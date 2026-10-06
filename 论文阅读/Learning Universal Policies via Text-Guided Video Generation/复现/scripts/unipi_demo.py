"""UniPi mechanism teaching demo; no official model/code/weights.

Pixel-space clean-video diffusion -> learned midpoint -> learned inverse -> toy env.
"""
import sys, json, time, hashlib
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
SIZE, FRAMES, LEVELS, SEED = 16, 4, 64, 83
DIRECTIONS = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
COMMANDS = ['move right', 'move left', 'move down', 'move up']

def render(pos):
    """Positions [...,2] in pixels -> Gaussian grayscale [...,1,16,16], [0,1]."""
    yy, xx = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing='ij')
    return torch.exp(-((xx-pos[..., 0, None, None])**2 +
                       (yy-pos[..., 1, None, None])**2) / (2*.85**2)).unsqueeze(-3)

def dataset(n, generator):
    start = 6 + torch.rand(n, 2, generator=generator)*3
    ids = torch.randint(4, (n,), generator=generator)
    # Eight 0.4-pixel controls; coarse planner sees future h=2,4,6,8.
    positions = start[:, None] + torch.arange(9)[None, :, None]*.4*DIRECTIONS[ids, None]
    video = render(positions)
    return video[:, 0], ids, video[:, 2::2], video, positions

class VideoDenoiser(nn.Module):
    """Small Conv3D x_clean predictor; concatenated first frame at every future h."""
    def __init__(self):
        super().__init__()
        self.text = nn.Embedding(5, 32)  # 4 commands + null instruction
        self.time = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.input = nn.Conv3d(3, 32, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv3d(32, 32, 3, padding=1) for _ in range(4)])
        self.output = nn.Conv3d(32, 1, 3, padding=1)
    def forward(self, noisy, k, first, ids):
        # Layout [B,C,F,Y,X], k [B], ids [B]; first [B,1,Y,X].
        tiled = first.unsqueeze(2).expand(-1, -1, FRAMES, -1, -1)
        frame_index = torch.linspace(0, 1, FRAMES).view(1,1,FRAMES,1,1).expand_as(noisy)
        z = F.silu(self.input(torch.cat([noisy, tiled, frame_index], dim=1)))
        cond = (self.text(ids) + self.time(k[:,None].float()/(LEVELS-1)))[:,:,None,None,None]
        for layer in self.blocks:
            z = z + F.silu(layer(z + cond))*.5
        return self.output(z)

class Midpoint(nn.Module):
    """Deterministic supervised teaching interpolator, NOT UniPi diffusion TSR."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2,32,3,padding=1), nn.SiLU(),
                                 nn.Conv2d(32,32,3,padding=1), nn.SiLU(),
                                 nn.Conv2d(32,1,3,padding=1))
    def forward(self, a, b):
        return self.net(torch.cat([a,b],1))

def moments(image):
    """Known toy camera feature extractor; threshold negative predicted intensity only.
    No coordinate, target direction, or true action is passed to inverse model.
    """
    weight = image.clamp(min=0)
    total = weight.sum((-2,-1)).clamp(min=1e-6)
    axis = torch.arange(SIZE, dtype=image.dtype)
    x = (weight*axis[None,None,None,:]).sum((-2,-1))/total
    y = (weight*axis[None,None,:,None]).sum((-2,-1))/total
    return torch.cat([x,y],1)

class InverseDynamics(nn.Module):
    """Learned 2-D control regressor after image moments; non-paper architecture."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,32),nn.SiLU(),nn.Linear(32,2))
    def forward(self, a, b):
        return self.net(moments(b)-moments(a))

class DotEnv:
    """Actual deterministic toy step, with bounded actions and workspace."""
    def reset(self, start, goal):
        self.position = start.clone()
        self.goal = goal.clone()
        self.steps = 0
        return render(self.position)
    def step(self, action):
        applied = action.clamp(-.8,.8)
        self.position = (self.position + applied).clamp(0,SIZE-1)
        self.steps += 1
        distance = torch.linalg.vector_norm(self.position-self.goal,dim=-1)
        return render(self.position), -distance, self.steps==8, {'distance':distance, 'applied':applied}

def schedule():
    beta = torch.linspace(.001,.2,LEVELS)
    return (1-beta).cumprod(0)

@torch.no_grad()
def sample(model, first, ids, noise, omega=0.):
    """All 64 deterministic DDIM steps, x0 parameterization, no output clipping.
    Null branch removes language only and RETAINS observed first frame.
    """
    ab = schedule()
    z = noise.clone()
    for k in reversed(range(LEVELS)):
        ts = torch.full((len(ids),),k)
        clean = model(z,ts,first,ids)
        if omega:
            null = model(z,ts,first,torch.full_like(ids,4))
            clean = (1+omega)*clean-omega*null
        eps = (z-ab[k].sqrt()*clean)/(1-ab[k]).sqrt()
        prev = ab[k-1] if k else torch.tensor(1.)
        z = prev.sqrt()*clean+(1-prev).sqrt()*eps
    return z.permute(0,2,1,3,4)

def densify(midpoint, first, coarse):
    prev = torch.cat([first[:,None], coarse[:,:-1]],1)
    b = len(first)
    mids = midpoint(prev.flatten(0,1),coarse.flatten(0,1)).view(b,FRAMES,1,SIZE,SIZE)
    return torch.stack([mids,coarse],2).flatten(1,2)

@torch.no_grad()
def execute(inverse, first, dense, starts, goals):
    # All actions are inferred BEFORE first step: paper-style open loop.
    whole = torch.cat([first[:,None],dense],1)
    actions = inverse(whole[:,:-1].flatten(0,1),whole[:,1:].flatten(0,1)).view(-1,8,2)
    env = DotEnv()
    observations = [env.reset(starts,goals)]
    applied=[]
    for h in range(8):
        observation,reward,done,info=env.step(actions[:,h])
        observations.append(observation);applied.append(info['applied'])
    assert done and env.steps==8
    return actions, torch.stack(observations,1), info['distance'], torch.stack(applied,1)

def weighted_pixel_loss(pred,target):
    # Toy foreground emphasis so empty background doesn't dominate sparse dot video.
    return ((pred-target)**2*(1+5*target)).mean()

def train():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    g=torch.Generator().manual_seed(SEED)
    first,ids,coarse,video,positions=dataset(512,g)
    eval_first,eval_ids,eval_coarse,eval_video,eval_positions=dataset(64,torch.Generator().manual_seed(198))
    model,mid,inverse=VideoDenoiser(),Midpoint(),InverseDynamics()
    ab=schedule();opt=torch.optim.Adam(model.parameters(),lr=.001)
    history=[];started=time.time()
    for step in range(1801):
        ix=torch.randint(512,(32,));k=torch.randint(LEVELS,(32,));noise=torch.randn(32,1,4,16,16)
        clean=coarse[ix].permute(0,2,1,3,4)
        a=ab[k][:,None,None,None,None]
        noisy=a.sqrt()*clean+(1-a).sqrt()*noise
        labels=ids[ix].clone();labels[torch.rand(32)<.15]=4
        loss=weighted_pixel_loss(model(noisy,k,first[ix],labels),clean)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        history.append({'stage':'video','step':step,'loss':loss.item()})
        if step%300==0:print('video',step,loss.item(),'seconds',round(time.time()-started,1),flush=True)
    opt=torch.optim.Adam(mid.parameters(),lr=.001)
    for step in range(801):
        ix=torch.randint(512,(32,));frames=video[ix]
        loss=weighted_pixel_loss(mid(frames[:,0:-1:2].flatten(0,1),frames[:,2::2].flatten(0,1)),frames[:,1::2].flatten(0,1))
        opt.zero_grad();loss.backward();opt.step()
        history.append({'stage':'midpoint','step':step,'loss':loss.item()})
    # Inverse is trained independently on random two-frame controls, including negative directions.
    starts=2+torch.rand(4096,2)*11;controls=(torch.rand(4096,2)-.5)*1.6
    a,b=render(starts),render(starts+controls)
    opt=torch.optim.Adam(inverse.parameters(),lr=.003)
    for step in range(801):
        ix=torch.randint(4096,(128,));loss=F.mse_loss(inverse(a[ix],b[ix]),controls[ix])
        opt.zero_grad();loss.backward();opt.step()
        history.append({'stage':'inverse','step':step,'loss':loss.item()})
    model.eval();mid.eval();inverse.eval()
    noise=torch.randn(64,1,4,16,16,generator=torch.Generator().manual_seed(977))
    generated=sample(model,eval_first,eval_ids,noise)
    dense=densify(mid,eval_first,generated)
    actions,observations,dist,applied=execute(inverse,eval_first,dense,eval_positions[:,0],eval_positions[:,-1])
    oracle_actions,oracle_obs,oracle_dist,_=execute(inverse,eval_first,eval_video[:,1:],eval_positions[:,0],eval_positions[:,-1])
    with torch.no_grad():
        teststart=3+torch.rand(256,2)*9;testcontrol=(torch.rand(256,2)-.5)*1.2
        inv_mse=F.mse_loss(inverse(render(teststart),render(teststart+testcontrol)),testcontrol).item()
    torch.save({'video':model.state_dict(),'midpoint':mid.state_dict(),'inverse':inverse.state_dict()},OUT/'teaching_state.pt')
    torch.save({'first':eval_first,'ids':eval_ids,'noise':noise,'coarse':generated,'dense':dense,'actions':actions,'observations':observations,'positions':eval_positions,'target_video':eval_video},OUT/'fixture.pt')
    report={'label':'教学机制验证，非官方UniPi复现','seed':SEED,'torch':torch.__version__,
            'video_updates':1801,'midpoint_updates':801,'inverse_updates':801,'dataset_train':512,'dataset_test':64,
            'video_parameters':sum(p.numel() for p in model.parameters()),'midpoint_parameters':sum(p.numel() for p in mid.parameters()),'inverse_parameters':sum(p.numel() for p in inverse.parameters()),
            'schedule':{'levels':LEVELS,'beta_start':.001,'beta_end':.2,'alpha_bar_last':ab[-1].item(),'DDIM_steps':64,'parameterization':'clean pixels','omega':0},
            'coarse_pixel_mse':F.mse_loss(generated,eval_coarse).item(),'dense_pixel_mse':F.mse_loss(dense,eval_video[:,1:]).item(),
            'inverse_heldout_mse':inv_mse,'mean_final_pixel_distance':dist.mean().item(),'max_final_pixel_distance':dist.max().item(),
            'toy_success_threshold_pixels':.5,'toy_success_count':int((dist<.5).sum()),'toy_trials':64,'oracle_video_final_distance':oracle_dist.mean().item(),
            'actions_clipped_components':int((actions!=applied).sum()),'generated_pixel_min':generated.min().item(),'generated_pixel_max':generated.max().item(),
            'wall_seconds':time.time()-started,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'训练曲线.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':train()
