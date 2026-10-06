"""Third-party GenieRedux small core training. No original Genie weights/code.
Action labels are held only in evaluation; optimization sees videos alone.
"""
import sys,json,time,hashlib,subprocess
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(4)
P=Path(__file__).resolve().parents[2];O=P/'复现'/'outputs';REPO=P/'代码'/'GenieRedux'
sys.path.insert(0,str(REPO))
from models import Tokenizer, LatentActionModel, MaskGIT, Dynamics
SEED=97
KW=dict(dim=32,image_size=8,patch_size=2,temporal_patch_size=1,num_blocks=1,heads=2,dim_head=16,channels=1,codebook_dim=16,vq_loss_w=.25)
DIRECTIONS=torch.tensor([[1,0],[-1,0],[0,1],[0,-1]])

def make_models():
    tok=Tokenizer(codebook_size=32,**KW)
    lam=LatentActionModel(codebook_size=8,**{**KW,'patch_size':4})
    maskgit=MaskGIT(dim=32,num_tokens=32,max_seq_len=256,image_size=8,patch_size=2,num_blocks=1,heads=2,dim_head=16,is_guided=False,use_token=False)
    dyn=Dynamics(maskgit=maskgit)
    return tok,lam,dyn

def render(positions):
    yy,xx=torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij')
    dx=(xx-positions[...,0,None,None]).abs();dx=torch.minimum(dx,8-dx)
    dy=(yy-positions[...,1,None,None]).abs();dy=torch.minimum(dy,8-dy)
    return (.05*((xx+yy)%2)+.75*torch.exp(-(dx**2+dy**2)/(2*.8**2))).unsqueeze(1)

def dataset(n,seed):
    g=torch.Generator().manual_seed(seed)
    positions=torch.randint(8,(n,2),generator=g)
    labels=torch.randint(4,(n,4),generator=g)
    path=[positions]
    for h in range(4):path.append((path[-1]+DIRECTIONS[labels[:,h]])%8)
    return render(torch.stack(path,1)),labels

def action_codes(lam,ids):
    # LAM's spatial 2x2 token grid differs from dynamics' 4x4 grid.
    # Retrieve projected actual VQ codes then singleton-broadcast over spatial axis.
    return lam.get_codes_from_indices(ids).unsqueeze(2).unsqueeze(2).detach()

@torch.no_grad()
def one_step(tok,lam,dyn,frame,ids,steps=5):
    prime=tok(frame,return_only_codebook_ids=True).flatten(1)
    # Explicit scalar args avoid upstream tuple defaults; actual source sampling used.
    nxt=dyn.sample(prime_token_ids=prime,actions=action_codes(lam,ids),num_tokens=16,
                   patch_shape=(2,4,4),inference_steps=steps,mask_schedule='cosine',sample_temperature=1.,tokenizer=tok)
    joined=torch.cat([prime,nxt],1).view(len(frame),2,4,4)
    return tok.decode_from_codebook_indices(joined)[:,:,1:],nxt

@torch.no_grad()
def evaluate(tok,lam,dyn,videos,labels):
    tok.eval();lam.eval();dyn.eval()
    reconstruction=tok(videos,return_recons_only=True)
    lam_rec=lam(videos,return_recons_only=True)
    ids=lam(videos,return_only_codebook_ids=True)
    hist=torch.bincount(ids.flatten(),minlength=8)
    confusion=torch.zeros(8,4,dtype=torch.long)
    for k in range(8):
        for d in range(4):confusion[k,d]=((ids==k)&(labels==d)).sum()
    purity=confusion.max(1).values.sum().item()/ids.numel()
    visualids=tok(videos,return_only_codebook_ids=True)
    torch.manual_seed(391)
    ce,logits=dyn(video_codebook_ids=visualids,actions=action_codes(lam,ids),return_logits=True)
    pred,generatedids=one_step(tok,lam,dyn,videos[:,:,:1],ids[:,:1])
    different=(ids[:,:1]+1)%8
    random_pred,_=one_step(tok,lam,dyn,videos[:,:,:1],different)
    mse=(pred-videos[:,:,1:2]).square().mean((1,2,3,4))
    wrong=(random_pred-videos[:,:,1:2]).square().mean((1,2,3,4))
    dpsnr=(10*torch.log10(wrong.clamp(min=1e-12)/mse.clamp(min=1e-12))).mean()
    all_frames=[]
    for k in range(8):
        p,_=one_step(tok,lam,dyn,videos[:1,:,:1],torch.tensor([[k]]))
        all_frames.append(p)
    # World rollout receives only first image and user-selected IDs, no future pixels.
    frames=videos[:1,:,:1];rollout=[frames]
    for h in range(6):
        frames,_=one_step(tok,lam,dyn,frames,torch.tensor([[h%8]]));rollout.append(frames)
    metrics={'tokenizer_pixel_mse':F.mse_loss(reconstruction,videos).item(),
             'LAM_pixel_mse':F.mse_loss(lam_rec,videos[:,:,1:]).item(),
             'LAM_code_histogram':hist.tolist(),'LAM_code_direction_counts':confusion.tolist(),
             'LAM_direction_purity_posthoc':purity,'heldout_dynamics_masked_CE':ce.item(),
             'one_step_generated_pixel_mse':mse.mean().item(),'different_code_pixel_mse':wrong.mean().item(),
             'toy_delta_PSNR_t1_dB':dpsnr.item(),'generated_visual_codes_used':torch.unique(generatedids).tolist(),
             'visual_codes_used':torch.unique(visualids).tolist(),
             'eight_user_codes_output_pair_max_difference':float((torch.cat(all_frames,0)-all_frames[0]).abs().max())}
    fixture={'video':videos,'labels_eval_only':labels,'lam_ids':ids,'tokenizer_reconstruction':reconstruction,
             'lam_reconstruction':lam_rec,'one_step':pred,'different_code_step':random_pred,
             'all_user_codes':torch.cat(all_frames,0),'rollout':torch.cat(rollout,2)}
    return metrics,fixture

def train():
    O.mkdir(parents=True,exist_ok=True);torch.manual_seed(SEED)
    videos,_=dataset(512,SEED) # Discard generating actions before training.
    tests,labels=dataset(64,SEED+1)
    tok,lam,dyn=make_models();history=[];start=time.time()
    for name,model,updates,lr in [('tokenizer',tok,1201,.001),('LAM',lam,1801,.001)]:
        model.train();opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0)
        for step in range(updates):
            ix=torch.randint(len(videos),(16,));loss=model(videos[ix])
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            history.append({'stage':name,'step':step,'loss':loss.item()})
            if step%400==0:print(name,step,loss.item(),'seconds',round(time.time()-start,1),flush=True)
        model.eval()
    # Freeze both label-producing modules. No gradients or true action IDs in dynamics.
    for m in [tok,lam]:
        for p in m.parameters():p.requires_grad_(False)
    with torch.no_grad():
        visual=tok(videos,return_only_codebook_ids=True)
        ids=lam(videos,return_only_codebook_ids=True)
        codes=action_codes(lam,ids)
    dyn.train();opt=torch.optim.AdamW(dyn.parameters(),lr=.001,weight_decay=0)
    for step in range(801):
        ix=torch.randint(len(videos),(16,));loss=dyn(video_codebook_ids=visual[ix],actions=codes[ix])
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(dyn.parameters(),1);opt.step()
        history.append({'stage':'dynamics','step':step,'loss':loss.item()})
        if step%400==0:print('dynamics',step,loss.item(),flush=True)
    metrics,fixture=evaluate(tok,lam,dyn,tests,labels)
    # Test temporal causality by corrupting only final future frame.
    modified=tests.clone();modified[:,:,-1]=torch.rand_like(modified[:,:,-1])
    with torch.no_grad():
        old=tok(tests,return_only_codebook_ids=True);new=tok(modified,return_only_codebook_ids=True)
        old_a=lam(tests,return_only_codebook_ids=True);new_a=lam(modified,return_only_codebook_ids=True)
    assert torch.equal(old[:,:-1],new[:,:-1]) and torch.equal(old_a[:,:-1],new_a[:,:-1])
    metrics.update(label='第三方GenieRedux小配置机制验证，非原Genie官方复现',seed=SEED,
        torch=torch.__version__,train_videos=512,test_videos=64,frames=5,
        updates={'tokenizer':1201,'LAM':1801,'dynamics':801},config=KW,
        parameters={n:sum(p.numel() for p in m.parameters()) for n,m in [('tokenizer',tok),('LAM',lam),('dynamics',dyn)]},
        future_frame_causality_test='passed (video prior codes and LAM prior actions unchanged)',
        no_true_actions_in_training=True,wall_seconds=time.time()-start,
        repo_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (REPO/'models').rglob('*.py')})
    torch.save({n:m.state_dict() for n,m in [('tokenizer',tok),('LAM',lam),('dynamics',dyn)]},O/'small_core_state.pt')
    torch.save(fixture,O/'fixture.pt')
    (O/'结果.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    (O/'训练曲线.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':train()
