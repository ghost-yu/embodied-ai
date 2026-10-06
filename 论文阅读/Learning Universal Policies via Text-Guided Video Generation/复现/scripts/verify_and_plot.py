"""Independent process checkpoint reload and open-loop execution audit."""
import json, hashlib
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unipi_demo import (OUT, VideoDenoiser, Midpoint, InverseDynamics, sample,
                        densify, execute, moments, DIRECTIONS, COMMANDS, schedule)

def main():
    state=torch.load(OUT/'teaching_state.pt',map_location='cpu',weights_only=True)
    f=torch.load(OUT/'fixture.pt',map_location='cpu',weights_only=True)
    model,mid,inverse=VideoDenoiser().eval(),Midpoint().eval(),InverseDynamics().eval()
    for name,obj in [('video',model),('midpoint',mid),('inverse',inverse)]:obj.load_state_dict(state[name],strict=True)
    times=[]
    handle=model.register_forward_pre_hook(lambda _,inputs: times.append(int(inputs[1][0])))
    coarse=sample(model,f['first'],f['ids'],f['noise']);handle.remove()
    assert times==list(reversed(range(64)))
    dense=densify(mid,f['first'],coarse)
    actions,obs,distance,applied=execute(inverse,f['first'],dense,f['positions'][:,0],f['positions'][:,-1])
    errors={key:float((value-f[key]).abs().max()) for key,value in [('coarse',coarse),('dense',dense),('actions',actions),('observations',obs)]}
    assert max(errors.values())<1e-6,errors
    # Task condition counterfactual: same first frame/noise, opposite command.
    opposite=torch.tensor([1,0,3,2])[f['ids']]
    reversed_video=sample(model,f['first'],opposite,f['noise'])
    pos=f['positions'][:,0,None]+torch.arange(1,9)[None,:,None]*.4*DIRECTIONS[opposite,None]
    from unipi_demo import render
    target_opposite=render(pos[:,1::2])
    correct_mse=float(torch.nn.functional.mse_loss(reversed_video,target_opposite))
    wrong_mse=float(torch.nn.functional.mse_loss(coarse,target_opposite))
    guided=sample(model,f['first'],f['ids'],f['noise'],omega=.5)
    guided_dense=densify(mid,f['first'],guided)
    _,_,guided_dist,_=execute(inverse,f['first'],guided_dense,f['positions'][:,0],f['positions'][:,-1])
    repeated=sample(model,f['first'],f['ids'],torch.randn(f['noise'].shape,generator=torch.Generator().manual_seed(978)))
    _,_,new_dist,_=execute(inverse,f['first'],densify(mid,f['first'],repeated),f['positions'][:,0],f['positions'][:,-1])
    report={'strict_reload':True,'reload_max_abs_errors':errors,'DDIM_model_times':times,
            'opposite_command_correct_target_mse':correct_mse,'original_command_wrong_target_mse':wrong_mse,
            'CFG_omega_0_5_mean_final_distance':guided_dist.mean().item(),
            'CFG_omega_0_5_toy_success_count':int((guided_dist<.5).sum()),
            'new_noise_mean_final_distance':new_dist.mean().item(),'new_noise_toy_success_count':int((new_dist<.5).sum()),
            'full_paper_reproduction':False}
    (OUT/'权重重载校验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    history=json.loads((OUT/'训练曲线.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(1,3,figsize=(13,3.4))
    for a,stage in zip(ax,['video','midpoint','inverse']):
        items=[r for r in history if r['stage']==stage]
        a.plot([r['step'] for r in items],[r['loss'] for r in items],linewidth=.6)
        a.set_yscale('log');a.set_xlabel('Update');a.set_ylabel('Training loss');a.set_title(stage)
    fig.tight_layout();fig.savefig(OUT/'训练曲线.png',dpi=140);plt.close(fig)
    # Fixed first test case per instruction; no cherry picking best success.
    indices=[int(torch.where(f['ids']==d)[0][0]) for d in range(4)]
    fig,axes=plt.subplots(12,9,figsize=(12,13))
    for row,i in enumerate(indices):
        plan=torch.cat([f['first'][i,None],dense[i]],0)
        for kind,data in enumerate([f['target_video'][i],plan,obs[i]]):
            for h in range(9):
                a=axes[row*3+kind,h];a.imshow(data[h,0].detach().numpy(),vmin=0,vmax=1,cmap='gray')
                a.set_xticks([]);a.set_yticks([])
                if h==0:a.set_ylabel(COMMANDS[row]+'\n'+['target','plan','executed'][kind],fontsize=7)
                if row==0 and kind==0:a.set_title('h='+str(h),fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'视频计划与执行.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4))
    for color,i in zip(['tab:blue','tab:orange','tab:green','tab:red'],indices):
        target=f['positions'][i].numpy()
        p=moments(torch.cat([f['first'][i,None],dense[i]],0)).detach().numpy()
        e=moments(obs[i]).numpy()
        ax.plot(target[:,0],target[:,1],':',color=color,label=COMMANDS[int(f['ids'][i])])
        ax.plot(p[:,0],p[:,1],'--',color=color);ax.plot(e[:,0],e[:,1],'-',color=color)
    ax.invert_yaxis();ax.set_aspect('equal');ax.set_xlabel('Pixel x');ax.set_ylabel('Pixel y')
    ax.set_title('Target dotted / plan dashed / execution solid');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(OUT/'轨迹.png',dpi=150);plt.close(fig)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
