"""Independent strict reload, source sampling diagnostics, and honest collapse plots."""
import json,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from genie_core_demo import *

def main():
    tok,lam,dyn=make_models()
    state=torch.load(O/'small_core_state.pt',weights_only=True)
    fixture=torch.load(O/'fixture.pt',weights_only=True)
    for name,m in [('tokenizer',tok),('LAM',lam),('dynamics',dyn)]:m.load_state_dict(state[name],strict=True)
    metrics,f=evaluate(tok,lam,dyn,fixture['video'],fixture['labels_eval_only'])
    errors={k:float((f[k].float()-fixture[k].float()).abs().max()) for k in f}
    assert max(errors.values())<1e-6,errors
    # Actual sampling hook: capture every input to MaskGIT, not just loop count.
    inputs=[]
    # Upstream calls maskgit.forward explicitly, bypassing Module-level hooks.
    # Its token_emb module is invoked normally and receives the actual token IDs.
    h=dyn.maskgit.token_emb.register_forward_pre_hook(lambda _,a: inputs.append(a[0].clone()))
    _,ids5=one_step(tok,lam,dyn,fixture['video'][:1,:,:1],fixture['lam_ids'][:1,:1],steps=5);h.remove()
    _,ids1=one_step(tok,lam,dyn,fixture['video'][:1,:,:1],fixture['lam_ids'][:1,:1],steps=1)
    assert len(inputs)==5
    same=all(torch.equal(inputs[0],x) for x in inputs)
    # Test tuple defaults separately; don't silently repair third-party source.
    try:
        dyn.sample(prime_token_ids=tok(fixture['video'][:1,:,:1],return_only_codebook_ids=True).flatten(1),
                   actions=action_codes(lam,fixture['lam_ids'][:1,:1]),num_tokens=16,patch_shape=(2,4,4))
        default_error=None
    except (TypeError,ValueError) as e:default_error=type(e).__name__+': '+str(e)
    lam.train();lam.zero_grad();loss=lam(fixture['video'][:4]);loss.backward()
    grads={n:float(p.grad.norm()) for n,p in lam.named_parameters() if p.grad is not None and n in ['to_action_emb.1.weight','vq._codebook.embed','to_pixels_first_frame.0.weight']}
    assert len(grads)==3 and all(torch.isfinite(torch.tensor(x)) and x>0 for x in grads.values())
    report={'strict_reload':True,'max_abs_errors':errors,'sampling_loop_calls':len(inputs),
            'all_refinement_network_inputs_equal':same,'sampling_input_shape':list(inputs[0].shape),
            'steps_1_vs_5_same_token_ids':torch.equal(ids1,ids5),'upstream_default_sampling_error':default_error,
            'LAM_nonzero_gradient_norms':grads,'full_paper_reproduction':False}
    second_state=torch.load(O/'vq001_state.pt',weights_only=True)
    second_fixture=torch.load(O/'vq001_fixture.pt',weights_only=True)
    for name,m in [('tokenizer',tok),('LAM',lam),('dynamics',dyn)]:m.load_state_dict(second_state[name],strict=True)
    lam.vq_loss_w=.01
    _,second_f=evaluate(tok,lam,dyn,second_fixture['video'],second_fixture['labels_eval_only'])
    second_errors={k:float((second_f[k].float()-second_fixture[k].float()).abs().max()) for k in second_f}
    assert max(second_errors.values())<1e-6,second_errors
    report['VQ001_reload_max_abs_errors']=second_errors
    (O/'权重重载校验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    curve=json.loads((O/'训练曲线.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(1,3,figsize=(12,3))
    for a,stage in zip(ax,['tokenizer','LAM','dynamics']):
        items=[x for x in curve if x['stage']==stage];a.plot([x['step'] for x in items],[x['loss'] for x in items],lw=.7)
        a.set_yscale('log');a.set_title(stage);a.set_xlabel('Update');a.set_ylabel('Loss')
    fig.tight_layout();fig.savefig(O/'训练曲线.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(8,3))
    ax[0].bar(range(8),metrics['LAM_code_histogram']);ax[0].set_xlabel('Latent code');ax[0].set_ylabel('Count / 256 transitions')
    ax[0].set_title('Original VQ weight .25')
    ab=json.loads((O/'VQ权重对照.json').read_text(encoding='utf-8'))
    ax[1].bar(range(8),ab['LAM_code_histogram']);ax[1].set_xlabel('Latent code');ax[1].set_title('VQ weight .01')
    fig.tight_layout();fig.savefig(O/'动作码使用率.png',dpi=150);plt.close(fig)
    fixture=second_fixture # Display improved run; original checkpoint/metrics retained.
    fig,axes=plt.subplots(4,5,figsize=(9,6.6))
    for i in range(5):
        for row,tensor in enumerate([fixture['video'][:1],fixture['tokenizer_reconstruction'][:1],
                                     fixture['lam_reconstruction'][:1],fixture['rollout'][:,:,:5]]):
            a=axes[row,i]
            if row==2 and i==0:a.text(.5,.5,'No prediction\nfor input frame',ha='center',va='center',fontsize=8)
            else:a.imshow(tensor[0,0,i-1 if row==2 else i].numpy(),cmap='gray',vmin=0,vmax=1)
            a.set_xticks([]);a.set_yticks([])
            if i==0:a.set_ylabel(['target','tokenizer','LAM teacher forcing','generated rollout'][row],fontsize=8)
            if row==0:a.set_title('t='+str(i))
    fig.tight_layout();fig.savefig(O/'重建与生成.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,9,figsize=(12,2))
    images=torch.cat([fixture['video'][:1,:,:1],fixture['all_user_codes']],0)
    for i,a in enumerate(axes):
        a.imshow(images[i,0,0].numpy(),cmap='gray',vmin=0,vmax=1);a.set_title('prompt' if i==0 else 'code '+str(i-1));a.axis('off')
    fig.tight_layout();fig.savefig(O/'用户动作码对照.png',dpi=150);plt.close(fig)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
