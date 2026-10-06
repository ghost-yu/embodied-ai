import json,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ijepa_demo import *

def main():
    state=torch.load(O/'small_core_state.pt',weights_only=True);f=torch.load(O/'fixture.pt',weights_only=True)
    enc,pred,target=make_models()
    for name,m in [('encoder',enc),('predictor',pred),('target',target)]:m.load_state_dict(state[name],strict=True);m.eval()
    metrics,z,h=evaluate(enc,pred,target,f['images'],f['masks_enc'],f['masks_pred'])
    errors={'prediction':float((z-f['prediction']).abs().max()),'target':float((h-f['target']).abs().max())}
    assert max(errors.values())<1e-6
    torch.manual_seed(SEED);_,_,anchor=make_models();anchor.eval()
    old_h=target_features(anchor,f['images'],f['masks_enc'],f['masks_pred'])
    comparison={'prediction_vs_initial_frozen_target_smooth_l1':F.smooth_l1_loss(z,old_h).item(),
                'final_vs_initial_target_mse':F.mse_loss(h,old_h).item(),
                'oracle_global_mean_final_target_baseline_smooth_l1':F.smooth_l1_loss(h.mean((0,1),keepdim=True).expand_as(h),h).item()}
    report={'strict_reload':True,'max_abs_error':errors,'reloaded_metrics':metrics,
            'target_motion_diagnostics':comparison,'full_paper_reproduction':False,'semantic_representation_quality_proven':False}
    (O/'权重重载校验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    history=json.loads((O/'训练曲线.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,3,figsize=(12,3.4))
    axes[0].plot([x['step'] for x in history],[x['loss'] for x in history]);axes[0].set_yscale('log');axes[0].set_title('Training smooth-L1')
    axes[1].plot([x['step'] for x in history],[x['lr'] for x in history]);axes[1].set_title('Official LR schedule')
    axes[2].plot([x['step'] for x in history],[x['momentum'] for x in history]);axes[2].set_title('Target EMA momentum')
    for a in axes:a.set_xlabel('Update')
    fig.tight_layout();fig.savefig(O/'训练曲线.png',dpi=140);plt.close(fig)
    image=f['images'][0].permute(1,2,0).numpy();visible=torch.zeros(8,8);visible.flatten()[f['masks_enc'][0][0]]=1
    view=image*visible.repeat_interleave(8,0).repeat_interleave(8,1).numpy()[...,None]
    fig,axes=plt.subplots(1,6,figsize=(13,2.5));axes[0].imshow(image.clip(0,1));axes[0].set_title('Full image / target')
    axes[1].imshow(view.clip(0,1));axes[1].set_title('Visible context')
    for i in range(4):
        mask=torch.zeros(8,8);mask.flatten()[f['masks_pred'][i][0]]=1
        axes[i+2].imshow(mask,vmin=0,vmax=1,cmap='magma');axes[i+2].set_title('Target block '+str(i))
    for a in axes:a.set_xticks([]);a.set_yticks([])
    fig.tight_layout();fig.savefig(O/'多块mask.png',dpi=140);plt.close(fig)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
