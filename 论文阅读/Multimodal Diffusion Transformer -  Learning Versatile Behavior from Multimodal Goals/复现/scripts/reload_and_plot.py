import json,sys
from pathlib import Path
import torch,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
sys.stdout.reconfigure(encoding='utf-8')
PAPER=Path(__file__).resolve().parents[2];OUT=PAPER/'复现'/'outputs'
sys.path.insert(0,str(PAPER/'代码'/'mdt_policy'))
from mdt.models.edm_diffusion.score_wrappers import GCDenoiser
from mdt.models.edm_diffusion.gc_sampling import sample_ddim
torch.set_num_threads(4)
r=json.loads((OUT/'结果.json').read_text(encoding='utf-8'))
model=GCDenoiser(OmegaConf.create(r['config']),sigma_data=r['sigma_data']).cpu()
model.load_state_dict(torch.load(OUT/'tiny_mdt_state.pt',map_location='cpu',weights_only=True),strict=True);model.eval()
f=torch.load(OUT/'evaluation_fixture.pt',map_location='cpu',weights_only=True)
sample=sample_ddim(model,f['state'],f['initial_noise'].clone(),f['goals'],f['sigmas'],disable=True)
np.testing.assert_allclose(sample.numpy(),f['expected_sample'].numpy(),atol=1e-7)
curve=json.loads((OUT/'训练曲线.json').read_text(encoding='utf-8'))
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for mod in ['vis','lang']:
 c=[v for v in curve if v['modality']==mod]
 ax[0].plot([v['step'] for v in c],[v['inner_target_loss'] for v in c],label=mod,alpha=.8)
ax[0].set(title='Synthetic training: internal target loss',xlabel='Optimizer updates',ylabel='Loss');ax[0].legend();ax[0].grid(alpha=.2)
for i,target in enumerate([-.7,.7]):
 ax[1].plot(range(10),sample[i,:,0].numpy(),marker='o',label=f'Generated (goal {target:+.1f})')
 ax[1].axhline(target,linestyle='--',alpha=.6,label=f'Target {target:+.1f}')
ax[1].set(title='Official 10-step DDIM: action dimension 0',xlabel='Position in action block',ylabel='Synthetic action value');ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
fig.savefig(OUT/'demo曲线.png',dpi=160);plt.close(fig)
(OUT/'权重重载校验.json').write_text(json.dumps({'strict_load':'passed','official_ddim_matches_saved_sample':True,'max_abs_error':float((sample-f['expected_sample']).abs().max()),'no_robot_evaluation':True},indent=2),encoding='utf-8')
print('PASS: independent process reload and official sampling match; figure saved.')