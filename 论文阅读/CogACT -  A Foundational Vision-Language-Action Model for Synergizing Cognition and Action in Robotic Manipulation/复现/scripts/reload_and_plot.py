import sys,json
from pathlib import Path
import torch,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8')
PAPER=Path(__file__).resolve().parents[2];REPO=PAPER/'代码'/'CogACT';OUT=PAPER/'复现'/'outputs'
sys.path.insert(0,str(REPO))
from action_model.action_model import ActionModel
torch.set_num_threads(4)
r=json.loads((OUT/'结果.json').read_text(encoding='utf-8'))
model=ActionModel(**r['config']).cpu()
model.load_state_dict(torch.load(OUT/'dit_s_state.pt',map_location='cpu',weights_only=True),strict=True);model.eval();ddim=model.create_ddim(10)
f=torch.load(OUT/'evaluation_fixture.pt',map_location='cpu',weights_only=True)
with torch.no_grad():out=ddim.ddim_sample_loop(model.net.forward,f['initial_noise'].shape,noise=f['initial_noise'].clone(),clip_denoised=False,model_kwargs={'z':f['cognition']},device=torch.device('cpu'),eta=0.,progress=False)
np.testing.assert_allclose(out.numpy(),f['expected_sample'].numpy(),atol=1e-6,rtol=1e-5)
curve=json.loads((OUT/'训练曲线.json').read_text(encoding='utf-8'))
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
ax[0].plot([v['step'] for v in curve],[v['epsilon_mse'] for v in curve],alpha=.75);ax[0].set(title='Official DiT-S: synthetic epsilon training',xlabel='Optimizer update',ylabel='Epsilon MSE');ax[0].grid(alpha=.2)
for i,z in [(0,-.7),(-1,.7)]:
 ax[1].plot(range(16),out[i,:,0].numpy(),marker='o',label=f'Generated (goal {z:+.1f})')
 ax[1].axhline(z,linestyle='--',alpha=.6,label=f'Target {z:+.1f}')
ax[1].set(title='Official 10-step DDIM: no output clipping',xlabel='Position within 16-action block',ylabel='Synthetic action dimension 0');ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
fig.savefig(OUT/'demo曲线.png',dpi=160);plt.close(fig)
(OUT/'权重重载校验.json').write_text(json.dumps({'strict_state_dict_load':'passed','independent_DDIM_sample':'passed','max_abs_error':float((out-f['expected_sample']).abs().max()),'atol':1e-6,'rtol':1e-5,'full_paper_reproduction':False},indent=2),encoding='utf-8')
print('PASS: independent process checkpoint reload and official DDIM sample agree; figure saved.')