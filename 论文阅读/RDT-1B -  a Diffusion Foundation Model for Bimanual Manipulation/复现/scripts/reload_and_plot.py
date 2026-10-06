import sys,json
from pathlib import Path
import torch,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8')
PAPER=Path(__file__).resolve().parents[2];OUT=PAPER/'复现'/'outputs'
sys.path.insert(0,str(PAPER/'代码'/'RoboticsDiffusionTransformer'))
from models.rdt_runner import RDTRunner
torch.set_num_threads(4)
r=json.loads((OUT/'结果.json').read_text(encoding='utf-8'))
runner=RDTRunner(**r['constructor'],dtype=torch.float32).cpu()
runner.load_state_dict(torch.load(OUT/'tiny_rdt_state.pt',map_location='cpu',weights_only=True),strict=True);runner.eval()
f=torch.load(OUT/'evaluation_fixture.pt',map_location='cpu',weights_only=True)
torch.manual_seed(f['sample_seed'])
with torch.no_grad():out=runner.predict_action(f['lang'],f['lang_mask'],f['img'],f['state'],f['mask'],f['freq'])
np.testing.assert_allclose(out.numpy(),f['expected_sample'].numpy(),atol=1e-7)
curve=json.loads((OUT/'训练曲线.json').read_text(encoding='utf-8'))
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
ax[0].semilogy([v['step'] for v in curve],[v['all_slots_training_mse'] for v in curve]);ax[0].set(title='Synthetic training: MSE over all 128 slots',xlabel='Optimizer update',ylabel='MSE (log scale)');ax[0].grid(alpha=.2)
for i,z in [(0,-.7),(-1,.7)]:
 ax[1].plot(range(64),out[i,:,0].numpy(),label=f'Generated: goal {z:+.1f}')
 ax[1].axhline(z,linestyle='--',alpha=.6,label=f'Target {z:+.1f}')
ax[1].set(title='Official five-step sampling: right joint 0',xlabel='Position within the 64-action block',ylabel='Synthetic joint value');ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
fig.savefig(OUT/'demo曲线.png',dpi=160);plt.close(fig)
(OUT/'权重重载校验.json').write_text(json.dumps({'strict_load':'passed','new_process_sampling_matches':True,'max_abs_error':float((out-f['expected_sample']).abs().max()),'real_robot_evaluation':False},indent=2),encoding='utf-8')
print('PASS: independent checkpoint reload and official sampling match; figure saved.')