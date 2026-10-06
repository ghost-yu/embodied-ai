import sys,json
from pathlib import Path
import numpy as np,jax,jax.numpy as jnp,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flax import nnx,serialization
from pi0_demo_support import build_model
from openpi.models import model as model_api
sys.stdout.reconfigure(encoding='utf-8')
PAPER=Path(__file__).resolve().parents[2];OUT=PAPER/'复现'/'outputs'
model=build_model();params=nnx.state(model,nnx.Param)
loaded=serialization.from_bytes(params.to_pure_dict(),(OUT/'tiny_pi0_params.msgpack').read_bytes())
params.replace_by_pure_dict(loaded);nnx.update(model,params)
f=np.load(OUT/'evaluation_fixture.npz')
obs=model_api.Observation(images={name:jnp.array(f['image']) for name in model_api.IMAGE_KEYS},image_masks={name:jnp.ones(4,dtype=bool) for name in model_api.IMAGE_KEYS},
 state=jnp.array(f['state']),tokenized_prompt=jnp.array(f['tokens']),tokenized_prompt_mask=jnp.array(f['token_mask']))
@nnx.jit
def sample(m,observation,noise):return m.sample_actions(jax.random.key(0),observation,num_steps=10,noise=noise)
result=np.asarray(sample(model,obs,jnp.array(f['noise'])))
np.testing.assert_allclose(result,f['expected'],atol=2e-5,rtol=1e-5)
curve=json.loads((OUT/'训练曲线.json').read_text(encoding='utf-8'))
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
ax[0].plot([v['step'] for v in curve],[v['mean_flow_mse'] for v in curve],alpha=.7);ax[0].set(title='Official Pi0: synthetic flow training',xlabel='Optimizer updates',ylabel='Mean flow MSE');ax[0].grid(alpha=.2)
for i,z in [(0,-.6),(3,.6)]:
 ax[1].plot(range(50),result[i,:,0],label=f'Generated: goal {z:+.1f}')
 ax[1].axhline(z,linestyle='--',alpha=.6,label=f'Target {z:+.1f}')
ax[1].set(title='Official 10-step cached Euler sampling',xlabel='Position within 50-action block',ylabel='Synthetic action dimension 0');ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
fig.savefig(OUT/'demo曲线.png',dpi=160);plt.close(fig)
(OUT/'参数重载校验.json').write_text(json.dumps({'parameter_schema_load':'passed','new_process_sampling_matches':True,'max_abs_error':float(np.max(np.abs(result-f['expected']))),'comparison_atol':2e-5,'comparison_rtol':1e-5,'paper_reproduction':False},indent=2),encoding='utf-8')
print('PASS: new process parameter reload, cached Euler sample agreement, and figure saved.')