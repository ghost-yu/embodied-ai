"""Official Pi0 JAX loss/cache/Euler with dummy Gemma and configured official mu/32 SigLIP."""
import hashlib,inspect,json,subprocess,sys,time
from pathlib import Path
import numpy as np,jax,jax.numpy as jnp,flax,optax
from flax import nnx,serialization
from pi0_demo_support import build_model,CONFIG
from openpi.models import pi0,model as model_api
sys.stdout.reconfigure(encoding='utf-8')
PAPER=Path(__file__).resolve().parents[2];REPO=PAPER/'代码'/'openpi';OUT=PAPER/'复现'/'outputs'
model=build_model();print('Constructed official Pi0: small config.',flush=True)
B,H,D=4,50,32
z=jnp.array([-.6,-.2,.2,.6])
images={name:jnp.broadcast_to(z[:,None,None,None],(B,224,224,3)) for name in model_api.IMAGE_KEYS}
state=jnp.zeros((B,D)).at[:,0].set(z)
obs=model_api.Observation(images=images,image_masks={name:jnp.ones(B,dtype=bool) for name in images},state=state,
 tokenized_prompt=jnp.array([[2,3,0,0]]*B,dtype=jnp.int32),tokenized_prompt_mask=jnp.array([[True,True,False,False]]*B))
actions=jnp.zeros((B,H,D))
for i,coefficient in enumerate([1.,.5,-.3,.2,.1,0.,.2]):actions=actions.at[:,:,i].set(z[:,None]*coefficient)
prefix,pm,pa=model.embed_prefix(obs)
noise=jax.random.normal(jax.random.key(641),(B,H,D));s=jnp.ones(B)*.6
suffix,sm,sa,ad=model.embed_suffix(obs,s[:,None,None]*noise+(1-s[:,None,None])*actions,s)
fullmask=pi0.make_attn_mask(jnp.concatenate([pm,sm],axis=1),jnp.concatenate([pa,sa]))
P=prefix.shape[1];S=suffix.shape[1]
assert suffix.shape==(B,51,64) and prefix.shape==(B,151,64) and ad is None
assert not np.asarray(fullmask[:,:P,P:]).any()
assert not np.asarray(fullmask[:,P:P+1,P+1:]).any()
assert np.asarray(fullmask[:,P+1:,P+1:]).all()
# Compare combined forward with official prefix-cache + suffix path.
pos=jnp.cumsum(jnp.concatenate([pm,sm],axis=1),axis=1)-1
(_,full_suffix),_=model.PaliGemma.llm([prefix,suffix],mask=fullmask,positions=pos)
_,cache=model.PaliGemma.llm([prefix,None],mask=pi0.make_attn_mask(pm,pa),positions=jnp.cumsum(pm,axis=1)-1)
cm=jnp.concatenate([jnp.broadcast_to(pm[:,None,:],(B,S,P)),pi0.make_attn_mask(sm,sa)],axis=-1)
cp=jnp.sum(pm,axis=-1)[:,None]+jnp.cumsum(sm,axis=-1)-1
(_,cached_suffix),_=model.PaliGemma.llm([None,suffix],mask=cm,positions=cp,kv_cache=cache)
cache_max=float(jnp.max(jnp.abs(full_suffix-cached_suffix)))
np.testing.assert_allclose(np.asarray(full_suffix),np.asarray(cached_suffix),atol=2e-5,rtol=1e-5)
# Exact stochastic loss reconstruction from official random-key split.
key=jax.random.key(643);_,nk,tk=jax.random.split(key,3)
n=jax.random.normal(nk,actions.shape);tm=jax.random.beta(tk,1.5,1,(B,))*.999+.001
xt=tm[:,None,None]*n+(1-tm[:,None,None])*actions
st,sma,sar,_=model.embed_suffix(obs,xt,tm)
mask=pi0.make_attn_mask(jnp.concatenate([pm,sma],1),jnp.concatenate([pa,sar]))
(_,so),_=model.PaliGemma.llm([prefix,st],mask=mask,positions=pos)
v=model.action_out_proj(so[:,-H:])
manual=jnp.mean((v-(n-actions))**2,axis=-1)
official=model.compute_loss(key,obs,actions,train=False)
np.testing.assert_allclose(np.asarray(official),np.asarray(manual),atol=2e-5,rtol=1e-5)
assert official.shape==(B,H)
print('PASS: masks, cache equivalence, exact flow loss.',flush=True)
@nnx.jit
def loss_eval(m,key,observation,target):return jnp.mean(m.compute_loss(key,observation,target,train=False))
@nnx.jit
def sample(m,observation,initial):return m.sample_actions(jax.random.key(0),observation,num_steps=10,noise=initial)
init_sample=sample(model,obs,noise);init_sample.block_until_ready()
initial_sample_mse=float(jnp.mean((init_sample-actions)**2))
initial_loss=float(loss_eval(model,jax.random.key(645),obs,actions))
optimizer=nnx.Optimizer(model,optax.adam(.002))
@nnx.jit
def train_step(m,opt,key,observation,target):
 def loss_fn(mm):return jnp.mean(mm.compute_loss(key,observation,target,train=False))
 loss,grads=nnx.value_and_grad(loss_fn)(m)
 # Aggregate gradients by major component; no replacement of forward/loss.
 totals={}
 for path,g in grads.flat_state().items():
  group='/'.join(str(x) for x in path[:2])
  totals[group]=totals.get(group,jnp.array(0.))+jnp.sum(jnp.square(g.value))
 norms={k:jnp.sqrt(v) for k,v in totals.items()}
 opt.update(grads)
 return loss,norms
curve=[];started=time.time();first_grad=None
for step in range(501):
 value,norms=train_step(model,optimizer,jax.random.fold_in(jax.random.key(646),step),obs,actions)
 val=float(value);assert np.isfinite(val)
 if first_grad is None:first_grad={k:float(v) for k,v in norms.items()}
 curve.append({'step':step,'mean_flow_mse':val})
 if step%20==0:print(json.dumps({'step':step,'loss':val,'elapsed':round(time.time()-started,1)}),flush=True)
final_loss=float(loss_eval(model,jax.random.key(645),obs,actions))
assert final_loss<initial_loss,(initial_loss,final_loss)
final_sample=sample(model,obs,noise);final_sample.block_until_ready()
new_noise=jax.random.normal(jax.random.key(647),noise.shape);new_sample=sample(model,obs,new_noise);new_sample.block_until_ready()
assert np.isfinite(np.asarray(final_sample)).all()
source_files=['src/openpi/models/pi0.py','src/openpi/models/pi0_config.py','src/openpi/models/gemma.py','src/openpi/models/siglip.py','src/openpi/models/model.py']
assert Path(inspect.getfile(pi0.Pi0)).resolve()==(REPO/source_files[0]).resolve()
report={'reference_date':'2026-10-03','repo_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
 'environment':{'python':sys.version,'jax':jax.__version__,'flax':flax.__version__,'numpy':np.__version__,'optax':optax.__version__,'device':str(jax.devices())},
 'config':CONFIG,'vision_override':'official SigLIP Module: So400m/14 -> mu/32 only at construction','parameter_count':sum(x.size for x in jax.tree.leaves(nnx.state(model,nnx.Param))),
 'source_sha256':{f:hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in source_files},
 'shapes':{'images':[B,224,224,3],'state':[B,32],'actions':[B,50,32],'prefix':list(prefix.shape),'suffix':list(suffix.shape),'loss':list(official.shape),'full_mask':list(fullmask.shape),'cache_leaves':[list(x.shape) for x in jax.tree.leaves(cache)]},
 'mask_checks':'prefix cannot read suffix; state cannot read action; actions read actions bidirectionally','cache_vs_full_max_error':cache_max,'manual_flow_loss_matches':True,
 'updates':501,'initial_new_key_flow_mse':initial_loss,'final_same_new_key_flow_mse':final_loss,'first_gradient_norms':first_grad,
 'initial_fixed_noise_sample_mse':initial_sample_mse,'final_fixed_noise_sample_mse':float(jnp.mean((final_sample-actions)**2)),'new_noise_same_conditions_sample_mse':float(jnp.mean((new_sample-actions)**2)),
 'sampling_steps':10,'dt':-.1,'sample_goal_extremes_dim0':np.asarray(final_sample[[0,3],:,0]).tolist(),
 'limitations':['synthetic RGB/IDs/states, not real data or pretrained semantics','same four conditions for training and sampling, no held-out task generalization','official tiny configurations; vision constructor adapter; not 3.3B model','train=False disables data augmentation only; gradients/optimizer still execute','JAX path only; PyTorch imported for repo dependencies but not executed','no paper benchmark or robot evaluation']}
(OUT/'结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
(OUT/'训练曲线.json').write_text(json.dumps(curve,indent=2),encoding='utf-8')
params=nnx.state(model,nnx.Param)
(OUT/'tiny_pi0_params.msgpack').write_bytes(serialization.to_bytes(params.to_pure_dict()))
np.savez_compressed(OUT/'evaluation_fixture.npz',state=np.asarray(state),tokens=np.asarray(obs.tokenized_prompt),token_mask=np.asarray(obs.tokenized_prompt_mask),image=np.asarray(images['base_0_rgb']),noise=np.asarray(noise),target=np.asarray(actions),expected=np.asarray(final_sample))
print(json.dumps(report,ensure_ascii=False,indent=2));print('PASS: official Pi0 loss, masks/cache, gradients, training and cached Euler sampling.')