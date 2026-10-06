"""Official ActionModel DiT-S, epsilon loss, DDIM/CFG and official adaptive ensemble."""
import hashlib,inspect,importlib.util,json,subprocess,sys,time
from pathlib import Path
import numpy as np,torch,timm
sys.stdout.reconfigure(encoding='utf-8')
PAPER=Path(__file__).resolve().parents[2];REPO=PAPER/'代码'/'CogACT';OUT=PAPER/'复现'/'outputs'
sys.path.insert(0,str(REPO))
from action_model.action_model import ActionModel
spec=importlib.util.spec_from_file_location('official_adaptive',REPO/'sim_cogact'/'adaptive_ensemble.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
AdaptiveEnsembler=module.AdaptiveEnsembler

torch.manual_seed(73);torch.set_num_threads(4)
config={'token_size':4096,'model_type':'DiT-S','in_channels':7,'future_action_window_size':15,'past_action_window_size':0,'diffusion_steps':100,'noise_schedule':'squaredcos_cap_v2'}
model=ActionModel(**config).cpu()
B,H,D=8,16,7
z=torch.linspace(-.7,.7,B)
cognition=torch.zeros(B,1,4096);cognition[:,:,0]=z[:,None];cognition[:,:,1]=1.;cognition[:,:,2]=.5*z[:,None]
clean=torch.stack([z,.5*z,-.3*z,.2*z,torch.zeros(B),torch.zeros(B),.5+.2*z],dim=-1)[:,None,:].repeat(1,H,1)
# Exact official random draw/epsilon MSE reconstruction in eval mode (no condition dropout).
model.eval();torch.manual_seed(730)
noise=torch.randn_like(clean);timesteps=torch.randint(0,100,(B,))
noisy=model.diffusion.q_sample(clean,timesteps,noise)
a=torch.tensor(model.diffusion.alphas_cumprod,dtype=torch.float32)[timesteps][:,None,None]
np.testing.assert_allclose(noisy.numpy(),(a.sqrt()*clean+(1-a).sqrt()*noise).numpy(),atol=1e-6)
trace=[]
hook=model.net.blocks[0].register_forward_pre_hook(lambda m,args:trace.append(list(args[0].shape)))
pred=model.net(noisy,timesteps,cognition);hook.remove()
manual=((pred-noise)**2).mean()
torch.manual_seed(730);actual=model.loss(clean,cognition)
np.testing.assert_allclose(float(actual),float(manual),rtol=1e-6)
assert trace[0]==[B,17,384] and pred.shape==clean.shape
actual.backward();initial_output_grad=float(model.net.final_layer.linear.weight.grad.norm());initial_condition_grad=float(model.net.z_embedder.linear.weight.grad.norm())
assert initial_output_grad>0 and initial_condition_grad==0
model.zero_grad(set_to_none=True)
# Explicitly forced label dropout must produce official learned uncondition feature.
forced=model.net.z_embedder.token_drop(cognition,torch.ones(B,dtype=torch.long))
np.testing.assert_allclose(forced.detach().numpy(),model.net.z_embedder.uncondition[None].expand(B,-1,-1).detach().numpy())
ddim=model.create_ddim(10)
initial_noise=torch.randn(B,H,D,generator=torch.Generator().manual_seed(731))

def sample(initial,condition):
 model.eval()
 with torch.no_grad():out=ddim.ddim_sample_loop(model.net.forward,initial.shape,noise=initial.clone(),clip_denoised=False,model_kwargs={'z':condition},progress=False,device=torch.device('cpu'),eta=0.)
 return out
initial_sample=sample(initial_noise,cognition)
initial_mse=float(((initial_sample-clean)**2).mean())
validation_noise=torch.randn_like(clean);validation_times=torch.tensor([10,20,30,40,50,60,70,80])
validation_input=model.diffusion.q_sample(clean,validation_times,validation_noise)
def eps_mse():
 model.eval()
 with torch.no_grad():return float(((model.net(validation_input,validation_times,cognition)-validation_noise)**2).mean())
initial_eps=eps_mse()
optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=0.)
curve=[];started=time.time()
for step in range(901):
 model.train();loss=model.loss(clean,cognition);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
 assert torch.isfinite(loss)
 curve.append({'step':step,'epsilon_mse':float(loss.detach())})
 if step%50==0:print(json.dumps({'step':step,'loss':float(loss.detach()),'elapsed':round(time.time()-started,1)}),flush=True)
final_eps=eps_mse();assert final_eps<initial_eps
calls=[]
hook=model.net.t_embedder.register_forward_pre_hook(lambda m,args:calls.append(args[0].detach().tolist()))
final_sample=sample(initial_noise,cognition);hook.remove()
assert torch.isfinite(final_sample).all() and len(calls)==10
new_sample=sample(torch.randn(B,H,D,generator=torch.Generator().manual_seed(732)),cognition)
# CFG formula uses conditional/unconditional pair with matching noisy inputs.
model.eval();half=initial_noise[:2];pair=torch.cat([half,half]);condition=cognition[:2]
uncondition=model.net.z_embedder.uncondition[None].expand(2,-1,-1)
z_pair=torch.cat([condition,uncondition],dim=0);t_pair=torch.ones(4,dtype=torch.long)*50
with torch.no_grad():
 raw=model.net(pair,t_pair,z_pair)
 cfg=model.net.forward_with_cfg(pair,t_pair,z_pair,1.5)
 expected=raw[2:]+1.5*(raw[:2]-raw[2:])
np.testing.assert_allclose(cfg[:2].numpy(),expected.numpy(),atol=1e-6);np.testing.assert_allclose(cfg[:2].numpy(),cfg[2:].numpy(),atol=1e-6)
with torch.no_grad():cfg_sample=ddim.ddim_sample_loop(model.net.forward_with_cfg,pair.shape,noise=pair.clone(),clip_denoised=False,model_kwargs={'z':z_pair,'cfg_scale':1.5},device=torch.device('cpu'),progress=False,eta=0.)[:2]
assert torch.isfinite(cfg_sample).all()
# Gradients after nonzero output head training; the history_embedder is intentionally unused.
model.train();value=model.loss(clean,cognition);model.zero_grad(set_to_none=True);value.backward()
gradients={}
for name,sub in [('condition',model.net.z_embedder),('time',model.net.t_embedder),('action',model.net.x_embedder),('first_block',model.net.blocks[0]),('output',model.net.final_layer)]:
 gs=[p.grad for p in sub.parameters() if p.grad is not None];assert gs and all(torch.isfinite(g).all() for g in gs)
 gradients[name]=float(torch.sqrt(sum((g*g).sum() for g in gs)));assert gradients[name]>0
history_unused=all(p.grad is None for p in model.net.history_embedder.parameters())
# Exercise official chunk alignment/history/reset with an independent manual computation.
ensembler=AdaptiveEnsembler(3,.1);chunks=[]
for i in range(3):
 chunk=np.zeros((16,7));chunk[:,0]=i*10+np.arange(16);chunk[:,1]=i+1;chunk[:,6]=.5;chunks.append(chunk)
 result=ensembler.ensemble_action(chunk)
aligned=np.stack([chunks[0][2],chunks[1][1],chunks[2][0]])
ref=aligned[-1];cos=(aligned*ref).sum(1)/(np.linalg.norm(aligned,axis=1)*np.linalg.norm(ref)+1e-7)
weights=np.exp(.1*cos);weights/=weights.sum();expected_ensemble=(aligned*weights[:,None]).sum(0)
np.testing.assert_allclose(result,expected_ensemble,atol=1e-8)
assert len(ensembler.action_history)==3;ensembler.reset();assert len(ensembler.action_history)==0
source_files=['action_model/action_model.py','action_model/models.py','action_model/__init__.py','action_model/gaussian_diffusion.py','action_model/respace.py','sim_cogact/adaptive_ensemble.py']
assert Path(inspect.getfile(ActionModel)).resolve()==(REPO/source_files[0]).resolve()
assert Path(inspect.getfile(AdaptiveEnsembler.ensemble_action)).resolve()==(REPO/source_files[5]).resolve()
report={'reference_date':'2026-10-03','repo_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),'environment':{'python':sys.version,'torch':torch.__version__,'timm':timm.__version__,'numpy':np.__version__,'device':'CPU'},
 'config':config,'parameter_count':sum(p.numel() for p in model.parameters()),'source_sha256':{f:hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in source_files},
 'shapes':{'cognition':[B,1,4096],'action':[B,16,7],'DiT_sequence':trace[0]},'q_sample_formula':'passed','official_epsilon_loss_matches':'passed','forced_condition_dropout':'passed',
 'initial_head_gradient':initial_output_grad,'initial_condition_gradient':initial_condition_grad,'updates':901,'initial_validation_epsilon_mse':initial_eps,'final_validation_epsilon_mse':final_eps,
 'initial_fixed_noise_sample_mse':initial_mse,'final_fixed_noise_sample_mse':float(((final_sample-clean)**2).mean()),'new_noise_same_conditions_sample_mse':float(((new_sample-clean)**2).mean()),
 'DDIM_original_timesteps':calls,'DDIM_timestep_map':ddim.timestep_map,'CFG_formula_and_batch_pair':'passed','CFG_sample_shape':list(cfg_sample.shape),'post_training_gradient_norms':gradients,'history_embedder_unused':history_unused,
 'ensemble_aligned_candidates':aligned.tolist(),'ensemble_weights':weights.tolist(),'ensemble_result':result.tolist(),'ensemble_history_and_reset':'passed',
 'sample_first_dimension_extremes':final_sample[[0,-1],:,0].tolist(),
 'limitations':['synthetic cognition features, not actual VLM/image/text','same eight training conditions used for evaluation, not held-out task generalization','official DiT-S random initialization, not paper DiT-B or pretrained checkpoint','no full CogACT.forward or real VLM hidden extraction','no robot normalization/unit conversion/command or SIMPLER benchmark']}
(OUT/'结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'训练曲线.json').write_text(json.dumps(curve,indent=2),encoding='utf-8')
torch.save(model.state_dict(),OUT/'dit_s_state.pt')
torch.save({'cognition':cognition,'initial_noise':initial_noise,'target':clean,'expected_sample':final_sample},OUT/'evaluation_fixture.pt')
print(json.dumps(report,ensure_ascii=False,indent=2));print('PASS: official CogACT action core training, DDIM/CFG, ensemble alignment/history/reset.')