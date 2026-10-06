"""Official RDTRunner/RDT + DDPM training + DPM-Solver; small random model, synthetic embeddings."""
import os
os.environ['MPLBACKEND']='Agg'
import hashlib,inspect,json,subprocess,sys
from pathlib import Path
import torch,numpy as np,timm,diffusers,huggingface_hub
sys.stdout.reconfigure(encoding='utf-8')
PAPER=Path(__file__).resolve().parents[2];REPO=PAPER/'代码'/'RoboticsDiffusionTransformer';OUT=PAPER/'复现'/'outputs'
OUT.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(REPO))
from models.rdt_runner import RDTRunner
from models.rdt.model import RDT
from configs.state_vec import STATE_VEC_IDX_MAPPING

torch.manual_seed(53);torch.set_num_threads(4)
config={'rdt':{'hidden_size':64,'depth':4,'num_heads':4},'lang_adaptor':'mlp2x_gelu','img_adaptor':'mlp2x_gelu','state_adaptor':'mlp3x_gelu',
 'noise_scheduler':{'num_train_timesteps':1000,'num_inference_timesteps':5,'beta_schedule':'squaredcos_cap_v2','prediction_type':'sample','clip_sample':False}}
kwargs={'action_dim':128,'pred_horizon':64,'config':config,'lang_token_dim':12,'img_token_dim':8,'state_token_dim':128,'max_lang_cond_len':4,'img_cond_len':6}
runner=RDTRunner(**kwargs,dtype=torch.float32).cpu()
B,H,D=16,64,128
z=torch.linspace(-.7,.7,B)
lang=torch.zeros(B,4,12);lang[:,0,0]=z;lang[:,0,1]=1;lang[:,1,2]=z*.5;lang[:,1,3]=1
lang[:,2:]=torch.randn(B,2,12)*.2
lang_mask=torch.tensor([True,True,False,False])[None].expand(B,-1)
img=torch.randn(B,6,8)*.05
state=torch.zeros(B,1,D)
slots=[STATE_VEC_IDX_MAPPING[f'{arm}_arm_joint_{i}_pos'] for arm in ['right','left'] for i in range(6)]
slots += [STATE_VEC_IDX_MAPPING['right_gripper_open'],STATE_VEC_IDX_MAPPING['left_gripper_open']]
mask=torch.zeros(B,1,D);mask[:,:,slots]=1
clean=torch.zeros(B,H,D)
for k,idx in enumerate(slots[:12]):clean[:,:,idx]=(1 if k<6 else -1)*z[:,None]*(1-(k%6)*.1)
clean[:,:,slots[12:]]=torch.stack([.5+.2*z,.5-.2*z],dim=-1)[:,None,:]
freq=torch.ones(B)*25
batch=(lang,lang_mask,img,state,clean,mask,freq)
# Trace real module calls and alternation, without altering model outputs.
calls=[];predictions=[]
hooks=[runner.model.register_forward_hook(lambda m,args,out:predictions.append(out.detach().clone()))]
for i,block in enumerate(runner.model.blocks):
 hooks.append(block.register_forward_pre_hook(lambda m,args,idx=i:calls.append({'block':idx,'query':list(args[0].shape),'condition':list(args[1].shape),'mask':None if args[2] is None else list(args[2].shape)})))
runner.eval();torch.manual_seed(901);loss=runner.compute_loss(*batch)
np.testing.assert_allclose(float(loss),float(((predictions[-1]-clean)**2).mean()),rtol=1e-6)
masked_mean=float((((predictions[-1]-clean)**2)*mask).sum()/(mask.sum()*H))
assert not np.isclose(float(loss),masked_mean)
# At official zero-initialized output, one invalid target slot of 1 adds exactly 1/128 to MSE.
changed=clean.clone();invalid=next(i for i in range(D) if i not in slots);changed[:,:,invalid]=1
changed_batch=(lang,lang_mask,img,state,changed,mask,freq)
torch.manual_seed(901);changed_loss=runner.compute_loss(*changed_batch)
np.testing.assert_allclose(float(changed_loss-loss),1/D,atol=1e-7)
trace=calls[:4]
for h in hooks:h.remove()
# Official output head starts at zero; record the resulting gradient boundary.
runner.train();loss.backward()
initial_head_grad=float(runner.model.final_layer.ffn_final.fc2.weight.grad.norm())
initial_upstream_grad=float(runner.lang_adaptor[0].weight.grad.norm())
assert initial_head_grad>0 and initial_upstream_grad==0
runner.zero_grad(set_to_none=True)
# Forward scheduler mathematical validation at selected timesteps.
noise=torch.randn_like(clean);t=torch.tensor([0,999]+[500]*(B-2),dtype=torch.long)
noisy=runner.noise_scheduler.add_noise(clean,noise,t)
a=runner.noise_scheduler.alphas_cumprod[t][:,None,None]
np.testing.assert_allclose(noisy.numpy(),(a.sqrt()*clean+(1-a).sqrt()*noise).numpy(),atol=1e-6)

def sample_and_metric(seed):
 runner.eval();torch.manual_seed(seed)
 with torch.no_grad():out=runner.predict_action(lang,lang_mask,img,state,mask,freq)
 assert out.shape==(B,H,D) and torch.isfinite(out).all()
 assert torch.count_nonzero(out[:,:,mask[0,0]==0])==0
 error=(out-clean).square()
 return out,{'all_slots_mse':float(error.mean()),'valid_slots_mse':float((error*mask).sum()/(mask.sum()*H))}
initial_sample,initial_metric=sample_and_metric(778)
opt=torch.optim.AdamW(runner.parameters(),lr=.0015,weight_decay=0)
curve=[]
for step in range(201):
 runner.train();value=runner(*batch);opt.zero_grad(set_to_none=True);value.backward();opt.step()
 assert torch.isfinite(value)
 curve.append({'step':step,'all_slots_training_mse':float(value.detach())})
 if step%50==0:print(json.dumps(curve[-1]),flush=True)
final_sample,final_metric=sample_and_metric(778)
assert final_metric['valid_slots_mse']<initial_metric['valid_slots_mse'],(initial_metric,final_metric)
# New initial noise, same conditions: evaluation is not unseen-task generalization.
new_sample,new_metric=sample_and_metric(779)
# Forward once with gradients after training: upstream gradient should now pass nonzero head.
runner.train();value=runner(*batch);runner.zero_grad(set_to_none=True);value.backward()
gradients={}
for name,module in [('lang_adaptor',runner.lang_adaptor),('img_adaptor',runner.img_adaptor),('state_adaptor',runner.state_adaptor),('first_block',runner.model.blocks[0]),('output',runner.model.final_layer)]:
 gs=[p.grad for p in module.parameters() if p.grad is not None]
 assert gs and all(torch.isfinite(g).all() for g in gs)
 gradients[name]=float(torch.sqrt(sum((g*g).sum() for g in gs)))
 assert gradients[name]>0
runner.eval()
# A trained model makes the language-padding test meaningful, unlike a zero output head.
state_action=torch.cat([state,clean],dim=1);m=mask.expand(-1,H+1,-1)
lc,ic,sc=runner.adapt_conditions(lang,img,torch.cat([state_action,m],dim=-1))
with torch.no_grad():
 reference=runner.model(sc,freq,torch.ones(B,dtype=torch.long)*500,lc,ic,lang_mask=lang_mask)
 altered=lang.clone();altered[:,2:]=999
 altered_lc=runner.lang_adaptor(altered)
 padding_check=runner.model(sc,freq,torch.ones(B,dtype=torch.long)*500,altered_lc,ic,lang_mask=lang_mask)
np.testing.assert_allclose(reference.numpy(),padding_check.numpy(),atol=1e-6)
source_files=['models/rdt_runner.py','models/rdt/model.py','models/rdt/blocks.py','models/hub_mixin.py','configs/state_vec.py']
assert Path(inspect.getfile(RDTRunner)).resolve()==(REPO/source_files[0]).resolve()
assert Path(inspect.getfile(RDT)).resolve()==(REPO/source_files[1]).resolve()
report={'reference_date':'2026-10-03','repo_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
 'environment':{'python':sys.version,'torch':torch.__version__,'timm':timm.__version__,'diffusers':diffusers.__version__,'huggingface_hub':huggingface_hub.__version__,'numpy':np.__version__,'device':'CPU'},
 'source_sha256':{s:hashlib.sha256((REPO/s).read_bytes()).hexdigest() for s in source_files},'constructor':kwargs,'seed':53,'parameter_count':sum(p.numel() for p in runner.parameters()),
 'shapes':{'lang':[B,4,12],'img':[B,6,8],'state':[B,1,128],'action':[B,64,128],'adapted_state_action':[B,65,64],'RDT_sequence':[B,67,64]},
 'valid_slot_indices':slots,'ACI_first_forward':trace,'training_loss_at_zero_head':float(loss),'valid_slots_mean_at_zero_head':masked_mean,
 'invalid_target_slot_added_loss':float(changed_loss-loss),'initial_output_head_gradient':initial_head_grad,'initial_language_adaptor_gradient':initial_upstream_grad,
 'scheduler_add_noise_matches_formula':True,'updates':201,'initial_same_seed_sampling':initial_metric,'final_same_seed_sampling':final_metric,
 'new_noise_same_conditions_sampling':new_metric,'post_training_gradient_norms':gradients,'masked_language_padding_max_error':float((reference-padding_check).abs().max()),
 'invalid_sample_dimensions_exact_zero':True,'inference_timesteps':runner.noise_scheduler_sample.timesteps.tolist(),
 'scheduler_runtime':dict(runner.noise_scheduler_sample.config),'sample_first_joint_goal_extremes':final_sample[[0,-1],:,slots[0]].tolist(),
 'limits':['synthetic embeddings; no T5/SigLIP','same 16 conditions in training and sampling; no held-out tasks','tiny random model, no pretrained weights','no real HDF5 data, robot unit/coordinate conversion, or robot execution','no paper benchmark reproduction']}
report['scheduler_runtime']['lambda_min_clipped']=str(report['scheduler_runtime']['lambda_min_clipped'])
(OUT/'结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
(OUT/'训练曲线.json').write_text(json.dumps(curve,indent=2),encoding='utf-8')
torch.save(runner.state_dict(),OUT/'tiny_rdt_state.pt')
torch.save({'lang':lang,'lang_mask':lang_mask,'img':img,'state':state,'mask':mask,'freq':freq,'clean':clean,'expected_sample':final_sample,'sample_seed':778},OUT/'evaluation_fixture.pt')
print(json.dumps(report,ensure_ascii=False,indent=2));print('PASS: official RDT forward/training, mask checks, ACI, five-step sampling and weights saved.')