"""官方 MDT Transformer + EDM wrapper + DDIM；缩小配置与合成特征，非完整论文。"""
import os
os.environ['MPLBACKEND']='Agg'
import hashlib,inspect,json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
import hydra,einops
from omegaconf import OmegaConf
sys.stdout.reconfigure(encoding='utf-8')
PAPER=Path(__file__).resolve().parents[2];REPO=PAPER/'代码'/'mdt_policy'
OUT=PAPER/'复现'/'outputs';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(REPO))
from mdt.models.networks.mdtv_transformer import MDTVTransformer
from mdt.models.edm_diffusion.score_wrappers import GCDenoiser
from mdt.models.edm_diffusion.gc_sampling import get_sigmas_exponential,sample_ddim

torch.manual_seed(43);torch.set_num_threads(4)
config={
 '_target_':'mdt.models.networks.mdtv_transformer.MDTVTransformer',
 'obs_dim':8,'goal_dim':12,'device':'cpu','n_obs_token':3,'goal_conditioned':True,
 'action_dim':7,'proprio_dim':4,'embed_dim':64,'embed_pdrob':0.,'attn_pdrop':0.,'resid_pdrop':0.,'mlp_pdrop':0.,
 'n_dec_layers':2,'n_enc_layers':2,'n_heads':4,'goal_seq_len':1,'obs_seq_len':1,'action_seq_len':10,
 'goal_drop':0.,'bias':False,'use_mlp_goal':True,'use_abs_pos_emb':True,'use_rot_embed':False,
 'linear_output':True,'use_ada_conditioning':True,'use_noise_encoder':False,'use_modality_encoder':True}
model=GCDenoiser(OmegaConf.create(config),sigma_data=.5).cpu()
# 16个训练条件；目标是七维常值动作块，目的在验证目标条件去噪，不模拟真实机器人。
B,H,D=16,10,7
z=torch.linspace(-.7,.7,B)
goals_vis=torch.zeros(B,1,12);goals_vis[:,:,0]=z[:,None];goals_vis[:,:,1]=1.
# 同一语义的合成语言embedding，使用不同输入位置；不是CLIP输出或真实文本。
goals_lang=torch.zeros_like(goals_vis);goals_lang[:,:,2]=z[:,None];goals_lang[:,:,3]=1.
clean=torch.stack([z,.5*z,-.3*z,.2*z,torch.zeros(B),torch.zeros(B),torch.ones(B)*.2],dim=-1)[:,None,:].repeat(1,H,1)
state={'state_images':torch.randn(B,3,8)*.2,'state_obs':torch.zeros(B,1,4),'modality':'vis'}
noise=torch.randn_like(clean);sigma=torch.ones(B)*.5
# 验证wrapper的训练target和forward组合，不假设inner_model输出直接是noise。
model.eval()
loss,inner_output=model.loss(state,clean,goals_vis,noise,sigma)
cs,co,ci=[v[:,None,None] for v in model.get_scalings(sigma)]
x=clean+noise*sigma[:,None,None]
expected_target=(clean-cs*x)/co
np.testing.assert_allclose(float(loss),float(((inner_output-expected_target)**2).mean()),rtol=1e-5)
den=model(state,x,goals_vis,sigma)
np.testing.assert_allclose(den.detach().numpy(),(cs*x+co*inner_output).detach().numpy(),atol=1e-6)
assert tuple(model.inner_model.latent_encoder_emb.shape)==(B,5,64)
# 梯度更新前，检查核心模块并记录位置参数是否实际参与forward。
model.train();loss.backward()
gradients={}
for name,submodule in [('encoder',model.inner_model.encoder),('decoder',model.inner_model.decoder),('goal_embedding',model.inner_model.goal_emb),('sigma_embedding',model.inner_model.sigma_emb),('action_output',model.inner_model.action_pred)]:
 g=[p.grad for p in submodule.parameters() if p.grad is not None]
 assert g and all(torch.isfinite(v).all() for v in g)
 gradients[name]=float(torch.sqrt(sum((v*v).sum() for v in g)))
position_grad_unused=model.inner_model.pos_emb.grad is None
model.zero_grad(set_to_none=True)
optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=0.)
# 固定不同seed产生的验证噪声，不用训练loss冒充去噪评估。
gen=torch.Generator().manual_seed(440)
validation_noise=torch.randn(clean.shape,generator=gen)
validation_sigma=torch.ones(B)*.5

def denoise_mse():
 model.eval()
 with torch.no_grad():
  total=[]
  for modality,goals in [('vis',goals_vis),('lang',goals_lang)]:
   state['modality']=modality
   result=model(state,clean+validation_noise*.5,goals,validation_sigma)
   total.append(float(((result-clean)**2).mean()))
 return dict(zip(['vis','lang'],total))
initial_mse=denoise_mse()
curve=[]
for step in range(241):
 model.train();state['modality']='vis' if step%2==0 else 'lang'
 goal=goals_vis if step%2==0 else goals_lang
 sigma=torch.exp(torch.rand(B)*(np.log(1.)-np.log(.03))+np.log(.03))
 train_noise=torch.randn_like(clean)
 loss,_=model.loss(state,clean,goal,train_noise,sigma)
 optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
 assert torch.isfinite(loss)
 curve.append({'step':step,'modality':state['modality'],'inner_target_loss':float(loss.detach())})
 if step%60==0:print(json.dumps(curve[-1]),flush=True)
final_mse=denoise_mse()
assert final_mse['vis']<initial_mse['vis'] and final_mse['lang']<initial_mse['lang']
model.eval()
# 从独立初始高斯噪声开始，调用官方10步DDIM；sigma末端为0，只在正sigma调用模型。
eval_state={'state_images':state['state_images'][:2],'state_obs':state['state_obs'][:2],'modality':'vis'}
goal_pair=goals_vis[[0,-1]]
init_noise=torch.randn(2,H,D,generator=gen)
sigmas=get_sigmas_exponential(10,.001,1.)
callbacks=[]
sampled=sample_ddim(model,eval_state,init_noise,goal_pair,sigmas,disable=True,callback=lambda data:callbacks.append({'step':int(data['i']),'sigma':float(data['sigma']),'denoised_mean':float(data['denoised'].mean())}))
assert sampled.shape==(2,10,7) and torch.isfinite(sampled).all() and len(callbacks)==10
# 固定同一条件/初始噪声，测试eta传参在保存实现中的实际行为。
repeat=sample_ddim(model,eval_state,init_noise.clone(),goal_pair,sigmas,disable=True,eta=0.)
np.testing.assert_allclose(sampled.detach().numpy(),repeat.detach().numpy(),atol=1e-7)

sources=['mdt/models/networks/mdtv_transformer.py','mdt/models/networks/transformers/transformer_blocks.py','mdt/models/networks/transformers/position_embeddings.py','mdt/models/edm_diffusion/score_wrappers.py','mdt/models/edm_diffusion/utils.py','mdt/models/edm_diffusion/gc_sampling.py']
for obj,relative in [(MDTVTransformer,sources[0]),(GCDenoiser,sources[3]),(sample_ddim,sources[5])]:assert Path(inspect.getfile(inspect.unwrap(obj))).resolve()==(REPO/relative).resolve()
report={'reference_date':'2026-10-03','scope':'official MDT action core in small configuration, synthetic features; no full MDTVAgent/robot evaluation',
 'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'hydra':hydra.__version__,'einops':einops.__version__,'device':'CPU'},
 'repo_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
 'source_sha256':{s:hashlib.sha256((REPO/s).read_bytes()).hexdigest() for s in sources},
 'seed':43,'config':config,'sigma_data':.5,'parameter_count':sum(p.numel() for p in model.parameters()),
 'shapes':{'state_features':list(state['state_images'].shape),'goals':list(goals_vis.shape),'action':list(clean.shape),'encoder_context':[B,5,64],'sampled':list(sampled.shape)},
 'wrapper_inner_target_and_combination':'passed','initial_gradient_norms':gradients,'pos_emb_has_no_gradient_in_this_path':position_grad_unused,
 'updates':241,'initial_same_conditions_new_noise_mse':initial_mse,'final_same_conditions_new_noise_mse':final_mse,
 'sampler_steps':len(callbacks),'sigma_schedule':sigmas.tolist(),'eta0_equals_eta1_for_fixed_inputs':True,
 'sampled_first_action_dimension':sampled[:,:,0].tolist(),'target_first_action_dimension':[-.7,.7],
 'sampled_goal_pair_mse':float(((sampled-clean[[0,-1]])**2).mean()),
 'limitations':['synthetic features, not image/language encoders','same 16 training conditions used for new-noise evaluation','no MGF/CLA training or full MDTVAgent construction','tiny parameters, changed noise schedule, no pretrained checkpoint or CALVIN/LIBERO']}
(OUT/'结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'训练曲线.json').write_text(json.dumps(curve,indent=2),encoding='utf-8')
(OUT/'采样记录.json').write_text(json.dumps(callbacks,indent=2),encoding='utf-8')
(OUT/'demo配置.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
torch.save(model.state_dict(),OUT/'tiny_mdt_state.pt')
torch.save({'state':eval_state,'goals':goal_pair,'initial_noise':init_noise,'sigmas':sigmas,'expected_sample':sampled},OUT/'evaluation_fixture.pt')
print(json.dumps(report,ensure_ascii=False,indent=2));print('PASS: official Transformer/EDM forward, gradients, synthetic training, and DDIM sampling.')
