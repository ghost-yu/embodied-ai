import json,sys
from pathlib import Path
import torch,numpy as np
sys.stdout.reconfigure(encoding='utf-8')
P=Path(__file__).resolve().parents[2];O=P/'复现'/'outputs';sys.path.insert(0,str(P/'代码'/'CogACT'))
from action_model.action_model import ActionModel
torch.set_num_threads(4)
r=json.loads((O/'结果.json').read_text(encoding='utf-8'));m=ActionModel(**r['config']);m.load_state_dict(torch.load(O/'dit_s_state.pt',map_location='cpu',weights_only=True),strict=True);m.eval();d=m.create_ddim(10)
f=torch.load(O/'evaluation_fixture.pt',map_location='cpu',weights_only=True)
B=f['cognition'].shape[0];z=torch.cat([f['cognition'],m.net.z_embedder.uncondition[None].expand(B,-1,-1)])
noise=torch.cat([f['initial_noise'],f['initial_noise']])
with torch.no_grad():out=d.ddim_sample_loop(m.net.forward_with_cfg,noise.shape,noise=noise,clip_denoised=False,model_kwargs={'z':z,'cfg_scale':1.5},device=torch.device('cpu'),progress=False,eta=0.)[:B]
assert torch.isfinite(out).all()
report={'scope':'same eight synthetic conditions, official CFG 1.5, no clipping','cfg_scale':1.5,'cfg_sample_mse':float(((out-f['target'])**2).mean()),'nonguided_sample_mse':r['final_fixed_noise_sample_mse'],'sample_extremes_dim0':out[[0,-1],:,0].tolist(),'real_task_evaluation':False}
(O/'CFG评估.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='sample_extremes_dim0'},ensure_ascii=False,indent=2))