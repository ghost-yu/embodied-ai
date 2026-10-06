"""直接运行官方 HF OpenVLA 类的缩小随机配置；不加载 7B 权重。"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import importlib.util,json,hashlib,platform,subprocess,sys,inspect
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import transformers,timm,tokenizers
from transformers import PreTrainedTokenizerFast,LogitsProcessor,LogitsProcessorList
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
sys.stdout.reconfigure(encoding='utf-8')
PAPER=Path(__file__).resolve().parents[2];REPO=PAPER/'代码'/'openvla';OUT=PAPER/'复现'/'outputs'
OUT.mkdir(parents=True,exist_ok=True)
torch.manual_seed(31);torch.set_num_threads(4)
hf=REPO/'prismatic/extern/hf'
Config=get_class_from_dynamic_module('configuration_prismatic.OpenVLAConfig',str(hf),local_files_only=True)
Model=get_class_from_dynamic_module('modeling_prismatic.OpenVLAForActionPrediction',str(hf),local_files_only=True)
loaded_source_files={}
for cls,name in [(Config,'configuration_prismatic.py'),(Model,'modeling_prismatic.py')]:
 loaded_file=Path(inspect.getfile(cls))
 loaded_digest=hashlib.sha256(loaded_file.read_bytes()).hexdigest()
 assert loaded_digest==hashlib.sha256((hf/name).read_bytes()).hexdigest()
 loaded_source_files[name]={'path':str(loaded_file),'sha256':loaded_digest}
# 真实 HF tokenizer，词表仅作动作 tokenizer 的合成 fixture，不冒充 Llama tokenizer。
vocab={f'tok{i}':i for i in range(32000)}
vocab['[UNK]']=vocab.pop('tok0')
backend=Tokenizer(WordLevel(vocab=vocab,unk_token='[UNK]'));backend.pre_tokenizer=Whitespace()
tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]')
f=REPO/'prismatic/vla/action_tokenizer.py'
spec=importlib.util.spec_from_file_location('official_action_tokenizer',f);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
act_tokenizer=module.ActionTokenizer(tokenizer)
probe=np.array([-1,0,1],np.float32)
encoded=act_tokenizer(probe)
ids=np.array(tokenizer(encoded,add_special_tokens=False).input_ids)
np.testing.assert_array_equal(ids,[31999,31872,31744])
np.testing.assert_allclose(act_tokenizer.decode_token_ids_to_actions(ids),[-254/255,0,254/255],atol=1e-8)

stats={'toy':{'action':{'q01':[-.2,-.3,-.4,-.5,-.6,-.7,0.],
 'q99':[.6,.3,.4,.5,.6,.7,1.], 'mask':[True]*6+[False]}}}
config=Config(vision_backbone_id='dinosiglip-vit-so-224px',llm_backbone_id='llama2-7b-pure',
 use_fused_vision_backbone=True,norm_stats=stats,pad_token_id=32000,pad_to_multiple_of=64,
 text_config={'vocab_size':32064,'hidden_size':64,'intermediate_size':128,'num_hidden_layers':2,
 'num_attention_heads':4,'num_key_value_heads':4,'max_position_embeddings':128,'pad_token_id':32000,'bos_token_id':1,'eos_token_id':2})
# 只改测试配置：保留官方 fused/backbone/projector/forward，两个随机 tiny ViT 替代巨大 DINO/SigLIP。
config.timm_model_ids=['vit_tiny_patch16_224','vit_tiny_patch16_224']
config.image_sizes=[32,32];config.timm_override_act_layers=[None,None]
config._attn_implementation='eager'
model=Model(config).cpu()
config.save_pretrained(OUT/'tiny_config')
shapes={}
def capture(name):
 def hook(mod,inputs,result):shapes[name]=list(result.shape)
 return hook
handles=[model.vision_backbone.register_forward_hook(capture('fused_features')),model.projector.register_forward_hook(capture('projected_features'))]
# 两个 3-channel 输入打包为 6 channels；不是同一个标准 RGB 预处理。
pixels=torch.randn(1,6,32,32)
prompt=torch.tensor([[1,5,8,29871]],dtype=torch.long)
action_ids=torch.tensor([[31999,31847,31872,31872,31872,31872,31745]],dtype=torch.long)
input_ids=torch.cat([prompt,action_ids],dim=1)
labels=input_ids.clone();labels[:,:prompt.shape[1]]=-100
# 官方collator：两个不同文本长度样本，右padding与labels=-100。
f=REPO/'prismatic/util/data_utils.py'
spec=importlib.util.spec_from_file_location('official_collator',f);collator_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=collator_module;spec.loader.exec_module(collator_module)
collator=collator_module.PaddedCollatorForActionPrediction(model_max_length=128,pad_token_id=32000)
batch=collator([{'input_ids':input_ids[0],'labels':labels[0],'pixel_values':pixels[0]},
 {'input_ids':input_ids[0,1:],'labels':labels[0,1:],'pixel_values':pixels[0]}])
assert list(batch['input_ids'].shape)==[2,11]
assert int(batch['input_ids'][1,-1])==32000 and int(batch['labels'][1,-1])==-100 and not bool(batch['attention_mask'][1,-1])
model.train()
output=model(input_ids=input_ids,pixel_values=pixels,attention_mask=torch.ones_like(input_ids),labels=labels,return_dict=True,use_cache=False)
assert list(output.logits.shape)==[1,15,32064]
assert torch.isfinite(output.loss)
output.loss.backward()
grad_norms={}
for name,group in [('vision',model.vision_backbone),('projector',model.projector),('language',model.language_model)]:
 grads=[p.grad for p in group.parameters() if p.grad is not None]
 assert grads and all(torch.isfinite(g).all() for g in grads)
 grad_norms[name]=float(torch.sqrt(sum((g.float()**2).sum() for g in grads)))
 assert grad_norms[name]>0
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
optimizer.step();optimizer.zero_grad(set_to_none=True)
model.eval()
# 教学附加 mask：官方 predict_action 不主动将 logits 屏蔽到动作 bank。
class OnlyActionTokens(LogitsProcessor):
 def __call__(self,input_ids,scores):
  result=torch.full_like(scores,-torch.inf);result[:,31744:32000]=scores[:,31744:32000];return result
kwargs={'pixel_values':pixels,'do_sample':False,'use_cache':True,'logits_processor':LogitsProcessorList([OnlyActionTokens()]),'pad_token_id':32000}
with torch.inference_mode():
 generated=model.generate(prompt,max_new_tokens=7,**kwargs)
 actual=model.predict_action(prompt,unnorm_key='toy',**kwargs)
 predicted_ids=generated[0,-7:].numpy()
 normalized=act_tokenizer.decode_token_ids_to_actions(predicted_ids)
 low=np.array(stats['toy']['action']['q01']);high=np.array(stats['toy']['action']['q99']);mask=np.array(stats['toy']['action']['mask'])
 expected=np.where(mask,.5*(normalized+1)*(high-low)+low,normalized)
np.testing.assert_allclose(actual,expected,atol=1e-8)
with torch.inference_mode():
 appended_action=model.predict_action(prompt[:,:-1],unnorm_key='toy',**kwargs)
np.testing.assert_allclose(appended_action,actual,atol=1e-8)
assert len(actual)==7 and bool(np.all((predicted_ids>=31744)&(predicted_ids<32000)))
# 数据统计集选择：多集 None 必须报错，显式key可选。
try:model._check_unnorm_key({'toy':{},'other':{}},None)
except AssertionError:ambiguous_key='rejected'
else:raise AssertionError('ambiguous key should be rejected')
for h in handles:h.remove()
report={'recorded_at_local':datetime.now().astimezone().isoformat(),'loaded_official_sources':loaded_source_files,'scope':'official HF OpenVLA classes; tiny random configuration; not pretrained 7B performance',
 'environment':{'python':sys.version,'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,
 'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'timm':timm.__version__,'numpy':np.__version__,'device':'CPU','os':platform.platform()},
 'seed':31,'repo_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
 'source_sha256':{str(f.relative_to(REPO)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [hf/'configuration_prismatic.py',hf/'modeling_prismatic.py',REPO/'prismatic/vla/action_tokenizer.py',REPO/'prismatic/util/data_utils.py']},
 'tokenizer_probe':{'encoded':encoded,'ids':ids.tolist(),'decoded':act_tokenizer.decode_token_ids_to_actions(ids).tolist()},
 'parameter_count':sum(p.numel() for p in model.parameters()),'vision_models':config.timm_model_ids,'shape_trail':shapes,
 'pixel_shape':list(pixels.shape),'input_ids_shape':list(input_ids.shape),'logits_shape':list(output.logits.shape),
 'loss_before_single_update':float(output.loss.detach()),'gradient_norms':grad_norms,'optimizer_updates':1,
 'generated_ids':predicted_ids.tolist(),'normalized_actions':normalized.tolist(),'predict_action_output':actual.tolist(),
 'collator_shapes':{k:list(v.shape) for k,v in batch.items()},'right_padding_labels_and_attention':'passed',
 'missing_29871_insertion':'passed','postprocess_matches_manual':'passed','ambiguous_unnorm_key':ambiguous_key,
 'extra_generation_constraint':'OnlyActionTokens is demo-added, not original OpenVLA default',
 'limits':['tiny randomly initialized vision/LLM','synthetic pixels and numeric prompt IDs','synthetic statistics, not any real dataset','no pretrained weights, real data, simulation or robot evaluation']}
(OUT/'结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2));print('PASS: official forward, backward, cached generation, action tokenizer, and predict_action postprocess.')
