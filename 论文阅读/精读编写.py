import ast, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PAPERS=json.loads((ROOT/'下载清单.json').read_text(encoding='utf-8'))
ROLES={
'RT-1':['sequence_agent.py','transformer_network.py','tokenizers/image_tokenizer.py','tokenizers/token_learner.py','tokenizers/action_tokenizer.py','film_efficientnet/film_conditioning_layer.py','transformer.py','requirements.txt'],
'OpenVLA':['vla-scripts/finetune.py','vla-scripts/train.py','prismatic/vla/datasets/datasets.py','prismatic/vla/action_tokenizer.py','prismatic/models/vlas/openvla.py','prismatic/models/vlms/prismatic.py','prismatic/models/backbones/vision/dinosiglip_vit.py','experiments/robot/libero/run_libero_eval.py','pyproject.toml','prismatic/extern/hf/configuration_prismatic.py','prismatic/extern/hf/modeling_prismatic.py','prismatic/util/data_utils.py'],
'MDT':['mdt/training.py','conf/config.yaml','conf/model/mdtv_agent.yaml','mdt/datasets/base_dataset.py','mdt/models/mdtv_agent.py','mdt/models/networks/mdtv_transformer.py','mdt/models/edm_diffusion/score_wrappers.py','mdt/models/img_generation/masked_transformer_decoder.py','mdt/evaluation/mdt_evaluate.py'],
'RDT-1B':['main.py','configs/base.yaml','configs/state_vec.py','train/dataset.py','train/train.py','models/rdt_runner.py','models/rdt/model.py','models/rdt/blocks.py','eval_sim/eval_rdt_maniskill.py','requirements.txt'],
'pi0':['scripts/train.py','src/openpi/training/config.py','src/openpi/training/data_loader.py','src/openpi/transforms.py','src/openpi/models/pi0.py','src/openpi/models/pi0_config.py','src/openpi/policies/policy.py','examples/libero/README.md','pyproject.toml'],
'CogACT':['scripts/train.py','vla/cogactvla.py','action_model/action_model.py','training/strategies/base_strategy_cogact.py','sim_cogact/cogact_policy.py','pyproject.toml'],
'I-JEPA':['main.py','configs/in1k_vith14_ep300.yaml','src/helper.py','src/train.py','src/datasets/imagenet1k.py','src/masks/multiblock.py','src/models/vision_transformer.py','src/utils/tensors.py'],
}

def write(short,body):
 r=next(x for x in PAPERS if x['short']==short);folder=Path(r['folder'])
 old=folder/'解释.md'
 backup=folder/'资料'/'阅读前导读.md'
 if not backup.exists(): backup.write_text(old.read_text(encoding='utf-8'),encoding='utf-8')
 header=f"# {r['title']}：论文与代码精读\n\n## 来源与阅读边界\n\n原论文：[arXiv {r['arxiv']}{r.get('version','')}](https://arxiv.org/abs/{r['arxiv']}{r.get('version','')})；本地 [原论文.pdf](<{(folder/'原论文.pdf').as_posix()}>)。此文以已保存的 PDF 为准，页码指 PDF 文件页码。版本和 SHA-256 见 `资料/来源.json`。\n\n"
 if r.get('repo'):header+=f"官方代码：[ {r['repo']} ](https://github.com/{r['repo']})，固定 commit `{r['commit']}`。本文的代码行为只对这个快照负责，不能推定所有版本都相同。\n\n"
 else:header+='本轮未找到可确认的完整官方实现。下面的公式解释与概念算法不能当成官方源代码。\n\n'
 header+=f"编写日期：2026-10-02。覆盖动机、核心方法、关键公式/图表、实现导航和复现边界；正文明确标注教学重写、实际代码和未核验项。跨篇基础见 [公共基础](<{(ROOT/'精读公共基础.md').as_posix()}>).\n\n"
 pages=sorted((folder/'图片').glob('精读-PDF-*.png'),key=lambda f:int(f.stem.rsplit('-',1)[1]))
 if pages:
  header+='原文图与公式核对页：'+ ' · '.join(f'[PDF第{p.stem.rsplit("-",1)[1]}页](<{p.as_posix()}>)' for p in pages)+'。这些图片直接渲染自本地论文，结合下文的箭头和变量解释阅读。\n\n'
 footer='\n## 保存版本的代码地图\n\n以下链接定位实际文件及对象；文件地图不等于运行成功。配置和参数文件需结合正文调用链阅读。\n\n|文件|实际对象与行号|\n|---|---|\n'
 if short in ROLES:
  checkout=Path(r['checkout'])
  for rel in ROLES[short]:
   f=checkout/rel;names=[]
   if not f.exists():footer+=f'|`{rel}`|未找到；仅检查了保存的仓库快照|\n';continue
   if f.suffix=='.py':
    try:
     tree=ast.parse(f.read_text(encoding='utf-8'))
     for n in tree.body:
      if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.AsyncFunctionDef)):
       names.append(f'`{n.name}`（L{n.lineno}）')
       if isinstance(n,ast.ClassDef):
        names.extend(f'`{n.name}.{m.name}`（L{m.lineno}）' for m in n.body if isinstance(m,ast.FunctionDef) and m.name in ['forward','call','compute_loss','loss','sample_actions','predict_action','tokenize','detokenize','__init__'])
    except Exception as e:names=['AST解析失败：'+type(e).__name__]
   footer+=f'|[{rel}](<{f.as_posix()}>)|'+('；'.join(names[:16]) or '配置/文档')+'|\n'
 else:footer+='|模型、数据、loss、训练和评估入口|未找到可确认的完整官方实现；检查范围见正文|\n'
 footer+='''
## 复现记录与置信度

**状态：未复现。** 已检查本地原论文、源码快照和关键表达式；独立数学例子的结果记录在全局校验文件中。未执行官方模型 import/forward、未进行论文训练或评测，因此不把源码阅读和 NumPy 例子记成“部分成功”的模型复现。

当前硬件与 Python 环境见公共基础。复现入口信息保留在本篇 `复现/README.md`；未下载新的大型数据集或模型权重。若仓库自带模型文件，是否完整可用仍需验证。涉及 Linux、CUDA 扩展或分布式训练的命令不能直接照搬到现有 Windows Python 环境。

置信度：原文与所列代码可直接支持的结论较高；教学公式和小例子用于说明机制，不是论文实验；未追踪到的数据坐标系、完整训练配置、所有附录实验和硬件运行行为会在正文或后续问题中注明。读完此文仍应沿一个真实样本完成 shape 日志和端到端检查。
'''
 state_path=folder/'复现'/'status.json'
 if state_path.exists():
  state=json.loads(state_path.read_text(encoding='utf-8'))
  footer=footer[:footer.index('## 复现记录与置信度')]+f"## 复现记录与置信度\n\n状态：**{state['status']}**（{state['scope']}）。证据日期：{state['date']}。详见 [复现记录](<{(folder/'复现'/'README.md').as_posix()}>)。官方模块运行与完整论文实验必须分别判断。未解决问题见正文。\n"
 old.write_text(header+body+footer,encoding='utf-8')
 if not (folder/'复现'/'README.md').exists(): (folder/'复现'/'README.md').write_text(f'# {short} 复现记录\n\n状态：未复现。\n\n证据：原论文方法与实验阅读、固定代码 commit 的静态定位；没有运行官方模型。环境：Windows，RTX 4060 Laptop GPU 8188 MiB；系统 Python 3.7.6，文档工具 Python 3.12.14。详见公共基础。\n\n后续先建立官方要求的独立环境，验证 import，再尝试最小 forward 或仿真 demo。每次记录命令、工作目录、配置、输出、失败位置与结果范围。\n',encoding='utf-8')

if __name__=='__main__':
 for f in (ROOT/'精读正文').glob('*.md'):write(f.stem,f.read_text(encoding='utf-8'))
 print('Wrote detailed notes')
