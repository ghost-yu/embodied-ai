from pathlib import Path
import json,importlib.util
ROOT=Path(r'G:\具身\论文阅读');P=ROOT/'OpenVLA -  An Open-Source Vision-Language-Action Model'
r=json.loads((P/'复现/outputs/结果.json').read_text(encoding='utf-8'));local_date=r['recorded_at_local'][:10]
body=ROOT/'精读正文/OpenVLA.md';s=body.read_text(encoding='utf-8')
extra='''

## 9. 本篇小 demo：官方类的缩小随机配置

已在CPU直接调用保存仓库的 `OpenVLAConfig`、`OpenVLAForActionPrediction`、`ActionTokenizer` 与 `PaddedCollatorForActionPrediction`。HF动态加载的两个源文件在cache中的SHA-256与仓库原文件相同；没有用fake模型替换forward或generate。

这是约15,855,424参数的随机配置：两侧视觉骨干均为tiny ViT，输入32×32；语言模型为两层、hidden64的Llama；保留32064个输出位置以兼容官方动作词表与29871格式token。真实原模型是预训练DINO/SigLIP+7B Llama；这次没有下载那些权重。

实际shape路径：

|位置|观测shape|说明|
|---|---|---|
|两个RGB打包|`[1,6,32,32]`|合成浮点特征输入，不是官方processor处理的照片|
|双tiny融合|`[1,4,384]`|4个patch，192+192 feature|
|projector输出|`[1,4,64]`|投射到缩小LLM宽度|
|文本+动作输入ids|`[1,11]`|4个prompt位置，7个监督动作|
|多模态logits|`[1,15,32064]`|在BOS后插入4个视觉位置|
|collator的两样本ids/labels/mask|`[2,11]`|较短样本右侧padding|
|collator打包pixels|`[2,6,32,32]`|这里只检查collator，不做batch2生成|

一次AdamW更新前loss=10.34685；视觉/projector/语言三组梯度范数约2.0294、8.2196、7.7347，均有限非零。这支持动作监督能回传到三组参数，不证明训练收敛或视觉语义正确。没有以一次更新的结果评价原论文训练recipe。

官方generate走了实际缓存生成。demo额外传入 `OnlyActionTokens` logits processor，只允许31744–31999的动作ID；这个processor是教学添加，**不是原版OpenVLA.predict_action自带的约束**。原版后处理会clip解码索引，不能把clip自动当成生成端的合法词表屏蔽。

生成后调用官方predict_action返回7维数组，与独立反量化/反归一化计算逐项一致。第一维示例：ID31830 → bin中心0.3294118 → 用toy q01=−0.2、q99=0.6恢复为0.3317647。最后维mask=False，保留归一化值0.9803922。所有统计值是合成fixture，没有真实机器人物理单位。

另外通过：tokenizer的−1/0/1端点测试；缺少末尾29871时官方方法补齐；两个统计集且unnorm_key=None被拒绝；collator右padding的ids=32000、labels=−100、attention=False。完整命令与对象说明见 [小demo运行记录](<G:/具身/论文阅读/OpenVLA -  An Open-Source Vision-Language-Action Model/复现/README.md>)。

当前状态：官方类在缩小配置上部分成功；真实预训练7B模型、实际RLDS数据与真机实验仍未复现。原始PIL/指令字符串接口没有运行；本轮执行的是HF的已处理tensor接口。真实Llama tokenizer/图像processor格式对齐仍是下一层要验证的内容。
'''
if '## 9. 本篇小 demo' not in s:s+=extra
body.write_text(s,encoding='utf-8')
readme='''# OpenVLA 小 demo 与复现记录

**状态：部分成功（官方类、缩小随机配置）；完整预训练7B模型与论文实验未复现。**

执行时间见 [结果JSON](<G:/具身/论文阅读/OpenVLA -  An Open-Source Vision-Language-Action Model/复现/outputs/结果.json>) 的 `recorded_at_local`。这次是独立CPU环境，不使用系统Python和GPU。

## 已执行的命令

```powershell
uv venv --python 'C:\\Users\\hgfhg\\AppData\\Local\\Programs\\Python\\Python311\\python.exe' 'G:\\具身\\论文阅读\\.venvs\\openvla'
# Torch wheel在下载后通过PyPI JSON发布的大小与SHA-256校验，再从本地安装
uv pip install --python 'G:\\具身\\论文阅读\\.venvs\\openvla\\Scripts\\python.exe' 'G:\\具身\\论文阅读\\依赖缓存\\torch-2.2.0-cp311-cp311-win_amd64.whl'
# 检查实际依赖；结果为 All installed packages are compatible
uv pip check --python 'G:\\具身\\论文阅读\\.venvs\\openvla\\Scripts\\python.exe'
# 执行；可从G:\\具身运行
& 'G:\\具身\\论文阅读\\.venvs\\openvla\\Scripts\\python.exe' 'G:\\具身\\论文阅读\\OpenVLA -  An Open-Source Vision-Language-Action Model\\复现\\scripts\\openvla_tiny_demo.py'
```

`--python`选隔离环境，最后的脚本用自身路径找官方仓库与outputs，不需要复制官方模型代码。`requirements.txt`记录直接依赖；[环境锁定](<G:/具身/论文阅读/OpenVLA -  An Open-Source Vision-Language-Action Model/复现/outputs/环境锁定.txt>)保存完整安装版本。首次准备还用 `uv pip install --no-deps` 安装小依赖并补齐mpmath，再由pip check和实际import确认完整性；直接复用环境时无需重复这段排障。

实际版本：Python3.11.0、torch2.2.0+cpu、torchvision0.17.0+cpu、transformers4.40.1、tokenizers0.19.1、timm0.9.10、NumPy1.26.4、huggingface-hub0.23.0。

网络失败与恢复见 [环境准备记录](<G:/具身/论文阅读/OpenVLA -  An Open-Source Vision-Language-Action Model/复现/环境准备记录.md>)；经过校验的wheel信息见 [下载校验](<G:/具身/论文阅读/依赖缓存/torch-wheel校验.json>)。没有关闭证书检查，也没有使用403镜像文件。

## 源码边界和配置

官方commit：`c8f03f48af692657d3060c19588038c7220e9af9`。核心代码位于 `prismatic/extern/hf`。`get_class_from_dynamic_module` 从本地目录读取这些自包含文件并缓存；脚本用inspect取得实际执行文件，再比较其SHA-256与本地仓库。源文件和cache的hash一致；官方代码未修改。

新建的是缩小config，不是下载后裁剪7B权重：

|设置|这次demo|原论文/默认路线|
|---|---|---|
|视觉模型|两个随机vit_tiny_patch16_224|预训练DINOv2/SigLIP|
|输入尺寸|32×32，每侧3通道|原论文224×224|
|视觉patch/融合宽度|4 / 384|256 / 2176|
|语言模型|两层Llama，hidden64，intermediate128，4个head|7B Llama 2|
|词表位置|32064，反量化基数32000|保留接口的对齐关系|
|模型参数|15,855,424|不当作原7B模型|
|权重/数据|随机权重、随机pixels、数字prompt、toy统计|预训练权重、真实示范/语言/图像|

config先通过原类构造，再明确覆写timm_model_ids和image_sizes。ID仍保留原融合路线标识，但实际测试骨干列表是两个tiny；不要仅看vision_backbone_id就误以为加载了DINO/SigLIP。配置保存于outputs/tiny_config，仅供本demo记录。

官方HF源码提示其初始化不适合论文从头训练。这里做前向与一次梯度更新来验证执行通路，不把随机配置训练称为OpenVLA原训练配方。

## 从脚本读数据路径

[demo源码](<G:/具身/论文阅读/OpenVLA -  An Open-Source Vision-Language-Action Model/复现/scripts/openvla_tiny_demo.py>)：本地加载官方类 → 建立实际WordLevel tokenizer → 官方ActionTokenizer端点核对 → 缩小config/模型 → collator → forward/loss/backward → 一次AdamW更新 → eval/generate → predict_action →独立后处理核对。

WordLevel tokenizer是实际PreTrainedTokenizerFast实例，拥有32000个合成词表项；它只用来测试动作ID编码，不是Llama tokenizer，更没有验证自然语言prompt格式。HF模型的prompt直接给数字IDs；pixel_values是合成浮点tensor，没有调用真实图像processor。

|对象/方法|输入与默认值|返回|状态和调用位置|
|---|---|---|---|
|ActionTokenizer.__call__|浮点动作数组|tokenizer解码的字符串/列表|无学习参数；构造动作目标|
|decode_token_ids_to_actions|整数IDs NumPy数组|同shape归一化动作|词表逆偏移与中心查表|
|PaddedCollatorForActionPrediction.__call__|样本列表，右padding，max_length128|pixels/ids/labels/mask batch字典|本次两个不同长度样本；不修改模型|
|OpenVLAForActionPrediction(config)|config与toy norm_stats|nn.Module模型|首次构造初始化随机参数|
|model.forward|ids/pixels/attention_mask/labels，use_cache=False|loss、logits等ModelOutput|这里train模式；不自行optimizer更新|
|loss.backward|标量loss|无返回；grad写入参数|autograd反传到视觉/projector/语言|
|AdamW.step|参数已有grad，lr0.0001|无任务输出|更新权重/optimizer状态，随后zero_grad|
|model.generate|prompt、pixels、max_new_tokens7、do_sampleFalse、use_cacheTrue|完整生成ID序列|eval下真实自回归缓存；不学习|
|model.predict_action|ids、unnorm_key toy、生成kwargs|NumPy七维动作|补29871→generate→解码→反归一化|
|_check_unnorm_key|norm_stats、key|选定key或assert|多集无key拒绝，避免任意选择统计|

`torch.manual_seed(31)`固定本次随机源；`.train()`/`.eval()`控制层行为，不能等同于“已经执行训练”。`loss.backward()`只写梯度，optimizer.step才改变参数。`register_forward_hook`记录模块实际输出shape；不会替换其前向。`torch.inference_mode()`用于推理关闭梯度；本次无需CUDA或FlashAttention。

## 真实观察

输入 `[1,6,32,32]`；融合 `[1,4,384]`；projector `[1,4,64]`；11个文本/动作位置与4个视觉位置融合后logits `[1,15,32064]`。loss=10.3468504（一次更新前）。三组梯度范数：vision2.02937、projector8.21959、language7.73474，均有限且非零。只进行了1次更新，没有训练曲线、收敛或成功率结论。

官方collator输出ids/labels/attention_mask `[2,11]`、pixels `[2,6,32,32]`；短样本尾部ids32000、labels−100、attentionFalse。这个batch仅用于collator核对；模型前向和生成使用batch1。

动作−1/0/1经官方ActionTokenizer得到IDs31999/31872/31744，反解约−0.996078/0/0.996078。256个边界、255个中心的端点处理与说明一致。

真实生成的7个IDs：`[31830,31751,31941,31747,31751,31941,31747]`。使用合成统计后，predict_action输出约 `[0.331765,0.284706,-0.216471,0.490196,0.569412,-0.378824,0.980392]`。这些值没有真实机器人单位或任务标签，不能解释成完成了一个抓取。

demo加入OnlyActionTokens logits processor来保证未训练模型只生成动作词表ID；这是外部测试约束，**原OpenVLA predict_action没有主动添加它**。仅靠解码clip得到范围内数字，不能证明生成的token原本合法。

另外核对：末尾29871缺失时补齐后结果与已补齐prompt相同；两个norm_stats集且key=None被拒绝；官方predict_action结果与独立NumPy反量化/反归一化逐项一致（atol1e-8），其中最后维maskFalse保留归一化量。

[结果JSON](<G:/具身/论文阅读/OpenVLA -  An Open-Source Vision-Language-Action Model/复现/outputs/结果.json>) · [运行日志](<G:/具身/论文阅读/OpenVLA -  An Open-Source Vision-Language-Action Model/复现/outputs/运行日志.txt>) · [依赖检查](<G:/具身/论文阅读/OpenVLA -  An Open-Source Vision-Language-Action Model/复现/outputs/依赖检查.txt>)

## 掌握要求、对应关系与未解决问题

【必须会】融合的feature轴、BOS后插入视觉位置、ignore labels不等于阻断梯度、32064/32000词表区别、动作统计key和mask、生成与反归一化各自职责。

【建议会】两个接口的pixel_values容器、collator右padding、cache首步与后续步骤、从源码hash区分实际加载文件。

【可以暂时不掌握】64卡训练、FSDP/FlashAttention、真实机器人驱动。

|Paper ↔ Code|本轮验证|仍未覆盖|
|---|---|---|
|视觉融合/projector|官方HF类在tiny配置前向|真实DINO/SigLIP预训练权重|
|动作词表|官方ActionTokenizer端点与HF逆映射|真实Llama分词/对话格式|
|模态序列与loss|官方forward、实际三组梯度|实际RLDS样本/原训练recipe|
|动作生成和后处理|实际cache生成、官方predict_action|原7B性能、PIL字符串接口|
|微调/实验|静态源码和论文分析|LoRA运行、真实数据、仿真与真机|

未解决：没有运行原始Prismatic整包的PIL/字符串接口；没有真实processor、RLDS样本、预训练7B或LoRA；没有量化与8GB显存测试；没有真实坐标系和闭环控制。参数变小的兼容测试不支持对原模型运行资源或精度作结论。环境是本demo所需子集，不是完整官方训练环境。
'''
(P/'复现/README.md').write_text(readme,encoding='utf-8')
(P/'复现/status.json').write_text(json.dumps({'status':'部分成功','scope':'官方HF类的缩小随机配置前向/梯度/生成；完整7B与论文实验未复现','date':local_date,'evidence':'outputs/结果.json'},ensure_ascii=False,indent=2),encoding='utf-8')
(P/'复现/requirements.txt').write_text('torch==2.2.0\ntorchvision==0.17.0\ntransformers==4.40.1\ntokenizers==0.19.1\ntimm==0.9.10\nnumpy==1.26.4\nhuggingface-hub==0.23.0\n',encoding='utf-8')
env=P/'复现/环境准备记录.md';t=env.read_text(encoding='utf-8-sig');t+='\n\n最终结果：分段87失败后，重新执行下载器时复用全部完整分段，仅补缺失部分；190个分段合并后，文件大小与PyPI JSON的SHA-256完全一致。安装本地wheel成功。pip check通过，实际import显示torch2.2.0+cpu/torchvision0.17.0+cpu。最终demo执行成功；本机记录时间为'+r['recorded_at_local']+'。详见结果JSON。\n';env.write_text(t,encoding='utf-8')
spec=importlib.util.spec_from_file_location('writer',ROOT/'精读编写.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.write('OpenVLA',s)
progress=ROOT/'逐篇进度.json';items=json.loads(progress.read_text(encoding='utf-8'))
for item in items:
 if item['paper']=='OpenVLA':item['reading']='核心方法、两种接口、监督/缓存与微调日志已详读';item['demo']='官方HF类缩小随机配置：前向、梯度、缓存生成与后处理部分成功'
progress.write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
index=ROOT/'精读索引.md';t=index.read_text(encoding='utf-8');t+='\n\nOpenVLA更新（本机执行时间 '+r['recorded_at_local']+'）：已完成官方HF类的缩小随机配置demo和接口补读。见 [OpenVLA运行记录](<G:/具身/论文阅读/OpenVLA -  An Open-Source Vision-Language-Action Model/复现/README.md>)。原7B和论文实验未复现，下一篇为MDT。\n';index.write_text(t,encoding='utf-8')
print('OpenVLA notes and verified reproduction records updated.')
