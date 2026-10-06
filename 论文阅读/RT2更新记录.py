from pathlib import Path
import json,importlib.util
ROOT=Path(r'G:\具身\论文阅读');P=ROOT/'RT-2 -  Vision-Language-Action Models Transfer Web Knowledge to Robotic Control'
body=ROOT/'精读正文/RT-2.md';s=body.read_text(encoding='utf-8')
s=s.replace('λ 表示采样或加权比例的概念，不在此猜论文所有模型共用一个数值；','λ 表示采样或加权比例的概念；附录 B/PDF19–20 页明确写 RT-2-PaLI-X 的机器人数据约占 50%，RT-2-PaLM-E 约占 66%，不能说两者统一使用 50%；')
s=s.replace('训练采样比例、具体动作区间和部署协议未得到源码验证；','混合比例已从附录核对，但具体采样实现、动作区间和部署协议未得到源码验证；')
extra='''

## 9. 附录补读：骨干、数据和实际训练量

附录 D/PDF20–21 页：PaLI-X-55B 的视觉侧为 ViT-22B，图像 patch tokens 经 projection 后，与文字一起送入约 32B、50 层的 encoder-decoder backbone；不能把这条路线直接写成 Llama 式 decoder-only。PaLM-E 路线采用 decoder-only LLM，将连续视觉/传感器信息投射到语言 embedding 空间；本文 12B 设置的视觉侧为 ViT-4B。两者都能自回归输出，但输入组织与 attention 路径并不相同。

附录 E/PDF21 页列出的 co-fine-tuning 设置：

|模型|学习率|batch size|梯度更新数|
|---|---:|---:|---:|
|RT-2-PaLI-X-55B|0.001|2048|80K|
|RT-2-PaLI-X-5B|0.001|2048|270K|
|RT-2-PaLM-E-12B|0.0004|512|1M|
|Language-Table 的 PaLI-3B 设置|0.001|128|300K|

这些是原文配置，不是本机已跑出的设置。更大的模型不一定有更多更新；比较参数规模时，也必须交代初始化、骨干与训练预算。混合比例是数据采样语境的比例，不自动等于每维 loss 的精确贡献。

附录 B 描述机器人示范沿用 RT-1 的移动操作机器人数据；正文说明采集使用 13 台机器人、持续 17 个月。WebLI 原始约 10B 图文对经过相似度筛选到约 1B，并混合其他视觉语言任务；不能将原始网页数量直接写成所有有效训练样本。Language-Table 设置另有动作、指令、机械臂位置、时间间隔和成功判断五种预测任务，它不是原八字段三维控制接口原封不动的缩小版。

附录 F 说明新对象/背景/环境指令每条测试 1–5 次，语义能力指令各 5 次。约 6k trials 是广泛评测的总量，不能将每个子表都描述为独立 6k 次测试。子场景与总体规模应分别阅读。

## 10. 小 demo：把动作当作共享 token 输出

本轮教学实现一个约 70,416 参数的共享 Transformer，同时完成“某颜色在哪里”的 VQA 与对应的八字段教学动作生成。它没有加载预训练 VLM，严格说是从随机初始化的**共同训练演示**，不验证 RT-2 的预训练知识迁移或 co-fine-tuning 优势。

18 个训练布局、6 个保留布局，每布局有红/绿/蓝三种查询。训练有 54 个图像查询样本；每个样本有动作与问答两个任务。保留集合有 18 个查询，布局与训练不重叠，但同布局内查询并非独立物理 trials。

|数据流|类型与shape|对应操作|
|---|---|---|
|图像|float32 `[B,2,2,3]`|flatten `[B,12]` → Dense `[B,64]`|
|颜色/任务|int32 `[B]`|两个 embedding；并非开放自然语言 tokenizer|
|prefix|float32 `[B,3,64]`|图像、颜色、任务各一个 token|
|之前输出 token|int32 `[B,S]`|BOS + 教师强制的之前标签；推理改为之前预测|
|完整序列|float32 `[B,3+S,64]`|位置 embedding → causal attention → MLP|
|输出 logits|float32 `[B,S,272]`|同一个 head 用于动作和 VQA|
|监督标签/mask|int32/float32 `[B,8]`|动作监督八项；VQA 只监督位置词和 EOS|

教学词表 ID 0–15 留给文字与控制，16–271 表示动作 bins 0–255。位置词的 ID 为 4–7。动作 bin=128 对应词表 ID=144；二者必须区分。连续量按教学等宽中心反解，bin128 对应 0.00390625，而不是严格的 0；这不宣称与 RT-2 原实现端点相同。

训练计算每样本的有效 token 平均 CE，再对动作与 VQA 样本分别平均，最后各乘 0.5。若直接把两任务所有有效 token 加总平均，八字段动作与两 token VQA 会得到不同任务权重，即使样本数量相等。推理逐字段生成，屏蔽任务不允许的 ID；终止字段额外只允许 0/1。

实际记录：混合 CE 5.9055 → 0.0013029；训练集动作八项全匹配与 VQA 位置准确率均为 100%；保留布局两项均为 13/18≈72.2%。保存权重在独立进程重载后复现了这些自由生成指标；不是只统计 teacher-forcing token accuracy。

![教学训练与推理示例](<G:/具身/论文阅读/RT-2 -  Vision-Language-Action Models Transfer Web Knowledge to Robotic Control/复现/outputs/demo结果.png>)

上图红色目标在右上，动作 bins 为 `[0,192,64,128,128,128,128,255]`。平移 x/y 反解约 +0.504/−0.496。其余大多为常量，这个任务只具有极小的控制复杂度；没有机器人坐标变换、碰撞、时间轨迹或闭环执行。72.2% 是合成离散序列匹配率，不能当作真实机器人成功率。

完整命令、函数解释和结果见 [教学 demo 说明](<G:/具身/论文阅读/RT-2 -  Vision-Language-Action Models Transfer Web Knowledge to Robotic Control/复现/README.md>)。官方 RT-2 源码仍未找到；不将教学代码填进上面的官方 Paper ↔ Code 表。
'''
if '## 9. 附录补读' not in s:s+=extra
body.write_text(s,encoding='utf-8')
readme='''# RT-2 教学 demo 与复现记录

**官方论文复现状态：未复现。教学 demo：成功跑通。** 未找到完整官方实现。下面是本地编写的 ToyVLA，用来学习共享 token 输出、共同训练与受限生成；不是 PaLI-X/PaLM-E，也没有互联网预训练阶段。

## 运行方法与环境

沿用隔离环境 `G:\\具身\\论文阅读\\.venvs\\rt1`，Python 3.11.0、TensorFlow 2.15.1、NumPy 1.26.4、CPU。画图使用 matplotlib 3.10.7。`requirements.txt` 是直接依赖，outputs 的环境锁定保存实际完整依赖。

```powershell
# 训练，并保存结果 JSON、曲线数据与 toy 权重
& 'G:\\具身\\论文阅读\\.venvs\\rt1\\Scripts\\python.exe' 'G:\\具身\\论文阅读\\RT-2 -  Vision-Language-Action Models Transfer Web Knowledge to Robotic Control\\复现\\scripts\\rt2_recipe_demo.py'
# 单独加载已保存权重，不重新训练；核对自回归评估和训练/保留布局分离
& 'G:\\具身\\论文阅读\\.venvs\\rt1\\Scripts\\python.exe' 'G:\\具身\\论文阅读\\RT-2 -  Vision-Language-Action Models Transfer Web Knowledge to Robotic Control\\复现\\scripts\\inspect_checkpoint.py'
# 读取已保存数据生成 PNG/SVG
& 'G:\\具身\\论文阅读\\.venvs\\rt1\\Scripts\\python.exe' 'G:\\具身\\论文阅读\\RT-2 -  Vision-Language-Action Models Transfer Web Knowledge to Robotic Control\\复现\\scripts\\plot_demo.py'
```

使用 `&` 调用带空格路径的解释器；三个脚本都按自身路径定位 outputs。可从 `G:\\具身` 运行。训练再次运行会覆盖同名输出，请先备份要保留的旧实验。没有下载新的模型或机器人数据。

## 数据、模型与 loss

四个格子中摆放三个不同颜色，一格空。24 种排列用 seed23 打乱，前18布局训练、后6保留；每布局查询三个颜色。每张输入图是 `[2,2,3]` 的合成 RGB 数组，不是现实照片。位置映射到左右/上下 ±0.5，其他动作基本固定。查询只有颜色和任务两个分类值，没有自然语言分词。

模型的图像投射、颜色 embedding、任务 embedding、Transformer 和输出 head 全部被两种任务共享。它有 70,416 参数，一个 4-head、64-hidden 的 causal attention block，FFN 中间128维。模型从随机初始化共同训练；没有经过 web 预训练后再微调，所以不能拿这个 demo 证明知识迁移。

动作词表 ID=16+bin，bin范围0–255。量化教学式：`floor(128*(clip(a,-1,1)+1))` 后 clip 到255；反解为 `-1+(bin+0.5)*2/256`。终止是离散 bin，单独处理；不能套连续反解式。VQA 使用位置文字 ID4–7 和 EOS2。

train_step 的每任务 loss 是有效 token 平均，再按样本平均；总 loss=`0.5*action_loss+0.5*vqa_loss`。动作八项有效，VQA 两项有效；不让 padding 参与 CE。教学 λ=.5 恰与论文 PaLI-X 约50%的数据混合相近，但没有声称它是相同训练实现；PaLM-E 原文机器人比例约66%。

teacher forcing 在训练时喂入真实之前 token；自由生成评估只能喂入自己之前预测。源码的 causal mask 阻止未来标签信息泄漏；保留评估用自回归 generate，不用真实动作前缀。

## 关键函数与语法

|对象|输入/默认值|返回|状态变化与位置|
|---|---|---|---|
|quantize(x)|NumPy连续数组；范围教学设为[-1,1]|同shape int32 bin|纯变换；构造标签|
|dequantize(b)|bin数组|同shape连续数组|纯变换；解释动作|
|data(layout_subset)|布局序列|图像、颜色、动作标签、VQA标签四数组|合成数据；不读取真实数据|
|ToyVLA.__init__|无额外参数|模型对象|创建可学习层；Dense参数在首次call构建|
|ToyVLA.call(inputs,training=False)|image,color,task,previous tuple|`[B,S,272]` logits|首次调用建参数；调用本身不执行optimizer|
|train_step()|闭包中固定训练张量|mixed/action/VQA三个标量Tensor|GradientTape反传，Adam更新模型与optimizer状态|
|generate(image,color,task,constrained=True)|图像、颜色、任务与是否词表约束|动作`[B,8]`或问答`[B,2]`整数IDs|逐步更新局部prefix，不更新参数|
|evaluate(dataset)|四个数据数组|匹配率/词表合法性字典|自由生成；不训练|
|inspect_checkpoint.predict|输入图、颜色、任务|序列IDs|载入权重后独立生成，无训练|

`class ToyVLA(tf.keras.Model)` 用 Keras 管理层和权重，`super().__init__()` 初始化父类状态。Embedding 将整数 ID 查成64维向量，不是按整数大小缩放；Dense 对末轴做线性变换。`tf.stack` 把三个 `[B,64]` 合成 `[B,3,64]`，`tf.concat` 把prefix与输出序列沿时间轴拼接。`position(...)[None,:,:]` 增加batch轴以广播。

`@tf.function` 让 train_step 追踪为TensorFlow图；GradientTape记录可微运算，apply_gradients才更新权重。`tf.nn.sparse_softmax_cross_entropy_with_logits` 接收整数标签与未softmax的logits。这里用的是平均每个样本有效token的loss；没有把短问答的padding当类别0监督。

generate 中 `np.where(allowed,logits,-np.inf)` 把不合法词表位置的分数设为负无穷，再argmax。教学额外限制了终止与EOS位置；合法数字输出不等于物理动作安全。

## 实际结果与解释

301 次Adam更新、学习率0.003、seed23。曲线记录每次更新前的loss，末点step300是第301次更新前；最终评估使用第301次更新后的权重。

|指标|训练54查询|保留18查询|
|---|---:|---:|
|动作八项全部匹配|54/54=100%|13/18≈72.2%|
|VQA位置正确|54/54=100%|13/18≈72.2%|
|动作词表合法|100%|100%|

混合 CE 从5.9055147降到0.0013029。约束的额外人工probe把文本ID4设置为最高logit：自由argmax选择4，动作词表约束后选择144（对应bin128）。它验证屏蔽逻辑，不是模型任务能力消融。

独立重载 toy.weights.h5 后，训练与保留自由生成指标一致，布局集合交集为空；同一保留图像分别查询红/绿/蓝时，记录中的三种动作位置不同且正确。保存结果、权重和源码hash在 checkpoint核对.json。只有一个seed、一个非常小的有限世界；每布局的三个查询相关，不能当作18次独立真实机器人trials。

[训练结果](<G:/具身/论文阅读/RT-2 -  Vision-Language-Action Models Transfer Web Knowledge to Robotic Control/复现/outputs/结果.json>) · [权重核对](<G:/具身/论文阅读/RT-2 -  Vision-Language-Action Models Transfer Web Knowledge to Robotic Control/复现/outputs/checkpoint核对.json>) · [运行日志](<G:/具身/论文阅读/RT-2 -  Vision-Language-Action Models Transfer Web Knowledge to Robotic Control/复现/outputs/运行日志.txt>)

## 掌握要求和边界

【必须会】bin与词表ID的差别、mask后按有效token归一化、两种任务共享head、训练真实前缀与推理预测前缀、合法token和任务成功的区别。

【建议会】切换数据混合比例、给不同任务统计独立loss、不同布局的保留评估。要判断共同微调优势，还需要相同预训练初始化下的robot-only与mixed对照，多seed，以及语义保持指标。本demo没有这些实验，不从loss下降推断共同微调优于单独微调。

【可以暂时不掌握】真实VLM预训练、大规模分布式训练和云端TPU serving。

|论文机制 ↔ 教学代码|教学定位|官方定位|
|---|---|---|
|动作语言化|quantize、OFFSET、标签格式|RT-2官方实现未找到|
|共享输出与共同训练|ToyVLA.head、train_step|未找到；此处从随机初始化训练|
|受限动作词表|generate的allowed mask|未找到；额外字段约束是教学选择|
|机器人闭环控制|未实现|未找到|

TensorFlow有API弃用warning，运行与重载均退出0。官方网络、预训练权重、真实示范、部署坐标系、闭环控制和论文实验仍未复现；不把这个本地教学实现标成“RT-2官方部分成功”。
'''
(P/'复现/README.md').write_text(readme,encoding='utf-8')
(P/'复现/status.json').write_text(json.dumps({'status':'未复现','scope':'完整官方模型；教学共同训练与受限解码 demo 已跑通','date':'2026-10-03','evidence':'outputs/结果.json'},ensure_ascii=False,indent=2),encoding='utf-8')
(P/'复现/requirements.txt').write_text('tensorflow-cpu==2.15.1\nnumpy==1.26.4\nmatplotlib==3.10.7\n',encoding='utf-8')
spec=importlib.util.spec_from_file_location('writer',ROOT/'精读编写.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.write('RT-2',s)
progress=ROOT/'逐篇进度.json';items=json.loads(progress.read_text(encoding='utf-8'))
for item in items:
 if item['paper']=='RT-2':item['reading']='核心方法与附录设置已核对；教学代码详细解释已写入';item['demo']='教学共享模型共同训练/受限解码成功；非官方复现'
progress.write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
index=ROOT/'精读索引.md';t=index.read_text(encoding='utf-8')
entry='\n\n2026-10-03 更新：RT-2 已补核对附录 B/D/E/F 的模型、混合比例、训练和评测设置，并完成教学共享模型的共同训练、受限解码、保留布局评估与权重重载。见 [RT-2 demo 记录](<G:/具身/论文阅读/RT-2 -  Vision-Language-Action Models Transfer Web Knowledge to Robotic Control/复现/README.md>)。该 demo 不是官方 RT-2 复现，下一篇为 OpenVLA。\n'
if 'RT-2 已补核对附录' not in t:t+=entry
index.write_text(t,encoding='utf-8')
print('RT-2 detailed notes, teaching demo record, and progress updated.')
