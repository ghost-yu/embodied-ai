# RT-2 教学 demo 与复现记录

**官方论文复现状态：未复现。教学 demo：成功跑通。** 未找到完整官方实现。下面是本地编写的 ToyVLA，用来学习共享 token 输出、共同训练与受限生成；不是 PaLI-X/PaLM-E，也没有互联网预训练阶段。

## 运行方法与环境

沿用隔离环境 `G:\具身\论文阅读\.venvs\rt1`，Python 3.11.0、TensorFlow 2.15.1、NumPy 1.26.4、CPU。画图使用 matplotlib 3.10.7。`requirements.txt` 是直接依赖，outputs 的环境锁定保存实际完整依赖。

```powershell
# 训练，并保存结果 JSON、曲线数据与 toy 权重
& 'G:\具身\论文阅读\.venvs\rt1\Scripts\python.exe' 'G:\具身\论文阅读\RT-2 -  Vision-Language-Action Models Transfer Web Knowledge to Robotic Control\复现\scripts\rt2_recipe_demo.py'
# 单独加载已保存权重，不重新训练；核对自回归评估和训练/保留布局分离
& 'G:\具身\论文阅读\.venvs\rt1\Scripts\python.exe' 'G:\具身\论文阅读\RT-2 -  Vision-Language-Action Models Transfer Web Knowledge to Robotic Control\复现\scripts\inspect_checkpoint.py'
# 读取已保存数据生成 PNG/SVG
& 'G:\具身\论文阅读\.venvs\rt1\Scripts\python.exe' 'G:\具身\论文阅读\RT-2 -  Vision-Language-Action Models Transfer Web Knowledge to Robotic Control\复现\scripts\plot_demo.py'
```

使用 `&` 调用带空格路径的解释器；三个脚本都按自身路径定位 outputs。可从 `G:\具身` 运行。训练再次运行会覆盖同名输出，请先备份要保留的旧实验。没有下载新的模型或机器人数据。

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
