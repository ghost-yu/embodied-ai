## 1. 问题、贡献与论文地图

机器人示范采集很贵。为一个对象、一个场景训练的策略容易换环境就失败。RT-1 的核心问题是：把大量任务的示范放在一起，能否得到一个根据语言指令执行动作、并泛化到新组合的策略？它使用模仿学习，不在本文训练过程中通过环境奖励寻找策略。

论文贡献要分为三部分看：能实时运行的视觉语言动作架构；13 台机器人、约 130k episodes、700 多任务的实际示范数据；对任务、干扰物、背景、异构数据和数据多样性的系统评估。不能把实验改进全部归因于 Transformer 一个模块。

阅读地图：§3 定义策略与机器人动作；§4 说明系统；§5.1/PDF 5–6 页和 Figure 3 讲模块；§5.2 讲数据；§6/Table 2 讲主比较；Table 7 讲数据量与多样性；附录 Table 13/PDF 30 页讲架构消融。Figure 2 是采集和评测场景，不是模型内部流程。

【必须会】用自己的话说明：RT-1 学的是“给定观察和指令，示范者下一步会做什么”，不是显式预测整个世界。

## 2. 沿 Figure 3 走一个样本

假设指令是“拿起桌上的罐子”。语言先经 Universal Sentence Encoder 得到句子向量。这不是 LLM 逐 token 推理，而是提供条件向量。六张历史图像各自经过带 FiLM 的 EfficientNet-B3。语言在视觉编码阶段调节通道，让特征更关注当前任务有关的内容。

论文给出的图像分辨率是 300×300；每张图的输出特征是 9×9×512。把空间两轴拉平得 81×512，再由 TokenLearner 压成 8×512。六帧形成 48 个视觉语言 token，加入位置信息后送入 decoder-only Transformer，得到动作类别 logits，最后反离散化为机器人控制量。

|变量|类型与形状|来源和去向|
|---|---|---|
|历史图像|图像数值，示意 `[B,6,300,300,3]`|相机历史→视觉编码器；代码预处理后的 dtype 需以实际样本核验|
|USE 条件|浮点 `[B,D_l]`|指令→FiLM；D_l 此处不猜具体模型配置|
|CNN 特征|浮点 `[B,6,9,9,512]`|论文架构→空间 flatten|
|压缩特征|浮点 `[B,6,8,512]`|TokenLearner→时序组装|
|视觉序列|浮点 `[B,48,512]`|说明论文图中的视觉内容；不是代码完整输入长度|
|动作标签|int32 `[B,T,D_a]`|动作 tokenizer→分类监督|
|动作 logits|浮点 `[B,T,D_a,256]`|模型输出→交叉熵或 argmax|

**论文图和代码组装要分开。** `transformer_network.py` 的 `_assemble_input_token_sequence` 还处理动作占位/token，训练序列长度按 `T × (视觉token数 + 动作token数)` 组织。不能把“48 个视觉 token”写成整个工程永远只有 48 个输入位置。Keras `call` 的训练外层 rank 是 `[B,T]`，推理外层 rank 是 `[B]`，历史缓存在 network state 中更新。

FiLM、TokenLearner、Transformer 有可学习参数；clip、reshape、argmax、查动作上下界没有学习参数。训练用真实动作标签计算 loss，推理通过模型预测并维护历史。机器人终止动作与环境 API 的 `done/terminated/truncated` 不是天然同一个变量，需要部署层转换。

## 3. 三个关键机制与公式

### 3.1 FiLM 为什么不破坏预训练特征【必须会】

下面是对论文图的教学重写，不是 RT-1 编号公式：

$$\operatorname{FiLM}(F,l)=(1+\gamma(l))\odot F+\beta(l).$$

F 是视觉激活；l 是语言条件；γ、β 是根据 l 产生的通道调节；⊙ 是逐元素乘。若 F 为 `[B,h,w,c]`，通道参数需广播为 `[B,1,1,c]`。γ=β=0 时输出等于 F，所以把调节层初始输出设为零可保留已有视觉网络功能。

标量例：F=2，γ=0.1，β=0.2，输出 2.4；初始 γ=β=0，输出 2。代码入口是 `film_efficientnet/film_conditioning_layer.py`，它只完成语言条件调节，不是整个图像编码器。此处保留论文“identity initialization”的含义，具体层的输入通道随 EfficientNet 层变化。

### 3.2 TokenLearner 是加权汇总，不是裁剪【必须会】

教学重写：

$$w_{s,i}=\operatorname{softmax}_{i}(g(X)_{s,i}),\qquad Z_{s,d}=\sum_{i=1}^{81}w_{s,i}X_{i,d}.$$

X 是 `[B,81,512]`；s∈{1,…,8} 是输出 token；i 是输入空间位置；d 是特征维。g 是 LayerNorm 和 MLP 产生的选择 logits。w 为 `[B,8,81]`，Z 为 `[B,8,512]`。softmax 沿 81 个位置计算，每个输出 token 的位置权重和为 1。

实际代码 `TokenLearnerModule.call(inputs, training=False)` 的关键语句：

```python
selected = tf.transpose(selected, [0, 2, 1])
selected = tf.nn.softmax(selected, axis=-1)
feat = tf.einsum("...si,...id->...sd", selected, inputs)
```

第一行把 `[B,81,8]` 变成 `[B,8,81]`；第二行把分数变权重；第三行按 i 汇总原始特征，而不是把 MLP 输出直接当视觉特征。方法返回压缩特征，不修改输入张量；有 LayerNorm/MLP 参数和训练时 dropout 行为。`einsum` 的省略号表示批次等前置轴，i 是被求和的轴。

小例子：两个位置的特征分别是 2 和 6，权重 0.25、0.75，输出是 5。这说明一个新 token 可以汇总多个位置，并不对应一个固定图像 patch。

### 3.3 动作离散化和 loss【必须会】

`RT1ActionTokenizer.tokenize(action)` 接受按名字组织的动作结构，`action_order` 决定 token 次序。对于浮点动作，实际实现等价于：

$$b=\left\lfloor255\frac{\operatorname{clip}(a,a_{min},a_{max})-a_{min}}{a_{max}-a_{min}}\right\rfloor,\quad
\hat a=a_{min}+\frac b{255}(a_{max}-a_{min}).$$

a 是某动作维，b 为 0–255 的 int32；上下界来自 action spec。连续值先 clip，再归一化，再乘 255，最后强制转整数。`detokenize` 按相同次序拆开并返回动作结构。终止/模式这类 int32 输入在该实现中以 one-hot 编码进来，经 argmax 得到类别；不能对它套连续缩放公式。

例：a=0.2、区间 [-1,1]，b=floor(153)=153，反解为 0.2。a=0 时会得到 127、反解约 -0.00392，说明离散化不是无损操作。论文完整动作是 7 个机械臂量、3 个底盘量、1 个模式量；库文件注释也给了更小 action spec 的例子，不能据此说论文只有 8 维。

交叉熵教学式：

$$L=-\frac1{|\Omega|}\sum_{(b,t,j)\in\Omega}\log p_\theta(y_{btj}\mid\text{可见观测和指令}).$$

Ω 是有效监督位置，y 是真实类别。若真实类别概率为 0.8，单项 loss 是 −ln(0.8)≈0.223。`TransformerNetwork.call` 使用 `SparseCategoricalCrossentropy(from_logits=True)`；“Sparse”表示标签为类别整数，不要求外部构造 256 维 one-hot。精确 mask 和归一化由训练实现决定，不能把教学平均式当成逐行等同的代码。

## 4. 实际调用链与容易误读的地方

该仓库是 TensorFlow/Keras 工程，不是 PyTorch；没有理由把它强行改写成 `nn.Module` 并称为原实现。可确认的库级链为：`SequenceAgent` 使用网络→`TransformerNetwork.call`→`_tokenize_images`/`_tokenize_actions`→`_assemble_input_token_sequence`→Transformer→动作位置 logits→loss 或 detokenize。README 提供组件测试与 SavedModel 说明，尚未找到可直接复制论文完整示范采集、训练数据管线和真机评估流程的独立脚本。

`call(observations, network_state, training=False)` 的 observations 是字典，network_state 带历史缓存，返回预测动作和更新后的 network state。training 影响层行为；函数内部另根据输入外层 rank 区分时序训练和单步推理，所以不能仅看到 `training=False` 就认定进入唯一推理分支。内部 `_loss` 等记录也会更新。

**RT-1 最终版本不逐维自回归生成动作。** Table 13 把 `w/ auto-regressive actions` 列为消融；正文说明最终不采用它。decoder-only、因果 mask 和动作逐维自回归不是同义词。`_generate_masks` 进一步处理动作之间的可见性。单步推理的循环/位置填充也不能脱离 mask 解读成 RT-2 的同一机制。

README 的安装/测试示例须先核对：其 `requirements.txt` 在此快照未找到；一个动作测试命令的模块名也应与实际文件名核验。不要把文档中能复制的命令当成已运行记录。

## 5. 主实验与消融怎样支持结论

Table 2/PDF 10 页是实机任务成功率，越高越好，不是动作 token 准确率：

|方法|Seen|Unseen|Distractors|Backgrounds|
|---|---:|---:|---:|---:|
|Gato|65|52|43|35|
|BC-Z|72|19|47|41|
|RT-1|97|76|83|59|

RT-1 相比该列最佳基线分别高 25、24、36、18 个百分点。不同列代表不同分布变化，不能把 97% 用来描述新背景上的表现；新背景只有 59%，说明泛化还不充分。比较在论文控制的训练/评测框架内成立，不代表任意机器人都能达到相同结果。

Table 7/PDF 14 页：保留全部任务、把数据减到 51% 时，seen 从 97 到 71、unseen 从 76 到 52；保留 97% 数据但只有 75% 任务时，seen 从 97 到 86、干扰物从 83 到 42。后一项损失 41 个百分点，支持任务多样性很重要；两种裁剪不是完美一一匹配的因果实验，不能推出“任意少量多样数据都胜过大量数据”。

Table 13/PDF 30 页：continuous actions 的 seen=68，较完整模型低 29 个百分点；无历史的 distractors=50，低 33 个百分点；自回归动作网络耗时 36 ms，完整模型 15 ms。这里是网络推理耗时，不是整机控制周期；论文系统为约 3 Hz。连续动作这个消融也不证明后来的扩散/flow 连续动作策略不行，因为其架构、训练目标与数据不同。

## 6. 掌握层次与未解决问题

【必须会】FiLM 语言条件、81→8 token 的加权压缩、动作分类与反离散化、训练/推理分支、成功率与 token 准确率的区别。

【建议会】追踪 action_order、注意力 mask 和 loss mask；核对无自回归动作的工程实现；分析任务多样性实验与异构数据实验。

【可以暂时不掌握】完整 EfficientNet 每个 MBConv、Bazel、真机驱动与多机器人采集调度。

未解决：当前 action spec 的完整坐标与单位需按部署模型核对；完整官方训练数据与采集系统未找到；没有运行 SavedModel；已确认 rt1main 的变量数据文件仍是 Git LFS 指针，实际权重未下载。论文的长时任务依赖 SayCan 上层规划，不能写成 RT-1 自己输出全部长时计划。

|Paper ↔ Code|可确认定位|边界|
|---|---|---|
|§5.1 视觉语言编码|image_tokenizer、FiLM encoder|输入实际 dtype/shape 未做运行记录|
|TokenLearner|TokenLearnerModule.call|加权汇总已静态核对|
|动作离散化|RT1ActionTokenizer.tokenize/detokenize|由实际 spec 决定动作维|
|分类训练与 mask|TransformerNetwork.call/_generate_masks|完整训练驱动未找到|
|实机评估、示范采集|未找到|仅在原文中解释|


## 7. 本篇小 demo：论文机制怎样落到运行结果

已在独立 TensorFlow CPU 环境直接运行官方 FiLM/TokenLearner 模块，完成最小前向、合成梯度训练和官方 TokenLearner 测试。完整说明、命令与源码见 [小 demo](<G:/具身/论文阅读/RT-1 -  Robotics Transformer for Real-World Control at Scale/复现/README.md>)。

六帧特征 `[6,9,9,512]` → FiLM → 每帧 TokenLearner `[6,8,512]` → 组合为 `[1,48,512]`。FiLM 恒等初始化误差 0；TokenLearner 权重和误差约 1.19×10⁻⁷。合成 marker 选择任务在 120 次更新后 MSE 从 3.888609 降至 2.0539×10⁻⁸。官方单项 TokenLearner 测试通过。

![合成训练 MSE 曲线](<G:/具身/论文阅读/RT-1 -  Robotics Transformer for Real-World Control at Scale/复现/outputs/训练曲线.png>)

这是同四样本上的训练误差，未评估泛化；图上纵轴为 log scale。condition 为合成 12 维向量，输入是合成特征，不是相机+真实指令。没有 CNN、USE、完整 Transformer、动作输出或真实机器人。这些成功仅支持模块行为，不支持论文任务成功率。
