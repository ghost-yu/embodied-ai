## 1. 问题、贡献与地图

双臂动作具有更强的多模态性：同一物体可能由左手拿、右手拿或两手协作。直接回归平均动作可能两种方式都做不成。不同机器人又有不同关节数、末端控制量与本体状态，简单混合数据会使维度语义冲突。

RDT 用连续扩散表达动作分布，设计适合机器人信号的 Transformer，并建立有物理语义的统一动作空间。它结合多机器人预训练和双臂 fine-tuning；不是一个以 Llama 直接输出动作词表的 VLA。语言 T5-XXL 与视觉 SigLIP 在这里作为冻结条件编码器。

地图：§3/PDF 3–4 页任务和挑战；§4.1/Eqs.(1–2)/Figure 3（PDF 4–7 页）为扩散与架构；§4.2 为数据/统一空间；附录 C/Table 4/PDF 20 页是动作槽位；Tables 2–3/PDF 9–10 页给消融/真机结果；附录 H/PDF 25–27 页给训练配置和基线设置。

## 2. Figure 3 的一条完整路径【必须会】

观察 $o_t=(X_{t-T_{img}+1:t},z_t,c)$：X 是多相机历史，z_t 是本体状态，c 是控制频率；ℓ 为语言。图像→SigLIP→视觉 tokens，语言→T5→文本 tokens。状态与带噪动作块经过 adaptor，加入 diffusion step 和 frequency 条件，进入 RDT。cross-attention 分层交替注入视觉和语言；输出 MLP 将隐变量投影回统一动作空间。

保存配置：T_img=2，camera=3，H=T_a=64，统一维 D=128；图像特征维1152，语言特征维4096，RDT hidden=2048、depth=28、heads=32。每头 hidden/head=64。图像 token 数还取决于 encoder patch 数，不能直接用 camera×history 当最终 token 数。

|变量|类型与形状|去向|
|---|---|---|
|lang_tokens|浮点 `[B,L,4096]`|语言 adaptor→`[B,L,2048]`|
|lang_attn_mask|bool `[B,L]`|屏蔽 padding 的语言位置|
|img_tokens|浮点 `[B,N_img,1152]`|图像 adaptor→2048|
|state_tokens|浮点 `[B,1,128]`|统一状态→与动作拼接|
|action_gt/noise/noisy_action|浮点 `[B,64,128]`|扩散与监督|
|action_mask|0/1 float `[B,1,128]`|描述有效物理槽位|
|ctrl_freqs|浮点 `[B]`|每个样本的 Hz 条件，不是 timestep|
|timesteps|long `[B]`|扩散时间类别|

时间轴拼 state 和动作后为 `[B,65,128]`；再在特征轴追加 mask 为 `[B,65,256]`；state adaptor 映射到 hidden=2048。注意力 mask 与动作有效维 mask 都叫 mask，但用途不同。

## 3. 128 维统一空间不是128个自由度

Table 4 按物理意义给槽位：右臂 joint positions [0,10)，右夹爪 [10,15)，右臂速度 [15,25)，右夹爪速度 [25,30)，右末端位置[30,33)，6D rotation 表示[33,39)；左臂对应从50起；底盘线速度[100,102)、角速度102；其余部分保留。6D pose 字段需按具体表示理解，不能把六个数字都叫欧拉角。

单臂映射到右臂槽位；六关节只填十个 joint position 槽的前六个。其余 padding=0，同时 mask=0。单位、相对/绝对控制、标准化和源数据坐标变换仍需 data loader 处理；“物理可解释”不自动消除这些差别。

例：六关节动作 `[0.1,…,0.6]` 写到 [0,6)，对应 mask=1；[6,10) 为 padding。空槽零与真实零动作要靠 mask 区分。本体状态与 action 可以占相同统一空间，但一个表示当前测量、一个表示目标控制，不能交换。

## 4. 扩散公式：预测干净动作【必须会】

Eq.(2) 的核心：

$$\tilde a_t=\sqrt{\bar\alpha_k}a_t+\sqrt{1-\bar\alpha_k}\epsilon,\qquad
L(\theta)=\operatorname{MSE}(a_t,f_\theta(\ell,o_t,\tilde a_t,k)).$$

k∈{1,…,K} 为扩散时间；α_k=1−β_k，$\bar\alpha_k=\prod_{i=1}^k\alpha_i$；ϵ 标准高斯；t 是机器人时间。动作块时 a_t 扩展成 `[B,H,D]`，噪声相同 shape，α 每样本选取后广播到时间和动作维。

标量例：a=0.2、ᾱ=0.64、ϵ=1，带噪输入=0.8×0.2+0.6×1=0.76。模型应该预测干净 a=0.2；若预测0.3，平方误差0.01。不是要预测噪声1。论文 Eq.(1) 的反向 DDPM 更新用干净预测替换未知 a⁰，再加 schedule 对应的高斯项；它只说明理论采样机制，实际 inference 使用 DPM-Solver++ 五步，不逐行手写原式1000次。

保存 `configs/base.yaml`：训练1000个 diffusion timesteps、squaredcos_cap_v2 schedule、`prediction_type: sample`、inference五步。1000是训练噪声等级数，五步是推理 solver 的时间网格，64是输出动作数；三个数不能混用。

## 5. 原实现 compute_loss 和 sampling

`RDTRunner.compute_loss(lang_tokens,lang_attn_mask,img_tokens,state_tokens,action_gt,action_mask,ctrl_freqs)` 返回标量 loss。关键原代码：

```python
noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)
state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
```

`randn` 采相同 dtype/device 的噪声；`randint` 为每样本选扩散时间；cat(dim=1) 加时间位置，cat(dim=2) 加有效维信息；expand 中 -1 表示保留原长度，扩展无需当作新的独立参数。`adapt_conditions` 对齐 token宽度，再进 RDT model。

接着 `prediction_type=='sample'` 用 action_gt 当 target，`'epsilon'` 用 noise。保存实现的最终 loss 是 `F.mse_loss(pred,target)`。

**重要边界：这个 compute_loss 没有在最终 MSE 上显式乘 action_mask。** mask 在输入上追加，sampling 最终又乘 mask，但不能因此写成“loss仅对有效维计算”。padding、noise与数据构造怎样共同影响无效维，需要继续追数据 loader。这是论文直觉与可执行细节必须分开的位置。

推理 `predict_action`→state和mask拼接→adapt_conditions→`conditional_sample`：先生成 `[B,64,128]` 随机动作，scheduler.set_timesteps，循环 model→scheduler.step→prev_sample，最后乘action_mask。模型参数不在采样中更新；scheduler 的 timesteps 被设置，是对象状态变化。返回统一空间动作，部署脚本再拆到机器人控制字段。

QKNorm 稳定注意力打分；RMSNorm 不做均值中心化；MLP decoder提供非线性投影；ACI交替注入条件以缓解图像token数量压过语言。它们是作者面向机器人信号的设计与实证解释，不是能对任意物理系统保证稳定的定理。

## 6. 训练和评估路径

`main.py`→config/参数→`train/train.py`→训练dataset→冻结多模态编码器→RDTRunner→compute_loss→optimizer/checkpoint。自定义 fine-tuning 数据从 `train/dataset.py` 进入 `data/hdf5_vla_dataset.py` 的 HDF5VLADataset 路径；跨机器人预训练数据另有转换/缓冲管线，不能只读一个 dataset class 就认定覆盖全部46个数据集。

配置 buffer 注释要求至少400GB，`/path/to/buffer` 是占位符；不是必须为最小教学前向创建400GB数据。论文预训练数据21TB、48 H100 80GB 的训练规模不是本机计划。附录说明 fine-tuning 从500K pretrain checkpoint开始，不是从最终1M步checkpoint开始，这影响复现版本选择。

## 7. 定量实验怎么读

Table 2/PDF 9 页选三个条件：unseen cup2整体成功、unseen room3整体成功、Pour Water-L-1/3的水量子目标。完整 RDT 为50、62.5、100；scratch为0、25、62.5，差50、37.5、37.5个百分点；small为37.5、62.5、25。小模型在选定新场景列与大模型相同，不能说扩大模型每列都提升。

这三列前两项是完整任务、第三项是子指标，不能直接无说明平均。每杯/房间/倒水指令只测8次，所以12.5个百分点对应一次试验，样本量有限。

Table 3/PDF 10 页：Fold Shorts仅一个技能示范的设置下，RDT=68%、scratch=40%，高28个百分点，测试25次；Handover完整任务RDT=40%、scratch=16%，高24个百分点。**one-shot不是模型只看过一条总示范**，它还使用大规模预训练及其他 fine-tuning 数据。Robot Dog的Total=76而Walk Straight=48，抓住遥控器/推摇杆与走直线是不同指标。

ACT不使用语言，因此不适合比较指令跟随；OpenVLA需改成双臂输出和对应训练设置，其原模型单臂接口与双臂实验不同。Table 7把大模型、多机器人数据、建模方法列出，有助于说明比较因素，不能凭主表推出离散策略普遍无效。

## 8. Figure 3 与 RDTRunner 的实际执行顺序【必须会】

先看原图：右侧历史图像经冻结 SigLIP 形成视觉条件，底部文本经冻结 T5-XXL 形成语言条件；中部状态、带噪动作、控制频率和扩散时间进入可训练动作网络。左侧统一动作空间是数据接口，不是另一套 LLM。输出为连续动作块。图中冻结符号约束编码器的参数更新，MLP、RDT blocks、adaptor 和可学习位置参数仍可训练。训练时带噪动作来自真实动作加噪；推理时来自高斯初始化。图像历史不包含预测未来，不能把 Figure 3 的切片终点 t+1 理解为读取未来一帧，Python 风格切片右端不含 t+1。

源码对象分两层：`RDTRunner` 管理条件 adaptor、训练 scheduler、采样 scheduler；`RDT` 是真正的动作 Transformer。二者继承 `nn.Module`，`runner(...)` 自动进入 `forward`，该方法把 `*args/**kwargs` 原样转交 `compute_loss`，所以训练调用 runner 并不会默认生成动作。

以本次小配置为例，B=16，H=64，统一动作维 D_a=128，hidden d=64。语言长度 L=4，图像特征 token 数 N_img=6。此处六个图像 token 是合成特征，**不是**三相机×两帧图像完整 SigLIP 的 token 数；真实编码器还会产生每张图的 patch tokens。

|顺序|方法及变量流|形状、返回与状态|
|---|---|---|
|1|`RDTRunner.__init__` 建立 `RDT`、三个 adaptor、DDPM/DPM-Solver scheduler|构造默认 dtype 为 bfloat16；demo 明确 float32，CPU|
|2|`compute_loss` 采 noise 与 timesteps|noise `[16,64,128]`；timesteps long `[16]`，取 0～999；由同 device/dtype 构建张量|
|3|`noise_scheduler.add_noise(action_gt,noise,timesteps)`|加噪结果 `[16,64,128]`；不是修改干净标签 action_gt|
|4|沿 dim=1 拼 state 与 noisy_action|`[16,1,128]` + `[16,64,128]` → `[16,65,128]`|
|5|mask `[16,1,128]` expand 到65个位置，再沿 dim=2 拼接|状态动作输入 `[16,65,256]`；前128数值，后128有效性指示|
|6|`adapt_conditions`|lang `[16,4,12]`→`[16,4,64]`；img `[16,6,8]`→`[16,6,64]`；state/action `[16,65,256]`→`[16,65,64]`|
|7|`RDT.forward(x,freq,t,lang_c,img_c,lang_mask=None,img_mask=None)`|freq与t分别嵌入为`[16,1,64]`；在序列前面拼接两项，得到`[16,67,64]`|
|8|位置嵌入 + 四个 RDTBlock|block0/2读取lang，block1/3读取img；当前序列长度始终67|
|9|`FinalLayer` + `x[:,-self.horizon:]`|先输出`[16,67,128]`，最后只保留64个动作，返回`[16,64,128]`|
|10|`F.mse_loss(pred,target)`|prediction_type=sample选干净action_gt；返回标量，不执行optimizer|

注意代码注释有两处不能照抄：RDT 的前向在 state+action 前拼了 **timestep 与 freq 两项**，实际长度 H+3，不是注释里的 T+1；FinalLayer 的注释说“inject language”，但调用 `self.final_layer(x)` 只传 x，没有额外 language 参数。语言已由前面的交替 cross-attention 注入，最终层是 RMSNorm 和 MLP。

`build_condition_adapter(projector_type,in_features,out_features)` 是 Runner 的实例方法，返回 `nn.Linear` 或 `nn.Sequential`，未知名称抛 ValueError。`mlp2x_gelu` 由 Linear(in,d)→GELU→Linear(d,d) 构成；`mlp3x_gelu` 多一个 GELU和Linear。源码用正则提取字符串里的层数，不是每种模态都有单独硬编码的 class。

`torch.cat` 合并现有轴，`torch.stack` 新增轴；`.expand` 用广播视图重复 mask，不建立不同时间位置的独立有效性。训练中的 mask 不参与最终平方误差加权；把输入中的 mask 与语言 attention mask混为一谈，会得出错误的 loss 解释。

## 9. ACI、QKNorm 和 RMSNorm 从维度看【建议会】

ACI 在本快照中由 `for i,block in enumerate(self.blocks)` 和 `conds[i%2]` 实现：偶数层读语言、奇数层读图像。是跨层交替，不是把两模态拼成一个巨大条件序列，也不是每一层先读语言再读图像。本次用真实 module hooks 捕获到条件长度依次4、6、4、6，与源码一致。hooks 只是记录输入，不返回替代输出。

每个 `RDTBlock.forward(x,c,mask=None)` 依次做 RMSNorm→self-attention→residual，RMSNorm→cross-attention→residual，RMSNorm→MLP→residual，返回与x同 shape。`origin_x=x` 保存残差引用；后续重新赋值x再加origin_x，不是覆盖输入的原始存储。可训练权重在backward后由optimizer更新。

`CrossAttention.forward(x,c,mask=None)` 是方法，x为query来源，c为条件key/value来源。四个heads、每头16维：q=`[16,4,67,16]`，语言k/v=`[16,4,4,16]`，打分=`[16,4,67,4]`；视觉层最后一维改成6。代码将mask `[B,L]` reshape为`[B,1,1,L]`，再expand为`[B,1,67,L]`，沿heads广播。**True表示可读条件**；False padding在非fused路径被置为负无穷，softmax后权重为0。fused路径使用PyTorch scaled_dot_product_attention的布尔mask。本demo在训练后把padding token改成999，预测最大差0，避免随机零输出头让任何输入都看似“不影响输出”的误判。全False语言mask的运行边界尚未测试。

教学公式：

$$\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\left(\frac{\operatorname{RMSNorm}(Q)\operatorname{RMSNorm}(K)^T}{\sqrt{d_h}}+M\right)V.$$

这里d_h是每头维，M在不可读条件上为−∞；真正的函数会用fused API或显式matmul，不会构造这个教学符号M。QKNorm分别对q和k的最后一维使用RmsNorm；v没有在这个位置做同样归一化。qkv_bias=True，与是否使用norm的缩放参数无关。

RMSNorm 用输入的均方根控制尺度，教学写法：

$$\operatorname{RMSNorm}(v)_j=\gamma_j\frac{v_j}{\sqrt{d^{-1}\sum_{r=1}^{d}v_r^2+\epsilon}}.$$

v是一个token的浮点向量；d为该归一化轴长度；γ是可学习缩放，ε避免除零。例v=[1,3]，γ=[1,1]，忽略微小ε，RMS=√5，结果约[0.4472,1.3416]。LayerNorm会先减均值2，结果近似[-1,1]，两者保留均值信息的方式不同。此保存环境使用timm1.0.3的RmsNorm，内部调用fast_rms_norm；不能声称本CPU demo运行了GPU FlashAttention。

位置参数也需要区分：`x_pos_embed [1,H+3,d]`、`lang_cond_pos_embed [1,L_max,d]`、`img_cond_pos_embed [1,N_img,d]` 用sin-cos值初始化，但均为 `nn.Parameter`，可学习，不是永远冻结的常量。语言位置按实际长度切片，视觉条件长度必须与构造时一致。demo用默认一维条件位置初始化，没有运行实际多维相机/历史/patch布局的配置。

## 10. 训练 MSE、采样误差与数据归一化的三个边界【必须会】

**训练调用链已补核对**：`train/train.py`取最后一个state，冻结视觉/语言encoder输出，`state_elem_mask.unsqueeze(1)`后直接调用rdt；随后`accelerator.backward(loss)`、clip_grad、optimizer.step。这个段落没有给compute_loss返回值补上有效维mask。因此，当前主训练链最终仍是全维MSE。

**采样日志是另一种 reduction**：`train/sample.py`在`predict_action`后计算elementwise MSE，再乘expanded_state_elem_mask、除以有效元素数；它另外计算 `sqrt(square_error)/(state_norm+1e-3)` 的有效维平均。state_norm在这里是尺度统计，不是对所有训练动作做的通用标准化操作。训练loss、masked sample MSE、sample_l2err 和真机success rate含义不同，数值不能直接混比。

可复查的数值例：demo14个有效槽位，128总槽位，初始输出为0。全维MSE=0.014289826，而有效维平均=0.130649835，两者比值14/128。将一个**无效**槽的干净target从0改为1，官方训练loss增加1/128=0.0078125。noise、step用相同随机seed；官方最终输出头初始为零，所以此处输出不受更改输入的影响。这只是验证损失边界，训练数据仍保持无效target为0。加噪阶段没有先把无效维noise归零，网络会看到这些槽的随机噪声与mask，需要学习输出padding目标。

**数据接口已纠正定位**：`train/dataset.py` 是训练包装与collator；自定义HDF5示例类实际在 `data/hdf5_vla_dataset.py`。示例读取14维 `[left6,left_gripper,right6,right_gripper]`，通过STATE_VEC_IDX_MAPPING写到统一128维；不足64步时用最后一个动作补齐，历史不足两帧时用第一帧补齐。不是用零动作假装之后全部停止。

示例对qpos的两个gripper分别除以4.7908/4.7888，对目标action的gripper分别除以11.8997/13.9231；这属于保存示例机器人/采集链的具体缩放，不能作为所有机器人的通用常数。它计算episode的state_mean/std/RMS(state_norm)并填入128维。训练包装可依据state_std注入state noise、按dataset mean替换被mask的状态；本文没有据此推断全部来源数据的单位与坐标。预训练46个数据集的转换需逐个检查，本次只建立代表路径。

## 11. Eq.(1) 与实际五步采样【必须会】

原文理论DDPM反向更新可以写为：

$$a_t^{k-1}=\frac{\sqrt{\bar\alpha_{k-1}}\beta_k}{1-\bar\alpha_k}\hat a_t^0+
\frac{\sqrt{\alpha_k}(1-\bar\alpha_{k-1})}{1-\bar\alpha_k}a_t^k+\sigma_k z.$$

a_t^k是第k个扩散噪声等级的动作，t是机器人时间；预测干净动作\(\hat a_t^0=f_\theta(\ell,o_t,a_t^k,k)\)，两个系数与sigma都由schedule决定。例取α_k=.8、ᾱ_{k−1}=.5、β_k=.2、ᾱ_k=.4，两个系数约.23570和.74536；若干净预测=.2、当前带噪=.76、z=0，则一步均值约.61361。这里教学上令z=0来展示均值，不代表所有DDPM步骤都没有随机项。k=1时末步无新的随机项。

实际 `RDTRunner.conditional_sample(lang_cond,lang_attn_mask,img_cond,state_traj,action_mask,ctrl_freqs)` 没有手写该DDPM式；它调用diffusers的DPMSolverMultistepScheduler。初始noisy_action为高斯`[B,64,128]`，每一轮追加mask、state_adaptor、与当前state context拼接，RDT预测干净动作，scheduler.step返回prev_sample。最后统一乘mask，确保无效输出维为0。这个方法修改scheduler的timesteps与内部多步历史，不修改模型参数；每次set_timesteps会重新开始一轮采样。

当前diffusers0.27.2默认solver_order=2、algorithm_type=dpmsolver++、solver_type=midpoint、lower_order_final=True；本次抓取的五个调用时间为999、799、599、400、200，末尾sigma为零由scheduler处理。不是必须调用模型timestep=0才得到最终动作。训练1000噪声等级与采样五个调用数仍是不同概念。

`predict_action`、`conditional_sample` 在Runner本身没有`@torch.no_grad`装饰，demo显式使用`with torch.no_grad()`。仅调用eval()不会关闭autograd。频率25是合成控制频率条件，不是本机每秒真的执行25次机器人指令；本demo没有环境reset/step、reward/done或真机发布动作。

## 12. 零初始化输出头为何先阻断上游梯度【建议会】

`RDT.initialize_weights` 对Linear使用Xavier初始化，对time/frequency MLP特定权重使用normal(std=.02)，对FinalLayer最后的 `fc2.weight/bias` 清零。所以初始预测动作是0。若y=W h+b，初始W=0，则对h的梯度为Wᵀ∂L/∂y=0；对W的梯度为(∂L/∂y)hᵀ仍可能非零。第一次backward中语言adaptor梯度为0，不能据此认定它断开计算图。输出头经optimizer更新后，上游才能收到梯度。

本demo实际记录到首轮输出头梯度norm约0.07386，语言adaptor为0；训练后再backward，语言/图像/state adaptor、first block和输出模块梯度全部有限且非零。AdamW.step改变参数；zero_grad(set_to_none=True)清空累计梯度，backward本身不会更新权重。模型state_dict保存参数和buffer，不能代替配置、tokenizer或encoder checkpoint。


## 13. 小 demo：官方 Runner 保留128维与64动作块

本次直接导入固定commit的 `RDTRunner`、`RDT` 和 `configs/state_vec.py`。模型不是重新写的教学Transformer；scheduler也是官方Runner实际调用的diffusers。hidden=64、depth=4、heads=4，合计 **260,608** 参数，随机初始化；动作维128、horizon64、1000训练噪声等级、cosine beta schedule、sample预测、五步DPM-Solver保持保存配置的这些接口。视觉/语言输入替换为合成特征，不实例化SigLIP/T5。float32 CPU、四线程、seed53。

右六关节与左六关节各占自己的物理槽位，加两个gripper共14维。有效索引为 `[0, 1, 2, 3, 4, 5, 50, 51, 52, 53, 54, 55, 10, 60]`。合成标量z有16个值，右关节目标为z乘不同系数，左关节为相反符号，两个gripper为0.5±0.2z，均重复64步。它们没有真实弧度/位移单位，不是可发布的机器人动作。图像输入是随机固定特征；任务主要由合成语言特征编码z，因此不能声称演示了视觉感知能力。

AdamW lr=.0015、weight_decay0，201次更新，loss由真实 `runner(*batch)` 计算。float32用于本机CPU，不是论文bfloat16大模型配置。初始模型、最终模型都用同一seed778进行官方五步采样，另用779换初始噪声；条件仍是训练中的16个条件，不是保留任务泛化。

|验证|实际结果|可支持的结论|
|---|---|---|
|add_noise与手工公式|一致|DDPM正向噪声广播与ᾱ索引正确|
|输入与ACI hooks|67个序列位置，条件长度4/6/4/6|time、frequency、state和64动作路径可执行|
|原始loss与全维MSE|一致|训练目标确实为sample，reduction为全维平均|
|无效target扰动|loss增加0.0078125|不是仅有效维loss|
|初始梯度|输出头非零，lang adaptor为0|零输出头暂时阻断上游梯度|
|训练后梯度|五个所检查模块全部有限非零|语言、图像、state与核心网络在该demo中收到梯度|
|有效槽位采样MSE，seed778|0.13064983 → 0.00003848|同条件合成动作拟合明显改善|
|换噪声seed779的有效槽位MSE|0.00004147|同条件换噪声仍接近合成目标|
|无效输出维|全部精确为0|采样最后乘mask生效|
|语言padding改为999|预测最大差0|当前布尔attention mask生效；在训练后的非零模型检查|
|独立进程重载|strict load通过，采样最大差0|保存权重与配置足以复查核心推理|

全维sample MSE为0.0000042089，明显小于有效维指标，因为114个无效槽位最终全部为0；不应用这个较小值夸大动作精度。训练最后一步全维loss约5.2676e−6，和采样MSE还存在不同噪声/调用方式，不能当作同一个测量。

![合成训练与官方采样](<G:/具身/论文阅读/RDT-1B -  a Diffusion Foundation Model for Bimanual Manipulation/复现/outputs/demo曲线.png>)

左图纵轴log scale，反映全128维训练MSE；右图横轴为64动作块的位置，**不是五个去噪时间步**。右图只展示右joint0，两种目标约−.7/+.7。没有物理环境step，无法从图推出轨迹可执行、稳定或真实成功率。

详见 [复现记录](<G:/具身/论文阅读/RDT-1B -  a Diffusion Foundation Model for Bimanual Manipulation/复现/README.md>)、[官方核心demo脚本](<G:/具身/论文阅读/RDT-1B -  a Diffusion Foundation Model for Bimanual Manipulation/复现/scripts/rdt_core_demo.py>)、[重载脚本](<G:/具身/论文阅读/RDT-1B -  a Diffusion Foundation Model for Bimanual Manipulation/复现/scripts/reload_and_plot.py>)、[原始结果](<G:/具身/论文阅读/RDT-1B -  a Diffusion Foundation Model for Bimanual Manipulation/复现/outputs/结果.json>). 状态为**部分成功**；1B预训练权重、完整前端、真实HDF5、论文benchmark和真机均未复现。

## 14. 掌握层次与 Paper ↔ Code

**必须掌握**：128维槽位不等于128自由度；state/action/action_mask各自含义；sample预测与epsilon预测的区别；64动作块、1000训练噪声等级、五步采样；输入mask、attention mask、训练reduction、评估reduction的区别；RDT输出怎样再映射回机器人接口。

**第二遍掌握**：QKNorm、RMSNorm、ACI与位置嵌入；零输出头初始梯度；DPM-Solver默认配置；模态随机mask与状态noise；one-shot/新场景测试协议和有限样本置信度。

**可以暂时不掌握**：21TB多机器人预训练、producer/consumer缓冲系统、48张H100训练与DeepSpeed；ALOHA/ROS设备驱动细节。不要先把大系统装完才理解公式。

|Paper ↔ Code|实际定位|证据与边界|
|---|---|---|
|Eq.(2)训练|RDTRunner.compute_loss → DDPMScheduler.add_noise|实际公式、target和全维MSE验证通过|
|Eq.(1)理论采样|论文定义；Runner实际用DPMSolverMultistepScheduler|DDPM理论式不能当五步solver实现|
|Figure 3动作网络|RDTRunner.adapt_conditions → RDT.forward → RDTBlock|缩小官方模型前向/反向通过|
|ACI/QKNorm|RDT.blocks[i%2]；CrossAttention、timm Attention/RmsNorm|交替hook与padding行为实际验证|
|Table 4统一空间|configs/state_vec.py；data/hdf5_vla_dataset.py|官方索引实际用于demo；全部源数据转换未审核|
|训练数据与总loss|train/dataset.py → train/train.py → Runner|代表主链静态核对；真实数据加载未运行|
|离线采样评估|train/sample.py.log_sample_res|静态核对masked MSE与按state_norm缩放误差；不是success rate|
|部署反映射|scripts/agilex_inference.py等官方部署入口|导航已定位，真实ROS/硬件执行未运行|
|论文真机成绩|论文Tables2/3与实验协议|只解释原文数字，未重测|
