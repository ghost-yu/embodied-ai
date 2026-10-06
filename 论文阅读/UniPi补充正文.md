## 8. 附录 A：论文到底训练了什么【建议会】

论文附录A.1/PDF14页不是空白的实现描述。视频生成采用Video U-Net：base channels=512、3个residual blocks、channel multiplier=[1,2,4]、attention resolutions=[6,12,24]、attention head dimension=64、condition embedding dimension=1024。这里512是基础通道数，1024是条件表示维度，不能解释成视频帧数。时间维使用temporal convolution而非temporal attention，目的为保持局部时间一致性；空间attention配置仍存在，不能据此说整个模型没有attention。

噪声schedule的log SNR范围是[−20,20]。信噪比SNR是信号方差/噪声方差，log SNR不是机器人时间；高SNR接近清晰视频、低SNR接近噪声。正文的α/σ要结合这个schedule理解，没有完整源码就不能把某个公开DDPM默认beta列表说成原训练设置。作者训练每个视频模型2M steps，batch2048，learning rate=1e−4，前10k linear warmup，使用256 TPU-v4 chips。batch与steps直接相乘为约40.96亿样本呈现次数，不等于40.96亿个独立视频；样本会重复。

文本由4.6B参数T5-XXL处理。仿真中粗视频10×48×64，每8个原始帧采一帧，模型1.7B；时间超分辨率20×48×64，每4帧采一帧，也是1.7B。10/20是视频帧数，48/64是空间分辨率，4/8是相对于源视频的采样间隔；没有源视频控制频率就不能换算为秒或Hz。真实视频实验列出了16×40×24、32×40×24、32×80×48、32×320×192等生成/超分辨率模型及1.7B、1.7B、1.4B、1.2B参数设置；不能将最终320×192套到Table4的低分辨率指标。

附录A.2：inverse dynamics输出7维控制，输入是图像观察，以MSE训练。网络为3×3 convolution、三层带residual connection的3×3 convolution、空间mean pooling、(128,7) MLP；Adam，gradient norm clip=1，lr=1e−4，2M steps，10k warmup。附录没有充分确定两帧/历史帧精确打包的可运行接口，不能只凭这段话发明官方`forward`的shape或默认参数。主文明确视觉转换功能，具体对象仍待源码验证。

附录A.4把动作解释为6个joint值+离散contact。对于连续预测，contact>0.5即接触，最近block附着到gripper；Pybullet joint controller执行，尝试达到预测joint状态，timeout2秒用于物理不可行预测。这不是论文动作全为末端笛卡尔delta xyz，也不是七个joint。组合任务只有20k视频带动作annotation，用于inverse训练，而video planner约200k示范；不能把两者都写成200k带动作。A.5 CLIPort部分inverse则用200k带动作视频。两组数据设置不同。

baseline训练细节沿用inverse的lr/warmup/clip等。Transformer BC为4个attention layers、8heads、hidden512，context4且每4帧采样；作者试context8未帮助。TT还预测未来8controls，autoregressive版本误差累积；state diffusion通过把控制复制到像素位置来复用U-Net结构。原文支持某些超参数共享，不代表所有baseline预训练、输入与计算成本完全相等。比较UniPi、状态diffusion与BC时应同时看视觉中间变量、语言、动作annotation和模型容量。

## 9. Figure 2 再看一遍：哪些模块真正有参数【必须会】

左上输入是真实观察x0和目标text；Tiling箭头把同一观察沿未来时间复制，复制没有可训练参数。灰色格子表示带噪的未来视频，video diffusion框是有参数的条件去噪器；它重复调用同一网络跨生成噪声时间k，不是每个机器人帧h各训练一个模型。输出稀疏图像进入有参数的temporal super resolution，再输出更密的执行图像。右侧inverse dynamics是另一组参数，把计划图像转换成机器人可执行控制；上方机器人示意是执行后果，不是另一个生成模型。

训练有两条监督路径：有文字的视频训练planner/TSR，带控制标签的transition训练inverse；推理时没有未来真实图像可喂入planner。若评价时把target未来帧给inverse，那是oracle诊断，不能叫生成计划执行。起始帧复制避免画面状态漂移，但没有形成刚体、碰撞、IK的硬约束；生成漂亮的图像仍可能不可执行。

论文§3.1还允许test-time prior h(τ)：classifier对任务约束，或Dirac delta固定某个目标图像。h是视觉轨迹上的额外约束/先验，不等于已经存在通用robot reward API。CFG强化语言/起始帧条件与额外prior guidance是两件事；本地demo只实现语言CFG，没有classifier prior、Dirac目标约束、互联网迁移或多任务组合泛化。

## 10. 本地教学实现：数据与代码地图

已保存官方网页仓库的recursive tree和index.html，核查范围及日期见`资料/官方资源核查.json`。该入口包含网页、样式、展示素材，没有找到Python模型、dataset、train/eval入口、checkpoint或依赖清单；这是对已查入口的结论，不宣称所有互联网实现都不存在。仓库tree SHA是网页快照，**不是官方训练代码commit**。本地脚本由本次编写，没有用AVDC或第三方实现冒充UniPi。

入口`unipi_demo.py -> train -> dataset/render -> VideoDenoiser -> weighted_pixel_loss -> Adam`。训练后`sample -> densify(Midpoint) -> execute(InverseDynamics) -> DotEnv.step`。`verify_and_plot.py`在独立Python进程strict加载三个权重，重算整条链、比较保存fixture、验证反向指令和CFG、绘图。这不是官方函数名，也不是论文架构的逐行实现。

|模块|论文作用|教学选择|保留内容与重要不同|
|---|---|---|---|
|视频数据|任务文字+视频|四条固定英文command、512合成轨迹|有图像/语言条件；无自由自然语言或T5|
|planner|Video U-Net|小Conv3D clean-pixel predictor|在像素视频上加噪/去噪；非1.7B U-Net|
|tiling|x0按时间复制并concat|`unsqueeze(2).expand`与`cat`|实际将首帧在每个未来帧处作为输入|
|时间层级|video diffusion TSR|监督学习Midpoint CNN|粗4帧→密8帧；补帧不是扩散TSR|
|inverse|卷积/mean pooling/MLP，7controls|图像moments→小MLP，2controls|独立学习动作；借助已知toy相机特征，非论文inverse|
|执行|Pybullet joint controller|DotEnv二维位置更新|真实调用step并渲染结果；无机器人/碰撞/IK|
|评估|任务基准、画质/分类指标|保留起点、像素MSE、终点距离|教学指标不能比论文success/FID/FVD|

所有模型随机初始化，训练/执行都在CPU、float32；复用OpenVLA的PyTorch2.2.0+cpu环境，没有下载大模型、真实数据或安装新框架。重用环境不表示重用OpenVLA算法。

数据来自`dataset(n,generator)`，起点每轴在[6,9)像素，方向为right/left/down/up，对应向量(1,0)/(−1,0)/(0,1)/(0,−1)。每次control为0.4×方向，连续8步；h=0…8共9帧。横轴x右为正，纵轴y向下为正，单位为像素，**不是米、joint角或牛顿**。`render`用σ=.85的Gaussian亮点，灰度[0,1]，背景0；不是仿真机械臂画面。

训练512轨迹，保留64轨迹由不同generator seed生成新的起点，指令仍是训练过的四种。它衡量同分布起点泛化，不能叫未见任务/语言组合泛化。coarse取h=2,4,6,8四帧；从第一帧和coarse各相邻端点学习h=1,3,5,7的midpoint。inverse另用4096随机transition，起点[2,13)，动作每轴[−.8,.8)，不要求是四种任务动作；planner与inverse的数据独立。有限16×16画幅近边缘可能有Gaussian截断，逆模型测试起点[3,12)减轻边缘影响，不代表已验证任意相机边界。

|变量名与命名来源|dtype/shape|流向|
|---|---|---|
|start/positions，环境位置|float32 `[B,2]` / `[B,9,2]`|仅用于render/真实环境reset/评价；不直接输入planner或inverse|
|first，首帧观察|float32 `[B,1,16,16]`|去噪条件和首transition|
|ids，command编号|int64 `[B]`，0…3，4=null|Embedding，不能处理任意字符串|
|coarse，稀疏真实视频|float32 `[B,4,1,16,16]`|permute→clean target|
|noisy/clean，Conv3D布局|float32 `[B,1,4,16,16]`|噪声等级k条件去噪|
|tiled，沿时间首帧|float32 `[B,1,4,16,16]`|输入拼接|
|frame_index，机器人帧编码|float32 `[B,1,4,16,16]`|0,1/3,2/3,1，区别于diffusion k|
|cond，文字与噪声时间|float32 `[B,32,1,1,1]`|broadcast给每个spatiotemporal feature|
|dense，补齐未来视频|float32 `[B,8,1,16,16]`|与first构成9帧/8transition|
|moments，图像加权坐标|float32 `[B,2]`|两帧之差输入MLP|
|actions/applied|float32 `[B,8,2]`|raw推断动作/环境限幅后真正动作|
|observations|float32 `[B,9,1,16,16]`|由执行状态重新render，区别于dense生成画面|

`permute`调换轴，不改变数据值；Conv3D期望channel在frame前。`flatten(0,1)`把batch和时间合并为B×F个独立两帧样本；`stack`新建一条轴，`cat`沿已有轴拼接；补帧用stack再flatten交错midpoint/keyframe，没有把8步当成8个batch。`expand`是共享原数据的view，复制无模型参数；本脚本没有对它做in-place写入。

## 11. 教学扩散训练与 DDIM 公式【建议会】

教学选择64级linear β_k，范围.001到.2，$\bar\alpha_k=\prod_{j=0}^{k}(1-\beta_j)$。这里ᾱ是累计系数，原文连续α_k记号是信号乘数，二者不能混写；教学加噪为：

$$z_k=\sqrt{\bar\alpha_k}\,v+\sqrt{1-\bar\alpha_k}\,\epsilon,$$

v是真实coarse像素视频，ϵ同shape标准正态。`a=ab[k][:,None,None,None,None]`把每个样本的系数广播到channel、frame、像素，不能把B不同样本误用同一k。例：ᾱ=.25、像素v=.8、ϵ=−.2，则z=.5×.8+√.75×(−.2)≈.226795。k=0依然有.001噪声，不是完全干净视频。

网络直接预测clean视频$\hat v_\theta(z_k,k,x_0,c)$，不是预测epsilon。教学损失：

$$L=\operatorname{mean}[(\hat v-v)^2(1+5v)].$$

Gaussian亮点很稀疏，纯背景会主导普通MSE，所以这里给亮点更大权重。权重以真实像素v确定，是**自选教学损失**，不是从论文抄来的精确training objective。v=.8时权重5，预测.6则该像素项=.04×5=.2；背景v=0时权重1。训练曲线是weighted clean-pixel loss，评估coarse/dense是普通MSE，不能直接比较数值当作同一指标下降。正文原文denoiser s的精确参数化没由源码验证，不能据本demo宣称论文也这样训练。

训练以15%概率将command ID设4，保留first frame。null分支只删除语言；正文CFG的unconditional s(τ_k,k)删除文字/图像，这里与它不同。教学omega=0主评估用conditional；额外omega=.5时仍使用(1+ω)conditional−ωnull形式，但加强的是给定first frame下的文字条件。四指令Embedding有可学习参数，却没有网络知识迁移或tokenizer。

`sample`从标准正态视频开始，k=63…0共64次。由clean预测反推noise：

$$\hat\epsilon_k=\frac{z_k-\sqrt{\bar\alpha_k}\hat v_k}{\sqrt{1-\bar\alpha_k}},\qquad
z_{k-1}=\sqrt{\bar\alpha_{k-1}}\hat v_k+\sqrt{1-\bar\alpha_{k-1}}\hat\epsilon_k.$$

这是本地deterministic DDIM式，η=0，没有每步重采噪声。最后k=0时取previous ᾱ=1，输出clean预测；不访问ab[−1]。示例ᾱ_k=.25、previous=.36、z=.226795、预测clean=.8，得ϵ≈−.2，previous=.6×.8+.8×(−.2)=.32。最后累计信号并非数学上严格0，因此标准高斯初始化是近似终点先验，具体终点ᾱ记录在结果中。无输出clip，生成像素可能<0或>1，图像显示限制[0,1]仅为可视化，不修改评估或执行输入。

每次forward输入拼接noisy、tiled、frame_index，得到3channels；Conv3D转32features。time MLP输入k/63，文字Embedding与time相加再广播。4个residual Conv3D通过`z = z + .5*silu(layer(z+cond))`混合空间与时间；最后32→1channel得到clean。残差表示保留原features同时学习修正，SiLU不是softmax，不输出概率。粗视频模型没有latent坐标diffusion：真正加噪的是16×16×4像素tensor。

## 12. 补帧、逆动力学与环境执行【必须会】

`Midpoint.forward(a,b)`将两端灰度图拼成2channel，CNN输出中间图。训练ground-truth端点/中间图；推理使用generated端点，存在训练/推理分布差。它学的是本环境匀速Gaussian亮点插帧，不解决抓取接触、遮挡、复杂运动；**原论文的时间超分辨率是扩散模型，本demo为deterministic CNN**。先生成keyframe再学补帧保留了阶段接口，而非完整重现论文层级模型。

`moments`将输入负灰度截为0，计算图像mass和坐标加权均值：

$$m_x(I)=\frac{\sum_{y,x}x\max(I_{y,x},0)}{\max(\sum_{y,x}\max(I_{y,x},0),10^{-6})},\qquad m_y(I)\text{ 同理}.$$

这里x/y是像素坐标index，1e−6防止空白图除0。单个亮点在(10,7)时提取近似(10,7)；多点、灰色背景或生成伪影会偏置重心，不能推广成通用robot perception。inverse收到$m(I_{h+1})-m(I_h)$，由2→32→2 MLP预测动作。它的输出不是直接返回坐标差，参数通过有标签control MSE学习；但已知toy camera与低维预处理显著降低任务难度。读者不能把这种适配器性能迁移到论文卷积inverse。

`execute`先构成whole=[first,dense]九帧，对八对相邻图一次性预测所有动作，然后才reset/step。这就是open-loop：真实执行observations未反馈到planner/inverse，goal仅用来评价距离，未输入模型。midpoint与inverse共同误差可能累积，生成图像的错误不由环境自动修正。oracle视频执行使用真实未来video给同一个inverse，以分离inverse本身误差和video生成误差；它是诊断上限，不计入生成计划成功率。

`DotEnv.reset(start,goal)`保存position、goal、steps=0，并render观察。`step(action)`按每维[−.8,.8]限幅，再位置限制[0,15]，steps加1，reward=负终点欧氏距离，done在steps==8。限幅是预先定义的环境控制范围，不是为测试修好模型输出；raw/applied不同数量独立报告。该简化API只返回done，没有Gymnasium的terminated/truncated分拆；无reset后持续step保护、无物理碰撞、无force/contact。没有Hz，时间单位是离散环境步。

|对象、所属、默认值|输入/返回|参数或状态变化、调用位置|
|---|---|---|
|`dataset(n,generator)`，函数，无默认值|n int、torch.Generator；五个tensor|推进generator RNG；train构造合成数据|
|`render(pos)`，函数|[...,2]→[...,1,16,16]|无模型状态，位置转camera观察|
|`VideoDenoiser.__init__()`，方法|无额外参数；建立Module|初始化Embedding、time MLP、Conv3D、blocks|
|`VideoDenoiser.forward(noisy,k,first,ids)`|[B,1,4,16,16]等→同shape|返回clean预测，前向无手动参数修改|
|`weighted_pixel_loss(pred,target)`|同shape视频→scalar|无状态；Adam训练前backward入口|
|`sample(model,first,ids,noise,omega=0.)`|初始噪声→[B,4,1,16,16]|clone噪声再更新局部z，无模型parameter变化|
|`Midpoint.forward(a,b)`|两幅[B,1,16,16]→中间图|监督学参数；由densify调用|
|`densify(midpoint,first,coarse)`|首帧+4keyframes→8未来帧|无环境变化；默认没no_grad，独立重载脚本可构建暂存graph|
|`InverseDynamics.forward(a,b)`|两幅图→[B,2]控制|无真实position输入；学习MLP参数|
|`DotEnv.reset/step`|位置/goal、action；图像/reward/done/info|改变self.position/goal/steps；execute调用|
|`execute(inverse,first,dense,starts,goals)`|八步计划/真实起点→raw/actions/距离/applied|no_grad；构造本地环境并实际step八次|
|`train()`，函数|无参数；写outputs|三个独立Adam训练、RNG、文件权重和日志有变化|

`class ... (nn.Module)`是可训练对象继承；`__init__`建层并通过`super().__init__()`注册Module状态；`model(...)`触发forward。`ModuleList`注册4个卷积层的参数，普通Python list不一定会被optimizer/state_dict发现。`opt.zero_grad()`清旧梯度，`backward()`计算这次梯度，`clip_grad_norm_`对video梯度限1，`opt.step()`更新权重。midpoint/inverse代码没有梯度clip，不能说三者配置相同。`@torch.no_grad()`阻止autograd记录，是推理内存选择，不是冻结/删除模型参数；`.eval()`改变训练模式，对此模型无dropout/batchnorm，因此不产生额外随机行为。command drop通过训练循环显式实现，eval时不存在15%drop。

`torch.save`保存三个state_dict而非完整Python模型对象；重载构造同样的类，`strict=True`要求key/shape全匹配。fixture保留输入噪声、first、ids、target和原结果，独立进程重新生成/补帧/执行后逐元素比较，比仅打印“加载成功”更强；它仍不能证明真实机器人接口正确。
