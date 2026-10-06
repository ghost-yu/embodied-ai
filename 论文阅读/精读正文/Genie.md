## 1. 只有视频，怎样得到可控世界

通常动作条件世界模型训练需要视频和动作，而互联网视频常没有控制标签。Genie学习latent action，让一个小离散码表示帧间变化，借此训练可按键控制的视频环境。它不是已经具备真实机械臂关节控制接口的机器人策略。

贡献是latent action model（LAM）、时空视频tokenizer、动作条件dynamics model的组合，以及无真实动作标签的可控生成/迁移实验。主要数据是2D平台游戏视频，同时测试机器人视频；不能把游戏结果直接扩展成现实物理精确模拟。

地图：§2/PDF3–5页和Figures3–8给三个模块、ST attention和推理；§3/PDF5–6页给数据/指标；Figure9给scaling；Figures13–15给控制/迁移；Tables2–3/PDF9页给LAM输入和tokenizer消融；附录给具体模型与训练规模。

## 2. Figures3–8：训练和推理分开【必须会】

视频tokenizer先训练，将原始视频编码成离散token并可解码重建。然后LAM从原始帧推断潜在动作，dynamics读取过去video tokens和latent action，预测下一帧tokens。LAM自身decoder给它训练信号，但推理交互时LAM encoder/decoder被丢弃，只保留action codebook；用户选择action ID，由codebook给embedding。

图中的未来帧在LAM训练用于推断“发生了什么变化”，不允许推理时偷看真实未来帧。推理：提示图像→video tokenizer→用户action ID→action embedding→dynamics生成下一帧tokens→video decoder→显示新帧，重复。

|变量|类型/shape|意义|
|---|---|---|
|x1:T|图像浮点 `[B,T,H,W,C]`，论文布局|原始视频；模块内布局以实现为准|
|video z1:T|整数 `[B,T,N]`|每帧N个压缩位置；N不是词表大小|
|video codebook|浮点 `[1024,32]`，主配置|1024种视觉码，32维embedding|
|action IDs|整数 `[B,T-1]`|帧间8选1，不是机器人动作向量|
|action codebook|浮点 `[8,32]`|8种潜在控制，每种32维embedding|
|dynamics logits|浮点 `[B,T-1,N,1024]`，概念shape|分类预测未来视觉token|

主设置训练序列16帧、10FPS。video tokenizer patch=4，LAM patch=16，二者不共用相同输入token数。论文video tokens中的D是每帧离散位置维度，codebook embedding32另有含义，不能混用。

## 3. 潜在动作怎样受约束

LAM encoder看到历史和下一帧，产生连续 $\tilde a_t$；量化到少量码；decoder仅靠历史+这个码重建下一帧。信息瓶颈迫使码描述有助于下一帧预测的变化，action vocabulary |A|=8也方便人交互。

VQ机制教学式，非Genie论文编号公式：

$$k^*=\arg\min_k\|h-e_k\|^2,\qquad q(h)=e_{k^*}.$$

h为连续latent `[D_e]`，e_k为codebook embedding，k为离散ID。例：h=[0.9,0.1]，两个码[1,0]与[0,1]，距离平方0.02与1.62，选择前者。这个例子只解释量化，不表示Genie真实codebook二维或只有两个码。

可用于理解VQ-VAE的通用loss为reconstruction+codebook+commitment：$L_{rec}+\|sg(h)-e\|^2+\beta\|h-sg(e)\|^2$。sg为stop-gradient，β约束encoder保持接近码。**这是通用教学公式**，本轮没有官方源码，不能声称它及reduction/更新方式逐行等于Genie实现。VQ、tokenizer和LAM的训练细节仍应以原论文附录为准。

无标签潜在动作不一定可唯一辨识：ID可能被置换，摄像机运动/场景变化也会影响帧差。作者用低容量和预测任务促成可控性，实验展示一致动作意义，不等于数学保证潜在码是真实物理力/关节量。

## 4. ST-transformer为什么省计算

令每帧N=H_pW_p个patch，共T帧。全attention对TN位置做两两打分，量级O(T²N²)。ST块先在每帧做空间attention，量级O(TN²)；再对每个空间位置沿时间做attention，量级O(NT²)。这是忽略hidden宽度的打分规模比较，不是整模型精确FLOPs。

例：T=16、N=100，全attention需2,560,000对，分解为160,000+25,600=185,600，约13.8倍差异。空间attention可读同帧不同位置；时间attention有因果mask，只读过去。时空混合要多层交换信息，所以计算更省与全attention表达结构并不完全相同。

Figure4中spatial、temporal、FFW构成一个block；论文省去post-spatial额外FFW。不能把两种attention都叫“全部位置互相看见”。tokenizer也采用因果时序编码，因此某帧编码不应包含尚未出现的未来信息。

## 5. Dynamics：帧间自回归，帧内MaskGIT

过去z和stopgrad latent actions作为条件，模型预测下一帧token分布，以交叉熵监督。教学式：

$$L_{dyn}=-\frac1{|\Omega|}\sum_{(b,t,n)\in\Omega}\log p_\theta(z_{b,t,n}\mid z_{<t},a_{<t},\text{可见/遮挡信息}).$$

Ω表示当前训练目标位置，具体mask/reduction需官方实现确认；标签是整数，logits末轴为visual vocabulary。每个正确token概率0.8，单项loss≈0.223。

训练随机mask率从[0.5,1]采样；latent action作为additive embedding注入，不是必定追加成独立序列位置。推理每帧做25次MaskGIT refinement，temperature2，再解码图像。**帧间是自回归时间推进，帧内是并行遮挡token的迭代生成；不是一个像素一个像素语言式解码。** 25为帧内采样迭代，16为训练帧长度，8为动作码数。

## 6. 可控性指标与定量消融

§3/PDF6页原公式：

$$\Delta_t\operatorname{PSNR}=\operatorname{PSNR}(x_t,\hat x_t)-\operatorname{PSNR}(x_t,\hat x'_t).$$

x_t是真实帧；x̂用从真实视频推断的action生成，x̂′用随机action生成。所有实验用t=4。越大说明与真实对应的action更有助于复原，随机action结果差异更明显。它不单独证明人类意图可控，也不验证机械臂动作安全可执行。

PSNR教学式 $10\log_{10}(MAX^2/MSE)$；MAX为像素最大值。若两种生成MSE分别0.01和0.04，MAX=1，则PSNR20与约13.979，差约6.021dB。pixel归一化、计算轴和aggregate规则影响实际值；本轮不是复现原评估实现。

Table2/PDF9页Platformers：token-input FVD38.8、ΔPSNR1.33；pixel-input Genie40.1、1.91。画质FVD反而差1.3，但可控性高0.58，说明指标不能互相替代；两者参数量2.3B/2.5B也不完全匹配。Robotics相同1B设置：FVD257.8→136.4下降121.4，Δ1.65→2.07高0.42。

Table3同patch10、batch128、T16下：ViT FVD114.5、Δ1.39；ST-ViViT81.4、1.66，分别改善33.1与0.27。C-ViViT272.7、1.37较差，但参数230M/225M/205M不同、memory有差异，应按该实验条件比较。

Figure15 CoinRun迁移：冻结LAM给专家视频打latent labels，学习latent policy，再用少量真实动作样本做latent→real mapping。原文说约200个expert samples即可与oracle score相近；因此迁移**并非完全没有真实动作校准**。图报100个levels、五seed和95%置信区间，没有逐柱精确原始数值时不编造精确成功率。

## 7. 阅读与实现边界

原版Genie公开论文可核对；本轮未找到完整原版官方训练源码。下面代码分析针对第三方GenieRedux的固定快照，明确标注原文与复现的差异，不能称为DeepMind官方行为。资源核查见 [记录](<G:/具身/论文阅读/Genie -  Generative Interactive Environments/资料/官方与复现资源核查.json>)。不混入Genie2/3、自动驾驶或相关论文的新基准结果。

## 8. 原论文训练预算和附录 F：小例子也有真实成本

本篇固定阅读arxiv2402.15391v1，共27页，SHA-256见来源文件；不是后续Genie2/3、Genie Envisioner或商业交互产品。主文§3/PDF6给video tokenizer约200M参数、LAM约300M、dynamics约10.1B，合计约10.6B，常四舍五入为11B。Table5/PDF22又列Platformers LAM encoder/decoder各20layers、d_model1024、16heads、8codes、latent_dim32、patch16；Table7列tokenizer encoder12layers/512/8heads、decoder20layers/1024/16heads、1024codes、patch4、latent32。参数量与表格配置只在其原训练上下文解释，不能把小demo计数拿去证明原模型大小。

主配置使用16frames、10FPS，视频原分辨率160×90；源片段为16秒（约160帧），不是16frames训练裁片。图像帧时间、空间patch大小和离散词表是不同轴。dynamics用bfloat16、QK normalization；主生成每帧25次MaskGIT、temperature2。温度不是环境动作强度，也不是帧率。本地实验float32、8×8、5frames，时间以合成离散步计，没有10FPS或真实渲染耗时的对应。

附录B/PDF21–22的数据筛选流程很重要。最初55M视频片段，最后6.8M（约12.36%）/30k小时。团队人工给10k视频标质量等级，训练11M ResNet18进行好/坏质量分类，再按置信度筛选。这些是**质量标签，不是控制动作标签**；“无动作标签学习”不等于整个数据工程完全无人标注。Table4同580M模型，原数据FVD61.4，筛后54.8，下降6.6（约10.75%）；支持质量过滤有用，但不是数据量越少越好或动作模型完全零人工成本。

附录C.2 tokenizer AdamW：300k steps，max_lr/min_lr均3e−4，betas(.9,.9)，weight_decay1e−4，10k warmup。文字称cosine decay，但表中最大最小相等意味着按该表不能推导有非零幅度的lr下降。C.3 scaling运行200k steps、batch256，每次750B tokens；Table10从41M到2.7B配置、硬件和时间都变，不是固定单卡的实时FLOPs对比。最终Table12 10.1B dynamics为48layers、36heads、d_model5120、k/q_size128，125ksteps、batch512、256TPUv5，总6.6×10²² FLOPs。不能强套d_model=heads×k/q_size：36×128=4608，而Q/K projection内部维度可不同于残差宽度5120。

附录F说单中档TPU/GPU一周内的CoinRun例子，仍需“hard”模式10k level seeds、每level1000step，总10M transitions、tokenizer300ksteps约三天，再LAM/dynamics各200ksteps。tokenizer8layers、dim512、8heads、1024codes、patch4、latent32，batch48×16frames；LAM8layers/512/8heads、**6codes**，并非所有设置均8codes。joint stage batch36×16frames，在16GB设备预算内。Table17将num_layers字段列了两次（12与8），没有官方配置源码就不能自行断言第二个字段本应是heads；保留表的歧义。温度1.0、25sampling steps对应这个case，不是主设置temperature2。

附录E进一步说明CoinRun迁移：冻结LAM从(x_t,x_{t+1})取得latent label a_t，policy只从观察历史预测a_t；推理前用少量带真实动作u_t的专家transition构造字典D[a_t]，每个latent对应一列可能动作，再采样真实u。约200labels是动作映射校准成本，不是无监督训练已自动发现键盘绑定。policy用4帧历史与过去action one-hot additive input、batch16。真实动作校准、latent imitation policy与视频生成是不同实验，不把它们合称“完全无动作监督控制机器人”。

## 9. 从图到可执行对象：原版、复现代码和本地实验分清

原论文Figure3是两条并行分支：蓝色video tokenizer将T帧压成T帧visual tokens；黄色LAM将视频变化压成T−1个action codes；二者喂给红色dynamics预测未来。LAM encoder训练必须看到要解释的下一帧，LAM decoder预测下一帧时只拿过去帧与动作瓶颈，不能旁路传递未来像素。Figure5中下面跳线是历史观察，不是未来图像绕过codebook。

Figure4中每个spatial attention读同帧所有patch，每个temporal attention读同空间位置历史；多层交换空间/时间信息。Figure6的video tokenizer encoder/decoder共同以重建训练，VQ视觉码在这条路径，动作码是另一套权重。Figure7利用latent action stopgrad，dynamics监督目标是visual token分类。Figure8交互时只保留动作codebook查表，不运行LAM encoder“猜未来动作”，用户ID代替其输出。video decoder显示预测画面，并不保证真实物理环境已执行相同控制。

本次重新检查[DeepMind论文入口](https://deepmind.google/research/publications/60474/)与原文，没有找到可确认的完整原版训练源码/权重。[Open-Genie](https://github.com/myscience/open-genie)明确自称unofficial；本次采用[GenieRedux](https://github.com/insait-institute/GenieRedux)的核心模块。后者是相关研究的开源复现/扩展框架，不是原Genie官方代码。固定commit **bc80fe43c8370fd79ac5f9fc1719c65c52e294db**，没有修改其源码；代码检查与运行结论仅针对该快照。README还介绍RetroAct、AutoExplore与guided variant，本轮没有运行这些扩展任务。

|仓库地图|实际定位|本轮范围|
|---|---|---|
|README、install、环境|README.md、install.sh、genie_redux_env.yaml|环境文件为Linux/Python3.13/Torch2.8参考，未整套照搬|
|完整训练入口|run.py、train_genie_redux.py、training/trainer.py|静态导航；数据/Accelerate全训练未执行|
|数据|data/data.py、configs/config/genie_redux.yaml|配置指向RetroAct与预训练tokenizer；本轮用自生成视频|
|video tokenizer|models/tokenizer.py → components/stvivit.py|实际小配置训练/编码/重建|
|LAM|models/lam.py → STViViT、VectorQuantize|实际无动作标签训练、查码与teacher-forced重建|
|dynamics|models/dynamics.py：MaskGIT、Dynamics|实际训练/采样；实现限制见后文|
|组合对象|models/genie_redux.py：GenieRedux/GenieReduxGuided|静态核对冻结与动作广播；本轮直接组合核心类|
|attention/VQ|components/attention.py、vector_quantize.py|追踪causal mask、codebook更新和梯度|
|eval|eval_genie_redux.py、training/evaluation.py|原bench/FVD/环境评估未执行|
|本地入口|genie_core_demo.py、vq_weight_ablation.py、verify_and_plot.py|合成训练、单因素对照、独立重载和源码采样诊断|

主路径是：自生成videos → Tokenizer训练 → LAM训练 → 冻结两者 → video IDs/LAM codes → Dynamics masked CE训练 → codebook user ID → sample → video decode。论文描述tokenizer先训练，LAM/dynamics联合/并行；本地为方便隔离问题而顺序训练LAM再训练dynamics。它是实际第三方核心的缩小实验，不能宣称复现全部原joint训练recipe。

## 10. 本地数据与 shape trail【必须会】

合成环境为8×8 periodic二维图像：一个Gaussian亮点（σ=.8、峰值.75）与固定.05checkerboard背景。真实位置是整数(x,y)，初始均匀0…7，每步随机选right/left/down/up使位置加(±1,0)/(0,±1)，再mod8，越界回到另一侧；没有重力、碰撞、joint、奖励或FPS。render按periodic距离画图，所以越界亮点会同时出现在两边，不是视频生成失真。训练512条、测试64条新视频，每条5frames/4transitions，seed97/98。新的序列不等于没见过的场景，因为只有64个起点和四方向，训练很可能覆盖全部位置。

训练入口使用`videos,_ = dataset(...)`丢弃真实方向标签；labels仅在evaluation用于事后统计。没有监督LAM“哪个ID是left”，没有将真实动作传给dynamics。为衡量重建，用LAM从真实测试视频推断动作是合理的oracle latent condition实验；为交互，从first image和用户选择ID生成，不提供真实未来视频。不要把这两种输入条件混在同一能力结论里。

|变量及来源|dtype/shape|实际流向|
|---|---|---|
|videos，render输出|float32 `[B,1,5,8,8]`|像素范围[0,.8]；两个encoder均从像素输入|
|tokenizer patch embedding|float32 `[B,5,4,4,32]`|patch2，即每帧16位置|
|packed video h|float32 `[B,80,32]`|5×4×4合并；送VQ projection|
|video IDs|int64 `[B,5,4,4]`|0…31，词表32不等于每帧16位置|
|visual codebook|float32 `[32,16]`|projection-in32→16/out16→32；未保持原1024×32|
|LAM patch embedding|float32 `[B,5,2,2,32]`|patch4，每帧4位置|
|LAM pooled/flattened action h|float32 `[B,4,32]`|去首帧，从下一帧encoding获得4个transition codes|
|action IDs/codebook|int64 `[B,4]` / float32 `[8,16]`|与visual codebook完全独立|
|projected action codes|float32 `[B,4,32]`|查码后project_out，不是2D真实动作|
|singleton action input|float32 `[B,4,1,1,32]`|给dynamics.expand到4×4网格|
|dynamics input/target|int64 `[B,4,4,4]`|分别video[:, :-1]/video[:, 1:]|
|logits|float32 `[B,64,32]`|4×4×4的位置、32分类词表|
|single-step prime/new IDs|int64 `[B,16]`|首帧16码与预测下一帧16码|
|decoded joined video|float32 `[B,1,2,8,8]`|解码首帧+下一帧，只返回下一帧|

`class Tokenizer(STViViT)`与`LatentActionModel(STViViT)`继承共同patch/attention/VQ/decoder基础，但对象实例不同，各自codebook不会共享。`super().__init__(...)`负责建层；`nn.ModuleList/Sequential`注册内部参数。`model(...)`通常进入forward并触发Module hooks；此快照sampler显式调用`maskgit.forward(...)`，会绕过该对象的Module调用hooks，因此观察采样用内部`token_emb`的正常调用hook。

`Rearrange`使用einops模式指明每个轴；`pack([tokens], 'b * d')`把frame/空间合并，同时保存packed_shape，方便unpack恢复IDs。`tokens[:,1:]`是排除起始帧，`enc_input_tokens[:,:-1]`是decoder使用过去观察；一个产生未来变化code，另一个提供过去背景，时间对齐不能互换。`repeat`产生空间复制，`unsqueeze(2/3)`只加长度1的维度，`expand`广播无模型参数。

小配置的LAM返回tokens是`[B,4,2,2,32]`，而dynamics空间为4×4。第一次直接传这个tensor触发expand维度错误：非singleton的2不能扩成4。修复在本地连接函数`action_codes`中用`get_codes_from_indices(ids)`取得未复制投影码`[B,4,32]`，再加两个singleton轴让dynamics正确广播。没有改第三方模型文件，方法语义仍为同一动作码加到所有空间位置。组合类`GenieRedux.get_codes_from_indices`也展示根据tokenizer网格复制码的思路。

## 11. STViViT、attention 与 VQ 的真实行为【建议会】

STViViT对first frame与rest frames分别patch embed：LayerNorm → Linear → LayerNorm。first只能输入一帧；rest按temporal_patch_size打包，本次为1；F−1必须可整除该值。encoder order='st'，decoder order='ts'，都启用temporal causal attention与causal PEG。PEG是有参数的局部位置卷积生成器，不是仅一个无参数位置index。

encoder先将[B,T,h,w,d]重排为[B×T,h×w,d]作spatial attention，再为[B×h×w,T,d]作temporal attention，最后恢复原shape。本次tokenizer T5/N16；全attention打分(5×16)²=6400，分解为5×16²+16×5²=1680，只比较score矩阵元素数，忽略QKV/FFW/卷积等开销。LAM N4，为5×4²+4×5²=180。不能拿这些小数字解释原11B实际速度。

此复现代码的STBlock在spatial和temporal各执行一套FFW；原论文说明省略post-spatial额外FFW，因此**不是完全一致的原ST block**。Attention先L2-normalize q/k，再各乘可学习逐维q_scale/k_scale，最终score乘默认8，加spatial ContinuousPositionBias或temporal ALiBi，causal upper-triangle mask为−inf，再softmax。这里不是简单原始qk/√d；softmax作用为加权value，归一化axis最后的key维，不是动作ID概率。

原注意力通用式为$\mathrm{softmax}(QK^\top/\sqrt{d_h}+M)V$，Q/K/V的head_dim是d_h；本复现实际替换为scaled normalized QK加bias/mask。因果mask允许当前和过去，屏蔽未来。实测把最后一帧替换为随机图，之前四帧visual IDs与前3个LAM IDs均不变；最后一个LAM ID允许变化，因为它解释最后transition。此测试证明该输入的离散输出因果性，不代表全模型所有浮点路径或所有mask边界都已证明。

VQ配置为learnable_codebook=True、ema_update=False、use_cosine_sim=True、commitment_weight=.25、kmeans_init=False。它从随机码开始，由optimizer更新，没有EMA中心更新，亦没有自动dead-code重置（默认threshold0）。wrapper先把32维h线性投影成16维u并normalize，再由CosineSimCodebook计算u与embedding的dot logits取最大ID。名称叫cosine，但该learnable+非EMA分支对embedding没有每步显式重归一化，不能在未检查其norm时把每次打分都视为严格单位球距离；原论文具体codebook算法没有原源码验证。代码执行比类名更可靠。

【必须会】训练时用straight-through：

$$q_{ST}=u+\operatorname{sg}(e_{k^*}-u).$$

前向数值是选中e，反向对u的局部导数为1，避免离散argmax阻断encoder的reconstruction梯度；codebook靠独立q_loss更新。标量u=.7、e=.9，forward=.9，对u仍传梯度。这里sg是detach，不是修改tensor数值。小demo实际LAM codebook、to_action_emb和pixel head梯度均非零，支持这些连接收到训练信号，但不能据此证明codebook语义正确。

真实无mask训练损失可解释为：

$$L_{total}=L_{pixel}+\lambda_{VQ}\bigl(L_q+.25L_{commit}\bigr),\quad
L_q=\operatorname{MSE}(e,\operatorname{sg}(u)),\quad
L_{commit}=\operatorname{MSE}(\operatorname{sg}(e),u).$$

u是normalize后的16维投影；MSE在batch、时间/空间和维度上取平均；decoder pixel MSE在全部video像素取平均。原始小配置λ=.25，对照LAM λ=.01，内部commitment .25不改。外部λ与内部β不是同一个参数；对照不等于“把commitment改到.01”。eval模式VQ附加loss为0，所以eval loss可只反映pixel error，不能与train total直接同名比较。此快照decoder直接Linear→像素重排，**没有论文附录描述的sigmoid**；本地输出允许超出[0,1]，显示色阶不会修改实际tensor。

LAM把每帧2×2×32flatten为128，再Linear128→32与LayerNorm得到每帧action h，然后去首帧、VQ，复制码到空间；decoder输入为过去patch embeddings加action embedding。动作不是每个像素一个ID，而是整个transition一个ID。小码本鼓励低容量变化解释，但完全可能坍塌或编码位置/风格；信息瓶颈不是自动得到可解释动作的保证。

## 12. Dynamics 与 sampling：代码中要额外注意的路径

MaskGIT的Embedding有33个索引，visual IDs0…31加mask_id32；mask也是一种可学习输入，不是“全部填0”。pos_emb按flatten后的位置编号，max_seq_len256限制输入位置数。本地use_token=False，只有token_emb；use_token=True另加冻结tokenizer projected codes×10，需要参数dim兼容，未运行。is_guided=False表示动作embedding与video features相加，不表示模型无条件。guided=True则concat动作特征并增加hidden宽度；本轮未使用真实动作/one-hot guided分支。

Dynamics.forward先input=visual[:, :-1]，target=visual[:,1:]，flatten为[B,64]；从[.5,1]采一个整个batch共用的mask概率，再每个位置Bernoulli，输入遮挡位置置32。logits经过MaskGIT后，只在token_mask=True处取分类CE，普通mean reduction。正确类概率.8时单位置−log(.8)≈.223；若仅4位置遮挡则除4，不除全部64。video_mask是有效位置，token_mask是随机遮挡，两者意义不同。若极小序列一次没有mask目标，空CE边界需额外处理，本次长度64没有触发也未验证该情况。

paper正文描述mask输入z_{2:T−1}，而此快照对所有previous input位置抽mask，包含最初观察；因此训练路径仍与原文有差异。本地未改source，不能说它原封不动实现原first-frame protection。maskgit梯度缩放使用`x*.1 + x.detach()*.9`，forward值不变，向token/pos embedding的梯度变为.1倍；动作加在该缩放之后，不被同一条式缩放。最后Linear32→32预测视觉词表。

sampling默认值有实际问题：Dynamics.__init__把inference_steps、temperature、schedule分别存为单元素tuple，直接使用default时`range(inference_steps)`报TypeError。本地调用明确提供scalar5、1.0、'cosine'，避免错误但没有修改源码；真实默认失败记录在重载校验。不能在笔记写“默认可直接运行”。

更重要的是`Dynamics.sample`把prime IDs与待生成IDs拼起来，重排为时间网格后又`[:, :-1]`，丢弃最后待生成帧。本次单帧prime条件下，每一轮网络只收到同样的first-frame IDs；生成的token没有反馈到下一轮网络输入。内部token_emb hook实测五次输入均[1,16]且完全相同，1步与5步最终token IDs相同。代码虽计算cosine mask schedule/低confidence重遮挡，不能据循环次数称为已验证的完整MaskGIT逐轮补全。

此结论限定**已固定快照、本地单步路径**；不是对原论文或所有GenieRedux分支的判断。本次没有静默“修好”source或用自写sampler替换后称为原模块成功。该路径的final temperature乘剩余步数比例为0，只留下约1e−10的Gumbel噪声，基本argmax；paper原temperature2的random generation不能由本结果验证。按源码的所有位置预测结果覆盖旧video_token_ids，也不是严格保留已确定位置的标准MaskGIT方式。

本地`one_step`只给真实/上一步生成的图像与用户码，编码prime、lookup action、sample、将prime+predicted IDs解码，只取下一帧。六步rollout依次用户ID0…5，首帧之外不提供真实video，故没有future-image leakage；每步只保留一帧历史重新编码，区别于原16帧上下文。user ID是latent code，不直接用于更新数据生成位置，没有把已知真实方向伪装成学会的控制器。

## 13. 核心函数、默认参数与状态变化

|对象/所属|输入、默认值与返回|调用位置和状态变化|
|---|---|---|
|`STViViT.__init__`|原默认dim512/codebook1024/image64/patch4/temporal1/blocks8/codebookdim32|创建encoder/decoder、VQ、pixel heads；小实验显式缩小|
|`STViViT.encode(tokens)`|[B,T,h,w,d]→同shape|spatial bias与ST encoder，无optimizer更新|
|`STViViT.decode(tokens)`|3D packed或5Dfeatures→[B,C,T,H,W]|decoder先ts，再first/rest pixel heads，无sigmoid|
|`Tokenizer.forward`|videos=None/mask=None；return_recons等默认False；step0/log_every50|默认loss，return_recons返回(loss,video)，return_only_codebook_ids返回整数；默认videos=None会assert|
|`LAM.forward`|videos必需/action_ids=None/mask=None；return_tokens_only、return_only_codebook_ids等False|默认loss；可返回空间复制tokens、IDs、recon；提供action_ids仍先运行encoder/VQ，不是精简交互API|
|`LAM.get_codes_from_indices(indices)`|整数[B,T]→[B,T,dim]|lookup16维codebook再project_out32维，交互无需encoder|
|`VectorQuantize.forward`|x、indices=None/mask=None/freeze_codebook=False等|返回quantized、IDs、VQ loss；训练启用STE/q+commit，非EMA参数由AdamW更新|
|`MaskGIT.forward`|IDs 2D或4D、actions、prompt=None/video_mask=None等|返回[B,T×N,K]logits或return_embeds；source sampling显式forward调用|
|`Dynamics.forward`|IDs/actions默认None需提供；return_logits/return_token_ids=False|mask RNG有变化，默认返回CE；可返回loss/logits|
|`Dynamics.sample`|prime/actions/num_tokens/patch_shape必需；steps等默认None|生成IDs；默认tuple存在错误；本地显式scalar；更改局部采样tensor和RNG，无optimizer更新|
|`dataset(n,seed)`本地函数|n条5帧→videos、真实标签|generator按seed生成；训练丢弃标签|
|`evaluate(tok,lam,dyn,videos,labels)`|返回metrics和fixture|设eval、seed391、oracle latent统计/用户交互rollout；无梯度|
|`vq_weight_ablation.main()`|无参数，权重.01|复用已训tokenizer，重建同初始化LAM/dynamics；仅LAM损失权重改变|
|`verify_and_plot.main()`|加载两套state/fixture|strict重载，hook诊断、额外梯度检查，写report/figures|

`@torch.no_grad`只禁止计算图记录，不等于传入模型参数已永久冻结；训练后另用requires_grad_(False)冻结tokenizer/LAM以训练dynamics。dynamics只看到detach后的codes，其CE不反传LAM；LAM自己的重建/VQ才是它的学习信号。`.eval()`对VQ切换辅助loss/STE路径，这里不能只解释成dropout关闭。optimizer仅在zero_grad/backward/step之间更新参数，保存的state_dict只含模型状态，没有optimizer恢复入口；复跑是从随机初始化重新训练，不支持中断续训。

缩小模型独立配置与原环境文件并存，没有升级到Linux/CUDA Torch2.8。直接使用已有pi0共享环境里的PyTorch2.2.0+cpu、einops.8、beartype.19、numpy1.26.4、matplotlib3.10.7；不等于把π0算法用于Genie。无checkpoint下载、API/key、外部视频数据、GPU或真实环境动作命令。


## 14. 小 demo 的实测结果与一次 VQ 权重对照

直接导入保存的第三方Tokenizer、LatentActionModel、MaskGIT/Dynamics，源码未改。参数分别 **56,842 / 62,258 / 37,314**，总156,414；没有原200M/300M/10.1B。CPU四线程，torch2.2.0+cpu、float32，seed97。初轮训练tokenizer1201、LAM1801、dynamics801 updates，约116.7秒；每步batch16，AdamW lr=.001、weight_decay0、gradient norm clip1，无原warmup/cosine/EMA。checkpoint没有optimizer状态，重新训练不保证跨设备bitwise相同。

第一轮LAM外部VQ权重.25，256个测试transition全部使用code6。重建loss已经下降却没有动作区分，这是明确的codebook collapse；不能拿pixel MSE变小证明无监督动作发现成功。保存第一轮全部权重/fixture/日志后，对照只将LAM外部VQ权重降到.01；tokenizer用原固定权重，LAM/dynamics回到同随机初始化、按原tokenizer minibatch RNG推进后使用同样minibatch序列，各再1801/801updates。其余结构、数据、内部commitment、optimizer、训练量不变，无真实动作标签加入；对照耗时63.6秒。

|指标（64测试视频/256transitions）|VQ .25第一轮|LAM VQ .01对照|解释|
|---|---|---|---|
|tokenizer pixel MSE|0.000347693|0.000347693|同一冻结tokenizer|
|LAM teacher-forced pixel MSE|0.009008363|0.001440435|下降约84.0%，不是自由rollout|
|动作码histogram|[0, 0, 0, 0, 0, 0, 256, 0]|[56, 51, 34, 9, 35, 34, 0, 37]|由1个使用码到7个；code6在对照未使用|
|事后direction purity|0.296875|0.566406|29.69%→56.64%，非独立动作分类泛化准确率|
|dynamics masked CE|1.061256|0.896374|同seed mask，在两个模型的不同动作条件下比较|
|推断latent条件的一步pixel MSE|0.013677403|0.010424165|推断ID时看过真实下一帧；oracle latent条件|
|将ID改为(k+1)%8的一步MSE|0.013512886|0.013544979|循环替代码，不是paper categorical random|
|toy ΔPSNR，t=1|-0.120420dB|1.614594dB|对照变正，不能与原t=4直接比|
|同prompt、八user ID输出最大差|0.273679|0.412985|有响应差异，不保证八个稳定物理动作|

direction purity定义为每个码在测试transition的四种方向计数中取最大，再把这些最大数求和除256。它允许同一方向分散到多个码，且“选最多方向”和“计算比例”用同一测试集，所以是描述性关联统计，不是训练映射后在另一个保留集评测的分类accuracy。对照code0的方向计数[5,3,25,23]，down/up仍混在一起；code1为[15,36,0,0]偏left但仍包含right，不能为按钮固定命名并说跨起点一致。

toy ΔPSNR逐样本计算$10\log_{10}(MSE_{different}/MSE_{inferred})$再取64样本平均，clamp最小1e−12防log0。**不等于先对所有样本平均MSE再求log**；主表平均MSE之比与报告dB不必相等。这里只t=1、单步、deterministic different ID，原论文t=4、多步、categorical随机动作，所以它是机制诊断，不能叫Genie ΔPSNR复现结果。第一轮均−.12dB，说明正确推断的坍塌码无优势；对照约+1.61dB支持动作条件开始承载信息，但语义一致性仍不足。

![训练曲线](<G:/具身/论文阅读/Genie -  Generative Interactive Environments/复现/outputs/训练曲线.png>)

图为第一轮，tokenizer/LAM总损失含pixel+VQ，dynamics是masked CE，三者数值不可当同一单位比较。LAM稳定低loss却全用code6，展示为何还必须检查usage与control。第二轮训练曲线原数据独立保存在VQ对照训练曲线.json，没有覆盖第一次结果。

![两次动作码使用](<G:/具身/论文阅读/Genie -  Generative Interactive Environments/复现/outputs/动作码使用率.png>)

左.25全部code6，右.01分散到七个码。均匀usage本身也不是正确动作语义的充分证据，应结合direction confusion、生成效果与跨环境测试。

![重建和生成](<G:/具身/论文阅读/Genie -  Generative Interactive Environments/复现/outputs/重建与生成.png>)

此图显示**.01对照**，固定测试数组第一条，未挑最好样本。target、tokenizer重建、LAM teacher forcing、用户ID0/1/2/3的生成rollout四行；最后一行用户ID序列不是target真实方向，因此不能凭轨迹不同说预测错误或成功。teacher forcing行每步提供真实过去观察与从未来推断的码，不代表自由rollout同样清晰。所有显示vmin0/vmax1只限制色阶，没有裁剪采样tensor或评价输入。

![八个用户动作码对照](<G:/具身/论文阅读/Genie -  Generative Interactive Environments/复现/outputs/用户动作码对照.png>)

同一个prompt下.01对照的八个ID生成画面有差异，但亮点会模糊、偏移，有些码近似同方向，未使用code6属于训练外码。交互输入仅提示图+user ID，查表后由dynamics生成；没有用真实位置/方向偷偷更新图像。

## 15. 重载、因果和采样问题的验证

第一轮与.01对照都在独立Python进程strict重载。所有fixture tensor、重建、动作IDs、一步/替代码生成、user-code输出与六步rollout重算，最大差均约 **2.98×10⁻⁷**（阈值1e−6），整数IDs一致；不是宣称bitwise全部0。参数/图像路径与源码hash已记录。LAM action projection、codebook、pixel head分别实测gradient norm约0.00079956、0.00198295、0.01913505，有限非零。

因果测试在第一轮把最后future frame改为随机，之前visual IDs和LAM action IDs不变，记录passed。不能把它扩大为所有采样路径均无信息泄漏。采样内部Embedding hook实际5次且输入相同、1步/5步token相同；default tuple报`TypeError: 'tuple' object cannot be interpreted as an integer`。这两点限制了对完整MaskGIT能力的判断，详见前节，源码保持原样。

失败记录：最初smoke test把LAM 2×2复制码直接传dynamics 4×4而报expand shape错误，改为lookup后singleton广播；第一次采样诊断在maskgit对象注册Module hook未收到调用，因为源码使用`.forward`，改在实际token_emb调用处采集，重新运行通过。二者为本地调用/观察修正，未改第三方模型源码；同时保留default tuple与collapse等实际问题，没有把它们隐藏成全成功。

[复跑说明](<G:/具身/论文阅读/Genie -  Generative Interactive Environments/复现/README.md>) · [核心训练](<G:/具身/论文阅读/Genie -  Generative Interactive Environments/复现/scripts/genie_core_demo.py>) · [VQ权重对照](<G:/具身/论文阅读/Genie -  Generative Interactive Environments/复现/scripts/vq_weight_ablation.py>) · [独立重载/绘图](<G:/具身/论文阅读/Genie -  Generative Interactive Environments/复现/scripts/verify_and_plot.py>) · [原始结果](<G:/具身/论文阅读/Genie -  Generative Interactive Environments/复现/outputs/结果.json>) · [对照结果](<G:/具身/论文阅读/Genie -  Generative Interactive Environments/复现/outputs/VQ权重对照.json>) · [重载/采样检查](<G:/具身/论文阅读/Genie -  Generative Interactive Environments/复现/outputs/权重重载校验.json>).

## 16. 掌握要求、Paper ↔ Code、复现状态与未解决问题

【必须会】video/action两个codebook的索引、宽度与含义；LAM训练用未来而交互不用encoder；causal空间/时间attention；VQ/STE与codebook collapse；action stopgrad与dynamics监督；teacher forcing、oracle latent与user control区分；latent ID不等于真实动作；参数/画质/可控性/交互速度分别判断。

【建议会】q_loss/commitment与外部VQ权重的区别；QK norm与真实ST block的FFW差异；mask input/label对齐与CE reduction；MaskGIT confidence/refinement真实输入核对；ΔPSNR的aggregation与公平对照；单因素训练实验；CoinRun少量真实动作校准；数据质量标签与动作标签区别。

【可以暂时不掌握】11B多TPU训练、完整采集过滤、AutoExplore/RL探索、全部RetroAct环境和真实机器人适配。先能从一段视频追踪每个码怎样被产生、监督和使用。

|Paper ↔ Code|本地实际对象|对应范围|
|---|---|---|
|Figure3整体|genie_core_demo.train → Tokenizer/LAM/Dynamics|第三方缩小顺序训练，非原joint recipe|
|Figure4 ST attention|STViViT.encode/decode、STBlock、Attention|真实运行causal；FFW数量与原文不同|
|Figure5 LAM|LatentActionModel.forward/get_codes_from_indices|原像素输入→transition码；两次无动作标签训练，语义不稳定|
|通用VQ说明|VectorQuantize、CosineSimCodebook|learnable非EMA、STE/q+commit实际代码；不是原Genie量化源码|
|Figure6 tokenizer|Tokenizer.forward、STViViT.decode_from_codebook_indices|实际visual codes/reconstruction；32×16缩小码本|
|Figure7 dynamics|Dynamics.forward → MaskGIT.forward|masked CE/stopgrad实际；mask范围和默认值有差异|
|Figure8 user交互|action_codes → one_step → Dynamics.sample → decode|真实查码/生成6步；只保留一帧，采样refinement输入缺陷|
|Tables2/3、scaling|原文实验阅读|没有复跑FVD、机器人视频、原scaling|
|ΔPSNR|evaluate的t1/inferred-vs-shifted-code|toy诊断，非原t4/random protocol|
|Figure15/附录E|真实动作字典D、BC policy|源码本轮未运行；无200labels迁移复现|
|附录F CoinRun|原文case-study配置|未采集10Mtransitions/训练原512dim模型|

**状态：部分成功（第三方核心小配置机制验证），完整原论文未复现。** tokenizer/LAM/dynamics训练、未来扰动因果检查、用户码生成、权重重载与VQ对照完成；第一轮codebook collapse，对照部分缓解，但潜在动作不稳定，实际快照的MaskGIT单步路径存在反馈缺陷与default bug。本地没有证据证明原版Genie可控性、跨域迁移或长期物理一致性。

未解决：完整原官方源码/权重未找到；第三方量化/FFW/sigmoid、mask和采样与原方法差异仍在；.01只做一次seed对照，未证明一般解决VQ坍塌；code语义可能掺入位置/背景且方向purity不足；相同测试集事后多数映射不构成泛化绑定；没有原t4随机动作/FVD评估、真实平台游戏/CoinRun、专家BC和真实动作校准；生成长rollout有模糊/漂移，未验证超过一帧历史的记忆或prompt跨域能力。原文和固定代码的可定位行为置信度较高，toy运行支持机制及已发现问题，原模型能力与完整复现结论均未取得。
