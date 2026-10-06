## 8. 原论文训练预算和附录 F：小例子也有真实成本

本篇固定阅读arxiv2402.15391v1，共27页，SHA-256见来源文件；不是后续Genie2/3、Genie Envisioner或商业交互产品。主文§3/PDF6给video tokenizer约200M参数、LAM约300M、dynamics约10.1B，合计约10.6B，常四舍五入为11B。Table5/PDF22又列Platformers LAM encoder/decoder各20layers、d_model1024、16heads、8codes、latent_dim32、patch16；Table7列tokenizer encoder12layers/512/8heads、decoder20layers/1024/16heads、1024codes、patch4、latent32。参数量与表格配置只在其原训练上下文解释，不能把小demo计数拿去证明原模型大小。

主配置使用16frames、10FPS，视频原分辨率160×90；图像帧时间、空间patch大小和离散词表是不同轴。dynamics用bfloat16、QK normalization；主生成每帧25次MaskGIT、temperature2。温度不是环境动作强度，也不是帧率。本地实验float32、8×8、5frames，时间以合成离散步计，没有10FPS或真实渲染耗时的对应。

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
