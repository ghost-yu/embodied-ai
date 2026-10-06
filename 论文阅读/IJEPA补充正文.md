## 8. Multi-block collator 的真实采样与边界【必须会】

`MaskCollator`的类默认context/target scale均(.2,.8)、aspect(.3,3)、nenc1/npred2/min_keep4；这些不是ViT-H配置值。配置文件覆盖为context(.85,1)、target(.15,.2)、aspect(.75,1.5)、4targets、min_keep10。本次8×8 patch网格显式min_keep4以便缩小实验，保持四target/一context，不能说完全照搬原配置。

输入图像先通过`default_collate`组成[B,C,H,W]。`step`在multiprocessing.Value锁下递增计数，初始−1→0；这个计数seed创建局部generator采block size。**block位置用全局torch.randint，不用该generator**，因此只恢复_itr_counter不能恢复完整随机mask。_sample_block_size用同一个随机u同时插值area scale与aspect，二者在这个函数内相关，并非两个独立uniform draws。一个batch共用target尺寸和context尺寸，图内四targets的坐标各自随机。

令网格height/width=8、面积scale=.1875、aspect=1，则max_keep=int(64×.1875)=12，h=w=round(√12)=3，实际9patch并不恰好等于目标12。离散round、长宽约束、context删除以及batch truncation都会改变实际比例。代码while h>=grid_height时将h减1，不允许块高覆盖全部网格；top用randint(0,grid_height−h)，上界exclusive，因此少了最后一个合法top-left位置。对于此固定代码，矩形不会覆盖最底/最右patch行列，不能按理想“所有位置均匀可采”解释。

target先采矩形，返回mask索引与其complement；context再与各target complement相乘。valid条件是`len(mask) > min_keep`，严格大于而非大于等于。若20次找不到足够context，tries增加，会逐步减少acceptable_regions数量，可能放松部分target排除约束。**allow_overlap=False并不构成所有输入下绝不重叠的强保证**。本次501batch检测context/target索引交集为0，但没有触发并验证退化高遮挡配置；初稿关于不重叠的描述是设计目标，需要这个实现例外。

各样本context保留数不同，collator最后将全batch/同类mask截断到最小长度，形成等长张量；target尺寸相同通常等长，但实现仍计算全局min。截断`cm[:min_keep]`按torch.nonzero得到的顺序保留较前索引，不是随机重新挑点。target不同blocks可以重叠并重复监督某位置，context排除的是完整矩形，而不仅最后截断后的target索引。mask是int64 patch index，非Bool像素mask。

## 9. 从 encoder 到 predictor 的对象与样本顺序

`VisionTransformer`先Conv2d(kernel=stride=patch)将图像patchify，得到全部patch tokens，加固定2D sin-cos位置表，再`apply_masks` gather可见context，最后才进Transformer。原图仍传入Conv2d，**只有可见patch token进入context attention**；不是在输入像素上把target涂黑然后把所有token送入encoder。patch卷积不重叠且不含跨patch操作，本次修改target对应像素时context输出完全不变。若以后换重叠卷积/全图预处理，不能沿用这个泄漏结论。

没有CLS token：encoder返回每patch特征[B,N_c,D]；full-image encoder返回[B,N,D]。全图representation通常平均patch特征，用于下游head。初始化有norm/Linear/Conv权重规则，并对每layer attention output projection和MLP末层按√(2×layer_id)缩放。本次使用相同官方类构造小配置，没有更改初始化方法。

`apply_masks(x,masks)`将每个[B,N_k]索引扩成[B,N_k,D]后torch.gather沿patch轴取元素，将不同mask结果沿**batch轴**cat。例如B2、M4时次序为target0的sample0/1，target1的sample0/1，依此；不是sample0的四targets后sample1的四targets。`repeat_interleave_batch(x,B,repeat)`按原B大小切组，再重复整个组；名字不是torch.repeat_interleave逐个样本的行为。

本次B2、C2contexts、M4targets的身份算例：原图0用patch值0…63，原图1用100…163，每target取图0的i、图1的i+1。最终序列为[0,101,0,101,1,102,1,102,2,103,2,103,3,104,3,104]。这对应(target,context,sample)三层顺序。predictor `x.repeat(len(masks),1,1)`按target复制全部context batch，与target feature的repeat顺序匹配；尺寸相同仍需检查身份，不能用reshape猜配对。

predictor开始Linear D_enc→D_pred；其context位置表先按masks_x gather，target位置表按masks_pred gather后依context数repeat，learnable mask_token与target位置表相加。拼context与target slots，进入无因果mask的Transformer blocks，再norm、切掉context、Linear D_pred→D_enc。target slots可以互相attention，但只含位置和可学习共享token，没有真实target特征。`x += pos`这里x已是Linear的新输出，不是在原context tensor上插入target内容。

固定位置表是requires_grad=False的Parameter，仍进state_dict和encoder/target参数zip。EMA会遍历它，但两个表从同一deepcopy开始且不会被optimizer梯度更新，所以保持一致；“Parameter”不等于必定可训练。官方helper把bias和一维参数放入WD_exclude组，其他矩阵用weight decay；列表推导产生parameter generator，AdamW构造时消费。它还包含固定位置Parameter，但requires_grad=False无梯度，optimizer不会按有梯度参数那样更新。

同分辨率路径下位置表直接返回；这个快照`interpolate_pos_encoding`仍含把第0位置当class_emb、N减1的旧逻辑，异分辨率插值要额外核查，**本次仅64×64等分辨率，不宣称该分支已验证**。square-grid sin-cos要求维度/网格适配，不拿这个类直接推广非方形输入。

## 10. 训练 loss、optimizer、EMA 和 checkpoint

本地沿原`src/train.py`顺序运行：target完整图像→末维F.layer_norm→target gather/repeat，在no_grad下；context masked encoder→predictor；`F.smooth_l1_loss`默认beta1/mean→官方AllReduce.apply；backward→optimizer.step→target EMA。单进程未初始化distributed，AllReduce早返回原tensor，未执行多GPU梯度聚合。没有运行DDP/AMP/scaler，float32 CPU；不宣称完整train入口跑通。

Smooth-L1的mean除以B×M×C×N_p×D_enc。原文每block平方L2与保存源码损失的区别已在前节说明。本次final MSE恰好约为smooth-L1两倍，因为这个批次所有相关误差落在二次区间；这不意味着两个loss一般只差常数，大于beta的误差进入线性区。

使用官方`helper.init_opt`及WarmupCosineSchedule/CosineWDSchedule。配置将一个“epoch”设为一次update以方便总501步：warmup20updates，start_lr1e−4、ref_lr1e−3、final_lr1e−5；wd .04→.1、ipe_scale1。原配置是300epochs/40epoch warmup、wd .04→.4，不能把本地501updates叫501个ImageNet epoch。scheduler.step会先将计数加1，所以第一update lr=.000145，而不是start_lr=.0001。wd分组bias/1D始终0，非排除组才随cosine变化。

EMA m=.996+step×.004/501，从.996逐渐接近1，最后约.999992；target新参数为m×旧teacher+(1−m)×刚optimizer更新后的context。本次每步对第一个trainable matrix（patch projection weight）用独立算式核对，最大差0；其他参数由同一源码式遍历更新，没有声称逐tensor独立校验全部。m≈1使最后teacher几乎固定，不能解释成target不会发生任何漂移。

checkpoint包含encoder、predictor和target三个state_dict；独立重载构造同配置，strict检查所有key/shape，并用固定16图像/masks重新输出prediction/target。没有optimizer、scheduler、mask-counter/RNG的保存恢复，故本demo支持推理重载，不支持断点训练恢复。官方full `load_checkpoint`路径会另处理optimizer/scaler/epoch；本轮未测试它。

|核心对象|参数默认与返回|状态、调用位置与边界|
|---|---|---|
|`MaskCollator.__init__`|input224/patch16、nenc1/npred2/min_keep4/overlapFalse|建立计数器/网格；本地覆盖为64/8、四target|
|`MaskCollator.__call__(batch)`|图像list→batch、context索引list、target索引list|递增计数、采RNG、裁截长度；可能放松排除|
|`apply_masks(x,masks)`|[B,N,D]与索引list→[B×mask数,N_keep,D]|gather/cat，无参数/RNG更新|
|`repeat_interleave_batch(x,B,repeat)`|B为原batch数；repeat为context数|按B组重复，不逐样本重复|
|`VisionTransformer.forward(x,masks=None)`|无mask为全图feature，mask时为可见patchfeature|Transformer参数由loss梯度更新；位置表固定|
|`VisionTransformerPredictor.forward(x,masks_x,masks)`|两个mask必需，缺失assert；返回targetfeatures|learnable mask_token，固定位置表；不含未来像素|
|`helper.init_opt`|wd/final_wd默认1e−6、final_lr0、bfloatFalse、ipe_scale1.25|返回optimizer/scaler/lr_scheduler/wd_scheduler；小demo显式覆盖|
|`WarmupCosineSchedule.step()`|无参数→新lr scalar|改_step和optimizer各group的lr|
|`CosineWDSchedule.step()`|无参数→新wd scalar|改_step和非WD_exclude的weight_decay|
|`target_features`本地函数|target、images、两类mask→h|no_grad；模仿官方target分支，非新模型|
|`evaluate`本地函数|三个模型/图像/masks→metrics、z、h|无梯度，仅输出loss/std诊断|
|`train`本地函数|无参数→权重/fixture/metrics文件|训练RNG/optimizer/teacher状态实际改变|

`with torch.no_grad()`阻止计算图，target的requires_grad=False进一步表明不由loss优化；`detach`断开更新语句的autograd；`mul_/add_`在原参数上修改。三者各有作用。`copy.deepcopy`提供独立teacher对象，不是`target=encoder`引用共享；若共享，就不再是慢teacher。此原始I-JEPA不输入语言/动作、没有text encoder/robot head，也没有生成像素decoder。

## 11. 附录消融与像素可视化：为什么不只看预训练 loss

附录C/PDF14–15，ViT-B/16、300epoch、1%ImageNet frozen linear probe：Table8 target(.075,.2)为19.2，(.15,.2)为54.2，提高35.0pp；但放大到(.2,.3)降至33.6，不是目标块越大越好。Table9 context(.40,1)31.2与(.85,1)54.2差23.0pp；Table10 target数1/2/3/4为9.0/22.0/48.5/54.2，4比1高45.2pp。块信息量/任务次数变化，不能在不同mask设置下将loss大小直接对应语义准确率。

Table11同ViT-H/16/300ep：output masking67.3%，input masking56.1%，高11.2pp。它直接支持完整图像target encoding再mask输出的重要性；输入target只裁块分别编码不是等价替代。Table12 ViT-L/16/500ep：predictor6层64.0%，12层66.9%，高2.9pp。Table14 width384为70.7，1024为68.4（600ep），窄瓶颈高2.3pp；原段落一处写386而表列384，按表和源码配置384解释，保留排印差异。

Table13 wd .04→.4 vs固定.05（同ViT-L/16/600ep）：1%fine-tune69.4→70.7提高1.3pp，但full linear evaluation77.8→76.4下降1.4pp。不同下游protocol的偏好可能相反，不能简单宣布较低wd更好。论文称线性增加wd，而保存官方源码CosineWDSchedule用cosine形式，也属于论文描述/实现差别。本次跑官方scheduler，不改成自选linear后称完全一致。

附录D Table15 full ImageNet fine-tune：I-JEPA87.1、MAE87.8，低.7pp；I-JEPA pretrain300 vs MAE1600epoch。I-JEPA并非在所有评估都领先。50epoch微调、lr1e−4、batch528、mixup.8/cutmix1/drop-path.25、wd.04、layer-decay.75、RandAugment均为另一路下游训练，不是预训练“无view augment”声明的反例或本地配置。

附录E RCDM另训diffusion decoder300kiterations，从representation s_x与noisy image x+ϵ预测ϵ；Figure6用predictor输出平均池化、Figures7/8用encoder平均池化。相同表征配多noise后共同结构反映保留信息，变化细节提示未保留信息；但生成decoder也有自身先验，不能只看几张图作为严格信息论证明。原I-JEPA主体没有解码器权重。本地mask图是输入可见性示意，**不是RCDM生成效果**。
