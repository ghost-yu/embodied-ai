import sys,json,re,hashlib,importlib.util,ast
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'G:\具身\论文阅读');P=ROOT/'π0 -  A Vision-Language-Action Flow Model for General Robot Control';R=P/'复现';O=R/'outputs';REPO=P/'代码'/'openpi'
r=json.loads((O/'结果.json').read_text(encoding='utf-8'));reload=json.loads((O/'参数重载校验.json').read_text(encoding='utf-8'))
b=ROOT/'精读正文'/'pi0.md';s=b.read_text(encoding='utf-8-sig')
start=s.index('## 7. 掌握层次与未解决问题');end=s.index('## 7. 从 Pi0Config',start);s=s[:start]+s[end:]
s=s.replace('最大输出差约1.7881e−6',f"最大输出差约{r['cache_vs_full_max_error']:.7g}")
s += f'''

## 12. 小 demo 的配置、验证与实际结果

本次运行保存版本的 **JAX Pi0**，没有重新实现flow模型。使用官方Gemma的dummy变体，动作接口保持action_dim32、horizon50，pi05=False；float32、CPU。Pi0构造器硬编码So400m/14视觉配置，demo支持文件临时替换其Module工厂的**variant构造参数**为官方支持的mu/32，构造后在finally恢复工厂。返回的仍是官方SigLIP实现，Pi0.compute_loss、sample_actions、Gemma Attention均保持原源码。此构造适配器是demo代码，不能说默认Pi0Config自行支持任意vision variant。

参数共 **{r['parameter_count']:,}**，大部分来自保留257,152词表的64维语言embedding；模型随机初始化，没有PaliGemma/SigLIP预训练语义。Python3.11.0、JAX0.5.3、Flax0.10.2、Optax0.2.4。训练Adam lr=.002、501次更新，NNX/Optax实际更新全部参数，没有冻结前端。train=False只用于关闭数据增强。

合成观测四个z=[−.6,−.2,.2,.6]；每个相机为对应z的常色RGB，state第一维也编码z，其余为0；prompt ID固定[2,3,0,0]，只有前两项有效，不是真实语言指令。七个动作维目标分别z乘[1,.5,−.3,.2,.1,0,.2]，重复50个位置，其余目标为0。由于图像与state同时含z，该任务不能单独证明模型必须使用视觉或语言，更不能证明指令理解。

|验证项|实际结果|解释与范围|
|---|---|---|
|官方模型构造|Pi0+Gemma+SigLIP成功|配置缩小且随机权重；非默认3.3B模型|
|loss与手工重建|同一key的noise/time/前向结果一致；loss`[4,50]`|flow目标为noise−action，函数尚未平均H|
|attention结构|prefix不读suffix；state不读action；action彼此双向|mask与实际代码组结构一致|
|完整前向与cache前向|最大差{r['cache_vs_full_max_error']:.8g}|float32容差内一致|
|梯度|vision/LLM/projections记录均有限非零|组件有梯度，不证明条件信息被正确利用|
|固定新key的flow MSE|{r['initial_new_key_flow_mse']:.8f} → {r['final_same_new_key_flow_mse']:.8f}|评估时重新采time/noise，但条件仍是训练四条件|
|固定noise的sample MSE|{r['initial_fixed_noise_sample_mse']:.8f} → {r['final_fixed_noise_sample_mse']:.8f}|官方10步缓存Euler输出与目标误差下降|
|换noise、同条件的sample MSE|{r['new_noise_same_conditions_sample_mse']:.8f}|不是保留任务/真实数据泛化|
|独立进程参数重载|最大差{reload['max_abs_error']:.8g}|参数schema加载通过，采样在规定容差内一致|

训练最后一步mean flow MSE约0.03156，而固定新key评估约0.21798；time/noise不同造成两者差异，不能只选择较小训练数值当最终性能。首轮vision参数梯度norm约84008.69，虽然有限，说明这个随机初始化与常色输入设置有很大的梯度尺度；demo未加梯度裁剪，Adam仍将loss降低，但不能据此称实际大模型训练稳定。

101次试跑时sample MSE仍约0.4666，随后增加到501次，结果明显改善。右图中有偏离±.6目标线的位置，残差仍存在，没有训练到所有动作精确相等。flow loss与sample MSE各自有意义：前者约束向量场，后者测十步积分后的动作，不应混为同一个指标。

![官方缩小模型的合成训练与采样](<{(O/'demo曲线.png').as_posix()}>)

图右横轴为动作块内部50个位置，而不是10个积分步；所有数值为合成动作坐标，没有米/弧度的物理单位。模型返回全部32维，demo没有假设未使用维会像RDT mask那样精确归零。没有环境reset/step、reward/done、控制频率实测或真机指令。

[运行说明](<{(R/'README.md').as_posix()}>)；[主脚本](<{(R/'scripts'/'pi0_core_demo.py').as_posix()}>)；[构造适配器](<{(R/'scripts'/'pi0_demo_support.py').as_posix()}>)；[重载脚本](<{(R/'scripts'/'reload_and_plot.py').as_posix()}>)；[原始结果](<{(O/'结果.json').as_posix()}>). 状态为**部分成功**：官方缩小配置的loss、梯度、训练、mask/cache和采样已运行；完整论文未复现。

## 13. 掌握层次与 Paper ↔ Code

**必须掌握**：PaliGemma与action expert的关系；连续token与词ID的区别；动作块50位置与flow时间；论文τ/源码s的换元、target符号、负dt积分；prefix/state/action三个attention组；标准化、delta/absolute与padding的接口。

**第二遍掌握**：Beta时间采样；专家不同宽度却共享head形状；GQA与RoPE；缓存与完整forward的一致性条件；train布尔参数、模型状态和梯度更新的区别；任务进度与成功率/高层指令的实验边界。

**可以暂时不掌握**：完整JAX FSDP分片、3.3B训练、全部机器人数据源与设备驱动。PyTorch变体和π₀.5/FAST可在理解原π₀后再分别读。

|Paper ↔ Code|固定快照中的定位|证据与边界|
|---|---|---|
|§IV显式flow路径与target|Pi0.compute_loss|代码反向时间，实际同key重建loss通过|
|Euler十步|Pi0.sample_actions|官方负dt/cache采样实际执行|
|Figure3条件输入|embed_prefix、embed_suffix、siglip.Module|真实官方模块小配置执行；视觉variant有demo适配器|
|双专家共享attention|gemma.Module/Attention|独立投影→拼QKV→按专家拆分；缓存一致性验证|
|prefix/suffix分组|make_attn_mask、positions计算|布尔group结构实际检查|
|完整训练loss汇总|scripts/train.py.loss_fn|静态核对外层mean及trainable_filter；没有执行数据/分布式全入口|
|机器人动作变换|transforms.DeltaActions/AbsoluteActions/Normalize/Unnormalize/PadStatesAndActions|静态说明与算例；未运行真实机器人数据链|
|原π₀与π₀.5配置|Pi0Config、training/config.py|pi05=False锁定；不把后续分支当原论文|
|Figure7/9任务比较|原论文与实验协议|只解释来源图，不补造柱值；未重新评测|
|完整3.3B预训练模型|checkpoint/weight loader入口|本次未下载或执行|
'''
b.write_text(s,encoding='utf-8')
(R/'status.json').write_text(json.dumps({'status':'部分成功','scope':'官方 JAX Pi0 缩小配置：flow/mask/cache、501更新、10步采样与参数重载；含视觉构造适配器，完整论文未复现','date':'2026-10-03','full_paper_reproduction':'未复现'},ensure_ascii=False,indent=2),encoding='utf-8')
(R/'requirements.txt').write_text('jax==0.5.3\njaxlib==0.5.3\nflax==0.10.2\noptax==0.2.4\norbax-checkpoint==0.11.13\nnumpy==1.26.4\njaxtyping==0.2.36\nbeartype==0.19.0\neinops==0.8.0\naugmax==0.3.4\nml_collections==1.0.0\nmatplotlib==3.10.7\ntorch==2.2.0\ntransformers==4.53.2\nsafetensors==0.5.3\nsentencepiece==0.2.0\n',encoding='utf-8')
readme=f'''# π₀ 官方 JAX 核心小 demo

状态：**部分成功**，完整论文未复现。源码commit `{r['repo_commit']}`，原文与代码说明见 [解释.md](<{(P/'解释.md').as_posix()}>).

## 环境与资源边界

Windows、Python3.11.0、CPU、float32；JAX0.5.3/JAXlib0.5.3、Flax0.10.2、Optax0.2.4、NumPy1.26.4。独立venv为 `G:\\具身\\论文阅读\\.venvs\\pi0`；pip check通过，完整实际版本在requirements-lock.txt。没有CUDA训练、API/key、3.3B checkpoint或真实数据下载；16,936,448随机参数，小模型参数msgpack约68MB量级。

仅安装此JAX demo用到的依赖，没有执行完整uv sync。仓库声明torch2.7.1和jax[cuda12]；本CPU demo为JAX0.5.3，torch2.2.0仅用于保存仓库JAX文件导入时的类型/PyTorch接口依赖，没有运行PyTorch PI0Pytorch。ml-dtypes/tensorstore等实际解析版本也不完全等于仓库uv override，详见lock。这些偏差不能支持完整PyTorch/分布式环境兼容结论。

## 复跑

工作目录 `G:\\具身`。PowerShell的&执行指定解释器，路径有空格须保持引号。主脚本训练并覆盖小demo outputs；第二条在新进程重新构建、加载msgpack参数，运行官方缓存采样并绘图。

```powershell
& 'G:\\具身\\论文阅读\\.venvs\\pi0\\Scripts\\python.exe' '{R/'scripts'/'pi0_core_demo.py'}'
& 'G:\\具身\\论文阅读\\.venvs\\pi0\\Scripts\\python.exe' '{R/'scripts'/'reload_and_plot.py'}'
```

scripts/pi0_demo_support.py仅调整构造配置：Gemma两个专家使用官方dummy、原始pi05=False、dtype=float32、prompt4、action_dim32、horizon50；SigLIP Module的硬编码So400m/14参数在构造期间改为官方mu/32，finally恢复factory。不替换前向/loss/attention/sampler，也不改官方源码。此适配器不是官方默认入口；不要用本msgpack配置加载3.3B权重。

## 数据与训练

四个z条件：常色合成RGB（真实通过官方视觉encoder）、state第一维z、固定整数prompt（非真实tokenizer结果）。七个目标维随z变化，全部50步重复，其余目标零；没有动作物理单位。图像与state同时携带z，任务不证明对视觉的必要依赖或指令理解。Adam lr=.002、501更新；train=False关闭图像augmentation，仍实际求梯度/更新参数。

真实调用链：Pi0Config.create → Pi0/Gemma/SigLIP → embed_prefix/embed_suffix → compute_loss；nnx.value_and_grad → NNX/Optax optimizer；sample_actions → prefix cache → 10次suffix/Euler。完整scripts/train.py/data_loader未运行。

## 检查与结果

公式使用同key的noise/time重建，loss shape `[4,50]`与官方一致。prefix `[4,151,64]`、suffix `[4,51,64]`、full mask `[4,202,202]`。group规则检查通过：prefix不读suffix，state不读action，动作彼此双向；完整与缓存forward最大差{r['cache_vs_full_max_error']:.8g}。

固定新key flow MSE {r['initial_new_key_flow_mse']:.8f}→{r['final_same_new_key_flow_mse']:.8f}；固定noise sample MSE {r['initial_fixed_noise_sample_mse']:.8f}→{r['final_fixed_noise_sample_mse']:.8f}；换noise同条件sample MSE {r['new_noise_same_conditions_sample_mse']:.8f}。501步最后训练loss约.03156，不可代替新key评估.21798。首轮视觉梯度norm约84008.69，有限但很大，不能宣称大模型稳定性。十步缓存Euler、dt=-.1，输出 `[4,50,32]`全部有限；右图仍有残差。

独立进程通过Flax参数schema恢复msgpack，官方采样最大差{reload['max_abs_error']:.8g}（atol2e-5/rtol1e-5），不是逐bit要求。所有条件仍是训练四条件，没有未见任务评估。

## 产物、失败与修复

outputs包含结果.json（config/环境/hash/shape/指标）、训练曲线.json、运行日志.txt、构造日志.txt、tiny_pi0_params.msgpack、evaluation_fixture.npz、demo曲线.png、参数重载校验.json。fixture是本地合成输入/期望输出，参数不包含预训练语义。

初次import检查在torch依赖安装尚未结束时提前启动，报Tensor导入错误；等安装完成后构造通过。首个训练脚本误把Flax0.10.2 `State.flat_state()`的dict当作pairs迭代，导致gradient日志报unpack错误；改成.items()后官方训练通过，没有改官方模块。101步时sample MSE约.4666，故增加到501步后重新完整执行。环境安装的跨盘hardlink提示已自动copy，pip check通过。

## 未复现范围

未下载原PaliGemma/SigLIP或π₀预训练权重；没有完整3.3B、原始数据recipe、LeRobot数据管线、GPU/JAX分片、π₀.5/FAST或PyTorch推理。没有真实坐标/归一化统计、机器人reset/step、reward/done/成功率或控制频率实测。结果为官方小配置的合成训练与机制验证，不是论文任务性能复现。
'''
(R/'README.md').write_text(readme,encoding='utf-8')
spec=importlib.util.spec_from_file_location('writer',ROOT/'精读编写.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for f in ['src/openpi/models/pi0_config.py','src/openpi/models/siglip.py','src/openpi/models/model.py','src/openpi/shared/nnx_utils.py','src/openpi/transforms.py']:
 if f not in m.ROLES['pi0']:m.ROLES['pi0'].append(f)
m.write('pi0',s)
p=P/'解释.md';notes=p.read_text(encoding='utf-8')
notes+='''\n## 未解决问题与置信度限制\n\n原文部分自采集数据及完整recipe未具备本地复现材料；真实dataset/标准化/delta-action回映射未运行；未测三视角真实语义、条件遮挡与padding极端边界；不同专家真实2048/1024宽度、GPU bfloat16和大模型显存占用未运行；未独立分离图像/状态对合成任务的贡献；PyTorch路径未执行。图表没有标出的精确柱值不补造。小配置的公式、缓存、训练与采样由实际记录支持，3.3B或真机能力仍未验证。\n'''
p.write_text(notes,encoding='utf-8')
progress=json.loads((ROOT/'逐篇进度.json').read_text(encoding='utf-8'))
for item in progress:
 if item['paper']=='pi0':item.update(reading='flow换元、专家QKV、prefix/state/action mask、缓存、数据变换与实验已详细核对',demo='官方JAX小配置：501更新，flow/mask/cache、10步Euler和参数重载部分成功；视觉构造适配器')
(ROOT/'逐篇进度.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
idx=ROOT/'精读索引.md';sidx=idx.read_text(encoding='utf-8').replace('RT-1、OpenVLA、MDT、RDT-1B 已完成官方核心模块','RT-1、OpenVLA、MDT、RDT-1B、π₀ 已完成官方核心模块')
sidx+=f'\nπ₀更新：原始pi05=False路径已详读；官方JAX小配置完成flow/mask/cache验证、501更新、10步采样及参数重载。见 [运行记录](<{(R/"README.md").as_posix()}>). 含视觉构造适配器，完整论文未复现，下一篇为CogACT。\n';idx.write_text(sidx,encoding='utf-8')
links=re.findall(r'\]\(<([^>]+)>\)',notes);missing=[v for v in links if not v.startswith('http') and not Path(v).exists()];assert not missing,missing
for src,digest in r['source_sha256'].items():assert hashlib.sha256((REPO/src).read_bytes()).hexdigest()==digest
for key in ['必须会','建议会','可以暂时不掌握','Paper ↔ Code','未解决','部分成功']:assert key in notes,key
assert reload['max_abs_error']<reload['comparison_atol'] and reload['new_process_sampling_matches']
assert r['updates']==501 and len(json.loads((O/'训练曲线.json').read_text(encoding='utf-8')))==501
assert r['final_same_new_key_flow_mse']<r['initial_new_key_flow_mse'] and r['final_fixed_noise_sample_mse']<r['initial_fixed_noise_sample_mse']
for script in (R/'scripts').glob('*.py'):ast.parse(script.read_text(encoding='utf-8'))
(O/'交付校验.json').write_text(json.dumps({'local_links_checked':len(links),'missing_links':missing,'source_hashes_match':True,'reading_structure':'passed','script_syntax':'passed','updates_recorded':501,'independent_reload':'passed within float32 tolerance','full_paper_reproduction':False},ensure_ascii=False,indent=2),encoding='utf-8')
print(f'PASS: Pi0 notes {len(notes):,} chars; {len(links)} local links; source hashes, run metrics and reload checked.')