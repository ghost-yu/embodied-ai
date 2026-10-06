import sys,json,re,hashlib,importlib.util,math
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'G:\具身\论文阅读');P=ROOT/'RDT-1B -  a Diffusion Foundation Model for Bimanual Manipulation';R=P/'复现';O=R/'outputs';REPO=P/'代码'/'RoboticsDiffusionTransformer'
def portable(v):
 if isinstance(v,float) and not math.isfinite(v):return '-Infinity' if v<0 else 'Infinity'
 if isinstance(v,dict):return {k:portable(x) for k,x in v.items()}
 if isinstance(v,list):return [portable(x) for x in v]
 return v
r=portable(json.loads((O/'结果.json').read_text(encoding='utf-8')))
(O/'结果.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
# Make future report exports strict JSON without changing numerical runtime/scheduler config.
script=R/'scripts'/'rdt_core_demo.py';code=script.read_text(encoding='utf-8')
code=code.replace("(OUT/'结果.json').write_text", "report['scheduler_runtime']['lambda_min_clipped']=str(report['scheduler_runtime']['lambda_min_clipped'])\n(OUT/'结果.json').write_text")
code=code.replace('ensure_ascii=False,indent=2),encoding=', 'ensure_ascii=False,indent=2,allow_nan=False),encoding=')
script.write_text(code,encoding='utf-8')
b=ROOT/'精读正文'/'RDT-1B.md';s=b.read_text(encoding='utf-8-sig')
start=s.index('## 8. 掌握层次与未解决问题');end=s.index('## 8. Figure 3',start);s=s[:start]+s[end:]
s=s.replace('§4.1/Eqs.(1–2)/Figure 3 为扩散与架构','§4.1/Eqs.(1–2)/Figure 3（PDF 4–7 页）为扩散与架构')
s=s.replace('自定义 fine-tuning 数据在 `train/dataset.py` 的 HDF5VLADataset 路径','自定义 fine-tuning 数据从 `train/dataset.py` 进入 `data/hdf5_vla_dataset.py` 的 HDF5VLADataset 路径')
s += f'''

## 13. 小 demo：官方 Runner 保留128维与64动作块

本次直接导入固定commit的 `RDTRunner`、`RDT` 和 `configs/state_vec.py`。模型不是重新写的教学Transformer；scheduler也是官方Runner实际调用的diffusers。hidden=64、depth=4、heads=4，合计 **{r['parameter_count']:,}** 参数，随机初始化；动作维128、horizon64、1000训练噪声等级、cosine beta schedule、sample预测、五步DPM-Solver保持保存配置的这些接口。视觉/语言输入替换为合成特征，不实例化SigLIP/T5。float32 CPU、四线程、seed53。

右六关节与左六关节各占自己的物理槽位，加两个gripper共14维。有效索引为 `{r['valid_slot_indices']}`。合成标量z有16个值，右关节目标为z乘不同系数，左关节为相反符号，两个gripper为0.5±0.2z，均重复64步。它们没有真实弧度/位移单位，不是可发布的机器人动作。图像输入是随机固定特征；任务主要由合成语言特征编码z，因此不能声称演示了视觉感知能力。

AdamW lr=.0015、weight_decay0，201次更新，loss由真实 `runner(*batch)` 计算。float32用于本机CPU，不是论文bfloat16大模型配置。初始模型、最终模型都用同一seed778进行官方五步采样，另用779换初始噪声；条件仍是训练中的16个条件，不是保留任务泛化。

|验证|实际结果|可支持的结论|
|---|---|---|
|add_noise与手工公式|一致|DDPM正向噪声广播与ᾱ索引正确|
|输入与ACI hooks|67个序列位置，条件长度4/6/4/6|time、frequency、state和64动作路径可执行|
|原始loss与全维MSE|一致|训练目标确实为sample，reduction为全维平均|
|无效target扰动|loss增加0.0078125|不是仅有效维loss|
|初始梯度|输出头非零，lang adaptor为0|零输出头暂时阻断上游梯度|
|训练后梯度|五个所检查模块全部有限非零|语言、图像、state与核心网络在该demo中收到梯度|
|有效槽位采样MSE，seed778|{r['initial_same_seed_sampling']['valid_slots_mse']:.8f} → {r['final_same_seed_sampling']['valid_slots_mse']:.8f}|同条件合成动作拟合明显改善|
|换噪声seed779的有效槽位MSE|{r['new_noise_same_conditions_sampling']['valid_slots_mse']:.8f}|同条件换噪声仍接近合成目标|
|无效输出维|全部精确为0|采样最后乘mask生效|
|语言padding改为999|预测最大差0|当前布尔attention mask生效；在训练后的非零模型检查|
|独立进程重载|strict load通过，采样最大差0|保存权重与配置足以复查核心推理|

全维sample MSE为{r['final_same_seed_sampling']['all_slots_mse']:.10f}，明显小于有效维指标，因为114个无效槽位最终全部为0；不应用这个较小值夸大动作精度。训练最后一步全维loss约5.2676e−6，和采样MSE还存在不同噪声/调用方式，不能当作同一个测量。

![合成训练与官方采样](<{(O/'demo曲线.png').as_posix()}>)

左图纵轴log scale，反映全128维训练MSE；右图横轴为64动作块的位置，**不是五个去噪时间步**。右图只展示右joint0，两种目标约−.7/+.7。没有物理环境step，无法从图推出轨迹可执行、稳定或真实成功率。

详见 [复现记录](<{(R/'README.md').as_posix()}>)、[官方核心demo脚本](<{(R/'scripts'/'rdt_core_demo.py').as_posix()}>)、[重载脚本](<{(R/'scripts'/'reload_and_plot.py').as_posix()}>)、[原始结果](<{(O/'结果.json').as_posix()}>). 状态为**部分成功**；1B预训练权重、完整前端、真实HDF5、论文benchmark和真机均未复现。

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
'''
b.write_text(s,encoding='utf-8')
(R/'status.json').write_text(json.dumps({'status':'部分成功','scope':'官方 RDTRunner/RDT 缩小配置，128维64动作块：训练、ACI/mask、五步采样和重载；完整论文未复现','date':'2026-10-03','full_paper_reproduction':'未复现'},ensure_ascii=False,indent=2),encoding='utf-8')
(R/'requirements.txt').write_text('torch==2.2.0\ntorchvision==0.17.0\nnumpy==1.26.4\ntimm==1.0.3\ndiffusers==0.27.2\nhuggingface-hub==0.23.0\nmatplotlib==3.10.7\nPyYAML==6.0.2\n',encoding='utf-8')
readme=f'''# RDT-1B 官方核心小 demo 运行记录

状态：**部分成功**。固定代码commit `{r['repo_commit']}`；完整论文未复现。原文解释见 [解释.md](<{(P/'解释.md').as_posix()}>).

## 环境与资源

Windows、Python3.11.0、CPU四线程、float32；torch2.2.0+cpu、timm1.0.3、diffusers0.27.2、huggingface-hub0.23.0、numpy1.26.4。独立环境 `G:\\具身\\论文阅读\\.venvs\\rdt`，与OpenVLA/MDT分开；requirements.txt列直接依赖，requirements-lock.txt记录完整实际版本，uv pip check通过。本机有RTX4060 Laptop，但没有在此demo使用CUDA。没有下载1B权重、T5-XXL/SigLIP或大数据集，不调用外部API、不需key。模型260,608参数，生成合成特征与动作，不创建400GB训练缓冲。

## 复跑命令

工作目录 `G:\\具身`。PowerShell中的&执行指定解释器；第一个脚本训练并覆盖demo outputs，第二个在新进程重载并校验采样。路径中有空格，需保持引号。

```powershell
& 'G:\\具身\\论文阅读\\.venvs\\rdt\\Scripts\\python.exe' '{R/'scripts'/'rdt_core_demo.py'}'
& 'G:\\具身\\论文阅读\\.venvs\\rdt\\Scripts\\python.exe' '{R/'scripts'/'reload_and_plot.py'}'
```

核心实际调用链：RDTRunner.__init__ → RDT/adaptors/schedulers → compute_loss（runner.forward代理）→ optimizer；predict_action → adapt_conditions → conditional_sample → 五次RDT/scheduler.step → action_mask。直接导入保存仓库；没有替换官方模型/采样器源码。结果.json保存来源路径对应hash，git快照保持干净。

## 配置与数据

保留action128、horizon64、1000噪声等级、cosine schedule、prediction_type=sample、五步采样。缩小hidden64/depth4/heads4；语言特征12维×4tokens，图像8维×6tokens，状态128维；float32替代默认bfloat16。语言是合成embedding，图像是固定随机特征；不是真实前端。官方14个双臂关节/夹爪槽位有效，其余零padding。目标右/左关节随标量z取相反符号；gripper为0.5±.2z，无真实单位。

AdamW lr=.0015、weight_decay0，201更新。训练随机采noise/timestep；初始和最终采样seed778相同，另以779换噪声。条件都是训练中16个条件，没有未见任务/视觉泛化评测。

## 实际验证

公式add_noise一致；真实hook记录ACI条件长度4/6/4/6、query长度67。初始全维loss={r['training_loss_at_zero_head']:.8f}，有效维平均={r['valid_slots_mean_at_zero_head']:.8f}；无效目标改为1增加1/128。输出头初始梯度非零，上游语言adaptor初始为0，训练后核心模块梯度全部有限非零。

有效维采样MSE：{r['initial_same_seed_sampling']['valid_slots_mse']:.8f}→{r['final_same_seed_sampling']['valid_slots_mse']:.8f}；换噪声={r['new_noise_same_conditions_sampling']['valid_slots_mse']:.8f}。无效输出精确为0。语言padding改999，训练后预测最大差0。实际五次模型调用时间为999/799/599/400/200，DPM-Solver末尾sigma零由库处理。独立进程strict权重加载、相同seed的官方采样最大差0。

## 文件与限制

scripts/rdt_core_demo.py为训练/验证；reload_and_plot.py为新进程重载/绘图。outputs内含结果.json、训练曲线.json、运行日志.txt、tiny_rdt_state.pt、evaluation_fixture.pt、demo曲线.png、权重重载校验.json。fixture为本地合成输入/期望输出，不是官方benchmark。结果.json将scheduler的非有限配置lambda_min_clipped用字符串保存，以便严格JSON解析；不改变库运行值。

此环境安装及模型demo没有终端失败；pip复制缓存wheel时提示跨盘hardlink转copy，不影响安装，pip check通过。未运行全训练入口，因此不能据此声称DeepSpeed/Accelerate/HDF5/ROS均可用。没有真实坐标/动作单位处理、环境reset/step或真机命令。论文1.2B配置、冻结encoder、预训练checkpoint、真实数据与success rate都待后续完整实验。
'''
(R/'README.md').write_text(readme,encoding='utf-8')
spec=importlib.util.spec_from_file_location('writer',ROOT/'精读编写.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for f in ['data/hdf5_vla_dataset.py','train/sample.py','models/hub_mixin.py','models/multimodal_encoder/t5_encoder.py','models/multimodal_encoder/siglip_encoder.py','scripts/agilex_inference.py']:
 if f not in m.ROLES['RDT-1B']:m.ROLES['RDT-1B'].append(f)
m.write('RDT-1B',s)
p=P/'解释.md';notes=p.read_text(encoding='utf-8')
notes+='''\n## 未解决问题与置信度限制\n\n所有46种来源的单位、坐标、相对/绝对动作、旋转表示转换未全量核对；真实HDF5加载与collator尚未运行；全部mask为零时评估分母、所有语言padding时attention边界未测试；未测真实图像前端、多维图像位置布局、GPU bfloat16与1B内存占用；完整checkpoint/论文基准/部署未执行。训练loss的mask边界已由主链源码和官方运行共同支持，不再列为未确认项。缩小官方动作核心的执行证据较高，真实任务能力不能从合成demo推出。\n'''
p.write_text(notes,encoding='utf-8')
progress=json.loads((ROOT/'逐篇进度.json').read_text(encoding='utf-8'))
for item in progress:
 if item['paper']=='RDT-1B':item.update(reading='核心公式、128维槽位、ACI/QKNorm、训练/评估reduction与采样已详细核对',demo='官方RDTRunner缩小配置：128维64动作块，训练/mask/ACI/五步采样和重载部分成功')
(ROOT/'逐篇进度.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
idx=ROOT/'精读索引.md';sidx=idx.read_text(encoding='utf-8')
sidx=sidx.replace('RT-1、OpenVLA、MDT 已完成官方核心模块','RT-1、OpenVLA、MDT、RDT-1B 已完成官方核心模块')
sidx+=f'\nRDT-1B更新：详细精读与官方核心demo完成，保留128维/64动作块，含201更新、mask/ACI验证和五步采样重载。见 [运行记录](<{(R/"README.md").as_posix()}>). 完整论文未复现，下一篇为π0。\n';idx.write_text(sidx,encoding='utf-8')
# Audit current deliverables instead of the obsolete all-paper verifier.
links=re.findall(r'\]\(<([^>]+)>\)',notes);missing=[v for v in links if not v.startswith('http') and not Path(v).exists()];assert not missing,missing
for src,digest in r['source_sha256'].items():assert hashlib.sha256((REPO/src).read_bytes()).hexdigest()==digest
for key in ['必须会','建议会','可以暂时不掌握','Paper ↔ Code','未解决','部分成功']:assert key in notes,key
reload=json.loads((O/'权重重载校验.json').read_text(encoding='utf-8'));assert reload['max_abs_error']==0
import ast
for file in (R/'scripts').glob('*.py'):ast.parse(file.read_text(encoding='utf-8'))
(O/'交付校验.json').write_text(json.dumps({'local_links_checked':len(links),'missing_links':missing,'source_hashes_match':True,'reading_structure':'passed','script_syntax':'passed','independent_reload':'passed','full_paper_reproduction':False},ensure_ascii=False,indent=2),encoding='utf-8')
print(f'PASS: RDT notes {len(notes):,} chars, {len(links)} links, source hashes and independent reload verified.')