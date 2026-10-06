import sys,json,re,hashlib,importlib.util,ast
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'G:\具身\论文阅读');P=ROOT/'CogACT -  A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation';R=P/'复现';O=R/'outputs';REPO=P/'代码'/'CogACT'
r=json.loads((O/'结果.json').read_text(encoding='utf-8'));cg=json.loads((O/'CFG评估.json').read_text(encoding='utf-8'));reload=json.loads((O/'权重重载校验.json').read_text(encoding='utf-8'))
b=ROOT/'精读正文'/'CogACT.md';s=b.read_text(encoding='utf-8-sig')
start=s.index('## 8. 掌握层次与未解决问题');end=s.index('## 8. ActionModel',start);s=s[:start]+s[end:]
s=s.replace('本轮只静态核对这条链，没有实样本日志。','完整VLM链仍只静态核对，没有真实样本日志；动作模块实际运行记录见后文。')
s += f'''

## 12. 实验协议补充：不能只看主表平均值

附录B/PDF14–15页说明：Google VM试验数分别为Pick Coke300、Move Near240、Drawer216、Place Apple108；VA分别825、600、378、189。WidowX各任务24种原始设定，使用不同随机seed重复5次，合计120试验。VM视觉匹配与VA变体聚合不是相同干扰条件；四任务的试验总数也不相同，表内任务均值不能自动解释成按总trial数量加权的总体成功率。

作者评估用A6000或A100，DDIM10步、CFG1.5、adaptive α=.1。历史窗口K按 `C=K×std` 的经验关系选择，std为6D动作每步变化的平均尺度，C=.2；这不是所有机器人同一个固定K，也不能解释成纯粹以Hz确定K。单位和动作统计仍需结合数据接口。

Google部分baseline数字来自SIMPLER原报告，OpenVLA用官方权重重新评估；WidowX的RT-1-X使用已有报告，Octo多seed复测。不是所有表中baseline均由本文同一训练入口/同一seed重新训练。DiT-S在附录TableIII标约13M参数，本保存实现本demo计数为{r['parameter_count']:,}，两者同量级；主结果常用DiT-B约89M，不能把S小demo性能直接对照Base表。

## 13. 官方 DiT-S 小 demo 与诚实的误差边界

直接导入官方ActionModel，并使用现成model_type='DiT-S'，没有更改官方模型、diffusion、CFG或ensemble源码。PyTorch2.2.0+cpu、timm0.9.10、NumPy1.26.4，CPU四线程，float32，seed73。模型随机初始化，共 **{r['parameter_count']:,}** 参数。保留4096维认知输入、16×7动作块、100噪声级、cosine schedule、10步DDIM，训练条件dropout仍为官方.1。

合成条件八个z均匀取−.7～+.7，cognition只有前三个位置携带z/常数/0.5z，其余0；这不是Llama输出分布。动作目标 `[z,.5z,-.3z,.2z,0,0,.5+.2z]`重复16个位置。没有真实图像/语言、单位、坐标或环境。AdamW lr=.0003、weight_decay0，901更新；每步官方loss采新noise/time并训练。小demo的目的为验证连接、损失、采样与融合，不具备论文成功率复现条件。

|验证项|实际结果|能支持的结论|
|---|---|---|
|q_sample公式|与ᾱ广播公式一致|正向加噪执行正确|
|epsilon loss|同seed手工MSE与官方一致|训练target是噪声，返回标量|
|DiT输入hook|`[8,17,384]`|条件+16动作实际经过官方block|
|强制条件drop|与learned uncondition一致|官方无条件分支构造生效|
|初始梯度|输出头norm{r['initial_head_gradient']:.6f}；condition为0|与零输出头的链式梯度解释一致|
|训练后梯度|condition/time/action/first_block/output均有限非零|所检查模块收到梯度|
|history_embedder|无梯度|保存forward未使用该历史模块|
|固定新noise/time的epsilon MSE|{r['initial_validation_epsilon_mse']:.8f} → {r['final_validation_epsilon_mse']:.8f}|同一八条件下噪声预测误差下降|
|固定noise、无CFG的sample MSE|{r['initial_fixed_noise_sample_mse']:.8f} → {r['final_fixed_noise_sample_mse']:.8f}|输出误差下降，但仍未准确拟合动作|
|换noise、同条件sample MSE|{r['new_noise_same_conditions_sample_mse']:.8f}|结果依赖初始噪声，不是保留任务泛化|
|论文CFG1.5设置sample MSE|{cg['cfg_sample_mse']:.8f}|仅轻微改变合成采样误差，不宣称真实任务改进|
|CFG公式与batch配对|与独立算式一致，两半guided epsilon相同|条件/无条件共享噪声的公式生效|
|DDIM时间hook|90,80,...,0，共10次|respacing映射按源码执行|
|ensemble时间对齐/reset|与独立手算一致，reset后history0|历史融合方法实际运行|
|独立进程权重重载|strict通过，采样最大差{reload['max_abs_error']:.8g}|保存/恢复动作模块可复查|

**不能把本demo叫“合成动作准确复现成功”。** 901步epsilon MSE约.0142，而sample MSE仍.2317；图中负目标−.7的输出多在−.35～+.43，正目标+.7的输出多在+.97～+1.22，仍有明显偏差。没有用后处理clip隐藏超范围输出，最终core sample继续设置clip_denoised=False。初始噪声预测为0时，不裁剪DDIM会放大噪声，所以初始sample MSE约51.71；相对下降很大也不能当最终动作已准确的证据。

301更新试跑的epsilon MSE约.0592、sample MSE约.8352，换noise约1.1672，已保存在301更新试跑结果.json。增加到901确实改善误差，但条件表示很稀疏、训练分布合成且优化不充分；究竟哪个因素导致偏差尚未做受控消融，不能擅自归因于官方架构缺陷。小demo中条件梯度非零也不能证明学会真实认知。

![官方 DiT-S 合成训练与未裁剪采样](<{(O/'demo曲线.png').as_posix()}>)

右图横轴为16个动作位置，不是10个扩散步；左图为每步epsilon训练loss，右图为采样后的动作表示。两张图的物理含义与量纲不同，低loss不能直接换算任务成功率。

ensemble手算候选第0维为2、11、20，归一化权重约 `{[round(v,6) for v in r['ensemble_weights']]}`，融合第0维为{r['ensemble_result'][0]:.8f}。没有用这个融合结果补偿上图误差，因为测试ensemble所用chunk是单独构造的数值样例，不是模型预测的真实闭环轨迹。

[复跑说明](<{(R/'README.md').as_posix()}>)；[官方核心训练脚本](<{(R/'scripts'/'cogact_core_demo.py').as_posix()}>)；[重载与绘图](<{(R/'scripts'/'reload_and_plot.py').as_posix()}>)；[CFG评估](<{(R/'scripts'/'cfg_eval.py').as_posix()}>)；[原始结果](<{(O/'结果.json').as_posix()}>). 状态为**部分成功**：官方动作核心的运行与机制验证通过，输出精度不足；完整论文未复现。

## 14. 掌握层次与 Paper ↔ Code

**必须掌握**：cognition hidden feature与显式思维文本的区别；训练与单样本/批推理的认知位置路径；epsilon target；N+1动作与条件token；CFG条件/无条件配对；历史同一时刻对齐、权重归一化与reset；noise loss与sample误差、任务成功率分别判断。

**第二遍掌握**：实际DiT的concat conditioning与LayerNorm；dropout learned uncondition；零输出头梯度；DDIM respacing/eta；端到端认知梯度；SIMPLER两协议与非同一重训baseline；窗口K的数据尺度。

**可以暂时不掌握**：FSDP/多机训练、全套Prismatic实现、ROS/真实IK与所有数据源。先沿一个条件到动作的真实计算链掌握机制。

|Paper ↔ Code|固定快照定位|运行证据与边界|
|---|---|---|
|Figure2认知|CogACT.forward / predict_action / predict_action_batch|静态追踪三条提取路径；真实VLM未运行|
|条件到动作|ActionModel → DiT.forward、LabelEmbedder|官方DiT-S实际前向/反向，cognition为合成|
|Eq.(3)|ActionModel.loss → q_sample → net|原公式/target/shape/MSE实际验证|
|CFG|DiT.forward_with_cfg|公式、配对及CFG1.5采样实际运行|
|DDIM|create_ddim → space_timesteps → ddim_sample_loop|10次net原时间hook与重载通过|
|Eqs.(4–5)|AdaptiveEnsembler.ensemble_action/reset|实际归一化、时间对齐、history/reset通过|
|训练入口|scripts/train.py → strategy → CogACT.forward|完整训练数据/策略未执行|
|反标准化与环境|CogACT.predict_action → CogACTInference → SIMPLER|源码导航已定位；真实stats/环境step未执行|
|Tables1/7/8/9|原论文实验与附录B|数字解释与公平性限制；没有重跑benchmark|
'''
b.write_text(s,encoding='utf-8')
(R/'status.json').write_text(json.dumps({'status':'部分成功','scope':'官方DiT-S动作核心：901更新、epsilon/DDIM/CFG和ensemble验证，重载通过；采样精度不足，完整论文未复现','date':'2026-10-03','full_paper_reproduction':'未复现'},ensure_ascii=False,indent=2),encoding='utf-8')
(R/'requirements.txt').write_text('torch==2.2.0\ntorchvision==0.17.0\nnumpy==1.26.4\ntimm==0.9.10\nmatplotlib==3.10.7\n',encoding='utf-8')
readme=f'''# CogACT 官方动作核心小 demo

状态：**部分成功**。官方模块的训练/采样/CFG/ensemble运行通过；合成采样精度不足；完整论文未复现。commit `{r['repo_commit']}`，详读见 [解释.md](<{(P/'解释.md').as_posix()}>).

## 环境与资源

Windows、Python3.11.0、torch2.2.0+cpu、timm0.9.10、numpy1.26.4，CPU四线程、float32。复用 `G:\\具身\\论文阅读\\.venvs\\openvla`，本篇没有安装新的框架或下载大权重；uv pip check通过。requirements.txt仅列核心demo直接依赖，requirements-lock.txt记录实际共享环境。完整CogACT依赖还含Prismatic/OpenVLA、训练数据/仿真组件，本demo没有验证这些全入口。

12,476,807随机参数，模型checkpoint约50MB量级；不需要GPU、API/key、真实机器人数据或外部账号。不创建环境episode/reset/step/reward/done，不测控制频率，不发布动作命令。

## 复跑命令

工作目录 `G:\\具身`。PowerShell的&执行指定解释器，脚本路径含空格须保留引号。第一条训练并覆盖outputs；第二条独立进程权重重载/采样/绘图；第三条按论文CFG1.5推理，记录与无CFG同条件的误差。

```powershell
& 'G:\\具身\\论文阅读\\.venvs\\openvla\\Scripts\\python.exe' '{R/'scripts'/'cogact_core_demo.py'}'
& 'G:\\具身\\论文阅读\\.venvs\\openvla\\Scripts\\python.exe' '{R/'scripts'/'reload_and_plot.py'}'
& 'G:\\具身\\论文阅读\\.venvs\\openvla\\Scripts\\python.exe' '{R/'scripts'/'cfg_eval.py'}'
```

导入保存的ActionModel，使用官方DiT-S：6layers/hidden384/4heads，cognition4096、动作16×7、100噪声级、cosine schedule、dropout.1；不缩改模型factory。AdaptiveEnsembler按原文件独立导入，避免sim_cogact.__init__同时导入整个VLM/仿真包。调用实际方法，没有伪模块替代。

## 数据与训练

八个合成z条件，4096维向量中前三项携带z/1/.5z；不是真实VLM特征。七维动作 `[z,.5z,-.3z,.2z,0,0,.5+.2z]`重复16步，没有物理单位。AdamW lr=.0003、weight_decay0，901更新。真实ActionModel.loss随机采noise/time，训练learned无条件分支；评估使用新noise/time但仍是相同八条件，不测未见任务。

## 验证和结果

q_sample公式和epsilon MSE重建一致；实际hook为 `[8,17,384]`。强制drop条件与uncondition一致。初始输出头梯度非零、上游condition为0；训练后所检查condition/time/action/block/output梯度有限非零；未用history_embedder无梯度。

epsilon MSE {r['initial_validation_epsilon_mse']:.8f}→{r['final_validation_epsilon_mse']:.8f}。无CFG且固定初始noise的sample MSE {r['initial_fixed_noise_sample_mse']:.8f}→{r['final_fixed_noise_sample_mse']:.8f}；换noise={r['new_noise_same_conditions_sample_mse']:.8f}；CFG1.5同noise={cg['cfg_sample_mse']:.8f}。输出仍明显偏离目标，部分值超过归一化范围，**不算动作准确复现成功**。保持clip_denoised=False，未用后处理隐藏误差。noise loss降低不能直接证明生成精度。

DDIM10步映射原时间90/80/.../0。CFG公式与batch两半验证一致。ensemble按历史同一时刻的2/11/20候选融合，第0维{r['ensemble_result'][0]:.8f}，history/reset通过；此为独立构造样例，不用于“修正”模型采样。新进程strict state_dict加载、官方DDIM采样最大差{reload['max_abs_error']:.8g}。

## 产物与修复记录

outputs包含结果.json（config/环境/hash/shape/指标）、训练曲线.json、运行日志.txt、dit_s_state.pt、evaluation_fixture.pt、demo曲线.png、权重重载校验.json、CFG评估.json和301更新试跑结果.json。fixture是本地合成输入/期望输出，没有机器人数据。

第一次主脚本在训练/采样完成后，inspect.getfile对直接按文件加载且未注册到sys.modules的class误判built-in，来源检查失败；改为定位AdaptiveEnsembler.ensemble_action的代码文件后重跑通过，官方源码没修改。301步epsilon loss已下降而sample仍.835，因此保存试跑记录后增加到901；输出误差进一步下降但仍不充分，没有把最终结果改称全成功。

## 未复现范围

没有执行完整CogACT.forward、Llama2/DINOv2/SigLIP、真实cognition token/padding契约、预训练DiT-B/大模型权重、OXE数据和全部训练recipe、真实归一化/坐标/夹爪接口、SIMPLER或真机。误差来源没有受控消融，不能单凭合成demo归因于官方架构。后续若做基准，需要完整checkpoint、数据/环境及论文推理配置。
'''
(R/'README.md').write_text(readme,encoding='utf-8')
spec=importlib.util.spec_from_file_location('writer',ROOT/'精读编写.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for f in ['action_model/__init__.py','action_model/gaussian_diffusion.py','action_model/respace.py','action_model/diffusion_utils.py','sim_cogact/adaptive_ensemble.py']:
 if f not in m.ROLES['CogACT']:m.ROLES['CogACT'].append(f)
m.write('CogACT',s)
p=P/'解释.md';notes=p.read_text(encoding='utf-8')
notes+='''\n## 未解决问题与置信度限制\n\n真实VLM视觉插入/BOS/padding与cognition位置未运行验证；认知特征分布和VLM→动作的端到端梯度未实测；全部mask为空、chunk历史窗口越界、真实norm stats与夹爪约定尚未测试；未加载checkpoint/评测SIMPLER/真机；目前合成DDIM输出仍有显著偏差，条件尺度、训练量、dropout/EMA/推理分布各因素没有单独消融。动作核心源码与机制运行证据较高，但不支持生成动作已准确或真实任务成功的结论。\n'''
p.write_text(notes,encoding='utf-8')
progress=json.loads((ROOT/'逐篇进度.json').read_text(encoding='utf-8'))
for item in progress:
 if item['paper']=='CogACT':item.update(reading='认知位置、DiT concat条件、epsilon/CFG/DDIM、ensemble及实验协议已详细核对',demo='官方DiT-S：901更新、DDIM/CFG/历史融合和重载部分成功；合成采样精度不足')
(ROOT/'逐篇进度.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
idx=ROOT/'精读索引.md';sidx=idx.read_text(encoding='utf-8').replace('RT-1、OpenVLA、MDT、RDT-1B、π₀ 已完成官方核心模块','RT-1、OpenVLA、MDT、RDT-1B、π₀、CogACT 已完成官方核心模块')
sidx+=f'\nCogACT更新：官方DiT-S动作核心完成901更新、DDIM/CFG、历史融合和权重重载；采样精度仍不足，详见 [运行记录](<{(R/"README.md").as_posix()}>). 核心详读完成，完整论文未复现；下一篇为UniPi。\n';idx.write_text(sidx,encoding='utf-8')
links=re.findall(r'\]\(<([^>]+)>\)',notes);missing=[v for v in links if not v.startswith('http') and not Path(v).exists()];assert not missing,missing
for src,digest in r['source_sha256'].items():assert hashlib.sha256((REPO/src).read_bytes()).hexdigest()==digest
for key in ['必须会','建议会','可以暂时不掌握','Paper ↔ Code','未解决','部分成功']:assert key in notes,key
assert r['updates']==901 and len(json.loads((O/'训练曲线.json').read_text(encoding='utf-8')))==901
assert reload['max_abs_error']==0 and r['ensemble_history_and_reset']=='passed' and r['CFG_formula_and_batch_pair']=='passed'
assert r['final_validation_epsilon_mse']<r['initial_validation_epsilon_mse']
for f in (R/'scripts').glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
(O/'交付校验.json').write_text(json.dumps({'local_links_checked':len(links),'missing_links':missing,'source_hashes_match':True,'reading_structure':'passed','script_syntax':'passed','updates_recorded':901,'independent_reload':'passed','accurate_action_reproduction':False,'full_paper_reproduction':False},ensure_ascii=False,indent=2),encoding='utf-8')
print(f'PASS: CogACT notes {len(notes):,} chars, {len(links)} links; hashes/records/reload verified. Sampling accuracy remains limited.')