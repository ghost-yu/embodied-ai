import sys,json,importlib.util,re,hashlib
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'G:\具身\论文阅读')
P=ROOT/'Multimodal Diffusion Transformer -  Learning Versatile Behavior from Multimodal Goals'
R=P/'复现';O=R/'outputs'
r=json.loads((O/'结果.json').read_text(encoding='utf-8'))
b=ROOT/'精读正文'/'MDT.md';s=b.read_text(encoding='utf-8-sig')
start=s.index('## 8. 掌握层次与未解决问题');end=s.index('## 8. Transformer',start)
s=s[:start]+s[end:]
s=s.replace('真实机器人 batch 可能有相同语言标签，直接把其他样本全当负例可能有问题；当前代码会把 lang_text 和 mode 传给 cont_loss，需检查实际配置，不能仅按最简 InfoNCE 认定所有实现细节。','真实机器人 batch 可能有相同语言标签，直接把其他样本全当负例可能有问题。已追踪到当前 loss 方法接收但不使用 lang_text，具体见第 9 节；实际 batch 是否存在重复文本仍未统计。')
s += f'''

## 10. 小 demo：官方动作核心实际运行

本次导入保存仓库的 `MDTVTransformer`、`GCDenoiser` 和 `sample_ddim`，通过 `inspect.unwrap` 核验原函数文件路径，并对六个关键源码保存 SHA-256。没有用伪模块替代依赖，也没有改动官方核心源码。模型随机初始化，共 **{r['parameter_count']:,}** 个参数；2 层 encoder、2 层 decoder、4 heads、hidden=64。PyTorch {r['environment']['torch']}，CPU，seed=43，241 次 AdamW 更新。此为缩小配置的官方模块 demo，状态为**部分成功；完整论文实验未复现**。

每个合成任务含状态特征、图像目标特征和语言目标特征；后两者使用不同输入位置编码同一标量 z。动作块的目标为 `[z,0.5z,-0.3z,0.2z,0,0,0.2]`，重复十个位置。z 取 16 个 -0.7 到 +0.7 的值；这些不是米、弧度或真实末端坐标。两个目标分支交替训练，没有生成真实图片或处理真实文字。`torch.stack(...,dim=-1)` 把七个长度 B 的向量变成 `[B,7]`；`[:,None,:]` 增加时间轴，再 `repeat(1,10,1)` 得 `[B,10,7]`。

|检查|实际结果|能支持的结论|
|---|---|---|
|EDM target 与 wrapper 组合|手工公式与官方输出一致|内部输出与干净动作输出关系正确|
|前向|encoder `[16,5,64]`，动作 `[16,10,7]`|缩小配置张量路径可执行|
|反向|encoder/decoder/goal/sigma/action head 梯度有限且非零|所检查核心模块收到梯度；sigma 梯度很小，不能因此宣称其训练充分|
|绝对位置参数|此路径 pos_emb.grad 为 None|与未调用 position helper 的静态定位一致|
|新噪声去噪评估，vis|{r['initial_same_conditions_new_noise_mse']['vis']:.8f} → {r['final_same_conditions_new_noise_mse']['vis']:.8f}|同一批训练条件，换噪声后误差下降|
|新噪声去噪评估，lang|{r['initial_same_conditions_new_noise_mse']['lang']:.8f} → {r['final_same_conditions_new_noise_mse']['lang']:.8f}|同上，不是保留任务泛化|
|官方 DDIM|10 次模型调用；输出 `[2,10,7]`，全部有限|核心采样可运行|
|eta 参数行为|固定输入时 eta=0 与 eta=1 一致|保存的当前 DDIM 实现没有使用 eta|
|权重重载|独立进程 strict load，采样最大差 0|保存和加载链路可复查|

采样两目标的动作块 MSE={r['sampled_goal_pair_mse']:.8f}。第一维目标 -0.7/+0.7 的生成值分别大约 -0.47 到 -0.29、+0.49 到 +0.65，方向已区分，仍有偏差。图右横轴为**动作块内的位置**，不是去噪步骤；十次去噪的 callback 数据单独存放在 `采样记录.json`。训练图展示内部 target loss，不能直接拿它与干净动作 MSE 的数值比较。

![合成训练与动作块采样](<{(O/'demo曲线.png').as_posix()}>)

官方默认 sigma_max=80，小 demo 改为 1；sigma_min=.001，十级 exponential schedule 加末尾零。训练 sigma 为 .03～1 的 log-uniform 分布，和默认配置的采样密度不同。结果不能用于对照论文分数。MGF/CLA 的完整训练、Voltron/CLIP 前端、CALVIN/LIBERO、真机控制都没有执行。没有下载论文预训练 checkpoint。

复跑命令、依赖、失败修复及脚本说明见 [复现记录](<{(R/'README.md').as_posix()}>)；[训练脚本](<{(R/'scripts'/'mdt_core_demo.py').as_posix()}>)；[原始结果](<{(O/'结果.json').as_posix()}>)。本 demo 未建立环境 episode、reset/step、reward/done 或控制频率，输出是合成动作张量，不能连接真机执行。

## 11. 掌握层次与 Paper ↔ Code

**必须掌握**：四种时间/长度 j、v、k、K 的区别；当前状态与目标怎样构成上下文；EDM 三个系数与内部 target；训练、采样各自的输入输出；MGF 的未来图像仅用于训练；CLA 的 batch 对角匹配；论文与配置权重差异。

**第二遍掌握**：SDE/probability-flow ODE、adaLN、sigma embedding；同文不同目标模态的监督来源；重复语言负例；CALVIN 长链统计与 seed 条件；MGF reduction 和图像 patch 内维度。

**可以暂时不掌握**：全部 sampler 变体、distributed gather、完整 CALVIN/机器人驱动安装。小模型需要这些核心知识，不需要先把所有基础库读一遍。

|Paper ↔ Code|定位|验证边界|
|---|---|---|
|Eq.(1) play/目标|数据 module → training_step → compute_input_embeddings|已建立导航；真实 batch 时间索引未运行|
|Figure 1 动作 Transformer|MDTVTransformer.forward/forward_enc_only/forward_dec_only；transformer_blocks|官方缩小配置前向/反向实际通过|
|Eq.(4) SM/EDM|GCDenoiser.loss/forward/get_scalings|实际输出与手工公式核对|
|Eq.(5) MGF|compute_img_gen_loss → MaskedTransformerImgDecoder.compute_loss|静态核对；按 mask.sum、两个时间位置平均；未执行训练|
|Eq.(6) CLA|compute_contrastive_loss → clip_auxiliary_loss|静态核对三个 mode；lang_text 未使用；未执行训练|
|Eq.(7) 总目标|MDTVAgent.training_step；conf/model/mdtv_agent.yaml|论文与保存默认权重有差异，最终实验覆写未全核对|
|动作推理|MDTVAgent.forward → sample_ddim|Agent 整体未运行；官方 sampler 独立运行通过|
|论文 checkpoint 与基准成绩|官方资源导航|本 demo 未加载 checkpoint；没有基准复现结果|
'''
b.write_text(s,encoding='utf-8')
(R/'status.json').write_text(json.dumps({'status':'部分成功','scope':'官方 MDT 动作核心缩小配置：前向、反向、合成训练与 DDIM；完整论文未复现','date':'2026-10-03','full_paper_reproduction':'未复现'},ensure_ascii=False,indent=2),encoding='utf-8')
req='torch==2.2.0\nnumpy==1.26.4\nhydra-core==1.3.2\neinops==0.8.0\nmatplotlib==3.10.7\nscipy==1.13.1\ntorchsde==0.2.6\ntorchdiffeq==0.2.4\n'
(R/'requirements.txt').write_text(req,encoding='utf-8')
readme=f'''# MDT 复现记录

状态：**部分成功（官方动作核心缩小配置）**；完整论文未复现。固定 commit `{r['repo_commit']}`。原文、公式、源码解释见 [解释.md](<{(P/'解释.md').as_posix()}>).

## 环境和范围

Windows，Python 3.11.0，PyTorch 2.2.0+cpu、NumPy 1.26.4、Hydra 1.3.2、einops 0.8.0。复用 `论文阅读/.venvs/openvla` 的 PyTorch 环境；实际使用 CPU、四线程，没有 CUDA 训练。新增依赖没有替换已有 torch/transformers；`uv pip check` 通过。直接需求见 requirements.txt，完整环境见 requirements-lock.txt。未加载预训练 checkpoint 或数据集；使用合成特征和七维合成动作，模型329,095参数。

## 复跑

工作目录 `G:\\具身`；以下 PowerShell 命令中的 `&` 执行指定解释器，路径带空格需引号。第一个脚本构建模型、训练、采样并覆盖本 demo outputs；第二个脚本在独立进程加载权重、验证采样并绘图。

```powershell
& 'G:\\具身\\论文阅读\\.venvs\\openvla\\Scripts\\python.exe' '{R / 'scripts' / 'mdt_core_demo.py'}'
& 'G:\\具身\\论文阅读\\.venvs\\openvla\\Scripts\\python.exe' '{R / 'scripts' / 'reload_and_plot.py'}'
```

脚本通过 sys.path 导入保存仓库，不需要安装完整 MDTVAgent 环境。Hydra `_target_` 指向真实 MDTVTransformer；GCDenoiser 实例化内部网络。实际检查函数来源时先 unwrap 装饰器；六个源文件的 hash 在结果.json，可和保存仓库逐一核对。

## 训练与验证

16 个 z 条件，两种合成目标 embedding；action `[16,10,7]`，context `[16,5,64]`。网络 hidden64、四 heads、2 encoder/2 decoder、dropout0、goal_drop0。AdamW lr=.002、weight_decay0，241 更新。训练 sigma 为 .03～1 的 log-uniform；评估固定 sigma=.5、使用新噪声但相同条件。

vis MSE {r['initial_same_conditions_new_noise_mse']['vis']:.8f}→{r['final_same_conditions_new_noise_mse']['vis']:.8f}；lang {r['initial_same_conditions_new_noise_mse']['lang']:.8f}→{r['final_same_conditions_new_noise_mse']['lang']:.8f}。这些是同条件新噪声的离线误差，不是保留任务泛化或成功率。官方10步DDIM、sigma 1→.001→0，生成两目标 `[2,10,7]`；MSE={r['sampled_goal_pair_mse']:.8f}。固定输入 eta0/1 一致；重载 strict=True 后采样最大差0。源码默认 sigma_max80，demo 改成1。

产物：结果.json（版本/config/shape/hash/指标），运行日志.txt，训练曲线.json，采样记录.json，demo配置.json，demo曲线.png，tiny_mdt_state.pt，evaluation_fixture.pt，权重重载校验.json。fixture 保存本地生成的输入与期望输出，用于重新加载验证，不是官方评测样本。

## 失败与修复

uv 首次下载 SciPy 1.13.1 遇到 TLS close_notify 中断；改用 Windows curl 分段下载官方 PyPI wheel，复用成功的分段，拼接后大小和 PyPI SHA-256 一致。元数据、下载脚本和校验在 `论文阅读/依赖缓存`，没有关闭 TLS 校验。首次脚本的来源断言在 `@torch.no_grad()` 包装函数上失败；用 inspect.unwrap 定位原函数后重跑全部验证通过。这个错误来自 demo 的检查逻辑，没有改官方模型。

## 限制

未实例化完整 MDTVAgent；未运行 Voltron/CLIP、MGF/CLA 联合训练；没有 CALVIN/LIBERO 或真机评估。网络大小、噪声采样、数据与论文训练不同。模型的偏差仍可从曲线看到。输出的数值不是实际机器人单位、坐标系或可执行命令。后续若做论文评测，需要真实 batch、预训练资源、完整配置覆写与官方环境。
'''
(R/'README.md').write_text(readme,encoding='utf-8')
spec=importlib.util.spec_from_file_location('writer',ROOT/'精读编写.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for f in ['mdt/models/networks/transformers/transformer_blocks.py','mdt/models/networks/transformers/position_embeddings.py','mdt/models/edm_diffusion/gc_sampling.py','mdt/models/edm_diffusion/utils.py']:
 if f not in m.ROLES['MDT']:m.ROLES['MDT'].append(f)
m.write('MDT',s)
p=P/'解释.md'
p.write_text(p.read_text(encoding='utf-8')+'''\n## 未解决问题与置信度限制\n\n没有读取真实 batch 验证时间索引、机器人动作坐标/归一化与 control frequency；没有核对全部 Hydra 实验覆写；MGF 零 mask 分母、实际 mask 数分布和联合训练未验证；CLA 真正 batch 的重复文本与分布式配对未统计。官方动作核心的缩小配置运行证据较高；完整数据/Agent/基准/真机结论不由此证明。\n''',encoding='utf-8')
progress=json.loads((ROOT/'逐篇进度.json').read_text(encoding='utf-8'))
for item in progress:
 if item['paper']=='MDT':item.update(reading='核心公式、Transformer路径、MGF/CLA和采样已详细核对',demo='官方动作核心缩小配置：前向、反向、合成训练、DDIM和权重重载部分成功')
(ROOT/'逐篇进度.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
index=ROOT/'精读索引.md';idx=index.read_text(encoding='utf-8')
idx=idx.replace('**完整论文实验：全部未复现。RT-1 已完成官方 FiLM/TokenLearner 模块 demo（部分成功）。**','**完整论文实验：全部未复现。RT-1、OpenVLA、MDT 已完成官方核心模块的不同范围 demo（部分成功）；RT-2 已完成教学 demo。**')
idx+=f'\nMDT 更新：已详读并运行官方 Transformer/EDM/DDIM 的缩小配置，完成合成训练和独立权重重载。见 [MDT 运行记录](<{(R/"README.md").as_posix()}>). 完整论文未复现，下一篇为 RDT-1B。\n'
index.write_text(idx,encoding='utf-8')
# 只读审计本篇，不改其他论文。
notes=p.read_text(encoding='utf-8')
links=re.findall(r'\]\(<([^>]+)>\)',notes)
missing=[v for v in links if not v.startswith('http') and not Path(v.split(':')[0] if not re.match(r'^[A-Za-z]:',v) else v).exists()]
assert not missing,missing
for src,digest in r['source_sha256'].items():assert hashlib.sha256((P/'代码'/'mdt_policy'/src).read_bytes()).hexdigest()==digest
for key in ['必须会','建议会','可以暂时不掌握','Paper ↔ Code','未解决','部分成功']:assert key in notes,key
reload=json.loads((O/'权重重载校验.json').read_text(encoding='utf-8'));assert reload['max_abs_error']==0
(O/'交付校验.json').write_text(json.dumps({'local_links_checked':len(links),'missing_links':missing,'six_source_hashes_match':True,'reading_structure':'passed','independent_reload':'passed','complete_paper_reproduction':False},ensure_ascii=False,indent=2),encoding='utf-8')
print(f'PASS: MDT notes {len(notes):,} chars; {len(links)} local links; source hashes and reload verified.')