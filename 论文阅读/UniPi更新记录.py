import sys,json,re,hashlib,importlib.util,ast,platform
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'G:\具身\论文阅读')
P=ROOT/'Learning Universal Policies via Text-Guided Video Generation'
R=P/'复现';O=R/'outputs';D=P/'资料'
r=json.loads((O/'结果.json').read_text(encoding='utf-8'))
v=json.loads((O/'权重重载校验.json').read_text(encoding='utf-8'))
tree=json.loads((D/'官方网页仓库树.json').read_text(encoding='utf-8'))
assert tree['truncated']==False
paths=[x['path'] for x in tree['tree'] if x['type']=='blob']
assert not any(x.endswith('.py') for x in paths)
audit={'date':'2026-10-05','paper':'2302.00111v3','primary_project':'https://universal-policy.github.io/unipi/',
       'primary_blog':'https://research.google/blog/unipi-learning-universal-policies-via-text-guided-video-generation/',
       'website_repository':'https://github.com/universal-policy/unipi','website_tree_sha':tree['sha'],
       'tree_truncated':False,'blob_count':len(paths),'python_files':[],
       'project_statement':'作者项目页说明原实验大部分在Google完成，源码不能发布，并推荐flow-diffusion/AVDC作为视频建模/训练基础。',
       'recommended_code':'https://github.com/flow-diffusion/AVDC',
       'recommended_code_identity':'另篇Learning to Act from Actionless Videos through Dense Correspondences官方实现，arxiv2310.08576；非原UniPi源码/权重。',
       'local_demo':'本地新写教学实现，未导入或执行AVDC。','confidence_limit':'限定已查作者网页、网页仓库、项目推荐repo与原文，不代表全互联网穷尽查找。'}
(D/'官方资源核查.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
base=(ROOT/'精读正文'/'UniPi.md').read_text(encoding='utf-8-sig')
base=base[:base.index('## 7. 官方代码缺失和实现边界')]
base+='''## 7. 官方资源：原源码不能发布，但有作者推荐代码基础

2026-10-05核查[作者项目页](https://universal-policy.github.io/unipi/)：Source Code部分明确说明主要实验在Google完成，原源码无法发布，并链接[AVDC](https://github.com/flow-diffusion/AVDC)作为可用于训练的视频建模开源基础。因此不能简单写“没有任何代码入口”。但AVDC README标明它是另一篇*Learning to Act from Actionless Videos through Dense Correspondences*的官方代码（arxiv2310.08576）；不能把AVDC的dense correspondence控制器、配置或成功率当作UniPi原inverse或原实验。

作者网站仓库`universal-policy/unipi`保存的是展示网页/素材。本次保存其完整recursive tree（未截断）和index.html；tree SHA `c93833e4ed865bf7537f7ada959a67f843666a20`仅用于标识网页快照，不是训练代码commit。原UniPi的dataset class、denoiser对象、optimizer入口、checkpoint与评估源码未找到可确认的完整公开版本。核查[记录](<G:/具身/论文阅读/Learning Universal Policies via Text-Guided Video Generation/资料/官方资源核查.json>)与[网页源码](<G:/具身/论文阅读/Learning Universal Policies via Text-Guided Video Generation/资料/官方项目页面源码.html>)已保留。

本次教学实现专注于验证Figure2的三个阶段和实际open-loop执行，不声称重现原模型。作者推荐AVDC是后续做真实视频规划的可用参考；本轮不把相关论文的完整工程引入已选十篇的研究结果。

'''
base+=(ROOT/'UniPi补充正文.md').read_text(encoding='utf-8')
base+=f'''

## 13. 实际运行结果：画面误差与执行误差分别报告

1801次video更新、801次midpoint更新、801次inverse更新；video/midpoint/inverse参数分别为{r['video_parameters']:,}、{r['midpoint_parameters']:,}、{r['inverse_parameters']:,}。视频与补帧Adam lr=.001，inverse Adam lr=.003，batch分别32轨迹/32轨迹（展开128对端点）/128transition，无weight_decay、无EMA。CPU四线程运行约{r['wall_seconds']:.1f}秒，seed83。64保留起点不在512训练轨迹数组内，四个指令种类相同；仍只有toy二维点。

|验证/指标|实际结果|范围和含义|
|---|---|---|
|普通coarse pixel MSE|{r['coarse_pixel_mse']:.9f}|与真实h=2/4/6/8图像对比，含大量背景|
|普通dense pixel MSE|{r['dense_pixel_mse']:.9f}|补帧后8future frames，不能叫FID/FVD|
|inverse新transition控制MSE|{r['inverse_heldout_mse']:.9f}|两个像素动作维度的平均平方误差|
|生成计划实际执行平均终点距离|{r['mean_final_pixel_distance']:.6f}像素|64条环境八步执行，目标移动3.2像素|
|最大终点距离|{r['max_final_pixel_distance']:.6f}像素|不能只看平均值|
|toy成功：距离<.5像素|{r['toy_success_count']}/{r['toy_trials']}（{100*r['toy_success_count']/r['toy_trials']:.3f}%）|阈值在写训练脚本时已设；不是论文任务成功率|
|oracle真实未来视频执行|平均{r['oracle_video_final_distance']:.6f}像素|独立诊断，不是生成计划成功率|
|raw动作被环境限幅的component数|{r['actions_clipped_components']}|此批没有通过限幅改善输出|
|生成coarse像素范围|{r['generated_pixel_min']:.6f}…{r['generated_pixel_max']:.6f}|未用像素clip隐藏越界；moments仅去负权重|
|独立进程strict重载|4阶段max abs error均0|coarse、dense、actions、observations一致|
|采样hook原时间|63,62,…,0，共64次|实际调用时间与预期一致|
|相同first/noise，改相反指令|正确目标MSE={v['opposite_command_correct_target_mse']:.9f}；原指令对相反目标={v['original_command_wrong_target_mse']:.9f}|模型实际响应四种ID条件，不是自然语言理解能力|
|CFG ω=.5|平均{v['CFG_omega_0_5_mean_final_distance']:.6f}像素，{v['CFG_omega_0_5_toy_success_count']}/64成功|同noise加强条件反而更差，不能默认CFG提升执行|
|换初始noise、同保留起点|平均{v['new_noise_mean_final_distance']:.6f}像素，{v['new_noise_toy_success_count']}/64成功|只检验另一个seed，不是全面鲁棒性|

本次机制链路运行通过，但**执行精度不足**。普通pixel MSE约.00019看似小，终点平均仍偏离.73像素，约等于3.2像素期望位移的22.9%。稀疏亮点背景占大多数像素，生成画面中的小伪影/偏移会改变inverse用的重心。oracle视频下误差.022，说明这个batch中inverse对真实图像本身较准确；把剩余误差全部定量分给video或midpoint仍需单独消融，不能只凭oracle差异认定某一模块的唯一责任。

![训练曲线](<{(O/'训练曲线.png').as_posix()}>)

三条曲线各有自己的update横轴；video/midpoint为weighted pixel loss，inverse为action MSE。损失不同量纲，曲线高度不能比较哪一阶段“更好”。video 0步loss约.11423，最后一步约.003035，随机batch/time使单步曲线波动，最后一个batch不是保留评估结果。

![目标、计划与实际执行](<{(O/'视频计划与执行.png').as_posix()}>)

每种指令固定选保留数组中第一条样本，没有挑最好效果；每组三行依次target、生成plan、actual executed。h=0是真实起始帧，h=1…8为环境步；显示vmin0/vmax1只限制色阶。实际执行图由环境状态render生成，并非将生成图直接当作“执行成功”的证据。

![二维轨迹](<{(O/'轨迹.png').as_posix()}>)

同色代表同一种指令；点线target、虚线plan图像重心、实线环境执行重心。图中plan/执行接近而一起偏离target，展示规划画面误差如何传给动作。像素y轴向下，与显示图一致。示例路径不能替代64条汇总统计。

[复跑说明](<{(R/'README.md').as_posix()}>) · [训练脚本](<{(R/'scripts'/'unipi_demo.py').as_posix()}>) · [独立重载与绘图](<{(R/'scripts'/'verify_and_plot.py').as_posix()}>) · [实际结果](<{(O/'结果.json').as_posix()}>) · [重载/指令/CFG对照](<{(O/'权重重载校验.json').as_posix()}>).

## 14. 掌握要求、Paper ↔ Code 和未解决问题

【必须会】视觉规划与动作策略的分工；H个未来帧加首帧形成H个transition；tiling的复制与网络利用条件；diffusion k与环境h的区别；open-loop/重规划；生成图、真实执行图与oracle图的区别；普通pixel MSE/动作MSE/任务成功率分别解释；Table4的surrogate不是真机执行。

【建议会】原文CFG与本教学language-only null的差别；clean/epsilon参数化换算与DDIM最后步；temporal super resolution与简化监督补帧；动作annotation子集与camera单位；生成/真实图像分布差；为什么增强条件可能增大动作误差；原U-Net/T5/TPU预算与baseline设置。

【可以暂时不掌握】T5-XXL/Imagen完整训练、TPU并行、真实机器人IK/夹爪和高分辨率多阶段权重工程。先能沿一个观察解释每个输出的shape、单位与证据，再进入完整训练。

|Paper ↔ Code|精确本地对象或未找到项|复现边界|
|---|---|---|
|§2.2 UPDP|dataset、render、DotEnv、sample|仅二维toy空间，不覆盖通用环境|
|§2.3条件视频扩散|VideoDenoiser.forward、schedule、weighted_pixel_loss、sample|clean参数化/64βschedule为教学自选；官方对象未找到|
|Figure2/§3.1 tiling|VideoDenoiser.forward：first.unsqueeze/expand/cat|实际输入首帧复制；非官方源码|
|§3.1文字条件|VideoDenoiser.text与time|四指令Embedding替代T5；无网络知识迁移|
|§3.1时间层级|Midpoint.forward、densify|粗到密接口实际执行；原扩散TSR对象未找到|
|§3.1 CFG|sample omega=.5对照|只去语言保留first；原双条件unconditional未复现|
|§3.1 extra prior h(τ)|未实现|classifier/Dirac/test-time约束未复现|
|§3.2 inverse|InverseDynamics.forward、moments|学二维control；原卷积7维模型接口未找到|
|§3.2 open-loop|execute → DotEnv.reset/step|真实toy八步执行，不反馈重规划|
|Tables1–3、附录A.4/A.5|原论文数字与训练协议核对|Pybullet、CLIPort、任务组合基准未复跑|
|Table4真实视频指标|未实现FID/FVD/CLIP/surrogate|本地pixel误差不能替代这些指标|
|作者推荐视频codebase|flow-diffusion/AVDC README|确认推荐及相关论文身份；未执行该repo|

**状态：部分成功（教学机制验证），完整论文未复现。** 学习、视频采样、补帧、动作适配、实际toy执行、独立重载与指令对照已完成；精度不足，不能叫完整UniPi成功复现。完整原模型、T5、互联网预训练、仿真7维动作、视频diffusion TSR、classifier prior、composition/跨环境泛化和真机执行都未验证。

未解决：原源码无法发布，具体参数化/官方接口需原作者更多资料；自选weighted clean loss与普通ε训练对执行误差的受控比较尚无；midpoint在生成图分布的误差未单独隔离；moments对伪影、空白、多物体、遮挡不稳健；当前四条指令是离散ID，不能推导自然语言组合理解；没有完整基准或物理动作安全/可执行性证明。所列原文/项目资源的结论置信度较高，toy机制由运行支持，机器人能力结论仍无本地证据。
'''
(ROOT/'精读正文'/'UniPi.md').write_text(base,encoding='utf-8')
(R/'status.json').write_text(json.dumps({'status':'部分成功','scope':'教学像素视频扩散/监督补帧/学习逆动力学/八步open-loop执行与独立重载；精度不足，非官方UniPi实现，完整论文未复现','date':'2026-10-05','full_paper_reproduction':'未复现'},ensure_ascii=False,indent=2),encoding='utf-8')
(R/'requirements.txt').write_text('torch==2.2.0\nnumpy==1.26.4\nmatplotlib==3.10.7\n',encoding='utf-8')
readme=f'''# UniPi 教学机制小 demo

状态：部分成功。视频扩散→监督补帧→学习逆动力学→实际二维环境执行链路完成，执行精度不足；完整论文未复现。[解释.md](<{(P/'解释.md').as_posix()}>)给出原文/教学代码的区别。

作者[官网](https://universal-policy.github.io/unipi/)说明原实验源码无法发布，推荐[AVDC](https://github.com/flow-diffusion/AVDC)作为视频训练代码基础。AVDC是另一篇论文的官方repo；本次未调用它。本地新写教学脚本没有官方训练commit。

## 环境、数据与资源

Windows、Python3.11、torch2.2.0+cpu、numpy1.26.4、matplotlib3.10.7；共享环境 `G:\\具身\\论文阅读\\.venvs\\openvla`，CPU四线程float32，GPU未用。不需要API key、账号、外部数据/权重。约126k总参数，checkpoint不到1MB；fixture含64合成轨迹与结果。实际训练约322秒；保存环境package列表见requirements-lock.json。

数据为16×16灰度Gaussian亮点、四条固定指令、512训练/64新起点测试轨迹。8个动作每步.4像素；planner生成h=2/4/6/8四keyframes。inverse另有4096随机有动作transition。没有真实视频、机器人坐标或自由语言。

## 命令

工作目录 `G:\\具身`。第一条训练并覆盖本篇outputs，第二条只加载已有权重并评价/绘图；路径含空格需引号，&用于运行指定解释器。

```powershell
& 'G:\\具身\\论文阅读\\.venvs\\openvla\\Scripts\\python.exe' '{R/'scripts'/'unipi_demo.py'}'
& 'G:\\具身\\论文阅读\\.venvs\\openvla\\Scripts\\python.exe' '{R/'scripts'/'verify_and_plot.py'}'
```

训练video1801次Adam .001、midpoint801次Adam .001、inverse801次Adam .003。video用64等级linearβ(.001,.2)、clean-pixel weighted loss、15%language drop（first保留），64步DDIM，主结果omega0。补帧为deterministic CNN，非论文diffusion TSR；inverse为pixel moments→MLP二维控制，非论文卷积7controls。DotEnv实际reset/step，8步open-loop，raw/applied动作分别记录，goal只用于metric。

## 实际结果

64保留起点平均终点距离{r['mean_final_pixel_distance']:.6f}像素，预设.5像素阈值成功{r['toy_success_count']}/64；coarse普通pixel MSE={r['coarse_pixel_mse']:.9f}，inverse新transition MSE={r['inverse_heldout_mse']:.9f}。oracle真实未来图执行距离{r['oracle_video_final_distance']:.6f}像素，不能计作生成成功。CFG omega.5误差{v['CFG_omega_0_5_mean_final_distance']:.6f}，更差；换noise误差{v['new_noise_mean_final_distance']:.6f}。raw动作限幅component数0；生成像素仍越界，未clip输出。

独立进程strict加载后，coarse/dense/actions/observations的max差均0；forward hook为63…0。反向指令对正确目标MSE=.000173，原指令对相反目标=.015994，证实有限ID条件产生行为差异。训练loss/生成画面MSE不能推导真实任务成功。

## 产物与限制

outputs/结果.json、训练曲线.json、teaching_state.pt、fixture.pt、权重重载校验.json、重载日志.txt、训练曲线.png、视频计划与执行.png、轨迹.png及交付校验.json；保存训练脚本hash和权重hash供复查。训练标准输出在本次工具运行记录中；结构化训练曲线和最终结果为持久证据，没有伪造原始训练stdout。

未复现：官方1.7B video U-Net/T5-XXL/256TPU训练、原schedule/参数化、diffusion时间超分辨率、7controls/Pybullet/CLIPort、classifier/Dirac prior、语言组合和互联网迁移、FID/FVD/CLIP、真机。toy执行仍失败46/64，不叫任务准确复现成功。进一步精度改进应隔离planner、补帧、inverse误差，不能用后处理或放宽阈值掩盖。
'''
(R/'README.md').write_text(readme,encoding='utf-8')
spec=importlib.util.spec_from_file_location('writer',ROOT/'精读编写.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.write('UniPi',base)
p=P/'解释.md';notes=p.read_text(encoding='utf-8').replace('编写日期：2026-10-02。','初稿：2026-10-02；本篇精读与demo更新：2026-10-05。')
# Put the final mastery/status/limits section after generated map/footer as required by skill.
start=notes.index('## 14. 掌握要求');end=notes.index('## 保存版本的代码地图',start)
closing=notes[start:end];notes=notes[:start]+notes[end:]+closing
p.write_text(notes,encoding='utf-8')
progress=json.loads((ROOT/'逐篇进度.json').read_text(encoding='utf-8'))
for item in progress:
    if item['paper']=='UniPi':item.update(reading='UPDP/视频条件/层级/附录网络和动作协议、官网代码推荐及教学全链路详读完成',demo='教学像素视频扩散、监督补帧、学习逆动力学与实际toy执行/重载部分成功；18/64，精度不足')
(ROOT/'逐篇进度.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
idx=ROOT/'精读索引.md';text=idx.read_text(encoding='utf-8')
text=text.replace('RT-2 已完成教学 demo。','RT-2、UniPi 已完成教学 demo。')
text+=f'\n2026-10-05 UniPi更新：原方法、附录A及作者AVDC推荐已核对；教学像素视频扩散→补帧→学习inverse→实际toy八步执行与重载完成，18/64达到预设阈值，精度不足。见 [运行记录](<{(R/"README.md").as_posix()}>). 完整论文未复现；下一篇为Genie。\n'
idx.write_text(text,encoding='utf-8')
links=re.findall(r'\]\(<([^>]+)>\)',notes)
missing=[x for x in links if not x.startswith('http') and not Path(x).exists()]
assert not missing,missing
pdfhash=hashlib.sha256((P/'原论文.pdf').read_bytes()).hexdigest()
assert pdfhash=='5ded462cfcad30a6317610a4f991d3c1eca7826cbe208dbd9929c3837529708e'
assert hashlib.sha256((R/'scripts'/'unipi_demo.py').read_bytes()).hexdigest()==r['source_sha256']
for f in (R/'scripts').glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
hist=json.loads((O/'训练曲线.json').read_text(encoding='utf-8'))
assert len(hist)==1801+801+801
assert v['strict_reload'] and max(v['reload_max_abs_errors'].values())==0
assert v['opposite_command_correct_target_mse']<v['original_command_wrong_target_mse']
assert all(x in notes for x in ['【必须会】','【建议会】','【可以暂时不掌握】','Paper ↔ Code','未解决','部分成功'])
outaudit={'date':'2026-10-05','links_checked':len(links),'missing_links':missing,'paper_pdf_sha256':pdfhash,
          'training_source_hash_matches':True,'independent_strict_reload':True,'script_syntax':'passed',
          'training_updates_recorded':len(hist),'environment_execution':'64 episodes x 8 steps, synthetic DotEnv',
          'toy_success_count':18,'toy_trials':64,'accurate_task_reproduction':False,'full_paper_reproduction':False,
          'artifact_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [O/'teaching_state.pt',O/'fixture.pt',R/'scripts'/'verify_and_plot.py']}}
(O/'交付校验.json').write_text(json.dumps(outaudit,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'PASS: UniPi notes {len(notes):,} chars, {len(links)} links; actual train/execute/reload checked. Accuracy limited, official full reproduction not done.')
