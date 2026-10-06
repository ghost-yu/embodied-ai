import sys,json,re,ast,hashlib,subprocess,importlib.util
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'G:\具身\论文阅读');P=ROOT/'Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture';R=P/'复现';O=R/'outputs';REPO=P/'代码'/'ijepa'
r=json.loads((O/'结果.json').read_text(encoding='utf-8'));v=json.loads((O/'权重重载校验.json').read_text(encoding='utf-8'))
b=ROOT/'精读正文'/'I-JEPA.md';s=b.read_text(encoding='utf-8-sig');s=s[:s.index('## 8. 掌握层次与未解决问题')]
s=s.replace('context和target可见区域避免重叠，','设计上context删除与target的重叠（collator在退化配置下可能放松约束，详见后文），')
s+=(ROOT/'IJEPA补充正文.md').read_text(encoding='utf-8')
s+=f'''

## 12. 官方核心小 demo：实际机制和表征质量分开

本机CPU四线程、Python3.11、torch2.2.0+cpu、float32、seed109；复用openvla环境。直接导入官方VisionTransformer、VisionTransformerPredictor、MaskCollator、helper.init_opt、AllReduce和scheduler，没有改官方源码。encoder64dim/2layers/4heads，predictor32dim/2layers/4heads，输入64×64 RGB、patch8、8×8网格；encoder参数{r['parameters']['encoder']:,}，predictor{r['parameters']['predictor']:,}，另外teacher deepcopy一套encoder。参数计数含固定位置表，不能说全是可训练参数。

数据为256张合成彩色圆/方形图，随机位置、大小、颜色与背景，另16张新seed图作固定mask评估；无ImageNet、自然照片、类别监督或机器人观测。render的一些像素可以略大于1，不使用真实图像标准化；仅mask显示限制[0,1]。501updates、batch8，约{r['wall_seconds']:.1f}秒，优化器/schedule沿官方helper，小配置超参数见上一节。该实验只验证训练机制，不测试下游语义能力。

|验证/指标|实际结果|解释|
|---|---|---|
|固定16图的smooth-L1|{r['initial_validation']['smooth_l1']:.9f} → {r['final_validation']['smooth_l1']:.9f}|teacher随EMA改变；不是固定监督目标的单一拟合问题|
|feature prediction MSE|{r['initial_validation']['prediction_mse']:.9f} → {r['final_validation']['prediction_mse']:.9f}|与smooth-L1不同；不是像素MSE|
|保留context形状|{r['shapes']['context']}|16图、10可见patch、64维feature|
|prediction/target形状|{r['shapes']['prediction']}|四targets×16图、每target12patch、64维|
|context/target交集|501batch总0|此配置没有检测到重叠，不覆盖collator fallback|
|修改target像素后的context最大差|0|nonoverlap patch gather防止这条路径泄漏|
|EMA独立算式最大差|0|每步第一trainable matrix；按新optimizer权重更新teacher|
|target loss梯度|全部None|no_grad/requires_grad False生效|
|context patch projection梯度|{r['initial_gradients']['patch_embed']:.9f}|初次backward有限非零|
|predictor mask token梯度|{r['initial_gradients']['mask_token']:.9f}|位置查询共享token收到梯度|
|predictor output projection梯度|{r['initial_gradients']['predictor_proj']:.9f}|预测head实际学习|
|独立进程strict重载|prediction/target最大差0|固定fixture输出一致，非训练resume|

**低loss不能据此证明语义表征已学好。** target patch feature在图像维的平均std从{r['initial_validation']['target_feature_std_across_images_mean']:.6f}降至{r['final_validation']['target_feature_std_across_images_mean']:.6f}，约缩小76.0%；平均池化feature的跨图std从{r['initial_validation']['pooled_feature_std_across_images_mean']:.6f}降至{r['final_validation']['pooled_feature_std_across_images_mean']:.6f}。特征尚非严格全相同，但更相似，存在表示退化风险。没有线性probe/分类/检索就不能只用非零std宣布“未collapse”或“语义成功”。

独立诊断用seed109重建初始teacher，固定同16图/masks：最终prediction对初始teacher的smooth-L1={v['target_motion_diagnostics']['prediction_vs_initial_frozen_target_smooth_l1']:.6f}，最终/初始teacher feature MSE={v['target_motion_diagnostics']['final_vs_initial_target_mse']:.6f}，证明teacher明显漂移；它不是原loss训练中的固定监督。最终teacher的全局mean feature作为oracle常量baseline，smooth-L1={v['target_motion_diagnostics']['oracle_global_mean_final_target_baseline_smooth_l1']:.6f}，比实际predictor的.001121高；predictor并非只精确输出同一全局均值。但这个baseline使用测试target自身均值，只是诊断，不能当独立泛化分类baseline，更不能排除所有形式的collapse。

![官方小配置训练与EMA](<{(O/'训练曲线.png').as_posix()}>)

图左每update随机mask/batch的train loss，中央实际lr scheduler，右EMA momentum。loss与最后固定评估不是相同batch，不能以曲线最后一点当测试性能。没有修改mask、loss或后处理来美化评估。

![完整target图、可见context与四块目标](<{(O/'多块mask.png').as_posix()}>)

固定第一个保留样本：target encoder看最左完整图，context attention只看第二图亮起的可见patch，四个右图为各target索引。某target只含背景仍可成为预测任务，四target可部分重叠；这不是四个独立裁剪视图。最底/最右patch行列未采，是前述randint边界的实际后果。图没有生成缺失像素，不是论文RCDM结果。

[复跑说明](<{(R/'README.md').as_posix()}>) · [官方核心训练脚本](<{(R/'scripts'/'ijepa_demo.py').as_posix()}>) · [独立重载与绘图](<{(R/'scripts'/'verify_and_plot.py').as_posix()}>) · [实测结果](<{(O/'结果.json').as_posix()}>) · [重载与teacher漂移诊断](<{(O/'权重重载校验.json').as_posix()}>).

## 13. 掌握要求、Paper ↔ Code、复现状态和未解决问题

【必须会】预测表征而非像素/动作；full-image target编码后选块；context tokens先gather再attention；固定位置与learnable mask token；多target/context沿batch展开顺序；EMA与stop-grad的分工；loss下降、teacher漂移和下游语义质量分开。

【建议会】collator随机数/几何round/截断/fallback；smooth-L1与论文L2差异；官方lr/wd分组和schedule；checkpoint与resume不同；multi-block/target output masking/predictor消融的评估协议；RCDM另训decoder；方形固定分辨率路径之外的position interpolation问题。

【可以暂时不掌握】多卡DDP/bfloat16/SLURM、完整ImageNet数据管道、所有ViT初始化和下游评估脚本。先能对一张图说清encoder和predictor各看见什么、为何没有真实target特征输入。

|Paper ↔ Code|官方快照/本地调用|证据与边界|
|---|---|---|
|Figure3 context/target|VisionTransformer.forward、target_features|实际full target+masked context，合成数据|
|Figure4 multi-block|MaskCollator.__call__/_sample_block_mask|实际四target，0交集；fallback及边界限制明确|
|predictor|VisionTransformerPredictor.forward|实际64→32→64，mask_token梯度、输出shape|
|论文L2|src/train.py loss_fn|实际smooth-L1，单进程AllReduce早返回|
|EMA|src/train.py动量式、本地train循环|501updates、无target梯度、第一矩阵独立数值校验|
|optimizer/schedule|helper.init_opt、utils/schedulers.py|官方param groups/lr/wd，缩小自选预算|
|多mask身份|apply_masks、repeat_interleave_batch|B2/C2/M4手工身份顺序验证；非所有组合穷举|
|checkpoint|state_dict保存/独立strict重载|实际prediction/target差0，原resume入口未跑|
|Tables1–15|原文主表/附录消融|数字/公平性解释，未复跑ImageNet/transfer|
|Figure6–8|附录RCDM|原另训diffusion decoder未运行；demo不生成像素|

**状态：部分成功（官方核心小配置机制验证），完整论文未复现。** 实际前向、反向、multi-block采样、无像素泄漏测试、EMA/顺序核对、501次合成训练与独立重载已完成。没有原ViT-H/14权重、ImageNet训练、线性/low-shot/full fine-tune、RCDM、机器人action/world rollout，因此不宣称论文效果或有效语义表征已复现。

未解决：真实自然图像与下游评测仍未运行；teacher std明显下降的成因与语义影响未受控隔离；mask fallback可能引入重叠，边界采样不含最后合法位置；异分辨率位置插值与非方形grid没测试；完整resume/RNG、多worker计数同步/AMP/DDP没有验证；原L2/linear-wd描述与保存smooth-L1/cosine-wd的版本历史未追溯。官方快照的运行机制置信度较高，合成学习结果不支持机器人能力、真实图像语义或严格不collapse保证。
'''
b.write_text(s,encoding='utf-8')
(R/'status.json').write_text(json.dumps({'status':'部分成功','scope':'官方小配置encoder/predictor/MaskCollator/optimizer/EMA、501更新与重载；合成图像特征方差缩小，语义质量未验证，完整论文未复现','date':'2026-10-05','full_paper_reproduction':'未复现'},ensure_ascii=False,indent=2),encoding='utf-8')
(R/'requirements.txt').write_text('torch==2.2.0\nnumpy==1.26.4\nmatplotlib==3.10.7\nPyYAML==6.0.2\n',encoding='utf-8')
readme=f'''# I-JEPA 官方核心小 demo

状态：部分成功。官方小配置模型、multi-block mask、optimizer/scheduler、EMA、501次合成训练与重载通过；语义质量未验证，完整论文未复现。详见 [解释.md](<{(P/'解释.md').as_posix()}>).

## 来源、环境与资源

官方facebookresearch/ijepa固定commit `{r['repo_commit']}`，source未修改。Windows、Python3.11.0、torch2.2.0+cpu、numpy1.26.4，CPU四线程float32；复用 `G:\\具身\\论文阅读\\.venvs\\openvla`。无API/key、GPU、真实ImageNet或大权重下载；环境package快照见requirements-lock.json。

encoder116,544、predictor31,744参数（含固定位置表），teacher另拷贝encoder；checkpoint约1.1MB量级。合成256张64×64 RGB圆/方形图，16新seed图测试，无类别labels；约10秒训练，不是原ImageNet模型训练时间。

## 复跑命令

工作目录 `G:\\具身`；&运行指定解释器，带空格路径保持引号。第一条重新训练并覆盖outputs，第二条独立加载/诊断/绘图。

```powershell
& 'G:\\具身\\论文阅读\\.venvs\\openvla\\Scripts\\python.exe' '{R/'scripts'/'ijepa_demo.py'}'
& 'G:\\具身\\论文阅读\\.venvs\\openvla\\Scripts\\python.exe' '{R/'scripts'/'verify_and_plot.py'}'
```

## 设置与结果

官方VisionTransformer embed64/depth2/heads4、patch8、grid8×8；官方predictor width32/depth2/heads4。一context/四targets，原scale/aspect范围保留，min_keep缩为4。CPU无DDP/AMP；loss用源码smooth-L1和官方AllReduce单进程路径。helper.init_opt给AdamW和warmup20→cosine lr(.0001→.001→.00001)、cosine wd(.04→.1)；batch8/501updates，EMA.996→约.999992。

固定eval smooth-L1 .44948265→.00112139，feature MSE .99950129→.00224277；输出[64,12,64]，context[16,10,64]。目标无梯度，context/learnable mask-token/output head有梯度；501batch overlap0；target像素修改后context差0；第一trainable matrix EMA独立算式每步差0；B2/C2/M4身份顺序验证。新进程严格重载prediction/target差0。

target跨图feature std .11709→.02806，pooled std .03390→.00838，表征变相似。最终预测对初始固定teacher smooth-L1 .81610，teacher明显漂移；不能把最终loss下降当固定目标拟合或语义成功。对最终teacher均值的oracle常量baseline .005278大于predictor .001121，仅说明不是简单全局均值，不排除语义退化。

## 边界与产物

MaskCollator fallback可能放松target排除、randint漏最后合法坐标；实际本配置未出现重叠。不是完全保证所有config无泄漏。输入非ImageNet分布、mask几何小网格离散化、teacher方差缩小，未做linear probe/分类/检索。源码L2 vs smooth-L1、linear vs cosine wd、异分辨率位置插值差异已在正文说明。

outputs保存结果/训练曲线、small_core_state.pt、fixture.pt、权重重载校验、两张图和交付校验；复现/运行日志.txt为本次实际stdout。权重不含optimizer/RNG/counter，支持推理重载，不支持训练resume。未运行官方train.py全入口、真实数据、全部下游/RCDM或机器人任务；完整论文未复现。
'''
(R/'README.md').write_text(readme,encoding='utf-8')
spec=importlib.util.spec_from_file_location('writer',ROOT/'精读编写.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for f in ['src/utils/schedulers.py','src/utils/distributed.py']:
    if f not in m.ROLES['I-JEPA']:m.ROLES['I-JEPA'].append(f)
m.write('I-JEPA',s)
p=P/'解释.md';notes=p.read_text(encoding='utf-8').replace('编写日期：2026-10-02。','初稿2026-10-02；本篇精读与demo更新2026-10-05。')
start=notes.index('## 13. 掌握要求');end=notes.index('## 保存版本的代码地图',start);closing=notes[start:end];notes=notes[:start]+notes[end:]+closing;p.write_text(notes,encoding='utf-8')
progress=json.loads((ROOT/'逐篇进度.json').read_text(encoding='utf-8'))
for item in progress:
    if item['paper']=='I-JEPA':item.update(reading='mask几何/fallback/样本身份、官方loss/optimizer/EMA、附录消融与RCDM已详读',demo='官方小配置501更新、mask/EMA/梯度/无泄漏和重载部分成功；特征方差缩小，语义质量未验证')
(ROOT/'逐篇进度.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
links=re.findall(r'\]\(<([^>]+)>\)',notes);missing=[x for x in links if not x.startswith('http') and not Path(x).exists()];assert not missing
assert hashlib.sha256((P/'原论文.pdf').read_bytes()).hexdigest()=='eddbdc093eb4d48662bcf4fbd1c6735ccd606205cc9070977ce317d68aff3941'
for path,digest in r['source_sha256'].items():assert hashlib.sha256((REPO/path).read_bytes()).hexdigest()==digest
assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
assert len(json.loads((O/'训练曲线.json').read_text(encoding='utf-8')))==501
assert max(v['max_abs_error'].values())==0 and r['context_target_overlap_index_count']==0
for f in (R/'scripts').glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
qa={'date':'2026-10-05','local_links_checked':len(links),'missing_links':missing,'source_hashes_match':True,'repo_clean':True,'updates':501,'strict_reload':True,
    'semantic_representation_quality_proven':False,'complete_paper_reproduction':False,
    'artifact_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [O/'small_core_state.pt',O/'fixture.pt',*list((R/'scripts').glob('*.py'))]}}
(O/'交付校验.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'PASS: I-JEPA {len(notes):,} chars, {len(links)} links; actual mechanisms/reload verified; semantic quality unproven.')
