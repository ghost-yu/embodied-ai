import sys,json,re,hashlib,importlib.util,ast,subprocess
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'G:\具身\论文阅读');P=ROOT/'Genie -  Generative Interactive Environments';R=P/'复现';O=R/'outputs';REPO=P/'代码'/'GenieRedux'
r=json.loads((O/'结果.json').read_text(encoding='utf-8'));a=json.loads((O/'VQ权重对照.json').read_text(encoding='utf-8'));v=json.loads((O/'权重重载校验.json').read_text(encoding='utf-8'))
audit={'date':'2026-10-05','paper':'2402.15391v1','official_page':'https://deepmind.google/research/publications/60474/',
       'original_complete_official_code':'未找到可确认的完整官方实现','checked':'本地原PDF、DeepMind论文入口、PMLR论文记录、Open-Genie README和GenieRedux README',
       'third_party_repo':'https://github.com/insait-institute/GenieRedux','third_party_commit':r['repo_commit'],
       'scope':'相关研究的复现框架，实际小配置核心使用；不是原Genie官方源码、权重或基准',
       'source_changed':False,'limits':'没有穷尽全网搜索，没有运行原11B/真实互联网训练集、RetroAct/CoinRun或扩展agent。'}
(P/'资料'/'官方与复现资源核查.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
b=ROOT/'精读正文'/'Genie.md';s=b.read_text(encoding='utf-8-sig');s=s[:s.index('## 7. 没有官方实现时保留哪些边界')]
s+='''## 7. 阅读与实现边界

原版Genie公开论文可核对；本轮未找到完整原版官方训练源码。下面代码分析针对第三方GenieRedux的固定快照，明确标注原文与复现的差异，不能称为DeepMind官方行为。资源核查见 [记录](<G:/具身/论文阅读/Genie -  Generative Interactive Environments/资料/官方与复现资源核查.json>)。不混入Genie2/3、自动驾驶或相关论文的新基准结果。

'''
s+=(ROOT/'Genie补充正文.md').read_text(encoding='utf-8')
s=s.replace('主配置使用16frames、10FPS，视频原分辨率160×90；','主配置使用16frames、10FPS，视频原分辨率160×90；源片段为16秒（约160帧），不是16frames训练裁片。')
s+=f'''

## 14. 小 demo 的实测结果与一次 VQ 权重对照

直接导入保存的第三方Tokenizer、LatentActionModel、MaskGIT/Dynamics，源码未改。参数分别 **{r['parameters']['tokenizer']:,} / {r['parameters']['LAM']:,} / {r['parameters']['dynamics']:,}**，总156,414；没有原200M/300M/10.1B。CPU四线程，torch2.2.0+cpu、float32，seed97。初轮训练tokenizer1201、LAM1801、dynamics801 updates，约{r['wall_seconds']:.1f}秒；每步batch16，AdamW lr=.001、weight_decay0、gradient norm clip1，无原warmup/cosine/EMA。checkpoint没有optimizer状态，重新训练不保证跨设备bitwise相同。

第一轮LAM外部VQ权重.25，256个测试transition全部使用code6。重建loss已经下降却没有动作区分，这是明确的codebook collapse；不能拿pixel MSE变小证明无监督动作发现成功。保存第一轮全部权重/fixture/日志后，对照只将LAM外部VQ权重降到.01；tokenizer用原固定权重，LAM/dynamics回到同随机初始化、按原tokenizer minibatch RNG推进后使用同样minibatch序列，各再1801/801updates。其余结构、数据、内部commitment、optimizer、训练量不变，无真实动作标签加入；对照耗时{a['wall_seconds']:.1f}秒。

|指标（64测试视频/256transitions）|VQ .25第一轮|LAM VQ .01对照|解释|
|---|---|---|---|
|tokenizer pixel MSE|{r['tokenizer_pixel_mse']:.9f}|{a['tokenizer_pixel_mse']:.9f}|同一冻结tokenizer|
|LAM teacher-forced pixel MSE|{r['LAM_pixel_mse']:.9f}|{a['LAM_pixel_mse']:.9f}|下降约84.0%，不是自由rollout|
|动作码histogram|{r['LAM_code_histogram']}|{a['LAM_code_histogram']}|由1个使用码到7个；code6在对照未使用|
|事后direction purity|{r['LAM_direction_purity_posthoc']:.6f}|{a['LAM_direction_purity_posthoc']:.6f}|29.69%→56.64%，非独立动作分类泛化准确率|
|dynamics masked CE|{r['heldout_dynamics_masked_CE']:.6f}|{a['heldout_dynamics_masked_CE']:.6f}|同seed mask，在两个模型的不同动作条件下比较|
|推断latent条件的一步pixel MSE|{r['one_step_generated_pixel_mse']:.9f}|{a['one_step_generated_pixel_mse']:.9f}|推断ID时看过真实下一帧；oracle latent条件|
|将ID改为(k+1)%8的一步MSE|{r['different_code_pixel_mse']:.9f}|{a['different_code_pixel_mse']:.9f}|循环替代码，不是paper categorical random|
|toy ΔPSNR，t=1|{r['toy_delta_PSNR_t1_dB']:.6f}dB|{a['toy_delta_PSNR_t1_dB']:.6f}dB|对照变正，不能与原t=4直接比|
|同prompt、八user ID输出最大差|{r['eight_user_codes_output_pair_max_difference']:.6f}|{a['eight_user_codes_output_pair_max_difference']:.6f}|有响应差异，不保证八个稳定物理动作|

direction purity定义为每个码在测试transition的四种方向计数中取最大，再把这些最大数求和除256。它允许同一方向分散到多个码，且“选最多方向”和“计算比例”用同一测试集，所以是描述性关联统计，不是训练映射后在另一个保留集评测的分类accuracy。对照code0的方向计数[5,3,25,23]，down/up仍混在一起；code1为[15,36,0,0]偏left但仍包含right，不能为按钮固定命名并说跨起点一致。

toy ΔPSNR逐样本计算$10\log_{{10}}(MSE_{{different}}/MSE_{{inferred}})$再取64样本平均，clamp最小1e−12防log0。**不等于先对所有样本平均MSE再求log**；主表平均MSE之比与报告dB不必相等。这里只t=1、单步、deterministic different ID，原论文t=4、多步、categorical随机动作，所以它是机制诊断，不能叫Genie ΔPSNR复现结果。第一轮均−.12dB，说明正确推断的坍塌码无优势；对照约+1.61dB支持动作条件开始承载信息，但语义一致性仍不足。

![训练曲线](<{(O/'训练曲线.png').as_posix()}>)

图为第一轮，tokenizer/LAM总损失含pixel+VQ，dynamics是masked CE，三者数值不可当同一单位比较。LAM稳定低loss却全用code6，展示为何还必须检查usage与control。第二轮训练曲线原数据独立保存在VQ对照训练曲线.json，没有覆盖第一次结果。

![两次动作码使用](<{(O/'动作码使用率.png').as_posix()}>)

左.25全部code6，右.01分散到七个码。均匀usage本身也不是正确动作语义的充分证据，应结合direction confusion、生成效果与跨环境测试。

![重建和生成](<{(O/'重建与生成.png').as_posix()}>)

此图显示**.01对照**，固定测试数组第一条，未挑最好样本。target、tokenizer重建、LAM teacher forcing、用户ID0/1/2/3的生成rollout四行；最后一行用户ID序列不是target真实方向，因此不能凭轨迹不同说预测错误或成功。teacher forcing行每步提供真实过去观察与从未来推断的码，不代表自由rollout同样清晰。所有显示vmin0/vmax1只限制色阶，没有裁剪采样tensor或评价输入。

![八个用户动作码对照](<{(O/'用户动作码对照.png').as_posix()}>)

同一个prompt下.01对照的八个ID生成画面有差异，但亮点会模糊、偏移，有些码近似同方向，未使用code6属于训练外码。交互输入仅提示图+user ID，查表后由dynamics生成；没有用真实位置/方向偷偷更新图像。

## 15. 重载、因果和采样问题的验证

第一轮与.01对照都在独立Python进程strict重载。所有fixture tensor、重建、动作IDs、一步/替代码生成、user-code输出与六步rollout重算，最大差均约 **2.98×10⁻⁷**（阈值1e−6），整数IDs一致；不是宣称bitwise全部0。参数/图像路径与源码hash已记录。LAM action projection、codebook、pixel head分别实测gradient norm约{v['LAM_nonzero_gradient_norms']['to_action_emb.1.weight']:.8f}、{v['LAM_nonzero_gradient_norms']['vq._codebook.embed']:.8f}、{v['LAM_nonzero_gradient_norms']['to_pixels_first_frame.0.weight']:.8f}，有限非零。

因果测试在第一轮把最后future frame改为随机，之前visual IDs和LAM action IDs不变，记录passed。不能把它扩大为所有采样路径均无信息泄漏。采样内部Embedding hook实际5次且输入相同、1步/5步token相同；default tuple报`TypeError: 'tuple' object cannot be interpreted as an integer`。这两点限制了对完整MaskGIT能力的判断，详见前节，源码保持原样。

失败记录：最初smoke test把LAM 2×2复制码直接传dynamics 4×4而报expand shape错误，改为lookup后singleton广播；第一次采样诊断在maskgit对象注册Module hook未收到调用，因为源码使用`.forward`，改在实际token_emb调用处采集，重新运行通过。二者为本地调用/观察修正，未改第三方模型源码；同时保留default tuple与collapse等实际问题，没有把它们隐藏成全成功。

[复跑说明](<{(R/'README.md').as_posix()}>) · [核心训练](<{(R/'scripts'/'genie_core_demo.py').as_posix()}>) · [VQ权重对照](<{(R/'scripts'/'vq_weight_ablation.py').as_posix()}>) · [独立重载/绘图](<{(R/'scripts'/'verify_and_plot.py').as_posix()}>) · [原始结果](<{(O/'结果.json').as_posix()}>) · [对照结果](<{(O/'VQ权重对照.json').as_posix()}>) · [重载/采样检查](<{(O/'权重重载校验.json').as_posix()}>).

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
'''
b.write_text(s,encoding='utf-8')
(R/'status.json').write_text(json.dumps({'status':'部分成功','scope':'第三方GenieRedux小配置tokenizer/LAM/dynamics训练、VQ单因素对照、user码生成与重载；动作语义不稳且采样存在限制，非原官方实现，完整论文未复现','date':'2026-10-05','full_paper_reproduction':'未复现'},ensure_ascii=False,indent=2),encoding='utf-8')
(R/'requirements.txt').write_text('torch==2.2.0\neinops==0.8.0\nbeartype==0.19.0\nnumpy==1.26.4\nmatplotlib==3.10.7\n',encoding='utf-8')
readme=f'''# Genie 第三方核心小 demo

**部分成功**：实际GenieRedux小配置完成video tokenizer/LAM/dynamics训练、用户码生成、VQ权重对照与独立重载；潜在动作语义不稳定，采样实现有限制；完整原论文未复现。详见 [解释.md](<{(P/'解释.md').as_posix()}>).

## 来源与环境

原论文arxiv2402.15391v1；非DeepMind官方源码。固定[GenieRedux](https://github.com/insait-institute/GenieRedux) commit `{r['repo_commit']}`，源码保持clean。第三方README/配置不等于原Genie接口/基准，本轮不运行guided真动作模型、RetroAct/AutoExplore。模型直接import，没有伪模块或自写替换网络。

Windows、Python3.11.0、torch2.2.0+cpu、einops.8、beartype.19、numpy1.26.4、matplotlib3.10.7；复用 `G:\\具身\\论文阅读\\.venvs\\pi0`，CPU四线程float32，无新框架安装/大模型权重下载、API/key或真实视频数据。全量环境列表见requirements-lock.json。总156,414参数，两个checkpoint各约.7MB。第一轮约117秒，对照约64秒，不是原论文FPS测量。

## 复跑命令

工作目录 `G:\\具身`，依次训练初轮、进行VQ对照、独立进程验证/绘图。脚本会覆盖本篇outputs的同名产物。

```powershell
& 'G:\\具身\\论文阅读\\.venvs\\pi0\\Scripts\\python.exe' '{R/'scripts'/'genie_core_demo.py'}'
& 'G:\\具身\\论文阅读\\.venvs\\pi0\\Scripts\\python.exe' '{R/'scripts'/'vq_weight_ablation.py'}'
& 'G:\\具身\\论文阅读\\.venvs\\pi0\\Scripts\\python.exe' '{R/'scripts'/'verify_and_plot.py'}'
```

PowerShell的&运行指定解释器；路径空格需要保留引号。脚本sys.path指向保存repo，requirements.txt为此demo直接依赖，不是完整Linux训练环境。需要先有克隆目录，不应调用pip安装同名models/genie替代。

## 配置与数据

512/64合成视频，每条5frames，8×8 periodic Gaussian亮点，四个随机真实移动方向。训练丢弃真实方向，测试labels仅用于事后confusion；无自由语言、平台跳跃或机械臂。

tokenizer32dim/1block/2heads/head16/patch2/visual codebook32×16；LAM同width、patch4、8×16动作码；dynamics同width、num_tokens32、is_guided=False/additive action、use_token=False。初轮VQ外权重.25，LAM对照.01；内部commitment .25不变。AdamW lr.001/wd0/clip1/batch16，tokenizer1201、LAM1801、dynamics801updates；对照冻结原tokenizer后重训LAM/dynamics。没有原warmup/bfloat16/11B或joint recipe。

## 结果

初轮全部256transition为code6，direction purity29.69%，LAM MSE.009008，toy t1 delta PSNR−.12042dB；不能称动作发现成功。对照使用7个码、purity56.64%、LAM MSE.001440，toy t1 delta PSNR+1.61459dB，一步视频MSE.010424。purity是测试集内按code取最多方向的描述统计，非独立泛化classification；delta对照用(k+1)%8而非论文随机category、t1而非t4，不与原表横比。

两套严格重载后最大差约2.98e−7，整数码一致。第一轮future-last-frame扰动不影响此前visual/action IDs；实际LAM action embedding、VQ和pixel head梯度有限非零。对照user ID响应仍不稳定，未训练过的code6也没有可靠意义。

## 实际问题与修正

LAM返回2×2复制码不能直接expand到dynamics4×4；本地action_codes使用原lookup再singleton广播解决，source未修改。采样没有scalar override会触发tuple TypeError；本地显式传5steps/temperature1/cosine避免默认问题。

源码sample拼接待生成帧后[:, :-1]丢掉该帧；single-step hook发现5次token输入完全相同、1/5steps最终码一致。不能说已完整验证MaskGIT逐轮补全。hook最初注册maskgit对象失败，因为source显式.forward；改为实际token_emb调用处后重跑。FFW数量、decoder无sigmoid、mask最初观察等与原文不同，详见正文。

## 产物与未复现范围

outputs有结果.json、VQ权重对照.json、训练曲线.json、VQ对照训练曲线.json，两套state/fixture、权重重载校验.json、源码hash、原始日志和四张图。图中重建/rollout与user ID对照展示.01实验，使用第一条测试样本；usage图同时展示两轮，没有隐藏初轮collapse。

未执行原11B/互联网预训练、RetroAct/CoinRun/BC/真实动作校准、完整25-step原采样、t4 ΔPSNR/FVD、机器人/长期物理记忆。当前小实验部分缓解坍塌但不证明稳定可控；完整论文未复现。源码bug保留并报告，没有修改源码后称原实现通过。
'''
(R/'README.md').write_text(readme,encoding='utf-8')
manifest=json.loads((ROOT/'下载清单.json').read_text(encoding='utf-8'))
for item in manifest:
    if item['short']=='Genie':item.update(third_party_repo='insait-institute/GenieRedux',third_party_commit=r['repo_commit'],checkout=str(REPO),code_status='原完整官方实现未找到；已保存第三方GenieRedux快照并运行小配置核心')
(ROOT/'下载清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
spec=importlib.util.spec_from_file_location('writer',ROOT/'精读编写.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.ROLES['Genie']=['README.md','models/tokenizer.py','models/lam.py','models/dynamics.py','models/genie_redux.py','models/components/stvivit.py','models/components/attention.py','models/components/vector_quantize.py','configs/config/genie_redux.yaml','train_genie_redux.py','data/data.py','training/optimizer.py']
m.write('Genie',s)
p=P/'解释.md';notes=p.read_text(encoding='utf-8').replace('编写日期：2026-10-02。','初稿2026-10-02；本篇精读与demo更新2026-10-05。')
start=notes.index('## 16. 掌握要求');end=notes.index('## 保存版本的代码地图',start);closing=notes[start:end];notes=notes[:start]+notes[end:]+closing
notes=notes.replace('## 保存版本的代码地图','## 保存版本的第三方代码地图')
p.write_text(notes,encoding='utf-8')
progress=json.loads((ROOT/'逐篇进度.json').read_text(encoding='utf-8'))
for item in progress:
    if item['paper']=='Genie':item.update(reading='原LAM/ST/tokenizer/dynamics、数据筛选/附录E/F及第三方VQ/mask/sampling详读完成',demo='第三方小配置训练/因果/user码生成/重载部分成功；初轮collapse，VQ对照部分缓解，语义与采样仍有限制')
(ROOT/'逐篇进度.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
idx=ROOT/'精读索引.md';st=idx.read_text(encoding='utf-8');st+=f'\n2026-10-05 Genie更新：原方法/附录和第三方GenieRedux核心已详读，完成无真实动作标签的小配置三模块训练、VQ权重对照、user码生成/因果/重载。初轮码本坍塌，对照使用7码但语义不稳，采样实现限制已报告。见 [运行记录](<{(R/"README.md").as_posix()}>). 完整原论文未复现；下一篇为I-JEPA。\n';idx.write_text(st,encoding='utf-8')
links=re.findall(r'\]\(<([^>]+)>\)',notes);missing=[x for x in links if not x.startswith('http') and not Path(x).exists()];assert not missing,missing
assert hashlib.sha256((P/'原论文.pdf').read_bytes()).hexdigest()=='94c63a5ced89355326706c88236591c2972ced12f3dece4f8982f020110c81f3'
for path,digest in r['source_sha256'].items():assert hashlib.sha256((REPO/path).read_bytes()).hexdigest()==digest
assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
assert len(json.loads((O/'训练曲线.json').read_text(encoding='utf-8')))==1201+1801+801
assert len(json.loads((O/'VQ对照训练曲线.json').read_text(encoding='utf-8')))==1801+801
assert r['LAM_code_histogram'][6]==256 and sum(x>0 for x in a['LAM_code_histogram'])==7
assert v['strict_reload'] and max(v['max_abs_errors'].values())<1e-6 and max(v['VQ001_reload_max_abs_errors'].values())<1e-6
assert v['all_refinement_network_inputs_equal'] and v['steps_1_vs_5_same_token_ids'] and 'TypeError' in v['upstream_default_sampling_error']
for f in (R/'scripts').glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
assert all(x in notes for x in ['【必须会】','【建议会】','【可以暂时不掌握】','Paper ↔ Code','未解决','部分成功'])
qa={'date':'2026-10-05','notes_chars':len(notes),'local_links_checked':len(links),'missing_links':missing,'source_hashes_match':True,'repo_clean':True,
    'strict_reload_two_runs':'passed tolerance 1e-6','training_updates_recorded':3803+2602,'first_run_collapse':True,'ablation_codes_used':7,
    'stable_latent_action_semantics_proven':False,'complete_MaskGIT_refinement_proven':False,'original_full_paper_reproduction':False,
    'artifact_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in [O/'small_core_state.pt',O/'fixture.pt',O/'vq001_state.pt',O/'vq001_fixture.pt',*list((R/'scripts').glob('*.py'))]}}
(O/'交付校验.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'PASS: Genie {len(notes):,} chars, {len(links)} links; source clean, both trained/reloaded. Stable semantics and original full reproduction NOT proven.')
