import ast, json
from pathlib import Path
from PIL import Image, ImageDraw
import pypdfium2 as pdfium

root=Path(__file__).resolve().parent
papers=json.loads((root/'下载清单.json').read_text(encoding='utf-8'))
notes={
'RT-1': ('先看机器人策略怎样组织图像历史、语言条件和离散动作，建立后续 VLA 的参照。', '语言条件图像编码 → TokenLearner → Transformer → 动作 token → 连续动作。', ['语言怎样影响视觉特征？','动作的离散化精度和控制频率有什么关系？','泛化测试究竟改变任务、对象还是背景？'], [('模型入口','transformer_network.py'),('视觉token','tokenizers/image_tokenizer.py'),('压缩token','tokenizers/token_learner.py'),('动作表示','tokenizers/action_tokenizer.py'),('训练代理','sequence_agent.py')]),
'RT-2': ('理解预训练视觉语言知识怎样迁移到机器人动作，重点比较 RT-1 和 OpenVLA。', '图像和文本 → 预训练 VLM → 用 token 表达动作 → 机器人控制。', ['动作如何表示为语言模型输出？','机器人数据和视觉语言数据怎样共同训练？','语义泛化与物理操作成功率有什么区别？'], []),
'OpenVLA': ('作为 VLA 代码主线：论文、模型和微调实现可以互相核对。', '图像 → 视觉特征与投影；指令 → 文本token；VLM生成动作token → 解码 → 反归一化。', ['视觉特征怎样进入语言模型？','ActionTokenizer如何在动作和token之间转换？','微调中哪些token参与loss，哪些参数被更新？'], [('微调入口','vla-scripts/finetune.py'),('预训练入口','vla-scripts/train.py'),('数据与预处理','prismatic/vla/datasets/datasets.py'),('动作离散化','prismatic/vla/action_tokenizer.py'),('动作推理','prismatic/models/vlas/openvla.py'),('融合模型','prismatic/models/vlms/prismatic.py'),('仿真评估','experiments/robot/libero/run_libero_eval.py'),('依赖','pyproject.toml')]),
'MDT': ('对应课程实战，是理解多模态目标条件与动作扩散的实践候选；是否能运行取决于数据、权重和环境。', '观测与语言/图像目标 → 条件表征 → 动作去噪；训练中加入辅助目标。', ['图像目标和语言目标怎样对齐？','扩散模型输出的是噪声、干净动作还是预条件化量？','辅助目标怎样改变训练，推理时是否还需要？'], [('训练入口','mdt/training.py'),('配置','conf/config.yaml'),('Agent','mdt/models/mdtv_agent.py'),('Transformer','mdt/models/networks/mdtv_transformer.py'),('去噪loss','mdt/models/edm_diffusion/score_wrappers.py'),('数据','mdt/datasets/base_dataset.py'),('评估','mdt/evaluation/mdt_evaluate.py')]),
'RDT-1B': ('在 MDT 后读，重点理解双臂动作、动作序列和跨机器人动作表示。', '图像、语言、本体状态与带噪动作块 → RDT → 迭代去噪 → 动作序列。', ['统一动作空间的每个槽位表示什么物理量？','动作mask怎样避免未使用维度影响训练？','一次生成多个动作怎样与闭环控制配合？'], [('入口','main.py'),('训练','train/train.py'),('数据','train/dataset.py'),('物理动作槽位','configs/state_vec.py'),('封装与loss','models/rdt_runner.py'),('Transformer','models/rdt/model.py'),('仿真评估','eval_sim/eval_rdt_maniskill.py'),('依赖','requirements.txt')]),
'pi0': ('在动作扩散之后阅读 flow matching，比较离散动作token、扩散动作和连续流三种建模方式。', '视觉语言条件、本体状态与带噪动作 → 动作专家预测向量场 → 数值积分得到动作块。', ['flow matching 的训练目标是什么？','视觉语言模块和动作专家如何交互？','训练时间变量与推理积分方向是否一致？'], [('训练入口','scripts/train.py'),('配置','src/openpi/training/config.py'),('数据加载','src/openpi/training/data_loader.py'),('原始pi0模型','src/openpi/models/pi0.py'),('模型配置','src/openpi/models/pi0_config.py'),('推理封装','src/openpi/policies/policy.py'),('数据变换','src/openpi/transforms.py'),('依赖','pyproject.toml')]),
'CogACT': ('第二轮与 OpenVLA、π0 对照，观察认知特征如何作为动作生成器条件。', '图像与指令 → VLM认知特征 → 扩散动作模块 → 连续动作。', ['哪些隐藏特征传给动作模块？','认知模块和动作模块的训练边界在哪里？','实验增益来自动作头、数据还是其他设置？'], [('训练入口','scripts/train.py'),('VLA封装','vla/cogactvla.py'),('动作模块','action_model/action_model.py'),('训练策略','training/strategies/base_strategy_cogact.py'),('仿真推理','sim_cogact/cogact_policy.py')]),
'UniPi': ('连接 VLA 与世界模型：看“先生成未来视频，再获得动作”的完整思路。', '当前图像与文字任务 → 视频生成 → 未来图像序列 → inverse dynamics → 动作。', ['视频计划如何变成可执行动作？','生成视频符合文字是否等于物理可执行？','跨环境统一图像表示后，动作空间差异怎样处理？'], []),
'Genie': ('通用世界模型主线，理解没有真实动作标签的视频怎样获得可控的潜在动作。', '视频tokenizer、latent action model与动力学模型共同支持逐帧交互。', ['潜在动作和真实机器人关节动作有什么差别？','推理时怎样选择潜在动作并生成下一帧？','长时间rollout误差和控制一致性如何评估？'], []),
'I-JEPA': ('用来理解 JEPA 的表征预测思想。它是静态图像方法；不是动作条件的视频动力学模型。', '上下文图像块 → context encoder → predictor；目标图像块 → target encoder → 目标表征。', ['为什么预测特征而不是像素？','目标编码器的stop-gradient和EMA分别做什么？','空间遮挡预测怎样扩展为时序或动作条件预测？'], [('启动入口','main.py'),('训练与loss','src/train.py'),('模型初始化','src/helper.py'),('编码器与predictor','src/models/vision_transformer.py'),('mask采样','src/masks/multiblock.py'),('配置','configs/in1k_vith14_ep300.yaml')]),
}

overview='''# VLA 与通用世界模型论文阅读路线

按你的课程目录筛选，方向为机械臂 VLA 与通用世界模型，排除自动驾驶相关工作。使用 paper-code-deep-reading 技能组织资料。此轮任务边界是选论文、下载原文、保存可确认的官方代码和建立阅读导航；各篇 `解释.md` 是阅读前导读，尚未完成逐公式、逐实验、逐函数精读，也未进行模型复现。

建议先读 RT-1、RT-2 的架构与动机，再精读 OpenVLA → MDT → RDT-1B → π0。世界模型路线可并行读 I-JEPA → Genie → UniPi；UniPi 用来连接未来预测和机器人动作。CogACT 放在第二轮对照。

如果只选六篇：OpenVLA、MDT、π0、I-JEPA、Genie、UniPi。若重点是双臂机械臂，再加入 RDT-1B。

## 选取结果

|论文|对应课程|阅读深度|为什么读|原文与导读|
|---|---|---|---|---|
'''
for r in papers:
    short=r['short']; folder=Path(r['folder']); purpose,flow,questions,roles=notes[short]
    pdfpath=(folder/'原论文.pdf').as_posix(); guidepath=(folder/'解释.md').as_posix()
    overview+=f"|{short}|{r['lesson']}|{r['priority']}|{purpose}|[PDF](<{pdfpath}>) · [导读](<{guidepath}>)|\n"
    s=f"# {r['title']}\n\n本文件是阅读前导读，范围为选题理由、论文地图、代码导航和待核对问题。完整论文与代码精读尚未完成。\n\n## 论文身份与来源\n\n- 原文：[arXiv {r['arxiv']}](https://arxiv.org/abs/{r['arxiv']})；下载版本：{r.get('version','待核验')}。\n- 对应课程：{r['lesson']}。优先级：{r['priority']}。\n- 本地原文：[原论文.pdf](<{pdfpath}>)，{r.get('pages','未知')}页。SHA-256 及下载地址见 `资料/来源.json`。\n"
    if r.get('repo'):
        s+=f"- 官方仓库：[ {r['repo']} ](https://github.com/{r['repo']})。保存 commit：`{r.get('commit','未确认')}`。\n"
    else:
        check={'RT-2':'RT-2 原论文与项目页 https://robotics-transformer2.github.io/；RT-1 仓库不能代替 RT-2 的完整实现。','UniPi':'论文、项目页 https://universal-policy.github.io/ 和 https://github.com/universal-policy/unipi；后者可见内容是网页素材，不认定为完整模型实现。','Genie':'Genie 原论文和 Google DeepMind 论文主页 https://deepmind.google/research/publications/60474/；本轮未找到可确认的完整官方训练代码。第三方 Genie 实现不能标成原论文官方代码。'}[short]
        s+=f"- 完整官方实现：未找到。已检查：{check}\n"
    s+=f"\n## 为什么选这篇\n\n{purpose}\n\n## 论文地图\n\n先读 Abstract/Introduction 明确输入、输出和任务；再按 Method 的模块关系读架构图；最后读实验设置、主结果、消融与限制。附录优先查动作表示、数据处理与超参数。本轮没有逐项核对公式编号和实验数值。\n\n概念数据流：{flow}\n\n阅读时回答：\n\n"+''.join(f'{i+1}. {q}\n' for i,q in enumerate(questions))
    s+='\n## Paper ↔ Code 导航\n\n以下为文件级定位，不能当作已验证的完整执行调用链。对象与行号是对下载 commit 的静态语法扫描结果；仍需沿入口、配置、数据、模型、loss和评估逐项追踪。\n\n|阅读对象|代码位置|顶层类或函数|\n|---|---|---|\n'
    if r.get('code_status')=='成功':
        checkout=Path(r['checkout'])
        for role,rel in roles:
            f=checkout/rel
            if not f.exists(): s+=f'|{role}|未找到：`{rel}`|未核验|\n'; continue
            objs=[]
            if f.suffix=='.py':
                try:
                    tree=ast.parse(f.read_text(encoding='utf-8'))
                    objs=[f'`{n.name}`（L{n.lineno}）' for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.AsyncFunctionDef))][:6]
                except Exception: objs=['静态语法扫描未成功']
            s+=f'|{role}|[{rel}](<{f.as_posix()}>)|'+('、'.join(objs) or '配置/声明')+'|\n'
    else: s+='|模型、loss、训练与评估|未找到可确认的完整官方实现|未找到|\n'
    if short=='pi0': s+='\n注意：保存的是 openpi 当前 commit，其中还包含后续模型。阅读本论文时先锁定 `pi0.py` 与原始 π0 配置，不能把 π0.5、FAST 或其他变体的行为直接归因于原论文。\n'
    if short=='I-JEPA': s+='\n课程“JEPA”没有注明论文版本，这里主动选择 I-JEPA 作为基础。进入动态世界模型研究后，应另选 V-JEPA/动作条件版本；此处没有把它们混称为同一篇论文。\n'
    s+='''
## 变量与形状检查

阅读代码时保持同一张表：变量名、含义、dtype、shape、坐标系、时间索引、变换前后位置。`B` 表示 batch，`T` 表示时间/动作块长度，`D_a` 表示动作维度；这些是导读记号，不替代论文符号。图像常见布局为 `[B,C,H,W]`，动作块常见布局为 `[B,T,D_a]`，必须以实际配置和代码为准。区分观测时间、机器人控制时间与扩散/流模型的时间变量。

## 实验阅读任务

记录数据集、训练/测试划分、任务、机器人、成功率定义、比较方法和消融设置。只在相同协议下比较数值。机器人成功率、语言理解、视频画质和表征评测各自回答不同问题，不能互相替代。本轮没有整理定量实验表。

## 必须掌握

【必须会】解释本篇的输入、输出、条件信息和训练目标；画出上面的数据流；说明动作或预测结果如何被使用。

## 第二轮知识

【建议会】把关键公式的符号、形状和代码对象逐一对齐；追踪训练与推理的差别；审查主要消融是否隔离了作者宣称的因素。

## 可暂缓的工程细节

【可以暂时不掌握】分布式训练、全量预训练、硬件驱动和完整机器人部署。先完成静态调用链，再考虑最小前向或官方 demo。

## 复现状态与未解决问题

状态：**未复现**。本轮只下载 PDF、提取文本并静态定位源文件，没有安装模型依赖、下载权重/数据集或运行训练与推理。代码下载成功不等于运行成功。

未解决：具体数据样本的 shape 与坐标系、关键公式与函数的精确对应、实验公平性、当前硬件能否运行。后续应先检查 GPU/显存、Python/CUDA、数据和权重要求，再选择 import test、最小前向或仿真评估；不直接启动全量训练。
'''
    (folder/'解释.md').write_text(s,encoding='utf-8')
    repro=folder/'复现'; repro.mkdir(exist_ok=True)
    (repro/'README.md').write_text('# 复现记录\n\n状态：未复现。\n\n此次完成资料下载与静态代码定位；未运行模型。未来运行时记录环境、commit、命令、输入、输出、失败原因及结果适用范围。\n',encoding='utf-8')

overview+='''
## 每一阶段做到什么程度

1. RT-1 / RT-2：能画出“图像+语言→动作”的路径，并说明预训练视觉语言知识的作用。第一遍不用逐行读 TensorFlow 工程。
2. OpenVLA：追踪一个样本从预处理到动作 token，再到真实量纲的动作；理解训练 labels 与 loss mask。这是 VLA 代码主线。
3. MDT：区分观测、目标、带噪动作和模型输出；读辅助目标与动作去噪 loss。作为第一个仿真实践候选，先确认 CALVIN 数据、权重和运行环境。
4. RDT-1B / π0：对照统一动作空间与 flow matching，比较动作块、采样步骤和推理延迟。不要用不同机器人实验的成功率直接排名。
5. I-JEPA / Genie：区分特征预测、视频生成、潜在动作与真实动作；解释“没有真实动作标签”对可控性意味着什么。
6. UniPi / CogACT：对照未来视频中介和认知特征条件两种连接方式，说明从预测或语义表示到可执行动作还需要哪些环节。

## 暂缓的课程工作

- 自动驾驶方向按你的要求排除：GAIA、UniScene、OpenDWM/MaskGWM、InstaDrive、OccWorld/OccLLaMA、HERMES、II-World，以及 BEV/Tesla/DriveVLA 驾驶线；没有下载对应论文。
- RoboFlamingo：第二轮补多模态融合与时间信息处理。第一轮优先 OpenVLA 的动作 token 主线。
- RoboDreamer、LAPA、GR 系列、RoboDual：先完成 UniPi、Genie 与动作生成基础，再读变体和连接方式。
- LCB、Marble、NWM：单凭课程缩写无法可靠确定具体论文版本，暂不据此猜论文。
- Transformer、扩散、NeRF/3DGS：课程中的基础知识模块不直接对应一篇指定论文。Transformer 与扩散需要补基础；NeRF/3DGS 可在选定三维世界模型课题后深入。

这些暂缓是为了控制第一轮阅读量，并不代表论文价值低。课程使用的“隐式/显式/分层”可作为讲解框架；阅读论文时还应核对是否真的存在高低层规划与控制，不能仅凭 VLM 和动作头分开就判定为分层策略。

## 怎么使用这些资料

每篇目录保留原论文 PDF、阅读前导读、来源记录和可确认的官方仓库。`资料/来源.json` 保存下载地址、版本、文件大小、页数、SHA-256 和仓库 commit；`下载清单.json` 汇总校验结果。英文 PDF 的文本提取文件供检索使用，不是中文翻译。未找到可靠中文译本，未生成 `中文论文.pdf`。

下载日期：2026-10-02。版本以下载时 arXiv 页面/PDF 内可核验信息为准。所有复现状态均为“未复现”。
'''
(root/'阅读路线.md').write_text(overview,encoding='utf-8')

# Render original first pages for identity and download sanity checks.
sheet=Image.new('RGB',(5*250,2*360),'#e5e7eb'); draw=ImageDraw.Draw(sheet)
for i,r in enumerate(papers):
    folder=Path(r['folder']); imgdir=folder/'图片'; imgdir.mkdir(exist_ok=True)
    doc=pdfium.PdfDocument(str(folder/'原论文.pdf'))
    page=doc[0]; image=page.render(scale=0.7).to_pil().convert('RGB')
    image.save(imgdir/'首页.png'); image.thumbnail((240,330))
    x=(i%5)*250; y=(i//5)*360
    sheet.paste(image,(x+5,y+25)); draw.text((x+5,y+5),r['short'],fill='black')
    page.close(); doc.close()
sheet.save(root/'论文首页校验.png')
print('Guides and first-page renders created:',len(papers))
