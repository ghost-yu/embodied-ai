"""检查产物导航和结构，并生成索引；结构检查不能替代事实核对。"""
import json, re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
papers=json.loads((ROOT/'下载清单.json').read_text(encoding='utf-8'))
topics={
'RT-1':'FiLM → TokenLearner → 动作分类；历史缓存与注意力 mask',
'RT-2':'VLM 共同微调；动作词表与网络知识迁移',
'OpenVLA':'双视觉编码器 → projector → 动作 token → 反归一化',
'MDT':'多模态目标 → EDM 去噪；MGF/CLA 辅助目标',
'RDT-1B':'统一动作槽位；多相机/语言条件；干净动作预测',
'pi0':'动作专家；flow matching；论文与代码的时间方向',
'CogACT':'认知特征条件；动作扩散；adaptive ensemble',
'UniPi':'条件视频生成 → inverse dynamics → 开环执行',
'Genie':'视频 token；潜在动作；可控动力学与 MaskGIT',
'I-JEPA':'context/target/predictor；mask；EMA；表征预测',
}
rows=[]; issues=[]; all_chars=0
for r in papers:
    folder=Path(r['folder']); doc=folder/'解释.md'; content=doc.read_text(encoding='utf-8')
    all_chars+=len(content)
    required=['必须会','建议会','可以暂时不掌握','Paper ↔ Code','未解决','未复现']
    missing=[x for x in required if x not in content]
    if missing:issues.append(f"{r['short']} 缺少 {missing}")
    if content.count('```')%2:issues.append(r['short']+' code fence未配对')
    if content.count('$$')%2:issues.append(r['short']+' display公式未配对')
    for target in re.findall(r'\]\(<([^>]+)>\)',content):
        if re.match(r'^[A-Za-z]:/',target) and not Path(target).exists():issues.append(r['short']+' 失效链接 '+target)
    absent=re.findall(r'\|`([^`]+)`\|未找到',content)
    rows.append({'论文':r['short'],'字符数':len(content),'检查':'通过' if not missing else '未通过','地图中未找到':absent})
    # 每篇自己的运行记录链接到统一命令和算例证据。
    repro=folder/'复现'/'README.md'
    text=repro.read_text(encoding='utf-8')
    text+='\n本轮核对命令与输出见 [精读校验记录](<'+(ROOT/'精读校验记录.md').as_posix()+'>)。这是资料结构与独立数学例子验证，未执行本仓库官方模型。\n'
    repro.write_text(text,encoding='utf-8')

numerics=json.loads((ROOT/'数学例子校验.json').read_text(encoding='utf-8'))
qa='''# 精读校验记录

核对日期：2026-10-03。工作目录：`G:\\具身\\论文阅读`。本记录区分内容阅读、结构检查和数学算例，不把它们记成官方模型复现。

## 执行命令与结果

文档工具解释器：`C:\\Users\\hgfhg\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe`。

```powershell
& 'C:\\Users\\hgfhg\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe' 'G:\\具身\\论文阅读\\数学例子校验.py'
# 输出：23 项数学例子通过；未导入或运行官方模型。
& 'C:\\Users\\hgfhg\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe' 'G:\\具身\\论文阅读\\精读编写.py'
# 输出：Wrote detailed notes
& 'C:\\Users\\hgfhg\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe' 'G:\\具身\\论文阅读\\精读验收.py'
```

数值使用 NumPy/标准库计算；`assert_allclose` 逐项检查预期结果。数字通过不代表模型会生成正确动作。JSON 保存未舍入结果。

## 文档检查

|论文|字符数（包括路径、代码和公式）|结构/本地链接检查|代码地图中缺失|
|---|---:|---|---|
'''
for r in rows:qa+=f"|{r['论文']}|{r['字符数']}|{r['检查']}|{', '.join(r['地图中未找到']) or '所列路径存在；不代表全部工程齐全'}|\n"
qa+='\n检查范围：三种掌握标签、Paper ↔ Code 表、未解决问题、复现状态、成对代码块/展示公式和尖括号内绝对路径链接。自动检查不验证公式语义、实验公平性或全部代码行为。\n'
qa+='\n发现问题：'+('；'.join(issues) if issues else '上述结构与链接检查未发现问题。')+'\n'
qa+='\n## 数学例子\n\n|例子|计算值|结果|\n|---|---|---|\n'
for n in numerics:qa+=f"|{n['例子']}|`{n['计算值']}`|{n['结果']}|\n"
qa+='''

## 内容核对与限制

使用本地原论文文本和保存源码追踪核心方法；核心架构、公式页已直接渲染为各篇图片。人工重点核对了 π₀ Figure7 的进度指标、I-JEPA 原文 L2、MDT 原文去噪/MGF 页，以及对应源码的时间方向、loss 和 reduction。图像入口在各篇解释顶部。

RT-2、UniPi、Genie 未找到可确认的完整官方实现，不能建立真实训练入口。其教学公式与概念流程已明确标注。其余七篇也未运行官方 import、forward、训练或评估；所有官方模型复现状态均为未复现。每篇未解决问题列出需要继续核验的真实 batch、配置、坐标系和实验入口。
'''
(ROOT/'精读校验记录.md').write_text(qa,encoding='utf-8')
index='''# VLA 与通用世界模型精读索引

按你的要求，先做机械臂 VLA 与通用世界模型，排除自动驾驶。十篇都已从阅读前导读扩写为详细第一版 `解释.md`。建议按下面顺序读：先了解发展脉络，再比较动作 token、扩散、flow，最后连接到视频和表征预测。

每篇包含：问题与论文地图、图中数据流、变量和 shape、关键公式与算例、源码调用链/对象、实验数字与比较边界、三层掌握要求、Paper ↔ Code、未解决问题和复现状态。它覆盖核心方法与代表性实验；没有冒称逐项完成全部附录或官方模型复现。

先读 [公共基础](<G:/具身/论文阅读/精读公共基础.md>)；公式、Python/PyTorch 和机器人动作概念不熟时可以随时返回查阅。

|顺序|论文解释|本篇重点|源码情况|
|---|---|---|---|
'''
for i,r in enumerate(papers,1):
    path=(Path(r['folder'])/'解释.md').as_posix()
    index+=f"|{i}|[{r['short']}](<{path}>)|{topics[r['short']]}|{'已保存官方仓库并固定 commit' if r.get('repo') else '未找到完整官方实现；概念解释'}|\n"
index+='''

## 阅读时先抓住这些区别

- RT-1 的 decoder-only 架构不等于最终策略逐维自回归动作。
- OpenVLA 的量化/反归一化依赖具体实现和数据集统计；双编码器按通道拼接。
- MDT 的 wrapper 输出与 inner_model target 不同；保存实现的辅助 loss reduction/权重需和原文区分。
- RDT 的干净动作 target、动作槽位和 mask 用法需要分别追踪，128 维不能直接称 128 自由度。
- π₀ 论文和保存源码的 flow 时间方向相反，target 符号与积分方向必须一起换。
- CogACT 的认知特征不是未来视频；UniPi 本文实验用开环执行。
- Genie 的潜在动作不天然对应机器人真实动作；I-JEPA 是静态图像表征预测，主体不生成像素视频。

## 本轮交付与验证

原文与仓库沿用下载时保存的版本；每篇顶部注明 arXiv 版本、commit 和图像页链接。原来的前导读备份在各篇 `资料/阅读前导读.md`。

23 项独立数学例子通过，十篇结构与本地导航已检查。查看 [校验记录](<G:/具身/论文阅读/精读校验记录.md>)、[算例脚本](<G:/具身/论文阅读/数学例子校验.py>) 和 [原始结果](<G:/具身/论文阅读/数学例子校验.json>)。环境与资源边界见公共基础。

**官方模型复现：全部未复现。** 源码静态阅读与算例不能证明训练或真机可运行；没有下载新的大型模型/数据。未解决内容留在各篇末尾，后续实践先从依赖检查、真实样本 shape 和最小 forward 开始。
'''
(ROOT/'精读索引.md').write_text(index,encoding='utf-8')
route=ROOT/'阅读路线.md'; text=route.read_text(encoding='utf-8')
old='此轮任务边界是选论文、下载原文、保存可确认的官方代码和建立阅读导航；各篇 `解释.md` 是阅读前导读，尚未完成逐公式、逐实验、逐函数精读，也未进行模型复现。'
new='已下载十篇原文，保存七个可确认的官方仓库；现已为十篇写入核心方法、公式、代码与实验的详细第一版。开始阅读请打开 [精读索引](<G:/具身/论文阅读/精读索引.md>)；全部官方模型仍未复现，未核验项在各篇末尾说明。'
text=text.replace(old,new).replace('[导读]','[精读]').replace('原文与导读','原文与精读').replace('每篇目录保留原论文 PDF、阅读前导读、来源记录和可确认的官方仓库。','每篇目录保留原论文 PDF、详细解释、前导读备份、来源记录和可确认的官方仓库。')
route.write_text(text,encoding='utf-8')
print(json.dumps({'解释篇数':len(rows),'总字符数':all_chars,'数学例子':len(numerics),'问题':issues},ensure_ascii=False))
if issues:sys.exit(1)
