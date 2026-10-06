"""核对已有交付物；不改训练记录，不重新训练，不代替论文内容审阅。"""
import ast
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = pathlib.Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

SCOPES = {
    'RT-1': ('官方 FiLM / TokenLearner 模块', '120 次合成训练；手算聚合及官方模块测试', '未运行完整 RT-1；保存的原模型文件含 LFS 指针'),
    'RT-2': ('教学共享 token 模型', '301 次共同训练；留出查询动作和问答各 13/18；权重重载', '非官方模型，未验证网络知识迁移'),
    'OpenVLA': ('官方 HF 类缩小随机配置', '前向、梯度、缓存生成、动作反归一化', '未加载 7B 预训练权重，未进行真实任务评估'),
    'MDT': ('官方动作核心缩小配置', '241 次训练、DDIM 采样及独立重载', '合成条件，未复现 CALVIN 或完整辅助训练任务'),
    'RDT-1B': ('官方 RDTRunner 缩小配置', '201 次训练；128 维、64 步动作块；mask、ACI、采样和重载', '非 1B 预训练模型；未验证双臂真实控制'),
    'pi0': ('官方 JAX 小配置及明确的视觉构造适配器', '501 次训练；flow、mask、cache、Euler 采样及独立参数重载', '随机初始化及合成输入，未复现原机器人结果'),
    'CogACT': ('官方 DiT-S 动作模块', '901 次训练；DDIM、CFG、历史融合及独立重载', '合成采样误差仍较大；未加载完整认知骨干'),
    'UniPi': ('教学视频计划到动作执行链', '视频扩散、补帧、逆动力学、8 步 toy 执行及重载；18/64 成功', '原源码不可发布；教学执行精度不足'),
    'Genie': ('第三方 GenieRedux 小配置', 'tokenizer / LAM / dynamics 训练、VQ 权重对照、用户码生成及重载', '潜在动作语义未证明；第三方采样循环存在正文列明的限制'),
    'I-JEPA': ('官方 encoder / predictor / mask / EMA 小配置', '501 次训练；梯度、无泄漏、EMA、样本顺序及独立重载', '特征方差缩小；未验证 ImageNet 语义质量'),
}

papers = read_json(ROOT / '下载清单.json')
progress = read_json(ROOT / '逐篇进度.json')
rows, failures = [], []
for paper in papers:
    folder = pathlib.Path(paper['folder'])
    demo, outputs = folder / '复现', folder / '复现' / 'outputs'
    scripts = demo / 'scripts'
    row = {'paper': paper['short'], 'scope': SCOPES[paper['short']][0], 'evidence': SCOPES[paper['short']][1], 'limits': SCOPES[paper['short']][2], 'checks': {}, 'hash_checks': []}
    checks = row['checks']
    pdf = folder / '原论文.pdf'
    checks['pdf_bytes_match'] = pdf.stat().st_size == paper['bytes']
    checks['pdf_sha256_match'] = sha(pdf) == paper['sha256']
    from pypdf import PdfReader
    checks['pdf_pages_match'] = len(PdfReader(pdf).pages) == paper['pages']
    text = (folder / '解释.md').read_text(encoding='utf-8-sig')
    row['notes_chars'] = len(text)
    # 结构检查只验证存在；正文解释、公式含义和结论边界须结合实际精读记录。
    checks['reading_structure'] = all(x in text for x in ('必须会', '建议会', '可以暂时不掌握', '未解决', '复现')) and len(text) > 7000
    checks['readme_requirements_sources'] = all(p.exists() for p in (demo / 'README.md', demo / 'requirements.txt', folder / '资料' / '来源.json'))
    missing, count = [], 0
    for doc in (folder / '解释.md', demo / 'README.md'):
        body = doc.read_text(encoding='utf-8-sig')
        for value in re.findall(r'!?\[[^\]]*\]\((<[^>]+>|[^)]+)\)', body):
            target = unquote(value.strip('<>')).split('#')[0]
            if not target or re.match(r'^(https?://|app:|codex:|mailto:)', target):
                continue
            target = re.sub(r':\d+$', '', target)
            path = pathlib.Path(target)
            if not path.is_absolute():
                path = doc.parent / path
            count += 1
            if not path.exists():
                missing.append(str(path))
    row['local_links_checked'], row['missing_links'] = count, missing
    checks['local_links_exist'] = not missing
    files = list(scripts.glob('*.py'))
    for path in files:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    row['scripts_parsed'] = len(files)
    checks['demo_scripts_syntax'] = bool(files)
    checkout = pathlib.Path(paper['checkout']) if paper.get('checkout') else None
    if checkout:
        head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
        row['commit'] = head
        checks['commit_matches'] = head == paper.get('commit', paper.get('third_party_commit'))
        dirty = subprocess.check_output(['git', '-C', str(checkout), 'diff', '--name-only', 'HEAD'], text=True).strip()
        checks['tracked_repository_files_unchanged'] = not dirty
    for report in outputs.glob('*.json'):
        data = read_json(report)
        if not isinstance(data, dict):
            continue
        for key in ('source_sha256', 'artifact_sha256'):
            hashes = data.get(key)
            if not hashes:
                continue
            if isinstance(hashes, str):
                candidates = [path for path in files if sha(path) == hashes]
                row['hash_checks'].append({'report': report.name, 'key': key, 'match': bool(candidates)})
                continue
            for name, expected in hashes.items():
                rel = pathlib.Path(name)
                candidates = [outputs / rel, scripts / rel, demo / rel]
                if checkout:
                    candidates.insert(0, checkout / rel)
                existing = [path for path in candidates if path.is_file()]
                matched = any(sha(path) == expected for path in existing)
                row['hash_checks'].append({'report': report.name, 'path': name, 'match': matched})
        if data.get('weights_sha256'):
            checks['RT2_checkpoint_hash_matches'] = sha(outputs / 'toy.weights.h5') == data['weights_sha256']
    checks['saved_source_artifact_hashes_match'] = bool(row['hash_checks']) and all(x['match'] for x in row['hash_checks'])
    if paper['short'] == 'OpenVLA':
        loaded = read_json(outputs / '结果.json')['loaded_official_sources']
        row['loaded_official_sources'] = loaded
        checks['actual_loaded_HF_sources_match'] = all(pathlib.Path(value['path']).is_file() and sha(pathlib.Path(value['path'])) == value['sha256'] for value in loaded.values())
    main_report = outputs / ('模块结果.json' if paper['short'] == 'RT-1' else '结果.json')
    checks['execution_result_exists'] = main_report.exists()
    reload_reports = list(outputs.glob('*重载校验.json')) + list(outputs.glob('checkpoint核对.json'))
    row['independent_reload_reports'] = [x.name for x in reload_reports]
    row['original_full_paper_reproduction'] = '未复现'
    row['passed'] = all(checks.values())
    if not row['passed']:
        failures.append({'paper': paper['short'], 'failed_checks': [key for key, value in checks.items() if not value], 'missing_links': missing, 'bad_hashes': [x for x in row['hash_checks'] if not x['match']]})
    rows.append(row)

report = {'date': '2026-10-05', 'scope': '十篇论文详细精读与小 demo 交付；排除自动驾驶；未制作整篇翻译', 'papers_count': len(rows), 'all_passed': len(rows) == 10 and not failures, 'checks_boundary': '检查当前 PDF、commit、源码/工件哈希、链接、语法和已保存运行证据；不重复训练，不将结构检查当作科学结论验证。', 'papers': rows, 'failures': failures}
(ROOT / '十篇交付核验.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
lines = ['# 十篇交付核验', '', '核验日期：2026-10-05。范围为详细精读和小 demo；不包含自动驾驶，也未制作整篇翻译。', '', '检查当前 PDF 的字节数、页数与 SHA-256，代码 commit 和受跟踪源码改动，保存的源码/权重/fixture 哈希，本地文档链接与 Python 语法。运行数值来自各篇保存的实际实验和独立重载记录，本次不重复训练。结构检查不能替代论文内容审阅或科学结论验证。', '', '| 论文 | demo 范围 | 已有运行证据 | 主要限制 | 核验 |', '|---|---|---|---|---|']
for row in rows:
    lines.append(f"| {row['paper']} | {row['scope']} | {row['evidence']} | {row['limits']} | {'通过' if row['passed'] else '需修复'} |")
lines += ['', f"共核验 {sum(x['local_links_checked'] for x in rows)} 个本地文档链接、{sum(x['scripts_parsed'] for x in rows)} 个 demo Python 文件；详细逐项结果见 [核验 JSON](十篇交付核验.json)。", '', '十篇原论文的完整数据、规模和机器人/正式基准结果均为 **未复现**。RT-2 与 UniPi 为教学实现，Genie 为第三方实现，其余仅使用正文说明的官方模块与小配置。小 demo 说明机制和代码行为，不证明原论文的泛化能力。', '', 'RT-1 与 OpenVLA 的范围是模块/前向验证，不声称完成独立整模型权重重载。其余八篇的独立重载证据保存在各自 outputs 中。', '', '重新核验：使用已安装 pypdf 的 Python 运行 `十篇交付核验.py`。']
if failures:
    lines += ['', '失败项：', '```json', json.dumps(failures, ensure_ascii=False, indent=2), '```']
(ROOT / '十篇交付核验.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'all_passed': report['all_passed'], 'failures': failures, 'papers': [{'paper': x['paper'], 'notes_chars': x['notes_chars'], 'links': x['local_links_checked'], 'hashes': len(x['hash_checks'])} for x in rows]}, ensure_ascii=False, indent=2))
sys.exit(0 if report['all_passed'] else 1)
