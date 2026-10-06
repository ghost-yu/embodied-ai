import concurrent.futures, hashlib, json, os, re, subprocess, time, urllib.request
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
PAPERS = [
 ('RT-1', 'RT-1: Robotics Transformer for Real-World Control at Scale', '2212.06817', 'google-research/robotics_transformer', '2.3', '发展脉络'),
 ('RT-2', 'RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control', '2307.15818', None, '2.5', '发展脉络'),
 ('OpenVLA', 'OpenVLA: An Open-Source Vision-Language-Action Model', '2406.09246', 'openvla/openvla', '2.5', '重点精读'),
 ('MDT', 'Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals', '2407.05996', 'intuitive-robots/mdt_policy', '2.6、2.8', '重点精读与实践候选'),
 ('RDT-1B', 'RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation', '2410.07864', 'thu-ml/RoboticsDiffusionTransformer', '2.7', '重点精读'),
 ('pi0', 'π0: A Vision-Language-Action Flow Model for General Robot Control', '2410.24164', 'Physical-Intelligence/openpi', '4.4', '重点精读'),
 ('CogACT', 'CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation', '2411.19650', 'microsoft/CogACT', '4.4', '第二轮对照阅读'),
 ('UniPi', 'Learning Universal Policies via Text-Guided Video Generation', '2302.00111', None, '3.2', 'VLA与世界模型桥梁'),
 ('Genie', 'Genie: Generative Interactive Environments', '2402.15391', None, '世界模型3.2', '通用世界模型重点'),
 ('I-JEPA', 'Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture', '2301.08243', 'facebookresearch/ijepa', '世界模型3.3', 'JEPA基础；课程未指定具体版本'),
]

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 (paper reading)'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read(), response.geturl()

def paper(item):
    short, title, aid, repo, lesson, priority = item
    folder = ROOT / re.sub(r'[<>:"/\\|?*]', ' - ', title)
    folder.mkdir(parents=True, exist_ok=True)
    assets = folder / '资料'
    assets.mkdir(exist_ok=True)
    result = dict(short=short, title=title, arxiv=aid, repo=repo, lesson=lesson, priority=priority, folder=str(folder))
    try:
        html, _ = fetch('https://arxiv.org/abs/' + aid)
        (assets/'arxiv页面.html').write_bytes(html)
        versions = re.findall(r'\b'+re.escape(aid)+r'v(\d+)\b', html.decode('utf-8','replace'))
        version = 'v'+str(max(map(int,versions))) if versions else ''
        url = 'https://arxiv.org/pdf/'+aid+version
        data, resolved = fetch(url)
        if not data.startswith(b'%PDF-'): raise ValueError('返回内容不是PDF')
        pdf = folder/'原论文.pdf'
        pdf.write_bytes(data)
        reader = PdfReader(pdf)
        text = '\n\n'.join(f'\n--- PDF page {i+1} ---\n'+(page.extract_text() or '') for i,page in enumerate(reader.pages))
        (assets/'原论文文本.txt').write_text(text,encoding='utf-8')
        result.update(version=version,pdf_url=url,resolved_url=resolved,bytes=len(data),pages=len(reader.pages),sha256=hashlib.sha256(data).hexdigest(),pdf_status='成功')
    except Exception as e:
        result.update(pdf_status='失败',error=str(e))
    if repo:
        checkout=folder/'代码'/repo.split('/')[-1]
        checkout.parent.mkdir(exist_ok=True)
        env=os.environ.copy(); env['GIT_LFS_SKIP_SMUDGE']='1'; env['GIT_TERMINAL_PROMPT']='0'
        try:
            p=subprocess.run(['git','-c','core.longpaths=true','clone','--depth','1','https://github.com/'+repo+'.git',str(checkout)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=150,env=env)
            (assets/'代码下载日志.txt').write_text(p.stdout+p.stderr,encoding='utf-8')
            if p.returncode: raise RuntimeError(p.stderr[-1000:])
            sha=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
            files=[str(x.relative_to(checkout)).replace('\\','/') for x in checkout.rglob('*') if x.is_file() and '.git' not in x.parts]
            (assets/'仓库文件列表.txt').write_text('\n'.join(sorted(files)),encoding='utf-8')
            result.update(code_status='成功',commit=sha,checkout=str(checkout))
        except Exception as e:
            result.update(code_status='失败',code_error=str(e))
    else:
        result.update(code_status='未找到可确认的完整官方实现')
    (assets/'来源.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(short, result['pdf_status'], result['code_status'], flush=True)
    return result

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(paper,PAPERS))
    (ROOT/'下载清单.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Finished',len(results),flush=True)
