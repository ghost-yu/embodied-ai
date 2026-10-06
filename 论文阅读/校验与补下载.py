import hashlib, json, re, urllib.request
from pathlib import Path
from pypdf import PdfReader

root=Path(__file__).resolve().parent
results=json.loads((root/'下载清单.json').read_text(encoding='utf-8'))
for r in results:
    if r['pdf_status']=='成功': continue
    folder=Path(r['folder'])
    for url in ['https://arxiv.org/pdf/'+r['arxiv'], 'https://export.arxiv.org/pdf/'+r['arxiv']]:
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60) as resp:
                data=resp.read()
            if not data.startswith(b'%PDF-'): raise ValueError('Not PDF')
            pdf=folder/'原论文.pdf'; pdf.write_bytes(data)
            reader=PdfReader(pdf)
            pages=[p.extract_text() or '' for p in reader.pages]
            text='\n\n'.join(f'--- PDF page {i+1} ---\n'+p for i,p in enumerate(pages))
            (folder/'资料'/'原论文文本.txt').write_text(text,encoding='utf-8')
            versions=re.findall(re.escape(r['arxiv'])+r'v(\d+)',text)
            r.update(pdf_status='成功',pdf_url=url,bytes=len(data),pages=len(pages),sha256=hashlib.sha256(data).hexdigest(),version='v'+versions[0] if versions else 'PDF内未找到版本号')
            r.pop('error',None)
            print(r['short'],'recovered',len(pages),flush=True)
            break
        except Exception as e: print(r['short'],str(e),flush=True)
    (folder/'资料'/'来源.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'下载清单.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
for r in results:
    pdf=Path(r['folder'])/'原论文.pdf'
    if pdf.exists():
        reader=PdfReader(pdf)
        print(r['short'],len(reader.pages),pdf.stat().st_size,flush=True)
assert all(r['pdf_status']=='成功' for r in results), 'Some downloads failed'
