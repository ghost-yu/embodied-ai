"""分段获取已经指定的 PyPI wheel，按 PyPI JSON 的 hash 校验后才发布本地文件。"""
import concurrent.futures,hashlib,json,math,subprocess,sys,time
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'G:\具身\论文阅读\依赖缓存');name='scipy-1.13.1-cp311-cp311-win_amd64.whl'
info=json.loads((ROOT/'scipy-1.13.1-pypi.json').read_text(encoding='utf-8'))
record=next(r for r in info['urls'] if r['filename']==name)
size=record['size'];url=record['url'];expected=record['digests']['sha256']
parts=ROOT/'scipy-parts';parts.mkdir(exist_ok=True);step=1024*1024;total=math.ceil(size/step)
def fetch(i):
 start=i*step;end=min(size,(i+1)*step)-1;file=parts/f'{i:04d}.part';length=end-start+1
 if file.exists() and file.stat().st_size==length:return i
 for attempt in range(3):
  result=subprocess.run([r'C:\Windows\System32\curl.exe','--fail','--location','--silent','--show-error','--connect-timeout','10','--max-time','45','--range',f'{start}-{end}','-o',str(file),url],capture_output=True,text=True)
  if result.returncode==0 and file.stat().st_size==length:return i
 raise RuntimeError(f'part {i} failed: {result.stderr[-250:]}')
started=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for n,future in enumerate(concurrent.futures.as_completed([pool.submit(fetch,i) for i in range(total)]),1):
  future.result()
  if n%20==0 or n==total:print(f'completed {n}/{total} chunks; elapsed {time.time()-started:.1f}s',flush=True)
temp=ROOT/(name+'.verified.tmp');digest=hashlib.sha256()
with temp.open('wb') as out:
 for i in range(total):
  data=(parts/f'{i:04d}.part').read_bytes();out.write(data);digest.update(data)
assert temp.stat().st_size==size
assert digest.hexdigest()==expected,(digest.hexdigest(),expected)
temp.replace(ROOT/name)
(ROOT/'scipy-wheel校验.json').write_text(json.dumps({'filename':name,'url':url,'bytes':size,'sha256':expected,'verification':'matched PyPI JSON','chunks':total},indent=2),encoding='utf-8')
print('PASS: wheel size and SHA-256 match the PyPI release metadata.',flush=True)
