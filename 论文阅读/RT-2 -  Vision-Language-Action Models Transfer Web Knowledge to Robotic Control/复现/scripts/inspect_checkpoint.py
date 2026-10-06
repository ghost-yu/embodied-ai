"""独立加载已保存的 toy 权重，核对自回归输出与记录；不再训练。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from toy_vla import ToyVLA
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'复现'/'outputs'
model=ToyVLA();model((tf.zeros([1,2,2,3]),tf.zeros([1],tf.int32),tf.zeros([1],tf.int32),tf.ones([1,1],tf.int32)))
model.load_weights(str(OUT/'toy.weights.h5'))
layouts=list(itertools.permutations(range(4),3));np.random.default_rng(23).shuffle(layouts)
def predict(image,color,task):
    previous=np.ones((len(image),1),np.int32);ids=[]
    for field in range(8 if task==0 else 2):
        logits=model((tf.constant(image),tf.constant(color),tf.fill([len(image)],task),tf.constant(previous)),training=False).numpy()[:,-1]
        allowed=np.zeros(272,bool)
        if task==0:
            allowed[16:]=True
            if field==0:allowed[:]=False;allowed[16:18]=True
        elif field==0:allowed[4:8]=True
        else:allowed[2]=True
        y=np.argmax(np.where(allowed[None],logits,-np.inf),axis=-1).astype(np.int32)
        ids.append(y);previous=np.concatenate([previous,y[:,None]],axis=1)
    return np.stack(ids,axis=1)
checks={};examples=[]
for split,subset in [('train',layouts[:18]),('heldout',layouts[18:])]:
    image=[];color=[];expected=[];positions=[]
    for layout in subset:
        im=np.zeros([2,2,3],np.float32)
        for c,pos in enumerate(layout):im[pos//2,pos%2,c]=1
        for c,pos in enumerate(layout):
            image.append(im);color.append(c);positions.append(pos)
            expected.append([16,16+(64 if pos%2==0 else 192),16+(64 if pos//2==0 else 192),144,144,144,144,271])
    image=np.array(image);color=np.array(color,np.int32);expected=np.array(expected)
    act=predict(image,color,0);web=predict(image,color,1)
    result={'robot_exact':float(np.mean(np.all(act==expected,axis=1))),'web_exact':float(np.mean(web[:,0]==np.array(positions)+4))}
    checks[split]=result
    report=json.loads((OUT/'结果.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose(result['robot_exact'],report[split]['robot_all_8_fields_exact'])
    np.testing.assert_allclose(result['web_exact'],report[split]['web_position_exact'])
    if split=='heldout':
        for i in range(3):examples.append({'same_scene':True,'query_color':int(color[i]),'expected_bins':(expected[i]-16).tolist(),'predicted_bins':(act[i]-16).tolist()})
# 验证训练/保留布局不重叠，而非复制样本做评估。
assert not set(layouts[:18]) & set(layouts[18:])
checks['same_scene_different_queries']=examples
checks['weights_sha256']=hashlib.sha256((OUT/'toy.weights.h5').read_bytes()).hexdigest()
checks['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('复现/scripts/*.py')}
(OUT/'checkpoint核对.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2));print('PASS: saved weights reproduce recorded free-running metrics; split layouts are disjoint.')
