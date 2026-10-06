"""RT-2 核心 recipe 的教学演示，非官方实现、非论文性能复现。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json,sys,itertools,platform
from pathlib import Path
import numpy as np
import tensorflow as tf
sys.stdout.reconfigure(encoding='utf-8')
tf.keras.utils.set_random_seed(23)
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'复现'/'outputs'; OUT.mkdir(parents=True,exist_ok=True)
V=272; OFFSET=16; BOS=1; EOS=2
# 0..15 留给文本/控制 token；16..271 是动作 token ID；不是 bin ID 本身。
# 教学量化选择等宽区间中心；不声称这就是 RT-2 未公开的端点实现。
def quantize(x):return np.clip(np.floor((np.clip(x,-1,1)+1)*128),0,255).astype(np.int32)
def dequantize(b):return -1+(np.asarray(b)+.5)*2/256
layouts=list(itertools.permutations(range(4),3))
rng=np.random.default_rng(23); rng.shuffle(layouts)
def data(layout_subset):
    images=[]; colors=[]; robot=[]; web=[]
    for layout in layout_subset:
        image=np.zeros((2,2,3),np.float32)
        for color,pos in enumerate(layout):image[pos//2,pos%2,color]=1
        for color,pos in enumerate(layout):
            # 相机坐标的教学平面目标：左右/上下分别 -0.5,+0.5。
            dx=-.5 if pos%2==0 else .5;dy=-.5 if pos//2==0 else .5
            fields=[0,int(quantize(dx)),int(quantize(dy)),int(quantize(0)),int(quantize(0)),int(quantize(0)),int(quantize(0)),255]
            images.append(image);colors.append(color);robot.append(np.array(fields)+OFFSET)
            # VQA 返回自然语言位置 token + EOS，剩余位置 padding，不参加 loss。
            web.append([4+pos,EOS]+[0]*6)
    return np.array(images),np.array(colors,np.int32),np.array(robot,np.int32),np.array(web,np.int32)
train=data(layouts[:18]); heldout=data(layouts[18:])
from toy_vla import ToyVLA
model=ToyVLA(); optimizer=tf.keras.optimizers.Adam(.003)
images,colors,robot,web=train
images=tf.constant(np.concatenate([images,images]));colors=tf.constant(np.concatenate([colors,colors]))
tasks=tf.constant([0]*len(robot)+[1]*len(web),tf.int32)
labels=tf.constant(np.concatenate([robot,web]))
mask=tf.constant(np.concatenate([np.ones_like(robot),np.tile([1,1,0,0,0,0,0,0],(len(web),1))]),tf.float32)
previous=tf.concat([tf.fill([len(labels),1],BOS),labels[:,:-1]],axis=1)
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        logits=model((images,colors,tasks,previous),training=True)
        token_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        per_sample=tf.reduce_sum(token_loss*mask,axis=1)/tf.reduce_sum(mask,axis=1)
        robot_loss=tf.reduce_mean(per_sample[:len(robot)])
        web_loss=tf.reduce_mean(per_sample[len(robot):])
        loss=.5*robot_loss+.5*web_loss
    grads=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    return loss,robot_loss,web_loss
curve=[]
for step in range(301):
    values=train_step()
    row={'step':step,**{name:float(v) for name,v in zip(['mixed_ce','robot_ce','web_ce'],values)}}
    if not all(np.isfinite(row[k]) for k in ['mixed_ce','robot_ce','web_ce']):raise RuntimeError(row)
    curve.append(row)
    if step%50==0:print(json.dumps(row),flush=True)
# step0记录在第一次更新前，step300记录在第301次更新前。
assert curve[-1]['mixed_ce']<curve[0]['mixed_ce']*.2

def generate(image,color,task,constrained=True):
    n=len(image); previous=np.full((n,1),BOS,np.int32); generated=[]
    for field in range(8 if task==0 else 2):
        logits=model((tf.constant(image),tf.constant(color),tf.fill([n],task),tf.constant(previous)),training=False).numpy()[:,-1,:]
        if constrained:
            allowed=np.zeros(V,bool)
            if task==0:
                allowed[OFFSET:]=True
                if field==0:allowed[:]=False;allowed[OFFSET:OFFSET+2]=True
            else:
                allowed[4:8]=True if field==0 else False
                if field==1:allowed[EOS]=True
            logits=np.where(allowed[None,:],logits,-np.inf)
        ids=np.argmax(logits,axis=-1).astype(np.int32)
        generated.append(ids);previous=np.concatenate([previous,ids[:,None]],axis=1)
    return np.stack(generated,axis=1)
def evaluate(dataset):
    im,col,robot_labels,web_labels=dataset
    actions=generate(im,col,0); answers=generate(im,col,1)
    return {'samples':len(im),'robot_all_8_fields_exact':float(np.mean(np.all(actions==robot_labels,axis=1))),
      'web_position_exact':float(np.mean(answers[:,0]==web_labels[:,0])),'robot_tokens_in_bank':bool(np.all((actions>=OFFSET)&(actions<OFFSET+256)))}
# 人为给一个文本 token 更高 logit，显示动作 bank 约束会排除它。
probe=np.zeros(V);probe[4]=20;probe[OFFSET+128]=10
unconstrained=int(np.argmax(probe)); constrained=int(np.argmax(np.where(np.arange(V)>=OFFSET,probe,-np.inf)))
assert unconstrained==4 and constrained==OFFSET+128
sample_image,sample_colors,_,_=heldout
ids=generate(sample_image[:1],sample_colors[:1],0)[0];bins=ids-OFFSET
report={'implementation':'teaching ToyVLA; not official RT-2, PaLI-X, or PaLM-E',
 'environment':{'python':sys.version,'tensorflow':tf.__version__,'numpy':np.__version__,'platform':platform.platform(),'device':'CPU'},
 'seed':23,'updates':301,'shared_parameters':model.count_params(),
 'data':{'train_layouts':18,'heldout_layouts':6,'queries_per_layout':3,'robot_and_web_mix':'.5/.5 per-sample mean CE','physical_world':'none'},
 'train':evaluate(train),'heldout':evaluate(heldout),
 'initial_mixed_ce':curve[0]['mixed_ce'],'final_recorded_mixed_ce':curve[-1]['mixed_ce'],
 'constraint_probe':{'free_id':unconstrained,'action_only_id':constrained},
 'example':{'query_color':int(sample_colors[0]),'image':sample_image[0].tolist(),'action_token_ids':ids.tolist(),'action_bin_ids':bins.tolist(),'terminate':int(bins[0]),'continuous_fields':dequantize(bins[1:]).tolist()},
 'limits':['synthetic images, tiny randomly initialized transformer','no web pretraining or PaLI-X/PaLM-E weights','no causal ablation of co-fine-tuning','no robot environment or physical safety check']}
(OUT/'结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'训练曲线.json').write_text(json.dumps(curve,indent=2),encoding='utf-8')
model.save_weights(str(OUT/'toy.weights.h5'))
print(json.dumps(report,ensure_ascii=False,indent=2));print('PASS: shared token training and constrained autoregressive decoding demo.',flush=True)
