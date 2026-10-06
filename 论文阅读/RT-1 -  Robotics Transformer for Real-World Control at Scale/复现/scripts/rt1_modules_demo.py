"""RT-1 官方 FiLM/TokenLearner 模块 demo；不是完整机器人策略复现。"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import importlib.util
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.stdout.reconfigure(encoding='utf-8')
PAPER = Path(__file__).resolve().parents[2]
REPO = PAPER/'代码'/'robotics_transformer'
OUT = PAPER/'复现'/'outputs'
OUT.mkdir(parents=True, exist_ok=True)

def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

film_source='film_efficientnet/film_conditioning_layer.py'
token_source='tokenizers/token_learner.py'
film_module=load('official_film',film_source)
token_module=load('official_token_learner',token_source)
tf.keras.utils.set_random_seed(17)

# 实际官方层，零初始化必须保持原特征。
features=tf.random.normal([6,9,9,512])
condition=tf.random.normal([6,12]) # 教学条件向量，不是 USE 语言 embedding。
film=film_module.FilmConditioning(512)
conditioned=film(features,condition)
identity_error=float(tf.reduce_max(tf.abs(conditioned-features)))
assert identity_error == 0, identity_error
# 指定 Dense 的 bias 后验证非恒等调节，kernel 仍为 0。
film._projection_mult.bias.assign(tf.fill([512],.1))
film._projection_add.bias.assign(tf.fill([512],.2))
changed=film(features,condition)
np.testing.assert_allclose(changed.numpy(),1.1*features.numpy()+.2,atol=1e-6)

learner=token_module.TokenLearnerModule(num_tokens=8,bottleneck_dim=64,dropout_rate=0.)
tokens=learner(conditioned,training=False)
assert tokens.shape == (6,8,512)
flat=tf.reshape(conditioned,[6,81,512])
scores=learner.mlp(learner.layernorm(flat),is_training=False)
weights=tf.nn.softmax(tf.transpose(scores,[0,2,1]),axis=-1)
manual=tf.einsum('bsi,bid->bsd',weights,flat)
np.testing.assert_allclose(tokens.numpy(),manual.numpy(),atol=1e-6)
weight_error=float(tf.reduce_max(tf.abs(tf.reduce_sum(weights,axis=-1)-1)))
assert weight_error < 1e-6
history=tf.reshape(tokens,[1,48,512])

# 在合成特征上训练选择器：每张图一个明显 marker，两个 token 都学习汇总该位置。
# 只验证梯度和选择学习，未训练 CNN、语言 encoder、Transformer 或机器人动作。
rng=np.random.default_rng(17)
x=rng.normal(0,.2,size=(4,9,9,16)).astype(np.float32)
x[...,0]=rng.uniform(-.1,.1,size=(4,9,9))
positions=[(1,2),(3,4),(5,6),(7,8)]
for b,(r,c) in enumerate(positions):x[b,r,c,0]=8.
target=np.stack([x[b,r,c] for b,(r,c) in enumerate(positions)])[:,None,:]
target=tf.constant(np.repeat(target,2,axis=1))
x=tf.constant(x)
small=token_module.TokenLearnerModule(num_tokens=2,bottleneck_dim=32,dropout_rate=0.)
optimizer=tf.keras.optimizers.Adam(learning_rate=.01)
curve=[]
for step in range(121):
    with tf.GradientTape() as tape:
        pred=small(x,training=True)
        loss=tf.reduce_mean(tf.square(pred-target))
    value=float(loss)
    assert np.isfinite(value)
    curve.append({'step':step,'mse':value})
    if step == 120:break
    gradients=tape.gradient(loss,small.trainable_variables)
    assert all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients)
    optimizer.apply_gradients(zip(gradients,small.trainable_variables))
assert curve[-1]['mse'] < .1*curve[0]['mse'], curve[-1]

checkpoint=REPO/'trained_checkpoints'/'rt1main'/'variables'/'variables.data-00000-of-00001'
header=checkpoint.read_bytes()[:200]
report={
 'scope':'official FiLM/TokenLearner forward + synthetic TokenLearner training; no complete RT-1 policy',
 'python':sys.version,'tensorflow':tf.__version__,'numpy':np.__version__,'os':platform.platform(),
 'device':[d.name for d in tf.config.list_physical_devices()], 'seed':17,
 'commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
 'source_sha256':{r:hashlib.sha256((REPO/r).read_bytes()).hexdigest() for r in [film_source,token_source]},
 'film_identity_max_abs_error':identity_error,'film_nonidentity_check':'passed',
 'input_shape':features.shape.as_list(),'token_shape':tokens.shape.as_list(),'history_visual_shape':history.shape.as_list(),
 'attention_sum_max_error':weight_error,'official_vs_manual_aggregation':'passed',
 'toy_initial_mse':curve[0]['mse'],'toy_final_mse':curve[-1]['mse'],'toy_updates':120,
 'checkpoint_is_lfs_pointer':header.startswith(b'version https://git-lfs.github.com/spec/v1'),
 'checkpoint_pointer_text':header.decode('ascii') if header.startswith(b'version ') else None,
}
(OUT/'模块结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'训练曲线.json').write_text(json.dumps(curve,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
print('PASS: official module forward, attention normalization, gradients, and synthetic training.')
