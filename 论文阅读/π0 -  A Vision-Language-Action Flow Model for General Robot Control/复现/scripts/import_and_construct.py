import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'代码'/'openpi'/'src'))
import jax,flax
from flax import nnx
from openpi.models import pi0,pi0_config,siglip
original=siglip.Module
calls=[]
def configured_vision(*args,**kwargs):
 calls.append({'original_variant':kwargs['variant'],'demo_variant':'mu/32'})
 kwargs['variant']='mu/32'
 return original(*args,**kwargs)
siglip.Module=configured_vision
try:
 config=pi0_config.Pi0Config(dtype='float32',paligemma_variant='dummy',action_expert_variant='dummy',pi05=False,max_token_len=4)
 model=config.create(jax.random.key(63))
finally:siglip.Module=original
print('PASS',jax.__version__,flax.__version__,jax.devices(),calls,flush=True)
print(sum(x.size for x in jax.tree.leaves(nnx.state(model,nnx.Param))),flush=True)