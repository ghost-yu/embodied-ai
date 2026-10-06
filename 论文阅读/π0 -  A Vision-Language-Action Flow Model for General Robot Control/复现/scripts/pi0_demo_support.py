"""Configuration adapter only; model methods and vision implementation stay official."""
import sys
from pathlib import Path
import jax
from flax import nnx
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'代码'/'openpi'/'src'))
from openpi.models import pi0_config,siglip
CONFIG={'dtype':'float32','paligemma_variant':'dummy','action_expert_variant':'dummy','pi05':False,'max_token_len':4,'action_dim':32,'action_horizon':50}
def build_model(seed=63):
 original=siglip.Module
 def configured_vision(*args,**kwargs):
  assert kwargs['variant']=='So400m/14'
  kwargs['variant']='mu/32'
  return original(*args,**kwargs)
 siglip.Module=configured_vision
 try:model=pi0_config.Pi0Config(**CONFIG).create(jax.random.key(seed))
 finally:siglip.Module=original
 return model