from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')
root=Path(r'G:\具身\论文阅读');paper=root/'Multimodal Diffusion Transformer -  Learning Versatile Behavior from Multimodal Goals'
for f in [root/'精读正文'/'MDT.md',paper/'解释.md']:
 s=f.read_text(encoding='utf-8')
 s=s.replace("sample_ddim(model,state,x,goal,sigmas,extra_args=None,callback=None,disable=False,eta=1,...)","sample_ddim(model,state,action,goal,sigmas,scaler=None,extra_args=None,callback=None,disable=None,eta=1.)")
 s=s.replace('循环 10 次，每次用当前正 sigma 调去噪器，callback 接收本次 i/sigma/denoised 等；返回连续动作块，不直接执行机器人。','循环 10 次，每次用当前正 sigma 调去噪器，callback 接收本次 i/sigma/denoised 等；返回连续动作块，不直接执行机器人。action 为初始 `[B,H,D_a]` 噪声动作，state 为条件字典，goal 为目标特征，sigmas 为长度 K+1 的一维 schedule；extra_args 展开后传给模型，disable 控制进度条，scaler 在此方法体中未使用。')
 f.write_text(s,encoding='utf-8')
print('Corrected sampler signature to the saved source defaults.')