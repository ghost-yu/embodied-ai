"""保存训练曲线与单个推理例子；读取结果，不重新训练。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
out=Path(__file__).resolve().parents[1]/'outputs'
r=json.loads((out/'结果.json').read_text(encoding='utf-8'))
curve=json.loads((out/'训练曲线.json').read_text(encoding='utf-8'))
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for key,label in [('mixed_ce','Mixed CE'),('robot_ce','Action CE'),('web_ce','VQA CE')]:
 axes[0].semilogy([x['step'] for x in curve],[x[key] for x in curve],label=label)
axes[0].set(title='Shared toy model training',xlabel='Update index (loss before update)',ylabel='Cross-entropy (log scale)')
axes[0].grid(alpha=.2);axes[0].legend()
axes[1].imshow(np.array(r['example']['image']),interpolation='nearest',extent=(0,2,2,0))
axes[1].set(xticks=[.5,1.5],yticks=[.5,1.5],title='Held-out scene: query red',xlabel='Toy dx: right = +0.5',ylabel='Toy dy: top = -0.5')
axes[1].text(.03,.03,'Predicted bins: '+str(r['example']['action_bin_ids']),transform=axes[1].transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.9))
fig.suptitle('RT-2 recipe teaching demo — synthetic data, no real robot',fontsize=12)
fig.savefig(out/'demo结果.png',dpi=150)
fig.savefig(out/'demo结果.svg')
