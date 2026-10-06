import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]/'outputs'
rows=json.loads((root/'训练曲线.json').read_text(encoding='utf-8'))
fig,ax=plt.subplots(figsize=(7,4.2),layout='constrained')
ax.semilogy([r['step'] for r in rows],[r['mse'] for r in rows],color='#2068aa',linewidth=2)
ax.set(xlabel='Optimizer updates',ylabel='Training MSE (log scale)',title='RT-1 official TokenLearner: synthetic marker fitting')
ax.grid(True,which='both',alpha=.2)
ax.text(.98,.95,'4 fixed synthetic samples\nNo held-out evaluation',transform=ax.transAxes,ha='right',va='top',fontsize=10)
fig.savefig(root/'训练曲线.png',dpi=160)
fig.savefig(root/'训练曲线.svg')
