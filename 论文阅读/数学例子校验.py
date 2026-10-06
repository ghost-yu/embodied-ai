"""独立数值核对；不导入官方模型、不构成论文复现。"""
import json, math, sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
records = []
def check(name, computed, expected):
    np.testing.assert_allclose(computed, expected, rtol=1e-6, atol=1e-8)
    records.append({'例子':name, '计算值':np.asarray(computed).tolist(), '预期值':np.asarray(expected).tolist(), '结果':'通过'})

check('RT-1 FiLM', (1+.1)*2+.2, 2.4)
check('RT-1 TokenLearner汇总', np.dot([.25,.75],[2,6]), 5)
check('RT-1 零点量化', -1+int(255*.5)/255*2, -1/255)
check('RT-2/OpenVLA交叉熵', -np.log(.8), .2231435513142097)
check('OpenVLA反归一化', .5*(.5+1)*(.6-(-.2))-.2, .4)
edges=np.linspace(-1,1,256); centers=(edges[:-1]+edges[1:])/2
bins=np.digitize([-1,0,1],edges)
check('OpenVLA边界索引', bins, [1,128,256])
check('OpenVLA端点反量化', centers[np.clip(bins-1,0,254)], [-254/255,0,254/255])
sd=sigma=.5; a=.2; x=a+sigma
cs=sd**2/(sigma**2+sd**2); co=sigma*sd/math.sqrt(sigma**2+sd**2)
target=(a-cs*x)/co
check('MDT inner target', target, -.4242640687119284)
check('MDT组合干净动作', cs*x+co*target, .2)
check('MDT CLA两样本', -math.log(math.e/(math.e+1)), .3132616875182228)
check('MDT MGF两种分母', [(1+3)/4,(1+3)/2], [1,2])
check('MDT LIBERO相对增益', (74.3-68.5)/68.5*100, 8.46715328467153)
# RDT sample target与noise target不同：相同预测对不同目标的MSE不同。
check('RDT clean/noise目标差别', [(.3-.2)**2,(.3-1)**2], [.01,.49])
tau=.25; eps=1; action=.2
check('pi0线性路径', tau*action+(1-tau)*eps, .8)
check('pi0换元', [(1-tau)*eps+tau*action, eps-action], [.8,.8])
check('pi0双时间方向Euler', [eps+10*.1*(action-eps),eps+10*(-.1)*(eps-action)], [.2,.2])
weights=np.exp(.1*np.array([1,-1])); weights/=weights.sum()
check('CogACT相反候选加权', np.dot(weights,[1,-1]), math.tanh(.1))
check('UniPi CFG', (1+2)*.8-2*.5, 1.4)
T,N=16,100
check('Genie注意力配对数', [(T*N)**2,T*N*N+N*T*T], [2560000,185600])
check('Genie PSNR差', 10*np.log10(.04/.01), 6.020599913279624)
error=np.array([1.2,1.8])-np.array([1,2])
check('I-JEPA L2/MSE/smoothL1', [np.sum(error**2),np.mean(error**2),np.mean(.5*error**2)], [.08,.04,.02])
check('I-JEPA EMA', .9*2+.1*4, 2.2)
check('I-JEPA patch数', [(224//14)**2,(224//16)**2], [256,196])
(ROOT/'数学例子校验.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'{len(records)} 项数学例子通过；未导入或运行官方模型。')
