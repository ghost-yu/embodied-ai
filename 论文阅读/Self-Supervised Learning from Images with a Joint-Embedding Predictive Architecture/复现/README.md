# I-JEPA 官方核心小 demo

状态：部分成功。官方小配置模型、multi-block mask、optimizer/scheduler、EMA、501次合成训练与重载通过；语义质量未验证，完整论文未复现。详见 [解释.md](<G:/具身/论文阅读/Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture/解释.md>).

## 来源、环境与资源

官方facebookresearch/ijepa固定commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048`，source未修改。Windows、Python3.11.0、torch2.2.0+cpu、numpy1.26.4，CPU四线程float32；复用 `G:\具身\论文阅读\.venvs\openvla`。无API/key、GPU、真实ImageNet或大权重下载；环境package快照见requirements-lock.json。

encoder116,544、predictor31,744参数（含固定位置表），teacher另拷贝encoder；checkpoint约1.1MB量级。合成256张64×64 RGB圆/方形图，16新seed图测试，无类别labels；约10秒训练，不是原ImageNet模型训练时间。

## 复跑命令

工作目录 `G:\具身`；&运行指定解释器，带空格路径保持引号。第一条重新训练并覆盖outputs，第二条独立加载/诊断/绘图。

```powershell
& 'G:\具身\论文阅读\.venvs\openvla\Scripts\python.exe' 'G:\具身\论文阅读\Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture\复现\scripts\ijepa_demo.py'
& 'G:\具身\论文阅读\.venvs\openvla\Scripts\python.exe' 'G:\具身\论文阅读\Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture\复现\scripts\verify_and_plot.py'
```

## 设置与结果

官方VisionTransformer embed64/depth2/heads4、patch8、grid8×8；官方predictor width32/depth2/heads4。一context/四targets，原scale/aspect范围保留，min_keep缩为4。CPU无DDP/AMP；loss用源码smooth-L1和官方AllReduce单进程路径。helper.init_opt给AdamW和warmup20→cosine lr(.0001→.001→.00001)、cosine wd(.04→.1)；batch8/501updates，EMA.996→约.999992。

固定eval smooth-L1 .44948265→.00112139，feature MSE .99950129→.00224277；输出[64,12,64]，context[16,10,64]。目标无梯度，context/learnable mask-token/output head有梯度；501batch overlap0；target像素修改后context差0；第一trainable matrix EMA独立算式每步差0；B2/C2/M4身份顺序验证。新进程严格重载prediction/target差0。

target跨图feature std .11709→.02806，pooled std .03390→.00838，表征变相似。最终预测对初始固定teacher smooth-L1 .81610，teacher明显漂移；不能把最终loss下降当固定目标拟合或语义成功。对最终teacher均值的oracle常量baseline .005278大于predictor .001121，仅说明不是简单全局均值，不排除语义退化。

## 边界与产物

MaskCollator fallback可能放松target排除、randint漏最后合法坐标；实际本配置未出现重叠。不是完全保证所有config无泄漏。输入非ImageNet分布、mask几何小网格离散化、teacher方差缩小，未做linear probe/分类/检索。源码L2 vs smooth-L1、linear vs cosine wd、异分辨率位置插值差异已在正文说明。

outputs保存结果/训练曲线、small_core_state.pt、fixture.pt、权重重载校验、两张图和交付校验；复现/运行日志.txt为本次实际stdout。权重不含optimizer/RNG/counter，支持推理重载，不支持训练resume。未运行官方train.py全入口、真实数据、全部下游/RCDM或机器人任务；完整论文未复现。
