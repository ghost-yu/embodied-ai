# π₀ 官方 JAX 核心小 demo

状态：**部分成功**，完整论文未复现。源码commit `215abfb217dbac7d5f1273282331b9b1866c0479`，原文与代码说明见 [解释.md](<G:/具身/论文阅读/π0 -  A Vision-Language-Action Flow Model for General Robot Control/解释.md>).

## 环境与资源边界

Windows、Python3.11.0、CPU、float32；JAX0.5.3/JAXlib0.5.3、Flax0.10.2、Optax0.2.4、NumPy1.26.4。独立venv为 `G:\具身\论文阅读\.venvs\pi0`；pip check通过，完整实际版本在requirements-lock.txt。没有CUDA训练、API/key、3.3B checkpoint或真实数据下载；16,936,448随机参数，小模型参数msgpack约68MB量级。

仅安装此JAX demo用到的依赖，没有执行完整uv sync。仓库声明torch2.7.1和jax[cuda12]；本CPU demo为JAX0.5.3，torch2.2.0仅用于保存仓库JAX文件导入时的类型/PyTorch接口依赖，没有运行PyTorch PI0Pytorch。ml-dtypes/tensorstore等实际解析版本也不完全等于仓库uv override，详见lock。这些偏差不能支持完整PyTorch/分布式环境兼容结论。

## 复跑

工作目录 `G:\具身`。PowerShell的&执行指定解释器，路径有空格须保持引号。主脚本训练并覆盖小demo outputs；第二条在新进程重新构建、加载msgpack参数，运行官方缓存采样并绘图。

```powershell
& 'G:\具身\论文阅读\.venvs\pi0\Scripts\python.exe' 'G:\具身\论文阅读\π0 -  A Vision-Language-Action Flow Model for General Robot Control\复现\scripts\pi0_core_demo.py'
& 'G:\具身\论文阅读\.venvs\pi0\Scripts\python.exe' 'G:\具身\论文阅读\π0 -  A Vision-Language-Action Flow Model for General Robot Control\复现\scripts\reload_and_plot.py'
```

scripts/pi0_demo_support.py仅调整构造配置：Gemma两个专家使用官方dummy、原始pi05=False、dtype=float32、prompt4、action_dim32、horizon50；SigLIP Module的硬编码So400m/14参数在构造期间改为官方mu/32，finally恢复factory。不替换前向/loss/attention/sampler，也不改官方源码。此适配器不是官方默认入口；不要用本msgpack配置加载3.3B权重。

## 数据与训练

四个z条件：常色合成RGB（真实通过官方视觉encoder）、state第一维z、固定整数prompt（非真实tokenizer结果）。七个目标维随z变化，全部50步重复，其余目标零；没有动作物理单位。图像与state同时携带z，任务不证明对视觉的必要依赖或指令理解。Adam lr=.002、501更新；train=False关闭图像augmentation，仍实际求梯度/更新参数。

真实调用链：Pi0Config.create → Pi0/Gemma/SigLIP → embed_prefix/embed_suffix → compute_loss；nnx.value_and_grad → NNX/Optax optimizer；sample_actions → prefix cache → 10次suffix/Euler。完整scripts/train.py/data_loader未运行。

## 检查与结果

公式使用同key的noise/time重建，loss shape `[4,50]`与官方一致。prefix `[4,151,64]`、suffix `[4,51,64]`、full mask `[4,202,202]`。group规则检查通过：prefix不读suffix，state不读action，动作彼此双向；完整与缓存forward最大差2.0861626e-06。

固定新key flow MSE 1.70825970→0.21798004；固定noise sample MSE 1.59779489→0.01292223；换noise同条件sample MSE 0.01392165。501步最后训练loss约.03156，不可代替新key评估.21798。首轮视觉梯度norm约84008.69，有限但很大，不能宣称大模型稳定性。十步缓存Euler、dt=-.1，输出 `[4,50,32]`全部有限；右图仍有残差。

独立进程通过Flax参数schema恢复msgpack，官方采样最大差3.4272671e-07（atol2e-5/rtol1e-5），不是逐bit要求。所有条件仍是训练四条件，没有未见任务评估。

## 产物、失败与修复

outputs包含结果.json（config/环境/hash/shape/指标）、训练曲线.json、运行日志.txt、构造日志.txt、tiny_pi0_params.msgpack、evaluation_fixture.npz、demo曲线.png、参数重载校验.json。fixture是本地合成输入/期望输出，参数不包含预训练语义。

初次import检查在torch依赖安装尚未结束时提前启动，报Tensor导入错误；等安装完成后构造通过。首个训练脚本误把Flax0.10.2 `State.flat_state()`的dict当作pairs迭代，导致gradient日志报unpack错误；改成.items()后官方训练通过，没有改官方模块。101步时sample MSE约.4666，故增加到501步后重新完整执行。环境安装的跨盘hardlink提示已自动copy，pip check通过。

## 未复现范围

未下载原PaliGemma/SigLIP或π₀预训练权重；没有完整3.3B、原始数据recipe、LeRobot数据管线、GPU/JAX分片、π₀.5/FAST或PyTorch推理。没有真实坐标/归一化统计、机器人reset/step、reward/done/成功率或控制频率实测。结果为官方小配置的合成训练与机制验证，不是论文任务性能复现。
