# RT-1 小 demo 与复现记录

状态：**部分成功（官方模块级）；完整 RT-1 论文未复现。**

本轮使用保存仓库的原始 `FilmConditioning` 和 `TokenLearnerModule`。通过 importlib 从实际源文件加载，没有将它们改成 NumPy 或另一种框架。固定 commit 与源码 SHA-256 在 [结果 JSON](<G:/具身/论文阅读/RT-1 -  Robotics Transformer for Real-World Control at Scale/复现/outputs/模块结果.json>) 中。

## 环境与命令

Windows，独立 Python 3.11.0，TensorFlow 2.15.1，NumPy 1.26.4，CPU。没有使用 CUDA，也没有改变系统 Python。环境目录为 `G:\具身\论文阅读\.venvs\rt1`。它是本轮模块兼容环境，不声称与论文训练环境相同。

```powershell
# 创建与安装（已执行）
uv venv --python 'C:\Users\hgfhg\AppData\Local\Programs\Python\Python311\python.exe' 'G:\具身\论文阅读\.venvs\rt1'
uv pip install --python 'G:\具身\论文阅读\.venvs\rt1\Scripts\python.exe' 'tensorflow-cpu==2.15.1'
# 运行 demo，工作目录可以是 G:\具身
& 'G:\具身\论文阅读\.venvs\rt1\Scripts\python.exe' 'G:\具身\论文阅读\RT-1 -  Robotics Transformer for Real-World Control at Scale\复现\scripts\rt1_modules_demo.py'
# 官方测试，先切换到本篇 代码 目录
& 'G:\具身\论文阅读\.venvs\rt1\Scripts\python.exe' -m robotics_transformer.tokenizers.token_learner_test
```

`uv venv` 创建隔离环境；`--python` 指定解释器。最后的 `-m` 按包名执行官方测试，需要在 `代码` 目录运行。demo 用绝对路径定位源码与输出，不要求手动复制模块。

安装经历：首次安装被会话中断，原进程句柄随后不存在；检查环境发现 tensorflow 只剩不完整 namespace，重新完成同版本安装。最终 demo 实际 import 和运行成功。Windows 上 tensorflow-cpu 引入 tensorflow-intel 作为后端。安装的跨磁盘 hardlink 提示只影响复制效率。

## 实际观察

|检查|输入/操作|结果|
|---|---|---|
|官方 FiLM 恒等初始化|特征 `[6,9,9,512]` + 教学条件 `[6,12]`|最大绝对误差 0|
|FiLM 非恒等变换|Dense bias 指定 gamma=0.1、beta=0.2|与 1.1F+0.2 一致|
|官方 TokenLearner 前向|每帧 81 个位置|输出 `[6,8,512]`|
|六帧视觉内容组合|reshape，不含动作占位|`[1,48,512]`|
|注意力权重|沿 81 位置 softmax|权重和最大误差约 1.19×10⁻⁷|
|聚合核对|实际输出对照权重×原始特征|一致|
|合成选择训练|4 个样本、16 通道、2 个 token、Adam 0.01、120 updates|MSE 3.888609 → 2.0539×10⁻⁸|
|官方 TokenLearner 测试|未改动原测试|1 个测试通过，退出码 0|

这些检查不是机器人成功率。六帧的第一轴在 demo 中当作独立样本送入模块，随后组合视觉序列；没有运行 RT-1 的时序缓存、动作位置或 Transformer。12 维条件是合成向量，没有加载 USE，也不能用它评估语言理解。

合成训练只让 TokenLearner 找到每张合成特征图中的明显 marker。两个输出 token 都监督到同一个 marker 特征，**不验证八个 token 学出不同语义**。固定四样本，未设置独立测试集，loss 下降是拟合/梯度证据，不是泛化证据。

## 从 demo 学习官方代码

读 [demo 源码](<G:/具身/论文阅读/RT-1 -  Robotics Transformer for Real-World Control at Scale/复现/scripts/rt1_modules_demo.py>)，顺序是 `load` → FilmConditioning → TokenLearnerModule → weights/aggregation → GradientTape/optimizer。

- `importlib.util.spec_from_file_location` 定位真实文件；`exec_module` 执行定义。这里不用 fake TensorFlow 或 dependency stub。
- `layer(features,condition)` 会触发 Keras 的构建与 `call`；第一次调用创建 Dense 参数。`bias.assign` 修改层状态，验证手工指定调节后的公式；这不是学习到的语言条件。
- TokenLearner 的 rank-4 输入先把 h/w 展平为 81，输出保持原 feature 通道 512。LayerNorm/MLP 用于算选择权重，而加权对象仍是原特征。
- `tf.GradientTape()` 记录运算；`tape.gradient` 求参数梯度；`apply_gradients` 才更新参数。loss 的 `.numpy()`/float 只用于记录，没有拿记录值反向传播。
- `set_random_seed(17)` 固定本次初始随机源；JSON 保存实际结果。oneDNN、平台与版本变化仍可能带来末位数值差别。

【必须会】说清楚 softmax 的轴、einsum 中求和的 i、81→8 如何保留 512 通道、loss/gradient/optimizer 的分工。

【建议会】把 2 个 token 改成 8 个，把 marker 位置改变后再跑；新运行要另存记录，不能覆盖本轮结果后仍使用旧数字。

【可以暂时不掌握】TensorFlow 图优化与 CPU 内核、完整 EfficientNet 和真机部署。

## 输出和边界

[运行日志](<G:/具身/论文阅读/RT-1 -  Robotics Transformer for Real-World Control at Scale/复现/outputs/运行日志.txt>)、[训练曲线数据](<G:/具身/论文阅读/RT-1 -  Robotics Transformer for Real-World Control at Scale/复现/outputs/训练曲线.json>)、[环境锁定](<G:/具身/论文阅读/RT-1 -  Robotics Transformer for Real-World Control at Scale/复现/outputs/环境锁定.txt>)。

运行出现 TensorFlow v1 API 弃用与 initializer 复用 warning；没有报错，全部断言与官方测试通过。保留 warning，不修改官方源码掩盖它们。

已确认 `rt1main/variables/variables.data-00000-of-00001` 是 134 字节 Git LFS 指针，指向 141841691 字节权重。当前没有实际权重，尚未加载 SavedModel。下一层实践需要取得实际权重并核对 policy spec/依赖，再做策略输入输出，不代表模块 demo 已覆盖该链。


曲线生成：`plot_demo.py` 读取已保存的 JSON，使用 matplotlib 3.10.7 生成 PNG/SVG。命令为本篇解释器加 `复现/scripts/plot_demo.py` 的绝对路径。`requirements.txt` 保存直接依赖，`outputs/环境锁定.txt` 保存完整实际环境。
