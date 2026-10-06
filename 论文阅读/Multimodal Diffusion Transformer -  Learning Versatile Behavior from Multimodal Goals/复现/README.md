# MDT 复现记录

状态：**部分成功（官方动作核心缩小配置）**；完整论文未复现。固定 commit `e7a78e744751724d3faa5d113fe8cb5f678c052a`。原文、公式、源码解释见 [解释.md](<G:/具身/论文阅读/Multimodal Diffusion Transformer -  Learning Versatile Behavior from Multimodal Goals/解释.md>).

## 环境和范围

Windows，Python 3.11.0，PyTorch 2.2.0+cpu、NumPy 1.26.4、Hydra 1.3.2、einops 0.8.0。复用 `论文阅读/.venvs/openvla` 的 PyTorch 环境；实际使用 CPU、四线程，没有 CUDA 训练。新增依赖没有替换已有 torch/transformers；`uv pip check` 通过。直接需求见 requirements.txt，完整环境见 requirements-lock.txt。未加载预训练 checkpoint 或数据集；使用合成特征和七维合成动作，模型329,095参数。

## 复跑

工作目录 `G:\具身`；以下 PowerShell 命令中的 `&` 执行指定解释器，路径带空格需引号。第一个脚本构建模型、训练、采样并覆盖本 demo outputs；第二个脚本在独立进程加载权重、验证采样并绘图。

```powershell
& 'G:\具身\论文阅读\.venvs\openvla\Scripts\python.exe' 'G:\具身\论文阅读\Multimodal Diffusion Transformer -  Learning Versatile Behavior from Multimodal Goals\复现\scripts\mdt_core_demo.py'
& 'G:\具身\论文阅读\.venvs\openvla\Scripts\python.exe' 'G:\具身\论文阅读\Multimodal Diffusion Transformer -  Learning Versatile Behavior from Multimodal Goals\复现\scripts\reload_and_plot.py'
```

脚本通过 sys.path 导入保存仓库，不需要安装完整 MDTVAgent 环境。Hydra `_target_` 指向真实 MDTVTransformer；GCDenoiser 实例化内部网络。实际检查函数来源时先 unwrap 装饰器；六个源文件的 hash 在结果.json，可和保存仓库逐一核对。

## 训练与验证

16 个 z 条件，两种合成目标 embedding；action `[16,10,7]`，context `[16,5,64]`。网络 hidden64、四 heads、2 encoder/2 decoder、dropout0、goal_drop0。AdamW lr=.002、weight_decay0，241 更新。训练 sigma 为 .03～1 的 log-uniform；评估固定 sigma=.5、使用新噪声但相同条件。

vis MSE 0.08448666→0.00780687；lang 0.08427044→0.01100310。这些是同条件新噪声的离线误差，不是保留任务泛化或成功率。官方10步DDIM、sigma 1→.001→0，生成两目标 `[2,10,7]`；MSE=0.01181233。固定输入 eta0/1 一致；重载 strict=True 后采样最大差0。源码默认 sigma_max80，demo 改成1。

产物：结果.json（版本/config/shape/hash/指标），运行日志.txt，训练曲线.json，采样记录.json，demo配置.json，demo曲线.png，tiny_mdt_state.pt，evaluation_fixture.pt，权重重载校验.json。fixture 保存本地生成的输入与期望输出，用于重新加载验证，不是官方评测样本。

## 失败与修复

uv 首次下载 SciPy 1.13.1 遇到 TLS close_notify 中断；改用 Windows curl 分段下载官方 PyPI wheel，复用成功的分段，拼接后大小和 PyPI SHA-256 一致。元数据、下载脚本和校验在 `论文阅读/依赖缓存`，没有关闭 TLS 校验。首次脚本的来源断言在 `@torch.no_grad()` 包装函数上失败；用 inspect.unwrap 定位原函数后重跑全部验证通过。这个错误来自 demo 的检查逻辑，没有改官方模型。

## 限制

未实例化完整 MDTVAgent；未运行 Voltron/CLIP、MGF/CLA 联合训练；没有 CALVIN/LIBERO 或真机评估。网络大小、噪声采样、数据与论文训练不同。模型的偏差仍可从曲线看到。输出的数值不是实际机器人单位、坐标系或可执行命令。后续若做论文评测，需要真实 batch、预训练资源、完整配置覆写与官方环境。
