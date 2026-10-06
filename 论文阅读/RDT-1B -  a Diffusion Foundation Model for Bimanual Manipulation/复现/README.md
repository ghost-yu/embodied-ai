# RDT-1B 官方核心小 demo 运行记录

状态：**部分成功**。固定代码commit `cd79363a1387e8f81c7724d070ef7e45fd23150f`；完整论文未复现。原文解释见 [解释.md](<G:/具身/论文阅读/RDT-1B -  a Diffusion Foundation Model for Bimanual Manipulation/解释.md>).

## 环境与资源

Windows、Python3.11.0、CPU四线程、float32；torch2.2.0+cpu、timm1.0.3、diffusers0.27.2、huggingface-hub0.23.0、numpy1.26.4。独立环境 `G:\具身\论文阅读\.venvs\rdt`，与OpenVLA/MDT分开；requirements.txt列直接依赖，requirements-lock.txt记录完整实际版本，uv pip check通过。本机有RTX4060 Laptop，但没有在此demo使用CUDA。没有下载1B权重、T5-XXL/SigLIP或大数据集，不调用外部API、不需key。模型260,608参数，生成合成特征与动作，不创建400GB训练缓冲。

## 复跑命令

工作目录 `G:\具身`。PowerShell中的&执行指定解释器；第一个脚本训练并覆盖demo outputs，第二个在新进程重载并校验采样。路径中有空格，需保持引号。

```powershell
& 'G:\具身\论文阅读\.venvs\rdt\Scripts\python.exe' 'G:\具身\论文阅读\RDT-1B -  a Diffusion Foundation Model for Bimanual Manipulation\复现\scripts\rdt_core_demo.py'
& 'G:\具身\论文阅读\.venvs\rdt\Scripts\python.exe' 'G:\具身\论文阅读\RDT-1B -  a Diffusion Foundation Model for Bimanual Manipulation\复现\scripts\reload_and_plot.py'
```

核心实际调用链：RDTRunner.__init__ → RDT/adaptors/schedulers → compute_loss（runner.forward代理）→ optimizer；predict_action → adapt_conditions → conditional_sample → 五次RDT/scheduler.step → action_mask。直接导入保存仓库；没有替换官方模型/采样器源码。结果.json保存来源路径对应hash，git快照保持干净。

## 配置与数据

保留action128、horizon64、1000噪声等级、cosine schedule、prediction_type=sample、五步采样。缩小hidden64/depth4/heads4；语言特征12维×4tokens，图像8维×6tokens，状态128维；float32替代默认bfloat16。语言是合成embedding，图像是固定随机特征；不是真实前端。官方14个双臂关节/夹爪槽位有效，其余零padding。目标右/左关节随标量z取相反符号；gripper为0.5±.2z，无真实单位。

AdamW lr=.0015、weight_decay0，201更新。训练随机采noise/timestep；初始和最终采样seed778相同，另以779换噪声。条件都是训练中16个条件，没有未见任务/视觉泛化评测。

## 实际验证

公式add_noise一致；真实hook记录ACI条件长度4/6/4/6、query长度67。初始全维loss=0.01428983，有效维平均=0.13064983；无效目标改为1增加1/128。输出头初始梯度非零，上游语言adaptor初始为0，训练后核心模块梯度全部有限非零。

有效维采样MSE：0.13064983→0.00003848；换噪声=0.00004147。无效输出精确为0。语言padding改999，训练后预测最大差0。实际五次模型调用时间为999/799/599/400/200，DPM-Solver末尾sigma零由库处理。独立进程strict权重加载、相同seed的官方采样最大差0。

## 文件与限制

scripts/rdt_core_demo.py为训练/验证；reload_and_plot.py为新进程重载/绘图。outputs内含结果.json、训练曲线.json、运行日志.txt、tiny_rdt_state.pt、evaluation_fixture.pt、demo曲线.png、权重重载校验.json。fixture为本地合成输入/期望输出，不是官方benchmark。结果.json将scheduler的非有限配置lambda_min_clipped用字符串保存，以便严格JSON解析；不改变库运行值。

此环境安装及模型demo没有终端失败；pip复制缓存wheel时提示跨盘hardlink转copy，不影响安装，pip check通过。未运行全训练入口，因此不能据此声称DeepSpeed/Accelerate/HDF5/ROS均可用。没有真实坐标/动作单位处理、环境reset/step或真机命令。论文1.2B配置、冻结encoder、预训练checkpoint、真实数据与success rate都待后续完整实验。
