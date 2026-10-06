# UniPi 教学机制小 demo

状态：部分成功。视频扩散→监督补帧→学习逆动力学→实际二维环境执行链路完成，执行精度不足；完整论文未复现。[解释.md](<G:/具身/论文阅读/Learning Universal Policies via Text-Guided Video Generation/解释.md>)给出原文/教学代码的区别。

作者[官网](https://universal-policy.github.io/unipi/)说明原实验源码无法发布，推荐[AVDC](https://github.com/flow-diffusion/AVDC)作为视频训练代码基础。AVDC是另一篇论文的官方repo；本次未调用它。本地新写教学脚本没有官方训练commit。

## 环境、数据与资源

Windows、Python3.11、torch2.2.0+cpu、numpy1.26.4、matplotlib3.10.7；共享环境 `G:\具身\论文阅读\.venvs\openvla`，CPU四线程float32，GPU未用。不需要API key、账号、外部数据/权重。约126k总参数，checkpoint不到1MB；fixture含64合成轨迹与结果。实际训练约322秒；保存环境package列表见requirements-lock.json。

数据为16×16灰度Gaussian亮点、四条固定指令、512训练/64新起点测试轨迹。8个动作每步.4像素；planner生成h=2/4/6/8四keyframes。inverse另有4096随机有动作transition。没有真实视频、机器人坐标或自由语言。

## 命令

工作目录 `G:\具身`。第一条训练并覆盖本篇outputs，第二条只加载已有权重并评价/绘图；路径含空格需引号，&用于运行指定解释器。

```powershell
& 'G:\具身\论文阅读\.venvs\openvla\Scripts\python.exe' 'G:\具身\论文阅读\Learning Universal Policies via Text-Guided Video Generation\复现\scripts\unipi_demo.py'
& 'G:\具身\论文阅读\.venvs\openvla\Scripts\python.exe' 'G:\具身\论文阅读\Learning Universal Policies via Text-Guided Video Generation\复现\scripts\verify_and_plot.py'
```

训练video1801次Adam .001、midpoint801次Adam .001、inverse801次Adam .003。video用64等级linearβ(.001,.2)、clean-pixel weighted loss、15%language drop（first保留），64步DDIM，主结果omega0。补帧为deterministic CNN，非论文diffusion TSR；inverse为pixel moments→MLP二维控制，非论文卷积7controls。DotEnv实际reset/step，8步open-loop，raw/applied动作分别记录，goal只用于metric。

## 实际结果

64保留起点平均终点距离0.732220像素，预设.5像素阈值成功18/64；coarse普通pixel MSE=0.000190639，inverse新transition MSE=0.000003148。oracle真实未来图执行距离0.022267像素，不能计作生成成功。CFG omega.5误差1.147050，更差；换noise误差0.732911。raw动作限幅component数0；生成像素仍越界，未clip输出。

独立进程strict加载后，coarse/dense/actions/observations的max差均0；forward hook为63…0。反向指令对正确目标MSE=.000173，原指令对相反目标=.015994，证实有限ID条件产生行为差异。训练loss/生成画面MSE不能推导真实任务成功。

## 产物与限制

outputs/结果.json、训练曲线.json、teaching_state.pt、fixture.pt、权重重载校验.json、重载日志.txt、训练曲线.png、视频计划与执行.png、轨迹.png及交付校验.json；保存训练脚本hash和权重hash供复查。训练标准输出在本次工具运行记录中；结构化训练曲线和最终结果为持久证据，没有伪造原始训练stdout。

未复现：官方1.7B video U-Net/T5-XXL/256TPU训练、原schedule/参数化、diffusion时间超分辨率、7controls/Pybullet/CLIPort、classifier/Dirac prior、语言组合和互联网迁移、FID/FVD/CLIP、真机。toy执行仍失败46/64，不叫任务准确复现成功。进一步精度改进应隔离planner、补帧、inverse误差，不能用后处理或放宽阈值掩盖。
