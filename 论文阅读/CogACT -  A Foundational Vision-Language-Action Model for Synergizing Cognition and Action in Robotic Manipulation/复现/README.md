# CogACT 官方动作核心小 demo

状态：**部分成功**。官方模块的训练/采样/CFG/ensemble运行通过；合成采样精度不足；完整论文未复现。commit `b174a1b86deedfab4d198d935207e7bb0527994e`，详读见 [解释.md](<G:/具身/论文阅读/CogACT -  A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation/解释.md>).

## 环境与资源

Windows、Python3.11.0、torch2.2.0+cpu、timm0.9.10、numpy1.26.4，CPU四线程、float32。复用 `G:\具身\论文阅读\.venvs\openvla`，本篇没有安装新的框架或下载大权重；uv pip check通过。requirements.txt仅列核心demo直接依赖，requirements-lock.txt记录实际共享环境。完整CogACT依赖还含Prismatic/OpenVLA、训练数据/仿真组件，本demo没有验证这些全入口。

12,476,807随机参数，模型checkpoint约50MB量级；不需要GPU、API/key、真实机器人数据或外部账号。不创建环境episode/reset/step/reward/done，不测控制频率，不发布动作命令。

## 复跑命令

工作目录 `G:\具身`。PowerShell的&执行指定解释器，脚本路径含空格须保留引号。第一条训练并覆盖outputs；第二条独立进程权重重载/采样/绘图；第三条按论文CFG1.5推理，记录与无CFG同条件的误差。

```powershell
& 'G:\具身\论文阅读\.venvs\openvla\Scripts\python.exe' 'G:\具身\论文阅读\CogACT -  A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation\复现\scripts\cogact_core_demo.py'
& 'G:\具身\论文阅读\.venvs\openvla\Scripts\python.exe' 'G:\具身\论文阅读\CogACT -  A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation\复现\scripts\reload_and_plot.py'
& 'G:\具身\论文阅读\.venvs\openvla\Scripts\python.exe' 'G:\具身\论文阅读\CogACT -  A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation\复现\scripts\cfg_eval.py'
```

导入保存的ActionModel，使用官方DiT-S：6layers/hidden384/4heads，cognition4096、动作16×7、100噪声级、cosine schedule、dropout.1；不缩改模型factory。AdaptiveEnsembler按原文件独立导入，避免sim_cogact.__init__同时导入整个VLM/仿真包。调用实际方法，没有伪模块替代。

## 数据与训练

八个合成z条件，4096维向量中前三项携带z/1/.5z；不是真实VLM特征。七维动作 `[z,.5z,-.3z,.2z,0,0,.5+.2z]`重复16步，没有物理单位。AdamW lr=.0003、weight_decay0，901更新。真实ActionModel.loss随机采noise/time，训练learned无条件分支；评估使用新noise/time但仍是相同八条件，不测未见任务。

## 验证和结果

q_sample公式和epsilon MSE重建一致；实际hook为 `[8,17,384]`。强制drop条件与uncondition一致。初始输出头梯度非零、上游condition为0；训练后所检查condition/time/action/block/output梯度有限非零；未用history_embedder无梯度。

epsilon MSE 0.93358707→0.01419737。无CFG且固定初始noise的sample MSE 51.71361923→0.23166592；换noise=0.41248041；CFG1.5同noise=0.22884667。输出仍明显偏离目标，部分值超过归一化范围，**不算动作准确复现成功**。保持clip_denoised=False，未用后处理隐藏误差。noise loss降低不能直接证明生成精度。

DDIM10步映射原时间90/80/.../0。CFG公式与batch两半验证一致。ensemble按历史同一时刻的2/11/20候选融合，第0维11.02005836，history/reset通过；此为独立构造样例，不用于“修正”模型采样。新进程strict state_dict加载、官方DDIM采样最大差0。

## 产物与修复记录

outputs包含结果.json（config/环境/hash/shape/指标）、训练曲线.json、运行日志.txt、dit_s_state.pt、evaluation_fixture.pt、demo曲线.png、权重重载校验.json、CFG评估.json和301更新试跑结果.json。fixture是本地合成输入/期望输出，没有机器人数据。

第一次主脚本在训练/采样完成后，inspect.getfile对直接按文件加载且未注册到sys.modules的class误判built-in，来源检查失败；改为定位AdaptiveEnsembler.ensemble_action的代码文件后重跑通过，官方源码没修改。301步epsilon loss已下降而sample仍.835，因此保存试跑记录后增加到901；输出误差进一步下降但仍不充分，没有把最终结果改称全成功。

## 未复现范围

没有执行完整CogACT.forward、Llama2/DINOv2/SigLIP、真实cognition token/padding契约、预训练DiT-B/大模型权重、OXE数据和全部训练recipe、真实归一化/坐标/夹爪接口、SIMPLER或真机。误差来源没有受控消融，不能单凭合成demo归因于官方架构。后续若做基准，需要完整checkpoint、数据/环境及论文推理配置。
