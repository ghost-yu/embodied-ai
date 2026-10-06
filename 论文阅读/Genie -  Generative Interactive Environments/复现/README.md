# Genie 第三方核心小 demo

**部分成功**：实际GenieRedux小配置完成video tokenizer/LAM/dynamics训练、用户码生成、VQ权重对照与独立重载；潜在动作语义不稳定，采样实现有限制；完整原论文未复现。详见 [解释.md](<G:/具身/论文阅读/Genie -  Generative Interactive Environments/解释.md>).

## 来源与环境

原论文arxiv2402.15391v1；非DeepMind官方源码。固定[GenieRedux](https://github.com/insait-institute/GenieRedux) commit `bc80fe43c8370fd79ac5f9fc1719c65c52e294db`，源码保持clean。第三方README/配置不等于原Genie接口/基准，本轮不运行guided真动作模型、RetroAct/AutoExplore。模型直接import，没有伪模块或自写替换网络。

Windows、Python3.11.0、torch2.2.0+cpu、einops.8、beartype.19、numpy1.26.4、matplotlib3.10.7；复用 `G:\具身\论文阅读\.venvs\pi0`，CPU四线程float32，无新框架安装/大模型权重下载、API/key或真实视频数据。全量环境列表见requirements-lock.json。总156,414参数，两个checkpoint各约.7MB。第一轮约117秒，对照约64秒，不是原论文FPS测量。

## 复跑命令

工作目录 `G:\具身`，依次训练初轮、进行VQ对照、独立进程验证/绘图。脚本会覆盖本篇outputs的同名产物。

```powershell
& 'G:\具身\论文阅读\.venvs\pi0\Scripts\python.exe' 'G:\具身\论文阅读\Genie -  Generative Interactive Environments\复现\scripts\genie_core_demo.py'
& 'G:\具身\论文阅读\.venvs\pi0\Scripts\python.exe' 'G:\具身\论文阅读\Genie -  Generative Interactive Environments\复现\scripts\vq_weight_ablation.py'
& 'G:\具身\论文阅读\.venvs\pi0\Scripts\python.exe' 'G:\具身\论文阅读\Genie -  Generative Interactive Environments\复现\scripts\verify_and_plot.py'
```

PowerShell的&运行指定解释器；路径空格需要保留引号。脚本sys.path指向保存repo，requirements.txt为此demo直接依赖，不是完整Linux训练环境。需要先有克隆目录，不应调用pip安装同名models/genie替代。

## 配置与数据

512/64合成视频，每条5frames，8×8 periodic Gaussian亮点，四个随机真实移动方向。训练丢弃真实方向，测试labels仅用于事后confusion；无自由语言、平台跳跃或机械臂。

tokenizer32dim/1block/2heads/head16/patch2/visual codebook32×16；LAM同width、patch4、8×16动作码；dynamics同width、num_tokens32、is_guided=False/additive action、use_token=False。初轮VQ外权重.25，LAM对照.01；内部commitment .25不变。AdamW lr.001/wd0/clip1/batch16，tokenizer1201、LAM1801、dynamics801updates；对照冻结原tokenizer后重训LAM/dynamics。没有原warmup/bfloat16/11B或joint recipe。

## 结果

初轮全部256transition为code6，direction purity29.69%，LAM MSE.009008，toy t1 delta PSNR−.12042dB；不能称动作发现成功。对照使用7个码、purity56.64%、LAM MSE.001440，toy t1 delta PSNR+1.61459dB，一步视频MSE.010424。purity是测试集内按code取最多方向的描述统计，非独立泛化classification；delta对照用(k+1)%8而非论文随机category、t1而非t4，不与原表横比。

两套严格重载后最大差约2.98e−7，整数码一致。第一轮future-last-frame扰动不影响此前visual/action IDs；实际LAM action embedding、VQ和pixel head梯度有限非零。对照user ID响应仍不稳定，未训练过的code6也没有可靠意义。

## 实际问题与修正

LAM返回2×2复制码不能直接expand到dynamics4×4；本地action_codes使用原lookup再singleton广播解决，source未修改。采样没有scalar override会触发tuple TypeError；本地显式传5steps/temperature1/cosine避免默认问题。

源码sample拼接待生成帧后[:, :-1]丢掉该帧；single-step hook发现5次token输入完全相同、1/5steps最终码一致。不能说已完整验证MaskGIT逐轮补全。hook最初注册maskgit对象失败，因为source显式.forward；改为实际token_emb调用处后重跑。FFW数量、decoder无sigmoid、mask最初观察等与原文不同，详见正文。

## 产物与未复现范围

outputs有结果.json、VQ权重对照.json、训练曲线.json、VQ对照训练曲线.json，两套state/fixture、权重重载校验.json、源码hash、原始日志和四张图。图中重建/rollout与user ID对照展示.01实验，使用第一条测试样本；usage图同时展示两轮，没有隐藏初轮collapse。

未执行原11B/互联网预训练、RetroAct/CoinRun/BC/真实动作校准、完整25-step原采样、t4 ΔPSNR/FVD、机器人/长期物理记忆。当前小实验部分缓解坍塌但不证明稳定可控；完整论文未复现。源码bug保留并报告，没有修改源码后称原实现通过。
