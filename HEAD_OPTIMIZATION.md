# BS-RoFormer 仅人声输出分支优化

这项实验研究：任务只需要人声时，能否跳过其他三个独立输出分支，在保留人声质量的同时减少计算与显存。它是预训练模型的推理优化，不训练新权重。

## 实现范围

固定 ZFTurbo MUSDB18-HQ 社区 checkpoint 和配置，先验证 SHA-256，再以 `weights_only=True`、`strict=True` 完整载入四分支模型。原始 stem 顺序为 drums、bass、other、vocals；人声是索引 3。

`--bs-heads vocals` 在严格载入之后，只在内存模型中保留已经训练好的人声 mask estimator，将输出分支数设为 1。共享特征网络、final norm 和人声分支权重不变。其他三个分支及其 ISTFT 不再执行；共享主干仍然执行，所以不能据此推断四倍加速。

固定版本的 vendor 源码不修改，并在载入时核对 provenance 中的本地源文件哈希。checkpoint 文件不修改，完整载入的 CPU 阶段和文件大小也不会因推理分支选择而缩减。保留参数量、移除参数量和原始各分支参数量都记录在 report.json。

四分支与人声单分支的 ISTFT 批形状不同，因此输出数值一致性需要实测，不能只依据结构就宣称逐比特相同。

## 命令用法

在项目目录、已安装项目依赖的 Python 环境中运行。输出目录必须是新目录。

```powershell
python evaluate_project.py --root data/musdb18-hq --subset train --track "Actions - One Minute Smile" --start 60 --duration 30 --system bsroformer --bs-heads vocals --precision amp --overlap 2 --device cuda --threads 1 --save-estimates --out results/head_demo_new
```

将 `--bs-heads vocals` 改成 `--bs-heads all` 可使用完整四分支推理；即使计算全部分支，当前项目对外仍然只返回人声。精度可设 `fp32` 或 CUDA 下的 `amp`。

完整预声明实验：

```powershell
python run_head_optimization.py --config head_optimization_experiments.json --out results/head_optimization_new
python build_head_optimization_report.py --run results/head_optimization_new
```

需要 data/musdb18-hq 中完整 MUSDB18-HQ train、已准备的固定 BS-RoFormer 权重，以及可用 CUDA。报告构建器仅做 CPU 后处理，不加载神经网络。

## 实验协议

- 四组单因素/组合设置：all_fp32、vocal_fp32、all_amp、vocal_amp。
- 质量：官方保留的 14 首验证歌曲，每首固定 60–90 秒，完整 11 秒输入上下文、overlap=2、无 TTA；全部估计保存为未归一化、未裁剪的 FLOAT WAV。
- 耗时：每组另做 3 次全新进程运行，每次固定三首歌共 90 秒，改变各轮设置顺序；排除模型载入、预热、文件读写和指标计算。
- 计时对象：从内存输入到 CPU 波形返回的一次分离调用，包含准备、传输、网络与拼接；同时保留网络前向计时。RTF = 调用秒数 / 音频秒数。
- 数值检查：每种精度的人声单分支对比同精度四分支人声输出，逐曲要求波形 NRMSE ≤ 1e-5 且 |SI-SDR变化| ≤ 0.001 dB。这是预声明的工程容差，不是听觉等价界限。
- 波形误差不做均值去除、增益拟合、归一化、时移搜索；使用 float64 累积误差。SI-SDR 则使用项目已有的 joint-stereo 参考评分，逐曲作差。
- all_amp、vocal_amp 还分别对比 all_fp32；这些跨精度比较不套用同精度分支选择的通过阈值。

## 报告与完整性

`build_head_optimization_report.py` 要求全部 16 个任务完成，检查 run.json 中每份 report.json 的 SHA、配置和任务名单，并把每条输入/参考采样哈希与预声明片段匹配。它确认 FLOAT WAV 的长度、采样率与声道，在分析文件中保存每个 WAV 的文件 SHA、float32 采样内容 SHA、逐曲误差和质量差值。

输出：

- `HEAD_OPTIMIZATION_RESULTS.md`：四组质量、参数、allocated/reserved 显存，三次独立计时、同精度数值检查和组合加速比。
- `head_optimization_analysis.json`：完整精确数值、逐曲数据、报告与波形哈希。分析生成器另记自己的 SHA，不加入推理代码快照，避免把后处理修改当成推理变化。

## 结论边界

公共权重的完整训练歌曲名单未知，开发片段可能已经用于权重训练；不能宣称独立留出泛化或复现原论文的官方测试分数。这里只验证固定片段，尚未覆盖整曲和真实麦克风条件。

分支选择保留双向注意力和未来上下文，不会让模型成为因果或流式模型。RTF 是吞吐指标，不能代替算法延迟、首个输出等待时间或音频设备端到端延迟。GPU 峰值 allocated/reserved 来自 PyTorch allocator；reserved 包含缓存，也不等于系统显存占用。

这项成果适合描述为：根据只提取人声的应用需求，理解输出结构、减少无用分支、验证数值误差并实测计算与显存变化；不应描述为发明新的分离架构。
