# VocalStream — CV text and claim boundaries

[Project](https://github.com/junxilucian-rgb/vocalstream) · [Evidence](RESULTS.md) · [中文面试学习路线](INTERVIEW_GUIDE.zh-CN.md)

Use the wording below for a **personal research/engineering project**, dated October 2026–Present. Development and experiment orchestration used Codex assistance. Keep the bullets whose implementation, measurements and limitations you can explain; a completed repository alone does not establish independent coding proficiency.

## 中文简历

**VocalStream｜音乐人声分离与推理效率优化**  
2026.10–至今｜个人研究项目，Codex辅助开发  
Python · PyTorch · 音频信号处理 · MUSDB18-HQ  
GitHub：https://github.com/junxilucian-rgb/vocalstream

- 复现 Open-Unmix 和 BS-RoFormer 预训练推理，在14首固定 MUSDB18-HQ 开发片段上建立 SI-SDR、实时比率与显存评测，并开展受控伴奏串音及人声保留实验。
- 实现仅人声输出分支，使运行模型参数量减少 **56.1%**，同精度14/14片段输出逐采样一致；结合混合精度，在 RTX 4060 Laptop GPU 上通过三次独立计时测得 **2.35倍吞吐量**，峰值已分配显存由 **1.80 GB降至0.95 GB**。
- 构建限制未来输入的流式评估，完成25项窗口推理对照（三首4秒开发片段，每曲/设置计时一次），将每块计算中位数约从 **183 ms降至91 ms**，并分析分离质量、输出等待及尚未满足的20/80 ms计算预算。

## English CV

**VocalStream — Music Vocal Separation and Inference Optimization**  
Oct 2026–Present | Personal research project; Codex-assisted development  
Python · PyTorch · Audio Signal Processing · MUSDB18-HQ  
GitHub: https://github.com/junxilucian-rgb/vocalstream

- Reproduced pretrained Open-Unmix and BS-RoFormer inference and evaluated SI-SDR, processing-time/audio-duration ratio and GPU memory on 14 fixed MUSDB18-HQ development excerpts, including controlled accompaniment bleed and vocal-preservation experiments.
- Implemented vocal-head selection, reducing active parameters by **56.1%** with sample-identical outputs on all 14 excerpts within each precision. Combined with mixed precision, achieved **2.35× throughput** over three independent timing runs and reduced peak allocated GPU memory from **1.80 to 0.95 GB** on an RTX 4060 Laptop GPU.
- Developed bounded-future streaming evaluation and a 25-job comparison on three 4 s development clips, with one timing run per song/setting; reduced block-compute medians from approximately **183 to 91 ms** while quantifying quality changes and remaining failures to meet 20/80 ms compute budgets.

## Keep the two optimizations separate

| Claim | Evidence and scope |
| --- | --- |
| 56.1% fewer active parameters | Removal of three independent output heads after strict checkpoint loading; the original checkpoint file stays unchanged. |
| 2.35× offline throughput | Vocal-only AMP vs four-head FP32; 11 s context, 14 fixed quality excerpts and three timing repetitions on three songs. Most of the speed benefit came from AMP. |
| Approximately 183 → 91 ms per emitted block | Nested OLA vs single-window inference; vocal-only AMP, 2 s context, three 4 s clips, one timing observation per song/schedule/policy. |
| 72 job reports | 16 head/precision + 13 earlier streaming + 18 refinement + 25 window-comparison jobs. Jobs are not independent songs or repeated timing trials. |

The two speed ratios use different workloads and baselines. They must not be multiplied into a combined project speedup. The stream-window change does alter predictions; it does not inherit the same-precision equivalence result from head selection.

Use “personal project”, not employment or an internship at Yamaha. The project did not train a new separation architecture, establish a state-of-the-art benchmark, demonstrate an acoustic-feedback closed loop, or measure soundcard end-to-end latency. Its pretrained checkpoint has an undisclosed full training-song list; the development scores do not establish held-out generalization. For interview questions, use [the learning guide](INTERVIEW_GUIDE.zh-CN.md).
