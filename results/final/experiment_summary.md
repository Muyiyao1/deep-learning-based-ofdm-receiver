# 实验摘要（由当前 CSV 自动生成）

本文件只读取 `results/final/` 下本轮实验生成的 CSV；不引用重构前的图或结果。

## 统计口径

- 主结果的置信区间是 3 个独立测试 Monte Carlo seed 上的 Student-t 95% CI。
- 多训练 seed CNN 的置信区间是 3 个独立模型训练 seed 的 Student-t 95% CI；每个模型均在同一组 3 个固定测试 seed、每个测试 seed 60 帧上评估。两类不确定性分别保存，不能混为一谈。
- 所有 BER 与 NMSE 图的原始逐 training-seed/test-seed/SNR 数据位于 `multiseed/training_seed_results.csv` 与 `ablation/ablation_training_seed_test_seed_results.csv`。

## 导频与场景

- FFT=64，每帧 14 个 OFDM 符号，guard=8，DC=1，有效子载波=55。
- 每个符号导频数为 6-7，全帧 97 个导频观测，导频开销 12.60%。
- staggered 导频的帧内并集覆盖 100% 有效子载波；每个有效子载波平均观测 1.764 次（最少 1、最多 2）。
- 因而该任务准确地说是“单符号稀疏 pilot、块衰落条件下的帧级观测融合和信道去噪实验”，而非所有子载波均未直接观测的纯插值任务。

## 单模型主结果：20 dB

| 方法 | BER (均值 +/- 95% CI) | 信道 NMSE (均值 +/- 95% CI) |
| --- | --- | --- |
| PerfectCSI+ZF | 1.2100e-02 +/- 8.69e-04 | 0.0000e+00 +/- 0.00e+00 |
| Frame-LS-linear+ZF | 1.8493e-02 +/- 6.19e-04 | 4.7356e-02 +/- 7.67e-03 |
| DFT-LS+ZF | 1.8577e-02 +/- 7.14e-04 | 3.4018e-02 +/- 6.48e-03 |
| LMMSE-oracle+ZF | 1.3400e-02 +/- 1.01e-03 | 2.0500e-03 +/- 3.81e-04 |
| ResidualCNN+ZF | 1.4991e-02 +/- 1.54e-03 | 1.1124e-02 +/- 4.39e-04 |

## 失配主结果：20 dB

| 方法 | BER (均值 +/- 95% CI) | 信道 NMSE (均值 +/- 95% CI) |
| --- | --- | --- |
| Frame-LS-linear+ZF | 2.1046e-02 +/- 8.98e-04 | 1.2412e-01 +/- 1.07e-02 |
| DFT-LS+ZF | 3.6291e-02 +/- 2.77e-03 | 8.6858e-02 +/- 7.31e-03 |
| LMMSE-oracle+ZF | 1.4730e-02 +/- 1.14e-03 | 8.8364e-04 +/- 5.05e-05 |
| LMMSE-train-prior+ZF | 1.5295e-02 +/- 1.17e-03 | 1.9427e-03 +/- 2.78e-04 |
| ResidualCNN+ZF | 1.7827e-02 +/- 1.42e-03 | 1.8362e-02 +/- 7.07e-04 |

## 多模型与 practical LMMSE：20 dB

| 场景 | 方法 | BER (均值 +/- 95% CI) | 信道 NMSE (均值 +/- 95% CI) |
| --- | --- | --- | --- |
| 匹配 12-tap 指数 PDP | DFT-LS+ZF | 1.7946e-02 +/- 4.30e-03 | 2.8074e-02 +/- 8.21e-03 |
| 匹配 12-tap 指数 PDP | LMMSE-oracle+ZF | 1.3255e-02 +/- 4.67e-03 | 2.0693e-03 +/- 5.37e-04 |
| 匹配 12-tap 指数 PDP | LMMSE-sample-10000+ZF | 1.6481e-02 +/- 5.87e-03 | 4.8932e-03 +/- 1.04e-03 |
| 匹配 12-tap 指数 PDP | ResidualCNN-single+ZF | 1.8857e-02 +/- 8.63e-04 | 2.7019e-02 +/- 2.79e-03 |
| 匹配 12-tap 指数 PDP | ResidualCNN-domain_randomized+ZF | 1.8735e-02 +/- 1.75e-03 | 3.1954e-02 +/- 5.75e-03 |
| 8-tap 均匀 PDP 失配 | DFT-LS+ZF | 3.3872e-02 +/- 3.66e-03 | 7.5062e-02 +/- 1.05e-02 |
| 8-tap 均匀 PDP 失配 | Frame-LS-linear+ZF | 2.1062e-02 +/- 1.35e-03 | 1.0804e-01 +/- 1.39e-02 |
| 8-tap 均匀 PDP 失配 | LMMSE-oracle+ZF | 1.4562e-02 +/- 1.17e-03 | 9.1969e-04 +/- 4.01e-04 |
| 8-tap 均匀 PDP 失配 | LMMSE-train-prior+ZF | 1.5158e-02 +/- 1.34e-03 | 1.9678e-03 +/- 2.72e-04 |
| 8-tap 均匀 PDP 失配 | LMMSE-sample-10000+ZF | 1.8803e-02 +/- 1.30e-03 | 5.0519e-03 +/- 7.84e-04 |
| 8-tap 均匀 PDP 失配 | ResidualCNN-single+ZF | 2.7710e-02 +/- 1.49e-03 | 7.2017e-02 +/- 1.17e-02 |
| 8-tap 均匀 PDP 失配 | ResidualCNN-domain_randomized+ZF | 2.4661e-02 +/- 1.36e-03 | 5.6525e-02 +/- 3.39e-03 |
| 未见 10-tap 陡峭指数 PDP | LMMSE-oracle+ZF | 1.0498e-02 +/- 1.05e-03 | 1.2481e-03 +/- 1.20e-04 |
| 未见 10-tap 陡峭指数 PDP | LMMSE-sample-10000+ZF | 1.3495e-02 +/- 1.45e-03 | 3.8605e-03 +/- 4.24e-04 |
| 未见 10-tap 陡峭指数 PDP | ResidualCNN-single+ZF | 1.2384e-02 +/- 4.85e-04 | 9.8441e-03 +/- 1.01e-03 |
| 未见 10-tap 陡峭指数 PDP | ResidualCNN-domain_randomized+ZF | 1.3982e-02 +/- 2.12e-03 | 2.3027e-02 +/- 7.77e-03 |

## 可审计结论

- 在匹配分布 20 dB，单一训练分布 CNN 的 BER 为 1.8857e-02，低于 DFT-LS 的 1.7946e-02，但高于 sample-covariance LMMSE (10,000 历史信道样本) 的 1.6481e-02；它也不优于 oracle LMMSE。
- 在 8-tap 均匀 PDP 失配 20 dB，domain-randomized CNN 的 BER 为 2.4661e-02，优于单一分布 CNN 的 2.7710e-02，但仍高于 sample-covariance LMMSE 的 1.8803e-02。
- 在未见 10-tap 陡峭指数 PDP，domain randomization 的 BER 为 1.3982e-02，反而高于单一分布 CNN 的 1.2384e-02；它不是对所有失配都稳健的万能改进。
- hard delay projection 将总体 NMSE 从 4.0473e-02 降为 3.2435e-02，但 BER 从 1.9999e-02 变为 2.0096e-02，且深衰落 NMSE 从 5.1818e-02 升为 5.7250e-02。因此保留“NMSE 改善不保证硬判决 BER 改善”的结论。
- 正确拆分离线/在线成本后，sample-covariance LMMSE (10,000 样本) 的协方差构建为 20.026 ms，缓存滤波器在线时间为 0.0032 +/- 0.0007 ms/帧；CNN checkpoint 加载 4.605 ms，warm-up 后在线时间为 0.0620 +/- 0.0131 ms/帧。
- 在匹配与 8-tap uniform 失配中，sample-covariance LMMSE 的 BER/NMSE 均优于当前 CNN 且缓存后更快；但在未见 10-tap 陡峭指数 PDP，single CNN 的 BER 为 1.2384e-02，略低于 sample LMMSE 的 1.3495e-02，同时 NMSE 更高（9.8441e-03 对 3.8605e-03）。两者 CI 分别来自模型 seed 与测试 seed，不能据此做跨层级显著性断言；更不能宣称 CNN 普遍替代 LMMSE。
