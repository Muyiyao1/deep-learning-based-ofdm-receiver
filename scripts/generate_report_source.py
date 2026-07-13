from __future__ import annotations

"""Generate the editable Chinese report from the final experiment CSVs."""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an editable Chinese report from current CSV results.")
    parser.add_argument("--results-dir", default="results/final")
    parser.add_argument("--output", default="report.md")
    return parser.parse_args()


def _row(frame: pd.DataFrame, method: str, snr_db: float = 20.0) -> pd.Series:
    rows = frame[(frame["method"] == method) & (frame["snr_db"] == snr_db)]
    if rows.empty:
        raise KeyError(f"Missing {method} at {snr_db} dB")
    return rows.iloc[0]


def _variant_row(frame: pd.DataFrame, variant: str, snr_db: float = 20.0) -> pd.Series:
    rows = frame[(frame["variant"] == variant) & (frame["snr_db"] == snr_db)]
    if rows.empty:
        raise KeyError(f"Missing {variant} at {snr_db} dB")
    return rows.iloc[0]


def _largest_sample_method(frame: pd.DataFrame) -> tuple[str, int]:
    """Return the largest sample-covariance LMMSE method present in an artifact."""

    candidates: list[tuple[int, str]] = []
    prefix = "LMMSE-sample-"
    for method in frame["method"].unique():
        if method.startswith(prefix) and method.endswith("+ZF"):
            candidates.append((int(method[len(prefix) : -len("+ZF")]), method))
    if not candidates:
        raise KeyError("No sample-covariance LMMSE method found.")
    sample_count, method = max(candidates)
    return method, sample_count


def _value(row: pd.Series, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.4e} +/- {row[f'{metric}_ci95']:.2e}"


def _table(frame: pd.DataFrame, methods: list[str], snr_db: float = 20.0) -> str:
    rows = ["| 方法 | BER (均值 +/- 95% CI) | 信道 NMSE (均值 +/- 95% CI) |", "| --- | --- | --- |"]
    for method in methods:
        row = _row(frame, method, snr_db)
        rows.append(f"| {method} | {_value(row, 'ber')} | {_value(row, 'channel_nmse')} |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    result_root = ROOT / args.results_dir
    matched = pd.read_csv(result_root / "matched" / "summary_results.csv")
    mismatch = pd.read_csv(result_root / "mismatch" / "summary_results.csv")
    practical_matched = pd.read_csv(result_root / "multiseed" / "practical_comparison_matched_12tap_exponential.csv")
    practical_mismatch = pd.read_csv(result_root / "multiseed" / "practical_comparison_mismatch_8tap_uniform.csv")
    practical_unseen = pd.read_csv(result_root / "multiseed" / "practical_comparison_unseen_10tap_steep_exponential.csv")
    ablation = pd.read_csv(result_root / "ablation" / "ablation_model_seed_summary.csv")
    pilot = pd.read_csv(result_root / "multiseed" / "pilot_pattern_statistics.csv").iloc[0]
    complexity = pd.read_csv(result_root / "multiseed" / "complexity_benchmark.csv")
    prior_meta = pd.read_csv(result_root / "multiseed" / "practical_lmmse_prior_metadata.csv")

    sample_method, sample_count = _largest_sample_method(practical_matched)
    sample_estimator = sample_method.removesuffix("+ZF")
    hard = _variant_row(ablation, "Residual CNN + hard delay projection")
    residual = _variant_row(ablation, "Residual CNN")
    single_matched = _row(practical_matched, "ResidualCNN-single+ZF")
    domain_matched = _row(practical_matched, "ResidualCNN-domain_randomized+ZF")
    single_mismatch = _row(practical_mismatch, "ResidualCNN-single+ZF")
    domain_mismatch = _row(practical_mismatch, "ResidualCNN-domain_randomized+ZF")
    sample_mismatch = _row(practical_mismatch, sample_method)
    single_unseen = _row(practical_unseen, "ResidualCNN-single+ZF")
    domain_unseen = _row(practical_unseen, "ResidualCNN-domain_randomized+ZF")
    sample_unseen = _row(practical_unseen, sample_method)
    cnn_complexity = complexity[complexity["estimator"] == "ResidualCNN"].iloc[0]
    oracle_complexity = complexity[complexity["estimator"] == "LMMSE-oracle"].iloc[0]
    sample_complexity = complexity[complexity["estimator"] == sample_estimator].iloc[0]

    ablation_rows = ["| 模型 | BER (模型 seed CI) | 总体 NMSE | pilot NMSE | data NMSE | deep-fade NMSE |", "| --- | --- | --- | --- | --- | --- |"]
    for _, row in ablation.iterrows():
        ablation_rows.append(
            f"| {row['variant']} | {_value(row, 'ber')} | {_value(row, 'channel_nmse')} | "
            f"{_value(row, 'pilot_nmse')} | {_value(row, 'data_nmse')} | {_value(row, 'deep_fade_nmse')} |"
        )
    practical_rows = ["| 场景 | 方法 | BER (均值 +/- 95% CI) | NMSE (均值 +/- 95% CI) |", "| --- | --- | --- | --- |"]
    for scenario, frame, methods in [
        ("匹配", practical_matched, ["DFT-LS+ZF", "LMMSE-oracle+ZF", sample_method, "ResidualCNN-single+ZF", "ResidualCNN-domain_randomized+ZF"]),
        ("8-tap 均匀失配", practical_mismatch, ["Frame-LS-linear+ZF", "LMMSE-train-prior+ZF", sample_method, "ResidualCNN-single+ZF", "ResidualCNN-domain_randomized+ZF"]),
        ("未见 10-tap 陡峭指数", practical_unseen, [sample_method, "ResidualCNN-single+ZF", "ResidualCNN-domain_randomized+ZF"]),
    ]:
        for method in methods:
            row = _row(frame, method)
            practical_rows.append(f"| {scenario} | {method} | {_value(row, 'ber')} | {_value(row, 'channel_nmse')} |")
    complexity_rows = [
        "| 估计器 | 离线协方差 / K 构建 (ms) | checkpoint 加载 (ms) | 缓存后在线 (ms/帧) | 存储 / 参数 |",
        "| --- | --- | --- | --- | --- |",
        f"| Residual CNN | - | {cnn_complexity['checkpoint_load_ms']:.3f} | {cnn_complexity['online_mean_ms_per_frame']:.4f} +/- {cnn_complexity['online_std_ms_per_frame']:.4f} | {int(cnn_complexity['parameter_count'])} 参数，{int(cnn_complexity['conv_macs_per_frame'])} MAC/帧，{int(cnn_complexity['checkpoint_bytes'])} B checkpoint |",
        f"| Oracle LMMSE | {oracle_complexity['offline_covariance_ms']:.3f} / {oracle_complexity['offline_filter_ms']:.3f} | - | {oracle_complexity['online_mean_ms_per_frame']:.4f} +/- {oracle_complexity['online_std_ms_per_frame']:.4f} | 0 可训练参数，K={int(oracle_complexity['filter_storage_bytes'])} B |",
        f"| Sample LMMSE ({sample_count:,}) | {sample_complexity['offline_covariance_ms']:.3f} / {sample_complexity['offline_filter_ms']:.3f} | - | {sample_complexity['online_mean_ms_per_frame']:.4f} +/- {sample_complexity['online_std_ms_per_frame']:.4f} | 0 可训练参数，K={int(sample_complexity['filter_storage_bytes'])} B |",
    ]
    prior_rows = ["| 先验 | 来源 | 历史信道样本数 | 协方差构建 (ms) |", "| --- | --- | --- | --- |"]
    for _, row in prior_meta.iterrows():
        prior_rows.append(f"| {row['name']} | {row['source']} | {int(row['sample_count'])} | {row['covariance_build_time_ms']:.3f} |")

    report = f"""# 深度学习辅助的 OFDM 信道估计：公平基线、物理先验与泛化分析

英文标题：Deep Learning-Aided OFDM Channel Estimation: Fair Baselines, Physical Priors, and Generalization

## 1. 问题定位与可复现实验链路

本项目实现的是传统 OFDM 接收链路中的深度学习辅助信道估计模块，不是端到端 learned receiver。处理链路为：

```text
bits -> 16QAM -> pilot/data/guard/DC mapping -> unitary IFFT + CP
-> block Rayleigh FIR + AWGN -> CP removal + unitary FFT
-> channel estimation -> one-tap ZF/MMSE -> hard demapping -> BER/SER/EVM
```

频域模型为 `Y[k] = H[k]X[k] + W[k]`。IFFT 使用 `ifft(X)*sqrt(N)`，FFT 使用 `fft(x)/sqrt(N)`；每个 Rayleigh 信道 realization 都归一化到 `sum |h[l]|^2 = 1`，且 CP=16 不小于最大信道长度。SNR 始终按信道输出的经验平均功率定义：`noise_power = mean(|channel_output|^2) / 10^(SNR_dB/10)`。训练、验证和评测共用同一帧生成路径，因此这里的 SNR 不是直接的 Eb/N0。

正式压力测试采用 FFT=64、14 个 OFDM 符号、16QAM、两侧 8 个 guard、1 个 DC null、12-tap 指数 PDP，并在 0--30 dB 评估。每个方法在同一帧复用完全相同的比特、信道与 AWGN realization；没有为了 CNN 调整测试数据或弱化基线。

## 2. Staggered 导频的物理含义

每个符号仅有 {int(pilot['pilot_observations_per_symbol_min'])}--{int(pilot['pilot_observations_per_symbol_max'])} 个 pilot，帧内合计 {int(pilot['pilot_observations_per_frame'])} 个观测，导频开销为 {pilot['pilot_overhead']:.2%}。55 个有效子载波在整帧的 pilot 并集覆盖率为 {pilot['pilot_union_coverage']:.0%}，每个有效子载波平均被直接观测 {pilot['mean_observations_per_active_subcarrier']:.3f} 次。

因此，这不是“所有子载波均未观测的纯频域插值”任务，而是单符号稀疏 pilot、块衰落条件下的帧级观测融合和信道去噪实验。Frame-LS 先合并重复观测再插值；DFT-LS 对该帧级估计做有限时延投影；LMMSE 对相同帧级 LS 向量使用统计滤波；CNN 以同一帧级 LS 与 mask 为输入。信息量对所有强基线和 CNN 一致。

![Staggered pilot pattern](results/final/multiseed/pilot_pattern.png)

## 3. 方法与统计口径

比较 Perfect CSI、Frame-LS、DFT-LS、oracle LMMSE、训练先验 LMMSE、sample-covariance LMMSE 与残差 1-D CNN。CNN 用 circular dilated residual blocks 从帧级线性 LS 预测频域残差；其 hard delay projection 在训练和推理均使用同一有效 tap 区间。MMSE 输出在硬判决前去除幅度偏置，因此在该未编码单抽头模型中，同一信道估计下 ZF 与 MMSE 判决可重合，这是预期现象。

主结果的 95% CI 是 3 个独立测试 Monte Carlo seed 的 Student-t 区间。多模型 CNN 的 95% CI 则是 3 个独立 training/model seed 的 Student-t 区间，且每个模型都在同一 3 个测试 seed、每个 seed 60 帧上评估。测试随机性和模型训练随机性分别保存，不能把 3 个测试 seed 误写成 3 次独立训练。

## 4. 匹配分布结果

以下为单一正式 checkpoint 在 3 个测试 seed、每个 seed 100 帧上的 20 dB 主结果。

{_table(matched, ['PerfectCSI+ZF', 'Frame-LS-linear+ZF', 'DFT-LS+ZF', 'LMMSE-oracle+ZF', 'ResidualCNN+ZF'])}

![Matched BER](results/final/matched/ber_vs_snr.png)

![Matched channel NMSE](results/final/matched/channel_nmse_vs_snr.png)

在匹配的可解析统计模型中，CNN 优于简单 Frame-LS/DFT-LS 的部分工作点，但不优于 oracle LMMSE；高 SNR 还出现模型误差地板。该结论与已知 PDP/时延支持时 LMMSE 可利用准确二阶统计先验相一致。

## 5. 多训练 seed、实用 LMMSE 与泛化

sample-covariance LMMSE 使用有限历史信道样本估计均值与频域协方差，再通过 diagonal loading 与最小特征值截断正则化。其线上仅执行缓存矩阵 `K @ H_LS`。历史先验均来自 12-tap 指数 PDP，未从测试帧拟合：

{chr(10).join(prior_rows)}

下表均为 20 dB。LMMSE/传统行的 CI 是测试 seed 变化；CNN 行的 CI 是模型训练 seed 变化，统计对象不同，不能将二者误解为同一层级的显著性检验。

{chr(10).join(practical_rows)}

![Practical comparison under matched statistics](results/final/multiseed/practical_ber_matched_12tap_exponential.png)

![Training-seed variability on the shared matched test set](results/final/multiseed/training_seed_variability_matched.png)

结论：匹配分布下，single CNN 的 20 dB BER 为 {single_matched['ber_mean']:.4e}，domain-randomized CNN 为 {domain_matched['ber_mean']:.4e}，均不优于 {sample_count:,} 样本 practical LMMSE。8-tap uniform 失配下，domain randomization 将 CNN BER 从 {single_mismatch['ber_mean']:.4e} 降至 {domain_mismatch['ber_mean']:.4e}，但仍高于 practical LMMSE 的 {sample_mismatch['ber_mean']:.4e}。在未见的 10-tap 陡峭指数 PDP 下，single CNN 的 BER 为 {single_unseen['ber_mean']:.4e}，略低于 sample LMMSE 的 {sample_unseen['ber_mean']:.4e}，但其 NMSE 更高（{single_unseen['channel_nmse_mean']:.4e} 对 {sample_unseen['channel_nmse_mean']:.4e}）；domain-randomized CNN 为 {domain_unseen['ber_mean']:.4e}，反而劣于 single CNN。因此 domain randomization 只对部分失配有帮助，且不能把不同 CI 统计层级的 BER 差直接解释为普适显著性结论。

## 6. Delay-domain prior 消融

四个模型训练预算、优化器、训练/验证/测试 seed 划分和网络宽度相同，仅改变残差连接与时延先验形式；每个变体训练 3 个独立模型。hard projection 的有效 tap indexing 由单元测试覆盖，投影长度严格等于配置的信道支持。

{chr(10).join(ablation_rows)}

![Delay-prior ablation](results/final/ablation/ablation_ber_nmse.png)

hard projection 把总体 NMSE 从 {residual['channel_nmse_mean']:.4e} 降至 {hard['channel_nmse_mean']:.4e}，也改善 pilot/data 平均 NMSE；但 BER 从 {residual['ber_mean']:.4e} 变为 {hard['ber_mean']:.4e}，其 Student-t 区间重叠，并且 deep-fade NMSE 从 {residual['deep_fade_nmse_mean']:.4e} 上升至 {hard['deep_fade_nmse_mean']:.4e}。这说明总体信道 MSE 的降低不必然与硬判决最敏感子载波上的误差同步降低；项目不将 delay prior 描述为 BER 的全面提升。

## 7. 修正后的复杂度口径

固定 pilot pattern、PDP、信道长度和 SNR 时，LMMSE 离线构建协方差与 `K = R_hp inv(R_pp + noise_cov)`，线上不再重复求逆。下表在同一 CPU、batch=64、5 次 warm-up、15 次重复测量下给出平均值与标准差。

{chr(10).join(complexity_rows)}

![Cached-online complexity](results/final/multiseed/complexity_benchmark.png)

## 8. 结论、边界与复现

1. CNN 在部分匹配工作点优于简单 LS/DFT-LS，但不优于 oracle LMMSE；匹配与 8-tap uniform 场景也不优于 sample-covariance LMMSE。未见陡峭指数 PDP 的 single CNN BER 略低、NMSE 更高，不能泛化为总体胜出。
2. domain randomization 缩小了 8-tap uniform 失配下的差距，却未改善未见陡峭指数 PDP。
3. hard delay projection 降低总体 NMSE，但未带来可分离的 BER 收益。项目的价值在于说明何时 CNN 有限有效、何时历史信道统计 LMMSE 更合适，而非预设 CNN 必胜。

边界：假设完美同步、帧内静态块衰落，不含 CFO、相位噪声、Doppler、IQ imbalance、功放非线性或信道编码。更复杂失配与硬件非理想仍需新实验验证。

"""
    output = ROOT / args.output
    output.write_text(report, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
