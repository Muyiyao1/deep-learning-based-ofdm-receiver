from __future__ import annotations

"""Create an auditable Chinese experiment digest from current CSV artifacts only."""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Chinese experiment summary from current CSV results.")
    parser.add_argument("--results-dir", default="results/final")
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


def _largest_sample_method(frame: pd.DataFrame) -> tuple[int, str]:
    """Return the largest sample-covariance LMMSE method present in an artifact."""

    candidates: list[tuple[int, str]] = []
    prefix = "LMMSE-sample-"
    for method in frame["method"].unique():
        if method.startswith(prefix) and method.endswith("+ZF"):
            candidates.append((int(method[len(prefix) : -len("+ZF")]), method))
    if not candidates:
        raise KeyError("No sample-covariance LMMSE method found.")
    return max(candidates)


def _metric(row: pd.Series, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.4e} +/- {row[f'{metric}_ci95']:.2e}"


def _table(frame: pd.DataFrame, methods: list[str], snr_db: float = 20.0) -> list[str]:
    lines = ["| 方法 | BER (均值 +/- 95% CI) | 信道 NMSE (均值 +/- 95% CI) |", "| --- | --- | --- |"]
    for method in methods:
        row = _row(frame, method, snr_db)
        lines.append(f"| {method} | {_metric(row, 'ber')} | {_metric(row, 'channel_nmse')} |")
    return lines


def main() -> None:
    args = parse_args()
    root = Path(args.results_dir)
    matched = pd.read_csv(root / "matched" / "summary_results.csv")
    mismatch = pd.read_csv(root / "mismatch" / "summary_results.csv")
    practical_matched = pd.read_csv(root / "multiseed" / "practical_comparison_matched_12tap_exponential.csv")
    practical_mismatch = pd.read_csv(root / "multiseed" / "practical_comparison_mismatch_8tap_uniform.csv")
    practical_unseen = pd.read_csv(root / "multiseed" / "practical_comparison_unseen_10tap_steep_exponential.csv")
    ablation = pd.read_csv(root / "ablation" / "ablation_model_seed_summary.csv")
    pilot = pd.read_csv(root / "multiseed" / "pilot_pattern_statistics.csv").iloc[0]
    complexity = pd.read_csv(root / "multiseed" / "complexity_benchmark.csv")

    sample_count, sample_method = _largest_sample_method(practical_matched)
    cnn_matched = _row(practical_matched, "ResidualCNN-single+ZF")
    dft_matched = _row(practical_matched, "DFT-LS+ZF")
    sample_matched = _row(practical_matched, sample_method)
    cnn_mismatch = _row(practical_mismatch, "ResidualCNN-single+ZF")
    domain_mismatch = _row(practical_mismatch, "ResidualCNN-domain_randomized+ZF")
    sample_mismatch = _row(practical_mismatch, sample_method)
    sample_unseen = _row(practical_unseen, sample_method)
    domain_unseen = _row(practical_unseen, "ResidualCNN-domain_randomized+ZF")
    single_unseen = _row(practical_unseen, "ResidualCNN-single+ZF")
    hard = _variant_row(ablation, "Residual CNN + hard delay projection")
    residual = _variant_row(ablation, "Residual CNN")
    cnn_time = complexity[complexity["estimator"] == "ResidualCNN"].iloc[0]
    lmmse_time = complexity[complexity["estimator"] == sample_method.removesuffix("+ZF")].iloc[0]

    lines = [
        "# 实验摘要（由当前 CSV 自动生成）",
        "",
        "本文件只读取 `results/final/` 下本轮实验生成的 CSV；不引用重构前的图或结果。",
        "",
        "## 统计口径",
        "",
        "- 主结果的置信区间是 3 个独立测试 Monte Carlo seed 上的 Student-t 95% CI。",
        "- 多训练 seed CNN 的置信区间是 3 个独立模型训练 seed 的 Student-t 95% CI；每个模型均在同一组 3 个固定测试 seed、每个测试 seed 60 帧上评估。两类不确定性分别保存，不能混为一谈。",
        "- 所有 BER 与 NMSE 图的原始逐 training-seed/test-seed/SNR 数据位于 `multiseed/training_seed_results.csv` 与 `ablation/ablation_training_seed_test_seed_results.csv`。",
        "",
        "## 导频与场景",
        "",
        f"- FFT={int(pilot['fft_size'])}，每帧 {int(pilot['ofdm_symbols'])} 个 OFDM 符号，guard={int(pilot['guard_subcarriers_total'])}，DC={int(pilot['dc_subcarriers'])}，有效子载波={int(pilot['active_subcarriers'])}。",
        f"- 每个符号导频数为 {int(pilot['pilot_observations_per_symbol_min'])}-{int(pilot['pilot_observations_per_symbol_max'])}，全帧 {int(pilot['pilot_observations_per_frame'])} 个导频观测，导频开销 {pilot['pilot_overhead']:.2%}。",
        f"- staggered 导频的帧内并集覆盖 {pilot['pilot_union_coverage']:.0%} 有效子载波；每个有效子载波平均观测 {pilot['mean_observations_per_active_subcarrier']:.3f} 次（最少 {int(pilot['minimum_observations_per_active_subcarrier'])}、最多 {int(pilot['maximum_observations_per_active_subcarrier'])}）。",
        "- 因而该任务准确地说是“单符号稀疏 pilot、块衰落条件下的帧级观测融合和信道去噪实验”，而非所有子载波均未直接观测的纯插值任务。",
        "",
        "## 单模型主结果：20 dB",
        "",
        *_table(matched, ["PerfectCSI+ZF", "Frame-LS-linear+ZF", "DFT-LS+ZF", "LMMSE-oracle+ZF", "ResidualCNN+ZF"]),
        "",
        "## 失配主结果：20 dB",
        "",
        *_table(mismatch, ["Frame-LS-linear+ZF", "DFT-LS+ZF", "LMMSE-oracle+ZF", "LMMSE-train-prior+ZF", "ResidualCNN+ZF"]),
        "",
        "## 多模型与 practical LMMSE：20 dB",
        "",
        "| 场景 | 方法 | BER (均值 +/- 95% CI) | 信道 NMSE (均值 +/- 95% CI) |",
        "| --- | --- | --- | --- |",
    ]
    for scenario, frame, methods in [
        ("匹配 12-tap 指数 PDP", practical_matched, ["DFT-LS+ZF", "LMMSE-oracle+ZF", sample_method, "ResidualCNN-single+ZF", "ResidualCNN-domain_randomized+ZF"]),
        ("8-tap 均匀 PDP 失配", practical_mismatch, ["DFT-LS+ZF", "Frame-LS-linear+ZF", "LMMSE-oracle+ZF", "LMMSE-train-prior+ZF", sample_method, "ResidualCNN-single+ZF", "ResidualCNN-domain_randomized+ZF"]),
        ("未见 10-tap 陡峭指数 PDP", practical_unseen, ["LMMSE-oracle+ZF", sample_method, "ResidualCNN-single+ZF", "ResidualCNN-domain_randomized+ZF"]),
    ]:
        for method in methods:
            row = _row(frame, method)
            lines.append(f"| {scenario} | {method} | {_metric(row, 'ber')} | {_metric(row, 'channel_nmse')} |")

    lines.extend(
        [
            "",
            "## 可审计结论",
            "",
            f"- 在匹配分布 20 dB，单一训练分布 CNN 的 BER 为 {cnn_matched['ber_mean']:.4e}，低于 DFT-LS 的 {dft_matched['ber_mean']:.4e}，但高于 sample-covariance LMMSE ({sample_count:,} 历史信道样本) 的 {sample_matched['ber_mean']:.4e}；它也不优于 oracle LMMSE。",
            f"- 在 8-tap 均匀 PDP 失配 20 dB，domain-randomized CNN 的 BER 为 {domain_mismatch['ber_mean']:.4e}，优于单一分布 CNN 的 {cnn_mismatch['ber_mean']:.4e}，但仍高于 sample-covariance LMMSE 的 {sample_mismatch['ber_mean']:.4e}。",
            f"- 在未见 10-tap 陡峭指数 PDP，domain randomization 的 BER 为 {domain_unseen['ber_mean']:.4e}，反而高于单一分布 CNN 的 {single_unseen['ber_mean']:.4e}；它不是对所有失配都稳健的万能改进。",
            f"- hard delay projection 将总体 NMSE 从 {residual['channel_nmse_mean']:.4e} 降为 {hard['channel_nmse_mean']:.4e}，但 BER 从 {residual['ber_mean']:.4e} 变为 {hard['ber_mean']:.4e}，且深衰落 NMSE 从 {residual['deep_fade_nmse_mean']:.4e} 升为 {hard['deep_fade_nmse_mean']:.4e}。因此保留“NMSE 改善不保证硬判决 BER 改善”的结论。",
            f"- 正确拆分离线/在线成本后，sample-covariance LMMSE ({sample_count:,} 样本) 的协方差构建为 {lmmse_time['offline_covariance_ms']:.3f} ms，缓存滤波器在线时间为 {lmmse_time['online_mean_ms_per_frame']:.4f} +/- {lmmse_time['online_std_ms_per_frame']:.4f} ms/帧；CNN checkpoint 加载 {cnn_time['checkpoint_load_ms']:.3f} ms，warm-up 后在线时间为 {cnn_time['online_mean_ms_per_frame']:.4f} +/- {cnn_time['online_std_ms_per_frame']:.4f} ms/帧。",
            f"- 在匹配与 8-tap uniform 失配中，sample-covariance LMMSE 的 BER/NMSE 均优于当前 CNN 且缓存后更快；但在未见 10-tap 陡峭指数 PDP，single CNN 的 BER 为 {single_unseen['ber_mean']:.4e}，略低于 sample LMMSE 的 {sample_unseen['ber_mean']:.4e}，同时 NMSE 更高（{single_unseen['channel_nmse_mean']:.4e} 对 {sample_unseen['channel_nmse_mean']:.4e}）。两者 CI 分别来自模型 seed 与测试 seed，不能据此做跨层级显著性断言；更不能宣称 CNN 普遍替代 LMMSE。",
        ]
    )
    output = root / "experiment_summary.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
