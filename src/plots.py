"""Consistent publication-style plots generated directly from result CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#000000"]
MARKERS = ["o", "s", "^", "D", "P", "X", "v"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-."]


def _method_sequence(df: pd.DataFrame, methods: list[str] | None) -> list[str]:
    if methods is not None:
        return [method for method in methods if method in set(df["method"])]
    return sorted(df["method"].unique())


def _apply_style(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.55, alpha=0.55)


def plot_ber(
    summary: pd.DataFrame,
    output_path: str | Path,
    title: str,
    methods: list[str] | None = None,
) -> None:
    """Plot seed-mean BER with 95% confidence intervals on a log scale."""

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=160)
    for index, method in enumerate(_method_sequence(summary, methods)):
        group = summary[summary["method"] == method].sort_values("snr_db")
        value = group["ber_plot"].to_numpy(dtype=float)
        ci = group["ber_ci95"].to_numpy(dtype=float)
        lower = np.minimum(ci, np.maximum(value - 1e-12, 0.0))
        upper = ci
        ax.errorbar(
            group["snr_db"],
            value,
            yerr=np.vstack([lower, upper]),
            color=PALETTE[index % len(PALETTE)],
            marker=MARKERS[index % len(MARKERS)],
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            linewidth=1.8,
            markersize=5.0,
            capsize=2.4,
            label=method,
        )
    ax.set_yscale("log")
    _apply_style(ax, "Received-signal SNR (dB)", "Bit error rate", title)
    ax.legend(fontsize=8, ncol=1, frameon=True)
    fig.tight_layout()
    fig.savefig(Path(output_path), bbox_inches="tight")
    plt.close(fig)


def plot_channel_nmse(
    summary: pd.DataFrame,
    output_path: str | Path,
    title: str,
    estimators: list[str] | None = None,
) -> None:
    """Plot channel NMSE once per estimator, avoiding duplicate ZF/MMSE curves."""

    source = summary[summary["equalizer"] == "ZF"]
    if estimators is not None:
        source = source[source["estimator"].isin(estimators)]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=160)
    for index, (estimator, group) in enumerate(source.groupby("estimator", sort=False)):
        group = group.sort_values("snr_db")
        value = np.maximum(group["channel_nmse_mean"].to_numpy(dtype=float), 1e-12)
        ci = group["channel_nmse_ci95"].to_numpy(dtype=float)
        ax.errorbar(
            group["snr_db"], value, yerr=ci, color=PALETTE[index % len(PALETTE)], marker=MARKERS[index % len(MARKERS)],
            linestyle=LINESTYLES[index % len(LINESTYLES)], linewidth=1.8, markersize=5.0, capsize=2.4, label=estimator,
        )
    ax.set_yscale("log")
    _apply_style(ax, "Received-signal SNR (dB)", "Channel NMSE", title)
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(Path(output_path), bbox_inches="tight")
    plt.close(fig)


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    title: str,
    ylabel: str,
    methods: list[str] | None = None,
    log_y: bool = False,
) -> None:
    """Plot a summary metric and its 95% CI when a matching CI column exists."""

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=160)
    for index, method in enumerate(_method_sequence(summary, methods)):
        group = summary[summary["method"] == method].sort_values("snr_db")
        value = group[f"{metric}_mean"].to_numpy(dtype=float)
        ci_key = f"{metric}_ci95"
        ci = group[ci_key].to_numpy(dtype=float) if ci_key in group else None
        ax.errorbar(
            group["snr_db"], value, yerr=ci, color=PALETTE[index % len(PALETTE)], marker=MARKERS[index % len(MARKERS)],
            linestyle=LINESTYLES[index % len(LINESTYLES)], linewidth=1.8, markersize=5.0, capsize=2.4, label=method,
        )
    if log_y:
        ax.set_yscale("log")
    _apply_style(ax, "Received-signal SNR (dB)", ylabel, title)
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(Path(output_path), bbox_inches="tight")
    plt.close(fig)


def plot_constellation(
    symbols_by_method: dict[str, np.ndarray],
    output_path: str | Path,
    max_points: int = 1600,
    methods: list[str] | None = None,
) -> None:
    """Save a representative equalized constellation comparison."""

    requested = [method for method in (methods or list(symbols_by_method)) if method in symbols_by_method]
    selected = [(method, symbols_by_method[method]) for method in requested[:3]]
    fig, axes = plt.subplots(1, len(selected), figsize=(3.15 * len(selected), 3.0), dpi=170, squeeze=False)
    for ax, (method, symbols) in zip(axes[0], selected):
        flat = np.asarray(symbols).reshape(-1)[:max_points]
        ax.scatter(flat.real, flat.imag, s=6, alpha=0.28, color="#0072B2", edgecolors="none")
        ax.axhline(0, color="#555555", linewidth=0.55)
        ax.axvline(0, color="#555555", linewidth=0.55)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_title(method, fontsize=8)
        ax.set_xlabel("I", fontsize=8)
        ax.set_ylabel("Q", fontsize=8)
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.45)
    fig.tight_layout()
    fig.savefig(Path(output_path), bbox_inches="tight")
    plt.close(fig)


def plot_training_loss(history: list[dict[str, float]], output_path: str | Path) -> None:
    """Save training and validation objective histories."""

    frame = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=160)
    ax.plot(frame["epoch"], frame["train_loss"], marker="o", color=PALETTE[0], label="train")
    ax.plot(frame["epoch"], frame["val_loss"], marker="s", color=PALETTE[1], label="validation")
    _apply_style(ax, "Epoch", "Composite loss", "Residual estimator training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_path), bbox_inches="tight")
    plt.close(fig)


def plot_ablation(ablation: pd.DataFrame, output_path: str | Path) -> None:
    """Plot BER and NMSE side by side for neural estimator variants."""

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=160)
    labels = ablation["variant"].tolist()
    for ax, metric, label in zip(axes, ["ber", "channel_nmse"], ["BER", "Channel NMSE"]):
        values = ablation[f"{metric}_mean"].to_numpy(dtype=float)
        errors = ablation[f"{metric}_ci95"].to_numpy(dtype=float)
        ax.bar(np.arange(len(labels)), values, yerr=errors, color=PALETTE[: len(labels)], capsize=4)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=12, ha="right")
        ax.set_yscale("log")
        _apply_style(ax, "Estimator variant", label, f"Ablation at {ablation['snr_db'].iloc[0]:.0f} dB")
    fig.tight_layout()
    fig.savefig(Path(output_path), bbox_inches="tight")
    plt.close(fig)


def plot_pilot_pattern(pilot_mask: np.ndarray, output_path: str | Path, title: str) -> None:
    """Visualize per-symbol pilot placement for the static-channel fusion experiment."""

    fig, ax = plt.subplots(figsize=(7.2, 3.7), dpi=160)
    ax.imshow(np.asarray(pilot_mask, dtype=float), aspect="auto", interpolation="nearest", cmap="Blues", origin="lower")
    _apply_style(ax, "Subcarrier index", "OFDM symbol index", title)
    fig.tight_layout()
    fig.savefig(Path(output_path), bbox_inches="tight")
    plt.close(fig)


def plot_training_seed_variability(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Plot model-seed uncertainty after averaging each model over fixed test streams."""

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=160)
    for index, (regime, group) in enumerate(summary.groupby("training_regime", sort=False)):
        group = group.sort_values("snr_db")
        for ax, metric, ylabel in zip(axes, ["ber", "channel_nmse"], ["BER", "Channel NMSE"]):
            ax.errorbar(
                group["snr_db"],
                np.maximum(group[f"{metric}_mean"], 1e-12),
                yerr=group[f"{metric}_ci95"],
                color=PALETTE[index % len(PALETTE)],
                marker=MARKERS[index % len(MARKERS)],
                linestyle=LINESTYLES[index % len(LINESTYLES)],
                capsize=2.4,
                label=regime,
            )
            ax.set_yscale("log")
            _apply_style(ax, "Received-signal SNR (dB)", ylabel, "Training-seed uncertainty (95% Student-t CI)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(output_path), bbox_inches="tight")
    plt.close(fig)


def plot_complexity_table(complexity: pd.DataFrame, output_path: str | Path) -> None:
    """Render the complexity CSV as a compact report-ready table image."""

    shown = complexity.copy()
    for column in shown.select_dtypes(include=[np.number]).columns:
        shown[column] = shown[column].map(lambda value: f"{value:.4g}")
    fig, ax = plt.subplots(figsize=(8.2, 1.0 + 0.36 * len(shown)), dpi=180)
    ax.axis("off")
    table = ax.table(cellText=shown.values, colLabels=shown.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.25)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E8F0F7")
    fig.tight_layout()
    fig.savefig(Path(output_path), bbox_inches="tight")
    plt.close(fig)


def plot_complexity_benchmark(complexity: pd.DataFrame, output_path: str | Path) -> None:
    """Plot readable offline/startup and cached-online complexity views."""

    labels = complexity["estimator"].tolist()
    online = complexity["online_mean_ms_per_frame"].to_numpy(dtype=float)
    online_std = complexity["online_std_ms_per_frame"].to_numpy(dtype=float)
    offline = (
        complexity["offline_covariance_ms"].to_numpy(dtype=float)
        + complexity["offline_filter_ms"].to_numpy(dtype=float)
        + complexity["checkpoint_load_ms"].to_numpy(dtype=float)
    )
    offline_labels = [
        "CNN checkpoint load" if name == "ResidualCNN" else f"{name} covariance + K"
        for name in labels
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), dpi=160)
    positions = np.arange(len(labels))
    axes[0].barh(positions, online, xerr=online_std, color=PALETTE[: len(labels)], capsize=3)
    axes[0].set_yticks(positions, labels)
    axes[0].set_xscale("log")
    axes[0].set_xlim(1e-3, 1.5e-1)
    axes[0].set_xticks([2e-3, 1e-2, 1e-1], ["0.002", "0.01", "0.1"])
    axes[0].minorticks_off()
    _apply_style(axes[0], "Cached online time (ms/frame, log)", "Estimator", "Warm online inference")
    axes[0].invert_yaxis()
    for y, value in enumerate(online):
        axes[0].text(value * 1.12, y, f"{value:.4f}", va="center", fontsize=8)

    axes[1].barh(positions, offline, color=PALETTE[: len(labels)])
    axes[1].set_yticks(positions, offline_labels)
    _apply_style(axes[1], "One-time initialization (ms)", "Estimator", "Offline / startup cost")
    axes[1].invert_yaxis()
    for y, value in enumerate(offline):
        axes[1].text(value + max(offline) * 0.015, y, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(output_path), bbox_inches="tight")
    plt.close(fig)
