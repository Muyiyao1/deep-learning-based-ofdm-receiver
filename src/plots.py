"""Plotting helpers for BER, channel MSE, training curves, and constellations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ber(df: pd.DataFrame, output_path: str | Path, title: str = "BER vs SNR") -> None:
    """Save a semilog BER curve grouped by method."""

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=140)
    method_order = ["LS+ZF", "LS+MMSE", "CNN+ZF", "CNN+MMSE"]
    methods = [method for method in method_order if method in set(df["method"])]
    methods.extend(method for method in sorted(set(df["method"])) if method not in methods)
    x_offsets = np.linspace(-0.16, 0.16, num=len(methods)) if len(methods) > 1 else [0.0]
    styles = {
        "LS+ZF": {"marker": "o", "linestyle": "-", "fillstyle": "full"},
        "LS+MMSE": {"marker": "s", "linestyle": "--", "fillstyle": "none"},
        "CNN+ZF": {"marker": "^", "linestyle": "-", "fillstyle": "full"},
        "CNN+MMSE": {"marker": "D", "linestyle": "--", "fillstyle": "none"},
    }
    for offset, method in zip(x_offsets, methods):
        group = df[df["method"] == method]
        group = group.sort_values("snr_db")
        ber = np.maximum(group["ber"].to_numpy(dtype=float), 1e-6)
        style = styles.get(method, {"marker": "o", "linestyle": "-", "fillstyle": "full"})
        ax.semilogy(
            group["snr_db"].to_numpy(dtype=float) + offset,
            ber,
            marker=style["marker"],
            linestyle=style["linestyle"],
            fillstyle=style["fillstyle"],
            linewidth=1.8,
            markersize=5.2,
            label=method,
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.65)
    ax.legend()
    if len(methods) > 1:
        ax.text(
            0.5,
            -0.18,
            "Small horizontal offsets are used only to reveal overlapping curves.",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#475569",
        )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_channel_mse(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save channel-estimation MSE curves grouped by method."""

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=140)
    for method, group in df.groupby("method"):
        group = group.sort_values("snr_db")
        mse = np.maximum(group["channel_mse"].to_numpy(dtype=float), 1e-8)
        ax.semilogy(group["snr_db"], mse, marker="s", linewidth=1.8, label=method)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Channel MSE")
    ax.set_title("Channel Estimation MSE vs SNR")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.65)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    title: str,
    ylabel: str,
    log_y: bool = False,
) -> None:
    """Plot an arbitrary receiver metric grouped by method."""

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=140)
    method_order = ["LS+ZF", "LS+MMSE", "CNN+ZF", "CNN+MMSE"]
    methods = [method for method in method_order if method in set(df["method"])]
    methods.extend(method for method in sorted(set(df["method"])) if method not in methods)
    styles = {
        "LS+ZF": {"marker": "o", "linestyle": "-"},
        "LS+MMSE": {"marker": "s", "linestyle": "--"},
        "CNN+ZF": {"marker": "^", "linestyle": "-"},
        "CNN+MMSE": {"marker": "D", "linestyle": "--"},
    }
    for method in methods:
        group = df[df["method"] == method].sort_values("snr_db")
        values = group[metric].to_numpy(dtype=float)
        if log_y:
            values = np.maximum(values, 1e-8)
        style = styles.get(method, {"marker": "o", "linestyle": "-"})
        plot_fn = ax.semilogy if log_y else ax.plot
        plot_fn(
            group["snr_db"].to_numpy(dtype=float),
            values,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=5.2,
            label=method,
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.65)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_relative_gain(
    df: pd.DataFrame,
    output_path: str | Path,
    baseline_method: str = "LS+MMSE",
    candidate_method: str = "CNN+MMSE",
) -> None:
    """Plot percentage reduction of BER, SER, and channel MSE versus a baseline."""

    output_path = Path(output_path)
    baseline = df[df["method"] == baseline_method].sort_values("snr_db")
    candidate = df[df["method"] == candidate_method].sort_values("snr_db")
    if baseline.empty or candidate.empty:
        return
    merged = baseline.merge(candidate, on="snr_db", suffixes=("_baseline", "_candidate"))
    metrics = {
        "BER reduction": ("ber_baseline", "ber_candidate"),
        "SER reduction": ("ser_baseline", "ser_candidate"),
        "Channel MSE reduction": ("channel_mse_baseline", "channel_mse_candidate"),
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=140)
    for label, (base_col, cand_col) in metrics.items():
        gain = 100.0 * (merged[base_col].to_numpy(dtype=float) - merged[cand_col].to_numpy(dtype=float))
        gain = gain / np.maximum(merged[base_col].to_numpy(dtype=float), 1e-12)
        ax.plot(merged["snr_db"], gain, marker="o", linewidth=1.8, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.65)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Reduction vs LS+MMSE (%)")
    ax.set_title("CNN+MMSE Relative Gain over LS+MMSE")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.65)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_training_loss(history: list[dict[str, float]], output_path: str | Path) -> None:
    """Save training and validation loss curves."""

    output_path = Path(output_path)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=140)
    ax.plot(epochs, train_loss, marker="o", label="Train")
    ax.plot(epochs, val_loss, marker="s", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("CNN Channel Estimator Training")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.65)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_constellation(
    symbols_by_method: dict[str, np.ndarray],
    output_path: str | Path,
    max_points: int = 2500,
) -> None:
    """Save equalized constellation examples for one or more methods."""

    output_path = Path(output_path)
    num_methods = len(symbols_by_method)
    fig, axes = plt.subplots(1, num_methods, figsize=(4.4 * num_methods, 4.2), dpi=140, squeeze=False)
    for ax, (method, symbols) in zip(axes[0], symbols_by_method.items()):
        flat = np.asarray(symbols).reshape(-1)
        if flat.size > max_points:
            flat = flat[:max_points]
        ax.scatter(np.real(flat), np.imag(flat), s=8, alpha=0.35, edgecolors="none")
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.axvline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.set_title(method)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.55)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
