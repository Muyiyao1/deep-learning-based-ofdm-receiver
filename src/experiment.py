"""Experiment output helpers shared by CLI scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import CNNChannelEstimator
from .plots import plot_ber, plot_channel_nmse, plot_constellation
from .utils import ensure_dir


def preferred_methods(summary: pd.DataFrame, include_train_prior: bool = False) -> list[str]:
    """Select non-duplicated ZF curves for compact report figures."""

    candidates = [
        "PerfectCSI+ZF",
        "Frame-LS-linear+ZF",
        "DFT-LS+ZF",
        "LMMSE-oracle+ZF",
        "ResidualCNN+ZF",
    ]
    if include_train_prior:
        candidates.insert(-1, "LMMSE-train-prior+ZF")
    return [method for method in candidates if method in set(summary["method"])]


def save_evaluation_artifacts(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    constellation: dict,
    output_dir: str | Path,
    title_prefix: str,
    cnn_model: CNNChannelEstimator | None = None,
    include_train_prior: bool = False,
    reference_snr_db: float = 20.0,
) -> pd.DataFrame:
    """Persist CSVs and the compact figure set directly derived from them."""

    output_dir = ensure_dir(output_dir)
    raw.to_csv(output_dir / "per_seed_results.csv", index=False)
    summary.to_csv(output_dir / "summary_results.csv", index=False)
    methods = preferred_methods(summary, include_train_prior=include_train_prior)
    plot_ber(
        summary,
        output_dir / "ber_vs_snr.png",
        f"{title_prefix}: BER (mean with 95% Student-t CI)",
        methods=methods,
    )
    estimators = [method.rsplit("+", 1)[0] for method in methods if method.endswith("+ZF") and not method.startswith("PerfectCSI")]
    plot_channel_nmse(
        summary,
        output_dir / "channel_nmse_vs_snr.png",
        f"{title_prefix}: channel NMSE (mean with 95% Student-t CI)",
        estimators=estimators,
    )
    if constellation:
        selected_methods = ["PerfectCSI+ZF", "Frame-LS-linear+ZF", "ResidualCNN+ZF"]
        selected = {key: value for key, value in constellation.items() if key in selected_methods}
        plot_constellation(selected, output_dir / "constellation_20db.png", methods=selected_methods)

    # Cached online complexity is benchmarked separately in results/*/multiseed.
    # Remove older per-frame timing tables that included repeated LMMSE solves.
    for stale_name in ("complexity.csv", "complexity_table.png"):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    return pd.DataFrame()
