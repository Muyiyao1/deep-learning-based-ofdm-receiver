"""Reproducible offline/online complexity measurements for channel estimators."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from .config import OFDMConfig
from .dataset import generate_frame
from .estimators import LMMSEPrior
from .evaluate import build_lmmse_filter_bank
from .models import CNNChannelEstimator


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cnn_conv_macs(model: CNNChannelEstimator, input_length: int) -> int:
    """Count convolution MACs for one forward pass, excluding FFT projection."""

    macs = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            macs += int(input_length * module.in_channels * module.out_channels * module.kernel_size[0] / module.groups)
    return int(macs)


def _measure(callable_fn, device: torch.device, repeats: int, frames: int) -> tuple[float, float]:
    measurements: list[float] = []
    for _ in range(repeats):
        _synchronize(device)
        start = perf_counter()
        callable_fn()
        _synchronize(device)
        measurements.append(1e3 * (perf_counter() - start) / frames)
    return float(np.mean(measurements)), float(np.std(measurements, ddof=1)) if len(measurements) > 1 else 0.0


def benchmark_cached_estimators(
    cfg: OFDMConfig,
    cnn_model: CNNChannelEstimator,
    cnn_device: torch.device,
    priors: dict[str, LMMSEPrior],
    snr_db: float = 20.0,
    checkpoint_path: str | Path | None = None,
    checkpoint_load_ms: float | None = None,
) -> pd.DataFrame:
    """Benchmark cached online LMMSE and warm CNN inference on the same batch.

    The reported LMMSE online time contains only ``K @ H_LS``.  Covariance and
    filter construction times are reported separately, which prevents a
    one-off matrix factorization from being misrepresented as per-frame cost.
    """

    frame_count = int(cfg.complexity_benchmark_frames)
    rng = np.random.default_rng(cfg.random_seed + 7_700_000)
    frames = [generate_frame(cfg, float(snr_db), rng) for _ in range(frame_count)]
    filters = build_lmmse_filter_bank(cfg, priors, [float(snr_db)])
    warmups = int(cfg.complexity_warmup_iterations)
    repeats = int(cfg.complexity_benchmark_repeats)
    cnn_inputs = torch.from_numpy(np.stack([frame.cnn_input for frame in frames])).float().to(cnn_device)
    checkpoint_bytes = Path(checkpoint_path).stat().st_size if checkpoint_path is not None and Path(checkpoint_path).exists() else 0
    rows: list[dict[str, float | int | str]] = []

    cnn_model.eval()
    with torch.no_grad():
        for _ in range(warmups):
            cnn_model(cnn_inputs)
        _synchronize(cnn_device)
        cnn_mean, cnn_std = _measure(lambda: cnn_model(cnn_inputs), cnn_device, repeats, frame_count)
    parameter_count = sum(parameter.numel() for parameter in cnn_model.parameters())
    approximate_memory = parameter_count * 4 + int(cnn_inputs.numel()) * 4 + frame_count * 2 * cfg.num_subcarriers * 4
    rows.append(
        {
            "estimator": "ResidualCNN",
            "offline_covariance_ms": 0.0,
            "offline_filter_ms": 0.0,
            "checkpoint_load_ms": 0.0 if checkpoint_load_ms is None else float(checkpoint_load_ms),
            "online_mean_ms_per_frame": cnn_mean,
            "online_std_ms_per_frame": cnn_std,
            "batch_size": frame_count,
            "warmup_iterations": warmups,
            "timing_repeats": repeats,
            "parameter_count": parameter_count,
            "conv_macs_per_frame": cnn_conv_macs(cnn_model, cfg.num_subcarriers),
            "checkpoint_bytes": int(checkpoint_bytes),
            "prior_storage_bytes": 0,
            "filter_storage_bytes": 0,
            "approximate_memory_bytes": approximate_memory,
        }
    )

    for name, prior in priors.items():
        prepared = filters[(name, float(snr_db))]
        observations = np.stack([frame.frame_ls[prepared.observed_indices] for frame in frames], axis=1).astype(np.complex128)

        def apply_filter() -> np.ndarray:
            return prepared.full_mean[:, None] + prepared.filter_matrix @ (observations - prepared.observed_mean[:, None])

        for _ in range(warmups):
            apply_filter()
        lmmse_mean, lmmse_std = _measure(apply_filter, torch.device("cpu"), repeats, frame_count)
        rows.append(
            {
                "estimator": name,
                "offline_covariance_ms": prior.covariance_build_time_ms,
                "offline_filter_ms": prepared.build_time_ms,
                "checkpoint_load_ms": 0.0,
                "online_mean_ms_per_frame": lmmse_mean,
                "online_std_ms_per_frame": lmmse_std,
                "batch_size": frame_count,
                "warmup_iterations": warmups,
                "timing_repeats": repeats,
                "parameter_count": 0,
                "conv_macs_per_frame": 0,
                "checkpoint_bytes": 0,
                "prior_storage_bytes": prior.storage_bytes,
                "filter_storage_bytes": prepared.storage_bytes,
                "approximate_memory_bytes": prepared.storage_bytes,
            }
        )
    return pd.DataFrame(rows)
