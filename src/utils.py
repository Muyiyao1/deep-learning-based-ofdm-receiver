"""General utilities for reproducibility and metrics."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bit_error_rate(reference_bits: np.ndarray, estimated_bits: np.ndarray) -> float:
    """Compute BER between two bit arrays."""

    ref = np.asarray(reference_bits).reshape(-1)
    est = np.asarray(estimated_bits).reshape(-1)
    if ref.size != est.size:
        raise ValueError(f"Bit arrays must have the same length, got {ref.size} and {est.size}.")
    if ref.size == 0:
        return 0.0
    return float(np.mean(ref != est))


def complex_mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute mean squared error for complex arrays."""

    ref = np.asarray(reference)
    est = np.asarray(estimate)
    if ref.shape != est.shape:
        raise ValueError(f"Arrays must have the same shape, got {ref.shape} and {est.shape}.")
    return float(np.mean(np.abs(ref - est) ** 2))


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def complex_to_two_channel(x: np.ndarray) -> np.ndarray:
    """Convert a complex array [T, N] to a two-channel real array [2, T, N]."""

    return np.stack([np.real(x), np.imag(x)], axis=0).astype(np.float32)


def two_channel_to_complex(x: np.ndarray) -> np.ndarray:
    """Convert a two-channel real array [2, T, N] to a complex array [T, N]."""

    x = np.asarray(x)
    if x.shape[0] != 2:
        raise ValueError("Expected first dimension to have size 2.")
    return (x[0] + 1j * x[1]).astype(np.complex64)
