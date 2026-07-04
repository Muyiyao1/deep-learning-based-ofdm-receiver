"""Traditional channel estimators and equalizers."""

from __future__ import annotations

import numpy as np


def sparse_ls_channel_estimate(
    received_grid: np.ndarray,
    transmitted_grid: np.ndarray,
    pilot_mask: np.ndarray,
) -> np.ndarray:
    """Compute LS channel estimates only at pilot positions."""

    y = np.asarray(received_grid, dtype=np.complex64)
    x = np.asarray(transmitted_grid, dtype=np.complex64)
    sparse = np.zeros_like(y, dtype=np.complex64)
    sparse[pilot_mask] = y[pilot_mask] / x[pilot_mask]
    return sparse


def interpolate_ls_channel(sparse_ls: np.ndarray, pilot_mask: np.ndarray) -> np.ndarray:
    """Linearly interpolate LS channel estimates over frequency.

    Real and imaginary parts are interpolated independently for every OFDM
    symbol. Edge subcarriers outside the pilot span use nearest extrapolation.
    """

    sparse_ls = np.asarray(sparse_ls, dtype=np.complex64)
    pilot_mask = np.asarray(pilot_mask, dtype=bool)
    if sparse_ls.shape != pilot_mask.shape:
        raise ValueError("sparse_ls and pilot_mask must have the same shape.")
    num_symbols, num_subcarriers = sparse_ls.shape
    subcarriers = np.arange(num_subcarriers)
    full = np.zeros_like(sparse_ls, dtype=np.complex64)
    for t in range(num_symbols):
        pilot_idx = np.flatnonzero(pilot_mask[t])
        if pilot_idx.size < 2:
            raise ValueError("At least two pilots per OFDM symbol are required for interpolation.")
        real = np.interp(subcarriers, pilot_idx, np.real(sparse_ls[t, pilot_idx]))
        imag = np.interp(subcarriers, pilot_idx, np.imag(sparse_ls[t, pilot_idx]))
        full[t] = real + 1j * imag
    return full.astype(np.complex64)


def ls_channel_estimate(
    received_grid: np.ndarray,
    transmitted_grid: np.ndarray,
    pilot_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sparse and interpolated LS channel estimates."""

    sparse = sparse_ls_channel_estimate(received_grid, transmitted_grid, pilot_mask)
    full = interpolate_ls_channel(sparse, pilot_mask)
    return sparse, full


def zf_equalize(received_grid: np.ndarray, channel_estimate: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-forcing equalization."""

    denom = np.where(np.abs(channel_estimate) < eps, eps + 0j, channel_estimate)
    return (received_grid / denom).astype(np.complex64)


def mmse_equalize(received_grid: np.ndarray, channel_estimate: np.ndarray, noise_var: float) -> np.ndarray:
    """Single-tap MMSE equalization for unit-average-power QAM symbols."""

    h = np.asarray(channel_estimate, dtype=np.complex64)
    y = np.asarray(received_grid, dtype=np.complex64)
    denom = np.abs(h) ** 2 + float(noise_var)
    return (np.conj(h) * y / denom).astype(np.complex64)
