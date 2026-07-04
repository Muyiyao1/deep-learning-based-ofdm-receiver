"""Frequency-selective Rayleigh channel and AWGN helpers."""

from __future__ import annotations

import numpy as np


def generate_rayleigh_channel(
    num_taps: int,
    rng: np.random.Generator | None = None,
    exponential_decay: float = 0.45,
) -> np.ndarray:
    """Generate a normalized complex Rayleigh multipath channel.

    The taps follow an exponentially decaying power-delay profile and are
    normalized so that sum_l |h_l|^2 = 1 for each realization.
    """

    if num_taps <= 0:
        raise ValueError("num_taps must be positive.")
    rng = np.random.default_rng() if rng is None else rng
    powers = np.exp(-exponential_decay * np.arange(num_taps, dtype=np.float64))
    powers = powers / np.sum(powers)
    real = rng.normal(size=num_taps)
    imag = rng.normal(size=num_taps)
    taps = (real + 1j * imag) * np.sqrt(powers / 2.0)
    energy = np.sum(np.abs(taps) ** 2)
    if energy == 0:
        taps[0] = 1.0 + 0j
    else:
        taps = taps / np.sqrt(energy)
    return taps.astype(np.complex64)


def channel_frequency_response(taps: np.ndarray, num_subcarriers: int) -> np.ndarray:
    """Return the N-point frequency response of a channel impulse response."""

    return np.fft.fft(np.asarray(taps), n=num_subcarriers).astype(np.complex64)


def apply_multipath_channel(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Apply a causal finite impulse response channel and keep the input length."""

    y = np.convolve(np.asarray(signal), np.asarray(taps), mode="full")
    return y[: len(signal)].astype(np.complex64)


def add_awgn(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Add complex AWGN using empirical received-signal power.

    SNR convention:
        noise_power = mean(|signal|^2) / 10^(SNR_dB / 10)
    """

    rng = np.random.default_rng() if rng is None else rng
    signal = np.asarray(signal)
    signal_power = float(np.mean(np.abs(signal) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape))
    return (signal + noise).astype(np.complex64), float(noise_power)


def transmit_through_channel(
    signal: np.ndarray,
    taps: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Apply multipath channel and AWGN."""

    faded = apply_multipath_channel(signal, taps)
    return add_awgn(faded, snr_db=snr_db, rng=rng)
