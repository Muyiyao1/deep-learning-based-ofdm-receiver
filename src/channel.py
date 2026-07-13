"""Frequency-selective Rayleigh channel, PDP, and AWGN helpers."""

from __future__ import annotations

import numpy as np


def power_delay_profile(num_taps: int, pdp: str = "exponential", exponential_decay: float = 0.45) -> np.ndarray:
    """Return a unit-sum power-delay profile used by generation and LMMSE."""

    if num_taps <= 0:
        raise ValueError("num_taps must be positive.")
    if pdp == "uniform":
        powers = np.ones(num_taps, dtype=np.float64)
    elif pdp == "exponential":
        powers = np.exp(-float(exponential_decay) * np.arange(num_taps, dtype=np.float64))
    else:
        raise ValueError(f"Unsupported PDP: {pdp}")
    return powers / np.sum(powers)


def generate_rayleigh_channel(
    num_taps: int,
    rng: np.random.Generator | None = None,
    pdp: str = "exponential",
    exponential_decay: float = 0.45,
) -> np.ndarray:
    """Generate one normalized complex Rayleigh FIR channel.

    Each realization is normalized to unit energy.  This keeps the received
    SNR convention stable while retaining the nominal PDP shape.
    """

    rng = np.random.default_rng() if rng is None else rng
    powers = power_delay_profile(num_taps, pdp=pdp, exponential_decay=exponential_decay)
    taps = (rng.normal(size=num_taps) + 1j * rng.normal(size=num_taps)) * np.sqrt(powers / 2.0)
    taps /= np.sqrt(np.sum(np.abs(taps) ** 2))
    return taps.astype(np.complex64)


def channel_frequency_response(taps: np.ndarray, num_subcarriers: int) -> np.ndarray:
    """Return physical channel response H[k] = FFT{h[l]} under unitary OFDM."""

    return np.fft.fft(np.asarray(taps), n=num_subcarriers).astype(np.complex64)


def apply_multipath_channel(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Apply causal convolution and retain waveform length.

    With CP >= channel length, the post-CP samples of every OFDM symbol have
    the circular-convolution relation used by the one-tap frequency response.
    """

    y = np.convolve(np.asarray(signal), np.asarray(taps), mode="full")
    return y[: len(signal)].astype(np.complex64)


def add_awgn(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Add complex AWGN with empirical post-channel received-signal SNR.

    ``noise_power = mean(abs(signal)**2) / 10**(snr_db / 10)``.  The returned
    noise variance is complex-sample variance E[|w|^2], also valid after the
    unitary OFDM FFT.
    """

    rng = np.random.default_rng() if rng is None else rng
    signal = np.asarray(signal, dtype=np.complex64)
    signal_power = float(np.mean(np.abs(signal) ** 2))
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape))
    return (signal + noise).astype(np.complex64), float(noise_power)


def transmit_through_channel(
    signal: np.ndarray,
    taps: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Apply multipath fading then AWGN using the project-wide SNR convention."""

    return add_awgn(apply_multipath_channel(signal, taps), snr_db=snr_db, rng=rng)
