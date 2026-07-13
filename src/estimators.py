"""Conventional OFDM channel estimators and one-tap equalizers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.interpolate import CubicSpline

from .channel import power_delay_profile


def sparse_ls_channel_estimate(
    received_grid: np.ndarray,
    transmitted_grid: np.ndarray,
    pilot_mask: np.ndarray,
) -> np.ndarray:
    """Compute LS estimates strictly at pilot locations and zero elsewhere."""

    y = np.asarray(received_grid, dtype=np.complex64)
    x = np.asarray(transmitted_grid, dtype=np.complex64)
    pilots = np.asarray(pilot_mask, dtype=bool)
    if y.shape != x.shape or y.shape != pilots.shape:
        raise ValueError("received_grid, transmitted_grid, and pilot_mask must share a shape.")
    sparse = np.zeros_like(y, dtype=np.complex64)
    sparse[pilots] = y[pilots] / x[pilots]
    return sparse


def _periodic_interpolate(values: np.ndarray, observed_mask: np.ndarray, kind: str = "linear") -> np.ndarray:
    """Interpolate a frequency response on the periodic DFT frequency axis."""

    values = np.asarray(values, dtype=np.complex64).reshape(-1)
    observed = np.asarray(observed_mask, dtype=bool).reshape(-1)
    if values.size != observed.size:
        raise ValueError("values and observed_mask must have equal length.")
    indices = np.flatnonzero(observed)
    if indices.size == 0:
        raise ValueError("At least one pilot observation is required.")
    if indices.size == 1:
        return np.full_like(values, values[indices[0]], dtype=np.complex64)

    n = values.size
    extended_indices = np.concatenate([indices - n, indices, indices + n])
    extended_values = np.concatenate([values[indices], values[indices], values[indices]])
    query = np.arange(n)
    if kind == "linear" or indices.size < 4:
        real = np.interp(query, extended_indices, extended_values.real)
        imag = np.interp(query, extended_indices, extended_values.imag)
    elif kind == "cubic":
        real = CubicSpline(extended_indices, extended_values.real)(query)
        imag = CubicSpline(extended_indices, extended_values.imag)(query)
    else:
        raise ValueError(f"Unsupported interpolation kind: {kind}")
    return (real + 1j * imag).astype(np.complex64)


def interpolate_ls_channel(sparse_ls: np.ndarray, pilot_mask: np.ndarray, kind: str = "linear") -> np.ndarray:
    """Interpolate per-symbol LS pilot estimates over frequency."""

    sparse = np.asarray(sparse_ls, dtype=np.complex64)
    pilots = np.asarray(pilot_mask, dtype=bool)
    if sparse.shape != pilots.shape or sparse.ndim != 2:
        raise ValueError("sparse_ls and pilot_mask must both have shape [T, N].")
    return np.stack([_periodic_interpolate(sparse[t], pilots[t], kind=kind) for t in range(sparse.shape[0])]).astype(np.complex64)


def ls_channel_estimate(
    received_grid: np.ndarray,
    transmitted_grid: np.ndarray,
    pilot_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible per-symbol LS plus periodic-linear interpolation."""

    sparse = sparse_ls_channel_estimate(received_grid, transmitted_grid, pilot_mask)
    return sparse, interpolate_ls_channel(sparse, pilot_mask, kind="linear")


def frame_average_ls(
    sparse_ls: np.ndarray,
    pilot_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average repeated frame-level pilot observations without using data tones.

    Returns ``(mean_ls, observed_mask, observation_count)``.  In a staggered
    pattern, the union can span more subcarriers than a single OFDM symbol.
    """

    sparse = np.asarray(sparse_ls, dtype=np.complex64)
    pilots = np.asarray(pilot_mask, dtype=bool)
    if sparse.shape != pilots.shape or sparse.ndim != 2:
        raise ValueError("sparse_ls and pilot_mask must both have shape [T, N].")
    count = pilots.sum(axis=0).astype(np.int32)
    observed = count > 0
    mean = np.zeros(sparse.shape[1], dtype=np.complex64)
    mean[observed] = sparse[:, observed].sum(axis=0) / count[observed]
    return mean, observed, count


def frame_average_interpolated_ls(
    sparse_ls: np.ndarray,
    pilot_mask: np.ndarray,
    kind: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Frame-average LS, interpolate once, and repeat over the static frame."""

    mean, observed, count = frame_average_ls(sparse_ls, pilot_mask)
    estimate = _periodic_interpolate(mean, observed, kind=kind)
    return np.repeat(estimate[np.newaxis, :], sparse_ls.shape[0], axis=0), mean, observed, count


def dft_denoise_channel(channel_estimate: np.ndarray, channel_length: int) -> np.ndarray:
    """Project a frequency response onto a finite delay support of known length."""

    h = np.asarray(channel_estimate, dtype=np.complex64)
    if h.ndim == 2:
        return np.stack([dft_denoise_channel(row, channel_length) for row in h]).astype(np.complex64)
    if h.ndim != 1:
        raise ValueError("channel_estimate must have shape [N] or [T, N].")
    if not 1 <= channel_length <= h.size:
        raise ValueError("channel_length must lie in [1, N].")
    delay = np.fft.ifft(h)
    delay[channel_length:] = 0.0
    return np.fft.fft(delay).astype(np.complex64)


def channel_covariance(
    num_subcarriers: int,
    num_taps: int,
    pdp: str = "exponential",
    exponential_decay: float = 0.45,
) -> np.ndarray:
    """Return R_H under H[k] = FFT{h[l]} and the supplied PDP."""

    bins = np.arange(num_subcarriers)
    lag = bins[:, np.newaxis] - bins[np.newaxis, :]
    taps = np.arange(num_taps)
    powers = power_delay_profile(num_taps, pdp=pdp, exponential_decay=exponential_decay)
    return np.sum(
        powers[np.newaxis, np.newaxis, :] * np.exp(-2j * np.pi * lag[:, :, np.newaxis] * taps / num_subcarriers),
        axis=2,
    ).astype(np.complex128)


@dataclass
class LMMSEPrior:
    """A channel mean/covariance prior used by a cached LMMSE filter."""

    name: str
    mean: np.ndarray
    covariance: np.ndarray
    source: str
    sample_count: int
    covariance_build_time_ms: float
    diagonal_loading: float = 0.0
    minimum_eigenvalue: float = 0.0

    @property
    def storage_bytes(self) -> int:
        return int(np.asarray(self.mean).nbytes + np.asarray(self.covariance).nbytes)


@dataclass
class PreparedLMMSEFilter:
    """Offline LMMSE filter for one pilot geometry and nominal noise variance."""

    name: str
    observed_indices: np.ndarray
    full_mean: np.ndarray
    observed_mean: np.ndarray
    filter_matrix: np.ndarray
    nominal_noise_var: float
    build_time_ms: float
    prior_storage_bytes: int

    @property
    def storage_bytes(self) -> int:
        return int(self.filter_matrix.nbytes + self.full_mean.nbytes + self.observed_mean.nbytes + self.prior_storage_bytes)

    def estimate(self, averaged_ls: np.ndarray) -> np.ndarray:
        """Apply only the cached matrix-vector product during online inference."""

        values = np.asarray(averaged_ls, dtype=np.complex128).reshape(-1)
        if values.size != self.full_mean.size:
            raise ValueError("averaged_ls has an incompatible subcarrier dimension.")
        estimate = self.full_mean + self.filter_matrix @ (values[self.observed_indices] - self.observed_mean)
        return estimate.astype(np.complex64)


def oracle_lmmse_prior(
    num_subcarriers: int,
    num_taps: int,
    pdp: str = "exponential",
    exponential_decay: float = 0.45,
    name: str = "LMMSE-oracle",
) -> LMMSEPrior:
    """Construct a zero-mean analytic covariance prior from a known PDP."""

    start = perf_counter()
    covariance = channel_covariance(num_subcarriers, num_taps, pdp=pdp, exponential_decay=exponential_decay)
    return LMMSEPrior(
        name=name,
        mean=np.zeros(num_subcarriers, dtype=np.complex128),
        covariance=covariance,
        source=f"analytic:{pdp}:{num_taps}",
        sample_count=0,
        covariance_build_time_ms=1e3 * (perf_counter() - start),
    )


def regularize_sample_covariance(
    covariance: np.ndarray,
    diagonal_loading: float = 0.02,
    minimum_eigenvalue: float = 1e-6,
) -> np.ndarray:
    """Shrink a sample covariance and enforce a positive numerical floor."""

    covariance = np.asarray(covariance, dtype=np.complex128)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    hermitian = 0.5 * (covariance + covariance.conj().T)
    diagonal = np.diag(np.diag(hermitian))
    shrunk = (1.0 - float(diagonal_loading)) * hermitian + float(diagonal_loading) * diagonal
    minimum = float(np.min(np.linalg.eigvalsh(shrunk)).real)
    if minimum < float(minimum_eigenvalue):
        shrunk = shrunk + np.eye(shrunk.shape[0], dtype=np.complex128) * (float(minimum_eigenvalue) - minimum)
    return (0.5 * (shrunk + shrunk.conj().T)).astype(np.complex128)


def sample_covariance_lmmse_prior(
    num_subcarriers: int,
    num_taps: int,
    pdp: str,
    exponential_decay: float,
    sample_count: int,
    seed: int,
    diagonal_loading: float = 0.02,
    minimum_eigenvalue: float = 1e-6,
    name: str | None = None,
) -> LMMSEPrior:
    """Estimate a practical LMMSE prior from independent historical channels."""

    if sample_count <= 1:
        raise ValueError("sample_count must exceed one for a sample covariance.")
    start = perf_counter()
    rng = np.random.default_rng(int(seed))
    powers = power_delay_profile(num_taps, pdp=pdp, exponential_decay=exponential_decay)
    taps = (rng.normal(size=(sample_count, num_taps)) + 1j * rng.normal(size=(sample_count, num_taps))) * np.sqrt(powers[None, :] / 2.0)
    taps /= np.sqrt(np.sum(np.abs(taps) ** 2, axis=1, keepdims=True))
    responses = np.fft.fft(taps, n=num_subcarriers, axis=1).astype(np.complex128)
    mean = responses.mean(axis=0)
    centered = responses - mean
    covariance = centered.T @ centered.conj() / float(sample_count - 1)
    covariance = regularize_sample_covariance(
        covariance,
        diagonal_loading=diagonal_loading,
        minimum_eigenvalue=minimum_eigenvalue,
    )
    return LMMSEPrior(
        name=name or f"LMMSE-sample-{sample_count}",
        mean=mean.astype(np.complex128),
        covariance=covariance,
        source=f"sample:{pdp}:{num_taps}",
        sample_count=int(sample_count),
        covariance_build_time_ms=1e3 * (perf_counter() - start),
        diagonal_loading=float(diagonal_loading),
        minimum_eigenvalue=float(minimum_eigenvalue),
    )


def prepare_lmmse_filter(
    prior: LMMSEPrior,
    observed_mask: np.ndarray,
    observation_count: np.ndarray,
    noise_var: float,
    pilot_energy: float,
) -> PreparedLMMSEFilter:
    """Precompute ``K = R_hp (R_pp + R_nn)^-1`` outside the frame loop."""

    observed = np.flatnonzero(np.asarray(observed_mask, dtype=bool))
    if observed.size == 0:
        raise ValueError("LMMSE requires pilot observations.")
    counts = np.asarray(observation_count, dtype=np.float64).reshape(-1)
    if counts.size != prior.mean.size:
        raise ValueError("observation_count has an incompatible subcarrier dimension.")
    start = perf_counter()
    covariance = np.asarray(prior.covariance, dtype=np.complex128)
    r_hp = covariance[:, observed]
    r_pp = covariance[np.ix_(observed, observed)]
    noise_diag = float(noise_var) / max(float(pilot_energy), 1e-12) / np.maximum(counts[observed], 1.0)
    filter_matrix = r_hp @ np.linalg.inv(r_pp + np.diag(noise_diag))
    return PreparedLMMSEFilter(
        name=prior.name,
        observed_indices=observed.astype(np.int64),
        full_mean=np.asarray(prior.mean, dtype=np.complex128),
        observed_mean=np.asarray(prior.mean, dtype=np.complex128)[observed],
        filter_matrix=filter_matrix.astype(np.complex128),
        nominal_noise_var=float(noise_var),
        build_time_ms=1e3 * (perf_counter() - start),
        prior_storage_bytes=prior.storage_bytes,
    )


def lmmse_channel_estimate(
    averaged_ls: np.ndarray,
    observed_mask: np.ndarray,
    observation_count: np.ndarray,
    noise_var: float,
    pilot_energy: float,
    num_taps: int,
    pdp: str,
    exponential_decay: float,
) -> np.ndarray:
    """Oracle covariance LMMSE estimator from frame-averaged pilot LS values.

    For one observation per pilot, this is exactly
    ``R_hp (R_pp + sigma^2/Ep I)^(-1) H_LS``.  Repeated pilot observations are
    averaged first, so their diagonal noise variance is ``sigma^2/(Ep*n_k)``.
    """

    avg = np.asarray(averaged_ls, dtype=np.complex64).reshape(-1)
    prior = oracle_lmmse_prior(avg.size, num_taps, pdp=pdp, exponential_decay=exponential_decay)
    prepared = prepare_lmmse_filter(prior, observed_mask, observation_count, noise_var, pilot_energy)
    return prepared.estimate(avg)


def zf_equalize(received_grid: np.ndarray, channel_estimate: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Single-tap zero-forcing equalization."""

    h = np.asarray(channel_estimate, dtype=np.complex64)
    denom = np.where(np.abs(h) < eps, eps + 0j, h)
    return (np.asarray(received_grid, dtype=np.complex64) / denom).astype(np.complex64)


def mmse_equalize(
    received_grid: np.ndarray,
    channel_estimate: np.ndarray,
    noise_var: float,
    debias: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Single-tap LMMSE equalizer with optional unbiased decision scaling.

    The raw linear-MMSE output has gain ``|H|^2/(|H|^2+sigma^2)``.  Before
    fixed QAM hard decisions we remove this gain.  In this uncoded single-tap
    model, the debiased result equals ZF for the same channel estimate; this is
    expected and avoids artificial 16QAM threshold bias.
    """

    h = np.asarray(channel_estimate, dtype=np.complex64)
    y = np.asarray(received_grid, dtype=np.complex64)
    power = np.abs(h) ** 2
    raw = np.conj(h) * y / np.maximum(power + float(noise_var), eps)
    if not debias:
        return raw.astype(np.complex64)
    gain = power / np.maximum(power + float(noise_var), eps)
    return (raw / np.maximum(gain, eps)).astype(np.complex64)
