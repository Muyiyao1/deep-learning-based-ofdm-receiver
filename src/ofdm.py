"""OFDM resource mapping, modulation, and demodulation."""

from __future__ import annotations

import numpy as np

from .config import OFDMConfig


def generate_guard_mask(cfg: OFDMConfig) -> np.ndarray:
    """Return guard tones at both occupied-band edges in FFT-bin order."""

    shifted = np.zeros(cfg.num_subcarriers, dtype=bool)
    count = cfg.guard_subcarriers_each_side
    if count:
        shifted[:count] = True
        shifted[-count:] = True
    return np.fft.ifftshift(shifted)


def generate_dc_mask(cfg: OFDMConfig) -> np.ndarray:
    """Return the DC-null mask; bin zero is the DC bin in NumPy FFT order."""

    mask = np.zeros(cfg.num_subcarriers, dtype=bool)
    if cfg.null_dc:
        mask[0] = True
    return mask


def generate_active_mask(cfg: OFDMConfig) -> np.ndarray:
    """Return tones available for pilots or data, excluding guard and DC."""

    return ~(generate_guard_mask(cfg) | generate_dc_mask(cfg))


def generate_pilot_mask(cfg: OFDMConfig) -> np.ndarray:
    """Create comb or staggered-comb pilot masks with shape ``[T, N]``.

    A staggered pattern rotates the comb across OFDM symbols.  In a static
    channel this provides additional *real pilot observations* across the
    frame; all conventional estimators are given the same union of these
    observations as the neural estimator.
    """

    active_indices = np.flatnonzero(generate_active_mask(cfg))
    mask = np.zeros((cfg.num_ofdm_symbols, cfg.num_subcarriers), dtype=bool)
    for t in range(cfg.num_ofdm_symbols):
        offset = cfg.pilot_offset
        if cfg.pilot_pattern == "staggered":
            offset = (offset + t * cfg.pilot_stagger_step) % cfg.pilot_spacing
        mask[t, active_indices[offset:: cfg.pilot_spacing]] = True
    return mask


def generate_data_mask(cfg: OFDMConfig) -> np.ndarray:
    """Return data mask, disjoint from pilot, guard, and DC tones."""

    active = generate_active_mask(cfg)[np.newaxis, :]
    return np.repeat(active, cfg.num_ofdm_symbols, axis=0) & ~generate_pilot_mask(cfg)


def generate_resource_masks(cfg: OFDMConfig) -> dict[str, np.ndarray]:
    """Build all disjoint resource masks used by transmitter and tests."""

    pilot = generate_pilot_mask(cfg)
    data = generate_data_mask(cfg)
    guard = np.repeat(generate_guard_mask(cfg)[np.newaxis, :], cfg.num_ofdm_symbols, axis=0)
    dc = np.repeat(generate_dc_mask(cfg)[np.newaxis, :], cfg.num_ofdm_symbols, axis=0)
    return {"pilot": pilot, "data": data, "guard": guard, "dc": dc}


def pilot_pattern_statistics(cfg: OFDMConfig) -> dict[str, float | int]:
    """Summarize the frame-level pilot observations used by all estimators.

    The values deliberately distinguish a sparse *single-symbol* pilot pattern
    from the union available to a block-fading, frame-level receiver.
    """

    masks = generate_resource_masks(cfg)
    pilot = masks["pilot"]
    active = generate_active_mask(cfg)
    observed = pilot.any(axis=0) & active
    counts = pilot[:, active].sum(axis=0)
    per_symbol = pilot.sum(axis=1)
    active_count = int(active.sum())
    observations = int(pilot.sum())
    return {
        "fft_size": int(cfg.num_subcarriers),
        "ofdm_symbols": int(cfg.num_ofdm_symbols),
        "guard_subcarriers_total": int(2 * cfg.guard_subcarriers_each_side),
        "dc_subcarriers": int(cfg.null_dc),
        "active_subcarriers": active_count,
        "pilot_observations_per_symbol_min": int(per_symbol.min()),
        "pilot_observations_per_symbol_max": int(per_symbol.max()),
        "pilot_observations_per_symbol_mean": float(per_symbol.mean()),
        "pilot_observations_per_frame": observations,
        "pilot_overhead": float(observations / max(cfg.num_ofdm_symbols * active_count, 1)),
        "pilot_union_coverage": float(observed.sum() / max(active_count, 1)),
        "mean_observations_per_active_subcarrier": float(counts.mean()) if counts.size else 0.0,
        "minimum_observations_per_active_subcarrier": int(counts.min()) if counts.size else 0,
        "maximum_observations_per_active_subcarrier": int(counts.max()) if counts.size else 0,
    }


def build_resource_grid(
    data_symbols: np.ndarray,
    cfg: OFDMConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert data and known pilots while leaving guard/DC tones at zero."""

    masks = generate_resource_masks(cfg)
    pilot_mask, data_mask = masks["pilot"], masks["data"]
    expected = int(np.sum(data_mask))
    data_symbols = np.asarray(data_symbols, dtype=np.complex64).reshape(-1)
    if data_symbols.size != expected:
        raise ValueError(f"Expected {expected} data symbols, got {data_symbols.size}.")
    grid = np.zeros((cfg.num_ofdm_symbols, cfg.num_subcarriers), dtype=np.complex64)
    grid[pilot_mask] = cfg.pilot_symbol
    grid[data_mask] = data_symbols
    return grid, pilot_mask, data_mask


def ofdm_modulate(freq_grid: np.ndarray, cp_length: int) -> np.ndarray:
    """Unitary IFFT, CP insertion, and OFDM-symbol serialization."""

    freq_grid = np.asarray(freq_grid, dtype=np.complex64)
    if freq_grid.ndim != 2:
        raise ValueError("freq_grid must have shape [num_symbols, num_subcarriers].")
    num_subcarriers = freq_grid.shape[1]
    time_grid = np.fft.ifft(freq_grid, axis=1) * np.sqrt(num_subcarriers)
    with_cp = np.concatenate([time_grid[:, -cp_length:], time_grid], axis=1)
    return with_cp.reshape(-1).astype(np.complex64)


def ofdm_demodulate(signal: np.ndarray, cfg: OFDMConfig) -> np.ndarray:
    """Remove CP and use the matched unitary FFT for an OFDM frame."""

    symbol_len = cfg.num_subcarriers + cfg.cp_length
    expected_len = cfg.num_ofdm_symbols * symbol_len
    signal = np.asarray(signal, dtype=np.complex64).reshape(-1)
    if signal.size < expected_len:
        raise ValueError(f"Expected at least {expected_len} samples, got {signal.size}.")
    framed = signal[:expected_len].reshape(cfg.num_ofdm_symbols, symbol_len)
    return (np.fft.fft(framed[:, cfg.cp_length :], axis=1) / np.sqrt(cfg.num_subcarriers)).astype(np.complex64)
