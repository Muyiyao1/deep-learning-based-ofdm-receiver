"""OFDM grid construction, modulation, and demodulation."""

from __future__ import annotations

import numpy as np

from .config import OFDMConfig


def generate_pilot_mask(cfg: OFDMConfig) -> np.ndarray:
    """Create a comb-type pilot mask with shape [T, N]."""

    mask = np.zeros((cfg.num_ofdm_symbols, cfg.num_subcarriers), dtype=bool)
    mask[:, cfg.pilot_offset :: cfg.pilot_spacing] = True
    return mask


def generate_data_mask(cfg: OFDMConfig) -> np.ndarray:
    """Return the complement of the pilot mask."""

    return ~generate_pilot_mask(cfg)


def build_resource_grid(data_symbols: np.ndarray, cfg: OFDMConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert data and known pilots into a frequency-domain OFDM grid."""

    pilot_mask = generate_pilot_mask(cfg)
    data_mask = ~pilot_mask
    expected = int(np.sum(data_mask))
    data_symbols = np.asarray(data_symbols, dtype=np.complex64).reshape(-1)
    if data_symbols.size != expected:
        raise ValueError(f"Expected {expected} data symbols, got {data_symbols.size}.")
    grid = np.zeros((cfg.num_ofdm_symbols, cfg.num_subcarriers), dtype=np.complex64)
    grid[pilot_mask] = cfg.pilot_symbol
    grid[data_mask] = data_symbols
    return grid, pilot_mask, data_mask


def ofdm_modulate(freq_grid: np.ndarray, cp_length: int) -> np.ndarray:
    """Convert a frequency-domain OFDM grid [T, N] into a serialized waveform."""

    freq_grid = np.asarray(freq_grid, dtype=np.complex64)
    if freq_grid.ndim != 2:
        raise ValueError("freq_grid must have shape [num_symbols, num_subcarriers].")
    num_subcarriers = freq_grid.shape[1]
    time_grid = np.fft.ifft(freq_grid, axis=1) * np.sqrt(num_subcarriers)
    cp = time_grid[:, -cp_length:]
    with_cp = np.concatenate([cp, time_grid], axis=1)
    return with_cp.reshape(-1).astype(np.complex64)


def ofdm_demodulate(signal: np.ndarray, cfg: OFDMConfig) -> np.ndarray:
    """Convert a serialized OFDM waveform back to a frequency-domain grid [T, N]."""

    symbol_len = cfg.num_subcarriers + cfg.cp_length
    expected_len = cfg.num_ofdm_symbols * symbol_len
    signal = np.asarray(signal, dtype=np.complex64).reshape(-1)
    if signal.size < expected_len:
        raise ValueError(f"Expected at least {expected_len} samples, got {signal.size}.")
    framed = signal[:expected_len].reshape(cfg.num_ofdm_symbols, symbol_len)
    no_cp = framed[:, cfg.cp_length :]
    freq_grid = np.fft.fft(no_cp, axis=1) / np.sqrt(cfg.num_subcarriers)
    return freq_grid.astype(np.complex64)
