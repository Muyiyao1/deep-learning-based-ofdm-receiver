"""Digital modulation and hard-demodulation utilities."""

from __future__ import annotations

import numpy as np


def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """Map bits to unit-power Gray-coded QPSK symbols.

    Mapping:
        b0 controls the imaginary sign, b1 controls the real sign.
    """

    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    if bits.size % 2 != 0:
        raise ValueError("QPSK modulation requires an even number of bits.")
    pairs = bits.reshape(-1, 2)
    real = np.where(pairs[:, 1] == 0, 1.0, -1.0)
    imag = np.where(pairs[:, 0] == 0, 1.0, -1.0)
    return ((real + 1j * imag) / np.sqrt(2.0)).astype(np.complex64)


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Hard-demodulate QPSK symbols back to bits."""

    symbols = np.asarray(symbols)
    bits = np.empty((symbols.size, 2), dtype=np.int8)
    bits[:, 0] = (np.imag(symbols.reshape(-1)) < 0).astype(np.int8)
    bits[:, 1] = (np.real(symbols.reshape(-1)) < 0).astype(np.int8)
    return bits.reshape(-1)


def qam16_modulate(bits: np.ndarray) -> np.ndarray:
    """Map bits to normalized Gray-coded 16QAM symbols."""

    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    if bits.size % 4 != 0:
        raise ValueError("16QAM modulation requires a multiple of 4 bits.")
    groups = bits.reshape(-1, 4)

    def axis_level(msb: np.ndarray, lsb: np.ndarray) -> np.ndarray:
        code = (msb << 1) | lsb
        return np.choose(code, [-3.0, -1.0, 3.0, 1.0])

    real = axis_level(groups[:, 0], groups[:, 1])
    imag = axis_level(groups[:, 2], groups[:, 3])
    return ((real + 1j * imag) / np.sqrt(10.0)).astype(np.complex64)


def qam16_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Hard-demodulate normalized Gray-coded 16QAM symbols."""

    scaled = np.asarray(symbols).reshape(-1) * np.sqrt(10.0)
    real = np.real(scaled)
    imag = np.imag(scaled)

    def axis_bits(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        levels = np.empty_like(values, dtype=np.int8)
        levels[values < -2.0] = 0
        levels[(values >= -2.0) & (values < 0.0)] = 1
        levels[(values >= 0.0) & (values < 2.0)] = 3
        levels[values >= 2.0] = 2
        msb = ((levels >> 1) & 1).astype(np.int8)
        lsb = (levels & 1).astype(np.int8)
        return msb, lsb

    r0, r1 = axis_bits(real)
    i0, i1 = axis_bits(imag)
    bits = np.stack([r0, r1, i0, i1], axis=1)
    return bits.reshape(-1)


def modulate_bits(bits: np.ndarray, modulation: str = "qpsk") -> np.ndarray:
    """Dispatch modulation by name."""

    mod = modulation.lower()
    if mod == "qpsk":
        return qpsk_modulate(bits)
    if mod == "16qam":
        return qam16_modulate(bits)
    raise ValueError(f"Unsupported modulation: {modulation}")


def demodulate_symbols(symbols: np.ndarray, modulation: str = "qpsk") -> np.ndarray:
    """Dispatch hard-demodulation by name."""

    mod = modulation.lower()
    if mod == "qpsk":
        return qpsk_demodulate(symbols)
    if mod == "16qam":
        return qam16_demodulate(symbols)
    raise ValueError(f"Unsupported modulation: {modulation}")
