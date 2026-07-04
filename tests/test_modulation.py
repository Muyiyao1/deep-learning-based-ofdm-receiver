from __future__ import annotations

import numpy as np

from src.modulation import qam16_demodulate, qam16_modulate, qpsk_demodulate, qpsk_modulate


def test_qpsk_round_trip_noiseless() -> None:
    rng = np.random.default_rng(123)
    bits = rng.integers(0, 2, size=2048, dtype=np.int8)
    symbols = qpsk_modulate(bits)
    recovered = qpsk_demodulate(symbols)
    np.testing.assert_array_equal(recovered, bits)


def test_16qam_round_trip_noiseless() -> None:
    rng = np.random.default_rng(321)
    bits = rng.integers(0, 2, size=4096, dtype=np.int8)
    symbols = qam16_modulate(bits)
    recovered = qam16_demodulate(symbols)
    np.testing.assert_array_equal(recovered, bits)
