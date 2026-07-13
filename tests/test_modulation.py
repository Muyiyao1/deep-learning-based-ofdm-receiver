from __future__ import annotations

import numpy as np

from src.modulation import qam16_demodulate, qam16_modulate, qpsk_demodulate, qpsk_modulate


def test_qpsk_round_trip_noiseless() -> None:
    rng = np.random.default_rng(123)
    bits = rng.integers(0, 2, size=2048, dtype=np.int8)
    symbols = qpsk_modulate(bits)
    recovered = qpsk_demodulate(symbols)
    np.testing.assert_array_equal(recovered, bits)
    np.testing.assert_allclose(np.mean(np.abs(symbols) ** 2), 1.0, atol=1e-6)


def test_16qam_round_trip_noiseless() -> None:
    rng = np.random.default_rng(321)
    bits = rng.integers(0, 2, size=4096, dtype=np.int8)
    symbols = qam16_modulate(bits)
    recovered = qam16_demodulate(symbols)
    np.testing.assert_array_equal(recovered, bits)
    np.testing.assert_allclose(np.mean(np.abs(symbols) ** 2), 1.0, atol=5e-2)


def test_qpsk_gray_adjacent_points_differ_by_one_bit() -> None:
    labels = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int8).reshape(-1)
    symbols = qpsk_modulate(labels).reshape(-1)
    for index in range(4):
        assert np.sum(np.abs(labels.reshape(-1, 2)[index] - labels.reshape(-1, 2)[(index + 1) % 4])) == 1
        assert abs(symbols[index] - symbols[(index + 1) % 4]) > 0
