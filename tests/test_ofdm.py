from __future__ import annotations

import numpy as np

from src.config import OFDMConfig
from src.modulation import demodulate_symbols, modulate_bits
from src.ofdm import build_resource_grid, ofdm_demodulate, ofdm_modulate
from src.utils import bit_error_rate


def test_ofdm_identity_channel_no_noise_has_zero_ber() -> None:
    cfg = OFDMConfig()
    rng = np.random.default_rng(456)
    bits = rng.integers(0, 2, size=cfg.num_bits_per_frame, dtype=np.int8)
    data_symbols = modulate_bits(bits, cfg.modulation)
    tx_grid, _, data_mask = build_resource_grid(data_symbols, cfg)

    waveform = ofdm_modulate(tx_grid, cfg.cp_length)
    rx_grid = ofdm_demodulate(waveform, cfg)
    rx_bits = demodulate_symbols(rx_grid[data_mask], cfg.modulation)

    assert np.allclose(rx_grid, tx_grid, atol=1e-6)
    assert bit_error_rate(bits, rx_bits) == 0.0
