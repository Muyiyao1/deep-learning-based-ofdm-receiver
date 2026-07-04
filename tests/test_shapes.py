from __future__ import annotations

import numpy as np
import torch

from src.channel import channel_frequency_response, generate_rayleigh_channel
from src.config import OFDMConfig
from src.dataset import OFDMFrameDataset
from src.models import CNNChannelEstimator


def test_dataset_and_cnn_shapes() -> None:
    cfg = OFDMConfig()
    ds = OFDMFrameDataset(cfg, num_samples=4, fixed_snr_db=10.0, seed=789)
    sample = ds[0]

    assert sample["cnn_input"].shape == (3, cfg.num_ofdm_symbols, cfg.num_subcarriers)
    assert sample["channel_target"].shape == (2, cfg.num_ofdm_symbols, cfg.num_subcarriers)
    assert sample["rx_grid"].shape == (2, cfg.num_ofdm_symbols, cfg.num_subcarriers)
    assert sample["tx_grid"].shape == (2, cfg.num_ofdm_symbols, cfg.num_subcarriers)
    assert sample["data_mask"].shape == (cfg.num_ofdm_symbols, cfg.num_subcarriers)
    assert sample["tx_bits"].shape == (cfg.num_bits_per_frame,)

    model = CNNChannelEstimator()
    x = sample["cnn_input"].unsqueeze(0)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 2, cfg.num_ofdm_symbols, cfg.num_subcarriers)


def test_rayleigh_channel_normalization() -> None:
    rng = np.random.default_rng(999)
    cfg = OFDMConfig()
    powers = []
    freq_powers = []
    for _ in range(200):
        taps = generate_rayleigh_channel(cfg.num_channel_taps, rng=rng)
        h = channel_frequency_response(taps, cfg.num_subcarriers)
        powers.append(np.sum(np.abs(taps) ** 2))
        freq_powers.append(np.mean(np.abs(h) ** 2))

    assert np.allclose(np.mean(powers), 1.0, atol=1e-5)
    assert np.allclose(np.mean(freq_powers), 1.0, atol=5e-2)
