from __future__ import annotations

import numpy as np
import torch

from src.channel import channel_frequency_response, generate_rayleigh_channel
from src.config import OFDMConfig, load_config
from src.dataset import OFDMFrameDataset, generate_frame
from src.models import CNNChannelEstimator


def test_dataset_and_cnn_shapes_are_finite() -> None:
    cfg = OFDMConfig()
    sample = OFDMFrameDataset(cfg, num_samples=4, fixed_snr_db=10.0, seed=789)[0]
    assert sample["cnn_input"].shape == (5, cfg.num_subcarriers)
    assert sample["channel_target"].shape == (2, cfg.num_subcarriers)
    assert sample["rx_grid"].shape == (2, cfg.num_ofdm_symbols, cfg.num_subcarriers)
    assert sample["data_mask"].shape == (cfg.num_ofdm_symbols, cfg.num_subcarriers)
    assert sample["tx_bits"].shape == (cfg.num_bits_per_frame,)
    model = CNNChannelEstimator(channel_length=cfg.effective_dft_taps)
    with torch.no_grad():
        output = model(sample["cnn_input"].unsqueeze(0))
    assert output.shape == (1, 2, cfg.num_subcarriers)
    assert torch.isfinite(output).all()


def test_rayleigh_channel_normalization() -> None:
    rng = np.random.default_rng(999)
    cfg = OFDMConfig()
    powers, freq_powers = [], []
    for _ in range(100):
        taps = generate_rayleigh_channel(cfg.num_channel_taps, rng=rng)
        powers.append(np.sum(np.abs(taps) ** 2))
        freq_powers.append(np.mean(np.abs(channel_frequency_response(taps, cfg.num_subcarriers)) ** 2))
    assert np.allclose(np.mean(powers), 1.0, atol=1e-5)
    assert np.allclose(np.mean(freq_powers), 1.0, atol=5e-2)


def test_fixed_seed_reproduces_frame_and_configs_load() -> None:
    cfg = OFDMConfig()
    first = generate_frame(cfg, 10.0, np.random.default_rng(2024))
    second = generate_frame(cfg, 10.0, np.random.default_rng(2024))
    np.testing.assert_array_equal(first.tx_bits, second.tx_bits)
    np.testing.assert_allclose(first.rx_grid, second.rx_grid)
    for path in ["configs/default_config.json", "configs/stress_test_config.json", "configs/mismatch_config.json", "configs/quick_experiment.json", "configs/final_experiment.json"]:
        load_config(path)
