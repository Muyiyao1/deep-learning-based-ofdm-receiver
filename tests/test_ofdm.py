from __future__ import annotations

import numpy as np

from src.channel import add_awgn, apply_multipath_channel, channel_frequency_response, generate_rayleigh_channel
from src.config import OFDMConfig
from src.dataset import generate_frame
from src.evaluate import evaluate_receiver_methods
from src.estimators import (
    dft_denoise_channel,
    frame_average_interpolated_ls,
    lmmse_channel_estimate,
    mmse_equalize,
    sparse_ls_channel_estimate,
    zf_equalize,
)
from src.modulation import demodulate_symbols, modulate_bits
from src.ofdm import build_resource_grid, generate_resource_masks, ofdm_demodulate, ofdm_modulate
from src.utils import bit_error_rate


def test_ofdm_identity_channel_no_noise_has_zero_ber() -> None:
    cfg = OFDMConfig()
    rng = np.random.default_rng(456)
    bits = rng.integers(0, 2, size=cfg.num_bits_per_frame, dtype=np.int8)
    tx_grid, _, data_mask = build_resource_grid(modulate_bits(bits, cfg.modulation), cfg)
    rx_grid = ofdm_demodulate(ofdm_modulate(tx_grid, cfg.cp_length), cfg)
    assert np.allclose(rx_grid, tx_grid, atol=1e-6)
    assert bit_error_rate(bits, demodulate_symbols(rx_grid[data_mask], cfg.modulation)) == 0.0


def test_unitary_fft_and_cp_channel_relation() -> None:
    cfg = OFDMConfig(num_channel_taps=4)
    rng = np.random.default_rng(99)
    bits = rng.integers(0, 2, size=cfg.num_bits_per_frame, dtype=np.int8)
    tx_grid, _, _ = build_resource_grid(modulate_bits(bits, cfg.modulation), cfg)
    waveform = ofdm_modulate(tx_grid, cfg.cp_length)
    assert np.isclose(np.sum(np.abs(tx_grid) ** 2), np.sum(np.abs(waveform.reshape(cfg.num_ofdm_symbols, -1)[:, cfg.cp_length:]) ** 2), rtol=1e-5)
    taps = generate_rayleigh_channel(cfg.num_channel_taps, rng=rng)
    rx_grid = ofdm_demodulate(apply_multipath_channel(waveform, taps), cfg)
    expected = tx_grid * channel_frequency_response(taps, cfg.num_subcarriers)[None, :]
    np.testing.assert_allclose(rx_grid, expected, atol=3e-5, rtol=3e-5)


def test_awgn_empirical_snr_matches_target() -> None:
    rng = np.random.default_rng(123)
    signal = (rng.normal(size=200_000) + 1j * rng.normal(size=200_000)).astype(np.complex64)
    noisy, noise_var = add_awgn(signal, 12.0, rng=rng)
    measured = 10 * np.log10(np.mean(np.abs(signal) ** 2) / np.mean(np.abs(noisy - signal) ** 2))
    assert abs(measured - 12.0) < 0.2
    assert noise_var > 0


def test_masks_are_disjoint_and_cover_resource_grid() -> None:
    cfg = OFDMConfig(modulation="16qam", pilot_pattern="staggered", pilot_spacing=8, num_channel_taps=12)
    masks = generate_resource_masks(cfg)
    total = masks["pilot"].astype(int) + masks["data"].astype(int) + masks["guard"].astype(int) + masks["dc"].astype(int)
    assert np.all(total == 1)
    assert masks["pilot"].sum() > 0 and masks["data"].sum() > 0


def test_ls_dft_lmmse_shapes_and_noiseless_pilots() -> None:
    cfg = OFDMConfig(num_channel_taps=4)
    rng = np.random.default_rng(654)
    frame = generate_frame(cfg, 100.0, rng)
    sparse = sparse_ls_channel_estimate(frame.rx_grid, frame.tx_grid, frame.pilot_mask)
    true_grid = np.repeat(frame.true_channel[None, :], cfg.num_ofdm_symbols, axis=0)
    np.testing.assert_allclose(sparse[frame.pilot_mask], true_grid[frame.pilot_mask], atol=2e-4)
    linear, averaged, observed, counts = frame_average_interpolated_ls(sparse, frame.pilot_mask)
    dft = dft_denoise_channel(linear[0], cfg.effective_dft_taps)
    lmmse = lmmse_channel_estimate(averaged, observed, counts, frame.noise_var, abs(cfg.pilot_symbol) ** 2, cfg.num_channel_taps, cfg.channel_pdp, cfg.exponential_decay)
    assert linear.shape == (cfg.num_ofdm_symbols, cfg.num_subcarriers)
    assert dft.shape == lmmse.shape == (cfg.num_subcarriers,)
    assert np.isfinite(dft).all() and np.isfinite(lmmse).all()


def test_debiased_mmse_matches_zf_for_hard_decision_path() -> None:
    rng = np.random.default_rng(77)
    y = (rng.normal(size=(3, 12)) + 1j * rng.normal(size=(3, 12))).astype(np.complex64)
    h = (rng.normal(size=(3, 12)) + 1j * rng.normal(size=(3, 12))).astype(np.complex64)
    np.testing.assert_allclose(mmse_equalize(y, h, 0.3), zf_equalize(y, h), atol=1e-5)


def test_perfect_csi_is_not_worse_than_sparse_per_symbol_ls() -> None:
    cfg = OFDMConfig(num_subcarriers=32, cp_length=8, num_ofdm_symbols=4, num_channel_taps=4, guard_subcarriers_each_side=2)
    result, _ = evaluate_receiver_methods(cfg, [10.0], num_frames=12, seed=31415)
    perfect = float(result[result["method"] == "PerfectCSI+ZF"]["ber"].iloc[0])
    sparse_ls = float(result[result["method"] == "LS-linear+ZF"]["ber"].iloc[0])
    assert perfect <= sparse_ls
