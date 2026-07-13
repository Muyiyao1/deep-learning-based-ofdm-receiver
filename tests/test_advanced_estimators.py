from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from src.benchmark import benchmark_cached_estimators
from src.config import OFDMConfig, load_config
from src.dataset import OFDMFrameDataset, generate_frame
from src.estimators import (
    lmmse_channel_estimate,
    oracle_lmmse_prior,
    prepare_lmmse_filter,
    sample_covariance_lmmse_prior,
)
from src.models import CNNChannelEstimator
from src.ofdm import pilot_pattern_statistics


def test_cached_lmmse_matches_direct_formulation_and_needs_no_online_inverse(monkeypatch) -> None:
    cfg = OFDMConfig(num_channel_taps=4)
    frame = generate_frame(cfg, 12.0, np.random.default_rng(12345))
    direct = lmmse_channel_estimate(
        frame.frame_ls,
        frame.frame_observed_mask,
        frame.frame_observation_count,
        frame.noise_var,
        abs(cfg.pilot_symbol) ** 2,
        cfg.num_channel_taps,
        cfg.channel_pdp,
        cfg.exponential_decay,
    )
    prior = oracle_lmmse_prior(cfg.num_subcarriers, cfg.num_channel_taps, cfg.channel_pdp, cfg.exponential_decay)
    prepared = prepare_lmmse_filter(
        prior,
        frame.frame_observed_mask,
        frame.frame_observation_count,
        frame.noise_var,
        abs(cfg.pilot_symbol) ** 2,
    )
    np.testing.assert_allclose(prepared.estimate(frame.frame_ls), direct, rtol=1e-5, atol=1e-5)
    monkeypatch.setattr(np.linalg, "inv", lambda *_: (_ for _ in ()).throw(AssertionError("inverse must be offline")))
    assert np.isfinite(prepared.estimate(frame.frame_ls)).all()


def test_sample_covariance_is_hermitian_and_regularized() -> None:
    prior = sample_covariance_lmmse_prior(
        num_subcarriers=32,
        num_taps=6,
        pdp="exponential",
        exponential_decay=0.4,
        sample_count=100,
        seed=99,
        diagonal_loading=0.05,
        minimum_eigenvalue=1e-5,
    )
    np.testing.assert_allclose(prior.covariance, prior.covariance.conj().T, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(prior.covariance)) >= 0.9e-5
    assert np.isfinite(prior.covariance).all()


def test_staggered_pilot_statistics_match_frame_level_observation_model() -> None:
    cfg = OFDMConfig(modulation="16qam", pilot_spacing=8, pilot_pattern="staggered", num_channel_taps=12)
    stats = pilot_pattern_statistics(cfg)
    assert stats["active_subcarriers"] == 55
    assert stats["pilot_observations_per_frame"] > stats["pilot_observations_per_symbol_max"]
    assert stats["pilot_union_coverage"] == 1.0
    assert stats["mean_observations_per_active_subcarrier"] > 1.0


def test_delay_projection_uses_first_configured_taps() -> None:
    model = CNNChannelEstimator(channel_length=4, delay_projection=True, delay_regularization=True)
    delay = torch.zeros(1, 16, dtype=torch.complex64)
    delay[0, :4] = torch.tensor([1 + 0j, 0.5j, -0.2 + 0.1j, 0.1 - 0.3j])
    delay[0, 8] = 1.0 + 2.0j
    frequency = torch.fft.fft(delay, dim=-1)
    two_channel = torch.stack((frequency.real, frequency.imag), dim=1)
    projected = model.delay_project(two_channel)
    projected_delay = torch.fft.ifft(torch.complex(projected[:, 0], projected[:, 1]), dim=-1)
    np.testing.assert_allclose(projected_delay.detach().numpy()[:, 4:], 0.0, atol=1e-6)
    np.testing.assert_allclose(projected_delay.detach().numpy()[:, :4], delay.numpy()[:, :4], atol=1e-6)


def test_training_validation_seeds_and_checkpoint_fingerprints_are_independent() -> None:
    cfg = OFDMConfig()
    train = OFDMFrameDataset(cfg, num_samples=2, fixed_snr_db=10.0, seed=cfg.random_seed + 10_000_000)
    validation = OFDMFrameDataset(cfg, num_samples=2, fixed_snr_db=10.0, seed=cfg.random_seed + 20_000_000)
    assert not np.array_equal(train[0]["tx_bits"].numpy(), validation[0]["tx_bits"].numpy())
    assert replace(cfg, random_seed=1101).fingerprint() != replace(cfg, random_seed=1201).fingerprint()


def test_cached_benchmark_uses_warmup_and_reports_online_metrics() -> None:
    cfg = OFDMConfig(
        num_subcarriers=32,
        cp_length=8,
        num_ofdm_symbols=4,
        guard_subcarriers_each_side=2,
        num_channel_taps=4,
        complexity_benchmark_frames=4,
        complexity_benchmark_repeats=2,
        complexity_warmup_iterations=1,
    )
    model = CNNChannelEstimator(hidden_channels=8, num_blocks=2, channel_length=4)
    prior = oracle_lmmse_prior(cfg.num_subcarriers, cfg.num_channel_taps, cfg.channel_pdp, cfg.exponential_decay)
    benchmark = benchmark_cached_estimators(cfg, model, torch.device("cpu"), {"LMMSE-oracle": prior})
    assert {"ResidualCNN", "LMMSE-oracle"}.issubset(set(benchmark["estimator"]))
    assert (benchmark["warmup_iterations"] == 1).all()
    assert (benchmark["online_mean_ms_per_frame"] >= 0.0).all()


def test_all_extended_configs_load() -> None:
    for path in [
        "configs/multiseed_config.json",
        "configs/domain_randomized_config.json",
        "configs/unseen_generalization_config.json",
    ]:
        load_config(path)
