"""Fair multi-estimator evaluation for the OFDM receiver."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

from .config import OFDMConfig, load_config
from .dataset import Frame, generate_frame
from .estimators import (
    LMMSEPrior,
    PreparedLMMSEFilter,
    dft_denoise_channel,
    frame_average_interpolated_ls,
    interpolate_ls_channel,
    mmse_equalize,
    oracle_lmmse_prior,
    prepare_lmmse_filter,
    sample_covariance_lmmse_prior,
    zf_equalize,
)
from .modulation import demodulate_symbols
from .models import CNNChannelEstimator
from .ofdm import generate_resource_masks
from .train import build_model, resolve_device
from .utils import complex_mse, two_channel_to_complex


def _count_bit_errors(reference_bits: np.ndarray, estimated_bits: np.ndarray) -> int:
    ref = np.asarray(reference_bits).reshape(-1)
    est = np.asarray(estimated_bits).reshape(-1)
    if ref.size != est.size:
        raise ValueError("Bit arrays must have the same length.")
    return int(np.sum(ref != est))


def _receiver_errors(
    equalized_grid: np.ndarray,
    frame: Frame,
    modulation: str,
) -> tuple[int, int, int, int, float, float, np.ndarray]:
    symbols = equalized_grid[frame.data_mask]
    bits = demodulate_symbols(symbols, modulation)
    bit_errors = _count_bit_errors(frame.tx_bits, bits)
    bps = 2 if modulation.lower() == "qpsk" else 4
    symbol_errors = int(np.sum(np.any(frame.tx_bits.reshape(-1, bps) != bits.reshape(-1, bps), axis=1)))
    tx_symbols = frame.tx_grid[frame.data_mask]
    return (
        bit_errors,
        int(frame.tx_bits.size),
        symbol_errors,
        int(tx_symbols.size),
        float(np.sum(np.abs(symbols - tx_symbols) ** 2)),
        float(np.sum(np.abs(tx_symbols) ** 2)),
        symbols,
    )


def _masked_channel_nmse(true_channel: np.ndarray, estimate_grid: np.ndarray, mask: np.ndarray) -> float:
    """Normalized channel error over a frame-shaped resource subset."""

    true_grid = _repeat_frame_channel(true_channel, estimate_grid.shape[0])
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != estimate_grid.shape:
        raise ValueError("mask and estimate_grid must share the frame shape.")
    error = np.sum(np.abs(true_grid[selected] - estimate_grid[selected]) ** 2)
    reference = np.sum(np.abs(true_grid[selected]) ** 2)
    return float(error / max(float(reference), 1e-12))


def _deep_fade_mask(frame: Frame) -> np.ndarray:
    """Select the weakest 20 percent of active tones in one true channel."""

    active = (frame.pilot_mask | frame.data_mask).any(axis=0)
    threshold = np.quantile(np.abs(frame.true_channel[active]), 0.20)
    selected = active & (np.abs(frame.true_channel) <= threshold)
    return np.repeat(selected[np.newaxis, :], frame.pilot_mask.shape[0], axis=0)


def load_cnn_model(
    checkpoint_path: str | Path,
    device: str = "auto",
    expected_cfg: OFDMConfig | None = None,
    allow_config_mismatch: bool = False,
) -> tuple[CNNChannelEstimator, torch.device, dict[str, Any]]:
    """Load a model and reject accidental incompatible checkpoint reuse."""

    device_obj = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    if checkpoint.get("schema_version") != 3:
        raise RuntimeError("Checkpoint uses an obsolete model schema. Retrain with the current code.")
    if expected_cfg is not None and checkpoint.get("config_fingerprint") != expected_cfg.fingerprint() and not allow_config_mismatch:
        raise RuntimeError("Checkpoint/config mismatch. Retrain or pass a documented mismatch evaluation path.")
    spec = checkpoint["model_spec"]
    model = CNNChannelEstimator(
        hidden_channels=int(spec["hidden_channels"]),
        num_blocks=int(spec["num_blocks"]),
        channel_length=int(spec["channel_length"]),
        in_channels=int(spec.get("in_channels", 5)),
        residual=bool(spec.get("residual", True)),
        delay_projection=bool(spec.get("delay_projection", True)),
        delay_regularization=bool(spec.get("delay_regularization", spec.get("delay_projection", True))),
    ).to(device_obj)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device_obj, checkpoint


def predict_cnn_channel(model: CNNChannelEstimator, device: torch.device, cnn_input: np.ndarray) -> np.ndarray:
    """Estimate one static frequency response from a frame-level [5, N] input."""

    with torch.no_grad():
        tensor = torch.from_numpy(cnn_input).unsqueeze(0).float().to(device)
        prediction = model(tensor).squeeze(0).cpu().numpy()
    return two_channel_to_complex(prediction)


def nominal_lmmse_noise_variance(cfg: OFDMConfig, snr_db: float) -> float:
    """Expected post-channel AWGN variance used by a filter cached per SNR.

    AWGN generation retains the project-wide empirical per-frame SNR rule.
    A cacheable LMMSE filter instead uses the corresponding ensemble-average
    received power: unit-energy QAM on ``N_active / N`` occupied tones.
    This avoids silently re-factorizing a matrix for every data realization.
    """

    expected_power = float(cfg.num_active_subcarriers) / float(cfg.num_subcarriers)
    return expected_power / (10.0 ** (float(snr_db) / 10.0))


def build_lmmse_priors(
    cfg: OFDMConfig,
    include_assumed_lmmse: bool = False,
    sample_covariance_sizes: list[int] | None = None,
    sample_prior_cfg: OFDMConfig | None = None,
) -> dict[str, LMMSEPrior]:
    """Build analytic and optional finite-history LMMSE priors offline."""

    priors = {
        "LMMSE-oracle": oracle_lmmse_prior(
            cfg.num_subcarriers,
            cfg.num_channel_taps,
            pdp=cfg.channel_pdp,
            exponential_decay=cfg.exponential_decay,
            name="LMMSE-oracle",
        )
    }
    if include_assumed_lmmse and cfg.lmmse_assumed_num_taps is not None and cfg.lmmse_assumed_pdp is not None:
        priors["LMMSE-train-prior"] = oracle_lmmse_prior(
            cfg.num_subcarriers,
            cfg.lmmse_assumed_num_taps,
            pdp=cfg.lmmse_assumed_pdp,
            exponential_decay=cfg.exponential_decay if cfg.lmmse_assumed_exponential_decay is None else cfg.lmmse_assumed_exponential_decay,
            name="LMMSE-train-prior",
        )
    history_cfg = cfg if sample_prior_cfg is None else sample_prior_cfg
    for index, sample_count in enumerate(sample_covariance_sizes or []):
        name = f"LMMSE-sample-{int(sample_count)}"
        priors[name] = sample_covariance_lmmse_prior(
            cfg.num_subcarriers,
            history_cfg.num_channel_taps,
            history_cfg.channel_pdp,
            history_cfg.exponential_decay,
            int(sample_count),
            seed=history_cfg.sample_covariance_seed + index,
            diagonal_loading=history_cfg.sample_covariance_diagonal_loading,
            minimum_eigenvalue=history_cfg.sample_covariance_min_eigenvalue,
            name=name,
        )
    return priors


def build_lmmse_filter_bank(
    cfg: OFDMConfig,
    priors: dict[str, LMMSEPrior],
    snr_values: list[int] | list[float],
) -> dict[tuple[str, float], PreparedLMMSEFilter]:
    """Build one reusable LMMSE matrix per prior and SNR before frame loops."""

    pilot_mask = generate_resource_masks(cfg)["pilot"]
    observed = pilot_mask.any(axis=0)
    counts = pilot_mask.sum(axis=0)
    pilot_energy = abs(cfg.pilot_symbol) ** 2
    return {
        (name, float(snr_db)): prepare_lmmse_filter(
            prior,
            observed,
            counts,
            nominal_lmmse_noise_variance(cfg, float(snr_db)),
            pilot_energy,
        )
        for name, prior in priors.items()
        for snr_db in snr_values
    }


def _repeat_frame_channel(channel: np.ndarray, num_symbols: int) -> np.ndarray:
    return np.repeat(np.asarray(channel, dtype=np.complex64)[np.newaxis, :], num_symbols, axis=0)


def _estimate_channels(
    frame: Frame,
    cfg: OFDMConfig,
    cnn_model: CNNChannelEstimator | None,
    cnn_device: torch.device | None,
    lmmse_filters: dict[str, PreparedLMMSEFilter],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Compute each estimator once from the same received frame and pilots."""

    estimates: dict[str, np.ndarray] = {}
    timing: dict[str, float] = {}

    start = perf_counter()
    estimates["LS-linear"] = interpolate_ls_channel(frame.sparse_ls, frame.pilot_mask, kind="linear")
    timing["LS-linear"] = perf_counter() - start

    start = perf_counter()
    frame_linear_grid, avg_ls, observed, counts = frame_average_interpolated_ls(frame.sparse_ls, frame.pilot_mask, kind="linear")
    estimates["Frame-LS-linear"] = frame_linear_grid
    timing["Frame-LS-linear"] = perf_counter() - start

    start = perf_counter()
    frame_cubic_grid, _, _, _ = frame_average_interpolated_ls(frame.sparse_ls, frame.pilot_mask, kind="cubic")
    estimates["Frame-LS-cubic"] = frame_cubic_grid
    timing["Frame-LS-cubic"] = perf_counter() - start

    start = perf_counter()
    estimates["DFT-LS"] = _repeat_frame_channel(dft_denoise_channel(frame_linear_grid[0], cfg.effective_dft_taps), cfg.num_ofdm_symbols)
    timing["DFT-LS"] = perf_counter() - start

    for estimator, prepared_filter in lmmse_filters.items():
        start = perf_counter()
        estimate = prepared_filter.estimate(avg_ls)
        estimates[estimator] = _repeat_frame_channel(estimate, cfg.num_ofdm_symbols)
        timing[estimator] = perf_counter() - start

    start = perf_counter()
    estimates["PerfectCSI"] = _repeat_frame_channel(frame.true_channel, cfg.num_ofdm_symbols)
    timing["PerfectCSI"] = perf_counter() - start

    if cnn_model is not None:
        if cnn_device is None:
            raise ValueError("cnn_device is required when cnn_model is supplied.")
        start = perf_counter()
        estimates["ResidualCNN"] = _repeat_frame_channel(predict_cnn_channel(cnn_model, cnn_device, frame.cnn_input), cfg.num_ofdm_symbols)
        timing["ResidualCNN"] = perf_counter() - start
    return estimates, timing


def evaluate_receiver_methods(
    cfg: OFDMConfig,
    snr_values: list[int] | list[float],
    num_frames: int,
    seed: int,
    cnn_model: CNNChannelEstimator | None = None,
    cnn_device: torch.device | None = None,
    constellation_snr_db: float = 20.0,
    include_assumed_lmmse: bool = False,
    lmmse_priors: dict[str, LMMSEPrior] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate every method on exactly the same frame/channel/noise realizations."""

    priors = build_lmmse_priors(cfg, include_assumed_lmmse=include_assumed_lmmse) if lmmse_priors is None else lmmse_priors
    filter_bank = build_lmmse_filter_bank(cfg, priors, snr_values)
    rows: list[dict[str, float | str | int]] = []
    constellation: dict[str, np.ndarray] = {}
    for snr_index, snr_db in enumerate(snr_values):
        stats: dict[str, dict[str, float]] = {}
        for frame_index in range(int(num_frames)):
            rng = np.random.default_rng(int(seed) + snr_index * 1_000_000 + frame_index)
            frame = generate_frame(cfg, snr_db=float(snr_db), rng=rng)
            lmmse_filters = {name: filter_bank[(name, float(snr_db))] for name in priors}
            estimates, timing = _estimate_channels(frame, cfg, cnn_model, cnn_device, lmmse_filters)
            for estimator, h_estimate in estimates.items():
                for equalizer_name, equalizer in (("ZF", zf_equalize), ("MMSE", mmse_equalize)):
                    method = f"{estimator}+{equalizer_name}"
                    if method not in stats:
                        stats[method] = {
                            "bit_errors": 0.0,
                            "total_bits": 0.0,
                            "symbol_errors": 0.0,
                            "total_symbols": 0.0,
                            "evm_error": 0.0,
                            "evm_reference": 0.0,
                            "mse": 0.0,
                            "nmse": 0.0,
                            "pilot_nmse": 0.0,
                            "data_nmse": 0.0,
                            "deep_fade_nmse": 0.0,
                            "inference_time": 0.0,
                        }
                    x_hat = equalizer(frame.rx_grid, h_estimate, frame.noise_var) if equalizer_name == "MMSE" else equalizer(frame.rx_grid, h_estimate)
                    bit_err, bit_total, sym_err, sym_total, evm_err, evm_ref, symbols = _receiver_errors(x_hat, frame, cfg.modulation)
                    values = stats[method]
                    values["bit_errors"] += bit_err
                    values["total_bits"] += bit_total
                    values["symbol_errors"] += sym_err
                    values["total_symbols"] += sym_total
                    values["evm_error"] += evm_err
                    values["evm_reference"] += evm_ref
                    values["mse"] += complex_mse(frame.true_channel, h_estimate[0])
                    values["nmse"] += float(
                        np.sum(np.abs(frame.true_channel - h_estimate[0]) ** 2)
                        / max(np.sum(np.abs(frame.true_channel) ** 2), 1e-12)
                    )
                    values["pilot_nmse"] += _masked_channel_nmse(frame.true_channel, h_estimate, frame.pilot_mask)
                    values["data_nmse"] += _masked_channel_nmse(frame.true_channel, h_estimate, frame.data_mask)
                    values["deep_fade_nmse"] += _masked_channel_nmse(frame.true_channel, h_estimate, _deep_fade_mask(frame))
                    values["inference_time"] += timing[estimator]
                    if abs(float(snr_db) - constellation_snr_db) < 1e-9 and frame_index == 0 and equalizer_name == "ZF":
                        constellation[method] = symbols.copy()
        for method, values in stats.items():
            evm_rms = np.sqrt(values["evm_error"] / max(values["evm_reference"], 1e-12))
            rows.append(
                {
                    "seed": int(seed),
                    "snr_db": float(snr_db),
                    "method": method,
                    "estimator": method.rsplit("+", 1)[0],
                    "equalizer": method.rsplit("+", 1)[1],
                    "frames": int(num_frames),
                    "bit_errors": int(values["bit_errors"]),
                    "total_bits": int(values["total_bits"]),
                    "ber": values["bit_errors"] / max(values["total_bits"], 1.0),
                    "symbol_errors": int(values["symbol_errors"]),
                    "total_symbols": int(values["total_symbols"]),
                    "ser": values["symbol_errors"] / max(values["total_symbols"], 1.0),
                    "evm_rms": evm_rms,
                    "evm_db": 20.0 * np.log10(max(evm_rms, 1e-12)),
                    "channel_mse": values["mse"] / num_frames,
                    "channel_nmse": values["nmse"] / num_frames,
                    "pilot_nmse": values["pilot_nmse"] / num_frames,
                    "data_nmse": values["data_nmse"] / num_frames,
                    "deep_fade_nmse": values["deep_fade_nmse"] / num_frames,
                    "estimator_time_ms": 1e3 * values["inference_time"] / num_frames,
                }
            )
    return pd.DataFrame(rows), constellation


def evaluate_cnn_only_receiver_methods(
    cfg: OFDMConfig,
    snr_values: list[int] | list[float],
    num_frames: int,
    seed: int,
    cnn_model: CNNChannelEstimator,
    cnn_device: torch.device,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate only the neural estimator for repeated model-seed studies.

    Conventional methods are deterministic with respect to a test stream and
    are evaluated once elsewhere.  Avoiding their repeated computation makes
    training-seed uncertainty experiments materially cheaper without changing
    any CNN/test-frame pairing.
    """

    rows: list[dict[str, float | str | int]] = []
    constellation: dict[str, np.ndarray] = {}
    for snr_index, snr_db in enumerate(snr_values):
        values = {
            "bit_errors": 0.0,
            "total_bits": 0.0,
            "symbol_errors": 0.0,
            "total_symbols": 0.0,
            "evm_error": 0.0,
            "evm_reference": 0.0,
            "mse": 0.0,
            "nmse": 0.0,
            "pilot_nmse": 0.0,
            "data_nmse": 0.0,
            "deep_fade_nmse": 0.0,
            "inference_time": 0.0,
        }
        for frame_index in range(int(num_frames)):
            rng = np.random.default_rng(int(seed) + snr_index * 1_000_000 + frame_index)
            frame = generate_frame(cfg, snr_db=float(snr_db), rng=rng)
            start = perf_counter()
            h_estimate = _repeat_frame_channel(predict_cnn_channel(cnn_model, cnn_device, frame.cnn_input), cfg.num_ofdm_symbols)
            values["inference_time"] += perf_counter() - start
            equalized = zf_equalize(frame.rx_grid, h_estimate)
            bit_err, bit_total, sym_err, sym_total, evm_err, evm_ref, symbols = _receiver_errors(equalized, frame, cfg.modulation)
            values["bit_errors"] += bit_err
            values["total_bits"] += bit_total
            values["symbol_errors"] += sym_err
            values["total_symbols"] += sym_total
            values["evm_error"] += evm_err
            values["evm_reference"] += evm_ref
            values["mse"] += complex_mse(frame.true_channel, h_estimate[0])
            values["nmse"] += float(np.sum(np.abs(frame.true_channel - h_estimate[0]) ** 2) / max(np.sum(np.abs(frame.true_channel) ** 2), 1e-12))
            values["pilot_nmse"] += _masked_channel_nmse(frame.true_channel, h_estimate, frame.pilot_mask)
            values["data_nmse"] += _masked_channel_nmse(frame.true_channel, h_estimate, frame.data_mask)
            values["deep_fade_nmse"] += _masked_channel_nmse(frame.true_channel, h_estimate, _deep_fade_mask(frame))
            if abs(float(snr_db) - 20.0) < 1e-9 and frame_index == 0:
                constellation["ResidualCNN+ZF"] = symbols.copy()
        evm_rms = np.sqrt(values["evm_error"] / max(values["evm_reference"], 1e-12))
        rows.append(
            {
                "seed": int(seed),
                "snr_db": float(snr_db),
                "method": "ResidualCNN+ZF",
                "estimator": "ResidualCNN",
                "equalizer": "ZF",
                "frames": int(num_frames),
                "bit_errors": int(values["bit_errors"]),
                "total_bits": int(values["total_bits"]),
                "ber": values["bit_errors"] / max(values["total_bits"], 1.0),
                "symbol_errors": int(values["symbol_errors"]),
                "total_symbols": int(values["total_symbols"]),
                "ser": values["symbol_errors"] / max(values["total_symbols"], 1.0),
                "evm_rms": evm_rms,
                "evm_db": 20.0 * np.log10(max(evm_rms, 1e-12)),
                "channel_mse": values["mse"] / num_frames,
                "channel_nmse": values["nmse"] / num_frames,
                "pilot_nmse": values["pilot_nmse"] / num_frames,
                "data_nmse": values["data_nmse"] / num_frames,
                "deep_fade_nmse": values["deep_fade_nmse"] / num_frames,
                "estimator_time_ms": 1e3 * values["inference_time"] / num_frames,
            }
        )
    return pd.DataFrame(rows), constellation


def evaluate_cnn_only_across_seeds(
    cfg: OFDMConfig,
    num_frames: int,
    seeds: list[int],
    cnn_model: CNNChannelEstimator,
    cnn_device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Aggregate CNN-only results across fixed test Monte Carlo streams."""

    results: list[pd.DataFrame] = []
    constellation: dict[str, np.ndarray] = {}
    for index, seed in enumerate(seeds):
        raw, example = evaluate_cnn_only_receiver_methods(cfg, cfg.snr_db_values, num_frames, int(seed), cnn_model, cnn_device)
        results.append(raw)
        if index == 0:
            constellation = example
    raw = pd.concat(results, ignore_index=True)
    return raw, aggregate_seed_results(raw), constellation


def aggregate_seed_results(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Return mean, standard deviation, and 95% CI across independent seeds."""

    metric_columns = [
        "ber",
        "ser",
        "evm_rms",
        "evm_db",
        "channel_mse",
        "channel_nmse",
        "pilot_nmse",
        "data_nmse",
        "deep_fade_nmse",
        "estimator_time_ms",
    ]
    rows: list[dict[str, float | str | int]] = []
    for (snr_db, method), group in seed_results.groupby(["snr_db", "method"], sort=True):
        row: dict[str, float | str | int] = {
            "snr_db": float(snr_db),
            "method": method,
            "estimator": str(group["estimator"].iloc[0]),
            "equalizer": str(group["equalizer"].iloc[0]),
            "num_seeds": int(group["seed"].nunique()),
            "frames_per_seed": int(group["frames"].iloc[0]),
            "total_bits": int(group["total_bits"].sum()),
            "bit_errors": int(group["bit_errors"].sum()),
            "total_symbols": int(group["total_symbols"].sum()),
            "symbol_errors": int(group["symbol_errors"].sum()),
        }
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            critical_value = float(student_t.ppf(0.975, values.size - 1)) if values.size > 1 else 0.0
            row[f"{metric}_ci95"] = float(critical_value * row[f"{metric}_std"] / np.sqrt(values.size))
        row["ber_pooled"] = int(row["bit_errors"]) / max(int(row["total_bits"]), 1)
        row["ser_pooled"] = int(row["symbol_errors"]) / max(int(row["total_symbols"]), 1)
        row["ber_plot"] = float(row["ber_mean"]) if float(row["ber_mean"]) > 0 else 3.0 / max(int(row["total_bits"]), 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "snr_db"]).reset_index(drop=True)


def evaluate_across_seeds(
    cfg: OFDMConfig,
    num_frames: int,
    seeds: list[int] | None = None,
    cnn_model: CNNChannelEstimator | None = None,
    cnn_device: torch.device | None = None,
    include_assumed_lmmse: bool = False,
    constellation_snr_db: float = 20.0,
    lmmse_priors: dict[str, LMMSEPrior] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate and aggregate independent random test streams."""

    all_results: list[pd.DataFrame] = []
    constellation: dict[str, np.ndarray] = {}
    for index, seed in enumerate(cfg.evaluation_seeds if seeds is None else seeds):
        result, example = evaluate_receiver_methods(
            cfg,
            cfg.snr_db_values,
            num_frames=num_frames,
            seed=int(seed),
            cnn_model=cnn_model,
            cnn_device=cnn_device,
            constellation_snr_db=constellation_snr_db,
            include_assumed_lmmse=include_assumed_lmmse,
            lmmse_priors=lmmse_priors,
        )
        all_results.append(result)
        if index == 0:
            constellation = example
    raw = pd.concat(all_results, ignore_index=True)
    return raw, aggregate_seed_results(raw), constellation


def evaluate_baselines(
    config_path: str | Path = "configs/default_config.json",
    num_frames: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Compatibility helper returning traditional baseline results for one seed."""

    cfg = load_config(config_path)
    actual_seed = cfg.evaluation_seeds[0] if seed is None else seed
    df, _ = evaluate_receiver_methods(cfg, cfg.snr_db_values, cfg.num_eval_frames if num_frames is None else num_frames, actual_seed)
    return df[df["estimator"] != "PerfectCSI"].reset_index(drop=True)
