"""Evaluation routines for traditional and CNN-based OFDM receivers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import OFDMConfig, load_config
from .dataset import generate_frame
from .estimators import ls_channel_estimate, mmse_equalize, zf_equalize
from .modulation import demodulate_symbols
from .models import CNNChannelEstimator
from .train import resolve_device
from .utils import complex_mse, two_channel_to_complex


def _count_bit_errors(reference_bits: np.ndarray, estimated_bits: np.ndarray) -> int:
    ref = np.asarray(reference_bits).reshape(-1)
    est = np.asarray(estimated_bits).reshape(-1)
    if ref.size != est.size:
        raise ValueError("Bit arrays must have the same length.")
    return int(np.sum(ref != est))


def _ber_for_equalized_symbols(
    equalized_grid: np.ndarray,
    data_mask: np.ndarray,
    tx_bits: np.ndarray,
    modulation: str,
) -> tuple[int, int, int, int, np.ndarray, np.ndarray]:
    data_symbols = equalized_grid[data_mask]
    estimated_bits = demodulate_symbols(data_symbols, modulation)
    bit_errors = _count_bit_errors(tx_bits, estimated_bits)
    bits_per_symbol = 2 if modulation.lower() == "qpsk" else 4
    tx_groups = tx_bits.reshape(-1, bits_per_symbol)
    est_groups = estimated_bits.reshape(-1, bits_per_symbol)
    symbol_errors = int(np.sum(np.any(tx_groups != est_groups, axis=1)))
    return bit_errors, int(tx_bits.size), symbol_errors, int(tx_groups.shape[0]), data_symbols, estimated_bits


def _squared_evm_error(equalized_symbols: np.ndarray, tx_symbols: np.ndarray) -> tuple[float, float]:
    """Return accumulated squared error and reference symbol energy."""

    err_power = float(np.sum(np.abs(equalized_symbols - tx_symbols) ** 2))
    ref_power = float(np.sum(np.abs(tx_symbols) ** 2))
    return err_power, ref_power


def load_cnn_model(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple[CNNChannelEstimator, torch.device, dict[str, Any]]:
    """Load a trained CNN checkpoint."""

    device_obj = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    model = CNNChannelEstimator().to(device_obj)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device_obj, checkpoint


def predict_cnn_channel(
    model: CNNChannelEstimator,
    device: torch.device,
    cnn_input: np.ndarray,
) -> np.ndarray:
    """Predict a complex channel grid [T, N] from a [3, T, N] input."""

    with torch.no_grad():
        x = torch.from_numpy(cnn_input).unsqueeze(0).float().to(device)
        pred = model(x).squeeze(0).cpu().numpy()
    return two_channel_to_complex(pred)


def evaluate_receiver_methods(
    cfg: OFDMConfig,
    snr_values: list[int] | list[float],
    num_frames: int,
    seed: int,
    cnn_model: CNNChannelEstimator | None = None,
    cnn_device: torch.device | None = None,
    constellation_snr_db: float = 20.0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate LS and optional CNN receiver methods over SNR values."""

    include_cnn = cnn_model is not None
    method_names = ["LS+ZF", "LS+MMSE"]
    if include_cnn:
        method_names.extend(["CNN+ZF", "CNN+MMSE"])

    rows: list[dict[str, float | str]] = []
    constellation: dict[str, np.ndarray] = {}
    for snr_idx, snr_db in enumerate(snr_values):
        stats = {}
        for method in method_names:
            stats[method] = {
                "bit_errors": 0,
                "total_bits": 0,
                "symbol_errors": 0,
                "total_symbols": 0,
                "mse_sum": 0.0,
                "evm_error_sum": 0.0,
                "evm_ref_sum": 0.0,
            }

        for frame_idx in range(num_frames):
            rng = np.random.default_rng(seed + snr_idx * 1_000_000 + frame_idx)
            frame = generate_frame(cfg, snr_db=float(snr_db), rng=rng)
            _, h_ls = ls_channel_estimate(frame.rx_grid, frame.tx_grid, frame.pilot_mask)

            estimates = {"LS": h_ls}
            if include_cnn:
                assert cnn_device is not None
                estimates["CNN"] = predict_cnn_channel(cnn_model, cnn_device, frame.cnn_input)

            equalized = {
                "LS+ZF": zf_equalize(frame.rx_grid, estimates["LS"]),
                "LS+MMSE": mmse_equalize(frame.rx_grid, estimates["LS"], frame.noise_var),
            }
            if include_cnn:
                equalized["CNN+ZF"] = zf_equalize(frame.rx_grid, estimates["CNN"])
                equalized["CNN+MMSE"] = mmse_equalize(frame.rx_grid, estimates["CNN"], frame.noise_var)

            for method, x_hat in equalized.items():
                bit_errors, total_bits, symbol_errors, total_symbols, data_symbols, _ = _ber_for_equalized_symbols(
                    x_hat,
                    frame.data_mask,
                    frame.tx_bits,
                    cfg.modulation,
                )
                evm_error, evm_ref = _squared_evm_error(data_symbols, frame.tx_grid[frame.data_mask])
                stats[method]["bit_errors"] += bit_errors
                stats[method]["total_bits"] += total_bits
                stats[method]["symbol_errors"] += symbol_errors
                stats[method]["total_symbols"] += total_symbols
                stats[method]["evm_error_sum"] += evm_error
                stats[method]["evm_ref_sum"] += evm_ref
                estimator_key = "CNN" if method.startswith("CNN") else "LS"
                stats[method]["mse_sum"] += complex_mse(frame.true_channel, estimates[estimator_key])

                if abs(float(snr_db) - constellation_snr_db) < 1e-9 and frame_idx == 0:
                    constellation[method] = data_symbols.copy()

        for method, values in stats.items():
            evm_rms = np.sqrt(values["evm_error_sum"] / max(values["evm_ref_sum"], 1e-12))
            rows.append(
                {
                    "snr_db": float(snr_db),
                    "method": method,
                    "ber": values["bit_errors"] / max(values["total_bits"], 1),
                    "ser": values["symbol_errors"] / max(values["total_symbols"], 1),
                    "evm_rms": evm_rms,
                    "evm_db": 20.0 * np.log10(max(evm_rms, 1e-12)),
                    "channel_mse": values["mse_sum"] / max(num_frames, 1),
                }
            )

    return pd.DataFrame(rows), constellation


def evaluate_baselines(
    config_path: str | Path = "configs/default_config.json",
    num_frames: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Evaluate LS+ZF and LS+MMSE baselines."""

    cfg = load_config(config_path)
    frames = cfg.num_eval_frames if num_frames is None else int(num_frames)
    eval_seed = cfg.random_seed + 20_000 if seed is None else int(seed)
    df, _ = evaluate_receiver_methods(cfg, cfg.snr_db_values, frames, eval_seed)
    return df
