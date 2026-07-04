"""Synthetic OFDM frame generation for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .channel import channel_frequency_response, generate_rayleigh_channel, transmit_through_channel
from .config import OFDMConfig
from .estimators import sparse_ls_channel_estimate
from .modulation import modulate_bits
from .ofdm import build_resource_grid, ofdm_demodulate, ofdm_modulate
from .utils import complex_to_two_channel


@dataclass
class Frame:
    """NumPy representation of one simulated OFDM frame."""

    cnn_input: np.ndarray
    channel_target: np.ndarray
    rx_grid: np.ndarray
    tx_grid: np.ndarray
    true_channel: np.ndarray
    sparse_ls: np.ndarray
    pilot_mask: np.ndarray
    data_mask: np.ndarray
    tx_bits: np.ndarray
    noise_var: float
    snr_db: float


def make_cnn_input(sparse_ls: np.ndarray, pilot_mask: np.ndarray) -> np.ndarray:
    """Create the [3, T, N] CNN input tensor from sparse LS and pilot mask."""

    return np.stack(
        [
            np.real(sparse_ls),
            np.imag(sparse_ls),
            pilot_mask.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def generate_frame(
    cfg: OFDMConfig,
    snr_db: float,
    rng: np.random.Generator,
) -> Frame:
    """Generate one random OFDM frame through a Rayleigh channel and AWGN."""

    tx_bits = rng.integers(0, 2, size=cfg.num_bits_per_frame, dtype=np.int8)
    data_symbols = modulate_bits(tx_bits, cfg.modulation)
    tx_grid, pilot_mask, data_mask = build_resource_grid(data_symbols, cfg)
    tx_waveform = ofdm_modulate(tx_grid, cfg.cp_length)

    taps = generate_rayleigh_channel(cfg.num_channel_taps, rng=rng)
    true_h = channel_frequency_response(taps, cfg.num_subcarriers)
    true_h_grid = np.repeat(true_h[np.newaxis, :], cfg.num_ofdm_symbols, axis=0).astype(np.complex64)

    rx_waveform, noise_var = transmit_through_channel(tx_waveform, taps, snr_db=snr_db, rng=rng)
    rx_grid = ofdm_demodulate(rx_waveform, cfg)
    sparse_ls = sparse_ls_channel_estimate(rx_grid, tx_grid, pilot_mask)

    return Frame(
        cnn_input=make_cnn_input(sparse_ls, pilot_mask),
        channel_target=complex_to_two_channel(true_h_grid),
        rx_grid=rx_grid.astype(np.complex64),
        tx_grid=tx_grid.astype(np.complex64),
        true_channel=true_h_grid,
        sparse_ls=sparse_ls.astype(np.complex64),
        pilot_mask=pilot_mask,
        data_mask=data_mask,
        tx_bits=tx_bits,
        noise_var=float(noise_var),
        snr_db=float(snr_db),
    )


class OFDMFrameDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset that generates synthetic OFDM frames on demand."""

    def __init__(
        self,
        cfg: OFDMConfig,
        num_samples: int,
        snr_range: tuple[float, float] = (0.0, 30.0),
        fixed_snr_db: float | None = None,
        seed: int = 1234,
    ) -> None:
        self.cfg = cfg
        self.num_samples = int(num_samples)
        self.snr_range = snr_range
        self.fixed_snr_db = fixed_snr_db
        self.seed = int(seed)
        self.epoch = 0
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Change the random seed offset used for on-the-fly generation."""

        self.epoch = int(epoch)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + self.epoch * self.num_samples + int(idx))
        if self.fixed_snr_db is None:
            snr_db = float(rng.uniform(self.snr_range[0], self.snr_range[1]))
        else:
            snr_db = float(self.fixed_snr_db)
        frame = generate_frame(self.cfg, snr_db=snr_db, rng=rng)
        return frame_to_torch(frame)


def frame_to_torch(frame: Frame) -> dict[str, torch.Tensor]:
    """Convert a NumPy frame to tensors with stable shapes and dtypes."""

    return {
        "cnn_input": torch.from_numpy(frame.cnn_input).float(),
        "channel_target": torch.from_numpy(frame.channel_target).float(),
        "rx_grid": torch.from_numpy(complex_to_two_channel(frame.rx_grid)).float(),
        "tx_grid": torch.from_numpy(complex_to_two_channel(frame.tx_grid)).float(),
        "data_mask": torch.from_numpy(frame.data_mask.astype(np.float32)).float(),
        "pilot_mask": torch.from_numpy(frame.pilot_mask.astype(np.float32)).float(),
        "tx_bits": torch.from_numpy(frame.tx_bits.astype(np.int64)).long(),
        "noise_var": torch.tensor(frame.noise_var, dtype=torch.float32),
        "snr_db": torch.tensor(frame.snr_db, dtype=torch.float32),
    }


def frame_to_dict(frame: Frame) -> dict[str, Any]:
    """Convert a frame to a plain dictionary for evaluation code."""

    return {
        "cnn_input": frame.cnn_input,
        "channel_target": frame.channel_target,
        "rx_grid": frame.rx_grid,
        "tx_grid": frame.tx_grid,
        "true_channel": frame.true_channel,
        "sparse_ls": frame.sparse_ls,
        "pilot_mask": frame.pilot_mask,
        "data_mask": frame.data_mask,
        "tx_bits": frame.tx_bits,
        "noise_var": frame.noise_var,
        "snr_db": frame.snr_db,
    }
