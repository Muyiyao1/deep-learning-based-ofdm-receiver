"""Independent synthetic OFDM frames for training, validation, and testing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .channel import channel_frequency_response, generate_rayleigh_channel, transmit_through_channel
from .config import OFDMConfig
from .estimators import frame_average_interpolated_ls, sparse_ls_channel_estimate
from .modulation import modulate_bits
from .ofdm import build_resource_grid, generate_resource_masks, ofdm_demodulate, ofdm_modulate
from .utils import complex_to_two_channel


@dataclass
class Frame:
    """One complete OFDM realization shared by every receiver method."""

    cnn_input: np.ndarray
    channel_target: np.ndarray
    rx_grid: np.ndarray
    tx_grid: np.ndarray
    true_channel: np.ndarray
    sparse_ls: np.ndarray
    frame_ls: np.ndarray
    frame_observed_mask: np.ndarray
    frame_observation_count: np.ndarray
    pilot_mask: np.ndarray
    data_mask: np.ndarray
    guard_mask: np.ndarray
    dc_mask: np.ndarray
    tx_bits: np.ndarray
    noise_var: float
    snr_db: float


def make_cnn_input(
    initial_channel: np.ndarray,
    averaged_sparse_ls: np.ndarray,
    observed_mask: np.ndarray,
) -> np.ndarray:
    """Build the residual estimator input ``[5, N]`` from frame-level pilots."""

    return np.stack(
        [
            np.real(initial_channel),
            np.imag(initial_channel),
            np.real(averaged_sparse_ls),
            np.imag(averaged_sparse_ls),
            np.asarray(observed_mask, dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def generate_frame(cfg: OFDMConfig, snr_db: float, rng: np.random.Generator) -> Frame:
    """Generate one block-fading OFDM frame using the project-wide SNR path."""

    masks = generate_resource_masks(cfg)
    tx_bits = rng.integers(0, 2, size=int(masks["data"].sum()) * cfg.bits_per_symbol, dtype=np.int8)
    data_symbols = modulate_bits(tx_bits, cfg.modulation)
    tx_grid, pilot_mask, data_mask = build_resource_grid(data_symbols, cfg)
    tx_waveform = ofdm_modulate(tx_grid, cfg.cp_length)

    taps = generate_rayleigh_channel(
        cfg.num_channel_taps,
        rng=rng,
        pdp=cfg.channel_pdp,
        exponential_decay=cfg.exponential_decay,
    )
    true_h = channel_frequency_response(taps, cfg.num_subcarriers)
    rx_waveform, noise_var = transmit_through_channel(tx_waveform, taps, snr_db=snr_db, rng=rng)
    rx_grid = ofdm_demodulate(rx_waveform, cfg)
    sparse_ls = sparse_ls_channel_estimate(rx_grid, tx_grid, pilot_mask)
    initial_grid, frame_ls, observed, counts = frame_average_interpolated_ls(sparse_ls, pilot_mask, kind="linear")
    initial_channel = initial_grid[0]

    return Frame(
        cnn_input=make_cnn_input(initial_channel, frame_ls, observed),
        channel_target=complex_to_two_channel(true_h),
        rx_grid=rx_grid.astype(np.complex64),
        tx_grid=tx_grid.astype(np.complex64),
        true_channel=true_h.astype(np.complex64),
        sparse_ls=sparse_ls.astype(np.complex64),
        frame_ls=frame_ls.astype(np.complex64),
        frame_observed_mask=observed,
        frame_observation_count=counts,
        pilot_mask=pilot_mask,
        data_mask=data_mask,
        guard_mask=masks["guard"],
        dc_mask=masks["dc"],
        tx_bits=tx_bits,
        noise_var=float(noise_var),
        snr_db=float(snr_db),
    )


class OFDMFrameDataset(Dataset[dict[str, torch.Tensor]]):
    """On-the-fly deterministic dataset with independent split seed streams."""

    def __init__(
        self,
        cfg: OFDMConfig,
        num_samples: int,
        snr_range: tuple[float, float] | None = None,
        fixed_snr_db: float | None = None,
        seed: int = 1234,
    ) -> None:
        self.cfg = cfg
        self.num_samples = int(num_samples)
        self.snr_range = tuple(cfg.train_snr_db_range) if snr_range is None else snr_range
        self.fixed_snr_db = fixed_snr_db
        self.seed = int(seed)
        self.epoch = 0
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sample_channel_config(self, rng: np.random.Generator) -> OFDMConfig:
        """Optionally randomize channel statistics while keeping the OFDM grid fixed."""

        if not self.cfg.domain_randomized_training:
            return self.cfg
        tap_min, tap_max = self.cfg.domain_tap_range
        taps = int(rng.integers(tap_min, tap_max + 1))
        pdp = str(rng.choice(self.cfg.domain_pdp_choices))
        decay = (
            float(rng.uniform(*self.cfg.domain_exponential_decay_range))
            if pdp == "exponential"
            else self.cfg.exponential_decay
        )
        return replace(
            self.cfg,
            num_channel_taps=taps,
            channel_pdp=pdp,
            exponential_decay=decay,
            dft_truncation_taps=max(self.cfg.effective_dft_taps, taps),
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + self.epoch * self.num_samples + int(idx))
        snr_db = float(rng.uniform(*self.snr_range)) if self.fixed_snr_db is None else float(self.fixed_snr_db)
        return frame_to_torch(generate_frame(self._sample_channel_config(rng), snr_db=snr_db, rng=rng))


def frame_to_torch(frame: Frame) -> dict[str, torch.Tensor]:
    """Convert one frame to fixed-shape tensors used by training and tests."""

    return {
        "cnn_input": torch.from_numpy(frame.cnn_input).float(),
        "channel_target": torch.from_numpy(frame.channel_target).float(),
        "rx_grid": torch.from_numpy(complex_to_two_channel(frame.rx_grid)).float(),
        "tx_grid": torch.from_numpy(complex_to_two_channel(frame.tx_grid)).float(),
        "data_mask": torch.from_numpy(frame.data_mask.astype(np.float32)).float(),
        "pilot_mask": torch.from_numpy(frame.pilot_mask.astype(np.float32)).float(),
        "frame_observed_mask": torch.from_numpy(frame.frame_observed_mask.astype(np.float32)).float(),
        "frame_ls": torch.from_numpy(complex_to_two_channel(frame.frame_ls)).float(),
        "tx_bits": torch.from_numpy(frame.tx_bits.astype(np.int64)).long(),
        "noise_var": torch.tensor(frame.noise_var, dtype=torch.float32),
        "snr_db": torch.tensor(frame.snr_db, dtype=torch.float32),
    }


def frame_to_dict(frame: Frame) -> dict[str, Any]:
    """Convert a frame to a lightweight mapping for evaluation utilities."""

    return frame.__dict__.copy()
