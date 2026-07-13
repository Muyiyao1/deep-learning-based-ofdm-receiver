"""Physics-informed residual neural channel estimator."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock1D(nn.Module):
    """Compact circular 1-D residual block over the OFDM frequency axis."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, padding_mode="circular")
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, padding_mode="circular")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.conv1(x))
        return F.gelu(self.conv2(x) + residual)


class ResidualCNNChannelEstimator(nn.Module):
    """Predict a correction to frame-averaged LS and project to delay support.

    Input channels are real/imag initial linear estimate, real/imag averaged
    sparse LS, and the union-pilot mask.  Since the channel is block-static,
    time samples are averaged before entering this 1-D model.
    """

    model_name = "residual_1d_cnn_delay_projected_v1"

    def __init__(
        self,
        hidden_channels: int = 32,
        num_blocks: int = 4,
        channel_length: int = 8,
        in_channels: int = 5,
        residual: bool = True,
        delay_projection: bool = True,
        delay_regularization: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.num_blocks = int(num_blocks)
        self.channel_length = int(channel_length)
        self.in_channels = int(in_channels)
        self.residual = bool(residual)
        self.delay_projection = bool(delay_projection)
        self.delay_regularization = bool(delay_regularization)
        self.stem = nn.Conv1d(in_channels, hidden_channels, 5, padding=2, padding_mode="circular")
        dilations = [2**index for index in range(num_blocks)]
        self.blocks = nn.Sequential(*(ResidualBlock1D(hidden_channels, dilation) for dilation in dilations))
        self.head = nn.Conv1d(hidden_channels, 2, 3, padding=1, padding_mode="circular")

    def model_spec(self) -> dict[str, int | str]:
        return {
            "model_name": self.model_name,
            "hidden_channels": self.hidden_channels,
            "num_blocks": self.num_blocks,
            "channel_length": self.channel_length,
            "in_channels": self.in_channels,
            "residual": self.residual,
            "delay_projection": self.delay_projection,
            "delay_regularization": self.delay_regularization,
        }

    def delay_project(self, channel: torch.Tensor) -> torch.Tensor:
        """Set delay taps outside configured support to zero in a differentiable way."""

        complex_channel = torch.complex(channel[:, 0], channel[:, 1])
        delay = torch.fft.ifft(complex_channel, dim=-1)
        mask = torch.zeros_like(delay)
        mask[:, : self.channel_length] = 1.0
        projected = torch.fft.fft(delay * mask, dim=-1)
        return torch.stack((projected.real, projected.imag), dim=1)

    def forward(self, x: torch.Tensor, return_raw: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return projected estimate, optionally together with pre-projection output."""

        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected input [B, {self.in_channels}, N], got {tuple(x.shape)}.")
        initial = x[:, :2]
        features = F.gelu(self.stem(x))
        correction = self.head(self.blocks(features))
        raw = initial + correction if self.residual else correction
        projected = self.delay_project(raw) if self.delay_projection else raw
        return (projected, raw) if return_raw else projected


def delay_domain_tail_energy(channel: torch.Tensor, channel_length: int) -> torch.Tensor:
    """Mean energy outside the valid delay support for a two-channel response."""

    complex_channel = torch.complex(channel[:, 0], channel[:, 1])
    delay = torch.fft.ifft(complex_channel, dim=-1)
    return torch.mean(torch.abs(delay[:, channel_length:]) ** 2)


def pilot_consistency_loss(prediction: torch.Tensor, cnn_input: torch.Tensor) -> torch.Tensor:
    """Penalize disagreement with averaged LS only on observed pilot tones."""

    mask = cnn_input[:, 4:5]
    observed = cnn_input[:, 2:4]
    squared = (prediction - observed) ** 2
    return torch.sum(squared * mask) / torch.clamp(torch.sum(mask) * 2.0, min=1.0)


# Backward-compatible public class name used by earlier scripts/tests.
CNNChannelEstimator = ResidualCNNChannelEstimator
