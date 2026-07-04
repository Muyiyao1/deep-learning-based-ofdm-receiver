"""PyTorch models for deep-learning-based OFDM channel estimation."""

from __future__ import annotations

import torch
from torch import nn


class CNNChannelEstimator(nn.Module):
    """Small 2D CNN mapping sparse pilot LS estimates to full channel response."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 48, out_channels: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor with shape [B, 3, T, N].

        Returns:
            Tensor with shape [B, 2, T, N].
        """

        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, T, N], got {tuple(x.shape)}.")
        return self.net(x)
