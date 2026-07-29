"""Target-aligned temporal encoders for inverse dynamics.

The legacy IDM reduces a centered window to the final state of a one-way GRU.
That makes the target frame (near the middle of the window) travel through all
remaining future updates before classification.  The encoder here instead
keeps one representation per native-rate frame.  Callers select only target
positions whose complete receptive field is present, so padding at an input
block boundary can never affect a reported or trained prediction.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


DEFAULT_TCN_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 32, 16, 8, 4)


class ResidualTemporalBlock(nn.Module):
    """One acausal dilated temporal update with a residual connection."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.0) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if dilation < 1:
            raise ValueError("dilation must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dilation = int(dilation)
        self.norm = nn.LayerNorm(channels)
        self.temporal = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=self.dilation,
            padding=self.dilation,
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.mix = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [B,T,C]")
        update = self.norm(sequence).transpose(1, 2)
        update = self.temporal(update)
        update = self.activation(update)
        update = self.dropout(update)
        update = self.mix(update).transpose(1, 2)
        return sequence + update


class AlignedTemporalTCN(nn.Module):
    """Dense, acausal sequence encoder with an explicit receptive radius.

    A width-five local stem mirrors the temporal operation VPT found important.
    Dilated residual blocks then expand context without downsampling, retaining
    an output aligned to every input frame.  With the default dilation pyramid,
    the receptive radius is exactly 189 native-rate frames: the past support of
    the existing 128-sample, stride-three centered recipe.
    """

    def __init__(
        self,
        input_dim: int,
        channels: int,
        *,
        dilations: Sequence[int] = DEFAULT_TCN_DILATIONS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or channels < 1:
            raise ValueError("input_dim and channels must be positive")
        parsed_dilations = tuple(int(value) for value in dilations)
        if not parsed_dilations or any(value < 1 for value in parsed_dilations):
            raise ValueError("dilations must be a non-empty positive sequence")

        self.input_dim = int(input_dim)
        self.channels = int(channels)
        self.dilations = parsed_dilations
        self.local_kernel_size = 5
        self.local = nn.Conv1d(
            self.input_dim,
            self.channels,
            kernel_size=self.local_kernel_size,
            padding=self.local_kernel_size // 2,
        )
        self.local_activation = nn.SiLU()
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(self.channels, dilation, dropout)
            for dilation in self.dilations
        )
        self.final_norm = nn.LayerNorm(self.channels)

    @property
    def receptive_radius(self) -> int:
        """Native-rate frames required on each side of an output position."""

        return self.local_kernel_size // 2 + sum(self.dilations)

    def valid_slice(self, length: int) -> slice:
        """Return positions unaffected by boundary padding for ``length``."""

        radius = self.receptive_radius
        if length < 2 * radius + 1:
            return slice(0, 0)
        return slice(radius, length - radius)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [B,T,C]")
        if sequence.shape[-1] != self.input_dim:
            raise ValueError(
                f"sequence feature dim is {sequence.shape[-1]}, expected "
                f"{self.input_dim}"
            )
        encoded = self.local(sequence.transpose(1, 2)).transpose(1, 2)
        encoded = self.local_activation(encoded)
        for block in self.blocks:
            encoded = block(encoded)
        return self.final_norm(encoded)
