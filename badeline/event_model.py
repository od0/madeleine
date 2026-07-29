"""Target-aligned IDM with separate state, onset, and release outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from badeline.model import BadelineIDM


class EventLatchIDM(nn.Module):
    """Compose Badeline's aligned encoder with explicit transition heads.

    The base model owns visual projection and temporal encoding. Its legacy
    state heads remain the exact seven-head parameterization initialized by
    the matched state-only TCN. Only onset and release heads are additional;
    constructing them after the base model preserves a controlled seed-zero
    initialization for the shared encoder and state heads.
    """

    def __init__(self, config: Mapping[str, Any] | object) -> None:
        super().__init__()
        self.encoder = BadelineIDM(config)
        if self.encoder.temporal_arch != "aligned_tcn":
            raise ValueError("EventLatchIDM requires temporal_arch=aligned_tcn")
        head_input_dim = self.encoder.heads[0].in_features
        self.onset_heads = nn.ModuleList(
            nn.Linear(head_input_dim, 1) for _ in self.encoder.key_order
        )
        self.release_heads = nn.ModuleList(
            nn.Linear(head_input_dim, 1) for _ in self.encoder.key_order
        )

    @property
    def key_order(self) -> list[str]:
        return self.encoder.key_order

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return self._outputs(self.encoder.encode_window(batch))

    def forward_segment(
        self, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return self._outputs(self.encoder.encode_segment(batch))

    def _outputs(self, encoded: torch.Tensor) -> dict[str, torch.Tensor]:
        def apply(heads: nn.ModuleList) -> torch.Tensor:
            return torch.cat([head(encoded) for head in heads], dim=-1)

        return {
            "state_logits": apply(self.encoder.heads),
            "onset_logits": apply(self.onset_heads),
            "release_logits": apply(self.release_heads),
        }
