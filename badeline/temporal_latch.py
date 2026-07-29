"""Explicit onset/release heads and legal key-state decoding.

The frame-state objective and sparse transition objectives deliberately remain
separate.  State logits are trained at natural prevalence so they can be
calibrated as probabilities; onset and release logits may use their own
class-balancing weights without shifting the state probability scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from data.schema import KEY_ORDER


@dataclass(frozen=True)
class TransitionTargets:
    """Dense held-state and sparse event targets for one contiguous sequence."""

    state: torch.Tensor
    onset: torch.Tensor
    release: torch.Tensor
    state_mask: torch.Tensor
    transition_mask: torch.Tensor


def _expanded_mask(mask: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    if mask.ndim == state.ndim - 1:
        mask = mask.unsqueeze(-1)
    try:
        return torch.broadcast_to(mask.bool(), state.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"valid mask shape {tuple(mask.shape)} does not broadcast to "
            f"state shape {tuple(state.shape)}"
        ) from exc


def make_transition_targets(
    state: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    previous_state: torch.Tensor | None = None,
    previous_valid: torch.Tensor | None = None,
) -> TransitionTargets:
    """Construct held, onset, and release targets without crossing gaps.

    Args:
        state: Binary key states with shape ``[..., T, K]``.
        valid_mask: Optional validity mask with shape ``[..., T]`` or
            ``[..., T, K]``.  A transition is valid only when both adjacent
            states are valid.  The first position never has a transition
            target unless ``previous_state`` is supplied.
        previous_state: Optional state immediately preceding the sequence,
            with shape ``[..., K]``. This lets adjacent fixed-length segments
            retain the event target at their first position.
        previous_valid: Optional validity for ``previous_state``, with shape
            ``[...]`` or ``[..., K]``. It defaults to valid when a previous
            state is supplied.
    """

    if state.ndim < 2:
        raise ValueError("state must have shape [...,T,K]")
    if state.shape[-1] != len(KEY_ORDER):
        raise ValueError(
            f"state has {state.shape[-1]} keys, expected {len(KEY_ORDER)}"
        )
    if not state.is_floating_point():
        state = state.float()
    if not torch.all((state == 0) | (state == 1)):
        raise ValueError("state targets must be binary")

    if valid_mask is None:
        state_mask = torch.ones_like(state, dtype=torch.bool)
    else:
        state_mask = _expanded_mask(valid_mask, state)

    onset = torch.zeros_like(state)
    release = torch.zeros_like(state)
    transition_mask = torch.zeros_like(state, dtype=torch.bool)
    if previous_state is not None:
        expected = state.shape[:-2] + (state.shape[-1],)
        if previous_state.shape != expected:
            raise ValueError(
                f"previous_state has shape {tuple(previous_state.shape)}, "
                f"expected {expected}"
            )
        previous = previous_state.to(dtype=state.dtype, device=state.device)
        if not torch.all((previous == 0) | (previous == 1)):
            raise ValueError("previous_state targets must be binary")
        onset[..., 0, :] = (
            (previous < 0.5) & (state[..., 0, :] >= 0.5)
        ).to(state.dtype)
        release[..., 0, :] = (
            (previous >= 0.5) & (state[..., 0, :] < 0.5)
        ).to(state.dtype)
        if previous_valid is None:
            valid_previous = torch.ones_like(previous, dtype=torch.bool)
        else:
            candidate = previous_valid
            if candidate.ndim == previous.ndim - 1:
                candidate = candidate.unsqueeze(-1)
            try:
                valid_previous = torch.broadcast_to(
                    candidate.bool(), previous.shape
                )
            except RuntimeError as exc:
                raise ValueError(
                    "previous_valid does not broadcast to previous_state"
                ) from exc
        transition_mask[..., 0, :] = (
            valid_previous & state_mask[..., 0, :]
        )
    elif previous_valid is not None:
        raise ValueError("previous_valid requires previous_state")
    if state.shape[-2] > 1:
        previous = state[..., :-1, :]
        current = state[..., 1:, :]
        onset[..., 1:, :] = ((previous < 0.5) & (current >= 0.5)).to(state.dtype)
        release[..., 1:, :] = ((previous >= 0.5) & (current < 0.5)).to(state.dtype)
        transition_mask[..., 1:, :] = (
            state_mask[..., :-1, :] & state_mask[..., 1:, :]
        )

    return TransitionTargets(
        state=state,
        onset=onset,
        release=release,
        state_mask=state_mask,
        transition_mask=transition_mask,
    )


class TemporalEventHeads(nn.Module):
    """Predict held state, press onset, and release from aligned features."""

    def __init__(self, input_dim: int, key_count: int = len(KEY_ORDER)) -> None:
        super().__init__()
        if input_dim < 1 or key_count < 1:
            raise ValueError("input_dim and key_count must be positive")
        self.state = nn.Linear(input_dim, key_count)
        self.onset = nn.Linear(input_dim, key_count)
        self.release = nn.Linear(input_dim, key_count)

    def forward(self, encoded: torch.Tensor) -> dict[str, torch.Tensor]:
        if encoded.ndim < 2:
            raise ValueError("encoded features must have shape [...,H]")
        return {
            "state_logits": self.state(encoded),
            "onset_logits": self.onset(encoded),
            "release_logits": self.release(encoded),
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values.masked_select(mask)
    if selected.numel() == 0:
        # Retain a differentiable zero for bounded smoke sequences that happen
        # to contain no valid transition positions.
        return values.sum() * 0.0
    return selected.mean()


class TemporalEventLoss(nn.Module):
    """Natural-prevalence state BCE plus separately balanced event BCE."""

    def __init__(
        self,
        *,
        onset_pos_weight: torch.Tensor | None = None,
        release_pos_weight: torch.Tensor | None = None,
        state_weight: float = 1.0,
        onset_weight: float = 1.0,
        release_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if min(state_weight, onset_weight, release_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        self.state_weight = float(state_weight)
        self.onset_weight = float(onset_weight)
        self.release_weight = float(release_weight)
        self.register_buffer("onset_pos_weight", onset_pos_weight)
        self.register_buffer("release_pos_weight", release_pos_weight)

    def _event_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        pos_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        values = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=pos_weight,
        )
        return _masked_mean(values, mask)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: TransitionTargets,
    ) -> dict[str, torch.Tensor]:
        required = {"state_logits", "onset_logits", "release_logits"}
        missing = required.difference(outputs)
        if missing:
            raise KeyError(f"missing event-head outputs: {sorted(missing)}")
        for name in required:
            if outputs[name].shape != targets.state.shape:
                raise ValueError(
                    f"{name} shape {tuple(outputs[name].shape)} does not "
                    f"match targets {tuple(targets.state.shape)}"
                )

        state_values = F.binary_cross_entropy_with_logits(
            outputs["state_logits"], targets.state, reduction="none"
        )
        state_loss = _masked_mean(state_values, targets.state_mask)
        onset_loss = self._event_loss(
            outputs["onset_logits"],
            targets.onset,
            targets.transition_mask,
            self.onset_pos_weight,
        )
        release_loss = self._event_loss(
            outputs["release_logits"],
            targets.release,
            targets.transition_mask,
            self.release_pos_weight,
        )
        total = (
            self.state_weight * state_loss
            + self.onset_weight * onset_loss
            + self.release_weight * release_loss
        )
        return {
            "loss": total,
            "state_loss": state_loss,
            "onset_loss": onset_loss,
            "release_loss": release_loss,
        }


def _threshold_tensor(
    value: float | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tensor.ndim > 1 or (tensor.ndim == 1 and tensor.shape[0] != reference.shape[-1]):
        raise ValueError("threshold must be scalar or one value per key")
    return tensor


@torch.no_grad()
def decode_event_latch(
    state_logits: torch.Tensor,
    onset_logits: torch.Tensor,
    release_logits: torch.Tensor,
    *,
    state_threshold: float | torch.Tensor = 0.5,
    onset_threshold: float | torch.Tensor = 0.5,
    release_threshold: float | torch.Tensor = 0.5,
    resync_on_threshold: float | torch.Tensor = 0.9,
    resync_off_threshold: float | torch.Tensor = 0.1,
    resync_patience: int = 3,
) -> torch.Tensor:
    """Decode legal held/released sequences from explicit event evidence.

    Events take priority.  At clip start, the direct state head initializes the
    latch.  Thereafter a sustained, high-confidence disagreement from the state
    head can repair a missed event after ``resync_patience`` frames.  There is
    intentionally no minimum duration: one-frame taps remain representable.
    """

    if state_logits.shape != onset_logits.shape or state_logits.shape != release_logits.shape:
        raise ValueError("state, onset, and release logits must have identical shape")
    if state_logits.ndim not in (2, 3):
        raise ValueError("logits must have shape [T,K] or [B,T,K]")
    if state_logits.shape[-1] != len(KEY_ORDER):
        raise ValueError(
            f"logits have {state_logits.shape[-1]} keys, expected {len(KEY_ORDER)}"
        )
    if resync_patience < 1:
        raise ValueError("resync_patience must be at least one")

    squeeze = state_logits.ndim == 2
    if squeeze:
        state_logits = state_logits.unsqueeze(0)
        onset_logits = onset_logits.unsqueeze(0)
        release_logits = release_logits.unsqueeze(0)

    state_prob = state_logits.sigmoid()
    onset_prob = onset_logits.sigmoid()
    release_prob = release_logits.sigmoid()
    state_t = _threshold_tensor(state_threshold, state_prob)
    onset_t = _threshold_tensor(onset_threshold, state_prob)
    release_t = _threshold_tensor(release_threshold, state_prob)
    resync_on_t = _threshold_tensor(resync_on_threshold, state_prob)
    resync_off_t = _threshold_tensor(resync_off_threshold, state_prob)
    if torch.any(resync_off_t >= resync_on_t):
        raise ValueError("resync_off_threshold must be below resync_on_threshold")

    batch, steps, keys = state_prob.shape
    decoded = torch.zeros((batch, steps, keys), dtype=torch.bool, device=state_prob.device)
    if steps == 0:
        return decoded.squeeze(0) if squeeze else decoded

    current = state_prob[:, 0] >= state_t
    decoded[:, 0] = current
    contrary_count = torch.zeros((batch, keys), dtype=torch.int64, device=state_prob.device)

    for step in range(1, steps):
        onset_candidate = (onset_prob[:, step] >= onset_t) & ~current
        release_candidate = (release_prob[:, step] >= release_t) & current
        # The current legal state makes the candidates mutually exclusive:
        # onset can fire only while released and release only while held.
        # Making that invariant explicit avoids an unreachable tie-break rule.
        choose_onset = onset_candidate
        choose_release = release_candidate

        had_event = choose_onset | choose_release
        current = (current | choose_onset) & ~choose_release
        contrary_count = torch.where(
            had_event,
            torch.zeros_like(contrary_count),
            contrary_count,
        )

        direct_on = state_prob[:, step] >= resync_on_t
        direct_off = state_prob[:, step] <= resync_off_t
        contrary = (~current & direct_on) | (current & direct_off)
        contrary_count = torch.where(
            contrary & ~had_event,
            contrary_count + 1,
            torch.zeros_like(contrary_count),
        )
        resync = contrary_count >= resync_patience
        current = torch.where(resync, direct_on, current)
        contrary_count = torch.where(
            resync, torch.zeros_like(contrary_count), contrary_count
        )
        decoded[:, step] = current

    return decoded.squeeze(0) if squeeze else decoded
