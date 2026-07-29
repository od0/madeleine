from __future__ import annotations

import torch

from badeline.temporal_latch import (
    TemporalEventHeads,
    TemporalEventLoss,
    decode_event_latch,
    make_transition_targets,
)


def _seven(states: list[list[int]]) -> torch.Tensor:
    tensor = torch.zeros((1, len(states), 7), dtype=torch.float32)
    for step, active in enumerate(states):
        tensor[0, step, active] = 1
    return tensor


def test_transition_targets_do_not_cross_invalid_boundary() -> None:
    state = _seven([[], [0], [0], [], [1]])
    valid = torch.tensor([[True, True, False, True, True]])
    targets = make_transition_targets(state, valid_mask=valid)

    assert targets.onset[0, 1, 0] == 1
    assert targets.release[0, 3, 0] == 1
    assert not targets.transition_mask[0, 2].any()
    assert not targets.transition_mask[0, 3].any()
    assert targets.transition_mask[0, 4].all()
    assert targets.onset[0, 4, 1] == 1


def test_previous_state_retains_first_segment_event() -> None:
    state = _seven([[0], [0], []])
    previous = torch.zeros((1, 7), dtype=torch.float32)
    targets = make_transition_targets(
        state,
        previous_state=previous,
        previous_valid=torch.tensor([True]),
    )
    assert targets.transition_mask[0, 0].all()
    assert targets.onset[0, 0, 0] == 1


def test_event_heads_and_loss_are_finite_and_differentiable() -> None:
    encoded = torch.randn(2, 6, 12, requires_grad=True)
    state = torch.randint(0, 2, (2, 6, 7), dtype=torch.float32)
    targets = make_transition_targets(state)
    heads = TemporalEventHeads(12)
    outputs = heads(encoded)
    criterion = TemporalEventLoss(
        onset_pos_weight=torch.full((7,), 4.0),
        release_pos_weight=torch.full((7,), 3.0),
    )

    losses = criterion(outputs, targets)
    assert set(losses) == {"loss", "state_loss", "onset_loss", "release_loss"}
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    assert encoded.grad is not None
    assert torch.isfinite(encoded.grad).all()


def test_latch_persists_without_redeciding_every_frame() -> None:
    state = torch.full((6, 7), -4.0)
    onset = torch.full((6, 7), -4.0)
    release = torch.full((6, 7), -4.0)
    onset[1, 0] = 5.0
    release[5, 0] = 5.0

    decoded = decode_event_latch(
        state,
        onset,
        release,
        resync_patience=10,
    )
    assert decoded[:, 0].tolist() == [False, True, True, True, True, False]


def test_latch_preserves_one_frame_tap() -> None:
    state = torch.full((4, 7), -4.0)
    onset = torch.full((4, 7), -4.0)
    release = torch.full((4, 7), -4.0)
    onset[1, 5] = 5.0
    release[2, 5] = 5.0

    decoded = decode_event_latch(state, onset, release)
    assert decoded[:, 5].tolist() == [False, True, False, False]


def test_latch_uses_legal_state_when_both_event_heads_are_high() -> None:
    state = torch.full((3, 7), -10.0)
    onset = torch.full((3, 7), -10.0)
    release = torch.full((3, 7), -10.0)
    onset[1:, 0] = 10.0
    release[1:, 0] = 10.0

    decoded = decode_event_latch(state, onset, release)

    # At frame 1 only onset is legal. At frame 2 the key is held, so only
    # release is legal, even though the raw head scores are tied both times.
    assert decoded[:, 0].tolist() == [False, True, False]


def test_state_head_repairs_missed_event_after_patience() -> None:
    state = torch.full((6, 7), -4.0)
    onset = torch.full((6, 7), -4.0)
    release = torch.full((6, 7), -4.0)
    state[1:, 2] = 5.0

    decoded = decode_event_latch(
        state,
        onset,
        release,
        resync_patience=2,
    )
    assert decoded[:, 2].tolist() == [False, False, True, True, True, True]
