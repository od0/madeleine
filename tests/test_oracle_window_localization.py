from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from data.schema import KEY_ORDER
from experiments.oracle_window_localization import (
    FeatureSession,
    OracleWindowDataset,
    OracleWindowEventModel,
    construct_oracle_examples,
    epoch_order,
    load_feature_session,
    requested_logits,
    state_dict_sha256,
    transition_matrix,
)


def _session(
    keys: np.ndarray,
    *,
    engine: np.ndarray | None = None,
    active: np.ndarray | None = None,
    session_id: str = "fixture",
) -> FeatureSession:
    length = len(keys)
    return FeatureSession(
        session_id=session_id,
        features=np.zeros((length, 512), dtype=np.float16),
        keys=keys.astype(np.uint8),
        engine_frame_idx=(
            np.arange(length, dtype=np.int64) if engine is None else engine
        ),
        input_active=(
            np.ones(length, dtype=np.uint8) if active is None else active
        ),
    )


def _construct(session: FeatureSession):
    return construct_oracle_examples(
        session,
        split="train",
        width=16,
        halo=8,
        assignment_seed=20260728,
        block_frames=600,
    )


def test_transition_targets_keep_polarity_and_require_valid_predecessor() -> None:
    keys = np.zeros((8, len(KEY_ORDER)), dtype=np.uint8)
    keys[2:5, 0] = 1
    keys[6:, 0] = 1
    engine = np.arange(8, dtype=np.int64)
    engine[6:] += 10

    events, counts = transition_matrix(
        keys, engine, np.ones(8, dtype=np.uint8)
    )

    assert events[2, 0]
    assert events[5, len(KEY_ORDER)]
    assert not events[6, 0]  # the apparent onset crosses an engine gap
    assert counts == {
        "raw_key_transitions": 3,
        "valid_active_contiguous_transitions": 2,
        "invalid_predecessor_transitions": 1,
    }


def test_all_offsets_are_feasible_and_balanced_before_assignment() -> None:
    transition_frames = [40 + 60 * index for index in range(32)]
    length = transition_frames[-1] + 50
    keys = np.zeros((length, len(KEY_ORDER)), dtype=np.uint8)
    state = 0
    for frame in transition_frames:
        state = 1 - state
        keys[frame:, 0] = state

    result = _construct(_session(keys))
    left_rows = [row for row in result.examples if row.key_index == 0]
    assert len(left_rows) == 32
    for polarity in (0, 1):
        offsets = np.bincount(
            [row.offset for row in left_rows if row.event_type_index == polarity],
            minlength=16,
        )
        assert offsets.tolist() == [1] * 16
    for row in left_rows:
        assert row.crop_start >= 0
        assert row.candidate_start == row.crop_start + 8
        assert row.array_index == row.candidate_start + row.offset


def test_construction_excludes_ambiguous_boundary_gap_and_inactive_events() -> None:
    length = 220
    keys = np.zeros((length, len(KEY_ORDER)), dtype=np.uint8)
    # Two left onsets only ten frames apart; neither is safe for every offset.
    keys[30:35, 0] = 1
    keys[40:80, 0] = 1
    # A clean right onset whose 47-frame union contains an inactive row.
    keys[110:, 1] = 1
    # A clean up onset whose union is split by an engine discontinuity.
    keys[180:, 2] = 1
    active = np.ones(length, dtype=np.uint8)
    active[125] = 0
    engine = np.arange(length, dtype=np.int64)
    engine[190:] += 5

    result = _construct(_session(keys, engine=engine, active=active))

    retained = {(row.array_index, row.head_index) for row in result.examples}
    assert (30, 0) not in retained
    assert (40, 0) not in retained
    assert (110, 1) not in retained
    assert (180, 2) not in retained
    assert result.counts["excluded_ambiguous_same_head"] >= 2
    assert result.counts["excluded_boundary_gap_or_inactive_union"] >= 2


def test_dataset_exposes_only_declared_model_inputs_and_targets() -> None:
    keys = np.zeros((200, len(KEY_ORDER)), dtype=np.uint8)
    keys[60:130, 0] = 1
    result = _construct(_session(keys))
    dataset = OracleWindowDataset(
        {"fixture": _session(keys)}, result.examples, width=16, halo=8
    )

    row = dataset[0]
    assert set(row) == {
        "features",
        "requested_head",
        "target_offset",
        "task_weight",
    }
    assert row["features"].shape == (32, 512)
    assert row["features"].dtype == torch.float32


def test_model_candidate_logits_have_no_padding_position_cue() -> None:
    torch.manual_seed(7)
    model = OracleWindowEventModel(
        feature_dim=512,
        projection_dim=32,
        temporal_dim=32,
        dilations=[1, 2, 3],
        width=16,
        halo=8,
    ).eval()
    features = torch.ones((2, 32, 512), dtype=torch.float32)

    logits = model(features)

    assert model.temporal.receptive_radius == 8
    assert logits.shape == (2, 16, 14)
    assert torch.allclose(logits, logits[:, :1].expand_as(logits), atol=1e-6)


def test_requested_head_selection_preserves_candidate_axis() -> None:
    dense = torch.arange(3 * 16 * 14).reshape(3, 16, 14)
    heads = torch.tensor([0, 7, 13])
    selected = requested_logits(dense, heads)
    assert selected.shape == (3, 16)
    assert torch.equal(selected[0], dense[0, :, 0])
    assert torch.equal(selected[1], dense[1, :, 7])
    assert torch.equal(selected[2], dense[2, :, 13])


def test_matched_arms_share_initial_bytes_and_batch_order() -> None:
    torch.manual_seed(19)
    initial = OracleWindowEventModel(
        feature_dim=8,
        projection_dim=12,
        temporal_dim=16,
        dilations=[1, 2, 3],
        width=16,
        halo=8,
    )
    left = copy.deepcopy(initial)
    right = copy.deepcopy(initial)
    assert state_dict_sha256(left) == state_dict_sha256(right)
    assert np.array_equal(epoch_order(101, 0, 3), epoch_order(101, 0, 3))
    assert not np.array_equal(epoch_order(101, 0, 3), epoch_order(101, 0, 4))


@pytest.mark.parametrize(
    "session_id",
    [
        "rec_20260725_025853",
        "rec_20260725_160450_b1",
        "rec_20260727_220000_test",
        "some-untouched-copy",
    ],
)
def test_embargoed_session_is_rejected_before_path_access(
    tmp_path: Path, session_id: str
) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        load_feature_session(tmp_path / "does-not-exist", session_id)
