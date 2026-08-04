from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from badeline.metrics import summarize
from experiments.eval_vpt_small import (
    center_supported_records,
    combine_phases,
    flatten_center_probabilities,
    retained_positions,
    resolve_center_overlap,
    restrict_to_reference_support,
    validate_temporal_support,
)


def _part(rows: list[int], phase: int) -> dict[str, np.ndarray]:
    count = len(rows)
    truth = np.zeros((count, 7), dtype=np.uint8)
    truth[:, 4] = np.asarray(rows) % 2
    return {
        "source_row": np.asarray(rows, dtype=np.int64),
        "engine_idx": np.asarray(rows, dtype=np.int64),
        "continuity": np.zeros(count, dtype=np.int32),
        "truth": truth,
        "active": np.ones(count, dtype=np.uint8),
        "probability": np.full((count, 7), 0.1 + phase * 0.1, dtype=np.float32),
    }


def test_three_phase_interleave_restores_native_order_and_splits_gaps() -> None:
    manifest = {
        "records": [
            {"session_id": "s", "phase": 0},
            {"session_id": "s", "phase": 1},
            {"session_id": "s", "phase": 2},
        ]
    }
    inferred = [
        _part([0, 3, 6, 12], 0),
        _part([1, 4, 7, 13], 1),
        _part([2, 5, 8, 14], 2),
    ]
    combined = combine_phases(manifest, None, inferred)  # type: ignore[arg-type]
    assert np.array_equal(combined["source_row"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14])
    assert np.array_equal(combined["stream_lengths"], [9, 3])
    assert combined["truth"].shape == combined["probability"].shape == (12, 7)


def test_phase0_deployment_support_uses_three_row_source_step() -> None:
    manifest = {"records": [{"session_id": "s", "phase": 0}]}
    combined = combine_phases(
        manifest,
        None,  # type: ignore[arg-type]
        [_part([0, 3, 6, 12], 0)],
        expected_source_row_step=3,
    )
    assert combined["source_row"].tolist() == [0, 3, 6, 12]
    assert combined["stream_lengths"].tolist() == [3, 1]


def test_reference_support_restriction_preserves_frozen_order(tmp_path: Path) -> None:
    combined = {
        "truth": np.asarray([[0] * 7, [1] * 7, [0] * 7], dtype=np.uint8),
        "probability": np.asarray([[0.1] * 7, [0.9] * 7, [0.2] * 7], dtype=np.float32),
        "active": np.ones(3, dtype=np.uint8),
        "source_row": np.asarray([10, 11, 12], dtype=np.int64),
        "engine_idx": np.asarray([110, 111, 112], dtype=np.int64),
        "row_session_id": np.asarray(["rec_a", "rec_a", "rec_a"]),
        "stream_lengths": np.asarray([3], dtype=np.int64),
        "stream_ids": np.asarray(["rec_a__run000__sub000"]),
    }
    reference = tmp_path / "reference.npz"
    np.savez_compressed(
        reference,
        y_true=combined["truth"][[2, 0]],
        y_prob=np.zeros((2, 7), dtype=np.float32),
        input_active=combined["active"][[2, 0]],
        source_row_index=combined["source_row"][[2, 0]],
        source_engine_frame_idx=combined["engine_idx"][[2, 0]],
        session_lengths=np.asarray([2], dtype=np.int64),
        session_ids=np.asarray(["rec_a__run000__sub000"]),
    )
    selected = restrict_to_reference_support(combined, reference)
    assert selected["source_row"].tolist() == [12, 10]
    assert selected["probability"][:, 0].tolist() == pytest.approx([0.2, 0.1])


def test_fixed_only_summary_does_not_fit_or_emit_oracle_thresholds() -> None:
    truth = np.zeros((12, 7), dtype=np.uint8)
    truth[3:6, 4] = 1
    probability = np.full((12, 7), 0.1, dtype=np.float32)
    probability[3:6, 4] = 0.9
    report = summarize(
        truth,
        probability,
        boundaries=[12],
        fixed_transition_thresholds={key: 0.5 for key in (
            "left", "right", "up", "down", "jump", "dash", "grab"
        )},
        include_oracle=False,
    )
    assert "transition_f1_oracle" not in report
    assert "transition_f1_oracle_collars" not in report
    assert report["transition_f1_at_0.5_collars"]["2"]["jump"]["event"]["f1"] == 1.0


def test_dense_center_probabilities_flatten_to_one_prediction_per_row() -> None:
    first = np.arange(2 * 64 * 7, dtype=np.float32).reshape(2, 64, 7)
    second = np.arange(64 * 7, dtype=np.float32).reshape(1, 64, 7)
    flattened = flatten_center_probabilities([first, second])
    assert flattened.shape == (3 * 64, 7)
    assert np.array_equal(flattened[: 2 * 64], first.reshape(-1, 7))
    assert np.array_equal(flattened[2 * 64 :], second.reshape(-1, 7))


def test_span_matched_geometry_retains_center_192() -> None:
    assert retained_positions(128, 64) == (32, 96)
    assert retained_positions(384, 192) == (96, 288)
    block = np.arange(2 * 192 * 7, dtype=np.float32).reshape(2, 192, 7)
    flattened = flatten_center_probabilities([block], retained=192)
    assert flattened.shape == (384, 7)


def test_tail_overlap_keeps_base_prediction_and_fills_missing_rows() -> None:
    rows = np.asarray([10, 11, 12, 12, 13], dtype=np.int64)
    probability = np.asarray([[0.1], [0.2], [0.3], [0.9], [0.4]], dtype=np.float32)
    kept_rows, kept_probability = resolve_center_overlap(
        rows, probability, policy="base-first-stable-tail-fill"
    )
    assert kept_rows.tolist() == [10, 11, 12, 13]
    assert kept_probability[:, 0].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_unfrozen_center_overlap_fails_closed() -> None:
    rows = np.asarray([10, 10], dtype=np.int64)
    probability = np.zeros((2, 7), dtype=np.float32)
    with pytest.raises(RuntimeError, match="without a frozen policy"):
        resolve_center_overlap(rows, probability, policy=None)


def test_wild_phase0_requires_explicit_deployment_support_mode() -> None:
    with pytest.raises(ValueError, match="requires all phases"):
        validate_temporal_support(
            derived_rate_hz=20,
            phases=[0],
            support_mode="native-grid",
        )
    validate_temporal_support(
        derived_rate_hz=20,
        phases=[0],
        support_mode="deployment-20hz-phase0",
    )


def test_deployment_support_mode_rejects_non_phase0_manifests() -> None:
    with pytest.raises(ValueError, match="requires derived_rate_hz=20 and phase 0"):
        validate_temporal_support(
            derived_rate_hz=20,
            phases=[0, 1, 2],
            support_mode="deployment-20hz-phase0",
        )


def test_zero_window_streams_are_excluded_from_evaluation() -> None:
    records = [
        {"session_id": "short", "windows": 0},
        {"session_id": "usable", "windows": 2},
    ]
    assert center_supported_records(records) == [records[1]]
    with pytest.raises(ValueError, match="no center-supported windows"):
        center_supported_records(records[:1])
