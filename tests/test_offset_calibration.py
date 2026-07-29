from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.schema import KEY_ORDER
from harvest.accept_wild_offset import accept_offset, verify_offset_acceptance
import harvest.calibrate_offset as calibration
from harvest.calibrate_offset import (
    OffsetPolicy,
    calibrate_offset,
    evaluate_offset_evidence,
    fingerprint_scores,
    write_contact_sheet,
)
from harvest.fetch_wild import sha256_file
from harvest.wild_layout import SCHEMA_VERSION, WildLayout
from tests.test_layout_acceptance import _layout_review_fixture


def _layout_dict(video_id: str = "offset_test") -> dict:
    cells = []
    for index, action in enumerate(KEY_ORDER):
        x = 0.05 + index * 0.085
        cells.append({
            "cell_id": f"cell_{action}",
            "action": action,
            "sample_rect": [x, 0.82, 0.06, 0.08],
            "reference_rect": [x, 0.92, 0.06, 0.03],
            "decoder": "local_contrast",
            "pressed_polarity": "high",
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "overlay_style": "synthetic_action_grid",
        "gameplay_rect": [0.0, 0.0, 1.0, 1.0],
        "gameplay_rect_source": "synthetic_review",
        "gameplay_rect_confidence": 1.0,
        "mask_rects": [[0.02, 0.78, 0.66, 0.20]],
        "cells": cells,
        "inference_source": "synthetic_fixture",
        "inference_confidence": 1.0,
        "human_reviewed": True,
        "evidence_frames_s": [1.0, 5.0, 9.0],
        "temporal_offset_frames": 0,
        "temporal_offset_source": "unmeasured",
        "temporal_offset_confidence": 0.0,
    }


def _motions_for_lags(
    event_lags: list[int], policy: OffsetPolicy
) -> tuple[np.ndarray, np.ndarray, int]:
    relative_start = policy.min_lag - 3
    relative_end = policy.max_lag + 4
    relative_frames = np.arange(relative_start, relative_end + 1)
    rows = []
    for event_number, lag in enumerate(event_lags):
        # Low deterministic variation prevents the fixture from depending on
        # accidental floating-point ties while retaining an exact fingerprint.
        row = 8.0 + 0.02 * np.sin(relative_frames + event_number)
        for relative_frame in (lag + 1, lag + 2, lag + 3):
            row[relative_frame - relative_start] = 0.20 + 0.001 * (event_number % 3)
        row[lag + 4 - relative_start] = 12.0 + 0.01 * (event_number % 5)
        rows.append(row)
    onsets = 100 + np.arange(len(rows), dtype=np.int64) * 300
    return np.stack(rows), onsets, relative_start


@pytest.mark.parametrize("known_lag", [-6, 0, 5])
def test_hitstop_estimator_recovers_known_integer_shift(known_lag: int) -> None:
    policy = OffsetPolicy()
    motions, onsets, relative_start = _motions_for_lags([known_lag] * 60, policy)
    result = evaluate_offset_evidence(motions, onsets, relative_start, policy)
    assert result["automatic_gates_passed"]
    assert result["verdict"] == "pass"
    assert result["best_candidate_offset_frames"] == known_lag
    assert result["per_event_modal_offset_frames"] == known_lag
    assert result["per_event_mode_fraction"] == 1.0
    assert result["per_event_collar_fraction"] == 1.0
    assert result["bootstrap_win_fraction"] == 1.0
    assert {row["winner_offset_frames"] for row in result["temporal_blocks"]} == {
        known_lag
    }


def test_max_freeze_penalizes_one_frame_shift_that_includes_rebound() -> None:
    policy = OffsetPolicy()
    motions, _, relative_start = _motions_for_lags([0], policy)
    scores = fingerprint_scores(motions, relative_start, np.asarray([0, 1]), 0.25)[0]
    assert scores[0] > 5.0
    assert scores[1] < 0.0


def test_static_scene_fails_closed() -> None:
    policy = OffsetPolicy()
    relative_start = policy.min_lag - 3
    width = policy.max_lag + 4 - relative_start + 1
    motions = np.zeros((60, width), dtype=np.float64)
    result = evaluate_offset_evidence(
        motions, np.arange(60, dtype=np.int64) * 300, relative_start, policy
    )
    assert not result["automatic_gates_passed"]
    assert result["verdict"] == "fail"
    assert result["best_candidate_offset_frames"] is None
    assert result["usable_events"] == 0


def test_multimodal_and_temporally_drifting_offset_fails_closed() -> None:
    policy = OffsetPolicy()
    # Each chronological third carries a different offset.  Averaging this
    # evidence into one offset would silently misalign most of the video.
    motions, onsets, relative_start = _motions_for_lags(
        [-4] * 20 + [0] * 20 + [4] * 20, policy
    )
    result = evaluate_offset_evidence(motions, onsets, relative_start, policy)
    assert not result["automatic_gates_passed"]
    assert result["verdict"] == "fail"
    assert any("temporal block" in reason for reason in result["failure_reasons"])


def test_contact_sheet_contains_ranked_repeated_events(tmp_path: Path) -> None:
    layout = WildLayout.from_dict(_layout_dict())
    video = tmp_path / "contact.mp4"
    width, height = 160, 90
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (width, height)
    )
    assert writer.isOpened()
    for frame_index in range(100):
        frame = np.full((height, width, 3), frame_index % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    out = tmp_path / "contact.png"
    policy = OffsetPolicy(contact_events=2, contact_frame_size=32)
    evidence = write_contact_sheet(
        video,
        layout,
        np.asarray([25, 70]),
        np.asarray([1.0, 2.0]),
        winner=-2,
        out_path=out,
        policy=policy,
    )
    image = cv2.imread(str(out))
    assert image is not None
    assert image.shape[:2] == (2 * (28 + 32), 6 * 32)
    assert [row["hud_onset_frame"] for row in evidence] == [70, 25]


def test_full_calibration_writes_pending_handoff_without_mutating_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = OffsetPolicy()
    video = tmp_path / "source.mp4"
    video.write_bytes(b"synthetic source identity")
    layout_path, layout_acceptance, _, _ = _layout_review_fixture(
        tmp_path,
        video_id="offset_test",
        source_sha256=sha256_file(video),
    )
    layout_before = layout_path.read_bytes()

    dash_onsets = 100 + np.arange(60, dtype=np.int64) * 300
    total_frames = int(dash_onsets[-1] + 100)
    dash = np.zeros(total_frames, dtype=bool)
    dash[dash_onsets] = True
    labels = tmp_path / "labels_raw.parquet"
    pq.write_table(pa.table({
        "video_frame_idx": np.arange(total_frames, dtype=np.int64),
        "dash": dash,
        "gameplay_allowed": np.ones(total_frames, dtype=bool),
    }), labels)
    decode = {
        "video_id": "offset_test",
        "layout": {"sha256": sha256_file(layout_path)},
        "source_video": {"sha256": sha256_file(video)},
        "rejection_reasons": ["HUD compositor offset is unmeasured"],
        "timing": {"pts": {"effective_fps": 60.0, "vfr_ratio_p99_p01": 1.0}},
        "raw_labels_sha256": sha256_file(labels),
    }
    decode_path = tmp_path / "decode_report.json"
    decode_path.write_text(json.dumps(decode))
    motions, _, relative_start = _motions_for_lags([-3] * 60, policy)

    def fake_collect(video_arg, layout_arg, onsets_arg, policy_arg):
        assert np.array_equal(onsets_arg, dash_onsets)
        return motions, onsets_arg, relative_start

    def fake_sheet(video_arg, layout_arg, onsets_arg, scores_arg, winner, out, policy_arg):
        assert winner == -3
        assert onsets_arg.size == 60
        Path(out).write_bytes(b"review me")
        return [{"hud_onset_frame": int(onsets_arg[0])}]

    monkeypatch.setattr(calibration, "collect_motion_evidence", fake_collect)
    monkeypatch.setattr(calibration, "write_contact_sheet", fake_sheet)
    out_dir = tmp_path / "calibration"
    report = calibrate_offset(
        video, layout_path, labels, decode_path, out_dir, policy=policy
    )

    assert report["automatic_gates_passed"]
    assert report["verdict"] == "pass"
    assert report["best_candidate_offset_frames"] == -3
    assert report["offset_uncertainty_frames"] == 1
    assert report["human_contact_sheet_review"] == "pending"
    assert not report["calibration_accepted"]
    assert not report["layout_was_modified"]
    assert layout_path.read_bytes() == layout_before
    assert (out_dir / "offset_calibration.json").is_file()
    assert (out_dir / "offset_calibration.sha256").is_file()
    assert (out_dir / "score_matrix.npz").is_file()
    assert report["score_matrix"]["sha256"] == sha256_file(out_dir / "score_matrix.npz")

    final_layout = tmp_path / "layout.final.json"
    acceptance = out_dir / "offset_acceptance.json"
    accepted = accept_offset(
        out_dir / "offset_calibration.json",
        layout_path,
        layout_acceptance,
        final_layout,
        acceptance,
        reviewer_identity="synthetic human reviewer",
        reviewer_kind="human",
        approved=True,
    )
    assert accepted["accepted_offset_frames"] == -3
    verified = verify_offset_acceptance(
        final_layout,
        WildLayout.load(final_layout),
        acceptance,
        source_sha256=sha256_file(video),
        layout_acceptance_path=layout_acceptance,
    )
    assert verified["reviewer_kind"] == "human"
