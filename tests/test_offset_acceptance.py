from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.schema import KEY_ORDER
from harvest.calibrate_offset import CALIBRATION_VERSION
from harvest.accept_wild_offset import (
    ACCEPTANCE_VERSION,
    accept_offset,
    verify_offset_acceptance,
)
from harvest.decode_wild import decode_video
from harvest.fetch_wild import sha256_file
from harvest.wild_layout import SCHEMA_VERSION, WildLayout
from tests.test_layout_acceptance import _layout_review_fixture


def _layout_dict(video_id: str = "acceptance_test") -> dict:
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
        "gameplay_rect_source": "reviewed_fixture",
        "gameplay_rect_confidence": 1.0,
        "mask_rects": [[0.02, 0.78, 0.66, 0.20]],
        "cells": cells,
        "inference_source": "reviewed_fixture",
        "inference_confidence": 1.0,
        "human_reviewed": True,
        "evidence_frames_s": [1.0, 5.0],
        "temporal_offset_frames": 0,
        "temporal_offset_source": "unmeasured",
        "temporal_offset_confidence": 0.0,
    }


def _write_calibration(
    directory: Path,
    layout_path: Path,
    *,
    winner: int = -3,
    source_sha256: str = "a" * 64,
    verdict: str = "pass",
) -> tuple[Path, Path]:
    directory.mkdir()
    contact = directory / "dash_offset_contact.png"
    contact.write_bytes(b"reviewed contact evidence")
    lags = list(range(-12, 13))
    medians = {lag: 0.0 for lag in lags}
    medians[winner] = 3.0
    runner = next(lag for lag in lags if lag != winner)
    # The runner-up is non-adjacent, so the non-adjacent median margin equals
    # winner minus runner: 2.0 for the pass fixture (at the floor) and 1.5 for
    # the uncertain_adjacent fixture (below it).
    medians[runner] = 1.0 if verdict == "pass" else 1.5
    margin = medians[winner] - medians[runner]
    candidates = []
    for lag in lags:
        candidates.append({
            "offset_frames": lag,
            "median_score": medians[lag],
            "mean_score": medians[lag],
            "event_wins": 60 if lag == winner else 0,
            "bootstrap_wins": 2_000 if lag == winner else 0,
            "bootstrap_fraction": 1.0 if lag == winner else 0.0,
        })
    calibration = {
        "format_version": CALIBRATION_VERSION,
        "video_id": WildLayout.load(layout_path).video_id,
        "inputs": {
            "layout_sha256": sha256_file(layout_path),
            "video_sha256": source_sha256,
            "labels_sha256": "b" * 64,
            "decode_report_sha256": "c" * 64,
        },
        "policy": {
            "min_lag": -12,
            "max_lag": 12,
            "min_events": 20,
            "min_effective_fps": 59.0,
            "max_effective_fps": 61.0,
            "max_vfr_ratio_p99_p01": 1.10,
            "min_local_motion_range": 0.50,
            "min_event_score": 3.0,
            "min_median_margin": 2.0,
            "mode_lag_collar": 1,
            "margin_nonadjacent_gap": 2,
            "bootstrap_samples": 2_000,
            "min_bootstrap_win_fraction": 0.95,
            "temporal_blocks": 3,
            "min_events_per_block": 4,
        },
        "events": {
            "dash_onsets_in_labels": 60,
            "motion_windows_decoded": 60,
            "usable_quality_matches": 60,
        },
        "candidates": candidates,
        "best_candidate_offset_frames": winner,
        "runner_up_offset_frames": runner,
        "median_score_margin": margin,
        "per_event_modal_offset_frames": winner,
        "per_event_mode_fraction": 1.0,
        "per_event_collar_fraction": 1.0,
        "bootstrap_win_fraction": 1.0,
        "temporal_blocks": [
            {
                "block": index,
                "events": 20,
                "first_onset_frame": index * 1_000,
                "last_onset_frame": index * 1_000 + 900,
                "winner_offset_frames": winner,
            }
            for index in range(3)
        ],
        "offset_uncertainty_frames": 1,
        "verdict": verdict,
        "automatic_gates_passed": verdict == "pass",
        "automatic_failure_reasons": [] if verdict == "pass" else [
            f"non-adjacent median margin {margin:.4f} < required 2.0000"
        ],
        "human_contact_sheet_review": "pending",
        "calibration_accepted": False,
        "layout_was_modified": False,
        "human_handoff": {
            "contact_sheet": contact.name,
            "contact_sheet_sha256": sha256_file(contact),
            "events": [],
        },
    }
    path = directory / "offset_calibration.json"
    path.write_text(json.dumps(calibration, indent=2) + "\n")
    path.with_suffix(".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n"
    )
    return path, contact


def _accepted_fixture(
    tmp_path: Path, *, reviewer_kind: str = "human_with_ai_assistance"
) -> tuple[Path, Path, Path, Path]:
    input_layout, layout_acceptance, _, _ = _layout_review_fixture(
        tmp_path, video_id="acceptance_test"
    )
    calibration, contact = _write_calibration(tmp_path / "calibration", input_layout)
    output_layout = tmp_path / "layout.final.json"
    acceptance = calibration.parent / "offset_acceptance.json"
    accept_offset(
        calibration,
        input_layout,
        layout_acceptance,
        output_layout,
        acceptance,
        reviewer_identity="Test Reviewer",
        reviewer_kind=reviewer_kind,
        approved=True,
        notes="Inspected all contact rows.",
    )
    return output_layout, acceptance, calibration, contact


def test_acceptance_binds_layout_calibration_contact_and_reviewer(tmp_path: Path) -> None:
    output_layout, acceptance_path, calibration, contact = _accepted_fixture(tmp_path)
    acceptance = json.loads(acceptance_path.read_text())
    layout = WildLayout.load(output_layout)

    assert acceptance["format_version"] == ACCEPTANCE_VERSION
    assert acceptance["decision"] == {
        "approved": True,
        "contact_sheet_reviewed": True,
        "reviewer_identity": "Test Reviewer",
        "reviewer_kind": "human_with_ai_assistance",
        "uncertain_tier_acknowledged": False,
        "notes": "Inspected all contact rows.",
    }
    assert acceptance["accepted_tier"] == "pass"
    assert acceptance["accepted_offset_uncertainty_frames"] == 1
    assert acceptance["calibration"]["sha256"] == sha256_file(calibration)
    assert acceptance["calibration"]["contact_sheet"]["sha256"] == sha256_file(contact)
    assert acceptance["output_layout"]["sha256"] == sha256_file(output_layout)
    assert layout.temporal_offset_frames == -3
    assert layout.temporal_offset_confidence == 1.0
    verified = verify_offset_acceptance(
        output_layout,
        layout,
        acceptance_path,
        source_sha256="a" * 64,
        layout_acceptance_path=tmp_path / "layout-review" / "layout_acceptance.json",
    )
    assert verified["reviewer_identity"] == "Test Reviewer"
    assert verified["reviewer_kind"] == "human_with_ai_assistance"
    assert verified["human_reviewed"] is True


def test_ai_agent_review_is_explicit_and_does_not_become_human_review(
    tmp_path: Path,
) -> None:
    output_layout, acceptance, _, _ = _accepted_fixture(
        tmp_path, reviewer_kind="ai_agent"
    )
    verified = verify_offset_acceptance(
        output_layout,
        WildLayout.load(output_layout),
        acceptance,
        source_sha256="a" * 64,
        layout_acceptance_path=tmp_path / "layout-review" / "layout_acceptance.json",
    )
    assert verified["reviewer_kind"] == "ai_agent"
    assert verified["human_reviewed"] is False


def test_ai_layout_acceptance_cannot_enter_offset_acceptance(tmp_path: Path) -> None:
    layout, layout_acceptance, _, _ = _layout_review_fixture(
        tmp_path,
        video_id="acceptance_test",
        reviewer_kind="ai_agent",
    )
    calibration, _ = _write_calibration(tmp_path / "calibration", layout)
    with pytest.raises(ValueError, match="was not reviewed by a human"):
        accept_offset(
            calibration,
            layout,
            layout_acceptance,
            tmp_path / "layout.final.json",
            calibration.parent / "offset_acceptance.json",
            reviewer_identity="Offset Reviewer",
            reviewer_kind="human",
            approved=True,
        )


def test_acceptance_refuses_failed_gate_and_nonapproval(tmp_path: Path) -> None:
    layout_path, layout_acceptance, _, _ = _layout_review_fixture(
        tmp_path, video_id="acceptance_test"
    )
    calibration, _ = _write_calibration(tmp_path / "calibration", layout_path)
    raw = json.loads(calibration.read_text())
    raw["verdict"] = "fail"
    raw["automatic_gates_passed"] = False
    raw["automatic_failure_reasons"] = ["weak evidence"]
    calibration.write_text(json.dumps(raw, indent=2) + "\n")
    calibration.with_suffix(".sha256").write_text(
        f"{sha256_file(calibration)}  {calibration.name}\n"
    )
    with pytest.raises(ValueError, match="verdict must be pass or uncertain_adjacent"):
        accept_offset(
            calibration,
            layout_path,
            layout_acceptance,
            tmp_path / "final.json",
            calibration.parent / "acceptance.json",
            reviewer_identity="Reviewer",
            reviewer_kind="human",
            approved=True,
        )
    with pytest.raises(ValueError, match="explicit contact-sheet approval"):
        accept_offset(
            calibration,
            layout_path,
            layout_acceptance,
            tmp_path / "final.json",
            calibration.parent / "acceptance.json",
            reviewer_identity="Reviewer",
            reviewer_kind="human",
            approved=False,
        )


def test_v2_calibration_remains_readable_but_is_not_acceptable(tmp_path: Path) -> None:
    layout_path, layout_acceptance, _, _ = _layout_review_fixture(
        tmp_path, video_id="acceptance_test"
    )
    calibration, _ = _write_calibration(tmp_path / "calibration", layout_path)
    raw = json.loads(calibration.read_text())
    raw["format_version"] = "madeleine.dash-hitstop-offset.v2"
    calibration.write_text(json.dumps(raw, indent=2) + "\n")
    calibration.with_suffix(".sha256").write_text(
        f"{sha256_file(calibration)}  {calibration.name}\n"
    )
    with pytest.raises(ValueError, match="unsupported calibration format_version"):
        accept_offset(
            calibration,
            layout_path,
            layout_acceptance,
            tmp_path / "final.json",
            calibration.parent / "acceptance.json",
            reviewer_identity="Reviewer",
            reviewer_kind="human",
            approved=True,
        )


def test_acceptance_and_verifier_fail_closed_on_overwrite_or_tamper(tmp_path: Path) -> None:
    output_layout, acceptance, calibration, contact = _accepted_fixture(tmp_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        accept_offset(
            calibration,
            tmp_path / "layout.reviewed.json",
            tmp_path / "layout-review" / "layout_acceptance.json",
            output_layout,
            acceptance,
            reviewer_identity="Another Reviewer",
            reviewer_kind="human",
            approved=True,
        )

    contact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="contact-sheet hash mismatch"):
        verify_offset_acceptance(
            output_layout,
            WildLayout.load(output_layout),
            acceptance,
            source_sha256="a" * 64,
            layout_acceptance_path=tmp_path / "layout-review" / "layout_acceptance.json",
        )


def test_measured_decode_requires_acceptance_before_reading_video(tmp_path: Path) -> None:
    output_layout, _, _, _ = _accepted_fixture(tmp_path)
    fetch = tmp_path / "fetch.json"
    fetch.write_text(json.dumps({"video_id": "acceptance_test"}))
    with pytest.raises(ValueError, match="requires a verified offset acceptance"):
        decode_video(
            fetch,
            output_layout,
            tmp_path / "missing-boundaries.json",
            tmp_path / "decoded",
        )
