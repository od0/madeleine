from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harvest.fetch_wild import sha256_file
from harvest.resegment_timer_trace import resegment_timer_trace
from harvest.timer_activity import (
    TimerActivityPolicy,
    TimerReviewContext,
    segment_timer_activity,
)


def _packet(tmp_path: Path) -> tuple[Path, Path]:
    fps = 60
    pts = np.arange(1200, dtype=np.float64) / fps
    change = np.zeros(pts.size, dtype=np.float64)
    change[1:481] = 1.0
    change[601:1081] = 1.0
    bright = np.full(pts.size, 5.0)
    dark = np.full(pts.size, 20.0)
    trace = tmp_path / "timer_trace.npz"
    np.savez_compressed(
        trace,
        pts_s=pts,
        change_score=change,
        bright_mask_mean=bright,
        dark_mask_mean=dark,
    )
    context = TimerReviewContext(
        video_id="video",
        source_sha256="a" * 64,
        timer_roi_normalized_xywh=(0.1, 0.1, 0.2, 0.1),
        timer_roi_evidence_reviewed=True,
        wall_clock_bounds_s=(0.0, 20.0),
        bounds_evidence_reviewed=True,
        reviewer_identity="AI diagnostic",
        reviewer_kind="ai_agent",
        nominal_loadless_duration_s=16.0,
        evidence=("timer_roi=fixture", "wall_clock_bounds=fixture"),
    )
    policy = TimerActivityPolicy(
        min_effective_fps=59.0,
        max_effective_fps=61.0,
    )
    report = segment_timer_activity(
        pts,
        change,
        context,
        policy,
        bright_mask_mean=bright,
        dark_mask_mean=dark,
    )
    report["trace_binding"] = {
        "format_version": "madeleine.wild-timer-trace.v1",
        "source_sha256": "a" * 64,
        "trace_npz_sha256": sha256_file(trace),
        "frames": pts.size,
    }
    proposal = tmp_path / "timer_activity_proposal.json"
    proposal.write_text(json.dumps(report, allow_nan=False) + "\n")
    return proposal, trace


def test_resegments_ai_trace_and_records_exact_ablation(tmp_path: Path) -> None:
    proposal, trace = _packet(tmp_path)
    output = tmp_path / "timer_activity_ablation.json"
    result = resegment_timer_trace(
        proposal,
        trace,
        output,
        policy_overrides={
            "min_bright_mask_mean": 0.0,
            "min_dark_mask_mean": 0.0,
            "max_bridge_s": 2.0,
        },
        rationale="fixture timer is rendered over changing gameplay",
    )
    loaded = json.loads(output.read_text())
    assert loaded == result
    assert loaded["auto_admitted"] is False
    assert loaded["review_provenance_gate_passed"] is False
    assert loaded["input_review"]["human_reviewed"] is False
    assert loaded["policy"]["max_bridge_s"] == 2.0
    ablation = loaded["policy_ablation"]
    assert ablation["parent_proposal_sha256"] == sha256_file(proposal)
    assert ablation["trace_npz_sha256"] == sha256_file(trace)
    assert ablation["overrides"] == {
        "max_bridge_s": 2.0,
        "min_bright_mask_mean": 0.0,
        "min_dark_mask_mean": 0.0,
    }
    assert ablation["admission_effect"].startswith("none")


def test_rejects_trace_mismatch_human_review_and_overwrite(tmp_path: Path) -> None:
    proposal, trace = _packet(tmp_path)
    bad_trace = tmp_path / "different_trace.npz"
    np.savez_compressed(
        bad_trace,
        pts_s=np.arange(2, dtype=np.float64),
        change_score=np.zeros(2),
        bright_mask_mean=np.zeros(2),
        dark_mask_mean=np.zeros(2),
    )
    with pytest.raises(ValueError, match="hash differs"):
        resegment_timer_trace(
            proposal,
            bad_trace,
            tmp_path / "bad.json",
            policy_overrides={"max_bridge_s": 2.0},
            rationale="test",
        )

    value = json.loads(proposal.read_text())
    value["input_review"].update(
        reviewer_kind="human_with_ai_assistance", human_reviewed=True
    )
    proposal.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="AI-only"):
        resegment_timer_trace(
            proposal,
            trace,
            tmp_path / "human.json",
            policy_overrides={"max_bridge_s": 2.0},
            rationale="test",
        )

    output = tmp_path / "existing.json"
    output.write_text("existing\n")
    with pytest.raises(FileExistsError, match="overwrite"):
        resegment_timer_trace(
            proposal,
            trace,
            output,
            policy_overrides={"max_bridge_s": 2.0},
            rationale="test",
        )
    assert output.read_text() == "existing\n"


def test_rejects_unscoped_policy_override(tmp_path: Path) -> None:
    proposal, trace = _packet(tmp_path)
    with pytest.raises(ValueError, match="unsupported"):
        resegment_timer_trace(
            proposal,
            trace,
            tmp_path / "output.json",
            policy_overrides={"min_allowed_s": 0.1},
            rationale="test",
        )
