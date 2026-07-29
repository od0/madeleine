from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harvest.fetch_wild import sha256_file
from harvest.materialize_ai_boundaries import materialize_ai_boundaries
from harvest.timer_activity import (
    TIMER_ACTIVITY_VERSION,
    TimerActivityPolicy,
    TimerReviewContext,
    segment_timer_activity,
)
from harvest.wild_boundaries import WildBoundaries


def _proposal() -> dict:
    source = "a" * 64
    return {
        "format_version": TIMER_ACTIVITY_VERSION,
        "video_id": "video",
        "status": "abstained",
        "auto_admitted": False,
        "signal_quality_gates_passed": True,
        "review_provenance_gate_passed": False,
        "input_review": {
            "video_id": "video",
            "source_sha256": source,
            "reviewer_kind": "ai_agent",
            "human_reviewed": False,
            "wall_clock_bounds_s": [10.0, 50.0],
        },
        "trace_binding": {
            "source_sha256": source,
            "trace_npz_sha256": "b" * 64,
        },
        "activity": {
            "candidate_ranges_before_gates": {
                "rows": [
                    {"range_s": [11.0, 20.0], "duration_s": 9.0, "frames": 540},
                    {"range_s": [25.0, 49.0], "duration_s": 24.0, "frames": 1440},
                ],
                "total": 2,
                "truncated": False,
            }
        },
        "proposal_quality": {
            "check": "passed",
            "candidate_range_count": 2,
            "candidate_seconds": 33.0,
        },
    }


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "timer_activity_proposal.json"
    path.write_text(json.dumps(value) + "\n")
    return path


def test_materializes_only_ai_reviewed_provisional_ranges(tmp_path: Path) -> None:
    proposal = _write(tmp_path, _proposal())
    output = tmp_path / "boundaries.ai.json"
    result = materialize_ai_boundaries(
        proposal,
        output,
        reviewer_identity="OpenAI Codex diagnostic",
    )
    loaded = WildBoundaries.load(output)
    assert result == loaded.to_dict()
    assert loaded.human_reviewed is False
    assert loaded.reviewer_kind == "ai_agent"
    assert loaded.ranges_s == ((11.0, 20.0), (25.0, 49.0))
    assert any("timer_proposal_sha256=" in value for value in loaded.evidence)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(status="review_required"), "abstained"),
        (lambda value: value.update(auto_admitted=True), "non-admitted"),
        (
            lambda value: value.update(signal_quality_gates_passed=False),
            "signal-quality",
        ),
        (
            lambda value: value.update(review_provenance_gate_passed=True),
            "normal human boundary path",
        ),
        (
            lambda value: value["input_review"].update(
                reviewer_kind="human", human_reviewed=True
            ),
            "AI-only",
        ),
        (
            lambda value: value["activity"]["candidate_ranges_before_gates"].update(
                truncated=True
            ),
            "truncated",
        ),
        (
            lambda value: value["proposal_quality"].update(candidate_seconds=34.0),
            "seconds differ",
        ),
    ],
)
def test_rejects_nonprovisional_or_inconsistent_proposals(
    tmp_path: Path, mutation, message: str
) -> None:
    value = _proposal()
    mutation(value)
    proposal = _write(tmp_path, value)
    with pytest.raises(ValueError, match=message):
        materialize_ai_boundaries(
            proposal,
            tmp_path / "boundaries.json",
            reviewer_identity="diagnostic",
        )


def test_rejects_incomplete_rows_and_refuses_overwrite(tmp_path: Path) -> None:
    value = _proposal()
    value["activity"]["candidate_ranges_before_gates"]["total"] = 3
    proposal = _write(tmp_path, value)
    output = tmp_path / "boundaries.json"
    with pytest.raises(ValueError, match="total"):
        materialize_ai_boundaries(
            proposal, output, reviewer_identity="diagnostic"
        )

    output.write_text("existing\n")
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_ai_boundaries(
            proposal, output, reviewer_identity="diagnostic"
        )
    assert output.read_text() == "existing\n"


def test_rebuilds_complete_candidate_rows_from_hash_bound_trace(tmp_path: Path) -> None:
    fps = 60
    pts = np.arange(720, dtype=np.float64) / fps
    change = np.zeros(pts.size, dtype=np.float64)
    change[1:181] = 1.0
    change[361:601] = 1.0
    bright = np.full(pts.size, 255.0)
    dark = np.full(pts.size, 255.0)
    context = TimerReviewContext(
        video_id="video",
        source_sha256="a" * 64,
        timer_roi_normalized_xywh=(0.1, 0.1, 0.1, 0.1),
        timer_roi_evidence_reviewed=True,
        wall_clock_bounds_s=(0.0, 12.0),
        bounds_evidence_reviewed=True,
        reviewer_identity="AI diagnostic",
        reviewer_kind="ai_agent",
        nominal_loadless_duration_s=7.0,
        evidence=("fixture",),
    )
    policy = TimerActivityPolicy(
        min_effective_fps=59.0,
        max_effective_fps=61.0,
        max_diagnostic_runs=1,
        min_allowed_s=2.0,
        segment_shape_min_envelope_s=300.0,
    )
    report = segment_timer_activity(
        pts,
        change,
        context,
        policy,
        bright_mask_mean=bright,
        dark_mask_mean=dark,
    )
    assert report["signal_quality_gates_passed"] is True
    assert report["activity"]["candidate_ranges_before_gates"]["truncated"] is True
    trace = tmp_path / "timer_trace.npz"
    np.savez_compressed(
        trace,
        pts_s=pts,
        change_score=change,
        bright_mask_mean=bright,
        dark_mask_mean=dark,
    )
    report["trace_binding"] = {
        "source_sha256": "a" * 64,
        "trace_npz_sha256": sha256_file(trace),
        "frames": pts.size,
    }
    proposal = _write(tmp_path, report)
    output = tmp_path / "boundaries.json"
    result = materialize_ai_boundaries(
        proposal,
        output,
        reviewer_identity="AI materializer",
        trace_path=trace,
    )
    assert result["human_reviewed"] is False
    assert len(result["allowed_ranges_s"]) == 2
    assert result["allowed_ranges_s"][0] == pytest.approx([1 / 60, 181 / 60])


def test_truncated_candidates_require_exact_bound_trace(tmp_path: Path) -> None:
    value = _proposal()
    value["activity"]["candidate_ranges_before_gates"].update(truncated=True)
    proposal = _write(tmp_path, value)
    with pytest.raises(ValueError, match="hash-bound timer trace is required"):
        materialize_ai_boundaries(
            proposal,
            tmp_path / "boundaries.json",
            reviewer_identity="AI materializer",
        )
