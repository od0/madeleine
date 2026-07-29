"""Materialize signal-valid timer diagnostics for provisional wild decode.

Timer proposals generated from AI-selected geometry deliberately expose no
``suggested_allowed_ranges_s`` because they cannot satisfy the human review
gate.  Their signal-valid candidate ranges are still useful for provisional
decode, threshold diagnostics, and offset-calibration preparation.  This tool
converts those ranges into an ordinary ``WildBoundaries`` v2 artifact whose
reviewer kind is unconditionally ``ai_agent`` and whose human gate is therefore
unconditionally false.

The output can never admit training data.  Human-reviewed boundaries must be
created separately after inspecting the timer packet.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from harvest.fetch_wild import sha256_file
from harvest.timer_activity import (
    TIMER_ACTIVITY_VERSION,
    TimerActivityPolicy,
    TimerReviewContext,
    segment_timer_activity,
)
from harvest.wild_boundaries import WildBoundaries


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("timer proposal must be a non-symlink regular file")

    def reject_constant(value: str) -> None:
        raise ValueError(f"timer proposal contains non-finite JSON number {value}")

    try:
        return _mapping(
            json.loads(path.read_text(), parse_constant=reject_constant),
            "timer proposal",
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("timer proposal is not valid UTF-8 JSON") from exc


def _complete_candidate_rows_from_trace(
    proposal: dict[str, Any],
    trace_path: Path,
) -> dict[str, Any]:
    """Re-run deterministic segmentation when JSON diagnostics were bounded."""

    if trace_path.is_symlink() or not trace_path.is_file():
        raise ValueError("timer trace must be a non-symlink regular file")
    trace_binding = _mapping(proposal.get("trace_binding"), "timer proposal.trace_binding")
    expected_hash = str(trace_binding.get("trace_npz_sha256", "")).strip().lower()
    if sha256_file(trace_path) != expected_hash:
        raise ValueError("timer trace hash differs from the proposal binding")
    with np.load(trace_path, allow_pickle=False) as trace:
        required = {"pts_s", "change_score", "bright_mask_mean", "dark_mask_mean"}
        if not required.issubset(trace.files):
            raise ValueError("timer trace is missing required scalar arrays")
        arrays = {name: np.asarray(trace[name]) for name in required}
    frames = arrays["pts_s"].size
    if frames <= 1 or any(value.ndim != 1 or value.size != frames for value in arrays.values()):
        raise ValueError("timer trace scalar arrays must be aligned one-dimensional vectors")
    if trace_binding.get("frames") != frames:
        raise ValueError("timer trace frame count differs from the proposal binding")

    review = _mapping(proposal.get("input_review"), "timer proposal.input_review")
    roi = _sequence(review.get("timer_roi_normalized_xywh"), "timer ROI")
    bounds = _sequence(review.get("wall_clock_bounds_s"), "wall-clock bounds")
    if len(roi) != 4 or len(bounds) != 2:
        raise ValueError("timer proposal review geometry is incomplete")
    context = TimerReviewContext(
        video_id=str(review.get("video_id", "")),
        source_sha256=str(review.get("source_sha256", "")),
        timer_roi_normalized_xywh=tuple(float(value) for value in roi),  # type: ignore[arg-type]
        timer_roi_evidence_reviewed=review.get("timer_roi_evidence_reviewed") is True,
        wall_clock_bounds_s=tuple(float(value) for value in bounds),  # type: ignore[arg-type]
        bounds_evidence_reviewed=review.get("bounds_evidence_reviewed") is True,
        reviewer_identity=str(review.get("reviewer_identity", "")),
        reviewer_kind=str(review.get("reviewer_kind", "")),
        nominal_loadless_duration_s=(
            float(review["nominal_loadless_duration_s"])
            if review.get("nominal_loadless_duration_s") is not None
            else None
        ),
        evidence=tuple(str(value) for value in _sequence(review.get("evidence"), "review evidence")),
    )
    policy = TimerActivityPolicy(**_mapping(proposal.get("policy"), "timer proposal.policy"))
    # max_diagnostic_runs only bounds report serialization; raising it cannot
    # change thresholding, presence, bridging, or candidate segmentation.
    complete_policy = replace(policy, max_diagnostic_runs=max(policy.max_diagnostic_runs, frames))
    rebuilt = segment_timer_activity(
        arrays["pts_s"],
        arrays["change_score"],
        context,
        complete_policy,
        bright_mask_mean=arrays["bright_mask_mean"],
        dark_mask_mean=arrays["dark_mask_mean"],
    )
    original_quality = _mapping(
        proposal.get("proposal_quality"), "timer proposal.proposal_quality"
    )
    rebuilt_quality = _mapping(
        rebuilt.get("proposal_quality"), "rebuilt timer proposal.proposal_quality"
    )
    for field in ("candidate_range_count", "candidate_seconds"):
        original = _finite(original_quality.get(field), f"proposal_quality.{field}")
        regenerated = _finite(rebuilt_quality.get(field), f"rebuilt proposal_quality.{field}")
        if not math.isclose(original, regenerated, rel_tol=0.0, abs_tol=2e-5):
            raise ValueError(f"rebuilt timer segmentation changed {field}")
    candidates = _mapping(
        _mapping(rebuilt.get("activity"), "rebuilt timer proposal.activity").get(
            "candidate_ranges_before_gates"
        ),
        "rebuilt candidate ranges",
    )
    if candidates.get("truncated") is not False:
        raise AssertionError("full-trace candidate regeneration unexpectedly truncated")
    return candidates


def materialize_ai_boundaries(
    proposal_path: str | Path,
    output_path: str | Path,
    *,
    reviewer_identity: str,
    trace_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write fail-closed AI-only boundaries from one signal-valid proposal."""

    proposal_file = Path(proposal_path)
    output_file = Path(output_path)
    reviewer = reviewer_identity.strip()
    if not reviewer:
        raise ValueError("reviewer_identity is required")
    if output_file.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output_file}")

    proposal = _read_json(proposal_file)
    if proposal.get("format_version") != TIMER_ACTIVITY_VERSION:
        raise ValueError(f"timer proposal must use {TIMER_ACTIVITY_VERSION}")
    if proposal.get("status") != "abstained":
        raise ValueError("only an abstained AI diagnostic may use this provisional path")
    if proposal.get("auto_admitted") is not False:
        raise ValueError("timer proposal must explicitly remain non-admitted")
    if proposal.get("signal_quality_gates_passed") is not True:
        raise ValueError("timer proposal signal-quality gates did not pass")
    if proposal.get("review_provenance_gate_passed") is not False:
        raise ValueError("reviewed proposals must use the normal human boundary path")

    review = _mapping(proposal.get("input_review"), "timer proposal.input_review")
    if review.get("reviewer_kind") != "ai_agent" or review.get("human_reviewed") is not False:
        raise ValueError("timer proposal is not explicitly AI-only")
    video_id = str(review.get("video_id", "")).strip()
    source_sha256 = str(review.get("source_sha256", "")).strip().lower()
    if proposal.get("video_id") != video_id:
        raise ValueError("timer proposal and input review name different videos")
    trace = _mapping(proposal.get("trace_binding"), "timer proposal.trace_binding")
    if trace.get("source_sha256") != source_sha256:
        raise ValueError("timer proposal trace and review name different sources")
    wall = _sequence(review.get("wall_clock_bounds_s"), "wall_clock_bounds_s")
    if len(wall) != 2:
        raise ValueError("wall_clock_bounds_s must contain start and end")
    wall_start, wall_end = (_finite(value, "wall_clock_bounds_s") for value in wall)

    activity = _mapping(proposal.get("activity"), "timer proposal.activity")
    candidates = _mapping(
        activity.get("candidate_ranges_before_gates"),
        "timer proposal.activity.candidate_ranges_before_gates",
    )
    if candidates.get("truncated") is not False:
        if trace_path is None:
            raise ValueError(
                "candidate ranges are truncated; the hash-bound timer trace is required"
            )
        candidates = _complete_candidate_rows_from_trace(proposal, Path(trace_path))
    rows = _sequence(candidates.get("rows"), "candidate ranges.rows")
    total = candidates.get("total")
    if isinstance(total, bool) or not isinstance(total, int) or total != len(rows):
        raise ValueError("candidate range total does not match complete row count")
    if not rows:
        raise ValueError("timer proposal contains no candidate ranges")

    ranges: list[list[float]] = []
    candidate_seconds = 0.0
    for index, value in enumerate(rows):
        row = _mapping(value, f"candidate ranges.rows[{index}]")
        range_s = _sequence(row.get("range_s"), f"candidate ranges.rows[{index}].range_s")
        if len(range_s) != 2:
            raise ValueError(f"candidate ranges.rows[{index}].range_s must have two values")
        start, end = (_finite(item, f"candidate ranges.rows[{index}].range_s") for item in range_s)
        duration = _finite(row.get("duration_s"), f"candidate ranges.rows[{index}].duration_s")
        frames = row.get("frames")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
            raise ValueError(f"candidate ranges.rows[{index}].frames must be positive")
        if not math.isclose(duration, end - start, rel_tol=0.0, abs_tol=2e-6):
            raise ValueError(f"candidate ranges.rows[{index}] duration is inconsistent")
        ranges.append([start, end])
        candidate_seconds += end - start

    quality = _mapping(proposal.get("proposal_quality"), "timer proposal.proposal_quality")
    if quality.get("check") != "passed":
        raise ValueError("timer proposal quality summary did not pass")
    if quality.get("candidate_range_count") != len(ranges):
        raise ValueError("proposal-quality range count differs from candidate rows")
    quality_seconds = _finite(
        quality.get("candidate_seconds"), "proposal_quality.candidate_seconds"
    )
    if not math.isclose(quality_seconds, candidate_seconds, rel_tol=0.0, abs_tol=2e-5):
        raise ValueError("proposal-quality seconds differ from candidate rows")

    proposal_hash = sha256_file(proposal_file)
    boundaries = WildBoundaries.from_dict({
        "format_version": "madeleine.wild-boundaries.v2",
        "video_id": video_id,
        "source_sha256": source_sha256,
        "wall_clock_range_s": [wall_start, wall_end],
        "allowed_ranges_s": ranges,
        "human_reviewed": False,
        "reviewer": reviewer,
        "reviewer_kind": "ai_agent",
        "evidence": [
            f"timer_proposal={proposal_file.name}",
            f"timer_proposal_sha256={proposal_hash}",
            f"source_trace_sha256={trace.get('trace_npz_sha256')}",
        ],
        "notes": (
            "AI-only provisional decode ranges materialized from a signal-valid "
            "timer diagnostic. Not reviewed for training admission."
        ),
    })
    payload = (json.dumps(boundaries.to_dict(), indent=2) + "\n").encode("utf-8")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output_file.unlink(missing_ok=True)
        raise
    return boundaries.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timer-proposal", type=Path, required=True)
    parser.add_argument("--timer-trace", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_ai_boundaries(
        args.timer_proposal,
        args.out,
        reviewer_identity=args.reviewer,
        trace_path=args.timer_trace,
    )
    print(json.dumps({
        "video_id": result["video_id"],
        "human_reviewed": result["human_reviewed"],
        "reviewer_kind": result["reviewer_kind"],
        "allowed_ranges": len(result["allowed_ranges_s"]),
        "output": str(args.out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
