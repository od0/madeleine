"""Re-segment a hash-bound AI timer trace under an explicit policy ablation.

This tool is intentionally limited to AI-reviewed diagnostics.  It reuses the
immutable scalar trace, records every policy override and the parent proposal
hash, and can never create reviewed boundaries or admit training data.  It is
useful when a timer style violates a generic presence assumption without
paying to decode the source video again.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
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


OVERRIDABLE_POLICY_FIELDS = frozenset({
    "min_bright_mask_mean",
    "min_dark_mask_mean",
    "max_bridge_s",
})
REQUIRED_TRACE_FIELDS = frozenset({
    "pts_s",
    "change_score",
    "bright_mask_mean",
    "dark_mask_mean",
})


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path} must be a non-symlink regular file")

    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON contains non-finite number {value}")

    try:
        return _mapping(
            json.loads(path.read_text(), parse_constant=reject_constant),
            "timer proposal",
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("timer proposal is not valid UTF-8 JSON") from exc


def _review_context(proposal: dict[str, Any]) -> TimerReviewContext:
    review = _mapping(proposal.get("input_review"), "input_review")
    roi = _sequence(review.get("timer_roi_normalized_xywh"), "timer ROI")
    bounds = _sequence(review.get("wall_clock_bounds_s"), "wall-clock bounds")
    evidence = _sequence(review.get("evidence"), "review evidence")
    if len(roi) != 4 or len(bounds) != 2:
        raise ValueError("timer proposal review geometry is incomplete")
    return TimerReviewContext(
        video_id=str(review.get("video_id", "")),
        source_sha256=str(review.get("source_sha256", "")),
        timer_roi_normalized_xywh=tuple(float(value) for value in roi),  # type: ignore[arg-type]
        timer_roi_evidence_reviewed=(
            review.get("timer_roi_evidence_reviewed") is True
        ),
        wall_clock_bounds_s=tuple(float(value) for value in bounds),  # type: ignore[arg-type]
        bounds_evidence_reviewed=review.get("bounds_evidence_reviewed") is True,
        reviewer_identity=str(review.get("reviewer_identity", "")),
        reviewer_kind=str(review.get("reviewer_kind", "")),
        nominal_loadless_duration_s=(
            float(review["nominal_loadless_duration_s"])
            if review.get("nominal_loadless_duration_s") is not None
            else None
        ),
        evidence=tuple(str(value) for value in evidence),
    )


def _trace_arrays(
    trace_path: Path,
    binding: dict[str, Any],
) -> dict[str, np.ndarray]:
    if trace_path.is_symlink() or not trace_path.is_file():
        raise ValueError("timer trace must be a non-symlink regular file")
    actual_hash = sha256_file(trace_path)
    if binding.get("trace_npz_sha256") != actual_hash:
        raise ValueError("timer trace hash differs from proposal binding")
    with np.load(trace_path, allow_pickle=False) as trace:
        if not REQUIRED_TRACE_FIELDS.issubset(trace.files):
            raise ValueError("timer trace is missing required scalar arrays")
        arrays = {
            name: np.asarray(trace[name]) for name in REQUIRED_TRACE_FIELDS
        }
    frames = arrays["pts_s"].size
    if frames <= 1 or any(
        values.ndim != 1 or values.size != frames for values in arrays.values()
    ):
        raise ValueError("timer trace scalar arrays must be aligned vectors")
    if binding.get("frames") != frames:
        raise ValueError("timer trace frame count differs from proposal binding")
    return arrays


def resegment_timer_trace(
    proposal_path: str | Path,
    trace_path: str | Path,
    output_path: str | Path,
    *,
    policy_overrides: dict[str, float],
    rationale: str,
) -> dict[str, Any]:
    """Write one provenance-bound, non-admitting timer-policy ablation."""

    proposal_file = Path(proposal_path)
    trace_file = Path(trace_path)
    output_file = Path(output_path)
    if output_file.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output_file}")
    if not rationale.strip():
        raise ValueError("a non-empty ablation rationale is required")
    if not policy_overrides:
        raise ValueError("at least one timer policy override is required")
    unknown = set(policy_overrides) - OVERRIDABLE_POLICY_FIELDS
    if unknown:
        raise ValueError(
            "unsupported timer policy override(s): " + ", ".join(sorted(unknown))
        )

    proposal = _read_json(proposal_file)
    if proposal.get("format_version") != TIMER_ACTIVITY_VERSION:
        raise ValueError(f"timer proposal must use {TIMER_ACTIVITY_VERSION}")
    if proposal.get("auto_admitted") is not False:
        raise ValueError("parent timer proposal must explicitly remain non-admitted")
    context = _review_context(proposal)
    if context.reviewer_kind != "ai_agent" or context.human_reviewed:
        raise ValueError("policy ablation is restricted to AI-only timer proposals")

    binding = dict(_mapping(proposal.get("trace_binding"), "trace_binding"))
    if binding.get("source_sha256") != context.source_sha256:
        raise ValueError("timer trace and review name different sources")
    arrays = _trace_arrays(trace_file, binding)
    base_policy = TimerActivityPolicy(
        **_mapping(proposal.get("policy"), "timer policy")
    )
    normalized_overrides = {
        name: float(value) for name, value in policy_overrides.items()
    }
    policy = replace(base_policy, **normalized_overrides)
    policy.validate()

    result = segment_timer_activity(
        arrays["pts_s"],
        arrays["change_score"],
        context,
        policy,
        bright_mask_mean=arrays["bright_mask_mean"],
        dark_mask_mean=arrays["dark_mask_mean"],
    )
    if result.get("auto_admitted") is not False:
        raise AssertionError("timer segmentation violated no-auto-admission contract")
    if result.get("review_provenance_gate_passed") is not False:
        raise AssertionError("AI-only ablation unexpectedly passed review provenance")
    result["trace_binding"] = binding
    result["policy_ablation"] = {
        "format_version": "madeleine.wild-timer-policy-ablation.v1",
        "parent_proposal_file": proposal_file.name,
        "parent_proposal_sha256": sha256_file(proposal_file),
        "trace_npz_sha256": binding["trace_npz_sha256"],
        "overrides": dict(sorted(normalized_overrides.items())),
        "rationale": rationale.strip(),
        "human_reviewed": False,
        "admission_effect": "none; diagnostic/provisional use only",
    }

    payload = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
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
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timer-proposal", type=Path, required=True)
    parser.add_argument("--timer-trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--min-bright-mask-mean", type=float)
    parser.add_argument("--min-dark-mask-mean", type=float)
    parser.add_argument("--max-bridge-s", type=float)
    args = parser.parse_args()
    overrides = {
        name: value
        for name, value in {
            "min_bright_mask_mean": args.min_bright_mask_mean,
            "min_dark_mask_mean": args.min_dark_mask_mean,
            "max_bridge_s": args.max_bridge_s,
        }.items()
        if value is not None
    }
    result = resegment_timer_trace(
        args.timer_proposal,
        args.timer_trace,
        args.out,
        policy_overrides=overrides,
        rationale=args.rationale,
    )
    quality = _mapping(result.get("proposal_quality"), "proposal quality")
    print(json.dumps({
        "video_id": result["video_id"],
        "status": result["status"],
        "signal_quality_gates_passed": result["signal_quality_gates_passed"],
        "candidate_ranges": quality["candidate_range_count"],
        "candidate_hours": quality["candidate_seconds"] / 3600.0,
        "human_reviewed": False,
        "output": str(args.out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
