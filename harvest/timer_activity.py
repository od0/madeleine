"""Evidence-first activity ranges from a reviewed on-screen run timer.

Speedrun.com reports loadless time, so it cannot define wall-clock gameplay
boundaries in a recording.  A reviewed official-timer ROI can: while the game
clock advances its pixels change, while loads, menus, and the finished screen
leave it absent or frozen.  This helper turns a bounded per-frame ROI trace
into *suggested* half-open PTS ranges.

The suggestion is deliberately not an admission artifact.  A reviewer must
first identify the timer ROI and wall-clock envelope, and must subsequently
review the suggested ranges against the source video before copying them into
``WildBoundaries``.  PTS irregularity, missing review evidence, or excessive
fragmentation causes abstention rather than guessed ranges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


TIMER_ACTIVITY_VERSION = "madeleine.timer-activity-evidence.v3"
REVIEWER_KINDS = ("human", "human_with_ai_assistance", "ai_agent")
HUMAN_REVIEWER_KINDS = frozenset(("human", "human_with_ai_assistance"))
Range = tuple[float, float]
Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class TimerActivityPolicy:
    """Fail-closed limits for timer-derived range suggestions."""

    max_frames: int = 2_000_000
    max_roi_pixels_per_frame: int = 65_536
    min_effective_fps: float = 50.0
    max_effective_fps: float = 61.0
    max_vfr_ratio_p99_p01: float = 1.05
    pts_interval_quantization_tolerance_s: float = 0.0011
    max_gap_s: float = 0.100
    max_gap_multiple: float = 2.5
    bright_pixel_threshold: float = 200.0
    dark_pixel_threshold: float = 32.0
    min_bright_mask_mean: float = 10.0
    min_dark_mask_mean: float = 100.0
    min_change_score: float = 1e-3
    threshold_fraction: float = 0.35
    min_bimodal_fraction: float = 0.05
    min_bimodal_samples: int = 3
    max_bridge_s: float = 0.500
    min_allowed_s: float = 2.0
    min_candidate_coverage_fraction: float = 0.25
    min_candidate_nominal_fraction: float = 0.50
    segment_shape_min_envelope_s: float = 300.0
    min_median_candidate_range_s: float = 5.0
    min_p90_candidate_range_s: float = 15.0
    max_diagnostic_runs: int = 256
    max_trace_points: int = 512

    def validate(self) -> None:
        if self.max_frames <= 1 or self.max_roi_pixels_per_frame <= 0:
            raise ValueError("frame and ROI limits must be positive")
        if not 0 < self.min_effective_fps <= self.max_effective_fps:
            raise ValueError("effective FPS bounds are invalid")
        if self.max_vfr_ratio_p99_p01 < 1.0:
            raise ValueError("VFR ratio must be at least one")
        if self.pts_interval_quantization_tolerance_s < 0:
            raise ValueError("PTS quantization tolerance must be non-negative")
        if self.max_gap_s <= 0 or self.max_gap_multiple <= 1:
            raise ValueError("PTS gap limits are invalid")
        if not 0 <= self.dark_pixel_threshold < self.bright_pixel_threshold <= 255:
            raise ValueError("timer presence pixel thresholds are invalid")
        if not 0 <= self.min_bright_mask_mean <= 255:
            raise ValueError("bright-mask mean threshold is invalid")
        if not 0 <= self.min_dark_mask_mean <= 255:
            raise ValueError("dark-mask mean threshold is invalid")
        if self.min_change_score < 0 or not 0 < self.threshold_fraction < 1:
            raise ValueError("activity threshold policy is invalid")
        if not 0 < self.min_bimodal_fraction < 0.5 or self.min_bimodal_samples <= 0:
            raise ValueError("bimodality policy is invalid")
        if self.max_bridge_s < 0 or self.min_allowed_s <= 0:
            raise ValueError("range duration limits are invalid")
        if not 0 < self.min_candidate_coverage_fraction <= 1:
            raise ValueError("candidate coverage floor must lie in (0,1]")
        if not 0 < self.min_candidate_nominal_fraction <= 1:
            raise ValueError("nominal-duration coverage floor must lie in (0,1]")
        if self.segment_shape_min_envelope_s <= 0:
            raise ValueError("segment-shape evaluation duration must be positive")
        if min(
            self.min_median_candidate_range_s,
            self.min_p90_candidate_range_s,
        ) <= 0:
            raise ValueError("candidate segment-shape floors must be positive")
        if self.min_p90_candidate_range_s < self.min_median_candidate_range_s:
            raise ValueError("candidate p90 floor must be at least the median floor")
        if min(self.max_diagnostic_runs, self.max_trace_points) <= 0:
            raise ValueError("diagnostic limits must be positive")


@dataclass(frozen=True)
class TimerReviewContext:
    """Evidence provenance for a timer trace and range proposal."""

    video_id: str
    source_sha256: str
    timer_roi_normalized_xywh: Rect | None
    timer_roi_evidence_reviewed: bool
    wall_clock_bounds_s: Range | None
    bounds_evidence_reviewed: bool
    reviewer_identity: str
    reviewer_kind: str
    nominal_loadless_duration_s: float | None = None
    evidence: tuple[str, ...] = ()

    def failures(self) -> list[str]:
        failures: list[str] = []
        if not self.video_id.strip():
            failures.append("video_id is missing")
        digest = self.source_sha256.strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            failures.append("source_sha256 must be a lowercase SHA-256 digest")
        if self.timer_roi_normalized_xywh is None:
            failures.append("timer ROI is missing")
        else:
            x, y, width, height = self.timer_roi_normalized_xywh
            values = np.asarray([x, y, width, height], dtype=np.float64)
            if (
                not np.all(np.isfinite(values))
                or x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > 1.0 + 1e-9
                or y + height > 1.0 + 1e-9
            ):
                failures.append("timer ROI must be a finite normalized xywh rectangle")
        if not self.timer_roi_evidence_reviewed:
            failures.append("timer ROI evidence has not been reviewed")
        if self.wall_clock_bounds_s is None:
            failures.append("reviewed wall-clock bounds are missing")
        else:
            start, end = self.wall_clock_bounds_s
            if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
                failures.append("wall-clock bounds must be a finite positive-duration range")
        if not self.bounds_evidence_reviewed:
            failures.append("wall-clock bounds evidence has not been reviewed")
        if not str(self.reviewer_identity).strip():
            failures.append("reviewer identity is required")
        if self.reviewer_kind not in REVIEWER_KINDS:
            failures.append(f"reviewer kind must be one of {REVIEWER_KINDS}")
        if self.nominal_loadless_duration_s is not None and (
            not np.isfinite(self.nominal_loadless_duration_s)
            or self.nominal_loadless_duration_s <= 0
        ):
            failures.append("nominal loadless duration must be finite and positive")
        if not any(str(item).strip() for item in self.evidence):
            failures.append("review evidence reference is required")
        return failures

    @property
    def human_reviewed(self) -> bool:
        return self.reviewer_kind in HUMAN_REVIEWER_KINDS

    def human_review_failures(self) -> list[str]:
        if self.human_reviewed:
            return []
        return [
            "timer ROI and wall-clock bounds lack review by a human or "
            "human-with-AI-assistance"
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "source_sha256": self.source_sha256,
            "timer_roi_normalized_xywh": (
                list(self.timer_roi_normalized_xywh)
                if self.timer_roi_normalized_xywh is not None
                else None
            ),
            "timer_roi_evidence_reviewed": self.timer_roi_evidence_reviewed,
            "wall_clock_bounds_s": (
                list(self.wall_clock_bounds_s)
                if self.wall_clock_bounds_s is not None
                else None
            ),
            "bounds_evidence_reviewed": self.bounds_evidence_reviewed,
            "reviewer_identity": str(self.reviewer_identity).strip(),
            "reviewer_kind": self.reviewer_kind,
            "human_reviewed": self.human_reviewed,
            "nominal_loadless_duration_s": self.nominal_loadless_duration_s,
            "evidence": list(self.evidence),
        }


def timer_change_scores(
    observations: np.ndarray,
    *,
    max_pixels_per_frame: int = TimerActivityPolicy.max_roi_pixels_per_frame,
) -> np.ndarray:
    """Reduce bounded ROI frames to normalized consecutive-frame differences.

    Production extraction may compute this scalar trace while streaming video
    instead of retaining every ROI image.  This in-memory helper exists for
    small evidence windows and deterministic tests.
    """

    values = np.asarray(observations)
    if values.ndim < 2:
        raise ValueError("timer ROI observations must have shape [frames, ...]")
    if values.shape[0] < 2:
        raise ValueError("at least two timer ROI observations are required")
    pixels = int(np.prod(values.shape[1:]))
    if pixels <= 0 or pixels > max_pixels_per_frame:
        raise ValueError(
            f"timer ROI has {pixels} values per frame; limit is {max_pixels_per_frame}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("timer ROI observations must be numeric")
    normalized = values.astype(np.float32)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("timer ROI observations must be finite")
    if np.issubdtype(values.dtype, np.integer):
        scale = float(np.iinfo(values.dtype).max)
        normalized /= max(scale, 1.0)
    else:
        minimum, maximum = float(normalized.min()), float(normalized.max())
        if minimum < 0:
            raise ValueError("floating timer ROI observations must be non-negative")
        if maximum > 1.0 + 1e-6:
            if maximum <= 255.0 + 1e-6:
                normalized /= 255.0
            else:
                raise ValueError("floating timer ROI observations must lie in [0,1] or [0,255]")
    flattened = normalized.reshape(normalized.shape[0], -1)
    scores = np.zeros(flattened.shape[0], dtype=np.float64)
    scores[1:] = np.mean(np.abs(flattened[1:] - flattened[:-1]), axis=1)
    return scores


def timer_presence_scores(
    observations: np.ndarray,
    *,
    bright_pixel_threshold: float = 200.0,
    dark_pixel_threshold: float = 32.0,
    max_pixels_per_frame: int = TimerActivityPolicy.max_roi_pixels_per_frame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame 0--255 binary-mask means for a grayscale timer ROI.

    The raw calibration for the reviewed Wild20 layouts showed that a visible
    timer contains both a small bright glyph population and a large dark
    background population.  Requiring both prevents arbitrary moving scene
    texture beneath an absent/translucent panel from becoming timer motion.
    """

    values = np.asarray(observations)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 3:
        raise ValueError(
            "timer presence observations must be grayscale [frames, height, width]"
        )
    if values.shape[0] < 2:
        raise ValueError("at least two timer ROI observations are required")
    pixels = int(np.prod(values.shape[1:]))
    if pixels <= 0 or pixels > max_pixels_per_frame:
        raise ValueError(
            f"timer ROI has {pixels} pixels per frame; limit is {max_pixels_per_frame}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("timer ROI observations must be numeric")
    gray = values.astype(np.float32)
    if not np.all(np.isfinite(gray)):
        raise ValueError("timer ROI observations must be finite")
    if np.issubdtype(values.dtype, np.integer):
        gray *= 255.0 / max(float(np.iinfo(values.dtype).max), 1.0)
    else:
        minimum, maximum = float(gray.min()), float(gray.max())
        if minimum < 0:
            raise ValueError("floating timer ROI observations must be non-negative")
        if maximum <= 1.0 + 1e-6:
            gray *= 255.0
        elif maximum > 255.0 + 1e-6:
            raise ValueError(
                "floating timer ROI observations must lie in [0,1] or [0,255]"
            )
    bright_mask_mean = np.mean(
        (gray >= bright_pixel_threshold).astype(np.float32) * 255.0,
        axis=(1, 2),
        dtype=np.float64,
    )
    dark_mask_mean = np.mean(
        (gray <= dark_pixel_threshold).astype(np.float32) * 255.0,
        axis=(1, 2),
        dtype=np.float64,
    )
    return bright_mask_mean, dark_mask_mean


def _empty_report(
    context: TimerReviewContext,
    policy: TimerActivityPolicy,
    failures: list[str],
) -> dict[str, Any]:
    return {
        "format_version": TIMER_ACTIVITY_VERSION,
        "video_id": context.video_id,
        "source_sha256": context.source_sha256,
        "status": "abstained",
        "automatic_gates_passed": False,
        "signal_quality_gates_passed": False,
        "review_provenance_gate_passed": context.human_reviewed,
        "auto_admitted": False,
        "requires_human_review": True,
        "failure_reasons": failures,
        "input_review": context.to_dict(),
        "policy": asdict(policy),
        "range_semantics": "half-open seconds relative to first decoded video PTS",
        "suggested_allowed_ranges_s": [],
        "review_checklist": _review_checklist(),
    }


def _review_checklist() -> list[str]:
    return [
        "Confirm the normalized ROI contains only the official advancing timer.",
        "Inspect every suggested start and end against raw source frames.",
        "Confirm hitstop/noise bridges are brief gameplay freezes, not loads or menus.",
        "Confirm every long frozen or absent-timer range remains excluded.",
        "Copy ranges into WildBoundaries only after naming a human reviewer.",
    ]


def _threshold(
    scores: np.ndarray,
    policy: TimerActivityPolicy,
) -> tuple[float, dict[str, Any]]:
    q10, median, q90 = (float(value) for value in np.percentile(scores, [10, 50, 90]))
    spread = q90 - q10
    if spread <= 1e-12:
        threshold = max(policy.min_change_score, q90)
    else:
        threshold = max(
            policy.min_change_score,
            q10 + policy.threshold_fraction * spread,
        )
    low = scores[scores < threshold]
    high = scores[scores >= threshold]
    low_median = float(np.median(low)) if low.size else None
    high_median = float(np.median(high)) if high.size else None
    separation = (
        float(high_median - low_median)
        if low_median is not None and high_median is not None
        else 0.0
    )
    minimum_population = max(
        policy.min_bimodal_samples,
        int(np.ceil(policy.min_bimodal_fraction * scores.size)),
    )
    bimodal = bool(
        low.size >= minimum_population
        and high.size >= minimum_population
        and spread >= policy.min_change_score
        and separation >= policy.min_change_score
    )
    return threshold, {
        "q10": q10,
        "median": median,
        "q90": q90,
        "spread_q90_q10": spread,
        "threshold": threshold,
        "eligible_samples": int(scores.size),
        "low_population": int(low.size),
        "high_population": int(high.size),
        "low_fraction": float(low.size / scores.size),
        "high_fraction": float(high.size / scores.size),
        "low_median": low_median,
        "high_median": high_median,
        "median_separation": separation,
        "minimum_population_required": minimum_population,
        "bimodal": bimodal,
        "check": "passed" if bimodal else "failed",
    }


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def _false_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    return _true_runs(~np.asarray(mask, dtype=bool))


def _run_range(
    start: int,
    end: int,
    pts: np.ndarray,
    median_dt: float,
    bounds: Range,
) -> Range:
    range_start = max(float(bounds[0]), float(pts[start]))
    next_pts = float(pts[end]) if end < pts.size else float(pts[-1] + median_dt)
    range_end = min(float(bounds[1]), next_pts)
    return range_start, range_end


def _bounded_rows(rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    return {
        "total": len(rows),
        "truncated": len(rows) > limit,
        "rows": rows[:limit],
    }


def segment_timer_activity(
    pts_s: np.ndarray,
    change_scores: np.ndarray | None,
    context: TimerReviewContext,
    policy: TimerActivityPolicy = TimerActivityPolicy(),
    *,
    bright_mask_mean: np.ndarray | None = None,
    dark_mask_mean: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return conservative timer-advancing range suggestions and diagnostics.

    ``change_scores[i]`` is the visual difference between timer ROI frame
    ``i-1`` and frame ``i``; index zero should be zero.  The two mask-mean
    traces are the raw 0--255 means of thresholded bright and dark masks.
    Timer presence gates motion before any range is proposed.  Even a
    successful report is review-required and is never a ``WildBoundaries``
    artifact.
    """

    policy.validate()
    failures = context.failures()
    if change_scores is None:
        failures.append("timer ROI observations are missing")
    if bright_mask_mean is None:
        failures.append("bright-mask mean trace is missing")
    if dark_mask_mean is None:
        failures.append("dark-mask mean trace is missing")
    if failures:
        return _empty_report(context, policy, failures)

    pts = np.asarray(pts_s, dtype=np.float64)
    scores = np.asarray(change_scores, dtype=np.float64)
    bright = np.asarray(bright_mask_mean, dtype=np.float64)
    dark = np.asarray(dark_mask_mean, dtype=np.float64)
    shape_failures: list[str] = []
    vectors = (pts, scores, bright, dark)
    if any(values.ndim != 1 for values in vectors) or len(
        {values.size for values in vectors}
    ) != 1:
        shape_failures.append(
            "PTS, change scores, and presence traces must be equal-length vectors"
        )
    elif pts.size < 2:
        shape_failures.append("at least two PTS observations are required")
    elif pts.size > policy.max_frames:
        shape_failures.append(f"frame count {pts.size} exceeds limit {policy.max_frames}")
    elif any(not np.all(np.isfinite(values)) for values in vectors):
        shape_failures.append("PTS, change scores, and presence traces must be finite")
    elif np.any(scores < 0):
        shape_failures.append("timer change scores must be non-negative")
    elif np.any(bright < 0) or np.any(bright > 255):
        shape_failures.append("bright-mask means must lie in [0,255]")
    elif np.any(dark < 0) or np.any(dark > 255):
        shape_failures.append("dark-mask means must lie in [0,255]")
    if shape_failures:
        return _empty_report(context, policy, shape_failures)

    delta = np.diff(pts)
    if np.any(delta <= 0):
        return _empty_report(context, policy, ["PTS must be strictly increasing"])
    median_dt = float(np.median(delta))
    p01, p99 = (float(value) for value in np.percentile(delta, [1, 99]))
    median_interval_fps = 1.0 / median_dt
    span_effective_fps = float((pts.size - 1) / (pts[-1] - pts[0]))
    vfr_ratio = p99 / max(p01, 1e-12)
    adjusted_p99 = max(p01, p99 - policy.pts_interval_quantization_tolerance_s)
    adjusted_vfr_ratio = adjusted_p99 / max(p01, 1e-12)
    gap_gate = max(policy.max_gap_s, policy.max_gap_multiple * median_dt)
    large_gaps = np.flatnonzero(delta > gap_gate)
    pts_diagnostics = {
        "frames": int(pts.size),
        "first_s": float(pts[0]),
        "last_s": float(pts[-1]),
        "median_dt_s": median_dt,
        "p01_dt_s": p01,
        "p99_dt_s": p99,
        "effective_fps": span_effective_fps,
        "span_effective_fps": span_effective_fps,
        "median_interval_fps": median_interval_fps,
        "vfr_ratio_p99_p01": vfr_ratio,
        "quantization_adjusted_vfr_ratio_p99_p01": adjusted_vfr_ratio,
        "gap_gate_s": gap_gate,
        "large_gap_intervals": int(large_gaps.size),
        "largest_gap_s": float(delta.max(initial=0.0)),
    }
    pts_failures: list[str] = []
    if not policy.min_effective_fps <= span_effective_fps <= policy.max_effective_fps:
        pts_failures.append(
            f"span effective FPS {span_effective_fps:.4f} is outside "
            f"[{policy.min_effective_fps}, {policy.max_effective_fps}]"
        )
    if adjusted_vfr_ratio > policy.max_vfr_ratio_p99_p01:
        pts_failures.append(
            "quantization-adjusted PTS VFR ratio "
            f"{adjusted_vfr_ratio:.4f} exceeds {policy.max_vfr_ratio_p99_p01}"
        )
    if large_gaps.size:
        pts_failures.append(f"PTS contains {large_gaps.size} large gap interval(s)")

    assert context.wall_clock_bounds_s is not None
    bounds = context.wall_clock_bounds_s
    coverage_end = float(pts[-1] + median_dt)
    if bounds[0] < pts[0] - 1e-9 or bounds[1] > coverage_end + 1e-6:
        pts_failures.append(
            f"reviewed bounds {bounds} lie outside PTS coverage "
            f"[{pts[0]:.6f}, {coverage_end:.6f})"
        )
    if pts_failures:
        report = _empty_report(context, policy, pts_failures)
        report["pts"] = pts_diagnostics
        return report

    selected = np.flatnonzero((pts >= bounds[0]) & (pts < bounds[1]))
    if selected.size < 2 or not np.array_equal(
        selected, np.arange(selected[0], selected[-1] + 1)
    ):
        report = _empty_report(
            context, policy, ["reviewed bounds do not select a contiguous PTS interval"]
        )
        report["pts"] = pts_diagnostics
        return report

    local_pts = pts[selected]
    local_scores = scores[selected]
    local_bright = bright[selected]
    local_dark = dark[selected]
    present = (
        (local_bright >= policy.min_bright_mask_mean)
        & (local_dark >= policy.min_dark_mask_mean)
    )
    absence_rows: list[dict[str, Any]] = []
    for start, end in _false_runs(present):
        absent_start, absent_end = _run_range(
            start, end, local_pts, median_dt, bounds
        )
        absence_rows.append({
            "range_s": [absent_start, absent_end],
            "duration_s": absent_end - absent_start,
            "frames": end - start,
            "reason": "timer_presence_guard_failed",
        })

    # A visual difference is timer motion only when both endpoints contain the
    # reviewed timer.  This also prevents scene changes at a panel boundary
    # from seeding a false timer-active range.
    motion_eligible = present.copy()
    motion_eligible[0] = False
    motion_eligible[1:] &= present[:-1]
    eligible_scores = local_scores[motion_eligible]
    if eligible_scores.size:
        threshold, threshold_diagnostics = _threshold(eligible_scores, policy)
    else:
        threshold = None
        threshold_diagnostics = {
            "eligible_samples": 0,
            "threshold": None,
            "low_population": 0,
            "high_population": 0,
            "low_fraction": 0.0,
            "high_fraction": 0.0,
            "low_median": None,
            "high_median": None,
            "median_separation": 0.0,
            "minimum_population_required": policy.min_bimodal_samples,
            "bimodal": False,
            "check": "failed",
            "reason": "no consecutive timer-present frames",
        }
    raw_active = (
        motion_eligible & (local_scores >= threshold)
        if threshold is not None
        else np.zeros_like(present)
    )

    bridged = raw_active.copy()
    bridged_rows: list[dict[str, Any]] = []
    for start, end in _false_runs(raw_active):
        gap_start, gap_end = _run_range(start, end, local_pts, median_dt, bounds)
        duration = gap_end - gap_start
        is_internal = start > 0 and end < raw_active.size
        absent_frames = int(np.count_nonzero(~present[start:end]))
        if is_internal and duration <= policy.max_bridge_s + 1e-9:
            bridged[start:end] = True
            bridged_rows.append({
                "range_s": [gap_start, gap_end],
                "duration_s": duration,
                "frames": end - start,
                "absent_timer_frames": absent_frames,
                "reason": (
                    "short_internal_timer_presence_dropout"
                    if absent_frames
                    else "short_internal_timer_freeze_or_visual_noise"
                ),
            })

    long_inactive_rows: list[dict[str, Any]] = []
    for start, end in _false_runs(bridged):
        inactive_start, inactive_end = _run_range(
            start, end, local_pts, median_dt, bounds
        )
        long_inactive_rows.append({
            "range_s": [inactive_start, inactive_end],
            "duration_s": inactive_end - inactive_start,
            "frames": end - start,
            "absent_timer_frames": int(np.count_nonzero(~present[start:end])),
            "reason": "frozen_or_absent_timer_not_bridged",
        })

    candidate_suggested: list[list[float]] = []
    candidate_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    for start, end in _true_runs(bridged):
        allowed_start, allowed_end = _run_range(
            start, end, local_pts, median_dt, bounds
        )
        duration = allowed_end - allowed_start
        row = {
            "range_s": [allowed_start, allowed_end],
            "duration_s": duration,
            "frames": end - start,
        }
        if duration < policy.min_allowed_s:
            dropped_rows.append({**row, "reason": "short_activity_island"})
        else:
            candidate_suggested.append([allowed_start, allowed_end])
            candidate_rows.append(row)

    reviewed_duration_s = float(bounds[1] - bounds[0])
    candidate_seconds = float(
        sum(end - start for start, end in candidate_suggested)
    )
    candidate_coverage_fraction = candidate_seconds / reviewed_duration_s
    candidate_range_count = len(candidate_suggested)
    candidate_ranges_per_hour = candidate_range_count / (reviewed_duration_s / 3600.0)
    envelope_coverage_passed = bool(
        candidate_suggested
        and candidate_coverage_fraction >= policy.min_candidate_coverage_fraction
    )
    nominal_duration_s = context.nominal_loadless_duration_s
    candidate_nominal_fraction = (
        candidate_seconds / nominal_duration_s
        if nominal_duration_s is not None
        else None
    )
    nominal_coverage_passed = (
        candidate_nominal_fraction >= policy.min_candidate_nominal_fraction
        if candidate_nominal_fraction is not None
        else None
    )
    coverage_passed = bool(
        envelope_coverage_passed and nominal_coverage_passed is not False
    )
    candidate_durations = np.asarray(
        [end - start for start, end in candidate_suggested], dtype=np.float64
    )
    median_candidate_s = (
        float(np.median(candidate_durations))
        if candidate_durations.size
        else None
    )
    p90_candidate_s = (
        float(np.percentile(candidate_durations, 90))
        if candidate_durations.size
        else None
    )
    max_candidate_s = (
        float(np.max(candidate_durations))
        if candidate_durations.size
        else None
    )
    segment_shape_evaluated = bool(
        candidate_durations.size
        and reviewed_duration_s >= policy.segment_shape_min_envelope_s
    )
    segment_shape_passed = bool(
        not segment_shape_evaluated
        or (
            median_candidate_s is not None
            and p90_candidate_s is not None
            and median_candidate_s >= policy.min_median_candidate_range_s
            and p90_candidate_s >= policy.min_p90_candidate_range_s
        )
    )
    quality = {
        "reviewed_envelope_seconds": reviewed_duration_s,
        "candidate_seconds": candidate_seconds,
        "candidate_coverage_fraction": candidate_coverage_fraction,
        "min_candidate_coverage_fraction": policy.min_candidate_coverage_fraction,
        "envelope_coverage_check": (
            "passed" if envelope_coverage_passed else "failed"
        ),
        "nominal_loadless_duration_seconds": nominal_duration_s,
        "candidate_nominal_fraction": candidate_nominal_fraction,
        "min_candidate_nominal_fraction": policy.min_candidate_nominal_fraction,
        "nominal_coverage_check": (
            "not_available"
            if nominal_coverage_passed is None
            else "passed" if nominal_coverage_passed else "failed"
        ),
        "coverage_check": "passed" if coverage_passed else "failed",
        "candidate_range_count": candidate_range_count,
        "candidate_ranges_per_hour": candidate_ranges_per_hour,
        "median_candidate_range_seconds": median_candidate_s,
        "p90_candidate_range_seconds": p90_candidate_s,
        "max_candidate_range_seconds": max_candidate_s,
        "segment_shape_min_envelope_seconds": policy.segment_shape_min_envelope_s,
        "segment_shape_evaluated": segment_shape_evaluated,
        "min_median_candidate_range_seconds": policy.min_median_candidate_range_s,
        "min_p90_candidate_range_seconds": policy.min_p90_candidate_range_s,
        "segment_shape_check": (
            "not_evaluated"
            if not segment_shape_evaluated
            else "passed" if segment_shape_passed else "failed"
        ),
        "check": (
            "passed"
            if coverage_passed and segment_shape_passed
            else "failed"
        ),
    }

    result_failures: list[str] = context.human_review_failures()
    if not threshold_diagnostics["bimodal"]:
        result_failures.append(
            "timer-motion scores lack separated quiet and changing populations"
        )
    if not candidate_suggested:
        result_failures.append("no timer-advancing interval passed the minimum duration")
    else:
        if not envelope_coverage_passed:
            result_failures.append(
                "candidate timer activity covers only "
                f"{candidate_coverage_fraction:.4f} of the reviewed envelope; "
                f"minimum is {policy.min_candidate_coverage_fraction:.4f}"
            )
        if nominal_coverage_passed is False:
            result_failures.append(
                "candidate timer activity covers only "
                f"{candidate_nominal_fraction:.4f} of nominal loadless duration; "
                f"minimum is {policy.min_candidate_nominal_fraction:.4f}"
            )
        if not segment_shape_passed:
            result_failures.append(
                "candidate timer ranges are implausibly fragmented: "
                f"median {median_candidate_s:.3f}s and p90 {p90_candidate_s:.3f}s; "
                f"minimums are {policy.min_median_candidate_range_s:.3f}s and "
                f"{policy.min_p90_candidate_range_s:.3f}s"
            )
    suggested = candidate_suggested if not result_failures else []

    trace_count = min(policy.max_trace_points, selected.size)
    trace_indices = np.unique(
        np.linspace(0, selected.size - 1, trace_count, dtype=np.int64)
    )
    trace = {
        "sampled_points": int(trace_indices.size),
        "total_points": int(selected.size),
        "truncated": int(trace_indices.size) < int(selected.size),
        "frame_index_in_reviewed_bounds": [int(value) for value in trace_indices],
        "pts_s": [float(value) for value in local_pts[trace_indices]],
        "change_score": [float(value) for value in local_scores[trace_indices]],
        "bright_mask_mean": [float(value) for value in local_bright[trace_indices]],
        "dark_mask_mean": [float(value) for value in local_dark[trace_indices]],
        "timer_present": [bool(value) for value in present[trace_indices]],
        "motion_eligible": [bool(value) for value in motion_eligible[trace_indices]],
        "raw_active": [bool(value) for value in raw_active[trace_indices]],
        "bridged_active": [bool(value) for value in bridged[trace_indices]],
    }
    automatic_pass = not result_failures
    signal_quality_pass = bool(
        threshold_diagnostics["bimodal"]
        and candidate_suggested
        and coverage_passed
        and segment_shape_passed
    )
    return {
        "format_version": TIMER_ACTIVITY_VERSION,
        "video_id": context.video_id,
        "source_sha256": context.source_sha256,
        "status": "review_required" if automatic_pass else "abstained",
        "automatic_gates_passed": automatic_pass,
        "signal_quality_gates_passed": signal_quality_pass,
        "review_provenance_gate_passed": context.human_reviewed,
        "auto_admitted": False,
        "requires_human_review": True,
        "failure_reasons": result_failures,
        "input_review": context.to_dict(),
        "policy": asdict(policy),
        "range_semantics": "half-open seconds relative to first decoded video PTS",
        "pts": pts_diagnostics,
        "presence": {
            "bright_mask_mean_threshold": policy.min_bright_mask_mean,
            "dark_mask_mean_threshold": policy.min_dark_mask_mean,
            "present_frames": int(np.count_nonzero(present)),
            "absent_frames": int(np.count_nonzero(~present)),
            "present_fraction": float(np.mean(present)),
            "absent_ranges": _bounded_rows(
                absence_rows, policy.max_diagnostic_runs
            ),
        },
        "threshold": threshold_diagnostics,
        "proposal_quality": quality,
        "activity": {
            "frames_in_reviewed_bounds": int(selected.size),
            "motion_eligible_frames": int(np.count_nonzero(motion_eligible)),
            "raw_active_frames": int(np.count_nonzero(raw_active)),
            "bridged_active_frames": int(np.count_nonzero(bridged)),
            "suggested_seconds": float(
                sum(end - start for start, end in suggested)
            ),
            "bridged_short_gaps": _bounded_rows(
                bridged_rows, policy.max_diagnostic_runs
            ),
            "long_inactive_ranges": _bounded_rows(
                long_inactive_rows, policy.max_diagnostic_runs
            ),
            "dropped_short_activity_islands": _bounded_rows(
                dropped_rows, policy.max_diagnostic_runs
            ),
            "candidate_range_count_before_gates": candidate_range_count,
            "candidate_seconds_before_gates": candidate_seconds,
            "candidate_ranges_before_gates": _bounded_rows(
                candidate_rows, policy.max_diagnostic_runs
            ),
        },
        "suggested_allowed_ranges_s": suggested,
        "diagnostic_trace": trace,
        "review_checklist": _review_checklist(),
    }


def segment_timer_roi_observations(
    pts_s: np.ndarray,
    observations: np.ndarray | None,
    context: TimerReviewContext,
    policy: TimerActivityPolicy = TimerActivityPolicy(),
) -> dict[str, Any]:
    """Convenience wrapper for bounded in-memory ROI frame observations."""

    if observations is None:
        return segment_timer_activity(pts_s, None, context, policy)
    values = np.asarray(observations)
    if values.ndim == 0:
        return _empty_report(
            context,
            policy,
            ["timer ROI observations must have shape [frames, ...]"],
        )
    if values.shape[0] > policy.max_frames:
        return _empty_report(
            context,
            policy,
            [f"frame count {values.shape[0]} exceeds limit {policy.max_frames}"],
        )
    try:
        scores = timer_change_scores(
            values, max_pixels_per_frame=policy.max_roi_pixels_per_frame
        )
        bright, dark = timer_presence_scores(
            values,
            bright_pixel_threshold=policy.bright_pixel_threshold,
            dark_pixel_threshold=policy.dark_pixel_threshold,
            max_pixels_per_frame=policy.max_roi_pixels_per_frame,
        )
    except ValueError as exc:
        return _empty_report(context, policy, [str(exc)])
    return segment_timer_activity(
        pts_s,
        scores,
        context,
        policy,
        bright_mask_mean=bright,
        dark_mask_mean=dark,
    )


def write_timer_activity_diagnostics(report: dict[str, Any], path: str | Path) -> Path:
    """Write deterministic, finite JSON suitable for a human review packet."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return destination
