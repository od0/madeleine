from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from harvest.timer_activity import (
    TimerActivityPolicy,
    TimerReviewContext,
    segment_timer_activity,
    segment_timer_roi_observations,
    timer_presence_scores,
    write_timer_activity_diagnostics,
)


FPS = 60.0


def _pts(frames: int) -> np.ndarray:
    return np.arange(frames, dtype=np.float64) / FPS


def _context(frames: int, *, roi: bool = True) -> TimerReviewContext:
    return TimerReviewContext(
        video_id="timer_fixture",
        source_sha256="a" * 64,
        timer_roi_normalized_xywh=(0.8, 0.02, 0.18, 0.08) if roi else None,
        timer_roi_evidence_reviewed=roi,
        wall_clock_bounds_s=(0.0, frames / FPS),
        bounds_evidence_reviewed=True,
        reviewer_identity="Synthetic Human Reviewer",
        reviewer_kind="human",
        evidence=("synthetic-reviewed-timer-roi.png",),
    )


def _advancing_timer(frames: int) -> np.ndarray:
    observations = np.zeros((frames, 4, 8), dtype=np.uint8)
    for index in range(frames):
        # A timer glyph remains bright on a dark background and advances every
        # other decoded frame, yielding real quiet/changing score populations.
        phase = (index // 2) % observations.shape[2]
        observations[index, 0, phase] = 255
        observations[index, 1, (phase + 3) % observations.shape[2]] = 255
    return observations


def _moving_scene_without_timer(frames: int) -> np.ndarray:
    observations = np.empty((frames, 4, 8), dtype=np.uint8)
    observations[::2] = 64
    observations[1::2] = 192
    return observations


def _uniformly_moving_timer(frames: int) -> np.ndarray:
    observations = np.zeros((frames, 4, 8), dtype=np.uint8)
    for index in range(frames):
        phase = index % observations.shape[2]
        observations[index, 0, phase] = 255
        observations[index, 1, (phase + 3) % observations.shape[2]] = 255
    return observations


def test_advancing_timer_proposes_one_review_required_half_open_range(
    tmp_path,
) -> None:
    frames = 301
    report = segment_timer_roi_observations(
        _pts(frames), _advancing_timer(frames), _context(frames)
    )
    assert report["automatic_gates_passed"]
    assert report["status"] == "review_required"
    assert report["auto_admitted"] is False
    assert report["requires_human_review"] is True
    [(start, end)] = report["suggested_allowed_ranges_s"]
    assert start == pytest.approx(2 / FPS)
    assert end == pytest.approx(frames / FPS)
    assert report["threshold"]["bimodal"] is True
    assert report["threshold"]["low_population"] > 0
    assert report["threshold"]["high_population"] > 0
    assert report["presence"]["present_fraction"] == 1.0

    path = write_timer_activity_diagnostics(report, tmp_path / "timer.json")
    serialized = json.loads(path.read_text())
    assert serialized["suggested_allowed_ranges_s"] == report["suggested_allowed_ranges_s"]


def test_short_hitstop_timer_freeze_is_bridged_but_not_split() -> None:
    frames = 600
    observations = _advancing_timer(frames)
    # [201,204) is exactly three frozen frame intervals.  Frame 204 resumes
    # with the unmodified timer, so half-open PTS semantics are unambiguous.
    observations[201:204] = observations[200]
    report = segment_timer_roi_observations(
        _pts(frames), observations, _context(frames)
    )
    assert report["automatic_gates_passed"]
    assert len(report["suggested_allowed_ranges_s"]) == 1
    bridges = report["activity"]["bridged_short_gaps"]
    freeze = next(row for row in bridges["rows"] if row["frames"] == 3)
    assert freeze["range_s"] == pytest.approx([201 / FPS, 204 / FPS])
    assert freeze["duration_s"] == pytest.approx(3 / FPS)


def test_long_frozen_timer_load_or_menu_remains_excluded() -> None:
    frames = 1_200
    observations = _advancing_timer(frames)
    observations[300:720] = observations[299]
    report = segment_timer_roi_observations(
        _pts(frames), observations, _context(frames)
    )
    assert report["automatic_gates_passed"]
    assert len(report["suggested_allowed_ranges_s"]) == 2
    first, second = report["suggested_allowed_ranges_s"]
    assert first[1] <= 300 / FPS + 1 / FPS
    assert second[0] >= 720 / FPS
    inactive = report["activity"]["long_inactive_ranges"]
    assert max(row["duration_s"] for row in inactive["rows"]) > 6.9


def test_moving_scene_without_timer_fails_presence_before_motion() -> None:
    frames = 300
    observations = _moving_scene_without_timer(frames)
    report = segment_timer_roi_observations(
        _pts(frames), observations, _context(frames)
    )
    assert report["status"] == "abstained"
    assert report["suggested_allowed_ranges_s"] == []
    assert report["presence"]["present_frames"] == 0
    assert report["threshold"]["eligible_samples"] == 0
    assert max(report["diagnostic_trace"]["change_score"]) > 0
    assert set(report["diagnostic_trace"]["bright_mask_mean"]) == {0.0}
    assert set(report["diagnostic_trace"]["dark_mask_mean"]) == {0.0}


def test_short_timer_presence_dropout_is_bridged() -> None:
    frames = 600
    observations = _advancing_timer(frames)
    observations[121:141] = 128  # absent for [121/60, 141/60), under 0.5 s
    report = segment_timer_roi_observations(
        _pts(frames), observations, _context(frames)
    )
    assert report["automatic_gates_passed"]
    assert len(report["suggested_allowed_ranges_s"]) == 1
    [absence] = report["presence"]["absent_ranges"]["rows"]
    assert absence["range_s"] == pytest.approx([121 / FPS, 141 / FPS])
    dropout_bridges = [
        row
        for row in report["activity"]["bridged_short_gaps"]["rows"]
        if row["absent_timer_frames"]
    ]
    assert len(dropout_bridges) == 1
    assert dropout_bridges[0]["absent_timer_frames"] == 20
    assert dropout_bridges[0]["duration_s"] <= 0.5


def test_long_timer_absence_remains_excluded() -> None:
    frames = 1_200
    observations = _advancing_timer(frames)
    observations[300:720] = 128
    report = segment_timer_roi_observations(
        _pts(frames), observations, _context(frames)
    )
    assert report["automatic_gates_passed"]
    assert len(report["suggested_allowed_ranges_s"]) == 2
    [absence] = report["presence"]["absent_ranges"]["rows"]
    assert absence["range_s"] == pytest.approx([300 / FPS, 720 / FPS])
    long_absence = max(
        report["activity"]["long_inactive_ranges"]["rows"],
        key=lambda row: row["absent_timer_frames"],
    )
    assert long_absence["absent_timer_frames"] == 420
    assert long_absence["duration_s"] > 7.0


def test_activity_island_under_two_seconds_is_pruned_conservatively() -> None:
    frames = 90
    report = segment_timer_roi_observations(
        _pts(frames), _advancing_timer(frames), _context(frames)
    )
    assert report["threshold"]["bimodal"] is True
    assert report["status"] == "abstained"
    assert report["suggested_allowed_ranges_s"] == []
    dropped = report["activity"]["dropped_short_activity_islands"]
    assert dropped["total"] == 1
    assert dropped["rows"][0]["duration_s"] < 2.0


def _scalar_activity_report(
    duration_s: int,
    active_ranges_s: list[tuple[int, int]],
    *,
    nominal_loadless_duration_s: float | None = None,
) -> dict:
    frames = int(duration_s * FPS)
    scores = np.full(frames, 0.001, dtype=np.float64)
    for start_s, end_s in active_ranges_s:
        scores[int(start_s * FPS):int(end_s * FPS)] = 0.1
    context = replace(
        _context(frames),
        nominal_loadless_duration_s=nominal_loadless_duration_s,
    )
    return segment_timer_activity(
        _pts(frames),
        scores,
        context,
        bright_mask_mean=np.full(frames, 20.0),
        dark_mask_mean=np.full(frames, 200.0),
    )


def test_low_coverage_candidate_ranges_fail_closed_but_keep_diagnostics() -> None:
    report = _scalar_activity_report(120, [(10, 30)])
    quality = report["proposal_quality"]
    assert quality["candidate_coverage_fraction"] == pytest.approx(1 / 6)
    assert quality["coverage_check"] == "failed"
    assert quality["segment_shape_check"] == "not_evaluated"
    assert not report["automatic_gates_passed"]
    assert report["suggested_allowed_ranges_s"] == []
    assert report["activity"]["candidate_range_count_before_gates"] == 1
    assert report["activity"]["candidate_ranges_before_gates"]["total"] == 1
    assert any("covers only" in reason for reason in report["failure_reasons"])


def test_short_segment_shape_fails_closed_even_with_high_coverage() -> None:
    ranges = [(start, start + 3) for start in range(0, 600, 4)]
    report = _scalar_activity_report(600, ranges)
    quality = report["proposal_quality"]
    assert quality["candidate_coverage_fraction"] > 0.70
    assert quality["coverage_check"] == "passed"
    assert quality["candidate_range_count"] == 150
    assert quality["candidate_ranges_per_hour"] == pytest.approx(900.0)
    assert quality["median_candidate_range_seconds"] == pytest.approx(3.0)
    assert quality["p90_candidate_range_seconds"] == pytest.approx(3.0)
    assert quality["segment_shape_check"] == "failed"
    assert not report["automatic_gates_passed"]
    assert report["suggested_allowed_ranges_s"] == []
    assert any("implausibly fragmented" in reason for reason in report["failure_reasons"])


def test_high_range_count_can_pass_when_coverage_and_segment_shape_are_good() -> None:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for index in range(60):
        duration = 10 if index < 48 else 25
        ranges.append((cursor, cursor + duration))
        cursor += duration + 2
    assert cursor == 900
    report = _scalar_activity_report(900, ranges)
    quality = report["proposal_quality"]
    assert quality["candidate_range_count"] == 60
    assert quality["candidate_ranges_per_hour"] == pytest.approx(240.0)
    assert quality["candidate_coverage_fraction"] > 0.85
    assert quality["median_candidate_range_seconds"] == pytest.approx(10.0)
    assert quality["p90_candidate_range_seconds"] == pytest.approx(25.0)
    assert quality["segment_shape_check"] == "passed"
    assert report["signal_quality_gates_passed"] is True
    assert report["automatic_gates_passed"] is True


def test_nominal_loadless_coverage_is_checked_when_provenance_supplies_it() -> None:
    report = _scalar_activity_report(
        600,
        [(0, 240)],
        nominal_loadless_duration_s=500.0,
    )
    quality = report["proposal_quality"]
    assert quality["envelope_coverage_check"] == "passed"
    assert quality["candidate_nominal_fraction"] == pytest.approx(0.48, abs=1e-4)
    assert quality["nominal_coverage_check"] == "failed"
    assert quality["segment_shape_check"] == "passed"
    assert not report["automatic_gates_passed"]
    assert any("nominal loadless" in reason for reason in report["failure_reasons"])


def test_ai_review_keeps_signal_diagnostics_but_cannot_pass_review_gate() -> None:
    frames = 301
    context = replace(
        _context(frames),
        reviewer_identity="OpenAI Codex visual draft",
        reviewer_kind="ai_agent",
    )
    report = segment_timer_roi_observations(
        _pts(frames), _advancing_timer(frames), context
    )
    assert report["signal_quality_gates_passed"] is True
    assert report["review_provenance_gate_passed"] is False
    assert report["automatic_gates_passed"] is False
    assert report["status"] == "abstained"
    assert report["suggested_allowed_ranges_s"] == []
    assert report["activity"]["candidate_range_count_before_gates"] == 1
    assert report["input_review"]["human_reviewed"] is False
    assert any("lack review by a human" in reason for reason in report["failure_reasons"])


def test_presence_mask_defaults_match_measured_wild20_calibration() -> None:
    observations = _advancing_timer(4)
    bright, dark = timer_presence_scores(observations)
    assert np.all(bright >= 10.0)
    assert np.all(dark >= 100.0)
    policy = TimerActivityPolicy()
    assert policy.min_bright_mask_mean == 10.0
    assert policy.min_dark_mask_mean == 100.0
    assert policy.max_bridge_s == 0.5
    assert policy.min_allowed_s == 2.0


def test_ambiguous_unimodal_motion_scores_abstain_with_populations_reported() -> None:
    frames = 300
    report = segment_timer_roi_observations(
        _pts(frames), _uniformly_moving_timer(frames), _context(frames)
    )
    threshold = report["threshold"]
    assert threshold["check"] == "failed"
    assert threshold["bimodal"] is False
    assert threshold["low_population"] == 0
    assert threshold["high_population"] == frames - 1
    assert report["activity"]["candidate_range_count_before_gates"] == 1
    assert report["status"] == "abstained"
    assert report["suggested_allowed_ranges_s"] == []
    assert any("quiet and changing" in reason for reason in report["failure_reasons"])


def test_missing_or_unreviewed_timer_roi_abstains() -> None:
    frames = 300
    report = segment_timer_roi_observations(
        _pts(frames), None, _context(frames, roi=False)
    )
    assert report["status"] == "abstained"
    assert not report["automatic_gates_passed"]
    assert report["suggested_allowed_ranges_s"] == []
    assert any("timer ROI" in reason for reason in report["failure_reasons"])


@pytest.mark.parametrize("kind", ["large_gap", "vfr"])
def test_pts_gap_or_vfr_abstains_without_suggesting_ranges(kind: str) -> None:
    frames = 600
    pts = _pts(frames)
    if kind == "large_gap":
        pts[300:] += 0.25
    else:
        intervals = np.where(np.arange(frames - 1) % 2, 1 / 50, 1 / 60)
        pts = np.r_[0.0, np.cumsum(intervals)]
    report = segment_timer_roi_observations(
        pts, _advancing_timer(frames), _context(frames)
    )
    assert report["status"] == "abstained"
    assert not report["automatic_gates_passed"]
    assert report["suggested_allowed_ranges_s"] == []
    assert any(
        marker in " ".join(report["failure_reasons"])
        for marker in ("gap", "VFR")
    )


def test_millisecond_quantized_60hz_pts_use_span_fps_not_median_fps() -> None:
    frames = 601
    intervals = np.resize(np.asarray([0.017, 0.017, 0.016]), frames - 1)
    pts = np.r_[0.0, np.cumsum(intervals)]
    context = TimerReviewContext(
        video_id="quantized_timer_fixture",
        source_sha256="b" * 64,
        timer_roi_normalized_xywh=(0.8, 0.02, 0.18, 0.08),
        timer_roi_evidence_reviewed=True,
        wall_clock_bounds_s=(0.0, float(pts[-1] + 0.017)),
        bounds_evidence_reviewed=True,
        reviewer_identity="Synthetic Human Reviewer",
        reviewer_kind="human",
        evidence=("reviewed-quantized-timer.png",),
    )
    report = segment_timer_roi_observations(
        pts, _advancing_timer(frames), context
    )
    cadence = report["pts"]
    assert report["automatic_gates_passed"]
    assert cadence["span_effective_fps"] == pytest.approx(60.0)
    assert cadence["median_interval_fps"] == pytest.approx(1 / 0.017)
    assert cadence["vfr_ratio_p99_p01"] > 1.05
    assert cadence["quantization_adjusted_vfr_ratio_p99_p01"] == pytest.approx(1.0)
