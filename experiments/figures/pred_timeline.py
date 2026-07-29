"""Shared prediction-timeline reconstruction and exhibit-window selection.

Prediction sidecars store one row per model target frame, concatenated in
stream order, with no explicit frame indices. This module places those rows
back onto the true 60 Hz engine timeline of the source session (capture drops
appear as real gaps, never bridged) and selects exhibit windows with
deterministic, stated rules. It backs `fig_piano_roll.py` and
`fig_pred_overlay_video.py`, which enumerate their concrete data sources in
their own headers.

Reconstruction contract (must match `badeline.train.contiguous_runs` and the
evaluation loader): a stream is a maximal run of strictly consecutive
`engine_frame_idx` values; a centered window of length W predicts target
`run_start + (W - 1) // 2` onward, producing `run_len - W + 1` targets per
run. The reconstruction is validated exactly against the sidecar before
anything downstream may use it.

Window-selection rules (both deterministic; candidates are enumerated in
ascending start order):

* median-onset: the earliest candidate whose all-key true-onset count equals
  the gated set's median. Event-dense but not cherry-picked.
* exhibit: restrict to candidates with onset count >= the gated set's median,
  then take the one whose micro accuracy at threshold 0.5 is nearest a stated
  target (ties break to the earliest start). Used when a clip must represent
  a run-level accuracy figure honestly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

THRESHOLD = 0.5


@dataclass(frozen=True)
class Timeline:
    """Predictions and truth on the engine grid. Slot 0 = first captured frame."""

    truth: np.ndarray            # [T,7] int8; -1 = capture drop (unknown)
    prob: np.ndarray             # [T,7] float64; NaN = no model prediction
    covered: np.ndarray          # [T] bool: slot has a prediction
    active: np.ndarray           # [T] bool: input_active (False on drops)
    onset: np.ndarray            # [T,7] bool: 0->1 with the previous slot captured
    slots: np.ndarray            # shard row -> slot
    pred_shard_pos: np.ndarray   # sidecar row -> shard row
    efi0: int                    # engine_frame_idx of slot 0


@dataclass(frozen=True)
class Candidate:
    """One eligible exhibit window."""

    start: int                   # timeline slot (or global sidecar row, stream mode)
    span: int
    coverage: float              # predicted slots / span
    activity: float              # active slots / captured slots
    micro: float                 # frame/key accuracy at THRESHOLD on scored rows
    joint: float                 # all-seven exact-match accuracy on scored rows
    onsets: int                  # all-key true onsets inside the window
    scored_frames: int           # rows entering micro/joint (predicted and active)
    stream: int = -1             # stream index (stream mode only)
    local_start: int = -1        # offset within the stream (stream mode only)


def _target_offset(window: int, window_mode: str) -> int:
    if window < 1:
        raise ValueError("window must be positive")
    if window_mode == "centered":
        return (window - 1) // 2
    if window_mode == "causal":
        return window - 1
    raise ValueError(f"unknown window_mode {window_mode!r}")


def reconstruct_timeline(
    preds: Mapping[str, np.ndarray],
    shard: Mapping[str, np.ndarray],
    *,
    window: int,
    window_mode: str = "centered",
) -> Timeline:
    """Place sidecar rows on the engine timeline and validate exactly."""

    y_true = np.asarray(preds["y_true"]).astype(bool)
    y_prob = np.asarray(preds["y_prob"]).astype(np.float64)
    lengths = np.asarray(preds["session_lengths"])
    pred_active = np.asarray(preds["input_active"]).astype(bool)

    efi = np.asarray(shard["engine_frame_idx"], dtype=np.int64)
    keys = np.asarray(shard["keys"]).astype(bool)
    active = np.asarray(shard["input_active"]).astype(bool)
    if efi.ndim != 1 or len(efi) != len(keys) or len(keys) != len(active):
        raise ValueError("shard arrays disagree about the frame count")

    delta = np.diff(efi)
    if np.any(delta < 1):
        # An engine-counter reset or duplicate index cannot be placed on a
        # single monotone timeline; such captures must be split upstream.
        raise ValueError("engine_frame_idx must be strictly increasing")

    offset = _target_offset(window, window_mode)

    # Contiguous runs exactly as badeline.train.contiguous_runs defines them.
    bounds = np.flatnonzero(delta != 1) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(efi)]))

    pred_shard_pos_parts = []
    recon_lengths = []
    for run_start, run_end in zip(starts, ends):
        n_windows = run_end - run_start - window + 1
        if n_windows < 1:
            continue
        target_start = run_start + offset
        pred_shard_pos_parts.append(
            np.arange(target_start, target_start + n_windows)
        )
        recon_lengths.append(n_windows)
    if not pred_shard_pos_parts:
        raise ValueError("no run is long enough for a single window")
    pred_shard_pos = np.concatenate(pred_shard_pos_parts)

    if recon_lengths != list(lengths):
        raise ValueError("stream reconstruction mismatch vs session_lengths")
    if not np.array_equal(keys[pred_shard_pos], y_true):
        raise ValueError("y_true mismatch vs shard keys")
    if not np.array_equal(active[pred_shard_pos], pred_active):
        raise ValueError("input_active mismatch vs shard")

    slots = efi - efi[0]
    total = int(slots[-1]) + 1
    truth = np.full((total, 7), -1, dtype=np.int8)
    truth[slots] = keys
    act_grid = np.zeros(total, dtype=bool)
    act_grid[slots] = active
    covered = np.zeros(total, dtype=bool)
    covered[slots[pred_shard_pos]] = True
    prob = np.full((total, 7), np.nan)
    prob[slots[pred_shard_pos]] = y_prob

    onset = (truth[1:] == 1) & (truth[:-1] == 0)
    onset = np.concatenate([np.zeros((1, 7), bool), onset])

    return Timeline(
        truth=truth,
        prob=prob,
        covered=covered,
        active=act_grid,
        onset=onset,
        slots=slots,
        pred_shard_pos=pred_shard_pos,
        efi0=int(efi[0]),
    )


def _scored_accuracy(
    truth: np.ndarray, prob: np.ndarray, scored: np.ndarray
) -> tuple[float, float, int]:
    """Return (micro, joint, scored_frames) at THRESHOLD over scored rows."""

    n = int(scored.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    correct = (prob[scored] >= THRESHOLD) == (truth[scored] == 1)
    return float(correct.mean()), float(correct.all(axis=1).mean()), n


def enumerate_windows(
    timeline: Timeline,
    *,
    span: int = 1800,
    stride: int = 60,
    min_coverage: float,
    min_activity: float,
) -> list[Candidate]:
    """All gated span-slot windows, in ascending start order."""

    if span < 1 or stride < 1:
        raise ValueError("span and stride must be positive")
    total = len(timeline.truth)
    out: list[Candidate] = []
    for start in range(0, total - span + 1, stride):
        window = slice(start, start + span)
        captured = timeline.truth[window, 0] >= 0
        coverage = float(timeline.covered[window].mean())
        if coverage < min_coverage:
            continue
        n_captured = int(captured.sum())
        n_active = int(timeline.active[window].sum())
        if n_active < min_activity * n_captured:
            continue
        scored = timeline.covered[window] & timeline.active[window]
        micro, joint, n_scored = _scored_accuracy(
            timeline.truth[window], timeline.prob[window], scored
        )
        out.append(
            Candidate(
                start=start,
                span=span,
                coverage=coverage,
                activity=n_active / n_captured if n_captured else 0.0,
                micro=micro,
                joint=joint,
                onsets=int(timeline.onset[window].sum()),
                scored_frames=n_scored,
            )
        )
    return out


def enumerate_stream_windows(
    preds: Mapping[str, np.ndarray],
    *,
    span: int = 1800,
    stride: int = 60,
    min_activity: float,
    source_frame: Callable[[int, int], int] | None = None,
    max_source_frame: int | None = None,
) -> list[Candidate]:
    """Gated windows inside single sidecar streams (gapless by construction).

    `source_frame(stream, local_index)` maps a row to an absolute source-video
    frame; windows whose last frame reaches `max_source_frame` are dropped
    (used to keep y4n windows inside the shorter local proxy video).
    """

    if span < 1 or stride < 1:
        raise ValueError("span and stride must be positive")
    if (max_source_frame is None) != (source_frame is None):
        raise ValueError("source_frame and max_source_frame come together")
    y_true = np.asarray(preds["y_true"]).astype(bool)
    y_prob = np.asarray(preds["y_prob"]).astype(np.float64)
    active = np.asarray(preds["input_active"]).astype(bool)
    lengths = np.asarray(preds["session_lengths"])
    if int(lengths.sum()) != len(y_true):
        raise ValueError("session_lengths must sum to the sidecar row count")

    out: list[Candidate] = []
    offset = 0
    for stream, length in enumerate(lengths):
        for local in range(0, int(length) - span + 1, stride):
            if source_frame is not None:
                assert max_source_frame is not None
                if source_frame(stream, local + span - 1) >= max_source_frame:
                    continue
            rows = slice(offset + local, offset + local + span)
            n_active = int(active[rows].sum())
            if n_active < min_activity * span:
                continue
            scored = active[rows]
            micro, joint, n_scored = _scored_accuracy(
                y_true[rows].astype(np.int8), y_prob[rows], scored
            )
            truth = y_true[rows]
            onsets = int((truth[1:] & ~truth[:-1]).sum())
            out.append(
                Candidate(
                    start=offset + local,
                    span=span,
                    coverage=1.0,
                    activity=n_active / span,
                    micro=micro,
                    joint=joint,
                    onsets=onsets,
                    scored_frames=n_scored,
                    stream=stream,
                    local_start=local,
                )
            )
        offset += int(length)
    return out


def _median_onsets(candidates: Sequence[Candidate]) -> int:
    counts = np.sort([c.onsets for c in candidates])
    return int(counts[len(counts) // 2])


def select_median_onset_window(candidates: Sequence[Candidate]) -> Candidate:
    """Earliest candidate at the median all-key onset count."""

    if not candidates:
        raise ValueError("no candidate windows")
    median = _median_onsets(candidates)
    return next(c for c in candidates if c.onsets == median)


def select_exhibit_window(
    candidates: Sequence[Candidate], *, target_micro: float
) -> Candidate:
    """Onset count >= median, micro nearest target, earliest start on ties."""

    if not candidates:
        raise ValueError("no candidate windows")
    median = _median_onsets(candidates)
    eligible = [c for c in candidates if c.onsets >= median]
    return min(eligible, key=lambda c: (abs(c.micro - target_micro), c.start))


def y4n_source_frame(
    stream: int,
    local_index: int,
    *,
    first_mapped_frame: int = 27_600,
    part_frames: int = 36_000,
    target_offset: int = 189,
) -> int:
    """Absolute 60 Hz source-video frame for a y4n sidecar row.

    The y4n mapped label span starts at source frame 27,600 (chunk 23 x 1,200),
    is split into 36,000-frame parts, and each part loses `target_offset`
    frames of leading context (window 128, stride 3, centered).
    """

    if stream < 0 or local_index < 0:
        raise ValueError("stream and local_index must be non-negative")
    return first_mapped_frame + part_frames * stream + target_offset + local_index
