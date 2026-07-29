"""Synthetic tests for the shared prediction-timeline library."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.figures.pred_timeline import (
    Candidate,
    Timeline,
    enumerate_stream_windows,
    enumerate_windows,
    reconstruct_timeline,
    select_exhibit_window,
    select_median_onset_window,
    y4n_source_frame,
)
from experiments.keypress_accuracy import score_sidecar

WINDOW = 8
OFFSET = (WINDOW - 1) // 2  # centered


def _synthetic(seed: int = 0):
    """A shard with two contiguous runs split by a capture drop, plus its
    sidecar built by an independent brute-force reference."""

    rng = np.random.default_rng(seed)
    efi = np.concatenate([np.arange(100, 140), np.arange(150, 180)])
    keys = rng.integers(0, 2, size=(len(efi), 7)).astype(np.uint8)
    active = rng.integers(0, 2, size=len(efi)).astype(np.uint8)
    shard = {"engine_frame_idx": efi, "keys": keys, "input_active": active}

    targets = []
    lengths = []
    for run_start, run_len in ((0, 40), (40, 30)):
        n_windows = run_len - WINDOW + 1
        first = run_start + OFFSET
        targets.append(np.arange(first, first + n_windows))
        lengths.append(n_windows)
    targets = np.concatenate(targets)
    preds = {
        "y_true": keys[targets],
        "y_prob": rng.random((len(targets), 7)).astype(np.float32),
        "input_active": active[targets],
        "session_lengths": np.asarray(lengths, dtype=np.int64),
    }
    return shard, preds, targets


def test_reconstruction_matches_brute_force():
    shard, preds, targets = _synthetic()
    timeline = reconstruct_timeline(preds, shard, window=WINDOW)

    assert np.array_equal(timeline.pred_shard_pos, targets)
    slots = shard["engine_frame_idx"] - 100
    assert timeline.efi0 == 100
    # Predictions land exactly on their engine slots.
    assert np.array_equal(np.flatnonzero(timeline.covered), slots[targets])
    assert np.array_equal(
        timeline.prob[slots[targets]], preds["y_prob"].astype(np.float64)
    )
    # The gap between the runs is a real hole: unknown truth, no predictions.
    gap = slice(40, 50)
    assert np.all(timeline.truth[gap] == -1)
    assert not timeline.covered[gap].any()
    assert not timeline.active[gap].any()
    # Onsets: 0 -> 1 with the previous slot captured, never across the gap.
    truth = timeline.truth
    expected_onset = (truth[1:] == 1) & (truth[:-1] == 0)
    assert np.array_equal(timeline.onset[1:], expected_onset)
    assert not timeline.onset[50].any()  # slot before it is the drop


def test_reconstruction_refuses_bad_inputs():
    shard, preds, _ = _synthetic()

    reset = dict(shard)
    reset["engine_frame_idx"] = np.concatenate(
        [np.arange(100, 140), np.arange(90, 120)]
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        reconstruct_timeline(preds, reset, window=WINDOW)

    tampered = {k: np.array(v) for k, v in preds.items()}
    tampered["y_true"] = tampered["y_true"].copy()
    tampered["y_true"][3, 2] ^= 1
    with pytest.raises(ValueError, match="y_true mismatch"):
        reconstruct_timeline(tampered, shard, window=WINDOW)

    short = {k: np.array(v) for k, v in preds.items()}
    short["session_lengths"] = short["session_lengths"][:-1]
    with pytest.raises(ValueError, match="reconstruction mismatch"):
        reconstruct_timeline(short, shard, window=WINDOW)


def _timeline(total: int, covered, active, truth, prob) -> Timeline:
    onset = (truth[1:] == 1) & (truth[:-1] == 0)
    onset = np.concatenate([np.zeros((1, 7), bool), onset])
    return Timeline(
        truth=truth,
        prob=prob,
        covered=covered,
        active=active,
        onset=onset,
        slots=np.arange(total),
        pred_shard_pos=np.flatnonzero(covered),
        efi0=0,
    )


def test_enumerate_windows_gates_and_scores():
    total, span = 200, 100
    truth = np.zeros((total, 7), dtype=np.int8)
    truth[10:20, 0] = 1  # one press -> one onset, inside the first window
    prob = np.full((total, 7), 0.1)
    prob[10:20, 0] = 0.9
    covered = np.ones(total, dtype=bool)
    active = np.ones(total, dtype=bool)
    covered[100:180] = False  # second window: 20% coverage

    timeline = _timeline(total, covered, active, truth, prob)
    cands = enumerate_windows(
        timeline, span=span, stride=100, min_coverage=0.5, min_activity=0.5
    )
    assert [c.start for c in cands] == [0]
    only = cands[0]
    assert only.coverage == 1.0 and only.activity == 1.0
    assert only.onsets == 1 and only.scored_frames == span
    # Every frame/key decision in the window is correct by construction.
    assert only.micro == 1.0 and only.joint == 1.0

    # Dropping activity below the gate removes the window.
    inactive = active.copy()
    inactive[:80] = False
    timeline = _timeline(total, covered, inactive, truth, prob)
    assert not enumerate_windows(
        timeline, span=span, stride=100, min_coverage=0.5, min_activity=0.5
    )


def _candidate(start: int, micro: float, onsets: int) -> Candidate:
    return Candidate(
        start=start,
        span=100,
        coverage=1.0,
        activity=1.0,
        micro=micro,
        joint=micro,
        onsets=onsets,
        scored_frames=100,
    )


def test_selection_rules():
    cands = [
        _candidate(0, 0.90, 1),
        _candidate(50, 0.60, 5),
        _candidate(100, 0.72, 5),
        _candidate(150, 0.70, 9),
    ]
    # Median onset count of [1, 5, 5, 9] is 5; earliest such window wins.
    assert select_median_onset_window(cands).start == 50
    # Exhibit rule: onsets >= 5 keeps starts 50/100/150; nearest 0.70 wins.
    assert select_exhibit_window(cands, target_micro=0.70).start == 150
    # Tie on |micro - target| breaks to the earliest start.
    tie = [_candidate(0, 0.68, 5), _candidate(60, 0.72, 5)]
    assert select_exhibit_window(tie, target_micro=0.70).start == 0
    with pytest.raises(ValueError):
        select_exhibit_window([], target_micro=0.7)


def test_window_tally_matches_keypress_accuracy(tmp_path):
    rng = np.random.default_rng(1)
    n = 600
    preds = {
        "y_true": rng.integers(0, 2, size=(n, 7)).astype(np.uint8),
        "y_prob": rng.random((n, 7)).astype(np.float32),
        "input_active": rng.integers(0, 2, size=n).astype(np.uint8),
        "session_lengths": np.asarray([n], dtype=np.int64),
    }
    (cand,) = enumerate_stream_windows(
        preds, span=n, stride=n, min_activity=0.0
    )
    sidecar = tmp_path / "preds.npz"
    np.savez(sidecar, **preds)
    report = score_sidecar(sidecar)
    assert cand.micro == pytest.approx(report["key_state_micro_accuracy"])
    assert cand.joint == pytest.approx(report["joint_exact_match_accuracy"])
    assert cand.scored_frames == report["frames"]


def test_y4n_mapping_and_proxy_bound():
    # First prediction of stream 0 targets source frame 27,600 + 189.
    assert y4n_source_frame(0, 0) == 27_789
    assert y4n_source_frame(2, 5) == 27_600 + 2 * 36_000 + 189 + 5
    with pytest.raises(ValueError):
        y4n_source_frame(-1, 0)

    rng = np.random.default_rng(2)
    n = 400
    preds = {
        "y_true": rng.integers(0, 2, size=(n, 7)).astype(np.uint8),
        "y_prob": rng.random((n, 7)).astype(np.float32),
        "input_active": np.ones(n, dtype=np.uint8),
        "session_lengths": np.asarray([n], dtype=np.int64),
    }
    unbounded = enumerate_stream_windows(
        preds, span=100, stride=100, min_activity=0.0
    )
    assert [c.local_start for c in unbounded] == [0, 100, 200, 300]
    bounded = enumerate_stream_windows(
        preds,
        span=100,
        stride=100,
        min_activity=0.0,
        source_frame=y4n_source_frame,
        max_source_frame=y4n_source_frame(0, 250),
    )
    # Windows whose final row maps at or past the proxy's end are dropped.
    assert [c.local_start for c in bounded] == [0, 100]
