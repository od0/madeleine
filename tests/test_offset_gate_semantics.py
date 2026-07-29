"""Gate-semantics regression for OffsetPolicy v3.

v2 (2026-07-27) replaced the v1 exact-lag mode gate with a winner±1 collar
after measuring that capture sampling phase spreads a 3-engine-frame hitstop
across adjacent video lags.  v3 (2026-07-28, owner-authorized) removes the
per-event collar fraction as a blocking gate entirely: the ground-truth
diagnostic (results/wild20/offset-gate-groundtruth-diagnostic/) measured
collar fractions of 0.63-0.93 on engine-truth sessions whose true offset is 0
by construction, so the statistic tracks per-event motion SNR, not offset
correctness.  The blocking gates are the ones ground truth validated: strong
events, non-adjacent median margin, bootstrap collar decisiveness, and
temporal-block unanimity within the collar.  A margin-only shortfall with a
decisive winner yields the ``uncertain_adjacent`` tier, admissible only
through a human acceptance that passes ``--accept-uncertain-tier``.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from harvest.accept_wild_offset import accept_offset
from harvest.calibrate_offset import (
    OffsetPolicy,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_UNCERTAIN_ADJACENT,
    evaluate_offset_evidence,
)
from harvest.fetch_wild import sha256_file
from tests.test_layout_acceptance import _layout_review_fixture
from tests.test_offset_acceptance import _write_calibration

REL_START = -16
WIDTH = 33
TRUE_LAG = 0


def _synthetic_motions(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    """Measured freeze-length jitter plus no-dash key-press contamination."""

    freeze_lengths = [2] * 11 + [3] * 51 + [4] * 26 + [5] * 9 + [6] * 22
    rng.shuffle(freeze_lengths)
    rows = []
    for length in freeze_lengths:
        motion = rng.normal(6.0, 1.0, WIDTH).clip(2.0, None)
        phase = int(rng.choice([-1, 0, 1], p=[0.15, 0.70, 0.15]))  # sampling phase
        start = TRUE_LAG + 1 + phase - REL_START
        motion[start:start + length] = rng.uniform(0.02, 0.10, length)
        rows.append(motion)
    strong = len(rows)
    for _ in range(60):  # no-dash key presses: motion but no freeze
        rows.append(rng.normal(6.0, 1.5, WIDTH).clip(1.0, None))
    onsets = np.arange(len(rows), dtype=np.int64) * 600
    return np.asarray(rows), onsets[: len(rows)], strong


def _clean_freeze_row(lag: int, jitter: float = 0.0) -> np.ndarray:
    row = 6.0 + 0.02 * np.sin(np.arange(WIDTH) + jitter)
    for relative in (lag + 1, lag + 2, lag + 3):
        row[relative - REL_START] = 0.05
    row[lag + 4 - REL_START] = 12.0
    return row


def test_v3_passes_jittered_ground_truth_with_uncertainty_one() -> None:
    motions, onsets, strong = _synthetic_motions(np.random.default_rng(20260727))
    result = evaluate_offset_evidence(motions, onsets, REL_START, OffsetPolicy())
    assert result["verdict"] == VERDICT_PASS
    assert result["automatic_gates_passed"], result["failure_reasons"]
    assert result["best_candidate_offset_frames"] == TRUE_LAG
    assert result["offset_uncertainty_frames"] == 1
    # The strong-event filter rejects the no-dash contamination.
    assert result["usable_events"] <= strong + 5

    # The v1 exact-lag criterion fails this same evidence: fewer than 80% of
    # events agree on the exact frame, because sampling phase spreads a
    # 3-engine-frame freeze across adjacent video lags.
    rows = {r["offset_frames"]: r for r in result["candidate_rows"]}
    exact_fraction = rows[TRUE_LAG]["event_wins"] / result["usable_events"]
    assert exact_fraction < 0.80 < result["per_event_collar_fraction"]


def test_v3_passes_low_snr_footage_with_a_correct_decisive_winner() -> None:
    # Models the rec_20260725_192824 ground-truth session (collar fraction
    # 0.630 at a true offset of 0) and the kd/Y6 wild rejections: a minority
    # of usable events lock onto unrelated strong motion at scattered
    # non-collar lags, but the aggregate winner stays correct with a
    # comfortable margin, a decisive bootstrap, and unanimous blocks.
    rng = np.random.default_rng(20260728)
    motions, onsets, strong = _synthetic_motions(rng)
    scattered_lags = [lag for lag in range(-10, 11) if abs(lag) >= 4]
    artifact_rows = [
        _clean_freeze_row(int(rng.choice(scattered_lags)), jitter=float(i))
        for i in range(40)
    ]
    rows = list(motions) + artifact_rows
    order = rng.permutation(len(rows))
    motions = np.asarray(rows)[order]
    onsets = np.arange(len(rows), dtype=np.int64) * 600

    result = evaluate_offset_evidence(motions, onsets, REL_START, OffsetPolicy())
    assert result["verdict"] == VERDICT_PASS, result["failure_reasons"]
    assert result["best_candidate_offset_frames"] == TRUE_LAG
    # The v2 collar-fraction floor would have rejected this correct winner.
    assert result["per_event_collar_fraction"] < 0.80
    assert result["median_score_margin"] >= 2.0
    assert result["bootstrap_win_fraction"] >= 0.95


def test_v3_uncertain_tier_for_margin_only_shortfall() -> None:
    # A correlated secondary motion dip compresses the non-adjacent margin
    # below the floor while the winner stays decisive: bootstrap 1.0 and
    # unanimous temporal blocks.  This is the nRM/6vE/v498/b43 signature.
    rows = []
    for index in range(60):
        row = _clean_freeze_row(TRUE_LAG, jitter=float(index))
        for relative in (7, 8, 9):
            row[relative - REL_START] = 0.30
        row[10 - REL_START] = 8.0
        rows.append(row)
    onsets = np.arange(len(rows), dtype=np.int64) * 600
    result = evaluate_offset_evidence(
        np.asarray(rows), onsets, REL_START, OffsetPolicy()
    )
    assert result["verdict"] == VERDICT_UNCERTAIN_ADJACENT
    assert not result["automatic_gates_passed"]
    assert result["best_candidate_offset_frames"] == TRUE_LAG
    assert result["offset_uncertainty_frames"] == 1
    assert result["median_score_margin"] < 2.0
    assert result["bootstrap_win_fraction"] >= 0.95
    assert all(
        reason.startswith("non-adjacent median margin")
        for reason in result["failure_reasons"]
    )


def test_v3_fails_a_split_cohort_without_any_tier() -> None:
    # Two genuinely different offsets in one video must still hard-fail: half
    # the strong events at lag 0, half at lag -3 (outside the collar).
    rng = np.random.default_rng(7)
    rows = []
    for i in range(60):
        motion = rng.normal(6.0, 1.0, WIDTH).clip(2.0, None)
        lag = TRUE_LAG if i % 2 == 0 else TRUE_LAG - 3
        start = lag + 1 - REL_START
        motion[start:start + 3] = 0.05
        rows.append(motion)
    onsets = np.arange(len(rows), dtype=np.int64) * 600
    result = evaluate_offset_evidence(np.asarray(rows), onsets, REL_START, OffsetPolicy())
    assert result["verdict"] == VERDICT_FAIL
    assert not result["automatic_gates_passed"]
    # At least one blocking failure beyond the margin: split evidence may not
    # be laundered into the uncertain tier.
    assert any(
        not reason.startswith("non-adjacent median margin")
        for reason in result["failure_reasons"]
    )


def test_v3_block_disagreement_beyond_collar_is_a_hard_fail() -> None:
    # A drifting offset (each chronological third at a different lag) must
    # remain a hard failure even when other statistics look decisive.
    rows = [_clean_freeze_row(lag, jitter=float(i)) for i, lag in enumerate(
        [-4] * 20 + [0] * 20 + [4] * 20
    )]
    onsets = np.arange(len(rows), dtype=np.int64) * 600
    result = evaluate_offset_evidence(
        np.asarray(rows), onsets, REL_START, OffsetPolicy()
    )
    assert result["verdict"] == VERDICT_FAIL
    assert any("temporal block" in reason for reason in result["failure_reasons"])


def test_uncertain_tier_acceptance_requires_the_explicit_flag(tmp_path: Path) -> None:
    input_layout, layout_acceptance, _, _ = _layout_review_fixture(
        tmp_path, video_id="acceptance_test"
    )
    calibration, _ = _write_calibration(
        tmp_path / "calibration", input_layout, verdict=VERDICT_UNCERTAIN_ADJACENT
    )
    with pytest.raises(ValueError, match="accept-uncertain-tier"):
        accept_offset(
            calibration,
            input_layout,
            layout_acceptance,
            tmp_path / "layout.final.json",
            calibration.parent / "offset_acceptance.json",
            reviewer_identity="Reviewer",
            reviewer_kind="human",
            approved=True,
        )
    acceptance_path = calibration.parent / "offset_acceptance.json"
    acceptance = accept_offset(
        calibration,
        input_layout,
        layout_acceptance,
        tmp_path / "layout.final.json",
        acceptance_path,
        reviewer_identity="Reviewer",
        reviewer_kind="human",
        approved=True,
        accept_uncertain_tier=True,
    )
    assert acceptance["accepted_tier"] == VERDICT_UNCERTAIN_ADJACENT
    assert acceptance["accepted_offset_uncertainty_frames"] == 1
    assert acceptance["decision"]["uncertain_tier_acknowledged"] is True
    written = json.loads(acceptance_path.read_text())
    assert written["accepted_tier"] == VERDICT_UNCERTAIN_ADJACENT
    layout_raw = json.loads((tmp_path / "layout.final.json").read_text())
    embedded = layout_raw["temporal_offset_acceptance"]
    assert embedded["tier"] == VERDICT_UNCERTAIN_ADJACENT
    assert embedded["offset_uncertainty_frames"] == 1
    assert embedded["calibration_sha256"] == sha256_file(calibration)


def test_pass_verdict_rejects_a_spurious_uncertain_flag(tmp_path: Path) -> None:
    input_layout, layout_acceptance, _, _ = _layout_review_fixture(
        tmp_path, video_id="acceptance_test"
    )
    calibration, _ = _write_calibration(tmp_path / "calibration", input_layout)
    with pytest.raises(ValueError, match="applies only to uncertain_adjacent"):
        accept_offset(
            calibration,
            input_layout,
            layout_acceptance,
            tmp_path / "layout.final.json",
            calibration.parent / "offset_acceptance.json",
            reviewer_identity="Reviewer",
            reviewer_kind="human",
            approved=True,
            accept_uncertain_tier=True,
        )
