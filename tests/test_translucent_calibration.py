"""Threshold calibration on raw float scores (madeleine.cell-threshold.v2).

The v1 calibrator min-max rescaled each cell's scores to uint8 before running
cv2's integer Otsu. When one cluster was much tighter than the full score
range — an opaque overlay's released state spans ~2 luma against a ~180-luma
range — the whole cluster collapsed into 2-3 uint8 bins, the back-mapped
integer level could land inside the cluster's quantization halo, and the
recentering rescue in the decoder could then freeze on the wrong side
(ofy37Fm6EgI bottom_grab: threshold 71.92, separation 0.17 against the same
data's 181.0 at the mid-gap threshold). These tests pin the v2 behavior: the
split is computed on the raw floats and lands at the midpoint of the two
cluster medians, in the empty gap.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest
import numpy as np

from harvest.decode_wild import _cluster_separation
from harvest.translucent_parser import (
    CALIBRATION_METHOD,
    calibrate_threshold,
    decode,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ofy37Fm6EgI_sampled_scores.npz"


def _v1_calibrate_and_recenter(column: np.ndarray) -> float:
    """The removed v1 path, kept here as the regression reference.

    uint8 min-max rescale, integer Otsu, back-map, then the decoder's
    median-midpoint recentering rescue. Reproduced verbatim so the tests can
    demonstrate the pathology the v2 calibrator fixes.
    """

    lo, hi = float(column.min()), float(column.max())
    scaled = np.clip((column - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    level, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = lo + (level / 255.0) * (hi - lo)
    low, high = column[column < threshold], column[column >= threshold]
    if low.size and high.size:
        threshold = (float(np.median(low)) + float(np.median(high))) / 2.0
    return threshold


def test_tight_cluster_quantization_regression() -> None:
    # Released cluster ~1 luma wide, split across two adjacent uint8 bins of
    # the v1 rescale with the dominant value in the upper bin; pressed cluster
    # at 253. This is the codec-halo shape from the ofy37Fm6EgI decode.
    released = np.concatenate([
        np.full(100, 71.6, dtype=np.float32),
        np.full(200, 72.3, dtype=np.float32),
    ])
    pressed = np.full(89, 253.0, dtype=np.float32)
    column = np.concatenate([released, pressed])
    truth = np.concatenate([
        np.zeros(released.size, dtype=bool), np.ones(pressed.size, dtype=bool)
    ])

    # The old path splits the released cluster against itself: its threshold
    # sits at or below the cluster's own values, so released frames decode as
    # pressed and the separation statistic collapses.
    old = _v1_calibrate_and_recenter(column)
    assert old <= float(released.max())
    old_states = column >= old
    assert (old_states & ~truth).any()          # false presses on released frames
    old_separation, _ = _cluster_separation(column, old)
    assert old_separation < 1.5                 # would fail the QC gate

    # The v2 calibrator lands in the empty gap and decodes exactly.
    new = float(calibrate_threshold(column[:, None])[0])
    assert float(released.max()) < new < float(pressed.min())
    assert np.array_equal(decode(column[:, None], np.array([new], np.float32))[:, 0], truth)
    new_separation, _ = _cluster_separation(column, new)
    assert new_separation > 100.0


def test_well_separated_invariance() -> None:
    # Wide, healthy clusters: the v2 threshold must land at the midpoint of
    # the cluster medians — the intended behavior of the old
    # calibrate-then-recenter pair — and inside the empty gap.
    rng = np.random.default_rng(20260728)
    released = rng.normal(40.0, 6.0, size=500).astype(np.float32)
    pressed = rng.normal(200.0, 6.0, size=180).astype(np.float32)
    column = np.concatenate([released, pressed])

    new = float(calibrate_threshold(column[:, None])[0])
    midgap = (float(np.median(released)) + float(np.median(pressed))) / 2.0
    assert abs(new - midgap) < 1.0
    assert float(released.max()) < new < float(pressed.min())
    assert abs(new - _v1_calibrate_and_recenter(column)) < 1.0


def test_degenerate_columns_keep_v1_contract() -> None:
    # All-NaN column -> 0.0; constant column -> a threshold that never fires.
    column = np.full((50, 1), np.nan, dtype=np.float32)
    assert float(calibrate_threshold(column)[0]) == 0.0
    constant = np.full((50, 1), 7.25, dtype=np.float32)
    threshold = calibrate_threshold(constant)
    assert not decode(constant, threshold).any()


@pytest.mark.requires_private_artifacts("tests/fixtures/ofy37Fm6EgI_sampled_scores.npz")
def test_ofy_fixture_all_cells_calibrate_mid_gap() -> None:
    # 400 frames re-scored from the ofy37Fm6EgI decode with the accepted
    # layout: nine opaque cells, released ~72, pressed ~253, empty gap ~180
    # luma. The six recent wild20 decodes calibrated every passing cell at
    # the mid-gap 162.5; the fix must reproduce that on every cell, including
    # bottom_grab where v1 froze inside the released cluster.
    data = np.load(FIXTURE)
    scores, cell_ids = data["scores"], data["cell_ids"]

    thresholds = calibrate_threshold(scores)
    for index, cell_id in enumerate(cell_ids):
        column = scores[:, index]
        threshold = float(thresholds[index])
        assert abs(threshold - 162.5) < 0.1, (cell_id, threshold)
        separation, minority = _cluster_separation(column, threshold)
        assert separation > 100.0, (cell_id, separation)
        assert minority > 0

    # bottom_grab decodes at its true prevalence (~22%, climb-heavy play),
    # not the 0.84 duty the in-cluster v1 threshold produced.
    grab = int(np.flatnonzero(cell_ids == "bottom_grab")[0])
    duty = float((scores[:, grab] >= thresholds[grab]).mean())
    assert abs(duty - 0.2225) < 1e-6

    # The v1 reference path fails on this same data for at least one cell —
    # which cell fails is a knife-edge function of the histogram, which is
    # exactly why the pre-quantization had to go.
    v1_separations = []
    for index in range(scores.shape[1]):
        old = _v1_calibrate_and_recenter(scores[:, index])
        v1_separations.append(_cluster_separation(scores[:, index], old)[0])
    assert min(v1_separations) < 1.5


@pytest.mark.requires_private_artifacts("tests/fixtures/ofy37Fm6EgI_sampled_scores.npz")
def test_ofy_fixture_passing_cells_labels_unchanged() -> None:
    # Every previously passing wild20 cell recorded threshold 162.5. The v2
    # thresholds must produce byte-identical labels to that recorded value on
    # every cell, so the fix changes nothing for admitted decodes.
    data = np.load(FIXTURE)
    scores = data["scores"]
    thresholds = calibrate_threshold(scores)
    recorded = np.full(scores.shape[1], 162.5, dtype=np.float32)
    assert np.array_equal(decode(scores, thresholds), decode(scores, recorded))


def test_calibration_method_is_versioned() -> None:
    assert CALIBRATION_METHOD == "madeleine.cell-threshold.v2-float-otsu-midgap"
