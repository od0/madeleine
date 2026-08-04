from __future__ import annotations

import numpy as np

from experiments.eval_vpt_small_native60_wild7 import (
    completed_window_starts,
    inference_window_slices,
)


def test_completed_window_starts_appends_end_aligned_tail() -> None:
    starts = completed_window_starts([(0, 1342)], window=384, stride=192)

    assert np.array_equal(starts, np.asarray([0, 192, 384, 576, 768, 958]))


def test_completed_window_starts_does_not_duplicate_aligned_tail() -> None:
    starts = completed_window_starts([(0, 448)], window=128, stride=64)

    assert np.array_equal(starts, np.asarray([0, 64, 128, 192, 256, 320]))


def test_inference_window_slices_pads_only_short_context_after_scored_center() -> None:
    assert inference_window_slices([(0, 382)], window=384, stride=192) == [(0, 382)]
    assert inference_window_slices([(0, 287)], window=384, stride=192) == []
