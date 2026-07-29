from __future__ import annotations

import numpy as np
import pytest

from harvest.propose_layout_family_pairs import (
    edge_similarity,
    reference_region,
    stable_edge_map,
)


def test_reference_region_bounds_seven_cells() -> None:
    layout = {
        "cells": [
            {"sample_rect": [0.1 + index * 0.05, 0.8, 0.01, 0.02]}
            for index in range(7)
        ]
    }

    assert reference_region(layout) == pytest.approx((0.075, 0.765, 0.435, 0.855))


def _moving_background_with_fixed_hud(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    frames = generator.random((16, 90, 160), dtype=np.float32) * 0.35
    frames[:, 70:72, 10:70] = 1.0
    frames[:, 70:86, 10:12] = 1.0
    frames[:, 84:86, 10:70] = 1.0
    frames[:, 70:86, 68:70] = 1.0
    return frames


def test_stable_edges_match_across_moving_backgrounds() -> None:
    first = stable_edge_map(_moving_background_with_fixed_hud(1), (0, 0, 0.5, 1))
    second = stable_edge_map(_moving_background_with_fixed_hud(2), (0, 0, 0.5, 1))
    metrics = edge_similarity(first, second)

    assert metrics["correlation"] > 0.8
    assert metrics["score"] > 0.7


def test_different_hud_geometry_scores_lower() -> None:
    first_frames = _moving_background_with_fixed_hud(1)
    second_frames = _moving_background_with_fixed_hud(2)
    second_frames[:, 70:86, 10:70] = second_frames[:, 70:86, 10:70] * 0.2
    second_frames[:, 50:52, 10:70] = 1.0
    second_frames[:, 50:66, 10:12] = 1.0
    second_frames[:, 64:66, 10:70] = 1.0
    second_frames[:, 50:66, 68:70] = 1.0

    first = stable_edge_map(first_frames, (0, 0, 0.5, 1))
    second = stable_edge_map(second_frames, (0, 0, 0.5, 1))

    assert edge_similarity(first, second)["score"] < 0.6
