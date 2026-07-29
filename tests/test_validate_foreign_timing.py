from __future__ import annotations

import numpy as np

from experiments.validate_foreign_timing import transition_lag_scores


def test_transition_lag_scores_recovers_zero_offset() -> None:
    keys = np.zeros((30, 2), dtype=np.uint8)
    keys[10:20, 0] = 1
    keys[15:, 1] = 1
    change = np.ones(30, dtype=np.float32)
    change[[10, 15, 20]] = 10

    scores, events = transition_lag_scores(keys, change, radius=3)

    assert events.tolist() == [10, 15, 20]
    assert max(scores, key=scores.get) == 0
    assert scores[0] == 10.0
