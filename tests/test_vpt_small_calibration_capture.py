from __future__ import annotations

import numpy as np

from data.schema import KEY_ORDER
from experiments.prepare_vpt_small_calibration_capture import count_positive_state_runs


def test_support_counter_excludes_boundaries_gaps_rooms_and_positive_start() -> None:
    rows = 14
    keys = np.zeros((rows, len(KEY_ORDER)), dtype=np.uint8)
    active = np.ones(rows, dtype=np.uint8)
    engine = np.asarray([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17])
    rooms = np.asarray(["a", "a", "a", "a", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b"])

    keys[0, 4] = 1  # positive segment start: excluded
    keys[2, 4] = 1  # ordinary jump onset: counted
    keys[4, 5] = 1  # dash onset at room boundary: excluded
    keys[6, 3] = 1  # down onset after engine-index gap: excluded
    active[8] = 0
    keys[8, 0] = 1  # inactive onset: excluded
    keys[10, 0] = 1  # ordinary left onset: counted
    keys[12, 1] = 1
    keys[12, 2] = 1  # simultaneous right/up onsets: both counted

    counts = count_positive_state_runs(keys, active, engine, rooms)
    assert counts == {
        "left": 1,
        "right": 1,
        "up": 1,
        "down": 0,
        "jump": 1,
        "dash": 0,
        "grab": 0,
    }


def test_support_counter_rejects_misaligned_arrays() -> None:
    keys = np.zeros((4, len(KEY_ORDER)), dtype=np.uint8)
    try:
        count_positive_state_runs(keys, np.ones(3), np.arange(4), np.asarray(["a"] * 4))
    except ValueError as error:
        assert "aligned" in str(error)
    else:
        raise AssertionError("misaligned support arrays were accepted")
