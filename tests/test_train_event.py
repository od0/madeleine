from __future__ import annotations

import numpy as np
import pytest

from badeline.train import SessionArrays
from badeline.train_event import EventSegmentSessionDataset, event_positive_weights
from data.schema import KEY_ORDER


def _session(
    keys: np.ndarray,
    *,
    active: np.ndarray | None = None,
    engine: np.ndarray | None = None,
) -> SessionArrays:
    frame_count = len(keys)
    return SessionArrays(
        "fixture",
        np.zeros((frame_count, 4), dtype=np.float32),
        keys.astype(np.uint8),
        np.arange(frame_count, dtype=np.int64) if engine is None else engine,
        np.ones(frame_count, dtype=np.uint8) if active is None else active,
    )


def test_event_segment_first_transition_requires_active_predecessor() -> None:
    keys = np.zeros((12, len(KEY_ORDER)), dtype=np.uint8)
    keys[5:, 0] = 1
    active = np.ones(12, dtype=np.uint8)
    active[4] = 0
    dataset = EventSegmentSessionDataset(
        [_session(keys, active=active)],
        window=3,
        window_mode="centered",
        input_config="pixels",
        history_len=1,
        segment_windows=2,
        active_targets_only=True,
        precomputed_features=True,
    )

    # The first full segment after the inactive target begins at window start
    # four, whose first target is frame five.  Frame four's label may exist,
    # but it cannot supervise an event across the invalid boundary.
    item_index = next(
        index
        for index, (_, start, _) in enumerate(dataset._locations)
        if start == 4
    )
    item = dataset[item_index]
    assert item["previous_target"][0].item() == 0
    assert item["target"][0, 0].item() == 1
    assert not item["previous_valid"].item()


def test_event_segment_retains_valid_predecessor_at_segment_boundary() -> None:
    keys = np.zeros((10, len(KEY_ORDER)), dtype=np.uint8)
    keys[3:, 0] = 1
    dataset = EventSegmentSessionDataset(
        [_session(keys)],
        window=3,
        window_mode="centered",
        input_config="pixels",
        history_len=1,
        segment_windows=2,
        active_targets_only=True,
        precomputed_features=True,
    )

    item_index = next(
        index
        for index, (_, start, _) in enumerate(dataset._locations)
        if start == 2
    )
    item = dataset[item_index]
    assert item["previous_target"][0].item() == 0
    assert item["target"][0, 0].item() == 1
    assert item["previous_valid"].item()


def test_event_positive_weights_use_only_valid_adjacent_pairs() -> None:
    keys = np.zeros((6, len(KEY_ORDER)), dtype=np.uint8)
    keys[:, 0] = [0, 1, 0, 0, 1, 0]
    active = np.asarray([1, 1, 0, 1, 1, 1], dtype=np.uint8)

    onset, release, counts = event_positive_weights(
        [_session(keys, active=active)], maximum=5.0
    )

    # Valid pairs are 0->1, 3->4, and 4->5.  The two pairs touching inactive
    # frame two are excluded from both the denominator and event counts.
    assert counts == {
        "valid_transition_frames": 3,
        "onsets": 2,
        "releases": 1,
    }
    assert onset[0] == pytest.approx(1.0)  # max((3 - 2) / 2, 1)
    assert release[0] == pytest.approx(2.0)
    assert np.all(onset[1:] == 3.0)
    assert np.all(release[1:] == 3.0)


def test_event_positive_weights_do_not_cross_engine_gaps() -> None:
    keys = np.zeros((4, len(KEY_ORDER)), dtype=np.uint8)
    keys[:, 0] = [0, 1, 0, 1]
    engine = np.asarray([0, 1, 9, 10], dtype=np.int64)

    onset, release, counts = event_positive_weights(
        [_session(keys, engine=engine)], maximum=5.0
    )

    assert counts == {
        "valid_transition_frames": 2,
        "onsets": 2,
        "releases": 0,
    }
    assert onset[0] == pytest.approx(1.0)
    assert release[0] == pytest.approx(2.0)


@pytest.mark.parametrize("maximum", [0.0, 0.99])
def test_event_positive_weights_reject_subunit_cap(maximum: float) -> None:
    keys = np.zeros((2, len(KEY_ORDER)), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least one"):
        event_positive_weights([_session(keys)], maximum=maximum)
