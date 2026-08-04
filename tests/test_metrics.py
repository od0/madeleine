from __future__ import annotations

import json

import numpy as np
import pytest

from badeline.metrics import (
    onset_timing_errors,
    per_key_ap,
    per_key_calibration,
    per_key_f1,
    summarize,
)
from data.schema import KEY_ORDER


def _empty_case(frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((frame_count, len(KEY_ORDER)), dtype=bool),
        np.zeros((frame_count, len(KEY_ORDER)), dtype=float),
    )


def test_tiny_hand_computed_ap_and_f1() -> None:
    y_true, y_prob = _empty_case(10)
    left = KEY_ORDER.index("left")
    right = KEY_ORDER.index("right")

    y_true[[0, 2], left] = True
    y_prob[:, left] = np.arange(0.9, -0.1, -0.1)
    y_true[[1, 3], right] = True
    y_prob[:, right] = [0.1, 0.9, 0.2, 0.8, 0.7, 0.6, 0.4, 0.3, 0.05, 0.0]

    ap = per_key_ap(y_true, y_prob)
    f1 = per_key_f1(y_true, y_prob)

    # Left positives rank 1st and 3rd: AP = (1/1 + 2/3) / 2 = 5/6.
    assert ap["left"] == pytest.approx(5 / 6)
    # Five predictions contain 2 TP and 3 FP: F1 = 4 / (4 + 3) = 4/7.
    assert f1["left"] == pytest.approx(4 / 7)
    # Right positives rank 1st and 2nd, so AP is exactly one.
    assert ap["right"] == pytest.approx(1.0)
    # Four predictions contain 2 TP and 2 FP: F1 = 4 / (4 + 2) = 2/3.
    assert f1["right"] == pytest.approx(2 / 3)


def test_zero_positive_key_is_nan_in_metrics_and_summary() -> None:
    y_true, y_prob = _empty_case(4)
    y_true[1, KEY_ORDER.index("left")] = True
    y_prob[1, KEY_ORDER.index("left")] = 0.9

    assert np.isnan(per_key_ap(y_true, y_prob)["grab"])
    assert np.isnan(per_key_f1(y_true, y_prob)["grab"])

    summary = summarize(y_true, y_prob)
    assert np.isnan(summary["per_key_ap"]["grab"])
    assert np.isnan(summary["per_key_f1"]["grab"])


def test_calibration_two_bins_matches_hand_arithmetic() -> None:
    y_true, y_prob = _empty_case(4)
    left = KEY_ORDER.index("left")
    y_true[:, left] = [False, True, False, True]
    y_prob[:, left] = [0.1, 0.3, 0.6, 0.8]

    calibration = per_key_calibration(y_true, y_prob, n_bins=2)["left"]

    assert calibration["bin_edges"] == pytest.approx([0.0, 0.5, 1.0])
    assert calibration["bin_confidence"] == pytest.approx([0.2, 0.7])
    assert calibration["bin_accuracy"] == pytest.approx([0.5, 0.5])
    assert calibration["bin_count"] == [2, 2]
    # ECE = 2/4 * |0.5 - 0.2| + 2/4 * |0.5 - 0.7| = 0.25.
    assert calibration["ece"] == pytest.approx(0.25)


def test_onset_timing_offsets_and_unmatched_count() -> None:
    y_true, y_prob = _empty_case(36)
    left = KEY_ORDER.index("left")
    right = KEY_ORDER.index("right")

    y_true[3:8, left] = True
    y_true[20:25, left] = True
    y_prob[5:7, left] = 0.9
    y_prob[18:20, left] = 0.9

    # A separate onset is too far from its only prediction to be matched.
    y_true[30:32, right] = True
    y_prob[20:22, right] = 0.9

    timing = onset_timing_errors(y_true, y_prob, max_lag=3)

    np.testing.assert_array_equal(timing["left"]["offsets"], [2, -2])
    assert timing["left"]["n_true_onsets"] == 2
    assert timing["left"]["n_matched"] == 2
    np.testing.assert_array_equal(
        timing["right"]["offsets"], np.array([], dtype=int)
    )
    assert timing["right"]["n_true_onsets"] == 1
    assert timing["right"]["n_matched"] == 0


def test_onset_timing_allows_exact_only_matching() -> None:
    y_true, y_prob = _empty_case(3)
    left = KEY_ORDER.index("left")
    y_true[1, left] = True
    y_prob[1, left] = 0.9

    timing = onset_timing_errors(y_true, y_prob, max_lag=0)["left"]

    np.testing.assert_array_equal(timing["offsets"], [0])
    assert timing["n_matched"] == 1


def test_onset_timing_nearest_event_tie_prefers_earlier_prediction() -> None:
    y_true, y_prob = _empty_case(12)
    left = KEY_ORDER.index("left")
    y_true[5, left] = True
    y_prob[3, left] = 0.9
    y_prob[7, left] = 0.9

    timing = onset_timing_errors(y_true, y_prob, max_lag=3)["left"]

    np.testing.assert_array_equal(timing["offsets"], [-2])


def test_summary_has_no_accuracy_field() -> None:
    y_true, y_prob = _empty_case(3)
    summary = summarize(y_true, y_prob)
    serialized = json.dumps(
        summary,
        default=lambda value: value.tolist()
        if isinstance(value, np.ndarray)
        else value,
    )
    decoded = json.loads(serialized)

    def assert_no_accuracy_key(value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                assert key != "accuracy"
                assert_no_accuracy_key(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_no_accuracy_key(nested)

    assert_no_accuracy_key(decoded)
    assert "bin_accuracy" in decoded["per_key_calibration"]["left"]


@pytest.mark.parametrize(
    "function",
    [
        per_key_ap,
        per_key_f1,
        per_key_calibration,
        onset_timing_errors,
        summarize,
    ],
)
@pytest.mark.parametrize(
    ("truth_shape", "probability_shape"),
    [
        ((10,), (10, len(KEY_ORDER))),
        ((10, len(KEY_ORDER) - 1), (10, len(KEY_ORDER) - 1)),
        ((10, len(KEY_ORDER)), (9, len(KEY_ORDER))),
        ((10, len(KEY_ORDER)), (10, len(KEY_ORDER) + 1)),
    ],
)
def test_shape_validation_raises(
    function: object,
    truth_shape: tuple[int, ...],
    probability_shape: tuple[int, ...],
) -> None:
    y_true = np.zeros(truth_shape, dtype=bool)
    y_prob = np.zeros(probability_shape, dtype=float)
    with pytest.raises(ValueError):
        function(y_true, y_prob)


# --- transition-event F1 (brief v3.1 primary metric) ---

from badeline.metrics import (  # noqa: E402
    match_event_counts,
    per_key_transition_f1,
    score_events,
    transition_events,
)


def test_transition_events_respect_segment_boundaries() -> None:
    stream = np.array([0, 1, 1, 1, 1, 1, 0, 0], dtype=bool)

    onsets, offsets = transition_events(stream)
    assert onsets.tolist() == [1]
    assert offsets.tolist() == [6]

    # Split into two sessions: the hold must not span the join. Frame 4 opens
    # segment two already held (counts as an onset); segment one's hold runs
    # into its end (no offset was ever observed).
    onsets, offsets = transition_events(stream, boundaries=[4, 4])
    assert onsets.tolist() == [1, 4]
    assert offsets.tolist() == [6]


def test_persistence_is_zero_at_collar0_and_perfect_at_collar1() -> None:
    y_true, y_prob = _empty_case(18)
    left = KEY_ORDER.index("left")
    y_true[3:7, left] = True
    y_true[10:14, left] = True
    # Persistence: prob[t] = true[t-1] — every event echoed one frame late.
    y_prob[1:, left] = y_true[:-1, left].astype(float)

    at_zero = per_key_transition_f1(y_true, y_prob, threshold=0.5, collar=0)
    assert at_zero["left"]["event"]["f1"] == pytest.approx(0.0)
    assert at_zero["left"]["event"]["n_true"] == 4
    assert at_zero["left"]["event"]["n_pred"] == 4

    at_one = per_key_transition_f1(y_true, y_prob, threshold=0.5, collar=1)
    assert at_one["left"]["event"]["f1"] == pytest.approx(1.0)

    # Keys with no events on either side are undefined, not zero.
    assert np.isnan(at_zero["grab"]["event"]["f1"])


def test_transition_f1_hand_arithmetic_and_one_to_one_matching() -> None:
    y_true, y_prob = _empty_case(50)
    jump = KEY_ORDER.index("jump")
    y_true[10:15, jump] = True   # onset 10, offset 15
    y_true[20:25, jump] = True   # onset 20, offset 25
    pred = np.zeros(50, dtype=bool)
    pred[12] = True              # onset 12, offset 13
    pred[14:16] = True           # onset 14, offset 16
    pred[40:42] = True           # onset 40, offset 42
    y_prob[:, jump] = pred.astype(float)

    result = per_key_transition_f1(y_true, y_prob, threshold=0.5, collar=2)
    # Onsets: true {10,20} vs pred {12,14,40}: one-to-one leaves only 10↔12.
    assert result["jump"]["onset"]["n_matched"] == 1
    assert result["jump"]["onset"]["f1"] == pytest.approx(0.4)
    # Offsets: true {15,25} vs pred {13,16,42}: only 15↔13 (|Δ|=2).
    assert result["jump"]["offset"]["n_matched"] == 1
    assert result["jump"]["offset"]["f1"] == pytest.approx(0.4)
    # Pooled: 2 matched of 4 true + 6 pred.
    assert result["jump"]["event"]["f1"] == pytest.approx(0.4)
    assert result["jump"]["event"]["precision"] == pytest.approx(1 / 3)
    assert result["jump"]["event"]["recall"] == pytest.approx(1 / 2)


def test_match_event_counts_greedy_is_one_to_one() -> None:
    assert match_event_counts(np.array([5]), np.array([3, 5]), 2) == 1
    assert match_event_counts(np.array([3, 5]), np.array([5]), 2) == 1
    assert match_event_counts(np.array([]), np.array([1, 2]), 2) == 0


@pytest.mark.parametrize("collar", [1, 2, 4])
def test_transition_matching_never_crosses_segment_boundaries(
    collar: int,
) -> None:
    y_true, y_prob = _empty_case(8)
    left = KEY_ORDER.index("left")
    y_true[3, left] = True
    y_prob[4:, left] = 1.0

    # The events are adjacent in concatenated coordinates but belong to
    # different sessions, so no collar may pair them across the join.
    unbounded = per_key_transition_f1(
        y_true, y_prob, threshold=0.5, collar=collar
    )
    bounded = per_key_transition_f1(
        y_true,
        y_prob,
        threshold=0.5,
        collar=collar,
        boundaries=[4, 4],
    )

    assert unbounded["left"]["onset"]["n_matched"] == 1
    assert bounded["left"]["onset"]["n_true"] == 1
    assert bounded["left"]["onset"]["n_pred"] == 1
    assert bounded["left"]["onset"]["n_matched"] == 0
    assert bounded["left"]["onset"]["f1"] == pytest.approx(0.0)


def test_segment_bounded_matcher_preserves_within_segment_semantics() -> None:
    assert match_event_counts(
        np.array([1, 5]),
        np.array([2, 6]),
        1,
        boundaries=[4, 4],
    ) == 2


def test_active_gating_drops_events_without_fragmenting_holds() -> None:
    y_true, y_prob = _empty_case(20)
    dash = KEY_ORDER.index("dash")
    y_true[5:9, dash] = True     # onset 5, offset 9
    y_prob[:, dash] = y_true[:, dash].astype(float)  # perfect predictor
    active = np.ones(20, dtype=bool)
    active[5] = False            # the onset lands on an inactive frame

    result = per_key_transition_f1(
        y_true, y_prob, threshold=0.5, collar=0, active=active
    )
    # Both true and predicted onsets are gated out; offsets still match.
    assert np.isnan(result["dash"]["onset"]["f1"])
    assert result["dash"]["offset"]["f1"] == pytest.approx(1.0)
    assert result["dash"]["event"]["f1"] == pytest.approx(1.0)

    # Contrast with subsetting, which would have manufactured a fake onset at
    # the first retained frame: gating keeps the stream contiguous.
    assert result["dash"]["event"]["n_true"] == 1


def test_oracle_threshold_recovers_low_confidence_events() -> None:
    y_true, y_prob = _empty_case(30)
    grab = KEY_ORDER.index("grab")
    y_true[10:15, grab] = True
    y_prob[:, grab] = 0.05
    y_prob[10:15, grab] = 0.3    # well-shaped but under-confident

    fixed = per_key_transition_f1(y_true, y_prob, threshold=0.5, collar=0)
    assert fixed["grab"]["event"]["f1"] == pytest.approx(0.0)

    oracle = per_key_transition_f1(y_true, y_prob, threshold=None, collar=0)
    assert oracle["grab"]["event"]["f1"] == pytest.approx(1.0)
    assert oracle["grab"]["threshold"] <= 0.3


def test_score_events_conventions() -> None:
    assert np.isnan(score_events(0, 0, 0)["f1"])
    assert score_events(0, 3, 0)["f1"] == pytest.approx(0.0)
    assert score_events(2, 0, 0)["f1"] == pytest.approx(0.0)


def test_summarize_includes_transition_metrics_with_gate() -> None:
    y_true, y_prob = _empty_case(40)
    left = KEY_ORDER.index("left")
    y_true[10:20, left] = True
    y_prob[:, left] = y_true[:, left].astype(float)
    active = np.ones(40, dtype=bool)

    summary = summarize(y_true, y_prob, boundaries=[20, 20], active=active)
    assert "transition_f1_oracle" in summary
    assert "transition_f1_at_0.5" in summary
    assert set(summary["transition_f1_oracle_collars"]) == {"1", "2", "4"}
    assert summary["transition_f1_at_0.5"]["left"]["event"]["f1"] == pytest.approx(1.0)


def test_summarize_applies_frozen_development_thresholds() -> None:
    y_true, y_prob = _empty_case(40)
    left = KEY_ORDER.index("left")
    y_true[10:20, left] = True
    y_prob[10:20, left] = 0.3
    thresholds = {key: 0.5 for key in KEY_ORDER}
    thresholds["left"] = 0.25

    summary = summarize(
        y_true, y_prob, fixed_transition_thresholds=thresholds
    )
    assert summary["transition_f1_fixed_dev"]["left"]["threshold"] == 0.25
    assert summary["transition_f1_fixed_dev"]["left"]["event"]["f1"] == pytest.approx(1.0)
    assert set(summary["transition_f1_fixed_dev_collars"]) == {"1", "2", "4"}
