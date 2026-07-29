"""Per-key metrics for Badeline predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from data.schema import KEY_ORDER


class CalibrationResult(TypedDict):
    """Reliability-diagram values for one key."""

    bin_edges: list[float]
    bin_confidence: list[float]
    bin_accuracy: list[float]
    bin_count: list[int]
    ece: float


class OnsetTimingResult(TypedDict):
    """Onset matching values for one key."""

    offsets: np.ndarray
    n_true_onsets: int
    n_matched: int


def _validate_arrays(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    probabilities = np.asarray(y_prob)
    key_count = len(KEY_ORDER)

    if truth.ndim != 2 or truth.shape[1] != key_count:
        raise ValueError(f"y_true must have shape [N,{key_count}]")
    if probabilities.ndim != 2 or probabilities.shape[1] != key_count:
        raise ValueError(f"y_prob must have shape [N,{key_count}]")
    if truth.shape != probabilities.shape:
        raise ValueError(
            "y_true and y_prob must have the same shape, "
            f"got {truth.shape} and {probabilities.shape}"
        )
    if truth.shape[0] == 0:
        raise ValueError("y_true and y_prob must contain at least one row")
    if not np.all(np.isin(truth, (False, True))):
        raise ValueError("y_true must contain only boolean or binary values")
    if not np.issubdtype(probabilities.dtype, np.number):
        raise ValueError("y_prob must contain numeric probabilities")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("y_prob must contain only finite probabilities")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("y_prob values must lie in [0, 1]")

    return truth.astype(bool, copy=False), probabilities.astype(float, copy=False)


def _validate_threshold(threshold: float) -> float:
    value = float(threshold)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    return value


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validate_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def per_key_ap(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Return average precision for each canonical key.

    A key absent from ``y_true`` has an undefined score and maps to ``nan``.
    """

    truth, probabilities = _validate_arrays(y_true, y_prob)
    result: dict[str, float] = {}
    for column, key in enumerate(KEY_ORDER):
        if not np.any(truth[:, column]):
            result[key] = float("nan")
        else:
            result[key] = float(
                average_precision_score(
                    truth[:, column], probabilities[:, column]
                )
            )
    return result


def per_key_f1(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Return F1 for each canonical key at ``y_prob >= threshold``.

    A key absent from ``y_true`` has an undefined score and maps to ``nan``.
    """

    truth, probabilities = _validate_arrays(y_true, y_prob)
    cutoff = _validate_threshold(threshold)
    predicted = probabilities >= cutoff

    result: dict[str, float] = {}
    for column, key in enumerate(KEY_ORDER):
        if not np.any(truth[:, column]):
            result[key] = float("nan")
        else:
            result[key] = float(
                f1_score(
                    truth[:, column],
                    predicted[:, column],
                    zero_division=0.0,
                )
            )
    return result


def per_key_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> dict[str, CalibrationResult]:
    """Return equal-width reliability bins and ECE for each canonical key.

    Bins are left-closed and right-open, except that the final bin includes
    probability 1. Empty bins retain a ``nan`` confidence and empirical
    frequency and contribute zero weight to ECE.
    """

    truth, probabilities = _validate_arrays(y_true, y_prob)
    bin_count = _validate_positive_integer(n_bins, "n_bins")
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    edge_values = edges.tolist()
    sample_count = truth.shape[0]

    result: dict[str, CalibrationResult] = {}
    for column, key in enumerate(KEY_ORDER):
        key_probabilities = probabilities[:, column]
        assignments = np.searchsorted(
            edges, key_probabilities, side="right"
        ) - 1
        assignments = np.clip(assignments, 0, bin_count - 1)
        counts = np.bincount(assignments, minlength=bin_count)

        confidence = np.full(bin_count, np.nan, dtype=float)
        empirical_frequency = np.full(bin_count, np.nan, dtype=float)
        ece = 0.0
        for bin_index, count in enumerate(counts):
            if count == 0:
                continue
            members = assignments == bin_index
            confidence[bin_index] = float(np.mean(key_probabilities[members]))
            empirical_frequency[bin_index] = float(
                np.mean(truth[members, column])
            )
            ece += (
                float(count)
                / sample_count
                * abs(
                    empirical_frequency[bin_index] - confidence[bin_index]
                )
            )

        result[key] = {
            "bin_edges": list(edge_values),
            "bin_confidence": confidence.tolist(),
            "bin_accuracy": empirical_frequency.tolist(),
            "bin_count": counts.astype(int, copy=False).tolist(),
            "ece": float(ece),
        }
    return result


def _onsets(active: np.ndarray) -> np.ndarray:
    previous = np.empty_like(active)
    previous[0] = False
    previous[1:] = active[:-1]
    return np.flatnonzero(active & ~previous)


def onset_timing_errors(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    max_lag: int = 30,
) -> dict[str, OnsetTimingResult]:
    """Match each true onset to its nearest predicted onset.

    Each offset is ``predicted frame - true frame``, so positive values mean
    the prediction is late and negative values mean it is early. Matches are
    inclusive of ``max_lag`` in either direction. Frame zero is an onset when
    active, assuming the key was inactive before the sequence began.
    """

    truth, probabilities = _validate_arrays(y_true, y_prob)
    cutoff = _validate_threshold(threshold)
    lag_limit = _validate_nonnegative_integer(max_lag, "max_lag")
    predicted = probabilities >= cutoff

    result: dict[str, OnsetTimingResult] = {}
    for column, key in enumerate(KEY_ORDER):
        true_onsets = _onsets(truth[:, column])
        predicted_onsets = _onsets(predicted[:, column])
        offsets: list[int] = []

        for true_onset in true_onsets:
            if predicted_onsets.size == 0:
                continue
            candidate_offsets = predicted_onsets - true_onset
            nearest_index = int(np.argmin(np.abs(candidate_offsets)))
            nearest_offset = int(candidate_offsets[nearest_index])
            if abs(nearest_offset) <= lag_limit:
                offsets.append(nearest_offset)

        result[key] = {
            "offsets": np.asarray(offsets, dtype=int),
            "n_true_onsets": int(true_onsets.size),
            "n_matched": len(offsets),
        }
    return result


class EventScore(TypedDict):
    """Precision/recall/F1 over matched transition events."""

    f1: float
    precision: float
    recall: float
    n_true: int
    n_pred: int
    n_matched: int


class TransitionF1Result(TypedDict):
    """Event-level scores for one key at one threshold and collar."""

    threshold: float
    collar: int
    onset: EventScore
    offset: EventScore
    event: EventScore


def _segment_bounds(
    n: int, boundaries: Sequence[int] | None
) -> list[tuple[int, int]]:
    if boundaries is None:
        return [(0, n)]
    lengths = [int(v) for v in boundaries]
    if any(length < 1 for length in lengths):
        raise ValueError("segment lengths must be positive")
    if sum(lengths) != n:
        raise ValueError(
            f"segment lengths sum to {sum(lengths)}, expected {n}"
        )
    bounds: list[tuple[int, int]] = []
    start = 0
    for length in lengths:
        bounds.append((start, start + length))
        start += length
    return bounds


def transition_events(
    active: np.ndarray, boundaries: Sequence[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (onset_times, offset_times) for one binary stream.

    Segments (sessions) are independent: no transition is ever manufactured
    across a segment join. Frame zero of a segment counts as an onset when
    active (assumed inactive before the segment); a hold running into the end
    of a segment produces no offset (the release was never observed).
    """

    stream = np.asarray(active).astype(bool)
    if stream.ndim != 1:
        raise ValueError("active must be one-dimensional")
    onsets: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    for start, end in _segment_bounds(len(stream), boundaries):
        segment = stream[start:end]
        previous = np.empty_like(segment)
        previous[0] = False
        previous[1:] = segment[:-1]
        onsets.append(np.flatnonzero(segment & ~previous) + start)
        offsets.append(np.flatnonzero(~segment & previous) + start)
    return (
        np.concatenate(onsets) if onsets else np.empty(0, dtype=int),
        np.concatenate(offsets) if offsets else np.empty(0, dtype=int),
    )


def match_event_counts(
    true_times: np.ndarray,
    pred_times: np.ndarray,
    collar: int,
    *,
    boundaries: Sequence[int] | None = None,
) -> int:
    """One-to-one matches between sorted event-time lists within ±collar.

    Greedy two-pointer matching in time order, which is optimal for interval
    matching on a line. collar 0 demands exact frame equality. When segment
    lengths are supplied, event times are absolute within the concatenated
    stream and matches are restricted to the same segment.
    """

    tolerance = _validate_nonnegative_integer(collar, "collar")

    def match_segment(
        segment_true: np.ndarray, segment_pred: np.ndarray
    ) -> int:
        i = j = matched = 0
        while i < len(segment_true) and j < len(segment_pred):
            delta = int(segment_pred[j]) - int(segment_true[i])
            if abs(delta) <= tolerance:
                matched += 1
                i += 1
                j += 1
            elif delta < -tolerance:
                j += 1
            else:
                i += 1
        return matched

    true = np.asarray(true_times)
    predicted = np.asarray(pred_times)
    if boundaries is None:
        return match_segment(true, predicted)

    lengths = [int(length) for length in boundaries]
    if not lengths:
        raise ValueError("segment lengths must be positive")
    frame_count = sum(lengths)
    bounds = _segment_bounds(frame_count, lengths)
    for name, times in (("true_times", true), ("pred_times", predicted)):
        if times.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if len(times) and (np.any(times < 0) or np.any(times >= frame_count)):
            raise ValueError(f"{name} must lie within the declared segments")

    matched = 0
    for start, end in bounds:
        true_start = int(np.searchsorted(true, start, side="left"))
        true_end = int(np.searchsorted(true, end, side="left"))
        pred_start = int(np.searchsorted(predicted, start, side="left"))
        pred_end = int(np.searchsorted(predicted, end, side="left"))
        matched += match_segment(
            true[true_start:true_end], predicted[pred_start:pred_end]
        )
    return matched


def score_events(n_true: int, n_pred: int, n_matched: int) -> EventScore:
    """F1 over event sets. No events on either side is undefined (nan)."""

    if n_true == 0 and n_pred == 0:
        f1 = float("nan")
    else:
        f1 = 2.0 * n_matched / (n_true + n_pred)
    return {
        "f1": f1,
        "precision": (n_matched / n_pred) if n_pred else 0.0,
        "recall": (n_matched / n_true) if n_true else 0.0,
        "n_true": int(n_true),
        "n_pred": int(n_pred),
        "n_matched": int(n_matched),
    }


def _gate(times: np.ndarray, active: np.ndarray | None) -> np.ndarray:
    if active is None or len(times) == 0:
        return times
    return times[active[times]]


def _transition_f1_single(
    truth: np.ndarray,
    predicted: np.ndarray,
    collar: int,
    boundaries: Sequence[int] | None,
    active: np.ndarray | None,
) -> tuple[EventScore, EventScore, EventScore]:
    true_on, true_off = transition_events(truth, boundaries)
    pred_on, pred_off = transition_events(predicted, boundaries)
    true_on, true_off = _gate(true_on, active), _gate(true_off, active)
    pred_on, pred_off = _gate(pred_on, active), _gate(pred_off, active)
    matched_on = match_event_counts(
        true_on, pred_on, collar, boundaries=boundaries
    )
    matched_off = match_event_counts(
        true_off, pred_off, collar, boundaries=boundaries
    )
    onset = score_events(len(true_on), len(pred_on), matched_on)
    offset = score_events(len(true_off), len(pred_off), matched_off)
    event = score_events(
        len(true_on) + len(true_off),
        len(pred_on) + len(pred_off),
        matched_on + matched_off,
    )
    return onset, offset, event


def _oracle_threshold(
    truth: np.ndarray,
    probabilities: np.ndarray,
    collar: int,
    boundaries: Sequence[int] | None,
    active: np.ndarray | None,
) -> float:
    """Per-key threshold maximizing pooled event F1 on this surface.

    An oracle (post-hoc) choice: it upper-bounds deployable event detection
    and is applied identically to every model, which is what makes arms with
    different calibration comparable. Reports quoting it must say "oracle
    threshold". Ties break toward the higher threshold (fewer predictions).
    """

    quantiles = np.unique(
        np.quantile(probabilities, np.linspace(0.005, 0.995, 199))
    )
    candidates = np.unique(np.concatenate([quantiles, [0.5]]))
    best_threshold, best_f1 = 0.5, float("-inf")
    for threshold in candidates:
        _, _, event = _transition_f1_single(
            truth, probabilities >= threshold, collar, boundaries, active
        )
        f1 = event["f1"]
        if np.isnan(f1):
            continue
        if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
            best_f1, best_threshold = f1, float(threshold)
    return best_threshold


def per_key_transition_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float | Mapping[str, float] | None = 0.5,
    collar: int = 0,
    boundaries: Sequence[int] | None = None,
    active: np.ndarray | None = None,
) -> dict[str, TransitionF1Result]:
    """Transition-event (onset and offset) F1 per key.

    The primary metric per brief v3.1: per-frame scores at 60Hz are
    autocorrelation-dominated (persistence reaches 0.912 per-frame AP), while
    at collar 0 a persistence predictor scores exactly zero here — its every
    event is an echo, one frame late. Loose collars re-admit that shortcut,
    so collar 0 is the primary setting and wider collars are sensitivity.

    ``threshold`` may be a float (applied to every key), a per-key mapping,
    or None to choose the oracle per-key threshold on this surface (stated
    as such wherever reported). ``boundaries`` gives per-segment (session)
    lengths so neither events nor tolerant matches span a session join.
    ``active`` gates which event times are scored without fragmenting the
    streams they are computed on.
    """

    truth, probabilities = _validate_arrays(y_true, y_prob)
    tolerance = _validate_nonnegative_integer(collar, "collar")
    bounds_check = _segment_bounds(truth.shape[0], boundaries)
    del bounds_check
    gate = None
    if active is not None:
        gate = np.asarray(active).astype(bool)
        if gate.shape != (truth.shape[0],):
            raise ValueError("active must have shape [N]")

    result: dict[str, TransitionF1Result] = {}
    for column, key in enumerate(KEY_ORDER):
        key_truth = truth[:, column]
        key_prob = probabilities[:, column]
        if threshold is None:
            cutoff = _oracle_threshold(
                key_truth, key_prob, tolerance, boundaries, gate
            )
        elif isinstance(threshold, Mapping):
            cutoff = _validate_threshold(threshold[key])
        else:
            cutoff = _validate_threshold(threshold)
        onset, offset, event = _transition_f1_single(
            key_truth, key_prob >= cutoff, tolerance, boundaries, gate
        )
        result[key] = {
            "threshold": float(cutoff),
            "collar": tolerance,
            "onset": onset,
            "offset": offset,
            "event": event,
        }
    return result


def summarize(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    boundaries: Sequence[int] | None = None,
    active: np.ndarray | None = None,
    fixed_transition_thresholds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Return the complete set of per-key metrics without aggregate accuracy.

    Per-frame metrics (AP, F1, calibration, onset timing) are computed on the
    ``active`` subset when a gate is given, matching the historical
    input_active_only surface. Transition-event metrics are computed on the
    full contiguous streams with ``active`` gating event times — subsetting
    would fragment holds into fictitious transitions.
    """

    truth, probabilities = _validate_arrays(y_true, y_prob)
    if active is not None:
        gate = np.asarray(active).astype(bool)
        frame_truth, frame_prob = truth[gate], probabilities[gate]
    else:
        frame_truth, frame_prob = truth, probabilities

    oracle = per_key_transition_f1(
        truth, probabilities, threshold=None, collar=0,
        boundaries=boundaries, active=active,
    )
    oracle_thresholds = {key: oracle[key]["threshold"] for key in oracle}
    collar_sensitivity = {
        str(collar): per_key_transition_f1(
            truth, probabilities, threshold=oracle_thresholds,
            collar=collar, boundaries=boundaries, active=active,
        )
        for collar in (1, 2, 4)
    }
    result: dict[str, object] = {
        "per_key_ap": per_key_ap(frame_truth, frame_prob),
        "per_key_f1": per_key_f1(frame_truth, frame_prob),
        "per_key_calibration": per_key_calibration(frame_truth, frame_prob),
        "onset_timing_errors": onset_timing_errors(frame_truth, frame_prob),
        "transition_f1_at_0.5": per_key_transition_f1(
            truth, probabilities, threshold=0.5, collar=0,
            boundaries=boundaries, active=active,
        ),
        "transition_f1_oracle": oracle,
        "transition_f1_oracle_collars": collar_sensitivity,
    }
    if fixed_transition_thresholds is not None:
        result["transition_f1_fixed_dev"] = per_key_transition_f1(
            truth, probabilities,
            threshold=fixed_transition_thresholds,
            collar=0, boundaries=boundaries, active=active,
        )
        result["transition_f1_fixed_dev_collars"] = {
            str(collar): per_key_transition_f1(
                truth, probabilities,
                threshold=fixed_transition_thresholds,
                collar=collar, boundaries=boundaries, active=active,
            )
            for collar in (1, 2, 4)
        }
    return result
