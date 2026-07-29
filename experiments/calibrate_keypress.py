"""Fit and evaluate per-key affine calibration for IDM prediction sidecars.

The calibrator is a monotone, per-key Platt transform::

    calibrated_logit = scale * raw_logit + bias,  scale > 0

Parameters are fit with unweighted binary negative log likelihood on explicitly
named whole streams.  A separate set of whole streams is then scored without
refitting.  Additional transfer sidecars may be scored with the frozen
parameters, but their labels never enter the fit.

This tool intentionally does not select a threshold for F1 or accuracy.  Both
the raw and calibrated policies use probability >= 0.5, and the report records
the equivalent raw-probability threshold induced by each affine transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from badeline.metrics import per_key_transition_f1
from data.schema import KEY_ORDER


_PROBABILITY_EPSILON = np.finfo(np.float64).eps
_EQUAL_MASS_BIN_COUNT = 15


@dataclass(frozen=True)
class Sidecar:
    """Validated prediction arrays and their stream metadata."""

    path: Path
    truth: np.ndarray
    probability: np.ndarray
    active: np.ndarray
    session_lengths: np.ndarray
    session_ids: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sigmoid(values: np.ndarray | float) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probability, dtype=np.float64),
        _PROBABILITY_EPSILON,
        1.0 - _PROBABILITY_EPSILON,
    )
    return np.log(clipped) - np.log1p(-clipped)


def load_sidecar(path: Path) -> Sidecar:
    """Load one sidecar and fail closed on malformed arrays or metadata."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "y_true",
            "y_prob",
            "input_active",
            "session_lengths",
            "session_ids",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{source}: missing arrays: {sorted(missing)}")
        truth = np.asarray(archive["y_true"])
        probability = np.asarray(archive["y_prob"])
        active = np.asarray(archive["input_active"])
        lengths = np.asarray(archive["session_lengths"])
        ids = np.asarray(archive["session_ids"])

    expected_shape = (truth.shape[0], len(KEY_ORDER))
    if truth.ndim != 2 or truth.shape != expected_shape:
        raise ValueError(
            f"{source}: y_true must have shape [N,{len(KEY_ORDER)}]"
        )
    if probability.shape != truth.shape:
        raise ValueError(f"{source}: y_prob shape does not match y_true")
    if not np.all(np.isin(truth, (0, 1))):
        raise ValueError(f"{source}: y_true is not binary")
    if not np.all(np.isfinite(probability)):
        raise ValueError(f"{source}: y_prob contains non-finite values")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError(f"{source}: y_prob lies outside [0, 1]")
    if active.shape != (truth.shape[0],) or not np.all(np.isin(active, (0, 1))):
        raise ValueError(f"{source}: input_active must be binary with shape [N]")
    if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError(f"{source}: session_lengths must be a one-dimensional integer array")
    if not len(lengths) or np.any(lengths <= 0) or int(lengths.sum()) != len(truth):
        raise ValueError(f"{source}: session_lengths must be positive and sum to N")
    if ids.ndim != 1 or len(ids) != len(lengths):
        raise ValueError(f"{source}: session_ids must have one entry per stream")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{source}: session_ids must be unique")

    return Sidecar(
        path=source,
        truth=truth.astype(bool, copy=False),
        probability=probability.astype(np.float64, copy=False),
        active=active.astype(bool, copy=False),
        session_lengths=lengths.astype(np.int64, copy=False),
        session_ids=ids.astype(str, copy=False),
    )


def _stream_slices(lengths: np.ndarray) -> list[slice]:
    ends = np.cumsum(lengths, dtype=np.int64)
    starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
    return [slice(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def select_streams(sidecar: Sidecar, session_ids: list[str]) -> Sidecar:
    """Return a compact sidecar containing exactly the named whole streams."""

    if not session_ids:
        raise ValueError("at least one session ID is required")
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("session ID selection contains duplicates")
    index = {session_id: i for i, session_id in enumerate(sidecar.session_ids.tolist())}
    unknown = sorted(set(session_ids).difference(index))
    if unknown:
        raise ValueError(f"{sidecar.path}: unknown session IDs: {unknown}")

    slices = _stream_slices(sidecar.session_lengths)
    chosen = [index[session_id] for session_id in session_ids]
    return Sidecar(
        path=sidecar.path,
        truth=np.concatenate([sidecar.truth[slices[i]] for i in chosen]),
        probability=np.concatenate([sidecar.probability[slices[i]] for i in chosen]),
        active=np.concatenate([sidecar.active[slices[i]] for i in chosen]),
        session_lengths=sidecar.session_lengths[chosen],
        session_ids=sidecar.session_ids[chosen],
    )


def _binary_nll(logits: np.ndarray, truth: np.ndarray) -> float:
    labels = np.asarray(truth, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def fit_affine_key(
    truth: np.ndarray,
    probability: np.ndarray,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
    minimum_scale: float = 1e-8,
) -> dict[str, float | int | bool]:
    """Fit a positive-scale affine logit transform with damped Newton steps."""

    labels = np.asarray(truth, dtype=np.float64)
    if labels.ndim != 1 or not len(labels) or not np.all(np.isin(labels, (0, 1))):
        raise ValueError("truth must be a non-empty binary vector")
    if labels.min() == labels.max():
        raise ValueError("calibration labels must contain both classes")
    logits = _logit(np.asarray(probability, dtype=np.float64))
    if logits.shape != labels.shape:
        raise ValueError("probability must have the same shape as truth")

    scale = 1.0
    bias = 0.0
    initial_loss = _binary_nll(logits, labels)
    loss = initial_loss
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        transformed = scale * logits + bias
        calibrated = _stable_sigmoid(transformed)
        residual = calibrated - labels
        variance = calibrated * (1.0 - calibrated)
        gradient = np.asarray(
            [np.mean(residual * logits), np.mean(residual)], dtype=np.float64
        )
        if float(np.max(np.abs(gradient))) <= tolerance:
            converged = True
            break
        hessian = np.asarray(
            [
                [np.mean(variance * logits * logits), np.mean(variance * logits)],
                [np.mean(variance * logits), np.mean(variance)],
            ],
            dtype=np.float64,
        )
        hessian.flat[::3] += 1e-12
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise ValueError("affine calibration Hessian is singular") from error

        accepted = False
        step_size = 1.0
        previous_loss = loss
        for _ in range(60):
            candidate_scale = scale - step_size * float(step[0])
            candidate_bias = bias - step_size * float(step[1])
            if candidate_scale >= minimum_scale:
                candidate_loss = _binary_nll(
                    candidate_scale * logits + candidate_bias, labels
                )
                if candidate_loss <= loss:
                    scale = candidate_scale
                    bias = candidate_bias
                    loss = candidate_loss
                    accepted = True
                    break
            step_size *= 0.5
        if not accepted:
            converged = float(np.max(np.abs(gradient))) <= 10 * tolerance
            break
        if abs(previous_loss - loss) <= tolerance * max(1.0, abs(previous_loss)):
            converged = True
            break

    equivalent_threshold = float(_stable_sigmoid(-bias / scale))
    required_raw_logit = float(-bias / scale)
    clipped_logit_minimum = float(_logit(np.asarray([0.0]))[0])
    clipped_logit_maximum = float(_logit(np.asarray([1.0]))[0])
    return {
        "scale": float(scale),
        "bias": float(bias),
        "calibration_positive_rate": float(labels.mean()),
        "required_raw_logit_at_calibrated_half": required_raw_logit,
        "equivalent_raw_probability_threshold": equivalent_threshold,
        "calibrated_half_reachable_with_clipped_float64_logit": bool(
            clipped_logit_minimum <= required_raw_logit <= clipped_logit_maximum
        ),
        "initial_unweighted_nll": initial_loss,
        "final_unweighted_nll": float(loss),
        "iterations": iterations,
        "converged": converged,
    }


def fit_affine_calibrators(sidecar: Sidecar) -> dict[str, dict[str, float | int | bool]]:
    """Fit one affine calibrator per key on active rows only."""

    if not np.any(sidecar.active):
        raise ValueError("calibration surface has no active rows")
    truth = sidecar.truth[sidecar.active]
    probability = sidecar.probability[sidecar.active]
    return {
        key: fit_affine_key(truth[:, column], probability[:, column])
        for column, key in enumerate(KEY_ORDER)
    }


def apply_affine_calibrators(
    probability: np.ndarray,
    parameters: dict[str, dict[str, float | int | bool]],
) -> np.ndarray:
    """Apply frozen positive-scale calibrators without changing key ranking."""

    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(KEY_ORDER):
        raise ValueError(f"probability must have shape [N,{len(KEY_ORDER)}]")
    logits = _logit(values)
    calibrated = np.empty_like(logits)
    for column, key in enumerate(KEY_ORDER):
        if key not in parameters:
            raise ValueError(f"missing calibrator for key {key}")
        scale = float(parameters[key]["scale"])
        bias = float(parameters[key]["bias"])
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(bias):
            raise ValueError(f"invalid calibrator for key {key}")
        calibrated[:, column] = _stable_sigmoid(scale * logits[:, column] + bias)
    return calibrated


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _equal_mass_ece(
    truth: np.ndarray,
    probability: np.ndarray,
    *,
    bin_count: int = _EQUAL_MASS_BIN_COUNT,
) -> dict[str, Any]:
    """Return equal-mass reliability bins and expected calibration error."""

    labels = np.asarray(truth, dtype=bool)
    probabilities = np.asarray(probability, dtype=np.float64)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels):
        raise ValueError("truth and probability must be non-empty matching vectors")
    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    order = np.argsort(probabilities, kind="stable")
    groups = [group for group in np.array_split(order, min(bin_count, len(order))) if len(group)]
    bins: list[dict[str, float | int]] = []
    ece = 0.0
    for group in groups:
        confidence = float(probabilities[group].mean())
        positive_rate = float(labels[group].mean())
        gap = abs(confidence - positive_rate)
        count = int(len(group))
        ece += count / len(labels) * gap
        bins.append(
            {
                "count": count,
                "minimum_probability": float(probabilities[group].min()),
                "maximum_probability": float(probabilities[group].max()),
                "mean_probability": confidence,
                "observed_positive_rate": positive_rate,
                "absolute_gap": gap,
            }
        )
    return {
        "kind": "equal_mass",
        "requested_bin_count": int(bin_count),
        "nonempty_bin_count": len(bins),
        "bin_counts": [int(entry["count"]) for entry in bins],
        "ece": float(ece),
        "bins": bins,
    }


def _event_score_for_json(score: dict[str, float | int]) -> dict[str, float | int | None]:
    return {
        name: (None if isinstance(value, float) and not np.isfinite(value) else value)
        for name, value in score.items()
    }


def _event_summary(
    sidecar: Sidecar,
    probability: np.ndarray,
    persistence_probability: np.ndarray,
) -> dict[str, Any]:
    """Score model and persistence transitions without crossing streams."""

    boundaries = sidecar.session_lengths.tolist()

    def score(values: np.ndarray, collar: int) -> dict[str, Any]:
        raw = per_key_transition_f1(
            sidecar.truth,
            values,
            threshold=0.5,
            collar=collar,
            boundaries=boundaries,
            active=sidecar.active,
        )
        per_key: dict[str, Any] = {}
        onset_values: list[float] = []
        release_values: list[float] = []
        combined_values: list[float] = []
        for key in KEY_ORDER:
            onset = _event_score_for_json(raw[key]["onset"])
            release = _event_score_for_json(raw[key]["offset"])
            combined = _event_score_for_json(raw[key]["event"])
            per_key[key] = {
                "threshold": float(raw[key]["threshold"]),
                "onset": onset,
                "release": release,
                "onset_plus_release": combined,
            }
            if onset["f1"] is not None:
                onset_values.append(float(onset["f1"]))
            if release["f1"] is not None:
                release_values.append(float(release["f1"]))
            if combined["f1"] is not None:
                combined_values.append(float(combined["f1"]))
        return {
            "collar_frames": collar,
            "macro_onset_f1": float(np.mean(onset_values)) if onset_values else None,
            "macro_release_f1": (
                float(np.mean(release_values)) if release_values else None
            ),
            "macro_onset_plus_release_f1": (
                float(np.mean(combined_values)) if combined_values else None
            ),
            "per_key": per_key,
        }

    return {
        "stream_boundary_policy": "stored session_lengths; no event crosses a stream",
        "active_policy": "events formed on full streams, then gated at input_active event times",
        "model": {
            "exact": score(probability, 0),
            "plus_or_minus_2_frames": score(probability, 2),
        },
        "persistence_baseline": {
            "definition": "copy previous true key vector within each stream; start released",
            "exact": score(persistence_probability, 0),
            "plus_or_minus_2_frames": score(persistence_probability, 2),
        },
    }


def _probability_diagnostics(
    sidecar: Sidecar,
    calibrated: np.ndarray,
    parameters: dict[str, dict[str, float | int | bool]],
) -> dict[str, Any]:
    """Count logit clipping and output saturation on active evaluation rows."""

    raw = sidecar.probability[sidecar.active]
    calibrated_active = calibrated[sidecar.active]
    clipped_logit_minimum = float(_logit(np.asarray([0.0]))[0])
    clipped_logit_maximum = float(_logit(np.asarray([1.0]))[0])
    per_key: dict[str, Any] = {}
    caveats: list[dict[str, Any]] = []
    for column, key in enumerate(KEY_ORDER):
        raw_key = raw[:, column]
        calibrated_key = calibrated_active[:, column]
        parameter = parameters[key]
        reachable = bool(
            parameter["calibrated_half_reachable_with_clipped_float64_logit"]
        )
        entry = {
            "raw_exact_zero_count": int(np.sum(raw_key == 0.0)),
            "raw_exact_one_count": int(np.sum(raw_key == 1.0)),
            "raw_low_logit_clip_count": int(np.sum(raw_key <= _PROBABILITY_EPSILON)),
            "raw_high_logit_clip_count": int(
                np.sum(raw_key >= 1.0 - _PROBABILITY_EPSILON)
            ),
            "raw_minimum_probability": float(raw_key.min()),
            "raw_maximum_probability": float(raw_key.max()),
            "calibrated_exact_zero_count": int(np.sum(calibrated_key == 0.0)),
            "calibrated_exact_one_count": int(np.sum(calibrated_key == 1.0)),
            "calibrated_minimum_probability": float(calibrated_key.min()),
            "calibrated_maximum_probability": float(calibrated_key.max()),
            "calibrated_positive_count_at_0.5": int(np.sum(calibrated_key >= 0.5)),
            "required_raw_logit_at_calibrated_half": float(
                parameter["required_raw_logit_at_calibrated_half"]
            ),
            "calibrated_half_reachable_with_clipped_float64_logit": reachable,
        }
        per_key[key] = entry
        if not reachable:
            caveats.append(
                {
                    "key": key,
                    "required_raw_logit": entry[
                        "required_raw_logit_at_calibrated_half"
                    ],
                    "maximum_logit_after_clipping": clipped_logit_maximum,
                    "effect": (
                        "calibrated probability cannot reach 0.5 under the declared "
                        "float64 clipping policy"
                    ),
                }
            )
    return {
        "scope": "input_active rows",
        "raw_probability_logit_clip_epsilon": _PROBABILITY_EPSILON,
        "clipped_logit_minimum": clipped_logit_minimum,
        "clipped_logit_maximum": clipped_logit_maximum,
        "aggregate_counts": {
            field: int(sum(int(entry[field]) for entry in per_key.values()))
            for field in (
                "raw_exact_zero_count",
                "raw_exact_one_count",
                "raw_low_logit_clip_count",
                "raw_high_logit_clip_count",
                "calibrated_exact_zero_count",
                "calibrated_exact_one_count",
            )
        },
        "per_key": per_key,
        "structurally_unreachable_half_thresholds": caveats,
    }


def _state_metrics(
    sidecar: Sidecar,
    probability: np.ndarray,
    *,
    prior_rates: dict[str, float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Score state decisions and same-support released/persistence baselines."""

    predicted_full = probability >= threshold
    persistence_full = np.zeros_like(sidecar.truth, dtype=bool)
    for stream_slice in _stream_slices(sidecar.session_lengths):
        start, end = stream_slice.start, stream_slice.stop
        persistence_full[start + 1 : end] = sidecar.truth[start : end - 1]

    truth = sidecar.truth[sidecar.active]
    probabilities = probability[sidecar.active]
    predicted = predicted_full[sidecar.active]
    persistence = persistence_full[sidecar.active]
    if not len(truth):
        raise ValueError("evaluation surface has no active rows")

    released = np.zeros_like(truth, dtype=bool)
    correct = predicted == truth
    clipped_probability = np.clip(
        probabilities, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON
    )
    truth_float = truth.astype(np.float64, copy=False)
    binary_cross_entropy = float(
        np.mean(
            -truth_float * np.log(clipped_probability)
            - (1.0 - truth_float) * np.log1p(-clipped_probability)
        )
    )
    brier_score = float(np.mean((probabilities - truth_float) ** 2))
    per_key: dict[str, dict[str, Any]] = {}
    per_key_ece: dict[str, Any] = {}
    for column, key in enumerate(KEY_ORDER):
        key_truth = truth[:, column]
        key_truth_float = truth_float[:, column]
        key_probability = probabilities[:, column]
        key_predicted = predicted[:, column]
        true_positive = int(np.sum(key_truth & key_predicted))
        false_positive = int(np.sum(~key_truth & key_predicted))
        false_negative = int(np.sum(key_truth & ~key_predicted))
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
        clipped_key_probability = clipped_probability[:, column]
        key_bce = float(
            np.mean(
                -key_truth_float * np.log(clipped_key_probability)
                - (1.0 - key_truth_float) * np.log1p(-clipped_key_probability)
            )
        )
        key_brier = float(np.mean((key_probability - key_truth_float) ** 2))
        if key not in prior_rates:
            raise ValueError(f"missing calibration prior rate for key {key}")
        prior_rate = float(prior_rates[key])
        if not 0.0 < prior_rate < 1.0:
            raise ValueError(f"calibration prior rate for {key} must lie in (0, 1)")
        prior_bce = float(
            np.mean(
                -key_truth_float * np.log(prior_rate)
                - (1.0 - key_truth_float) * np.log1p(-prior_rate)
            )
        )
        prior_brier = float(np.mean((prior_rate - key_truth_float) ** 2))
        per_key[key] = {
            "accuracy": float(correct[:, column].mean()),
            "average_precision": float(
                average_precision_score(key_truth, key_probability)
            ),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "truth_positive_rate": float(key_truth.mean()),
            "predicted_positive_rate": float(key_predicted.mean()),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "unweighted_binary_cross_entropy": key_bce,
            "brier_score": key_brier,
            "calibration_prior_rate_baseline": {
                "prior_positive_rate": prior_rate,
                "binary_cross_entropy": prior_bce,
                "brier_score": prior_brier,
            },
        }
        per_key_ece[key] = _equal_mass_ece(key_truth, key_probability)

    prior_bce = float(
        np.mean(
            [
                entry["calibration_prior_rate_baseline"]["binary_cross_entropy"]
                for entry in per_key.values()
            ]
        )
    )
    prior_brier = float(
        np.mean(
            [
                entry["calibration_prior_rate_baseline"]["brier_score"]
                for entry in per_key.values()
            ]
        )
    )
    event_metrics = _event_summary(
        sidecar,
        probability,
        persistence_full.astype(np.float64),
    )

    return {
        "threshold": threshold,
        "frames": int(len(truth)),
        "streams": int(len(sidecar.session_lengths)),
        "key_state_micro_accuracy": float(correct.mean()),
        "joint_exact_match_accuracy": float(correct.all(axis=1).mean()),
        "macro_average_precision": float(
            np.mean([entry["average_precision"] for entry in per_key.values()])
        ),
        "macro_state_f1": float(np.mean([entry["f1"] for entry in per_key.values()])),
        "macro_precision": float(
            np.mean([entry["precision"] for entry in per_key.values()])
        ),
        "macro_recall": float(np.mean([entry["recall"] for entry in per_key.values()])),
        "unweighted_binary_cross_entropy": binary_cross_entropy,
        "brier_score": brier_score,
        "equal_mass_expected_calibration_error": {
            "bin_count": _EQUAL_MASS_BIN_COUNT,
            "macro_ece": float(
                np.mean([entry["ece"] for entry in per_key_ece.values()])
            ),
            "per_key": per_key_ece,
        },
        "calibration_prior_rate_baseline": {
            "source": "calibration-only active rows",
            "unweighted_binary_cross_entropy": prior_bce,
            "brier_score": prior_brier,
        },
        "probability_skill_scores": {
            "definition": "1 - model_loss / calibration_prior_rate_baseline_loss",
            "binary_cross_entropy": float(1.0 - binary_cross_entropy / prior_bce),
            "brier": float(1.0 - brier_score / prior_brier),
        },
        "truth_positive_rate": float(truth.mean()),
        "predicted_positive_rate": float(predicted.mean()),
        "always_released_key_state_micro_accuracy": float((released == truth).mean()),
        "always_released_joint_exact_match_accuracy": float(
            (released == truth).all(axis=1).mean()
        ),
        "persistence_key_state_micro_accuracy": float((persistence == truth).mean()),
        "persistence_joint_exact_match_accuracy": float(
            (persistence == truth).all(axis=1).mean()
        ),
        "transition_event_f1": event_metrics,
        "per_key": per_key,
    }


def _surface_report(
    sidecar: Sidecar,
    parameters: dict[str, dict[str, float | int | bool]],
    *,
    role: str,
    labels_used_for_fit: bool,
) -> dict[str, Any]:
    calibrated = apply_affine_calibrators(sidecar.probability, parameters)
    prior_rates = {
        key: float(parameters[key]["calibration_positive_rate"])
        for key in KEY_ORDER
    }
    before = _state_metrics(
        sidecar,
        sidecar.probability,
        prior_rates=prior_rates,
    )
    after = _state_metrics(
        sidecar,
        calibrated,
        prior_rates=prior_rates,
    )
    per_key_drift = {
        key: float(
            after["per_key"][key]["average_precision"]
            - before["per_key"][key]["average_precision"]
        )
        for key in KEY_ORDER
    }
    return {
        "role": role,
        "labels_used_for_fit": labels_used_for_fit,
        "source_sidecar": str(sidecar.path),
        "source_sidecar_sha256": _sha256(sidecar.path),
        "session_ids": sidecar.session_ids.tolist(),
        "before_raw_probability_at_0.5": before,
        "after_affine_probability_at_0.5": after,
        "probability_clipping_and_saturation": _probability_diagnostics(
            sidecar,
            calibrated,
            parameters,
        ),
        "average_precision_invariance": {
            "reason": "positive affine scale is strictly monotone in the raw logit",
            "per_key_difference": per_key_drift,
            "maximum_absolute_difference": float(
                max(abs(value) for value in per_key_drift.values())
            ),
        },
    }


def load_roles(path: Path) -> dict[str, Any]:
    """Load a pinned whole-stream calibration/evaluation role declaration."""

    roles = json.loads(Path(path).read_text())
    required = {"calibration_session_ids", "evaluation_session_ids"}
    missing = required.difference(roles)
    if missing:
        raise ValueError(f"{path}: missing role fields: {sorted(missing)}")
    calibration_ids = [str(value) for value in roles["calibration_session_ids"]]
    evaluation_ids = [str(value) for value in roles["evaluation_session_ids"]]
    if not calibration_ids or not evaluation_ids:
        raise ValueError("both calibration and evaluation roles must be non-empty")
    overlap = sorted(set(calibration_ids).intersection(evaluation_ids))
    if overlap:
        raise ValueError(f"calibration and evaluation roles overlap: {overlap}")
    return roles


def calibrate_sidecar(
    fit_sidecar_path: Path,
    roles_path: Path,
    *,
    transfers: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Fit on named calibration streams and score disjoint/transfer surfaces."""

    roles = load_roles(roles_path)
    source = load_sidecar(fit_sidecar_path)
    calibration_ids = [str(value) for value in roles["calibration_session_ids"]]
    evaluation_ids = [str(value) for value in roles["evaluation_session_ids"]]
    calibration = select_streams(source, calibration_ids)
    evaluation = select_streams(source, evaluation_ids)
    parameters = fit_affine_calibrators(calibration)
    if not all(bool(parameters[key]["converged"]) for key in KEY_ORDER):
        unconverged = [key for key in KEY_ORDER if not parameters[key]["converged"]]
        raise RuntimeError(f"affine calibration did not converge for: {unconverged}")

    report: dict[str, Any] = {
        "schema_version": 2,
        "method": {
            "name": "per-key positive-scale affine logit calibration",
            "formula": "sigmoid(scale[key] * logit(raw_probability) + bias[key])",
            "fit_objective": "unweighted binary negative log likelihood",
            "decision_rule": "calibrated_probability >= 0.5",
            "threshold_selection": "none; 0.5 is fixed before evaluation",
            "active_rows_only_for_fit": True,
            "whole_stream_role_boundary": True,
            "reliability_metric": (
                f"per-key equal-mass ECE with {_EQUAL_MASS_BIN_COUNT} bins"
            ),
            "logit_probability_clip_epsilon": _PROBABILITY_EPSILON,
            "transition_event_metric": (
                "badeline.metrics.per_key_transition_f1 with stored session_lengths; "
                "boundary-safe tolerant matcher introduced by commit 98f5a42"
            ),
        },
        "provenance": {
            "roles_file": str(roles_path),
            "roles_file_sha256": _sha256(roles_path),
            "surface": roles.get("surface"),
            "policy": roles.get("policy"),
            "calibration_session_ids": calibration_ids,
            "evaluation_session_ids": evaluation_ids,
            "b1_labels_used_for_fit": False,
            "untouched_test_used_for_fit_or_scoring": False,
        },
        "fit": {
            "source_sidecar": str(source.path),
            "source_sidecar_sha256": _sha256(source.path),
            "active_frames": int(calibration.active.sum()),
            "session_ids": calibration_ids,
            "parameters": parameters,
        },
        "surfaces": {
            "mapped_y4n_disjoint_evaluation": _surface_report(
                evaluation,
                parameters,
                role="disjoint whole-stream mapped-label development evaluation",
                labels_used_for_fit=False,
            )
        },
    }
    for name, transfer_path in (transfers or {}).items():
        if name in report["surfaces"]:
            raise ValueError(f"duplicate surface name: {name}")
        transfer = load_sidecar(transfer_path)
        report["surfaces"][name] = _surface_report(
            transfer,
            parameters,
            role="frozen-parameter transfer evaluation",
            labels_used_for_fit=False,
        )
    return report


def _parse_transfer(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError("--transfer must be NAME=PATH")
        if name in result:
            raise ValueError(f"duplicate transfer name: {name}")
        result[name] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fit_sidecar", type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument(
        "--transfer",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="score a sidecar with frozen parameters; labels never enter fitting",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = calibrate_sidecar(
        args.fit_sidecar,
        args.roles,
        transfers=_parse_transfer(args.transfer),
    )
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)


if __name__ == "__main__":
    main()
