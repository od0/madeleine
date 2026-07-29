"""Boundary-safe evaluation for the onset/release/state IDM.

The direct state head is reported on the same target support and through the
same metrics as :mod:`badeline.eval`.  The deterministic latch is decoded once
per contiguous engine stream, never once per inference chunk, so changing the
chunk size cannot reset held state or manufacture transitions.  Explicit event
heads retain their own probabilities and sparse transition targets in the
prediction sidecar for threshold-free rescoring.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from badeline.event_model import EventLatchIDM
from badeline.metrics import (
    match_event_counts,
    onset_timing_errors,
    per_key_ap,
    per_key_calibration,
    per_key_f1,
    per_key_transition_f1,
    score_events,
    summarize,
)
from badeline.temporal_latch import decode_event_latch
from badeline.train import contiguous_runs, history_block, load_session, target_offset
from data.schema import KEY_ORDER


DEFAULT_DECODE_CONFIG: dict[str, float | int] = {
    "state_threshold": 0.5,
    "onset_threshold": 0.5,
    "release_threshold": 0.5,
    "resync_on_threshold": 0.9,
    "resync_off_threshold": 0.1,
    "resync_patience": 3,
}


def deweight_event_logits(
    logits: torch.Tensor,
    positive_weight: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Undo weighted-BCE prior shift before probability use or decoding.

    For a binary loss with positive ``pos_weight=w``, the optimum logit is
    shifted upward by ``log(w)`` relative to the natural-prevalence log odds.
    Subtracting that fixed training weight preserves ranking/AP while making a
    0.5 event threshold consistent with the unweighted state head.
    """

    weight = torch.as_tensor(
        positive_weight, dtype=logits.dtype, device=logits.device
    )
    if weight.shape != (len(KEY_ORDER),):
        raise ValueError(f"positive weight must have shape [{len(KEY_ORDER)}]")
    if not torch.all(torch.isfinite(weight)) or torch.any(weight < 1):
        raise ValueError("positive weight must be finite and at least one")
    return logits - weight.log()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _bounds(lengths: Sequence[int], frame_count: int) -> list[tuple[int, int]]:
    parsed = [int(length) for length in lengths]
    if not parsed or any(length < 1 for length in parsed):
        raise ValueError("stream lengths must be positive")
    if sum(parsed) != frame_count:
        raise ValueError("stream lengths must sum to the frame count")
    ends = np.cumsum(parsed, dtype=np.int64)
    starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _decision_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    active: np.ndarray,
    lengths: Sequence[int],
) -> dict[str, object]:
    """State-decision metrics and baselines on one explicitly gated surface."""

    if truth.shape != predicted.shape or truth.ndim != 2:
        raise ValueError("truth and predicted state must share shape [N,K]")
    if active.shape != (len(truth),):
        raise ValueError("active must have shape [N]")
    if not np.any(active):
        raise ValueError("decision surface is empty")

    truth = truth.astype(bool, copy=False)
    predicted = predicted.astype(bool, copy=False)
    persistence = np.zeros_like(truth, dtype=bool)
    for start, end in _bounds(lengths, len(truth)):
        persistence[start + 1 : end] = truth[start : end - 1]

    selected_truth = truth[active]
    selected_predicted = predicted[active]
    selected_persistence = persistence[active]
    always_released = np.zeros_like(selected_truth, dtype=bool)

    def accuracy(values: np.ndarray) -> tuple[float, float]:
        correct = values == selected_truth
        return float(correct.mean()), float(correct.all(axis=1).mean())

    micro, joint = accuracy(selected_predicted)
    released_micro, released_joint = accuracy(always_released)
    persistence_micro, persistence_joint = accuracy(selected_persistence)
    per_key: dict[str, dict[str, float | int]] = {}
    for column, key in enumerate(KEY_ORDER):
        target = selected_truth[:, column]
        decision = selected_predicted[:, column]
        tp = int(np.sum(target & decision))
        fp = int(np.sum(~target & decision))
        fn = int(np.sum(target & ~decision))
        denominator = 2 * tp + fp + fn
        per_key[key] = {
            "accuracy": float(np.mean(target == decision)),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "f1": float(2 * tp / denominator) if denominator else float("nan"),
            "prevalence": float(target.mean()),
            "predicted_positive_rate": float(decision.mean()),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }

    powers = (1 << np.arange(len(KEY_ORDER), dtype=np.uint16)).reshape(1, -1)
    truth_codes = np.sum(selected_truth.astype(np.uint16) * powers, axis=1)
    predicted_codes = np.sum(selected_predicted.astype(np.uint16) * powers, axis=1)
    truth_configurations = np.unique(truth_codes)

    return {
        "frames": int(len(selected_truth)),
        "binary_decisions": int(selected_truth.size),
        "key_state_micro_accuracy": micro,
        "joint_exact_match_accuracy": joint,
        "per_key": per_key,
        "truth_joint_configurations": int(len(truth_configurations)),
        "predicted_joint_configurations": int(len(np.unique(predicted_codes))),
        "unseen_joint_configuration_rate": float(
            np.mean(~np.isin(predicted_codes, truth_configurations))
        ),
        "baselines": {
            "always_released": {
                "key_state_micro_accuracy": released_micro,
                "joint_exact_match_accuracy": released_joint,
            },
            "one_frame_persistence": {
                "key_state_micro_accuracy": persistence_micro,
                "joint_exact_match_accuracy": persistence_joint,
            },
        },
    }


def _event_score(
    truth: np.ndarray,
    probability: np.ndarray,
    valid: np.ndarray,
    *,
    lengths: Sequence[int],
    threshold: float,
    collar: int,
    column: int,
) -> dict[str, float | int]:
    predicted = probability[:, column] >= threshold
    target = truth[:, column].astype(bool, copy=False)
    true_times = np.flatnonzero(target & valid)
    pred_times = np.flatnonzero(predicted & valid)
    n_true = len(true_times)
    n_pred = len(pred_times)
    n_matched = match_event_counts(
        true_times, pred_times, collar, boundaries=lengths
    )
    result = score_events(n_true, n_pred, n_matched)
    return {
        "threshold": float(threshold),
        "collar": int(collar),
        **result,
    }


def _oracle_exact_threshold(
    truth: np.ndarray,
    probability: np.ndarray,
    valid: np.ndarray,
    column: int,
) -> float:
    labels = truth[valid, column].astype(bool, copy=False)
    values = probability[valid, column]
    candidates = np.unique(
        np.concatenate(
            (np.quantile(values, np.linspace(0.005, 0.995, 199)), [0.5])
        )
    )
    n_true = int(labels.sum())
    best_threshold = 0.5
    best_f1 = float("-inf")
    for candidate in candidates:
        predicted = values >= candidate
        n_pred = int(predicted.sum())
        matched = int(np.sum(predicted & labels))
        denominator = n_true + n_pred
        f1 = 2.0 * matched / denominator if denominator else float("nan")
        if np.isnan(f1):
            continue
        if f1 > best_f1 or (f1 == best_f1 and candidate > best_threshold):
            best_f1 = f1
            best_threshold = float(candidate)
    return best_threshold


def _event_head_metrics(
    truth: np.ndarray,
    probability: np.ndarray,
    valid: np.ndarray,
    *,
    lengths: Sequence[int],
    allow_oracle_thresholds: bool,
) -> dict[str, object]:
    if truth.shape != probability.shape or truth.shape[1] != len(KEY_ORDER):
        raise ValueError("event truth and probability must share shape [N,7]")
    if valid.shape != (len(truth),) or not np.any(valid):
        raise ValueError("event validity must select at least one frame")
    _bounds(lengths, len(truth))

    def scores(
        thresholds: float | Mapping[str, float], collar: int
    ) -> dict[str, object]:
        return {
            key: _event_score(
                truth,
                probability,
                valid,
                lengths=lengths,
                threshold=(
                    float(thresholds[key])
                    if isinstance(thresholds, Mapping)
                    else float(thresholds)
                ),
                collar=collar,
                column=column,
            )
            for column, key in enumerate(KEY_ORDER)
        }

    result: dict[str, object] = {
        "per_key_ap": per_key_ap(truth[valid], probability[valid]),
        "at_0.5": {
            str(collar): scores(0.5, collar) for collar in (0, 1, 2, 4)
        },
    }
    if allow_oracle_thresholds:
        oracle_thresholds = {
            key: _oracle_exact_threshold(truth, probability, valid, column)
            for column, key in enumerate(KEY_ORDER)
        }
        result["oracle_exact_thresholds"] = oracle_thresholds
        result["at_oracle_exact_threshold"] = {
            str(collar): scores(oracle_thresholds, collar)
            for collar in (0, 1, 2, 4)
        }
    return result


def _summarize_fixed_thresholds(
    truth: np.ndarray,
    probability: np.ndarray,
    *,
    boundaries: Sequence[int],
    active: np.ndarray | None = None,
) -> dict[str, object]:
    """State metrics with no data-fitted threshold of any kind."""

    if active is None:
        frame_truth = truth
        frame_probability = probability
    else:
        frame_truth = truth[active]
        frame_probability = probability[active]
    return {
        "per_key_ap": per_key_ap(frame_truth, frame_probability),
        "per_key_f1": per_key_f1(frame_truth, frame_probability),
        "per_key_calibration": per_key_calibration(
            frame_truth, frame_probability
        ),
        "onset_timing_errors": onset_timing_errors(
            frame_truth, frame_probability
        ),
        "transition_f1_at_0.5": per_key_transition_f1(
            truth,
            probability,
            threshold=0.5,
            collar=0,
            boundaries=boundaries,
            active=active,
        ),
        "transition_f1_at_0.5_collars": {
            str(collar): per_key_transition_f1(
                truth,
                probability,
                threshold=0.5,
                collar=collar,
                boundaries=boundaries,
                active=active,
            )
            for collar in (1, 2, 4)
        },
    }


def evaluate_event_latch(
    model: EventLatchIDM,
    model_config: Mapping[str, object],
    data_dir: Path,
    session_ids: Sequence[str],
    device: str,
    *,
    preds_out: Path | None = None,
    segment_span: int = 512,
    decode_config: Mapping[str, float | int] | None = None,
    onset_positive_weight: Sequence[float] | torch.Tensor | None = None,
    release_positive_weight: Sequence[float] | torch.Tensor | None = None,
    allow_oracle_thresholds: bool = True,
) -> dict[str, object]:
    """Evaluate direct state, explicit events, and the decoded state latch."""

    if segment_span < 1:
        raise ValueError("segment_span must be positive")
    decode = dict(DEFAULT_DECODE_CONFIG)
    if decode_config is not None:
        unknown = set(decode_config).difference(decode)
        if unknown:
            raise ValueError(f"unknown decode settings: {sorted(unknown)}")
        decode.update(decode_config)
    onset_weight = (
        torch.ones(len(KEY_ORDER), dtype=torch.float32)
        if onset_positive_weight is None
        else torch.as_tensor(onset_positive_weight, dtype=torch.float32)
    )
    release_weight = (
        torch.ones(len(KEY_ORDER), dtype=torch.float32)
        if release_positive_weight is None
        else torch.as_tensor(release_positive_weight, dtype=torch.float32)
    )
    # Validate eagerly, before loading data or running GPU inference.
    deweight_event_logits(torch.zeros((1, len(KEY_ORDER))), onset_weight)
    deweight_event_logits(torch.zeros((1, len(KEY_ORDER))), release_weight)

    model.eval().to(device)
    config = dict(model_config)
    window = int(config.get("window", 2))
    frame_stride = int(config.get("frame_stride", 1))
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least one")
    offset = target_offset(window, str(config.get("window_mode", "centered")))
    frame_span = (window - 1) * frame_stride + 1
    input_config = str(config.get("input_config", "pixels"))
    uses_pixels = input_config in ("pixels", "pixels_plus_history")
    uses_history = input_config in ("history", "pixels_plus_history")
    precomputed = bool(config.get("precomputed_features", False))
    history_len = int(config.get("history_len", 8))
    history_gap = int(config.get("history_gap", 0))

    all_true: list[np.ndarray] = []
    all_state: list[np.ndarray] = []
    all_onset: list[np.ndarray] = []
    all_release: list[np.ndarray] = []
    all_onset_raw: list[np.ndarray] = []
    all_release_raw: list[np.ndarray] = []
    all_latch: list[np.ndarray] = []
    all_onset_true: list[np.ndarray] = []
    all_release_true: list[np.ndarray] = []
    all_event_valid: list[np.ndarray] = []
    all_active: list[np.ndarray] = []
    stream_lengths: list[int] = []
    stream_ids: list[str] = []

    for session_id in session_ids:
        arrays = load_session(
            data_dir, session_id, precomputed_features=precomputed
        )
        assert arrays.engine_frame_idx is not None
        assert arrays.input_active is not None
        for run_index, (run_start, run_end) in enumerate(
            contiguous_runs(arrays.engine_frame_idx)
        ):
            n_windows = run_end - run_start - frame_span + 1
            if n_windows < 1:
                continue
            output_chunks: dict[str, list[torch.Tensor]] = {
                "state_logits": [],
                "onset_logits": [],
                "release_logits": [],
            }
            with torch.no_grad():
                for relative_start in range(0, n_windows, segment_span):
                    count = min(segment_span, n_windows - relative_start)
                    start = run_start + relative_start
                    inputs: dict[str, torch.Tensor] = {}
                    if uses_pixels:
                        block = arrays.frames[
                            start : start + count + frame_span - 1
                        ]
                        visual = torch.from_numpy(block.copy()).to(
                            dtype=torch.float32
                        )
                        if precomputed:
                            inputs["features"] = visual.unsqueeze(0).to(device)
                        else:
                            inputs["frames"] = (
                                visual.permute(0, 3, 1, 2)
                                .div_(255.0)
                                .unsqueeze(0)
                                .to(device)
                            )
                    if uses_history:
                        target_indices = [
                            start + step + offset * frame_stride
                            for step in range(count)
                        ]
                        inputs["history"] = (
                            torch.from_numpy(
                                history_block(
                                    arrays.keys,
                                    target_indices,
                                    history_len,
                                    history_gap,
                                    floor=run_start,
                                )
                            )
                            .unsqueeze(0)
                            .to(device)
                        )
                    outputs = model.forward_segment(inputs)
                    for name in output_chunks:
                        logits = outputs[name]
                        if logits.shape != (1, count, len(KEY_ORDER)):
                            raise ValueError(
                                f"{name} returned shape {tuple(logits.shape)}, "
                                f"expected {(1, count, len(KEY_ORDER))}"
                            )
                        output_chunks[name].append(
                            logits[0].to(dtype=torch.float32, device="cpu")
                        )

            logits = {
                name: torch.cat(chunks, dim=0)
                for name, chunks in output_chunks.items()
            }
            onset_logits = deweight_event_logits(
                logits["onset_logits"], onset_weight
            )
            release_logits = deweight_event_logits(
                logits["release_logits"], release_weight
            )
            state_probability = torch.sigmoid(logits["state_logits"]).numpy()
            onset_raw_probability = torch.sigmoid(logits["onset_logits"]).numpy()
            release_raw_probability = torch.sigmoid(
                logits["release_logits"]
            ).numpy()
            onset_probability = torch.sigmoid(onset_logits).numpy()
            release_probability = torch.sigmoid(release_logits).numpy()
            latched = decode_event_latch(
                logits["state_logits"],
                onset_logits,
                release_logits,
                **decode,
            ).numpy()

            target_start = run_start + offset * frame_stride
            target_stop = target_start + n_windows
            state_truth = arrays.keys[target_start:target_stop].astype(bool)
            active = arrays.input_active[target_start:target_stop].astype(bool)
            previous_indices = np.arange(target_start, target_stop) - 1
            event_valid = previous_indices >= run_start
            event_valid &= active
            safe_previous = np.maximum(previous_indices, run_start)
            event_valid &= arrays.input_active[safe_previous].astype(bool)
            previous = arrays.keys[safe_previous].astype(bool)
            onset_truth = (~previous & state_truth) & event_valid[:, None]
            release_truth = (previous & ~state_truth) & event_valid[:, None]

            all_true.append(state_truth)
            all_state.append(state_probability)
            all_onset.append(onset_probability)
            all_release.append(release_probability)
            all_onset_raw.append(onset_raw_probability)
            all_release_raw.append(release_raw_probability)
            all_latch.append(latched)
            all_onset_true.append(onset_truth)
            all_release_true.append(release_truth)
            all_event_valid.append(event_valid)
            all_active.append(active)
            stream_lengths.append(n_windows)
            stream_ids.append(f"{session_id}__stream{run_index:03d}")

    if not all_true:
        raise ValueError("no contiguous evaluation window in requested sessions")

    truth = np.concatenate(all_true)
    state_probability = np.concatenate(all_state)
    onset_probability = np.concatenate(all_onset)
    release_probability = np.concatenate(all_release)
    onset_raw_probability = np.concatenate(all_onset_raw)
    release_raw_probability = np.concatenate(all_release_raw)
    latched = np.concatenate(all_latch)
    onset_truth = np.concatenate(all_onset_true)
    release_truth = np.concatenate(all_release_true)
    event_valid = np.concatenate(all_event_valid)
    active = np.concatenate(all_active)
    lengths = np.asarray(stream_lengths, dtype=np.int64)

    for name, values in (
        ("state_probability", state_probability),
        ("onset_probability", onset_probability),
        ("release_probability", release_probability),
        ("onset_raw_probability", onset_raw_probability),
        ("release_raw_probability", release_raw_probability),
    ):
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{name} is not finite probability data")

    if preds_out is not None:
        preds_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            preds_out,
            y_true=truth.astype(np.uint8),
            y_prob=state_probability.astype(np.float32),
            y_latch=latched.astype(np.uint8),
            onset_true=onset_truth.astype(np.uint8),
            onset_prob=onset_probability.astype(np.float32),
            onset_raw_prob=onset_raw_probability.astype(np.float32),
            release_true=release_truth.astype(np.uint8),
            release_prob=release_probability.astype(np.float32),
            release_raw_prob=release_raw_probability.astype(np.float32),
            onset_positive_weight=onset_weight.numpy().astype(np.float32),
            release_positive_weight=release_weight.numpy().astype(np.float32),
            event_valid=event_valid.astype(np.uint8),
            input_active=active.astype(np.uint8),
            session_lengths=lengths,
            session_ids=np.asarray(stream_ids),
        )

    summarize_state = summarize if allow_oracle_thresholds else _summarize_fixed_thresholds
    state_all = summarize_state(truth, state_probability, boundaries=lengths)
    latch_all = summarize_state(
        truth, latched.astype(np.float32), boundaries=lengths
    )
    state_all_decisions = _decision_metrics(
        truth,
        state_probability >= 0.5,
        active=np.ones(len(truth), dtype=bool),
        lengths=lengths,
    )
    latch_all_decisions = _decision_metrics(
        truth,
        latched,
        active=np.ones(len(truth), dtype=bool),
        lengths=lengths,
    )
    if np.all(active):
        state_active = state_all
        latch_active = latch_all
        state_active_decisions = state_all_decisions
        latch_active_decisions = latch_all_decisions
    else:
        state_active = summarize_state(
            truth, state_probability, boundaries=lengths, active=active
        )
        latch_active = summarize_state(
            truth,
            latched.astype(np.float32),
            boundaries=lengths,
            active=active,
        )
        state_active_decisions = _decision_metrics(
            truth,
            state_probability >= 0.5,
            active=active,
            lengths=lengths,
        )
        latch_active_decisions = _decision_metrics(
            truth, latched, active=active, lengths=lengths
        )

    report = {
        "all_frames": {
            "n": int(len(truth)),
            "metrics": state_all,
            "decision_metrics": state_all_decisions,
            "latch_metrics": latch_all,
            "latch_decision_metrics": latch_all_decisions,
        },
        "input_active_only": {
            "n": int(active.sum()),
            "metrics": state_active,
            "decision_metrics": state_active_decisions,
            "latch_metrics": latch_active,
            "latch_decision_metrics": latch_active_decisions,
        },
        "event_heads": {
            "valid_transition_frames": int(event_valid.sum()),
            "onset": _event_head_metrics(
                onset_truth,
                onset_probability,
                event_valid,
                lengths=lengths,
                allow_oracle_thresholds=allow_oracle_thresholds,
            ),
            "release": _event_head_metrics(
                release_truth,
                release_probability,
                event_valid,
                lengths=lengths,
                allow_oracle_thresholds=allow_oracle_thresholds,
            ),
        },
        "threshold_policy": {
            "data_fitted_thresholds_enabled": bool(allow_oracle_thresholds),
            "fixed_state_threshold": 0.5,
            "fixed_event_threshold": 0.5,
        },
        "decode": decode,
        "event_probability_adjustment": {
            "method": "subtract_log_training_positive_weight_from_logits",
            "purpose": (
                "undo weighted-BCE prior shift before fixed-0.5 event "
                "decoding; ranking and AP are unchanged"
            ),
            "onset_positive_weight": {
                key: float(onset_weight[index])
                for index, key in enumerate(KEY_ORDER)
            },
            "release_positive_weight": {
                key: float(release_weight[index])
                for index, key in enumerate(KEY_ORDER)
            },
            "sidecar_probability_fields": {
                "deweighted": ["onset_prob", "release_prob"],
                "raw_weighted": ["onset_raw_prob", "release_raw_prob"],
            },
        },
        "streams": len(lengths),
        "input_active_is_placeholder": bool(active.all()),
    }
    return _to_jsonable(report)
