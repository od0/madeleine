#!/usr/bin/env python3
"""Fail-closed fixed evaluation for exploratory dynamics-pretrained C/D arms.

Only the final state of the matched 25.7M frozen-feature GRU is accepted.  The
only evaluation surface is the canonical mapped-y4n later-eight split and the
only decision threshold is 0.5.  There is deliberately no B1, threshold-fit,
calibration-fit, checkpoint-selection, or alternate-surface interface.

Metrics are recomputed from the serialized prediction sidecar before anything
is published.  The report, sidecar, and terminal marker are then published as
one fail-closed bundle whose hashes bind the downstream checkpoint, supervised
feature assembly, and label-free SSL checkpoint.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from badeline.metrics import (
    match_event_counts,
    per_key_transition_f1,
    score_events,
    transition_events,
)
from badeline.model import BadelineIDM
from badeline.train import read_session_ids
from data.schema import KEY_ORDER
from experiments.eval_tcn_control_lr_b1 import infer_fixed_state
from experiments.assemble_dynamics_supervised_features import (
    ASSEMBLY_SCHEMA,
    COMPLETION_SCHEMA as ASSEMBLY_COMPLETION_SCHEMA,
    VALIDATION_SCHEMA as ASSEMBLY_VALIDATION_SCHEMA,
)
from experiments.export_dynamics_features import (
    load_checkpoint_contract,
    load_inventory,
)


SCHEMA_VERSION = "madeleine.dynamics-downstream-fixed-eval.v1"
MARKER_SCHEMA_VERSION = "madeleine.dynamics-downstream-fixed-eval-complete.v1"
STUDY_ID = "photon_inspired_celeste_dynamics_exploratory_cd_s0_v1"
SURFACE = "mapped_y4n_later_eight"
EXPECTED_FINAL_STEP = 20_458
EXPECTED_TRAINABLE_PARAMETERS = 25_719_815
EXPECTED_SSL_STEPS = 30_000
EXPECTED_SEED = 0
ECE_BIN_COUNT = 15
SHUFFLE_REPETITIONS = 10
SHUFFLE_SEED = 0
Y4N_BASE_SESSION_IDS = tuple(
    f"y4nQHqYSObI__r{index:03d}" for index in range(8, 16)
)
Y4N_STREAM_IDS = tuple(
    f"{session_id}__stream000" for session_id in Y4N_BASE_SESSION_IDS
)
Y4N_STREAM_LENGTHS = (35_619,) * 7 + (20_019,)
Y4N_FRAMES = 269_352
Y4N_TRUTH_SHA256 = (
    "f61a0de4076f4683f01494837f01c3e314873ab0d78ee131b43e8e9f6e576a01"
)
REQUIRED_SIDECAR_FIELDS = {
    "y_true",
    "y_prob",
    "input_active",
    "session_lengths",
    "session_ids",
}
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")

ARM_RUN_IDS = {
    "C": "dynamics_c_full_210train_y4n_holdout_26m_128x3_s0",
    "D": "dynamics_d_full_210train_y4n_holdout_26m_128x3_s0",
}

REQUIRED_CONFIG = {
    "active_targets_only": True,
    "backbone_feature_dim": 512,
    "batch_size": 1536,
    "class_balance": True,
    "class_balance_max": 10.0,
    "embedding_dim": 1024,
    "eval_batch_size": 3072,
    "eval_interval": EXPECTED_FINAL_STEP,
    "feature_deltas": True,
    "frame_stride": 3,
    "initial_train_eval": False,
    "input_config": "pixels",
    "learning_rate": 0.0003,
    "linear_lr_decay": True,
    "max_steps": EXPECTED_FINAL_STEP,
    "optimizer": "adamw",
    "precomputed_features": True,
    "seed": EXPECTED_SEED,
    "segment_windows": 96,
    "temporal_dim": 2048,
    "transition_weight": 8.0,
    "weight_decay": 0.01,
    "window": 128,
    "window_mode": "centered",
}


@dataclass(frozen=True)
class SidecarArrays:
    truth: np.ndarray
    probability: np.ndarray
    active: np.ndarray
    lengths: np.ndarray
    session_ids: np.ndarray
    support: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_array_sha256(value: np.ndarray) -> str:
    """Historical y4n receipt: shape followed directly by C-order bytes."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _require_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{description} is not a lowercase SHA-256")
    return value


def _reject_forbidden_identity(value: str, description: str) -> None:
    folded = str(value).casefold()
    compact = re.sub(r"[^a-z0-9]+", "", folded)
    if (
        "untouched" in folded
        or "rec_20260727_220000_test" in folded
        or folded == "b1"
        or folded.startswith(("b1_", "b1-", ".b1"))
        or compact.startswith(("b1pixels", "b1features", "b1engine"))
        or re.match(r"^val[-_]?[ab](?:$|[-_])", folded) is not None
    ):
        raise ValueError(f"{description} identifies a forbidden surface")


def _reject_forbidden_path(path: Path, description: str) -> None:
    for component in Path(path).parts:
        _reject_forbidden_identity(component, description)


def _stream_slices(lengths: np.ndarray, frames: int) -> list[slice]:
    if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError("session_lengths must be a one-dimensional integer array")
    if not len(lengths) or np.any(lengths <= 0) or int(lengths.sum()) != frames:
        raise ValueError("session_lengths do not exactly partition the sidecar")
    starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64))
    )
    return [slice(int(start), int(start + length)) for start, length in zip(starts, lengths)]


def load_validate_sidecar(
    path: Path,
    *,
    expected_truth_sha256: str | None = None,
) -> SidecarArrays:
    """Load the exact canonical y4n later-eight sidecar and fail closed."""

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_SIDECAR_FIELDS:
            raise ValueError("prediction sidecar member set changed")
        truth = np.asarray(archive["y_true"])
        probability = np.asarray(archive["y_prob"])
        active = np.asarray(archive["input_active"])
        lengths = np.asarray(archive["session_lengths"])
        session_ids = np.asarray(archive["session_ids"])

    expected_shape = (Y4N_FRAMES, len(KEY_ORDER))
    if truth.dtype != np.uint8 or truth.shape != expected_shape:
        raise ValueError("y4n truth schema or support changed")
    if probability.dtype != np.float32 or probability.shape != expected_shape:
        raise ValueError("y4n probability schema or support changed")
    if active.dtype != np.uint8 or active.shape != (Y4N_FRAMES,):
        raise ValueError("y4n activity schema or support changed")
    if lengths.dtype != np.int64 or lengths.tolist() != list(Y4N_STREAM_LENGTHS):
        raise ValueError("y4n stream boundaries changed")
    if session_ids.ndim != 1 or session_ids.tolist() != list(Y4N_STREAM_IDS):
        raise ValueError("y4n stream identities changed")
    if not np.all(np.isin(truth, (0, 1))):
        raise ValueError("y4n truth is not binary")
    if not np.all(active == 1):
        raise ValueError("y4n later-eight activity support changed")
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("y4n probabilities are not finite values in [0,1]")
    _stream_slices(lengths, len(truth))
    truth_sha = canonical_array_sha256(truth)
    if expected_truth_sha256 is None:
        expected_truth_sha256 = Y4N_TRUTH_SHA256
    if truth_sha != _require_sha256(expected_truth_sha256, "expected truth hash"):
        raise ValueError("canonical y4n mapped-label truth receipt changed")
    support = {
        "all_frames": Y4N_FRAMES,
        "input_active_frames": Y4N_FRAMES,
        "streams": len(Y4N_STREAM_IDS),
        "session_ids": list(Y4N_STREAM_IDS),
        "stream_lengths": list(Y4N_STREAM_LENGTHS),
        "truth_sha256": truth_sha,
        "probability_sha256": canonical_array_sha256(probability),
        "input_active_sha256": canonical_array_sha256(active),
        "finite_aligned_arrays": True,
    }
    return SidecarArrays(truth, probability, active, lengths, session_ids, support)


def _safe_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    true_positive = int(np.sum(truth & predicted))
    false_positive = int(np.sum(~truth & predicted))
    false_negative = int(np.sum(truth & ~predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else float(2 * true_positive / denominator)


def _average_precision(truth: np.ndarray, probability: np.ndarray) -> float:
    """Dependency-light non-interpolated AP with score ties kept together."""

    truth = np.asarray(truth, dtype=bool)
    positives = int(truth.sum())
    if positives == 0:
        raise ValueError("canonical y4n key has no positive support")
    scores = np.asarray(probability, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    ordered_truth = truth[order]
    cumulative = np.cumsum(ordered_truth, dtype=np.int64)
    ordered_scores = scores[order]
    # Precision/recall points exist after the final member of each tied-score
    # group.  Treating tied examples in arbitrary stable order would inflate
    # a constant-score prevalence baseline.
    group_ends = np.flatnonzero(
        np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    )
    true_positive = cumulative[group_ends].astype(np.float64)
    precision = true_positive / (group_ends.astype(np.float64) + 1.0)
    recall = true_positive / positives
    recall_gain = np.diff(np.r_[0.0, recall])
    return float(np.sum(precision * recall_gain))


def _equal_mass_ece(
    truth: np.ndarray, probability: np.ndarray, bin_count: int = ECE_BIN_COUNT
) -> dict[str, Any]:
    if bin_count < 1:
        raise ValueError("ECE bin count must be positive")
    order = np.argsort(np.asarray(probability), kind="mergesort")
    groups = [group for group in np.array_split(order, bin_count) if len(group)]
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for group in groups:
        confidence = float(np.mean(probability[group]))
        observed = float(np.mean(truth[group]))
        gap = abs(confidence - observed)
        ece += len(group) / len(truth) * gap
        bins.append(
            {
                "count": int(len(group)),
                "minimum_probability": float(np.min(probability[group])),
                "maximum_probability": float(np.max(probability[group])),
                "mean_probability": confidence,
                "observed_positive_rate": observed,
                "absolute_gap": float(gap),
            }
        )
    return {
        "kind": "equal_mass",
        "requested_bin_count": bin_count,
        "nonempty_bin_count": len(bins),
        "ece": float(ece),
        "bins": bins,
    }


def _event_score(
    truth: np.ndarray,
    probability: np.ndarray,
    active: np.ndarray,
    lengths: Sequence[int],
    collar: int,
) -> dict[str, Any]:
    raw = per_key_transition_f1(
        truth,
        probability,
        threshold=0.5,
        collar=collar,
        boundaries=[int(value) for value in lengths],
        active=active,
    )
    per_key: dict[str, Any] = {}
    onset: list[float] = []
    release: list[float] = []
    combined: list[float] = []
    for key in KEY_ORDER:
        key_row: dict[str, Any] = {}
        for source_name, output_name in (
            ("onset", "onset"),
            ("offset", "release"),
            ("event", "combined"),
        ):
            source = raw[key][source_name]
            row = {
                "precision": float(source["precision"]),
                "recall": float(source["recall"]),
                "f1": float(source["f1"]),
                "n_true": int(source["n_true"]),
                "n_pred": int(source["n_pred"]),
                "n_matched": int(source["n_matched"]),
            }
            if not all(math.isfinite(float(row[name])) for name in ("precision", "recall", "f1")):
                raise ValueError("canonical y4n event metric is non-finite")
            key_row[output_name] = row
        per_key[key] = key_row
        onset.append(key_row["onset"]["f1"])
        release.append(key_row["release"]["f1"])
        combined.append(key_row["combined"]["f1"])
    return {
        "collar_frames": collar,
        "per_key": per_key,
        "macro_onset_f1": float(np.mean(onset)),
        "macro_release_f1": float(np.mean(release)),
        "macro_combined_f1": float(np.mean(combined)),
    }


def _state_core(
    truth: np.ndarray,
    probability: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    gate = np.asarray(active, dtype=bool)
    frame_truth = np.asarray(truth, dtype=bool)[gate]
    frame_probability = np.asarray(probability, dtype=np.float64)[gate]
    if not len(frame_truth):
        raise ValueError("evaluation surface has no active rows")
    predicted = frame_probability >= 0.5
    clipped = np.clip(frame_probability, 1e-7, 1.0 - 1e-7)
    truth_float = frame_truth.astype(np.float64)
    correct = predicted == frame_truth
    per_key_ap: dict[str, float] = {}
    prevalence: dict[str, float] = {}
    state_f1: dict[str, float] = {}
    predicted_rate: dict[str, float] = {}
    bce: dict[str, float] = {}
    brier: dict[str, float] = {}
    ece: dict[str, Any] = {}
    for column, key in enumerate(KEY_ORDER):
        key_truth = frame_truth[:, column]
        key_probability = frame_probability[:, column]
        key_truth_float = truth_float[:, column]
        per_key_ap[key] = _average_precision(key_truth, key_probability)
        prevalence[key] = float(np.mean(key_truth))
        state_f1[key] = _safe_f1(key_truth, predicted[:, column])
        predicted_rate[key] = float(np.mean(predicted[:, column]))
        bce[key] = float(
            np.mean(
                -key_truth_float * np.log(clipped[:, column])
                - (1.0 - key_truth_float) * np.log1p(-clipped[:, column])
            )
        )
        brier[key] = float(np.mean((key_probability - key_truth_float) ** 2))
        ece[key] = _equal_mass_ece(key_truth, key_probability)
    return {
        "per_key_ap": per_key_ap,
        "macro_ap": float(np.mean(list(per_key_ap.values()))),
        "per_key_prevalence": prevalence,
        "macro_prevalence": float(np.mean(list(prevalence.values()))),
        "per_key_state_f1_fixed_0_5": state_f1,
        "macro_state_f1_fixed_0_5": float(np.mean(list(state_f1.values()))),
        "key_state_accuracy_fixed_0_5": {
            "micro": float(np.mean(correct)),
            "joint_exact_match": float(np.mean(np.all(correct, axis=1))),
        },
        "predicted_positive_rate_fixed_0_5": {
            "micro": float(np.mean(predicted)),
            "per_key": predicted_rate,
        },
        "per_key_bce": bce,
        "macro_bce": float(np.mean(list(bce.values()))),
        "per_key_brier": brier,
        "macro_brier": float(np.mean(list(brier.values()))),
        "equal_mass_ece": {
            "bin_count": ECE_BIN_COUNT,
            "macro_ece": float(np.mean([row["ece"] for row in ece.values()])),
            "per_key": ece,
        },
    }


def _persistence_probability(truth: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    persistence = np.zeros_like(truth, dtype=np.float64)
    for stream in _stream_slices(lengths, len(truth)):
        start, stop = int(stream.start), int(stream.stop)
        persistence[start + 1 : stop] = truth[start : stop - 1]
    return persistence


def _compact_baseline(
    truth: np.ndarray,
    probability: np.ndarray,
    active: np.ndarray,
    lengths: np.ndarray,
) -> dict[str, Any]:
    state = _state_core(truth, probability, active)
    return {
        **state,
        "events_fixed_0_5": {
            "exact": _event_score(truth, probability, active, lengths, 0),
            "plus_minus_2": _event_score(truth, probability, active, lengths, 2),
        },
    }


def _shuffled_event_baseline(
    truth: np.ndarray,
    active: np.ndarray,
    lengths: np.ndarray,
    *,
    seed: int = SHUFFLE_SEED,
    repetitions: int = SHUFFLE_REPETITIONS,
) -> dict[str, Any]:
    """Shuffle onset/release times within each stream, preserving counts."""

    rng = np.random.default_rng(seed)
    stream_slices = _stream_slices(lengths, len(truth))
    collars: dict[str, Any] = {}
    for collar, name in ((0, "exact"), (2, "plus_minus_2")):
        per_key: dict[str, Any] = {}
        macro_by_type = {"onset": [], "release": [], "combined": []}
        for column, key in enumerate(KEY_ORDER):
            repetition_scores = {"onset": [], "release": [], "combined": []}
            repetition_matched = {"onset": [], "release": [], "combined": []}
            n_true_onset = n_true_release = 0
            truth_by_stream: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for stream in stream_slices:
                key_truth = truth[stream, column]
                gate = active[stream].astype(bool)
                true_onset, true_release = transition_events(key_truth)
                true_onset = true_onset[gate[true_onset]]
                true_release = true_release[gate[true_release]]
                candidates = np.flatnonzero(gate)
                if len(candidates) < max(len(true_onset), len(true_release)):
                    raise ValueError("shuffled-event baseline lacks active candidates")
                truth_by_stream.append((true_onset, true_release, candidates))
                n_true_onset += len(true_onset)
                n_true_release += len(true_release)
            for _ in range(repetitions):
                matched_onset = matched_release = 0
                for true_onset, true_release, candidates in truth_by_stream:
                    random_onset = np.sort(
                        rng.choice(candidates, size=len(true_onset), replace=False)
                    )
                    random_release = np.sort(
                        rng.choice(candidates, size=len(true_release), replace=False)
                    )
                    matched_onset += match_event_counts(true_onset, random_onset, collar)
                    matched_release += match_event_counts(true_release, random_release, collar)
                for event_type, n_true, matched in (
                    ("onset", n_true_onset, matched_onset),
                    ("release", n_true_release, matched_release),
                    (
                        "combined",
                        n_true_onset + n_true_release,
                        matched_onset + matched_release,
                    ),
                ):
                    row = score_events(n_true, n_true, matched)
                    repetition_scores[event_type].append(float(row["f1"]))
                    repetition_matched[event_type].append(int(matched))
            key_row: dict[str, Any] = {}
            for event_type, n_true in (
                ("onset", n_true_onset),
                ("release", n_true_release),
                ("combined", n_true_onset + n_true_release),
            ):
                scores = repetition_scores[event_type]
                key_row[event_type] = {
                    "mean_f1": float(np.mean(scores)),
                    "std_f1": float(np.std(scores)),
                    "n_true": int(n_true),
                    "n_pred_per_repetition": int(n_true),
                    "mean_matched": float(np.mean(repetition_matched[event_type])),
                }
                macro_by_type[event_type].append(float(np.mean(scores)))
            per_key[key] = key_row
        collars[name] = {
            "collar_frames": collar,
            "per_key": per_key,
            "macro_onset_f1": float(np.mean(macro_by_type["onset"])),
            "macro_release_f1": float(np.mean(macro_by_type["release"])),
            "macro_combined_f1": float(np.mean(macro_by_type["combined"])),
        }
    return {
        "definition": (
            "true onset and release counts placed uniformly without replacement "
            "on active rows independently within each stored stream"
        ),
        "seed": seed,
        "repetitions": repetitions,
        **collars,
    }


def score_fixed_surface(
    truth: np.ndarray,
    probability: np.ndarray,
    active: np.ndarray,
    lengths: Sequence[int],
) -> dict[str, Any]:
    """Score the sole frozen 0.5 surface; this function fits nothing."""

    truth = np.asarray(truth)
    probability = np.asarray(probability)
    active = np.asarray(active)
    lengths_array = np.asarray(lengths, dtype=np.int64)
    if truth.ndim != 2 or truth.shape[1] != len(KEY_ORDER):
        raise ValueError("truth must have canonical [frames,keys] shape")
    if probability.shape != truth.shape or active.shape != (len(truth),):
        raise ValueError("metric arrays are not aligned")
    if not np.all(np.isin(truth, (0, 1))) or not np.all(np.isin(active, (0, 1))):
        raise ValueError("truth and active arrays must be binary")
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("probabilities must be finite values in [0,1]")
    _stream_slices(lengths_array, len(truth))
    truth_bool = truth.astype(bool, copy=False)
    active_bool = active.astype(bool, copy=False)
    state = _state_core(truth_bool, probability, active_bool)
    prevalence_probability = np.broadcast_to(
        np.asarray([state["per_key_prevalence"][key] for key in KEY_ORDER]),
        probability.shape,
    )
    always_released = np.zeros_like(probability, dtype=np.float64)
    persistence = _persistence_probability(truth_bool, lengths_array)
    return {
        **state,
        "events_fixed_0_5": {
            "boundary_policy": "stored session_lengths; no transition crosses a stream",
            "active_policy": "events formed within full streams and gated at event time",
            "exact": _event_score(
                truth_bool, probability, active_bool, lengths_array, 0
            ),
            "plus_minus_2": _event_score(
                truth_bool, probability, active_bool, lengths_array, 2
            ),
        },
        "baselines": {
            "always_released": _compact_baseline(
                truth_bool, always_released, active_bool, lengths_array
            ),
            "persistence": {
                "definition": "previous true state within each stream; first row released",
                **_compact_baseline(
                    truth_bool, persistence, active_bool, lengths_array
                ),
            },
            "prevalence": {
                "definition": "constant per-key probability equal to this surface's prevalence",
                **_compact_baseline(
                    truth_bool, prevalence_probability, active_bool, lengths_array
                ),
            },
            "shuffled_events": _shuffled_event_baseline(
                truth_bool, active_bool, lengths_array
            ),
        },
        "threshold_policy": {
            "state_probability": 0.5,
            "transition_probability": 0.5,
            "data_fitted_thresholds_used": False,
            "calibration_parameters_fitted": False,
        },
    }


def validate_run_config(config: Mapping[str, Any], arm: str) -> None:
    if arm not in ARM_RUN_IDS:
        raise ValueError("dynamics downstream arm must be C or D")
    for key, expected in REQUIRED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"downstream run config changed {key}")
    if "temporal_arch" in config and config.get("temporal_arch") != "gru":
        raise ValueError("downstream model is not the matched GRU")
    forbidden = {
        "event_latch",
        "event_loss_weight",
        "onset_positive_weight",
        "release_positive_weight",
        "source_sampling",
    }
    if forbidden.intersection(config):
        raise ValueError("downstream config contains a non-matched objective")
    if set(config) != set(REQUIRED_CONFIG) | {"_note"}:
        raise ValueError("downstream run config key set changed")
    note = config.get("_note")
    if not isinstance(note, str) or str(EXPECTED_FINAL_STEP) not in note:
        raise ValueError("downstream config lacks fixed-endpoint note")
    if f"dynamics {arm}" not in note and f"dynamics-{arm}" not in note:
        raise ValueError("downstream config note does not identify its arm")


def _argument_value(argv: object, flag: str) -> str:
    if not isinstance(argv, list):
        raise ValueError("run metadata argv is missing")
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"run metadata must contain {flag} exactly once")
    return str(argv[positions[0] + 1])


def _same_path(first: object, second: Path) -> bool:
    return isinstance(first, str) and Path(first).resolve() == Path(second).resolve()


def _read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_run_and_provenance(
    *,
    arm: str,
    run_id: str,
    run_dir: Path,
    data_dir: Path,
    later8_sessions_path: Path,
    assembly_validation_path: Path,
    split_receipt_path: Path,
    ssl_checkpoint_path: Path,
    ssl_checkpoint_sha256: str,
    inventory_path: Path,
    inventory_sha256: str,
) -> tuple[dict[str, Any], BadelineIDM, dict[str, Any]]:
    """Validate the complete C/D downstream provenance chain and final state."""

    if ARM_RUN_IDS.get(arm) != run_id or run_dir.name != run_id:
        raise ValueError("arm-specific dynamics downstream run identity changed")
    for path, name in (
        (run_dir, "run path"),
        (data_dir, "data path"),
        (later8_sessions_path, "session-list path"),
        (assembly_validation_path, "assembly-validation path"),
        (split_receipt_path, "split-receipt path"),
    ):
        _reject_forbidden_path(path, name)

    ssl_sha = _require_sha256(ssl_checkpoint_sha256, "SSL checkpoint hash")
    inventory_sha = _require_sha256(inventory_sha256, "inventory hash")
    if sha256_file(ssl_checkpoint_path) != ssl_sha:
        raise ValueError("SSL checkpoint SHA-256 mismatch")
    if sha256_file(inventory_path) != inventory_sha:
        raise ValueError("SSL inventory SHA-256 mismatch")
    ssl_contract = load_checkpoint_contract(
        ssl_checkpoint_path,
        ssl_sha,
        expected_arm=arm,
        expected_completed_steps=EXPECTED_SSL_STEPS,
    )
    if ssl_contract.horizons != (1, 2, 4):
        raise ValueError("terminal SSL checkpoint horizon contract changed")
    inventory_contract = load_inventory(inventory_path, inventory_sha)
    if inventory_contract.frames != 32_598_000:
        raise ValueError("terminal SSL inventory frame support changed")

    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "model.pt"
    run_meta_path = run_dir / "run_meta.json"
    dynamics_meta_path = run_dir / "dynamics_downstream_meta.json"
    for path in (config_path, checkpoint_path, run_meta_path, dynamics_meta_path):
        if not path.is_file():
            raise ValueError(f"required downstream run artifact is missing: {path.name}")
    config = _json(config_path, "downstream config")
    validate_run_config(config, arm)

    assembly_validation = _json(assembly_validation_path, "assembly validation")
    if assembly_validation.get("schema_version") != ASSEMBLY_VALIDATION_SCHEMA:
        raise ValueError("assembly-validation schema changed")
    if assembly_validation.get("ok") is not True or assembly_validation.get("arm") != arm:
        raise ValueError("assembly validation is not a passing receipt for this arm")
    if assembly_validation.get("checkpoint_sha256") != ssl_sha:
        raise ValueError("assembly validation SSL-checkpoint binding changed")
    if assembly_validation.get("inventory_sha256") != inventory_sha:
        raise ValueError("assembly validation inventory binding changed")
    if not _same_path(assembly_validation.get("output_root"), data_dir):
        raise ValueError("assembly validation does not bind the evaluated data root")
    if assembly_validation.get("deep_shards") is not True or assembly_validation.get("deep_sources") is not True:
        raise ValueError("assembly validation is not deep")
    if assembly_validation.get("counts") != {
        "expected_sessions": 1_554,
        "expected_frames": 32_598_000,
        "checked_sessions": 1_554,
        "checked_frames": 32_598_000,
    }:
        raise ValueError("assembly validation exact support changed")
    if assembly_validation.get("failures") != []:
        raise ValueError("assembly validation contains failures")

    assembly_manifest_path = data_dir / "supervised_assembly_manifest.json"
    assembly_completion_path = data_dir / "supervised_assembly_complete.json"
    assembly_manifest = _json(assembly_manifest_path, "assembly manifest")
    assembly_completion = _json(assembly_completion_path, "assembly completion")
    if assembly_manifest.get("schema_version") != ASSEMBLY_SCHEMA:
        raise ValueError("assembly manifest schema changed")
    if assembly_completion.get("schema_version") != ASSEMBLY_COMPLETION_SCHEMA:
        raise ValueError("assembly completion schema changed")
    for payload in (assembly_manifest, assembly_completion):
        if payload.get("checkpoint_sha256") != ssl_sha or payload.get("inventory_sha256") != inventory_sha:
            raise ValueError("assembly provenance binding changed")
    if assembly_completion.get("assembly_manifest_sha256") != sha256_file(assembly_manifest_path):
        raise ValueError("assembly completion manifest hash changed")
    if assembly_manifest.get("labels_first_opened_after_terminal_validation") is not True:
        raise ValueError("assembly label-access chronology changed")
    expected_counts = {
        "videos": 211,
        "sessions": 1_554,
        "frames": 32_598_000,
        "train_sessions": 1_538,
        "y4n_sessions": 16,
        "y4n_later8_sessions": 8,
    }
    if assembly_manifest.get("counts") != expected_counts or assembly_completion.get("counts") != expected_counts:
        raise ValueError("assembly counts changed")

    if _read_ids(later8_sessions_path) != list(Y4N_BASE_SESSION_IDS):
        raise ValueError("evaluation list is not canonical y4n later-eight")
    all_path = data_dir / "all_sessions.txt"
    train_path = data_dir / "train_sessions.txt"
    val_path = data_dir / "val_sessions.txt"
    canonical_later8_path = data_dir / "y4n_later8_sessions.txt"
    all_ids = _read_ids(all_path)
    train_ids = _read_ids(train_path)
    val_ids = _read_ids(val_path)
    if canonical_later8_path.resolve() != later8_sessions_path.resolve():
        raise ValueError("evaluation must use the assembly's canonical later-eight list")
    if len(all_ids) != 1_554 or len(train_ids) != 1_538 or len(val_ids) != 16:
        raise ValueError("assembled split counts changed")
    if set(train_ids) & set(val_ids) or set(train_ids) | set(val_ids) != set(all_ids):
        raise ValueError("assembled train/holdout partition changed")
    if any(value.startswith("y4nQHqYSObI__") for value in train_ids):
        raise ValueError("y4n leaked into downstream training")
    if any(not value.startswith("y4nQHqYSObI__") for value in val_ids):
        raise ValueError("downstream validation includes a non-y4n session")
    if any(value.startswith("rec_") for value in all_ids):
        raise ValueError("local or sealed session entered the downstream corpus")

    split_receipt = _json(split_receipt_path, "downstream split receipt")
    if split_receipt.get("schema_version") != "madeleine.dynamics-downstream-split.v1":
        raise ValueError("downstream split-receipt schema changed")
    for key, expected in {
        "arm": arm,
        "run_id": run_id,
        "seed": EXPECTED_SEED,
        "max_steps": EXPECTED_FINAL_STEP,
        "all_sessions": 1_554,
        "train_sessions": 1_538,
        "validation_sessions": 16,
        "y4n_later8_sessions": 8,
        "train_videos": 210,
        "checkpoint_sha256": ssl_sha,
        "inventory_sha256": inventory_sha,
        "assembly_manifest_sha256": sha256_file(assembly_manifest_path),
        "assembly_completion_sha256": sha256_file(assembly_completion_path),
        "assembly_validation_sha256": sha256_file(assembly_validation_path),
    }.items():
        if split_receipt.get(key) != expected:
            raise ValueError(f"downstream split receipt changed {key}")
    expected_list_hashes = {
        "all_sessions_sha256": sha256_file(all_path),
        "train_sessions_sha256": sha256_file(train_path),
        "val_sessions_sha256": sha256_file(val_path),
        "y4n_later8_sessions_sha256": sha256_file(canonical_later8_path),
        "config_sha256": sha256_file(config_path),
    }
    for key, expected in expected_list_hashes.items():
        if split_receipt.get(key) != expected:
            raise ValueError(f"downstream split receipt changed {key}")
    policy = split_receipt.get("evaluation_policy")
    if policy != {
        "weights": "final_state_dict",
        "surface": SURFACE,
        "threshold": 0.5,
        "data_fitted_thresholds": False,
        "calibration_fit": False,
        "b1_access": False,
    }:
        raise ValueError("downstream fixed evaluation policy changed")
    nested_counts = split_receipt.get("counts")
    if nested_counts != {
        "all_sessions": 1_554,
        "train_sessions": 1_538,
        "train_videos": 210,
        "validation_sessions": 16,
        "later_eight_sessions": 8,
        "frames": 32_598_000,
    }:
        raise ValueError("downstream nested split counts changed")
    nested_lists = split_receipt.get("lists")
    if not isinstance(nested_lists, dict):
        raise ValueError("downstream nested list receipts are missing")
    for name, path, values in (
        ("all_sessions.txt", all_path, all_ids),
        ("train_sessions.txt", train_path, train_ids),
        ("val_sessions.txt", val_path, val_ids),
        ("y4n_later8_sessions.txt", canonical_later8_path, list(Y4N_BASE_SESSION_IDS)),
    ):
        if nested_lists.get(name) != {
            "path": str(path.resolve()),
            "count": len(values),
            "sha256": sha256_file(path),
        }:
            raise ValueError(f"downstream nested list receipt changed {name}")
    nested_assembly = split_receipt.get("assembly")
    if not isinstance(nested_assembly, dict) or nested_assembly != {
        "root": str(data_dir.resolve()),
        "manifest": str(assembly_manifest_path.resolve()),
        "manifest_sha256": sha256_file(assembly_manifest_path),
        "completion": str(assembly_completion_path.resolve()),
        "completion_sha256": sha256_file(assembly_completion_path),
        "validation": str(assembly_validation_path.resolve()),
        "validation_sha256": sha256_file(assembly_validation_path),
    }:
        raise ValueError("downstream nested assembly receipt changed")
    nested_ssl = split_receipt.get("ssl_checkpoint")
    if nested_ssl != {
        "path": str(ssl_checkpoint_path.resolve()),
        "sha256": ssl_sha,
        "arm": arm,
    }:
        raise ValueError("downstream nested SSL-checkpoint receipt changed")
    nested_inventory = split_receipt.get("pretraining_inventory")
    if nested_inventory != {
        "path": str(inventory_path.resolve()),
        "sha256": inventory_sha,
    }:
        raise ValueError("downstream nested inventory receipt changed")
    nested_training = split_receipt.get("training")
    if nested_training != {
        "model": "BadelineIDM default GRU",
        "seed": EXPECTED_SEED,
        "max_steps": EXPECTED_FINAL_STEP,
        "eval_interval": EXPECTED_FINAL_STEP,
        "initialized_from": None,
    }:
        raise ValueError("downstream nested training receipt changed")
    nested_evaluation = split_receipt.get("evaluation")
    expected_nested_evaluation = {
        "surface": "mapped-y4n-later-eight",
        "weights": "final_state_dict",
        "threshold": 0.5,
        "threshold_source": "fixed",
        "checkpoint_reselection": False,
        "oracle_thresholds": False,
        "calibration": False,
        "b1_accessed": False,
        "local_or_sealed_session_accessed": False,
    }
    if nested_evaluation != expected_nested_evaluation:
        raise ValueError("downstream nested evaluation receipt changed")
    nested_config = split_receipt.get("config")
    if not isinstance(nested_config, dict):
        raise ValueError("downstream nested config receipt is missing")
    resolved_control_config = Path(str(nested_config.get("resolved", "")))
    if (
        not resolved_control_config.is_file()
        or nested_config.get("resolved_sha256") != sha256_file(resolved_control_config)
        or nested_config.get("resolved_sha256") != sha256_file(config_path)
    ):
        raise ValueError("downstream resolved config receipt changed")

    run_meta = _json(run_meta_path, "Badeline run metadata")
    if run_meta.get("seed") != EXPECTED_SEED or run_meta.get("config") != config:
        raise ValueError("Badeline run metadata recipe changed")
    if run_meta.get("initialized_from") is not None:
        raise ValueError("downstream GRU was unexpectedly initialized")
    split = run_meta.get("split")
    if not isinstance(split, dict) or split.get("train") != train_ids or split.get("val") != val_ids:
        raise ValueError("Badeline run metadata split changed")
    argv = run_meta.get("argv")
    for flag, expected in (
        ("--data", data_dir),
        ("--train-sessions", train_path),
        ("--val-sessions", val_path),
        ("--out", run_dir),
    ):
        if not _same_path(_argument_value(argv, flag), expected):
            raise ValueError(f"Badeline argv changed {flag}")
    if int(_argument_value(argv, "--max-steps")) != EXPECTED_FINAL_STEP:
        raise ValueError("Badeline argv endpoint changed")
    if int(_argument_value(argv, "--seed")) != EXPECTED_SEED:
        raise ValueError("Badeline argv seed changed")
    if _argument_value(argv, "--device") != "cuda":
        raise ValueError("Badeline argv device changed")
    shard_hashes = _json(data_dir / "shard_hashes.json", "assembly shard hashes")
    run_shards = run_meta.get("shard_sha256")
    if not isinstance(run_shards, dict) or set(run_shards) != set(all_ids):
        raise ValueError("Badeline run metadata shard membership changed")
    for session_id in all_ids:
        row = shard_hashes.get(session_id)
        if not isinstance(row, dict) or run_shards.get(session_id) != row.get("sha256"):
            raise ValueError(f"Badeline run shard hash changed for {session_id}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("config") != config:
        raise ValueError("downstream checkpoint config changed")
    if checkpoint.get("key_order") != list(KEY_ORDER):
        raise ValueError("downstream checkpoint key order changed")
    if checkpoint.get("steps") != EXPECTED_FINAL_STEP:
        raise ValueError("downstream checkpoint is not the final endpoint")
    if checkpoint.get("initialized_from") is not None:
        raise ValueError("downstream checkpoint was unexpectedly initialized")
    if checkpoint.get("source_sampling_receipt") is not None:
        raise ValueError("matched all-valid GRU unexpectedly used source sampling")
    final_state = checkpoint.get("final_state_dict")
    selected_state = checkpoint.get("model_state_dict")
    if not isinstance(final_state, Mapping) or not final_state:
        raise ValueError("downstream checkpoint lacks final_state_dict")
    if not isinstance(selected_state, Mapping) or not selected_state:
        raise ValueError("downstream checkpoint lacks selected-state receipt")
    positive_weight = checkpoint.get("positive_weight")
    if (
        not isinstance(positive_weight, list)
        or len(positive_weight) != len(KEY_ORDER)
        or not all(
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and 1.0 <= float(value) <= 10.0
            for value in positive_weight
        )
    ):
        raise ValueError("downstream checkpoint positive weights changed")
    model = BadelineIDM(config)
    if model.temporal_arch != "gru" or not isinstance(model.temporal, torch.nn.GRUCell):
        raise ValueError("downstream checkpoint does not instantiate matched GRU")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError("downstream trainable parameter count changed")
    model.load_state_dict(final_state, strict=True)
    checkpoint_sha = sha256_file(checkpoint_path)

    dynamics_meta = _json(dynamics_meta_path, "dynamics downstream metadata")
    required_meta = {
        "arm": arm,
        "run_id": run_id,
        "seed": EXPECTED_SEED,
        "max_steps": EXPECTED_FINAL_STEP,
        "data": str(data_dir.resolve()),
        "train_sessions": str(train_path.resolve()),
        "val_sessions": str(val_path.resolve()),
        "y4n_later8_sessions": str(canonical_later8_path.resolve()),
        "config": str(config_path.resolve()),
        "assembly_validation": str(assembly_validation_path.resolve()),
        "assembly_manifest": str(assembly_manifest_path.resolve()),
        "assembly_completion": str(assembly_completion_path.resolve()),
        "ssl_checkpoint": str(ssl_checkpoint_path.resolve()),
        "inventory": str(inventory_path.resolve()),
        "run_meta_sha256": sha256_file(run_meta_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": checkpoint_sha,
        "assembly_validation_sha256": sha256_file(assembly_validation_path),
        "assembly_manifest_sha256": sha256_file(assembly_manifest_path),
        "assembly_completion_sha256": sha256_file(assembly_completion_path),
        "ssl_checkpoint_sha256": ssl_sha,
        "inventory_sha256": inventory_sha,
    }
    for key, expected in required_meta.items():
        if dynamics_meta.get(key) != expected:
            raise ValueError(f"dynamics downstream metadata changed {key}")
    source_commit = dynamics_meta.get("source_commit")
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("dynamics downstream source commit is malformed")
    expected_inputs = {
        "data_root": str(data_dir.resolve()),
        "split_receipt": {
            "path": str(split_receipt_path.resolve()),
            "sha256": sha256_file(split_receipt_path),
        },
        "assembly_validation": {
            "path": str(assembly_validation_path.resolve()),
            "sha256": sha256_file(assembly_validation_path),
        },
        "assembly_manifest": {
            "path": str(assembly_manifest_path.resolve()),
            "sha256": sha256_file(assembly_manifest_path),
        },
        "assembly_completion": {
            "path": str(assembly_completion_path.resolve()),
            "sha256": sha256_file(assembly_completion_path),
        },
        "ssl_checkpoint": {
            "path": str(ssl_checkpoint_path.resolve()),
            "sha256": ssl_sha,
        },
        "pretraining_inventory": {
            "path": str(inventory_path.resolve()),
            "sha256": inventory_sha,
        },
        "resolved_config": {
            "path": str(resolved_control_config.resolve()),
            "sha256": sha256_file(resolved_control_config),
        },
    }
    if dynamics_meta.get("inputs") != expected_inputs:
        raise ValueError("dynamics downstream input bindings changed")
    expected_binding = {
        "schema_version": "madeleine.dynamics-downstream-run.v1",
        "arm": arm,
        "run_id": run_id,
        "source_commit": source_commit,
        "seed": EXPECTED_SEED,
        "max_steps": EXPECTED_FINAL_STEP,
        "weights_for_release": "final_state_dict",
        "checkpoint_reselection": False,
        "inputs": expected_inputs,
        "evaluation": {
            key: value
            for key, value in expected_nested_evaluation.items()
            if key != "weights"
        },
    }
    if run_meta.get("dynamics_downstream") != expected_binding:
        raise ValueError("Badeline run metadata dynamics binding changed")
    for key, expected in expected_binding.items():
        if dynamics_meta.get(key) != expected:
            raise ValueError(f"dynamics downstream binding changed {key}")
    if dynamics_meta.get("status") != "trained" or not isinstance(
        dynamics_meta.get("completed_at"), str
    ):
        raise ValueError("dynamics downstream completion state changed")
    log_path = run_dir / "log.jsonl"
    try:
        log_rows = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("downstream training log is unreadable") from error
    if [row.get("step") for row in log_rows] != [0, EXPECTED_FINAL_STEP]:
        raise ValueError("downstream training log endpoints changed")
    def finite_tree(value: object) -> bool:
        if isinstance(value, bool) or value is None:
            return True
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        if isinstance(value, Mapping):
            return all(finite_tree(item) for item in value.values())
        if isinstance(value, list):
            return all(finite_tree(item) for item in value)
        return True
    if not all(finite_tree(row) for row in log_rows):
        raise ValueError("downstream training log contains non-finite values")
    expected_artifacts = {
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha,
        },
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "log": {"path": str(log_path.resolve()), "sha256": sha256_file(log_path)},
        "run_meta": {
            "path": str(run_meta_path.resolve()),
            "sha256": sha256_file(run_meta_path),
        },
    }
    if dynamics_meta.get("artifacts") != expected_artifacts:
        raise ValueError("dynamics downstream artifact hashes changed")

    identical = set(selected_state) == set(final_state) and all(
        torch.equal(selected_state[key], final_state[key]) for key in final_state
    )
    receipt = {
        "arm": arm,
        "run_id": run_id,
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": checkpoint_sha,
        "run_meta_sha256": sha256_file(run_meta_path),
        "dynamics_downstream_meta_sha256": sha256_file(dynamics_meta_path),
        "split_receipt_sha256": sha256_file(split_receipt_path),
        "assembly_validation_sha256": sha256_file(assembly_validation_path),
        "assembly_manifest_sha256": sha256_file(assembly_manifest_path),
        "assembly_completion_sha256": sha256_file(assembly_completion_path),
        "ssl_checkpoint_sha256": ssl_sha,
        "inventory_sha256": inventory_sha,
        "checkpoint_steps": EXPECTED_FINAL_STEP,
        "parameter_count": parameter_count,
        "temporal_architecture": "gru",
        "evaluation_weights": "final_state_dict",
        "selected_final_tensors_identical": identical,
    }
    return config, model, receipt


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def _refuse_existing(paths: Sequence[Path]) -> None:
    for path in paths:
        if os.path.lexists(path):
            raise ValueError(f"refusing to overwrite fixed evaluation artifact: {path}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _publish(
    temporary_sidecar: Path,
    sidecar: Path,
    temporary_report: Path,
    report: Path,
    temporary_marker: Path,
    marker: Path,
) -> None:
    published: list[Path] = []
    try:
        temporary_sidecar.replace(sidecar)
        published.append(sidecar)
        temporary_report.replace(report)
        published.append(report)
        temporary_marker.replace(marker)
        published.append(marker)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise


def validate_release(
    report_path: Path,
    sidecar_path: Path,
    marker_path: Path,
    *,
    expected_arm: str | None = None,
) -> dict[str, Any]:
    """Independently recompute a published release from its bound sidecar."""

    report = _json(report_path, "dynamics evaluation report")
    marker = _json(marker_path, "dynamics evaluation marker")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("dynamics evaluation report schema changed")
    if marker.get("schema_version") != MARKER_SCHEMA_VERSION or marker.get("status") != "complete":
        raise ValueError("dynamics evaluation marker is not complete")
    arm = report.get("arm")
    if arm not in ARM_RUN_IDS or (expected_arm is not None and arm != expected_arm):
        raise ValueError("dynamics evaluation arm changed")
    if report.get("study_id") != STUDY_ID or report.get("surface") != SURFACE:
        raise ValueError("dynamics evaluation identity changed")
    if report.get("run_id") != ARM_RUN_IDS[arm] or report.get("weights") != "final":
        raise ValueError("dynamics evaluation run or weight policy changed")
    sidecar_receipt = report.get("prediction_sidecar")
    if not isinstance(sidecar_receipt, dict):
        raise ValueError("dynamics report lacks prediction-sidecar receipt")
    if not _same_path(sidecar_receipt.get("path"), sidecar_path):
        raise ValueError("dynamics report prediction-sidecar path changed")
    sidecar_sha = sha256_file(sidecar_path)
    if sidecar_receipt.get("sha256") != sidecar_sha:
        raise ValueError("dynamics report prediction-sidecar hash changed")
    arrays = load_validate_sidecar(sidecar_path)
    if report.get("support") != arrays.support:
        raise ValueError("dynamics report support receipt changed")
    recomputed = score_fixed_surface(
        arrays.truth, arrays.probability, arrays.active, arrays.lengths
    )
    if report.get("metrics") != recomputed:
        raise ValueError("dynamics report metrics differ from serialized sidecar")
    run_receipt = report.get("run_receipt")
    assembly_validation = report.get("assembly_validation")
    if not isinstance(run_receipt, dict) or not isinstance(assembly_validation, dict):
        raise ValueError("dynamics report provenance receipt is missing")
    expected_marker = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "status": "complete",
        "study_id": STUDY_ID,
        "arm": arm,
        "run_id": ARM_RUN_IDS[arm],
        "surface": SURFACE,
        "weights": "final",
        "checkpoint_sha256": run_receipt.get("checkpoint_sha256"),
        "assembly_validation_sha256": assembly_validation.get("sha256"),
        "report_sha256": sha256_file(report_path),
        "sidecar_sha256": sidecar_sha,
    }
    if marker != expected_marker:
        raise ValueError("dynamics evaluation completion marker changed")
    return {
        "arm": arm,
        "run_id": ARM_RUN_IDS[arm],
        "report": str(report_path),
        "report_sha256": expected_marker["report_sha256"],
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sidecar_sha,
        "marker": str(marker_path),
        "marker_sha256": sha256_file(marker_path),
        "metrics_recomputed": True,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--arm", choices=("C", "D"), required=True)
    value.add_argument("--run-id", required=True)
    value.add_argument("--run", type=Path, required=True)
    value.add_argument("--data", type=Path, required=True)
    value.add_argument("--sessions", type=Path, required=True)
    value.add_argument("--assembly-validation", type=Path, required=True)
    value.add_argument("--split-receipt", type=Path, required=True)
    value.add_argument("--ssl-checkpoint", type=Path, required=True)
    value.add_argument("--ssl-checkpoint-sha256", required=True)
    value.add_argument("--inventory", type=Path, required=True)
    value.add_argument("--inventory-sha256", required=True)
    value.add_argument("--out", type=Path, required=True)
    value.add_argument("--sidecar", type=Path, required=True)
    value.add_argument("--completion-marker", type=Path, required=True)
    value.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config, model, run_receipt = validate_run_and_provenance(
        arm=args.arm,
        run_id=args.run_id,
        run_dir=args.run,
        data_dir=args.data,
        later8_sessions_path=args.sessions,
        assembly_validation_path=args.assembly_validation,
        split_receipt_path=args.split_receipt,
        ssl_checkpoint_path=args.ssl_checkpoint,
        ssl_checkpoint_sha256=args.ssl_checkpoint_sha256,
        inventory_path=args.inventory,
        inventory_sha256=args.inventory_sha256,
    )
    output_paths = (args.out, args.sidecar, args.completion_marker)
    for path in output_paths:
        _reject_forbidden_path(path, "evaluation output path")
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix != ".json" or args.sidecar.suffix != ".npz" or args.completion_marker.suffix != ".json":
        raise ValueError("report/marker must be JSON and sidecar must be NPZ")
    temporary_report = _temporary(args.out)
    temporary_sidecar = _temporary(args.sidecar)
    temporary_marker = _temporary(args.completion_marker)
    _refuse_existing([*output_paths, temporary_report, temporary_sidecar, temporary_marker])
    try:
        session_ids = read_session_ids(args.sessions)
        if session_ids != list(Y4N_BASE_SESSION_IDS):
            raise ValueError("evaluation session list changed")
        infer_fixed_state(
            model,
            config,
            args.data,
            session_ids,
            args.device,
            temporary_sidecar,
        )
        arrays = load_validate_sidecar(temporary_sidecar)
        # This re-load makes the serialized sidecar, rather than transient
        # inference arrays, the sole metric input.
        metrics = score_fixed_surface(
            arrays.truth, arrays.probability, arrays.active, arrays.lengths
        )
        assembly_validation_sha = sha256_file(args.assembly_validation)
        report = {
            "schema_version": SCHEMA_VERSION,
            "study_id": STUDY_ID,
            "arm": args.arm,
            "run_id": args.run_id,
            "surface": SURFACE,
            "weights": "final",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "label_kind": "mapped_foreign_nitrogen",
            "label_notice": (
                "Noisy mapped-label development comparison on canonical y4n "
                "later-eight; not engine-truth or final-test evidence."
            ),
            "sessions": list(Y4N_BASE_SESSION_IDS),
            "support": arrays.support,
            "metrics": metrics,
            "run_receipt": run_receipt,
            "assembly_validation": {
                "path": str(args.assembly_validation.resolve()),
                "sha256": assembly_validation_sha,
            },
            "prediction_sidecar": {
                "path": str(args.sidecar.resolve()),
                "sha256": sha256_file(temporary_sidecar),
            },
            "evaluation_policy": {
                "raw_sigmoid_probabilities": True,
                "fixed_state_threshold": 0.5,
                "fixed_event_threshold": 0.5,
                "threshold_parameters_fitted": False,
                "calibration_parameters_fitted": False,
                "checkpoint_selected_on_this_surface": False,
                "development_surfaces_accessed": [SURFACE],
            },
        }
        _write_json(temporary_report, report)
        marker = {
            "schema_version": MARKER_SCHEMA_VERSION,
            "status": "complete",
            "study_id": STUDY_ID,
            "arm": args.arm,
            "run_id": args.run_id,
            "surface": SURFACE,
            "weights": "final",
            "checkpoint_sha256": run_receipt["checkpoint_sha256"],
            "assembly_validation_sha256": assembly_validation_sha,
            "report_sha256": sha256_file(temporary_report),
            "sidecar_sha256": sha256_file(temporary_sidecar),
        }
        _write_json(temporary_marker, marker)
        _publish(
            temporary_sidecar,
            args.sidecar,
            temporary_report,
            args.out,
            temporary_marker,
            args.completion_marker,
        )
        validate_release(
            args.out,
            args.sidecar,
            args.completion_marker,
            expected_arm=args.arm,
        )
    finally:
        temporary_report.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        temporary_marker.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "run_id": args.run_id,
                "surface": SURFACE,
                "weights": "final",
                "fixed_threshold": 0.5,
                "report": str(args.out),
                "sidecar": str(args.sidecar),
                "completion_marker": str(args.completion_marker),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
