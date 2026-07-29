"""Score the preregistered aligned-TCN control/LR decision on mapped y4n.

This command deliberately has a narrow interface.  It reads the frozen study
contract plus a separate, exact artifact-receipt manifest, loads the *final*
prediction sidecar for every disclosed run, and scores only the eight frozen
later streams.  It never loads calibration parameters, oracle thresholds, or
B1 artifacts.

The receipt manifest has one entry per role in ``consulted_runs``::

    {
      "schema_version": 1,
      "runs": {
        "weighted_tcn_lr3e4": {
          "report_path": "results/idm/<run>_final_nitrogen_val.json",
          "sidecar_path": "results/idm/<run>_final_nitrogen_val_preds.npz",
          "config_path": "results/idm/<run>_config.json",
          "launcher_path": "experiments/run_vptlite_tcn_holdout.sh",
          "checkpoint_sha256": "<64 lowercase hex characters>",
          "implementation_git_commit": "<40 lowercase hex characters>",
          "training_start_utc": "<timezone-aware ISO-8601 timestamp>",
          "training_end_utc": "<timezone-aware ISO-8601 timestamp>",
          "final_step": 14265,
          "expected_final_step": 14265,
          "weights": "final",
          "b1_used_before_decision": false
        }
      }
    }

Paths are resolved relative to the repository root.  Checkpoint hashes are
receipts because historical checkpoints may remain on the training machine;
all repository-resident config, launcher, and prediction files are hashed by
this command.  Missing or malformed input fails before the output is written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from badeline.metrics import per_key_ap, per_key_f1, per_key_transition_f1
from data.schema import KEY_ORDER


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_PREREGISTRATION_COMMIT = "505fb01c5f2acf773423796362fd5743f184aa0b"
_PREREGISTRATION_RELATIVE_PATH = Path(
    "experiments/configs/tcn_control_lr_decision.json"
)
_NEW_RUN_ROLES = {
    "natural_tcn_control",
    "weighted_tcn_lr1e4",
    "weighted_tcn_lr1e3",
}
_HIGHER_IS_BETTER_METRICS = (
    "macro_ap",
    "macro_state_f1_fixed_0_5",
    "key_state_micro_accuracy_fixed_0_5",
    "joint_exact_match_accuracy_fixed_0_5",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve(repo: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        path = repo / path
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


def _relative(repo: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path.resolve())


def _require_hex(value: object, pattern: re.Pattern[str], field: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{field} has an invalid hexadecimal receipt")
    return value


def _parse_utc(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _verify_committed_preregistration(
    repo: Path, path: Path, commit: str
) -> str:
    """Require the study contract to equal its pre-launch committed bytes."""

    commit = _require_hex(commit, _GIT_SHA_RE, "preregistration Git commit")
    expected_path = (repo / _PREREGISTRATION_RELATIVE_PATH).resolve()
    if path.resolve() != expected_path:
        raise ValueError(
            "preregistration path must be "
            f"{_PREREGISTRATION_RELATIVE_PATH.as_posix()}"
        )
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if resolved.returncode != 0 or resolved.stdout.strip() != commit:
        raise ValueError("preregistration commit is not a resolvable full commit")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("preregistration commit is not an ancestor of HEAD")
    committed = subprocess.run(
        [
            "git",
            "cat-file",
            "blob",
            f"{commit}:{_PREREGISTRATION_RELATIVE_PATH.as_posix()}",
        ],
        cwd=repo,
        check=False,
        capture_output=True,
    )
    if committed.returncode != 0:
        raise ValueError("preregistration is absent from its declared commit")
    if committed.stdout != path.read_bytes():
        raise ValueError("preregistration differs from its pre-launch commit")
    return commit


def _require_canonical_result_path(
    repo: Path,
    path: Path,
    run_id: str,
    suffix: str,
    field: str,
) -> None:
    expected = (repo / "results" / "idm" / f"{run_id}{suffix}").resolve()
    if path.resolve() != expected:
        raise ValueError(f"{field} is not the canonical final artifact path")


def _validate_final_report(
    path: Path, run_id: str, sidecar: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the sidecar to its raw final-weight mapped-y4n evaluation."""

    report = _load_json(path)
    if report.get("weights") != "final":
        raise ValueError(f"{run_id}: evaluation report is not final weights")
    if report.get("label_kind") != "mapped_foreign_nitrogen":
        raise ValueError(f"{run_id}: evaluation report label surface changed")
    run_value = report.get("run")
    if not isinstance(run_value, str) or Path(run_value).name != run_id:
        raise ValueError(f"{run_id}: evaluation report run identity changed")
    full_stream_ids = list(sidecar["full_session_ids"])
    suffix = "__stream000"
    if any(not session_id.endswith(suffix) for session_id in full_stream_ids):
        raise ValueError(f"{run_id}: final sidecar contains a noncanonical stream ID")
    full_session_ids = [
        session_id[: -len(suffix)] for session_id in full_stream_ids
    ]
    if report.get("sessions") != full_session_ids:
        raise ValueError(f"{run_id}: evaluation report sessions changed")
    full_frames = int(sidecar["full_frames"])
    if report.get("all_frames", {}).get("n") != full_frames:
        raise ValueError(f"{run_id}: evaluation report frame support changed")
    if report.get("input_active_only", {}).get("n") != full_frames:
        raise ValueError(f"{run_id}: evaluation report active support changed")
    return report


def _stream_slices(lengths: np.ndarray) -> list[slice]:
    ends = np.cumsum(lengths, dtype=np.int64)
    starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
    return [
        slice(int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
    ]


def _load_frozen_sidecar(
    path: Path,
    session_ids: Sequence[str],
    expected_lengths: Mapping[str, int],
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "y_true",
            "y_prob",
            "input_active",
            "session_lengths",
            "session_ids",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        truth = np.asarray(archive["y_true"])
        probability = np.asarray(archive["y_prob"])
        active = np.asarray(archive["input_active"])
        lengths = np.asarray(archive["session_lengths"])
        ids = np.asarray(archive["session_ids"])

    expected_shape = (truth.shape[0], len(KEY_ORDER))
    if truth.ndim != 2 or truth.shape != expected_shape:
        raise ValueError(f"{path}: y_true must have shape [N,{len(KEY_ORDER)}]")
    if probability.shape != truth.shape:
        raise ValueError(f"{path}: y_prob shape does not match y_true")
    if not np.all(np.isin(truth, (0, 1))):
        raise ValueError(f"{path}: y_true is not binary")
    if not np.all(np.isfinite(probability)):
        raise ValueError(f"{path}: y_prob contains non-finite values")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError(f"{path}: y_prob lies outside [0,1]")
    if active.shape != (truth.shape[0],) or not np.all(np.isin(active, (0, 1))):
        raise ValueError(f"{path}: input_active must be a binary [N] array")
    if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError(f"{path}: session_lengths must be an integer vector")
    if np.any(lengths <= 0) or int(lengths.sum()) != len(truth):
        raise ValueError(f"{path}: invalid session_lengths")
    if ids.ndim != 1 or len(ids) != len(lengths):
        raise ValueError(f"{path}: session_ids must align with session_lengths")

    id_values = ids.astype(str).tolist()
    if len(set(id_values)) != len(id_values):
        raise ValueError(f"{path}: duplicate session IDs")
    if not np.all(active == 1):
        raise ValueError(f"{path}: full y4n active support changed")
    index = {session_id: position for position, session_id in enumerate(id_values)}
    missing_ids = [session_id for session_id in session_ids if session_id not in index]
    if missing_ids:
        raise ValueError(f"{path}: missing frozen streams {missing_ids}")

    slices = _stream_slices(lengths.astype(np.int64, copy=False))
    selected_positions = [index[session_id] for session_id in session_ids]
    selected_lengths = np.asarray(
        [int(lengths[position]) for position in selected_positions],
        dtype=np.int64,
    )
    required_lengths = np.asarray(
        [int(expected_lengths[session_id]) for session_id in session_ids],
        dtype=np.int64,
    )
    if not np.array_equal(selected_lengths, required_lengths):
        raise ValueError(
            f"{path}: frozen stream lengths changed: "
            f"{selected_lengths.tolist()} != {required_lengths.tolist()}"
        )

    selected_slices = [slices[position] for position in selected_positions]
    selected_truth = np.concatenate([truth[item] for item in selected_slices])
    selected_probability = np.concatenate(
        [probability[item] for item in selected_slices]
    )
    selected_active = np.concatenate([active[item] for item in selected_slices])
    if not np.all(selected_active == 1):
        raise ValueError(f"{path}: frozen population contains inactive rows")

    return {
        "truth": selected_truth.astype(bool, copy=False),
        "probability": selected_probability.astype(np.float64, copy=False),
        "active": selected_active.astype(bool, copy=False),
        "lengths": selected_lengths,
        "session_ids": list(session_ids),
        "full_session_ids": id_values,
        "full_session_lengths": lengths.astype(np.int64, copy=False),
        "full_frames": int(len(truth)),
    }


def _json_floats(values: Mapping[str, float]) -> dict[str, float | None]:
    return {
        key: float(value) if np.isfinite(value) else None
        for key, value in values.items()
    }


def _finite_mean(values: Sequence[float], field: str) -> float:
    result = float(np.mean(np.asarray(values, dtype=np.float64)))
    if not np.isfinite(result):
        raise ValueError(f"{field} is undefined on the frozen population")
    return result


def _macro_event_f1(
    truth: np.ndarray,
    probability: np.ndarray,
    lengths: np.ndarray,
    collar: int,
) -> tuple[float, dict[str, Any]]:
    result = per_key_transition_f1(
        truth,
        probability,
        threshold=0.5,
        collar=collar,
        boundaries=lengths.tolist(),
    )
    per_key = {key: float(result[key]["event"]["f1"]) for key in KEY_ORDER}
    return _finite_mean(list(per_key.values()), f"collar-{collar} event F1"), {
        key: {
            "f1": per_key[key],
            "n_true": int(result[key]["event"]["n_true"]),
            "n_pred": int(result[key]["event"]["n_pred"]),
            "n_matched": int(result[key]["event"]["n_matched"]),
        }
        for key in KEY_ORDER
    }


def _accuracy(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    correct = prediction == truth
    return float(correct.mean()), float(correct.all(axis=1).mean())


def _persistence(truth: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    result = np.zeros_like(truth, dtype=bool)
    for stream_slice in _stream_slices(lengths):
        start = int(stream_slice.start)
        end = int(stream_slice.stop)
        result[start + 1 : end] = truth[start : end - 1]
    return result


def _score_run(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    truth = np.asarray(sidecar["truth"], dtype=bool)
    probability = np.asarray(sidecar["probability"], dtype=np.float64)
    lengths = np.asarray(sidecar["lengths"], dtype=np.int64)
    session_ids = list(sidecar["session_ids"])
    predicted = probability >= 0.5

    key_ap = per_key_ap(truth, probability)
    key_f1 = per_key_f1(truth, probability, threshold=0.5)
    micro, joint = _accuracy(predicted, truth)
    exact_event, exact_by_key = _macro_event_f1(
        truth, probability, lengths, collar=0
    )
    collar_event, collar_by_key = _macro_event_f1(
        truth, probability, lengths, collar=2
    )

    prevalence = {
        key: float(truth[:, column].mean())
        for column, key in enumerate(KEY_ORDER)
    }
    released = np.zeros_like(truth, dtype=bool)
    released_micro, released_joint = _accuracy(released, truth)
    persistence = _persistence(truth, lengths)
    persistence_micro, persistence_joint = _accuracy(persistence, truth)
    persistence_probability = persistence.astype(np.float64)
    persistence_exact, persistence_exact_by_key = _macro_event_f1(
        truth, persistence_probability, lengths, collar=0
    )
    persistence_collar, persistence_collar_by_key = _macro_event_f1(
        truth, persistence_probability, lengths, collar=2
    )

    per_stream: dict[str, Any] = {}
    for session_id, stream_slice in zip(
        session_ids, _stream_slices(lengths), strict=True
    ):
        stream_ap = per_key_ap(truth[stream_slice], probability[stream_slice])
        finite_ap = [float(value) for value in stream_ap.values() if np.isfinite(value)]
        if not finite_ap:
            raise ValueError(f"{session_id}: no defined per-key AP values")
        per_stream[session_id] = {
            "frames": int(stream_slice.stop - stream_slice.start),
            "macro_ap": float(np.mean(finite_ap)),
            "defined_ap_keys": len(finite_ap),
            "per_key_ap": _json_floats(stream_ap),
        }

    return {
        "metrics": {
            "macro_ap": _finite_mean(list(key_ap.values()), "macro AP"),
            "per_key_ap": _json_floats(key_ap),
            "macro_state_f1_fixed_0_5": _finite_mean(
                list(key_f1.values()), "macro state F1"
            ),
            "per_key_state_f1_fixed_0_5": _json_floats(key_f1),
            "key_state_micro_accuracy_fixed_0_5": micro,
            "joint_exact_match_accuracy_fixed_0_5": joint,
            "predicted_positive_rate_fixed_0_5": float(predicted.mean()),
            "segment_bounded_combined_event_f1_fixed_0_5": {
                "exact": exact_event,
                "plus_minus_2": collar_event,
                "exact_per_key": exact_by_key,
                "plus_minus_2_per_key": collar_by_key,
            },
        },
        "baselines": {
            "prevalence_macro": float(np.mean(list(prevalence.values()))),
            "prevalence_per_key": prevalence,
            "always_released": {
                "key_state_micro_accuracy": released_micro,
                "joint_exact_match_accuracy": released_joint,
            },
            "one_frame_persistence": {
                "key_state_micro_accuracy": persistence_micro,
                "joint_exact_match_accuracy": persistence_joint,
                "segment_bounded_combined_event_f1": {
                    "exact": persistence_exact,
                    "plus_minus_2": persistence_collar,
                    "exact_per_key": persistence_exact_by_key,
                    "plus_minus_2_per_key": persistence_collar_by_key,
                },
            },
        },
        "per_stream": per_stream,
    }


def _strip_keys(value: Mapping[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in keys}


def _validate_configs(configs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    weighted = configs["weighted_tcn_lr3e4"]
    natural = configs["natural_tcn_control"]
    event = configs["event_head_tcn_direct"]

    if _strip_keys(weighted, {"_note", "class_balance", "transition_weight"}) != _strip_keys(
        natural, {"_note", "class_balance", "transition_weight"}
    ):
        raise ValueError(
            "natural control differs from weighted TCN beyond the frozen objective fields"
        )
    if weighted.get("class_balance") is not True or float(
        weighted.get("transition_weight", -1)
    ) != 8.0:
        raise ValueError("weighted TCN objective receipt changed")
    if natural.get("class_balance") is not False or float(
        natural.get("transition_weight", -1)
    ) != 1.0:
        raise ValueError("natural TCN objective receipt changed")
    forbidden = {
        "event_latch",
        "event_class_balance_max",
        "state_loss_weight",
        "onset_loss_weight",
        "release_loss_weight",
    }
    if forbidden.intersection(natural):
        raise ValueError("natural control unexpectedly contains event-head fields")

    for role, expected_lr in (
        ("weighted_tcn_lr1e4", 0.0001),
        ("weighted_tcn_lr1e3", 0.001),
    ):
        candidate = configs[role]
        if _strip_keys(weighted, {"_note", "learning_rate"}) != _strip_keys(
            candidate, {"_note", "learning_rate"}
        ):
            raise ValueError(f"{role} differs from 3e-4 beyond learning rate")
        if float(candidate.get("learning_rate", -1)) != expected_lr:
            raise ValueError(f"{role} has the wrong learning rate")

    event_only = {
        "_note",
        "class_balance_max",
        "initial_train_eval",
        "event_class_balance_max",
        "event_latch",
        "state_loss_weight",
        "onset_loss_weight",
        "release_loss_weight",
    }
    if _strip_keys(natural, event_only) != _strip_keys(event, event_only):
        raise ValueError(
            "event-head run and natural control differ beyond event-head fields"
        )
    expected_event = {
        "event_latch": True,
        "state_loss_weight": 1.0,
        "onset_loss_weight": 0.5,
        "release_loss_weight": 0.5,
    }
    for field, expected in expected_event.items():
        if event.get(field) != expected:
            raise ValueError(f"event-head config has unexpected {field}")

    for role, config in configs.items():
        if int(config.get("seed", -1)) != 0:
            raise ValueError(f"{role}: seed is not zero")
    return {
        "natural_control_differs_only_in_objective_fields": True,
        "lr_variants_differ_only_in_learning_rate": True,
        "event_head_differs_only_in_declared_event_fields": True,
    }


def _contrast(
    left_role: str,
    right_role: str,
    role_to_run: Mapping[str, str],
    runs: Mapping[str, Mapping[str, Any]],
    study: Mapping[str, Any],
) -> dict[str, Any]:
    left_id, right_id = role_to_run[left_role], role_to_run[right_role]
    left, right = runs[left_id], runs[right_id]
    left_metrics = left["metrics"]
    right_metrics = right["metrics"]
    aggregate_delta = float(left_metrics["macro_ap"] - right_metrics["macro_ap"])
    stream_deltas = {
        session_id: float(
            left["per_stream"][session_id]["macro_ap"]
            - right["per_stream"][session_id]["macro_ap"]
        )
        for session_id in left["support_session_ids"]
    }
    positive = sum(delta > 0.0 for delta in stream_deltas.values())
    negative = sum(delta < 0.0 for delta in stream_deltas.values())
    threshold = float(
        study["decision_rules"]["material_macro_ap_effect"][
            "minimum_absolute_macro_ap_delta"
        ]
    )
    minimum_streams = int(
        study["decision_rules"]["material_macro_ap_effect"][
            "minimum_same_direction_stream_deltas_out_of_8"
        ]
    )
    if aggregate_delta >= threshold and positive >= minimum_streams:
        effect = "materially_positive"
    elif aggregate_delta <= -threshold and negative >= minimum_streams:
        effect = "materially_negative"
    else:
        effect = "inconclusive_at_preregistered_effect_size"

    exact_delta = float(
        left_metrics["segment_bounded_combined_event_f1_fixed_0_5"]["exact"]
        - right_metrics["segment_bounded_combined_event_f1_fixed_0_5"]["exact"]
    )
    collar_delta = float(
        left_metrics["segment_bounded_combined_event_f1_fixed_0_5"]["plus_minus_2"]
        - right_metrics["segment_bounded_combined_event_f1_fixed_0_5"][
            "plus_minus_2"
        ]
    )
    guard_rule = study["decision_rules"]["event_regression_guards"]
    exact_pass = exact_delta >= float(guard_rule["exact_event_f1_minimum_delta"])
    collar_pass = collar_delta >= float(
        guard_rule["plus_minus_2_event_f1_minimum_delta"]
    )
    guard_pass = exact_pass and collar_pass
    qualified = effect
    if effect == "materially_positive" and not guard_pass:
        qualified = "materially_positive_with_fixed_event_timing_regression"

    return {
        "left_role": left_role,
        "left_run_id": left_id,
        "right_role": right_role,
        "right_run_id": right_id,
        "macro_ap_delta": aggregate_delta,
        "per_stream_macro_ap_delta": stream_deltas,
        "strictly_positive_streams": positive,
        "strictly_negative_streams": negative,
        "effect": effect,
        "qualified_conclusion": qualified,
        "event_regression_guard": {
            "exact_event_f1_delta": exact_delta,
            "plus_minus_2_event_f1_delta": collar_delta,
            "exact_floor_passed": exact_pass,
            "plus_minus_2_floor_passed": collar_pass,
            "passed": guard_pass,
        },
    }


def _resolve_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return _require_hex(result, _GIT_SHA_RE, "decision Git commit")


def build_decision(
    *,
    repo: Path,
    preregistration_path: Path,
    preregistration_git_commit: str,
    receipt_manifest_path: Path,
    decision_git_commit: str,
) -> dict[str, Any]:
    """Build and hash a complete decision record without writing it."""

    repo = repo.resolve()
    preregistration_path = preregistration_path.resolve()
    receipt_manifest_path = receipt_manifest_path.resolve()
    preregistration_git_commit = _verify_committed_preregistration(
        repo, preregistration_path, preregistration_git_commit
    )
    study = _load_json(preregistration_path)
    receipts = _load_json(receipt_manifest_path)
    if receipts.get("schema_version") != 1 or not isinstance(receipts.get("runs"), dict):
        raise ValueError("receipt manifest must have schema_version 1 and a runs object")
    decision_git_commit = _require_hex(
        decision_git_commit, _GIT_SHA_RE, "decision Git commit"
    )

    population = study["frozen_population"]
    session_ids = list(population["session_ids"])
    expected_lengths = population["expected_frames_by_session"]
    expected_frames = int(population["expected_active_frames"])
    consulted = study["consulted_runs"]
    if set(receipts["runs"]) != set(consulted):
        raise ValueError(
            "receipt roles must exactly equal consulted_runs: "
            f"{sorted(receipts['runs'])} != {sorted(consulted)}"
        )

    runs: dict[str, Any] = {}
    role_to_run: dict[str, str] = {}
    loaded: dict[str, dict[str, Any]] = {}
    configs: dict[str, Mapping[str, Any]] = {}
    reference_truth: np.ndarray | None = None
    reference_active: np.ndarray | None = None

    for role, disclosed in consulted.items():
        receipt = receipts["runs"][role]
        if not isinstance(receipt, dict):
            raise ValueError(f"{role}: receipt must be an object")
        run_id = disclosed["run_id"]
        if receipt.get("run_id", run_id) != run_id:
            raise ValueError(f"{role}: receipt run_id differs from preregistration")
        report_path = _resolve(repo, receipt.get("report_path"), f"{role}.report_path")
        sidecar_path = _resolve(repo, receipt.get("sidecar_path"), f"{role}.sidecar_path")
        _require_canonical_result_path(
            repo,
            report_path,
            run_id,
            "_final_nitrogen_val.json",
            f"{role}.report_path",
        )
        _require_canonical_result_path(
            repo,
            sidecar_path,
            run_id,
            "_final_nitrogen_val_preds.npz",
            f"{role}.sidecar_path",
        )
        config_path = _resolve(repo, receipt.get("config_path"), f"{role}.config_path")
        launcher_path = _resolve(
            repo, receipt.get("launcher_path"), f"{role}.launcher_path"
        )
        weights = receipt.get("weights")
        if weights != "final":
            raise ValueError(f"{role}: only final weights are admissible")
        final_step = int(receipt.get("final_step", -1))
        expected_final_step = int(receipt.get("expected_final_step", -2))
        if final_step < 1 or final_step != expected_final_step:
            raise ValueError(f"{role}: final checkpoint is not the fixed endpoint")
        config = _load_json(config_path)
        if int(config.get("max_steps", -1)) != expected_final_step:
            raise ValueError(f"{role}: config endpoint does not match receipt")
        configs[role] = config

        start = _parse_utc(receipt.get("training_start_utc"), f"{role}.training_start_utc")
        end = _parse_utc(receipt.get("training_end_utc"), f"{role}.training_end_utc")
        if datetime.fromisoformat(end.replace("Z", "+00:00")) < datetime.fromisoformat(
            start.replace("Z", "+00:00")
        ):
            raise ValueError(f"{role}: training_end_utc precedes training_start_utc")
        checkpoint_sha = _require_hex(
            receipt.get("checkpoint_sha256"), _SHA256_RE, f"{role}.checkpoint_sha256"
        )
        implementation_commit = _require_hex(
            receipt.get("implementation_git_commit"),
            _GIT_SHA_RE,
            f"{role}.implementation_git_commit",
        )
        if role in _NEW_RUN_ROLES and receipt.get("b1_used_before_decision") is not False:
            raise ValueError(f"{role}: B1 embargo receipt is not false")

        sidecar = _load_frozen_sidecar(sidecar_path, session_ids, expected_lengths)
        _validate_final_report(report_path, run_id, sidecar)
        if int(sidecar["truth"].shape[0]) != expected_frames:
            raise ValueError(f"{role}: frozen support is not {expected_frames} rows")
        if reference_truth is None:
            reference_truth = sidecar["truth"]
            reference_active = sidecar["active"]
        elif not np.array_equal(reference_truth, sidecar["truth"]):
            raise ValueError(f"{role}: truth differs from the first consulted run")
        elif not np.array_equal(reference_active, sidecar["active"]):
            raise ValueError(f"{role}: active support differs from the first consulted run")

        score = _score_run(sidecar)
        score.update(
            {
                "run_id": run_id,
                "study_role": role,
                "inferential_role": disclosed["role"],
                "config_path": _relative(repo, config_path),
                "config_sha256": _sha256(config_path),
                "launcher_path": _relative(repo, launcher_path),
                "launcher_sha256": _sha256(launcher_path),
                "checkpoint_sha256": checkpoint_sha,
                "evaluation_report_path": _relative(repo, report_path),
                "evaluation_report_sha256": _sha256(report_path),
                "prediction_sidecar_path": _relative(repo, sidecar_path),
                "prediction_sidecar_sha256": _sha256(sidecar_path),
                "implementation_git_commit": implementation_commit,
                "training_start_utc": start,
                "training_end_utc": end,
                "final_step": final_step,
                "expected_final_step": expected_final_step,
                "evaluation_weights": "final",
                "support_session_ids": session_ids,
                "active_frames": expected_frames,
                "finite_aligned_arrays": True,
                "valid": True,
            }
        )
        if run_id in runs:
            raise ValueError(f"duplicate run ID in consulted runs: {run_id}")
        runs[run_id] = score
        role_to_run[role] = run_id
        loaded[role] = sidecar

    config_gates = _validate_configs(configs)
    objective = _contrast(
        "natural_tcn_control", "weighted_tcn_lr3e4", role_to_run, runs, study
    )
    event_gradient = _contrast(
        "event_head_tcn_direct", "natural_tcn_control", role_to_run, runs, study
    )
    lr_contrasts = {
        role: _contrast(role, "weighted_tcn_lr3e4", role_to_run, runs, study)
        for role in ("weighted_tcn_lr1e4", "weighted_tcn_lr1e3")
    }
    lr_sensitive = any(
        contrast["effect"] == "materially_positive"
        for contrast in lr_contrasts.values()
    )

    gru = runs[role_to_run["matched_gru"]]["metrics"]
    gru_crossings: dict[str, list[str]] = {}
    for role in ("weighted_tcn_lr1e4", "weighted_tcn_lr1e3"):
        candidate = runs[role_to_run[role]]["metrics"]
        crossed = [
            metric
            for metric in _HIGHER_IS_BETTER_METRICS
            if float(candidate[metric]) > float(gru[metric])
        ]
        for metric in ("exact", "plus_minus_2"):
            if float(
                candidate["segment_bounded_combined_event_f1_fixed_0_5"][metric]
            ) > float(gru["segment_bounded_combined_event_f1_fixed_0_5"][metric]):
                crossed.append(f"segment_bounded_{metric}_event_f1_fixed_0_5")
        gru_crossings[role] = crossed

    truth_digest = hashlib.sha256()
    assert reference_truth is not None
    truth_digest.update(str(reference_truth.shape).encode("ascii"))
    truth_digest.update(reference_truth.astype(np.uint8, copy=False).tobytes())
    result: dict[str, Any] = {
        "schema_version": 1,
        "study_id": study["study_id"],
        "preregistration": {
            "path": _relative(repo, preregistration_path),
            "sha256": _sha256(preregistration_path),
            "git_commit": preregistration_git_commit,
            "status": study["status"],
            "preregistered_at_utc": study["preregistered_at_utc"],
        },
        "artifact_receipts": {
            "path": _relative(repo, receipt_manifest_path),
            "sha256": _sha256(receipt_manifest_path),
            "schema_version": receipts["schema_version"],
        },
        "evaluation_population": {
            "surface": population["surface"],
            "role": population["role"],
            "session_ids": session_ids,
            "frames_by_session": expected_lengths,
            "active_frames": expected_frames,
            "keys": list(KEY_ORDER),
            "truth_sha256": truth_digest.hexdigest(),
            "probability_policy": population["probability_policy"],
            "threshold_policy": population["threshold_policy"],
            "event_policy": population["event_policy"],
            "oracle_metrics_used": False,
            "calibration_used": False,
            "b1_used": False,
        },
        "config_validity_gates": config_gates,
        "runs": runs,
        "decision": {
            "all_consulted_runs": [
                {
                    "study_role": role,
                    "run_id": role_to_run[role],
                    "inferential_role": consulted[role]["role"],
                }
                for role in consulted
            ],
            "multiplicity_disclosure": study["multiplicity"],
            "objective_contrast": objective,
            "event_head_gradient_contrast": event_gradient,
            "lr_sensitivity": {
                "reference_role": "weighted_tcn_lr3e4",
                "candidate_contrasts": lr_contrasts,
                "optimizer_recipe_sensitive": lr_sensitive,
                "conclusion": (
                    "optimizer_recipe_sensitive_at_preregistered_screen"
                    if lr_sensitive
                    else "no_material_lr_sensitivity_detected_at_tested_rates"
                ),
                "headline_remains_weighted_tcn_lr3e4": True,
                "sensitivity_candidate_promoted": False,
            },
            "event_regression_guards": {
                "objective_contrast": objective["event_regression_guard"],
                "event_head_gradient_contrast": event_gradient[
                    "event_regression_guard"
                ],
                **{
                    role: contrast["event_regression_guard"]
                    for role, contrast in lr_contrasts.items()
                },
            },
            "gru_crossing": {
                "metrics_exceeding_gru_by_lr_candidate": gru_crossings,
                "matched_architecture_win_allowed": False,
                "reason": study["decision_rules"]["gru_crossing"]["rule"],
            },
            "single_seed_limit": study["decision_rules"]["single_seed"]["rule"],
            "y4n_decision_frozen_before_b1": all(
                receipts["runs"][role].get("b1_used_before_decision") is False
                for role in _NEW_RUN_ROLES
            ),
            "decision_record_sha256_scope": (
                "canonical JSON with decision_record_sha256 set to null; "
                "the self-field is excluded to avoid a circular hash"
            ),
            "decision_record_sha256": None,
            "decision_git_commit": decision_git_commit,
        },
    }
    result["decision"]["decision_record_sha256"] = _canonical_sha256(result)
    return result


def write_decision(path: Path, result: Mapping[str, Any]) -> None:
    """Atomically write one deterministic JSON decision record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("experiments/configs/tcn_control_lr_decision.json"),
    )
    parser.add_argument(
        "--preregistration-git-commit",
        default=_PREREGISTRATION_COMMIT,
        help="pre-launch commit containing the exact frozen study contract",
    )
    parser.add_argument(
        "--receipts",
        type=Path,
        default=Path("results/idm/tcn_control_lr_y4n_receipts.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/idm/tcn_control_lr_y4n_decision.json"),
    )
    parser.add_argument(
        "--decision-git-commit",
        help="40-character repository HEAD under which the decision is authored",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    preregistration = args.preregistration
    receipts = args.receipts
    output = args.out
    if not preregistration.is_absolute():
        preregistration = repo / preregistration
    if not receipts.is_absolute():
        receipts = repo / receipts
    if not output.is_absolute():
        output = repo / output
    commit = args.decision_git_commit or _resolve_head(repo)
    result = build_decision(
        repo=repo,
        preregistration_path=preregistration,
        preregistration_git_commit=args.preregistration_git_commit,
        receipt_manifest_path=receipts,
        decision_git_commit=commit,
    )
    write_decision(output, result)
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": result["decision"]["decision_record_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
