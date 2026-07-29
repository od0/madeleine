"""Build the frozen three-run provisional-blend decision on y4n later-eight.

This scorer has no interface for B1, val-B, or the sealed untouched session.
It validates the committed blend contract, the already-committed pure-NitroGen
later-eight evidence, and both fixed-only final-weight blend releases.  It then
recomputes every decision metric and the preregistered candidate rule from the
prediction arrays.  No threshold or calibration parameter is fit here.

The pure-NitroGen prediction artifact contains all sixteen y4n streams.  Its
later-eight row is derived by the exact stream IDs frozen in the blend
contract; the full source report and sidecar hashes remain in the receipt so a
derived row cannot be mistaken for a standalone release.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from badeline.metrics import (
    match_event_counts,
    score_events,
    transition_events,
)
from data.schema import KEY_ORDER
from experiments.eval_provisional_blend_gru import (
    ARM_SPECS,
    COMPARISON_SCHEMA_VERSION,
    CONTRACT_RELATIVE_PATH,
    MARKER_SCHEMA_VERSION,
    REFERENCE_RUN_ID,
    SCHEMA_VERSION as ARM_REPORT_SCHEMA_VERSION,
    STUDY_ID,
    Y4N_SURFACE,
    _sampling_receipt,
    validate_comparison_value,
    validate_contract,
)
from experiments.eval_tcn_control_lr_b1 import fixed_metric_report, sha256_file
from experiments.eval_wild_provisional_gru import (
    Y4N_BASE_SESSION_IDS,
    Y4N_FRAMES,
    Y4N_STREAM_IDS,
    Y4N_STREAM_LENGTHS,
    Y4N_TRUTH_SHA256,
)
from experiments.score_tcn_control_lr_decision import (
    _canonical_sha256,
    _load_frozen_sidecar,
    _score_run,
)


CONTRACT_COMMIT = "8e98f949aab976d89f801e9e6fdca0cb4ab9b53a"
PURE_N_DECISION_RELATIVE_PATH = Path("results/idm/tcn_control_lr_y4n_decision.json")
OUTPUT_RELATIVE_PATH = Path("results/idm/provisional_blend_y4n_decision.json")
PURE_N_DECISION_STUDY_ID = "tcn_control_lr_y4n_later8_s0"
PURE_N_ROLE = "matched_gru"
WRAPPER_MARKER_SCHEMA_VERSION = "madeleine.provisional-blend-arm-wrapper.v1"
SHUFFLE_SEEDS = tuple(range(10))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MINIMUM_MACRO_AP_DELTA = 0.005
MINIMUM_PER_KEY_AP_WINS = 4
MAXIMUM_PLUS_MINUS_2_EVENT_F1_LOSS = 0.005


def _json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _relative(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _canonical_path(repo: Path, path: Path, relative: Path, field: str) -> Path:
    expected = (repo.resolve() / relative).resolve()
    if path.resolve() != expected:
        raise ValueError(f"{field} must be {relative.as_posix()}")
    if not path.is_file():
        raise FileNotFoundError(f"{field} is missing: {path}")
    return path


def _canonical_array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _same_number(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    return left == right


def _require_nested_equal(
    observed: object,
    expected: object,
    field: str,
    *,
    tolerance: float = 1e-12,
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ValueError(f"{field} object keys changed")
        for key, value in expected.items():
            _require_nested_equal(
                observed[key], value, f"{field}.{key}", tolerance=tolerance
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"{field} list shape changed")
        for index, value in enumerate(expected):
            _require_nested_equal(
                observed[index], value, f"{field}[{index}]", tolerance=tolerance
            )
        return
    if not _same_number(observed, expected, tolerance=tolerance):
        raise ValueError(f"{field} changed: {observed!r} != {expected!r}")


def _stream_slices(lengths: Sequence[int]) -> list[slice]:
    result: list[slice] = []
    start = 0
    for length in lengths:
        end = start + int(length)
        result.append(slice(start, end))
        start = end
    return result


def _load_exact_later_eight(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "y_true",
            "y_prob",
            "input_active",
            "session_lengths",
            "session_ids",
        }
        if set(archive.files) != required:
            raise ValueError(f"{path}: prediction sidecar fields changed")
        truth = np.asarray(archive["y_true"])
        probability = np.asarray(archive["y_prob"])
        active = np.asarray(archive["input_active"])
        lengths = np.asarray(archive["session_lengths"])
        session_ids = np.asarray(archive["session_ids"])

    expected_shape = (Y4N_FRAMES, len(KEY_ORDER))
    if truth.shape != expected_shape or probability.shape != expected_shape:
        raise ValueError(f"{path}: later-eight tensor support changed")
    if truth.dtype != np.uint8 or probability.dtype != np.float32:
        raise ValueError(f"{path}: truth/probability dtypes changed")
    if active.dtype != np.uint8 or active.shape != (Y4N_FRAMES,):
        raise ValueError(f"{path}: active-mask dtype or support changed")
    if lengths.dtype != np.int64 or lengths.tolist() != Y4N_STREAM_LENGTHS:
        raise ValueError(f"{path}: stream boundaries changed")
    if session_ids.tolist() != Y4N_STREAM_IDS:
        raise ValueError(f"{path}: stream identities changed")
    if not np.all(active == 1):
        raise ValueError(f"{path}: later-eight contains inactive rows")
    if not np.all(np.isin(truth, (0, 1))):
        raise ValueError(f"{path}: truth is not binary")
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError(f"{path}: probabilities are not finite values in [0,1]")
    if _canonical_array_sha256(truth) != Y4N_TRUTH_SHA256:
        raise ValueError(f"{path}: mapped-label truth receipt changed")
    return {
        "truth": truth.astype(bool, copy=False),
        "probability": probability.astype(np.float64, copy=False),
        "active": active.astype(bool, copy=False),
        "lengths": lengths,
        "session_ids": session_ids.tolist(),
    }


def segment_bounded_shuffled_event_baseline(
    truth: np.ndarray,
    lengths: Sequence[int],
    *,
    seeds: Sequence[int] = SHUFFLE_SEEDS,
) -> dict[str, Any]:
    """Place true event counts uniformly within their original stream.

    Onset and release counts are preserved independently for every key and
    stream.  Sampling and matching are both stream bounded, so neither the
    random anchor nor its collar can cross a concatenation boundary.
    """

    truth = np.asarray(truth).astype(bool, copy=False)
    lengths = [int(value) for value in lengths]
    if truth.shape != (sum(lengths), len(KEY_ORDER)):
        raise ValueError("shuffled-event truth and stream lengths do not align")
    if not seeds or len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("shuffled-event seeds must be a non-empty unique list")
    bounds = _stream_slices(lengths)
    collars = {"exact": 0, "plus_minus_2": 2}
    seed_scores: dict[str, dict[str, dict[str, float]]] = {}

    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        key_scores: dict[str, dict[str, float]] = {}
        for column, key in enumerate(KEY_ORDER):
            true_on, true_off = transition_events(
                truth[:, column], boundaries=lengths
            )
            random_on: list[np.ndarray] = []
            random_off: list[np.ndarray] = []
            for stream_slice in bounds:
                start, end = int(stream_slice.start), int(stream_slice.stop)
                on_count = int(
                    np.searchsorted(true_on, end)
                    - np.searchsorted(true_on, start)
                )
                off_count = int(
                    np.searchsorted(true_off, end)
                    - np.searchsorted(true_off, start)
                )
                frames = np.arange(start, end, dtype=np.int64)
                random_on.append(
                    np.sort(rng.choice(frames, size=on_count, replace=False))
                )
                random_off.append(
                    np.sort(rng.choice(frames, size=off_count, replace=False))
                )
            predicted_on = np.sort(np.concatenate(random_on))
            predicted_off = np.sort(np.concatenate(random_off))
            combined_true = len(true_on) + len(true_off)
            combined_pred = len(predicted_on) + len(predicted_off)
            key_scores[key] = {}
            for name, collar in collars.items():
                matched = match_event_counts(
                    true_on, predicted_on, collar, boundaries=lengths
                ) + match_event_counts(
                    true_off, predicted_off, collar, boundaries=lengths
                )
                key_scores[key][name] = float(
                    score_events(combined_true, combined_pred, matched)["f1"]
                )
        seed_scores[str(int(seed))] = key_scores

    per_key: dict[str, dict[str, float]] = {}
    for key in KEY_ORDER:
        per_key[key] = {
            name: float(
                np.mean(
                    [seed_scores[str(int(seed))][key][name] for seed in seeds]
                )
            )
            for name in collars
        }
    macro = {
        name: float(np.mean([per_key[key][name] for key in KEY_ORDER]))
        for name in collars
    }
    per_seed_macro = {
        str(int(seed)): {
            name: float(
                np.mean(
                    [seed_scores[str(int(seed))][key][name] for key in KEY_ORDER]
                )
            )
            for name in collars
        }
        for seed in seeds
    }
    return {
        "policy": (
            "preserve onset and release counts independently per key and "
            "stream; sample uniformly without replacement within that stream; "
            "one-to-one segment-bounded matching"
        ),
        "seeds": [int(seed) for seed in seeds],
        "per_key_mean": per_key,
        "macro_mean": macro,
        "per_seed_macro": per_seed_macro,
    }


def _validate_pure_n_decision(
    repo: Path, path: Path, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _canonical_path(
        repo,
        path,
        PURE_N_DECISION_RELATIVE_PATH,
        "pure-N later-eight decision",
    )
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain",
            "--",
            PURE_N_DECISION_RELATIVE_PATH.as_posix(),
        ],
        check=False,
        capture_output=True,
    )
    if status.returncode or status.stdout:
        raise ValueError("pure-N later-eight decision is not clean and committed")
    decision = _json_object(path, "pure-N later-eight decision")
    if decision.get("study_id") != PURE_N_DECISION_STUDY_ID:
        raise ValueError("pure-N later-eight decision study changed")
    population = decision.get("evaluation_population")
    expected_population = {
        "session_ids": Y4N_STREAM_IDS,
        "active_frames": Y4N_FRAMES,
        "truth_sha256": Y4N_TRUTH_SHA256,
        "oracle_metrics_used": False,
        "calibration_used": False,
        "b1_used": False,
    }
    if not isinstance(population, Mapping):
        raise ValueError("pure-N decision lacks evaluation population")
    for field, expected in expected_population.items():
        if population.get(field) != expected:
            raise ValueError(f"pure-N decision changed population.{field}")
    internal = decision.get("decision")
    if not isinstance(internal, Mapping):
        raise ValueError("pure-N decision lacks its decision receipt")
    internal_hash = _require_sha256(
        internal.get("decision_record_sha256"),
        "pure-N decision canonical hash",
    )
    unhashed = copy.deepcopy(decision)
    unhashed["decision"]["decision_record_sha256"] = None
    if _canonical_sha256(unhashed) != internal_hash:
        raise ValueError("pure-N decision canonical hash changed")
    run = decision.get("runs", {}).get(REFERENCE_RUN_ID)
    if not isinstance(run, Mapping) or run.get("study_role") != PURE_N_ROLE:
        raise ValueError("pure-N reference row is missing")
    if run.get("evaluation_weights") != "final" or run.get("valid") is not True:
        raise ValueError("pure-N reference is not a valid final-weight row")
    if run.get("checkpoint_sha256") != contract["reference"]["checkpoint_sha256"]:
        raise ValueError("pure-N checkpoint differs from blend contract")

    report_path = _canonical_path(
        repo,
        repo / str(run.get("evaluation_report_path", "")),
        Path("results/idm") / f"{REFERENCE_RUN_ID}_final_nitrogen_val.json",
        "pure-N full source report",
    )
    sidecar_path = _canonical_path(
        repo,
        repo / str(run.get("prediction_sidecar_path", "")),
        Path("results/idm") / f"{REFERENCE_RUN_ID}_final_nitrogen_val_preds.npz",
        "pure-N full source sidecar",
    )
    report_sha = sha256_file(report_path)
    sidecar_sha = sha256_file(sidecar_path)
    if report_sha != run.get("evaluation_report_sha256"):
        raise ValueError("pure-N full source report SHA-256 changed")
    if sidecar_sha != run.get("prediction_sidecar_sha256"):
        raise ValueError("pure-N full source sidecar SHA-256 changed")
    source_report = _json_object(report_path, "pure-N full source report")
    if (
        source_report.get("weights") != "final"
        or source_report.get("label_kind") != "mapped_foreign_nitrogen"
        or Path(str(source_report.get("run", ""))).name != REFERENCE_RUN_ID
        or len(source_report.get("sessions", [])) != 16
    ):
        raise ValueError("pure-N full source report semantics changed")

    selected = _load_frozen_sidecar(
        sidecar_path,
        Y4N_STREAM_IDS,
        dict(zip(Y4N_STREAM_IDS, Y4N_STREAM_LENGTHS, strict=True)),
    )
    if _canonical_array_sha256(
        np.asarray(selected["truth"], dtype=np.uint8)
    ) != Y4N_TRUTH_SHA256:
        raise ValueError("pure-N derived later-eight truth changed")
    score = _score_run(selected)
    _require_nested_equal(score["metrics"], run.get("metrics"), "pure-N metrics")
    _require_nested_equal(
        score["baselines"], run.get("baselines"), "pure-N baselines"
    )
    release = {
        "complete": True,
        "weights": "final",
        "checkpoint_sha256": run["checkpoint_sha256"],
        "report_sha256": report_sha,
        "sidecar_sha256": sidecar_sha,
        # The committed derived decision is the completion receipt for this
        # historical all-sixteen source artifact.
        "completion_marker_sha256": sha256_file(path),
        "completion_marker_kind": "committed_derived_later_eight_decision",
        "source_report_path": _relative(repo, report_path),
        "source_report_support": "all_sixteen_y4n_streams",
        "source_sidecar_path": _relative(repo, sidecar_path),
        "source_sidecar_support": "all_sixteen_y4n_streams",
        "derived_evidence_path": _relative(repo, path),
        "derived_evidence_sha256": sha256_file(path),
        "derived_evidence_canonical_sha256": internal_hash,
        "metrics": score["metrics"],
        "baselines": score["baselines"],
        "source_sampling": None,
        "memorization_receipts": {
            "status": "not_part_of_historical_reference_release",
            "candidate_rule_dependency": False,
        },
    }
    return release, selected, decision


def _mean_mapping(value: Mapping[str, Any], field: str) -> float:
    numbers = [float(value[key]) for key in KEY_ORDER]
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError(f"{field} contains a non-finite value")
    return float(np.mean(numbers))


def _source_membership_receipt(contract: Mapping[str, Any]) -> dict[str, Any]:
    sources = contract["sources"]
    local = sources["local"]
    wild = sources["wild_provisional"]
    nitrogen = sources["nitrogen"]
    return {
        "nitrogen": {
            "session_list_sha256": nitrogen["session_list_sha256"],
            "feature_validation_sha256": nitrogen["feature_validation_sha256"],
        },
        "local": {
            "generation": "own_v3",
            "build_manifest_sha256": local["build_manifest_sha256"],
            "train_sessions_sha256": local["train_sessions_sha256"],
            "train_shard_sha256": local["train_shard_sha256"],
        },
        "wild_provisional": {
            "admitted_hours": 0.0,
            "source_manifest_sha256": wild["source_manifest_sha256"],
            "session_list_sha256": wild["session_list_sha256"],
            "feature_validation_marker_sha256": wild[
                "feature_validation_marker_sha256"
            ],
        },
    }


def _validate_blend_release(
    repo: Path,
    contract: Mapping[str, Any],
    contract_sha256: str,
    contract_commit: str,
    arm_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = ARM_SPECS[arm_name]
    run_id = spec["run_id"]
    results = repo / "results" / "idm"
    report_path = _canonical_path(
        repo,
        results / f"{run_id}_final_y4n_later8_fixed.json",
        Path("results/idm") / f"{run_id}_final_y4n_later8_fixed.json",
        f"{arm_name} report",
    )
    sidecar_path = _canonical_path(
        repo,
        results / f"{run_id}_final_y4n_later8_fixed_preds.npz",
        Path("results/idm") / f"{run_id}_final_y4n_later8_fixed_preds.npz",
        f"{arm_name} sidecar",
    )
    marker_path = _canonical_path(
        repo,
        results / f".{run_id}_final_y4n_later8_fixed_done.json",
        Path("results/idm") / f".{run_id}_final_y4n_later8_fixed_done.json",
        f"{arm_name} completion marker",
    )
    wrapper_path = _canonical_path(
        repo,
        results / f".{run_id}_train_and_y4n_done.json",
        Path("results/idm") / f".{run_id}_train_and_y4n_done.json",
        f"{arm_name} wrapper marker",
    )
    run_dir = results / run_id
    paths = {
        "config": run_dir / "config.json",
        "checkpoint": run_dir / "model.pt",
        "run_meta": run_dir / "run_meta.json",
        "sampling": run_dir / "source_sampling_receipt.json",
        "log": run_dir / "log.jsonl",
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError(f"{arm_name} run receipt is incomplete")

    report = _json_object(report_path, f"{arm_name} report")
    marker = _json_object(marker_path, f"{arm_name} completion marker")
    wrapper = _json_object(wrapper_path, f"{arm_name} wrapper marker")
    exact_report = {
        "schema_version": ARM_REPORT_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "arm": arm_name,
        "run_id": run_id,
        "surface": Y4N_SURFACE,
        "weights": "final",
        "sessions": Y4N_BASE_SESSION_IDS,
        "label_kind": "mapped_foreign_nitrogen",
    }
    for field, expected in exact_report.items():
        if report.get(field) != expected:
            raise ValueError(f"{arm_name} report changed {field}")
    report_contract = report.get("contract")
    if (
        not isinstance(report_contract, Mapping)
        or report_contract.get("sha256") != contract_sha256
        or report_contract.get("commit") != contract_commit
        or not str(report_contract.get("path", "")).endswith(
            CONTRACT_RELATIVE_PATH.as_posix()
        )
    ):
        raise ValueError(f"{arm_name} report contract hash changed")
    expected_support = {
        "all_frames": Y4N_FRAMES,
        "input_active_frames": Y4N_FRAMES,
        "streams": len(Y4N_STREAM_IDS),
        "session_ids": Y4N_STREAM_IDS,
        "stream_lengths": Y4N_STREAM_LENGTHS,
        "truth_sha256": Y4N_TRUTH_SHA256,
        "truth_hash_includes_shape": True,
        "finite_aligned_arrays": True,
    }
    _require_nested_equal(
        report.get("support"), expected_support, f"{arm_name} report support"
    )
    policy = report.get("evaluation_policy")
    if not isinstance(policy, Mapping) or policy != {
        "raw_sigmoid_probabilities": True,
        "fixed_state_threshold": 0.5,
        "fixed_event_threshold": 0.5,
        "threshold_parameters_fitted": False,
        "calibration_parameters_fitted": False,
        "checkpoint_selected_on_this_surface": False,
        "sealed_untouched_session_accessed": False,
    }:
        raise ValueError(f"{arm_name} report is not the fixed-only release")

    sidecar = _load_exact_later_eight(sidecar_path)
    recomputed_fixed = fixed_metric_report(
        sidecar["truth"],
        sidecar["probability"],
        sidecar["active"],
        sidecar["lengths"].tolist(),
    )
    _require_nested_equal(
        report.get("fixed_metrics"),
        recomputed_fixed,
        f"{arm_name} fixed metrics",
    )
    score = _score_run(sidecar)
    report_sha = sha256_file(report_path)
    sidecar_sha = sha256_file(sidecar_path)
    report_sidecar = report.get("prediction_sidecar")
    if (
        not isinstance(report_sidecar, Mapping)
        or Path(str(report_sidecar.get("path", ""))).name != sidecar_path.name
        or report_sidecar.get("sha256") != sidecar_sha
    ):
        raise ValueError(f"{arm_name} report sidecar binding changed")

    run_receipt = report.get("run_receipt")
    if not isinstance(run_receipt, Mapping):
        raise ValueError(f"{arm_name} report lacks a run receipt")
    expected_hashes = {
        "config_sha256": sha256_file(paths["config"]),
        "checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "run_meta_sha256": sha256_file(paths["run_meta"]),
        "source_sampling_receipt_sha256": sha256_file(paths["sampling"]),
        "training_log_sha256": sha256_file(paths["log"]),
    }
    for field, expected in expected_hashes.items():
        if run_receipt.get(field) != expected:
            raise ValueError(f"{arm_name} run receipt changed {field}")
    if (
        run_receipt.get("checkpoint_steps") != 14_265
        or run_receipt.get("evaluation_weights") != "final_state_dict"
        or run_receipt.get("initialization") != "from_scratch"
    ):
        raise ValueError(f"{arm_name} is not the fixed final from-scratch run")

    sampling = _json_object(paths["sampling"], f"{arm_name} sampling receipt")
    validated_sampling = _sampling_receipt(sampling, contract, arm_name)
    if run_receipt.get("source_sampling") != validated_sampling:
        raise ValueError(f"{arm_name} embedded source sampling changed")
    run_meta = _json_object(paths["run_meta"], f"{arm_name} run metadata")
    if run_meta.get("source_sampling") != validated_sampling:
        raise ValueError(f"{arm_name} run metadata source sampling changed")
    if run_meta.get("initialized_from") is not None or run_meta.get("seed") != 0:
        raise ValueError(f"{arm_name} run initialization or seed changed")

    log_rows = [
        json.loads(line)
        for line in paths["log"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [row.get("step") for row in log_rows] != [0, 14_265]:
        raise ValueError(f"{arm_name} training log endpoint changed")
    final_log = log_rows[-1]
    train_bce = final_log.get("train_bce_per_key")
    validation_bce = final_log.get("val_bce_per_key")
    if not isinstance(train_bce, Mapping) or set(train_bce) != set(KEY_ORDER):
        raise ValueError(f"{arm_name} final train BCE receipt changed")
    if not isinstance(validation_bce, Mapping) or set(validation_bce) != set(KEY_ORDER):
        raise ValueError(f"{arm_name} final y4n validation BCE receipt changed")

    expected_marker = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "status": "complete",
        "study_id": STUDY_ID,
        "arm": arm_name,
        "run_id": run_id,
        "surface": Y4N_SURFACE,
        "weights": "final",
        "contract_sha256": contract_sha256,
        "checkpoint_sha256": expected_hashes["checkpoint_sha256"],
        "run_meta_sha256": expected_hashes["run_meta_sha256"],
        "source_sampling_receipt_sha256": expected_hashes[
            "source_sampling_receipt_sha256"
        ],
        "report_sha256": report_sha,
        "sidecar_sha256": sidecar_sha,
    }
    _require_nested_equal(marker, expected_marker, f"{arm_name} completion marker")
    if (
        wrapper.get("schema_version") != WRAPPER_MARKER_SCHEMA_VERSION
        or wrapper.get("status") != "complete"
        or wrapper.get("arm") != arm_name
        or wrapper.get("run_id") != run_id
        or wrapper.get("contract_sha256") != contract_sha256
        or wrapper.get("b1_accessed") is not False
        or wrapper.get("sealed_untouched_session_accessed") is not False
    ):
        raise ValueError(f"{arm_name} wrapper release gate changed")
    wrapper_artifacts = wrapper.get("artifacts")
    expected_wrapper_hashes = {
        "report": (report_path.name, report_sha),
        "sidecar": (sidecar_path.name, sidecar_sha),
        "marker": (marker_path.name, sha256_file(marker_path)),
    }
    if not isinstance(wrapper_artifacts, Mapping):
        raise ValueError(f"{arm_name} wrapper lacks artifact bindings")
    for name, (basename, digest) in expected_wrapper_hashes.items():
        item = wrapper_artifacts.get(name)
        if (
            not isinstance(item, Mapping)
            or Path(str(item.get("path", ""))).name != basename
            or item.get("sha256") != digest
        ):
            raise ValueError(f"{arm_name} wrapper changed {name} binding")

    source_exposure = {
        name: {
            field: receipt[field]
            for field in (
                "session_count",
                "segment_items",
                "scheduled_draws",
                "actual_draws",
                "unique_segment_items_drawn",
                "repeat_draws",
                "effective_pool_passes",
                "completed_pool_passes",
                "minimum_draws_per_item",
                "maximum_draws_per_item",
                "mean_draws_per_item",
            )
        }
        for name, receipt in validated_sampling["sources"].items()
    }
    release = {
        "complete": True,
        "weights": "final",
        "checkpoint_sha256": expected_hashes["checkpoint_sha256"],
        "report_sha256": report_sha,
        "sidecar_sha256": sidecar_sha,
        "completion_marker_sha256": sha256_file(marker_path),
        "wrapper_marker_sha256": sha256_file(wrapper_path),
        "report_path": _relative(repo, report_path),
        "sidecar_path": _relative(repo, sidecar_path),
        "completion_marker_path": _relative(repo, marker_path),
        "wrapper_marker_path": _relative(repo, wrapper_path),
        "run_receipt": dict(run_receipt),
        "metrics": score["metrics"],
        "baselines": score["baselines"],
        "source_sampling": validated_sampling,
        "memorization_receipts": {
            "sampling_exposure": source_exposure,
            "final_mixed_batch_train_bce_per_key": {
                key: float(train_bce[key]) for key in KEY_ORDER
            },
            "final_mixed_batch_train_bce_macro": _mean_mapping(
                train_bce, f"{arm_name} train BCE"
            ),
            "final_y4n_all_sixteen_validation_bce_per_key": {
                key: float(validation_bce[key]) for key in KEY_ORDER
            },
            "final_y4n_all_sixteen_validation_bce_macro": _mean_mapping(
                validation_bce, f"{arm_name} validation BCE"
            ),
            "source_membership_and_shard_hashes": _source_membership_receipt(
                contract
            ),
            "required_receipt_status": {
                "complete": False,
                "available": [
                    "scheduled and actual source draws",
                    "unique segment items and repeat distribution per source",
                    "local effective pool passes",
                    "source membership and shard hashes",
                    "final mixed-batch train BCE",
                ],
                "not_emitted_by_training_artifacts": [
                    "per-source final train BCE",
                    "corrected local-train versus val-A BCE/AP/F1 gap",
                ],
                "candidate_rule_dependency": False,
            },
        },
    }
    return release, sidecar


def _candidate_rule(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    reference_metrics = reference["metrics"]
    candidate_metrics = candidate["metrics"]
    macro_delta = float(
        candidate_metrics["macro_ap"] - reference_metrics["macro_ap"]
    )
    per_key_delta = {
        key: float(
            candidate_metrics["per_key_ap"][key]
            - reference_metrics["per_key_ap"][key]
        )
        for key in KEY_ORDER
    }
    improved_keys = [key for key in KEY_ORDER if per_key_delta[key] > 0.0]
    candidate_event = float(
        candidate_metrics["segment_bounded_combined_event_f1_fixed_0_5"]
        ["plus_minus_2"]
    )
    reference_event = float(
        reference_metrics["segment_bounded_combined_event_f1_fixed_0_5"]
        ["plus_minus_2"]
    )
    event_delta = candidate_event - reference_event
    event_loss = reference_event - candidate_event
    gates = {
        "macro_ap_delta_at_least_0_005": macro_delta
        >= MINIMUM_MACRO_AP_DELTA,
        "at_least_four_of_seven_per_key_ap_improvements": len(improved_keys)
        >= MINIMUM_PER_KEY_AP_WINS,
        "plus_minus_2_event_f1_loss_at_most_0_005": event_loss
        <= MAXIMUM_PLUS_MINUS_2_EVENT_F1_LOSS,
    }
    return {
        "macro_ap_delta": macro_delta,
        "per_key_ap_delta": per_key_delta,
        "improved_keys": improved_keys,
        "improved_key_count": len(improved_keys),
        "plus_minus_2_event_f1_delta": event_delta,
        "plus_minus_2_event_f1_loss": event_loss,
        "gates": gates,
        "eligible": all(gates.values()),
    }


def _metric_deltas(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    left_metrics, right_metrics = left["metrics"], right["metrics"]
    scalar = (
        "macro_ap",
        "macro_state_f1_fixed_0_5",
        "key_state_micro_accuracy_fixed_0_5",
        "joint_exact_match_accuracy_fixed_0_5",
        "predicted_positive_rate_fixed_0_5",
    )
    return {
        "left_minus_right": {
            **{
                field: float(left_metrics[field] - right_metrics[field])
                for field in scalar
            },
            "per_key_ap": {
                key: float(
                    left_metrics["per_key_ap"][key]
                    - right_metrics["per_key_ap"][key]
                )
                for key in KEY_ORDER
            },
            "segment_bounded_event_f1_fixed_0_5": {
                collar: float(
                    left_metrics[
                        "segment_bounded_combined_event_f1_fixed_0_5"
                    ][collar]
                    - right_metrics[
                        "segment_bounded_combined_event_f1_fixed_0_5"
                    ][collar]
                )
                for collar in ("exact", "plus_minus_2")
            },
        }
    }


def build_decision(
    *,
    repo: Path,
    contract_path: Path,
    contract_commit: str,
    pure_n_decision_path: Path,
) -> dict[str, Any]:
    """Validate all inputs and return one deterministic comparison record."""

    repo = repo.resolve()
    contract_path = contract_path.resolve()
    contract_sha256 = sha256_file(contract_path)
    contract = validate_contract(
        repo,
        contract_path,
        contract_sha256,
        contract_commit,
    )
    reference, reference_sidecar, pure_n_decision = _validate_pure_n_decision(
        repo, pure_n_decision_path.resolve(), contract
    )
    arm_releases: dict[str, dict[str, Any]] = {}
    sidecars: list[dict[str, Any]] = [reference_sidecar]
    for arm_name in ARM_SPECS:
        release, sidecar = _validate_blend_release(
            repo, contract, contract_sha256, contract_commit, arm_name
        )
        arm_releases[arm_name] = release
        sidecars.append(sidecar)
    reference_truth = np.asarray(reference_sidecar["truth"], dtype=bool)
    for arm_name, sidecar in zip(ARM_SPECS, sidecars[1:], strict=True):
        if not np.array_equal(reference_truth, sidecar["truth"]):
            raise ValueError(f"{arm_name} truth differs from pure-N later-eight")
        if list(sidecar["session_ids"]) != Y4N_STREAM_IDS:
            raise ValueError(f"{arm_name} support differs from pure-N later-eight")

    shuffled = segment_bounded_shuffled_event_baseline(
        reference_truth, Y4N_STREAM_LENGTHS
    )
    reference["baselines"]["shuffled_events"] = shuffled
    for release in arm_releases.values():
        release["baselines"]["shuffled_events"] = shuffled

    runs = {
        REFERENCE_RUN_ID: reference,
        **{
            ARM_SPECS[arm_name]["run_id"]: release
            for arm_name, release in arm_releases.items()
        },
    }
    rules = {
        arm_name: _candidate_rule(reference, arm_releases[arm_name])
        for arm_name in ARM_SPECS
    }
    eligible = [arm_name for arm_name in ARM_SPECS if rules[arm_name]["eligible"]]
    winner: str | None = None
    outcome: str
    if not eligible:
        outcome = "no_arm_eligible"
    elif len(eligible) == 1:
        winner = eligible[0]
        outcome = "one_arm_eligible"
    else:
        ordered = sorted(
            eligible,
            key=lambda arm_name: (
                -float(arm_releases[arm_name]["metrics"]["macro_ap"]),
                -float(
                    arm_releases[arm_name]["metrics"]
                    ["segment_bounded_combined_event_f1_fixed_0_5"]
                    ["plus_minus_2"]
                ),
            ),
        )
        first, second = ordered[:2]
        first_key = (
            arm_releases[first]["metrics"]["macro_ap"],
            arm_releases[first]["metrics"]
            ["segment_bounded_combined_event_f1_fixed_0_5"]["plus_minus_2"],
        )
        second_key = (
            arm_releases[second]["metrics"]["macro_ap"],
            arm_releases[second]["metrics"]
            ["segment_bounded_combined_event_f1_fixed_0_5"]["plus_minus_2"],
        )
        if first_key == second_key:
            outcome = "unresolved_exact_preregistered_tie"
        else:
            winner = first
            outcome = "tie_break_selected_one_arm"

    result: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "contract": {
            "path": _relative(repo, contract_path),
            "sha256": contract_sha256,
            "commit": contract_commit,
        },
        "pure_n_source_evidence": {
            "path": _relative(repo, pure_n_decision_path),
            "sha256": sha256_file(pure_n_decision_path),
            "canonical_sha256": pure_n_decision["decision"][
                "decision_record_sha256"
            ],
        },
        "evaluation_population": {
            "surface": Y4N_SURFACE,
            "session_ids": Y4N_STREAM_IDS,
            "stream_lengths": Y4N_STREAM_LENGTHS,
            "active_frames": Y4N_FRAMES,
            "truth_sha256": Y4N_TRUTH_SHA256,
            "fixed_threshold": 0.5,
            "fitted_thresholds_used": False,
            "fitted_calibration_used": False,
            "b1_used": False,
            "weights": "final",
            "label_kind": "mapped_foreign_nitrogen",
        },
        "shared_baselines": {
            **copy.deepcopy(reference["baselines"]),
            "identical_support_for_every_run": True,
        },
        "runs": runs,
        "decision": {
            "comparison_frozen_before_b1": True,
            "all_runs_consulted": [
                REFERENCE_RUN_ID,
                ARM_SPECS["NL_90_10"]["run_id"],
                ARM_SPECS["NLW_70_20_10"]["run_id"],
            ],
            "candidate_rule": {
                "surface": "mapped_y4n_later_eight_only",
                "minimum_macro_ap_delta": MINIMUM_MACRO_AP_DELTA,
                "minimum_per_key_ap_improvements_out_of_7": (
                    MINIMUM_PER_KEY_AP_WINS
                ),
                "maximum_plus_minus_2_event_f1_loss": (
                    MAXIMUM_PLUS_MINUS_2_EVENT_F1_LOSS
                ),
                "tie_break": ["higher_macro_ap", "higher_plus_minus_2_event_f1"],
                "source_text": contract["fresh_capture_candidate_rule"],
            },
            "arm_results": rules,
            "eligible_arms": eligible,
            "winner_arm": winner,
            "winner_run_id": ARM_SPECS[winner]["run_id"] if winner else None,
            "outcome": outcome,
            "scientific_contrasts": {
                "local_marginal_NL_90_10_minus_pure_N": _metric_deltas(
                    arm_releases["NL_90_10"], reference
                ),
                "provisional_wild_marginal_NLW_minus_NL": _metric_deltas(
                    arm_releases["NLW_70_20_10"],
                    arm_releases["NL_90_10"],
                ),
            },
            "interpretation_limits": {
                "single_seed": True,
                "not_a_full_factorial": True,
                "not_an_additive_hours_experiment": True,
                "wild_tier": "provisional_not_train_ready",
                "wild_admitted_hours": 0.0,
                "mapped_labels_are_noisy": True,
                "memorization_receipts_complete": all(
                    release["memorization_receipts"]["required_receipt_status"]
                    ["complete"]
                    for release in arm_releases.values()
                ),
            },
            "decision_record_sha256_scope": (
                "canonical JSON with decision_record_sha256 set to null"
            ),
            "decision_record_sha256": None,
        },
    }
    result["decision"]["decision_record_sha256"] = _canonical_sha256(result)
    for arm_name, release in arm_releases.items():
        validate_comparison_value(
            result,
            arm_name=arm_name,
            arm_release=release,
        )
    return result


def write_decision(path: Path, result: Mapping[str, Any]) -> None:
    """Atomically write deterministic JSON and refuse every overwrite."""

    if os.path.lexists(path):
        raise ValueError(f"refusing to overwrite blend decision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if os.path.lexists(temporary):
        raise ValueError(f"refusing to overwrite blend decision temp file: {temporary}")
    try:
        temporary.write_text(
            json.dumps(result, indent=2, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--contract", type=Path, default=CONTRACT_RELATIVE_PATH)
    parser.add_argument("--contract-commit", default=CONTRACT_COMMIT)
    parser.add_argument(
        "--pure-n-decision", type=Path, default=PURE_N_DECISION_RELATIVE_PATH
    )
    parser.add_argument("--out", type=Path, default=OUTPUT_RELATIVE_PATH)
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo = args.repo.resolve()
    contract = args.contract if args.contract.is_absolute() else repo / args.contract
    pure_n = (
        args.pure_n_decision
        if args.pure_n_decision.is_absolute()
        else repo / args.pure_n_decision
    )
    output = args.out if args.out.is_absolute() else repo / args.out
    result = build_decision(
        repo=repo,
        contract_path=contract,
        contract_commit=args.contract_commit,
        pure_n_decision_path=pure_n,
    )
    write_decision(output, result)
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": sha256_file(output),
                "decision_record_sha256": result["decision"]
                ["decision_record_sha256"],
                "outcome": result["decision"]["outcome"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
