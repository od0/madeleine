"""Fixed-policy B1 transfer for the preregistered state-only TCN study.

This entry point deliberately does not call :mod:`badeline.eval`: that
evaluator computes same-surface oracle transition thresholds as diagnostics.
Here B1 is available only after the complete y4n decision has been committed,
and it is scored with final weights and the natural 0.5 threshold only.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np
import torch

from badeline.eval_event import _decision_metrics
from badeline.metrics import per_key_ap, per_key_f1, per_key_transition_f1
from badeline.model import BadelineIDM
from badeline.train import (
    contiguous_runs,
    load_session,
    read_session_ids,
    target_offset,
)
from data.schema import KEY_ORDER
from experiments.eval_event_b1 import B1_SESSION_ID, validate_b1_manifest


DECISION_RELATIVE_PATH = Path(
    "results/idm/tcn_control_lr_y4n_decision.json"
)
STUDY_ID = "tcn_control_lr_y4n_later8_s0"
EXPECTED_FINAL_STEP = 14_265
EXPECTED_B1_FRAMES = 12_925
EXPECTED_B1_ACTIVE_FRAMES = 9_202
EXPECTED_B1_STREAMS = 73
EXPECTED_Y4N_ACTIVE_FRAMES = 269_352
EXPECTED_Y4N_SESSIONS = [
    f"y4nQHqYSObI__r{run:03d}__stream000" for run in range(8, 16)
]
EXPECTED_B1_RUN_INDICES = [
    13, 15, 19, 23, 26, 31, 33, 37, 38, 41, 43, 49, 52, 54, 56, 57,
    62, 71, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 87, 88, 89,
    94, 95, 96, 97, 98, 102, 103, 105, 106, 107, 112, 114, 117, 118,
    121, 126, 132, 134, 143, 146, 149, 153, 154, 164, 166, 168, 170,
    171, 172, 174, 177, 179, 181, 198, 200, 204, 210, 211, 212, 213,
    214,
]
EXPECTED_B1_STREAM_IDS = [
    f"{B1_SESSION_ID}__stream{index:03d}"
    for index in EXPECTED_B1_RUN_INDICES
]
EXPECTED_B1_STREAM_LENGTHS = [
    216, 102, 39, 138, 102, 217, 101, 183, 218, 132, 92, 216, 186,
    217, 217, 219, 152, 108, 218, 219, 218, 219, 218, 219, 218, 218,
    219, 190, 122, 218, 219, 218, 217, 219, 219, 219, 218, 219, 218,
    217, 218, 218, 92, 217, 149, 113, 216, 92, 217, 90, 79, 92, 28,
    91, 219, 218, 217, 216, 217, 219, 103, 56, 217, 217, 65, 210, 45,
    217, 217, 219, 219, 219, 169,
]
EXPECTED_B1_TRUTH_SHA256 = (
    "e51e1e9dc945fd41ab61fb3b8fe2b91cc7644ef15e5780eae41748dece26436c"
)
EXPECTED_B1_ACTIVE_SHA256 = (
    "bd9109c03fa0e572d2c1aa2ac7ddbe8919948fcbe9247cf4f7df9b090f2df6ab"
)
INFERENCE_SOURCE_PATHS = (
    "badeline/model.py",
    "badeline/temporal.py",
    "badeline/train.py",
    "data/schema.py",
)
REGISTERED_RUNS: dict[str, dict[str, float | bool]] = {
    "nitrogen_unflagged_92train_y4n_holdout_26m_vptlite_tcn_natural_s0": {
        "learning_rate": 0.0003,
        "class_balance": False,
        "transition_weight": 1.0,
    },
    "nitrogen_unflagged_92train_y4n_holdout_26m_vptlite_tcn_lr1e4_s0": {
        "learning_rate": 0.0001,
        "class_balance": True,
        "transition_weight": 8.0,
    },
    "nitrogen_unflagged_92train_y4n_holdout_26m_vptlite_tcn_lr1e3_s0": {
        "learning_rate": 0.001,
        "class_balance": True,
        "transition_weight": 8.0,
    },
}
REQUIRED_DECISION_FIELDS = {
    "all_consulted_runs",
    "multiplicity_disclosure",
    "objective_contrast",
    "event_head_gradient_contrast",
    "lr_sensitivity",
    "event_regression_guards",
    "single_seed_limit",
    "y4n_decision_frozen_before_b1",
    "decision_record_sha256",
    "decision_git_commit",
}
REQUIRED_RUN_RECEIPT_FIELDS = {
    "run_id",
    "config_sha256",
    "launcher_sha256",
    "checkpoint_sha256",
    "prediction_sidecar_sha256",
    "implementation_git_commit",
    "training_start_utc",
    "training_end_utc",
    "final_step",
    "expected_final_step",
    "support_session_ids",
    "active_frames",
    "finite_aligned_arrays",
    "valid",
}
HEX_64 = re.compile(r"[0-9a-f]{64}")
HEX_40 = re.compile(r"[0-9a-f]{40}")


def sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 digest for ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
    )


def _verify_inference_source(
    repo: Path, implementation_commit: object
) -> dict[str, Any]:
    """Bind checkpoint execution to the source tree that trained the run."""

    if not isinstance(implementation_commit, str) or not HEX_40.fullmatch(
        implementation_commit
    ):
        raise ValueError("implementation Git commit is malformed")
    resolved = _git(
        repo, "rev-parse", "--verify", f"{implementation_commit}^{{commit}}"
    )
    if (
        resolved.returncode != 0
        or resolved.stdout.strip().decode() != implementation_commit
    ):
        raise ValueError("implementation Git commit is not resolvable")
    ancestor = _git(
        repo, "merge-base", "--is-ancestor", implementation_commit, "HEAD"
    )
    if ancestor.returncode != 0:
        raise ValueError("implementation Git commit is not an ancestor of checkout")

    files: dict[str, str] = {}
    for relative in INFERENCE_SOURCE_PATHS:
        working = repo / relative
        if not working.is_file():
            raise ValueError(f"inference source is missing: {relative}")
        committed = _git(
            repo, "cat-file", "blob", f"{implementation_commit}:{relative}"
        )
        if committed.returncode != 0:
            raise ValueError(f"inference source is absent from commit: {relative}")
        working_bytes = working.read_bytes()
        if committed.stdout != working_bytes:
            raise ValueError(
                f"inference source differs from training commit: {relative}"
            )
        files[relative] = hashlib.sha256(working_bytes).hexdigest()
    return {
        "implementation_git_commit": implementation_commit,
        "verified_files_sha256": files,
    }


def _consulted_run_ids(value: object) -> set[str]:
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            if isinstance(item, str):
                result.add(item)
            elif isinstance(item, dict) and isinstance(item.get("run_id"), str):
                result.add(item["run_id"])
            else:
                raise ValueError(
                    "all_consulted_runs list entries must declare run_id"
                )
        return result
    if isinstance(value, dict):
        result: set[str] = set()
        for key, item in value.items():
            if isinstance(item, dict) and isinstance(item.get("run_id"), str):
                result.add(item["run_id"])
            elif isinstance(key, str):
                result.add(key)
        return result
    raise ValueError("all_consulted_runs must be a list or object")


def validate_decision_release(
    repo: Path,
    receipt_path: Path,
    expected_sha256: str,
    decision_commit: str,
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the immutable y4n decision before any B1 path is accessed."""

    if run_id not in REGISTERED_RUNS:
        raise ValueError(f"unregistered TCN study run: {run_id}")
    if not HEX_64.fullmatch(expected_sha256):
        raise ValueError("decision SHA-256 must be 64 lowercase hex characters")
    if not HEX_40.fullmatch(decision_commit):
        raise ValueError("decision commit must be a full 40-character SHA")

    repo = repo.resolve()
    expected_path = (repo / DECISION_RELATIVE_PATH).resolve()
    if receipt_path.resolve() != expected_path:
        raise ValueError(
            f"decision receipt must be {DECISION_RELATIVE_PATH.as_posix()}"
        )
    if not receipt_path.is_file():
        raise ValueError("committed y4n decision receipt is missing")
    receipt_bytes = receipt_path.read_bytes()
    observed_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("y4n decision receipt SHA-256 mismatch")

    resolved = _git(repo, "rev-parse", "--verify", f"{decision_commit}^{{commit}}")
    if resolved.returncode != 0 or resolved.stdout.strip().decode() != decision_commit:
        raise ValueError("decision commit is not a resolvable full commit")
    ancestor = _git(repo, "merge-base", "--is-ancestor", decision_commit, "HEAD")
    if ancestor.returncode != 0:
        raise ValueError("decision commit is not an ancestor of the checkout")
    committed = _git(
        repo,
        "cat-file",
        "blob",
        f"{decision_commit}:{DECISION_RELATIVE_PATH.as_posix()}",
    )
    if committed.returncode != 0:
        raise ValueError("decision receipt is absent from the declared commit")
    if committed.stdout != receipt_bytes:
        raise ValueError("working y4n decision differs from its committed bytes")
    dirty = _git(
        repo,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        DECISION_RELATIVE_PATH.as_posix(),
    )
    if dirty.returncode != 0 or dirty.stdout:
        raise ValueError("working y4n decision receipt is not clean")

    try:
        receipt = json.loads(receipt_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("y4n decision receipt is not valid JSON") from error
    if not isinstance(receipt, dict):
        raise ValueError("y4n decision receipt must be an object")
    if receipt.get("study_id") != STUDY_ID:
        raise ValueError("y4n decision receipt has the wrong study ID")
    population = receipt.get("evaluation_population")
    if not isinstance(population, dict):
        raise ValueError("y4n decision lacks its evaluation population")
    if population.get("session_ids") != EXPECTED_Y4N_SESSIONS:
        raise ValueError("decision population is not y4n later-eight")
    if population.get("active_frames") != EXPECTED_Y4N_ACTIVE_FRAMES:
        raise ValueError("decision population support changed")
    if population.get("oracle_metrics_used") is not False:
        raise ValueError("decision population used oracle metrics")
    if population.get("calibration_used") is not False:
        raise ValueError("decision population used fitted calibration")
    if population.get("b1_used") is not False:
        raise ValueError("decision population was not isolated from B1")
    if population.get("threshold_policy") != (
        "probability >= 0.5 for fixed state and state-transition metrics"
    ):
        raise ValueError("decision threshold policy is not fixed 0.5")

    decision = receipt.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("y4n decision receipt lacks its decision object")
    missing_fields = REQUIRED_DECISION_FIELDS.difference(decision)
    if missing_fields:
        raise ValueError(
            f"y4n decision receipt lacks fields: {sorted(missing_fields)}"
        )
    if decision["y4n_decision_frozen_before_b1"] is not True:
        raise ValueError("y4n decision is not frozen before B1")
    internal_hash = decision["decision_record_sha256"]
    if not isinstance(internal_hash, str) or not HEX_64.fullmatch(internal_hash):
        raise ValueError("decision canonical hash is malformed")
    canonical = copy.deepcopy(receipt)
    canonical["decision"]["decision_record_sha256"] = None
    canonical_bytes = json.dumps(
        canonical,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if hashlib.sha256(canonical_bytes).hexdigest() != internal_hash:
        raise ValueError("decision canonical hash does not validate")
    authored_commit = decision["decision_git_commit"]
    if not isinstance(authored_commit, str) or not HEX_40.fullmatch(
        authored_commit
    ):
        raise ValueError("decision authoring commit is malformed")
    authored_ancestor = _git(
        repo, "merge-base", "--is-ancestor", authored_commit, decision_commit
    )
    if authored_ancestor.returncode != 0:
        raise ValueError("decision authoring commit is not in its commit history")

    consulted = _consulted_run_ids(decision["all_consulted_runs"])
    if not set(REGISTERED_RUNS).issubset(consulted):
        raise ValueError("decision omits one or more registered new runs")
    runs = receipt.get("runs")
    if not isinstance(runs, dict) or run_id not in runs:
        raise ValueError("decision lacks the requested run receipt")
    run_receipt = runs[run_id]
    if not isinstance(run_receipt, dict):
        raise ValueError("requested run receipt must be an object")
    missing_run_fields = REQUIRED_RUN_RECEIPT_FIELDS.difference(run_receipt)
    if missing_run_fields:
        raise ValueError(
            f"requested run receipt lacks fields: {sorted(missing_run_fields)}"
        )
    if run_receipt["run_id"] != run_id:
        raise ValueError("requested run receipt identity mismatch")
    if run_receipt["valid"] is not True:
        raise ValueError("requested run is not valid in the frozen y4n decision")
    if run_receipt["final_step"] != EXPECTED_FINAL_STEP or run_receipt[
        "expected_final_step"
    ] != EXPECTED_FINAL_STEP:
        raise ValueError("requested run did not reach the fixed final endpoint")
    if run_receipt["support_session_ids"] != EXPECTED_Y4N_SESSIONS:
        raise ValueError("requested run y4n support membership changed")
    if run_receipt["active_frames"] != EXPECTED_Y4N_ACTIVE_FRAMES:
        raise ValueError("requested run y4n support count changed")
    if run_receipt["finite_aligned_arrays"] is not True:
        raise ValueError("requested run y4n arrays were not validated")
    if run_receipt.get("evaluation_weights") != "final":
        raise ValueError("requested run decision did not use final weights")
    if not HEX_64.fullmatch(str(run_receipt["checkpoint_sha256"])):
        raise ValueError("requested run checkpoint receipt is not SHA-256")
    return receipt, run_receipt


def validate_run(
    repo: Path,
    run_dir: Path,
    run_id: str,
    run_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor], dict[str, Any]]:
    """Validate the exact registered recipe and return final weights only."""

    if run_id not in REGISTERED_RUNS or run_dir.name != run_id:
        raise ValueError("run directory is not an allowed registered run")
    source_receipt = _verify_inference_source(
        repo.resolve(), run_receipt.get("implementation_git_commit")
    )
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "model.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise ValueError("completed run config or checkpoint is missing")
    config = json.loads(config_path.read_text())
    expected: dict[str, object] = {
        "temporal_arch": "aligned_tcn",
        "precomputed_features": True,
        "window": 128,
        "frame_stride": 3,
        "window_mode": "centered",
        "input_config": "pixels",
        "active_targets_only": True,
        "seed": 0,
        "max_steps": EXPECTED_FINAL_STEP,
        **REGISTERED_RUNS[run_id],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"registered run config mismatch for {key}")
    for forbidden in (
        "event_latch",
        "event_class_balance_max",
        "state_loss_weight",
        "onset_loss_weight",
        "release_loss_weight",
    ):
        if forbidden in config:
            raise ValueError(f"state-only B1 evaluator forbids {forbidden}")
    if sha256_file(config_path) != run_receipt["config_sha256"]:
        raise ValueError("config differs from the frozen y4n receipt")
    if sha256_file(checkpoint_path) != run_receipt["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from the frozen y4n receipt")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if checkpoint.get("steps") != EXPECTED_FINAL_STEP:
        raise ValueError("checkpoint is not the fixed final endpoint")
    final_state = checkpoint.get("final_state_dict")
    if not isinstance(final_state, Mapping) or not final_state:
        raise ValueError("checkpoint lacks final_state_dict")
    return config, final_state, source_receipt


def validate_b1_surface(
    data_dir: Path, sessions_path: Path, marker_path: Path
) -> list[str]:
    """Validate the one frozen B1 feature surface after embargo release."""

    if not marker_path.is_file():
        raise ValueError("B1 frozen-feature validation marker is missing")
    session_ids = read_session_ids(sessions_path)
    validate_b1_manifest(data_dir, session_ids)
    shard = data_dir / f"{B1_SESSION_ID}.npz"
    if not shard.is_file():
        raise ValueError("frozen B1 feature shard is missing")
    return session_ids


def infer_fixed_state(
    model: BadelineIDM,
    config: Mapping[str, Any],
    data_dir: Path,
    session_ids: Sequence[str],
    device: str,
    sidecar_path: Path,
    *,
    segment_span: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Run state-only inference without invoking any fitted metric path."""

    if segment_span < 1:
        raise ValueError("segment_span must be positive")
    model.eval().to(device)
    window = int(config["window"])
    frame_stride = int(config["frame_stride"])
    frame_span = (window - 1) * frame_stride + 1
    offset = target_offset(window, str(config["window_mode"]))
    all_true: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []
    all_active: list[np.ndarray] = []
    stream_lengths: list[int] = []
    stream_ids: list[str] = []

    for session_id in session_ids:
        arrays = load_session(
            data_dir, session_id, precomputed_features=True
        )
        assert arrays.engine_frame_idx is not None
        assert arrays.input_active is not None
        for run_index, (run_start, run_end) in enumerate(
            contiguous_runs(arrays.engine_frame_idx)
        ):
            n_windows = run_end - run_start - frame_span + 1
            if n_windows < 1:
                continue
            chunks: list[np.ndarray] = []
            with torch.no_grad():
                for relative_start in range(0, n_windows, segment_span):
                    count = min(segment_span, n_windows - relative_start)
                    start = run_start + relative_start
                    block = arrays.frames[
                        start : start + count + frame_span - 1
                    ]
                    features = (
                        torch.from_numpy(block.copy())
                        .to(dtype=torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    )
                    logits = model.forward_segment({"features": features})
                    chunks.append(
                        torch.sigmoid(logits)[0]
                        .to(torch.float32)
                        .cpu()
                        .numpy()
                    )
            probability = np.concatenate(chunks)
            target_start = run_start + offset * frame_stride
            truth = arrays.keys[
                target_start : target_start + len(probability)
            ].astype(bool)
            active = arrays.input_active[
                target_start : target_start + len(probability)
            ].astype(bool)
            all_true.append(truth)
            all_prob.append(probability)
            all_active.append(active)
            stream_lengths.append(len(probability))
            stream_ids.append(f"{session_id}__stream{run_index:03d}")

    if not all_true:
        raise ValueError("B1 contains no eligible contiguous evaluation window")
    truth = np.concatenate(all_true)
    probability = np.concatenate(all_prob)
    active = np.concatenate(all_active)
    lengths = np.asarray(stream_lengths, dtype=np.int64)
    np.savez_compressed(
        sidecar_path,
        y_true=truth.astype(np.uint8),
        y_prob=probability.astype(np.float32),
        input_active=active.astype(np.uint8),
        session_lengths=lengths,
        session_ids=np.asarray(stream_ids),
    )
    return truth, probability, active, lengths, stream_ids


def validate_b1_sidecar(
    path: Path,
    *,
    expected_truth_sha256: str = EXPECTED_B1_TRUTH_SHA256,
    expected_active_sha256: str = EXPECTED_B1_ACTIVE_SHA256,
) -> dict[str, Any]:
    """Require exact B1 support, identities, binary truth and finite scores."""

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "y_true",
            "y_prob",
            "input_active",
            "session_lengths",
            "session_ids",
        }
        if set(archive.files) != required:
            raise ValueError("B1 sidecar fields changed")
        truth = np.asarray(archive["y_true"])
        probability = np.asarray(archive["y_prob"])
        active = np.asarray(archive["input_active"])
        lengths = np.asarray(archive["session_lengths"])
        session_ids = np.asarray(archive["session_ids"])
    expected_shape = (EXPECTED_B1_FRAMES, len(KEY_ORDER))
    if truth.shape != expected_shape or probability.shape != expected_shape:
        raise ValueError("B1 state sidecar support changed")
    if truth.dtype != np.uint8 or active.dtype != np.uint8:
        raise ValueError("B1 truth and active receipts must use uint8")
    if lengths.dtype != np.int64:
        raise ValueError("B1 stream-length receipt must use int64")
    if not np.all(np.isin(truth, (0, 1))):
        raise ValueError("B1 truth is not binary")
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("B1 probabilities are not finite values in [0,1]")
    if active.shape != (EXPECTED_B1_FRAMES,) or not np.all(
        np.isin(active, (0, 1))
    ):
        raise ValueError("B1 input-active mask changed")
    if int(active.sum()) != EXPECTED_B1_ACTIVE_FRAMES:
        raise ValueError("B1 active support changed")
    if lengths.shape != (EXPECTED_B1_STREAMS,) or np.any(lengths < 1):
        raise ValueError("B1 stream-length support changed")
    if lengths.tolist() != EXPECTED_B1_STREAM_LENGTHS:
        raise ValueError("B1 stream lengths or boundaries changed")
    if session_ids.tolist() != EXPECTED_B1_STREAM_IDS:
        raise ValueError("B1 stream identities changed")
    truth_sha256 = hashlib.sha256(
        np.ascontiguousarray(truth).tobytes()
    ).hexdigest()
    active_sha256 = hashlib.sha256(
        np.ascontiguousarray(active).tobytes()
    ).hexdigest()
    if truth_sha256 != expected_truth_sha256:
        raise ValueError("B1 engine-truth receipt changed")
    if active_sha256 != expected_active_sha256:
        raise ValueError("B1 active-mask receipt changed")
    return {
        "all_frames": EXPECTED_B1_FRAMES,
        "input_active_frames": EXPECTED_B1_ACTIVE_FRAMES,
        "streams": EXPECTED_B1_STREAMS,
        "session_ids": session_ids.tolist(),
        "stream_lengths": lengths.tolist(),
        "truth_sha256": truth_sha256,
        "input_active_sha256": active_sha256,
        "finite_aligned_arrays": True,
    }


def _macro(values: Mapping[str, float]) -> float:
    return float(np.nanmean(np.asarray(list(values.values()), dtype=float)))


def fixed_metric_report(
    truth: np.ndarray,
    probability: np.ndarray,
    active: np.ndarray,
    lengths: Sequence[int],
) -> dict[str, Any]:
    """Return AP and fixed-0.5 metrics without fitting B1 parameters."""

    truth = np.asarray(truth).astype(bool, copy=False)
    probability = np.asarray(probability)
    active = np.asarray(active).astype(bool, copy=False)
    lengths = [int(value) for value in lengths]

    def surface(gate: np.ndarray) -> dict[str, Any]:
        frame_truth = truth[gate]
        frame_probability = probability[gate]
        ap = per_key_ap(frame_truth, frame_probability)
        state_f1 = per_key_f1(frame_truth, frame_probability, threshold=0.5)
        decisions = _decision_metrics(
            truth,
            probability >= 0.5,
            active=gate,
            lengths=lengths,
        )
        exact = per_key_transition_f1(
            truth,
            probability,
            threshold=0.5,
            collar=0,
            boundaries=lengths,
            active=gate,
        )
        plus_minus_2 = per_key_transition_f1(
            truth,
            probability,
            threshold=0.5,
            collar=2,
            boundaries=lengths,
            active=gate,
        )
        exact_f1 = {key: float(exact[key]["event"]["f1"]) for key in KEY_ORDER}
        plus_minus_2_f1 = {
            key: float(plus_minus_2[key]["event"]["f1"])
            for key in KEY_ORDER
        }
        return {
            "n": int(gate.sum()),
            "per_key_ap": ap,
            "macro_ap": _macro(ap),
            "per_key_state_f1_fixed_0_5": state_f1,
            "macro_state_f1_fixed_0_5": _macro(state_f1),
            "decision_metrics_fixed_0_5": decisions,
            "per_key_combined_event_f1_fixed_0_5": {
                "exact": exact_f1,
                "plus_minus_2": plus_minus_2_f1,
            },
            "macro_combined_event_f1_fixed_0_5": {
                "exact": _macro(exact_f1),
                "plus_minus_2": _macro(plus_minus_2_f1),
            },
        }

    return {
        "all_frames": surface(np.ones(len(truth), dtype=bool)),
        "input_active_only": surface(active),
        "threshold_policy": {
            "state_probability": 0.5,
            "transition_probability": 0.5,
            "data_fitted_thresholds_used": False,
            "calibration_parameters_fitted": False,
        },
    }


def _refuse_existing(paths: Sequence[Path]) -> None:
    for path in paths:
        if os.path.lexists(path):
            raise ValueError(f"refusing to overwrite B1 artifact: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--decision-receipt", type=Path, required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--decision-commit", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--b1-marker", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--completion-marker", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Embargo gate: do not stat, read or load any B1 path above this line.
    _, run_receipt = validate_decision_release(
        args.repo,
        args.decision_receipt,
        args.decision_sha256,
        args.decision_commit,
        args.run_id,
    )
    config, final_state, source_receipt = validate_run(
        args.repo, args.run, args.run_id, run_receipt
    )

    artifacts = [
        args.out,
        args.sidecar,
        args.validation_receipt,
        args.completion_marker,
    ]
    _refuse_existing(artifacts)
    session_ids = validate_b1_surface(
        args.data, args.sessions, args.b1_marker
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.out.with_name(f".{args.out.name}.tmp")
    temporary_sidecar = args.sidecar.with_name(
        f".{args.sidecar.stem}.tmp.npz"
    )
    temporary_validation = args.validation_receipt.with_name(
        f".{args.validation_receipt.name}.tmp"
    )
    _refuse_existing(
        [temporary_report, temporary_sidecar, temporary_validation]
    )
    try:
        model = BadelineIDM(config)
        model.load_state_dict(final_state)
        truth, probability, active, lengths, stream_ids = infer_fixed_state(
            model,
            config,
            args.data,
            session_ids,
            args.device,
            temporary_sidecar,
        )
        support = validate_b1_sidecar(temporary_sidecar)
        if stream_ids != EXPECTED_B1_STREAM_IDS:
            raise ValueError("inference stream identities changed")
        report = fixed_metric_report(
            truth, probability, active, lengths.tolist()
        )
        report.update(
            {
                "run_id": args.run_id,
                "run": str(args.run),
                "weights": "final",
                "label_kind": "engine_truth_development_b1",
                "label_notice": (
                    "Post-y4n-decision development transfer; B1 cannot alter "
                    "the frozen y4n decision or promote a single-seed model."
                ),
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "sessions": session_ids,
                "support": support,
                "b1_policy": {
                    "post_y4n_decision_only": True,
                    "used_for_training": False,
                    "used_for_checkpoint_selection": False,
                    "used_for_threshold_fitting": False,
                    "used_for_calibration_fitting": False,
                    "fixed_threshold": 0.5,
                },
                "y4n_decision_receipt": {
                    "path": str(args.decision_receipt),
                    "sha256": args.decision_sha256,
                    "commit": args.decision_commit,
                },
                "checkpoint_sha256": run_receipt["checkpoint_sha256"],
                "inference_source_receipt": source_receipt,
                "predictions": str(args.sidecar),
            }
        )
        temporary_report.write_text(
            json.dumps(
                report, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n"
        )
        validation = {
            "status": "passed",
            "run_id": args.run_id,
            "weights": "final",
            "label_kind": "engine_truth_development_b1",
            "role": "post_y4n_decision_development_transfer",
            "threshold": 0.5,
            "threshold_or_calibration_fitting_used": False,
            "support": support,
            "decision_receipt_sha256": args.decision_sha256,
            "decision_commit": args.decision_commit,
            "checkpoint_sha256": run_receipt["checkpoint_sha256"],
            "inference_source_receipt": source_receipt,
            "report": str(args.out),
            "report_sha256": sha256_file(temporary_report),
            "sidecar": str(args.sidecar),
            "sidecar_sha256": sha256_file(temporary_sidecar),
        }
        temporary_validation.write_text(
            json.dumps(
                validation, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n"
        )
        temporary_sidecar.replace(args.sidecar)
        temporary_report.replace(args.out)
        temporary_validation.replace(args.validation_receipt)
    finally:
        for path in (
            temporary_report,
            temporary_sidecar,
            temporary_validation,
        ):
            path.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "weights": "final",
                "input_active_frames": EXPECTED_B1_ACTIVE_FRAMES,
                "role": "post_y4n_decision_development_transfer",
                "threshold": 0.5,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
