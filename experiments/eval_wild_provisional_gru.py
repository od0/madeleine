"""Fixed-only evaluation for the preregistered provisional-wild GRU.

The experiment is intentionally narrow.  It evaluates the fixed final
checkpoint first on the exact mapped-y4n later-eight development surface and,
only after that result has been committed, on the frozen B1 engine-truth
development surface.  Decisions use raw sigmoid probabilities at 0.5.  This
module does not fit thresholds, calibration parameters, or any other
evaluation-time parameter.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from badeline.model import BadelineIDM
from badeline.train import read_session_ids
from data.schema import KEY_ORDER
from experiments.eval_tcn_control_lr_b1 import (
    EXPECTED_B1_ACTIVE_FRAMES,
    EXPECTED_B1_FRAMES,
    EXPECTED_B1_STREAM_IDS,
    EXPECTED_B1_STREAM_LENGTHS,
    EXPECTED_B1_STREAMS,
    EXPECTED_B1_ACTIVE_SHA256,
    EXPECTED_B1_TRUTH_SHA256,
    fixed_metric_report,
    infer_fixed_state,
    sha256_file,
    validate_b1_sidecar,
    validate_b1_surface,
)


SCHEMA_VERSION = "madeleine.wild-provisional-gru-fixed-eval.v1"
MARKER_SCHEMA_VERSION = "madeleine.wild-provisional-gru-fixed-marker.v1"
CONTRACT_SCHEMA_VERSION = "madeleine.wild-provisional-gru-decision.v1"
STUDY_ID = "wild_provisional_broad7_gru_y4n_b1_s0"
RUN_ID = "wild_provisional_broad7_7train_y4n_holdout_26m_128x3_s0"
CONTRACT_RELATIVE_PATH = Path(
    "experiments/configs/wild_provisional_broad7_gru_decision.json"
)
CONTRACT_SHA256 = (
    "488a19cf906e51ef10954dc289ba747d652b7f56b5a0f373629acc78579c0ca6"
)
TEMPLATE_SHA256 = (
    "9c92ee27ac37115389980490f656af1af5bf0f3389952e652b323f6b279bfb95"
)
EXPECTED_FINAL_STEP = 2_598
EXPECTED_TRAINABLE_PARAMETERS = 25_719_815

Y4N_SURFACE = "mapped_y4n_later_eight"
B1_SURFACE = "engine_truth_b1_development"
Y4N_BASE_SESSION_IDS = [
    f"y4nQHqYSObI__r{index:03d}" for index in range(8, 16)
]
Y4N_STREAM_IDS = [f"{session_id}__stream000" for session_id in Y4N_BASE_SESSION_IDS]
Y4N_STREAM_LENGTHS = [35_619] * 7 + [20_019]
Y4N_FRAMES = 269_352
Y4N_TRUTH_SHA256 = (
    "f61a0de4076f4683f01494837f01c3e314873ab0d78ee131b43e8e9f6e576a01"
)
EXPECTED_NOTE = (
    "25.7M matched GRU trained for one exact pass over the immutable "
    "broad-seven provisional wild-overlay corpus; diagnostic noisy "
    "supervision only; final weights at 2598 steps"
)
REQUIRED_SIDECAR_FIELDS = {
    "y_true",
    "y_prob",
    "input_active",
    "session_lengths",
    "session_ids",
}


def _json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _canonical_array_sha256(array: np.ndarray) -> str:
    """Hash an array with its shape so support changes cannot collide."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _contains_disallowed_metric_language(value: object) -> bool:
    """Return whether a serialized report tries to expose a fitted metric."""

    if isinstance(value, Mapping):
        return any(
            "oracle" in str(key).lower()
            or _contains_disallowed_metric_language(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_disallowed_metric_language(item) for item in value)
    return isinstance(value, str) and "oracle" in value.lower()


def validate_contract(
    repo: Path, contract_path: Path, expected_sha256: str
) -> dict[str, Any]:
    """Bind evaluation to the exact preregistered experiment contract."""

    if expected_sha256 != CONTRACT_SHA256:
        raise ValueError("contract SHA-256 is not the preregistered digest")
    expected_path = (repo.resolve() / CONTRACT_RELATIVE_PATH).resolve()
    if contract_path.resolve() != expected_path:
        raise ValueError(
            f"contract must be {CONTRACT_RELATIVE_PATH.as_posix()}"
        )
    if not contract_path.is_file() or sha256_file(contract_path) != CONTRACT_SHA256:
        raise ValueError("preregistered contract bytes changed")
    contract = _json_object(contract_path, "preregistered contract")
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("preregistered contract schema changed")
    if contract.get("study_id") != STUDY_ID:
        raise ValueError("preregistered study identity changed")

    model_contract = contract.get("model_contract")
    evaluation = contract.get("evaluation_contract")
    if not isinstance(model_contract, dict) or not isinstance(evaluation, dict):
        raise ValueError("preregistered model or evaluation contract is missing")
    exact_model_values: dict[str, object] = {
        "run_id": RUN_ID,
        "template": (
            "experiments/configs/"
            "takeover_features_26m_128x3frame_full_holdout.json"
        ),
        "seed": 0,
        "window": 128,
        "frame_stride": 3,
        "frame_span": 382,
        "window_mode": "centered",
        "segment_windows": 96,
        "loader_batch_items": 16,
        "expected_max_steps": EXPECTED_FINAL_STEP,
        "weights_reported": "final",
    }
    for key, expected in exact_model_values.items():
        if model_contract.get(key) != expected:
            raise ValueError(f"preregistered model contract changed {key}")

    y4n = evaluation.get("mapped_y4n_primary")
    b1 = evaluation.get("b1_secondary")
    if not isinstance(y4n, dict) or not isinstance(b1, dict):
        raise ValueError("preregistered evaluation surfaces are missing")
    if y4n.get("session_ids") != Y4N_BASE_SESSION_IDS:
        raise ValueError("preregistered y4n membership changed")
    if y4n.get("expected_stream_ids") != Y4N_STREAM_IDS:
        raise ValueError("preregistered y4n stream identity changed")
    if y4n.get("expected_stream_lengths") != Y4N_STREAM_LENGTHS:
        raise ValueError("preregistered y4n boundaries changed")
    if y4n.get("expected_active_frames") != Y4N_FRAMES:
        raise ValueError("preregistered y4n support changed")
    if y4n.get("truth_sha256") != Y4N_TRUTH_SHA256:
        raise ValueError("preregistered y4n truth receipt changed")
    expected_b1: dict[str, object] = {
        "expected_all_frames": EXPECTED_B1_FRAMES,
        "expected_active_frames": EXPECTED_B1_ACTIVE_FRAMES,
        "expected_streams": EXPECTED_B1_STREAMS,
        "truth_sha256": EXPECTED_B1_TRUTH_SHA256,
        "input_active_sha256": EXPECTED_B1_ACTIVE_SHA256,
    }
    for key, expected in expected_b1.items():
        if b1.get(key) != expected:
            raise ValueError(f"preregistered B1 contract changed {key}")
    if evaluation.get("threshold_policy") != (
        "raw sigmoid probabilities; state and event decisions fixed at 0.5"
    ):
        raise ValueError("preregistered threshold policy changed")
    if evaluation.get("oracle_thresholds_allowed") is not False:
        raise ValueError("preregistered contract permits fitted thresholds")
    if evaluation.get("calibration_allowed") is not False:
        raise ValueError("preregistered contract permits calibration")
    return contract


def expected_run_config(repo: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only run config accepted by this evaluator."""

    model_contract = contract["model_contract"]
    template_path = repo.resolve() / str(model_contract["template"])
    if not template_path.is_file() or sha256_file(template_path) != TEMPLATE_SHA256:
        raise ValueError("matched GRU template bytes changed")
    config = _json_object(template_path, "matched GRU template")
    if "temporal_arch" in config:
        raise ValueError("matched GRU template must use the default GRU")
    config["max_steps"] = EXPECTED_FINAL_STEP
    config["eval_interval"] = EXPECTED_FINAL_STEP
    config["_note"] = EXPECTED_NOTE
    return config


def _state_dicts_identical(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
) -> bool:
    if set(first) != set(second):
        return False
    return all(torch.equal(first[key], second[key]) for key in first)


def validate_run(
    repo: Path,
    run_dir: Path,
    run_id: str,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], BadelineIDM, dict[str, Any]]:
    """Validate the exact completed GRU run and load final weights only."""

    if run_id != RUN_ID or run_dir.name != RUN_ID:
        raise ValueError("run directory is not the preregistered wild GRU run")
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "model.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise ValueError("completed run config or checkpoint is missing")
    config = _json_object(config_path, "run config")
    expected = expected_run_config(repo, contract)
    if config != expected:
        changed = sorted(
            key
            for key in set(config) | set(expected)
            if config.get(key) != expected.get(key)
        )
        raise ValueError(f"run config differs from frozen recipe: {changed}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping")
    if checkpoint.get("config") != config:
        raise ValueError("checkpoint config differs from run config")
    if checkpoint.get("key_order") != list(KEY_ORDER):
        raise ValueError("checkpoint key order changed")
    if checkpoint.get("steps") != EXPECTED_FINAL_STEP:
        raise ValueError("checkpoint is not the fixed final endpoint")
    if checkpoint.get("initialized_from") is not None:
        raise ValueError("wild GRU checkpoint was unexpectedly initialized")
    best_step = checkpoint.get("best_val_step")
    if best_step not in (0, EXPECTED_FINAL_STEP):
        raise ValueError("checkpoint best-val step is outside evaluated endpoints")
    best_bce = checkpoint.get("best_val_mean_bce")
    if not isinstance(best_bce, (float, int)) or not math.isfinite(float(best_bce)):
        raise ValueError("checkpoint best-val loss is not finite")
    positive_weight = checkpoint.get("positive_weight")
    if (
        not isinstance(positive_weight, list)
        or len(positive_weight) != len(KEY_ORDER)
        or not all(
            isinstance(value, (float, int))
            and math.isfinite(float(value))
            and 1.0 <= float(value) <= 10.0
            for value in positive_weight
        )
    ):
        raise ValueError("checkpoint class-balance weights are invalid")
    final_state = checkpoint.get("final_state_dict")
    selected_state = checkpoint.get("model_state_dict")
    if not isinstance(final_state, Mapping) or not final_state:
        raise ValueError("checkpoint lacks final_state_dict")
    if not isinstance(selected_state, Mapping) or not selected_state:
        raise ValueError("checkpoint lacks model_state_dict receipt")

    model = BadelineIDM(config)
    if model.temporal_arch != "gru" or not isinstance(model.temporal, torch.nn.GRUCell):
        raise ValueError("checkpoint recipe does not instantiate the matched GRU")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError("matched GRU parameter count changed")
    model.load_state_dict(final_state, strict=True)

    receipt = {
        "run_config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_steps": EXPECTED_FINAL_STEP,
        "best_val_step": int(best_step),
        "selected_final_tensors_identical": _state_dicts_identical(
            selected_state, final_state
        ),
        "parameter_count": parameter_count,
        "temporal_architecture": "gru",
        "temporal_module": "GRUCell",
        "evaluation_weights": "final_state_dict",
    }
    return config, model, receipt


def validate_y4n_sidecar(
    path: Path,
    *,
    expected_truth_sha256: str = Y4N_TRUTH_SHA256,
) -> dict[str, Any]:
    """Require the exact later-eight support and aligned finite arrays."""

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_SIDECAR_FIELDS:
            raise ValueError("y4n sidecar fields changed")
        truth = np.asarray(archive["y_true"])
        probability = np.asarray(archive["y_prob"])
        active = np.asarray(archive["input_active"])
        lengths = np.asarray(archive["session_lengths"])
        session_ids = np.asarray(archive["session_ids"])
    expected_shape = (Y4N_FRAMES, len(KEY_ORDER))
    if truth.shape != expected_shape or probability.shape != expected_shape:
        raise ValueError("y4n sidecar support changed")
    if truth.dtype != np.uint8 or active.dtype != np.uint8:
        raise ValueError("y4n truth and active receipts must use uint8")
    if probability.dtype != np.float32:
        raise ValueError("y4n probability receipt must use float32")
    if lengths.dtype != np.int64:
        raise ValueError("y4n stream-length receipt must use int64")
    if not np.all(np.isin(truth, (0, 1))):
        raise ValueError("y4n truth is not binary")
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("y4n probabilities are not finite values in [0,1]")
    if active.shape != (Y4N_FRAMES,) or not np.all(active == 1):
        raise ValueError("y4n later-eight active support changed")
    if lengths.tolist() != Y4N_STREAM_LENGTHS:
        raise ValueError("y4n stream lengths or boundaries changed")
    if session_ids.tolist() != Y4N_STREAM_IDS:
        raise ValueError("y4n stream identities changed")
    truth_sha256 = _canonical_array_sha256(truth)
    if truth_sha256 != expected_truth_sha256:
        raise ValueError("y4n mapped-label truth receipt changed")
    return {
        "all_frames": Y4N_FRAMES,
        "input_active_frames": Y4N_FRAMES,
        "streams": len(Y4N_STREAM_IDS),
        "session_ids": session_ids.tolist(),
        "stream_lengths": lengths.tolist(),
        "truth_sha256": truth_sha256,
        "truth_hash_includes_shape": True,
        "finite_aligned_arrays": True,
    }


def _load_sidecar_arrays(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return (
            np.asarray(archive["y_true"]),
            np.asarray(archive["y_prob"]),
            np.asarray(archive["input_active"]),
            np.asarray(archive["session_lengths"]),
        )


def validate_y4n_release(
    report_path: Path,
    marker_path: Path,
    *,
    contract_sha256: str,
    checkpoint_sha256: str,
    expected_truth_sha256: str = Y4N_TRUTH_SHA256,
) -> dict[str, Any]:
    """Validate the completed y4n result before any B1 data is accessed."""

    if not report_path.is_file() or not marker_path.is_file():
        raise ValueError("completed y4n report and marker are required before B1")
    report = _json_object(report_path, "completed y4n report")
    marker = _json_object(marker_path, "completed y4n marker")
    if _contains_disallowed_metric_language(report) or _contains_disallowed_metric_language(marker):
        raise ValueError("y4n release contains a fitted-metric diagnostic")
    expected_common: dict[str, object] = {
        "study_id": STUDY_ID,
        "run_id": RUN_ID,
        "surface": Y4N_SURFACE,
        "contract_sha256": contract_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("y4n report schema changed")
    if report.get("weights") != "final":
        raise ValueError("y4n report did not use final weights")
    contract_receipt = report.get("contract")
    run_receipt = report.get("run_receipt")
    if not isinstance(contract_receipt, dict) or not isinstance(run_receipt, dict):
        raise ValueError("y4n report lacks contract or run receipt")
    for key, expected in expected_common.items():
        if key == "contract_sha256":
            observed = contract_receipt.get("sha256")
        elif key == "checkpoint_sha256":
            observed = run_receipt.get("checkpoint_sha256")
        else:
            observed = report.get(key)
        if observed != expected:
            raise ValueError(f"y4n report changed {key}")
    support = report.get("support")
    expected_support: dict[str, object] = {
        "all_frames": Y4N_FRAMES,
        "input_active_frames": Y4N_FRAMES,
        "streams": len(Y4N_STREAM_IDS),
        "session_ids": Y4N_STREAM_IDS,
        "stream_lengths": Y4N_STREAM_LENGTHS,
        "truth_sha256": expected_truth_sha256,
        "finite_aligned_arrays": True,
    }
    if not isinstance(support, dict) or any(
        support.get(key) != expected for key, expected in expected_support.items()
    ):
        raise ValueError("y4n release support receipt changed")
    fixed_metrics = report.get("fixed_metrics")
    if not isinstance(fixed_metrics, dict):
        raise ValueError("y4n release lacks fixed metrics")
    threshold_policy = fixed_metrics.get("threshold_policy")
    if threshold_policy != {
        "state_probability": 0.5,
        "transition_probability": 0.5,
        "data_fitted_thresholds_used": False,
        "calibration_parameters_fitted": False,
    }:
        raise ValueError("y4n release is not fixed-only")

    if marker.get("schema_version") != MARKER_SCHEMA_VERSION:
        raise ValueError("y4n completion marker schema changed")
    if marker.get("status") != "complete":
        raise ValueError("y4n completion marker is not complete")
    for key, expected in expected_common.items():
        if marker.get(key) != expected:
            raise ValueError(f"y4n completion marker changed {key}")
    if marker.get("report_sha256") != sha256_file(report_path):
        raise ValueError("y4n completion marker report hash changed")
    sidecar_receipt = report.get("prediction_sidecar")
    if not isinstance(sidecar_receipt, dict):
        raise ValueError("y4n report lacks its prediction-sidecar receipt")
    sidecar_value = sidecar_receipt.get("path")
    sidecar_sha256 = sidecar_receipt.get("sha256")
    if not isinstance(sidecar_value, str) or not isinstance(sidecar_sha256, str):
        raise ValueError("y4n prediction-sidecar receipt is malformed")
    sidecar_path = Path(sidecar_value)
    if not sidecar_path.is_file() or sha256_file(sidecar_path) != sidecar_sha256:
        raise ValueError("y4n prediction sidecar is missing or changed")
    validate_y4n_sidecar(
        sidecar_path, expected_truth_sha256=expected_truth_sha256
    )
    if marker.get("sidecar_sha256") != sidecar_sha256:
        raise ValueError("y4n completion marker sidecar hash changed")
    return {
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "completion_marker": str(marker_path),
        "completion_marker_sha256": sha256_file(marker_path),
        "sidecar_sha256": sidecar_sha256,
        "surface": Y4N_SURFACE,
        "weights": "final",
    }


def _refuse_existing(paths: Sequence[Path]) -> None:
    for path in paths:
        if os.path.lexists(path):
            raise ValueError(f"refusing to overwrite fixed-evaluation artifact: {path}")


def _temporary_path(path: Path, *, npz: bool = False) -> Path:
    if npz:
        return path.with_name(f".{path.stem}.tmp.npz")
    return path.with_name(f".{path.name}.tmp")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _publish_result(
    *,
    report_path: Path,
    temporary_report: Path,
    sidecar_path: Path,
    temporary_sidecar: Path,
    marker_path: Path,
    marker: Mapping[str, Any],
) -> None:
    """Publish two artifacts and a final commit marker, rolling back on error."""

    temporary_marker = _temporary_path(marker_path)
    _refuse_existing(
        [report_path, sidecar_path, marker_path, temporary_marker]
    )
    _atomic_json(temporary_marker, marker)
    published: list[Path] = []
    try:
        temporary_sidecar.replace(sidecar_path)
        published.append(sidecar_path)
        temporary_report.replace(report_path)
        published.append(report_path)
        temporary_marker.replace(marker_path)
        published.append(marker_path)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        temporary_marker.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", choices=("y4n-later8", "b1"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--completion-marker", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--b1-marker", type=Path)
    parser.add_argument("--y4n-report", type=Path)
    parser.add_argument("--y4n-completion-marker", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    contract = validate_contract(args.repo, args.contract, args.contract_sha256)
    config, model, run_receipt = validate_run(
        args.repo, args.run, args.run_id, contract
    )

    y4n_release: dict[str, Any] | None = None
    if args.surface == "b1":
        if (
            args.b1_marker is None
            or args.y4n_report is None
            or args.y4n_completion_marker is None
        ):
            raise ValueError(
                "B1 requires its validation marker and the completed y4n release"
            )
        # Release gate: do not stat or read any B1 input or output before this.
        y4n_release = validate_y4n_release(
            args.y4n_report,
            args.y4n_completion_marker,
            contract_sha256=args.contract_sha256,
            checkpoint_sha256=run_receipt["checkpoint_sha256"],
        )

    artifacts = [args.out, args.sidecar, args.completion_marker]
    _refuse_existing(artifacts)
    if args.out.suffix != ".json" or args.sidecar.suffix != ".npz":
        raise ValueError("report must be JSON and prediction sidecar must be NPZ")
    for artifact in artifacts:
        artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = _temporary_path(args.out)
    temporary_sidecar = _temporary_path(args.sidecar, npz=True)
    _refuse_existing([temporary_report, temporary_sidecar])

    if args.surface == "y4n-later8":
        if args.b1_marker is not None:
            raise ValueError("y4n evaluation must not receive a B1 marker")
        session_ids = read_session_ids(args.sessions)
        if session_ids != Y4N_BASE_SESSION_IDS:
            raise ValueError("y4n session list is not the exact later-eight split")
        surface = Y4N_SURFACE
        label_kind = "mapped_foreign_nitrogen"
        label_notice = (
            "Primary matched development comparison against noisy mapped "
            "NitroGen labels; this is not engine-truth performance."
        )
    else:
        assert args.b1_marker is not None
        session_ids = validate_b1_surface(args.data, args.sessions, args.b1_marker)
        surface = B1_SURFACE
        label_kind = "engine_truth_development_b1"
        label_notice = (
            "Fixed post-y4n engine-truth development transfer. B1 has been "
            "consulted repeatedly and cannot select, tune, or promote this model."
        )

    try:
        truth, probability, active, lengths, stream_ids = infer_fixed_state(
            model,
            config,
            args.data,
            session_ids,
            args.device,
            temporary_sidecar,
        )
        if surface == Y4N_SURFACE:
            support = validate_y4n_sidecar(temporary_sidecar)
            if stream_ids != Y4N_STREAM_IDS:
                raise ValueError("y4n inference stream identities changed")
        else:
            support = validate_b1_sidecar(temporary_sidecar)
            if stream_ids != EXPECTED_B1_STREAM_IDS:
                raise ValueError("B1 inference stream identities changed")

        # Reload the serialized arrays: metrics are computed from exactly the
        # sidecar bytes that will be published, not transient inference values.
        truth, probability, active, lengths = _load_sidecar_arrays(
            temporary_sidecar
        )
        metrics = fixed_metric_report(truth, probability, active, lengths.tolist())
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "study_id": STUDY_ID,
            "run_id": RUN_ID,
            "surface": surface,
            "weights": "final",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "label_kind": label_kind,
            "label_notice": label_notice,
            "sessions": list(session_ids),
            "support": support,
            "fixed_metrics": metrics,
            "evaluation_policy": {
                "raw_sigmoid_probabilities": True,
                "fixed_state_threshold": 0.5,
                "fixed_event_threshold": 0.5,
                "threshold_parameters_fitted": False,
                "calibration_parameters_fitted": False,
                "checkpoint_selected_on_this_surface": False,
            },
            "contract": {
                "path": str(args.contract),
                "sha256": args.contract_sha256,
            },
            "run_receipt": run_receipt,
            "prediction_sidecar": {
                "path": str(args.sidecar),
                "sha256": sha256_file(temporary_sidecar),
            },
        }
        if y4n_release is not None:
            report["y4n_release_gate"] = y4n_release
            report["b1_policy"] = {
                "post_y4n_release_only": True,
                "used_for_training": False,
                "used_for_checkpoint_selection": False,
                "used_for_threshold_fitting": False,
                "used_for_calibration_fitting": False,
            }
        if _contains_disallowed_metric_language(report):
            raise AssertionError("fixed-only report contains disallowed metric output")
        _atomic_json(temporary_report, report)
        marker = {
            "schema_version": MARKER_SCHEMA_VERSION,
            "status": "complete",
            "study_id": STUDY_ID,
            "run_id": RUN_ID,
            "surface": surface,
            "weights": "final",
            "contract_sha256": args.contract_sha256,
            "checkpoint_sha256": run_receipt["checkpoint_sha256"],
            "report_sha256": sha256_file(temporary_report),
            "sidecar_sha256": sha256_file(temporary_sidecar),
        }
        if _contains_disallowed_metric_language(marker):
            raise AssertionError("fixed-only marker contains disallowed metric output")
        _publish_result(
            report_path=args.out,
            temporary_report=temporary_report,
            sidecar_path=args.sidecar,
            temporary_sidecar=temporary_sidecar,
            marker_path=args.completion_marker,
            marker=marker,
        )
    finally:
        temporary_report.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "surface": surface,
                "weights": "final",
                "input_active_frames": support["input_active_frames"],
                "fixed_threshold": 0.5,
                "report": str(args.out),
                "sidecar": str(args.sidecar),
                "completion_marker": str(args.completion_marker),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
