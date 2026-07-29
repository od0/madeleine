"""Independently validate and bind a completed oracle-window seed-zero run.

This is a post-run audit, not a second decision surface.  It rebuilds the
frozen dataset identities, reloads both final checkpoints, reproduces all
three prediction families, recomputes the fixed score, verifies the original
completion marker, and publishes one non-overwriting supplementary receipt.
It never changes the frozen run, decision report, marker, or contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.oracle_window_localization import (
    HEAD_NAMES,
    OracleWindowDataset,
    OracleWindowEventModel,
    _array_sha256,
    _dataset_manifest,
    _example_arrays,
    _prepare_data,
    predict_current_dense_reference,
    predict_probabilities,
    sha256_file,
    state_dict_sha256,
)
from experiments.score_oracle_window_localization import (
    load_prediction_sidecar,
    score_experiment,
)


AUDIT_SCHEMA_VERSION = "madeleine.oracle-window-run-audit.v1"
RUN_SCHEMA_VERSION = "madeleine.oracle-window-run.v1"
MANIFEST_SCHEMA_VERSION = "madeleine.oracle-window-dataset.v1"
SCORE_SCHEMA_VERSION = "madeleine.oracle-window-score.v1"
MARKER_SCHEMA_VERSION = "madeleine.oracle-window-complete.v1"
EXPECTED_RUN_INVENTORY = frozenset(
    {
        "conditional_model.pt",
        "dataset_manifest.json",
        "dense_model.pt",
        "predictions.npz",
        "run_receipt.json",
        "training_log.json",
    }
)
CHECKPOINT_INVENTORY = frozenset(
    {
        "config_sha256",
        "feature_receipt_sha256",
        "initial_state_sha256",
        "seed",
        "epochs",
        "arm",
        "model_state_dict",
    }
)


def _json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same_path(observed: object, expected: Path, label: str) -> None:
    _require(isinstance(observed, str), f"{label} path is not a string")
    _require(
        Path(observed).resolve() == Path(expected).resolve(),
        f"{label} path changed: {observed}",
    )


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _arrays_equal(observed: np.ndarray, expected: np.ndarray) -> bool:
    if observed.dtype != expected.dtype or observed.shape != expected.shape:
        return False
    if np.issubdtype(expected.dtype, np.floating):
        return bool(np.array_equal(observed, expected, equal_nan=True))
    return bool(np.array_equal(observed, expected))


def _require_array_equal(
    label: str, observed: np.ndarray, expected: np.ndarray
) -> None:
    if _arrays_equal(np.asarray(observed), np.asarray(expected)):
        return
    detail = (
        f"observed dtype/shape={observed.dtype}/{observed.shape}, "
        f"expected={expected.dtype}/{expected.shape}"
    )
    if (
        observed.shape == expected.shape
        and np.issubdtype(observed.dtype, np.floating)
        and np.issubdtype(expected.dtype, np.floating)
    ):
        finite = np.isfinite(observed) & np.isfinite(expected)
        maximum = (
            float(np.max(np.abs(observed[finite] - expected[finite])))
            if np.any(finite)
            else None
        )
        detail += f", maximum finite absolute difference={maximum}"
    raise ValueError(f"{label} changed: {detail}")


def _model_from_config(config: Mapping[str, Any]) -> OracleWindowEventModel:
    model = config["model"]
    dataset = config["dataset"]
    return OracleWindowEventModel(
        feature_dim=int(model["feature_dim"]),
        projection_dim=int(model["projection_dim"]),
        temporal_dim=int(model["temporal_dim"]),
        dilations=tuple(int(value) for value in model["tcn_dilations"]),
        width=int(dataset["candidate_width"]),
        halo=int(dataset["context_halo"]),
    )


def validate_frozen_config(config_path: Path, repo: Path) -> dict[str, Any]:
    """Validate every decision-bearing scorer/trainer knob used by this audit."""

    config = _json(config_path, "decision config")
    _require(
        config.get("schema_version") == "madeleine.oracle-window-decision.v1",
        "decision config schema changed",
    )
    _require(
        config.get("status") == "preregistered_before_validation_inference",
        "decision config is not the frozen preregistration",
    )
    dataset = config.get("dataset")
    training = config.get("training")
    evaluation = config.get("evaluation")
    _require(isinstance(dataset, Mapping), "decision config lacks dataset policy")
    _require(isinstance(training, Mapping), "decision config lacks training policy")
    _require(isinstance(evaluation, Mapping), "decision config lacks evaluation policy")
    _require(int(dataset["candidate_width"]) == 16, "candidate width changed")
    _require(int(dataset["context_halo"]) == 8, "context halo changed")
    _require(int(training["seed"]) == 0, "primary seed changed")
    _require(int(training["epochs"]) == 40, "fixed endpoint changed")
    _require(int(evaluation["bootstrap_replicates"]) == 5_000, "bootstrap count changed")
    _require(int(evaluation["bootstrap_block_frames"]) == 600, "block size changed")
    implementation = config.get("implementation")
    _require(isinstance(implementation, Mapping), "implementation receipt is missing")
    expected_hashes = implementation.get("sha256")
    _require(
        isinstance(expected_hashes, Mapping) and bool(expected_hashes),
        "frozen implementation hashes are missing",
    )
    observed_hashes: dict[str, str] = {}
    for relative, expected in expected_hashes.items():
        _require(
            isinstance(relative, str) and isinstance(expected, str),
            "implementation hash map is malformed",
        )
        path = repo / relative
        _require(path.is_file(), f"frozen implementation file is missing: {relative}")
        observed = sha256_file(path)
        _require(observed == expected, f"frozen implementation changed: {relative}")
        observed_hashes[relative] = observed
    return {"config": config, "implementation_sha256": observed_hashes}


def validate_feature_receipt(
    *,
    feature_receipt_path: Path,
    feature_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = config["dataset"]
    expected_sha = str(dataset["feature_receipt_sha256"])
    observed_sha = sha256_file(feature_receipt_path)
    _require(observed_sha == expected_sha, "feature receipt hash changed")
    receipt = _json(feature_receipt_path, "feature receipt")
    _require(receipt.get("status") == "complete", "feature receipt is incomplete")
    _same_path(receipt.get("published_output"), feature_root, "feature receipt")
    _require(
        receipt.get("content_sha256") == dataset["feature_content_sha256"],
        "feature content hash changed",
    )
    checks = receipt.get("checks")
    _require(
        isinstance(checks, Mapping)
        and bool(checks)
        and all(value is True for value in checks.values()),
        "feature receipt checks are not all true",
    )
    return {"receipt": receipt, "sha256": observed_sha}


def validate_run_receipt(
    *,
    receipt_path: Path,
    config_path: Path,
    feature_receipt_path: Path,
    manifest_path: Path,
    predictions_path: Path,
    config: Mapping[str, Any],
    implementation_sha256: Mapping[str, str],
    device_name: str,
) -> dict[str, Any]:
    receipt = _json(receipt_path, "run receipt")
    _require(receipt.get("schema_version") == RUN_SCHEMA_VERSION, "run schema changed")
    _require(
        receipt.get("status") == "predictions_complete_unscored",
        "run receipt status changed",
    )
    _same_path(receipt.get("config_path"), config_path, "run config")
    _same_path(
        receipt.get("feature_receipt_path"), feature_receipt_path, "run feature receipt"
    )
    _require(receipt.get("config_sha256") == sha256_file(config_path), "run/config hash mismatch")
    _require(
        receipt.get("feature_receipt_sha256") == sha256_file(feature_receipt_path),
        "run/feature-receipt hash mismatch",
    )
    _require(
        receipt.get("feature_content_sha256")
        == config["dataset"]["feature_content_sha256"],
        "run feature-content identity changed",
    )
    _require(
        receipt.get("dataset_manifest_sha256") == sha256_file(manifest_path),
        "run/manifest hash mismatch",
    )
    _require(
        receipt.get("prediction_sidecar_sha256") == sha256_file(predictions_path),
        "run/prediction hash mismatch",
    )
    _require(int(receipt.get("seed", -1)) == 0 == int(config["training"]["seed"]), "run is not seed zero")
    _require(
        int(receipt.get("epochs", -1))
        == int(receipt.get("configured_epochs", -2))
        == int(config["training"]["epochs"])
        == 40,
        "run did not reach the frozen 40-epoch endpoint",
    )
    for field in ("matched_initialization", "matched_batch_order", "final_weights_only"):
        _require(receipt.get(field) is True, f"run receipt does not prove {field}")
    _require(receipt.get("device") == device_name, "validation device differs from run device")
    run_implementation = receipt.get("implementation")
    _require(isinstance(run_implementation, Mapping), "run lacks implementation receipt")
    _require(
        run_implementation.get("relevant_file_sha256") == dict(implementation_sha256),
        "run implementation hashes differ from frozen bytes",
    )
    return receipt


def rebuild_dataset_evidence(
    *,
    feature_root: Path,
    config: Mapping[str, Any],
    stored_manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Rebuild all examples and require exact manifest and sidecar identities."""

    sessions, train_examples, val_examples, construction = _prepare_data(
        feature_root=feature_root, config=config
    )
    rebuilt_manifest = _dataset_manifest(
        sessions=sessions,
        train_examples=train_examples,
        val_examples=val_examples,
        construction=construction,
        config=config,
    )
    _require(
        rebuilt_manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION,
        "rebuilt manifest schema changed",
    )
    _require(rebuilt_manifest == stored_manifest, "stored dataset manifest does not rebuild exactly")
    _require(
        rebuilt_manifest.get("head_names") == list(HEAD_NAMES),
        "manifest head order changed",
    )
    expected_support = config["dataset"]["expected_support"]
    _require(
        int(rebuilt_manifest["train_examples"])
        == int(expected_support["training_examples"]),
        "training support changed from preregistration",
    )
    _require(
        int(rebuilt_manifest["validation_examples"])
        == int(expected_support["validation_examples"]),
        "validation support changed from preregistration",
    )
    _require(
        int(rebuilt_manifest["validation_block_count"])
        == int(expected_support["validation_blocks"]),
        "validation block support changed from preregistration",
    )
    expected_identity = _example_arrays(val_examples)
    for name, expected in expected_identity.items():
        _require(name in arrays, f"prediction sidecar lacks identity array: {name}")
        _require_array_equal(f"prediction identity {name}", arrays[name], expected)
    identity = list(
        zip(
            arrays["session_id"].astype(str).tolist(),
            arrays["run_index"].astype(np.int64).tolist(),
            arrays["array_index"].astype(np.int64).tolist(),
            arrays["engine_frame_idx"].astype(np.int64).tolist(),
            arrays["head_index"].astype(np.int64).tolist(),
            arrays["true_offset"].astype(np.int64).tolist(),
            strict=True,
        )
    )
    _require(len(identity) == len(set(identity)), "prediction sidecar repeats an example identity")
    return {
        "sessions": sessions,
        "train_examples": train_examples,
        "val_examples": val_examples,
        "manifest": rebuilt_manifest,
        "identity_arrays": expected_identity,
    }


def validate_training_log(path: Path, *, epochs: int) -> dict[str, Any]:
    value = _json(path, "training log")
    _require(
        set(value) == {
            "conditional_softmax",
            "dense_bce",
            "fixed_final_epoch",
            "configured_final_epoch",
            "validation_used_for_training_or_selection",
        },
        "training log inventory changed",
    )
    _require(
        int(value["fixed_final_epoch"])
        == int(value["configured_final_epoch"])
        == epochs,
        "training log endpoint changed",
    )
    _require(
        value["validation_used_for_training_or_selection"] is False,
        "validation was used for training or checkpoint selection",
    )
    for arm in ("conditional_softmax", "dense_bce"):
        rows = value[arm]
        _require(isinstance(rows, list) and len(rows) == epochs, f"{arm} log length changed")
        for index, row in enumerate(rows, start=1):
            _require(isinstance(row, Mapping), f"{arm} log row is malformed")
            _require(float(row.get("epoch", -1)) == float(index), f"{arm} epoch order changed")
            loss = float(row.get("loss", float("nan")))
            _require(np.isfinite(loss) and loss >= 0, f"{arm} loss is invalid")
    return value


def load_final_checkpoint(
    *,
    path: Path,
    arm: str,
    config: Mapping[str, Any],
    run_receipt: Mapping[str, Any],
) -> tuple[OracleWindowEventModel, dict[str, Any]]:
    """Hash, reload, strictly validate, and return one final model."""

    _require(path.is_file() and not path.is_symlink(), f"checkpoint is not a regular file: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), f"checkpoint is not a mapping: {path}")
    _require(set(payload) == CHECKPOINT_INVENTORY, f"checkpoint inventory changed: {path}")
    expected_scalars = {
        "config_sha256": run_receipt["config_sha256"],
        "feature_receipt_sha256": run_receipt["feature_receipt_sha256"],
        "initial_state_sha256": run_receipt["initial_state_sha256"],
        "seed": 0,
        "epochs": 40,
        "arm": arm,
    }
    for name, expected in expected_scalars.items():
        _require(payload.get(name) == expected, f"{path.name} changed {name}")
    state = payload.get("model_state_dict")
    _require(isinstance(state, Mapping) and bool(state), f"{path.name} has no model state")
    for name, tensor in state.items():
        _require(isinstance(name, str) and isinstance(tensor, torch.Tensor), f"{path.name} state is malformed")
        _require(bool(torch.isfinite(tensor).all()), f"{path.name} has non-finite tensor {name}")
    model = _model_from_config(config)
    model.load_state_dict(state, strict=True)
    receipt = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "state_dict_sha256": state_dict_sha256(model),
        "arm": arm,
        "seed": int(payload["seed"]),
        "epochs": int(payload["epochs"]),
        "config_sha256": str(payload["config_sha256"]),
        "feature_receipt_sha256": str(payload["feature_receipt_sha256"]),
        "initial_state_sha256": str(payload["initial_state_sha256"]),
        "strict_reload": True,
        "all_tensors_finite": True,
    }
    return model, receipt


def validate_completion_marker(
    *,
    marker_path: Path,
    report_path: Path,
    predictions_path: Path,
    manifest_path: Path,
    config_path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    marker = _json(marker_path, "completion marker")
    _require(set(marker) == {
        "schema_version", "status", "study_id", "report", "predictions",
        "dataset_manifest", "config", "decision", "content_sha256"
    }, "completion marker inventory changed")
    _require(marker.get("schema_version") == MARKER_SCHEMA_VERSION, "completion marker schema changed")
    _require(marker.get("status") == "complete", "completion marker is incomplete")
    _require(marker.get("study_id") == report.get("study_id"), "marker study changed")
    for name, path in (
        ("report", report_path),
        ("predictions", predictions_path),
        ("dataset_manifest", manifest_path),
        ("config", config_path),
    ):
        binding = marker.get(name)
        _require(isinstance(binding, Mapping), f"marker lacks {name} binding")
        _same_path(binding.get("path"), path, f"marker {name}")
        _require(binding.get("sha256") == sha256_file(path), f"marker {name} hash changed")
    _require(marker.get("decision") == report["decision_gate"]["decision"], "marker decision changed")
    without_hash = dict(marker)
    content_sha = without_hash.pop("content_sha256")
    _require(content_sha == _canonical_json_sha256(without_hash), "marker content hash changed")
    return marker


def _write_json_exclusive_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Fsync and publish a fully serialized JSON object without overwrite."""

    target = Path(path)
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite audit receipt: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _require(_json(temporary, "serialized audit receipt") == value, "audit receipt changed on reload")
        os.link(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _finalize_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    _require("content_sha256" not in result, "audit receipt already has a content hash")
    result["content_sha256"] = _canonical_json_sha256(result)
    return result


def validate_oracle_window_run(
    *,
    repo: Path,
    run: Path,
    config_path: Path,
    feature_receipt_path: Path,
    current_checkpoint_path: Path,
    report_path: Path,
    marker_path: Path,
    output_path: Path,
    device_name: str,
) -> dict[str, Any]:
    """Perform the complete independent post-run audit and publish its receipt."""

    paths = [repo, run, config_path, feature_receipt_path, current_checkpoint_path, report_path, marker_path]
    _require(all(Path(path).exists() for path in paths), "an audit input is missing")
    if os.path.lexists(output_path):
        raise FileExistsError(f"refusing to overwrite audit receipt: {output_path}")
    repo = repo.resolve()
    run = run.resolve()
    actual_inventory = {path.name for path in run.iterdir()}
    _require(actual_inventory == EXPECTED_RUN_INVENTORY, "run artifact inventory changed")
    manifest_path = run / "dataset_manifest.json"
    predictions_path = run / "predictions.npz"
    run_receipt_path = run / "run_receipt.json"
    training_log_path = run / "training_log.json"

    frozen = validate_frozen_config(config_path, repo)
    config = frozen["config"]
    feature_root_value = Path(str(config["dataset"]["feature_root"]))
    feature_root = feature_root_value if feature_root_value.is_absolute() else repo / feature_root_value
    feature = validate_feature_receipt(
        feature_receipt_path=feature_receipt_path,
        feature_root=feature_root,
        config=config,
    )
    manifest = _json(manifest_path, "dataset manifest")
    width = int(config["dataset"]["candidate_width"])
    arrays = load_prediction_sidecar(predictions_path, width=width)
    run_receipt = validate_run_receipt(
        receipt_path=run_receipt_path,
        config_path=config_path,
        feature_receipt_path=feature_receipt_path,
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        config=config,
        implementation_sha256=frozen["implementation_sha256"],
        device_name=device_name,
    )
    rebuilt = rebuild_dataset_evidence(
        feature_root=feature_root,
        config=config,
        stored_manifest=manifest,
        arrays=arrays,
    )
    _require(int(run_receipt["train_examples"]) == len(rebuilt["train_examples"]), "run training count changed")
    _require(int(run_receipt["validation_examples"]) == len(rebuilt["val_examples"]), "run validation count changed")
    training_log = validate_training_log(training_log_path, epochs=40)

    torch.manual_seed(0)
    initial_model = _model_from_config(config)
    _require(
        state_dict_sha256(initial_model) == run_receipt["initial_state_sha256"],
        "seed-zero initial model does not reproduce",
    )
    conditional_model, conditional_checkpoint = load_final_checkpoint(
        path=run / "conditional_model.pt",
        arm="conditional_softmax",
        config=config,
        run_receipt=run_receipt,
    )
    dense_model, dense_checkpoint = load_final_checkpoint(
        path=run / "dense_model.pt",
        arm="dense_bce",
        config=config,
        run_receipt=run_receipt,
    )

    if device_name == "mps":
        _require(torch.backends.mps.is_available(), "MPS is unavailable for exact replay")
    elif device_name == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable for exact replay")
    elif device_name != "cpu":
        raise ValueError("device must be cpu, mps, or cuda")
    device = torch.device(device_name)
    torch.use_deterministic_algorithms(True, warn_only=(device.type != "cuda"))
    val_dataset = OracleWindowDataset(
        rebuilt["sessions"],
        rebuilt["val_examples"],
        width=width,
        halo=int(config["dataset"]["context_halo"]),
    )
    batch_size = int(config["training"]["eval_batch_size"])
    conditional_replay = predict_probabilities(
        conditional_model.to(device), val_dataset, device=device, batch_size=batch_size
    )
    dense_replay = predict_probabilities(
        dense_model.to(device), val_dataset, device=device, batch_size=batch_size
    )
    _require_array_equal("conditional checkpoint predictions", conditional_replay, arrays["conditional_prob"])
    _require_array_equal("dense checkpoint predictions", dense_replay, arrays["dense_prob"])

    current_config = config["current_dense_reference"]
    current_replay, current_support, current_receipt = predict_current_dense_reference(
        checkpoint_path=current_checkpoint_path,
        checkpoint_sha256=str(current_config["checkpoint_sha256"]),
        sessions=rebuilt["sessions"],
        examples=rebuilt["val_examples"],
        width=width,
        device=device,
        target_chunk=int(current_config["target_chunk"]),
    )
    _require_array_equal("retained-reference support", current_support, arrays["current_dense_support"])
    _require_array_equal("retained-reference predictions", current_replay, arrays["current_dense_prob"])
    _require(current_receipt == run_receipt["current_dense_reference"], "retained-reference receipt changed")
    _require(
        int(current_support.sum()) == int(current_config["expected_common_validation_examples"]),
        "retained-reference common support changed",
    )

    report = _json(report_path, "score report")
    _require(report.get("schema_version") == SCORE_SCHEMA_VERSION, "score schema changed")
    _require(report.get("status") == "complete", "score report is incomplete")
    _require(report.get("study_id") == config["study_id"], "score study changed")
    _require(report.get("config_sha256") == sha256_file(config_path), "score/config binding changed")
    _require(report.get("dataset_manifest_sha256") == sha256_file(manifest_path), "score/manifest binding changed")
    _require(report.get("prediction_sidecar_sha256") == sha256_file(predictions_path), "score/prediction binding changed")
    recomputed_report = score_experiment(
        predictions_path=predictions_path,
        dataset_manifest_path=manifest_path,
        config_path=config_path,
    )
    _require(recomputed_report == report, "fixed-policy score does not reproduce exactly")
    marker = validate_completion_marker(
        marker_path=marker_path,
        report_path=report_path,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        config_path=config_path,
        report=report,
    )

    sidecar_arrays = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
        }
        for name, value in sorted(arrays.items())
    }
    audit_base = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": config["study_id"],
        "scope": "post-run validation only; frozen decision bytes and result are unchanged",
        "validator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "platform": sys.platform,
        },
        "bindings": {
            "config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
            "feature_receipt": {
                "path": str(feature_receipt_path.resolve()),
                "sha256": feature["sha256"],
                "content_sha256": feature["receipt"]["content_sha256"],
            },
            "run_receipt": {"path": str(run_receipt_path), "sha256": sha256_file(run_receipt_path)},
            "training_log": {"path": str(training_log_path), "sha256": sha256_file(training_log_path)},
            "dataset_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "prediction_sidecar": {
                "path": str(predictions_path),
                "sha256": sha256_file(predictions_path),
                "arrays": sidecar_arrays,
            },
            "conditional_checkpoint": conditional_checkpoint,
            "dense_checkpoint": dense_checkpoint,
            "retained_reference_checkpoint": {
                "path": str(current_checkpoint_path.resolve()),
                "bytes": current_checkpoint_path.stat().st_size,
                "sha256": sha256_file(current_checkpoint_path),
            },
            "score_report": {"path": str(report_path.resolve()), "sha256": sha256_file(report_path)},
            "completion_marker": {
                "path": str(marker_path.resolve()),
                "sha256": sha256_file(marker_path),
                "content_sha256": marker["content_sha256"],
            },
        },
        "reproduction": {
            "manifest_exact": True,
            "sidecar_identity_exact": True,
            "initial_state_exact": True,
            "conditional_predictions_exact": True,
            "dense_predictions_exact": True,
            "retained_reference_predictions_exact": True,
            "score_report_exact": True,
            "completion_marker_valid": True,
            "conditional_prediction_sha256": _array_sha256(conditional_replay),
            "dense_prediction_sha256": _array_sha256(dense_replay),
            "retained_reference_prediction_sha256": _array_sha256(current_replay),
            "retained_reference_support_sha256": _array_sha256(current_support),
            "train_examples": len(rebuilt["train_examples"]),
            "validation_examples": len(rebuilt["val_examples"]),
            "validation_blocks": int(manifest["validation_block_count"]),
        },
        "run_contract": {
            "seed": int(run_receipt["seed"]),
            "epochs": int(run_receipt["epochs"]),
            "configured_epochs": int(run_receipt["configured_epochs"]),
            "matched_initialization": run_receipt["matched_initialization"],
            "matched_batch_order": run_receipt["matched_batch_order"],
            "final_weights_only": run_receipt["final_weights_only"],
            "validation_used_for_training_or_selection": training_log[
                "validation_used_for_training_or_selection"
            ],
            "frozen_implementation_sha256": dict(frozen["implementation_sha256"]),
        },
        "decision": {
            "original": report["decision_gate"]["decision"],
            "passed": report["decision_gate"]["passed"],
            "unchanged_by_audit": True,
        },
    }
    audit = _finalize_receipt(audit_base)
    _write_json_exclusive_atomic(output_path, audit)
    _require(_json(output_path, "published audit receipt") == audit, "published audit changed")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--feature-receipt", required=True, type=Path)
    parser.add_argument("--current-checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    audit = validate_oracle_window_run(
        repo=args.repo,
        run=args.run,
        config_path=args.config,
        feature_receipt_path=args.feature_receipt,
        current_checkpoint_path=args.current_checkpoint,
        report_path=args.report,
        marker_path=args.marker,
        output_path=args.out,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "decision": audit["decision"]["original"],
                "audit": str(args.out),
                "content_sha256": audit["content_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
