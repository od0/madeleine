from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.oracle_window_localization import (
    HEAD_NAMES,
    OracleExample,
    state_dict_sha256,
)
from experiments.validate_oracle_window_run import (
    _canonical_json_sha256,
    _finalize_receipt,
    _model_from_config,
    _require_array_equal,
    _write_json_exclusive_atomic,
    load_final_checkpoint,
    rebuild_dataset_evidence,
    validate_completion_marker,
    validate_run_receipt,
    validate_training_log,
)


def _config() -> dict:
    return {
        "study_id": "fixture",
        "dataset": {
            "candidate_width": 16,
            "context_halo": 3,
            "feature_content_sha256": "feature-content",
            "expected_support": {
                "training_examples": 1,
                "validation_examples": 1,
                "validation_blocks": 1,
            },
        },
        "model": {
            "feature_dim": 4,
            "projection_dim": 5,
            "temporal_dim": 6,
            "tcn_dilations": [1],
        },
        "training": {"seed": 0, "epochs": 40},
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_array_validation_accepts_nan_identity_and_rejects_drift() -> None:
    expected = np.asarray([[1.0, np.nan]], dtype=np.float32)
    _require_array_equal("fixture", expected.copy(), expected)

    changed = expected.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(2.0))
    with pytest.raises(ValueError, match="fixture changed"):
        _require_array_equal("fixture", changed, expected)

    with pytest.raises(ValueError, match="dtype/shape"):
        _require_array_equal("fixture", expected.astype(np.float64), expected)


def test_run_receipt_binds_seed_endpoint_flags_and_input_hashes(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    feature_path = tmp_path / "feature.json"
    manifest_path = tmp_path / "manifest.json"
    predictions_path = tmp_path / "predictions.npz"
    for path in (config_path, feature_path, manifest_path):
        _write_json(path, {})
    predictions_path.write_bytes(b"predictions")
    receipt_path = tmp_path / "run_receipt.json"
    receipt = {
        "schema_version": "madeleine.oracle-window-run.v1",
        "status": "predictions_complete_unscored",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "feature_receipt_path": str(feature_path),
        "feature_receipt_sha256": _sha256(feature_path),
        "feature_content_sha256": "feature-content",
        "dataset_manifest_sha256": _sha256(manifest_path),
        "prediction_sidecar_sha256": _sha256(predictions_path),
        "seed": 0,
        "epochs": 40,
        "configured_epochs": 40,
        "matched_initialization": True,
        "matched_batch_order": True,
        "final_weights_only": True,
        "device": "cpu",
        "implementation": {"relevant_file_sha256": {"f": "h"}},
    }
    _write_json(receipt_path, receipt)

    observed = validate_run_receipt(
        receipt_path=receipt_path,
        config_path=config_path,
        feature_receipt_path=feature_path,
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        config=_config(),
        implementation_sha256={"f": "h"},
        device_name="cpu",
    )
    assert observed["seed"] == 0

    receipt["epochs"] = 1
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="40-epoch endpoint"):
        validate_run_receipt(
            receipt_path=receipt_path,
            config_path=config_path,
            feature_receipt_path=feature_path,
            manifest_path=manifest_path,
            predictions_path=predictions_path,
            config=_config(),
            implementation_sha256={"f": "h"},
            device_name="cpu",
        )


def test_checkpoint_is_hashed_strictly_reloaded_and_identity_checked(tmp_path: Path) -> None:
    config = _config()
    torch.manual_seed(0)
    model = _model_from_config(config)
    initial_sha = state_dict_sha256(model)
    run_receipt = {
        "config_sha256": "config",
        "feature_receipt_sha256": "feature",
        "initial_state_sha256": initial_sha,
    }
    path = tmp_path / "conditional_model.pt"
    torch.save(
        {
            **run_receipt,
            "seed": 0,
            "epochs": 40,
            "arm": "conditional_softmax",
            "model_state_dict": model.state_dict(),
        },
        path,
    )

    reloaded, receipt = load_final_checkpoint(
        path=path,
        arm="conditional_softmax",
        config=config,
        run_receipt=run_receipt,
    )
    assert receipt["sha256"] == _sha256(path)
    assert receipt["state_dict_sha256"] == state_dict_sha256(reloaded) == initial_sha
    assert receipt["strict_reload"]

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["epochs"] = 39
    torch.save(payload, path)
    with pytest.raises(ValueError, match="epochs"):
        load_final_checkpoint(
            path=path,
            arm="conditional_softmax",
            config=config,
            run_receipt=run_receipt,
        )


def test_rebuild_requires_exact_manifest_identity_and_uniqueness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.validate_oracle_window_run as validator

    example = OracleExample(
        split="validation",
        session_id="s",
        run_index=0,
        array_index=10,
        engine_frame_idx=10,
        head_index=0,
        key_index=0,
        event_type_index=0,
        offset=3,
        crop_start=4,
        candidate_start=7,
        block_id="s:run0:block0",
    )
    train_example = OracleExample(**{**example.__dict__, "split": "train"})
    manifest = {
        "schema_version": "madeleine.oracle-window-dataset.v1",
        "head_names": list(HEAD_NAMES),
        "train_examples": 1,
        "validation_examples": 1,
        "validation_block_count": 1,
    }
    arrays = validator._example_arrays([example])
    monkeypatch.setattr(
        validator,
        "_prepare_data",
        lambda **kwargs: ({"s": object()}, (train_example,), (example,), {}),
    )
    monkeypatch.setattr(validator, "_dataset_manifest", lambda **kwargs: manifest)

    result = rebuild_dataset_evidence(
        feature_root=tmp_path,
        config=_config(),
        stored_manifest=manifest,
        arrays=arrays,
    )
    assert result["val_examples"] == (example,)

    changed = dict(arrays)
    changed["engine_frame_idx"] = np.asarray([11], dtype=np.int64)
    with pytest.raises(ValueError, match="engine_frame_idx"):
        rebuild_dataset_evidence(
            feature_root=tmp_path,
            config=_config(),
            stored_manifest=manifest,
            arrays=changed,
        )


def test_marker_content_hash_and_every_input_binding_are_validated(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    predictions_path = tmp_path / "predictions.npz"
    manifest_path = tmp_path / "manifest.json"
    config_path = tmp_path / "config.json"
    for path in (predictions_path,):
        path.write_bytes(b"prediction")
    for path in (manifest_path, config_path):
        _write_json(path, {})
    report = {
        "study_id": "fixture",
        "decision_gate": {"decision": "reject"},
    }
    _write_json(report_path, report)
    marker_path = tmp_path / "complete.json"
    marker = {
        "schema_version": "madeleine.oracle-window-complete.v1",
        "status": "complete",
        "study_id": "fixture",
        "report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "predictions": {
            "path": str(predictions_path),
            "sha256": _sha256(predictions_path),
        },
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "decision": "reject",
    }
    marker["content_sha256"] = _canonical_json_sha256(marker)
    _write_json(marker_path, marker)

    validated = validate_completion_marker(
        marker_path=marker_path,
        report_path=report_path,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        config_path=config_path,
        report=report,
    )
    assert validated["content_sha256"] == marker["content_sha256"]

    predictions_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="predictions hash"):
        validate_completion_marker(
            marker_path=marker_path,
            report_path=report_path,
            predictions_path=predictions_path,
            manifest_path=manifest_path,
            config_path=config_path,
            report=report,
        )


def test_training_log_and_audit_publication_are_finite_atomic_and_exclusive(
    tmp_path: Path,
) -> None:
    training_path = tmp_path / "training.json"
    rows = [{"epoch": float(index), "loss": 1.0 / index} for index in range(1, 41)]
    training = {
        "conditional_softmax": rows,
        "dense_bce": rows,
        "fixed_final_epoch": 40,
        "configured_final_epoch": 40,
        "validation_used_for_training_or_selection": False,
    }
    _write_json(training_path, training)
    assert validate_training_log(training_path, epochs=40) == training

    receipt = _finalize_receipt(
        {"schema_version": "fixture", "status": "complete", "value": 1}
    )
    expected_hash = receipt["content_sha256"]
    without_hash = dict(receipt)
    without_hash.pop("content_sha256")
    assert expected_hash == _canonical_json_sha256(without_hash)
    output = tmp_path / "audit.json"
    _write_json_exclusive_atomic(output, receipt)
    assert json.loads(output.read_text()) == receipt
    with pytest.raises(FileExistsError, match="overwrite"):
        _write_json_exclusive_atomic(output, receipt)
