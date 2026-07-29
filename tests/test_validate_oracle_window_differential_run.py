from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.oracle_window_differential_followup import (
    METADATA_NAMES,
    DifferentialCandidateModel,
)
from experiments.validate_oracle_window_differential_run import (
    _array_sha256,
    _canonical_json_sha256,
    _finalize_receipt,
    _json,
    _sha256_file,
    _state_dict_sha256,
    _write_json_exclusive_atomic,
    load_final_checkpoint,
    validate_cache_receipt,
    validate_completion_marker,
    validate_run_receipt,
    validate_score_report,
    validate_training_log,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metadata(count: int = 1, *, session: str = "session-a") -> dict[str, np.ndarray]:
    truth = (np.arange(count) % 16).astype(np.int8)
    heads = (np.arange(count) % 14).astype(np.int16)
    crop_start = np.arange(count, dtype=np.int64) * 40
    return {
        "session_id": np.asarray([session] * count),
        "run_index": np.zeros(count, dtype=np.int32),
        "array_index": crop_start + 8 + truth.astype(np.int64),
        "engine_frame_idx": np.arange(100, 100 + count, dtype=np.int64),
        "head_index": heads,
        "key_index": (heads % 7).astype(np.int8),
        "event_type_index": (heads // 7).astype(np.int8),
        "true_offset": truth,
        "crop_start": crop_start,
        "block_id": np.asarray([f"{session}:run0:block0"] * count),
    }


def _cache_archive(path: Path, *, session: str) -> dict:
    rgb = np.zeros((1, 32, 32, 32, 3), dtype=np.uint8)
    np.savez_compressed(path, rgb=rgb, **_metadata(session=session))
    return {
        "bytes": path.stat().st_size,
        "file_sha256": _sha256_file(path),
        "rgb_sha256": _array_sha256(rgb),
        "examples": 1,
        "rgb_shape": list(rgb.shape),
        "rgb_dtype": "uint8",
    }


def test_json_reader_rejects_duplicate_keys_and_nonfinite_values(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value":1,"value":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _json(duplicate, "fixture")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON value"):
        _json(nonfinite, "fixture")


def test_cache_receipt_closes_manifest_files_content_and_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.validate_oracle_window_differential_run as validator

    monkeypatch.setattr(validator, "EXPECTED_TRAIN_EXAMPLES", 1)
    monkeypatch.setattr(validator, "EXPECTED_VALIDATION_EXAMPLES", 1)
    cache = tmp_path / "cache"
    cache.mkdir()
    train_evidence = _cache_archive(cache / "train.npz", session="train-session")
    validation_evidence = _cache_archive(
        cache / "validation.npz", session="validation-session"
    )
    base_manifest = tmp_path / "base_manifest.json"
    _write_json(base_manifest, {"schema_version": "fixture"})
    cache_manifest = {
        "schema_version": "madeleine.oracle-window-pixel-crops.v1",
        "base_config_sha256": "base-config",
        "base_dataset_manifest_sha256": _sha256_file(base_manifest),
        "feature_receipt_sha256": "feature-receipt",
        "source_build_manifest_sha256": "source-build",
        "crop_frames": 32,
        "source_frame_size": 128,
        "output_size": 32,
        "downsampling": "fixture",
        "color_order": "source RGB preserved",
        "cache": {
            "train": train_evidence,
            "validation": validation_evidence,
        },
        "source_checks": {
            session: {
                "frames_sha256": "a" * 64,
                "source_npz_sha256": "b" * 64,
                "masked_regions": ["frame_index_strip", "input_overlay"],
                "supervision_equal_to_features": True,
            }
            for session in ("train-session", "validation-session")
        },
    }
    cache_manifest_path = cache / "cache_manifest.json"
    _write_json(cache_manifest_path, cache_manifest)
    receipt = {
        "schema_version": "madeleine.oracle-window-pixel-crops-complete.v1",
        "status": "complete",
        "published_output": str(cache),
        "base_config_sha256": "base-config",
        "base_dataset_manifest_sha256": _sha256_file(base_manifest),
        "feature_receipt_sha256": "feature-receipt",
        "cache_manifest_sha256": _sha256_file(cache_manifest_path),
        "cache": {
            "cache_manifest.json": {
                "bytes": cache_manifest_path.stat().st_size,
                "sha256": _sha256_file(cache_manifest_path),
            },
            "train.npz": {
                "bytes": train_evidence["bytes"],
                "sha256": train_evidence["file_sha256"],
            },
            "validation.npz": {
                "bytes": validation_evidence["bytes"],
                "sha256": validation_evidence["file_sha256"],
            },
        },
        "checks": {"all": True},
    }
    receipt["content_sha256"] = _canonical_json_sha256(receipt)
    receipt_path = tmp_path / "cache_complete.json"
    _write_json(receipt_path, receipt)
    config = {
        "dataset": {
            "pixel_cache_receipt_sha256": _sha256_file(receipt_path),
            "pixel_cache_content_sha256": receipt["content_sha256"],
            "source_build_manifest_sha256": "source-build",
        }
    }

    validated = validate_cache_receipt(
        cache_root=cache,
        receipt_path=receipt_path,
        base_manifest_path=base_manifest,
        config=config,
    )
    assert validated["split_evidence"]["train"] == train_evidence

    receipt["checks"]["all"] = False
    _write_json(receipt_path, receipt)
    config["dataset"]["pixel_cache_receipt_sha256"] = _sha256_file(receipt_path)
    with pytest.raises(ValueError, match="content hash"):
        validate_cache_receipt(
            cache_root=cache,
            receipt_path=receipt_path,
            base_manifest_path=base_manifest,
            config=config,
        )


def _run_receipt_fixture(tmp_path: Path) -> tuple[dict, dict[str, Path], dict[str, Path]]:
    paths = {
        name: tmp_path / name
        for name in (
            "config.json",
            "cache.json",
            "baseline.npz",
            "baseline_audit.json",
            "predictions.npz",
            "training.json",
        )
    }
    for path in paths.values():
        path.write_bytes(path.name.encode("ascii"))
    checkpoints = {
        "ordered_pair": tmp_path / "ordered.pt",
        "symmetric_pair": tmp_path / "symmetric.pt",
    }
    for arm, path in checkpoints.items():
        path.write_bytes(arm.encode("ascii"))
    implementation = {
        "git_head_at_execution": "head",
        "git_head_at_freeze": "head",
        "relevant_file_sha256": {"f": "h"},
        "authority": "exact relevant working bytes; no commit created for this study",
    }
    receipt = {
        "schema_version": "madeleine.oracle-window-differential-run.v1",
        "status": "predictions_complete_unscored",
        "config_path": str(paths["config.json"]),
        "config_sha256": _sha256_file(paths["config.json"]),
        "pixel_cache_receipt_path": str(paths["cache.json"]),
        "pixel_cache_receipt_sha256": _sha256_file(paths["cache.json"]),
        "baseline_sidecar_path": str(paths["baseline.npz"]),
        "baseline_sidecar_sha256": _sha256_file(paths["baseline.npz"]),
        "baseline_audit_path": str(paths["baseline_audit.json"]),
        "baseline_audit_sha256": _sha256_file(paths["baseline_audit.json"]),
        "checkpoints": {
            arm: {
                "path": str(path),
                "sha256": _sha256_file(path),
                "model_state_sha256": arm[0] * 64,
            }
            for arm, path in checkpoints.items()
        },
        "initial_state_sha256": "i" * 64,
        "training_log_sha256": _sha256_file(paths["training.json"]),
        "prediction_sidecar_sha256": _sha256_file(paths["predictions.npz"]),
        "seed": 0,
        "epochs": 20,
        "configured_epochs": 20,
        "final_weights_only": True,
        "validation_used_for_training_or_selection": False,
        "matched_initialization": True,
        "matched_batch_order": True,
        "train_examples": 4554,
        "validation_examples": 1150,
        "device": "mps",
        "implementation": implementation,
    }
    return receipt, checkpoints, paths


def test_run_receipt_requires_exact_inventory_hashes_flags_and_mps(tmp_path: Path) -> None:
    receipt, checkpoints, paths = _run_receipt_fixture(tmp_path)
    receipt_path = tmp_path / "run.json"
    _write_json(receipt_path, receipt)
    implementation = receipt["implementation"]
    validated = validate_run_receipt(
        receipt_path=receipt_path,
        config_path=paths["config.json"],
        cache_receipt_path=paths["cache.json"],
        baseline_sidecar_path=paths["baseline.npz"],
        baseline_audit_path=paths["baseline_audit.json"],
        predictions_path=paths["predictions.npz"],
        training_log_path=paths["training.json"],
        checkpoint_paths=checkpoints,
        config={},
        implementation=implementation,
    )
    assert validated["device"] == "mps"

    receipt["device"] = "cpu"
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="device"):
        validate_run_receipt(
            receipt_path=receipt_path,
            config_path=paths["config.json"],
            cache_receipt_path=paths["cache.json"],
            baseline_sidecar_path=paths["baseline.npz"],
            baseline_audit_path=paths["baseline_audit.json"],
            predictions_path=paths["predictions.npz"],
            training_log_path=paths["training.json"],
            checkpoint_paths=checkpoints,
            config={},
            implementation=implementation,
        )

    receipt["device"] = "mps"
    receipt["extra"] = True
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="inventory"):
        validate_run_receipt(
            receipt_path=receipt_path,
            config_path=paths["config.json"],
            cache_receipt_path=paths["cache.json"],
            baseline_sidecar_path=paths["baseline.npz"],
            baseline_audit_path=paths["baseline_audit.json"],
            predictions_path=paths["predictions.npz"],
            training_log_path=paths["training.json"],
            checkpoint_paths=checkpoints,
            config={},
            implementation=implementation,
        )


def test_training_log_requires_exact_rows_endpoint_and_finite_losses(tmp_path: Path) -> None:
    rows = [{"epoch": float(index), "loss": 1.0 / index} for index in range(1, 21)]
    value = {
        "ordered_pair": rows,
        "symmetric_pair": rows,
        "fixed_final_epoch": 20,
        "configured_final_epoch": 20,
        "validation_used_for_training_or_selection": False,
        "matched_batch_order": True,
    }
    path = tmp_path / "training.json"
    _write_json(path, value)
    assert validate_training_log(path, epochs=20) == value

    value["symmetric_pair"][3]["loss"] = -1.0
    _write_json(path, value)
    with pytest.raises(ValueError, match="training loss"):
        validate_training_log(path, epochs=20)


def test_checkpoint_requires_exact_schema_dtype_and_state_hash(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = DifferentialCandidateModel()
    path = tmp_path / "ordered.pt"
    payload = {
        "schema_version": "madeleine.oracle-window-differential-checkpoint.v1",
        "config_sha256": "config",
        "pixel_cache_receipt_sha256": "cache",
        "baseline_sidecar_sha256": "baseline",
        "seed": 0,
        "epochs": 20,
        "initial_state_sha256": "initial",
        "arm": "ordered_pair",
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    run = {
        "config_sha256": "config",
        "pixel_cache_receipt_sha256": "cache",
        "baseline_sidecar_sha256": "baseline",
        "initial_state_sha256": "initial",
        "checkpoints": {
            "ordered_pair": {
                "sha256": _sha256_file(path),
                "model_state_sha256": _state_dict_sha256(model),
            }
        },
    }
    loaded, evidence = load_final_checkpoint(
        path=path,
        arm="ordered_pair",
        config={"model": {"embedding_dim": 64}},
        run_receipt=run,
    )
    assert evidence["state_dict_sha256"] == _state_dict_sha256(loaded)

    first = next(iter(payload["model_state_dict"]))
    payload["model_state_dict"][first] = payload["model_state_dict"][first].double()
    torch.save(payload, path)
    run["checkpoints"]["ordered_pair"]["sha256"] = _sha256_file(path)
    with pytest.raises(ValueError, match="tensor schema"):
        load_final_checkpoint(
            path=path,
            arm="ordered_pair",
            config={"model": {"embedding_dim": 64}},
            run_receipt=run,
        )


def _score_fixture(bindings: dict) -> dict:
    bootstrap = {"replicates_requested": 5000, "replicates_valid": 5000}
    gate = {"passed": False}
    return {
        "schema_version": "madeleine.oracle-window-differential-score.v1",
        "status": "complete",
        "study_id": "fixture",
        "scope": "fixture",
        "config_sha256": "",
        "prediction_sidecar_sha256": "",
        "pixel_cache_receipt_sha256": "",
        "base_dataset_manifest_sha256": "",
        "bindings": bindings,
        "support": {},
        "chance": {},
        "arms": {},
        "primary_comparison": {"paired_block_bootstrap": bootstrap},
        "attribution_comparisons": {
            "symmetric_pair_minus_frozen_feature": {
                "paired_block_bootstrap": bootstrap
            },
            "ordered_pair_minus_symmetric_pair": {
                "paired_block_bootstrap": bootstrap
            },
        },
        "decision_gate": {
            "ordered_pair_vs_frozen_feature": gate,
            "symmetric_pair_vs_frozen_feature": gate,
            "ordered_pair_vs_symmetric_pair": gate,
            "original_phase_2_decision_remains_rejected": True,
            "passed_primary_pixel_rescue": False,
            "passed_differential_attribution": False,
            "decision": "no_bounded_pixel_rescue_no_phase_2",
        },
    }


def test_score_report_requires_exact_reproduction_bootstrap_and_hierarchy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.validate_oracle_window_differential_run as validator

    paths = {
        name: tmp_path / name
        for name in ("config.json", "predictions.npz", "cache.json", "manifest.json")
    }
    for path in paths.values():
        path.write_bytes(path.name.encode("ascii"))
    bindings = {"run_receipt_sha256": "run"}
    report = _score_fixture(bindings)
    report["config_sha256"] = _sha256_file(paths["config.json"])
    report["prediction_sidecar_sha256"] = _sha256_file(paths["predictions.npz"])
    report["pixel_cache_receipt_sha256"] = _sha256_file(paths["cache.json"])
    report["base_dataset_manifest_sha256"] = _sha256_file(paths["manifest.json"])
    report_path = tmp_path / "report.json"
    _write_json(report_path, report)
    monkeypatch.setattr(validator, "score_followup", lambda **kwargs: report)
    validated = validate_score_report(
        report_path=report_path,
        predictions_path=paths["predictions.npz"],
        config_path=paths["config.json"],
        cache_receipt_path=paths["cache.json"],
        base_manifest_path=paths["manifest.json"],
        arrays={},
        config={"study_id": "fixture"},
        base_manifest={},
        bindings=bindings,
    )
    assert validated["decision_gate"]["decision"] == "no_bounded_pixel_rescue_no_phase_2"

    report["primary_comparison"]["paired_block_bootstrap"]["replicates_valid"] = 4999
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="bootstrap support"):
        validate_score_report(
            report_path=report_path,
            predictions_path=paths["predictions.npz"],
            config_path=paths["config.json"],
            cache_receipt_path=paths["cache.json"],
            base_manifest_path=paths["manifest.json"],
            arrays={},
            config={"study_id": "fixture"},
            base_manifest={},
            bindings=bindings,
        )


def test_completion_marker_closes_every_direct_and_transitive_binding(
    tmp_path: Path,
) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "report.json",
            "config.json",
            "predictions.npz",
            "run.json",
            "ordered.pt",
            "symmetric.pt",
            "cache.json",
        )
    }
    for path in paths.values():
        path.write_bytes(path.name.encode("ascii"))
    report = {
        "study_id": "fixture",
        "pixel_cache_receipt_sha256": _sha256_file(paths["cache.json"]),
        "bindings": {
            "run_receipt_sha256": _sha256_file(paths["run.json"]),
            "checkpoints": {
                "ordered_pair": {"sha256": _sha256_file(paths["ordered.pt"])},
                "symmetric_pair": {"sha256": _sha256_file(paths["symmetric.pt"])},
            },
        },
        "decision_gate": {"decision": "reject"},
    }
    _write_json(paths["report.json"], report)
    marker = {
        "schema_version": "madeleine.oracle-window-differential-complete.v1",
        "status": "complete",
        "study_id": "fixture",
        "decision": "reject",
        "report": {
            "path": str(paths["report.json"]),
            "sha256": _sha256_file(paths["report.json"]),
        },
        "config": {
            "path": str(paths["config.json"]),
            "sha256": _sha256_file(paths["config.json"]),
        },
        "predictions": {
            "path": str(paths["predictions.npz"]),
            "sha256": _sha256_file(paths["predictions.npz"]),
        },
        "run_receipt": {
            "path": str(paths["run.json"]),
            "sha256": _sha256_file(paths["run.json"]),
        },
        "checkpoints": {
            "ordered_pair": {
                "path": str(paths["ordered.pt"]),
                "sha256": _sha256_file(paths["ordered.pt"]),
            },
            "symmetric_pair": {
                "path": str(paths["symmetric.pt"]),
                "sha256": _sha256_file(paths["symmetric.pt"]),
            },
        },
        "pixel_cache_receipt": {
            "path": str(paths["cache.json"]),
            "sha256": _sha256_file(paths["cache.json"]),
        },
    }
    marker["content_sha256"] = _canonical_json_sha256(marker)
    marker_path = tmp_path / "marker.json"
    _write_json(marker_path, marker)
    validated = validate_completion_marker(
        marker_path=marker_path,
        report_path=paths["report.json"],
        config_path=paths["config.json"],
        predictions_path=paths["predictions.npz"],
        run_receipt_path=paths["run.json"],
        checkpoint_paths={
            "ordered_pair": paths["ordered.pt"],
            "symmetric_pair": paths["symmetric.pt"],
        },
        cache_receipt_path=paths["cache.json"],
        report=report,
    )
    assert validated["content_sha256"] == marker["content_sha256"]

    paths["ordered.pt"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint hash"):
        validate_completion_marker(
            marker_path=marker_path,
            report_path=paths["report.json"],
            config_path=paths["config.json"],
            predictions_path=paths["predictions.npz"],
            run_receipt_path=paths["run.json"],
            checkpoint_paths={
                "ordered_pair": paths["ordered.pt"],
                "symmetric_pair": paths["symmetric.pt"],
            },
            cache_receipt_path=paths["cache.json"],
            report=report,
        )


def test_audit_receipt_is_canonical_atomic_and_non_overwriting(tmp_path: Path) -> None:
    receipt = _finalize_receipt(
        {"schema_version": "fixture", "status": "complete", "value": 1}
    )
    without_hash = dict(receipt)
    content_sha = without_hash.pop("content_sha256")
    assert content_sha == _canonical_json_sha256(without_hash)
    output = tmp_path / "audit.json"
    _write_json_exclusive_atomic(output, receipt)
    assert _json(output, "audit") == receipt
    with pytest.raises(FileExistsError, match="overwrite"):
        _write_json_exclusive_atomic(output, receipt)


def test_metadata_fixture_tracks_validator_inventory() -> None:
    assert set(_metadata()) == set(METADATA_NAMES)
