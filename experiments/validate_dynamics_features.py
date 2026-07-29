#!/usr/bin/env python3
"""Independently validate a final-EMA C/D feature export.

The validator reconstructs exact membership from the content-bound explicit
inventory, verifies the terminal streaming checkpoint and completion chain,
and reconciles every feature-only NPZ with its sidecar.  ``--deep-shards``
loads all feature values to check finiteness and canonical array hashes.
``--deep-references`` additionally re-hashes every reference shard and
compares only its engine/input/session metadata; it never reads ``keys`` or
old ``features`` members.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from badeline.dynamics_pretraining import REPRESENTATION_DIM
from experiments.export_dynamics_features import (
    CHECKPOINT_SCHEMA,
    COMPLETION_SCHEMA,
    EXPORT_ONLY_ROLE,
    INVENTORY_SCHEMA,
    MANIFEST_SCHEMA,
    PRODUCTION_COUNTS,
    SHARD_SIDECAR_SCHEMA,
    TRAIN_ROLE,
    CheckpointContract,
    ExpectedCounts,
    Inventory,
    SessionSpec,
    _expected_previous_index_sha256,
    _npz_headers,
    _reject_forbidden_identity,
    array_sha256,
    instantiate_final_ema_target,
    load_checkpoint_contract,
    load_inventory,
    load_reference_metadata,
    sha256_file,
)


REPORT_SCHEMA = "madeleine.dynamics-feature-validation.v1"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_sidecar_bindings(
    session: SessionSpec,
    *,
    inventory: Inventory,
    checkpoint: CheckpointContract,
) -> dict[str, Any]:
    return {
        "schema_version": SHARD_SIDECAR_SCHEMA,
        "session_id": session.session_id,
        "video_id": session.video_id,
        "role": session.role,
        "arm": checkpoint.arm,
        "checkpoint_sha256": checkpoint.sha256,
        "inventory_sha256": inventory.sha256,
        "reference_shard": str(session.reference_shard),
        "reference_shard_sha256": session.reference_shard_sha256,
        "source_frame_range": [session.start_frame, session.end_frame],
        "frames": session.frames,
        "feature_dim": REPRESENTATION_DIM,
        "normalization": "none_raw_target_encoder_output",
        "supervision_phase": "label_free_feature_export_no_keys_member_read_or_written",
        "npz": f"{session.session_id}.npz",
    }


def _validate_shard(
    session: SessionSpec,
    *,
    inventory: Inventory,
    checkpoint: CheckpointContract,
    out_dir: Path,
    manifest_row: dict[str, Any],
    deep_shards: bool,
    deep_references: bool,
) -> dict[str, Any]:
    shard = out_dir / f"{session.session_id}.npz"
    sidecar_path = out_dir / f"{session.session_id}.json"
    sidecar = _read_json(sidecar_path)
    if not isinstance(sidecar, dict):
        raise ValueError(f"{sidecar_path}: root must be an object")
    if sidecar != manifest_row:
        raise ValueError(f"{session.session_id}: manifest row differs from sidecar")
    for key, expected in _expected_sidecar_bindings(
        session, inventory=inventory, checkpoint=checkpoint
    ).items():
        if sidecar.get(key) != expected:
            raise ValueError(f"{session.session_id}: sidecar {key} mismatch")
    if sidecar.get("source_video_sha256") is None:
        raise ValueError(f"{session.session_id}: missing source video binding")
    video = next(item for item in inventory.videos if item.video_id == session.video_id)
    if sidecar["source_video_sha256"] != video.video_sha256:
        raise ValueError(f"{session.session_id}: source video SHA binding mismatch")
    if sidecar.get("decoder_mode") != video.decoder_mode:
        raise ValueError(f"{session.session_id}: decoder mode mismatch")
    expected_format = (
        f"dynamics_{checkpoint.arm.lower()}_final_ema_raw_avgpool_float16_v1"
    )
    if sidecar.get("feature_format") != expected_format:
        raise ValueError(f"{session.session_id}: feature format mismatch")
    if not isinstance(sidecar.get("imputed_tail_frames"), int):
        raise ValueError(f"{session.session_id}: invalid imputation count")
    if not 0 <= sidecar["imputed_tail_frames"] <= 3:
        raise ValueError(f"{session.session_id}: excessive imputed tail")
    if video.decoder_mode == "opencv_native_60hz" and sidecar["imputed_tail_frames"]:
        raise ValueError(f"{session.session_id}: native decode claims imputation")
    if checkpoint.arm == "D":
        policy = "previous_equals_current_at_explicit_session_start_then_prior_frame"
        if sidecar.get("D_boundary_policy") != policy:
            raise ValueError(f"{session.session_id}: D boundary policy mismatch")
    elif sidecar.get("D_boundary_policy") is not None:
        raise ValueError(f"{session.session_id}: C export has a D policy")

    if not shard.is_file():
        raise FileNotFoundError(f"missing output feature shard: {shard}")
    if sidecar.get("npz_sha256") != sha256_file(shard):
        raise ValueError(f"{session.session_id}: output NPZ SHA mismatch")
    headers = _npz_headers(shard)
    expected_headers = {
        "features": ((session.frames, REPRESENTATION_DIM), np.dtype(np.float16)),
        "engine_frame_idx": ((session.frames,), np.dtype(np.int64)),
        "input_active": ((session.frames,), np.dtype(np.uint8)),
    }
    for key, expected in expected_headers.items():
        if headers.get(key) != expected:
            raise ValueError(f"{session.session_id}: {key} header mismatch")
    sid_shape, sid_dtype = headers["session_id"]
    if sid_shape != () or sid_dtype.kind not in {"U", "S"}:
        raise ValueError(f"{session.session_id}: session_id header mismatch")
    arrays = sidecar.get("arrays")
    if not isinstance(arrays, dict) or set(arrays) != {
        "features_sha256", "engine_frame_idx_sha256", "input_active_sha256"
    }:
        raise ValueError(f"{session.session_id}: array hash map differs")

    expected_engine = np.arange(
        session.start_frame, session.end_frame, dtype=np.int64
    )
    expected_active = np.ones(session.frames, dtype=np.uint8)
    if arrays["engine_frame_idx_sha256"] != array_sha256(expected_engine):
        raise ValueError(f"{session.session_id}: engine hash mismatch")
    if arrays["input_active_sha256"] != array_sha256(expected_active):
        raise ValueError(f"{session.session_id}: activity hash mismatch")
    if checkpoint.arm == "D":
        if sidecar.get("previous_engine_frame_idx_sha256") != (
            _expected_previous_index_sha256(expected_engine)
        ):
            raise ValueError(f"{session.session_id}: D previous-index policy mismatch")
    elif sidecar.get("previous_engine_frame_idx_sha256") is not None:
        raise ValueError(f"{session.session_id}: unexpected C previous-index hash")

    feature_min: float | None = None
    feature_max: float | None = None
    if deep_shards:
        with np.load(shard, allow_pickle=False) as archive:
            if set(archive.files) != {
                "features", "engine_frame_idx", "input_active", "session_id"
            }:
                raise ValueError(f"{session.session_id}: deep member set differs")
            features = np.asarray(archive["features"])
            engine = np.asarray(archive["engine_frame_idx"])
            active = np.asarray(archive["input_active"])
            stored_session = archive["session_id"]
        if not np.array_equal(engine, expected_engine):
            raise ValueError(f"{session.session_id}: deep engine values differ")
        if not np.array_equal(active, expected_active):
            raise ValueError(f"{session.session_id}: deep activity values differ")
        if str(stored_session.reshape(()).item()) != session.session_id:
            raise ValueError(f"{session.session_id}: deep session identity differs")
        if not np.isfinite(features).all():
            raise ValueError(f"{session.session_id}: non-finite feature values")
        if array_sha256(features) != arrays["features_sha256"]:
            raise ValueError(f"{session.session_id}: feature array hash mismatch")
        if array_sha256(engine) != arrays["engine_frame_idx_sha256"]:
            raise ValueError(f"{session.session_id}: engine array hash mismatch")
        if array_sha256(active) != arrays["input_active_sha256"]:
            raise ValueError(f"{session.session_id}: activity array hash mismatch")
        feature_min = float(features.min(initial=np.inf))
        feature_max = float(features.max(initial=-np.inf))
    if deep_references:
        reference_engine, reference_active = load_reference_metadata(session)
        if not np.array_equal(reference_engine, expected_engine):
            raise ValueError(f"{session.session_id}: reference engine values differ")
        if not np.array_equal(reference_active, expected_active):
            raise ValueError(f"{session.session_id}: reference activity values differ")
    return {
        "session_id": session.session_id,
        "frames": session.frames,
        "npz_sha256": sidecar["npz_sha256"],
        "feature_min": feature_min,
        "feature_max": feature_max,
    }


def validate_export(
    *,
    inventory: Inventory,
    checkpoint: CheckpointContract,
    out_dir: Path,
    expected_counts: ExpectedCounts = PRODUCTION_COUNTS,
    deep_shards: bool,
    deep_references: bool,
    strict_checkpoint_state: bool = True,
) -> dict[str, Any]:
    """Return a complete report; validation failures are recorded, not hidden."""

    out_dir = Path(out_dir).resolve()
    for component in out_dir.parts:
        _reject_forbidden_identity(component, name="feature directory")
    failures: list[str] = []
    checked: list[dict[str, Any]] = []
    try:
        if inventory.provenance is not None:
            expected_terminal = {
                "schema_version": CHECKPOINT_SCHEMA,
                "sha256": checkpoint.sha256,
                "arm": checkpoint.arm,
                "completed_steps": checkpoint.completed_steps,
            }
            if inventory.provenance.get("terminal_checkpoint") != expected_terminal:
                raise ValueError("inventory terminal-checkpoint provenance mismatch")
            if inventory.provenance.get(
                "y4n_hashed_after_terminal_checkpoint_validation"
            ) is not True:
                raise ValueError("inventory lacks post-terminal y4n hash proof")
        if strict_checkpoint_state:
            # CPU construction and strict state loading prove model-state
            # compatibility without creating or using a CUDA context.
            instantiate_final_ema_target(checkpoint, device=np_device_cpu())
        manifest_path = out_dir / "feature_export_manifest.json"
        completion_path = out_dir / "feature_export_complete.json"
        manifest = _read_json(manifest_path)
        completion = _read_json(completion_path)
        if not isinstance(manifest, dict) or not isinstance(completion, dict):
            raise ValueError("manifest/completion roots must be objects")
        if manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise ValueError("manifest schema mismatch")
        if completion.get("schema_version") != COMPLETION_SCHEMA:
            raise ValueError("completion schema mismatch")
        if completion.get("manifest") != manifest_path.name:
            raise ValueError("completion manifest filename mismatch")
        if completion.get("manifest_sha256") != sha256_file(manifest_path):
            raise ValueError("completion manifest SHA mismatch")
        for key, expected in {
            "inventory_sha256": inventory.sha256,
            "checkpoint_sha256": checkpoint.sha256,
            "arm": checkpoint.arm,
        }.items():
            if completion.get(key) != expected:
                raise ValueError(f"completion {key} mismatch")
        if manifest.get("arm") != checkpoint.arm:
            raise ValueError("manifest arm mismatch")
        if manifest.get("inventory") != {
            "path": str(inventory.path), "sha256": inventory.sha256
        }:
            raise ValueError("manifest inventory binding mismatch")
        checkpoint_row = manifest.get("checkpoint")
        if not isinstance(checkpoint_row, dict):
            raise ValueError("manifest checkpoint row missing")
        expected_checkpoint = {
            "path": str(checkpoint.path),
            "sha256": checkpoint.sha256,
            "schema_version": CHECKPOINT_SCHEMA,
            "completed_steps": checkpoint.completed_steps,
            "horizons": list(checkpoint.horizons),
            "selection": "final_weights_only",
            "encoder_state": "final_ema_target_only",
        }
        if checkpoint_row != expected_checkpoint:
            raise ValueError("manifest checkpoint binding mismatch")
        expected_format = (
            f"dynamics_{checkpoint.arm.lower()}_final_ema_raw_avgpool_float16_v1"
        )
        for key, expected in {
            "feature_format": expected_format,
            "feature_dim": REPRESENTATION_DIM,
            "dtype": "float16",
            "normalization": "none_raw_target_encoder_output",
            "supervision_phase": "label_free_feature_export_no_keys_member_read_or_written",
            "y4n_policy": "downstream_export_only_after_terminal_ssl_never_pretraining",
        }.items():
            if manifest.get(key) != expected:
                raise ValueError(f"manifest {key} mismatch")
        expected_policy = (
            "previous_equals_current_at_explicit_session_start_then_prior_frame"
            if checkpoint.arm == "D"
            else None
        )
        if manifest.get("D_boundary_policy") != expected_policy:
            raise ValueError("manifest D boundary policy mismatch")
        expected_count_row = {
            "videos": expected_counts.videos,
            "sessions": expected_counts.sessions,
            "frames": expected_counts.frames,
            "train_videos": expected_counts.train_videos,
            "downstream_export_only_videos": 1,
        }
        if manifest.get("counts") != expected_count_row:
            raise ValueError("manifest count row mismatch")
        if completion.get("counts") != expected_count_row:
            raise ValueError("completion count row mismatch")
        rows = manifest.get("sessions")
        if not isinstance(rows, list) or len(rows) != len(inventory.sessions):
            raise ValueError("manifest session rows differ in count")
        rows_by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("session_id"), str):
                raise ValueError("manifest contains invalid session row")
            if row["session_id"] in rows_by_id:
                raise ValueError("manifest contains duplicate session row")
            rows_by_id[row["session_id"]] = row
        if set(rows_by_id) != {item.session_id for item in inventory.sessions}:
            raise ValueError("manifest session membership differs from inventory")
        for session in inventory.sessions:
            checked.append(
                _validate_shard(
                    session,
                    inventory=inventory,
                    checkpoint=checkpoint,
                    out_dir=out_dir,
                    manifest_row=rows_by_id[session.session_id],
                    deep_shards=deep_shards,
                    deep_references=deep_references,
                )
            )
        expected_files = {
            "feature_export_manifest.json",
            "feature_export_complete.json",
        }
        expected_files.update(f"{item.session_id}.npz" for item in inventory.sessions)
        expected_files.update(f"{item.session_id}.json" for item in inventory.sessions)
        actual_entries = {item.name for item in out_dir.iterdir()}
        if actual_entries != expected_files:
            missing = sorted(expected_files - actual_entries)
            extra = sorted(actual_entries - expected_files)
            raise ValueError(f"output file set differs: missing={missing} extra={extra}")
        temporary = sorted(str(item) for item in out_dir.rglob("*.tmp.*"))
        if temporary:
            raise ValueError(f"temporary outputs remain: {temporary[0]}")
    except Exception as error:  # report the exact failure and exit nonzero in CLI
        failures.append(f"{type(error).__name__}: {error}")
    return {
        "schema_version": REPORT_SCHEMA,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "ok": not failures,
        "inventory": {
            "schema_version": INVENTORY_SCHEMA,
            "path": str(inventory.path),
            "sha256": inventory.sha256,
        },
        "checkpoint": {
            "schema_version": CHECKPOINT_SCHEMA,
            "path": str(checkpoint.path),
            "sha256": checkpoint.sha256,
            "arm": checkpoint.arm,
            "completed_steps": checkpoint.completed_steps,
        },
        "out_dir": str(out_dir.resolve()),
        "deep_shards": bool(deep_shards),
        "deep_references": bool(deep_references),
        "counts": {
            "expected_videos": expected_counts.videos,
            "expected_sessions": expected_counts.sessions,
            "expected_frames": expected_counts.frames,
            "checked_sessions": len(checked),
            "checked_frames": sum(item["frames"] for item in checked),
        },
        "failures": failures,
    }


def np_device_cpu():
    """Import torch lazily only when strict model-state validation is requested."""

    import torch

    return torch.device("cpu")


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    if temporary.exists():
        raise FileExistsError(f"refusing existing temporary report: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--arm", choices=("C", "D"), required=True)
    parser.add_argument("--expected-completed-steps", type=int, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--deep-shards", action="store_true")
    parser.add_argument("--deep-references", action="store_true")
    args = parser.parse_args()
    try:
        if args.report.resolve().is_relative_to(args.features.resolve()):
            raise ValueError("validation report must be outside the feature directory")
        inventory = load_inventory(args.inventory, args.inventory_sha256)
        checkpoint = load_checkpoint_contract(
            args.checkpoint,
            args.checkpoint_sha256,
            expected_arm=args.arm,
            expected_completed_steps=args.expected_completed_steps,
        )
        report = validate_export(
            inventory=inventory,
            checkpoint=checkpoint,
            out_dir=args.features,
            deep_shards=args.deep_shards,
            deep_references=args.deep_references,
        )
    except Exception as error:
        report = {
            "schema_version": REPORT_SCHEMA,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "failures": [f"{type(error).__name__}: {error}"],
        }
    _write_report(args.report, report)
    print(json.dumps(report, indent=2))
    if not report.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
