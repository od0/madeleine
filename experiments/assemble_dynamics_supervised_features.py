#!/usr/bin/env python3
"""Assemble standard supervised Badeline shards after terminal C/D export.

This is the only bridge that opens ``keys``.  Before it does so, it requires a
passing deep validation of the complete final-EMA feature-only export,
including deep reference metadata checks and exact checkpoint/inventory
bindings.  It then joins each feature-only shard to keys from its explicit
reference shard, verifies identical session/engine/activity alignment, and
writes the standard Badeline NPZ member set atomically.

The reference all-session inventory is partitioned into explicit downstream
lists: all 1,554 sessions for audit, 1,538 non-y4n sessions for training, all
16 y4n sessions for endpoint monitoring, and r008--r015 as the sole reported
later-eight surface.  The full-corpus manifest is preserved structurally but
receives a new visual-representation format and content-bound provenance;
output shard hashes are rebuilt, never copied from the ImageNet reference
corpus.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from badeline.dynamics_pretraining import REPRESENTATION_DIM
from data.schema import KEY_ORDER
from experiments.export_dynamics_features import (
    PRODUCTION_COUNTS,
    CheckpointContract,
    ExpectedCounts,
    Inventory,
    SessionSpec,
    _reject_forbidden_identity,
    array_sha256,
    canonical_json_sha256,
    load_checkpoint_contract,
    load_inventory,
    sha256_file,
)
from experiments.validate_dynamics_features import validate_export


ASSEMBLY_SCHEMA = "madeleine.dynamics-supervised-feature-assembly.v1"
COMPLETION_SCHEMA = "madeleine.dynamics-supervised-feature-assembly-complete.v1"
VALIDATION_SCHEMA = "madeleine.dynamics-supervised-feature-assembly-validation.v1"
HOLDOUT_VIDEO_ID = "y4nQHqYSObI"
Y4N_LATER8_SESSION_IDS = tuple(
    f"{HOLDOUT_VIDEO_ID}__r{index:03d}" for index in range(8, 16)
)


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp.json")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing existing publication artifact: {path}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _copy_bytes_exact(source: Path, destination: Path) -> None:
    value = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != value:
            raise ValueError(f"existing copied artifact differs: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary copied artifact remains: {temporary}")
    temporary.write_bytes(value)
    temporary.replace(destination)


def _write_lines_atomic(path: Path, values: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    contents = "".join(f"{value}\n" for value in values)
    if path.exists():
        if path.read_text(encoding="utf-8") != contents:
            raise ValueError(f"existing session-list artifact differs: {path}")
        return
    if temporary.exists():
        raise FileExistsError(f"temporary session-list artifact remains: {temporary}")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def _canonical_split_lists(
    session_ids: Sequence[str],
    unflagged_ids: Sequence[str],
    *,
    production: bool,
) -> dict[str, list[str]]:
    all_ids = sorted(str(value) for value in session_ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("full-session inventory contains duplicates")
    prefix = f"{HOLDOUT_VIDEO_ID}__"
    train_ids = [value for value in all_ids if not value.startswith(prefix)]
    val_ids = [value for value in all_ids if value.startswith(prefix)]
    unflagged_train = sorted(
        value for value in unflagged_ids if not value.startswith(prefix)
    )
    later8 = [value for value in Y4N_LATER8_SESSION_IDS if value in set(val_ids)]
    if set(train_ids) & set(val_ids):
        raise AssertionError("holdout split construction overlapped")
    if set(train_ids) | set(val_ids) != set(all_ids):
        raise AssertionError("holdout split construction lost sessions")
    if len(unflagged_train) != len(set(unflagged_train)) or not set(
        unflagged_train
    ).issubset(train_ids):
        raise ValueError("unflagged training list is not a unique train subset")
    if production:
        if len(all_ids) != 1_554 or len(train_ids) != 1_538 or len(val_ids) != 16:
            raise ValueError("production all/train/y4n session counts changed")
        if later8 != list(Y4N_LATER8_SESSION_IDS):
            raise ValueError("production y4n later-eight membership changed")
    return {
        "all_sessions.txt": all_ids,
        "train_sessions.txt": train_ids,
        "unflagged_sessions.txt": unflagged_train,
        "val_sessions.txt": val_ids,
        "y4n_later8_sessions.txt": later8,
    }


def _require_terminal_validation(
    report: Mapping[str, Any],
    *,
    inventory: Inventory,
    checkpoint: CheckpointContract,
    expected_counts: ExpectedCounts,
) -> None:
    if report.get("ok") is not True:
        raise ValueError("feature-only export lacks passing validation")
    if report.get("deep_shards") is not True or report.get("deep_references") is not True:
        raise ValueError("supervised assembly requires deep shard/reference validation")
    inventory_row = report.get("inventory")
    checkpoint_row = report.get("checkpoint")
    if not isinstance(inventory_row, dict) or inventory_row.get("sha256") != inventory.sha256:
        raise ValueError("terminal validation inventory binding differs")
    expected_checkpoint = {
        "sha256": checkpoint.sha256,
        "arm": checkpoint.arm,
        "completed_steps": checkpoint.completed_steps,
    }
    if not isinstance(checkpoint_row, dict):
        raise ValueError("terminal validation lacks checkpoint binding")
    for key, expected in expected_checkpoint.items():
        if checkpoint_row.get(key) != expected:
            raise ValueError(f"terminal validation checkpoint {key} differs")
    counts = report.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("terminal validation lacks counts")
    if counts.get("checked_sessions") != expected_counts.sessions:
        raise ValueError("terminal validation did not check every session")
    if counts.get("checked_frames") != expected_counts.frames:
        raise ValueError("terminal validation did not check every frame")


def _load_feature_only(
    path: Path, session: SessionSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "features", "engine_frame_idx", "input_active", "session_id"
        }:
            raise ValueError(f"{path}: feature-only member set differs")
        features = np.asarray(archive["features"])
        engine = np.asarray(archive["engine_frame_idx"])
        active = np.asarray(archive["input_active"])
        stored = archive["session_id"]
    if features.dtype != np.float16 or features.shape != (
        session.frames,
        REPRESENTATION_DIM,
    ):
        raise ValueError(f"{path}: feature tensor schema differs")
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: feature tensor is non-finite")
    if engine.dtype != np.int64 or engine.shape != (session.frames,):
        raise ValueError(f"{path}: engine metadata schema differs")
    if active.dtype != np.uint8 or active.shape != (session.frames,):
        raise ValueError(f"{path}: activity metadata schema differs")
    if str(stored.reshape(()).item()) != session.session_id:
        raise ValueError(f"{path}: feature-only session identity differs")
    return features, engine, active


def _load_supervision(
    session: SessionSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = session.reference_shard
    if sha256_file(path) != session.reference_shard_sha256:
        raise ValueError(f"{session.session_id}: reference changed before key join")
    with np.load(path, allow_pickle=False) as archive:
        required = {"keys", "engine_frame_idx", "input_active", "session_id"}
        if not required.issubset(archive.files):
            raise ValueError(f"{path}: reference supervision members missing")
        # This is intentionally the first label access in the C/D pipeline.
        keys = np.asarray(archive["keys"])
        engine = np.asarray(archive["engine_frame_idx"])
        active = np.asarray(archive["input_active"])
        stored = archive["session_id"]
    if keys.dtype != np.uint8 or keys.shape != (session.frames, len(KEY_ORDER)):
        raise ValueError(f"{path}: keys schema differs")
    if np.any((keys != 0) & (keys != 1)):
        raise ValueError(f"{path}: keys are not binary")
    if engine.dtype != np.int64 or engine.shape != (session.frames,):
        raise ValueError(f"{path}: reference engine schema differs")
    if active.dtype != np.uint8 or active.shape != (session.frames,):
        raise ValueError(f"{path}: reference activity schema differs")
    if str(stored.reshape(()).item()) != session.session_id:
        raise ValueError(f"{path}: reference session identity differs")
    return keys, engine, active


def _valid_resumable_output(
    path: Path,
    *,
    session: SessionSpec,
    features: np.ndarray,
    keys: np.ndarray,
    engine: np.ndarray,
    active: np.ndarray,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                set(archive.files)
                == {"features", "keys", "engine_frame_idx", "input_active", "session_id"}
                and np.array_equal(archive["features"], features)
                and np.array_equal(archive["keys"], keys)
                and np.array_equal(archive["engine_frame_idx"], engine)
                and np.array_equal(archive["input_active"], active)
                and str(archive["session_id"].reshape(()).item())
                == session.session_id
            )
    except (OSError, ValueError, KeyError):
        return False


def _write_standard_shard(
    path: Path,
    *,
    session: SessionSpec,
    features: np.ndarray,
    keys: np.ndarray,
    engine: np.ndarray,
    active: np.ndarray,
) -> None:
    temporary = path.with_suffix(".tmp.npz")
    if temporary.exists():
        raise FileExistsError(f"temporary supervised shard remains: {temporary}")
    np.savez(
        temporary,
        features=np.asarray(features, dtype=np.float16),
        keys=np.asarray(keys, dtype=np.uint8),
        engine_frame_idx=np.asarray(engine, dtype=np.int64),
        input_active=np.asarray(active, dtype=np.uint8),
        session_id=np.asarray(session.session_id),
    )
    temporary.replace(path)


def _read_session_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def assemble_supervised_features(
    *,
    inventory: Inventory,
    checkpoint: CheckpointContract,
    feature_root: Path,
    reference_root: Path,
    output_root: Path,
    terminal_validation: Mapping[str, Any],
    expected_counts: ExpectedCounts = PRODUCTION_COUNTS,
) -> dict[str, Any]:
    """Join labels only after terminal validation has passed in full."""

    _require_terminal_validation(
        terminal_validation,
        inventory=inventory,
        checkpoint=checkpoint,
        expected_counts=expected_counts,
    )
    feature_root = feature_root.resolve()
    reference_root = reference_root.resolve()
    output_root = output_root.resolve()
    for component in output_root.parts:
        _reject_forbidden_identity(component, name="supervised output path")
    output_root.mkdir(parents=True, exist_ok=True)
    publication_names = {
        "full_corpus_manifest.json",
        "shard_hashes.json",
        "supervised_assembly_manifest.json",
        "supervised_assembly_complete.json",
    }
    if any((output_root / name).exists() for name in publication_names):
        raise FileExistsError("supervised assembly publication already exists")
    temporary = sorted(output_root.rglob("*.tmp*"))
    if temporary:
        raise FileExistsError(f"temporary supervised output remains: {temporary[0]}")

    source_manifest_path = reference_root / "full_corpus_manifest.json"
    source_hashes_path = reference_root / "shard_hashes.json"
    source_manifest = _json(source_manifest_path, "reference full-corpus manifest")
    source_hashes = _json(source_hashes_path, "reference shard hashes")
    source_videos = source_manifest.get("videos")
    if not isinstance(source_videos, list) or len(source_videos) != expected_counts.videos:
        raise ValueError("reference full-corpus video membership changed")
    if int(source_manifest.get("session_count", -1)) != expected_counts.sessions:
        raise ValueError("reference full-corpus session count changed")
    if int(source_manifest.get("train_frames", -1)) != expected_counts.frames:
        raise ValueError("reference full-corpus frame count changed")

    expected_ids = sorted(item.session_id for item in inventory.sessions)
    train_path = reference_root / "train_sessions.txt"
    unflagged_path = reference_root / "unflagged_sessions.txt"
    val_path = reference_root / "val_sessions.txt"
    train_ids = _read_session_list(train_path)
    unflagged_ids = _read_session_list(unflagged_path)
    val_ids = _read_session_list(val_path)
    if train_ids != expected_ids:
        raise ValueError("reference train_sessions differs from exact inventory")
    if len(unflagged_ids) != len(set(unflagged_ids)) or not set(unflagged_ids).issubset(train_ids):
        raise ValueError("reference unflagged_sessions is not a unique subset")
    if val_ids:
        raise ValueError("reference base val_sessions must remain empty")
    split_lists = _canonical_split_lists(
        expected_ids,
        unflagged_ids,
        production=expected_counts == PRODUCTION_COUNTS,
    )

    records: list[dict[str, Any]] = []
    new_hashes: dict[str, dict[str, Any]] = {}
    for session in inventory.sessions:
        if session.reference_shard.parent.resolve() != reference_root:
            raise ValueError(f"{session.session_id}: reference root differs")
        reference_receipt = source_hashes.get(session.session_id)
        if not isinstance(reference_receipt, dict):
            raise ValueError(f"{session.session_id}: reference hash receipt missing")
        if reference_receipt.get("sha256") != session.reference_shard_sha256:
            raise ValueError(f"{session.session_id}: inventory/reference hash differs")
        feature_path = feature_root / f"{session.session_id}.npz"
        features, feature_engine, feature_active = _load_feature_only(
            feature_path, session
        )
        keys, reference_engine, reference_active = _load_supervision(session)
        if not np.array_equal(feature_engine, reference_engine):
            raise ValueError(f"{session.session_id}: engine alignment differs")
        if not np.array_equal(feature_active, reference_active):
            raise ValueError(f"{session.session_id}: activity alignment differs")
        destination = output_root / f"{session.session_id}.npz"
        resumed = _valid_resumable_output(
            destination,
            session=session,
            features=features,
            keys=keys,
            engine=reference_engine,
            active=reference_active,
        )
        if not resumed:
            if destination.exists():
                raise ValueError(
                    f"{session.session_id}: mismatched supervised shard requires quarantine"
                )
            _write_standard_shard(
                destination,
                session=session,
                features=features,
                keys=keys,
                engine=reference_engine,
                active=reference_active,
            )
        output_sha = sha256_file(destination)
        destination_stat = destination.stat()
        new_hashes[session.session_id] = {
            "sha256": output_sha,
            "size": destination_stat.st_size,
            "mtime": destination_stat.st_mtime,
        }
        records.append(
            {
                "session_id": session.session_id,
                "frames": session.frames,
                "feature_only_npz": str(feature_path),
                "feature_only_npz_sha256": sha256_file(feature_path),
                "reference_npz": str(session.reference_shard),
                "reference_npz_sha256": session.reference_shard_sha256,
                "features_sha256": array_sha256(features),
                "keys_sha256": array_sha256(keys),
                "engine_frame_idx_sha256": array_sha256(reference_engine),
                "input_active_sha256": array_sha256(reference_active),
                "output_npz": destination.name,
                "output_npz_sha256": output_sha,
                "resumed": resumed,
            }
        )

    for name, values in split_lists.items():
        _write_lines_atomic(output_root / name, values)
    export_manifest_path = feature_root / "feature_export_manifest.json"
    export_manifest_sha = sha256_file(export_manifest_path)
    visual_format = (
        f"dynamics_{checkpoint.arm.lower()}_final_ema_raw_avgpool_float16_v1"
    )
    adapted_manifest = dict(source_manifest)
    adapted_manifest.update(
        {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "format": visual_format,
            "source_visual_representation": source_manifest.get("format"),
            "source_full_corpus_manifest": str(source_manifest_path),
            "source_full_corpus_manifest_sha256": sha256_file(source_manifest_path),
            "dynamics_feature_export_manifest": str(export_manifest_path),
            "dynamics_feature_export_manifest_sha256": export_manifest_sha,
            "terminal_checkpoint_sha256": checkpoint.sha256,
            "post_ssl_supervised_assembly": True,
            "downstream_splits": {
                "all_sessions": "all_sessions.txt",
                "train_non_y4n": "train_sessions.txt",
                "training_monitor_all_y4n": "val_sessions.txt",
                "reported_y4n_later_eight": "y4n_later8_sessions.txt",
            },
        }
    )
    _write_json_atomic(output_root / "full_corpus_manifest.json", adapted_manifest)
    _write_json_atomic(output_root / "shard_hashes.json", new_hashes)
    assembly_manifest = {
        "schema_version": ASSEMBLY_SCHEMA,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "arm": checkpoint.arm,
        "checkpoint_sha256": checkpoint.sha256,
        "inventory_sha256": inventory.sha256,
        "terminal_validation_sha256": canonical_json_sha256(terminal_validation),
        "feature_export_manifest_sha256": export_manifest_sha,
        "source_full_corpus_manifest_sha256": sha256_file(source_manifest_path),
        "labels_first_opened_after_terminal_validation": True,
        "format": visual_format,
        "counts": {
            "videos": expected_counts.videos,
            "sessions": len(records),
            "frames": sum(item["frames"] for item in records),
            "train_sessions": len(split_lists["train_sessions.txt"]),
            "y4n_sessions": len(split_lists["val_sessions.txt"]),
            "y4n_later8_sessions": len(split_lists["y4n_later8_sessions.txt"]),
        },
        "split_lists": {
            name: {
                "count": len(values),
                "sha256": sha256_file(output_root / name),
            }
            for name, values in split_lists.items()
        },
        "sessions": records,
    }
    assembly_path = output_root / "supervised_assembly_manifest.json"
    _write_json_atomic(assembly_path, assembly_manifest)
    completion = {
        "schema_version": COMPLETION_SCHEMA,
        "assembly_manifest_sha256": sha256_file(assembly_path),
        "full_corpus_manifest_sha256": sha256_file(
            output_root / "full_corpus_manifest.json"
        ),
        "shard_hashes_sha256": sha256_file(output_root / "shard_hashes.json"),
        "checkpoint_sha256": checkpoint.sha256,
        "inventory_sha256": inventory.sha256,
        "counts": assembly_manifest["counts"],
    }
    _write_json_atomic(output_root / "supervised_assembly_complete.json", completion)
    expected_files = {
        *(f"{item.session_id}.npz" for item in inventory.sessions),
        *split_lists,
        *publication_names,
    }
    actual_files = {item.name for item in output_root.iterdir()}
    if actual_files != expected_files:
        raise ValueError(
            "supervised output file set differs: "
            f"missing={sorted(expected_files-actual_files)} "
            f"extra={sorted(actual_files-expected_files)}"
        )
    return completion


def _load_and_validate_standard_output(
    path: Path,
    *,
    session: SessionSpec,
    record: Mapping[str, Any],
) -> None:
    if sha256_file(path) != record.get("output_npz_sha256"):
        raise ValueError(f"{session.session_id}: assembled NPZ hash differs")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "features", "keys", "engine_frame_idx", "input_active", "session_id"
        }:
            raise ValueError(f"{session.session_id}: assembled member set differs")
        features = np.asarray(archive["features"])
        keys = np.asarray(archive["keys"])
        engine = np.asarray(archive["engine_frame_idx"])
        active = np.asarray(archive["input_active"])
        stored = archive["session_id"]
    if features.dtype != np.float16 or features.shape != (
        session.frames, REPRESENTATION_DIM
    ):
        raise ValueError(f"{session.session_id}: assembled feature schema differs")
    if not np.isfinite(features).all():
        raise ValueError(f"{session.session_id}: assembled features are non-finite")
    if keys.dtype != np.uint8 or keys.shape != (session.frames, len(KEY_ORDER)):
        raise ValueError(f"{session.session_id}: assembled key schema differs")
    if np.any((keys != 0) & (keys != 1)):
        raise ValueError(f"{session.session_id}: assembled keys are non-binary")
    if engine.dtype != np.int64 or engine.shape != (session.frames,):
        raise ValueError(f"{session.session_id}: assembled engine schema differs")
    if active.dtype != np.uint8 or active.shape != (session.frames,):
        raise ValueError(f"{session.session_id}: assembled activity schema differs")
    if str(stored.reshape(()).item()) != session.session_id:
        raise ValueError(f"{session.session_id}: assembled identity differs")
    for field, array in (
        ("features_sha256", features),
        ("keys_sha256", keys),
        ("engine_frame_idx_sha256", engine),
        ("input_active_sha256", active),
    ):
        if record.get(field) != array_sha256(array):
            raise ValueError(f"{session.session_id}: assembled {field} differs")


def validate_supervised_features(
    *,
    inventory: Inventory,
    checkpoint: CheckpointContract,
    feature_root: Path,
    reference_root: Path,
    output_root: Path,
    terminal_validation: Mapping[str, Any],
    expected_counts: ExpectedCounts = PRODUCTION_COUNTS,
) -> dict[str, Any]:
    """Independently deep-validate a published supervised assembly."""

    failures: list[str] = []
    checked_sessions = 0
    checked_frames = 0
    try:
        _require_terminal_validation(
            terminal_validation,
            inventory=inventory,
            checkpoint=checkpoint,
            expected_counts=expected_counts,
        )
        feature_root = feature_root.resolve()
        reference_root = reference_root.resolve()
        output_root = output_root.resolve()
        for component in output_root.parts:
            _reject_forbidden_identity(component, name="supervised output path")

        assembly_path = output_root / "supervised_assembly_manifest.json"
        completion_path = output_root / "supervised_assembly_complete.json"
        full_manifest_path = output_root / "full_corpus_manifest.json"
        hashes_path = output_root / "shard_hashes.json"
        assembly = _json(assembly_path, "supervised assembly manifest")
        completion = _json(completion_path, "supervised assembly completion")
        full_manifest = _json(full_manifest_path, "adapted full-corpus manifest")
        hashes = _json(hashes_path, "assembled shard hashes")
        if assembly.get("schema_version") != ASSEMBLY_SCHEMA:
            raise ValueError("supervised assembly schema differs")
        if completion.get("schema_version") != COMPLETION_SCHEMA:
            raise ValueError("supervised completion schema differs")
        expected_bindings = {
            "checkpoint_sha256": checkpoint.sha256,
            "inventory_sha256": inventory.sha256,
        }
        for key, expected in expected_bindings.items():
            if assembly.get(key) != expected or completion.get(key) != expected:
                raise ValueError(f"supervised assembly {key} binding differs")
        if completion.get("assembly_manifest_sha256") != sha256_file(assembly_path):
            raise ValueError("supervised completion manifest hash differs")
        if completion.get("full_corpus_manifest_sha256") != sha256_file(
            full_manifest_path
        ):
            raise ValueError("supervised completion full manifest hash differs")
        if completion.get("shard_hashes_sha256") != sha256_file(hashes_path):
            raise ValueError("supervised completion shard-hash receipt differs")
        if assembly.get("labels_first_opened_after_terminal_validation") is not True:
            raise ValueError("supervised label-access chronology receipt differs")
        expected_format = (
            f"dynamics_{checkpoint.arm.lower()}_final_ema_raw_avgpool_float16_v1"
        )
        if assembly.get("format") != expected_format:
            raise ValueError("supervised representation format differs")
        if full_manifest.get("format") != expected_format:
            raise ValueError("adapted full-corpus representation format differs")
        if full_manifest.get("post_ssl_supervised_assembly") is not True:
            raise ValueError("adapted manifest lacks post-SSL receipt")
        if full_manifest.get("terminal_checkpoint_sha256") != checkpoint.sha256:
            raise ValueError("adapted manifest checkpoint binding differs")

        expected_ids = sorted(item.session_id for item in inventory.sessions)
        source_unflagged = _read_session_list(
            reference_root / "unflagged_sessions.txt"
        )
        split_lists = _canonical_split_lists(
            expected_ids,
            source_unflagged,
            production=expected_counts == PRODUCTION_COUNTS,
        )
        split_receipts = assembly.get("split_lists")
        if not isinstance(split_receipts, dict) or set(split_receipts) != set(
            split_lists
        ):
            raise ValueError("supervised split receipt set differs")
        for name, expected_values in split_lists.items():
            path = output_root / name
            if _read_session_list(path) != expected_values:
                raise ValueError(f"supervised {name} membership differs")
            expected_receipt = {
                "count": len(expected_values),
                "sha256": sha256_file(path),
            }
            if split_receipts.get(name) != expected_receipt:
                raise ValueError(f"supervised {name} receipt differs")

        expected_count_row = {
            "videos": expected_counts.videos,
            "sessions": expected_counts.sessions,
            "frames": expected_counts.frames,
            "train_sessions": len(split_lists["train_sessions.txt"]),
            "y4n_sessions": len(split_lists["val_sessions.txt"]),
            "y4n_later8_sessions": len(split_lists["y4n_later8_sessions.txt"]),
        }
        if assembly.get("counts") != expected_count_row:
            raise ValueError("supervised assembly counts differ")
        if completion.get("counts") != expected_count_row:
            raise ValueError("supervised completion counts differ")

        rows = assembly.get("sessions")
        if not isinstance(rows, list):
            raise ValueError("supervised assembly session records are missing")
        rows_by_id: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(
                row.get("session_id"), str
            ):
                raise ValueError("supervised assembly has malformed session record")
            if row["session_id"] in rows_by_id:
                raise ValueError("supervised assembly has duplicate session record")
            rows_by_id[str(row["session_id"])] = row
        if set(rows_by_id) != set(expected_ids) or set(hashes) != set(expected_ids):
            raise ValueError("supervised assembly session membership differs")
        for session in inventory.sessions:
            row = rows_by_id[session.session_id]
            feature_path = feature_root / f"{session.session_id}.npz"
            reference_path = session.reference_shard.resolve()
            if Path(str(row.get("feature_only_npz", ""))).resolve() != feature_path:
                raise ValueError(f"{session.session_id}: feature source path differs")
            if sha256_file(feature_path) != row.get("feature_only_npz_sha256"):
                raise ValueError(f"{session.session_id}: feature source hash differs")
            if Path(str(row.get("reference_npz", ""))).resolve() != reference_path:
                raise ValueError(f"{session.session_id}: reference path differs")
            if sha256_file(reference_path) != row.get("reference_npz_sha256"):
                raise ValueError(f"{session.session_id}: reference hash differs")
            output_path = output_root / f"{session.session_id}.npz"
            _load_and_validate_standard_output(
                output_path,
                session=session,
                record=row,
            )
            expected_hash_row = {
                "sha256": row["output_npz_sha256"],
                "size": output_path.stat().st_size,
                "mtime": output_path.stat().st_mtime,
            }
            if hashes.get(session.session_id) != expected_hash_row:
                raise ValueError(f"{session.session_id}: shard hash map differs")
            checked_sessions += 1
            checked_frames += session.frames

        expected_files = {
            *(f"{item.session_id}.npz" for item in inventory.sessions),
            *split_lists,
            "full_corpus_manifest.json",
            "shard_hashes.json",
            "supervised_assembly_manifest.json",
            "supervised_assembly_complete.json",
        }
        actual_files = {item.name for item in output_root.iterdir()}
        if actual_files != expected_files:
            raise ValueError("supervised output file set differs")
        temporary = sorted(output_root.rglob("*.tmp*"))
        if temporary:
            raise ValueError(f"temporary supervised output remains: {temporary[0]}")
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")
    return {
        "schema_version": VALIDATION_SCHEMA,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "ok": not failures,
        "arm": checkpoint.arm,
        "checkpoint_sha256": checkpoint.sha256,
        "inventory_sha256": inventory.sha256,
        "output_root": str(Path(output_root).resolve()),
        "deep_shards": True,
        "deep_sources": True,
        "counts": {
            "expected_sessions": expected_counts.sessions,
            "expected_frames": expected_counts.frames,
            "checked_sessions": checked_sessions,
            "checked_frames": checked_frames,
        },
        "failures": failures,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--inventory", type=Path, required=True)
    value.add_argument("--inventory-sha256", required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--checkpoint-sha256", required=True)
    value.add_argument("--arm", choices=("C", "D"), required=True)
    value.add_argument("--expected-completed-steps", type=int, required=True)
    value.add_argument("--feature-root", type=Path, required=True)
    value.add_argument("--reference-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--validate-only", action="store_true")
    value.add_argument(
        "--report",
        type=Path,
        help="write the independent deep-validation receipt outside --output",
    )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    inventory = load_inventory(args.inventory, args.inventory_sha256)
    checkpoint = load_checkpoint_contract(
        args.checkpoint,
        args.checkpoint_sha256,
        expected_arm=args.arm,
        expected_completed_steps=args.expected_completed_steps,
    )
    terminal_validation = validate_export(
        inventory=inventory,
        checkpoint=checkpoint,
        out_dir=args.feature_root,
        deep_shards=True,
        deep_references=True,
    )
    if args.report is not None and args.report.resolve().is_relative_to(
        args.output.resolve()
    ):
        raise ValueError("supervised validation report must be outside --output")
    completion: Mapping[str, Any] | None = None
    if not args.validate_only:
        completion = assemble_supervised_features(
            inventory=inventory,
            checkpoint=checkpoint,
            feature_root=args.feature_root,
            reference_root=args.reference_root,
            output_root=args.output,
            terminal_validation=terminal_validation,
        )
    report = validate_supervised_features(
        inventory=inventory,
        checkpoint=checkpoint,
        feature_root=args.feature_root,
        reference_root=args.reference_root,
        output_root=args.output,
        terminal_validation=terminal_validation,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(args.report, report)
    print(
        json.dumps(
            {"completion": completion, "validation": report},
            indent=2,
            sort_keys=True,
        )
    )
    if report.get("ok") is not True:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
