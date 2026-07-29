"""Fail-closed validation for the four-session corrected own-v3 features.

This validator deliberately knows the exact historical train/validation split.
It refuses discovery, extra shards, temporary files, or a fifth session.  The
``preflight`` command validates only the corrected RGB source.  The ``validate``
command additionally verifies a clean feature build and writes one atomic,
content-bound JSON receipt; the launcher publishes that receipt only after the
validated directory is atomically renamed into place.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torchvision

from data.precompute_features import (
    BACKBONE_FEATURE_DIM,
    FRAME_SIZE,
)
from data.schema import KEY_ORDER


SCHEMA_VERSION = "madeleine.own-v3-features-validation.v1"
FEATURE_FORMAT = "resnet18_imagenet_avgpool_float16_v1"
TRAIN_SESSION_IDS = (
    "rec_20260724_190233",
    "rec_20260725_015612",
    "rec_20260725_021338",
)
VAL_SESSION_IDS = ("rec_20260724_171305_5min",)
SESSION_IDS = (*TRAIN_SESSION_IDS, *VAL_SESSION_IDS)
SOURCE_FILES = {
    "build_manifest.json",
    "train_sessions.txt",
    "val_sessions.txt",
    *(f"{session_id}.npz" for session_id in SESSION_IDS),
}
FEATURE_FILES = {
    "build_manifest.json",
    "feature_build_manifest.json",
    "train_sessions.txt",
    "val_sessions.txt",
    *(f"{session_id}.npz" for session_id in SESSION_IDS),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _lines(path: Path) -> list[str]:
    try:
        values = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as error:
        raise ValueError(f"split list is not readable: {path}") from error
    if len(values) != len(set(values)):
        raise ValueError(f"split list contains duplicate sessions: {path}")
    return values


def _temporary_files(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name.startswith(".")
            or path.name.endswith(".tmp")
            or ".tmp." in path.name
            or path.name.endswith(".tmp.npz")
        )
    )


def _require_exact_files(root: Path, expected: set[str], description: str) -> None:
    if not root.is_dir():
        raise ValueError(f"{description} directory is missing: {root}")
    entries = list(root.iterdir())
    nonregular = sorted(
        path.name
        for path in entries
        if path.is_symlink() or not path.is_file()
    )
    if nonregular:
        raise ValueError(f"{description} has non-regular entries: {nonregular}")
    observed = {path.name for path in entries}
    if observed != expected:
        raise ValueError(
            f"{description} inventory changed: "
            f"missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )
    temporary = _temporary_files(root)
    if temporary:
        raise ValueError(f"{description} has temporary files: {temporary}")


def _manifest_sessions(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = manifest.get("sessions")
    if not isinstance(rows, list) or len(rows) != len(SESSION_IDS):
        raise ValueError("own-v3 build manifest must contain exactly four sessions")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("own-v3 manifest session rows must be objects")
        session_id = row.get("session_id")
        if not isinstance(session_id, str) or session_id in result:
            raise ValueError("own-v3 manifest has a missing or duplicate session ID")
        result[session_id] = row
    if set(result) != set(SESSION_IDS):
        raise ValueError(
            "own-v3 manifest membership changed: "
            f"expected={sorted(SESSION_IDS)} observed={sorted(result)}"
        )
    return result


def _validate_split_contract(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    train = _lines(root / "train_sessions.txt")
    val = _lines(root / "val_sessions.txt")
    if train != list(TRAIN_SESSION_IDS):
        raise ValueError(f"own-v3 train split changed: {train}")
    if val != list(VAL_SESSION_IDS):
        raise ValueError(f"own-v3 validation split changed: {val}")
    split = manifest.get("split")
    expected = {
        "train": list(TRAIN_SESSION_IDS),
        "val": list(VAL_SESSION_IDS),
        "unit": "session",
    }
    if split != expected:
        raise ValueError(f"own-v3 manifest split changed: {split!r}")
    return {
        "train": train,
        "validation": val,
        "train_sha256": sha256_file(root / "train_sessions.txt"),
        "validation_sha256": sha256_file(root / "val_sessions.txt"),
    }


def _stored_session_id(value: np.ndarray, path: Path) -> str:
    if value.size != 1:
        raise ValueError(f"{path}: session_id must contain one scalar string")
    return str(value.reshape(()).item())


def _validate_source_shard(
    path: Path, session_id: str, manifest_row: Mapping[str, Any]
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "frames",
            "keys",
            "engine_frame_idx",
            "input_active",
            "session_id",
        }
        if set(archive.files) != required:
            raise ValueError(
                f"{path}: RGB shard fields changed: "
                f"missing={sorted(required - set(archive.files))} "
                f"extra={sorted(set(archive.files) - required)}"
            )
        frames = np.asarray(archive["frames"])
        keys = np.asarray(archive["keys"])
        engine = np.asarray(archive["engine_frame_idx"])
        active = np.asarray(archive["input_active"])
        session_id_array = np.asarray(archive["session_id"])
        stored_id = _stored_session_id(session_id_array, path)

    if stored_id != session_id:
        raise ValueError(f"{path}: stored session ID changed: {stored_id!r}")
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (
        FRAME_SIZE,
        FRAME_SIZE,
        3,
    ):
        raise ValueError(f"{path}: frames must be uint8 [N,128,128,3]")
    frame_count = len(frames)
    if keys.dtype != np.uint8 or keys.shape != (frame_count, len(KEY_ORDER)):
        raise ValueError(f"{path}: keys must be uint8 [N,{len(KEY_ORDER)}]")
    if engine.dtype != np.int64 or engine.shape != (frame_count,):
        raise ValueError(f"{path}: engine_frame_idx must be int64 [N]")
    if active.dtype != np.uint8 or active.shape != (frame_count,):
        raise ValueError(f"{path}: input_active must be uint8 [N]")
    if not np.all(np.isin(keys, (0, 1))) or not np.all(np.isin(active, (0, 1))):
        raise ValueError(f"{path}: supervision must be binary")
    if manifest_row.get("npz") != path.name:
        raise ValueError(f"{path}: manifest NPZ name changed")
    if manifest_row.get("frames") != frame_count:
        raise ValueError(f"{path}: manifest frame count changed")
    return {
        "session_id": session_id,
        "frames": frame_count,
        "source_npz_sha256": sha256_file(path),
        "arrays": {
            "frames_sha256": _array_sha256(frames),
            "keys_sha256": _array_sha256(keys),
            "engine_frame_idx_sha256": _array_sha256(engine),
            "input_active_sha256": _array_sha256(active),
            "session_id_sha256": _array_sha256(session_id_array),
            "session_id": stored_id,
        },
    }


def validate_source(root: Path) -> dict[str, Any]:
    """Validate the exact corrected four-session RGB generation."""

    _require_exact_files(root, SOURCE_FILES, "own-v3 source")
    manifest_path = root / "build_manifest.json"
    manifest = _json(manifest_path, "own-v3 build manifest")
    if manifest.get("frame_size") != FRAME_SIZE:
        raise ValueError("own-v3 source frame size changed")
    grid = manifest.get("grid")
    if not isinstance(grid, Mapping) or grid.get("engine_hz") != 60:
        raise ValueError("own-v3 source engine grid changed")
    split = _validate_split_contract(root, manifest)
    rows = _manifest_sessions(manifest)
    sessions = [
        _validate_source_shard(root / f"{session_id}.npz", session_id, rows[session_id])
        for session_id in SESSION_IDS
    ]
    return {
        "root": str(root.resolve()),
        "build_manifest_sha256": sha256_file(manifest_path),
        "split": split,
        "sessions": sessions,
        "session_count": len(sessions),
        "frame_count": sum(int(row["frames"]) for row in sessions),
        "exact_inventory": sorted(SOURCE_FILES),
    }


def _validate_feature_manifest(
    path: Path, source_root: Path, source: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = _json(path, "own-v3 feature build manifest")
    expected_header = {
        "format": FEATURE_FORMAT,
        "backbone_feature_dim": BACKBONE_FEATURE_DIM,
        "frame_size": FRAME_SIZE,
        "source_kind": "audited_rgb_shards",
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(f"feature build manifest changed {key}")
    rows = manifest.get("sessions")
    if not isinstance(rows, list) or len(rows) != len(SESSION_IDS):
        raise ValueError("feature build manifest must contain exactly four sessions")
    by_id: dict[str, Mapping[str, Any]] = {}
    source_frames = {
        row["session_id"]: row["frames"] for row in source["sessions"]
    }
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("feature manifest session rows must be objects")
        session_id = row.get("session_id")
        if not isinstance(session_id, str) or session_id in by_id:
            raise ValueError("feature manifest has a missing or duplicate session ID")
        if session_id not in SESSION_IDS:
            raise ValueError(f"feature manifest contains unexpected session {session_id}")
        if row.get("frames") != source_frames[session_id]:
            raise ValueError(f"feature manifest frame count changed for {session_id}")
        if row.get("npz") != f"{session_id}.npz":
            raise ValueError(f"feature manifest output name changed for {session_id}")
        source_value = row.get("source")
        if not isinstance(source_value, str) or Path(source_value).resolve() != (
            source_root / f"{session_id}.npz"
        ).resolve():
            raise ValueError(f"feature manifest source changed for {session_id}")
        if row.get("resumed") is not False:
            raise ValueError(f"feature build unexpectedly resumed {session_id}")
        by_id[session_id] = row
    if set(by_id) != set(SESSION_IDS):
        raise ValueError("feature manifest membership changed")
    if [row.get("session_id") for row in rows] != list(SESSION_IDS):
        raise ValueError("feature manifest session order changed")
    return manifest


def _validate_copied_build_manifest(
    path: Path, source_root: Path, source_manifest: Mapping[str, Any]
) -> None:
    observed = _json(path, "feature engine-truth build manifest")
    source_value = observed.pop("source_build_manifest", None)
    if not isinstance(source_value, str) or Path(source_value).resolve() != (
        source_root / "build_manifest.json"
    ).resolve():
        raise ValueError("feature build manifest source pointer changed")
    if observed.pop("visual_representation", None) != FEATURE_FORMAT:
        raise ValueError("feature build manifest visual representation changed")
    if observed.pop("backbone_feature_dim", None) != BACKBONE_FEATURE_DIM:
        raise ValueError("feature build manifest backbone dimension changed")
    if observed != source_manifest:
        raise ValueError("feature engine-truth manifest differs from RGB source")


def _validate_feature_shard(
    source_path: Path,
    feature_path: Path,
    session_id: str,
) -> dict[str, Any]:
    with np.load(source_path, allow_pickle=False) as source_archive, np.load(
        feature_path, allow_pickle=False
    ) as feature_archive:
        required = {
            "features",
            "keys",
            "engine_frame_idx",
            "input_active",
            "session_id",
        }
        if set(feature_archive.files) != required:
            raise ValueError(
                f"{feature_path}: feature shard fields changed: "
                f"missing={sorted(required - set(feature_archive.files))} "
                f"extra={sorted(set(feature_archive.files) - required)}"
            )
        features = np.asarray(feature_archive["features"])
        feature_keys = np.asarray(feature_archive["keys"])
        feature_engine = np.asarray(feature_archive["engine_frame_idx"])
        feature_active = np.asarray(feature_archive["input_active"])
        feature_id_array = np.asarray(feature_archive["session_id"])
        source_keys = np.asarray(source_archive["keys"])
        source_engine = np.asarray(source_archive["engine_frame_idx"])
        source_active = np.asarray(source_archive["input_active"])
        source_id_array = np.asarray(source_archive["session_id"])

    frame_count = len(source_keys)
    if features.dtype != np.float16 or features.shape != (
        frame_count,
        BACKBONE_FEATURE_DIM,
    ):
        raise ValueError(f"{feature_path}: features must be float16 [N,512]")
    if not np.all(np.isfinite(features)):
        raise ValueError(f"{feature_path}: features contain non-finite values")
    comparisons = {
        "keys": feature_keys.dtype == source_keys.dtype
        and np.array_equal(feature_keys, source_keys),
        "engine_frame_idx": feature_engine.dtype == source_engine.dtype
        and np.array_equal(feature_engine, source_engine),
        "input_active": feature_active.dtype == source_active.dtype
        and np.array_equal(feature_active, source_active),
        "session_id": feature_id_array.dtype == source_id_array.dtype
        and np.array_equal(feature_id_array, source_id_array),
    }
    changed = [key for key, equal in comparisons.items() if not equal]
    if changed:
        raise ValueError(
            f"{feature_path}: supervision differs from own-v3 pixels: {changed}"
        )
    stored_id = _stored_session_id(feature_id_array, feature_path)
    if stored_id != session_id:
        raise ValueError(f"{feature_path}: stored session ID changed")
    return {
        "session_id": session_id,
        "frames": frame_count,
        "source_npz_sha256": sha256_file(source_path),
        "feature_npz_sha256": sha256_file(feature_path),
        "features_sha256": _array_sha256(features),
        "keys_sha256": _array_sha256(feature_keys),
        "engine_frame_idx_sha256": _array_sha256(feature_engine),
        "input_active_sha256": _array_sha256(feature_active),
        "session_id_sha256": _array_sha256(feature_id_array),
        "session_id_value": stored_id,
        "supervision_equal_to_source": comparisons,
        "features_finite": True,
    }


def _git_receipt(repo: Path, expected_commit: str | None) -> dict[str, Any]:
    if not repo.is_dir():
        raise ValueError(f"repository is missing: {repo}")
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if expected_commit is not None and commit != expected_commit:
        raise ValueError(
            f"repository commit changed: expected {expected_commit}, observed {commit}"
        )
    relevant = (
        "data/precompute_features.py",
        "data/schema.py",
        "experiments/validate_own_v3_features.py",
        "experiments/run_own_v3_features.sh",
    )
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--", *relevant],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(f"feature-build implementation is dirty: {dirty}")
    return {
        "commit": commit,
        "relevant_files": {
            relative: sha256_file(repo / relative) for relative in relevant
        },
        "relevant_files_clean": True,
    }


def validate_features(
    *,
    source_root: Path,
    feature_root: Path,
    published_output: Path,
    repo: Path,
    expected_commit: str | None,
    source_snapshot_path: Path,
) -> dict[str, Any]:
    source = validate_source(source_root)
    source_snapshot = _json(source_snapshot_path, "own-v3 source preflight snapshot")
    if source_snapshot != source:
        raise ValueError(
            "own-v3 source changed after preflight; refusing to bind features"
        )
    _require_exact_files(feature_root, FEATURE_FILES, "own-v3 feature staging")
    source_manifest = _json(
        source_root / "build_manifest.json", "own-v3 build manifest"
    )
    feature_manifest = _validate_feature_manifest(
        feature_root / "feature_build_manifest.json", source_root, source
    )
    _validate_copied_build_manifest(
        feature_root / "build_manifest.json", source_root, source_manifest
    )
    output_split = _validate_split_contract(feature_root, source_manifest)
    if output_split["train_sha256"] != source["split"]["train_sha256"]:
        raise ValueError("feature train split bytes differ from own-v3 source")
    if output_split["validation_sha256"] != source["split"]["validation_sha256"]:
        raise ValueError("feature validation split bytes differ from own-v3 source")

    sessions = [
        _validate_feature_shard(
            source_root / f"{session_id}.npz",
            feature_root / f"{session_id}.npz",
            session_id,
        )
        for session_id in SESSION_IDS
    ]
    git = _git_receipt(repo, expected_commit)
    inventory = {
        name: {
            "bytes": (feature_root / name).stat().st_size,
            "sha256": sha256_file(feature_root / name),
        }
        for name in sorted(FEATURE_FILES)
    }
    content = {
        "source": source,
        "feature_inventory": inventory,
        "feature_build_manifest_sha256": sha256_file(
            feature_root / "feature_build_manifest.json"
        ),
        "sessions": sessions,
        "split": output_split,
        "implementation": git,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "source_preflight_snapshot_sha256": sha256_file(source_snapshot_path),
    }
    canonical = json.dumps(
        content, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root.resolve()),
        "published_output": str(published_output.resolve()),
        "feature_format": FEATURE_FORMAT,
        "backbone_feature_dim": BACKBONE_FEATURE_DIM,
        "session_count": len(SESSION_IDS),
        "frame_count": sum(int(row["frames"]) for row in sessions),
        "train_sessions": list(TRAIN_SESSION_IDS),
        "validation_sessions": list(VAL_SESSION_IDS),
        "checks": {
            "exact_four_source_sessions": True,
            "exact_four_feature_sessions": True,
            "split_lists_preserved_byte_for_byte": True,
            "supervision_arrays_equal_to_source": True,
            "feature_arrays_finite_float16_512d": True,
            "no_extra_or_temporary_files": True,
            "implementation_clean_and_commit_bound": True,
        },
        "content_sha256": hashlib.sha256(canonical).hexdigest(),
        "content": content,
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise ValueError(f"refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if os.path.lexists(temporary):
        raise ValueError(f"stale temporary receipt exists: {temporary}")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--source", type=Path, required=True)
    preflight.add_argument("--out", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--source", type=Path, required=True)
    validate.add_argument("--features", type=Path, required=True)
    validate.add_argument("--published-output", type=Path, required=True)
    validate.add_argument("--repo", type=Path, required=True)
    validate.add_argument("--source-commit")
    validate.add_argument("--source-snapshot", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "preflight":
        report = validate_source(args.source)
        _write_json_atomic(args.out, report)
        print(
            json.dumps(
                {
                    "status": "source_valid",
                    "session_count": report["session_count"],
                    "frame_count": report["frame_count"],
                    "build_manifest_sha256": report["build_manifest_sha256"],
                    "snapshot": str(args.out),
                },
                indent=2,
            )
        )
        return
    report = validate_features(
        source_root=args.source,
        feature_root=args.features,
        published_output=args.published_output,
        repo=args.repo,
        expected_commit=args.source_commit,
        source_snapshot_path=args.source_snapshot,
    )
    _write_json_atomic(args.out, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "session_count": report["session_count"],
                "frame_count": report["frame_count"],
                "content_sha256": report["content_sha256"],
                "receipt": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
