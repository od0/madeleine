"""Build the content-bound 128px frame store for Study H.

The builder copies exactly four already-masked own-v3 shards into memory-mapped
uint8 frame arrays and publishes compact oracle-example indices.  It rebuilds
the completed oracle dataset manifest exactly before any output is promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.oracle_window_localization import (
    FORBIDDEN_SESSION_IDS,
    _dataset_manifest,
    _example_arrays,
    _prepare_data,
    sha256_file,
)
from experiments.prepare_oracle_window_pixel_crops import _source_authority


CACHE_SCHEMA = "madeleine.oracle-window-fullres-cache.v1"
RECEIPT_SCHEMA = "madeleine.oracle-window-fullres-cache-complete.v1"
EXPECTED_MASKS = frozenset({"frame_index_strip", "input_overlay"})


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if _json(path) != value:
        raise ValueError(f"serialized JSON changed on reload: {path}")


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise ValueError(f"NPZ inventory changed: {path}")
        for name, expected in arrays.items():
            observed = np.asarray(archive[name])
            if observed.dtype != expected.dtype or not np.array_equal(observed, expected):
                raise ValueError(f"NPZ array changed on reload: {path}:{name}")


def _copy_frame_memmap(frames: np.ndarray, path: Path, *, chunk: int = 1024) -> None:
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (128, 128, 3):
        raise ValueError("source frames must be uint8 [N,128,128,3]")
    target = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint8, shape=frames.shape
    )
    for start in range(0, len(frames), chunk):
        target[start : start + chunk] = frames[start : start + chunk]
        target.flush()
    del target
    observed = np.load(path, mmap_mode="r", allow_pickle=False)
    if observed.dtype != np.uint8 or observed.shape != frames.shape:
        raise ValueError(f"frame memmap changed shape or dtype: {path}")
    for start in range(0, len(frames), chunk):
        if not np.array_equal(observed[start : start + chunk], frames[start : start + chunk]):
            raise ValueError(f"frame memmap changed content: {path}")


def _file_receipt(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_fullres_cache(
    *,
    source_root: Path,
    feature_root: Path,
    feature_receipt_path: Path,
    base_config_path: Path,
    base_dataset_manifest_path: Path,
    output: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    for target in (output, receipt_path):
        if os.path.lexists(target):
            raise ValueError(f"refusing to overwrite full-resolution artifact: {target}")
    staging = output.with_name(f".{output.name}.staging")
    if os.path.lexists(staging):
        raise ValueError(f"stale full-resolution staging exists: {staging}")

    base_config = _json(base_config_path)
    if base_config.get("status") != "preregistered_before_validation_inference":
        raise ValueError("base oracle contract is not frozen")
    feature_receipt = _json(feature_receipt_path)
    if feature_receipt.get("status") != "complete":
        raise ValueError("feature receipt is incomplete")
    expected_feature_sha = str(base_config["dataset"]["feature_receipt_sha256"])
    if sha256_file(feature_receipt_path) != expected_feature_sha:
        raise ValueError("feature receipt hash changed")
    if Path(str(feature_receipt.get("source_root"))).resolve() != source_root.resolve():
        raise ValueError("feature receipt points to a different source root")

    source_manifest_path = source_root / "build_manifest.json"
    source_manifest = _json(source_manifest_path)
    if sha256_file(source_manifest_path) != str(base_config["dataset"]["source_build_manifest_sha256"]):
        raise ValueError("source build manifest hash changed")
    if int(source_manifest.get("frame_size", -1)) != 128:
        raise ValueError("source frame size changed")
    source_rows = source_manifest.get("sessions")
    if not isinstance(source_rows, list):
        raise ValueError("source manifest sessions are missing")
    source_manifest_by_id = {str(row["session_id"]): row for row in source_rows}

    sessions, train_examples, val_examples, construction = _prepare_data(
        feature_root=feature_root, config=base_config
    )
    rebuilt = _dataset_manifest(
        sessions=sessions,
        train_examples=train_examples,
        val_examples=val_examples,
        construction=construction,
        config=base_config,
    )
    expected_manifest = _json(base_dataset_manifest_path)
    if rebuilt != expected_manifest:
        raise ValueError("completed oracle dataset manifest did not rebuild exactly")

    allowed_ids = tuple(
        (*base_config["dataset"]["train_sessions"], *base_config["dataset"]["validation_sessions"])
    )
    if len(allowed_ids) != 4 or len(set(allowed_ids)) != 4:
        raise ValueError("full-resolution cache requires exactly four unique sessions")
    if set(allowed_ids).intersection(FORBIDDEN_SESSION_IDS):
        raise ValueError("forbidden session entered the full-resolution cache")
    if set(source_manifest_by_id) != set(allowed_ids):
        raise ValueError("source view must contain every and only the four allowed sessions")
    for session_id, row in source_manifest_by_id.items():
        if set(row.get("masked_regions", [])) != EXPECTED_MASKS:
            raise ValueError(f"answer-key mask record changed: {session_id}")

    authority = _source_authority(feature_receipt)
    if set(authority) != set(allowed_ids):
        raise ValueError("source authority must contain every and only allowed session")

    staging.mkdir(parents=True, exist_ok=False)
    (staging / "frames").mkdir()
    (staging / "supervision").mkdir()
    files: dict[str, dict[str, Any]] = {}
    source_checks: dict[str, Any] = {}
    try:
        for session_id in allowed_ids:
            source_path = source_root / f"{session_id}.npz"
            expected_source_sha = str(authority[session_id]["source_npz_sha256"])
            observed_source_sha = sha256_file(source_path)
            if observed_source_sha != expected_source_sha:
                raise ValueError(f"source NPZ hash changed: {session_id}")
            with np.load(source_path, allow_pickle=False) as archive:
                required = {
                    "frames", "keys", "engine_frame_idx", "input_active", "session_id"
                }
                if set(archive.files) != required:
                    raise ValueError(f"source NPZ inventory changed: {session_id}")
                frames = np.asarray(archive["frames"])
                keys = np.asarray(archive["keys"])
                engine = np.asarray(archive["engine_frame_idx"])
                active = np.asarray(archive["input_active"])
                stored_id = str(np.asarray(archive["session_id"]).reshape(()).item())
            feature_session = sessions[session_id]
            if stored_id != session_id:
                raise ValueError(f"stored session identity changed: {session_id}")
            if not (
                np.array_equal(keys, feature_session.keys)
                and np.array_equal(engine, feature_session.engine_frame_idx)
                and np.array_equal(active, feature_session.input_active)
            ):
                raise ValueError(f"source supervision differs from feature authority: {session_id}")

            frame_path = staging / "frames" / f"{session_id}.npy"
            _copy_frame_memmap(frames, frame_path)
            supervision_path = staging / "supervision" / f"{session_id}.npz"
            _write_npz(
                supervision_path,
                {
                    "keys": keys,
                    "engine_frame_idx": engine,
                    "input_active": active,
                    "session_id": np.asarray(session_id),
                },
            )
            files[str(frame_path.relative_to(staging))] = _file_receipt(frame_path)
            files[str(supervision_path.relative_to(staging))] = _file_receipt(supervision_path)
            source_checks[session_id] = {
                "source_npz_sha256": observed_source_sha,
                "rows": len(frames),
                "frame_shape": list(frames.shape),
                "mask_regions": sorted(EXPECTED_MASKS),
            }

        split_examples = {"train": train_examples, "validation": val_examples}
        for split, examples in split_examples.items():
            index_path = staging / f"{split}_examples.npz"
            _write_npz(index_path, _example_arrays(examples))
            files[index_path.name] = _file_receipt(index_path)

        manifest = {
            "schema_version": CACHE_SCHEMA,
            "builder_sha256": sha256_file(Path(__file__)),
            "base_config_sha256": sha256_file(base_config_path),
            "base_dataset_manifest_sha256": sha256_file(base_dataset_manifest_path),
            "feature_receipt_sha256": sha256_file(feature_receipt_path),
            "source_build_manifest_sha256": sha256_file(source_manifest_path),
            "frame_size": 128,
            "crop_frames": int(base_config["dataset"]["crop_frames"]),
            "candidate_width": int(base_config["dataset"]["candidate_width"]),
            "context_halo": int(base_config["dataset"]["context_halo"]),
            "sessions": list(allowed_ids),
            "train_examples": len(train_examples),
            "validation_examples": len(val_examples),
            "validation_blocks": len({row.block_id for row in val_examples}),
            "source_checks": source_checks,
            "files": files,
        }
        manifest_path = staging / "manifest.json"
        _write_json(manifest_path, manifest)
        files[manifest_path.name] = _file_receipt(manifest_path)
        # The manifest intentionally binds all payload files but not itself.
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output)
    except Exception:
        # Preserve staging for diagnosis; never reuse it silently.
        raise

    receipt_base = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        "published_output": str(output.resolve()),
        "builder_sha256": sha256_file(Path(__file__)),
        "base_config_sha256": sha256_file(base_config_path),
        "base_dataset_manifest_sha256": sha256_file(base_dataset_manifest_path),
        "feature_receipt_sha256": sha256_file(feature_receipt_path),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "payload": {
            relative: _file_receipt(output / relative)
            for relative in sorted(files)
        },
        "checks": {
            "exact_four_source_sessions": True,
            "source_hashes_exact": True,
            "answer_key_masks_recorded": True,
            "supervision_equal_to_feature_authority": True,
            "base_dataset_manifest_rebuilt_exactly": True,
            "forbidden_sessions_absent": True,
            "memmaps_reloaded_exactly": True,
            "example_indices_reloaded_exactly": True,
        },
    }
    receipt = dict(receipt_base)
    receipt["content_sha256"] = _canonical_sha256(receipt_base)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_name(f".{receipt_path.name}.tmp")
    if os.path.lexists(temporary):
        raise ValueError(f"stale receipt temporary exists: {temporary}")
    _write_json(temporary, receipt)
    temporary.replace(receipt_path)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--feature-root", required=True, type=Path)
    parser.add_argument("--feature-receipt", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--base-dataset-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    receipt = build_fullres_cache(
        source_root=args.source_root,
        feature_root=args.feature_root,
        feature_receipt_path=args.feature_receipt,
        base_config_path=args.base_config,
        base_dataset_manifest_path=args.base_dataset_manifest,
        output=args.out,
        receipt_path=args.receipt,
    )
    print(json.dumps({"status": receipt["status"], "receipt": str(args.receipt)}))


if __name__ == "__main__":
    main()
