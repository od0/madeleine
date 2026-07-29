"""Build a content-bound, downsampled masked-pixel cache for the oracle follow-up.

The cache contains only the already frozen oracle crops.  It never changes split
membership or offset assignment, and it verifies the source-pixel hashes recorded
by the completed canonical-feature receipt before publishing a completion receipt.
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
    OracleExample,
    _array_sha256,
    _dataset_manifest,
    _example_arrays,
    _prepare_data,
    sha256_file,
)


CACHE_SCHEMA = "madeleine.oracle-window-pixel-crops.v1"
RECEIPT_SCHEMA = "madeleine.oracle-window-pixel-crops-complete.v1"


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


def downsample_rgb_area(frames: np.ndarray, *, output_size: int) -> np.ndarray:
    """Integer area-average NHWC uint8 RGB frames without interpolation drift."""

    value = np.asarray(frames)
    if value.dtype != np.uint8 or value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError("frames must be uint8 [N,H,W,3]")
    if value.shape[1] != value.shape[2] or value.shape[1] % output_size:
        raise ValueError("square input size must be divisible by output_size")
    factor = value.shape[1] // output_size
    blocked = value.reshape(
        len(value), output_size, factor, output_size, factor, 3
    ).astype(np.uint32)
    summed = blocked.sum(axis=(2, 4), dtype=np.uint32)
    divisor = factor * factor
    return ((summed + divisor // 2) // divisor).astype(np.uint8)


def extract_crops(
    frames: np.ndarray,
    examples: Sequence[OracleExample],
    *,
    crop_frames: int,
    output_size: int,
    batch_examples: int = 8,
) -> np.ndarray:
    """Extract and downsample examples while bounding temporary raw-pixel memory."""

    if batch_examples < 1:
        raise ValueError("batch_examples must be positive")
    output = np.empty(
        (len(examples), crop_frames, output_size, output_size, 3), dtype=np.uint8
    )
    for start in range(0, len(examples), batch_examples):
        rows = examples[start : start + batch_examples]
        indices = np.stack(
            [
                np.arange(row.crop_start, row.crop_start + crop_frames, dtype=np.int64)
                for row in rows
            ]
        )
        if np.any(indices < 0) or np.any(indices >= len(frames)):
            raise ValueError("oracle crop escaped its source shard")
        selected = frames[indices.reshape(-1)]
        reduced = downsample_rgb_area(selected, output_size=output_size)
        output[start : start + len(rows)] = reduced.reshape(
            len(rows), crop_frames, output_size, output_size, 3
        )
    return output


def _archive_arrays(examples: Sequence[OracleExample], rgb: np.ndarray) -> dict[str, np.ndarray]:
    arrays = {"rgb": rgb, **_example_arrays(examples)}
    if len(rgb) != len(examples):
        raise ValueError("pixel crop count does not match example count")
    return arrays


def validate_cache_archive(
    path: Path,
    *,
    expected_examples: Sequence[OracleExample],
    crop_frames: int,
    output_size: int,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        expected_names = {"rgb", *_example_arrays(expected_examples)}
        if set(archive.files) != expected_names:
            raise ValueError("pixel cache array inventory changed")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    rgb = arrays.pop("rgb")
    expected_shape = (
        len(expected_examples),
        crop_frames,
        output_size,
        output_size,
        3,
    )
    if rgb.dtype != np.uint8 or rgb.shape != expected_shape:
        raise ValueError(f"rgb must be uint8 {expected_shape}")
    expected_metadata = _example_arrays(expected_examples)
    for name, expected in expected_metadata.items():
        observed = arrays[name]
        if observed.dtype != expected.dtype or not np.array_equal(observed, expected):
            raise ValueError(f"pixel cache metadata changed: {name}")
    return {
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "rgb_sha256": _array_sha256(rgb),
        "examples": len(expected_examples),
        "rgb_shape": list(rgb.shape),
        "rgb_dtype": str(rgb.dtype),
    }


def _source_authority(feature_receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    content = feature_receipt.get("content")
    if not isinstance(content, Mapping):
        raise ValueError("feature receipt lacks content authority")
    source = content.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("feature receipt lacks source authority")
    sessions = source.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("feature receipt lacks source sessions")
    result: dict[str, Mapping[str, Any]] = {}
    for row in sessions:
        if not isinstance(row, Mapping):
            raise ValueError("invalid source session receipt")
        session_id = str(row.get("session_id"))
        result[session_id] = row
    return result


def build_pixel_cache(
    *,
    source_root: Path,
    feature_root: Path,
    feature_receipt_path: Path,
    base_config_path: Path,
    dataset_manifest_path: Path,
    output: Path,
    receipt_path: Path,
    output_size: int,
) -> dict[str, Any]:
    for target in (output, receipt_path):
        if os.path.lexists(target):
            raise ValueError(f"refusing to overwrite pixel-cache artifact: {target}")
    config = _json(base_config_path)
    if config.get("status") != "preregistered_before_validation_inference":
        raise ValueError("base oracle decision config is not frozen")
    dataset_config = config["dataset"]
    if output_size != 32:
        raise ValueError("the bounded follow-up freezes 32x32 pixel crops")
    crop_frames = int(dataset_config["crop_frames"])
    if crop_frames != 32:
        raise ValueError("base oracle crop length changed")
    forbidden = set(str(value) for value in dataset_config["forbidden_sessions"])
    if forbidden != set(FORBIDDEN_SESSION_IDS):
        raise ValueError("forbidden-session contract changed")

    feature_receipt = _json(feature_receipt_path)
    if sha256_file(feature_receipt_path) != str(dataset_config["feature_receipt_sha256"]):
        raise ValueError("canonical feature receipt hash changed")
    if feature_receipt.get("status") != "complete":
        raise ValueError("canonical feature receipt is incomplete")
    if Path(str(feature_receipt.get("source_root"))).resolve() != source_root.resolve():
        raise ValueError("feature receipt points to a different pixel source")
    source_manifest_path = source_root / "build_manifest.json"
    if sha256_file(source_manifest_path) != str(dataset_config["source_build_manifest_sha256"]):
        raise ValueError("source build manifest hash changed")
    source_manifest = _json(source_manifest_path)
    if int(source_manifest.get("frame_size", -1)) != 128:
        raise ValueError("source frame size changed")
    expected_masks = {"frame_index_strip", "input_overlay"}
    for row in source_manifest.get("sessions", []):
        if set(row.get("masked_regions", [])) != expected_masks:
            raise ValueError(f"answer-key masks changed for {row.get('session_id')}")

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
    if rebuilt_manifest != _json(dataset_manifest_path):
        raise ValueError("base oracle dataset manifest did not rebuild exactly")

    authority = _source_authority(feature_receipt)
    split_examples = {"train": train_examples, "validation": val_examples}
    cache_arrays: dict[str, dict[str, np.ndarray]] = {}
    source_checks: dict[str, Any] = {}
    for split, examples in split_examples.items():
        rgb = np.empty(
            (len(examples), crop_frames, output_size, output_size, 3), dtype=np.uint8
        )
        filled = np.zeros(len(examples), dtype=bool)
        for session_id in sorted({row.session_id for row in examples}):
            if session_id in forbidden:
                raise ValueError(f"forbidden session reached pixel loading: {session_id}")
            path = source_root / f"{session_id}.npz"
            expected_source = authority.get(session_id)
            if expected_source is None:
                raise ValueError(f"source authority missing session: {session_id}")
            source_sha = sha256_file(path)
            if source_sha != str(expected_source.get("source_npz_sha256")):
                raise ValueError(f"source NPZ hash changed: {session_id}")
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "frames",
                    "keys",
                    "engine_frame_idx",
                    "input_active",
                    "session_id",
                }
                if set(archive.files) != required:
                    raise ValueError(f"source array inventory changed: {session_id}")
                frames = np.asarray(archive["frames"])
                keys = np.asarray(archive["keys"])
                engine = np.asarray(archive["engine_frame_idx"])
                active = np.asarray(archive["input_active"])
                stored_id = str(np.asarray(archive["session_id"]).reshape(()).item())
            feature_session = sessions[session_id]
            if stored_id != session_id:
                raise ValueError(f"stored source session ID changed: {session_id}")
            if frames.dtype != np.uint8 or frames.shape != (len(keys), 128, 128, 3):
                raise ValueError(f"source frames changed shape or dtype: {session_id}")
            if not (
                np.array_equal(keys, feature_session.keys)
                and np.array_equal(engine, feature_session.engine_frame_idx)
                and np.array_equal(active, feature_session.input_active)
            ):
                raise ValueError(f"source supervision differs from features: {session_id}")
            frame_sha = _array_sha256(frames)
            expected_arrays = expected_source.get("arrays")
            if not isinstance(expected_arrays, Mapping) or frame_sha != str(
                expected_arrays.get("frames_sha256")
            ):
                raise ValueError(f"source frame-content hash changed: {session_id}")
            positions = [
                index for index, row in enumerate(examples) if row.session_id == session_id
            ]
            selected_examples = [examples[index] for index in positions]
            reduced = extract_crops(
                frames,
                selected_examples,
                crop_frames=crop_frames,
                output_size=output_size,
            )
            rgb[np.asarray(positions)] = reduced
            filled[np.asarray(positions)] = True
            source_checks[session_id] = {
                "source_npz_sha256": source_sha,
                "frames_sha256": frame_sha,
                "supervision_equal_to_features": True,
                "masked_regions": sorted(expected_masks),
            }
        if not filled.all():
            raise AssertionError(f"pixel cache did not fill every {split} example")
        cache_arrays[split] = _archive_arrays(examples, rgb)

    staging = output.with_name(f".{output.name}.staging")
    if os.path.lexists(staging):
        raise ValueError(f"stale pixel-cache staging path exists: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for split, arrays in cache_arrays.items():
            np.savez_compressed(staging / f"{split}.npz", **arrays)
        cache_receipts = {
            split: validate_cache_archive(
                staging / f"{split}.npz",
                expected_examples=examples,
                crop_frames=crop_frames,
                output_size=output_size,
            )
            for split, examples in split_examples.items()
        }
        manifest = {
            "schema_version": CACHE_SCHEMA,
            "base_config_sha256": sha256_file(base_config_path),
            "base_dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "feature_receipt_sha256": sha256_file(feature_receipt_path),
            "source_build_manifest_sha256": sha256_file(source_manifest_path),
            "crop_frames": crop_frames,
            "source_frame_size": 128,
            "output_size": output_size,
            "downsampling": "per-channel integer 4x4 area mean with round-half-up",
            "color_order": "source RGB preserved",
            "cache": cache_receipts,
            "source_checks": source_checks,
        }
        manifest_path = staging / "cache_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if _json(manifest_path) != manifest:
            raise ValueError("serialized pixel-cache manifest changed")
        staging.replace(output)
    except Exception:
        # Preserve staging for diagnosis; never silently reuse it.
        raise

    final_manifest_path = output / "cache_manifest.json"
    receipt_content = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        "published_output": str(output.resolve()),
        "base_config_sha256": sha256_file(base_config_path),
        "base_dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "feature_receipt_sha256": sha256_file(feature_receipt_path),
        "cache_manifest_sha256": sha256_file(final_manifest_path),
        "cache": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": sha256_file(output / name),
            }
            for name in ("cache_manifest.json", "train.npz", "validation.npz")
        },
        "checks": {
            "source_pixels_hash_matched": True,
            "answer_key_masks_recorded": True,
            "supervision_equal_to_features": True,
            "base_dataset_manifest_rebuilt_exactly": True,
            "cache_metadata_equal_to_oracle_examples": True,
            "serialized_arrays_reloaded": True,
            "forbidden_sessions_absent": True,
        },
    }
    receipt_content["content_sha256"] = _canonical_sha256(receipt_content)
    temporary_receipt = receipt_path.with_name(f".{receipt_path.name}.tmp")
    if os.path.lexists(temporary_receipt):
        raise ValueError(f"stale pixel-cache receipt temporary exists: {temporary_receipt}")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt.write_text(
        json.dumps(receipt_content, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if _json(temporary_receipt) != receipt_content:
        raise ValueError("serialized pixel-cache receipt changed")
    temporary_receipt.replace(receipt_path)
    return receipt_content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--feature-root", required=True, type=Path)
    parser.add_argument("--feature-receipt", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output-size", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    receipt = build_pixel_cache(
        source_root=args.source_root,
        feature_root=args.feature_root,
        feature_receipt_path=args.feature_receipt,
        base_config_path=args.base_config,
        dataset_manifest_path=args.dataset_manifest,
        output=args.out,
        receipt_path=args.receipt,
        output_size=args.output_size,
    )
    print(json.dumps({"status": receipt["status"], "receipt": str(args.receipt)}))


if __name__ == "__main__":
    main()
