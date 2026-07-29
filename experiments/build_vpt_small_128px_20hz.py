#!/usr/bin/env python3
"""Derive hash-bound 20 Hz VPT streams from canonical 60 Hz 128px shards.

The input shards already contain the exact mapped labels and 128x128 RGB rows
used by the Tier-B end-to-end study.  This builder performs no video decode and
no label reconstruction: it only phase-subsamples within contiguous engine
runs and writes mmap-friendly arrays plus a content-bound manifest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from badeline.train import contiguous_runs, read_session_ids
from data.schema import KEY_ORDER


SCHEMA = "madeleine.vpt-small-20hz-shards.v1"
MARKER_SCHEMA = "madeleine.vpt-small-20hz-complete.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_phases(raw: str) -> tuple[int, ...]:
    phases = tuple(int(item) for item in raw.split(",") if item.strip())
    if not phases or any(phase not in (0, 1, 2) for phase in phases):
        raise ValueError("phases must be a comma-separated subset of 0,1,2")
    if len(phases) != len(set(phases)):
        raise ValueError("phases contain duplicates")
    return phases


def load_expected_hashes(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "shard_sha256" in raw:
        raw = raw["shard_sha256"]
    if not isinstance(raw, dict):
        raise ValueError("expected hash JSON must be a mapping or run_meta")
    return {str(key).removesuffix(".npz"): str(value) for key, value in raw.items()}


def load_source_shard(
    path: Path, session_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"frames", "keys", "engine_frame_idx", "input_active", "session_id"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        frames = archive["frames"]
        keys = archive["keys"]
        engine_frame_idx = archive["engine_frame_idx"]
        input_active = archive["input_active"]
        stored_id = str(archive["session_id"].reshape(()).item())
    if stored_id != session_id:
        raise ValueError(f"{path}: session id {stored_id!r} != {session_id!r}")
    if frames.dtype != np.uint8 or frames.shape[1:] != (128, 128, 3):
        raise ValueError(f"{path}: frames must be uint8 [N,128,128,3]")
    if keys.dtype != np.uint8 or keys.shape != (len(frames), len(KEY_ORDER)):
        raise ValueError(f"{path}: keys must be uint8 [N,{len(KEY_ORDER)}]")
    if engine_frame_idx.dtype != np.int64 or engine_frame_idx.shape != (len(frames),):
        raise ValueError(f"{path}: engine_frame_idx must be int64 [N]")
    if input_active.dtype != np.uint8 or input_active.shape != (len(frames),):
        raise ValueError(f"{path}: input_active must be uint8 [N]")
    if not np.all((keys == 0) | (keys == 1)):
        raise ValueError(f"{path}: keys are not binary")
    if not np.all((input_active == 0) | (input_active == 1)):
        raise ValueError(f"{path}: input_active is not binary")
    return frames, keys, engine_frame_idx, input_active


def selected_rows(
    engine_frame_idx: np.ndarray, phase: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    rows: list[np.ndarray] = []
    continuity: list[np.ndarray] = []
    output_runs: list[tuple[int, int]] = []
    cursor = 0
    for run_number, (start, end) in enumerate(contiguous_runs(engine_frame_idx)):
        chosen = np.arange(start + phase, end, 3, dtype=np.int64)
        if not len(chosen):
            continue
        rows.append(chosen)
        continuity.append(np.full(len(chosen), run_number, dtype=np.int32))
        output_runs.append((cursor, cursor + len(chosen)))
        cursor += len(chosen)
    if not rows:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32), []
    return np.concatenate(rows), np.concatenate(continuity), output_runs


def write_array(directory: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = directory / f"{name}.npy"
    np.save(path, value, allow_pickle=False)
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }


def derive_one(
    *,
    source_root: Path,
    output_root: Path,
    session_id: str,
    phase: int,
    expected_hash: str | None,
    window: int,
    stride: int,
) -> dict[str, Any]:
    source = source_root / f"{session_id}.npz"
    if not source.is_file():
        raise FileNotFoundError(f"missing declared source shard: {source}")
    source_hash = sha256_file(source)
    if expected_hash is not None and source_hash != expected_hash:
        raise RuntimeError(
            f"{source}: sha256 {source_hash} != declared {expected_hash}"
        )
    frames, keys, engine_frame_idx, input_active = load_source_shard(source, session_id)
    rows, continuity_id, runs = selected_rows(engine_frame_idx, phase)
    if not len(rows):
        raise RuntimeError(f"{source}: phase {phase} selects no rows")
    target = output_root / f"{session_id}__p{phase}"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite derived shard: {target}")
    target.mkdir(parents=True)
    windows = np.concatenate(
        [
            np.arange(start, end - window + 1, stride, dtype=np.int64)
            for start, end in runs
            if end - start >= window
        ]
    ) if any(end - start >= window for start, end in runs) else np.empty(0, dtype=np.int64)
    arrays = {
        "frames": write_array(target, "frames", frames[rows]),
        "keys": write_array(target, "keys", keys[rows]),
        "input_active": write_array(target, "input_active", input_active[rows]),
        "source_engine_frame_idx": write_array(
            target, "source_engine_frame_idx", engine_frame_idx[rows]
        ),
        "source_row_index": write_array(target, "source_row_index", rows),
        "continuity_id": write_array(target, "continuity_id", continuity_id),
        "window_start": write_array(target, "window_start", windows),
    }
    metadata = {
        "schema_version": SCHEMA,
        "session_id": session_id,
        "phase": phase,
        "source": {
            "filename": source.name,
            "bytes": source.stat().st_size,
            "sha256": source_hash,
            "source_rows": len(frames),
        },
        "derived_rows": len(rows),
        "windows": len(windows),
        "contiguous_runs": len(runs),
        "window": window,
        "stride": stride,
        "arrays": arrays,
    }
    metadata["content_sha256"] = canonical_sha256(metadata)
    metadata_path = target / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata["metadata_file"] = {
        "file": metadata_path.name,
        "bytes": metadata_path.stat().st_size,
        "sha256": sha256_file(metadata_path),
    }
    return metadata


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-hashes", type=Path)
    parser.add_argument("--phases", default="0")
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.window != 128 or args.stride != 64:
        raise ValueError("production VPT-small derivation requires window 128, stride 64")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing nonempty output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    sessions = read_session_ids(args.sessions)
    phases = parse_phases(args.phases)
    expected = load_expected_hashes(args.expected_hashes)
    if expected:
        absent = sorted(set(sessions).difference(expected))
        if absent:
            raise ValueError(f"declared sessions missing expected hashes: {absent}")

    records = [
        derive_one(
            source_root=args.source_root,
            output_root=args.output_root,
            session_id=session_id,
            phase=phase,
            expected_hash=expected.get(session_id),
            window=args.window,
            stride=args.stride,
        )
        for session_id in sessions
        for phase in phases
    ]
    manifest = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source_root),
        "sessions_file": {
            "path": str(args.sessions),
            "sha256": sha256_file(args.sessions),
            "sessions": sessions,
        },
        "phases": list(phases),
        "source_rate_hz": 60,
        "derived_rate_hz": 20,
        "phase_rule": "within each contiguous engine-frame run, rows phase, phase+3, ...",
        "window": args.window,
        "stride": args.stride,
        "key_order": list(KEY_ORDER),
        "records": records,
        "totals": {
            "source_sessions": len(sessions),
            "derived_streams": len(records),
            "derived_rows": sum(int(record["derived_rows"]) for record in records),
            "windows": sum(int(record["windows"]) for record in records),
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    manifest_path = args.output_root / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker = {
        "schema_version": MARKER_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "file": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "content_sha256": manifest["content_sha256"],
        },
        "derived_streams": len(records),
        "derived_rows": manifest["totals"]["derived_rows"],
        "windows": manifest["totals"]["windows"],
    }
    marker_path = args.output_root / "complete.json"
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(marker, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
