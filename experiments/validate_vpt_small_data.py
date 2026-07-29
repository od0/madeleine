#!/usr/bin/env python3
"""Deeply validate a content-bound VPT-small derived-data generation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


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


def verify_content_hash(value: dict[str, Any], *, name: str) -> None:
    declared = value.get("content_sha256")
    payload = copy.deepcopy(value)
    payload.pop("content_sha256", None)
    if canonical_sha256(payload) != declared:
        raise RuntimeError(f"{name}: content_sha256 mismatch")


def validate(root: Path) -> dict[str, Any]:
    manifest_path = root / "build_manifest.json"
    marker_path = root / "complete.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "madeleine.vpt-small-20hz-shards.v1":
        raise ValueError("unexpected VPT data manifest schema")
    if marker.get("schema_version") != "madeleine.vpt-small-20hz-complete.v1":
        raise ValueError("unexpected VPT data marker schema")
    if sha256_file(manifest_path) != marker["manifest"]["sha256"]:
        raise RuntimeError("completion marker does not bind manifest bytes")
    if manifest["content_sha256"] != marker["manifest"]["content_sha256"]:
        raise RuntimeError("completion marker does not bind manifest content")
    verify_content_hash(manifest, name="manifest")

    rows = 0
    windows = 0
    streams = 0
    for record in manifest["records"]:
        streams += 1
        directory = root / f"{record['session_id']}__p{record['phase']}"
        metadata_path = directory / "metadata.json"
        if sha256_file(metadata_path) != record["metadata_file"]["sha256"]:
            raise RuntimeError(f"metadata hash mismatch: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        verify_content_hash(metadata, name=str(metadata_path))
        arrays: dict[str, np.ndarray] = {}
        for name, receipt in metadata["arrays"].items():
            path = directory / receipt["file"]
            if path.stat().st_size != int(receipt["bytes"]):
                raise RuntimeError(f"size mismatch: {path}")
            if sha256_file(path) != receipt["sha256"]:
                raise RuntimeError(f"sha256 mismatch: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != receipt["shape"] or str(array.dtype) != receipt["dtype"]:
                raise RuntimeError(f"shape/dtype mismatch: {path}")
            arrays[name] = array
        expected_rows = int(record["derived_rows"])
        if arrays["frames"].shape != (expected_rows, 128, 128, 3):
            raise ValueError(f"invalid frame shape: {directory}")
        if arrays["frames"].dtype != np.uint8:
            raise ValueError(f"invalid frame dtype: {directory}")
        if arrays["keys"].shape != (expected_rows, 7):
            raise ValueError(f"invalid key shape: {directory}")
        if not np.all((arrays["keys"] == 0) | (arrays["keys"] == 1)):
            raise ValueError(f"nonbinary keys: {directory}")
        if not np.all((arrays["input_active"] == 0) | (arrays["input_active"] == 1)):
            raise ValueError(f"nonbinary active mask: {directory}")
        continuity = np.asarray(arrays["continuity_id"])
        engine = np.asarray(arrays["source_engine_frame_idx"])
        source_rows = np.asarray(arrays["source_row_index"])
        same = continuity[1:] == continuity[:-1]
        if not np.all(np.diff(engine)[same] == 3):
            raise ValueError(f"derived engine spacing is not 20 Hz: {directory}")
        if not np.all(np.diff(source_rows)[same] == 3):
            raise ValueError(f"derived source-row spacing is not 3: {directory}")
        starts = np.asarray(arrays["window_start"])
        for start in starts:
            stop = int(start) + 128
            if stop > expected_rows:
                raise ValueError(f"window exceeds stream: {directory}")
            if np.any(continuity[int(start):stop] != continuity[int(start)]):
                raise ValueError(f"window crosses continuity boundary: {directory}")
        if len(starts) != int(record["windows"]):
            raise RuntimeError(f"window count mismatch: {directory}")
        rows += expected_rows
        windows += len(starts)

    if rows != int(manifest["totals"]["derived_rows"]):
        raise RuntimeError("derived-row total mismatch")
    if windows != int(manifest["totals"]["windows"]):
        raise RuntimeError("window total mismatch")
    if streams != int(manifest["totals"]["derived_streams"]):
        raise RuntimeError("stream total mismatch")
    return {
        "ok": True,
        "root": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "content_sha256": manifest["content_sha256"],
        "derived_streams": streams,
        "derived_rows": rows,
        "windows": windows,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate(args.root)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
