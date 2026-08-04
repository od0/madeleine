#!/usr/bin/env python3
"""Add deterministic tail-aligned full-context windows to VPT derived data."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.build_vpt_small_128px_20hz import canonical_sha256, sha256_file
from experiments.validate_vpt_small_data import validate


OVERLAP_POLICY = "base-first-stable-tail-fill"


def _array_receipt(path: Path, value: np.ndarray) -> dict[str, Any]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }


def _tail_starts(continuity: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    boundaries = np.flatnonzero(continuity[1:] != continuity[:-1]) + 1
    runs = zip(np.r_[0, boundaries], np.r_[boundaries, len(continuity)], strict=True)
    existing = set(int(value) for value in starts)
    tails = [int(end - window) for begin, end in runs if end - begin >= window and int(end - window) not in existing]
    return np.asarray(tails, dtype=np.int64)


def complete_tail_windows(source_root: Path, output_root: Path) -> dict[str, Any]:
    validation = validate(source_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing nonempty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_manifest_path = source_root / "build_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("center_overlap_policy") is not None:
        raise ValueError("source generation already declares a center-overlap policy")
    window = int(source_manifest["window"])
    stride = int(source_manifest["stride"])
    records = []
    for source_record in source_manifest["records"]:
        name = f"{source_record['session_id']}__p{source_record['phase']}"
        source_dir = source_root / name
        output_dir = output_root / name
        output_dir.mkdir()
        metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
        arrays = copy.deepcopy(metadata["arrays"])
        for array_name, receipt in arrays.items():
            if array_name == "window_start":
                continue
            os.link(source_dir / receipt["file"], output_dir / receipt["file"])
        continuity = np.load(source_dir / arrays["continuity_id"]["file"], mmap_mode="r", allow_pickle=False)
        base_starts = np.load(source_dir / arrays["window_start"]["file"], allow_pickle=False).astype(np.int64)
        tails = _tail_starts(np.asarray(continuity), base_starts, window)
        completed_starts = np.concatenate((base_starts, tails))
        starts_path = output_dir / arrays["window_start"]["file"]
        np.save(starts_path, completed_starts, allow_pickle=False)
        arrays["window_start"] = _array_receipt(starts_path, completed_starts)
        record = copy.deepcopy(source_record)
        record.pop("metadata_file", None)
        record.pop("content_sha256", None)
        record["arrays"] = arrays
        record["windows"] = len(completed_starts)
        record["center_overlap_policy"] = OVERLAP_POLICY
        record["tail_completion"] = {
            "base_windows": len(base_starts),
            "tail_windows": len(tails),
            "tail_start_rule": "end_of_continuity_run_minus_window",
            "retained_positions": [(window - stride) // 2, (window + stride) // 2],
        }
        record["content_sha256"] = canonical_sha256(record)
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        record["metadata_file"] = {
            "file": metadata_path.name,
            "bytes": metadata_path.stat().st_size,
            "sha256": sha256_file(metadata_path),
        }
        records.append(record)

    manifest = copy.deepcopy(source_manifest)
    manifest.pop("content_sha256", None)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["source_generation"] = {
        "root": str(source_root),
        "manifest_sha256": sha256_file(source_manifest_path),
        "content_sha256": source_manifest["content_sha256"],
    }
    manifest["center_overlap_policy"] = OVERLAP_POLICY
    manifest["records"] = records
    manifest["totals"]["windows"] = sum(int(record["windows"]) for record in records)
    manifest["content_sha256"] = canonical_sha256(manifest)
    manifest_path = output_root / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker = {
        "schema_version": "madeleine.vpt-small-20hz-complete.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "file": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "content_sha256": manifest["content_sha256"],
        },
        "derived_streams": len(records),
        "derived_rows": int(manifest["totals"]["derived_rows"]),
        "windows": int(manifest["totals"]["windows"]),
        "window": window,
        "center_overlap_policy": OVERLAP_POLICY,
    }
    marker_path = output_root / "complete.json"
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = validate(output_root)
    result["source_validation"] = validation
    result["tail_windows"] = sum(record["tail_completion"]["tail_windows"] for record in records)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = complete_tail_windows(args.source_root, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
