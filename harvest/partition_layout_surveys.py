"""Freeze and balance a layout-survey wave from raw-complete nominations.

The input IDs are an immutable snapshot of R2 ``upload_complete.json``
prefixes.  Emitted rows remain machine nominations: this schedules sparse
layout evidence and never implies human review or training admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_WAVE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {line_number} is not valid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}: line {line_number} is not an object")
        rows.append(row)
    return rows


def nominal_hours(row: dict[str, Any], label: str) -> float:
    value = row.get("nominal_hours")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} has missing or non-numeric nominal_hours")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} has non-finite or negative nominal_hours")
    return result


def unique_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = str(row.get("video_id", ""))
        if not _SAFE_ID.fullmatch(video_id):
            raise ValueError(f"{label} contains an unsafe video_id")
        if video_id in indexed:
            raise ValueError(f"{label} contains duplicate video_id {video_id}")
        nominal_hours(row, f"{label} row {video_id}")
        indexed[video_id] = row
    return indexed


def load_completed_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("raw-complete snapshot contains duplicate video IDs")
    if any(not _SAFE_ID.fullmatch(video_id) for video_id in ids):
        raise ValueError("raw-complete snapshot contains an unsafe video ID")
    return sorted(ids)


def build_wave(
    nominations: list[dict[str, Any]],
    completed_ids: set[str],
    excluded_ids: set[str],
) -> list[dict[str, Any]]:
    indexed = unique_rows(nominations, "nominations")
    eligible = sorted(completed_ids.intersection(indexed).difference(excluded_ids))
    rows: list[dict[str, Any]] = []
    for video_id in eligible:
        row = dict(indexed[video_id])
        if row.get("human_reviewed") is not False:
            raise ValueError(f"nomination {video_id} must remain human_reviewed=false")
        if row.get("training_admitted") not in (None, False):
            raise ValueError(f"nomination {video_id} is unexpectedly training-admitted")
        row.update({
            "survey_reason": (
                "raw-complete machine nominee; full-source AI layout stability triage"
            ),
            "human_reviewed": False,
            "training_admitted": False,
        })
        rows.append(row)
    return rows


def balance_by_nominal_hours(
    rows: list[dict[str, Any]], shard_count: int
) -> list[list[dict[str, Any]]]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    hours = [0.0] * shard_count
    ordered = sorted(
        rows,
        key=lambda row: (-nominal_hours(row, str(row["video_id"])), row["video_id"]),
    )
    for row in ordered:
        index = min(range(shard_count), key=lambda value: (hours[value], value))
        shards[index].append(row)
        hours[index] += nominal_hours(row, str(row["video_id"]))
    for shard in shards:
        shard.sort(key=lambda row: row["video_id"])
    return shards


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(jsonl_text(rows))


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_immutable_files(files: dict[Path, bytes]) -> None:
    """Publish complete files atomically, refusing any conflicting prior wave."""

    conflicts = [
        path for path, content in files.items()
        if path.exists() and (not path.is_file() or path.read_bytes() != content)
    ]
    if conflicts:
        raise FileExistsError(
            f"refusing to overwrite immutable wave artifact: {conflicts[0]}"
        )
    for path, content in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if not path.is_file() or path.read_bytes() != content:
                    raise FileExistsError(
                        f"concurrent immutable wave conflict: {path}"
                    ) from None
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominations", type=Path, required=True)
    parser.add_argument("--raw-complete-ids", type=Path, required=True)
    parser.add_argument("--exclude-queue", type=Path, action="append", default=[])
    parser.add_argument(
        "--exclude-ids",
        type=Path,
        action="append",
        default=[],
        help="newline-delimited video IDs to exclude from this immutable wave",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--wave", required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if not _SAFE_WAVE.fullmatch(args.wave):
        parser.error("wave must be a safe filename component")

    completed = load_completed_ids(args.raw_complete_ids)
    exclusions: set[str] = set()
    exclusion_inputs = []
    for path in args.exclude_queue:
        rows = load_jsonl(path)
        ids = set(unique_rows(rows, str(path)))
        overlap = exclusions.intersection(ids)
        if overlap:
            raise ValueError(f"excluded queues overlap: {sorted(overlap)[:3]}")
        exclusions.update(ids)
        exclusion_inputs.append({"path": str(path), "sha256": sha256_file(path)})
    for path in args.exclude_ids:
        ids = set(load_completed_ids(path))
        overlap = exclusions.intersection(ids)
        if overlap:
            raise ValueError(f"excluded inputs overlap: {sorted(overlap)[:3]}")
        exclusions.update(ids)
        exclusion_inputs.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "kind": "video_id_snapshot",
        })

    nominations = load_jsonl(args.nominations)
    rows = build_wave(nominations, set(completed), exclusions)
    shards = balance_by_nominal_hours(rows, args.shards)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = args.out_dir / f"layout-survey-{args.wave}.raw-complete-ids.txt"
    snapshot_bytes = "".join(video_id + "\n" for video_id in completed).encode()
    master_path = args.out_dir / f"layout-survey-{args.wave}-master.jsonl"
    master_bytes = jsonl_text(rows).encode()
    artifacts: dict[Path, bytes] = {
        snapshot_path: snapshot_bytes,
        master_path: master_bytes,
    }

    shard_rows = []
    for index, shard in enumerate(shards, start=1):
        path = args.out_dir / f"layout-survey-{args.wave}-batch-{index:02d}.jsonl"
        content = jsonl_text(shard).encode()
        artifacts[path] = content
        shard_rows.append({
            "index": index,
            "path": path.name,
            "rows": len(shard),
            "nominal_hours": sum(
                nominal_hours(row, str(row["video_id"])) for row in shard
            ),
            "sha256": sha256_bytes(content),
        })

    manifest = {
        "schema_version": 1,
        "wave": args.wave,
        "semantics": "machine-only sparse layout survey scheduling",
        "human_reviewed": False,
        "training_admitted": False,
        "nominations": {
            "path": str(args.nominations),
            "rows": len(nominations),
            "sha256": sha256_file(args.nominations),
        },
        "raw_complete_snapshot": {
            "path": snapshot_path.name,
            "rows": len(completed),
            "sha256": sha256_bytes(snapshot_bytes),
        },
        "excluded_inputs": exclusion_inputs,
        "excluded_video_ids": len(exclusions),
        "eligible_rows": len(rows),
        "eligible_nominal_hours": sum(
            nominal_hours(row, str(row["video_id"])) for row in rows
        ),
        "master": {
            "path": master_path.name,
            "sha256": sha256_bytes(master_bytes),
        },
        "partition_method": "largest-nominal-hours-first greedy balance",
        "shards": shard_rows,
    }
    manifest_path = args.out_dir / f"layout-survey-{args.wave}.manifest.json"
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    artifacts[manifest_path] = manifest_bytes
    write_immutable_files(artifacts)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
