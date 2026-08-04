#!/usr/bin/env python3
"""Freeze a deterministic mostly-replay manifest for targeted down fine-tuning."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from data.schema import KEY_ORDER
from experiments.build_vpt_small_128px_20hz import canonical_sha256, sha256_file


SCHEMA = "madeleine.vpt-small-20hz-shards.v1"
RECEIPT_SCHEMA = "madeleine.vpt-small-down-finetune-mix.v1"


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA:
        raise ValueError(f"unsupported VPT-small manifest: {path}")
    if (int(value.get("window", -1)), int(value.get("stride", -1))) != (128, 64):
        raise ValueError(f"unexpected window geometry: {path}")
    if list(value.get("key_order", [])) != list(KEY_ORDER):
        raise ValueError(f"unexpected key order: {path}")
    if set(value.get("phases", [])) != {0}:
        raise ValueError(f"fine-tune training inputs must contain phase 0 only: {path}")
    return value


def record_order(record: dict[str, Any], *, salt: str) -> tuple[str, str]:
    session_id = str(record["session_id"])
    digest = hashlib.sha256(f"{salt}\0{session_id}".encode()).hexdigest()
    return digest, session_id


def select_replay_records(
    replay_records: list[dict[str, Any]],
    *,
    targeted_windows: int,
    maximum_targeted_fraction: float,
    salt: str,
) -> list[dict[str, Any]]:
    if not 0.0 < maximum_targeted_fraction < 1.0:
        raise ValueError(
            "maximum targeted-capture fraction must be strictly between zero and one"
        )
    target_replay_windows = math.ceil(
        targeted_windows
        * (1.0 - maximum_targeted_fraction)
        / maximum_targeted_fraction
    )
    selected: list[dict[str, Any]] = []
    selected_windows = 0
    for record in sorted(replay_records, key=lambda row: record_order(row, salt=salt)):
        selected.append(record)
        selected_windows += int(record["windows"])
        if selected_windows >= target_replay_windows:
            return selected
    raise ValueError(
        "replay manifest cannot satisfy the requested targeted-capture fraction"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument(
        "--targeted-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maximum-targeted-fraction", type=float, default=0.05)
    parser.add_argument("--salt", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing existing output directory: {args.out}")
    replay = load_manifest(args.replay_manifest)
    targeted_manifests = [load_manifest(path) for path in args.targeted_manifest]
    targeted_records = [
        record for manifest in targeted_manifests for record in manifest["records"]
    ]
    targeted_record_ids = [str(record["session_id"]) for record in targeted_records]
    if len(targeted_record_ids) != len(set(targeted_record_ids)):
        raise ValueError("targeted manifests contain duplicate session IDs")
    targeted_windows = sum(int(record["windows"]) for record in targeted_records)
    if targeted_windows < 1:
        raise ValueError("targeted manifest contains no training windows")
    selected_replay = select_replay_records(
        list(replay["records"]),
        targeted_windows=targeted_windows,
        maximum_targeted_fraction=args.maximum_targeted_fraction,
        salt=args.salt,
    )
    replay_ids = {str(record["session_id"]) for record in selected_replay}
    targeted_ids = {str(record["session_id"]) for record in targeted_records}
    overlap = replay_ids.intersection(targeted_ids)
    if overlap:
        raise ValueError(f"replay/targeted session overlap: {sorted(overlap)}")
    records = selected_replay + targeted_records
    replay_windows = sum(int(record["windows"]) for record in selected_replay)
    total_windows = replay_windows + targeted_windows
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": SCHEMA,
        "created_at": created_at,
        "source_root": "staged immutable replay and targeted-capture objects",
        "sessions_file": {
            "path": "selected_sessions.txt",
            "sessions": [str(record["session_id"]) for record in records],
        },
        "sources": {
            "replay_manifest": {
                "path": str(args.replay_manifest),
                "sha256": sha256_file(args.replay_manifest),
                "content_sha256": replay["content_sha256"],
            },
            "targeted_manifests": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "content_sha256": manifest["content_sha256"],
                }
                for path, manifest in zip(
                    args.targeted_manifest, targeted_manifests, strict=True
                )
            ],
        },
        "selection": {
            "unit": "whole_phase0_stream",
            "algorithm": "sha256(salt + NUL + session_id), ascending until replay-window target",
            "salt": args.salt,
            "maximum_targeted_fraction": args.maximum_targeted_fraction,
            "observed_targeted_fraction": targeted_windows / total_windows,
            "replay_windows": replay_windows,
            "targeted_windows": targeted_windows,
        },
        "phases": [0],
        "source_rate_hz": 60,
        "derived_rate_hz": 20,
        "phase_rule": "within each contiguous engine-frame run, rows phase, phase+3, ...",
        "window": 128,
        "stride": 64,
        "key_order": list(KEY_ORDER),
        "records": records,
        "totals": {
            "source_sessions": len(records),
            "derived_streams": len(records),
            "derived_rows": sum(int(record["derived_rows"]) for record in records),
            "windows": total_windows,
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    args.out.mkdir(parents=True)
    manifest_path = args.out / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out / "selected_replay_sessions.txt").write_text(
        "".join(f"{record['session_id']}\n" for record in selected_replay),
        encoding="utf-8",
    )
    (args.out / "selected_targeted_sessions.txt").write_text(
        "".join(f"{record['session_id']}\n" for record in targeted_records),
        encoding="utf-8",
    )
    inventory_rows: list[dict[str, Any]] = []
    for record in records:
        directory = f"{record['session_id']}__p{record['phase']}"
        metadata = record["metadata_file"]
        inventory_rows.append(
            {
                "relative_path": f"{directory}/{metadata['file']}",
                "bytes": int(metadata["bytes"]),
                "sha256": metadata["sha256"],
            }
        )
        for array in record["arrays"].values():
            inventory_rows.append(
                {
                    "relative_path": f"{directory}/{array['file']}",
                    "bytes": int(array["bytes"]),
                    "sha256": array["sha256"],
                }
            )
    inventory_rows.sort(key=lambda row: row["relative_path"])
    inventory_path = args.out / "object_inventory.jsonl"
    inventory_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in inventory_rows),
        encoding="utf-8",
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "created_at": created_at,
        "manifest": {
            "file": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "content_sha256": manifest["content_sha256"],
        },
        "replay_streams": len(selected_replay),
        "targeted_streams": len(targeted_records),
        "replay_windows": replay_windows,
        "targeted_windows": targeted_windows,
        "total_windows": total_windows,
        "observed_targeted_fraction": targeted_windows / total_windows,
        "object_inventory": {
            "file": inventory_path.name,
            "objects": len(inventory_rows),
            "bytes": sum(int(row["bytes"]) for row in inventory_rows),
            "sha256": sha256_file(inventory_path),
        },
    }
    receipt_path = args.out / "mix_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
