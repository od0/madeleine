#!/usr/bin/env python3
"""Derive hash-bound native-60-Hz VPT streams from canonical 60-Hz shards.

Unlike the paper-faithful 20-Hz builder, this generation retains every row
inside each contiguous engine run. It still materializes mmap-friendly arrays
and stride-64/window-128 starts so the proven VPT-small trainer can consume an
immutable manifest without decoding or reconstructing labels on the worker.
The production study permits two physically meaningful geometries: 128/64
for the short-context arms and 384/192 for the span-matched arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from badeline.train import read_session_ids
from data.schema import KEY_ORDER
from experiments.build_vpt_small_128px_20hz import (
    canonical_sha256,
    derive_one,
    load_expected_hashes,
    sha256_file,
)


SCHEMA = "madeleine.vpt-small-60hz-shards.v1"
MARKER_SCHEMA = "madeleine.vpt-small-60hz-complete.v1"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-hashes", type=Path)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument(
        "--created-at",
        default="2026-07-30T00:00:00+00:00",
        help="frozen generation identity; use the contract value in production",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if (args.window, args.stride) not in {(128, 64), (384, 192)}:
        raise ValueError(
            "production native-rate VPT derivation requires window/stride "
            "128/64 or 384/192"
        )
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing nonempty output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    sessions = read_session_ids(args.sessions)
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
            phase=0,
            expected_hash=expected.get(session_id),
            window=args.window,
            stride=args.stride,
            row_step=1,
            schema=SCHEMA,
        )
        for session_id in sessions
    ]
    manifest = {
        "schema_version": SCHEMA,
        "created_at": args.created_at,
        "source_root": str(args.source_root),
        "sessions_file": {
            "path": str(args.sessions),
            "sha256": sha256_file(args.sessions),
            "sessions": sessions,
        },
        "phases": [0],
        "source_rate_hz": 60,
        "derived_rate_hz": 60,
        "phase_rule": "within each contiguous engine-frame run, retain every row",
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
        "created_at": args.created_at,
        "manifest": {
            "file": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "content_sha256": manifest["content_sha256"],
        },
        "derived_streams": len(records),
        "derived_rows": manifest["totals"]["derived_rows"],
        "windows": manifest["totals"]["windows"],
        "window": args.window,
        "stride": args.stride,
    }
    marker_path = args.output_root / "complete.json"
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(marker, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
