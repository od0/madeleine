#!/usr/bin/env python3
"""Normalize stream metadata for a 20 Hz phase-0 VPT evaluation sidecar."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(input_path: Path, output_path: Path) -> dict[str, object]:
    with np.load(input_path, allow_pickle=False) as archive:
        required = {
            "y_true",
            "y_prob",
            "input_active",
            "session_lengths",
            "session_ids",
            "source_row_index",
            "source_engine_frame_idx",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"input sidecar lacks {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    lengths = arrays["session_lengths"].astype(np.int64, copy=False)
    stream_ids = arrays["session_ids"].astype(str, copy=False)
    row_count = len(arrays["y_true"])
    if int(lengths.sum()) != row_count or len(lengths) != len(stream_ids):
        raise RuntimeError("input stream metadata does not cover the sidecar rows")
    row_sessions = np.concatenate(
        [
            np.full(int(length), stream_id.split("__run", 1)[0])
            for stream_id, length in zip(stream_ids, lengths, strict=True)
        ]
    )
    source_rows = arrays["source_row_index"].astype(np.int64, copy=False)
    boundary = np.ones(row_count, dtype=bool)
    boundary[1:] = (row_sessions[1:] != row_sessions[:-1]) | (
        np.diff(source_rows) != 3
    )
    starts = np.flatnonzero(boundary)
    ends = np.concatenate((starts[1:], np.asarray([row_count], dtype=np.int64)))
    normalized_lengths = ends - starts
    counters: dict[str, int] = {}
    normalized_ids: list[str] = []
    for start in starts:
        session_id = str(row_sessions[int(start)])
        subrun = counters.get(session_id, 0)
        counters[session_id] = subrun + 1
        normalized_ids.append(f"{session_id}__run000__sub{subrun:03d}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        y_true=arrays["y_true"],
        y_prob=arrays["y_prob"],
        input_active=arrays["input_active"],
        session_lengths=normalized_lengths,
        session_ids=np.asarray(normalized_ids),
        source_row_index=source_rows,
        source_engine_frame_idx=arrays["source_engine_frame_idx"],
    )
    return {
        "schema_version": "madeleine.vpt-phase0-sidecar-normalization.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "streams": len(lengths),
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "streams": len(normalized_lengths),
            "rows": row_count,
        },
        "source_row_step": 3,
        "truth_probability_rows_changed": 0,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = normalize(args.input, args.output)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
