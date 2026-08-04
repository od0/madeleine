#!/usr/bin/env python3
"""Build a deterministic, training-only validation view for VPT smoke runs.

The output manifest references an exact prefix of positive-window records from
an already staged VPT-small training generation. It copies no arrays and must
live beside the source manifest so the existing manifest-relative dataset
loader resolves the same immutable session directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "madeleine.vpt-small-20hz-shards.v1"
RECEIPT_SCHEMA = "madeleine.vpt-small-training-smoke-val.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate_source(source: dict[str, Any], path: Path, expected_sha256: str) -> None:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("source manifest SHA-256 mismatch")
    if source.get("schema_version") != SCHEMA:
        raise ValueError("source is not a 20 Hz VPT-small manifest")
    if source.get("phases") != [0] or source.get("window") != 128 or source.get("stride") != 64:
        raise ValueError("source manifest is not phase0/window128/stride64")
    payload = dict(source)
    claimed = payload.pop("content_sha256", None)
    if not isinstance(claimed, str) or canonical_sha256(payload) != claimed:
        raise RuntimeError("source manifest content hash mismatch")
    records = list(source.get("records", []))
    identities = [(str(row["session_id"]), int(row["phase"])) for row in records]
    if len(identities) != len(set(identities)) or any(phase != 0 for _, phase in identities):
        raise RuntimeError("source records are not unique phase-0 streams")


def build(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.output_manifest, args.output_sessions, args.output_receipt):
        if path.exists():
            raise FileExistsError(f"refusing existing output: {path}")
        if path.parent.resolve() != args.source_manifest.parent.resolve():
            raise ValueError("all outputs must live beside the source manifest")
    if args.minimum_windows < 1:
        raise ValueError("minimum windows must be positive")
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    validate_source(source, args.source_manifest, args.source_manifest_sha256)

    selected: list[dict[str, Any]] = []
    windows = 0
    for record in source["records"]:
        if int(record["windows"]) <= 0:
            continue
        selected.append(record)
        windows += int(record["windows"])
        if windows >= args.minimum_windows:
            break
    if windows < args.minimum_windows:
        raise RuntimeError("source manifest cannot satisfy smoke validation window floor")

    sessions = [str(record["session_id"]) for record in selected]
    args.output_sessions.write_bytes(("\n".join(sessions) + "\n").encode("utf-8"))
    result = {
        key: value
        for key, value in source.items()
        if key not in {"content_sha256", "records", "sessions_file", "totals"}
    }
    result.update(
        {
            "sessions_file": {
                "path": args.output_sessions.name,
                "sha256": sha256_file(args.output_sessions),
                "sessions": sessions,
            },
            "records": selected,
            "totals": {
                "source_sessions": len(selected),
                "derived_streams": len(selected),
                "derived_rows": sum(int(record["derived_rows"]) for record in selected),
                "windows": windows,
            },
            "smoke_subset": {
                "source_manifest_sha256": args.source_manifest_sha256,
                "selection": "source record order; positive-window records through first cumulative minimum",
                "minimum_windows": args.minimum_windows,
            },
        }
    )
    result["content_sha256"] = canonical_sha256(result)
    write_json(args.output_manifest, result)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "source_manifest_sha256": args.source_manifest_sha256,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "output_manifest_sha256": sha256_file(args.output_manifest),
        "output_manifest_content_sha256": result["content_sha256"],
        "output_sessions_sha256": sha256_file(args.output_sessions),
        "selected_sessions": len(selected),
        "selected_rows": result["totals"]["derived_rows"],
        "selected_windows": windows,
        "minimum_windows": args.minimum_windows,
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    write_json(args.output_receipt, receipt)
    return receipt


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--minimum-windows", type=int, default=32)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-sessions", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    receipt = build(parse_args(argv))
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
