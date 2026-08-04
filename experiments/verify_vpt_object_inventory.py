#!/usr/bin/env python3
"""Stream-verify one immutable VPT object inventory against a local root."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify(inventory: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    rows = [json.loads(line) for line in inventory.read_text().splitlines() if line]
    seen: set[str] = set()
    total_bytes = 0
    for index, row in enumerate(rows, start=1):
        relative = str(row["relative_path"])
        if relative in seen:
            raise RuntimeError(f"duplicate inventory path: {relative}")
        seen.add(relative)
        path = (root / relative).resolve()
        if root not in path.parents:
            raise RuntimeError(f"inventory path escapes root: {relative}")
        expected_bytes = int(row["bytes"])
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise RuntimeError(f"inventory size mismatch: {relative}")
        if sha256_file(path) != str(row["sha256"]):
            raise RuntimeError(f"inventory SHA-256 mismatch: {relative}")
        total_bytes += expected_bytes
        if index % 250 == 0:
            print(json.dumps({"verified_objects": index, "verified_bytes": total_bytes}))
    result = {
        "schema_version": "madeleine.vpt-object-inventory-verification.v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "inventory_sha256": sha256_file(inventory),
        "objects": len(rows),
        "bytes": total_bytes,
        "status": "pass",
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")
    result = verify(args.inventory, args.root)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
