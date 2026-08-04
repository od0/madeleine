#!/usr/bin/env python3
"""Derive independently verifiable tier inventories from a frozen split stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.bind_vpt_max_composite import (
    canonical_sha256,
    raw_inventory,
    sha256_file,
    write_json,
)


SCHEMA = "madeleine.vpt-small-split-tier-inventories.v1"
MARKER_SCHEMA = "madeleine.vpt-small-split-tier-inventories-complete.v1"


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise RuntimeError(f"refusing existing tier inventory root: {args.output_dir}")
    assignment = json.loads(args.assignment.read_text(encoding="utf-8"))
    claimed = assignment.get("content_sha256")
    payload = dict(assignment)
    payload.pop("content_sha256", None)
    if not isinstance(claimed, str) or canonical_sha256(payload) != claimed:
        raise RuntimeError("split assignment content hash differs")
    mapping = {
        (str(row["source"]), str(row["directory"])): str(row["tier"])
        for row in assignment["streams"]
    }
    if len(mapping) != len(assignment["streams"]):
        raise RuntimeError("split assignment contains duplicate streams")
    by_tier: dict[str, list[dict[str, Any]]] = {
        str(name): [] for name in assignment["tiers"]
    }
    for row in raw_inventory(args.inventory):
        source = str(row["source"])
        relative = str(row["relative_path"])
        directory, separator, _ = relative.partition("/")
        if not separator:
            raise RuntimeError(f"inventory path is not stream-relative: {relative}")
        tier = mapping.get((source, directory))
        if tier is None:
            raise RuntimeError(f"inventory stream lacks a tier assignment: {source}/{directory}")
        by_tier[tier].append(
            {
                "relative_path": relative,
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
            }
        )
    args.output_dir.mkdir(parents=True)
    tier_rows = {}
    for tier, rows in sorted(by_tier.items()):
        rows.sort(key=lambda row: row["relative_path"])
        path = args.output_dir / f"{tier}.jsonl"
        path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        expected = assignment["tiers"][tier]
        if len(rows) != int(expected["objects"]) or sum(
            row["bytes"] for row in rows
        ) != int(expected["used_bytes"]):
            raise RuntimeError(f"tier inventory totals differ: {tier}")
        tier_rows[tier] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "objects": len(rows),
            "bytes": sum(row["bytes"] for row in rows),
            "stage_root": str(expected["path"]),
        }
    receipt = {
        "schema_version": SCHEMA,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "source_inventory_sha256": sha256_file(args.inventory),
        "split_assignment_sha256": sha256_file(args.assignment),
        "split_assignment_content_sha256": claimed,
        "tiers": tier_rows,
        "objects": sum(row["objects"] for row in tier_rows.values()),
        "bytes": sum(row["bytes"] for row in tier_rows.values()),
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    receipt_path = args.output_dir / "tier_inventory_receipt.json"
    write_json(receipt_path, receipt)
    marker = {
        "schema_version": MARKER_SCHEMA,
        "tier_inventory_receipt_sha256": sha256_file(receipt_path),
        "tier_inventory_receipt_content_sha256": receipt["content_sha256"],
        "objects": receipt["objects"],
        "bytes": receipt["bytes"],
    }
    write_json(args.output_dir / "TIER_INVENTORIES_COMPLETE.json", marker)
    print(json.dumps(marker, sort_keys=True))
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
