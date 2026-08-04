#!/usr/bin/env python3
"""Freeze a small record-level subset of a composite VPT manifest for resume smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.bind_vpt_max_composite import (
    canonical_sha256,
    manifest_records,
    require_sha,
    sha256_bytes,
    sha256_file,
    write_json,
)


SUBSET_SCHEMA = "madeleine.vpt-small-composite-preflight-subset.v1"
MARKER_SCHEMA = "madeleine.vpt-small-composite-preflight-subset-complete.v1"


def select_until(
    records: list[dict[str, Any]], target_windows: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    windows = 0
    for record in records:
        selected.append(record)
        windows += int(record["windows"])
        if windows >= target_windows:
            return selected
    raise RuntimeError(
        f"component has only {windows} windows, below requested {target_windows}"
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    require_sha(args.parent_manifest, args.parent_manifest_sha256)
    parent = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    records = manifest_records(parent, "preflight parent manifest")
    policy = parent.get("composite_policy", {})
    names = list(policy.get("component_order", []))
    if len(names) != 2:
        raise RuntimeError("preflight parent must have exactly two frozen components")
    if args.first_component_sessions <= 0 or args.first_component_sessions >= len(records):
        raise ValueError("invalid first-component session boundary")
    first_records = records[: args.first_component_sessions]
    second_records = records[args.first_component_sessions :]
    first_selected = select_until(first_records, args.first_component_target_windows)
    second_selected = select_until(second_records, args.second_component_target_windows)
    selected = [*first_selected, *second_selected]
    sessions = [str(record["session_id"]) for record in selected]
    if len(sessions) != len(set(sessions)):
        raise RuntimeError("preflight subset has duplicate sessions")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing nonempty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions_path = args.output_dir / "train_sessions.txt"
    sessions_path.write_bytes(("\n".join(sessions) + "\n").encode("utf-8"))
    totals = {
        "source_sessions": len(selected),
        "derived_streams": len(selected),
        "derived_rows": sum(int(record["derived_rows"]) for record in selected),
        "windows": sum(int(record["windows"]) for record in selected),
    }
    subset = {
        key: value
        for key, value in parent.items()
        if key not in {"content_sha256", "records", "sessions_file", "source_root", "totals"}
    }
    subset.update(
        {
            "source_root": f"preflight-subset:{args.parent_manifest_sha256}",
            "sessions_file": {
                "path": sessions_path.name,
                "sha256": sha256_file(sessions_path),
                "sessions": sessions,
            },
            "records": selected,
            "totals": totals,
            "subset_policy": {
                "parent_manifest_sha256": args.parent_manifest_sha256,
                "parent_manifest_content_sha256": parent["content_sha256"],
                "selection": "whole records from the start of each frozen component order until its target window count is met",
                "components": [
                    {
                        "name": names[0],
                        "target_windows": args.first_component_target_windows,
                        "sessions": len(first_selected),
                        "windows": sum(int(record["windows"]) for record in first_selected),
                        "membership_sha256": sha256_bytes(
                            (
                                "\n".join(str(record["session_id"]) for record in first_selected)
                                + "\n"
                            ).encode("utf-8")
                        ),
                    },
                    {
                        "name": names[1],
                        "target_windows": args.second_component_target_windows,
                        "sessions": len(second_selected),
                        "windows": sum(int(record["windows"]) for record in second_selected),
                        "membership_sha256": sha256_bytes(
                            (
                                "\n".join(str(record["session_id"]) for record in second_selected)
                                + "\n"
                            ).encode("utf-8")
                        ),
                    },
                ],
            },
        }
    )
    subset["content_sha256"] = canonical_sha256(subset)
    manifest_path = args.output_dir / "build_manifest.json"
    write_json(manifest_path, subset)
    receipt = {
        "schema_version": SUBSET_SCHEMA,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "parent_manifest_sha256": args.parent_manifest_sha256,
        "parent_manifest_content_sha256": parent["content_sha256"],
        "subset_manifest_sha256": sha256_file(manifest_path),
        "subset_manifest_content_sha256": subset["content_sha256"],
        "membership_sha256": sha256_file(sessions_path),
        "totals": totals,
        "components": subset["subset_policy"]["components"],
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    receipt_path = args.output_dir / "subset_receipt.json"
    write_json(receipt_path, receipt)
    marker = {
        "schema_version": MARKER_SCHEMA,
        "subset_receipt_sha256": sha256_file(receipt_path),
        "subset_receipt_content_sha256": receipt["content_sha256"],
        "subset_manifest_sha256": sha256_file(manifest_path),
        "membership_sha256": sha256_file(sessions_path),
        "windows": totals["windows"],
    }
    write_json(args.output_dir / "PREFLIGHT_SUBSET_COMPLETE.json", marker)
    print(json.dumps(marker, sort_keys=True))
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--parent-manifest-sha256", required=True)
    parser.add_argument("--first-component-sessions", type=int, required=True)
    parser.add_argument("--first-component-target-windows", type=int, required=True)
    parser.add_argument("--second-component-target-windows", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
