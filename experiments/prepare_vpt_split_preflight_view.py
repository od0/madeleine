#!/usr/bin/env python3
"""Expose a frozen preflight manifest through an already verified split stage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

from experiments.bind_vpt_max_composite import (
    canonical_sha256,
    manifest_records,
    require_sha,
    sha256_file,
    write_json,
)


STAGE_SCHEMA = "madeleine.vpt-small-full-foreign-split-stage.v1"
SUBSET_MARKER_SCHEMA = "madeleine.vpt-small-composite-preflight-subset-complete.v1"
VIEW_SCHEMA = "madeleine.vpt-small-split-preflight-view.v1"
MARKER_SCHEMA = "madeleine.vpt-small-split-preflight-view-complete.v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        raise RuntimeError(f"refusing existing preflight view: {args.output_root}")
    stage = read_json(args.full_root / "STAGE_COMPLETE.json")
    if stage.get("schema_version") != STAGE_SCHEMA:
        raise ValueError("full split stage marker schema differs")
    require_sha(args.full_root / "build_manifest.json", stage["build_manifest_sha256"])
    require_sha(args.full_root / "train_sessions.txt", stage["membership_sha256"])

    subset_marker_path = args.subset_root / "PREFLIGHT_SUBSET_COMPLETE.json"
    subset_marker = read_json(subset_marker_path)
    if subset_marker.get("schema_version") != SUBSET_MARKER_SCHEMA:
        raise ValueError("preflight subset marker schema differs")
    subset_manifest_path = args.subset_root / "build_manifest.json"
    subset_sessions_path = args.subset_root / "train_sessions.txt"
    subset_receipt_path = args.subset_root / "subset_receipt.json"
    require_sha(subset_manifest_path, subset_marker["subset_manifest_sha256"])
    require_sha(subset_sessions_path, subset_marker["membership_sha256"])
    require_sha(subset_receipt_path, subset_marker["subset_receipt_sha256"])
    subset_manifest = read_json(subset_manifest_path)
    records = manifest_records(subset_manifest, "split preflight subset")
    sessions = subset_sessions_path.read_text(encoding="utf-8").splitlines()
    if [str(record["session_id"]) for record in records] != sessions:
        raise RuntimeError("preflight subset records differ from frozen membership")

    args.output_root.mkdir(parents=True)
    copied = {}
    for name in (
        "build_manifest.json",
        "train_sessions.txt",
        "subset_receipt.json",
        "PREFLIGHT_SUBSET_COMPLETE.json",
    ):
        source = args.subset_root / name
        destination = args.output_root / name
        shutil.copy2(source, destination)
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"preflight binding copy differs: {name}")
        copied[name] = sha256_file(destination)
    for record in records:
        directory = f"{record['session_id']}__p{record['phase']}"
        source = args.full_root / directory
        if not source.is_dir():
            raise RuntimeError(f"selected stream is absent from full split stage: {directory}")
        os.symlink(source, args.output_root / directory, target_is_directory=True)

    receipt = {
        "schema_version": VIEW_SCHEMA,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "full_stage_receipt_sha256": sha256_file(args.full_root / "STAGE_COMPLETE.json"),
        "full_composite_reference_sha256": stage["composite_reference_sha256"],
        "subset_artifacts": copied,
        "streams": len(records),
        "windows": int(subset_manifest["totals"]["windows"]),
        "view": "one absolute directory symlink per frozen selected stream into the independently SHA-verified full split stage",
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    receipt_path = args.output_root / "preflight_view_receipt.json"
    write_json(receipt_path, receipt)
    marker = {
        "schema_version": MARKER_SCHEMA,
        "preflight_view_receipt_sha256": sha256_file(receipt_path),
        "preflight_view_receipt_content_sha256": receipt["content_sha256"],
        "subset_manifest_sha256": copied["build_manifest.json"],
        "membership_sha256": copied["train_sessions.txt"],
        "streams": len(records),
        "windows": int(subset_manifest["totals"]["windows"]),
    }
    write_json(args.output_root / "PREFLIGHT_VIEW_COMPLETE.json", marker)
    print(json.dumps(marker, sort_keys=True))
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--subset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
