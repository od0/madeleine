#!/usr/bin/env python3
"""Validate the frozen native-60-Hz VPT data and endpoint contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from experiments.validate_vpt_small_data import validate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contract(
    contract_path: Path, train_root: Path, val_root: Path
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != "madeleine.vpt-small-native60-generation.v1":
        raise ValueError("unexpected native-rate generation contract schema")
    reports = {
        "train": validate(train_root),
        "validation": validate(val_root),
    }
    roots = {"train": train_root, "validation": val_root}
    for name, root in roots.items():
        expected = contract["generations"][name]
        manifest = root / "build_manifest.json"
        marker = root / "complete.json"
        if sha256_file(manifest) != expected["manifest_sha256"]:
            raise RuntimeError(f"{name} manifest SHA-256 differs")
        if sha256_file(marker) != expected["marker_sha256"]:
            raise RuntimeError(f"{name} marker SHA-256 differs")
        for field in ("derived_rows", "windows", "derived_streams"):
            if int(reports[name][field]) != int(expected[field]):
                raise RuntimeError(f"{name} {field} differs")
        for field in ("window", "stride"):
            if int(reports[name][field]) != int(contract["geometry"][field]):
                raise RuntimeError(f"{name} {field} differs from contract geometry")
    global_batch = int(contract["training"]["global_batch"])
    epochs = int(contract["training"]["full_epochs"])
    steps_per_epoch = math.ceil(reports["train"]["windows"] / global_batch)
    full_endpoint = steps_per_epoch * epochs
    if steps_per_epoch != int(contract["training"]["optimizer_steps_per_epoch"]):
        raise RuntimeError("optimizer steps per epoch differ")
    if full_endpoint != int(contract["training"]["full_endpoint_optimizer_steps"]):
        raise RuntimeError("full optimizer endpoint differs")
    report = {
        "ok": True,
        "contract_sha256": sha256_file(contract_path),
        "train_manifest_sha256": reports["train"]["manifest_sha256"],
        "validation_manifest_sha256": reports["validation"]["manifest_sha256"],
        "train_windows": reports["train"]["windows"],
        "validation_windows": reports["validation"]["windows"],
        "optimizer_steps_per_epoch": steps_per_epoch,
        "full_endpoint_optimizer_steps": full_endpoint,
        "window": int(contract["geometry"]["window"]),
        "stride": int(contract["geometry"]["stride"]),
    }
    short_endpoint = contract["training"].get("short_endpoint_optimizer_steps")
    if short_endpoint is not None:
        short_endpoint = int(short_endpoint)
        if short_endpoint != 2340 or short_endpoint >= full_endpoint:
            raise RuntimeError("short optimizer endpoint is invalid")
        report["short_endpoint_optimizer_steps"] = short_endpoint
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_contract(args.contract, args.train_root, args.val_root)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
