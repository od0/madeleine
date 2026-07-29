#!/usr/bin/env python3
"""Publish immutable VPT-small epoch checkpoints to R2 as they appear."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Iterable

from experiments.eval_vpt_small import sha256_file


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run(*args: str, capture: bool = False) -> str:
    completed = subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return completed.stdout if capture else ""


def publish(checkpoint: Path, remote: str) -> dict[str, object]:
    digest = sha256_file(checkpoint)
    destination = f"{remote.rstrip('/')}/{checkpoint.name}"
    run("rclone", "copyto", str(checkpoint), destination, "--immutable")
    reader = subprocess.Popen(["rclone", "cat", destination], stdout=subprocess.PIPE)
    assert reader.stdout is not None
    remote_hasher = hashlib.sha256()
    for chunk in iter(lambda: reader.stdout.read(1024 * 1024), b""):
        remote_hasher.update(chunk)
    return_code = reader.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, ["rclone", "cat", destination])
    streamed_digest = remote_hasher.hexdigest()
    if streamed_digest != digest:
        raise RuntimeError(f"R2 hash mismatch for {checkpoint.name}")
    receipt = {
        "schema_version": "madeleine.vpt-small-epoch-publication.v1",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "filename": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": digest,
        },
        "r2_object": destination,
        "r2_streamed_sha256": streamed_digest,
    }
    receipt_path = checkpoint.with_suffix(".r2.json")
    atomic_json(receipt_path, receipt)
    run(
        "rclone",
        "copyto",
        str(receipt_path),
        f"{remote.rstrip('/')}/{receipt_path.name}",
        "--immutable",
    )
    return receipt


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs != 20:
        raise ValueError("production VPT-small watcher requires exactly 20 epochs")
    if args.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    while True:
        for epoch in range(1, args.epochs + 1):
            checkpoint = args.run_dir / f"checkpoint_epoch_{epoch:02d}.pt"
            receipt = checkpoint.with_suffix(".r2.json")
            if checkpoint.is_file() and not receipt.exists():
                published = publish(checkpoint, args.remote)
                print(json.dumps(published, sort_keys=True), flush=True)
        completion = args.run_dir / "complete.json"
        receipts = list(args.run_dir.glob("checkpoint_epoch_*.r2.json"))
        if completion.is_file():
            complete = json.loads(completion.read_text(encoding="utf-8"))
            if complete.get("completed_production_endpoint") is True:
                if len(receipts) != args.epochs:
                    raise RuntimeError(
                        "training completed but not all epoch checkpoints exist for publication"
                    )
                return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
