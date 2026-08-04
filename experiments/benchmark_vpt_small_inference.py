#!/usr/bin/env python3
"""Benchmark the SHA-verified VPT-small graph on synthetic 128-frame inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import subprocess
import time
from typing import Iterable

import torch

from badeline.vpt_small import VPTSmallConfig, VPTSmallIDM, maybe_autocast
from experiments.eval_vpt_small import sha256_file


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark(
    checkpoint: Path,
    *,
    expected_sha256: str,
    device: torch.device,
    batch_size: int,
    warmups: int,
    timed_batches: int,
) -> dict:
    actual_hash = sha256_file(checkpoint)
    if actual_hash != expected_sha256:
        raise RuntimeError("VPT-small checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "madeleine.vpt-small-checkpoint.v1":
        raise ValueError("unsupported VPT-small checkpoint")
    model = VPTSmallIDM(VPTSmallConfig.from_dict(payload["config"]["model"]))
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    synthetic = torch.zeros(
        batch_size, 128, 3, 128, 128, device=device, dtype=torch.float32
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmups):
            with maybe_autocast(device, dtype):
                output = model(synthetic)
            if output.shape != (batch_size, 128, 7, 2):
                raise RuntimeError(f"unexpected model output shape {tuple(output.shape)}")
        synchronize(device)
        started = time.perf_counter()
        for _ in range(timed_batches):
            with maybe_autocast(device, dtype):
                output = model(synthetic)
        synchronize(device)
        elapsed = time.perf_counter() - started
    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(device))
    elif device.type == "mps":
        peak = int(torch.mps.driver_allocated_memory())
    else:
        maximum_resident_set = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak = maximum_resident_set if platform.system() == "Darwin" else maximum_resident_set * 1024
    sequences = batch_size * timed_batches
    repo = Path(__file__).resolve().parents[1]
    implementation_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    return {
        "schema_version": "madeleine.vpt-small-synthetic-inference-benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"path": str(checkpoint), "bytes": checkpoint.stat().st_size, "sha256": actual_hash},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()},
        "implementation_commit": implementation_commit,
        "device": {"requested": str(device), "type": device.type, "name": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type, "precision": str(dtype)},
        "batch_size": batch_size,
        "warmups": warmups,
        "timed_batches": timed_batches,
        "sequences": sequences,
        "elapsed_seconds": elapsed,
        "sequences_per_second": sequences / elapsed,
        "peak_memory_bytes": peak,
        "input_shape": [batch_size, 128, 3, 128, 128],
        "output_shape": list(output.shape),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timed-batches", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = benchmark(
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        warmups=args.warmups,
        timed_batches=args.timed_batches,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
