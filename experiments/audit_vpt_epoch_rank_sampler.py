#!/usr/bin/env python3
"""Freeze exact step-major sample order for the VPT EpochRankSampler."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.bind_vpt_max_composite import canonical_sha256, sha256_file, write_json
from experiments.train_vpt_small import EpochRankSampler


SCHEMA = "madeleine.epoch-rank-sampler-audit.v1"


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing existing sampler audit: {args.output}")
    samplers = [
        EpochRankSampler(
            args.population_windows,
            global_batch=args.global_batch,
            rank=rank,
            world_size=args.world_size,
            seed=args.seed,
        )
        for rank in range(args.world_size)
    ]
    steps = samplers[0].steps
    local_batch = samplers[0].local_batch
    expected_samples = steps * args.global_batch
    padding = expected_samples - args.population_windows
    rows = []
    for epoch in range(args.epochs):
        rank_orders = []
        for sampler in samplers:
            sampler.set_epoch(epoch)
            rank_orders.append(sampler.epoch_indices())
        global_order = []
        for step in range(steps):
            begin = step * local_batch
            stop = begin + local_batch
            for rank_order in rank_orders:
                global_order.extend(rank_order[begin:stop])
        counts = [0] * args.population_windows
        for index in global_order:
            counts[index] += 1
        row = {
            "epoch": epoch + 1,
            "global_order_sha256": hashlib.sha256(
                array("Q", global_order).tobytes()
            ).hexdigest(),
            "global_samples": len(global_order),
            "unique_samples": sum(count > 0 for count in counts),
            "missing_base_samples": sum(count == 0 for count in counts),
            "padding_repeats": sum(max(0, count - 1) for count in counts),
            "max_occurrence": max(counts),
            "rank_samples": [len(order) for order in rank_orders],
        }
        if row != {
            **row,
            "global_samples": expected_samples,
            "unique_samples": args.population_windows,
            "missing_base_samples": 0,
            "padding_repeats": padding,
            "max_occurrence": 2 if padding else 1,
            "rank_samples": [steps * local_batch] * args.world_size,
        }:
            raise RuntimeError(f"sampler coverage gate failed at epoch {epoch + 1}")
        rows.append(row)
    receipt = {
        "schema_version": SCHEMA,
        "source_commit": args.source_commit,
        "train_script_sha256": args.train_script_sha256,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "population_windows": args.population_windows,
        "seed": args.seed,
        "world_size": args.world_size,
        "global_batch": args.global_batch,
        "local_batch_per_rank": local_batch,
        "optimizer_steps_per_epoch": steps,
        "epochs": args.epochs,
        "audit_method": "Reconstruct each step-major global order from all EpochRankSampler rank partitions and count exact base-index multiplicities.",
        "rows": rows,
        "result": "pass",
        "interpretation": f"Every epoch covers all base windows once and repeats exactly {padding} deterministic leading-permutation indices only to complete the final global batch.",
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    write_json(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-windows", type=int, required=True)
    parser.add_argument("--global-batch", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--train-script-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
