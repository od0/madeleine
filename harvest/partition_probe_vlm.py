"""Partition unfinished probe classifications into deterministic ID shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_ids(path: Path) -> set[str]:
    return {
        json.loads(line)["video_id"]
        for line in path.read_text().splitlines()
        if line.strip()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--completed-predictions", type=Path, action="append", default=[])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shards", type=int, required=True)
    args = ap.parse_args()
    if args.shards < 1:
        raise ValueError("--shards must be positive")

    eligible = {
        row["video_id"]
        for row in (
            json.loads(line) for line in args.scan.read_text().splitlines() if line.strip()
        )
        if row.get("error") is None
    }
    completed: set[str] = set()
    for path in args.completed_predictions:
        completed.update(jsonl_ids(path))
    unexpected_completed = sorted(completed - eligible)
    remaining = sorted(eligible - completed)
    shards = [remaining[index :: args.shards] for index in range(args.shards)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_rows = []
    for index, video_ids in enumerate(shards):
        path = args.out_dir / f"shard-{index:02d}.ids"
        path.write_text("".join(video_id + "\n" for video_id in video_ids))
        shard_rows.append({
            "shard_index": index,
            "rows": len(video_ids),
            "path": path.name,
            "sha256": sha256(path),
        })
    manifest = {
        "schema_version": 1,
        "scan": str(args.scan.resolve()),
        "scan_sha256": sha256(args.scan),
        "eligible_rows": len(eligible),
        "completed_rows": len(eligible & completed),
        "unexpected_completed_ids": unexpected_completed,
        "remaining_rows": len(remaining),
        "partition_method": "sorted_video_ids_round_robin",
        "shards": shard_rows,
        "completed_prediction_inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in args.completed_predictions
        ],
    }
    (args.out_dir / "partition_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
