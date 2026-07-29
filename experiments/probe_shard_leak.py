"""Probe built shards for the input-overlay answer-key leak.

Re-verification for the 2026-07-26 masking audit: on the leaked v2 shards,
mean brightness of the leftmost leaked cell columns separated `left` frames
at AUC 1.000. After the manifest-geometry fix and rebuild, this probe must
show, per session shard:

1. the masked zone (the manifest's input_overlay rect_norm at shard
   resolution, with build_dataset's 1 px dilation) is identically zero in
   EVERY frame — not just the first frame the builder asserts on;
2. single-feature AUC (zone mean brightness vs each key) is undefined on
   the zero zone, and in the thin band of game pixels just above the zone
   it sits at gameplay levels, far from 1.0. Gameplay correlation is real
   signal (it is the research task), so the band is reported, not gated;
   the hard requirement is (1).

Usage:
  uv run python experiments/probe_shard_leak.py \
      --shards data/shards_v2 --sessions-roots sessions data/sessions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# data.schema.KEY_ORDER, inlined so the probe runs standalone against any
# checkout's shards.
KEY_ORDER = ["left", "right", "up", "down", "jump", "dash", "grab"]

BAND_ROWS = 5


def _zone(rect_norm: list[float], size: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect_norm
    # Mirrors build_dataset's output-resolution re-mask, including dilation.
    return (
        max(0, int(x0 * size) - 1),
        max(0, int(y0 * size) - 1),
        min(size, int(np.ceil(x1 * size)) + 1),
        min(size, int(np.ceil(y1 * size)) + 1),
    )


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    return float(
        (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
        / (len(pos) * len(neg))
    )


def probe_shard(npz_path: Path, manifest_path: Path) -> dict:
    data = np.load(npz_path)
    frames = data["frames"]
    keys = data["keys"].astype(bool)
    size = frames.shape[1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    regions = {r["name"]: r for r in manifest["masked_regions"]}

    report: dict = {"session_id": str(data["session_id"]), "frames": len(frames)}
    for name in ("input_overlay", "wild_overlay"):
        if name not in regions:
            continue
        zx0, zy0, zx1, zy1 = _zone(regions[name]["rect_norm"], size)
        zone_max = int(frames[:, zy0:zy1, zx0:zx1].max()) if len(frames) else 0
        band_y0 = max(0, zy0 - BAND_ROWS)
        band = frames[:, band_y0:zy0, zx0:zx1]
        band_mean = band.reshape(len(frames), -1).mean(axis=1)
        report[name] = {
            "zone_xyxy_at_shard_res": [zx0, zy0, zx1, zy1],
            "zone_max_over_all_frames": zone_max,
            "zone_is_all_zero": zone_max == 0,
            "band_above_auc_per_key": {
                key: (None if (a := _auc(band_mean, keys[:, k])) is None
                      else round(a, 4))
                for k, key in enumerate(KEY_ORDER)
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--sessions-roots", nargs="+", type=Path, required=True)
    args = parser.parse_args()

    failures = 0
    reports = []
    for npz_path in sorted(args.shards.glob("rec_*.npz")):
        session_id = npz_path.stem
        manifest_path = next(
            (
                root / session_id / "manifest.json"
                for root in args.sessions_roots
                if (root / session_id / "manifest.json").is_file()
            ),
            None,
        )
        if manifest_path is None:
            raise SystemExit(f"{session_id}: session manifest not found")
        report = probe_shard(npz_path, manifest_path)
        reports.append(report)
        for name in ("input_overlay", "wild_overlay"):
            if name in report and not report[name]["zone_is_all_zero"]:
                failures += 1
    print(json.dumps(reports, indent=2))
    if failures:
        raise SystemExit(f"{failures} masked zone(s) not identically zero")


if __name__ == "__main__":
    main()
