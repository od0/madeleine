"""Distill per-chunk frame anchoring into one portable parquet.

The mapped labels (labels_native.parquet) are chunk-local: frame_idx counts
from 0 within each 20 s chunk. The anchor into the source video —
original_video.start_frame / end_frame — lives only in the extraction's
per-chunk metadata.json. The foreign shard builder runs on the pod, where the
19 GB extraction does not; this module distills the anchoring for chosen
videos into a single small parquet that travels with the mapped labels.

Row per chunk: video_id, chunk_id, start_frame, end_frame, start_time,
end_time, grid_hz, n_rows (from the chunk's declared size). grid_hz is
recomputed here as n_rows / duration and must agree with the chunk index —
disagreement is refused, not papered over.

FRAME-BOUND CONVENTION (verified on the data, not assumed). The extraction's
``original_video.end_frame`` is INCLUSIVE — it is the last frame of the chunk,
not one past it. Measured three ways on y4nQHqYSObI: consecutive chunks satisfy
``start[i+1] - end[i] == 1`` for all 466 pairs; ``start_frame == start_time *
60`` exactly; and ``end_time - start_time == 20 s == 1200`` frames against a
declared ``n_rows`` of 1200 while ``end - start`` is 1199.

This file therefore emits ``end_frame`` converted to the HALF-OPEN convention
(``end_frame + 1``), so that everywhere downstream ``end - start == n_rows``,
``np.arange(start, end)`` is exactly the chunk's frames, and abutting chunks
satisfy ``end[i] == start[i+1]`` — which is what the foreign builder's
run-contiguity test requires. Reading the source bound verbatim would both
fail the row-count check and silently fragment every video into 20-second
runs. One frame of drift here costs 4.5% macro-F1 (E4), so the conversion is
asserted per chunk rather than trusted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def collect(
    actions_root: Path, video_ids: list[str]
) -> tuple[pa.Table, dict[str, str]]:
    """Return (table, refusals). A video with ANY inconsistent chunk is dropped
    whole, never partially: partial coverage of a video is indistinguishable
    downstream from a video that simply had fewer chunks, and that is exactly
    the kind of silent shortfall this project refuses. VFR sources fail here
    structurally — 20 s of wall time holds a variable number of frames, so the
    span and the declared row count diverge."""

    rows: list[dict] = []
    refusals: dict[str, str] = {}
    for shard_dir in sorted(Path(actions_root).iterdir()):
        if not shard_dir.is_dir():
            continue
        for vid in video_ids:
            video_dir = shard_dir / vid
            if not video_dir.is_dir():
                continue
            for chunk_dir in sorted(video_dir.iterdir()):
                meta_path = chunk_dir / "metadata.json"
                if not meta_path.is_file():
                    continue
                meta = json.loads(meta_path.read_text())
                original = meta["original_video"]
                duration = float(original["duration"])
                start_frame = int(original["start_frame"])
                # Source bound is inclusive; store half-open (see module docstring).
                end_frame = int(original["end_frame"]) + 1
                n_rows = int(meta.get("chunk_size", end_frame - start_frame))
                if duration <= 0 or end_frame <= start_frame:
                    refusals[vid] = f"{chunk_dir.name}: degenerate chunk timing"
                    continue
                if end_frame - start_frame != n_rows:
                    refusals.setdefault(vid, (
                        f"{chunk_dir.name}: half-open span "
                        f"{end_frame - start_frame} != declared rows {n_rows} "
                        f"(variable frame rate, or the source's frame-bound "
                        f"convention changed — re-verify on the data)"
                    ))
                    continue
                rows.append({
                    "video_id": vid,
                    "chunk_id": chunk_dir.name,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": float(original["start_time"]),
                    "end_time": float(original["end_time"]),
                    "grid_hz": n_rows / duration,
                    "n_rows": n_rows,
                })
    rows = [r for r in rows if r["video_id"] not in refusals]
    if not rows:
        raise SystemExit("no usable chunks found for the requested videos")
    rows.sort(key=lambda r: (r["video_id"], r["start_frame"]))
    return pa.table({k: [r[k] for r in rows] for k in rows[0]}), refusals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actions-root", required=True, type=Path)
    ap.add_argument("--videos", required=True,
                    help="comma-separated video ids")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    vids = [v.strip() for v in args.videos.split(",") if v.strip()]
    table, refusals = collect(args.actions_root, vids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)
    kept = len(set(table["video_id"].to_pylist()))
    print(f"{args.out}: {table.num_rows} chunks over {kept} videos")
    for vid, why in sorted(refusals.items()):
        print(f"  REFUSED {vid}: {why}")
    if refusals:
        (args.out.parent / "chunk_frames_refusals.json").write_text(
            json.dumps(refusals, indent=2, sort_keys=True)
        )


if __name__ == "__main__":
    main()
