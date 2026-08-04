#!/usr/bin/env python3
"""Raw-evidence scan behind ``results/idm/nitrogen_mapping_code_audit_v1``.

For every video in the corrected-v2 210-video NitroGen population this tool
reads the immutable raw controller chunks, the v1 (contaminated) and v2
(corrected) mapped label trees, and both mapping-report generations, and
emits one JSON record per video plus a corpus aggregate covering:

- ``j_left``/``j_right`` value statistics, stick dtype census, and
  malformedness counts (NaN/inf, out-of-range, non-binary buttons);
- d-pad totals and four-way d-pad/stick coincidence counts with run
  structure and d-pad-edge adjacency;
- H1 (negative-y-up) versus H2 (positive-y-up) accounting for the vertical
  axis plus the analogous horizontal check;
- exact-threshold, near-threshold, and impossible-simultaneity counts;
- independent reconstruction of the v1 and v2 label columns from the raw
  formula under both hypotheses and both bind maps, so the affected-video
  list is derived from data rather than trusted.

The scan is read-only against every input tree.  It intentionally does not
import :mod:`nitrogen.map_actions`; the direction formulas are restated here
so the reconstruction is independent of the code under audit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

KEY_ORDER = ("left", "right", "up", "down", "jump", "dash", "grab")
ACTION_NAMES = ("jump", "dash", "grab")
BUTTON_COLUMNS = (
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east",
    "guide", "left_shoulder", "left_thumb", "left_trigger", "north",
    "right_shoulder", "right_thumb", "right_trigger", "south", "start", "west",
)
T = 0.5

_WORKER: dict[str, Path] = {}


def _init_worker(actions_root: str, v1_root: str, v2_root: str) -> None:
    _WORKER["actions"] = Path(actions_root)
    _WORKER["v1"] = Path(v1_root)
    _WORKER["v2"] = Path(v2_root)


def read_raw(path: Path):
    table = pq.read_table(path)
    n = table.num_rows
    buttons = {
        b: table[b].combine_chunks().to_numpy(zero_copy_only=False)
        for b in BUTTON_COLUMNS
    }

    def stick(name: str):
        col = table[name].combine_chunks()
        lens = pa.compute.list_value_length(col)
        if col.null_count or not pa.compute.all(pa.compute.equal(lens, 2)).as_py():
            raise ValueError(f"{path}: malformed {name}")
        vals = col.flatten().to_numpy(zero_copy_only=False)
        return (
            np.asarray(vals, dtype=np.float64).reshape(n, 2),
            pa.types.is_integer(col.type.value_type),
        )

    j_left, jl_int = stick("j_left")
    j_right, _ = stick("j_right")
    return buttons, j_left, j_right, jl_int, n


def read_labels(path: Path, n: int):
    table = pq.read_table(path)
    if table.num_rows != n:
        raise ValueError(f"{path}: row count {table.num_rows} != raw {n}")
    frame_idx = table["frame_idx"].combine_chunks().to_numpy(zero_copy_only=False)
    if not np.array_equal(frame_idx, np.arange(n, dtype=np.int64)):
        raise ValueError(f"{path}: frame_idx not dense")
    return {
        k: table[k].combine_chunks().to_numpy(zero_copy_only=False).astype(bool)
        for k in KEY_ORDER
    }


def run_stats(mask: np.ndarray):
    if not mask.any():
        return {"frames": 0, "runs": 0, "max_run": 0, "median_run": 0.0}
    delta = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    lengths = (np.flatnonzero(delta == -1) - np.flatnonzero(delta == 1)).astype(np.int64)
    return {
        "frames": int(mask.sum()),
        "runs": int(len(lengths)),
        "max_run": int(lengths.max()),
        "median_run": float(np.median(lengths)),
    }


def or_columns(buttons, names):
    out = np.zeros_like(next(iter(buttons.values())), dtype=bool)
    for name in names:
        out |= buttons[name].astype(bool)
    return out


def scan_video(args):
    video_id, chunk_ids, shard, bind_v1, bind_v2 = args
    video_dir = _WORKER["actions"] / shard / video_id
    concat = {name: [] for name in ("dpad_up", "dpad_down", "y")}
    button_press = Counter()
    y_counter = Counter()
    rec = {
        "video_id": video_id,
        "chunks": len(chunk_ids),
        "rows": 0,
        "int_stick_chunks": 0,
        "float_stick_chunks": 0,
        "nan_or_inf": 0,
        "out_of_range": 0,
        "nonbinary_button_rows": 0,
        "y_min": None, "y_max": None,
        "x_min": None, "x_max": None,
        "counts": Counter(),
        "mismatch": Counter(),
    }
    counts = rec["counts"]
    mismatch = rec["mismatch"]

    for chunk_id in chunk_ids:
        chunk_dir = video_dir / f"{video_id}_chunk_{chunk_id:04d}"
        buttons, j_left, j_right, jl_int, n = read_raw(chunk_dir / "actions_raw.parquet")
        rec["rows"] += n
        rec["int_stick_chunks" if jl_int else "float_stick_chunks"] += 1

        for b in BUTTON_COLUMNS:
            arr = buttons[b]
            rec["nonbinary_button_rows"] += int((~np.isin(arr, (0, 1))).sum())
            button_press[b] += int(np.count_nonzero(arr))

        x = j_left[:, 0]
        y = j_left[:, 1]
        finite = np.isfinite(x) & np.isfinite(y)
        rec["nan_or_inf"] += int((~finite).sum())
        rec["out_of_range"] += int(((np.abs(x) > 1) | (np.abs(y) > 1)).sum())
        for field, arr in (("y", y), ("x", x)):
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            rec[f"{field}_min"] = lo if rec[f"{field}_min"] is None else min(rec[f"{field}_min"], lo)
            rec[f"{field}_max"] = hi if rec[f"{field}_max"] is None else max(rec[f"{field}_max"], hi)

        dpad_up = buttons["dpad_up"].astype(bool)
        dpad_down = buttons["dpad_down"].astype(bool)
        dpad_left = buttons["dpad_left"].astype(bool)
        dpad_right = buttons["dpad_right"].astype(bool)
        stick_up = y < -T
        stick_down = y > T
        stick_left = x < -T
        stick_right = x > T

        counts["stick_up_h1"] += int(stick_up.sum())
        counts["stick_down_h1"] += int(stick_down.sum())
        counts["stick_left"] += int(stick_left.sum())
        counts["stick_right"] += int(stick_right.sum())
        counts["jl_y_zero"] += int((y == 0.0).sum())
        counts["jl_y_exact_pos_half"] += int((y == T).sum())
        counts["jl_y_exact_neg_half"] += int((y == -T).sum())
        counts["jl_x_exact_pos_half"] += int((x == T).sum())
        counts["jl_x_exact_neg_half"] += int((x == -T).sum())
        counts["jl_y_near_threshold"] += int(((np.abs(np.abs(y) - T) < 0.05) & (y != 0)).sum())
        counts["jr_engaged"] += int((np.abs(j_right) > T).any(axis=1).sum())
        counts["dpad_up"] += int(dpad_up.sum())
        counts["dpad_down"] += int(dpad_down.sum())
        counts["dpad_left"] += int(dpad_left.sum())
        counts["dpad_right"] += int(dpad_right.sum())
        counts["dpad_up_and_down"] += int((dpad_up & dpad_down).sum())
        counts["dpad_left_and_right"] += int((dpad_left & dpad_right).sum())
        counts["up_with_stick_up"] += int((dpad_up & stick_up).sum())
        counts["up_with_stick_down"] += int((dpad_up & stick_down).sum())
        counts["down_with_stick_up"] += int((dpad_down & stick_up).sum())
        counts["down_with_stick_down"] += int((dpad_down & stick_down).sum())
        counts["left_with_stick_left"] += int((dpad_left & stick_left).sum())
        counts["left_with_stick_right"] += int((dpad_left & stick_right).sum())
        counts["right_with_stick_left"] += int((dpad_right & stick_left).sum())
        counts["right_with_stick_right"] += int((dpad_right & stick_right).sum())

        y_counter.update(np.round(y, 3))

        pred = {
            "h1_up": dpad_up | stick_up,
            "h1_down": dpad_down | stick_down,
            "h2_up": dpad_up | stick_down,
            "h2_down": dpad_down | stick_up,
            "left": dpad_left | stick_left,
            "right": dpad_right | stick_right,
        }
        v1 = read_labels(_WORKER["v1"] / video_id / chunk_dir.name / "labels_native.parquet", n)
        v2 = read_labels(_WORKER["v2"] / video_id / chunk_dir.name / "labels_native.parquet", n)
        for gen_name, gen in (("v1", v1), ("v2", v2)):
            mismatch[f"{gen_name}_up_ne_h1"] += int((gen["up"] != pred["h1_up"]).sum())
            mismatch[f"{gen_name}_down_ne_h1"] += int((gen["down"] != pred["h1_down"]).sum())
            mismatch[f"{gen_name}_up_ne_h2"] += int((gen["up"] != pred["h2_up"]).sum())
            mismatch[f"{gen_name}_down_ne_h2"] += int((gen["down"] != pred["h2_down"]).sum())
            mismatch[f"{gen_name}_left_ne_formula"] += int((gen["left"] != pred["left"]).sum())
            mismatch[f"{gen_name}_right_ne_formula"] += int((gen["right"] != pred["right"]).sum())
        for action in ACTION_NAMES:
            mismatch[f"v1_{action}_ne_bind"] += int((v1[action] != or_columns(buttons, bind_v1[action])).sum())
            mismatch[f"v2_{action}_ne_bind"] += int((v2[action] != or_columns(buttons, bind_v2[action])).sum())
        counts["v1_up_and_down_both"] += int((v1["up"] & v1["down"]).sum())
        counts["v2_up_and_down_both"] += int((v2["up"] & v2["down"]).sum())
        counts["v2_left_and_right_both"] += int((v2["left"] & v2["right"]).sum())

        concat["dpad_up"].append(dpad_up)
        concat["dpad_down"].append(dpad_down)
        concat["y"].append(y)

    dpad_up = np.concatenate(concat["dpad_up"])
    dpad_down = np.concatenate(concat["dpad_down"])
    y = np.concatenate(concat["y"])
    stick_up = y < -T
    stick_down = y > T
    rec["coincidence_runs"] = {
        "up_with_stick_up": run_stats(dpad_up & stick_up),
        "up_with_stick_down": run_stats(dpad_up & stick_down),
        "down_with_stick_up": run_stats(dpad_down & stick_up),
        "down_with_stick_down": run_stats(dpad_down & stick_down),
        "dpad_up": run_stats(dpad_up),
        "dpad_down": run_stats(dpad_down),
        "stick_up_h1": run_stats(stick_up),
        "stick_down_h1": run_stats(stick_down),
    }
    contradiction = (dpad_up & stick_down) | (dpad_down & stick_up)
    rec["contradiction_frames_total"] = int(contradiction.sum())
    near_edge = 0
    if contradiction.any():
        edges = np.zeros_like(contradiction)
        for arr in (dpad_up, dpad_down):
            edge_at = np.flatnonzero(np.diff(arr.astype(np.int8)) != 0)
            for offset in (-1, 0, 1, 2):
                idx = edge_at + offset
                idx = idx[(idx >= 0) & (idx < len(edges))]
                edges[idx] = True
        near_edge = int((contradiction & edges).sum())
    rec["contradiction_frames_near_dpad_edge"] = near_edge

    rec["counts"] = dict(counts)
    rec["mismatch"] = dict(mismatch)
    rec["button_press_frames"] = dict(button_press)
    rec["y_unique_values"] = len(y_counter)
    rec["y_top_values"] = [[float(v), int(c)] for v, c in y_counter.most_common(8)]

    def classify(prefix: str) -> str:
        h1 = mismatch[f"{prefix}_up_ne_h1"] == 0 and mismatch[f"{prefix}_down_ne_h1"] == 0
        h2 = mismatch[f"{prefix}_up_ne_h2"] == 0 and mismatch[f"{prefix}_down_ne_h2"] == 0
        if h1 and h2:
            return "both_vacuous"
        if h1:
            return "H1"
        if h2:
            return "H2"
        return "neither"

    rec["v1_vertical_class"] = classify("v1")
    rec["v2_vertical_class"] = classify("v2")
    return rec


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-index", type=Path, required=True)
    parser.add_argument("--actions-root", type=Path, required=True)
    parser.add_argument("--v1-mapped-root", type=Path, required=True)
    parser.add_argument("--v2-mapped-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    index_rows = pq.read_table(args.chunk_index).to_pylist()
    chunks_by_video: dict[str, list[int]] = {}
    for row in index_rows:
        chunks_by_video.setdefault(str(row["video_id"]), []).append(int(row["chunk_id"]))
    for video_id in chunks_by_video:
        chunks_by_video[video_id].sort()

    shard_of = {
        video_dir.name: shard.name
        for shard in args.actions_root.glob("SHARD_*")
        for video_dir in shard.iterdir()
        if video_dir.is_dir()
    }
    binds = {}
    for video_id in chunks_by_video:
        report_v1 = json.loads((args.v1_mapped_root / video_id / "mapping_report.json").read_text())
        report_v2 = json.loads((args.v2_mapped_root / video_id / "mapping_report.json").read_text())
        binds[video_id] = (report_v1["bind_map"], report_v2["bind_map"])

    tasks = [
        (v, chunks_by_video[v], shard_of[v], binds[v][0], binds[v][1])
        for v in sorted(chunks_by_video)
    ]
    records = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(args.actions_root), str(args.v1_mapped_root), str(args.v2_mapped_root)),
    ) as pool:
        for done, rec in enumerate(pool.map(scan_video, tasks, chunksize=1)):
            records.append(rec)
            if (done + 1) % 20 == 0:
                print(f"{done + 1}/{len(tasks)} videos scanned", flush=True)

    records.sort(key=lambda r: r["video_id"])
    with (args.out_dir / "per_video_scan.jsonl").open("w") as stream:
        for rec in records:
            stream.write(json.dumps(rec, sort_keys=True) + "\n")

    aggregate = Counter()
    mismatch_aggregate = Counter()
    classes = {"v1": Counter(), "v2": Counter()}
    for rec in records:
        aggregate.update(rec["counts"])
        mismatch_aggregate.update(rec["mismatch"])
        classes["v1"][rec["v1_vertical_class"]] += 1
        classes["v2"][rec["v2_vertical_class"]] += 1
        for field in ("nan_or_inf", "out_of_range", "nonbinary_button_rows",
                      "int_stick_chunks", "float_stick_chunks", "rows"):
            aggregate[field] += rec[field]
    summary = {
        "videos": len(records),
        "aggregate_counts": dict(aggregate),
        "aggregate_mismatches": dict(mismatch_aggregate),
        "v1_vertical_class_counts": dict(classes["v1"]),
        "v2_vertical_class_counts": dict(classes["v2"]),
        "v1_H2_videos": sorted(r["video_id"] for r in records if r["v1_vertical_class"] == "H2"),
        "v1_neither_videos": sorted(r["video_id"] for r in records if r["v1_vertical_class"] == "neither"),
        "v2_not_H1_videos": sorted(
            r["video_id"] for r in records
            if r["v2_vertical_class"] not in ("H1", "both_vacuous")
        ),
    }
    (args.out_dir / "aggregate_scan.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: summary[k] for k in ("v1_vertical_class_counts", "v2_vertical_class_counts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
