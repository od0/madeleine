"""Measure rendered answer-key geometry directly from session pixels.

The 2026-07-26 masking audit found the declared input_overlay rect misses
the top of the rendered key cells (the mod's on-screen transform is not the
one specs/overlay_spec.md assumes). Manifest rects must therefore come from
measured pixels, not from logical-space constants. This script is that
measurement: for each session it correlates per-pixel brightness with
engine-truth key state and reports, per declared masked region, the true
extent of state-correlated and static-widget pixels next to the declared
rect.

Method, per session:
  1. From alignment.parquet + truth.parquet, pick up to --per-bucket frames
     per bucket (each of the 7 keys down; all keys up), evenly spread over
     the video, decode_status ok, non-duplicate, input_active only.
  2. One sequential pass (grab all, retrieve selected) accumulates per-pixel
     sums for key-down/key-up per key, plus mean/std over all samples.
  3. A pixel is state-correlated for key k when |mean_down_k - mean_up_k|
     exceeds --state-thresh. A pixel is static-dark (the opaque backing bar)
     when its std is < 6 and mean < 30.
  4. Report the bounding box of state-correlated pixels near each declared
     region, the static-dark bar box containing the input_overlay cells, and
     whether the declared rect covers them.

Output: one JSON report per session (--out), plus a human-readable table.
Nothing here edits manifests; corrections are a separate, reviewed step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

KEY_ORDER = ["left", "right", "up", "down", "jump", "dash", "grab"]

STATIC_STD_MAX = 6.0
STATIC_DARK_MAX = 30.0


def _spread(indices: np.ndarray, count: int) -> np.ndarray:
    if len(indices) <= count:
        return indices
    pick = np.linspace(0, len(indices) - 1, count).round().astype(int)
    return indices[np.unique(pick)]


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    # [x0, y0, x1, y1] half-open, capture pixels.
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _covers(rect_xyxy: list[int], bbox: list[int] | None) -> bool:
    if bbox is None:
        return True
    return (
        rect_xyxy[0] <= bbox[0]
        and rect_xyxy[1] <= bbox[1]
        and rect_xyxy[2] >= bbox[2]
        and rect_xyxy[3] >= bbox[3]
    )


def _profile_extent(
    mask: np.ndarray,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    row_min_hits: float,
    col_min_hits: float,
) -> list[int] | None:
    """Extent of a dense widget inside a search zone, ignoring scatter.

    Raw bounding boxes overstate widget extent whenever ordinary gameplay
    pixels near the widget correlate with key state (short sessions have too
    little visual diversity to average that out). Widget cells are DENSE:
    a cell row lights up hundreds of contiguous columns. So threshold the
    row/column hit profiles instead of taking the bbox of every hit.
    """
    r0, r1 = row_range
    c0, c1 = col_range
    zone = mask[r0:r1, c0:c1]
    row_hits = zone.sum(axis=1)
    rows = np.nonzero(row_hits >= row_min_hits * (c1 - c0))[0]
    if len(rows) == 0:
        return None
    cell_rows = zone[rows.min() : rows.max() + 1]
    col_hits = cell_rows.sum(axis=0)
    cols = np.nonzero(col_hits >= col_min_hits * len(cell_rows))[0]
    if len(cols) == 0:
        return None
    return [
        int(c0 + cols.min()),
        int(r0 + rows.min()),
        int(c0 + cols.max()) + 1,
        int(r0 + rows.max()) + 1,
    ]


def measure_session(
    session_dir: Path, per_bucket: int, state_thresh: float
) -> dict:
    manifest = json.loads((session_dir / "manifest.json").read_text())
    width, height = manifest["capture"]["resolution"]
    regions = {r["name"]: r for r in manifest["masked_regions"]}

    truth = pq.read_table(session_dir / "truth.parquet")
    alignment = pq.read_table(session_dir / "alignment.parquet")
    status = np.asarray(alignment["decode_status"].to_pylist())
    dup = np.asarray(alignment["is_duplicate"].to_pylist(), dtype=bool)
    engine_idx = np.asarray(
        alignment["engine_frame_idx"].to_pylist(), dtype=np.int64
    )
    truth_base = int(truth["frame_idx"][0].as_py())
    keys_all = np.stack(
        [np.asarray(truth[k].to_pylist(), dtype=bool) for k in KEY_ORDER], axis=1
    )
    active_all = np.asarray(truth["input_active"].to_pylist(), dtype=bool)

    rows = engine_idx - truth_base
    in_range = (rows >= 0) & (rows < len(keys_all)) & (status == "ok") & (~dup)
    rows_safe = np.clip(rows, 0, len(keys_all) - 1)
    frame_keys = keys_all[rows_safe] & in_range[:, None]
    frame_active = active_all[rows_safe] & in_range

    chosen: set[int] = set()
    eligible = frame_active
    for k in range(len(KEY_ORDER)):
        down = np.nonzero(eligible & frame_keys[:, k])[0]
        chosen.update(_spread(down, per_bucket).tolist())
    all_up = np.nonzero(eligible & ~frame_keys.any(axis=1))[0]
    chosen.update(_spread(all_up, per_bucket).tolist())
    if not chosen:
        raise SystemExit(f"{session_dir.name}: no eligible sample frames")

    sum_down = np.zeros((len(KEY_ORDER), height, width), np.float64)
    sum_up = np.zeros((len(KEY_ORDER), height, width), np.float64)
    n_down = np.zeros(len(KEY_ORDER), np.int64)
    n_up = np.zeros(len(KEY_ORDER), np.int64)
    g_sum = np.zeros((height, width), np.float64)
    g_sumsq = np.zeros((height, width), np.float64)
    n_total = 0

    cap = cv2.VideoCapture(str(session_dir / "video.mkv"))
    last = max(chosen)
    for video_frame in range(last + 1):
        if not cap.grab():
            raise SystemExit(
                f"{session_dir.name}: video ended at frame {video_frame}"
            )
        if video_frame not in chosen:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            raise SystemExit(
                f"{session_dir.name}: retrieve failed at frame {video_frame}"
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        kv = frame_keys[video_frame]
        for k in range(len(KEY_ORDER)):
            if kv[k]:
                sum_down[k] += gray
                n_down[k] += 1
            else:
                sum_up[k] += gray
                n_up[k] += 1
        g_sum += gray
        g_sumsq += gray * gray
        n_total += 1
    cap.release()

    mean = g_sum / n_total
    std = np.sqrt(np.maximum(g_sumsq / n_total - mean * mean, 0.0))
    static_dark = (std < STATIC_STD_MAX) & (mean < STATIC_DARK_MAX)

    state = np.zeros((height, width), bool)
    per_key_boxes: dict[str, list[int] | None] = {}
    for k, name in enumerate(KEY_ORDER):
        if n_down[k] < 10 or n_up[k] < 10:
            per_key_boxes[name] = None
            continue
        diff = np.abs(sum_down[k] / n_down[k] - sum_up[k] / n_up[k])
        key_mask = diff > state_thresh
        state |= key_mask
        per_key_boxes[name] = _bbox(key_mask)

    report: dict = {
        "session_id": session_dir.name,
        "capture_px": [width, height],
        "samples": {
            "total": int(n_total),
            "per_key_down": {
                name: int(n_down[k]) for k, name in enumerate(KEY_ORDER)
            },
        },
        "state_thresh": state_thresh,
        "per_key_state_bbox": per_key_boxes,
        "regions": {},
    }

    def rect_xyxy(region: dict) -> list[int]:
        x, y, w, h = region["rect_px"]
        return [x, y, x + w, y + h]

    remaining_state = state.copy()

    if "input_overlay" in regions:
        rect = rect_xyxy(regions["input_overlay"])
        # Search zone: declared columns padded, from well above the declared
        # rect to the frame bottom.
        pad_x, pad_y = 40, int(0.12 * height)
        zone_rows = (max(0, rect[1] - pad_y), height)
        zone_cols = (max(0, rect[0] - pad_x), min(width, rect[2] + pad_x))
        cells = _profile_extent(
            state, zone_rows, zone_cols, row_min_hits=0.15, col_min_hits=0.3
        )
        raw = _bbox(
            state[zone_rows[0] : zone_rows[1], zone_cols[0] : zone_cols[1]]
        )
        if raw is not None:
            raw = [
                raw[0] + zone_cols[0],
                raw[1] + zone_rows[0],
                raw[2] + zone_cols[0],
                raw[3] + zone_rows[0],
            ]
        bar = None
        if cells is not None:
            # The opaque backing bar: static-dark rows around the cells that
            # span most of the widget columns.
            bar = _profile_extent(
                static_dark,
                (max(0, cells[1] - pad_y), height),
                (max(0, min(cells[0], rect[0]) - pad_x),
                 min(width, max(cells[2], rect[2]) + pad_x)),
                row_min_hits=0.5,
                col_min_hits=0.5,
            )
        report["regions"]["input_overlay"] = {
            "declared_xyxy": rect,
            "state_bbox_xyxy": cells,
            "state_raw_bbox_in_zone_xyxy": raw,
            "static_bar_bbox_xyxy": bar,
            "declared_covers_state": _covers(rect, cells),
            "declared_covers_bar": _covers(rect, bar),
        }
        nb = np.zeros_like(state)
        nb[zone_rows[0] :, zone_cols[0] : zone_cols[1]] = True
        remaining_state &= ~nb

    if "wild_overlay" in regions:
        rect = rect_xyxy(regions["wild_overlay"])
        pad = 60
        zone_rows = (max(0, rect[1] - pad), min(height, rect[3] + pad))
        zone_cols = (max(0, rect[0] - pad), min(width, rect[2] + pad))
        nb = np.zeros_like(state)
        nb[zone_rows[0] : zone_rows[1], zone_cols[0] : zone_cols[1]] = True
        # The wild panel's buttons are sparser than the opaque cells; a
        # gentler row threshold, but still profile-based.
        panel_state = _profile_extent(
            state, zone_rows, zone_cols, row_min_hits=0.05, col_min_hits=0.05
        )
        report["regions"]["wild_overlay"] = {
            "declared_xyxy": rect,
            "state_bbox_xyxy": panel_state,
            "state_raw_bbox_in_zone_xyxy": _bbox(state & nb),
            "declared_covers_state": _covers(rect, panel_state),
        }
        remaining_state &= ~nb

    if "frame_index_strip" in regions:
        rect = rect_xyxy(regions["frame_index_strip"])
        # Std cannot separate the strip from ordinary gameplay pixels (both
        # churn); the strip's coverage evidence is that frame-index decode
        # succeeded on every aligned frame, which requires the cells inside
        # the rect. Only state-correlated pixels are meaningful here.
        pad = 20
        nb = np.zeros_like(state)
        nb[: rect[3] + pad, max(0, rect[0] - pad) : rect[2] + pad] = True
        strip_state = _bbox(state & nb)
        report["regions"]["frame_index_strip"] = {
            "declared_xyxy": rect,
            "state_bbox_xyxy": strip_state,
            "declared_covers_state": _covers(rect, strip_state),
            "note": "coverage evidence is frame-index decode success",
        }
        remaining_state &= ~nb

    # Anything state-correlated outside every declared neighborhood is either
    # genuine gameplay correlation or an undeclared widget; report it.
    report["state_pixels_outside_declared_neighborhoods"] = {
        "count": int(remaining_state.sum()),
        "bbox_xyxy": _bbox(remaining_state),
    }
    return report, mean, std, state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-bucket", type=int, default=120)
    ap.add_argument("--state-thresh", type=float, default=60.0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for session_dir in args.sessions:
        report, mean, std, state = measure_session(
            session_dir, args.per_bucket, args.state_thresh
        )
        out_path = args.out / f"{report['session_id']}.json"
        out_path.write_text(json.dumps(report, indent=2))
        np.savez_compressed(
            args.out / f"{report['session_id']}_meanstd.npz",
            mean=mean.astype(np.float16),
            std=std.astype(np.float16),
            state=state,
        )
        io = report["regions"].get("input_overlay", {})
        print(
            f"{report['session_id']:36s} "
            f"declared={io.get('declared_xyxy')} "
            f"cells={io.get('state_bbox_xyxy')} "
            f"bar={io.get('static_bar_bbox_xyxy')} "
            f"covers_state={io.get('declared_covers_state')}"
        )


if __name__ == "__main__":
    main()
