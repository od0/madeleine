"""Cheap keyboard-overlay detection, without downloading whole videos.

The speedrun.com Celeste leaderboards link roughly 6,400 PC runs. Most are
useless to us because they carry no input overlay, and a full 720p60 fetch of a
two-hour run costs gigabytes — so the filter has to run BEFORE the fetch. This
module pulls an ~8-second section at low resolution (a couple of MB) and decides
from the pixels whether an overlay is present.

The detector is classical and cheap by design; per the brief, VLMs do one-time
layout inference and nothing per-frame. It keys on a signature that an input
overlay has and the game does not:

  * A NohBoard-style panel is SPATIALLY STATIC — its background pixels do not
    change at all across a window of frames, where the game's pixels change
    almost everywhere (camera scroll, parallax, particles, animation).
  * Its key cells are BIMODAL — each cell sits at one of exactly two levels
    (up colour, down colour), toggling between them, rather than taking a
    continuous spread of values the way game pixels do.

Neither signal alone suffices: a letterbox bar is static but not bimodal, and a
flickering game light is bimodal but not static, so the detector requires both
in the same rectangular island and then demands several such islands clustered
together. That is also exactly the cell table the layout step needs, so the same
function serves detection and (later) per-cell decoding.

Output is one JSONL row per video: video_id, has_overlay, score, panel_rect,
cells, so the fetch stage can consume a filtered list.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from harvest.fetch_wild import FetchPolicy

# An overlay must show at least this many bimodal cells clustered together.
MIN_CELLS = 4
# A cell must be at least this rectangular (filled pixels / bbox area).
MIN_FILL = 0.55
# Cell area bounds as a fraction of frame area — rejects single pixels and
# whole-screen flashes.
MIN_CELL_AREA_FRAC = 2e-5
MAX_CELL_AREA_FRAC = 5e-3
# A cell must actually be pressed sometimes and released sometimes across the
# probe window, or it is a static decoration rather than a key.
MIN_DUTY = 0.005
MAX_DUTY = 0.95
# Fraction of frames whose value must fall near one of the two levels for a
# pixel to count as bimodal.
BIMODAL_PURITY = 0.90


def build_probe_command(
    url: str,
    out_path: Path,
    start_s: float,
    seconds: float = 8.0,
    policy: FetchPolicy = FetchPolicy(),
) -> list[str]:
    """Build one deliberately low-rate, Deno-backed probe command."""

    end_s = start_s + seconds
    return [
        policy.yt_dlp_path, "--quiet", "--no-warnings",
        "--no-playlist",
        "--js-runtimes", f"deno:{policy.deno_path}",
        "--concurrent-fragments", "1",
        "--sleep-requests", str(policy.sleep_requests_s),
        "--sleep-interval", str(policy.sleep_min_s),
        "--max-sleep-interval", str(policy.sleep_max_s),
        "--retries", "5",
        "--fragment-retries", "5",
        "--download-sections", f"*{start_s:.0f}-{end_s:.0f}",
        # force_keyframes is deliberately OFF: it re-encodes, which is slow and
        # would smear the hard cell edges the detector depends on.
        "-f", "bv*[height<=480]/bv*[height<=720]/b",
        "-o", str(out_path), url,
    ]


def fetch_section(
    url: str,
    out_path: Path,
    start_s: float,
    seconds: float = 8.0,
    policy: FetchPolicy = FetchPolicy(),
) -> None:
    """Download one short section without hiding the required JS runtime."""

    cmd = build_probe_command(url, out_path, start_s, seconds, policy)
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)


def read_frames(path: Path, max_frames: int = 480, stride: int = 1) -> np.ndarray:
    """Read up to max_frames grayscale frames as a [T,H,W] uint8 array."""

    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    index = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        index += 1
    cap.release()
    if not frames:
        raise ValueError(f"{path}: no decodable frames")
    return np.stack(frames)


def bimodal_static_mask(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (bimodal_mask, duty) for a [T,H,W] stack.

    A pixel is bimodal when almost every sample sits near its own min or its own
    max, with a real gap between the two levels — the signature of a cell that
    toggles between an up colour and a down colour. ``duty`` is the fraction of
    frames spent at the high level, used to reject decorations that never move.
    """

    stack = frames.astype(np.float32)
    low = stack.min(axis=0)
    high = stack.max(axis=0)
    spread = high - low
    midpoint = (low + high) / 2.0

    # A pixel needs a real two-level gap; 40 is the same order as the mod's own
    # up/down separation (0x28 vs 0xff) seen through video compression.
    has_gap = spread >= 40.0

    above = stack >= midpoint
    duty = above.mean(axis=0)

    # Purity: samples must cluster AT the two levels, not spread between them.
    near_high = (stack >= high - spread * 0.25).mean(axis=0)
    near_low = (stack <= low + spread * 0.25).mean(axis=0)
    pure = (near_high + near_low) >= BIMODAL_PURITY

    toggles = (duty >= MIN_DUTY) & (duty <= MAX_DUTY)
    return (has_gap & pure & toggles), duty


def find_cells(frames: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Rectangular bimodal islands — candidate key cells. (x, y, w, h)."""

    mask, _ = bimodal_static_mask(frames)
    height, width = mask.shape
    frame_area = height * width
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=4
    )
    cells: list[tuple[int, int, int, int]] = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if not (MIN_CELL_AREA_FRAC * frame_area <= area <= MAX_CELL_AREA_FRAC * frame_area):
            continue
        if w < 3 or h < 3:
            continue
        if area / float(w * h) < MIN_FILL:      # must be rectangle-ish
            continue
        cells.append((int(x), int(y), int(w), int(h)))
    return cells


def cluster_cells(
    cells: list[tuple[int, int, int, int]], shape: tuple[int, int]
) -> list[list[tuple[int, int, int, int]]]:
    """Group cells into panels by proximity (single-link, gap-based).

    A speedrun frame carries SEVERAL static changing panels — a run timer, a
    LiveSplit column, and (if we are lucky) an input display. Taking one
    bounding box over every cell spans the whole screen and rejects all of
    them, so panels are separated first and judged individually.
    """

    height, width = shape
    # Cells of one panel sit within a few cell-widths of each other.
    gap = max(12, int(0.05 * max(height, width)))
    remaining = list(cells)
    groups: list[list[tuple[int, int, int, int]]] = []
    while remaining:
        group = [remaining.pop()]
        changed = True
        while changed:
            changed = False
            for cell in list(remaining):
                x, y, w, h = cell
                for gx, gy, gw, gh in group:
                    if (x < gx + gw + gap and gx < x + w + gap
                            and y < gy + gh + gap and gy < y + h + gap):
                        group.append(cell)
                        remaining.remove(cell)
                        changed = True
                        break
        groups.append(group)
    return groups


def _panel_of(group: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x0 = min(c[0] for c in group); y0 = min(c[1] for c in group)
    x1 = max(c[0] + c[2] for c in group); y1 = max(c[1] + c[3] for c in group)
    return (x0, y0, x1 - x0, y1 - y0)


def _score_group(
    group: list[tuple[int, int, int, int]], shape: tuple[int, int]
) -> float:
    if len(group) < MIN_CELLS:
        return 0.0
    height, width = shape
    panel = _panel_of(group)
    frac = (panel[2] * panel[3]) / float(height * width)
    if frac > 0.25:
        return 0.0
    return float(len(group)) * (1.0 - frac)


def cell_dynamics(
    frames: np.ndarray, cells: list[tuple[int, int, int, int]], fps: float
) -> list[dict]:
    """Per-cell toggle statistics — the resolution-free identity signal.

    Reading a panel's text needs pixels we do not have at probe resolution, but
    a panel's BEHAVIOUR separates the three families cleanly in an 8-second
    window: input-HUD cells toggle at gameplay frequency with varied duty
    cycles, a run timer churns its digits at a fixed cadence, and a splits
    column barely changes at all between splits.
    """

    stats: list[dict] = []
    for x, y, w, h in cells:
        series = frames[:, y:y + h, x:x + w].reshape(len(frames), -1).mean(axis=1)
        low, high = float(series.min()), float(series.max())
        if high - low < 8.0:
            stats.append({"transitions_per_s": 0.0, "duty": 0.0})
            continue
        on = series >= (low + high) / 2.0
        transitions = int(np.count_nonzero(on[1:] != on[:-1]))
        stats.append({
            "transitions_per_s": round(transitions / max(len(frames) / fps, 1e-6), 3),
            "duty": round(float(on.mean()), 3),
        })
    return stats


# A key that a human is playing with toggles a few times a second at most, and
# is neither permanently down nor never pressed.
GAMEPLAY_RATE = (0.15, 12.0)
GAMEPLAY_DUTY = (0.01, 0.92)
MIN_ACTIVE_CELLS = 3


def score_input_hud(stats: list[dict]) -> dict:
    """How much does this panel behave like a live input display?"""

    active = [
        s for s in stats
        if GAMEPLAY_RATE[0] <= s["transitions_per_s"] <= GAMEPLAY_RATE[1]
        and GAMEPLAY_DUTY[0] <= s["duty"] <= GAMEPLAY_DUTY[1]
    ]
    duties = [s["duty"] for s in active]
    # Real keys are used unequally (movement constantly, dash in bursts); a
    # timer's digit cells all behave alike, so duty spread discriminates.
    spread = float(np.std(duties)) if len(duties) > 1 else 0.0
    return {
        "n_active_cells": len(active),
        "duty_spread": round(spread, 3),
        "is_input_hud": bool(len(active) >= MIN_ACTIVE_CELLS and spread >= 0.03),
        "mean_rate": round(float(np.mean([s["transitions_per_s"] for s in active])), 3)
        if active else 0.0,
    }


def detect_overlay(frames: np.ndarray, fps: float = 60.0) -> dict:
    cells = find_cells(frames)
    shape = frames.shape[1:]
    groups = cluster_cells(cells, shape)
    scored = sorted(
        ((_score_group(g, shape), g) for g in groups),
        key=lambda t: t[0], reverse=True,
    )
    best_score, best_group = (scored[0] if scored else (0.0, []))
    panels = []
    for s, group in scored:
        if s <= 0.0:
            continue
        verdict = score_input_hud(cell_dynamics(frames, group, fps))
        panels.append({
            "panel_rect": list(_panel_of(group)), "n_cells": len(group),
            "score": round(s, 3), **verdict,
        })
    hud = [p for p in panels if p["is_input_hud"]]
    # Prefer the most cell-rich input-HUD panel; a video with none is not a
    # harvest candidate no matter how many timers and splits panels it shows.
    hud.sort(key=lambda p: -p["n_active_cells"])
    return {
        "has_overlay": bool(best_score > 0.0),
        "has_input_hud": bool(hud),
        "hud_panel": hud[0] if hud else None,
        "score": round(best_score, 3),
        "n_cells": len(best_group),
        "panel_rect": list(_panel_of(best_group)) if best_group else None,
        "cells": [list(c) for c in best_group],
        "panels": panels,
        "n_candidate_panels": len(panels),
        "probe_shape": [int(frames.shape[2]), int(frames.shape[1])],
    }


def probe_video(
    url: str, video_id: str, start_s: float, seconds: float = 8.0
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{video_id}.mp4"
        try:
            fetch_section(url, path, start_s, seconds)
            found = sorted(Path(tmp).glob(f"{video_id}.*"))
            if not found:
                raise FileNotFoundError("yt-dlp produced no file")
            frames = read_frames(found[0])
        except Exception as exc:                      # noqa: BLE001 - reported, not raised
            return {"video_id": video_id, "url": url, "has_overlay": None,
                    "error": f"{type(exc).__name__}: {exc}"[:300]}
    report = detect_overlay(frames)
    report.update({"video_id": video_id, "url": url, "error": None,
                   "probe_start_s": start_s, "probe_frames": int(len(frames))})
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, required=True,
                    help="JSONL with video_id, url, and optionally duration_s")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["video_id"] for l in args.out.read_text().splitlines() if l.strip()}
    todo = [r for r in rows if r["video_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        for i, row in enumerate(todo, 1):
            # Probe a third of the way in: past intros and menus, inside real play.
            start = float(row.get("duration_s") or 600) / 3.0
            report = probe_video(row["url"], row["video_id"], start, args.seconds)
            fh.write(json.dumps(report) + "\n")
            fh.flush()
            flag = ("OVERLAY" if report.get("has_overlay")
                    else "err" if report.get("error") else "none")
            print(f"[{i}/{len(todo)}] {report['video_id']}: {flag} "
                  f"cells={report.get('n_cells')} score={report.get('score')}")


if __name__ == "__main__":
    main()
