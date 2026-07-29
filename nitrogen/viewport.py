"""Classify a harvested video as full-screen gameplay or a stream layout.

The Celeste slice is 97% Twitch VODs, and Twitch VODs are usually STREAM
LAYOUTS: the game occupies a sub-rectangle while a facecam, a chat column, a
splits panel and a donation bar fill the rest. Our own recordings are
full-screen game. Resizing a layout frame to 128px therefore spends most of the
tensor on furniture and shrinks the actual gameplay to a fraction of the
resolution our own data carries — a domain shift and a resolution loss at the
same time, and one that would surface only as "foreign data does not help".

NitroGen ships no game-area bbox (verified: chunk metadata carries
``bbox_controller_overlay`` and nothing else), so the viewport has to be found
here. This module answers the cheaper question first — IS this video
full-screen? — because filtering to full-screen videos removes the confound
outright, whereas cropping introduces a detector that can fail silently.

The signal is temporal: in full-screen gameplay every border strip moves
(camera scroll, parallax, particles), while a layout's furniture strips are
static or near-static. Sampled at three points in the video so a pause menu,
a cutscene or a loading screen cannot decide the verdict alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MOVING_STD = 3.0        # per-pixel temporal std above which a pixel "moves"
BORDER_FRAC = 0.10      # strip thickness as a fraction of each dimension
FULLSCREEN_MIN = 0.25   # every border strip must be at least this alive
PROBE_POINTS = (0.25, 0.5, 0.75)


def border_motion(video_path: Path, at_frac: float, n_frames: int = 150) -> dict | None:
    """Fraction of moving pixels in each border strip at one probe point."""

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * at_frac))
    frames = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) < 30:
        return None

    moving = np.stack(frames).astype(np.float32).std(axis=0) > MOVING_STD
    height, width = moving.shape
    bh = max(1, int(height * BORDER_FRAC))
    bw = max(1, int(width * BORDER_FRAC))
    return {
        "left": float(moving[:, :bw].mean()),
        "right": float(moving[:, -bw:].mean()),
        "top": float(moving[:bh, :].mean()),
        "bottom": float(moving[-bh:, :].mean()),
        "overall": float(moving.mean()),
    }


def classify(video_path: Path) -> dict:
    probes = [p for p in (border_motion(video_path, f) for f in PROBE_POINTS) if p]
    if not probes:
        return {"video_id": video_path.stem, "verdict": "undecodable",
                "probes": 0}
    # Take the most-alive probe: a pause menu or cutscene makes a full-screen
    # video look static, but nothing makes a layout's furniture move.
    best = max(probes, key=lambda p: min(p["left"], p["right"], p["bottom"]))
    edges = min(best["left"], best["right"], best["bottom"])
    return {
        "video_id": video_path.stem,
        "verdict": "fullscreen" if edges >= FULLSCREEN_MIN else "layout",
        "min_border_motion": round(edges, 3),
        "probes": len(probes),
        **{k: round(v, 3) for k, v in best.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    reports = []
    for path in args.videos:
        report = classify(path)
        reports.append(report)
        print(f"{report['video_id']:16s} {report['verdict']:11s} "
              f"min_border={report.get('min_border_motion')} "
              f"overall={report.get('overall')}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=2))
        counts: dict[str, int] = {}
        for report in reports:
            counts[report["verdict"]] = counts.get(report["verdict"], 0) + 1
        print(f"\n{counts}  -> {args.out}")


if __name__ == "__main__":
    main()
