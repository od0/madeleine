"""Scan every speedrun.com candidate for a CelesteTAS-style action HUD.

Goal: reduce ~7,000 linked PC runs to the subset carrying a decodable input
overlay, without downloading a single full video. Each candidate costs one
~6-second low-resolution section (a couple of MB), from which we keep:

  * the panel geometry found by the classical detector, and
  * a PNG crop of each candidate panel, plus one full frame.

Crops are kept deliberately: identifying WHICH panel is an input display (as
opposed to a run timer, a splits column or chat) is a vision judgement, and
saving the evidence means that judgement — by OCR, by a one-time VLM call, or
by eye — never requires re-fetching anything. The scan is the expensive part
and it runs once.

Resumable by design: one JSONL row per video, appended and flushed, and a
video already present in the output is skipped. Kill and restart freely.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np

from harvest.overlay_probe import detect_overlay, fetch_section, read_frames
from harvest.fetch_wild import FetchPolicy

WRITE_LOCK = threading.Lock()
MAX_CROPS = 4          # keep the best few panels per video
CROP_PAD = 6


def format_probe_error(exc: Exception, limit: int = 1000) -> str:
    """Preserve the platform failure reason without emitting unbounded logs."""

    parts = [f"{type(exc).__name__}: {exc}"]
    if isinstance(exc, subprocess.CalledProcessError):
        for label, payload in (("stderr", exc.stderr), ("stdout", exc.stdout)):
            if not payload:
                continue
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8", errors="replace")
            compact = " ".join(str(payload).split())
            if compact:
                parts.append(f"{label}: {compact}")
    return " | ".join(parts)[:limit]


def _save_crops(frame: np.ndarray, report: dict, crops_dir: Path, video_id: str) -> list[str]:
    saved: list[str] = []
    height, width = frame.shape[:2]
    for i, panel in enumerate(report.get("panels", [])[:MAX_CROPS]):
        x, y, w, h = panel["panel_rect"]
        x0 = max(0, x - CROP_PAD); y0 = max(0, y - CROP_PAD)
        x1 = min(width, x + w + CROP_PAD); y1 = min(height, y + h + CROP_PAD)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = frame[y0:y1, x0:x1]
        # Upscale small panels so text stays legible to OCR / a reader.
        if max(crop.shape[:2]) < 200:
            scale = 200.0 / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_NEAREST)
        name = f"{video_id}__p{i}.png"
        cv2.imwrite(str(crops_dir / name), crop)
        saved.append(name)
    return saved


def scan_one(
    row: dict,
    seconds: float,
    crops_dir: Path,
    frames_dir: Path,
    policy: FetchPolicy = FetchPolicy(),
) -> dict:
    video_id = row["video_id"]
    duration = float(row.get("duration_s") or 900)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{video_id}.mp4"
        try:
            # A third of the way in: past intros, inside real play. A single
            # probe point occasionally lands on a cutscene; that is recorded
            # as a weak result rather than a negative.
            fetch_section(row["url"], path, duration / 3.0, seconds, policy)
            found = sorted(Path(tmp).glob(f"{video_id}.*"))
            if not found:
                raise FileNotFoundError("yt-dlp produced no file")
            frames = read_frames(found[0], max_frames=300)
            cap = cv2.VideoCapture(str(found[0]))
            ok, bgr = cap.read()
            cap.release()
        except Exception as exc:                       # noqa: BLE001
            return {"video_id": video_id, "url": row["url"],
                    "error": format_probe_error(exc),
                    "has_overlay": None}
    report = detect_overlay(frames)
    if ok:
        cv2.imwrite(str(frames_dir / f"{video_id}.png"), bgr)
        report["crops"] = _save_crops(bgr, report, crops_dir, video_id)
    report.update({k: row.get(k) for k in
                   ("video_id", "url", "category", "place", "duration_s", "source")})
    report["error"] = None
    # Cells are useful later but bulky; the panels summary is enough per row.
    report.pop("cells", None)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crops-dir", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=1,
                    help="host concurrency; keep at 1 on production IPs")
    ap.add_argument("--yt-dlp", default="/opt/wildenv/bin/yt-dlp")
    ap.add_argument("--deno", default="/usr/local/bin/deno")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-probe videos whose previous row recorded an error "
                         "(YouTube bot-blocks are transient and IP-scoped, so a "
                         "failed row is not a verdict about the video)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.candidates.read_text().splitlines() if l.strip()]
    # Resume by video id. Errored rows count as done by default — but a
    # rate-limit block marks thousands of perfectly good videos as failed, so
    # --retry-errors lets a later pass revisit exactly those.
    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if args.retry_errors and row.get("error"):
                continue
            done.add(row["video_id"])
    todo = [r for r in rows if r["video_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    for directory in (args.crops_dir, args.frames_dir, args.out.parent):
        directory.mkdir(parents=True, exist_ok=True)
    print(f"scanning {len(todo)} videos ({len(done)} already done) "
          f"with {args.workers} workers", flush=True)
    policy = FetchPolicy(yt_dlp_path=args.yt_dlp, deno_path=args.deno)

    hits = 0
    with args.out.open("a") as fh, concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = [pool.submit(scan_one, r, args.seconds, args.crops_dir,
                               args.frames_dir, policy) for r in todo]
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            report = future.result()
            with WRITE_LOCK:
                fh.write(json.dumps(report) + "\n")
                fh.flush()
            if report.get("has_overlay"):
                hits += 1
            if n % 25 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] panels-found={hits}", flush=True)
    print(f"scan complete: {len(todo)} probed, {hits} with candidate panels")


if __name__ == "__main__":
    main()
