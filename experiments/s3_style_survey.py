"""Survey what input-overlay styles Celeste speedruns actually use.

Before committing to a decoder, and before asking Bryan to record a calibration
session, we need to know which overlay style owns the hours. Two are known to
exist in the wild, and they have very different decode difficulty:

  * an OPAQUE key grid (NohBoard and friends) — cells hold two fixed colours,
    trivially thresholdable, the case `harvest/overlay_parser.py` already
    handles for our own mod;
  * a TRANSLUCENT info HUD (CelesteTAS-style action names) — alpha-blended over
    moving game content, so a cell's "released" value tracks whatever is behind
    it and a fixed threshold cannot work.

Calibrating against the wrong one would validate a decoder we never run, so the
sample is stratified across leaderboard rank (run length and play style both
vary strongly with rank) and its frames are rendered to contact sheets for
direct visual classification.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from harvest.overlay_probe import detect_overlay, fetch_section, read_frames

# Rank bands: world-record routes and back-of-the-pack runs differ in length,
# play style, and (we suspect) overlay tooling.
BANDS = [(1, 10), (11, 100), (101, 500), (501, 2000), (2001, 10_000)]


def sample(rows: list[dict], per_band: int) -> list[dict]:
    picked: list[dict] = []
    for low, high in BANDS:
        band = [r for r in rows if r.get("place") and low <= r["place"] <= high]
        # Longest first: more hours per fetch, and long runs are the ones the
        # scaling axis actually wants.
        band.sort(key=lambda r: -(r.get("duration_s") or 0))
        picked.extend(band[:per_band])
    return picked


def probe_one(row: dict, seconds: float, frames_dir: Path) -> dict:
    video_id = row["video_id"]
    start = float(row.get("duration_s") or 900) / 3.0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{video_id}.mp4"
        try:
            fetch_section(row["url"], path, start, seconds)
            found = sorted(Path(tmp).glob(f"{video_id}.*"))
            if not found:
                raise FileNotFoundError("yt-dlp produced no file")
            frames = read_frames(found[0], max_frames=360)
            cap = cv2.VideoCapture(str(found[0]))
            ok, bgr = cap.read()
            cap.release()
            if ok:
                cv2.imwrite(str(frames_dir / f"{video_id}.png"), bgr)
        except Exception as exc:                       # noqa: BLE001
            return {**row, "has_overlay": None,
                    "error": f"{type(exc).__name__}: {exc}"[:200]}
    report = detect_overlay(frames)
    report.update({k: row[k] for k in ("video_id", "url", "category", "place",
                                       "duration_s", "source")})
    report["error"] = None
    return report


def contact_sheet(frames_dir: Path, ids: list[str], out_path: Path,
                  cols: int = 4, tile: tuple[int, int] = (480, 270)) -> int:
    tiles = []
    for video_id in ids:
        image_path = frames_dir / f"{video_id}.png"
        if not image_path.is_file():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = cv2.resize(image, tile, interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (tile[0] - 1, 18), (0, 0, 0), -1)
        cv2.putText(image, video_id, (3, 13), cv2.FONT_HERSHEY_PLAIN, 0.9,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(image)
    if not tiles:
        return 0
    while len(tiles) % cols:
        tiles.append(np.zeros((tile[1], tile[0], 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out_path), np.vstack(rows))
    return len(tiles)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, default=Path("results/wild/candidates.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/wild/style_survey.jsonl"))
    ap.add_argument("--frames-dir", type=Path, default=Path("results/wild/frames"))
    ap.add_argument("--per-band", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.candidates.read_text().splitlines() if l.strip()]
    picked = sample(rows, args.per_band)
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"probing {len(picked)} videos across {len(BANDS)} rank bands")

    reports: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(probe_one, r, args.seconds, args.frames_dir): r
                   for r in picked}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            report = future.result()
            reports.append(report)
            print(f"[{done}/{len(picked)}] {report['video_id']} "
                  f"place={report.get('place')} panels={report.get('n_candidate_panels')} "
                  f"{'ERR' if report.get('error') else ''}")

    with args.out.open("w") as fh:
        for report in reports:
            fh.write(json.dumps(report) + "\n")

    ok = [r for r in reports if not r.get("error")]
    ok.sort(key=lambda r: (r.get("place") or 0))
    sheets = 0
    for start in range(0, len(ok), 12):
        batch = [r["video_id"] for r in ok[start:start + 12]]
        made = contact_sheet(args.frames_dir, batch,
                             args.out.parent / f"sheet_{start // 12:02d}.png")
        sheets += bool(made)
    print(f"\nprobed ok: {len(ok)}/{len(reports)}; wrote {sheets} contact sheets")


if __name__ == "__main__":
    main()
