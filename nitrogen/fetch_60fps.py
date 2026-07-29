"""Fetch harvested Celeste source video at native 60 fps to match 60 Hz labels.

Designed to run ON THE POD. The claim that YouTube blocks datacenter IPs was
tested on this pod (LightEdge AS11320, US datacenter) and is false: a 720p60
download succeeded first try, unauthenticated. Fetching where the disk and GPU
already live removes a tens-of-GB rsync from every downstream step.

Why 60 fps matters more than resolution here: the labels are one row per source
frame (60 Hz for most Celeste videos). A 480p rendition is 30 fps, which forces
an unverifiable "video frame i == source frame 2i" mapping and hides inputs that
land on odd frames. E4 measured the cost of misalignment at ~4.5% macro-F1 per
frame, so a rendition whose frame rate equals the label rate is worth far more
than extra pixels.

Works from the committed chunk index (data/celeste_chunk_index.parquet), not the
19 GB actions tree, so the pod needs no dataset copy. Append-only report, safe to
re-run: completed videos are skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

# Prefer >=50 fps at <=720p (one video frame per label frame); degrade
# gracefully rather than failing a video that has no high-fps rendition.
FORMAT = "bv*[fps>=50][height<=720]/bv*[fps>=50][height<=1080]/bv*[height<=720]/bv*"

_report_lock = threading.Lock()


def video_table(chunk_index: Path) -> list[dict]:
    """One row per video: id, source, url, label rate, chunk count."""
    t = pq.read_table(chunk_index).to_pydict()
    out: dict[str, dict] = {}
    for i, vid in enumerate(t["video_id"]):
        rec = out.get(vid)
        if rec is None:
            out[vid] = {
                "video_id": vid,
                "source": t["source"][i],
                "url": t["url"][i],
                "label_hz": float(t["grid_hz"][i]),
                "chunks": 1,
            }
        else:
            rec["chunks"] += 1
    for rec in out.values():
        rec["chunk_hours"] = rec["chunks"] * 20 / 3600
    return list(out.values())


def probe(path: Path) -> dict:
    """Measured properties of what we actually got — never assumed."""
    fields = "stream=width,height,r_frame_rate,nb_read_packets:format=duration"
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", fields, "-of", "json", str(path)],
        capture_output=True, text=True, timeout=600,
    )
    info = json.loads(proc.stdout or "{}")
    st = (info.get("streams") or [{}])[0]
    num, _, den = (st.get("r_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {
        "width": st.get("width"),
        "height": st.get("height"),
        "fps": round(fps, 3),
        "frames": int(st.get("nb_read_packets") or 0),
        "duration_s": round(float((info.get("format") or {}).get("duration") or 0), 2),
    }


def fetch_one(rec: dict, out_dir: Path, report_path: Path, fmt: str) -> dict:
    import yt_dlp

    dest = out_dir / f"{rec['video_id']}.mp4"
    row = {**rec, "requested_format": fmt,
           "fetched_at": datetime.now(timezone.utc).isoformat()}
    try:
        if not (dest.exists() and dest.stat().st_size > 0):
            opts = {
                "format": fmt,
                "outtmpl": str(out_dir / f"{rec['video_id']}.%(ext)s"),
                "merge_output_format": "mp4",
                "quiet": True,
                "no_warnings": True,
                "retries": 2,
                "fragment_retries": 3,
                "socket_timeout": 30,
                "noprogress": True,
            }
            with yt_dlp.YoutubeDL(opts) as y:
                y.download([rec["url"]])
        found = next((p for p in out_dir.glob(f"{rec['video_id']}.*")
                      if p.suffix in (".mp4", ".mkv", ".webm")), None)
        if found is None:
            raise FileNotFoundError("download reported success but no file")
        info = probe(found)
        # ratio 1.0 means one video frame per label frame — the whole point.
        ratio = (rec["label_hz"] / info["fps"]) if info["fps"] else None
        row.update({
            "status": "ok", "error": None, "path": found.name,
            "bytes": found.stat().st_size, **info,
            "label_to_video_ratio": round(ratio, 4) if ratio else None,
            "aligned_1to1": bool(ratio and abs(ratio - 1.0) < 0.02),
        })
    except Exception as exc:  # link rot, geo, removed, format gone
        row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]})

    with _report_lock:
        with report_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-index", type=Path, default=Path("data/celeste_chunk_index.parquet"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video-ids", type=str, default=None,
                    help="comma-separated ids, or a path to a file of ids")
    ap.add_argument("--alive-census", type=Path, default=None,
                    help="jsonl of {video_id,status} to skip known-dead links")
    ap.add_argument("--sources", type=str, default="youtube,twitch")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--format", dest="fmt", type=str, default=FORMAT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    videos = video_table(args.chunk_index)

    if args.video_ids:
        p = Path(args.video_ids)
        wanted = set((p.read_text().split() if p.exists()
                      else args.video_ids.split(",")))
        videos = [v for v in videos if v["video_id"] in wanted]
    if args.alive_census and args.alive_census.exists():
        dead = {json.loads(l)["video_id"] for l in args.alive_census.open()
                if json.loads(l)["status"] != "alive"}
        videos = [v for v in videos if v["video_id"] not in dead]
    sources = [s.strip() for s in args.sources.split(",")]
    videos = [v for v in videos if v["source"] in sources]
    # biggest first: more label-hours per download, and failures surface early
    videos.sort(key=lambda v: -v["chunk_hours"])
    if args.limit:
        videos = videos[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "fetch60_report.jsonl"
    done = set()
    if report_path.exists():
        for line in report_path.open():
            r = json.loads(line)
            if r.get("status") == "ok":
                done.add(r["video_id"])
    todo = [v for v in videos if v["video_id"] not in done]

    total_h = sum(v["chunk_hours"] for v in todo)
    print(f"videos selected={len(videos)} already-ok={len(done)} todo={len(todo)} "
          f"label-hours-todo={total_h:.1f}")
    if args.dry_run:
        for v in todo[:10]:
            print(f"  {v['video_id']:>13} {v['source']:>7} {v['chunk_hours']:6.2f}h {v['url']}")
        return

    ok = fail = 0
    hours = 0.0
    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch_one, v, args.out, report_path, args.fmt) for v in todo]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            r = fut.result()
            if r["status"] == "ok":
                ok += 1
                hours += (r.get("duration_s") or 0) / 3600
                flag = "" if r.get("aligned_1to1") else f" RATIO={r.get('label_to_video_ratio')}"
                print(f"[{i}/{len(todo)}] ok {r['video_id']} "
                      f"{r.get('width')}x{r.get('height')}@{r.get('fps')} "
                      f"{(r.get('duration_s') or 0)/3600:.2f}h{flag}", flush=True)
            else:
                fail += 1
                print(f"[{i}/{len(todo)}] FAIL {r['video_id']} {r['error'][:90]}", flush=True)
    print(f"DONE ok={ok} failed={fail} video-hours={hours:.1f}")


if __name__ == "__main__":
    main()
