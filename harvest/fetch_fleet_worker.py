"""Sequential full-video fetch worker for a completion-gated fleet queue.

The queue is deliberately serial: one process performs at most one source
extraction at a time on a public IP.  Durable completion is determined only by
the R2 ``upload_complete.json`` marker written by :mod:`harvest.worker_wild`
after SHA-256 readback.  Local partial downloads and failed progress rows are
therefore resumable, but never mistaken for published raw evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import time
from typing import Any

from harvest.fetch_wild import FetchPolicy
from harvest.worker_wild import run_worker, worker_preflight


FORMAT_VERSION = "madeleine.wild-fetch-fleet-progress.v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_SOURCE_BLOCK_MARKERS = (
    "sign in to confirm you're not a bot",
    "sign in to confirm you’re not a bot",
    "http error 429",
    "too many requests",
)


def format_fetch_error(exc: Exception, limit: int = 1000) -> str:
    """Keep a bounded source failure reason without importing probe tooling."""

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


def is_source_block_error(exc: Exception) -> bool:
    """Return true only for errors that implicate the worker IP, not the video."""

    payloads: list[str] = [str(exc)]
    if isinstance(exc, subprocess.CalledProcessError):
        payloads.extend(str(value) for value in (exc.stderr, exc.stdout) if value)
    normalized = " ".join(payloads).lower()
    return any(marker in normalized for marker in _SOURCE_BLOCK_MARKERS)


def load_queue(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every queue row must be a JSON object")
    video_ids = [str(row.get("video_id", "")) for row in rows]
    if any(not _SAFE_ID.fullmatch(video_id) for video_id in video_ids):
        raise ValueError("queue contains a missing or unsafe video_id")
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("queue contains duplicate video_id rows")
    for row in rows:
        if not row.get("url") or row.get("duration_s") is None:
            raise ValueError(f"{row['video_id']}: queue row lacks URL or duration_s")
    return rows


def remote_complete(remote_dir: str) -> bool:
    result = subprocess.run(
        ["rclone", "lsf", remote_dir, "--max-depth", "1", "--files-only"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = " ".join((result.stderr or result.stdout).split())[:1000]
        raise RuntimeError(f"cannot audit remote completion for {remote_dir}: {detail}")
    return "upload_complete.json" in {
        line.strip() for line in result.stdout.splitlines() if line.strip()
    }


def append_progress(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def run_queue(
    queue: list[dict[str, Any]],
    out_root: Path,
    remote_root: str,
    progress: Path,
    policy: FetchPolicy,
    min_free_gb: float,
    inter_video_sleep_min_s: float = 0.0,
    inter_video_sleep_max_s: float = 0.0,
    stop_on_source_block: bool = True,
    delete_local_after_verified: bool = False,
) -> dict[str, int]:
    if ":" not in remote_root:
        raise ValueError("remote_root must be an rclone remote path")
    out_root.mkdir(parents=True, exist_ok=True)
    if inter_video_sleep_min_s < 0 or inter_video_sleep_max_s < inter_video_sleep_min_s:
        raise ValueError("invalid inter-video sleep range")
    counts = {"queued": len(queue), "ok": 0, "error": 0, "skipped": 0}
    min_free_bytes = int(min_free_gb * 1024**3)

    for index, candidate in enumerate(queue, 1):
        video_id = str(candidate["video_id"])
        remote_dir = f"{remote_root.rstrip('/')}/{video_id}"
        if remote_complete(remote_dir):
            counts["skipped"] += 1
            print(f"[{index}/{len(queue)}] {video_id}: skip-complete", flush=True)
            continue
        free_bytes = shutil.disk_usage(out_root).free
        if free_bytes < min_free_bytes:
            raise RuntimeError(
                f"free-space gate failed before {video_id}: "
                f"{free_bytes / 1024**3:.2f} GiB < {min_free_gb:.2f} GiB"
            )

        started_at = datetime.now(timezone.utc).isoformat()
        try:
            completion = run_worker(
                candidate, out_root, remote_root, policy=policy
            )
            status = "ok"
            error = None
            published_bytes = int(completion["total_bytes"])
            source_blocked = False
            if delete_local_after_verified:
                local_directory = out_root / video_id
                if local_directory.parent.resolve() != out_root.resolve():
                    raise ValueError("refusing to clean a path outside the output root")
                shutil.rmtree(local_directory)
        except Exception as exc:  # noqa: BLE001 - one dead source must not stop a fleet
            status = "error"
            error = format_fetch_error(exc)
            published_bytes = 0
            source_blocked = is_source_block_error(exc)
        counts[status] += 1
        append_progress(progress, {
            "format_version": FORMAT_VERSION,
            "index": index,
            "video_id": video_id,
            "status": status,
            "error": error,
            "remote_dir": remote_dir,
            "published_bytes": published_bytes,
            "source_blocked": source_blocked,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"[{index}/{len(queue)}] {video_id}: {status}", flush=True)
        if source_blocked and stop_on_source_block:
            print(
                f"source-block gate tripped after {video_id}; preserving queue tail",
                flush=True,
            )
            break
        if index < len(queue) and inter_video_sleep_max_s > 0:
            delay = random.uniform(
                inter_video_sleep_min_s, inter_video_sleep_max_s
            )
            print(f"inter-video cooldown: {delay:.1f}s", flush=True)
            time.sleep(delay)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--inter-video-sleep-min-s", type=float, default=0.0)
    parser.add_argument("--inter-video-sleep-max-s", type=float, default=0.0)
    parser.add_argument(
        "--no-stop-on-source-block", action="store_true",
        help="continue after a detected IP-wide source block (unsafe by default)",
    )
    parser.add_argument(
        "--delete-local-after-verified", action="store_true",
        help="remove one local video only after complete SHA-256 R2 readback",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--yt-dlp", default="/opt/wildenv/bin/yt-dlp")
    parser.add_argument("--deno", default="/usr/local/bin/deno")
    args = parser.parse_args()
    if args.min_free_gb < 0:
        parser.error("--min-free-gb must be non-negative")
    if (
        args.inter_video_sleep_min_s < 0
        or args.inter_video_sleep_max_s < args.inter_video_sleep_min_s
    ):
        parser.error("invalid inter-video sleep range")
    rows = load_queue(args.queue)
    if args.limit:
        rows = rows[: args.limit]
    policy = FetchPolicy(yt_dlp_path=args.yt_dlp, deno_path=args.deno)
    worker_preflight(policy)
    progress = args.progress or args.out / "fetch_fleet_progress.jsonl"
    counts = run_queue(
        rows, args.out, args.remote_root, progress, policy, args.min_free_gb,
        inter_video_sleep_min_s=args.inter_video_sleep_min_s,
        inter_video_sleep_max_s=args.inter_video_sleep_max_s,
        stop_on_source_block=not args.no_stop_on_source_block,
        delete_local_after_verified=args.delete_local_after_verified,
    )
    print(json.dumps(counts, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
