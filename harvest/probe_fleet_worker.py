"""Sequential Wild-corpus probe worker with immutable R2 checkpoints.

Each process owns one explicit JSONL queue and performs one network extraction
at a time.  Every attempt is published under a campaign/worker-specific R2
prefix and receives its completion marker only after every named object was
uploaded immutably and verified by SHA-256 readback.  Failed attempts remain
auditable without poisoning a later retry on a different IP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from harvest.fetch_wild import FetchPolicy, sha256_file
from harvest.scan_corpus import scan_one
from harvest.worker_wild import _copy_verified, worker_preflight


FORMAT_VERSION = "madeleine.wild-probe-attempt.v1"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def _remote_files(remote_dir: str) -> set[str]:
    completed = subprocess.run(
        ["rclone", "lsf", remote_dir, "--max-depth", "1", "--files-only"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return set()
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def _publish_attempt(directory: Path, report: dict[str, Any], remote_dir: str) -> dict[str, Any]:
    evidence = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.name not in {"objects.json", "probe_complete.json"}
    )
    object_rows = [
        {
            "name": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in evidence
    ]
    manifest = {
        "format_version": FORMAT_VERSION,
        "video_id": report["video_id"],
        "status": "error" if report.get("error") else "ok",
        "objects": object_rows,
    }
    manifest_path = directory / "objects.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    evidence.append(manifest_path)

    verified = [
        _copy_verified(path, f"{remote_dir}/{path.relative_to(directory).as_posix()}")
        for path in evidence
    ]
    completion = {
        "format_version": FORMAT_VERSION,
        "video_id": report["video_id"],
        "status": manifest["status"],
        "remote_dir": remote_dir,
        "verification": "every object SHA-256 hashed through rclone cat",
        "objects": verified,
        "total_bytes": sum(row["size_bytes"] for row in verified),
    }
    completion_path = directory / "probe_complete.json"
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    _copy_verified(completion_path, f"{remote_dir}/probe_complete.json")
    return completion


def run_queue(
    queue: Path,
    out_root: Path,
    remote_root: str,
    policy: FetchPolicy,
    seconds: float,
) -> dict[str, int]:
    if ":" not in remote_root:
        raise ValueError("remote_root must be an rclone remote path")
    rows = [json.loads(line) for line in queue.read_text().splitlines() if line.strip()]
    counts = {"queued": len(rows), "ok": 0, "error": 0, "skipped": 0}
    progress = out_root / "worker_progress.jsonl"
    out_root.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows, 1):
        video_id = str(row["video_id"])
        if not _SAFE_COMPONENT.fullmatch(video_id):
            raise ValueError(f"unsafe video_id: {video_id!r}")
        remote_dir = f"{remote_root.rstrip('/')}/{video_id}"
        directory = out_root / video_id
        marker = directory / "probe_complete.json"
        if marker.is_file() or "probe_complete.json" in _remote_files(remote_dir):
            counts["skipped"] += 1
            print(f"[{index}/{len(rows)}] {video_id}: skip-complete", flush=True)
            continue

        crops = directory / "crops"
        frames = directory / "frames"
        crops.mkdir(parents=True, exist_ok=True)
        frames.mkdir(parents=True, exist_ok=True)
        report = scan_one(row, seconds, crops, frames, policy)
        report.update({key: row.get(key) for key in (
            "video_id", "url", "category", "place", "duration_s", "source"
        )})
        report_path = directory / "probe.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        completion = _publish_attempt(directory, report, remote_dir)
        status = completion["status"]
        counts[status] += 1
        with progress.open("a") as handle:
            handle.write(json.dumps({
                "index": index,
                "video_id": video_id,
                "status": status,
                "remote_dir": remote_dir,
            }) + "\n")
        print(f"[{index}/{len(rows)}] {video_id}: {status}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--yt-dlp", default="/opt/wildenv/bin/yt-dlp")
    parser.add_argument("--deno", default="/usr/local/bin/deno")
    args = parser.parse_args()
    policy = FetchPolicy(yt_dlp_path=args.yt_dlp, deno_path=args.deno)
    worker_preflight(policy)
    counts = run_queue(args.queue, args.out, args.remote_root, policy, args.seconds)
    print(json.dumps(counts, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
