"""One-candidate wild fetch worker with verified R2 handoff.

The worker never inspects or configures rclone credentials.  It assumes the
host already has a named remote, fetches exactly one candidate at low rate,
uploads an explicit object manifest, reads every object back through rclone to
verify its SHA-256, and writes ``upload_complete.json`` last.  Consumers must
ignore remote prefixes without that completion object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np

from harvest.fetch_wild import FetchPolicy, fetch_candidate, load_candidate, sha256_file


UPLOAD_VERSION = "madeleine.wild-upload.v1"
PTS_PUBLICATION_VERSION = "madeleine.wild-pts-publication.v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def worker_preflight(policy: FetchPolicy) -> None:
    """Fail before a long scan when the worker-user runtime is half-installed."""

    for label, command in (("yt-dlp", policy.yt_dlp_path), ("deno", policy.deno_path)):
        resolved = command if "/" in command else shutil.which(command)
        if resolved is None or not Path(resolved).is_file() or not os.access(resolved, os.X_OK):
            raise RuntimeError(f"{label} is not executable by the worker user: {command}")
    # A root-owned 0750 site-package directory can import as an empty namespace
    # (np.__file__ is None) and fail only after ffprobe has scanned for minutes.
    if np.asarray([1]).item() != 1 or np.__file__ is None:
        raise RuntimeError("NumPy is not a functional worker-user installation")


def _remote_sha256(remote_path: str) -> tuple[str, int]:
    """Hash a remote object as streamed by rclone, without printing bytes."""

    digest = hashlib.sha256()
    count = 0
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            ["rclone", "cat", remote_path], stdout=subprocess.PIPE, stderr=errors
        )
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            count += len(chunk)
        return_code = process.wait()
        if return_code:
            errors.seek(0)
            detail = errors.read(1000).decode("utf-8", errors="replace")
            raise subprocess.CalledProcessError(return_code, process.args, stderr=detail)
    return digest.hexdigest(), count


def _object_rows(directory: Path, fetch: dict[str, Any]) -> list[dict[str, Any]]:
    names = {"fetch.json", fetch["source_file"]}
    names.update(path.name for path in directory.glob("*.info.json"))
    names.update(name for name in ("frame_pts.npy", "frame_pts.json")
                 if (directory / name).is_file())
    rows = []
    for name in sorted(names):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"required upload object is missing: {path}")
        rows.append({
            "name": name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def _copy_verified(local: Path, remote: str) -> dict[str, Any]:
    expected_hash = sha256_file(local)
    expected_size = local.stat().st_size
    _run([
        "rclone", "copyto", str(local), remote,
        "--immutable", "--size-only", "--transfers", "1",
        "--retries", "5", "--low-level-retries", "10",
    ])
    remote_hash, remote_size = _remote_sha256(remote)
    if remote_size != expected_size or remote_hash != expected_hash:
        raise ValueError(
            f"remote verification failed for {local.name}: "
            f"size {remote_size}/{expected_size} sha {remote_hash}/{expected_hash}"
        )
    return {
        "name": local.name,
        "size_bytes": expected_size,
        "sha256": expected_hash,
        "remote_path": remote,
        "verified": "sha256_readback",
    }


def publish_pts_evidence(
    directory: str | Path,
    fetch: dict[str, Any],
    remote_root: str,
) -> dict[str, Any]:
    """Publish derived PTS evidence without mutating a legacy raw prefix."""

    source_dir = Path(directory)
    video_id = str(fetch["video_id"])
    if not _SAFE_ID.fullmatch(video_id):
        raise ValueError(f"unsafe video_id for remote path: {video_id!r}")
    if ":" not in remote_root or not remote_root.strip():
        raise ValueError("PTS evidence remote root must be an rclone remote path")
    pts_manifest = source_dir / "frame_pts.json"
    pts_vector = source_dir / "frame_pts.npy"
    if not pts_manifest.is_file() or not pts_vector.is_file():
        raise FileNotFoundError("frame PTS evidence has not been generated locally")
    parsed = json.loads(pts_manifest.read_text())
    if parsed.get("source_sha256") != fetch.get("sha256"):
        raise ValueError("PTS evidence source hash differs from fetch evidence")

    remote_dir = f"{remote_root.rstrip('/')}/{video_id}"
    verified = [
        _copy_verified(path, f"{remote_dir}/{path.name}")
        for path in (pts_vector, pts_manifest)
    ]
    completion = {
        "format_version": PTS_PUBLICATION_VERSION,
        "video_id": video_id,
        "source_sha256": fetch["sha256"],
        "remote_dir": remote_dir,
        "verification": "every object SHA-256 hashed through rclone cat",
        "objects": verified,
        "total_bytes": sum(row["size_bytes"] for row in verified),
    }
    marker = source_dir / "pts_evidence_complete.json"
    marker.write_text(json.dumps(completion, indent=2) + "\n")
    _copy_verified(marker, f"{remote_dir}/{marker.name}")
    return completion


def upload_verified(
    directory: str | Path,
    fetch: dict[str, Any],
    remote_root: str,
) -> dict[str, Any]:
    """Upload exact files, read them back, then publish completion last."""

    source_dir = Path(directory)
    video_id = str(fetch["video_id"])
    if not _SAFE_ID.fullmatch(video_id):
        raise ValueError(f"unsafe video_id for remote path: {video_id!r}")
    if ":" not in remote_root or not remote_root.strip():
        raise ValueError("remote_root must be an rclone remote path")
    remote_dir = f"{remote_root.rstrip('/')}/{video_id}"
    object_manifest = {
        "format_version": UPLOAD_VERSION,
        "video_id": video_id,
        "objects": _object_rows(source_dir, fetch),
    }
    manifest_path = source_dir / "objects.json"
    manifest_path.write_text(json.dumps(object_manifest, indent=2) + "\n")
    object_manifest["objects"].append({
        "name": manifest_path.name,
        "size_bytes": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
    })

    verified = []
    for row in object_manifest["objects"]:
        local = source_dir / row["name"]
        remote = f"{remote_dir}/{row['name']}"
        verified.append(_copy_verified(local, remote))

    completion = {
        "format_version": UPLOAD_VERSION,
        "video_id": video_id,
        "remote_dir": remote_dir,
        "verification": "every object SHA-256 hashed through rclone cat",
        "objects": verified,
        "total_bytes": sum(row["size_bytes"] for row in verified),
    }
    completion_path = source_dir / "upload_complete.json"
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    completion_remote = f"{remote_dir}/{completion_path.name}"
    _run([
        "rclone", "copyto", str(completion_path), completion_remote,
        "--immutable", "--size-only", "--transfers", "1",
    ])
    marker_hash, marker_size = _remote_sha256(completion_remote)
    if marker_hash != sha256_file(completion_path) or marker_size != completion_path.stat().st_size:
        raise ValueError("remote completion-marker verification failed")
    return completion


def run_worker(
    candidate: dict[str, Any],
    out_root: str | Path,
    remote_root: str | None,
    *,
    pts_evidence_remote_root: str | None = None,
    explicit_start_s: float | None = None,
    explicit_end_s: float | None = None,
    policy: FetchPolicy = FetchPolicy(),
) -> dict[str, Any]:
    fetch = fetch_candidate(
        candidate, out_root, policy=policy, explicit_start_s=explicit_start_s,
        explicit_end_s=explicit_end_s,
    )
    directory = Path(out_root) / str(candidate["video_id"])
    if fetch.get("format_version") == "madeleine.wild-fetch.v1":
        if pts_evidence_remote_root is None:
            raise ValueError(
                "legacy raw completion is immutable; provide a separate "
                "pts_evidence_remote_root instead of republishing raw"
            )
        return publish_pts_evidence(directory, fetch, pts_evidence_remote_root)
    if remote_root is None:
        raise ValueError("remote_root is required for a v2 raw publication")
    return upload_verified(directory, fetch, remote_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True,
                        help="one candidate JSON object or one-row JSONL")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remote-root",
                        help=("for example "
                              "object-store:example-bucket/wild/v1/raw"))
    parser.add_argument("--pts-evidence-remote-root",
                        help="separate derived prefix required for legacy v1 canaries")
    parser.add_argument("--start-s", type=float)
    parser.add_argument("--end-s", type=float)
    parser.add_argument("--yt-dlp", default="/opt/wildenv/bin/yt-dlp")
    parser.add_argument("--deno", default="/usr/local/bin/deno")
    args = parser.parse_args()
    try:
        candidate = load_candidate(args.candidate)
    except ValueError as exc:
        parser.error(str(exc))
    policy = FetchPolicy(yt_dlp_path=args.yt_dlp, deno_path=args.deno)
    worker_preflight(policy)
    result = run_worker(candidate, args.out, args.remote_root,
                        explicit_start_s=args.start_s,
                        explicit_end_s=args.end_s,
                        pts_evidence_remote_root=args.pts_evidence_remote_root,
                        policy=policy)
    print(json.dumps({
        "video_id": result["video_id"],
        "remote_dir": result["remote_dir"],
        "objects": len(result["objects"]),
        "total_bytes": result["total_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
