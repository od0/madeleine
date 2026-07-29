"""Build a source-bound sparse layout survey from one completed wild video.

This is deliberately an AI-review staging artifact.  It extracts exact decoded
frame indices across the full source, assembles a contact sheet, binds every
image to the immutable fetch/PTS evidence, and can publish the result with
SHA-256 readback.  It never claims layout approval or training admission.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import numpy as np

from harvest.worker_wild import _copy_verified


FORMAT_VERSION = "madeleine.wild-layout-survey.v1"
PUBLICATION_VERSION = "madeleine.wild-layout-survey-publication.v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def declared_artifact_path(
    directory: Path, declared: dict[str, Any]
) -> Path:
    """Resolve and validate one manifest-relative artifact before publication."""

    relative = Path(str(declared.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("declared artifact path must stay inside its directory")
    root = directory.resolve()
    path = (directory / relative).resolve()
    if path.parent != root and root not in path.parents:
        raise ValueError("declared artifact resolves outside its directory")
    if not path.is_file():
        raise FileNotFoundError(f"declared artifact is missing: {relative}")
    expected_size = declared.get("size_bytes")
    expected_hash = str(declared.get("sha256", ""))
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ValueError(f"declared size differs from current bytes: {relative}")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"declared hash differs from current bytes: {relative}")
    return path


def sample_indices(frame_count: int, sample_count: int) -> list[int]:
    """Return unique full-source frame indices spanning 2% through 98%."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    count = min(frame_count, sample_count)
    if count == frame_count:
        return list(range(frame_count))
    indices = np.rint(
        np.linspace(0.02, 0.98, num=count, dtype=np.float64)
        * (frame_count - 1)
    ).astype(np.int64)
    unique = sorted({int(value) for value in indices})
    if len(unique) != count:
        # This can happen only for very short sources.  Preserve the requested
        # count deterministically without pretending duplicated frames differ.
        unique = np.linspace(0, frame_count - 1, num=count, dtype=np.int64).tolist()
        unique = sorted({int(value) for value in unique})
    if len(unique) != count:
        raise ValueError("could not choose the requested number of unique frames")
    return unique


def validate_pts_order(pts: np.ndarray) -> int:
    """Validate decoded-frame order and return duplicate timestamp intervals.

    Some real containers assign the same presentation timestamp to adjacent
    decoded frames.  Exact frame indices remain usable for layout evidence;
    backward timestamps do not.  Duplicate timestamps stay explicit so a
    later timing/admission gate can handle them without discarding the source
    before machine triage.
    """

    if pts.ndim != 1 or pts.size == 0:
        raise ValueError("PTS vector must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(pts)):
        raise ValueError("PTS vector must be finite")
    intervals = np.diff(pts)
    if np.any(intervals < 0):
        raise ValueError("PTS vector must be nondecreasing")
    return int(np.count_nonzero(intervals == 0))


def exact_extract_command(
    source: Path,
    destination_pattern: Path,
    indices: list[int],
    hwaccel: str,
) -> list[str]:
    if not indices:
        raise ValueError("at least one frame index is required")
    selection = "+".join(f"eq(n\\,{index})" for index in indices)
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    filters = [f"select={selection}"]
    if hwaccel != "none":
        command.extend(["-hwaccel", hwaccel, "-hwaccel_output_format", hwaccel])
        filters.extend(["hwdownload", "format=nv12"])
    command.extend([
        "-i", str(source),
        "-vf", ",".join(filters),
        "-frames:v", str(len(indices)),
        "-vsync", "0",
        str(destination_pattern),
    ])
    return command


def contact_sheet_command(
    source_pattern: Path,
    destination: Path,
    sample_count: int,
    columns: int = 4,
    tile_width: int = 480,
) -> list[str]:
    rows = math.ceil(sample_count / columns)
    filter_graph = (
        f"scale={tile_width}:-1,"
        "drawtext=fontcolor=white:fontsize=20:box=1:boxcolor=black@0.70:"
        "text='sample %{n}':x=8:y=8,"
        f"tile={columns}x{rows}:padding=4:margin=4:color=black"
    )
    return [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", "1", "-i", str(source_pattern),
        "-vf", filter_graph, "-frames:v", "1", str(destination),
    ]


def _load_bound_source(source_dir: Path) -> tuple[dict[str, Any], np.ndarray, Path]:
    fetch_path = source_dir / "fetch.json"
    pts_path = source_dir / "frame_pts.npy"
    pts_manifest_path = source_dir / "frame_pts.json"
    completion_path = source_dir / "upload_complete.json"
    for path in (fetch_path, pts_path, pts_manifest_path, completion_path):
        if not path.is_file():
            raise FileNotFoundError(f"required completed-source artifact missing: {path}")

    fetch = json.loads(fetch_path.read_text())
    pts_manifest = json.loads(pts_manifest_path.read_text())
    completion = json.loads(completion_path.read_text())
    video_id = str(fetch.get("video_id", ""))
    if not _SAFE_ID.fullmatch(video_id) or source_dir.name != video_id:
        raise ValueError("source directory and fetch video_id do not match safely")
    if completion.get("video_id") != video_id:
        raise ValueError("raw completion marker belongs to another video")
    source = source_dir / str(fetch.get("source_file", ""))
    if not source.is_file():
        raise FileNotFoundError("fetch.json source file is missing")
    if sha256_file(source) != fetch.get("sha256"):
        raise ValueError("source video hash differs from immutable fetch evidence")
    if pts_manifest.get("source_sha256") != fetch.get("sha256"):
        raise ValueError("PTS evidence belongs to another source video")
    if sha256_file(pts_path) != pts_manifest.get("sha256"):
        raise ValueError("PTS vector hash differs from its manifest")
    pts = np.load(pts_path, allow_pickle=False)
    if pts.ndim != 1 or pts.size != int(pts_manifest.get("frames", -1)):
        raise ValueError("PTS vector shape/count differs from its manifest")
    validate_pts_order(pts)

    completion_objects = {
        str(row.get("name")): row for row in completion.get("objects", [])
    }
    for name, expected_hash in (
        (fetch_path.name, sha256_file(fetch_path)),
        (pts_path.name, sha256_file(pts_path)),
        (pts_manifest_path.name, sha256_file(pts_manifest_path)),
        (source.name, str(fetch["sha256"])),
    ):
        if completion_objects.get(name, {}).get("sha256") != expected_hash:
            raise ValueError(f"raw completion marker does not bind {name}")
    return fetch, pts.astype(np.float64, copy=False), source


def build_survey(
    source_dir: Path,
    out_root: Path,
    *,
    sample_count: int = 16,
    hwaccel: str = "cuda",
) -> Path:
    fetch, pts, source = _load_bound_source(source_dir)
    video_id = str(fetch["video_id"])
    destination = out_root / video_id
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"survey output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    staging = destination / ".frames"
    staging.mkdir()

    indices = sample_indices(int(pts.size), sample_count)
    pattern = staging / "sample-%03d.png"
    subprocess.run(
        exact_extract_command(source, pattern, indices, hwaccel), check=True
    )
    extracted = sorted(staging.glob("sample-*.png"))
    if len(extracted) != len(indices):
        raise ValueError(
            f"ffmpeg produced {len(extracted)} frames for {len(indices)} indices"
        )
    contact_path = destination / "contact-sheet.png"
    subprocess.run(
        contact_sheet_command(pattern, contact_path, len(indices)), check=True
    )

    frame_rows = []
    for order, (temporary, index) in enumerate(zip(extracted, indices, strict=True)):
        pts_us = int(round(float(pts[index]) * 1_000_000))
        final = destination / f"sample-{order:02d}-frame-{index:09d}-pts-{pts_us}.png"
        temporary.rename(final)
        frame_rows.append({
            "sample_order": order,
            "exact_frame_index": index,
            "exact_pts_s": float(pts[index]),
            "path": final.name,
            "size_bytes": final.stat().st_size,
            "sha256": sha256_file(final),
        })
    staging.rmdir()

    manifest = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video_id": video_id,
        "purpose": "AI-only full-source layout and activity-stability nomination",
        "human_reviewed": False,
        "training_admitted": False,
        "source": {
            "path": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": fetch["sha256"],
            "fetch_json_sha256": sha256_file(source_dir / "fetch.json"),
            "raw_completion_sha256": sha256_file(source_dir / "upload_complete.json"),
        },
        "pts": {
            "path": "frame_pts.npy",
            "sha256": sha256_file(source_dir / "frame_pts.npy"),
            "frames": int(pts.size),
            "first_s": float(pts[0]),
            "last_s": float(pts[-1]),
            "duplicate_timestamp_intervals": validate_pts_order(pts),
        },
        "sampling": {
            "scheme": "uniform decoded-frame quantiles from 0.02 through 0.98",
            "requested_count": sample_count,
            "actual_count": len(indices),
            "ffmpeg_hwaccel": hwaccel,
            "exact_index_filter": "ffmpeg select=eq(n,index), one ordered full decode",
        },
        "frames": frame_rows,
        "contact_sheet": {
            "path": contact_path.name,
            "size_bytes": contact_path.stat().st_size,
            "sha256": sha256_file(contact_path),
            "sample_order": "row-major; labels are zero-based sample_order",
        },
    }
    manifest_path = destination / "survey.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def publish_survey(manifest_path: Path, remote_root: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    video_id = str(manifest.get("video_id", ""))
    if not _SAFE_ID.fullmatch(video_id):
        raise ValueError("survey manifest contains an unsafe video_id")
    if ":" not in remote_root:
        raise ValueError("remote_root must be an rclone remote path")
    if (
        manifest.get("human_reviewed") is not False
        or manifest.get("training_admitted") is not False
    ):
        raise ValueError("AI survey publication cannot claim review or admission")
    directory = manifest_path.parent
    remote_dir = f"{remote_root.rstrip('/')}/{video_id}"
    declared_rows = [*manifest["frames"], manifest["contact_sheet"]]
    declared_paths = [
        declared_artifact_path(directory, row) for row in declared_rows
    ]
    if len(declared_paths) != len(set(declared_paths)):
        raise ValueError("survey manifest declares the same artifact more than once")
    paths = declared_paths + [manifest_path.resolve()]
    verified = []
    for index, path in enumerate(paths):
        relative = path.relative_to(directory.resolve()).as_posix()
        copied = _copy_verified(path, f"{remote_dir}/{relative}")
        if index < len(declared_rows):
            declared = declared_rows[index]
            if (
                copied.get("sha256") != declared.get("sha256")
                or int(copied.get("size_bytes", -1))
                != int(declared.get("size_bytes", -2))
            ):
                raise ValueError(
                    f"published bytes differ from survey manifest: {relative}"
                )
        verified.append(copied)
    completion = {
        "format_version": PUBLICATION_VERSION,
        "video_id": video_id,
        "source_sha256": manifest["source"]["sha256"],
        "survey_sha256": sha256_file(manifest_path),
        "remote_dir": remote_dir,
        "verification": "every object SHA-256 hashed through rclone cat",
        "objects": verified,
        "total_bytes": sum(int(row["size_bytes"]) for row in verified),
        "human_reviewed": False,
        "training_admitted": False,
    }
    completion_path = directory / "survey_complete.json"
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    _copy_verified(completion_path, f"{remote_dir}/{completion_path.name}")
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--hwaccel", default="cuda", choices=("cuda", "none"))
    parser.add_argument("--remote-root")
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required")
    existing_manifest = None
    fetch_path = args.source_dir / "fetch.json"
    if fetch_path.is_file():
        existing_video_id = str(json.loads(fetch_path.read_text()).get("video_id", ""))
        if _SAFE_ID.fullmatch(existing_video_id):
            destination = args.out_root / existing_video_id
            candidate = destination / "survey.json"
            if candidate.is_file():
                fetch, pts, _ = _load_bound_source(args.source_dir)
                existing = json.loads(candidate.read_text())
                if (
                    existing.get("video_id") != existing_video_id
                    or existing.get("source", {}).get("sha256") != fetch.get("sha256")
                    or existing.get("pts", {}).get("sha256")
                    != sha256_file(args.source_dir / "frame_pts.npy")
                    or int(existing.get("pts", {}).get("frames", -1)) != int(pts.size)
                ):
                    raise ValueError("existing survey does not bind the completed source")
                existing_manifest = candidate
            elif destination.exists() and any(destination.iterdir()):
                failed_root = args.out_root / ".failed"
                failed_root.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                destination.rename(failed_root / f"{existing_video_id}-{stamp}")
    manifest = existing_manifest or build_survey(
        args.source_dir,
        args.out_root,
        sample_count=args.sample_count,
        hwaccel=args.hwaccel,
    )
    result: dict[str, Any] = {
        "video_id": json.loads(manifest.read_text())["video_id"],
        "manifest": str(manifest),
    }
    if args.remote_root:
        result["publication"] = publish_survey(manifest, args.remote_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
