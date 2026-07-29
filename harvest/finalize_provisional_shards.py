"""Verify concurrent provisional shard jobs and publish one immutable manifest.

Each video builds in its own directory, so builders never race on a shared
manifest.  This finalizer takes an explicit video-id set, verifies every input
binding and NPZ byte hash, and creates the aggregate with an atomic
no-overwrite operation.  It never turns provisional data into admitted data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import zipfile

import numpy as np

from harvest.build_wild import PROVISIONAL_BUILD_VERSION
from harvest.fetch_wild import sha256_file


AGGREGATE_VERSION = "madeleine.wild-provisional-corpus.v1"
EXPECTED_ARRAY_NAMES = {
    "frames.npy",
    "keys.npy",
    "engine_frame_idx.npy",
    "pts_s.npy",
    "input_active.npy",
    "session_id.npy",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_relative_basename(value: Any, field: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{field} must be a string")
    path = Path(value)
    _require(
        not path.is_absolute() and path.name == value and value not in {".", ".."},
        f"{field} must be a relative basename",
    )
    return value


def _require_video_id(value: Any) -> str:
    """Keep an explicit video inventory from escaping configured roots."""

    video_id = _require_relative_basename(value, "video ID")
    _require(
        re.fullmatch(r"[A-Za-z0-9_-]+", video_id) is not None,
        "video ID contains unsupported characters",
    )
    return video_id


def _require_sha256(value: Any, field: str) -> str:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{field} SHA-256 is missing or invalid",
    )
    return value


def _npy_member_header(
    archive: zipfile.ZipFile,
    member_name: str,
) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    """Read an NPY header without inflating its array payload.

    This helper is intentionally local. The finalizer must be able to audit
    shards produced by the frozen a7bbb10 builder, which predates the builder's
    private resume helper.
    """

    with archive.open(member_name) as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(member)
        if version == (2, 0):
            return np.lib.format.read_array_header_2_0(member)
        raise ValueError(f"unsupported npy header version {version} in {member_name}")


def _verify_bound_file(path: Path, expected_sha256: Any, field: str) -> int:
    _require(path.is_file() and not path.is_symlink(), f"missing regular {field}: {path}")
    expected = _require_sha256(expected_sha256, field)
    actual = sha256_file(path)
    _require(actual == expected, f"{field} SHA-256 mismatch: {path}")
    return path.stat().st_size


def _verify_npz(
    path: Path,
    part: dict[str, Any],
    *,
    expected_frame_size: int,
) -> dict[str, int | float]:
    expected_frames = int(part.get("frames", 0))
    _require(expected_frames > 0, f"invalid frame count for {path.name}")
    _verify_bound_file(path, part.get("sha256"), "part")
    try:
        with zipfile.ZipFile(path) as archive:
            _require(
                set(archive.namelist()) == EXPECTED_ARRAY_NAMES,
                f"unexpected NPZ members: {path}",
            )
            shape, fortran, dtype = _npy_member_header(archive, "frames.npy")
            _require(
                shape
                == (expected_frames, expected_frame_size, expected_frame_size, 3)
                and not fortran
                and dtype == np.dtype(np.uint8),
                f"frame array contract mismatch: {path}",
            )
        with np.load(path, allow_pickle=False) as stored:
            keys = stored["keys"]
            indices = stored["engine_frame_idx"]
            pts = stored["pts_s"]
            active = stored["input_active"]
            _require(
                keys.shape == (expected_frames, 7) and keys.dtype == np.uint8,
                f"key array contract mismatch: {path}",
            )
            _require(
                np.all((keys == 0) | (keys == 1)),
                f"key values are not binary: {path}",
            )
            _require(
                indices.shape == (expected_frames,) and indices.dtype == np.int64,
                f"frame-index contract mismatch: {path}",
            )
            _require(
                int(indices[0]) >= 0 and np.all(np.diff(indices) == 1),
                f"frame indices are not dense and ordered: {path}",
            )
            _require(
                pts.shape == (expected_frames,)
                and pts.dtype == np.float64
                and np.all(np.isfinite(pts))
                and np.all(np.diff(pts) > 0),
                f"PTS contract mismatch: {path}",
            )
            _require(
                active.shape == (expected_frames,)
                and active.dtype == np.uint8
                and np.all(active == 1),
                f"activity contract mismatch: {path}",
            )
            _require(
                str(stored["session_id"]) == str(part.get("session_id")),
                f"session ID mismatch: {path}",
            )
            source_range = part.get("source_frame_range")
            _require(
                source_range == [int(indices[0]), int(indices[-1]) + 1]
                and int(indices[-1]) - int(indices[0]) + 1 == expected_frames,
                f"source frame range mismatch: {path}",
            )
            pts_range = part.get("pts_range_s")
            _require(
                isinstance(pts_range, list)
                and len(pts_range) == 2
                and float(pts_range[0]) == float(pts[0])
                and float(pts_range[1]) == float(pts[-1]),
                f"PTS range mismatch: {path}",
            )
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"invalid NPZ {path}: {error}") from error
    return {
        "npz_bytes": path.stat().st_size,
        "first_frame": int(indices[0]),
        "last_frame": int(indices[-1]),
        "first_pts_s": float(pts[0]),
        "last_pts_s": float(pts[-1]),
    }


def _verify_video(
    *,
    video_id: str,
    jobs_root: Path,
    decode_root: Path,
    layout_root: Path,
    boundary_store: Path,
    builder_sha256: str,
    verify_source_bytes: bool,
    expected_frame_size: int,
    seen_sessions: set[str],
) -> dict[str, Any]:
    part_dir = jobs_root / video_id / "parts"
    report_path = part_dir / "wild_provisional_build_report.json"
    _require(
        report_path.is_file() and not report_path.is_symlink(),
        f"missing regular completed build report for {video_id}",
    )
    report = json.loads(report_path.read_text())
    _require(report.get("format_version") == PROVISIONAL_BUILD_VERSION,
             f"wrong report format for {video_id}")
    _require(report.get("video_id") == video_id, f"report video ID mismatch: {video_id}")
    _require(report.get("label_kind") == "wild_overlay_provisional",
             f"wrong label kind for {video_id}")
    _require(report.get("admission_tier") == "provisional_not_train_ready",
             f"wrong admission tier for {video_id}")
    _require(report.get("timing_authority") == "presentation_timestamp",
             f"wrong timing authority for {video_id}")
    _require(report.get("train_ready_frames") == 0, f"nonzero train-ready frames: {video_id}")
    _require(float(report.get("train_ready_hours", -1)) == 0.0,
             f"nonzero train-ready hours: {video_id}")
    implementation = report.get("implementation", {})
    _require(
        implementation == {
            "module": "harvest/build_wild.py",
            "sha256": builder_sha256,
        },
        f"builder implementation mismatch for {video_id}",
    )

    inputs = report.get("inputs", {})
    decode_binding = inputs.get("decode_report", {})
    decode_name = _require_relative_basename(
        decode_binding.get("path"), "decode report path"
    )
    decode_path = decode_root / video_id / decode_name
    _verify_bound_file(decode_path, decode_binding.get("sha256"), "decode report")
    decode = json.loads(decode_path.read_text())
    _require(decode.get("format_version") == "madeleine.wild-decode.v1",
             f"wrong decode format for {video_id}")
    _require(decode.get("video_id") == video_id, f"decode video ID mismatch: {video_id}")
    _require(decode.get("admitted") is False, f"provisional decode is admitted: {video_id}")
    _require(int(report.get("decoded_frames", -1)) == int(decode.get("decoded_frames", -2)),
             f"decoded frame total mismatch: {video_id}")
    _require(float(report.get("decoded_hours", -1)) == float(decode.get("decoded_hours", -2)),
             f"decoded hour total mismatch: {video_id}")
    _require(
        report.get("unresolved_admission_reasons") == decode.get("rejection_reasons"),
        f"unresolved admission reasons mismatch: {video_id}",
    )

    labels_binding = inputs.get("labels", {})
    labels_name = _require_relative_basename(labels_binding.get("path"), "labels path")
    labels_path = decode_path.parent / labels_name
    _verify_bound_file(labels_path, labels_binding.get("sha256"), "labels")
    _require(labels_binding.get("sha256") == decode.get("labels_sha256"),
             f"decode/labels binding mismatch: {video_id}")

    layout_binding = inputs.get("layout", {})
    layout_name = _require_relative_basename(layout_binding.get("path"), "layout path")
    layout_path = layout_root / layout_name
    _verify_bound_file(layout_path, layout_binding.get("sha256"), "layout")
    _require(layout_binding.get("sha256") == decode.get("layout", {}).get("sha256"),
             f"decode/layout binding mismatch: {video_id}")

    boundaries = decode.get("boundaries", {})
    boundary_sha256 = boundaries.get("sha256")
    _require(inputs.get("boundaries_sha256") == boundary_sha256,
             f"decode/boundaries binding mismatch: {video_id}")
    boundary_sha256 = _require_sha256(boundary_sha256, "boundaries")
    # Decode reports preserve the worker-local original path, which is not a
    # portable locator. Relocation is explicit and content-addressed instead:
    # the handoff store must contain exactly the bytes bound by the decode.
    boundary_path = boundary_store / f"{boundary_sha256}.json"
    _verify_bound_file(boundary_path, boundary_sha256, "boundaries")

    source = decode.get("source_video", {})
    source_path = Path(str(source.get("path", "")))
    _require(source_path.is_absolute(), f"source path is not absolute: {video_id}")
    source_sha256 = _require_sha256(source.get("sha256"), "source video")
    _require(inputs.get("source_video_sha256") == source_sha256,
             f"decode/source binding mismatch: {video_id}")
    _require(source_path.is_file() and not source_path.is_symlink(),
             f"missing regular source video: {source_path}")
    if verify_source_bytes:
        _verify_bound_file(source_path, source_sha256, "source video")

    fps = float(report.get("effective_grid_hz", 0.0))
    _require(np.isfinite(fps) and fps > 0, f"invalid grid for {video_id}")
    decode_fps = float(decode.get("timing", {}).get("pts", {}).get("effective_fps", 0.0))
    _require(fps == decode_fps, f"decode/build grid mismatch: {video_id}")
    parts = report.get("parts")
    _require(isinstance(parts, list) and parts, f"empty part inventory: {video_id}")
    declared_npz_names = {
        _require_relative_basename(part.get("npz"), "part path")
        for part in parts
        if isinstance(part, dict)
    }
    _require(
        len(declared_npz_names) == len(parts),
        f"duplicate or invalid part inventory: {video_id}",
    )
    actual_npz_names = {path.name for path in part_dir.glob("*.npz") if path.is_file()}
    _require(
        actual_npz_names == declared_npz_names,
        f"unreported or missing NPZ files: {video_id}",
    )
    inventory: list[dict[str, Any]] = []
    previous_end = -1
    previous_pts = -np.inf
    total_frames = 0
    for part_number, part in enumerate(parts):
        _require(isinstance(part, dict), f"invalid part row for {video_id}")
        expected_session = f"wild_provisional_{video_id}__r{part_number:03d}"
        _require(part.get("session_id") == expected_session,
                 f"noncanonical session ID for {video_id} part {part_number}")
        _require(expected_session not in seen_sessions,
                 f"duplicate session ID: {expected_session}")
        seen_sessions.add(expected_session)
        npz_name = _require_relative_basename(part.get("npz"), "part path")
        _require(
            npz_name == f"{expected_session}.npz",
            f"noncanonical part filename for {video_id} part {part_number}",
        )
        npz_path = part_dir / npz_name
        source_range = part.get("source_frame_range")
        _require(
            isinstance(source_range, list)
            and len(source_range) == 2
            and int(source_range[0]) >= previous_end,
            f"overlapping or unordered source ranges: {video_id} part {part_number}",
        )
        _require(
            int(source_range[1]) > int(source_range[0]),
            f"empty or reversed source range: {video_id} part {part_number}",
        )
        verified = _verify_npz(
            npz_path,
            part,
            expected_frame_size=expected_frame_size,
        )
        _require(
            float(verified["first_pts_s"]) > previous_pts,
            f"overlapping or unordered PTS ranges: {video_id} part {part_number}",
        )
        previous_end = int(source_range[1])
        previous_pts = float(verified["last_pts_s"])
        total_frames += int(part["frames"])
        inventory.append({
            **part,
            "path": npz_path.relative_to(jobs_root).as_posix(),
            "npz_bytes": int(verified["npz_bytes"]),
        })

    declared_frames = int(report.get("provisional_trainable_frames", -1))
    _require(total_frames == declared_frames,
             f"part/frame total mismatch for {video_id}")
    computed_hours = total_frames / fps / 3600.0
    _require(
        abs(computed_hours - float(report.get("provisional_trainable_hours", -1))) < 1e-12,
        f"part/hour total mismatch for {video_id}",
    )
    return {
        "video_id": video_id,
        "build_report": {
            "path": report_path.relative_to(jobs_root).as_posix(),
            "sha256": sha256_file(report_path),
        },
        "decode_report_sha256": decode_binding["sha256"],
        "boundaries_sha256": boundary_sha256,
        "source_video_sha256": source_sha256,
        "effective_grid_hz": fps,
        "decoded_frames": int(report["decoded_frames"]),
        "decoded_hours": float(report["decoded_hours"]),
        "train_ready_frames": 0,
        "train_ready_hours": 0.0,
        "provisional_trainable_frames": total_frames,
        "provisional_trainable_hours": computed_hours,
        "part_count": len(inventory),
        "parts": inventory,
    }


def _publish_no_overwrite(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link is atomic and fails if another finalizer won the race.
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def finalize_provisional_corpus(
    *,
    jobs_root: str | Path,
    decode_root: str | Path,
    layout_root: str | Path,
    boundary_store: str | Path,
    builder_file: str | Path,
    video_ids: list[str],
    output: str | Path,
    verify_source_bytes: bool = True,
    expected_frame_size: int = 128,
) -> dict[str, Any]:
    _require(bool(video_ids), "an explicit nonempty video-id set is required")
    checked_video_ids = [_require_video_id(video_id) for video_id in video_ids]
    _require(
        len(checked_video_ids) == len(set(checked_video_ids)),
        "duplicate requested video IDs",
    )
    _require(expected_frame_size > 0, "expected frame size must be positive")
    jobs = Path(jobs_root).resolve()
    decodes = Path(decode_root).resolve()
    layouts = Path(layout_root).resolve()
    boundaries = Path(boundary_store).resolve()
    builder = Path(builder_file).resolve()
    builder_sha256 = sha256_file(builder)
    seen_sessions: set[str] = set()
    videos = [
        _verify_video(
            video_id=video_id,
            jobs_root=jobs,
            decode_root=decodes,
            layout_root=layouts,
            boundary_store=boundaries,
            builder_sha256=builder_sha256,
            verify_source_bytes=verify_source_bytes,
            expected_frame_size=expected_frame_size,
            seen_sessions=seen_sessions,
        )
        for video_id in sorted(checked_video_ids)
    ]
    manifest = {
        "format_version": AGGREGATE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "admission_tier": "provisional_not_train_ready",
        "timing_authority": "presentation_timestamp",
        "builder": {
            "path": str(builder),
            "sha256": builder_sha256,
        },
        "verification": {
            "explicit_video_set": sorted(checked_video_ids),
            "expected_frame_shape": [expected_frame_size, expected_frame_size, 3],
            "source_files_rehashed": verify_source_bytes,
            "part_archives_rehashed": True,
            "part_array_contracts_verified": True,
            "output_policy": "atomic_no_overwrite",
        },
        "video_count": len(videos),
        "session_count": len(seen_sessions),
        "decoded_frames": sum(row["decoded_frames"] for row in videos),
        "decoded_hours": sum(row["decoded_hours"] for row in videos),
        "train_ready_frames": 0,
        "train_ready_hours": 0.0,
        "provisional_trainable_frames": sum(
            row["provisional_trainable_frames"] for row in videos
        ),
        "provisional_trainable_hours": sum(
            row["provisional_trainable_hours"] for row in videos
        ),
        "videos": videos,
        "warning": (
            "Diagnostic/noisy-supervision corpus only. Human layout, timing, "
            "and gameplay-boundary admission gates remain unresolved."
        ),
    }
    _publish_no_overwrite(Path(output), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--decode-root", type=Path, required=True)
    parser.add_argument("--layout-root", type=Path, required=True)
    parser.add_argument(
        "--boundary-store",
        type=Path,
        required=True,
        help="directory of content-addressed <sha256>.json boundary files",
    )
    parser.add_argument("--builder-file", type=Path, required=True)
    parser.add_argument("--video-id", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--frame-size",
        type=int,
        default=128,
        help="required square frame size in every shard (default: 128)",
    )
    parser.add_argument(
        "--skip-source-byte-hash",
        action="store_true",
        help="trust decode/report source hash binding without rereading source bytes",
    )
    args = parser.parse_args()
    manifest = finalize_provisional_corpus(
        jobs_root=args.jobs_root,
        decode_root=args.decode_root,
        layout_root=args.layout_root,
        boundary_store=args.boundary_store,
        builder_file=args.builder_file,
        video_ids=args.video_id,
        output=args.out,
        verify_source_bytes=not args.skip_source_byte_hash,
        expected_frame_size=args.frame_size,
    )
    print(json.dumps({
        "video_count": manifest["video_count"],
        "session_count": manifest["session_count"],
        "train_ready_hours": manifest["train_ready_hours"],
        "provisional_trainable_hours": manifest["provisional_trainable_hours"],
        "output": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
