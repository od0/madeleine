"""Stream a reviewed timer ROI into source-bound activity evidence.

This bridge reads a dense, reviewed PTS interval from one fetched video while
retaining only scalar traces and the previous ROI.  It verifies the immutable
source and its separately hashed PTS vector, writes a compact trace, and asks
``timer_activity`` for review-required range suggestions.  It never creates a
``WildBoundaries`` artifact and never admits footage automatically.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

import cv2
import numpy as np

from harvest.fetch_wild import PTS_SIDECAR_VERSION, sha256_file, summarize_pts
from harvest.timer_activity import (
    Rect,
    REVIEWER_KINDS,
    TimerActivityPolicy,
    TimerReviewContext,
    segment_timer_activity,
    write_timer_activity_diagnostics,
)
from harvest.wild_layout import rect_to_pixels


TRACE_VERSION = "madeleine.wild-timer-trace.v1"
MANIFEST_VERSION = "madeleine.wild-timer-trace-manifest.v1"
TRACE_FILE = "timer_trace.npz"
PROPOSAL_FILE = "timer_activity_proposal.json"
MANIFEST_FILE = "timer_trace_manifest.json"
REQUIRED_EVIDENCE_NAMES = frozenset({"timer_roi", "wall_clock_bounds"})
DECODE_BACKENDS = ("opencv", "ffmpeg")
FFMPEG_BACKEND_VERSION = "madeleine.ffmpeg-timer-raw-pipe.v2"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _safe_child(directory: Path, name: Any, field: str) -> Path:
    text = str(name or "").strip()
    if not text or Path(text).name != text:
        raise ValueError(f"{field} must be a plain file name")
    return directory / text


def _review_evidence(evidence_refs: Mapping[str, str]) -> dict[str, str]:
    evidence = {
        str(name).strip(): str(reference).strip()
        for name, reference in evidence_refs.items()
    }
    if any(not name or not reference for name, reference in evidence.items()):
        raise ValueError("evidence reference names and values must be non-empty")
    missing = REQUIRED_EVIDENCE_NAMES - evidence.keys()
    if missing:
        raise ValueError(
            "named review evidence is missing: " + ", ".join(sorted(missing))
        )
    return dict(sorted(evidence.items()))


def _nominal_loadless_duration(fetch: Mapping[str, Any]) -> float | None:
    """Read optional leaderboard duration without using it as a wall bound."""

    raw_values: list[tuple[str, Any]] = []
    candidate = fetch.get("candidate")
    if isinstance(candidate, Mapping) and candidate.get("duration_s") is not None:
        raw_values.append(("candidate.duration_s", candidate.get("duration_s")))
    run_window = fetch.get("run_window")
    if (
        isinstance(run_window, Mapping)
        and run_window.get("nominal_loadless_duration_s") is not None
    ):
        raw_values.append((
            "run_window.nominal_loadless_duration_s",
            run_window.get("nominal_loadless_duration_s"),
        ))
    if not raw_values:
        return None
    parsed: list[tuple[str, float]] = []
    for field, raw in raw_values:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"fetch report {field} must be numeric")
        value = float(raw)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"fetch report {field} must be finite and positive")
        parsed.append((field, value))
    first_field, first_value = parsed[0]
    for field, value in parsed[1:]:
        if not np.isclose(value, first_value, rtol=1e-9, atol=1e-6):
            raise ValueError(
                f"fetch report {first_field} and {field} disagree"
            )
    return first_value


def _load_verified_pts(
    fetch_path: Path,
    fetch: dict[str, Any],
    source_sha256: str,
    evidence_dir: str | Path | None,
    *,
    verify_hash: bool,
) -> tuple[np.ndarray, dict[str, Any], Path, Path]:
    directory = Path(evidence_dir) if evidence_dir is not None else fetch_path.parent
    declared = fetch.get("pts_sidecar")
    manifest_name = (
        declared.get("manifest", "frame_pts.json")
        if isinstance(declared, dict)
        else "frame_pts.json"
    )
    manifest_path = _safe_child(directory, manifest_name, "PTS manifest path")
    if not manifest_path.is_file():
        raise ValueError(f"hashed PTS manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read PTS manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("PTS manifest must contain one JSON object")
    if manifest.get("format_version") != PTS_SIDECAR_VERSION:
        raise ValueError("unsupported frame PTS sidecar version")
    if manifest.get("source_sha256") != source_sha256:
        raise ValueError("frame PTS sidecar belongs to a different source video")
    vector_path = _safe_child(
        directory, manifest.get("path", "frame_pts.npy"), "PTS vector path"
    )
    if not vector_path.is_file():
        raise ValueError(f"hashed PTS vector is missing: {vector_path}")
    actual_vector_hash = sha256_file(vector_path)
    if verify_hash and manifest.get("sha256") != actual_vector_hash:
        raise ValueError("frame PTS sidecar hash mismatch")
    try:
        pts = np.load(vector_path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load PTS vector: {vector_path}") from exc
    if pts.ndim != 1 or not np.issubdtype(pts.dtype, np.number):
        raise ValueError("frame PTS sidecar must be a one-dimensional numeric vector")
    if int(manifest.get("frames", -1)) != pts.size:
        raise ValueError("frame PTS sidecar shape/count mismatch")
    summary = summarize_pts(pts)
    if summary["nonmonotonic_intervals"]:
        raise ValueError("frame PTS sidecar is not strictly increasing")
    if isinstance(declared, dict):
        if verify_hash and declared.get("sha256") != actual_vector_hash:
            raise ValueError("PTS evidence hash differs from fetch report")
        if int(declared.get("frames", -1)) != pts.size:
            raise ValueError("PTS evidence length differs from fetch report")
    reported_frames = ((fetch.get("media") or {}).get("pts") or {}).get("frames")
    if reported_frames is not None and int(reported_frames) != pts.size:
        raise ValueError("PTS evidence length differs from media audit")
    manifest_copy = dict(manifest)
    manifest_copy["verified_vector_sha256"] = actual_vector_hash
    manifest_copy["verified_summary"] = summary
    return pts, manifest_copy, manifest_path, vector_path


def _selected_interval(
    all_pts: np.ndarray,
    bounds: tuple[float, float],
    policy: TimerActivityPolicy,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    start_s, end_s = (float(value) for value in bounds)
    if not np.isfinite(start_s) or not np.isfinite(end_s):
        raise ValueError("reviewed wall-clock bounds must be finite")
    if start_s < 0 or end_s <= start_s:
        raise ValueError("reviewed wall-clock bounds must have positive duration")
    relative = np.asarray(all_pts, dtype=np.float64) - float(all_pts[0])
    selected = np.flatnonzero((relative >= start_s) & (relative < end_s))
    if selected.size < 2:
        raise ValueError("reviewed wall-clock interval contains fewer than two frames")
    expected = np.arange(selected[0], selected[-1] + 1, dtype=np.int64)
    if not np.array_equal(selected, expected):
        raise ValueError("reviewed wall-clock frame interval is not dense")
    selected_pts = relative[selected]
    summary = dict(summarize_pts(selected_pts))
    median_dt = float(summary["median_dt_s"])
    median_interval_fps = 1.0 / median_dt
    span_effective_fps = float(
        (selected_pts.size - 1) / (selected_pts[-1] - selected_pts[0])
    )
    raw_vfr_ratio = float(summary["vfr_ratio_p99_p01"])
    p01 = float(summary["p01_dt_s"])
    p99 = float(summary["p99_dt_s"])
    adjusted_p99 = max(p01, p99 - policy.pts_interval_quantization_tolerance_s)
    adjusted_vfr_ratio = adjusted_p99 / max(p01, 1e-12)
    summary.update({
        "effective_fps": span_effective_fps,
        "span_effective_fps": span_effective_fps,
        "median_interval_fps": median_interval_fps,
        "vfr_ratio_p99_p01": raw_vfr_ratio,
        "quantization_adjusted_vfr_ratio_p99_p01": adjusted_vfr_ratio,
    })
    if summary["nonmonotonic_intervals"]:
        raise ValueError("selected PTS interval is not strictly increasing")
    if summary["large_gap_intervals"]:
        raise ValueError("selected PTS interval contains large gaps")
    fps = span_effective_fps
    if not policy.min_effective_fps <= fps <= policy.max_effective_fps:
        raise ValueError(
            f"selected PTS effective FPS {fps:.4f} is outside "
            f"[{policy.min_effective_fps}, {policy.max_effective_fps}]"
        )
    if adjusted_vfr_ratio > policy.max_vfr_ratio_p99_p01:
        raise ValueError("selected PTS interval cadence is too variable")
    next_index = int(selected[-1]) + 1
    coverage_end = (
        float(relative[next_index])
        if next_index < relative.size
        else float(selected_pts[-1] + median_dt)
    )
    if abs(float(selected_pts[0]) - start_s) > 1e-6:
        raise ValueError("reviewed start must match a persisted frame PTS")
    if abs(coverage_end - end_s) > 1e-6:
        raise ValueError("reviewed end must match a half-open persisted frame boundary")
    return selected.astype(np.int64, copy=False), selected_pts, summary


def _stream_scalar_trace(
    video: Path,
    source_frame_indices: np.ndarray,
    timer_roi: Rect,
    policy: TimerActivityPolicy,
    expected_total_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open source video: {video}")
    count = int(source_frame_indices.size)
    change = np.zeros(count, dtype=np.float64)
    bright = np.empty(count, dtype=np.float64)
    dark = np.empty(count, dtype=np.float64)
    first_index = int(source_frame_indices[0])
    try:
        declared_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if declared_count > 0 and declared_count != expected_total_frames:
            raise ValueError(
                "OpenCV/source PTS decode count mismatch: "
                f"{declared_count} != {expected_total_frames}"
            )
        if first_index:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(first_index)):
                raise ValueError(f"OpenCV could not seek to source frame {first_index}")
            position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if np.isfinite(position) and abs(position - first_index) > 0.5:
                raise ValueError(
                    f"OpenCV seek landed at frame {position}, expected {first_index}"
                )

        previous: np.ndarray | None = None
        frame_shape: tuple[int, int] | None = None
        pixel_rect: tuple[int, int, int, int] | None = None
        for row, source_index in enumerate(source_frame_indices):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"video decode failed at source frame {source_index}")
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                raise ValueError("decoded frame must be uint8 BGR with shape [H,W,3]")
            shape = (int(frame.shape[0]), int(frame.shape[1]))
            if frame_shape is None:
                frame_shape = shape
                pixel_rect = rect_to_pixels(timer_roi, shape[1], shape[0])
                x0, y0, x1, y1 = pixel_rect
                pixels = (x1 - x0) * (y1 - y0)
                if pixels <= 0 or pixels > policy.max_roi_pixels_per_frame:
                    raise ValueError(
                        f"timer ROI has {pixels} pixels; "
                        f"limit is {policy.max_roi_pixels_per_frame}"
                    )
            elif shape != frame_shape:
                raise ValueError("decoded frame dimensions changed inside selected interval")
            assert pixel_rect is not None
            x0, y0, x1, y1 = pixel_rect
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                raise ValueError("reviewed timer ROI produced an empty crop")
            normalized = roi.astype(np.float32) / 255.0
            if previous is not None:
                change[row] = np.mean(np.abs(normalized - previous))
            bright[row] = np.mean(
                (roi >= policy.bright_pixel_threshold).astype(np.float32) * 255.0,
                dtype=np.float64,
            )
            dark[row] = np.mean(
                (roi <= policy.dark_pixel_threshold).astype(np.float32) * 255.0,
                dtype=np.float64,
            )
            previous = normalized
            position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            expected_position = int(source_index) + 1
            if np.isfinite(position) and position > 0 and abs(position - expected_position) > 0.5:
                raise ValueError(
                    f"OpenCV decode position {position} differs from expected "
                    f"{expected_position}"
                )
    finally:
        capture.release()
    if frame_shape is None or pixel_rect is None:
        raise ValueError("selected frame interval decoded no frames")
    return change, bright, dark, {
        "decode_backend": "opencv",
        "encoded_resolution_wh": [frame_shape[1], frame_shape[0]],
        "timer_roi_pixels_xyxy": list(pixel_rect),
        "timer_roi_pixels": int(
            (pixel_rect[2] - pixel_rect[0]) * (pixel_rect[3] - pixel_rect[1])
        ),
        "decoded_frames": count,
    }


def _ffprobe_video(video: Path) -> tuple[int, int, dict[str, Any]]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name",
        "-of", "json", str(video),
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace")[-2000:]
        raise ValueError(f"ffprobe failed ({completed.returncode}): {detail}")
    try:
        payload = json.loads(completed.stdout)
        streams = payload["streams"]
        stream = streams[0]
        width, height = int(stream["width"]), int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("ffprobe did not return one usable video stream") from exc
    if len(streams) != 1 or width <= 0 or height <= 0:
        raise ValueError("ffprobe video resolution is invalid or ambiguous")
    return width, height, {
        "codec_name": str(stream.get("codec_name", "")),
        "encoded_resolution_wh": [width, height],
        "ffprobe_command": command[:-1] + [video.name],
    }


def _read_exact(handle: Any, destination: memoryview) -> int:
    """Accumulate short pipe reads until one raw frame is full or EOF occurs."""

    offset = 0
    while offset < len(destination):
        count = handle.readinto(destination[offset:])
        if not count:
            break
        offset += int(count)
    return offset


def _stream_scalar_trace_ffmpeg(
    video: Path,
    source_frame_indices: np.ndarray,
    timer_roi: Rect,
    policy: TimerActivityPolicy,
    expected_total_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Decode an exact dense index interval through a bounded gray8 pipe.

    FFmpeg starts at source frame zero: there is no timestamp or keyframe seek
    whose rounding could silently shift the PTS binding. ``crop=...:exact=1``
    prevents chroma-subsampled input from rounding an odd ROI dimension. Every
    pipe read is accumulated to exactly one ROI frame, and both final row count
    and EOF are required before evidence is written.
    """

    indices = np.asarray(source_frame_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size < 2:
        raise ValueError("ffmpeg timer selection requires at least two frame indices")
    expected = np.arange(int(indices[0]), int(indices[-1]) + 1, dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise ValueError("ffmpeg timer selection must be a dense source-index interval")
    if indices[0] < 0 or indices[-1] >= expected_total_frames:
        raise ValueError("ffmpeg timer selection lies outside the PTS vector")

    width, height, probe = _ffprobe_video(video)
    x0, y0, x1, y1 = rect_to_pixels(timer_roi, width, height)
    crop_width, crop_height = x1 - x0, y1 - y0
    pixels = crop_width * crop_height
    if pixels <= 0 or pixels > policy.max_roi_pixels_per_frame:
        raise ValueError(
            f"timer ROI has {pixels} pixels; limit is {policy.max_roi_pixels_per_frame}"
        )

    first, last = int(indices[0]), int(indices[-1])
    frame_count = int(indices.size)
    # ``exact=1`` is load-bearing on 4:2:0 input. Without it ffmpeg 4.4, for
    # example, silently emits a requested 200x47 crop as 200x46.
    video_filter = (
        f"select=between(n\\,{first}\\,{last}),"
        f"crop={crop_width}:{crop_height}:{x0}:{y0}:exact=1,format=gray"
    )
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-map", "0:v:0", "-vf", video_filter,
        "-vsync", "0", "-frames:v", str(frame_count),
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    change = np.zeros(frame_count, dtype=np.float64)
    bright = np.empty(frame_count, dtype=np.float64)
    dark = np.empty(frame_count, dtype=np.float64)
    raw = bytearray(pixels)
    raw_view = memoryview(raw)
    previous: np.ndarray | None = None
    decoded_rows = 0
    failure: str | None = None

    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file)
        assert process.stdout is not None
        try:
            for row in range(frame_count):
                received = _read_exact(process.stdout, raw_view)
                if received != pixels:
                    failure = (
                        "ffmpeg raw pipe ended mid-frame: "
                        f"row={row} bytes={received}/{pixels}"
                    )
                    break
                roi = np.frombuffer(raw, dtype=np.uint8).reshape(
                    crop_height, crop_width
                )
                normalized = roi.astype(np.float32) / 255.0
                if previous is not None:
                    change[row] = np.mean(np.abs(normalized - previous))
                bright[row] = np.mean(
                    (roi >= policy.bright_pixel_threshold).astype(np.float32) * 255.0,
                    dtype=np.float64,
                )
                dark[row] = np.mean(
                    (roi <= policy.dark_pixel_threshold).astype(np.float32) * 255.0,
                    dtype=np.float64,
                )
                previous = normalized
                decoded_rows += 1
            if failure is None and process.stdout.read(1):
                failure = "ffmpeg raw pipe emitted more rows than requested"
            process.stdout.close()
            if failure is not None and process.poll() is None:
                process.kill()
            return_code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        stderr_file.flush()
        stderr_file.seek(0)
        stderr_data = stderr_file.read()

    stderr_tail = stderr_data.decode("utf-8", "replace")[-4000:]
    if failure is not None:
        raise ValueError(
            f"{failure}; ffmpeg exit={return_code}; stderr_tail={stderr_tail!r}"
        )
    if return_code != 0:
        raise ValueError(f"ffmpeg failed ({return_code}): {stderr_tail}")
    if decoded_rows != frame_count:
        raise ValueError(
            f"ffmpeg decoded row count {decoded_rows} != requested {frame_count}"
        )

    version = subprocess.run(
        ["ffmpeg", "-version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    return change, bright, dark, {
        **probe,
        "decode_backend": FFMPEG_BACKEND_VERSION,
        "decode_from_source_frame_zero": True,
        "timer_roi_pixels_xyxy": [x0, y0, x1, y1],
        "timer_roi_pixels": pixels,
        "decoded_frames": frame_count,
        "selected_source_frame_range": [first, last + 1],
        "raw_pixel_format": "gray8",
        "raw_frame_bytes": pixels,
        "raw_bytes_verified": pixels * frame_count,
        "exact_row_count_verified": True,
        "pipe_eof_verified": True,
        "ffmpeg_crop_exact": True,
        "ffmpeg_filter": video_filter,
        "ffmpeg_version": version,
        "ffmpeg_stderr_bytes": len(stderr_data),
        "ffmpeg_stderr_sha256": hashlib.sha256(stderr_data).hexdigest(),
    }


def _write_trace_npz(
    destination: Path,
    source_frame_indices: np.ndarray,
    pts_s: np.ndarray,
    change_score: np.ndarray,
    bright_mask_mean: np.ndarray,
    dark_mask_mean: np.ndarray,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                format_version=np.asarray(TRACE_VERSION),
                source_frame_idx=source_frame_indices.astype(np.int64, copy=False),
                pts_s=np.asarray(pts_s, dtype=np.float64),
                change_score=np.asarray(change_score, dtype=np.float64),
                bright_mask_mean=np.asarray(bright_mask_mean, dtype=np.float64),
                dark_mask_mean=np.asarray(dark_mask_mean, dtype=np.float64),
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def extract_timer_trace(
    fetch_report_path: str | Path,
    timer_roi_normalized_xywh: Rect,
    wall_clock_bounds_s: tuple[float, float],
    evidence_refs: Mapping[str, str],
    out_dir: str | Path,
    *,
    reviewer_identity: str,
    reviewer_kind: str,
    pts_evidence_dir: str | Path | None = None,
    policy: TimerActivityPolicy = TimerActivityPolicy(),
    verify_source_hash: bool = True,
    verify_pts_hash: bool = True,
    decode_backend: str = "opencv",
) -> dict[str, Any]:
    """Extract, persist, and propose ranges without creating admission state."""

    policy.validate()
    if decode_backend not in DECODE_BACKENDS:
        raise ValueError(f"decode_backend must be one of {DECODE_BACKENDS}")
    reviewed_evidence = _review_evidence(evidence_refs)
    fetch_path = Path(fetch_report_path)
    try:
        fetch = json.loads(fetch_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read fetch report: {fetch_path}") from exc
    if not isinstance(fetch, dict):
        raise ValueError("fetch report must contain one JSON object")
    video_id = str(fetch.get("video_id", "")).strip()
    source_sha256 = str(fetch.get("sha256", "")).strip().lower()
    if not video_id:
        raise ValueError("fetch report video_id is missing")
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise ValueError("fetch report source SHA-256 is invalid")
    video = _safe_child(fetch_path.parent, fetch.get("source_file"), "source_file")
    if not video.is_file():
        raise ValueError(f"fetch report source video is missing: {video}")
    actual_source_hash = sha256_file(video) if verify_source_hash else None
    if verify_source_hash and actual_source_hash != source_sha256:
        raise ValueError("source video hash does not match fetch report")

    context = TimerReviewContext(
        video_id=video_id,
        source_sha256=source_sha256,
        timer_roi_normalized_xywh=timer_roi_normalized_xywh,
        timer_roi_evidence_reviewed=True,
        wall_clock_bounds_s=wall_clock_bounds_s,
        bounds_evidence_reviewed=True,
        reviewer_identity=reviewer_identity,
        reviewer_kind=reviewer_kind,
        nominal_loadless_duration_s=_nominal_loadless_duration(fetch),
        evidence=tuple(f"{name}={value}" for name, value in reviewed_evidence.items()),
    )
    context_failures = context.failures()
    if context_failures:
        raise ValueError("; ".join(context_failures))

    all_pts, pts_manifest, pts_manifest_path, pts_vector_path = _load_verified_pts(
        fetch_path,
        fetch,
        source_sha256,
        pts_evidence_dir,
        verify_hash=verify_pts_hash,
    )
    selected_indices, selected_pts, selected_pts_summary = _selected_interval(
        all_pts, wall_clock_bounds_s, policy
    )
    stream = (
        _stream_scalar_trace
        if decode_backend == "opencv"
        else _stream_scalar_trace_ffmpeg
    )
    change, bright, dark, decode = stream(
        video,
        selected_indices,
        timer_roi_normalized_xywh,
        policy,
        expected_total_frames=int(all_pts.size),
    )
    if decode["decoded_frames"] != selected_indices.size:
        raise ValueError("selected PTS/decode count mismatch")

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trace_path = destination / TRACE_FILE
    _write_trace_npz(
        trace_path, selected_indices, selected_pts, change, bright, dark
    )
    trace_hash = sha256_file(trace_path)
    fetch_hash = sha256_file(fetch_path)
    pts_manifest_hash = sha256_file(pts_manifest_path)
    pts_vector_hash = str(pts_manifest["verified_vector_sha256"])
    trace_binding = {
        "format_version": TRACE_VERSION,
        "source_sha256": source_sha256,
        "fetch_report_sha256": fetch_hash,
        "pts_manifest_sha256": pts_manifest_hash,
        "pts_vector_sha256": pts_vector_hash,
        "trace_npz_sha256": trace_hash,
        "source_frame_range": [
            int(selected_indices[0]), int(selected_indices[-1]) + 1
        ],
        "frames": int(selected_indices.size),
    }
    proposal = segment_timer_activity(
        selected_pts,
        change,
        context,
        policy,
        bright_mask_mean=bright,
        dark_mask_mean=dark,
    )
    if proposal.get("auto_admitted") is not False:
        raise AssertionError("timer activity helper violated the no-auto-admission contract")
    proposal["trace_binding"] = trace_binding
    proposal_path = destination / PROPOSAL_FILE
    write_timer_activity_diagnostics(proposal, proposal_path)
    proposal_hash = sha256_file(proposal_path)

    manifest = {
        "format_version": MANIFEST_VERSION,
        "video_id": video_id,
        "source_video": {
            "file": video.name,
            "sha256": source_sha256,
            "hash_verified": verify_source_hash,
            "actual_sha256": actual_source_hash,
        },
        "fetch_report": {
            "file": fetch_path.name,
            "sha256": fetch_hash,
        },
        "pts_evidence": {
            "manifest_file": pts_manifest_path.name,
            "manifest_sha256": pts_manifest_hash,
            "vector_file": pts_vector_path.name,
            "vector_sha256": pts_vector_hash,
            "vector_hash_verified": verify_pts_hash,
            "frames": int(all_pts.size),
        },
        "review": {
            "timer_roi_normalized_xywh": list(timer_roi_normalized_xywh),
            "wall_clock_bounds_s": list(wall_clock_bounds_s),
            "evidence_refs": reviewed_evidence,
            "reviewer_identity": str(reviewer_identity).strip(),
            "reviewer_kind": reviewer_kind,
            "human_reviewed": context.human_reviewed,
            "nominal_loadless_duration_s": context.nominal_loadless_duration_s,
            "human_review_required_for_proposal": True,
        },
        "selection": {
            "source_frame_range": trace_binding["source_frame_range"],
            "frames": int(selected_indices.size),
            "pts_summary": selected_pts_summary,
            **decode,
        },
        "policy": asdict(policy),
        "trace": {
            "file": trace_path.name,
            "sha256": trace_hash,
            "arrays": {
                "source_frame_idx": "int64",
                "pts_s": "float64",
                "change_score": "float64",
                "bright_mask_mean": "float64",
                "dark_mask_mean": "float64",
            },
        },
        "proposal": {
            "file": proposal_path.name,
            "sha256": proposal_hash,
            "status": proposal["status"],
            "automatic_gates_passed": proposal["automatic_gates_passed"],
            "auto_admitted": False,
            "requires_human_review": True,
        },
        "admission": {
            "wild_boundaries_created": False,
            "auto_admitted": False,
            "next_step": (
                "Review raw source frames and every proposed half-open range; "
                "create WildBoundaries separately only after human approval."
            ),
        },
    }
    manifest_path = destination / MANIFEST_FILE
    _write_json(manifest_path, manifest)
    return manifest


def _parse_evidence_refs(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        name, separator, reference = value.partition("=")
        name, reference = name.strip(), reference.strip()
        if not separator or not name or not reference:
            raise ValueError("--evidence-ref must be NAME=REFERENCE")
        if name in parsed:
            raise ValueError(f"duplicate evidence reference name: {name}")
        parsed[name] = reference
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-report", type=Path, required=True)
    parser.add_argument(
        "--timer-roi", type=float, nargs=4, metavar=("X", "Y", "W", "H"),
        required=True, help="evidence-reviewed normalized timer ROI"
    )
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument(
        "--evidence-ref", action="append", default=[], metavar="NAME=REFERENCE",
        help="repeat; timer_roi and wall_clock_bounds names are required",
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewer-kind", choices=REVIEWER_KINDS, required=True)
    parser.add_argument("--pts-evidence-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-verify-source-hash", action="store_true")
    parser.add_argument("--no-verify-pts-hash", action="store_true")
    parser.add_argument(
        "--decode-backend", choices=DECODE_BACKENDS, default="opencv",
        help="frame reader; ffmpeg is a from-zero exact-index fallback for codecs OpenCV cannot read",
    )
    args = parser.parse_args()
    try:
        evidence = _parse_evidence_refs(args.evidence_ref)
        manifest = extract_timer_trace(
            args.fetch_report,
            tuple(args.timer_roi),
            (args.start_s, args.end_s),
            evidence,
            args.out,
            reviewer_identity=args.reviewer,
            reviewer_kind=args.reviewer_kind,
            pts_evidence_dir=args.pts_evidence_dir,
            verify_source_hash=not args.no_verify_source_hash,
            verify_pts_hash=not args.no_verify_pts_hash,
            decode_backend=args.decode_backend,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({
        "video_id": manifest["video_id"],
        "frames": manifest["selection"]["frames"],
        "proposal": manifest["proposal"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
