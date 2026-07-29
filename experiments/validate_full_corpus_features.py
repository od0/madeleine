"""Validate the assembled all-video NitroGen feature corpus without changing it.

The builder already validates shards while assembling them.  This program is
an independent, read-only handoff check: it reconstructs corpus membership and
part boundaries from source metadata, checks every NPZ header and hard link,
and reconciles all manifests and split lists.  ``--deep-shards`` additionally
loads array contents, checks feature finiteness, and compares supervision to
the mapped parquet labels.  That pass reads the full feature corpus and is
therefore deliberately opt-in.

The only file this program writes is the JSON report named by ``--out``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable
import zipfile

import cv2
import numpy as np
import pyarrow.parquet as pq

from data.build_dataset import (
    MAX_PART_FRAMES,
    MIN_RUN_FRAMES,
    _foreign_runs,
    _run_keys,
)
from data.schema import KEY_ORDER
from experiments.audit_corpus_contiguity import VideoPlan, build_plans
from experiments.build_full_corpus_features import (
    CHUNK_FRAME_SCHEMA,
    FEATURE_DIM,
)


NATIVE_MODE = "opencv_native_60hz"
RESAMPLED_MODE = "ffmpeg_timestamp_resample_60hz"
FRAMES_PER_HOUR_60HZ = 216_000
MAX_RESAMPLED_TAIL_REPEAT = 3


@dataclass(frozen=True)
class CorpusExpectations:
    valid_videos: int
    rejected_videos: int
    chunk_rows: int
    source_label_frames: int
    sessions: int
    train_frames: int
    native_videos: int
    resampled_videos: int
    native_sessions: int
    resampled_sessions: int
    native_frames: int
    resampled_frames: int
    unflagged_videos: int
    flagged_videos: int
    unflagged_sessions: int
    flagged_sessions: int
    unflagged_frames: int
    flagged_frames: int
    axis_sign_indeterminate: int
    tail_truncated_frames: int
    skipped_short_frames: int


FULL_211_EXPECTATIONS = CorpusExpectations(
    valid_videos=211,
    rejected_videos=21,
    chunk_rows=27_165,
    source_label_frames=32_598_000,
    sessions=1_554,
    train_frames=32_598_000,
    native_videos=194,
    resampled_videos=17,
    native_sessions=1_460,
    resampled_sessions=94,
    native_frames=30_706_800,
    resampled_frames=1_891_200,
    unflagged_videos=93,
    flagged_videos=118,
    unflagged_sessions=1_078,
    flagged_sessions=476,
    unflagged_frames=22_896_000,
    flagged_frames=9_702_000,
    axis_sign_indeterminate=181,
    tail_truncated_frames=0,
    skipped_short_frames=0,
)


@dataclass(frozen=True)
class VideoMetadata:
    average_fps: float
    decoded_frames: int


@dataclass(frozen=True)
class PlannedPart:
    session_id: str
    start_frame: int
    end_frame: int

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


def read_video_metadata(path: Path) -> VideoMetadata:
    """Read only the container metadata needed to reproduce decoder policy."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video metadata: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0 or frames < 1:
        raise ValueError(f"invalid video metadata for {path}: {fps=} {frames=}")
    return VideoMetadata(average_fps=fps, decoded_frames=frames)


def _decoder_plan(metadata: VideoMetadata) -> tuple[str, int]:
    if abs(metadata.average_fps - 60.0) <= 0.1:
        return NATIVE_MODE, metadata.decoded_frames
    duration = metadata.decoded_frames / metadata.average_fps
    return RESAMPLED_MODE, int(round(duration * 60.0))


def _group_runs(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: int(row["start_frame"]))
    runs: list[list[dict[str, Any]]] = []
    for row in ordered:
        if runs and int(runs[-1][-1]["end_frame"]) == int(row["start_frame"]):
            runs[-1].append(row)
        else:
            runs.append([row])
    return runs


def _plan_parts(
    video_id: str,
    rows: list[dict[str, Any]],
    timeline_frames: int,
) -> tuple[list[PlannedPart], int, int, int]:
    parts: list[PlannedPart] = []
    truncated = 0
    skipped = 0
    runs = _group_runs(rows)
    for run in runs:
        start = int(run[0]["start_frame"])
        end = int(run[-1]["end_frame"])
        if end > timeline_frames:
            truncated += end - timeline_frames
            end = timeline_frames
        for part_start in range(start, end, MAX_PART_FRAMES):
            part_end = min(part_start + MAX_PART_FRAMES, end)
            if part_end - part_start < MIN_RUN_FRAMES:
                skipped += part_end - part_start
                continue
            index = len(parts)
            parts.append(PlannedPart(
                session_id=f"{video_id}__r{index:03d}",
                start_frame=part_start,
                end_frame=part_end,
            ))
    return parts, len(runs), truncated, skipped


def _read_npy_header(
    archive: zipfile.ZipFile, member: str
) -> tuple[tuple[int, ...], np.dtype[Any]]:
    with archive.open(member) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in {(2, 0), (3, 0)}:
            shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY version {version} in {member}")
    return tuple(int(value) for value in shape), np.dtype(dtype)


def _npz_headers(path: Path) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        expected = {
            "features.npy", "keys.npy", "engine_frame_idx.npy",
            "input_active.npy", "session_id.npy",
        }
        if names != expected:
            raise ValueError(
                f"{path}: NPZ members differ: missing={sorted(expected-names)} "
                f"extra={sorted(names-expected)}"
            )
        return {name[:-4]: _read_npy_header(archive, name) for name in names}


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-10)


def _git_commit(repo: Path | None) -> str | None:
    if repo is None or not repo.is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def validate_full_corpus_features(
    *,
    raw_root: Path,
    chunk_index: Path,
    fetch_report: Path,
    mapped_root: Path,
    chunk_frames: Path,
    feature_root: Path,
    output_root: Path,
    completion_marker: Path,
    build_log: Path | None,
    video_log_root: Path | None,
    expectations: CorpusExpectations = FULL_211_EXPECTATIONS,
    deep_shards: bool = False,
    repo: Path | None = None,
    metadata_reader: Callable[[Path], VideoMetadata] = read_video_metadata,
) -> dict[str, Any]:
    """Return a machine-readable validation report; never mutate inputs."""

    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(completion_marker.is_file(), f"completion marker missing: {completion_marker}")
    manifest_path = output_root / "full_corpus_manifest.json"
    if not manifest_path.is_file():
        errors.append(f"final manifest missing: {manifest_path}")
        return {
            "ok": False,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "deep_shards": deep_shards,
            "git_commit": _git_commit(repo),
            "expected": asdict(expectations),
            "observed": {},
            "errors": errors,
        }

    try:
        plans, rejected = build_plans(raw_root, chunk_index, fetch_report)
    except Exception as error:  # noqa: BLE001 - evidence belongs in report
        errors.append(f"source planning failed: {type(error).__name__}: {error}")
        plans, rejected = [], []
    plan_by_id = {plan.video_id: plan for plan in plans}
    plan_ids = set(plan_by_id)
    require(len(plans) == expectations.valid_videos,
            f"valid videos {len(plans)} != {expectations.valid_videos}")
    require(len(rejected) == expectations.rejected_videos,
            f"rejected videos {len(rejected)} != {expectations.rejected_videos}")

    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("format") == "resnet18_imagenet_avgpool_float16_v1",
            "final manifest feature format mismatch")
    require(manifest.get("source_kind") == "mapped_foreign_video",
            "final manifest source kind mismatch")
    if completion_marker.is_file():
        require(completion_marker.stat().st_mtime >= manifest_path.stat().st_mtime,
                "completion marker predates final manifest")
    manifest_videos = manifest.get("videos", [])
    manifest_by_id = {
        str(row.get("video_id")): row for row in manifest_videos
        if isinstance(row, dict) and row.get("video_id") is not None
    }
    require(len(manifest_videos) == len(manifest_by_id),
            "final manifest contains duplicate or malformed video rows")
    require(set(manifest_by_id) == plan_ids,
            "final manifest video membership differs from source plans")

    try:
        table = pq.read_table(chunk_frames)
        require(table.schema.equals(CHUNK_FRAME_SCHEMA),
                f"chunk-frame schema differs: {table.schema}")
        chunk_rows = table.to_pylist()
    except Exception as error:  # noqa: BLE001
        errors.append(f"chunk-frame read failed: {type(error).__name__}: {error}")
        chunk_rows = []
    require(len(chunk_rows) == expectations.chunk_rows,
            f"chunk rows {len(chunk_rows)} != {expectations.chunk_rows}")
    rows_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in chunk_rows:
        rows_by_video.setdefault(str(row["video_id"]), []).append(row)
        require(float(row["grid_hz"]) == 60.0, f"non-60-Hz chunk: {row}")
        require(int(row["n_rows"]) == 1_200, f"non-1200-row chunk: {row}")
        require(int(row["end_frame"]) - int(row["start_frame"]) == 1_200,
                f"non-exclusive chunk range: {row}")
    require(set(rows_by_video) == plan_ids,
            "chunk-frame video membership differs from source plans")

    # Independently reconstruct the generated chunk table from the source
    # chunk index, but do not call build_chunk_frames because it writes.
    if plan_ids:
        source_rows = pq.read_table(chunk_index).to_pylist()
        expected_chunk_rows: list[dict[str, Any]] = []
        for source in source_rows:
            video_id = str(source["video_id"])
            if video_id not in plan_ids:
                continue
            count = int(source["chunk_size"])
            number = int(source["chunk_id"])
            expected_chunk_rows.append({
                "video_id": video_id,
                "chunk_id": f"{video_id}_chunk_{number:04d}",
                "start_frame": number * count,
                "end_frame": (number + 1) * count,
                "start_time": number * 20.0,
                "end_time": (number + 1) * 20.0,
                "grid_hz": float(source["grid_hz"]),
                "n_rows": count,
            })
        expected_chunk_rows.sort(key=lambda row: (row["video_id"], row["start_frame"]))
        observed_sorted = sorted(
            chunk_rows, key=lambda row: (str(row["video_id"]), int(row["start_frame"]))
        )
        require(observed_sorted == expected_chunk_rows,
                "chunk-frame table does not exactly match source chunk index")

    mapping_ids = {
        path.parent.name for path in mapped_root.glob("*/mapping_report.json")
    }
    feature_manifest_ids = {
        path.parent.name
        for path in feature_root.glob("*/feature_build_manifest.json")
    }
    require(mapping_ids == plan_ids, "mapping-report membership differs from plans")
    require(feature_manifest_ids == plan_ids,
            "feature-manifest membership differs from plans")

    expected_sessions: dict[str, tuple[str, PlannedPart]] = {}
    expected_unflagged: set[str] = set()
    mode_video_counts: Counter[str] = Counter()
    mode_session_counts: Counter[str] = Counter()
    mode_frame_counts: Counter[str] = Counter()
    total_train_frames = 0
    total_source_frames = sum(plan.label_frames for plan in plans)
    total_truncated = 0
    total_skipped = 0
    total_imputed = 0
    flagged_videos = 0
    axis_indeterminate = 0
    unflagged_frames = 0
    flagged_frames = 0
    confidences: list[float] = []
    shard_headers_checked = 0
    hardlinks_checked = 0
    deep_shards_checked = 0

    for video_id in sorted(plan_ids):
        plan = plan_by_id[video_id]
        mapping_path = mapped_root / video_id / "mapping_report.json"
        feature_manifest_path = feature_root / video_id / "feature_build_manifest.json"
        if not mapping_path.is_file() or not feature_manifest_path.is_file():
            continue
        mapping = json.loads(mapping_path.read_text())
        feature_manifest = json.loads(feature_manifest_path.read_text())
        reports = feature_manifest.get("videos", [])
        if len(reports) != 1:
            errors.append(f"{video_id}: expected one feature video report")
            continue
        feature_report = reports[0]
        require(feature_report.get("video_id") == video_id,
                f"{video_id}: feature report identity mismatch")
        require(feature_manifest.get("format") == "resnet18_imagenet_avgpool_float16_v1",
                f"{video_id}: feature format mismatch")

        try:
            metadata = metadata_reader(Path(plan.path))
        except Exception as error:  # noqa: BLE001
            errors.append(f"{video_id}: metadata read failed: {error}")
            continue
        mode, timeline_frames = _decoder_plan(metadata)
        mode_video_counts[mode] += 1
        rows = rows_by_video.get(video_id, [])
        parts, run_count, truncated, skipped = _plan_parts(
            video_id, rows, timeline_frames
        )
        total_truncated += truncated
        total_skipped += skipped

        reported_parts = feature_report.get("parts", [])
        require(len(reported_parts) == len(parts),
                f"{video_id}: reported parts {len(reported_parts)} != {len(parts)}")
        require(int(feature_report.get("runs", -1)) == run_count,
                f"{video_id}: run count mismatch")
        require(feature_report.get("decoder_mode") == mode,
                f"{video_id}: decoder mode mismatch")
        video_meta = feature_report.get("video", {})
        require(_close(video_meta.get("average_fps", -1), metadata.average_fps),
                f"{video_id}: average FPS mismatch")
        require(int(video_meta.get("decoded_frames", -1)) == metadata.decoded_frames,
                f"{video_id}: decoded frame count mismatch")
        require(int(video_meta.get("nominal_timeline_frames", -1)) == timeline_frames,
                f"{video_id}: nominal timeline mismatch")
        require(int(feature_report.get("tail_truncated_frames", -1)) == truncated,
                f"{video_id}: tail truncation mismatch")
        require(int(feature_report.get("skipped_short_frames", -1)) == skipped,
                f"{video_id}: skipped-short mismatch")
        require(feature_report.get("bind_map") == mapping.get("bind_map"),
                f"{video_id}: bind map mismatch")
        require(_close(feature_report.get("bind_confidence", -1), mapping.get("confidence", -2)),
                f"{video_id}: feature bind confidence mismatch")

        flagged = bool(mapping.get("flagged"))
        flagged_videos += int(flagged)
        axis_indeterminate += int(bool(mapping.get("axis_sign_indeterminate")))
        confidence = float(mapping.get("confidence", float("nan")))
        require(math.isfinite(confidence), f"{video_id}: non-finite bind confidence")
        confidences.append(confidence)
        require(int(mapping.get("chunks_mapped", -1)) == len(rows),
                f"{video_id}: mapped chunk count mismatch")
        require(int(mapping.get("chunks_skipped", -1)) == 0,
                f"{video_id}: mapping reports skipped chunks")

        manifest_video = manifest_by_id.get(video_id, {})
        planned_ids = [part.session_id for part in parts]
        require(manifest_video.get("sessions") == planned_ids,
                f"{video_id}: final-manifest session list mismatch")
        video_frames = sum(part.frames for part in parts)
        require(int(manifest_video.get("frames", -1)) == video_frames,
                f"{video_id}: final-manifest frame count mismatch")
        require(int(manifest_video.get("source_label_frames", -1)) == plan.label_frames,
                f"{video_id}: source-label frame count mismatch")
        require(_close(manifest_video.get("label_hours", -1),
                       video_frames / FRAMES_PER_HOUR_60HZ),
                f"{video_id}: train-ready hours mismatch")
        require(_close(manifest_video.get("train_to_source_fraction", -1),
                       video_frames / plan.label_frames),
                f"{video_id}: train/source fraction mismatch")
        require(bool(manifest_video.get("bind_flagged")) == flagged,
                f"{video_id}: final bind flag mismatch")
        require(_close(manifest_video.get("bind_confidence", -1), confidence),
                f"{video_id}: final bind confidence mismatch")
        require(int(manifest_video.get("label_run_count", -1)) == plan.label_run_count,
                f"{video_id}: continuity run count mismatch")
        require(_close(manifest_video.get("long_context_fraction", -1),
                       plan.long_context_fraction),
                f"{video_id}: long-context fraction mismatch")
        require(manifest_video.get("decoder_mode") == mode,
                f"{video_id}: final decoder mode mismatch")
        require(_close(manifest_video.get("source_average_fps", -1),
                       metadata.average_fps),
                f"{video_id}: final source FPS mismatch")
        require(int(manifest_video.get("source_decoded_frames", -1)) ==
                metadata.decoded_frames,
                f"{video_id}: final source decoded-frame count mismatch")
        require(int(manifest_video.get("nominal_timeline_frames", -1)) ==
                timeline_frames,
                f"{video_id}: final nominal timeline mismatch")

        part_imputed = 0
        for index, part in enumerate(parts):
            if index >= len(reported_parts):
                break
            reported = reported_parts[index]
            require(reported.get("session_id") == part.session_id,
                    f"{part.session_id}: reported ID mismatch")
            require(reported.get("source_frame_range") == [
                part.start_frame, part.end_frame
            ], f"{part.session_id}: source range mismatch")
            require(int(reported.get("frames", -1)) == part.frames,
                    f"{part.session_id}: reported frame count mismatch")
            require(reported.get("npz") == f"{part.session_id}.npz",
                    f"{part.session_id}: NPZ filename mismatch")
            require(reported.get("decoder_mode") == mode,
                    f"{part.session_id}: part decoder mode mismatch")
            imputed = int(reported.get("imputed_tail_frames", -1))
            require(0 <= imputed <= MAX_RESAMPLED_TAIL_REPEAT,
                    f"{part.session_id}: invalid tail-imputation count {imputed}")
            if mode == NATIVE_MODE:
                require(imputed == 0,
                        f"{part.session_id}: native shard reports imputation")
            part_imputed += max(0, imputed)
            sidecar_path = feature_root / video_id / f"{part.session_id}.decode.json"
            if mode == RESAMPLED_MODE:
                require(sidecar_path.is_file(),
                        f"{part.session_id}: resampled decode sidecar missing")
            if sidecar_path.is_file():
                sidecar = json.loads(sidecar_path.read_text())
                require(sidecar.get("decoder_mode") == mode,
                        f"{part.session_id}: sidecar decoder mode mismatch")
                require(sidecar.get("source_frame_range") == [
                    part.start_frame, part.end_frame
                ], f"{part.session_id}: sidecar range mismatch")
                require(int(sidecar.get("imputed_tail_frames", -1)) == imputed,
                        f"{part.session_id}: sidecar imputation mismatch")

            expected_sessions[part.session_id] = (video_id, part)
            if not flagged:
                expected_unflagged.add(part.session_id)
            mode_session_counts[mode] += 1
            mode_frame_counts[mode] += part.frames
            total_train_frames += part.frames

        require(int(feature_report.get("imputed_tail_frames", -1)) == part_imputed,
                f"{video_id}: feature imputation aggregate mismatch")
        require(int(manifest_video.get("imputed_tail_frames", -1)) == part_imputed,
                f"{video_id}: final imputation aggregate mismatch")
        require(int(manifest_video.get("tail_truncated_frames", -1)) == truncated,
                f"{video_id}: final truncation mismatch")
        require(int(manifest_video.get("skipped_short_frames", -1)) == skipped,
                f"{video_id}: final skipped-short mismatch")
        total_imputed += part_imputed
        if flagged:
            flagged_frames += video_frames
        else:
            unflagged_frames += video_frames

    train_path = output_root / "train_sessions.txt"
    unflagged_path = output_root / "unflagged_sessions.txt"
    val_path = output_root / "val_sessions.txt"
    train_ids = _read_lines(train_path) if train_path.is_file() else []
    unflagged_ids = _read_lines(unflagged_path) if unflagged_path.is_file() else []
    require(train_path.is_file(), "train_sessions.txt missing")
    require(unflagged_path.is_file(), "unflagged_sessions.txt missing")
    require(val_path.is_file(), "val_sessions.txt missing")
    require(not val_path.is_file() or val_path.read_text() == "",
            "val_sessions.txt must be empty")
    expected_ids = sorted(expected_sessions)
    require(train_ids == expected_ids,
            "train_sessions.txt is not the exact sorted session set")
    require(unflagged_ids == sorted(expected_unflagged),
            "unflagged_sessions.txt is not the exact mapped-quality subset")

    output_npzs = {path.stem for path in output_root.glob("*.npz")}
    source_npzs = {
        path.relative_to(feature_root).as_posix()
        for path in feature_root.glob("*/*.npz")
    }
    expected_source_npzs = {
        f"{video_id}/{session_id}.npz"
        for session_id, (video_id, _part) in expected_sessions.items()
    }
    require(output_npzs == set(expected_ids),
            "assembled NPZ membership differs from expected sessions")
    require(source_npzs == expected_source_npzs,
            "per-video NPZ membership differs from expected sessions")

    deep_keys: dict[str, np.ndarray] = {}
    if deep_shards:
        for video_id in sorted(plan_ids):
            metadata = metadata_reader(Path(plan_by_id[video_id].path))
            _mode, timeline_frames = _decoder_plan(metadata)
            runs = _foreign_runs(chunk_frames, mapped_root / video_id, video_id)
            part_index = 0
            for run in runs:
                run_keys = _run_keys(run)
                start = int(run[0]["start_frame"])
                end = int(run[-1]["end_frame"])
                if end > timeline_frames:
                    end = timeline_frames
                    run_keys = run_keys[:max(0, end - start)]
                for part_start in range(start, end, MAX_PART_FRAMES):
                    part_end = min(part_start + MAX_PART_FRAMES, end)
                    if part_end - part_start < MIN_RUN_FRAMES:
                        continue
                    session_id = f"{video_id}__r{part_index:03d}"
                    deep_keys[session_id] = run_keys[
                        part_start - start:part_end - start
                    ]
                    part_index += 1

    for session_id, (video_id, part) in expected_sessions.items():
        source = feature_root / video_id / f"{session_id}.npz"
        destination = output_root / f"{session_id}.npz"
        if not source.is_file() or not destination.is_file():
            errors.append(f"{session_id}: source or assembled NPZ missing")
            continue
        try:
            same_file = os.path.samefile(source, destination)
            require(same_file,
                    f"{session_id}: assembled file is not source hard link")
            hardlinks_checked += int(same_file)
            require(destination.stat().st_nlink >= 2,
                    f"{session_id}: assembled inode has fewer than two links")
            headers = _npz_headers(destination)
            shard_headers_checked += 1
            require(headers["features"] == ((part.frames, FEATURE_DIM), np.dtype(np.float16)),
                    f"{session_id}: feature header mismatch {headers['features']}")
            require(headers["keys"] == ((part.frames, len(KEY_ORDER)), np.dtype(np.uint8)),
                    f"{session_id}: key header mismatch {headers['keys']}")
            require(headers["engine_frame_idx"] == ((part.frames,), np.dtype(np.int64)),
                    f"{session_id}: engine-index header mismatch")
            require(headers["input_active"] == ((part.frames,), np.dtype(np.uint8)),
                    f"{session_id}: active-mask header mismatch")
            with np.load(destination, allow_pickle=False) as archive:
                stored_id = str(archive["session_id"].reshape(()).item())
                require(stored_id == session_id,
                        f"{session_id}: stored session ID is {stored_id!r}")
                if deep_shards:
                    features = archive["features"]
                    keys = archive["keys"]
                    engine = archive["engine_frame_idx"]
                    active = archive["input_active"]
                    require(bool(np.isfinite(features).all()),
                            f"{session_id}: non-finite feature value")
                    require(np.array_equal(keys, deep_keys.get(session_id)),
                            f"{session_id}: keys differ from mapped labels")
                    require(np.array_equal(
                        engine,
                        np.arange(part.start_frame, part.end_frame, dtype=np.int64),
                    ), f"{session_id}: engine indices are not dense source range")
                    require(bool(np.all(active == 1)),
                            f"{session_id}: input_active is not all one")
                    deep_shards_checked += 1
        except Exception as error:  # noqa: BLE001
            errors.append(f"{session_id}: shard validation failed: {error}")

    temporary_artifacts = sorted(str(path) for root in (feature_root, output_root)
                                 for path in root.rglob("*.tmp.*"))
    chunk_temp = chunk_frames.with_suffix(".tmp.parquet")
    if chunk_temp.exists():
        temporary_artifacts.append(str(chunk_temp))
    require(not temporary_artifacts,
            f"temporary artifacts remain: {temporary_artifacts[:10]}")

    require(len(expected_sessions) == expectations.sessions,
            f"sessions {len(expected_sessions)} != {expectations.sessions}")
    require(total_source_frames == expectations.source_label_frames,
            f"source frames {total_source_frames} != {expectations.source_label_frames}")
    require(total_train_frames == expectations.train_frames,
            f"train frames {total_train_frames} != {expectations.train_frames}")
    require(total_truncated == expectations.tail_truncated_frames,
            f"truncated frames {total_truncated} != {expectations.tail_truncated_frames}")
    require(total_skipped == expectations.skipped_short_frames,
            f"skipped frames {total_skipped} != {expectations.skipped_short_frames}")
    require(mode_video_counts[NATIVE_MODE] == expectations.native_videos,
            "native video count mismatch")
    require(mode_video_counts[RESAMPLED_MODE] == expectations.resampled_videos,
            "resampled video count mismatch")
    require(mode_session_counts[NATIVE_MODE] == expectations.native_sessions,
            "native session count mismatch")
    require(mode_session_counts[RESAMPLED_MODE] == expectations.resampled_sessions,
            "resampled session count mismatch")
    require(mode_frame_counts[NATIVE_MODE] == expectations.native_frames,
            "native frame count mismatch")
    require(mode_frame_counts[RESAMPLED_MODE] == expectations.resampled_frames,
            "resampled frame count mismatch")
    require(flagged_videos == expectations.flagged_videos,
            "flagged video count mismatch")
    require(len(plans) - flagged_videos == expectations.unflagged_videos,
            "unflagged video count mismatch")
    require(len(expected_unflagged) == expectations.unflagged_sessions,
            "unflagged session count mismatch")
    require(len(expected_sessions) - len(expected_unflagged) == expectations.flagged_sessions,
            "flagged session count mismatch")
    require(unflagged_frames == expectations.unflagged_frames,
            "unflagged frame count mismatch")
    require(flagged_frames == expectations.flagged_frames,
            "flagged frame count mismatch")
    require(axis_indeterminate == expectations.axis_sign_indeterminate,
            "axis-sign-indeterminate video count mismatch")

    expected_mode_manifest = {mode: count for mode, count in {
        NATIVE_MODE: expectations.native_videos,
        RESAMPLED_MODE: expectations.resampled_videos,
    }.items() if count}
    require(manifest.get("decoder_mode_counts") == expected_mode_manifest,
            "final decoder-mode aggregate mismatch")
    scalar_expectations = {
        "video_count": expectations.valid_videos,
        "session_count": expectations.sessions,
        "train_frames": expectations.train_frames,
        "source_label_frames": expectations.source_label_frames,
        "tail_truncated_frames": expectations.tail_truncated_frames,
        "skipped_short_frames": expectations.skipped_short_frames,
        "unflagged_video_count": expectations.unflagged_videos,
        "unflagged_session_count": expectations.unflagged_sessions,
    }
    for key, expected in scalar_expectations.items():
        require(int(manifest.get(key, -1)) == expected,
                f"final manifest {key}={manifest.get(key)!r} != {expected}")
    require(int(manifest.get("imputed_tail_frames", -1)) == total_imputed,
            "final imputed-tail aggregate mismatch")
    require(_close(manifest.get("train_label_hours_at_60hz", -1),
                   total_train_frames / FRAMES_PER_HOUR_60HZ),
            "final train hours mismatch")
    require(_close(manifest.get("source_label_hours_at_60hz", -1),
                   total_source_frames / FRAMES_PER_HOUR_60HZ),
            "final source hours mismatch")
    require(_close(manifest.get("train_to_source_fraction", -1),
                   total_train_frames / total_source_frames),
            "final train/source fraction mismatch")

    if build_log is not None:
        require(build_log.is_file(), f"build log missing: {build_log}")
        if build_log.is_file():
            statuses = re.findall(
                rf"^\[(\d+)/{expectations.valid_videos}\] (\S+) (ok|ERROR)$",
                build_log.read_text(),
                flags=re.MULTILINE,
            )
            latest = statuses[-expectations.valid_videos:]
            require(len(latest) == expectations.valid_videos,
                    "build log lacks a complete final status sequence")
            if len(latest) == expectations.valid_videos:
                require([int(row[0]) for row in latest] == list(
                    range(1, expectations.valid_videos + 1)
                ), "build log final counters are not contiguous")
                require({row[1] for row in latest} == plan_ids,
                        "build log final video membership mismatch")
                require(all(row[2] == "ok" for row in latest),
                        "build log final sequence contains ERROR")
    if video_log_root is not None:
        video_logs = {path.stem for path in video_log_root.glob("*.log")}
        require(video_logs == plan_ids, "per-video log membership mismatch")
        for video_id in plan_ids & video_logs:
            require((video_log_root / f"{video_id}.log").stat().st_size > 0,
                    f"{video_id}: per-video log is empty")

    long_context_targets = sum(plan.long_context_targets for plan in plans)
    confidence_summary: dict[str, float] = {}
    if confidences:
        confidence_summary = {
            "min": float(np.min(confidences)),
            "p10": float(np.percentile(confidences, 10)),
            "median": float(np.median(confidences)),
            "p90": float(np.percentile(confidences, 90)),
            "max": float(np.max(confidences)),
        }
    observed = {
        "valid_videos": len(plans),
        "rejected_videos": len(rejected),
        "chunk_rows": len(chunk_rows),
        "source_bytes": sum(plan.bytes for plan in plans),
        "source_label_frames": total_source_frames,
        "source_label_hours_at_60hz": total_source_frames / FRAMES_PER_HOUR_60HZ,
        "sessions": len(expected_sessions),
        "shard_headers_checked": shard_headers_checked,
        "hardlinks_checked": hardlinks_checked,
        "deep_shards_checked": deep_shards_checked,
        "train_frames": total_train_frames,
        "train_hours_at_60hz": total_train_frames / FRAMES_PER_HOUR_60HZ,
        "train_to_source_fraction": (
            total_train_frames / total_source_frames if total_source_frames else 0.0
        ),
        "decoder_mode_video_counts": dict(mode_video_counts),
        "decoder_mode_session_counts": dict(mode_session_counts),
        "decoder_mode_frame_counts": dict(mode_frame_counts),
        "imputed_tail_frames": total_imputed,
        "tail_truncated_frames": total_truncated,
        "skipped_short_frames": total_skipped,
        "unflagged_videos": len(plans) - flagged_videos,
        "flagged_videos": flagged_videos,
        "unflagged_sessions": len(expected_unflagged),
        "flagged_sessions": len(expected_sessions) - len(expected_unflagged),
        "unflagged_frames": unflagged_frames,
        "flagged_frames": flagged_frames,
        "axis_sign_indeterminate": axis_indeterminate,
        "bind_confidence": confidence_summary,
        "long_context_target_frames": long_context_targets,
        "long_context_target_hours_at_60hz": (
            long_context_targets / FRAMES_PER_HOUR_60HZ
        ),
        "weighted_long_context_fraction": (
            long_context_targets / total_source_frames if total_source_frames else 0.0
        ),
        "temporary_artifacts": temporary_artifacts,
    }
    return {
        "ok": not errors,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "deep_shards": deep_shards,
        "git_commit": _git_commit(repo),
        "paths": {
            "raw_root": str(raw_root),
            "chunk_index": str(chunk_index),
            "fetch_report": str(fetch_report),
            "mapped_root": str(mapped_root),
            "chunk_frames": str(chunk_frames),
            "feature_root": str(feature_root),
            "output_root": str(output_root),
            "completion_marker": str(completion_marker),
        },
        "expected": asdict(expectations),
        "observed": observed,
        "errors": errors,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/harvest60"))
    parser.add_argument("--chunk-index", type=Path,
                        default=Path("data/celeste_chunk_index.parquet"))
    parser.add_argument("--fetch-report", type=Path,
                        default=Path("data/harvest60/fetch60_report.jsonl"))
    parser.add_argument("--mapped-root", type=Path,
                        default=Path("data/mapped_full_60fps"))
    parser.add_argument("--chunk-frames", type=Path,
                        default=Path("data/mapped_full_60fps/chunk_frames.parquet"))
    parser.add_argument("--feature-root", type=Path,
                        default=Path("data/features_by_video"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/full_corpus_features"))
    parser.add_argument("--completion-marker", type=Path,
                        default=Path("data/.full_corpus_features_done"))
    parser.add_argument("--build-log", type=Path,
                        default=Path("build_logs/full_corpus_features/build.log"))
    parser.add_argument("--video-log-root", type=Path,
                        default=Path("build_logs/full_corpus_features/videos"))
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--deep-shards", action="store_true",
        help=("load all shard arrays, verify feature finiteness and exact mapped "
              "supervision; default validation reads only NPZ headers and IDs"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    corpus_roots = [args.feature_root.resolve(), args.output_root.resolve()]
    report_path = args.out.resolve()
    if any(report_path == root or root in report_path.parents for root in corpus_roots):
        raise SystemExit("--out must be outside the feature corpus")
    try:
        report = validate_full_corpus_features(
            raw_root=args.raw_root,
            chunk_index=args.chunk_index,
            fetch_report=args.fetch_report,
            mapped_root=args.mapped_root,
            chunk_frames=args.chunk_frames,
            feature_root=args.feature_root,
            output_root=args.output_root,
            completion_marker=args.completion_marker,
            build_log=args.build_log,
            video_log_root=args.video_log_root,
            deep_shards=args.deep_shards,
            repo=args.repo,
        )
    except Exception as error:  # noqa: BLE001 - preserve failed validation evidence
        report = {
            "ok": False,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "deep_shards": args.deep_shards,
            "git_commit": _git_commit(args.repo),
            "expected": asdict(FULL_211_EXPECTATIONS),
            "observed": {},
            "errors": [f"validator crashed: {type(error).__name__}: {error}"],
        }
    _write_report(args.out, report)
    print(json.dumps({
        "ok": report["ok"],
        "out": str(args.out),
        "errors": len(report["errors"]),
        "observed": report["observed"],
    }, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
