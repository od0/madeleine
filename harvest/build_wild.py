"""Build masked, train-ready shards from admitted wild-overlay labels.

Long no-input stretches are not silently called gameplay.  A frame is active
only when it lies near an observed input; contiguous active runs become shard
parts, so model windows cannot bridge a cutscene or a long paused/menu span.
The source video remains untouched and the report records both decoded and
train-ready hours.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import os
import json
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import cv2
import numpy as np
import pyarrow.parquet as pq

from data.schema import KEY_ORDER
from harvest.decode_wild import masked_resize, mask_rects_in_gameplay
from harvest.fetch_wild import sha256_file
from harvest.wild_layout import WildLayout


BUILD_VERSION = "madeleine.wild-shards.v1"
PROVISIONAL_BUILD_VERSION = "madeleine.wild-provisional-shards.v1"
MIN_RUN_FRAMES = 240
MAX_PART_FRAMES = 9_000  # 2.5 minutes at 60 Hz; bounds peak RAM per worker

PROVISIONAL_ONLY_REJECTION_REASONS = frozenset({
    "HUD compositor offset confidence below admission threshold",
    "HUD compositor offset is unmeasured",
    "gameplay boundaries were not reviewed by a human",
    "layout inference confidence below admission threshold",
    "layout lacks a verified hash-bound review acceptance",
})
PROVISIONAL_MIN_CELL_SEPARATION = 20.0
PROVISIONAL_MAX_SINGLE_FRAME_RUN_FRACTION = 0.05
PROVISIONAL_NATIVE30_MAX_SINGLE_FRAME_RUN_FRACTION = 0.10
PROVISIONAL_NATIVE24_MAX_SINGLE_FRAME_RUN_FRACTION = 0.10
PROVISIONAL_MIN_LAYOUT_CONFIDENCE = 0.75
PART_COMPLETION_VERSION = "madeleine.wild-shard-part.v1"


def activity_mask(keys: np.ndarray, radius_frames: int) -> np.ndarray:
    """Dilate any observed input by a fixed temporal radius."""

    if keys.ndim != 2 or keys.shape[1] != len(KEY_ORDER):
        raise ValueError(f"keys must have shape [N,{len(KEY_ORDER)}]")
    if radius_frames < 0:
        raise ValueError("radius_frames must be non-negative")
    active = np.any(keys, axis=1).astype(np.uint8)
    if radius_frames and active.size:
        kernel = np.ones(2 * radius_frames + 1, dtype=np.int64)
        full = np.convolve(active, kernel, mode="full")
        active = (full[radius_frames : radius_frames + active.size] > 0).astype(np.uint8)
    return active.astype(bool)


def contiguous_true_runs(mask: np.ndarray, min_frames: int = MIN_RUN_FRAMES) -> list[tuple[int, int]]:
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    padded = np.r_[False, mask.astype(bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(start), int(end))
        for start, end in edges.reshape(-1, 2)
        if end - start >= min_frames
    ]


def _read_label_table(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(path)
    required = {"video_frame_idx", "pts_s", "gameplay_allowed", *KEY_ORDER}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"labels missing columns: {sorted(missing)}")
    indices = np.asarray(table["video_frame_idx"].to_pylist(), dtype=np.int64)
    pts = np.asarray(table["pts_s"].to_pylist(), dtype=np.float64)
    keys = np.stack([
        np.asarray(table[key].to_pylist(), dtype=np.uint8) for key in KEY_ORDER
    ], axis=1)
    gameplay_allowed = np.asarray(
        table["gameplay_allowed"].to_pylist(), dtype=bool
    )
    if indices.size == 0 or np.any(np.diff(indices) != 1):
        raise ValueError("wild labels must cover a dense source-frame run")
    if np.any(np.diff(pts) <= 0):
        raise ValueError("wild label PTS must be strictly increasing")
    return indices, pts, keys, gameplay_allowed


def _validate_low_dynamic_scan(
    decode: dict[str, Any], scan_validation_path: str | Path | None
) -> tuple[set[str], dict[str, Any] | None]:
    if scan_validation_path is None:
        return set(), None
    path = Path(scan_validation_path)
    validation = json.loads(path.read_text())
    score_source = decode.get("score_source")
    if (
        validation.get("format_version")
        != "madeleine.wild-layout-family-scan-validation.v1"
        or validation.get("video_id") != decode.get("video_id")
        or validation.get("validation_policy") not in {
            "absolute_luma_or_low_dynamic_binary_v1",
            (
                "absolute_luma_or_disjoint_stable_pressed_or_"
                "low_dynamic_binary_v2"
            ),
        }
        or not isinstance(score_source, dict)
        or score_source.get("kind") != "hash_bound_full_cell_scan"
        or validation.get("scan_report_sha256")
        != score_source.get("report_sha256")
        or validation.get("layout_sha256") != decode.get("layout", {}).get("sha256")
    ):
        raise ValueError("low-dynamic scan validation does not bind decode inputs")
    rows = validation.get("cell_validation")
    if not isinstance(rows, list):
        raise ValueError("low-dynamic scan validation lacks per-cell evidence")
    allowed = set()
    for row in rows:
        mode = row.get("validation_mode")
        shared = (
            int(row.get("minority_frames", 0)) >= 1_000
            and float(row.get("single_frame_positive_run_fraction", 1.0)) <= 0.05
        )
        low_dynamic = (
            mode == "low_dynamic_binary"
            and float(row.get("absolute_gap_luma", 0.0)) >= 12.0
            and max(
                float(row.get("low_state_mad_luma", float("inf"))),
                float(row.get("high_state_mad_luma", float("inf"))),
            )
            <= 0.5
            and float(row.get("decoder_cluster_separation_floor1", 0.0)) >= 12.0
            and shared
        )
        disjoint_stable_pressed = (
            mode == "disjoint_stable_pressed_state"
            and float(row.get("absolute_gap_luma", 0.0)) >= 20.0
            and float(row.get("inter_cluster_support_gap_luma", 0.0)) >= 64.0
            and float(row.get("pressed_state_mad_luma", float("inf"))) <= 0.5
            and float(row.get("pressed_state_range_luma", float("inf"))) <= 8.0
            and shared
        )
        if low_dynamic or disjoint_stable_pressed:
            allowed.add(str(row.get("cell_id")))
    summary = {
        "path": path.name,
        "sha256": sha256_file(path),
        "policy": validation["validation_policy"],
        "allowed_cells": sorted(allowed),
    }
    return allowed, summary


def _validate_provisional_decode(
    decode: dict[str, Any],
    *,
    low_dynamic_cells: set[str] | None = None,
) -> dict[str, Any]:
    """Require strong mechanical QC while preserving every unresolved gate."""

    if decode.get("admitted") is not False:
        raise ValueError("provisional mode requires an explicitly unadmitted decode")
    reasons = decode.get("rejection_reasons")
    if not isinstance(reasons, list) or not reasons:
        raise ValueError("provisional decode must list its admission rejections")
    unexpected = set(str(reason) for reason in reasons) - PROVISIONAL_ONLY_REJECTION_REASONS
    if unexpected:
        raise ValueError(
            "provisional decode has non-provenance QC rejection(s): "
            + ", ".join(sorted(unexpected))
        )

    layout = decode.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("provisional decode lacks layout provenance")
    layout_confidence = float(layout.get("inference_confidence", 0.0))
    if layout_confidence < PROVISIONAL_MIN_LAYOUT_CONFIDENCE:
        raise ValueError(
            "provisional layout confidence "
            f"{layout_confidence:.3f} < {PROVISIONAL_MIN_LAYOUT_CONFIDENCE:.3f}"
        )

    pts = decode.get("timing", {}).get("pts", {})
    if pts.get("nonmonotonic_intervals") != 0 or pts.get("large_gap_intervals") != 0:
        raise ValueError("provisional decode timing is not contiguous and monotonic")

    cell_qc = decode.get("cell_qc")
    if not isinstance(cell_qc, list) or not cell_qc:
        raise ValueError("provisional decode lacks per-cell QC")
    low_dynamic_cells = low_dynamic_cells or set()
    weak_cells = [
        str(row.get("cell_id", "unknown"))
        for row in cell_qc
        if float(row.get("cluster_separation", 0.0)) < PROVISIONAL_MIN_CELL_SEPARATION
        and not (
            str(row.get("cell_id", "")) in low_dynamic_cells
            and float(row.get("cluster_separation", 0.0)) >= 12.0
        )
    ]
    if weak_cells:
        raise ValueError(
            "provisional decode has weak state separation: "
            + ", ".join(weak_cells)
        )

    effective_fps = float(decode.get("timing", {}).get("pts", {}).get(
        "effective_fps", 0.0
    ))
    if 50.0 <= effective_fps <= 61.0:
        cadence_tier = "native60"
        max_single_frame_fraction = PROVISIONAL_MAX_SINGLE_FRAME_RUN_FRACTION
        flicker_rationale = (
            "native-60 source: one-frame runs are ~16.7 ms; retain the original "
            "5% provisional flicker ceiling"
        )
    elif 29.0 <= effective_fps <= 31.0:
        cadence_tier = "native30"
        max_single_frame_fraction = (
            PROVISIONAL_NATIVE30_MAX_SINGLE_FRAME_RUN_FRACTION
        )
        flicker_rationale = (
            "verified native-29-31 Hz source: one-frame runs are ~33 ms and can "
            "represent legitimate taps; 10% is bounded by the mechanically "
            "clean, AI-only elDs reference decode observation (7.586%)"
        )
    elif 23.0 <= effective_fps <= 25.0:
        cadence_tier = "native24"
        max_single_frame_fraction = (
            PROVISIONAL_NATIVE24_MAX_SINGLE_FRAME_RUN_FRACTION
        )
        flicker_rationale = (
            "verified native-23-25 Hz source: one-frame runs are ~40-43 ms and "
            "can represent legitimate taps; retain the conservative 10% native-30 "
            "ceiling rather than extrapolating a looser threshold without a clean "
            "native-24 reference decode"
        )
    else:
        raise ValueError(
            f"provisional decode cadence {effective_fps:.6f} has no flicker policy"
        )

    action_qc = decode.get("action_qc")
    if not isinstance(action_qc, dict) or not action_qc:
        raise ValueError("provisional decode lacks per-action QC")
    flickery: list[str] = []
    fractions: dict[str, float] = {}
    for action, row in action_qc.items():
        transitions = int(row.get("transitions", 0))
        positive_runs_upper_bound = max(1.0, transitions / 2.0)
        fraction = float(row.get("single_frame_runs", 0)) / positive_runs_upper_bound
        fractions[str(action)] = fraction
        if fraction > max_single_frame_fraction:
            flickery.append(str(action))
    if flickery:
        raise ValueError(
            "provisional decode has excessive single-frame flicker: "
            + ", ".join(sorted(flickery))
        )
    return {
        "allowed_rejection_reasons": sorted(PROVISIONAL_ONLY_REJECTION_REASONS),
        "min_cell_separation": PROVISIONAL_MIN_CELL_SEPARATION,
        "min_layout_confidence": PROVISIONAL_MIN_LAYOUT_CONFIDENCE,
        "observed_layout_confidence": layout_confidence,
        "cadence_tier": cadence_tier,
        "effective_grid_hz": effective_fps,
        "max_single_frame_run_fraction": max_single_frame_fraction,
        "single_frame_run_policy_rationale": flicker_rationale,
        "observed_single_frame_run_fraction": dict(sorted(fractions.items())),
    }


def _part_completion_path(part_path: Path) -> Path:
    return part_path.with_name(part_path.name + ".complete.json")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Publish JSON only after its complete contents are durable."""

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
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _npy_member_header(
    archive: zipfile.ZipFile,
    member_name: str,
) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    with archive.open(member_name) as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(member)
        if version == (2, 0):
            return np.lib.format.read_array_header_2_0(member)
        raise ValueError(f"unsupported npy header version {version} in {member_name}")


def _part_bindings(
    *,
    implementation_sha256: str,
    decode: dict[str, Any],
) -> dict[str, str]:
    return {
        "implementation_sha256": implementation_sha256,
        "source_video_sha256": str(decode["source_video"]["sha256"]),
        "labels_sha256": str(decode["labels_sha256"]),
        "layout_sha256": str(decode["layout"]["sha256"]),
        "boundaries_sha256": str(decode["boundaries"]["sha256"]),
    }


def _write_part_atomic(
    *,
    part_path: Path,
    row_without_hash: dict[str, Any],
    bindings: dict[str, str],
    frames: np.ndarray,
    keys: np.ndarray,
    frame_indices: np.ndarray,
    pts_s: np.ndarray,
) -> dict[str, Any]:
    """Compress one part off the decoder thread, then atomically publish it."""

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=part_path.parent,
            prefix=f".{part_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                frames=frames,
                keys=keys,
                engine_frame_idx=frame_indices,
                pts_s=pts_s,
                input_active=np.ones(frame_indices.size, dtype=np.uint8),
                session_id=row_without_hash["session_id"],
            )
            handle.flush()
            os.fsync(handle.fileno())
        part_sha256 = sha256_file(temporary)
        part_bytes = temporary.stat().st_size
        os.replace(temporary, part_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    row = {**row_without_hash, "sha256": part_sha256}
    completion = {
        "format_version": PART_COMPLETION_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row": row,
        "npz_bytes": part_bytes,
        "bindings": bindings,
        "arrays": {
            "frames": {
                "shape": list(frames.shape),
                "dtype": frames.dtype.str,
            },
            "keys": {"shape": list(keys.shape), "dtype": keys.dtype.str},
            "engine_frame_idx": {
                "shape": list(frame_indices.shape),
                "dtype": frame_indices.dtype.str,
            },
            "pts_s": {"shape": list(pts_s.shape), "dtype": pts_s.dtype.str},
            "input_active": {
                "shape": [int(frame_indices.size)],
                "dtype": np.dtype(np.uint8).str,
            },
        },
    }
    _atomic_json(_part_completion_path(part_path), completion)
    return row


def _resume_part(
    *,
    part_path: Path,
    row_without_hash: dict[str, Any],
    bindings: dict[str, str],
    frame_size: int,
    expected_keys: np.ndarray,
    expected_indices: np.ndarray,
    expected_pts: np.ndarray,
) -> dict[str, Any] | None:
    """Return a verified existing row, or fail closed and request a rebuild."""

    completion_path = _part_completion_path(part_path)
    if not part_path.is_file() or not completion_path.is_file():
        return None
    try:
        completion = json.loads(completion_path.read_text())
        row = completion["row"]
        if completion.get("format_version") != PART_COMPLETION_VERSION:
            return None
        if completion.get("bindings") != bindings:
            return None
        if {key: row.get(key) for key in row_without_hash} != row_without_hash:
            return None
        actual_sha256 = sha256_file(part_path)
        if row.get("sha256") != actual_sha256:
            return None
        if int(completion.get("npz_bytes", -1)) != part_path.stat().st_size:
            return None

        expected_names = {
            "frames.npy",
            "keys.npy",
            "engine_frame_idx.npy",
            "pts_s.npy",
            "input_active.npy",
            "session_id.npy",
        }
        with zipfile.ZipFile(part_path) as archive:
            if set(archive.namelist()) != expected_names:
                return None
            shape, fortran, dtype = _npy_member_header(archive, "frames.npy")
            if (
                shape != (expected_indices.size, frame_size, frame_size, 3)
                or fortran
                or dtype != np.dtype(np.uint8)
            ):
                return None
        # These arrays are small and encode the exact label/frame/PTS pairing.
        # The large frame payload is bound by the archive hash, source hash,
        # layout hash, and exact builder implementation hash above.
        with np.load(part_path, allow_pickle=False) as stored:
            if not np.array_equal(stored["keys"], expected_keys):
                return None
            if not np.array_equal(stored["engine_frame_idx"], expected_indices):
                return None
            if not np.array_equal(stored["pts_s"], expected_pts):
                return None
            if not np.array_equal(
                stored["input_active"],
                np.ones(expected_indices.size, dtype=np.uint8),
            ):
                return None
            if str(stored["session_id"]) != str(row_without_hash["session_id"]):
                return None
        return {**row_without_hash, "sha256": actual_sha256}
    except (KeyError, OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError):
        return None


def build_wild_video(
    decode_report_path: str | Path,
    layout_path: str | Path,
    out_dir: str | Path,
    *,
    frame_size: int = 128,
    idle_context_s: float = 3.0,
    provisional: bool = False,
    workers: int = 1,
    resume: bool = False,
    scan_validation_path: str | Path | None = None,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least one")
    decode_path = Path(decode_report_path)
    decode = json.loads(decode_path.read_text())
    provisional_qc = None
    low_dynamic_cells, low_dynamic_summary = _validate_low_dynamic_scan(
        decode, scan_validation_path
    )
    if provisional:
        provisional_qc = _validate_provisional_decode(
            decode, low_dynamic_cells=low_dynamic_cells
        )
    elif not decode.get("admitted"):
        raise ValueError("decode report is not admitted; refusing to build training shards")
    layout_file = Path(layout_path)
    layout = WildLayout.load(layout_file)
    if layout.video_id != decode.get("video_id"):
        raise ValueError("layout and decode report video IDs differ")
    if sha256_file(layout_file) != decode["layout"]["sha256"]:
        raise ValueError("layout changed after decoding")
    labels_path = decode_path.parent / decode["labels"]
    if sha256_file(labels_path) != decode["labels_sha256"]:
        raise ValueError("labels changed after decoding")
    video_path = Path(decode["source_video"]["path"])
    if sha256_file(video_path) != decode["source_video"]["sha256"]:
        raise ValueError("source video changed after decoding")

    frame_indices, pts_s, keys, gameplay_allowed = _read_label_table(labels_path)
    fps = float(decode["timing"]["pts"]["effective_fps"])
    radius = int(round(idle_context_s * fps))
    # Reviewed ranges are a hard gate.  The action-radius heuristic is only an
    # additional activity filter and can never re-admit menu/load frames.
    active = activity_mask(keys, radius) & gameplay_allowed
    runs = contiguous_true_runs(active)
    if not runs:
        raise ValueError("no train-ready active run passed the minimum length")

    parts: list[tuple[int, int]] = []
    skipped_short = 0
    for run_start, run_end in runs:
        for start in range(run_start, run_end, MAX_PART_FRAMES):
            end = min(start + MAX_PART_FRAMES, run_end)
            if end - start < MIN_RUN_FRAMES:
                skipped_short += end - start
            else:
                parts.append((start, end))

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    implementation_sha256 = sha256_file(Path(__file__).resolve())
    bindings = _part_bindings(
        implementation_sha256=implementation_sha256,
        decode=decode,
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open {video_path}")
    cursor = 0
    rows_by_part: dict[int, dict[str, Any]] = {}
    pending: dict[Future[dict[str, Any]], int] = {}
    resumed_parts = 0

    def collect_finished() -> None:
        if not pending:
            return
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            part_number = pending.pop(future)
            rows_by_part[part_number] = future.result()

    try:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="wild-shard-writer",
        ) as executor:
            for part_number, (row_start, row_end) in enumerate(parts):
                source_start = int(frame_indices[row_start])
                source_end = int(frame_indices[row_end - 1]) + 1
                while cursor < source_start:
                    if not capture.grab():
                        raise ValueError(
                            f"video ended while seeking to source frame {source_start}"
                        )
                    cursor += 1
                prefix = "wild_provisional" if provisional else "wild"
                session_id = f"{prefix}_{layout.video_id}__r{part_number:03d}"
                part_path = destination / f"{session_id}.npz"
                row_without_hash = {
                    "session_id": session_id,
                    "npz": part_path.name,
                    "frames": row_end - row_start,
                    "source_frame_range": [source_start, source_end],
                    "pts_range_s": [
                        float(pts_s[row_start]),
                        float(pts_s[row_end - 1]),
                    ],
                }
                expected_keys = keys[row_start:row_end]
                expected_indices = frame_indices[row_start:row_end]
                expected_pts = pts_s[row_start:row_end]
                existing = None
                if resume:
                    existing = _resume_part(
                        part_path=part_path,
                        row_without_hash=row_without_hash,
                        bindings=bindings,
                        frame_size=frame_size,
                        expected_keys=expected_keys,
                        expected_indices=expected_indices,
                        expected_pts=expected_pts,
                    )
                if existing is not None:
                    while cursor < source_end:
                        if not capture.grab():
                            raise ValueError(
                                "video ended while advancing across verified part "
                                f"{part_number}"
                            )
                        cursor += 1
                    rows_by_part[part_number] = existing
                    resumed_parts += 1
                    continue

                frames = np.empty(
                    (row_end - row_start, frame_size, frame_size, 3), dtype=np.uint8
                )
                for local in range(row_end - row_start):
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"video decode failed at source frame {cursor}")
                    cursor += 1
                    small = masked_resize(frame, layout, frame_size)
                    frames[local] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                future = executor.submit(
                    _write_part_atomic,
                    part_path=part_path,
                    row_without_hash=row_without_hash,
                    bindings=bindings,
                    frames=frames,
                    keys=expected_keys,
                    frame_indices=expected_indices,
                    pts_s=expected_pts,
                )
                pending[future] = part_number
                # The number of live frame tensors is bounded by the writer
                # count, keeping RAM predictable even for 9,000-frame parts.
                if len(pending) >= workers:
                    collect_finished()
            while pending:
                collect_finished()
    finally:
        capture.release()

    rows = [rows_by_part[index] for index in range(len(parts))]
    train_frames = sum(row["frames"] for row in rows)
    train_hours = train_frames / fps / 3600.0
    report = {
        "format_version": (
            PROVISIONAL_BUILD_VERSION if provisional else BUILD_VERSION
        ),
        "video_id": layout.video_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_kind": (
            "wild_overlay_provisional" if provisional else "wild_overlay"
        ),
        "admission_tier": (
            "provisional_not_train_ready" if provisional else "admitted_train_ready"
        ),
        "timing_authority": "presentation_timestamp",
        "implementation": {
            "module": "harvest/build_wild.py",
            "sha256": implementation_sha256,
        },
        "workers": workers,
        "resume_enabled": resume,
        "resumed_parts": resumed_parts,
        "effective_grid_hz": fps,
        "decoded_frames": int(frame_indices.size),
        "decoded_hours": float(decode["decoded_hours"]),
        "train_ready_frames": 0 if provisional else int(train_frames),
        "train_ready_hours": 0.0 if provisional else train_hours,
        "provisional_trainable_frames": int(train_frames) if provisional else 0,
        "provisional_trainable_hours": train_hours if provisional else 0.0,
        "idle_context_s": idle_context_s,
        "activity_policy": "within idle_context_s of any decoded pressed action",
        "inputs": {
            "decode_report": {
                "path": decode_path.name,
                "sha256": sha256_file(decode_path),
            },
            "labels": {
                "path": labels_path.name,
                "sha256": decode["labels_sha256"],
            },
            "layout": {
                "path": layout_file.name,
                "sha256": decode["layout"]["sha256"],
            },
            "boundaries_sha256": decode["boundaries"]["sha256"],
            "source_video_sha256": decode["source_video"]["sha256"],
        },
        "reviewed_gameplay_frames": int(np.count_nonzero(gameplay_allowed)),
        "reviewed_gameplay_hours": float(np.count_nonzero(gameplay_allowed)) / fps / 3600.0,
        "excluded_by_reviewed_ranges": int(frame_indices.size - np.count_nonzero(gameplay_allowed)),
        "excluded_inactive_within_reviewed_ranges": int(
            np.count_nonzero(gameplay_allowed & ~active)
        ),
        "skipped_short_frames": skipped_short,
        "mask_rects": [list(rect) for rect in layout.mask_rects],
        "gameplay_crop": {
            "normalized_xywh": list(layout.gameplay_rect),
            "source": layout.gameplay_rect_source,
            "confidence": layout.gameplay_rect_confidence,
            "layout_sha256": decode["layout"]["sha256"],
            "mask_rects_in_crop": [
                list(rect) for rect in mask_rects_in_gameplay(layout)
            ],
        },
        "parts": rows,
    }
    if provisional:
        report["provisional_qc"] = provisional_qc
        if low_dynamic_summary is not None:
            report["inputs"]["scan_validation"] = low_dynamic_summary
        report["unresolved_admission_reasons"] = list(decode["rejection_reasons"])
        report["warning"] = (
            "Diagnostic/noisy-supervision shards only; timing, layout, and/or "
            "gameplay review remains unresolved. This is not train-ready data."
        )
    report_name = (
        "wild_provisional_build_report.json"
        if provisional
        else "wild_build_report.json"
    )
    _atomic_json(destination / report_name, report)
    return report


def update_corpus_manifest(out_dir: str | Path, video_report: dict[str, Any]) -> Path:
    if video_report.get("format_version") != BUILD_VERSION:
        raise ValueError("canonical corpus manifest accepts admitted build reports only")
    directory = Path(out_dir)
    manifest_path = directory / "wild_corpus_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "format_version": BUILD_VERSION,
        "videos": [],
    }
    videos = [row for row in old["videos"] if row["video_id"] != video_report["video_id"]]
    videos.append(video_report)
    videos.sort(key=lambda row: row["video_id"])
    old.update({
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "videos": videos,
        "video_count": len(videos),
        "decoded_hours": sum(float(row["decoded_hours"]) for row in videos),
        "train_ready_hours": sum(float(row["train_ready_hours"]) for row in videos),
    })
    manifest_path.write_text(json.dumps(old, indent=2) + "\n")
    return manifest_path


def update_provisional_corpus_manifest(
    out_dir: str | Path,
    video_report: dict[str, Any],
) -> Path:
    if video_report.get("format_version") != PROVISIONAL_BUILD_VERSION:
        raise ValueError("provisional corpus manifest requires a provisional build report")
    directory = Path(out_dir)
    manifest_path = directory / "wild_provisional_corpus_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "format_version": PROVISIONAL_BUILD_VERSION,
        "admission_tier": "provisional_not_train_ready",
        "videos": [],
    }
    videos = [
        row for row in old["videos"]
        if row["video_id"] != video_report["video_id"]
    ]
    videos.append(video_report)
    videos.sort(key=lambda row: row["video_id"])
    old.update({
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "videos": videos,
        "video_count": len(videos),
        "train_ready_hours": 0.0,
        "provisional_trainable_hours": sum(
            float(row["provisional_trainable_hours"]) for row in videos
        ),
        "warning": "This manifest is not an admitted Wild20 corpus.",
    })
    manifest_path.write_text(json.dumps(old, indent=2) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-report", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scan-validation", type=Path)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--idle-context-s", type=float, default=3.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "number of concurrent NPZ writers; source decode remains a single "
            "ordered stream"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only parts with valid hash-bound completion sidecars",
    )
    parser.add_argument(
        "--provisional",
        action="store_true",
        help=(
            "build separately marked noisy-supervision shards from an "
            "unadmitted but mechanically clean decode"
        ),
    )
    args = parser.parse_args()
    report = build_wild_video(
        args.decode_report, args.layout, args.out,
        frame_size=args.frame_size, idle_context_s=args.idle_context_s,
        provisional=args.provisional, workers=args.workers, resume=args.resume,
        scan_validation_path=args.scan_validation,
    )
    if args.provisional:
        manifest = update_provisional_corpus_manifest(args.out.parent, report)
    else:
        manifest = update_corpus_manifest(args.out.parent, report)
    print(json.dumps({
        "video_id": report["video_id"],
        "decoded_hours": report["decoded_hours"],
        "train_ready_hours": report["train_ready_hours"],
        "provisional_trainable_hours": report["provisional_trainable_hours"],
        "parts": len(report["parts"]),
        "corpus_manifest": str(manifest),
    }, indent=2))


if __name__ == "__main__":
    main()
