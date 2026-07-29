"""Decode, time-align, quality-score, and mask a reviewed wild input HUD.

Labels live on the source video's presentation-timestamp timeline.  Frame
indices are retained for efficient pairing, but PTS—not a container FPS stamp—
is the timing authority.  The decoder emits labels even when quality gates
fail; only the manifest's explicit ``admitted`` status permits training.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.schema import KEY_ORDER
from harvest.accept_layout_confidence import verify_confidence_override
from harvest.accept_wild_layout import verify_layout_acceptance
from harvest.accept_wild_offset import verify_offset_acceptance
from harvest.fetch_wild import (
    PTS_SIDECAR_VERSION,
    load_pts_evidence,
    sha256_file,
    summarize_pts,
)
from harvest.translucent_parser import CALIBRATION_METHOD, calibrate_threshold
from harvest.wild_boundaries import WildBoundaries
from harvest.wild_layout import CellSpec, WildLayout, rect_to_pixels


DECODE_VERSION = "madeleine.wild-decode.v1"
DECODE_COMPLETION_VERSION = "madeleine.wild-decode-complete.v1"
DECODE_COMPLETION_NAME = "decode_complete.json"
WILD_LABEL_SCHEMA = pa.schema([
    pa.field("video_frame_idx", pa.int64()),
    pa.field("pts_s", pa.float64()),
    *(pa.field(key, pa.bool_()) for key in KEY_ORDER),
    pa.field("gameplay_allowed", pa.bool_()),
])


@dataclass(frozen=True)
class QCPolicy:
    min_effective_fps: float = 50.0
    max_effective_fps: float = 61.0
    max_large_gap_fraction: float = 1e-4
    max_vfr_ratio_p99_p01: float = 1.10
    min_cell_minority_frames: int = 30
    min_cell_separation: float = 1.5
    max_action_transition_hz: float = 15.0
    min_layout_confidence: float = 0.80
    min_offset_confidence: float = 0.95


def _mean(gray: np.ndarray, rect: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = rect_to_pixels(rect, gray.shape[1], gray.shape[0])
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("layout produced an empty pixel sample")
    return float(crop.mean())


def cell_score(gray: np.ndarray, cell: CellSpec) -> float:
    score = _mean(gray, cell.sample_rect)
    if cell.decoder == "local_contrast":
        assert cell.reference_rect is not None
        score -= _mean(gray, cell.reference_rect)
    return score


def extract_scores(frame: np.ndarray, layout: WildLayout) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("frame must be uint8 BGR with shape [H,W,3]")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.asarray([cell_score(gray, cell) for cell in layout.cells], dtype=np.float32)


def mask_frame(frame: np.ndarray, layout: WildLayout, copy: bool = True) -> np.ndarray:
    """Zero every answer-key region before a frame can reach a model."""

    out = frame.copy() if copy else frame
    for rect in layout.mask_rects:
        x0, y0, x1, y1 = rect_to_pixels(rect, out.shape[1], out.shape[0])
        out[y0:y1, x0:x1] = 0
    return out


def mask_rects_in_gameplay(layout: WildLayout) -> list[tuple[float, float, float, float]]:
    """Intersect frame masks with gameplay crop and normalize to that crop."""

    gx, gy, gw, gh = layout.gameplay_rect
    transformed = []
    for mx, my, mw, mh in layout.mask_rects:
        x0, y0 = max(gx, mx), max(gy, my)
        x1, y1 = min(gx + gw, mx + mw), min(gy + gh, my + mh)
        if x1 <= x0 or y1 <= y0:
            continue
        transformed.append(((x0 - gx) / gw, (y0 - gy) / gh,
                            (x1 - x0) / gw, (y1 - y0) / gh))
    return transformed


def masked_resize(frame: np.ndarray, layout: WildLayout, size: int) -> np.ndarray:
    """Mask, crop reviewed gameplay, resize, then re-zero transformed masks."""

    if size <= 0:
        raise ValueError("size must be positive")
    masked = mask_frame(frame, layout)
    gx0, gy0, gx1, gy1 = rect_to_pixels(
        layout.gameplay_rect, masked.shape[1], masked.shape[0]
    )
    gameplay = masked[gy0:gy1, gx0:gx1]
    if gameplay.size == 0:
        raise ValueError("reviewed gameplay_rect produced an empty crop")
    small = cv2.resize(gameplay, (size, size), interpolation=cv2.INTER_AREA)
    for rect in mask_rects_in_gameplay(layout):
        x0, y0, x1, y1 = rect_to_pixels(rect, size, size)
        x0, y0 = max(0, x0 - 1), max(0, y0 - 1)
        x1, y1 = min(size, x1 + 1), min(size, y1 + 1)
        small[y0:y1, x0:x1] = 0
        if small[y0:y1, x0:x1].size:
            assert int(small[y0:y1, x0:x1].max()) == 0
    return small


def _cluster_separation(values: np.ndarray, threshold: float) -> tuple[float, int]:
    low, high = values[values < threshold], values[values >= threshold]
    minority = min(low.size, high.size)
    if minority == 0:
        return 0.0, 0
    low_med, high_med = float(np.median(low)), float(np.median(high))
    low_mad = float(np.median(np.abs(low - low_med)))
    high_mad = float(np.median(np.abs(high - high_med)))
    robust_scale = 1.4826 * (low_mad + high_mad) / 2.0
    # Quantization can make both clusters exactly constant; a one-luma floor
    # keeps the statistic finite without punishing a perfect binary overlay.
    return (high_med - low_med) / max(robust_scale, 1.0), int(minority)


def _transition_stats(states: np.ndarray, duration_s: float) -> dict[str, float | int]:
    transitions = int(np.count_nonzero(states[1:] != states[:-1]))
    if states.size < 2:
        single = 0
    else:
        edges = np.flatnonzero(np.r_[True, states[1:] != states[:-1], True])
        single = int(np.count_nonzero(np.diff(edges) == 1))
    return {
        "duty": float(states.mean()) if states.size else 0.0,
        "transitions": transitions,
        "transitions_hz": transitions / max(duration_s, 1e-9),
        "single_frame_runs": single,
    }


def apply_temporal_offset(
    observed: np.ndarray, source_indices: np.ndarray, pts_s: np.ndarray, offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map overlay frame i to gameplay frame i+offset and drop uncovered edges."""

    n = observed.shape[0]
    overlay_idx = np.arange(n, dtype=np.int64)
    game_idx = overlay_idx + int(offset)
    keep = (game_idx >= 0) & (game_idx < n)
    aligned = np.empty((int(np.count_nonzero(keep)), observed.shape[1]), dtype=bool)
    aligned_order = np.argsort(game_idx[keep])
    aligned[:] = observed[overlay_idx[keep][aligned_order]]
    selected_game = game_idx[keep][aligned_order]
    return aligned, source_indices[selected_game], pts_s[selected_game]


def _decode_states(scores: np.ndarray, layout: WildLayout) -> tuple[np.ndarray, list[dict[str, Any]]]:
    # The calibrator's float-Otsu partition already places each threshold at
    # the midpoint of the two cluster medians (see CALIBRATION_METHOD), so no
    # post-hoc recentering runs here.  The old uint8-Otsu + recentering pair
    # could freeze inside a tight released cluster's quantization halo when
    # the provisional split left that cluster's dominant value on the high
    # side (ofy37Fm6EgI bottom_grab: threshold 71.92, separation 0.17).
    thresholds = calibrate_threshold(scores)
    physical = np.zeros(scores.shape, dtype=bool)
    diagnostics = []
    for index, (cell, threshold) in enumerate(zip(layout.cells, thresholds, strict=True)):
        high = scores[:, index] >= threshold
        physical[:, index] = high if cell.pressed_polarity == "high" else ~high
        separation, minority = _cluster_separation(scores[:, index], float(threshold))
        diagnostics.append({
            "cell_id": cell.cell_id,
            "action": cell.action,
            "decoder": cell.decoder,
            "pressed_polarity": cell.pressed_polarity,
            "calibration_method": CALIBRATION_METHOD,
            "threshold": float(threshold),
            "score_min": float(np.min(scores[:, index])),
            "score_max": float(np.max(scores[:, index])),
            "cluster_separation": float(separation),
            "minority_frames": minority,
            "observed_duty": float(physical[:, index].mean()),
        })
    actions = np.zeros((scores.shape[0], len(KEY_ORDER)), dtype=bool)
    for action_index, action in enumerate(KEY_ORDER):
        columns = [i for i, cell in enumerate(layout.cells) if cell.action == action]
        actions[:, action_index] = np.any(physical[:, columns], axis=1)
    return actions, diagnostics


def _read_scores(
    video: Path,
    layout: WildLayout,
    start_index: int,
    count: int,
    target: np.ndarray,
) -> None:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open {video}")
    try:
        for index in range(start_index):
            if not capture.grab():
                raise ValueError(f"video ended while seeking to frame {start_index} at {index}")
        for row in range(count):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"video decode failed at frame {start_index + row}")
            target[row] = extract_scores(frame, layout)
    finally:
        capture.release()


def _load_precomputed_scan_scores(
    scan_report_path: Path,
    layout: WildLayout,
    fetch: dict[str, Any],
    pts_manifest: dict[str, Any],
    frame_count: int,
) -> tuple[np.memmap, dict[str, Any]]:
    """Reuse a complete physical scan only after exact geometry binding."""

    unsupported = [
        cell.cell_id
        for cell in layout.cells
        if cell.decoder != "luma" or cell.reference_rect is not None
    ]
    if unsupported:
        raise ValueError(
            "precomputed cell scans currently support only plain luma cells "
            "without reference_rect: " + ", ".join(unsupported)
        )

    scan_report_path = _regular_file(
        scan_report_path, "precomputed scan report"
    )
    report = json.loads(scan_report_path.read_text())
    if report.get("format_version") != "madeleine.wild-cell-activity-scan.v1":
        raise ValueError("unsupported precomputed cell scan report")
    if (
        report.get("video_id") != layout.video_id
        or report.get("source", {}).get("sha256") != fetch.get("sha256")
        or report.get("pts", {}).get("sha256") != pts_manifest.get("sha256")
        or int(report.get("source", {}).get("frames", -1)) != frame_count
        or report.get("human_reviewed") is not False
        or report.get("training_admitted") is not False
    ):
        raise ValueError("precomputed cell scan does not bind source/PTS/layout video")
    score_row = report.get("scores")
    if not isinstance(score_row, dict):
        raise ValueError("precomputed cell scan lacks scores binding")
    score_path = _safe_child(
        scan_report_path.parent, score_row.get("path"), "precomputed cell scores"
    )
    source_range = report.get("source", {}).get(
        "source_frame_range", [0, frame_count]
    )
    if (
        not isinstance(source_range, list)
        or len(source_range) != 2
        or any(not isinstance(value, int) for value in source_range)
        or source_range[0] < 0
        or source_range[1] <= source_range[0]
        or source_range[1] > frame_count
    ):
        raise ValueError("precomputed scan has an invalid source frame range")
    scanned_frames = source_range[1] - source_range[0]
    shape = score_row.get("shape")
    expected_shape = [scanned_frames, len(layout.cells)]
    if shape != expected_shape or score_row.get("dtype") != "float32":
        raise ValueError("precomputed cell score shape/dtype mismatch")
    if (
        score_path.stat().st_size != scanned_frames * len(layout.cells) * 4
        or sha256_file(score_path) != score_row.get("sha256")
    ):
        raise ValueError("precomputed cell score bytes differ from scan report")

    spec_row = report.get("spec")
    if not isinstance(spec_row, dict):
        raise ValueError("precomputed cell scan lacks spec binding")
    spec_path = _safe_child(
        scan_report_path.parent, spec_row.get("path"), "precomputed scan spec"
    )
    if sha256_file(spec_path) != spec_row.get("sha256"):
        raise ValueError("precomputed cell scan spec hash mismatch")
    spec = json.loads(spec_path.read_text())
    if (
        spec.get("video_id") != layout.video_id
        or spec.get("source_sha256") != fetch.get("sha256")
        or spec.get("pts_sha256") != pts_manifest.get("sha256")
    ):
        raise ValueError("precomputed scan spec has incompatible bindings")
    resolution = fetch.get("media", {}).get("resolution_wh")
    if spec.get("frame_size_wh") != resolution:
        raise ValueError("precomputed scan dimensions differ from target source")
    width, height = int(resolution[0]), int(resolution[1])
    spec_cells = spec.get("cells")
    if not isinstance(spec_cells, list) or len(spec_cells) != len(layout.cells):
        raise ValueError("precomputed scan cell count differs from target layout")
    for index, (scan_cell, layout_cell) in enumerate(
        zip(spec_cells, layout.cells, strict=True)
    ):
        x0, y0, x1, y1 = rect_to_pixels(layout_cell.sample_rect, width, height)
        expected_rect = [x0, y0, x1 - x0, y1 - y0]
        if (
            scan_cell.get("cell_id") != layout_cell.cell_id
            or scan_cell.get("pressed_polarity") != layout_cell.pressed_polarity
            or scan_cell.get("sample_rect_px") != expected_rect
            or (
                "decoder" in scan_cell
                and scan_cell.get("decoder") != layout_cell.decoder
            )
            or scan_cell.get("semantic_action_from_reference", layout_cell.action)
            != layout_cell.action
        ):
            raise ValueError(
                f"precomputed scan cell {index} differs from target layout"
            )
    scores = np.memmap(
        score_path,
        dtype=np.float32,
        mode="r",
        shape=(scanned_frames, len(layout.cells)),
    )
    provenance = {
        "kind": "hash_bound_full_cell_scan",
        "report_path": str(scan_report_path),
        "report_sha256": sha256_file(scan_report_path),
        "spec_path": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "scores_path": str(score_path),
        "scores_sha256": sha256_file(score_path),
        "shape": expected_shape,
        "source_frame_range": source_range,
    }
    return scores, provenance


def _atomic_json(path: Path, value: dict[str, Any], *, replace: bool) -> None:
    payload = (json.dumps(value, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        else:
            try:
                os.link(temporary, path)
                _fsync_directory(path.parent)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise ValueError(f"existing completion marker differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage_parquet(
    destination: Path,
    name: str,
    table: pa.Table,
    metadata: dict[bytes, bytes],
) -> Path:
    """Write and fsync a Parquet file without exposing its final name."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{name}.", suffix=".tmp", dir=destination
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(
            table.replace_schema_metadata(metadata), temporary, compression="zstd"
        )
        _fsync_file(temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _regular_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or not a regular file")
    return path


def _safe_child(directory: Path, name: Any, label: str) -> Path:
    value = str(name or "")
    if not value or Path(value).name != value:
        raise ValueError(f"{label} has an unsafe path")
    return _regular_file(directory / value, label)


def _existing_pts_evidence(
    fetch_path: Path,
    fetch: dict[str, Any],
    video: Path,
    pts_evidence_dir: str | Path | None,
) -> tuple[np.ndarray, dict[str, Any], Path, Path]:
    """Read a complete PTS sidecar without regenerating missing evidence."""

    directory = (
        Path(pts_evidence_dir) if pts_evidence_dir is not None else fetch_path.parent
    )
    manifest_path = _regular_file(directory / "frame_pts.json", "PTS manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != PTS_SIDECAR_VERSION:
        raise ValueError("unsupported PTS manifest version")
    if manifest.get("source_sha256") != fetch.get("sha256"):
        raise ValueError("PTS manifest belongs to a different source video")
    if manifest.get("source_file") != video.name:
        raise ValueError("PTS manifest source filename differs from fetch report")
    vector_path = _safe_child(directory, manifest.get("path"), "PTS vector")
    vector_hash = sha256_file(vector_path)
    if vector_hash != manifest.get("sha256"):
        raise ValueError("PTS vector hash differs from manifest")
    pts = np.load(vector_path, allow_pickle=False)
    if pts.ndim != 1 or int(manifest.get("frames", -1)) != pts.size:
        raise ValueError("PTS vector shape/count differs from manifest")
    summarize_pts(pts)
    declared = fetch.get("pts_sidecar")
    if declared is not None and (
        declared.get("sha256") != vector_hash
        or int(declared.get("frames", -1)) != pts.size
    ):
        raise ValueError("PTS evidence differs from fetch report")
    reported_frames = ((fetch.get("media") or {}).get("pts") or {}).get("frames")
    if reported_frames is not None and int(reported_frames) != pts.size:
        raise ValueError("PTS evidence length differs from media audit")
    return pts.astype(np.float64, copy=False), manifest, manifest_path, vector_path


def _validate_label_table(
    path: Path,
    *,
    expected_rows: int | None,
    pts: np.ndarray,
) -> pa.Table:
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise ValueError(f"decode label artifact is not readable Parquet: {path.name}") from exc
    if not table.schema.remove_metadata().equals(WILD_LABEL_SCHEMA):
        raise ValueError(f"decode label schema differs: {path.name}")
    if expected_rows is not None and table.num_rows != expected_rows:
        raise ValueError(f"decode label row count differs: {path.name}")
    if table.num_rows <= 0:
        raise ValueError(f"decode label artifact is empty: {path.name}")
    indices = table.column("video_frame_idx").to_numpy(zero_copy_only=False)
    if (
        indices.ndim != 1
        or np.any(indices < 0)
        or np.any(indices >= pts.size)
        or not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1))
    ):
        raise ValueError(f"decode label frame indices are invalid: {path.name}")
    relative_pts = pts - pts[0]
    stored_pts = table.column("pts_s").to_numpy(zero_copy_only=False)
    if not np.allclose(stored_pts, relative_pts[indices], rtol=0.0, atol=1e-9):
        raise ValueError(f"decode label PTS values differ from evidence: {path.name}")
    return table


def _completion_value(
    destination: Path,
    report: dict[str, Any],
    *,
    fetch_path: Path,
    source_path: Path,
    layout_path: Path,
    boundaries_path: Path,
    pts_manifest_path: Path,
    pts_vector_path: Path,
    precomputed_scan_report_path: Path | None,
) -> dict[str, Any]:
    report_path = destination / "decode_report.json"
    artifacts: dict[str, dict[str, Any]] = {}
    for key, hash_key in (
        ("raw_labels", "raw_labels_sha256"),
        ("labels", "labels_sha256"),
    ):
        path = _safe_child(destination, report.get(key), f"decode {key}")
        digest = sha256_file(path)
        if digest != report.get(hash_key):
            raise ValueError(f"decode {key} bytes differ from report")
        artifacts[key] = {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        }
    return {
        "format_version": DECODE_COMPLETION_VERSION,
        "video_id": report["video_id"],
        "report": {
            "path": report_path.name,
            "size_bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
        },
        "artifacts": artifacts,
        "bindings": {
            "fetch_report_sha256": sha256_file(fetch_path),
            "source_sha256": report["source_video"]["sha256"],
            "layout_sha256": sha256_file(layout_path),
            "boundaries_sha256": sha256_file(boundaries_path),
            "pts_manifest_sha256": sha256_file(pts_manifest_path),
            "pts_sha256": sha256_file(pts_vector_path),
            "scan_report_sha256": (
                sha256_file(precomputed_scan_report_path)
                if precomputed_scan_report_path is not None
                else None
            ),
        },
        "source": {
            "path": source_path.name,
            "size_bytes": source_path.stat().st_size,
        },
        "admitted": bool(report.get("admitted")),
    }


def validate_decode_output(
    out_dir: str | Path,
    fetch_report_path: str | Path,
    layout_path: str | Path,
    boundaries_path: str | Path,
    *,
    pts_evidence_dir: str | Path | None = None,
    precomputed_scan_report_path: str | Path | None = None,
    backfill_completion: bool = False,
    verify_source_hash: bool = True,
) -> dict[str, Any]:
    """Validate an immutable decode chain and optionally add its marker.

    This is also the only safe resume predicate used by the worker. A bare
    ``decode_report.json`` is insufficient because a crash may leave it beside
    partial or changed Parquet files.
    """

    if backfill_completion and not verify_source_hash:
        raise ValueError("completion backfill requires source-byte verification")
    destination = Path(out_dir)
    report_path = _regular_file(
        destination / "decode_report.json", "decode report"
    )
    report = json.loads(report_path.read_text())
    fetch_path = _regular_file(Path(fetch_report_path), "fetch report")
    layout_file = _regular_file(Path(layout_path), "layout")
    boundaries_file = _regular_file(Path(boundaries_path), "boundaries")
    fetch = json.loads(fetch_path.read_text())
    layout = WildLayout.load(layout_file)
    boundaries = WildBoundaries.load(boundaries_file)
    video = _safe_child(fetch_path.parent, fetch.get("source_file"), "source video")
    if verify_source_hash and sha256_file(video) != fetch.get("sha256"):
        raise ValueError("source video hash does not match fetch report")
    pts, pts_manifest, pts_manifest_path, pts_vector_path = _existing_pts_evidence(
        fetch_path, fetch, video, pts_evidence_dir
    )
    if (
        report.get("format_version") != DECODE_VERSION
        or report.get("video_id") != layout.video_id
        or report.get("video_id") != fetch.get("video_id")
        or boundaries.video_id != layout.video_id
        or report.get("source_video", {}).get("sha256") != fetch.get("sha256")
        or report.get("layout", {}).get("sha256") != sha256_file(layout_file)
        or report.get("boundaries", {}).get("sha256")
        != sha256_file(boundaries_file)
    ):
        raise ValueError("decode report bindings differ from requested inputs")
    timing_evidence = (report.get("timing") or {}).get("pts_evidence") or {}
    if (
        timing_evidence.get("sha256") != pts_manifest.get("sha256")
        or int(timing_evidence.get("frames", -1)) != pts.size
    ):
        raise ValueError("decode report PTS binding differs from evidence")
    score_source = report.get("score_source")
    if precomputed_scan_report_path is not None:
        scan_path = _regular_file(
            Path(precomputed_scan_report_path), "precomputed scan report"
        )
        _, scan_provenance = _load_precomputed_scan_scores(
            scan_path, layout, fetch, pts_manifest, int(pts.size)
        )
        if (
            not isinstance(score_source, dict)
            or score_source.get("kind") != "hash_bound_full_cell_scan"
            or any(
                score_source.get(key) != scan_provenance[key]
                for key in (
                    "report_sha256",
                    "spec_sha256",
                    "scores_sha256",
                    "shape",
                    "source_frame_range",
                )
            )
        ):
            raise ValueError("decode report does not bind requested cell scan")
    elif isinstance(score_source, dict) and score_source.get("kind") == (
        "hash_bound_full_cell_scan"
    ):
        raise ValueError("precomputed scan path is required to validate this decode")

    label_tables: dict[str, pa.Table] = {}
    for key, hash_key in (
        ("raw_labels", "raw_labels_sha256"),
        ("labels", "labels_sha256"),
    ):
        path = _safe_child(destination, report.get(key), f"decode {key}")
        if sha256_file(path) != report.get(hash_key):
            raise ValueError(f"decode {key} bytes differ from report")
        expected_rows = int(report.get("decoded_frames", -1)) if key == "labels" else None
        label_tables[key] = _validate_label_table(
            path, expected_rows=expected_rows, pts=pts
        )
    native = label_tables["labels"]
    native_indices = native.column("video_frame_idx").to_numpy(zero_copy_only=False)
    source_range = report.get("source_video", {}).get("source_frame_range")
    if source_range != [int(native_indices[0]), int(native_indices[-1]) + 1]:
        raise ValueError("decode report source frame range differs from labels")
    allowed = native.column("gameplay_allowed").to_numpy(zero_copy_only=False)
    if int(report.get("gameplay_allowed_frames", -1)) != int(np.count_nonzero(allowed)):
        raise ValueError("decode report gameplay frame count differs from labels")

    scan_path_or_none = (
        Path(precomputed_scan_report_path)
        if precomputed_scan_report_path is not None
        else None
    )
    completion = _completion_value(
        destination,
        report,
        fetch_path=fetch_path,
        source_path=video,
        layout_path=layout_file,
        boundaries_path=boundaries_file,
        pts_manifest_path=pts_manifest_path,
        pts_vector_path=pts_vector_path,
        precomputed_scan_report_path=scan_path_or_none,
    )
    completion_path = destination / DECODE_COMPLETION_NAME
    if backfill_completion:
        _atomic_json(completion_path, completion, replace=False)
    _regular_file(completion_path, "decode completion marker")
    observed = json.loads(completion_path.read_text())
    if observed != completion:
        raise ValueError("decode completion marker differs from current bytes")
    return completion


def decode_video(
    fetch_report_path: str | Path,
    layout_path: str | Path,
    boundaries_path: str | Path,
    out_dir: str | Path,
    *,
    pts_evidence_dir: str | Path | None = None,
    layout_acceptance_path: str | Path | None = None,
    offset_acceptance_path: str | Path | None = None,
    confidence_override_path: str | Path | None = None,
    precomputed_scan_report_path: str | Path | None = None,
    policy: QCPolicy = QCPolicy(),
    verify_source_hash: bool = True,
) -> dict[str, Any]:
    fetch_path = Path(fetch_report_path)
    fetch = json.loads(fetch_path.read_text())
    layout_file = Path(layout_path)
    layout = WildLayout.load(layout_file)
    if layout.video_id != fetch.get("video_id"):
        raise ValueError("layout video_id does not match fetch report")
    layout_acceptance: dict[str, Any] | None = None
    if layout_acceptance_path is not None:
        layout_acceptance = verify_layout_acceptance(
            layout_file,
            layout,
            layout_acceptance_path,
            source_sha256=str(fetch.get("sha256", "")),
            allow_timing_derivative=layout.temporal_offset_source != "unmeasured",
        )
    offset_acceptance: dict[str, Any] | None = None
    if layout.temporal_offset_source == "unmeasured":
        if offset_acceptance_path is not None:
            raise ValueError("an unmeasured layout may not claim an offset acceptance")
    else:
        if offset_acceptance_path is None:
            raise ValueError(
                "a measured HUD compositor offset requires a verified offset acceptance artifact"
            )
        if layout_acceptance_path is None:
            raise ValueError(
                "a measured layout requires a verified hash-bound layout acceptance artifact"
            )
        offset_acceptance = verify_offset_acceptance(
            layout_file,
            layout,
            offset_acceptance_path,
            source_sha256=str(fetch.get("sha256", "")),
            layout_acceptance_path=layout_acceptance_path,
        )
    confidence_override: dict[str, Any] | None = None
    if confidence_override_path is not None:
        if layout_acceptance is None:
            raise ValueError(
                "a layout-confidence override requires a verified hash-bound layout acceptance artifact"
            )
        confidence_override = verify_confidence_override(
            layout_file,
            layout,
            confidence_override_path,
            layout_acceptance=layout_acceptance,
            source_sha256=str(fetch.get("sha256", "")),
            min_layout_confidence=policy.min_layout_confidence,
        )
    boundaries_file = Path(boundaries_path)
    boundaries = WildBoundaries.load(boundaries_file)
    if boundaries.video_id != layout.video_id:
        raise ValueError("boundaries video_id does not match layout/fetch report")
    if boundaries.source_sha256 != fetch.get("sha256"):
        raise ValueError("boundaries source hash does not match fetch report")
    video = fetch_path.parent / fetch["source_file"]
    if verify_source_hash and sha256_file(video) != fetch.get("sha256"):
        raise ValueError("source video hash does not match fetch report")

    all_pts, pts_manifest = load_pts_evidence(
        fetch_path, fetch, video, evidence_dir=pts_evidence_dir
    )
    relative_pts = all_pts - all_pts[0]
    start_s, end_s = boundaries.wall_clock_range_s
    selected = np.flatnonzero((relative_pts >= start_s) & (relative_pts < end_s))
    if selected.size == 0:
        raise ValueError("reviewed wall-clock range contains no decoded frames")
    if not np.array_equal(selected, np.arange(selected[0], selected[-1] + 1)):
        raise ValueError("reviewed wall-clock frame selection is not contiguous")
    selected_pts = relative_pts[selected]
    pts_report = summarize_pts(selected_pts)

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    score_provenance: dict[str, Any]
    if precomputed_scan_report_path is not None:
        complete_scores, score_provenance = _load_precomputed_scan_scores(
            Path(precomputed_scan_report_path),
            layout,
            fetch,
            pts_manifest,
            int(all_pts.size),
        )
        scan_start, scan_end = score_provenance["source_frame_range"]
        if int(selected[0]) < scan_start or int(selected[-1]) >= scan_end:
            raise ValueError("decode boundaries leave the precomputed scan frame range")
        selected_scores = complete_scores[
            int(selected[0]) - scan_start:int(selected[-1]) + 1 - scan_start
        ]
        observed, cell_qc = _decode_states(np.asarray(selected_scores), layout)
    else:
        with tempfile.TemporaryDirectory(prefix="wild-scores-", dir=destination) as temp:
            score_path = Path(temp) / "scores.f32"
            scores = np.memmap(
                score_path, dtype=np.float32, mode="w+",
                shape=(selected.size, len(layout.cells)),
            )
            _read_scores(video, layout, int(selected[0]), int(selected.size), scores)
            observed, cell_qc = _decode_states(np.asarray(scores), layout)
        score_provenance = {"kind": "decoded_from_source_video"}

    destination_metadata = {
        b"label_source": b"wild_overlay",
        b"timing_authority": b"presentation_timestamp",
        b"effective_grid_hz": str(pts_report["effective_fps"]).encode("ascii"),
    }
    raw_allowed = boundaries.gameplay_mask(selected_pts)
    raw_labels_path = destination / "labels_raw.parquet"
    raw_columns: dict[str, Any] = {
        "video_frame_idx": selected.astype(np.int64),
        "pts_s": selected_pts,
        **{key: observed[:, i] for i, key in enumerate(KEY_ORDER)},
        "gameplay_allowed": raw_allowed,
    }
    raw_table = pa.Table.from_pydict(raw_columns, schema=WILD_LABEL_SCHEMA)

    actions, source_indices, aligned_pts = apply_temporal_offset(
        observed, selected.astype(np.int64), selected_pts,
        layout.temporal_offset_frames,
    )
    duration = max(float(aligned_pts[-1] - aligned_pts[0]), 1e-9)
    action_qc = {
        action: _transition_stats(actions[:, index], duration)
        for index, action in enumerate(KEY_ORDER)
    }
    labels_path = destination / "labels_native.parquet"
    columns: dict[str, Any] = {
        "video_frame_idx": source_indices,
        "pts_s": aligned_pts,
    }
    columns.update({key: actions[:, i] for i, key in enumerate(KEY_ORDER)})
    aligned_allowed = boundaries.gameplay_mask(aligned_pts)
    columns["gameplay_allowed"] = aligned_allowed
    table = pa.Table.from_pydict(columns, schema=WILD_LABEL_SCHEMA)

    reasons: list[str] = []
    fps = float(pts_report["effective_fps"])
    if not policy.min_effective_fps <= fps <= policy.max_effective_fps:
        reasons.append(f"effective_fps {fps:.4f} outside admitted range")
    if pts_report["nonmonotonic_intervals"]:
        reasons.append("non-monotonic presentation timestamps")
    intervals = max(1, int(pts_report["frames"]) - 1)
    if pts_report["large_gap_intervals"] / intervals > policy.max_large_gap_fraction:
        reasons.append("too many large PTS gaps")
    if pts_report["vfr_ratio_p99_p01"] > policy.max_vfr_ratio_p99_p01:
        reasons.append("presentation-timestamp cadence is too variable")
    if not boundaries.human_reviewed:
        reasons.append("gameplay boundaries were not reviewed by a human")
    if layout_acceptance is None:
        reasons.append("layout lacks a verified hash-bound review acceptance")
    elif not layout_acceptance["human_reviewed"]:
        reasons.append("layout acceptance was not reviewed by a human")
    if (
        layout.inference_confidence < policy.min_layout_confidence
        and confidence_override is None
    ):
        reasons.append("layout inference confidence below admission threshold")
    if layout.gameplay_rect_confidence < policy.min_layout_confidence:
        reasons.append("gameplay crop confidence below admission threshold")
    if layout.temporal_offset_source == "unmeasured":
        reasons.append("HUD compositor offset is unmeasured")
    if layout.temporal_offset_confidence < policy.min_offset_confidence:
        reasons.append("HUD compositor offset confidence below admission threshold")
    if offset_acceptance is not None and not offset_acceptance["human_reviewed"]:
        reasons.append("HUD compositor offset acceptance was not human-reviewed")
    for diagnostic in cell_qc:
        if diagnostic["minority_frames"] < policy.min_cell_minority_frames:
            reasons.append(f"cell {diagnostic['cell_id']} did not show both states often enough")
        if diagnostic["cluster_separation"] < policy.min_cell_separation:
            reasons.append(f"cell {diagnostic['cell_id']} has weak state separation")
    for action, diagnostic in action_qc.items():
        if diagnostic["transitions_hz"] > policy.max_action_transition_hz:
            reasons.append(f"action {action} toggles implausibly fast")
    admitted = not reasons
    gameplay_allowed_hours = float(np.count_nonzero(aligned_allowed)) / fps / 3600.0

    report = {
        "format_version": DECODE_VERSION,
        "video_id": layout.video_id,
        "source_video": {
            "path": str(video),
            "sha256": fetch["sha256"],
            "source_frame_range": [int(source_indices[0]), int(source_indices[-1]) + 1],
        },
        "boundaries": {
            "path": str(boundaries_file),
            "sha256": sha256_file(boundaries_file),
            "wall_clock_range_s": list(boundaries.wall_clock_range_s),
            "policy_mode": boundaries.policy_mode,
            "ranges_s": [list(value) for value in boundaries.ranges_s],
            "reviewer": boundaries.reviewer,
            "reviewer_kind": boundaries.reviewer_kind,
            "human_reviewed": boundaries.human_reviewed,
        },
        "layout": {
            "path": str(layout_file),
            "sha256": sha256_file(layout_file),
            "overlay_style": layout.overlay_style,
            "gameplay_rect": list(layout.gameplay_rect),
            "gameplay_rect_source": layout.gameplay_rect_source,
            "gameplay_rect_confidence": layout.gameplay_rect_confidence,
            "mask_rects": [list(rect) for rect in layout.mask_rects],
            "inference_source": layout.inference_source,
            "inference_confidence": layout.inference_confidence,
            "human_reviewed": layout.human_reviewed,
            "review_acceptance": layout_acceptance,
        },
        "timing": {
            "authority": "presentation_timestamp",
            "pts": pts_report,
            "pts_evidence": {
                "manifest": str(
                    (Path(pts_evidence_dir) if pts_evidence_dir is not None else fetch_path.parent)
                    / "frame_pts.json"
                ),
                "sha256": pts_manifest["sha256"],
                "frames": pts_manifest["frames"],
            },
            "temporal_offset_frames": layout.temporal_offset_frames,
            "temporal_offset_source": layout.temporal_offset_source,
            "temporal_offset_confidence": layout.temporal_offset_confidence,
            "offset_acceptance": offset_acceptance,
        },
        "score_source": score_provenance,
        "decoded_frames": int(actions.shape[0]),
        "decoded_hours": duration / 3600.0,
        "gameplay_allowed_frames": int(np.count_nonzero(aligned_allowed)),
        "gameplay_allowed_hours": gameplay_allowed_hours,
        "admitted_hours": gameplay_allowed_hours if admitted else 0.0,
        "raw_labels": raw_labels_path.name,
        "labels": labels_path.name,
        "cell_qc": cell_qc,
        "action_qc": action_qc,
        "qc_policy": asdict(policy),
        "admitted": admitted,
        "rejection_reasons": sorted(set(reasons)),
    }
    if confidence_override is not None:
        # The gate did fire; the report must show it was overridden by a
        # recorded human ruling, never that it passed.
        report["layout"]["confidence_override"] = confidence_override
        report["admission_overrides"] = [{
            "gate": "layout inference confidence below admission threshold",
            "outcome": "overridden_by_recorded_human_ruling",
            "inference_confidence": layout.inference_confidence,
            "min_layout_confidence": policy.min_layout_confidence,
            "override_sha256": confidence_override["sha256"],
            "reviewer_identity": confidence_override["reviewer_identity"],
            "reviewer_kind": confidence_override["reviewer_kind"],
            "rationale": confidence_override["rationale"],
        }]
    raw_staged = _stage_parquet(
        destination, raw_labels_path.name, raw_table, destination_metadata
    )
    labels_staged: Path | None = None
    try:
        labels_staged = _stage_parquet(
            destination, labels_path.name, table, destination_metadata
        )
        report["raw_labels_sha256"] = sha256_file(raw_staged)
        report["labels_sha256"] = sha256_file(labels_staged)

        # Only expose final names once both complete Parquets have been staged
        # and fsynced.  A crash between replacements invalidates any older
        # completion marker because that marker binds both artifact hashes.
        os.replace(raw_staged, raw_labels_path)
        os.replace(labels_staged, labels_path)
        _fsync_directory(destination)

        report_path = destination / "decode_report.json"
        _atomic_json(report_path, report, replace=True)
        evidence_directory = (
            Path(pts_evidence_dir)
            if pts_evidence_dir is not None
            else fetch_path.parent
        )
        pts_manifest_path = _regular_file(
            evidence_directory / "frame_pts.json", "PTS manifest"
        )
        pts_vector_path = _safe_child(
            evidence_directory, pts_manifest.get("path"), "PTS vector"
        )
        completion = _completion_value(
            destination,
            report,
            fetch_path=fetch_path,
            source_path=video,
            layout_path=layout_file,
            boundaries_path=boundaries_file,
            pts_manifest_path=pts_manifest_path,
            pts_vector_path=pts_vector_path,
            precomputed_scan_report_path=(
                Path(precomputed_scan_report_path)
                if precomputed_scan_report_path is not None
                else None
            ),
        )
        _atomic_json(
            destination / DECODE_COMPLETION_NAME, completion, replace=True
        )
    finally:
        raw_staged.unlink(missing_ok=True)
        if labels_staged is not None:
            labels_staged.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-report", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--pts-evidence-dir", type=Path)
    parser.add_argument("--layout-acceptance", type=Path)
    parser.add_argument("--offset-acceptance", type=Path)
    parser.add_argument("--confidence-override", type=Path)
    parser.add_argument("--precomputed-scan-report", type=Path)
    parser.add_argument(
        "--min-effective-fps", type=float, default=QCPolicy().min_effective_fps
    )
    parser.add_argument(
        "--max-effective-fps", type=float, default=QCPolicy().max_effective_fps
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-verify-source-hash", action="store_true")
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="validate an existing decode and its completion marker instead of decoding",
    )
    parser.add_argument(
        "--backfill-completion",
        action="store_true",
        help="after full existing-chain validation, create a missing completion marker",
    )
    args = parser.parse_args()
    if args.backfill_completion and args.no_verify_source_hash:
        parser.error("--backfill-completion requires source-byte verification")
    if args.validate_existing or args.backfill_completion:
        completion = validate_decode_output(
            args.out,
            args.fetch_report,
            args.layout,
            args.boundaries,
            pts_evidence_dir=args.pts_evidence_dir,
            precomputed_scan_report_path=args.precomputed_scan_report,
            backfill_completion=args.backfill_completion,
            verify_source_hash=not args.no_verify_source_hash,
        )
        print(json.dumps({
            "video_id": completion["video_id"],
            "decode_complete": True,
            "completion_sha256": sha256_file(args.out / DECODE_COMPLETION_NAME),
        }, indent=2))
        return
    report = decode_video(
        args.fetch_report, args.layout, args.boundaries, args.out,
        pts_evidence_dir=args.pts_evidence_dir,
        layout_acceptance_path=args.layout_acceptance,
        offset_acceptance_path=args.offset_acceptance,
        confidence_override_path=args.confidence_override,
        precomputed_scan_report_path=args.precomputed_scan_report,
        policy=QCPolicy(
            min_effective_fps=args.min_effective_fps,
            max_effective_fps=args.max_effective_fps,
        ),
        verify_source_hash=not args.no_verify_source_hash,
    )
    print(json.dumps({
        "video_id": report["video_id"],
        "decoded_hours": report["decoded_hours"],
        "admitted": report["admitted"],
        "rejection_reasons": report["rejection_reasons"],
    }, indent=2))


if __name__ == "__main__":
    main()
