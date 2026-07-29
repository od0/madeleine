"""Validate one frozen-format session directory."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.schema import (
    ALIGNMENT_DECODE_STATUSES,
    ALIGNMENT_SCHEMA,
    FOREIGN_STREAMS,
    GRID_HZ_METADATA_KEY,
    KEY_ORDER,
    LABELS_NATIVE_SCHEMA,
    MANIFEST_REQUIRED_FIELDS,
    PROVENANCE_SOURCES,
    RECORDED_STREAMS,
    STREAM_FILE_KEYS,
    TRUTH_SCHEMA,
)


def _required_field_violations(
    value: Any, tree: Mapping[str, Any], prefix: str = "manifest"
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{prefix} must be an object"]
    violations: list[str] = []
    for name, children in tree.items():
        path = f"{prefix}.{name}"
        if name not in value:
            violations.append(f"{path} is required")
        elif children is not None:
            violations.extend(_required_field_violations(value[name], children, path))
    return violations


def _read_manifest(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.is_file():
        return None, ["manifest.json is required"]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, [f"manifest.json is not valid JSON: {exc}"]
    if not isinstance(value, dict):
        return None, ["manifest.json must contain an object"]
    return value, []


def _schema_violations(
    path: Path, expected: pa.Schema, display_name: str
) -> tuple[pa.Table | None, list[str]]:
    try:
        actual_schema = pq.read_schema(path)
    except Exception as exc:
        return None, [f"{display_name} cannot be read as Parquet: {exc}"]

    violations: list[str] = []
    same_columns_and_types = (
        actual_schema.names == expected.names
        and [field.type for field in actual_schema]
        == [field.type for field in expected]
    )
    if not same_columns_and_types:
        expected_description = ", ".join(
            f"{field.name}:{field.type}" for field in expected
        )
        actual_description = ", ".join(
            f"{field.name}:{field.type}" for field in actual_schema
        )
        violations.append(
            f"{display_name} schema must be [{expected_description}], "
            f"found [{actual_description}]"
        )
        return None, violations

    try:
        table = pq.read_table(path)
    except Exception as exc:
        return None, [f"{display_name} cannot be read as Parquet: {exc}"]

    for name in expected.names:
        null_count = table.column(name).null_count
        if null_count:
            violations.append(
                f"{display_name}.{name} contains {null_count} null value(s)"
            )
    if violations:
        return None, violations
    return table, violations


def _dense_index_violations(
    table: pa.Table, column_name: str, display_name: str, start_at_zero: bool = False
) -> list[str]:
    column = table.column(column_name).combine_chunks()
    if len(column) == 0:
        return [f"{display_name}.{column_name} must not be empty"]
    values = column.to_numpy(zero_copy_only=False)
    violations: list[str] = []
    if start_at_zero and int(values[0]) != 0:
        violations.append(f"{display_name}.{column_name} must start at 0")
    if len(values) > 1:
        bad = np.flatnonzero(np.diff(values) != 1)
        if len(bad):
            row = int(bad[0]) + 1
            violations.append(
                f"{display_name}.{column_name} must be dense and monotonic; "
                f"gap or reversal at row {row}"
            )
    return violations


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction(value: Any) -> Fraction | None:
    if not isinstance(value, str) or value in {"", "N/A", "0/0"}:
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _probe_video(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    command = [
        "ffprobe",
        "-v",
        "error",
        # Count packets, not frames: -count_frames decodes every frame (minutes
        # on a long clip); -count_packets is demux-only (seconds). For our CFR
        # H.264 one packet == one frame.
        "-count_packets",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate,nb_frames,nb_read_packets,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, [f"video.mkv could not be inspected with ffprobe: {exc}"]
    if result.returncode != 0:
        detail = result.stderr.strip().replace("\n", " ") or "unknown ffprobe error"
        return None, [f"video.mkv could not be inspected with ffprobe: {detail}"]
    try:
        probe = json.loads(result.stdout)
        stream = probe["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        return None, [f"video.mkv ffprobe output is incomplete: {exc}"]

    violations: list[str] = []
    rate = _fraction(stream.get("r_frame_rate"))
    average_rate = _fraction(stream.get("avg_frame_rate"))
    if rate is None or average_rate is None:
        violations.append("video.mkv frame rates are missing from ffprobe output")
    elif rate != average_rate:
        violations.append(
            "video.mkv must be CFR: r_frame_rate does not equal avg_frame_rate"
        )
    elif rate != Fraction(60, 1):
        violations.append(f"video.mkv frame rate must be 60 fps, found {rate}")

    frame_count_value = stream.get("nb_frames")
    if _number(frame_count_value) is None:
        frame_count_value = stream.get("nb_read_packets")
    frame_count_number = _number(frame_count_value)
    frame_count = (
        int(round(frame_count_number)) if frame_count_number is not None else None
    )
    if frame_count is None:
        violations.append("video.mkv frame count is missing from ffprobe output")

    duration = _number(stream.get("duration"))
    if duration is None:
        duration = _number(probe.get("format", {}).get("duration"))
    if duration is None:
        violations.append("video.mkv duration is missing from ffprobe output")

    if (
        rate is not None
        and frame_count is not None
        and duration is not None
        and abs(frame_count - duration * float(rate)) > 1.0
    ):
        violations.append(
            "video.mkv frame count is inconsistent with duration × fps by more "
            "than 1 frame"
        )

    return {
        "frame_count": frame_count,
        "duration": duration,
        "fps": float(rate) if rate is not None else None,
    }, violations


def _manifest_shape_violations(
    manifest: dict[str, Any], session_dir: Path
) -> list[str]:
    violations: list[str] = []
    if manifest["format_version"] != "1":
        violations.append("manifest.format_version must equal \"1\"")
    if (
        not isinstance(manifest["session_id"], str)
        or manifest["session_id"] != session_dir.name
    ):
        violations.append("manifest.session_id must match the session directory name")
    created_at = manifest["created_at"]
    if not isinstance(created_at, str) or not created_at:
        violations.append("manifest.created_at must be an ISO-8601 string")
    else:
        try:
            datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            violations.append("manifest.created_at must be an ISO-8601 string")

    regions = manifest["masked_regions"]
    if not isinstance(regions, list):
        violations.append("manifest.masked_regions must be a list")
    else:
        for index, region in enumerate(regions):
            prefix = f"manifest.masked_regions[{index}]"
            if not isinstance(region, dict):
                violations.append(f"{prefix} must be an object")
                continue
            for field in ("name", "space", "applied", "rect_px", "rect_norm"):
                if field not in region:
                    violations.append(f"{prefix}.{field} is required")
            for field in ("name", "space", "applied"):
                if field in region and (
                    not isinstance(region[field], str) or not region[field]
                ):
                    violations.append(f"{prefix}.{field} must be a non-empty string")
            rect_px = region.get("rect_px")
            if (
                not isinstance(rect_px, list)
                or len(rect_px) != 4
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or _number(value) is None
                    for value in rect_px
                )
                or (
                    len(rect_px) == 4
                    and (
                        rect_px[0] < 0
                        or rect_px[1] < 0
                        or rect_px[2] <= 0
                        or rect_px[3] <= 0
                    )
                )
            ):
                violations.append(
                    f"{prefix}.rect_px must be [x, y, width, height] with "
                    "non-negative origin and positive size"
                )
            rect_norm = region.get("rect_norm")
            if (
                not isinstance(rect_norm, list)
                or len(rect_norm) != 4
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or _number(value) is None
                    for value in rect_norm
                )
                or (
                    len(rect_norm) == 4
                    and not (
                        0 <= rect_norm[0] < rect_norm[2] <= 1
                        and 0 <= rect_norm[1] < rect_norm[3] <= 1
                    )
                )
            ):
                violations.append(
                    f"{prefix}.rect_norm must be normalized [x0, y0, x1, y1]"
                )

    if manifest["actions"]["keys"] != KEY_ORDER:
        violations.append(f"manifest.actions.keys must equal {KEY_ORDER}")
    source = manifest["provenance"]["source"]
    if not isinstance(source, str) or source not in PROVENANCE_SOURCES:
        violations.append(
            f"manifest.provenance.source must be one of {list(PROVENANCE_SOURCES)}"
        )
    if not isinstance(manifest["integrity"]["sha256"], dict):
        violations.append("manifest.integrity.sha256 must be an object")
    for field in ("video_frames", "duplicates", "drops"):
        value = manifest["integrity"][field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            violations.append(
                f"manifest.integrity.{field} must be a non-negative integer"
            )
    resolution = manifest["capture"]["resolution"]
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in resolution
        )
    ):
        violations.append(
            "manifest.capture.resolution must be [width, height] with positive integers"
        )
    for field in ("requested_fps", "achieved_fps"):
        raw_value = manifest["capture"][field]
        value = _number(raw_value)
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or value is None
            or value <= 0
        ):
            violations.append(
                f"manifest.capture.{field} must be a positive number"
            )
    for field in ("game", "everest", "mod"):
        if not isinstance(manifest["env"][field], str):
            violations.append(f"manifest.env.{field} must be a string")
    for field in ("tool", "encode"):
        if not isinstance(manifest["capture"][field], str):
            violations.append(f"manifest.capture.{field} must be a string")
    if manifest["streams"]["overlay_style"] not in (
        "none",
        "input-display",
        "nohboard",
    ):
        violations.append(
            "manifest.streams.overlay_style must be one of "
            "\"none\", \"input-display\", or \"nohboard\""
        )
    return violations


def validate_session(session_dir: str | Path) -> list[str]:
    """Return one string per format violation; an empty list means valid."""
    root = Path(session_dir)
    if not root.is_dir():
        return [f"session directory does not exist: {root}"]

    manifest, violations = _read_manifest(root / "manifest.json")
    if manifest is None:
        return violations
    required_violations = _required_field_violations(
        manifest, MANIFEST_REQUIRED_FIELDS
    )
    violations.extend(required_violations)
    if required_violations:
        return violations
    shape_violations = _manifest_shape_violations(manifest, root)
    violations.extend(shape_violations)

    source = manifest["provenance"]["source"]
    streams = manifest["streams"]
    if not isinstance(streams, dict):
        return violations + ["manifest.streams must be an object"]
    grid = manifest["grid"]
    if not isinstance(grid, dict):
        return violations + ["manifest.grid must be an object"]

    if source == "recorded":
        for stream_name, filename in RECORDED_STREAMS.items():
            if streams.get(stream_name) != filename:
                violations.append(
                    f"manifest.streams.{stream_name} must equal \"{filename}\" "
                    "for a recorded session"
                )
            if not (root / filename).is_file():
                violations.append(f"{filename} is required for a recorded session")
        if (root / "labels_native.parquet").exists():
            violations.append(
                "labels_native.parquet is not allowed for a recorded session"
            )
        if manifest["label_kind"] != "engine_truth":
            violations.append(
                "manifest.label_kind must equal \"engine_truth\" for a recorded session"
            )
        if grid.get("engine_hz") != 60:
            violations.append(
                "manifest.grid.engine_hz must equal 60 for a recorded session"
            )
        if "video" in streams:
            regions = (
                manifest["masked_regions"]
                if isinstance(manifest["masked_regions"], list)
                else []
            )
            has_strip = any(
                isinstance(region, dict)
                and region.get("name") == "frame_index_strip"
                for region in regions
            )
            if not has_strip:
                violations.append(
                    "manifest.masked_regions must contain frame_index_strip "
                    "for a recorded video"
                )
    elif source in ("nitrogen", "wild"):
        if (root / "truth.parquet").exists():
            violations.append(
                "truth.parquet is forbidden when provenance.source is not recorded"
            )
        if streams.get("labels") != FOREIGN_STREAMS["labels"]:
            violations.append(
                "manifest.streams.labels must equal \"labels_native.parquet\" "
                "for a foreign session"
            )
        if not (root / "labels_native.parquet").is_file():
            violations.append(
                "labels_native.parquet is required for a foreign session"
            )
        if manifest["label_kind"] != "mapped":
            violations.append(
                "manifest.label_kind must equal \"mapped\" for a foreign session"
            )
        raw_grid_hz = grid.get("grid_hz")
        grid_hz = _number(raw_grid_hz)
        if (
            isinstance(raw_grid_hz, bool)
            or not isinstance(raw_grid_hz, (int, float))
            or grid_hz is None
            or grid_hz <= 0
        ):
            violations.append(
                "manifest.grid.grid_hz must be a positive number for a foreign session"
            )

    tables: dict[str, pa.Table] = {}
    parquet_specs = (
        ("truth.parquet", TRUTH_SCHEMA),
        ("alignment.parquet", ALIGNMENT_SCHEMA),
        ("labels_native.parquet", LABELS_NATIVE_SCHEMA),
    )
    for filename, expected_schema in parquet_specs:
        path = root / filename
        if not path.is_file():
            continue
        table, table_violations = _schema_violations(
            path, expected_schema, filename
        )
        violations.extend(table_violations)
        if table is not None:
            tables[filename] = table

    for filename in ("truth.parquet", "labels_native.parquet"):
        if filename in tables:
            violations.extend(
                _dense_index_violations(tables[filename], "frame_idx", filename)
            )

    truth = tables.get("truth.parquet")
    if truth is not None and isinstance(manifest["session_id"], str):
        truth_session_ids = set(truth.column("session_id").to_pylist())
        if truth_session_ids != {manifest["session_id"]}:
            violations.append(
                "truth.parquet.session_id must match manifest.session_id in every row"
            )

    alignment = tables.get("alignment.parquet")
    if alignment is not None:
        violations.extend(
            _dense_index_violations(
                alignment,
                "video_frame_idx",
                "alignment.parquet",
                start_at_zero=True,
            )
        )
        statuses = alignment.column("decode_status").to_pylist()
        invalid_statuses = sorted(
            {status for status in statuses if status not in ALIGNMENT_DECODE_STATUSES}
        )
        if invalid_statuses:
            violations.append(
                "alignment.parquet.decode_status contains invalid value(s): "
                + ", ".join(repr(value) for value in invalid_statuses)
            )
        duplicate_count = sum(alignment.column("is_duplicate").to_pylist())
        drop_counts = alignment.column("preceded_by_drop_count").to_pylist()
        if any(value < 0 for value in drop_counts):
            violations.append(
                "alignment.parquet.preceded_by_drop_count must be non-negative"
            )
        if manifest["integrity"]["duplicates"] != duplicate_count:
            violations.append(
                "manifest.integrity.duplicates must match alignment.parquet"
            )
        if manifest["integrity"]["drops"] != sum(drop_counts):
            violations.append("manifest.integrity.drops must match alignment.parquet")

    labels_path = root / "labels_native.parquet"
    if source in ("nitrogen", "wild") and labels_path.is_file():
        try:
            metadata = pq.read_schema(labels_path).metadata or {}
            raw_grid_hz = metadata.get(GRID_HZ_METADATA_KEY)
            parquet_grid_hz = (
                _number(raw_grid_hz.decode("ascii")) if raw_grid_hz else None
            )
        except (OSError, UnicodeError, pa.ArrowException):
            parquet_grid_hz = None
        manifest_grid_hz = _number(grid.get("grid_hz"))
        if parquet_grid_hz is None:
            violations.append(
                "labels_native.parquet metadata must declare numeric grid_hz"
            )
        elif (
            manifest_grid_hz is not None
            and not np.isclose(parquet_grid_hz, manifest_grid_hz)
        ):
            violations.append(
                "labels_native.parquet grid_hz metadata must match "
                "manifest.grid.grid_hz"
            )

    sha_entries = manifest["integrity"]["sha256"]
    if not isinstance(sha_entries, dict):
        sha_entries = {}
    declared_files: set[str] = set()
    for stream_name in STREAM_FILE_KEYS:
        filename = streams.get(stream_name)
        if isinstance(filename, str):
            if Path(filename).name != filename:
                violations.append(
                    f"manifest.streams.{stream_name} must name a file in the "
                    "session directory"
                )
            else:
                declared_files.add(filename)
    for filename in sorted(declared_files):
        path = root / filename
        if filename not in sha_entries:
            violations.append(f"manifest.integrity.sha256.{filename} is required")
        elif not path.is_file():
            violations.append(f"manifest stream file does not exist: {filename}")
        elif sha_entries[filename] != _sha256(path):
            violations.append(f"manifest sha256 mismatch for {filename}")
    for filename, expected_digest in sha_entries.items():
        if not isinstance(filename, str) or not isinstance(expected_digest, str):
            violations.append("manifest.integrity.sha256 entries must be strings")
            continue
        if Path(filename).name != filename:
            violations.append(
                "manifest.integrity.sha256 keys must name files in the "
                "session directory"
            )
            continue
        path = root / filename
        if not path.is_file():
            violations.append(f"manifest sha256 references missing file: {filename}")
        elif filename not in declared_files and expected_digest != _sha256(path):
            violations.append(f"manifest sha256 mismatch for {filename}")

    video_probe: dict[str, Any] | None = None
    video_path = root / "video.mkv"
    if streams.get("video") == "video.mkv" and video_path.is_file():
        video_probe, video_violations = _probe_video(video_path)
        violations.extend(video_violations)
    if video_probe is not None and video_probe["frame_count"] is not None:
        frame_count = video_probe["frame_count"]
        if manifest["integrity"]["video_frames"] != frame_count:
            violations.append(
                "manifest.integrity.video_frames must match video.mkv frame count"
            )
        if alignment is not None and alignment.num_rows != frame_count:
            violations.append(
                "alignment.parquet row count must match video.mkv frame count"
            )

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir")
    args = parser.parse_args(argv)
    violations = validate_session(args.session_dir)
    for violation in violations:
        print(violation.replace("\n", " "), file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
