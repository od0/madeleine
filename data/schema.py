"""Frozen v1 session schemas shared by writers and validators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import pyarrow as pa


KEY_ORDER = ["left", "right", "up", "down", "jump", "dash", "grab"]

TRUTH_SCHEMA = pa.schema(
    [
        pa.field("frame_idx", pa.int64()),
        *(pa.field(key, pa.bool_()) for key in KEY_ORDER),
        pa.field("input_active", pa.bool_()),
        pa.field("room_id", pa.string()),
        pa.field("pos_x", pa.float64()),
        pa.field("pos_y", pa.float64()),
        pa.field("speed_x", pa.float64()),
        pa.field("speed_y", pa.float64()),
        pa.field("dash_count", pa.int32()),
        pa.field("stamina", pa.float64()),
        pa.field("on_ground", pa.bool_()),
        pa.field("death", pa.bool_()),
        pa.field("session_id", pa.string()),
    ]
)

ALIGNMENT_DECODE_STATUSES = ("ok", "unreadable", "out_of_session")
ALIGNMENT_SCHEMA = pa.schema(
    [
        pa.field("video_frame_idx", pa.int64()),
        pa.field("engine_frame_idx", pa.int64()),
        pa.field("decode_status", pa.string()),
        pa.field("is_duplicate", pa.bool_()),
        pa.field("preceded_by_drop_count", pa.int32()),
    ]
)

LABELS_NATIVE_SCHEMA = pa.schema(
    [
        pa.field("frame_idx", pa.int64()),
        *(pa.field(key, pa.bool_()) for key in KEY_ORDER),
    ]
)
GRID_HZ_METADATA_KEY = b"grid_hz"


def labels_native_schema(grid_hz: float) -> pa.Schema:
    """Return the minimal mapped-label schema with its required grid metadata."""
    return LABELS_NATIVE_SCHEMA.with_metadata(
        {GRID_HZ_METADATA_KEY: str(float(grid_hz)).encode("ascii")}
    )


def _column_types(schema: pa.Schema) -> Mapping[str, pa.DataType]:
    return {field.name: field.type for field in schema}


TRUTH_COLUMNS = TRUTH_SCHEMA.names
TRUTH_DTYPES = _column_types(TRUTH_SCHEMA)
ALIGNMENT_COLUMNS = ALIGNMENT_SCHEMA.names
ALIGNMENT_DTYPES = _column_types(ALIGNMENT_SCHEMA)
LABELS_NATIVE_COLUMNS = LABELS_NATIVE_SCHEMA.names
LABELS_NATIVE_DTYPES = _column_types(LABELS_NATIVE_SCHEMA)

RequiredFieldTree: TypeAlias = Mapping[str, "RequiredFieldTree | None"]

# Values of None are leaves. Empty mappings are required objects whose
# source-specific contents are checked separately by the validator.
MANIFEST_REQUIRED_FIELDS: RequiredFieldTree = {
    "format_version": None,
    "session_id": None,
    "created_at": None,
    "env": {
        "game": None,
        "everest": None,
        "mod": None,
    },
    "capture": {
        "tool": None,
        "requested_fps": None,
        "achieved_fps": None,
        "encode": None,
        "resolution": None,
    },
    "streams": {
        "overlay_style": None,
    },
    "grid": {},
    "label_kind": None,
    "masked_regions": None,
    "integrity": {
        "video_frames": None,
        "duplicates": None,
        "drops": None,
        "sha256": None,
    },
    "actions": {
        "keys": None,
    },
    "provenance": {
        "source": None,
        "origin_url": None,
        "mapping_report": None,
    },
}

PROVENANCE_SOURCES = ("recorded", "nitrogen", "wild")
RECORDED_STREAMS = {
    "video": "video.mkv",
    "truth": "truth.parquet",
    "alignment": "alignment.parquet",
}
FOREIGN_STREAMS = {"labels": "labels_native.parquet"}
STREAM_FILE_KEYS = ("video", "truth", "alignment", "labels")
