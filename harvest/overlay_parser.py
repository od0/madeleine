"""Decode the frozen seven-key input overlay from captured video frames.

Portability caveat (2026-07-26 masking audit): the ROIs below assume the
overlay renders at its logical draw constants scaled uniformly by
capture/1920x1080. That holds on the external-display rig, where E4
validated this parser (exact-match 1.0 on active frames,
results/e4_15min.json). On the built-in-display 1710-px rigs the render
pass lands the overlay ~33 logical px higher (report/findings_log.md), so
these ROIs sample blank bar there. Do not run this parser against 1710-px
captures without re-deriving geometry from measured pixels.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.schema import KEY_ORDER


# Frozen v1 overlay geometry, in logical 1920x1080 game-window coordinates.
LOGICAL_FRAME_SIZE = (1920, 1080)
BAR_RECT = (0, 1032, 416, 48)
QUIET_ZONE_INSET = 8
CELL_RECTS = tuple((16 + 56 * index, 1040, 40, 32) for index in range(7))

# The four non-overlapping bands forming the bar's eight-pixel quiet zone.
QUIET_ZONE_RECTS = (
    (BAR_RECT[0], BAR_RECT[1], BAR_RECT[2], QUIET_ZONE_INSET),
    (
        BAR_RECT[0],
        BAR_RECT[1] + BAR_RECT[3] - QUIET_ZONE_INSET,
        BAR_RECT[2],
        QUIET_ZONE_INSET,
    ),
    (
        BAR_RECT[0],
        BAR_RECT[1] + QUIET_ZONE_INSET,
        QUIET_ZONE_INSET,
        BAR_RECT[3] - 2 * QUIET_ZONE_INSET,
    ),
    (
        BAR_RECT[0] + BAR_RECT[2] - QUIET_ZONE_INSET,
        BAR_RECT[1] + QUIET_ZONE_INSET,
        QUIET_ZONE_INSET,
        BAR_RECT[3] - 2 * QUIET_ZONE_INSET,
    ),
)
# Black-to-white contrast is nominally 255. Black-to-up-cell contrast is 40,
# so values below this gate do not provide a trustworthy white reference.
MIN_CONTRAST = 80.0

PARSED_OVERLAY_SCHEMA = pa.schema(
    [
        pa.field("video_frame_idx", pa.int64()),
        *(pa.field(key, pa.bool_(), nullable=True) for key in KEY_ORDER),
    ]
)

Rect = tuple[int, int, int, int]


def _scaled_rect(rect: Rect, frame_size: tuple[int, int]) -> Rect:
    """Scale a logical rectangle by scaling both pairs of boundary edges."""
    logical_width, logical_height = LOGICAL_FRAME_SIZE
    frame_width, frame_height = frame_size
    x, y, width, height = rect
    x0 = round(x * frame_width / logical_width)
    y0 = round(y * frame_height / logical_height)
    x1 = round((x + width) * frame_width / logical_width)
    y1 = round((y + height) * frame_height / logical_height)
    return x0, y0, x1 - x0, y1 - y0


def _crop(gray: np.ndarray, rect: Rect, frame_size: tuple[int, int]) -> np.ndarray:
    x, y, width, height = _scaled_rect(rect, frame_size)
    return gray[y : y + height, x : x + width]


def _central_half(crop: np.ndarray) -> np.ndarray:
    height, width = crop.shape
    sample_width = max(1, round(width * 0.5))
    sample_height = max(1, round(height * 0.5))
    x = (width - sample_width) // 2
    y = (height - sample_height) // 2
    return crop[y : y + sample_height, x : x + sample_width]


def parse_overlay(
    frame: np.ndarray,
    frame_size: tuple[int, int] | None = None,
) -> dict[str, bool | None]:
    """Decode one BGR frame, returning null keys when contrast is insufficient."""
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (height, width, 3)")
    if frame.dtype != np.uint8:
        raise ValueError("frame must have dtype uint8")

    actual_size = (frame.shape[1], frame.shape[0])
    if frame_size is None:
        frame_size = actual_size
    if (
        len(frame_size) != 2
        or frame_size[0] <= 0
        or frame_size[1] <= 0
    ):
        raise ValueError("frame_size must be a positive (width, height) tuple")
    if frame_size != actual_size:
        raise ValueError("frame_size must match the supplied frame dimensions")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bar = _crop(gray, BAR_RECT, frame_size)
    quiet_samples = [
        _crop(gray, rect, frame_size).reshape(-1) for rect in QUIET_ZONE_RECTS
    ]
    if bar.size == 0 or any(sample.size == 0 for sample in quiet_samples):
        raise ValueError("frame is too small for the scaled overlay geometry")

    dark_level = float(np.concatenate(quiet_samples).mean())
    white_level = float(bar.max())
    if white_level - dark_level < MIN_CONTRAST:
        return dict.fromkeys(KEY_ORDER, None)

    threshold = (dark_level + white_level) / 2.0
    decoded: dict[str, bool | None] = {}
    for key, cell_rect in zip(KEY_ORDER, CELL_RECTS, strict=True):
        sample = _central_half(_crop(gray, cell_rect, frame_size))
        if sample.size == 0:
            raise ValueError("frame is too small for the scaled cell geometry")
        decoded[key] = bool(float(sample.mean()) >= threshold)
    return decoded


def parse_video(
    video_path: str | PathLike[str],
    out_parquet: str | PathLike[str],
) -> None:
    """Decode every video frame into a parsed-overlay Parquet file."""
    output_path = Path(out_parquet)
    if output_path.name == "truth.parquet":
        raise ValueError("parsed-overlay labels must not be written as truth.parquet")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open video: {video_path}")

    columns: dict[str, list[int | bool | None]] = {
        "video_frame_idx": [],
        **{key: [] for key in KEY_ORDER},
    }
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded = parse_overlay(frame)
            columns["video_frame_idx"].append(frame_index)
            for key in KEY_ORDER:
                columns[key].append(decoded[key])
            frame_index += 1
    finally:
        capture.release()

    table = pa.Table.from_pydict(columns, schema=PARSED_OVERLAY_SCHEMA)
    pq.write_table(table, output_path)
