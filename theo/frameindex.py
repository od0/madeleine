"""Decode the v1 frame-index strip from captured video."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.schema import ALIGNMENT_SCHEMA


CELL_COUNT = 30
BACKING_WIDTH_CELLS = 32
BACKING_HEIGHT_CELLS = 3
PATCH_SIZE = 5
PAYLOAD_BITS = 24

Rect = tuple[int, int, int, int]


def _grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError("strip image must be grayscale, BGR, or BGRA")
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("strip image must be grayscale, BGR, or BGRA")


def _backing_bar_threshold(gray: np.ndarray) -> float:
    low, high = np.percentile(gray, (5, 95))
    if high <= low:
        # Valid low-valued indices can occupy less than 5% of the full backing
        # bar, making the prescribed upper percentile equal to the black level.
        # Recover the second level from this same fixed rect; never inspect the
        # surrounding frame or another video frame.
        observed_high = float(np.max(gray))
        if observed_high > high:
            high = observed_high
    return (float(low) + float(high)) / 2.0


def extract_cells(strip: np.ndarray) -> tuple[int, ...]:
    """Extract the 30 frame-index bits from a full backing-bar image."""
    gray = _grayscale(np.asarray(strip))
    if gray.size == 0 or gray.shape[0] < PATCH_SIZE or gray.shape[1] < PATCH_SIZE:
        raise ValueError("strip image is too small for 5x5 cell samples")

    height, width = gray.shape
    threshold = _backing_bar_threshold(gray)
    center_y = int(np.floor(height / 2.0))
    radius = PATCH_SIZE // 2
    bits: list[int] = []
    for cell_index in range(CELL_COUNT):
        center_x = int(
            np.floor((1.5 + cell_index) * width / BACKING_WIDTH_CELLS)
        )
        patch = gray[
            center_y - radius : center_y + radius + 1,
            center_x - radius : center_x + radius + 1,
        ]
        if patch.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError("strip geometry cannot provide 5x5 cell samples")
        bits.append(int(float(np.mean(patch)) >= threshold))
    return tuple(bits)


def _bits_to_int(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _checksum(frame_index: int) -> int:
    checksum = 0
    for shift in range(20, -1, -4):
        checksum ^= (frame_index >> shift) & 0xF
    return checksum


def decode_strip(strip: np.ndarray) -> int | None:
    """Return the encoded engine frame index, or ``None`` when unreadable."""
    cells = extract_cells(strip)
    if cells[:2] != (1, 0):
        return None
    frame_index = _bits_to_int(cells[2 : 2 + PAYLOAD_BITS])
    encoded_checksum = _bits_to_int(cells[2 + PAYLOAD_BITS :])
    if encoded_checksum != _checksum(frame_index):
        return None
    return frame_index


def _normalize_rect(rect: Sequence[int]) -> Rect:
    if len(rect) != 4:
        raise ValueError("rect must contain X,Y,W,H")
    x, y, width, height = (int(value) for value in rect)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("rect coordinates must be non-negative and dimensions positive")
    if width * BACKING_HEIGHT_CELLS != height * BACKING_WIDTH_CELLS:
        raise ValueError("rect must have the frame-index backing bar's 32:3 aspect ratio")
    if width < BACKING_WIDTH_CELLS * PATCH_SIZE / 3:
        raise ValueError("rect is too small for 5x5 cell samples")
    return x, y, width, height


def _alignment_table(decoded: Sequence[int | None]) -> pa.Table:
    row_count = len(decoded)
    readable_rows = [row for row, value in enumerate(decoded) if value is not None]
    first_readable = readable_rows[0] if readable_rows else row_count
    last_readable = readable_rows[-1] if readable_rows else -1

    engine_indices: list[int] = []
    statuses: list[str] = []
    duplicates: list[bool] = []
    drop_counts: list[int] = []
    previous_readable: int | None = None

    for video_frame_idx, value in enumerate(decoded):
        if value is None:
            engine_indices.append(-1)
            if video_frame_idx < first_readable or video_frame_idx > last_readable:
                statuses.append("out_of_session")
            else:
                statuses.append("unreadable")
            duplicates.append(False)
            drop_counts.append(0)
            continue

        engine_index = int(value)
        engine_indices.append(engine_index)
        statuses.append("ok")
        if previous_readable is None:
            duplicates.append(False)
            drop_counts.append(0)
        else:
            duplicates.append(engine_index == previous_readable)
            drop_counts.append(max(0, engine_index - previous_readable - 1))
        previous_readable = engine_index

    arrays = [
        pa.array(np.arange(row_count, dtype=np.int64), type=pa.int64()),
        pa.array(engine_indices, type=pa.int64()),
        pa.array(statuses, type=pa.string()),
        pa.array(duplicates, type=pa.bool_()),
        pa.array(drop_counts, type=pa.int32()),
    ]
    return pa.Table.from_arrays(arrays, schema=ALIGNMENT_SCHEMA)


def decode_video(video_path: str | Path, rect: Sequence[int]) -> pa.Table:
    """Decode every frame in ``video_path`` into the frozen alignment schema."""
    x, y, width, height = _normalize_rect(rect)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"could not open video: {video_path}")

    decoded: list[int | None] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_height, frame_width = frame.shape[:2]
            if x + width > frame_width or y + height > frame_height:
                raise ValueError(
                    f"rect {x},{y},{width},{height} exceeds video frame "
                    f"{frame_width}x{frame_height}"
                )
            decoded.append(decode_strip(frame[y : y + height, x : x + width]))
    finally:
        capture.release()

    return _alignment_table(decoded)


def _parse_rect(value: str) -> Rect:
    try:
        parts = tuple(int(part.strip()) for part in value.split(","))
        return _normalize_rect(parts)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--rect", required=True, type=_parse_rect, metavar="X,Y,W,H")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    table = decode_video(args.video, args.rect)
    pq.write_table(table, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
