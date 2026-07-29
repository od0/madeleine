from __future__ import annotations

from pathlib import Path
import subprocess

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.schema import KEY_ORDER
from harvest.overlay_parser import (
    BAR_RECT,
    CELL_RECTS,
    LOGICAL_FRAME_SIZE,
    MIN_CONTRAST,
    PARSED_OVERLAY_SCHEMA,
    parse_overlay,
    parse_video,
)


def _scaled_rect(
    rect: tuple[int, int, int, int],
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    logical_width, logical_height = LOGICAL_FRAME_SIZE
    frame_width, frame_height = frame_size
    x, y, width, height = rect
    x0 = round(x * frame_width / logical_width)
    y0 = round(y * frame_height / logical_height)
    x1 = round((x + width) * frame_width / logical_width)
    y1 = round((y + height) * frame_height / logical_height)
    return x0, y0, x1 - x0, y1 - y0


def _fill_rect(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    x, y, width, height = _scaled_rect(
        rect, (frame.shape[1], frame.shape[0])
    )
    cv2.rectangle(
        frame,
        (x, y),
        (x + width - 1, y + height - 1),
        color,
        thickness=cv2.FILLED,
    )


def _synthetic_frame(
    frame_size: tuple[int, int],
    state: tuple[bool, ...],
    *,
    seed: int,
    background: np.ndarray | None = None,
) -> np.ndarray:
    width, height = frame_size
    if background is None:
        rng = np.random.default_rng(seed)
        frame = rng.integers(
            128, 256, size=(height, width, 3), dtype=np.uint8
        )
    else:
        frame = background.copy()

    _fill_rect(frame, BAR_RECT, (0, 0, 0))
    for is_down, rect in zip(state, CELL_RECTS, strict=True):
        fill = (255, 255, 255) if is_down else (40, 40, 40)
        _fill_rect(frame, rect, fill)
    return frame


def _scripted_states(frame_count: int) -> list[tuple[bool, ...]]:
    states = []
    for frame_index in range(frame_count):
        state = tuple(
            bool((frame_index // (2**key_index)) & 1)
            for key_index in range(len(KEY_ORDER))
        )
        if not any(state):
            forced_key = (frame_index // 128) % len(KEY_ORDER)
            state = tuple(
                key_index == forced_key
                for key_index in range(len(KEY_ORDER))
            )
        states.append(state)
    return states


def _precision_recall(
    expected: list[tuple[bool, ...]],
    actual: list[dict[str, bool | None]],
) -> dict[str, tuple[float, float]]:
    metrics = {}
    for key_index, key in enumerate(KEY_ORDER):
        expected_values = [state[key_index] for state in expected]
        actual_values = [row[key] for row in actual]
        true_positives = sum(
            expected_value and actual_value is True
            for expected_value, actual_value in zip(
                expected_values, actual_values, strict=True
            )
        )
        false_positives = sum(
            not expected_value and actual_value is True
            for expected_value, actual_value in zip(
                expected_values, actual_values, strict=True
            )
        )
        false_negatives = sum(
            expected_value and actual_value is not True
            for expected_value, actual_value in zip(
                expected_values, actual_values, strict=True
            )
        )
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        metrics[key] = (precision, recall)
    return metrics


@pytest.fixture(scope="module")
def transcoded_video(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, list[tuple[bool, ...]], float]:
    output_path = tmp_path_factory.mktemp("overlay_transcode") / "overlay.mp4"
    frame_size = (1280, 720)
    fps = 30
    states = _scripted_states(200)
    rng = np.random.default_rng(20250724)
    background = rng.integers(
        128,
        256,
        size=(frame_size[1], frame_size[0], 3),
        dtype=np.uint8,
    )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{frame_size[0]}x{frame_size[1]}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-b:v",
        "1M",
        "-maxrate",
        "1M",
        "-bufsize",
        "2M",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    encoder = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    try:
        for frame_index, state in enumerate(states):
            frame = _synthetic_frame(
                frame_size,
                state,
                seed=frame_index,
                background=background,
            )
            encoder.stdin.write(frame.tobytes())
    finally:
        encoder.stdin.close()
    stderr = (
        encoder.stderr.read().decode("utf-8", errors="replace")
        if encoder.stderr
        else ""
    )
    return_code = encoder.wait()
    assert return_code == 0, stderr

    duration_seconds = len(states) / fps
    bitrate_mbps = output_path.stat().st_size * 8 / duration_seconds / 1_000_000
    assert 0.6 <= bitrate_mbps <= 1.4
    return output_path, states, bitrate_mbps


def test_frozen_geometry_constants() -> None:
    assert LOGICAL_FRAME_SIZE == (1920, 1080)
    assert BAR_RECT == (0, 1032, 416, 48)
    assert CELL_RECTS == tuple(
        (16 + 56 * index, 1040, 40, 32) for index in range(7)
    )


@pytest.mark.parametrize("frame_size", [(1920, 1080), (1280, 720)])
def test_clean_frames_match_script_exactly(
    frame_size: tuple[int, int],
) -> None:
    states = [
        tuple(key_index == active_key for key_index in range(len(KEY_ORDER)))
        for active_key in range(len(KEY_ORDER))
    ]
    states.extend(
        [
            tuple(key_index % 2 == 0 for key_index in range(len(KEY_ORDER))),
            tuple(key_index % 2 == 1 for key_index in range(len(KEY_ORDER))),
        ]
    )
    actual = []
    for frame_index, state in enumerate(states):
        frame = _synthetic_frame(frame_size, state, seed=frame_index)
        row = parse_overlay(frame, frame_size)
        assert row == dict(zip(KEY_ORDER, state, strict=True))
        actual.append(row)

    assert _precision_recall(states, actual) == {
        key: (1.0, 1.0) for key in KEY_ORDER
    }


def test_transcoded_video_precision_and_recall(
    transcoded_video: tuple[Path, list[tuple[bool, ...]], float],
) -> None:
    video_path, states, bitrate_mbps = transcoded_video
    capture = cv2.VideoCapture(str(video_path))
    assert capture.isOpened()
    actual = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            actual.append(parse_overlay(frame))
    finally:
        capture.release()

    assert len(actual) == len(states)
    assert all(value is not None for row in actual for value in row.values())
    metrics = _precision_recall(states, actual)
    print(f"transcode bitrate: {bitrate_mbps:.3f} Mbps")
    for key, (precision, recall) in metrics.items():
        print(f"{key}: precision={precision:.4f}, recall={recall:.4f}")
        assert precision >= 0.95
        assert recall >= 0.95


def test_low_contrast_overlay_returns_nulls() -> None:
    state = (True, False, True, False, True, False, True)
    frame = _synthetic_frame((1920, 1080), state, seed=1)
    x, y, width, height = _scaled_rect(BAR_RECT, (1920, 1080))
    rng = np.random.default_rng(11)
    frame[y : y + height, x : x + width] = rng.integers(
        96, 128, size=(height, width, 3), dtype=np.uint8
    )

    assert MIN_CONTRAST > 0
    assert parse_overlay(frame) == dict.fromkeys(KEY_ORDER, None)


def test_parse_video_writes_exact_nullable_schema(
    transcoded_video: tuple[Path, list[tuple[bool, ...]], float],
    tmp_path: Path,
) -> None:
    video_path, states, _ = transcoded_video
    output_path = tmp_path / "parsed_overlay.parquet"
    parse_video(video_path, output_path)
    table = pq.read_table(output_path)

    expected_schema = pa.schema(
        [
            pa.field("video_frame_idx", pa.int64()),
            *(pa.field(key, pa.bool_(), nullable=True) for key in KEY_ORDER),
        ]
    )
    assert PARSED_OVERLAY_SCHEMA == expected_schema
    assert table.schema == expected_schema
    assert table.num_rows == len(states)
    assert table["video_frame_idx"].to_pylist() == list(range(len(states)))
    assert all(field.nullable for field in list(table.schema)[1:])
