from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from data.schema import ALIGNMENT_SCHEMA
from data.toy_sessions import generate_sessions, render_frame_index_strip
from theo.frameindex import decode_strip, decode_video, extract_cells


VECTORS_PATH = ROOT / "specs" / "frameindex_test_vectors.json"


def _frame_index_rect(session_dir: Path) -> tuple[int, int, int, int]:
    manifest = json.loads(
        (session_dir / "manifest.json").read_text(encoding="utf-8")
    )
    region = next(
        region
        for region in manifest["masked_regions"]
        if region["name"] == "frame_index_strip"
    )
    return tuple(region["rect_px"])


def _run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope="module")
def toy_sessions(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    root = tmp_path_factory.mktemp("frameindex_toys")
    return [
        generate_sessions(root / f"seed_{seed}", 1, 0.6, seed)[0]
        for seed in (1, 2)
    ]


def test_decode_toy_sessions_matches_frozen_alignment(
    toy_sessions: list[Path],
) -> None:
    for session_dir in toy_sessions:
        expected = pq.read_table(session_dir / "alignment.parquet")
        actual = decode_video(session_dir / "video.mkv", _frame_index_rect(session_dir))

        assert actual.schema == ALIGNMENT_SCHEMA
        assert actual["video_frame_idx"].to_pylist() == expected[
            "video_frame_idx"
        ].to_pylist()
        assert actual["engine_frame_idx"].to_pylist() == expected[
            "engine_frame_idx"
        ].to_pylist()
        assert actual["decode_status"].to_pylist() == ["ok"] * actual.num_rows
        assert actual["is_duplicate"].to_pylist() == [False] * actual.num_rows
        assert actual["preceded_by_drop_count"].to_pylist() == [0] * actual.num_rows


def test_duplicate_and_drop_accounting(
    toy_sessions: list[Path], tmp_path: Path
) -> None:
    session_dir = toy_sessions[0]
    source = session_dir / "video.mkv"
    modified = tmp_path / "duplicate_and_drop.mkv"
    filter_graph = (
        "[0:v]trim=start_frame=0:end_frame=7,setpts=PTS-STARTPTS[a];"
        "[0:v]trim=start_frame=6:end_frame=7,setpts=PTS-STARTPTS[b];"
        "[0:v]trim=start_frame=8,setpts=PTS-STARTPTS[c];"
        "[a][b][c]concat=n=3:v=1:a=0,setpts=N/(60*TB)[v]"
    )
    _run_ffmpeg(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "cfr",
            "-r",
            "60",
            str(modified),
        ]
    )

    original = pq.read_table(session_dir / "alignment.parquet")[
        "engine_frame_idx"
    ].to_pylist()
    expected_indices = original[:7] + [original[6]] + original[8:]
    actual = decode_video(modified, _frame_index_rect(session_dir))

    assert actual["decode_status"].to_pylist() == ["ok"] * len(expected_indices)
    assert actual["engine_frame_idx"].to_pylist() == expected_indices
    duplicate_rows = [
        row
        for row, duplicate in enumerate(actual["is_duplicate"].to_pylist())
        if duplicate
    ]
    drop_counts = actual["preceded_by_drop_count"].to_pylist()
    assert duplicate_rows == [7]
    assert [row for row, count in enumerate(drop_counts) if count] == [8]
    assert drop_counts[8] == 1
    assert sum(actual["is_duplicate"].to_pylist()) == 1
    assert sum(drop_counts) == 1


def _write_corrupted_video(
    source: Path,
    destination: Path,
    rect: tuple[int, int, int, int],
    corrupt_rows: set[int],
) -> None:
    capture = cv2.VideoCapture(str(source))
    assert capture.isOpened()
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
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
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "cfr",
        "-r",
        str(fps),
        str(destination),
    ]
    encoder = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    rng = np.random.default_rng(20250723)
    x, y, strip_width, strip_height = rect
    cell_size = strip_width // 32
    row = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if row in corrupt_rows:
                frame[y : y + strip_height, x : x + strip_width] = rng.integers(
                    0,
                    256,
                    size=(strip_height, strip_width, 3),
                    dtype=np.uint8,
                )
                # Keep the corruption random while guaranteeing bad sync after
                # compression by blacking the complete S1 cell.
                frame[
                    y + cell_size : y + 2 * cell_size,
                    x + cell_size : x + 2 * cell_size,
                ] = 0
            encoder.stdin.write(frame.tobytes())
            row += 1
    finally:
        capture.release()
        encoder.stdin.close()
    stderr = (
        encoder.stderr.read().decode("utf-8", errors="replace")
        if encoder.stderr
        else ""
    )
    return_code = encoder.wait()
    assert return_code == 0, stderr


def test_corrupted_strips_are_unreadable_without_inference(
    toy_sessions: list[Path], tmp_path: Path
) -> None:
    session_dir = toy_sessions[0]
    rect = _frame_index_rect(session_dir)
    corrupt_rows = {5, 11, 17, 23, 29}
    corrupted = tmp_path / "corrupted.mkv"
    _write_corrupted_video(
        session_dir / "video.mkv", corrupted, rect, corrupt_rows
    )

    expected = pq.read_table(session_dir / "alignment.parquet")[
        "engine_frame_idx"
    ].to_pylist()
    actual = decode_video(corrupted, rect)
    statuses = actual["decode_status"].to_pylist()
    indices = actual["engine_frame_idx"].to_pylist()

    assert [row for row, status in enumerate(statuses) if status == "unreadable"] == [
        5,
        11,
        17,
        23,
        29,
    ]
    assert all(
        status == ("unreadable" if row in corrupt_rows else "ok")
        for row, status in enumerate(statuses)
    )
    assert all(indices[row] == -1 for row in corrupt_rows)
    assert all(
        indices[row] == expected[row]
        for row in range(len(expected))
        if row not in corrupt_rows
    )
    assert all(
        not actual["is_duplicate"][row].as_py()
        and actual["preceded_by_drop_count"][row].as_py() == 0
        for row in corrupt_rows
    )


def test_30_fps_transcode_reports_two_to_one_decimation(
    toy_sessions: list[Path], tmp_path: Path
) -> None:
    session_dir = toy_sessions[1]
    transcoded = tmp_path / "internet_grade_30fps.mkv"
    _run_ffmpeg(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(session_dir / "video.mkv"),
            "-an",
            "-vf",
            "select='not(mod(n,2))',setpts=N/(30*TB)",
            "-c:v",
            "libx264",
            "-b:v",
            "1M",
            "-maxrate",
            "1M",
            "-bufsize",
            "2M",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "cfr",
            "-r",
            "30",
            str(transcoded),
        ]
    )

    original = set(
        pq.read_table(session_dir / "alignment.parquet")[
            "engine_frame_idx"
        ].to_pylist()
    )
    actual = decode_video(transcoded, _frame_index_rect(session_dir))
    statuses = actual["decode_status"].to_pylist()
    indices = actual["engine_frame_idx"].to_pylist()

    assert statuses == ["ok"] * actual.num_rows
    assert all(index in original for index in indices)
    assert np.diff(indices).tolist() == [2] * (len(indices) - 1)
    assert actual["is_duplicate"].to_pylist() == [False] * len(indices)
    assert actual["preceded_by_drop_count"].to_pylist() == [0] + [1] * (
        len(indices) - 1
    )


def test_cell_extraction_conforms_to_frozen_vectors() -> None:
    vectors = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))["vectors"]
    for vector in vectors:
        strip = render_frame_index_strip(vector["frame_idx"])
        cells = "".join(str(bit) for bit in extract_cells(strip))
        assert cells == vector["cells"]
        assert decode_strip(strip) == vector["frame_idx"]
