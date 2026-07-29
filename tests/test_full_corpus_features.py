from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.build_full_corpus_features import build_chunk_frames


def test_build_chunk_frames_uses_exclusive_native_60_ranges(tmp_path: Path) -> None:
    source = tmp_path / "index.parquet"
    pq.write_table(pa.Table.from_pylist([
        {
            "video_id": "video_a",
            "chunk_id": 3,
            "chunk_size": 1200,
            "grid_hz": 60.0,
        },
        {
            "video_id": "video_a",
            "chunk_id": 5,
            "chunk_size": 1200,
            "grid_hz": 60.0,
        },
    ]), source)

    rows = build_chunk_frames(
        source, {"video_a"}, tmp_path / "chunk_frames.parquet"
    )

    assert rows == [
        {
            "video_id": "video_a",
            "chunk_id": "video_a_chunk_0003",
            "start_frame": 3600,
            "end_frame": 4800,
            "start_time": 60.0,
            "end_time": 80.0,
            "grid_hz": 60.0,
            "n_rows": 1200,
        },
        {
            "video_id": "video_a",
            "chunk_id": "video_a_chunk_0005",
            "start_frame": 6000,
            "end_frame": 7200,
            "start_time": 100.0,
            "end_time": 120.0,
            "grid_hz": 60.0,
            "n_rows": 1200,
        },
    ]


def test_build_chunk_frames_rejects_non_60_source(tmp_path: Path) -> None:
    source = tmp_path / "index.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "video_id": "video_a",
        "chunk_id": 0,
        "chunk_size": 600,
        "grid_hz": 30.0,
    }]), source)

    try:
        build_chunk_frames(source, {"video_a"}, tmp_path / "out.parquet")
    except ValueError as error:
        assert "expected 1,200 rows at 60 Hz" in str(error)
    else:
        raise AssertionError("non-60 source was accepted")
