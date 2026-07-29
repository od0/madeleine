from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from nitrogen import slice as nitrogen_slice


VIDEO_SPECS = [
    ("SHARD_0000", "video_a", "celeste", "youtube", [720, 1280], 600, 2),
    ("SHARD_0000", "video_b", "celeste", "twitch", [1080, 1920], 1200, 3),
    ("SHARD_0001", "video_c", "celeste", "twitch", [900, 1600], 900, 2),
    ("SHARD_0001", "other_game", "hollow_knight", "youtube", [720, 1280], 1200, 1),
]


def _metadata(
    video_id: str,
    chunk_index: int,
    game: str,
    source: str,
    resolution: list[int],
    chunk_size: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "chunk_id": f"{chunk_index:04d}",
        "chunk_size": chunk_size,
        "original_video": {
            "video_id": video_id,
            "source": source,
            "url": f"https://example.test/{source}/{video_id}",
            "resolution": resolution,
            "duration": 20,
        },
        "game": game,
        "controller_type": "xboxone",
        # Deliberately overflows a 720-high frame; the census must preserve it.
        "bbox_controller_overlay": [11, 690, 321, 155],
    }
    if video_id == "video_b":
        metadata["bbox_game_area"] = [0.0, 0.0, 1.0, 0.9]
    return metadata


@pytest.fixture
def actions_root(tmp_path: Path) -> Path:
    root = tmp_path / "actions"
    for (
        shard,
        video_id,
        game,
        source,
        resolution,
        chunk_size,
        n_chunks,
    ) in VIDEO_SPECS:
        for chunk_index in range(n_chunks):
            chunk_dir = (
                root
                / shard
                / video_id
                / f"{video_id}_chunk_{chunk_index:04d}"
            )
            chunk_dir.mkdir(parents=True)
            (chunk_dir / "metadata.json").write_text(
                json.dumps(
                    _metadata(
                        video_id,
                        chunk_index,
                        game,
                        source,
                        resolution,
                        chunk_size,
                    )
                ),
                encoding="utf-8",
            )
            if (video_id, chunk_index) in {("video_a", 0), ("video_b", 2)}:
                (chunk_dir / "actions_processed.parquet").touch()
    return root


def test_chunk_index_exact_native_grid_and_raw_layout(
    actions_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "chunk_index.parquet"

    assert (
        nitrogen_slice.main(
            [
                "--actions-root",
                str(actions_root),
                "--game",
                "celeste",
                "--out",
                str(output),
            ]
        )
        == 0
    )

    table = pq.read_table(output)
    assert table.schema == nitrogen_slice.CHUNK_INDEX_SCHEMA
    rows = table.to_pylist()
    assert len(rows) == 7
    assert [(row["video_id"], row["chunk_id"]) for row in rows] == [
        ("video_a", 0),
        ("video_a", 1),
        ("video_b", 0),
        ("video_b", 1),
        ("video_b", 2),
        ("video_c", 0),
        ("video_c", 1),
    ]

    by_chunk = {(row["video_id"], row["chunk_id"]): row for row in rows}
    assert by_chunk[("video_a", 0)] == {
        "video_id": "video_a",
        "chunk_id": 0,
        "source": "youtube",
        "url": "https://example.test/youtube/video_a",
        "chunk_size": 600,
        "grid_hz": 30.0,
        "metadata_resolution_h": 720,
        "metadata_resolution_w": 1280,
        "controller_type": "xboxone",
        "bbox_x": 11,
        "bbox_y": 690,
        "bbox_w": 321,
        "bbox_h": 155,
        "has_processed": True,
        "has_game_area": False,
    }
    assert by_chunk[("video_b", 1)]["grid_hz"] == 60.0
    assert by_chunk[("video_c", 0)]["grid_hz"] == 45.0
    assert by_chunk[("video_b", 2)]["has_processed"] is True
    assert by_chunk[("video_b", 0)]["has_game_area"] is True
    assert by_chunk[("video_a", 0)]["bbox_y"] == 690
    assert by_chunk[("video_a", 0)]["bbox_h"] == 155

    summary = capsys.readouterr().out.strip()
    assert "videos=3 chunks=7 chunk-hours=0.038889" in summary
    assert (
        'per-source={"twitch":{"chunks":5,"videos":2},'
        '"youtube":{"chunks":2,"videos":1}}'
    ) in summary
