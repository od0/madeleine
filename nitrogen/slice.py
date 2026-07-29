"""Build a point-in-time chunk census for a NitroGen game slice.

NitroGen chunks are nominally 20 seconds long.  ``grid_hz`` is therefore the
chunk's own row rate (``chunk_size / 20``), not a request to convert it to any
other grid.  Resolution and controller-overlay fields are copied in their
measured dataset-card order: resolution is ``[height, width]`` and the box is
raw ``[x, y, width, height]``.  In particular, overflowing boxes are preserved
for the downstream masking stage to handle.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


CHUNK_SECONDS = 20

CHUNK_INDEX_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string(), nullable=False),
        pa.field("chunk_id", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("chunk_size", pa.int64(), nullable=False),
        pa.field("grid_hz", pa.float64(), nullable=False),
        pa.field("metadata_resolution_h", pa.int64(), nullable=False),
        pa.field("metadata_resolution_w", pa.int64(), nullable=False),
        pa.field("controller_type", pa.string(), nullable=False),
        pa.field("bbox_x", pa.int64()),
        pa.field("bbox_y", pa.int64()),
        pa.field("bbox_w", pa.int64()),
        pa.field("bbox_h", pa.int64()),
        pa.field("has_processed", pa.bool_(), nullable=False),
        pa.field("has_game_area", pa.bool_(), nullable=False),
    ]
)


def _chunk_directories(actions_root: Path) -> list[Path]:
    chunk_dirs: list[Path] = []
    for shard_dir in sorted(path for path in actions_root.glob("SHARD_*") if path.is_dir()):
        for video_dir in sorted(path for path in shard_dir.iterdir() if path.is_dir()):
            prefix = f"{video_dir.name}_chunk_"
            with os.scandir(video_dir) as entries:
                chunk_dirs.extend(
                    video_dir / name
                    for name in sorted(
                        entry.name
                        for entry in entries
                        if entry.is_dir() and entry.name.startswith(prefix)
                    )
                )
    return chunk_dirs


def _required_pair(
    value: Any,
    *,
    field: str,
    metadata_path: Path,
) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be a two-element list in {metadata_path}")
    return int(value[0]), int(value[1])


def _optional_bbox(
    value: Any,
    *,
    metadata_path: Path,
) -> tuple[int | None, int | None, int | None, int | None]:
    if value is None:
        return None, None, None, None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(
            f"bbox_controller_overlay must be a four-element list in {metadata_path}"
        )
    return tuple(int(coordinate) for coordinate in value)  # type: ignore[return-value]


def discover_chunks(actions_root: str | Path, game: str) -> list[dict[str, Any]]:
    """Return one census row per chunk whose metadata game exactly matches."""

    root = Path(actions_root)
    rows: list[dict[str, Any]] = []
    for chunk_dir in _chunk_directories(root):
        metadata_path = chunk_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        if metadata.get("game") != game:
            continue

        original_video = metadata["original_video"]
        height, width = _required_pair(
            original_video["resolution"],
            field="original_video.resolution",
            metadata_path=metadata_path,
        )
        bbox_x, bbox_y, bbox_w, bbox_h = _optional_bbox(
            metadata.get("bbox_controller_overlay"),
            metadata_path=metadata_path,
        )
        chunk_size = int(metadata["chunk_size"])
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive in {metadata_path}")

        rows.append(
            {
                "video_id": str(original_video.get("video_id", chunk_dir.parent.name)),
                "chunk_id": int(metadata["chunk_id"]),
                "source": str(original_video["source"]),
                "url": str(original_video["url"]),
                "chunk_size": chunk_size,
                "grid_hz": chunk_size / CHUNK_SECONDS,
                "metadata_resolution_h": height,
                "metadata_resolution_w": width,
                "controller_type": str(metadata["controller_type"]),
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "has_processed": (chunk_dir / "actions_processed.parquet").is_file(),
                "has_game_area": metadata.get("bbox_game_area") is not None,
            }
        )

    rows.sort(key=lambda row: (row["video_id"], row["chunk_id"]))
    return rows


def write_chunk_index(rows: Sequence[dict[str, Any]], out: str | Path) -> None:
    """Write census rows using the stable chunk-index schema."""

    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows), schema=CHUNK_INDEX_SCHEMA)
    pq.write_table(table, destination)


def _summary(rows: Sequence[dict[str, Any]]) -> str:
    videos_by_source: dict[str, set[str]] = defaultdict(set)
    chunks_by_source: dict[str, int] = defaultdict(int)
    for row in rows:
        source = row["source"]
        videos_by_source[source].add(row["video_id"])
        chunks_by_source[source] += 1
    per_source = {
        source: {
            "videos": len(videos_by_source[source]),
            "chunks": chunks_by_source[source],
        }
        for source in sorted(chunks_by_source)
    }
    chunks = len(rows)
    videos = len({row["video_id"] for row in rows})
    return (
        f"videos={videos} chunks={chunks} "
        f"chunk-hours={chunks * CHUNK_SECONDS / 3600:.6f} "
        f"per-source={json.dumps(per_source, sort_keys=True, separators=(',', ':'))}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a native-grid NitroGen chunk census."
    )
    parser.add_argument("--actions-root", required=True, type=Path)
    parser.add_argument("--game", required=True)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = discover_chunks(args.actions_root, args.game)
    write_chunk_index(rows, args.out)
    print(_summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
