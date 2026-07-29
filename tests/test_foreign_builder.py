from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.build_dataset import build_foreign_video
from data.schema import KEY_ORDER

W, H = 320, 180
FPS = 60.0
N_FRAMES = 600
WHITE_FRAME = 200          # single bright frame: exact pairing sentinel
META_RES_HW = (180, 320)   # metadata [h, w] equals video here
BBOX = (10.0, 120.0, 60.0, 50.0)


def _write_video(path: Path, fps: float = FPS, n: int = N_FRAMES) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
    )
    assert writer.isOpened()
    for f in range(n):
        value = 255 if f == WHITE_FRAME else 30
        writer.write(np.full((H, W, 3), value, dtype=np.uint8))
    writer.release()


def _labels(path: Path, n_rows: int, right_on: tuple[int, int] | None) -> None:
    cols: dict[str, list] = {"frame_idx": list(range(n_rows))}
    for key in KEY_ORDER:
        cols[key] = [False] * n_rows
    if right_on is not None:
        for i in range(*right_on):
            cols["right"][i] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path)


def _fixture(
    tmp_path: Path,
    video_fps: float = FPS,
    bad_grid: bool = False,
    inclusive_end: bool = False,
):
    video = tmp_path / "vid_test.mp4"
    _write_video(video, fps=video_fps)

    mapped = tmp_path / "mapped"
    vdir = mapped / "vid_test"
    vdir.mkdir(parents=True)
    (vdir / "mapping_report.json").write_text(json.dumps(
        {"confidence": 0.9, "bind_map": {"jump": ["south"]}}
    ))
    # Chunks: [60,180)+[180,300) form one 240-frame run; [420,540) is a
    # 120-frame fragment below MIN_RUN_FRAMES and must be skipped.
    chunks = [
        ("c0", 60, 180, (40, 90)),    # right held on rows 40..90 → frames 100..150
        ("c1", 180, 300, None),
        ("c2", 420, 540, None),
    ]
    rows = []
    for name, start, end, right_on in chunks:
        _labels(vdir / name / "labels_native.parquet", end - start, right_on)
        rows.append({
            "video_id": "vid_test", "chunk_id": name,
            "start_frame": start,
            "end_frame": end - 1 if inclusive_end else end,
            "start_time": start / FPS, "end_time": end / FPS,
            "grid_hz": 30.0 if bad_grid else 60.0,
            "n_rows": end - start,
        })
    chunk_frames = tmp_path / "chunk_frames.parquet"
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in rows[0]}), chunk_frames)

    index = tmp_path / "chunk_index.parquet"
    pq.write_table(pa.table({
        "video_id": ["vid_test"] * 3,
        "bbox_x": [BBOX[0]] * 3, "bbox_y": [BBOX[1]] * 3,
        "bbox_w": [BBOX[2]] * 3, "bbox_h": [BBOX[3]] * 3,
        "metadata_resolution_h": [META_RES_HW[0]] * 3,
        "metadata_resolution_w": [META_RES_HW[1]] * 3,
    }), index)
    return video, mapped, chunk_frames, index


def test_foreign_build_runs_pairing_masking_and_short_skip(tmp_path: Path) -> None:
    video, mapped, chunk_frames, index = _fixture(tmp_path)
    out = tmp_path / "shards"
    report = build_foreign_video(
        video, "vid_test", mapped, chunk_frames, index, out, frame_size=128
    )

    assert report["label_kind"] == "mapped"
    assert report["runs"] == 2
    assert len(report["parts"]) == 1            # the 120-frame run was skipped
    assert report["skipped_short_frames"] == 120
    part = report["parts"][0]
    assert part["source_frame_range"] == [60, 300]

    with np.load(out / f"{part['session_id']}.npz") as z:
        frames, keys = z["frames"], z["keys"]
        engine_idx = z["engine_frame_idx"]
    assert frames.shape == (240, 128, 128, 3) and keys.shape == (240, 7)
    assert engine_idx[0] == 60 and engine_idx[-1] == 299

    # Exact pairing: source frame 200 is the single white frame and lands at
    # position 200-60=140. Codec bleed tolerated on neighbours, not the peak.
    bright = frames.mean(axis=(1, 2, 3))
    assert int(np.argmax(bright)) == 140
    assert bright[140] > 150 and bright[139] < 100 and bright[141] < 100

    # Labels: right held on chunk-local rows 40..90 of c0 → positions 40..90.
    right = keys[:, KEY_ORDER.index("right")]
    assert right[40] == 1 and right[89] == 1
    assert right[39] == 0 and right[90] == 0 and right.sum() == 50

    # Mask: the scaled rect is exactly zero on every stored frame.
    sx0 = max(0, int(BBOX[0] / W * 128) - 1)
    sy0 = max(0, int(BBOX[1] / H * 128) - 1)
    sx1 = min(128, int(np.ceil((BBOX[0] + BBOX[2]) / W * 128)) + 1)
    sy1 = min(128, int(np.ceil((BBOX[1] + BBOX[3]) / H * 128)) + 1)
    assert frames[:, sy0:sy1, sx0:sx1].max() == 0


def test_foreign_build_refuses_wrong_fps(tmp_path: Path) -> None:
    video, mapped, chunk_frames, index = _fixture(tmp_path, video_fps=30.0)
    with pytest.raises(SystemExit, match="fps"):
        build_foreign_video(
            video, "vid_test", mapped, chunk_frames, index, tmp_path / "s"
        )


def test_foreign_build_refuses_non60_grid(tmp_path: Path) -> None:
    video, mapped, chunk_frames, index = _fixture(tmp_path, bad_grid=True)
    with pytest.raises(SystemExit, match="grid_hz"):
        build_foreign_video(
            video, "vid_test", mapped, chunk_frames, index, tmp_path / "s"
        )


def test_foreign_build_normalizes_inclusive_end_frames(tmp_path: Path) -> None:
    video, mapped, chunk_frames, index = _fixture(
        tmp_path, inclusive_end=True
    )
    out = tmp_path / "shards"
    report = build_foreign_video(
        video, "vid_test", mapped, chunk_frames, index, out, frame_size=128
    )

    assert report["end_frame_conventions"] == ["inclusive"]
    assert report["parts"][0]["source_frame_range"] == [60, 300]
    with np.load(out / f"{report['parts'][0]['session_id']}.npz") as shard:
        assert shard["frames"].shape[0] == 240
        assert shard["engine_frame_idx"][[0, -1]].tolist() == [60, 299]
