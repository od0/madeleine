from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nitrogen.mask import (
    MaskRectError,
    apply_mask,
    scale_bbox,
    video_mask_rect,
)


def test_scale_bbox_identity_resolution_pads_and_rounds() -> None:
    # Same resolution: rect is bbox ± pad.
    rect = scale_bbox((10, 20, 30, 40), (720, 1280), (1280, 720), pad_px=4)
    assert rect == (6, 16, 44, 64)


def test_scale_bbox_scales_1080p_bbox_onto_720p_video() -> None:
    # 1920x1080 metadata, 1280x720 fetch: scale factor 2/3 on both axes.
    rect = scale_bbox((3, 872, 256, 217), (1080, 1920), (1280, 720), pad_px=0)
    x0, y0, x1, y1 = rect
    assert (x0, y0) == (2, 581)          # floor(3*2/3), floor(872*2/3)
    assert x1 == int(np.ceil((3 + 256) * 2 / 3))
    assert y1 == 720                     # ceil((872+217)*2/3)=727 clips to 720


def test_scale_bbox_clips_overflow_but_rejects_fully_outside() -> None:
    # Overflowing edge (observed in G2: widgets render partly off-frame).
    rect = scale_bbox((1200, 600, 200, 200), (720, 1280), (1280, 720), pad_px=0)
    assert rect == (1200, 600, 1280, 720)
    with pytest.raises(MaskRectError):
        scale_bbox((1290, 600, 50, 50), (720, 1280), (1280, 720), pad_px=0)


def test_apply_mask_zeroes_rect_in_place() -> None:
    frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
    apply_mask(frame, (6, 16, 44, 64))
    assert frame[16:64, 6:44].max() == 0
    assert frame[15, 6].max() == 200 and frame[16, 44].max() == 200


def _index(tmp_path: Path, rows: list[dict]) -> Path:
    cols = ["video_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "metadata_resolution_h", "metadata_resolution_w"]
    table = pa.table({c: [r[c] for r in rows] for c in cols})
    path = tmp_path / "index.parquet"
    pq.write_table(table, path)
    return path


def test_video_mask_rect_happy_path_and_refusals(tmp_path: Path) -> None:
    base = dict(bbox_x=3.0, bbox_y=578.0, bbox_w=173.0, bbox_h=146.0,
                metadata_resolution_h=720, metadata_resolution_w=1280)
    index = _index(tmp_path, [
        {**base, "video_id": "vid_ok"},
        {**base, "video_id": "vid_ok"},
        {**base, "video_id": "vid_varies"},
        {**base, "video_id": "vid_varies", "bbox_x": 900.0},
        {**base, "video_id": "vid_missing", "bbox_x": float("nan")},
    ])

    rect = video_mask_rect(index, "vid_ok", (1280, 720), pad_px=0)
    assert rect == (3, 578, 176, 720)   # 578+146=724 clips to 720

    with pytest.raises(MaskRectError, match="varies"):
        video_mask_rect(index, "vid_varies", (1280, 720))
    with pytest.raises(MaskRectError, match="missing"):
        video_mask_rect(index, "vid_missing", (1280, 720))
    with pytest.raises(MaskRectError, match="not in chunk index"):
        video_mask_rect(index, "vid_absent", (1280, 720))
