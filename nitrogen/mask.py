"""Controller-widget mask rects for re-fetched NitroGen videos.

Orchestrator-owned: an IDM that can see a gamepad widget is reading the
answer key, and a mask that silently lands in the wrong place poisons every
downstream number. This module only RESOLVES rects; applying them (mask at
full resolution before resize, re-mask dilated after resize) is the
builder's job, mirroring data/build_dataset.py's discipline on own sessions.

Conventions, measured at G2 and re-verified on pixels (findings log
2026-07-23): metadata `resolution` is [height, width]; `bbox_controller_overlay`
is [x, y, w, h] in SOURCE pixels at that metadata resolution; bboxes may
overflow the frame edge (observed +5 px and +125 px — widgets render partly
off-frame), so rects are clipped after scaling. Measured on the fetched
corpus (chunk index, 2026-07-25): the bbox is constant across every chunk of
a video, so one rect per video is the contract — a video whose chunks
disagree is refused, not averaged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Safety pad in FETCHED-video pixels around the scaled rect: absorbs bbox
# rounding across resolutions and widget anti-aliasing halos.
PAD_PX = 4


class MaskRectError(ValueError):
    """The chunk index cannot supply a trustworthy rect for this video."""


def scale_bbox(
    bbox_xywh: tuple[float, float, float, float],
    meta_res_hw: tuple[int, int],
    video_wh: tuple[int, int],
    pad_px: int = PAD_PX,
) -> tuple[int, int, int, int]:
    """Scale a source-pixel bbox onto the fetched video and clip to frame.

    Returns (x0, y0, x1, y1) in fetched-video pixels, end-exclusive.
    """

    x, y, w, h = (float(v) for v in bbox_xywh)
    meta_h, meta_w = (int(v) for v in meta_res_hw)
    video_w, video_h = (int(v) for v in video_wh)
    if min(meta_h, meta_w, video_w, video_h) <= 0:
        raise MaskRectError("non-positive resolution")
    if w <= 0 or h <= 0:
        raise MaskRectError(f"degenerate bbox {bbox_xywh}")

    sx, sy = video_w / meta_w, video_h / meta_h
    x0 = int(np.floor(x * sx)) - pad_px
    y0 = int(np.floor(y * sy)) - pad_px
    x1 = int(np.ceil((x + w) * sx)) + pad_px
    y1 = int(np.ceil((y + h) * sy)) + pad_px

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(video_w, x1), min(video_h, y1)
    if x0 >= x1 or y0 >= y1:
        # A bbox may legally overflow an edge, but a rect entirely outside
        # the frame means the conventions were misread somewhere upstream.
        raise MaskRectError(
            f"bbox {bbox_xywh} at meta {meta_res_hw} scaled to empty rect "
            f"on video {video_wh}"
        )
    return x0, y0, x1, y1


def video_mask_rect(
    chunk_index: Path | str,
    video_id: str,
    video_wh: tuple[int, int],
    pad_px: int = PAD_PX,
) -> tuple[int, int, int, int]:
    """Resolve THE mask rect for one fetched video from the chunk index.

    Refuses (raises MaskRectError) when the video is absent, any chunk lacks
    a bbox, or chunks disagree about it. The refusal is the point: a video
    without a trustworthy rect does not enter training until a layout call
    supplies one (S1 concern; rung 2 curates around it).
    """

    table = pq.read_table(
        chunk_index,
        columns=[
            "video_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "metadata_resolution_h", "metadata_resolution_w",
        ],
    )
    rows = table.filter(pc.equal(table["video_id"], video_id))
    if rows.num_rows == 0:
        raise MaskRectError(f"{video_id}: not in chunk index")

    def unique_finite(col: str) -> float:
        values = np.asarray(rows[col].to_pylist(), dtype=float)
        if np.isnan(values).any():
            raise MaskRectError(f"{video_id}: {col} missing on some chunks")
        distinct = np.unique(values)
        if len(distinct) != 1:
            raise MaskRectError(
                f"{video_id}: {col} varies across chunks ({distinct[:4]}...)"
            )
        return float(distinct[0])

    bbox = tuple(unique_finite(c) for c in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
    meta_hw = (
        int(unique_finite("metadata_resolution_h")),
        int(unique_finite("metadata_resolution_w")),
    )
    return scale_bbox(bbox, meta_hw, video_wh, pad_px=pad_px)


def apply_mask(frame: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    """Zero the rect in place. (x0, y0, x1, y1), end-exclusive."""

    x0, y0, x1, y1 = rect
    frame[y0:y1, x0:x1] = 0
