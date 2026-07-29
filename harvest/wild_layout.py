"""Versioned, resolution-independent layout contract for wild input HUDs.

The wild corpus contains several unrelated overlay families.  Geometry is
therefore data, not code: every admitted video carries one reviewed layout
file whose rectangles are normalized to the encoded video frame.  Decoders
may combine multiple physical cells into one canonical action, but every
canonical action must be observable; missing cells are never silently treated
as released keys.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal

from data.schema import KEY_ORDER


SCHEMA_VERSION = "madeleine.wild-layout.v1"
Rect = tuple[float, float, float, float]
DecoderKind = Literal["luma", "local_contrast"]
Polarity = Literal["high", "low"]


def _rect(value: Any, field: str) -> Rect:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field} must be [x, y, width, height]")
    try:
        x, y, width, height = (float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain numbers") from exc
    if not all(math.isfinite(v) for v in (x, y, width, height)):
        raise ValueError(f"{field} must contain finite numbers")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"{field} must have non-negative origin and positive size")
    if x + width > 1.0 + 1e-9 or y + height > 1.0 + 1e-9:
        raise ValueError(f"{field} must lie inside normalized frame coordinates")
    return x, y, width, height


def rect_contains(outer: Rect, inner: Rect, tolerance: float = 1e-6) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return (
        ix >= ox - tolerance
        and iy >= oy - tolerance
        and ix + iw <= ox + ow + tolerance
        and iy + ih <= oy + oh + tolerance
    )


def rect_to_pixels(rect: Rect, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    """Scale normalized edges, returning a clipped half-open ``xyxy`` rect."""

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    x, y, width, height = rect
    x0 = max(0, min(frame_width - 1, round(x * frame_width)))
    y0 = max(0, min(frame_height - 1, round(y * frame_height)))
    x1 = max(x0 + 1, min(frame_width, round((x + width) * frame_width)))
    y1 = max(y0 + 1, min(frame_height, round((y + height) * frame_height)))
    return x0, y0, x1, y1


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    action: str
    sample_rect: Rect
    decoder: DecoderKind
    pressed_polarity: Polarity
    reference_rect: Rect | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], index: int) -> "CellSpec":
        field = f"cells[{index}]"
        cell_id = str(raw.get("cell_id", "")).strip()
        action = str(raw.get("action", "")).strip()
        decoder = str(raw.get("decoder", ""))
        polarity = str(raw.get("pressed_polarity", ""))
        if not cell_id:
            raise ValueError(f"{field}.cell_id is required")
        if action not in KEY_ORDER:
            raise ValueError(f"{field}.action must be one of {KEY_ORDER}")
        if decoder not in ("luma", "local_contrast"):
            raise ValueError(f"{field}.decoder must be luma or local_contrast")
        if polarity not in ("high", "low"):
            raise ValueError(f"{field}.pressed_polarity must be high or low")
        sample = _rect(raw.get("sample_rect"), f"{field}.sample_rect")
        reference_raw = raw.get("reference_rect")
        reference = (
            _rect(reference_raw, f"{field}.reference_rect")
            if reference_raw is not None
            else None
        )
        if decoder == "local_contrast" and reference is None:
            raise ValueError(f"{field}.reference_rect is required for local_contrast")
        return cls(
            cell_id=cell_id,
            action=action,
            sample_rect=sample,
            decoder=decoder,  # type: ignore[arg-type]
            pressed_polarity=polarity,  # type: ignore[arg-type]
            reference_rect=reference,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cell_id": self.cell_id,
            "action": self.action,
            "sample_rect": list(self.sample_rect),
            "decoder": self.decoder,
            "pressed_polarity": self.pressed_polarity,
        }
        if self.reference_rect is not None:
            out["reference_rect"] = list(self.reference_rect)
        return out


@dataclass(frozen=True)
class WildLayout:
    video_id: str
    overlay_style: str
    gameplay_rect: Rect
    gameplay_rect_source: str
    gameplay_rect_confidence: float
    mask_rects: tuple[Rect, ...]
    cells: tuple[CellSpec, ...]
    inference_source: str
    inference_confidence: float
    human_reviewed: bool
    evidence_frames: tuple[float, ...]
    # A decoded overlay state observed at frame i is assigned to gameplay
    # frame i + temporal_offset_frames.  A late compositor therefore has a
    # negative offset.  Zero is valid only when it has actually been measured.
    temporal_offset_frames: int
    temporal_offset_source: str
    temporal_offset_confidence: float
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WildLayout":
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema_version must equal {SCHEMA_VERSION!r}")
        video_id = str(raw.get("video_id", "")).strip()
        overlay_style = str(raw.get("overlay_style", "")).strip()
        if not video_id or not overlay_style:
            raise ValueError("video_id and overlay_style are required")
        gameplay_rect = _rect(raw.get("gameplay_rect"), "gameplay_rect")
        gameplay_source = str(raw.get("gameplay_rect_source", "")).strip()
        gameplay_confidence = float(raw.get("gameplay_rect_confidence", 0.0))
        if not gameplay_source or not 0.0 <= gameplay_confidence <= 1.0:
            raise ValueError(
                "gameplay_rect_source and gameplay_rect_confidence in [0,1] are required"
            )
        mask_raw = raw.get("mask_rects")
        if not isinstance(mask_raw, list) or not mask_raw:
            raise ValueError("mask_rects must contain at least one rectangle")
        masks = tuple(_rect(value, f"mask_rects[{i}]") for i, value in enumerate(mask_raw))
        cells_raw = raw.get("cells")
        if not isinstance(cells_raw, list) or not cells_raw:
            raise ValueError("cells must contain at least one cell")
        cells = tuple(CellSpec.from_dict(value, i) for i, value in enumerate(cells_raw))
        ids = [cell.cell_id for cell in cells]
        if len(ids) != len(set(ids)):
            raise ValueError("cell_id values must be unique")
        missing = sorted(set(KEY_ORDER) - {cell.action for cell in cells})
        if missing:
            raise ValueError(
                "layout does not observe every canonical action; missing " + ", ".join(missing)
            )
        for i, cell in enumerate(cells):
            if not any(rect_contains(mask, cell.sample_rect) for mask in masks):
                raise ValueError(f"cells[{i}].sample_rect is not covered by mask_rects")
            if cell.reference_rect is not None and not any(
                rect_contains(mask, cell.reference_rect) for mask in masks
            ):
                raise ValueError(f"cells[{i}].reference_rect is not covered by mask_rects")

        confidence = float(raw.get("inference_confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("inference_confidence must lie in [0, 1]")
        evidence = tuple(float(value) for value in raw.get("evidence_frames_s", []))
        if any(not math.isfinite(value) or value < 0 for value in evidence):
            raise ValueError("evidence frame times must be finite and non-negative")
        human_reviewed = raw.get("human_reviewed")
        if not isinstance(human_reviewed, bool):
            raise ValueError("human_reviewed must be an explicit boolean")
        offset = raw.get("temporal_offset_frames")
        if isinstance(offset, bool) or not isinstance(offset, int) or abs(offset) > 120:
            raise ValueError("temporal_offset_frames must be an integer in [-120, 120]")
        offset_source = str(raw.get("temporal_offset_source", "")).strip()
        if not offset_source:
            raise ValueError("temporal_offset_source is required (use 'unmeasured' explicitly)")
        offset_confidence = float(raw.get("temporal_offset_confidence", 0.0))
        if not 0.0 <= offset_confidence <= 1.0:
            raise ValueError("temporal_offset_confidence must lie in [0, 1]")
        return cls(
            video_id=video_id,
            overlay_style=overlay_style,
            gameplay_rect=gameplay_rect,
            gameplay_rect_source=gameplay_source,
            gameplay_rect_confidence=gameplay_confidence,
            mask_rects=masks,
            cells=cells,
            inference_source=str(raw.get("inference_source", "unknown")),
            inference_confidence=confidence,
            human_reviewed=human_reviewed,
            evidence_frames=evidence,
            temporal_offset_frames=offset,
            temporal_offset_source=offset_source,
            temporal_offset_confidence=offset_confidence,
        )

    @classmethod
    def load(cls, path: str | Path) -> "WildLayout":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "overlay_style": self.overlay_style,
            "gameplay_rect": list(self.gameplay_rect),
            "gameplay_rect_source": self.gameplay_rect_source,
            "gameplay_rect_confidence": self.gameplay_rect_confidence,
            "mask_rects": [list(rect) for rect in self.mask_rects],
            "cells": [cell.to_dict() for cell in self.cells],
            "inference_source": self.inference_source,
            "inference_confidence": self.inference_confidence,
            "human_reviewed": self.human_reviewed,
            "evidence_frames_s": list(self.evidence_frames),
            "temporal_offset_frames": self.temporal_offset_frames,
            "temporal_offset_source": self.temporal_offset_source,
            "temporal_offset_confidence": self.temporal_offset_confidence,
        }
