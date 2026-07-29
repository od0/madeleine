"""Decode a TRANSLUCENT input overlay, where absolute brightness is useless.

The opaque parser in `overlay_parser.py` thresholds a cell against a global
dark/white reference, which works because our own recording overlay is drawn
with `BlendState.Opaque` — a cell is exactly white or exactly 0x282828 no
matter what is behind it. Harvested speedrun HUDs are not like that. They are
alpha-blended over live game content, so a released cell's pixel value tracks
the background: the same cell reads 40 over a dark cave and 200 over a bright
sky, and no fixed threshold separates released-over-sky from pressed-over-cave.

What survives translucency is the LOCAL CONTRAST between a cell and the panel
immediately around it. Both are composited over nearly the same background, so
subtracting the surrounding panel cancels the background and leaves the
overlay's own contribution:

    cell_down = alpha_down * white + (1 - alpha_down) * bg
    panel     = alpha_panel * black + (1 - alpha_panel) * bg
    delta     = cell - panel  ->  large and positive only when pressed

The reference ring is sampled from the panel gaps beside each cell rather than
from a global patch, so a background gradient across the panel cancels too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.schema import KEY_ORDER

# Geometry of the calibration overlay, in Celeste's 1920x1080 logical space —
# mirrors granny/InputTruth/Source/WildOverlay.cs. Kept in one place here so a
# change on either side shows up as a decode failure, not a silent drift.
PANEL = (1472, 904, 432, 160)
COLUMNS = 4
CELL_W, CELL_H, CELL_GAP, PANEL_PAD = 96, 64, 8, 12

RING_PX = 5            # width of the panel-background reference ring

# Sample a strip across the TOP of each cell rather than its centre. Measured:
# sampling the centre runs straight through the label glyphs, and when a HUD
# renders its label in a state-dependent colour the text partially cancels the
# very signal being read. On the calibration session that cost jump 0.609 and
# dash 0.817 oracle F1; moving the sample above the text took every key to
# AUC 1.000 and oracle F1 >= 0.980. Label rows are the one part of a cell that
# is not a clean function of the key's state, so they are excluded on purpose.
STRIP_X_FRAC = 0.15    # inset from the left/right cell edges (avoid borders)
STRIP_W_FRAC = 0.70
STRIP_Y_FRAC = 0.08    # start just below the top border
STRIP_H_FRAC = 0.22    # shallow enough to stay clear of centred glyphs


@dataclass(frozen=True)
class CellGeometry:
    """Cell and its local reference ring, in capture pixels."""

    cell: tuple[int, int, int, int]
    ring: tuple[int, int, int, int]


def logical_cell_rect(index: int) -> tuple[int, int, int, int]:
    px, py, _, _ = PANEL
    column, row = index % COLUMNS, index // COLUMNS
    return (
        px + PANEL_PAD + column * (CELL_W + CELL_GAP),
        py + PANEL_PAD + row * (CELL_H + CELL_GAP),
        CELL_W, CELL_H,
    )


def scale_geometry(panel_px: tuple[int, int, int, int]) -> list[CellGeometry]:
    """Map logical cell rects onto the captured panel rect."""

    x0, y0, w, h = panel_px
    sx, sy = w / PANEL[2], h / PANEL[3]
    out: list[CellGeometry] = []
    for index in range(len(KEY_ORDER)):
        lx, ly, lw, lh = logical_cell_rect(index)
        cx = x0 + int(round((lx - PANEL[0]) * sx))
        cy = y0 + int(round((ly - PANEL[1]) * sy))
        cw, ch = int(round(lw * sx)), int(round(lh * sy))
        out.append(CellGeometry(
            cell=(cx, cy, cw, ch),
            ring=(cx - RING_PX, cy - RING_PX,
                  cw + 2 * RING_PX, ch + 2 * RING_PX),
        ))
    return out


def _mean(gray: np.ndarray, rect: tuple[int, int, int, int]) -> float:
    x, y, w, h = rect
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(gray.shape[1], x + w), min(gray.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return float("nan")
    return float(gray[y0:y1, x0:x1].mean())


def cell_deltas(frame_gray: np.ndarray, geometry: list[CellGeometry]) -> np.ndarray:
    """Per-cell (cell − surrounding panel) contrast for one frame."""

    deltas = np.empty(len(geometry), dtype=np.float32)
    for i, geo in enumerate(geometry):
        x, y, w, h = geo.cell
        inner = (
            x + int(w * STRIP_X_FRAC), y + int(h * STRIP_Y_FRAC),
            int(w * STRIP_W_FRAC), int(h * STRIP_H_FRAC),
        )
        rx, ry, rw, rh = geo.ring
        # Ring mean = (ring box total − cell box total) / ring-only area.
        ring_total = _mean(frame_gray, (rx, ry, rw, rh)) * rw * rh
        cell_total = _mean(frame_gray, (x, y, w, h)) * w * h
        ring_area = rw * rh - w * h
        ring_mean = (ring_total - cell_total) / ring_area if ring_area > 0 else np.nan
        deltas[i] = _mean(frame_gray, inner) - ring_mean
    return deltas


# Recorded into every decode report's per-cell QC so a report always names
# the calibrator that produced its thresholds. v1 min-max rescaled each
# column to uint8 and ran cv2's integer Otsu; when one cluster was much
# tighter than the full score range (an opaque overlay's released state spans
# ~2 luma against a ~180-luma range) the whole cluster collapsed into 2-3
# uint8 bins and the back-mapped integer level could land inside the
# cluster's quantization halo, splitting the cluster against itself instead
# of against the other state. v2 removes the lossy pre-quantization.
CALIBRATION_METHOD = "madeleine.cell-threshold.v2-float-otsu-midgap"


def _float_otsu_midgap(column: np.ndarray) -> float:
    """Two-cluster split of raw float scores; threshold mid-gap.

    Runs Otsu's between-class-variance criterion exactly on the empirical
    float distribution — every boundary between consecutive distinct sorted
    values is a candidate split, with no binning — then places the threshold
    at the midpoint of the two clusters' medians. For a bimodal distribution
    that midpoint sits in the centre of the empty gap, far from either
    cluster's quantization halo, and is robust to skew inside a cluster.
    """

    values = np.sort(column.astype(np.float64))
    n = values.size
    prefix = np.cumsum(values)
    total = prefix[-1]
    sizes_low = np.arange(1, n, dtype=np.float64)
    sizes_high = n - sizes_low
    mean_low = prefix[:-1] / sizes_low
    mean_high = (total - prefix[:-1]) / sizes_high
    between = sizes_low * sizes_high * (mean_low - mean_high) ** 2
    # A split is only real between two distinct values; a boundary inside a
    # run of ties is exactly the halo interior the v1 rescale tripped on.
    between[values[1:] <= values[:-1]] = -np.inf
    split = int(np.argmax(between)) + 1
    low, high = values[:split], values[split:]
    return float((np.median(low) + np.median(high)) / 2.0)


def calibrate_threshold(deltas: np.ndarray) -> np.ndarray:
    """Per-key threshold from the delta distribution (float Otsu per column).

    Per-video, per-key, and unsupervised: an input-dense stream makes each
    key's delta distribution cleanly bimodal, and a per-key threshold absorbs
    cells that differ in size or label brightness. The split is computed on
    the raw float scores (see CALIBRATION_METHOD) and lands at the midpoint
    of the two cluster medians, so downstream `>=` decoding never sits on a
    cluster's own quantized values.
    """

    thresholds = np.empty(deltas.shape[1], dtype=np.float32)
    for k in range(deltas.shape[1]):
        column = deltas[:, k]
        column = column[np.isfinite(column)]
        if column.size == 0:
            thresholds[k] = 0.0
            continue
        lo, hi = float(column.min()), float(column.max())
        if hi - lo < 1e-3:
            thresholds[k] = hi + 1.0        # never fires: cell never changed
            continue
        thresholds[k] = _float_otsu_midgap(column)
    return thresholds


def decode(deltas: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return deltas >= thresholds[None, :]
