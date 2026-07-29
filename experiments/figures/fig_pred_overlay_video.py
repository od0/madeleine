# fig_pred_overlay_video.py — "what the model sees": a 30 s prediction-overlay
# video. Two exhibits share one composite renderer:
#
#   val_a  PUBLIC. Self-recorded session, engine-truth labels, end-to-end
#          pixel model. Produces the mp4 master and (with --webp, or the
#          `webp` subcommand from an existing master) the README animation
#          results/figures/fig_pred_overlay.webp. Animated WebP is used
#          because GitHub renders it inline and a 30 s GIF of this content
#          cannot reach readable quality at comparable size (measured
#          2026-07-27: the best GIF rung was 3.9 MB and still coarse; the
#          media budget is a repository policy choice, not a GitHub limit).
#   y4n    INTERNAL ONLY. Third-party NitroGen holdout video scored by the
#          feature-based GRU run; mapped labels, not engine truth. The mp4 is
#          never committed or exported (third-party footage; the repository's
#          fair-use posture covers single-frame exhibits only).
#
# Data sources (val_a):
#   results/idm/foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0_val_a_preds.npz
#       y_true/y_prob/input_active per predicted frame, stream metadata
#   results/idm/foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0/config.json
#       window=32, window_mode=centered -> target offset 15
#   data/shards_v2/rec_20260724_171305_5min.npz
#       engine_frame_idx / keys / input_active, plus frames [M,128,128,3]:
#       the literal masked, aspect-squashed model inputs
#   sessions/rec_20260724_171305_5min/{video.mkv,alignment.parquet,
#       truth.parquet,manifest.json}
#       capture video, video->engine alignment, engine truth, mask geometry
#   model checkpoint for run foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0
#       (selected endpoint, best_val_step 6000) — passed via --checkpoint;
#       used for gradient saliency and a live logit-vs-sidecar parity check
#
# Data sources (y4n):
#   results/idm/nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0_final_nitrogen_val_preds.npz
#   data/celeste_chunk_index.parquet (controller-widget mask bbox)
#   a local re-fetch of source video y4nQHqYSObI — passed via --proxy-video;
#       the corpus build used the 1280x720@60 rendition, this proxy is
#       640x360@30 on the same timeline (proxy frame = 60 Hz frame // 2)
#
# Window-selection rule (deterministic; implemented in pred_timeline.py and
# frozen below by asserted constants): enumerate 1800-slot windows at stride
# 60 on the engine timeline; gate at prediction coverage >= 0.85 and
# input-active >= 0.80 of captured slots (y4n streams are gapless, gates pass
# trivially, and windows must fit inside the proxy video); restrict to
# windows with all-key true-onset count >= the gated set's median; pick the
# window whose micro accuracy at threshold 0.5 is nearest the run-level
# figure (val_a 0.67623, y4n 0.68484); ties break to the earliest start.
#
# Per-frame accuracy is deliberately not a headline metric in this project:
# the always-released baseline scores 0.829 (val-A) / 0.808 (y4n) on the same
# frames. Both captions state this. See results/idm/KEYPRESS_ACCURACY.md.

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments/figures")
import style  # noqa: E402
from pred_timeline import (  # noqa: E402
    enumerate_stream_windows,
    enumerate_windows,
    reconstruct_timeline,
    select_exhibit_window,
    y4n_source_frame,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from nitrogen.mask import video_mask_rect  # noqa: E402
from theo.frameindex import decode_strip  # noqa: E402

HZ = 60
SPAN = 1800  # 30 s
STRIDE = 60
THRESHOLD = 0.5

VAL_A_RUN = "foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0"
VAL_A_PREDS = ROOT / f"results/idm/{VAL_A_RUN}_val_a_preds.npz"
VAL_A_CONFIG = ROOT / f"results/idm/{VAL_A_RUN}/config.json"
VAL_A_SHARD = ROOT / "data/shards_v2/rec_20260724_171305_5min.npz"
VAL_A_SESSION = ROOT / "sessions/rec_20260724_171305_5min"
# Run-level micro accuracy at 0.5 on active frames (experiments/keypress_accuracy.py).
VAL_A_TARGET_MICRO = 0.67623
VAL_A_BASELINE = 0.829
# The published exhibit window, frozen. A changed sidecar changes the pick;
# update these constants only as a deliberate re-selection of the exhibit.
VAL_A_EXPECTED_START = 27_660

Y4N_RUN = "nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0"
Y4N_PREDS = ROOT / f"results/idm/{Y4N_RUN}_final_nitrogen_val_preds.npz"
Y4N_CHUNK_INDEX = ROOT / "data/celeste_chunk_index.parquet"
Y4N_VIDEO_ID = "y4nQHqYSObI"
Y4N_TARGET_MICRO = 0.68484
Y4N_BASELINE = 0.808
Y4N_EXPECTED = (0, 26_640)  # (stream, local start), frozen as above
Y4N_WINDOW, Y4N_STRIDE_F, Y4N_OFFSET = 128, 3, 63  # samples, frame stride, center

WEBP_LADDER = [  # (width, fps, q); the first attempt <= WEBP_MAX_BYTES wins
    (720, 10, 45), (720, 10, 40), (680, 10, 35),
    (640, 10, 32), (600, 10, 32),
]
WEBP_MAX_BYTES = 10_000_000
# Public distribution re-encode of the mp4 master (full 1080p60). The cap
# is GitHub's inline video-player limit: at or below it, the plain relative
# README link opens the file page with GitHub's own player; above it the
# blob viewer refuses the file (measured 2026-07-27: CRF 24 -> 15.7 MB
# refused, CRF 28 -> 10.12 MB still over). First ladder rung under the cap
# wins.
MP4_DIST_CRF_LADDER = (29, 30)
MP4_DIST_MAX_BYTES = 10_000_000

# ------------------------------------------------------------------ geometry
CANVAS = (1920, 1080)
MAIN = (40, 76, 1104, 621)            # x, y, w, h (16:9)
BAND = (116, 744, 1028, 182)          # piano-roll plot area; labels sit left
INSET_A = (1184, 76, 320, 320)        # model input
INSET_B = (1560, 76, 320, 320)        # saliency (val_a) / note card (y4n)
HUD = (1184, 470, 696, 224)           # 7 rows x 32 px
TALLY = (1184, 712)
CAPTION = (1184, 806)
STRIP = (40, 958, 1840, 74)
FOOTER_Y = 1046
BAND_SLOTS = 360  # 6 s of history in the piano-roll band

WHITE = (255, 255, 255)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


INK = _rgb(style.INK)
INK_MUTED = _rgb(style.INK_MUTED)
GRID = _rgb(style.GRID)
BASELINE = _rgb(style.BASELINE)
KEY_RGB = {k: _rgb(v) for k, v in style.KEY_COLORS.items()}
DROP_GRAY = (216, 215, 209)  # matches fig_piano_roll's capture-drop sliver
MAGMA = (matplotlib.colormaps["magma"](np.linspace(0, 1, 256))[:, :3] * 255).astype(
    np.uint8
)

_FONT_FILE = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
_FONT_BOLD_FILE = font_manager.findfont(
    font_manager.FontProperties(family="DejaVu Sans", weight="bold")
)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_FONT_BOLD_FILE if bold else _FONT_FILE, size)


def _dashed_rect(draw: ImageDraw.ImageDraw, box, dash=7, gap=5) -> None:
    """White casing under an ink dashed stroke (fig_rig_frame's mask style)."""

    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=WHITE, width=3)
    edges = [
        ((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0)),
    ]
    for (ax, ay), (bx, by) in edges:
        length = max(abs(bx - ax), abs(by - ay))
        if length == 0:
            continue
        ux, uy = (bx - ax) / length, (by - ay) / length
        pos = 0.0
        while pos < length:
            end = min(pos + dash, length)
            draw.line(
                [(ax + ux * pos, ay + uy * pos), (ax + ux * end, ay + uy * end)],
                fill=INK, width=1,
            )
            pos = end + gap


def _chip(draw: ImageDraw.ImageDraw, xy, text, fill, fg=WHITE,
          align="left") -> None:
    f = _font(15, bold=True)
    tw = draw.textlength(text, font=f)
    x, y = xy
    if align == "right":
        x -= tw + 18
    draw.rounded_rectangle((x, y, x + tw + 18, y + 26), radius=6, fill=fill)
    draw.text((x + 9, y + 5), text, font=f, fill=fg)


def _paste(img: Image.Image, array: np.ndarray, xy) -> None:
    img.paste(Image.fromarray(array), xy)


# ------------------------------------------------------------------ ffmpeg io
class RawVideoReader:
    """Sequential rawvideo reader over an inclusive source frame range.

    ``get(n)`` returns frame ``n`` (source indexing); requests must be
    non-decreasing. Frames are decoded once and held until passed.
    """

    def __init__(self, path: Path, first: int, last: int, out_wh, scale_flags):
        self.first, self.last = first, last
        self.w, self.h = out_wh
        vf = (
            f"select=between(n\\,{first}\\,{last}),"
            f"scale={self.w}:{self.h}:flags={scale_flags}"
        )
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                "-vf", vf, "-vsync", "0", "-an",
                "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.pos = first - 1
        self.frame: np.ndarray | None = None

    def _read_one(self) -> np.ndarray:
        assert self.proc.stdout is not None
        need = self.w * self.h * 3
        buf = bytearray()
        while len(buf) < need:
            chunk = self.proc.stdout.read(need - len(buf))
            if not chunk:
                stderr = (self.proc.stderr.read() or b"").decode(errors="replace")
                raise RuntimeError(
                    f"video ended at frame {self.pos} (< {self.last}): {stderr[-300:]}"
                )
            buf.extend(chunk)
        return np.frombuffer(bytes(buf), np.uint8).reshape(self.h, self.w, 3)

    def get(self, n: int) -> np.ndarray:
        if n < self.pos:
            raise ValueError(f"non-monotonic read: {n} after {self.pos}")
        while self.pos < n:
            self.frame = self._read_one()
            self.pos += 1
        assert self.frame is not None
        return self.frame

    def close(self) -> None:
        if self.proc.stdout:
            self.proc.stdout.close()
        self.proc.terminate()
        self.proc.wait()


def open_mp4_writer(path: Path, wh, fps: int) -> subprocess.Popen:
    """rawvideo -> libx264 stdin pipe (data/toy_sessions.py's encoder shape)."""

    command = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{wh[0]}x{wh[1]}",
        "-r", str(fps), "-i", "-", "-an",
        "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-r", str(fps), str(path),
    ]
    return subprocess.Popen(
        command, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def close_writer(proc: subprocess.Popen) -> None:
    assert proc.stdin is not None
    proc.stdin.close()
    if proc.wait() != 0:
        stderr = (proc.stderr.read() or b"").decode(errors="replace")
        raise RuntimeError(f"ffmpeg encode failed: {stderr[-400:]}")


def write_webp(master: Path, out_webp: Path) -> None:
    """Animated WebP from the master; walk the ladder until the budget holds."""

    for width, fps, quality in WEBP_LADDER:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(master),
             "-vf", f"fps={fps},scale={width}:-1:flags=lanczos",
             "-c:v", "libwebp", "-q:v", str(quality),
             "-compression_level", "6", "-loop", "0", "-an", str(out_webp)],
            check=True,
        )
        size = out_webp.stat().st_size
        print(f"webp attempt width={width} fps={fps} q={quality}: {size:,} bytes")
        if size <= WEBP_MAX_BYTES:
            sha = hashlib.sha256(out_webp.read_bytes()).hexdigest()
            print(f"webp accepted: {out_webp} ({size:,} bytes, sha256 {sha})")
            return
    raise SystemExit(f"webp exceeds {WEBP_MAX_BYTES:,} bytes at every ladder rung")


def write_dist_mp4(master: Path, out_mp4: Path) -> None:
    """Distribution re-encode of the master for the public repository."""

    for crf in MP4_DIST_CRF_LADDER:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(master), "-an",
             "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(out_mp4)],
            check=True,
        )
        size = out_mp4.stat().st_size
        print(f"dist mp4 attempt crf={crf}: {size:,} bytes")
        if size <= MP4_DIST_MAX_BYTES:
            sha = hashlib.sha256(out_mp4.read_bytes()).hexdigest()
            print(f"dist mp4: {out_mp4} ({size:,} bytes, sha256 {sha})")
            return
    raise SystemExit(
        f"distribution mp4 exceeds {MP4_DIST_MAX_BYTES:,} bytes at every rung"
    )


# ------------------------------------------------------------------ composite
@dataclass
class FrameState:
    main: np.ndarray                    # panel-sized RGB
    inset: np.ndarray                   # 128x128 model input
    saliency: np.ndarray | None         # 128x128 uint8 normalized, or None
    truth: np.ndarray                   # [7] int8, -1 unknown
    prob: np.ndarray                    # [7] float, NaN = no prediction
    covered: bool
    active: bool
    dropped: bool
    thumbs: list[np.ndarray]            # strip thumbnails
    strip_weights: np.ndarray | None    # per-cell saliency weights [cells]
    band_truth: np.ndarray              # [BAND_SLOTS,7]
    band_prob: np.ndarray               # [BAND_SLOTS,7]
    band_onset: np.ndarray              # [BAND_SLOTS,7] bool
    tally: tuple[int, int, int, int]    # micro correct/total, joint correct/total
    t_session: float
    clip_i: int


class Composer:
    def __init__(self, mode: str, cells: int, cell_w: int, strip_note: str,
                 title: str, subtitle: str, caption_lines: list[str],
                 chips_static: list[tuple[str, tuple[int, int, int]]]):
        self.mode = mode
        self.cells = cells
        self.cell_w = cell_w
        self.chips_static = chips_static
        base = Image.new("RGB", CANVAS, WHITE)
        d = ImageDraw.Draw(base)

        d.text((40, 14), title, font=_font(26, bold=True), fill=INK)
        d.text((40, 50), subtitle, font=_font(15), fill=INK_MUTED)

        for x, y, w, h in (MAIN, INSET_A, INSET_B):
            d.rectangle((x - 1, y - 1, x + w, y + h), outline=INK_MUTED, width=1)
        d.text((INSET_A[0], INSET_A[1] + INSET_A[3] + 8),
               "model input (128 x 128, masked, squashed)",
               font=_font(14), fill=INK_MUTED)
        label_b = ("gradient saliency (relative)" if mode == "val_a"
                   else "saliency unavailable for this run")
        d.text((INSET_B[0], INSET_B[1] + INSET_B[3] + 8),
               label_b, font=_font(14), fill=INK_MUTED)

        # HUD chrome: swatches, key names, bar frames, threshold ticks.
        f_key = _font(16, bold=True)
        for i, key in enumerate(style.KEY_ORDER):
            y = HUD[1] + i * 32
            d.rectangle((HUD[0], y + 7, HUD[0] + 14, y + 21), fill=KEY_RGB[key])
            d.text((HUD[0] + 22, y + 4), key, font=f_key, fill=INK)
            bx0, bx1 = self._bar_x()
            d.rectangle((bx0 - 1, y + 6, bx1 + 1, y + 22), outline=GRID, width=1)
            tx = bx0 + (bx1 - bx0) * THRESHOLD
            d.line([(tx, y + 4), (tx, y + 24)], fill=BASELINE, width=1)
        d.text((HUD[0] + 74, HUD[1] - 22), "truth", font=_font(13),
               fill=INK_MUTED)
        d.text((self._bar_x()[0], HUD[1] - 22),
               f"p(pressed), threshold {THRESHOLD}", font=_font(13),
               fill=INK_MUTED)

        d.text((TALLY[0], TALLY[1] - 24), "running tally (scored frames only)",
               font=_font(13), fill=INK_MUTED)

        f_cap = _font(13)
        for j, line in enumerate(caption_lines):
            d.text((CAPTION[0], CAPTION[1] + j * 18), line, font=f_cap,
                   fill=INK_MUTED)

        # Piano-roll band chrome: key labels and frame.
        d.rectangle((BAND[0] - 1, BAND[1] - 1, BAND[0] + BAND[2], BAND[1] + BAND[3]),
                    outline=GRID, width=1)
        row_h = BAND[3] / 7
        for i, key in enumerate(style.KEY_ORDER):
            d.text((MAIN[0], BAND[1] + i * row_h + row_h / 2 - 8), key,
                   font=_font(13), fill=INK)
        d.text((BAND[0], BAND[1] - 20),
               f"last {BAND_SLOTS // HZ} s: truth spans (color), "
               "model p(pressed) (ink), p = 0.5 (dashed)",
               font=_font(13), fill=INK_MUTED)

        # Film-strip chrome.
        d.text((STRIP[0], STRIP[1] - 20), strip_note, font=_font(13),
               fill=INK_MUTED)

        self.base = base
        self.row_h = row_h

    def _bar_x(self) -> tuple[int, int]:
        return HUD[0] + 120, HUD[0] + HUD[2] - 60

    def _draw_band(self, d: ImageDraw.ImageDraw, st: FrameState) -> None:
        x0, y0, w, h = BAND
        px = w / BAND_SLOTS
        row_h = self.row_h
        for i, key in enumerate(style.KEY_ORDER):
            ry0 = y0 + i * row_h + 2
            ry1 = y0 + (i + 1) * row_h - 2
            color = KEY_RGB[key]
            fill = tuple(int(0.70 * 255 + 0.30 * c) for c in color)
            truth_row = st.band_truth[:, i]
            pressed = np.flatnonzero(truth_row == 1)
            if len(pressed):
                edges = np.flatnonzero(np.diff(pressed) > 1)
                seg_start = np.concatenate(([0], edges + 1))
                seg_end = np.concatenate((edges, [len(pressed) - 1]))
                for a, b in zip(seg_start, seg_end):
                    d.rectangle(
                        (x0 + pressed[a] * px, ry0,
                         x0 + (pressed[b] + 1) * px, ry1),
                        fill=fill,
                    )
            for t in np.flatnonzero(st.band_onset[:, i]):
                d.line([(x0 + t * px, ry1 - 6), (x0 + t * px, ry1)],
                       fill=color, width=2)
            mid = (ry0 + ry1) / 2
            for sx in range(0, w, 12):
                d.line([(x0 + sx, mid), (x0 + min(sx + 5, w), mid)],
                       fill=BASELINE, width=1)
            prob_row = st.band_prob[:, i]
            finite = np.isfinite(prob_row)
            idx = np.flatnonzero(finite)
            if len(idx):
                breaks = np.flatnonzero(np.diff(idx) > 1)
                seg_start = np.concatenate(([0], breaks + 1))
                seg_end = np.concatenate((breaks, [len(idx) - 1]))
                for a, b in zip(seg_start, seg_end):
                    pts = [
                        (x0 + t * px,
                         ry1 - float(prob_row[t]) * (ry1 - ry0))
                        for t in idx[a : b + 1]
                    ]
                    if len(pts) > 1:
                        d.line(pts, fill=INK, width=1)
        drops = np.flatnonzero(st.band_truth[:, 0] == -1)
        for t in drops:
            d.rectangle((x0 + t * px, y0, x0 + (t + 1) * px, y0 + h),
                        fill=DROP_GRAY)
        d.line([(x0 + w - 1, y0), (x0 + w - 1, y0 + h)], fill=INK, width=2)

    def frame(self, st: FrameState) -> np.ndarray:
        img = self.base.copy()
        d = ImageDraw.Draw(img)

        _paste(img, st.main, (MAIN[0], MAIN[1]))
        inset = cv2.resize(st.inset, (INSET_A[2], INSET_A[3]),
                           interpolation=cv2.INTER_NEAREST)
        _paste(img, inset, (INSET_A[0], INSET_A[1]))
        if self.mode == "val_a":
            if st.saliency is not None:
                alpha = (st.saliency.astype(np.float32) / 255.0)[..., None]
                heat = MAGMA[st.saliency]
                dim = st.inset.astype(np.float32) * 0.35
                overlay = dim * (1 - alpha * 0.9) + heat * (alpha * 0.9)
                sal = cv2.resize(overlay.astype(np.uint8),
                                 (INSET_B[2], INSET_B[3]),
                                 interpolation=cv2.INTER_NEAREST)
                _paste(img, sal, (INSET_B[0], INSET_B[1]))
            else:
                d.rectangle((INSET_B[0], INSET_B[1],
                             INSET_B[0] + INSET_B[2], INSET_B[1] + INSET_B[3]),
                            fill=GRID)
        else:
            d.rectangle((INSET_B[0], INSET_B[1],
                         INSET_B[0] + INSET_B[2], INSET_B[1] + INSET_B[3]),
                        fill=(250, 249, 246))
            for j, line in enumerate((
                "feature-based run:", "this model consumes frozen",
                "ResNet-18 features of the", "masked input; pixel",
                "gradients are not available.",
            )):
                d.text((INSET_B[0] + 18, INSET_B[1] + 100 + j * 22), line,
                       font=_font(15), fill=INK_MUTED)

        chips_x = MAIN[0] + MAIN[2] - 12
        chips_y = MAIN[1] + 12
        for text, color in self.chips_static:
            _chip(d, (chips_x, chips_y), text, color, align="right")
            chips_y += 34
        if st.dropped:
            _chip(d, (chips_x, chips_y), "capture drop", (150, 60, 50),
                  align="right")
        elif not st.covered:
            _chip(d, (chips_x, chips_y), "no prediction", INK_MUTED,
                  align="right")

        # HUD rows.
        bx0, bx1 = self._bar_x()
        f_mark = _font(16, bold=True)
        for i, key in enumerate(style.KEY_ORDER):
            y = HUD[1] + i * 32
            truth = int(st.truth[i])
            if truth == 1:
                d.rectangle((HUD[0] + 82, y + 7, HUD[0] + 106, y + 21),
                            fill=KEY_RGB[key])
            elif truth == 0:
                d.rectangle((HUD[0] + 82, y + 7, HUD[0] + 106, y + 21),
                            outline=INK_MUTED, width=1)
            else:
                d.text((HUD[0] + 88, y + 4), "?", font=f_mark, fill=INK_MUTED)
            p = float(st.prob[i])
            if np.isfinite(p):
                pred = p >= THRESHOLD
                color = KEY_RGB[key] if pred else tuple(
                    int(0.55 * 255 + 0.45 * c) for c in KEY_RGB[key]
                )
                d.rectangle((bx0, y + 8, bx0 + (bx1 - bx0) * p, y + 20),
                            fill=color)
                if truth in (0, 1) and st.active:
                    ok = pred == bool(truth)
                    d.text((bx1 + 14, y + 3), "OK" if ok else "X",
                           font=f_mark,
                           fill=(27, 122, 84) if ok else (178, 56, 44))
            else:
                d.text((bx0 + 4, y + 4), "-", font=f_mark, fill=INK_MUTED)

        # Tally.
        c, n, jc, jn = st.tally
        pct = f"{100 * c / n:.1f}%" if n else "-"
        d.text((TALLY[0], TALLY[1]),
               f"{c:,} / {n:,} frame-key decisions correct",
               font=_font(19, bold=True), fill=INK)
        d.text((TALLY[0], TALLY[1] + 28), f"running micro accuracy {pct}",
               font=_font(16), fill=INK)
        jpct = f"{100 * jc / jn:.1f}%" if jn else "-"
        d.text((TALLY[0], TALLY[1] + 52),
               f"all-seven exact frames: {jc:,} / {jn:,} ({jpct})",
               font=_font(14), fill=INK_MUTED)

        self._draw_band(d, st)

        # Film strip.
        x = STRIP[0]
        thumb_h = STRIP[3] - 20
        for c_i, thumb in enumerate(st.thumbs):
            cell = cv2.resize(thumb, (self.cell_w, thumb_h),
                              interpolation=cv2.INTER_AREA)
            if not st.covered:
                cell = (cell.astype(np.float32) * 0.35 + 255 * 0.65).astype(np.uint8)
            _paste(img, cell, (x, STRIP[1]))
            if st.strip_weights is not None and st.covered:
                w8 = int(np.clip(st.strip_weights[c_i], 0, 1) * 255)
                d.rectangle((x, STRIP[1] + thumb_h + 3,
                             x + self.cell_w, STRIP[1] + thumb_h + 9),
                            fill=tuple(MAGMA[w8]))
            x += self.cell_w + 2
        marker = self._now_marker_x()
        d.rectangle((marker - 1, STRIP[1] - 3, marker + 1,
                     STRIP[1] + thumb_h + 12), fill=INK)
        d.text((marker + 6, STRIP[1] + thumb_h + 1), "now",
               font=_font(12), fill=INK)

        d.text((STRIP[0] + STRIP[2] - 340, FOOTER_Y - 12),
               f"session t = {st.t_session:7.2f} s    "
               f"clip {st.clip_i / HZ:5.2f} / {SPAN / HZ:.0f} s",
               font=_font(14), fill=INK_MUTED)
        d.text((40, FOOTER_Y - 12),
               "clip window chosen by a fixed documented rule "
               "(coverage and activity gates, onset count >= median, "
               "accuracy nearest run level)",
               font=_font(13), fill=INK_MUTED)
        return np.asarray(img)

    def _now_marker_x(self) -> int:
        if self.mode == "val_a":
            frac = (15 + 0.5) / 32
        else:
            frac = (Y4N_OFFSET + 0.5) / Y4N_WINDOW
        total = self.cells * (self.cell_w + 2) - 2
        return int(STRIP[0] + frac * total)


def run_tally(covered: bool, active: bool, truth: np.ndarray, prob: np.ndarray,
              tally: list[int]) -> None:
    if not (covered and active) or truth[0] < 0:
        return
    correct = (prob >= THRESHOLD) == (truth == 1)
    tally[0] += int(correct.sum())
    tally[1] += 7
    tally[2] += int(correct.all())
    tally[3] += 1


# ------------------------------------------------------------------- val_a
def saliency_pass(args, shard_frames, shard_rows, pred_rows, y_prob):
    """Per-target |grad| maps and per-window-frame norms via the checkpoint."""

    import torch

    from badeline.model import BadelineIDM

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert ck.get("best_val_step") == 6000, ck.get("best_val_step")
    model = BadelineIDM(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    print(f"saliency: device {device}, every {args.saliency_every} frame(s), "
          f"{len(shard_rows)} covered targets")

    n = len(shard_rows)
    sal = np.zeros((n, 128, 128), dtype=np.uint8)
    norms = np.zeros((n, 32), dtype=np.float32)
    deltas: list[float] = []
    todo = list(range(0, n, args.saliency_every))
    batch = 4
    for start in range(0, len(todo), batch):
        idxs = todo[start : start + batch]
        block = np.stack(
            [shard_frames[shard_rows[i] - 15 : shard_rows[i] + 17] for i in idxs]
        )
        x = (
            torch.from_numpy(block).to(device).float().div_(255.0)
            .permute(0, 1, 4, 2, 3).contiguous()
        )
        x.requires_grad_(True)
        logits = model({"frames": x})
        sign = torch.where(torch.sigmoid(logits) >= THRESHOLD, 1.0, -1.0)
        (logits * sign).sum().backward()
        assert x.grad is not None
        g = x.grad.abs()
        spatial = g[:, 15].mean(dim=1)
        per_frame = g.mean(dim=(2, 3, 4))
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        for j, i in enumerate(idxs):
            m = spatial[j].detach().cpu().numpy()
            peak = float(m.max())
            if peak > 0:
                sal[i] = np.clip(m / peak * 255, 0, 255).astype(np.uint8)
            v = per_frame[j].detach().cpu().numpy()
            vmax = float(v.max())
            norms[i] = v / vmax if vmax > 0 else 0.0
            deltas.append(float(np.abs(probs[j] - y_prob[pred_rows[i]]).max()))
        if (start // batch) % 60 == 0:
            print(f"  saliency {start}/{len(todo)}")
    if args.saliency_every > 1:
        for i in range(n):
            src = todo[min(i // args.saliency_every, len(todo) - 1)]
            sal[i] = sal[src]
            norms[i] = norms[src]
    # Parity gate against the sidecar. Local fp32 inference drifts from the
    # pod-side A100 evaluation; measured on the exhibit window (2026-07-27,
    # 1,539 targets): correct selected-endpoint weights give per-window max
    # deltas with median 0.030 / p95 0.083 / max 0.153, while pairing the
    # final endpoint against the selected sidecar gives median 0.139. The
    # MEDIAN separates the regimes; the max does not. The gate catches wrong
    # weights or wrong window assembly, not bit inequality.
    d = np.asarray(deltas)
    print("saliency parity vs sidecar |sigmoid(logit) - prob|: "
          f"median {np.median(d):.4f}, p95 {np.percentile(d, 95):.4f}, "
          f"max {d.max():.4f} over {len(d)} windows")
    if float(np.median(d)) > 0.08:
        raise SystemExit(
            "saliency-pass probabilities diverge from the sidecar; "
            "wrong checkpoint endpoint or window assembly"
        )
    return sal, norms


def run_val_a(args) -> None:
    config = json.loads(VAL_A_CONFIG.read_text())
    window = int(config["window"])
    assert window == 32 and config["window_mode"] == "centered"
    offset = (window - 1) // 2

    preds = dict(np.load(VAL_A_PREDS, allow_pickle=False))
    shard = np.load(VAL_A_SHARD, allow_pickle=False)
    timeline = reconstruct_timeline(preds, shard, window=window)
    total = len(timeline.truth)

    cands = enumerate_windows(
        timeline, span=SPAN, stride=STRIDE, min_coverage=0.85, min_activity=0.80
    )
    pick = select_exhibit_window(cands, target_micro=VAL_A_TARGET_MICRO)
    print(f"val_a window: start slot {pick.start} (t = {pick.start / HZ:.1f} s), "
          f"coverage {pick.coverage:.3f}, activity {pick.activity:.3f}, "
          f"micro {pick.micro:.4f}, joint {pick.joint:.4f}, "
          f"onsets {pick.onsets}, scored {pick.scored_frames}, "
          f"candidates {len(cands)}")
    assert pick.start == VAL_A_EXPECTED_START, pick.start

    efi = shard["engine_frame_idx"].astype(np.int64)
    slot_to_shard = np.full(total, -1, dtype=np.int64)
    slot_to_shard[timeline.slots] = np.arange(len(efi))
    pred_row_of_shard = np.full(len(efi), -1, dtype=np.int64)
    pred_row_of_shard[timeline.pred_shard_pos] = np.arange(
        len(timeline.pred_shard_pos)
    )

    # Alignment: kept video frames must be a bijection with shard rows.
    align = pq.read_table(VAL_A_SESSION / "alignment.parquet").to_pydict()
    keep = [
        i for i in range(len(align["video_frame_idx"]))
        if align["decode_status"][i] == "ok" and not align["is_duplicate"][i]
    ]
    kept_video = np.asarray([align["video_frame_idx"][i] for i in keep])
    kept_engine = np.asarray([align["engine_frame_idx"][i] for i in keep])
    assert np.array_equal(kept_engine, efi), "alignment/shard bijection broken"
    slot_to_video = np.full(total, -1, dtype=np.int64)
    slot_to_video[timeline.slots] = kept_video

    manifest = json.loads((VAL_A_SESSION / "manifest.json").read_text())
    cap_w, cap_h = manifest["capture"]["resolution"]
    mask_rects = {
        r["name"]: tuple(r["rect_px"]) for r in manifest["masked_regions"]
    }

    # Truth spot checks straight from truth.parquet.
    truth_table = pq.read_table(
        VAL_A_SESSION / "truth.parquet",
        columns=["frame_idx", *style.KEY_ORDER, "input_active"],
    ).to_pydict()
    truth_index = {f: i for i, f in enumerate(truth_table["frame_idx"])}
    window_slots = range(pick.start, pick.start + SPAN)
    covered_slots = [s for s in window_slots if timeline.covered[s]]
    captured_slots = [s for s in window_slots if timeline.truth[s, 0] >= 0]
    for s in np.linspace(0, len(covered_slots) - 1, 5).astype(int):
        slot = covered_slots[s]
        row = truth_index[timeline.efi0 + slot]
        expect = [int(truth_table[k][row]) for k in style.KEY_ORDER]
        assert expect == list(timeline.truth[slot]), (slot, expect)
        assert bool(truth_table["input_active"][row]) == bool(
            timeline.active[slot]
        )
    print("truth spot-check: 5 slots match truth.parquet")

    # Frame-index strip decode on three full-resolution frames.
    sx, sy, sw, sh = mask_rects["frame_index_strip"]
    for s in (covered_slots[0], covered_slots[len(covered_slots) // 2],
              covered_slots[-1]):
        vframe = int(slot_to_video[s])
        with tempfile.TemporaryDirectory() as td:
            out_png = Path(td) / "frame.png"
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-i",
                 str(VAL_A_SESSION / "video.mkv"),
                 "-vf", f"select=eq(n\\,{vframe})", "-vsync", "0",
                 "-frames:v", "1", str(out_png)],
                check=True,
            )
            frame = np.asarray(Image.open(out_png).convert("L"))
        decoded = decode_strip(frame[sy : sy + sh, sx : sx + sw])
        assert decoded == timeline.efi0 + s, (decoded, timeline.efi0 + s)
    print("strip decode: 3 frames decode to their engine frame")

    # Model-input frames for the clip (and the film strip / saliency windows).
    shard_lo = max(0, int(slot_to_shard[captured_slots[0]]) - (window + 1))
    shard_hi = int(slot_to_shard[captured_slots[-1]]) + (window + 1)
    frames_all = shard["frames"]
    shard_frames = np.array(frames_all[shard_lo:shard_hi])
    del frames_all

    covered_shard_rows = [int(slot_to_shard[s]) - shard_lo for s in covered_slots]
    covered_pred_rows = [
        int(pred_row_of_shard[int(slot_to_shard[s])]) for s in covered_slots
    ]
    sal, norms = saliency_pass(
        args, shard_frames, covered_shard_rows, covered_pred_rows,
        preds["y_prob"].astype(np.float64),
    )
    sal_of_slot = {s: i for i, s in enumerate(covered_slots)}

    thumbs = {
        r: cv2.resize(shard_frames[r], (55, 54), interpolation=cv2.INTER_AREA)
        for r in range(len(shard_frames))
    }

    composer = Composer(
        mode="val_a", cells=32, cell_w=55,
        strip_note="the model's input window: 32 consecutive frames, "
                   "centered on now (0.53 s); underline = gradient magnitude "
                   "per frame",
        title="What the model sees - engine-truth session",
        subtitle="self-recorded gameplay - engine-truth labels - "
                 "end-to-end model trained only on mapped NitroGen labels - "
                 "60 Hz",
        caption_lines=[
            "Tally scores frame/key agreement at p >= 0.5, only where a",
            "prediction exists and gameplay is active.",
            f"Context: always predicting released scores "
            f"{VAL_A_BASELINE:.1%} here;",
            "per-frame accuracy is not this project's headline metric.",
        ],
        chips_static=[("mask regions dashed: hidden from the model",
                       (70, 70, 68))],
    )

    reader = RawVideoReader(
        VAL_A_SESSION / "video.mkv",
        int(slot_to_video[captured_slots[0]]),
        int(slot_to_video[captured_slots[-1]]),
        (MAIN[2], MAIN[3]), "area",
    )
    scale_x, scale_y = MAIN[2] / cap_w, MAIN[3] / cap_h

    master = args.out_dir / "fig_pred_overlay_val_a.mp4"
    writer = open_mp4_writer(master, CANVAS, HZ)
    tally = [0, 0, 0, 0]
    last_main = np.full((MAIN[3], MAIN[2], 3), 245, dtype=np.uint8)
    last_inset = shard_frames[covered_shard_rows[0]]
    stills = []

    last_row = covered_shard_rows[0]
    for i in range(SPAN):
        slot = pick.start + i
        dropped = timeline.truth[slot, 0] < 0
        covered = bool(timeline.covered[slot])
        if not dropped:
            main = reader.get(int(slot_to_video[slot])).copy()
            pil = Image.fromarray(main)
            dm = ImageDraw.Draw(pil)
            for x, y, w, h in mask_rects.values():
                _dashed_rect(dm, (x * scale_x, y * scale_y,
                                  (x + w) * scale_x, (y + h) * scale_y))
            last_main = np.asarray(pil)
            last_row = int(slot_to_shard[slot]) - shard_lo
            last_inset = shard_frames[last_row]
        sal_i = sal_of_slot.get(slot)
        band = slice(slot - BAND_SLOTS + 1, slot + 1)
        run_tally(covered, bool(timeline.active[slot]),
                  timeline.truth[slot], timeline.prob[slot], tally)
        if covered:
            strip = [thumbs[r]
                     for r in range(last_row - offset, last_row - offset + window)]
            weights = norms[sal_i]
        else:
            # No prediction here (the window would cross a capture drop); show
            # the frames around the current capture, clamped and dimmed.
            strip = [
                thumbs[max(0, min(len(thumbs) - 1, r))]
                for r in range(last_row - offset, last_row - offset + window)
            ]
            weights = None
        state = FrameState(
            main=last_main, inset=last_inset,
            saliency=sal[sal_i] if sal_i is not None else None,
            truth=timeline.truth[slot], prob=timeline.prob[slot],
            covered=covered, active=bool(timeline.active[slot]),
            dropped=bool(dropped),
            thumbs=strip, strip_weights=weights,
            band_truth=timeline.truth[band], band_prob=timeline.prob[band],
            band_onset=timeline.onset[band],
            tally=tuple(tally),
            t_session=(timeline.efi0 + slot) / HZ, clip_i=i,
        )
        frame = composer.frame(state)
        assert writer.stdin is not None
        writer.stdin.write(frame.tobytes())
        if args.stills_dir and i % (SPAN // 6) == 0:
            stills.append((i, frame.copy()))
        if i % 300 == 0:
            print(f"  composite {i}/{SPAN}")
    reader.close()
    close_writer(writer)

    micro = tally[0] / tally[1]
    joint = tally[2] / tally[3]
    assert abs(micro - pick.micro) < 1e-9, (micro, pick.micro)
    assert abs(joint - pick.joint) < 1e-9, (joint, pick.joint)
    assert tally[3] == pick.scored_frames
    print(f"tally check: rendered micro {micro:.4f} == window micro; "
          f"{tally[3]} scored frames")
    print(f"master: {master} ({master.stat().st_size:,} bytes)")

    if args.stills_dir:
        args.stills_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in stills:
            Image.fromarray(frame).save(
                args.stills_dir / f"val_a_still_{i:04d}.png"
            )
        print(f"stills: {len(stills)} -> {args.stills_dir}")

    if args.webp:
        write_webp(master, args.out_dir / "fig_pred_overlay.webp")


# -------------------------------------------------------------------- y4n
def run_y4n(args) -> None:
    preds = dict(np.load(Y4N_PREDS, allow_pickle=False))
    probe = cv2.VideoCapture(str(args.proxy_video))
    assert probe.isOpened(), args.proxy_video
    pw = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    ph = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_proxy = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    print(f"proxy: {pw}x{ph}, {n_proxy} frames")
    assert (pw, ph) == (640, 360), "expected the 640x360@30 proxy fetch"

    cands = enumerate_stream_windows(
        preds, span=SPAN, stride=STRIDE, min_activity=0.80,
        source_frame=y4n_source_frame, max_source_frame=2 * n_proxy,
    )
    pick = select_exhibit_window(cands, target_micro=Y4N_TARGET_MICRO)
    print(f"y4n window: stream {pick.stream}, local {pick.local_start}, "
          f"micro {pick.micro:.4f}, joint {pick.joint:.4f}, "
          f"onsets {pick.onsets}, candidates {len(cands)}")
    assert (pick.stream, pick.local_start) == Y4N_EXPECTED

    rect = video_mask_rect(Y4N_CHUNK_INDEX, Y4N_VIDEO_ID, (pw, ph))
    assert rect == (421, 309, 486, 360), rect
    rx0, ry0, rx1, ry1 = rect
    # 128-space re-mask exactly as data/precompute_features.build_foreign_video.
    sx0 = max(0, int(rx0 / pw * 128) - 1)
    sy0 = max(0, int(ry0 / ph * 128) - 1)
    sx1 = min(128, int(np.ceil(rx1 / pw * 128)) + 1)
    sy1 = min(128, int(np.ceil(ry1 / ph * 128)) + 1)
    print(f"mask rect {rect} -> 128-space ({sx0},{sy0})-({sx1},{sy1})")

    src0 = y4n_source_frame(pick.stream, pick.local_start)
    src_frames = np.arange(src0, src0 + SPAN)
    # Reconstructed model-input buffer must cover the film-strip window too.
    buf_lo = (src0 - Y4N_OFFSET * Y4N_STRIDE_F) // 2
    buf_hi = (src_frames[-1] + (Y4N_WINDOW - 1 - Y4N_OFFSET) * Y4N_STRIDE_F) // 2
    native = RawVideoReader(args.proxy_video, buf_lo, buf_hi, (pw, ph), "area")
    recon = np.empty((buf_hi - buf_lo + 1, 128, 128, 3), dtype=np.uint8)
    for n in range(buf_lo, buf_hi + 1):
        frame = native.get(n).copy()
        frame[ry0:ry1, rx0:rx1] = 0
        small = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
        small[sy0:sy1, sx0:sx1] = 0
        recon[n - buf_lo] = small
    native.close()
    assert int(recon[:, sy0:sy1, sx0:sx1].max(initial=0)) == 0
    print(f"reconstructed {len(recon)} model-input frames from the proxy")
    thumbs = {
        n: cv2.resize(recon[n], (110, 54), interpolation=cv2.INTER_AREA)
        for n in range(len(recon))
    }

    rows = slice(pick.start, pick.start + SPAN)
    y_true = preds["y_true"].astype(np.int8)
    y_prob = preds["y_prob"].astype(np.float64)
    stream_start = pick.start - pick.local_start
    onset_all = np.zeros_like(y_true, dtype=bool)
    onset_all[stream_start + 1 :] = (
        y_true[stream_start + 1 :] == 1
    ) & (y_true[stream_start:-1] == 0)

    composer = Composer(
        mode="y4n", cells=16, cell_w=110,
        strip_note="the model's input window: 128 frames sampled at stride 3 "
                   "(6.4 s span, centered on now); every 8th sample shown",
        title="What the model sees - NitroGen holdout (internal exhibit)",
        subtitle="third-party source video - labels are mapped NitroGen "
                 "gamepad annotations, NOT engine truth - "
                 "reconstructed from a lower-resolution proxy",
        caption_lines=[
            "Tally scores agreement with mapped NitroGen labels at",
            "p >= 0.5; label noise counts against the model.",
            f"Context: always predicting released scores "
            f"{Y4N_BASELINE:.1%} here;",
            "per-frame accuracy is not this project's headline metric.",
        ],
        chips_static=[
            ("mapped NitroGen labels - not engine truth", (150, 60, 50)),
            ("lower-resolution proxy of the source video", (70, 70, 68)),
        ],
    )

    reader = RawVideoReader(
        args.proxy_video, src_frames[0] // 2, src_frames[-1] // 2,
        (MAIN[2], MAIN[3]), "lanczos",
    )
    master = args.out_dir / "fig_pred_overlay_y4n.mp4"
    writer = open_mp4_writer(master, CANVAS, HZ)
    tally = [0, 0, 0, 0]
    stills = []
    ks = list(range(0, Y4N_WINDOW, 8))
    for i in range(SPAN):
        row = pick.start + i
        src = int(src_frames[i])
        main = reader.get(src // 2)
        pil = Image.fromarray(main.copy())
        dm = ImageDraw.Draw(pil)
        _dashed_rect(dm, (rx0 * MAIN[2] / pw, ry0 * MAIN[3] / ph,
                          rx1 * MAIN[2] / pw, ry1 * MAIN[3] / ph))
        main = np.asarray(pil)
        inset = recon[src // 2 - buf_lo]
        strip = [
            thumbs[(src + (k - Y4N_OFFSET) * Y4N_STRIDE_F) // 2 - buf_lo]
            for k in ks
        ]
        band_lo = row - BAND_SLOTS + 1
        band_truth = y_true[band_lo : row + 1]
        band_prob = y_prob[band_lo : row + 1]
        band_onset = onset_all[band_lo : row + 1]
        run_tally(True, True, y_true[row], y_prob[row], tally)
        state = FrameState(
            main=main, inset=inset, saliency=None,
            truth=y_true[row], prob=y_prob[row],
            covered=True, active=True, dropped=False,
            thumbs=strip, strip_weights=None,
            band_truth=band_truth, band_prob=band_prob, band_onset=band_onset,
            tally=tuple(tally),
            t_session=src / HZ, clip_i=i,
        )
        frame = composer.frame(state)
        assert writer.stdin is not None
        writer.stdin.write(frame.tobytes())
        if args.stills_dir and i % (SPAN // 6) == 0:
            stills.append((i, frame.copy()))
        if i % 300 == 0:
            print(f"  composite {i}/{SPAN}")
    reader.close()
    close_writer(writer)

    micro = tally[0] / tally[1]
    assert abs(micro - pick.micro) < 1e-9, (micro, pick.micro)
    print(f"tally check: rendered micro {micro:.4f} == window micro")
    print(f"master: {master} ({master.stat().st_size:,} bytes) - "
          "INTERNAL ONLY, never committed or exported")

    if args.stills_dir:
        args.stills_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in stills:
            Image.fromarray(frame).save(
                args.stills_dir / f"y4n_still_{i:04d}.png"
            )
        print(f"stills: {len(stills)} -> {args.stills_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    a = sub.add_parser("val_a", help="public exhibit: own session, e2e model")
    a.add_argument("--checkpoint", type=Path, required=True,
                   help="path to the run's .pt checkpoint (kept out of Git; "
                   "never a default, paths are machine-local)")
    a.add_argument("--webp", action="store_true",
                   help="also derive results/figures/fig_pred_overlay.webp")
    a.add_argument("--saliency-every", type=int, default=1)

    b = sub.add_parser("y4n", help="internal exhibit: NitroGen holdout, GRU")
    b.add_argument("--proxy-video", type=Path, required=True,
                   help="local re-fetch of the source video (machine-local)")

    w = sub.add_parser(
        "webp",
        help="derive the README animation (and optionally the public "
        "distribution mp4) from an existing master",
    )
    w.add_argument("--master", type=Path,
                   default=ROOT / "results/figures/fig_pred_overlay_val_a.mp4")
    w.add_argument("--mp4", action="store_true",
                   help="also derive results/figures/fig_pred_overlay.mp4")

    for p in (a, b, w):
        p.add_argument("--out-dir", type=Path,
                       default=ROOT / "results/figures")
    for p in (a, b):
        p.add_argument("--stills-dir", type=Path, default=None,
                       help="optionally dump composite stills for QC")

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "val_a":
        run_val_a(args)
    elif args.mode == "y4n":
        run_y4n(args)
    else:
        write_webp(args.master, args.out_dir / "fig_pred_overlay.webp")
        if args.mp4:
            write_dist_mp4(args.master, args.out_dir / "fig_pred_overlay.mp4")


if __name__ == "__main__":
    main()
