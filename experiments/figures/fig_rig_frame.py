# fig_rig_frame.py — the annotated rig frame: everything the rig renders,
# everything the model never sees. OWN footage only.
#
# Data sources:
#   data/sessions/rec_20260726_020745_calib/video.mkv
#       Phase-A calibration session capture (sha256-pinned in the manifest);
#       one frame extracted at full 1710x962 resolution with ffmpeg.
#   data/sessions/rec_20260726_020745_calib/manifest.json
#       masked_regions rect_px geometry for the frame-index strip, the opaque
#       input overlay, and the translucent wild overlay (the dashed rects are
#       drawn from these values, never placed by eye).
#   data/sessions/rec_20260726_020745_calib/alignment.parquet
#       video_frame_idx -> engine_frame_idx for the chosen frame (decode ok,
#       not a duplicate).
#   data/sessions/rec_20260726_020745_calib/truth.parquet
#       engine-truth key chord at that engine frame (label_kind engine_truth).
#   results/s3_calibration.json
#       Phase-A wild-decode score vs engine truth: macro-F1 0.9977 at zero
#       shift over 35,119 scored frames.
#   specs/frameindex_encoding.md + theo/frameindex.py
#       strip decode, re-run here on the extracted frame as a consistency
#       check (decoded index must equal the alignment table's).
#
# Frame selection (deterministic, stated): video frame 9564 — inside the
# middle half of the session, decode_status ok, not a duplicate, input_active,
# and a five-key chord (left+up+jump+dash+grab) so both overlays show mixed
# pressed/released states. All properties are asserted below.

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, "experiments/figures")
import style  # noqa: E402

style.apply()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theo.frameindex import decode_strip  # noqa: E402

SESSION = ROOT / "data/sessions/rec_20260726_020745_calib"
CALIB = ROOT / "results/s3_calibration.json"
VIDEO_FRAME = 9564

# ------------------------------------------------------------------ load facts
manifest = json.loads((SESSION / "manifest.json").read_text())
assert manifest["session_id"] == "rec_20260726_020745_calib"
W, H = manifest["capture"]["resolution"]
rects = {r["name"]: tuple(r["rect_px"]) for r in manifest["masked_regions"]}
assert set(rects) == {"frame_index_strip", "input_overlay", "wild_overlay"}

align = pq.read_table(SESSION / "alignment.parquet").to_pydict()
row = align["video_frame_idx"].index(VIDEO_FRAME)
assert align["decode_status"][row] == "ok" and not align["is_duplicate"][row]
ENGINE_FRAME = align["engine_frame_idx"][row]

truth = pq.read_table(SESSION / "truth.parquet").to_pydict()
trow = truth["frame_idx"].index(ENGINE_FRAME)
assert truth["input_active"][trow]
chord = [k for k in style.KEY_ORDER if truth[k][trow]]
assert chord == ["left", "up", "jump", "dash", "grab"]

calib = json.loads(CALIB.read_text())
assert calib["session"] == manifest["session_id"]
# The decoder's anchor is the region's panel_rect_px since the 2026-07-26
# mask-geometry correction (rect_px is the larger measured mask rect; the
# decoder keeps the geometry it was validated with).
wild_region = next(
    r for r in manifest["masked_regions"] if r["name"] == "wild_overlay"
)
decoder_anchor = tuple(wild_region.get("panel_rect_px", wild_region["rect_px"]))
assert tuple(calib["panel_rect_px"]) == decoder_anchor
PHASE_A_F1 = calib["macro_f1_at_zero_shift"]  # 0.9977

# ------------------------------------------------- extract + verify the frame
with tempfile.TemporaryDirectory() as td:
    out_png = Path(td) / "frame.png"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(SESSION / "video.mkv"),
         "-vf", f"select=eq(n\\,{VIDEO_FRAME})", "-vsync", "0",
         "-frames:v", "1", str(out_png)],
        check=True,
    )
    frame = np.asarray(Image.open(out_png).convert("RGB"))
assert frame.shape == (H, W, 3), frame.shape

# Independent alignment check: the rendered strip must decode to the same
# engine frame the alignment table assigns to this video frame.
sx, sy, sw, sh = rects["frame_index_strip"]
gray = np.asarray(Image.fromarray(frame[sy:sy + sh, sx:sx + sw]).convert("L"))
decoded = decode_strip(gray)
assert decoded == ENGINE_FRAME, (decoded, ENGINE_FRAME)

# ---------------------------------------------------------------------- figure
# Image at 1:1 device pixels (1710 px / 200 dpi), margins for title + callouts.
IMG_W_IN, IMG_H_IN = W / 200, H / 200          # 8.55 x 4.81 in
M_SIDE, M_TOP, M_BOT = 0.18, 1.30, 0.62        # inches
FIG_W = IMG_W_IN + 2 * M_SIDE
FIG_H = IMG_H_IN + M_TOP + M_BOT

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([M_SIDE / FIG_W, M_BOT / FIG_H,
                   IMG_W_IN / FIG_W, IMG_H_IN / FIG_H])
ax.imshow(frame, extent=(0, W, H, 0), interpolation="none")
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_edgecolor(style.INK_MUTED)
    sp.set_linewidth(0.8)

# Dashed mask rects, exactly at manifest geometry: white casing under an ink
# dashed stroke so the outline reads on both the black strip and the scene.
for x, y, w, h in rects.values():
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="white",
                           lw=2.2, zorder=5, clip_on=False))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=style.INK,
                           lw=1.1, ls=(0, (4, 3)), zorder=6, clip_on=False))

BOX = dict(boxstyle="square,pad=0.38", fc="white", ec=style.INK, lw=0.8)
LEAD = dict(arrowstyle="-", color=style.INK, lw=0.9, shrinkA=2, shrinkB=0)


def callout(text, xy, xytext, ha):
    ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va="center",
                fontsize=9.5, color=style.INK, bbox=BOX, zorder=7,
                annotation_clip=False, arrowprops=LEAD)


# (1) binary frame-index strip (top-left, rect touches the frame corner).
callout("frame index, decoded per frame:\nclock-free alignment",
        xy=(390, 12), xytext=(478, -58), ha="left")

# (2) opaque input overlay (bottom-left).
callout("rendered ground truth (E4 instrument)",
        xy=(185, 952), xytext=(30, H + 52), ha="left")

# (3) translucent action HUD (bottom-right).
callout(f"wild-decode calibration target (Phase A: {PHASE_A_F1:.4f} F1)",
        xy=(1500, 938), xytext=(W - 18, H + 52), ha="right")

# (4) the dashed rects themselves: mask geometry from the session manifest.
ax.text(W - 18, -58, "masked from every model input\n"
        "(dashed: masked_regions, manifest.json)",
        ha="right", va="center", fontsize=9.5, color=style.INK, zorder=7,
        bbox=dict(boxstyle="square,pad=0.38", fc="white", ec=style.INK,
                  lw=0.9, ls=(0, (4, 3))))

fig.text(M_SIDE / FIG_W, 1 - 0.10 / FIG_H,
         "The rig frame: everything the rig renders, everything the model never sees",
         fontsize=13, color=style.INK, va="top")
fig.text(M_SIDE / FIG_W, 1 - 0.42 / FIG_H,
         f"own footage, session {manifest['session_id']}  ·  video frame "
         f"{VIDEO_FRAME:,} — the strip decodes to engine frame "
         f"{ENGINE_FRAME:,}, matching the alignment table",
         fontsize=9, color=style.INK_MUTED, va="top")
fig.text(M_SIDE / FIG_W, 1 - 0.585 / FIG_H,
         f"engine truth at this frame: {' + '.join(chord)}",
         fontsize=9, color=style.INK_MUTED, va="top")

out = style.save(fig, "fig_rig_frame")
print(f"saved {out}")
print(f"video frame {VIDEO_FRAME} -> engine frame {ENGINE_FRAME} "
      f"(strip decode {decoded})")
print(f"chord: {chord}")
print(f"mask rects (x, y, w, h): {rects}")
print(f"Phase A macro-F1 at zero shift: {PHASE_A_F1}")
