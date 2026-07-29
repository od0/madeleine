# fig_wild_exhibits: two fair-use exhibit images for the wild-harvest section.
#
#   fig_wild_panels  one probe frame (ss3nhAUaScE, YouTube) with every panel
#                    proposal of the classical bimodal-cell detector drawn and
#                    identified. Rects and scores come verbatim from the
#                    style-survey record; the identity of each region is a
#                    human judgment recorded here.
#   fig_wild_styles  one exemplar crop per overlay style with the measured
#                    survey shares.
#
# Data sources (all read at render time; nothing re-detected here):
#   results/wild/style_survey.jsonl   per-video detector output for the
#                                     60-video style survey: panel_rect list,
#                                     scores, probe_shape (55 rows error-free)
#   results/wild/style_labels.json    human style classification of the 55
#                                     usable probe frames (overlay style lists,
#                                     probed_ok, unclassifiable ids)
#   results/wild/frames/<id>.png      the survey probe frames themselves
#   results/wild/frame_b43KAaem61g.png  probe frame for the translucent-HUD
#                                     exemplar (this video's survey frame was
#                                     saved at repo top level)
#   results/wild/candidates.jsonl     platform (youtube/twitch) per video id
#
# Face policy: every exhibit was visually inspected. ss3nhAUaScE, fJcUr6CXD1I,
# b43KAaem61g and v1509603803 contain no camera feed; elDsFg-S8YA is cropped to
# the keyboard region (hands only); v378693976 is cropped to the controller
# glyph, excluding the webcam above it.
#
# Note on ids: style_labels.json spells the on-screen-keyboard video
# "eIDsFg-S8YA" (capital I); the survey row, its URL and the saved frame use
# "elDsFg-S8YA" (lowercase l), which is the id recorded here.

import json
import sys
from pathlib import Path

sys.path.insert(0, "experiments/figures")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import style

style.apply()

WILD = Path("results/wild")

# ---------------------------------------------------------------- survey data
survey = {}
for line in (WILD / "style_survey.jsonl").read_text().splitlines():
    d = json.loads(line)
    if not d.get("error"):
        survey[d["video_id"]] = d

labels = json.loads((WILD / "style_labels.json").read_text())
probed_ok = labels["probed_ok"]                      # 55
ov = labels["has_input_overlay"]
n_hud = len(ov["action_hud_translucent"])            # 6
n_grid = len(ov["opaque_key_grid"])                  # 1
n_kbd = len(ov["onscreen_keyboard_graphic"])         # 1
n_pad = len(ov["gamepad_display"])                   # 1
n_unclass = len(labels["unclassifiable_probe_hit_non_gameplay"])  # 2
n_timer = probed_ok - n_hud - n_grid - n_kbd - n_pad - n_unclass  # 44

# proposal-count distribution across the 55 usable probe frames
counts = sorted(len(d["panels"]) for d in survey.values())
med = int(np.median(counts))
q1, q3 = int(np.percentile(counts, 25)), int(np.percentile(counts, 75))

# ============================================================ fig_wild_panels
VID = "ss3nhAUaScE"
frame = plt.imread(WILD / "frames" / f"{VID}.png")
panels = survey[VID]["panels"]  # already sorted by descending score

# Human identification of each ranked proposal (True = real overlay region).
IDENT = [
    ("spike bed — game art", False),
    ("crystal cluster — game art", False),
    ("spike row — game art", False),
    ("input HUD + first split rows", True),
    ("cloud bank — game art (clips split-panel edge)", False),
    ("cliff edge — game art", False),
    ("run timer", True),
    ("screen-transition artifact", False),
]
assert len(IDENT) == len(panels) == 8

fig = plt.figure(figsize=(10.6, 4.15))
gs = fig.add_gridspec(1, 2, width_ratios=[6.4, 3.7],
                      left=0.005, right=0.995, top=0.86, bottom=0.03,
                      wspace=0.04)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(frame, interpolation="nearest")
ax.set_axis_off()

# badge placement tweaks per rank
BADGE = {
    4: ("outleft",),  # input HUD: outside the box's left edge, off the HUD
    7: ("below",),    # run timer: under the box, off the digits
    8: ("side",),     # sliver box on the left edge: badge to its right
}
for rank, (p, (name, real)) in enumerate(zip(panels, IDENT), start=1):
    x, y, w, h = p["panel_rect"]
    if real:
        ec, ls, lw = style.ACCENT, "-", 1.8
    else:
        ec, ls, lw = style.BASELINE, (0, (4, 2)), 1.3
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=ec,
                           linestyle=ls, linewidth=lw, zorder=4))
    mode = BADGE.get(rank, ("inside",))[0]
    if mode == "side":
        bx, by, ha = x + w + 5, y + 4, "left"
    elif mode == "outleft":
        bx, by, ha = x - 7, y + 6, "right"
    elif mode == "below":
        bx, by, ha = x + 3, y + h + 6, "left"
    else:
        bx, by, ha = x + 6, y + 6, "left"
    ax.text(bx, by, str(rank), fontsize=7.5, fontweight="bold",
            color=style.INK, ha=ha, va="top", zorder=5,
            bbox=dict(boxstyle="square,pad=0.28", facecolor="white",
                      edgecolor=ec, linewidth=1.0))
ax.text(6, 354, f"source: youtube {VID}", fontsize=6.5,
        color=style.INK_MUTED, ha="left", va="bottom", zorder=5,
        bbox=dict(boxstyle="square,pad=0.25", facecolor="white",
                  edgecolor="none", alpha=0.85))

# ranked list on the right
axl = fig.add_subplot(gs[0, 1])
axl.set_axis_off()
axl.set_xlim(0, 1)
axl.set_ylim(0, 1)
axl.text(0.02, 1.00, "8 proposals, ranked by detector score",
         fontsize=9.5, fontweight="bold", color=style.INK, va="top")
y0, dy = 0.885, 0.093
for rank, (p, (name, real)) in enumerate(zip(panels, IDENT), start=1):
    yy = y0 - (rank - 1) * dy
    ec = style.ACCENT if real else style.BASELINE
    axl.text(0.045, yy, str(rank), fontsize=7.5, fontweight="bold",
             color=style.INK, ha="center", va="center",
             bbox=dict(boxstyle="square,pad=0.28", facecolor="white",
                       edgecolor=ec, linewidth=1.0))
    txt = name
    weight = "bold" if rank == 4 else "normal"
    col = style.INK if real else style.INK_MUTED
    axl.text(0.115, yy, txt, fontsize=8.2, color=col, va="center",
             fontweight=weight)
    axl.text(1.00, yy, f"{p['score']:.1f}", fontsize=7.5,
             color=style.INK_MUTED, ha="right", va="center")
axl.text(0.02, y0 - 8 * dy - 0.005,
         "2 of 8 proposals are overlay; the three top scores\n"
         "are game art, and the input HUD ranks only 4th.",
         fontsize=8.2, color=style.INK, va="top")

fig.suptitle("A wild probe frame is full of panel-shaped things — identity, "
             "not detection, is the vision task",
             x=0.005, y=0.975, ha="left", fontsize=11, fontweight="bold")
fig.text(0.005, 0.895,
         f"classical bimodal-cell detector on one mid-run frame "
         f"(640×360) of speedrun video {VID} · solid blue = real "
         f"overlay, gray dashed = false alarm · across the 55-frame "
         f"survey a frame yields {counts[0]}–{counts[-1]} proposals "
         f"(median {med}, IQR {q1}–{q3})",
         fontsize=7.5, color=style.INK_MUTED, ha="left")
out1 = style.save(fig, "fig_wild_panels")
print(out1)

# ============================================================ fig_wild_styles
# (id, platform, crop x0,y0,x1,y1, frame path, style name, count)
EXEMPLARS = [
    ("fJcUr6CXD1I", "youtube", (0, 0, 168, 92),
     WILD / "frames" / "fJcUr6CXD1I.png",
     "timer / LiveSplit only", n_timer),
    ("b43KAaem61g", "youtube", (494, 0, 702, 104),
     WILD / "frame_b43KAaem61g.png",
     "translucent action HUD", n_hud),
    ("v1509603803", "twitch", (445, 770, 1355, 1058),
     WILD / "frames" / "v1509603803.png",
     "opaque key grid", n_grid),
    ("elDsFg-S8YA", "youtube", (152, 256, 640, 360),
     WILD / "frames" / "elDsFg-S8YA.png",
     "on-screen keyboard + handcam", n_kbd),
    ("v378693976", "twitch", (0, 213, 276, 342),
     WILD / "frames" / "v378693976.png",
     "gamepad display", n_pad),
]

aspects = [(x1 - x0) / (y1 - y0) for _, _, (x0, y0, x1, y1), _, _, _ in EXEMPLARS]
fig = plt.figure(figsize=(13.0, 2.6))
gs = fig.add_gridspec(1, 5, width_ratios=aspects,
                      left=0.005, right=0.995, top=0.76, bottom=0.155,
                      wspace=0.05)
for i, (vid, platform, (x0, y0, x1, y1), path, name, n) in enumerate(EXEMPLARS):
    axc = fig.add_subplot(gs[0, i])
    img = plt.imread(path)
    axc.imshow(img[y0:y1, x0:x1], interpolation="nearest")
    axc.set_axis_off()
    axc.add_patch(Rectangle((0, 0), 1, 1, transform=axc.transAxes, fill=False,
                            edgecolor=style.INK_MUTED, linewidth=0.7,
                            clip_on=False, zorder=6))
    axc.set_title(name, fontsize=9, fontweight="bold", color=style.INK, pad=5)
    pct = n / probed_ok
    axc.text(0.5, -0.075, f"{pct:.0%} · {n}/{probed_ok}",
             transform=axc.transAxes, ha="center", va="top", fontsize=9.5,
             color=style.INK)
    axc.text(0.02, 0.035, f"source: {platform} {vid}",
             transform=axc.transAxes, fontsize=6.0, color=style.INK_MUTED,
             ha="left", va="bottom", zorder=7,
             bbox=dict(boxstyle="square,pad=0.22", facecolor="white",
                       edgecolor="none", alpha=0.85))

fig.suptitle("Input-overlay styles in the wild — one probe frame per "
             "video, 55 usable frames of 60 probed",
             x=0.005, y=0.985, ha="left", fontsize=11, fontweight="bold")
fig.text(0.005, 0.005,
         f"remaining {n_unclass}/{probed_ok} probes landed on non-gameplay "
         "screens (unclassifiable, not negative) · exemplar crops; the "
         "handcam exemplar is cropped to the keyboard region and the gamepad "
         "exemplar to the controller glyph, so no faces appear",
         fontsize=7.5, color=style.INK_MUTED, ha="left", va="bottom")
out2 = style.save(fig, "fig_wild_styles")
print(out2)

# ------------------------------------------------------------- source record
sources = [
    {"figure": "fig_wild_panels", "video_id": VID, "platform": "youtube",
     "note": "style-survey probe frame; all 8 detector proposals drawn from "
             "style_survey.jsonl; translucent CelesteTAS-style input HUD at "
             "top right; no camera feed in frame"},
    {"figure": "fig_wild_styles", "video_id": "fJcUr6CXD1I",
     "platform": "youtube",
     "note": "exemplar: timer/LiveSplit-only class (44/55); top-left run-timer "
             "crop"},
    {"figure": "fig_wild_styles", "video_id": "b43KAaem61g",
     "platform": "youtube",
     "note": "exemplar: translucent action HUD (6/55); probe frame stored at "
             "results/wild/frame_b43KAaem61g.png"},
    {"figure": "fig_wild_styles", "video_id": "v1509603803",
     "platform": "twitch",
     "note": "exemplar: opaque key grid (1/55); bottom key-grid crop"},
    {"figure": "fig_wild_styles", "video_id": "elDsFg-S8YA",
     "platform": "youtube",
     "note": "exemplar: on-screen keyboard + handcam (1/55); cropped to "
             "keyboard region, hands only, no face; spelled eIDsFg-S8YA in "
             "style_labels.json"},
    {"figure": "fig_wild_styles", "video_id": "v378693976",
     "platform": "twitch",
     "note": "exemplar: gamepad display (1/55); cropped to controller glyph, "
             "webcam above the crop excluded"},
]
src_path = style.FIGDIR / "wild_exhibit_sources.json"
src_path.write_text(json.dumps(sources, indent=1) + "\n")
print(src_path)
