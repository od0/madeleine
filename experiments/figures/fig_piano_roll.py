# fig_piano_roll.py — prediction-vs-truth key timeline ("piano roll").
#
# Data sources:
#   results/idm/foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0_val_a_preds.npz
#       y_true [N,7], y_prob [N,7], input_active [N], session_lengths,
#       session_ids (contiguous streams; key order = style.KEY_ORDER; 60 Hz)
#   results/idm/foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0_val_a.json
#       per-key AP (all frames) for the right-margin annotations
#   results/idm/foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0/config.json
#       window=32, window_mode=centered  ->  target offset 15, frame span 32
#   data/shards_v2/rec_20260724_171305_5min.npz
#       engine_frame_idx / keys / input_active for the val-A session, used to
#       place every predicted frame back on the true 60 Hz engine timeline
#       (capture drops appear as real gaps, never bridged). The reconstruction
#       is validated exactly against the preds npz before anything is drawn.
#
# Reconstruction and window enumeration live in pred_timeline.py (shared with
# fig_pred_overlay_video.py); this script keeps its own relaxation ladder.
#
# Window selection (stated in the caption): among all 30 s windows that are
# >=90% input-active and >=97% prediction-covered, take the one whose true-
# onset count (all keys) is the MEDIAN — event-dense but not cherry-picked.

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments/figures")
import style  # noqa: E402
from pred_timeline import (  # noqa: E402
    enumerate_windows,
    reconstruct_timeline,
    select_median_onset_window,
)

style.apply()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PREDS = ROOT / "results/idm/foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0_val_a_preds.npz"
REPORT = ROOT / "results/idm/foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0_val_a.json"
SHARD = ROOT / "data/shards_v2/rec_20260724_171305_5min.npz"

HZ = 60
WINDOW = 32          # model input window (config.json)
OFFSET = 15          # centered target: (window - 1) // 2
SPAN_S = 30
SPAN = SPAN_S * HZ   # 1800 engine frames
STRIDE = 60          # candidate windows slide by 1 s

# ---------------------------------------------------------------- load + align
preds = np.load(PREDS)
y_true = preds["y_true"].astype(bool)          # [N,7]

with np.load(SHARD, allow_pickle=False) as shard:
    timeline = reconstruct_timeline(preds, shard, window=WINDOW)

truth = timeline.truth
prob = timeline.prob
onset = timeline.onset
T = len(truth)

# ------------------------------------------------------- median-density window
# val-A continuity runs are short (median ~3 s), so a fully covered 30 s
# window may not exist; relax span, then coverage, until candidates appear,
# and state the chosen span/coverage in the title.
cands = []
for span_s in (30, 20, 15, 10, 6):
    for min_cov, min_act in ((0.97, 0.90), (0.85, 0.80), (0.70, 0.70)):
        SPAN_S, SPAN = span_s, span_s * HZ
        cands = enumerate_windows(
            timeline, span=SPAN, stride=STRIDE,
            min_coverage=min_cov, min_activity=min_act,
        )
        if cands:
            break
    if cands:
        break
assert cands, "no eligible window at any relaxation"
print(f"window: {SPAN_S}s, coverage>={min_cov}, activity>={min_act}, n={len(cands)}")
dens = np.array([c.onsets for c in cands])
w0 = select_median_onset_window(cands).start
wsl = slice(w0, w0 + SPAN)
t0, t1 = w0 / HZ, (w0 + SPAN) / HZ
n_onsets = int(onset[wsl].sum())

report = json.loads(REPORT.read_text())
ap = report["all_frames"]["metrics"]["per_key_ap"]
rate = {k: float(y_true[:, i].mean()) for i, k in enumerate(style.KEY_ORDER)}

# ---------------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(12.8, 6.4))
tt = np.arange(w0, w0 + SPAN) / HZ
PAD, BH = 0.10, 0.80  # band padding / height inside each unit row

for i, key in enumerate(style.KEY_ORDER):
    yb = (6 - i) + PAD
    color = style.KEY_COLORS[key]
    kt = truth[wsl, i]

    # Ground-truth press spans (translucent, key color); never bridge a drop.
    pressed = kt == 1
    edges = np.flatnonzero(np.diff(pressed.astype(np.int8)))
    seg_starts = np.concatenate(([0], edges + 1))
    seg_ends = np.concatenate((edges + 1, [SPAN]))
    bars = [((w0 + a) / HZ, (b - a) / HZ)
            for a, b in zip(seg_starts, seg_ends) if pressed[a]]
    ax.broken_barh(bars, (yb, BH), facecolors=color, alpha=0.30,
                   linewidth=0, zorder=2)

    # True-onset ticks at the base of the band.
    ots = (w0 + np.flatnonzero(onset[wsl, i])) / HZ
    ax.vlines(ots, yb, yb + 0.16, color=color, linewidth=1.3, zorder=4)

    # p = 0.5 hairline (decision-threshold reference), gray dashed.
    ax.hlines(yb + 0.5 * BH, t0, t1, color=style.BASELINE, linestyle=(0, (2, 3)),
              linewidth=0.5, zorder=1)

    # Model probability, thin ink line scaled 0..1 into the band.
    ax.plot(tt, yb + prob[wsl, i] * BH, color=style.INK, linewidth=0.75,
            zorder=3, solid_capstyle="butt")

    # Right margin: AP vs chance (= press rate on val-A, the AP of a
    # constant predictor).
    ax.text(t1 + 0.35, yb + 0.5 * BH,
            f"AP {ap[key]:.2f}", ha="left", va="center",
            fontsize=8.5, color=style.INK)
    ax.text(t1 + 2.45, yb + 0.5 * BH,
            f"chance {rate[key]:.2f}", ha="left", va="center",
            fontsize=8.5, color=style.BASELINE)

# Capture drops: thin gray slivers across all rows (real gaps in the record).
drop_slots = np.flatnonzero(truth[wsl, 0] < 0)
for a in (w0 + drop_slots) / HZ:
    ax.axvspan(a, a + 1 / HZ, color="#d8d7d1", alpha=0.9, linewidth=0, zorder=1.5)

# Row separators.
for y in range(1, 7):
    ax.axhline(y, color=style.GRID, linewidth=0.8, zorder=1)

ax.set_xlim(t0, t1)
ax.set_ylim(0, 7)
ax.set_yticks([(6 - i) + 0.5 for i in range(7)])
ax.set_yticklabels(style.KEY_ORDER, fontsize=10, color=style.INK)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("session time (s)")
ax.grid(axis="y", visible=False)
ax.grid(axis="x", color=style.GRID, linewidth=0.8)
ax.spines["left"].set_visible(False)

ax.set_title(
    "Where 13.45 h of mapped NitroGen labels gets you: probability vs engine truth, key by key\n"
    "113M end-to-end IDM, foreign labels only — val-A engine truth (dev split), "
    f"30 s window at median onset density ({n_onsets} onsets)",
    loc="left", pad=12,
)

handles = [
    Rectangle((0, 0), 1, 1, facecolor=style.KEY_COLORS["right"], alpha=0.30,
              linewidth=0),
    Line2D([], [], color=style.INK, linewidth=0.9),
    Line2D([], [], color=style.KEY_COLORS["right"], linewidth=1.4),
    Line2D([], [], color=style.BASELINE, linestyle=(0, (2, 3)), linewidth=0.8),
    Rectangle((0, 0), 1, 1, facecolor="#d8d7d1", linewidth=0),
]
labels = ["engine-truth press", "model p(pressed), 0–1 per row",
          "true onset", "p = 0.5", "capture drop"]
ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.045),
          ncol=5, fontsize=8.5, handlelength=1.4, handleheight=1.0,
          borderaxespad=0, columnspacing=1.4)

fig.subplots_adjust(right=0.86)
out = style.save(fig, "fig_piano_roll")
print(f"saved {out}")
print(f"window: t = [{t0:.1f}, {t1:.1f}] s   onsets = {n_onsets}   "
      f"eligible windows = {len(cands)}   density range = "
      f"[{dens.min()}, {dens.max()}]   drops in window = {len(drop_slots)}")
print("per-key AP:", {k: round(ap[k], 3) for k in style.KEY_ORDER})
print("per-key chance:", {k: round(rate[k], 3) for k in style.KEY_ORDER})
