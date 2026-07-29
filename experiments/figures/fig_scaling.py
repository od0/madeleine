"""fig_scaling — two panels, one message: more own gold is flat; mapped
foreign supervision moves it.

Left panel: own-data scaling of the 16-frame non-causal grid-v3 model.
Right panel: the takeover 3-seed paired comparison (engine-truth-only vs
+13.45 h mapped NitroGen pretraining, zero-shot, no fine-tune).
Shared y-quantity: collar-0 exact transition-event F1 (macro over the seven
keys, oracle per-key thresholds) on the val-A development session,
input-active frames. The two panels use different model families and
loaders (grid-v3 end-to-end ResNet-18 vs takeover frozen-feature GRU), so
levels are comparable only as the same measured quantity, not as an
ablation of one model.

Data sources (verified against these files, not restated from memory):
  results/grid_summary.json
      val_a.transition_f1_oracle_collar0.scale{1,2,3}_16f._macro
      -> 0.022119 / 0.019052 / 0.020809 (one seed per point)
  results/grid_v3/scale{1,2,3}_16f.json
      the underlying per-run eval reports (runs_v3/scale*_16f)
  results/baselines.json
      val-A shard rec_20260724_171305_5min:
      shuffled_event_f1_macro["0"] = 0.004999 (the luck anchor: true event
      counts placed uniformly at random over active frames, 10 seeds, mean)
  data/shards_v2/train_scale{1,2,3}.txt + data/shards_v2/build_manifest.json
      retained frames per scale level -> minutes of own gold:
      53,369 / 89,670 / 143,451 frames = 14.8 / 24.9 / 39.8 min at 60 Hz
  results/idm/SUMMARY.md ("Primary three-seed result", lines 39-54)
      engine-truth-only exact F1: 0.0801 / 0.0789 / 0.0786 (0.0792 +/- 0.0008)
      mapped-pretrained exact F1: 0.0911 / 0.0942 / 0.0907 (0.0920 +/- 0.0019)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "experiments/figures")
import matplotlib.pyplot as plt  # noqa: E402
import style  # noqa: E402

style.apply()

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- left panel data
summary = json.loads((ROOT / "results/grid_summary.json").read_text())
collar0 = summary["val_a"]["transition_f1_oracle_collar0"]
scale_f1 = [collar0[f"scale{i}_16f"]["_macro"][0] for i in (1, 2, 3)]

manifest = json.loads((ROOT / "data/shards_v2/build_manifest.json").read_text())
frames_by_session = {s["session_id"]: s["frames"] for s in manifest["sessions"]}
scale_minutes = []
for i in (1, 2, 3):
    sessions = (ROOT / f"data/shards_v2/train_scale{i}.txt").read_text().split()
    scale_minutes.append(sum(frames_by_session[s] for s in sessions) / 60.0 / 60.0)

baselines = json.loads((ROOT / "results/baselines.json").read_text())
val_a = next(b for b in baselines if b["shard"].startswith("rec_20260724_171305"))
luck = val_a["shuffled_event_f1_macro"]["0"]

# --------------------------------------------------------------- right panel data
# results/idm/SUMMARY.md, "Primary three-seed result" table (lines 39-54).
seeds = [0, 1, 2]
own_only = [0.0801, 0.0789, 0.0786]     # engine-truth-only exact F1
mapped = [0.0911, 0.0942, 0.0907]       # +13.45 h mapped NitroGen, zero-shot
own_mean, own_sd = 0.0792, 0.0008
map_mean, map_sd = 0.0920, 0.0019
mean_gain = 0.0128

# -------------------------------------------------------------------------- figure
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(8.8, 3.9), sharey=True,
    gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.10},
)
YMAX = 0.105

# Left: own-data scaling is flat just above the luck anchor.
ax1.plot(scale_minutes, scale_f1, "-o", color=style.ACCENT,
         linewidth=1.4, markersize=4.5, zorder=3)
for x, y in zip(scale_minutes, scale_f1):
    ax1.annotate(f"{y:.4f}", (x, y), xytext=(0, 7),
                 textcoords="offset points", ha="center",
                 fontsize=8, color=style.INK)
style.baseline_line(ax1, luck, f"shuffled-events luck {luck:.3f}")
ax1.annotate("2.7× more gold buys nothing",
             xy=(27.3, 0.0335), ha="center", fontsize=9.5,
             color=style.INK, style="italic")
ax1.set_xlim(11, 44)
ax1.set_xticks([15, 25, 40])
ax1.set_xlabel("own engine-truth data (min)")
ax1.set_ylabel("exact transition-event F1 (collar 0, val-A)")
ax1.set_ylim(0, YMAX)
ax1.set_title("",
              loc="left", fontsize=10.5)
ax1.text(0, 1.013, "16-frame non-causal ResNet-18, one seed per point",
         transform=ax1.transAxes, fontsize=7.5, color=style.INK_MUTED,
         va="bottom", ha="left")
ax1.set_title("")  # subtitle handled below; keep a single title block
ax1.text(0, 1.075, "Own gold only: 15 → 40 min",
         transform=ax1.transAxes, fontsize=11, color=style.INK,
         va="bottom", ha="left")

# Right: paired per-seed comparison; every seed improves.
XL, XR = 0.0, 1.0
for s, y0, y1 in zip(seeds, own_only, mapped):
    ax2.plot([XL, XR], [y0, y1], "-o", color=style.INK_MUTED,
             linewidth=1.0, markersize=3.6, zorder=3)
ax2.errorbar([XL - 0.13], [own_mean], yerr=[own_sd], fmt="o",
             color=style.ACCENT, markersize=5, capsize=3,
             linewidth=1.4, zorder=4)
ax2.errorbar([XR + 0.13], [map_mean], yerr=[map_sd], fmt="o",
             color=style.ACCENT, markersize=5, capsize=3,
             linewidth=1.4, zorder=4)
ax2.annotate(f"{own_mean:.4f}\n±{own_sd:.4f}", (XL - 0.13, own_mean),
             xytext=(-8, -4), textcoords="offset points", ha="right",
             va="center", fontsize=8, color=style.INK)
ax2.annotate(f"{map_mean:.4f}\n±{map_sd:.4f}", (XR + 0.13, map_mean),
             xytext=(8, 4), textcoords="offset points", ha="left",
             va="center", fontsize=8, color=style.INK)
ax2.annotate(f"+{mean_gain:.4f} mean gain — every seed improved",
             xy=(0.5, 0.101), ha="center", fontsize=9.5,
             color=style.INK, style="italic")
ax2.set_xlim(-0.55, 1.55)
ax2.set_xticks([XL, XR])
ax2.set_xticklabels(["engine-truth only", "+13.45 h mapped\nNitroGen (zero-shot)"])
ax2.set_ylim(0, YMAX)
ax2.text(0, 1.075, "Mapped foreign supervision, 3 paired seeds",
         transform=ax2.transAxes, fontsize=11, color=style.INK,
         va="bottom", ha="left")
ax2.text(0, 1.013, "32-frame frozen-feature GRU; seeds gray, mean ± SD blue",
         transform=ax2.transAxes, fontsize=7.5, color=style.INK_MUTED,
         va="bottom", ha="left")

out = style.save(fig, "fig_scaling")
print(f"wrote {out}")
print(f"left: minutes={['%.1f' % m for m in scale_minutes]} "
      f"f1={['%.4f' % v for v in scale_f1]} luck={luck:.4f}")
print(f"right: own={own_only} mapped={mapped}")
