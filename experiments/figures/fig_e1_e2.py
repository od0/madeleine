# fig_e1_e2: the E1 (causality) x E2 (blindfold) grid with the autocorrelation trap exposed.
#
# Data sources (all committed):
#   results/grid_summary.json
#       val_a.transition_f1_oracle_collar0.<arm>._macro  -> mean/sd bars (grid v3, 3 seeds,
#           val-A, oracle per-key thresholds, collar-0 transition-event F1, macro over 7 keys)
#       val_a.ap.e2_history._macro                       -> 0.933 per-frame AP for the leaky
#           history arm (the copy-keys[t-1] shortcut callout)
#   results/grid_v3/<arm>_s{0,1,2}.json
#       input_active_only.metrics.transition_f1_oracle[key].event.f1 -> per-seed dots
#       (recomputed macro is asserted to match grid_summary to 1e-9)
#   results/baselines.json  (val-A shard rec_20260724_171305_5min; grid_v3 runs list this
#       shard as their eval session)
#       shuffled_event_f1_macro["0"] = 0.0050  -> gray dashed luck anchor
#       persistence_ap_macro = 0.912, persistence_event_f1_macro["0"] = 0.000
#           -> cited in the caption (the per-frame autocorrelation trap)
#
# Message: at 40 training minutes the causal question is not answerable (all pixel arms
# within seed noise of each other and barely above shuffled-timing luck), and the apparent
# ungapped-history advantage is label leakage through time, not dynamics signal.

import json
import sys
from pathlib import Path

sys.path.insert(0, "experiments/figures")
import matplotlib.pyplot as plt  # noqa: E402

import style  # noqa: E402

style.apply()

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
KEYS = style.KEY_ORDER

# ---------------------------------------------------------------- load
summary = json.loads((RESULTS / "grid_summary.json").read_text())
grid = summary["val_a"]["transition_f1_oracle_collar0"]
leaky_ap = summary["val_a"]["ap"]["e2_history"]["_macro"][0]  # 0.933

baselines = json.loads((RESULTS / "baselines.json").read_text())
val_a_base = next(b for b in baselines if b["shard"] == "rec_20260724_171305_5min.npz")
luck = val_a_base["shuffled_event_f1_macro"]["0"]           # 0.0050
persist_ap = val_a_base["persistence_ap_macro"]             # 0.912 (caption)
persist_ev = val_a_base["persistence_event_f1_macro"]["0"]  # 0.000 (caption)


def per_seed_macro(arm: str) -> list[float]:
    vals = []
    for s in range(3):
        run = json.loads((RESULTS / "grid_v3" / f"{arm}_s{s}.json").read_text())
        m = run["input_active_only"]["metrics"]["transition_f1_oracle"]
        vals.append(sum(m[k]["event"]["f1"] for k in KEYS) / len(KEYS))
    return vals


ARMS_E1 = [
    ("e1_2frame", "2-frame"),
    ("e1_16frame_pastonly", "16-frame\npast-only"),
    ("e1_16frame_noncausal", "16-frame\nnon-causal"),
]
ARMS_E2 = [
    ("e1_16frame_noncausal", "pixels\nonly*"),
    ("e2_history", "history\nungapped"),
    ("e2_history_gap30", "history\ngap 0.5 s"),
    ("e2_pixels_history", "pixels+hist\nungapped"),
    ("e2_pixels_history_gap30", "pixels+hist\ngap 0.5 s"),
]
LEAKY = {"e2_history", "e2_pixels_history"}

for arm, _ in ARMS_E1 + ARMS_E2:
    seeds = per_seed_macro(arm)
    assert abs(sum(seeds) / 3 - grid[arm]["_macro"][0]) < 1e-9, arm

# ---------------------------------------------------------------- draw
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(9.6, 4.5), sharey=True,
    gridspec_kw={"width_ratios": [3, 5], "wspace": 0.05},
)

FILL = "#dfe9f8"      # light tint of the accent
EDGE = style.ACCENT
JITTER = (-0.10, 0.0, 0.10)


def draw_arms(ax, arms):
    tops = []
    for x, (arm, _) in enumerate(arms):
        mean = grid[arm]["_macro"][0]
        seeds = per_seed_macro(arm)
        hatch = "////" if arm in LEAKY else None
        ax.bar(x, mean, width=0.56, facecolor=FILL, edgecolor=EDGE,
               linewidth=1.1, hatch=hatch, zorder=2)
        ax.scatter([x + j for j in JITTER], seeds, s=15, color=style.INK,
                   edgecolor="white", linewidth=0.5, zorder=4)
        top = max(seeds + [mean])
        ax.text(x, top + 0.0016, f"{mean:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=style.INK, zorder=5)
        tops.append(top)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([lbl for _, lbl in arms], fontsize=9)
    ax.set_xlim(-0.65, len(arms) - 0.35)
    ax.grid(axis="x", visible=False)
    return tops


tops1 = draw_arms(ax1, ARMS_E1)
tops2 = draw_arms(ax2, ARMS_E2)

ax1.set_ylim(0, 0.054)
ax1.set_ylabel("Transition-event F1  (macro over 7 keys, collar 0)")
ax1.set_title("E1 — temporal context (pixels only)", loc="left", fontsize=10)
ax2.set_title("E2 — input modality (16-frame non-causal)", loc="left", fontsize=10)

# luck anchor: gray dashed in both panels, labeled once (right panel)
ax1.axhline(luck, color=style.BASELINE, linestyle="--", linewidth=1.2, zorder=1)
ax1.text(0.02, luck, f" shuffled-timing luck {luck:.3f}",
         transform=ax1.get_yaxis_transform(), ha="left", va="bottom",
         fontsize=8, color=style.BASELINE)
style.baseline_line(ax2, luck, f"shuffled-timing luck {luck:.3f}")

# E1: the future buys +0.0001 -- bracket past-only vs non-causal
d_future = grid["e1_16frame_noncausal"]["_macro"][0] - grid["e1_16frame_pastonly"]["_macro"][0]
yb = 0.0295
ax1.plot([1, 1, 2, 2], [yb - 0.0012, yb, yb, yb - 0.0012],
         color=style.INK_MUTED, linewidth=0.9, zorder=3)
ax1.text(1.5, yb + 0.0012, f"future context adds {d_future:+.4f}",
         ha="center", va="bottom", fontsize=8.5, color=style.INK)
ax1.text(1.5, 0.0455, "all three arms within seed noise:\nthe causal question is not\nanswerable at 40 train-min",
         ha="center", va="top", fontsize=8.5, color=style.INK_MUTED, linespacing=1.35)

# E2: the leak callout on the ungapped-history bar
ax2.annotate(
    "the “advantage” is label leakage:\nsame arm scores 0.933 per-frame AP\n"
    "— the copy-keys[t−1] shortcut.\ngapping history by 0.5 s closes it",
    xy=(1.08, 0.0405), xytext=(2.62, 0.0525),
    ha="center", va="top", fontsize=8.5, color=style.INK, linespacing=1.35,
    arrowprops=dict(arrowstyle="-", color=style.INK_MUTED, linewidth=0.9,
                    connectionstyle="arc3,rad=-0.18"),
)

fig.text(0.005, -0.045,
         "*same runs as the E1 16-frame non-causal arm.  Hatched bars: action history enters "
         "ungapped (frames t−15..t−1 include keys adjacent to the target).",
         fontsize=8, color=style.INK_MUTED)

fig.suptitle(
    "Grid v3, val-A, 3 seeds, oracle thresholds — nothing beats luck by much, except the leak",
    x=0.005, y=1.005, ha="left", fontsize=11.5,
)

out = style.save(fig, "fig_e1_e2")
print(f"wrote {out}")
print(f"luck={luck:.4f} persist_ap={persist_ap:.3f} persist_event_f1={persist_ev:.3f} leaky_ap={leaky_ap:.3f}")
for arm, _ in ARMS_E1 + ARMS_E2:
    m, s = grid[arm]["_macro"]
    print(f"{arm}: {m:.4f} +/- {s:.4f}  seeds={[round(v, 4) for v in per_seed_macro(arm)]}")
