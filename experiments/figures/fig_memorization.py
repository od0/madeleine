# fig_memorization: train vs validation BCE — the memorization-without-transfer exhibit.
#
# Data sources (all committed):
#   runs_pod/e1_16frame_noncausal_s0/log.jsonl
#   runs_pod/e1_2frame_s0/log.jsonl
#       step, train_bce_per_key, val_bce_per_key (7 keys, nats) every 100 steps, 0..4000.
#       Macro = unweighted mean over style.KEY_ORDER.
#   runs_pod/*/run_meta.json
#       both runs train on the same three sessions and validate on
#       rec_20260724_171305_5min (checked below), so one baseline shard applies to both.
#   results/baselines.json  (shard rec_20260724_171305_5min)
#       chance_ap_per_key = per-key positive prevalence on the val session (a random
#       ranking's AP equals prevalence). Base-rate BCE floor = macro mean of the
#       Bernoulli entropy -(p ln p + (1-p) ln(1-p)) in nats -> 0.399.
#
# Message: every pixels model at this data scale memorizes its three training
# sessions. Val BCE bottoms at ~0.43 by step 200-300 — barely below the trivial
# base-rate floor — then climbs past 0.8 while train BCE falls to 0.05-0.07.

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "experiments/figures")
import matplotlib.pyplot as plt  # noqa: E402

import style  # noqa: E402

style.apply()

ROOT = Path(__file__).resolve().parents[2]
KEYS = style.KEY_ORDER
VAL_SHARD = "rec_20260724_171305_5min"


def load_run(name: str):
    rows = [json.loads(l) for l in (ROOT / "runs_pod" / name / "log.jsonl").read_text().splitlines()]
    meta = json.loads((ROOT / "runs_pod" / name / "run_meta.json").read_text())
    assert meta["split"]["val"] == [VAL_SHARD], name  # baseline shard must match
    steps = [r["step"] for r in rows]
    train = [sum(r["train_bce_per_key"][k] for k in KEYS) / len(KEYS) for r in rows]
    val = [sum(r["val_bce_per_key"][k] for k in KEYS) / len(KEYS) for r in rows]
    return steps, train, val


steps16, train16, val16 = load_run("e1_16frame_noncausal_s0")
steps2, train2, val2 = load_run("e1_2frame_s0")

baselines = json.loads((ROOT / "results" / "baselines.json").read_text())
shard = next(b for b in baselines if b["shard"] == f"{VAL_SHARD}.npz")
prev = shard["chance_ap_per_key"]  # random-ranking AP == prevalence
base_bce = sum(-(p * math.log(p) + (1 - p) * math.log(1 - p)) for p in prev.values()) / len(prev)

i16 = min(range(len(val16)), key=val16.__getitem__)
i2 = min(range(len(val2)), key=val2.__getitem__)

# ---------------------------------------------------------------- draw
fig, ax = plt.subplots(figsize=(8.4, 4.7))

VAL2 = "#9fc2e8"    # lighter tone of the accent for the secondary run
TRAIN2 = "#aaa9a2"  # lighter tone of ink

ax.plot(steps2, val2, color=VAL2, linewidth=1.2, zorder=2)
ax.plot(steps2, train2, color=TRAIN2, linewidth=1.2, zorder=2)
ax.plot(steps16, val16, color=style.ACCENT, linewidth=1.7, zorder=3)
ax.plot(steps16, train16, color=style.INK, linewidth=1.7, zorder=3)

# base-rate floor of the val session: gray dashed, labeled at the right edge
style.baseline_line(ax, base_bce, f"base-rate BCE {base_bce:.2f} (val prevalence)")

# mark the val minima
ax.scatter([steps16[i16]], [val16[i16]], s=18, color=style.ACCENT,
           edgecolor="white", linewidth=0.6, zorder=4)
ax.scatter([steps2[i2]], [val2[i2]], s=13, color=VAL2,
           edgecolor="white", linewidth=0.6, zorder=4)

# direct labels at the right ends (text is ink, never series-colored)
x_lab = steps16[-1] + 60
ax.text(x_lab, 0.875, f"val — 16-frame   {val16[-1]:.2f}", ha="left", va="center",
        fontsize=8.5, color=style.INK)
ax.text(x_lab, 0.800, f"val — 2-frame    {val2[-1]:.2f}", ha="left", va="center",
        fontsize=8.5, color=style.INK_MUTED)
ax.text(x_lab, 0.100, f"train — 16-frame  {train16[-1]:.2f}", ha="left", va="center",
        fontsize=8.5, color=style.INK)
ax.text(x_lab, 0.028, f"train — 2-frame   {train2[-1]:.2f}", ha="left", va="center",
        fontsize=8.5, color=style.INK_MUTED)

# the story, in two annotations
ax.annotate(
    f"val bottoms at step {steps16[i16]} ({val16[i16]:.2f}),\n"
    "barely past the base-rate floor,\nthen climbs for 3,700 more steps",
    xy=(steps16[i16] + 30, val16[i16] + 0.012), xytext=(430, 0.815),
    ha="left", va="top", fontsize=8.5, color=style.INK, linespacing=1.35,
    arrowprops=dict(arrowstyle="-", color=style.INK_MUTED, linewidth=0.9,
                    connectionstyle="arc3,rad=0.15"),
)
ax.annotate(
    "meanwhile train BCE falls to 0.05–0.07:\nthe three training sessions are memorized",
    xy=(2100, train16[21]), xytext=(2320, 0.345),
    ha="left", va="top", fontsize=8.5, color=style.INK, linespacing=1.35,
    arrowprops=dict(arrowstyle="-", color=style.INK_MUTED, linewidth=0.9,
                    connectionstyle="arc3,rad=-0.2"),
)

ax.set_xlim(0, steps16[-1])
ax.set_ylim(0, 0.95)
ax.set_xlabel("training step  (batch 64, eval every 100 steps)")
ax.set_ylabel("BCE, macro over 7 keys  (nats)")
ax.set_xticks(range(0, 4001, 1000))

fig.suptitle(
    "Pixels-only E1 runs, seed 0 — the generalization gap opens by step 300 and never closes",
    x=0.005, y=1.005, ha="left", fontsize=11.5,
)

out = style.save(fig, "fig_memorization")
print(f"wrote {out}")
print(f"base_rate_bce={base_bce:.4f}")
print(f"16-frame: val_min={val16[i16]:.4f}@{steps16[i16]}  val_final={val16[-1]:.4f}  train_final={train16[-1]:.4f}")
print(f"2-frame:  val_min={val2[i2]:.4f}@{steps2[i2]}  val_final={val2[-1]:.4f}  train_final={train2[-1]:.4f}")
