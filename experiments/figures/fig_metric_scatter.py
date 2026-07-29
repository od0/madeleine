"""fig_metric_scatter — per-frame accuracy vs collar-0 transition-event F1:
the metric-decoupling exhibit.

One dot per model on the val-A development session (rec_20260724_171305_5min,
25,028 input-active frames). x is per-key micro key-state accuracy at the
untuned 0.5 threshold; y is exact (collar-0) transition-event F1, macro over
the seven keys, oracle per-key thresholds fit on val-A (the same quantity as
fig_scaling / fig_e1_e2). The one-frame persistence copy rule tops accuracy
at 98.95% while scoring exactly 0.000 on collar-0 events; the learned models
lose 20-40 accuracy points to it yet are the only points above shuffled-events
luck on timing.

Data sources (verified against these files, not restated from memory):
  results/idm/KEYPRESS_ACCURACY.md
      canonical accuracy inventory, threshold 0.5, input_active surface:
      - own_features_32nc (matched clean-only frozen features, selected):
        73.80 / 76.48 / 78.73 % micro (listed seed order s0,s1,s2 -- the
        same listing convention the Tier-B seed list uses, which is
        verified below against retained npz sidecars)
      - foreign_tier_b_13p45h_32nc (selected): 61.53 / 61.41 / 58.36 %
      - 36.9M end-to-end val-A: 67.37 %; 112.95M end-to-end val-A: 67.62 %
      - always released 82.85 % micro; persistence 98.95 % micro (93.32 %
        joint), constructed before the activity gate
  results/idm/<run>_val_a.json  (selected-checkpoint eval sidecars)
      input_active_only.metrics.transition_f1_oracle[key].event.f1
      -> macro over style.KEY_ORDER = the "Exact event F1" column of
         results/idm/SUMMARY.md (0.0801 own s0, 0.0911 tier-B s0,
         0.0764 36.9M, 0.0810 113M, ...)
  results/idm/<run>_val_a_preds.npz + experiments/keypress_accuracy.py
      (five retained sidecars) -> asserts the documented accuracies and
      baseline figures above, and the seed listing order
  results/baselines.json  (val-A shard rec_20260724_171305_5min)
      shuffled_event_f1_macro["0"] = 0.0050  -> gray dashed luck anchor
      persistence_event_f1_macro["0"] = 0.000
  VPT reported 90.6% keypress accuracy at 1,962 h: VPT paper Fig. 3 as
      transcribed in KEYPRESS_ACCURACY.md; aggregation/threshold/denominator
      unpublished, 20 buttons vs 7 controls -> context band, not comparison.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "experiments/figures")
sys.path.insert(0, ".")  # repo root, for experiments.keypress_accuracy
import matplotlib.pyplot as plt  # noqa: E402

import style  # noqa: E402

style.apply()

ROOT = Path(__file__).resolve().parents[2]
TAKE = ROOT / "results" / "takeover"
KEYS = style.KEY_ORDER

# ---------------------------------------------------------------- accuracy (x)
# Documented in results/idm/KEYPRESS_ACCURACY.md (micro %, threshold 0.5,
# input_active, selected checkpoints, val-A).
ACC = {
    "own_features_32nc_s0": 73.80,
    "own_features_32nc_s1": 76.48,
    "own_features_32nc_s2": 78.73,
    "foreign_tier_b_13p45h_32nc_s0": 61.53,
    "foreign_tier_b_13p45h_32nc_s1": 61.41,
    "foreign_tier_b_13p45h_32nc_s2": 58.36,
    "foreign_tier_b_13p45h_37m_e2e_aug_32nc_s0": 67.37,
    "foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0": 67.62,
}
ALWAYS_RELEASED = 82.85
PERSISTENCE = 98.95
VPT_REPORTED = 90.6

# Verify every documented number that still has a retained prediction sidecar.
from experiments.keypress_accuracy import score_sidecar  # noqa: E402

for run, documented in ACC.items():
    npz = TAKE / f"{run}_val_a_preds.npz"
    if not npz.exists():
        continue  # own_features npz were not retained; doc numbers stand
    rep = score_sidecar(npz)
    assert abs(rep["key_state_micro_accuracy"] * 100 - documented) < 0.005, run
    assert abs(rep["always_released_key_state_micro_accuracy"] * 100
               - ALWAYS_RELEASED) < 0.005, run
    assert abs(rep["persistence_key_state_micro_accuracy"] * 100
               - PERSISTENCE) < 0.005, run

# ----------------------------------------------------- transition-event F1 (y)
def macro_event_f1(run: str) -> float:
    sidecar = json.loads((TAKE / f"{run}_val_a.json").read_text())
    m = sidecar["input_active_only"]["metrics"]["transition_f1_oracle"]
    assert all(m[k]["collar"] == 0 for k in KEYS), run
    return sum(m[k]["event"]["f1"] for k in KEYS) / len(KEYS)


F1 = {run: macro_event_f1(run) for run in ACC}

baselines = json.loads((ROOT / "results" / "baselines.json").read_text())
val_a = next(b for b in baselines if b["shard"] == "rec_20260724_171305_5min.npz")
LUCK = val_a["shuffled_event_f1_macro"]["0"]                 # 0.0050
assert val_a["persistence_event_f1_macro"]["0"] == 0.0

# Spearman rank correlation between the two metrics across the 8 learned
# models (no ties in either metric, so the classic d^2 formula is exact).
def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for rank, i in enumerate(order, start=1):
        ranks[i] = rank
    return ranks


_runs = list(ACC)
_ra, _rf = _ranks([ACC[r] for r in _runs]), _ranks([F1[r] for r in _runs])
_n = len(_runs)
SPEARMAN = 1 - 6 * sum((a - f) ** 2 for a, f in zip(_ra, _rf)) / (_n * (_n**2 - 1))

COHORTS = {
    "own": [f"own_features_32nc_s{s}" for s in range(3)],
    "tier_b": [f"foreign_tier_b_13p45h_32nc_s{s}" for s in range(3)],
    "e2e_37m": ["foreign_tier_b_13p45h_37m_e2e_aug_32nc_s0"],
    "e2e_113m": ["foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0"],
}

# ---------------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(8.8, 4.8))

# VPT context band: reported number, unusable as a comparison.
ax.axvspan(VPT_REPORTED - 0.4, VPT_REPORTED + 0.4, color=style.GRID,
           alpha=0.55, zorder=0, linewidth=0)
ax.text(VPT_REPORTED, 0.1095, "VPT reported 90.6%\n(different game/keys;\nmetric reading undefined)",
        ha="center", va="top", fontsize=7.5,
        color=style.INK_MUTED, linespacing=1.45)

# Luck anchor for the y-quantity.
ax.axhline(LUCK, color=style.BASELINE, linestyle="--", linewidth=1.2, zorder=1)
ax.text(0.012, LUCK, f" shuffled-events luck {LUCK:.3f}",
        transform=ax.get_yaxis_transform(), ha="left", va="bottom",
        fontsize=8, color=style.BASELINE)

# Trivial copy rules: gray X markers on the event-F1 floor.
ax.scatter([ALWAYS_RELEASED, PERSISTENCE], [0.0, 0.0], marker="x",
           s=55, color=style.BASELINE, linewidth=1.6, zorder=4)
ax.annotate("always released\n82.85%, F1 0", (ALWAYS_RELEASED, 0.0),
            xytext=(0, 17), textcoords="offset points", ha="center",
            va="bottom", fontsize=8, color=style.BASELINE, linespacing=1.4)
ax.annotate("one-frame persistence\n98.95%, F1 0.000\n(every transition\none frame late)",
            (PERSISTENCE, 0.0), xytext=(2, 17), textcoords="offset points",
            ha="right", va="bottom", fontsize=8, color=style.BASELINE,
            linespacing=1.4)

# Learned models.
def pts(cohort):
    return ([ACC[r] for r in COHORTS[cohort]],
            [F1[r] for r in COHORTS[cohort]])


x, y = pts("tier_b")
ax.scatter(x, y, s=34, color=style.ACCENT, zorder=5)
x, y = pts("own")
ax.scatter(x, y, s=34, facecolor="white", edgecolor=style.ACCENT,
           linewidth=1.3, zorder=5)
x, y = pts("e2e_37m")
ax.scatter(x, y, s=40, marker="D", color=style.INK, zorder=5)
x, y = pts("e2e_113m")
ax.scatter(x, y, s=40, marker="D", facecolor="white",
           edgecolor=style.INK, linewidth=1.3, zorder=5)

# Direct cohort labels.
ax.text(62.0, 0.0985, "0.725M frozen features\nTier-B mapped, zero-shot (3 seeds)",
        ha="center", va="bottom", fontsize=8.5, color=style.INK, linespacing=1.4)
ax.text(76.3, 0.0715, "0.725M frozen features\nengine-truth only (3 seeds)",
        ha="center", va="top", fontsize=8.5, color=style.INK, linespacing=1.4)
ax.annotate("36.9M end-to-end", (ACC["foreign_tier_b_13p45h_37m_e2e_aug_32nc_s0"],
                                 F1["foreign_tier_b_13p45h_37m_e2e_aug_32nc_s0"]),
            xytext=(-6, -11), textcoords="offset points", ha="center",
            fontsize=8.5, color=style.INK)
ax.annotate("113M end-to-end", (ACC["foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0"],
                                F1["foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0"]),
            xytext=(0, 8), textcoords="offset points", ha="center",
            fontsize=8.5, color=style.INK)

# Headline extremes.
best_f1_run = max(F1, key=F1.get)
ax.annotate(f"{F1[best_f1_run]:.4f}", (ACC[best_f1_run], F1[best_f1_run]),
            xytext=(8, 2), textcoords="offset points", ha="left",
            va="center", fontsize=8, color=style.INK)
best_acc_run = max(ACC, key=ACC.get)
ax.annotate(f"{ACC[best_acc_run]:.1f}%", (ACC[best_acc_run], F1[best_acc_run]),
            xytext=(8, 2), textcoords="offset points", ha="left",
            va="center", fontsize=8, color=style.INK)

# The message.
ax.text(55.5, 0.050,
        "the copy rule wins accuracy outright\nand scores zero on exact events;\n"
        "among learned models, the accuracy\norder roughly inverts the timing order\n"
        f"(Spearman ρ = {SPEARMAN:+.2f}, n = {_n})",
        ha="left", va="top", fontsize=9,
        color=style.INK, style="italic", linespacing=1.55)

ax.set_xlim(54, 101)
ax.set_ylim(-0.0045, 0.112)
ax.set_xticks(range(55, 101, 5))
ax.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
ax.set_xlabel("per-key micro key-state accuracy at 0.5 threshold (%)")
ax.set_ylabel("exact transition-event F1 (collar 0, macro)")
ax.text(0, 1.075, "State accuracy and event timing measure different skills",
        transform=ax.transAxes, fontsize=11.5, color=style.INK,
        va="bottom", ha="left")
ax.text(0, 1.018, "val-A development session, 25,028 input-active frames; "
        "y uses oracle per-key thresholds fit on val-A, x the untuned 0.5 rule",
        transform=ax.transAxes, fontsize=7.5, color=style.INK_MUTED,
        va="bottom", ha="left")

out = style.save(fig, "fig_metric_scatter")
print(f"wrote {out}")
for run in ACC:
    print(f"{run}: acc={ACC[run]:.2f}% f1={F1[run]:.4f}")
print(f"anchors: always-released ({ALWAYS_RELEASED}%, 0.0) "
      f"persistence ({PERSISTENCE}%, 0.000) luck={LUCK:.4f} vpt_band={VPT_REPORTED}%")
print(f"spearman(acc, f1) over learned models = {SPEARMAN:+.3f} (n={_n})")
