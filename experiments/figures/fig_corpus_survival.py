# fig_corpus_survival: what survives of a public label corpus at re-fetch time.
#
# Data sources (verified against files, 2026-07-26):
#   data/dataset_card.md
#     - NitroGen Celeste release: 411 videos, 123,111 chunks, 684 nominal label-hours
#     - Availability census 2026-07-25: 245/411 alive; 14/14 YouTube alive,
#       231/397 Twitch alive (166 dead were Twitch VODs); 244 recovered
#       successfully carrying 213.0889 label-hours
#   results/provenance/census_alive.jsonl
#     - 397 censused rows, 231 alive / 166 dead (the Twitch-side census)
#   results/corpus_trainable.json
#     - trainable_label_hours: 192.9, trainable_video_ids: 221 entries
#       (matches the "221 videos" quoted in report/findings_log.md; 23 excluded)
#   results/idm/CORPUS_AUDIT.md
#     - strict durable feature-eligible corpus: 211 videos, 150.9167 label-hours;
#       ten videos of the 221-video pool (41.9833 h) absent from the durable
#       archive / rejected in validation
#
# Note: the stage-3 loss is 192.9 - 150.9167 = 42.0 h (per CORPUS_AUDIT.md),
# not the 13.5 h quoted in the figure request; the files win.

import sys

sys.path.insert(0, "experiments/figures")

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import style

style.apply()

# ---------------------------------------------------------------- data (fixed)
RELEASED_H = 684.0
ALIVE_H = 213.0889     # dataset_card.md (213.1 quoted)
TRAINABLE_H = 192.9    # corpus_trainable.json
STRICT_H = 150.9167    # CORPUS_AUDIT.md

STAGES = [
    dict(hours=RELEASED_H, videos=411, name="Released",
         sub="NitroGen Celeste labels"),
    dict(hours=ALIVE_H, videos=245, name="Alive at source",
         sub="census 2026-07-25"),
    dict(hours=TRAINABLE_H, videos=221, name="60 Hz-eligible",
         sub="per-chunk rate rule"),
    dict(hours=STRICT_H, videos=211, name="Durable + validated",
         sub="strict feature corpus"),
]

YT_ALIVE, YT_N = 14, 14
TW_ALIVE, TW_N = 231, 397

# Sequential single-hue (light -> dark) for the surviving mass; gray for losses.
SURVIVE = ["#c3d7ef", "#8db4e3", "#5b93d9", "#2a78d6"]
LOSS_GRAY = "#dbd9d2"

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(10.6, 5.0))
gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1.0], wspace=0.16,
                      left=0.065, right=0.975, top=0.845, bottom=0.24)
ax = fig.add_subplot(gs[0, 0])
axp = fig.add_subplot(gs[0, 1])

# ---- main panel: decay bar sequence (waterfall: gray cap = loss at that stage)
W = 0.62
xs = range(len(STAGES))
prev = None
for i, st in enumerate(STAGES):
    ax.bar(i, st["hours"], width=W, color=SURVIVE[i], edgecolor="white",
           linewidth=0.8, zorder=3)
    if prev is not None:
        ax.bar(i, prev - st["hours"], width=W, bottom=st["hours"],
               color=LOSS_GRAY, edgecolor="white", linewidth=0.8, zorder=3)
        # dotted carry-over connector at the previous survivor level
        ax.plot([i - 1 + W / 2, i - W / 2], [prev, prev],
                color=style.BASELINE, linestyle=":", linewidth=1.0, zorder=2)
    prev = st["hours"]

ax.set_xlim(-0.62, 3.62)
ax.set_ylim(0, 740)
ax.set_ylabel("Label-hours")
ax.set_xticks([])
ax.grid(axis="x", visible=False)

# loss annotation 1: inside the big link-rot block
ax.text(1, 505, "−470.9 h", ha="center", va="center",
        fontsize=10.5, fontweight="semibold", color=style.INK, zorder=4)
ax.text(1, 448, "link rot (Twitch retention)\n−69% of released hours",
        ha="center", va="center", fontsize=8.5, color=style.INK_MUTED, zorder=4)

# loss annotations 2 and 3: arrows down to the thin gray caps
ax.annotate("−20.2 h\nrate rule (per-chunk 60 Hz)",
            xy=(2, 208), xytext=(2, 330), ha="center", va="bottom",
            fontsize=8.5, color=style.INK_MUTED,
            arrowprops=dict(arrowstyle="-", color=style.BASELINE, linewidth=0.9))
ax.annotate("−42.0 h\ndurability / validation (10 videos)",
            xy=(3, 175), xytext=(3, 297), ha="center", va="bottom",
            fontsize=8.5, color=style.INK_MUTED,
            arrowprops=dict(arrowstyle="-", color=style.BASELINE, linewidth=0.9))

# headline: the survival fraction at source
ax.annotate("31% of the released hours\nsurvive at source",
            xy=(1 + W / 2, ALIVE_H), xytext=(1.62, 120), ha="left", va="center",
            fontsize=9.5, color=style.INK,
            arrowprops=dict(arrowstyle="-", color=style.BASELINE, linewidth=0.9))

# stat row under the axis: stage name / detail / hours / videos+percent
blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
for i, st in enumerate(STAGES):
    pct = st["hours"] / RELEASED_H * 100
    ax.text(i, -0.055, st["name"], transform=blend, ha="center", va="top",
            fontsize=9.5, color=style.INK)
    ax.text(i, -0.115, st["sub"], transform=blend, ha="center", va="top",
            fontsize=8, color=style.INK_MUTED)
    ax.text(i, -0.185, f"{st['hours']:.1f} h" if i else f"{st['hours']:.0f} h",
            transform=blend, ha="center", va="top",
            fontsize=10.5, fontweight="semibold", color=style.INK)
    ax.text(i, -0.26, f"{st['videos']} videos · {pct:.0f}%",
            transform=blend, ha="center", va="top",
            fontsize=8, color=style.INK_MUTED)

# ---- side panel: platform split at the census
axp.set_title("Alive at source, by platform", fontsize=9.5, color=style.INK,
              loc="left", pad=8)
frac_yt = YT_ALIVE / YT_N
frac_tw = TW_ALIVE / TW_N
axp.barh(1, frac_yt, height=0.52, color=style.ACCENT, edgecolor="white",
         linewidth=0.8, zorder=3)
axp.barh(0, frac_tw, height=0.52, color=style.ACCENT, edgecolor="white",
         linewidth=0.8, zorder=3)
axp.barh(0, 1 - frac_tw, height=0.52, left=frac_tw, color=LOSS_GRAY,
         edgecolor="white", linewidth=0.8, zorder=3)

axp.text(0.03, 1, f"{YT_ALIVE}/{YT_N} alive · 100%", ha="left",
         va="center", fontsize=8.5, color="white", zorder=4)
axp.text(0.03, 0, f"{TW_ALIVE}/{TW_N} alive · 58%", ha="left",
         va="center", fontsize=8.5, color="white", zorder=4)
axp.text(frac_tw + 0.03, 0, "166 dead", ha="left", va="center",
         fontsize=8.5, color=style.INK_MUTED, zorder=4)

axp.set_yticks([0, 1])
axp.set_yticklabels(["Twitch\nVODs", "YouTube"], fontsize=9, color=style.INK)
axp.set_ylim(-0.75, 1.75)
axp.set_xlim(0, 1)
axp.set_xticks([0, 0.5, 1.0])
axp.set_xticklabels(["0%", "50%", "100%"], fontsize=8)
axp.grid(axis="y", visible=False)
axp.tick_params(axis="y", length=0)

axp.text(0, -0.24, "Twitch VODs expire on a retention schedule;\n"
         "every censused death was a Twitch VOD.",
         transform=axp.transAxes, ha="left", va="top", fontsize=8,
         color=style.INK_MUTED)

# ---- titles and footnote
fig.text(0.065, 0.955, "What survives of a public label corpus at re-fetch time",
         ha="left", va="top", fontsize=12, fontweight="semibold",
         color=style.INK)
fig.text(0.065, 0.905,
         "NitroGen Celeste slice: released label-hours → alive at source "
         "→ rate-eligible → durable strict corpus",
         ha="left", va="top", fontsize=9, color=style.INK_MUTED)
fig.text(0.065, 0.015,
         "245 sources alive at census; 244 recovered successfully (one failed), "
         "carrying 213.1 of the 684 released label-hours.",
         ha="left", va="bottom", fontsize=7.5, color=style.INK_MUTED)

out = style.save(fig, "fig_corpus_survival")
print(out)
