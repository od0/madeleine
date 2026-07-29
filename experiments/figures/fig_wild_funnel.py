# fig_wild_funnel: the wild-harvest admission funnel that ends at zero.
#
# Every number below was re-measured against its source artifact before drawing
# (values hard-coded so the figure reproduces from the repo alone; the wild20
# worktree is read-only and lives outside this repo):
#
#   Stage 1  results/wild/candidates.jsonl
#            7,071 lines; sum(duration_s)/3600 = 6,757.5 video-hours.
#   Stage 2  results/wild/style_labels.json
#            55 probed frames; 9 show an input overlay (16.4%), of which 8 are
#            keyboard-channel (opaque key grid / translucent action HUD /
#            on-screen keyboard) and 1 is a gamepad display. 8/55 = 14.5% ~ 15%
#            -> 7,071 * 8/55 ~ 1,028 videos, 6,757.5 * 8/55 = 982.9 h (estimate,
#            drawn hatched: extrapolation, not a measurement).
#   Stage 3  harvest/wild20_tranche.json in the private acquisition workspace
#            frozen_queue_video_count = 11; frozen_queue_nominal_hours = 27.4016.
#   Stage 4  results/wild20/raw/*/fetch.json in that workspace
#            (upload_complete.json confirms SHA-256 read-back of every object to
#            object-store:example-bucket/wild/v1/raw/<id>/); the sum of
#            media.duration_s/3600 over the
#            11 videos = 33.2401 media hours. Exceeds nominal because Twitch VODs
#            carry pre/post-run content and leaderboard times are loadless.
#   Stage 5  ai-v3 boundary proposals inside decode_report.json for
#            {Y6AeZFCU4LY, nRMVyWdNsTo, ofy37Fm6EgI, v1509603803}:
#            sum(gameplay_allowed_hours) = 1.7360 + 3.7490 + 4.2740 + 0.5999
#            = 10.359 h (brief said 10.4; measured 10.36). reviewer_kind =
#            "ai_agent", human_reviewed = false on every proposal.
#   Stage 6  results/wild20/ in the private acquisition workspace:
#            {v1509603803,nRMVyWdNsTo}/provisional-ai-v3-381b380/decode_report.json
#            decoded_hours = 0.80000 + 4.64491 = 5.4449 h.
#   Stage 7  every decode_report.json: admitted = false, admitted_hours = 0.0;
#            rejection_reasons are uniformly the pending human sign-offs
#            (compositor offset unmeasured, boundaries not human-reviewed,
#            no hash-bound layout acceptance). No layout_acceptance / offset
#            acceptance artifact exists anywhere under results/wild20/.

import sys

sys.path.insert(0, "experiments/figures")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import style

style.apply()

# (name, sub-annotation, hours, printed value, kind)
STAGES = [
    ("Fresh PC candidates enumerated",
     "7,071 videos · speedrun.com leaderboards",
     6757.5, "6,757 h", "measured"),
    ("Carrying an input display",
     "estimate: 8/55 style-survey frames keyboard-visible (≈15%) → ~1,000 videos",
     982.9, "~980 h est.", "estimate"),
    ("Frozen first tranche",
     "11 videos · wild20_tranche.json, frozen queue",
     27.4016, "27.4 h nominal", "measured"),
    ("Fetched + byte-verified to R2",
     "11/11 videos · SHA-256 read-back · exceeds nominal: VOD padding, loadless timing",
     33.2401, "33.2 h media", "measured"),
    ("AI-proposed gameplay windows",
     "ai-v3 boundary proposals · 4 videos · not human-reviewed",
     10.359, "10.4 h", "measured"),
    ("Provisionally decoded",
     "2 videos (v1509603803, nRMVyWdNsTo) · AI layout + AI boundaries, unreviewed",
     5.4449, "5.44 h", "measured"),
    ("Admitted / train-ready", None, 0.0, None, "zero"),
]

XMIN, XMAX = 0.5, 30000.0
BAR_H = 0.36

fig, ax = plt.subplots(figsize=(9.8, 5.6))

n = len(STAGES)
for i, (name, sub, hours, value, kind) in enumerate(STAGES):
    y = n - 1 - i  # top row first

    if kind == "zero":
        # No bar can be drawn at zero on any scale; the annotation is the mark.
        ax.text(XMIN * 1.04, y + 0.30, name, fontsize=9.5, fontweight="bold",
                color=style.INK, va="bottom")
        ax.text(XMIN * 1.04, y, "0 h admitted — every gate requires human "
                "sign-off (fail-closed)",
                fontsize=11, fontweight="bold", color=style.INK, va="center")
        ax.text(XMIN * 1.04, y - 0.40,
                "pending sign-offs: HUD compositor-offset measurement · "
                "gameplay-boundary review · hash-bound layout acceptance",
                fontsize=7.5, color=style.INK_MUTED, va="center")
        continue

    if kind == "estimate":
        ax.barh(y, hours - XMIN, left=XMIN, height=BAR_H,
                facecolor="#cfe0f5", edgecolor=style.ACCENT,
                linewidth=0.9, linestyle=(0, (3, 2)), zorder=3)
    else:
        ax.barh(y, hours - XMIN, left=XMIN, height=BAR_H,
                facecolor=style.ACCENT, edgecolor="none", zorder=3)

    label = name + (" (estimate)" if kind == "estimate" else "")
    ax.text(XMIN * 1.04, y + 0.30, label, fontsize=9.5, fontweight="bold",
            color=style.INK, va="bottom")
    if sub:
        ax.text(hours * 1.10, y - 0.26, sub, fontsize=7.5,
                color=style.INK_MUTED, va="center", ha="left",
                clip_on=False) if False else None
    # value at the bar end, sub-annotation under the stage name
    ax.text(hours * 1.12, y, value, fontsize=9.5, fontweight="bold",
            color=style.INK, va="center")
    if sub:
        ax.text(XMIN * 1.04, y - 0.32, sub, fontsize=7.5,
                color=style.INK_MUTED, va="center")
    if name == "Provisionally decoded":
        ax.text(hours * 1.12, y - 0.32, "0.08% of enumerated hours",
                fontsize=7.5, color=style.INK_MUTED, va="center")

ax.set_xscale("log")
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(-0.85, n - 0.15)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.xaxis.set_major_locator(mticker.FixedLocator([1, 10, 100, 1000, 10000]))
ax.xaxis.set_major_formatter(mticker.FixedFormatter(
    ["1", "10", "100", "1,000", "10,000"]))
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlabel("hours (log scale — raw values printed at every stage)")
ax.set_title("Wild-harvest admission funnel — 6,757 candidate hours in, "
             "zero admitted", loc="left", fontweight="bold", pad=12)

fig.text(0.01, -0.02,
         "Hatched bar is a survey extrapolation, not a measurement. AI-proposed "
         "windows and provisional decodes carry no human review; admission is "
         "fail-closed, so train-ready hours remain zero until each sign-off "
         "exists as a hash-bound artifact.",
         fontsize=7.5, color=style.INK_MUTED, ha="left", va="top")

out = style.save(fig, "fig_wild_funnel")
print(out)
