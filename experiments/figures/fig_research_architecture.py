# fig_research_architecture.py — the current MADELEINE research system:
# evidence channels, trust boundaries, durable state, and execution resources.
#
# This complements (rather than replaces) fig_architecture.py, which remains the
# dated 2026-07-23 → 26 build-sprint snapshot used by the project history.
# Tracked sources for the dated facts and architectural claims:
# - data/sessions/INDEX.md: 60 Hz capture and role-frozen session surfaces;
# - data/dataset_card.md, results/corpus_trainable.json, and the findings
#   log (private working repository): NitroGen funnel and the 2026-07-25
#   census;
# - README.md and PROGRESS.md: wild admission, as of 2026-07-28;
# - results/idm/untouched_test/UNTOUCHED_TEST.md: executed one-pass test;
# - infra/MACHINES.md: execution classes, durable storage, and lifecycle;
# - docs/PUBLIC_RELEASE.md and public_release/allowlist.json: export gate.
# Regenerate with:
#   uv run python experiments/figures/fig_research_architecture.py

import sys

sys.path.insert(0, "experiments/figures")
import style  # noqa: E402

style.apply()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED, GRID = style.INK, style.INK_MUTED, style.GRID
BLUE = "#2a78d6"     # data and evidence
ORANGE = "#eb6834"   # disposable execution resources
GREEN = "#1b8f68"    # validated / durable state
VIOLET = "#4a3aa7"   # human or independent review
FILL = "#fafaf8"
GROUP = "#f1f0ec"
SOFT_BLUE = "#f3f7fc"
SOFT_GREEN = "#f1f8f5"
SOFT_ORANGE = "#fff6f1"

fig, ax = plt.subplots(figsize=(16, 10.8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10.8)
ax.axis("off")
ax.grid(False)


def group(x0, y0, x1, y1, label):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.06", linewidth=1.0,
        edgecolor=GRID, facecolor=GROUP, zorder=1,
    ))
    ax.text(x0 + 0.14, y1 - 0.12, label, fontsize=9.2, fontweight="bold",
            color=MUTED, va="top", zorder=3)


def box(x0, y0, x1, y1, title, body, edge=INK, face=FILL, dashed=False,
        tag=None, tag_color=None, title_size=7.8, body_size=6.7):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.04", linewidth=1.2,
        edgecolor=edge, facecolor=face,
        linestyle=(0, (4, 3)) if dashed else "solid", zorder=2,
    ))
    ax.text((x0 + x1) / 2, y1 - 0.14, title, fontsize=title_size,
            fontweight="bold", color=INK, ha="center", va="top", zorder=3)
    ax.text((x0 + x1) / 2, y1 - 0.42, body, fontsize=body_size, color=MUTED,
            ha="center", va="top", zorder=3, linespacing=1.35)
    if tag:
        ax.text(x1 - 0.10, y0 + 0.09, tag, fontsize=6.2, style="italic",
                color=tag_color or MUTED, ha="right", va="bottom", zorder=3)


def arrow(x0, y0, x1, y1, label=None, lx=None, ly=None, both=False,
          color=MUTED, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="<->" if both else "->", mutation_scale=9,
        linewidth=1.05, color=color, zorder=4, shrinkA=2, shrinkB=2,
        connectionstyle=f"arc3,rad={rad}",
    ))
    if label:
        ax.text(lx if lx is not None else (x0 + x1) / 2,
                ly if ly is not None else (y0 + y1) / 2 + 0.06,
                label, fontsize=6.2, color=color, ha="center", zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})


ax.text(0.15, 10.65, "MADELEINE research architecture",
        fontsize=13.5, fontweight="bold", color=INK, va="top")
ax.text(
    0.15, 10.31,
    "Three evidence channels converge through explicit trust gates; durable state lives in object storage and git, "
    "while local and cloud execution is replaceable. Current through 2026-07-28.",
    fontsize=8.2, color=MUTED, va="top",
)

# ------------------------------------------------ governance and coordination
group(0.15, 8.48, 15.85, 10.02, "Governance and coordination")
box(0.35, 8.66, 3.25, 9.72, "Bryan — human authority",
    "records play · reviews wild gates\nspend and lifecycle approval · final claims",
    edge=VIOLET)
box(3.45, 8.66, 7.15, 9.72, "Agent workstreams",
    "orchestration · capture · training · harvest\nparallel research probes and reproducible handoffs",
    edge=ORANGE)
box(7.35, 8.66, 10.45, 9.72, "Independent review",
    "read-only audits · study-plan critique\ndefect confirmation before claims move",
    edge=VIOLET)
box(10.65, 8.66, 15.65, 9.72, "Git + evidence ledger",
    "plans · frozen configs · manifests · hashes\nreview decisions · compact results · public/private boundary",
    edge=GREEN, face=SOFT_GREEN)

# ------------------------------------------------------- evidence channels
group(0.15, 1.72, 4.35, 8.20, "Evidence channels")
box(0.35, 6.65, 4.15, 7.88, "Engine truth — local rig",
    "Celeste + Everest + granny at 60 Hz\nframe-index alignment · role-frozen captures\ntrain / dev / diagnostic / calibration / executed one-pass test",
    edge=BLUE, face=SOFT_BLUE, tag="highest-trust labels", tag_color=BLUE)
box(0.35, 5.15, 4.15, 6.38, "Mapped NitroGen labels",
    "released gamepad-overlay corpus\nsource recovery · controller masks · bind mapping\n192.9 historically eligible label-h (2026-07-25 census)",
    edge=BLUE, face=SOFT_BLUE, tag="curated foreign supervision", tag_color=BLUE)
box(0.35, 3.55, 4.15, 4.88, "Wild keyboard-overlay candidates",
    "speedrun.com discovery → YouTube / Twitch media\ndistributed source acquisition · layout surveys\nraw-complete candidates are not labels",
    edge=BLUE, face=SOFT_BLUE, tag="candidate media", tag_color=BLUE)
box(0.35, 1.95, 4.15, 3.20, "Wild admission boundary",
    "decode evidence → named human review\nmechanical publication · 1.5 h admitted as of 2026-07-28",
    edge=VIOLET, face=SOFT_GREEN, tag="fail closed", tag_color=VIOLET,
    body_size=6.5)
arrow(2.25, 3.55, 2.25, 3.20, "reviewable evidence", lx=3.04, ly=3.31)

# ----------------------------------------------------- durable model pipeline
group(4.55, 1.72, 9.75, 8.20, "Trust, data, and model pipeline")
box(4.75, 6.55, 9.55, 7.88, "Normalize + validate",
    "source binding · exact alignment / PTS · gap accounting\nanswer-key masks · curation and admission provenance",
    edge=GREEN, face=SOFT_GREEN)
box(4.75, 4.78, 9.55, 6.25, "Object storage — durable data home",
    "Cloudflare R2 in this build · content-addressed corpora\nraw / review artifacts · shards · features · checkpoints\nrehydration source of truth; machines hold replaceable caches",
    edge=GREEN, face=SOFT_GREEN)
box(4.75, 3.75, 9.55, 4.56, "Unified training + evaluation",
    "the same model and metric stack across label channels\nfrozen recipes · held-out splits · transition-aware scoring",
    edge=ORANGE, face=SOFT_ORANGE)
box(4.75, 2.89, 9.55, 3.50, "Results and evidence",
    "evaluation sidecars · predictions · run manifests · checkpoint hashes · figures",
    edge=GREEN, face=SOFT_GREEN, body_size=6.2, title_size=7.5)
box(4.75, 1.90, 9.55, 2.67, "Allowlisted export gate",
    "exact paths · SHA-256-pinned media · text / link / clean-clone checks\n→ one-root public repository",
    edge=VIOLET, face=FILL, body_size=6.1, title_size=7.4)

arrow(4.15, 7.26, 4.75, 7.26, "truth")
arrow(4.15, 5.76, 4.75, 7.05, "mapped", lx=4.39, ly=6.47)
arrow(4.15, 2.58, 4.75, 6.83, "admitted", lx=4.41, ly=4.68, rad=-0.08)
arrow(7.15, 6.55, 7.15, 6.25, "publish", lx=7.63, ly=6.36)
arrow(7.15, 4.78, 7.15, 4.56, "stage", lx=7.55, ly=4.59, both=True)
arrow(7.15, 3.75, 7.15, 3.50)
arrow(7.15, 2.89, 7.15, 2.67)

# ------------------------------------------------ disposable execution pool
group(9.95, 1.72, 15.85, 8.20, "Replaceable execution resources")
box(10.15, 6.75, 15.65, 7.88, "Local Mac",
    "capture rig · MPS / CPU probes and exact replay\nresidential fallback source lane",
    edge=ORANGE, face=SOFT_ORANGE)
box(10.15, 5.55, 15.65, 6.48, "A100 research lanes",
    "feature generation · independent training arms\nend-to-end scaling and frozen evaluations",
    edge=ORANGE, face=SOFT_ORANGE)
box(10.15, 4.30, 15.65, 5.28, "Distributed CPU + network lanes",
    "queue controller · per-IP source-health gates\nfetch · exact PTS · survey · decode · review packets",
    edge=ORANGE, face=SOFT_ORANGE)
box(10.15, 3.25, 15.65, 4.05, "Dedicated VLM triage",
    "A6000 + local Qwen · machine nomination only; never admission",
    edge=ORANGE, face=SOFT_ORANGE, body_size=6.5)
box(10.15, 1.95, 15.65, 3.00, "Retired burst and egress capacity",
    "A10 · A6000 · RTX 6000 Ada · H200 — short-lived source / survey lanes",
    edge=ORANGE, face=FILL, dashed=True, body_size=6.45,
    tag="not architectural dependencies", tag_color=ORANGE)

arrow(10.15, 7.30, 9.55, 7.30, "capture / replay", lx=9.82, ly=7.43)
arrow(10.15, 6.00, 9.55, 6.00, "stage / publish", lx=9.82, ly=6.13, both=True)
arrow(10.15, 4.79, 9.55, 5.04, "publish", lx=9.84, ly=5.08)

# ------------------------------------------------------ lifecycle principle
box(0.35, 0.55, 15.65, 1.35, "Resource lifecycle — a cross-cutting gate",
    "content-addressed queue → real source-health check → resumable work → completion-last publication "
    "→ readback verification → cost-state check → teardown; no machine retains unique state",
    edge=GREEN, face=SOFT_GREEN, title_size=8.0, body_size=6.8)

out = style.save(fig, "fig_research_architecture")
print(out)
