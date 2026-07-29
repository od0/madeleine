# fig_architecture.py — the full build architecture: people, agents, local rig,
# cloud fleet, label channels, durable storage, and the flows between them.
#
# Every number shown is recorded in the repo: PROGRESS.md, infra notes,
# data/dataset_card.md, results/idm/*, harvest/WILD20.md, findings log.
# Agent-to-machine relationships are shown as tags on the boxes (border color
# identifies the actor); arrows are reserved for data flow. Regenerate with:
#   uv run python experiments/figures/fig_architecture.py

import sys

sys.path.insert(0, "experiments/figures")
import style  # noqa: E402

style.apply()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED, GRID = style.INK, style.INK_MUTED, style.GRID
BLUE = "#2a78d6"    # Claude sessions
ORANGE = "#eb6834"  # Codex sessions
VIOLET = "#4a3aa7"  # independent review
FILL = "#fafaf8"
GROUP = "#f1f0ec"

fig, ax = plt.subplots(figsize=(16, 10.6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10.6)
ax.axis("off")
ax.grid(False)


def group(x0, y0, x1, y1, label):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.06", linewidth=1.0,
                                edgecolor=GRID, facecolor=GROUP, zorder=1))
    ax.text(x0 + 0.12, y1 - 0.10, label, fontsize=9, fontweight="bold",
            color=MUTED, va="top", zorder=3)


def box(x0, y0, x1, y1, title, body, edge=INK, dashed=False, tag=None,
        tag_color=None):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.04", linewidth=1.2,
                                edgecolor=edge, facecolor=FILL,
                                linestyle=(0, (4, 3)) if dashed else "solid",
                                zorder=2))
    ax.text((x0 + x1) / 2, y1 - 0.14, title, fontsize=7.8, fontweight="bold",
            color=INK, ha="center", va="top", zorder=3)
    ax.text((x0 + x1) / 2, y1 - 0.40, body, fontsize=6.6, color=MUTED,
            ha="center", va="top", zorder=3, linespacing=1.35)
    if tag:
        ax.text(x1 - 0.10, y0 + 0.09, tag, fontsize=6.2, style="italic",
                color=tag_color or MUTED, ha="right", va="bottom", zorder=3)


def arrow(x0, y0, x1, y1, label=None, lx=None, ly=None, both=False):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="<->" if both else "->",
                                 mutation_scale=9, linewidth=1.0,
                                 color=MUTED, zorder=4, shrinkA=2, shrinkB=2))
    if label:
        ax.text(lx if lx is not None else (x0 + x1) / 2,
                ly if ly is not None else (y0 + y1) / 2 + 0.06,
                label, fontsize=6.2, color=MUTED, ha="center", zorder=4)


ax.text(0.15, 10.45, "MADELEINE build architecture",
        fontsize=13, fontweight="bold", color=INK, va="top")
ax.text(0.15, 10.12,
        "People, agents, machines, and data flow, 2026-07-23 → 26. "
        "Compute is disposable; data and evidence are durable (object storage + git). "
        "Box tags name the operator; border color identifies the actor.",
        fontsize=8.2, color=MUTED, va="top")

# ---------------------------------------------------------- people and agents
group(0.15, 8.05, 15.85, 9.85, "People and agents")
box(0.35, 8.20, 2.45, 9.45, "Bryan (human)",
    "plays and records\nreviews harvest gates\ncredentials, spend\nfinal claims")
box(2.65, 8.20, 5.35, 9.45, "Claude — orchestrator",
    "7 sessions\nplan, board, claims\nexperiment grids, report\nrecon + figure subagent fleets",
    edge=BLUE)
box(5.55, 8.20, 8.05, 9.45, "Claude — capture",
    "smaller model, checklist-bound\nrecords sessions\nflags, never fixes",
    edge=BLUE)
box(8.25, 8.20, 10.75, 9.45, "Independent review",
    "read-only audit\n6 recommendations adopted\nconfirmed 2 loader defects",
    edge=VIOLET)
box(10.95, 8.20, 13.35, 9.45, "Codex — trainer",
    "overnight training campaigns\n65 commits\ntransfer + capacity runs",
    edge=ORANGE)
box(13.55, 8.20, 15.65, 9.45, "Codex — harvest",
    "wild-overlay harvest pipeline\nfail-closed admission gates",
    edge=ORANGE)

# ------------------------------------------------------------ coordination bar
box(0.35, 7.10, 15.65, 7.80, "Git repository — the coordination surface",
    "PROGRESS.md board · session INDEX · evidence manifests · findings + engineering logs · "
    "pull before edit · small atomic commits · no work that is not on the board")
arrow(8.0, 8.20, 8.0, 7.80, both=True)
ax.text(8.35, 7.95, "every agent reads and writes here", fontsize=6.2,
        color=MUTED, ha="left", zorder=4)

# ------------------------------------------------------------------ middle row
group(0.15, 2.95, 4.85, 6.85, "Foreign label channels")
box(0.35, 4.95, 4.65, 6.45, "NitroGen release (HF)",
    "684 h Celeste labels released\n213.1 h alive at source (31%)\n192.9 h trainable after 60 Hz rule\n"
    "bind mapping + curation tiers\n13.45 h / 40.6 h / 150.9 h",
    tag="fetched by orchestrator lanes", tag_color=BLUE)
box(0.35, 3.10, 4.65, 4.75, "speedrun.com wild harvest",
    "7,071 videos / 6,757 h enumerated\n~15% carry an input HUD\nprobe → classify → polite fetch\n"
    "→ decode → gates\n0 h admitted so far (by design)",
    tag="pipeline: Codex-harvest · gates: Bryan", tag_color=ORANGE)

group(5.05, 2.95, 10.15, 6.85, "Local rig — MacBook")
box(5.25, 5.75, 9.95, 6.55, "Celeste + Everest + granny mod",
    "engine-truth CSV · frame-index strip · overlays, 60 Hz",
    tag="played by Bryan")
box(5.25, 4.80, 9.95, 5.55, "theo capture",
    "CFR 60 fps screen recording",
    tag="run by the capture session", tag_color=BLUE)
box(5.25, 3.90, 9.95, 4.60, "assembly + validation",
    "strip decode · drop/dup accounting · masks · manifests")
box(5.25, 3.05, 9.95, 3.70, "11 sessions → masked 128 px shards",
    "roles frozen per session: train / dev / diagnostic / test")

group(10.35, 2.95, 15.85, 6.85, "Cloud compute — disposable lanes")
box(10.55, 5.92, 15.65, 6.45, "single-GPU pods (retired)",
    "E1/E2/E3 grids v1–v3 · 3 seeds per arm", dashed=True,
    tag="orchestrator", tag_color=BLUE)
box(10.55, 5.05, 15.65, 5.78, "8×A100 lane node (terminated)",
    "60 fps corpus re-fetch, 221 GB · 7,071-video wild scan", dashed=True,
    tag="orchestrator", tag_color=BLUE)
box(10.55, 4.05, 15.65, 4.95, "2×A100 node (active, $3.30/h)",
    "Tier-B/C transfer · 36.9M / 113M end-to-end\nfeature builds · full-corpus pair in flight",
    tag="Codex-trainer", tag_color=ORANGE)
box(10.55, 3.10, 15.65, 3.92, "8 small CPU workers (~$0.85/h total)",
    "wild fetch + decode · one video per IP, polite rates",
    tag="Codex-harvest", tag_color=ORANGE)

# ------------------------------------------------------------------ bottom row
box(0.35, 1.10, 4.65, 2.30, "Results and evidence",
    "eval sidecars · prediction arrays\ncheckpoint hashes · run manifests\nfigures — all committed to git")
box(5.05, 1.10, 15.65, 2.30, "Cloudflare R2 — durable data home",
    "29,113 objects / 370 GB, byte-for-byte verified · corpus video 237 GB · shards · features · "
    "wild raw 44.7 GB\n~$5.55/month, free egress — any node rehydrates from here")

# ------------------------------------------------------------------ data flows
arrow(4.65, 5.70, 5.25, 5.35, "mapped labels", lx=4.35, ly=5.92)
arrow(2.5, 3.15, 6.6, 2.30, "raw media, byte-verified", lx=4.05, ly=2.42)
arrow(8.6, 3.10, 8.6, 2.30, "sessions, shards", lx=9.55, ly=2.58)
arrow(12.5, 3.10, 12.5, 2.30, "stage in / write back", lx=13.65, ly=2.58, both=True)
arrow(10.55, 2.87, 3.6, 2.32, "sidecars + hashes", lx=6.35, ly=2.72)

out = style.save(fig, "fig_architecture")
print(out)
