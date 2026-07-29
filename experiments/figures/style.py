"""Shared figure style for MADELEINE report figures.

Every figure script imports this module so the whole set reads as one system:
one categorical palette (CVD-validated, fixed key order), one baseline-line
convention, one save path. Figures are static PNGs for GitHub rendering:
white surface, recessive grid, direct labels over legends where possible.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIGDIR = Path(__file__).resolve().parents[2] / "results" / "figures"

# Fixed key order and hues (validated: adjacent-pair CVD dE >= 8, normal >= 15).
# Color follows the key everywhere; never reassign by rank or subset.
KEY_ORDER = ["left", "right", "up", "down", "jump", "dash", "grab"]
KEY_COLORS = {
    "left": "#2a78d6",
    "right": "#eb6834",
    "up": "#1baf7a",
    "down": "#eda100",
    "jump": "#e87ba4",
    "dash": "#008300",
    "grab": "#4a3aa7",
}

INK = "#1a1a19"          # primary text/marks
INK_MUTED = "#6b6a63"    # secondary text, axis labels
GRID = "#e6e5df"         # recessive gridlines
BASELINE = "#8a8983"     # reference lines (chance/persistence/shuffle): gray, dashed
ACCENT = "#2a78d6"       # single-series accent when no key identity applies


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "text.color": INK,
    })


def baseline_line(ax, y: float, label: str, x_text: float = 0.99) -> None:
    """Reference line for a trivial baseline: gray dashed, small right-aligned label."""
    ax.axhline(y, color=BASELINE, linestyle="--", linewidth=1.2, zorder=1)
    ax.text(x_text, y, f" {label}", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color=BASELINE)


def save(fig, name: str) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"{name}.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out
