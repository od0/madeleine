"""Create deterministic summary figures for experiments E3 and E4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


BLUE = "#2563eb"
GRAY = "#6b7280"
RED = "#dc2626"


def _finish_axis(ax: plt.Axes) -> None:
    """Apply the shared, deliberately minimal axis styling."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def fig_e3(data: dict) -> Figure:
    """Build the E3 state-ambiguity and future-divergence figure."""
    eps_values = [0.25, 0.5, 1.0, 2.0, 4.0]
    eps_rates = [
        data["ambiguity_rate_vs_eps"][f"eps_{eps}"]
        for eps in eps_values
    ]
    horizons = [1, 2, 4, 8, 16]
    divergence = data["future_divergence_vs_horizon"]
    mean_l2 = [divergence[f"h_{h}"]["mean_L2"] for h in horizons]
    median_l2 = [divergence[f"h_{h}"]["median_L2"] for h in horizons]

    fig = plt.figure(figsize=(12, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(2, 2, height_ratios=(0.12, 1.0))
    header = fig.add_subplot(grid[0, :])
    ax_rate = fig.add_subplot(grid[1, 0])
    ax_div = fig.add_subplot(grid[1, 1])
    header.set_axis_off()
    header.text(
        0.5,
        0.78,
        "State ambiguity: fraction of active frames whose near-identical "
        "state was revisited with a different action",
        transform=header.transAxes,
        fontsize=11,
        ha="center",
        va="center",
    )
    header.text(
        0.5,
        0.15,
        f"n={data['total_active_frames']:,} active frames, "
        f"{data['rooms_analyzed']} rooms",
        transform=header.transAxes,
        color=GRAY,
        fontsize=9,
        ha="center",
        va="center",
    )

    ax_rate.set_facecolor("white")
    ax_rate.plot(
        eps_values,
        eps_rates,
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=6,
    )
    for eps, rate in zip(eps_values, eps_rates, strict=True):
        ax_rate.annotate(
            f"{rate:.1%}",
            (eps, rate),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=9,
            ha="center",
            va="bottom",
        )
    ax_rate.set_title("(a) Ambiguity rate vs. ε", fontsize=11)
    ax_rate.set_xlabel("ε (normalized-state radius)", fontsize=10)
    ax_rate.set_ylabel("ambiguity rate", fontsize=10)
    ax_rate.set_xscale("log", base=2)
    ax_rate.set_xticks(eps_values, ["0.25", "0.5", "1", "2", "4"])
    ax_rate.set_ylim(0.0, 1.0)
    ax_rate.minorticks_off()
    _finish_axis(ax_rate)

    ax_div.set_facecolor("white")
    ax_div.axvspan(8, 16, color=GRAY, alpha=0.1, linewidth=0)
    ax_div.plot(
        horizons,
        mean_l2,
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=6,
        label="mean L2",
    )
    ax_div.plot(
        horizons,
        median_l2,
        color=GRAY,
        linewidth=2,
        marker="o",
        markersize=6,
        label="median L2",
    )
    ax_div.text(
        11.3,
        0.94,
        "disambiguating\nhorizon",
        transform=ax_div.get_xaxis_transform(),
        color=GRAY,
        fontsize=9,
        ha="center",
        va="top",
    )
    ax_div.set_title("(b) Future divergence vs. horizon", fontsize=11)
    ax_div.set_xlabel("future horizon H (frames)", fontsize=10)
    ax_div.set_ylabel("normalized state L2 at t+H", fontsize=10)
    ax_div.set_xscale("log", base=2)
    ax_div.set_xticks(horizons, [str(h) for h in horizons])
    ax_div.minorticks_off()
    ax_div.legend(frameon=False, fontsize=9)
    _finish_axis(ax_div)

    return fig


def _macro_f1(value: float | dict) -> float:
    """Accept either the compact or expanded jitter result schema."""
    if isinstance(value, dict):
        return float(value["macro_f1"])
    return float(value)


def _profile_label(profile: str) -> str:
    """Turn a machine-readable transcode profile into a compact tick label."""
    parts = []
    for part in profile.split("_"):
        if part == "fullres":
            continue
        if part.endswith("M") and part[:-1].replace(".", "", 1).isdigit():
            part = f"{part}bps"
        elif part.endswith("k") and part[:-1].replace(".", "", 1).isdigit():
            part = f"{part}bps"
        parts.append(part)
    return " ".join(parts)


def fig_e4(data: dict) -> Figure:
    """Build the E4 acquisition-degradation figure."""
    timing_labels = ["clean", "jitter ±1f", "jitter ±2f", "jitter ±4f"]
    jitter = data["leg2_jitter"]
    timing_f1 = [
        float(data["leg1_clean"]["macro_f1"]),
        _macro_f1(jitter["shift_1"]),
        _macro_f1(jitter["shift_2"]),
        _macro_f1(jitter["shift_4"]),
    ]
    transcodes = data["leg3_transcode"]
    transcode_labels = [_profile_label(item["profile"]) for item in transcodes]
    transcode_f1 = [float(item["macro_f1"]) for item in transcodes]

    timing_x = list(range(len(timing_labels)))
    transcode_x = list(
        range(len(timing_labels), len(timing_labels) + len(transcodes))
    )

    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.5, zorder=1)
    ax.plot(
        timing_x,
        timing_f1,
        color=RED,
        linewidth=2,
        marker="o",
        markersize=7,
        label="timing offset",
        zorder=3,
    )
    ax.scatter(
        transcode_x,
        transcode_f1,
        color=BLUE,
        marker="s",
        s=52,
        label="transcode",
        zorder=3,
    )
    for x, value in zip(timing_x, timing_f1, strict=True):
        ax.annotate(
            f"{value:.3f}",
            (x, value),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=9,
            ha="center",
            va="bottom",
        )

    all_labels = timing_labels + transcode_labels
    ax.set_title(
        "Video quality costs nothing, timing costs 4.5% per frame",
        fontsize=11,
    )
    ax.set_ylabel("macro-F1", fontsize=10)
    ax.set_xlabel("acquisition condition", fontsize=10)
    ax.set_xticks(timing_x + transcode_x, all_labels, rotation=20, ha="right")
    ax.set_ylim(0.78, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _finish_axis(ax)

    return fig


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Input file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic summary figures for E3 and E4."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/figures"),
    )
    parser.add_argument(
        "--e3",
        type=Path,
        default=Path("results/e3_ambiguity.json"),
    )
    parser.add_argument(
        "--e4",
        type=Path,
        default=Path("results/e4_5min_full.json"),
    )
    args = parser.parse_args()

    e3_data = _read_json(args.e3)
    e4_data = _read_json(args.e4)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    e3_path = args.out_dir / "e3_ambiguity.png"
    e4_path = args.out_dir / "e4_degradation.png"
    e3_figure = fig_e3(e3_data)
    e4_figure = fig_e4(e4_data)
    e3_figure.savefig(
        e3_path,
        dpi=200,
        facecolor="white",
        metadata={"Software": "matplotlib"},
    )
    e4_figure.savefig(
        e4_path,
        dpi=200,
        facecolor="white",
        metadata={"Software": "matplotlib"},
    )
    plt.close(e3_figure)
    plt.close(e4_figure)

    print(e3_path)
    print(e4_path)


if __name__ == "__main__":
    main()
