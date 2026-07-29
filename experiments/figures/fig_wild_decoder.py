"""Render README-ready exhibits from the reviewed wild-overlay decoder packet.

The input packet is deliberately not part of the public export: it contains
source-bound frame evidence from a public gameplay video.  This script emits
three compact fair-use research exhibits into ``results/figures``:

* ``fig_wild_decoder_layout.png``: model crop, HUD mask, and sample cells;
* ``fig_wild_decoder_states.png``: three reviewed released/pressed pairs; and
* ``fig_wild_decoder_sequence.png``: exact-PTS evidence for U -> dash.

Pass the private evidence root explicitly.  For example::

    uv run python experiments/figures/fig_wild_decoder.py \
        --evidence-root /path/to/private/results/wild20

The renderer verifies that it is reading the human-reviewed, still-unmeasured
layout.  These figures demonstrate geometry and semantic decoding; they do not
claim a measured compositor offset or train-ready admission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / "results" / "figures"
VIDEO_ID = "v498642684"
SOURCE_URL = "https://www.twitch.tv/videos/498642684"

INK = "#1a1a19"
INK_MUTED = "#6b6a63"
SURFACE = "#ffffff"
CARD = "#f5f5f2"
GRID = "#deddd7"
CYAN = "#00b8d4"
MAGENTA = "#d81b9c"
GREEN = "#169c55"
GREEN_PALE = "#e3f5ea"
GRAY_PALE = "#eeeeea"
BLUE = "#2a78d6"

FRAME_WH = (1280, 720)
GAMEPLAY_XYXY = (0, 0, 1058, 594)
PANEL_XYXY = (974, 594, 1280, 720)


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    size: int,
    *,
    color: str = INK,
    bold: bool = False,
    anchor: str | None = None,
) -> None:
    draw.text(xy, text, fill=color, font=font(size, bold=bold), anchor=anchor)


def source_rect(normalized: list[float]) -> tuple[int, int, int, int]:
    x, y, w, h = normalized
    return (
        round(x * FRAME_WH[0]),
        round(y * FRAME_WH[1]),
        round((x + w) * FRAME_WH[0]),
        round((y + h) * FRAME_WH[1]),
    )


def resize_to(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height), Image.Resampling.LANCZOS)


def rounded_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = CARD,
    outline: str = GRID,
    radius: int = 18,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def badge(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: str,
    color: str = INK,
    size: int = 20,
) -> tuple[int, int, int, int]:
    fnt = font(size, bold=True)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=fnt)
    w = right - left + 28
    h = bottom - top + 18
    box = (xy[0], xy[1], xy[0] + w, xy[1] + h)
    draw.rounded_rectangle(box, radius=h // 2, fill=fill)
    draw.text((xy[0] + 14, xy[1] + 7), text, fill=color, font=fnt)
    return box


def wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    size: int,
    *,
    bold: bool = False,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    fnt = font(size, bold=bold)
    for word in words:
        proposal = f"{current} {word}".strip()
        if draw.textbbox((0, 0), proposal, font=fnt)[2] <= max_width:
            current = proposal
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def save_png(image: Image.Image, name: str) -> Path:
    """Save a compact, GitHub-friendly PNG below the release size ceiling."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    image.save(path, optimize=True, compress_level=9)
    if path.stat().st_size > 2_000_000:
        adaptive = image.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        adaptive.save(path, optimize=True, compress_level=9)
    return path


def load_packet(evidence_root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    case = evidence_root / VIDEO_ID
    packet = case / "review_packet"
    layout = json.loads((packet / "layout.reviewed-unmeasured.json").read_text())
    cells = json.loads((packet / "cell_states.json").read_text())
    if layout.get("video_id") != VIDEO_ID or cells.get("video_id") != VIDEO_ID:
        raise ValueError("unexpected evidence video")
    if layout.get("human_reviewed") is not True:
        raise ValueError("renderer requires the reviewed layout")
    if layout.get("temporal_offset_source") != "unmeasured":
        raise ValueError("figure copy assumes the offset remains unmeasured")
    if len(layout.get("cells", [])) != 7 or len(cells.get("cells", [])) != 7:
        raise ValueError("expected seven canonical actions")
    return case, layout, cells


def draw_sample_on_panel(
    panel: Image.Image,
    normalized_rect: list[float],
    *,
    color: str = GREEN,
    width: int = 5,
) -> Image.Image:
    panel = panel.copy()
    draw = ImageDraw.Draw(panel)
    x0, y0, x1, y1 = source_rect(normalized_rect)
    draw.rectangle(
        (x0 - PANEL_XYXY[0], y0 - PANEL_XYXY[1],
         x1 - PANEL_XYXY[0], y1 - PANEL_XYXY[1]),
        outline=color,
        width=width,
    )
    return panel


def render_layout(case: Path, layout: dict[str, Any]) -> Path:
    canvas = Image.new("RGB", (1800, 1130), SURFACE)
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (38, 24), "From a public speedrun to model-safe pixels", 48, bold=True)
    draw_text(
        draw,
        (40, 84),
        "A reviewed packet separates the gameplay viewport, the answer-key HUD, and the seven cells used to recover actions.",
        23,
        color=INK_MUTED,
    )

    geometry = Image.open(case / "review_packet" / "geometry.png").convert("RGB")
    geometry_body = geometry.crop((0, 64, 1280, 784))
    geometry_body = resize_to(geometry_body, 1360, 765)
    canvas.paste(geometry_body, (38, 150))
    draw.rectangle((38, 150, 1398, 915), outline=GRID, width=2)

    rounded_card(draw, (1430, 150, 1762, 915), fill="#f3f5f6")
    draw_text(draw, (1455, 174), "Decoder view", 28, bold=True)

    source = Image.open(case / "review_packet" / "frames" / "u-pressed-f783126.png").convert("RGB")
    panel = source.crop(PANEL_XYXY)
    for cell in layout["cells"]:
        panel = draw_sample_on_panel(panel, cell["sample_rect"], width=3)
    panel = resize_to(panel, 282, 116)
    canvas.paste(panel, (1455, 222))
    draw.rectangle((1455, 222, 1737, 338), outline=GREEN, width=3)
    draw_text(draw, (1455, 350), "Seven tiny luma probes; the rest of the HUD is ignored.", 17, color=INK_MUTED)

    legend = [
        (CYAN, "MODEL VIEW", "gameplay crop"),
        (MAGENTA, "MASKED", "keyboard + stream furniture"),
        (GREEN, "DECODED", "seven action cells"),
    ]
    y = 424
    for color, label, detail in legend:
        draw.rounded_rectangle((1455, y, 1483, y + 28), radius=6, fill=color)
        draw_text(draw, (1497, y - 1), label, 19, bold=True)
        draw_text(draw, (1497, y + 24), detail, 17, color=INK_MUTED)
        y += 82

    draw.line((1455, 672, 1737, 672), fill=GRID, width=2)
    draw_text(draw, (1455, 694), "Reviewed mapping", 22, bold=True)
    mapping = ("A / D / W / S  →  directions\n"
               "Space  →  jump\nU  →  dash\nI  →  grab")
    draw.multiline_text((1455, 734), mapping, fill=INK, font=font(19), spacing=13)
    badge(draw, (1455, 855), "human approved", fill=GREEN_PALE, color=GREEN, size=17)

    draw_text(
        draw,
        (40, 958),
        "What this proves: stable geometry and semantic key mapping. What it does not prove: compositor timing or train-ready admission.",
        23,
        bold=True,
    )
    draw_text(
        draw,
        (40, 1010),
        f"source: Twitch {VIDEO_ID} · reviewed with AI assistance · timing offset still unmeasured · fair-use research exhibit",
        18,
        color=INK_MUTED,
    )
    return save_png(canvas, "fig_wild_decoder_layout")


def render_state_pair(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    source_path: Path,
    state: str,
    frame_index: int,
    time_s: float,
    sample_rect: list[float],
) -> None:
    x0, y0, x1, y1 = box
    rounded_card(draw, box, fill="#f6f6f3")
    state_fill = GREEN_PALE if state == "pressed" else GRAY_PALE
    state_color = GREEN if state == "pressed" else INK_MUTED
    badge(draw, (x0 + 18, y0 + 14), state, fill=state_fill, color=state_color, size=17)
    draw_text(draw, (x1 - 18, y0 + 23), f"frame {frame_index} · PTS {time_s:.3f}s", 16,
              color=INK_MUTED, anchor="ra")

    source = Image.open(source_path).convert("RGB")
    gameplay = resize_to(source.crop(GAMEPLAY_XYXY), 700, 393)
    canvas.paste(gameplay, (x0 + 18, y0 + 58))

    panel = draw_sample_on_panel(source.crop(PANEL_XYXY), sample_rect)
    panel = resize_to(panel, 294, 121)
    px, py = x1 - 18 - 294, y1 - 18 - 121
    canvas.paste(panel, (px, py))
    draw.rectangle((px, py, px + 294, py + 121), outline=state_color, width=4)


def render_states(case: Path, layout: dict[str, Any], cells: dict[str, Any]) -> Path:
    canvas = Image.new("RGB", (1800, 1685), SURFACE)
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (38, 24), "Pixels become seven binary action labels", 48, bold=True)
    draw_text(
        draw,
        (40, 84),
        "Three of the seven human-approved mappings are shown. Each pair contrasts an isolated released state with a pressed state and visible game response.",
        22,
        color=INK_MUTED,
    )

    layout_by_id = {row["cell_id"]: row for row in layout["cells"]}
    evidence_by_id = {row["cell_id"]: row for row in cells["cells"]}
    selected = (
        ("a", "LEFT", "A", BLUE),
        ("space", "JUMP", "SPACE", "#c1497c"),
        ("u", "DASH", "U", GREEN),
    )
    row_tops = (154, 632, 1110)
    for (cell_id, action, physical, color), top in zip(selected, row_tops):
        evidence = evidence_by_id[cell_id]
        layout_cell = layout_by_id[cell_id]
        draw.rounded_rectangle((38, top, 184, top + 62), radius=18, fill=color)
        draw_text(draw, (111, top + 31), action, 23, color="white", bold=True, anchor="mm")
        draw_text(draw, (40, top + 78), f"physical key: {physical}", 17, color=INK_MUTED)
        desc_lines = wrapped_lines(draw, evidence["observation"], 145, 16)
        draw.multiline_text((40, top + 108), "\n".join(desc_lines), fill=INK, font=font(16), spacing=7)

        for column, state in enumerate(("released", "pressed")):
            row = evidence[state]
            render_state_pair(
                canvas,
                draw,
                box=(205 + column * 790, top, 955 + column * 790, top + 460),
                source_path=case / "review_packet" / row["path"],
                state=state,
                frame_index=int(row["frame_index"]),
                time_s=float(row["time_s"]),
                sample_rect=layout_cell["sample_rect"],
            )

    draw_text(
        draw,
        (40, 1615),
        "Pairs are distributed semantic evidence, not adjacent before/after frames. The reviewed mapping is source-bound; timing-offset calibration is a separate gate.",
        18,
        color=INK_MUTED,
    )
    return save_png(canvas, "fig_wild_decoder_states")


def render_sequence_card(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    frame_path: Path,
    frame_index: int,
    pts_s: float,
    relative_ms: int,
    physical_cells: list[str],
    label: str,
    u_rect: list[float],
) -> None:
    x0, y0, x1, y1 = box
    pressed = "u" in physical_cells
    border = GREEN if pressed else GRID
    rounded_card(draw, box, fill="#f7f7f4", outline=border, width=3 if pressed else 2)
    draw_text(draw, (x0 + 18, y0 + 17), label, 21, bold=True)
    rel = f"{relative_ms:+d} ms" if relative_ms else "0 ms"
    draw_text(draw, (x1 - 18, y0 + 24), rel, 18, color=GREEN if pressed else INK_MUTED,
              bold=pressed, anchor="ra")

    source = Image.open(frame_path).convert("RGB")
    gameplay = resize_to(source.crop(GAMEPLAY_XYXY), 522, 293)
    canvas.paste(gameplay, (x0 + 18, y0 + 58))
    panel = draw_sample_on_panel(source.crop(PANEL_XYXY), u_rect,
                                 color=GREEN if pressed else INK_MUTED)
    panel = resize_to(panel, 244, 100)
    px, py = x1 - 18 - 244, y1 - 18 - 100
    canvas.paste(panel, (px, py))
    draw.rectangle((px, py, px + 244, py + 100), outline=border, width=3)
    decoded = "+".join(physical_cells) if physical_cells else "released"
    draw_text(draw, (x0 + 18, y1 - 76), f"decoded: {decoded}", 17,
              color=GREEN if pressed else INK_MUTED, bold=pressed)
    draw_text(draw, (x0 + 18, y1 - 46), f"source frame {frame_index} · PTS {pts_s:.3f}s", 15,
              color=INK_MUTED)


def render_sequence(case: Path, layout: dict[str, Any]) -> Path:
    report = json.loads((case / "exact_packet" / "exact_pts_evidence.json").read_text())
    by_index = {int(row["frame_index"]): row for row in report["frames"]}
    indices = (783117, 783119, 783121, 783123, 783126, 783127)
    labels = (
        "released",
        "U first fills",
        "U held",
        "U held",
        "blue dash burst",
        "U released; motion continues",
    )
    onset_s = float(by_index[783119]["persisted_pts_s"])
    u_rect = next(row["sample_rect"] for row in layout["cells"] if row["cell_id"] == "u")

    canvas = Image.new("RGB", (1800, 1230), SURFACE)
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (38, 24), "The visual sequence behind U → dash", 48, bold=True)
    draw_text(
        draw,
        (40, 84),
        "Exact source timestamps make the semantic review auditable: the key fills, the blue dash effect appears, and motion persists after release.",
        22,
        color=INK_MUTED,
    )

    card_w, card_h = 560, 480
    xs = (38, 620, 1202)
    ys = (146, 648)
    for i, (index, label) in enumerate(zip(indices, labels)):
        row = by_index[index]
        pts_s = float(row["persisted_pts_s"])
        render_sequence_card(
            canvas,
            draw,
            box=(xs[i % 3], ys[i // 3], xs[i % 3] + card_w, ys[i // 3] + card_h),
            frame_path=case / "exact_packet" / "evidence" / f"frame_{index:06d}.png",
            frame_index=index,
            pts_s=pts_s,
            relative_ms=round((pts_s - onset_s) * 1000),
            physical_cells=list(row.get("physical_cells", [])),
            label=label,
            u_rect=u_rect,
        )

    draw_text(
        draw,
        (40, 1160),
        "This sequence supports the semantic mapping. It is intentionally not presented as compositor-offset calibration or admitted supervision.",
        18,
        color=INK_MUTED,
    )
    return save_png(canvas, "fig_wild_decoder_sequence")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help="Private directory containing v498642684/review_packet and exact_packet",
    )
    args = parser.parse_args()
    case, layout, cells = load_packet(args.evidence_root)
    outputs = (
        render_layout(case, layout),
        render_states(case, layout, cells),
        render_sequence(case, layout),
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
