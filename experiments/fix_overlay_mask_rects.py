"""Apply measured overlay mask geometry to session manifests.

The 2026-07-26 masking audit found that `input_overlay` rects derived from
the mod's logical constants undershoot the rendered overlay on the 1710-px
built-in-display rigs (the render pass's vertical transform is not the
uniform canvas scale the assembler assumed; horizontal is exact). This
script rewrites the affected rects with geometry measured from pixels by
experiments/measure_overlay_geometry.py.

The corrected rects are an explicit per-family table below, not an
automatic union of measurements: short fixture sessions contaminate
per-session estimates with gameplay correlation, so each family rect was
chosen from the sessions with reliable estimates and then verified on every
member with data.mask_coverage (band statistic outside the rect ~ zero).
The table records what each rect covers and which measurements support it.

For the calibration session's `wild_overlay`, the previous rect is retained
in a new `panel_rect_px` field: the translucent decoder was validated
(macro-F1 0.9977, results/s3_calibration.json) with cell geometry scaled
from that rect, so it must keep anchoring the decoder even though the mask
rect grows. Masking and instrument geometry are separate duties.

Dry-run by default; `--apply` rewrites manifest.json in place, preserving
the superseded rect in `supersedes` for the audit trail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MEASURED_AT = "2026-07-26"
METHOD = (
    "key-state-correlated pixel measurement vs engine truth "
    "(experiments/measure_overlay_geometry.py); see report/findings_log.md"
)

# rect_px is [x, y, w, h] in capture pixels, keyed by capture resolution.
# Sources: measured cell extents and static backing-bar extents, plus a
# safety margin; every applied rect is re-verified by data.mask_coverage.
INPUT_OVERLAY_RECTS: dict[tuple[int, int], list[int]] = {
    # Built-in display, letterbox-cropped. Cells measured at
    # [14, 896, 349, 924], backing bar [0, 890, 370, 962] on all six
    # sessions (identical to the pixel). Rect = bar + 6 px margin
    # up/right, anchored to the frame's left/bottom edges.
    (1710, 962): [0, 884, 376, 78],
    # Built-in display, pre-crop (letterboxed window; canvas top at y=121).
    # Cells measured at [14, 1006, 349, 1034] — the declared rect started at
    # y=1034, missing the ENTIRE cell row on this family. Static bar from
    # y=1000 merging into the bottom letterbox. Rect = bar + 6 px margin
    # up/right, to the frame bottom.
    (1710, 1112): [0, 994, 420, 118],
    # External display: the render matches the logical constants and the
    # old rect already covered the measured cells [21, 1387, 523, 1429];
    # widened by 8 px top/right for margin and family consistency.
    (2560, 1440): [0, 1368, 563, 72],
}

# wild_overlay (calibration sessions only): measured state-correlated
# pixels [1322, 790, 1685, 907]; the translucent panel's background starts
# ~y 780. Rect covers panel + margin. panel_rect_px keeps the decoder's
# validated anchor.
WILD_OVERLAY_RECT: list[int] = [1305, 774, 397, 179]


def _rect_norm(rect_px: list[int], resolution: list[int]) -> list[float]:
    x, y, w, h = rect_px
    cw, ch = resolution
    return [x / cw, y / ch, (x + w) / cw, (y + h) / ch]


def fix_session(session_dir: Path, apply: bool) -> list[str]:
    manifest_path = session_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolution = tuple(manifest["capture"]["resolution"])
    changes: list[str] = []
    for region in manifest["masked_regions"]:
        if region["name"] == "input_overlay":
            new_rect = INPUT_OVERLAY_RECTS.get(resolution)
            if new_rect is None:
                raise SystemExit(
                    f"{session_dir.name}: no measured input_overlay rect for "
                    f"capture resolution {resolution}; measure before fixing"
                )
        elif region["name"] == "wild_overlay":
            new_rect = WILD_OVERLAY_RECT
        else:
            continue
        if region["rect_px"] == new_rect:
            continue
        old_rect = region["rect_px"]
        if region["name"] == "wild_overlay" and "panel_rect_px" not in region:
            region["panel_rect_px"] = list(old_rect)
        region["supersedes"] = {
            "rect_px": old_rect,
            "reason": "undershot the rendered overlay (2026-07-26 audit)",
        }
        region["rect_px"] = list(new_rect)
        region["rect_norm"] = _rect_norm(new_rect, list(resolution))
        region["geometry_provenance"] = {
            "method": METHOD,
            "measured_at": MEASURED_AT,
        }
        changes.append(
            f"{session_dir.name}: {region['name']} {old_rect} -> {new_rect}"
        )
    if changes and apply:
        manifest_path.write_text(
            json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
        )
    return changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", nargs="+", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    total = 0
    for session_dir in args.sessions:
        for change in fix_session(session_dir, args.apply):
            print(("APPLIED " if args.apply else "DRY-RUN ") + change)
            total += 1
    if not total:
        print("no changes needed")


if __name__ == "__main__":
    main()
