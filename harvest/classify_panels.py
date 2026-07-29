"""Classify scanned panel crops: which videos carry a CelesteTAS action HUD?

The corpus scan finds every static-and-changing panel in a frame, but a
speedrun frame holds several — a run timer, a LiveSplit column, chat, and
sometimes the input display we actually want. Telling them apart is a reading
task, and the panels are text, so OCR answers it deterministically and for
free rather than paying a VLM per video.

The three panel families are separable by vocabulary:

  * INPUT HUD  — Celeste action names: jump, dash, grab, demo, pause, talk,
    and the crouch/climb variants, plus direction glyphs.
  * SPLITS     — Celeste chapter names: prologue, city, site, resort, ridge,
    temple, reflection, summit, farewell, epilogue, core.
  * TIMER      — digits and colons and little else.

A video is accepted only when some panel shows at least MIN_ACTION_HITS
distinct action words, which rejects a splits panel that happens to contain
one matching token. Everything is recorded — the OCR text included — so a
disputed call can be re-read without re-fetching anything.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract

ACTION_WORDS = {
    "jump", "dash", "grab", "demo", "pause", "talk", "climb", "crouch",
    "left", "right", "up", "down", "quick", "restart", "confirm",
}
SPLIT_WORDS = {
    "prologue", "city", "site", "resort", "ridge", "temple", "reflection",
    "summit", "farewell", "epilogue", "core", "chapter", "total", "segment",
    "best", "comparing", "possible",
}
MIN_ACTION_HITS = 2
TIME_RE = re.compile(r"\d+[:.]\d+")


def ocr_text(image_path: Path) -> str:
    image = cv2.imread(str(image_path))
    if image is None:
        return ""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if max(gray.shape) < 400:                      # OCR likes bigger glyphs
        scale = 400.0 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    # Translucent HUDs come in both polarities; read both and keep the richer.
    best = ""
    for candidate in (gray, cv2.bitwise_not(gray)):
        _, binary = cv2.threshold(candidate, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            text = pytesseract.image_to_string(binary, config="--psm 6")
        except Exception:                          # noqa: BLE001
            text = ""
        if len(text.strip()) > len(best.strip()):
            best = text
    return best


def classify_text(text: str) -> dict:
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    actions = sorted(tokens & ACTION_WORDS)
    splits = sorted(tokens & SPLIT_WORDS)
    times = len(TIME_RE.findall(text))
    if len(actions) >= MIN_ACTION_HITS and len(actions) > len(splits):
        kind = "input_hud"
    elif splits:
        kind = "splits"
    elif times:
        kind = "timer"
    else:
        kind = "unknown"
    return {"kind": kind, "actions": actions, "splits": splits,
            "n_times": times}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--crops-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.scan.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("crops")]
    if args.limit:
        rows = rows[: args.limit]

    done: set[str] = set()
    if args.out.exists():
        done = {json.loads(l)["video_id"]
                for l in args.out.read_text().splitlines() if l.strip()}
    rows = [r for r in rows if r["video_id"] not in done]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    hits = 0
    with args.out.open("a") as fh:
        for n, row in enumerate(rows, 1):
            panels = []
            for name in row["crops"]:
                text = ocr_text(args.crops_dir / name)
                verdict = classify_text(text)
                verdict["crop"] = name
                verdict["text"] = text.strip()[:300]
                panels.append(verdict)
            hud = [p for p in panels if p["kind"] == "input_hud"]
            record = {
                "video_id": row["video_id"], "url": row.get("url"),
                "category": row.get("category"), "place": row.get("place"),
                "duration_s": row.get("duration_s"),
                "has_input_hud": bool(hud),
                "hud_actions": sorted({a for p in hud for a in p["actions"]}),
                "hud_panel_rect": next(
                    (pr["panel_rect"] for pr, p in zip(row.get("panels", []), panels)
                     if p["kind"] == "input_hud"), None),
                "panels": panels,
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            hits += bool(hud)
            if n % 50 == 0 or n == len(rows):
                print(f"[{n}/{len(rows)}] input-HUD videos: {hits}", flush=True)
    print(f"done: {hits} videos with a CelesteTAS-style input HUD")


if __name__ == "__main__":
    main()
