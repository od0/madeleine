"""G1 capture glue: mss region grab at 60Hz piped to ffmpeg x264, CFR by construction.

Orchestrator-owned day-0 tool. theo/capture.py (packet A3) supersedes this for
real sessions; this exists so the G1 gate doesn't wait on it. The strip rect it
reports is measured from the probe frame, not assumed.

Usage:
  uv run python -m theo.g1_capture --probe            # save one PNG of the game window
  uv run python -m theo.g1_capture --record SECONDS   # capture video + capture_meta.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mss
import numpy as np
import Quartz

FPS = 60


def find_celeste_window() -> dict[str, int]:
    """Global logical desktop coords of the Celeste window, title bar excluded."""
    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
    )
    display = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    for w in windows:
        owner = str(w.get("kCGWindowOwnerName", ""))
        if "celeste" not in owner.lower():
            continue
        # Fullscreen mode also lists a phantom alpha-0 title-bar window.
        if float(w.get("kCGWindowAlpha", 1.0)) == 0.0:
            continue
        b = w["kCGWindowBounds"]
        rect = {
            "left": int(b["X"]),
            "top": int(b["Y"]),
            "width": int(b["Width"]),
            "height": int(b["Height"]),
        }
        if rect["width"] < 320 or rect["height"] < 180:
            continue
        fullscreen = (
            rect["left"] == 0
            and rect["top"] == 0
            and rect["width"] == int(display.size.width)
            and rect["height"] == int(display.size.height)
        )
        if not fullscreen:
            # Windowed bounds include the title bar; the content area is what
            # we capture. 28 logical px is the standard macOS title bar.
            rect["top"] += 28
            rect["height"] -= 28
        return rect
    raise SystemExit("Celeste window not found (is the game running, not minimized?)")


GAME_ASPECT = 1920 / 1080  # Celeste renders a fixed 16:9 canvas


def display_rect() -> dict[str, int]:
    """Full main-display region, for fullscreen-game capture."""
    with mss.mss() as sct:
        m = sct.monitors[1]
    return {"left": m["left"], "top": m["top"],
            "width": m["width"], "height": m["height"]}


def game_canvas_rect(display: dict[str, int]) -> dict[str, int]:
    """The letterboxed game canvas inside a fullscreen display.

    Celeste renders 16:9 and pads to fit the screen, so a display with a
    different aspect (the 1710x1112 built-in screen: 1.54 vs 1.78) records
    black bands — ~13% of every frame, encoded for nothing. Capturing the
    canvas alone keeps every pixel useful, cuts encode bandwidth (helping
    hold 60fps), and makes the strip geometry deterministic again: the strip
    sits at the canvas origin, so rect = (0, 0, 512*s, 48*s) with s = cw/1920.
    """
    w, h = display["width"], display["height"]
    if w / h > GAME_ASPECT:      # display wider than 16:9 -> bands left/right
        cw, ch = round(h * GAME_ASPECT), h
    else:                        # display taller -> bands top/bottom
        cw, ch = w, round(w / GAME_ASPECT)
    return {"left": display["left"] + (w - cw) // 2,
            "top": display["top"] + (h - ch) // 2,
            "width": cw, "height": ch}


def capture_rect(use_display: bool, crop_letterbox: bool = True) -> dict[str, int]:
    """Ask the OS where the game is; only model geometry as a fallback.

    Quartz reports the exact window bounds whether Celeste is windowed or
    fullscreen, so prefer it — guessing a centred letterbox is wrong for a
    window positioned off-centre (measured: a 1680x945 canvas at (0,115) on a
    1710x1112 screen). The 16:9 crop then only trims the game's own padding
    when the window aspect differs, and is a no-op on a 16:9 canvas.
    """
    rect = None
    try:
        rect = find_celeste_window()
    except SystemExit:
        if not use_display:
            raise
    if rect is None:
        rect = display_rect()
    return game_canvas_rect(rect) if crop_letterbox else rect


def probe(out_dir: Path, use_display: bool = False,
          crop_letterbox: bool = True) -> None:
    rect = capture_rect(use_display, crop_letterbox)
    with mss.mss() as sct:
        img = np.asarray(sct.grab(rect))
    import cv2

    out = out_dir / "g1_probe.png"
    cv2.imwrite(str(out), img[:, :, :3])
    print(json.dumps({"window_rect": rect, "probe_png": str(out),
                      "captured_px": [img.shape[1], img.shape[0]]}))


def record(seconds: float, out_dir: Path, use_display: bool = False,
           encoder: str = "x264", crop_letterbox: bool = True) -> None:
    disp = display_rect() if use_display else None
    rect = capture_rect(use_display, crop_letterbox)
    with mss.mss() as sct:
        first = np.asarray(sct.grab(rect))
    h, w = first.shape[:2]

    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "video.mkv"
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{w}x{h}",
            "-r", str(FPS), "-i", "-",
            *(["-c:v", "h264_videotoolbox", "-b:v", "20M"]
              if encoder == "vt"
              else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16"]),
            "-pix_fmt", "yuv420p", str(video_path),
        ],
        stdin=subprocess.PIPE,
    )
    assert ffmpeg.stdin is not None

    n_frames = int(round(seconds * FPS))
    started_at = datetime.now(timezone.utc).isoformat()
    grab_times: list[float] = []
    period = 1.0 / FPS
    with mss.mss() as sct:
        t_next = time.perf_counter()
        for _ in range(n_frames):
            now = time.perf_counter()
            if now < t_next:
                time.sleep(t_next - now)
            grab_times.append(time.perf_counter())
            frame = sct.grab(rect)
            ffmpeg.stdin.write(frame.raw)
            t_next += period
    ffmpeg.stdin.close()
    ffmpeg.wait()

    deltas = np.diff(grab_times)
    meta = {
        "tool": "g1_capture(mss+ffmpeg)",
        "window_rect": rect,
        "display_rect": disp,
        "letterbox_cropped": bool(use_display and crop_letterbox),
        "canvas_scale": round(w / 1920.0, 6),
        "captured_px": [w, h],
        "requested_fps": FPS,
        "achieved_fps": round(1.0 / float(np.mean(deltas)), 3),
        "tick_jitter_ms_p99": round(float(np.percentile(deltas, 99)) * 1e3, 2),
        "frames_written": n_frames,
        "encode": ("h264_videotoolbox b20M yuv420p cfr60" if encoder == "vt"
                   else "libx264 crf16 veryfast yuv420p cfr60"),
        "started_at": started_at,
    }
    (out_dir / "capture_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--probe", action="store_true")
    group.add_argument("--record", type=float, metavar="SECONDS")
    parser.add_argument("--out", type=Path, default=Path("sessions/_g1_capture"))
    # Legacy capture only composites regular desktop Spaces. The user talks to
    # the orchestrator from a fullscreen app, so grabs must fire after they've
    # switched to the game's Space.
    parser.add_argument("--delay", type=float, default=0.0, metavar="SECONDS")
    parser.add_argument("--display", action="store_true",
                        help="capture the full main display (fullscreen game)")
    parser.add_argument("--encoder", choices=["x264", "vt"], default="x264")
    parser.add_argument("--no-crop-letterbox", action="store_true",
                        help="record the whole display incl. the game's own "
                             "letterbox bands (default: crop to game canvas)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.delay > 0:
        time.sleep(args.delay)
    if args.probe:
        probe(args.out, use_display=args.display,
              crop_letterbox=not args.no_crop_letterbox)
    else:
        record(args.record, args.out, use_display=args.display,
               encoder=args.encoder,
               crop_letterbox=not args.no_crop_letterbox)


if __name__ == "__main__":
    main()
