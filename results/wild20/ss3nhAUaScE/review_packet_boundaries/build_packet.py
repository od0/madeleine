"""Build the ss3nhAUaScE human boundary-review packet.

Prerequisite: run ``regenerate_timer_trace.py`` first; it writes and verifies
``inputs/timer_trace.npz`` against the recorded proposal (512-point diagnostic
agreement, exact 106-range reproduction, and every recorded population count).
This script refuses to run unless that verification exists and passed.

The packet asks one question: are the 106 AI-proposed gameplay ranges in
``../boundaries.v3-ai.json`` believable enough to adopt as human-reviewed
boundaries?  It creates no acceptance, no boundaries file, and no admission
state.

Outputs (all under this packet directory, all packet-relative in the
manifest):

- ``boundary_review_full.png``  — annotated full-video trace with all 106
  ranges drawn; the range count is asserted against the authoritative list at
  render time and stated in the caption.
- ``spot_checks/range-***.jpg`` — annotated exact-frame evidence for a
  deterministic sample of range starts/ends (earliest, latest, longest,
  shortest, second-longest, and 8 seeded-random ranges).
- ``spot_checks/gap-***.jpg``   — the two largest excluded gaps, which must
  show a frozen or absent timer.
- ``spot_checks/bridge-***.jpg``— the highest-risk bridged gaps inside
  proposed ranges (these must be brief gameplay freezes, not loads or menus).
- ``spot_check.json``           — the machine-readable sample table with
  YouTube timestamps for online verification.
- ``source_artifacts/``         — hash-bound copies of the reviewed inputs.
- ``REVIEW.md`` and ``review_manifest.json``.

Every displayed edge frame's timer-ROI bright/dark mask means are recomputed
from the extracted frame and must equal the verified full-resolution trace at
that exact frame index, so a wrong seek cannot silently show the wrong frame.

Usage (from the repository root):

    uv run python results/wild20/ss3nhAUaScE/review_packet_boundaries/\
build_packet.py --video-dir <dir containing ss3nhAUaScE.mp4>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

PACKET = Path(__file__).resolve().parent
REPO = PACKET.parents[3]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

VIDEO_ID = "ss3nhAUaScE"
SOURCE_DIR = PACKET.parent
BOUNDARIES_PATH = SOURCE_DIR / "boundaries.v3-ai.json"
PROPOSAL_PATH = SOURCE_DIR / "timer-official-v3-ai" / "timer_activity_proposal.json"
TRACE_MANIFEST_PATH = SOURCE_DIR / "timer-official-v3-ai" / "timer_trace_manifest.json"
FETCH_PATH = SOURCE_DIR / "fetch.json"
EVIDENCE_MANIFEST_PATH = SOURCE_DIR / "evidence_manifest.json"

SOURCE_SHA256 = "b6b5302c5d79eafb8d96f289bdfd175a5b80c438e4d12f5725cd51823b715544"
PTS_SHA256 = "28b31524e24bac3484703ffc569306656ec232abfa36d46fca250359a8006ec1"
EXPECTED_RANGE_COUNT = 106
WALL_BOUNDS_S = (12.233333, 6867.083333)
ROI_XYXY = (18, 30, 244, 64)  # exact recorded pixel ROI of the official timer
FIRST_SELECTED_FRAME = 734
VIDEO_URL = "https://youtu.be/ss3nhAUaScE"
RANDOM_SEED = 20260728
RANDOM_SAMPLE_SIZE = 8
WALL_EDGE_EVIDENCE = (
    "t000012233_start_zero.jpg",
    "t000012250_start_advancing.jpg",
    "t006867067_last_advancing.jpg",
    "t006867083_first_final.jpg",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hms(seconds: float) -> str:
    total = float(seconds)
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    return f"{hours}:{minutes:02d}:{total % 60:06.3f}"


def yt_url(seconds: float) -> str:
    return f"{VIDEO_URL}?t={int(seconds)}"


def font(size: int) -> ImageFont.ImageFont:
    for name in ("Menlo.ttc", "Monaco.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class FrameGrabber:
    """Index-exact frame access through the trace's own OpenCV pipeline.

    Using the identical decode and BGR2GRAY conversion as
    ``harvest.extract_timer_trace._stream_scalar_trace`` makes the displayed
    frame's timer-ROI scalars exactly comparable to the regenerated trace, so
    a mis-seek cannot silently show the wrong frame.
    """

    def __init__(self, video: Path) -> None:
        self.capture = cv2.VideoCapture(str(video))
        if not self.capture.isOpened():
            raise ValueError(f"cannot open source video: {video}")

    def grab(self, frame_index: int) -> tuple[Image.Image, np.ndarray]:
        if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index)):
            raise ValueError(f"OpenCV could not seek to source frame {frame_index}")
        ok, frame = self.capture.read()
        if not ok or frame.ndim != 3 or frame.dtype != np.uint8:
            raise ValueError(f"decode failed at source frame {frame_index}")
        position = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        if np.isfinite(position) and abs(position - (frame_index + 1)) > 0.5:
            raise ValueError(
                f"OpenCV seek for frame {frame_index} landed at {position}"
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x0, y0, x1, y1 = ROI_XYXY
        roi = gray[y0:y1, x0:x1]
        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return rgb, roi

    def __enter__(self) -> "FrameGrabber":
        return self

    def __exit__(self, *_exc) -> None:
        self.capture.release()


def mask_means(roi: np.ndarray) -> tuple[float, float]:
    bright = float(np.mean((roi >= 200.0).astype(np.float32) * 255.0, dtype=np.float64))
    dark = float(np.mean((roi <= 32.0).astype(np.float32) * 255.0, dtype=np.float64))
    return bright, dark


class Trace:
    def __init__(self, npz_path: Path, policy: dict) -> None:
        data = np.load(npz_path)
        self.source_frame_idx = data["source_frame_idx"]
        self.pts_s = data["pts_s"]
        self.change = data["change_score"]
        self.bright = data["bright_mask_mean"]
        self.dark = data["dark_mask_mean"]
        self.present = (self.bright >= policy["min_bright_mask_mean"]) & (
            self.dark >= policy["min_dark_mask_mean"]
        )

    def position(self, source_frame: int) -> int | None:
        offset = source_frame - int(self.source_frame_idx[0])
        if 0 <= offset < self.source_frame_idx.size:
            return offset
        return None


def frame_index_for_pts(all_pts: np.ndarray, pts: float, label: str) -> int:
    index = int(np.searchsorted(all_pts, pts, side="left"))
    if index >= all_pts.size or all_pts[index] != pts:
        raise ValueError(f"{label}: {pts!r} does not equal a persisted frame PTS")
    return index


def edge_verify_text(
    trace: Trace, source_frame: int, measured: tuple[float, float]
) -> str:
    position = trace.position(source_frame)
    if position is None:
        return "outside the reviewed trace interval (no recorded scalars)"
    recorded = (float(trace.bright[position]), float(trace.dark[position]))
    if measured != recorded:
        raise ValueError(
            f"frame f{source_frame}: extracted ROI mask means {measured} do not "
            f"equal the verified trace {recorded}; the seek landed on the wrong "
            "frame"
        )
    state = "present" if trace.present[position] else "ABSENT/occluded"
    return (
        f"ROI bright/dark {measured[0]:.1f}/{measured[1]:.1f} == verified trace; "
        f"timer {state}"
    )


def tile(
    frame_image: Image.Image,
    heading: str,
    subtext: list[str],
    accent: str,
) -> Image.Image:
    thumb_w, thumb_h = 620, 349
    zoom_scale = 3
    x0, y0, x1, y1 = ROI_XYXY
    zoom_w, zoom_h = (x1 - x0) * zoom_scale, (y1 - y0) * zoom_scale
    canvas = Image.new("RGB", (648, 620), "#11151a")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), heading, fill=accent, font=font(17))
    source = frame_image.convert("RGB")
    thumb = source.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    tdraw = ImageDraw.Draw(thumb)
    scale_x, scale_y = thumb_w / source.width, thumb_h / source.height
    tdraw.rectangle(
        (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y),
        outline=accent,
        width=2,
    )
    canvas.paste(thumb, (14, 34))
    zoom = source.crop(ROI_XYXY).resize((zoom_w, zoom_h), Image.Resampling.NEAREST)
    canvas.paste(zoom, (14, 34 + thumb_h + 8))
    draw.rectangle(
        (14, 34 + thumb_h + 8, 14 + zoom_w, 34 + thumb_h + 8 + zoom_h),
        outline=accent,
        width=1,
    )
    y = 34 + thumb_h + 8 + zoom_h + 6
    for line in subtext:
        draw.text((10, y), line, fill="#c5d0dc", font=font(12))
        y += 17
    return canvas


def compose_page(
    title_lines: list[str], tiles: list[Image.Image], destination: Path
) -> None:
    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    header_h = 30 + 22 * len(title_lines)
    page = Image.new(
        "RGB", (14 + 662 * columns, header_h + 634 * rows), "#0b0e12"
    )
    draw = ImageDraw.Draw(page)
    y = 10
    for index, line in enumerate(title_lines):
        draw.text(
            (14, y),
            line,
            fill="white" if index == 0 else "#9fb0c0",
            font=font(19 if index == 0 else 14),
        )
        y += 26 if index == 0 else 20
    for index, image in enumerate(tiles):
        x = 14 + (index % columns) * 662
        row_y = header_h + (index // columns) * 634
        page.paste(image, (x, row_y))
    page.save(destination, quality=88, optimize=True)


def render_overview(
    ranges: list[list[float]],
    trace: Trace,
    proposal: dict,
    sampled: dict[int, str],
    destination: Path,
) -> dict:
    # Render-time completeness assertion against the authoritative artifact.
    authoritative = json.loads(BOUNDARIES_PATH.read_text())["allowed_ranges_s"]
    assert len(ranges) == EXPECTED_RANGE_COUNT, (
        f"render aborted: {len(ranges)} ranges != authoritative "
        f"{EXPECTED_RANGE_COUNT}"
    )
    assert ranges == [[float(a), float(b)] for a, b in authoritative], (
        "render aborted: range list does not equal boundaries.v3-ai.json"
    )
    recorded_total = int(
        proposal["activity"]["candidate_ranges_before_gates"]["total"]
    )
    assert recorded_total == EXPECTED_RANGE_COUNT and (
        proposal["activity"]["candidate_ranges_before_gates"]["truncated"] is False
    ), "render aborted: proposal candidate-range accounting mismatch"

    durations = [end - start for start, end in ranges]
    total_s = sum(durations)
    envelope_s = WALL_BOUNDS_S[1] - WALL_BOUNDS_S[0]
    caption = (
        f"{VIDEO_ID} — boundary review: all {len(ranges)} of "
        f"{EXPECTED_RANGE_COUNT} proposed ranges drawn "
        "(render-time assertion against untruncated boundaries.v3-ai.json); "
        f"{total_s:,.1f} s = {total_s / 3600:.4f} h gameplay of a "
        f"{envelope_s:,.1f} s reviewed envelope "
        f"({100 * total_s / envelope_s:.1f}% coverage); unshaded = excluded"
    )

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(19, 10),
        dpi=100,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.32},
    )
    for index, (start, end) in enumerate(ranges):
        top.axvspan(start, end, color="#f5b895", alpha=0.65, linewidth=0)
        if index in sampled:
            top.axvspan(
                start, end, facecolor="none", edgecolor="#c02020", linewidth=1.2
            )
    top.plot(
        trace.pts_s,
        np.where(trace.present, 0.86, 0.0),
        color="#6b5b4f",
        linewidth=0.5,
        label="timer present (full 411,291-frame trace)",
    )
    active = np.zeros(trace.pts_s.size)
    for start, end in ranges:
        active[(trace.pts_s >= start) & (trace.pts_s < end)] = 0.62
    top.plot(
        trace.pts_s,
        active,
        color="#3465c0",
        linewidth=0.6,
        label="proposed gameplay (bridged timer activity)",
    )
    label_step = 5
    for index, (start, end) in enumerate(ranges):
        if index in sampled or index % label_step == 0:
            middle = (start + end) / 2
            special = index in sampled
            top.text(
                middle,
                1.08 if special else 1.02,
                str(index),
                ha="center",
                fontsize=7.5 if special else 5.5,
                color="#c02020" if special else "#8a7360",
                fontweight="bold" if special else "normal",
            )
    top.set_xlim(0, WALL_BOUNDS_S[1] + 40)
    top.set_ylim(0, 1.14)
    top.set_yticks([])
    top.set_xlabel("video time (s)  [equals YouTube ?t= seconds]")
    top.legend(loc="lower right", fontsize=9, framealpha=0.9)
    top.set_title(
        "red outline + red index = spot-checked in this packet; "
        f"small indices label every {label_step}th range",
        fontsize=9,
        color="#555555",
        pad=30,
    )

    positions = np.arange(len(ranges))
    colors = ["#c02020" if i in sampled else "#f0a06a" for i in positions]
    bottom.bar(positions, durations, width=0.85, color=colors, linewidth=0)
    bottom.set_yscale("log")
    bottom.set_xlim(-1, len(ranges))
    bottom.set_xticks(np.arange(0, len(ranges), 5))
    bottom.tick_params(axis="x", labelsize=7)
    bottom.set_xlabel(
        f"range index (0..{len(ranges) - 1}) — every one of the "
        f"{len(ranges)} ranges appears as one bar"
    )
    bottom.set_ylabel("duration (s, log)")
    bottom.axhline(2.0, color="#888888", linewidth=0.7, linestyle="--")
    bottom.annotate(
        "2 s policy minimum",
        xy=(len(ranges) * 0.35, 2.0),
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444444",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
    )
    fig.suptitle(caption, fontsize=13.5, x=0.01, ha="left")
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return {
        "caption": caption,
        "ranges_drawn": len(ranges),
        "authoritative_count": EXPECTED_RANGE_COUNT,
        "assertion": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    args = parser.parse_args()
    video_path = args.video_dir / f"{VIDEO_ID}.mp4"

    verification_path = PACKET / "inputs" / "trace_verification.json"
    verification = json.loads(verification_path.read_text())
    if not all(row["passed"] for row in verification["checks"]):
        raise SystemExit("inputs/trace_verification.json records a failed check")
    if verification["source_sha256"] != SOURCE_SHA256:
        raise SystemExit("trace verification is bound to a different source")

    video_sha = sha256_file(video_path)
    if video_sha != SOURCE_SHA256:
        raise SystemExit(f"source video hash mismatch: {video_sha}")

    boundaries = json.loads(BOUNDARIES_PATH.read_text())
    proposal = json.loads(PROPOSAL_PATH.read_text())
    ranges = [[float(a), float(b)] for a, b in boundaries["allowed_ranges_s"]]
    assert len(ranges) == EXPECTED_RANGE_COUNT, (
        f"{len(ranges)} ranges in boundaries.v3-ai.json; expected "
        f"{EXPECTED_RANGE_COUNT}"
    )
    trace = Trace(PACKET / "inputs" / "timer_trace.npz", proposal["policy"])
    all_pts = np.load(args.video_dir / "frame_pts.npy")

    durations = np.asarray([end - start for start, end in ranges])
    order = np.argsort(durations)
    shortest = int(order[0])
    longest = int(order[-1])
    second_longest = int(order[-2])
    special = {
        0: "earliest range",
        len(ranges) - 1: "latest range",
        longest: "longest range",
        shortest: "shortest range",
        second_longest: "second-longest range",
    }
    if longest == len(ranges) - 1:
        special[longest] = "latest range (also the longest)"
    rng = np.random.default_rng(RANDOM_SEED)
    remaining = [index for index in range(len(ranges)) if index not in special]
    random_sample = sorted(
        int(value)
        for value in rng.choice(remaining, size=RANDOM_SAMPLE_SIZE, replace=False)
    )
    sampled = dict(special)
    for index in random_sample:
        sampled[index] = "random sample"
    sampled = dict(sorted(sampled.items()))

    spot_dir = PACKET / "spot_checks"
    source_artifacts = PACKET / "source_artifacts"
    for directory in (spot_dir, source_artifacts):
        if directory.exists():
            raise SystemExit(f"refusing to overwrite {directory}")
        directory.mkdir(parents=True)

    overview_path = PACKET / "boundary_review_full.png"
    overview_result = render_overview(
        ranges, trace, proposal, sampled, overview_path
    )
    print(json.dumps(overview_result, indent=2))

    spot_rows = []
    with FrameGrabber(video_path) as grabber:
        for page_number, (index, why) in enumerate(sampled.items(), start=1):
            start, end = ranges[index]
            start_frame = frame_index_for_pts(all_pts, start, f"range {index} start")
            end_frame = frame_index_for_pts(all_pts, end, f"range {index} end")
            spec = [
                (
                    start_frame - 1,
                    "start-1 (EXCLUDED)",
                    "verify: timer absent or frozen here",
                    "#e0e050",
                ),
                (
                    start_frame,
                    "start (first included)",
                    "verify: official timer visible",
                    "#3cff73",
                ),
                (
                    start_frame + 1,
                    "start+1 (included)",
                    "verify: digits ADVANCING vs previous tile",
                    "#3cff73",
                ),
                (
                    end_frame - 2,
                    "end-2 (included)",
                    "verify: digits still advancing",
                    "#3cff73",
                ),
                (
                    end_frame - 1,
                    "end-1 (last included)",
                    "verify: last advancing-timer frame",
                    "#3cff73",
                ),
                (
                    end_frame,
                    "end (first EXCLUDED)",
                    "verify: timer frozen or absent from here",
                    "#e0e050",
                ),
            ]
            tiles = []
            frame_rows = []
            for frame_index, heading, instruction, accent in spec:
                pts = float(all_pts[frame_index])
                frame_image, roi = grabber.grab(frame_index)
                measured = mask_means(roi)
                verify = edge_verify_text(trace, frame_index, measured)
                tiles.append(
                    tile(
                        frame_image,
                        heading,
                        [
                            f"f{frame_index} @ {pts:.6f}s ({hms(pts)})",
                            instruction,
                            verify,
                        ],
                        accent,
                    )
                )
                frame_rows.append(
                    {
                        "frame_index": frame_index,
                        "time_s": pts,
                        "role": heading,
                        "roi_bright_mask_mean": measured[0],
                        "roi_dark_mask_mean": measured[1],
                        "trace_verified": "== verified trace" in verify
                        or "outside" in verify,
                    }
                )
            page_path = spot_dir / f"range-{index:03d}.jpg"
            compose_page(
                [
                    f"range {index} of 0..105 ({why}) — "
                    f"[{start:.6f}, {end:.6f}) s = {hms(start)} .. {hms(end)}, "
                    f"{end - start:.2f} s",
                    "top row: start edge — timer must BEGIN advancing. "
                    "bottom row: end edge — timer must freeze or disappear.",
                    f"online check: start {yt_url(start)}   end {yt_url(end)}",
                ],
                tiles,
                page_path,
            )
            spot_rows.append(
                {
                    "kind": "range",
                    "range_index": index,
                    "reason": why,
                    "range_s": [start, end],
                    "duration_s": end - start,
                    "start_hms": hms(start),
                    "end_hms": hms(end),
                    "start_url": yt_url(start),
                    "end_url": yt_url(end),
                    "page": page_path.relative_to(PACKET).as_posix(),
                    "frames": frame_rows,
                }
            )
            print(f"spot page {page_number}/{len(sampled)}: {page_path.name}")

        # The two largest excluded gaps: the timer must be frozen or absent.
        gaps = sorted(
            (
                (ranges[i + 1][0] - ranges[i][1], ranges[i][1], ranges[i + 1][0], i)
                for i in range(len(ranges) - 1)
            ),
            reverse=True,
        )[:2]
        for duration, gap_start, gap_end, after_index in gaps:
            middle = (gap_start + gap_end) / 2
            tiles = []
            frame_rows = []
            for pts, heading in (
                (gap_start + 0.5, "gap start +0.5 s (EXCLUDED)"),
                (middle, "gap middle (EXCLUDED)"),
                (gap_end - 0.5, "gap end -0.5 s (EXCLUDED)"),
            ):
                frame_index = int(np.searchsorted(all_pts, pts, side="right")) - 1
                exact_pts = float(all_pts[frame_index])
                frame_image, roi = grabber.grab(frame_index)
                measured = mask_means(roi)
                verify = edge_verify_text(trace, frame_index, measured)
                tiles.append(
                    tile(
                        frame_image,
                        heading,
                        [
                            f"f{frame_index} @ {exact_pts:.6f}s ({hms(exact_pts)})",
                            "verify: NO advancing official timer (menu/load/"
                            "cutscene/frozen)",
                            verify,
                        ],
                        "#e0e050",
                    )
                )
                frame_rows.append(
                    {
                        "frame_index": frame_index,
                        "time_s": exact_pts,
                        "role": heading,
                        "roi_bright_mask_mean": measured[0],
                        "roi_dark_mask_mean": measured[1],
                    }
                )
            page_path = spot_dir / f"gap-after-range-{after_index:03d}.jpg"
            compose_page(
                [
                    f"excluded gap between ranges {after_index} and "
                    f"{after_index + 1} — [{gap_start:.6f}, {gap_end:.6f}) s = "
                    f"{hms(gap_start)} .. {hms(gap_end)}, {duration:.2f} s",
                    "this whole interval is EXCLUDED from training; the "
                    "official timer must not advance anywhere inside it.",
                    f"online check: {yt_url(gap_start)} through {yt_url(gap_end)}",
                ],
                tiles,
                page_path,
            )
            spot_rows.append(
                {
                    "kind": "excluded_gap",
                    "after_range_index": after_index,
                    "range_s": [gap_start, gap_end],
                    "duration_s": duration,
                    "start_hms": hms(gap_start),
                    "end_hms": hms(gap_end),
                    "start_url": yt_url(gap_start),
                    "end_url": yt_url(gap_end),
                    "page": page_path.relative_to(PACKET).as_posix(),
                    "frames": frame_rows,
                }
            )
            print(f"gap page: {page_path.name}")

        # Highest-risk bridges inside proposed ranges (checklist item 3).
        risky = [
            json.loads(line)
            for line in (PACKET / "inputs" / "bridge_risk_population.jsonl")
            .read_text()
            .splitlines()
        ]
        risky.sort(
            key=lambda row: (row["absent_timer_frames"], row["duration_s"]),
            reverse=True,
        )
        for row in risky[:3]:
            bridge_start, bridge_end = row["range_s"]
            containing = next(
                (
                    i
                    for i, (a, b) in enumerate(ranges)
                    if a <= bridge_start and bridge_end <= b
                ),
                None,
            )
            middle = (bridge_start + bridge_end) / 2
            tiles = []
            frame_rows = []
            for pts, heading in (
                (bridge_start - 0.1, "before bridge (active)"),
                (middle, "inside bridge (INCLUDED, timer not advancing)"),
                (bridge_end + 0.1, "after bridge (active)"),
            ):
                frame_index = int(np.searchsorted(all_pts, pts, side="right")) - 1
                exact_pts = float(all_pts[frame_index])
                frame_image, roi = grabber.grab(frame_index)
                measured = mask_means(roi)
                verify = edge_verify_text(trace, frame_index, measured)
                tiles.append(
                    tile(
                        frame_image,
                        heading,
                        [
                            f"f{frame_index} @ {exact_pts:.6f}s ({hms(exact_pts)})",
                            "verify: brief gameplay freeze (hitstop/screen "
                            "shake), NOT a load or menu",
                            verify,
                        ],
                        "#7dd3fc",
                    )
                )
                frame_rows.append(
                    {
                        "frame_index": frame_index,
                        "time_s": exact_pts,
                        "role": heading,
                        "roi_bright_mask_mean": measured[0],
                        "roi_dark_mask_mean": measured[1],
                    }
                )
            page_path = spot_dir / (
                f"bridge-{int(bridge_start * 1000):09d}ms.jpg"
            )
            compose_page(
                [
                    f"bridged gap inside range {containing} — "
                    f"[{bridge_start:.6f}, {bridge_end:.6f}) s = "
                    f"{hms(bridge_start)}, {row['duration_s'] * 1000:.0f} ms, "
                    f"{row['absent_timer_frames']} absent-timer frames",
                    "these frames ARE included in the proposed range although "
                    "the timer was not advancing; they must be a brief "
                    "gameplay freeze, not a load or menu.",
                    f"online check: {yt_url(bridge_start)}",
                ],
                tiles,
                page_path,
            )
            spot_rows.append(
                {
                    "kind": "included_bridge",
                    "containing_range_index": containing,
                    "range_s": [bridge_start, bridge_end],
                    "duration_s": row["duration_s"],
                    "absent_timer_frames": row["absent_timer_frames"],
                    "start_hms": hms(bridge_start),
                    "start_url": yt_url(bridge_start),
                    "page": page_path.relative_to(PACKET).as_posix(),
                    "frames": frame_rows,
                }
            )
            print(f"bridge page: {page_path.name}")

    # --- source artifact copies ----------------------------------------------
    copied_sources = {}
    for source in (
        BOUNDARIES_PATH,
        PROPOSAL_PATH,
        TRACE_MANIFEST_PATH,
        FETCH_PATH,
        EVIDENCE_MANIFEST_PATH,
    ):
        target = source_artifacts / source.name
        shutil.copyfile(source, target)
        copied_sources[source.name] = {
            "packet_path": target.relative_to(PACKET).as_posix(),
            "sha256": sha256_file(target),
            "original_repo_path": source.relative_to(REPO).as_posix(),
        }
        if sha256_file(source) != copied_sources[source.name]["sha256"]:
            raise SystemExit(f"copy mismatch for {source}")
    evidence_manifest = json.loads(EVIDENCE_MANIFEST_PATH.read_text())
    recorded_frame_hashes = {
        Path(row["path"]).name: row["sha256"]
        for row in evidence_manifest["frames"]
    }
    wall_dir = source_artifacts / "wall_clock"
    wall_dir.mkdir()
    for name in WALL_EDGE_EVIDENCE:
        source = SOURCE_DIR / "evidence" / name
        target = wall_dir / name
        shutil.copyfile(source, target)
        digest = sha256_file(target)
        if recorded_frame_hashes[name] != digest:
            raise SystemExit(f"wall-clock evidence hash mismatch: {name}")
        copied_sources[f"wall_clock/{name}"] = {
            "packet_path": target.relative_to(PACKET).as_posix(),
            "sha256": digest,
            "recorded_in_evidence_manifest": True,
        }

    spot_check = {
        "format_version": "madeleine.wild20-boundary-spot-check.v1",
        "video_id": VIDEO_ID,
        "status": "awaiting_human_decision",
        "human_reviewed": False,
        "source_sha256": SOURCE_SHA256,
        "video_url": VIDEO_URL,
        "timeline_note": (
            "range seconds are relative to the first decoded video PTS, which "
            "is 0.0 for this source, so YouTube ?t= seconds equal range seconds"
        ),
        "sample_policy": {
            "special": {str(k): v for k, v in special.items()},
            "random_seed": RANDOM_SEED,
            "random_sample_size": RANDOM_SAMPLE_SIZE,
            "random_indices": random_sample,
            "excluded_gap_rule": "two largest excluded gaps",
            "bridge_rule": (
                "three riskiest bridges by (absent_timer_frames, duration) "
                "from the complete recomputed bridge population"
            ),
        },
        "edge_frame_binding": (
            "every displayed edge frame's timer-ROI bright/dark mask means "
            "were recomputed from the extracted frame and must equal the "
            "verified full-resolution trace at that frame index"
        ),
        "checks": spot_rows,
    }
    spot_check_path = PACKET / "spot_check.json"
    spot_check_path.write_text(json.dumps(spot_check, indent=2) + "\n")

    write_review_md(
        ranges, sampled, special, random_sample, spot_rows, verification, proposal
    )

    # --- manifest -------------------------------------------------------------
    bound = sorted(
        path
        for path in PACKET.rglob("*")
        if path.is_file()
        and path.name != "review_manifest.json"
        and "__pycache__" not in path.parts
        and not path.name.endswith(".pyc")
    )
    manifest = {
        "format_version": "madeleine.wild20-boundary-review-request.v1",
        "status": "awaiting_human_decision",
        "video_id": VIDEO_ID,
        "human_reviewed": False,
        "decision_requested": (
            "adopt the 106 proposed half-open gameplay ranges as "
            "human-reviewed WildBoundaries, or list corrections"
        ),
        "source_video": {
            "sha256": SOURCE_SHA256,
            "verified_at_build": video_sha == SOURCE_SHA256,
            "url": VIDEO_URL,
        },
        "pts_sha256": PTS_SHA256,
        "reviewed_artifacts": copied_sources,
        "authoritative_range_count": EXPECTED_RANGE_COUNT,
        "files": [
            {
                "path": path.relative_to(PACKET).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in bound
        ],
    }
    manifest_path = PACKET / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "packet_built_awaiting_human_review",
                "packet": str(PACKET),
                "files_bound": len(manifest["files"]),
                "manifest_sha256": sha256_file(manifest_path),
                "human_reviewed": False,
            },
            indent=2,
        )
    )


def write_review_md(
    ranges: list[list[float]],
    sampled: dict[int, str],
    special: dict[int, str],
    random_sample: list[int],
    spot_rows: list[dict],
    verification: dict,
    proposal: dict,
) -> None:
    durations = [end - start for start, end in ranges]
    total_s = sum(durations)
    bridge_population = verification["bridge_population"]
    range_rows = [row for row in spot_rows if row["kind"] == "range"]
    gap_rows = [row for row in spot_rows if row["kind"] == "excluded_gap"]
    bridge_rows = [row for row in spot_rows if row["kind"] == "included_bridge"]
    checklist = proposal["review_checklist"]

    lines = [
        "# ss3nhAUaScE gameplay-boundary human review",
        "",
        "Status: **awaiting a named human decision**. Nothing in this packet is",
        "an acceptance, a reviewed boundary set, or train-ready data.",
        "",
        "## What you are deciding",
        "",
        f"Whether to adopt the {len(ranges)} AI-proposed half-open gameplay",
        "ranges in `source_artifacts/boundaries.v3-ai.json` (byte-identical",
        "copy of `results/wild20/ss3nhAUaScE/boundaries.v3-ai.json`) as",
        "human-reviewed boundaries:",
        "",
        f"- {total_s:,.1f} s = {total_s / 3600:.4f} h of proposed gameplay",
        f"  inside the reviewed wall-clock envelope [{WALL_BOUNDS_S[0]},",
        f"  {WALL_BOUNDS_S[1]}) s;",
        "- the proposal's signal-quality gates all PASSED; it abstained only",
        "  because its ROI and wall-clock inputs were reviewed by an AI agent,",
        "  not a human (`review_provenance_gate_passed: false`) — exactly the",
        "  gap this review closes;",
        "- ranges come from official in-game total-timer activity: the timer",
        "  advances during play and freezes/disappears on loads, maps, menus,",
        "  and after the finish.",
        "",
        "## Review checklist (from the recorded proposal)",
        "",
    ]
    lines += [f"{i}. {item}" for i, item in enumerate(checklist, start=1)]
    lines += [
        "",
        "The wall-clock envelope edges themselves (LiveSplit 0.00 -> 0.01 at",
        "12.233/12.250 s; final official value first shown at 6867.083 s) are",
        "evidenced by the four exact frames in `source_artifacts/wall_clock/`,",
        "hash-bound in the layout evidence manifest.",
        "",
        "## Completeness assertions (all enforced by the generating scripts)",
        "",
        f"- `boundaries.v3-ai.json` holds exactly {len(ranges)} ranges; the",
        "  proposal's candidate accounting records the same total and is NOT",
        "  truncated (`candidate_ranges_before_gates: rows 106, total 106,",
        "  truncated false`).",
        "- `boundary_review_full.png` asserts `len(ranges) == 106` against the",
        "  authoritative artifact at render time and draws every range twice:",
        "  as a shaded span on the full-video trace and as one bar per range",
        "  in the indexed duration panel.",
        "- The full-resolution scalar trace behind the image was",
        "  independently re-decoded locally from the hash-verified source",
        "  video with the same OpenCV pipeline (decoder builds are not",
        "  bit-portable across hosts, so agreement is tolerance-bounded and",
        "  the measured deviations are recorded): all 512 recorded diagnostic",
        "  samples agree within "
        f"max change-score deviation "
        f"{verification['scalar_deviations_at_512_diagnostic_samples']['change_score']['max']:.2g}"
        " and",
        "  max mask-mean deviation "
        f"{max(verification['scalar_deviations_at_512_diagnostic_samples'][k]['max'] for k in ('bright_mask_mean', 'dark_mask_mean')):.3g}"
        " of 255, and an independent",
        "  re-segmentation reproduces the authoritative range set:",
        f"  {verification['independent_segmentation']['ranges_reproduced_exactly']}"
        f"/106 ranges float-exact, max edge deviation",
        f"  {verification['independent_segmentation']['max_edge_deviation_s']:.4f} s,"
        f" and {len(verification['independent_segmentation']['near_threshold_flips'])}"
        " near-threshold split/merge",
        "  flip (each recorded in `inputs/trace_verification.json`).",
        "- The proposal's bridge diagnostic list IS truncated (256 of its",
        f"  recorded {proposal['activity']['bridged_short_gaps']['total']:,} rows;"
        f" the independent re-decode finds",
        f"  {bridge_population['total_bridges']:,}), so the complete bridge"
        " population was recomputed; the",
        f"  {bridge_population['risky_bridges']:,} risky bridges",
        f"  ({bridge_population['risky_definition']}) are bound in",
        "  `inputs/bridge_risk_population.jsonl`.",
        "",
        "## Spot-check evidence",
        "",
        "Sampled ranges (deterministic: earliest, latest, longest, shortest,",
        f"second-longest, plus {len(random_sample)} random with seed",
        f"{RANDOM_SEED}): every page shows six exact frames — three at the",
        "start edge (timer must begin advancing) and three at the end edge",
        "(timer must freeze or disappear). Every displayed edge frame's",
        "timer-ROI scalars were re-measured and equal the verified trace, so",
        "the pages cannot silently show the wrong frame.",
        "",
        "| page | range | why | interval (video time) | duration | online check |",
        "|---|---|---|---|---|---|",
    ]
    for row in range_rows:
        lines.append(
            f"| `{row['page']}` | {row['range_index']} | {row['reason']} | "
            f"{row['start_hms']} .. {row['end_hms']} | {row['duration_s']:.2f} s | "
            f"[start]({row['start_url']}) [end]({row['end_url']}) |"
        )
    lines += [
        "",
        "Excluded-gap checks (timer must NOT advance anywhere inside):",
        "",
        "| page | between ranges | interval | duration |",
        "|---|---|---|---|",
    ]
    for row in gap_rows:
        lines.append(
            f"| `{row['page']}` | {row['after_range_index']} and "
            f"{row['after_range_index'] + 1} | {row['start_hms']} .. "
            f"{row['end_hms']} | {row['duration_s']:.2f} s |"
        )
    lines += [
        "",
        "Included-bridge checks (frames inside proposed ranges where the timer",
        "was briefly not advancing; must be gameplay freezes, not loads or",
        "menus):",
        "",
        "| page | inside range | at | duration | absent-timer frames |",
        "|---|---|---|---|---|",
    ]
    for row in bridge_rows:
        lines.append(
            f"| `{row['page']}` | {row['containing_range_index']} | "
            f"{row['start_hms']} | {row['duration_s'] * 1000:.0f} ms | "
            f"{row['absent_timer_frames']} |"
        )
    gap_durations = [
        ranges[i + 1][0] - ranges[i][1] for i in range(len(ranges) - 1)
    ]
    near_bridge_gaps = sum(1 for gap in gap_durations if gap < 0.6)
    short_ranges = sum(1 for duration in durations if duration < 6.0)
    lines += [
        "",
        "## Flagged for attention",
        "",
        "- **Range 82 (shortest, 2.13 s at 1:21:24.9)**: its frames sit",
        "  inside a black screen-wipe transition — the gameplay viewport is",
        "  mostly covered, and the file timer is only partially revealed",
        "  (`47.926` with the hour/minute glyphs still occluded) while its",
        "  visible digits advance. Look at `spot_checks/range-082.jpg` and",
        "  decide whether this 2-second transition sliver is gameplay worth",
        "  keeping; rejecting it costs 2.1 s.",
        f"- **{near_bridge_gaps} excluded gaps are barely over the 0.5 s",
        "  bridge limit** (typically 0.517 s): freezes of this length are",
        "  bridged when <= 0.5 s but split ranges when slightly longer, so",
        "  several adjacent ranges (for example 81/82/83) are conservative",
        "  splits of continuous play, not content changes. Approving the",
        "  split ranges as proposed only drops the frozen frames themselves.",
        "- **Range 105 is both the latest and the longest (440.2 s)** and",
        "  ends exactly at the reviewed wall end: its end edge is the",
        "  final-timer-value frame evidenced in `source_artifacts/wall_clock/`.",
        "- **The first ~1.6 s of play (12.25-13.87 s) are excluded** because",
        "  the official timer is not yet visible right after the LiveSplit",
        "  start; the proposal is conservative at the wall start.",
        f"- **{short_ranges} ranges are shorter than 6 s** (minimum 2.13 s;",
        "  policy floor 2.0 s) — brief gameplay slivers between deaths,",
        "  menus, or transitions. The shortest is spot-checked above.",
        "- The very large bridge population "
        f"({bridge_population['total_bridges']:,} gaps) is the normal",
        "  signature of this change-score detector (kdQbIoMxzZw showed the",
        "  same raw-to-bridged pattern); only",
        f"  {bridge_population['bridges_with_presence_dropout']} bridges",
        "  contain any timer-presence dropout. **The riskiest bridges are",
        "  full-screen wipe transitions (death/room changes) of exactly",
        "  500 ms with the timer absent for ~29 frames** — included inside",
        "  ranges by the bridging rule, as in the previously approved",
        "  videos; the top three are rendered above.",
        "",
        "## What approval unlocks",
        "",
        f"Adopting the ranges closes the boundary gate for {total_s / 3600:.2f}",
        "proposed gameplay hours: it authorizes decoding labels inside these",
        "ranges and running the dash-hitstop offset calibration for this",
        "video. It does NOT admit training data by itself — the offset gate",
        "and decode-time QC remain, and this video's current decode fails",
        "action QC (8-19% single-frame-run rates), which no boundary approval",
        "can waive.",
        "",
        "## Recording a decision",
        "",
        "Do not approve by implication and never edit `human_reviewed` flags.",
        "Either list corrections/rejected range indices, or approve with an",
        "explicit statement naming both hashes, for example:",
        "",
        "> I approve all 106 ss3nhAUaScE candidate half-open ranges from",
        f"> proposal `{verification['proposal_sha256']}`",
        "> after reviewing packet manifest `<review_manifest.json sha256>`.",
        "",
        "A separate step then materializes",
        "`boundaries.human-reviewed-<date>.json` with `reviewer_kind`",
        "`human_with_ai_assistance`, binding the reviewer name, the source",
        "hash, and the evidence consulted (as in the seven previously",
        "reviewed videos). This packet itself records no decision.",
        "",
    ]
    (PACKET / "REVIEW.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
