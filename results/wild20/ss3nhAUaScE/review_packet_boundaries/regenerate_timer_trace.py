"""Independently re-decode the ss3nhAUaScE official-timer trace and verify it.

The recorded trace manifest (`../timer-official-v3-ai/timer_trace_manifest.json`)
names a `timer_trace.npz` whose bytes were never kept locally or published to
durable storage, so this packet regenerates the scalar trace from the
hash-verified source video with the same OpenCV pixel pipeline as
`harvest.extract_timer_trace._stream_scalar_trace` over the exact recorded
frame interval and ROI.  The decode runs in parallel seeked chunks; every
chunk boundary frame is decoded twice (once by each neighbouring worker) and
the duplicate scalars must agree exactly, which fails closed on any seek or
decode inconsistency.

Decoder builds differ between this host and the original worker (the YUV to
BGR conversion inside OpenCV is not bit-portable), so float-exact equality
with the recorded proposal is not attainable: a first ffmpeg Y-plane attempt
deviated by up to 0.0042 on change scores, and the matching local OpenCV
decode still deviates by up to ~0.0024 at the recorded 512-point diagnostic
samples.  The verification is therefore:

- pts values must match the recorded diagnostic samples exactly (they come
  from the hash-verified PTS sidecar, not the decoder);
- scalar traces must agree at all 512 diagnostic samples within small
  explicit tolerances, with the actual maximum deviations recorded;
- an INDEPENDENT re-segmentation of the recomputed trace must reproduce the
  authoritative 106 ranges of `../boundaries.v3-ai.json` up to explicitly
  classified near-threshold differences: edge shifts within a strict
  tolerance, and split/merge flips only at gaps or islands within a small
  neighbourhood of the 0.5 s bridge limit / 2.0 s island floor, every one
  recorded for the reviewer.  Any unclassifiable difference fails the run.

It writes, under this packet directory only:

- `inputs/timer_trace.npz` — the recomputed full-resolution scalar trace;
- `inputs/trace_verification.json` — every check result, measured deviations,
  the classified segmentation differences, and hashes;
- `inputs/bridge_risk_population.jsonl` — the COMPLETE recomputed list of
  risky bridged gaps (any timer-presence dropout, or duration >= 0.35 s).
  The proposal itself persists only the first 256 of all bridge diagnostics,
  so per the review-round rule the complete inventory is recomputed and
  bound rather than trusting the truncated list.

A scalar cache is kept next to the shared raw-video cache so a verification
re-run never has to pay for the decode again.

No acceptance, no boundaries file, and no admission state is created.

Usage (from the repository root):

    uv run python results/wild20/ss3nhAUaScE/review_packet_boundaries/\
regenerate_timer_trace.py --video-dir <dir containing ss3nhAUaScE.mp4>
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PACKET = Path(__file__).resolve().parent
REPO = PACKET.parents[3]
sys.path.insert(0, str(REPO))

from harvest.extract_timer_trace import (  # noqa: E402
    _selected_interval,
    _write_trace_npz,
)
from harvest.timer_activity import (  # noqa: E402
    TimerActivityPolicy,
    _false_runs,
    _run_range,
    _threshold,
    _true_runs,
)

VIDEO_ID = "ss3nhAUaScE"
SOURCE_DIR = PACKET.parent
PROPOSAL_PATH = SOURCE_DIR / "timer-official-v3-ai" / "timer_activity_proposal.json"
TRACE_MANIFEST_PATH = SOURCE_DIR / "timer-official-v3-ai" / "timer_trace_manifest.json"
BOUNDARIES_PATH = SOURCE_DIR / "boundaries.v3-ai.json"

SOURCE_SHA256 = "b6b5302c5d79eafb8d96f289bdfd175a5b80c438e4d12f5725cd51823b715544"
PTS_SHA256 = "28b31524e24bac3484703ffc569306656ec232abfa36d46fca250359a8006ec1"
RECORDED_TRACE_NPZ_SHA256 = (
    "07feffd92723b788c72ff09fcc7f3f80f9f3ecc48de977a4948aee5e16190079"
)
WALL_BOUNDS_S = (12.233333, 6867.083333)
TIMER_ROI_NORMALIZED = (0.0140625, 0.041666666666666664, 0.1765625, 0.04722222222222222)
ROI_XYXY = (18, 30, 244, 64)
SOURCE_FRAME_RANGE = (734, 412025)
TOTAL_SOURCE_FRAMES = 420180
DECODE_WORKERS = 8

# Cross-build decoder tolerance at the 512 recorded diagnostic samples.
# change_score is on the normalized [0,1] pixel scale; the mask means are on
# the 0-255 scale.
CHANGE_TOLERANCE = 0.005
MASK_MEAN_TOLERANCE = 8.0
MIN_BOOL_SAMPLE_AGREEMENT = 500  # of 512
EDGE_TOLERANCE_S = 0.1  # six frames at 60 Hz; measured deviations recorded
# A split/merge flip is acceptable only at a gap within this neighbourhood of
# the 0.5 s bridge limit, or an island within it of the 2.0 s floor.
FLIP_NEIGHBOURHOOD_S = 0.15
MAX_CLASSIFIED_FLIPS = 6
MAX_COVERAGE_DIFFERENCE_S = 3.0

RISKY_BRIDGE_MIN_DURATION_S = 0.35


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(results: list, name: str, passed: bool, detail: str) -> None:
    results.append({"check": name, "passed": bool(passed), "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}", flush=True)
    if not passed:
        raise SystemExit(f"verification failed: {name}: {detail}")


def decode_chunk(payload: tuple) -> dict:
    """Decode scalars for one dense frame chunk through a seeked capture.

    Mirrors ``harvest.extract_timer_trace._stream_scalar_trace`` pixel-exactly:
    BGR2GRAY grayscale, ROI slice, normalized mean-absolute change, and 0-255
    bright/dark threshold-mask means.  For every chunk except the global
    first, the frame before the chunk is decoded too: it provides the
    previous frame for the chunk's first change score and a duplicate scalar
    pair the assembler checks against the neighbouring chunk.
    """

    video, start_index, end_index, global_first = payload
    x0, y0, x1, y1 = ROI_XYXY
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise ValueError(f"cannot open source video: {video}")
    try:
        is_global_first = start_index == global_first
        seek_to = start_index if is_global_first else start_index - 1
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(seek_to)):
            raise ValueError(f"OpenCV could not seek to source frame {seek_to}")
        position = capture.get(cv2.CAP_PROP_POS_FRAMES)
        if np.isfinite(position) and abs(position - seek_to) > 0.5:
            raise ValueError(f"seek landed at {position}, expected {seek_to}")
        count = end_index - start_index
        change = np.zeros(count, dtype=np.float64)
        bright = np.empty(count, dtype=np.float64)
        dark = np.empty(count, dtype=np.float64)
        previous = None
        boundary_bright = boundary_dark = None
        for source_index in range(seek_to, end_index):
            ok, frame = capture.read()
            if not ok or frame.ndim != 3 or frame.dtype != np.uint8:
                raise ValueError(f"decode failed at source frame {source_index}")
            position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if (
                np.isfinite(position)
                and position > 0
                and abs(position - (source_index + 1)) > 0.5
            ):
                raise ValueError(
                    f"decode position {position} differs from expected "
                    f"{source_index + 1}"
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[y0:y1, x0:x1]
            normalized = roi.astype(np.float32) / 255.0
            row = source_index - start_index
            if row < 0:
                boundary_bright = float(
                    np.mean(
                        (roi >= 200.0).astype(np.float32) * 255.0, dtype=np.float64
                    )
                )
                boundary_dark = float(
                    np.mean(
                        (roi <= 32.0).astype(np.float32) * 255.0, dtype=np.float64
                    )
                )
            else:
                if previous is not None:
                    change[row] = np.mean(np.abs(normalized - previous))
                bright[row] = np.mean(
                    (roi >= 200.0).astype(np.float32) * 255.0, dtype=np.float64
                )
                dark[row] = np.mean(
                    (roi <= 32.0).astype(np.float32) * 255.0, dtype=np.float64
                )
            previous = normalized
    finally:
        capture.release()
    return {
        "start": start_index,
        "end": end_index,
        "change": change,
        "bright": bright,
        "dark": dark,
        "boundary_bright": boundary_bright,
        "boundary_dark": boundary_dark,
    }


def parallel_decode(
    video: Path, indices: np.ndarray, cache_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    first, last_exclusive = int(indices[0]), int(indices[-1]) + 1
    if cache_path.exists():
        cached = np.load(cache_path)
        if np.array_equal(cached["source_frame_idx"], indices):
            print(f"loaded scalar cache {cache_path}", flush=True)
            return (
                cached["change_score"],
                cached["bright_mask_mean"],
                cached["dark_mask_mean"],
                {"decode_backend": "opencv-parallel-chunks", "cache_hit": True},
            )
        raise ValueError(f"scalar cache {cache_path} indexes different frames")

    edges = np.linspace(first, last_exclusive, DECODE_WORKERS + 1).astype(int)
    chunks = [
        (str(video), int(edges[i]), int(edges[i + 1]), first)
        for i in range(DECODE_WORKERS)
        if edges[i + 1] > edges[i]
    ]
    print(
        f"decoding {last_exclusive - first} ROI frames in {len(chunks)} "
        "parallel seeked chunks...",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=DECODE_WORKERS) as pool:
        results = list(pool.map(decode_chunk, chunks))
    results.sort(key=lambda row: row["start"])
    for left, right in zip(results, results[1:]):
        if left["end"] != right["start"]:
            raise ValueError("decoded chunks are not contiguous")
        # The boundary frame was decoded by both workers; scalars must agree.
        if (
            right["boundary_bright"] != left["bright"][-1]
            or right["boundary_dark"] != left["dark"][-1]
        ):
            raise ValueError(
                "cross-chunk duplicate-decode mismatch at source frame "
                f"{right['start'] - 1}: "
                f"{(right['boundary_bright'], right['boundary_dark'])} != "
                f"{(left['bright'][-1], left['dark'][-1])}"
            )
    change = np.concatenate([row["change"] for row in results])
    bright = np.concatenate([row["bright"] for row in results])
    dark = np.concatenate([row["dark"] for row in results])
    if change.size != indices.size:
        raise ValueError("assembled scalar length mismatch")
    np.savez_compressed(
        cache_path,
        source_frame_idx=indices,
        change_score=change,
        bright_mask_mean=bright,
        dark_mask_mean=dark,
    )
    return change, bright, dark, {
        "decode_backend": "opencv-parallel-chunks",
        "cache_hit": False,
        "chunks": len(chunks),
        "cross_chunk_boundary_checks": len(chunks) - 1,
    }


def classify_differences(
    recorded: list[list[float]],
    recomputed: list[list[float]],
) -> tuple[list[dict], list[dict], float]:
    """Align two candidate-range lists, classifying near-threshold flips.

    Returns (matches, flips, max_edge_deviation).  Exits on any difference
    that is not an edge shift within tolerance, a split at a near-limit gap,
    a merge of a near-limit recorded gap, or an island flip near the 2 s
    duration floor.
    """

    matches: list[dict] = []
    flips: list[dict] = []
    max_edge_dev = 0.0
    i = j = 0
    while i < len(recorded) or j < len(recomputed):
        record = recorded[i] if i < len(recorded) else None
        candidate = recomputed[j] if j < len(recomputed) else None
        if record is not None and candidate is not None:
            start_dev = abs(candidate[0] - record[0])
            end_dev = abs(candidate[1] - record[1])
            if start_dev <= EDGE_TOLERANCE_S and end_dev <= EDGE_TOLERANCE_S:
                max_edge_dev = max(max_edge_dev, start_dev, end_dev)
                matches.append(
                    {
                        "recorded_index": i,
                        "start_dev_s": start_dev,
                        "end_dev_s": end_dev,
                        "exact": start_dev == 0.0 and end_dev == 0.0,
                    }
                )
                i += 1
                j += 1
                continue
            # Split flip: a recomputed pair covers one recorded range with an
            # internal gap near the 0.5 s bridge limit.
            if (
                start_dev <= EDGE_TOLERANCE_S
                and j + 1 < len(recomputed)
                and abs(recomputed[j + 1][1] - record[1]) <= EDGE_TOLERANCE_S
            ):
                gap = recomputed[j + 1][0] - candidate[1]
                if abs(gap - 0.5) <= FLIP_NEIGHBOURHOOD_S:
                    flips.append(
                        {
                            "kind": "split_at_near_limit_gap",
                            "recorded_index": i,
                            "recorded_s": record,
                            "recomputed_s": [candidate, recomputed[j + 1]],
                            "internal_gap_s": gap,
                        }
                    )
                    i += 1
                    j += 2
                    continue
            # Merge flip: one recomputed range covers two recorded ranges
            # whose recorded gap is near the bridge limit.
            if (
                start_dev <= EDGE_TOLERANCE_S
                and i + 1 < len(recorded)
                and abs(candidate[1] - recorded[i + 1][1]) <= EDGE_TOLERANCE_S
            ):
                gap = recorded[i + 1][0] - record[1]
                if abs(gap - 0.5) <= FLIP_NEIGHBOURHOOD_S:
                    flips.append(
                        {
                            "kind": "merge_of_near_limit_gap",
                            "recorded_indices": [i, i + 1],
                            "recorded_s": [record, recorded[i + 1]],
                            "recomputed_s": candidate,
                            "recorded_gap_s": gap,
                        }
                    )
                    i += 2
                    j += 1
                    continue
        # Island flips near the 2.0 s duration floor.
        if candidate is not None and (
            record is None or candidate[1] <= record[0] + EDGE_TOLERANCE_S
        ):
            duration = candidate[1] - candidate[0]
            if abs(duration - 2.0) <= FLIP_NEIGHBOURHOOD_S:
                flips.append(
                    {
                        "kind": "extra_island_near_duration_floor",
                        "recomputed_s": candidate,
                        "duration_s": duration,
                    }
                )
                j += 1
                continue
        if record is not None and (
            candidate is None or record[1] <= candidate[0] + EDGE_TOLERANCE_S
        ):
            duration = record[1] - record[0]
            if abs(duration - 2.0) <= FLIP_NEIGHBOURHOOD_S:
                flips.append(
                    {
                        "kind": "missing_island_near_duration_floor",
                        "recorded_index": i,
                        "recorded_s": record,
                        "duration_s": duration,
                    }
                )
                i += 1
                continue
        raise SystemExit(
            "unclassifiable segmentation difference near recorded index "
            f"{i} ({record}) / recomputed index {j} ({candidate})"
        )
    return matches, flips, max_edge_dev


def coverage_difference_s(
    recorded: list[list[float]], recomputed: list[list[float]]
) -> float:
    """Total length of the symmetric difference between the two range sets."""

    events: list[tuple[float, int, int]] = []
    for start, end in recorded:
        events.append((start, 0, 1))
        events.append((end, 0, -1))
    for start, end in recomputed:
        events.append((start, 1, 1))
        events.append((end, 1, -1))
    events.sort()
    depth = [0, 0]
    previous_position = None
    total = 0.0
    for position, which, delta in events:
        if previous_position is not None and (depth[0] > 0) != (depth[1] > 0):
            total += position - previous_position
        depth[which] += delta
        previous_position = position
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="directory holding the hash-verified ss3nhAUaScE.mp4 + frame_pts.npy",
    )
    args = parser.parse_args()

    inputs_dir = PACKET / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    results: list = []

    proposal = json.loads(PROPOSAL_PATH.read_text())
    manifest = json.loads(TRACE_MANIFEST_PATH.read_text())
    boundaries = json.loads(BOUNDARIES_PATH.read_text())

    check(
        results,
        "boundaries_source_binding",
        boundaries["source_sha256"] == SOURCE_SHA256
        and boundaries["video_id"] == VIDEO_ID
        and boundaries["human_reviewed"] is False,
        "boundaries.v3-ai.json binds the expected source hash and remains unreviewed",
    )
    authoritative_ranges = [
        [float(a), float(b)] for a, b in boundaries["allowed_ranges_s"]
    ]
    recorded_count = int(
        proposal["activity"]["candidate_ranges_before_gates"]["total"]
    )
    check(
        results,
        "authoritative_range_count",
        len(authoritative_ranges) == 106 == recorded_count
        and proposal["activity"]["candidate_ranges_before_gates"]["truncated"] is False,
        f"boundaries.v3-ai.json has {len(authoritative_ranges)} ranges; proposal "
        f"records {recorded_count} candidates (untruncated)",
    )
    check(
        results,
        "recorded_ranges_equal_proposal_rows",
        [
            [float(a), float(b)]
            for a, b in (
                row["range_s"]
                for row in proposal["activity"]["candidate_ranges_before_gates"]["rows"]
            )
        ]
        == authoritative_ranges,
        "boundaries.v3-ai.json equals the proposal's untruncated candidate rows",
    )

    video_path = args.video_dir / f"{VIDEO_ID}.mp4"
    pts_path = args.video_dir / "frame_pts.npy"
    video_sha = sha256_file(video_path)
    check(
        results,
        "source_video_sha256",
        video_sha == SOURCE_SHA256,
        f"{video_path.name} sha256 {video_sha}",
    )
    pts_sha = sha256_file(pts_path)
    check(
        results,
        "pts_vector_sha256",
        pts_sha == PTS_SHA256,
        f"{pts_path.name} sha256 {pts_sha}",
    )

    policy = TimerActivityPolicy(**proposal["policy"])
    policy.validate()

    all_pts = np.load(pts_path)
    check(
        results,
        "pts_vector_shape",
        all_pts.ndim == 1 and all_pts.size == TOTAL_SOURCE_FRAMES,
        f"{all_pts.size} persisted PTS values",
    )
    indices, selected_pts, _summary = _selected_interval(
        all_pts, WALL_BOUNDS_S, policy
    )
    check(
        results,
        "selected_interval",
        int(indices[0]) == SOURCE_FRAME_RANGE[0]
        and int(indices[-1]) + 1 == SOURCE_FRAME_RANGE[1]
        and indices.size == int(proposal["trace_binding"]["frames"]),
        f"dense source frames [{int(indices[0])}, {int(indices[-1]) + 1}) "
        f"({indices.size} frames)",
    )
    expected_roi = [
        round(TIMER_ROI_NORMALIZED[0] * 1280),
        round(TIMER_ROI_NORMALIZED[1] * 720),
        round((TIMER_ROI_NORMALIZED[0] + TIMER_ROI_NORMALIZED[2]) * 1280),
        round((TIMER_ROI_NORMALIZED[1] + TIMER_ROI_NORMALIZED[3]) * 720),
    ]
    check(
        results,
        "roi_pixel_geometry",
        list(ROI_XYXY) == expected_roi
        and list(ROI_XYXY) == list(manifest["selection"]["timer_roi_pixels_xyxy"]),
        f"ROI pixels xyxy {list(ROI_XYXY)}",
    )

    cache_path = args.video_dir.parent / "timer_trace_scalar_cache_opencv.npz"
    change, bright, dark, decode_info = parallel_decode(
        video_path, indices, cache_path
    )

    # --- bounded agreement with the 512-point recorded diagnostic trace -------
    diag = proposal["diagnostic_trace"]
    sample = np.asarray(diag["frame_index_in_reviewed_bounds"], dtype=np.int64)
    recorded_pts = np.asarray(diag["pts_s"], dtype=np.float64)
    check(
        results,
        "diagnostic_trace_pts_exact",
        np.array_equal(recorded_pts, selected_pts[sample]),
        "512/512 sampled PTS values match exactly",
    )
    scalar_deviations = {}
    for name, recomputed, tolerance in (
        ("change_score", change, CHANGE_TOLERANCE),
        ("bright_mask_mean", bright, MASK_MEAN_TOLERANCE),
        ("dark_mask_mean", dark, MASK_MEAN_TOLERANCE),
    ):
        recorded_values = np.asarray(diag[name], dtype=np.float64)
        deviation = np.abs(recorded_values - recomputed[sample])
        scalar_deviations[name] = {
            "max": float(deviation.max()),
            "mean": float(deviation.mean()),
            "tolerance": tolerance,
            "exact_matches": int(np.count_nonzero(deviation == 0)),
        }
        check(
            results,
            f"diagnostic_trace_{name}_within_tolerance",
            float(deviation.max()) <= tolerance,
            f"max |dev| {deviation.max():.4g} (mean {deviation.mean():.4g}, "
            f"tolerance {tolerance}) across 512 samples; decoder builds "
            "differ between hosts so exact equality is not expected",
        )

    # --- independent segmentation at full resolution --------------------------
    median_dt = float(np.median(np.diff(selected_pts)))
    present = (bright >= policy.min_bright_mask_mean) & (
        dark >= policy.min_dark_mask_mean
    )
    motion_eligible = present.copy()
    motion_eligible[0] = False
    motion_eligible[1:] &= present[:-1]
    threshold, threshold_diag = _threshold(change[motion_eligible], policy)
    check(
        results,
        "activity_threshold_bimodal",
        bool(threshold_diag["bimodal"]),
        f"recomputed threshold {threshold:.6g} (recorded "
        f"{proposal['threshold']['threshold']:.6g}); bimodality check "
        f"{threshold_diag['check']}",
    )
    raw_active = motion_eligible & (change >= threshold)

    bridged = raw_active.copy()
    bridge_rows = []
    for start, end in _false_runs(raw_active):
        gap_start, gap_end = _run_range(
            start, end, selected_pts, median_dt, WALL_BOUNDS_S
        )
        duration = gap_end - gap_start
        is_internal = start > 0 and end < raw_active.size
        if is_internal and duration <= policy.max_bridge_s + 1e-9:
            absent_frames = int(np.count_nonzero(~present[start:end]))
            bridged[start:end] = True
            bridge_rows.append(
                {
                    "range_s": [gap_start, gap_end],
                    "duration_s": duration,
                    "frames": int(end - start),
                    "absent_timer_frames": absent_frames,
                }
            )

    long_inactive = []
    for start, end in _false_runs(bridged):
        inactive_start, inactive_end = _run_range(
            start, end, selected_pts, median_dt, WALL_BOUNDS_S
        )
        long_inactive.append(
            {
                "range_s": [inactive_start, inactive_end],
                "duration_s": inactive_end - inactive_start,
                "frames": int(end - start),
                "absent_timer_frames": int(np.count_nonzero(~present[start:end])),
            }
        )

    candidates = []
    dropped = []
    for start, end in _true_runs(bridged):
        allowed_start, allowed_end = _run_range(
            start, end, selected_pts, median_dt, WALL_BOUNDS_S
        )
        duration = allowed_end - allowed_start
        if duration < policy.min_allowed_s:
            dropped.append([allowed_start, allowed_end])
        else:
            candidates.append([allowed_start, allowed_end])

    population_comparison = {
        "motion_eligible_frames": [
            int(np.count_nonzero(motion_eligible)),
            int(proposal["activity"]["motion_eligible_frames"]),
        ],
        "raw_active_frames": [
            int(np.count_nonzero(raw_active)),
            int(proposal["activity"]["raw_active_frames"]),
        ],
        "bridged_active_frames": [
            int(np.count_nonzero(bridged)),
            int(proposal["activity"]["bridged_active_frames"]),
        ],
        "bridge_total": [
            len(bridge_rows),
            int(proposal["activity"]["bridged_short_gaps"]["total"]),
        ],
        "long_inactive_total": [
            len(long_inactive),
            int(proposal["activity"]["long_inactive_ranges"]["total"]),
        ],
        "dropped_islands_total": [
            len(dropped),
            int(proposal["activity"]["dropped_short_activity_islands"]["total"]),
        ],
    }
    for name, (recomputed_value, recorded_value) in population_comparison.items():
        relative = abs(recomputed_value - recorded_value) / max(recorded_value, 1)
        check(
            results,
            f"population_{name}",
            relative <= 0.02,
            f"recomputed {recomputed_value} vs recorded {recorded_value} "
            f"({100 * relative:.3f}% relative deviation, limit 2%)",
        )

    matches, flips, max_edge_dev = classify_differences(
        authoritative_ranges, candidates
    )
    coverage_diff = coverage_difference_s(authoritative_ranges, candidates)
    exact_matches = sum(1 for row in matches if row["exact"])
    check(
        results,
        "independent_segmentation_alignment",
        len(flips) <= MAX_CLASSIFIED_FLIPS
        and max_edge_dev <= EDGE_TOLERANCE_S
        and coverage_diff <= MAX_COVERAGE_DIFFERENCE_S,
        f"independent re-decode produced {len(candidates)} ranges; "
        f"{exact_matches}/{len(authoritative_ranges)} recorded ranges "
        f"reproduced float-exactly, {len(matches) - exact_matches} with edge "
        f"shifts (max {max_edge_dev:.4f} s), {len(flips)} near-threshold "
        f"split/merge/island flips (limit {MAX_CLASSIFIED_FLIPS}), total "
        f"coverage difference {coverage_diff:.3f} s of "
        f"{sum(b - a for a, b in authoritative_ranges):.1f} s "
        f"(limit {MAX_COVERAGE_DIFFERENCE_S} s)",
    )

    bool_agreement = {}
    for name, values in (
        ("timer_present", present),
        ("motion_eligible", motion_eligible),
        ("raw_active", raw_active),
        ("bridged_active", bridged),
    ):
        recorded_bools = np.asarray(diag[name], dtype=bool)
        agreeing = int(np.count_nonzero(recorded_bools == values[sample]))
        bool_agreement[name] = agreeing
        check(
            results,
            f"diagnostic_trace_{name}_agreement",
            agreeing >= MIN_BOOL_SAMPLE_AGREEMENT,
            f"{agreeing}/512 sampled boolean states agree "
            f"(minimum {MIN_BOOL_SAMPLE_AGREEMENT}; near-threshold samples "
            "may flip under cross-build decoder deviations)",
        )

    # --- persist the recomputed trace ----------------------------------------
    trace_path = inputs_dir / "timer_trace.npz"
    if trace_path.exists():
        raise SystemExit(f"refusing to overwrite {trace_path}")
    _write_trace_npz(trace_path, indices, selected_pts, change, bright, dark)
    trace_sha = sha256_file(trace_path)

    risky = [
        row
        for row in bridge_rows
        if row["absent_timer_frames"] > 0
        or row["duration_s"] >= RISKY_BRIDGE_MIN_DURATION_S
    ]
    risk_path = inputs_dir / "bridge_risk_population.jsonl"
    if risk_path.exists():
        raise SystemExit(f"refusing to overwrite {risk_path}")
    with risk_path.open("w") as handle:
        for row in risky:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    durations = np.asarray([row["duration_s"] for row in bridge_rows])
    verification = {
        "format_version": "madeleine.wild20-trace-regeneration-verification.v3",
        "video_id": VIDEO_ID,
        "status": "independently_regenerated_and_verified_within_tolerance",
        "human_reviewed": False,
        "source_sha256": SOURCE_SHA256,
        "pts_sha256": PTS_SHA256,
        "proposal_sha256": sha256_file(PROPOSAL_PATH),
        "trace_manifest_sha256": sha256_file(TRACE_MANIFEST_PATH),
        "boundaries_v3_ai_sha256": sha256_file(BOUNDARIES_PATH),
        "recomputed_trace_npz_sha256": trace_sha,
        "recorded_trace_npz_sha256": RECORDED_TRACE_NPZ_SHA256,
        "trace_npz_byte_identical_to_recorded": trace_sha
        == RECORDED_TRACE_NPZ_SHA256,
        "reproduction_note": (
            "The recorded worker trace bytes were never preserved; this trace "
            "is an independent local re-decode with the same OpenCV pixel "
            "pipeline (parallel seeked chunks, duplicate-decoded boundary "
            "frames checked exactly). Decoder builds are not bit-portable, so "
            "agreement with the recorded proposal is verified within recorded "
            "tolerances and by reproduction of the authoritative range set up "
            "to explicitly classified near-threshold flips."
        ),
        "decode": decode_info,
        "scalar_deviations_at_512_diagnostic_samples": scalar_deviations,
        "boolean_agreement_at_512_diagnostic_samples": bool_agreement,
        "recomputed_threshold": threshold,
        "recorded_threshold": proposal["threshold"]["threshold"],
        "population_comparison_recomputed_vs_recorded": population_comparison,
        "independent_segmentation": {
            "range_count": len(candidates),
            "authoritative_range_count": len(authoritative_ranges),
            "ranges_reproduced_exactly": exact_matches,
            "edge_shifted_matches": len(matches) - exact_matches,
            "max_edge_deviation_s": max_edge_dev,
            "edge_tolerance_s": EDGE_TOLERANCE_S,
            "near_threshold_flips": flips,
            "flip_neighbourhood_s": FLIP_NEIGHBOURHOOD_S,
            "coverage_difference_s": coverage_diff,
        },
        "checks": results,
        "bridge_population": {
            "total_bridges": len(bridge_rows),
            "risky_bridges": len(risky),
            "risky_definition": (
                "absent_timer_frames > 0 or duration_s >= "
                f"{RISKY_BRIDGE_MIN_DURATION_S}"
            ),
            "bridges_with_presence_dropout": int(
                sum(1 for row in bridge_rows if row["absent_timer_frames"] > 0)
            ),
            "duration_histogram_s": {
                "edges": [0.0, 0.05, 0.1, 0.2, 0.35, 0.5],
                "counts": [
                    int(value)
                    for value in np.histogram(
                        durations, bins=[0.0, 0.05, 0.1, 0.2, 0.35, 0.5 + 1e-9]
                    )[0]
                ],
            },
            "note": (
                "The proposal's bridged_short_gaps list is truncated to 256 "
                f"of {len(bridge_rows)} rows; this complete population was "
                "recomputed from the verified full-resolution trace."
            ),
        },
    }
    verification_path = inputs_dir / "trace_verification.json"
    if verification_path.exists():
        raise SystemExit(f"refusing to overwrite {verification_path}")
    verification_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "trace": str(trace_path),
                "verification": str(verification_path),
                "ranges_reproduced_exactly": exact_matches,
                "near_threshold_flips": len(flips),
                "max_edge_deviation_s": max_edge_dev,
                "coverage_difference_s": coverage_diff,
                "risky_bridges": len(risky),
                "total_bridges": len(bridge_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
