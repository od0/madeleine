"""Rank fetched NitroGen videos by long-context label and visual continuity.

There are two different failure modes:

* missing NitroGen chunks, which are measured from the chunk index without
  decoding video; and
* repeated/frozen visual frames inside an otherwise nominal 60-fps source,
  which require decoding pixels.

Visual scanning is resumable.  It decodes only labeled, native-60-Hz ranges,
masks the controller overlay, reduces frames to grayscale 32x18 fingerprints,
and records exact and near-duplicate consecutive-frame rates.  Exact duplicate
pixels are a strong frozen-frame signal, not proof of a capture fault: menus,
pause screens, hitstop, and legitimate static scenes can also produce them.
The output therefore ranks videos for review instead of declaring them clean.
"""

from __future__ import annotations

import argparse
from collections import deque
import concurrent.futures
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq


WRITE_LOCK = threading.Lock()
CHUNK_SECONDS = 20.0
LONG_CONTEXT_RAW_FRAMES = 382


@dataclass(frozen=True)
class LabelRun:
    start_s: float
    duration_s: float
    expected_frames: int


@dataclass(frozen=True)
class VideoPlan:
    video_id: str
    path: str
    bytes: int
    width: int
    height: int
    fps: float
    label_frames: int
    label_hours: float
    label_run_count: int
    long_context_targets: int
    long_context_fraction: float
    mask_rect: tuple[int, int, int, int] | None
    runs: tuple[LabelRun, ...]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _latest_fetch_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        rows[str(row["video_id"])] = row
    return rows


def _label_runs(rows: list[dict[str, Any]]) -> list[LabelRun]:
    """Join consecutive native-60-Hz NitroGen chunks into source-time runs."""

    ordered = sorted(rows, key=lambda row: int(row["chunk_id"]))
    if not ordered:
        return []
    groups: list[list[dict[str, Any]]] = []
    current = [ordered[0]]
    for row in ordered[1:]:
        previous = current[-1]
        if (
            int(row["chunk_id"]) == int(previous["chunk_id"]) + 1
            and float(row["grid_hz"]) == float(previous["grid_hz"])
        ):
            current.append(row)
        else:
            groups.append(current)
            current = [row]
    groups.append(current)

    runs: list[LabelRun] = []
    for group in groups:
        if float(group[0]["grid_hz"]) != 60.0:
            continue
        frames = sum(int(row["chunk_size"]) for row in group)
        runs.append(LabelRun(
            start_s=int(group[0]["chunk_id"]) * CHUNK_SECONDS,
            duration_s=frames / 60.0,
            expected_frames=frames,
        ))
    return runs


def build_plans(
    raw_root: Path,
    chunk_index: Path,
    fetch_report: Path,
) -> tuple[list[VideoPlan], list[dict[str, Any]]]:
    """Return metadata-valid scan plans and explicit rejected-video rows."""

    raw_paths = {
        path.stem: path
        for path in raw_root.iterdir()
        if path.suffix.lower() in {".mp4", ".webm", ".mkv"}
    }
    fetch = _latest_fetch_rows(fetch_report)
    chunk_rows = pq.read_table(chunk_index).to_pylist()
    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in chunk_rows:
        video_id = str(row["video_id"])
        if video_id in raw_paths:
            by_video.setdefault(video_id, []).append(row)

    plans: list[VideoPlan] = []
    rejected: list[dict[str, Any]] = []
    for video_id, path in sorted(raw_paths.items()):
        rows = by_video.get(video_id, [])
        fetch_row = fetch.get(video_id, {})
        reasons: list[str] = []
        if not rows:
            reasons.append("no_label_chunks")
        if rows and {float(row["grid_hz"]) for row in rows} != {60.0}:
            reasons.append("label_grid_not_uniform_60hz")
        if fetch_row.get("aligned_1to1") is not True:
            reasons.append("fetch_not_marked_1to1")
        if float(fetch_row.get("fps") or 0.0) < 59.0:
            reasons.append("video_fps_below_59")
        width = int(fetch_row.get("width") or 0)
        height = int(fetch_row.get("height") or 0)
        if width < 1 or height < 1:
            reasons.append("missing_video_dimensions")
        if reasons:
            rejected.append({"video_id": video_id, "reasons": reasons})
            continue

        runs = _label_runs(rows)
        label_frames = sum(run.expected_frames for run in runs)
        long_targets = sum(
            max(0, run.expected_frames - LONG_CONTEXT_RAW_FRAMES + 1)
            for run in runs
        )
        sample = rows[0]
        metadata_width = int(sample["metadata_resolution_w"])
        metadata_height = int(sample["metadata_resolution_h"])
        bbox_values = (
            sample.get("bbox_x"), sample.get("bbox_y"),
            sample.get("bbox_w"), sample.get("bbox_h"),
        )
        mask_rect: tuple[int, int, int, int] | None = None
        if all(value is not None for value in bbox_values):
            x, y, w, h = (int(value) for value in bbox_values)
            x0 = max(0, round(x * width / metadata_width))
            y0 = max(0, round(y * height / metadata_height))
            x1 = min(width, round((x + w) * width / metadata_width))
            y1 = min(height, round((y + h) * height / metadata_height))
            if x1 > x0 and y1 > y0:
                mask_rect = (x0, y0, x1 - x0, y1 - y0)

        plans.append(VideoPlan(
            video_id=video_id,
            path=str(path),
            bytes=int(path.stat().st_size),
            width=width,
            height=height,
            fps=float(fetch_row["fps"]),
            label_frames=label_frames,
            label_hours=label_frames / 216_000.0,
            label_run_count=len(runs),
            long_context_targets=long_targets,
            long_context_fraction=(long_targets / label_frames),
            mask_rect=mask_rect,
            runs=tuple(runs),
        ))
    return plans, rejected


def _scaled_mask_filter(
    plan: VideoPlan, width: int, height: int
) -> str:
    filters = [
        f"scale_cuda={width}:{height}",
        "hwdownload",
        "format=nv12",
        "format=gray",
    ]
    if plan.mask_rect is not None:
        x, y, w, h = plan.mask_rect
        x0 = max(0, round(x * width / plan.width))
        y0 = max(0, round(y * height / plan.height))
        x1 = min(width, round((x + w) * width / plan.width))
        y1 = min(height, round((y + h) * height / plan.height))
        if x1 > x0 and y1 > y0:
            filters.append(
                f"drawbox=x={x0}:y={y0}:w={x1-x0}:h={y1-y0}:"
                "color=black:t=fill"
            )
    return ",".join(filters)


def _scan_run(
    plan: VideoPlan,
    run: LabelRun,
    *,
    gpu: int,
    width: int,
    height: int,
    near_threshold: float,
) -> dict[str, int]:
    frame_bytes = width * height
    command = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-hwaccel", "cuda", "-hwaccel_device", str(gpu),
        "-hwaccel_output_format", "cuda",
        "-ss", f"{run.start_s:.6f}", "-i", plan.path,
        "-t", f"{run.duration_s:.6f}", "-an",
        "-vf", _scaled_mask_filter(plan, width, height),
        # The cloud image ships an older FFmpeg without ``-fps_mode``.
        # ``-vsync 0`` is its equivalent and prevents synthesized duplicates.
        "-vsync", "0", "-pix_fmt", "gray",
        "-f", "rawvideo", "pipe:1",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.stdout is not None
    assert process.stderr is not None
    # FFmpeg can emit one non-monotonic-DTS diagnostic per frame for some
    # otherwise decodable sources.  Waiting to read stderr until stdout reaches
    # EOF deadlocks once stderr's ~64-KiB pipe fills: FFmpeg cannot produce more
    # stdout, while this thread is waiting for stdout.  Drain concurrently and
    # retain only a bounded tail for a useful non-zero-exit error message.
    stderr_tail: deque[bytes] = deque(maxlen=64)

    def drain_stderr() -> None:
        while chunk := process.stderr.read(4096):
            stderr_tail.append(chunk)

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()
    previous: np.ndarray | None = None
    remainder = b""
    frames = pairs = exact = near = 0
    current_exact_run = longest_exact_run = 0
    while True:
        chunk = process.stdout.read(frame_bytes * 4096)
        if not chunk:
            break
        data = remainder + chunk
        count = len(data) // frame_bytes
        remainder = data[count * frame_bytes :]
        if count == 0:
            continue
        batch = np.frombuffer(
            data[: count * frame_bytes], dtype=np.uint8
        ).reshape(count, frame_bytes)
        if previous is not None:
            batch = np.vstack((previous, batch))
        if len(batch) > 1:
            difference = np.abs(
                batch[1:].astype(np.int16) - batch[:-1].astype(np.int16)
            )
            exact_flags = np.all(difference == 0, axis=1)
            near_flags = difference.mean(axis=1) <= near_threshold
            pairs += len(exact_flags)
            exact += int(exact_flags.sum())
            near += int(near_flags.sum())
            for flag in exact_flags:
                if flag:
                    current_exact_run += 1
                    longest_exact_run = max(longest_exact_run, current_exact_run)
                else:
                    current_exact_run = 0
        frames += count
        previous = batch[-1].copy()
    return_code = process.wait()
    stderr_thread.join()
    stderr = b"".join(stderr_tail).decode("utf-8", errors="replace")
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg exited {return_code}: {stderr[-500:].strip()}"
        )
    if remainder:
        raise RuntimeError(f"partial raw frame: {len(remainder)} bytes")
    return {
        "decoded_frames": frames,
        "pairs": pairs,
        "exact_pairs": exact,
        "near_pairs": near,
        "longest_exact_pair_run": longest_exact_run,
    }


def scan_video(
    plan: VideoPlan,
    *,
    gpu: int,
    width: int,
    height: int,
    near_threshold: float,
) -> dict[str, Any]:
    started = time.monotonic()
    totals = {
        "decoded_frames": 0,
        "pairs": 0,
        "exact_pairs": 0,
        "near_pairs": 0,
        "longest_exact_pair_run": 0,
    }
    try:
        for run in plan.runs:
            values = _scan_run(
                plan, run, gpu=gpu, width=width, height=height,
                near_threshold=near_threshold,
            )
            for key in ("decoded_frames", "pairs", "exact_pairs", "near_pairs"):
                totals[key] += values[key]
            totals["longest_exact_pair_run"] = max(
                totals["longest_exact_pair_run"],
                values["longest_exact_pair_run"],
            )
        elapsed = time.monotonic() - started
        pairs = totals["pairs"]
        return {
            **{key: value for key, value in asdict(plan).items() if key != "runs"},
            **totals,
            "exact_pair_fraction": totals["exact_pairs"] / max(1, pairs),
            "near_pair_fraction": totals["near_pairs"] / max(1, pairs),
            "decoded_to_expected": totals["decoded_frames"] / plan.label_frames,
            "scan_seconds": elapsed,
            "realtime_multiple": (plan.label_hours * 3600.0) / max(elapsed, 1e-9),
            "gpu": gpu,
            "error": None,
        }
    except Exception as error:  # noqa: BLE001 - error becomes audit evidence
        return {
            **{key: value for key, value in asdict(plan).items() if key != "runs"},
            **totals,
            "gpu": gpu,
            "error": f"{type(error).__name__}: {error}"[:1000],
            "scan_seconds": time.monotonic() - started,
        }


def _write_summary(
    path: Path,
    plans: list[VideoPlan],
    rejected: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
) -> None:
    successful = [row for row in visual_rows if row.get("error") is None]
    video_rows = visual_rows or [
        {
            **{
                key: value
                for key, value in asdict(plan).items()
                if key != "runs"
            },
            "scan_status": "not_run",
        }
        for plan in plans
    ]
    summary = {
        "long_context_raw_frames": LONG_CONTEXT_RAW_FRAMES,
        "metadata_valid_videos": len(plans),
        "metadata_rejected_videos": len(rejected),
        "label_hours": sum(plan.label_hours for plan in plans),
        "long_context_target_hours": (
            sum(plan.long_context_targets for plan in plans) / 216_000.0
        ),
        "weighted_long_context_fraction": (
            sum(plan.long_context_targets for plan in plans)
            / sum(plan.label_frames for plan in plans)
        ),
        "visual_scans_ok": len(successful),
        "visual_scans_failed": len(visual_rows) - len(successful),
        "visual_pairs": sum(int(row["pairs"]) for row in successful),
        "visual_exact_pairs": sum(
            int(row["exact_pairs"]) for row in successful
        ),
        "visual_near_pairs": sum(
            int(row["near_pairs"]) for row in successful
        ),
        "rejected": rejected,
        "videos": sorted(
            video_rows,
            key=lambda row: (
                row.get("error") is not None,
                float(row.get("exact_pair_fraction", 1.0)),
            ),
        ),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--chunk-index", type=Path, required=True)
    parser.add_argument("--fetch-report", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=18)
    parser.add_argument("--near-threshold", type=float, default=0.5)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    plans, rejected = build_plans(
        args.raw_root, args.chunk_index, args.fetch_report
    )
    if args.video_id:
        selected = set(args.video_id)
        plans = [plan for plan in plans if plan.video_id in selected]
    if args.limit:
        plans = plans[: args.limit]
    if args.metadata_only:
        _write_summary(args.summary, plans, rejected, [])
        print(
            f"metadata-valid={len(plans)} rejected={len(rejected)} "
            f"label-hours={sum(plan.label_hours for plan in plans):.2f}",
            flush=True,
        )
        return

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, Any]] = {}
    if args.out_jsonl.exists():
        for row in _read_jsonl(args.out_jsonl):
            done[str(row["video_id"])] = row
    todo = [plan for plan in plans if plan.video_id not in done]
    # Long videos first keeps the worker tail short.
    todo.sort(key=lambda plan: plan.label_frames, reverse=True)
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    print(
        f"scanning {len(todo)} videos ({len(done)} resumed), "
        f"{sum(plan.label_hours for plan in todo):.2f} label-hours, "
        f"workers={args.workers}, gpus={gpus}",
        flush=True,
    )

    with args.out_jsonl.open("a", encoding="utf-8") as output:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as pool:
            futures = {
                pool.submit(
                    scan_video,
                    plan,
                    gpu=gpus[index % len(gpus)],
                    width=args.width,
                    height=args.height,
                    near_threshold=args.near_threshold,
                ): plan
                for index, plan in enumerate(todo)
            }
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), 1
            ):
                row = future.result()
                with WRITE_LOCK:
                    output.write(json.dumps(row, sort_keys=True) + "\n")
                    output.flush()
                done[row["video_id"]] = row
                status = "ok" if row.get("error") is None else "ERROR"
                print(
                    f"[{completed}/{len(todo)}] {row['video_id']} {status} "
                    f"exact={row.get('exact_pair_fraction', float('nan')):.3%} "
                    f"speed={row.get('realtime_multiple', 0.0):.1f}x",
                    flush=True,
                )
    _write_summary(args.summary, plans, rejected, list(done.values()))
    print(f"wrote {args.summary}", flush=True)


if __name__ == "__main__":
    main()
