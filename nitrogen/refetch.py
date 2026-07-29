"""Refetch NitroGen source videos at video-level granularity."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Sequence, TypedDict

import cv2
import yt_dlp


CHUNK_SECONDS = 20
DOWNLOAD_FORMAT = "bv*[height<=480]/b[height<=480]/b"
REPORT_NAME = "fetch_report.jsonl"
SOCKET_TIMEOUT_SECONDS = 30
MAX_WORKERS = 3
SUPPORTED_SOURCES = ("youtube", "twitch")


class VideoRecord(TypedDict):
    video_id: str
    source: str
    url: str
    metadata_resolution: list[int]
    n_chunks: int
    chunk_hours: float


class FetchReport(TypedDict):
    video_id: str
    game: str
    source: str
    url: str
    status: str
    error: str | None
    metadata_resolution: list[int]
    fetched_width: int | None
    fetched_height: int | None
    fetched_fps: float | None
    fetched_frames: int | None
    n_chunks: int
    chunk_hours: float
    fetched_at: str


def discover_videos(actions_root: str | Path, game: str) -> list[VideoRecord]:
    """Discover one record per video, filtered by exact metadata game."""

    root = Path(actions_root)
    records: list[VideoRecord] = []
    seen_video_ids: set[str] = set()

    for shard_dir in sorted(path for path in root.glob("SHARD_*") if path.is_dir()):
        for video_dir in sorted(path for path in shard_dir.iterdir() if path.is_dir()):
            video_id = video_dir.name
            chunk_prefix = f"{video_id}_chunk_"
            with os.scandir(video_dir) as entries:
                chunk_dirs = [
                    video_dir / name
                    for name in sorted(
                        entry.name
                        for entry in entries
                        if entry.is_dir() and entry.name.startswith(chunk_prefix)
                    )
                ]
            metadata_path = next(
                (
                    chunk_dir / "metadata.json"
                    for chunk_dir in chunk_dirs
                    if (chunk_dir / "metadata.json").is_file()
                ),
                None,
            )
            if metadata_path is None:
                continue

            # Every chunk repeats the video fields, so only the first is read.
            with metadata_path.open(encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            if metadata["game"] != game:
                continue

            original_video = metadata["original_video"]
            source = original_video["source"]
            if source not in SUPPORTED_SOURCES:
                raise ValueError(
                    f"unsupported source {source!r} in {metadata_path}"
                )
            if video_id in seen_video_ids:
                continue
            seen_video_ids.add(video_id)

            # NitroGen resolution is [height, width], not [width, height].
            metadata_resolution = list(original_video["resolution"])
            n_chunks = len(chunk_dirs)
            records.append(
                {
                    "video_id": video_id,
                    "source": source,
                    "url": original_video["url"],
                    "metadata_resolution": metadata_resolution,
                    "n_chunks": n_chunks,
                    "chunk_hours": n_chunks * CHUNK_SECONDS / 3600,
                }
            )

    return records


def sample_videos(
    candidates: Sequence[VideoRecord],
    n_videos: int,
    source_priority: Sequence[str] | str,
    seed: int,
) -> list[VideoRecord]:
    """Shuffle within source groups, preserving source-priority ordering."""

    if n_videos < 0:
        raise ValueError("n_videos must be non-negative")
    priorities = _parse_source_priority(source_priority)
    rng = random.Random(seed)
    sampled: list[VideoRecord] = []

    for source in priorities:
        group = sorted(
            (record for record in candidates if record["source"] == source),
            key=lambda record: record["video_id"],
        )
        rng.shuffle(group)
        sampled.extend(group)

    return sampled[:n_videos]


def _parse_source_priority(value: Sequence[str] | str) -> list[str]:
    if isinstance(value, str):
        priorities = [part.strip() for part in value.split(",") if part.strip()]
    else:
        priorities = list(value)
    if not priorities:
        raise ValueError("source priority cannot be empty")
    if len(set(priorities)) != len(priorities):
        raise ValueError("source priority cannot contain duplicates")
    unsupported = [source for source in priorities if source not in SUPPORTED_SOURCES]
    if unsupported:
        raise ValueError(f"unsupported source priority: {', '.join(unsupported)}")
    return priorities


def _retry_sleep_seconds(retry_number: int) -> int:
    """Return the requested 5s/15s delay for yt-dlp's two retries."""

    return 5 if retry_number == 0 else 15


def _download_options(record: VideoRecord, out_dir: Path) -> dict[str, Any]:
    retry_sleep_functions = {
        retry_type: _retry_sleep_seconds
        for retry_type in ("http", "fragment", "file_access", "extractor")
    }
    return {
        "format": DOWNLOAD_FORMAT,
        "outtmpl": str(out_dir / f"{record['video_id']}.%(ext)s"),
        "noplaylist": True,
        "socket_timeout": SOCKET_TIMEOUT_SECONDS,
        "retries": 2,
        "fragment_retries": 2,
        "extractor_retries": 2,
        "file_access_retries": 2,
        "retry_sleep_functions": retry_sleep_functions,
        "continuedl": True,
        "overwrites": False,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }


def _find_existing_video(out_dir: Path, video_id: str) -> Path | None:
    for path in sorted(out_dir.glob(f"{video_id}.*")):
        if (
            path.is_file()
            and path.stat().st_size > 0
            and path.name != REPORT_NAME
            and path.suffix not in {".part", ".ytdl", ".temp", ".json", ".jsonl"}
        ):
            return path
    return None


def _probe_video(path: Path) -> tuple[int, int, float, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cv2 could not open fetched video: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    if width <= 0 or height <= 0 or fps <= 0 or frames <= 0:
        raise RuntimeError(
            "cv2 returned invalid video properties "
            f"for {path}: width={width}, height={height}, fps={fps}, frames={frames}"
        )
    return width, height, fps, frames


def _base_report(record: VideoRecord, game: str) -> FetchReport:
    return {
        "video_id": record["video_id"],
        "game": game,
        "source": record["source"],
        "url": record["url"],
        "status": "failed",
        "error": None,
        "metadata_resolution": record["metadata_resolution"],
        "fetched_width": None,
        "fetched_height": None,
        "fetched_fps": None,
        "fetched_frames": None,
        "n_chunks": record["n_chunks"],
        "chunk_hours": record["chunk_hours"],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _fetch_one(record: VideoRecord, game: str, out_dir: Path) -> FetchReport:
    report = _base_report(record, game)
    try:
        video_path = _find_existing_video(out_dir, record["video_id"])
        if video_path is None:
            with yt_dlp.YoutubeDL(_download_options(record, out_dir)) as downloader:
                downloader.extract_info(record["url"], download=True)
            video_path = _find_existing_video(out_dir, record["video_id"])
            if video_path is None:
                raise RuntimeError("yt-dlp completed without producing a video file")

        width, height, fps, frames = _probe_video(video_path)
        report.update(
            {
                "status": "ok",
                "fetched_width": width,
                "fetched_height": height,
                "fetched_fps": fps,
                "fetched_frames": frames,
            }
        )
    except Exception as error:
        report["error"] = str(error) or type(error).__name__
    report["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return report


def _load_ok_video_ids(report_path: Path) -> set[str]:
    if not report_path.exists():
        return set()

    ok_video_ids: set[str] = set()
    with report_path.open(encoding="utf-8") as report_file:
        for line_number, line in enumerate(report_file, start=1):
            if not line.strip():
                continue
            try:
                report = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {report_path} line {line_number}: {error}"
                ) from error
            if report.get("status") == "ok":
                ok_video_ids.add(report["video_id"])
    return ok_video_ids


def _append_report(report_path: Path, report: FetchReport) -> None:
    with report_path.open("a", encoding="utf-8") as report_file:
        report_file.write(json.dumps(report, separators=(",", ":")) + "\n")


def fetch_videos(
    records: Sequence[VideoRecord],
    game: str,
    out_dir: str | Path,
    workers: int = 1,
) -> tuple[list[FetchReport], list[VideoRecord]]:
    """Fetch selected videos, returning new reports and already-ok skips."""

    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / REPORT_NAME
    already_ok = _load_ok_video_ids(report_path)
    skipped = [record for record in records if record["video_id"] in already_ok]
    pending = [record for record in records if record["video_id"] not in already_ok]
    reports: list[FetchReport] = []

    if workers == 1:
        for record in pending:
            report = _fetch_one(record, game, destination)
            _append_report(report_path, report)
            reports.append(report)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_fetch_one, record, game, destination): record
                for record in pending
            }
            for future in as_completed(futures):
                report = future.result()
                _append_report(report_path, report)
                reports.append(report)

    return reports, skipped


def _source_counts(
    reports: Iterable[FetchReport], skipped: Iterable[VideoRecord]
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"ok": 0, "failed": 0, "skipped_ok": 0}
    )
    for report in reports:
        counts[report["source"]][report["status"]] += 1
    for record in skipped:
        counts[record["source"]]["skipped_ok"] += 1
    return dict(sorted(counts.items()))


def _print_summary(
    requested: int,
    reports: Sequence[FetchReport],
    skipped: Sequence[VideoRecord],
) -> None:
    fetched_ok = [report for report in reports if report["status"] == "ok"]
    failed = [report for report in reports if report["status"] == "failed"]
    breakdown = json.dumps(
        _source_counts(reports, skipped), sort_keys=True, separators=(",", ":")
    )
    total_hours = sum(report["chunk_hours"] for report in fetched_ok)
    print(
        f"requested={requested} fetched-ok={len(fetched_ok)} "
        f"skipped-ok={len(skipped)} failed={len(failed)} "
        f"per-source={breakdown} total-chunk-hours-fetched={total_hours:.6f}"
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _worker_count(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_WORKERS:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {MAX_WORKERS}, inclusive"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refetch NitroGen source videos with yt-dlp."
    )
    parser.add_argument("--actions-root", required=True, type=Path)
    parser.add_argument("--game", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-videos", type=_positive_int, default=30)
    parser.add_argument("--source-priority", default="youtube,twitch")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=_worker_count, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        priorities = _parse_source_priority(args.source_priority)
    except ValueError as error:
        parser.error(str(error))

    candidates = discover_videos(args.actions_root, args.game)
    sampled = sample_videos(candidates, args.n_videos, priorities, args.seed)

    if args.dry_run:
        for record in sampled:
            print(f"dry-run {json.dumps(record, separators=(',', ':'))}")
        source_counts: dict[str, int] = defaultdict(int)
        for record in sampled:
            source_counts[record["source"]] += 1
        source_summary = json.dumps(
            dict(sorted(source_counts.items())), separators=(",", ":")
        )
        print(
            f"dry-run requested={len(sampled)} "
            f"per-source={source_summary} "
            f"total-chunk-hours={sum(record['chunk_hours'] for record in sampled):.6f}"
        )
        return 0

    reports, skipped = fetch_videos(sampled, args.game, args.out, args.workers)
    _print_summary(len(sampled), reports, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
