"""Polite, idempotent full-video fetch and timestamp audit for wild footage.

This module deliberately owns no fleet logic and no credentials.  A worker is
given one candidate at a time.  YouTube extraction explicitly uses Deno and a
single fragment; host-level concurrency and pacing stay with the orchestrator.

The resulting report separates the speedrun's nominal duration from the media
timeline.  A missing run-start offset is recorded as unresolved when the media
contains appreciable lead-in/out footage; downstream builders refuse to guess.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np


REPORT_VERSION = "madeleine.wild-fetch.v2"
PTS_SIDECAR_VERSION = "madeleine.wild-pts.v1"
_HMS_PART = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>[hms])")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_timestamp(value: str | int | float | None) -> float | None:
    """Parse seconds or compact values such as ``1h02m03.5s``."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if float(value) >= 0 else None
    text = value.strip().lower()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        seconds = 0.0
        cursor = 0
        for match in _HMS_PART.finditer(text):
            if match.start() != cursor:
                return None
            amount = float(match.group("value"))
            seconds += amount * {"h": 3600.0, "m": 60.0, "s": 1.0}[match.group("unit")]
            cursor = match.end()
        if cursor != len(text) or cursor == 0:
            return None
    return seconds if seconds >= 0 else None


def url_start_time(url: str) -> float | None:
    """Read an explicit YouTube/Twitch start offset without inventing one."""

    parsed = urlparse(url)
    values = parse_qs(parsed.query)
    for key in ("t", "start", "time_continue"):
        if key in values:
            parsed_time = parse_timestamp(values[key][0])
            if parsed_time is not None:
                return parsed_time
    fragment = parsed.fragment
    if fragment.startswith("t="):
        return parse_timestamp(fragment[2:])
    return None


def _rate(value: str | None) -> float | None:
    if not value or value in ("0/0", "N/A"):
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


@dataclass(frozen=True)
class FetchPolicy:
    yt_dlp_path: str = "yt-dlp"
    deno_path: str = "deno"
    max_height: int = 720
    min_preferred_fps: int = 50
    sleep_requests_s: float = 1.0
    sleep_min_s: float = 2.0
    sleep_max_s: float = 5.0


def build_fetch_command(
    url: str, output_template: Path, policy: FetchPolicy = FetchPolicy()
) -> list[str]:
    """Build the argv used by a single low-rate worker.

    There is intentionally no cookie/browser import and no parallel playlist
    behavior.  Authentication, if ever required, must be an explicit separate
    operational decision.
    """

    selector = (
        f"bv*[height<={policy.max_height}][fps>={policy.min_preferred_fps}]/"
        f"bv*[height<={policy.max_height}]/b[height<={policy.max_height}]"
        # Some Twitch VODs expose only a slightly taller combined source
        # rendition (for example 1364x768) and no <=720p variant. Preserve the
        # bandwidth preference above, but do not reject otherwise valid media
        # solely because the provider omitted that exact ladder rung.
        "/b"
    )
    return [
        policy.yt_dlp_path,
        "--no-playlist",
        "--js-runtimes", f"deno:{policy.deno_path}",
        "--concurrent-fragments", "1",
        "--sleep-requests", str(policy.sleep_requests_s),
        "--sleep-interval", str(policy.sleep_min_s),
        "--max-sleep-interval", str(policy.sleep_max_s),
        "--retries", "5",
        "--fragment-retries", "5",
        # Keep captured failure diagnostics bounded; fleet workers need the
        # final provider error, not a multi-hour in-memory progress stream.
        "--no-progress",
        "--write-info-json",
        "--no-overwrites",
        "-f", selector,
        "-o", str(output_template),
        url,
    ]


def frame_pts(path: str | Path) -> np.ndarray:
    """Return decoded presentation timestamps from ffprobe in frame order."""

    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    values = []
    for line in result.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    if not values:
        raise ValueError(f"{path}: ffprobe returned no video frame timestamps")
    return np.asarray(values, dtype=np.float64)


def summarize_pts(pts: np.ndarray) -> dict[str, Any]:
    if pts.ndim != 1 or pts.size == 0 or not np.all(np.isfinite(pts)):
        raise ValueError("PTS must be a non-empty finite vector")
    delta = np.diff(pts)
    positive = delta[delta > 0]
    if positive.size == 0:
        raise ValueError("PTS has no positive frame intervals")
    median = float(np.median(positive))
    p01, p99 = (float(v) for v in np.percentile(positive, [1, 99]))
    gap_gate = max(0.100, 2.5 * median)
    # Millisecond-quantized 60 Hz timelines (notably Twitch VODs) alternate
    # 0.016 and 0.017 second intervals.  ``1 / median`` calls that 58.82 Hz
    # even though its actual cadence is 60 Hz.  Average only ordinary positive
    # intervals so real timestamp gaps do not depress the cadence estimate.
    cadence_intervals = positive[positive <= gap_gate]
    if cadence_intervals.size == 0:
        raise ValueError("PTS has no intervals inside the cadence gate")
    mean_cadence = float(np.mean(cadence_intervals))
    span = float(pts[-1] - pts[0])
    return {
        "frames": int(pts.size),
        "first_s": float(pts[0]),
        "last_s": float(pts[-1]),
        "median_dt_s": median,
        "mean_cadence_dt_s": mean_cadence,
        "p01_dt_s": p01,
        "p99_dt_s": p99,
        "effective_fps": 1.0 / mean_cadence,
        "span_fps": (pts.size - 1) / span if pts.size > 1 and span > 0 else None,
        "nonmonotonic_intervals": int(np.count_nonzero(delta <= 0)),
        "large_gap_intervals": int(np.count_nonzero(delta > gap_gate)),
        "largest_gap_s": float(delta.max(initial=0.0)),
        "vfr_ratio_p99_p01": p99 / max(p01, 1e-12),
    }


def probe_media(path: str | Path, scan_pts: bool = True) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=index,codec_name,width,height,r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_frames",
        "-show_entries", "format=duration,size,format_name",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    raw = json.loads(result.stdout)
    streams = raw.get("streams") or []
    if len(streams) != 1:
        raise ValueError(f"{path}: expected one selected video stream")
    stream = streams[0]
    fmt = raw.get("format") or {}
    report: dict[str, Any] = {
        "codec": stream.get("codec_name"),
        "resolution_wh": [int(stream["width"]), int(stream["height"])],
        "r_frame_rate": _rate(stream.get("r_frame_rate")),
        "avg_frame_rate": _rate(stream.get("avg_frame_rate")),
        "time_base": stream.get("time_base"),
        "start_time_s": float(stream.get("start_time") or 0.0),
        "duration_s": float(stream.get("duration") or fmt.get("duration") or 0.0),
        "declared_frames": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
        "container": fmt.get("format_name"),
        "size_bytes": int(fmt.get("size") or Path(path).stat().st_size),
    }
    if scan_pts:
        report["pts"] = summarize_pts(frame_pts(path))
    return report


def ensure_pts_sidecar(
    directory: str | Path,
    video: str | Path,
    source_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create or verify the exact presentation-timestamp evidence once."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    video_path = Path(video)
    vector_path = destination / "frame_pts.npy"
    manifest_path = destination / "frame_pts.json"
    if vector_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        pts = np.load(vector_path, allow_pickle=False)
        if manifest.get("format_version") != PTS_SIDECAR_VERSION:
            raise ValueError("unsupported frame PTS sidecar version")
        if manifest.get("source_sha256") != source_sha256:
            raise ValueError("frame PTS sidecar belongs to a different source video")
        if manifest.get("sha256") != sha256_file(vector_path):
            raise ValueError("frame PTS sidecar hash mismatch")
        if pts.ndim != 1 or int(manifest.get("frames", -1)) != pts.size:
            raise ValueError("frame PTS sidecar shape/count mismatch")
        summarize_pts(pts)  # finite/monotonic evidence validation
        return pts.astype(np.float64, copy=False), manifest
    if vector_path.exists() or manifest_path.exists():
        raise ValueError("incomplete frame PTS sidecar; refusing to overwrite evidence")

    pts = frame_pts(video_path)
    np.save(vector_path, pts, allow_pickle=False)
    manifest = {
        "format_version": PTS_SIDECAR_VERSION,
        "source_file": video_path.name,
        "source_sha256": source_sha256,
        "path": vector_path.name,
        "sha256": sha256_file(vector_path),
        "frames": int(pts.size),
        "summary": summarize_pts(pts),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return pts, manifest


def load_pts_evidence(
    fetch_report_path: str | Path,
    fetch: dict[str, Any],
    video: str | Path,
    evidence_dir: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load verified PTS evidence, regenerating legacy evidence if absent.

    ``evidence_dir`` supports rehydrated v1 canaries whose immutable raw prefix
    has a separate derived-evidence publication.
    """

    directory = Path(evidence_dir) if evidence_dir is not None else Path(fetch_report_path).parent
    source_hash = str(fetch["sha256"])
    pts, manifest = ensure_pts_sidecar(directory, video, source_hash)
    declared = fetch.get("pts_sidecar")
    if declared is not None:
        if declared.get("sha256") != manifest.get("sha256"):
            raise ValueError("PTS evidence hash differs from fetch report")
        if int(declared.get("frames", -1)) != pts.size:
            raise ValueError("PTS evidence length differs from fetch report")
    reported_frames = ((fetch.get("media") or {}).get("pts") or {}).get("frames")
    if reported_frames is not None and int(reported_frames) != pts.size:
        raise ValueError("PTS evidence length differs from media audit")
    return pts, manifest


def resolve_run_window(
    url: str,
    nominal_duration_s: float,
    media_duration_s: float,
    explicit_start_s: float | None = None,
    explicit_end_s: float | None = None,
    tolerance_s: float = 15.0,
) -> dict[str, Any]:
    """Resolve independent wall-clock boundaries without using loadless time.

    Leaderboard duration excludes loads and therefore cannot determine the end
    of a video interval.  It remains provenance metadata only.  The sole safe
    inference is a duration match, where the entire media timeline is the run.
    """

    duration_match = abs(media_duration_s - nominal_duration_s) <= tolerance_s
    url_start = url_start_time(url)
    if explicit_start_s is not None:
        start, start_source = float(explicit_start_s), "explicit"
    elif url_start is not None:
        start, start_source = float(url_start), "url_timestamp"
    elif duration_match:
        start, start_source = 0.0, "duration_match"
    else:
        start, start_source = None, "unresolved"

    if explicit_end_s is not None:
        end, end_source = float(explicit_end_s), "explicit"
    elif duration_match:
        end, end_source = float(media_duration_s), "duration_match"
    else:
        end, end_source = None, "unresolved"

    errors = []
    if start is not None and not 0.0 <= start < media_duration_s:
        errors.append(f"start {start:.3f}s lies outside media")
    if end is not None and not 0.0 < end <= media_duration_s + tolerance_s:
        errors.append(f"end {end:.3f}s lies outside media")
    if start is not None and end is not None and end <= start:
        errors.append("end must be greater than start")
    if errors:
        raise ValueError("; ".join(errors))

    unresolved = []
    if start is None:
        unresolved.append("start boundary needs reviewed wall-clock evidence")
    if end is None:
        unresolved.append(
            "end boundary needs reviewed wall-clock evidence; leaderboard duration is loadless"
        )
    return {
        "resolved": start is not None and end is not None,
        "start_resolved": start is not None,
        "end_resolved": end is not None,
        "start_s": start,
        "end_s": end,
        "nominal_loadless_duration_s": float(nominal_duration_s),
        "start_source": start_source,
        "end_source": end_source,
        "reason": "; ".join(unresolved) or None,
    }


def _source_file(directory: Path, video_id: str) -> Path | None:
    candidates = [
        path for path in directory.glob(f"{video_id}.*")
        if not path.name.endswith((".info.json", ".part", ".ytdl", ".fetch.json"))
        and path.is_file()
    ]
    return sorted(candidates)[0] if len(candidates) == 1 else None


def fetch_candidate(
    candidate: dict[str, Any],
    out_dir: str | Path,
    *,
    policy: FetchPolicy = FetchPolicy(),
    explicit_start_s: float | None = None,
    explicit_end_s: float | None = None,
    run_download: bool = True,
) -> dict[str, Any]:
    video_id = str(candidate["video_id"])
    url = str(candidate["url"])
    nominal_duration = float(candidate["duration_s"])
    destination = Path(out_dir) / video_id
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "fetch.json"
    if report_path.exists():
        old = json.loads(report_path.read_text())
        source = destination / old.get("source_file", "")
        if source.is_file():
            actual_hash = sha256_file(source)
            if old.get("sha256") == actual_hash:
                # Backward-compatible canaries keep their immutable v1
                # fetch.json.  Add separately hashed PTS evidence beside it;
                # reviewed boundaries also live in a separate artifact.
                ensure_pts_sidecar(destination, source, actual_hash)
                return old
            raise ValueError(
                f"{video_id}: cached source hash differs from fetch.json; "
                "refusing to bless or overwrite it"
            )

    if run_download:
        command = build_fetch_command(url, destination / f"{video_id}.%(ext)s", policy)
        # Preserve the source's bounded diagnostic on failure so the fleet
        # controller can distinguish a dead video from an IP-wide bot block.
        # Download progress is operational noise and does not belong in the
        # durable worker log; successful output is discarded with the result.
        subprocess.run(command, check=True, capture_output=True, text=True)
    source = _source_file(destination, video_id)
    if source is None:
        raise FileNotFoundError(f"{video_id}: expected exactly one downloaded media file")
    source_hash = sha256_file(source)
    pts, pts_sidecar = ensure_pts_sidecar(destination, source, source_hash)
    media = probe_media(source, scan_pts=False)
    media["pts"] = summarize_pts(pts)
    run_window = resolve_run_window(
        url, nominal_duration, media["duration_s"],
        explicit_start_s=explicit_start_s, explicit_end_s=explicit_end_s,
    )
    report = {
        "format_version": REPORT_VERSION,
        "video_id": video_id,
        "source": candidate.get("source"),
        "origin_url": url,
        "source_file": source.name,
        "sha256": source_hash,
        "pts_sidecar": {
            "path": pts_sidecar["path"],
            "manifest": "frame_pts.json",
            "sha256": pts_sidecar["sha256"],
            "frames": pts_sidecar["frames"],
        },
        "candidate": {
            key: candidate.get(key)
            for key in ("run_id", "category", "place", "duration_s", "platform")
        },
        "policy": asdict(policy),
        "media": media,
        "run_window": run_window,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_candidate(path: str | Path) -> dict[str, Any]:
    """Load exactly one JSON object, including normal multi-line JSON.

    A one-row JSONL file is also a JSON object. Multiple JSONL rows are
    intentionally rejected instead of silently running only the first one.
    """

    candidate_path = Path(path)
    try:
        candidate = json.loads(candidate_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "candidate must contain exactly one valid JSON object"
        ) from exc
    if not isinstance(candidate, dict):
        raise ValueError("candidate must contain exactly one JSON object")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True,
                        help="one candidate JSON object or one-row JSONL")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-s", type=float)
    parser.add_argument("--end-s", type=float)
    parser.add_argument("--no-download", action="store_true",
                        help="audit a previously downloaded source file")
    args = parser.parse_args()
    candidate = load_candidate(args.candidate)
    report = fetch_candidate(
        candidate, args.out, explicit_start_s=args.start_s,
        explicit_end_s=args.end_s,
        run_download=not args.no_download,
    )
    print(json.dumps({
        "video_id": report["video_id"],
        "source_file": report["source_file"],
        "run_window": report["run_window"],
        "pts": report["media"]["pts"],
    }, indent=2))


if __name__ == "__main__":
    main()
