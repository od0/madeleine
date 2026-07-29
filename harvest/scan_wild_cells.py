"""Scan every frame of a wild-video HUD using an AI-nominated physical layout.

The output describes physical cell activity only.  It is useful for rejecting
frozen or unstable overlays before semantic mapping, but it cannot approve a
layout, assign gameplay actions, establish compositor offset, or admit data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import cv2
import numpy as np

from harvest.survey_wild_layout import (
    _load_bound_source,
    contact_sheet_command,
    declared_artifact_path,
    exact_extract_command,
    sha256_file,
)
from harvest.worker_wild import _copy_verified


SPEC_VERSION = "madeleine.wild-cell-scan-spec.v1"
REPORT_VERSION = "madeleine.wild-cell-activity-scan.v1"
PUBLICATION_VERSION = "madeleine.wild-cell-activity-publication.v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text())
    if spec.get("format_version") != SPEC_VERSION:
        raise ValueError("unsupported cell scan spec")
    video_id = str(spec.get("video_id", ""))
    if not _SAFE_ID.fullmatch(video_id):
        raise ValueError("cell scan spec has an unsafe video_id")
    for field in (
        "source_sha256",
        "pts_sha256",
        "survey_sha256",
        "survey_contact_sheet_sha256",
    ):
        value = str(spec.get(field, ""))
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{field} must be a lowercase SHA-256")
    frame_size = spec.get("frame_size_wh")
    if (
        not isinstance(frame_size, list)
        or len(frame_size) != 2
        or any(not isinstance(value, int) or value <= 0 for value in frame_size)
    ):
        raise ValueError("frame_size_wh must contain two positive integers")
    scan_range = spec.get("scan_range_s")
    if scan_range is not None:
        if (
            not isinstance(scan_range, list)
            or len(scan_range) != 2
            or any(not isinstance(value, (int, float)) for value in scan_range)
            or not np.all(np.isfinite(scan_range))
            or float(scan_range[0]) < 0
            or float(scan_range[1]) <= float(scan_range[0])
        ):
            raise ValueError("scan_range_s must be a finite [start_s, end_s]")
    cells = spec.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("cell scan spec must contain cells")
    cell_ids = []
    for row in cells:
        if not isinstance(row, dict):
            raise ValueError("every cell must be an object")
        cell_id = str(row.get("cell_id", ""))
        if not _SAFE_ID.fullmatch(cell_id):
            raise ValueError("cell scan spec has an unsafe cell_id")
        cell_ids.append(cell_id)
        rect = row.get("sample_rect_px")
        if (
            not isinstance(rect, list)
            or len(rect) != 4
            or any(not isinstance(value, int) for value in rect)
        ):
            raise ValueError(f"{cell_id}: sample_rect_px must contain four integers")
        x, y, width, height = rect
        if width <= 0 or height <= 0 or x < 0 or y < 0:
            raise ValueError(f"{cell_id}: invalid sample rectangle")
        if x + width > frame_size[0] or y + height > frame_size[1]:
            raise ValueError(f"{cell_id}: sample rectangle leaves the source frame")
        if row.get("pressed_polarity") not in ("high", "low"):
            raise ValueError(f"{cell_id}: pressed_polarity must be high or low")
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("cell scan spec contains duplicate cell_id values")
    if spec.get("human_reviewed") is not False or spec.get("training_admitted") is not False:
        raise ValueError("AI cell scan specs cannot claim review or admission")
    return spec


def score_filter_graph(cells: list[dict[str, Any]]) -> tuple[str, int]:
    split = f"[0:v]format=gray,split={len(cells)}" + "".join(
        f"[s{index}]" for index in range(len(cells))
    )
    crops = []
    widths = []
    max_height = max(int(cell["sample_rect_px"][3]) for cell in cells)
    for index, cell in enumerate(cells):
        x, y, width, cell_height = cell["sample_rect_px"]
        widths.append(width)
        crop = f"[s{index}]crop={width}:{cell_height}:{x}:{y}"
        if cell_height != max_height:
            crop += f",pad={width}:{max_height}:0:0:black"
        crops.append(crop + f"[c{index}]")
    stack = "".join(f"[c{index}]" for index in range(len(cells)))
    graph = ";".join((split, *crops, f"{stack}hstack=inputs={len(cells)}[out]"))
    return graph, sum(widths) * max_height


def ranged_score_filter_graph(
    cells: list[dict[str, Any]], start_index: int, count: int
) -> tuple[str, int]:
    if start_index < 0 or count <= 0:
        raise ValueError("score frame range must be non-negative and non-empty")
    graph, frame_bytes = score_filter_graph(cells)
    selection = (
        f"[0:v]select=between(n\\,{start_index}\\,{start_index + count - 1}),"
    )
    if not graph.startswith("[0:v]"):
        raise AssertionError("unexpected score filter graph source")
    return selection + graph[len("[0:v]"):], frame_bytes


def score_decode_command(
    video: Path,
    cells: list[dict[str, Any]],
    hwaccel: str = "cuda",
    *,
    start_index: int = 0,
    count: int | None = None,
) -> tuple[list[str], int]:
    if hwaccel not in ("none", "cuda"):
        raise ValueError("score hwaccel must be none or cuda")
    graph, frame_bytes = (
        ranged_score_filter_graph(cells, start_index, count)
        if count is not None
        else score_filter_graph(cells)
    )
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-threads", "8",
    ]
    if hwaccel == "cuda":
        command.extend(["-hwaccel", "cuda"])
    command.extend([
        "-i", str(video),
        "-filter_complex", graph, "-map", "[out]", "-pix_fmt", "gray",
        # Rawvideo otherwise inherits ffmpeg's default CFR synchronizer and
        # can duplicate VFR source frames, breaking the persisted PTS binding.
        "-vsync", "0",
    ])
    if count is not None:
        command.extend(["-frames:v", str(count)])
    command.extend(["-f", "rawvideo", "pipe:1"])
    return command, frame_bytes


def transition_stats(states: np.ndarray, duration_s: float) -> dict[str, int | float]:
    changes = int(np.count_nonzero(states[1:] != states[:-1]))
    edges = np.flatnonzero(np.diff(np.r_[False, states, False].astype(np.int8)))
    positive_lengths = (edges[1::2] - edges[::2]).astype(np.int64)
    return {
        "pressed_frames": int(np.count_nonzero(states)),
        "duty": float(states.mean()),
        "transitions": changes,
        "transitions_hz": changes / max(duration_s, 1e-9),
        "positive_runs": int(positive_lengths.size),
        "single_frame_positive_runs": int(np.count_nonzero(positive_lengths == 1)),
    }


def decode_scores(
    video: Path,
    pts: np.ndarray,
    cells: list[dict[str, Any]],
    destination: Path,
    *,
    hwaccel: str = "cuda",
    start_index: int = 0,
) -> np.memmap:
    command, frame_bytes = score_decode_command(
        video, cells, hwaccel, start_index=start_index, count=int(pts.size)
    )
    widths = [int(cell["sample_rect_px"][2]) for cell in cells]
    heights = [int(cell["sample_rect_px"][3]) for cell in cells]
    max_height = max(heights)
    scores = np.memmap(
        destination,
        dtype=np.float32,
        mode="w+",
        shape=(int(pts.size), len(cells)),
    )
    carry = bytearray()
    row_index = 0
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        assert process.stdout is not None
        try:
            while True:
                block = process.stdout.read(frame_bytes * 4096)
                if not block:
                    break
                carry.extend(block)
                complete = len(carry) // frame_bytes
                if complete == 0:
                    continue
                if row_index + complete > pts.size:
                    raise ValueError(
                        "ffmpeg emitted more frames than the persisted PTS vector"
                    )
                raw = np.frombuffer(
                    carry, dtype=np.uint8, count=complete * frame_bytes
                ).copy()
                raw = raw.reshape(complete, max_height, sum(widths))
                offset = 0
                for cell_index, (width, cell_height) in enumerate(
                    zip(widths, heights, strict=True)
                ):
                    scores[row_index:row_index + complete, cell_index] = raw[
                        :, :cell_height, offset:offset + width
                    ].mean(axis=(1, 2))
                    offset += width
                row_index += complete
                del carry[:complete * frame_bytes]
            return_code = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        errors.seek(0)
        stderr = errors.read().decode("utf-8", errors="replace")
    scores.flush()
    if return_code:
        raise RuntimeError(f"ffmpeg exited {return_code}: {stderr[-2000:]}")
    if carry:
        raise ValueError("ffmpeg left a partial raw score frame")
    if row_index != pts.size:
        raise ValueError(f"decoded {row_index} frames but PTS has {pts.size}")
    return scores


def validate_scan_inputs(
    source_dir: Path,
    spec_path: Path,
    survey_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, Path]:
    """Validate the completed raw -> survey -> AI spec provenance chain."""

    spec = load_spec(spec_path)
    video_id = spec["video_id"]
    if source_dir.name != video_id or survey_dir.name != video_id:
        raise ValueError("source/survey directory does not match scan spec video_id")
    fetch, pts, video = _load_bound_source(source_dir)
    if fetch.get("sha256") != spec["source_sha256"]:
        raise ValueError("scan spec source hash differs from completed source")
    if sha256_file(source_dir / "frame_pts.npy") != spec["pts_sha256"]:
        raise ValueError("scan spec PTS hash differs from completed source")
    if fetch.get("media", {}).get("resolution_wh") != spec["frame_size_wh"]:
        raise ValueError("source dimensions differ from cell scan spec")

    survey_path = survey_dir / "survey.json"
    completion_path = survey_dir / "survey_complete.json"
    if not survey_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError("completed survey manifest/marker is missing")
    survey = json.loads(survey_path.read_text())
    completion = json.loads(completion_path.read_text())
    survey_hash = sha256_file(survey_path)
    if survey_hash != spec["survey_sha256"]:
        raise ValueError("cell scan spec does not bind the actual survey")
    if (
        survey.get("video_id") != video_id
        or survey.get("source", {}).get("sha256") != spec["source_sha256"]
        or survey.get("pts", {}).get("sha256") != spec["pts_sha256"]
    ):
        raise ValueError("survey does not bind the completed source/PTS")
    if (
        survey.get("human_reviewed") is not False
        or survey.get("training_admitted") is not False
        or completion.get("human_reviewed") is not False
        or completion.get("training_admitted") is not False
    ):
        raise ValueError("AI survey chain cannot claim review or admission")
    contact_row = survey.get("contact_sheet", {})
    contact_path = declared_artifact_path(survey_dir, contact_row)
    if sha256_file(contact_path) != spec["survey_contact_sheet_sha256"]:
        raise ValueError("cell scan spec does not bind the survey contact sheet")
    if (
        completion.get("video_id") != video_id
        or completion.get("source_sha256") != spec["source_sha256"]
        or completion.get("survey_sha256") != survey_hash
    ):
        raise ValueError("survey completion marker has incompatible bindings")
    completion_objects = {
        str(row.get("remote_path", "")).rsplit("/", 1)[-1]: row
        for row in completion.get("objects", [])
    }
    for name, expected_hash in (
        (survey_path.name, survey_hash),
        (contact_path.name, spec["survey_contact_sheet_sha256"]),
    ):
        if completion_objects.get(name, {}).get("sha256") != expected_hash:
            raise ValueError(f"survey completion marker does not bind {name}")
    return spec, fetch, pts, video


def classify_cells(
    scores: np.ndarray,
    pts: np.ndarray,
    cells: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, list[str]]]:
    duration_s = float(pts[-1] - pts[0]) if pts.size > 1 else 0.0
    diagnostics = []
    evidence_indices: dict[int, list[str]] = {}
    for index, cell in enumerate(cells):
        values = np.asarray(scores[:, index])
        low, high = float(values.min()), float(values.max())
        scaled = np.clip((values - low) / max(high - low, 1e-9) * 255, 0, 255).astype(
            np.uint8
        )
        level, _ = cv2.threshold(
            scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        provisional = low + float(level) / 255.0 * (high - low)
        low_cluster = values[values <= provisional]
        high_cluster = values[values > provisional]
        if low_cluster.size and high_cluster.size:
            low_median = float(np.median(low_cluster))
            high_median = float(np.median(high_cluster))
            threshold = (low_median + high_median) / 2.0
        else:
            low_median = high_median = threshold = float(np.median(values))
        high_state = values >= threshold
        states = high_state if cell["pressed_polarity"] == "high" else ~high_state
        minority = int(min(np.count_nonzero(states), np.count_nonzero(~states)))
        separation = high_median - low_median
        changing = minority >= 30 and separation >= 5.0
        if changing:
            pressed = np.flatnonzero(states)
            released = np.flatnonzero(~states)
            pressed_target = high_median if cell["pressed_polarity"] == "high" else low_median
            released_target = low_median if cell["pressed_polarity"] == "high" else high_median
            pressed_index = int(pressed[np.argmin(np.abs(values[pressed] - pressed_target))])
            released_index = int(released[np.argmin(np.abs(values[released] - released_target))])
            evidence_indices.setdefault(pressed_index, []).append(
                f"{cell['cell_id']}:pressed"
            )
            evidence_indices.setdefault(released_index, []).append(
                f"{cell['cell_id']}:released"
            )
        diagnostics.append({
            "cell_id": cell["cell_id"],
            "physical_label": cell.get("physical_label"),
            "sample_rect_px": cell["sample_rect_px"],
            "pressed_polarity": cell["pressed_polarity"],
            "threshold": threshold,
            "low_median": low_median,
            "high_median": high_median,
            "cluster_separation_luma": separation,
            "minority_frames": minority,
            "changing": changing,
            **transition_stats(states, duration_s),
        })
    return diagnostics, evidence_indices


def build_scan(
    source_dir: Path,
    spec_path: Path,
    survey_dir: Path,
    out_root: Path,
    *,
    score_hwaccel: str = "cuda",
    evidence_hwaccel: str = "none",
) -> Path:
    spec, fetch, pts, video = validate_scan_inputs(
        source_dir, spec_path, survey_dir
    )
    video_id = spec["video_id"]
    pts_path = source_dir / "frame_pts.npy"

    destination = out_root / video_id
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"cell scan output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    relative_pts = pts - pts[0]
    if spec.get("scan_range_s") is None:
        selected_indices = np.arange(pts.size, dtype=np.int64)
    else:
        scan_start, scan_end = (float(value) for value in spec["scan_range_s"])
        selected_indices = np.flatnonzero(
            (relative_pts >= scan_start) & (relative_pts < scan_end)
        )
        if selected_indices.size == 0:
            raise ValueError("scan_range_s contains no source frames")
        if not np.array_equal(
            selected_indices,
            np.arange(selected_indices[0], selected_indices[-1] + 1),
        ):
            raise ValueError("scan_range_s did not select a contiguous frame run")
    selected_pts = pts[selected_indices]
    source_start = int(selected_indices[0])
    source_end = int(selected_indices[-1]) + 1
    scores_path = destination / "cell_scores.f32"
    scores = decode_scores(
        video,
        selected_pts,
        spec["cells"],
        scores_path,
        hwaccel=score_hwaccel,
        start_index=source_start,
    )
    diagnostics, relative_evidence_indices = classify_cells(
        scores, selected_pts, spec["cells"]
    )
    evidence_indices = {
        source_start + index: roles
        for index, roles in relative_evidence_indices.items()
    }

    evidence_dir = destination / "evidence"
    evidence_dir.mkdir()
    ordered_indices = sorted(evidence_indices)
    evidence_rows = []
    if ordered_indices:
        temporary_pattern = evidence_dir / "temporary-%03d.png"
        subprocess.run(
            exact_extract_command(
                video, temporary_pattern, ordered_indices, evidence_hwaccel
            ),
            check=True,
        )
        temporary = sorted(evidence_dir.glob("temporary-*.png"))
        if len(temporary) != len(ordered_indices):
            raise ValueError("exact evidence extraction returned the wrong frame count")
        contact_path = destination / "evidence-contact-sheet.png"
        subprocess.run(
            contact_sheet_command(
                temporary_pattern,
                contact_path,
                len(ordered_indices),
                columns=min(4, len(ordered_indices)),
            ),
            check=True,
        )
        for order, (path, frame_index) in enumerate(
            zip(temporary, ordered_indices, strict=True)
        ):
            final = evidence_dir / f"evidence-{order:02d}-frame-{frame_index:09d}.png"
            path.rename(final)
            evidence_rows.append({
                "sample_order": order,
                "exact_frame_index": frame_index,
                "exact_pts_s": float(pts[frame_index]),
                "roles": evidence_indices[frame_index],
                "path": str(final.relative_to(destination)),
                "size_bytes": final.stat().st_size,
                "sha256": sha256_file(final),
            })
        contact_row: dict[str, Any] | None = {
            "path": contact_path.name,
            "size_bytes": contact_path.stat().st_size,
            "sha256": sha256_file(contact_path),
        }
    else:
        contact_row = None

    report = {
        "format_version": REPORT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video_id": video_id,
        "purpose": "AI-only physical-cell activity nomination before semantic mapping",
        "source": {
            "path": video.name,
            "sha256": spec["source_sha256"],
            "frames": int(pts.size),
            "duration_s": float(pts[-1] - pts[0]),
            "source_frame_range": [source_start, source_end],
            "scanned_frames": int(selected_indices.size),
            "scan_range_s": [
                float(relative_pts[source_start]),
                float(relative_pts[source_end - 1]),
            ],
        },
        "pts": {
            "path": pts_path.name,
            "sha256": spec["pts_sha256"],
            "first_s": float(selected_pts[0]),
            "last_s": float(selected_pts[-1]),
        },
        "spec": {
            "path": spec_path.name,
            "size_bytes": spec_path.stat().st_size,
            "sha256": sha256_file(spec_path),
            "reviewer_kind": spec.get("reviewer_kind"),
            "survey_sha256": spec.get("survey_sha256"),
        },
        "survey": {
            "path": "survey.json",
            "sha256": spec["survey_sha256"],
            "contact_sheet_sha256": spec["survey_contact_sheet_sha256"],
        },
        "cells": diagnostics,
        "changing_cell_ids": [row["cell_id"] for row in diagnostics if row["changing"]],
        "evidence": evidence_rows,
        "evidence_contact_sheet": contact_row,
        "evidence_extract_hwaccel": evidence_hwaccel,
        "score_extract_hwaccel": score_hwaccel,
        "scores": {
            "path": scores_path.name,
            "size_bytes": scores_path.stat().st_size,
            "sha256": sha256_file(scores_path),
            "dtype": "float32",
            "shape": [int(selected_indices.size), len(spec["cells"])],
        },
        "limitations": [
            "changing physical cells do not establish semantic action mapping",
            "full-source duration includes menus, cutscenes, and inactive HUD intervals",
            "compositor offset is unmeasured",
        ],
        "human_reviewed": False,
        "training_admitted": False,
    }
    report_path = destination / "cell_activity_scan.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    shutil_spec = destination / spec_path.name
    shutil_spec.write_bytes(spec_path.read_bytes())
    return report_path


def publish_scan(report_path: Path, remote_root: str) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    video_id = str(report.get("video_id", ""))
    if not _SAFE_ID.fullmatch(video_id) or ":" not in remote_root:
        raise ValueError("unsafe publication target")
    if (
        report.get("human_reviewed") is not False
        or report.get("training_admitted") is not False
    ):
        raise ValueError("AI scan publication cannot claim review or admission")
    directory = report_path.parent
    remote_dir = f"{remote_root.rstrip('/')}/{video_id}"
    declared_rows = [report["scores"], report["spec"], *report["evidence"]]
    if report["evidence_contact_sheet"] is not None:
        declared_rows.append(report["evidence_contact_sheet"])
    declared_paths = [
        declared_artifact_path(directory, row) for row in declared_rows
    ]
    if len(declared_paths) != len(set(declared_paths)):
        raise ValueError("scan report declares the same artifact more than once")
    paths = [report_path.resolve(), *declared_paths]
    verified = []
    for index, path in enumerate(paths):
        relative = path.relative_to(directory.resolve()).as_posix()
        copied = _copy_verified(path, f"{remote_dir}/{relative}")
        if index:
            declared = declared_rows[index - 1]
            size_matches = (
                "size_bytes" not in declared
                or int(copied.get("size_bytes", -1)) == int(declared["size_bytes"])
            )
            if copied.get("sha256") != declared.get("sha256") or not size_matches:
                raise ValueError(
                    f"published bytes differ from scan report: {relative}"
                )
        verified.append(copied)
    completion = {
        "format_version": PUBLICATION_VERSION,
        "video_id": video_id,
        "source_sha256": report["source"]["sha256"],
        "report_sha256": sha256_file(report_path),
        "remote_dir": remote_dir,
        "verification": "every object SHA-256 hashed through rclone cat",
        "objects": verified,
        "total_bytes": sum(int(row["size_bytes"]) for row in verified),
        "human_reviewed": False,
        "training_admitted": False,
    }
    completion_path = directory / "cell_activity_complete.json"
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    _copy_verified(completion_path, f"{remote_dir}/{completion_path.name}")
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--survey-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--remote-root")
    parser.add_argument(
        "--score-hwaccel", default="cuda", choices=("none", "cuda")
    )
    parser.add_argument(
        "--evidence-hwaccel", default="none", choices=("none", "cuda")
    )
    args = parser.parse_args()
    spec = load_spec(args.spec)
    destination = args.out_root / spec["video_id"]
    existing_report = destination / "cell_activity_scan.json"
    if existing_report.is_file():
        validate_scan_inputs(args.source_dir, args.spec, args.survey_dir)
        existing = json.loads(existing_report.read_text())
        if (
            existing.get("video_id") != spec["video_id"]
            or existing.get("source", {}).get("sha256") != spec["source_sha256"]
            or existing.get("spec", {}).get("sha256") != sha256_file(args.spec)
        ):
            raise ValueError("existing cell scan does not bind the requested inputs")
        report = existing_report
    else:
        if destination.exists() and any(destination.iterdir()):
            failed_root = args.out_root / ".failed"
            failed_root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            destination.rename(failed_root / f"{spec['video_id']}-{stamp}")
        report = build_scan(
            args.source_dir,
            args.spec,
            args.survey_dir,
            args.out_root,
            score_hwaccel=args.score_hwaccel,
            evidence_hwaccel=args.evidence_hwaccel,
        )
    result: dict[str, Any] = {"report": str(report)}
    if args.remote_root:
        result["publication"] = publish_scan(report, args.remote_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
