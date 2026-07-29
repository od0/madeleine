"""Bind a proven HUD-family template to a new, source-bound provisional video.

This tool deliberately does not approve data.  It turns an explicit AI-only
layout-family assessment into three immutable inputs for the existing full
cell scan and decoder: a target layout, a physical-cell scan spec, and an
outer source boundary.  Full-scan mechanical QC must still pass before shard
building, and every generated artifact remains unreviewed and unadmitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from harvest.fetch_wild import sha256_file
from harvest.scan_wild_cells import SPEC_VERSION
from harvest.survey_wild_layout import _load_bound_source, declared_artifact_path
from harvest.wild_boundaries import WildBoundaries
from harvest.wild_layout import WildLayout, rect_to_pixels


EVIDENCE_VERSION = "madeleine.wild-layout-family-transfer.v1"
DEFAULT_ASSESSMENT_NOTE = (
    "AI visual audit found an exact source-bound printed-semantic HUD layout; "
    "the selected target interval remains subject to a complete cell scan."
)


def _write_resumable(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise FileExistsError(f"refusing to overwrite changed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def _write_bytes_resumable(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite changed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _validate_survey(
    source_dir: Path, survey_dir: Path
) -> tuple[dict[str, Any], np.ndarray, Path, dict[str, Any], Path, str]:
    fetch, pts, video = _load_bound_source(source_dir)
    survey_path = survey_dir / "survey.json"
    completion_path = survey_dir / "survey_complete.json"
    if not survey_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError("completed target survey is required")
    survey = json.loads(survey_path.read_text())
    completion = json.loads(completion_path.read_text())
    survey_sha = sha256_file(survey_path)
    pts_sha = sha256_file(source_dir / "frame_pts.npy")
    source_sha = str(fetch.get("sha256", ""))
    video_id = str(fetch.get("video_id", ""))
    if source_dir.name != video_id or survey_dir.name != video_id:
        raise ValueError("source/survey directory name must equal target video_id")
    if (
        survey.get("video_id") != video_id
        or survey.get("source", {}).get("sha256") != source_sha
        or survey.get("pts", {}).get("sha256") != pts_sha
        or survey.get("human_reviewed") is not False
        or survey.get("training_admitted") is not False
    ):
        raise ValueError("survey is not an unreviewed binding to target source/PTS")
    if (
        completion.get("video_id") != video_id
        or completion.get("source_sha256") != source_sha
        or completion.get("survey_sha256") != survey_sha
        or completion.get("human_reviewed") is not False
        or completion.get("training_admitted") is not False
    ):
        raise ValueError("survey completion marker does not bind target survey")
    contact = declared_artifact_path(survey_dir, survey["contact_sheet"])
    if sha256_file(contact) != survey["contact_sheet"]["sha256"]:
        raise ValueError("survey contact sheet hash mismatch")
    return fetch, pts, video, survey, contact, survey_sha


def _sample_diagnostics(
    survey_dir: Path,
    survey: dict[str, Any],
    layout: WildLayout,
    expected_wh: tuple[int, int],
    *,
    bounded: bool = False,
) -> list[dict[str, Any]]:
    minimum_frames = 4 if bounded else 8
    frames = survey.get("frames")
    if not isinstance(frames, list) or len(frames) < minimum_frames:
        raise ValueError(
            "layout-family transfer requires at least "
            f"{minimum_frames} exact survey frames"
        )
    values: list[list[float]] = [[] for _ in layout.cells]
    for row in frames:
        path = declared_artifact_path(survey_dir, row)
        if sha256_file(path) != row.get("sha256"):
            raise ValueError(f"survey frame hash mismatch: {path.name}")
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or (image.shape[1], image.shape[0]) != expected_wh:
            raise ValueError(f"survey frame dimensions differ from source: {path.name}")
        for index, cell in enumerate(layout.cells):
            x0, y0, x1, y1 = rect_to_pixels(
                cell.sample_rect, image.shape[1], image.shape[0]
            )
            values[index].append(float(image[y0:y1, x0:x1].mean()))
    return [
        {
            "cell_id": cell.cell_id,
            "action": cell.action,
            "survey_sample_min_luma": min(cell_values),
            "survey_sample_max_luma": max(cell_values),
            "survey_sample_range_luma": max(cell_values) - min(cell_values),
        }
        for cell, cell_values in zip(layout.cells, values, strict=True)
    ]


def prepare_transfer(
    *,
    source_dir: Path,
    survey_dir: Path,
    reference_layout_path: Path,
    reference_video_id: str,
    assessment_note: str = DEFAULT_ASSESSMENT_NOTE,
    out_dir: Path,
    scan_start_s: float | None = None,
    scan_end_s: float | None = None,
) -> dict[str, Path]:
    fetch, pts, _video, survey, contact, survey_sha = _validate_survey(
        source_dir, survey_dir
    )
    target_id = str(fetch["video_id"])
    reference = WildLayout.load(reference_layout_path)
    if reference.video_id != reference_video_id:
        raise ValueError("reference layout video_id does not match explicit reference")
    if not assessment_note.strip():
        raise ValueError("an explicit AI layout-family assessment note is required")
    resolution = fetch.get("media", {}).get("resolution_wh")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(not isinstance(value, int) or value <= 0 for value in resolution)
    ):
        raise ValueError("fetch report lacks source resolution")
    width, height = int(resolution[0]), int(resolution[1])
    # Some capture encoders round a 16:9 raster by one column (for example
    # 854x480).  Permit at most two source pixels of horizontal rounding while
    # still rejecting genuinely different aspect/layout families.
    if abs(width - height * 16 / 9) > 2.0:
        raise ValueError("template family transfer currently requires a 16:9 source")
    source_duration_s = float(pts[-1] - pts[0])
    bounded = scan_start_s is not None or scan_end_s is not None
    range_start_s = 0.0 if scan_start_s is None else float(scan_start_s)
    range_end_s = source_duration_s if scan_end_s is None else float(scan_end_s)
    if (
        not np.isfinite(range_start_s)
        or not np.isfinite(range_end_s)
        or range_start_s < 0.0
        or range_end_s > source_duration_s
        or range_end_s <= range_start_s
    ):
        raise ValueError("explicit scan range must lie inside the source timeline")
    evidence_survey = survey
    if bounded:
        first_pts = float(pts[0])
        evidence_survey = {
            **survey,
            "frames": [
                row for row in survey["frames"]
                if range_start_s
                <= float(row["exact_pts_s"]) - first_pts
                < range_end_s
            ],
        }
    diagnostics = _sample_diagnostics(
        survey_dir,
        evidence_survey,
        reference,
        (width, height),
        # Full-source family claims keep the original eight-frame floor.
        # A deliberately bounded interval may use four sparse source-bound
        # frames only because every frame in that interval must subsequently
        # pass the exhaustive cell scan and decode QC.
        bounded=bounded,
    )
    source_sha = str(fetch["sha256"])
    pts_sha = sha256_file(source_dir / "frame_pts.npy")
    contact_sha = sha256_file(contact)
    reference_sha = sha256_file(reference_layout_path)
    reference_copy_path = out_dir / "reference-layout.source.json"
    _write_bytes_resumable(reference_copy_path, reference_layout_path.read_bytes())
    if sha256_file(reference_copy_path) != reference_sha:
        raise ValueError("copied reference layout hash mismatch")
    evidence_frames = tuple(
        float(row["exact_pts_s"]) for row in evidence_survey["frames"]
    )

    target_layout = WildLayout.from_dict({
        **reference.to_dict(),
        "video_id": target_id,
        "gameplay_rect_source": (
            "AI-only source-bound same-family transfer; reference="
            f"{reference_video_id} layout_sha256={reference_sha}; "
            f"target_survey_sha256={survey_sha}"
        ),
        "gameplay_rect_confidence": min(reference.gameplay_rect_confidence, 0.90),
        "inference_source": (
            "AI-only semantic/geometry transfer from a visually matched uploader "
            f"layout family; reference={reference_video_id}; target source, PTS, "
            "survey, and contact sheet are hash-bound in transfer_evidence.json"
        ),
        "inference_confidence": min(reference.inference_confidence, 0.85),
        "human_reviewed": False,
        "evidence_frames_s": list(evidence_frames),
        "temporal_offset_frames": 0,
        "temporal_offset_source": "unmeasured",
        "temporal_offset_confidence": 0.0,
    })
    layout_path = out_dir / "layout.family-transfer-ai.json"
    _write_resumable(layout_path, target_layout.to_dict())

    scan_cells = []
    for cell in target_layout.cells:
        x0, y0, x1, y1 = rect_to_pixels(cell.sample_rect, width, height)
        scan_cells.append({
            "cell_id": cell.cell_id,
            "physical_label": cell.cell_id,
            "semantic_action_from_reference": cell.action,
            "sample_rect_px": [x0, y0, x1 - x0, y1 - y0],
            "decoder": cell.decoder,
            "pressed_polarity": cell.pressed_polarity,
        })
    spec = {
        "format_version": SPEC_VERSION,
        "video_id": target_id,
        "source_sha256": source_sha,
        "pts_sha256": pts_sha,
        "survey_sha256": survey_sha,
        "survey_contact_sheet_sha256": contact_sha,
        "frame_size_wh": [width, height],
        "geometry_basis": (
            "AI-only source-bound transfer from visually matched uploader/layout "
            f"family {reference_video_id}; full target scan required"
        ),
        "cells": scan_cells,
        "reviewer_kind": "ai_agent",
        "human_reviewed": False,
        "training_admitted": False,
        "limitations": [
            "layout-family equivalence was assessed by AI, not a human",
            "target full-cell scan and decode QC must pass before provisional use",
            "gameplay ranges and compositor offset remain unreviewed",
        ],
    }
    if bounded:
        spec["scan_range_s"] = [range_start_s, range_end_s]
    spec_path = out_dir / "cell-scan-spec.family-transfer-ai.json"
    _write_resumable(spec_path, spec)

    dt = float(np.median(np.diff(pts)))
    start_s = range_start_s if bounded else 0.0
    end_s = range_end_s if bounded else float(pts[-1] - pts[0] + dt)
    boundary = WildBoundaries.from_dict({
        "format_version": "madeleine.wild-boundaries.v2",
        "video_id": target_id,
        "source_sha256": source_sha,
        "wall_clock_range_s": [start_s, end_s],
        "allowed_ranges_s": [[start_s, end_s]],
        "human_reviewed": False,
        "reviewer": "OpenAI Codex layout-family provisional pipeline",
        "reviewer_kind": "ai_agent",
        "evidence": [
            "transfer_evidence.json",
            f"target_survey_sha256={survey_sha}",
            f"target_contact_sheet_sha256={contact_sha}",
        ],
        "notes": (
            "AI-only outer source envelope. Activity-aware shard building removes "
            "long no-input spans, but menus, deaths, loads, and pauses are not "
            "human-reviewed and this source is not training-admitted."
        ),
    })
    boundaries_path = out_dir / "boundaries.outer-ai.json"
    _write_resumable(boundaries_path, boundary.to_dict())

    evidence_path = out_dir / "transfer_evidence.json"
    # Transfer evidence is a content-addressed handoff.  Do not put a fresh
    # wall-clock value into new artifacts: an independently validated
    # preflight and the stock worker must produce identical bytes from the
    # same immutable inputs.  Historical v1 artifacts may already contain a
    # created_at field, so retain it in-place to preserve resumability.
    existing_evidence = (
        json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
    )
    evidence = {
        "format_version": EVIDENCE_VERSION,
        **(
            {"created_at": existing_evidence["created_at"]}
            if "created_at" in existing_evidence
            else {}
        ),
        "video_id": target_id,
        "reference_video_id": reference_video_id,
        "assessment": {
            "reviewer_kind": "ai_agent",
            "note": assessment_note.strip(),
            "coordinate_policy": "normalized coordinate reuse after exact 16:9 validation",
            "semantic_policy": "reuse only the explicit reference canonical mapping",
        },
        "bindings": {
            "target_source_sha256": source_sha,
            "target_pts_sha256": pts_sha,
            "target_survey_sha256": survey_sha,
            "target_contact_sheet_sha256": contact_sha,
            "reference_layout_sha256": reference_sha,
            "generated_layout_sha256": sha256_file(layout_path),
            "generated_scan_spec_sha256": sha256_file(spec_path),
            "generated_boundaries_sha256": sha256_file(boundaries_path),
        },
        "target": {
            "resolution_wh": [width, height],
            "frames": int(pts.size),
            "duration_s": source_duration_s,
            "survey_frames": len(evidence_survey["frames"]),
            "scan_range_s": [range_start_s, range_end_s] if bounded else None,
            "evidence_policy": (
                "bounded_sparse_then_exhaustive_scan_decode"
                if bounded
                else "full_source_minimum_eight_sparse_frames"
            ),
            "bounded_sparse_evidence_frames": (
                len(evidence_survey["frames"]) if bounded else None
            ),
        },
        "survey_cell_diagnostics": diagnostics,
        "required_next_gates": [
            (
                "exhaustive bounded target-cell scan with every canonical action "
                "mechanically valid"
                if bounded
                else "full target-cell scan with every canonical action mechanically valid"
            ),
            "exhaustive selected-range native decode timing and per-cell/action QC",
            "provisional shard validation",
            "human layout/range/offset review before any training admission",
        ],
        "human_reviewed": False,
        "training_admitted": False,
    }
    _write_resumable(evidence_path, evidence)
    return {
        "reference_layout": reference_copy_path,
        "layout": layout_path,
        "scan_spec": spec_path,
        "boundaries": boundaries_path,
        "evidence": evidence_path,
    }


def validate_full_scan(report_path: Path, layout_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    layout = WildLayout.load(layout_path)
    if report.get("video_id") != layout.video_id:
        raise ValueError("scan/layout video mismatch")
    rows = report.get("cells")
    if not isinstance(rows, list):
        raise ValueError("scan report lacks cell diagnostics")
    by_id = {str(row.get("cell_id")): row for row in rows}
    score_row = report.get("scores")
    if not isinstance(score_row, dict):
        raise ValueError("scan report lacks score bytes")
    score_path = report_path.parent / str(score_row.get("path", ""))
    shape = score_row.get("shape")
    if (
        not score_path.is_file()
        or not isinstance(shape, list)
        or shape != [int(shape[0]), len(layout.cells)]
        or score_row.get("dtype") != "float32"
        or score_path.stat().st_size != int(shape[0]) * len(layout.cells) * 4
        or sha256_file(score_path) != score_row.get("sha256")
    ):
        raise ValueError("scan score binding is invalid")
    scores = np.memmap(
        score_path,
        dtype=np.float32,
        mode="r",
        shape=(int(shape[0]), len(layout.cells)),
    )
    weak = []
    cell_validation = []
    for index, cell in enumerate(layout.cells):
        row = by_id.get(cell.cell_id)
        if row is None:
            weak.append(f"{cell.cell_id}:missing")
        elif not row.get("changing"):
            weak.append(f"{cell.cell_id}:not-changing")
        else:
            gap = float(row.get("cluster_separation_luma", 0.0))
            threshold = float(row.get("threshold", float("nan")))
            values = np.asarray(scores[:, index])
            low = values[values <= threshold]
            high = values[values > threshold]
            if not low.size or not high.size:
                weak.append(f"{cell.cell_id}:empty-state-cluster")
                continue
            low_median = float(np.median(low))
            high_median = float(np.median(high))
            low_mad = float(np.median(np.abs(low - low_median)))
            high_mad = float(np.median(np.abs(high - high_median)))
            support_gap = float(high.min() - low.max())
            pressed = high if cell.pressed_polarity == "high" else low
            pressed_mad = high_mad if cell.pressed_polarity == "high" else low_mad
            pressed_range = float(np.ptp(pressed))
            robust_floor1 = (high_median - low_median) / max(
                1.4826 * (low_mad + high_mad) / 2.0,
                1.0,
            )
            positive_runs = max(1, int(row.get("positive_runs", 0)))
            one_frame_fraction = (
                int(row.get("single_frame_positive_runs", 0)) / positive_runs
            )
            low_dynamic = (
                12.0 <= gap < 20.0
                and max(low_mad, high_mad) <= 0.5
                and robust_floor1 >= 12.0
                and int(row.get("minority_frames", 0)) >= 1_000
                and one_frame_fraction <= 0.05
            )
            # A translucent unpressed cell can legitimately inherit a broad
            # range of gameplay luminance while the pressed state is a nearly
            # constant opaque fill.  Accept that shape only when the complete
            # observed supports are separated by at least one quarter of the
            # 8-bit luma range and the pressed support itself is tightly
            # bounded.  This remains stricter than a large median gap: broad
            # two-cluster scene noise cannot satisfy the stable-state bound.
            disjoint_stable_pressed = (
                gap >= 20.0
                and support_gap >= 64.0
                and pressed_mad <= 0.5
                and pressed_range <= 8.0
                and int(row.get("minority_frames", 0)) >= 1_000
                and one_frame_fraction <= 0.05
            )
            mode = (
                "absolute_luma_gap"
                if gap >= 20.0 and robust_floor1 >= 20.0
                else "disjoint_stable_pressed_state"
                if disjoint_stable_pressed
                else "low_dynamic_binary"
                if low_dynamic
                else "rejected"
            )
            cell_validation.append({
                "cell_id": cell.cell_id,
                "validation_mode": mode,
                "absolute_gap_luma": gap,
                "low_state_mad_luma": low_mad,
                "high_state_mad_luma": high_mad,
                "inter_cluster_support_gap_luma": support_gap,
                "pressed_state_mad_luma": pressed_mad,
                "pressed_state_range_luma": pressed_range,
                "decoder_cluster_separation_floor1": robust_floor1,
                "minority_frames": int(row.get("minority_frames", 0)),
                "single_frame_positive_run_fraction": one_frame_fraction,
            })
            if mode == "rejected":
                weak.append(f"{cell.cell_id}:no-valid-separation-policy")
    if weak:
        raise ValueError("full target scan failed provisional gates: " + ", ".join(weak))
    bounded_summary: dict[str, Any] = {}
    spec_row = report.get("spec")
    if not isinstance(spec_row, dict) or not isinstance(spec_row.get("path"), str):
        raise ValueError("scan report lacks copied scan spec")
    spec_path = report_path.parent / spec_row["path"]
    if not spec_path.is_file() or sha256_file(spec_path) != spec_row.get("sha256"):
        raise ValueError("scan report copied spec binding is invalid")
    spec = json.loads(spec_path.read_text())
    if spec.get("scan_range_s") is not None:
        source = report.get("source")
        evidence = report.get("evidence")
        if not isinstance(source, dict):
            raise ValueError("bounded scan lacks source-range provenance")
        frame_range = source.get("source_frame_range")
        if (
            not isinstance(frame_range, list)
            or len(frame_range) != 2
            or any(not isinstance(value, int) for value in frame_range)
            or frame_range[0] < 0
            or frame_range[1] <= frame_range[0]
            or source.get("scanned_frames") != frame_range[1] - frame_range[0]
        ):
            raise ValueError("bounded scan source-frame coverage is not exhaustive")
        # Four sparse survey frames may nominate a deliberately bounded layout,
        # but the subsequent scan must contribute a broader exact-frame audit.
        if not isinstance(evidence, list) or len(evidence) < 8:
            raise ValueError("bounded scan requires at least eight exact evidence frames")
        bounded_summary = {
            "scan_range_s": spec["scan_range_s"],
            "source_frame_range": frame_range,
            "scanned_frames": int(source["scanned_frames"]),
            "exact_scan_evidence_frames": len(evidence),
            "evidence_policy": "bounded_sparse_then_exhaustive_scan_decode",
        }
    result = {
        "format_version": "madeleine.wild-layout-family-scan-validation.v1",
        "video_id": layout.video_id,
        "scan_report_sha256": sha256_file(report_path),
        "layout_sha256": sha256_file(layout_path),
        "validated_cells": len(layout.cells),
        "validated_actions": sorted({cell.action for cell in layout.cells}),
        "minimum_cluster_separation_luma": min(
            float(by_id[cell.cell_id]["cluster_separation_luma"])
            for cell in layout.cells
        ),
        "validation_policy": (
            "absolute_luma_or_disjoint_stable_pressed_or_low_dynamic_binary_v2"
        ),
        "cell_validation": cell_validation,
        **bounded_summary,
        "human_reviewed": False,
        "training_admitted": False,
    }
    validation_path = report_path.parent / "family_transfer_scan_validation.json"
    _write_resumable(validation_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source-dir", type=Path, required=True)
    prepare.add_argument("--survey-dir", type=Path, required=True)
    prepare.add_argument("--reference-layout", type=Path, required=True)
    prepare.add_argument("--reference-video-id", required=True)
    prepare.add_argument(
        "--assessment-note",
        default=DEFAULT_ASSESSMENT_NOTE,
        help=(
            "source-bound AI assessment; omit to use the canonical worker "
            "provenance text"
        ),
    )
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--scan-start-s", type=float)
    prepare.add_argument("--scan-end-s", type=float)
    validate = sub.add_parser("validate-scan")
    validate.add_argument("--scan-report", type=Path, required=True)
    validate.add_argument("--layout", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        paths = prepare_transfer(
            source_dir=args.source_dir,
            survey_dir=args.survey_dir,
            reference_layout_path=args.reference_layout,
            reference_video_id=args.reference_video_id,
            assessment_note=args.assessment_note,
            out_dir=args.out,
            scan_start_s=args.scan_start_s,
            scan_end_s=args.scan_end_s,
        )
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    else:
        print(json.dumps(validate_full_scan(args.scan_report, args.layout), indent=2))


if __name__ == "__main__":
    main()
