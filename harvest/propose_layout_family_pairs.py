"""Deterministically propose exact HUD-layout pairs for pairwise review.

The proposal score deliberately looks only at the normalized region occupied by
the seven validated action cells of each reference layout.  It measures stable
edge structure across the 16 exact survey frames, so changing Celeste scenes
are suppressed while a fixed keyboard/action overlay remains.  Results are
machine nominations only: a VLM pair check and a mechanical full-video cell
scan are still required before a layout can be transferred.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from harvest.classify_layout_families_vlm import (
    REFERENCE_VIDEO_IDS,
    sha256_file,
    survey_binding,
)


ALGORITHM_VERSION = "stable-hud-edges-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}: blank line {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: line {line_number} is not an object")
        rows.append(value)
    return rows


def reference_region(layout: dict[str, Any], *, pad_x: float = 0.025,
                     pad_y: float = 0.035) -> tuple[float, float, float, float]:
    """Return a bounded normalized ROI around all seven action cells."""
    cells = layout.get("cells")
    if not isinstance(cells, list) or len(cells) != 7:
        raise ValueError("reference layout must contain exactly seven cells")
    rects = [cell.get("sample_rect") for cell in cells]
    if any(
        not isinstance(rect, list)
        or len(rect) != 4
        or any(not isinstance(value, (int, float)) for value in rect)
        for rect in rects
    ):
        raise ValueError("reference layout has malformed sample rectangles")
    x0 = max(0.0, min(float(rect[0]) for rect in rects) - pad_x)
    y0 = max(0.0, min(float(rect[1]) for rect in rects) - pad_y)
    x1 = min(1.0, max(float(rect[0]) + float(rect[2]) for rect in rects) + pad_x)
    y1 = min(1.0, max(float(rect[1]) + float(rect[3]) for rect in rects) + pad_y)
    if x1 - x0 < 0.04 or y1 - y0 < 0.04:
        raise ValueError("reference action-cell region is implausibly small")
    return x0, y0, x1, y1


def _bound_survey_frames(survey_root: Path, video_id: str) -> tuple[dict[str, Any], list[Path]]:
    binding = survey_binding(survey_root, video_id)
    survey = json.loads(binding["survey_path"].read_text())
    frames = survey.get("frames")
    if not isinstance(frames, list) or len(frames) != 16:
        raise ValueError(f"{video_id}: expected exactly 16 survey frames")
    ordered_paths: dict[int, Path] = {}
    seen_orders: set[int] = set()
    for frame in frames:
        order = frame.get("sample_order")
        relative = frame.get("path")
        expected_sha = frame.get("sha256")
        if not isinstance(order, int) or order in seen_orders:
            raise ValueError(f"{video_id}: invalid sample order")
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise ValueError(f"{video_id}: unsafe survey frame path")
        path = binding["survey_path"].parent / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"{video_id}: survey frame hash mismatch")
        seen_orders.add(order)
        ordered_paths[order] = path
    if seen_orders != set(range(16)):
        raise ValueError(f"{video_id}: incomplete sample orders")
    paths = [ordered_paths[index] for index in range(16)]
    return binding, paths


def load_normalized_gray_frames(
    paths: Iterable[Path], *, width: int = 320, height: int = 180
) -> np.ndarray:
    frames: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as opened:
            image = opened.convert("L").resize((width, height), Image.Resampling.LANCZOS)
            frames.append(np.asarray(image, dtype=np.float32) / 255.0)
    value = np.stack(frames)
    if value.shape != (16, height, width):
        raise ValueError(f"unexpected normalized survey shape {value.shape}")
    return value


def stable_edge_map(frames: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
    """Extract edge energy persistent across most of the 16 survey frames."""
    if frames.ndim != 3 or frames.shape[0] != 16:
        raise ValueError("stable edge extraction requires 16 grayscale frames")
    _, height, width = frames.shape
    x0, y0, x1, y1 = region
    left = max(0, min(width - 2, round(x0 * width)))
    top = max(0, min(height - 2, round(y0 * height)))
    right = max(left + 2, min(width, round(x1 * width)))
    bottom = max(top + 2, min(height, round(y1 * height)))
    crop = frames[:, top:bottom, left:right]

    # A centered two-pixel derivative is cheap, deterministic, and sufficient
    # for the fixed cell borders/text.  The temporal lower quartile rejects
    # edges that appear in only a few changing gameplay scenes.
    gx = np.zeros_like(crop)
    gy = np.zeros_like(crop)
    gx[:, :, 1:-1] = np.abs(crop[:, :, 2:] - crop[:, :, :-2]) * 0.5
    gy[:, 1:-1, :] = np.abs(crop[:, 2:, :] - crop[:, :-2, :]) * 0.5
    gradient = np.maximum(gx, gy)
    persistent = np.quantile(gradient, 0.25, axis=0).astype(np.float32)
    scale = float(np.quantile(persistent, 0.97))
    if scale <= 1e-6:
        return np.zeros_like(persistent)
    return np.clip(persistent / scale, 0.0, 1.0)


def edge_similarity(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    if first.shape != second.shape:
        raise ValueError("edge maps must have identical shapes")
    first_flat = first.reshape(-1).astype(np.float64)
    second_flat = second.reshape(-1).astype(np.float64)
    first_centered = first_flat - first_flat.mean()
    second_centered = second_flat - second_flat.mean()
    denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    correlation = (
        float(np.dot(first_centered, second_centered) / denominator)
        if denominator > 1e-12 else 0.0
    )
    first_binary = first_flat >= float(np.quantile(first_flat, 0.85))
    second_binary = second_flat >= float(np.quantile(second_flat, 0.85))
    binary_denominator = int(first_binary.sum()) + int(second_binary.sum())
    dice = (
        2.0 * float(np.logical_and(first_binary, second_binary).sum())
        / binary_denominator
        if binary_denominator else 0.0
    )
    score = 0.7 * max(-1.0, correlation) + 0.3 * dice
    return {
        "score": round(score, 8),
        "correlation": round(correlation, 8),
        "edge_dice": round(dice, 8),
    }


def rank_references(
    candidate_frames: np.ndarray,
    references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked = []
    for reference in references:
        region = tuple(reference["region"])
        candidate_edges = stable_edge_map(candidate_frames, region)
        metrics = edge_similarity(candidate_edges, reference["edges"])
        ranked.append({
            "reference_video_id": reference["video_id"],
            "reference_survey_sha256": reference["survey_sha256"],
            "reference_contact_sha256": reference["contact_sha256"],
            "reference_layout_sha256": reference["layout_sha256"],
            "region_normalized": list(region),
            **metrics,
        })
    return sorted(ranked, key=lambda row: (-row["score"], row["reference_video_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-root", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--reference-layout-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.28)
    parser.add_argument("--min-margin", type=float, default=0.06)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    predictions = _jsonl(args.candidate_predictions)
    candidates = [
        row for row in predictions
        if row.get("full_scan_candidate") is True
        and str(row.get("video_id")) not in REFERENCE_VIDEO_IDS
    ]
    reference_records: list[dict[str, Any]] = []
    for video_id in REFERENCE_VIDEO_IDS:
        layout_path = args.reference_layout_root / video_id / "layout.exact-ai-nomination.json"
        if not layout_path.is_file():
            raise FileNotFoundError(f"missing reference layout: {layout_path}")
        layout = json.loads(layout_path.read_text())
        if layout.get("video_id") != video_id:
            raise ValueError(f"{video_id}: reference layout binding mismatch")
        binding, frame_paths = _bound_survey_frames(args.survey_root, video_id)
        frames = load_normalized_gray_frames(frame_paths)
        region = reference_region(layout)
        reference_records.append({
            "video_id": video_id,
            "survey_sha256": binding["survey_sha256"],
            "contact_sha256": binding["contact_sha256"],
            "layout_sha256": sha256_file(layout_path),
            "region": region,
            "edges": stable_edge_map(frames, region),
        })

    records: list[dict[str, Any]] = []
    for prediction in candidates:
        video_id = str(prediction["video_id"])
        binding, frame_paths = _bound_survey_frames(args.survey_root, video_id)
        if prediction.get("survey_sha256") != binding["survey_sha256"]:
            raise ValueError(f"{video_id}: candidate prediction survey hash mismatch")
        frames = load_normalized_gray_frames(frame_paths)
        ranking = rank_references(frames, reference_records)
        top = ranking[0]
        runner_up = ranking[1]
        margin = float(top["score"]) - float(runner_up["score"])
        proposed = top["score"] >= args.min_score and margin >= args.min_margin
        records.append({
            "schema_version": 1,
            "algorithm_version": ALGORITHM_VERSION,
            "video_id": video_id,
            "candidate_survey_sha256": binding["survey_sha256"],
            "candidate_contact_sha256": binding["contact_sha256"],
            "candidate_class": prediction.get("class"),
            "candidate_confidence": prediction.get("confidence"),
            "proposed_reference_video_id": top["reference_video_id"] if proposed else None,
            "proposal_passed": proposed,
            "top_score": top["score"],
            "runner_up_score": runner_up["score"],
            "margin": round(margin, 8),
            "ranking": ranking,
            "human_reviewed": False,
            "training_admitted": False,
            "pairwise_vlm_required": True,
            "mechanical_full_scan_required": True,
            "created_at": utc_now(),
        })
    records.sort(key=lambda row: (-row["margin"], -row["top_score"], row["video_id"]))
    if args.limit is not None:
        records = records[:args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "created_at": utc_now(),
        "human_reviewed": False,
        "training_admitted": False,
        "pairwise_vlm_required": True,
        "mechanical_full_scan_required": True,
        "inputs": {
            "candidate_predictions": str(args.candidate_predictions),
            "candidate_predictions_sha256": sha256_file(args.candidate_predictions),
            "reference_layout_root": str(args.reference_layout_root),
            "reference_video_ids": list(REFERENCE_VIDEO_IDS),
        },
        "thresholds": {"min_score": args.min_score, "min_margin": args.min_margin},
        "results": {
            "rows": len(records),
            "proposals": sum(row["proposal_passed"] for row in records),
            "sha256": sha256_file(args.out),
        },
    }
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest["results"], sort_keys=True))


if __name__ == "__main__":
    main()
