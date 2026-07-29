"""Verify nominated input HUDs with fixed-crop, pairwise Qwen 7B evidence.

This stage mechanically rejects static overlay artwork after a video has been
nominated by the survey reclassifier.  Each generative call sees exactly two
crops of the same fixed screen region at two different exact survey frames and
answers only whether an input-indicator state changed.  The region is never
taken from the stage-A ``location`` field — calibration showed that signal is
unreliable — instead the four overlapping quadrants are each swept with the
same pairwise question and the per-quadrant verdicts are reduced
conservatively.

Every queue, stage-A file, survey, source image, crop, prompt, model revision,
and output prefix is SHA-256 bound.  The JSONL is append-only and fsynced one
record at a time; its atomic manifest makes a crash between the row append and
manifest update safely resumable.  Every result remains an AI nomination with
``human_reviewed=false`` and ``training_admitted=false``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harvest.reclassify_layout_surveys_vlm7b import (
    ALLOWED_CLASSES,
    DEFAULT_MODEL,
    MANIFEST_SCHEMA_VERSION,
    RECORD_SCHEMA_VERSION,
    RETRY_POLICY,
    _IMMUTABLE_REVISION,
    _manifest_prediction_state,
    _png_bytes,
    _strict_json_object,
    available_surveys,
    build_corrective_message,
    canonical_sha256,
    choose_evidence_frames,
    finalize_response_attempts,
    load_queue_rows,
    quadrant_boxes,
    sha256_file,
    sha256_text,
    utc_now,
    validate_resume_manifest,
    write_json_atomic,
)


PROMPT_VERSION = "celeste-hud-change-pair-stacked-v2-qwen7b"
PROMPT_TEMPLATE = """The image shows the SAME fixed screen region captured at
two different moments of one video: the TOP half is moment A and the BOTTOM
half is moment B, separated by a solid red bar.

Compare the two halves element by element. Do any INPUT-INDICATOR elements
(key caps, action-label cells, controller buttons, d-pad arms, stick
positions) show a DIFFERENT pressed/active/filled state between moment A and
moment B? A key cap or button that is filled, highlighted, or darkened in one
half but not the other is a different state. Ignore gameplay, timers, splits
text, chat, and camera content.

Respond with exactly:
{"verdict":"different_state|same_state|no_input_overlay|illegible","differing_controls":["..."],"evidence":"..."}

Return ONLY one JSON object with EXACTLY the required keys and allowed values,
no markdown."""

ALLOWED_VERDICTS = {
    "different_state",
    "same_state",
    "no_input_overlay",
    "illegible",
}
PAIR_RESPONSE_KEYS = {"verdict", "differing_controls", "evidence"}
STAGE_A_CANDIDATE_CLASSES = {"decodable_input_hud", "uncertain"}
AGGREGATE_RESULTS = {
    "changing_overlay_confirmed",
    "static_overlay",
    "no_overlay",
    "insufficient_evidence",
}
STAGE_B_CLASSES = {"decodable_input_hud", "non_decodable", "uncertain"}
EVIDENCE_FRAME_COUNT = 6
CROP_SPEC_VERSION = "overlap-quadrant-sweep-stacked-v3"
STACK_DIVIDER_PX = 8
SWEEP_REGIONS = ("top_left", "top_right", "bottom_left", "bottom_right")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def load_stage_a_predictions(path: Path) -> list[dict[str, Any]]:
    """Load stage-A rows with the minimal identity checks needed by stage B."""

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f"stage-A line {line_number} is blank")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"stage-A line {line_number} is invalid JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"stage-A line {line_number} is not an object")
        video_id = str(record.get("video_id", ""))
        if not _SAFE_ID.fullmatch(video_id):
            raise ValueError(f"stage-A line {line_number} has unsafe video_id")
        if video_id in seen:
            raise ValueError(f"stage-A predictions contain duplicate video_id {video_id}")
        if record.get("class") not in ALLOWED_CLASSES:
            raise ValueError(f"stage-A prediction {video_id} has invalid class")
        if record.get("human_reviewed") is not False:
            raise ValueError(f"stage-A prediction {video_id} must not claim human review")
        seen.add(video_id)
        if record["class"] in STAGE_A_CANDIDATE_CLASSES:
            output.append(record)
    return output



def construct_pairs(frame_count_or_frames: int | Sequence[Any]) -> list[tuple[int, int]]:
    """Return ordered adjacent frame indices plus the first/last pair."""

    frame_count = (
        frame_count_or_frames
        if type(frame_count_or_frames) is int
        else len(frame_count_or_frames)
    )
    if frame_count < 2:
        return []
    proposed = [
        *((index, index + 1) for index in range(frame_count - 1)),
        (0, frame_count - 1),
    ]
    output: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair in proposed:
        if pair not in seen:
            output.append(pair)
            seen.add(pair)
    return output


def build_frame_pairs(
    frame_count_or_frames: int | Sequence[Any],
) -> list[tuple[int, int]]:
    """Descriptive alias for deterministic pair construction."""

    return construct_pairs(frame_count_or_frames)


def stack_pair(top: Any, bottom: Any) -> Any:
    """Stack two same-size crops vertically with a solid red divider.

    A single composite image is presented to the model because Qwen 7B
    compares fine element states far more reliably within one image than
    across two separate images.
    """

    from PIL import Image

    if top.size != bottom.size:
        raise ValueError("stacked crops must share dimensions")
    width, height = top.size
    composite = Image.new(
        "RGB", (width, height * 2 + STACK_DIVIDER_PX), (255, 0, 0)
    )
    composite.paste(top, (0, 0))
    composite.paste(bottom, (0, height + STACK_DIVIDER_PX))
    return composite


def _fail_closed_pair(errors: Iterable[str]) -> dict[str, Any]:
    error_list = list(errors)
    return {
        "verdict": "illegible",
        "differing_controls": [],
        "evidence": "model output failed strict validation",
        "validation_errors": error_list or ["unknown_validation_error"],
    }


def parse_pair_response(raw: str) -> dict[str, Any]:
    """Strictly validate one pair response, failing closed to illegible."""

    value, errors = _strict_json_object(raw)
    if value is None:
        return _fail_closed_pair(errors)
    if set(value) != PAIR_RESPONSE_KEYS:
        missing = sorted(PAIR_RESPONSE_KEYS - set(value))
        extra = sorted(set(value) - PAIR_RESPONSE_KEYS)
        if missing:
            errors.append("missing_keys:" + ",".join(missing))
        if extra:
            errors.append("extra_keys:" + ",".join(extra))

    verdict = value.get("verdict")
    differing_controls = value.get("differing_controls")
    evidence = value.get("evidence")
    if not isinstance(verdict, str) or verdict not in ALLOWED_VERDICTS:
        errors.append("invalid_verdict")
    if (
        not isinstance(differing_controls, list)
        or len(differing_controls) > 16
        or any(
            not isinstance(item, str) or not item.strip() or len(item) > 64
            for item in differing_controls
        )
    ):
        errors.append("invalid_differing_controls")
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 400:
        errors.append("invalid_evidence")
    if isinstance(verdict, str) and isinstance(differing_controls, list):
        if verdict == "different_state" and not differing_controls:
            errors.append("different_state_lacks_differing_controls")
        if verdict in ALLOWED_VERDICTS - {"different_state"} and differing_controls:
            errors.append("non_different_state_has_differing_controls")
    if errors:
        return _fail_closed_pair(errors)

    assert isinstance(verdict, str)
    assert isinstance(differing_controls, list)
    assert isinstance(evidence, str)
    return {
        "verdict": verdict,
        "differing_controls": [
            " ".join(control.split()) for control in differing_controls
        ],
        "evidence": " ".join(evidence.split()),
        "validation_errors": [],
    }


def parse_response(raw: str) -> dict[str, Any]:
    """Mirror the stage-A parser name for callers of this classifier."""

    return parse_pair_response(raw)


def aggregate_pair_verdicts(
    pair_verdicts: Iterable[str | dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate valid pair verdicts using the fixed stage-B truth table."""

    counts = {
        "n_diff": 0,
        "n_same": 0,
        "n_none": 0,
        "n_illeg": 0,
    }
    invalid = 0
    count_key = {
        "different_state": "n_diff",
        "same_state": "n_same",
        "no_input_overlay": "n_none",
        "illegible": "n_illeg",
    }
    for item in pair_verdicts:
        if isinstance(item, dict):
            errors = item.get("validation_errors", [])
            verdict = item.get("verdict")
            if errors:
                invalid += 1
                continue
        else:
            verdict = item
        if verdict not in count_key:
            invalid += 1
            continue
        counts[count_key[verdict]] += 1

    valid = counts["n_diff"] + counts["n_same"] + counts["n_none"]
    if counts["n_diff"] >= 2:
        result = "changing_overlay_confirmed"
    elif (
        counts["n_diff"] == 0
        and counts["n_same"] >= max(3, math.ceil(valid / 2))
    ):
        result = "static_overlay"
    elif (
        counts["n_none"] > counts["n_diff"] + counts["n_same"]
        and counts["n_none"] >= 3
    ):
        result = "no_overlay"
    else:
        result = "insufficient_evidence"
    return {
        **counts,
        "invalid_pairs": invalid,
        "valid": valid,
        "result": result,
    }


def aggregate_regions(
    pair_verdicts: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-region pair verdicts, then reduce conservatively.

    A changing overlay anywhere confirms the video; a same-state overlay seen
    in some region without any confirmed change is static; ``no_overlay`` is
    claimed only when every swept region independently concludes it.
    """

    by_region: dict[str, list[dict[str, Any]]] = {}
    for verdict in pair_verdicts:
        region = str(verdict.get("region", "unknown"))
        by_region.setdefault(region, []).append(verdict)
    region_aggregates = {
        region: aggregate_pair_verdicts(verdicts)
        for region, verdicts in sorted(by_region.items())
    }
    results = [aggregate["result"] for aggregate in region_aggregates.values()]
    if any(result == "changing_overlay_confirmed" for result in results):
        reduced = "changing_overlay_confirmed"
    elif any(result == "static_overlay" for result in results):
        reduced = "static_overlay"
    elif results and all(result == "no_overlay" for result in results):
        reduced = "no_overlay"
    else:
        reduced = "insufficient_evidence"
    return {"regions": region_aggregates, "result": reduced}


def route_aggregate_result(result: str) -> dict[str, Any]:
    if result == "changing_overlay_confirmed":
        return {
            "stage_b_class": "decodable_input_hud",
            "non_decodable_reason": "none",
            "review_candidate": True,
        }
    if result == "static_overlay":
        return {
            "stage_b_class": "non_decodable",
            "non_decodable_reason": "static_or_frozen",
            "review_candidate": False,
        }
    if result == "no_overlay":
        return {
            "stage_b_class": "non_decodable",
            "non_decodable_reason": "no_input_hud",
            "review_candidate": False,
        }
    if result == "insufficient_evidence":
        return {
            "stage_b_class": "uncertain",
            "non_decodable_reason": "none",
            "review_candidate": True,
        }
    raise ValueError(f"unsupported aggregate result {result!r}")


@dataclass
class PairEvidence:
    images: tuple[Any, Any]
    descriptor: dict[str, Any]


@dataclass
class EvidenceBundle:
    status: str
    crop_boxes_pixel_xyxy: dict[str, list[int]]
    selected_sample_orders: list[int]
    frame_sizes: list[list[int]]
    pairs: list[PairEvidence]
    image_bundle_sha256: str

    @property
    def descriptors(self) -> list[dict[str, Any]]:
        return [pair.descriptor for pair in self.pairs]


def prepare_evidence(row: dict[str, Any], stage_a: dict[str, Any]) -> EvidenceBundle:
    """Prepare the quadrant-sweep crops and deterministic frame pairs.

    The same four overlapping quadrant pixel boxes are applied to every
    selected frame; the stage-A ``location`` is recorded but never trusted
    for geometry.
    """

    from PIL import Image

    selected = choose_evidence_frames(row["frames"], EVIDENCE_FRAME_COUNT)
    sample_orders = [int(frame["sample_order"]) for frame in selected]

    frames: list[Any] = []
    crops: list[Any] = []
    try:
        for frame in selected:
            with Image.open(frame["absolute_path"]) as opened:
                frames.append(opened.convert("RGB"))
        frame_sizes = [[image.width, image.height] for image in frames]
        first_size = frames[0].size
        boxes = quadrant_boxes(*first_size)
        crop_boxes = {
            region: list(boxes[region]) for region in SWEEP_REGIONS
        }
        if any(image.size != first_size for image in frames[1:]):
            return EvidenceBundle(
                status="frame_geometry_mismatch",
                crop_boxes_pixel_xyxy=crop_boxes,
                selected_sample_orders=sample_orders,
                frame_sizes=frame_sizes,
                pairs=[],
                image_bundle_sha256=canonical_sha256([]),
            )

        pairs: list[PairEvidence] = []
        pair_order = 0
        index_pairs = construct_pairs(selected)
        for region in SWEEP_REGIONS:
            box = tuple(crop_boxes[region])
            region_crops: list[Any] = []
            region_hashes: list[str] = []
            for image in frames:
                crop = image.crop(box)
                crops.append(crop)
                region_crops.append(crop)
                region_hashes.append(
                    hashlib.sha256(_png_bytes(crop)).hexdigest()
                )
            for first, second in index_pairs:
                composite = stack_pair(
                    region_crops[first], region_crops[second]
                )
                crops.append(composite)
                descriptor = {
                    "pair_order": pair_order,
                    "region": region,
                    "sample_orders": [
                        int(selected[first]["sample_order"]),
                        int(selected[second]["sample_order"]),
                    ],
                    "crop_image_sha256s": [
                        region_hashes[first],
                        region_hashes[second],
                    ],
                    "composite_sha256": hashlib.sha256(
                        _png_bytes(composite)
                    ).hexdigest(),
                }
                pairs.append(
                    PairEvidence(
                        images=(composite,),
                        descriptor=descriptor,
                    )
                )
                pair_order += 1
        descriptors = [pair.descriptor for pair in pairs]
        return EvidenceBundle(
            status="classified",
            crop_boxes_pixel_xyxy=crop_boxes,
            selected_sample_orders=sample_orders,
            frame_sizes=frame_sizes,
            pairs=pairs,
            image_bundle_sha256=canonical_sha256(descriptors),
        )
    except Exception:
        for crop in crops:
            try:
                crop.close()
            except Exception:
                pass
        raise
    finally:
        for image in frames:
            image.close()


def close_evidence(bundle: EvidenceBundle) -> None:
    seen: set[int] = set()
    for pair in bundle.pairs:
        for image in pair.images:
            if id(image) not in seen:
                image.close()
                seen.add(id(image))


def _base_record(
    row: dict[str, Any],
    stage_a: dict[str, Any],
    bundle: EvidenceBundle,
    *,
    queue_sha256: str,
    stage_a_sha256: str,
    model: str,
    resolved_model_revision: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "video_id": str(row["video_id"]),
        "queue_sha256": queue_sha256,
        "stage_a_sha256": stage_a_sha256,
        "stage_a_class": stage_a["class"],
        "stage_a_modality": str(stage_a.get("modality", "unknown")),
        "stage_a_location": str(stage_a.get("location", "unknown")),
        "survey_sha256": row["survey_sha256"],
        "source_sha256": row["source_sha256"],
        "processing_status": bundle.status,
        "crop_boxes_pixel_xyxy": bundle.crop_boxes_pixel_xyxy,
        "selected_sample_orders": bundle.selected_sample_orders,
        "selected_frame_sizes": bundle.frame_sizes,
        "image_bundle_sha256": bundle.image_bundle_sha256,
        "pair_descriptors": bundle.descriptors,
        "model": model,
        "resolved_model_revision": resolved_model_revision,
        "prompt_version": PROMPT_VERSION,
        "prompt_template_sha256": sha256_text(PROMPT_TEMPLATE),
        "crop_spec_version": CROP_SPEC_VERSION,
        "max_new_tokens": max_new_tokens,
        "classified_at": utc_now(),
        "human_reviewed": False,
        "training_admitted": False,
    }


def _nonclassified_record(
    row: dict[str, Any],
    stage_a: dict[str, Any],
    bundle: EvidenceBundle,
    *,
    queue_sha256: str,
    stage_a_sha256: str,
    model: str,
    resolved_model_revision: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    flat = aggregate_pair_verdicts([])
    regional = aggregate_regions([])
    return {
        **_base_record(
            row,
            stage_a,
            bundle,
            queue_sha256=queue_sha256,
            stage_a_sha256=stage_a_sha256,
            model=model,
            resolved_model_revision=resolved_model_revision,
            max_new_tokens=max_new_tokens,
        ),
        "pair_verdicts": [],
        "aggregate_counts": {
            key: flat[key] for key in ("n_diff", "n_same", "n_none", "n_illeg")
        },
        "invalid_pair_count": flat["invalid_pairs"],
        "region_aggregates": regional["regions"],
        "aggregate_result": regional["result"],
        **route_aggregate_result(regional["result"]),
    }


def validate_record_identity(
    record: dict[str, Any],
    row: dict[str, Any],
    stage_a: dict[str, Any],
    bundle: EvidenceBundle,
    *,
    queue_sha256: str,
    stage_a_sha256: str,
    model: str,
    resolved_model_revision: str,
    max_new_tokens: int,
) -> None:
    video_id = str(row["video_id"])
    expected = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "video_id": video_id,
        "queue_sha256": queue_sha256,
        "stage_a_sha256": stage_a_sha256,
        "stage_a_class": stage_a["class"],
        "stage_a_modality": str(stage_a.get("modality", "unknown")),
        "stage_a_location": str(stage_a.get("location", "unknown")),
        "survey_sha256": row["survey_sha256"],
        "source_sha256": row["source_sha256"],
        "processing_status": bundle.status,
        "crop_boxes_pixel_xyxy": bundle.crop_boxes_pixel_xyxy,
        "selected_sample_orders": bundle.selected_sample_orders,
        "selected_frame_sizes": bundle.frame_sizes,
        "image_bundle_sha256": bundle.image_bundle_sha256,
        "pair_descriptors": bundle.descriptors,
        "model": model,
        "resolved_model_revision": resolved_model_revision,
        "prompt_version": PROMPT_VERSION,
        "prompt_template_sha256": sha256_text(PROMPT_TEMPLATE),
        "crop_spec_version": CROP_SPEC_VERSION,
        "max_new_tokens": max_new_tokens,
        "human_reviewed": False,
        "training_admitted": False,
    }
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            raise ValueError(f"prediction {video_id} has stale or mismatched {field}")
    if record.get("aggregate_result") not in AGGREGATE_RESULTS:
        raise ValueError(f"prediction {video_id} has invalid aggregate result")
    if record.get("stage_b_class") not in STAGE_B_CLASSES:
        raise ValueError(f"prediction {video_id} has invalid stage-B class")
    expected_route = route_aggregate_result(str(record["aggregate_result"]))
    for field, expected_value in expected_route.items():
        if record.get(field) != expected_value:
            raise ValueError(f"prediction {video_id} has inconsistent {field}")
    pair_verdicts = record.get("pair_verdicts")
    if not isinstance(pair_verdicts, list):
        raise ValueError(f"prediction {video_id} has invalid pair verdicts")
    if bundle.status == "classified" and len(pair_verdicts) != len(bundle.pairs):
        raise ValueError(f"prediction {video_id} has incomplete pair verdicts")
    if bundle.status != "classified" and pair_verdicts:
        raise ValueError(f"prediction {video_id} has unexpected pair verdicts")
    for pair_index, pair_verdict in enumerate(pair_verdicts):
        if not isinstance(pair_verdict, dict):
            raise ValueError(f"prediction {video_id} has malformed pair verdict")
        descriptor = bundle.pairs[pair_index].descriptor
        for field in ("pair_order", "region", "sample_orders", "crop_image_sha256s"):
            if pair_verdict.get(field) != descriptor[field]:
                raise ValueError(
                    f"prediction {video_id} has mismatched pair verdict {field}"
                )
        if pair_verdict.get("verdict") not in ALLOWED_VERDICTS:
            raise ValueError(f"prediction {video_id} has invalid pair verdict")
        retry_count = pair_verdict.get("retry_count")
        if type(retry_count) is not int or retry_count not in {0, 1}:
            raise ValueError(f"prediction {video_id} has invalid pair retry_count")
        has_retry_raw = "raw_response_retry" in pair_verdict
        has_retry_errors = "retry_validation_errors" in pair_verdict
        if retry_count == 0 and (has_retry_raw or has_retry_errors):
            raise ValueError(f"prediction {video_id} has inconsistent pair retry")
        if retry_count == 1 and not isinstance(
            pair_verdict.get("raw_response_retry"), str
        ):
            raise ValueError(f"prediction {video_id} lacks pair retry response")
        if has_retry_errors and (
            not isinstance(pair_verdict["retry_validation_errors"], list)
            or not pair_verdict["retry_validation_errors"]
        ):
            raise ValueError(f"prediction {video_id} has invalid pair retry errors")
    flat = aggregate_pair_verdicts(pair_verdicts)
    expected_counts = {
        key: flat[key] for key in ("n_diff", "n_same", "n_none", "n_illeg")
    }
    if record.get("aggregate_counts") != expected_counts:
        raise ValueError(f"prediction {video_id} has inconsistent aggregate counts")
    if record.get("invalid_pair_count") != flat["invalid_pairs"]:
        raise ValueError(f"prediction {video_id} has inconsistent invalid pair count")
    regional = aggregate_regions(pair_verdicts)
    if record.get("region_aggregates") != regional["regions"]:
        raise ValueError(f"prediction {video_id} has inconsistent region aggregates")
    if record.get("aggregate_result") != regional["result"]:
        raise ValueError(f"prediction {video_id} has inconsistent aggregation")
    if not isinstance(record.get("classified_at"), str) or not record["classified_at"]:
        raise ValueError(f"prediction {video_id} lacks classified_at")


def load_existing_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    stage_a_by_id: dict[str, dict[str, Any]],
    bundles: dict[str, EvidenceBundle],
    *,
    queue_sha256: str,
    stage_a_sha256: str,
    model: str,
    resolved_model_revision: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    row_by_id = {str(row["video_id"]): row for row in rows}
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f"prediction line {line_number} is blank")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"prediction line {line_number} is invalid JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"prediction line {line_number} is not an object")
        video_id = str(record.get("video_id", ""))
        if video_id in seen:
            raise ValueError(f"predictions contain duplicate video_id {video_id}")
        if video_id not in row_by_id:
            raise ValueError(f"prediction {video_id} lacks a current stage-B input")
        validate_record_identity(
            record,
            row_by_id[video_id],
            stage_a_by_id[video_id],
            bundles[video_id],
            queue_sha256=queue_sha256,
            stage_a_sha256=stage_a_sha256,
            model=model,
            resolved_model_revision=resolved_model_revision,
            max_new_tokens=max_new_tokens,
        )
        seen.add(video_id)
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--survey-root", type=Path, required=True)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    args = parser.parse_args()

    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        raise ValueError("invalid processor pixel bounds")
    if args.max_new_tokens < 64:
        raise ValueError("--max-new-tokens is too small for the strict response schema")

    import torch
    import transformers

    resolved_config = transformers.AutoConfig.from_pretrained(
        args.model, revision=args.revision
    )
    revision = str(getattr(resolved_config, "_commit_hash", "") or "")
    if not _IMMUTABLE_REVISION.fullmatch(revision):
        raise RuntimeError(
            "model revision did not resolve to an immutable commit; "
            "use a Hugging Face commit revision"
        )

    queue_sha256 = sha256_file(args.queue)
    stage_a_sha256 = sha256_file(args.stage_a)
    queue_rows = load_queue_rows(args.queue)
    queue_ids = {str(row["video_id"]) for row in queue_rows}
    stage_a_rows = load_stage_a_predictions(args.stage_a)
    unknown_stage_a_ids = {
        str(record["video_id"]) for record in stage_a_rows
    } - queue_ids
    if unknown_stage_a_ids:
        raise ValueError(
            "stage-A predictions are absent from the queue: "
            + ",".join(sorted(unknown_stage_a_ids))
        )
    stage_a_by_id = {
        str(record["video_id"]): record
        for record in stage_a_rows
    }
    available = available_surveys(
        args.queue, args.survey_root, available_only=args.available_only
    )
    rows = [
        row for row in available if str(row["video_id"]) in stage_a_by_id
    ]
    bundles: dict[str, EvidenceBundle] = {}
    try:
        for row in rows:
            video_id = str(row["video_id"])
            bundles[video_id] = prepare_evidence(row, stage_a_by_id[video_id])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        existing = load_existing_predictions(
            args.out,
            rows,
            stage_a_by_id,
            bundles,
            queue_sha256=queue_sha256,
            stage_a_sha256=stage_a_sha256,
            model=args.model,
            resolved_model_revision=revision,
            max_new_tokens=args.max_new_tokens,
        )
        done = {str(record["video_id"]) for record in existing}
        pending = [row for row in rows if str(row["video_id"]) not in done]
        manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
        manifest_identity = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "task": "celeste_pairwise_hud_change_verification",
            "classification_is_human_review": False,
            "training_admission": False,
            "queue_sha256": queue_sha256,
            "stage_a_sha256": stage_a_sha256,
            "prompt_version": PROMPT_VERSION,
            "prompt_template_sha256": sha256_text(PROMPT_TEMPLATE),
            "model": args.model,
            "resolved_model_revision": revision,
            "evidence_frame_count": EVIDENCE_FRAME_COUNT,
            "crop_spec_version": CROP_SPEC_VERSION,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "max_new_tokens": args.max_new_tokens,
            "retry_policy": RETRY_POLICY,
        }
        created_at = validate_resume_manifest(
            manifest_path, manifest_identity, args.out, len(existing)
        )
        args.out.touch(exist_ok=True)
        manifest = {
            **manifest_identity,
            "created_at": created_at,
            "updated_at": utc_now(),
            "state": "running" if pending else "complete",
            "queue": str(args.queue.resolve()),
            "survey_root": str(args.survey_root.resolve()),
            "stage_a": str(args.stage_a.resolve()),
            "available_rows": len(rows),
            "already_complete_on_start": len(existing),
            "stage_b_classes": sorted(STAGE_B_CLASSES),
            "aggregate_results": sorted(AGGREGATE_RESULTS),
            "predictions": _manifest_prediction_state(args.out, existing),
        }
        write_json_atomic(manifest_path, manifest)
        if not pending:
            manifest["completed_at"] = utc_now()
            manifest["class_counts_total"] = {
                label: sum(record["stage_b_class"] == label for record in existing)
                for label in STAGE_B_CLASSES
            }
            write_json_atomic(manifest_path, manifest)
            print(f"[0/0] validated {len(existing)} completed predictions")
            return

        requires_model = any(
            bundles[str(row["video_id"])].status == "classified"
            for row in pending
        )
        processor = None
        model = None
        process_vision_info = None
        if requires_model:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for production 7B verification")
            from qwen_vl_utils import process_vision_info as qwen_process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            process_vision_info = qwen_process_vision_info
            processor = AutoProcessor.from_pretrained(
                args.model,
                revision=revision,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
            processor.tokenizer.padding_side = "left"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model,
                revision=revision,
                config=resolved_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map="cuda",
            ).eval()
            if str(getattr(model.config, "_commit_hash", "") or "") != revision:
                raise RuntimeError("loaded model revision differs from resolved config")
            manifest["cuda_device"] = torch.cuda.get_device_name(0)
            manifest["torch_version"] = torch.__version__
            manifest["transformers_version"] = transformers.__version__
            write_json_atomic(manifest_path, manifest)

        def generate(conversation: list[dict[str, Any]]) -> str:
            assert processor is not None
            assert model is not None
            assert process_vision_info is not None
            rendered = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info([conversation])
            inputs = processor(
                text=[rendered],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            trimmed = generated[0][len(inputs.input_ids[0]) :]
            return processor.decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        started = time.monotonic()
        new_records: list[dict[str, Any]] = []
        with args.out.open("a") as handle, torch.inference_mode():
            for index, row in enumerate(pending, 1):
                video_id = str(row["video_id"])
                stage_a = stage_a_by_id[video_id]
                bundle = bundles[video_id]
                if bundle.status != "classified":
                    record = _nonclassified_record(
                        row,
                        stage_a,
                        bundle,
                        queue_sha256=queue_sha256,
                        stage_a_sha256=stage_a_sha256,
                        model=args.model,
                        resolved_model_revision=revision,
                        max_new_tokens=args.max_new_tokens,
                    )
                else:
                    pair_verdicts: list[dict[str, Any]] = []
                    for pair in bundle.pairs:
                        content = [
                            {
                                "type": "image",
                                "image": image,
                                "min_pixels": args.min_pixels,
                                "max_pixels": args.max_pixels,
                            }
                            for image in pair.images
                        ]
                        content.append({"type": "text", "text": PROMPT_TEMPLATE})
                        conversation = [{"role": "user", "content": content}]
                        raw = generate(conversation)
                        first_parsed = parse_pair_response(raw)
                        if first_parsed["validation_errors"]:
                            retry_conversation = [
                                *conversation,
                                {"role": "assistant", "content": raw[:4000]},
                                {
                                    "role": "user",
                                    "content": build_corrective_message(
                                        first_parsed["validation_errors"]
                                    ),
                                },
                            ]
                            retry_raw = generate(retry_conversation)
                            retry_parsed = parse_pair_response(retry_raw)
                            parsed = finalize_response_attempts(
                                raw,
                                first_parsed,
                                retry_raw=retry_raw,
                                retry_parsed=retry_parsed,
                            )
                        else:
                            parsed = finalize_response_attempts(raw, first_parsed)
                        pair_verdicts.append({**pair.descriptor, **parsed})
                    flat = aggregate_pair_verdicts(pair_verdicts)
                    regional = aggregate_regions(pair_verdicts)
                    record = {
                        **_base_record(
                            row,
                            stage_a,
                            bundle,
                            queue_sha256=queue_sha256,
                            stage_a_sha256=stage_a_sha256,
                            model=args.model,
                            resolved_model_revision=revision,
                            max_new_tokens=args.max_new_tokens,
                        ),
                        "pair_verdicts": pair_verdicts,
                        "aggregate_counts": {
                            key: flat[key]
                            for key in ("n_diff", "n_same", "n_none", "n_illeg")
                        },
                        "invalid_pair_count": flat["invalid_pairs"],
                        "region_aggregates": regional["regions"],
                        "aggregate_result": regional["result"],
                        **route_aggregate_result(regional["result"]),
                    }
                validate_record_identity(
                    record,
                    row,
                    stage_a,
                    bundle,
                    queue_sha256=queue_sha256,
                    stage_a_sha256=stage_a_sha256,
                    model=args.model,
                    resolved_model_revision=revision,
                    max_new_tokens=args.max_new_tokens,
                )
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                new_records.append(record)
                all_records = [*existing, *new_records]
                manifest["updated_at"] = utc_now()
                manifest["predictions"] = _manifest_prediction_state(
                    args.out, all_records
                )
                manifest["class_counts_total"] = {
                    label: sum(
                        completed["stage_b_class"] == label
                        for completed in all_records
                    )
                    for label in STAGE_B_CLASSES
                }
                write_json_atomic(manifest_path, manifest)
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[{index}/{len(pending)}] {video_id} "
                    f"{record['stage_b_class']} rate={index / elapsed:.3f}/s",
                    flush=True,
                )
        manifest["state"] = "complete"
        manifest["updated_at"] = utc_now()
        manifest["completed_at"] = utc_now()
        manifest["newly_completed"] = len(new_records)
        write_json_atomic(manifest_path, manifest)
    finally:
        for bundle in bundles.values():
            close_evidence(bundle)


if __name__ == "__main__":
    main()
