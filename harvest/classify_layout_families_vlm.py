"""AI-only visual layout-family matching for surveyed action HUDs.

This is a scheduling aid, not human review or training admission.  It watches
the append-only layout-stability predictions, compares positive candidates to
an explicit set of source-bound reference surveys, and emits either an exact
normalized-layout match, a complete normalized seven-action geometry, or an
unknown decision.  Mechanical full-cell validation remains mandatory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any


REFERENCE_VIDEO_IDS = (
    "0H53eYMpGsg",
    "FbUnym2i63o",
    "Fkzu6WzQi9U",
    "HUUMakvu-Cw",
    "I8C6-fhrXsQ",
    "IFD6cFcejkk",
    "JLDcOChdUl0",
    "Lud9iWXM9Fo",
    "NufclndH-8M",
    "PWEZdSuC44o",
    "Qfl1vGKauVg",
    "Reo4YFyPqB0",
    "S7ZV8PeeutY",
    "pkDueqiepaQ",
    "wHrVwjK7dDw",
)
ACTIONS = ("grab", "dash", "jump", "up", "left", "down", "right")
ALLOWED_DECISIONS = {"same_normalized_layout", "explicit_geometry", "unknown"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
PROMPT_VERSION = "celeste-layout-family-geometry-v2-order-consistency"
PROMPT_TEMPLATE = """You are conservatively matching one Celeste keyboard/action
HUD against 15 validated reference layouts. Gameplay, timers, LiveSplit, chat,
and game UI are irrelevant.

Images 1-5 are the CANDIDATE: its 4x4 contact sheet, then its four quadrants in
reading order. Images 6-20 are REFERENCE contact sheets in this exact order:
{reference_ids}

Choose exactly one decision:
- same_normalized_layout: the candidate has exactly the same normalized HUD
  position, scale, grid geometry, and action-cell ordering as one reference.
  Cosmetic colors/translucency may differ. Return that reference ID.
- explicit_geometry: no exact reference match, but all seven action centers are
  directly legible. Return normalized [x,y] centers in the source frame for
  grab, dash, jump, up, left, down, right, each coordinate in [0,1].
- unknown: anything else, including a missing/frozen/non-keyboard HUD, an
  approximate family resemblance, ambiguous labels, or fewer than seven
  directly legible centers.

Do not guess. Exact means exact normalized geometry, not merely the same HUD
software or visual style. Every field is mandatory. For a reference match,
centers_normalized must be null. For explicit geometry, reference_video_id must
be null. For unknown, both must be null. Return ONLY one compact JSON object:
{{"decision":"same_normalized_layout|explicit_geometry|unknown",
"reference_video_id":"ID or null","confidence":"high|medium|low",
"reason":"short visible evidence",
"centers_normalized":{{"grab":[x,y],"dash":[x,y],"jump":[x,y],
"up":[x,y],"left":[x,y],"down":[x,y],"right":[x,y]}} or null
}}
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}: blank line {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: line {line_number} is not an object")
        rows.append(value)
    return rows


def survey_binding(survey_root: Path, video_id: str) -> dict[str, Any]:
    directory = survey_root / video_id
    survey_path = directory / "survey.json"
    contact_path = directory / "contact-sheet.png"
    completion_path = directory / "survey_complete.json"
    if not all(path.is_file() for path in (survey_path, contact_path, completion_path)):
        raise FileNotFoundError(f"{video_id}: completed survey is missing")
    survey = json.loads(survey_path.read_text())
    completion = json.loads(completion_path.read_text())
    survey_sha256 = sha256_file(survey_path)
    contact_sha256 = sha256_file(contact_path)
    if survey.get("video_id") != video_id or completion.get("video_id") != video_id:
        raise ValueError(f"{video_id}: survey video binding mismatch")
    if survey.get("contact_sheet", {}).get("sha256") != contact_sha256:
        raise ValueError(f"{video_id}: contact-sheet hash mismatch")
    if completion.get("survey_sha256") != survey_sha256:
        raise ValueError(f"{video_id}: survey completion hash mismatch")
    if survey.get("human_reviewed") is not False or survey.get("training_admitted") is not False:
        raise ValueError(f"{video_id}: survey cannot claim review or admission")
    return {
        "video_id": video_id,
        "survey_path": survey_path,
        "survey_sha256": survey_sha256,
        "contact_path": contact_path,
        "contact_sha256": contact_sha256,
    }


def _valid_center(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) and 0 <= item <= 1 for item in value)
    )


def parse_response(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {
            "decision": "unknown",
            "reference_video_id": None,
            "confidence": "low",
            "centers_normalized": None,
            "reason": "model response lacked a JSON object",
            "parse_error": "missing_json_object",
        }
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {
            "decision": "unknown",
            "reference_video_id": None,
            "confidence": "low",
            "centers_normalized": None,
            "reason": "model response contained malformed JSON",
            "parse_error": f"invalid_json:{exc.msg}",
        }
    decision = str(value.get("decision", "")).strip().lower()
    confidence = str(value.get("confidence", "")).strip().lower()
    reference = value.get("reference_video_id")
    centers = value.get("centers_normalized")
    reason = " ".join(str(value.get("reason", "")).split())[:700]
    errors: list[str] = []
    if decision not in ALLOWED_DECISIONS:
        errors.append(f"invalid_decision:{decision}")
        decision = "unknown"
    if confidence not in ALLOWED_CONFIDENCE:
        errors.append(f"invalid_confidence:{confidence}")
        confidence = "low"
    if decision == "same_normalized_layout":
        if reference not in REFERENCE_VIDEO_IDS:
            errors.append("invalid_reference")
            decision = "unknown"
            reference = None
        centers = None
    elif decision == "explicit_geometry":
        if not isinstance(centers, dict) or set(centers) != set(ACTIONS):
            errors.append("incomplete_geometry")
            decision = "unknown"
            centers = None
        elif not all(_valid_center(centers[action]) for action in ACTIONS):
            errors.append("invalid_geometry")
            decision = "unknown"
            centers = None
        reference = None
    else:
        reference = None
        centers = None
    if not reason:
        reason = "model supplied no visible evidence"
        errors.append("missing_reason")
        decision = "unknown"
        reference = None
        centers = None
        confidence = "low"
    result = {
        "decision": decision,
        "reference_video_id": reference,
        "confidence": confidence,
        "centers_normalized": centers,
        "reason": reason,
    }
    if errors:
        result["parse_error"] = ",".join(errors)
    return result


def combine_order_checks(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    geometry_tolerance: float = 0.03,
) -> dict[str, Any]:
    """Require a decision to survive an independent reference reordering."""
    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    confidence = min(
        (str(first["confidence"]), str(second["confidence"])),
        key=confidence_rank.__getitem__,
    )
    if (
        first["decision"] == "same_normalized_layout"
        and second["decision"] == "same_normalized_layout"
        and first["reference_video_id"] == second["reference_video_id"]
    ):
        return {
            "decision": "same_normalized_layout",
            "reference_video_id": first["reference_video_id"],
            "confidence": confidence,
            "centers_normalized": None,
            "reason": (
                f"order A: {first['reason']} | order B: {second['reason']}"
            )[:700],
            "order_consistent": True,
        }
    if first["decision"] == second["decision"] == "explicit_geometry":
        first_centers = first["centers_normalized"]
        second_centers = second["centers_normalized"]
        maximum_delta = max(
            abs(float(first_centers[action][axis]) - float(second_centers[action][axis]))
            for action in ACTIONS
            for axis in (0, 1)
        )
        if maximum_delta <= geometry_tolerance:
            centers = {
                action: [
                    round(
                        (
                            float(first_centers[action][axis])
                            + float(second_centers[action][axis])
                        )
                        / 2,
                        6,
                    )
                    for axis in (0, 1)
                ]
                for action in ACTIONS
            }
            return {
                "decision": "explicit_geometry",
                "reference_video_id": None,
                "confidence": confidence,
                "centers_normalized": centers,
                "reason": (
                    f"order A: {first['reason']} | order B: {second['reason']}"
                )[:700],
                "order_consistent": True,
                "maximum_center_delta": maximum_delta,
            }
    return {
        "decision": "unknown",
        "reference_video_id": None,
        "confidence": "low",
        "centers_normalized": None,
        "reason": (
            "reference-order disagreement: "
            f"A={first['decision']}:{first.get('reference_video_id')}; "
            f"B={second['decision']}:{second.get('reference_video_id')}"
        ),
        "parse_error": "order_inconsistent",
        "order_consistent": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--survey-root", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--max-pixels-side", type=int, default=336)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--target-prediction-rows", type=int, default=630)
    args = parser.parse_args()

    import torch
    import transformers
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    queue_sha256 = sha256_file(args.queue)
    references = [survey_binding(args.survey_root, item) for item in REFERENCE_VIDEO_IDS]
    reference_orders = (tuple(references), tuple(reversed(references)))
    prompts = tuple(
        PROMPT_TEMPLATE.format(
            reference_ids=", ".join(item["video_id"] for item in reference_order)
        )
        for reference_order in reference_orders
    )
    prompt_sha256 = sha256_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "prompts": prompts,
                "combination": "same reference across canonical and reversed orders",
            },
            sort_keys=True,
        )
    )
    reference_order_ids = [
        [item["video_id"] for item in order] for order in reference_orders
    ]
    reference_manifest = [
        {
            "video_id": item["video_id"],
            "survey_sha256": item["survey_sha256"],
            "contact_sha256": item["contact_sha256"],
        }
        for item in references
    ]
    resolved_config = transformers.AutoConfig.from_pretrained(
        args.model, revision=args.revision
    )
    resolved_revision = str(getattr(resolved_config, "_commit_hash", "") or "")
    if resolved_revision != args.revision:
        raise RuntimeError("model revision did not resolve to the requested commit")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for layout-family matching")

    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        min_pixels=224 * 224,
        max_pixels=args.max_pixels_side * args.max_pixels_side,
    )
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.revision,
        config=resolved_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    ).eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(exist_ok=True)
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    completed = {str(row["video_id"]) for row in load_jsonl(args.out)}
    created_at = utc_now()
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        created_at = str(prior.get("created_at") or created_at)
        for field, expected in {
            "queue_sha256": queue_sha256,
            "prompt_sha256": prompt_sha256,
            "resolved_model_revision": resolved_revision,
            "reference_surveys": reference_manifest,
            "max_pixels_side": args.max_pixels_side,
            "reference_orders": reference_order_ids,
        }.items():
            if prior.get(field) != expected:
                raise ValueError(f"resume manifest has stale {field}")

    def update_manifest(state: str, candidate_rows: int) -> None:
        rows = load_jsonl(args.out)
        write_json_atomic(
            manifest_path,
            {
                "schema_version": 1,
                "task": "ai_only_visual_layout_family_and_geometry",
                "state": state,
                "created_at": created_at,
                "updated_at": utc_now(),
                "human_reviewed": False,
                "training_admitted": False,
                "mechanical_full_scan_required": True,
                "queue_sha256": queue_sha256,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_sha256,
                "model": args.model,
                "resolved_model_revision": resolved_revision,
                "max_pixels_side": args.max_pixels_side,
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
                "reference_surveys": reference_manifest,
                "reference_orders": reference_order_ids,
                "candidate_prediction_rows_seen": candidate_rows,
                "results": {
                    "rows": len(rows),
                    "unique_video_ids": len({str(row["video_id"]) for row in rows}),
                    "sha256": sha256_file(args.out),
                },
            },
        )

    update_manifest("running", 0)
    with torch.inference_mode():
        while True:
            candidates = load_jsonl(args.candidate_predictions)
            pending = [
                item
                for item in candidates
                if item.get("full_scan_candidate") is True
                and str(item.get("video_id")) not in completed
            ]
            if not pending:
                update_manifest(
                    "complete" if len(candidates) >= args.target_prediction_rows else "waiting",
                    len(candidates),
                )
                if len(candidates) >= args.target_prediction_rows:
                    return
                time.sleep(args.poll_seconds)
                continue
            for candidate_prediction in pending:
                video_id = str(candidate_prediction["video_id"])
                candidate = survey_binding(args.survey_root, video_id)
                if candidate_prediction.get("survey_sha256") != candidate["survey_sha256"]:
                    raise ValueError(f"{video_id}: candidate prediction survey hash mismatch")
                with Image.open(candidate["contact_path"]) as opened:
                    contact = opened.convert("RGB")
                    width, height = contact.size
                    candidate_images = [
                        contact,
                        contact.crop((0, 0, width // 2, height // 2)),
                        contact.crop((width // 2, 0, width, height // 2)),
                        contact.crop((0, height // 2, width // 2, height)),
                        contact.crop((width // 2, height // 2, width, height)),
                    ]
                    order_checks = []
                    for reference_order, prompt in zip(reference_orders, prompts):
                        reference_images = []
                        for reference in reference_order:
                            with Image.open(reference["contact_path"]) as reference_opened:
                                reference_images.append(reference_opened.convert("RGB"))
                        content: list[dict[str, Any]] = []
                        for index, image in enumerate(candidate_images, 1):
                            content.extend(
                                (
                                    {
                                        "type": "text",
                                        "text": f"Candidate image {index}",
                                    },
                                    {"type": "image", "image": image},
                                )
                            )
                        for index, (reference, image) in enumerate(
                            zip(reference_order, reference_images), 6
                        ):
                            content.extend(
                                (
                                    {
                                        "type": "text",
                                        "text": (
                                            f"Reference image {index}: "
                                            f"{reference['video_id']}"
                                        ),
                                    },
                                    {"type": "image", "image": image},
                                )
                            )
                        content.append({"type": "text", "text": prompt})
                        messages = [{"role": "user", "content": content}]
                        rendered = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        image_inputs, video_inputs = process_vision_info(messages)
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
                        raw = processor.decode(
                            trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        order_checks.append(
                            {
                                "reference_order": [
                                    item["video_id"] for item in reference_order
                                ],
                                "parsed": parse_response(raw),
                                "raw_response": raw[:1600],
                            }
                        )
                parsed = combine_order_checks(
                    order_checks[0]["parsed"], order_checks[1]["parsed"]
                )
                record = {
                    "schema_version": 1,
                    "video_id": video_id,
                    "survey_sha256": candidate["survey_sha256"],
                    "contact_sha256": candidate["contact_sha256"],
                    "candidate_class": candidate_prediction.get("class"),
                    "candidate_confidence": candidate_prediction.get("confidence"),
                    **parsed,
                    "reference_survey": next(
                        (
                            item
                            for item in reference_manifest
                            if item["video_id"] == parsed.get("reference_video_id")
                        ),
                        None,
                    ),
                    "human_reviewed": False,
                    "training_admitted": False,
                    "mechanical_full_scan_required": True,
                    "queue_sha256": queue_sha256,
                    "model": args.model,
                    "resolved_model_revision": resolved_revision,
                    "prompt_version": PROMPT_VERSION,
                    "prompt_sha256": prompt_sha256,
                    "classified_at": utc_now(),
                    "order_checks": order_checks,
                }
                with args.out.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                completed.add(video_id)
                update_manifest("running", len(candidates))
                print(
                    f"{video_id} {record['decision']} "
                    f"reference={record['reference_video_id']} "
                    f"confidence={record['confidence']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
