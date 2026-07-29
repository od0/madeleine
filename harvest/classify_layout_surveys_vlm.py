"""Resumably nominate full-cell scans from 16-frame layout contact sheets.

This local VLM pass is a scheduling aid only.  Its strongest positive still
requires a full physical-cell activity scan, semantic mapping, gameplay-range
review, and offset calibration before labels can approach admission.
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


PROMPT_VERSION = "celeste-layout-stability-v1-contact-quadrants"
PROMPT = """You are reviewing one 4x4 contact sheet made from 16 evenly spaced,
exact source frames across a Celeste speedrun video. The first image is the
whole contact sheet. The next four images are its quadrants in reading order,
each containing four larger samples.

Judge the keyboard/action HUD only. Ignore LiveSplit, timers, chat, webcam,
emotes, game UI, and controller diagrams.

Choose exactly one class:
- stable_active_candidate: the same keyboard or labeled action-cell HUD is
  visible with stable geometry in at least 12 of 16 samples, AND visible cell
  fills/highlights differ across samples in a way consistent with real inputs.
- stable_activity_uncertain: a keyboard/action HUD is stable in at least 12 of
  16 samples, but the images are too small, the highlight is ambiguous, or it
  might be frozen. This class still goes to a mechanical full-cell scan.
- unstable_or_missing: the candidate HUD is absent, moved, obscured, or changes
  layout in more than four samples.
- non_keyboard: the apparent panel is not a keyboard/action HUD (for example a
  gamepad/controller, timer, splits, chat, or game UI).

Do not infer activity merely because gameplay changes. Compare the HUD cells
themselves. A single highlighted cell that never changes is not proof of an
active label source. Describe the HUD location and the specific cross-sample
change or ambiguity.

Return ONLY one compact JSON object:
{"class":"stable_active_candidate|stable_activity_uncertain|unstable_or_missing|non_keyboard","confidence":"high|medium|low","evidence":"short visual description"}
"""

ALLOWED_CLASSES = {
    "stable_active_candidate",
    "stable_activity_uncertain",
    "unstable_or_missing",
    "non_keyboard",
}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
MANIFEST_SCHEMA_VERSION = 2
SURVEY_FORMAT_VERSION = "madeleine.wild-layout-survey.v1"
SURVEY_PUBLICATION_VERSION = "madeleine.wild-layout-survey-publication.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_response(raw: str) -> dict[str, str]:
    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if not match:
        return {
            "class": "stable_activity_uncertain",
            "confidence": "low",
            "evidence": "model response lacked a JSON object",
            "parse_error": "missing_json_object",
        }
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {
            "class": "stable_activity_uncertain",
            "confidence": "low",
            "evidence": "model response contained malformed JSON",
            "parse_error": f"invalid_json:{exc.msg}",
        }
    label = str(value.get("class", "")).strip().lower()
    confidence = str(value.get("confidence", "")).strip().lower()
    evidence = " ".join(str(value.get("evidence", "")).split())[:500]
    errors = []
    if label not in ALLOWED_CLASSES:
        errors.append(f"invalid_class:{label}")
        label = "stable_activity_uncertain"
    if confidence not in ALLOWED_CONFIDENCE:
        errors.append(f"invalid_confidence:{confidence}")
        confidence = "low"
    if not evidence:
        errors.append("missing_evidence")
        evidence = "model supplied no visual evidence"
    result = {"class": label, "confidence": confidence, "evidence": evidence}
    if errors:
        result["parse_error"] = ",".join(errors)
    return result


def load_queue_rows(queue: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(queue.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"queue line {line_number} is not valid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"queue line {line_number} is not an object")
        video_id = str(row.get("video_id", ""))
        if not _SAFE_ID.fullmatch(video_id):
            raise ValueError(f"queue line {line_number} has an unsafe video_id")
        if video_id in seen:
            raise ValueError(f"queue contains duplicate video_id {video_id}")
        if row.get("human_reviewed") is not False:
            raise ValueError(f"queue row {video_id} must remain human_reviewed=false")
        if row.get("training_admitted") not in (None, False):
            raise ValueError(f"queue row {video_id} is unexpectedly training-admitted")
        seen.add(video_id)
        rows.append(row)
    return rows


def available_surveys(
    queue: Path,
    survey_root: Path,
    *,
    available_only: bool,
) -> list[dict[str, Any]]:
    rows = load_queue_rows(queue)
    output = []
    for row in rows:
        video_id = str(row["video_id"])
        directory = survey_root / video_id
        manifest_path = directory / "survey.json"
        contact_path = directory / "contact-sheet.png"
        completion_path = directory / "survey_complete.json"
        if not all(path.is_file() for path in (manifest_path, contact_path, completion_path)):
            if available_only:
                continue
            raise FileNotFoundError(f"{video_id}: completed survey is missing")
        manifest = json.loads(manifest_path.read_text())
        completion = json.loads(completion_path.read_text())
        if manifest.get("format_version") != SURVEY_FORMAT_VERSION:
            raise ValueError(f"{video_id}: unsupported survey format")
        if completion.get("format_version") != SURVEY_PUBLICATION_VERSION:
            raise ValueError(f"{video_id}: unsupported survey completion format")
        if manifest.get("video_id") != video_id or completion.get("video_id") != video_id:
            raise ValueError(f"{video_id}: survey binding mismatch")
        if (
            manifest.get("human_reviewed") is not False
            or manifest.get("training_admitted") is not False
            or completion.get("human_reviewed") is not False
            or completion.get("training_admitted") is not False
        ):
            raise ValueError(f"{video_id}: survey cannot claim review or admission")
        contact_sha256 = sha256_file(contact_path)
        survey_sha256 = sha256_file(manifest_path)
        if manifest.get("contact_sheet", {}).get("sha256") != contact_sha256:
            raise ValueError(f"{video_id}: contact sheet hash mismatch")
        if completion.get("survey_sha256") != survey_sha256:
            raise ValueError(f"{video_id}: completion marker survey hash mismatch")
        if completion.get("source_sha256") != manifest.get("source", {}).get("sha256"):
            raise ValueError(f"{video_id}: completion marker source hash mismatch")
        output.append({
            **row,
            "survey_path": str(manifest_path),
            "survey_sha256": survey_sha256,
            "contact_path": str(contact_path),
            "contact_sha256": contact_sha256,
        })
    return output


def load_existing_predictions(
    path: Path,
    available_rows: list[dict[str, Any]],
    *,
    model: str,
    resolved_model_revision: str,
    prompt_sha256: str,
    max_new_tokens: int = 160,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    available = {str(row["video_id"]): row for row in available_rows}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f"prediction line {line_number} is blank")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"prediction line {line_number} is not valid JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"prediction line {line_number} is not an object")
        video_id = str(record.get("video_id", ""))
        if not _SAFE_ID.fullmatch(video_id):
            raise ValueError(f"prediction line {line_number} has an unsafe video_id")
        if video_id in seen:
            raise ValueError(f"predictions contain duplicate video_id {video_id}")
        if video_id not in available:
            raise ValueError(
                f"prediction {video_id} is outside the queue or its survey is unavailable"
            )
        survey = available[video_id]
        expected = {
            "survey_sha256": survey["survey_sha256"],
            "contact_sha256": survey["contact_sha256"],
            "model": model,
            "resolved_model_revision": resolved_model_revision,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": prompt_sha256,
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise ValueError(
                    f"prediction {video_id} has stale or mismatched {field}"
                )
        recorded_max_new_tokens = record.get("max_new_tokens")
        if recorded_max_new_tokens is None:
            # v1 records predated this field and always used the default.
            recorded_max_new_tokens = 160
        if recorded_max_new_tokens != max_new_tokens:
            raise ValueError(
                f"prediction {video_id} has stale or mismatched max_new_tokens"
            )
        if record.get("class") not in ALLOWED_CLASSES:
            raise ValueError(f"prediction {video_id} has an invalid class")
        if record.get("confidence") not in ALLOWED_CONFIDENCE:
            raise ValueError(f"prediction {video_id} has invalid confidence")
        if (
            record.get("human_reviewed") is not False
            or record.get("training_admitted") is not False
        ):
            raise ValueError(f"prediction {video_id} cannot claim review or admission")
        expected_candidate = record["class"] in {
            "stable_active_candidate",
            "stable_activity_uncertain",
        }
        if record.get("full_scan_candidate") is not expected_candidate:
            raise ValueError(f"prediction {video_id} has inconsistent scan candidacy")
        seen.add(video_id)
        records.append(record)
    return records


def _sha256_line_prefix(path: Path, lines: int) -> str:
    digest = hashlib.sha256()
    consumed = 0
    if not path.exists():
        if lines:
            raise ValueError("prediction output is missing")
        return digest.hexdigest()
    with path.open("rb") as handle:
        for raw_line in handle:
            if consumed == lines:
                break
            digest.update(raw_line)
            consumed += 1
    if consumed != lines:
        raise ValueError("prediction output is shorter than its manifest")
    return digest.hexdigest()


def validate_resume_manifest(
    manifest_path: Path,
    expected: dict[str, Any],
    predictions_path: Path,
    prediction_rows: int,
) -> str:
    if not manifest_path.exists():
        if prediction_rows:
            raise ValueError("prediction output exists without its manifest")
        return utc_now()
    manifest = json.loads(manifest_path.read_text())
    for field in (
        "task",
        "classification_is_human_review",
        "training_admission",
        "queue_sha256",
        "prompt_version",
        "prompt_sha256",
        "model",
        "resolved_model_revision",
        "max_new_tokens",
    ):
        actual = manifest.get(field)
        if field == "max_new_tokens" and actual is None:
            # Legacy v1 manifests did not record the fixed default.
            actual = 160
        if actual != expected.get(field):
            raise ValueError(f"resume manifest has stale or mismatched {field}")
    prior = manifest.get("predictions")
    if prior is not None:
        if not isinstance(prior, dict):
            raise ValueError("resume manifest predictions field is malformed")
        prior_rows = prior.get("rows")
        prior_sha256 = prior.get("sha256")
        if not isinstance(prior_rows, int) or prior_rows < 0:
            raise ValueError("resume manifest prediction row count is invalid")
        if prior_rows > prediction_rows:
            raise ValueError("prediction output is shorter than its manifest")
        if prior_sha256 != _sha256_line_prefix(predictions_path, prior_rows):
            raise ValueError("prediction output prefix differs from its manifest")
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("resume manifest is missing created_at")
    return created_at


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--survey-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    import torch
    import transformers

    resolved_config = transformers.AutoConfig.from_pretrained(
        args.model, revision=args.revision
    )
    revision = str(getattr(resolved_config, "_commit_hash", "") or "")
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise RuntimeError(
            "model revision did not resolve to an immutable commit; "
            "use a Hugging Face commit revision"
        )

    available = available_surveys(
        args.queue, args.survey_root, available_only=args.available_only
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prompt_sha256 = sha256_text(PROMPT)
    existing = load_existing_predictions(
        args.out,
        available,
        model=args.model,
        resolved_model_revision=revision,
        prompt_sha256=prompt_sha256,
        max_new_tokens=args.max_new_tokens,
    )
    done = {str(record["video_id"]) for record in existing}
    rows = [row for row in available if row["video_id"] not in done]

    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest_identity = {
        "task": "celeste_layout_stability_nomination",
        "classification_is_human_review": False,
        "training_admission": False,
        "queue_sha256": sha256_file(args.queue),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "model": args.model,
        "resolved_model_revision": revision,
        "max_new_tokens": args.max_new_tokens,
    }
    created_at = validate_resume_manifest(
        manifest_path,
        manifest_identity,
        args.out,
        len(existing),
    )
    args.out.touch(exist_ok=True)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        **manifest_identity,
        "created_at": created_at,
        "updated_at": utc_now(),
        "state": "running" if rows else "complete",
        "queue": str(args.queue.resolve()),
        "survey_root": str(args.survey_root.resolve()),
        "prompt": PROMPT,
        "classes": sorted(ALLOWED_CLASSES),
        "full_scan_classes": [
            "stable_active_candidate",
            "stable_activity_uncertain",
        ],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "available_rows": len(available),
        "already_complete_on_start": len(done),
        "predictions": {
            "rows": len(existing),
            "unique_video_ids": len(existing),
            "sha256": sha256_file(args.out),
        },
    }
    write_json_atomic(manifest_path, manifest)

    if not rows:
        manifest["completed_at"] = utc_now()
        manifest["newly_completed"] = 0
        manifest["class_counts_this_run"] = {
            label: 0 for label in ALLOWED_CLASSES
        }
        manifest["class_counts_total"] = {
            label: sum(record["class"] == label for record in existing)
            for label in ALLOWED_CLASSES
        }
        write_json_atomic(manifest_path, manifest)
        print(f"[0/0] no pending surveys; validated {len(existing)} predictions")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VLM survey triage")
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=revision,
        min_pixels=224 * 224,
        max_pixels=448 * 448,
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
    if getattr(model.config, "_commit_hash", None) != revision:
        raise RuntimeError("loaded model revision differs from resolved configuration")
    manifest["cuda_device"] = torch.cuda.get_device_name(0)
    write_json_atomic(manifest_path, manifest)

    started = time.monotonic()
    counts = {label: 0 for label in ALLOWED_CLASSES}
    new_records: list[dict[str, Any]] = []
    with args.out.open("a") as handle, torch.inference_mode():
        for index, row in enumerate(rows, 1):
            with Image.open(row["contact_path"]) as opened:
                contact = opened.convert("RGB")
                width, height = contact.size
                quadrants = [
                    contact.crop((0, 0, width // 2, height // 2)),
                    contact.crop((width // 2, 0, width, height // 2)),
                    contact.crop((0, height // 2, width // 2, height)),
                    contact.crop((width // 2, height // 2, width, height)),
                ]
                content: list[dict[str, Any]] = [
                    {"type": "image", "image": contact}
                ]
                content.extend({"type": "image", "image": image} for image in quadrants)
                content.append({"type": "text", "text": PROMPT})
                messages = [[{"role": "user", "content": content}]]
                text = processor.apply_chat_template(
                    messages[0], tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
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
                trimmed = generated[0][len(inputs.input_ids[0]):]
                raw = processor.decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            parsed = parse_response(raw)
            counts[parsed["class"]] += 1
            record = {
                "schema_version": 1,
                "video_id": row["video_id"],
                "source": row.get("source"),
                "nominal_hours": row.get("nominal_hours"),
                "survey_sha256": row["survey_sha256"],
                "contact_sha256": row["contact_sha256"],
                "class": parsed["class"],
                "confidence": parsed["confidence"],
                "evidence": parsed["evidence"],
                "parse_error": parsed.get("parse_error"),
                "full_scan_candidate": parsed["class"] in {
                    "stable_active_candidate", "stable_activity_uncertain"
                },
                "human_reviewed": False,
                "training_admitted": False,
                "model": args.model,
                "resolved_model_revision": revision,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": sha256_text(PROMPT),
                "max_new_tokens": args.max_new_tokens,
                "classified_at": utc_now(),
                "raw_response": raw[:1000],
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            new_records.append(record)
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = index / elapsed
            print(
                f"[{index}/{len(rows)}] {row['video_id']} {parsed['class']} "
                f"rate={rate:.3f}/s counts={counts}",
                flush=True,
            )
    all_records = [*existing, *new_records]
    manifest["state"] = "complete"
    manifest["updated_at"] = utc_now()
    manifest["completed_at"] = utc_now()
    manifest["newly_completed"] = len(rows)
    manifest["class_counts_this_run"] = counts
    manifest["class_counts_total"] = {
        label: sum(record["class"] == label for record in all_records)
        for label in ALLOWED_CLASSES
    }
    manifest["predictions"] = {
        "rows": len(all_records),
        "unique_video_ids": len(all_records),
        "sha256": sha256_file(args.out),
    }
    write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
