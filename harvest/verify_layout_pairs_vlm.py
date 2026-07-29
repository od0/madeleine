"""Order-invariant pairwise VLM verification of proposed HUD-layout matches.

This is a conservative machine verification stage.  It never marks human
review or training admission, and a positive result still requires the
mechanical full-video scan before any layout transfer is usable.
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
from typing import Any

from harvest.classify_layout_families_vlm import sha256_file, survey_binding


PROMPT_VERSION = "celeste-layout-pair-v2-specific-evidence"
PROMPT = """Decide whether two Celeste action-HUD surveys have EXACTLY the same
normalized layout. Ignore gameplay, timers, splits, chat, colors, and opacity.

Each survey has a full 4x4 temporal contact sheet and an enlarged 2x2 montage
of the same normalized action-HUD region. The region coordinates are
{region}. The first two images belong to survey A; the last two belong to
survey B.

Return same only when the HUD has the same position within the source frame,
same scale, same cell borders/centers, and same seven-action ordering. A shared
HUD style or approximately similar grid is not enough. Return different for a
visible geometric mismatch. Return unknown if either HUD is unclear, absent in
several samples, or the evidence is insufficient. Do not guess.

The evidence must name at least two visible geometric facts (for example HUD
corner, row/column count, cell border alignment, or a specific action-cell
position). Never repeat the phrase "short visible comparison".

Return ONLY one compact JSON object:
{{"decision":"same|different|unknown","confidence":"high|medium|low",
"evidence":"short visible comparison"}}
"""
ALLOWED_DECISIONS = {"same", "different", "unknown"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}: blank line {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: line {line_number} is not an object")
        rows.append(value)
    return rows


def parse_response(raw: str) -> dict[str, str]:
    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if not match:
        return {
            "decision": "unknown",
            "confidence": "low",
            "evidence": "model response lacked a JSON object",
            "parse_error": "missing_json_object",
        }
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {
            "decision": "unknown",
            "confidence": "low",
            "evidence": "model response contained malformed JSON",
            "parse_error": f"invalid_json:{exc.msg}",
        }
    decision = str(value.get("decision", "")).strip().lower()
    confidence = str(value.get("confidence", "")).strip().lower()
    evidence = " ".join(str(value.get("evidence", "")).split())[:700]
    errors = []
    if decision not in ALLOWED_DECISIONS:
        errors.append(f"invalid_decision:{decision}")
        decision = "unknown"
    if confidence not in ALLOWED_CONFIDENCE:
        errors.append(f"invalid_confidence:{confidence}")
        confidence = "low"
    if len(evidence.split()) < 8 or evidence.lower() == "short visible comparison":
        errors.append("missing_specific_evidence")
        evidence = "model supplied no visible comparison"
        decision = "unknown"
        confidence = "low"
    result = {"decision": decision, "confidence": confidence, "evidence": evidence}
    if errors:
        result["parse_error"] = ",".join(errors)
    return result


def combine_order_checks(first: dict[str, str], second: dict[str, str]) -> dict[str, Any]:
    rank = {"low": 0, "medium": 1, "high": 2}
    confidence = min(
        (str(first["confidence"]), str(second["confidence"])),
        key=rank.__getitem__,
    )
    if first["decision"] == second["decision"] == "same" and rank[confidence] >= 1:
        return {
            "decision": "same",
            "confidence": confidence,
            "order_consistent": True,
            "pair_confirmed": True,
            "reason": f"order A: {first['evidence']} | order B: {second['evidence']}"[:1000],
        }
    if first["decision"] == second["decision"] == "different":
        return {
            "decision": "different",
            "confidence": confidence,
            "order_consistent": True,
            "pair_confirmed": False,
            "reason": f"order A: {first['evidence']} | order B: {second['evidence']}"[:1000],
        }
    return {
        "decision": "unknown",
        "confidence": "low",
        "order_consistent": first["decision"] == second["decision"],
        "pair_confirmed": False,
        "reason": (
            f"order-invariant verification failed: "
            f"A={first['decision']}:{first['confidence']}; "
            f"B={second['decision']}:{second['confidence']}"
        ),
    }


def _survey_images(
    survey_root: Path,
    video_id: str,
    region: tuple[float, float, float, float],
) -> tuple[dict[str, Any], Any, Any]:
    from PIL import Image

    binding = survey_binding(survey_root, video_id)
    survey = json.loads(binding["survey_path"].read_text())
    frames = survey.get("frames")
    if not isinstance(frames, list) or len(frames) != 16:
        raise ValueError(f"{video_id}: expected 16 survey frames")
    selected = []
    for index in (0, 5, 10, 15):
        frame = frames[index]
        path = binding["survey_path"].parent / str(frame["path"])
        if not path.is_file() or sha256_file(path) != frame.get("sha256"):
            raise ValueError(f"{video_id}: survey frame hash mismatch")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            x0, y0, x1, y1 = region
            crop = image.crop((
                round(x0 * width), round(y0 * height),
                round(x1 * width), round(y1 * height),
            ))
            selected.append(crop.resize((336, 224), Image.Resampling.LANCZOS))
    montage = Image.new("RGB", (672, 448), "black")
    for index, image in enumerate(selected):
        montage.paste(image, ((index % 2) * 336, (index // 2) * 224))
    with Image.open(binding["contact_path"]) as opened:
        contact = opened.convert("RGB")
    return binding, contact, montage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--survey-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-pixels-side", type=int, default=448)
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--min-score", type=float, default=0.32)
    parser.add_argument("--min-margin", type=float, default=0.14)
    args = parser.parse_args()

    import torch
    import transformers
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proposal_sha256 = sha256_file(args.proposals)
    proposals = [
        row for row in load_jsonl(args.proposals)
        if row.get("proposal_passed") is True
        and float(row.get("top_score", 0)) >= args.min_score
        and float(row.get("margin", 0)) >= args.min_margin
    ][:args.max_pairs]
    resolved_config = transformers.AutoConfig.from_pretrained(
        args.model, revision=args.revision
    )
    resolved_revision = str(getattr(resolved_config, "_commit_hash", "") or "")
    if resolved_revision != args.revision:
        raise RuntimeError("model revision did not resolve to requested commit")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pairwise verification")
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

    prompt_sha256 = sha256_text(PROMPT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(exist_ok=True)
    existing = load_jsonl(args.out) if args.out.stat().st_size else []
    completed = {(row["video_id"], row["reference_video_id"]) for row in existing}
    with torch.inference_mode(), args.out.open("a") as handle:
        for proposal in proposals:
            video_id = str(proposal["video_id"])
            reference_id = str(proposal["proposed_reference_video_id"])
            if (video_id, reference_id) in completed:
                continue
            top = proposal["ranking"][0]
            region = tuple(float(value) for value in top["region_normalized"])
            candidate, candidate_contact, candidate_montage = _survey_images(
                args.survey_root, video_id, region
            )
            reference, reference_contact, reference_montage = _survey_images(
                args.survey_root, reference_id, region
            )
            expected = {
                "candidate_survey_sha256": candidate["survey_sha256"],
                "candidate_contact_sha256": candidate["contact_sha256"],
                "reference_survey_sha256": reference["survey_sha256"],
                "reference_contact_sha256": reference["contact_sha256"],
            }
            for field, value in expected.items():
                recorded = proposal.get(field) if field.startswith("candidate") else top.get(field)
                if recorded != value:
                    raise ValueError(f"{video_id}: proposal has stale {field}")

            order_checks = []
            for label, images in (
                ("candidate-first", (candidate_contact, candidate_montage,
                                     reference_contact, reference_montage)),
                ("reference-first", (reference_contact, reference_montage,
                                     candidate_contact, candidate_montage)),
            ):
                content: list[dict[str, Any]] = []
                for index, image in enumerate(images, 1):
                    content.extend((
                        {"type": "text", "text": f"Survey {'A' if index <= 2 else 'B'} image {index}"},
                        {"type": "image", "image": image},
                    ))
                content.append({
                    "type": "text",
                    "text": PROMPT.format(region=json.dumps(region)),
                })
                messages = [{"role": "user", "content": content}]
                rendered = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[rendered], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt",
                ).to("cuda")
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
                )
                trimmed = generated[0][len(inputs.input_ids[0]):]
                raw = processor.decode(
                    trimmed, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                order_checks.append({
                    "image_order": label,
                    "parsed": parse_response(raw),
                    "raw_response": raw[:1600],
                })
            combined = combine_order_checks(
                order_checks[0]["parsed"], order_checks[1]["parsed"]
            )
            record = {
                "schema_version": 1,
                "video_id": video_id,
                "reference_video_id": reference_id,
                **expected,
                "reference_layout_sha256": top["reference_layout_sha256"],
                "proposal_sha256": proposal_sha256,
                "proposal_algorithm_version": proposal["algorithm_version"],
                "proposal_score": proposal["top_score"],
                "proposal_margin": proposal["margin"],
                "region_normalized": list(region),
                **combined,
                "human_reviewed": False,
                "training_admitted": False,
                "mechanical_full_scan_required": True,
                "model": args.model,
                "resolved_model_revision": resolved_revision,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_sha256,
                "verified_at": utc_now(),
                "order_checks": order_checks,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(
                f"{video_id} vs {reference_id}: {record['decision']} "
                f"confirmed={record['pair_confirmed']}", flush=True
            )
    rows = load_jsonl(args.out)
    write_json_atomic(args.out.with_suffix(args.out.suffix + ".manifest.json"), {
        "schema_version": 1,
        "task": "order_invariant_pairwise_layout_verification",
        "state": "complete",
        "created_at": utc_now(),
        "human_reviewed": False,
        "training_admitted": False,
        "mechanical_full_scan_required": True,
        "proposal_sha256": proposal_sha256,
        "model": args.model,
        "resolved_model_revision": resolved_revision,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "thresholds": {
            "min_score": args.min_score,
            "min_margin": args.min_margin,
            "max_pairs": args.max_pairs,
        },
        "results": {
            "rows": len(rows),
            "confirmed": sum(row["pair_confirmed"] is True for row in rows),
            "sha256": sha256_file(args.out),
        },
    })


if __name__ == "__main__":
    main()
