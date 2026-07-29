"""Resumably triage saved speedrun probe frames with a local vision model.

This is a *nomination* stage, not human review.  It deliberately separates
``target_action_hud`` from ``non_target`` and ``uncertain`` and sends both the
positive and uncertain classes to the downstream review queue.  A malformed
model response is therefore fail-closed as ``uncertain`` rather than silently
becoming a negative.

The output JSONL is append-only and one row is flushed after every batch.  A
sidecar manifest records the exact prompt hash, model identifier and resolved
model revision.  Restarting with the same output skips already written video
IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROMPT_VERSION = "celeste-action-hud-triage-v4-crops-tiles"
PROMPT = """You are triaging ONE saved frame from a Celeste speedrun video.
The FIRST image is the full video frame. Later images are enlarged views:
first any regions selected by a noisy geometric detector, then (when present)
four fixed full-frame quadrants in reading order. Most detector crops are false
positives such as scenery, the speedrun timer, or LiveSplit. The fixed
quadrants exist so a small real HUD missed by the detector remains legible.

Choose exactly one class:
- target_action_hud: at least one supplied image DIRECTLY AND VISIBLY contains a keyboard/input/action display that can plausibly reveal the player's actions over time. Examples include labeled cells such as Jump/Dash/Grab/Left/Right/Up/Down, a recognizable grid of keyboard keys that lights up, or an on-screen keyboard graphic. It can be opaque or translucent and can sit anywhere in the frame.
- non_target: ordinary gameplay with no such display, or only a timer, LiveSplit chapter/split list, chat, webcam, controller/gamepad visualization, subtitles, or game UI.
- uncertain: reserve this for a menu/loading/cutscene/non-gameplay frame, or for a SPECIFIC visible panel that has recognizable key-like structure but is too small/blurred/occluded to decide. Uncertain must be rare.

Important:
- Do NOT call a timer or LiveSplit splits an action HUD.
- Do NOT call a gamepad/controller diagram a target; this project is looking for keyboard/action HUDs.
- Celeste's pixel-art tiles, spikes, character, berries, dialog boxes and inventory-like game art are NOT evidence of keyboard inputs. Never infer an action HUD merely because this is a Celeste speedrun.
- To choose target_action_hud, name the exact visible cue and where it is. If you cannot point to a literal keyboard/key grid/action label, do not choose target_action_hud.
- Most frames are non_target. If you do not see direct evidence of an input display, choose non_target. Do not choose uncertain merely because an input display could theoretically be hidden or because a detector supplied irrelevant crops.
- Before answering, inspect every supplied image and distinguish each candidate crop as action/keyboard HUD, timer/splits/text, game scenery/UI, or genuinely ambiguous. Base the final class on what is actually visible.

Return ONLY one compact JSON object, with no markdown:
{"class":"target_action_hud|non_target|uncertain","confidence":"high|medium|low","evidence":"short visual description"}
"""

ALLOWED_CLASSES = {"target_action_hud", "non_target", "uncertain"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_rows(
    scan_path: Path,
    frames_dir: Path,
    fallback_frames_dir: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in scan_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        direct_frame = row.get("frame_path")
        if direct_frame:
            frame = Path(direct_frame)
            if not frame.is_absolute():
                frame = scan_path.parent / frame
        else:
            frame = frames_dir / f"{row['video_id']}.png"
        if not frame.is_file() and fallback_frames_dir is not None:
            fallback = fallback_frames_dir / f"{row['video_id']}.png"
            if fallback.is_file():
                frame = fallback
        if row.get("error") is not None and not frame.is_file():
            continue
        if not frame.is_file():
            raise FileNotFoundError(f"successful probe has no frame: {frame}")
        rows.append({**row, "frame_path": str(frame.resolve())})
    rows.sort(key=lambda row: row["video_id"])
    return rows


def load_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if isinstance(value, dict):
            value = value.get("video_ids", list(value))
        return {str(item) for item in value}
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def parse_response(raw: str) -> dict[str, str]:
    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if not match:
        return {
            "class": "uncertain",
            "confidence": "low",
            "evidence": "model response was not a JSON object",
            "parse_error": "missing_json_object",
        }
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {
            "class": "uncertain",
            "confidence": "low",
            "evidence": "model response contained malformed JSON",
            "parse_error": f"invalid_json:{exc.msg}",
        }
    label = str(value.get("class", "")).strip().lower()
    confidence = str(value.get("confidence", "")).strip().lower()
    evidence = " ".join(str(value.get("evidence", "")).split())[:300]
    errors = []
    if label not in ALLOWED_CLASSES:
        errors.append(f"invalid_class:{label}")
        label = "uncertain"
    if confidence not in ALLOWED_CONFIDENCE:
        errors.append(f"invalid_confidence:{confidence}")
        confidence = "low"
    if not evidence:
        errors.append("missing_evidence")
        evidence = "model supplied no visual evidence"
    parsed = {"class": label, "confidence": confidence, "evidence": evidence}
    if errors:
        parsed["parse_error"] = ",".join(errors)
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--fallback-frames-dir", type=Path)
    ap.add_argument("--crops-dir", type=Path)
    ap.add_argument("--max-crops", type=int, default=4)
    ap.add_argument(
        "--auto-tiles",
        action="store_true",
        help="append four overlapping full-frame quadrants after detector crops",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--ids-file", type=Path)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument(
        "--classical-uncertain-score",
        type=float,
        default=0.0,
        help=(
            "if positive, convert VLM non_target rows at or above this legacy "
            "geometric score into uncertain review nominees"
        ),
    )
    ap.add_argument(
        "--classical-input-hud-uncertain",
        action="store_true",
        help=(
            "convert VLM non_target rows nominated by the newer temporal "
            "has_input_hud detector into uncertain review candidates"
        ),
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # Heavy imports stay below argument parsing so --help works without the GPU
    # environment installed.
    import torch
    import transformers
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the production VLM triage")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    wanted = load_ids(args.ids_file)
    rows = load_rows(args.scan, args.frames_dir, args.fallback_frames_dir)
    if wanted is not None:
        rows = [row for row in rows if row["video_id"] in wanted]
        missing = wanted - {row["video_id"] for row in rows}
        if missing:
            raise ValueError(f"requested IDs lack successful frames: {sorted(missing)[:10]}")
    if args.limit:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["video_id"])
    rows = [row for row in rows if row["video_id"] not in done]

    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=224 * 224,
        # Capping every full frame/crop bounds attention memory even for
        # unusually large source screenshots. Enlarged crops preserve the
        # small HUD text that would otherwise motivate a larger full frame.
        max_pixels=448 * 448,
    )
    # Qwen is decoder-only. Batched generation must left-pad so that every
    # sequence begins decoding after its own prompt rather than after padding.
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    revision = getattr(model.config, "_commit_hash", None)
    manifest = {
        "schema_version": 1,
        "task": "celeste_probe_frame_action_hud_triage",
        "created_at": utc_now(),
        "classification_is_human_review": False,
        "classes": sorted(ALLOWED_CLASSES),
        "review_queue_classes": ["target_action_hud", "uncertain"],
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256_text(PROMPT),
        "prompt": PROMPT,
        "model": args.model,
        "resolved_model_revision": revision,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "classical_uncertain_score": args.classical_uncertain_score,
        "classical_input_hud_uncertain": args.classical_input_hud_uncertain,
        "scan": str(args.scan.resolve()),
        "scan_sha256": sha256_file(args.scan),
        "frames_dir": str(args.frames_dir.resolve()),
        "fallback_frames_dir": (
            str(args.fallback_frames_dir.resolve())
            if args.fallback_frames_dir is not None
            else None
        ),
        "crops_dir": (
            str(args.crops_dir.resolve()) if args.crops_dir is not None else None
        ),
        "max_crops": args.max_crops,
        "auto_tiles": args.auto_tiles,
        "selected_rows": len(rows) + len(done),
        "already_complete_on_start": len(done),
        "pid": os.getpid(),
    }
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    started = time.monotonic()
    counts = {label: 0 for label in ALLOWED_CLASSES}
    with args.out.open("a") as fh, torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset : offset + args.batch_size]
            from PIL import Image

            crop_paths_by_row: list[list[str]] = []
            auto_tile_names_by_row: list[list[str]] = []
            messages = []
            for row in batch:
                crop_paths: list[str] = []
                if row.get("crop_paths"):
                    for raw_path in row["crop_paths"][: args.max_crops]:
                        crop_path = Path(raw_path)
                        if not crop_path.is_absolute():
                            crop_path = args.scan.parent / crop_path
                        if crop_path.is_file():
                            crop_paths.append(str(crop_path.resolve()))
                elif args.crops_dir is not None:
                    for crop_name in row.get("crops", [])[: args.max_crops]:
                        crop_path = args.crops_dir / crop_name
                        if crop_path.is_file():
                            crop_paths.append(str(crop_path.resolve()))
                crop_paths_by_row.append(crop_paths)
                content = [{"type": "image", "image": row["frame_path"]}]
                content.extend(
                    {"type": "image", "image": crop_path}
                    for crop_path in crop_paths
                )
                auto_tile_names: list[str] = []
                if args.auto_tiles:
                    with Image.open(row["frame_path"]) as source_image:
                        source_image = source_image.convert("RGB")
                        width, height = source_image.size
                        # Ten-percent overlap keeps HUDs on a quadrant boundary
                        # wholly visible in at least one tile.
                        x_mid, y_mid = width // 2, height // 2
                        x_overlap = max(1, width // 20)
                        y_overlap = max(1, height // 20)
                        boxes = [
                            (0, 0, min(width, x_mid + x_overlap), min(height, y_mid + y_overlap)),
                            (max(0, x_mid - x_overlap), 0, width, min(height, y_mid + y_overlap)),
                            (0, max(0, y_mid - y_overlap), min(width, x_mid + x_overlap), height),
                            (max(0, x_mid - x_overlap), max(0, y_mid - y_overlap), width, height),
                        ]
                        for index, box in enumerate(boxes):
                            content.append(
                                {"type": "image", "image": source_image.crop(box)}
                            )
                            auto_tile_names.append(f"quadrant_{index}:{box}")
                auto_tile_names_by_row.append(auto_tile_names)
                content.append({"type": "text", "text": PROMPT})
                messages.append([{"role": "user", "content": content}])
            texts = [
                processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                for message in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=texts,
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
            trimmed = [
                output[len(input_ids) :]
                for input_ids, output in zip(inputs.input_ids, generated)
            ]
            decoded = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for row, crop_paths, auto_tile_names, raw in zip(
                batch, crop_paths_by_row, auto_tile_names_by_row, decoded
            ):
                parsed = parse_response(raw)
                vlm_class = parsed["class"]
                calibration_reason = None
                classical_score = row.get("score")
                score_nominated = (
                    args.classical_uncertain_score > 0
                    and isinstance(classical_score, (int, float))
                    and classical_score >= args.classical_uncertain_score
                )
                temporal_nominated = (
                    args.classical_input_hud_uncertain
                    and row.get("has_input_hud") is True
                )
                if vlm_class == "non_target" and (
                    score_nominated or temporal_nominated
                ):
                    parsed["class"] = "uncertain"
                    parsed["confidence"] = "low"
                    reasons = []
                    if score_nominated:
                        reasons.append(
                            "legacy_geometric_score_at_or_above_"
                            f"{args.classical_uncertain_score:g}"
                        )
                    if temporal_nominated:
                        reasons.append("temporal_has_input_hud_nomination")
                    calibration_reason = "+".join(reasons)
                counts[parsed["class"]] += 1
                record = {
                    "schema_version": 1,
                    "video_id": row["video_id"],
                    "url": row.get("url"),
                    "input_campaign_id": row.get("campaign_id"),
                    "input_worker_id": row.get("worker_id"),
                    "input_attempt_path": row.get("attempt_path"),
                    "frame_file": Path(row["frame_path"]).name,
                    "crop_files": [Path(path).name for path in crop_paths],
                    "auto_tiles": auto_tile_names,
                    "class": parsed["class"],
                    "vlm_class_before_calibration": vlm_class,
                    "confidence": parsed["confidence"],
                    "evidence": parsed["evidence"],
                    "review_candidate": parsed["class"] != "non_target",
                    "parse_error": parsed.get("parse_error"),
                    "calibration_reason": calibration_reason,
                    "classical_has_overlay": row.get("has_overlay"),
                    "classical_has_input_hud": row.get("has_input_hud"),
                    "classical_score": classical_score,
                    "raw_response": raw[:1000],
                    "model": args.model,
                    "resolved_model_revision": revision,
                    "prompt_version": PROMPT_VERSION,
                    "prompt_sha256": sha256_text(PROMPT),
                    "classified_at": utc_now(),
                }
                fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            completed = offset + len(batch)
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = completed / elapsed
            eta_s = (len(rows) - completed) / rate if rate else float("inf")
            print(
                f"[{completed}/{len(rows)}] {rate:.3f} frames/s "
                f"eta={eta_s / 60:.1f}m counts={counts}",
                flush=True,
            )

    manifest["completed_at"] = utc_now()
    manifest["newly_completed"] = len(rows)
    manifest["class_counts_this_run"] = counts
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"complete: {args.out}", flush=True)


if __name__ == "__main__":
    main()
