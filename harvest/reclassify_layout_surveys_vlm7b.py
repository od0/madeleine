"""Resumably reclassify full-source layout surveys with crop-first Qwen 7B evidence.

This stage answers only whether the supplied survey visibly contains a
changing input HUD whose physical controls can be decoded over time. Keyboard,
labeled-action, and controller/gamepad visualizers are all valid; modality is
kept explicit only so later geometry/semantics work can be routed correctly.
Its output is an AI nomination: every record is permanently
``human_reviewed=false`` and ``training_admitted=false``.

The evidence order is crop-first.  Four overlapping quadrants from each of a
small, deterministic set of original survey frames are shown before the full
contact sheet and original frames.  Every queue, survey, source image, derived
tile, prompt, model revision, and output prefix is SHA-256 bound.  The JSONL is
append-only and fsynced one record at a time; its atomic manifest makes a crash
between the row append and manifest update safely resumable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Iterable


PROMPT_VERSION = "celeste-decodable-input-hud-v2-qwen7b-crop-first"
PROMPT_TEMPLATE = """You are auditing ONE full-source Celeste gameplay survey.

The images are ordered crop-first. The evidence map below identifies every
image. Images called `tile` are enlarged overlapping quadrants of exact source
frames. The contact sheet contains all sparse samples across the full video.
The final images are the uncropped exact source frames used for the tiles.

Classify only what is DIRECTLY VISIBLE in the supplied images. The goal is to
find ANY input visualizer whose individual physical controls visibly change
over time. Keyboard, labeled-action cells, and controller/gamepad visualizers
are all valuable. Most videos do not have a usable input HUD.

Choose exactly one class:
- decodable_input_hud: the SAME fixed-geometry input visualizer is visible in
  multiple samples, at least one named physical control visibly changes state,
  and its physical controls are distinguishable well enough for a later
  mechanical scan. This includes keyboards, action-labeled cells, controllers,
  gamepads, analog sticks, D-pads, and mixed layouts.
- non_decodable: there is no input HUD, the apparent overlay is only a timer or
  splits, the controls are illegible, or the visualizer is static/frozen so
  changing inputs cannot be decoded from these images.
- uncertain: a specific input panel is visible, but the supplied evidence is
  too small, blurred, occluded, or internally inconsistent to prove changing
  physical controls.

Rules:
- Controller/gamepad HUDs are VALID decodable input HUDs. Use modality
  `controller`; do not reject them merely for not being keyboards.
- A timer or LiveSplit is NEVER an action HUD.
- Do not infer a HUD from changing gameplay, a geometric detector, or the fact
  that this is Celeste. Point to literal visible structure.
- `physical_control_labels` contains only literal physical control text or
  symbols you can actually read, such as W, A, S, D, Z, X, C, D-pad Left,
  button A, Jump, Dash, or Grab. Never invent a mapping.
- `changing_controls` names the literal controls whose fill, highlight,
  displacement, or pressed state visibly differs between cited samples. A
  decodable_input_hud requires at least one such control and at least two cited
  sample orders. Gameplay motion does not count.
- `evidence_sample_orders` contains the zero-based survey sample orders that
  visibly support the decision. Use [] only when no specific sample is legible.
- `modality` is keyboard, labeled_actions, controller, mixed, none, or unknown.
  It is a routing label, not a quality ranking.
- `non_decodable_reason` is one of timer_splits, no_input_hud,
  static_or_frozen, illegible, other, or none. Use none for
  decodable_input_hud and uncertain.
- Describe position using one allowed location value. Use `none` only when no
  panel exists and `unknown` only when its position cannot be established.

Allowed locations: top_left, top_center, top_right, center_left, center,
center_right, bottom_left, bottom_center, bottom_right, multiple, none, unknown.

Return ONLY one JSON object with EXACTLY these keys and types, no markdown:
{"class":"decodable_input_hud|non_decodable|uncertain","modality":"keyboard|labeled_actions|controller|mixed|none|unknown","confidence":"high|medium|low","evidence":"short literal cross-frame evidence","location":"allowed location","physical_control_labels":["literal control label"],"changing_controls":["literal changing control"],"layout_description":"short geometry description without inferred mappings","evidence_sample_orders":[0,1],"non_decodable_reason":"timer_splits|no_input_hud|static_or_frozen|illegible|other|none"}

Evidence map:
{evidence_map}
"""

ALLOWED_CLASSES = {
    "decodable_input_hud",
    "non_decodable",
    "uncertain",
}
# Classes routed onward to human/modality review. ``non_decodable`` is the
# only terminal negative; a decodable nomination is never an admission.
REVIEW_CANDIDATE_CLASSES = frozenset({"decodable_input_hud", "uncertain"})
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
ALLOWED_MODALITIES = {
    "keyboard",
    "labeled_actions",
    "controller",
    "mixed",
    "none",
    "unknown",
}
ALLOWED_NON_DECODABLE_REASONS = {
    "timer_splits",
    "no_input_hud",
    "static_or_frozen",
    "illegible",
    "other",
    "none",
}
ALLOWED_LOCATIONS = {
    "top_left",
    "top_center",
    "top_right",
    "center_left",
    "center",
    "center_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
    "multiple",
    "none",
    "unknown",
}
RESPONSE_KEYS = {
    "class",
    "modality",
    "confidence",
    "evidence",
    "location",
    "physical_control_labels",
    "changing_controls",
    "layout_description",
    "evidence_sample_orders",
    "non_decodable_reason",
}
SURVEY_FORMAT_VERSION = "madeleine.wild-layout-survey.v1"
SURVEY_PUBLICATION_VERSION = "madeleine.wild-layout-survey-publication.v1"
MANIFEST_SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
EVIDENCE_SPEC_VERSION = "crop-first-overlap-quadrants-v1"
RETRY_POLICY = "one-corrective-retry-v1"
REGIONS = ("top_left", "top_right", "bottom_left", "bottom_right")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def load_queue_rows(queue_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(queue_path.read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f"queue line {line_number} is blank")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"queue line {line_number} is not valid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"queue line {line_number} is not an object")
        video_id = str(row.get("video_id", ""))
        if not _SAFE_ID.fullmatch(video_id):
            raise ValueError(f"queue line {line_number} has unsafe video_id")
        if video_id in seen:
            raise ValueError(f"queue contains duplicate video_id {video_id}")
        if row.get("human_reviewed") not in (None, False):
            raise ValueError(f"queue row {video_id} unexpectedly claims human review")
        if row.get("training_admitted") not in (None, False):
            raise ValueError(f"queue row {video_id} unexpectedly claims admission")
        seen.add(video_id)
        rows.append(row)
    return rows


def _safe_declared_path(directory: Path, raw_path: Any) -> Path:
    relative = Path(str(raw_path))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("survey artifact path must stay inside its directory")
    root = directory.resolve()
    path = (directory / relative).resolve()
    if path.parent != root and root not in path.parents:
        raise ValueError("survey artifact path resolves outside its directory")
    return path


def _verify_declared_file(
    directory: Path, declaration: dict[str, Any], *, context: str
) -> Path:
    path = _safe_declared_path(directory, declaration.get("path"))
    if not path.is_file():
        raise FileNotFoundError(f"{context} is missing: {path}")
    size = declaration.get("size_bytes")
    if not isinstance(size, int) or size < 0 or path.stat().st_size != size:
        raise ValueError(f"{context} size differs from survey manifest")
    expected_hash = str(declaration.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError(f"{context} has invalid SHA-256")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{context} hash differs from survey manifest")
    return path


def available_surveys(
    queue_path: Path, survey_root: Path, *, available_only: bool
) -> list[dict[str, Any]]:
    """Load and byte-verify completion-last surveys in queue order."""

    output: list[dict[str, Any]] = []
    for queue_row in load_queue_rows(queue_path):
        video_id = str(queue_row["video_id"])
        directory = survey_root / video_id
        survey_path = directory / "survey.json"
        completion_path = directory / "survey_complete.json"
        contact_path = directory / "contact-sheet.png"
        required = (survey_path, completion_path, contact_path)
        if not all(path.is_file() for path in required):
            if available_only:
                continue
            raise FileNotFoundError(f"{video_id}: completed survey is missing")
        survey = json.loads(survey_path.read_text())
        completion = json.loads(completion_path.read_text())
        if survey.get("format_version") != SURVEY_FORMAT_VERSION:
            raise ValueError(f"{video_id}: unsupported survey format")
        if completion.get("format_version") != SURVEY_PUBLICATION_VERSION:
            raise ValueError(f"{video_id}: unsupported survey completion format")
        if survey.get("video_id") != video_id or completion.get("video_id") != video_id:
            raise ValueError(f"{video_id}: survey identity mismatch")
        for artifact in (survey, completion):
            if artifact.get("human_reviewed") is not False:
                raise ValueError(f"{video_id}: survey cannot claim human review")
            if artifact.get("training_admitted") is not False:
                raise ValueError(f"{video_id}: survey cannot claim training admission")
        survey_sha256 = sha256_file(survey_path)
        source_sha256 = str(survey.get("source", {}).get("sha256", ""))
        if completion.get("survey_sha256") != survey_sha256:
            raise ValueError(f"{video_id}: completion does not bind survey.json")
        if completion.get("source_sha256") != source_sha256:
            raise ValueError(f"{video_id}: completion does not bind source")
        contact_declaration = survey.get("contact_sheet")
        if not isinstance(contact_declaration, dict):
            raise ValueError(f"{video_id}: survey contact declaration is missing")
        verified_contact = _verify_declared_file(
            directory, contact_declaration, context=f"{video_id} contact sheet"
        )
        if verified_contact != contact_path.resolve():
            raise ValueError(f"{video_id}: contact sheet must use canonical path")
        raw_frames = survey.get("frames")
        if not isinstance(raw_frames, list) or len(raw_frames) < 2:
            raise ValueError(f"{video_id}: survey needs at least two frames")
        frames: list[dict[str, Any]] = []
        sample_orders: set[int] = set()
        for declaration in raw_frames:
            if not isinstance(declaration, dict):
                raise ValueError(f"{video_id}: frame declaration is malformed")
            order = declaration.get("sample_order")
            if not isinstance(order, int) or order < 0 or order in sample_orders:
                raise ValueError(f"{video_id}: invalid or duplicate sample_order")
            sample_orders.add(order)
            frame_path = _verify_declared_file(
                directory,
                declaration,
                context=f"{video_id} sample {order}",
            )
            frames.append({**declaration, "absolute_path": str(frame_path)})
        frames.sort(key=lambda row: int(row["sample_order"]))
        output.append(
            {
                **queue_row,
                "survey_path": str(survey_path.resolve()),
                "survey_sha256": survey_sha256,
                "source_sha256": source_sha256,
                "contact_path": str(contact_path.resolve()),
                "contact_sha256": str(contact_declaration["sha256"]),
                "frames": frames,
            }
        )
    return output


def choose_evidence_frames(
    frames: list[dict[str, Any]], frame_count: int
) -> list[dict[str, Any]]:
    if frame_count < 2:
        raise ValueError("evidence_frame_count must be at least two")
    if len(frames) <= frame_count:
        return frames.copy()
    indices = [
        round(index * (len(frames) - 1) / (frame_count - 1))
        for index in range(frame_count)
    ]
    if len(set(indices)) != frame_count:
        raise ValueError("could not select unique evidence frames")
    return [frames[index] for index in indices]


def quadrant_boxes(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    if width < 2 or height < 2:
        raise ValueError("survey frame is too small to tile")
    overlap_x = max(1, width // 20)
    overlap_y = max(1, height // 20)
    mid_x, mid_y = width // 2, height // 2
    return {
        "top_left": (0, 0, min(width, mid_x + overlap_x), min(height, mid_y + overlap_y)),
        "top_right": (max(0, mid_x - overlap_x), 0, width, min(height, mid_y + overlap_y)),
        "bottom_left": (0, max(0, mid_y - overlap_y), min(width, mid_x + overlap_x), height),
        "bottom_right": (max(0, mid_x - overlap_x), max(0, mid_y - overlap_y), width, height),
    }


def _png_bytes(image: Any) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


@dataclass
class EvidenceBundle:
    images: list[Any]
    descriptors: list[dict[str, Any]]
    prompt: str
    prompt_sha256: str
    image_bundle_sha256: str
    all_sample_orders: frozenset[int]


def prepare_evidence(
    row: dict[str, Any], *, evidence_frame_count: int
) -> EvidenceBundle:
    """Build the exact ordered image bundle supplied to the model."""

    from PIL import Image

    selected = choose_evidence_frames(row["frames"], evidence_frame_count)
    images: list[Any] = []
    descriptors: list[dict[str, Any]] = []
    opened_frames: list[tuple[dict[str, Any], Any]] = []
    try:
        for frame in selected:
            opened = Image.open(frame["absolute_path"]).convert("RGB")
            opened_frames.append((frame, opened))
        position = 1
        for frame, opened in opened_frames:
            for region in REGIONS:
                box = quadrant_boxes(*opened.size)[region]
                tile = opened.crop(box)
                tile_hash = hashlib.sha256(_png_bytes(tile)).hexdigest()
                images.append(tile)
                descriptors.append(
                    {
                        "position": position,
                        "kind": "tile",
                        "sample_order": int(frame["sample_order"]),
                        "region": region,
                        "crop_box_xyxy": list(box),
                        "source_sha256": str(frame["sha256"]),
                        "sha256": tile_hash,
                        "width": tile.width,
                        "height": tile.height,
                    }
                )
                position += 1
        contact = Image.open(row["contact_path"]).convert("RGB")
        images.append(contact)
        descriptors.append(
            {
                "position": position,
                "kind": "contact_sheet",
                "sha256": row["contact_sha256"],
                "width": contact.width,
                "height": contact.height,
                "sample_orders": [
                    int(frame["sample_order"]) for frame in row["frames"]
                ],
            }
        )
        position += 1
        for frame, opened in opened_frames:
            images.append(opened)
            descriptors.append(
                {
                    "position": position,
                    "kind": "full_frame",
                    "sample_order": int(frame["sample_order"]),
                    "sha256": str(frame["sha256"]),
                    "width": opened.width,
                    "height": opened.height,
                }
            )
            position += 1
        evidence_map = "\n".join(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
            for descriptor in descriptors
        )
        # str.format would treat the literal JSON braces in the response
        # schema as replacement fields, so substitute the placeholder directly.
        prompt = PROMPT_TEMPLATE.replace("{evidence_map}", evidence_map)
        return EvidenceBundle(
            images=images,
            descriptors=descriptors,
            prompt=prompt,
            prompt_sha256=sha256_text(prompt),
            image_bundle_sha256=canonical_sha256(descriptors),
            all_sample_orders=frozenset(
                int(frame["sample_order"]) for frame in row["frames"]
            ),
        )
    except Exception:
        for image in images:
            try:
                image.close()
            except Exception:
                pass
        for _, image in opened_frames:
            try:
                image.close()
            except Exception:
                pass
        raise


def close_evidence(bundle: EvidenceBundle) -> None:
    seen: set[int] = set()
    for image in bundle.images:
        if id(image) not in seen:
            image.close()
            seen.add(id(image))


def _strict_json_object(raw: str) -> tuple[dict[str, Any] | None, list[str]]:
    stripped = raw.strip()
    decoder = json.JSONDecoder()
    if not stripped.startswith("{"):
        return None, ["response_not_bare_json_object"]
    try:
        value, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        return None, [f"invalid_json:{exc.msg}"]
    if stripped[end:].strip():
        return None, ["extraneous_text_after_json"]
    if not isinstance(value, dict):
        return None, ["response_is_not_object"]
    return value, []


def _fail_closed(errors: Iterable[str]) -> dict[str, Any]:
    error_list = list(errors)
    return {
        "class": "uncertain",
        "modality": "unknown",
        "confidence": "low",
        "evidence": "model output failed strict validation",
        "location": "unknown",
        "physical_control_labels": [],
        "changing_controls": [],
        "layout_description": "unvalidated model response",
        "evidence_sample_orders": [],
        "non_decodable_reason": "none",
        "validation_errors": error_list or ["unknown_validation_error"],
    }


def parse_response(raw: str, valid_sample_orders: set[int] | frozenset[int]) -> dict[str, Any]:
    """Strictly validate one response, downgrading every defect to uncertain."""

    value, errors = _strict_json_object(raw)
    if value is None:
        return _fail_closed(errors)
    if set(value) != RESPONSE_KEYS:
        missing = sorted(RESPONSE_KEYS - set(value))
        extra = sorted(set(value) - RESPONSE_KEYS)
        if missing:
            errors.append("missing_keys:" + ",".join(missing))
        if extra:
            errors.append("extra_keys:" + ",".join(extra))

    label = value.get("class")
    modality = value.get("modality")
    confidence = value.get("confidence")
    evidence = value.get("evidence")
    location = value.get("location")
    labels = value.get("physical_control_labels")
    changing_controls = value.get("changing_controls")
    description = value.get("layout_description")
    sample_orders = value.get("evidence_sample_orders")
    non_decodable_reason = value.get("non_decodable_reason")

    if not isinstance(label, str) or label not in ALLOWED_CLASSES:
        errors.append("invalid_class")
    if not isinstance(modality, str) or modality not in ALLOWED_MODALITIES:
        errors.append("invalid_modality")
    if not isinstance(confidence, str) or confidence not in ALLOWED_CONFIDENCE:
        errors.append("invalid_confidence")
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 800:
        errors.append("invalid_evidence")
    if not isinstance(location, str) or location not in ALLOWED_LOCATIONS:
        errors.append("invalid_location")
    if not isinstance(description, str) or not description.strip() or len(description) > 800:
        errors.append("invalid_layout_description")
    if (
        not isinstance(labels, list)
        or len(labels) > 32
        or any(not isinstance(item, str) or not item.strip() or len(item) > 64 for item in labels)
    ):
        errors.append("invalid_physical_control_labels")
    if (
        not isinstance(changing_controls, list)
        or len(changing_controls) > 32
        or any(
            not isinstance(item, str) or not item.strip() or len(item) > 64
            for item in changing_controls
        )
    ):
        errors.append("invalid_changing_controls")
    if (
        not isinstance(sample_orders, list)
        or len(sample_orders) > len(valid_sample_orders)
        or any(type(item) is not int for item in sample_orders)
        or len(set(sample_orders)) != len(sample_orders)
        or any(item not in valid_sample_orders for item in sample_orders)
    ):
        errors.append("invalid_evidence_sample_orders")
    if (
        not isinstance(non_decodable_reason, str)
        or non_decodable_reason not in ALLOWED_NON_DECODABLE_REASONS
    ):
        errors.append("invalid_non_decodable_reason")

    if errors:
        return _fail_closed(errors)

    assert isinstance(label, str)
    assert isinstance(modality, str)
    assert isinstance(confidence, str)
    assert isinstance(evidence, str)
    assert isinstance(location, str)
    assert isinstance(labels, list)
    assert isinstance(changing_controls, list)
    assert isinstance(description, str)
    assert isinstance(sample_orders, list)
    assert isinstance(non_decodable_reason, str)
    normalized_labels = [" ".join(item.split()) for item in labels]
    normalized_changes = [" ".join(item.split()) for item in changing_controls]

    contradictions: list[str] = []
    if label == "decodable_input_hud":
        if modality in {"none", "unknown"}:
            contradictions.append("decodable_hud_lacks_routable_modality")
        if location in {"none", "unknown"}:
            contradictions.append("decodable_hud_has_nonvisible_location")
        if len(sample_orders) < 2:
            contradictions.append("decodable_hud_lacks_cross_frame_evidence")
        if not normalized_labels:
            contradictions.append("decodable_hud_lacks_physical_control_labels")
        if not normalized_changes:
            contradictions.append("decodable_hud_lacks_visible_changing_controls")
        if non_decodable_reason != "none":
            contradictions.append("decodable_hud_has_non_decodable_reason")
    elif label == "non_decodable":
        if non_decodable_reason == "none":
            contradictions.append("non_decodable_lacks_reason")
        if normalized_changes:
            contradictions.append("non_decodable_claims_changing_controls")
        if non_decodable_reason == "no_input_hud":
            if modality != "none":
                contradictions.append("no_input_hud_has_modality")
            if location != "none":
                contradictions.append("no_input_hud_has_visible_location")
            if normalized_labels:
                contradictions.append("no_input_hud_has_control_labels")
        elif modality == "unknown":
            contradictions.append("non_decodable_panel_has_unknown_modality")
    elif label == "uncertain":
        if non_decodable_reason != "none":
            contradictions.append("uncertain_has_non_decodable_reason")
        if location == "none":
            contradictions.append("uncertain_uses_none_location")

    if contradictions:
        failed = _fail_closed(contradictions)
        failed["model_claimed_class"] = label
        return failed
    return {
        "class": label,
        "modality": modality,
        "confidence": confidence,
        "evidence": " ".join(evidence.split()),
        "location": location,
        "physical_control_labels": normalized_labels,
        "changing_controls": normalized_changes,
        "layout_description": " ".join(description.split()),
        "evidence_sample_orders": sample_orders,
        "non_decodable_reason": non_decodable_reason,
        "validation_errors": [],
    }


def build_corrective_message(validation_errors: Iterable[str]) -> str:
    errors = list(validation_errors)
    return (
        "The previous response failed validation with these errors:\n"
        + "\n".join(str(error) for error in errors)
        + "\nReturn ONLY one JSON object with EXACTLY the required keys and "
        "allowed values, no markdown."
    )


def finalize_response_attempts(
    first_raw: str,
    first_parsed: dict[str, Any],
    *,
    retry_raw: str | None = None,
    retry_parsed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select a validated retry or retain the first fail-closed response."""

    first_errors = first_parsed.get("validation_errors")
    if not isinstance(first_errors, list):
        raise ValueError("first parsed response lacks validation_errors")
    if not first_errors:
        if retry_raw is not None or retry_parsed is not None:
            raise ValueError("validated first responses must not have a retry")
        return {
            **first_parsed,
            "retry_count": 0,
            "raw_response": first_raw[:4000],
        }
    if retry_raw is None or retry_parsed is None:
        raise ValueError("invalid first responses require one retry result")
    retry_errors = retry_parsed.get("validation_errors")
    if not isinstance(retry_errors, list):
        raise ValueError("retry parsed response lacks validation_errors")
    selected = retry_parsed if not retry_errors else first_parsed
    result = {
        **selected,
        "retry_count": 1,
        "raw_response": first_raw[:4000],
        "raw_response_retry": retry_raw[:4000],
    }
    if retry_errors:
        result["retry_validation_errors"] = list(retry_errors)
    return result


def _sha256_line_prefix(path: Path, rows: int) -> str:
    digest = hashlib.sha256()
    consumed = 0
    if path.exists():
        with path.open("rb") as handle:
            for line in handle:
                if consumed == rows:
                    break
                digest.update(line)
                consumed += 1
    if consumed != rows:
        raise ValueError("prediction output is shorter than manifest prefix")
    return digest.hexdigest()


def validate_record_identity(
    record: dict[str, Any],
    row: dict[str, Any],
    bundle: EvidenceBundle,
    *,
    queue_sha256: str,
    model: str,
    resolved_model_revision: str,
    max_new_tokens: int,
) -> None:
    video_id = str(row["video_id"])
    expected = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "video_id": video_id,
        "queue_sha256": queue_sha256,
        "survey_sha256": row["survey_sha256"],
        "source_sha256": row["source_sha256"],
        "contact_sha256": row["contact_sha256"],
        "image_bundle_sha256": bundle.image_bundle_sha256,
        "input_images": bundle.descriptors,
        "model": model,
        "resolved_model_revision": resolved_model_revision,
        "prompt_version": PROMPT_VERSION,
        "prompt_template_sha256": sha256_text(PROMPT_TEMPLATE),
        "prompt_sha256": bundle.prompt_sha256,
        "evidence_spec_version": EVIDENCE_SPEC_VERSION,
        "max_new_tokens": max_new_tokens,
        "human_reviewed": False,
        "training_admitted": False,
    }
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            raise ValueError(f"prediction {video_id} has stale or mismatched {field}")
    if record.get("class") not in ALLOWED_CLASSES:
        raise ValueError(f"prediction {video_id} has invalid class")
    if record.get("confidence") not in ALLOWED_CONFIDENCE:
        raise ValueError(f"prediction {video_id} has invalid confidence")
    retry_count = record.get("retry_count")
    if type(retry_count) is not int or retry_count not in {0, 1}:
        raise ValueError(f"prediction {video_id} has invalid retry_count")
    has_retry_raw = "raw_response_retry" in record
    has_retry_errors = "retry_validation_errors" in record
    if retry_count == 0 and (has_retry_raw or has_retry_errors):
        raise ValueError(f"prediction {video_id} has inconsistent retry metadata")
    if retry_count == 1 and not isinstance(record.get("raw_response_retry"), str):
        raise ValueError(f"prediction {video_id} is missing retry response")
    if has_retry_errors and (
        not isinstance(record["retry_validation_errors"], list)
        or not record["retry_validation_errors"]
        or any(not isinstance(error, str) for error in record["retry_validation_errors"])
    ):
        raise ValueError(f"prediction {video_id} has invalid retry validation errors")
    expected_candidate = record["class"] in REVIEW_CANDIDATE_CLASSES
    if record.get("review_candidate") is not expected_candidate:
        raise ValueError(f"prediction {video_id} has inconsistent review candidacy")


def load_existing_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    bundles: dict[str, EvidenceBundle],
    *,
    queue_sha256: str,
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
            raise ValueError(f"prediction {video_id} lacks a completed current survey")
        validate_record_identity(
            record,
            row_by_id[video_id],
            bundles[video_id],
            queue_sha256=queue_sha256,
            model=model,
            resolved_model_revision=resolved_model_revision,
            max_new_tokens=max_new_tokens,
        )
        seen.add(video_id)
        output.append(record)
    return output


def validate_resume_manifest(
    manifest_path: Path,
    expected_identity: dict[str, Any],
    predictions_path: Path,
    prediction_rows: int,
) -> str:
    if not manifest_path.exists():
        if prediction_rows:
            raise ValueError("prediction output exists without its manifest")
        return utc_now()
    manifest = json.loads(manifest_path.read_text())
    for field, expected in expected_identity.items():
        if manifest.get(field) != expected:
            raise ValueError(f"resume manifest has stale or mismatched {field}")
    prefix = manifest.get("predictions")
    if not isinstance(prefix, dict):
        raise ValueError("resume manifest is missing prediction prefix binding")
    prior_rows = prefix.get("rows")
    if not isinstance(prior_rows, int) or prior_rows < 0 or prior_rows > prediction_rows:
        raise ValueError("resume manifest prediction row count is invalid")
    if prefix.get("sha256") != _sha256_line_prefix(predictions_path, prior_rows):
        raise ValueError("prediction output prefix differs from its manifest")
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("resume manifest is missing created_at")
    return created_at


def _manifest_prediction_state(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "unique_video_ids": len({record["video_id"] for record in records}),
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--survey-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--evidence-frame-count", type=int, default=4)
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    args = parser.parse_args()

    if args.evidence_frame_count < 2:
        raise ValueError("--evidence-frame-count must be at least two")
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
    rows = available_surveys(
        args.queue, args.survey_root, available_only=args.available_only
    )
    bundles: dict[str, EvidenceBundle] = {}
    try:
        for row in rows:
            bundles[str(row["video_id"])] = prepare_evidence(
                row, evidence_frame_count=args.evidence_frame_count
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        existing = load_existing_predictions(
            args.out,
            rows,
            bundles,
            queue_sha256=queue_sha256,
            model=args.model,
            resolved_model_revision=revision,
            max_new_tokens=args.max_new_tokens,
        )
        done = {str(record["video_id"]) for record in existing}
        pending = [row for row in rows if str(row["video_id"]) not in done]
        manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
        manifest_identity = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "task": "celeste_decodable_input_hud_reclassification",
            "classification_is_human_review": False,
            "training_admission": False,
            "queue_sha256": queue_sha256,
            "prompt_version": PROMPT_VERSION,
            "prompt_template_sha256": sha256_text(PROMPT_TEMPLATE),
            "model": args.model,
            "resolved_model_revision": revision,
            "evidence_spec_version": EVIDENCE_SPEC_VERSION,
            "evidence_frame_count": args.evidence_frame_count,
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
            "available_rows": len(rows),
            "already_complete_on_start": len(existing),
            "classes": sorted(ALLOWED_CLASSES),
            "review_candidate_classes": sorted(REVIEW_CANDIDATE_CLASSES),
            "predictions": _manifest_prediction_state(args.out, existing),
        }
        write_json_atomic(manifest_path, manifest)
        if not pending:
            manifest["completed_at"] = utc_now()
            manifest["class_counts_total"] = {
                label: sum(record["class"] == label for record in existing)
                for label in ALLOWED_CLASSES
            }
            write_json_atomic(manifest_path, manifest)
            print(f"[0/0] validated {len(existing)} completed predictions")
            return

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for production 7B reclassification")
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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

        started = time.monotonic()
        new_records: list[dict[str, Any]] = []
        with args.out.open("a") as handle, torch.inference_mode():
            for index, row in enumerate(pending, 1):
                video_id = str(row["video_id"])
                bundle = bundles[video_id]
                # qwen_vl_utils resizes from the per-image bounds, not the
                # processor config; omitting them sends multi-megapixel frames
                # and exhausts device memory on 21-image bundles.
                content = [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": args.min_pixels,
                        "max_pixels": args.max_pixels,
                    }
                    for image in bundle.images
                ]
                content.append({"type": "text", "text": bundle.prompt})
                messages = [[{"role": "user", "content": content}]]
                rendered = processor.apply_chat_template(
                    messages[0], tokenize=False, add_generation_prompt=True
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
                del generated, inputs, trimmed
                first_parsed = parse_response(raw, bundle.all_sample_orders)
                if first_parsed["validation_errors"]:
                    retry_messages = [
                        *messages[0],
                        {"role": "assistant", "content": raw[:4000]},
                        {
                            "role": "user",
                            "content": build_corrective_message(
                                first_parsed["validation_errors"]
                            ),
                        },
                    ]
                    retry_rendered = processor.apply_chat_template(
                        retry_messages, tokenize=False, add_generation_prompt=True
                    )
                    retry_image_inputs, retry_video_inputs = process_vision_info(
                        [retry_messages]
                    )
                    retry_inputs = processor(
                        text=[retry_rendered],
                        images=retry_image_inputs,
                        videos=retry_video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to("cuda")
                    retry_generated = model.generate(
                        **retry_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                    retry_trimmed = retry_generated[0][
                        len(retry_inputs.input_ids[0]) :
                    ]
                    retry_raw = processor.decode(
                        retry_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    retry_parsed = parse_response(
                        retry_raw, bundle.all_sample_orders
                    )
                    parsed = finalize_response_attempts(
                        raw,
                        first_parsed,
                        retry_raw=retry_raw,
                        retry_parsed=retry_parsed,
                    )
                else:
                    parsed = finalize_response_attempts(raw, first_parsed)
                record = {
                    "schema_version": RECORD_SCHEMA_VERSION,
                    "video_id": video_id,
                    "source": row.get("source"),
                    "nominal_hours": row.get("nominal_hours"),
                    "queue_sha256": queue_sha256,
                    "survey_sha256": row["survey_sha256"],
                    "source_sha256": row["source_sha256"],
                    "contact_sha256": row["contact_sha256"],
                    "image_bundle_sha256": bundle.image_bundle_sha256,
                    "input_images": bundle.descriptors,
                    **parsed,
                    "review_candidate": parsed["class"] in REVIEW_CANDIDATE_CLASSES,
                    "human_reviewed": False,
                    "training_admitted": False,
                    "model": args.model,
                    "resolved_model_revision": revision,
                    "prompt_version": PROMPT_VERSION,
                    "prompt_template_sha256": sha256_text(PROMPT_TEMPLATE),
                    "prompt_sha256": bundle.prompt_sha256,
                    "evidence_spec_version": EVIDENCE_SPEC_VERSION,
                    "max_new_tokens": args.max_new_tokens,
                    "classified_at": utc_now(),
                }
                validate_record_identity(
                    record,
                    row,
                    bundle,
                    queue_sha256=queue_sha256,
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
                    label: sum(record["class"] == label for record in all_records)
                    for label in ALLOWED_CLASSES
                }
                write_json_atomic(manifest_path, manifest)
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[{index}/{len(pending)}] {video_id} {parsed['class']} "
                    f"rate={index / elapsed:.3f}/s",
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
