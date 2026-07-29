"""Accept wild-HUD layout geometry through a portable, hash-bound review packet.

An editable ``human_reviewed`` boolean is not provenance.  This module turns an
unreviewed layout draft and a deliberately small review packet into two new,
immutable artifacts:

* a reviewed layout that embeds a reference to the acceptance; and
* an acceptance that binds the draft, source-video identity, portable review
  manifest, every required review artifact, every evidence-frame image, and an
  explicit reviewer identity/kind.

The v2 review manifest is intentionally whitelist-only.  It contains relative
paths, sizes, hashes, frame times, and the source SHA-256, but no origin URL,
host name, raw-video path, credentials, free-form notes, or reviewer identity.
It can therefore be committed and verified from a clean clone without access
to the raw video or any cloud credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Iterable

from harvest.fetch_wild import sha256_file
from harvest.wild_layout import WildLayout


REVIEW_MANIFEST_VERSION = "madeleine.wild-layout-review.v2"
LAYOUT_ACCEPTANCE_VERSION = "madeleine.wild-layout-acceptance.v1"
REVIEWER_KINDS = ("human", "human_with_ai_assistance", "ai_agent")
HUMAN_REVIEWER_KINDS = frozenset(("human", "human_with_ai_assistance"))
REQUIRED_ARTIFACT_ROLES = frozenset(
    ("geometry_overlay", "cell_state_evidence", "cell_state_contact_sheet")
)

_IMAGE_SUFFIXES = frozenset((".jpg", ".jpeg", ".png"))
_TIMING_FIELDS = frozenset(
    (
        "temporal_offset_frames",
        "temporal_offset_source",
        "temporal_offset_confidence",
        "temporal_offset_acceptance",
    )
)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{field} fields do not match the v2 privacy whitelist; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _sha256(value: Any, field: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return result


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _video_id(value: Any) -> str:
    result = str(value).strip()
    if (
        not result
        or result in (".", "..")
        or "/" in result
        or "\\" in result
        or any(ord(char) < 32 for char in result)
    ):
        raise ValueError("video_id must be one safe identifier")
    return result


def _relative_path(value: Any, field: str) -> str:
    """Return one portable POSIX path with no traversal or URL syntax."""

    result = str(value).strip()
    path = PurePosixPath(result)
    if (
        not result
        or "\\" in result
        or ":" in result
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError(f"{field} must be a clean-clone-safe relative POSIX path")
    return path.as_posix()


def _local_name(value: Any, field: str) -> str:
    result = _relative_path(value, field)
    if PurePosixPath(result).name != result:
        raise ValueError(f"{field} must be one local file name")
    return result


def _regular_file(path: Path, field: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field} must be an existing non-symlink regular file")
    return path


def _load_json(path: Path, field: str) -> dict[str, Any]:
    _regular_file(path, field)

    def reject_constant(value: str) -> None:
        raise ValueError(f"{field} contains non-finite JSON number {value}")

    try:
        value = json.loads(path.read_text(), parse_constant=reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{field} is not valid UTF-8 JSON") from exc
    return _mapping(value, field)


def _resolve_packet_path(base: Path, relative: str, field: str) -> Path:
    candidate = base.joinpath(*PurePosixPath(relative).parts)
    _regular_file(candidate, field)
    try:
        candidate.resolve().relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} resolves outside the review packet") from exc
    return candidate


def _entry_for(path: Path, *, base: Path) -> dict[str, Any]:
    local = _regular_file(path, "review-packet input")
    try:
        relative = local.resolve().relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(
            f"review-packet inputs must live under the manifest directory: {local}"
        ) from exc
    return {
        "path": PurePosixPath(*relative.parts).as_posix(),
        "sha256": sha256_file(local),
        "size_bytes": local.stat().st_size,
    }


def _validate_file_entry(
    raw: Any,
    *,
    base: Path,
    field: str,
    extra_keys: set[str] = frozenset(),
) -> tuple[dict[str, Any], Path]:
    row = _mapping(raw, field)
    _exact_keys(row, {"path", "sha256", "size_bytes", *extra_keys}, field)
    relative = _relative_path(row.get("path"), f"{field}.path")
    expected_hash = _sha256(row.get("sha256"), f"{field}.sha256")
    expected_size = _positive_integer(row.get("size_bytes"), f"{field}.size_bytes")
    local = _resolve_packet_path(base, relative, field)
    if local.stat().st_size != expected_size:
        raise ValueError(f"{field} size does not match the manifest")
    if sha256_file(local) != expected_hash:
        raise ValueError(f"{field} hash does not match the manifest")
    normalized = {
        key: row[key]
        for key in ("role", "time_s", "path", "sha256", "size_bytes")
        if key in row
    }
    normalized.update({
        "path": relative,
        "sha256": expected_hash,
        "size_bytes": expected_size,
    })
    return normalized, local


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reviewed_core_hash(layout_raw: dict[str, Any]) -> str:
    return _canonical_hash(
        {key: value for key, value in layout_raw.items() if key not in _TIMING_FIELDS}
    )


def _atomic_write_new(path: Path, payload: bytes) -> None:
    """Atomically create ``path`` and fail instead of replacing any bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def write_review_manifest(
    output_path: str | Path,
    draft_layout_path: str | Path,
    *,
    source_sha256: str,
    artifacts: dict[str, str | Path],
    evidence_frames: Iterable[tuple[float, str | Path]],
) -> dict[str, Any]:
    """Write one deterministic, portable v2 review manifest without metadata spill."""

    output = Path(output_path)
    base = output.parent
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    draft_file = _regular_file(Path(draft_layout_path), "draft layout")
    draft = WildLayout.load(draft_file)
    if draft.human_reviewed:
        raise ValueError("review manifest input must be an unreviewed draft layout")
    if draft.temporal_offset_source != "unmeasured":
        raise ValueError("review manifest input layout must have an unmeasured offset")
    source_hash = _sha256(source_sha256, "source_sha256")
    if set(artifacts) != REQUIRED_ARTIFACT_ROLES:
        raise ValueError(
            "review artifacts must contain exactly the required roles: "
            + ", ".join(sorted(REQUIRED_ARTIFACT_ROLES))
        )

    artifact_rows = []
    for role in sorted(artifacts):
        local = Path(artifacts[role])
        if role == "cell_state_evidence":
            if local.suffix.lower() != ".json":
                raise ValueError("cell_state_evidence must be JSON")
        elif local.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError(f"{role} must be a PNG or JPEG image")
        artifact_rows.append({"role": role, **_entry_for(local, base=base)})

    frame_rows = []
    seen_times: set[float] = set()
    seen_paths: set[str] = set()
    for index, (time_s, frame_path) in enumerate(evidence_frames):
        time_value = _finite_nonnegative(time_s, f"evidence_frames[{index}].time_s")
        if time_value in seen_times:
            raise ValueError("evidence frame times must be unique")
        local = Path(frame_path)
        if local.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError("evidence frames must be PNG or JPEG images")
        entry = _entry_for(local, base=base)
        if entry["path"] in seen_paths:
            raise ValueError("evidence frame paths must be unique")
        seen_times.add(time_value)
        seen_paths.add(entry["path"])
        frame_rows.append({"time_s": time_value, **entry})
    frame_rows.sort(key=lambda row: (row["time_s"], row["path"]))

    frame_times = [float(row["time_s"]) for row in frame_rows]
    if len(frame_times) != len(draft.evidence_frames) or any(
        not any(math.isclose(expected, actual, rel_tol=0.0, abs_tol=1e-6)
                for actual in frame_times)
        for expected in draft.evidence_frames
    ):
        raise ValueError(
            "review manifest evidence frames must exactly cover layout.evidence_frames_s"
        )
    _validate_cell_state_evidence(
        Path(artifacts["cell_state_evidence"]),
        layout=draft,
        frame_rows=frame_rows,
    )

    manifest = {
        "format_version": REVIEW_MANIFEST_VERSION,
        "video_id": draft.video_id,
        "source_video_sha256": source_hash,
        "draft_layout": _entry_for(draft_file, base=base),
        "review_artifacts": artifact_rows,
        "evidence_frames": frame_rows,
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_new(output, payload)
    return manifest


def _validate_cell_state_evidence(
    evidence_file: Path,
    *,
    layout: WildLayout,
    frame_rows: list[dict[str, Any]],
) -> None:
    evidence = _load_json(evidence_file, "cell-state evidence")
    if evidence.get("video_id") != layout.video_id:
        raise ValueError("cell-state evidence video_id differs from the layout")
    cells = _sequence(evidence.get("cells"), "cell-state evidence.cells")
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(cells):
        row = _mapping(raw, f"cell-state evidence.cells[{index}]")
        cell_id = str(row.get("cell_id", "")).strip()
        if not cell_id or cell_id in by_id:
            raise ValueError("cell-state evidence cell_id values must be nonempty and unique")
        by_id[cell_id] = row
    frame_hashes = {
        (str(row["path"]), str(row["sha256"])) for row in frame_rows
    }
    for cell in layout.cells:
        row = by_id.get(cell.cell_id)
        if row is None or row.get("action") != cell.action:
            raise ValueError(f"cell-state evidence is missing layout cell {cell.cell_id}")
        for state in ("released", "pressed"):
            state_row = _mapping(row.get(state), f"cell {cell.cell_id}.{state}")
            path = _relative_path(
                state_row.get("path"), f"cell {cell.cell_id}.{state}.path"
            )
            digest = _sha256(
                state_row.get("sha256"), f"cell {cell.cell_id}.{state}.sha256"
            )
            if (path, digest) not in frame_hashes:
                raise ValueError(
                    f"cell {cell.cell_id} {state} frame is not hash-bound in evidence_frames"
                )


def validate_review_manifest(
    manifest_path: str | Path,
    *,
    expected_draft_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify one v2 manifest and every local byte it names."""

    manifest_file = Path(manifest_path)
    manifest = _load_json(manifest_file, "review manifest")
    _exact_keys(
        manifest,
        {
            "format_version",
            "video_id",
            "source_video_sha256",
            "draft_layout",
            "review_artifacts",
            "evidence_frames",
        },
        "review manifest",
    )
    if manifest.get("format_version") != REVIEW_MANIFEST_VERSION:
        raise ValueError(
            f"unsupported review manifest format; expected {REVIEW_MANIFEST_VERSION}"
        )
    video_id = _video_id(manifest.get("video_id"))
    source_hash = _sha256(
        manifest.get("source_video_sha256"), "review manifest.source_video_sha256"
    )
    base = manifest_file.parent
    draft_row, draft_file = _validate_file_entry(
        manifest.get("draft_layout"), base=base, field="review manifest.draft_layout"
    )
    if expected_draft_path is not None and draft_file.resolve() != Path(
        expected_draft_path
    ).resolve():
        raise ValueError("review manifest is bound to a different draft layout path")
    draft = WildLayout.load(draft_file)
    if draft.video_id != video_id:
        raise ValueError("review manifest video_id differs from the draft layout")
    if draft.human_reviewed:
        raise ValueError("review manifest draft must remain unreviewed")
    if draft.temporal_offset_source != "unmeasured":
        raise ValueError("review manifest draft must have an unmeasured offset")

    artifact_rows: list[dict[str, Any]] = []
    artifact_files: dict[str, Path] = {}
    roles: set[str] = set()
    paths = {draft_row["path"]}
    for index, raw in enumerate(
        _sequence(manifest.get("review_artifacts"), "review manifest.review_artifacts")
    ):
        row, local = _validate_file_entry(
            raw,
            base=base,
            field=f"review manifest.review_artifacts[{index}]",
            extra_keys={"role"},
        )
        role = str(row.get("role", "")).strip()
        if role not in REQUIRED_ARTIFACT_ROLES or role in roles:
            raise ValueError("review artifact roles must be required and unique")
        if row["path"] in paths:
            raise ValueError("review-packet paths must be unique")
        if role == "cell_state_evidence":
            if local.suffix.lower() != ".json":
                raise ValueError("cell_state_evidence must be JSON")
        elif local.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError(f"{role} must be a PNG or JPEG image")
        roles.add(role)
        paths.add(row["path"])
        artifact_rows.append(row)
        artifact_files[role] = local
    if roles != REQUIRED_ARTIFACT_ROLES:
        raise ValueError(
            "review manifest is missing required review artifact roles: "
            + ", ".join(sorted(REQUIRED_ARTIFACT_ROLES - roles))
        )

    frame_rows: list[dict[str, Any]] = []
    frame_times: list[float] = []
    for index, raw in enumerate(
        _sequence(manifest.get("evidence_frames"), "review manifest.evidence_frames")
    ):
        row, local = _validate_file_entry(
            raw,
            base=base,
            field=f"review manifest.evidence_frames[{index}]",
            extra_keys={"time_s"},
        )
        time_s = _finite_nonnegative(
            row.get("time_s"), f"review manifest.evidence_frames[{index}].time_s"
        )
        if local.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError("review evidence frames must be PNG or JPEG images")
        if row["path"] in paths:
            raise ValueError("review-packet paths must be unique")
        if any(math.isclose(time_s, prior, rel_tol=0.0, abs_tol=1e-9) for prior in frame_times):
            raise ValueError("review evidence-frame times must be unique")
        paths.add(row["path"])
        frame_times.append(time_s)
        row["time_s"] = time_s
        frame_rows.append(row)
    if not frame_rows:
        raise ValueError("review manifest must bind at least one source-frame image")
    if len(frame_times) != len(draft.evidence_frames) or any(
        not any(math.isclose(expected, actual, rel_tol=0.0, abs_tol=1e-6)
                for actual in frame_times)
        for expected in draft.evidence_frames
    ):
        raise ValueError(
            "review manifest evidence frames must exactly cover layout.evidence_frames_s"
        )
    _validate_cell_state_evidence(
        artifact_files["cell_state_evidence"], layout=draft, frame_rows=frame_rows
    )

    return {
        "format_version": REVIEW_MANIFEST_VERSION,
        "path": str(manifest_file),
        "sha256": sha256_file(manifest_file),
        "video_id": video_id,
        "source_video_sha256": source_hash,
        "draft_layout": draft_row,
        "draft_layout_path": str(draft_file),
        "review_artifacts": artifact_rows,
        "evidence_frames": frame_rows,
    }


def accept_layout(
    review_manifest_path: str | Path,
    draft_layout_path: str | Path,
    output_layout_path: str | Path,
    acceptance_path: str | Path,
    *,
    reviewer_identity: str,
    reviewer_kind: str,
    approved: bool,
) -> dict[str, Any]:
    """Create an immutable reviewed layout and hash-bound acceptance."""

    manifest_file = Path(review_manifest_path)
    draft_file = Path(draft_layout_path)
    output_file = Path(output_layout_path)
    acceptance_file = Path(acceptance_path)
    identity = reviewer_identity.strip()
    if not identity:
        raise ValueError("reviewer_identity is required")
    if reviewer_kind not in REVIEWER_KINDS:
        raise ValueError(f"reviewer_kind must be one of {REVIEWER_KINDS}")
    if approved is not True:
        raise ValueError("explicit layout-review approval is required")
    if output_file.resolve() == draft_file.resolve():
        raise ValueError("reviewed layout must be written to a new path")
    if acceptance_file.parent.resolve() != manifest_file.parent.resolve():
        raise ValueError("layout acceptance must live beside its review manifest")
    if output_file.exists() or acceptance_file.exists():
        raise FileExistsError("refusing to overwrite reviewed layout or acceptance")

    checked = validate_review_manifest(
        manifest_file, expected_draft_path=draft_file
    )
    draft_raw = _load_json(draft_file, "draft layout")
    draft = WildLayout.from_dict(draft_raw)
    human_reviewed = reviewer_kind in HUMAN_REVIEWER_KINDS
    embedded = {
        "format_version": LAYOUT_ACCEPTANCE_VERSION,
        "artifact": acceptance_file.name,
        "review_manifest_sha256": checked["sha256"],
        "source_video_sha256": checked["source_video_sha256"],
    }
    output_raw = dict(draft_raw)
    output_raw["human_reviewed"] = human_reviewed
    output_raw["layout_review_acceptance"] = embedded
    output_layout = WildLayout.from_dict(output_raw)
    if output_layout.video_id != draft.video_id:
        raise AssertionError("reviewed layout unexpectedly changed video_id")
    layout_bytes = (json.dumps(output_raw, indent=2) + "\n").encode("utf-8")
    output_hash = hashlib.sha256(layout_bytes).hexdigest()

    acceptance = {
        "format_version": LAYOUT_ACCEPTANCE_VERSION,
        "video_id": draft.video_id,
        "source_video_sha256": checked["source_video_sha256"],
        "decision": {
            "approved": True,
            "reviewer_identity": identity,
            "reviewer_kind": reviewer_kind,
            "human_reviewed": human_reviewed,
        },
        "review_manifest": {
            "path": manifest_file.name,
            "sha256": checked["sha256"],
            "format_version": REVIEW_MANIFEST_VERSION,
        },
        "draft_layout": checked["draft_layout"],
        "review_artifacts": checked["review_artifacts"],
        "evidence_frames": checked["evidence_frames"],
        "output_layout": {
            "path": output_file.name,
            "sha256": output_hash,
            "reviewed_core_sha256": _reviewed_core_hash(output_raw),
        },
    }
    acceptance_bytes = (
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_new(output_file, layout_bytes)
    _atomic_write_new(acceptance_file, acceptance_bytes)
    return acceptance


def verify_layout_acceptance(
    layout_path: str | Path,
    layout: WildLayout,
    acceptance_path: str | Path,
    *,
    source_sha256: str,
    allow_timing_derivative: bool = False,
) -> dict[str, Any]:
    """Verify a reviewed layout (or its offset-only derivative) from local bytes."""

    layout_file = Path(layout_path)
    acceptance_file = Path(acceptance_path)
    acceptance = _load_json(acceptance_file, "layout acceptance")
    _exact_keys(
        acceptance,
        {
            "format_version",
            "video_id",
            "source_video_sha256",
            "decision",
            "review_manifest",
            "draft_layout",
            "review_artifacts",
            "evidence_frames",
            "output_layout",
        },
        "layout acceptance",
    )
    if acceptance.get("format_version") != LAYOUT_ACCEPTANCE_VERSION:
        raise ValueError("unsupported or legacy layout acceptance format_version")
    if acceptance.get("video_id") != layout.video_id:
        raise ValueError("layout acceptance video_id differs from layout")
    accepted_source = _sha256(
        acceptance.get("source_video_sha256"), "layout acceptance.source_video_sha256"
    )
    if accepted_source != _sha256(source_sha256, "source_sha256"):
        raise ValueError("layout acceptance is bound to a different source video")

    decision = _mapping(acceptance.get("decision"), "layout acceptance.decision")
    _exact_keys(
        decision,
        {"approved", "reviewer_identity", "reviewer_kind", "human_reviewed"},
        "layout acceptance.decision",
    )
    if decision.get("approved") is not True:
        raise ValueError("layout acceptance lacks explicit approval")
    identity = str(decision.get("reviewer_identity", "")).strip()
    kind = str(decision.get("reviewer_kind", "")).strip()
    if not identity or kind not in REVIEWER_KINDS:
        raise ValueError("layout acceptance reviewer identity/kind is invalid")
    human_reviewed = kind in HUMAN_REVIEWER_KINDS
    if decision.get("human_reviewed") is not human_reviewed:
        raise ValueError("layout human gate must be derived from reviewer_kind")
    if layout.human_reviewed is not human_reviewed:
        raise ValueError("layout human_reviewed differs from its verified reviewer kind")

    manifest_row = _mapping(
        acceptance.get("review_manifest"), "layout acceptance.review_manifest"
    )
    _exact_keys(
        manifest_row, {"path", "sha256", "format_version"},
        "layout acceptance.review_manifest",
    )
    if manifest_row.get("format_version") != REVIEW_MANIFEST_VERSION:
        raise ValueError("layout acceptance names a legacy review manifest")
    manifest_name = _local_name(
        manifest_row.get("path"), "layout acceptance.review_manifest.path"
    )
    manifest_file = acceptance_file.parent / manifest_name
    if _sha256(
        manifest_row.get("sha256"), "layout acceptance.review_manifest.sha256"
    ) != sha256_file(_regular_file(manifest_file, "review manifest")):
        raise ValueError("layout acceptance review-manifest hash mismatch")
    checked = validate_review_manifest(manifest_file)
    if checked["video_id"] != layout.video_id:
        raise ValueError("review manifest video_id differs from layout acceptance")
    if checked["source_video_sha256"] != accepted_source:
        raise ValueError("review manifest and layout acceptance name different source videos")
    if acceptance.get("draft_layout") != checked["draft_layout"]:
        raise ValueError("layout acceptance draft-layout binding differs from review manifest")
    if acceptance.get("review_artifacts") != checked["review_artifacts"]:
        raise ValueError("layout acceptance artifact bindings differ from review manifest")
    if acceptance.get("evidence_frames") != checked["evidence_frames"]:
        raise ValueError("layout acceptance frame bindings differ from review manifest")

    raw_layout = _load_json(layout_file, "reviewed layout")
    if WildLayout.from_dict(raw_layout) != layout:
        raise ValueError("layout object differs from the reviewed layout bytes")
    embedded = _mapping(
        raw_layout.get("layout_review_acceptance"),
        "layout.layout_review_acceptance",
    )
    _exact_keys(
        embedded,
        {
            "format_version",
            "artifact",
            "review_manifest_sha256",
            "source_video_sha256",
        },
        "layout.layout_review_acceptance",
    )
    if embedded.get("format_version") != LAYOUT_ACCEPTANCE_VERSION:
        raise ValueError("layout embeds an unsupported review acceptance")
    if embedded.get("artifact") != acceptance_file.name:
        raise ValueError("layout names a different review-acceptance artifact")
    if _sha256(
        embedded.get("review_manifest_sha256"),
        "layout.layout_review_acceptance.review_manifest_sha256",
    ) != checked["sha256"]:
        raise ValueError("layout embeds a different review-manifest hash")
    if _sha256(
        embedded.get("source_video_sha256"),
        "layout.layout_review_acceptance.source_video_sha256",
    ) != accepted_source:
        raise ValueError("layout embeds a different source-video hash")

    output = _mapping(acceptance.get("output_layout"), "layout acceptance.output_layout")
    _exact_keys(
        output, {"path", "sha256", "reviewed_core_sha256"},
        "layout acceptance.output_layout",
    )
    output_name = _local_name(output.get("path"), "layout acceptance.output_layout.path")
    output_hash = _sha256(output.get("sha256"), "layout acceptance.output_layout.sha256")
    core_hash = _sha256(
        output.get("reviewed_core_sha256"),
        "layout acceptance.output_layout.reviewed_core_sha256",
    )

    # Reconstruct the exact accepted output from the still-hash-bound draft.
    draft_raw = _load_json(Path(checked["draft_layout_path"]), "draft layout")
    expected_raw = dict(draft_raw)
    expected_raw["human_reviewed"] = human_reviewed
    expected_raw["layout_review_acceptance"] = embedded
    expected_bytes = (json.dumps(expected_raw, indent=2) + "\n").encode("utf-8")
    if hashlib.sha256(expected_bytes).hexdigest() != output_hash:
        raise ValueError("layout acceptance output hash is not derivable from its bound draft")
    if _reviewed_core_hash(expected_raw) != core_hash:
        raise ValueError("layout acceptance reviewed-core hash is inconsistent")

    if allow_timing_derivative:
        if _reviewed_core_hash(raw_layout) != core_hash:
            raise ValueError("timed layout changes fields outside the accepted review core")
    else:
        if layout_file.name != output_name:
            raise ValueError("layout acceptance names a different output layout")
        if sha256_file(layout_file) != output_hash:
            raise ValueError("layout acceptance is bound to different reviewed-layout bytes")

    return {
        "path": str(acceptance_file),
        "sha256": sha256_file(acceptance_file),
        "format_version": LAYOUT_ACCEPTANCE_VERSION,
        "review_manifest_path": str(manifest_file),
        "review_manifest_sha256": checked["sha256"],
        "reviewer_identity": identity,
        "reviewer_kind": kind,
        "human_reviewed": human_reviewed,
        "source_video_sha256": accepted_source,
        "draft_layout_sha256": checked["draft_layout"]["sha256"],
        "reviewed_layout_sha256": output_hash,
        "review_artifacts": checked["review_artifacts"],
        "evidence_frames": checked["evidence_frames"],
    }


def _parse_artifact(value: str) -> tuple[str, Path]:
    role, separator, path = value.partition("=")
    if not separator or not role or not path:
        raise argparse.ArgumentTypeError("artifact must be ROLE=PATH")
    return role, Path(path)


def _parse_frame(value: str) -> tuple[float, Path]:
    time_text, separator, path = value.partition("=")
    if not separator or not path:
        raise argparse.ArgumentTypeError("frame must be SECONDS=PATH")
    try:
        time_s = float(time_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frame seconds must be numeric") from exc
    return time_s, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--draft-layout", type=Path, required=True)
    manifest_parser.add_argument("--source-sha256", required=True)
    manifest_parser.add_argument("--artifact", type=_parse_artifact, action="append", required=True)
    manifest_parser.add_argument("--frame", type=_parse_frame, action="append", required=True)
    manifest_parser.add_argument("--out", type=Path, required=True)

    accept_parser = subparsers.add_parser("accept")
    accept_parser.add_argument("--review-manifest", type=Path, required=True)
    accept_parser.add_argument("--draft-layout", type=Path, required=True)
    accept_parser.add_argument("--output-layout", type=Path, required=True)
    accept_parser.add_argument("--acceptance-out", type=Path, required=True)
    accept_parser.add_argument("--reviewer", required=True)
    accept_parser.add_argument("--reviewer-kind", choices=REVIEWER_KINDS, required=True)
    accept_parser.add_argument("--approve", action="store_true", required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--layout", type=Path, required=True)
    verify_parser.add_argument("--acceptance", type=Path, required=True)
    verify_parser.add_argument("--source-sha256", required=True)
    verify_parser.add_argument("--allow-timing-derivative", action="store_true")

    args = parser.parse_args()
    if args.command == "manifest":
        artifact_items = dict(args.artifact)
        if len(artifact_items) != len(args.artifact):
            parser.error("artifact roles must be unique")
        result = write_review_manifest(
            args.out,
            args.draft_layout,
            source_sha256=args.source_sha256,
            artifacts=artifact_items,
            evidence_frames=args.frame,
        )
        summary = {
            "format_version": result["format_version"],
            "video_id": result["video_id"],
            "manifest": str(args.out),
            "review_artifacts": len(result["review_artifacts"]),
            "evidence_frames": len(result["evidence_frames"]),
        }
    elif args.command == "accept":
        result = accept_layout(
            args.review_manifest,
            args.draft_layout,
            args.output_layout,
            args.acceptance_out,
            reviewer_identity=args.reviewer,
            reviewer_kind=args.reviewer_kind,
            approved=args.approve,
        )
        summary = {
            "format_version": result["format_version"],
            "video_id": result["video_id"],
            "reviewer_kind": result["decision"]["reviewer_kind"],
            "human_reviewed": result["decision"]["human_reviewed"],
            "layout": str(args.output_layout),
            "acceptance": str(args.acceptance_out),
        }
    else:
        layout = WildLayout.load(args.layout)
        result = verify_layout_acceptance(
            args.layout,
            layout,
            args.acceptance,
            source_sha256=args.source_sha256,
            allow_timing_derivative=args.allow_timing_derivative,
        )
        summary = result
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
