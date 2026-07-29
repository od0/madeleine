"""Publish one completed AI-only family-transfer build, completion marker last.

The publisher has no discovery policy: every object is named by a validated
transfer, scan, decode, or provisional-build report. Existing remote objects
are verified and skipped, missing objects are copied with ``--immutable``, and
the completion marker is copied only after the payload inventory is exact.
Completed reruns are read-only validation passes.
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
import subprocess
import tempfile
from typing import Any

from harvest.build_wild import PART_COMPLETION_VERSION, PROVISIONAL_BUILD_VERSION
from harvest.decode_wild import DECODE_COMPLETION_NAME, DECODE_COMPLETION_VERSION
from harvest.fetch_wild import sha256_file
from harvest.transfer_wild_layout_family import EVIDENCE_VERSION
from harvest.wild_boundaries import WildBoundaries
from harvest.wild_layout import WildLayout


# Bucket identity is operational and stays out of Git (the contributor guide
# requires placeholders or environment variables for host and account
# identifiers). Real publishes set MADELEINE_R2_BUCKET_URI; the placeholder
# default keeps provenance fields structurally complete in dry contexts.
R2_BUCKET_URI = os.environ.get("MADELEINE_R2_BUCKET_URI", "r2:<bucket>")

PUBLICATION_VERSION = "madeleine.wild-family-transfer-provisional-publication.v2"
COMPLETION_VERSION = "madeleine.wild-family-transfer-provisional-complete.v2"
SCAN_VERSION = "madeleine.wild-cell-activity-scan.v1"
SCAN_VALIDATION_VERSION = "madeleine.wild-layout-family-scan-validation.v1"
DECODE_VERSION = "madeleine.wild-decode.v1"
MANIFEST_NAME = "publication-manifest.json"
COMPLETION_NAME = "publication-complete.json"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CADENCE_HZ = {
    "native60": (50.0, 61.0),
    "native30": (29.0, 31.0),
    "native24": (23.0, 25.0),
}
_SCAN_VALIDATION_MODES = {
    "absolute_luma_or_low_dynamic_binary_v1": frozenset(
        {"absolute_luma_gap", "low_dynamic_binary"}
    ),
    "absolute_luma_or_disjoint_stable_pressed_or_low_dynamic_binary_v2": frozenset(
        {
            "absolute_luma_gap",
            "disjoint_stable_pressed_state",
            "low_dynamic_binary",
        }
    ),
}


@dataclass(frozen=True)
class Artifact:
    kind: str
    local: Path
    relative_path: str
    sha256: str
    size_bytes: int
    npz_part: bool = False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_id(value: Any) -> str:
    result = str(value).strip()
    _require(_SAFE_ID.fullmatch(result) is not None, "unsafe video ID")
    return result


def _safe_relative(value: Any, field: str) -> str:
    result = str(value).strip()
    path = Path(result)
    _require(
        bool(result)
        and not path.is_absolute()
        and "\\" not in result
        and all(
            part not in ("", ".", "..") and _SAFE_COMPONENT.fullmatch(part)
            for part in path.parts
        ),
        f"{field} is not a safe relative path",
    )
    return "/".join(path.parts)


def _safe_name(value: Any, field: str) -> str:
    result = _safe_relative(value, field)
    _require("/" not in result, f"{field} must be a basename")
    return result


def _sha(value: Any, field: str) -> str:
    result = str(value).strip().lower()
    _require(re.fullmatch(r"[0-9a-f]{64}", result) is not None, f"invalid {field} SHA-256")
    return result


def _number(value: Any, field: str) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{field} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"{field} must be finite")
    return result


def _regular(path: Path, field: str) -> Path:
    _require(path.is_file() and not path.is_symlink(), f"missing regular {field}: {path}")
    return path


def _json(path: Path, field: str) -> dict[str, Any]:
    _regular(path, field)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {field}: {path}") from error
    _require(isinstance(value, dict), f"{field} must contain an object")
    return value


def _artifact(
    kind: str,
    local: Path,
    relative_path: str,
    *,
    expected_sha: Any | None = None,
    expected_size: Any | None = None,
    npz_part: bool = False,
) -> Artifact:
    _regular(local, kind)
    digest = sha256_file(local)
    size = local.stat().st_size
    if expected_sha is not None:
        _require(digest == _sha(expected_sha, kind), f"{kind} SHA-256 mismatch")
    if expected_size is not None:
        _require(size == int(expected_size), f"{kind} size mismatch")
    return Artifact(kind, local, _safe_relative(relative_path, kind), digest, size, npz_part)


def _declared_artifact(
    directory: Path, row: Any, field: str, *, remote_prefix: str = "scan"
) -> Artifact:
    _require(isinstance(row, dict), f"{field} must be an object")
    relative = _safe_relative(row.get("path"), f"{field}.path")
    return _artifact(
        field,
        directory.joinpath(*Path(relative).parts),
        f"{remote_prefix}/{relative}",
        expected_sha=row.get("sha256"),
        expected_size=row.get("size_bytes") if "size_bytes" in row else None,
    )


def _false_flags(record: dict[str, Any], field: str) -> None:
    _require(record.get("human_reviewed") is False, f"{field} must be human_reviewed=false")
    _require(record.get("training_admitted") is False, f"{field} must be training_admitted=false")


def collect_artifacts(
    work_root: str | Path, video_id: str, cadence_tier: str
) -> tuple[list[Artifact], dict[str, Any]]:
    """Validate the complete local chain and return its sole upload allowlist."""

    root = Path(work_root)
    video_id = _safe_id(video_id)
    _require(cadence_tier in _CADENCE_HZ, "unsupported cadence tier")
    transfer_dir = root / "transfer"
    scan_dir = root / "scan" / video_id
    decode_dir = root / "decode"
    parts_dir = root / "shards" / "parts"

    layout_path = transfer_dir / "layout.family-transfer-ai.json"
    spec_path = transfer_dir / "cell-scan-spec.family-transfer-ai.json"
    boundaries_path = transfer_dir / "boundaries.outer-ai.json"
    evidence_path = transfer_dir / "transfer_evidence.json"
    layout = WildLayout.load(_regular(layout_path, "family-transfer layout"))
    boundaries = WildBoundaries.load(_regular(boundaries_path, "outer boundaries"))
    spec = _json(spec_path, "family-transfer scan spec")
    evidence = _json(evidence_path, "family-transfer evidence")
    _require(evidence.get("format_version") == EVIDENCE_VERSION, "wrong transfer evidence version")
    _false_flags(evidence, "transfer evidence")
    _false_flags(spec, "scan spec")
    _require(layout.video_id == video_id and not layout.human_reviewed, "layout is not target AI-only layout")
    _require(layout.temporal_offset_source == "unmeasured", "family-transfer layout offset must remain unmeasured")
    _require(boundaries.video_id == video_id and not boundaries.human_reviewed, "boundaries are not target AI-only boundaries")
    _require(boundaries.reviewer_kind == "ai_agent", "family-transfer boundaries must be AI-only")
    _require(spec.get("video_id") == video_id, "scan spec video mismatch")
    _require(evidence.get("video_id") == video_id, "transfer evidence video mismatch")
    bindings = evidence.get("bindings")
    _require(isinstance(bindings, dict), "transfer evidence lacks bindings")
    source_sha = _sha(bindings.get("target_source_sha256"), "target source")
    pts_sha = _sha(bindings.get("target_pts_sha256"), "target PTS")
    survey_sha = _sha(bindings.get("target_survey_sha256"), "target survey")
    contact_sha = _sha(
        bindings.get("target_contact_sheet_sha256"),
        "target survey contact sheet",
    )
    reference_sha = _sha(bindings.get("reference_layout_sha256"), "reference layout")
    _require(
        spec.get("survey_sha256") == survey_sha,
        "scan spec survey binding differs from transfer evidence",
    )
    _require(
        spec.get("survey_contact_sheet_sha256") == contact_sha,
        "scan spec contact-sheet binding differs from transfer evidence",
    )
    _require(spec.get("source_sha256") == source_sha, "scan spec source binding mismatch")
    _require(spec.get("pts_sha256") == pts_sha, "scan spec PTS binding mismatch")
    _require(boundaries.source_sha256 == source_sha, "boundaries source binding mismatch")
    reference_path = transfer_dir / "reference-layout.source.json"
    _regular(reference_path, "source reference layout")
    _require(
        sha256_file(reference_path) == reference_sha,
        "source reference layout bytes differ from transfer evidence",
    )
    for key, path in (
        ("generated_layout_sha256", layout_path),
        ("generated_scan_spec_sha256", spec_path),
        ("generated_boundaries_sha256", boundaries_path),
    ):
        _require(_sha(bindings.get(key), key) == sha256_file(path), f"transfer {key} mismatch")

    scan_report_path = scan_dir / "cell_activity_scan.json"
    scan_validation_path = scan_dir / "family_transfer_scan_validation.json"
    scan_completion_path = scan_dir / "cell_activity_complete.json"
    scan = _json(scan_report_path, "full-cell scan report")
    scan_validation = _json(scan_validation_path, "family-transfer scan validation")
    scan_completion = _json(scan_completion_path, "cell-scan completion")
    _require(scan.get("format_version") == SCAN_VERSION, "wrong cell-scan version")
    _false_flags(scan, "cell-scan report")
    _false_flags(scan_validation, "cell-scan validation")
    _false_flags(scan_completion, "cell-scan completion")
    _require(scan.get("video_id") == video_id, "cell-scan video mismatch")
    _require(scan.get("source", {}).get("sha256") == source_sha, "cell-scan source mismatch")
    _require(scan.get("pts", {}).get("sha256") == pts_sha, "cell-scan PTS mismatch")
    scan_spec_row = scan.get("spec")
    _require(isinstance(scan_spec_row, dict), "cell-scan report lacks spec")
    copied_spec_name = _safe_name(scan_spec_row.get("path"), "cell-scan copied spec")
    copied_spec_path = scan_dir / copied_spec_name
    scan_spec_sha = _sha(scan_spec_row.get("sha256"), "cell-scan spec")
    _require(scan_spec_sha == sha256_file(spec_path), "cell-scan uses a different transfer spec")
    _artifact("cell-scan copied spec", copied_spec_path, f"scan/{copied_spec_name}", expected_sha=scan_spec_sha, expected_size=scan_spec_row.get("size_bytes"))
    _require(scan_validation.get("format_version") == SCAN_VALIDATION_VERSION, "wrong scan-validation version")
    _require(scan_validation.get("video_id") == video_id, "scan-validation video mismatch")
    _require(_sha(scan_validation.get("scan_report_sha256"), "scan-validation report") == sha256_file(scan_report_path), "scan-validation report binding mismatch")
    _require(_sha(scan_validation.get("layout_sha256"), "scan-validation layout") == sha256_file(layout_path), "scan-validation layout binding mismatch")
    _require(int(scan_validation.get("validated_cells", -1)) == len(layout.cells), "scan validation did not validate every layout cell")
    _require(sorted(scan_validation.get("validated_actions", [])) == sorted({cell.action for cell in layout.cells}), "scan validation action set mismatch")
    scan_cells = scan.get("cells")
    _require(
        isinstance(scan_cells, list)
        and len(scan_cells) == len(layout.cells)
        and {str(row.get("cell_id")) for row in scan_cells}
        == {cell.cell_id for cell in layout.cells}
        and all(row.get("changing") is True for row in scan_cells),
        "scan report lacks every unique changing layout cell",
    )
    validation_policy = scan_validation.get("validation_policy")
    if validation_policy is None:
        # Historical PWE/wHr v2 artifacts were already immutable when the
        # richer policy rows landed. Grandfather only the old absolute-gap
        # evidence, recomputed directly from the bound scan report.
        _require(
            float(scan_validation.get("minimum_cluster_separation_luma", 0.0))
            >= 20.0
            and all(
                float(row.get("cluster_separation_luma", 0.0)) >= 20.0
                for row in scan_cells
            ),
            "legacy scan validation lacks independently strong absolute gaps",
        )
    else:
        accepted_modes = _SCAN_VALIDATION_MODES.get(validation_policy)
        _require(
            accepted_modes is not None,
            "unsupported scan-validation policy",
        )
        validation_rows = scan_validation.get("cell_validation")
        _require(
            isinstance(validation_rows, list)
            and len(validation_rows) == len(layout.cells)
            and {str(row.get("cell_id")) for row in validation_rows}
            == {cell.cell_id for cell in layout.cells}
            and all(
                row.get("validation_mode") in accepted_modes
                for row in validation_rows
            ),
            "scan-validation policy lacks every accepted cell",
        )
    _require(scan_completion.get("video_id") == video_id, "scan completion video mismatch")
    _require(scan_completion.get("source_sha256") == source_sha, "scan completion source mismatch")
    _require(_sha(scan_completion.get("report_sha256"), "scan completion report") == sha256_file(scan_report_path), "scan completion report binding mismatch")

    score_artifact = _declared_artifact(scan_dir, scan.get("scores"), "cell scores")
    scan_artifacts = [
        _artifact("cell-scan report", scan_report_path, "scan/cell_activity_scan.json"),
        _artifact("cell-scan validation", scan_validation_path, "scan/family_transfer_scan_validation.json"),
        _artifact("cell-scan completion", scan_completion_path, "scan/cell_activity_complete.json"),
        _declared_artifact(scan_dir, scan_spec_row, "cell-scan copied spec"),
        score_artifact,
    ]
    evidence_rows = scan.get("evidence")
    _require(isinstance(evidence_rows, list), "cell-scan evidence must be a list")
    scan_artifacts.extend(
        _declared_artifact(scan_dir, row, f"cell-scan evidence {index}")
        for index, row in enumerate(evidence_rows)
    )
    if scan.get("evidence_contact_sheet") is not None:
        scan_artifacts.append(_declared_artifact(scan_dir, scan["evidence_contact_sheet"], "cell-scan contact sheet"))

    decode_report_path = decode_dir / "decode_report.json"
    decode_completion_path = decode_dir / DECODE_COMPLETION_NAME
    decode = _json(decode_report_path, "decode report")
    decode_completion = _json(decode_completion_path, "decode completion")
    _require(decode.get("format_version") == DECODE_VERSION, "wrong decode version")
    _require(decode.get("video_id") == video_id and decode.get("admitted") is False, "decode is not target provisional decode")
    _require(isinstance(decode.get("rejection_reasons"), list) and decode["rejection_reasons"], "provisional decode lacks rejection reasons")
    _require(decode.get("source_video", {}).get("sha256") == source_sha, "decode source binding mismatch")
    _require(decode.get("layout", {}).get("human_reviewed") is False, "decode layout incorrectly claims human review")
    _require(_sha(decode.get("layout", {}).get("sha256"), "decode layout") == sha256_file(layout_path), "decode layout binding mismatch")
    _require(_sha(decode.get("boundaries", {}).get("sha256"), "decode boundaries") == sha256_file(boundaries_path), "decode boundaries binding mismatch")
    _require(decode.get("boundaries", {}).get("human_reviewed") is False, "decode boundaries incorrectly claim human review")
    score_source = decode.get("score_source")
    _require(isinstance(score_source, dict) and score_source.get("kind") == "hash_bound_full_cell_scan", "decode lacks hash-bound full-cell scan provenance")
    _require(_sha(score_source.get("report_sha256"), "decode scan report") == sha256_file(scan_report_path), "decode scan-report binding mismatch")
    _require(_sha(score_source.get("spec_sha256"), "decode scan spec") == scan_spec_sha, "decode scan-spec binding mismatch")
    _require(_sha(score_source.get("scores_sha256"), "decode cell scores") == score_artifact.sha256, "decode score binding mismatch")
    raw_name = _safe_name(decode.get("raw_labels"), "raw labels")
    labels_name = _safe_name(decode.get("labels"), "native labels")
    raw_labels = _artifact("raw labels", decode_dir / raw_name, "decode/labels_raw.parquet", expected_sha=decode.get("raw_labels_sha256"))
    labels = _artifact("native labels", decode_dir / labels_name, "decode/labels_native.parquet", expected_sha=decode.get("labels_sha256"))
    _require(
        decode_completion.get("format_version") == DECODE_COMPLETION_VERSION
        and decode_completion.get("video_id") == video_id
        and decode_completion.get("admitted") is False,
        "decode completion identity/version mismatch",
    )
    completion_report = decode_completion.get("report")
    completion_artifacts = decode_completion.get("artifacts")
    completion_bindings = decode_completion.get("bindings")
    _require(
        isinstance(completion_report, dict)
        and completion_report.get("path") == decode_report_path.name
        and int(completion_report.get("size_bytes", -1))
        == decode_report_path.stat().st_size
        and completion_report.get("sha256") == sha256_file(decode_report_path),
        "decode completion report binding mismatch",
    )
    _require(
        isinstance(completion_artifacts, dict)
        and completion_artifacts.get("raw_labels", {}).get("path") == raw_name
        and completion_artifacts.get("raw_labels", {}).get("sha256")
        == raw_labels.sha256
        and completion_artifacts.get("raw_labels", {}).get("size_bytes")
        == raw_labels.size_bytes
        and completion_artifacts.get("labels", {}).get("path") == labels_name
        and completion_artifacts.get("labels", {}).get("sha256") == labels.sha256
        and completion_artifacts.get("labels", {}).get("size_bytes")
        == labels.size_bytes,
        "decode completion label binding mismatch",
    )
    _require(
        isinstance(completion_bindings, dict)
        and completion_bindings.get("source_sha256") == source_sha
        and completion_bindings.get("layout_sha256") == sha256_file(layout_path)
        and completion_bindings.get("boundaries_sha256")
        == sha256_file(boundaries_path)
        and completion_bindings.get("scan_report_sha256")
        == sha256_file(scan_report_path),
        "decode completion input binding mismatch",
    )

    build_report_path = parts_dir / "wild_provisional_build_report.json"
    build = _json(build_report_path, "provisional build report")
    _require(build.get("format_version") == PROVISIONAL_BUILD_VERSION, "wrong provisional build version")
    _require(build.get("video_id") == video_id, "build video mismatch")
    _require(build.get("label_kind") == "wild_overlay_provisional", "wrong provisional label kind")
    _require(build.get("admission_tier") == "provisional_not_train_ready", "wrong build admission tier")
    _require(build.get("timing_authority") == "presentation_timestamp", "wrong build timing authority")
    _require(int(build.get("train_ready_frames", -1)) == 0 and _number(build.get("train_ready_hours"), "train-ready hours") == 0.0, "provisional build claims train-ready data")
    provisional_frames = int(build.get("provisional_trainable_frames", 0))
    provisional_hours = _number(build.get("provisional_trainable_hours"), "provisional hours")
    _require(provisional_frames > 0 and provisional_hours > 0, "provisional build has no usable frames")
    fps = _number(build.get("effective_grid_hz"), "effective grid")
    low, high = _CADENCE_HZ[cadence_tier]
    _require(low <= fps <= high, f"effective grid {fps:.4f} does not match {cadence_tier}")
    _require(abs(provisional_frames / fps / 3600.0 - provisional_hours) < 1e-9, "provisional hours do not match frames/grid")
    inputs = build.get("inputs")
    _require(isinstance(inputs, dict), "build lacks input bindings")
    build_decode = inputs.get("decode_report")
    build_labels = inputs.get("labels")
    build_layout = inputs.get("layout")
    _require(isinstance(build_decode, dict) and _safe_name(build_decode.get("path"), "build decode path") == decode_report_path.name and _sha(build_decode.get("sha256"), "build decode") == sha256_file(decode_report_path), "build decode binding mismatch")
    _require(isinstance(build_labels, dict) and _safe_name(build_labels.get("path"), "build labels path") == labels_name and _sha(build_labels.get("sha256"), "build labels") == labels.sha256, "build label binding mismatch")
    _require(isinstance(build_layout, dict) and _safe_name(build_layout.get("path"), "build layout path") == layout_path.name and _sha(build_layout.get("sha256"), "build layout") == sha256_file(layout_path), "build layout binding mismatch")
    _require(_sha(inputs.get("source_video_sha256"), "build source") == source_sha, "build source binding mismatch")
    _require(_sha(inputs.get("boundaries_sha256"), "build boundaries") == sha256_file(boundaries_path), "build boundaries binding mismatch")
    build_scan_validation = inputs.get("scan_validation")
    low_dynamic_cells = {
        str(row.get("cell_id"))
        for row in scan_validation.get("cell_validation", [])
        if isinstance(row, dict) and row.get("validation_mode") == "low_dynamic_binary"
    }
    if low_dynamic_cells:
        _require(
            isinstance(build_scan_validation, dict),
            "low-dynamic build lacks scan-validation binding",
        )
    if build_scan_validation is not None:
        _require(
            isinstance(build_scan_validation, dict)
            and _safe_name(
                build_scan_validation.get("path"), "build scan-validation path"
            )
            == scan_validation_path.name
            and _sha(
                build_scan_validation.get("sha256"), "build scan validation"
            )
            == sha256_file(scan_validation_path),
            "build scan-validation binding mismatch",
        )
    implementation = build.get("implementation")
    _require(isinstance(implementation, dict) and implementation.get("module") == "harvest/build_wild.py", "build implementation binding missing")
    implementation_sha = _sha(implementation.get("sha256"), "builder implementation")

    parts = build.get("parts")
    _require(isinstance(parts, list) and parts, "provisional build has no parts")
    part_artifacts: list[Artifact] = []
    summed_frames = 0
    expected_bindings = {
        "implementation_sha256": implementation_sha,
        "source_video_sha256": source_sha,
        "labels_sha256": labels.sha256,
        "layout_sha256": sha256_file(layout_path),
        "boundaries_sha256": sha256_file(boundaries_path),
    }
    declared_part_names: set[str] = set()
    for index, part_row in enumerate(parts):
        _require(isinstance(part_row, dict), f"part {index} is not an object")
        session_id = f"wild_provisional_{video_id}__r{index:03d}"
        name = _safe_name(part_row.get("npz"), f"part {index} name")
        _require(part_row.get("session_id") == session_id and name == f"{session_id}.npz", "part IDs are not canonical and contiguous")
        _require(name not in declared_part_names, "duplicate build part")
        declared_part_names.add(name)
        frames = int(part_row.get("frames", 0))
        _require(frames > 0, "part frame count must be positive")
        summed_frames += frames
        part_path = parts_dir / name
        part = _artifact("NPZ part", part_path, f"shards/{name}", expected_sha=part_row.get("sha256"), npz_part=True)
        sidecar_path = part_path.with_name(part_path.name + ".complete.json")
        sidecar = _json(sidecar_path, f"part {index} sidecar")
        _require(sidecar.get("format_version") == PART_COMPLETION_VERSION, "wrong part-sidecar version")
        _require(sidecar.get("row") == part_row, "part sidecar row differs from build report")
        _require(int(sidecar.get("npz_bytes", -1)) == part.size_bytes, "part sidecar byte count mismatch")
        _require(sidecar.get("bindings") == expected_bindings, "part sidecar bindings mismatch")
        arrays = sidecar.get("arrays")
        _require(isinstance(arrays, dict), "part sidecar lacks array contract")
        _require(arrays.get("frames", {}).get("shape", [None])[0] == frames, "part sidecar frame shape mismatch")
        _require(arrays.get("keys", {}).get("shape") == [frames, 7], "part sidecar key shape mismatch")
        part_artifacts.extend((part, _artifact("NPZ part completion", sidecar_path, f"shards/{sidecar_path.name}")))
    _require(summed_frames == provisional_frames, "part frames do not sum to provisional frame total")

    corpus_manifest_path = parts_dir.parent / "wild_provisional_corpus_manifest.json"
    corpus = _json(corpus_manifest_path, "provisional corpus manifest")
    _require(corpus.get("format_version") == PROVISIONAL_BUILD_VERSION, "wrong provisional corpus version")
    _require(corpus.get("admission_tier") == "provisional_not_train_ready", "wrong corpus admission tier")
    _require(int(corpus.get("video_count", -1)) == 1 and corpus.get("videos") == [build], "per-video corpus manifest does not bind exactly this build")
    _require(_number(corpus.get("train_ready_hours"), "corpus train-ready hours") == 0.0, "provisional corpus claims train-ready hours")
    _require(abs(_number(corpus.get("provisional_trainable_hours"), "corpus provisional hours") - provisional_hours) < 1e-12, "corpus provisional hours mismatch")

    artifacts = [
        _artifact(
            "source reference layout",
            reference_path,
            "transfer/reference-layout.source.json",
            expected_sha=reference_sha,
        ),
        _artifact("family-transfer layout", layout_path, "transfer/layout.family-transfer-ai.json"),
        _artifact("family-transfer scan spec", spec_path, "transfer/cell-scan-spec.family-transfer-ai.json"),
        _artifact("outer AI boundaries", boundaries_path, "transfer/boundaries.outer-ai.json"),
        _artifact("family-transfer evidence", evidence_path, "transfer/transfer_evidence.json"),
        *scan_artifacts,
        _artifact("decode report", decode_report_path, "decode/decode_report.json"),
        _artifact(
            "decode completion",
            decode_completion_path,
            f"decode/{DECODE_COMPLETION_NAME}",
        ),
        raw_labels,
        labels,
        _artifact("provisional build report", build_report_path, "shards/wild_provisional_build_report.json"),
        _artifact("provisional corpus manifest", corpus_manifest_path, "shards/wild_provisional_corpus_manifest.json"),
        *part_artifacts,
    ]
    relative = [row.relative_path for row in artifacts]
    _require(len(relative) == len(set(relative)), "duplicate publication object path")
    return artifacts, build


def _remote_root(value: str) -> str:
    result = value.strip().rstrip("/")
    _require(result.count(":") == 1 and not any(char.isspace() for char in result), "invalid remote root")
    remote, suffix = result.split(":", 1)
    _require(_SAFE_COMPONENT.fullmatch(remote) is not None, "invalid remote name")
    if suffix:
        _safe_relative(suffix, "remote root suffix")
    return result


def _remote_inventory(remote_dir: str) -> dict[str, int]:
    result = subprocess.run(
        ["rclone", "lsjson", "--recursive", "--files-only", remote_dir],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError("rclone remote inventory failed")
    try:
        rows = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as error:
        raise RuntimeError("rclone returned invalid remote inventory") from error
    _require(isinstance(rows, list), "remote inventory is not a list")
    inventory: dict[str, int] = {}
    for row in rows:
        _require(isinstance(row, dict), "invalid remote inventory row")
        path = _safe_relative(row.get("Path"), "remote inventory path")
        _require(path not in inventory and int(row.get("Size", -1)) >= 0, "invalid duplicate/size in remote inventory")
        inventory[path] = int(row["Size"])
    return inventory


def _remote_sha256(remote_path: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(["rclone", "cat", remote_path], stdout=subprocess.PIPE, stderr=errors)
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
        returncode = process.wait()
    if returncode:
        raise RuntimeError("rclone remote hash readback failed")
    return digest.hexdigest(), size


def _remote_size(remote_path: str) -> int:
    result = subprocess.run(
        ["rclone", "size", remote_path, "--json"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError("rclone remote stat failed")
    try:
        row = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("rclone returned invalid remote stat") from error
    _require(
        isinstance(row, dict)
        and int(row.get("count", -1)) == 1
        and int(row.get("bytes", -1)) >= 0,
        "invalid remote stat",
    )
    return int(row["bytes"])


def _verify_remote(artifact: Artifact, remote_dir: str, *, npz_size_only: bool) -> None:
    if artifact.npz_part and npz_size_only:
        _require(
            _remote_size(f"{remote_dir}/{artifact.relative_path}")
            == artifact.size_bytes,
            f"remote NPZ size mismatch: {artifact.relative_path}",
        )
        return
    digest, size = _remote_sha256(f"{remote_dir}/{artifact.relative_path}")
    _require(digest == artifact.sha256 and size == artifact.size_bytes, f"remote hash/size mismatch: {artifact.relative_path}")


def _copy_missing(artifact: Artifact, remote_dir: str, *, npz_size_only: bool) -> None:
    _require(sha256_file(artifact.local) == artifact.sha256 and artifact.local.stat().st_size == artifact.size_bytes, f"local artifact changed before upload: {artifact.local}")
    result = subprocess.run(
        [
            "rclone", "copyto", str(artifact.local),
            f"{remote_dir}/{artifact.relative_path}", "--immutable", "--transfers", "1",
            "--retries", "5", "--low-level-retries", "10",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode:
        # A concurrent identical immutable publisher is safe only if readback proves it.
        try:
            _verify_remote(artifact, remote_dir, npz_size_only=npz_size_only)
            return
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"immutable upload failed: {artifact.relative_path}") from error
    _require(sha256_file(artifact.local) == artifact.sha256 and artifact.local.stat().st_size == artifact.size_bytes, f"local artifact changed during upload: {artifact.local}")
    _verify_remote(artifact, remote_dir, npz_size_only=npz_size_only)


def _write_resumable(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        _require(path.is_file() and not path.is_symlink() and path.read_bytes() == payload, f"refusing changed publication state: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"publication state appeared concurrently: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def publish(
    work_root: str | Path,
    video_id: str,
    state_dir: str | Path,
    remote_root: str,
    cadence_tier: str,
    *,
    npz_size_only: bool = False,
) -> dict[str, Any]:
    """Publish or idempotently validate one provisional video prefix."""

    video_id = _safe_id(video_id)
    artifacts, build = collect_artifacts(work_root, video_id, cadence_tier)
    transfer_bindings = _json(
        Path(work_root) / "transfer" / "transfer_evidence.json",
        "family-transfer evidence",
    )["bindings"]
    remote_dir = f"{_remote_root(remote_root)}/{cadence_tier}/{video_id}"
    state = Path(state_dir)
    _require(not state.is_symlink(), "publication state_dir may not be a symlink")
    verification = lambda row: (
        "remote_size_only; local_and_build_sha256_do_not_verify_remote_content"
        if row.npz_part and npz_size_only
        else "sha256_and_size_readback"
    )
    manifest = {
        "format_version": PUBLICATION_VERSION,
        "video_id": video_id,
        "cadence_tier": cadence_tier,
        "admission_tier": "provisional_not_train_ready",
        "provisional_trainable_frames": int(build["provisional_trainable_frames"]),
        "provisional_trainable_hours": float(build["provisional_trainable_hours"]),
        "objects": [
            {
                "kind": row.kind,
                "path": row.relative_path,
                "size_bytes": row.size_bytes,
                "sha256": row.sha256,
                "remote_verification": verification(row),
            }
            for row in artifacts
        ],
        "object_count": len(artifacts),
        "total_bytes": sum(row.size_bytes for row in artifacts),
        "reconstruction_sources": {
            "raw_r2_prefix": f"{R2_BUCKET_URI}/wild/v1/raw/{video_id}",
            "layout_survey_r2_prefix": (
                f"{R2_BUCKET_URI}/wild/v1/layout-surveys-ai-v1/{video_id}"
            ),
            "reference_layout_publication_path": (
                "transfer/reference-layout.source.json"
            ),
            "target_source_sha256": build["inputs"]["source_video_sha256"],
            "target_pts_sha256": transfer_bindings["target_pts_sha256"],
            "target_survey_sha256": transfer_bindings["target_survey_sha256"],
            "target_contact_sheet_sha256": (
                transfer_bindings["target_contact_sheet_sha256"]
            ),
            "reference_layout_sha256": (
                transfer_bindings["reference_layout_sha256"]
            ),
        },
        "human_reviewed": False,
        "training_admitted": False,
    }
    manifest_path = state / MANIFEST_NAME
    _write_resumable(manifest_path, manifest)
    manifest_artifact = _artifact("publication manifest", manifest_path, MANIFEST_NAME)
    payload = [*artifacts, manifest_artifact]
    expected_payload = {row.relative_path: row.size_bytes for row in payload}
    completion = {
        "format_version": COMPLETION_VERSION,
        "video_id": video_id,
        "cadence_tier": cadence_tier,
        "remote_dir": remote_dir,
        "admission_tier": "provisional_not_train_ready",
        "manifest_sha256": manifest_artifact.sha256,
        "payload_objects": len(payload),
        "payload_bytes": sum(expected_payload.values()),
        "npz_verification": (
            "remote_size_only; local_and_build_sha256_do_not_verify_remote_content"
            if npz_size_only
            else "sha256_and_size_readback"
        ),
        "completion_policy": "marker uploaded last after exact payload inventory",
        "human_reviewed": False,
        "training_admitted": False,
    }
    completion_path = state / COMPLETION_NAME
    _write_resumable(completion_path, completion)
    marker = _artifact("publication completion", completion_path, COMPLETION_NAME)

    observed = _remote_inventory(remote_dir)
    allowed = set(expected_payload) | {COMPLETION_NAME}
    unexpected = set(observed) - allowed
    _require(not unexpected, f"remote prefix contains unexpected objects: {sorted(unexpected)}")
    if COMPLETION_NAME in observed:
        expected_complete = {**expected_payload, COMPLETION_NAME: marker.size_bytes}
        _require(observed == expected_complete, "completed remote inventory is incomplete or changed")
        for artifact in payload:
            _verify_remote(artifact, remote_dir, npz_size_only=npz_size_only)
        _verify_remote(marker, remote_dir, npz_size_only=False)
        return {**completion, "publication_status": "already_complete_validated"}

    for artifact in payload:
        if artifact.relative_path in observed:
            _require(observed[artifact.relative_path] == artifact.size_bytes, f"existing remote size mismatch: {artifact.relative_path}")
            _verify_remote(artifact, remote_dir, npz_size_only=npz_size_only)
        else:
            _copy_missing(artifact, remote_dir, npz_size_only=npz_size_only)
    _require(_remote_inventory(remote_dir) == expected_payload, "remote payload inventory differs before completion")
    _copy_missing(marker, remote_dir, npz_size_only=False)
    expected_complete = {**expected_payload, COMPLETION_NAME: marker.size_bytes}
    _require(_remote_inventory(remote_dir) == expected_complete, "remote inventory differs after completion")
    return {**completion, "publication_status": "published"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--cadence-tier", choices=tuple(_CADENCE_HZ), required=True)
    parser.add_argument(
        "--npz-size-only",
        action="store_true",
        help="verify large NPZ remotely by exact size; local/build/sidecar SHA-256 remains mandatory",
    )
    args = parser.parse_args()
    print(json.dumps(publish(
        args.work_root,
        args.video_id,
        args.state_dir,
        args.remote_root,
        args.cadence_tier,
        npz_size_only=args.npz_size_only,
    ), indent=2))


if __name__ == "__main__":
    main()
