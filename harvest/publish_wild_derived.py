"""Publish one train-ready wild video's derived artifacts with SHA readback.

This publisher accepts only the named artifact classes in the v1 wild
pipeline.  It never walks a directory to decide what to upload: decoded files
are named by the decode report and shard files are named by the build report.
Directory enumeration is used only as a fail-closed stale-file check.

Every local object is hashed into a deterministic manifest, copied with
``rclone --immutable``, streamed back through ``rclone cat`` for SHA-256 and
size verification, and listed in a completion marker that is copied last.
Credentials are neither discovered nor configured by this module.  Aggregate
corpus publication is intentionally a separate operation.
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

from harvest.accept_wild_layout import validate_review_manifest, verify_layout_acceptance
from harvest.accept_wild_offset import verify_offset_acceptance
from harvest.fetch_wild import sha256_file
from harvest.wild_boundaries import WildBoundaries
from harvest.wild_layout import WildLayout


PUBLICATION_VERSION = "madeleine.wild-derived-publication.v1"
DECODE_VERSION = "madeleine.wild-decode.v1"
BUILD_VERSION = "madeleine.wild-shards.v1"
MANIFEST_NAME = "derived_objects.json"
COMPLETION_NAME = "derived_complete.json"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class Artifact:
    kind: str
    local: Path
    relative_path: str


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _sha256(value: Any, field: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return result


def _safe_component(value: Any, field: str) -> str:
    result = str(value).strip()
    if not _SAFE_COMPONENT.fullmatch(result) or result in (".", ".."):
        raise ValueError(f"{field} is not a safe file name")
    return result


def _safe_relative(value: Any, field: str) -> str:
    result = str(value).strip()
    path = Path(result)
    if (
        not result
        or path.is_absolute()
        or "\\" in result
        or any(part in ("", ".", "..") for part in path.parts)
        or any(not _SAFE_COMPONENT.fullmatch(part) for part in path.parts)
    ):
        raise ValueError(f"{field} is not a safe relative object path")
    return "/".join(path.parts)


def _remote_root(value: str) -> str:
    result = value.strip().rstrip("/")
    if not result or any(char.isspace() or ord(char) < 32 for char in result):
        raise ValueError("remote_root contains unsafe whitespace/control characters")
    if result.count(":") != 1:
        raise ValueError("remote_root must be one rclone remote path")
    remote, suffix = result.split(":", 1)
    if not _SAFE_COMPONENT.fullmatch(remote):
        raise ValueError("remote_root has an unsafe remote name")
    if suffix:
        _safe_relative(suffix, "remote_root suffix")
    return result


def _regular_file(path: str | Path, field: str) -> Path:
    result = Path(path)
    if result.is_symlink() or not result.is_file():
        raise ValueError(f"{field} must be an existing regular non-symlink file")
    return result


def _same_path(actual: Path, expected: Path, field: str) -> None:
    if actual.absolute() != expected.absolute():
        raise ValueError(f"{field} is not the exact path named by its report")


def _exact_directory(directory: Path, expected_names: set[str], field: str) -> None:
    actual: set[str] = set()
    for entry in directory.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"{field} contains a non-regular or nested entry: {entry.name}")
        actual.add(entry.name)
    if actual != expected_names:
        missing = sorted(expected_names - actual)
        stale = sorted(actual - expected_names)
        raise ValueError(
            f"{field} does not exactly match its allowlist; missing={missing}, stale={stale}"
        )


def _artifact(kind: str, local: Path, relative_path: str) -> Artifact:
    return Artifact(kind, _regular_file(local, kind), _safe_relative(relative_path, kind))


def collect_artifacts(
    *,
    video_id: str,
    input_layout_path: str | Path,
    layout_acceptance_path: str | Path,
    layout_path: str | Path,
    boundaries_path: str | Path,
    calibration_path: str | Path,
    calibration_sha256_path: str | Path,
    contact_sheet_path: str | Path,
    acceptance_path: str | Path,
    decode_report_path: str | Path,
    labels_raw_path: str | Path,
    labels_native_path: str | Path,
    build_report_path: str | Path,
    shard_dir: str | Path,
) -> list[Artifact]:
    """Validate cross-artifact bindings and return the sole upload allowlist."""

    if not _SAFE_ID.fullmatch(video_id):
        raise ValueError("video_id contains unsafe characters")
    input_layout_file = _regular_file(input_layout_path, "reviewed unmeasured layout")
    layout_acceptance_file = _regular_file(
        layout_acceptance_path, "layout review acceptance"
    )
    layout_file = _regular_file(layout_path, "final layout")
    boundaries_file = _regular_file(boundaries_path, "boundaries")
    calibration_file = _regular_file(calibration_path, "calibration")
    calibration_sha_file = _regular_file(
        calibration_sha256_path, "calibration SHA-256 sidecar"
    )
    contact_file = _regular_file(contact_sheet_path, "offset contact sheet")
    acceptance_file = _regular_file(acceptance_path, "offset acceptance")
    decode_file = _regular_file(decode_report_path, "decode report")
    raw_labels_file = _regular_file(labels_raw_path, "raw labels")
    native_labels_file = _regular_file(labels_native_path, "native labels")
    build_file = _regular_file(build_report_path, "build report")
    shards = Path(shard_dir)
    if shards.is_symlink() or not shards.is_dir():
        raise ValueError("shard_dir must be an existing non-symlink directory")

    input_layout = WildLayout.load(input_layout_file)
    layout = WildLayout.load(layout_file)
    boundaries = WildBoundaries.load(boundaries_file)
    if (
        input_layout.video_id != video_id
        or layout.video_id != video_id
        or boundaries.video_id != video_id
    ):
        raise ValueError("layout/boundaries video_id differs from requested video")
    if not input_layout.human_reviewed or input_layout.temporal_offset_source != "unmeasured":
        raise ValueError("calibration input layout must be reviewed and explicitly unmeasured")
    if not boundaries.human_reviewed:
        raise ValueError("AI-only gameplay boundaries are not train-ready publication evidence")

    decode = _mapping(json.loads(decode_file.read_text()), "decode report")
    if decode.get("format_version") != DECODE_VERSION:
        raise ValueError("unsupported decode report format_version")
    if decode.get("video_id") != video_id:
        raise ValueError("decode report video_id differs from requested video")
    if decode.get("admitted") is not True or decode.get("rejection_reasons") != []:
        raise ValueError("only an admitted final decode with no rejections may be published")
    source = _mapping(decode.get("source_video"), "decode.source_video")
    source_hash = _sha256(source.get("sha256"), "decode.source_video.sha256")
    if boundaries.source_sha256 != source_hash:
        raise ValueError("boundaries and decode report name different source videos")
    verified_layout_review = verify_layout_acceptance(
        input_layout_file,
        input_layout,
        layout_acceptance_file,
        source_sha256=source_hash,
    )
    if not verified_layout_review["human_reviewed"]:
        raise ValueError("AI-only layout review is not train-ready publication evidence")
    layout_row = _mapping(decode.get("layout"), "decode.layout")
    if _sha256(layout_row.get("sha256"), "decode.layout.sha256") != sha256_file(layout_file):
        raise ValueError("decode report is bound to a different layout")
    decode_layout_review = _mapping(
        layout_row.get("review_acceptance"), "decode.layout.review_acceptance"
    )
    if _sha256(
        decode_layout_review.get("sha256"), "decode layout-review acceptance SHA-256"
    ) != sha256_file(layout_acceptance_file):
        raise ValueError("decode report is bound to a different layout review acceptance")
    boundary_row = _mapping(decode.get("boundaries"), "decode.boundaries")
    if _sha256(boundary_row.get("sha256"), "decode.boundaries.sha256") != sha256_file(
        boundaries_file
    ):
        raise ValueError("decode report is bound to different boundaries")

    raw_name = _safe_component(decode.get("raw_labels"), "decode.raw_labels")
    native_name = _safe_component(decode.get("labels"), "decode.labels")
    _same_path(raw_labels_file, decode_file.parent / raw_name, "raw labels")
    _same_path(native_labels_file, decode_file.parent / native_name, "native labels")
    if _sha256(decode.get("raw_labels_sha256"), "decode.raw_labels_sha256") != sha256_file(
        raw_labels_file
    ):
        raise ValueError("raw-label bytes differ from decode report")
    if _sha256(decode.get("labels_sha256"), "decode.labels_sha256") != sha256_file(
        native_labels_file
    ):
        raise ValueError("native-label bytes differ from decode report")
    _exact_directory(
        decode_file.parent,
        {decode_file.name, raw_labels_file.name, native_labels_file.name},
        "decoded artifact directory",
    )

    timing = _mapping(decode.get("timing"), "decode.timing")
    acceptance_summary = _mapping(
        timing.get("offset_acceptance"), "decode.timing.offset_acceptance"
    )
    if _sha256(acceptance_summary.get("sha256"), "offset_acceptance.sha256") != sha256_file(
        acceptance_file
    ):
        raise ValueError("decode report is bound to a different offset acceptance")
    verified_acceptance = verify_offset_acceptance(
        layout_file,
        layout,
        acceptance_file,
        source_sha256=source_hash,
        layout_acceptance_path=layout_acceptance_file,
    )
    if not verified_acceptance["human_reviewed"]:
        raise ValueError("AI-only offset review is not train-ready publication evidence")
    acceptance = _mapping(json.loads(acceptance_file.read_text()), "offset acceptance")
    acceptance_input = _mapping(
        acceptance.get("input_layout"), "offset acceptance.input_layout"
    )
    if _sha256(
        acceptance_input.get("sha256"), "offset acceptance input-layout SHA-256"
    ) != sha256_file(input_layout_file):
        raise ValueError("offset acceptance is bound to a different input layout")
    if _safe_component(
        acceptance_input.get("path"), "offset acceptance input-layout path"
    ) != input_layout_file.name:
        raise ValueError("offset acceptance names a different input layout")

    layout_acceptance = _mapping(
        json.loads(layout_acceptance_file.read_text()), "layout acceptance"
    )
    review_manifest_row = _mapping(
        layout_acceptance.get("review_manifest"), "layout acceptance.review_manifest"
    )
    review_manifest_name = _safe_component(
        review_manifest_row.get("path"), "layout review manifest path"
    )
    review_manifest_file = _regular_file(
        layout_acceptance_file.parent / review_manifest_name, "layout review manifest"
    )
    reviewed_packet = validate_review_manifest(review_manifest_file)

    calibration = _mapping(json.loads(calibration_file.read_text()), "calibration")
    handoff = _mapping(calibration.get("human_handoff"), "calibration.human_handoff")
    contact_name = _safe_component(handoff.get("contact_sheet"), "contact sheet")
    _same_path(contact_file, calibration_file.parent / contact_name, "contact sheet")
    _same_path(
        calibration_sha_file,
        calibration_file.with_suffix(".sha256"),
        "calibration SHA-256 sidecar",
    )
    _same_path(acceptance_file, calibration_file.parent / acceptance_file.name, "acceptance")
    _exact_directory(
        calibration_file.parent,
        {
            calibration_file.name,
            calibration_sha_file.name,
            contact_file.name,
            acceptance_file.name,
        },
        "calibration artifact directory",
    )

    build = _mapping(json.loads(build_file.read_text()), "build report")
    if build.get("format_version") != BUILD_VERSION:
        raise ValueError("unsupported build report format_version")
    if build.get("video_id") != video_id:
        raise ValueError("build report video_id differs from requested video")
    if _number(build.get("train_ready_hours"), "build.train_ready_hours") <= 0:
        raise ValueError("build report has no train-ready hours")
    build_inputs = _mapping(build.get("inputs"), "build.inputs")
    build_decode = _mapping(build_inputs.get("decode_report"), "build.inputs.decode_report")
    if _sha256(build_decode.get("sha256"), "build decode SHA-256") != sha256_file(decode_file):
        raise ValueError("build report is bound to a different decode report")
    if _safe_component(build_decode.get("path"), "build decode path") != decode_file.name:
        raise ValueError("build report names a different decode report")
    build_labels = _mapping(build_inputs.get("labels"), "build.inputs.labels")
    if _safe_component(build_labels.get("path"), "build labels path") != native_labels_file.name:
        raise ValueError("build report names different native labels")
    if _sha256(build_labels.get("sha256"), "build labels SHA-256") != sha256_file(
        native_labels_file
    ):
        raise ValueError("build report is bound to different native labels")
    if _sha256(build_inputs.get("boundaries_sha256"), "build boundaries SHA-256") != sha256_file(
        boundaries_file
    ):
        raise ValueError("build report is bound to different boundaries")
    if _sha256(build_inputs.get("source_video_sha256"), "build source SHA-256") != source_hash:
        raise ValueError("build report is bound to a different source video")
    build_layout = _mapping(build_inputs.get("layout"), "build.inputs.layout")
    if _safe_component(build_layout.get("path"), "build layout path") != layout_file.name:
        raise ValueError("build report names a different layout")
    if _sha256(build_layout.get("sha256"), "build layout SHA-256") != sha256_file(
        layout_file
    ):
        raise ValueError("build report is bound to a different layout")

    parts = _sequence(build.get("parts"), "build.parts")
    if not parts:
        raise ValueError("build report contains no shard parts")
    part_artifacts: list[Artifact] = []
    part_names: set[str] = set()
    total_frames = 0
    for index, raw in enumerate(parts):
        part = _mapping(raw, f"build.parts[{index}]")
        expected_session = f"wild_{video_id}__r{index:03d}"
        if part.get("session_id") != expected_session:
            raise ValueError("build shard session IDs are not canonical and contiguous")
        name = _safe_component(part.get("npz"), f"build.parts[{index}].npz")
        if name != f"{expected_session}.npz" or name in part_names:
            raise ValueError("build shard file names are not canonical and unique")
        part_names.add(name)
        local = _regular_file(shards / name, f"build shard {name}")
        if _sha256(part.get("sha256"), f"build.parts[{index}].sha256") != sha256_file(local):
            raise ValueError(f"build shard hash mismatch: {name}")
        frames = _integer(part.get("frames"), f"build.parts[{index}].frames")
        if frames <= 0:
            raise ValueError("build shard frames must be positive")
        total_frames += frames
        part_artifacts.append(_artifact("shard_npz", local, f"shards/{name}"))
    if total_frames != _integer(build.get("train_ready_frames"), "build.train_ready_frames"):
        raise ValueError("build part frames do not sum to train_ready_frames")
    _same_path(build_file, shards / build_file.name, "build report")
    _exact_directory(shards, {build_file.name, *part_names}, "shard artifact directory")

    review_packet_artifacts = [
        _artifact(
            "layout_review_draft",
            Path(reviewed_packet["draft_layout_path"]),
            f"layout/review_packet/{reviewed_packet['draft_layout']['path']}",
        ),
        _artifact(
            "layout_review_manifest",
            review_manifest_file,
            f"layout/review_packet/{review_manifest_file.name}",
        ),
        _artifact(
            "layout_review_acceptance",
            layout_acceptance_file,
            f"layout/review_packet/{layout_acceptance_file.name}",
        ),
    ]
    for row in reviewed_packet["review_artifacts"]:
        review_packet_artifacts.append(_artifact(
            f"layout_review_{row['role']}",
            review_manifest_file.parent.joinpath(*Path(row["path"]).parts),
            f"layout/review_packet/{row['path']}",
        ))
    for row in reviewed_packet["evidence_frames"]:
        review_packet_artifacts.append(_artifact(
            "layout_review_evidence_frame",
            review_manifest_file.parent.joinpath(*Path(row["path"]).parts),
            f"layout/review_packet/{row['path']}",
        ))

    artifacts = [
        *review_packet_artifacts,
        _artifact(
            "reviewed_unmeasured_layout",
            input_layout_file,
            "layout/reviewed_unmeasured.json",
        ),
        _artifact("final_layout", layout_file, "layout/final.json"),
        _artifact("reviewed_boundaries", boundaries_file, "boundaries/boundaries.json"),
        _artifact("offset_calibration", calibration_file, "calibration/offset_calibration.json"),
        _artifact(
            "offset_calibration_sha256",
            calibration_sha_file,
            "calibration/offset_calibration.sha256",
        ),
        _artifact("offset_contact_sheet", contact_file, "calibration/dash_offset_contact.png"),
        _artifact("offset_acceptance", acceptance_file, "calibration/offset_acceptance.json"),
        _artifact("decode_report", decode_file, "decoded/decode_report.json"),
        _artifact("raw_labels", raw_labels_file, "decoded/labels_raw.parquet"),
        _artifact("native_labels", native_labels_file, "decoded/labels_native.parquet"),
        _artifact("build_report", build_file, "shards/wild_build_report.json"),
        *part_artifacts,
    ]
    relative_paths = [row.relative_path for row in artifacts]
    local_paths = [row.local.absolute() for row in artifacts]
    if len(relative_paths) != len(set(relative_paths)) or len(local_paths) != len(set(local_paths)):
        raise ValueError("derived artifact allowlist contains duplicate paths")
    return artifacts


def _atomic_write_new(path: Path, payload: bytes) -> None:
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
            raise FileExistsError(f"refusing to overwrite publication state: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _remote_files(remote_dir: str) -> set[str]:
    result = subprocess.run(
        ["rclone", "lsf", remote_dir, "--recursive", "--files-only"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError("rclone remote inventory failed")
    names = set()
    for line in result.stdout.splitlines():
        if line.strip():
            names.add(_safe_relative(line.strip(), "remote inventory object"))
    return names


def _remote_sha256(remote_path: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            ["rclone", "cat", remote_path], stdout=subprocess.PIPE, stderr=errors
        )
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            count += len(chunk)
        return_code = process.wait()
    if return_code:
        raise RuntimeError("rclone remote SHA-256 readback failed")
    return digest.hexdigest(), count


def _copy_verified(
    local: Path,
    remote: str,
    *,
    expected_sha256: str,
    expected_size: int,
) -> dict[str, Any]:
    if local.stat().st_size != expected_size or sha256_file(local) != expected_sha256:
        raise ValueError(f"local artifact changed before upload: {local.name}")
    result = subprocess.run(
        [
            "rclone", "copyto", str(local), remote,
            "--immutable", "--size-only", "--transfers", "1",
            "--retries", "5", "--low-level-retries", "10",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode:
        raise RuntimeError("immutable rclone copy failed")
    remote_hash, remote_size = _remote_sha256(remote)
    if local.stat().st_size != expected_size or sha256_file(local) != expected_sha256:
        raise ValueError(f"local artifact changed during upload: {local.name}")
    if remote_size != expected_size or remote_hash != expected_sha256:
        raise ValueError(f"remote SHA-256 readback mismatch: {local.name}")
    return {
        "relative_path": remote.rsplit("/", 1)[-1],
        "size_bytes": expected_size,
        "sha256": expected_sha256,
        "verified": "sha256_readback",
    }


def publish_derived(
    *,
    video_id: str,
    input_layout_path: str | Path,
    layout_acceptance_path: str | Path,
    layout_path: str | Path,
    boundaries_path: str | Path,
    calibration_path: str | Path,
    calibration_sha256_path: str | Path,
    contact_sheet_path: str | Path,
    acceptance_path: str | Path,
    decode_report_path: str | Path,
    labels_raw_path: str | Path,
    labels_native_path: str | Path,
    build_report_path: str | Path,
    shard_dir: str | Path,
    state_dir: str | Path,
    remote_root: str,
) -> dict[str, Any]:
    """Publish exactly one video's validated derived allowlist."""

    state = Path(state_dir)
    if state.exists():
        raise FileExistsError("refusing to reuse publication state_dir")
    artifacts = collect_artifacts(
        video_id=video_id,
        input_layout_path=input_layout_path,
        layout_acceptance_path=layout_acceptance_path,
        layout_path=layout_path,
        boundaries_path=boundaries_path,
        calibration_path=calibration_path,
        calibration_sha256_path=calibration_sha256_path,
        contact_sheet_path=contact_sheet_path,
        acceptance_path=acceptance_path,
        decode_report_path=decode_report_path,
        labels_raw_path=labels_raw_path,
        labels_native_path=labels_native_path,
        build_report_path=build_report_path,
        shard_dir=shard_dir,
    )
    state_resolved = state.resolve()
    protected_directories = {
        Path(layout_acceptance_path).parent.resolve(),
        Path(calibration_path).parent.resolve(),
        Path(decode_report_path).parent.resolve(),
        Path(shard_dir).resolve(),
    }
    if any(
        state_resolved == protected
        or state_resolved.is_relative_to(protected)
        or protected.is_relative_to(state_resolved)
        for protected in protected_directories
    ):
        raise ValueError("publication state_dir may not overlap an artifact directory")
    root = _remote_root(remote_root)
    remote_dir = f"{root}/{video_id}"
    object_rows = [
        {
            "kind": artifact.kind,
            "relative_path": artifact.relative_path,
            "source_name": artifact.local.name,
            "size_bytes": artifact.local.stat().st_size,
            "sha256": sha256_file(artifact.local),
        }
        for artifact in artifacts
    ]
    manifest = {
        "format_version": PUBLICATION_VERSION,
        "video_id": video_id,
        "objects": object_rows,
        "object_count": len(object_rows),
        "total_bytes": sum(row["size_bytes"] for row in object_rows),
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    state.mkdir(parents=True)
    manifest_path = state / MANIFEST_NAME
    _atomic_write_new(manifest_path, manifest_bytes)
    manifest_row = {
        "kind": "object_manifest",
        "relative_path": MANIFEST_NAME,
        "source_name": MANIFEST_NAME,
        "size_bytes": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
    }
    publish_rows = [*object_rows, manifest_row]
    artifact_by_relative = {row.relative_path: row.local for row in artifacts}
    artifact_by_relative[MANIFEST_NAME] = manifest_path

    expected_remote = {row["relative_path"] for row in publish_rows} | {COMPLETION_NAME}
    existing = _remote_files(remote_dir)
    if COMPLETION_NAME in existing:
        raise FileExistsError("remote derived publication is already complete")
    unexpected = existing - expected_remote
    if unexpected:
        raise ValueError(f"remote derived prefix contains stale objects: {sorted(unexpected)}")

    verified = []
    for row in publish_rows:
        relative = row["relative_path"]
        result = _copy_verified(
            artifact_by_relative[relative],
            f"{remote_dir}/{relative}",
            expected_sha256=row["sha256"],
            expected_size=row["size_bytes"],
        )
        result["kind"] = row["kind"]
        result["relative_path"] = relative
        verified.append(result)

    completion = {
        "format_version": PUBLICATION_VERSION,
        "video_id": video_id,
        "remote_dir": remote_dir,
        "verification": "every named object SHA-256 hashed through rclone cat",
        "manifest_sha256": manifest_row["sha256"],
        "objects": verified,
        "object_count": len(verified),
        "total_bytes": sum(row["size_bytes"] for row in verified),
    }
    completion_path = state / COMPLETION_NAME
    _atomic_write_new(
        completion_path, (json.dumps(completion, indent=2) + "\n").encode("utf-8")
    )
    marker_hash = sha256_file(completion_path)
    marker_size = completion_path.stat().st_size
    _copy_verified(
        completion_path,
        f"{remote_dir}/{COMPLETION_NAME}",
        expected_sha256=marker_hash,
        expected_size=marker_size,
    )
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--input-layout", type=Path, required=True)
    parser.add_argument("--layout-acceptance", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--offset-calibration", type=Path, required=True)
    parser.add_argument("--offset-calibration-sha256", type=Path, required=True)
    parser.add_argument("--offset-contact-sheet", type=Path, required=True)
    parser.add_argument("--offset-acceptance", type=Path, required=True)
    parser.add_argument("--decode-report", type=Path, required=True)
    parser.add_argument("--labels-raw", type=Path, required=True)
    parser.add_argument("--labels-native", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    args = parser.parse_args()
    result = publish_derived(
        video_id=args.video_id,
        input_layout_path=args.input_layout,
        layout_acceptance_path=args.layout_acceptance,
        layout_path=args.layout,
        boundaries_path=args.boundaries,
        calibration_path=args.offset_calibration,
        calibration_sha256_path=args.offset_calibration_sha256,
        contact_sheet_path=args.offset_contact_sheet,
        acceptance_path=args.offset_acceptance,
        decode_report_path=args.decode_report,
        labels_raw_path=args.labels_raw,
        labels_native_path=args.labels_native,
        build_report_path=args.build_report,
        shard_dir=args.shard_dir,
        state_dir=args.state_dir,
        remote_root=args.remote_root,
    )
    print(json.dumps({
        "video_id": result["video_id"],
        "remote_dir": result["remote_dir"],
        "object_count": result["object_count"],
        "total_bytes": result["total_bytes"],
        "completion": COMPLETION_NAME,
    }, indent=2))


if __name__ == "__main__":
    main()
