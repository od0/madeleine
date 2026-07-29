"""Publish an AI-only existing-layout provisional build without invented provenance.

The source video remains in the immutable raw corpus; this prefix contains its
hash-bound fetch/PTS evidence, the exact AI-only layout and boundaries, decoded
labels, and provisional shard payload. Publication is missing-only and the
completion marker is always uploaded last. A completed rerun is validation-only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from harvest.build_wild import PART_COMPLETION_VERSION, PROVISIONAL_BUILD_VERSION
from harvest.fetch_wild import sha256_file
from harvest.publish_provisional_family_transfer import (
    Artifact,
    _artifact,
    _copy_missing,
    _json,
    _number,
    _remote_inventory,
    _remote_root,
    _safe_id,
    _safe_name,
    _sha,
    _verify_remote,
    _write_resumable,
)
from harvest.wild_boundaries import WildBoundaries
from harvest.wild_layout import WildLayout


PUBLICATION_VERSION = "madeleine.wild-existing-layout-provisional-publication.v1"
COMPLETION_VERSION = "madeleine.wild-existing-layout-provisional-complete.v1"
DECODE_VERSION = "madeleine.wild-decode.v1"
MANIFEST_NAME = "publication-manifest.json"
COMPLETION_NAME = "publication-complete.json"
_CADENCE_HZ = {"native60": (55.0, 65.0), "native30": (27.0, 33.0)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_path(actual: Any, expected: Path, field: str) -> None:
    _require(Path(str(actual)).resolve() == expected.resolve(), f"{field} path mismatch")


def collect_artifacts(
    *,
    work_root: str | Path,
    raw_dir: str | Path,
    layout_path: str | Path,
    boundaries_path: str | Path,
    video_id: str,
    cadence_tier: str,
) -> tuple[list[Artifact], dict[str, Any], dict[str, Any]]:
    """Validate source/PTS/layout/decode/build bindings and name the payload."""

    video_id = _safe_id(video_id)
    _require(cadence_tier in _CADENCE_HZ, "unsupported cadence tier")
    work = Path(work_root)
    raw = Path(raw_dir)
    layout_file = Path(layout_path)
    boundaries_file = Path(boundaries_path)
    decode_dir = work / "decode"
    parts_dir = work / "shards" / "parts"

    fetch_path = raw / "fetch.json"
    pts_manifest_path = raw / "frame_pts.json"
    fetch = _json(fetch_path, "fetch report")
    pts_manifest = _json(pts_manifest_path, "PTS manifest")
    _require(
        fetch.get("format_version")
        in ("madeleine.wild-fetch.v1", "madeleine.wild-fetch.v2"),
        "wrong fetch-report version",
    )
    _require(fetch.get("video_id") == video_id, "fetch report video mismatch")
    source_name = _safe_name(fetch.get("source_file"), "source video")
    source_path = raw / source_name
    source_sha = _sha(fetch.get("sha256"), "source video")
    source_artifact = _artifact(
        "source video validation",
        source_path,
        "source-not-uploaded.mp4",
        expected_sha=source_sha,
    )
    _require(pts_manifest.get("format_version") == "madeleine.wild-pts.v1", "wrong PTS-manifest version")
    _require(pts_manifest.get("source_file") == source_name, "PTS source-file mismatch")
    _require(pts_manifest.get("source_sha256") == source_sha, "PTS source-hash mismatch")
    pts_name = _safe_name(pts_manifest.get("path"), "PTS array")
    pts_path = raw / pts_name
    pts_artifact = _artifact(
        "PTS array",
        pts_path,
        f"source/{pts_name}",
        expected_sha=pts_manifest.get("sha256"),
    )
    pts = np.load(pts_path, mmap_mode="r", allow_pickle=False)
    _require(
        pts.ndim == 1
        and pts.size == int(pts_manifest.get("frames", -1))
        and pts.size > 1
        and np.all(np.isfinite(pts))
        and np.all(np.diff(pts) > 0),
        "PTS array is not a finite strictly increasing bound grid",
    )

    layout = WildLayout.load(layout_file)
    boundaries = WildBoundaries.load(boundaries_file)
    _require(layout.video_id == video_id and layout.human_reviewed is False, "layout is not target AI-only layout")
    _require(layout.temporal_offset_source == "unmeasured", "existing layout must retain unmeasured timing")
    _require(boundaries.video_id == video_id and boundaries.human_reviewed is False, "boundaries are not target AI-only boundaries")
    _require(boundaries.reviewer_kind == "ai_agent", "boundaries reviewer is not AI-only")
    _require(boundaries.source_sha256 == source_sha, "boundaries source binding mismatch")

    decode_path = decode_dir / "decode_report.json"
    decode = _json(decode_path, "decode report")
    _require(decode.get("format_version") == DECODE_VERSION, "wrong decode version")
    _require(decode.get("video_id") == video_id and decode.get("admitted") is False, "decode is not target provisional decode")
    _require(isinstance(decode.get("rejection_reasons"), list) and decode["rejection_reasons"], "provisional decode lacks rejection reasons")
    _require(decode.get("source_video", {}).get("sha256") == source_sha, "decode source binding mismatch")
    _exact_path(decode.get("source_video", {}).get("path"), source_path, "decode source")
    _require(decode.get("layout", {}).get("human_reviewed") is False, "decode layout claims human review")
    _require(_sha(decode.get("layout", {}).get("sha256"), "decode layout") == sha256_file(layout_file), "decode layout hash mismatch")
    _exact_path(decode.get("layout", {}).get("path"), layout_file, "decode layout")
    _require(decode.get("boundaries", {}).get("human_reviewed") is False, "decode boundaries claim human review")
    _require(_sha(decode.get("boundaries", {}).get("sha256"), "decode boundaries") == sha256_file(boundaries_file), "decode boundaries hash mismatch")
    _exact_path(decode.get("boundaries", {}).get("path"), boundaries_file, "decode boundaries")
    timing = decode.get("timing")
    _require(isinstance(timing, dict) and timing.get("authority") == "presentation_timestamp", "decode timing authority mismatch")
    pts_evidence = timing.get("pts_evidence")
    _require(isinstance(pts_evidence, dict), "decode lacks PTS evidence")
    _exact_path(pts_evidence.get("manifest"), pts_manifest_path, "decode PTS manifest")
    _require(_sha(pts_evidence.get("sha256"), "decode PTS") == pts_artifact.sha256, "decode PTS hash mismatch")
    _require(int(pts_evidence.get("frames", -1)) == pts.size, "decode PTS frame count mismatch")
    _require(int(decode.get("decoded_frames", -1)) == pts.size, "decode frame count differs from PTS grid")
    _require(decode.get("score_source") in (None, {"kind": "decoded_from_source_video"}), "unexpected external score provenance")
    raw_name = _safe_name(decode.get("raw_labels"), "raw labels")
    labels_name = _safe_name(decode.get("labels"), "native labels")
    raw_labels = _artifact("raw labels", decode_dir / raw_name, "decode/labels_raw.parquet", expected_sha=decode.get("raw_labels_sha256"))
    labels = _artifact("native labels", decode_dir / labels_name, "decode/labels_native.parquet", expected_sha=decode.get("labels_sha256"))

    report_path = parts_dir / "wild_provisional_build_report.json"
    build = _json(report_path, "provisional build report")
    _require(build.get("format_version") == PROVISIONAL_BUILD_VERSION, "wrong provisional-build version")
    _require(build.get("video_id") == video_id, "build video mismatch")
    _require(build.get("label_kind") == "wild_overlay_provisional", "wrong provisional label kind")
    _require(build.get("admission_tier") == "provisional_not_train_ready", "wrong build admission tier")
    _require(build.get("timing_authority") == "presentation_timestamp", "wrong build timing authority")
    _require(int(build.get("train_ready_frames", -1)) == 0 and _number(build.get("train_ready_hours"), "train-ready hours") == 0.0, "provisional build claims train-ready data")
    frames = int(build.get("provisional_trainable_frames", 0))
    hours = _number(build.get("provisional_trainable_hours"), "provisional hours")
    fps = _number(build.get("effective_grid_hz"), "effective grid")
    low, high = _CADENCE_HZ[cadence_tier]
    _require(frames > 0 and hours > 0, "provisional build has no usable frames")
    _require(low <= fps <= high, f"effective grid {fps:.4f} does not match {cadence_tier}")
    _require(math.isclose(frames / fps / 3600.0, hours, abs_tol=1e-9), "provisional hours mismatch")
    _require(int(build.get("decoded_frames", -1)) == int(decode.get("decoded_frames", -2)), "build/decode frame mismatch")
    _require(float(build.get("decoded_hours", -1)) == float(decode.get("decoded_hours", -2)), "build/decode hour mismatch")
    _require(build.get("unresolved_admission_reasons") == decode.get("rejection_reasons"), "build/decode rejection mismatch")
    inputs = build.get("inputs")
    _require(isinstance(inputs, dict), "build lacks bindings")
    decode_binding = inputs.get("decode_report")
    labels_binding = inputs.get("labels")
    layout_binding = inputs.get("layout")
    _require(isinstance(decode_binding, dict) and _safe_name(decode_binding.get("path"), "build decode") == decode_path.name and _sha(decode_binding.get("sha256"), "build decode") == sha256_file(decode_path), "build decode binding mismatch")
    _require(isinstance(labels_binding, dict) and _safe_name(labels_binding.get("path"), "build labels") == labels_name and _sha(labels_binding.get("sha256"), "build labels") == labels.sha256, "build labels binding mismatch")
    _require(isinstance(layout_binding, dict) and _safe_name(layout_binding.get("path"), "build layout") == layout_file.name and _sha(layout_binding.get("sha256"), "build layout") == sha256_file(layout_file), "build layout binding mismatch")
    _require(_sha(inputs.get("source_video_sha256"), "build source") == source_sha, "build source binding mismatch")
    _require(_sha(inputs.get("boundaries_sha256"), "build boundaries") == sha256_file(boundaries_file), "build boundaries binding mismatch")
    implementation = build.get("implementation")
    _require(isinstance(implementation, dict) and implementation.get("module") == "harvest/build_wild.py", "build implementation binding missing")
    implementation_sha = _sha(implementation.get("sha256"), "builder implementation")

    parts = build.get("parts")
    _require(isinstance(parts, list) and parts, "provisional build has no parts")
    part_artifacts: list[Artifact] = []
    total_frames = 0
    sidecar_bindings = {
        "implementation_sha256": implementation_sha,
        "source_video_sha256": source_sha,
        "labels_sha256": labels.sha256,
        "layout_sha256": sha256_file(layout_file),
        "boundaries_sha256": sha256_file(boundaries_file),
    }
    for index, row in enumerate(parts):
        _require(isinstance(row, dict), f"part {index} is not an object")
        session_id = f"wild_provisional_{video_id}__r{index:03d}"
        name = _safe_name(row.get("npz"), f"part {index}")
        _require(row.get("session_id") == session_id and name == f"{session_id}.npz", "parts are not canonical/contiguous")
        part_frames = int(row.get("frames", 0))
        _require(part_frames > 0, "part has no frames")
        total_frames += part_frames
        part_path = parts_dir / name
        part = _artifact("NPZ part", part_path, f"shards/{name}", expected_sha=row.get("sha256"), npz_part=True)
        sidecar_path = part_path.with_name(part_path.name + ".complete.json")
        sidecar = _json(sidecar_path, f"part {index} sidecar")
        _require(sidecar.get("format_version") == PART_COMPLETION_VERSION, "wrong part-sidecar version")
        _require(sidecar.get("row") == row, "part sidecar row mismatch")
        _require(int(sidecar.get("npz_bytes", -1)) == part.size_bytes, "part sidecar size mismatch")
        _require(sidecar.get("bindings") == sidecar_bindings, "part sidecar bindings mismatch")
        part_artifacts.extend((part, _artifact("NPZ part completion", sidecar_path, f"shards/{sidecar_path.name}")))
    _require(total_frames == frames, "parts do not sum to provisional frame total")

    corpus_path = parts_dir.parent / "wild_provisional_corpus_manifest.json"
    corpus = _json(corpus_path, "provisional corpus manifest")
    _require(corpus.get("format_version") == PROVISIONAL_BUILD_VERSION, "wrong corpus version")
    _require(corpus.get("admission_tier") == "provisional_not_train_ready", "wrong corpus admission tier")
    _require(corpus.get("videos") == [build] and int(corpus.get("video_count", -1)) == 1, "corpus does not bind exactly this build")
    _require(_number(corpus.get("train_ready_hours"), "corpus train-ready hours") == 0.0, "corpus claims train-ready data")

    artifacts = [
        _artifact("fetch report", fetch_path, "source/fetch.json"),
        _artifact("PTS manifest", pts_manifest_path, "source/frame_pts.json"),
        pts_artifact,
        _artifact("AI-only layout", layout_file, "layout/layout.draft.json"),
        _artifact("AI-only boundaries", boundaries_file, "layout/boundaries.outer-ai.json"),
        _artifact("decode report", decode_path, "decode/decode_report.json"),
        raw_labels,
        labels,
        _artifact("provisional build report", report_path, "shards/wild_provisional_build_report.json"),
        _artifact("provisional corpus manifest", corpus_path, "shards/wild_provisional_corpus_manifest.json"),
        *part_artifacts,
    ]
    paths = [row.relative_path for row in artifacts]
    _require(len(paths) == len(set(paths)), "duplicate publication paths")
    source = {
        "video_id": video_id,
        "source_file": source_name,
        "source_sha256": source_sha,
        "source_bytes": source_artifact.size_bytes,
        "source_video_uploaded_here": False,
        "source_video_location": "immutable raw corpus",
        "pts_sha256": pts_artifact.sha256,
        "pts_frames": int(pts.size),
    }
    return artifacts, build, source


def publish(
    *,
    work_root: str | Path,
    raw_dir: str | Path,
    layout_path: str | Path,
    boundaries_path: str | Path,
    video_id: str,
    state_dir: str | Path,
    remote_root: str,
    cadence_tier: str,
    npz_size_only: bool = False,
) -> dict[str, Any]:
    artifacts, build, source = collect_artifacts(
        work_root=work_root,
        raw_dir=raw_dir,
        layout_path=layout_path,
        boundaries_path=boundaries_path,
        video_id=video_id,
        cadence_tier=cadence_tier,
    )
    video_id = _safe_id(video_id)
    remote_dir = f"{_remote_root(remote_root)}/{cadence_tier}/{video_id}"
    state = Path(state_dir)
    _require(not state.is_symlink(), "state directory may not be a symlink")
    manifest = {
        "format_version": PUBLICATION_VERSION,
        "video_id": video_id,
        "cadence_tier": cadence_tier,
        "admission_tier": "provisional_not_train_ready",
        "source": source,
        "provisional_trainable_frames": int(build["provisional_trainable_frames"]),
        "provisional_trainable_hours": float(build["provisional_trainable_hours"]),
        "objects": [
            {
                "kind": row.kind,
                "path": row.relative_path,
                "size_bytes": row.size_bytes,
                "sha256": row.sha256,
                "remote_verification": (
                    "exact_size_against_local_and_build_sha256"
                    if row.npz_part and npz_size_only
                    else "sha256_and_size_readback"
                ),
            }
            for row in artifacts
        ],
        "object_count": len(artifacts),
        "total_bytes": sum(row.size_bytes for row in artifacts),
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
        "npz_verification": "exact_size_against_local_and_build_sha256" if npz_size_only else "sha256_and_size_readback",
        "completion_policy": "marker uploaded last after exact payload inventory",
        "human_reviewed": False,
        "training_admitted": False,
    }
    completion_path = state / COMPLETION_NAME
    _write_resumable(completion_path, completion)
    marker = _artifact("publication completion", completion_path, COMPLETION_NAME)
    observed = _remote_inventory(remote_dir)
    unexpected = set(observed) - (set(expected_payload) | {COMPLETION_NAME})
    _require(not unexpected, f"unexpected remote objects: {sorted(unexpected)}")
    if COMPLETION_NAME in observed:
        expected_complete = {**expected_payload, COMPLETION_NAME: marker.size_bytes}
        _require(observed == expected_complete, "completed inventory is incomplete or changed")
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
    _require(_remote_inventory(remote_dir) == expected_payload, "payload inventory differs before completion")
    _copy_missing(marker, remote_dir, npz_size_only=False)
    expected_complete = {**expected_payload, COMPLETION_NAME: marker.size_bytes}
    _require(_remote_inventory(remote_dir) == expected_complete, "inventory differs after completion")
    return {**completion, "publication_status": "published"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--cadence-tier", choices=tuple(_CADENCE_HZ), required=True)
    parser.add_argument("--npz-size-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(publish(
        work_root=args.work_root,
        raw_dir=args.raw_dir,
        layout_path=args.layout,
        boundaries_path=args.boundaries,
        video_id=args.video_id,
        state_dir=args.state_dir,
        remote_root=args.remote_root,
        cadence_tier=args.cadence_tier,
        npz_size_only=args.npz_size_only,
    ), indent=2))


if __name__ == "__main__":
    main()
