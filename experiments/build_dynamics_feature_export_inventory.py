#!/usr/bin/env python3
"""Build the exact post-SSL 211-video C/D feature-export inventory.

The 210 training-video paths, hashes, masks, and session ranges are reused
from the already validated label-free SSL inventory; they are not re-hashed.
The held-out y4n source is added only after a terminal C/D checkpoint has
passed its exact schema, arm, completed-step, and SHA-256 checks.  Only then
is the y4n video opened bytewise for its first and only source hash.

Full-corpus validation, manifest membership, reference-shard receipts, and
the fetch report are reconciled before the new inventory is atomically
published.  No key or old feature array is opened by this builder.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from data.precompute_features import FRAME_SIZE, _nominal_timeline_frames
from experiments.build_dynamics_pretraining_inventory import (
    SCHEMA as SSL_INVENTORY_SCHEMA,
    canonical_sha256,
)
from experiments.export_dynamics_features import (
    CHECKPOINT_SCHEMA,
    EXPORT_ONLY_ROLE,
    INVENTORY_SCHEMA,
    PRODUCTION_COUNTS,
    TRAIN_ROLE,
    Y4N_VIDEO_ID,
    ExpectedCounts,
    CheckpointContract,
    load_checkpoint_contract,
    sha256_file,
    validate_inventory_payload,
)


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _fetch_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            video_id = str(row.get("video_id", ""))
            if not video_id or video_id in rows:
                raise ValueError("fetch report has invalid/duplicate successful video")
            rows[video_id] = row
    return rows


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite export inventory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _verify_ssl_inventory(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("SSL inventory SHA-256 mismatch")
    payload = _json(path, "SSL inventory")
    if payload.get("schema_version") != SSL_INVENTORY_SCHEMA:
        raise ValueError("SSL inventory schema mismatch")
    if payload.get("labels_consumed") is not False:
        raise ValueError("SSL inventory is not label-free")
    proof = payload.get("forbidden_exclusion_proof")
    if not isinstance(proof, dict) or proof.get("whole_y4n_absent") is not True:
        raise ValueError("SSL inventory lacks whole-y4n exclusion proof")
    recorded_content = payload.get("inventory_content_sha256")
    without_receipt = dict(payload)
    without_receipt.pop("inventory_content_sha256", None)
    if recorded_content != canonical_sha256(without_receipt):
        raise ValueError("SSL inventory internal content hash mismatch")
    serialized = json.dumps(
        {
            "nitrogen_videos": payload.get("nitrogen_videos"),
            "sessions": [
                row
                for row in payload.get("sessions", [])
                if isinstance(row, dict) and row.get("source") == "nitrogen"
            ],
        }
    )
    if Y4N_VIDEO_ID in serialized:
        raise ValueError("SSL inventory contains y4n despite exclusion receipt")
    return payload


def _scaled_mask(
    rect: Sequence[int], width: int, height: int
) -> list[int]:
    x0, y0, x1, y1 = map(int, rect)
    result = [
        max(0, int(x0 / width * FRAME_SIZE) - 1),
        max(0, int(y0 / height * FRAME_SIZE) - 1),
        min(FRAME_SIZE, int(np.ceil(x1 / width * FRAME_SIZE)) + 1),
        min(FRAME_SIZE, int(np.ceil(y1 / height * FRAME_SIZE)) + 1),
    ]
    if not (0 <= result[0] < result[2] <= FRAME_SIZE):
        raise ValueError("scaled mask has invalid horizontal extent")
    if not (0 <= result[1] < result[3] <= FRAME_SIZE):
        raise ValueError("scaled mask has invalid vertical extent")
    return result


def _validate_full_inputs(
    *,
    full_root: Path,
    validation_path: Path,
    raw_root: Path,
    fetch_report: Path,
    feature_root: Path,
    expected_counts: ExpectedCounts,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    validation = _json(validation_path, "full-corpus validation")
    if validation.get("ok") is not True or validation.get("deep_shards") is not True:
        raise ValueError("full corpus lacks passing deep validation")
    observed = validation.get("observed")
    if not isinstance(observed, dict):
        raise ValueError("full validation lacks observed counts")
    for name, expected in {
        "valid_videos": expected_counts.videos,
        "sessions": expected_counts.sessions,
        "train_frames": expected_counts.frames,
        "deep_shards_checked": expected_counts.sessions,
    }.items():
        if observed.get(name) != expected:
            raise ValueError(f"full validation {name} changed")
    paths = validation.get("paths")
    if isinstance(paths, dict):
        for name, expected in {
            "output_root": full_root,
            "raw_root": raw_root,
            "fetch_report": fetch_report,
            "feature_root": feature_root,
        }.items():
            recorded = paths.get(name)
            if recorded is not None and Path(recorded).resolve() != expected.resolve():
                raise ValueError(f"full validation {name} path differs")
    manifest_path = full_root / "full_corpus_manifest.json"
    hashes_path = full_root / "shard_hashes.json"
    manifest = _json(manifest_path, "full-corpus manifest")
    hashes = _json(hashes_path, "full-corpus shard hashes")
    videos = manifest.get("videos")
    if not isinstance(videos, list) or len(videos) != expected_counts.videos:
        raise ValueError("full-corpus video membership/count changed")
    if int(manifest.get("session_count", -1)) != expected_counts.sessions:
        raise ValueError("full-corpus session count changed")
    if int(manifest.get("train_frames", -1)) != expected_counts.frames:
        raise ValueError("full-corpus frame count changed")
    fetch = _fetch_rows(fetch_report)
    if set(str(row.get("video_id")) for row in videos) - set(fetch):
        raise ValueError("full-corpus video is absent from successful fetch report")
    return validation, manifest, hashes, fetch


def _train_video_rows(
    *,
    ssl: dict[str, Any],
    full_manifest: dict[str, Any],
    shard_hashes: dict[str, Any],
    fetch: dict[str, dict[str, Any]],
    raw_root: Path,
    full_root: Path,
    expected_counts: ExpectedCounts,
) -> list[dict[str, Any]]:
    full_by_id = {
        str(row.get("video_id")): row for row in full_manifest.get("videos", [])
    }
    ssl_videos = ssl.get("nitrogen_videos")
    if not isinstance(ssl_videos, list):
        raise ValueError("SSL inventory lacks nitrogen_videos")
    expected_train_ids = set(full_by_id) - {Y4N_VIDEO_ID}
    ssl_by_id = {str(row.get("video_id")): row for row in ssl_videos}
    if set(ssl_by_id) != expected_train_ids or len(ssl_by_id) != expected_counts.train_videos:
        raise ValueError("SSL/full-corpus training video membership differs")
    nitrogen_sessions = {
        str(row.get("session_id")): row
        for row in ssl.get("sessions", [])
        if isinstance(row, dict) and row.get("source") == "nitrogen"
    }
    rows: list[dict[str, Any]] = []
    for video_id in sorted(expected_train_ids):
        source = ssl_by_id[video_id]
        full = full_by_id[video_id]
        fetch_row = fetch[video_id]
        source_path = Path(str(source.get("video_path", ""))).resolve()
        expected_path = (raw_root / str(fetch_row.get("path", ""))).resolve()
        if source_path != expected_path:
            raise ValueError(f"{video_id}: SSL/fetch source path differs")
        if not source_path.is_file() or source_path.stat().st_size != int(fetch_row["bytes"]):
            raise ValueError(f"{video_id}: source file size changed")
        # Intentionally reuse the SSL hash.  The 210 files are not re-hashed.
        source_sha = str(source.get("video_sha256", ""))
        if len(source_sha) != 64:
            raise ValueError(f"{video_id}: SSL raw hash is invalid")
        width = int(source["source_width"])
        height = int(source["source_height"])
        fps = float(source["source_fps"])
        decoded = int(source["source_frames"])
        resampled, timeline = _nominal_timeline_frames(decoded, fps)
        mode = (
            "ffmpeg_timestamp_resample_60hz"
            if resampled
            else "opencv_native_60hz"
        )
        if source.get("decoder_mode") != mode or full.get("decoder_mode") != mode:
            raise ValueError(f"{video_id}: decoder mode differs")
        session_ids = [str(value) for value in full.get("sessions", [])]
        if session_ids != list(source.get("sessions", [])):
            raise ValueError(f"{video_id}: SSL/full session list differs")
        sessions: list[dict[str, Any]] = []
        for session_id in session_ids:
            record = nitrogen_sessions.get(session_id)
            if record is None:
                raise ValueError(f"{video_id}: SSL session row missing {session_id}")
            start = int(record["engine_frame_start"])
            end = int(record["engine_frame_end_exclusive"])
            if end - start != int(record["frames"]):
                raise ValueError(f"{session_id}: SSL frame range changed")
            reference = (full_root / f"{session_id}.npz").resolve()
            if Path(str(record["reference_shard"])).resolve() != reference:
                raise ValueError(f"{session_id}: reference path changed")
            hash_row = shard_hashes.get(session_id)
            if not isinstance(hash_row, dict):
                raise ValueError(f"{session_id}: full-corpus hash receipt missing")
            if str(record["reference_shard_sha256"]) != str(hash_row.get("sha256")):
                raise ValueError(f"{session_id}: SSL/full reference hash differs")
            if not reference.is_file() or reference.stat().st_size != int(hash_row["size"]):
                raise ValueError(f"{session_id}: reference shard size changed")
            sessions.append(
                {
                    "session_id": session_id,
                    "start_frame": start,
                    "end_frame": end,
                    "reference_shard": str(reference),
                    "reference_shard_sha256": str(hash_row["sha256"]),
                }
            )
        rows.append(
            {
                "video_id": video_id,
                "role": TRAIN_ROLE,
                "video_path": str(source_path),
                "video_sha256": source_sha,
                "decoder_mode": mode,
                "video": {
                    "average_fps": fps,
                    "decoded_frames": decoded,
                    "nominal_timeline_frames": timeline,
                    "resolution_wh": [width, height],
                },
                "mask_rect_xyxy": list(map(int, source["mask_rect_source_xyxy"])),
                "resized_mask_rect_xyxy": list(
                    map(int, source["mask_rect_128_xyxy"])
                ),
                "sessions": sessions,
            }
        )
    return rows


def _y4n_row(
    *,
    full_manifest: dict[str, Any],
    shard_hashes: dict[str, Any],
    fetch: dict[str, dict[str, Any]],
    raw_root: Path,
    full_root: Path,
    feature_root: Path,
) -> dict[str, Any]:
    full_by_id = {
        str(row.get("video_id")): row for row in full_manifest.get("videos", [])
    }
    full = full_by_id.get(Y4N_VIDEO_ID)
    if not isinstance(full, dict):
        raise ValueError("full corpus lacks exact y4n holdout")
    fetch_row = fetch.get(Y4N_VIDEO_ID)
    if fetch_row is None:
        raise ValueError("fetch report lacks successful y4n source")
    source_path = (raw_root / str(fetch_row.get("path", ""))).resolve()
    if not source_path.is_file() or source_path.stat().st_size != int(fetch_row["bytes"]):
        raise ValueError("y4n source file size changed")
    per_video_path = feature_root / Y4N_VIDEO_ID / "feature_build_manifest.json"
    per_video = _json(per_video_path, "y4n feature build manifest")
    if per_video.get("format") != "resnet18_imagenet_avgpool_float16_v1":
        raise ValueError("y4n reference feature format changed")
    reports = per_video.get("videos")
    if not isinstance(reports, list) or len(reports) != 1:
        raise ValueError("y4n feature manifest must contain one video report")
    report = reports[0]
    if report.get("video_id") != Y4N_VIDEO_ID:
        raise ValueError("y4n feature manifest identity changed")
    metadata = report.get("video")
    if not isinstance(metadata, dict):
        raise ValueError("y4n feature report lacks video metadata")
    fps = float(metadata["average_fps"])
    decoded = int(metadata["decoded_frames"])
    timeline = int(metadata["nominal_timeline_frames"])
    resolution = metadata.get("resolution_wh")
    if not isinstance(resolution, list) or len(resolution) != 2:
        resolution = [int(fetch_row["width"]), int(fetch_row["height"])]
    width, height = map(int, resolution)
    resampled, expected_timeline = _nominal_timeline_frames(decoded, fps)
    mode = (
        "ffmpeg_timestamp_resample_60hz" if resampled else "opencv_native_60hz"
    )
    if timeline != expected_timeline or report.get("decoder_mode") != mode:
        raise ValueError("y4n decoder plan changed")
    if full.get("decoder_mode") != mode:
        raise ValueError("y4n full/per-video decoder modes differ")
    parts = report.get("parts")
    if not isinstance(parts, list):
        raise ValueError("y4n feature report lacks parts")
    full_sessions = [str(value) for value in full.get("sessions", [])]
    if [str(part.get("session_id")) for part in parts] != full_sessions:
        raise ValueError("y4n part membership differs from full manifest")
    sessions: list[dict[str, Any]] = []
    for part in parts:
        session_id = str(part["session_id"])
        frame_range = part.get("source_frame_range")
        if not isinstance(frame_range, list) or len(frame_range) != 2:
            raise ValueError(f"{session_id}: y4n source frame range missing")
        start, end = map(int, frame_range)
        if end - start != int(part["frames"]):
            raise ValueError(f"{session_id}: y4n frame range changed")
        hash_row = shard_hashes.get(session_id)
        if not isinstance(hash_row, dict):
            raise ValueError(f"{session_id}: y4n reference hash missing")
        reference = (full_root / f"{session_id}.npz").resolve()
        if not reference.is_file() or reference.stat().st_size != int(hash_row["size"]):
            raise ValueError(f"{session_id}: y4n reference shard size changed")
        sessions.append(
            {
                "session_id": session_id,
                "start_frame": start,
                "end_frame": end,
                "reference_shard": str(reference),
                "reference_shard_sha256": str(hash_row["sha256"]),
            }
        )
    rect = list(map(int, report["mask_rect_xyxy"]))
    # This is the only source hash computed here, and this function is called
    # only after load_checkpoint_contract has validated terminal C/D state.
    y4n_sha256 = sha256_file(source_path)
    return {
        "video_id": Y4N_VIDEO_ID,
        "role": EXPORT_ONLY_ROLE,
        "video_path": str(source_path),
        "video_sha256": y4n_sha256,
        "decoder_mode": mode,
        "video": {
            "average_fps": fps,
            "decoded_frames": decoded,
            "nominal_timeline_frames": timeline,
            "resolution_wh": [width, height],
        },
        "mask_rect_xyxy": rect,
        "resized_mask_rect_xyxy": _scaled_mask(rect, width, height),
        "sessions": sessions,
    }


def build_export_inventory(
    *,
    ssl_inventory_path: Path,
    ssl_inventory_sha256: str,
    full_root: Path,
    full_validation_path: Path,
    raw_root: Path,
    fetch_report: Path,
    feature_root: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    arm: str,
    expected_completed_steps: int,
    expected_counts: ExpectedCounts = PRODUCTION_COUNTS,
) -> dict[str, Any]:
    # Ordering is a data-embargo property: terminal checkpoint validation must
    # complete before _y4n_row is reachable and hashes the held-out raw video.
    checkpoint: CheckpointContract = load_checkpoint_contract(
        checkpoint_path,
        checkpoint_sha256,
        expected_arm=arm,
        expected_completed_steps=expected_completed_steps,
    )
    ssl = _verify_ssl_inventory(ssl_inventory_path, ssl_inventory_sha256)
    validation, manifest, hashes, fetch = _validate_full_inputs(
        full_root=full_root,
        validation_path=full_validation_path,
        raw_root=raw_root,
        fetch_report=fetch_report,
        feature_root=feature_root,
        expected_counts=expected_counts,
    )
    videos = _train_video_rows(
        ssl=ssl,
        full_manifest=manifest,
        shard_hashes=hashes,
        fetch=fetch,
        raw_root=raw_root,
        full_root=full_root,
        expected_counts=expected_counts,
    )
    videos.append(
        _y4n_row(
            full_manifest=manifest,
            shard_hashes=hashes,
            fetch=fetch,
            raw_root=raw_root,
            full_root=full_root,
            feature_root=feature_root,
        )
    )
    payload: dict[str, Any] = {
        "schema_version": INVENTORY_SCHEMA,
        "population": {
            "videos": expected_counts.videos,
            "sessions": expected_counts.sessions,
            "frames": expected_counts.frames,
            "train_videos": expected_counts.train_videos,
        },
        "provenance": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "terminal_checkpoint": {
                "schema_version": CHECKPOINT_SCHEMA,
                "sha256": checkpoint.sha256,
                "arm": checkpoint.arm,
                "completed_steps": checkpoint.completed_steps,
            },
            "ssl_inventory": {
                "schema_version": SSL_INVENTORY_SCHEMA,
                "path": str(ssl_inventory_path.resolve()),
                "sha256": ssl_inventory_sha256,
                "train_video_hashes_reused_without_rehash": True,
            },
            "full_corpus_validation": {
                "path": str(full_validation_path.resolve()),
                "sha256": sha256_file(full_validation_path),
                "ok": validation.get("ok"),
                "deep_shards": validation.get("deep_shards"),
            },
            "full_corpus_manifest_sha256": sha256_file(
                full_root / "full_corpus_manifest.json"
            ),
            "full_corpus_shard_hashes_sha256": sha256_file(
                full_root / "shard_hashes.json"
            ),
            "fetch_report_sha256": sha256_file(fetch_report),
            "y4n_hashed_after_terminal_checkpoint_validation": True,
        },
        "videos": videos,
    }
    # Reuse the exporter's independent schema/count/embargo validator before
    # allowing these paths to become executable input.
    validate_inventory_payload(
        payload,
        path=Path("/content-bound/unwritten-export-inventory.json"),
        sha256="0" * 64,
        expected_counts=expected_counts,
    )
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--ssl-inventory", type=Path, required=True)
    value.add_argument("--ssl-inventory-sha256", required=True)
    value.add_argument("--full-feature-root", type=Path, required=True)
    value.add_argument("--full-validation", type=Path, required=True)
    value.add_argument("--raw-root", type=Path, required=True)
    value.add_argument("--fetch-report", type=Path, required=True)
    value.add_argument("--feature-root", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--checkpoint-sha256", required=True)
    value.add_argument("--arm", choices=("C", "D"), required=True)
    value.add_argument("--expected-completed-steps", type=int, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = build_export_inventory(
        ssl_inventory_path=args.ssl_inventory.resolve(),
        ssl_inventory_sha256=args.ssl_inventory_sha256,
        full_root=args.full_feature_root.resolve(),
        full_validation_path=args.full_validation.resolve(),
        raw_root=args.raw_root.resolve(),
        fetch_report=args.fetch_report.resolve(),
        feature_root=args.feature_root.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        checkpoint_sha256=args.checkpoint_sha256,
        arm=args.arm,
        expected_completed_steps=args.expected_completed_steps,
    )
    _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
                "population": payload["population"],
                "terminal_checkpoint": payload["provenance"]["terminal_checkpoint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
