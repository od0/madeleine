#!/usr/bin/env python3
"""Build the exact label-free source inventory for exploratory C/D SSL.

The inventory is deliberately produced before any RGB cache is opened.  It
binds every allowed source object to a content hash, proves the forbidden
members absent, and records only temporal metadata required to choose safe
windows.  Supervision arrays named ``keys`` are never indexed or read.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from nitrogen.mask import video_mask_rect


SCHEMA = "madeleine.dynamics-pretraining-inventory.v1"
CONTRACT_SCHEMA = "madeleine.dynamics-pretraining-exploratory-cd.v1"
WILD_SCHEMA = "madeleine.wild-provisional-corpus.v1"
HOLDOUT_VIDEO = "y4nQHqYSObI"
SEALED_SESSION = "rec_20260727_220000_test"
OWN_VAL_A = "rec_20260724_171305_5min"
EXPECTED_OWN_TRAIN = (
    "rec_20260724_190233",
    "rec_20260725_015612",
    "rec_20260725_021338",
)
FORBIDDEN_TOKENS = (
    SEALED_SESSION.casefold(),
    "untouched",
    "b1_",
    "/b1/",
    "val-b",
    "val_b",
)


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def reject_forbidden_text(value: str, *, allow_y4n_metadata: bool = False) -> None:
    folded = value.casefold()
    if any(token in folded for token in FORBIDDEN_TOKENS):
        raise ValueError(f"forbidden identity/path in SSL inventory: {value}")
    if not allow_y4n_metadata and HOLDOUT_VIDEO.casefold() in folded:
        raise ValueError(f"holdout video in SSL inventory: {value}")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite inventory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _required_npz_metadata(
    path: Path,
    *,
    expected_session: str,
    expected_frames: int | None,
    require_all_active: bool,
    require_contiguous: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Read temporal metadata without ever touching pixel or key members."""

    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        required = {"engine_frame_idx", "input_active", "session_id"}
        if not required.issubset(names):
            raise ValueError(f"{path}: missing temporal metadata")
        # Merely checking member names does not decompress their values.  The
        # only members indexed below are the three label-free metadata arrays.
        engine = np.asarray(archive["engine_frame_idx"], dtype=np.int64)
        active = np.asarray(archive["input_active"], dtype=np.uint8)
        stored = str(archive["session_id"].reshape(()).item())
    if stored != expected_session:
        raise ValueError(f"{path}: session identity changed")
    if engine.ndim != 1 or active.shape != engine.shape:
        raise ValueError(f"{path}: temporal metadata is not aligned")
    if expected_frames is not None and len(engine) != expected_frames:
        raise ValueError(f"{path}: frame count changed")
    if require_contiguous and len(engine) and np.any(np.diff(engine) != 1):
        raise ValueError(f"{path}: engine timeline is not contiguous")
    if np.any((active != 0) & (active != 1)):
        raise ValueError(f"{path}: input_active is not binary")
    if require_all_active and not bool(active.all()):
        raise ValueError(f"{path}: mapped session is not all-active")
    return engine, active, int(active.sum())


def _eligible_windows(engine: np.ndarray, active: np.ndarray, max_horizon: int) -> int:
    if len(engine) < max_horizon + 2:
        return 0
    usable = active.astype(bool, copy=True)
    if len(engine) > 1:
        # A row may start a safe run only if it follows the previous engine
        # frame exactly.  Mark both sides of a gap as a boundary without
        # deleting either row from its respective run.
        consecutive = np.diff(engine) == 1
    else:
        consecutive = np.empty(0, dtype=bool)
    count = 0
    run_start: int | None = None
    for index, is_active in enumerate(usable):
        if index and not consecutive[index - 1] and run_start is not None:
            run_length = index - run_start
            count += max(0, run_length - max_horizon - 1)
            run_start = None
        if is_active and run_start is None:
            run_start = index
        if run_start is not None and (not is_active or index + 1 == len(usable)):
            run_end = index if not is_active else index + 1
            run_length = run_end - run_start
            count += max(0, run_length - max_horizon - 1)
            run_start = None
    return count


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
                raise ValueError("invalid or duplicate fetch-report video")
            rows[video_id] = row
    return rows


def _scaled_mask(
    rect: Sequence[int], width: int, height: int, frame_size: int = 128
) -> list[int]:
    x0, y0, x1, y1 = map(int, rect)
    return [
        max(0, int(x0 / width * frame_size) - 1),
        max(0, int(y0 / height * frame_size) - 1),
        min(frame_size, int(np.ceil(x1 / width * frame_size)) + 1),
        min(frame_size, int(np.ceil(y1 / height * frame_size)) + 1),
    ]


def _nitrogen_inventory(
    *,
    full_root: Path,
    validation_path: Path,
    raw_root: Path,
    fetch_report_path: Path,
    chunk_index_path: Path,
    max_horizon: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validation = load_json(validation_path, "full-corpus validation")
    expected_observed = {
        "valid_videos": 211,
        "sessions": 1554,
        "train_frames": 32598000,
        "deep_shards_checked": 1554,
    }
    if validation.get("ok") is not True or validation.get("deep_shards") is not True:
        raise ValueError("full-corpus features lack passing deep validation")
    observed = validation.get("observed")
    if not isinstance(observed, dict):
        raise ValueError("full-corpus validation lacks observed counts")
    for name, expected in expected_observed.items():
        if observed.get(name) != expected:
            raise ValueError(f"full-corpus validation changed {name}")

    manifest_path = full_root / "full_corpus_manifest.json"
    hashes_path = full_root / "shard_hashes.json"
    manifest = load_json(manifest_path, "full-corpus manifest")
    shard_hashes = load_json(hashes_path, "full-corpus shard hashes")
    videos = manifest.get("videos")
    if not isinstance(videos, list) or len(videos) != 211:
        raise ValueError("full-corpus manifest video count changed")
    fetch = _fetch_rows(fetch_report_path)
    video_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    total_frames = 0
    total_eligible = 0
    for video in sorted(videos, key=lambda row: str(row.get("video_id"))):
        video_id = str(video.get("video_id", ""))
        if video_id == HOLDOUT_VIDEO:
            continue
        reject_forbidden_text(video_id)
        fetch_row = fetch.get(video_id)
        if fetch_row is None:
            raise ValueError(f"missing fetched source for {video_id}")
        relative = str(fetch_row.get("path", ""))
        reject_forbidden_text(relative)
        video_path = (raw_root / relative).resolve()
        if not video_path.is_file() or video_path.stat().st_size != int(fetch_row["bytes"]):
            raise ValueError(f"source video changed: {video_id}")
        width, height = int(fetch_row["width"]), int(fetch_row["height"])
        rect = list(video_mask_rect(chunk_index_path, video_id, (width, height)))
        decoder_mode = str(video.get("decoder_mode", ""))
        if decoder_mode not in {
            "opencv_native_60hz",
            "ffmpeg_timestamp_resample_60hz",
        }:
            raise ValueError(f"unsupported decoder mode for {video_id}")
        video_sessions: list[str] = []
        for session_id_value in video.get("sessions", []):
            session_id = str(session_id_value)
            reject_forbidden_text(session_id)
            if not session_id.startswith(f"{video_id}__r"):
                raise ValueError("NitroGen session/video identity changed")
            reference = (full_root / f"{session_id}.npz").resolve()
            record = shard_hashes.get(session_id)
            if not isinstance(record, dict):
                raise ValueError(f"missing feature-shard receipt for {session_id}")
            if not reference.is_file() or reference.stat().st_size != int(record["size"]):
                raise ValueError(f"feature reference changed: {session_id}")
            engine, active, active_frames = _required_npz_metadata(
                reference,
                expected_session=session_id,
                expected_frames=None,
                require_all_active=True,
                require_contiguous=True,
            )
            frames = len(engine)
            eligible = _eligible_windows(engine, active, max_horizon)
            if frames < 1 or eligible < 1:
                raise ValueError(f"empty NitroGen session: {session_id}")
            row = {
                "source": "nitrogen",
                "session_id": session_id,
                "video_id": video_id,
                "reference_shard": str(reference),
                "reference_shard_sha256": str(record["sha256"]),
                "frames": frames,
                "active_frames": active_frames,
                "eligible_windows": eligible,
                "engine_frame_start": int(engine[0]),
                "engine_frame_end_exclusive": int(engine[-1]) + 1,
            }
            session_rows.append(row)
            video_sessions.append(session_id)
            total_frames += frames
            total_eligible += eligible
        video_rows.append(
            {
                "source": "nitrogen",
                "video_id": video_id,
                "video_path": str(video_path),
                "video_sha256": sha256_file(video_path),
                "video_bytes": video_path.stat().st_size,
                "decoder_mode": decoder_mode,
                "source_width": width,
                "source_height": height,
                "source_fps": float(fetch_row["fps"]),
                "source_frames": int(fetch_row["frames"]),
                "mask_rect_source_xyxy": rect,
                "mask_rect_128_xyxy": _scaled_mask(rect, width, height),
                "sessions": video_sessions,
            }
        )
    if len(video_rows) != 210:
        raise ValueError(f"expected 210 NitroGen training videos, found {len(video_rows)}")
    if any(row["video_id"] == HOLDOUT_VIDEO for row in video_rows):
        raise AssertionError("whole-video holdout exclusion failed")
    return video_rows, session_rows, {
        "videos": len(video_rows),
        "sessions": len(session_rows),
        "frames": total_frames,
        "eligible_windows": total_eligible,
        "manifest_sha256": sha256_file(manifest_path),
        "shard_hashes_sha256": sha256_file(hashes_path),
        "validation_sha256": sha256_file(validation_path),
        "fetch_report_sha256": sha256_file(fetch_report_path),
        "chunk_index_sha256": sha256_file(chunk_index_path),
    }


def _npz_source_row(
    *,
    source: str,
    path: Path,
    expected_sha256: str | None,
    expected_session: str,
    expected_frames: int | None,
    max_horizon: int,
) -> dict[str, Any]:
    reject_forbidden_text(str(path))
    reject_forbidden_text(expected_session)
    if not path.is_file():
        raise ValueError(f"missing {source} shard: {path}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{source} shard hash changed: {expected_session}")
    engine, active, active_frames = _required_npz_metadata(
        path,
        expected_session=expected_session,
        expected_frames=expected_frames,
        require_all_active=False,
        require_contiguous=False,
    )
    eligible = _eligible_windows(engine, active, max_horizon)
    if eligible < 1:
        raise ValueError(f"{source} shard has no safe windows: {expected_session}")
    return {
        "source": source,
        "session_id": expected_session,
        "shard_path": str(path.resolve()),
        "shard_sha256": digest,
        "frames": len(engine),
        "active_frames": active_frames,
        "eligible_windows": eligible,
        "engine_frame_start": int(engine[0]),
        "engine_frame_end_exclusive": int(engine[-1]) + 1,
    }


def _wild_inventory(
    *, wild_root: Path, aggregate_path: Path, max_horizon: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aggregate = load_json(aggregate_path, "provisional-wild aggregate")
    expected = {
        "format_version": WILD_SCHEMA,
        "admission_tier": "provisional_not_train_ready",
        "video_count": 7,
        "session_count": 2058,
        "provisional_trainable_frames": 4835638,
        "train_ready_frames": 0,
        "train_ready_hours": 0.0,
    }
    for name, value in expected.items():
        if aggregate.get(name) != value:
            raise ValueError(f"wild aggregate changed {name}")
    rows: list[dict[str, Any]] = []
    for video in aggregate.get("videos", []):
        video_id = str(video.get("video_id", ""))
        reject_forbidden_text(video_id)
        for part in video.get("parts", []):
            session_id = str(part.get("session_id", ""))
            path = (wild_root / str(part.get("path", ""))).resolve()
            row = _npz_source_row(
                source="wild_provisional",
                path=path,
                expected_sha256=str(part.get("sha256", "")),
                expected_session=session_id,
                expected_frames=int(part.get("frames", -1)),
                max_horizon=max_horizon,
            )
            row["video_id"] = video_id
            rows.append(row)
    if len(rows) != 2058 or sum(int(row["frames"]) for row in rows) != 4835638:
        raise ValueError("wild inventory accounting changed")
    return rows, {
        "videos": 7,
        "sessions": len(rows),
        "frames": sum(int(row["frames"]) for row in rows),
        "eligible_windows": sum(int(row["eligible_windows"]) for row in rows),
        "admitted_hours": 0.0,
        "aggregate_sha256": sha256_file(aggregate_path),
    }


def _own_inventory(
    *, own_root: Path, max_horizon: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    build_path = own_root / "build_manifest.json"
    build = load_json(build_path, "own-v3 build manifest")
    split = build.get("split")
    if not isinstance(split, dict) or tuple(split.get("train", ())) != EXPECTED_OWN_TRAIN:
        raise ValueError("own-v3 training split changed")
    if split.get("val") != [OWN_VAL_A]:
        raise ValueError("own-v3 validation identity changed")
    sessions = {
        str(row.get("session_id")): row for row in build.get("sessions", [])
    }
    rows: list[dict[str, Any]] = []
    for session_id in EXPECTED_OWN_TRAIN:
        record = sessions.get(session_id)
        if not isinstance(record, dict):
            raise ValueError(f"missing own-v3 session {session_id}")
        path = (own_root / str(record.get("npz", ""))).resolve()
        rows.append(
            _npz_source_row(
                source="local",
                path=path,
                expected_sha256=None,
                expected_session=session_id,
                expected_frames=int(record.get("frames", -1)),
                max_horizon=max_horizon,
            )
        )
    return rows, {
        "sessions": len(rows),
        "frames": sum(int(row["frames"]) for row in rows),
        "eligible_windows": sum(int(row["eligible_windows"]) for row in rows),
        "build_manifest_sha256": sha256_file(build_path),
    }


def build_inventory(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_json(args.contract, "exploratory C/D contract")
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("unsupported exploratory C/D contract")
    if contract.get("status") != (
        "owner_authorized_post_phase0_exploratory_override_before_optimizer_step_one"
    ):
        raise ValueError("C/D override is not explicitly authorized")
    horizons = tuple(int(value) for value in contract["data"]["horizons_native_frames"])
    if horizons != (1, 2, 4):
        raise ValueError("exploratory horizon set changed")
    max_horizon = max(horizons)

    nitrogen_videos, nitrogen_sessions, nitrogen_summary = _nitrogen_inventory(
        full_root=args.full_feature_root.resolve(),
        validation_path=args.full_validation.resolve(),
        raw_root=args.raw_root.resolve(),
        fetch_report_path=args.fetch_report.resolve(),
        chunk_index_path=args.chunk_index.resolve(),
        max_horizon=max_horizon,
    )
    wild_sessions, wild_summary = _wild_inventory(
        wild_root=args.wild_root.resolve(),
        aggregate_path=args.wild_aggregate.resolve(),
        max_horizon=max_horizon,
    )
    own_sessions, own_summary = _own_inventory(
        own_root=args.own_root.resolve(), max_horizon=max_horizon
    )
    sessions = [*nitrogen_sessions, *wild_sessions, *own_sessions]
    identities = [str(row["session_id"]) for row in sessions]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate source-session identity")
    # Final proof applies to the complete serializable payload, before any
    # downstream cache process gets a path it could open.
    for value in identities:
        reject_forbidden_text(value)
    serialized = json.dumps(
        {"videos": nitrogen_videos, "sessions": sessions}, sort_keys=True
    )
    reject_forbidden_text(serialized)
    if HOLDOUT_VIDEO in serialized:
        raise AssertionError("y4n survived pretraining inventory exclusion")

    source_eligible = {
        "nitrogen": int(nitrogen_summary["eligible_windows"]),
        "wild_provisional": int(wild_summary["eligible_windows"]),
        "local": int(own_summary["eligible_windows"]),
    }
    total_eligible = sum(source_eligible.values())
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "study_id": contract["study_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": str(args.contract.resolve()),
            "sha256": sha256_file(args.contract),
        },
        "labels_consumed": False,
        "forbidden_exclusion_proof": {
            "sealed_untouched_absent": True,
            "whole_y4n_absent": True,
            "own_val_a_absent": True,
            "B1_absent": True,
            "val_B_absent": True,
            "checked_before_cache_RGB_access": True,
        },
        "horizons_native_frames": list(horizons),
        "nitrogen_videos": nitrogen_videos,
        "sessions": sessions,
        "summary": {
            "nitrogen": nitrogen_summary,
            "wild_provisional": wild_summary,
            "local": own_summary,
            "sessions": len(sessions),
            "eligible_windows": total_eligible,
            "eligible_fraction_by_source": {
                source: count / total_eligible
                for source, count in source_eligible.items()
            },
        },
    }
    payload["inventory_content_sha256"] = canonical_sha256(payload)
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--full-feature-root", type=Path, required=True)
    value.add_argument("--full-validation", type=Path, required=True)
    value.add_argument("--raw-root", type=Path, required=True)
    value.add_argument("--fetch-report", type=Path, required=True)
    value.add_argument("--chunk-index", type=Path, required=True)
    value.add_argument("--wild-root", type=Path, required=True)
    value.add_argument("--wild-aggregate", type=Path, required=True)
    value.add_argument("--own-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = build_inventory(args)
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
