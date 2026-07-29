#!/usr/bin/env python3
"""Export final-EMA C/D visual features from an explicit corpus inventory.

This is deliberately a *post-pretraining* program.  It loads one terminal
streaming dynamics checkpoint, verifies its caller-supplied SHA-256, and runs
only the frozen EMA target encoder.  It never discovers source videos, never
opens action labels, and never reads the ``keys`` or old ``features`` members
of reference feature shards.  Reference shards provide only the explicit
session identity, engine-frame indices, and activity boundary metadata.

The production inventory names all 211 videos and all 1,554 sessions.  The
210 training videos are marked ``pretraining_train``; the y4n holdout must be
the sole ``downstream_export_only`` video.  The latter role allows final-EMA
feature export after SSL has ended, but never pretraining access.

Each feature-only shard and its provenance sidecar are written atomically.
Interrupted runs resume only from pairs whose hashes and contracts validate;
orphaned, mismatched, and temporary files fail closed for manual quarantine.
Arm D duplicates the current frame as its previous frame at the first sample
of *every* explicit session and never crosses a session boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import cv2
import numpy as np
import torch
from torch import nn

from badeline.dynamics_pretraining import (
    EMADynamicsPretrainer,
    REPRESENTATION_DIM,
)
from data.precompute_features import (
    FRAME_SIZE,
    MAX_SEQUENTIAL_GAP,
    _decode_resampled_part,
    _nominal_timeline_frames,
)


CHECKPOINT_SCHEMA = "madeleine.dynamics-streaming-checkpoint.v1"
INVENTORY_SCHEMA = "madeleine.dynamics-feature-export-inventory.v1"
SHARD_SIDECAR_SCHEMA = "madeleine.dynamics-feature-shard.v1"
MANIFEST_SCHEMA = "madeleine.dynamics-feature-export.v1"
COMPLETION_SCHEMA = "madeleine.dynamics-feature-export-complete.v1"

NATIVE_MODE = "opencv_native_60hz"
RESAMPLED_MODE = "ffmpeg_timestamp_resample_60hz"
TRAIN_ROLE = "pretraining_train"
EXPORT_ONLY_ROLE = "downstream_export_only"
Y4N_VIDEO_ID = "y4nQHqYSObI"
SEALED_UNTOUCHED_SESSION_ID = "rec_20260727_220000_test"
KNOWN_VAL_A_SESSION_ID = "rec_20260724_171305_5min"

PRODUCTION_VIDEO_COUNT = 211
PRODUCTION_SESSION_COUNT = 1_554
PRODUCTION_FRAME_COUNT = 32_598_000
PRODUCTION_TRAIN_VIDEO_COUNT = 210

HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]+__r[0-9]{3}$")


@dataclass(frozen=True)
class ExpectedCounts:
    videos: int
    sessions: int
    frames: int
    train_videos: int


PRODUCTION_COUNTS = ExpectedCounts(
    videos=PRODUCTION_VIDEO_COUNT,
    sessions=PRODUCTION_SESSION_COUNT,
    frames=PRODUCTION_FRAME_COUNT,
    train_videos=PRODUCTION_TRAIN_VIDEO_COUNT,
)


@dataclass(frozen=True)
class SessionSpec:
    video_id: str
    role: str
    session_id: str
    start_frame: int
    end_frame: int
    reference_shard: Path
    reference_shard_sha256: str

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class VideoSpec:
    video_id: str
    role: str
    video_path: Path
    video_sha256: str
    decoder_mode: str
    average_fps: float
    decoded_frames: int
    nominal_timeline_frames: int
    resolution_wh: tuple[int, int]
    mask_rect_xyxy: tuple[int, int, int, int]
    resized_mask_rect_xyxy: tuple[int, int, int, int]
    sessions: tuple[SessionSpec, ...]


@dataclass(frozen=True)
class Inventory:
    path: Path
    sha256: str
    videos: tuple[VideoSpec, ...]
    sessions: tuple[SessionSpec, ...]
    frames: int
    provenance: Mapping[str, Any] | None


@dataclass(frozen=True)
class CheckpointContract:
    path: Path
    sha256: str
    arm: str
    horizons: tuple[int, ...]
    completed_steps: int
    model_state: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash an array's exact dtype, shape, and C-order value bytes."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    header = {
        "dtype": array.dtype.str,
        "shape": [int(item) for item in array.shape],
    }
    digest.update(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    )
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _reject_forbidden_identity(value: str, *, name: str) -> None:
    """Reject embargoed/development identities before any data file access."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    folded = value.casefold()
    compact = re.sub(r"[^a-z0-9]+", "", folded)
    if folded in {
        SEALED_UNTOUCHED_SESSION_ID.casefold(),
        KNOWN_VAL_A_SESSION_ID.casefold(),
    }:
        raise ValueError(f"{name} identifies forbidden evaluation data")
    if "untouched" in folded:
        raise ValueError(f"{name} identifies forbidden untouched data")
    b1_identity = (
        folded == "b1"
        or folded.startswith(("b1_", "b1-", ".b1"))
        or compact.startswith(("b1pixels", "b1features", "b1engine"))
    )
    val_identity = re.match(r"^val[-_]?[ab](?:$|[-_])", folded) is not None
    if b1_identity or val_identity:
        raise ValueError(f"{name} identifies forbidden development data")


def _require_absolute_file_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be an absolute path string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute: {path}")
    for component in path.parts:
        _reject_forbidden_identity(component, name=name)
    return path


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_rect(value: Any, name: str, *, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must contain four integers")
    x0, y0, x1, y1 = (
        _require_int(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"{name} lies outside {width}x{height}")
    return x0, y0, x1, y1


def _parse_session(
    raw: Any, *, video_id: str, role: str, timeline_frames: int
) -> SessionSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"{video_id}: every session must be an object")
    expected_fields = {
        "session_id",
        "start_frame",
        "end_frame",
        "reference_shard",
        "reference_shard_sha256",
    }
    if set(raw) != expected_fields:
        raise ValueError(
            f"{video_id}: session fields differ: "
            f"missing={sorted(expected_fields-set(raw))} "
            f"extra={sorted(set(raw)-expected_fields)}"
        )
    session_id = raw["session_id"]
    _reject_forbidden_identity(session_id, name="session_id")
    if SESSION_ID.fullmatch(session_id) is None:
        raise ValueError(f"invalid explicit session_id: {session_id}")
    if not session_id.startswith(f"{video_id}__r"):
        raise ValueError(f"{session_id}: session does not belong to {video_id}")
    start = _require_int(raw["start_frame"], f"{session_id}.start_frame")
    end = _require_int(raw["end_frame"], f"{session_id}.end_frame", minimum=1)
    if start >= end or end > timeline_frames:
        raise ValueError(
            f"{session_id}: invalid [{start},{end}) within {timeline_frames}"
        )
    reference = _require_absolute_file_path(
        raw["reference_shard"], f"{session_id}.reference_shard"
    )
    if reference.suffix != ".npz":
        raise ValueError(f"{session_id}: reference shard must be NPZ")
    if reference.stem != session_id:
        raise ValueError(f"{session_id}: reference filename stem differs")
    return SessionSpec(
        video_id=video_id,
        role=role,
        session_id=session_id,
        start_frame=start,
        end_frame=end,
        reference_shard=reference,
        reference_shard_sha256=_require_sha256(
            raw["reference_shard_sha256"],
            f"{session_id}.reference_shard_sha256",
        ),
    )


def validate_inventory_payload(
    payload: Any,
    *,
    path: Path,
    sha256: str,
    expected_counts: ExpectedCounts = PRODUCTION_COUNTS,
) -> Inventory:
    """Validate all identities and counts without opening named data files."""

    if not isinstance(payload, dict):
        raise ValueError("inventory root must be an object")
    if payload.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError(f"inventory schema must equal {INVENTORY_SCHEMA}")
    allowed_root = {"schema_version", "population", "videos", "provenance"}
    if not {"schema_version", "population", "videos"}.issubset(payload):
        raise ValueError("inventory root lacks schema/population/videos")
    if not set(payload).issubset(allowed_root):
        raise ValueError("inventory root contains unsupported fields")
    provenance = payload.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("inventory provenance must be an object")
    if expected_counts == PRODUCTION_COUNTS and provenance is None:
        raise ValueError("production inventory requires terminal-build provenance")
    if expected_counts == PRODUCTION_COUNTS:
        assert isinstance(provenance, dict)
        terminal = provenance.get("terminal_checkpoint")
        if not isinstance(terminal, dict) or terminal.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("production inventory terminal provenance differs")
        _require_sha256(terminal.get("sha256"), "terminal checkpoint SHA-256")
        if terminal.get("arm") not in {"C", "D"}:
            raise ValueError("production inventory terminal arm differs")
        _require_int(
            terminal.get("completed_steps"),
            "terminal completed_steps",
            minimum=1,
        )
        ssl = provenance.get("ssl_inventory")
        if not isinstance(ssl, dict):
            raise ValueError("production inventory lacks SSL provenance")
        _require_sha256(ssl.get("sha256"), "SSL inventory SHA-256")
        if ssl.get("train_video_hashes_reused_without_rehash") is not True:
            raise ValueError("production train-video hashes were not reused")
        validation = provenance.get("full_corpus_validation")
        if not isinstance(validation, dict):
            raise ValueError("production inventory lacks corpus-validation provenance")
        _require_sha256(validation.get("sha256"), "corpus validation SHA-256")
        if validation.get("ok") is not True or validation.get("deep_shards") is not True:
            raise ValueError("production inventory lacks passing deep corpus validation")
        for field in (
            "full_corpus_manifest_sha256",
            "full_corpus_shard_hashes_sha256",
            "fetch_report_sha256",
        ):
            _require_sha256(provenance.get(field), field)
        if provenance.get("y4n_hashed_after_terminal_checkpoint_validation") is not True:
            raise ValueError("production inventory lacks post-terminal y4n proof")
    population = payload["population"]
    if not isinstance(population, dict) or set(population) != {
        "videos", "sessions", "frames", "train_videos"
    }:
        raise ValueError("inventory population fields differ from contract")
    declared = ExpectedCounts(
        videos=_require_int(population["videos"], "population.videos", minimum=1),
        sessions=_require_int(population["sessions"], "population.sessions", minimum=1),
        frames=_require_int(population["frames"], "population.frames", minimum=1),
        train_videos=_require_int(
            population["train_videos"], "population.train_videos", minimum=0
        ),
    )
    if declared != expected_counts:
        raise ValueError(
            f"inventory population {declared} differs from required {expected_counts}"
        )
    raw_videos = payload["videos"]
    if not isinstance(raw_videos, list):
        raise ValueError("inventory videos must be a list")

    videos: list[VideoSpec] = []
    sessions: list[SessionSpec] = []
    seen_videos: set[str] = set()
    seen_sessions: set[str] = set()
    seen_paths: set[Path] = set()
    for raw in raw_videos:
        if not isinstance(raw, dict):
            raise ValueError("every video inventory row must be an object")
        expected_fields = {
            "video_id",
            "role",
            "video_path",
            "video_sha256",
            "decoder_mode",
            "video",
            "mask_rect_xyxy",
            "resized_mask_rect_xyxy",
            "sessions",
        }
        if set(raw) != expected_fields:
            raise ValueError("video inventory fields differ from contract")
        video_id = raw["video_id"]
        _reject_forbidden_identity(video_id, name="video_id")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
            raise ValueError(f"invalid video_id: {video_id}")
        if video_id in seen_videos:
            raise ValueError(f"duplicate video_id: {video_id}")
        seen_videos.add(video_id)
        role = raw["role"]
        if role not in {TRAIN_ROLE, EXPORT_ONLY_ROLE}:
            raise ValueError(f"{video_id}: invalid role {role!r}")
        if (video_id == Y4N_VIDEO_ID) != (role == EXPORT_ONLY_ROLE):
            raise ValueError(
                f"{video_id}: y4n must be the sole downstream-export-only video"
            )
        video_path = _require_absolute_file_path(
            raw["video_path"], f"{video_id}.video_path"
        )
        if video_path in seen_paths:
            raise ValueError(f"duplicate explicit video path: {video_path}")
        seen_paths.add(video_path)
        metadata = raw["video"]
        if not isinstance(metadata, dict) or set(metadata) != {
            "average_fps",
            "decoded_frames",
            "nominal_timeline_frames",
            "resolution_wh",
        }:
            raise ValueError(f"{video_id}: video metadata fields differ")
        average_fps = metadata["average_fps"]
        if (
            isinstance(average_fps, bool)
            or not isinstance(average_fps, (int, float))
            or not math.isfinite(float(average_fps))
            or float(average_fps) <= 0
        ):
            raise ValueError(f"{video_id}: invalid average_fps")
        decoded = _require_int(
            metadata["decoded_frames"], f"{video_id}.decoded_frames", minimum=1
        )
        timeline = _require_int(
            metadata["nominal_timeline_frames"],
            f"{video_id}.nominal_timeline_frames",
            minimum=1,
        )
        resolution = metadata["resolution_wh"]
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise ValueError(f"{video_id}: resolution_wh must have two integers")
        width = _require_int(resolution[0], f"{video_id}.width", minimum=1)
        height = _require_int(resolution[1], f"{video_id}.height", minimum=1)
        decoder_mode = raw["decoder_mode"]
        if decoder_mode not in {NATIVE_MODE, RESAMPLED_MODE}:
            raise ValueError(f"{video_id}: invalid decoder mode")
        should_resample, computed_timeline = _nominal_timeline_frames(
            decoded, float(average_fps)
        )
        computed_mode = RESAMPLED_MODE if should_resample else NATIVE_MODE
        if decoder_mode != computed_mode or timeline != computed_timeline:
            raise ValueError(f"{video_id}: decoder plan differs from metadata")
        mask = _require_rect(
            raw["mask_rect_xyxy"],
            f"{video_id}.mask_rect_xyxy",
            width=width,
            height=height,
        )
        small_mask = _require_rect(
            raw["resized_mask_rect_xyxy"],
            f"{video_id}.resized_mask_rect_xyxy",
            width=FRAME_SIZE,
            height=FRAME_SIZE,
        )
        expected_small = (
            max(0, int(mask[0] / width * FRAME_SIZE) - 1),
            max(0, int(mask[1] / height * FRAME_SIZE) - 1),
            min(FRAME_SIZE, int(np.ceil(mask[2] / width * FRAME_SIZE)) + 1),
            min(FRAME_SIZE, int(np.ceil(mask[3] / height * FRAME_SIZE)) + 1),
        )
        if small_mask != expected_small:
            raise ValueError(f"{video_id}: resized mask differs from contract")
        raw_sessions = raw["sessions"]
        if not isinstance(raw_sessions, list) or not raw_sessions:
            raise ValueError(f"{video_id}: sessions must be a non-empty list")
        video_sessions = tuple(
            _parse_session(
                item, video_id=video_id, role=role, timeline_frames=timeline
            )
            for item in raw_sessions
        )
        expected_ids = [f"{video_id}__r{index:03d}" for index in range(len(video_sessions))]
        if [item.session_id for item in video_sessions] != expected_ids:
            raise ValueError(f"{video_id}: sessions are not exact ordered r000.. sequence")
        previous_end = -1
        for session in video_sessions:
            if session.session_id in seen_sessions:
                raise ValueError(f"duplicate session_id: {session.session_id}")
            seen_sessions.add(session.session_id)
            if session.start_frame < previous_end:
                raise ValueError(f"{session.session_id}: sessions overlap/out of order")
            previous_end = session.end_frame
            sessions.append(session)
        videos.append(
            VideoSpec(
                video_id=video_id,
                role=role,
                video_path=video_path,
                video_sha256=_require_sha256(
                    raw["video_sha256"], f"{video_id}.video_sha256"
                ),
                decoder_mode=decoder_mode,
                average_fps=float(average_fps),
                decoded_frames=decoded,
                nominal_timeline_frames=timeline,
                resolution_wh=(width, height),
                mask_rect_xyxy=mask,
                resized_mask_rect_xyxy=small_mask,
                sessions=video_sessions,
            )
        )

    actual = ExpectedCounts(
        videos=len(videos),
        sessions=len(sessions),
        frames=sum(item.frames for item in sessions),
        train_videos=sum(item.role == TRAIN_ROLE for item in videos),
    )
    if actual != expected_counts:
        raise ValueError(f"actual inventory {actual} differs from {expected_counts}")
    if sum(item.role == EXPORT_ONLY_ROLE for item in videos) != 1:
        raise ValueError("exactly one downstream-export-only video is required")
    return Inventory(
        path=path,
        sha256=sha256,
        videos=tuple(videos),
        sessions=tuple(sessions),
        frames=actual.frames,
        provenance=provenance,
    )


def load_inventory(
    path: Path,
    expected_sha256: str,
    *,
    expected_counts: ExpectedCounts = PRODUCTION_COUNTS,
) -> Inventory:
    expected_sha256 = _require_sha256(expected_sha256, "inventory SHA-256")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"inventory is absent: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError("inventory SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_inventory_payload(
        payload,
        path=path.resolve(),
        sha256=actual,
        expected_counts=expected_counts,
    )


def load_checkpoint_contract(
    path: Path,
    expected_sha256: str,
    *,
    expected_arm: str,
    expected_completed_steps: int,
) -> CheckpointContract:
    """Verify and parse the strict production streaming checkpoint schema."""

    expected_sha256 = _require_sha256(expected_sha256, "checkpoint SHA-256")
    if expected_arm not in {"C", "D"}:
        raise ValueError("feature exporter supports only Arm C or D")
    if expected_completed_steps < 1:
        raise ValueError("expected completed steps must be positive")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is absent: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a dictionary")
    required = {
        "schema_version",
        "model_state",
        "arm",
        "horizons",
        "completed_steps",
        "kind",
        "selection_eligible",
        "resumable",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint missing required fields: {sorted(missing)}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError(f"checkpoint schema must equal {CHECKPOINT_SCHEMA}")
    if payload["kind"] != "final":
        raise ValueError("feature export requires a terminal final checkpoint")
    if payload["selection_eligible"] is not True or payload["resumable"] is not False:
        raise ValueError("feature export requires final-only checkpoint flags")
    arm = payload["arm"]
    if arm != expected_arm:
        raise ValueError(f"checkpoint arm {arm!r} differs from {expected_arm!r}")
    horizons_raw = payload["horizons"]
    if not isinstance(horizons_raw, (list, tuple)):
        raise ValueError("checkpoint horizons must be a list or tuple")
    horizons = tuple(_require_int(item, "checkpoint horizon", minimum=1) for item in horizons_raw)
    if not horizons or len(horizons) != len(set(horizons)):
        raise ValueError("checkpoint horizons must be unique and non-empty")
    completed = _require_int(
        payload["completed_steps"], "checkpoint completed_steps", minimum=1
    )
    if completed != expected_completed_steps:
        raise ValueError(
            f"checkpoint completed {completed} steps, expected {expected_completed_steps}"
        )
    state = payload["model_state"]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint model_state must be a non-empty mapping")
    return CheckpointContract(
        path=path.resolve(),
        sha256=actual,
        arm=arm,
        horizons=horizons,
        completed_steps=completed,
        model_state=state,
    )


def instantiate_final_ema_target(
    contract: CheckpointContract,
    *,
    device: torch.device,
) -> nn.Module:
    model = EMADynamicsPretrainer(
        contract.arm,  # type: ignore[arg-type]
        horizons=contract.horizons,
        weights=None,
    )
    model.load_state_dict(contract.model_state, strict=True)
    target = model.target_encoder.eval().requires_grad_(False).to(device)
    return target


@torch.inference_mode()
def encode_rgb_frames(
    frames_rgb: np.ndarray,
    *,
    target_encoder: nn.Module,
    arm: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return raw, unnormalized final-EMA features for one bounded session."""

    if arm not in {"C", "D"}:
        raise ValueError("arm must be C or D")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if (
        frames_rgb.dtype != np.uint8
        or frames_rgb.ndim != 4
        or frames_rgb.shape[1:] != (FRAME_SIZE, FRAME_SIZE, 3)
        or len(frames_rgb) < 1
    ):
        raise ValueError("frames must be non-empty uint8 [N,128,128,3]")
    output = np.empty((len(frames_rgb), REPRESENTATION_DIM), dtype=np.float16)
    for start in range(0, len(frames_rgb), batch_size):
        stop = min(start + batch_size, len(frames_rgb))
        current = (
            torch.from_numpy(frames_rgb[start:stop].copy())
            .permute(0, 3, 1, 2)
            .to(device=device, dtype=torch.float32)
            .div_(255.0)
        )
        if arm == "C":
            latent = target_encoder(current)
        else:
            if start == 0:
                previous_rgb = np.concatenate(
                    (frames_rgb[:1], frames_rgb[: stop - 1]), axis=0
                )
            else:
                previous_rgb = frames_rgb[start - 1 : stop - 1]
            previous = (
                torch.from_numpy(previous_rgb.copy())
                .permute(0, 3, 1, 2)
                .to(device=device, dtype=torch.float32)
                .div_(255.0)
            )
            latent = target_encoder(previous, current)
        if latent.shape != (stop - start, REPRESENTATION_DIM):
            raise RuntimeError(f"target encoder returned shape {tuple(latent.shape)}")
        if not bool(torch.isfinite(latent).all().item()):
            raise RuntimeError("target encoder emitted non-finite float32 features")
        encoded = latent.to(torch.float16).cpu().numpy()
        if not np.isfinite(encoded).all():
            raise RuntimeError("target encoder overflowed float16 feature export")
        output[start:stop] = encoded
    return output


def _npz_headers(path: Path) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    result: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
    with zipfile.ZipFile(path) as archive:
        expected = {
            "features.npy", "engine_frame_idx.npy", "input_active.npy", "session_id.npy"
        }
        names = set(archive.namelist())
        if names != expected:
            raise ValueError(
                f"{path}: output member set differs: "
                f"missing={sorted(expected-names)} extra={sorted(names-expected)}"
            )
        for member in names:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version in {(2, 0), (3, 0)}:
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"{path}: unsupported NPY version {version}")
            result[member[:-4]] = (tuple(int(x) for x in shape), np.dtype(dtype))
    return result


def load_reference_metadata(session: SessionSpec) -> tuple[np.ndarray, np.ndarray]:
    """Load only non-label metadata arrays from one exact reference shard.

    The byte-level file hash binds the explicit reference artifact but no old
    feature or key member is decoded or indexed by this function.
    """

    path = session.reference_shard
    if not path.is_file():
        raise FileNotFoundError(f"missing reference shard: {path}")
    if sha256_file(path) != session.reference_shard_sha256:
        raise ValueError(f"{session.session_id}: reference shard SHA mismatch")
    with np.load(path, allow_pickle=False) as archive:
        required = {"engine_frame_idx", "input_active", "session_id"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing metadata arrays {sorted(missing)}")
        stored = archive["session_id"]
        engine = np.asarray(archive["engine_frame_idx"])
        active = np.asarray(archive["input_active"])
    if stored.size != 1 or stored.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{path}: session_id must be one string")
    if str(stored.reshape(()).item()) != session.session_id:
        raise ValueError(f"{path}: stored session_id differs")
    expected_engine = np.arange(
        session.start_frame, session.end_frame, dtype=np.int64
    )
    if engine.dtype != np.int64 or not np.array_equal(engine, expected_engine):
        raise ValueError(f"{path}: engine_frame_idx differs from explicit range")
    if active.dtype != np.uint8 or active.shape != (session.frames,):
        raise ValueError(f"{path}: input_active has invalid schema")
    if not np.all(active == 1):
        raise ValueError(f"{path}: inactive frames require a separate session boundary")
    return engine.copy(), active.copy()


def _decode_native_session(
    capture: cv2.VideoCapture,
    *,
    cursor: int,
    session: SessionSpec,
    mask_xyxy: tuple[int, int, int, int],
    resized_mask_xyxy: tuple[int, int, int, int],
) -> tuple[np.ndarray, int]:
    start, end = session.start_frame, session.end_frame
    if cursor >= 0 and 0 < start - cursor <= MAX_SEQUENTIAL_GAP:
        while cursor < start:
            if not capture.grab():
                raise RuntimeError(f"{session.video_id}: ended before frame {start}")
            cursor += 1
    elif cursor != start:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise RuntimeError(f"{session.video_id}: failed seek to {start}")
        reported = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        if reported != start:
            raise RuntimeError(
                f"{session.video_id}: seek landed at {reported}, expected {start}"
            )
        cursor = start
    frames = np.empty((session.frames, FRAME_SIZE, FRAME_SIZE, 3), np.uint8)
    x0, y0, x1, y1 = mask_xyxy
    sx0, sy0, sx1, sy1 = resized_mask_xyxy
    for index in range(session.frames):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"{session.video_id}: decode failed at {cursor}")
        cursor += 1
        frame[y0:y1, x0:x1] = 0
        small = cv2.resize(
            frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA
        )
        small[sy0:sy1, sx0:sx1] = 0
        frames[index] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return frames, cursor


def _verify_video_metadata(video: VideoSpec, capture: cv2.VideoCapture) -> None:
    if not capture.isOpened():
        raise ValueError(f"cannot open explicit source video: {video.video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isclose(fps, video.average_fps, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{video.video_id}: source FPS changed")
    if frames != video.decoded_frames or (width, height) != video.resolution_wh:
        raise ValueError(f"{video.video_id}: source video metadata changed")
    resample, timeline = _nominal_timeline_frames(frames, fps)
    mode = RESAMPLED_MODE if resample else NATIVE_MODE
    if mode != video.decoder_mode or timeline != video.nominal_timeline_frames:
        raise ValueError(f"{video.video_id}: source decoder plan changed")


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(".tmp.json")
    if temporary.exists():
        raise FileExistsError(f"refusing existing temporary file: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_feature_only_shard(
    path: Path,
    *,
    features: np.ndarray,
    engine_frame_idx: np.ndarray,
    input_active: np.ndarray,
    session_id: str,
) -> None:
    temporary = path.with_suffix(".tmp.npz")
    if temporary.exists():
        raise FileExistsError(f"refusing existing temporary file: {temporary}")
    np.savez(
        temporary,
        features=np.asarray(features, dtype=np.float16),
        engine_frame_idx=np.asarray(engine_frame_idx, dtype=np.int64),
        input_active=np.asarray(input_active, dtype=np.uint8),
        session_id=np.asarray(session_id),
    )
    temporary.replace(path)


def _expected_previous_index_sha256(engine: np.ndarray) -> str:
    previous = engine.copy()
    previous[1:] = engine[:-1]
    return array_sha256(previous)


def _validate_resumable_pair(
    shard: Path,
    sidecar: Path,
    *,
    session: SessionSpec,
    checkpoint: CheckpointContract,
    inventory: Inventory,
) -> dict[str, Any] | None:
    if not shard.exists() and not sidecar.exists():
        return None
    if not shard.is_file() or not sidecar.is_file():
        raise ValueError(f"{session.session_id}: orphaned output requires quarantine")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SHARD_SIDECAR_SCHEMA:
        raise ValueError(f"{session.session_id}: sidecar schema mismatch")
    bindings = {
        "session_id": session.session_id,
        "arm": checkpoint.arm,
        "checkpoint_sha256": checkpoint.sha256,
        "inventory_sha256": inventory.sha256,
        "reference_shard_sha256": session.reference_shard_sha256,
        "source_frame_range": [session.start_frame, session.end_frame],
    }
    for key, expected in bindings.items():
        if metadata.get(key) != expected:
            raise ValueError(f"{session.session_id}: resumed {key} mismatch")
    if metadata.get("npz_sha256") != sha256_file(shard):
        raise ValueError(f"{session.session_id}: resumed NPZ SHA mismatch")
    headers = _npz_headers(shard)
    if headers.get("features") != ((session.frames, REPRESENTATION_DIM), np.dtype(np.float16)):
        raise ValueError(f"{session.session_id}: resumed feature header mismatch")
    return metadata


def _export_one_session(
    *,
    session: SessionSpec,
    video: VideoSpec,
    frames: np.ndarray,
    target_encoder: nn.Module,
    checkpoint: CheckpointContract,
    inventory: Inventory,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    imputed_tail_frames: int,
) -> dict[str, Any]:
    shard = out_dir / f"{session.session_id}.npz"
    sidecar = out_dir / f"{session.session_id}.json"
    resumed = _validate_resumable_pair(
        shard,
        sidecar,
        session=session,
        checkpoint=checkpoint,
        inventory=inventory,
    )
    if resumed is not None:
        return resumed
    engine, active = load_reference_metadata(session)
    features = encode_rgb_frames(
        frames,
        target_encoder=target_encoder,
        arm=checkpoint.arm,
        device=device,
        batch_size=batch_size,
    )
    _write_feature_only_shard(
        shard,
        features=features,
        engine_frame_idx=engine,
        input_active=active,
        session_id=session.session_id,
    )
    metadata: dict[str, Any] = {
        "schema_version": SHARD_SIDECAR_SCHEMA,
        "session_id": session.session_id,
        "video_id": session.video_id,
        "role": session.role,
        "arm": checkpoint.arm,
        "checkpoint_sha256": checkpoint.sha256,
        "inventory_sha256": inventory.sha256,
        "reference_shard": str(session.reference_shard),
        "reference_shard_sha256": session.reference_shard_sha256,
        "source_video_sha256": video.video_sha256,
        "source_frame_range": [session.start_frame, session.end_frame],
        "frames": session.frames,
        "decoder_mode": video.decoder_mode,
        "imputed_tail_frames": int(imputed_tail_frames),
        "feature_format": f"dynamics_{checkpoint.arm.lower()}_final_ema_raw_avgpool_float16_v1",
        "feature_dim": REPRESENTATION_DIM,
        "normalization": "none_raw_target_encoder_output",
        "supervision_phase": "label_free_feature_export_no_keys_member_read_or_written",
        "D_boundary_policy": (
            "previous_equals_current_at_explicit_session_start_then_prior_frame"
            if checkpoint.arm == "D"
            else None
        ),
        "previous_engine_frame_idx_sha256": (
            _expected_previous_index_sha256(engine)
            if checkpoint.arm == "D"
            else None
        ),
        "arrays": {
            "features_sha256": array_sha256(features),
            "engine_frame_idx_sha256": array_sha256(engine),
            "input_active_sha256": array_sha256(active),
        },
        "npz": shard.name,
        "npz_sha256": sha256_file(shard),
    }
    _write_json_atomic(sidecar, metadata)
    return metadata


def export_inventory(
    *,
    inventory: Inventory,
    checkpoint: CheckpointContract,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if inventory.provenance is not None:
        terminal = inventory.provenance.get("terminal_checkpoint")
        expected_terminal = {
            "schema_version": CHECKPOINT_SCHEMA,
            "sha256": checkpoint.sha256,
            "arm": checkpoint.arm,
            "completed_steps": checkpoint.completed_steps,
        }
        if terminal != expected_terminal:
            raise ValueError("inventory terminal-checkpoint provenance mismatch")
        if inventory.provenance.get(
            "y4n_hashed_after_terminal_checkpoint_validation"
        ) is not True:
            raise ValueError("inventory lacks post-terminal y4n hash proof")
    out_dir = Path(out_dir).resolve()
    for component in out_dir.parts:
        _reject_forbidden_identity(component, name="output path")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "feature_export_manifest.json"
    completion_path = out_dir / "feature_export_complete.json"
    if manifest_path.exists() or completion_path.exists():
        raise FileExistsError(
            "completed output exists; run the independent validator instead"
        )
    unexpected_temps = sorted(out_dir.glob("*.tmp.*"))
    if unexpected_temps:
        raise FileExistsError(f"temporary outputs require quarantine: {unexpected_temps[0]}")
    target = instantiate_final_ema_target(checkpoint, device=device)
    records: list[dict[str, Any]] = []
    for video in inventory.videos:
        if not video.video_path.is_file():
            raise FileNotFoundError(f"missing source video: {video.video_path}")
        if sha256_file(video.video_path) != video.video_sha256:
            raise ValueError(f"{video.video_id}: source video SHA mismatch")
        capture = cv2.VideoCapture(str(video.video_path))
        try:
            _verify_video_metadata(video, capture)
            cursor = 0
            for session in video.sessions:
                shard = out_dir / f"{session.session_id}.npz"
                sidecar = out_dir / f"{session.session_id}.json"
                resumed = _validate_resumable_pair(
                    shard,
                    sidecar,
                    session=session,
                    checkpoint=checkpoint,
                    inventory=inventory,
                )
                if resumed is not None:
                    records.append(resumed)
                    cursor = -1
                    continue
                if video.decoder_mode == RESAMPLED_MODE:
                    frames, imputed = _decode_resampled_part(
                        video.video_path,
                        start_frame=session.start_frame,
                        end_frame=session.end_frame,
                        mask_xyxy=video.resized_mask_rect_xyxy,
                    )
                    cursor = -1
                else:
                    frames, cursor = _decode_native_session(
                        capture,
                        cursor=cursor,
                        session=session,
                        mask_xyxy=video.mask_rect_xyxy,
                        resized_mask_xyxy=video.resized_mask_rect_xyxy,
                    )
                    imputed = 0
                sx0, sy0, sx1, sy1 = video.resized_mask_rect_xyxy
                if int(frames[:, sy0:sy1, sx0:sx1].max(initial=0)) != 0:
                    raise AssertionError(f"{video.video_id}: resized controller mask is not black")
                records.append(
                    _export_one_session(
                        session=session,
                        video=video,
                        frames=frames,
                        target_encoder=target,
                        checkpoint=checkpoint,
                        inventory=inventory,
                        out_dir=out_dir,
                        device=device,
                        batch_size=batch_size,
                        imputed_tail_frames=imputed,
                    )
                )
        finally:
            capture.release()
    if len(records) != len(inventory.sessions):
        raise RuntimeError("export record count differs from explicit inventory")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "arm": checkpoint.arm,
        "checkpoint": {
            "path": str(checkpoint.path),
            "sha256": checkpoint.sha256,
            "schema_version": CHECKPOINT_SCHEMA,
            "completed_steps": checkpoint.completed_steps,
            "horizons": list(checkpoint.horizons),
            "selection": "final_weights_only",
            "encoder_state": "final_ema_target_only",
        },
        "inventory": {"path": str(inventory.path), "sha256": inventory.sha256},
        "feature_format": f"dynamics_{checkpoint.arm.lower()}_final_ema_raw_avgpool_float16_v1",
        "feature_dim": REPRESENTATION_DIM,
        "dtype": "float16",
        "normalization": "none_raw_target_encoder_output",
        "supervision_phase": "label_free_feature_export_no_keys_member_read_or_written",
        "y4n_policy": "downstream_export_only_after_terminal_ssl_never_pretraining",
        "D_boundary_policy": (
            "previous_equals_current_at_explicit_session_start_then_prior_frame"
            if checkpoint.arm == "D"
            else None
        ),
        "counts": {
            "videos": len(inventory.videos),
            "sessions": len(records),
            "frames": sum(int(item["frames"]) for item in records),
            "train_videos": sum(item.role == TRAIN_ROLE for item in inventory.videos),
            "downstream_export_only_videos": sum(
                item.role == EXPORT_ONLY_ROLE for item in inventory.videos
            ),
        },
        "sessions": records,
    }
    _write_json_atomic(manifest_path, manifest)
    completion = {
        "schema_version": COMPLETION_SCHEMA,
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "inventory_sha256": inventory.sha256,
        "checkpoint_sha256": checkpoint.sha256,
        "arm": checkpoint.arm,
        "counts": manifest["counts"],
    }
    _write_json_atomic(completion_path, completion)
    return completion


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--arm", choices=("C", "D"), required=True)
    parser.add_argument("--expected-completed-steps", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    inventory = load_inventory(args.inventory, args.inventory_sha256)
    checkpoint = load_checkpoint_contract(
        args.checkpoint,
        args.checkpoint_sha256,
        expected_arm=args.arm,
        expected_completed_steps=args.expected_completed_steps,
    )
    completion = export_inventory(
        inventory=inventory,
        checkpoint=checkpoint,
        out_dir=args.out,
        device=_device(args.device),
        batch_size=args.batch_size,
    )
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
