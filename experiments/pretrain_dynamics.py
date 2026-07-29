#!/usr/bin/env python3
"""Fail-closed real-data smoke trainer for Celeste dynamics pretraining.

The trainer intentionally accepts only explicitly named RGB NPZ shards and an
explicit allowlist of session IDs.  It does not scan a directory, and it never
retains, indexes, or uses the values in the ``keys`` array; NumPy necessarily
decompresses that member while its dtype and shape are checked as part of the
source-shard schema.  Temporal samples are bounded
by both consecutive ``engine_frame_idx`` runs and ``input_active`` runs.

Arms B/C/D are implemented by :class:`badeline.dynamics_pretraining.
EMADynamicsPretrainer`.  This executable owns the parts that deliberately stay
outside that model primitive: data contracts, sampling, tuple-consistent
augmentations, the EMA schedule, and atomic evidence publication.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

# Required by deterministic CUDA matrix multiplication.  It must be set before
# the first CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights

from badeline.dynamics_pretraining import (
    CollapseDiagnostics,
    DynamicsArm,
    EMADynamicsPretrainer,
    collapse_diagnostics,
)
from data.schema import KEY_ORDER


CHECKPOINT_SCHEMA = "madeleine.dynamics-pretraining-checkpoint.v1"
RECEIPT_SCHEMA = "madeleine.dynamics-pretraining-smoke.v1"
KNOWN_EMBARGOED_SESSION_IDS = frozenset({"rec_20260727_220000_test"})
EXPECTED_FRAME_SHAPE = (128, 128, 3)
RESNET18_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
RESNET18_WEIGHTS_URL = (
    "https://download.pytorch.org/models/resnet18-f37072fd.pth"
)
RESNET18_WEIGHTS_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)


@dataclass(frozen=True)
class RGBShard:
    """One validated masked RGB shard; action values are intentionally absent."""

    session_id: str
    path: Path
    sha256: str
    frames: np.ndarray
    engine_frame_idx: np.ndarray
    input_active: np.ndarray


@dataclass(frozen=True)
class CandidateIndex:
    """Columnar, session-bounded temporal samples and their frozen strata."""

    session: np.ndarray
    online_previous: np.ndarray
    online_current: np.ndarray
    target_previous: np.ndarray
    target_current: np.ndarray
    horizon: np.ndarray
    motion_score: np.ndarray
    motion_threshold: float
    stratum: np.ndarray

    def __post_init__(self) -> None:
        length = len(self.session)
        for name in (
            "online_previous",
            "online_current",
            "target_previous",
            "target_current",
            "horizon",
            "motion_score",
            "stratum",
        ):
            if len(getattr(self, name)) != length:
                raise ValueError(f"candidate column {name} has inconsistent length")

    def __len__(self) -> int:
        return int(len(self.session))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_imagenet_initialization() -> dict[str, Any]:
    """Download-if-needed, then validate the exact preregistered weight bytes."""

    if RESNET18_WEIGHTS.url != RESNET18_WEIGHTS_URL:
        raise RuntimeError("torchvision ResNet-18 weight URL differs from contract")
    # This is the standard torchvision cache path used by get_state_dict and
    # model construction.  check_hash also checks the filename's abbreviated
    # hash; MADELEINE additionally checks the complete preregistered digest.
    RESNET18_WEIGHTS.get_state_dict(progress=True, check_hash=True)
    filename = Path(urlparse(RESNET18_WEIGHTS.url).path).name
    cache = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not cache.is_file():
        raise FileNotFoundError(f"torchvision weight cache is absent: {cache}")
    actual = sha256_file(cache)
    if actual != RESNET18_WEIGHTS_SHA256:
        raise RuntimeError(
            "cached ResNet-18 ImageNet weight SHA-256 differs from contract"
        )
    return {
        "architecture": "torchvision resnet18",
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "url": RESNET18_WEIGHTS_URL,
        "cached_path": str(cache.resolve()),
        "sha256": actual,
        "validated_before_step_one": True,
    }


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_forbidden_identity(value: str) -> None:
    folded = value.casefold()
    if value in KNOWN_EMBARGOED_SESSION_IDS or "untouched" in folded:
        raise ValueError(f"embargoed/untouched session is forbidden: {value}")


def validate_explicit_inputs(
    shard_paths: Sequence[Path],
    allowed_session_ids: Sequence[str],
    expected_sha256: Mapping[str, str] | None = None,
) -> None:
    """Validate identities before any NPZ path is opened."""

    if not shard_paths:
        raise ValueError("at least one explicit --shard is required")
    if not allowed_session_ids:
        raise ValueError("at least one explicit --allowed-session-id is required")
    if len(shard_paths) != len(set(map(str, shard_paths))):
        raise ValueError("duplicate explicit shard path")
    if len(allowed_session_ids) != len(set(allowed_session_ids)):
        raise ValueError("duplicate allowed session ID")
    for session_id in allowed_session_ids:
        _reject_forbidden_identity(session_id)
    for path in shard_paths:
        _reject_forbidden_identity(Path(path).stem)
        if Path(path).suffix != ".npz":
            raise ValueError(f"explicit shard must be an NPZ file: {path}")
    if len(shard_paths) != len(allowed_session_ids):
        raise ValueError("explicit shard count must equal allowed session-ID count")
    if expected_sha256 is not None:
        if set(expected_sha256) != set(allowed_session_ids):
            raise ValueError(
                "expected SHA-256 map must bind every and only allowed session ID"
            )
        for session_id, digest in expected_sha256.items():
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"invalid lowercase SHA-256 for {session_id}")


def load_explicit_rgb_shards(
    shard_paths: Sequence[Path],
    *,
    allowed_session_ids: Sequence[str],
    expected_sha256: Mapping[str, str] | None = None,
) -> list[RGBShard]:
    """Load exactly the paths supplied by the caller, with no discovery.

    ``keys`` must exist with the expected dtype and shape because it is part of
    the source-shard schema.  Loading the member for that schema check
    decompresses it, but its values are never indexed, converted, checked for
    binary membership, retained, or returned.
    """

    paths = [Path(path) for path in shard_paths]
    validate_explicit_inputs(paths, allowed_session_ids, expected_sha256)
    allowed = set(allowed_session_ids)
    observed: set[str] = set()
    result: list[RGBShard] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing explicit RGB shard: {path}")
        digest = sha256_file(path)
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "frames",
                "keys",
                "engine_frame_idx",
                "input_active",
                "session_id",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{path}: missing arrays: {sorted(missing)}")
            stored = archive["session_id"]
            if stored.size != 1 or stored.dtype.kind not in ("U", "S"):
                raise ValueError(f"{path}: session_id must be one string")
            session_id = str(stored.reshape(()).item())
            _reject_forbidden_identity(session_id)
            if session_id not in allowed:
                raise ValueError(f"{path}: session_id is not explicitly allowed")
            if session_id in observed:
                raise ValueError(f"duplicate loaded session ID: {session_id}")
            if path.stem != session_id:
                raise ValueError(
                    f"{path}: filename stem does not match stored session_id"
                )

            frames = archive["frames"]
            keys = archive["keys"]  # Schema only: values are never consumed.
            engine = archive["engine_frame_idx"]
            active = archive["input_active"]
            if frames.dtype != np.uint8 or frames.ndim != 4:
                raise ValueError(f"{path}: frames must be uint8 [N,128,128,3]")
            if frames.shape[1:] != EXPECTED_FRAME_SHAPE:
                raise ValueError(f"{path}: frames must have shape [N,128,128,3]")
            if keys.dtype != np.uint8 or keys.shape != (
                len(frames),
                len(KEY_ORDER),
            ):
                raise ValueError(
                    f"{path}: keys must be uint8 [N,{len(KEY_ORDER)}]"
                )
            if engine.dtype != np.int64 or engine.shape != (len(frames),):
                raise ValueError(f"{path}: engine_frame_idx must be int64 [N]")
            if active.dtype != np.uint8 or active.shape != (len(frames),):
                raise ValueError(f"{path}: input_active must be uint8 [N]")
            # input_active is a sampling boundary, so unlike keys its values are
            # necessarily consumed and validated.
            if not np.all(np.isin(active, (0, 1))):
                raise ValueError(f"{path}: input_active must be binary")
            if not len(frames):
                raise ValueError(f"{path}: empty RGB shard")

            expected = None if expected_sha256 is None else expected_sha256[session_id]
            if expected is not None and digest != expected:
                raise ValueError(f"{path}: SHA-256 mismatch for {session_id}")
            observed.add(session_id)
            result.append(
                RGBShard(
                    session_id=session_id,
                    path=path.resolve(),
                    sha256=digest,
                    frames=np.asarray(frames),
                    engine_frame_idx=np.asarray(engine),
                    input_active=np.asarray(active),
                )
            )
    if observed != allowed:
        raise ValueError(
            "loaded session membership differs from explicit allowlist: "
            f"missing={sorted(allowed - observed)}, extra={sorted(observed - allowed)}"
        )
    return result


def active_contiguous_runs(shard: RGBShard) -> list[tuple[int, int]]:
    """Half-open ranges that are both engine-consecutive and input-active."""

    engine = shard.engine_frame_idx
    active = shard.input_active.astype(bool, copy=False)
    if not len(engine):
        return []
    boundary = np.ones(len(engine), dtype=bool)
    boundary[1:] = (np.diff(engine) != 1) | (~active[:-1]) | (~active[1:])
    starts = np.flatnonzero(boundary & active)
    if not len(starts):
        return []
    # A run ends before the next engine/active boundary, or at array end.
    all_boundaries = np.flatnonzero(boundary)
    ends_by_start = {
        int(start): int(all_boundaries[index + 1])
        if index + 1 < len(all_boundaries)
        else len(engine)
        for index, start in enumerate(all_boundaries)
    }
    return [(int(start), ends_by_start[int(start)]) for start in starts]


def _luma_endpoint_difference(
    frames: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    chunk_size: int = 512,
) -> np.ndarray:
    """Mean absolute Rec.601-ish luma difference without action access."""

    if starts.shape != ends.shape:
        raise ValueError("luma endpoint indices must have identical shapes")
    score = np.empty(len(starts), dtype=np.float32)
    for offset in range(0, len(starts), chunk_size):
        sl = slice(offset, min(offset + chunk_size, len(starts)))
        first = frames[starts[sl]].astype(np.int32)
        second = frames[ends[sl]].astype(np.int32)
        first_luma = (
            54 * first[..., 0] + 183 * first[..., 1] + 19 * first[..., 2] + 128
        ) >> 8
        second_luma = (
            54 * second[..., 0] + 183 * second[..., 1] + 19 * second[..., 2] + 128
        ) >> 8
        score[sl] = np.abs(second_luma - first_luma).mean(axis=(1, 2))
    return score


def build_candidate_index(
    shards: Sequence[RGBShard],
    *,
    arm: DynamicsArm,
    horizons: Sequence[int],
) -> CandidateIndex:
    """Enumerate every eligible ordered sample without crossing boundaries."""

    values = tuple(int(value) for value in horizons)
    if not values or any(value <= 0 for value in values):
        raise ValueError(
            "Arms B, C, and D require the same positive matched horizon schedule"
        )
    if len(values) != len(set(values)):
        raise ValueError("horizons must be unique")

    columns: dict[str, list[np.ndarray]] = defaultdict(list)
    score_parts: list[np.ndarray] = []
    for session_index, shard in enumerate(shards):
        for start, end in active_contiguous_runs(shard):
            # Every arm uses the same future-bounded anchor population and the
            # same (t,t+h) masked-pixel motion stratum.  Arm B alone replaces
            # the future target with a second augmented view of t; h remains a
            # matched nuisance token and sampling quota.
            for horizon in values:
                online_current = np.arange(
                    start,
                    end - horizon,
                    dtype=np.int64,
                )
                if not len(online_current):
                    continue
                future_current = online_current + horizon
                target_current = (
                    online_current if arm == "B" else future_current
                )
                online_previous = np.where(
                    online_current > start, online_current - 1, online_current
                ).astype(np.int64)
                target_previous = np.where(
                    target_current > start, target_current - 1, target_current
                ).astype(np.int64)
                score = _luma_endpoint_difference(
                    shard.frames, online_current, future_current
                )
                count = len(online_current)
                columns["session"].append(
                    np.full(count, session_index, dtype=np.int32)
                )
                columns["online_previous"].append(online_previous)
                columns["online_current"].append(online_current)
                columns["target_previous"].append(target_previous)
                columns["target_current"].append(target_current)
                columns["horizon"].append(
                    np.full(count, horizon, dtype=np.int16)
                )
                score_parts.append(score)
    if not score_parts:
        raise ValueError("no eligible active, contiguous temporal samples")
    motion_score = np.concatenate(score_parts)
    threshold = float(np.median(motion_score))
    stratum = (motion_score > threshold).astype(np.uint8)
    if not np.any(stratum == 0) or not np.any(stratum == 1):
        raise ValueError(
            "masked-pixel motion split lacks both static and change-heavy samples"
        )

    def combined(name: str, dtype: np.dtype[Any]) -> np.ndarray:
        return np.concatenate(columns[name]).astype(dtype, copy=False)

    index = CandidateIndex(
        session=combined("session", np.dtype(np.int32)),
        online_previous=combined("online_previous", np.dtype(np.int64)),
        online_current=combined("online_current", np.dtype(np.int64)),
        target_previous=combined("target_previous", np.dtype(np.int64)),
        target_current=combined("target_current", np.dtype(np.int64)),
        horizon=combined("horizon", np.dtype(np.int16)),
        motion_score=motion_score,
        motion_threshold=threshold,
        stratum=stratum,
    )
    for horizon in values:
        for stratum_value in (0, 1):
            if not np.any(
                (index.horizon == horizon) & (index.stratum == stratum_value)
            ):
                raise ValueError(
                    f"horizon {horizon} lacks stratum {stratum_value} candidates"
                )
    return index


class BalancedCandidateSampler:
    """Deterministic cycling sampler with exact horizon/stratum quotas."""

    def __init__(
        self,
        candidates: CandidateIndex,
        *,
        horizons: Sequence[int],
        batch_size: int,
        seed: int,
    ) -> None:
        self.horizons = tuple(int(value) for value in horizons)
        divisor = 2 * len(self.horizons)
        if batch_size < divisor or batch_size % divisor:
            raise ValueError(
                f"batch_size must be divisible by 2 * horizons ({divisor})"
            )
        self.batch_size = int(batch_size)
        self.quota = self.batch_size // divisor
        self.rng = np.random.default_rng(seed)
        self._buckets: dict[tuple[int, int], np.ndarray] = {}
        self._position: dict[tuple[int, int], int] = {}
        self.draw_counts = {
            f"h{horizon}:{'change' if stratum else 'static'}": 0
            for horizon in self.horizons
            for stratum in (0, 1)
        }
        for horizon in self.horizons:
            for stratum in (0, 1):
                key = (horizon, stratum)
                values = np.flatnonzero(
                    (candidates.horizon == horizon)
                    & (candidates.stratum == stratum)
                ).astype(np.int64)
                if not len(values):
                    raise ValueError(f"empty candidate bucket {key}")
                self._buckets[key] = self.rng.permutation(values)
                self._position[key] = 0

    def _take(self, key: tuple[int, int], count: int) -> np.ndarray:
        pieces: list[np.ndarray] = []
        remaining = count
        while remaining:
            bucket = self._buckets[key]
            position = self._position[key]
            available = len(bucket) - position
            take = min(available, remaining)
            pieces.append(bucket[position : position + take])
            position += take
            remaining -= take
            if position == len(bucket):
                bucket = self.rng.permutation(bucket)
                position = 0
                self._buckets[key] = bucket
            self._position[key] = position
        return np.concatenate(pieces)

    def next_batch(self) -> np.ndarray:
        parts: list[np.ndarray] = []
        for horizon in self.horizons:
            for stratum in (0, 1):
                key = (horizon, stratum)
                parts.append(self._take(key, self.quota))
                receipt_key = f"h{horizon}:{'change' if stratum else 'static'}"
                self.draw_counts[receipt_key] += self.quota
        return self.rng.permutation(np.concatenate(parts))


def _gather_frames(
    shards: Sequence[RGBShard],
    sessions: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    batch = np.empty((len(indices), *EXPECTED_FRAME_SHAPE), dtype=np.uint8)
    for session_index in np.unique(sessions):
        selection = np.flatnonzero(sessions == session_index)
        batch[selection] = shards[int(session_index)].frames[indices[selection]]
    return (
        torch.from_numpy(batch)
        .to(device=device, dtype=torch.float32)
        .permute(0, 3, 1, 2)
        .div_(255.0)
    )


def augment_ordered_tuple(
    frames: torch.Tensor,
    *,
    generator: torch.Generator,
    crop_scale_min: float = 0.8,
    brightness: float = 0.2,
    contrast: float = 0.2,
) -> torch.Tensor:
    """Apply one no-flip crop/color transform per sample, shared over time.

    ``frames`` has shape ``[B,T,3,H,W]``.  Online and target calls use
    independent generator draws, but all T frames in one ordered tuple receive
    identical parameters.  There is deliberately no horizontal or vertical
    flip branch.
    """

    if frames.ndim != 5 or frames.shape[2] != 3:
        raise ValueError("ordered augmentation input must be [B,T,3,H,W]")
    if not 0.0 < crop_scale_min <= 1.0:
        raise ValueError("crop_scale_min must lie in (0,1]")
    if brightness < 0.0 or contrast < 0.0:
        raise ValueError("color jitter magnitudes must be non-negative")
    batch, time, channels, height, width = frames.shape
    output = torch.empty_like(frames)
    random_values = torch.rand((batch, 5), generator=generator, device="cpu")
    for index in range(batch):
        scale = crop_scale_min + (1.0 - crop_scale_min) * float(
            random_values[index, 0]
        )
        crop_h = max(1, min(height, int(round(height * math.sqrt(scale)))))
        crop_w = max(1, min(width, int(round(width * math.sqrt(scale)))))
        top = int(float(random_values[index, 1]) * (height - crop_h + 1))
        left = int(float(random_values[index, 2]) * (width - crop_w + 1))
        value = frames[index, :, :, top : top + crop_h, left : left + crop_w]
        value = F.interpolate(
            value,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        brightness_factor = 1.0 + brightness * (
            2.0 * float(random_values[index, 3]) - 1.0
        )
        contrast_factor = 1.0 + contrast * (
            2.0 * float(random_values[index, 4]) - 1.0
        )
        mean = value.mean(dim=(-2, -1), keepdim=True)
        value = (value - mean) * contrast_factor + mean
        output[index] = (value * brightness_factor).clamp_(0.0, 1.0)
    return output


def _batch_inputs(
    shards: Sequence[RGBShard],
    candidates: CandidateIndex,
    selection: np.ndarray,
    *,
    arm: DynamicsArm,
    device: torch.device,
    online_generator: torch.Generator,
    target_generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    session = candidates.session[selection]
    online_current = _gather_frames(
        shards, session, candidates.online_current[selection], device=device
    )
    target_current = _gather_frames(
        shards, session, candidates.target_current[selection], device=device
    )
    result: dict[str, torch.Tensor]
    if arm == "D":
        online_previous = _gather_frames(
            shards, session, candidates.online_previous[selection], device=device
        )
        target_previous = _gather_frames(
            shards, session, candidates.target_previous[selection], device=device
        )
        online_pair = augment_ordered_tuple(
            torch.stack((online_previous, online_current), dim=1),
            generator=online_generator,
        )
        target_pair = augment_ordered_tuple(
            torch.stack((target_previous, target_current), dim=1),
            generator=target_generator,
        )
        result = {
            "online_previous": online_pair[:, 0],
            "online_current": online_pair[:, 1],
            "target_previous": target_pair[:, 0],
            "target_current": target_pair[:, 1],
        }
    else:
        result = {
            "online_current": augment_ordered_tuple(
                online_current[:, None], generator=online_generator
            )[:, 0],
            "target_current": augment_ordered_tuple(
                target_current[:, None], generator=target_generator
            )[:, 0],
        }
    result["horizon"] = torch.from_numpy(
        candidates.horizon[selection].astype(np.int64)
    ).to(device)
    return result


def linear_ema_momentum(
    step_index: int,
    total_steps: int,
    *,
    start: float = 0.998,
    end: float = 1.0,
) -> float:
    if total_steps < 1 or not 0 <= step_index < total_steps:
        raise ValueError("EMA step must lie inside a positive fixed step budget")
    if total_steps == 1:
        return float(start)
    return float(start + (end - start) * step_index / (total_steps - 1))


def _diagnostic_summary(value: CollapseDiagnostics) -> dict[str, float]:
    per_dimension_std = value.per_dimension_std.detach().float().cpu()
    return {
        "per_dimension_std_min": float(per_dimension_std.min()),
        "per_dimension_std_median": float(per_dimension_std.median()),
        "per_dimension_std_mean": float(per_dimension_std.mean()),
        "per_dimension_std_max": float(per_dimension_std.max()),
        "covariance_effective_rank": float(
            value.covariance_effective_rank.detach().cpu()
        ),
        "mean_cosine_similarity": float(
            value.mean_cosine_similarity.detach().cpu()
        ),
    }


def _candidate_support(
    shards: Sequence[RGBShard], candidates: CandidateIndex
) -> dict[str, Any]:
    by_horizon_stratum: dict[str, int] = {}
    by_session: dict[str, int] = {}
    for horizon in sorted(set(map(int, candidates.horizon.tolist()))):
        for stratum, label in ((0, "static"), (1, "change")):
            by_horizon_stratum[f"h{horizon}:{label}"] = int(
                np.sum(
                    (candidates.horizon == horizon)
                    & (candidates.stratum == stratum)
                )
            )
    for index, shard in enumerate(shards):
        by_session[shard.session_id] = int(np.sum(candidates.session == index))
    return {
        "eligible_samples": len(candidates),
        "motion_definition": (
            "mean absolute masked-pixel luma endpoint difference (t,t+h) for "
            "every arm; Arm B uses h only for matched sampling/conditioning"
        ),
        "motion_threshold_training_median": candidates.motion_threshold,
        "motion_score_min": float(candidates.motion_score.min()),
        "motion_score_max": float(candidates.motion_score.max()),
        "eligible_by_horizon_stratum": by_horizon_stratum,
        "eligible_by_session": by_session,
        "active_frames": {
            shard.session_id: int(shard.input_active.sum()) for shard in shards
        },
        "contiguous_active_runs": {
            shard.session_id: len(active_contiguous_runs(shard)) for shard in shards
        },
    }


def save_final_checkpoint_atomic(
    output_dir: Path,
    *,
    model: EMADynamicsPretrainer,
    optimizer: torch.optim.Optimizer,
    checkpoint_metadata: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Path]:
    """Publish exactly one final checkpoint and one receipt by directory rename."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        checkpoint_path = temporary / "final.pt"
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA,
                **dict(checkpoint_metadata),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        checkpoint_hash = sha256_file(checkpoint_path)
        final_receipt = dict(receipt)
        final_receipt["artifacts"] = {
            "checkpoint": "final.pt",
            "checkpoint_sha256": checkpoint_hash,
            "inventory": ["final.pt", "run_receipt.json"],
        }
        receipt_path = temporary / "run_receipt.json"
        receipt_path.write_text(
            json.dumps(final_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "checkpoint": output / "final.pt",
        "receipt": output / "run_receipt.json",
    }


def load_final_checkpoint(
    path: Path,
    model: EMADynamicsPretrainer,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Strictly reload a final checkpoint into an already-constructed model."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported dynamics checkpoint schema")
    if payload.get("arm") != model.arm:
        raise ValueError("checkpoint arm does not match model")
    if tuple(payload.get("horizons", ())) != model.predictor.horizons:
        raise ValueError("checkpoint horizons do not match model")
    model.load_state_dict(payload["model_state"], strict=True)
    return payload


def train_fixed_steps(
    *,
    shards: Sequence[RGBShard],
    candidates: CandidateIndex,
    model: EMADynamicsPretrainer,
    arm: DynamicsArm,
    horizons: Sequence[int],
    output_dir: Path,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    diagnostic_interval: int = 10,
    initialization_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the exact fixed-step smoke and atomically publish final evidence."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if diagnostic_interval < 1:
        raise ValueError("diagnostic_interval must be positive")
    values = tuple(int(value) for value in horizons)
    if model.arm != arm or model.predictor.horizons != values:
        raise ValueError("model identity does not match requested arm/horizons")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    sampler = BalancedCandidateSampler(
        candidates, horizons=values, batch_size=batch_size, seed=seed
    )
    online_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    target_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    loss_values: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    last_momentum = 0.998
    for step_index in range(steps):
        selection = sampler.next_batch()
        inputs = _batch_inputs(
            shards,
            candidates,
            selection,
            arm=arm,
            device=device,
            online_generator=online_generator,
            target_generator=target_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(**inputs)
        if not bool(torch.isfinite(output.loss)):
            raise FloatingPointError(f"non-finite loss at step {step_index + 1}")
        output.loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(
                    f"non-finite gradient at step {step_index + 1}"
                )
        optimizer.step()
        last_momentum = linear_ema_momentum(step_index, steps)
        model.update_target(last_momentum)
        loss_values.append(float(output.loss.detach().cpu()))
        if (
            step_index == 0
            or step_index + 1 == steps
            or (step_index + 1) % diagnostic_interval == 0
        ):
            diagnostics.append(
                {
                    "step": step_index + 1,
                    "online": _diagnostic_summary(
                        collapse_diagnostics(output.online.detach())
                    ),
                    "prediction": _diagnostic_summary(
                        collapse_diagnostics(output.prediction.detach())
                    ),
                    "target": _diagnostic_summary(
                        collapse_diagnostics(output.target.detach())
                    ),
                }
            )

    loss_array = np.asarray(loss_values, dtype=np.float64)
    support = _candidate_support(shards, candidates)
    shard_records = [
        {
            "session_id": shard.session_id,
            "path": str(shard.path),
            "sha256": shard.sha256,
            "frames": len(shard.frames),
        }
        for shard in shards
    ]
    config = {
        "arm": arm,
        "horizons": list(values),
        "steps": steps,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "ema": {"start": 0.998, "end": 1.0, "schedule": "linear"},
        "seed": seed,
        "augmentation": {
            "crop_scale_min": 0.8,
            "brightness": 0.2,
            "contrast": 0.2,
            "spatial_or_vertical_flips": False,
            "tuple_consistent": True,
            "online_target_views_independent": True,
        },
        "labels_consumed": False,
        "initialization": dict(initialization_receipt or {"kind": "injected-test-model"}),
    }
    checkpoint_metadata = {
        "arm": arm,
        "horizons": list(values),
        "completed_steps": steps,
        "ema_momentum_final_update": last_momentum,
        "config": config,
        "config_sha256": _canonical_json_sha256(config),
        "support": support,
        "shards": shard_records,
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "completed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_contract": {
            "explicit_paths_only": True,
            "session_allowlist_exact": True,
            "no_session_discovery": True,
            "labels_consumed": False,
            "shards": shard_records,
        },
        "config": config,
        "config_sha256": _canonical_json_sha256(config),
        "support": support,
        "sampling": {
            "quota_per_horizon_stratum_per_step": sampler.quota,
            "draw_counts": sampler.draw_counts,
            "equal_horizon_quota": True,
            "equal_static_change_quota": True,
        },
        "loss": {
            "name": "normalized_l1",
            "steps": len(loss_values),
            "first": loss_values[0],
            "final": loss_values[-1],
            "minimum": float(loss_array.min()),
            "maximum": float(loss_array.max()),
            "mean": float(loss_array.mean()),
            "trace_float64_sha256": hashlib.sha256(loss_array.tobytes()).hexdigest(),
            "trace": loss_values,
        },
        "collapse_diagnostics": diagnostics,
    }
    artifacts = save_final_checkpoint_atomic(
        output_dir,
        model=model,
        optimizer=optimizer,
        checkpoint_metadata=checkpoint_metadata,
        receipt=receipt,
    )
    return {
        "checkpoint": str(artifacts["checkpoint"]),
        "receipt": str(artifacts["receipt"]),
        "loss": receipt["loss"],
        "support": support,
    }


def _parse_expected_hashes(values: Sequence[str]) -> dict[str, str] | None:
    if not values:
        return None
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("expected hash entries must be SESSION_ID=SHA256")
        session_id, digest = value.split("=", 1)
        if not session_id or session_id in result:
            raise ValueError("invalid or duplicate expected hash session ID")
        result[session_id] = digest
    return result


def _parse_horizons(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("horizon list is empty")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("B", "C", "D"))
    parser.add_argument("--horizons", required=True, help="comma-separated native frames")
    parser.add_argument("--shard", action="append", type=Path, required=True)
    parser.add_argument("--allowed-session-id", action="append", required=True)
    parser.add_argument(
        "--expected-sha256",
        action="append",
        default=[],
        metavar="SESSION_ID=SHA256",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostic-interval", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arm: DynamicsArm = args.arm
    horizons = _parse_horizons(args.horizons)
    expected = _parse_expected_hashes(args.expected_sha256)
    shards = load_explicit_rgb_shards(
        args.shard,
        allowed_session_ids=args.allowed_session_id,
        expected_sha256=expected,
    )
    candidates = build_candidate_index(shards, arm=arm, horizons=horizons)
    initialization = validate_imagenet_initialization()
    model = EMADynamicsPretrainer(
        arm, horizons=horizons, weights=RESNET18_WEIGHTS
    )
    # Guard against replacement between the initial validation and model
    # construction.  This still occurs before optimizer step one.
    if sha256_file(Path(initialization["cached_path"])) != RESNET18_WEIGHTS_SHA256:
        raise RuntimeError("ResNet-18 ImageNet cache changed during model creation")
    result = train_fixed_steps(
        shards=shards,
        candidates=candidates,
        model=model,
        arm=arm,
        horizons=horizons,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=torch.device(args.device),
        diagnostic_interval=args.diagnostic_interval,
        initialization_receipt=initialization,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
