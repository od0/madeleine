#!/usr/bin/env python3
"""Production streaming trainer for matched Celeste dynamics-pretraining arms.

This module consumes only the label-free cache produced by the dynamics RGB
cache builder: a C-contiguous ``rgb.npy`` memmap, an arm-independent
``index.npz``, and a content-bound completion manifest.  Cache membership and
the embargo proof are validated before the RGB memmap is opened.

The schedule is counter based.  Given the same cache, horizons, batch size,
and seed, Arms B/C/D consume the same tuple IDs in the same order, including
after resume.  Periodic checkpoints are recovery artifacts only; ``final.pt``
is the sole selectable SSL checkpoint.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import random
import sys
import tempfile
import threading
import time
from typing import Any, Literal

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Deterministic CUDA matrix multiplication requires this before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights

from badeline.dynamics_pretraining import (
    DynamicsArm,
    EMADynamicsPretrainer,
)
from experiments.pretrain_dynamics import (
    validate_imagenet_initialization,
)


CACHE_SCHEMA = "madeleine.dynamics-rgb-cache.v1"
CHECKPOINT_SCHEMA = "madeleine.dynamics-streaming-checkpoint.v1"
RUN_SCHEMA = "madeleine.dynamics-streaming-run.v1"
COMPLETION_SCHEMA = "madeleine.dynamics-streaming-complete.v1"
FAILURE_SCHEMA = "madeleine.dynamics-streaming-failure.v1"
SCHEDULE_VERSION = "counter-fair-source-affine-cycles-v2"
EXPECTED_RGB_SHAPE = (128, 128, 3)
INDEX_FIELDS = (
    "tuple_id",
    "window_id",
    "source_id",
    "session_index",
    "run_id",
    "anchor_engine_frame",
    "online_previous",
    "online_current",
    "target_previous",
    "target_current",
    "horizon",
    "motion_score",
    "stratum",
    "session_ids",
    "source_names",
)
INDEX_DTYPES = {
    "tuple_id": np.dtype(np.uint64),
    "window_id": np.dtype(np.int64),
    "source_id": np.dtype(np.uint8),
    "session_index": np.dtype(np.int32),
    "run_id": np.dtype(np.int32),
    "anchor_engine_frame": np.dtype(np.int64),
    "online_previous": np.dtype(np.int64),
    "online_current": np.dtype(np.int64),
    "target_previous": np.dtype(np.int64),
    "target_current": np.dtype(np.int64),
    "horizon": np.dtype(np.int16),
    "motion_score": np.dtype(np.float32),
    "stratum": np.dtype(np.uint8),
}
FORBIDDEN_EXACT_IDS = frozenset(
    {
        "rec_20260727_220000_test",
        "rec_20260724_171305_5min",
        "rec_20260725_025853",
        "rec_20260725_160450_b1",
    }
)
FORBIDDEN_VIDEO_ID = "y4nQHqYSObI"
MASK64 = (1 << 64) - 1


class CacheContractError(ValueError):
    """The immutable cache does not satisfy the label-free data contract."""


class RepresentationCollapse(RuntimeError):
    """Three consecutive fixed-panel collapse checks failed."""


class ThroughputBudgetExceeded(RuntimeError):
    """The measured 100-step projection exceeds the frozen run budget."""


@dataclass(frozen=True)
class CacheBundle:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    rgb_path: Path
    rgb_sha256: str
    index_path: Path
    index_sha256: str
    rgb: np.ndarray
    index: dict[str, np.ndarray]
    source_probabilities: dict[int, float]


@dataclass(frozen=True)
class TrainConfig:
    arm: DynamicsArm
    horizons: tuple[int, ...]
    max_steps: int
    global_batch_size: int
    microbatch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    schedule_seed: int
    checkpoint_interval: int
    diagnostic_interval: int
    panel_size: int
    prefetch_depth: int
    throughput_steps: int
    max_projected_seconds: float
    collapse_relative_floor: float
    collapse_effective_rank_min: float
    collapse_mean_cosine_max: float
    collapse_nn_unique_fraction_min: float
    collapse_consecutive_failures: int
    max_cuda_memory_gib: float
    device: str
    study_id: str
    run_id: str
    implementation_commit: str


@dataclass(frozen=True)
class HostBatch:
    step: int
    rows: torch.Tensor
    tuple_id: torch.Tensor
    horizon: torch.Tensor
    stratum: torch.Tensor
    online_previous: torch.Tensor
    online_current: torch.Tensor
    target_previous: torch.Tensor
    target_current: torch.Tensor


@dataclass(frozen=True)
class CollapseGateState:
    consecutive_failures: int
    initial: dict[str, float]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_native(value: Any) -> Any:
    """Normalize tuples and scalar containers to their JSON representation."""

    return json.loads(
        json.dumps(
            value, sort_keys=True, ensure_ascii=True, allow_nan=False
        )
    )


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CacheContractError(f"{description} is not readable JSON") from error
    if not isinstance(value, dict):
        raise CacheContractError(f"{description} must be a JSON object")
    return value


def _read_json_value(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not readable JSON") from error


def _is_forbidden_identity(value: str) -> bool:
    folded = value.casefold()
    return (
        value in FORBIDDEN_EXACT_IDS
        or FORBIDDEN_VIDEO_ID.casefold() in folded
        or "untouched" in folded
        or folded.endswith("_b1")
    )


def reject_forbidden_path_before_access(path: Path) -> None:
    for component in Path(path).parts:
        if _is_forbidden_identity(component):
            raise CacheContractError(
                f"forbidden evaluation identity appears in path: {component}"
            )


def _regular_unsymlinked(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise CacheContractError(f"{description} must be a regular non-symlink file")


def _artifact_record(
    manifest: Mapping[str, Any], name: Literal["rgb", "index"]
) -> Mapping[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CacheContractError("cache manifest lacks artifacts")
    record = artifacts.get(name)
    if not isinstance(record, Mapping):
        raise CacheContractError(f"cache manifest lacks {name} artifact")
    return record


def _validate_exclusion_and_labels(manifest: Mapping[str, Any]) -> None:
    labels = manifest.get("labels")
    if (
        not isinstance(labels, Mapping)
        or labels.get("loaded") is not False
        or labels.get("arrays_accessed") != []
    ):
        raise CacheContractError("cache is not provably label-free")
    proof = manifest.get("exclusion_proof")
    if not isinstance(proof, Mapping):
        raise CacheContractError("cache lacks exclusion proof")
    required = (
        "validated_before_source_access",
        "whole_y4n_absent",
        "val_a_absent",
        "val_b_absent",
        "b1_absent",
        "sealed_untouched_absent",
    )
    failed = [name for name in required if proof.get(name) is not True]
    if failed:
        raise CacheContractError(f"cache exclusion proof failed: {failed}")
    forbidden_ids = proof.get("forbidden_ids")
    if not isinstance(forbidden_ids, list) or not all(
        isinstance(value, str) for value in forbidden_ids
    ):
        raise CacheContractError("cache forbidden-ID receipt is malformed")
    if not FORBIDDEN_EXACT_IDS.issubset(set(forbidden_ids)):
        raise CacheContractError("cache forbidden-ID receipt is incomplete")
    if not any(FORBIDDEN_VIDEO_ID in value for value in forbidden_ids):
        raise CacheContractError("cache forbidden-ID receipt omits y4n")


def _validate_manifest_before_cache_access(
    manifest: Mapping[str, Any], manifest_path: Path
) -> tuple[Path, Path]:
    if manifest.get("schema_version") != CACHE_SCHEMA:
        raise CacheContractError("unsupported dynamics RGB cache schema")
    if manifest.get("status") != "complete":
        raise CacheContractError("dynamics RGB cache is not complete")
    _validate_exclusion_and_labels(manifest)
    horizons = manifest.get("horizons")
    if (
        not isinstance(horizons, list)
        or not horizons
        or any(not isinstance(value, int) or value <= 0 for value in horizons)
        or len(horizons) != len(set(horizons))
    ):
        raise CacheContractError("cache horizon contract is malformed")
    expected_span = max(horizons) + 2
    if manifest.get("window_span_frames") != expected_span:
        raise CacheContractError("cache window span does not match horizons")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, Mapping) or not all(
        isinstance(inventory.get(name), str)
        for name in ("path", "sha256", "schema_version")
    ):
        raise CacheContractError("cache inventory receipt is incomplete")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise CacheContractError("cache source inventory is malformed")
    for source in sources:
        if not isinstance(source, Mapping):
            raise CacheContractError("cache source row is malformed")
        for key in ("name", "source", "source_id", "session_id", "video_id"):
            value = source.get(key)
            if isinstance(value, str) and _is_forbidden_identity(value):
                raise CacheContractError(
                    f"forbidden identity appears in source inventory: {value}"
                )

    root = manifest_path.parent.resolve()
    rgb_record = _artifact_record(manifest, "rgb")
    index_record = _artifact_record(manifest, "index")
    if rgb_record.get("path") != "rgb.npy" or index_record.get("path") != "index.npz":
        raise CacheContractError("cache artifact paths must be canonical basenames")
    rgb_path = root / "rgb.npy"
    index_path = root / "index.npz"
    for path in (rgb_path, index_path):
        reject_forbidden_path_before_access(path)
        if path.parent.resolve() != root:
            raise CacheContractError("cache artifact escapes cache root")
    return rgb_path, index_path


def load_cache(manifest_path: Path, *, expected_horizons: Sequence[int]) -> CacheBundle:
    """Validate immutable metadata and bytes, then open the RGB memmap."""

    manifest_path = Path(manifest_path)
    reject_forbidden_path_before_access(manifest_path)
    _regular_unsymlinked(manifest_path, "cache manifest")
    manifest = _read_json(manifest_path, "cache manifest")
    rgb_path, index_path = _validate_manifest_before_cache_access(
        manifest, manifest_path
    )
    _regular_unsymlinked(rgb_path, "RGB cache")
    _regular_unsymlinked(index_path, "tuple index")
    values = tuple(int(value) for value in expected_horizons)
    if list(values) != manifest.get("horizons"):
        raise CacheContractError("requested horizons differ from cache")

    rgb_record = _artifact_record(manifest, "rgb")
    index_record = _artifact_record(manifest, "index")
    for record, path, name in (
        (rgb_record, rgb_path, "rgb"),
        (index_record, index_path, "index"),
    ):
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise CacheContractError(f"{name} SHA-256 receipt is malformed")
        if record.get("bytes") != path.stat().st_size:
            raise CacheContractError(f"{name} byte size differs from manifest")
        if sha256_file(path) != expected_hash:
            raise CacheContractError(f"{name} SHA-256 differs from manifest")

    # The index contains metadata only.  No action, truth, key, or label array
    # is allowed to exist, even if a future caller promises not to read it.
    with np.load(index_path, allow_pickle=False) as archive:
        observed = tuple(archive.files)
        if set(observed) != set(INDEX_FIELDS):
            raise CacheContractError(
                "tuple index field inventory changed: "
                f"missing={sorted(set(INDEX_FIELDS)-set(observed))}, "
                f"extra={sorted(set(observed)-set(INDEX_FIELDS))}"
            )
        forbidden_field_tokens = ("key", "action", "button", "truth", "label")
        if any(
            token in name.casefold()
            for name in observed
            for token in forbidden_field_tokens
        ):
            raise CacheContractError("tuple index contains a supervision field")
        index = {name: np.asarray(archive[name]) for name in observed}

    fields_receipt = index_record.get("fields")
    if isinstance(fields_receipt, Mapping):
        receipt_names = set(fields_receipt)
    elif isinstance(fields_receipt, list) and all(
        isinstance(value, str) for value in fields_receipt
    ):
        receipt_names = set(fields_receipt)
    elif isinstance(fields_receipt, list) and all(
        isinstance(value, Mapping) and isinstance(value.get("name"), str)
        for value in fields_receipt
    ):
        receipt_names = {str(value["name"]) for value in fields_receipt}
    else:
        receipt_names = set()
    if receipt_names != set(INDEX_FIELDS):
        raise CacheContractError("index field receipt differs from index")
    row_count = len(index["tuple_id"])
    if index_record.get("rows") != row_count or manifest.get("tuples") != row_count:
        raise CacheContractError("tuple count differs from manifest")
    for name, dtype in INDEX_DTYPES.items():
        value = index[name]
        if value.dtype != dtype or value.shape != (row_count,):
            raise CacheContractError(
                f"index {name} must be {dtype} [{row_count}]"
            )
    for name in ("session_ids", "source_names"):
        value = index[name]
        if value.ndim != 1 or value.dtype.kind != "U":
            raise CacheContractError(f"index {name} must be a Unicode vector")
        for identity in value.tolist():
            if _is_forbidden_identity(str(identity)):
                raise CacheContractError(
                    f"forbidden identity appears in tuple index: {identity}"
                )
    if len(np.unique(index["tuple_id"])) != row_count:
        raise CacheContractError("tuple IDs are not unique")
    if not np.isfinite(index["motion_score"]).all() or np.any(
        index["motion_score"] < 0
    ):
        raise CacheContractError("motion scores must be finite and nonnegative")
    if set(np.unique(index["stratum"]).tolist()) != {0, 1}:
        raise CacheContractError("tuple index must contain both motion strata")
    if tuple(sorted(np.unique(index["horizon"]).tolist())) != tuple(sorted(values)):
        raise CacheContractError("tuple index horizons differ from contract")
    if np.any(index["session_index"] < 0) or np.any(
        index["session_index"] >= len(index["session_ids"])
    ):
        raise CacheContractError("session index is outside lookup table")
    if np.any(index["source_id"] >= len(index["source_names"])):
        raise CacheContractError("source index is outside lookup table")

    source_rows = manifest["sources"]
    source_probabilities: dict[int, float] = {}
    seen_source_names: set[str] = set()
    for row in source_rows:
        source_id = row.get("source_id")
        name = row.get("name")
        probability = row.get("eligible_probability")
        if (
            not isinstance(source_id, int)
            or not 0 <= source_id < len(index["source_names"])
            or not isinstance(name, str)
            or str(index["source_names"][source_id]) != name
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or float(probability) <= 0
            or source_id in source_probabilities
            or name in seen_source_names
        ):
            raise CacheContractError("cache source-probability receipt is malformed")
        source_probabilities[source_id] = float(probability)
        seen_source_names.add(name)
    if set(source_probabilities) != set(range(len(index["source_names"]))) or not math.isclose(
        sum(source_probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise CacheContractError("cache source probabilities are incomplete")
    if not np.array_equal(index["online_current"], index["online_previous"] + 1):
        raise CacheContractError("online pair geometry changed")
    if not np.array_equal(index["target_current"], index["target_previous"] + 1):
        raise CacheContractError("target pair geometry changed")
    horizon64 = index["horizon"].astype(np.int64)
    if not np.array_equal(
        index["target_previous"] - index["online_previous"], horizon64
    ) or not np.array_equal(
        index["target_current"] - index["online_current"], horizon64
    ):
        raise CacheContractError("target offsets differ from native horizons")

    rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    expected_shape = rgb_record.get("shape")
    if (
        rgb.dtype != np.uint8
        or rgb.ndim != 4
        or tuple(rgb.shape[1:]) != EXPECTED_RGB_SHAPE
        or list(rgb.shape) != expected_shape
        or rgb_record.get("dtype") != "uint8"
        or rgb_record.get("c_order") is not True
        or not rgb.flags.c_contiguous
    ):
        raise CacheContractError("RGB memmap shape/dtype/order differs from manifest")
    if row_count and (
        min(int(index[name].min()) for name in (
            "online_previous", "online_current", "target_previous", "target_current"
        )) < 0
        or max(int(index[name].max()) for name in (
            "online_previous", "online_current", "target_previous", "target_current"
        )) >= len(rgb)
    ):
        raise CacheContractError("tuple frame index is outside RGB memmap")

    return CacheBundle(
        root=manifest_path.parent.resolve(),
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        rgb_path=rgb_path.resolve(),
        rgb_sha256=str(rgb_record["sha256"]),
        index_path=index_path.resolve(),
        index_sha256=str(index_record["sha256"]),
        rgb=rgb,
        index=index,
        source_probabilities=source_probabilities,
    )


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def _coprime_stride(seed: int, size: int) -> int:
    if size == 1:
        return 0
    candidate = (splitmix64(seed) % size) | 1
    while math.gcd(candidate, size) != 1:
        candidate = (candidate + 2) % size
        if candidate == 0:
            candidate = 1
    return int(candidate)


def cumulative_weighted_source_plan(
    probabilities: Mapping[int, float],
    *,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, dict[int, float]]:
    """Return a deterministic low-discrepancy categorical sequence.

    At draw ``n`` the source with the largest deficit relative to its desired
    cumulative allocation is selected.  Constant seed-derived tie ranks make
    the sequence reproducible without consuming mutable RNG state.  The
    resulting maximum absolute cumulative error is recorded and must remain
    at most one draw for every source.
    """

    if draws < 0 or not probabilities:
        raise ValueError("weighted source plan requires nonnegative draws")
    source_ids = tuple(sorted(int(value) for value in probabilities))
    weights = [float(probabilities[value]) for value in source_ids]
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("source probabilities must be finite and positive")
    total = sum(weights)
    weights = [value / total for value in weights]
    tie_order = sorted(
        range(len(source_ids)),
        key=lambda position: splitmix64(seed ^ source_ids[position]),
    )
    tie_rank = {position: rank for rank, position in enumerate(tie_order)}
    counts = [0] * len(source_ids)
    maximum_error = [0.0] * len(source_ids)
    plan = np.empty(draws, dtype=np.uint8)
    for draw in range(draws):
        after = draw + 1
        deficits = [after * weights[i] - counts[i] for i in range(len(source_ids))]
        chosen = max(
            range(len(source_ids)),
            key=lambda i: (deficits[i], -tie_rank[i]),
        )
        counts[chosen] += 1
        plan[draw] = source_ids[chosen]
        for i in range(len(source_ids)):
            maximum_error[i] = max(
                maximum_error[i], abs(counts[i] - after * weights[i])
            )
    errors = {
        source_id: float(maximum_error[position])
        for position, source_id in enumerate(source_ids)
    }
    if any(value > 1.0 + 1e-9 for value in errors.values()):
        raise RuntimeError("weighted source schedule exceeded one-draw discrepancy")
    return plan, errors


class CounterSchedule:
    """Arm-independent schedule with exact h/stratum and fair source draws."""

    def __init__(
        self,
        index: Mapping[str, np.ndarray],
        *,
        horizons: Sequence[int],
        batch_size: int,
        seed: int,
        max_steps: int,
        source_probabilities: Mapping[int, float],
    ) -> None:
        self.horizons = tuple(int(value) for value in horizons)
        divisor = 2 * len(self.horizons)
        if batch_size < divisor or batch_size % divisor:
            raise ValueError(f"batch size must be divisible by {divisor}")
        self.batch_size = int(batch_size)
        self.quota = self.batch_size // divisor
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("schedule endpoint must be positive")
        self.source_probabilities = {
            int(key): float(value) for key, value in source_probabilities.items()
        }
        observed_sources = set(map(int, np.unique(index["source_id"])))
        if set(self.source_probabilities) != observed_sources:
            raise ValueError("schedule source probabilities differ from index")
        self.buckets: dict[tuple[int, int, int], np.ndarray] = {}
        self.parameters: dict[tuple[int, int, int], tuple[int, int]] = {}
        self.cell_row_plans: dict[tuple[int, int], np.ndarray] = {}
        self.cell_source_plans: dict[tuple[int, int], np.ndarray] = {}
        self.cell_maximum_errors: dict[tuple[int, int], dict[int, float]] = {}
        for h_index, horizon in enumerate(self.horizons):
            for stratum in (0, 1):
                cell = (horizon, stratum)
                cell_seed = splitmix64(
                    self.seed ^ (h_index << 32) ^ (stratum << 48) ^ 0xCE115EED
                )
                source_plan, errors = cumulative_weighted_source_plan(
                    self.source_probabilities,
                    draws=self.max_steps * self.quota,
                    seed=cell_seed,
                )
                row_plan = np.empty(len(source_plan), dtype=np.int64)
                for source_id in sorted(self.source_probabilities):
                    key = (source_id, horizon, stratum)
                    rows = np.flatnonzero(
                        (index["source_id"] == source_id)
                        & (index["horizon"] == horizon)
                        & (index["stratum"] == stratum)
                    ).astype(np.int64)
                    if not len(rows):
                        raise ValueError(f"empty schedule bucket {key}")
                    bucket_seed = splitmix64(cell_seed ^ (source_id << 16))
                    offset = int(splitmix64(bucket_seed ^ 0xA11CE) % len(rows))
                    stride = _coprime_stride(bucket_seed ^ 0x5EED, len(rows))
                    self.buckets[key] = rows
                    self.parameters[key] = (offset, stride)
                    positions = np.flatnonzero(source_plan == source_id)
                    counters = np.arange(len(positions), dtype=np.int64)
                    row_plan[positions] = rows[
                        (offset + counters * stride) % len(rows)
                    ]
                self.cell_source_plans[cell] = source_plan
                self.cell_row_plans[cell] = row_plan
                self.cell_maximum_errors[cell] = errors

    def rows_for_step(self, step: int) -> np.ndarray:
        if step < 0:
            raise ValueError("schedule step must be nonnegative")
        if step >= self.max_steps:
            raise ValueError("schedule step exceeds fixed endpoint")
        selected: list[np.ndarray] = []
        draw_start = step * self.quota
        for horizon in self.horizons:
            for stratum in (0, 1):
                selected.append(
                    self.cell_row_plans[(horizon, stratum)][
                        draw_start : draw_start + self.quota
                    ]
                )
        result = np.concatenate(selected)
        # Counter-derived stable permutation avoids bucket-ordered microbatches.
        keys = np.fromiter(
            (
                splitmix64(
                    self.seed
                    ^ 0xBADC0FFEE
                    ^ (step << 24)
                    ^ position
                )
                for position in range(len(result))
            ),
            dtype=np.uint64,
            count=len(result),
        )
        return result[np.argsort(keys, kind="stable")]

    def receipt(self, tuple_ids: np.ndarray) -> dict[str, Any]:
        buckets: dict[str, Any] = {}
        for key in sorted(self.buckets):
            offset, stride = self.parameters[key]
            buckets[
                f"source{key[0]}:h{key[1]}:{'change' if key[2] else 'static'}"
            ] = {
                "rows": len(self.buckets[key]),
                "offset": offset,
                "stride": stride,
            }
        cells: dict[str, Any] = {}
        for key in sorted(self.cell_row_plans):
            row_plan = self.cell_row_plans[key]
            source_plan = self.cell_source_plans[key]
            cells[f"h{key[0]}:{'change' if key[1] else 'static'}"] = {
                "draws": len(row_plan),
                "row_plan_sha256": array_sha256(row_plan),
                "tuple_id_plan_sha256": array_sha256(tuple_ids[row_plan]),
                "source_plan_sha256": array_sha256(source_plan),
                "maximum_cumulative_allocation_error_by_source": {
                    str(source): error
                    for source, error in sorted(self.cell_maximum_errors[key].items())
                },
            }
        preview_rows = np.concatenate(
            [self.rows_for_step(step) for step in range(min(10, self.max_steps))]
        )
        core = {
            "version": SCHEDULE_VERSION,
            "seed": self.seed,
            "horizons": list(self.horizons),
            "max_steps": self.max_steps,
            "global_batch_size": self.batch_size,
            "quota_per_horizon_stratum": self.quota,
            "source_probabilities": {
                str(key): value for key, value in sorted(self.source_probabilities.items())
            },
            "buckets": buckets,
            "cells": cells,
            "first_ten_steps_tuple_id_sha256": array_sha256(tuple_ids[preview_rows]),
        }
        return {**core, "canonical_plan_sha256": canonical_json_sha256(core)}

    def panel_rows(self, total: int) -> np.ndarray:
        """Build a unique fixed panel balanced across h/stratum cells."""

        cells = [
            (horizon, stratum)
            for horizon in self.horizons
            for stratum in (0, 1)
        ]
        if total < 2 * len(cells):
            raise ValueError("collapse panel is too small")
        base, remainder = divmod(total, len(cells))
        selected: list[np.ndarray] = []
        for cell_position, (horizon, stratum) in enumerate(cells):
            count = base + int(cell_position < remainder)
            source_plan, _ = cumulative_weighted_source_plan(
                self.source_probabilities,
                draws=count,
                seed=splitmix64(
                    self.seed
                    ^ 0x9A4E1
                    ^ (cell_position << 32)
                ),
            )
            cell_rows = np.empty(count, dtype=np.int64)
            for source_id in sorted(self.source_probabilities):
                key = (source_id, horizon, stratum)
                bucket = self.buckets[key]
                positions = np.flatnonzero(source_plan == source_id)
                if len(positions) > len(bucket):
                    raise ValueError(
                        f"collapse panel requires duplicate rows from bucket {key}"
                    )
                offset, stride = self.parameters[key]
                cell_rows[positions] = bucket[
                    (offset + np.arange(len(positions), dtype=np.int64) * stride)
                    % len(bucket)
                ]
            selected.append(cell_rows)
        result = np.concatenate(selected)
        if len(result) != total or len(np.unique(result)) != total:
            raise ValueError("fixed collapse panel contains duplicate index rows")
        return result


def fixed_panel_rows(schedule: CounterSchedule, *, total: int) -> np.ndarray:
    return schedule.panel_rows(total)


def _pinned_tensor(array: np.ndarray, *, pin: bool) -> torch.Tensor:
    contiguous = np.ascontiguousarray(array)
    output = torch.empty(
        contiguous.shape, dtype=torch.from_numpy(contiguous).dtype, pin_memory=pin
    )
    output.copy_(torch.from_numpy(contiguous))
    return output


class BatchPrefetcher(Iterator[HostBatch]):
    """Bounded background memmap gather into pinned host tensors."""

    _STOP = object()

    def __init__(
        self,
        cache: CacheBundle,
        schedule: CounterSchedule,
        *,
        start_step: int,
        stop_step: int,
        depth: int,
        pin_memory: bool,
    ) -> None:
        if depth < 1:
            raise ValueError("prefetch depth must be positive")
        self.cache = cache
        self.schedule = schedule
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self.pin_memory = bool(pin_memory)
        self.queue: queue.Queue[HostBatch | BaseException | object] = queue.Queue(depth)
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _put_unless_closed(self, value: HostBatch | BaseException | object) -> bool:
        while not self._closed.is_set():
            try:
                self.queue.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _worker(self) -> None:
        try:
            for step in range(self.start_step, self.stop_step):
                if self._closed.is_set():
                    break
                rows = self.schedule.rows_for_step(step)
                index = self.cache.index
                payload = HostBatch(
                    step=step,
                    rows=_pinned_tensor(rows, pin=self.pin_memory),
                    tuple_id=_pinned_tensor(index["tuple_id"][rows], pin=self.pin_memory),
                    horizon=_pinned_tensor(index["horizon"][rows], pin=self.pin_memory),
                    stratum=_pinned_tensor(index["stratum"][rows], pin=self.pin_memory),
                    online_previous=_pinned_tensor(
                        self.cache.rgb[index["online_previous"][rows]],
                        pin=self.pin_memory,
                    ),
                    online_current=_pinned_tensor(
                        self.cache.rgb[index["online_current"][rows]],
                        pin=self.pin_memory,
                    ),
                    target_previous=_pinned_tensor(
                        self.cache.rgb[index["target_previous"][rows]],
                        pin=self.pin_memory,
                    ),
                    target_current=_pinned_tensor(
                        self.cache.rgb[index["target_current"][rows]],
                        pin=self.pin_memory,
                    ),
                )
                if not self._put_unless_closed(payload):
                    return
        except BaseException as error:
            self._put_unless_closed(error)
        finally:
            self._put_unless_closed(self._STOP)

    def __iter__(self) -> "BatchPrefetcher":
        return self

    def __next__(self) -> HostBatch:
        value = self.queue.get()
        if value is self._STOP:
            raise StopIteration
        if isinstance(value, BaseException):
            raise value
        assert isinstance(value, HostBatch)
        return value

    def close(self) -> None:
        self._closed.set()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise RuntimeError("prefetch worker did not stop")

    def __enter__(self) -> "BatchPrefetcher":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _uint8_to_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return (
        value.to(device=device, dtype=torch.float32, non_blocking=True)
        .permute(0, 3, 1, 2)
        .div_(255.0)
    )


def augment_tuple_vectorized(
    frames: torch.Tensor,
    *,
    generator: torch.Generator,
    crop_scale_min: float = 0.8,
    brightness: float = 0.2,
    contrast: float = 0.2,
) -> torch.Tensor:
    """Independent per-sample views with one transform shared over tuple time."""

    if frames.ndim != 5 or frames.shape[2] != 3:
        raise ValueError("augmentation input must be [B,T,3,H,W]")
    batch, time_count, channels, height, width = frames.shape
    draws = torch.rand((batch, 5), generator=generator, device="cpu").to(
        frames.device
    )
    scale = crop_scale_min + (1.0 - crop_scale_min) * draws[:, 0]
    side = scale.sqrt()
    max_offset = 1.0 - side
    center_x = -1.0 + side + 2.0 * draws[:, 1] * max_offset
    center_y = -1.0 + side + 2.0 * draws[:, 2] * max_offset
    theta = frames.new_zeros((batch, 2, 3))
    theta[:, 0, 0] = side
    theta[:, 1, 1] = side
    theta[:, 0, 2] = center_x
    theta[:, 1, 2] = center_y
    flat = frames.reshape(batch * time_count, channels, height, width)
    repeated = theta.repeat_interleave(time_count, dim=0)
    grid = F.affine_grid(repeated, flat.shape, align_corners=False)
    value = F.grid_sample(
        flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).reshape(batch, time_count, channels, height, width)
    brightness_factor = 1.0 + brightness * (2.0 * draws[:, 3] - 1.0)
    contrast_factor = 1.0 + contrast * (2.0 * draws[:, 4] - 1.0)
    mean = value.mean(dim=(-2, -1), keepdim=True)
    value = (value - mean) * contrast_factor[:, None, None, None, None] + mean
    value = value * brightness_factor[:, None, None, None, None]
    return value.clamp_(0.0, 1.0)


def _arm_inputs(
    batch: HostBatch,
    selection: slice,
    *,
    arm: DynamicsArm,
    device: torch.device,
    online_generator: torch.Generator,
    target_generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    current = _uint8_to_device(batch.online_current[selection], device)
    target_current = (
        current if arm == "B" else _uint8_to_device(batch.target_current[selection], device)
    )
    if arm == "D":
        previous = _uint8_to_device(batch.online_previous[selection], device)
        target_previous = _uint8_to_device(batch.target_previous[selection], device)
        online_pair = augment_tuple_vectorized(
            torch.stack((previous, current), dim=1), generator=online_generator
        )
        target_pair = augment_tuple_vectorized(
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
            "online_current": augment_tuple_vectorized(
                current[:, None], generator=online_generator
            )[:, 0],
            "target_current": augment_tuple_vectorized(
                target_current[:, None], generator=target_generator
            )[:, 0],
        }
    result["horizon"] = batch.horizon[selection].to(
        device=device, dtype=torch.long, non_blocking=True
    )
    return result


def initialize_shared_predictor(model: EMADynamicsPretrainer, seed: int) -> str:
    """Reset predictor independently of D's extra-convolution RNG consumption."""

    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed ^ 0x51A2D1C5)
        for module in model.predictor.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
    digest = hashlib.sha256()
    for name, value in sorted(model.predictor.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model(config: TrainConfig) -> tuple[EMADynamicsPretrainer, dict[str, Any]]:
    initialization = validate_imagenet_initialization()
    torch.manual_seed(config.seed)
    model = EMADynamicsPretrainer(
        config.arm,
        horizons=config.horizons,
        weights=ResNet18_Weights.IMAGENET1K_V1,
    )
    predictor_sha256 = initialize_shared_predictor(model, config.seed)
    initialization = {
        **initialization,
        "shared_predictor_seed": config.seed ^ 0x51A2D1C5,
        "shared_predictor_sha256": predictor_sha256,
    }
    return model, initialization


def normalized_l1_per_example(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return (
        F.normalize(prediction.float(), dim=-1)
        - F.normalize(target.float(), dim=-1)
    ).abs().mean(dim=-1)


def nearest_neighbor_unique_fraction(
    representations: torch.Tensor, *, chunk_size: int = 512
) -> float:
    value = F.normalize(representations.float(), dim=-1)
    count = len(value)
    if count < 2:
        raise ValueError("nearest-neighbor diversity requires two samples")
    neighbors: list[torch.Tensor] = []
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        similarity = value[start:stop] @ value.T
        rows = torch.arange(stop - start, device=value.device)
        columns = torch.arange(start, stop, device=value.device)
        similarity[rows, columns] = -torch.inf
        neighbors.append(similarity.argmax(dim=1).cpu())
    return float(torch.unique(torch.cat(neighbors)).numel() / count)


def representation_metrics(value: torch.Tensor) -> dict[str, float]:
    representations = value.float()
    centered = representations - representations.mean(dim=0, keepdim=True)
    std = centered.square().mean(dim=0).sqrt()
    trace = centered.square().sum() / len(centered)
    gram = centered @ centered.T / len(centered)
    denominator = gram.square().sum()
    rank = (
        float((trace.square() / denominator.clamp_min(1e-12)).detach().cpu())
        if float(denominator.detach().cpu()) > 0
        else 0.0
    )
    normalized = F.normalize(representations, dim=-1)
    cosine = (
        normalized.sum(dim=0).square().sum() - normalized.square().sum()
    ) / (len(normalized) * (len(normalized) - 1))
    return {
        "per_dimension_std_min": float(std.min().cpu()),
        "per_dimension_std_median": float(std.median().cpu()),
        "per_dimension_std_mean": float(std.mean().cpu()),
        "effective_rank": rank,
        "mean_pairwise_cosine": float(cosine.cpu()),
        "angular_diversity": float((1.0 - cosine).cpu()),
        "nearest_neighbor_unique_fraction": nearest_neighbor_unique_fraction(
            representations
        ),
    }


@torch.inference_mode()
def evaluate_fixed_panel(
    model: EMADynamicsPretrainer,
    cache: CacheBundle,
    panel_rows: np.ndarray,
    *,
    arm: DynamicsArm,
    device: torch.device,
    microbatch_size: int,
    step: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    index = cache.index
    prediction_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    identity_parts: list[torch.Tensor] = []
    online_parts: list[torch.Tensor] = []
    stratum_parts: list[torch.Tensor] = []
    for start in range(0, len(panel_rows), microbatch_size):
        rows = panel_rows[start : start + microbatch_size]
        current = torch.from_numpy(
            np.ascontiguousarray(cache.rgb[index["online_current"][rows]])
        )
        current = _uint8_to_device(current, device)
        target_current = (
            current
            if arm == "B"
            else _uint8_to_device(
                torch.from_numpy(
                    np.ascontiguousarray(cache.rgb[index["target_current"][rows]])
                ),
                device,
            )
        )
        horizon = torch.from_numpy(index["horizon"][rows].astype(np.int64)).to(device)
        if arm == "D":
            previous = _uint8_to_device(
                torch.from_numpy(
                    np.ascontiguousarray(cache.rgb[index["online_previous"][rows]])
                ),
                device,
            )
            target_previous = _uint8_to_device(
                torch.from_numpy(
                    np.ascontiguousarray(cache.rgb[index["target_previous"][rows]])
                ),
                device,
            )
            kwargs = {
                "online_previous": previous,
                "target_previous": target_previous,
            }
        else:
            previous = target_previous = None
            kwargs = {}
        with (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        ):
            output = model(
                online_current=current,
                target_current=target_current,
                horizon=horizon,
                **kwargs,
            )
            identity = model._encode(  # Data-boundary semantics live in model.
                model.target_encoder, current, previous
            )
        prediction_parts.append(output.prediction.float().cpu())
        target_parts.append(output.target.float().cpu())
        identity_parts.append(identity.float().cpu())
        online_parts.append(output.online.float().cpu())
        stratum_parts.append(torch.from_numpy(index["stratum"][rows].astype(np.int64)))
    prediction = torch.cat(prediction_parts)
    target = torch.cat(target_parts)
    identity = torch.cat(identity_parts)
    online = torch.cat(online_parts)
    strata = torch.cat(stratum_parts)
    metrics = {
        "online": representation_metrics(online),
        "target": representation_metrics(target),
    }
    losses: dict[str, Any] = {}
    for stratum, label in ((0, "static"), (1, "change")):
        mask = strata == stratum
        selected_prediction = prediction[mask]
        selected_target = target[mask]
        selected_identity = identity[mask]
        if len(selected_target) < 2:
            raise ValueError(f"fixed panel {label} stratum is too small")
        mean_target = selected_target.mean(dim=0, keepdim=True).expand_as(
            selected_target
        )
        shuffled = torch.roll(selected_target, shifts=1, dims=0)
        losses[label] = {
            "examples": int(mask.sum()),
            "model_normalized_l1": float(
                normalized_l1_per_example(
                    selected_prediction, selected_target
                ).mean()
            ),
            "identity_current_normalized_l1": float(
                normalized_l1_per_example(
                    selected_identity, selected_target
                ).mean()
            ),
            "mean_target_normalized_l1": float(
                normalized_l1_per_example(mean_target, selected_target).mean()
            ),
            "shuffled_future_normalized_l1": float(
                normalized_l1_per_example(
                    selected_prediction, shuffled
                ).mean()
            ),
        }
    if was_training:
        model.train()
    return {
        "step": int(step),
        "panel_rows": len(panel_rows),
        "representations": metrics,
        "baselines_by_motion_stratum": losses,
    }


def apply_collapse_gate(
    diagnostic: Mapping[str, Any],
    state: CollapseGateState | None,
    *,
    relative_floor: float,
    effective_rank_min: float,
    mean_cosine_max: float,
    nn_unique_fraction_min: float,
) -> tuple[CollapseGateState, dict[str, Any]]:
    representations = diagnostic.get("representations")
    if not isinstance(representations, Mapping):
        raise ValueError("collapse diagnostic lacks online/target representations")
    current = {
        f"{side}.{metric}": float(representations[side][metric])
        for side in ("online", "target")
        for metric in (
            "per_dimension_std_median",
            "effective_rank",
            "mean_pairwise_cosine",
            "nearest_neighbor_unique_fraction",
        )
    }
    relative_names = tuple(
        f"{side}.{metric}"
        for side in ("online", "target")
        for metric in ("per_dimension_std_median", "effective_rank")
    )
    initial = current if state is None else state.initial

    def safe_ratio(name: str) -> float:
        denominator = initial[name]
        numerator = current[name]
        if not math.isfinite(numerator) or not math.isfinite(denominator):
            return math.nan
        if denominator <= 0:
            return 1.0 if numerator == denominator else math.inf
        return numerator / denominator

    ratios = {name: safe_ratio(name) for name in relative_names}
    relative_failed = [
        name
        for name in relative_names
        if not math.isfinite(ratios[name]) or ratios[name] < relative_floor
    ]
    absolute_failed: list[str] = []
    for side in ("online", "target"):
        rank_name = f"{side}.effective_rank"
        cosine_name = f"{side}.mean_pairwise_cosine"
        nn_name = f"{side}.nearest_neighbor_unique_fraction"
        if not math.isfinite(current[rank_name]) or current[rank_name] < effective_rank_min:
            absolute_failed.append(rank_name)
        if not math.isfinite(current[cosine_name]) or current[cosine_name] > mean_cosine_max:
            absolute_failed.append(cosine_name)
        if not math.isfinite(current[nn_name]) or current[nn_name] < nn_unique_fraction_min:
            absolute_failed.append(nn_name)
    failed = sorted(set(relative_failed + absolute_failed))
    previous_consecutive = 0 if state is None else state.consecutive_failures
    consecutive = previous_consecutive + 1 if failed else 0
    next_state = CollapseGateState(
        consecutive_failures=consecutive, initial=initial
    )
    return next_state, {
        "phase": "step_zero_baseline" if state is None else "training",
        "status": "fail" if failed else "pass",
        "thresholds": {
            "relative_floor": relative_floor,
            "effective_rank_min": effective_rank_min,
            "mean_pairwise_cosine_max": mean_cosine_max,
            "nearest_neighbor_unique_fraction_min": nn_unique_fraction_min,
        },
        "relative_ratios": ratios,
        "relative_failed_metrics": relative_failed,
        "absolute_failed_metrics": absolute_failed,
        "failed_metrics": failed,
        "consecutive_failures": consecutive,
    }


def _capture_rng(
    online_generator: torch.Generator, target_generator: torch.Generator
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "online_augmentation": online_generator.get_state(),
        "target_augmentation": target_generator.get_state(),
    }


def _restore_rng(
    state: Mapping[str, Any],
    online_generator: torch.Generator,
    target_generator: torch.Generator,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    online_generator.set_state(state["online_augmentation"])
    target_generator.set_state(state["target_augmentation"])


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def checkpoint_payload(
    *,
    kind: Literal["resume", "final", "failed_forensic"],
    model: EMADynamicsPretrainer,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    config_sha256: str,
    cache: CacheBundle,
    schedule_receipt: Mapping[str, Any],
    completed_steps: int,
    online_generator: torch.Generator,
    target_generator: torch.Generator,
    collapse_state: CollapseGateState,
    diagnostics: list[dict[str, Any]],
    losses: list[float],
    training_seconds: float,
    preflight_step_seconds: list[float],
    peak_cuda_memory_reserved_bytes: int,
    resumable: bool,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "kind": kind,
        "selection_eligible": kind == "final",
        "resumable": resumable,
        "created_at": utc_now(),
        "arm": config.arm,
        "horizons": list(config.horizons),
        "completed_steps": completed_steps,
        "next_step": completed_steps,
        "config": json_native(asdict(config)),
        "config_sha256": config_sha256,
        "cache": {
            "manifest_sha256": cache.manifest_sha256,
            "rgb_sha256": cache.rgb_sha256,
            "index_sha256": cache.index_sha256,
        },
        "schedule": dict(schedule_receipt),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "amp": {
            "enabled": config.device.startswith("cuda"),
            "dtype": "bfloat16" if config.device.startswith("cuda") else "float32",
            "gradient_scaler": None,
        },
        "rng_state": _capture_rng(online_generator, target_generator),
        "collapse_gate": asdict(collapse_state),
        "diagnostics": diagnostics,
        "losses": losses,
        "training_seconds": training_seconds,
        "preflight": {
            "step_seconds": preflight_step_seconds,
            "peak_cuda_memory_reserved_bytes": peak_cuda_memory_reserved_bytes,
        },
    }


def validate_train_config(config: TrainConfig) -> None:
    if config.arm not in ("B", "C", "D"):
        raise ValueError("arm must be B, C, or D")
    if (
        not config.horizons
        or any(value <= 0 for value in config.horizons)
        or tuple(sorted(set(config.horizons))) != config.horizons
    ):
        raise ValueError("horizons must be sorted unique positive integers")
    if config.max_steps < config.throughput_steps or config.throughput_steps < 1:
        raise ValueError("run must include the complete throughput gate")
    divisor = 2 * len(config.horizons)
    if config.global_batch_size % divisor:
        raise ValueError("global batch is not divisible by horizon/stratum buckets")
    if (
        config.microbatch_size < 1
        or config.global_batch_size % config.microbatch_size
    ):
        raise ValueError("microbatch must divide global batch")
    if config.checkpoint_interval < 1 or config.diagnostic_interval < 1:
        raise ValueError("checkpoint and diagnostic intervals must be positive")
    if config.prefetch_depth < 1:
        raise ValueError("prefetch depth must be positive")
    if config.panel_size < 2 * divisor:
        raise ValueError("fixed panel is too small for every horizon/stratum")
    if config.collapse_consecutive_failures != 3:
        raise ValueError("collapse stop rule is frozen at three consecutive failures")
    if not math.isfinite(config.collapse_relative_floor) or not (
        0 < config.collapse_relative_floor < 1
    ):
        raise ValueError("collapse relative floor must lie in (0,1)")
    if (
        not math.isfinite(config.collapse_effective_rank_min)
        or config.collapse_effective_rank_min <= 0
        or not math.isfinite(config.collapse_mean_cosine_max)
        or not 0 < config.collapse_mean_cosine_max < 1
        or not math.isfinite(config.collapse_nn_unique_fraction_min)
        or not 0 < config.collapse_nn_unique_fraction_min <= 1
    ):
        raise ValueError("absolute collapse thresholds are malformed")
    if not math.isfinite(config.max_projected_seconds) or config.max_projected_seconds < 0:
        raise ValueError("throughput budget must be nonnegative")
    if not math.isfinite(config.max_cuda_memory_gib) or config.max_cuda_memory_gib <= 0:
        raise ValueError("CUDA memory budget must be positive")
    if not config.study_id or not config.run_id:
        raise ValueError("study and run IDs are required")
    if (
        not math.isfinite(config.learning_rate)
        or config.learning_rate <= 0
        or not math.isfinite(config.weight_decay)
        or config.weight_decay < 0
    ):
        raise ValueError("optimizer hyperparameters are malformed")
    if not 0 <= config.seed < 2**63 or not 0 <= config.schedule_seed < 2**63:
        raise ValueError("training and schedule seeds must be nonnegative int64")
    if len(config.implementation_commit) != 40 or any(
        value not in "0123456789abcdef" for value in config.implementation_commit
    ):
        raise ValueError("implementation commit must be a full lowercase SHA-1")


def _write_static_receipts(
    output: Path,
    config: TrainConfig,
    config_sha256: str,
    cache: CacheBundle,
    schedule_receipt: Mapping[str, Any],
    panel_rows: np.ndarray,
) -> None:
    _atomic_json(output / "resolved_config.json", {
        "schema_version": RUN_SCHEMA,
        "config": asdict(config),
        "config_sha256": config_sha256,
        "selection_policy": "final_weights_only",
        "post_result_override": True,
    })
    _atomic_json(output / "cache_receipt.json", {
        "manifest_path": str(cache.manifest_path),
        "manifest_sha256": cache.manifest_sha256,
        "rgb_sha256": cache.rgb_sha256,
        "index_sha256": cache.index_sha256,
        "labels_loaded": False,
        "exclusion_proof": cache.manifest["exclusion_proof"],
    })
    _atomic_json(output / "schedule_receipt.json", schedule_receipt)
    tuple_ids = cache.index["tuple_id"][panel_rows]
    _atomic_json(output / "panel_receipt.json", {
        "kind": "fixed_training_only_collapse_panel",
        "rows": len(panel_rows),
        "balanced_horizon_stratum_cells": True,
        "index_rows_sha256": array_sha256(panel_rows),
        "tuple_ids_sha256": array_sha256(tuple_ids),
        "uses_evaluation_data": False,
    })


def _validate_resume_payload(
    payload: Mapping[str, Any],
    *,
    config: TrainConfig,
    config_sha256: str,
    cache: CacheBundle,
    schedule_receipt: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("resume checkpoint schema changed")
    if payload.get("resumable") is not True or payload.get("kind") != "resume":
        raise ValueError("checkpoint is not a resume artifact")
    if (
        payload.get("config_sha256") != config_sha256
        or payload.get("config") != json_native(asdict(config))
    ):
        raise ValueError("resume config differs from current run")
    expected_cache = {
        "manifest_sha256": cache.manifest_sha256,
        "rgb_sha256": cache.rgb_sha256,
        "index_sha256": cache.index_sha256,
    }
    if payload.get("cache") != expected_cache:
        raise ValueError("resume cache identity differs")
    if payload.get("schedule") != dict(schedule_receipt):
        raise ValueError("resume schedule differs")


def run_training(
    *,
    manifest_path: Path,
    output_dir: Path,
    config: TrainConfig,
    resume_checkpoint: Path | None = None,
    model_factory: Callable[
        [TrainConfig], tuple[EMADynamicsPretrainer, dict[str, Any]]
    ] = build_model,
    step_hook: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run or exactly resume one immutable arm."""

    validate_train_config(config)
    output = Path(output_dir)
    reject_forbidden_path_before_access(output)
    if resume_checkpoint is None:
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to reuse output directory: {output}")
        output.mkdir(parents=True)
    else:
        if not output.is_dir() or output.is_symlink():
            raise ValueError("resume output must be an existing regular directory")
        if any(
            (output / name).exists()
            for name in ("final.pt", "run_receipt.json", "completion.json")
        ):
            raise ValueError("refusing to resume an already completed output")
        resume_checkpoint = Path(resume_checkpoint)
        reject_forbidden_path_before_access(resume_checkpoint)
        _regular_unsymlinked(resume_checkpoint, "resume checkpoint")
        if resume_checkpoint.parent.parent.resolve() != output.resolve():
            raise ValueError("resume checkpoint must belong to output/resume")
        available_resume = sorted((output / "resume").glob("step_*.pt"))
        if not available_resume or resume_checkpoint.resolve() != available_resume[-1].resolve():
            raise ValueError("resume checkpoint must be the latest published recovery state")

    cache: CacheBundle | None = None
    model: EMADynamicsPretrainer | None = None
    optimizer: torch.optim.Optimizer | None = None
    online_generator = torch.Generator(device="cpu")
    target_generator = torch.Generator(device="cpu")
    completed_steps = 0
    training_seconds = 0.0
    preflight_step_seconds: list[float] = []
    peak_cuda_memory_reserved_bytes = 0
    losses: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    collapse_state: CollapseGateState | None = None
    config_sha256 = canonical_json_sha256(asdict(config))
    schedule_receipt: dict[str, Any] = {}
    in_flight = False
    try:
        cache = load_cache(manifest_path, expected_horizons=config.horizons)
        schedule = CounterSchedule(
            cache.index,
            horizons=config.horizons,
            batch_size=config.global_batch_size,
            seed=config.schedule_seed,
            max_steps=config.max_steps,
            source_probabilities=cache.source_probabilities,
        )
        schedule_receipt = schedule.receipt(cache.index["tuple_id"])
        panel_rows = fixed_panel_rows(schedule, total=config.panel_size)
        if resume_checkpoint is None:
            _write_static_receipts(
                output, config, config_sha256, cache, schedule_receipt, panel_rows
            )
        else:
            for required in (
                "resolved_config.json", "cache_receipt.json",
                "schedule_receipt.json", "panel_receipt.json",
            ):
                if not (output / required).is_file():
                    raise ValueError(f"resume output lacks {required}")
            if _read_json(output / "schedule_receipt.json", "schedule receipt") != schedule_receipt:
                raise ValueError("published schedule receipt changed")

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(True)
        device = torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device does not support required bfloat16 autocast")
        if device.type not in ("cpu", "cuda"):
            raise ValueError("streaming trainer supports only CPU tests or CUDA runs")
        model, initialization = model_factory(config)
        model.to(device).train()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        online_generator.manual_seed(config.seed + 1)
        target_generator.manual_seed(config.seed + 2)
        if resume_checkpoint is not None:
            payload = torch.load(
                resume_checkpoint, map_location="cpu", weights_only=False
            )
            _validate_resume_payload(
                payload,
                config=config,
                config_sha256=config_sha256,
                cache=cache,
                schedule_receipt=schedule_receipt,
            )
            model.load_state_dict(payload["model_state"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state"])
            _optimizer_to_device(optimizer, device)
            completed_steps = int(payload["completed_steps"])
            losses = list(payload["losses"])
            diagnostics = list(payload["diagnostics"])
            training_seconds = float(payload["training_seconds"])
            preflight = payload.get("preflight")
            if not isinstance(preflight, Mapping):
                raise ValueError("resume checkpoint lacks preflight state")
            preflight_step_seconds = list(preflight.get("step_seconds", []))
            peak_cuda_memory_reserved_bytes = int(
                preflight.get("peak_cuda_memory_reserved_bytes", 0)
            )
            if len(preflight_step_seconds) != min(
                completed_steps, config.throughput_steps
            ):
                raise ValueError("resume preflight timing state changed")
            collapse_state = CollapseGateState(**payload["collapse_gate"])
            _restore_rng(
                payload["rng_state"], online_generator, target_generator
            )

        if collapse_state is None:
            initial_diagnostic = evaluate_fixed_panel(
                model,
                cache,
                panel_rows,
                arm=config.arm,
                device=device,
                microbatch_size=config.microbatch_size,
                step=0,
            )
            collapse_state, gate = apply_collapse_gate(
                initial_diagnostic,
                None,
                relative_floor=config.collapse_relative_floor,
                effective_rank_min=config.collapse_effective_rank_min,
                mean_cosine_max=config.collapse_mean_cosine_max,
                nn_unique_fraction_min=config.collapse_nn_unique_fraction_min,
            )
            initial_diagnostic["collapse_gate"] = gate
            diagnostics.append(initial_diagnostic)
            _atomic_json(output / "diagnostics.json", diagnostics)

        pin = device.type == "cuda"
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        next_step_started = time.perf_counter()
        with BatchPrefetcher(
            cache,
            schedule,
            start_step=completed_steps,
            stop_step=config.max_steps,
            depth=config.prefetch_depth,
            pin_memory=pin,
        ) as batches:
            for host_batch in batches:
                if host_batch.step != completed_steps:
                    raise RuntimeError("prefetch schedule step changed")
                in_flight = True
                step_started = next_step_started
                optimizer.zero_grad(set_to_none=True)
                micro_count = config.global_batch_size // config.microbatch_size
                weighted_loss = 0.0
                for start in range(0, config.global_batch_size, config.microbatch_size):
                    selection = slice(start, start + config.microbatch_size)
                    inputs = _arm_inputs(
                        host_batch,
                        selection,
                        arm=config.arm,
                        device=device,
                        online_generator=online_generator,
                        target_generator=target_generator,
                    )
                    amp = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if device.type == "cuda"
                        else nullcontext()
                    )
                    with amp:
                        output_value = model(**inputs)
                        loss = output_value.loss / micro_count
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(
                            f"non-finite loss at step {completed_steps + 1}"
                        )
                    loss.backward()
                    weighted_loss += float(loss.detach().cpu())
                for parameter in model.parameters():
                    if parameter.grad is not None and not bool(
                        torch.isfinite(parameter.grad).all()
                    ):
                        raise FloatingPointError(
                            f"non-finite gradient at step {completed_steps + 1}"
                        )
                optimizer.step()
                if config.max_steps == 1:
                    momentum = 0.998
                else:
                    momentum = 0.998 + 0.002 * completed_steps / (
                        config.max_steps - 1
                    )
                model.update_target(momentum)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                completed_steps += 1
                losses.append(weighted_loss)
                elapsed = time.perf_counter() - step_started
                training_seconds += elapsed
                if len(preflight_step_seconds) < config.throughput_steps:
                    preflight_step_seconds.append(elapsed)
                if device.type == "cuda":
                    peak_cuda_memory_reserved_bytes = max(
                        peak_cuda_memory_reserved_bytes,
                        int(torch.cuda.max_memory_reserved(device)),
                    )
                in_flight = False

                if (
                    completed_steps % config.diagnostic_interval == 0
                    or completed_steps == config.max_steps
                ):
                    diagnostic = evaluate_fixed_panel(
                        model,
                        cache,
                        panel_rows,
                        arm=config.arm,
                        device=device,
                        microbatch_size=config.microbatch_size,
                        step=completed_steps,
                    )
                    collapse_state, gate = apply_collapse_gate(
                        diagnostic,
                        collapse_state,
                        relative_floor=config.collapse_relative_floor,
                        effective_rank_min=config.collapse_effective_rank_min,
                        mean_cosine_max=config.collapse_mean_cosine_max,
                        nn_unique_fraction_min=config.collapse_nn_unique_fraction_min,
                    )
                    diagnostic["collapse_gate"] = gate
                    diagnostics.append(diagnostic)
                    _atomic_json(output / "diagnostics.json", diagnostics)
                    if (
                        collapse_state.consecutive_failures
                        >= config.collapse_consecutive_failures
                    ):
                        raise RepresentationCollapse(
                            "three consecutive fixed-panel collapse gates failed"
                        )

                if completed_steps == config.throughput_steps and not (
                    output / "throughput.json"
                ).exists():
                    if len(preflight_step_seconds) != config.throughput_steps:
                        raise RuntimeError("throughput timing window is incomplete")
                    measured = sum(preflight_step_seconds)
                    projected = measured / config.throughput_steps * config.max_steps
                    max_cuda_memory_bytes = int(
                        config.max_cuda_memory_gib * 1024**3
                    )
                    time_pass = (
                        config.max_projected_seconds == 0
                        or projected <= config.max_projected_seconds
                    )
                    memory_pass = (
                        device.type != "cuda"
                        or peak_cuda_memory_reserved_bytes <= max_cuda_memory_bytes
                    )
                    throughput = {
                        "schema_version": "madeleine.dynamics-throughput.v1",
                        "measured_steps": config.throughput_steps,
                        "measured_training_seconds": measured,
                        "projected_total_training_seconds": projected,
                        "max_projected_seconds": config.max_projected_seconds,
                        "global_batch_size": config.global_batch_size,
                        "microbatch_size": config.microbatch_size,
                        "arm": config.arm,
                        "peak_cuda_memory_reserved_bytes": peak_cuda_memory_reserved_bytes,
                        "max_cuda_memory_bytes": max_cuda_memory_bytes,
                        "cache_manifest_sha256": cache.manifest_sha256,
                        "time_gate_passed": time_pass,
                        "memory_gate_passed": memory_pass,
                        "status": "pass" if time_pass and memory_pass else "fail",
                    }
                    _atomic_json(output / "throughput.json", throughput)
                    if throughput["status"] == "fail":
                        raise ThroughputBudgetExceeded(
                            "100-step throughput projection exceeds budget"
                        )

                if (
                    completed_steps % config.checkpoint_interval == 0
                    and completed_steps < config.max_steps
                ):
                    checkpoint = checkpoint_payload(
                        kind="resume",
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        config_sha256=config_sha256,
                        cache=cache,
                        schedule_receipt=schedule_receipt,
                        completed_steps=completed_steps,
                        online_generator=online_generator,
                        target_generator=target_generator,
                        collapse_state=collapse_state,
                        diagnostics=diagnostics,
                        losses=losses,
                        training_seconds=training_seconds,
                        preflight_step_seconds=preflight_step_seconds,
                        peak_cuda_memory_reserved_bytes=peak_cuda_memory_reserved_bytes,
                        resumable=True,
                    )
                    _atomic_torch_save(
                        output / "resume" / f"step_{completed_steps:08d}.pt",
                        checkpoint,
                    )
                if step_hook is not None:
                    step_hook(completed_steps)
                next_step_started = time.perf_counter()

        assert model is not None and optimizer is not None and collapse_state is not None
        final_payload = checkpoint_payload(
            kind="final",
            model=model,
            optimizer=optimizer,
            config=config,
            config_sha256=config_sha256,
            cache=cache,
            schedule_receipt=schedule_receipt,
            completed_steps=completed_steps,
            online_generator=online_generator,
            target_generator=target_generator,
            collapse_state=collapse_state,
            diagnostics=diagnostics,
            losses=losses,
            training_seconds=training_seconds,
            preflight_step_seconds=preflight_step_seconds,
            peak_cuda_memory_reserved_bytes=peak_cuda_memory_reserved_bytes,
            resumable=False,
        )
        final_path = output / "final.pt"
        _atomic_torch_save(final_path, final_payload)
        final_hash = sha256_file(final_path)
        receipt = {
            "schema_version": RUN_SCHEMA,
            "status": "complete",
            "completed_at": utc_now(),
            "study_id": config.study_id,
            "run_id": config.run_id,
            "implementation_commit": config.implementation_commit,
            "arm": config.arm,
            "horizons": list(config.horizons),
            "completed_steps": completed_steps,
            "selection_policy": "final_weights_only",
            "config_sha256": config_sha256,
            "initialization": initialization,
            "cache_manifest_sha256": cache.manifest_sha256,
            "schedule_sha256": canonical_json_sha256(schedule_receipt),
            "final_checkpoint": {
                "path": "final.pt",
                "sha256": final_hash,
                "selection_eligible": True,
            },
            "loss": {
                "first": losses[0],
                "final": losses[-1],
                "minimum": min(losses),
                "mean": float(np.mean(losses)),
                "trace_sha256": array_sha256(np.asarray(losses, dtype=np.float64)),
            },
            "diagnostics": {
                "path": "diagnostics.json",
                "sha256": sha256_file(output / "diagnostics.json"),
                "checks": len(diagnostics),
                "terminal_gate": diagnostics[-1]["collapse_gate"],
            },
            "throughput": {
                "path": "throughput.json",
                "sha256": sha256_file(output / "throughput.json"),
            },
        }
        _atomic_json(output / "run_receipt.json", receipt)
        completion = {
            "schema_version": COMPLETION_SCHEMA,
            "status": "complete",
            "run_receipt_sha256": sha256_file(output / "run_receipt.json"),
            "final_checkpoint_sha256": final_hash,
            "config_sha256": config_sha256,
            "cache_manifest_sha256": cache.manifest_sha256,
        }
        _atomic_json(output / "completion.json", completion)
        return receipt
    except BaseException as error:
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "status": "failed",
            "failed_at": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_steps": completed_steps,
            "in_flight_step_mutated": in_flight,
            "config_sha256": config_sha256,
            "cache_manifest_sha256": (
                None if cache is None else cache.manifest_sha256
            ),
            "preserved": True,
        }
        failure_path = output / f"failure_{int(time.time_ns())}.json"
        _atomic_json(failure_path, failure)
        if (
            model is not None
            and optimizer is not None
            and cache is not None
            and collapse_state is not None
        ):
            forensic = checkpoint_payload(
                kind="failed_forensic",
                model=model,
                optimizer=optimizer,
                config=config,
                config_sha256=config_sha256,
                cache=cache,
                schedule_receipt=schedule_receipt,
                completed_steps=completed_steps,
                online_generator=online_generator,
                target_generator=target_generator,
                collapse_state=collapse_state,
                diagnostics=diagnostics,
                losses=losses,
                training_seconds=training_seconds,
                preflight_step_seconds=preflight_step_seconds,
                peak_cuda_memory_reserved_bytes=peak_cuda_memory_reserved_bytes,
                resumable=False,
            )
            _atomic_torch_save(
                output / f"failed_state_{int(time.time_ns())}.pt", forensic
            )
        raise


def validate_completed_run(output_dir: Path) -> dict[str, Any]:
    """Independently bind a completed output inventory and final checkpoint."""

    root = Path(output_dir)
    reject_forbidden_path_before_access(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("completed run root must be a regular directory")
    required = {
        "resolved_config.json",
        "cache_receipt.json",
        "schedule_receipt.json",
        "panel_receipt.json",
        "diagnostics.json",
        "throughput.json",
        "final.pt",
        "run_receipt.json",
        "completion.json",
    }
    observed_files = {path.name for path in root.iterdir() if path.is_file()}
    missing = required - observed_files
    if missing:
        raise ValueError(f"completed run lacks artifacts: {sorted(missing)}")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("completed run contains symlink artifacts")
    if any(path.name.startswith(".") or ".tmp" in path.name for path in root.rglob("*")):
        raise ValueError("completed run contains temporary artifacts")
    allowed_direct = required | {
        name
        for name in observed_files
        if name.startswith("failure_") and name.endswith(".json")
        or name.startswith("failed_state_") and name.endswith(".pt")
    }
    unknown_direct = observed_files - allowed_direct
    if unknown_direct:
        raise ValueError(f"completed run contains unknown artifacts: {sorted(unknown_direct)}")
    directories = {path.name for path in root.iterdir() if path.is_dir()}
    if directories - {"resume"}:
        raise ValueError("completed run contains unknown directories")
    completion = _read_json(root / "completion.json", "completion marker")
    receipt = _read_json(root / "run_receipt.json", "run receipt")
    resolved = _read_json(root / "resolved_config.json", "resolved config")
    if completion.get("schema_version") != COMPLETION_SCHEMA:
        raise ValueError("completion schema changed")
    if receipt.get("schema_version") != RUN_SCHEMA or receipt.get("status") != "complete":
        raise ValueError("run receipt is not complete")
    if completion.get("run_receipt_sha256") != sha256_file(root / "run_receipt.json"):
        raise ValueError("run receipt hash changed")
    if completion.get("final_checkpoint_sha256") != sha256_file(root / "final.pt"):
        raise ValueError("final checkpoint hash changed")
    if completion.get("config_sha256") != resolved.get("config_sha256"):
        raise ValueError("completion config hash changed")
    resolved_config = resolved.get("config")
    if (
        not isinstance(resolved_config, Mapping)
        or canonical_json_sha256(resolved_config) != resolved.get("config_sha256")
    ):
        raise ValueError("resolved config content hash changed")
    if receipt.get("config_sha256") != resolved.get("config_sha256"):
        raise ValueError("run receipt config hash changed")
    for path_key, hash_key in (
        ("diagnostics.json", "diagnostics"),
        ("throughput.json", "throughput"),
    ):
        record = receipt.get(hash_key)
        if (
            not isinstance(record, Mapping)
            or record.get("path") != path_key
            or record.get("sha256") != sha256_file(root / path_key)
        ):
            raise ValueError(f"{hash_key} receipt hash changed")
    payload = torch.load(root / "final.pt", map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA
        or payload.get("kind") != "final"
        or payload.get("selection_eligible") is not True
        or payload.get("resumable") is not False
    ):
        raise ValueError("final checkpoint policy changed")
    if payload.get("completed_steps") != receipt.get("completed_steps"):
        raise ValueError("final checkpoint endpoint changed")
    if payload.get("config_sha256") != resolved.get("config_sha256"):
        raise ValueError("final checkpoint config changed")
    if payload.get("config") != resolved_config:
        raise ValueError("final checkpoint resolved config changed")
    if (
        payload.get("arm") != resolved_config.get("arm")
        or payload.get("horizons") != resolved_config.get("horizons")
        or receipt.get("arm") != resolved_config.get("arm")
        or receipt.get("horizons") != resolved_config.get("horizons")
    ):
        raise ValueError("final arm/horizon identity changed")
    if payload.get("completed_steps") != resolved_config.get("max_steps"):
        raise ValueError("final checkpoint is not at the fixed endpoint")
    cache_receipt = _read_json(root / "cache_receipt.json", "cache receipt")
    expected_cache = {
        "manifest_sha256": cache_receipt.get("manifest_sha256"),
        "rgb_sha256": cache_receipt.get("rgb_sha256"),
        "index_sha256": cache_receipt.get("index_sha256"),
    }
    if payload.get("cache") != expected_cache:
        raise ValueError("final checkpoint cache identity changed")
    schedule_receipt = _read_json(
        root / "schedule_receipt.json", "schedule receipt"
    )
    if payload.get("schedule") != schedule_receipt:
        raise ValueError("final checkpoint schedule identity changed")
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("final checkpoint model state is absent")
    if not any(str(name).startswith("online_encoder.") for name in model_state) or not any(
        str(name).startswith("target_encoder.") for name in model_state
    ):
        raise ValueError("final checkpoint lacks online/EMA encoder state")
    for name, value in model_state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"model state {name} is not a tensor")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"model state {name} is non-finite")
    throughput = _read_json(root / "throughput.json", "throughput receipt")
    if throughput.get("status") != "pass":
        raise ValueError("throughput gate did not pass")
    diagnostics = _read_json_value(root / "diagnostics.json", "diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise ValueError("fixed-panel diagnostics are absent")
    if diagnostics[-1]["collapse_gate"]["status"] != "pass":
        raise ValueError("terminal collapse gate did not pass")
    if diagnostics[-1]["collapse_gate"].get("consecutive_failures") != 0:
        raise ValueError("terminal collapse failure counter is nonzero")
    failure_files = sorted(
        path.name for path in root.glob("failure_*.json") if path.is_file()
    )
    failed_state_files = sorted(
        path.name for path in root.glob("failed_state_*.pt") if path.is_file()
    )
    audit = {
        "schema_version": "madeleine.dynamics-streaming-audit.v1",
        "ok": True,
        "arm": receipt["arm"],
        "completed_steps": receipt["completed_steps"],
        "final_checkpoint_sha256": sha256_file(root / "final.pt"),
        "run_receipt_sha256": sha256_file(root / "run_receipt.json"),
        "completion_sha256": sha256_file(root / "completion.json"),
        "preserved_prior_failures": {
            "failure_receipts": failure_files,
            "forensic_states": failed_state_files,
        },
    }
    return audit


def _parse_horizons(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result:
        raise argparse.ArgumentTypeError("horizons must not be empty")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--cache-manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--resume-checkpoint", type=Path)
    train.add_argument("--arm", choices=("B", "C", "D"), required=True)
    train.add_argument("--horizons", type=_parse_horizons, required=True)
    train.add_argument("--max-steps", type=int, required=True)
    train.add_argument("--global-batch-size", type=int, default=192)
    train.add_argument("--microbatch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--schedule-seed", type=int, default=2026072802)
    train.add_argument("--checkpoint-interval", type=int, default=1000)
    train.add_argument("--diagnostic-interval", type=int, default=250)
    train.add_argument("--panel-size", type=int, default=2048)
    train.add_argument("--prefetch-depth", type=int, default=2)
    train.add_argument("--throughput-steps", type=int, default=100)
    train.add_argument("--max-projected-seconds", type=float, required=True)
    train.add_argument("--collapse-relative-floor", type=float, default=0.25)
    train.add_argument("--collapse-effective-rank-min", type=float, default=8.0)
    train.add_argument("--collapse-mean-cosine-max", type=float, default=0.995)
    train.add_argument(
        "--collapse-nn-unique-fraction-min", type=float, default=0.25
    )
    train.add_argument("--max-cuda-memory-gib", type=float, default=76.0)
    train.add_argument("--device", default="cuda")
    train.add_argument("--study-id", required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--implementation-commit", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        print(json.dumps(validate_completed_run(args.output_dir), indent=2, sort_keys=True))
        return 0
    config = TrainConfig(
        arm=args.arm,
        horizons=args.horizons,
        max_steps=args.max_steps,
        global_batch_size=args.global_batch_size,
        microbatch_size=args.microbatch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        schedule_seed=args.schedule_seed,
        checkpoint_interval=args.checkpoint_interval,
        diagnostic_interval=args.diagnostic_interval,
        panel_size=args.panel_size,
        prefetch_depth=args.prefetch_depth,
        throughput_steps=args.throughput_steps,
        max_projected_seconds=args.max_projected_seconds,
        collapse_relative_floor=args.collapse_relative_floor,
        collapse_effective_rank_min=args.collapse_effective_rank_min,
        collapse_mean_cosine_max=args.collapse_mean_cosine_max,
        collapse_nn_unique_fraction_min=args.collapse_nn_unique_fraction_min,
        collapse_consecutive_failures=3,
        max_cuda_memory_gib=args.max_cuda_memory_gib,
        device=args.device,
        study_id=args.study_id,
        run_id=args.run_id,
        implementation_commit=args.implementation_commit,
    )
    result = run_training(
        manifest_path=args.cache_manifest,
        output_dir=args.output_dir,
        config=config,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
