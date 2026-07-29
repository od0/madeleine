"""Train matched dense and conditional oracle-window event localizers.

This is the Phase-1 diagnostic for coarse-to-fine transition timing.  Every
example names a key and transition polarity and guarantees exactly one such
event in a 16-frame candidate region.  Candidate placement is assigned only
after an event has been shown to support every possible offset, so crop
construction cannot reveal the target position.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from badeline.event_model import EventLatchIDM
from badeline.temporal import AlignedTemporalTCN
from badeline.train import contiguous_runs, read_session_ids, validate_splits
from data.schema import KEY_ORDER


EVENT_TYPES = ("onset", "release")
HEAD_NAMES = tuple(
    f"{key}:{event_type}"
    for event_type in EVENT_TYPES
    for key in KEY_ORDER
)
DEFAULT_TRAIN_IDS = (
    "rec_20260724_190233",
    "rec_20260725_015612",
    "rec_20260725_021338",
)
DEFAULT_VAL_IDS = ("rec_20260724_171305_5min",)
FORBIDDEN_SESSION_IDS = frozenset(
    {
        "rec_20260725_025853",  # val-B
        "rec_20260725_160450_b1",  # repeatedly consulted B1
        "rec_20260727_220000_test",  # sealed untouched test
    }
)


@dataclass(frozen=True)
class FeatureSession:
    session_id: str
    features: np.ndarray
    keys: np.ndarray
    engine_frame_idx: np.ndarray
    input_active: np.ndarray


@dataclass(frozen=True)
class OracleExample:
    split: str
    session_id: str
    run_index: int
    array_index: int
    engine_frame_idx: int
    head_index: int
    key_index: int
    event_type_index: int
    offset: int
    crop_start: int
    candidate_start: int
    block_id: str


@dataclass(frozen=True)
class ConstructionResult:
    examples: tuple[OracleExample, ...]
    counts: Mapping[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(*parts: object) -> int:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_binary(name: str, value: np.ndarray) -> None:
    if not np.all(np.isin(value, (0, 1))):
        raise ValueError(f"{name} must be binary")


def transition_matrix(
    keys: np.ndarray,
    engine_frame_idx: np.ndarray,
    input_active: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return valid onset/release targets in onset-then-release head order."""

    keys = np.asarray(keys)
    engine = np.asarray(engine_frame_idx)
    active = np.asarray(input_active)
    if keys.dtype != np.uint8 or keys.ndim != 2 or keys.shape[1] != len(KEY_ORDER):
        raise ValueError("keys must be uint8 [N,7]")
    if engine.dtype != np.int64 or engine.shape != (len(keys),):
        raise ValueError("engine_frame_idx must be int64 [N]")
    if active.dtype != np.uint8 or active.shape != (len(keys),):
        raise ValueError("input_active must be uint8 [N]")
    _validate_binary("keys", keys)
    _validate_binary("input_active", active)

    raw_change = keys[1:] != keys[:-1]
    valid_pair = (
        (engine[1:] == engine[:-1] + 1)
        & active[:-1].astype(bool)
        & active[1:].astype(bool)
    )
    events = np.zeros((len(keys), len(HEAD_NAMES)), dtype=bool)
    events[1:, : len(KEY_ORDER)] = (
        (keys[:-1] == 0) & (keys[1:] == 1) & valid_pair[:, None]
    )
    events[1:, len(KEY_ORDER) :] = (
        (keys[:-1] == 1) & (keys[1:] == 0) & valid_pair[:, None]
    )
    return events, {
        "raw_key_transitions": int(raw_change.sum()),
        "valid_active_contiguous_transitions": int(events.sum()),
        "invalid_predecessor_transitions": int(
            raw_change.sum() - events.sum()
        ),
    }


def load_feature_session(root: Path, session_id: str) -> FeatureSession:
    """Load one explicitly allowed feature shard, rejecting embargoes first."""

    if session_id in FORBIDDEN_SESSION_IDS or "untouched" in session_id.lower():
        raise ValueError(f"forbidden evaluation session: {session_id}")
    path = root / f"{session_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing requested feature shard: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "features",
            "keys",
            "engine_frame_idx",
            "input_active",
            "session_id",
        }
        if set(archive.files) != required:
            raise ValueError(
                f"{path}: feature fields changed: "
                f"missing={sorted(required - set(archive.files))} "
                f"extra={sorted(set(archive.files) - required)}"
            )
        features = np.asarray(archive["features"])
        keys = np.asarray(archive["keys"])
        engine = np.asarray(archive["engine_frame_idx"])
        active = np.asarray(archive["input_active"])
        stored_id = str(np.asarray(archive["session_id"]).reshape(()).item())
    if stored_id != session_id:
        raise ValueError(f"{path}: stored session ID changed")
    if features.dtype != np.float16 or features.shape != (len(keys), 512):
        raise ValueError(f"{path}: features must be float16 [N,512]")
    if not np.all(np.isfinite(features)):
        raise ValueError(f"{path}: features contain non-finite values")
    transition_matrix(keys, engine, active)
    return FeatureSession(session_id, features, keys, engine, active)


def _run_lookup(engine_frame_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    run_index = np.full(len(engine_frame_idx), -1, dtype=np.int32)
    run_start = np.full(len(engine_frame_idx), -1, dtype=np.int64)
    run_stop = np.full(len(engine_frame_idx), -1, dtype=np.int64)
    for index, (start, stop) in enumerate(contiguous_runs(engine_frame_idx)):
        run_index[start:stop] = index
        run_start[start:stop] = start
        run_stop[start:stop] = stop
    if len(engine_frame_idx) and np.any(run_index < 0):
        raise AssertionError("run lookup did not cover every row")
    return run_index, run_start, run_stop


def construct_oracle_examples(
    session: FeatureSession,
    *,
    split: str,
    width: int,
    halo: int,
    assignment_seed: int,
    block_frames: int,
) -> ConstructionResult:
    """Construct one all-offset-safe crop per exact transition."""

    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    if width < 2 or halo < 1 or block_frames < 1:
        raise ValueError("width, halo, and block_frames must be positive")
    events, base_counts = transition_matrix(
        session.keys, session.engine_frame_idx, session.input_active
    )
    run_index, run_start, run_stop = _run_lookup(session.engine_frame_idx)
    radius = width - 1 + halo
    by_head: dict[int, list[int]] = {head: [] for head in range(len(HEAD_NAMES))}
    excluded_boundary = 0
    excluded_ambiguous = 0
    eligible_by_head = np.zeros(len(HEAD_NAMES), dtype=np.int64)

    for event_index, head_index in zip(*np.nonzero(events), strict=True):
        event_index = int(event_index)
        head_index = int(head_index)
        start = int(run_start[event_index])
        stop = int(run_stop[event_index])
        union_start = event_index - radius
        union_stop = event_index + radius + 1
        if (
            union_start < start
            or union_stop > stop
            or not bool(session.input_active[union_start:union_stop].all())
        ):
            excluded_boundary += 1
            continue
        possible_candidate_start = event_index - (width - 1)
        possible_candidate_stop = event_index + width
        same_head = events[
            possible_candidate_start:possible_candidate_stop, head_index
        ]
        if int(same_head.sum()) != 1:
            excluded_ambiguous += 1
            continue
        by_head[head_index].append(event_index)
        eligible_by_head[head_index] += 1

    examples: list[OracleExample] = []
    offset_counts: dict[str, list[int]] = {}
    for head_index, event_indices in by_head.items():
        if not event_indices:
            offset_counts[HEAD_NAMES[head_index]] = [0] * width
            continue
        rng = np.random.default_rng(
            _stable_seed(
                "oracle-window-offset-v1",
                assignment_seed,
                split,
                session.session_id,
                head_index,
            )
        )
        event_order = rng.permutation(len(event_indices))
        offset_order = rng.permutation(width)
        counts = np.zeros(width, dtype=np.int64)
        for order_index, position in enumerate(event_order):
            event_index = int(event_indices[int(position)])
            offset = int(offset_order[order_index % width])
            crop_start = event_index - offset - halo
            candidate_start = crop_start + halo
            crop_stop = crop_start + width + 2 * halo
            if crop_start < 0 or crop_stop > len(session.features):
                raise AssertionError("all-offset-safe crop escaped the shard")
            if event_index != candidate_start + offset:
                raise AssertionError("assigned target does not match its offset")
            if not bool(session.input_active[crop_start:crop_stop].all()):
                raise AssertionError("assigned crop contains an inactive row")
            if not np.all(
                np.diff(session.engine_frame_idx[crop_start:crop_stop]) == 1
            ):
                raise AssertionError("assigned crop crosses a continuity boundary")
            head_events = events[
                candidate_start : candidate_start + width, head_index
            ]
            if int(head_events.sum()) != 1 or not bool(head_events[offset]):
                raise AssertionError("candidate does not contain one requested event")
            this_run = int(run_index[event_index])
            run_engine_start = int(session.engine_frame_idx[int(run_start[event_index])])
            block = (int(session.engine_frame_idx[event_index]) - run_engine_start) // block_frames
            examples.append(
                OracleExample(
                    split=split,
                    session_id=session.session_id,
                    run_index=this_run,
                    array_index=event_index,
                    engine_frame_idx=int(session.engine_frame_idx[event_index]),
                    head_index=head_index,
                    key_index=head_index % len(KEY_ORDER),
                    event_type_index=head_index // len(KEY_ORDER),
                    offset=offset,
                    crop_start=crop_start,
                    candidate_start=candidate_start,
                    block_id=f"{session.session_id}:run{this_run}:block{block}",
                )
            )
            counts[offset] += 1
        if int(counts.max()) - int(counts.min()) > 1:
            raise AssertionError("offset assignment is not balanced within task")
        offset_counts[HEAD_NAMES[head_index]] = counts.tolist()

    examples.sort(key=lambda row: (row.session_id, row.array_index, row.head_index))
    return ConstructionResult(
        examples=tuple(examples),
        counts={
            **base_counts,
            "excluded_boundary_gap_or_inactive_union": excluded_boundary,
            "excluded_ambiguous_same_head": excluded_ambiguous,
            "eligible": len(examples),
            "eligible_by_head": {
                name: int(eligible_by_head[index])
                for index, name in enumerate(HEAD_NAMES)
            },
            "offset_counts_by_head": offset_counts,
        },
    )


class OracleWindowDataset:
    """In-memory view that exposes no answer-key metadata to the model."""

    def __init__(
        self,
        sessions: Mapping[str, FeatureSession],
        examples: Sequence[OracleExample],
        *,
        width: int,
        halo: int,
    ) -> None:
        if not examples:
            raise ValueError("oracle-window dataset is empty")
        self.sessions = dict(sessions)
        self.examples = tuple(examples)
        self.width = int(width)
        self.halo = int(halo)
        counts = np.bincount(
            [example.head_index for example in examples], minlength=len(HEAD_NAMES)
        )
        present = counts > 0
        self.task_weights = np.zeros(len(HEAD_NAMES), dtype=np.float64)
        self.task_weights[present] = (
            len(examples) / (int(present.sum()) * counts[present])
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        session = self.sessions[example.session_id]
        crop_stop = example.crop_start + self.width + 2 * self.halo
        features = session.features[example.crop_start:crop_stop]
        return {
            "features": torch.from_numpy(features.astype(np.float32, copy=True)),
            "requested_head": torch.tensor(example.head_index, dtype=torch.long),
            "target_offset": torch.tensor(example.offset, dtype=torch.long),
            "task_weight": torch.tensor(
                self.task_weights[example.head_index], dtype=torch.float32
            ),
        }


class OracleWindowEventModel(nn.Module):
    """A radius-eight aligned encoder with fourteen dense event heads."""

    def __init__(
        self,
        *,
        feature_dim: int,
        projection_dim: int,
        temporal_dim: int,
        dilations: Sequence[int],
        width: int,
        halo: int,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.width = int(width)
        self.halo = int(halo)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(projection_dim),
        )
        self.temporal = AlignedTemporalTCN(
            projection_dim,
            temporal_dim,
            dilations=dilations,
            dropout=0.0,
        )
        if self.temporal.receptive_radius != halo:
            raise ValueError(
                "temporal receptive radius must equal the declared halo: "
                f"{self.temporal.receptive_radius} != {halo}"
            )
        self.heads = nn.Linear(temporal_dim, len(HEAD_NAMES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [B,T,D]")
        expected_frames = self.width + 2 * self.halo
        if features.shape[1:] != (expected_frames, self.feature_dim):
            raise ValueError(
                f"features must have shape [B,{expected_frames},{self.feature_dim}]"
            )
        projected = self.projection(features)
        encoded = self.temporal(projected)
        candidate = encoded[:, self.halo : self.halo + self.width]
        return self.heads(candidate)


def requested_logits(
    dense_logits: torch.Tensor, requested_head: torch.Tensor
) -> torch.Tensor:
    if dense_logits.ndim != 3 or dense_logits.shape[-1] != len(HEAD_NAMES):
        raise ValueError("dense logits must have shape [B,W,14]")
    if requested_head.shape != (dense_logits.shape[0],):
        raise ValueError("requested_head must have shape [B]")
    index = requested_head[:, None, None].expand(-1, dense_logits.shape[1], 1)
    return dense_logits.gather(2, index).squeeze(2)


def state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def epoch_order(length: int, seed: int, epoch: int) -> np.ndarray:
    if length < 1 or epoch < 0:
        raise ValueError("length must be positive and epoch non-negative")
    return np.random.default_rng(
        _stable_seed("oracle-window-batch-order-v1", seed, epoch)
    ).permutation(length)


def _collate(
    dataset: OracleWindowDataset, indices: Sequence[int], device: torch.device
) -> dict[str, torch.Tensor]:
    rows = [dataset[int(index)] for index in indices]
    return {
        name: torch.stack([row[name] for row in rows]).to(device)
        for name in rows[0]
    }


def train_arm(
    model: OracleWindowEventModel,
    dataset: OracleWindowDataset,
    *,
    arm: str,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> list[dict[str, float]]:
    """Train one fixed-endpoint arm with the shared deterministic batch order."""

    if arm not in ("conditional_softmax", "dense_bce"):
        raise ValueError("unknown training arm")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    log: list[dict[str, float]] = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_examples = 0
        order = epoch_order(len(dataset), seed, epoch)
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch = _collate(dataset, batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            dense = model(batch["features"])
            logits = requested_logits(dense, batch["requested_head"])
            if arm == "conditional_softmax":
                per_example = F.cross_entropy(
                    logits, batch["target_offset"], reduction="none"
                )
            else:
                target = F.one_hot(
                    batch["target_offset"], num_classes=dataset.width
                ).to(dtype=logits.dtype)
                per_position = F.binary_cross_entropy_with_logits(
                    logits,
                    target,
                    reduction="none",
                    pos_weight=logits.new_tensor(float(dataset.width - 1)),
                )
                per_example = per_position.mean(dim=1)
            weights = batch["task_weight"]
            loss = (per_example * weights).sum() / weights.sum()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_indices)
            total_examples += len(batch_indices)
        log.append({"epoch": float(epoch + 1), "loss": total_loss / total_examples})
    return log


@torch.inference_mode()
def predict_probabilities(
    model: OracleWindowEventModel,
    dataset: OracleWindowDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    for start in range(0, len(dataset), batch_size):
        indices = np.arange(start, min(start + batch_size, len(dataset)))
        batch = _collate(dataset, indices, device)
        logits = requested_logits(
            model(batch["features"]), batch["requested_head"]
        )
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
    result = np.concatenate(probabilities).astype(np.float32, copy=False)
    if result.shape != (len(dataset), dataset.width):
        raise AssertionError("prediction shape changed")
    if not np.all(np.isfinite(result)) or not np.allclose(
        result.sum(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError("model probabilities are not finite and normalized")
    return result


def _external_valid_targets(
    session: FeatureSession, *, past: int, future: int
) -> np.ndarray:
    valid = np.zeros(len(session.features), dtype=bool)
    for start, stop in contiguous_runs(session.engine_frame_idx):
        first = start + past
        last = stop - future
        if last > first:
            valid[first:last] = True
    return valid


@torch.inference_mode()
def predict_current_dense_reference(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    sessions: Mapping[str, FeatureSession],
    examples: Sequence[OracleExample],
    width: int,
    device: torch.device,
    target_chunk: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Restrict the retained Track-3 dense logits to each oracle candidate."""

    observed_sha = sha256_file(checkpoint_path)
    if observed_sha != checkpoint_sha256:
        raise ValueError(
            f"current dense checkpoint hash changed: {observed_sha}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("current dense checkpoint lacks a config")
    model = EventLatchIDM(config)
    state = checkpoint.get("final_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("current dense checkpoint lacks final_state_dict")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    past = int(model.encoder.raw_target_offset)
    future = int(model.encoder.frame_span - 1 - past)
    if past != 189 or future != 192:
        raise ValueError(
            f"current dense support changed: past={past}, future={future}"
        )

    dense_by_session: dict[str, np.ndarray] = {}
    valid_by_session: dict[str, np.ndarray] = {}
    for session_id in sorted({row.session_id for row in examples}):
        session = sessions[session_id]
        valid = _external_valid_targets(session, past=past, future=future)
        logits = np.full((len(session.features), len(HEAD_NAMES)), np.nan, np.float32)
        for run_start, run_stop in contiguous_runs(session.engine_frame_idx):
            valid_start = run_start + past
            valid_stop = run_stop - future
            for target_start in range(valid_start, valid_stop, target_chunk):
                target_stop = min(valid_stop, target_start + target_chunk)
                source_start = target_start - past
                source_stop = target_stop + future
                features = torch.from_numpy(
                    session.features[source_start:source_stop].astype(
                        np.float32, copy=True
                    )
                )[None].to(device)
                outputs = model.forward_segment({"features": features})
                onset = outputs["onset_logits"][0].cpu().numpy()
                release = outputs["release_logits"][0].cpu().numpy()
                block = np.concatenate((onset, release), axis=1).astype(np.float32)
                if block.shape != (target_stop - target_start, len(HEAD_NAMES)):
                    raise AssertionError("current dense segment output changed")
                logits[target_start:target_stop] = block
        if not np.all(np.isfinite(logits[valid])):
            raise ValueError("current dense inference left a valid target unscored")
        dense_by_session[session_id] = logits
        valid_by_session[session_id] = valid

    probabilities = np.full((len(examples), width), np.nan, dtype=np.float32)
    support = np.zeros(len(examples), dtype=bool)
    for index, example in enumerate(examples):
        positions = np.arange(example.candidate_start, example.candidate_start + width)
        valid = valid_by_session[example.session_id]
        if positions[-1] >= len(valid) or not bool(valid[positions].all()):
            continue
        logits = dense_by_session[example.session_id][positions, example.head_index]
        shifted = logits.astype(np.float64) - float(np.max(logits))
        probability = np.exp(shifted)
        probability /= probability.sum()
        probabilities[index] = probability.astype(np.float32)
        support[index] = True
    if np.any(support) and not np.allclose(
        probabilities[support].sum(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError("current dense probabilities are not normalized")
    return probabilities, support, {
        "checkpoint_sha256": observed_sha,
        "past_support": past,
        "future_support": future,
        "common_examples": int(support.sum()),
        "total_examples": len(examples),
        "role": "unmatched historical Track-3 reference; excluded from gate",
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    if os.path.lexists(path):
        raise ValueError(f"refusing to overwrite prediction sidecar: {path}")
    temporary = path.with_suffix(".tmp.npz")
    if os.path.lexists(temporary):
        raise ValueError(f"stale prediction temporary exists: {temporary}")
    np.savez(temporary, **arrays)
    with np.load(temporary, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise ValueError("serialized prediction inventory changed")
        for name, expected in arrays.items():
            observed = np.asarray(archive[name])
            equal = (
                np.array_equal(observed, expected, equal_nan=True)
                if np.issubdtype(expected.dtype, np.floating)
                else np.array_equal(observed, expected)
            )
            if observed.dtype != expected.dtype or not equal:
                raise ValueError(f"serialized prediction array changed: {name}")
    temporary.replace(path)


def _validate_feature_receipt(
    receipt_path: Path, feature_root: Path, expected_sha256: str
) -> dict[str, Any]:
    observed_sha = sha256_file(receipt_path)
    if observed_sha != expected_sha256:
        raise ValueError(f"feature receipt hash changed: {observed_sha}")
    receipt = _json(receipt_path)
    if receipt.get("status") != "complete":
        raise ValueError("feature receipt is not complete")
    if Path(str(receipt.get("published_output"))).resolve() != feature_root.resolve():
        raise ValueError("feature receipt points to a different output")
    checks = receipt.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
        raise ValueError("feature receipt did not pass every check")
    return receipt


def _validate_implementation(
    config: Mapping[str, Any], *, repo: Path
) -> dict[str, Any]:
    declaration = config.get("implementation")
    if not isinstance(declaration, Mapping):
        raise ValueError("decision config lacks implementation hashes")
    expected = declaration.get("sha256")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("decision config has no implementation file hashes")
    observed: dict[str, str] = {}
    for relative, expected_sha in expected.items():
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise ValueError("implementation hashes must map paths to strings")
        path = repo / relative
        if not path.is_file():
            raise ValueError(f"missing frozen implementation file: {relative}")
        observed_sha = sha256_file(path)
        if observed_sha != expected_sha:
            raise ValueError(
                f"implementation bytes changed for {relative}: {observed_sha}"
            )
        observed[relative] = observed_sha
    git_head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_head_at_execution": git_head,
        "git_head_at_freeze": declaration.get("git_head_at_freeze"),
        "relevant_file_sha256": observed,
        "authority": "exact relevant working bytes; no commit created for this study",
    }


def _prepare_data(
    *, feature_root: Path, config: Mapping[str, Any]
) -> tuple[
    dict[str, FeatureSession],
    tuple[OracleExample, ...],
    tuple[OracleExample, ...],
    dict[str, Any],
]:
    dataset_config = config["dataset"]
    train_ids = tuple(dataset_config["train_sessions"])
    val_ids = tuple(dataset_config["validation_sessions"])
    if train_ids != DEFAULT_TRAIN_IDS or val_ids != DEFAULT_VAL_IDS:
        raise ValueError("own-v3 split membership changed")
    on_disk_train = tuple(read_session_ids(feature_root / "train_sessions.txt"))
    on_disk_val = tuple(read_session_ids(feature_root / "val_sessions.txt"))
    if on_disk_train != train_ids or on_disk_val != val_ids:
        raise ValueError("feature split bytes do not name the frozen sessions")
    validate_splits(train_ids, val_ids)
    if FORBIDDEN_SESSION_IDS.intersection((*train_ids, *val_ids)):
        raise ValueError("a forbidden session entered the split")
    sessions = {
        session_id: load_feature_session(feature_root, session_id)
        for session_id in (*train_ids, *val_ids)
    }
    width = int(dataset_config["candidate_width"])
    halo = int(dataset_config["context_halo"])
    assignment_seed = int(dataset_config["offset_assignment_seed"])
    block_frames = int(config["evaluation"]["bootstrap_block_frames"])
    train_examples: list[OracleExample] = []
    val_examples: list[OracleExample] = []
    construction: dict[str, Any] = {"train": {}, "validation": {}}
    for split, ids, destination in (
        ("train", train_ids, train_examples),
        ("validation", val_ids, val_examples),
    ):
        for session_id in ids:
            result = construct_oracle_examples(
                sessions[session_id],
                split=split,
                width=width,
                halo=halo,
                assignment_seed=assignment_seed,
                block_frames=block_frames,
            )
            destination.extend(result.examples)
            construction[split][session_id] = result.counts
    return sessions, tuple(train_examples), tuple(val_examples), construction


def _example_arrays(examples: Sequence[OracleExample]) -> dict[str, np.ndarray]:
    return {
        "session_id": np.asarray([row.session_id for row in examples]),
        "run_index": np.asarray([row.run_index for row in examples], np.int32),
        "array_index": np.asarray([row.array_index for row in examples], np.int64),
        "engine_frame_idx": np.asarray(
            [row.engine_frame_idx for row in examples], np.int64
        ),
        "head_index": np.asarray([row.head_index for row in examples], np.int16),
        "key_index": np.asarray([row.key_index for row in examples], np.int8),
        "event_type_index": np.asarray(
            [row.event_type_index for row in examples], np.int8
        ),
        "true_offset": np.asarray([row.offset for row in examples], np.int8),
        "crop_start": np.asarray([row.crop_start for row in examples], np.int64),
        "block_id": np.asarray([row.block_id for row in examples]),
    }


def _dataset_manifest(
    *,
    sessions: Mapping[str, FeatureSession],
    train_examples: Sequence[OracleExample],
    val_examples: Sequence[OracleExample],
    construction: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    train_counts_by_session_head: dict[str, dict[str, int]] = {}
    for session_id in config["dataset"]["train_sessions"]:
        rows = [row for row in train_examples if row.session_id == session_id]
        counts = np.bincount(
            [row.head_index for row in rows], minlength=len(HEAD_NAMES)
        )
        train_counts_by_session_head[session_id] = {
            name: int(counts[index]) for index, name in enumerate(HEAD_NAMES)
        }
    val_counts = np.bincount(
        [row.head_index for row in val_examples], minlength=len(HEAD_NAMES)
    )
    offset_counts = np.bincount(
        [row.offset for row in val_examples],
        minlength=int(config["dataset"]["candidate_width"]),
    )
    return {
        "schema_version": "madeleine.oracle-window-dataset.v1",
        "head_names": list(HEAD_NAMES),
        "sessions": {
            session_id: {
                "rows": len(session.features),
                "features_sha256": _array_sha256(session.features),
                "keys_sha256": _array_sha256(session.keys),
                "engine_frame_idx_sha256": _array_sha256(
                    session.engine_frame_idx
                ),
                "input_active_sha256": _array_sha256(session.input_active),
            }
            for session_id, session in sessions.items()
        },
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "train_counts_by_session_head": train_counts_by_session_head,
        "val_counts_by_head": {
            name: int(val_counts[index]) for index, name in enumerate(HEAD_NAMES)
        },
        "validation_offset_counts": offset_counts.tolist(),
        "validation_block_count": len({row.block_id for row in val_examples}),
        "construction": construction,
        "construction_policy": config["dataset"]["construction_policy"],
    }


def run_experiment(
    *,
    feature_root: Path,
    feature_receipt_path: Path,
    config_path: Path,
    output: Path,
    current_checkpoint: Path,
    device_name: str,
    epoch_override: int | None = None,
    seed_override: int | None = None,
) -> Path:
    if os.path.lexists(output):
        raise ValueError(f"refusing to overwrite experiment output: {output}")
    config = _json(config_path)
    if config.get("status") != "preregistered_before_validation_inference":
        raise ValueError("decision config is not frozen for inference")
    feature_root = feature_root.resolve()
    repo = config_path.resolve().parents[2]
    implementation_receipt = _validate_implementation(config, repo=repo)
    expected_receipt_sha = str(config["dataset"]["feature_receipt_sha256"])
    feature_receipt = _validate_feature_receipt(
        feature_receipt_path, feature_root, expected_receipt_sha
    )
    sessions, train_examples, val_examples, construction = _prepare_data(
        feature_root=feature_root, config=config
    )
    manifest = _dataset_manifest(
        sessions=sessions,
        train_examples=train_examples,
        val_examples=val_examples,
        construction=construction,
        config=config,
    )
    width = int(config["dataset"]["candidate_width"])
    halo = int(config["dataset"]["context_halo"])
    train_dataset = OracleWindowDataset(
        sessions, train_examples, width=width, halo=halo
    )
    val_dataset = OracleWindowDataset(sessions, val_examples, width=width, halo=halo)

    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    primary_seed = int(config["training"]["seed"])
    allowed_seeds = {
        primary_seed,
        *(int(value) for value in config["training"]["confirmation_seeds"]),
    }
    seed = primary_seed if seed_override is None else int(seed_override)
    if seed not in allowed_seeds:
        raise ValueError(f"seed {seed} is not preregistered: {sorted(allowed_seeds)}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=(device.type != "cuda"))
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model_config = config["model"]
    initial_model = OracleWindowEventModel(
        feature_dim=int(model_config["feature_dim"]),
        projection_dim=int(model_config["projection_dim"]),
        temporal_dim=int(model_config["temporal_dim"]),
        dilations=tuple(model_config["tcn_dilations"]),
        width=width,
        halo=halo,
    )
    initial_hash = state_dict_sha256(initial_model)
    conditional = copy.deepcopy(initial_model).to(device)
    dense = copy.deepcopy(initial_model).to(device)
    if state_dict_sha256(conditional) != initial_hash or state_dict_sha256(dense) != initial_hash:
        raise AssertionError("matched arms do not share exact initialization")

    configured_epochs = int(config["training"]["epochs"])
    epochs = configured_epochs if epoch_override is None else int(epoch_override)
    if epochs < 1 or epochs > configured_epochs:
        raise ValueError("epoch override must lie within the frozen endpoint")
    common_train = {
        "device": device,
        "seed": seed,
        "epochs": epochs,
        "batch_size": int(config["training"]["batch_size"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "weight_decay": float(config["training"]["weight_decay"]),
    }
    conditional_log = train_arm(
        conditional,
        train_dataset,
        arm="conditional_softmax",
        **common_train,
    )
    dense_log = train_arm(
        dense,
        train_dataset,
        arm="dense_bce",
        **common_train,
    )
    eval_batch = int(config["training"]["eval_batch_size"])
    conditional_prob = predict_probabilities(
        conditional, val_dataset, device=device, batch_size=eval_batch
    )
    dense_prob = predict_probabilities(
        dense, val_dataset, device=device, batch_size=eval_batch
    )
    external_config = config["current_dense_reference"]
    current_prob, current_support, current_receipt = predict_current_dense_reference(
        checkpoint_path=current_checkpoint,
        checkpoint_sha256=str(external_config["checkpoint_sha256"]),
        sessions=sessions,
        examples=val_examples,
        width=width,
        device=device,
        target_chunk=int(external_config["target_chunk"]),
    )

    output.mkdir(parents=True, exist_ok=False)
    config_sha = sha256_file(config_path)
    dataset_path = output / "dataset_manifest.json"
    _write_json(dataset_path, manifest)
    checkpoint_payload = {
        "config_sha256": config_sha,
        "feature_receipt_sha256": expected_receipt_sha,
        "initial_state_sha256": initial_hash,
        "seed": seed,
        "epochs": epochs,
    }
    torch.save(
        {
            **checkpoint_payload,
            "arm": "conditional_softmax",
            "model_state_dict": conditional.state_dict(),
        },
        output / "conditional_model.pt",
    )
    torch.save(
        {
            **checkpoint_payload,
            "arm": "dense_bce",
            "model_state_dict": dense.state_dict(),
        },
        output / "dense_model.pt",
    )
    _write_json(
        output / "training_log.json",
        {
            "conditional_softmax": conditional_log,
            "dense_bce": dense_log,
            "fixed_final_epoch": epochs,
            "configured_final_epoch": configured_epochs,
            "validation_used_for_training_or_selection": False,
        },
    )
    sidecar_path = output / "predictions.npz"
    _write_npz_atomic(
        sidecar_path,
        **_example_arrays(val_examples),
        conditional_prob=conditional_prob,
        dense_prob=dense_prob,
        current_dense_prob=current_prob,
        current_dense_support=current_support,
    )
    run_receipt = {
        "schema_version": "madeleine.oracle-window-run.v1",
        "status": "predictions_complete_unscored",
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha,
        "feature_receipt_path": str(feature_receipt_path.resolve()),
        "feature_receipt_sha256": expected_receipt_sha,
        "feature_content_sha256": feature_receipt["content_sha256"],
        "dataset_manifest_sha256": sha256_file(dataset_path),
        "prediction_sidecar_sha256": sha256_file(sidecar_path),
        "initial_state_sha256": initial_hash,
        "matched_initialization": True,
        "matched_batch_order": True,
        "final_weights_only": True,
        "epochs": epochs,
        "configured_epochs": configured_epochs,
        "seed": seed,
        "device": str(device),
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "current_dense_reference": current_receipt,
        "implementation": implementation_receipt,
    }
    _write_json(output / "run_receipt.json", run_receipt)
    return sidecar_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--feature-receipt", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--current-checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sidecar = run_experiment(
        feature_root=args.features,
        feature_receipt_path=args.feature_receipt,
        config_path=args.config,
        output=args.out,
        current_checkpoint=args.current_checkpoint,
        device_name=args.device,
        epoch_override=args.epochs,
        seed_override=args.seed,
    )
    print(json.dumps({"status": "predictions_complete", "sidecar": str(sidecar)}))


if __name__ == "__main__":
    main()
