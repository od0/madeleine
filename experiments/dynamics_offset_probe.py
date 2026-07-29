"""Measure when Celeste input transitions become visible in RGB frames.

For every requested offset ``o``, a linear probe receives the frame pair
``(frame[t + o - 1], frame[t + o])`` and predicts the engine-truth onset and
release at ``t``.  Candidate construction is run- and session-bounded: the
target transition, the observed pair, and every frame between them must belong
to one strictly consecutive engine-frame run.

Two frozen ImageNet ResNet-18 representations are compared:

``pooled_same_frame``
    The globally pooled feature for the second observed frame only.

``pooled_pair``
    Global features for the two frames, their signed difference, and their
    absolute difference.

``spatial_motion``
    Current-frame, signed-difference, and absolute-difference features formed
    on the layer-3 8x8 map *before* global pooling, then averaged to a 4x4
grid.  This is a cheap control for motion evidence discarded downstream.

``spatial_same_frame``
    The current layer-3 4x4 feature alone, paired with ``spatial_motion`` in
    the same way that ``pooled_same_frame`` is paired with ``pooled_pair``.

All offsets use the identical target rows.  With the candidate -4..+16 sweep
plus the +17 adjacent-confirmation offset, every row from t-5 through t+17
must be consecutive and active.  The probe
uses capped inverse-prevalence BCE weights (no event resampling), a fixed
optimization budget, and a fixed 0.5 diagnostic threshold.  AP is always
measured on the validation split at its natural prevalence; validation data is
never used for fitting or model selection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

# Required by deterministic CUDA matrix multiplication.  It must be set before
# the first CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.models import ResNet18_Weights, resnet18

from badeline.train import contiguous_runs
from data.schema import KEY_ORDER


DEFAULT_OFFSETS = tuple(range(-4, 18))
FEATURE_VARIANTS = (
    "pooled_same_frame",
    "pooled_pair",
    "spatial_same_frame",
    "spatial_motion",
)
OUTPUT_NAMES = tuple(
    [f"{key}:onset" for key in KEY_ORDER]
    + [f"{key}:release" for key in KEY_ORDER]
)
KNOWN_EMBARGOED_SESSION_IDS = frozenset({"rec_20260727_220000_test"})
EXPECTED_BACKBONE_CONTRACT = {
    "architecture": "torchvision resnet18",
    "weights": "ResNet18_Weights.IMAGENET1K_V1",
    "weights_url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "weights_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    "mode": "frozen eval",
}


@dataclass(frozen=True)
class RawSession:
    """One validated engine-truth RGB shard."""

    session_id: str
    path: Path
    shard_sha256: str
    frames: np.ndarray
    keys: np.ndarray
    engine_frame_idx: np.ndarray
    input_active: np.ndarray


@dataclass(frozen=True)
class EncodedSession:
    """Frozen frame representations retained in host memory."""

    session_id: str
    path: Path
    shard_sha256: str
    pooled: np.ndarray
    coarse_spatial: np.ndarray
    keys: np.ndarray
    engine_frame_idx: np.ndarray
    input_active: np.ndarray


@dataclass(frozen=True)
class EncodedCorpus:
    """Concatenated representations plus immutable session boundaries."""

    session_ids: tuple[str, ...]
    shard_records: tuple[dict[str, Any], ...]
    pooled: np.ndarray
    coarse_spatial: np.ndarray
    keys: np.ndarray
    input_active: np.ndarray
    engine_frame_idx: tuple[np.ndarray, ...]
    starts: tuple[int, ...]


@dataclass(frozen=True)
class CandidateSet:
    """Global array indices for one offset's observed and target pairs."""

    observed_previous: np.ndarray
    observed_current: np.ndarray
    target_previous: np.ndarray
    target_current: np.ndarray
    per_session: dict[str, int]

    def __len__(self) -> int:
        return int(len(self.target_current))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_array_sha256(array: np.ndarray) -> str:
    """Hash shape, dtype, and little-endian contiguous bytes."""

    value = np.asarray(array)
    dtype = value.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(value.astype(dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def read_explicit_session_ids(path: Path) -> list[str]:
    ids = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not ids:
        raise ValueError(f"empty session list: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate session ID in list: {path}")
    return ids


def require_canonical_session_list_bytes(
    path: Path,
    expected_ids: Sequence[str],
) -> None:
    expected = ("\n".join(expected_ids) + "\n").encode("utf-8")
    observed = Path(path).read_bytes()
    if observed != expected:
        raise ValueError(
            f"{path}: split-list bytes must exactly equal one preregistered "
            "session ID per line with a final newline"
        )


def reject_forbidden_sessions(session_ids: Sequence[str]) -> None:
    """Reject known sealed-test names before any shard path is opened."""

    forbidden = [
        session_id
        for session_id in session_ids
        if session_id in KNOWN_EMBARGOED_SESSION_IDS
        or "untouched" in session_id.casefold()
    ]
    if forbidden:
        raise ValueError(
            "embargoed/untouched sessions are forbidden for this diagnostic: "
            + ", ".join(forbidden)
        )


def validate_split_ids(train_ids: Sequence[str], validation_ids: Sequence[str]) -> None:
    reject_forbidden_sessions([*train_ids, *validation_ids])
    overlap = sorted(set(train_ids).intersection(validation_ids))
    if overlap:
        raise ValueError(f"train/validation overlap: {', '.join(overlap)}")


def load_and_validate_contract(
    path: Path,
    *,
    data_dir: Path,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    offsets: Sequence[int],
    variants: Sequence[str],
    seeds: Sequence[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_train_samples: int,
    positive_weight_cap: float,
) -> dict[str, Any]:
    """Bind a run to the committed preregistration before shard access."""

    contract_path = Path(path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != "madeleine.dynamics-offset-probe.v1":
        raise ValueError("unsupported dynamics-offset probe contract schema")
    data = contract.get("data", {})
    fit = contract.get("fit", {})
    expected = {
        "train sessions": (list(data.get("train_sessions", [])), list(train_ids)),
        "validation sessions": (
            list(data.get("validation_sessions", [])),
            list(validation_ids),
        ),
        "offsets": (list(contract.get("offsets_native_frames", [])), list(offsets)),
        "seeds": (list(fit.get("random_seeds", [])), list(seeds)),
        "epochs": (fit.get("epochs"), epochs),
        "batch size": (fit.get("batch_size"), batch_size),
        "learning rate": (fit.get("learning_rate"), learning_rate),
        "weight decay": (fit.get("weight_decay"), weight_decay),
        "key order": (list(contract.get("targets", {}).get("key_order", [])), KEY_ORDER),
        "data root": (data.get("root"), str(Path(data_dir))),
    }
    mismatch = [
        f"{name}: contract={contract_value!r}, run={run_value!r}"
        for name, (contract_value, run_value) in expected.items()
        if contract_value != run_value
    ]
    if mismatch:
        raise ValueError("run does not match preregistered contract: " + "; ".join(mismatch))
    if tuple(variants) != FEATURE_VARIANTS:
        raise ValueError(
            "run does not match preregistered contract: all four feature "
            f"variants are required in order {FEATURE_VARIANTS}"
        )
    if max_train_samples != 0:
        raise ValueError(
            "run does not match preregistered contract: max_train_samples must be 0"
        )
    if positive_weight_cap != 50.0:
        raise ValueError(
            "run does not match preregistered contract: positive_weight_cap must be 50"
        )
    low_relative = min(-1, min(int(offset) - 1 for offset in offsets))
    high_relative = max(0, max(int(offset) for offset in offsets))
    if (low_relative, high_relative) != (-5, 17):
        raise ValueError(
            "run does not match preregistered contract: common support must "
            "span t-5 through t+17"
        )
    if ResNet18_Weights.DEFAULT is not ResNet18_Weights.IMAGENET1K_V1:
        raise RuntimeError("torchvision ResNet18 default weight identity changed")
    if contract.get("probe_surfaces", {}).get("backbone") != EXPECTED_BACKBONE_CONTRACT:
        raise ValueError("contract does not bind the exact pretrained backbone weights")
    embargo = contract.get("embargo", {}).get("sealed_untouched_session")
    if embargo not in KNOWN_EMBARGOED_SESSION_IDS:
        raise ValueError("contract does not preserve the known untouched-session embargo")
    train_hashes = data.get("train_shard_sha256")
    validation_hashes = data.get("validation_shard_sha256")
    if set(train_hashes or {}) != set(train_ids):
        raise ValueError("contract train shard hashes do not bind every train session")
    if set(validation_hashes or {}) != set(validation_ids):
        raise ValueError(
            "contract validation shard hashes do not bind every validation session"
        )
    return contract


def verify_cached_backbone_weights(contract: dict[str, Any]) -> dict[str, str]:
    """Verify the exact bytes torchvision loaded before any shard is opened."""

    backbone = contract["probe_surfaces"]["backbone"]
    filename = backbone["weights_url"].rsplit("/", 1)[-1]
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"torchvision did not materialize expected weight file: {weight_path}"
        )
    observed = sha256_file(weight_path)
    if observed != backbone["weights_sha256"]:
        raise ValueError(
            f"pretrained weight SHA-256 {observed} does not match contract "
            f"{backbone['weights_sha256']}"
        )
    return {
        "enum": backbone["weights"],
        "url": backbone["weights_url"],
        "path": str(weight_path),
        "sha256": observed,
    }


def load_rgb_session(data_dir: Path, session_id: str) -> RawSession:
    """Load exactly one named shard and enforce the probe's frozen schema."""

    reject_forbidden_sessions([session_id])
    shard = Path(data_dir) / f"{session_id}.npz"
    if not shard.is_file():
        raise FileNotFoundError(f"missing requested session shard: {shard}")
    shard_hash = sha256_file(shard)
    with np.load(shard, allow_pickle=False) as archive:
        required = {"frames", "keys", "engine_frame_idx", "input_active"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{shard}: missing arrays: {sorted(missing)}")
        frames = np.asarray(archive["frames"])
        keys = np.asarray(archive["keys"])
        engine_frame_idx = np.asarray(archive["engine_frame_idx"])
        input_active = np.asarray(archive["input_active"])
        if "session_id" in archive.files:
            stored = np.asarray(archive["session_id"])
            if stored.size != 1 or str(stored.reshape(()).item()) != session_id:
                raise ValueError(f"{shard}: stored session_id does not match filename")

    if frames.dtype != np.uint8 or frames.ndim != 4:
        raise ValueError(f"{shard}: frames must be uint8 [N,128,128,3]")
    if frames.shape[1:] != (128, 128, 3):
        raise ValueError(f"{shard}: frames must have shape [N,128,128,3]")
    if keys.dtype != np.uint8 or keys.shape != (len(frames), len(KEY_ORDER)):
        raise ValueError(f"{shard}: keys must be uint8 [N,{len(KEY_ORDER)}]")
    if not np.all(np.isin(keys, (0, 1))):
        raise ValueError(f"{shard}: keys must be binary")
    if engine_frame_idx.dtype != np.int64 or engine_frame_idx.shape != (len(frames),):
        raise ValueError(f"{shard}: engine_frame_idx must be int64 [N]")
    if input_active.dtype != np.uint8 or input_active.shape != (len(frames),):
        raise ValueError(f"{shard}: input_active must be uint8 [N]")
    if not np.all(np.isin(input_active, (0, 1))):
        raise ValueError(f"{shard}: input_active must be binary")
    if len(frames) < 2:
        raise ValueError(f"{shard}: at least two frames are required")
    return RawSession(
        session_id=session_id,
        path=shard,
        shard_sha256=shard_hash,
        frames=frames,
        keys=keys,
        engine_frame_idx=engine_frame_idx,
        input_active=input_active,
    )


def candidate_indices_for_run(
    start: int,
    end: int,
    offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return local candidate indices for one contiguous half-open run."""

    if start < 0 or end < start:
        raise ValueError("invalid run bounds")
    # A target transition uses t-1,t.  The observed pair uses t+o-1,t+o.
    # Constraining all four indices to this run also guarantees the complete
    # interval between cause and effect does not cross a gap.
    lower = max(start + 1, start - offset + 1)
    upper = min(end - 1, end - offset - 1)
    if upper < lower:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy(), empty.copy()
    target_current = np.arange(lower, upper + 1, dtype=np.int64)
    observed_current = target_current + int(offset)
    return (
        observed_current - 1,
        observed_current,
        target_current - 1,
        target_current,
    )


def common_target_indices_for_run(
    start: int,
    end: int,
    offsets: Sequence[int],
    input_active: np.ndarray,
) -> np.ndarray:
    """Return targets sharing one active, contiguous support across offsets."""

    if not offsets:
        raise ValueError("offset set must not be empty")
    active = np.asarray(input_active, dtype=np.uint8)
    if active.ndim != 1 or end > len(active):
        raise ValueError("input_active does not cover the requested run")
    low_relative = min(-1, min(int(offset) - 1 for offset in offsets))
    high_relative = max(0, max(int(offset) for offset in offsets))
    lower = start - low_relative
    upper = end - 1 - high_relative
    if upper < lower:
        return np.empty(0, dtype=np.int64)
    targets = np.arange(lower, upper + 1, dtype=np.int64)
    inactive = (active == 0).astype(np.int64)
    prefix = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(inactive)))
    window_start = targets + low_relative
    window_end = targets + high_relative + 1
    active_window = (prefix[window_end] - prefix[window_start]) == 0
    return targets[active_window]


def build_candidates(
    corpus: EncodedCorpus,
    offset: int,
    *,
    common_offsets: Sequence[int] | None = None,
) -> CandidateSet:
    support_offsets = tuple(common_offsets) if common_offsets is not None else (offset,)
    if offset not in support_offsets:
        raise ValueError("observed offset must be in common support offsets")
    observed_previous: list[np.ndarray] = []
    observed_current: list[np.ndarray] = []
    target_previous: list[np.ndarray] = []
    target_current: list[np.ndarray] = []
    per_session: dict[str, int] = {}

    for session_id, base, frame_idx in zip(
        corpus.session_ids,
        corpus.starts,
        corpus.engine_frame_idx,
        strict=True,
    ):
        count = 0
        active = corpus.input_active[base : base + len(frame_idx)]
        for start, end in contiguous_runs(frame_idx):
            tc = common_target_indices_for_run(
                start, end, support_offsets, active
            )
            if len(tc):
                oc = tc + int(offset)
                op = oc - 1
                tp = tc - 1
                observed_previous.append(op + base)
                observed_current.append(oc + base)
                target_previous.append(tp + base)
                target_current.append(tc + base)
                count += len(tc)
        per_session[session_id] = int(count)

    def combine(parts: list[np.ndarray]) -> np.ndarray:
        return (
            np.concatenate(parts).astype(np.int64, copy=False)
            if parts
            else np.empty(0, dtype=np.int64)
        )

    return CandidateSet(
        observed_previous=combine(observed_previous),
        observed_current=combine(observed_current),
        target_previous=combine(target_previous),
        target_current=combine(target_current),
        per_session=per_session,
    )


def transition_targets(keys: np.ndarray, candidates: CandidateSet) -> np.ndarray:
    previous = np.asarray(keys[candidates.target_previous], dtype=bool)
    current = np.asarray(keys[candidates.target_current], dtype=bool)
    onset = np.logical_and(~previous, current)
    release = np.logical_and(previous, ~current)
    return np.concatenate((onset, release), axis=1).astype(np.float32)


def per_session_target_support(
    corpus: EncodedCorpus,
    candidates: CandidateSet,
) -> dict[str, Any]:
    targets = transition_targets(corpus.keys, candidates)
    result: dict[str, Any] = {}
    start = 0
    for session_id in corpus.session_ids:
        count = int(candidates.per_session[session_id])
        result[session_id] = _support_record(targets[start : start + count])
        start += count
    if start != len(targets):
        raise AssertionError("per-session candidate support does not sum to total")
    return result


def target_identity_arrays(
    corpus: EncodedCorpus,
    candidates: CandidateSet,
) -> dict[str, np.ndarray]:
    """Stable session/run/engine identifiers aligned to candidate targets."""

    engine = np.concatenate(corpus.engine_frame_idx).astype(np.int64, copy=False)
    session_index = np.empty(len(corpus.keys), dtype=np.int32)
    run_id = np.empty(len(corpus.keys), dtype=np.int64)
    next_run_id = 0
    for index, (base, frame_idx) in enumerate(
        zip(corpus.starts, corpus.engine_frame_idx, strict=True)
    ):
        session_index[base : base + len(frame_idx)] = index
        for start, end in contiguous_runs(frame_idx):
            run_id[base + start : base + end] = next_run_id
            next_run_id += 1
    target = candidates.target_current
    return {
        "target_global_index": target.astype(np.int64, copy=False),
        "target_engine_frame_idx": engine[target],
        "target_session_index": session_index[target],
        "target_run_id": run_id[target],
        "session_ids": np.asarray(corpus.session_ids),
        "session_lengths": np.asarray(
            [candidates.per_session[session_id] for session_id in corpus.session_ids],
            dtype=np.int64,
        ),
    }


def target_key_context(
    corpus: EncodedCorpus,
    candidates: CandidateSet,
    *,
    low_relative: int = -5,
    high_relative: int = 17,
) -> np.ndarray:
    relative = np.arange(low_relative, high_relative + 1, dtype=np.int64)
    indices = candidates.target_current[:, None] + relative[None, :]
    if np.any(indices < 0) or np.any(indices >= len(corpus.keys)):
        raise ValueError("target key context lies outside corpus arrays")
    return corpus.keys[indices].astype(np.uint8, copy=False)


def candidate_session_slices(
    corpus: EncodedCorpus,
    candidates: CandidateSet,
) -> dict[str, slice]:
    result: dict[str, slice] = {}
    start = 0
    for session_id in corpus.session_ids:
        end = start + int(candidates.per_session[session_id])
        result[session_id] = slice(start, end)
        start = end
    if start != len(candidates):
        raise AssertionError("candidate session slices do not cover support")
    return result


class FrozenResNet18SpatialFeatures(nn.Module):
    """Frozen ImageNet global layer-4 and coarse layer-3 representations."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        for module in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            module.requires_grad_(False)
            module.eval()
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def train(self, mode: bool = True) -> "FrozenResNet18SpatialFeatures":
        super().train(mode)
        for module in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            module.eval()
        return self

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (frames - self.image_mean) / self.image_std
        layer2 = self.layer2(self.layer1(self.stem(normalized)))
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        if layer3.shape[1:] != (256, 8, 8):
            raise ValueError(
                "128px ResNet layer-3 map must be [256,8,8], got "
                f"{tuple(layer3.shape[1:])}"
            )
        if layer4.shape[1:] != (512, 4, 4):
            raise ValueError(
                "128px ResNet layer-4 map must be [512,4,4], got "
                f"{tuple(layer4.shape[1:])}"
            )
        pooled = layer4.mean(dim=(-2, -1))
        coarse = F.avg_pool2d(layer3, kernel_size=2, stride=2)
        return pooled, coarse


@torch.inference_mode()
def encode_raw_session(
    session: RawSession,
    encoder: FrozenResNet18SpatialFeatures,
    *,
    device: torch.device,
    batch_size: int,
) -> EncodedSession:
    if batch_size < 1:
        raise ValueError("encode batch size must be positive")
    pooled = np.empty((len(session.frames), 512), dtype=np.float16)
    coarse = np.empty((len(session.frames), 256, 4, 4), dtype=np.float16)
    for start in range(0, len(session.frames), batch_size):
        end = min(start + batch_size, len(session.frames))
        batch = (
            torch.from_numpy(session.frames[start:end].copy())
            .permute(0, 3, 1, 2)
            .to(device=device, dtype=torch.float32)
            .div_(255.0)
        )
        batch_pooled, batch_coarse = encoder(batch)
        pooled[start:end] = batch_pooled.to(torch.float16).cpu().numpy()
        coarse[start:end] = batch_coarse.to(torch.float16).cpu().numpy()
    return EncodedSession(
        session_id=session.session_id,
        path=session.path,
        shard_sha256=session.shard_sha256,
        pooled=pooled,
        coarse_spatial=coarse,
        keys=session.keys,
        engine_frame_idx=session.engine_frame_idx,
        input_active=session.input_active,
    )


def concatenate_encoded_sessions(sessions: Sequence[EncodedSession]) -> EncodedCorpus:
    if not sessions:
        raise ValueError("at least one encoded session is required")
    starts: list[int] = []
    position = 0
    for session in sessions:
        starts.append(position)
        position += len(session.keys)
    return EncodedCorpus(
        session_ids=tuple(session.session_id for session in sessions),
        shard_records=tuple(
            {
                "session_id": session.session_id,
                "path": str(session.path),
                "sha256": session.shard_sha256,
                "frames": int(len(session.keys)),
                "contiguous_runs": int(len(contiguous_runs(session.engine_frame_idx))),
            }
            for session in sessions
        ),
        pooled=np.concatenate([session.pooled for session in sessions]),
        coarse_spatial=np.concatenate(
            [session.coarse_spatial for session in sessions]
        ),
        keys=np.concatenate([session.keys for session in sessions]),
        input_active=np.concatenate([session.input_active for session in sessions]),
        engine_frame_idx=tuple(session.engine_frame_idx for session in sessions),
        starts=tuple(starts),
    )


def pair_features(
    corpus: EncodedCorpus,
    candidates: CandidateSet,
    selection: np.ndarray,
    *,
    variant: str,
    device: torch.device,
) -> torch.Tensor:
    """Materialize one minibatch's unstandardized representation."""

    if variant not in FEATURE_VARIANTS:
        raise ValueError(f"unknown feature variant: {variant}")
    chosen = np.asarray(selection, dtype=np.int64)
    previous_idx = candidates.observed_previous[chosen]
    current_idx = candidates.observed_current[chosen]
    previous = torch.from_numpy(corpus.pooled[previous_idx].astype(np.float32)).to(device)
    current = torch.from_numpy(corpus.pooled[current_idx].astype(np.float32)).to(device)
    if variant == "pooled_same_frame":
        return current
    signed_delta = current - previous
    if variant == "pooled_pair":
        return torch.cat((previous, current, signed_delta, signed_delta.abs()), dim=1)
    if variant in ("spatial_same_frame", "spatial_motion"):
        previous_spatial = torch.from_numpy(
            corpus.coarse_spatial[previous_idx].reshape(len(chosen), -1).astype(np.float32)
        ).to(device)
        current_spatial = torch.from_numpy(
            corpus.coarse_spatial[current_idx].reshape(len(chosen), -1).astype(np.float32)
        ).to(device)
        if variant == "spatial_same_frame":
            return current_spatial
        spatial_delta = current_spatial - previous_spatial
        return torch.cat(
            (current_spatial, spatial_delta, spatial_delta.abs()), dim=1
        )
    raise AssertionError("unreachable feature variant")


def feature_dimension(variant: str) -> int:
    if variant == "pooled_same_frame":
        return 512
    if variant == "pooled_pair":
        return 4 * 512
    if variant == "spatial_same_frame":
        return 256 * 4 * 4
    if variant == "spatial_motion":
        return 3 * 256 * 4 * 4
    raise ValueError(f"unknown feature variant: {variant}")


@dataclass(frozen=True)
class FeatureStandardizer:
    """Per-dimension statistics fit only on selected training rows."""

    mean: torch.Tensor
    scale: torch.Tensor


GIB = 1024**3
WORST_CASE_TRAIN_SAMPLES = 143_451
WORST_CASE_VALIDATION_SAMPLES = 35_074
MAX_PROJECTED_RUNTIME_SECONDS = 2 * 60 * 60
BENCHMARK_RUNTIME_MULTIPLIER = 1.35
BENCHMARK_FIXED_OVERHEAD_SECONDS = 15 * 60


def materialized_feature_bytes(
    train_samples: int,
    validation_samples: int,
    variant: str,
) -> int:
    """Exact bytes for the two persistent float32 feature matrices."""

    return int((train_samples + validation_samples) * feature_dimension(variant) * 4)


def require_materialization_capacity(
    *,
    required_bytes: int,
    configured_limit_bytes: int,
    device: torch.device,
) -> dict[str, int | str]:
    """Fail before allocation instead of paging a large matrix silently."""

    if configured_limit_bytes < 1:
        raise ValueError("materialized feature byte limit must be positive")
    if required_bytes > configured_limit_bytes:
        raise MemoryError(
            "materialized train+validation feature matrices require "
            f"{required_bytes / GIB:.3f} GiB, above configured limit "
            f"{configured_limit_bytes / GIB:.3f} GiB"
        )
    receipt: dict[str, int | str] = {
        "device": str(device),
        "required_bytes": int(required_bytes),
        "configured_limit_bytes": int(configured_limit_bytes),
    }
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        # Keep a fixed 2 GiB or ten-percent reserve (whichever is larger) for
        # the frozen encoder, minibatches, CUDA libraries, and probe optimizer.
        reserve_bytes = max(2 * GIB, int(total_bytes * 0.10))
        receipt.update(
            {
                "cuda_free_bytes_before_materialization": int(free_bytes),
                "cuda_total_bytes": int(total_bytes),
                "cuda_required_reserve_bytes": int(reserve_bytes),
            }
        )
        if required_bytes + reserve_bytes > free_bytes:
            raise MemoryError(
                "materialized feature matrices plus reserve require "
                f"{(required_bytes + reserve_bytes) / GIB:.3f} GiB, but CUDA "
                f"reports only {free_bytes / GIB:.3f} GiB free"
            )
    return receipt


@torch.inference_mode()
def materialize_feature_matrix(
    corpus: EncodedCorpus,
    candidates: CandidateSet,
    *,
    variant: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Materialize exactly one aligned feature matrix directly on device."""

    if batch_size < 1:
        raise ValueError("materialization batch size must be positive")
    matrix = torch.empty(
        (len(candidates), feature_dimension(variant)),
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, len(candidates), batch_size):
        end = min(start + batch_size, len(candidates))
        selection = np.arange(start, end, dtype=np.int64)
        matrix[start:end].copy_(
            pair_features(
                corpus,
                candidates,
                selection,
                variant=variant,
                device=device,
            )
        )
    return matrix


@torch.inference_mode()
def fit_matrix_standardizer(
    matrix: torch.Tensor,
    selection: np.ndarray,
    *,
    batch_size: int,
) -> FeatureStandardizer:
    """Fit train-only per-dimension moments without copying the full matrix."""

    selected = np.asarray(selection, dtype=np.int64)
    if not len(selected):
        raise ValueError("cannot standardize empty training selection")
    total = torch.zeros(matrix.shape[1], dtype=torch.float64, device=matrix.device)
    total_square = torch.zeros_like(total)
    for start in range(0, len(selected), batch_size):
        indices = torch.from_numpy(selected[start : start + batch_size]).to(matrix.device)
        values = matrix.index_select(0, indices).to(torch.float64)
        total += values.sum(dim=0)
        total_square += values.square().sum(dim=0)
    mean64 = total / len(selected)
    variance64 = (total_square / len(selected) - mean64.square()).clamp_min_(0.0)
    scale64 = variance64.sqrt()
    scale64 = torch.where(scale64 > 1e-6, scale64, torch.ones_like(scale64))
    return FeatureStandardizer(mean64.float(), scale64.float())


@torch.inference_mode()
def standardize_matrix_in_place(
    matrix: torch.Tensor,
    standardizer: FeatureStandardizer,
    *,
    batch_size: int,
) -> None:
    for start in range(0, len(matrix), batch_size):
        end = min(start + batch_size, len(matrix))
        matrix[start:end].sub_(standardizer.mean).div_(standardizer.scale)


@torch.inference_mode()
def fit_feature_standardizer(
    corpus: EncodedCorpus,
    candidates: CandidateSet,
    selection: np.ndarray,
    *,
    variant: str,
    device: torch.device,
    batch_size: int,
) -> FeatureStandardizer:
    if not len(selection):
        raise ValueError("cannot standardize empty training selection")
    dimension = feature_dimension(variant)
    total = torch.zeros(dimension, dtype=torch.float64, device=device)
    total_square = torch.zeros(dimension, dtype=torch.float64, device=device)
    seen = 0
    for start in range(0, len(selection), batch_size):
        chosen = selection[start : start + batch_size]
        features = pair_features(
            corpus, candidates, chosen, variant=variant, device=device
        ).to(torch.float64)
        total += features.sum(dim=0)
        total_square += features.square().sum(dim=0)
        seen += len(chosen)
    mean64 = total / seen
    variance64 = (total_square / seen - mean64.square()).clamp_min_(0.0)
    scale64 = variance64.sqrt()
    # Constant training dimensions carry no information.  A unit scale keeps
    # them at zero after centering and avoids division-by-zero behavior.
    scale64 = torch.where(scale64 > 1e-6, scale64, torch.ones_like(scale64))
    return FeatureStandardizer(
        mean=mean64.to(torch.float32),
        scale=scale64.to(torch.float32),
    )


def standardize_features(
    features: torch.Tensor,
    standardizer: FeatureStandardizer,
) -> torch.Tensor:
    return (features - standardizer.mean) / standardizer.scale


def _standardizer_record(standardizer: FeatureStandardizer) -> dict[str, str]:
    return {
        "mean_sha256": canonical_array_sha256(
            standardizer.mean.detach().cpu().numpy().astype(np.float32)
        ),
        "scale_sha256": canonical_array_sha256(
            standardizer.scale.detach().cpu().numpy().astype(np.float32)
        ),
    }


class LinearTransitionProbe(nn.Module):
    def __init__(self, input_dim: int, prevalence: np.ndarray) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, len(OUTPUT_NAMES))
        nn.init.zeros_(self.linear.weight)
        clipped = np.clip(np.asarray(prevalence, dtype=np.float64), 1e-6, 1 - 1e-6)
        initial_bias = np.log(clipped) - np.log1p(-clipped)
        with torch.no_grad():
            self.linear.bias.copy_(torch.from_numpy(initial_bias.astype(np.float32)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def _derived_seed(seed: int, variant: str, offset: int) -> int:
    payload = f"{seed}\0{variant}\0{offset}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def select_natural_training_examples(
    sample_count: int,
    *,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Uniformly subsample candidates without changing class prevalence."""

    if sample_count < 1:
        raise ValueError("training candidate set is empty")
    if maximum < 0:
        raise ValueError("maximum training samples cannot be negative")
    if maximum == 0 or sample_count <= maximum:
        return np.arange(sample_count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(sample_count, size=maximum, replace=False)).astype(np.int64)


def _state_sha256(model: LinearTransitionProbe) -> str:
    buffer = io.BytesIO()
    for name, tensor in sorted(model.state_dict().items()):
        buffer.write(name.encode("utf-8"))
        buffer.write(b"\0")
        array = tensor.detach().cpu().numpy().astype("<f4", copy=False)
        buffer.write(str(array.shape).encode("ascii"))
        buffer.write(b"\0")
        buffer.write(array.tobytes(order="C"))
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def fit_linear_probe(
    corpus: EncodedCorpus,
    candidates: CandidateSet,
    *,
    variant: str,
    offset: int,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_train_samples: int,
    positive_weight_cap: float,
) -> tuple[LinearTransitionProbe, FeatureStandardizer, dict[str, Any]]:
    if (
        epochs < 1
        or batch_size < 1
        or learning_rate <= 0
        or weight_decay < 0
        or positive_weight_cap < 1
    ):
        raise ValueError("invalid fixed optimization budget")
    derived_seed = _derived_seed(seed, variant, offset)
    selected = select_natural_training_examples(
        len(candidates), maximum=max_train_samples, seed=derived_seed
    )
    all_targets = transition_targets(corpus.keys, candidates)
    targets = all_targets[selected]
    positive = targets.sum(axis=0, dtype=np.float64)
    negative = len(targets) - positive
    prevalence = positive / len(targets)
    estimable = (positive > 0) & (negative > 0)
    positive_weight = np.ones(len(OUTPUT_NAMES), dtype=np.float32)
    positive_weight[estimable] = np.minimum(
        negative[estimable] / positive[estimable], positive_weight_cap
    ).astype(np.float32)
    standardizer = fit_feature_standardizer(
        corpus,
        candidates,
        selected,
        variant=variant,
        device=device,
        batch_size=batch_size,
    )

    torch.manual_seed(derived_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(derived_seed)
    model = LinearTransitionProbe(feature_dimension(variant), prevalence).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    positive_weight_tensor = torch.from_numpy(positive_weight).to(device)
    generator = np.random.default_rng(derived_seed)
    final_loss = math.nan
    model.train()
    for _ in range(epochs):
        order = generator.permutation(len(selected))
        loss_numerator = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            candidate_selection = selected[positions]
            features = pair_features(
                corpus,
                candidates,
                candidate_selection,
                variant=variant,
                device=device,
            )
            features = standardize_features(features, standardizer)
            labels = torch.from_numpy(targets[positions]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=positive_weight_tensor
            )
            loss.backward()
            optimizer.step()
            loss_numerator += float(loss.detach().cpu()) * len(positions)
            seen += len(positions)
        final_loss = loss_numerator / seen

    return model.eval(), standardizer, {
        "derived_seed": int(derived_seed),
        "candidate_samples": int(len(candidates)),
        "fitted_samples": int(len(selected)),
        "uniform_natural_prevalence_subsample": bool(len(selected) < len(candidates)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "loss": "class-balanced BCE on natural uniformly sampled rows",
        "positive_weight_cap": float(positive_weight_cap),
        "per_output_positive_weight": {
            name: float(positive_weight[column])
            for column, name in enumerate(OUTPUT_NAMES)
        },
        "per_output_estimable": {
            name: bool(estimable[column])
            for column, name in enumerate(OUTPUT_NAMES)
        },
        "standardization": {
            "fit_rows": "selected training rows only",
            **_standardizer_record(standardizer),
        },
        "final_weighted_bce": float(final_loss),
        "model_state_sha256": _state_sha256(model),
        "support": _support_record(targets),
    }


def fit_linear_probe_from_matrix(
    train_features: torch.Tensor,
    train_targets: np.ndarray,
    selection: np.ndarray,
    *,
    variant: str,
    offset: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    positive_weight_cap: float,
) -> tuple[LinearTransitionProbe, dict[str, Any]]:
    """Fit one seed using a standardized feature matrix already on device."""

    targets = np.asarray(train_targets, dtype=np.float32)
    selected = np.asarray(selection, dtype=np.int64)
    if train_features.ndim != 2 or train_features.shape[1] != feature_dimension(variant):
        raise ValueError("materialized train feature matrix has the wrong shape")
    if targets.shape != (len(train_features), len(OUTPUT_NAMES)):
        raise ValueError("materialized train targets have the wrong shape")
    if not len(selected):
        raise ValueError("training selection is empty")
    selected_targets = targets[selected]
    positive = selected_targets.sum(axis=0, dtype=np.float64)
    negative = len(selected_targets) - positive
    prevalence = positive / len(selected_targets)
    estimable = (positive > 0) & (negative > 0)
    positive_weight = np.ones(len(OUTPUT_NAMES), dtype=np.float32)
    positive_weight[estimable] = np.minimum(
        negative[estimable] / positive[estimable], positive_weight_cap
    ).astype(np.float32)
    derived_seed = _derived_seed(seed, variant, offset)
    torch.manual_seed(derived_seed)
    if train_features.device.type == "cuda":
        torch.cuda.manual_seed_all(derived_seed)
    model = LinearTransitionProbe(feature_dimension(variant), prevalence).to(
        train_features.device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    target_tensor = torch.from_numpy(targets).to(train_features.device)
    positive_weight_tensor = torch.from_numpy(positive_weight).to(train_features.device)
    generator = np.random.default_rng(derived_seed)
    final_loss = math.nan
    model.train()
    for _ in range(epochs):
        order = generator.permutation(len(selected))
        loss_numerator = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            row_indices_numpy = selected[positions]
            row_indices = torch.from_numpy(row_indices_numpy).to(train_features.device)
            features = train_features.index_select(0, row_indices)
            labels = target_tensor.index_select(0, row_indices)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=positive_weight_tensor
            )
            loss.backward()
            optimizer.step()
            loss_numerator += float(loss.detach().cpu()) * len(positions)
            seen += len(positions)
        final_loss = loss_numerator / seen
    return model.eval(), {
        "derived_seed": int(derived_seed),
        "candidate_samples": int(len(train_features)),
        "fitted_samples": int(len(selected)),
        "uniform_natural_prevalence_subsample": bool(
            len(selected) < len(train_features)
        ),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "loss": "class-balanced BCE on natural uniformly sampled rows",
        "positive_weight_cap": float(positive_weight_cap),
        "per_output_positive_weight": {
            name: float(positive_weight[column])
            for column, name in enumerate(OUTPUT_NAMES)
        },
        "per_output_estimable": {
            name: bool(estimable[column])
            for column, name in enumerate(OUTPUT_NAMES)
        },
        "final_weighted_bce": float(final_loss),
        "model_state_sha256": _state_sha256(model),
        "support": _support_record(selected_targets),
    }


@torch.inference_mode()
def predict_probe_from_matrix(
    model: LinearTransitionProbe,
    validation_features: torch.Tensor,
    *,
    batch_size: int,
) -> np.ndarray:
    result = np.empty(
        (len(validation_features), len(OUTPUT_NAMES)), dtype=np.float32
    )
    for start in range(0, len(validation_features), batch_size):
        end = min(start + batch_size, len(validation_features))
        result[start:end] = (
            torch.sigmoid(model(validation_features[start:end])).cpu().numpy()
        )
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise RuntimeError("probe produced invalid probabilities")
    return result


@torch.inference_mode()
def predict_probe(
    model: LinearTransitionProbe,
    standardizer: FeatureStandardizer,
    corpus: EncodedCorpus,
    candidates: CandidateSet,
    *,
    variant: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty((len(candidates), len(OUTPUT_NAMES)), dtype=np.float32)
    for start in range(0, len(candidates), batch_size):
        end = min(start + batch_size, len(candidates))
        selection = np.arange(start, end, dtype=np.int64)
        features = pair_features(
            corpus, candidates, selection, variant=variant, device=device
        )
        features = standardize_features(features, standardizer)
        result[start:end] = torch.sigmoid(model(features)).cpu().numpy()
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise RuntimeError("probe produced invalid probabilities")
    return result


def _support_record(targets: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(targets, dtype=np.float64)
    return {
        "samples": int(len(labels)),
        "per_output": {
            name: {
                "positives": int(labels[:, column].sum()),
                "negatives": int(len(labels) - labels[:, column].sum()),
                "prevalence": float(labels[:, column].mean()),
                "estimable": bool(
                    0 < labels[:, column].sum() < len(labels)
                ),
            }
            for column, name in enumerate(OUTPUT_NAMES)
        },
    }


def binary_diagnostic_metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(truth, dtype=np.uint8)
    scores = np.asarray(probability, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("truth and probability must be aligned vectors")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("truth must be binary")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("probability must be finite in [0,1]")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    prediction = scores >= 0.5
    true_positive = int(np.logical_and(prediction, labels == 1).sum())
    false_positive = int(np.logical_and(prediction, labels == 0).sum())
    false_negative = int(np.logical_and(~prediction, labels == 1).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    prevalence = positives / len(labels) if len(labels) else 0.0
    estimable = bool(positives > 0 and negatives > 0)
    # One-class and empty surfaces are valid evidence about support, but they
    # cannot estimate discriminative AP skill.  Keep the report JSON-finite
    # and make that fact explicit instead of aborting the entire probe.
    average_precision = (
        float(average_precision_score(labels, scores))
        if estimable
        else float(prevalence)
    )
    predicted_positive_rate = float(prediction.mean()) if len(labels) else 0.0
    return {
        "estimable": estimable,
        "inestimable_reason": (
            None
            if estimable
            else ("empty support" if not len(labels) else "one-class support")
        ),
        "positives": positives,
        "negatives": negatives,
        "prevalence_chance_ap": float(prevalence),
        "average_precision": average_precision,
        "normalized_ap_skill": (
            float((average_precision - prevalence) / (1.0 - prevalence))
            if estimable
            else 0.0
        ),
        "fixed_threshold": 0.5,
        "fixed_f1": float(f1),
        "fixed_precision": float(precision),
        "fixed_recall": float(recall),
        "predicted_positives": int(prediction.sum()),
        "predicted_positive_rate": predicted_positive_rate,
    }


def score_predictions(
    truth: np.ndarray,
    probability: np.ndarray,
    *,
    post_state: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.asarray(truth)
    scores = np.asarray(probability)
    if labels.shape != scores.shape or labels.shape[1:] != (len(OUTPUT_NAMES),):
        raise ValueError("transition truth/probability must have shape [N,14]")
    per_output = {
        name: binary_diagnostic_metrics(labels[:, column], scores[:, column])
        for column, name in enumerate(OUTPUT_NAMES)
    }
    onset = [per_output[f"{key}:onset"] for key in KEY_ORDER]
    release = [per_output[f"{key}:release"] for key in KEY_ORDER]
    report = {
        "support": _support_record(labels),
        "per_output": per_output,
        "macro": {
            "onset_average_precision": float(np.mean([row["average_precision"] for row in onset])),
            "release_average_precision": float(np.mean([row["average_precision"] for row in release])),
            "all_event_average_precision": float(np.mean([row["average_precision"] for row in per_output.values()])),
            "onset_fixed_f1": float(np.mean([row["fixed_f1"] for row in onset])),
            "release_fixed_f1": float(np.mean([row["fixed_f1"] for row in release])),
            "all_event_fixed_f1": float(np.mean([row["fixed_f1"] for row in per_output.values()])),
            "all_event_prevalence_chance_ap": float(np.mean([row["prevalence_chance_ap"] for row in per_output.values()])),
        },
        "truth_sha256": canonical_array_sha256(labels.astype(np.uint8)),
        "probability_sha256": canonical_array_sha256(scores.astype(np.float32)),
    }
    if post_state is not None:
        state = np.asarray(post_state, dtype=np.uint8)
        if state.shape != (len(labels), len(KEY_ORDER)) or not np.all(
            np.isin(state, (0, 1))
        ):
            raise ValueError("post_state must be binary [N,7]")
        conditioned: dict[str, dict[str, Any]] = {}
        for column, key in enumerate(KEY_ORDER):
            onset_mask = state[:, column] == 1
            release_mask = state[:, column] == 0
            conditioned[f"{key}:onset"] = binary_diagnostic_metrics(
                labels[onset_mask, column], scores[onset_mask, column]
            )
            release_column = column + len(KEY_ORDER)
            conditioned[f"{key}:release"] = binary_diagnostic_metrics(
                labels[release_mask, release_column], scores[release_mask, release_column]
            )
        conditioned_onset = [conditioned[f"{key}:onset"] for key in KEY_ORDER]
        conditioned_release = [conditioned[f"{key}:release"] for key in KEY_ORDER]
        report["post_state_conditioned_per_output"] = conditioned
        report["macro"].update(
            {
                "conditioned_onset_average_precision": float(
                    np.mean([row["average_precision"] for row in conditioned_onset])
                ),
                "conditioned_release_average_precision": float(
                    np.mean([row["average_precision"] for row in conditioned_release])
                ),
                "conditioned_all_event_average_precision": float(
                    np.mean([row["average_precision"] for row in conditioned.values()])
                ),
                "conditioned_all_event_normalized_ap_skill": float(
                    np.mean([row["normalized_ap_skill"] for row in conditioned.values()])
                ),
            }
        )
    return report


def aggregate_seed_scores(seed_scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not seed_scores:
        raise ValueError("at least one seed score is required")
    metric_names = tuple(seed_scores[0]["macro"])
    return {
        name: {
            "mean": float(np.mean([score["macro"][name] for score in seed_scores])),
            "std_population": float(
                np.std([score["macro"][name] for score in seed_scores], ddof=0)
            ),
        }
        for name in metric_names
    }


def raw_pair_lift(
    results: dict[str, Any],
    *,
    pair_variant: str,
    control_variant: str,
    offsets: Sequence[int],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Uncertainty-free pair-minus-same-frame AP receipts for later scoring."""

    output: dict[str, Any] = {}
    for offset in offsets:
        offset_key = str(offset)
        seed_rows: dict[str, Any] = {}
        for seed in seeds:
            seed_key = str(seed)
            pair = results[pair_variant][offset_key]["seeds"][seed_key]["validation"]
            control = results[control_variant][offset_key]["seeds"][seed_key][
                "validation"
            ]
            per_output = {
                name: float(
                    pair["post_state_conditioned_per_output"][name][
                        "normalized_ap_skill"
                    ]
                    - control["post_state_conditioned_per_output"][name][
                        "normalized_ap_skill"
                    ]
                )
                for name in OUTPUT_NAMES
            }
            seed_rows[seed_key] = {
                "conditioned_normalized_ap_skill_lift_per_output": per_output,
                "macro_conditioned_normalized_ap_skill_lift": float(
                    np.mean(list(per_output.values()))
                ),
                "natural_support_macro_ap_lift": float(
                    pair["macro"]["all_event_average_precision"]
                    - control["macro"]["all_event_average_precision"]
                ),
            }
        output[offset_key] = seed_rows
    return output


def prediction_array_name(variant: str, offset: int, seed: int) -> str:
    if variant not in FEATURE_VARIANTS:
        raise ValueError(f"unknown feature variant: {variant}")
    offset_token = f"m{abs(offset):02d}" if offset < 0 else f"p{offset:02d}"
    return f"y_prob__{variant}__offset_{offset_token}__seed_{seed}"


def loso_prediction_array_name(variant: str, offset: int, seed: int) -> str:
    return prediction_array_name(variant, offset, seed).replace(
        "y_prob__", "loso_prob__", 1
    )


def validate_prediction_sidecar_arrays(arrays: dict[str, np.ndarray]) -> None:
    required = {
        "y_true",
        "post_state",
        "key_context",
        "context_relative_offsets",
        "target_global_index",
        "target_engine_frame_idx",
        "target_session_index",
        "target_run_id",
        "session_ids",
        "session_lengths",
        "offsets",
        "variants",
        "seeds",
        "train_y_true",
        "train_post_state",
        "train_key_context",
        "train_target_global_index",
        "train_target_engine_frame_idx",
        "train_target_session_index",
        "train_target_run_id",
        "train_session_ids",
        "train_session_lengths",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"prediction sidecar missing arrays: {sorted(missing)}")
    truth = np.asarray(arrays["y_true"])
    post_state = np.asarray(arrays["post_state"])
    if truth.dtype != np.uint8 or truth.ndim != 2 or truth.shape[1] != len(OUTPUT_NAMES):
        raise ValueError("sidecar y_true must be uint8 [N,14]")
    if post_state.dtype != np.uint8 or post_state.shape != (len(truth), len(KEY_ORDER)):
        raise ValueError("sidecar post_state must be uint8 [N,7]")
    if not np.all(np.isin(truth, (0, 1))) or not np.all(
        np.isin(post_state, (0, 1))
    ):
        raise ValueError("sidecar truth/state arrays must be binary")
    relative_offsets = np.asarray(arrays["context_relative_offsets"])
    expected_relative_offsets = np.arange(-5, 18, dtype=np.int16)
    if (
        relative_offsets.dtype != np.int16
        or not np.array_equal(relative_offsets, expected_relative_offsets)
    ):
        raise ValueError("sidecar context_relative_offsets must be int16 -5..17")
    key_context = np.asarray(arrays["key_context"])
    if (
        key_context.dtype != np.uint8
        or key_context.shape
        != (len(truth), len(expected_relative_offsets), len(KEY_ORDER))
        or not np.all(np.isin(key_context, (0, 1)))
    ):
        raise ValueError("sidecar key_context must be binary uint8 [N,23,7]")
    zero_position = int(np.flatnonzero(relative_offsets == 0)[0])
    previous_position = int(np.flatnonzero(relative_offsets == -1)[0])
    if not np.array_equal(key_context[:, zero_position], post_state):
        raise ValueError("sidecar key_context at t must equal post_state")
    context_onset = (key_context[:, previous_position] == 0) & (
        key_context[:, zero_position] == 1
    )
    context_release = (key_context[:, previous_position] == 1) & (
        key_context[:, zero_position] == 0
    )
    if not np.array_equal(
        np.concatenate((context_onset, context_release), axis=1).astype(np.uint8),
        truth,
    ):
        raise ValueError("sidecar key_context transition at t must equal y_true")
    for name in (
        "target_global_index",
        "target_engine_frame_idx",
        "target_session_index",
        "target_run_id",
    ):
        value = np.asarray(arrays[name])
        if value.shape != (len(truth),) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"sidecar {name} must be integer [N]")
    session_ids = np.asarray(arrays["session_ids"])
    session_lengths = np.asarray(arrays["session_lengths"])
    if session_ids.ndim != 1 or session_lengths.shape != session_ids.shape:
        raise ValueError("sidecar session IDs/lengths must be aligned vectors")
    if len(set(session_ids.astype(str).tolist())) != len(session_ids):
        raise ValueError("sidecar session IDs must be unique")
    if not np.issubdtype(session_lengths.dtype, np.integer) or np.any(
        session_lengths <= 0
    ):
        raise ValueError("sidecar session lengths must be positive integers")
    if int(session_lengths.sum()) != len(truth):
        raise ValueError("sidecar session lengths must sum to N")
    offsets = np.asarray(arrays["offsets"])
    variants = np.asarray(arrays["variants"])
    seeds = np.asarray(arrays["seeds"])
    if (
        offsets.ndim != 1
        or not len(offsets)
        or not np.issubdtype(offsets.dtype, np.integer)
        or variants.ndim != 1
        or not len(variants)
        or seeds.ndim != 1
        or not len(seeds)
        or not np.issubdtype(seeds.dtype, np.integer)
    ):
        raise ValueError("sidecar offsets/variants/seeds must be non-empty vectors")
    offset_values = [int(value) for value in offsets]
    variant_values = variants.astype(str).tolist()
    seed_values = [int(value) for value in seeds]
    if (
        len(set(offset_values)) != len(offset_values)
        or len(set(variant_values)) != len(variant_values)
        or len(set(seed_values)) != len(seed_values)
    ):
        raise ValueError("sidecar offsets/variants/seeds must be unique")
    expected_probability_names = {
        prediction_array_name(variant, offset, seed)
        for variant in variant_values
        for offset in offset_values
        for seed in seed_values
    }
    probability_names = {
        name for name in arrays if name.startswith("y_prob__")
    }
    if probability_names != expected_probability_names:
        raise ValueError(
            "prediction sidecar y_prob member names do not match metadata: "
            f"missing={sorted(expected_probability_names - probability_names)}, "
            f"extra={sorted(probability_names - expected_probability_names)}"
        )
    for name in probability_names:
        probability = np.asarray(arrays[name])
        if probability.dtype != np.float32 or probability.shape != truth.shape:
            raise ValueError(f"sidecar {name} must be float32 [N,14]")
        if not np.all(np.isfinite(probability)) or np.any(
            (probability < 0) | (probability > 1)
        ):
            raise ValueError(f"sidecar {name} must be finite in [0,1]")
    train_truth = np.asarray(arrays["train_y_true"])
    train_state = np.asarray(arrays["train_post_state"])
    if train_truth.dtype != np.uint8 or train_truth.shape[1:] != (
        len(OUTPUT_NAMES),
    ):
        raise ValueError("sidecar train_y_true must be uint8 [M,14]")
    if train_state.dtype != np.uint8 or train_state.shape != (
        len(train_truth),
        len(KEY_ORDER),
    ):
        raise ValueError("sidecar train_post_state must be uint8 [M,7]")
    train_context = np.asarray(arrays["train_key_context"])
    if (
        train_context.dtype != np.uint8
        or train_context.shape
        != (len(train_truth), len(expected_relative_offsets), len(KEY_ORDER))
        or not np.all(np.isin(train_context, (0, 1)))
    ):
        raise ValueError("sidecar train_key_context must be binary uint8 [M,23,7]")
    if not np.array_equal(train_context[:, zero_position], train_state):
        raise ValueError("sidecar train_key_context at t must equal train_post_state")
    train_onset = (train_context[:, previous_position] == 0) & (
        train_context[:, zero_position] == 1
    )
    train_release = (train_context[:, previous_position] == 1) & (
        train_context[:, zero_position] == 0
    )
    if not np.array_equal(
        np.concatenate((train_onset, train_release), axis=1).astype(np.uint8),
        train_truth,
    ):
        raise ValueError("sidecar train key_context transition at t must equal train_y_true")
    for name in (
        "train_target_global_index",
        "train_target_engine_frame_idx",
        "train_target_session_index",
        "train_target_run_id",
    ):
        value = np.asarray(arrays[name])
        if value.shape != (len(train_truth),) or not np.issubdtype(
            value.dtype, np.integer
        ):
            raise ValueError(f"sidecar {name} must be integer [M]")
    train_ids = np.asarray(arrays["train_session_ids"])
    train_lengths = np.asarray(arrays["train_session_lengths"])
    if train_ids.ndim != 1 or train_lengths.shape != train_ids.shape:
        raise ValueError("sidecar train session IDs/lengths must align")
    if int(train_lengths.sum()) != len(train_truth):
        raise ValueError("sidecar train session lengths must sum to M")
    expected_loso_names = {
        loso_prediction_array_name(variant, offset, seed_values[0])
        for variant in variant_values
        for offset in offset_values
    }
    loso_names = {name for name in arrays if name.startswith("loso_prob__")}
    if loso_names != expected_loso_names:
        raise ValueError(
            "prediction sidecar loso_prob member names do not match metadata: "
            f"missing={sorted(expected_loso_names - loso_names)}, "
            f"extra={sorted(loso_names - expected_loso_names)}"
        )
    for name in loso_names:
        probability = np.asarray(arrays[name])
        if probability.dtype != np.float32 or probability.shape != train_truth.shape:
            raise ValueError(f"sidecar {name} must be float32 [M,14]")
        if not np.all(np.isfinite(probability)) or np.any(
            (probability < 0) | (probability > 1)
        ):
            raise ValueError(f"sidecar {name} must be finite in [0,1]")


def write_prediction_sidecar(
    path: Path,
    arrays: dict[str, np.ndarray],
) -> str:
    """Validate, compress, fsync, and exclusively publish an NPZ receipt."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing prediction sidecar: {destination}"
        )
    validate_prediction_sidecar_arrays(arrays)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        validate_prediction_sidecar(temporary_path)
        # Hard-link publication is atomic and, unlike replace(), cannot
        # overwrite an artifact that appeared during the run.
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return sha256_file(destination)


def write_json_exclusive_atomic(path: Path, payload: dict[str, Any]) -> str:
    """Serialize, fsync, validate, and publish JSON without overwrite."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite JSON artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        json.loads(temporary_path.read_text(encoding="utf-8"))
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return sha256_file(destination)


def validate_prediction_sidecar(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    validate_prediction_sidecar_arrays(arrays)
    return arrays


def parse_offsets(value: str) -> tuple[int, ...]:
    text = value.strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("offset range must be START:END")
        try:
            start, end = (int(part) for part in parts)
        except ValueError as error:
            raise argparse.ArgumentTypeError("offsets must be integers") from error
        if end < start:
            raise argparse.ArgumentTypeError("offset range END must be >= START")
        offsets = tuple(range(start, end + 1))
    else:
        try:
            offsets = tuple(int(part.strip()) for part in text.split(",") if part.strip())
        except ValueError as error:
            raise argparse.ArgumentTypeError("offsets must be integers") from error
    if not offsets or len(offsets) != len(set(offsets)):
        raise argparse.ArgumentTypeError("offsets must be non-empty and unique")
    return offsets


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def _device_identity(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return str(device)


def load_benchmark_contract(path: Path) -> dict[str, Any]:
    """Validate scientific benchmark knobs without touching any data path."""

    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    fit = contract.get("fit", {})
    required = {
        "schema_version": contract.get("schema_version"),
        "offsets": contract.get("offsets_native_frames"),
        "seeds": fit.get("random_seeds"),
        "epochs": fit.get("epochs"),
        "batch_size": fit.get("batch_size"),
        "learning_rate": fit.get("learning_rate"),
        "weight_decay": fit.get("weight_decay"),
        "backbone": contract.get("probe_surfaces", {}).get("backbone"),
    }
    expected = {
        "schema_version": "madeleine.dynamics-offset-probe.v1",
        "offsets": list(DEFAULT_OFFSETS),
        "seeds": [0, 1, 2],
        "epochs": 40,
        "batch_size": 2048,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "backbone": EXPECTED_BACKBONE_CONTRACT,
    }
    mismatch = [
        name for name in expected if required[name] != expected[name]
    ]
    if mismatch:
        raise ValueError(
            "benchmark contract does not preserve frozen knobs: "
            + ", ".join(mismatch)
        )
    return contract


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_synthetic_worst_case_benchmark(
    *,
    contract_path: Path,
    output: Path,
    device_name: str,
    max_materialized_bytes: int,
) -> dict[str, Any]:
    """Benchmark one synthetic cell per surface without opening real shards."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite benchmark: {destination}")
    contract = load_benchmark_contract(contract_path)
    device = _resolve_device(device_name)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    rng = np.random.default_rng(20260727)
    prevalence = np.linspace(0.005, 0.04, len(OUTPUT_NAMES))
    train_targets = rng.binomial(
        1,
        prevalence,
        size=(WORST_CASE_TRAIN_SAMPLES, len(OUTPUT_NAMES)),
    ).astype(np.float32)
    # Ensure the deterministic fixture has both classes regardless of a future
    # RNG implementation change.
    train_targets[0] = 0
    train_targets[1] = 1
    seeds = tuple(contract["fit"]["random_seeds"])
    batch_size = int(contract["fit"]["batch_size"])
    epochs = int(contract["fit"]["epochs"])
    learning_rate = float(contract["fit"]["learning_rate"])
    weight_decay = float(contract["fit"]["weight_decay"])
    cells: dict[str, Any] = {}
    for variant in FEATURE_VARIANTS:
        persistent_bytes = materialized_feature_bytes(
            WORST_CASE_TRAIN_SAMPLES,
            WORST_CASE_VALIDATION_SAMPLES,
            variant,
        )
        clone_bytes = int(
            WORST_CASE_TRAIN_SAMPLES * feature_dimension(variant) * 4
        )
        capacity = require_materialization_capacity(
            required_bytes=persistent_bytes + clone_bytes,
            configured_limit_bytes=max_materialized_bytes,
            device=device,
        )
        _synchronize(device)
        start_time = time.perf_counter()
        generator = torch.Generator(device=device)
        generator.manual_seed(_derived_seed(0, variant, 16))
        train_features = torch.randn(
            (WORST_CASE_TRAIN_SAMPLES, feature_dimension(variant)),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        validation_features = torch.randn(
            (WORST_CASE_VALIDATION_SAMPLES, feature_dimension(variant)),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        boundaries = np.linspace(
            0, WORST_CASE_TRAIN_SAMPLES, 4, dtype=np.int64
        )
        for fold in range(3):
            heldout = np.arange(boundaries[fold], boundaries[fold + 1])
            fit_indices = np.concatenate(
                (
                    np.arange(0, boundaries[fold]),
                    np.arange(boundaries[fold + 1], WORST_CASE_TRAIN_SAMPLES),
                )
            ).astype(np.int64)
            scaler = fit_matrix_standardizer(
                train_features, fit_indices, batch_size=batch_size
            )
            fold_features = train_features.clone()
            standardize_matrix_in_place(
                fold_features, scaler, batch_size=batch_size
            )
            model, _ = fit_linear_probe_from_matrix(
                fold_features,
                train_targets,
                fit_indices,
                variant=variant,
                offset=16,
                seed=int(seeds[0]),
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                positive_weight_cap=50.0,
            )
            heldout_tensor = torch.from_numpy(heldout).to(device)
            predict_probe_from_matrix(
                model,
                fold_features.index_select(0, heldout_tensor),
                batch_size=batch_size,
            )
            del fold_features, scaler, model
        selection = np.arange(WORST_CASE_TRAIN_SAMPLES, dtype=np.int64)
        scaler = fit_matrix_standardizer(
            train_features, selection, batch_size=batch_size
        )
        standardize_matrix_in_place(train_features, scaler, batch_size=batch_size)
        standardize_matrix_in_place(
            validation_features, scaler, batch_size=batch_size
        )
        for seed in seeds:
            model, _ = fit_linear_probe_from_matrix(
                train_features,
                train_targets,
                selection,
                variant=variant,
                offset=16,
                seed=int(seed),
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                positive_weight_cap=50.0,
            )
            predict_probe_from_matrix(
                model, validation_features, batch_size=batch_size
            )
            del model
        _synchronize(device)
        elapsed = time.perf_counter() - start_time
        cells[variant] = {
            "elapsed_seconds": float(elapsed),
            "feature_dimension": feature_dimension(variant),
            "capacity": capacity,
            "loso_fits": 3,
            "final_fits": 3,
            "epochs_per_fit": epochs,
        }
        del train_features, validation_features, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
    measured_cell_seconds = float(
        sum(cell["elapsed_seconds"] for cell in cells.values())
    )
    projected_seconds = float(
        measured_cell_seconds
        * len(DEFAULT_OFFSETS)
        * BENCHMARK_RUNTIME_MULTIPLIER
        + BENCHMARK_FIXED_OVERHEAD_SECONDS
    )
    # Import lazily to avoid a module-import cycle: the scorer imports this
    # module's schema helpers, while this synthetic gate must exercise the
    # scorer before any real shard is opened.
    from experiments.score_dynamics_offset_probe import (
        run_synthetic_null_benchmark,
    )

    scorer_script_path = Path(__file__).with_name(
        "score_dynamics_offset_probe.py"
    )
    contract_sha256 = sha256_file(Path(contract_path))
    scorer_script_sha256 = sha256_file(scorer_script_path)
    null_benchmark = {
        "schema_version": "madeleine.dynamics-offset-null-runtime.v1",
        "contract_sha256": contract_sha256,
        "scorer_script_sha256": scorer_script_sha256,
        **run_synthetic_null_benchmark(
            device_name=str(device),
        ),
    }
    allowed = (
        projected_seconds <= MAX_PROJECTED_RUNTIME_SECONDS
        and null_benchmark["allowed_to_open_real_shards"] is True
    )
    report = {
        "schema_version": "madeleine.dynamics-offset-probe-runtime.v1",
        "status": "pass" if allowed else "blocked_over_two_hours",
        "real_data_or_validation_opened": False,
        "contract_sha256": contract_sha256,
        "script_sha256": sha256_file(Path(__file__)),
        "scorer_script_sha256": scorer_script_sha256,
        "device": str(device),
        "device_identity": _device_identity(device),
        "worst_case_samples": {
            "train": WORST_CASE_TRAIN_SAMPLES,
            "validation": WORST_CASE_VALIDATION_SAMPLES,
        },
        "scientific_knobs": {
            "offsets": list(DEFAULT_OFFSETS),
            "variants": list(FEATURE_VARIANTS),
            "seeds": list(seeds),
            "loso_folds": 3,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "cells": cells,
        "scoring_null_benchmark": null_benchmark,
        "projection": {
            "measured_one_offset_all_surfaces_seconds": measured_cell_seconds,
            "offset_count": len(DEFAULT_OFFSETS),
            "runtime_multiplier_for_host_materialization_and_variance": (
                BENCHMARK_RUNTIME_MULTIPLIER
            ),
            "fixed_encoding_compression_verification_seconds": (
                BENCHMARK_FIXED_OVERHEAD_SECONDS
            ),
            "projected_full_runtime_seconds": projected_seconds,
            "hard_limit_seconds": MAX_PROJECTED_RUNTIME_SECONDS,
            "allowed_to_open_real_shards": allowed,
        },
    }
    write_json_exclusive_atomic(destination, report)
    return report


def validate_runtime_benchmark_receipt(
    path: Path,
    *,
    contract_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "madeleine.dynamics-offset-probe-runtime.v1":
        raise ValueError("unsupported runtime benchmark receipt schema")
    if receipt.get("real_data_or_validation_opened") is not False:
        raise ValueError("runtime benchmark was not synthetic-only")
    if receipt.get("contract_sha256") != sha256_file(Path(contract_path)):
        raise ValueError("runtime benchmark contract hash is stale")
    if receipt.get("script_sha256") != sha256_file(Path(__file__)):
        raise ValueError("runtime benchmark script hash is stale")
    scorer_script_path = Path(__file__).with_name(
        "score_dynamics_offset_probe.py"
    )
    if receipt.get("scorer_script_sha256") != sha256_file(scorer_script_path):
        raise ValueError("runtime benchmark scorer script hash is stale")
    if receipt.get("device_identity") != _device_identity(device):
        raise ValueError("runtime benchmark device does not match run device")
    samples = receipt.get("worst_case_samples", {})
    if samples.get("train", 0) < WORST_CASE_TRAIN_SAMPLES or samples.get(
        "validation", 0
    ) < WORST_CASE_VALIDATION_SAMPLES:
        raise ValueError("runtime benchmark sample support is too small")
    knobs = receipt.get("scientific_knobs", {})
    if (
        knobs.get("offsets") != list(DEFAULT_OFFSETS)
        or knobs.get("variants") != list(FEATURE_VARIANTS)
        or knobs.get("seeds") != [0, 1, 2]
        or knobs.get("loso_folds") != 3
        or knobs.get("epochs") != 40
        or knobs.get("batch_size") != 2048
    ):
        raise ValueError("runtime benchmark scientific knobs do not match run")
    cells = receipt.get("cells", {})
    if set(cells) != set(FEATURE_VARIANTS):
        raise ValueError("runtime benchmark cell set is incomplete")
    elapsed_cells: list[float] = []
    for variant in FEATURE_VARIANTS:
        cell = cells[variant]
        elapsed = cell.get("elapsed_seconds", math.nan)
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"runtime benchmark {variant} elapsed time is invalid")
        if (
            cell.get("feature_dimension") != feature_dimension(variant)
            or cell.get("loso_fits") != 3
            or cell.get("final_fits") != 3
            or cell.get("epochs_per_fit") != 40
        ):
            raise ValueError(f"runtime benchmark {variant} cell metadata is invalid")
        elapsed_cells.append(float(elapsed))
    measured = float(sum(elapsed_cells))
    projection = receipt.get("projection", {})
    expected_projected = float(
        measured * len(DEFAULT_OFFSETS) * BENCHMARK_RUNTIME_MULTIPLIER
        + BENCHMARK_FIXED_OVERHEAD_SECONDS
    )
    projection_values_match = (
        math.isclose(
            projection.get("measured_one_offset_all_surfaces_seconds", math.nan),
            measured,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        and projection.get("offset_count") == len(DEFAULT_OFFSETS)
        and projection.get("runtime_multiplier_for_host_materialization_and_variance")
        == BENCHMARK_RUNTIME_MULTIPLIER
        and projection.get("fixed_encoding_compression_verification_seconds")
        == BENCHMARK_FIXED_OVERHEAD_SECONDS
        and math.isclose(
            projection.get("projected_full_runtime_seconds", math.nan),
            expected_projected,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        and projection.get("hard_limit_seconds") == MAX_PROJECTED_RUNTIME_SECONDS
    )
    if not projection_values_match:
        raise ValueError("runtime benchmark projection arithmetic is invalid")

    from experiments.score_dynamics_offset_probe import (
        MAX_NULL_BENCHMARK_SECONDS,
        NULL_REPLICATES,
        NULL_STATISTIC_CONTRACT,
    )

    null = receipt.get("scoring_null_benchmark", {})
    null_elapsed = null.get("elapsed_seconds", math.nan)
    null_valid = (
        null.get("schema_version") == "madeleine.dynamics-offset-null-runtime.v1"
        and null.get("contract_sha256") == sha256_file(Path(contract_path))
        and null.get("scorer_script_sha256") == sha256_file(scorer_script_path)
        and null.get("status") == "pass"
        and null.get("real_data_or_validation_opened") is False
        and null.get("samples") == WORST_CASE_VALIDATION_SAMPLES
        and null.get("replicates") == NULL_REPLICATES
        and null.get("device") == str(device)
        and null.get("device_identity") == _device_identity(device)
        and null.get("statistic") == NULL_STATISTIC_CONTRACT
        and isinstance(null_elapsed, (int, float))
        and math.isfinite(null_elapsed)
        and 0 < null_elapsed <= MAX_NULL_BENCHMARK_SECONDS
        and null.get("hard_limit_seconds") == MAX_NULL_BENCHMARK_SECONDS
        and null.get("allowed_to_open_real_shards") is True
    )
    if not null_valid:
        raise RuntimeError("synthetic circular-null benchmark receipt is unsafe")
    projected = projection.get("projected_full_runtime_seconds", math.inf)
    expected_allowed = (
        expected_projected <= MAX_PROJECTED_RUNTIME_SECONDS and null_valid
    )
    if (
        receipt.get("status") != "pass"
        or projection.get("allowed_to_open_real_shards") is not expected_allowed
        or not math.isfinite(projected)
        or projected > MAX_PROJECTED_RUNTIME_SECONDS
    ):
        raise RuntimeError(
            "synthetic benchmark projects more than two GPU-hours; refusing "
            "to open real shards"
        )
    return receipt


def run_probe(
    *,
    data_dir: Path,
    train_list: Path,
    validation_list: Path,
    contract_path: Path,
    output: Path,
    predictions_output: Path,
    benchmark_receipt_path: Path,
    offsets: Sequence[int] = DEFAULT_OFFSETS,
    variants: Sequence[str] = FEATURE_VARIANTS,
    device_name: str = "cuda",
    encode_batch_size: int = 256,
    probe_batch_size: int = 2048,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_train_samples: int = 0,
    positive_weight_cap: float = 50.0,
    seeds: Sequence[int] = (0, 1, 2),
    max_materialized_bytes: int = 24 * GIB,
) -> dict[str, Any]:
    """Encode explicit splits, fit fixed probes, and write one JSON receipt."""

    output = Path(output)
    predictions_output = Path(predictions_output)
    if output == predictions_output:
        raise ValueError("JSON report and prediction sidecar paths must differ")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    if predictions_output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing prediction sidecar: {predictions_output}"
        )
    if not offsets or len(offsets) != len(set(offsets)):
        raise ValueError("offsets must be non-empty and unique")
    if not variants or len(variants) != len(set(variants)):
        raise ValueError("feature variants must be non-empty and unique")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("probe seeds must be non-empty and unique")
    unknown_variants = sorted(set(variants).difference(FEATURE_VARIANTS))
    if unknown_variants:
        raise ValueError(f"unknown feature variants: {unknown_variants}")

    train_ids = read_explicit_session_ids(train_list)
    validation_ids = read_explicit_session_ids(validation_list)
    validate_split_ids(train_ids, validation_ids)
    contract = load_and_validate_contract(
        contract_path,
        data_dir=data_dir,
        train_ids=train_ids,
        validation_ids=validation_ids,
        offsets=offsets,
        variants=variants,
        seeds=seeds,
        epochs=epochs,
        batch_size=probe_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_train_samples=max_train_samples,
        positive_weight_cap=positive_weight_cap,
    )
    require_canonical_session_list_bytes(train_list, train_ids)
    require_canonical_session_list_bytes(validation_list, validation_ids)
    device = _resolve_device(device_name)
    runtime_benchmark_receipt = validate_runtime_benchmark_receipt(
        benchmark_receipt_path,
        contract_path=contract_path,
        device=device,
    )
    manifest_path = Path(data_dir) / "build_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing preregistered build manifest: {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != contract["data"]["build_manifest_sha256"]:
        raise ValueError(
            f"build manifest SHA-256 {manifest_sha256} does not match contract "
            f"{contract['data']['build_manifest_sha256']}"
        )
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seeds[0]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seeds[0]))

    encoder = FrozenResNet18SpatialFeatures().eval().to(device)
    backbone_weight_receipt = verify_cached_backbone_weights(contract)

    def encode_ids(
        ids: Sequence[str], expected_hashes: dict[str, str]
    ) -> EncodedCorpus:
        encoded: list[EncodedSession] = []
        for session_id in ids:
            raw = load_rgb_session(data_dir, session_id)
            if raw.shard_sha256 != expected_hashes[session_id]:
                raise ValueError(
                    f"{raw.path}: SHA-256 {raw.shard_sha256} does not match "
                    f"preregistered {expected_hashes[session_id]}"
                )
            encoded.append(
                encode_raw_session(
                    raw,
                    encoder,
                    device=device,
                    batch_size=encode_batch_size,
                )
            )
        return concatenate_encoded_sessions(encoded)

    train_corpus = encode_ids(
        train_ids, contract["data"]["train_shard_sha256"]
    )
    validation_corpus = encode_ids(
        validation_ids, contract["data"]["validation_shard_sha256"]
    )
    results: dict[str, Any] = {variant: {} for variant in variants}
    prediction_arrays: dict[str, np.ndarray] = {
        "offsets": np.asarray(offsets, dtype=np.int16),
        "variants": np.asarray(variants),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "context_relative_offsets": np.arange(-5, 18, dtype=np.int16),
    }
    train_target_reference: np.ndarray | None = None
    validation_target_reference: np.ndarray | None = None
    train_support_sha256: str | None = None
    validation_support_sha256: str | None = None
    for offset in offsets:
        train_candidates = build_candidates(
            train_corpus, int(offset), common_offsets=offsets
        )
        validation_candidates = build_candidates(
            validation_corpus, int(offset), common_offsets=offsets
        )
        if not len(train_candidates) or not len(validation_candidates):
            raise ValueError(f"offset {offset} has empty train or validation support")
        current_train_hash = canonical_array_sha256(train_candidates.target_current)
        current_validation_hash = canonical_array_sha256(
            validation_candidates.target_current
        )
        if train_target_reference is None:
            train_target_reference = train_candidates.target_current.copy()
            validation_target_reference = validation_candidates.target_current.copy()
            train_support_sha256 = current_train_hash
            validation_support_sha256 = current_validation_hash
        else:
            assert validation_target_reference is not None
            if (
                current_train_hash != train_support_sha256
                or current_validation_hash != validation_support_sha256
                or not np.array_equal(
                    train_candidates.target_current, train_target_reference
                )
                or not np.array_equal(
                    validation_candidates.target_current,
                    validation_target_reference,
                )
            ):
                raise RuntimeError(
                    "offsets do not share exact common target support"
                )
        train_truth = transition_targets(train_corpus.keys, train_candidates)
        train_post_state = train_corpus.keys[train_candidates.target_current]
        validation_truth = transition_targets(validation_corpus.keys, validation_candidates)
        validation_post_state = validation_corpus.keys[
            validation_candidates.target_current
        ]
        train_context = target_key_context(train_corpus, train_candidates)
        validation_context = target_key_context(
            validation_corpus, validation_candidates
        )
        if "y_true" not in prediction_arrays:
            prediction_arrays.update(
                {
                    "y_true": validation_truth.astype(np.uint8),
                    "post_state": validation_post_state.astype(np.uint8),
                    "key_context": validation_context,
                    **target_identity_arrays(
                        validation_corpus, validation_candidates
                    ),
                    "train_y_true": train_truth.astype(np.uint8),
                    "train_post_state": train_post_state.astype(np.uint8),
                    "train_key_context": train_context,
                    **{
                        f"train_{name}": value
                        for name, value in target_identity_arrays(
                            train_corpus, train_candidates
                        ).items()
                    },
                }
            )
        elif (
            not np.array_equal(prediction_arrays["y_true"], validation_truth)
            or not np.array_equal(
                prediction_arrays["post_state"], validation_post_state
            )
            or not np.array_equal(prediction_arrays["train_y_true"], train_truth)
            or not np.array_equal(
                prediction_arrays["train_post_state"], train_post_state
            )
            or not np.array_equal(
                prediction_arrays["key_context"], validation_context
            )
            or not np.array_equal(
                prediction_arrays["train_key_context"], train_context
            )
        ):
            raise RuntimeError("truth/state/context changed across offsets")
        for variant in variants:
            selected = select_natural_training_examples(
                len(train_candidates),
                maximum=max_train_samples,
                seed=_derived_seed(0, variant, int(offset)),
            )
            persistent_bytes = materialized_feature_bytes(
                len(train_candidates), len(validation_candidates), variant
            )
            loso_clone_bytes = int(
                len(train_candidates) * feature_dimension(variant) * 4
            )
            required_bytes = persistent_bytes + loso_clone_bytes
            capacity_receipt = require_materialization_capacity(
                required_bytes=required_bytes,
                configured_limit_bytes=max_materialized_bytes,
                device=device,
            )
            train_features = materialize_feature_matrix(
                train_corpus,
                train_candidates,
                variant=variant,
                device=device,
                batch_size=probe_batch_size,
            )
            validation_features = materialize_feature_matrix(
                validation_corpus,
                validation_candidates,
                variant=variant,
                device=device,
                batch_size=probe_batch_size,
            )
            # LOSO is run on the three training sessions before the final
            # val-A fit.  One preregistered seed is sufficient for its sign
            # gate; the three-seed final fit remains unchanged below.
            loso_records: dict[str, Any] = {}
            loso_scores: list[dict[str, Any]] = []
            loso_probability = np.empty_like(train_truth, dtype=np.float32)
            session_slices = candidate_session_slices(
                train_corpus, train_candidates
            )
            loso_seed = int(seeds[0])
            for heldout_session in train_corpus.session_ids:
                heldout_slice = session_slices[heldout_session]
                heldout_indices = np.arange(
                    heldout_slice.start, heldout_slice.stop, dtype=np.int64
                )
                fit_indices = np.concatenate(
                    (
                        np.arange(0, heldout_slice.start, dtype=np.int64),
                        np.arange(
                            heldout_slice.stop,
                            len(train_candidates),
                            dtype=np.int64,
                        ),
                    )
                )
                fold_standardizer = fit_matrix_standardizer(
                    train_features,
                    fit_indices,
                    batch_size=probe_batch_size,
                )
                fold_features = train_features.clone()
                standardize_matrix_in_place(
                    fold_features,
                    fold_standardizer,
                    batch_size=probe_batch_size,
                )
                fold_model, fold_fit = fit_linear_probe_from_matrix(
                    fold_features,
                    train_truth,
                    fit_indices,
                    variant=variant,
                    offset=int(offset),
                    seed=loso_seed,
                    epochs=epochs,
                    batch_size=probe_batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    positive_weight_cap=positive_weight_cap,
                )
                fold_fit["standardization"] = {
                    "fit_rows": "two non-held-out training sessions only",
                    **_standardizer_record(fold_standardizer),
                }
                heldout_tensor_indices = torch.from_numpy(heldout_indices).to(
                    device
                )
                heldout_probability = predict_probe_from_matrix(
                    fold_model,
                    fold_features.index_select(0, heldout_tensor_indices),
                    batch_size=probe_batch_size,
                )
                loso_probability[heldout_indices] = heldout_probability
                fold_score = score_predictions(
                    train_truth[heldout_indices],
                    heldout_probability,
                    post_state=train_post_state[heldout_indices],
                )
                loso_scores.append(fold_score)
                loso_records[heldout_session] = {
                    "heldout_session": heldout_session,
                    "fit_session_ids": [
                        session_id
                        for session_id in train_corpus.session_ids
                        if session_id != heldout_session
                    ],
                    "seed": loso_seed,
                    "fit": fold_fit,
                    "heldout": fold_score,
                }
                del fold_features, fold_model, fold_standardizer
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            prediction_arrays[
                loso_prediction_array_name(variant, int(offset), loso_seed)
            ] = loso_probability
            standardizer = fit_matrix_standardizer(
                train_features,
                selected,
                batch_size=probe_batch_size,
            )
            standardize_matrix_in_place(
                train_features, standardizer, batch_size=probe_batch_size
            )
            standardize_matrix_in_place(
                validation_features, standardizer, batch_size=probe_batch_size
            )
            seed_records: dict[str, Any] = {}
            seed_scores: list[dict[str, Any]] = []
            for seed in seeds:
                model, fit = fit_linear_probe_from_matrix(
                    train_features,
                    train_truth,
                    selected,
                    variant=variant,
                    offset=int(offset),
                    seed=int(seed),
                    epochs=epochs,
                    batch_size=probe_batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    positive_weight_cap=positive_weight_cap,
                )
                fit["standardization"] = {
                    "fit_rows": "selected training rows only; shared across seeds",
                    **_standardizer_record(standardizer),
                }
                probability = predict_probe_from_matrix(
                    model,
                    validation_features,
                    batch_size=probe_batch_size,
                )
                prediction_arrays[
                    prediction_array_name(variant, int(offset), int(seed))
                ] = probability
                score = score_predictions(
                    validation_truth,
                    probability,
                    post_state=validation_post_state,
                )
                seed_scores.append(score)
                seed_records[str(seed)] = {"fit": fit, "validation": score}
                del model
            results[variant][str(offset)] = {
                "observed_pair_relative_to_target": [int(offset - 1), int(offset)],
                "train_candidates_per_session": train_candidates.per_session,
                "validation_candidates_per_session": validation_candidates.per_session,
                "train_target_support_per_session": per_session_target_support(
                    train_corpus, train_candidates
                ),
                "validation_target_support_per_session": per_session_target_support(
                    validation_corpus, validation_candidates
                ),
                "common_target_support": True,
                "train_target_index_sha256": canonical_array_sha256(
                    train_candidates.target_current
                ),
                "validation_target_index_sha256": canonical_array_sha256(
                    validation_candidates.target_current
                ),
                "materialization": {
                    **capacity_receipt,
                    "persistent_train_plus_validation_bytes": int(
                        persistent_bytes
                    ),
                    "peak_loso_clone_bytes": int(loso_clone_bytes),
                    "feature_dimension": feature_dimension(variant),
                    "train_shape": list(train_features.shape),
                    "validation_shape": list(validation_features.shape),
                    "dtype": str(train_features.dtype),
                    "materialized_once_and_reused_across_seeds": True,
                },
                "leave_one_training_session_out": {
                    "seed_policy": "one fixed seed (first preregistered seed) per fold",
                    "seed": loso_seed,
                    "folds": loso_records,
                    "fold_macro_aggregate": aggregate_seed_scores(loso_scores),
                },
                "seeds": seed_records,
                "seed_aggregate": aggregate_seed_scores(seed_scores),
            }
            del train_features, validation_features, standardizer
            if device.type == "cuda":
                torch.cuda.empty_cache()

    assert train_support_sha256 is not None
    assert validation_support_sha256 is not None

    comparisons: dict[str, Any] = {}
    if {"pooled_same_frame", "pooled_pair"}.issubset(results):
        comparisons["pooled_pair_minus_same_frame"] = raw_pair_lift(
            results,
            pair_variant="pooled_pair",
            control_variant="pooled_same_frame",
            offsets=offsets,
            seeds=seeds,
        )
    if {"spatial_same_frame", "spatial_motion"}.issubset(results):
        comparisons["spatial_motion_minus_same_frame"] = raw_pair_lift(
            results,
            pair_variant="spatial_motion",
            control_variant="spatial_same_frame",
            offsets=offsets,
            seeds=seeds,
        )

    prediction_sidecar_sha256 = write_prediction_sidecar(
        predictions_output, prediction_arrays
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "engine_truth_dynamics_identifiability_probe",
        "status": "raw_probe_evidence_complete",
        "protocol": {
            "target": "engine-truth onset and release at t",
            "observed_pair": "(frame[t+offset-1], frame[t+offset])",
            "offsets": [int(offset) for offset in offsets],
            "feature_variants": list(variants),
            "feature_dimensions": {
                variant: feature_dimension(variant) for variant in variants
            },
            "backbone": "torchvision ImageNet ResNet-18, frozen; pooled layer4 and 4x4-pooled layer3 controls",
            "probe": "one linear 14-output capped class-balanced BCE probe per variant/offset/seed",
            "validation_used_for_fit_or_selection": False,
            "event_resampling": False,
            "validation_ap_uses_natural_prevalence": True,
            "primary_metric_surface": "post-state-conditioned onset/release average precision",
            "feature_standardization": "per dimension, fit on training rows only",
            "leave_one_training_session_out": (
                "three folds run before final val-A; one fixed seed per fold, "
                "with fold-specific train-only standardization"
            ),
            "positive_weight_cap": float(positive_weight_cap),
            "fixed_diagnostic_threshold": 0.5,
            "common_support_relative_rows": [
                min(-1, min(int(offset) - 1 for offset in offsets)),
                max(0, max(int(offset) for offset in offsets)),
            ],
            "gap_policy": "target, observed pair, and intervening frames remain within one strictly consecutive engine-frame run",
            "follow_up_statistical_scorer_required_for_preregistered_decision": [
                "post-state persistence and intervening-action subsets",
                "continuity-bounded block bootstrap simultaneous bands",
                "within-run circular-shift null",
                "BH-FDR decision gates",
            ],
        },
        "optimization": {
            "seeds": [int(seed) for seed in seeds],
            "epochs": int(epochs),
            "encode_batch_size": int(encode_batch_size),
            "probe_batch_size": int(probe_batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "max_train_samples": int(max_train_samples),
            "deterministic_algorithms": True,
            "device": str(device),
            "max_materialized_bytes": int(max_materialized_bytes),
        },
        "common_support": {
            "identical_across_all_offsets": True,
            "train_target_index_sha256": train_support_sha256,
            "validation_target_index_sha256": validation_support_sha256,
            "relative_rows_inclusive": [-5, 17],
            "validation_key_context_sha256": canonical_array_sha256(
                prediction_arrays["key_context"]
            ),
            "train_key_context_sha256": canonical_array_sha256(
                prediction_arrays["train_key_context"]
            ),
        },
        "provenance": {
            "script_sha256": sha256_file(Path(__file__)),
            "contract": {
                "path": str(Path(contract_path)),
                "sha256": sha256_file(Path(contract_path)),
                "schema_version": contract["schema_version"],
                "study_id": contract["study_id"],
            },
            "train_list": {
                "path": str(Path(train_list)),
                "sha256": sha256_file(Path(train_list)),
                "session_ids": train_ids,
            },
            "validation_list": {
                "path": str(Path(validation_list)),
                "sha256": sha256_file(Path(validation_list)),
                "session_ids": validation_ids,
            },
            "build_manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
            },
            "backbone_weights": backbone_weight_receipt,
            "prediction_sidecar": {
                "path": str(predictions_output),
                "sha256": prediction_sidecar_sha256,
                "compressed_npz": True,
                "independently_validated_before_publication": True,
            },
            "runtime_benchmark": {
                "path": str(Path(benchmark_receipt_path)),
                "sha256": sha256_file(Path(benchmark_receipt_path)),
                "projected_full_runtime_seconds": runtime_benchmark_receipt[
                    "projection"
                ]["projected_full_runtime_seconds"],
                "hard_limit_seconds": MAX_PROJECTED_RUNTIME_SECONDS,
                "scoring_null_benchmark": runtime_benchmark_receipt[
                    "scoring_null_benchmark"
                ],
            },
            "train_shards": list(train_corpus.shard_records),
            "validation_shards": list(validation_corpus.shard_records),
            "known_embargoed_sessions_rejected_before_npz_access": sorted(
                KNOWN_EMBARGOED_SESSION_IDS
            ),
            "untouched_test_used": False,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "resnet_weights": str(ResNet18_Weights.IMAGENET1K_V1),
        },
        "results": results,
        "raw_pair_lift_comparisons": comparisons,
    }
    write_json_exclusive_atomic(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--train-list", type=Path)
    parser.add_argument("--validation-list", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--benchmark-receipt", type=Path)
    parser.add_argument("--offsets", type=parse_offsets, default=DEFAULT_OFFSETS)
    parser.add_argument(
        "--feature-variant",
        action="append",
        choices=FEATURE_VARIANTS,
        dest="variants",
        help="repeat to run a subset; default runs both pair/control families",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--probe-batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--positive-weight-cap", type=float, default=50.0)
    parser.add_argument("--max-materialized-gib", type=float, default=24.0)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        help="repeat for independent fixed probe seeds; default: 0, 1, 2",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    max_materialized_bytes = int(args.max_materialized_gib * GIB)
    if args.benchmark_only:
        run_synthetic_worst_case_benchmark(
            contract_path=args.contract,
            output=args.output,
            device_name=args.device,
            max_materialized_bytes=max_materialized_bytes,
        )
        return
    required = {
        "--data-dir": args.data_dir,
        "--train-list": args.train_list,
        "--validation-list": args.validation_list,
        "--predictions-output": args.predictions_output,
        "--benchmark-receipt": args.benchmark_receipt,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("real run requires " + ", ".join(missing))
    run_probe(
        data_dir=args.data_dir,
        train_list=args.train_list,
        validation_list=args.validation_list,
        contract_path=args.contract,
        output=args.output,
        predictions_output=args.predictions_output,
        benchmark_receipt_path=args.benchmark_receipt,
        offsets=args.offsets,
        variants=args.variants or FEATURE_VARIANTS,
        device_name=args.device,
        encode_batch_size=args.encode_batch_size,
        probe_batch_size=args.probe_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_train_samples=args.max_train_samples,
        positive_weight_cap=args.positive_weight_cap,
        seeds=args.seeds or (0, 1, 2),
        max_materialized_bytes=max_materialized_bytes,
    )


if __name__ == "__main__":
    main()
