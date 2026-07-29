"""Train one matched Study-H oracle-window arm at a fixed endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from experiments.oracle_window_localization import (
    HEAD_NAMES,
    _array_sha256,
    epoch_order,
    sha256_file,
    state_dict_sha256,
)
from experiments.pretrain_dynamics import validate_imagenet_initialization


ARMS = ("h32_q", "h128_g", "h128_q")
RUN_SCHEMA = "madeleine.oracle-window-highres-run.v1"
CHECKPOINT_SCHEMA = "madeleine.oracle-window-highres-checkpoint.v1"
SIDECAR_SCHEMA = "madeleine.oracle-window-highres-predictions.v1"
INDEX_FIELDS = frozenset(
    {
        "session_id",
        "run_index",
        "array_index",
        "engine_frame_idx",
        "head_index",
        "key_index",
        "event_type_index",
        "true_offset",
        "crop_start",
        "block_id",
    }
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if _json(path) != value:
        raise ValueError(f"serialized JSON changed on reload: {path}")


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.npz")
    if os.path.lexists(temporary):
        raise ValueError(f"stale sidecar temporary exists: {temporary}")
    np.savez_compressed(temporary, **arrays)
    with np.load(temporary, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise ValueError("serialized prediction inventory changed")
        for name, expected in arrays.items():
            observed = np.asarray(archive[name])
            if observed.dtype != expected.dtype or not np.array_equal(observed, expected):
                raise ValueError(f"serialized prediction array changed: {name}")
    temporary.replace(path)


class FullResolutionOracleDataset:
    """Memory-mapped exact oracle crops without model-visible metadata."""

    def __init__(self, cache: Path, *, split: str) -> None:
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        self.cache = cache.resolve()
        manifest = _json(self.cache / "manifest.json")
        self.crop_frames = int(manifest["crop_frames"])
        self.width = int(manifest["candidate_width"])
        self.halo = int(manifest["context_halo"])
        if (self.crop_frames, self.width, self.halo) != (32, 16, 8):
            raise ValueError("Study-H oracle geometry changed")
        index_path = self.cache / f"{split}_examples.npz"
        with np.load(index_path, allow_pickle=False) as archive:
            if set(archive.files) != INDEX_FIELDS:
                raise ValueError(f"Study-H index inventory changed: {split}")
            self.metadata = {name: np.asarray(archive[name]) for name in archive.files}
        lengths = {len(value) for value in self.metadata.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise ValueError(f"Study-H index columns are inconsistent: {split}")
        self.session_ids = tuple(str(value) for value in manifest["sessions"])
        self.frames = {
            session_id: np.load(
                self.cache / "frames" / f"{session_id}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            for session_id in self.session_ids
        }
        for session_id, frames in self.frames.items():
            if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (128, 128, 3):
                raise ValueError(f"Study-H frame map changed: {session_id}")
        counts = np.bincount(
            self.metadata["head_index"].astype(np.int64), minlength=len(HEAD_NAMES)
        )
        present = counts > 0
        self.task_weights = np.zeros(len(HEAD_NAMES), dtype=np.float64)
        self.task_weights[present] = len(self) / (int(present.sum()) * counts[present])

    def __len__(self) -> int:
        return int(len(self.metadata["true_offset"]))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        session_id = str(self.metadata["session_id"][index])
        if session_id not in self.frames:
            raise ValueError(f"indexed session is outside the cache: {session_id}")
        crop_start = int(self.metadata["crop_start"][index])
        crop_stop = crop_start + self.crop_frames
        frames = self.frames[session_id]
        if crop_start < 0 or crop_stop > len(frames):
            raise ValueError("indexed crop escaped its frame map")
        rgb = np.array(frames[crop_start:crop_stop], dtype=np.uint8, copy=True)
        head = int(self.metadata["head_index"][index])
        return {
            "rgb": torch.from_numpy(rgb),
            "requested_head": torch.tensor(head, dtype=torch.long),
            "target_offset": torch.tensor(
                int(self.metadata["true_offset"][index]), dtype=torch.long
            ),
            "task_weight": torch.tensor(self.task_weights[head], dtype=torch.float32),
        }


class OracleDatasetView:
    """A deterministic smoke-only view that preserves the parent tensors."""

    def __init__(
        self, dataset: FullResolutionOracleDataset, indices: Sequence[int]
    ) -> None:
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1 or not len(selected):
            raise ValueError("Study-H dataset view must contain at least one row")
        if selected.min() < 0 or selected.max() >= len(dataset):
            raise ValueError("Study-H dataset view index is out of bounds")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Study-H dataset view contains duplicate rows")
        self.parent = dataset
        self.metadata = {
            name: np.asarray(values[selected])
            for name, values in dataset.metadata.items()
        }
        self.indices = selected
        self.task_weights = dataset.task_weights
        self.width = dataset.width

    def __len__(self) -> int:
        return int(len(self.metadata["true_offset"]))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.parent[int(self.indices[index])]


def smoke_subset(
    dataset: FullResolutionOracleDataset, *, maximum: int, require_every_head: bool
) -> OracleDatasetView:
    """Select a stable bounded prefix, seeding one row per head when required."""

    if maximum < 1 or maximum > len(dataset):
        raise ValueError("Study-H smoke limit is outside dataset support")
    selected: list[int] = []
    if require_every_head:
        heads = dataset.metadata["head_index"].astype(np.int64)
        for head in range(len(HEAD_NAMES)):
            rows = np.flatnonzero(heads == head)
            if not len(rows):
                raise ValueError(f"Study-H smoke support lacks head {head}")
            selected.append(int(rows[0]))
        if maximum < len(selected):
            raise ValueError("Study-H validation smoke limit must cover every head")
    present = set(selected)
    for index in range(len(dataset)):
        if len(selected) == maximum:
            break
        if index not in present:
            selected.append(index)
    return OracleDatasetView(dataset, selected)


def collate_indices(
    dataset: FullResolutionOracleDataset | OracleDatasetView,
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rows = [dataset[int(index)] for index in indices]
    return {
        name: torch.stack([row[name] for row in rows]).to(device, non_blocking=True)
        for name in rows[0]
    }


class HighResolutionRegionalLocalizer(nn.Module):
    """Shared spatial-map pair encoder with matched regional/global readouts."""

    def __init__(
        self,
        *,
        token_dim: int = 128,
        temporal_kernel: int = 16,
        imagenet_weights: bool = True,
    ) -> None:
        super().__init__()
        backbone = resnet18(
            weights=(ResNet18_Weights.IMAGENET1K_V1 if imagenet_weights else None)
        )
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        self.pair_projection = nn.Conv2d(256 * 4, token_dim, kernel_size=1)
        self.task_embedding = nn.Embedding(len(HEAD_NAMES), token_dim)
        # Both readout modes use these same tensors. Query mode uses q/k dot
        # products; global mode uses them as a parameter-matched nonspatial MLP.
        self.readout_first = nn.Linear(token_dim, token_dim)
        self.readout_second = nn.Linear(token_dim, token_dim)
        self.pair_norm = nn.LayerNorm(token_dim)
        self.temporal = nn.Conv1d(token_dim, token_dim, kernel_size=temporal_kernel)
        self.temporal_norm = nn.LayerNorm(token_dim)
        self.output = nn.Linear(token_dim, 1)
        self.token_dim = int(token_dim)
        self.temporal_kernel = int(temporal_kernel)
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> "HighResolutionRegionalLocalizer":
        super().train(mode)
        # Preserve ImageNet BatchNorm statistics while allowing weight gradients.
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def _encode_frames(self, rgb: torch.Tensor, *, input_size: int) -> torch.Tensor:
        if rgb.dtype != torch.uint8 or rgb.ndim != 5 or rgb.shape[1:] != (32, 128, 128, 3):
            raise ValueError("rgb must be uint8 [B,32,128,128,3]")
        batch = rgb.shape[0]
        frames = rgb.permute(0, 1, 4, 2, 3).reshape(batch * 32, 3, 128, 128)
        frames = frames.to(dtype=torch.float32).div_(255.0)
        if input_size == 32:
            frames = F.avg_pool2d(frames, kernel_size=4, stride=4)
        elif input_size != 128:
            raise ValueError("Study-H input size must be 32 or 128")
        normalized = (frames - self.image_mean) / self.image_std
        maps = self.backbone(normalized)
        expected_grid = 2 if input_size == 32 else 8
        if maps.shape != (batch * 32, 256, expected_grid, expected_grid):
            raise ValueError("ResNet layer-3 map geometry changed")
        return maps.reshape(batch, 32, 256, expected_grid, expected_grid)

    def forward(
        self,
        rgb: torch.Tensor,
        requested_head: torch.Tensor,
        *,
        input_size: int,
        readout_mode: str,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if requested_head.shape != (rgb.shape[0],):
            raise ValueError("requested_head must have shape [B]")
        if readout_mode not in ("query", "global"):
            raise ValueError("readout_mode must be query or global")
        maps = self._encode_frames(rgb, input_size=input_size)
        previous = maps[:, :-1]
        current = maps[:, 1:]
        pair = torch.cat(
            (previous, current, current - previous, (current - previous).abs()), dim=2
        )
        batch, pairs, channels, height, width = pair.shape
        projected = self.pair_projection(
            pair.reshape(batch * pairs, channels, height, width)
        ).reshape(batch, pairs, self.token_dim, height * width)
        tokens = projected.permute(0, 1, 3, 2)
        task = self.task_embedding(requested_head)
        if readout_mode == "query":
            query = self.readout_first(task)
            keys = self.readout_second(tokens)
            attention = torch.softmax(
                torch.einsum("bd,btnd->btn", query, keys) / self.token_dim**0.5,
                dim=-1,
            )
            pooled = torch.einsum("btn,btnd->btd", attention, tokens)
        else:
            attention = tokens.new_full(
                (batch, pairs, height * width), 1.0 / float(height * width)
            )
            mean = tokens.mean(dim=2) + task[:, None, :]
            pooled = self.readout_second(F.silu(self.readout_first(mean)))
        pair_embedding = self.pair_norm(pooled + task[:, None, :])
        temporal = self.temporal(pair_embedding.transpose(1, 2)).transpose(1, 2)
        if temporal.shape[1] != 16:
            raise ValueError("valid temporal mapping must produce 16 candidates")
        logits = self.output(self.temporal_norm(F.silu(temporal))).squeeze(-1)
        if return_attention:
            return logits, attention
        return logits


def arm_geometry(arm: str) -> tuple[int, str]:
    if arm == "h32_q":
        return 32, "query"
    if arm == "h128_g":
        return 128, "global"
    if arm == "h128_q":
        return 128, "query"
    raise ValueError(f"unknown Study-H arm: {arm}")


def validate_implementation(config: Mapping[str, Any], *, repo: Path) -> dict[str, Any]:
    declaration = config.get("implementation")
    if not isinstance(declaration, Mapping):
        raise ValueError("Study-H contract lacks implementation hashes")
    expected = declaration.get("sha256")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("Study-H contract lacks dependency hashes")
    observed: dict[str, str] = {}
    for relative, digest in expected.items():
        path = repo / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"missing Study-H dependency: {path}")
        observed[str(relative)] = sha256_file(path)
        if observed[str(relative)] != str(digest):
            raise ValueError(f"Study-H dependency hash changed: {relative}")
    git_head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"git_head": git_head, "relevant_file_sha256": observed}


def validate_cache(
    cache: Path, receipt_path: Path, *, expected_receipt_sha256: str
) -> dict[str, Any]:
    if sha256_file(receipt_path) != expected_receipt_sha256:
        raise ValueError("Study-H cache receipt hash changed")
    receipt = _json(receipt_path)
    if receipt.get("status") != "complete":
        raise ValueError("Study-H cache receipt is incomplete")
    published_output = receipt.get("published_output")
    if not isinstance(published_output, str) or not published_output:
        raise ValueError("Study-H cache receipt lacks original publication provenance")
    manifest_path = cache / "manifest.json"
    if sha256_file(manifest_path) != str(receipt.get("manifest_sha256")):
        raise ValueError("Study-H cache manifest hash changed")
    for relative, binding in receipt.get("payload", {}).items():
        path = cache / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Study-H cache payload is missing: {path}")
        if path.stat().st_size != int(binding["bytes"]) or sha256_file(path) != str(binding["sha256"]):
            raise ValueError(f"Study-H cache payload changed: {relative}")
    canonical = dict(receipt)
    content_sha = canonical.pop("content_sha256", None)
    if content_sha != _canonical_sha256(canonical):
        raise ValueError("Study-H cache receipt content hash changed")
    return receipt


def _arm_model(config: Mapping[str, Any], *, seed: int) -> HighResolutionRegionalLocalizer:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_config = config["model"]
    return HighResolutionRegionalLocalizer(
        token_dim=int(model_config["token_dim"]),
        temporal_kernel=int(model_config["temporal_kernel"]),
        imagenet_weights=bool(model_config["imagenet_weights"]),
    )


def train_model(
    model: HighResolutionRegionalLocalizer,
    dataset: FullResolutionOracleDataset | OracleDatasetView,
    *,
    arm: str,
    config: Mapping[str, Any],
    device: torch.device,
    epochs: int,
) -> list[dict[str, float]]:
    training = config["training"]
    seed = int(training["seed"])
    effective_batch = int(training["effective_batch_size"])
    microbatch = int(training["microbatch_size"])
    if effective_batch < microbatch or effective_batch % microbatch:
        raise ValueError("effective batch must be a positive microbatch multiple")
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    backbone = [parameter for parameter in model.parameters() if id(parameter) in backbone_ids]
    new = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": float(training["encoder_learning_rate"])},
            {"params": new, "lr": float(training["new_layer_learning_rate"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    input_size, readout = arm_geometry(arm)
    use_bf16 = device.type == "cuda" and bool(training["cuda_bf16"])
    log: list[dict[str, float]] = []
    model.train()
    for epoch in range(epochs):
        order = epoch_order(len(dataset), seed, epoch)
        epoch_numerator = 0.0
        epoch_weight = 0.0
        for batch_start in range(0, len(order), effective_batch):
            effective_indices = order[batch_start : batch_start + effective_batch]
            effective_weights = np.asarray(
                [
                    dataset.task_weights[
                        int(dataset.metadata["head_index"][int(index)])
                    ]
                    for index in effective_indices
                ],
                dtype=np.float64,
            )
            denominator = float(effective_weights.sum())
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, len(effective_indices), microbatch):
                indices = effective_indices[micro_start : micro_start + microbatch]
                batch = collate_indices(dataset, indices, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    logits = model(
                        batch["rgb"],
                        batch["requested_head"],
                        input_size=input_size,
                        readout_mode=readout,
                    )
                    per_example = F.cross_entropy(
                        logits.float(), batch["target_offset"], reduction="none"
                    )
                    numerator = (per_example * batch["task_weight"]).sum()
                    loss = numerator / denominator
                loss.backward()
                epoch_numerator += float(numerator.detach())
                epoch_weight += float(batch["task_weight"].sum())
            optimizer.step()
        log.append({"epoch": float(epoch + 1), "loss": epoch_numerator / epoch_weight})
    return log


@torch.inference_mode()
def predict_model(
    model: HighResolutionRegionalLocalizer,
    dataset: FullResolutionOracleDataset | OracleDatasetView,
    *,
    arm: str,
    device: torch.device,
    batch_size: int,
    cuda_bf16: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_size, readout = arm_geometry(arm)
    use_bf16 = device.type == "cuda" and cuda_bf16
    probabilities: list[np.ndarray] = []
    entropies: list[np.ndarray] = []
    attention_sum: np.ndarray | None = None
    attention_count = np.zeros(len(HEAD_NAMES), dtype=np.int64)
    model.eval()
    for start in range(0, len(dataset), batch_size):
        indices = np.arange(start, min(start + batch_size, len(dataset)))
        batch = collate_indices(dataset, indices, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            logits, attention = model(
                batch["rgb"],
                batch["requested_head"],
                input_size=input_size,
                readout_mode=readout,
                return_attention=True,
            )
        probability = logits.float().softmax(dim=1)
        probabilities.append(probability.cpu().numpy())
        attention_float = attention.float()
        entropy = -(attention_float.clamp_min(1e-12).log() * attention_float).sum(dim=-1).mean(dim=1)
        entropies.append(entropy.cpu().numpy())
        grid = int(attention.shape[-1] ** 0.5)
        if grid * grid != attention.shape[-1]:
            raise ValueError("Study-H attention token count is not square")
        if attention_sum is None:
            attention_sum = np.zeros((len(HEAD_NAMES), grid, grid), dtype=np.float64)
        mean_pair_attention = attention_float.mean(dim=1).cpu().numpy().reshape(-1, grid, grid)
        for row, head in zip(mean_pair_attention, batch["requested_head"].cpu().numpy(), strict=True):
            attention_sum[int(head)] += row
            attention_count[int(head)] += 1
    prob = np.concatenate(probabilities).astype(np.float32, copy=False)
    spatial_entropy = np.concatenate(entropies).astype(np.float32, copy=False)
    if attention_sum is None:
        raise ValueError("Study-H validation dataset is empty")
    for head in range(len(HEAD_NAMES)):
        if attention_count[head]:
            attention_sum[head] /= attention_count[head]
    attention_mean = attention_sum.astype(np.float32)
    if prob.shape != (len(dataset), dataset.width) or not np.all(np.isfinite(prob)):
        raise ValueError("Study-H probabilities changed shape or finiteness")
    if not np.allclose(prob.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Study-H probabilities are not normalized")
    return prob, spatial_entropy, attention_mean


def run_arm(
    *,
    cache: Path,
    cache_receipt_path: Path,
    config_path: Path,
    output: Path,
    arm: str,
    device_name: str,
    epoch_override: int | None = None,
    smoke: bool = False,
    max_train_examples: int | None = None,
    max_validation_examples: int | None = None,
) -> Path:
    if arm not in ARMS:
        raise ValueError(f"unknown Study-H arm: {arm}")
    if os.path.lexists(output):
        raise ValueError(f"refusing to overwrite Study-H output: {output}")
    staging = output.with_name(f".{output.name}.staging")
    if os.path.lexists(staging):
        raise ValueError(f"stale Study-H staging exists: {staging}")
    config = _json(config_path)
    if config.get("status") != "preregistered_before_validation_inference":
        raise ValueError("Study-H contract is not frozen")
    repo = config_path.resolve().parents[2]
    implementation = validate_implementation(config, repo=repo)
    expected_cache_sha = str(config["dataset"]["cache_receipt_sha256"])
    cache_receipt = validate_cache(
        cache.resolve(), cache_receipt_path.resolve(), expected_receipt_sha256=expected_cache_sha
    )
    train_dataset_full = FullResolutionOracleDataset(cache, split="train")
    val_dataset_full = FullResolutionOracleDataset(cache, split="validation")
    support = config["dataset"]["expected_support"]
    if len(train_dataset_full) != int(support["training_examples"]) or len(val_dataset_full) != int(support["validation_examples"]):
        raise ValueError("Study-H cache support changed")
    if not smoke and (
        epoch_override is not None
        or max_train_examples is not None
        or max_validation_examples is not None
    ):
        raise ValueError("Study-H production endpoint cannot be overridden")
    if smoke:
        if max_train_examples is None or max_validation_examples is None:
            raise ValueError("Study-H smoke requires explicit train and validation limits")
        train_dataset = smoke_subset(
            train_dataset_full,
            maximum=int(max_train_examples),
            require_every_head=False,
        )
        val_dataset = smoke_subset(
            val_dataset_full,
            maximum=int(max_validation_examples),
            require_every_head=True,
        )
    else:
        train_dataset = train_dataset_full
        val_dataset = val_dataset_full

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(device_name)
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=(device.type != "cuda"))
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)
    imagenet = validate_imagenet_initialization()
    model = _arm_model(config, seed=seed)
    initial_hash = state_dict_sha256(model)
    model.to(device)
    configured_epochs = int(config["training"]["epochs"])
    epochs = configured_epochs if epoch_override is None else int(epoch_override)
    if epochs < 1 or epochs > configured_epochs:
        raise ValueError("Study-H epoch override is outside the frozen endpoint")
    started = time.monotonic()
    training_log = train_model(
        model, train_dataset, arm=arm, config=config, device=device, epochs=epochs
    )
    probability, attention_entropy, attention_mean = predict_model(
        model,
        val_dataset,
        arm=arm,
        device=device,
        batch_size=int(config["training"]["eval_batch_size"]),
        cuda_bf16=bool(config["training"]["cuda_bf16"]),
    )
    wall_seconds = time.monotonic() - started

    staging.mkdir(parents=True, exist_ok=False)
    config_sha = sha256_file(config_path)
    checkpoint_path = staging / "model.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "arm": arm,
            "config_sha256": config_sha,
            "cache_receipt_sha256": expected_cache_sha,
            "initial_state_sha256": initial_hash,
            "model_state_sha256": state_dict_sha256(model),
            "seed": seed,
            "epochs": epochs,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    training_log_path = staging / "training_log.json"
    _write_json(
        training_log_path,
        {
            "arm": arm,
            "epochs": training_log,
            "fixed_final_epoch": epochs,
            "configured_final_epoch": configured_epochs,
            "validation_used_for_training_or_selection": False,
        },
    )
    sidecar_path = staging / "predictions.npz"
    sidecar_arrays = {name: value for name, value in val_dataset.metadata.items()}
    sidecar_arrays.update(
        {
            "schema_version": np.asarray(SIDECAR_SCHEMA),
            "arm": np.asarray(arm),
            "probability": probability,
            "spatial_attention_entropy": attention_entropy,
            "attention_mean_by_head": attention_mean,
        }
    )
    _write_npz_atomic(sidecar_path, **sidecar_arrays)
    receipt_base = {
        "schema_version": RUN_SCHEMA,
        "status": (
            "smoke_predictions_complete_unscored"
            if smoke
            else "predictions_complete_unscored"
        ),
        "execution_mode": "smoke" if smoke else "production",
        "arm": arm,
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha,
        "cache_receipt_path": str(cache_receipt_path.resolve()),
        "cache_receipt_sha256": expected_cache_sha,
        "cache_content_sha256": cache_receipt["content_sha256"],
        "implementation": implementation,
        "imagenet_initialization": imagenet,
        "initial_state_sha256": initial_hash,
        "model_state_sha256": state_dict_sha256(model),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_log_sha256": sha256_file(training_log_path),
        "prediction_sidecar_sha256": sha256_file(sidecar_path),
        "seed": seed,
        "epochs": epochs,
        "configured_epochs": configured_epochs,
        "device": str(device),
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "full_train_examples": len(train_dataset_full),
        "full_validation_examples": len(val_dataset_full),
        "matched_batch_order": True,
        "final_weights_only": True,
        "validation_used_for_training_or_selection": False,
        "runtime": {
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024)
            ),
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
        "model": {
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "input_size": arm_geometry(arm)[0],
            "readout_mode": arm_geometry(arm)[1],
        },
    }
    receipt = dict(receipt_base)
    receipt["content_sha256"] = _canonical_sha256(receipt_base)
    _write_json(staging / "run_receipt.json", receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(output)
    return output / "predictions.npz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--cache-receipt", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sidecar = run_arm(
        cache=args.cache,
        cache_receipt_path=args.cache_receipt,
        config_path=args.config,
        output=args.out,
        arm=args.arm,
        device_name=args.device,
        epoch_override=args.epochs,
        smoke=args.smoke,
        max_train_examples=args.max_train_examples,
        max_validation_examples=args.max_validation_examples,
    )
    print(json.dumps({"status": "complete", "arm": args.arm, "predictions": str(sidecar)}))


if __name__ == "__main__":
    main()
