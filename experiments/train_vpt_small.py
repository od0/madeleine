#!/usr/bin/env python3
"""Train VPT-small with exact global-batch DDP and epoch-boundary receipts."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor, distributed as dist, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

from badeline.vpt_small import (
    ClipConsistentAugmentation,
    VPTAugmentationConfig,
    VPTSmallConfig,
    VPTSmallIDM,
    maybe_autocast,
    natural_factored_nll,
    parameter_inventory,
)
from data.schema import KEY_ORDER


RUN_META_SCHEMA = "madeleine.vpt-small-run-meta.v1"
CHECKPOINT_SCHEMA = "madeleine.vpt-small-checkpoint.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def git_commit() -> str:
    declared = os.environ.get("MADELEINE_SOURCE_COMMIT")
    if declared is not None:
        normalized = declared.strip().lower()
        if len(normalized) != 40 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("MADELEINE_SOURCE_COMMIT must be a 40-character SHA-1")
        return normalized
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class VPTWindowDataset(Dataset[dict[str, Tensor]]):
    """Mmap-backed windows declared by one VPT derived-data manifest."""

    def __init__(self, manifest_path: Path, *, cache_streams: int = 8) -> None:
        self.manifest_path = manifest_path.resolve()
        self.root = self.manifest_path.parent
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") not in {
            "madeleine.vpt-small-20hz-shards.v1",
            "madeleine.vpt-small-60hz-shards.v1",
        }:
            raise ValueError(f"unsupported data manifest: {self.manifest_path}")
        self.window = int(self.manifest.get("window", -1))
        self.stride = int(self.manifest.get("stride", -1))
        if (self.window, self.stride) not in {(128, 64), (384, 192)}:
            raise ValueError(
                "VPT-small data manifest must freeze window/stride 128/64 or 384/192"
            )
        self.records = list(self.manifest["records"])
        self.locations: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            directory = self.root / f"{record['session_id']}__p{record['phase']}"
            metadata_path = directory / "metadata.json"
            expected_metadata = record["metadata_file"]
            if sha256_file(metadata_path) != expected_metadata["sha256"]:
                raise RuntimeError(f"metadata hash mismatch: {metadata_path}")
            starts = np.load(directory / "window_start.npy", allow_pickle=False)
            if starts.dtype != np.int64 or starts.ndim != 1:
                raise ValueError(f"invalid window_start: {directory}")
            self.locations.extend((record_index, int(start)) for start in starts)
        if len(self.locations) != int(self.manifest["totals"]["windows"]):
            raise RuntimeError("manifest window total does not match stream indexes")
        if not self.locations:
            raise ValueError("manifest contains no complete VPT windows")
        self.cache_streams = int(cache_streams)
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.locations)

    def _load_stream(self, record_index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.pop(record_index, None)
        if cached is not None:
            self._cache[record_index] = cached
            return cached
        record = self.records[record_index]
        directory = self.root / f"{record['session_id']}__p{record['phase']}"
        frames = np.load(directory / "frames.npy", mmap_mode="r", allow_pickle=False)
        keys = np.load(directory / "keys.npy", mmap_mode="r", allow_pickle=False)
        expected_rows = int(record["derived_rows"])
        if frames.dtype != np.uint8 or frames.shape != (expected_rows, 128, 128, 3):
            raise ValueError(f"invalid frames array: {directory}")
        if keys.dtype != np.uint8 or keys.shape != (expected_rows, len(KEY_ORDER)):
            raise ValueError(f"invalid keys array: {directory}")
        cached = (frames, keys)
        self._cache[record_index] = cached
        while len(self._cache) > self.cache_streams:
            self._cache.popitem(last=False)
        return cached

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record_index, start = self.locations[index]
        frames, keys = self._load_stream(record_index)
        stop = start + self.window
        frame_block = np.array(frames[start:stop], copy=True)
        key_block = np.array(keys[start:stop], copy=True)
        if frame_block.shape[0] != self.window:
            raise RuntimeError("window index crosses a stream boundary")
        return {
            "frames": torch.from_numpy(frame_block).permute(0, 3, 1, 2),
            "target": torch.from_numpy(key_block).to(torch.long),
        }


class EpochRankSampler(Sampler[int]):
    """Pad each epoch to exact global batches, then partition each batch by rank."""

    def __init__(
        self,
        size: int,
        *,
        global_batch: int,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        if size < 1:
            raise ValueError("sampler size must be positive")
        if global_batch % world_size:
            raise ValueError("global batch must be divisible by world size")
        self.size = size
        self.global_batch = global_batch
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.steps = math.ceil(size / global_batch)
        self.local_batch = global_batch // world_size

    @property
    def repeated_per_epoch(self) -> int:
        return self.steps * self.global_batch - self.size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def epoch_indices(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        needed = self.steps * self.global_batch - len(order)
        if needed:
            repeats = (order * math.ceil(needed / len(order)))[:needed]
            order.extend(repeats)
        result: list[int] = []
        offset = self.rank * self.local_batch
        for step in range(self.steps):
            begin = step * self.global_batch + offset
            result.extend(order[begin : begin + self.local_batch])
        return result

    def __iter__(self) -> Iterator[int]:
        return iter(self.epoch_indices())

    def __len__(self) -> int:
        return self.steps * self.local_batch


class RankStrideSampler(Sampler[int]):
    """Partition deterministic validation rows without padding duplicates."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.indices = list(range(rank, size, world_size))

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def unwrap(model: nn.Module) -> VPTSmallIDM:
    return model.module if isinstance(model, DistributedDataParallel) else model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def production_device(local_rank: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def validate_nll(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    world_size: int,
) -> dict[str, Any]:
    evaluation_model = unwrap(model)
    evaluation_model.eval()
    total = torch.zeros(len(KEY_ORDER), device=device, dtype=torch.float64)
    count = torch.zeros(len(KEY_ORDER), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for batch in loader:
            frames = batch["frames"].to(device=device, non_blocking=True).float().div_(255.0)
            target = batch["target"].to(device=device, non_blocking=True)
            with maybe_autocast(device, dtype):
                logits = evaluation_model(frames)
                losses = F.cross_entropy(
                    logits.reshape(-1, 2), target.reshape(-1), reduction="none"
                ).reshape_as(target)
            total += losses.double().sum(dim=(0, 1))
            count += torch.tensor(
                target.shape[0] * target.shape[1], device=device, dtype=torch.float64
            )
    if world_size > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    per_key = total / count
    return {
        "natural_nll": float(per_key.sum().item()),
        "per_key_nll": {
            key: float(per_key[column].item())
            for column, key in enumerate(KEY_ORDER)
        },
        "positions": int(count[0].item()),
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    epoch: int,
    optimizer_step: int,
    best_validation_nll: float,
    best_epoch: int,
    completed_epoch: bool,
    rng_by_rank: list[dict[str, Any]],
    train_manifest_sha256: str,
    val_manifest_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "epoch": epoch,
        "optimizer_step": optimizer_step,
        "best_validation_nll": best_validation_nll,
        "best_epoch": best_epoch,
        "completed_epoch": completed_epoch,
        "config": config,
        "source_commit": git_commit(),
        "train_manifest_sha256": train_manifest_sha256,
        "val_manifest_sha256": val_manifest_sha256,
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_by_rank": rng_by_rank,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "epoch": epoch,
        "optimizer_step": optimizer_step,
    }


def resume_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rank: int,
    device: torch.device,
    augmentation_generator: torch.Generator,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported VPT-small checkpoint")
    if payload.get("completed_epoch") is not True:
        raise ValueError("VPT-small resume requires a completed epoch checkpoint")
    unwrap(model).load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    rng_by_rank = payload.get("rng_by_rank")
    if not isinstance(rng_by_rank, list) or rank >= len(rng_by_rank):
        raise ValueError("checkpoint does not contain the requested DDP rank RNG state")
    rng = rng_by_rank[rank]
    torch.set_rng_state(rng["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(rng["cuda_rng"], device=device)
    np.random.set_state(rng["numpy_rng"])
    random.setstate(rng["python_rng"])
    augmentation_generator.set_state(rng["augmentation_generator_rng"])
    return payload


def initialize_model_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    expected_sha256: str,
    expected_model_config: dict[str, Any],
) -> dict[str, Any]:
    """Load model weights only from one hash-bound completed checkpoint."""

    normalized_sha256 = expected_sha256.strip().lower()
    if len(normalized_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_sha256
    ):
        raise ValueError("initialize-from SHA-256 must be 64 lowercase hex characters")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != normalized_sha256:
        raise ValueError(
            f"initialize-from checkpoint SHA-256 mismatch: {observed_sha256}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported VPT-small initialize-from checkpoint")
    if payload.get("completed_epoch") is not True:
        raise ValueError(
            "VPT-small initialize-from requires a completed epoch checkpoint"
        )
    source_config = payload.get("config")
    if (
        not isinstance(source_config, dict)
        or source_config.get("model") != expected_model_config
    ):
        raise ValueError("initialize-from checkpoint model config differs")
    unwrap(model).load_state_dict(payload["model"])
    return {
        "path": str(path.resolve()),
        "sha256": observed_sha256,
        "source_commit": payload.get("source_commit"),
        "epoch": int(payload["epoch"]),
        "optimizer_step": int(payload["optimizer_step"]),
        "train_manifest_sha256": payload.get("train_manifest_sha256"),
        "val_manifest_sha256": payload.get("val_manifest_sha256"),
    }


def capture_rng_state(
    *, device: torch.device, augmentation_generator: torch.Generator
) -> dict[str, Any]:
    """Capture every random stream that can affect a resumed training epoch."""

    return {
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state(device=device)
            if device.type == "cuda"
            else torch.empty(0, dtype=torch.uint8)
        ),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "augmentation_generator_rng": augmentation_generator.get_state(),
    }


def gather_rng_states(
    local_state: dict[str, Any], *, world_size: int
) -> list[dict[str, Any]]:
    """Return rank-ordered RNG states on every rank for one shared checkpoint."""

    if world_size == 1:
        return [local_state]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_state)
    if any(state is None for state in gathered):
        raise RuntimeError("failed to gather all DDP rank RNG states")
    return [state for state in gathered if state is not None]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--initialize-from-sha256")
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--stop-after-optimizer-steps", type=int)
    parser.add_argument("--microbatch", type=int)
    return parser.parse_args(argv)


def resolve_optimizer_endpoints(
    production_steps: int,
    *,
    max_optimizer_steps: int | None,
    stop_after_optimizer_steps: int | None,
) -> tuple[int, int]:
    """Return the scheduler endpoint and this invocation's stop point."""

    scheduler_endpoint = min(production_steps, max_optimizer_steps or production_steps)
    stop_endpoint = min(
        scheduler_endpoint, stop_after_optimizer_steps or scheduler_endpoint
    )
    if scheduler_endpoint < 1 or stop_endpoint < 1:
        raise ValueError("optimizer endpoints must be positive")
    return scheduler_endpoint, stop_endpoint


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume is not None and args.initialize_from is not None:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
    if (args.initialize_from is None) != (args.initialize_from_sha256 is None):
        raise ValueError(
            "--initialize-from and --initialize-from-sha256 must be provided together"
        )
    if args.out.exists() and any(args.out.iterdir()) and args.resume is None:
        raise FileExistsError(f"refusing nonempty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_config = VPTSmallConfig.from_dict(config["model"])
    augmentation_config = VPTAugmentationConfig.from_dict(config["augmentation"])
    training = config["training"]
    run_kind = str(training.get("run_kind", "production"))
    if run_kind not in {"production", "finetune"}:
        raise ValueError("training.run_kind must be production or finetune")
    if training["optimizer"] != "adam" or training["loss"] != "natural_factored_nll":
        raise ValueError("VPT-small production path requires Adam and natural NLL")
    if (
        run_kind == "production"
        and int(training["epochs"]) != 20
        and args.max_optimizer_steps is None
    ):
        raise ValueError("production VPT-small requires exactly 20 epochs")
    if run_kind == "finetune":
        if args.initialize_from is None and args.resume is None:
            raise ValueError("fine-tuning requires --initialize-from or --resume")
        if int(training["epochs"]) < 1:
            raise ValueError("fine-tuning epochs must be positive")
    elif args.initialize_from is not None:
        raise ValueError("--initialize-from requires training.run_kind=finetune")

    rank, local_rank, world_size = setup_distributed()
    device = production_device(local_rank)
    seed = int(training["seed"])
    seed_everything(seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(training.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(training.get("allow_tf32", True))

    train_dataset = VPTWindowDataset(args.train_manifest)
    val_dataset = VPTWindowDataset(args.val_manifest)
    if train_dataset.window != model_config.frames or val_dataset.window != model_config.frames:
        raise ValueError("model frame count differs from a data-manifest window")
    if (train_dataset.window, train_dataset.stride) != (
        val_dataset.window,
        val_dataset.stride,
    ):
        raise ValueError("train and validation window geometry differs")
    global_batch = int(training["global_batch"])
    microbatch = int(args.microbatch or training["microbatch"])
    if global_batch % (world_size * microbatch):
        raise ValueError("global batch must divide world_size * microbatch exactly")
    accumulation = global_batch // (world_size * microbatch)
    sampler = EpochRankSampler(
        len(train_dataset),
        global_batch=global_batch,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )
    val_sampler = RankStrideSampler(len(val_dataset), rank, world_size)
    workers = int(training["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=microbatch,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training["eval_batch_size"]),
        sampler=val_sampler,
        num_workers=max(0, workers // 2),
        pin_memory=device.type == "cuda",
    )

    model = VPTSmallIDM(model_config).to(device)
    inventory = parameter_inventory(model)
    if not 70_000_000 <= inventory["total"] <= 120_000_000:
        raise RuntimeError(f"parameter count outside preregistered range: {inventory['total']}")
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    augmentation = ClipConsistentAugmentation(augmentation_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(training["betas"]),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    full_exposure_steps = sampler.steps * int(training["epochs"])
    declared_endpoint = training.get("endpoint_optimizer_steps")
    production_steps = (
        int(declared_endpoint) if declared_endpoint is not None else full_exposure_steps
    )
    if production_steps <= 0 or production_steps > full_exposure_steps:
        raise ValueError("declared optimizer endpoint is outside the 20-epoch population")
    endpoint, stop_endpoint = resolve_optimizer_endpoints(
        production_steps,
        max_optimizer_steps=args.max_optimizer_steps,
        stop_after_optimizer_steps=args.stop_after_optimizer_steps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: max(0.0, 1.0 - step / endpoint)
    )
    generator = torch.Generator(device=device).manual_seed(seed + 10_000 + rank)
    start_epoch = 0
    optimizer_step = 0
    best_validation_nll = math.inf
    best_epoch = -1
    initialization: dict[str, Any] | None = None
    if args.resume is not None:
        resumed = resume_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=rank,
            device=device,
            augmentation_generator=generator,
        )
        start_epoch = int(resumed["epoch"])
        optimizer_step = int(resumed["optimizer_step"])
        best_validation_nll = float(resumed["best_validation_nll"])
        best_epoch = int(resumed["best_epoch"])
    elif args.initialize_from is not None:
        initialization = initialize_model_checkpoint(
            args.initialize_from,
            model=model,
            expected_sha256=str(args.initialize_from_sha256),
            expected_model_config=config["model"],
        )

    dtype = torch.bfloat16 if training["precision"] == "bf16" else torch.float32
    run_meta = {
        "schema_version": RUN_META_SCHEMA,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(),
        "config": config,
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": train_dataset.manifest_sha256,
        "val_manifest": str(args.val_manifest.resolve()),
        "val_manifest_sha256": val_dataset.manifest_sha256,
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "world_size": world_size,
        "global_batch": global_batch,
        "microbatch_per_rank": microbatch,
        "accumulation_steps": accumulation,
        "optimizer_steps_per_epoch": sampler.steps,
        "repeated_windows_per_epoch": sampler.repeated_per_epoch,
        "production_optimizer_steps": production_steps,
        "full_exposure_optimizer_steps": full_exposure_steps,
        "actual_endpoint": endpoint,
        "stop_after_optimizer_steps": stop_endpoint,
        "parameter_inventory": inventory,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "resume": str(args.resume) if args.resume else None,
        "initialize_from": initialization,
    }
    if rank == 0:
        atomic_json(args.out / "run_meta.json", run_meta)
    if world_size > 1:
        dist.barrier()

    history_path = args.out / "training_history.json"
    history: list[dict[str, Any]] = []
    if args.resume is not None and history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if len(history) != start_epoch:
            raise RuntimeError("resume history does not end at the checkpoint epoch")
    training_started = time.monotonic()
    stop = False
    final_checkpoint_record: dict[str, Any] | None = None
    for epoch in range(start_epoch, int(training["epochs"])):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_microbatches = 0
        for micro_index, batch in enumerate(train_loader):
            final_accumulation = (micro_index + 1) % accumulation == 0
            synchronize = nullcontext()
            if isinstance(model, DistributedDataParallel) and not final_accumulation:
                synchronize = model.no_sync()
            frames = batch["frames"].to(device=device, non_blocking=True).float().div_(255.0)
            targets = batch["target"].to(device=device, non_blocking=True)
            frames = augmentation(frames, generator=generator)
            with synchronize:
                with maybe_autocast(device, dtype):
                    logits = model(frames)
                    loss = natural_factored_nll(logits, targets) / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {optimizer_step}")
                loss.backward()
            epoch_loss += float(loss.detach().item()) * accumulation
            epoch_microbatches += 1
            if not final_accumulation:
                continue
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            if rank == 0 and optimizer_step % int(training["log_interval"]) == 0:
                elapsed = time.monotonic() - training_started
                record = {
                    "optimizer_step": optimizer_step,
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "mean_recent_nll": epoch_loss / epoch_microbatches,
                    "elapsed_seconds": elapsed,
                    "sequences_per_second": optimizer_step * global_batch / elapsed,
                    "peak_vram_bytes": (
                        torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else 0
                    ),
                }
                with (args.out / "train.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            if optimizer_step >= stop_endpoint:
                stop = True
                break

        completed_epoch = not stop or optimizer_step == (epoch + 1) * sampler.steps
        validation = validate_nll(
            model, val_loader, device=device, dtype=dtype, world_size=world_size
        )
        validation_nll = float(validation["natural_nll"])
        improved = validation_nll < best_validation_nll
        if improved:
            best_validation_nll = validation_nll
            best_epoch = epoch + 1
        rng_by_rank = gather_rng_states(
            capture_rng_state(
                device=device, augmentation_generator=generator
            ),
            world_size=world_size,
        )
        if rank == 0:
            history_row = {
                "epoch": epoch + 1,
                "completed_epoch": completed_epoch,
                "optimizer_step": optimizer_step,
                "train_nll_mean_microbatch": epoch_loss / max(1, epoch_microbatches),
                "validation": validation,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(history_row)
            checkpoint_name = (
                f"checkpoint_epoch_{epoch + 1:02d}.pt"
                if completed_epoch
                else f"checkpoint_step_{optimizer_step:08d}.pt"
            )
            checkpoint_record = save_checkpoint(
                args.out / checkpoint_name,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch + 1,
                optimizer_step=optimizer_step,
                best_validation_nll=best_validation_nll,
                best_epoch=best_epoch,
                completed_epoch=completed_epoch,
                rng_by_rank=rng_by_rank,
                train_manifest_sha256=train_dataset.manifest_sha256,
                val_manifest_sha256=val_dataset.manifest_sha256,
            )
            if improved:
                atomic_json(args.out / "best_checkpoint.json", checkpoint_record)
            atomic_json(history_path, history)
            final_checkpoint_record = checkpoint_record
        if world_size > 1:
            dist.barrier()
        if stop:
            break

    if rank == 0:
        if final_checkpoint_record is None:
            raise RuntimeError("training produced no checkpoint")
        elapsed = time.monotonic() - training_started
        completion = {
            "schema_version": "madeleine.vpt-small-training-complete.v1",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "completed_production_endpoint": optimizer_step == production_steps,
            "optimizer_steps": optimizer_step,
            "production_optimizer_steps": production_steps,
            "full_exposure_optimizer_steps": full_exposure_steps,
            "epochs_recorded": len(history),
            "best_epoch": best_epoch,
            "best_validation_nll": best_validation_nll,
            "elapsed_seconds": elapsed,
            "mean_sequences_per_second": optimizer_step * global_batch / max(elapsed, 1e-9),
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "final_checkpoint": final_checkpoint_record,
        }
        atomic_json(args.out / "complete.json", completion)
        print(json.dumps(completion, sort_keys=True))
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
