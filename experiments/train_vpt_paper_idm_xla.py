#!/usr/bin/env python3
"""Train the released-artifact VPT paper IDM with PyTorch/XLA SPMD FSDP.

The module keeps XLA imports inside ``main`` so recipe and sampler invariants
remain unit-testable on machines without libtpu.  Production is deliberately
single-process SPMD: the batch and model state are partitioned by GSPMD, while
one host-side RNG stream defines the exact global sample/augmentation order.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from badeline.vpt_paper_idm import (
    VPTPaperIDM,
    VPTPaperIDMConfig,
    natural_factored_nll,
    parameter_inventory,
)
from badeline.vpt_small import VPTAugmentationConfig
from badeline.vpt_xla import (
    apply_clip_consistent_augmentation,
    sample_clip_augmentation_parameters,
)
from data.schema import KEY_ORDER
from experiments.train_vpt_small import VPTWindowDataset


RUN_META_SCHEMA = "madeleine.vpt-paper-idm-xla-run-meta.v1"
CHECKPOINT_SCHEMA = "madeleine.vpt-paper-idm-xla-checkpoint.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def git_commit() -> str:
    transported = os.environ.get("MADELEINE_SOURCE_COMMIT")
    if transported is not None:
        normalized = transported.strip().lower()
        if len(normalized) != 40 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("MADELEINE_SOURCE_COMMIT must be a full SHA-1")
        return normalized
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def padded_epoch_order(
    size: int, *, global_batch: int, seed: int, epoch: int
) -> list[int]:
    """Return the exact padded global permutation for one epoch."""

    if size < 1 or global_batch < 1:
        raise ValueError("size and global_batch must be positive")
    generator = torch.Generator().manual_seed(int(seed) + int(epoch))
    order = torch.randperm(size, generator=generator).tolist()
    needed = math.ceil(size / global_batch) * global_batch - size
    if needed:
        order.extend((order * math.ceil(needed / len(order)))[:needed])
    return order


class GlobalEpochSampler(Sampler[int]):
    def __init__(self, size: int, *, global_batch: int, seed: int) -> None:
        self.size = int(size)
        self.global_batch = int(global_batch)
        self.seed = int(seed)
        self.epoch = 0

    @property
    def steps(self) -> int:
        return math.ceil(self.size / self.global_batch)

    @property
    def repeated_per_epoch(self) -> int:
        return self.steps * self.global_batch - self.size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        return iter(
            padded_epoch_order(
                self.size,
                global_batch=self.global_batch,
                seed=self.seed,
                epoch=self.epoch,
            )
        )

    def __len__(self) -> int:
        return self.steps * self.global_batch


class PaddedEvaluationDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, base: Dataset[dict[str, Tensor]], *, batch_size: int) -> None:
        if len(base) < 1:
            raise ValueError("evaluation dataset must not be empty")
        self.base = base
        self.total = math.ceil(len(base) / batch_size) * batch_size

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        result = dict(self.base[index % len(self.base)])
        result["valid_sequence"] = torch.tensor(index < len(self.base))
        return result


class SyntheticWindowDataset(Dataset[dict[str, Tensor]]):
    """Static deterministic fixtures for the v6e-1 compile gate."""

    def __init__(self, size: int, *, seed: int) -> None:
        self.size = int(size)
        generator = torch.Generator().manual_seed(seed)
        self.frames = torch.randint(
            0,
            256,
            (self.size, 128, 3, 128, 128),
            dtype=torch.uint8,
            generator=generator,
        )
        self.targets = torch.randint(
            0,
            2,
            (self.size, 128, len(KEY_ORDER)),
            dtype=torch.long,
            generator=generator,
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {"frames": self.frames[index], "target": self.targets[index]}


def validate_recipe(config: dict[str, Any], *, qualification: bool) -> None:
    model = VPTPaperIDMConfig.from_dict(config["model"])
    if model.to_dict() != config["model"]:
        raise ValueError("model config is not normalized")
    training = config["training"]
    expected = {
        "epochs": 20,
        "global_batch": 128,
        "learning_rate": 0.003,
        "weight_decay": 0.01,
        "optimizer": "adam_coupled_weight_decay",
        "schedule": "linear_to_zero",
        "loss": "natural_factored_nll",
        "precision": "bf16",
        "seed": 0,
        "production_optimizer_steps": 2340,
    }
    for key, value in expected.items():
        if training.get(key) != value:
            raise ValueError(f"recipe mismatch for {key}: {training.get(key)!r}")
    if not qualification and config["xla"]["production_accelerator_type"] != "v6e-4":
        raise ValueError("matched production is frozen to v6e-4")


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    *,
    initial_lr: float,
    optimizer_step: int,
    endpoint: int,
) -> float:
    value = initial_lr * max(0.0, 1.0 - optimizer_step / endpoint)
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def mark_empty_parameters_replicated(
    model: nn.Module, *, mesh: Any, xs: Any
) -> list[str]:
    """Keep inert released-artifact tensors out of FSDPv2 partitioning.

    The released IDM state contains two ``[10, 0]`` relative-attention
    parameters. They contribute zero trainable scalars and are algebraically
    inert, but multi-device FSDPv2 attempts to pad every unmarked parameter
    before sharding it. XLA cannot pad a zero-length dimension. Marking only
    these empty tensors replicated preserves the exact state-dict keys and
    parameter inventory while FSDPv2 partitions every non-empty tensor.
    """

    marked: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.numel() == 0:
            xs.mark_sharding(parameter, mesh, tuple(None for _ in parameter.shape))
            marked.append(name)
    return marked


def tensor_collection_stats(tensors: Iterable[Tensor]) -> dict[str, float | bool]:
    """Synchronously summarize a tensor collection for bounded qualification."""

    present = [tensor.detach() for tensor in tensors if tensor.numel()]
    if not present:
        raise ValueError("tensor diagnostics require at least one non-empty tensor")
    finite = torch.stack([torch.isfinite(tensor).all() for tensor in present]).all()
    maximum = torch.stack([tensor.abs().max().float() for tensor in present]).max()
    values = torch.stack([finite.float(), maximum]).cpu().tolist()
    return {"finite": bool(values[0]), "max_abs": float(values[1])}


def _move_and_shard_batch(
    batch: dict[str, Tensor],
    *,
    device: torch.device,
    mesh: Any,
    xs: Any,
    augmentation_config: VPTAugmentationConfig,
    augmentation_generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    frames = batch["frames"].to(device=device).float().div_(255.0)
    targets = batch["target"].to(device=device)
    parameters = sample_clip_augmentation_parameters(
        frames.shape[0], augmentation_config, generator=augmentation_generator
    )
    xs.mark_sharding(frames, mesh, ("fsdp", None, None, None, None))
    xs.mark_sharding(targets, mesh, ("fsdp", None, None))
    for name, value in tuple(parameters.items()):
        parameters[name] = value.to(device=device)
        spec = ("fsdp", None, None) if value.ndim == 3 else ("fsdp",)
        xs.mark_sharding(parameters[name], mesh, spec)
    return apply_clip_consistent_augmentation(frames, parameters), targets


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    augmentation_generator: torch.Generator,
    epoch: int,
    optimizer_step: int,
    endpoint: int,
    config_sha256: str,
    train_manifest_sha256: str,
    val_manifest_sha256: str | None,
    sample_order_sha256: str,
) -> dict[str, Any]:
    import torch.distributed.checkpoint as dist_cp
    import torch_xla.experimental.distributed_checkpoint as xc

    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    temporary.mkdir(parents=True)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    dist_cp.save(
        state_dict=state,
        storage_writer=dist_cp.FileSystemWriter(temporary / "state"),
        planner=xc.SPMDSavePlanner(),
    )
    torch.save(
        {
            "torch_rng": torch.get_rng_state(),
            "augmentation_generator_rng": augmentation_generator.get_state(),
        },
        temporary / "rng.pt",
    )
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(),
        "config_sha256": config_sha256,
        "train_manifest_sha256": train_manifest_sha256,
        "val_manifest_sha256": val_manifest_sha256,
        "completed_epoch": True,
        "epoch": epoch,
        "optimizer_step": optimizer_step,
        "production_optimizer_steps": endpoint,
        "sample_order_sha256": sample_order_sha256,
    }
    atomic_json(temporary / "manifest.json", manifest)
    os.replace(temporary, path)
    return manifest


def _load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    augmentation_generator: torch.Generator,
    expected_config_sha256: str,
    expected_train_manifest_sha256: str,
) -> dict[str, Any]:
    import torch.distributed.checkpoint as dist_cp
    import torch_xla.experimental.distributed_checkpoint as xc

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported checkpoint schema")
    if manifest.get("completed_epoch") is not True:
        raise ValueError("resume requires a completed epoch")
    if manifest.get("config_sha256") != expected_config_sha256:
        raise ValueError("resume config hash mismatch")
    if manifest.get("train_manifest_sha256") != expected_train_manifest_sha256:
        raise ValueError("resume train-manifest hash mismatch")
    xc.prime_optimizer(optimizer)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    dist_cp.load(
        state_dict=state,
        storage_reader=dist_cp.FileSystemReader(path / "state"),
        planner=xc.SPMDLoadPlanner(),
    )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    rng = torch.load(path / "rng.pt", map_location="cpu", weights_only=True)
    torch.set_rng_state(rng["torch_rng"])
    augmentation_generator.set_state(rng["augmentation_generator_rng"])
    return manifest


def _validate_nll(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    *,
    device: torch.device,
    mesh: Any,
    xs: Any,
    xm: Any,
) -> dict[str, Any]:
    model.eval()
    totals = torch.zeros(len(KEY_ORDER), device=device, dtype=torch.float64)
    counts = torch.zeros(len(KEY_ORDER), device=device, dtype=torch.float64)
    # torch.inference_mode() is incompatible with XLA FSDP's parameter
    # materialization because the wrapper updates tensor version counters.
    # no_grad() preserves identical validation semantics without creating
    # inference tensors.
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device=device).float().div_(255.0)
            targets = batch["target"].to(device=device)
            valid = batch["valid_sequence"].to(device=device)
            xs.mark_sharding(frames, mesh, ("fsdp", None, None, None, None))
            xs.mark_sharding(targets, mesh, ("fsdp", None, None))
            xs.mark_sharding(valid, mesh, ("fsdp",))
            with torch.autocast("xla", dtype=torch.bfloat16):
                logits = model(frames)
                losses = F.cross_entropy(
                    logits.reshape(-1, 2), targets.reshape(-1), reduction="none"
                ).reshape_as(targets)
            supported = valid[:, None, None].expand_as(targets)
            totals += (losses.double() * supported).sum(dim=(0, 1))
            counts += supported.sum(dim=(0, 1)).double()
            xm.mark_step()
    per_key = totals / counts
    values = per_key.cpu().tolist()
    return {
        "natural_nll": float(sum(values)),
        "per_key_nll": dict(zip(KEY_ORDER, map(float, values))),
        "positions": int(counts[0].cpu().item()),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--qualification-windows", type=int)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--numerical-diagnostic", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_recipe(config, qualification=args.qualification)
    if args.synthetic and not args.qualification:
        raise ValueError("synthetic data is qualification-only")
    if args.max_optimizer_steps is not None and not args.qualification:
        raise ValueError("production endpoint cannot be shortened")
    if args.qualification_windows is not None and not args.qualification:
        raise ValueError("qualification window limits are smoke-only")
    if args.numerical_diagnostic and not args.qualification:
        raise ValueError("numerical diagnostics are qualification-only")
    if args.out.exists() and any(args.out.iterdir()) and args.resume is None:
        raise FileExistsError(f"refusing nonempty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.distributed.xla_backend  # noqa: F401; registers xla://
    import torch_xla.runtime as xr
    import torch.distributed as dist
    from torch_xla.distributed.spmd import Mesh
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        SpmdFullyShardedDataParallel as FSDPv2,
    )

    xr.use_spmd()
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo",
            init_method="xla://",
            rank=xr.process_index(),
            world_size=xr.process_count(),
        )
    device = xm.xla_device()
    device_count = xr.global_runtime_device_count()
    mesh = Mesh(
        np.arange(device_count), (device_count, 1), ("fsdp", "model")
    )
    if args.microbatch % device_count:
        raise ValueError("global microbatch must be divisible by TPU device count")

    training = config["training"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    augmentation_generator = torch.Generator().manual_seed(seed + 10_000)
    model = VPTPaperIDM(VPTPaperIDMConfig.from_dict(config["model"]))
    inventory = parameter_inventory(model)
    if inventory["total"] != 482_133_390:
        raise RuntimeError(f"paper-IDM parameter mismatch: {inventory['total']}")
    model = model.to(device)
    empty_parameters = mark_empty_parameters_replicated(model, mesh=mesh, xs=xs)
    if empty_parameters != [
        "net.recurrent_layer.blocks.0.r.orc_block.b_nd",
        "net.recurrent_layer.blocks.1.r.orc_block.b_nd",
    ]:
        raise RuntimeError(
            f"released-artifact empty-parameter inventory changed: {empty_parameters}"
        )
    model = FSDPv2(model, mesh=mesh)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(training["betas"]),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )

    if args.synthetic:
        synthetic_size = args.qualification_windows or int(training["global_batch"])
        train_dataset: Dataset[dict[str, Tensor]] = SyntheticWindowDataset(
            synthetic_size, seed=seed + 71
        )
        train_manifest_sha256 = "synthetic:" + canonical_sha256(
            {"size": synthetic_size, "seed": seed + 71}
        )
    else:
        if args.train_manifest is None:
            raise ValueError("--train-manifest is required for real data")
        real_dataset = VPTWindowDataset(args.train_manifest)
        train_manifest_sha256 = real_dataset.manifest_sha256
        expected_manifest = config["data"]["train_manifest_sha256"]
        if train_manifest_sha256 != expected_manifest:
            raise ValueError("train manifest hash differs from frozen config")
        if args.qualification_windows is not None:
            if not 1 <= args.qualification_windows <= len(real_dataset):
                raise ValueError("invalid qualification window count")
            train_dataset = Subset(real_dataset, range(args.qualification_windows))
        else:
            train_dataset = real_dataset

    global_batch = int(training["global_batch"])
    if global_batch % args.microbatch:
        raise ValueError("global batch must be divisible by global microbatch")
    accumulation = global_batch // args.microbatch
    sampler = GlobalEpochSampler(
        len(train_dataset), global_batch=global_batch, seed=seed
    )
    workers = int(training["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.microbatch,
        sampler=sampler,
        num_workers=workers if not args.synthetic else 0,
        persistent_workers=workers > 0 and not args.synthetic,
        drop_last=True,
    )
    production_endpoint = int(training["production_optimizer_steps"])
    if not args.qualification:
        if sampler.steps != int(training["optimizer_steps_per_epoch"]):
            raise ValueError("production sampler steps differ from frozen endpoint")
        if sampler.repeated_per_epoch != int(training["repeated_windows_per_epoch"]):
            raise ValueError("production sampler repeats differ from frozen endpoint")
        endpoint = production_endpoint
        epochs = int(training["epochs"])
    else:
        endpoint = args.max_optimizer_steps or sampler.steps
        if endpoint % sampler.steps:
            raise ValueError("qualification endpoint must end on an epoch boundary")
        epochs = endpoint // sampler.steps

    val_manifest_sha256: str | None = None
    val_loader: DataLoader[dict[str, Tensor]] | None = None
    if not args.skip_validation:
        if args.val_manifest is None:
            raise ValueError("--val-manifest is required unless validation is skipped")
        val_dataset = VPTWindowDataset(args.val_manifest)
        val_manifest_sha256 = val_dataset.manifest_sha256
        if val_manifest_sha256 != config["data"]["val_manifest_sha256"]:
            raise ValueError("validation manifest hash differs from frozen config")
        eval_batch = int(training["eval_global_microbatch"])
        if eval_batch % device_count:
            raise ValueError("evaluation batch must be divisible by device count")
        val_loader = DataLoader(
            PaddedEvaluationDataset(val_dataset, batch_size=eval_batch),
            batch_size=eval_batch,
            shuffle=False,
            num_workers=max(0, workers // 2),
            drop_last=True,
        )

    config_sha256 = sha256_file(args.config)
    start_epoch = 0
    optimizer_step = 0
    if args.resume is not None:
        resumed = _load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            augmentation_generator=augmentation_generator,
            expected_config_sha256=config_sha256,
            expected_train_manifest_sha256=train_manifest_sha256,
        )
        start_epoch = int(resumed["epoch"])
        optimizer_step = int(resumed["optimizer_step"])
        if optimizer_step != start_epoch * sampler.steps:
            raise ValueError("resume checkpoint does not end at this epoch boundary")

    run_meta = {
        "schema_version": RUN_META_SCHEMA,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(),
        "config_sha256": config_sha256,
        "config": config,
        "qualification": args.qualification,
        "synthetic": args.synthetic,
        "train_manifest_sha256": train_manifest_sha256,
        "val_manifest_sha256": val_manifest_sha256,
        "train_windows": len(train_dataset),
        "global_batch": global_batch,
        "global_microbatch": args.microbatch,
        "accumulation_steps": accumulation,
        "optimizer_steps_per_epoch": sampler.steps,
        "repeated_windows_per_epoch": sampler.repeated_per_epoch,
        "endpoint": endpoint,
        "parameter_inventory": inventory,
        "replicated_empty_parameters": empty_parameters,
        "xla_devices": device_count,
        "torch": torch.__version__,
        "torch_xla": getattr(__import__("torch_xla"), "__version__", "unknown"),
    }
    atomic_json(args.out / "run_meta.json", run_meta)

    history_path = args.out / "training_history.json"
    history: list[dict[str, Any]] = []
    if args.resume is not None and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    started = time.monotonic()
    sample_order_digest = hashlib.sha256()
    for prior_epoch in range(start_epoch):
        prior_order = padded_epoch_order(
            len(train_dataset),
            global_batch=global_batch,
            seed=seed,
            epoch=prior_epoch,
        )
        sample_order_digest.update(
            np.asarray(prior_order, dtype="<i8").tobytes()
        )
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        epoch_order = padded_epoch_order(
            len(train_dataset), global_batch=global_batch, seed=seed, epoch=epoch
        )
        sample_order_digest.update(np.asarray(epoch_order, dtype="<i8").tobytes())
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        microbatches = 0
        step_loss: Tensor | None = None
        for micro_index, batch in enumerate(train_loader):
            frames, targets = _move_and_shard_batch(
                batch,
                device=device,
                mesh=mesh,
                xs=xs,
                augmentation_config=VPTAugmentationConfig.from_dict(
                    config["augmentation"]
                ),
                augmentation_generator=augmentation_generator,
            )
            with torch.autocast("xla", dtype=torch.bfloat16):
                logits = model(frames)
                loss = natural_factored_nll(logits, targets) / accumulation
            loss.backward()
            detached = loss.detach() * accumulation
            step_loss = detached if step_loss is None else step_loss + detached
            microbatches += 1
            final_accumulation = (micro_index + 1) % accumulation == 0
            if final_accumulation:
                # Materialize the complete accumulated backward graph before
                # tracing Adam. Without this boundary, XLA 2.6 links the next
                # lazy backward graph through optimizer state and produces a
                # nonfinite third-step loss on v6e SPMD. This is an execution
                # barrier only; it does not change gradients or optimizer math.
                xm.mark_step()
                if args.numerical_diagnostic:
                    if step_loss is None:
                        raise RuntimeError("missing accumulated step loss")
                    pre_update = {
                        "loss": float(step_loss.cpu().item()) / accumulation,
                        "parameters": tensor_collection_stats(model.parameters()),
                        "gradients": tensor_collection_stats(
                            parameter.grad
                            for parameter in model.parameters()
                            if parameter.grad is not None
                        ),
                    }
                learning_rate = set_optimizer_lr(
                    optimizer,
                    initial_lr=float(training["learning_rate"]),
                    optimizer_step=optimizer_step,
                    endpoint=production_endpoint,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            xm.mark_step()
            if final_accumulation:
                if step_loss is None:
                    raise RuntimeError("missing accumulated step loss")
                step_loss_value = float(step_loss.cpu().item()) / accumulation
                if args.numerical_diagnostic:
                    optimizer_tensors = [
                        value
                        for state in optimizer.state.values()
                        for value in state.values()
                        if isinstance(value, Tensor) and value.device.type == "xla"
                    ]
                    post_update = {
                        "parameters": tensor_collection_stats(model.parameters()),
                        "optimizer": tensor_collection_stats(optimizer_tensors),
                    }
                    with (args.out / "numerical_diagnostic.jsonl").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "optimizer_step": optimizer_step,
                                    "learning_rate": learning_rate,
                                    "pre_update": pre_update,
                                    "post_update": post_update,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    if not post_update["parameters"]["finite"]:
                        raise FloatingPointError(
                            f"nonfinite parameter after step {optimizer_step}"
                        )
                if not math.isfinite(step_loss_value):
                    raise FloatingPointError(
                        f"nonfinite loss at step {optimizer_step}"
                    )
                loss_total += step_loss_value * accumulation
                step_loss = None
            if (
                final_accumulation
                and optimizer_step % int(training["log_interval"]) == 0
            ):
                elapsed = time.monotonic() - started
                with (args.out / "train.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "optimizer_step": optimizer_step,
                                "epoch": epoch + 1,
                                "learning_rate": learning_rate,
                                "mean_epoch_nll": loss_total / microbatches,
                                "elapsed_seconds": elapsed,
                                "sequences_per_second": optimizer_step
                                * global_batch
                                / elapsed,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        if optimizer_step != (epoch + 1) * sampler.steps:
            raise RuntimeError("epoch did not end at the declared optimizer step")
        validation = (
            _validate_nll(
                model, val_loader, device=device, mesh=mesh, xs=xs, xm=xm
            )
            if val_loader is not None
            else None
        )
        row = {
            "epoch": epoch + 1,
            "optimizer_step": optimizer_step,
            "train_nll_mean_microbatch": loss_total / microbatches,
            "validation": validation,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(row)
        atomic_json(history_path, history)
        checkpoint_path = args.out / f"checkpoint_epoch_{epoch + 1:02d}"
        _save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            augmentation_generator=augmentation_generator,
            epoch=epoch + 1,
            optimizer_step=optimizer_step,
            endpoint=production_endpoint,
            config_sha256=config_sha256,
            train_manifest_sha256=train_manifest_sha256,
            val_manifest_sha256=val_manifest_sha256,
            sample_order_sha256=sample_order_digest.hexdigest(),
        )
        if optimizer_step >= endpoint:
            break

    complete = {
        "schema_version": "madeleine.vpt-paper-idm-xla-training-complete.v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "completed_endpoint": optimizer_step == endpoint,
        "optimizer_steps": optimizer_step,
        "endpoint": endpoint,
        "epochs_recorded": len(history),
        "elapsed_seconds": time.monotonic() - started,
        "sample_order_sha256": sample_order_digest.hexdigest(),
    }
    atomic_json(args.out / "complete.json", complete)
    import torch_xla.debug.metrics as met

    (args.out / "xla_metrics.txt").write_text(
        met.metrics_report(), encoding="utf-8"
    )
    print(json.dumps(complete, sort_keys=True))
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
