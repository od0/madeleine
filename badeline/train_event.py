"""Train the target-aligned IDM with held-state, onset, and release heads."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from badeline.event_model import EventLatchIDM
from badeline.temporal_latch import (
    TemporalEventLoss,
    TransitionTargets,
    make_transition_targets,
)
from badeline.train import (
    SegmentSessionDataset,
    SessionArrays,
    _run_metadata,
    contiguous_runs,
    load_session,
    read_session_ids,
    validate_splits,
)
from data.schema import KEY_ORDER


class EventSegmentSessionDataset(SegmentSessionDataset):
    """Segment dataset that retains the state before the first target."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = super().__getitem__(index)
        session_index, start, run_start = self._locations[index]
        session = self.sessions[session_index]
        first_target = start + self.target_in_window * self.frame_stride
        assert session.input_active is not None
        previous_valid = (
            first_target > run_start
            and bool(session.input_active[first_target - 1])
            and bool(session.input_active[first_target])
        )
        previous = (
            session.keys[first_target - 1]
            if previous_valid
            else np.zeros(len(KEY_ORDER), dtype=np.uint8)
        )
        example["previous_target"] = torch.from_numpy(
            previous.astype(np.float32, copy=True)
        )
        example["previous_valid"] = torch.tensor(previous_valid)
        return example


def _move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> tuple[dict[str, torch.Tensor], TransitionTargets]:
    inputs = {
        name: value.to(device)
        for name, value in batch.items()
        if name in ("frames", "features", "history")
    }
    state = batch["target"].to(device)
    previous = batch["previous_target"].to(device)
    previous_valid = batch["previous_valid"].to(device)
    targets = make_transition_targets(
        state,
        previous_state=previous,
        previous_valid=previous_valid,
    )
    return inputs, targets


def _cycle(
    loader: DataLoader[dict[str, torch.Tensor]],
) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def event_positive_weights(
    sessions: Sequence[SessionArrays], *, maximum: float
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Compute per-key event weights without crossing sequence boundaries."""

    if maximum < 1:
        raise ValueError("event class-balance maximum must be at least one")
    onset = np.zeros(len(KEY_ORDER), dtype=np.float64)
    release = np.zeros(len(KEY_ORDER), dtype=np.float64)
    valid_pairs = 0
    for session in sessions:
        assert session.engine_frame_idx is not None
        assert session.input_active is not None
        for start, stop in contiguous_runs(session.engine_frame_idx):
            if stop - start < 2:
                continue
            previous = session.keys[start : stop - 1]
            current = session.keys[start + 1 : stop]
            valid = (
                session.input_active[start : stop - 1].astype(bool)
                & session.input_active[start + 1 : stop].astype(bool)
            )
            if not np.any(valid):
                continue
            previous = previous[valid]
            current = current[valid]
            onset += ((previous == 0) & (current == 1)).sum(axis=0)
            release += ((previous == 1) & (current == 0)).sum(axis=0)
            valid_pairs += int(valid.sum())
    if valid_pairs == 0:
        raise ValueError("training data contains no valid adjacent target pairs")
    onset_weight = np.clip(
        (valid_pairs - onset) / np.maximum(onset, 1.0), 1.0, maximum
    )
    release_weight = np.clip(
        (valid_pairs - release) / np.maximum(release, 1.0), 1.0, maximum
    )
    counts = {
        "valid_transition_frames": valid_pairs,
        "onsets": int(onset.sum()),
        "releases": int(release.sum()),
    }
    return onset_weight, release_weight, counts


@torch.no_grad()
def evaluate_losses(
    model: EventLatchIDM,
    loader: DataLoader[dict[str, torch.Tensor]],
    criterion: TemporalEventLoss,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals = {name: 0.0 for name in ("loss", "state_loss", "onset_loss", "release_loss")}
    examples = 0
    for batch in loader:
        inputs, targets = _move_batch(batch, device)
        outputs = model.forward_segment(inputs)
        losses = criterion(outputs, targets)
        count = int(targets.state.shape[0] * targets.state.shape[1])
        for name, value in losses.items():
            totals[name] += float(value) * count
        examples += count
    model.train(was_training)
    if examples == 0:
        raise ValueError("cannot evaluate an empty event dataset")
    return {name: value / examples for name, value in totals.items()}


def _event_weight_dict(values: Sequence[float]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in zip(KEY_ORDER, values, strict=True)
    }


def run_event_training(
    *,
    data_dir: str | Path,
    train_sessions: str | Path,
    val_sessions: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    max_steps: int | None = None,
    device_name: str | None = None,
    seed_override: int | None = None,
) -> Path:
    train_ids = read_session_ids(train_sessions)
    val_ids = read_session_ids(val_sessions)
    validate_splits(train_ids, val_ids)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    config["event_latch"] = True
    config["class_balance"] = False
    config["transition_weight"] = 1.0
    if seed_override is not None:
        config["seed"] = int(seed_override)

    steps = int(max_steps if max_steps is not None else config.get("max_steps", 300))
    eval_interval = int(config.get("eval_interval", max(1, steps)))
    segment_windows = int(config.get("segment_windows", 96))
    nominal_batch = int(config.get("batch_size", 1536))
    nominal_eval_batch = int(config.get("eval_batch_size", nominal_batch))
    loader_batch = max(1, round(nominal_batch / segment_windows))
    loader_eval_batch = max(1, round(nominal_eval_batch / segment_windows))
    learning_rate = float(config.get("learning_rate", 3e-4))
    weight_decay = float(config.get("weight_decay", 0.01))
    linear_lr_decay = bool(config.get("linear_lr_decay", True))
    seed = int(config.get("seed", 0))
    if steps < 0 or eval_interval < 1:
        raise ValueError("max_steps must be non-negative and eval_interval positive")

    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    device = torch.device(device_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=(device.type != "cuda"))
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    precomputed = bool(config.get("precomputed_features", False))
    train_arrays = [
        load_session(data_dir, session_id, precomputed_features=precomputed)
        for session_id in train_ids
    ]
    val_arrays = [
        load_session(data_dir, session_id, precomputed_features=precomputed)
        for session_id in val_ids
    ]
    dataset_kwargs = {
        "window": int(config.get("window", 2)),
        "window_mode": str(config.get("window_mode", "centered")),
        "input_config": str(config.get("input_config", "pixels")),
        "history_len": int(config.get("history_len", 8)),
        "history_gap": int(config.get("history_gap", 0)),
        "segment_windows": segment_windows,
        "active_targets_only": bool(config.get("active_targets_only", True)),
        "transition_weight": 1.0,
        "precomputed_features": precomputed,
        "frame_stride": int(config.get("frame_stride", 1)),
    }
    train_dataset = EventSegmentSessionDataset(train_arrays, **dataset_kwargs)
    val_dataset = EventSegmentSessionDataset(val_arrays, **dataset_kwargs)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_batch,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_eval_batch,
        shuffle=False,
        num_workers=0,
    )

    event_maximum = float(config.get("event_class_balance_max", 50.0))
    onset_weight, release_weight, event_counts = event_positive_weights(
        train_arrays, maximum=event_maximum
    )
    model = EventLatchIDM(config).to(device)
    criterion = TemporalEventLoss(
        onset_pos_weight=torch.tensor(onset_weight, dtype=torch.float32, device=device),
        release_pos_weight=torch.tensor(release_weight, dtype=torch.float32, device=device),
        state_weight=float(config.get("state_loss_weight", 1.0)),
        onset_weight=float(config.get("onset_loss_weight", 0.5)),
        release_weight=float(config.get("release_loss_weight", 0.5)),
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = None
    if linear_lr_decay and steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda completed: max(0.0, 1.0 - completed / steps),
        )

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_meta: dict[str, Any] = _run_metadata(
        config=config,
        device=device,
        seed=seed,
        data_dir=Path(data_dir),
        train_ids=train_ids,
        val_ids=val_ids,
    )
    run_meta.update(
        {
            "event_counts": event_counts,
            "onset_positive_weight": _event_weight_dict(onset_weight),
            "release_positive_weight": _event_weight_dict(release_weight),
            "state_probability_objective": "natural_prevalence_bce",
            "event_objective": "separately_class_balanced_bce",
        }
    )
    (output / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    initial_val = evaluate_losses(model, val_loader, criterion, device)
    best_value = float(initial_val["loss"])
    best_step = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    batches = _cycle(train_loader)
    running = {name: 0.0 for name in initial_val}
    running_steps = 0
    log_path = output / "log.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"step": 0, "train": None, "val": initial_val}) + "\n")
        log.flush()
        for step in range(1, steps + 1):
            inputs, targets = _move_batch(next(batches), device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model.forward_segment(inputs)
            losses = criterion(outputs, targets)
            losses["loss"].backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            for name, value in losses.items():
                running[name] += float(value.detach())
            running_steps += 1

            if step % eval_interval == 0 or step == steps:
                train_values = {
                    name: value / running_steps for name, value in running.items()
                }
                val_values = evaluate_losses(model, val_loader, criterion, device)
                log.write(
                    json.dumps(
                        {"step": step, "train": train_values, "val": val_values},
                        sort_keys=True,
                    )
                    + "\n"
                )
                log.flush()
                if val_values["loss"] < best_value:
                    best_value = float(val_values["loss"])
                    best_step = step
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                running = {name: 0.0 for name in running}
                running_steps = 0

    checkpoint = {
        "config": config,
        "key_order": list(KEY_ORDER),
        "model_state_dict": best_state,
        "final_state_dict": model.state_dict(),
        "steps": steps,
        "best_val_step": best_step,
        "best_val_objective": best_value,
        "selection_objective": (
            "state_bce + onset_loss_weight*balanced_onset_bce + "
            "release_loss_weight*balanced_release_bce"
        ),
        "onset_positive_weight": onset_weight.tolist(),
        "release_positive_weight": release_weight.tolist(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    model_path = output / "model.pt"
    torch.save(checkpoint, model_path)
    return model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--train-sessions", required=True, type=Path)
    parser.add_argument("--val-sessions", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_event_training(
        data_dir=args.data,
        train_sessions=args.train_sessions,
        val_sessions=args.val_sessions,
        config_path=args.config,
        out_dir=args.out,
        max_steps=args.max_steps,
        device_name=args.device,
        seed_override=args.seed,
    )


if __name__ == "__main__":
    main()
