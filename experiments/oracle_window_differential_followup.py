"""Bounded learned masked-pixel differential follow-up for oracle localization."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.oracle_window_localization import (
    HEAD_NAMES,
    _validate_implementation,
    _write_json,
    _write_npz_atomic,
    epoch_order,
    sha256_file,
    state_dict_sha256,
)
from experiments.score_oracle_window_localization import load_prediction_sidecar


METADATA_NAMES = (
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
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_pixel_receipt(
    receipt_path: Path, cache_root: Path, expected_sha256: str
) -> dict[str, Any]:
    observed = sha256_file(receipt_path)
    if observed != expected_sha256:
        raise ValueError(f"pixel-cache receipt hash changed: {observed}")
    receipt = _json(receipt_path)
    if receipt.get("status") != "complete":
        raise ValueError("pixel-cache receipt is incomplete")
    if Path(str(receipt.get("published_output"))).resolve() != cache_root.resolve():
        raise ValueError("pixel-cache receipt points to another output")
    checks = receipt.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
        raise ValueError("pixel-cache receipt did not pass every check")
    return receipt


def load_pixel_split(
    cache_root: Path,
    receipt: Mapping[str, Any],
    *,
    split: str,
    expected_examples: int,
) -> dict[str, np.ndarray]:
    if split not in ("train", "validation"):
        raise ValueError("unknown pixel-cache split")
    path = cache_root / f"{split}.npz"
    expected_file = receipt.get("cache", {}).get(f"{split}.npz")
    if not isinstance(expected_file, Mapping):
        raise ValueError(f"pixel receipt lacks {split} cache authority")
    if path.stat().st_size != int(expected_file.get("bytes", -1)):
        raise ValueError(f"pixel {split} cache byte count changed")
    if sha256_file(path) != str(expected_file.get("sha256")):
        raise ValueError(f"pixel {split} cache hash changed")
    with np.load(path, allow_pickle=False) as archive:
        expected_names = {"rgb", *METADATA_NAMES}
        if set(archive.files) != expected_names:
            raise ValueError(f"pixel {split} cache inventory changed")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    rgb = arrays["rgb"]
    if rgb.dtype != np.uint8 or rgb.shape != (expected_examples, 32, 32, 32, 3):
        raise ValueError(f"pixel {split} rgb geometry changed")
    for name in METADATA_NAMES:
        if len(arrays[name]) != expected_examples:
            raise ValueError(f"pixel {split} metadata length changed: {name}")
    heads = arrays["head_index"]
    truth = arrays["true_offset"]
    if heads.dtype != np.int16 or np.any((heads < 0) | (heads >= len(HEAD_NAMES))):
        raise ValueError("pixel cache head indices changed")
    if truth.dtype != np.int8 or np.any((truth < 0) | (truth >= 16)):
        raise ValueError("pixel cache target offsets changed")
    if not np.array_equal(arrays["key_index"], heads % 7):
        raise ValueError("pixel cache key identity changed")
    if not np.array_equal(arrays["event_type_index"], heads // 7):
        raise ValueError("pixel cache event identity changed")
    return arrays


def frame_pair_inputs(rgb: torch.Tensor, *, arm: str) -> torch.Tensor:
    """Return 31 ordered pairs or their exact-support symmetric-pair ablation."""

    if rgb.dtype != torch.uint8 or rgb.ndim != 5 or rgb.shape[1:] != (32, 32, 32, 3):
        raise ValueError("rgb must be uint8 [B,32,32,32,3]")
    if arm not in ("ordered_pair", "symmetric_pair"):
        raise ValueError("unknown differential arm")
    value = rgb.to(dtype=torch.float32).permute(0, 1, 4, 2, 3) / 255.0
    previous = value[:, :-1]
    current = value[:, 1:]
    if arm == "ordered_pair":
        pair = (previous * 2.0 - 1.0, current * 2.0 - 1.0, current - previous)
    else:
        mean = (previous + current) * 0.5
        pair = (mean * 2.0 - 1.0, mean * 2.0 - 1.0, torch.zeros_like(mean))
    return torch.cat(pair, dim=2)


class DifferentialCandidateModel(nn.Module):
    """Shared pair encoder plus valid temporal convolution for 16 offsets."""

    def __init__(self, *, embedding_dim: int = 64) -> None:
        super().__init__()
        if embedding_dim != 64:
            raise ValueError("the bounded adapter freezes a 64-dimensional embedding")
        self.embedding_dim = int(embedding_dim)
        self.pair_encoder = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, groups=16),
            nn.Conv2d(16, 24, kernel_size=1),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1, groups=24),
            nn.Conv2d(24, 32, kernel_size=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.spatial = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, embedding_dim),
            nn.SiLU(),
        )
        self.temporal = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=16)
        self.temporal_activation = nn.SiLU()
        self.temporal_norm = nn.LayerNorm(embedding_dim)
        self.heads = nn.Linear(embedding_dim, len(HEAD_NAMES))

    def forward(self, rgb: torch.Tensor, *, arm: str) -> torch.Tensor:
        pairs = frame_pair_inputs(rgb, arm=arm)
        batch, pair_count = pairs.shape[:2]
        if pair_count != 31:
            raise AssertionError("32 frames must produce 31 adjacent pairs")
        encoded = self.pair_encoder(pairs.reshape(batch * pair_count, 9, 32, 32))
        embedded = self.spatial(encoded).reshape(batch, pair_count, self.embedding_dim)
        candidate = self.temporal(embedded.transpose(1, 2)).transpose(1, 2)
        if candidate.shape[1] != 16:
            raise AssertionError("valid width-16 temporal kernel must produce 16 offsets")
        candidate = self.temporal_norm(self.temporal_activation(candidate))
        return self.heads(candidate)


def requested_logits(dense: torch.Tensor, requested_head: torch.Tensor) -> torch.Tensor:
    if dense.ndim != 3 or dense.shape[1:] != (16, len(HEAD_NAMES)):
        raise ValueError("differential logits must be [B,16,14]")
    if requested_head.dtype != torch.long or requested_head.shape != (len(dense),):
        raise ValueError("requested heads must be int64 [B]")
    index = requested_head[:, None, None].expand(-1, 16, 1)
    return dense.gather(2, index).squeeze(2)


class PixelOracleDataset:
    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.arrays = dict(arrays)
        heads = self.arrays["head_index"].astype(np.int64)
        counts = np.bincount(heads, minlength=len(HEAD_NAMES))
        present = counts > 0
        self.task_weights = np.zeros(len(HEAD_NAMES), dtype=np.float32)
        self.task_weights[present] = (
            len(heads) / (int(present.sum()) * counts[present])
        ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.arrays["rgb"])


def _batch(
    dataset: PixelOracleDataset, indices: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = np.asarray(indices, dtype=np.int64)
    heads = dataset.arrays["head_index"][selected].astype(np.int64, copy=False)
    truth = dataset.arrays["true_offset"][selected].astype(np.int64, copy=False)
    return (
        torch.from_numpy(dataset.arrays["rgb"][selected].copy()).to(device),
        torch.from_numpy(heads.copy()).to(device),
        torch.from_numpy(truth.copy()).to(device),
        torch.from_numpy(dataset.task_weights[heads].copy()).to(device),
    )


def train_model(
    model: DifferentialCandidateModel,
    dataset: PixelOracleDataset,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    arm: str,
) -> list[dict[str, float]]:
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
            indices = order[start : start + batch_size]
            rgb, heads, truth, weights = _batch(dataset, indices, device)
            optimizer.zero_grad(set_to_none=True)
            logits = requested_logits(model(rgb, arm=arm), heads)
            per_example = F.cross_entropy(logits, truth, reduction="none")
            loss = (per_example * weights).sum() / weights.sum()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            total_examples += len(indices)
        log.append({"epoch": float(epoch + 1), "loss": total_loss / total_examples})
    return log


@torch.inference_mode()
def predict_probabilities(
    model: DifferentialCandidateModel,
    dataset: PixelOracleDataset,
    *,
    device: torch.device,
    batch_size: int,
    arm: str,
) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    for start in range(0, len(dataset), batch_size):
        indices = np.arange(start, min(start + batch_size, len(dataset)))
        rgb, heads, _, _ = _batch(dataset, indices, device)
        probability = requested_logits(model(rgb, arm=arm), heads).softmax(dim=1)
        result.append(probability.cpu().numpy())
    combined = np.concatenate(result).astype(np.float32, copy=False)
    if combined.shape != (len(dataset), 16):
        raise AssertionError("differential prediction shape changed")
    if not np.all(np.isfinite(combined)) or not np.allclose(
        combined.sum(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError("differential predictions are not finite and normalized")
    return combined


def _validate_frozen_baseline(
    *,
    baseline_sidecar_path: Path,
    audit_path: Path,
    config: Mapping[str, Any],
    validation_arrays: Mapping[str, np.ndarray],
) -> np.ndarray:
    baseline_config = config["frozen_feature_baseline"]
    if sha256_file(baseline_sidecar_path) != str(
        baseline_config["prediction_sidecar_sha256"]
    ):
        raise ValueError("frozen feature-baseline sidecar hash changed")
    if sha256_file(audit_path) != str(baseline_config["audit_receipt_sha256"]):
        raise ValueError("frozen feature-baseline audit hash changed")
    audit = _json(audit_path)
    if audit.get("status") != "complete" or audit.get("decision", {}).get("passed"):
        raise ValueError("feature-baseline audit or original rejection changed")
    baseline = load_prediction_sidecar(baseline_sidecar_path, width=16)
    for name in METADATA_NAMES:
        if not np.array_equal(baseline[name], validation_arrays[name]):
            raise ValueError(f"pixel validation identity differs from baseline: {name}")
    return baseline["conditional_prob"]


def run_followup(
    *,
    cache_root: Path,
    cache_receipt_path: Path,
    config_path: Path,
    output: Path,
    baseline_sidecar_path: Path,
    baseline_audit_path: Path,
    device_name: str,
    epoch_override: int | None = None,
) -> Path:
    if os.path.lexists(output):
        raise ValueError(f"refusing to overwrite differential output: {output}")
    config = _json(config_path)
    if config.get("status") != "preregistered_before_validation_inference":
        raise ValueError("differential follow-up config is not frozen")
    repo = config_path.resolve().parents[2]
    implementation = _validate_implementation(config, repo=repo)
    cache_root = cache_root.resolve()
    receipt = _validate_pixel_receipt(
        cache_receipt_path,
        cache_root,
        str(config["dataset"]["pixel_cache_receipt_sha256"]),
    )
    expected = config["dataset"]["expected_support"]
    train_arrays = load_pixel_split(
        cache_root,
        receipt,
        split="train",
        expected_examples=int(expected["training_examples"]),
    )
    validation_arrays = load_pixel_split(
        cache_root,
        receipt,
        split="validation",
        expected_examples=int(expected["validation_examples"]),
    )
    baseline_probability = _validate_frozen_baseline(
        baseline_sidecar_path=baseline_sidecar_path,
        audit_path=baseline_audit_path,
        config=config,
        validation_arrays=validation_arrays,
    )
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=(device.type != "cuda"))
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    initial = DifferentialCandidateModel(
        embedding_dim=int(config["model"]["embedding_dim"])
    )
    initial_sha = state_dict_sha256(initial)
    ordered_model = copy.deepcopy(initial).to(device)
    symmetric_pair_model = copy.deepcopy(initial).to(device)
    if (
        state_dict_sha256(ordered_model) != initial_sha
        or state_dict_sha256(symmetric_pair_model) != initial_sha
    ):
        raise AssertionError("matched differential initialization changed")
    configured_epochs = int(config["training"]["epochs"])
    epochs = configured_epochs if epoch_override is None else int(epoch_override)
    if epochs < 1 or epochs > configured_epochs:
        raise ValueError("epoch override is outside the frozen endpoint")
    train_dataset = PixelOracleDataset(train_arrays)
    validation_dataset = PixelOracleDataset(validation_arrays)
    ordered_log = train_model(
        ordered_model,
        train_dataset,
        device=device,
        seed=seed,
        epochs=epochs,
        batch_size=int(config["training"]["batch_size"]),
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        arm="ordered_pair",
    )
    symmetric_pair_log = train_model(
        symmetric_pair_model,
        train_dataset,
        device=device,
        seed=seed,
        epochs=epochs,
        batch_size=int(config["training"]["batch_size"]),
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        arm="symmetric_pair",
    )
    ordered_probability = predict_probabilities(
        ordered_model,
        validation_dataset,
        device=device,
        batch_size=int(config["training"]["eval_batch_size"]),
        arm="ordered_pair",
    )
    symmetric_pair_probability = predict_probabilities(
        symmetric_pair_model,
        validation_dataset,
        device=device,
        batch_size=int(config["training"]["eval_batch_size"]),
        arm="symmetric_pair",
    )

    output.mkdir(parents=True, exist_ok=False)
    config_sha = sha256_file(config_path)
    checkpoint_paths = {
        "ordered_pair": output / "ordered_pair_model.pt",
        "symmetric_pair": output / "symmetric_pair_model.pt",
    }
    for arm, model in (
        ("ordered_pair", ordered_model),
        ("symmetric_pair", symmetric_pair_model),
    ):
        torch.save(
            {
                "schema_version": "madeleine.oracle-window-differential-checkpoint.v1",
                "config_sha256": config_sha,
                "pixel_cache_receipt_sha256": sha256_file(cache_receipt_path),
                "baseline_sidecar_sha256": sha256_file(baseline_sidecar_path),
                "seed": seed,
                "epochs": epochs,
                "initial_state_sha256": initial_sha,
                "arm": arm,
                "model_state_dict": model.state_dict(),
            },
            checkpoint_paths[arm],
        )
    training_log_path = output / "training_log.json"
    _write_json(
        training_log_path,
        {
            "ordered_pair": ordered_log,
            "symmetric_pair": symmetric_pair_log,
            "fixed_final_epoch": epochs,
            "configured_final_epoch": configured_epochs,
            "validation_used_for_training_or_selection": False,
            "matched_batch_order": True,
        },
    )
    predictions_path = output / "predictions.npz"
    _write_npz_atomic(
        predictions_path,
        **{name: validation_arrays[name] for name in METADATA_NAMES},
        ordered_pair_prob=ordered_probability,
        symmetric_pair_prob=symmetric_pair_probability,
        feature_conditional_prob=baseline_probability,
    )
    receipt_content = {
        "schema_version": "madeleine.oracle-window-differential-run.v1",
        "status": "predictions_complete_unscored",
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha,
        "pixel_cache_receipt_path": str(cache_receipt_path.resolve()),
        "pixel_cache_receipt_sha256": sha256_file(cache_receipt_path),
        "baseline_sidecar_path": str(baseline_sidecar_path.resolve()),
        "baseline_sidecar_sha256": sha256_file(baseline_sidecar_path),
        "baseline_audit_path": str(baseline_audit_path.resolve()),
        "baseline_audit_sha256": sha256_file(baseline_audit_path),
        "checkpoints": {
            arm: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "model_state_sha256": state_dict_sha256(model),
            }
            for (arm, model), path in zip(
                (("ordered_pair", ordered_model), ("symmetric_pair", symmetric_pair_model)),
                (checkpoint_paths["ordered_pair"], checkpoint_paths["symmetric_pair"]),
                strict=True,
            )
        },
        "initial_state_sha256": initial_sha,
        "training_log_sha256": sha256_file(training_log_path),
        "prediction_sidecar_sha256": sha256_file(predictions_path),
        "seed": seed,
        "epochs": epochs,
        "configured_epochs": configured_epochs,
        "final_weights_only": True,
        "validation_used_for_training_or_selection": False,
        "matched_initialization": True,
        "matched_batch_order": True,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "device": str(device),
        "implementation": implementation,
    }
    _write_json(output / "run_receipt.json", receipt_content)
    return predictions_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--cache-receipt", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline-sidecar", required=True, type=Path)
    parser.add_argument("--baseline-audit", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sidecar = run_followup(
        cache_root=args.cache,
        cache_receipt_path=args.cache_receipt,
        config_path=args.config,
        output=args.out,
        baseline_sidecar_path=args.baseline_sidecar,
        baseline_audit_path=args.baseline_audit,
        device_name=args.device,
        epoch_override=args.epochs,
    )
    print(json.dumps({"status": "predictions_complete", "sidecar": str(sidecar)}))


if __name__ == "__main__":
    main()
