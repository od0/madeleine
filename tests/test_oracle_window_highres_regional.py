from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.oracle_window_highres_regional import (
    FullResolutionOracleDataset,
    HighResolutionRegionalLocalizer,
    arm_geometry,
    smoke_subset,
    train_model,
    validate_cache,
)
from experiments.oracle_window_localization import HEAD_NAMES, state_dict_sha256


def _index_arrays(session_id: str, *, crop_start: int = 1) -> dict[str, np.ndarray]:
    return {
        "session_id": np.asarray([session_id]),
        "run_index": np.asarray([0], np.int32),
        "array_index": np.asarray([9], np.int64),
        "engine_frame_idx": np.asarray([9], np.int64),
        "head_index": np.asarray([3], np.int16),
        "key_index": np.asarray([3], np.int8),
        "event_type_index": np.asarray([0], np.int8),
        "true_offset": np.asarray([0], np.int8),
        "crop_start": np.asarray([crop_start], np.int64),
        "block_id": np.asarray([f"{session_id}:run0:block0"]),
    }


def test_memory_mapped_dataset_exposes_only_model_tensors(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    (cache / "frames").mkdir(parents=True)
    session_id = "train"
    frames = np.lib.format.open_memmap(
        cache / "frames" / f"{session_id}.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(40, 128, 128, 3),
    )
    frames[:] = np.arange(40, dtype=np.uint8)[:, None, None, None]
    frames.flush()
    del frames
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "crop_frames": 32,
                "candidate_width": 16,
                "context_halo": 8,
                "sessions": [session_id],
            }
        ),
        encoding="utf-8",
    )
    np.savez(cache / "train_examples.npz", **_index_arrays(session_id))
    dataset = FullResolutionOracleDataset(cache, split="train")
    row = dataset[0]
    assert set(row) == {"rgb", "requested_head", "target_offset", "task_weight"}
    assert row["rgb"].shape == (32, 128, 128, 3)
    assert torch.all(row["rgb"][0] == 1)
    assert torch.all(row["rgb"][-1] == 32)


def test_dataset_rejects_crop_past_session_boundary(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    (cache / "frames").mkdir(parents=True)
    np.save(cache / "frames" / "train.npy", np.zeros((32, 128, 128, 3), np.uint8))
    (cache / "manifest.json").write_text(
        json.dumps({"crop_frames": 32, "candidate_width": 16, "context_halo": 8, "sessions": ["train"]}),
        encoding="utf-8",
    )
    np.savez(cache / "train_examples.npz", **_index_arrays("train", crop_start=1))
    dataset = FullResolutionOracleDataset(cache, split="train")
    with pytest.raises(ValueError, match="escaped"):
        dataset[0]


def test_all_arms_have_expected_geometry_and_parameter_identity() -> None:
    torch.manual_seed(7)
    first = HighResolutionRegionalLocalizer(token_dim=16, imagenet_weights=False)
    torch.manual_seed(7)
    second = HighResolutionRegionalLocalizer(token_dim=16, imagenet_weights=False)
    assert state_dict_sha256(first) == state_dict_sha256(second)
    assert arm_geometry("h32_q") == (32, "query")
    assert arm_geometry("h128_g") == (128, "global")
    assert arm_geometry("h128_q") == (128, "query")
    parameter_count = sum(parameter.numel() for parameter in first.parameters())
    assert parameter_count == sum(parameter.numel() for parameter in second.parameters())


@pytest.mark.parametrize(
    ("input_size", "readout", "grid"),
    [(32, "query", 4), (128, "global", 64), (128, "query", 64)],
)
def test_model_maps_31_pairs_to_16_logits_without_position_inputs(
    input_size: int, readout: str, grid: int
) -> None:
    torch.manual_seed(3)
    model = HighResolutionRegionalLocalizer(token_dim=16, imagenet_weights=False).eval()
    rgb = torch.zeros((1, 32, 128, 128, 3), dtype=torch.uint8)
    head = torch.tensor([len(HEAD_NAMES) - 1])
    with torch.inference_mode():
        logits, attention = model(
            rgb,
            head,
            input_size=input_size,
            readout_mode=readout,
            return_attention=True,
        )
    assert logits.shape == (1, 16)
    assert attention.shape == (1, 31, grid)
    assert torch.allclose(attention.sum(-1), torch.ones((1, 31)), atol=1e-6)
    if readout == "global":
        assert torch.allclose(attention, torch.full_like(attention, 1.0 / grid))


def test_model_rejects_answer_key_metadata_by_interface() -> None:
    parameters = HighResolutionRegionalLocalizer.forward.__annotations__
    assert "session_id" not in parameters
    assert "crop_start" not in parameters
    assert "engine_frame_idx" not in parameters


def test_smoke_subset_is_bounded_and_covers_every_head() -> None:
    class Parent:
        metadata = {
            "head_index": np.repeat(np.arange(len(HEAD_NAMES), dtype=np.int16), 2),
            "true_offset": np.tile(np.asarray([0, 1], np.int8), len(HEAD_NAMES)),
        }
        task_weights = np.ones(len(HEAD_NAMES), np.float64)
        width = 16

        def __len__(self) -> int:
            return len(self.metadata["head_index"])

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {"index": torch.tensor(index)}

    selected = smoke_subset(Parent(), maximum=len(HEAD_NAMES), require_every_head=True)
    assert len(selected) == len(HEAD_NAMES)
    assert np.array_equal(selected.metadata["head_index"], np.arange(len(HEAD_NAMES)))
    assert [int(selected[index]["index"]) for index in range(len(selected))] == list(
        range(0, 2 * len(HEAD_NAMES), 2)
    )
    with pytest.raises(ValueError, match="cover every head"):
        smoke_subset(Parent(), maximum=len(HEAD_NAMES) - 1, require_every_head=True)


def test_training_handles_final_short_effective_and_microbatch() -> None:
    class TinyDataset:
        metadata = {
            "head_index": np.zeros(5, dtype=np.int16),
            "true_offset": np.arange(5, dtype=np.int8),
        }
        task_weights = np.ones(len(HEAD_NAMES), np.float64)

        def __len__(self) -> int:
            return 5

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {
                "rgb": torch.zeros((1,), dtype=torch.uint8),
                "requested_head": torch.tensor(0, dtype=torch.long),
                "target_offset": torch.tensor(index, dtype=torch.long),
                "task_weight": torch.tensor(1.0),
            }

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(1, 1)
            self.logits = torch.nn.Parameter(torch.zeros(16))

        def forward(
            self,
            rgb: torch.Tensor,
            requested_head: torch.Tensor,
            *,
            input_size: int,
            readout_mode: str,
        ) -> torch.Tensor:
            del requested_head, input_size, readout_mode
            return self.logits.unsqueeze(0).expand(len(rgb), -1)

    model = TinyModel()
    before = model.logits.detach().clone()
    config = {
        "training": {
            "seed": 0,
            "effective_batch_size": 4,
            "microbatch_size": 2,
            "encoder_learning_rate": 1e-3,
            "new_layer_learning_rate": 1e-3,
            "weight_decay": 0.0,
            "cuda_bf16": False,
        }
    }
    log = train_model(
        model,
        TinyDataset(),
        arm="h32_q",
        config=config,
        device=torch.device("cpu"),
        epochs=1,
    )
    assert len(log) == 1 and np.isfinite(log[0]["loss"])
    assert not torch.equal(before, model.logits.detach())


def test_cache_validation_accepts_only_hash_verified_relocation(tmp_path: Path) -> None:
    cache = tmp_path / "relocated-cache"
    cache.mkdir()
    manifest = cache / "manifest.json"
    manifest.write_text('{"cache":"exact"}\n', encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    receipt_base = {
        "status": "complete",
        "published_output": "/original/control-machine/cache",
        "manifest_sha256": manifest_hash,
        "payload": {
            "manifest.json": {
                "bytes": manifest.stat().st_size,
                "sha256": manifest_hash,
            }
        },
    }
    receipt = {
        **receipt_base,
        "content_sha256": hashlib.sha256(
            json.dumps(
                receipt_base, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }
    receipt_path = tmp_path / "complete.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert validate_cache(
        cache, receipt_path, expected_receipt_sha256=receipt_hash
    ) == receipt
    manifest.write_text('{"cache":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        validate_cache(cache, receipt_path, expected_receipt_sha256=receipt_hash)
