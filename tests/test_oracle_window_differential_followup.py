from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.oracle_window_differential_followup import (
    DifferentialCandidateModel,
    PixelOracleDataset,
    frame_pair_inputs,
    requested_logits,
)


def _metadata(count: int) -> dict[str, np.ndarray]:
    heads = np.arange(count, dtype=np.int16) % 14
    return {
        "rgb": np.zeros((count, 32, 32, 32, 3), dtype=np.uint8),
        "session_id": np.asarray(["s"] * count),
        "run_index": np.zeros(count, dtype=np.int32),
        "array_index": np.arange(count, dtype=np.int64),
        "engine_frame_idx": np.arange(count, dtype=np.int64),
        "head_index": heads,
        "key_index": (heads % 7).astype(np.int8),
        "event_type_index": (heads // 7).astype(np.int8),
        "true_offset": (np.arange(count) % 16).astype(np.int8),
        "crop_start": np.arange(count, dtype=np.int64),
        "block_id": np.asarray([f"b{i}" for i in range(count)]),
    }


def test_ordered_and_symmetric_pair_pair_inputs_are_explicit() -> None:
    rgb = torch.zeros((1, 32, 32, 32, 3), dtype=torch.uint8)
    rgb[:, 12] = 255
    ordered = frame_pair_inputs(rgb, arm="ordered_pair")
    symmetric_pair = frame_pair_inputs(rgb, arm="symmetric_pair")
    assert ordered.shape == symmetric_pair.shape == (1, 31, 9, 32, 32)
    assert torch.all(ordered[:, 11, 6:] == 1.0)
    assert torch.equal(symmetric_pair[:, :, :3], symmetric_pair[:, :, 3:6])
    assert torch.count_nonzero(symmetric_pair[:, :, 6:]) == 0
    assert torch.all(symmetric_pair[:, 11, :6] == 0.0)
    assert torch.all(symmetric_pair[:, 12, :6] == 0.0)
    assert torch.all(symmetric_pair[:, 10, :6] == -1.0)


def test_constant_pixels_cannot_reveal_candidate_position() -> None:
    model = DifferentialCandidateModel()
    rgb = torch.full((3, 32, 32, 32, 3), 91, dtype=torch.uint8)
    for arm in ("ordered_pair", "symmetric_pair"):
        dense = model(rgb, arm=arm)
        assert dense.shape == (3, 16, 14)
        assert torch.equal(dense, dense[:, :1].expand_as(dense))
        heads = torch.tensor([0, 7, 13], dtype=torch.long)
        selected = requested_logits(dense, heads)
        assert selected.shape == (3, 16)
        assert torch.equal(selected, selected[:, :1].expand_as(selected))


def test_valid_temporal_convolution_is_shift_equivariant() -> None:
    torch.manual_seed(7)
    model = DifferentialCandidateModel().eval()
    stream = torch.randint(0, 256, (1, 33, 32, 32, 3), dtype=torch.uint8)
    windows = torch.cat((stream[:, :32], stream[:, 1:]), dim=0)
    with torch.inference_mode():
        logits = model(windows, arm="ordered_pair")
    assert torch.allclose(logits[0, 1:], logits[1, :-1], atol=1e-6, rtol=1e-6)


def test_dataset_weights_equalize_present_tasks() -> None:
    arrays = _metadata(18)
    arrays["head_index"] = np.asarray([0] * 16 + [1, 1], dtype=np.int16)
    arrays["key_index"] = (arrays["head_index"] % 7).astype(np.int8)
    arrays["event_type_index"] = (arrays["head_index"] // 7).astype(np.int8)
    dataset = PixelOracleDataset(arrays)
    weighted = np.bincount(
        arrays["head_index"],
        weights=dataset.task_weights[arrays["head_index"]],
        minlength=14,
    )
    assert weighted[0] == pytest.approx(weighted[1])


def test_requested_logits_rejects_wrong_head_dtype() -> None:
    with pytest.raises(ValueError, match="int64"):
        requested_logits(
            torch.zeros((2, 16, 14)), torch.zeros(2, dtype=torch.int32)
        )
