from __future__ import annotations

import hashlib
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from experiments.train_vpt_paper_idm_xla import (
    GlobalEpochSampler,
    KEY_ORDER,
    _validate_nll,
    git_commit,
    mark_empty_parameters_replicated,
    padded_epoch_order,
    set_optimizer_lr,
    tensor_collection_stats,
)


def _order_digest(epochs: range) -> str:
    digest = hashlib.sha256()
    for epoch in epochs:
        order = padded_epoch_order(14_921, global_batch=128, seed=0, epoch=epoch)
        digest.update(np.asarray(order, dtype="<i8").tobytes())
    return digest.hexdigest()


def test_tier_b_epoch_order_has_exact_padding_and_support() -> None:
    sampler = GlobalEpochSampler(14_921, global_batch=128, seed=0)
    assert sampler.steps == 117
    assert sampler.repeated_per_epoch == 55
    order = list(sampler)
    assert len(order) == 14_976
    assert set(order) == set(range(14_921))
    counts = np.bincount(order, minlength=14_921)
    assert np.count_nonzero(counts == 2) == 55
    assert np.count_nonzero(counts == 1) == 14_921 - 55


def test_epoch_order_and_resume_digest_are_deterministic() -> None:
    assert _order_digest(range(20)) == _order_digest(range(20))
    first = hashlib.sha256()
    for epoch in range(7):
        first.update(
            np.asarray(
                padded_epoch_order(14_921, global_batch=128, seed=0, epoch=epoch),
                dtype="<i8",
            ).tobytes()
        )
    for epoch in range(7, 20):
        first.update(
            np.asarray(
                padded_epoch_order(14_921, global_batch=128, seed=0, epoch=epoch),
                dtype="<i8",
            ).tobytes()
        )
    assert first.hexdigest() == _order_digest(range(20))


def test_linear_learning_rate_matches_frozen_2340_step_schedule() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0.003, weight_decay=0.01)
    assert set_optimizer_lr(
        optimizer, initial_lr=0.003, optimizer_step=0, endpoint=2340
    ) == 0.003
    halfway = set_optimizer_lr(
        optimizer, initial_lr=0.003, optimizer_step=1170, endpoint=2340
    )
    assert halfway == 0.0015
    last = set_optimizer_lr(
        optimizer, initial_lr=0.003, optimizer_step=2339, endpoint=2340
    )
    assert last == pytest.approx(0.003 / 2340)
    assert optimizer.param_groups[0]["lr"] == last


def test_transported_source_commit_is_explicit_and_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "2bc325f943a309672d17b403079a6d8da9c50bd7"
    monkeypatch.setenv("MADELEINE_SOURCE_COMMIT", commit.upper())
    assert git_commit() == commit
    monkeypatch.setenv("MADELEINE_SOURCE_COMMIT", "2bc325f")
    with pytest.raises(ValueError, match="full SHA-1"):
        git_commit()


def test_only_empty_parameters_are_marked_replicated() -> None:
    class RecordingSharding:
        def __init__(self) -> None:
            self.calls = []

        def mark_sharding(self, parameter, mesh, spec) -> None:
            self.calls.append((parameter, mesh, spec))

    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(2, 3)))
    model.register_parameter("empty", torch.nn.Parameter(torch.empty(10, 0)))
    sharding = RecordingSharding()

    names = mark_empty_parameters_replicated(model, mesh="mesh", xs=sharding)

    assert names == ["empty"]
    assert len(sharding.calls) == 1
    assert sharding.calls[0][0] is model.empty
    assert sharding.calls[0][1:] == ("mesh", (None, None))


def test_tensor_collection_stats_detects_nonfinite_values() -> None:
    assert tensor_collection_stats([torch.tensor([1.0, -3.0]), torch.empty(0)]) == {
        "finite": True,
        "max_abs": 3.0,
    }
    result = tensor_collection_stats([torch.tensor([1.0, float("nan")])])
    assert result["finite"] is False
    assert torch.isnan(torch.tensor(result["max_abs"]))


def test_validation_uses_no_grad_without_inference_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model(torch.nn.Module):
        def forward(self, frames: torch.Tensor) -> torch.Tensor:
            assert not torch.is_grad_enabled()
            assert not torch.is_inference_mode_enabled()
            batch, steps = frames.shape[:2]
            return torch.zeros(batch, steps, len(KEY_ORDER), 2)

    class Sharding:
        @staticmethod
        def mark_sharding(*args, **kwargs) -> None:
            return None

    class Runtime:
        @staticmethod
        def mark_step() -> None:
            return None

    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    batch = {
        "frames": torch.zeros(2, 3, 3, 4, 4, dtype=torch.uint8),
        "target": torch.zeros(2, 3, len(KEY_ORDER), dtype=torch.long),
        "valid_sequence": torch.ones(2, dtype=torch.bool),
    }

    result = _validate_nll(
        Model(),
        [batch],
        device=torch.device("cpu"),
        mesh=object(),
        xs=Sharding(),
        xm=Runtime(),
    )

    assert result["positions"] == 6
    assert result["natural_nll"] == pytest.approx(len(KEY_ORDER) * np.log(2))
