from __future__ import annotations

from pathlib import Path

import pytest
import torch

from badeline.vpt_small import natural_factored_nll
from experiments.train_vpt_small import (
    EpochRankSampler,
    capture_rng_state,
    resume_checkpoint,
    save_checkpoint,
)


def test_two_rank_sampler_reconstructs_padded_global_batches() -> None:
    left = EpochRankSampler(259, global_batch=128, rank=0, world_size=2, seed=7)
    right = EpochRankSampler(259, global_batch=128, rank=1, world_size=2, seed=7)
    left.set_epoch(2)
    right.set_epoch(2)
    left_rows = left.epoch_indices()
    right_rows = right.epoch_indices()
    assert left.steps == 3
    assert left.repeated_per_epoch == 125
    assert len(left_rows) == len(right_rows) == 192
    for step in range(3):
        global_rows = left_rows[step * 64 : (step + 1) * 64]
        global_rows += right_rows[step * 64 : (step + 1) * 64]
        assert len(global_rows) == 128
    assert left.epoch_indices() == left_rows


def test_ddp_mean_of_two_rank_gradients_matches_global_batch_gradient() -> None:
    torch.manual_seed(12)
    full_logits = torch.randn(4, 3, 7, 2, requires_grad=True)
    targets = torch.randint(0, 2, (4, 3, 7))
    full_loss = natural_factored_nll(full_logits, targets)
    full_gradient, = torch.autograd.grad(full_loss, full_logits)

    rank_logits = full_logits.detach().clone().requires_grad_(True)
    left = natural_factored_nll(rank_logits[:2], targets[:2])
    right = natural_factored_nll(rank_logits[2:], targets[2:])
    # DDP averages corresponding parameter gradients across ranks. For this
    # logits-level surrogate, multiplying each local input gradient by 1/2
    # reconstructs the gradient contribution of the global mean.
    rank_gradient, = torch.autograd.grad((left + right) / 2.0, rank_logits)
    assert torch.allclose(full_gradient, rank_gradient, atol=1e-7, rtol=1e-6)


def test_linear_schedule_starts_at_declared_lr_and_ends_at_zero() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0.003, weight_decay=0.01)
    endpoint = 5
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: max(0.0, 1.0 - step / endpoint)
    )
    assert optimizer.param_groups[0]["lr"] == 0.003
    for _ in range(endpoint):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.0


def test_epoch_checkpoint_restores_augmentation_rng_and_rejects_partial(
    tmp_path: Path,
) -> None:
    torch.manual_seed(31)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    generator = torch.Generator().manual_seed(77)
    _ = torch.rand(4, generator=generator)
    rng = capture_rng_state(
        device=torch.device("cpu"), augmentation_generator=generator
    )
    expected_next = torch.rand(4, generator=generator)
    checkpoint = tmp_path / "epoch.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config={},
        epoch=1,
        optimizer_step=5,
        best_validation_nll=1.0,
        best_epoch=1,
        completed_epoch=True,
        rng_by_rank=[rng],
        train_manifest_sha256="a" * 64,
        val_manifest_sha256="b" * 64,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    generator.manual_seed(999)
    resume_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        rank=0,
        device=torch.device("cpu"),
        augmentation_generator=generator,
    )
    assert torch.equal(torch.rand(4, generator=generator), expected_next)
    assert any(torch.count_nonzero(parameter) for parameter in model.parameters())

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["completed_epoch"] = False
    partial = tmp_path / "partial.pt"
    torch.save(payload, partial)
    with pytest.raises(ValueError, match="completed epoch"):
        resume_checkpoint(
            partial,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=0,
            device=torch.device("cpu"),
            augmentation_generator=generator,
        )
