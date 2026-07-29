from __future__ import annotations

import pytest
import torch
from torch import nn

from badeline.dynamics_pretraining import (
    EMADynamicsPretrainer,
    REPRESENTATION_DIM,
    HorizonConditionedPredictor,
    OrderedPairResNet18Encoder,
    ResNet18FrameEncoder,
    collapse_diagnostics,
    normalized_l1_loss,
)


class _TinyFrameEncoder(nn.Module):
    output_dim = REPRESENTATION_DIM

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, REPRESENTATION_DIM, bias=False)
        self.register_buffer("running", torch.tensor([2.0]))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        return self.projection(current.mean(dim=(-2, -1)))


class _TinyPairEncoder(nn.Module):
    output_dim = REPRESENTATION_DIM

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(6, REPRESENTATION_DIM, bias=False)

    def forward(
        self, previous: torch.Tensor, current: torch.Tensor
    ) -> torch.Tensor:
        pair = torch.cat(
            (previous.mean(dim=(-2, -1)), current.mean(dim=(-2, -1))), dim=-1
        )
        return self.projection(pair)


def test_pair_encoder_is_current_frame_encoder_at_initialization() -> None:
    torch.manual_seed(12)
    current_encoder = ResNet18FrameEncoder(weights=None).eval()
    pair_encoder = OrderedPairResNet18Encoder.from_frame_encoder(
        current_encoder
    ).eval()

    previous = torch.rand(2, 3, 32, 32)
    current = torch.rand(2, 3, 32, 32)
    pair_conv = pair_encoder.backbone.conv1.weight
    assert torch.count_nonzero(pair_conv[:, :3]) == 0
    assert torch.equal(
        pair_conv[:, 3:6], current_encoder.backbone.conv1.weight
    )
    assert torch.count_nonzero(pair_conv[:, 6:]) == 0

    with torch.no_grad():
        expected = current_encoder(current)
        actual = pair_encoder(previous, current)
    assert torch.equal(actual, expected)


def test_pair_encoder_duplicate_policy_is_explicit_and_zero_difference() -> None:
    encoder = OrderedPairResNet18Encoder(weights=None)
    current = torch.rand(2, 3, 16, 16)
    ordered = encoder.ordered_input(current, current)

    assert ordered.shape == (2, 9, 16, 16)
    assert torch.equal(ordered[:, :3], ordered[:, 3:6])
    assert torch.count_nonzero(ordered[:, 6:]) == 0
    with pytest.raises(TypeError):
        encoder(current)  # type: ignore[call-arg]


def test_pair_encoder_rejects_misaligned_inputs() -> None:
    encoder = OrderedPairResNet18Encoder(weights=None)
    with pytest.raises(ValueError, match="identical shapes"):
        encoder(
            torch.rand(2, 3, 16, 16),
            torch.rand(2, 3, 15, 16),
        )


def test_horizon_predictor_supports_mixed_native_frame_offsets() -> None:
    predictor = HorizonConditionedPredictor([1, 2, 4, 8, 16])
    latent = torch.randn(5, REPRESENTATION_DIM)
    horizons = torch.tensor([1, 2, 4, 8, 16])

    output = predictor(latent, horizons)
    assert output.shape == latent.shape
    assert predictor.horizons == (1, 2, 4, 8, 16)
    with pytest.raises(ValueError, match="unsupported horizon"):
        predictor(latent, torch.tensor([1, 2, 4, 8, 12]))


def test_normalized_l1_is_scale_invariant_and_detaches_target() -> None:
    prediction = torch.randn(3, REPRESENTATION_DIM, requires_grad=True)
    target = torch.randn(3, REPRESENTATION_DIM, requires_grad=True)

    loss = normalized_l1_loss(prediction, target)
    scaled = normalized_l1_loss(9.0 * prediction, 0.2 * target)
    torch.testing.assert_close(loss, scaled)
    loss.backward()
    assert prediction.grad is not None
    assert target.grad is None


@pytest.mark.parametrize("arm", ["B", "C", "D"])
def test_pretrainer_supports_all_three_arms_without_target_gradients(
    arm: str,
) -> None:
    torch.manual_seed(3)
    encoder: nn.Module
    horizons: tuple[int, ...]
    kwargs: dict[str, torch.Tensor]
    if arm == "D":
        encoder = _TinyPairEncoder()
        horizons = (1, 2)
        kwargs = {
            "online_previous": torch.rand(4, 3, 4, 4),
            "target_previous": torch.rand(4, 3, 4, 4),
        }
    else:
        encoder = _TinyFrameEncoder()
        horizons = (1, 2)
        kwargs = {}
    model = EMADynamicsPretrainer(
        arm,  # type: ignore[arg-type]
        horizons=horizons,
        online_encoder=encoder,
    ).train()
    output = model(
        online_current=torch.rand(4, 3, 4, 4),
        target_current=torch.rand(4, 3, 4, 4),
        horizon=horizons[0],
        **kwargs,
    )

    assert output.online.shape == (4, REPRESENTATION_DIM)
    assert output.prediction.shape == output.target.shape
    assert not output.target.requires_grad
    assert not model.target_encoder.training
    output.loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.online_encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.target_encoder.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())


def test_arm_d_refuses_to_invent_previous_frames() -> None:
    model = EMADynamicsPretrainer(
        "D", horizons=(1,), online_encoder=_TinyPairEncoder()
    )
    with pytest.raises(ValueError, match="explicit previous frame"):
        model(
            online_current=torch.rand(2, 3, 4, 4),
            target_current=torch.rand(2, 3, 4, 4),
            horizon=1,
        )


def test_ema_update_is_name_aligned_deterministic_and_updates_buffers() -> None:
    torch.manual_seed(19)
    first = EMADynamicsPretrainer(
        "C", horizons=(1,), online_encoder=_TinyFrameEncoder()
    )
    second = EMADynamicsPretrainer(
        "C", horizons=(1,), online_encoder=_TinyFrameEncoder()
    )
    second.load_state_dict(first.state_dict())
    with torch.no_grad():
        for model in (first, second):
            model.online_encoder.projection.weight.fill_(4.0)
            model.online_encoder.running.fill_(6.0)
            model.online_encoder.updates.fill_(7)
    initial_target = first.target_encoder.projection.weight.detach().clone()

    first.update_target(momentum=0.25)
    second.update_target(momentum=0.25)

    expected = initial_target * 0.25 + 4.0 * 0.75
    torch.testing.assert_close(first.target_encoder.projection.weight, expected)
    for first_value, second_value in zip(
        first.target_encoder.state_dict().values(),
        second.target_encoder.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(first_value, second_value)
    torch.testing.assert_close(first.target_encoder.running, torch.tensor([5.0]))
    assert first.target_encoder.updates.item() == 7


def test_collapse_diagnostics_distinguish_collapsed_and_diverse_latents() -> None:
    collapsed = torch.ones(8, 4)
    collapsed_metrics = collapse_diagnostics(collapsed)
    assert torch.count_nonzero(collapsed_metrics.per_dimension_std) == 0
    assert collapsed_metrics.covariance_effective_rank.item() == 0.0
    torch.testing.assert_close(
        collapsed_metrics.mean_cosine_similarity, torch.tensor(1.0)
    )

    diverse = torch.eye(4)
    diverse_metrics = collapse_diagnostics(diverse)
    torch.testing.assert_close(
        diverse_metrics.per_dimension_std,
        torch.full((4,), (3.0 / 16.0) ** 0.5),
    )
    torch.testing.assert_close(
        diverse_metrics.covariance_effective_rank,
        torch.tensor(3.0),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        diverse_metrics.mean_cosine_similarity, torch.tensor(0.0)
    )


def test_invalid_arm_and_horizon_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive matched horizon"):
        EMADynamicsPretrainer(
            "B", horizons=(0,), online_encoder=_TinyFrameEncoder()
        )
    with pytest.raises(ValueError, match="positive matched horizon"):
        EMADynamicsPretrainer(
            "C", horizons=(0, 1), online_encoder=_TinyFrameEncoder()
        )
    with pytest.raises(ValueError, match="EMA momentum"):
        EMADynamicsPretrainer(
            "C",
            horizons=(1,),
            online_encoder=_TinyFrameEncoder(),
            ema_momentum=1.1,
        )
