from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from badeline.vpt_paper_idm import (
    VPTPaperIDM,
    VPTPaperIDMConfig,
    _AttentionCore,
    named_parameter_shapes,
    natural_factored_nll,
    parameter_inventory,
)
from experiments.validate_vpt_paper_idm_reference import validate
from experiments.validate_vpt_paper_idm_fp32_parity import (
    compare_outputs,
    compare_parameter_gradients,
)


def tiny_config(**overrides: object) -> VPTPaperIDMConfig:
    values: dict[str, object] = {
        "frames": 4,
        "image_size": 8,
        "temporal_channels": 4,
        "spatial_widths": (4, 4, 4),
        "frame_hidden": 4,
        "d_model": 4,
        "attention_heads": 1,
        "transformer_blocks": 2,
        "activation_checkpointing": False,
    }
    values.update(overrides)
    return VPTPaperIDMConfig(**values)


def test_released_body_and_celeste_head_parameter_counts_on_meta_device() -> None:
    with torch.device("meta"):
        model = VPTPaperIDM()

    inventory = parameter_inventory(model)
    shapes = named_parameter_shapes(model)

    assert inventory == {
        "total": 482_133_390,
        "components": {"net": 482_076_032, "pi_head": 57_358},
    }
    assert model.net.img_process.cnn.flattened_width == 131_072
    assert len(model.net.img_process.cnn.stacks) == 3
    assert len(model.net.recurrent_layer.blocks) == 2
    assert len(shapes) == 98
    assert shapes["net.conv3d_layer.layer.weight"] == [128, 3, 5, 1, 1]
    assert shapes["net.img_process.cnn.dense.layer.weight"] == [256, 131_072]
    assert shapes["net.img_process.linear.layer.weight"] == [4096, 256]
    assert shapes["net.recurrent_layer.blocks.0.mlp0.layer.weight"] == [16_384, 4096]
    assert shapes["net.recurrent_layer.blocks.0.r.orc_block.b_nd"] == [10, 0]
    assert shapes["net.recurrent_layer.blocks.0.r.orc_block.r_layer.weight"] == [320, 4096]
    assert shapes["net.lastlayer.layer.weight"] == [4096, 4096]
    assert shapes["pi_head.linear_layer.weight"] == [14, 4096]


def test_tiny_graph_produces_dense_logits_and_gradients() -> None:
    torch.manual_seed(17)
    model = VPTPaperIDM(tiny_config())
    frames = torch.rand(1, 4, 3, 8, 8)
    targets = torch.randint(0, 2, (1, 4, 7))

    logits = model(frames)
    loss = natural_factored_nll(logits, targets)
    loss.backward()

    assert logits.shape == (1, 4, 7, 2)
    assert torch.isfinite(loss)
    assert model.net.conv3d_layer.layer.weight.grad is not None
    assert model.net.recurrent_layer.blocks[0].r.orc_block.r_layer.weight.grad is not None
    assert torch.count_nonzero(
        model.net.recurrent_layer.blocks[0].r.orc_block.r_layer.weight.grad
    ) == 0


def test_attention_uses_released_inverse_head_width_scaling() -> None:
    config = tiny_config()
    core = _AttentionCore(config, init_scale=1.0)
    with torch.no_grad():
        for layer in (core.q_layer, core.k_layer, core.v_layer, core.proj_layer):
            layer.weight.copy_(torch.eye(4))
            if layer.bias is not None:
                layer.bias.zero_()
        core.r_layer.weight.zero_()
        core.r_layer.bias.zero_()
    value = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]])

    actual = core(value)

    scores = torch.matmul(value, value.transpose(-1, -2)) / 4.0
    expected = value + torch.matmul(torch.softmax(scores, dim=-1), value)
    sqrt_scaled = value + torch.matmul(
        torch.softmax(torch.matmul(value, value.transpose(-1, -2)) / math.sqrt(4.0), dim=-1),
        value,
    )
    assert torch.allclose(actual, expected, atol=1e-6)
    assert not torch.allclose(actual, sqrt_scaled, atol=1e-4)


def test_released_fan_in_initialization_scales_are_frozen() -> None:
    torch.manual_seed(19)
    model = VPTPaperIDM(tiny_config())

    def row_norms(weight: torch.Tensor) -> torch.Tensor:
        return weight.flatten(1).norm(dim=1)

    block = model.net.recurrent_layer.blocks[0]
    expected_attention_residual_scale = 2 ** -0.25
    assert torch.allclose(
        row_norms(model.net.conv3d_layer.layer.weight), torch.ones(4), atol=1e-6
    )
    assert torch.allclose(
        row_norms(model.net.img_process.cnn.dense.layer.weight),
        torch.full((4,), 1.4),
        atol=1e-6,
    )
    assert torch.allclose(
        row_norms(block.r.orc_block.q_layer.weight),
        torch.full((4,), 0.1),
        atol=1e-6,
    )
    assert torch.allclose(
        row_norms(block.r.orc_block.k_layer.weight),
        torch.full((4,), 0.2),
        atol=1e-6,
    )
    assert torch.allclose(
        row_norms(block.r.orc_block.v_layer.weight),
        torch.full((4,), expected_attention_residual_scale),
        atol=1e-6,
    )
    assert torch.allclose(
        row_norms(block.mlp1.layer.weight), torch.full((4,), 0.5), atol=1e-6
    )
    assert torch.allclose(
        row_norms(model.pi_head.linear_layer.weight),
        torch.full((14,), 0.01),
        atol=1e-6,
    )


def test_unmasked_attention_allows_future_pixels_to_change_earlier_logits() -> None:
    torch.manual_seed(23)
    model = VPTPaperIDM(tiny_config()).eval()
    first = torch.rand(1, 4, 3, 8, 8)
    second = first.clone()
    second[:, -1] += 5.0

    with torch.no_grad():
        before = model(first)
        after = model(second)

    assert torch.max(torch.abs(before[:, 0] - after[:, 0])) > 1e-6


def test_post_transformer_dense_layer_is_not_discarded() -> None:
    torch.manual_seed(29)
    model = VPTPaperIDM(tiny_config()).eval()
    frames = torch.rand(1, 4, 3, 8, 8)
    with torch.no_grad():
        model.net.lastlayer.layer.weight.zero_()
        logits = model(frames)
    assert torch.equal(logits, torch.zeros_like(logits))


@pytest.mark.requires_private_artifacts(
    "results/idm/vpt_paper_idm_reference/official_artifacts.json"
)
def test_tracked_official_receipt_matches_reconstructed_body() -> None:
    audit = validate(
        Path("results/idm/vpt_paper_idm_reference/official_artifacts.json")
    )
    assert audit["result"] == "pass"
    assert audit["body_name_shape_match"] == {
        "matched_tensors": 96,
        "missing": [],
        "extra": [],
        "shape_mismatches": [],
    }
    assert audit["parameter_counts"]["celeste_reconstruction_total"] == 482_133_390


def test_fp32_parity_comparator_fails_closed() -> None:
    reference = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    result = compare_outputs("fixture", reference.clone(), reference)
    assert result["result"] == "pass"
    assert result["exact"] is True

    with pytest.raises(ValueError, match="FP32 parity failed"):
        compare_outputs("fixture", reference + 0.1, reference)


def test_fp32_gradient_parity_comparator_fails_closed() -> None:
    reproduction = torch.nn.Linear(2, 2)
    upstream = torch.nn.Linear(2, 2)
    upstream.load_state_dict(reproduction.state_dict())
    fixture = torch.tensor([[0.25, -0.5]])

    reproduction(fixture).square().sum().backward()
    upstream(fixture).square().sum().backward()
    result = compare_parameter_gradients("fixture", reproduction, upstream)
    assert result["result"] == "pass"
    assert result["exact"] is True
    assert result["byte_exact"] is True

    upstream.zero_grad(set_to_none=True)
    upstream(fixture).sum().backward()
    with pytest.raises(ValueError, match="FP32 gradient parity failed"):
        compare_parameter_gradients("fixture", reproduction, upstream)
