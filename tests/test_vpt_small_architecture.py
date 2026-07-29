from __future__ import annotations

import torch

from badeline.vpt_small import (
    ClipConsistentAugmentation,
    VPTAugmentationConfig,
    VPTSmallConfig,
    VPTSmallIDM,
    natural_factored_nll,
    parameter_inventory,
)


def tiny_config(**overrides) -> VPTSmallConfig:
    values = {
        "frames": 8,
        "image_size": 16,
        "temporal_channels": 8,
        "spatial_widths": (8, 8, 8),
        "frame_hidden": 16,
        "d_model": 16,
        "attention_heads": 4,
        "mlp_width": 32,
        "transformer_blocks": 2,
        "activation_checkpointing": False,
    }
    values.update(overrides)
    return VPTSmallConfig(**values)


def test_production_parameter_count_and_topology_on_meta_device() -> None:
    with torch.device("meta"):
        model = VPTSmallIDM(VPTSmallConfig())
    inventory = parameter_inventory(model)
    assert inventory["total"] == 105_696_398
    assert len(model.spatial_stacks) == 3
    assert all(len(stack.blocks) == 2 for stack in model.spatial_stacks)
    assert len(model.transformer) == 4
    assert len(model.action_heads) == 7
    assert model.flattened_width == 32_768


def test_temporal_conv_receptive_field_is_exactly_five_positions() -> None:
    model = VPTSmallIDM(tiny_config())
    with torch.no_grad():
        model.temporal_conv.weight.fill_(1.0)
        model.temporal_conv.bias.zero_()
    impulse = torch.zeros(1, 3, 8, 1, 1)
    impulse[:, :, 4] = 1.0
    response = model.temporal_conv(impulse)[0, 0, :, 0, 0]
    assert torch.equal(
        torch.nonzero(response, as_tuple=True)[0], torch.tensor([2, 3, 4, 5, 6])
    )


def test_dense_logits_and_edge_and_center_gradients() -> None:
    torch.manual_seed(3)
    model = VPTSmallIDM(tiny_config())
    frames = torch.rand(1, 8, 3, 16, 16)
    targets = torch.randint(0, 2, (1, 8, 7))
    logits = model(frames)
    assert logits.shape == (1, 8, 7, 2)
    logits.retain_grad()
    natural_factored_nll(logits, targets).backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 0])
    assert torch.count_nonzero(logits.grad[:, 4])
    assert torch.count_nonzero(logits.grad[:, -1])


def test_unmasked_attention_allows_future_to_change_earlier_output() -> None:
    torch.manual_seed(4)
    model = VPTSmallIDM(tiny_config(transformer_blocks=1)).eval()
    first = torch.rand(1, 8, 3, 16, 16)
    second = first.clone()
    second[:, -1] += 0.5
    with torch.no_grad():
        before = model(first)
        after = model(second)
    assert not torch.allclose(before[:, 0], after[:, 0])


def test_natural_factored_nll_matches_hand_computation() -> None:
    logits = torch.tensor([[[[2.0, -1.0], [0.5, 0.0]]]])
    targets = torch.tensor([[[0, 1]]])
    expected = -torch.log_softmax(logits, dim=-1)[0, 0, 0, 0]
    expected += -torch.log_softmax(logits, dim=-1)[0, 0, 1, 1]
    assert torch.allclose(natural_factored_nll(logits, targets), expected)


def test_augmentation_draw_is_clip_consistent() -> None:
    height = width = 16
    base = torch.linspace(0.0, 1.0, height * width).reshape(1, 1, height, width)
    frame = base.repeat(1, 3, 1, 1)
    clips = frame.repeat(2, 4, 1, 1, 1)
    augmentation = ClipConsistentAugmentation(VPTAugmentationConfig())
    output = augmentation(clips, generator=torch.Generator().manual_seed(9))
    assert torch.equal(output[0, 0], output[0, 1])
    assert torch.equal(output[1, 0], output[1, -1])
    assert not torch.equal(output[0, 0], output[1, 0])
