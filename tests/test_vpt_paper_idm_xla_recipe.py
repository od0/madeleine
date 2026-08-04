from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from badeline.vpt_paper_idm import VPTPaperIDMConfig
from badeline.vpt_small import (
    ClipConsistentAugmentation,
    VPTAugmentationConfig,
)
from badeline.vpt_xla import (
    PARAMETER_ORDER,
    apply_clip_consistent_augmentation,
    sample_clip_augmentation_parameters,
)


def test_xla_parameter_draw_order_matches_completed_cuda_augmentation() -> None:
    config = VPTAugmentationConfig()
    expected_generator = torch.Generator().manual_seed(519)
    actual_generator = torch.Generator().manual_seed(519)
    completed = ClipConsistentAugmentation(config)
    expected = [
        completed.sample_parameters(device=torch.device("cpu"), generator=expected_generator)
        for _ in range(3)
    ]
    actual = sample_clip_augmentation_parameters(
        3, config, generator=actual_generator
    )
    for row, expected_row in enumerate(expected):
        for name in PARAMETER_ORDER:
            expected_value = expected_row[name]
            if name.startswith("translate_"):
                expected_value = int(round(expected_value))
            assert actual[name][row].item() == expected_value


def test_xla_augmentation_matches_torchvision_tensor_semantics() -> None:
    config = VPTAugmentationConfig()
    generator = torch.Generator().manual_seed(1907)
    parameters = sample_clip_augmentation_parameters(2, config, generator=generator)
    frames = torch.rand((2, 4, 3, 32, 32), generator=torch.Generator().manual_seed(91))
    completed = ClipConsistentAugmentation(config)
    expected_clips = []
    for row, clip in enumerate(frames):
        row_parameters = {
            name: float(parameters[name][row]) for name in PARAMETER_ORDER
        }
        expected_clips.append(completed.apply_parameters(clip, row_parameters))
    expected = torch.stack(expected_clips)
    actual = apply_clip_consistent_augmentation(frames, parameters)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_xla_augmentation_is_clip_consistent() -> None:
    config = VPTAugmentationConfig()
    parameters = sample_clip_augmentation_parameters(
        1, config, generator=torch.Generator().manual_seed(4)
    )
    frame = torch.rand((1, 1, 3, 24, 24), generator=torch.Generator().manual_seed(5))
    clip = frame.expand(-1, 5, -1, -1, -1).clone()
    result = apply_clip_consistent_augmentation(clip, parameters)
    for index in range(1, result.shape[1]):
        torch.testing.assert_close(result[:, 0], result[:, index], rtol=0, atol=0)


@pytest.mark.requires_private_artifacts(
    "experiments/configs/vpt_paper_idm_tpu_tier_b.json"
)
def test_tier_b_xla_config_freezes_paper_recipe_and_endpoint() -> None:
    path = Path("experiments/configs/vpt_paper_idm_tpu_tier_b.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    model = VPTPaperIDMConfig.from_dict(config["model"])
    training = config["training"]
    assert model.to_dict() == config["model"]
    assert training["epochs"] == 20
    assert training["global_batch"] == 128
    assert training["optimizer"] == "adam_coupled_weight_decay"
    assert training["learning_rate"] == 0.003
    assert training["weight_decay"] == 0.01
    assert training["loss"] == "natural_factored_nll"
    steps = math.ceil(training["train_windows"] / training["global_batch"])
    assert steps == training["optimizer_steps_per_epoch"] == 117
    assert steps * training["global_batch"] - training["train_windows"] == 55
    assert steps * training["epochs"] == training["production_optimizer_steps"] == 2340
    assert config["xla"]["project"] == "madeleine-idm"
    assert config["xla"]["zone"] == "us-east5-a"
    assert config["xla"]["production_accelerator_type"] == "v6e-4"
    assert config["inference"]["fixed_probability_threshold"] == 0.5
