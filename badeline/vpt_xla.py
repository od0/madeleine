"""XLA-safe mechanics for the paper-IDM TPU trainer.

The scientific augmentation is defined by :class:`VPTAugmentationConfig` and
the completed CUDA VPT-small trainer.  This module keeps the same draw order
and torchvision tensor semantics while expressing the image operations with
static-shape PyTorch operators that PyTorch/XLA can lower as one graph.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from badeline.vpt_small import VPTAugmentationConfig


PARAMETER_ORDER = (
    "hue",
    "saturation",
    "brightness",
    "contrast",
    "rotation",
    "scale",
    "shear",
    "translate_x",
    "translate_y",
)


def _inverse_affine_matrix(
    *, angle: float, translate_x: int, translate_y: int, scale: float, shear: float
) -> list[float]:
    """Match torchvision's centered inverse affine matrix for a zero center."""

    rotation = math.radians(angle)
    shear_x = math.radians(shear)
    a = math.cos(rotation)
    b = -math.cos(rotation) * math.tan(shear_x) - math.sin(rotation)
    c = math.sin(rotation)
    d = -math.sin(rotation) * math.tan(shear_x) + math.cos(rotation)
    matrix = [d / scale, -b / scale, 0.0, -c / scale, a / scale, 0.0]
    matrix[2] += matrix[0] * -translate_x + matrix[1] * -translate_y
    matrix[5] += matrix[3] * -translate_x + matrix[4] * -translate_y
    return matrix


def sample_clip_augmentation_parameters(
    batch_size: int,
    config: VPTAugmentationConfig,
    *,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    """Draw one host-side parameter row per clip in the CUDA trainer's order."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    draws = torch.rand((batch_size, len(PARAMETER_ORDER)), generator=generator)
    lows = torch.tensor(
        [
            -config.hue,
            1.0 - config.saturation,
            1.0 - config.brightness,
            1.0 - config.contrast,
            -config.rotation_degrees,
            1.0 - config.scale,
            -config.shear_degrees,
            -config.translate_pixels,
            -config.translate_pixels,
        ],
        dtype=torch.float32,
    )
    spans = torch.tensor(
        [
            2.0 * config.hue,
            2.0 * config.saturation,
            2.0 * config.brightness,
            2.0 * config.contrast,
            2.0 * config.rotation_degrees,
            2.0 * config.scale,
            2.0 * config.shear_degrees,
            2.0 * config.translate_pixels,
            2.0 * config.translate_pixels,
        ],
        dtype=torch.float32,
    )
    values = lows + spans * draws
    result = {name: values[:, index] for index, name in enumerate(PARAMETER_ORDER)}
    translate_x = [int(round(float(value))) for value in result["translate_x"]]
    translate_y = [int(round(float(value))) for value in result["translate_y"]]
    result["translate_x"] = torch.tensor(translate_x, dtype=torch.int64)
    result["translate_y"] = torch.tensor(translate_y, dtype=torch.int64)
    matrices = [
        _inverse_affine_matrix(
            angle=float(result["rotation"][index]),
            translate_x=translate_x[index],
            translate_y=translate_y[index],
            scale=float(result["scale"][index]),
            shear=float(result["shear"][index]),
        )
        for index in range(batch_size)
    ]
    result["affine_matrix"] = torch.tensor(matrices, dtype=torch.float32).reshape(
        batch_size, 2, 3
    )
    return result


def _per_clip(value: Tensor, frames: Tensor) -> Tensor:
    return value.to(device=frames.device, dtype=frames.dtype).reshape(-1, 1, 1, 1, 1)


def _blend(value: Tensor, other: Tensor, ratio: Tensor) -> Tensor:
    return (ratio * value + (1.0 - ratio) * other).clamp(0.0, 1.0)


def _grayscale(value: Tensor) -> Tensor:
    weights = value.new_tensor((0.2989, 0.5870, 0.1140)).reshape(1, 1, 3, 1, 1)
    return (value * weights).sum(dim=2, keepdim=True)


def _rgb_to_hsv(value: Tensor) -> Tensor:
    red, green, blue = value.unbind(dim=2)
    maximum = value.max(dim=2).values
    minimum = value.min(dim=2).values
    equal = maximum == minimum
    chroma = maximum - minimum
    ones = torch.ones_like(maximum)
    saturation = chroma / torch.where(equal, ones, maximum)
    divisor = torch.where(equal, ones, chroma)
    red_component = (maximum - red) / divisor
    green_component = (maximum - green) / divisor
    blue_component = (maximum - blue) / divisor
    hue_red = (maximum == red) * (blue_component - green_component)
    hue_green = ((maximum == green) & (maximum != red)) * (
        2.0 + red_component - blue_component
    )
    hue_blue = ((maximum != green) & (maximum != red)) * (
        4.0 + green_component - red_component
    )
    hue = torch.fmod((hue_red + hue_green + hue_blue) / 6.0 + 1.0, 1.0)
    return torch.stack((hue, saturation, maximum), dim=2)


def _hsv_to_rgb(value: Tensor) -> Tensor:
    hue, saturation, maximum = value.unbind(dim=2)
    sector_float = torch.floor(hue * 6.0)
    fraction = hue * 6.0 - sector_float
    sector = sector_float.to(dtype=torch.int32) % 6
    p = (maximum * (1.0 - saturation)).clamp(0.0, 1.0)
    q = (maximum * (1.0 - saturation * fraction)).clamp(0.0, 1.0)
    t = (maximum * (1.0 - saturation * (1.0 - fraction))).clamp(0.0, 1.0)
    choices = torch.stack(
        (
            torch.stack((maximum, t, p), dim=2),
            torch.stack((q, maximum, p), dim=2),
            torch.stack((p, maximum, t), dim=2),
            torch.stack((p, q, maximum), dim=2),
            torch.stack((t, p, maximum), dim=2),
            torch.stack((maximum, p, q), dim=2),
        ),
        dim=2,
    )
    index = sector[:, :, None, None, :, :].expand(-1, -1, 1, 3, -1, -1)
    return torch.gather(choices, dim=2, index=index).squeeze(2)


def _affine_grid(theta: Tensor, *, width: int, height: int) -> Tensor:
    half = 0.5
    x = torch.linspace(
        -width * 0.5 + half,
        width * 0.5 + half - 1,
        steps=width,
        device=theta.device,
        dtype=theta.dtype,
    )
    y = torch.linspace(
        -height * 0.5 + half,
        height * 0.5 + half - 1,
        steps=height,
        device=theta.device,
        dtype=theta.dtype,
    )
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    base = torch.stack((x_grid, y_grid, torch.ones_like(x_grid)), dim=-1)
    base = base.reshape(1, height * width, 3).expand(theta.shape[0], -1, -1)
    denominator = theta.new_tensor((0.5 * width, 0.5 * height))
    grid = torch.bmm(base, theta.transpose(1, 2) / denominator)
    return grid.reshape(theta.shape[0], height, width, 2)


def apply_clip_consistent_augmentation(
    frames: Tensor, parameters: Mapping[str, Tensor]
) -> Tensor:
    """Apply torchvision-equivalent color and affine operations to static clips."""

    if frames.ndim != 5 or frames.shape[2] != 3:
        raise ValueError("frames must have shape [B,T,3,H,W]")
    batch, steps, channels, height, width = frames.shape
    if parameters["hue"].shape != (batch,):
        raise ValueError("augmentation parameters do not match batch size")

    value = _blend(
        frames,
        torch.zeros_like(frames),
        _per_clip(parameters["brightness"], frames),
    )
    contrast_mean = _grayscale(value).mean(dim=(2, 3, 4), keepdim=True)
    value = _blend(
        value, contrast_mean, _per_clip(parameters["contrast"], frames)
    )
    value = _blend(
        value, _grayscale(value), _per_clip(parameters["saturation"], frames)
    )
    hsv = _rgb_to_hsv(value)
    hue, saturation, maximum = hsv.unbind(dim=2)
    hue = torch.remainder(
        hue + parameters["hue"].to(value.device, value.dtype)[:, None, None, None],
        1.0,
    )
    value = _hsv_to_rgb(torch.stack((hue, saturation, maximum), dim=2))

    theta = parameters["affine_matrix"].to(value.device, value.dtype)
    theta = theta.repeat_interleave(steps, dim=0)
    flattened = value.reshape(batch * steps, channels, height, width)
    mask = torch.ones(
        (batch * steps, 1, height, width), device=value.device, dtype=value.dtype
    )
    combined = torch.cat((flattened, mask), dim=1)
    transformed = F.grid_sample(
        combined,
        _affine_grid(theta, width=width, height=height),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    image, coverage = transformed[:, :channels], transformed[:, channels:]
    image = image * coverage
    return image.reshape(batch, steps, channels, height, width)
