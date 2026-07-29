"""Paper-shaped, width-reduced Video PreTraining inverse-dynamics model.

This module intentionally does not share the historical Badeline GRU path.
The production configuration keeps VPT's 128-frame raw-pixel input, early
non-causal temporal convolution, Appendix-D spatial stack, dense temporal
predictions, natural factored key likelihood, and four unmasked Transformer
blocks.  Only the Transformer width is reduced.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class VPTSmallConfig:
    """Architecture-only fields for the frozen VPT-small graph."""

    frames: int = 128
    image_size: int = 128
    input_channels: int = 3
    temporal_channels: int = 128
    temporal_kernel: int = 5
    spatial_widths: tuple[int, int, int] = (64, 128, 128)
    residual_blocks_per_stack: int = 2
    frame_hidden: int = 256
    d_model: int = 1408
    attention_heads: int = 11
    mlp_width: int = 5632
    transformer_blocks: int = 4
    action_keys: int = 7
    activation_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.frames < 1 or self.image_size < 8:
            raise ValueError("frames must be positive and image_size at least 8")
        if self.temporal_kernel < 1 or self.temporal_kernel % 2 != 1:
            raise ValueError("temporal_kernel must be a positive odd integer")
        if len(self.spatial_widths) != 3:
            raise ValueError("Appendix-D path requires exactly three stacks")
        if self.residual_blocks_per_stack != 2:
            raise ValueError("Appendix-D path requires two blocks per stack")
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if self.action_keys < 1:
            raise ValueError("action_keys must be positive")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VPTSmallConfig":
        values = dict(raw)
        if "spatial_widths" in values:
            values["spatial_widths"] = tuple(values["spatial_widths"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["spatial_widths"] = list(self.spatial_widths)
        return result


@dataclass(frozen=True)
class VPTAugmentationConfig:
    """VPT's clip-consistent visual augmentation ranges."""

    hue: float = 0.2
    saturation: float = 0.2
    brightness: float = 0.2
    contrast: float = 0.2
    rotation_degrees: float = 2.0
    scale: float = 0.02
    shear_degrees: float = 2.0
    translate_pixels: float = 2.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VPTAugmentationConfig":
        return cls(**raw)


def _fan_in_init(module: nn.Module, scale: float = 1.0) -> None:
    """Apply released VPT's output-filter-normalized fan-in initialization."""

    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
        with torch.no_grad():
            dimensions = tuple(range(1, module.weight.ndim))
            norm = module.weight.norm(dim=dimensions, p=2, keepdim=True)
            module.weight.mul_(float(scale) / norm.clamp_min(1e-12))
            if module.bias is not None:
                module.bias.zero_()


class PreNormConv2d(nn.Module):
    """LayerNorm-equivalent channel normalization before a spatial conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, in_channels)
        self.conv = nn.Conv2d(
            in_channels, out_channels, 3, padding=1, bias=False
        )

    def forward(self, value: Tensor) -> Tensor:
        return F.relu(self.conv(self.norm(value)))


class VPTResidualBlock(nn.Module):
    """Two-convolution pre-normalized residual block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv0 = PreNormConv2d(channels, channels)
        self.conv1 = PreNormConv2d(channels, channels)

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = self.conv0(value)
        value = self.conv1(value)
        return value + residual


class VPTSpatialStack(nn.Module):
    """One Appendix-D convolution, pool, and two-block stack."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.entry = PreNormConv2d(in_channels, out_channels)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.blocks = nn.ModuleList(
            [VPTResidualBlock(out_channels), VPTResidualBlock(out_channels)]
        )

    def forward(self, value: Tensor) -> Tensor:
        value = self.pool(self.entry(value))
        for block in self.blocks:
            value = block(value)
        return value


class VPTTransformerBlock(nn.Module):
    """Pre-normalized, fully bidirectional residual Transformer block."""

    def __init__(self, d_model: int, heads: int, mlp_width: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, heads, dropout=0.0, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp_in = nn.Linear(d_model, mlp_width)
        self.mlp_out = nn.Linear(mlp_width, d_model)

    def forward(self, value: Tensor) -> Tensor:
        normalized = self.attention_norm(value)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
            attn_mask=None,
            is_causal=False,
        )
        value = value + attended
        hidden = F.relu(self.mlp_in(self.mlp_norm(value)))
        return value + self.mlp_out(hidden)


class ClipConsistentAugmentation(nn.Module):
    """Sample one color/affine transform and reuse it for every clip frame."""

    def __init__(self, config: VPTAugmentationConfig) -> None:
        super().__init__()
        self.config = config

    @staticmethod
    def _uniform(
        low: float, high: float, *, device: torch.device, generator: torch.Generator | None
    ) -> float:
        draw = torch.rand((), device=device, generator=generator)
        return float((low + (high - low) * draw).item())

    def sample_parameters(
        self, *, device: torch.device, generator: torch.Generator | None = None
    ) -> dict[str, float]:
        cfg = self.config
        return {
            "hue": self._uniform(-cfg.hue, cfg.hue, device=device, generator=generator),
            "saturation": self._uniform(
                1.0 - cfg.saturation,
                1.0 + cfg.saturation,
                device=device,
                generator=generator,
            ),
            "brightness": self._uniform(
                1.0 - cfg.brightness,
                1.0 + cfg.brightness,
                device=device,
                generator=generator,
            ),
            "contrast": self._uniform(
                1.0 - cfg.contrast,
                1.0 + cfg.contrast,
                device=device,
                generator=generator,
            ),
            "rotation": self._uniform(
                -cfg.rotation_degrees,
                cfg.rotation_degrees,
                device=device,
                generator=generator,
            ),
            "scale": self._uniform(
                1.0 - cfg.scale,
                1.0 + cfg.scale,
                device=device,
                generator=generator,
            ),
            "shear": self._uniform(
                -cfg.shear_degrees,
                cfg.shear_degrees,
                device=device,
                generator=generator,
            ),
            "translate_x": self._uniform(
                -cfg.translate_pixels,
                cfg.translate_pixels,
                device=device,
                generator=generator,
            ),
            "translate_y": self._uniform(
                -cfg.translate_pixels,
                cfg.translate_pixels,
                device=device,
                generator=generator,
            ),
        }

    @staticmethod
    def apply_parameters(frames: Tensor, parameters: dict[str, float]) -> Tensor:
        """Apply one parameter dictionary to ``[T,C,H,W]`` frames."""

        value = TF.adjust_brightness(frames, parameters["brightness"])
        value = TF.adjust_contrast(value, parameters["contrast"])
        value = TF.adjust_saturation(value, parameters["saturation"])
        value = TF.adjust_hue(value, parameters["hue"])
        return TF.affine(
            value,
            angle=parameters["rotation"],
            translate=[
                int(round(parameters["translate_x"])),
                int(round(parameters["translate_y"])),
            ],
            scale=parameters["scale"],
            shear=[parameters["shear"], 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
            center=None,
        )

    def forward(
        self, frames: Tensor, *, generator: torch.Generator | None = None
    ) -> Tensor:
        if frames.ndim != 5:
            raise ValueError("augmentation expects [B,T,C,H,W]")
        transformed = []
        for clip in frames:
            parameters = self.sample_parameters(
                device=clip.device, generator=generator
            )
            transformed.append(self.apply_parameters(clip, parameters))
        return torch.stack(transformed)


class VPTSmallIDM(nn.Module):
    """Width-reduced VPT architecture with dense seven-key predictions."""

    def __init__(self, config: VPTSmallConfig) -> None:
        super().__init__()
        self.config = config
        self.temporal_conv = nn.Conv3d(
            config.input_channels,
            config.temporal_channels,
            kernel_size=(config.temporal_kernel, 1, 1),
            padding=(config.temporal_kernel // 2, 0, 0),
        )
        spatial: list[VPTSpatialStack] = []
        input_width = config.temporal_channels
        for width in config.spatial_widths:
            spatial.append(VPTSpatialStack(input_width, width))
            input_width = width
        self.spatial_stacks = nn.ModuleList(spatial)

        final_size = config.image_size
        for _ in config.spatial_widths:
            final_size = math.ceil(final_size / 2)
        self.spatial_output_size = final_size
        self.flattened_width = input_width * final_size * final_size
        self.frame_norm = nn.LayerNorm(self.flattened_width)
        self.frame_hidden = nn.Linear(
            self.flattened_width, config.frame_hidden, bias=False
        )
        self.frame_projection_norm = nn.LayerNorm(config.frame_hidden)
        self.frame_projection = nn.Linear(
            config.frame_hidden, config.d_model, bias=False
        )
        self.transformer = nn.ModuleList(
            [
                VPTTransformerBlock(
                    config.d_model, config.attention_heads, config.mlp_width
                )
                for _ in range(config.transformer_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.action_heads = nn.ModuleList(
            [nn.Linear(config.d_model, 2) for _ in range(config.action_keys)]
        )
        self.apply(_fan_in_init)

    def encode_frames(self, frames: Tensor) -> Tensor:
        """Return dense embeddings from normalized ``[B,T,C,H,W]`` pixels."""

        if frames.ndim != 5:
            raise ValueError("frames must have shape [B,T,C,H,W]")
        batch, steps, channels, height, width = frames.shape
        expected = (
            self.config.frames,
            self.config.input_channels,
            self.config.image_size,
            self.config.image_size,
        )
        if (steps, channels, height, width) != expected:
            raise ValueError(
                "frames have incorrect shape after batch dimension: "
                f"{(steps, channels, height, width)} != {expected}"
            )
        value = frames.permute(0, 2, 1, 3, 4)
        value = F.relu(self.temporal_conv(value))
        value = value.permute(0, 2, 1, 3, 4).reshape(
            batch * steps, self.config.temporal_channels, height, width
        )
        for stack in self.spatial_stacks:
            value = stack(value)
        value = value.flatten(1)
        value = F.relu(self.frame_hidden(self.frame_norm(value)))
        value = F.relu(self.frame_projection(self.frame_projection_norm(value)))
        return value.reshape(batch, steps, self.config.d_model)

    def forward(self, frames: Tensor) -> Tensor:
        value = self.encode_frames(frames)
        use_checkpoint = self.config.activation_checkpointing and self.training
        for block in self.transformer:
            if use_checkpoint:
                value = checkpoint(block, value, use_reentrant=False)
            else:
                value = block(value)
        value = self.output_norm(value)
        return torch.stack([head(value) for head in self.action_heads], dim=2)


def natural_factored_nll(
    logits: Tensor, targets: Tensor, supported: Tensor | None = None
) -> Tensor:
    """Sum seven natural two-class NLLs, each averaged on its support."""

    if logits.ndim != 4 or logits.shape[-1] != 2:
        raise ValueError("logits must have shape [B,T,K,2]")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must match logits [B,T,K]")
    if supported is None:
        supported = torch.ones_like(targets, dtype=torch.bool)
    if supported.shape != targets.shape:
        raise ValueError("supported must match targets")
    losses = F.cross_entropy(
        logits.reshape(-1, 2), targets.to(torch.long).reshape(-1), reduction="none"
    ).reshape_as(targets)
    per_key: list[Tensor] = []
    for column in range(logits.shape[2]):
        mask = supported[..., column]
        if not torch.any(mask):
            raise ValueError(f"key {column} has no label-supported positions")
        per_key.append(losses[..., column][mask].mean())
    return torch.stack(per_key).sum()


def parameter_inventory(model: nn.Module) -> dict[str, Any]:
    """Return content-ready exact trainable counts by top-level component."""

    by_component: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            component = name.split(".", 1)[0]
            by_component[component] = by_component.get(component, 0) + parameter.numel()
    return {
        "total": sum(by_component.values()),
        "by_component": dict(sorted(by_component.items())),
    }


def maybe_autocast(device: torch.device, dtype: torch.dtype):
    """Small shared helper that keeps CPU unit tests unambiguous."""

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()
