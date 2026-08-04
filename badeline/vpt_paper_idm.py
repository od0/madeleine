"""Released-artifact reconstruction of OpenAI's approximately 0.5B VPT IDM.

This is deliberately separate from :mod:`badeline.vpt_small`.  The module
names and tensor shapes reconstruct the body recorded by the official
``4x_idm.model`` and ``4x_idm.weights`` artifacts.  The Minecraft action heads
are replaced with seven natural binary Celeste heads.

OpenAI did not release the original IDM training source.  The forward path
therefore follows the released config, weight inventory, and the pinned public
VPT graph.  In particular it retains the public base-policy post-Transformer
dense layer; the public demo's IDM override computes that layer and then
accidentally discards it, one reason the project does not claim bit-exact
private-code reproduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class VPTPaperIDMConfig:
    frames: int = 128
    image_size: int = 128
    input_channels: int = 3
    temporal_channels: int = 128
    temporal_kernel: int = 5
    spatial_widths: tuple[int, int, int] = (256, 512, 512)
    residual_blocks_per_stack: int = 2
    frame_hidden: int = 256
    d_model: int = 4096
    attention_heads: int = 32
    pointwise_ratio: int = 4
    transformer_blocks: int = 2
    relative_attention_bases: int = 10
    action_keys: int = 7
    action_temperature: float = 4.0
    activation_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.frames < 1 or self.image_size < 8:
            raise ValueError("frames must be positive and image_size at least 8")
        if self.temporal_kernel < 1 or self.temporal_kernel % 2 != 1:
            raise ValueError("temporal kernel must be a positive odd integer")
        if len(self.spatial_widths) != 3:
            raise ValueError("released IDM requires three spatial stacks")
        if self.residual_blocks_per_stack != 2:
            raise ValueError("released IDM requires two residual blocks per stack")
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if self.transformer_blocks != 2:
            raise ValueError("released 4x artifact requires two Transformer blocks")
        if self.pointwise_ratio != 4:
            raise ValueError("released 4x artifact requires pointwise ratio four")
        if self.action_keys < 1 or self.action_temperature <= 0:
            raise ValueError("action head dimensions and temperature must be positive")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VPTPaperIDMConfig":
        values = dict(raw)
        if "spatial_widths" in values:
            values["spatial_widths"] = tuple(values["spatial_widths"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["spatial_widths"] = list(self.spatial_widths)
        return result


def _normalize_output_filters(weight: Tensor, scale: float) -> None:
    with torch.no_grad():
        dimensions = tuple(range(1, weight.ndim))
        norm = weight.norm(dim=dimensions, p=2, keepdim=True)
        weight.mul_(float(scale) / norm.clamp_min(1e-12))


class _FanInLayer(nn.Module):
    """Public VPT FanInInitReLULayer with stable submodule names."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        kind: str,
        scale: float = 1.0,
        layer_norm: bool = False,
        group_norm_groups: int | None = None,
        use_activation: bool = True,
        kernel_size: int | tuple[int, int, int] = 3,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__()
        if layer_norm and group_norm_groups is not None:
            raise ValueError("a fan-in layer can have only one normalization")
        if layer_norm:
            self.norm: nn.Module | None = nn.LayerNorm(in_features)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, in_features)
        else:
            self.norm = None
        bias = self.norm is None
        if kind == "linear":
            self.layer: nn.Module = nn.Linear(in_features, out_features, bias=bias)
        elif kind == "conv2d":
            self.layer = nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            )
        elif kind == "conv3d":
            self.layer = nn.Conv3d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            )
        else:
            raise ValueError(f"unknown fan-in layer kind: {kind}")
        _normalize_output_filters(self.layer.weight, scale)  # type: ignore[attr-defined]
        if self.layer.bias is not None:  # type: ignore[attr-defined]
            nn.init.zeros_(self.layer.bias)  # type: ignore[attr-defined]
        self.use_activation = use_activation

    def forward(self, value: Tensor) -> Tensor:
        if self.norm is not None:
            value = self.norm(value)
        value = self.layer(value)
        return F.relu(value) if self.use_activation else value


class _CnnBasicBlock(nn.Module):
    def __init__(self, channels: int, *, init_scale: float) -> None:
        super().__init__()
        scale = math.sqrt(init_scale)
        self.conv0 = _FanInLayer(
            channels,
            channels,
            kind="conv2d",
            scale=scale,
            group_norm_groups=1,
            kernel_size=3,
            padding=1,
        )
        self.conv1 = _FanInLayer(
            channels,
            channels,
            kind="conv2d",
            scale=scale,
            group_norm_groups=1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.conv1(self.conv0(value))


class _CnnDownStack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stack_count: int) -> None:
        super().__init__()
        self.firstconv = _FanInLayer(
            in_channels,
            out_channels,
            kind="conv2d",
            group_norm_groups=1,
            kernel_size=3,
            padding=1,
        )
        self.n = nn.GroupNorm(1, out_channels)
        block_init_scale = math.sqrt(stack_count) / 2**0.5
        self.blocks = nn.ModuleList(
            [
                _CnnBasicBlock(out_channels, init_scale=block_init_scale),
                _CnnBasicBlock(out_channels, init_scale=block_init_scale),
            ]
        )

    def forward(self, value: Tensor) -> Tensor:
        value = self.firstconv(value)
        value = F.max_pool2d(value, kernel_size=3, stride=2, padding=1)
        value = self.n(value)
        for block in self.blocks:
            value = block(value)
        return value


class _ImpalaCNN(nn.Module):
    def __init__(self, config: VPTPaperIDMConfig) -> None:
        super().__init__()
        stacks = []
        input_width = config.temporal_channels
        for output_width in config.spatial_widths:
            stacks.append(
                _CnnDownStack(
                    input_width, output_width, stack_count=len(config.spatial_widths)
                )
            )
            input_width = output_width
        self.stacks = nn.ModuleList(stacks)
        final_size = config.image_size
        for _ in config.spatial_widths:
            final_size = math.ceil(final_size / 2)
        self.output_size = final_size
        self.flattened_width = input_width * final_size * final_size
        self.dense = _FanInLayer(
            self.flattened_width,
            config.frame_hidden,
            kind="linear",
            scale=1.4,
            layer_norm=True,
        )

    def forward(self, value: Tensor) -> Tensor:
        for stack in self.stacks:
            value = stack(value)
        return self.dense(value.flatten(1))


class _ImageProcess(nn.Module):
    def __init__(self, config: VPTPaperIDMConfig) -> None:
        super().__init__()
        self.cnn = _ImpalaCNN(config)
        self.linear = _FanInLayer(
            config.frame_hidden,
            config.d_model,
            kind="linear",
            layer_norm=True,
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.linear(self.cnn(value))


def _fan_in_linear(
    in_features: int, out_features: int, *, scale: float, bias: bool
) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    _normalize_output_filters(layer.weight, scale)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class _AttentionCore(nn.Module):
    """Unmasked released attention, including inert relative tensors."""

    def __init__(self, config: VPTPaperIDMConfig, *, init_scale: float) -> None:
        super().__init__()
        width = config.d_model
        root_scale = math.sqrt(init_scale)
        self.b_nd = nn.Parameter(torch.empty(config.relative_attention_bases, 0))
        self.q_layer = _fan_in_linear(width, width, scale=0.1, bias=True)
        self.k_layer = _fan_in_linear(width, width, scale=0.2, bias=False)
        self.v_layer = _fan_in_linear(width, width, scale=root_scale, bias=False)
        self.proj_layer = _fan_in_linear(width, width, scale=root_scale, bias=True)
        self.r_layer = _fan_in_linear(
            width,
            config.relative_attention_bases * config.attention_heads,
            scale=0.1,
            bias=True,
        )
        self.heads = config.attention_heads
        self.head_width = width // config.attention_heads

    def forward(self, value: Tensor) -> Tensor:
        batch, steps, width = value.shape
        def split_heads(item: Tensor) -> Tensor:
            return item.reshape(batch, steps, self.heads, self.head_width).transpose(1, 2)

        query = split_heads(self.q_layer(value))
        key = split_heads(self.k_layer(value))
        projected_value = split_heads(self.v_layer(value))
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
        scores = scores * (1.0 / self.head_width)

        # The released config has memory_size == timesteps, hence b_nd is
        # [10, 0] and the relative score is identically zero.  Retain both
        # tensors in the graph so their gradients are exactly zero, not absent.
        relative = self.r_layer(value).sum(dim=-1, keepdim=True) * 0.0
        relative = relative[:, None].expand(batch, self.heads, steps, steps)
        scores = scores + relative + self.b_nd.sum() * 0.0
        weights = torch.softmax(scores, dim=-1).to(projected_value.dtype)
        attended = torch.matmul(weights, projected_value)
        attended = attended.transpose(1, 2).reshape(batch, steps, width)
        return value + self.proj_layer(attended)


class _ReleasedAttention(nn.Module):
    def __init__(self, config: VPTPaperIDMConfig, *, init_scale: float) -> None:
        super().__init__()
        self.orc_block = _AttentionCore(config, init_scale=init_scale)

    def forward(self, value: Tensor) -> Tensor:
        return self.orc_block(value)


class _TransformerBlock(nn.Module):
    def __init__(self, config: VPTPaperIDMConfig, *, init_scale: float) -> None:
        super().__init__()
        width = config.d_model
        self.mlp0 = _FanInLayer(
            width,
            width * config.pointwise_ratio,
            kind="linear",
            layer_norm=True,
        )
        self.mlp1 = _FanInLayer(
            width * config.pointwise_ratio,
            width,
            kind="linear",
            scale=init_scale * 2**-0.5,
            use_activation=False,
        )
        self.pre_r_ln = nn.LayerNorm(width)
        self.r = _ReleasedAttention(config, init_scale=init_scale)

    def forward(self, value: Tensor) -> Tensor:
        # The attention residual is around the pre-normalized value in the
        # released public graph; there is no second outer attention residual.
        value = self.r(self.pre_r_ln(value))
        return value + self.mlp1(self.mlp0(value))


class _RecurrentLayer(nn.Module):
    def __init__(self, config: VPTPaperIDMConfig) -> None:
        super().__init__()
        init_scale = config.transformer_blocks**-0.5
        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(config, init_scale=init_scale)
                for _ in range(config.transformer_blocks)
            ]
        )


class _ReleasedBody(nn.Module):
    def __init__(self, config: VPTPaperIDMConfig) -> None:
        super().__init__()
        self.config = config
        self.img_process = _ImageProcess(config)
        self.recurrent_layer = _RecurrentLayer(config)
        self.lastlayer = _FanInLayer(
            config.d_model,
            config.d_model,
            kind="linear",
            layer_norm=True,
        )
        self.final_ln = nn.LayerNorm(config.d_model)
        self.conv3d_layer = _FanInLayer(
            config.input_channels,
            config.temporal_channels,
            kind="conv3d",
            kernel_size=(config.temporal_kernel, 1, 1),
            padding=(config.temporal_kernel // 2, 0, 0),
        )

    def encode_frames(self, frames: Tensor) -> Tensor:
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
        value = self.conv3d_layer(value)
        value = value.permute(0, 2, 1, 3, 4).reshape(
            batch * steps, self.config.temporal_channels, height, width
        )
        value = self.img_process(value).reshape(batch, steps, self.config.d_model)
        return value

    def forward(self, frames: Tensor) -> Tensor:
        value = self.encode_frames(frames)
        use_checkpoint = self.config.activation_checkpointing and self.training
        for block in self.recurrent_layer.blocks:
            if use_checkpoint:
                if value.device.type == "xla":
                    from torch_xla.utils.checkpoint import (
                        checkpoint as xla_checkpoint,
                    )

                    value = xla_checkpoint(block, value, use_reentrant=True)
                else:
                    value = checkpoint(block, value, use_reentrant=False)
            else:
                value = block(value)
        value = F.relu(value)
        value = self.lastlayer(value)
        return self.final_ln(value)


class _CelesteActionHead(nn.Module):
    def __init__(self, config: VPTPaperIDMConfig) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(config.d_model, config.action_keys * 2)
        _normalize_output_filters(self.linear_layer.weight, 0.01)
        nn.init.zeros_(self.linear_layer.bias)
        self.action_keys = config.action_keys
        self.temperature = config.action_temperature

    def forward(self, value: Tensor) -> Tensor:
        logits = self.linear_layer(value)
        return logits.reshape(*logits.shape[:-1], self.action_keys, 2) / self.temperature


class VPTPaperIDM(nn.Module):
    """482,133,390-parameter seven-key released-artifact reconstruction."""

    def __init__(self, config: VPTPaperIDMConfig | None = None) -> None:
        super().__init__()
        self.config = config or VPTPaperIDMConfig()
        self.net = _ReleasedBody(self.config)
        self.pi_head = _CelesteActionHead(self.config)

    def forward(self, frames: Tensor) -> Tensor:
        return self.pi_head(self.net(frames))


def natural_factored_nll(
    logits: Tensor, targets: Tensor, supported: Tensor | None = None
) -> Tensor:
    if logits.ndim != 4 or logits.shape[-1] != 2:
        raise ValueError("logits must have shape [B,T,K,2]")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must match logits [B,T,K]")
    if supported is None:
        supported = torch.ones_like(targets, dtype=torch.bool)
    if supported.shape != targets.shape:
        raise ValueError("supported must match targets")
    losses = F.cross_entropy(
        logits.reshape(-1, 2), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    weights = supported.to(dtype=losses.dtype)
    counts = weights.sum(dim=(0, 1))
    if logits.device.type != "xla" and bool(torch.any(counts == 0).item()):
        raise ValueError("every key must have at least one supported label")
    per_key = (losses * weights).sum(dim=(0, 1)) / counts.clamp_min(1)
    return per_key.sum()


def parameter_inventory(model: nn.Module) -> dict[str, Any]:
    components: dict[str, int] = {}
    total = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        root = name.split(".", 1)[0]
        components[root] = components.get(root, 0) + count
    return {"total": total, "components": components}


def named_parameter_shapes(model: nn.Module) -> dict[str, list[int]]:
    return {name: list(parameter.shape) for name, parameter in model.named_parameters()}
