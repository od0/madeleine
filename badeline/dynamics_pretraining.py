"""Photon-inspired representation learning primitives for Celeste video.

The public Photon-1 description motivates future-latent prediction and an
ordered frame-pair encoder, but does not specify this continuous-target
training recipe.  This module therefore keeps the MADELEINE adaptation small
and explicit: an online encoder predicts a stop-gradient EMA target latent.

Data selection, temporal sampling, augmentations, and first-frame handling are
deliberately caller responsibilities.  In particular, callers that want a
duplicate first-frame policy for the ordered-pair encoder must pass
``previous=current`` themselves; the encoder never crosses or invents a
segment boundary.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


DynamicsArm = Literal["B", "C", "D"]
EncoderInput = torch.Tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REPRESENTATION_DIM = 512


def _validate_frame(name: str, frame: torch.Tensor) -> None:
    if frame.ndim != 4 or frame.shape[1] != 3:
        raise ValueError(f"{name} must have shape [B,3,H,W]")
    if not frame.is_floating_point():
        raise TypeError(f"{name} must be floating point")


class ResNet18FrameEncoder(nn.Module):
    """ImageNet-initializable ResNet-18 encoder with a 512-D output.

    Inputs are floating-point RGB images in the usual ``[0, 1]`` range.  The
    module performs ImageNet normalization and returns the globally pooled
    layer-4 representation before the classification layer.
    """

    output_dim = REPRESENTATION_DIM

    def __init__(
        self,
        *,
        weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
    ) -> None:
        super().__init__()
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.register_buffer(
            "image_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
        )

    def normalize(self, frame: torch.Tensor) -> torch.Tensor:
        _validate_frame("frame", frame)
        return (frame - self.image_mean) / self.image_std

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        latent = self.backbone(self.normalize(frame))
        if latent.shape[-1] != self.output_dim:
            raise RuntimeError(
                f"ResNet-18 produced {latent.shape[-1]} features, expected "
                f"{self.output_dim}"
            )
        return latent


class OrderedPairResNet18Encoder(nn.Module):
    """Encode ``[previous, current, current-previous]`` before pooling.

    Every ResNet-18 parameter is copied from ``source_encoder``.  The input
    convolution is expanded from three to nine channels: its ImageNet weights
    are copied only into the current-frame slice (channels 3:6), while the
    previous-frame and signed-difference slices are exactly zero.  Consequently
    this encoder is functionally the source current-frame encoder at
    initialization while retaining learnable pre-pooling motion pathways.
    """

    output_dim = REPRESENTATION_DIM

    def __init__(
        self,
        *,
        weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
        source_encoder: ResNet18FrameEncoder | None = None,
    ) -> None:
        super().__init__()
        if source_encoder is not None and weights is not ResNet18_Weights.DEFAULT:
            raise ValueError(
                "pass either source_encoder or non-default weights, not both"
            )
        source = source_encoder or ResNet18FrameEncoder(weights=weights)
        self.backbone = deepcopy(source.backbone)
        original = self.backbone.conv1
        expanded = nn.Conv2d(
            9,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            dilation=original.dilation,
            groups=original.groups,
            bias=original.bias is not None,
            padding_mode=original.padding_mode,
            device=original.weight.device,
            dtype=original.weight.dtype,
        )
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, 3:6].copy_(original.weight)
            if expanded.bias is not None:
                assert original.bias is not None
                expanded.bias.copy_(original.bias)
        self.backbone.conv1 = expanded
        self.register_buffer("image_mean", source.image_mean.detach().clone())
        self.register_buffer("image_std", source.image_std.detach().clone())

    @classmethod
    def from_frame_encoder(
        cls, source_encoder: ResNet18FrameEncoder
    ) -> "OrderedPairResNet18Encoder":
        """Build an exact ordered-pair expansion of ``source_encoder``."""

        return cls(source_encoder=source_encoder)

    def ordered_input(
        self, previous: torch.Tensor, current: torch.Tensor
    ) -> torch.Tensor:
        """Return the normalized ordered nine-channel pair representation."""

        _validate_frame("previous", previous)
        _validate_frame("current", current)
        if previous.shape != current.shape:
            raise ValueError("previous and current must have identical shapes")
        previous_normalized = (previous - self.image_mean) / self.image_std
        current_normalized = (current - self.image_mean) / self.image_std
        difference = current_normalized - previous_normalized
        return torch.cat(
            (previous_normalized, current_normalized, difference), dim=1
        )

    def forward(
        self, previous: torch.Tensor, current: torch.Tensor
    ) -> torch.Tensor:
        latent = self.backbone(self.ordered_input(previous, current))
        if latent.shape[-1] != self.output_dim:
            raise RuntimeError(
                f"ResNet-18 produced {latent.shape[-1]} features, expected "
                f"{self.output_dim}"
            )
        return latent


class HorizonConditionedPredictor(nn.Module):
    """Two-layer 512-D predictor with a learned native-frame horizon token."""

    def __init__(self, horizons: tuple[int, ...] | list[int]) -> None:
        super().__init__()
        values = tuple(int(value) for value in horizons)
        if not values:
            raise ValueError("horizons must not be empty")
        if len(values) != len(set(values)):
            raise ValueError("horizons must be unique")
        if any(value < 0 for value in values):
            raise ValueError("horizons must be non-negative")
        self.register_buffer(
            "horizon_values", torch.tensor(values, dtype=torch.long)
        )
        self.horizon_embedding = nn.Embedding(len(values), REPRESENTATION_DIM)
        self.layers = nn.Sequential(
            nn.Linear(REPRESENTATION_DIM, REPRESENTATION_DIM),
            nn.GELU(),
            nn.Linear(REPRESENTATION_DIM, REPRESENTATION_DIM),
        )

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.horizon_values.tolist())

    def _indices(
        self, horizon: int | torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if isinstance(horizon, int):
            requested = torch.full(
                (batch_size,), horizon, dtype=torch.long, device=device
            )
        else:
            requested = horizon.to(device=device, dtype=torch.long)
            if requested.ndim == 0:
                requested = requested.expand(batch_size)
            if requested.shape != (batch_size,):
                raise ValueError("horizon tensor must be scalar or have shape [B]")
        values = self.horizon_values.to(device=device)
        matches = requested[:, None] == values[None, :]
        if not bool(matches.any(dim=1).all()):
            invalid = requested[~matches.any(dim=1)].unique().tolist()
            raise ValueError(
                f"unsupported horizon(s) {invalid}; configured {self.horizons}"
            )
        return matches.to(dtype=torch.int64).argmax(dim=1)

    def forward(
        self, latent: torch.Tensor, horizon: int | torch.Tensor
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != REPRESENTATION_DIM:
            raise ValueError(
                f"latent must have shape [B,{REPRESENTATION_DIM}]"
            )
        indices = self._indices(horizon, latent.shape[0], latent.device)
        conditioned = latent + self.horizon_embedding(indices)
        return self.layers(conditioned)


def normalized_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean per-example L1 distance between L2-normalized representations."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim != 2 or prediction.shape[1] != REPRESENTATION_DIM:
        raise ValueError(
            f"prediction and target must have shape [B,{REPRESENTATION_DIM}]"
        )
    prediction_normalized = F.normalize(prediction, dim=-1, eps=eps)
    target_normalized = F.normalize(target.detach(), dim=-1, eps=eps)
    return (prediction_normalized - target_normalized).abs().mean(dim=-1).mean()


@dataclass(frozen=True)
class DynamicsPretrainingOutput:
    """One online/target comparison and its scalar training loss."""

    online: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    loss: torch.Tensor


@dataclass(frozen=True)
class CollapseDiagnostics:
    """Scale-independent diagnostics for a batch of representations."""

    per_dimension_std: torch.Tensor
    covariance_effective_rank: torch.Tensor
    mean_cosine_similarity: torch.Tensor


def collapse_diagnostics(
    representations: torch.Tensor, *, eps: float = 1e-12
) -> CollapseDiagnostics:
    """Measure variance, covariance participation rank, and mean cosine.

    ``covariance_effective_rank`` is the stable participation-ratio proxy
    ``trace(C)^2 / ||C||_F^2``.  It equals the number of equally energetic
    covariance directions and avoids an eigendecomposition.  Mean cosine is
    computed across distinct, unordered-equivalent sample pairs (the ordered
    average has the same value).
    """

    if representations.ndim != 2:
        raise ValueError("representations must have shape [N,D]")
    if representations.shape[0] < 2:
        raise ValueError("collapse diagnostics require at least two samples")
    if not representations.is_floating_point():
        raise TypeError("representations must be floating point")

    sample_count, dimension = representations.shape
    centered = representations - representations.mean(dim=0, keepdim=True)
    per_dimension_std = centered.square().mean(dim=0).sqrt()
    trace = centered.square().sum() / sample_count
    if sample_count <= dimension:
        gram = centered @ centered.transpose(0, 1) / sample_count
        covariance_frobenius_squared = gram.square().sum()
    else:
        covariance = centered.transpose(0, 1) @ centered / sample_count
        covariance_frobenius_squared = covariance.square().sum()
    effective_rank = torch.where(
        covariance_frobenius_squared > eps,
        trace.square() / covariance_frobenius_squared.clamp_min(eps),
        trace.new_zeros(()),
    )

    normalized = F.normalize(representations, dim=-1, eps=eps)
    total_similarity = normalized.sum(dim=0).square().sum()
    diagonal_similarity = normalized.square().sum()
    mean_cosine = (total_similarity - diagonal_similarity) / (
        sample_count * (sample_count - 1)
    )
    return CollapseDiagnostics(
        per_dimension_std=per_dimension_std,
        covariance_effective_rank=effective_rank,
        mean_cosine_similarity=mean_cosine,
    )


class EMADynamicsPretrainer(nn.Module):
    """Online predictor plus a stop-gradient EMA target for Arms B, C, and D.

    Arm B compares two views of the same frame while retaining C/D's positive
    horizon tokens as matched nuisance variables.  This lets the data sampler
    use the exact same ``(session, t, h, motion-stratum)`` population for B and
    C; only the target frame differs.  Arm C predicts future single-frame
    latents.  Arm D predicts future ordered-pair latents and requires explicit
    previous frames for both sides.  Augmentations and temporal indexing stay
    outside this module.
    """

    def __init__(
        self,
        arm: DynamicsArm,
        *,
        horizons: tuple[int, ...] | list[int],
        weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
        online_encoder: nn.Module | None = None,
        ema_momentum: float = 0.998,
    ) -> None:
        super().__init__()
        if arm not in ("B", "C", "D"):
            raise ValueError("arm must be one of B, C, or D")
        values = tuple(int(value) for value in horizons)
        if not values or any(value <= 0 for value in values):
            raise ValueError(
                "Arms B, C, and D require the same positive matched horizon "
                "schedule"
            )
        self._validate_momentum(ema_momentum)
        self.arm: DynamicsArm = arm
        self.ema_momentum = float(ema_momentum)

        if online_encoder is None:
            if arm == "D":
                online_encoder = OrderedPairResNet18Encoder(weights=weights)
            else:
                online_encoder = ResNet18FrameEncoder(weights=weights)
        if getattr(online_encoder, "output_dim", None) != REPRESENTATION_DIM:
            raise ValueError(
                f"online_encoder.output_dim must equal {REPRESENTATION_DIM}"
            )
        self.online_encoder = online_encoder
        self.target_encoder = deepcopy(online_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = HorizonConditionedPredictor(values)

    @staticmethod
    def _validate_momentum(momentum: float) -> None:
        if not 0.0 <= float(momentum) <= 1.0:
            raise ValueError("EMA momentum must be in [0, 1]")

    def train(self, mode: bool = True) -> "EMADynamicsPretrainer":
        super().train(mode)
        # The target is a deterministic moving average, never a source of
        # BatchNorm-stat updates or stochastic training-time behavior.
        self.target_encoder.eval()
        return self

    def _encode(
        self,
        encoder: nn.Module,
        current: EncoderInput,
        previous: EncoderInput | None,
    ) -> torch.Tensor:
        if self.arm == "D":
            if previous is None:
                raise ValueError(
                    "Arm D requires an explicit previous frame; pass a "
                    "duplicate at a segment's first frame if that is the "
                    "caller-selected policy"
                )
            return encoder(previous, current)
        if previous is not None:
            raise ValueError(f"Arm {self.arm} does not accept previous frames")
        return encoder(current)

    def forward(
        self,
        *,
        online_current: EncoderInput,
        target_current: EncoderInput,
        horizon: int | torch.Tensor,
        online_previous: EncoderInput | None = None,
        target_previous: EncoderInput | None = None,
    ) -> DynamicsPretrainingOutput:
        online = self._encode(
            self.online_encoder, online_current, online_previous
        )
        with torch.no_grad():
            target = self._encode(
                self.target_encoder, target_current, target_previous
            ).detach()
        prediction = self.predictor(online, horizon)
        return DynamicsPretrainingOutput(
            online=online,
            prediction=prediction,
            target=target,
            loss=normalized_l1_loss(prediction, target),
        )

    @torch.no_grad()
    def update_target(self, momentum: float | None = None) -> None:
        """Apply a deterministic, name-aligned EMA update to target state."""

        resolved = self.ema_momentum if momentum is None else float(momentum)
        self._validate_momentum(resolved)
        online_parameters = dict(self.online_encoder.named_parameters())
        target_parameters = dict(self.target_encoder.named_parameters())
        if online_parameters.keys() != target_parameters.keys():
            raise RuntimeError("online and target parameter names differ")
        for name in sorted(online_parameters):
            target_parameters[name].mul_(resolved).add_(
                online_parameters[name], alpha=1.0 - resolved
            )

        online_buffers = dict(self.online_encoder.named_buffers())
        target_buffers = dict(self.target_encoder.named_buffers())
        if online_buffers.keys() != target_buffers.keys():
            raise RuntimeError("online and target buffer names differ")
        for name in sorted(online_buffers):
            online_buffer = online_buffers[name]
            target_buffer = target_buffers[name]
            if target_buffer.is_floating_point() or target_buffer.is_complex():
                target_buffer.mul_(resolved).add_(
                    online_buffer, alpha=1.0 - resolved
                )
            else:
                target_buffer.copy_(online_buffer)
        self.target_encoder.eval()
