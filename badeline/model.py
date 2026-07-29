"""The Badeline inverse-dynamics model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from badeline.temporal import AlignedTemporalTCN, DEFAULT_TCN_DILATIONS
from data.schema import KEY_ORDER


WINDOW_MODES = ("centered", "past_only")
INPUT_CONFIGS = ("pixels", "history", "pixels_plus_history", "state_meta")
TEMPORAL_ARCHITECTURES = ("gru", "aligned_tcn")


def _value(config: Mapping[str, Any] | object, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _swap_batchnorm_for_groupnorm(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            groups = 32 if child.num_features % 32 == 0 else 8
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            _swap_batchnorm_for_groupnorm(child)


class ResNetFrameEncoder(nn.Module):
    """Encode one frame while retaining a small spatial feature grid."""

    def __init__(self, embedding_dim: int, spatial_size: int = 4) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        # BatchNorm training is numerically unreliable on the MPS backend
        # (torch 2.13) and our batches are small; GroupNorm sidesteps both.
        _swap_batchnorm_for_groupnorm(backbone)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        # AdaptiveAvgPool2d's CUDA backward is nondeterministic and raises
        # under torch.use_deterministic_algorithms(True). At the frozen 128px
        # input, layer4's output is already exactly (spatial_size,
        # spatial_size), so the pool was a pass-through; we assert that
        # instead of pooling. Other input sizes are a contract violation.
        self.spatial_size = spatial_size
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * spatial_size * spatial_size, embedding_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.features(frames)
        if features.shape[-2:] != (self.spatial_size, self.spatial_size):
            raise ValueError(
                f"encoder expects a {self.spatial_size}x{self.spatial_size} "
                f"feature map (128px input); got {tuple(features.shape[-2:])}"
            )
        return self.projection(features)


class FrozenImageNetFrameEncoder(nn.Module):
    """ImageNet ResNet-18 with a small trainable projection.

    The backbone can be frozen or fine-tuned, but remains in evaluation mode
    so BatchNorm statistics cannot drift toward temporally correlated gameplay
    batches. ImageNet normalization lives here rather than in the data builder;
    this keeps stored pixels and the masking audit unchanged.
    """

    def __init__(self, embedding_dim: int, *, trainable: bool = False) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.features.requires_grad_(trainable)
        self.features.eval()
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embedding_dim),
        )
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def train(self, mode: bool = True) -> "FrozenImageNetFrameEncoder":
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        normalized = (frames - self.image_mean) / self.image_std
        features = self.features(normalized).mean(dim=(-2, -1))
        return self.projection(features)


class BadelineIDM(nn.Module):
    """Predict the seven canonical key logits for a configured input window."""

    def __init__(self, config: Mapping[str, Any] | object) -> None:
        super().__init__()

        self.window = int(_value(config, "window", 2))
        self.frame_stride = int(_value(config, "frame_stride", 1))
        self.window_mode = str(_value(config, "window_mode", "centered"))
        self.input_config = str(_value(config, "input_config", "pixels"))
        self.history_len = int(_value(config, "history_len", 8))
        self.embedding_dim = int(_value(config, "embedding_dim", 64))
        self.temporal_dim = int(_value(config, "temporal_dim", 64))
        self.temporal_arch = str(_value(config, "temporal_arch", "gru"))
        spatial_size = int(_value(config, "spatial_size", 4))
        frozen_dino = bool(_value(config, "frozen_dino", False))
        self.pretrained_encoder = bool(
            _value(config, "pretrained_encoder", False)
        )
        self.trainable_encoder = bool(
            _value(config, "trainable_encoder", False)
        )
        self.precomputed_features = bool(
            _value(config, "precomputed_features", False)
        )
        self.backbone_feature_dim = int(
            _value(config, "backbone_feature_dim", 512)
        )
        self.feature_deltas = bool(_value(config, "feature_deltas", False))
        self.video_augmentation = bool(
            _value(config, "video_augmentation", False)
        )
        self.temporal_input_dim = self.embedding_dim * (
            2 if self.feature_deltas else 1
        )

        if frozen_dino:
            raise NotImplementedError("frozen-DINO lands later")
        if self.trainable_encoder and not self.pretrained_encoder:
            raise ValueError("trainable_encoder requires pretrained_encoder")
        if self.video_augmentation and self.precomputed_features:
            raise ValueError(
                "video_augmentation requires pixels, not precomputed features"
            )
        if self.temporal_arch == "aligned_tcn" and self.feature_deltas:
            raise ValueError(
                "aligned_tcn learns native-rate differences in its width-five "
                "stem; explicit feature_deltas would extend the declared "
                "left context by one frame"
            )
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if self.history_len < 1:
            raise ValueError("history_len must be at least 1")
        if self.window_mode not in WINDOW_MODES:
            raise ValueError(
                f"window_mode must be one of {WINDOW_MODES}, got {self.window_mode!r}"
            )
        if self.temporal_arch not in TEMPORAL_ARCHITECTURES:
            raise ValueError(
                "temporal_arch must be one of "
                f"{TEMPORAL_ARCHITECTURES}, got {self.temporal_arch!r}"
            )
        if self.input_config not in INPUT_CONFIGS:
            raise ValueError(
                f"input_config must be one of {INPUT_CONFIGS}, "
                f"got {self.input_config!r}"
            )
        if self.input_config == "state_meta":
            raise NotImplementedError("state_meta lands with real data")
        if spatial_size < 1:
            raise ValueError("spatial_size must be at least 1")

        self.key_order = list(KEY_ORDER)
        uses_pixels = self.input_config in ("pixels", "pixels_plus_history")
        uses_history = self.input_config in ("history", "pixels_plus_history")
        if self.temporal_arch == "aligned_tcn" and not uses_pixels:
            raise ValueError("aligned_tcn requires visual inputs")

        self.frame_span = (self.window - 1) * self.frame_stride + 1
        self.target_in_window = (
            (self.window - 1) // 2
            if self.window_mode == "centered"
            else self.window - 1
        )
        self.raw_target_offset = self.target_in_window * self.frame_stride

        if uses_pixels:
            self.frame_encoder: (
                ResNetFrameEncoder | FrozenImageNetFrameEncoder | None
            )
            self.feature_projection: nn.Module | None
            if self.precomputed_features:
                self.frame_encoder = None
                self.feature_projection = nn.Sequential(
                    nn.Linear(self.backbone_feature_dim, self.embedding_dim),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(self.embedding_dim),
                )
            elif self.pretrained_encoder:
                self.frame_encoder = FrozenImageNetFrameEncoder(
                    self.embedding_dim, trainable=self.trainable_encoder
                )
                self.feature_projection = None
            else:
                self.frame_encoder = ResNetFrameEncoder(
                    self.embedding_dim, spatial_size
                )
                self.feature_projection = None
            self.temporal: nn.GRUCell | AlignedTemporalTCN | None
            if self.temporal_arch == "aligned_tcn":
                dilations = _value(
                    config, "tcn_dilations", list(DEFAULT_TCN_DILATIONS)
                )
                if not isinstance(dilations, (list, tuple)):
                    raise ValueError("tcn_dilations must be a list of integers")
                self.temporal = AlignedTemporalTCN(
                    self.temporal_input_dim,
                    self.temporal_dim,
                    dilations=dilations,
                    dropout=float(_value(config, "tcn_dropout", 0.0)),
                )
                past = self.raw_target_offset
                future = self.frame_span - 1 - self.raw_target_offset
                if self.temporal.receptive_radius > min(past, future):
                    raise ValueError(
                        "aligned_tcn receptive radius "
                        f"{self.temporal.receptive_radius} exceeds centered "
                        f"window support ({past} past, {future} future)"
                    )
            else:
                # GRUCell unrolled manually: the fused nn.GRU kernel trains
                # incorrectly on the MPS backend (torch 2.13 — val BCE plateaus
                # ~0.42-0.46 where CPU reaches 0.34 on the same fixture); the
                # cell's elementwise ops do not.
                self.temporal = nn.GRUCell(
                    input_size=self.temporal_input_dim,
                    hidden_size=self.temporal_dim,
                )
        else:
            self.frame_encoder = None
            self.feature_projection = None
            self.temporal = None

        if uses_history:
            self.history_encoder: nn.Module | None = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.history_len * len(KEY_ORDER), self.temporal_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.temporal_dim),
            )
        else:
            self.history_encoder = None

        head_input_dim = self.temporal_dim * (
            int(uses_pixels) + int(uses_history)
        )
        # ModuleList order is deliberately the frozen data.schema.KEY_ORDER.
        self.heads = nn.ModuleList(
            nn.Linear(head_input_dim, 1) for _ in self.key_order
        )

    def _temporal_features(self, per_frame: torch.Tensor) -> torch.Tensor:
        if not self.feature_deltas:
            return per_frame
        delta = torch.zeros_like(per_frame)
        delta[:, 1:] = per_frame[:, 1:] - per_frame[:, :-1]
        return torch.cat((per_frame, delta), dim=-1)

    def _augment_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply VPT-range transforms once per segment, consistently in time."""

        if not self.training or not self.video_augmentation:
            return frames
        augmented: list[torch.Tensor] = []
        for segment in frames:
            def uniform(low: float, high: float) -> float:
                return float(
                    torch.empty((), device=segment.device).uniform_(low, high)
                )

            transformed = transform_functional.adjust_brightness(
                segment, uniform(0.8, 1.2)
            )
            transformed = transform_functional.adjust_contrast(
                transformed, uniform(0.8, 1.2)
            )
            transformed = transform_functional.adjust_saturation(
                transformed, uniform(0.8, 1.2)
            )
            transformed = transform_functional.adjust_hue(
                transformed, uniform(-0.2, 0.2)
            )
            transformed = transform_functional.affine(
                transformed,
                angle=uniform(-2.0, 2.0),
                translate=[
                    round(uniform(-2.0, 2.0)),
                    round(uniform(-2.0, 2.0)),
                ],
                scale=uniform(0.98, 1.02),
                shear=[uniform(-2.0, 2.0), 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            augmented.append(transformed)
        return torch.stack(augmented)

    def _pixel_embedding(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [B,T,C,H,W]")
        batch_size, frame_count, channels, height, width = frames.shape
        expected = self.frame_span if self.temporal_arch == "aligned_tcn" else self.window
        if frame_count != expected:
            raise ValueError(
                f"frames window is {frame_count}, expected {expected} for "
                f"{self.temporal_arch}"
            )
        if channels != 3:
            raise ValueError(f"frames must have 3 channels, got {channels}")
        if not frames.is_floating_point():
            raise TypeError("frames must be floating point and normalized to [0, 1]")

        frames = self._augment_frames(frames)
        assert self.frame_encoder is not None
        assert self.temporal is not None
        per_frame = self.frame_encoder(
            frames.reshape(batch_size * frame_count, channels, height, width)
        )
        per_frame = per_frame.reshape(batch_size, frame_count, self.embedding_dim)
        per_frame = self._temporal_features(per_frame)
        if isinstance(self.temporal, AlignedTemporalTCN):
            return self.temporal(per_frame)[:, self.raw_target_offset]
        hidden = per_frame.new_zeros(batch_size, self.temporal_dim)
        for step in range(frame_count):
            hidden = self.temporal(per_frame[:, step], hidden)
        return hidden

    def _precomputed_embedding(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [B,T,D]")
        batch_size, frame_count, feature_dim = features.shape
        expected = self.frame_span if self.temporal_arch == "aligned_tcn" else self.window
        if frame_count != expected:
            raise ValueError(
                f"features window is {frame_count}, expected {expected} for "
                f"{self.temporal_arch}"
            )
        if feature_dim != self.backbone_feature_dim:
            raise ValueError(
                f"features dim is {feature_dim}, configured backbone feature "
                f"dim is {self.backbone_feature_dim}"
            )
        if not features.is_floating_point():
            raise TypeError("features must be floating point")
        assert self.feature_projection is not None
        assert self.temporal is not None
        per_frame = self.feature_projection(
            features.reshape(batch_size * frame_count, feature_dim)
        ).reshape(batch_size, frame_count, self.embedding_dim)
        per_frame = self._temporal_features(per_frame)
        if isinstance(self.temporal, AlignedTemporalTCN):
            return self.temporal(per_frame)[:, self.raw_target_offset]
        hidden = per_frame.new_zeros(batch_size, self.temporal_dim)
        for step in range(frame_count):
            hidden = self.temporal(per_frame[:, step], hidden)
        return hidden

    def _history_embedding(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("history must have shape [B,history_len,7]")
        if history.shape[1:] != (self.history_len, len(KEY_ORDER)):
            raise ValueError(
                "history shape after the batch dimension must be "
                f"[{self.history_len},{len(KEY_ORDER)}]"
            )
        if not history.is_floating_point():
            raise TypeError("history must be floating point")

        assert self.history_encoder is not None
        return self.history_encoder(history)

    def encode_window(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the target-aligned hidden representation ``[B, H]``."""

        embeddings: list[torch.Tensor] = []
        if self.input_config in ("pixels", "pixels_plus_history"):
            if self.precomputed_features:
                if "features" not in batch:
                    raise KeyError("features")
                embeddings.append(self._precomputed_embedding(batch["features"]))
            else:
                if "frames" not in batch:
                    raise KeyError("frames")
                embeddings.append(self._pixel_embedding(batch["frames"]))
        if self.input_config in ("history", "pixels_plus_history"):
            if "history" not in batch:
                raise KeyError("history")
            embeddings.append(self._history_embedding(batch["history"]))

        fused = torch.cat(embeddings, dim=-1) if len(embeddings) > 1 else embeddings[0]
        return fused

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return unsquashed logits with shape ``[B, 7]``."""

        encoded = self.encode_window(batch)
        return torch.cat([head(encoded) for head in self.heads], dim=1)

    def encode_segment(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode target-aligned representations with shape ``[B, S, H]``.

        ``frames`` contains the segment's unique contiguous raw frames. Each
        frame is encoded exactly once, and the S overlapping, possibly
        dilated windows are assembled by index. Encoding a frame
        per window would inflate every epoch estimate by roughly the window
        length; this path exists so that can never happen. The per-window
        math (GRU unroll over the same per-frame embeddings, same heads) is
        identical to ``forward``, which the equivalence test asserts.
        """

        embeddings: list[torch.Tensor] = []
        n_windows: int | None = None

        if self.input_config in ("pixels", "pixels_plus_history"):
            if self.precomputed_features:
                features = batch["features"]
                if features.ndim != 3:
                    raise ValueError("features must have shape [B,F,D]")
                batch_size, frame_count, feature_dim = features.shape
                if feature_dim != self.backbone_feature_dim:
                    raise ValueError(
                        f"features dim is {feature_dim}, expected "
                        f"{self.backbone_feature_dim}"
                    )
            else:
                frames = self._augment_frames(batch["frames"])
                if frames.ndim != 5:
                    raise ValueError("frames must have shape [B,F,C,H,W]")
                batch_size, frame_count = frames.shape[0], frames.shape[1]
            frame_span = self.frame_span
            n_windows = frame_count - frame_span + 1
            if n_windows < 1:
                raise ValueError(
                    f"segment of {frame_count} frames is shorter than "
                    f"raw span {frame_span} for window {self.window} and "
                    f"stride {self.frame_stride}"
                )
            assert self.temporal is not None
            if self.precomputed_features:
                assert self.feature_projection is not None
                per_frame = self.feature_projection(
                    features.reshape(batch_size * frame_count, feature_dim)
                ).reshape(batch_size, frame_count, self.embedding_dim)
            else:
                assert self.frame_encoder is not None
                per_frame = self.frame_encoder(
                    frames.reshape(batch_size * frame_count, *frames.shape[2:])
                ).reshape(batch_size, frame_count, self.embedding_dim)
            if isinstance(self.temporal, AlignedTemporalTCN):
                # Dense native-rate sequence tagging. Only positions with the
                # entire configured receptive field inside this contiguous
                # block are returned, so same-padding never reaches a loss or
                # evaluation target.
                dense = self._temporal_features(per_frame)
                encoded = self.temporal(dense)
                start = self.raw_target_offset
                stop = start + n_windows
                radius = self.temporal.receptive_radius
                if start < radius or frame_count - stop < radius:
                    raise ValueError(
                        "segment does not provide boundary-safe TCN context: "
                        f"frames={frame_count}, targets=[{start},{stop}), "
                        f"radius={radius}"
                    )
                embeddings.append(encoded[:, start:stop])
            else:
                index = (
                    torch.arange(self.window, device=per_frame.device)
                    * self.frame_stride
                    + torch.arange(n_windows, device=per_frame.device).unsqueeze(1)
                )                                          # [S, window]
                windows = per_frame[:, index]              # [B, S, window, E]
                flat = windows.reshape(
                    batch_size * n_windows, self.window, self.embedding_dim
                )
                # Deltas are window-relative: the first sampled frame has a zero
                # delta, exactly as in ``forward``. Computing them before window
                # assembly leaks the frame preceding each window and is also wrong
                # for a dilated stride.
                flat = self._temporal_features(flat)
                hidden = flat.new_zeros(batch_size * n_windows, self.temporal_dim)
                for step in range(self.window):
                    hidden = self.temporal(flat[:, step], hidden)
                embeddings.append(
                    hidden.reshape(batch_size, n_windows, self.temporal_dim)
                )

        if self.input_config in ("history", "pixels_plus_history"):
            history = batch["history"]
            if history.ndim != 4:
                raise ValueError("history must have shape [B,S,history_len,7]")
            if n_windows is None:
                n_windows = history.shape[1]
            if history.shape[1] != n_windows:
                raise ValueError(
                    f"history has {history.shape[1]} windows, frames imply "
                    f"{n_windows}"
                )
            batch_size = history.shape[0]
            assert self.history_encoder is not None
            encoded = self.history_encoder(
                history.reshape(batch_size * n_windows, *history.shape[2:])
            )
            embeddings.append(
                encoded.reshape(batch_size, n_windows, self.temporal_dim)
            )

        fused = (
            torch.cat(embeddings, dim=-1) if len(embeddings) > 1
            else embeddings[0]
        )
        return fused

    def forward_segment(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Segment path (brief v3.2): unsquashed logits ``[B, S, 7]``."""

        encoded = self.encode_segment(batch)
        return torch.cat([head(encoded) for head in self.heads], dim=-1)
