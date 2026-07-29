from __future__ import annotations

import pytest
import torch

from badeline.model import BadelineIDM
from badeline.temporal import AlignedTemporalTCN
from data.schema import KEY_ORDER


def test_aligned_tcn_reports_exact_valid_support() -> None:
    encoder = AlignedTemporalTCN(12, 16, dilations=[1, 2, 4])

    assert encoder.receptive_radius == 9
    assert encoder.valid_slice(18) == slice(0, 0)
    assert encoder.valid_slice(23) == slice(9, 14)
    assert encoder(torch.rand(2, 23, 12)).shape == (2, 23, 16)


def test_aligned_tcn_segment_matches_individual_dense_windows() -> None:
    torch.manual_seed(7)
    config = {
        "window": 7,
        "frame_stride": 1,
        "window_mode": "centered",
        "input_config": "pixels",
        "precomputed_features": True,
        "backbone_feature_dim": 12,
        "embedding_dim": 8,
        "temporal_dim": 10,
        "temporal_arch": "aligned_tcn",
        "tcn_dilations": [1],
    }
    model = BadelineIDM(config).eval()
    features = torch.rand(11, 12)  # seven-frame span, five aligned targets

    with torch.no_grad():
        encoded = model.encode_segment({"features": features.unsqueeze(0)})
        individual = torch.cat([
            model({"features": features[start : start + 7].unsqueeze(0)})
            for start in range(5)
        ])
        segment = model.forward_segment({"features": features.unsqueeze(0)})[0]

    assert encoded.shape == (1, 5, 10)
    assert segment.shape == (5, len(KEY_ORDER))
    assert torch.allclose(individual, segment, atol=1e-6)


def test_aligned_tcn_sequence_loss_backpropagates() -> None:
    torch.manual_seed(11)
    model = BadelineIDM({
        "window": 7,
        "window_mode": "centered",
        "input_config": "pixels",
        "precomputed_features": True,
        "backbone_feature_dim": 12,
        "embedding_dim": 8,
        "temporal_dim": 10,
        "temporal_arch": "aligned_tcn",
        "tcn_dilations": [1],
    }).train()
    logits = model.forward_segment({"features": torch.rand(2, 11, 12)})
    target = torch.randint(0, 2, logits.shape, dtype=torch.float32)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, target).backward()

    assert logits.shape == (2, 5, len(KEY_ORDER))
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_aligned_tcn_rejects_boundary_unsafe_context() -> None:
    with pytest.raises(ValueError, match="exceeds centered window support"):
        BadelineIDM({
            "window": 5,
            "frame_stride": 1,
            "window_mode": "centered",
            "input_config": "pixels",
            "precomputed_features": True,
            "backbone_feature_dim": 12,
            "embedding_dim": 8,
            "temporal_dim": 10,
            "temporal_arch": "aligned_tcn",
            "tcn_dilations": [1],  # radius 3, but only two frames per side
        })


def test_aligned_tcn_rejects_explicit_feature_deltas() -> None:
    with pytest.raises(ValueError, match="extend the declared left context"):
        BadelineIDM({
            "window": 7,
            "input_config": "pixels",
            "precomputed_features": True,
            "feature_deltas": True,
            "temporal_arch": "aligned_tcn",
            "tcn_dilations": [1],
        })
