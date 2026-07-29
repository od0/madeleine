from __future__ import annotations

import torch

from badeline.event_model import EventLatchIDM


def _config() -> dict:
    return {
        "window": 9,
        "frame_stride": 1,
        "window_mode": "centered",
        "input_config": "pixels",
        "precomputed_features": True,
        "backbone_feature_dim": 8,
        "embedding_dim": 12,
        "temporal_dim": 16,
        "temporal_arch": "aligned_tcn",
        "tcn_dilations": [1],
    }


def test_event_model_returns_three_aligned_outputs() -> None:
    model = EventLatchIDM(_config())
    batch = {"features": torch.randn(2, 11, 8)}
    outputs = model.forward_segment(batch)
    assert set(outputs) == {"state_logits", "onset_logits", "release_logits"}
    assert all(value.shape == (2, 3, 7) for value in outputs.values())

    loss = sum(value.square().mean() for value in outputs.values())
    loss.backward()
    assert model.encoder.heads[0].weight.grad is not None
    assert model.onset_heads[0].weight.grad is not None
    assert model.release_heads[0].weight.grad is not None
    assert model.encoder.temporal.local.weight.grad is not None


def test_shared_state_initialization_matches_state_only_tcn() -> None:
    from badeline.model import BadelineIDM

    torch.manual_seed(7)
    state_only = BadelineIDM(_config())
    torch.manual_seed(7)
    event_model = EventLatchIDM(_config())
    for baseline, event in zip(
        state_only.state_dict().items(),
        event_model.encoder.state_dict().items(),
        strict=True,
    ):
        baseline_name, baseline_value = baseline
        event_name, event_value = event
        assert baseline_name == event_name
        assert torch.equal(baseline_value, event_value)


def test_event_model_requires_aligned_tcn() -> None:
    config = _config()
    config["temporal_arch"] = "gru"
    try:
        EventLatchIDM(config)
    except ValueError as exc:
        assert "aligned_tcn" in str(exc)
    else:
        raise AssertionError("EventLatchIDM accepted a non-aligned encoder")
