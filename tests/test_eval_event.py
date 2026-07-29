from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from badeline.eval_event import deweight_event_logits, evaluate_event_latch
from badeline.event_model import EventLatchIDM
from data.schema import KEY_ORDER


def _config() -> dict[str, object]:
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
        "event_latch": True,
    }


def _write_shard(path: Path) -> None:
    frame_count = 26
    features = np.random.default_rng(4).normal(
        size=(frame_count, 8)
    ).astype(np.float32)
    keys = np.zeros((frame_count, len(KEY_ORDER)), dtype=np.uint8)
    keys[5:8, 0] = 1
    keys[17:21, 5] = 1
    engine = np.concatenate(
        (np.arange(13, dtype=np.int64), np.arange(40, 53, dtype=np.int64))
    )
    active = np.ones(frame_count, dtype=np.uint8)
    active[6] = 0
    np.savez_compressed(
        path,
        features=features,
        keys=keys,
        engine_frame_idx=engine,
        input_active=active,
        session_id=np.asarray(path.stem),
    )


def test_event_evaluation_writes_rescorable_boundary_safe_sidecar(
    tmp_path: Path,
) -> None:
    session_id = "fixture"
    _write_shard(tmp_path / f"{session_id}.npz")
    sidecar = tmp_path / "predictions.npz"
    torch.manual_seed(0)
    model = EventLatchIDM(_config())

    report = evaluate_event_latch(
        model,
        _config(),
        tmp_path,
        [session_id],
        "cpu",
        preds_out=sidecar,
        segment_span=2,
        allow_oracle_thresholds=False,
    )

    # Each 13-frame contiguous run supplies 13 - 9 + 1 = five targets.
    assert report["all_frames"]["n"] == 10
    assert report["streams"] == 2
    assert report["event_heads"]["valid_transition_frames"] == 8
    assert report["decode"]["resync_patience"] == 3
    assert report["input_active_only"]["decision_metrics"]["frames"] == 9
    assert "per_key_ap" in report["event_heads"]["onset"]
    assert not report["threshold_policy"]["data_fitted_thresholds_enabled"]

    def contains_oracle(value: object) -> bool:
        if isinstance(value, dict):
            return any(
                "oracle" in str(key).lower() or contains_oracle(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(contains_oracle(item) for item in value)
        return False

    assert not contains_oracle(report)

    with np.load(sidecar, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "y_true",
            "y_prob",
            "y_latch",
            "onset_true",
            "onset_prob",
            "onset_raw_prob",
            "release_true",
            "release_prob",
            "release_raw_prob",
            "onset_positive_weight",
            "release_positive_weight",
            "event_valid",
            "input_active",
            "session_lengths",
            "session_ids",
        }
        assert archive["y_true"].shape == (10, len(KEY_ORDER))
        assert archive["session_lengths"].tolist() == [5, 5]
        assert archive["session_ids"].tolist() == [
            "fixture__stream000",
            "fixture__stream001",
        ]
        for name in ("y_prob", "onset_prob", "release_prob"):
            assert np.isfinite(archive[name]).all()
            assert np.all((archive[name] >= 0) & (archive[name] <= 1))
        # A decode starts from the direct state head independently for each
        # engine stream, rather than carrying state through the index gap.
        for start in (0, 5):
            np.testing.assert_array_equal(
                archive["y_latch"][start], archive["y_prob"][start] >= 0.5
            )


def test_event_logit_deweighting_undoes_fixed_positive_weight_shift() -> None:
    natural_logits = torch.tensor(
        [[-0.4, 0.7, -1.2, 0.0, 1.1, -2.0, 0.3]], dtype=torch.float32
    )
    weight = torch.tensor([50.0, 8.0, 3.0, 1.0, 12.0, 2.0, 5.0])
    weighted_logits = natural_logits + weight.log()

    recovered = deweight_event_logits(weighted_logits, weight)

    torch.testing.assert_close(recovered, natural_logits)
    # A raw 0.5 threshold would fire key zero; deweighting correctly retains
    # its negative natural-prevalence logit as a non-event.
    assert weighted_logits[0, 0] > 0
    assert recovered[0, 0] < 0


def test_event_evaluation_is_independent_of_inference_chunk_size(
    tmp_path: Path,
) -> None:
    session_id = "fixture"
    _write_shard(tmp_path / f"{session_id}.npz")
    torch.manual_seed(0)
    model = EventLatchIDM(_config())
    small = tmp_path / "small.npz"
    large = tmp_path / "large.npz"

    evaluate_event_latch(
        model,
        _config(),
        tmp_path,
        [session_id],
        "cpu",
        preds_out=small,
        segment_span=1,
    )
    evaluate_event_latch(
        model,
        _config(),
        tmp_path,
        [session_id],
        "cpu",
        preds_out=large,
        segment_span=32,
    )

    with np.load(small, allow_pickle=False) as first, np.load(
        large, allow_pickle=False
    ) as second:
        assert first.files == second.files
        for name in first.files:
            if np.issubdtype(first[name].dtype, np.floating):
                np.testing.assert_allclose(first[name], second[name], atol=1e-6)
            else:
                np.testing.assert_array_equal(first[name], second[name])
