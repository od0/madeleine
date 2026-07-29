from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.oracle_window_differential_followup import (
    METADATA_NAMES,
    DifferentialCandidateModel,
)
from experiments.score_oracle_window_differential_followup import (
    _validate_state_dict_exact,
    extended_uniform_chance,
    load_differential_sidecar,
    publish_followup,
)


def _probability(truth: np.ndarray) -> np.ndarray:
    result = np.full((len(truth), 16), 0.01 / 15, dtype=np.float32)
    result[np.arange(len(truth)), truth] = 0.99
    result /= result.sum(axis=1, keepdims=True)
    return result.astype(np.float32)


def _sidecar_arrays(count: int = 16) -> dict[str, np.ndarray]:
    heads = (np.arange(count) % 14).astype(np.int16)
    truth = (np.arange(count) % 16).astype(np.int8)
    probability = _probability(truth)
    return {
        "session_id": np.asarray(["s"] * count),
        "run_index": np.zeros(count, dtype=np.int32),
        "array_index": np.arange(count, dtype=np.int64),
        "engine_frame_idx": np.arange(count, dtype=np.int64),
        "head_index": heads,
        "key_index": (heads % 7).astype(np.int8),
        "event_type_index": (heads // 7).astype(np.int8),
        "true_offset": truth,
        "crop_start": np.arange(count, dtype=np.int64),
        "block_id": np.asarray([f"b{i}" for i in range(count)]),
        "ordered_pair_prob": probability,
        "symmetric_pair_prob": probability.copy(),
        "feature_conditional_prob": probability.copy(),
    }


def test_extended_uniform_chance_reports_every_requested_metric() -> None:
    truth = np.tile(np.arange(16), 10)
    chance = extended_uniform_chance(truth)
    assert chance["exact"] == pytest.approx(0.0625)
    assert chance["within_1"] == pytest.approx(46 / 256)
    assert chance["within_2"] == pytest.approx(74 / 256)
    assert chance["nll"] == pytest.approx(np.log(16))
    assert chance["entropy"] == pytest.approx(np.log(16))
    assert chance["mean_signed_error"] == pytest.approx(0.0)
    assert chance["early_rate"] == pytest.approx(0.46875)
    assert chance["late_rate"] == pytest.approx(0.46875)


def test_differential_sidecar_validates_exact_inventory_and_dtypes(tmp_path: Path) -> None:
    arrays = _sidecar_arrays()
    path = tmp_path / "predictions.npz"
    np.savez(path, **arrays)
    loaded = load_differential_sidecar(path)
    assert set(loaded) == {
        *METADATA_NAMES,
        "ordered_pair_prob",
        "symmetric_pair_prob",
        "feature_conditional_prob",
    }
    arrays["true_offset"] = arrays["true_offset"].astype(np.int64)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="dtype changed"):
        load_differential_sidecar(path)


def test_checkpoint_schema_validation_rejects_silent_dtype_cast() -> None:
    model = DifferentialCandidateModel()
    expected = model.state_dict()
    observed = {name: value.clone() for name, value in expected.items()}
    first = next(iter(observed))
    observed[first] = observed[first].double()
    with pytest.raises(ValueError, match="tensor schema"):
        _validate_state_dict_exact(observed, expected)


def test_followup_publication_binds_run_and_checkpoint_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    predictions = tmp_path / "predictions.npz"
    run_receipt = tmp_path / "run.json"
    ordered_checkpoint = tmp_path / "ordered.pt"
    symmetric_pair_checkpoint = tmp_path / "same.pt"
    cache_receipt = tmp_path / "cache.json"
    for path, payload in [
        (config, b"{}\n"),
        (predictions, b"predictions"),
        (run_receipt, b"{}\n"),
        (ordered_checkpoint, b"ordered-checkpoint"),
        (symmetric_pair_checkpoint, b"symmetric-pair-checkpoint"),
        (cache_receipt, b"{}\n"),
    ]:
        path.write_bytes(payload)
    from experiments.oracle_window_localization import sha256_file

    report = {
        "study_id": "fixture",
        "config_sha256": sha256_file(config),
        "prediction_sidecar_sha256": sha256_file(predictions),
        "decision_gate": {"decision": "bounded_pixel_followup_negative_no_phase_2"},
    }
    out = tmp_path / "report.json"
    marker = tmp_path / "complete.json"
    publish_followup(
        report=report,
        out=out,
        marker=marker,
        config_path=config,
        predictions_path=predictions,
        run_receipt_path=run_receipt,
        ordered_checkpoint_path=ordered_checkpoint,
        symmetric_pair_checkpoint_path=symmetric_pair_checkpoint,
        cache_receipt_path=cache_receipt,
    )
    saved = json.loads(marker.read_text())
    assert saved["run_receipt"]["sha256"] == sha256_file(run_receipt)
    assert saved["checkpoints"]["ordered_pair"]["sha256"] == sha256_file(
        ordered_checkpoint
    )
    assert saved["checkpoints"]["symmetric_pair"]["sha256"] == sha256_file(
        symmetric_pair_checkpoint
    )
    assert saved["content_sha256"]
    with pytest.raises(ValueError, match="overwrite"):
        publish_followup(
            report=report,
            out=out,
            marker=marker,
            config_path=config,
            predictions_path=predictions,
            run_receipt_path=run_receipt,
            ordered_checkpoint_path=ordered_checkpoint,
            symmetric_pair_checkpoint_path=symmetric_pair_checkpoint,
            cache_receipt_path=cache_receipt,
        )
