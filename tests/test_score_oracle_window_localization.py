from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.oracle_window_localization import HEAD_NAMES
from experiments.score_oracle_window_localization import (
    analytic_uniform_chance,
    apply_seed_zero_gate,
    arm_metrics,
    estimable_heads,
    load_prediction_sidecar,
    paired_block_bootstrap,
    publish_score,
    summarize_probabilities,
)


def _probabilities(truth: np.ndarray, *, correct: bool) -> np.ndarray:
    result = np.full((len(truth), 16), 0.01 / 15, dtype=np.float32)
    prediction = truth if correct else (truth + 1) % 16
    result[np.arange(len(truth)), prediction] = 0.99
    result /= result.sum(axis=1, keepdims=True)
    return result.astype(np.float32)


def test_uniform_chance_matches_width_16_formulas() -> None:
    truth = np.tile(np.arange(16), 10)
    chance = analytic_uniform_chance(truth, width=16)
    assert chance["exact"] == pytest.approx(0.0625)
    assert chance["within_1"] == pytest.approx(46 / 256)
    assert chance["within_2"] == pytest.approx(74 / 256)
    assert chance["nll"] == pytest.approx(np.log(16))
    assert chance["entropy"] == pytest.approx(np.log(16))


def test_metrics_keep_early_and_late_error_signs() -> None:
    truth = np.asarray([3, 4, 5, 6])
    probability = np.eye(16, dtype=np.float32)[[2, 4, 6, 8]]
    metrics = summarize_probabilities(probability, truth, width=16)
    assert metrics["exact"] == pytest.approx(0.25)
    assert metrics["within_1"] == pytest.approx(0.75)
    assert metrics["mean_signed_error"] == pytest.approx(0.5)
    assert metrics["early_rate"] == pytest.approx(0.25)
    assert metrics["late_rate"] == pytest.approx(0.5)


def test_sidecar_requires_current_reference_nan_only_off_support(tmp_path: Path) -> None:
    count = 5
    truth = np.arange(count, dtype=np.int8)
    conditional = _probabilities(truth, correct=True)
    dense = _probabilities(truth, correct=False)
    support = np.asarray([1, 0, 1, 0, 1], dtype=bool)
    current = conditional.copy()
    current[~support] = np.nan
    path = tmp_path / "predictions.npz"
    np.savez(
        path,
        session_id=np.asarray(["s"] * count),
        run_index=np.zeros(count, np.int32),
        array_index=np.arange(count, dtype=np.int64),
        engine_frame_idx=np.arange(count, dtype=np.int64),
        head_index=np.zeros(count, np.int16),
        key_index=np.zeros(count, np.int8),
        event_type_index=np.zeros(count, np.int8),
        true_offset=truth,
        crop_start=np.arange(count, dtype=np.int64),
        block_id=np.asarray([f"b{i}" for i in range(count)]),
        conditional_prob=conditional,
        dense_prob=dense,
        current_dense_prob=current,
        current_dense_support=support,
    )
    loaded = load_prediction_sidecar(path, width=16)
    assert np.array_equal(loaded["current_dense_support"], support)

    current[1, 0] = 0.5
    np.savez(path, **{**loaded, "current_dense_prob": current})
    with pytest.raises(ValueError, match="all-NaN"):
        load_prediction_sidecar(path, width=16)


def test_paired_block_bootstrap_preserves_positive_exact_delta() -> None:
    heads = np.repeat(np.arange(4), 30)
    blocks = np.asarray([f"b{index // 4}" for index in range(len(heads))])
    conditional = np.ones(len(heads))
    dense = np.zeros(len(heads))
    result = paired_block_bootstrap(
        conditional,
        dense,
        heads,
        blocks,
        selected_heads=[0, 1, 2, 3],
        replicates=500,
        seed=17,
    )
    assert result["delta_macro_95"] == pytest.approx([1.0, 1.0, 1.0])
    assert result["block_count"] == 30


def _gate_config() -> dict:
    return {
        "decision_gate": {
            "estimability": {
                "minimum_validation_events_per_head": 30,
                "minimum_training_events_per_head_per_session": 20,
                "minimum_training_sessions": 2,
                "minimum_estimable_heads": 7,
                "minimum_distinct_keys": 4,
                "require_both_event_types": True,
                "minimum_validation_blocks": 20,
            },
            "seed_zero": {
                "minimum_conditional_macro_exact": 0.125,
                "minimum_conditional_exact_ci_low": 0.0625,
                "minimum_macro_exact_delta": 0.03,
                "minimum_macro_exact_delta_ci_low": 0.0,
                "minimum_positive_estimable_heads": 7,
                "minimum_positive_distinct_keys": 4,
                "require_positive_both_event_types": True,
                "minimum_macro_within_2_delta": -0.01,
                "require_lower_macro_nll": True,
            },
        }
    }


def test_seed_zero_gate_passes_only_the_preregistered_joint_rule() -> None:
    config = _gate_config()
    selected_names = [
        "left:onset",
        "right:onset",
        "up:onset",
        "jump:onset",
        "left:release",
        "right:release",
        "grab:release",
    ]
    manifest = {
        "validation_block_count": 25,
        "val_counts_by_head": {name: 40 for name in HEAD_NAMES},
        "train_counts_by_session_head": {
            "a": {name: 25 for name in HEAD_NAMES},
            "b": {name: 25 for name in HEAD_NAMES},
        },
    }
    estimable = estimable_heads(manifest, config["decision_gate"]["estimability"])
    assert estimable == list(HEAD_NAMES)
    truth = np.tile(np.arange(16), len(HEAD_NAMES) * 4)
    heads = np.repeat(np.arange(len(HEAD_NAMES)), 64)
    conditional = arm_metrics(_probabilities(truth, correct=True), truth, heads, width=16)
    dense = arm_metrics(_probabilities(truth, correct=False), truth, heads, width=16)
    gate = apply_seed_zero_gate(
        config=config,
        manifest=manifest,
        conditional_metrics=conditional,
        dense_metrics=dense,
        bootstrap={
            "conditional_macro_95": [0.9, 1.0, 1.0],
            "delta_macro_95": [0.9, 1.0, 1.0],
        },
        estimable=selected_names,
    )
    assert gate["passed"]
    assert gate["decision"] == "replicate_unchanged_seeds_1_and_2"

    failed = copy.deepcopy(gate)
    failed_config = copy.deepcopy(config)
    failed_config["decision_gate"]["seed_zero"]["minimum_macro_exact_delta"] = 1.1
    failed = apply_seed_zero_gate(
        config=failed_config,
        manifest=manifest,
        conditional_metrics=conditional,
        dense_metrics=dense,
        bootstrap={
            "conditional_macro_95": [0.9, 1.0, 1.0],
            "delta_macro_95": [0.9, 1.0, 1.0],
        },
        estimable=selected_names,
    )
    assert not failed["passed"]


def test_publication_writes_marker_last_and_refuses_overwrite(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.npz"
    predictions.write_bytes(b"predictions")
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}\n")
    config = tmp_path / "config.json"
    config.write_text("{}\n")
    out = tmp_path / "report.json"
    marker = tmp_path / "complete.json"
    report = {
        "study_id": "fixture",
        "decision_gate": {"decision": "reject_phase_2_at_seed_zero_gate"},
    }

    publish_score(
        report=report,
        out=out,
        marker=marker,
        predictions_path=predictions,
        dataset_manifest_path=dataset,
        config_path=config,
    )

    saved = json.loads(marker.read_text())
    assert out.is_file() and marker.is_file()
    assert saved["report"]["sha256"]
    with pytest.raises(ValueError, match="overwrite"):
        publish_score(
            report=report,
            out=out,
            marker=marker,
            predictions_path=predictions,
            dataset_manifest_path=dataset,
            config_path=config,
        )
