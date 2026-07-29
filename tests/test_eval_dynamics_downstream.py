from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from data.schema import KEY_ORDER
from experiments import eval_dynamics_downstream as module


def _small_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray([36, 42], dtype=np.int64)
    frames = int(lengths.sum())
    truth = np.zeros((frames, len(KEY_ORDER)), dtype=np.uint8)
    start = 0
    for length in lengths:
        for column in range(len(KEY_ORDER)):
            first = start + 2 + column
            truth[first : min(first + 4 + column, start + length), column] = 1
            second = start + 20 + column
            truth[second : min(second + 3, start + length), column] = 1
        start += int(length)
    probability = np.where(truth, 0.91, 0.07).astype(np.float32)
    active = np.ones(frames, dtype=np.uint8)
    return truth, probability, active, lengths


@pytest.fixture(scope="module")
def exact_sidecar(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    root = tmp_path_factory.mktemp("dynamics-eval")
    frames = module.Y4N_FRAMES
    truth = np.zeros((frames, len(KEY_ORDER)), dtype=np.uint8)
    start = 0
    for stream_length in module.Y4N_STREAM_LENGTHS:
        stop = start + stream_length
        for column in range(len(KEY_ORDER)):
            period = 37 + column * 3
            positions = np.arange(start + column + 2, stop - 5, period)
            for position in positions:
                truth[position : position + 3 + column % 3, column] = 1
        start = stop
    probability = np.where(truth, 0.9, 0.1).astype(np.float32)
    active = np.ones(frames, dtype=np.uint8)
    path = root / "preds.npz"
    np.savez_compressed(
        path,
        y_true=truth,
        y_prob=probability,
        input_active=active,
        session_lengths=np.asarray(module.Y4N_STREAM_LENGTHS, dtype=np.int64),
        session_ids=np.asarray(module.Y4N_STREAM_IDS),
    )
    return path, module.canonical_array_sha256(truth)


def test_score_fixed_surface_has_complete_fixed_schema() -> None:
    truth, probability, active, lengths = _small_surface()
    report = module.score_fixed_surface(truth, probability, active, lengths)

    assert report["macro_ap"] == pytest.approx(1.0)
    assert report["macro_state_f1_fixed_0_5"] == pytest.approx(1.0)
    assert report["events_fixed_0_5"]["exact"]["macro_combined_f1"] == pytest.approx(1.0)
    assert report["events_fixed_0_5"]["plus_minus_2"]["macro_combined_f1"] == pytest.approx(1.0)
    assert report["key_state_accuracy_fixed_0_5"] == {
        "micro": 1.0,
        "joint_exact_match": 1.0,
    }
    assert set(report["per_key_ap"]) == set(KEY_ORDER)
    assert set(report["per_key_prevalence"]) == set(KEY_ORDER)
    assert set(report["per_key_bce"]) == set(KEY_ORDER)
    assert set(report["per_key_brier"]) == set(KEY_ORDER)
    assert report["threshold_policy"] == {
        "state_probability": 0.5,
        "transition_probability": 0.5,
        "data_fitted_thresholds_used": False,
        "calibration_parameters_fitted": False,
    }
    assert set(report["baselines"]) == {
        "always_released",
        "persistence",
        "prevalence",
        "shuffled_events",
    }
    # The entire report must be strict JSON; no NaN can quietly escape.
    json.dumps(report, allow_nan=False)


def test_average_precision_keeps_constant_score_ties_together() -> None:
    truth = np.asarray([0, 1, 0, 0, 1, 0, 1, 0], dtype=np.uint8)
    probability = np.full(len(truth), truth.mean(), dtype=np.float64)
    assert module._average_precision(truth, probability) == pytest.approx(truth.mean())


def test_events_are_segment_bounded() -> None:
    truth, probability, active, lengths = _small_surface()
    # Begin the second stream held for every key.  That stream-start onset is
    # valid, but must never match an event at the end of stream one.
    boundary = int(lengths[0])
    truth[boundary : boundary + 3] = 1
    probability = np.where(truth, 0.9, 0.1).astype(np.float32)
    report = module.score_fixed_surface(truth, probability, active, lengths)
    assert report["events_fixed_0_5"]["exact"]["macro_combined_f1"] == pytest.approx(1.0)
    assert report["events_fixed_0_5"]["boundary_policy"].startswith("stored session_lengths")


def test_shuffled_event_baseline_is_deterministic() -> None:
    truth, probability, active, lengths = _small_surface()
    first = module.score_fixed_surface(truth, probability, active, lengths)
    second = module.score_fixed_surface(truth, probability, active, lengths)
    assert first["baselines"]["shuffled_events"] == second["baselines"]["shuffled_events"]
    assert first["baselines"]["shuffled_events"]["seed"] == 0
    assert first["baselines"]["shuffled_events"]["repetitions"] == 10


def test_load_validate_sidecar_requires_exact_y4n_receipt(
    exact_sidecar: tuple[Path, str],
) -> None:
    path, truth_sha = exact_sidecar
    arrays = module.load_validate_sidecar(path, expected_truth_sha256=truth_sha)
    assert arrays.support["all_frames"] == module.Y4N_FRAMES
    assert arrays.support["input_active_frames"] == module.Y4N_FRAMES
    assert arrays.support["session_ids"] == list(module.Y4N_STREAM_IDS)
    assert arrays.support["truth_sha256"] == truth_sha
    assert arrays.probability.dtype == np.float32

    with pytest.raises(ValueError, match="truth receipt changed"):
        module.load_validate_sidecar(path, expected_truth_sha256="0" * 64)


def test_load_validate_sidecar_rejects_nonfinite_probability(
    tmp_path: Path,
) -> None:
    truth = np.zeros((module.Y4N_FRAMES, len(KEY_ORDER)), dtype=np.uint8)
    probability = np.zeros_like(truth, dtype=np.float32)
    probability[0, 0] = np.nan
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        y_true=truth,
        y_prob=probability,
        input_active=np.ones(module.Y4N_FRAMES, dtype=np.uint8),
        session_lengths=np.asarray(module.Y4N_STREAM_LENGTHS, dtype=np.int64),
        session_ids=np.asarray(module.Y4N_STREAM_IDS),
    )
    with pytest.raises(ValueError, match="not finite"):
        module.load_validate_sidecar(
            path,
            expected_truth_sha256=module.canonical_array_sha256(truth),
        )


@pytest.mark.parametrize("arm", ["C", "D"])
def test_validate_run_config_accepts_only_matched_recipe(arm: str) -> None:
    config = dict(module.REQUIRED_CONFIG)
    config["_note"] = f"25.7M dynamics {arm} fixed endpoint 20458"
    module.validate_run_config(config, arm)

    changed = dict(config, max_steps=20_457)
    with pytest.raises(ValueError, match="max_steps"):
        module.validate_run_config(changed, arm)
    with pytest.raises(ValueError, match="non-matched objective"):
        module.validate_run_config({**config, "event_latch": True}, arm)


def test_cli_has_no_tuning_or_alternate_surface_interface() -> None:
    options = {
        option
        for action in module.parser()._actions
        for option in action.option_strings
    }
    assert "--threshold" not in options
    assert "--surface" not in options
    assert "--b1" not in options
    assert "--weights" not in options
    assert "--calibration" not in options


def test_validate_release_recomputes_serialized_sidecar(
    exact_sidecar: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, truth_sha = exact_sidecar
    monkeypatch.setattr(module, "Y4N_TRUTH_SHA256", truth_sha)
    sidecar = tmp_path / "dynamics_c_preds.npz"
    sidecar.write_bytes(source.read_bytes())
    arrays = module.load_validate_sidecar(sidecar)
    metrics = module.score_fixed_surface(
        arrays.truth, arrays.probability, arrays.active, arrays.lengths
    )
    report_path = tmp_path / "dynamics_c.json"
    marker_path = tmp_path / "dynamics_c_complete.json"
    checkpoint_sha = "1" * 64
    assembly_sha = "2" * 64
    report = {
        "schema_version": module.SCHEMA_VERSION,
        "study_id": module.STUDY_ID,
        "arm": "C",
        "run_id": module.ARM_RUN_IDS["C"],
        "surface": module.SURFACE,
        "weights": "final",
        "support": arrays.support,
        "metrics": metrics,
        "run_receipt": {"checkpoint_sha256": checkpoint_sha},
        "assembly_validation": {"sha256": assembly_sha},
        "prediction_sidecar": {
            "path": str(sidecar.resolve()),
            "sha256": module.sha256_file(sidecar),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    marker = {
        "schema_version": module.MARKER_SCHEMA_VERSION,
        "status": "complete",
        "study_id": module.STUDY_ID,
        "arm": "C",
        "run_id": module.ARM_RUN_IDS["C"],
        "surface": module.SURFACE,
        "weights": "final",
        "checkpoint_sha256": checkpoint_sha,
        "assembly_validation_sha256": assembly_sha,
        "report_sha256": module.sha256_file(report_path),
        "sidecar_sha256": module.sha256_file(sidecar),
    }
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")

    receipt = module.validate_release(
        report_path, sidecar, marker_path, expected_arm="C"
    )
    assert receipt["metrics_recomputed"] is True
    assert receipt["sidecar_sha256"] == module.sha256_file(sidecar)

    tampered = json.loads(report_path.read_text())
    tampered["metrics"]["macro_ap"] = 0.0
    report_path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="metrics differ"):
        module.validate_release(report_path, sidecar, marker_path)
