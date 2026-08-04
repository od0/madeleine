from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from data.schema import KEY_ORDER
from experiments.vpt_small_calibration import (
    affine_probabilities,
    calibration_validity_guard,
    candidate_decision,
    fit_calibrators,
    fit_from_sidecar,
    parameter_matrix,
)


def calibration_fixture(rows: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4815)
    raw = rng.uniform(0.001, 0.35, size=(rows, len(KEY_ORDER)))
    truth = np.zeros_like(raw, dtype=np.uint8)
    for column in range(len(KEY_ORDER)):
        threshold = 0.07 + column * 0.015
        truth[:, column] = raw[:, column] > threshold
    active = np.ones(rows, dtype=np.uint8)
    return truth, raw.astype(np.float32), active


def write_sidecar(path: Path, truth: np.ndarray, probability: np.ndarray, active: np.ndarray) -> None:
    rows = len(truth)
    np.savez_compressed(
        path,
        y_true=truth,
        y_prob=probability,
        input_active=active,
        session_lengths=np.asarray([rows], dtype=np.int64),
        session_ids=np.asarray(["c1__run000"]),
        source_row_index=np.arange(rows, dtype=np.int64),
        source_engine_frame_idx=np.arange(1000, 1000 + rows, dtype=np.int64),
    )


def test_positive_affine_fit_is_deterministic_and_ap_invariant() -> None:
    truth, raw, active = calibration_fixture()
    first = fit_calibrators(truth, raw, active)
    second = fit_calibrators(truth, raw, active)
    assert np.array_equal(first["parameter_matrix"], second["parameter_matrix"])
    assert first["raw_ap"] == first["calibrated_ap"]
    assert first["calibrated_brier"] < first["raw_brier"]
    assert float(np.sum(first["calibrated_nll_per_key"])) < float(
        np.sum(first["raw_nll_per_key"])
    )
    assert np.all(first["parameter_matrix"][:, 0] > 0)


def test_affine_application_refuses_nonfinite_and_nonpositive_slopes() -> None:
    _, raw, _ = calibration_fixture(20)
    params = np.tile([1.0, 0.0], (len(KEY_ORDER), 1))
    params[0, 0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        affine_probabilities(raw, params)
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        affine_probabilities(raw, np.tile([1.0, 0.0], (len(KEY_ORDER), 1)))


def test_fit_receipt_is_c1_only_and_binds_frozen_e1_command(tmp_path: Path) -> None:
    truth, raw, active = calibration_fixture()
    sidecar = tmp_path / "c1.npz"
    write_sidecar(sidecar, truth, raw, active)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen-parent")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    command = tmp_path / "e1-command.txt"
    command.write_text("python -m experiments.eval_vpt_small --surface prospective-e1\n")
    capture = tmp_path / "capture.json"
    manifest_hash = "a" * 64
    capture.write_text(json.dumps({
        "role": "c1",
        "decision": "accepted",
        "model_accessed": False,
        "violations": [],
        "derived": {"build_manifest_sha256": manifest_hash},
        "support": {"rows": len(truth), "active_rows": int(active.sum())},
    }))
    evaluation = tmp_path / "c1-eval.json"
    evaluation.write_text(json.dumps({
        "schema_version": "madeleine.vpt-small-eval.v1",
        "weights": {"sha256": checkpoint_hash},
        "data": {"manifest_sha256": manifest_hash, "rows": len(truth)},
        "sidecar": {"sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest()},
    }))
    receipt_path = tmp_path / "calibrator.json"
    receipt = fit_from_sidecar(
        sidecar,
        c1_receipt=capture,
        c1_eval_report=evaluation,
        checkpoint=checkpoint,
        expected_checkpoint_sha256=checkpoint_hash,
        e1_command=command,
        out=receipt_path,
        calibrated_sidecar=tmp_path / "c1-calibrated.npz",
        repo=Path(__file__).resolve().parents[1],
    )
    assert receipt["fit_role"] == "c1_only"
    assert receipt["e1_command"]["sha256"] == hashlib.sha256(command.read_bytes()).hexdigest()
    assert parameter_matrix(receipt).shape == (7, 2)
    assert receipt_path.with_suffix(".json.sha256").is_file()

    capture.write_text(json.dumps({"role": "e1", "decision": "accepted"}))
    with pytest.raises(RuntimeError, match="accepted C1"):
        fit_from_sidecar(
            sidecar,
            c1_receipt=capture,
            c1_eval_report=evaluation,
            checkpoint=checkpoint,
            expected_checkpoint_sha256=checkpoint_hash,
            e1_command=command,
            out=tmp_path / "invalid.json",
            calibrated_sidecar=tmp_path / "invalid.npz",
            repo=Path(__file__).resolve().parents[1],
        )


def test_seventh_guard_requires_both_no_worse_and_one_improvement() -> None:
    raw = {"natural_nll": 2.0, "brier": 0.2}
    assert calibration_validity_guard(raw, {"natural_nll": 1.9, "brier": 0.2})["pass"]
    assert not calibration_validity_guard(raw, {"natural_nll": 2.0, "brier": 0.2})["pass"]
    assert not calibration_validity_guard(raw, {"natural_nll": 1.9, "brier": 0.21})["pass"]


def _report(*, macro_ap: float, state_f1: float, event_f1: float, nll: float, brier: float, ap: float) -> dict:
    return {
        "variant": "fixture",
        "natural_nll": nll,
        "brier": brier,
        "aggregate": {
            "macro_ap": macro_ap,
            "macro_state_f1": state_f1,
            "macro_event_f1_collar_2_native_frames": event_f1,
        },
        "per_key": {
            key: {
                "ap": ap,
                "recall": 0.5,
                "state_f1": state_f1,
                "prevalence": 0.1,
                "predicted_positive_rate": 0.1,
            }
            for key in KEY_ORDER
        },
        "key_state_accuracy": {
            "key_state_micro_accuracy": 0.90,
            "joint_exact_match_accuracy": 0.50,
            "always_released_key_state_micro_accuracy": 0.85,
            "always_released_joint_exact_match_accuracy": 0.45,
        },
    }


def test_candidate_decision_applies_all_seven_guards() -> None:
    raw = _report(macro_ap=0.4, state_f1=0.4, event_f1=0.2, nll=3.0, brier=0.2, ap=0.4)
    calibrated = _report(macro_ap=0.4, state_f1=0.4, event_f1=0.2, nll=2.9, brier=0.19, ap=0.4)
    gru_112 = _report(macro_ap=0.25, state_f1=0.3, event_f1=0.1, nll=4.0, brier=0.3, ap=0.2)
    gru_36 = _report(macro_ap=0.30, state_f1=0.3, event_f1=0.1, nll=4.0, brier=0.3, ap=0.3)
    decision = candidate_decision(
        raw_vpt=raw, calibrated_vpt=calibrated, gru_112m95=gru_112, gru_36m9=gru_36
    )
    assert decision["pass"] is True
    assert set(decision["diagnosis"]) == set(KEY_ORDER)
    assert all(
        entry["label"] == "positioning recovered"
        for entry in decision["diagnosis"].values()
    )
    calibrated["per_key"]["down"]["recall"] = 0.0
    failed = candidate_decision(
        raw_vpt=raw, calibrated_vpt=calibrated, gru_112m95=gru_112, gru_36m9=gru_36
    )
    assert failed["pass"] is False
    assert failed["diagnosis"]["down"]["label"] == "representation-limited"
