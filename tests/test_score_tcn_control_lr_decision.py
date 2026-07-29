from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from data.schema import KEY_ORDER
from experiments.score_tcn_control_lr_decision import (
    _canonical_sha256,
    _persistence,
    build_decision,
    write_decision,
)


ROLES = (
    "matched_gru",
    "weighted_tcn_lr3e4",
    "event_head_tcn_direct",
    "natural_tcn_control",
    "weighted_tcn_lr1e4",
    "weighted_tcn_lr1e3",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _base_config() -> dict[str, object]:
    return {
        "_note": "weighted",
        "active_targets_only": True,
        "class_balance": True,
        "class_balance_max": 10.0,
        "initial_train_eval": False,
        "learning_rate": 0.0003,
        "max_steps": 10,
        "seed": 0,
        "temporal_arch": "aligned_tcn",
        "transition_weight": 8.0,
    }


def _configs() -> dict[str, dict[str, object]]:
    weighted = _base_config()
    natural = copy.deepcopy(weighted)
    natural.update(
        {
            "_note": "natural",
            "class_balance": False,
            "transition_weight": 1.0,
        }
    )
    event = copy.deepcopy(natural)
    event.pop("class_balance_max")
    event.pop("initial_train_eval")
    event.update(
        {
            "_note": "events",
            "event_class_balance_max": 50.0,
            "event_latch": True,
            "state_loss_weight": 1.0,
            "onset_loss_weight": 0.5,
            "release_loss_weight": 0.5,
        }
    )
    lr1e4 = copy.deepcopy(weighted)
    lr1e4.update({"_note": "lr1e4", "learning_rate": 0.0001})
    lr1e3 = copy.deepcopy(weighted)
    lr1e3.update({"_note": "lr1e3", "learning_rate": 0.001})
    gru = copy.deepcopy(weighted)
    gru["_note"] = "gru context"
    return {
        "matched_gru": gru,
        "weighted_tcn_lr3e4": weighted,
        "event_head_tcn_direct": event,
        "natural_tcn_control": natural,
        "weighted_tcn_lr1e4": lr1e4,
        "weighted_tcn_lr1e3": lr1e3,
    }


def _truth() -> np.ndarray:
    truth = np.zeros((18, len(KEY_ORDER)), dtype=np.uint8)
    for column in range(len(KEY_ORDER)):
        start = 1 + column
        truth[start : start + 2, column] = 1
        truth[10 + (column % 3) : 12 + (column % 3), column] = 1
    return truth


def _probability(truth: np.ndarray, role: str) -> np.ndarray:
    # Aggregate ordering: lr1e3 > natural > event > lr1e4 > weighted > GRU.
    scales = {
        "matched_gru": 0.42,
        "weighted_tcn_lr3e4": 0.52,
        "event_head_tcn_direct": 0.68,
        "natural_tcn_control": 0.72,
        "weighted_tcn_lr1e4": 0.60,
        "weighted_tcn_lr1e3": 0.82,
    }
    scale = scales[role]
    row_signal = np.linspace(0.0, 0.08, len(truth), dtype=np.float64)[:, None]
    probability = 0.12 + row_signal + scale * truth
    probability += np.arange(len(KEY_ORDER), dtype=np.float64)[None, :] * 0.002
    return np.clip(probability, 0.0, 0.99).astype(np.float32)


def _fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    session_ids = ["y4n__r008__stream000", "y4n__r009__stream000"]
    lengths = [9, 9]
    run_ids = {role: f"run_{role}" for role in ROLES}
    study = {
        "schema_version": 1,
        "study_id": "fixture",
        "status": "preregistered_before_new_run_launch",
        "preregistered_at_utc": "2026-01-01T00:00:00Z",
        "frozen_population": {
            "surface": "mapped fixture",
            "role": "development",
            "session_ids": session_ids,
            "expected_active_frames": sum(lengths),
            "expected_frames_by_session": dict(zip(session_ids, lengths, strict=True)),
            "probability_policy": "raw",
            "threshold_policy": ">= 0.5",
            "event_policy": "segment bounded",
        },
        "consulted_runs": {
            role: {"run_id": run_ids[role], "role": f"role for {role}"}
            for role in ROLES
        },
        "multiplicity": {
            "number_of_new_runs": 3,
            "number_of_lr_candidates_compared_with_lr3e4": 2,
            "policy": "disclose all",
        },
        "decision_rules": {
            "material_macro_ap_effect": {
                "minimum_absolute_macro_ap_delta": 0.005,
                "minimum_same_direction_stream_deltas_out_of_8": 1,
            },
            "event_regression_guards": {
                "exact_event_f1_minimum_delta": -0.002,
                "plus_minus_2_event_f1_minimum_delta": -0.005,
            },
            "gru_crossing": {"rule": "sensitivity only"},
            "single_seed": {"rule": "do not promote seed zero"},
        },
    }
    preregistration = (
        tmp_path / "experiments" / "configs" / "tcn_control_lr_decision.json"
    )
    _write_json(preregistration, study)
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    subprocess.run(
        ["git", "-C", tmp_path, "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", tmp_path, "config", "user.name", "Test"], check=True
    )
    subprocess.run(
        ["git", "-C", tmp_path, "add", preregistration.relative_to(tmp_path)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", tmp_path, "commit", "-q", "-m", "preregister"],
        check=True,
    )
    preregistration_commit = subprocess.run(
        ["git", "-C", tmp_path, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    launcher = tmp_path / "launcher.sh"
    launcher.write_text("#!/bin/sh\nexit 0\n")
    truth = _truth()
    configs = _configs()
    receipt_runs: dict[str, object] = {}
    for role in ROLES:
        config_path = tmp_path / f"{role}.json"
        _write_json(config_path, configs[role])
        artifact_root = tmp_path / "results" / "idm"
        sidecar = artifact_root / f"{run_ids[role]}_final_nitrogen_val_preds.npz"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sidecar,
            y_true=truth,
            y_prob=_probability(truth, role),
            input_active=np.ones(len(truth), dtype=np.uint8),
            session_lengths=np.asarray(lengths, dtype=np.int64),
            session_ids=np.asarray(session_ids),
        )
        report = artifact_root / f"{run_ids[role]}_final_nitrogen_val.json"
        _write_json(
            report,
            {
                "run": f"/ephemeral/results/idm/{run_ids[role]}",
                "weights": "final",
                "label_kind": "mapped_foreign_nitrogen",
                "sessions": [
                    session_id.removesuffix("__stream000")
                    for session_id in session_ids
                ],
                "all_frames": {"n": sum(lengths)},
                "input_active_only": {"n": sum(lengths)},
            },
        )
        receipt_runs[role] = {
            "run_id": run_ids[role],
            "report_path": str(report),
            "sidecar_path": str(sidecar),
            "config_path": str(config_path),
            "launcher_path": str(launcher),
            "checkpoint_sha256": "b" * 64,
            "implementation_git_commit": "c" * 40,
            "training_start_utc": "2026-01-01T00:00:00Z",
            "training_end_utc": "2026-01-01T00:10:00Z",
            "final_step": 10,
            "expected_final_step": 10,
            "weights": "final",
            "b1_used_before_decision": False,
        }
    receipts = tmp_path / "receipts.json"
    _write_json(receipts, {"schema_version": 1, "runs": receipt_runs})
    return preregistration, receipts, preregistration_commit


def _build(tmp_path: Path) -> dict[str, object]:
    preregistration, receipts, preregistration_commit = _fixture(tmp_path)
    return build_decision(
        repo=tmp_path,
        preregistration_path=preregistration,
        preregistration_git_commit=preregistration_commit,
        receipt_manifest_path=receipts,
        decision_git_commit="d" * 40,
    )


def test_builds_complete_deterministic_preregistered_decision(tmp_path: Path) -> None:
    first = _build(tmp_path)
    preregistration = (
        tmp_path / "experiments" / "configs" / "tcn_control_lr_decision.json"
    )
    receipts = tmp_path / "receipts.json"
    preregistration_commit = subprocess.run(
        ["git", "-C", tmp_path, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    second = build_decision(
        repo=tmp_path,
        preregistration_path=preregistration,
        preregistration_git_commit=preregistration_commit,
        receipt_manifest_path=receipts,
        decision_git_commit="d" * 40,
    )
    assert first == second
    assert len(first["runs"]) == 6
    assert first["evaluation_population"]["active_frames"] == 18
    assert first["evaluation_population"]["oracle_metrics_used"] is False
    assert first["evaluation_population"]["calibration_used"] is False
    assert first["evaluation_population"]["b1_used"] is False

    decision = first["decision"]
    required = {
        "all_consulted_runs",
        "multiplicity_disclosure",
        "objective_contrast",
        "event_head_gradient_contrast",
        "lr_sensitivity",
        "event_regression_guards",
        "single_seed_limit",
        "y4n_decision_frozen_before_b1",
        "decision_record_sha256",
        "decision_git_commit",
    }
    assert required.issubset(decision)
    assert decision["y4n_decision_frozen_before_b1"] is True
    assert decision["lr_sensitivity"]["headline_remains_weighted_tcn_lr3e4"] is True
    assert decision["lr_sensitivity"]["sensitivity_candidate_promoted"] is False
    digest = decision["decision_record_sha256"]
    unhashed = copy.deepcopy(first)
    unhashed["decision"]["decision_record_sha256"] = None
    assert digest == _canonical_sha256(unhashed)

    for run in first["runs"].values():
        assert run["active_frames"] == 18
        assert run["finite_aligned_arrays"] is True
        assert run["valid"] is True
        assert run["evaluation_report_path"].endswith(
            "_final_nitrogen_val.json"
        )
        assert len(run["evaluation_report_sha256"]) == 64
        assert set(run["per_stream"]) == {
            "y4n__r008__stream000",
            "y4n__r009__stream000",
        }
        assert "exact" in run["metrics"][
            "segment_bounded_combined_event_f1_fixed_0_5"
        ]
        assert "plus_minus_2" in run["baselines"]["one_frame_persistence"][
            "segment_bounded_combined_event_f1"
        ]

    serialized = json.dumps(first, allow_nan=False, sort_keys=True).lower()
    assert "oracle_threshold" not in serialized
    assert "calibrated" not in serialized


def test_truth_mismatch_fails_closed(tmp_path: Path) -> None:
    preregistration, receipts_path, preregistration_commit = _fixture(tmp_path)
    receipts = json.loads(receipts_path.read_text())
    sidecar_path = Path(receipts["runs"]["weighted_tcn_lr1e4"]["sidecar_path"])
    with np.load(sidecar_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["y_true"] = arrays["y_true"].copy()
    arrays["y_true"][0, 0] = 1 - arrays["y_true"][0, 0]
    np.savez_compressed(sidecar_path, **arrays)

    with pytest.raises(ValueError, match="truth differs"):
        build_decision(
            repo=tmp_path,
            preregistration_path=preregistration,
            preregistration_git_commit=preregistration_commit,
            receipt_manifest_path=receipts_path,
            decision_git_commit="d" * 40,
        )


def test_missing_sidecar_never_writes_a_partial_decision(tmp_path: Path) -> None:
    preregistration, receipts_path, preregistration_commit = _fixture(tmp_path)
    receipts = json.loads(receipts_path.read_text())
    Path(receipts["runs"]["weighted_tcn_lr1e3"]["sidecar_path"]).unlink()
    output = tmp_path / "decision.json"

    with pytest.raises(FileNotFoundError):
        result = build_decision(
            repo=tmp_path,
            preregistration_path=preregistration,
            preregistration_git_commit=preregistration_commit,
            receipt_manifest_path=receipts_path,
            decision_git_commit="d" * 40,
        )
        write_decision(output, result)
    assert not output.exists()


def test_modified_preregistration_fails_closed(tmp_path: Path) -> None:
    preregistration, receipts, preregistration_commit = _fixture(tmp_path)
    preregistration.write_text(preregistration.read_text() + "\n")

    with pytest.raises(ValueError, match="differs from its pre-launch commit"):
        build_decision(
            repo=tmp_path,
            preregistration_path=preregistration,
            preregistration_git_commit=preregistration_commit,
            receipt_manifest_path=receipts,
            decision_git_commit="d" * 40,
        )


def test_selected_or_renamed_sidecar_is_not_admissible(tmp_path: Path) -> None:
    preregistration, receipts_path, preregistration_commit = _fixture(tmp_path)
    receipts = json.loads(receipts_path.read_text())
    role = "weighted_tcn_lr1e4"
    final_path = Path(receipts["runs"][role]["sidecar_path"])
    selected_path = final_path.with_name(
        final_path.name.replace("_final_", "_selected_")
    )
    selected_path.write_bytes(final_path.read_bytes())
    receipts["runs"][role]["sidecar_path"] = str(selected_path)
    _write_json(receipts_path, receipts)

    with pytest.raises(ValueError, match="canonical final artifact path"):
        build_decision(
            repo=tmp_path,
            preregistration_path=preregistration,
            preregistration_git_commit=preregistration_commit,
            receipt_manifest_path=receipts_path,
            decision_git_commit="d" * 40,
        )


def test_paired_report_must_be_final_mapped_y4n(tmp_path: Path) -> None:
    preregistration, receipts_path, preregistration_commit = _fixture(tmp_path)
    receipts = json.loads(receipts_path.read_text())
    report_path = Path(
        receipts["runs"]["natural_tcn_control"]["report_path"]
    )
    report = json.loads(report_path.read_text())
    report["weights"] = "selected"
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="not final weights"):
        build_decision(
            repo=tmp_path,
            preregistration_path=preregistration,
            preregistration_git_commit=preregistration_commit,
            receipt_manifest_path=receipts_path,
            decision_git_commit="d" * 40,
        )


def test_persistence_is_reset_at_each_stream_boundary(tmp_path: Path) -> None:
    truth = np.zeros((6, len(KEY_ORDER)), dtype=bool)
    truth[2, 0] = True
    truth[3, 1] = True
    prediction = _persistence(truth, np.asarray([3, 3], dtype=np.int64))

    assert not prediction[2, 0]
    # A concatenation-based implementation would copy frame 2's left press
    # into frame 3.  The stream-aware implementation resets every key.
    assert not prediction[3].any()
    assert prediction[4, 1]
