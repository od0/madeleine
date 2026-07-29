from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.schema import KEY_ORDER
import experiments.eval_wild_provisional_gru as evaluator


requires_private_contract = pytest.mark.requires_private_artifacts(
    "experiments/configs/wild_provisional_broad7_gru_decision.json"
)


def _valid_y4n_sidecar(path: Path) -> str:
    truth = np.zeros(
        (evaluator.Y4N_FRAMES, len(KEY_ORDER)), dtype=np.uint8
    )
    probability = np.full(truth.shape, 0.25, dtype=np.float32)
    active = np.ones(evaluator.Y4N_FRAMES, dtype=np.uint8)
    np.savez_compressed(
        path,
        y_true=truth,
        y_prob=probability,
        input_active=active,
        session_lengths=np.asarray(
            evaluator.Y4N_STREAM_LENGTHS, dtype=np.int64
        ),
        session_ids=np.asarray(evaluator.Y4N_STREAM_IDS),
    )
    return evaluator._canonical_array_sha256(truth)


def _rewrite_sidecar(path: Path, **updates: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    arrays.update(updates)
    np.savez_compressed(path, **arrays)


@requires_private_contract
def test_contract_is_bound_to_exact_preregistered_bytes() -> None:
    repo = Path.cwd()
    path = repo / evaluator.CONTRACT_RELATIVE_PATH

    contract = evaluator.validate_contract(
        repo, path, evaluator.CONTRACT_SHA256
    )

    assert contract["study_id"] == evaluator.STUDY_ID
    assert contract["model_contract"]["run_id"] == evaluator.RUN_ID
    with pytest.raises(ValueError, match="not the preregistered digest"):
        evaluator.validate_contract(repo, path, "0" * 64)


@requires_private_contract
def test_expected_run_config_is_the_exact_one_pass_gru_recipe() -> None:
    repo = Path.cwd()
    contract = evaluator.validate_contract(
        repo,
        repo / evaluator.CONTRACT_RELATIVE_PATH,
        evaluator.CONTRACT_SHA256,
    )

    config = evaluator.expected_run_config(repo, contract)

    assert config["max_steps"] == evaluator.EXPECTED_FINAL_STEP
    assert config["eval_interval"] == evaluator.EXPECTED_FINAL_STEP
    assert config["feature_deltas"] is True
    assert "temporal_arch" not in config
    assert config["_note"] == evaluator.EXPECTED_NOTE


@requires_private_contract
def test_exact_recipe_instantiates_the_declared_gru_capacity() -> None:
    repo = Path.cwd()
    contract = evaluator.validate_contract(
        repo,
        repo / evaluator.CONTRACT_RELATIVE_PATH,
        evaluator.CONTRACT_SHA256,
    )
    model = evaluator.BadelineIDM(
        evaluator.expected_run_config(repo, contract)
    )

    assert model.temporal_arch == "gru"
    assert isinstance(model.temporal, torch.nn.GRUCell)
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        evaluator.EXPECTED_TRAINABLE_PARAMETERS
    )


class _TinyGRU(torch.nn.Module):
    def __init__(self, config: object) -> None:
        super().__init__()
        self.temporal_arch = "gru"
        self.temporal = torch.nn.GRUCell(1, 1)


def _write_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    config_update: dict[str, object] | None = None,
    include_final: bool = True,
) -> tuple[Path, dict[str, object]]:
    repo = Path.cwd()
    contract = evaluator.validate_contract(
        repo,
        repo / evaluator.CONTRACT_RELATIVE_PATH,
        evaluator.CONTRACT_SHA256,
    )
    config = evaluator.expected_run_config(repo, contract)
    if config_update:
        config.update(config_update)
    run = tmp_path / evaluator.RUN_ID
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model = _TinyGRU(config)
    final_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    checkpoint: dict[str, object] = {
        "config": config,
        "key_order": list(KEY_ORDER),
        "model_state_dict": final_state,
        "steps": evaluator.EXPECTED_FINAL_STEP,
        "best_val_step": evaluator.EXPECTED_FINAL_STEP,
        "best_val_mean_bce": 0.5,
        "initialized_from": None,
        "positive_weight": [2.0] * len(KEY_ORDER),
    }
    if include_final:
        checkpoint["final_state_dict"] = final_state
    torch.save(checkpoint, run / "model.pt")
    monkeypatch.setattr(evaluator, "BadelineIDM", _TinyGRU)
    monkeypatch.setattr(
        evaluator,
        "EXPECTED_TRAINABLE_PARAMETERS",
        sum(parameter.numel() for parameter in model.parameters()),
    )
    return run, contract


@requires_private_contract
def test_run_validation_loads_final_gru_weights_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, contract = _write_run(tmp_path, monkeypatch)

    config, model, receipt = evaluator.validate_run(
        Path.cwd(), run, evaluator.RUN_ID, contract
    )

    assert config["max_steps"] == evaluator.EXPECTED_FINAL_STEP
    assert model.temporal_arch == "gru"
    assert receipt["evaluation_weights"] == "final_state_dict"
    assert receipt["selected_final_tensors_identical"] is True


@requires_private_contract
def test_run_validation_rejects_recipe_drift_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, contract = _write_run(
        tmp_path, monkeypatch, config_update={"learning_rate": 0.001}
    )

    with pytest.raises(ValueError, match="differs from frozen recipe"):
        evaluator.validate_run(Path.cwd(), run, evaluator.RUN_ID, contract)


@requires_private_contract
def test_run_validation_rejects_selected_only_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, contract = _write_run(
        tmp_path, monkeypatch, include_final=False
    )

    with pytest.raises(ValueError, match="lacks final_state_dict"):
        evaluator.validate_run(Path.cwd(), run, evaluator.RUN_ID, contract)


def test_y4n_sidecar_requires_exact_support_and_truth_receipt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.npz"
    truth_sha256 = _valid_y4n_sidecar(path)

    support = evaluator.validate_y4n_sidecar(
        path, expected_truth_sha256=truth_sha256
    )

    assert support["all_frames"] == evaluator.Y4N_FRAMES
    assert support["input_active_frames"] == evaluator.Y4N_FRAMES
    assert support["session_ids"] == evaluator.Y4N_STREAM_IDS
    assert support["stream_lengths"] == evaluator.Y4N_STREAM_LENGTHS


def test_y4n_sidecar_rejects_shifted_boundary(tmp_path: Path) -> None:
    path = tmp_path / "predictions.npz"
    truth_sha256 = _valid_y4n_sidecar(path)
    lengths = np.asarray(evaluator.Y4N_STREAM_LENGTHS, dtype=np.int64)
    lengths[0] -= 1
    lengths[1] += 1
    _rewrite_sidecar(path, session_lengths=lengths)

    with pytest.raises(ValueError, match="lengths or boundaries changed"):
        evaluator.validate_y4n_sidecar(
            path, expected_truth_sha256=truth_sha256
        )


def test_y4n_sidecar_rejects_inactive_or_nonfinite_rows(tmp_path: Path) -> None:
    path = tmp_path / "predictions.npz"
    truth_sha256 = _valid_y4n_sidecar(path)
    active = np.ones(evaluator.Y4N_FRAMES, dtype=np.uint8)
    active[0] = 0
    _rewrite_sidecar(path, input_active=active)
    with pytest.raises(ValueError, match="active support changed"):
        evaluator.validate_y4n_sidecar(
            path, expected_truth_sha256=truth_sha256
        )

    _valid_y4n_sidecar(path)
    with np.load(path, allow_pickle=False) as archive:
        probability = np.asarray(archive["y_prob"])
    probability[0, 0] = np.nan
    _rewrite_sidecar(path, y_prob=probability)
    with pytest.raises(ValueError, match="not finite"):
        evaluator.validate_y4n_sidecar(
            path, expected_truth_sha256=truth_sha256
        )


def _valid_y4n_release(
    report_path: Path,
    marker_path: Path,
    *,
    checkpoint_sha256: str,
) -> str:
    sidecar = report_path.with_name("y4n_predictions.npz")
    truth_sha256 = _valid_y4n_sidecar(sidecar)
    sidecar_sha256 = evaluator.sha256_file(sidecar)
    report = {
        "schema_version": evaluator.SCHEMA_VERSION,
        "study_id": evaluator.STUDY_ID,
        "run_id": evaluator.RUN_ID,
        "surface": evaluator.Y4N_SURFACE,
        "weights": "final",
        "contract": {"sha256": evaluator.CONTRACT_SHA256},
        "run_receipt": {"checkpoint_sha256": checkpoint_sha256},
        "support": {
            "all_frames": evaluator.Y4N_FRAMES,
            "input_active_frames": evaluator.Y4N_FRAMES,
            "streams": len(evaluator.Y4N_STREAM_IDS),
            "session_ids": evaluator.Y4N_STREAM_IDS,
            "stream_lengths": evaluator.Y4N_STREAM_LENGTHS,
            "truth_sha256": truth_sha256,
            "finite_aligned_arrays": True,
        },
        "fixed_metrics": {
            "threshold_policy": {
                "state_probability": 0.5,
                "transition_probability": 0.5,
                "data_fitted_thresholds_used": False,
                "calibration_parameters_fitted": False,
            }
        },
        "prediction_sidecar": {
            "path": str(sidecar),
            "sha256": sidecar_sha256,
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker = {
        "schema_version": evaluator.MARKER_SCHEMA_VERSION,
        "status": "complete",
        "study_id": evaluator.STUDY_ID,
        "run_id": evaluator.RUN_ID,
        "surface": evaluator.Y4N_SURFACE,
        "contract_sha256": evaluator.CONTRACT_SHA256,
        "checkpoint_sha256": checkpoint_sha256,
        "report_sha256": evaluator.sha256_file(report_path),
        "sidecar_sha256": sidecar_sha256,
    }
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return truth_sha256


def test_b1_release_gate_requires_complete_exact_y4n_result(
    tmp_path: Path,
) -> None:
    report = tmp_path / "y4n.json"
    marker = tmp_path / ".y4n_done"
    checkpoint_sha256 = "b" * 64

    with pytest.raises(ValueError, match="required before B1"):
        evaluator.validate_y4n_release(
            report,
            marker,
            contract_sha256=evaluator.CONTRACT_SHA256,
            checkpoint_sha256=checkpoint_sha256,
        )

    truth_sha256 = _valid_y4n_release(
        report, marker, checkpoint_sha256=checkpoint_sha256
    )
    receipt = evaluator.validate_y4n_release(
        report,
        marker,
        contract_sha256=evaluator.CONTRACT_SHA256,
        checkpoint_sha256=checkpoint_sha256,
        expected_truth_sha256=truth_sha256,
    )
    assert receipt["surface"] == evaluator.Y4N_SURFACE
    assert receipt["weights"] == "final"


def test_y4n_release_gate_rejects_fitted_metric_output(tmp_path: Path) -> None:
    report = tmp_path / "y4n.json"
    marker = tmp_path / ".y4n_done"
    checkpoint_sha256 = "b" * 64
    truth_sha256 = _valid_y4n_release(
        report, marker, checkpoint_sha256=checkpoint_sha256
    )
    value = json.loads(report.read_text())
    value["oracle_threshold"] = 0.25
    report.write_text(json.dumps(value) + "\n")

    with pytest.raises(ValueError, match="fitted-metric diagnostic"):
        evaluator.validate_y4n_release(
            report,
            marker,
            contract_sha256=evaluator.CONTRACT_SHA256,
            checkpoint_sha256=checkpoint_sha256,
            expected_truth_sha256=truth_sha256,
        )


def test_fixed_metrics_have_only_natural_thresholds() -> None:
    truth = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    probability = np.where(truth, 0.9, 0.1).astype(np.float32)
    metrics = evaluator.fixed_metric_report(
        truth, probability, np.ones(4, dtype=bool), [2, 2]
    )

    assert not evaluator._contains_disallowed_metric_language(metrics)
    assert metrics["threshold_policy"] == {
        "state_probability": 0.5,
        "transition_probability": 0.5,
        "data_fitted_thresholds_used": False,
        "calibration_parameters_fitted": False,
    }


def test_publication_uses_completion_marker_as_commit_point(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    sidecar = tmp_path / "predictions.npz"
    marker = tmp_path / ".done"
    temporary_report = evaluator._temporary_path(report)
    temporary_sidecar = evaluator._temporary_path(sidecar, npz=True)
    temporary_report.write_text("{}\n", encoding="utf-8")
    temporary_sidecar.write_bytes(b"npz receipt")
    marker_value = {
        "status": "complete",
        "report_sha256": hashlib.sha256(b"{}\n").hexdigest(),
        "sidecar_sha256": hashlib.sha256(b"npz receipt").hexdigest(),
    }

    evaluator._publish_result(
        report_path=report,
        temporary_report=temporary_report,
        sidecar_path=sidecar,
        temporary_sidecar=temporary_sidecar,
        marker_path=marker,
        marker=marker_value,
    )

    assert report.is_file() and sidecar.is_file() and marker.is_file()
    assert json.loads(marker.read_text()) == marker_value
    assert not temporary_report.exists() and not temporary_sidecar.exists()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        evaluator._publish_result(
            report_path=report,
            temporary_report=temporary_report,
            sidecar_path=sidecar,
            temporary_sidecar=temporary_sidecar,
            marker_path=marker,
            marker=marker_value,
        )
