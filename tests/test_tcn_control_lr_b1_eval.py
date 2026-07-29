from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from data.schema import KEY_ORDER
from experiments.eval_tcn_control_lr_b1 import (
    DECISION_RELATIVE_PATH,
    EXPECTED_B1_ACTIVE_FRAMES,
    EXPECTED_B1_FRAMES,
    EXPECTED_B1_RUN_INDICES,
    EXPECTED_B1_STREAM_IDS,
    EXPECTED_B1_STREAM_LENGTHS,
    EXPECTED_B1_STREAMS,
    EXPECTED_Y4N_ACTIVE_FRAMES,
    EXPECTED_Y4N_SESSIONS,
    INFERENCE_SOURCE_PATHS,
    REGISTERED_RUNS,
    REQUIRED_DECISION_FIELDS,
    STUDY_ID,
    _verify_inference_source,
    fixed_metric_report,
    sha256_file,
    validate_b1_sidecar,
    validate_decision_release,
    validate_run,
)


RUN_ID = next(iter(REGISTERED_RUNS))


def _decision(authoring_commit: str) -> dict[str, object]:
    runs: dict[str, object] = {}
    for run_id in REGISTERED_RUNS:
        runs[run_id] = {
            "run_id": run_id,
            "inferential_role": "test",
            "config_sha256": "1" * 64,
            "launcher_sha256": "2" * 64,
            "checkpoint_sha256": "3" * 64,
            "prediction_sidecar_sha256": "4" * 64,
            "implementation_git_commit": "5" * 40,
            "training_start_utc": "2026-07-27T00:00:00Z",
            "training_end_utc": "2026-07-27T00:10:00Z",
            "final_step": 14265,
            "expected_final_step": 14265,
            "support_session_ids": EXPECTED_Y4N_SESSIONS,
            "active_frames": EXPECTED_Y4N_ACTIVE_FRAMES,
            "finite_aligned_arrays": True,
            "evaluation_weights": "final",
            "valid": True,
        }
    decision_fields: dict[str, object] = {
        field: "documented" for field in REQUIRED_DECISION_FIELDS
    }
    decision_fields.update(
        {
            "all_consulted_runs": [
                {"run_id": run_id} for run_id in REGISTERED_RUNS
            ],
            "y4n_decision_frozen_before_b1": True,
            "decision_record_sha256": None,
            "decision_git_commit": authoring_commit,
        }
    )
    record: dict[str, object] = {
        "study_id": STUDY_ID,
        "evaluation_population": {
            "session_ids": EXPECTED_Y4N_SESSIONS,
            "active_frames": EXPECTED_Y4N_ACTIVE_FRAMES,
            "oracle_metrics_used": False,
            "calibration_used": False,
            "b1_used": False,
            "threshold_policy": (
                "probability >= 0.5 for fixed state and "
                "state-transition metrics"
            ),
        },
        "runs": runs,
        "decision": decision_fields,
    }
    canonical = copy.deepcopy(record)
    encoded = json.dumps(
        canonical,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    record["decision"]["decision_record_sha256"] = hashlib.sha256(
        encoded
    ).hexdigest()
    return record


def _committed_decision(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", repo], check=True)
    subprocess.run(
        ["git", "-C", repo, "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", repo, "config", "user.name", "Test"], check=True
    )
    (repo / "README.md").write_text("test repository\n")
    subprocess.run(["git", "-C", repo, "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", repo, "commit", "-q", "-m", "base"], check=True
    )
    authoring_commit = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt = repo / DECISION_RELATIVE_PATH
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps(_decision(authoring_commit), indent=2) + "\n"
    )
    subprocess.run(
        ["git", "-C", repo, "add", DECISION_RELATIVE_PATH.as_posix()],
        check=True,
    )
    subprocess.run(
        ["git", "-C", repo, "commit", "-q", "-m", "decision"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    return receipt, digest, commit


def test_decision_release_requires_exact_clean_committed_receipt(
    tmp_path: Path,
) -> None:
    receipt, digest, commit = _committed_decision(tmp_path)

    decision, run = validate_decision_release(
        tmp_path / "repo", receipt, digest, commit, RUN_ID
    )

    assert decision["decision"]["y4n_decision_frozen_before_b1"] is True
    assert run["run_id"] == RUN_ID


def test_decision_release_rejects_working_bytes_even_with_new_hash(
    tmp_path: Path,
) -> None:
    receipt, _, commit = _committed_decision(tmp_path)
    receipt.write_text(receipt.read_text() + "\n")
    changed_digest = hashlib.sha256(receipt.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="differs from its committed bytes"):
        validate_decision_release(
            tmp_path / "repo", receipt, changed_digest, commit, RUN_ID
        )


def test_decision_release_rejects_unregistered_run_before_receipt_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unregistered TCN study run"):
        validate_decision_release(
            tmp_path,
            tmp_path / "missing.json",
            "0" * 64,
            "0" * 40,
            "not_registered",
        )


def test_decision_release_rejects_invalid_registered_run(
    tmp_path: Path,
) -> None:
    receipt, _, _ = _committed_decision(tmp_path)
    decision = json.loads(receipt.read_text())
    decision["runs"][RUN_ID]["valid"] = False
    decision["decision"]["decision_record_sha256"] = None
    encoded = json.dumps(
        decision,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    decision["decision"]["decision_record_sha256"] = hashlib.sha256(
        encoded
    ).hexdigest()
    receipt.write_text(json.dumps(decision, indent=2) + "\n")
    subprocess.run(
        ["git", "-C", tmp_path / "repo", "add", DECISION_RELATIVE_PATH],
        check=True,
    )
    subprocess.run(
        ["git", "-C", tmp_path / "repo", "commit", "-q", "-m", "invalid"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", tmp_path / "repo", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="not valid in the frozen y4n"):
        validate_decision_release(
            tmp_path / "repo", receipt, digest, commit, RUN_ID
        )


def _run_receipt(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    run = tmp_path / RUN_ID
    run.mkdir()
    recipe = REGISTERED_RUNS[RUN_ID]
    config = {
        "temporal_arch": "aligned_tcn",
        "precomputed_features": True,
        "window": 128,
        "frame_stride": 3,
        "window_mode": "centered",
        "input_config": "pixels",
        "active_targets_only": True,
        "seed": 0,
        "max_steps": 14265,
        **recipe,
    }
    (run / "config.json").write_text(json.dumps(config) + "\n")
    torch.save(
        {"steps": 14265, "final_state_dict": {"value": torch.ones(1)}},
        run / "model.pt",
    )
    implementation_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return run, {
        "config_sha256": sha256_file(run / "config.json"),
        "checkpoint_sha256": sha256_file(run / "model.pt"),
        "implementation_git_commit": implementation_commit,
    }


def test_registered_run_returns_final_weights_only(tmp_path: Path) -> None:
    run, receipt = _run_receipt(tmp_path)

    config, final_state, source = validate_run(
        Path.cwd(), run, RUN_ID, receipt
    )

    assert config["learning_rate"] == REGISTERED_RUNS[RUN_ID]["learning_rate"]
    assert torch.equal(final_state["value"], torch.ones(1))
    assert source["implementation_git_commit"] == receipt[
        "implementation_git_commit"
    ]


def test_registered_run_rejects_selected_only_checkpoint(tmp_path: Path) -> None:
    run, receipt = _run_receipt(tmp_path)
    torch.save(
        {"steps": 14265, "model_state_dict": {"value": torch.ones(1)}},
        run / "model.pt",
    )
    receipt["checkpoint_sha256"] = sha256_file(run / "model.pt")

    with pytest.raises(ValueError, match="lacks final_state_dict"):
        validate_run(Path.cwd(), run, RUN_ID, receipt)


def _valid_sidecar(path: Path) -> tuple[str, str]:
    lengths = np.asarray(EXPECTED_B1_STREAM_LENGTHS, dtype=np.int64)
    active = np.zeros(EXPECTED_B1_FRAMES, dtype=np.uint8)
    active[:EXPECTED_B1_ACTIVE_FRAMES] = 1
    truth = np.zeros((EXPECTED_B1_FRAMES, len(KEY_ORDER)), dtype=np.uint8)
    np.savez_compressed(
        path,
        y_true=truth,
        y_prob=np.full(
            (EXPECTED_B1_FRAMES, len(KEY_ORDER)), 0.25, dtype=np.float32
        ),
        input_active=active,
        session_lengths=lengths,
        session_ids=np.asarray(EXPECTED_B1_STREAM_IDS),
    )
    return (
        hashlib.sha256(truth.tobytes()).hexdigest(),
        hashlib.sha256(active.tobytes()).hexdigest(),
    )


def test_b1_sidecar_requires_exact_support_and_stream_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "preds.npz"
    truth_sha, active_sha = _valid_sidecar(path)

    support = validate_b1_sidecar(
        path,
        expected_truth_sha256=truth_sha,
        expected_active_sha256=active_sha,
    )

    assert support["all_frames"] == EXPECTED_B1_FRAMES
    assert support["input_active_frames"] == EXPECTED_B1_ACTIVE_FRAMES
    assert support["streams"] == EXPECTED_B1_STREAMS
    assert support["stream_lengths"] == EXPECTED_B1_STREAM_LENGTHS
    assert EXPECTED_B1_RUN_INDICES[:3] == [13, 15, 19]


def test_b1_sidecar_rejects_nonfinite_probability(tmp_path: Path) -> None:
    path = tmp_path / "preds.npz"
    truth_sha, active_sha = _valid_sidecar(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["y_prob"][0, 0] = np.nan
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="not finite"):
        validate_b1_sidecar(
            path,
            expected_truth_sha256=truth_sha,
            expected_active_sha256=active_sha,
        )


def test_b1_sidecar_rejects_shifted_stream_boundary(tmp_path: Path) -> None:
    path = tmp_path / "preds.npz"
    truth_sha, active_sha = _valid_sidecar(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["session_lengths"] = arrays["session_lengths"].copy()
    arrays["session_lengths"][0] -= 1
    arrays["session_lengths"][1] += 1
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="lengths or boundaries changed"):
        validate_b1_sidecar(
            path,
            expected_truth_sha256=truth_sha,
            expected_active_sha256=active_sha,
        )


def test_inference_source_must_equal_training_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", repo], check=True)
    subprocess.run(
        ["git", "-C", repo, "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", repo, "config", "user.name", "Test"], check=True
    )
    for relative in INFERENCE_SOURCE_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n")
    subprocess.run(["git", "-C", repo, "add", "."], check=True)
    subprocess.run(
        ["git", "-C", repo, "commit", "-q", "-m", "source"], check=True
    )
    commit = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    receipt = _verify_inference_source(repo, commit)
    assert receipt["implementation_git_commit"] == commit

    (repo / INFERENCE_SOURCE_PATHS[0]).write_text("changed\n")
    with pytest.raises(ValueError, match="differs from training commit"):
        _verify_inference_source(repo, commit)


def test_fixed_metric_report_has_no_fitted_threshold_or_calibrator() -> None:
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
    active = np.ones(4, dtype=bool)

    report = fixed_metric_report(truth, probability, active, [2, 2])
    rendered = json.dumps(report, sort_keys=True)

    assert "oracle" not in rendered.lower()
    assert report["threshold_policy"] == {
        "state_probability": 0.5,
        "transition_probability": 0.5,
        "data_fitted_thresholds_used": False,
        "calibration_parameters_fitted": False,
    }
    assert report["input_active_only"][
        "macro_state_f1_fixed_0_5"
    ] == pytest.approx(1.0)


@pytest.mark.requires_private_artifacts(
    "experiments/run_tcn_control_lr_b1_eval.sh"
)
def test_wrapper_is_final_only_allowlisted_and_receipt_gated() -> None:
    wrapper = Path("experiments/run_tcn_control_lr_b1_eval.sh").read_text()

    assert all(run_id in wrapper for run_id in REGISTERED_RUNS)
    assert "--decision-receipt" in wrapper
    assert "--decision-sha256" in wrapper
    assert "--decision-commit" in wrapper
    assert "_final_b1_postdecision" in wrapper
    assert "selected" not in wrapper
    assert "run_full_corpus_b1_eval.sh" in wrapper
