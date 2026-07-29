from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from data.schema import KEY_ORDER
import experiments.eval_provisional_blend_gru as blend_evaluator
from experiments.eval_tcn_control_lr_b1 import fixed_metric_report
import experiments.score_provisional_blend_y4n_decision as scorer
from experiments.score_tcn_control_lr_decision import _canonical_sha256, _score_run


SOURCE_REPO = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit(repo: Path, message: str) -> str:
    subprocess.run(["git", "-C", repo, "add", "."], check=True)
    subprocess.run(
        ["git", "-C", repo, "commit", "-q", "-m", message], check=True
    )
    return subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sampling(contract: dict[str, object], arm_name: str) -> dict[str, object]:
    spec = scorer.ARM_SPECS[arm_name]
    sources = contract["sources"]  # type: ignore[assignment]
    pools = {
        "nitrogen": 228_237,
        "local": int(sources["local"]["complete_segment_items"]),
        "wild_provisional": int(
            sources["wild_provisional"]["complete_segment_items"]
        ),
    }
    sessions = {"nitrogen": 1062, "local": 3, "wild_provisional": 2058}
    sources: dict[str, object] = {}
    for name, draws in spec["draws"].items():
        pool = pools[name]
        unique = min(pool, draws)
        sources[name] = {
            "session_count": sessions[name],
            "segment_items": pool,
            "scheduled_draws": draws,
            "actual_draws": draws,
            "unique_segment_items_drawn": unique,
            "repeat_draws": draws - unique,
            "effective_pool_passes": draws / pool,
            "completed_pool_passes": draws // pool,
            "minimum_draws_per_item": draws // pool,
            "maximum_draws_per_item": (draws + pool - 1) // pool,
            "mean_draws_per_item": draws / pool,
        }
    return {
        "format_version": "madeleine.source-balanced-batch.v1",
        "seed": 0,
        "cycle_steps": 5,
        "cycle_items": 80,
        "batch_items": 16,
        "scheduled_steps": 14_265,
        "actual_steps": 14_265,
        "step_cycle": [
            {key: row[key] for key in sorted(row)} for row in spec["cycle"]
        ],
        "complete": True,
        "sources": sources,
    }


def _truth(streams: int = 8, length: int = 4) -> np.ndarray:
    segment = np.asarray([0, 1, 1, 0], dtype=np.uint8)
    matrix = np.repeat(segment[:, None], len(KEY_ORDER), axis=1).reshape(
        length, len(KEY_ORDER)
    )
    return np.tile(matrix, (streams, 1))


def _probability(truth: np.ndarray, adjustment: float = 0.0) -> np.ndarray:
    row = np.arange(len(truth), dtype=np.float32)[:, None] * 1e-5
    key = np.arange(len(KEY_ORDER), dtype=np.float32)[None, :] * 1e-4
    result = 0.2 + 0.6 * truth.astype(np.float32) + row + key + adjustment
    return np.clip(result, 0.001, 0.999).astype(np.float32)


def _sidecar_value(
    truth: np.ndarray,
    probability: np.ndarray,
    session_ids: list[str],
    lengths: list[int],
) -> dict[str, np.ndarray]:
    return {
        "y_true": truth.astype(np.uint8),
        "y_prob": probability.astype(np.float32),
        "input_active": np.ones(len(truth), dtype=np.uint8),
        "session_lengths": np.asarray(lengths, dtype=np.int64),
        "session_ids": np.asarray(session_ids),
    }


def _score_sidecar(
    truth: np.ndarray,
    probability: np.ndarray,
    session_ids: list[str],
    lengths: list[int],
) -> dict[str, object]:
    return _score_run(
        {
            "truth": truth.astype(bool),
            "probability": probability.astype(np.float64),
            "active": np.ones(len(truth), dtype=bool),
            "lengths": np.asarray(lengths, dtype=np.int64),
            "session_ids": session_ids,
        }
    )


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str]:
    repo = tmp_path
    subprocess.run(["git", "init", "-q", repo], check=True)
    subprocess.run(
        ["git", "-C", repo, "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", repo, "config", "user.name", "Test"], check=True
    )

    base_ids = [f"y4nQHqYSObI__r{index:03d}" for index in range(8, 16)]
    stream_ids = [f"{value}__stream000" for value in base_ids]
    stream_lengths = [4] * 8
    frames = sum(stream_lengths)
    truth = _truth()
    truth_sha = scorer._canonical_array_sha256(truth)
    monkeypatch.setattr(scorer, "Y4N_BASE_SESSION_IDS", base_ids)
    monkeypatch.setattr(scorer, "Y4N_STREAM_IDS", stream_ids)
    monkeypatch.setattr(scorer, "Y4N_STREAM_LENGTHS", stream_lengths)
    monkeypatch.setattr(scorer, "Y4N_FRAMES", frames)
    monkeypatch.setattr(scorer, "Y4N_TRUTH_SHA256", truth_sha)
    monkeypatch.setattr(blend_evaluator, "Y4N_FRAMES", frames)
    monkeypatch.setattr(blend_evaluator, "Y4N_TRUTH_SHA256", truth_sha)

    contract = json.loads(
        (SOURCE_REPO / scorer.CONTRACT_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    contract["evaluation_contract"][scorer.Y4N_SURFACE]["active_rows"] = frames
    contract["evaluation_contract"][scorer.Y4N_SURFACE]["truth_sha256"] = truth_sha
    contract_path = repo / scorer.CONTRACT_RELATIVE_PATH
    _write_json(contract_path, contract)
    template_source = SOURCE_REPO / blend_evaluator.TEMPLATE_RELATIVE_PATH
    template_path = repo / blend_evaluator.TEMPLATE_RELATIVE_PATH
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_bytes(template_source.read_bytes())

    results = repo / "results" / "idm"
    results.mkdir(parents=True)
    reference_probability = _probability(truth)
    all_base_ids = [f"y4nQHqYSObI__r{index:03d}" for index in range(16)]
    all_stream_ids = [f"{value}__stream000" for value in all_base_ids]
    all_truth = np.concatenate([truth, truth])
    all_probability = np.concatenate([reference_probability, reference_probability])
    pure_sidecar = results / f"{scorer.REFERENCE_RUN_ID}_final_nitrogen_val_preds.npz"
    np.savez_compressed(
        pure_sidecar,
        **_sidecar_value(all_truth, all_probability, all_stream_ids, [4] * 16),
    )
    pure_report = results / f"{scorer.REFERENCE_RUN_ID}_final_nitrogen_val.json"
    _write_json(
        pure_report,
        {
            "run": f"/ephemeral/results/idm/{scorer.REFERENCE_RUN_ID}",
            "weights": "final",
            "label_kind": "mapped_foreign_nitrogen",
            "sessions": all_base_ids,
            "all_frames": {"n": len(all_truth)},
            "input_active_only": {"n": len(all_truth)},
        },
    )
    pure_score = _score_sidecar(
        truth, reference_probability, stream_ids, stream_lengths
    )
    pure_run = {
        "run_id": scorer.REFERENCE_RUN_ID,
        "study_role": scorer.PURE_N_ROLE,
        "evaluation_weights": "final",
        "valid": True,
        "checkpoint_sha256": contract["reference"]["checkpoint_sha256"],
        "evaluation_report_path": pure_report.relative_to(repo).as_posix(),
        "evaluation_report_sha256": _sha256(pure_report),
        "prediction_sidecar_path": pure_sidecar.relative_to(repo).as_posix(),
        "prediction_sidecar_sha256": _sha256(pure_sidecar),
        "metrics": pure_score["metrics"],
        "baselines": pure_score["baselines"],
    }
    pure_decision = {
        "schema_version": 1,
        "study_id": scorer.PURE_N_DECISION_STUDY_ID,
        "evaluation_population": {
            "session_ids": stream_ids,
            "active_frames": frames,
            "truth_sha256": truth_sha,
            "oracle_metrics_used": False,
            "calibration_used": False,
            "b1_used": False,
        },
        "runs": {scorer.REFERENCE_RUN_ID: pure_run},
        "decision": {"decision_record_sha256": None},
    }
    pure_decision["decision"]["decision_record_sha256"] = _canonical_sha256(
        pure_decision
    )
    _write_json(repo / scorer.PURE_N_DECISION_RELATIVE_PATH, pure_decision)

    contract_commit = _commit(repo, "fixture preregistration and reference")
    contract_sha = _sha256(contract_path)
    for index, arm_name in enumerate(scorer.ARM_SPECS):
        run_id = scorer.ARM_SPECS[arm_name]["run_id"]
        run_dir = results / run_id
        run_dir.mkdir()
        config_path = run_dir / "config.json"
        checkpoint_path = run_dir / "model.pt"
        run_meta_path = run_dir / "run_meta.json"
        sampling_path = run_dir / "source_sampling_receipt.json"
        log_path = run_dir / "log.jsonl"
        _write_json(config_path, {"run_id": run_id, "seed": 0})
        checkpoint_path.write_bytes(f"checkpoint-{arm_name}".encode())
        sampling = _sampling(contract, arm_name)
        _write_json(sampling_path, sampling)
        _write_json(
            run_meta_path,
            {"seed": 0, "initialized_from": None, "source_sampling": sampling},
        )
        bce = {key: 0.4 + 0.01 * column for column, key in enumerate(KEY_ORDER)}
        log_path.write_text(
            json.dumps({"step": 0, "train_bce_per_key": None})
            + "\n"
            + json.dumps(
                {
                    "step": 14_265,
                    "train_bce_per_key": bce,
                    "val_bce_per_key": {key: value + 0.1 for key, value in bce.items()},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        probability = _probability(truth, adjustment=index * 1e-4)
        sidecar_path = results / f"{run_id}_final_y4n_later8_fixed_preds.npz"
        np.savez_compressed(
            sidecar_path,
            **_sidecar_value(truth, probability, stream_ids, stream_lengths),
        )
        run_receipt = {
            "arm": arm_name,
            "config_sha256": _sha256(config_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "run_meta_sha256": _sha256(run_meta_path),
            "source_sampling_receipt_sha256": _sha256(sampling_path),
            "training_log_sha256": _sha256(log_path),
            "checkpoint_steps": 14_265,
            "best_val_step": 14_265,
            "selected_final_tensors_identical": True,
            "parameter_count": 25_719_815,
            "evaluation_weights": "final_state_dict",
            "initialization": "from_scratch",
            "source_sampling": sampling,
        }
        fixed = fixed_metric_report(
            truth, probability, np.ones(frames, dtype=bool), stream_lengths
        )
        report_path = results / f"{run_id}_final_y4n_later8_fixed.json"
        _write_json(
            report_path,
            {
                "schema_version": scorer.ARM_REPORT_SCHEMA_VERSION,
                "study_id": scorer.STUDY_ID,
                "arm": arm_name,
                "run_id": run_id,
                "surface": scorer.Y4N_SURFACE,
                "weights": "final",
                "label_kind": "mapped_foreign_nitrogen",
                "sessions": base_ids,
                "support": {
                    "all_frames": frames,
                    "input_active_frames": frames,
                    "streams": 8,
                    "session_ids": stream_ids,
                    "stream_lengths": stream_lengths,
                    "truth_sha256": truth_sha,
                    "truth_hash_includes_shape": True,
                    "finite_aligned_arrays": True,
                },
                "fixed_metrics": fixed,
                "evaluation_policy": {
                    "raw_sigmoid_probabilities": True,
                    "fixed_state_threshold": 0.5,
                    "fixed_event_threshold": 0.5,
                    "threshold_parameters_fitted": False,
                    "calibration_parameters_fitted": False,
                    "checkpoint_selected_on_this_surface": False,
                    "sealed_untouched_session_accessed": False,
                },
                "contract": {
                    "path": f"/ephemeral/madeleine/{scorer.CONTRACT_RELATIVE_PATH}",
                    "sha256": contract_sha,
                    "commit": contract_commit,
                },
                "run_receipt": run_receipt,
                "prediction_sidecar": {
                    "path": f"/ephemeral/results/idm/{sidecar_path.name}",
                    "sha256": _sha256(sidecar_path),
                },
            },
        )
        marker_path = results / f".{run_id}_final_y4n_later8_fixed_done.json"
        _write_json(
            marker_path,
            {
                "schema_version": scorer.MARKER_SCHEMA_VERSION,
                "status": "complete",
                "study_id": scorer.STUDY_ID,
                "arm": arm_name,
                "run_id": run_id,
                "surface": scorer.Y4N_SURFACE,
                "weights": "final",
                "contract_sha256": contract_sha,
                "checkpoint_sha256": _sha256(checkpoint_path),
                "run_meta_sha256": _sha256(run_meta_path),
                "source_sampling_receipt_sha256": _sha256(sampling_path),
                "report_sha256": _sha256(report_path),
                "sidecar_sha256": _sha256(sidecar_path),
            },
        )
        wrapper_path = results / f".{run_id}_train_and_y4n_done.json"
        _write_json(
            wrapper_path,
            {
                "schema_version": scorer.WRAPPER_MARKER_SCHEMA_VERSION,
                "status": "complete",
                "arm": arm_name,
                "run_id": run_id,
                "contract_sha256": contract_sha,
                "b1_accessed": False,
                "sealed_untouched_session_accessed": False,
                "artifacts": {
                    "report": {
                        "path": f"/ephemeral/results/idm/{report_path.name}",
                        "sha256": _sha256(report_path),
                    },
                    "sidecar": {
                        "path": f"/ephemeral/results/idm/{sidecar_path.name}",
                        "sha256": _sha256(sidecar_path),
                    },
                    "marker": {
                        "path": f"/ephemeral/results/idm/{marker_path.name}",
                        "sha256": _sha256(marker_path),
                    },
                },
            },
        )
    return repo, contract_commit


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    repo, commit = _fixture(tmp_path, monkeypatch)
    return scorer.build_decision(
        repo=repo,
        contract_path=repo / scorer.CONTRACT_RELATIVE_PATH,
        contract_commit=commit,
        pure_n_decision_path=repo / scorer.PURE_N_DECISION_RELATIVE_PATH,
    )


def test_build_is_deterministic_and_mechanically_scores_three_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _build(tmp_path, monkeypatch)
    repo = tmp_path
    commit = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    second = scorer.build_decision(
        repo=repo,
        contract_path=repo / scorer.CONTRACT_RELATIVE_PATH,
        contract_commit=commit,
        pure_n_decision_path=repo / scorer.PURE_N_DECISION_RELATIVE_PATH,
    )

    assert first == second
    assert first["schema_version"] == scorer.COMPARISON_SCHEMA_VERSION
    assert len(first["runs"]) == 3
    assert first["evaluation_population"]["b1_used"] is False
    assert first["evaluation_population"]["fitted_thresholds_used"] is False
    assert first["decision"]["comparison_frozen_before_b1"] is True
    assert first["decision"]["outcome"] == "no_arm_eligible"
    assert first["decision"]["winner_arm"] is None
    assert first["shared_baselines"]["shuffled_events"]["seeds"] == list(
        range(10)
    )
    digest = first["decision"]["decision_record_sha256"]
    unhashed = copy.deepcopy(first)
    unhashed["decision"]["decision_record_sha256"] = None
    assert digest == _canonical_sha256(unhashed)
    for arm_name in scorer.ARM_SPECS:
        run_id = scorer.ARM_SPECS[arm_name]["run_id"]
        receipt = first["runs"][run_id]["memorization_receipts"]
        assert receipt["sampling_exposure"]["local"]["effective_pool_passes"] > 143
        assert receipt["required_receipt_status"]["complete"] is False
        assert "per-source final train BCE" in receipt[
            "required_receipt_status"
        ]["not_emitted_by_training_artifacts"]


def test_candidate_rule_requires_all_three_gates() -> None:
    def run(macro: float, per_key: list[float], event: float) -> dict[str, object]:
        return {
            "metrics": {
                "macro_ap": macro,
                "per_key_ap": dict(zip(KEY_ORDER, per_key, strict=True)),
                "segment_bounded_combined_event_f1_fixed_0_5": {
                    "plus_minus_2": event
                },
            }
        }

    reference = run(0.20, [0.1] * 7, 0.04)
    passing = run(0.206, [0.11] * 4 + [0.09] * 3, 0.036)
    result = scorer._candidate_rule(reference, passing)
    assert result["eligible"] is True
    assert result["improved_key_count"] == 4

    failing = run(0.206, [0.11] * 4 + [0.09] * 3, 0.034)
    assert scorer._candidate_rule(reference, failing)["eligible"] is False


def test_shuffled_event_anchor_is_deterministic_and_segment_bounded() -> None:
    truth = np.zeros((12, len(KEY_ORDER)), dtype=np.uint8)
    truth[1:3] = 1
    truth[4:6] = 1
    truth[9:11] = 1

    first = scorer.segment_bounded_shuffled_event_baseline(
        truth, [4, 4, 4], seeds=[0, 1, 2]
    )
    second = scorer.segment_bounded_shuffled_event_baseline(
        truth, [4, 4, 4], seeds=[0, 1, 2]
    )

    assert first == second
    assert "within that stream" in first["policy"]
    assert set(first["macro_mean"]) == {"exact", "plus_minus_2"}

    # Each one-frame stream has only one possible placement.  A global
    # shuffle could move these onsets across the join; the stream-bounded
    # implementation must reproduce them exactly for every seed.
    boundary_truth = np.zeros((2, len(KEY_ORDER)), dtype=np.uint8)
    boundary_truth[0] = 1
    forced = scorer.segment_bounded_shuffled_event_baseline(
        boundary_truth, [1, 1], seeds=[0, 1, 2]
    )
    assert forced["macro_mean"]["exact"] == 1.0


def test_tampered_fixed_report_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, commit = _fixture(tmp_path, monkeypatch)
    run_id = scorer.ARM_SPECS["NL_90_10"]["run_id"]
    report_path = repo / "results" / "idm" / f"{run_id}_final_y4n_later8_fixed.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["fixed_metrics"]["input_active_only"]["macro_ap"] += 0.01
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="fixed metrics"):
        scorer.build_decision(
            repo=repo,
            contract_path=repo / scorer.CONTRACT_RELATIVE_PATH,
            contract_commit=commit,
            pure_n_decision_path=repo / scorer.PURE_N_DECISION_RELATIVE_PATH,
        )


def test_write_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "decision.json"
    scorer.write_decision(output, {"ok": True})
    original = output.read_bytes()

    with pytest.raises(ValueError, match="refusing to overwrite"):
        scorer.write_decision(output, {"ok": False})
    assert output.read_bytes() == original
