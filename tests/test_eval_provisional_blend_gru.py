from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from data.schema import KEY_ORDER
import experiments.eval_provisional_blend_gru as evaluator


def _contract() -> dict[str, object]:
    return json.loads(
        (Path.cwd() / evaluator.CONTRACT_RELATIVE_PATH).read_text(encoding="utf-8")
    )


def test_current_contract_has_both_exact_arms_and_hard_embargo() -> None:
    contract = _contract()

    evaluator.validate_contract_value(contract)

    assert [arm["name"] for arm in contract["arms"]] == list(evaluator.ARM_SPECS)
    assert contract["sources"]["local"]["tier"] == "engine_truth_corrected_own_v3"
    assert contract["sources"]["wild_provisional"]["admitted_hours"] == 0.0
    assert contract["embargo"]["sealed_untouched_session"] == (
        evaluator.UNTOUCHED_SESSION_ID
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model_contract", "maximum_steps"), 14_264, "maximum_steps"),
        (("sources", "local", "tier"), "legacy_own", "corrected own-v3"),
        (("sources", "wild_provisional", "admitted_hours"), 1.0, "zero-admission"),
        (
            ("evaluation_contract", "oracle_thresholds_allowed"),
            True,
            "fitted thresholds",
        ),
        (("embargo", "sealed_untouched_session"), "other", "identity changed"),
    ],
)
def test_contract_rejects_decision_bearing_drift(
    path: tuple[str, ...], value: object, message: str
) -> None:
    contract = _contract()
    target: dict[str, object] = contract
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        evaluator.validate_contract_value(contract)


def _source_ids(contract: dict[str, object]) -> dict[str, list[str]]:
    nitrogen = [f"n_{index:04d}" for index in range(1062)]
    local = [f"own_v3_{index}" for index in range(3)]
    wild = [f"wild_{index:04d}" for index in range(2058)]
    contract["sources"]["nitrogen"]["session_list_sha256"] = (  # type: ignore[index]
        evaluator._line_list_sha256(nitrogen)
    )
    contract["sources"]["local"]["train_sessions_sha256"] = (  # type: ignore[index]
        evaluator._line_list_sha256(local)
    )
    contract["sources"]["wild_provisional"]["session_list_sha256"] = (  # type: ignore[index]
        evaluator._line_list_sha256(wild)
    )
    return {"nitrogen": nitrogen, "local": local, "wild_provisional": wild}


def _sampling_receipt(
    contract: dict[str, object], arm_name: str, source_ids: dict[str, list[str]]
) -> dict[str, object]:
    spec = evaluator.ARM_SPECS[arm_name]
    segment_items = {
        "nitrogen": 300_000,
        "local": contract["sources"]["local"]["complete_segment_items"],  # type: ignore[index]
        "wild_provisional": contract["sources"]["wild_provisional"][  # type: ignore[index]
            "complete_segment_items"
        ],
    }
    sources: dict[str, object] = {}
    for name, draws in spec["draws"].items():
        pool = int(segment_items[name])
        unique = min(pool, draws)
        sources[name] = {
            "session_count": len(source_ids[name]),
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
        "scheduled_steps": evaluator.EXPECTED_FINAL_STEP,
        "actual_steps": evaluator.EXPECTED_FINAL_STEP,
        "step_cycle": [
            {key: row[key] for key in sorted(row)} for row in spec["cycle"]
        ],
        "complete": True,
        "sources": sources,
    }


def test_sampling_receipt_binds_draws_repeats_and_local_pool() -> None:
    contract = _contract()
    ids = _source_ids(contract)
    receipt = _sampling_receipt(contract, "NLW_70_20_10", ids)

    observed = evaluator._sampling_receipt(
        receipt, contract, "NLW_70_20_10"
    )

    assert observed["sources"]["local"]["repeat_draws"] == 22_665
    assert observed["sources"]["wild_provisional"]["repeat_draws"] == 4_081
    changed = copy.deepcopy(receipt)
    changed["sources"]["local"]["actual_draws"] -= 1
    with pytest.raises(ValueError, match="local.actual_draws"):
        evaluator._sampling_receipt(changed, contract, "NLW_70_20_10")


class _TinyGRU(torch.nn.Module):
    def __init__(self, config: object) -> None:
        super().__init__()
        self.temporal_arch = "gru"
        self.temporal = torch.nn.GRUCell(1, 1)


def _write_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    initialized_from: str | None = None,
    change_draws: bool = False,
) -> tuple[Path, dict[str, object]]:
    contract = _contract()
    ids = _source_ids(contract)
    arm_name = "NL_90_10"
    spec = evaluator.ARM_SPECS[arm_name]
    config = {
        "source_sampling": {
            "sources": {"nitrogen": ids["nitrogen"], "local": ids["local"]}
        }
    }
    receipt = _sampling_receipt(contract, arm_name, ids)
    if change_draws:
        receipt["sources"]["local"]["actual_draws"] -= 1
    run = tmp_path / spec["run_id"]
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")
    (run / "source_sampling_receipt.json").write_text(
        json.dumps(receipt) + "\n", encoding="utf-8"
    )
    weights = {
        key: float(value)
        for key, value in zip(
            KEY_ORDER,
            contract["model_contract"]["positive_weight"],
            strict=True,
        )
    }
    run_meta = {
        "git": f"{'a' * 40}-declared",
        "seed": 0,
        "config": config,
        "split": {
            "train": [*ids["nitrogen"], *ids["local"]],
            "val": [f"y4nQHqYSObI__r{index:03d}" for index in range(16)],
        },
        "initialized_from": initialized_from,
        "positive_weight": weights,
        "source_sampling": receipt,
    }
    (run / "run_meta.json").write_text(json.dumps(run_meta) + "\n", encoding="utf-8")
    (run / "log.jsonl").write_text(
        json.dumps({"step": 0})
        + "\n"
        + json.dumps({"step": evaluator.EXPECTED_FINAL_STEP})
        + "\n",
        encoding="utf-8",
    )
    model = _TinyGRU(config)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    checkpoint = {
        "config": config,
        "key_order": list(KEY_ORDER),
        "model_state_dict": state,
        "final_state_dict": state,
        "steps": evaluator.EXPECTED_FINAL_STEP,
        "best_val_step": evaluator.EXPECTED_FINAL_STEP,
        "best_val_mean_bce": 0.5,
        "initialized_from": initialized_from,
        "positive_weight": [weights[key] for key in KEY_ORDER],
        "source_sampling_receipt": receipt,
    }
    torch.save(checkpoint, run / "model.pt")
    monkeypatch.setattr(
        evaluator, "expected_run_config", lambda repo, c, a, observed: observed
    )
    monkeypatch.setattr(evaluator, "BadelineIDM", _TinyGRU)
    monkeypatch.setattr(
        evaluator,
        "EXPECTED_TRAINABLE_PARAMETERS",
        sum(parameter.numel() for parameter in model.parameters()),
    )
    monkeypatch.setattr(
        evaluator,
        "_verify_inference_source",
        lambda repo, commit: {"implementation_git_commit": commit},
    )
    return run, contract


def test_run_validation_requires_final_from_scratch_checkpoint_and_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, contract = _write_run(tmp_path, monkeypatch)

    _, model, receipt = evaluator.validate_run(
        Path.cwd(),
        run,
        evaluator.ARM_SPECS["NL_90_10"]["run_id"],
        contract,
        "NL_90_10",
    )

    assert model.temporal_arch == "gru"
    assert receipt["checkpoint_steps"] == evaluator.EXPECTED_FINAL_STEP
    assert receipt["initialization"] == "from_scratch"
    assert receipt["source_sampling"]["complete"] is True
    assert len(receipt["checkpoint_sha256"]) == 64


def test_run_validation_rejects_initialization_and_incomplete_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialized, contract = _write_run(
        tmp_path / "initialized",
        monkeypatch,
        initialized_from="reference.pt",
    )
    with pytest.raises(ValueError, match="not from scratch"):
        evaluator.validate_run(
            Path.cwd(),
            initialized,
            evaluator.ARM_SPECS["NL_90_10"]["run_id"],
            contract,
            "NL_90_10",
        )

    bad, contract = _write_run(
        tmp_path / "sampling",
        monkeypatch,
        change_draws=True,
    )
    with pytest.raises(ValueError, match="local.actual_draws"):
        evaluator.validate_run(
            Path.cwd(),
            bad,
            evaluator.ARM_SPECS["NL_90_10"]["run_id"],
            contract,
            "NL_90_10",
        )


def _comparison_value(arm_name: str) -> tuple[dict[str, object], dict[str, str]]:
    release = {
        "checkpoint_sha256": "a" * 64,
        "report_sha256": "b" * 64,
        "sidecar_sha256": "c" * 64,
        "completion_marker_sha256": "d" * 64,
    }
    runs = {}
    for run_id in (
        evaluator.REFERENCE_RUN_ID,
        evaluator.ARM_SPECS["NL_90_10"]["run_id"],
        evaluator.ARM_SPECS["NLW_70_20_10"]["run_id"],
    ):
        runs[run_id] = {
            "complete": True,
            "weights": "final",
            "checkpoint_sha256": release["checkpoint_sha256"],
            "report_sha256": release["report_sha256"],
            "sidecar_sha256": release["sidecar_sha256"],
            "completion_marker_sha256": release["completion_marker_sha256"],
        }
    value = {
        "schema_version": evaluator.COMPARISON_SCHEMA_VERSION,
        "study_id": evaluator.STUDY_ID,
        "evaluation_population": {
            "surface": evaluator.Y4N_SURFACE,
            "session_ids": evaluator.Y4N_STREAM_IDS,
            "active_frames": evaluator.Y4N_FRAMES,
            "truth_sha256": evaluator.Y4N_TRUTH_SHA256,
            "fixed_threshold": 0.5,
            "fitted_thresholds_used": False,
            "fitted_calibration_used": False,
            "b1_used": False,
        },
        "decision": {
            "comparison_frozen_before_b1": True,
            "all_runs_consulted": [
                evaluator.REFERENCE_RUN_ID,
                evaluator.ARM_SPECS["NL_90_10"]["run_id"],
                evaluator.ARM_SPECS["NLW_70_20_10"]["run_id"],
            ],
        },
        "runs": runs,
    }
    return value, release


def test_b1_gate_requires_complete_three_run_y4n_comparison() -> None:
    value, release = _comparison_value("NL_90_10")

    evaluator.validate_comparison_value(
        value, arm_name="NL_90_10", arm_release=release
    )

    value["evaluation_population"]["b1_used"] = True
    with pytest.raises(ValueError, match="population.b1_used"):
        evaluator.validate_comparison_value(
            value, arm_name="NL_90_10", arm_release=release
        )


def test_output_publication_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    sidecar = tmp_path / "preds.npz"
    marker = tmp_path / ".done"
    temporary_report = evaluator._temporary_path(report)
    temporary_sidecar = evaluator._temporary_path(sidecar, npz=True)
    temporary_report.write_text("{}\n", encoding="utf-8")
    temporary_sidecar.write_bytes(b"sidecar")
    marker_value = {
        "status": "complete",
        "report_sha256": hashlib.sha256(b"{}\n").hexdigest(),
        "sidecar_sha256": hashlib.sha256(b"sidecar").hexdigest(),
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
    with pytest.raises(ValueError, match="refusing to overwrite"):
        evaluator._publish_result(
            report_path=report,
            temporary_report=temporary_report,
            sidecar_path=sidecar,
            temporary_sidecar=temporary_sidecar,
            marker_path=marker,
            marker=marker_value,
        )
