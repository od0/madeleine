from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import experiments.validate_provisional_blend_memorization as validator


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _diagnostic_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    module = repo / validator.DIAGNOSTIC_RELATIVE_PATH
    module.parent.mkdir(parents=True)
    module.write_text("DIAGNOSTIC = 'committed'\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", validator.DIAGNOSTIC_RELATIVE_PATH.as_posix())
    _git(repo, "commit", "-qm", "diagnostic")
    commit = _git(repo, "rev-parse", "HEAD")
    digest = hashlib.sha256(module.read_bytes()).hexdigest()
    return repo, commit, digest


def _tiny_source_specs(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    memberships = {
        "nitrogen": ["n1", "n2"],
        "local": ["l1"],
        "local_val_a": [validator.LOCAL_VAL_A_ID],
    }
    specs = copy.deepcopy(validator.SOURCE_SPECS)
    for source, ids in memberships.items():
        specs[source]["sessions"] = len(ids)
        specs[source]["membership_sha256"] = validator._line_list_sha256(ids)
    specs["nitrogen"]["segment_items"] = 3
    specs["local"]["segment_items"] = 1
    specs["local_val_a"]["segment_items"] = 1
    monkeypatch.setattr(validator, "SOURCE_SPECS", specs)
    return memberships


def _metrics(
    *, bce: float = 0.4, ap: float = 0.5, f1: float = 0.5
) -> dict[str, object]:
    values = {
        "unweighted_bce": bce,
        "average_precision": ap,
        "state_f1_fixed_0_5": f1,
        "prevalence": 0.25,
        "predicted_positive_rate_fixed_0_5": 0.25,
    }
    return {
        name: {
            "per_key": {key: value for key in validator.KEY_ORDER},
            "macro": value,
        }
        for name, value in values.items()
    }


def _sampling(arm: str) -> dict[str, object]:
    spec = validator.ARM_SPECS[arm]
    return {
        "format_version": "madeleine.source-balanced-batch.v1",
        "seed": 0,
        "cycle_steps": 5,
        "cycle_items": 80,
        "batch_items": 16,
        "scheduled_steps": validator.FINAL_STEP,
        "actual_steps": validator.FINAL_STEP,
        "step_cycle": list(spec["step_cycle"]),
        "complete": True,
        "sources": {
            source: validator._expected_sampling_source(
                source, int(spec["draws"][source])
            )
            for source in spec["sources"]
        },
    }


def _shard_receipt(source: str, ids: list[str]) -> dict[str, object]:
    shards = {
        session_id: {
            "bytes": 100 + index,
            "sha256": f"{index + 1:064x}",
        }
        for index, session_id in enumerate(ids)
    }
    return {
        "sessions": len(ids),
        "total_bytes": sum(row["bytes"] for row in shards.values()),
        "membership_sha256": validator.SOURCE_SPECS[source][
            "membership_sha256"
        ],
        "shard_set_sha256": validator._canonical_json_sha256(shards),
        "shards": shards,
    }


def _support(source: str, ids: list[str]) -> dict[str, object]:
    items = int(validator.SOURCE_SPECS[source]["segment_items"])
    counts = {session_id: 0 for session_id in ids}
    for index in range(items):
        counts[ids[index % len(ids)]] += 1
    frames = items * validator.SEGMENT_WINDOWS
    return {
        "sessions": len(ids),
        "sessions_with_complete_segments": sum(value > 0 for value in counts.values()),
        "sessions_without_complete_segments": sum(value == 0 for value in counts.values()),
        "segment_items": items,
        "segment_windows": validator.SEGMENT_WINDOWS,
        "target_frames": frames,
        "binary_labels": frames * len(validator.KEY_ORDER),
        "session_segment_items": counts,
        "session_segment_items_sha256": validator._canonical_json_sha256(counts),
        "truth_sha256": "a" * 64,
        "probability_sha256": "b" * 64,
    }


def _source_contract(source: str) -> dict[str, object]:
    spec = validator.SOURCE_SPECS[source]
    if source == "nitrogen":
        return {
            "tier": spec["tier"],
            "sessions": spec["sessions"],
            "session_list_sha256": spec["membership_sha256"],
        }
    if source == "local":
        return {
            "tier": spec["tier"],
            "complete_segment_items": spec["segment_items"],
            "train_sessions_sha256": spec["membership_sha256"],
            "val_sessions_sha256": validator.SOURCE_SPECS["local_val_a"][
                "membership_sha256"
            ],
            "forbidden_generation": "/ephemeral/data/own_features",
        }
    raise AssertionError(source)


def _surface(
    source: str,
    ids: list[str],
    sampling: dict[str, object],
    *,
    metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    if source == "local_val_a":
        return {
            "support": _support(source, ids),
            "metrics": metrics or _metrics(),
            "role": "corrected_local_val_a_complete_segment_pool",
            "tier": validator.SOURCE_SPECS[source]["tier"],
            "sampling_receipt": {
                "scheduled_draws": 0,
                "actual_draws": 0,
                "repeat_draws": 0,
                "note": "held-out local development surface; never sampled",
            },
            "source_rgb_shard_sha256": {
                validator.LOCAL_VAL_A_ID: validator.LOCAL_VAL_A_RGB_SHA256
            },
            "shard_receipt": _shard_receipt(source, ids),
        }
    return {
        "support": _support(source, ids),
        "metrics": metrics or _metrics(),
        "role": "unique_complete_training_segment_pool",
        "tier": validator.SOURCE_SPECS[source]["tier"],
        "sampling_receipt": sampling["sources"][source],
        "source_contract": _source_contract(source),
        "shard_receipt": _shard_receipt(source, ids),
    }


def _local_gap(local: dict[str, object], val: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    conventions = {
        "unweighted_bce": "val_a_minus_train",
        "average_precision": "train_minus_val_a",
        "state_f1_fixed_0_5": "train_minus_val_a",
    }
    for metric, direction in conventions.items():
        train = local["metrics"][metric]
        validation = val["metrics"][metric]
        values = {
            key: (
                validation["per_key"][key] - train["per_key"][key]
                if direction == "val_a_minus_train"
                else train["per_key"][key] - validation["per_key"][key]
            )
            for key in validator.KEY_ORDER
        }
        macro = (
            validation["macro"] - train["macro"]
            if direction == "val_a_minus_train"
            else train["macro"] - validation["macro"]
        )
        result[metric] = {"direction": direction, "per_key": values, "macro": macro}
    return result


def _report(
    commit: str,
    diagnostic_sha256: str,
    memberships: dict[str, list[str]],
) -> dict[str, object]:
    arm = "NL_90_10"
    spec = validator.ARM_SPECS[arm]
    sampling = _sampling(arm)
    surfaces = {
        source: _surface(source, memberships[source], sampling)
        for source in spec["sources"]
    }
    surfaces["local_val_a"] = _surface(
        "local_val_a", memberships["local_val_a"], sampling
    )
    generated_names = {
        "train_nl_90_10_sessions.txt",
        "config_nl_90_10.json",
        "train_nlw_70_20_10_sessions.txt",
        "config_nlw_70_20_10.json",
        "val_sessions.txt",
        "later_eight_sessions.txt",
        "local_val_a_sessions.txt",
        "hardlink_inventory.jsonl",
        "shard_hashes.json",
    }
    inventory_sha = "c" * 64
    return {
        "schema_version": validator.REPORT_SCHEMA_VERSION,
        "study_id": validator.STUDY_ID,
        "arm": arm,
        "run_id": spec["run_id"],
        "surface": validator.SURFACE,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "weights": "final_state_dict",
        "contract": {
            "path": f"/repo/{validator.CONTRACT_RELATIVE_PATH.as_posix()}",
            "sha256": validator.CONTRACT_SHA256,
            "commit": validator.CONTRACT_COMMIT,
        },
        "diagnostic_source": {
            "git_commit": commit,
            "relative_path": validator.DIAGNOSTIC_RELATIVE_PATH.as_posix(),
            "sha256": diagnostic_sha256,
        },
        "run_receipt": {
            "arm": arm,
            "config_sha256": spec["config_sha256"],
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "run_meta_sha256": spec["run_meta_sha256"],
            "source_sampling_receipt_sha256": spec[
                "source_sampling_receipt_sha256"
            ],
            "training_log_sha256": spec["training_log_sha256"],
            "checkpoint_steps": validator.FINAL_STEP,
            "best_val_step": validator.FINAL_STEP,
            "selected_final_tensors_identical": True,
            "parameter_count": validator.PARAMETER_COUNT,
            "evaluation_weights": "final_state_dict",
            "initialization": "from_scratch",
            "positive_weight": validator.POSITIVE_WEIGHT,
            "source_sampling": sampling,
            "inference_source": {
                "implementation_git_commit": validator.TRAINING_IMPLEMENTATION_COMMIT,
                "verified_files_sha256": validator.TRAINING_SOURCE_HASHES,
            },
        },
        "feature_view": {
            "path": "/data/view",
            "receipt_path": "/data/view/blend_feature_view_receipt.json",
            "receipt_sha256": "d" * 64,
            "hardlink_inventory_path": "/data/view/hardlink_inventory.jsonl",
            "hardlink_inventory_sha256": inventory_sha,
            "hardlink_inventory_rows": 3_140,
            "generated_files": {
                name: inventory_sha if name == "hardlink_inventory.jsonl" else "e" * 64
                for name in generated_names
            },
            "source_sessions": {
                source: memberships[source] for source in spec["sources"]
            },
            "local_val_a_sessions": memberships["local_val_a"],
        },
        "scope_guard": {
            "accessed": [
                "unique complete training segment pools",
                "corrected own-v3 local val-A complete segment pool",
            ],
            "not_accessed": ["B1", "val-B", "mapped-y4n", "sealed untouched test"],
            "known_forbidden_session_ids": sorted(validator.FORBIDDEN_SESSION_IDS),
            "forbidden_session_ids_accessed": [],
        },
        "method": {
            "segment_pool": (
                "the exact SegmentSessionDataset construction used for training; "
                "full 96-window items only; each unique item scored once"
            ),
            "loss": (
                "unweighted binary cross-entropy over unique target frames; "
                "training class, transition, and draw-repeat weights excluded"
            ),
            "average_precision": "raw final sigmoid probabilities; threshold-free",
            "state_f1": "raw final sigmoid probabilities at fixed threshold 0.5",
            "checkpoint_selection": "none; final_state_dict only",
            "calibration": "none",
        },
        "source_sampling_receipt": sampling,
        "surfaces": surfaces,
        "local_generalization_gap": _local_gap(
            surfaces["local"], surfaces["local_val_a"]
        ),
    }


def _write_pair(
    root: Path,
    report: dict[str, object],
    *,
    marker_changes: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    run_id = report["run_id"]
    output = root / "results"
    output.mkdir(exist_ok=True)
    report_path = output / f"{run_id}_final_memorization.json"
    marker_path = output / f".{run_id}_final_memorization_done.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    run = report["run_receipt"]
    diagnostic = report["diagnostic_source"]
    feature_view = report["feature_view"]
    marker = {
        "schema_version": validator.MARKER_SCHEMA_VERSION,
        "status": "complete",
        "study_id": report["study_id"],
        "arm": report["arm"],
        "run_id": run_id,
        "surface": report["surface"],
        "weights": report["weights"],
        "contract_sha256": report["contract"]["sha256"],
        "training_implementation_git_commit": run["inference_source"][
            "implementation_git_commit"
        ],
        "diagnostic_git_commit": diagnostic["git_commit"],
        "diagnostic_module_sha256": diagnostic["sha256"],
        "config_sha256": run["config_sha256"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "run_meta_sha256": run["run_meta_sha256"],
        "training_log_sha256": run["training_log_sha256"],
        "source_sampling_receipt_sha256": run[
            "source_sampling_receipt_sha256"
        ],
        "selected_final_tensors_identical": run[
            "selected_final_tensors_identical"
        ],
        "feature_view_receipt_sha256": feature_view["receipt_sha256"],
        "forbidden_surfaces_accessed": False,
        "report": str(report_path.resolve()),
        "report_sha256": validator.sha256_file(report_path),
    }
    marker.update(marker_changes or {})
    marker_path.write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
    return report_path, marker_path


@pytest.fixture
def valid_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str, dict[str, object], dict[str, list[str]]]:
    memberships = _tiny_source_specs(monkeypatch)
    repo, commit, digest = _diagnostic_repo(tmp_path)
    monkeypatch.setattr(
        validator,
        "_load_contract_sources",
        lambda unused_repo: {
            source: _source_contract(source) for source in ("nitrogen", "local")
        },
    )
    return repo, commit, _report(commit, digest, memberships), memberships


def test_validates_complete_read_only_artifact_pair(
    tmp_path: Path,
    valid_case: tuple[Path, str, dict[str, object], dict[str, list[str]]],
) -> None:
    repo, commit, report, _ = valid_case
    report_path, marker_path = _write_pair(tmp_path, report)

    receipt = validator.validate_artifacts(
        repo,
        report_path,
        marker_path,
        arm="NL_90_10",
        diagnostic_commit=commit,
    )

    assert receipt["status"] == "valid"
    assert receipt["report_sha256"] == validator.sha256_file(report_path)
    assert receipt["forbidden_surfaces_accessed"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda report: report["surfaces"]["local"]["metrics"][
                "average_precision"
            ]["per_key"].pop("dash"),
            "exactly seven keys",
        ),
        (
            lambda report: report["surfaces"]["nitrogen"]["support"].__setitem__(
                "target_frames", 7
            ),
            "target_frames changed",
        ),
        (
            lambda report: report["source_sampling_receipt"]["sources"][
                "local"
            ].__setitem__("repeat_draws", 0),
            "source-sampling local.repeat_draws changed",
        ),
        (
            lambda report: report["local_generalization_gap"][
                "average_precision"
            ].__setitem__("macro", 0.25),
            "macro gap arithmetic changed",
        ),
        (
            lambda report: report["scope_guard"].__setitem__(
                "forbidden_session_ids_accessed", [validator.LOCAL_VAL_A_ID]
            ),
            "forbidden-session access",
        ),
    ],
)
def test_fails_closed_on_metric_support_sampling_gap_or_scope_change(
    tmp_path: Path,
    valid_case: tuple[Path, str, dict[str, object], dict[str, list[str]]],
    mutator: object,
    message: str,
) -> None:
    repo, commit, base, _ = valid_case
    report = copy.deepcopy(base)
    mutator(report)
    report_path, marker_path = _write_pair(tmp_path, report)

    with pytest.raises(ValueError, match=message):
        validator.validate_artifacts(
            repo,
            report_path,
            marker_path,
            arm="NL_90_10",
            diagnostic_commit=commit,
        )


def test_rejects_nonfinite_metric_and_wrong_checkpoint(
    tmp_path: Path,
    valid_case: tuple[Path, str, dict[str, object], dict[str, list[str]]],
) -> None:
    repo, commit, base, _ = valid_case
    report = copy.deepcopy(base)
    report["surfaces"]["local"]["metrics"]["unweighted_bce"]["per_key"][
        "left"
    ] = float("nan")
    report_path, marker_path = _write_pair(tmp_path, report)
    with pytest.raises(ValueError, match="not finite"):
        validator.validate_artifacts(
            repo,
            report_path,
            marker_path,
            arm="NL_90_10",
            diagnostic_commit=commit,
        )

    report = copy.deepcopy(base)
    report["run_receipt"]["checkpoint_sha256"] = "0" * 64
    report_path, marker_path = _write_pair(tmp_path, report)
    with pytest.raises(ValueError, match="checkpoint_sha256 changed"):
        validator.validate_artifacts(
            repo,
            report_path,
            marker_path,
            arm="NL_90_10",
            diagnostic_commit=commit,
        )


def test_rejects_marker_hash_status_and_diagnostic_binding(
    tmp_path: Path,
    valid_case: tuple[Path, str, dict[str, object], dict[str, list[str]]],
) -> None:
    repo, commit, base, _ = valid_case
    report_path, marker_path = _write_pair(
        tmp_path, base, marker_changes={"status": "partial"}
    )
    with pytest.raises(ValueError, match="schema or status"):
        validator.validate_artifacts(
            repo,
            report_path,
            marker_path,
            arm="NL_90_10",
            diagnostic_commit=commit,
        )

    report_path, marker_path = _write_pair(
        tmp_path, base, marker_changes={"report_sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="hash differs"):
        validator.validate_artifacts(
            repo,
            report_path,
            marker_path,
            arm="NL_90_10",
            diagnostic_commit=commit,
        )

    report = copy.deepcopy(base)
    report["diagnostic_source"]["sha256"] = "0" * 64
    report_path, marker_path = _write_pair(tmp_path, report)
    with pytest.raises(ValueError, match="committed blob"):
        validator.validate_artifacts(
            repo,
            report_path,
            marker_path,
            arm="NL_90_10",
            diagnostic_commit=commit,
        )


def test_cli_exposes_no_inference_data_or_output_path() -> None:
    help_text = validator._parser().format_help()
    assert "--report" in help_text
    assert "--completion-marker" in help_text
    assert "--diagnostic-commit" in help_text
    for forbidden_option in (
        "--out",
        "--data",
        "--run",
        "--device",
        "--sessions",
        "--checkpoint",
    ):
        assert forbidden_option not in help_text
