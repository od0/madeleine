"""Fixed-only evaluation for the preregistered provisional blend GRUs.

The two blend arms share one evaluator so their validation and serialization
paths cannot drift.  Mapped-y4n later-eight is always released first.  B1 is
unavailable until both arm releases and the pure-NitroGen reference have been
bound into a clean, committed comparison decision.  All decisions use raw
sigmoid probabilities at 0.5; this module has no threshold-fitting or
calibration path.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np
import torch

from badeline.model import BadelineIDM
from badeline.train import read_session_ids
from data.schema import KEY_ORDER
from experiments.eval_tcn_control_lr_b1 import (
    EXPECTED_B1_ACTIVE_FRAMES,
    EXPECTED_B1_ACTIVE_SHA256,
    EXPECTED_B1_FRAMES,
    EXPECTED_B1_STREAM_IDS,
    EXPECTED_B1_STREAMS,
    EXPECTED_B1_TRUTH_SHA256,
    _verify_inference_source,
    fixed_metric_report,
    infer_fixed_state,
    sha256_file,
    validate_b1_sidecar,
    validate_b1_surface,
)
from experiments.eval_wild_provisional_gru import (
    Y4N_BASE_SESSION_IDS,
    Y4N_FRAMES,
    Y4N_STREAM_IDS,
    Y4N_STREAM_LENGTHS,
    Y4N_TRUTH_SHA256,
    _contains_disallowed_metric_language,
    _load_sidecar_arrays,
    _publish_result,
    _refuse_existing,
    _state_dicts_identical,
    _temporary_path,
    _atomic_json,
    validate_y4n_sidecar,
)


SCHEMA_VERSION = "madeleine.provisional-blend-gru-fixed-eval.v1"
MARKER_SCHEMA_VERSION = "madeleine.provisional-blend-gru-fixed-marker.v1"
CONTRACT_SCHEMA_VERSION = "madeleine.provisional-blend-gru-decision.v1"
COMPARISON_SCHEMA_VERSION = "madeleine.provisional-blend-y4n-decision.v1"
STUDY_ID = "provisional_blend_gru_y4n_b1_s0"
CONTRACT_RELATIVE_PATH = Path(
    "experiments/configs/provisional_blend_gru_decision.json"
)
COMPARISON_RELATIVE_PATH = Path(
    "results/idm/provisional_blend_y4n_decision.json"
)
TEMPLATE_RELATIVE_PATH = Path(
    "experiments/configs/takeover_features_26m_128x3frame_full_holdout.json"
)
EXPECTED_FINAL_STEP = 14_265
EXPECTED_TRAINABLE_PARAMETERS = 25_719_815
REFERENCE_RUN_ID = "nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0"
UNTOUCHED_SESSION_ID = "rec_20260727_220000_test"
Y4N_SURFACE = "mapped_y4n_later_eight"
B1_SURFACE = "engine_truth_b1_development"
HEX_40 = re.compile(r"[0-9a-f]{40}")
HEX_64 = re.compile(r"[0-9a-f]{64}")

ARM_SPECS: dict[str, dict[str, Any]] = {
    "NL_90_10": {
        "run_id": "blend_provisional_nl90_10_92train_y4n_holdout_26m_128x3_s0",
        "mix": {"nitrogen": 90, "local": 10},
        "cycle": [
            {"nitrogen": 14, "local": 2},
            {"nitrogen": 14, "local": 2},
            {"nitrogen": 14, "local": 2},
            {"nitrogen": 15, "local": 1},
            {"nitrogen": 15, "local": 1},
        ],
        "draws": {"nitrogen": 205_416, "local": 22_824},
    },
    "NLW_70_20_10": {
        "run_id": "blend_provisional_nlw70_20_10_92train_y4n_holdout_26m_128x3_s0",
        "mix": {"nitrogen": 70, "wild_provisional": 20, "local": 10},
        "cycle": [
            {"nitrogen": 11, "wild_provisional": 3, "local": 2},
            {"nitrogen": 11, "wild_provisional": 3, "local": 2},
            {"nitrogen": 11, "wild_provisional": 3, "local": 2},
            {"nitrogen": 12, "wild_provisional": 3, "local": 1},
            {"nitrogen": 11, "wild_provisional": 4, "local": 1},
        ],
        "draws": {
            "nitrogen": 159_768,
            "wild_provisional": 45_648,
            "local": 22_824,
        },
    },
}


def _json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
    )


def _committed_file(
    repo: Path,
    path: Path,
    relative_path: Path,
    expected_sha256: str,
    commit: str,
    description: str,
) -> bytes:
    """Return exact bytes only when they are clean and committed as declared."""

    repo = repo.resolve()
    if path.resolve() != (repo / relative_path).resolve():
        raise ValueError(f"{description} must be {relative_path.as_posix()}")
    if not HEX_64.fullmatch(expected_sha256):
        raise ValueError(f"{description} SHA-256 is malformed")
    if not HEX_40.fullmatch(commit):
        raise ValueError(f"{description} commit is malformed")
    if not path.is_file():
        raise ValueError(f"{description} is missing")
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected_sha256:
        raise ValueError(f"{description} SHA-256 mismatch")
    resolved = _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if resolved.returncode or resolved.stdout.strip().decode() != commit:
        raise ValueError(f"{description} commit is not resolvable")
    if _git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode:
        raise ValueError(f"{description} commit is not an ancestor of HEAD")
    blob = _git(repo, "cat-file", "blob", f"{commit}:{relative_path.as_posix()}")
    if blob.returncode or blob.stdout != contents:
        raise ValueError(f"{description} differs from committed bytes")
    status = _git(
        repo,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        relative_path.as_posix(),
    )
    if status.returncode or status.stdout:
        raise ValueError(f"{description} is not clean")
    return contents


def _arm(contract: Mapping[str, Any], arm_name: str) -> dict[str, Any]:
    if arm_name not in ARM_SPECS:
        raise ValueError(f"unknown provisional blend arm: {arm_name}")
    arms = contract.get("arms")
    if not isinstance(arms, list):
        raise ValueError("blend contract arms are missing")
    matches = [value for value in arms if isinstance(value, dict) and value.get("name") == arm_name]
    if len(matches) != 1:
        raise ValueError("blend contract does not contain the requested arm exactly once")
    return matches[0]


def _line_list_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _normalized_cycle(value: object) -> list[dict[str, int]]:
    if not isinstance(value, list):
        raise ValueError("source-sampling step cycle is missing")
    result: list[dict[str, int]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("source-sampling cycle row is not an object")
        result.append({str(key): int(item) for key, item in sorted(row.items())})
    return result


def validate_contract_value(contract: Mapping[str, Any]) -> None:
    """Validate every decision-bearing term shared by both blend arms."""

    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("blend contract schema changed")
    if contract.get("study_id") != STUDY_ID:
        raise ValueError("blend study identity changed")
    model = contract.get("model_contract")
    evaluation = contract.get("evaluation_contract")
    sources = contract.get("sources")
    embargo = contract.get("embargo")
    if not all(isinstance(item, Mapping) for item in (model, evaluation, sources, embargo)):
        raise ValueError("blend model, source, evaluation, or embargo contract is missing")
    exact_model = {
        "initialization": "from scratch; seed 0; no checkpoint initialization",
        "template": TEMPLATE_RELATIVE_PATH.as_posix(),
        "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "window": 128,
        "frame_stride": 3,
        "frame_span": 382,
        "window_mode": "centered",
        "segment_windows": 96,
        "segment_items_per_step": 16,
        "maximum_steps": EXPECTED_FINAL_STEP,
        "evaluation_steps": [0, EXPECTED_FINAL_STEP],
        "learning_rate": 0.0003,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "linear_learning_rate_decay": True,
        "transition_weight": 8.0,
        "weights_reported": "final",
        "checkpoint_selection": "none",
        "positive_weight_policy": "freeze the pure-103h reference vector unchanged",
    }
    for key, expected in exact_model.items():
        if model.get(key) != expected:
            raise ValueError(f"blend model contract changed {key}")
    weights = model.get("positive_weight")
    if (
        model.get("positive_weight_key_order") != list(KEY_ORDER)
        or not isinstance(weights, list)
        or len(weights) != len(KEY_ORDER)
        or not all(isinstance(value, (float, int)) and math.isfinite(float(value)) for value in weights)
    ):
        raise ValueError("blend frozen positive-weight vector changed")

    reference = contract.get("reference")
    if not isinstance(reference, Mapping) or reference.get("run_id") != REFERENCE_RUN_ID:
        raise ValueError("pure-NitroGen comparison reference changed")
    if reference.get("training_mix_percent") != {
        "nitrogen": 100,
        "local": 0,
        "wild_provisional": 0,
    }:
        raise ValueError("pure-NitroGen reference mix changed")
    if not HEX_64.fullmatch(str(reference.get("checkpoint_sha256", ""))):
        raise ValueError("pure-NitroGen reference checkpoint hash is malformed")

    for arm_name, spec in ARM_SPECS.items():
        arm = _arm(contract, arm_name)
        expected = {
            "run_id": spec["run_id"],
            "training_mix_percent": spec["mix"],
            "five_step_cycle": spec["cycle"],
            "expected_draws": spec["draws"],
        }
        for key, value in expected.items():
            if arm.get(key) != value:
                raise ValueError(f"blend arm {arm_name} changed {key}")

    nitrogen = sources.get("nitrogen")
    local = sources.get("local")
    wild = sources.get("wild_provisional")
    if not all(isinstance(value, Mapping) for value in (nitrogen, local, wild)):
        raise ValueError("blend source contracts are incomplete")
    if nitrogen.get("tier") != "mapped_unflagged" or nitrogen.get("sessions") != 1062:
        raise ValueError("NitroGen source tier or membership changed")
    if local.get("tier") != "engine_truth_corrected_own_v3":
        raise ValueError("local source is not the mandatory corrected own-v3 generation")
    if local.get("forbidden_generation") != "/ephemeral/data/own_features":
        raise ValueError("pre-mask-fix local generation is not explicitly forbidden")
    if local.get("complete_segment_items") != 159:
        raise ValueError("corrected local segment pool changed")
    if wild.get("tier") != "provisional_not_train_ready" or wild.get("admitted_hours") != 0.0:
        raise ValueError("wild source lost its provisional zero-admission label")
    if wild.get("sessions") != 2058 or wild.get("complete_segment_items") != 41567:
        raise ValueError("provisional wild source membership changed")

    if evaluation.get("order") != [
        Y4N_SURFACE,
        "commit_and_freeze_y4n_comparison",
        B1_SURFACE,
    ]:
        raise ValueError("blend evaluation order changed")
    if evaluation.get("threshold_policy") != (
        "raw sigmoid probabilities; state and event decisions fixed at 0.5"
    ):
        raise ValueError("blend fixed threshold policy changed")
    if evaluation.get("oracle_thresholds_allowed") is not False:
        raise ValueError("blend contract permits fitted thresholds")
    if evaluation.get("calibration_allowed") is not False:
        raise ValueError("blend contract permits fitted calibration")
    y4n = evaluation.get(Y4N_SURFACE)
    b1 = evaluation.get(B1_SURFACE)
    if not isinstance(y4n, Mapping) or not isinstance(b1, Mapping):
        raise ValueError("blend evaluation surfaces are missing")
    if y4n.get("sessions") != 8 or y4n.get("active_rows") != Y4N_FRAMES or y4n.get("truth_sha256") != Y4N_TRUTH_SHA256:
        raise ValueError("blend y4n surface changed")
    if b1.get("active_rows") != EXPECTED_B1_ACTIVE_FRAMES or b1.get("streams") != EXPECTED_B1_STREAMS or b1.get("truth_sha256") != EXPECTED_B1_TRUTH_SHA256 or b1.get("input_active_sha256") != EXPECTED_B1_ACTIVE_SHA256:
        raise ValueError("blend B1 surface changed")
    if embargo.get("sealed_untouched_session") != UNTOUCHED_SESSION_ID:
        raise ValueError("sealed untouched-session identity changed")
    rule = str(embargo.get("rule", ""))
    if not all(term in rule for term in ("forbidden", "training", "inference", "selection")):
        raise ValueError("sealed untouched-session embargo weakened")


def validate_contract(
    repo: Path,
    contract_path: Path,
    expected_sha256: str,
    contract_commit: str,
) -> dict[str, Any]:
    contents = _committed_file(
        repo,
        contract_path,
        CONTRACT_RELATIVE_PATH,
        expected_sha256,
        contract_commit,
        "blend decision contract",
    )
    try:
        contract = json.loads(contents)
    except json.JSONDecodeError as error:
        raise ValueError("blend decision contract is invalid JSON") from error
    if not isinstance(contract, dict):
        raise ValueError("blend decision contract must be an object")
    validate_contract_value(contract)
    # The contract names a recipe template rather than copying it.  Bind that
    # dependency to the same preregistration commit so a later template edit
    # cannot silently redefine either arm.
    template_path = repo.resolve() / TEMPLATE_RELATIVE_PATH
    if not template_path.is_file():
        raise ValueError("matched GRU template is missing")
    committed_template = _git(
        repo.resolve(),
        "cat-file",
        "blob",
        f"{contract_commit}:{TEMPLATE_RELATIVE_PATH.as_posix()}",
    )
    if committed_template.returncode or committed_template.stdout != template_path.read_bytes():
        raise ValueError("matched GRU template differs from preregistered bytes")
    template_status = _git(
        repo.resolve(),
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        TEMPLATE_RELATIVE_PATH.as_posix(),
    )
    if template_status.returncode or template_status.stdout:
        raise ValueError("matched GRU template is not clean")
    return contract


def _source_lists(
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    arm_name: str,
) -> dict[str, list[str]]:
    sampling = config.get("source_sampling")
    if not isinstance(sampling, Mapping) or not isinstance(sampling.get("sources"), Mapping):
        raise ValueError("run config lacks exact source-sampling membership")
    sources = {
        str(name): [str(value) for value in values]
        for name, values in sampling["sources"].items()
        if isinstance(values, list)
    }
    if set(sources) != set(ARM_SPECS[arm_name]["mix"]):
        raise ValueError("run source names differ from the preregistered arm")
    if any(not values or len(values) != len(set(values)) for values in sources.values()):
        raise ValueError("run source membership is empty or duplicated")
    if any(UNTOUCHED_SESSION_ID in values for values in sources.values()):
        raise ValueError("sealed untouched test appeared in blend training")
    source_contract = contract["sources"]
    expected_membership = {
        "nitrogen": (1062, source_contract["nitrogen"]["session_list_sha256"]),
        "local": (3, source_contract["local"]["train_sessions_sha256"]),
    }
    if arm_name == "NLW_70_20_10":
        expected_membership["wild_provisional"] = (
            2058,
            source_contract["wild_provisional"]["session_list_sha256"],
        )
    for name, (count, digest) in expected_membership.items():
        if len(sources[name]) != count or _line_list_sha256(sources[name]) != digest:
            raise ValueError(f"run source membership changed for {name}")
    return sources


def expected_run_config(
    repo: Path,
    contract: Mapping[str, Any],
    arm_name: str,
    observed_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently rebuild the only accepted config for one arm."""

    template_path = repo.resolve() / TEMPLATE_RELATIVE_PATH
    template = _json_object(template_path, "matched GRU template")
    if "temporal_arch" in template:
        raise ValueError("matched blend template must instantiate the default GRU")
    source_lists = _source_lists(observed_config, contract, arm_name)
    spec = ARM_SPECS[arm_name]
    expected = copy.deepcopy(template)
    expected.update(
        {
            "max_steps": EXPECTED_FINAL_STEP,
            "eval_interval": EXPECTED_FINAL_STEP,
            "seed": 0,
            "frozen_positive_weight": {
                key: float(value)
                for key, value in zip(
                    KEY_ORDER,
                    contract["model_contract"]["positive_weight"],
                    strict=True,
                )
            },
            "source_sampling": {
                "format_version": "madeleine.source-balanced-batch.v1",
                "expected_steps": EXPECTED_FINAL_STEP,
                "cycle_steps": 5,
                "cycle_items": 80,
                "sources": source_lists,
                "step_cycle": spec["cycle"],
            },
            "_note": (
                "provisional blend diagnostic; exact source-balanced 14265-step "
                f"endpoint; final weights only; arm={arm_name}"
            ),
        }
    )
    return expected


def _sampling_receipt(
    receipt: object,
    contract: Mapping[str, Any],
    arm_name: str,
) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise ValueError("completed source-sampling receipt is missing")
    spec = ARM_SPECS[arm_name]
    exact = {
        "format_version": "madeleine.source-balanced-batch.v1",
        "seed": 0,
        "cycle_steps": 5,
        "cycle_items": 80,
        "batch_items": 16,
        "scheduled_steps": EXPECTED_FINAL_STEP,
        "actual_steps": EXPECTED_FINAL_STEP,
        "step_cycle": _normalized_cycle(spec["cycle"]),
        "complete": True,
    }
    for key, value in exact.items():
        observed = _normalized_cycle(receipt.get(key)) if key == "step_cycle" else receipt.get(key)
        if observed != value:
            raise ValueError(f"source-sampling receipt changed {key}")
    sources = receipt.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(spec["draws"]):
        raise ValueError("source-sampling receipt source set changed")
    expected_segments = {
        "local": int(contract["sources"]["local"]["complete_segment_items"]),
        "wild_provisional": int(
            contract["sources"]["wild_provisional"]["complete_segment_items"]
        ),
    }
    expected_sessions = {"nitrogen": 1062, "local": 3, "wild_provisional": 2058}
    for name, expected_draws in spec["draws"].items():
        source = sources[name]
        if not isinstance(source, Mapping):
            raise ValueError(f"source-sampling receipt is malformed for {name}")
        segment_items = source.get("segment_items")
        if not isinstance(segment_items, int) or segment_items < 1:
            raise ValueError(f"source-sampling pool is invalid for {name}")
        if name in expected_segments and segment_items != expected_segments[name]:
            raise ValueError(f"source-sampling segment pool changed for {name}")
        expected_unique = min(segment_items, expected_draws)
        expected_values = {
            "session_count": expected_sessions[name],
            "scheduled_draws": expected_draws,
            "actual_draws": expected_draws,
            "unique_segment_items_drawn": expected_unique,
            "repeat_draws": expected_draws - expected_unique,
            "completed_pool_passes": expected_draws // segment_items,
            "minimum_draws_per_item": expected_draws // segment_items,
            "maximum_draws_per_item": (
                expected_draws + segment_items - 1
            )
            // segment_items,
        }
        for key, value in expected_values.items():
            if source.get(key) != value:
                raise ValueError(f"source-sampling receipt changed {name}.{key}")
        if not math.isclose(
            float(source.get("effective_pool_passes", math.nan)),
            expected_draws / segment_items,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"source-sampling pool-pass receipt changed for {name}")
        if not math.isclose(
            float(source.get("mean_draws_per_item", math.nan)),
            expected_draws / segment_items,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"source-sampling mean-draw receipt changed for {name}")
    return receipt


def _declared_commit(run_meta: Mapping[str, Any]) -> str:
    value = run_meta.get("git")
    if not isinstance(value, str) or not value.endswith("-declared"):
        raise ValueError("run metadata lacks a declared clean implementation commit")
    commit = value.removesuffix("-declared")
    if not HEX_40.fullmatch(commit):
        raise ValueError("run implementation commit is malformed")
    return commit


def validate_run(
    repo: Path,
    run_dir: Path,
    run_id: str,
    contract: Mapping[str, Any],
    arm_name: str,
) -> tuple[dict[str, Any], BadelineIDM, dict[str, Any]]:
    """Validate a completed from-scratch final checkpoint and all receipts."""

    spec = ARM_SPECS.get(arm_name)
    if spec is None or run_id != spec["run_id"] or run_dir.name != run_id:
        raise ValueError("run directory is not the requested preregistered blend arm")
    paths = {
        "config": run_dir / "config.json",
        "checkpoint": run_dir / "model.pt",
        "run_meta": run_dir / "run_meta.json",
        "sampling": run_dir / "source_sampling_receipt.json",
        "log": run_dir / "log.jsonl",
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("completed blend run or provenance receipt is missing")
    config = _json_object(paths["config"], "blend run config")
    expected = expected_run_config(repo, contract, arm_name, config)
    if config != expected:
        changed = sorted(
            key
            for key in set(config) | set(expected)
            if config.get(key) != expected.get(key)
        )
        raise ValueError(f"blend run config differs from frozen recipe: {changed}")
    run_meta = _json_object(paths["run_meta"], "blend run metadata")
    sampling = _sampling_receipt(
        _json_object(paths["sampling"], "source-sampling receipt"),
        contract,
        arm_name,
    )
    source_lists = _source_lists(config, contract, arm_name)
    expected_train = {value for values in source_lists.values() for value in values}
    split = run_meta.get("split")
    if not isinstance(split, Mapping) or set(split.get("train", [])) != expected_train:
        raise ValueError("run metadata training membership changed")
    val_ids = split.get("val") if isinstance(split, Mapping) else None
    expected_val_ids = [f"y4nQHqYSObI__r{index:03d}" for index in range(16)]
    if val_ids != expected_val_ids:
        raise ValueError("run metadata validation membership changed")
    if UNTOUCHED_SESSION_ID in expected_train or UNTOUCHED_SESSION_ID in val_ids:
        raise ValueError("sealed untouched session appeared in the blend run")
    if run_meta.get("config") != config or run_meta.get("seed") != 0:
        raise ValueError("run metadata config or seed changed")
    if run_meta.get("initialized_from") is not None:
        raise ValueError("blend run metadata is not from scratch")
    if run_meta.get("source_sampling") != sampling:
        raise ValueError("run metadata source-sampling receipt changed")
    positive_weight_dict = {
        key: float(value)
        for key, value in zip(
            KEY_ORDER,
            contract["model_contract"]["positive_weight"],
            strict=True,
        )
    }
    if run_meta.get("positive_weight") != positive_weight_dict:
        raise ValueError("run metadata frozen positive weights changed")

    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("blend checkpoint must be an object")
    if checkpoint.get("config") != config or checkpoint.get("key_order") != list(KEY_ORDER):
        raise ValueError("blend checkpoint config or key order changed")
    if checkpoint.get("steps") != EXPECTED_FINAL_STEP:
        raise ValueError("blend checkpoint is not the fixed 14265-step endpoint")
    if checkpoint.get("initialized_from") is not None:
        raise ValueError("blend checkpoint was not trained from scratch")
    positive_weight = checkpoint.get("positive_weight")
    expected_weights = [positive_weight_dict[key] for key in KEY_ORDER]
    if positive_weight != expected_weights:
        raise ValueError("blend checkpoint frozen positive weights changed")
    if checkpoint.get("source_sampling_receipt") != sampling:
        raise ValueError("blend checkpoint source-sampling receipt changed")
    final_state = checkpoint.get("final_state_dict")
    selected_state = checkpoint.get("model_state_dict")
    if not isinstance(final_state, Mapping) or not final_state:
        raise ValueError("blend checkpoint lacks final_state_dict")
    if not isinstance(selected_state, Mapping) or not selected_state:
        raise ValueError("blend checkpoint lacks selected-state receipt")
    best_bce = checkpoint.get("best_val_mean_bce")
    if not isinstance(best_bce, (float, int)) or not math.isfinite(float(best_bce)):
        raise ValueError("blend checkpoint best validation loss is not finite")
    if checkpoint.get("best_val_step") not in (0, EXPECTED_FINAL_STEP):
        raise ValueError("blend checkpoint selected outside its two evaluated endpoints")

    model = BadelineIDM(config)
    if model.temporal_arch != "gru" or not isinstance(model.temporal, torch.nn.GRUCell):
        raise ValueError("blend recipe does not instantiate the matched GRU")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError("matched blend GRU parameter count changed")
    model.load_state_dict(final_state, strict=True)
    implementation_commit = _declared_commit(run_meta)
    source_receipt = _verify_inference_source(repo.resolve(), implementation_commit)

    log_records = [
        json.loads(line)
        for line in paths["log"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [record.get("step") for record in log_records] != [0, EXPECTED_FINAL_STEP]:
        raise ValueError("blend training log lacks the fixed final endpoint")
    receipt = {
        "arm": arm_name,
        "config_sha256": sha256_file(paths["config"]),
        "checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "run_meta_sha256": sha256_file(paths["run_meta"]),
        "source_sampling_receipt_sha256": sha256_file(paths["sampling"]),
        "training_log_sha256": sha256_file(paths["log"]),
        "checkpoint_steps": EXPECTED_FINAL_STEP,
        "best_val_step": int(checkpoint.get("best_val_step", -1)),
        "selected_final_tensors_identical": _state_dicts_identical(
            selected_state, final_state
        ),
        "parameter_count": parameter_count,
        "evaluation_weights": "final_state_dict",
        "initialization": "from_scratch",
        "positive_weight": positive_weight_dict,
        "source_sampling": sampling,
        "inference_source": source_receipt,
    }
    return config, model, receipt


def validate_y4n_release(
    report_path: Path,
    marker_path: Path,
    *,
    arm_name: str,
    contract_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Validate one complete fixed-only arm release before comparison freeze."""

    if not report_path.is_file() or not marker_path.is_file():
        raise ValueError("completed blend y4n report and marker are required")
    report = _json_object(report_path, "blend y4n report")
    marker = _json_object(marker_path, "blend y4n completion marker")
    if _contains_disallowed_metric_language(report) or _contains_disallowed_metric_language(marker):
        raise ValueError("blend y4n release contains a fitted-metric diagnostic")
    run_id = ARM_SPECS[arm_name]["run_id"]
    expected = {
        "study_id": STUDY_ID,
        "arm": arm_name,
        "run_id": run_id,
        "surface": Y4N_SURFACE,
        "weights": "final",
    }
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("blend y4n report schema changed")
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"blend y4n report changed {key}")
    if report.get("contract", {}).get("sha256") != contract_sha256:
        raise ValueError("blend y4n report contract hash changed")
    if report.get("run_receipt", {}).get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("blend y4n report checkpoint hash changed")
    support = report.get("support")
    support_exact = {
        "all_frames": Y4N_FRAMES,
        "input_active_frames": Y4N_FRAMES,
        "streams": len(Y4N_STREAM_IDS),
        "session_ids": Y4N_STREAM_IDS,
        "stream_lengths": Y4N_STREAM_LENGTHS,
        "truth_sha256": Y4N_TRUTH_SHA256,
        "finite_aligned_arrays": True,
    }
    if not isinstance(support, Mapping) or any(support.get(key) != value for key, value in support_exact.items()):
        raise ValueError("blend y4n report support changed")
    threshold = report.get("fixed_metrics", {}).get("threshold_policy")
    if threshold != {
        "state_probability": 0.5,
        "transition_probability": 0.5,
        "data_fitted_thresholds_used": False,
        "calibration_parameters_fitted": False,
    }:
        raise ValueError("blend y4n release is not fixed-only")
    if marker.get("schema_version") != MARKER_SCHEMA_VERSION or marker.get("status") != "complete":
        raise ValueError("blend y4n completion marker changed")
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"blend y4n marker changed {key}")
    if marker.get("contract_sha256") != contract_sha256 or marker.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("blend y4n marker provenance changed")
    if marker.get("report_sha256") != sha256_file(report_path):
        raise ValueError("blend y4n marker report hash changed")
    sidecar_receipt = report.get("prediction_sidecar")
    if not isinstance(sidecar_receipt, Mapping):
        raise ValueError("blend y4n report lacks a sidecar receipt")
    sidecar_path = Path(str(sidecar_receipt.get("path", "")))
    sidecar_sha = str(sidecar_receipt.get("sha256", ""))
    if not sidecar_path.is_file() or sha256_file(sidecar_path) != sidecar_sha:
        raise ValueError("blend y4n prediction sidecar changed")
    validate_y4n_sidecar(sidecar_path)
    if marker.get("sidecar_sha256") != sidecar_sha:
        raise ValueError("blend y4n marker sidecar hash changed")
    return {
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "completion_marker": str(marker_path),
        "completion_marker_sha256": sha256_file(marker_path),
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sidecar_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "surface": Y4N_SURFACE,
        "weights": "final",
    }


def validate_comparison_value(
    value: Mapping[str, Any],
    *,
    arm_name: str,
    arm_release: Mapping[str, Any],
) -> None:
    """Validate the semantic content of the post-y4n B1 release gate."""

    if value.get("schema_version") != COMPARISON_SCHEMA_VERSION or value.get("study_id") != STUDY_ID:
        raise ValueError("frozen blend y4n comparison identity changed")
    population = value.get("evaluation_population")
    if not isinstance(population, Mapping):
        raise ValueError("frozen blend comparison lacks its population")
    exact_population = {
        "surface": Y4N_SURFACE,
        "session_ids": Y4N_STREAM_IDS,
        "active_frames": Y4N_FRAMES,
        "truth_sha256": Y4N_TRUTH_SHA256,
        "fixed_threshold": 0.5,
        "fitted_thresholds_used": False,
        "fitted_calibration_used": False,
        "b1_used": False,
    }
    for key, expected in exact_population.items():
        if population.get(key) != expected:
            raise ValueError(f"frozen blend comparison changed population.{key}")
    decision = value.get("decision")
    if not isinstance(decision, Mapping) or decision.get("comparison_frozen_before_b1") is not True:
        raise ValueError("blend y4n comparison was not frozen before B1")
    if decision.get("all_runs_consulted") != [
        REFERENCE_RUN_ID,
        ARM_SPECS["NL_90_10"]["run_id"],
        ARM_SPECS["NLW_70_20_10"]["run_id"],
    ]:
        raise ValueError("frozen blend comparison did not consult exactly three runs")
    runs = value.get("runs")
    required = {
        REFERENCE_RUN_ID,
        ARM_SPECS["NL_90_10"]["run_id"],
        ARM_SPECS["NLW_70_20_10"]["run_id"],
    }
    if not isinstance(runs, Mapping) or set(runs) != required:
        raise ValueError("frozen blend comparison run set changed")
    requested = runs[ARM_SPECS[arm_name]["run_id"]]
    if not isinstance(requested, Mapping):
        raise ValueError("frozen blend comparison arm receipt is malformed")
    for key in ("checkpoint_sha256", "report_sha256", "sidecar_sha256", "completion_marker_sha256"):
        if requested.get(key) != arm_release[key]:
            raise ValueError(f"frozen blend comparison changed requested arm {key}")
    if requested.get("weights") != "final" or requested.get("complete") is not True:
        raise ValueError("requested blend arm is incomplete in frozen comparison")
    for run_id, run in runs.items():
        if not isinstance(run, Mapping) or run.get("complete") is not True:
            raise ValueError(f"run is incomplete in frozen blend comparison: {run_id}")
        if run.get("weights") != "final":
            raise ValueError(f"run did not use final weights in frozen comparison: {run_id}")
        for key in (
            "checkpoint_sha256",
            "report_sha256",
            "sidecar_sha256",
            "completion_marker_sha256",
        ):
            if not HEX_64.fullmatch(str(run.get(key, ""))):
                raise ValueError(
                    f"run has malformed {key} in frozen comparison: {run_id}"
                )


def validate_committed_comparison(
    repo: Path,
    path: Path,
    expected_sha256: str,
    commit: str,
    *,
    arm_name: str,
    arm_release: Mapping[str, Any],
) -> dict[str, Any]:
    contents = _committed_file(
        repo,
        path,
        COMPARISON_RELATIVE_PATH,
        expected_sha256,
        commit,
        "frozen blend y4n comparison",
    )
    try:
        value = json.loads(contents)
    except json.JSONDecodeError as error:
        raise ValueError("frozen blend y4n comparison is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("frozen blend y4n comparison must be an object")
    validate_comparison_value(value, arm_name=arm_name, arm_release=arm_release)
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "commit": commit,
        "comparison_frozen_before_b1": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", choices=("y4n-later8", "b1"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--contract-commit", required=True)
    parser.add_argument("--arm", choices=tuple(ARM_SPECS), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--completion-marker", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--b1-marker", type=Path)
    parser.add_argument("--y4n-report", type=Path)
    parser.add_argument("--y4n-completion-marker", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--comparison-sha256")
    parser.add_argument("--comparison-commit")
    return parser


def main() -> None:
    args = _parser().parse_args()
    contract = validate_contract(
        args.repo,
        args.contract,
        args.contract_sha256,
        args.contract_commit,
    )
    config, model, run_receipt = validate_run(
        args.repo,
        args.run,
        args.run_id,
        contract,
        args.arm,
    )

    y4n_release: dict[str, Any] | None = None
    comparison_release: dict[str, Any] | None = None
    if args.surface == "b1":
        required = (
            args.b1_marker,
            args.y4n_report,
            args.y4n_completion_marker,
            args.comparison,
            args.comparison_sha256,
            args.comparison_commit,
        )
        if any(value is None for value in required):
            raise ValueError(
                "B1 requires the completed arm y4n release and committed comparison gate"
            )
        y4n_release = validate_y4n_release(
            args.y4n_report,
            args.y4n_completion_marker,
            arm_name=args.arm,
            contract_sha256=args.contract_sha256,
            checkpoint_sha256=run_receipt["checkpoint_sha256"],
        )
        comparison_release = validate_committed_comparison(
            args.repo,
            args.comparison,
            args.comparison_sha256,
            args.comparison_commit,
            arm_name=args.arm,
            arm_release=y4n_release,
        )

    artifacts = [args.out, args.sidecar, args.completion_marker]
    _refuse_existing(artifacts)
    if args.out.suffix != ".json" or args.sidecar.suffix != ".npz":
        raise ValueError("report must be JSON and prediction sidecar must be NPZ")
    for path in artifacts:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = _temporary_path(args.out)
    temporary_sidecar = _temporary_path(args.sidecar, npz=True)
    _refuse_existing([temporary_report, temporary_sidecar])

    if args.surface == "y4n-later8":
        if any(
            value is not None
            for value in (
                args.b1_marker,
                args.y4n_report,
                args.y4n_completion_marker,
                args.comparison,
                args.comparison_sha256,
                args.comparison_commit,
            )
        ):
            raise ValueError("y4n evaluation must not receive any B1 release input")
        session_ids = read_session_ids(args.sessions)
        if session_ids != Y4N_BASE_SESSION_IDS:
            raise ValueError("y4n session list is not the exact later-eight split")
        surface = Y4N_SURFACE
        label_kind = "mapped_foreign_nitrogen"
        label_notice = (
            "Primary fixed-policy development comparison against noisy mapped "
            "NitroGen labels; wild supervision remains provisional."
        )
    else:
        assert args.b1_marker is not None
        # No B1 path is statted or read until both y4n gates above pass.
        session_ids = validate_b1_surface(args.data, args.sessions, args.b1_marker)
        surface = B1_SURFACE
        label_kind = "engine_truth_development_b1"
        label_notice = (
            "Post-decision fixed-policy engine-truth development transfer. "
            "B1 cannot train, tune, select, or promote a blend arm."
        )

    try:
        truth, probability, active, lengths, stream_ids = infer_fixed_state(
            model,
            config,
            args.data,
            session_ids,
            args.device,
            temporary_sidecar,
        )
        if surface == Y4N_SURFACE:
            support = validate_y4n_sidecar(temporary_sidecar)
            if stream_ids != Y4N_STREAM_IDS:
                raise ValueError("blend y4n inference stream identities changed")
        else:
            support = validate_b1_sidecar(temporary_sidecar)
            if stream_ids != EXPECTED_B1_STREAM_IDS:
                raise ValueError("blend B1 inference stream identities changed")
        truth, probability, active, lengths = _load_sidecar_arrays(temporary_sidecar)
        metrics = fixed_metric_report(truth, probability, active, lengths.tolist())
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "study_id": STUDY_ID,
            "arm": args.arm,
            "run_id": args.run_id,
            "surface": surface,
            "weights": "final",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "label_kind": label_kind,
            "label_notice": label_notice,
            "sessions": list(session_ids),
            "support": support,
            "fixed_metrics": metrics,
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
                "path": str(args.contract),
                "sha256": args.contract_sha256,
                "commit": args.contract_commit,
            },
            "run_receipt": run_receipt,
            "prediction_sidecar": {
                "path": str(args.sidecar),
                "sha256": sha256_file(temporary_sidecar),
            },
        }
        if y4n_release is not None and comparison_release is not None:
            report["y4n_arm_release_gate"] = y4n_release
            report["y4n_comparison_release_gate"] = comparison_release
            report["b1_policy"] = {
                "post_committed_y4n_comparison_only": True,
                "used_for_training": False,
                "used_for_checkpoint_selection": False,
                "used_for_threshold_fitting": False,
                "used_for_calibration_fitting": False,
            }
        if _contains_disallowed_metric_language(report):
            raise AssertionError("fixed-only blend report contains fitted-metric output")
        _atomic_json(temporary_report, report)
        marker = {
            "schema_version": MARKER_SCHEMA_VERSION,
            "status": "complete",
            "study_id": STUDY_ID,
            "arm": args.arm,
            "run_id": args.run_id,
            "surface": surface,
            "weights": "final",
            "contract_sha256": args.contract_sha256,
            "checkpoint_sha256": run_receipt["checkpoint_sha256"],
            "run_meta_sha256": run_receipt["run_meta_sha256"],
            "source_sampling_receipt_sha256": run_receipt[
                "source_sampling_receipt_sha256"
            ],
            "report_sha256": sha256_file(temporary_report),
            "sidecar_sha256": sha256_file(temporary_sidecar),
        }
        _publish_result(
            report_path=args.out,
            temporary_report=temporary_report,
            sidecar_path=args.sidecar,
            temporary_sidecar=temporary_sidecar,
            marker_path=args.completion_marker,
            marker=marker,
        )
    finally:
        temporary_report.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "study_id": STUDY_ID,
                "arm": args.arm,
                "run_id": args.run_id,
                "surface": surface,
                "weights": "final",
                "input_active_frames": support["input_active_frames"],
                "fixed_threshold": 0.5,
                "report": str(args.out),
                "sidecar": str(args.sidecar),
                "completion_marker": str(args.completion_marker),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
