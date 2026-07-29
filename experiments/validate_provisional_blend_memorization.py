"""Read-only validator for provisional-blend memorization diagnostics.

This validator reads exactly two evaluation artifacts (one JSON report and its
JSON completion marker) plus Git object metadata for the declared diagnostic
commit.  It never opens feature shards, checkpoints, prediction arrays, B1,
val-B, mapped-y4n, or the sealed untouched session.  It has no output-file
argument and cannot rerun inference.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any


REPORT_SCHEMA_VERSION = "madeleine.provisional-blend-memorization.v1"
MARKER_SCHEMA_VERSION = "madeleine.provisional-blend-memorization-marker.v1"
STUDY_ID = "provisional_blend_gru_y4n_b1_s0"
SURFACE = "complete_training_segment_pools_and_corrected_local_val_a"
CONTRACT_RELATIVE_PATH = Path(
    "experiments/configs/provisional_blend_gru_decision.json"
)
DIAGNOSTIC_RELATIVE_PATH = Path(
    "experiments/eval_provisional_blend_memorization.py"
)
CONTRACT_SHA256 = (
    "ee194ea370bde2ad8f4797e59d6373030101d7d4dac140f6375f5f7db40630b5"
)
CONTRACT_COMMIT = "8e98f949aab976d89f801e9e6fdca0cb4ab9b53a"
TRAINING_IMPLEMENTATION_COMMIT = CONTRACT_COMMIT
FINAL_STEP = 14_265
PARAMETER_COUNT = 25_719_815
SEGMENT_WINDOWS = 96
KEY_ORDER = ("left", "right", "up", "down", "jump", "dash", "grab")
POSITIVE_WEIGHT = {
    "left": 6.5241737274432445,
    "right": 2.165329049329906,
    "up": 5.234555340345659,
    "down": 10.0,
    "jump": 6.478684563686479,
    "dash": 10.0,
    "grab": 1.7741724096558231,
}
TRAINING_SOURCE_HASHES = {
    "badeline/model.py": (
        "ce535fd129363510eeaa378cd7413e75d5053d7e0784de0aacb8422bd90f0209"
    ),
    "badeline/temporal.py": (
        "c47eed4ef43d6e93b5d7f1c5730768630586b0d4826a9d57301770328d212f2d"
    ),
    "badeline/train.py": (
        "eb103786a1e54dac11af39b80e28a144afa4c1d59df6d4be8cc37bda7c224aed"
    ),
    "data/schema.py": (
        "3712ec20318a76a7939cd6561eafd75c67990a55556fabea18524fb79c52e0a2"
    ),
}
LOCAL_VAL_A_ID = "rec_20260724_171305_5min"
LOCAL_VAL_A_RGB_SHA256 = (
    "6abde83a24da4202e7c148722f2ac0ceaa0f204e13905795c5662c1d267b3d62"
)
FORBIDDEN_SESSION_IDS = frozenset(
    {
        "rec_20260727_220000_test",
        "rec_20260725_025853",
        "rec_20260725_160450_b1",
    }
)
HEX = frozenset("0123456789abcdef")

SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "nitrogen": {
        "tier": "mapped_unflagged",
        "sessions": 1_062,
        "segment_items": 228_237,
        "membership_sha256": (
            "e668ccbd0aa02fb1bda79c2da621df6ae4cb0d183280580076e20b9f00996943"
        ),
    },
    "local": {
        "tier": "engine_truth_corrected_own_v3",
        "sessions": 3,
        "segment_items": 159,
        "membership_sha256": (
            "aa0ead8291970707d283515843c1d2776c8d31afc710654accac8e91d3e28a09"
        ),
    },
    "wild_provisional": {
        "tier": "provisional_not_train_ready",
        "sessions": 2_058,
        "segment_items": 41_567,
        "membership_sha256": (
            "c670af69f1805ba7260055473f79b30935e68f378c24ed8b286b85e1bbe6b3f4"
        ),
    },
    "local_val_a": {
        "tier": "engine_truth_corrected_own_v3_development",
        "sessions": 1,
        "segment_items": 5,
        "membership_sha256": (
            "fe324018ea818ab573973853fc9df90f0c5d001dee0f1391b8aaae6776141b54"
        ),
    },
}

ARM_SPECS: dict[str, dict[str, Any]] = {
    "NL_90_10": {
        "run_id": "blend_provisional_nl90_10_92train_y4n_holdout_26m_128x3_s0",
        "sources": ("nitrogen", "local"),
        "draws": {"nitrogen": 205_416, "local": 22_824},
        "step_cycle": (
            {"nitrogen": 14, "local": 2},
            {"nitrogen": 14, "local": 2},
            {"nitrogen": 14, "local": 2},
            {"nitrogen": 15, "local": 1},
            {"nitrogen": 15, "local": 1},
        ),
        "config_sha256": (
            "1275a39882d8f8220ebf7764cea0e44358b32d35e8ab4e4069aed02b26c96052"
        ),
        "checkpoint_sha256": (
            "e5e194172a31e3c6a14a2ba9d1c5233c1b37a112504b3275c2e2e1d1a55e7bf9"
        ),
        "run_meta_sha256": (
            "d565ae3ac9357dc08b755cc860af4e313a1bb22978daa5d77a53c62f7e8dda23"
        ),
        "source_sampling_receipt_sha256": (
            "f05e0551da635cd573b0e97213a8a1ef83ef0c8ae67be50d755bd0e0ddf20b99"
        ),
        "training_log_sha256": (
            "c689f1154cec4ab5d8a4eb84fcc03b51d4795b537fa0adbd46f703d8f3402e7a"
        ),
    },
    "NLW_70_20_10": {
        "run_id": (
            "blend_provisional_nlw70_20_10_92train_y4n_holdout_26m_128x3_s0"
        ),
        "sources": ("nitrogen", "wild_provisional", "local"),
        "draws": {
            "nitrogen": 159_768,
            "wild_provisional": 45_648,
            "local": 22_824,
        },
        "step_cycle": (
            {"nitrogen": 11, "wild_provisional": 3, "local": 2},
            {"nitrogen": 11, "wild_provisional": 3, "local": 2},
            {"nitrogen": 11, "wild_provisional": 3, "local": 2},
            {"nitrogen": 12, "wild_provisional": 3, "local": 1},
            {"nitrogen": 11, "wild_provisional": 4, "local": 1},
        ),
        "config_sha256": (
            "388465e6f2f587b21ea546d2e7a28d43f690f2f444739c48122e7d1bdbd00aad"
        ),
        "checkpoint_sha256": (
            "b4d7a677fa8cfc981f043f79da7f87dc01536a2a9a942ecb26652d9aa96c7cfd"
        ),
        "run_meta_sha256": (
            "925a584a1b2b947d28ae08a108084fe983bff7a89d4a10bd0748a95e56f46891"
        ),
        "source_sampling_receipt_sha256": (
            "3c1dd1819b439e68da0947b518b13d3c2ddd4ce1072ea22eb7b9ce4e25a35bbc"
        ),
        "training_log_sha256": (
            "c0ef376121e1996b3b310873de11a71f384040f93c97256817ec5dc8803ac60c"
        ),
    },
}

REPORT_KEYS = {
    "schema_version",
    "study_id",
    "arm",
    "run_id",
    "surface",
    "evaluated_at",
    "weights",
    "contract",
    "diagnostic_source",
    "run_receipt",
    "feature_view",
    "scope_guard",
    "method",
    "source_sampling_receipt",
    "surfaces",
    "local_generalization_gap",
}
MARKER_KEYS = {
    "schema_version",
    "status",
    "study_id",
    "arm",
    "run_id",
    "surface",
    "weights",
    "contract_sha256",
    "training_implementation_git_commit",
    "diagnostic_git_commit",
    "diagnostic_module_sha256",
    "config_sha256",
    "checkpoint_sha256",
    "run_meta_sha256",
    "training_log_sha256",
    "source_sampling_receipt_sha256",
    "selected_final_tensors_identical",
    "feature_view_receipt_sha256",
    "forbidden_surfaces_accessed",
    "report",
    "report_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _line_list_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def _is_commit(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= HEX


def _json_object(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not readable JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} fields changed: missing={sorted(expected - set(value))} "
            f"extra={sorted(set(value) - expected)}"
        )


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _close(left: object, right: object, name: str) -> None:
    left_value = _finite(left, name)
    right_value = _finite(right, name)
    if not math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} arithmetic changed")


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
    )


def _verify_diagnostic_source(
    repo: Path,
    expected_commit: str,
    source: Mapping[str, Any],
) -> None:
    _exact_keys(source, {"git_commit", "relative_path", "sha256"}, "diagnostic source")
    if not _is_commit(expected_commit) or source.get("git_commit") != expected_commit:
        raise ValueError("diagnostic commit does not match the expected commit")
    if source.get("relative_path") != DIAGNOSTIC_RELATIVE_PATH.as_posix():
        raise ValueError("diagnostic module path changed")
    digest = source.get("sha256")
    if not _is_sha256(digest):
        raise ValueError("diagnostic module SHA-256 is malformed")
    repo = repo.resolve()
    resolved = _git(repo, "rev-parse", "--verify", f"{expected_commit}^{{commit}}")
    if resolved.returncode or resolved.stdout.strip().decode() != expected_commit:
        raise ValueError("diagnostic commit is not resolvable")
    if _git(repo, "merge-base", "--is-ancestor", expected_commit, "HEAD").returncode:
        raise ValueError("diagnostic commit is not an ancestor of HEAD")
    blob = _git(
        repo,
        "cat-file",
        "blob",
        f"{expected_commit}:{DIAGNOSTIC_RELATIVE_PATH.as_posix()}",
    )
    if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != digest:
        raise ValueError("diagnostic module does not match its committed blob")


def _load_contract_sources(repo: Path) -> Mapping[str, Any]:
    """Read the frozen contract blob from Git without following report paths."""

    repo = repo.resolve()
    resolved = _git(repo, "rev-parse", "--verify", f"{CONTRACT_COMMIT}^{{commit}}")
    if resolved.returncode or resolved.stdout.strip().decode() != CONTRACT_COMMIT:
        raise ValueError("frozen contract commit is not resolvable")
    if _git(repo, "merge-base", "--is-ancestor", CONTRACT_COMMIT, "HEAD").returncode:
        raise ValueError("frozen contract commit is not an ancestor of HEAD")
    blob = _git(
        repo,
        "cat-file",
        "blob",
        f"{CONTRACT_COMMIT}:{CONTRACT_RELATIVE_PATH.as_posix()}",
    )
    if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != CONTRACT_SHA256:
        raise ValueError("frozen contract blob or SHA-256 changed")
    try:
        contract = json.loads(blob.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("frozen contract blob is not valid JSON") from error
    sources = contract.get("sources") if isinstance(contract, Mapping) else None
    if not isinstance(sources, Mapping):
        raise ValueError("frozen contract lacks source contracts")
    return sources


def _expected_sampling_source(source: str, draws: int) -> dict[str, Any]:
    spec = SOURCE_SPECS[source]
    items = int(spec["segment_items"])
    return {
        "session_count": int(spec["sessions"]),
        "segment_items": items,
        "scheduled_draws": draws,
        "actual_draws": draws,
        "unique_segment_items_drawn": min(items, draws),
        "repeat_draws": max(0, draws - items),
        "completed_pool_passes": draws // items,
        "effective_pool_passes": draws / items,
        "mean_draws_per_item": draws / items,
        "minimum_draws_per_item": draws // items,
        "maximum_draws_per_item": (draws + items - 1) // items,
    }


def _validate_sampling(value: Mapping[str, Any], arm: str) -> None:
    spec = ARM_SPECS[arm]
    expected_keys = {
        "format_version",
        "seed",
        "cycle_steps",
        "cycle_items",
        "batch_items",
        "scheduled_steps",
        "actual_steps",
        "step_cycle",
        "complete",
        "sources",
    }
    _exact_keys(value, expected_keys, "source-sampling receipt")
    expected_scalar = {
        "format_version": "madeleine.source-balanced-batch.v1",
        "seed": 0,
        "cycle_steps": 5,
        "cycle_items": 80,
        "batch_items": 16,
        "scheduled_steps": FINAL_STEP,
        "actual_steps": FINAL_STEP,
        "complete": True,
    }
    for key, expected in expected_scalar.items():
        if value.get(key) != expected:
            raise ValueError(f"source-sampling {key} changed")
    if value.get("step_cycle") != list(spec["step_cycle"]):
        raise ValueError("source-sampling step cycle changed")
    sources = value.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(spec["sources"]):
        raise ValueError("source-sampling source set changed")
    for source in spec["sources"]:
        row = sources[source]
        expected = _expected_sampling_source(source, int(spec["draws"][source]))
        if not isinstance(row, Mapping) or set(row) != set(expected):
            raise ValueError(f"source-sampling fields changed for {source}")
        for key, expected_value in expected.items():
            observed = row.get(key)
            if isinstance(expected_value, float):
                _close(observed, expected_value, f"source-sampling {source}.{key}")
            elif observed != expected_value:
                raise ValueError(f"source-sampling {source}.{key} changed")


def _validate_run_receipt(value: Mapping[str, Any], arm: str) -> None:
    expected_keys = {
        "arm",
        "config_sha256",
        "checkpoint_sha256",
        "run_meta_sha256",
        "source_sampling_receipt_sha256",
        "training_log_sha256",
        "checkpoint_steps",
        "best_val_step",
        "selected_final_tensors_identical",
        "parameter_count",
        "evaluation_weights",
        "initialization",
        "positive_weight",
        "source_sampling",
        "inference_source",
    }
    _exact_keys(value, expected_keys, "run receipt")
    spec = ARM_SPECS[arm]
    exact = {
        "arm": arm,
        "config_sha256": spec["config_sha256"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "run_meta_sha256": spec["run_meta_sha256"],
        "source_sampling_receipt_sha256": spec[
            "source_sampling_receipt_sha256"
        ],
        "training_log_sha256": spec["training_log_sha256"],
        "checkpoint_steps": FINAL_STEP,
        "best_val_step": FINAL_STEP,
        "selected_final_tensors_identical": True,
        "parameter_count": PARAMETER_COUNT,
        "evaluation_weights": "final_state_dict",
        "initialization": "from_scratch",
        "positive_weight": POSITIVE_WEIGHT,
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise ValueError(f"run receipt {key} changed")
    inference = value.get("inference_source")
    expected_inference = {
        "implementation_git_commit": TRAINING_IMPLEMENTATION_COMMIT,
        "verified_files_sha256": TRAINING_SOURCE_HASHES,
    }
    if inference != expected_inference:
        raise ValueError("training implementation provenance changed")
    sampling = value.get("source_sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("run receipt source sampling is missing")
    _validate_sampling(sampling, arm)


def _validate_metric_bundle(metrics: Mapping[str, Any]) -> None:
    metric_names = {
        "unweighted_bce",
        "average_precision",
        "state_f1_fixed_0_5",
        "prevalence",
        "predicted_positive_rate_fixed_0_5",
    }
    _exact_keys(metrics, metric_names, "surface metrics")
    for metric_name in sorted(metric_names):
        metric = metrics[metric_name]
        if not isinstance(metric, Mapping):
            raise ValueError(f"{metric_name} metric is not an object")
        _exact_keys(metric, {"per_key", "macro"}, metric_name)
        per_key = metric.get("per_key")
        if not isinstance(per_key, Mapping) or set(per_key) != set(KEY_ORDER):
            raise ValueError(f"{metric_name} does not contain exactly seven keys")
        values = [_finite(per_key[key], f"{metric_name}.{key}") for key in KEY_ORDER]
        if metric_name == "unweighted_bce":
            if any(value < 0.0 for value in values):
                raise ValueError("unweighted BCE is negative")
        elif any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(f"{metric_name} is outside [0, 1]")
        macro = _finite(metric.get("macro"), f"{metric_name}.macro")
        _close(macro, sum(values) / len(values), f"{metric_name}.macro")


def _validate_decision_metric_counts(
    metrics: Mapping[str, Any], target_frames: int
) -> None:
    """Require prevalence, PPR, and F1 to describe feasible integer counts."""

    prevalence = metrics["prevalence"]["per_key"]
    predicted_rate = metrics["predicted_positive_rate_fixed_0_5"]["per_key"]
    state_f1 = metrics["state_f1_fixed_0_5"]["per_key"]
    for key in KEY_ORDER:
        positive_float = float(prevalence[key]) * target_frames
        predicted_float = float(predicted_rate[key]) * target_frames
        positive = round(positive_float)
        predicted = round(predicted_float)
        if not math.isclose(positive_float, positive, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{key} prevalence is incompatible with frame support")
        if not math.isclose(predicted_float, predicted, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"{key} predicted-positive rate is incompatible with frame support"
            )
        if not 0 < positive < target_frames or not 0 <= predicted <= target_frames:
            raise ValueError(f"{key} decision support lacks a valid class count")
        denominator = positive + predicted
        true_positive_float = float(state_f1[key]) * denominator / 2.0
        true_positive = round(true_positive_float)
        if not math.isclose(
            true_positive_float, true_positive, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(f"{key} state F1 is incompatible with integer counts")
        minimum = max(0, positive + predicted - target_frames)
        maximum = min(positive, predicted)
        if not minimum <= true_positive <= maximum:
            raise ValueError(f"{key} state F1 implies an impossible confusion matrix")
        expected_f1 = 0.0 if denominator == 0 else 2.0 * true_positive / denominator
        _close(state_f1[key], expected_f1, f"{key} state F1")


def _validate_shard_receipt(
    value: Mapping[str, Any],
    source: str,
    session_ids: Sequence[str],
) -> None:
    _exact_keys(
        value,
        {"sessions", "total_bytes", "membership_sha256", "shard_set_sha256", "shards"},
        f"{source} shard receipt",
    )
    spec = SOURCE_SPECS[source]
    if value.get("sessions") != spec["sessions"]:
        raise ValueError(f"{source} shard session count changed")
    if value.get("membership_sha256") != spec["membership_sha256"]:
        raise ValueError(f"{source} shard membership hash changed")
    shards = value.get("shards")
    if not isinstance(shards, Mapping) or set(shards) != set(session_ids):
        raise ValueError(f"{source} shard membership changed")
    total_bytes = 0
    for session_id, row in shards.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"{source} shard row is not an object")
        _exact_keys(row, {"bytes", "sha256"}, f"{source} shard row")
        byte_count = row.get("bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
            raise ValueError(f"{source} shard byte count is invalid")
        if not _is_sha256(row.get("sha256")):
            raise ValueError(f"{source} shard SHA-256 is malformed")
        total_bytes += byte_count
    if value.get("total_bytes") != total_bytes:
        raise ValueError(f"{source} shard byte total changed")
    if value.get("shard_set_sha256") != _canonical_json_sha256(shards):
        raise ValueError(f"{source} shard-set hash changed")


def _validate_support(
    value: Mapping[str, Any],
    source: str,
    session_ids: Sequence[str],
) -> None:
    keys = {
        "sessions",
        "sessions_with_complete_segments",
        "sessions_without_complete_segments",
        "segment_items",
        "segment_windows",
        "target_frames",
        "binary_labels",
        "session_segment_items",
        "session_segment_items_sha256",
        "truth_sha256",
        "probability_sha256",
    }
    _exact_keys(value, keys, f"{source} support")
    spec = SOURCE_SPECS[source]
    expected_items = int(spec["segment_items"])
    expected_sessions = int(spec["sessions"])
    exact = {
        "sessions": expected_sessions,
        "segment_items": expected_items,
        "segment_windows": SEGMENT_WINDOWS,
        "target_frames": expected_items * SEGMENT_WINDOWS,
        "binary_labels": expected_items * SEGMENT_WINDOWS * len(KEY_ORDER),
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise ValueError(f"{source} support {key} changed")
    item_counts = value.get("session_segment_items")
    if not isinstance(item_counts, Mapping) or set(item_counts) != set(session_ids):
        raise ValueError(f"{source} per-session support membership changed")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in item_counts.values()
    ):
        raise ValueError(f"{source} per-session item count is invalid")
    if sum(item_counts.values()) != expected_items:
        raise ValueError(f"{source} per-session item support does not sum")
    with_segments = sum(count > 0 for count in item_counts.values())
    without_segments = expected_sessions - with_segments
    if value.get("sessions_with_complete_segments") != with_segments:
        raise ValueError(f"{source} sessions-with-support count changed")
    if value.get("sessions_without_complete_segments") != without_segments:
        raise ValueError(f"{source} sessions-without-support count changed")
    if value.get("session_segment_items_sha256") != _canonical_json_sha256(item_counts):
        raise ValueError(f"{source} per-session support hash changed")
    for key in ("truth_sha256", "probability_sha256"):
        if not _is_sha256(value.get(key)):
            raise ValueError(f"{source} {key} is malformed")


def _validate_source_contract(source: str, value: Mapping[str, Any]) -> None:
    spec = SOURCE_SPECS[source]
    if value.get("tier") != spec["tier"]:
        raise ValueError(f"{source} source tier changed")
    if source == "nitrogen":
        if value.get("sessions") != spec["sessions"] or value.get(
            "session_list_sha256"
        ) != spec["membership_sha256"]:
            raise ValueError("NitroGen source contract changed")
    elif source == "local":
        if (
            value.get("complete_segment_items") != spec["segment_items"]
            or value.get("train_sessions_sha256") != spec["membership_sha256"]
            or value.get("val_sessions_sha256")
            != SOURCE_SPECS["local_val_a"]["membership_sha256"]
            or value.get("forbidden_generation") != "/ephemeral/data/own_features"
        ):
            raise ValueError("local source contract changed")
    elif source == "wild_provisional":
        if (
            value.get("sessions") != spec["sessions"]
            or value.get("complete_segment_items") != spec["segment_items"]
            or value.get("session_list_sha256") != spec["membership_sha256"]
            or value.get("admitted_hours") != 0.0
        ):
            raise ValueError("wild provisional source contract changed")


def _validate_surface(
    value: Mapping[str, Any],
    source: str,
    session_ids: Sequence[str],
    sampling: Mapping[str, Any],
    expected_source_contract: Mapping[str, Any] | None,
) -> None:
    training = source != "local_val_a"
    expected_keys = {
        "support",
        "metrics",
        "role",
        "tier",
        "sampling_receipt",
        "shard_receipt",
        "source_contract" if training else "source_rgb_shard_sha256",
    }
    _exact_keys(value, expected_keys, f"{source} surface")
    expected_role = (
        "unique_complete_training_segment_pool"
        if training
        else "corrected_local_val_a_complete_segment_pool"
    )
    if value.get("role") != expected_role or value.get("tier") != SOURCE_SPECS[source]["tier"]:
        raise ValueError(f"{source} surface role or tier changed")
    support = value.get("support")
    metrics = value.get("metrics")
    shards = value.get("shard_receipt")
    if not all(isinstance(item, Mapping) for item in (support, metrics, shards)):
        raise ValueError(f"{source} surface body is malformed")
    _validate_support(support, source, session_ids)
    _validate_metric_bundle(metrics)
    _validate_decision_metric_counts(metrics, int(support["target_frames"]))
    _validate_shard_receipt(shards, source, session_ids)
    if training:
        if value.get("sampling_receipt") != sampling:
            raise ValueError(f"{source} surface sampling receipt changed")
        contract = value.get("source_contract")
        if not isinstance(contract, Mapping):
            raise ValueError(f"{source} source contract is missing")
        if expected_source_contract is None or contract != expected_source_contract:
            raise ValueError(f"{source} embedded source contract changed")
        _validate_source_contract(source, contract)
    else:
        expected_sampling = {
            "scheduled_draws": 0,
            "actual_draws": 0,
            "repeat_draws": 0,
            "note": "held-out local development surface; never sampled",
        }
        if value.get("sampling_receipt") != expected_sampling:
            raise ValueError("local val-A sampling receipt changed")
        if value.get("source_rgb_shard_sha256") != {
            LOCAL_VAL_A_ID: LOCAL_VAL_A_RGB_SHA256
        }:
            raise ValueError("local val-A RGB source binding changed")


def _validate_feature_view(value: Mapping[str, Any], arm: str) -> dict[str, list[str]]:
    keys = {
        "path",
        "receipt_path",
        "receipt_sha256",
        "hardlink_inventory_path",
        "hardlink_inventory_sha256",
        "hardlink_inventory_rows",
        "generated_files",
        "source_sessions",
        "local_val_a_sessions",
    }
    _exact_keys(value, keys, "feature-view receipt")
    if not all(
        isinstance(value.get(key), str)
        for key in ("path", "receipt_path", "hardlink_inventory_path")
    ):
        raise ValueError("feature-view paths are malformed")
    if Path(value["receipt_path"]).name != "blend_feature_view_receipt.json":
        raise ValueError("feature-view receipt path changed")
    if Path(value["hardlink_inventory_path"]).name != "hardlink_inventory.jsonl":
        raise ValueError("feature-view inventory path changed")
    if value.get("hardlink_inventory_rows") != 3_140:
        raise ValueError("feature-view inventory row count changed")
    for key in ("receipt_sha256", "hardlink_inventory_sha256"):
        if not _is_sha256(value.get(key)):
            raise ValueError(f"feature-view {key} is malformed")
    generated = value.get("generated_files")
    expected_generated = {
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
    if not isinstance(generated, Mapping) or set(generated) != expected_generated:
        raise ValueError("feature-view generated-file set changed")
    if not all(_is_sha256(digest) for digest in generated.values()):
        raise ValueError("feature-view generated-file hash is malformed")
    if generated["hardlink_inventory.jsonl"] != value["hardlink_inventory_sha256"]:
        raise ValueError("feature-view inventory hash binding changed")

    raw_sources = value.get("source_sessions")
    expected_sources = set(ARM_SPECS[arm]["sources"])
    if not isinstance(raw_sources, Mapping) or set(raw_sources) != expected_sources:
        raise ValueError("feature-view source membership set changed")
    result: dict[str, list[str]] = {}
    seen: set[str] = set()
    for source in ARM_SPECS[arm]["sources"]:
        ids = raw_sources[source]
        spec = SOURCE_SPECS[source]
        if (
            not isinstance(ids, list)
            or not all(isinstance(item, str) and item for item in ids)
            or len(ids) != spec["sessions"]
            or len(ids) != len(set(ids))
            or _line_list_sha256(ids) != spec["membership_sha256"]
        ):
            raise ValueError(f"feature-view membership changed for {source}")
        if seen.intersection(ids) or FORBIDDEN_SESSION_IDS.intersection(ids):
            raise ValueError("feature-view source overlap or forbidden membership")
        seen.update(ids)
        result[source] = list(ids)
    local_val = value.get("local_val_a_sessions")
    if local_val != [LOCAL_VAL_A_ID] or LOCAL_VAL_A_ID in seen:
        raise ValueError("feature-view local val-A membership changed or overlaps")
    result["local_val_a"] = [LOCAL_VAL_A_ID]
    return result


def _validate_local_gap(
    value: Mapping[str, Any],
    local_train: Mapping[str, Any],
    local_val: Mapping[str, Any],
) -> None:
    conventions = {
        "unweighted_bce": "val_a_minus_train",
        "average_precision": "train_minus_val_a",
        "state_f1_fixed_0_5": "train_minus_val_a",
    }
    _exact_keys(value, set(conventions), "local generalization gap")
    train_metrics = local_train["metrics"]
    val_metrics = local_val["metrics"]
    for metric_name, direction in conventions.items():
        row = value[metric_name]
        if not isinstance(row, Mapping):
            raise ValueError(f"local {metric_name} gap is malformed")
        _exact_keys(row, {"direction", "per_key", "macro"}, f"local {metric_name} gap")
        if row.get("direction") != direction:
            raise ValueError(f"local {metric_name} gap direction changed")
        per_key = row.get("per_key")
        if not isinstance(per_key, Mapping) or set(per_key) != set(KEY_ORDER):
            raise ValueError(f"local {metric_name} gap key set changed")
        expected_values: list[float] = []
        for key in KEY_ORDER:
            train = float(train_metrics[metric_name]["per_key"][key])
            val = float(val_metrics[metric_name]["per_key"][key])
            expected = val - train if direction == "val_a_minus_train" else train - val
            _close(per_key[key], expected, f"local {metric_name}.{key} gap")
            expected_values.append(expected)
        train_macro = float(train_metrics[metric_name]["macro"])
        val_macro = float(val_metrics[metric_name]["macro"])
        expected_macro = (
            val_macro - train_macro
            if direction == "val_a_minus_train"
            else train_macro - val_macro
        )
        _close(row.get("macro"), expected_macro, f"local {metric_name}.macro gap")
        _close(row.get("macro"), sum(expected_values) / len(expected_values), f"local {metric_name}.mean gap")


def _validate_scope(value: Mapping[str, Any]) -> None:
    _exact_keys(
        value,
        {"accessed", "not_accessed", "known_forbidden_session_ids", "forbidden_session_ids_accessed"},
        "scope guard",
    )
    if value.get("accessed") != [
        "unique complete training segment pools",
        "corrected own-v3 local val-A complete segment pool",
    ]:
        raise ValueError("diagnostic accessed-surface declaration changed")
    if set(value.get("not_accessed", [])) != {
        "B1",
        "val-B",
        "mapped-y4n",
        "sealed untouched test",
    }:
        raise ValueError("diagnostic not-accessed declaration changed")
    if set(value.get("known_forbidden_session_ids", [])) != FORBIDDEN_SESSION_IDS:
        raise ValueError("known forbidden-session declaration changed")
    if value.get("forbidden_session_ids_accessed") != []:
        raise ValueError("diagnostic declares forbidden-session access")


def validate_artifacts(
    repo: Path,
    report_path: Path,
    marker_path: Path,
    *,
    arm: str,
    diagnostic_commit: str,
) -> dict[str, Any]:
    """Validate two published JSON artifacts without opening model/data inputs."""

    if arm not in ARM_SPECS:
        raise ValueError("unknown provisional blend arm")
    spec = ARM_SPECS[arm]
    run_id = str(spec["run_id"])
    if report_path.name != f"{run_id}_final_memorization.json":
        raise ValueError("memorization report filename changed")
    if marker_path.name != f".{run_id}_final_memorization_done.json":
        raise ValueError("memorization marker filename changed")
    report = _json_object(report_path, "memorization report")
    marker = _json_object(marker_path, "memorization marker")
    _exact_keys(report, REPORT_KEYS, "memorization report")
    _exact_keys(marker, MARKER_KEYS, "memorization marker")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("memorization report schema changed")
    if marker.get("schema_version") != MARKER_SCHEMA_VERSION or marker.get("status") != "complete":
        raise ValueError("memorization marker schema or status changed")
    expected_identity = {
        "study_id": STUDY_ID,
        "arm": arm,
        "run_id": run_id,
        "surface": SURFACE,
        "weights": "final_state_dict",
    }
    for key, expected in expected_identity.items():
        if report.get(key) != expected or marker.get(key) != expected:
            raise ValueError(f"memorization report/marker {key} changed")
    try:
        timestamp = datetime.fromisoformat(str(report.get("evaluated_at")))
    except ValueError as error:
        raise ValueError("memorization evaluation timestamp is invalid") from error
    if timestamp.tzinfo is None:
        raise ValueError("memorization evaluation timestamp lacks a timezone")
    if Path(str(marker.get("report"))).resolve() != report_path.resolve():
        raise ValueError("memorization marker is bound to another report path")
    report_sha256 = sha256_file(report_path)
    if marker.get("report_sha256") != report_sha256:
        raise ValueError("memorization report hash differs from marker")

    contract = report.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("memorization contract receipt is missing")
    _exact_keys(contract, {"path", "sha256", "commit"}, "contract receipt")
    if (
        not str(contract.get("path", "")).endswith(CONTRACT_RELATIVE_PATH.as_posix())
        or contract.get("sha256") != CONTRACT_SHA256
        or contract.get("commit") != CONTRACT_COMMIT
        or marker.get("contract_sha256") != CONTRACT_SHA256
    ):
        raise ValueError("memorization contract provenance changed")
    diagnostic = report.get("diagnostic_source")
    if not isinstance(diagnostic, Mapping):
        raise ValueError("diagnostic source receipt is missing")
    _verify_diagnostic_source(repo, diagnostic_commit, diagnostic)
    frozen_source_contracts = _load_contract_sources(repo)

    run_receipt = report.get("run_receipt")
    if not isinstance(run_receipt, Mapping):
        raise ValueError("memorization run receipt is missing")
    _validate_run_receipt(run_receipt, arm)
    marker_bindings = {
        "training_implementation_git_commit": TRAINING_IMPLEMENTATION_COMMIT,
        "diagnostic_git_commit": diagnostic_commit,
        "diagnostic_module_sha256": diagnostic["sha256"],
        "config_sha256": spec["config_sha256"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "run_meta_sha256": spec["run_meta_sha256"],
        "training_log_sha256": spec["training_log_sha256"],
        "source_sampling_receipt_sha256": spec["source_sampling_receipt_sha256"],
        "selected_final_tensors_identical": True,
        "forbidden_surfaces_accessed": False,
    }
    for key, expected in marker_bindings.items():
        if marker.get(key) != expected:
            raise ValueError(f"memorization marker {key} changed")

    top_sampling = report.get("source_sampling_receipt")
    if top_sampling != run_receipt.get("source_sampling"):
        raise ValueError("top-level and run source-sampling receipts differ")
    feature_view = report.get("feature_view")
    if not isinstance(feature_view, Mapping):
        raise ValueError("feature-view receipt is missing")
    memberships = _validate_feature_view(feature_view, arm)
    if marker.get("feature_view_receipt_sha256") != feature_view.get("receipt_sha256"):
        raise ValueError("marker feature-view receipt binding changed")

    surfaces = report.get("surfaces")
    expected_surfaces = {*spec["sources"], "local_val_a"}
    if not isinstance(surfaces, Mapping) or set(surfaces) != expected_surfaces:
        raise ValueError("memorization surface set changed")
    sampling_sources = run_receipt["source_sampling"]["sources"]
    for source in spec["sources"]:
        if not isinstance(surfaces[source], Mapping):
            raise ValueError(f"{source} surface is not an object")
        _validate_surface(
            surfaces[source],
            source,
            memberships[source],
            sampling_sources[source],
            frozen_source_contracts.get(source),
        )
    local_val = surfaces["local_val_a"]
    if not isinstance(local_val, Mapping):
        raise ValueError("local val-A surface is not an object")
    _validate_surface(
        local_val,
        "local_val_a",
        memberships["local_val_a"],
        {},
        None,
    )
    gap = report.get("local_generalization_gap")
    if not isinstance(gap, Mapping):
        raise ValueError("local generalization gap is missing")
    _validate_local_gap(gap, surfaces["local"], local_val)

    scope = report.get("scope_guard")
    if not isinstance(scope, Mapping):
        raise ValueError("scope guard is missing")
    _validate_scope(scope)
    method = report.get("method")
    expected_method = {
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
    }
    if method != expected_method:
        raise ValueError("memorization method declaration changed")

    return {
        "status": "valid",
        "study_id": STUDY_ID,
        "arm": arm,
        "run_id": run_id,
        "report_sha256": report_sha256,
        "diagnostic_git_commit": diagnostic_commit,
        "diagnostic_module_sha256": diagnostic["sha256"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "forbidden_surfaces_accessed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--completion-marker", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARM_SPECS), required=True)
    parser.add_argument("--diagnostic-commit", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    receipt = validate_artifacts(
        args.repo,
        args.report,
        args.completion_marker,
        arm=args.arm,
        diagnostic_commit=args.diagnostic_commit,
    )
    print(json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
