"""Independently validate a completed oracle-window differential follow-up.

This is a post-run audit, not another decision surface.  It validates the
frozen configuration and every transitive input, reloads both final models,
replays their validation predictions exactly on MPS, recomputes the frozen
score, closes the report and completion-marker bindings, and only then writes
one non-overwriting supplementary receipt.  It never changes the experiment,
its artifacts, or its decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.oracle_window_differential_followup import (
    METADATA_NAMES,
    DifferentialCandidateModel,
    PixelOracleDataset,
    predict_probabilities,
)
from experiments.oracle_window_localization import HEAD_NAMES
from experiments.score_oracle_window_differential_followup import (
    load_differential_sidecar,
    score_followup,
)
from experiments.score_oracle_window_localization import (
    estimable_heads,
    load_prediction_sidecar,
)


AUDIT_SCHEMA_VERSION = "madeleine.oracle-window-differential-run-audit.v1"
CONFIG_SCHEMA_VERSION = "madeleine.oracle-window-differential-decision.v1"
RUN_SCHEMA_VERSION = "madeleine.oracle-window-differential-run.v1"
CHECKPOINT_SCHEMA_VERSION = "madeleine.oracle-window-differential-checkpoint.v1"
CACHE_RECEIPT_SCHEMA_VERSION = "madeleine.oracle-window-pixel-crops-complete.v1"
CACHE_MANIFEST_SCHEMA_VERSION = "madeleine.oracle-window-pixel-crops.v1"
BASE_MANIFEST_SCHEMA_VERSION = "madeleine.oracle-window-dataset.v1"
SCORE_SCHEMA_VERSION = "madeleine.oracle-window-differential-score.v1"
MARKER_SCHEMA_VERSION = "madeleine.oracle-window-differential-complete.v1"

FROZEN_CONFIG_SHA256 = (
    "cac63698326a3dbe1d64b8158a25a563aeb514294ec06ef295e6321d9fe338c8"
)
EXPECTED_SEED = 0
EXPECTED_EPOCHS = 20
EXPECTED_DEVICE = "mps"
EXPECTED_TRAIN_EXAMPLES = 4_554
EXPECTED_VALIDATION_EXAMPLES = 1_150
EXPECTED_VALIDATION_BLOCKS = 122
EXPECTED_ESTIMABLE_HEADS = 12

EXPECTED_RUN_INVENTORY = frozenset(
    {
        "ordered_pair_model.pt",
        "symmetric_pair_model.pt",
        "predictions.npz",
        "run_receipt.json",
        "training_log.json",
    }
)
RUN_RECEIPT_INVENTORY = frozenset(
    {
        "baseline_audit_path",
        "baseline_audit_sha256",
        "baseline_sidecar_path",
        "baseline_sidecar_sha256",
        "checkpoints",
        "config_path",
        "config_sha256",
        "configured_epochs",
        "device",
        "epochs",
        "final_weights_only",
        "implementation",
        "initial_state_sha256",
        "matched_batch_order",
        "matched_initialization",
        "pixel_cache_receipt_path",
        "pixel_cache_receipt_sha256",
        "prediction_sidecar_sha256",
        "schema_version",
        "seed",
        "status",
        "train_examples",
        "training_log_sha256",
        "validation_examples",
        "validation_used_for_training_or_selection",
    }
)
CHECKPOINT_INVENTORY = frozenset(
    {
        "schema_version",
        "config_sha256",
        "pixel_cache_receipt_sha256",
        "baseline_sidecar_sha256",
        "seed",
        "epochs",
        "initial_state_sha256",
        "arm",
        "model_state_dict",
    }
)
CACHE_RECEIPT_INVENTORY = frozenset(
    {
        "schema_version",
        "status",
        "published_output",
        "base_config_sha256",
        "base_dataset_manifest_sha256",
        "feature_receipt_sha256",
        "cache_manifest_sha256",
        "cache",
        "checks",
        "content_sha256",
    }
)
CACHE_MANIFEST_INVENTORY = frozenset(
    {
        "schema_version",
        "base_config_sha256",
        "base_dataset_manifest_sha256",
        "feature_receipt_sha256",
        "source_build_manifest_sha256",
        "crop_frames",
        "source_frame_size",
        "output_size",
        "downsampling",
        "color_order",
        "cache",
        "source_checks",
    }
)
CACHE_DIRECTORY_INVENTORY = frozenset(
    {"cache_manifest.json", "train.npz", "validation.npz"}
)
SCORE_INVENTORY = frozenset(
    {
        "schema_version",
        "status",
        "study_id",
        "scope",
        "config_sha256",
        "prediction_sidecar_sha256",
        "pixel_cache_receipt_sha256",
        "base_dataset_manifest_sha256",
        "bindings",
        "support",
        "chance",
        "arms",
        "primary_comparison",
        "attribution_comparisons",
        "decision_gate",
    }
)
MARKER_INVENTORY = frozenset(
    {
        "schema_version",
        "status",
        "study_id",
        "decision",
        "report",
        "config",
        "predictions",
        "run_receipt",
        "checkpoints",
        "pixel_cache_receipt",
        "content_sha256",
    }
)


def _json(path: Path, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, item in pairs:
            if name in result:
                raise ValueError(f"{label} contains a duplicate JSON key: {name}")
            result[name] = item
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains a non-finite JSON value: {value}")

    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same_path(observed: object, expected: Path, label: str) -> None:
    _require(isinstance(observed, str), f"{label} path is not a string")
    _require(
        Path(observed).resolve() == Path(expected).resolve(),
        f"{label} path changed: {observed}",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes())
    return digest.hexdigest()


def _state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _arrays_equal(observed: np.ndarray, expected: np.ndarray) -> bool:
    if observed.dtype != expected.dtype or observed.shape != expected.shape:
        return False
    if np.issubdtype(expected.dtype, np.floating):
        return bool(np.array_equal(observed, expected, equal_nan=True))
    return bool(np.array_equal(observed, expected))


def _require_array_equal(
    label: str, observed: np.ndarray, expected: np.ndarray
) -> None:
    observed = np.asarray(observed)
    expected = np.asarray(expected)
    if _arrays_equal(observed, expected):
        return
    detail = (
        f"observed dtype/shape={observed.dtype}/{observed.shape}, "
        f"expected={expected.dtype}/{expected.shape}"
    )
    if (
        observed.shape == expected.shape
        and np.issubdtype(observed.dtype, np.floating)
        and np.issubdtype(expected.dtype, np.floating)
    ):
        finite = np.isfinite(observed) & np.isfinite(expected)
        maximum = (
            float(np.max(np.abs(observed[finite] - expected[finite])))
            if np.any(finite)
            else None
        )
        detail += f", maximum finite absolute difference={maximum}"
    raise ValueError(f"{label} changed: {detail}")


def _require_regular_file(path: Path, label: str) -> None:
    _require(
        path.is_file() and not path.is_symlink(),
        f"{label} is not a regular non-symlink file: {path}",
    )


def _require_canonical_content_hash(value: Mapping[str, Any], label: str) -> str:
    without_hash = dict(value)
    content_sha = without_hash.pop("content_sha256", None)
    _require(isinstance(content_sha, str), f"{label} lacks a content hash")
    _require(
        content_sha == _canonical_json_sha256(without_hash),
        f"{label} content hash changed",
    )
    return content_sha


def _require_deterministic_json(path: Path, value: Mapping[str, Any], label: str) -> None:
    expected = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _require(path.read_bytes() == expected, f"{label} serialization changed")


def validate_frozen_config(config_path: Path, repo: Path) -> dict[str, Any]:
    """Require the externally recorded preregistration and all declared bytes."""

    _require_regular_file(config_path, "decision config")
    observed_config_sha = _sha256_file(config_path)
    _require(
        observed_config_sha == FROZEN_CONFIG_SHA256,
        f"decision config hash changed: {observed_config_sha}",
    )
    config = _json(config_path, "decision config")
    _require(config.get("schema_version") == CONFIG_SCHEMA_VERSION, "config schema changed")
    _require(
        config.get("status") == "preregistered_before_validation_inference",
        "config is not the frozen preregistration",
    )
    _require(config.get("study_id") == "own_v3_oracle_window_differential_w16_s0", "study ID changed")
    dataset = config.get("dataset")
    training = config.get("training")
    evaluation = config.get("evaluation")
    model = config.get("model")
    _require(isinstance(dataset, Mapping), "config lacks dataset policy")
    _require(isinstance(training, Mapping), "config lacks training policy")
    _require(isinstance(evaluation, Mapping), "config lacks evaluation policy")
    _require(isinstance(model, Mapping), "config lacks model policy")
    _require(int(dataset["candidate_width"]) == 16, "candidate width changed")
    _require(int(dataset["context_halo"]) == 8, "context halo changed")
    _require(int(dataset["crop_frames"]) == 32, "crop length changed")
    _require(int(dataset["cached_frame_size"]) == 32, "cached frame size changed")
    support = dataset.get("expected_support")
    _require(isinstance(support, Mapping), "config lacks expected support")
    _require(int(support["training_examples"]) == EXPECTED_TRAIN_EXAMPLES, "training support changed")
    _require(int(support["validation_examples"]) == EXPECTED_VALIDATION_EXAMPLES, "validation support changed")
    _require(int(support["validation_blocks"]) == EXPECTED_VALIDATION_BLOCKS, "validation blocks changed")
    _require(int(support["estimable_heads"]) == EXPECTED_ESTIMABLE_HEADS, "estimable-head count changed")
    _require(int(training["seed"]) == EXPECTED_SEED, "training seed changed")
    _require(int(training["epochs"]) == EXPECTED_EPOCHS, "training endpoint changed")
    _require(training.get("device") == EXPECTED_DEVICE, "frozen device is not MPS")
    _require(int(model["embedding_dim"]) == 64, "model embedding dimension changed")
    _require(int(evaluation["bootstrap_replicates"]) == 5_000, "bootstrap count changed")

    declaration = config.get("implementation")
    _require(isinstance(declaration, Mapping), "config lacks implementation authority")
    expected_hashes = declaration.get("sha256")
    _require(isinstance(expected_hashes, Mapping) and bool(expected_hashes), "implementation hashes are missing")
    repo = repo.resolve()
    observed_hashes: dict[str, str] = {}
    for relative, expected_sha in expected_hashes.items():
        _require(isinstance(relative, str) and isinstance(expected_sha, str), "implementation hash map is malformed")
        path = (repo / relative).resolve()
        _require(path.is_relative_to(repo), f"implementation path escapes repo: {relative}")
        _require_regular_file(path, f"implementation {relative}")
        observed_sha = _sha256_file(path)
        _require(observed_sha == expected_sha, f"frozen implementation changed: {relative}")
        observed_hashes[relative] = observed_sha
    git_head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    freeze_head = declaration.get("git_head_at_freeze")
    _require(git_head == freeze_head, "Git HEAD differs from the frozen implementation HEAD")
    implementation_receipt = {
        "git_head_at_execution": git_head,
        "git_head_at_freeze": freeze_head,
        "relevant_file_sha256": observed_hashes,
        "authority": "exact relevant working bytes; no commit created for this study",
    }
    return {
        "config": config,
        "config_sha256": observed_config_sha,
        "implementation": implementation_receipt,
    }


def _load_cache_split(
    *,
    path: Path,
    split: str,
    expected_examples: int,
    retain_rgb: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _require_regular_file(path, f"pixel cache {split}")
    with np.load(path, allow_pickle=False) as archive:
        expected_names = {"rgb", *METADATA_NAMES}
        _require(set(archive.files) == expected_names, f"pixel cache {split} inventory changed")
        rgb = np.asarray(archive["rgb"])
        metadata = {name: np.asarray(archive[name]) for name in METADATA_NAMES}
    expected_shape = (expected_examples, 32, 32, 32, 3)
    _require(rgb.dtype == np.uint8 and rgb.shape == expected_shape, f"pixel cache {split} RGB schema changed")
    expected_dtypes = {
        "run_index": np.dtype("int32"),
        "array_index": np.dtype("int64"),
        "engine_frame_idx": np.dtype("int64"),
        "head_index": np.dtype("int16"),
        "key_index": np.dtype("int8"),
        "event_type_index": np.dtype("int8"),
        "true_offset": np.dtype("int8"),
        "crop_start": np.dtype("int64"),
    }
    for name, value in metadata.items():
        _require(value.shape == (expected_examples,), f"pixel cache {split} metadata shape changed: {name}")
        if name in expected_dtypes:
            _require(value.dtype == expected_dtypes[name], f"pixel cache {split} metadata dtype changed: {name}")
        elif name in {"session_id", "block_id"}:
            _require(value.dtype.kind == "U", f"pixel cache {split} string dtype changed: {name}")
    heads = metadata["head_index"]
    truth = metadata["true_offset"]
    _require(np.all((heads >= 0) & (heads < len(HEAD_NAMES))), f"pixel cache {split} head changed")
    _require(np.all((truth >= 0) & (truth < 16)), f"pixel cache {split} offset changed")
    _require_array_equal(f"pixel cache {split} key identity", metadata["key_index"], (heads % 7).astype(np.int8))
    _require_array_equal(f"pixel cache {split} event identity", metadata["event_type_index"], (heads // 7).astype(np.int8))
    target_index = metadata["crop_start"] + 8 + truth.astype(np.int64)
    _require_array_equal(f"pixel cache {split} crop/offset alignment", metadata["array_index"], target_index)
    identity = list(
        zip(
            metadata["session_id"].astype(str).tolist(),
            metadata["run_index"].astype(np.int64).tolist(),
            metadata["array_index"].astype(np.int64).tolist(),
            heads.astype(np.int64).tolist(),
            truth.astype(np.int64).tolist(),
            strict=True,
        )
    )
    _require(len(identity) == len(set(identity)), f"pixel cache {split} repeats an example identity")
    evidence = {
        "bytes": path.stat().st_size,
        "file_sha256": _sha256_file(path),
        "rgb_sha256": _array_sha256(rgb),
        "examples": expected_examples,
        "rgb_shape": list(rgb.shape),
        "rgb_dtype": str(rgb.dtype),
    }
    arrays = dict(metadata)
    if retain_rgb:
        arrays["rgb"] = rgb
    return arrays, evidence


def validate_cache_receipt(
    *,
    cache_root: Path,
    receipt_path: Path,
    base_manifest_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the content-bound cache receipt, manifest, files, and support."""

    _require_regular_file(receipt_path, "pixel-cache receipt")
    observed_receipt_sha = _sha256_file(receipt_path)
    dataset = config["dataset"]
    _require(observed_receipt_sha == dataset["pixel_cache_receipt_sha256"], "pixel-cache receipt hash changed")
    receipt = _json(receipt_path, "pixel-cache receipt")
    _require(set(receipt) == CACHE_RECEIPT_INVENTORY, "pixel-cache receipt inventory changed")
    _require(receipt.get("schema_version") == CACHE_RECEIPT_SCHEMA_VERSION, "pixel-cache receipt schema changed")
    _require(receipt.get("status") == "complete", "pixel-cache receipt is incomplete")
    _same_path(receipt.get("published_output"), cache_root, "pixel-cache receipt")
    content_sha = _require_canonical_content_hash(receipt, "pixel-cache receipt")
    _require(content_sha == dataset["pixel_cache_content_sha256"], "pixel-cache content identity changed")
    _require(receipt.get("base_dataset_manifest_sha256") == _sha256_file(base_manifest_path), "pixel-cache/base-manifest binding changed")
    checks = receipt.get("checks")
    _require(isinstance(checks, Mapping) and bool(checks) and all(value is True for value in checks.values()), "pixel-cache checks are not all true")

    cache_root = cache_root.resolve()
    _require(cache_root.is_dir() and not cache_root.is_symlink(), "pixel-cache root is not a regular directory")
    _require({path.name for path in cache_root.iterdir()} == CACHE_DIRECTORY_INVENTORY, "pixel-cache directory inventory changed")
    for name in CACHE_DIRECTORY_INVENTORY:
        _require_regular_file(cache_root / name, f"pixel-cache artifact {name}")

    manifest_path = cache_root / "cache_manifest.json"
    manifest_sha = _sha256_file(manifest_path)
    _require(manifest_sha == receipt["cache_manifest_sha256"], "pixel-cache manifest hash changed")
    manifest = _json(manifest_path, "pixel-cache manifest")
    _require(set(manifest) == CACHE_MANIFEST_INVENTORY, "pixel-cache manifest inventory changed")
    _require(manifest.get("schema_version") == CACHE_MANIFEST_SCHEMA_VERSION, "pixel-cache manifest schema changed")
    _require(manifest.get("base_dataset_manifest_sha256") == _sha256_file(base_manifest_path), "cache manifest/base-manifest binding changed")
    _require(manifest.get("source_build_manifest_sha256") == dataset["source_build_manifest_sha256"], "cache source-build identity changed")
    _require(int(manifest.get("crop_frames", -1)) == 32, "cache crop length changed")
    _require(int(manifest.get("source_frame_size", -1)) == 128, "cache source frame size changed")
    _require(int(manifest.get("output_size", -1)) == 32, "cache output size changed")
    _require(manifest.get("color_order") == "source RGB preserved", "cache color order changed")

    train_arrays, train_evidence = _load_cache_split(
        path=cache_root / "train.npz",
        split="train",
        expected_examples=EXPECTED_TRAIN_EXAMPLES,
        retain_rgb=False,
    )
    validation_arrays, validation_evidence = _load_cache_split(
        path=cache_root / "validation.npz",
        split="validation",
        expected_examples=EXPECTED_VALIDATION_EXAMPLES,
        retain_rgb=True,
    )
    _require(manifest.get("cache") == {"train": train_evidence, "validation": validation_evidence}, "cache manifest split evidence changed")
    expected_files = {
        "cache_manifest.json": {"bytes": manifest_path.stat().st_size, "sha256": manifest_sha},
        "train.npz": {"bytes": train_evidence["bytes"], "sha256": train_evidence["file_sha256"]},
        "validation.npz": {"bytes": validation_evidence["bytes"], "sha256": validation_evidence["file_sha256"]},
    }
    _require(receipt.get("cache") == expected_files, "pixel-cache receipt file bindings changed")
    source_checks = manifest.get("source_checks")
    _require(isinstance(source_checks, Mapping) and bool(source_checks), "cache source checks are missing")
    observed_sessions = set(train_arrays["session_id"].astype(str)) | set(validation_arrays["session_id"].astype(str))
    _require(set(source_checks) == observed_sessions, "cache source-check session inventory changed")
    for session_id, row in source_checks.items():
        _require(isinstance(row, Mapping), f"cache source check is malformed: {session_id}")
        _require(row.get("masked_regions") == ["frame_index_strip", "input_overlay"], f"cache answer-key masks changed: {session_id}")
        _require(row.get("supervision_equal_to_features") is True, f"cache supervision check failed: {session_id}")
        for field in ("frames_sha256", "source_npz_sha256"):
            value = row.get(field)
            _require(isinstance(value, str) and len(value) == 64, f"cache source hash changed: {session_id}:{field}")
    return {
        "receipt": receipt,
        "receipt_sha256": observed_receipt_sha,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha,
        "train_arrays": train_arrays,
        "validation_arrays": validation_arrays,
        "split_evidence": {"train": train_evidence, "validation": validation_evidence},
    }


def validate_base_manifest_and_support(
    *,
    manifest_path: Path,
    config: Mapping[str, Any],
    train_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    _require_regular_file(manifest_path, "base dataset manifest")
    observed_sha = _sha256_file(manifest_path)
    _require(observed_sha == config["dataset"]["base_dataset_manifest_sha256"], "base dataset manifest hash changed")
    manifest = _json(manifest_path, "base dataset manifest")
    _require(manifest.get("schema_version") == BASE_MANIFEST_SCHEMA_VERSION, "base dataset manifest schema changed")
    _require(manifest.get("head_names") == list(HEAD_NAMES), "base manifest head order changed")
    _require(int(manifest.get("train_examples", -1)) == EXPECTED_TRAIN_EXAMPLES, "base manifest training support changed")
    _require(int(manifest.get("validation_examples", -1)) == EXPECTED_VALIDATION_EXAMPLES, "base manifest validation support changed")
    _require(int(manifest.get("validation_block_count", -1)) == EXPECTED_VALIDATION_BLOCKS, "base manifest validation blocks changed")

    train_sessions = [str(value) for value in config["dataset"]["train_sessions"]]
    val_sessions = [str(value) for value in config["dataset"]["validation_sessions"]]
    forbidden = set(str(value) for value in config["dataset"]["forbidden_sessions"])
    observed_train_sessions = set(train_arrays["session_id"].astype(str))
    observed_val_sessions = set(validation_arrays["session_id"].astype(str))
    _require(observed_train_sessions <= set(train_sessions), "pixel cache contains an undeclared training session")
    _require(observed_val_sessions == set(val_sessions), "pixel cache validation sessions changed")
    _require(not ((observed_train_sessions | observed_val_sessions) & forbidden), "forbidden session reached pixel cache")

    train_heads = train_arrays["head_index"].astype(np.int64)
    train_session_values = train_arrays["session_id"].astype(str)
    actual_train_counts = {
        session: {
            name: int(np.sum((train_session_values == session) & (train_heads == index)))
            for index, name in enumerate(HEAD_NAMES)
        }
        for session in train_sessions
    }
    _require(actual_train_counts == manifest.get("train_counts_by_session_head"), "training per-session/head support changed")
    val_heads = validation_arrays["head_index"].astype(np.int64)
    actual_val_counts = {
        name: int(np.sum(val_heads == index)) for index, name in enumerate(HEAD_NAMES)
    }
    _require(actual_val_counts == manifest.get("val_counts_by_head"), "validation per-head support changed")
    truth = validation_arrays["true_offset"].astype(np.int64)
    _require(np.bincount(truth, minlength=16).tolist() == manifest.get("validation_offset_counts"), "validation offset support changed")
    actual_blocks = len(set(validation_arrays["block_id"].astype(str).tolist()))
    _require(actual_blocks == EXPECTED_VALIDATION_BLOCKS, "validation block identities changed")
    estimable = estimable_heads(manifest, config["decision_gate"]["estimability"])
    _require(len(estimable) == EXPECTED_ESTIMABLE_HEADS, "estimable-head support changed")
    return {"manifest": manifest, "sha256": observed_sha, "estimable_heads": estimable}


def validate_baseline(
    *,
    sidecar_path: Path,
    audit_path: Path,
    config: Mapping[str, Any],
    validation_arrays: Mapping[str, np.ndarray],
    differential_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    baseline_config = config["frozen_feature_baseline"]
    _require_regular_file(sidecar_path, "frozen baseline sidecar")
    _require_regular_file(audit_path, "frozen baseline audit")
    sidecar_sha = _sha256_file(sidecar_path)
    audit_sha = _sha256_file(audit_path)
    _require(sidecar_sha == baseline_config["prediction_sidecar_sha256"], "frozen baseline sidecar hash changed")
    _require(audit_sha == baseline_config["audit_receipt_sha256"], "frozen baseline audit hash changed")
    audit = _json(audit_path, "frozen baseline audit")
    _require(
        set(audit)
        == {
            "schema_version",
            "status",
            "audited_at_utc",
            "study_id",
            "scope",
            "validator",
            "environment",
            "bindings",
            "reproduction",
            "run_contract",
            "decision",
            "content_sha256",
        },
        "frozen baseline audit inventory changed",
    )
    _require(
        audit.get("schema_version") == "madeleine.oracle-window-run-audit.v1",
        "frozen baseline audit schema changed",
    )
    audit_content_sha = _require_canonical_content_hash(audit, "frozen baseline audit")
    _require(audit_content_sha == baseline_config["audit_content_sha256"], "frozen baseline audit content changed")
    _require(audit.get("status") == "complete", "frozen baseline audit is incomplete")
    decision = audit.get("decision")
    _require(isinstance(decision, Mapping), "frozen baseline audit decision is missing")
    _require(decision.get("passed") is False and decision.get("unchanged_by_audit") is True, "frozen baseline decision changed")
    bindings = audit.get("bindings")
    _require(isinstance(bindings, Mapping) and bool(bindings), "frozen baseline audit bindings are missing")
    for name, binding in bindings.items():
        _require(isinstance(binding, Mapping), f"frozen baseline binding is malformed: {name}")
        path_value = binding.get("path")
        expected_sha = binding.get("sha256")
        _require(isinstance(path_value, str) and isinstance(expected_sha, str), f"frozen baseline binding is incomplete: {name}")
        bound_path = Path(path_value)
        _require_regular_file(bound_path, f"frozen baseline binding {name}")
        _require(_sha256_file(bound_path) == expected_sha, f"frozen baseline binding hash changed: {name}")
    prediction_binding = bindings.get("prediction_sidecar")
    _require(isinstance(prediction_binding, Mapping), "frozen baseline prediction binding is missing")
    _same_path(prediction_binding.get("path"), sidecar_path, "frozen baseline prediction")
    _require(prediction_binding.get("sha256") == sidecar_sha, "frozen baseline prediction audit binding changed")
    validator_binding = audit.get("validator")
    _require(isinstance(validator_binding, Mapping), "frozen baseline validator binding is missing")
    validator_path = Path(str(validator_binding.get("path")))
    _require_regular_file(validator_path, "frozen baseline validator")
    _require(_sha256_file(validator_path) == validator_binding.get("sha256"), "frozen baseline validator changed")

    prior = config["prior_decision"]
    _require(audit_sha == prior["audit_receipt_sha256"], "prior-decision audit hash changed")
    _require(audit_content_sha == prior["audit_content_sha256"], "prior-decision audit content changed")
    _require(bindings["score_report"]["sha256"] == prior["score_report_sha256"], "prior score-report identity changed")
    _require(bindings["completion_marker"]["sha256"] == prior["completion_marker_sha256"], "prior marker identity changed")

    baseline = load_prediction_sidecar(sidecar_path, width=16)
    for name in METADATA_NAMES:
        _require_array_equal(f"baseline/cache identity {name}", baseline[name], validation_arrays[name])
        _require_array_equal(f"baseline/differential identity {name}", baseline[name], differential_arrays[name])
    _require_array_equal(
        "frozen baseline conditional predictions",
        baseline["conditional_prob"],
        differential_arrays["feature_conditional_prob"],
    )
    return {
        "sidecar": baseline,
        "sidecar_sha256": sidecar_sha,
        "audit": audit,
        "audit_sha256": audit_sha,
        "audit_content_sha256": audit_content_sha,
    }


def validate_training_log(path: Path, *, epochs: int) -> dict[str, Any]:
    _require_regular_file(path, "training log")
    value = _json(path, "training log")
    expected_inventory = {
        "ordered_pair",
        "symmetric_pair",
        "fixed_final_epoch",
        "configured_final_epoch",
        "validation_used_for_training_or_selection",
        "matched_batch_order",
    }
    _require(set(value) == expected_inventory, "training log inventory changed")
    _require(
        int(value["fixed_final_epoch"])
        == int(value["configured_final_epoch"])
        == epochs,
        "training log endpoint changed",
    )
    _require(value["validation_used_for_training_or_selection"] is False, "validation was used for training or selection")
    _require(value["matched_batch_order"] is True, "training log does not prove matched batch order")
    for arm in ("ordered_pair", "symmetric_pair"):
        rows = value[arm]
        _require(isinstance(rows, list) and len(rows) == epochs, f"{arm} training-log length changed")
        for index, row in enumerate(rows, start=1):
            _require(isinstance(row, Mapping) and set(row) == {"epoch", "loss"}, f"{arm} training-log row changed")
            _require(float(row["epoch"]) == float(index), f"{arm} training-log epoch order changed")
            loss = float(row["loss"])
            _require(np.isfinite(loss) and loss >= 0.0, f"{arm} training loss is invalid")
    return value


def validate_run_receipt(
    *,
    receipt_path: Path,
    config_path: Path,
    cache_receipt_path: Path,
    baseline_sidecar_path: Path,
    baseline_audit_path: Path,
    predictions_path: Path,
    training_log_path: Path,
    checkpoint_paths: Mapping[str, Path],
    config: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    _require_regular_file(receipt_path, "run receipt")
    receipt = _json(receipt_path, "run receipt")
    _require(set(receipt) == RUN_RECEIPT_INVENTORY, "run receipt inventory changed")
    _require(receipt.get("schema_version") == RUN_SCHEMA_VERSION, "run receipt schema changed")
    _require(receipt.get("status") == "predictions_complete_unscored", "run receipt status changed")
    _same_path(receipt.get("config_path"), config_path, "run config")
    _same_path(receipt.get("pixel_cache_receipt_path"), cache_receipt_path, "run cache receipt")
    _same_path(receipt.get("baseline_sidecar_path"), baseline_sidecar_path, "run baseline sidecar")
    _same_path(receipt.get("baseline_audit_path"), baseline_audit_path, "run baseline audit")
    expected_scalars = {
        "config_sha256": _sha256_file(config_path),
        "pixel_cache_receipt_sha256": _sha256_file(cache_receipt_path),
        "baseline_sidecar_sha256": _sha256_file(baseline_sidecar_path),
        "baseline_audit_sha256": _sha256_file(baseline_audit_path),
        "prediction_sidecar_sha256": _sha256_file(predictions_path),
        "training_log_sha256": _sha256_file(training_log_path),
        "seed": EXPECTED_SEED,
        "epochs": EXPECTED_EPOCHS,
        "configured_epochs": EXPECTED_EPOCHS,
        "device": EXPECTED_DEVICE,
        "train_examples": EXPECTED_TRAIN_EXAMPLES,
        "validation_examples": EXPECTED_VALIDATION_EXAMPLES,
        "final_weights_only": True,
        "validation_used_for_training_or_selection": False,
        "matched_initialization": True,
        "matched_batch_order": True,
        "implementation": dict(implementation),
    }
    for name, expected in expected_scalars.items():
        _require(receipt.get(name) == expected, f"run receipt changed: {name}")
    initial_sha = receipt.get("initial_state_sha256")
    _require(isinstance(initial_sha, str) and len(initial_sha) == 64, "run initial-state hash changed")
    checkpoint_receipts = receipt.get("checkpoints")
    _require(isinstance(checkpoint_receipts, Mapping) and set(checkpoint_receipts) == set(checkpoint_paths), "run checkpoint inventory changed")
    for arm, path in checkpoint_paths.items():
        binding = checkpoint_receipts[arm]
        _require(isinstance(binding, Mapping) and set(binding) == {"path", "sha256", "model_state_sha256"}, f"run checkpoint binding changed: {arm}")
        _same_path(binding.get("path"), path, f"run checkpoint {arm}")
        _require(binding.get("sha256") == _sha256_file(path), f"run checkpoint hash changed: {arm}")
        state_sha = binding.get("model_state_sha256")
        _require(isinstance(state_sha, str) and len(state_sha) == 64, f"run checkpoint state hash changed: {arm}")
    return receipt


def load_final_checkpoint(
    *,
    path: Path,
    arm: str,
    config: Mapping[str, Any],
    run_receipt: Mapping[str, Any],
) -> tuple[DifferentialCandidateModel, dict[str, Any]]:
    _require_regular_file(path, f"{arm} checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping), f"{arm} checkpoint is not a mapping")
    _require(set(payload) == CHECKPOINT_INVENTORY, f"{arm} checkpoint inventory changed")
    expected_scalars = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_sha256": run_receipt["config_sha256"],
        "pixel_cache_receipt_sha256": run_receipt["pixel_cache_receipt_sha256"],
        "baseline_sidecar_sha256": run_receipt["baseline_sidecar_sha256"],
        "seed": EXPECTED_SEED,
        "epochs": EXPECTED_EPOCHS,
        "initial_state_sha256": run_receipt["initial_state_sha256"],
        "arm": arm,
    }
    for name, expected in expected_scalars.items():
        _require(payload.get(name) == expected, f"{arm} checkpoint changed: {name}")
    model = DifferentialCandidateModel(embedding_dim=int(config["model"]["embedding_dim"]))
    state = payload.get("model_state_dict")
    reference = model.state_dict()
    _require(isinstance(state, Mapping) and set(state) == set(reference), f"{arm} checkpoint state inventory changed")
    tensor_schema: dict[str, Any] = {}
    for name, expected in reference.items():
        value = state[name]
        _require(isinstance(value, torch.Tensor), f"{arm} checkpoint tensor is malformed: {name}")
        _require(value.dtype == expected.dtype and value.shape == expected.shape, f"{arm} checkpoint tensor schema changed: {name}")
        _require(bool(torch.isfinite(value).all()), f"{arm} checkpoint tensor is non-finite: {name}")
        tensor_schema[name] = {"dtype": str(value.dtype), "shape": list(value.shape)}
    model.load_state_dict(state, strict=True)
    state_sha = _state_dict_sha256(model)
    run_binding = run_receipt["checkpoints"][arm]
    checkpoint_sha = _sha256_file(path)
    _require(checkpoint_sha == run_binding["sha256"], f"{arm} checkpoint file hash changed")
    _require(state_sha == run_binding["model_state_sha256"], f"{arm} checkpoint state hash changed")
    return model, {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": checkpoint_sha,
        "state_dict_sha256": state_sha,
        "schema_version": payload["schema_version"],
        "arm": arm,
        "seed": int(payload["seed"]),
        "epochs": int(payload["epochs"]),
        "config_sha256": str(payload["config_sha256"]),
        "pixel_cache_receipt_sha256": str(payload["pixel_cache_receipt_sha256"]),
        "baseline_sidecar_sha256": str(payload["baseline_sidecar_sha256"]),
        "initial_state_sha256": str(payload["initial_state_sha256"]),
        "strict_reload": True,
        "all_tensors_finite": True,
        "tensor_schema": tensor_schema,
    }


def validate_score_report(
    *,
    report_path: Path,
    predictions_path: Path,
    config_path: Path,
    cache_receipt_path: Path,
    base_manifest_path: Path,
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    base_manifest: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    _require_regular_file(report_path, "score report")
    report = _json(report_path, "score report")
    _require(set(report) == SCORE_INVENTORY, "score report inventory changed")
    _require(report.get("schema_version") == SCORE_SCHEMA_VERSION, "score report schema changed")
    _require(report.get("status") == "complete", "score report is incomplete")
    _require(report.get("study_id") == config["study_id"], "score report study changed")
    expected_hashes = {
        "config_sha256": _sha256_file(config_path),
        "prediction_sidecar_sha256": _sha256_file(predictions_path),
        "pixel_cache_receipt_sha256": _sha256_file(cache_receipt_path),
        "base_dataset_manifest_sha256": _sha256_file(base_manifest_path),
    }
    for name, expected in expected_hashes.items():
        _require(report.get(name) == expected, f"score report binding changed: {name}")
    _require(report.get("bindings") == dict(bindings), "score report artifact bindings changed")
    recomputed = score_followup(
        arrays=arrays,
        config=config,
        base_manifest=base_manifest,
        bindings=bindings,
        config_sha256=expected_hashes["config_sha256"],
        predictions_sha256=expected_hashes["prediction_sidecar_sha256"],
        cache_receipt_sha256=expected_hashes["pixel_cache_receipt_sha256"],
        base_manifest_sha256=expected_hashes["base_dataset_manifest_sha256"],
    )
    _require(recomputed == report, "fixed-policy differential score does not reproduce exactly")
    primary = report["primary_comparison"]["paired_block_bootstrap"]
    attribution = report["attribution_comparisons"]
    bootstraps = {
        "primary": primary,
        "symmetric-vs-feature": attribution["symmetric_pair_minus_frozen_feature"]["paired_block_bootstrap"],
        "ordered-vs-symmetric": attribution["ordered_pair_minus_symmetric_pair"]["paired_block_bootstrap"],
    }
    for name, bootstrap in bootstraps.items():
        _require(
            int(bootstrap["replicates_requested"])
            == int(bootstrap["replicates_valid"])
            == 5_000,
            f"score report bootstrap support changed: {name}",
        )
    decision_gate = report["decision_gate"]
    _require(
        decision_gate.get("original_phase_2_decision_remains_rejected") is True,
        "score report reopened the rejected cascade",
    )
    ordered_passed = bool(decision_gate["ordered_pair_vs_frozen_feature"]["passed"])
    symmetric_passed = bool(decision_gate["symmetric_pair_vs_frozen_feature"]["passed"])
    attribution_passed = bool(decision_gate["ordered_pair_vs_symmetric_pair"]["passed"])
    _require(
        decision_gate["passed_primary_pixel_rescue"] is ordered_passed,
        "score primary-gate summary changed",
    )
    _require(
        decision_gate["passed_differential_attribution"]
        is (ordered_passed and attribution_passed),
        "score attribution-gate summary changed",
    )
    if ordered_passed:
        expected_decision = (
            "differential_specific_pixel_rescue_original_phase_2_still_rejected"
            if attribution_passed
            else "learned_pixel_rescue_not_differentially_attributed_original_phase_2_still_rejected"
        )
    elif symmetric_passed:
        expected_decision = "appearance_only_pixel_rescue_pair_hypothesis_rejected_no_phase_2"
    else:
        expected_decision = "no_bounded_pixel_rescue_no_phase_2"
    _require(
        decision_gate["decision"] == expected_decision,
        "score hierarchical decision changed",
    )
    _require_deterministic_json(report_path, report, "score report")
    return report


def validate_completion_marker(
    *,
    marker_path: Path,
    report_path: Path,
    config_path: Path,
    predictions_path: Path,
    run_receipt_path: Path,
    checkpoint_paths: Mapping[str, Path],
    cache_receipt_path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    _require_regular_file(marker_path, "completion marker")
    marker = _json(marker_path, "completion marker")
    _require(set(marker) == MARKER_INVENTORY, "completion marker inventory changed")
    _require(marker.get("schema_version") == MARKER_SCHEMA_VERSION, "completion marker schema changed")
    _require(marker.get("status") == "complete", "completion marker is incomplete")
    _require(marker.get("study_id") == report.get("study_id"), "completion marker study changed")
    _require(marker.get("decision") == report["decision_gate"]["decision"], "completion marker decision changed")
    for name, path in (
        ("report", report_path),
        ("config", config_path),
        ("predictions", predictions_path),
        ("run_receipt", run_receipt_path),
        ("pixel_cache_receipt", cache_receipt_path),
    ):
        binding = marker.get(name)
        _require(isinstance(binding, Mapping) and set(binding) == {"path", "sha256"}, f"completion marker binding changed: {name}")
        _same_path(binding.get("path"), path, f"completion marker {name}")
        _require(binding.get("sha256") == _sha256_file(path), f"completion marker hash changed: {name}")
    checkpoint_bindings = marker.get("checkpoints")
    _require(isinstance(checkpoint_bindings, Mapping) and set(checkpoint_bindings) == set(checkpoint_paths), "completion marker checkpoint inventory changed")
    for arm, path in checkpoint_paths.items():
        binding = checkpoint_bindings[arm]
        _require(isinstance(binding, Mapping) and set(binding) == {"path", "sha256"}, f"completion marker checkpoint binding changed: {arm}")
        _same_path(binding.get("path"), path, f"completion marker checkpoint {arm}")
        _require(binding.get("sha256") == _sha256_file(path), f"completion marker checkpoint hash changed: {arm}")
        _require(report["bindings"]["checkpoints"][arm]["sha256"] == binding["sha256"], f"report/marker checkpoint closure changed: {arm}")
    _require(report["bindings"]["run_receipt_sha256"] == marker["run_receipt"]["sha256"], "report/marker run-receipt closure changed")
    _require(report["pixel_cache_receipt_sha256"] == marker["pixel_cache_receipt"]["sha256"], "report/marker cache closure changed")
    _require_canonical_content_hash(marker, "completion marker")
    _require_deterministic_json(marker_path, marker, "completion marker")
    return marker


def _write_json_exclusive_atomic(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite differential audit receipt: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _require(_json(temporary, "serialized differential audit") == value, "differential audit changed on reload")
        os.link(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _finalize_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    _require("content_sha256" not in result, "differential audit already has a content hash")
    result["content_sha256"] = _canonical_json_sha256(result)
    return result


def validate_oracle_window_differential_run(
    *,
    repo: Path,
    run: Path,
    config_path: Path,
    cache_root: Path,
    cache_receipt_path: Path,
    baseline_sidecar_path: Path,
    baseline_audit_path: Path,
    base_manifest_path: Path,
    report_path: Path,
    marker_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run the complete MPS audit and publish its content-bound receipt."""

    required_paths = (
        repo,
        run,
        config_path,
        cache_root,
        cache_receipt_path,
        baseline_sidecar_path,
        baseline_audit_path,
        base_manifest_path,
        report_path,
        marker_path,
    )
    _require(all(Path(path).exists() for path in required_paths), "a differential audit input is missing")
    if os.path.lexists(output_path):
        raise FileExistsError(f"refusing to overwrite differential audit receipt: {output_path}")
    repo = repo.resolve()
    run = run.resolve()
    _require(run.is_dir() and not run.is_symlink(), "differential run is not a regular directory")
    _require({path.name for path in run.iterdir()} == EXPECTED_RUN_INVENTORY, "differential run inventory changed")
    for name in EXPECTED_RUN_INVENTORY:
        _require_regular_file(run / name, f"differential run artifact {name}")

    predictions_path = run / "predictions.npz"
    run_receipt_path = run / "run_receipt.json"
    training_log_path = run / "training_log.json"
    checkpoint_paths = {
        "ordered_pair": run / "ordered_pair_model.pt",
        "symmetric_pair": run / "symmetric_pair_model.pt",
    }
    all_bound_paths = [
        config_path,
        cache_receipt_path,
        baseline_sidecar_path,
        baseline_audit_path,
        base_manifest_path,
        report_path,
        marker_path,
        predictions_path,
        run_receipt_path,
        training_log_path,
        *checkpoint_paths.values(),
        output_path,
    ]
    resolved_paths = [Path(path).resolve() for path in all_bound_paths]
    _require(len(resolved_paths) == len(set(resolved_paths)), "differential audit paths alias")

    frozen = validate_frozen_config(config_path, repo)
    config = frozen["config"]
    def contract_path(value: object, label: str) -> Path:
        _require(isinstance(value, str), f"{label} contract path is not a string")
        path = Path(value)
        return path if path.is_absolute() else repo / path

    _require(
        run == contract_path(config["outputs"]["run"], "run").resolve(),
        "run path differs from the frozen contract",
    )
    _require(
        report_path.resolve()
        == contract_path(config["outputs"]["report"], "report").resolve(),
        "report path differs from the frozen contract",
    )
    _require(
        marker_path.resolve()
        == contract_path(config["outputs"]["completion_marker"], "marker").resolve(),
        "marker path differs from the frozen contract",
    )
    _require(
        cache_root.resolve()
        == contract_path(config["dataset"]["pixel_cache_root"], "cache").resolve(),
        "cache path differs from the frozen contract",
    )
    _require(
        cache_receipt_path.resolve()
        == contract_path(config["dataset"]["pixel_cache_receipt"], "cache receipt").resolve(),
        "cache-receipt path differs from the frozen contract",
    )
    _require(
        base_manifest_path.resolve()
        == contract_path(config["dataset"]["base_dataset_manifest"], "base manifest").resolve(),
        "base-manifest path differs from the frozen contract",
    )
    _require(
        baseline_sidecar_path.resolve()
        == contract_path(config["frozen_feature_baseline"]["prediction_sidecar"], "baseline sidecar").resolve(),
        "baseline-sidecar path differs from the frozen contract",
    )
    _require(
        baseline_audit_path.resolve()
        == contract_path(config["frozen_feature_baseline"]["audit_receipt"], "baseline audit").resolve(),
        "baseline-audit path differs from the frozen contract",
    )
    expected_output = repo / "results/idm/oracle_window_differential_s0_audit.json"
    _require(
        output_path.resolve() == expected_output.resolve(),
        "audit output path differs from the approved supplementary receipt",
    )
    cache = validate_cache_receipt(
        cache_root=cache_root,
        receipt_path=cache_receipt_path,
        base_manifest_path=base_manifest_path,
        config=config,
    )
    support = validate_base_manifest_and_support(
        manifest_path=base_manifest_path,
        config=config,
        train_arrays=cache["train_arrays"],
        validation_arrays=cache["validation_arrays"],
    )
    arrays = load_differential_sidecar(predictions_path)
    _require(len(arrays["true_offset"]) == EXPECTED_VALIDATION_EXAMPLES, "differential sidecar support changed")
    for name in METADATA_NAMES:
        _require_array_equal(f"differential sidecar/cache identity {name}", arrays[name], cache["validation_arrays"][name])
    sidecar_identity = list(
        zip(
            arrays["session_id"].astype(str).tolist(),
            arrays["run_index"].astype(np.int64).tolist(),
            arrays["array_index"].astype(np.int64).tolist(),
            arrays["head_index"].astype(np.int64).tolist(),
            arrays["true_offset"].astype(np.int64).tolist(),
            strict=True,
        )
    )
    _require(len(sidecar_identity) == len(set(sidecar_identity)), "differential sidecar repeats an example identity")
    baseline = validate_baseline(
        sidecar_path=baseline_sidecar_path,
        audit_path=baseline_audit_path,
        config=config,
        validation_arrays=cache["validation_arrays"],
        differential_arrays=arrays,
    )
    _require(
        cache["receipt"]["base_dataset_manifest_sha256"]
        == baseline["audit"]["bindings"]["dataset_manifest"]["sha256"],
        "cache/baseline base-manifest closure changed",
    )
    _require(
        cache["receipt"]["feature_receipt_sha256"]
        == baseline["audit"]["bindings"]["feature_receipt"]["sha256"],
        "cache/baseline feature-receipt closure changed",
    )
    training_log = validate_training_log(training_log_path, epochs=EXPECTED_EPOCHS)
    run_receipt = validate_run_receipt(
        receipt_path=run_receipt_path,
        config_path=config_path,
        cache_receipt_path=cache_receipt_path,
        baseline_sidecar_path=baseline_sidecar_path,
        baseline_audit_path=baseline_audit_path,
        predictions_path=predictions_path,
        training_log_path=training_log_path,
        checkpoint_paths=checkpoint_paths,
        config=config,
        implementation=frozen["implementation"],
    )
    immutable_inputs = {
        "validator": Path(__file__).resolve(),
        "config": config_path,
        "cache_receipt": cache_receipt_path,
        "cache_manifest": cache["manifest_path"],
        "cache_train": cache_root / "train.npz",
        "cache_validation": cache_root / "validation.npz",
        "base_manifest": base_manifest_path,
        "baseline_sidecar": baseline_sidecar_path,
        "baseline_audit": baseline_audit_path,
        "run_receipt": run_receipt_path,
        "training_log": training_log_path,
        "predictions": predictions_path,
        "ordered_checkpoint": checkpoint_paths["ordered_pair"],
        "symmetric_checkpoint": checkpoint_paths["symmetric_pair"],
        "report": report_path,
        "marker": marker_path,
    }
    immutable_hashes = {
        name: _sha256_file(path) for name, path in immutable_inputs.items()
    }

    random.seed(EXPECTED_SEED)
    np.random.seed(EXPECTED_SEED)
    torch.manual_seed(EXPECTED_SEED)
    initial_model = DifferentialCandidateModel(embedding_dim=int(config["model"]["embedding_dim"]))
    initial_state_sha = _state_dict_sha256(initial_model)
    _require(initial_state_sha == run_receipt["initial_state_sha256"], "seed-zero initial model does not reproduce")
    models: dict[str, DifferentialCandidateModel] = {}
    checkpoint_evidence: dict[str, Any] = {}
    for arm, path in checkpoint_paths.items():
        model, evidence = load_final_checkpoint(
            path=path,
            arm=arm,
            config=config,
            run_receipt=run_receipt,
        )
        models[arm] = model
        checkpoint_evidence[arm] = evidence

    _require(torch.backends.mps.is_available(), "MPS is unavailable for the required exact replay")
    device = torch.device(EXPECTED_DEVICE)
    torch.use_deterministic_algorithms(True, warn_only=True)
    validation_dataset = PixelOracleDataset(cache["validation_arrays"])
    replay: dict[str, np.ndarray] = {}
    probability_names = {
        "ordered_pair": "ordered_pair_prob",
        "symmetric_pair": "symmetric_pair_prob",
    }
    for arm in ("ordered_pair", "symmetric_pair"):
        probability = predict_probabilities(
            models[arm].to(device),
            validation_dataset,
            device=device,
            batch_size=int(config["training"]["eval_batch_size"]),
            arm=arm,
        )
        _require_array_equal(f"{arm} MPS checkpoint predictions", probability, arrays[probability_names[arm]])
        replay[arm] = probability
        checkpoint_evidence[arm]["predictions_reproduced_exactly_on_mps"] = True

    score_bindings = {
        "run_receipt_sha256": _sha256_file(run_receipt_path),
        "checkpoints": {
            arm: {
                "sha256": checkpoint_evidence[arm]["sha256"],
                "state_dict_sha256": checkpoint_evidence[arm]["state_dict_sha256"],
                "predictions_reproduced_exactly": True,
            }
            for arm in ("ordered_pair", "symmetric_pair")
        },
        "training_log_sha256": _sha256_file(training_log_path),
        "model_predictions_reproduced_exactly": True,
        "pixel_cache_identity_exact": True,
        "feature_baseline_identity_exact": True,
    }
    report = validate_score_report(
        report_path=report_path,
        predictions_path=predictions_path,
        config_path=config_path,
        cache_receipt_path=cache_receipt_path,
        base_manifest_path=base_manifest_path,
        arrays=arrays,
        config=config,
        base_manifest=support["manifest"],
        bindings=score_bindings,
    )
    marker = validate_completion_marker(
        marker_path=marker_path,
        report_path=report_path,
        config_path=config_path,
        predictions_path=predictions_path,
        run_receipt_path=run_receipt_path,
        checkpoint_paths=checkpoint_paths,
        cache_receipt_path=cache_receipt_path,
        report=report,
    )
    for name, path in immutable_inputs.items():
        _require(
            _sha256_file(path) == immutable_hashes[name],
            f"audit input changed during validation: {name}",
        )

    sidecar_arrays = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
        }
        for name, value in sorted(arrays.items())
    }
    audit_base = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": config["study_id"],
        "scope": "post-run validation only; frozen experiment bytes, report, marker, and decision are unchanged",
        "validator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__)),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "platform": sys.platform,
        },
        "bindings": {
            "config": {"path": str(config_path.resolve()), "sha256": frozen["config_sha256"]},
            "implementation": frozen["implementation"],
            "pixel_cache_receipt": {
                "path": str(cache_receipt_path.resolve()),
                "sha256": cache["receipt_sha256"],
                "content_sha256": cache["receipt"]["content_sha256"],
            },
            "pixel_cache_manifest": {
                "path": str(cache["manifest_path"].resolve()),
                "sha256": cache["manifest_sha256"],
            },
            "pixel_cache_splits": cache["split_evidence"],
            "base_dataset_manifest": {
                "path": str(base_manifest_path.resolve()),
                "sha256": support["sha256"],
            },
            "frozen_baseline_sidecar": {
                "path": str(baseline_sidecar_path.resolve()),
                "sha256": baseline["sidecar_sha256"],
                "conditional_prediction_sha256": _array_sha256(baseline["sidecar"]["conditional_prob"]),
            },
            "frozen_baseline_audit": {
                "path": str(baseline_audit_path.resolve()),
                "sha256": baseline["audit_sha256"],
                "content_sha256": baseline["audit_content_sha256"],
            },
            "run_receipt": {"path": str(run_receipt_path.resolve()), "sha256": _sha256_file(run_receipt_path)},
            "training_log": {"path": str(training_log_path.resolve()), "sha256": _sha256_file(training_log_path)},
            "prediction_sidecar": {
                "path": str(predictions_path.resolve()),
                "sha256": _sha256_file(predictions_path),
                "arrays": sidecar_arrays,
            },
            "checkpoints": checkpoint_evidence,
            "score_report": {"path": str(report_path.resolve()), "sha256": _sha256_file(report_path)},
            "completion_marker": {
                "path": str(marker_path.resolve()),
                "sha256": _sha256_file(marker_path),
                "content_sha256": marker["content_sha256"],
            },
        },
        "reproduction": {
            "run_inventory_exact": True,
            "device_exact_mps": True,
            "dependency_hashes_exact": True,
            "cache_receipt_and_files_exact": True,
            "cache_support_exact": True,
            "baseline_audit_closure_exact": True,
            "baseline_identity_and_predictions_exact": True,
            "initial_state_exact": True,
            "ordered_pair_predictions_exact_on_mps": True,
            "symmetric_pair_predictions_exact_on_mps": True,
            "score_report_exact": True,
            "completion_marker_content_bound": True,
            "ordered_pair_prediction_sha256": _array_sha256(replay["ordered_pair"]),
            "symmetric_pair_prediction_sha256": _array_sha256(replay["symmetric_pair"]),
            "train_examples": EXPECTED_TRAIN_EXAMPLES,
            "validation_examples": EXPECTED_VALIDATION_EXAMPLES,
            "validation_blocks": EXPECTED_VALIDATION_BLOCKS,
            "estimable_heads": support["estimable_heads"],
        },
        "run_contract": {
            "seed": int(run_receipt["seed"]),
            "epochs": int(run_receipt["epochs"]),
            "configured_epochs": int(run_receipt["configured_epochs"]),
            "device": run_receipt["device"],
            "matched_initialization": run_receipt["matched_initialization"],
            "matched_batch_order": run_receipt["matched_batch_order"],
            "final_weights_only": run_receipt["final_weights_only"],
            "validation_used_for_training_or_selection": training_log["validation_used_for_training_or_selection"],
            "frozen_implementation_sha256": dict(config["implementation"]["sha256"]),
        },
        "decision": {
            "original": report["decision_gate"]["decision"],
            "passed_primary_pixel_rescue": report["decision_gate"]["passed_primary_pixel_rescue"],
            "passed_differential_attribution": report["decision_gate"]["passed_differential_attribution"],
            "original_phase_2_decision_remains_rejected": report["decision_gate"]["original_phase_2_decision_remains_rejected"],
            "unchanged_by_audit": True,
        },
    }
    audit = _finalize_receipt(audit_base)
    _write_json_exclusive_atomic(output_path, audit)
    published = _json(output_path, "published differential audit")
    _require(published == audit, "published differential audit changed")
    _require_canonical_content_hash(published, "published differential audit")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--cache-receipt", required=True, type=Path)
    parser.add_argument("--baseline-sidecar", required=True, type=Path)
    parser.add_argument("--baseline-audit", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    audit = validate_oracle_window_differential_run(
        repo=args.repo,
        run=args.run,
        config_path=args.config,
        cache_root=args.cache,
        cache_receipt_path=args.cache_receipt,
        baseline_sidecar_path=args.baseline_sidecar,
        baseline_audit_path=args.baseline_audit,
        base_manifest_path=args.base_manifest,
        report_path=args.report,
        marker_path=args.marker,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "decision": audit["decision"]["original"],
                "audit": str(args.out),
                "content_sha256": audit["content_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
