"""Posthoc memorization diagnostics for the completed provisional blend GRUs.

This program has one deliberately narrow surface: every *complete* 96-window
segment item in each source pool used by a completed blend arm, plus the one
corrected own-v3 val-A session.  Each unique segment item is scored once with
the final checkpoint weights.  Training draw multiplicity, positive-class
weights, and transition weights are receipts, not evaluation weights.

The sealed untouched test, B1, mapped-y4n, and val-B have no command-line path
through this module.  Their known session identities are rejected before any
NPZ is opened.  The evaluator writes only one JSON report and a final
content-bound completion marker; it never writes prediction arrays or changes
the immutable feature view.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from badeline.model import BadelineIDM
from badeline.train import SegmentSessionDataset, load_session, read_session_ids
from data.schema import KEY_ORDER
from experiments.eval_provisional_blend_gru import (
    ARM_SPECS,
    STUDY_ID,
    UNTOUCHED_SESSION_ID,
    sha256_file,
    validate_contract,
    validate_run,
)


REPORT_SCHEMA_VERSION = "madeleine.provisional-blend-memorization.v1"
MARKER_SCHEMA_VERSION = "madeleine.provisional-blend-memorization-marker.v1"
FEATURE_VIEW_SCHEMA_VERSION = "madeleine.provisional-blend-feature-view.v1"
SURFACE = "complete_training_segment_pools_and_corrected_local_val_a"
FEATURE_VIEW_RECEIPT = "blend_feature_view_receipt.json"
HARDLINK_INVENTORY = "hardlink_inventory.jsonl"
EXPECTED_FEATURE_VIEW_SOURCE_COUNTS = {
    "nitrogen": 1_078,  # 1,062 train + 16 mapped validation hard links.
    "wild_provisional": 2_058,
    "local": 4,  # Three corrected train sessions + local val-A.
}
LOCAL_VAL_A_LIST = "local_val_a_sessions.txt"
LOCAL_VAL_A_ID = "rec_20260724_171305_5min"
VAL_B_ID = "rec_20260725_025853"
B1_ID = "rec_20260725_160450_b1"
FORBIDDEN_SESSION_IDS = frozenset(
    {UNTOUCHED_SESSION_ID, VAL_B_ID, B1_ID}
)
HEX_64 = frozenset("0123456789abcdef")
DIAGNOSTIC_RELATIVE_PATH = Path(
    "experiments/eval_provisional_blend_memorization.py"
)


def _json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX_64)
    )


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and set(value).issubset(HEX_64)
    )


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
    )


def bind_diagnostic_source(
    repo: Path,
    diagnostic_commit: str,
    *,
    loaded_module_path: Path | None = None,
) -> dict[str, str]:
    """Bind this evaluator's exact clean bytes to a declared Git commit."""

    repo = repo.resolve()
    if not _is_git_commit(diagnostic_commit):
        raise ValueError("diagnostic Git commit must be 40 lowercase hex characters")
    declared_module_path = repo / DIAGNOSTIC_RELATIVE_PATH
    if not declared_module_path.is_file() or declared_module_path.is_symlink():
        raise ValueError("memorization diagnostic must be a regular source file")
    module_path = declared_module_path.resolve()
    try:
        module_path.relative_to(repo)
    except ValueError as error:
        raise ValueError("memorization diagnostic resolves outside --repo") from error
    loaded_path = (
        Path(__file__).resolve()
        if loaded_module_path is None
        else loaded_module_path.resolve()
    )
    if loaded_path != module_path:
        raise ValueError("loaded memorization diagnostic is outside --repo")

    resolved = _git(repo, "rev-parse", "--verify", f"{diagnostic_commit}^{{commit}}")
    if resolved.returncode or resolved.stdout.strip().decode() != diagnostic_commit:
        raise ValueError("diagnostic Git commit is not resolvable")
    if _git(
        repo, "merge-base", "--is-ancestor", diagnostic_commit, "HEAD"
    ).returncode:
        raise ValueError("diagnostic Git commit is not an ancestor of HEAD")
    working_bytes = module_path.read_bytes()
    committed = _git(
        repo,
        "cat-file",
        "blob",
        f"{diagnostic_commit}:{DIAGNOSTIC_RELATIVE_PATH.as_posix()}",
    )
    if committed.returncode or committed.stdout != working_bytes:
        raise ValueError("memorization diagnostic differs from declared commit")
    status = _git(
        repo,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        DIAGNOSTIC_RELATIVE_PATH.as_posix(),
    )
    if status.returncode or status.stdout:
        raise ValueError("memorization diagnostic source is not clean")
    return {
        "git_commit": diagnostic_commit,
        "relative_path": DIAGNOSTIC_RELATIVE_PATH.as_posix(),
        "sha256": hashlib.sha256(working_bytes).hexdigest(),
    }


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _line_list_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(
        ("\n".join(values) + "\n").encode("utf-8")
    ).hexdigest()


def _canonical_array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    byte_view = memoryview(contiguous).cast("B")
    for offset in range(0, len(byte_view), 1024 * 1024):
        digest.update(byte_view[offset : offset + 1024 * 1024])
    return digest.hexdigest()


def _require_allowed_session_ids(
    values: Sequence[str], description: str
) -> list[str]:
    result = [str(value) for value in values]
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{description} is empty or contains duplicate sessions")
    forbidden = sorted(set(result).intersection(FORBIDDEN_SESSION_IDS))
    if forbidden:
        raise ValueError(
            f"{description} contains an embargoed session: {', '.join(forbidden)}"
        )
    return result


def _argument_value(argv: Sequence[object], option: str) -> str:
    values = [str(value) for value in argv]
    positions = [index for index, value in enumerate(values) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(values):
        raise ValueError(f"run metadata must contain exactly one {option}")
    return values[positions[0] + 1]


def _read_hardlink_inventory(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"hard-link inventory is not readable: {path}") from error
    if not lines:
        raise ValueError("hard-link inventory is empty")
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"hard-link inventory row {line_number} is invalid JSON"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(
                f"hard-link inventory row {line_number} is not an object"
            )
        session_id = row.get("session_id")
        source = row.get("source")
        byte_count = row.get("bytes")
        digest = row.get("sha256")
        if (
            not isinstance(session_id, str)
            or not session_id
            or source not in {"nitrogen", "wild_provisional", "local"}
            or not isinstance(byte_count, int)
            or byte_count < 1
            or not _is_sha256(digest)
        ):
            raise ValueError(
                f"hard-link inventory row {line_number} is malformed"
            )
        if session_id in rows:
            raise ValueError(f"duplicate hard-link inventory session: {session_id}")
        _require_allowed_session_ids([session_id], "hard-link inventory")
        rows[session_id] = {
            "source": str(source),
            "bytes": byte_count,
            "sha256": str(digest),
        }
    return rows


def validate_feature_view(
    data_dir: Path,
    *,
    contract_sha256: str,
    source_sessions: Mapping[str, Sequence[str]],
    local_val_a_sessions: Sequence[str],
) -> dict[str, Any]:
    """Validate immutable view metadata without opening an evaluation shard."""

    if data_dir.is_symlink():
        raise ValueError("blend feature view must not be a symlink")
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise ValueError("blend feature view must be a real directory")
    receipt_path = data_dir / FEATURE_VIEW_RECEIPT
    receipt = _json_object(receipt_path, "blend feature-view receipt")
    if receipt.get("schema_version") != FEATURE_VIEW_SCHEMA_VERSION:
        raise ValueError("blend feature-view schema changed")
    if receipt.get("study_id") != STUDY_ID:
        raise ValueError("blend feature-view study identity changed")
    contract = receipt.get("contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("sha256") != contract_sha256
    ):
        raise ValueError("blend feature view is bound to another contract")
    if receipt.get("sealed_untouched_session_present") is not False:
        raise ValueError("blend feature view does not exclude the sealed test")
    if receipt.get("temporary_files_present") is not False:
        raise ValueError("blend feature view is not a complete atomic publication")

    hardlinks = receipt.get("hardlinks")
    if (
        not isinstance(hardlinks, Mapping)
        or hardlinks.get("verified") is not True
        or hardlinks.get("inventory_file") != HARDLINK_INVENTORY
    ):
        raise ValueError("blend feature view lacks a verified hard-link receipt")
    inventory_path = data_dir / HARDLINK_INVENTORY
    inventory_sha256 = sha256_file(inventory_path)
    if hardlinks.get("inventory_sha256") != inventory_sha256:
        raise ValueError("blend hard-link inventory hash changed")
    inventory = _read_hardlink_inventory(inventory_path)
    if hardlinks.get("files") != len(inventory):
        raise ValueError("blend hard-link inventory count changed")
    observed_source_counts = {
        source: sum(row["source"] == source for row in inventory.values())
        for source in EXPECTED_FEATURE_VIEW_SOURCE_COUNTS
    }
    if observed_source_counts != EXPECTED_FEATURE_VIEW_SOURCE_COUNTS:
        raise ValueError("blend hard-link inventory source counts changed")

    generated = receipt.get("generated_files")
    if not isinstance(generated, Mapping) or not generated:
        raise ValueError("blend feature view lacks generated-file receipts")
    for name, digest in generated.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not _is_sha256(digest)
        ):
            raise ValueError("blend generated-file receipt is malformed")
        path = data_dir / name
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise ValueError(f"blend generated file changed: {name}")

    selected: dict[str, list[str]] = {}
    seen: set[str] = set()
    for source, raw_ids in source_sessions.items():
        if source not in {"nitrogen", "wild_provisional", "local"}:
            raise ValueError(f"unknown blend source: {source}")
        session_ids = _require_allowed_session_ids(
            raw_ids, f"{source} training membership"
        )
        if seen.intersection(session_ids):
            raise ValueError("blend source memberships overlap")
        seen.update(session_ids)
        for session_id in session_ids:
            row = inventory.get(session_id)
            if row is None or row["source"] != source:
                raise ValueError(
                    f"hard-link inventory source changed for {session_id}"
                )
        selected[source] = session_ids

    local_val_ids = _require_allowed_session_ids(
        local_val_a_sessions, "corrected local val-A membership"
    )
    if local_val_ids != [LOCAL_VAL_A_ID]:
        raise ValueError("corrected local val-A identity changed")
    if seen.intersection(local_val_ids):
        raise ValueError("local val-A overlaps blend training membership")
    for session_id in local_val_ids:
        row = inventory.get(session_id)
        if row is None or row["source"] != "local":
            raise ValueError("corrected local val-A feature inventory changed")

    return {
        "path": str(data_dir),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "hardlink_inventory_path": str(inventory_path),
        "hardlink_inventory_sha256": inventory_sha256,
        "hardlink_inventory_rows": len(inventory),
        "generated_files": dict(generated),
        "inventory": inventory,
        "source_sessions": selected,
        "local_val_a_sessions": local_val_ids,
    }


def validate_selected_shards(
    data_dir: Path,
    *,
    source: str,
    session_ids: Sequence[str],
    inventory: Mapping[str, Mapping[str, Any]],
    run_meta_shards: Mapping[str, object] | None,
) -> dict[str, Any]:
    """Hash every selected feature shard before any model inference."""

    session_ids = _require_allowed_session_ids(session_ids, f"{source} shards")
    rows: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for session_id in session_ids:
        expected = inventory.get(session_id)
        if not isinstance(expected, Mapping) or expected.get("source") != source:
            raise ValueError(f"feature inventory changed for {session_id}")
        path = data_dir / f"{session_id}.npz"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"selected feature shard is not a regular file: {path}")
        stat = path.stat()
        if stat.st_size != expected.get("bytes") or stat.st_nlink < 2:
            raise ValueError(f"selected feature hard link changed: {session_id}")
        digest = sha256_file(path)
        if digest != expected.get("sha256"):
            raise ValueError(f"selected feature shard hash changed: {session_id}")
        if (
            run_meta_shards is not None
            and run_meta_shards.get(session_id) != digest
        ):
            raise ValueError(f"training-time shard hash changed: {session_id}")
        rows[session_id] = {"bytes": stat.st_size, "sha256": digest}
        total_bytes += stat.st_size
    return {
        "sessions": len(rows),
        "total_bytes": total_bytes,
        "membership_sha256": _line_list_sha256(session_ids),
        "shard_set_sha256": _canonical_json_sha256(rows),
        "shards": rows,
    }


def _segment_dataset(
    data_dir: Path,
    session_id: str,
    config: Mapping[str, Any],
) -> SegmentSessionDataset | None:
    _require_allowed_session_ids([session_id], "segment-pool session")
    arrays = load_session(data_dir, session_id, precomputed_features=True)
    try:
        return SegmentSessionDataset(
            [arrays],
            window=int(config["window"]),
            window_mode=str(config["window_mode"]),
            input_config=str(config["input_config"]),
            history_len=int(config.get("history_len", 16)),
            history_gap=int(config.get("history_gap", 0)),
            segment_windows=int(config["segment_windows"]),
            active_targets_only=bool(config.get("active_targets_only", True)),
            transition_weight=float(config.get("transition_weight", 1.0)),
            precomputed_features=True,
            frame_stride=int(config.get("frame_stride", 1)),
        )
    except ValueError as error:
        if str(error).startswith("no session is long enough for one "):
            return None
        raise


def count_complete_segment_items(
    data_dir: Path,
    session_ids: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[int, dict[str, int]]:
    """Count the exact full-segment pool using the training dataset class."""

    counts: dict[str, int] = {}
    total = 0
    for session_id in _require_allowed_session_ids(
        session_ids, "complete-segment membership"
    ):
        dataset = _segment_dataset(data_dir, session_id, config)
        count = 0 if dataset is None else len(dataset)
        counts[session_id] = count
        total += count
    return total, counts


def _macro(values: Mapping[str, float]) -> float:
    array = np.asarray(list(values.values()), dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("per-key diagnostic is not finite for all seven keys")
    return float(array.mean())


def _close_memmap(value: np.memmap) -> None:
    value.flush()
    mapping = getattr(value, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _per_key_rank_and_decision_metrics(
    truth: np.ndarray, probability: np.ndarray
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Compute exact metrics one key at a time to bound peak host memory."""

    ap: dict[str, float] = {}
    state_f1: dict[str, float] = {}
    predicted_positive_rate: dict[str, float] = {}
    rows = len(truth)
    for index, key in enumerate(KEY_ORDER):
        key_truth = np.asarray(truth[:, index], dtype=bool)
        key_probability = np.asarray(probability[:, index])
        key_prediction = key_probability >= 0.5
        ap[key] = float(average_precision_score(key_truth, key_probability))
        true_positive = int(np.count_nonzero(key_truth & key_prediction))
        false_positive = int(np.count_nonzero(~key_truth & key_prediction))
        false_negative = int(np.count_nonzero(key_truth & ~key_prediction))
        denominator = 2 * true_positive + false_positive + false_negative
        state_f1[key] = (
            0.0 if denominator == 0 else 2.0 * true_positive / denominator
        )
        predicted_positive_rate[key] = float(
            np.count_nonzero(key_prediction) / rows
        )
    return ap, state_f1, predicted_positive_rate


def evaluate_complete_segment_pool(
    model: BadelineIDM | torch.nn.Module,
    config: Mapping[str, Any],
    data_dir: Path,
    session_ids: Sequence[str],
    *,
    expected_segment_items: int,
    device: str,
    batch_segments: int,
    scratch_dir: Path | None,
) -> dict[str, Any]:
    """Score every unique complete training segment exactly once.

    Probability/truth matrices live in temporary memory maps.  This keeps the
    exhaustive NitroGen pool bounded in resident memory while retaining exact
    (not histogram-approximated) average precision.
    """

    session_ids = _require_allowed_session_ids(
        session_ids, "complete-segment evaluation membership"
    )
    if expected_segment_items < 1:
        raise ValueError("expected_segment_items must be positive")
    if batch_segments < 1:
        raise ValueError("batch_segments must be positive")
    segment_windows = int(config["segment_windows"])
    expected_rows = expected_segment_items * segment_windows
    key_count = len(KEY_ORDER)
    if scratch_dir is not None:
        scratch_dir = scratch_dir.resolve()
        scratch_dir.mkdir(parents=True, exist_ok=True)

    model.eval().to(device)
    bce_sum = np.zeros(key_count, dtype=np.float64)
    session_items: dict[str, int] = {}
    cursor = 0
    prefix = "madeleine-blend-memorization-"
    with tempfile.TemporaryDirectory(
        prefix=prefix,
        dir=str(scratch_dir) if scratch_dir is not None else None,
    ) as temporary_name:
        temporary = Path(temporary_name)
        truth_path = temporary / "truth.uint8.mmap"
        probability_path = temporary / "probability.float32.mmap"
        truth_store = np.memmap(
            truth_path,
            mode="w+",
            dtype=np.uint8,
            shape=(expected_rows, key_count),
        )
        probability_store = np.memmap(
            probability_path,
            mode="w+",
            dtype=np.float32,
            shape=(expected_rows, key_count),
        )
        try:
            with torch.inference_mode():
                for session_id in session_ids:
                    dataset = _segment_dataset(data_dir, session_id, config)
                    item_count = 0 if dataset is None else len(dataset)
                    session_items[session_id] = item_count
                    if dataset is None:
                        continue
                    loader = DataLoader(
                        dataset,
                        batch_size=min(batch_segments, item_count),
                        shuffle=False,
                        num_workers=0,
                        pin_memory=str(device).startswith("cuda"),
                    )
                    for batch in loader:
                        target = batch["target"].to(
                            device=device,
                            dtype=torch.float32,
                            non_blocking=True,
                        )
                        model_batch = {
                            name: value.to(device=device, non_blocking=True)
                            for name, value in batch.items()
                            if name in {"features", "frames", "history"}
                        }
                        logits = model.forward_segment(model_batch)
                        if logits.shape != target.shape:
                            raise ValueError(
                                "segment model output and target shapes differ"
                            )
                        if not torch.isfinite(logits).all():
                            raise ValueError("segment model emitted non-finite logits")
                        per_label_bce = F.binary_cross_entropy_with_logits(
                            logits, target, reduction="none"
                        )
                        bce_sum += (
                            per_label_bce.to(torch.float64)
                            .sum(dim=(0, 1))
                            .cpu()
                            .numpy()
                        )
                        probability = torch.sigmoid(logits).to(torch.float32)
                        flat_probability = (
                            probability.cpu().numpy().reshape(-1, key_count)
                        )
                        flat_truth = (
                            target.to(torch.uint8)
                            .cpu()
                            .numpy()
                            .reshape(-1, key_count)
                        )
                        stop = cursor + len(flat_truth)
                        if stop > expected_rows:
                            raise ValueError(
                                "observed complete-segment support exceeds receipt"
                            )
                        truth_store[cursor:stop] = flat_truth
                        probability_store[cursor:stop] = flat_probability
                        cursor = stop

            observed_items = sum(session_items.values())
            if observed_items != expected_segment_items or cursor != expected_rows:
                raise ValueError(
                    "complete-segment support changed: "
                    f"expected {expected_segment_items} items/{expected_rows} rows, "
                    f"observed {observed_items} items/{cursor} rows"
                )
            truth_store.flush()
            probability_store.flush()
            truth = np.asarray(truth_store)
            probability = np.asarray(probability_store)
            if not np.all(np.isin(truth, (0, 1))):
                raise ValueError("complete-segment truth is not binary")
            if not np.all(np.isfinite(probability)) or np.any(
                (probability < 0.0) | (probability > 1.0)
            ):
                raise ValueError("complete-segment probabilities are invalid")
            positive = truth.sum(axis=0, dtype=np.int64)
            if np.any(positive == 0) or np.any(positive == expected_rows):
                raise ValueError(
                    "complete-segment surface lacks both classes for a key"
                )
            ap, state_f1, predicted_positive_rate = (
                _per_key_rank_and_decision_metrics(truth, probability)
            )
            bce = {
                key: float(bce_sum[index] / expected_rows)
                for index, key in enumerate(KEY_ORDER)
            }
            prevalence = {
                key: float(positive[index] / expected_rows)
                for index, key in enumerate(KEY_ORDER)
            }
            result = {
                "support": {
                    "sessions": len(session_ids),
                    "sessions_with_complete_segments": sum(
                        count > 0 for count in session_items.values()
                    ),
                    "sessions_without_complete_segments": sum(
                        count == 0 for count in session_items.values()
                    ),
                    "segment_items": observed_items,
                    "segment_windows": segment_windows,
                    "target_frames": expected_rows,
                    "binary_labels": expected_rows * key_count,
                    "session_segment_items": session_items,
                    "session_segment_items_sha256": _canonical_json_sha256(
                        session_items
                    ),
                    "truth_sha256": _canonical_array_sha256(truth),
                    "probability_sha256": _canonical_array_sha256(probability),
                },
                "metrics": {
                    "unweighted_bce": {
                        "per_key": bce,
                        "macro": _macro(bce),
                    },
                    "average_precision": {
                        "per_key": ap,
                        "macro": _macro(ap),
                    },
                    "state_f1_fixed_0_5": {
                        "per_key": state_f1,
                        "macro": _macro(state_f1),
                    },
                    "prevalence": {
                        "per_key": prevalence,
                        "macro": _macro(prevalence),
                    },
                    "predicted_positive_rate_fixed_0_5": {
                        "per_key": predicted_positive_rate,
                        "macro": _macro(predicted_positive_rate),
                    },
                },
            }
        finally:
            _close_memmap(truth_store)
            _close_memmap(probability_store)
    return result


def local_generalization_gap(
    local_train: Mapping[str, Any], local_val_a: Mapping[str, Any]
) -> dict[str, Any]:
    """Return signed gaps whose names state which direction is positive."""

    train_metrics = local_train.get("metrics")
    val_metrics = local_val_a.get("metrics")
    if not isinstance(train_metrics, Mapping) or not isinstance(
        val_metrics, Mapping
    ):
        raise ValueError("local train/val-A metrics are missing")

    result: dict[str, Any] = {}
    conventions = {
        "unweighted_bce": "val_a_minus_train",
        "average_precision": "train_minus_val_a",
        "state_f1_fixed_0_5": "train_minus_val_a",
    }
    for metric, direction in conventions.items():
        train = train_metrics.get(metric)
        val = val_metrics.get(metric)
        if not isinstance(train, Mapping) or not isinstance(val, Mapping):
            raise ValueError(f"local metric is missing: {metric}")
        train_per_key = train.get("per_key")
        val_per_key = val.get("per_key")
        if not isinstance(train_per_key, Mapping) or not isinstance(
            val_per_key, Mapping
        ):
            raise ValueError(f"local per-key metric is missing: {metric}")
        if direction == "val_a_minus_train":
            subtract = lambda key: float(val_per_key[key]) - float(
                train_per_key[key]
            )
            macro = float(val["macro"]) - float(train["macro"])
        else:
            subtract = lambda key: float(train_per_key[key]) - float(
                val_per_key[key]
            )
            macro = float(train["macro"]) - float(val["macro"])
        result[metric] = {
            "direction": direction,
            "per_key": {key: subtract(key) for key in KEY_ORDER},
            "macro": macro,
        }
    return result


def _artifact_path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def publish_atomic_report(
    report_path: Path,
    marker_path: Path,
    report: Mapping[str, Any],
    marker_fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish report first and its completion marker last, with rollback."""

    if report_path.suffix != ".json" or marker_path.suffix != ".json":
        raise ValueError("memorization report and marker must be JSON")
    if report_path.resolve() == marker_path.resolve():
        raise ValueError("memorization report and marker paths must differ")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_name(f".{report_path.name}.tmp")
    temporary_marker = marker_path.with_name(f".{marker_path.name}.tmp")
    for path in (report_path, marker_path, temporary_report, temporary_marker):
        if os.path.lexists(path):
            raise ValueError(f"refusing to overwrite memorization artifact: {path}")
    published: list[Path] = []
    try:
        _write_json(temporary_report, report)
        report_sha256 = sha256_file(temporary_report)
        reserved = {"schema_version", "status", "report", "report_sha256"}
        if reserved.intersection(marker_fields):
            raise ValueError("marker fields attempt to replace publication metadata")
        marker = {
            **dict(marker_fields),
            "schema_version": MARKER_SCHEMA_VERSION,
            "status": "complete",
            "report": str(report_path),
            "report_sha256": report_sha256,
        }
        _write_json(temporary_marker, marker)
        temporary_report.replace(report_path)
        published.append(report_path)
        temporary_marker.replace(marker_path)
        published.append(marker_path)
        return marker
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        temporary_report.unlink(missing_ok=True)
        temporary_marker.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--contract-commit", required=True)
    parser.add_argument("--diagnostic-commit", required=True)
    parser.add_argument("--arm", choices=tuple(ARM_SPECS), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--completion-marker", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-segments", type=int, default=32)
    parser.add_argument("--scratch-dir", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.run_id != ARM_SPECS[args.arm]["run_id"]:
        raise ValueError("run ID differs from the requested blend arm")
    for artifact in (args.out, args.completion_marker):
        if _artifact_path_is_within(artifact, args.data) or _artifact_path_is_within(
            artifact, args.run
        ):
            raise ValueError("diagnostic artifacts may not modify data or run roots")
    if args.scratch_dir is not None and (
        _artifact_path_is_within(args.scratch_dir, args.data)
        or _artifact_path_is_within(args.scratch_dir, args.run)
    ):
        raise ValueError("diagnostic scratch space may not modify data or run roots")

    diagnostic_source = bind_diagnostic_source(
        args.repo,
        args.diagnostic_commit,
    )

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
    if run_receipt.get("evaluation_weights") != "final_state_dict":
        raise ValueError("memorization diagnostics require final weights")

    run_meta_path = args.run / "run_meta.json"
    run_meta = _json_object(run_meta_path, "blend run metadata")
    run_data = Path(_argument_value(run_meta.get("argv", []), "--data"))
    if run_data.resolve() != args.data.resolve():
        raise ValueError("diagnostic data root differs from the training view")
    run_meta_shards = run_meta.get("shard_sha256")
    if not isinstance(run_meta_shards, Mapping):
        raise ValueError("blend run metadata lacks shard hashes")

    source_sessions_raw = config.get("source_sampling", {}).get("sources")
    if not isinstance(source_sessions_raw, Mapping):
        raise ValueError("blend config lacks source membership")
    source_sessions = {
        str(source): _require_allowed_session_ids(
            values, f"{source} source membership"
        )
        for source, values in source_sessions_raw.items()
        if isinstance(values, list)
    }
    if set(source_sessions) != set(ARM_SPECS[args.arm]["mix"]):
        raise ValueError("blend source set changed")
    local_val_path = args.data / LOCAL_VAL_A_LIST
    local_val_ids = read_session_ids(local_val_path)
    if _line_list_sha256(local_val_ids) != contract["sources"]["local"][
        "val_sessions_sha256"
    ]:
        raise ValueError("corrected local val-A membership hash changed")

    view = validate_feature_view(
        args.data,
        contract_sha256=args.contract_sha256,
        source_sessions=source_sessions,
        local_val_a_sessions=local_val_ids,
    )
    inventory = view.pop("inventory")
    source_shards: dict[str, dict[str, Any]] = {}
    for source, session_ids in source_sessions.items():
        source_shards[source] = validate_selected_shards(
            args.data,
            source=source,
            session_ids=session_ids,
            inventory=inventory,
            run_meta_shards=run_meta_shards,
        )
    local_val_shards = validate_selected_shards(
        args.data,
        source="local",
        session_ids=local_val_ids,
        inventory=inventory,
        run_meta_shards=None,
    )

    surfaces: dict[str, dict[str, Any]] = {}
    sampling_sources = run_receipt["source_sampling"]["sources"]
    for source, session_ids in source_sessions.items():
        sampling = sampling_sources[source]
        expected_items = int(sampling["segment_items"])
        result = evaluate_complete_segment_pool(
            model,
            config,
            args.data,
            session_ids,
            expected_segment_items=expected_items,
            device=args.device,
            batch_segments=args.batch_segments,
            scratch_dir=args.scratch_dir,
        )
        result.update(
            {
                "role": "unique_complete_training_segment_pool",
                "tier": contract["sources"][source]["tier"],
                "source_contract": contract["sources"][source],
                "sampling_receipt": sampling,
                "shard_receipt": source_shards[source],
            }
        )
        surfaces[source] = result

    local_val_items, local_val_item_counts = count_complete_segment_items(
        args.data, local_val_ids, config
    )
    local_val = evaluate_complete_segment_pool(
        model,
        config,
        args.data,
        local_val_ids,
        expected_segment_items=local_val_items,
        device=args.device,
        batch_segments=args.batch_segments,
        scratch_dir=args.scratch_dir,
    )
    if local_val["support"]["session_segment_items"] != local_val_item_counts:
        raise ValueError("local val-A support changed between count and inference")
    local_val.update(
        {
            "role": "corrected_local_val_a_complete_segment_pool",
            "tier": "engine_truth_corrected_own_v3_development",
            "sampling_receipt": {
                "scheduled_draws": 0,
                "actual_draws": 0,
                "repeat_draws": 0,
                "note": "held-out local development surface; never sampled",
            },
            "source_rgb_shard_sha256": contract["sources"]["local"][
                "val_a_shard_sha256"
            ],
            "shard_receipt": local_val_shards,
        }
    )
    surfaces["local_val_a"] = local_val

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "arm": args.arm,
        "run_id": args.run_id,
        "surface": SURFACE,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "weights": "final_state_dict",
        "contract": {
            "path": str(args.contract),
            "sha256": args.contract_sha256,
            "commit": args.contract_commit,
        },
        "diagnostic_source": diagnostic_source,
        "run_receipt": run_receipt,
        "feature_view": view,
        "scope_guard": {
            "accessed": [
                "unique complete training segment pools",
                "corrected own-v3 local val-A complete segment pool",
            ],
            "not_accessed": [
                "B1",
                "val-B",
                "mapped-y4n",
                "sealed untouched test",
            ],
            "known_forbidden_session_ids": sorted(FORBIDDEN_SESSION_IDS),
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
        "source_sampling_receipt": run_receipt["source_sampling"],
        "surfaces": surfaces,
        "local_generalization_gap": local_generalization_gap(
            surfaces["local"], surfaces["local_val_a"]
        ),
    }
    marker_fields = {
        "study_id": STUDY_ID,
        "arm": args.arm,
        "run_id": args.run_id,
        "surface": SURFACE,
        "weights": "final_state_dict",
        "contract_sha256": args.contract_sha256,
        "training_implementation_git_commit": run_receipt["inference_source"][
            "implementation_git_commit"
        ],
        "diagnostic_git_commit": diagnostic_source["git_commit"],
        "diagnostic_module_sha256": diagnostic_source["sha256"],
        "config_sha256": run_receipt["config_sha256"],
        "checkpoint_sha256": run_receipt["checkpoint_sha256"],
        "run_meta_sha256": run_receipt["run_meta_sha256"],
        "training_log_sha256": run_receipt["training_log_sha256"],
        "source_sampling_receipt_sha256": run_receipt[
            "source_sampling_receipt_sha256"
        ],
        "selected_final_tensors_identical": run_receipt[
            "selected_final_tensors_identical"
        ],
        "feature_view_receipt_sha256": view["receipt_sha256"],
        "forbidden_surfaces_accessed": False,
    }
    if bind_diagnostic_source(args.repo, args.diagnostic_commit) != diagnostic_source:
        raise ValueError("memorization diagnostic source changed during evaluation")
    marker = publish_atomic_report(
        args.out,
        args.completion_marker,
        report,
        marker_fields,
    )
    print(json.dumps(marker, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
