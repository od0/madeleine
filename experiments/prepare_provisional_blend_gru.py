"""Assemble the immutable feature view for the provisional GRU blend study.

The assembler does not copy or transform supervision.  It hard-links three
already validated feature sources into one read-only training view, writes the
two exact source-mixture configs, and publishes a content-bound receipt by an
atomic directory rename.  The sealed untouched test is rejected by identity
and is never an input to this program.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from data.schema import KEY_ORDER


SCHEMA_VERSION = "madeleine.provisional-blend-feature-view.v1"
CONTRACT_SCHEMA = "madeleine.provisional-blend-gru-decision.v1"
SAMPLER_SCHEMA = "madeleine.source-balanced-batch.v1"
UNTOUCHED_SESSION = "rec_20260727_220000_test"
MAX_STEPS = 14_265
CYCLE_STEPS = 5
CYCLE_ITEMS = 80
EXPECTED_TEMPLATE_SHA256 = (
    "9c92ee27ac37115389980490f656af1af5bf0f3389952e652b323f6b279bfb95"
)
EXPECTED_REFERENCE_SHA256 = (
    "cf55f612382bfa7b9a1b67038b5223a1629782782995f0b008311ba380b34f94"
)
EXPECTED_POSITIVE_WEIGHT = (
    6.5241737274432445,
    2.165329049329906,
    5.234555340345659,
    10.0,
    6.478684563686479,
    10.0,
    1.7741724096558231,
)
LOCAL_TRAIN_IDS = (
    "rec_20260724_190233",
    "rec_20260725_015612",
    "rec_20260725_021338",
)
LOCAL_VAL_IDS = ("rec_20260724_171305_5min",)
LOCAL_FEATURE_FILES = {
    "build_manifest.json",
    "feature_build_manifest.json",
    "train_sessions.txt",
    "val_sessions.txt",
    *(f"{session_id}.npz" for session_id in (*LOCAL_TRAIN_IDS, *LOCAL_VAL_IDS)),
}
EXPECTED_ARMS = {
    "NL_90_10": {
        "run_id": "blend_provisional_nl90_10_92train_y4n_holdout_26m_128x3_s0",
        "training_mix_percent": {"nitrogen": 90, "local": 10},
        "five_step_cycle": [
            {"nitrogen": 14, "local": 2},
            {"nitrogen": 14, "local": 2},
            {"nitrogen": 14, "local": 2},
            {"nitrogen": 15, "local": 1},
            {"nitrogen": 15, "local": 1},
        ],
        "expected_draws": {"nitrogen": 205_416, "local": 22_824},
    },
    "NLW_70_20_10": {
        "run_id": "blend_provisional_nlw70_20_10_92train_y4n_holdout_26m_128x3_s0",
        "training_mix_percent": {
            "nitrogen": 70,
            "wild_provisional": 20,
            "local": 10,
        },
        "five_step_cycle": [
            {"nitrogen": 11, "wild_provisional": 3, "local": 2},
            {"nitrogen": 11, "wild_provisional": 3, "local": 2},
            {"nitrogen": 11, "wild_provisional": 3, "local": 2},
            {"nitrogen": 12, "wild_provisional": 3, "local": 1},
            {"nitrogen": 11, "wild_provisional": 4, "local": 1},
        ],
        "expected_draws": {
            "nitrogen": 159_768,
            "wild_provisional": 45_648,
            "local": 22_824,
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _sessions(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"session list is empty or contains duplicates: {path}")
    if UNTOUCHED_SESSION in values:
        raise ValueError("sealed untouched test appeared in a blend input")
    return values


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_sha(path: Path, expected: str, description: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing {description}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{description} SHA-256 changed: expected {expected}, got {observed}"
        )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_contract(contract: Mapping[str, Any]) -> None:
    """Reject drift in the two-arm preregistration before touching features."""

    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("blend contract schema changed")
    if contract.get("study_id") != "provisional_blend_gru_y4n_b1_s0":
        raise ValueError("blend study identity changed")
    reference = contract.get("reference")
    if not isinstance(reference, Mapping) or reference.get(
        "checkpoint_sha256"
    ) != EXPECTED_REFERENCE_SHA256:
        raise ValueError("pure-N reference checkpoint changed")
    if reference.get("training_mix_percent") != {
        "nitrogen": 100,
        "local": 0,
        "wild_provisional": 0,
    }:
        raise ValueError("pure-N reference mixture changed")

    model = contract.get("model_contract")
    if not isinstance(model, Mapping):
        raise ValueError("model contract is missing")
    exact_model = {
        "initialization": "from scratch; seed 0; no checkpoint initialization",
        "template_sha256": EXPECTED_TEMPLATE_SHA256,
        "window": 128,
        "frame_stride": 3,
        "window_mode": "centered",
        "segment_windows": 96,
        "segment_items_per_step": 16,
        "maximum_steps": MAX_STEPS,
        "evaluation_steps": [0, MAX_STEPS],
        "learning_rate": 0.0003,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "linear_learning_rate_decay": True,
        "transition_weight": 8.0,
        "weights_reported": "final",
        "checkpoint_selection": "none",
    }
    for key, expected in exact_model.items():
        if model.get(key) != expected:
            raise ValueError(f"blend model contract changed {key}")
    if model.get("positive_weight_key_order") != list(KEY_ORDER):
        raise ValueError("frozen positive-weight key order changed")
    if model.get("positive_weight") != list(EXPECTED_POSITIVE_WEIGHT):
        raise ValueError("frozen positive-weight vector changed")

    raw_arms = contract.get("arms")
    if not isinstance(raw_arms, list):
        raise ValueError("blend arms must be a list")
    arms: dict[str, Mapping[str, Any]] = {}
    for arm in raw_arms:
        if not isinstance(arm, Mapping) or not isinstance(arm.get("name"), str):
            raise ValueError("blend arm is malformed")
        name = str(arm["name"])
        if name in arms:
            raise ValueError(f"duplicate blend arm: {name}")
        arms[name] = arm
    if set(arms) != set(EXPECTED_ARMS):
        raise ValueError("blend arm set changed")
    for name, expected in EXPECTED_ARMS.items():
        arm = arms[name]
        for key, value in expected.items():
            if arm.get(key) != value:
                raise ValueError(f"blend arm {name} changed {key}")
        cycle = arm["five_step_cycle"]
        calculated = {
            source: (MAX_STEPS // CYCLE_STEPS)
            * sum(int(row[source]) for row in cycle)
            for source in arm["training_mix_percent"]
        }
        if calculated != arm["expected_draws"]:
            raise ValueError(f"blend arm {name} draw arithmetic changed")

    sources = contract.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("blend source contract is missing")
    nitrogen = sources.get("nitrogen")
    local = sources.get("local")
    wild = sources.get("wild_provisional")
    if not isinstance(nitrogen, Mapping) or nitrogen.get("sessions") != 1062:
        raise ValueError("NitroGen source membership count changed")
    if not isinstance(local, Mapping) or {
        "source_frames": local.get("source_frames"),
        "complete_segment_items": local.get("complete_segment_items"),
        "expected_draws_per_arm": local.get("expected_draws_per_arm"),
        "feature_validation_schema": local.get("feature_validation_schema"),
        "forbidden_generation": local.get("forbidden_generation"),
    } != {
        "source_frames": 143451,
        "complete_segment_items": 159,
        "expected_draws_per_arm": 22824,
        "feature_validation_schema": "madeleine.own-v3-features-validation.v1",
        "forbidden_generation": "/ephemeral/data/own_features",
    }:
        raise ValueError("corrected local source contract changed")
    if not isinstance(wild, Mapping) or {
        "tier": wild.get("tier"),
        "admitted_hours": wild.get("admitted_hours"),
        "sessions": wild.get("sessions"),
        "frames": wild.get("frames"),
        "complete_segment_items": wild.get("complete_segment_items"),
    } != {
        "tier": "provisional_not_train_ready",
        "admitted_hours": 0.0,
        "sessions": 2058,
        "frames": 4835638,
        "complete_segment_items": 41567,
    }:
        raise ValueError("provisional wild source contract changed")

    embargo = contract.get("embargo")
    if not isinstance(embargo, Mapping) or embargo.get(
        "sealed_untouched_session"
    ) != UNTOUCHED_SESSION:
        raise ValueError("sealed-session embargo changed")
    evaluation = contract.get("evaluation_contract")
    mapped = (
        evaluation.get("mapped_y4n_later_eight")
        if isinstance(evaluation, Mapping)
        else None
    )
    if not isinstance(mapped, Mapping) or {
        "sessions": mapped.get("sessions"),
        "all_sixteen_session_list_sha256": mapped.get(
            "all_sixteen_session_list_sha256"
        ),
        "truth_sha256": mapped.get("truth_sha256"),
    } != {
        "sessions": 8,
        "all_sixteen_session_list_sha256": (
            "5cc9428034fc07ef4a3d47781044e4c8ce5f89a3659530c70ee0400761ee4690"
        ),
        "truth_sha256": (
            "f61a0de4076f4683f01494837f01c3e314873ab0d78ee131b43e8e9f6e576a01"
        ),
    }:
        raise ValueError("mapped-y4n evaluation contract changed")


def _validate_nitrogen_receipt(
    path: Path, *, root: Path, expected_sha256: str
) -> None:
    _require_sha(path, expected_sha256, "full-corpus feature validation report")
    report = _json(path, "full-corpus feature validation report")
    if report.get("ok") is not True or report.get("errors") != []:
        raise ValueError("full-corpus feature validation did not pass cleanly")
    if report.get("deep_shards") is not True:
        raise ValueError("full-corpus feature validation was not deep")
    observed = report.get("observed")
    if not isinstance(observed, Mapping):
        raise ValueError("full-corpus feature validation lacks observations")
    exact = {
        "sessions": 1554,
        "unflagged_sessions": 1078,
        "deep_shards_checked": 1554,
        "hardlinks_checked": 1554,
        "skipped_short_frames": 0,
        "tail_truncated_frames": 0,
        "imputed_tail_frames": 0,
    }
    for key, expected in exact.items():
        if observed.get(key) != expected:
            raise ValueError(f"full-corpus validation changed {key}")
    paths = report.get("paths")
    output_root = paths.get("output_root") if isinstance(paths, Mapping) else None
    if not isinstance(output_root, str) or Path(output_root).resolve() != root.resolve():
        raise ValueError("full-corpus validation is bound to another output root")


def _validate_wild_receipt(
    path: Path,
    *,
    root: Path,
    expected_sha256: str,
    expected_source_sha256: str,
    wild_ids: Sequence[str],
    y4n_ids: Sequence[str],
) -> None:
    _require_sha(path, expected_sha256, "provisional wild feature validation marker")
    report = _json(path, "provisional wild feature validation marker")
    if report.get("format_version") != (
        "madeleine.wild-provisional-gru-features-validated.v1"
    ):
        raise ValueError("provisional wild feature receipt schema changed")
    if report.get("source_manifest_sha256") != expected_source_sha256:
        raise ValueError("provisional wild source manifest changed")
    if report.get("training_shards") != len(wild_ids):
        raise ValueError("provisional wild training shard count changed")
    if report.get("validation_shards") != len(y4n_ids):
        raise ValueError("provisional wild validation shard count changed")
    if report.get("total_hardlinks") != len(wild_ids) + len(y4n_ids):
        raise ValueError("provisional wild hard-link count changed")
    checks = report.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise ValueError("provisional wild feature validation checks changed")
    files = report.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("provisional wild feature receipt lacks file hashes")
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError("provisional wild feature file receipt is malformed")
        _require_sha(root / name, expected, f"provisional wild {name}")
    expected_npz = {f"{session_id}.npz" for session_id in (*wild_ids, *y4n_ids)}
    observed_npz = {path.name for path in root.glob("*.npz") if path.is_file()}
    if observed_npz != expected_npz:
        raise ValueError(
            "provisional wild NPZ inventory changed: "
            f"missing={sorted(expected_npz - observed_npz)[:3]} "
            f"extra={sorted(observed_npz - expected_npz)[:3]}"
        )


def _validate_local_feature_receipt(
    path: Path,
    *,
    root: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind corrected RGB source hashes to every generated own-v3 feature."""

    report = _json(path, "own-v3 feature validation marker")
    if report.get("schema_version") != contract.get("feature_validation_schema"):
        raise ValueError("own-v3 feature validation schema changed")
    if report.get("status") != "complete":
        raise ValueError("own-v3 feature build is incomplete")
    published = report.get("published_output")
    if not isinstance(published, str) or Path(published).resolve() != root.resolve():
        raise ValueError("own-v3 feature receipt is bound to another output root")
    if report.get("train_sessions") != list(LOCAL_TRAIN_IDS) or report.get(
        "validation_sessions"
    ) != list(LOCAL_VAL_IDS):
        raise ValueError("own-v3 feature receipt split changed")
    checks = report.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise ValueError("own-v3 feature validation checks changed")
    content = report.get("content")
    if not isinstance(content, Mapping):
        raise ValueError("own-v3 feature receipt lacks content")
    if report.get("content_sha256") != _canonical_sha256(content):
        raise ValueError("own-v3 feature receipt content hash changed")
    source = content.get("source")
    if not isinstance(source, Mapping) or source.get(
        "build_manifest_sha256"
    ) != contract.get("build_manifest_sha256"):
        raise ValueError("own-v3 RGB build manifest changed")
    split = source.get("split")
    if not isinstance(split, Mapping) or split.get(
        "train_sha256"
    ) != contract.get("train_sessions_sha256") or split.get(
        "validation_sha256"
    ) != contract.get("val_sessions_sha256"):
        raise ValueError("own-v3 RGB split receipts changed")
    expected_source_hashes = {
        **contract.get("train_shard_sha256", {}),
        **contract.get("val_a_shard_sha256", {}),
    }
    source_rows = source.get("sessions")
    if not isinstance(source_rows, list):
        raise ValueError("own-v3 RGB receipt lacks session rows")
    observed_source_hashes = {
        row.get("session_id"): row.get("source_npz_sha256")
        for row in source_rows
        if isinstance(row, Mapping)
    }
    if observed_source_hashes != expected_source_hashes:
        raise ValueError("own-v3 RGB shard hashes changed")

    entries = list(root.iterdir()) if root.is_dir() else []
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("own-v3 feature root contains non-regular entries")
    if {path.name for path in entries} != LOCAL_FEATURE_FILES:
        raise ValueError("own-v3 feature root inventory changed")
    inventory = content.get("feature_inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != LOCAL_FEATURE_FILES:
        raise ValueError("own-v3 feature receipt inventory changed")
    for name in sorted(LOCAL_FEATURE_FILES):
        row = inventory[name]
        if not isinstance(row, Mapping):
            raise ValueError(f"own-v3 feature receipt is malformed for {name}")
        feature_path = root / name
        if row.get("bytes") != feature_path.stat().st_size:
            raise ValueError(f"own-v3 feature byte count changed for {name}")
        _require_sha(feature_path, str(row.get("sha256")), f"own-v3 feature {name}")
    feature_rows = content.get("sessions")
    if not isinstance(feature_rows, list):
        raise ValueError("own-v3 feature receipt lacks session rows")
    for row in feature_rows:
        if not isinstance(row, Mapping):
            raise ValueError("own-v3 feature session receipt is malformed")
        session_id = row.get("session_id")
        expected_source = expected_source_hashes.get(session_id)
        if expected_source is None or row.get("source_npz_sha256") != expected_source:
            raise ValueError("own-v3 feature-to-RGB binding changed")
        expected_feature = inventory[f"{session_id}.npz"]["sha256"]
        if row.get("feature_npz_sha256") != expected_feature:
            raise ValueError("own-v3 feature shard receipt changed")
        supervision = row.get("supervision_equal_to_source")
        if not isinstance(supervision, Mapping) or not all(
            value is True for value in supervision.values()
        ):
            raise ValueError("own-v3 feature supervision is not source-identical")
    return report


def _link_sessions(
    *,
    source: Path,
    destination: Path,
    session_ids: Sequence[str],
    inventory: list[dict[str, Any]],
    source_name: str,
) -> None:
    for session_id in session_ids:
        source_path = source / f"{session_id}.npz"
        target_path = destination / f"{session_id}.npz"
        if not source_path.is_file():
            raise ValueError(f"missing {source_name} feature shard: {source_path}")
        if os.path.lexists(target_path):
            raise ValueError(f"feature-view session collision: {session_id}")
        os.link(source_path, target_path)
        source_stat = source_path.stat()
        target_stat = target_path.stat()
        if (
            source_stat.st_dev != target_stat.st_dev
            or source_stat.st_ino != target_stat.st_ino
            or source_stat.st_size != target_stat.st_size
            or target_stat.st_nlink < 2
        ):
            raise ValueError(f"hard-link verification failed: {session_id}")
        inventory.append(
            {
                "session_id": session_id,
                "source": source_name,
                "bytes": source_stat.st_size,
                "mtime_ns": source_stat.st_mtime_ns,
                "sha256": sha256_file(source_path),
            }
        )


def _write_hardlink_inventory(
    path: Path, inventory: Sequence[Mapping[str, Any]]
) -> str:
    ordered = sorted(inventory, key=lambda item: str(item["session_id"]))
    if len(ordered) != len({str(item["session_id"]) for item in ordered}):
        raise ValueError("hard-link inventory contains duplicate sessions")
    payload = "".join(
        json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in ordered
    )
    _write_text(path, payload)
    return sha256_file(path)


def _write_shard_hash_cache(
    path: Path, inventory: Sequence[Mapping[str, Any]]
) -> str:
    """Preseed train.py's cache with hashes already read for the receipt."""

    cache = {
        str(row["session_id"]): {
            "size": int(row["bytes"]),
            "mtime": (path.parent / f"{row['session_id']}.npz").stat().st_mtime,
            "sha256": str(row["sha256"]),
        }
        for row in inventory
    }
    # Match badeline.train._shard_hashes byte-for-byte.  The trainer always
    # rewrites this cache, even when every entry is a hit; identical formatting
    # keeps the content-bound assembled view unchanged during both arms.
    path.write_text(
        json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8"
    )
    return sha256_file(path)


def _sampling_config(
    *,
    sources: Mapping[str, Sequence[str]],
    step_cycle: Sequence[Mapping[str, int]],
) -> dict[str, Any]:
    return {
        "format_version": SAMPLER_SCHEMA,
        "expected_steps": MAX_STEPS,
        "cycle_steps": CYCLE_STEPS,
        "cycle_items": CYCLE_ITEMS,
        "sources": {name: list(values) for name, values in sources.items()},
        "step_cycle": [dict(row) for row in step_cycle],
    }


def _run_config(
    *,
    template: Mapping[str, Any],
    contract: Mapping[str, Any],
    arm: Mapping[str, Any],
    sources: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    config = dict(template)
    model = contract["model_contract"]
    positive_values = model["positive_weight"]
    if len(positive_values) != len(KEY_ORDER):
        raise ValueError("frozen positive-weight vector changed")
    config.update(
        {
            "max_steps": MAX_STEPS,
            "eval_interval": MAX_STEPS,
            "seed": 0,
            "frozen_positive_weight": {
                key: float(value)
                for key, value in zip(KEY_ORDER, positive_values, strict=True)
            },
            "source_sampling": _sampling_config(
                sources=sources,
                step_cycle=arm["five_step_cycle"],
            ),
            "_note": (
                "provisional blend diagnostic; exact source-balanced 14265-step "
                f"endpoint; final weights only; arm={arm['name']}"
            ),
        }
    )
    exact = {
        "window": 128,
        "frame_stride": 3,
        "window_mode": "centered",
        "segment_windows": 96,
        "batch_size": 1536,
        "learning_rate": 0.0003,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "linear_lr_decay": True,
        "transition_weight": 8.0,
        "class_balance": True,
        "precomputed_features": True,
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise ValueError(f"matched GRU template changed {key}")
    if config.get("input_config") != "pixels" or "temporal_arch" in config:
        raise ValueError("matched GRU architecture changed")
    return config


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    contract = _json(args.contract, "blend decision contract")
    _validate_contract(contract)
    contract_sha256 = sha256_file(args.contract)

    _require_sha(args.template, EXPECTED_TEMPLATE_SHA256, "matched GRU template")
    template = _json(args.template, "matched GRU template")
    source_contract = contract["sources"]
    nitrogen_contract = source_contract["nitrogen"]
    wild_contract = source_contract["wild_provisional"]
    local_contract = source_contract["local"]

    _require_sha(
        args.nitrogen_train,
        nitrogen_contract["session_list_sha256"],
        "NitroGen unflagged session list",
    )
    _validate_nitrogen_receipt(
        args.nitrogen_validation_report,
        root=args.nitrogen_root,
        expected_sha256=nitrogen_contract["feature_validation_sha256"],
    )
    _require_sha(
        args.wild_train,
        wild_contract["session_list_sha256"],
        "provisional wild session list",
    )

    nitrogen_ids = _sessions(args.nitrogen_train)
    wild_ids = _sessions(args.wild_train)
    local_ids = _sessions(args.local_root / "train_sessions.txt")
    local_val_ids = _sessions(args.local_root / "val_sessions.txt")
    y4n_ids = _sessions(args.y4n_sessions)
    _require_sha(
        args.y4n_sessions,
        contract["evaluation_contract"]["mapped_y4n_later_eight"][
            "all_sixteen_session_list_sha256"
        ],
        "mapped-y4n 16-session list",
    )
    _validate_wild_receipt(
        args.wild_validation_marker,
        root=args.wild_root,
        expected_sha256=wild_contract["feature_validation_marker_sha256"],
        expected_source_sha256=wild_contract["source_manifest_sha256"],
        wild_ids=wild_ids,
        y4n_ids=y4n_ids,
    )
    local_receipt = _validate_local_feature_receipt(
        args.local_validation_marker,
        root=args.local_root,
        contract=local_contract,
    )
    if len(nitrogen_ids) != nitrogen_contract["sessions"]:
        raise ValueError("NitroGen unflagged membership count changed")
    if len(wild_ids) != wild_contract["sessions"]:
        raise ValueError("provisional wild membership count changed")
    if local_ids != list(LOCAL_TRAIN_IDS) or local_val_ids != list(LOCAL_VAL_IDS):
        raise ValueError("own-v3 train/val-A membership changed")
    if len(y4n_ids) != 16 or any(
        not session.startswith("y4nQHqYSObI__") for session in y4n_ids
    ):
        raise ValueError("mapped-y4n holdout membership changed")
    groups = [set(nitrogen_ids), set(wild_ids), set(local_ids), set(local_val_ids), set(y4n_ids)]
    for index, first in enumerate(groups):
        for second in groups[index + 1 :]:
            if first & second:
                raise ValueError("blend source or evaluation membership overlaps")

    destination = args.out.resolve()
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if os.path.lexists(destination) or os.path.lexists(temporary):
        raise ValueError("refusing to overwrite blend feature view")
    temporary.mkdir(parents=True)
    inventory: list[dict[str, Any]] = []
    try:
        _link_sessions(
            source=args.nitrogen_root,
            destination=temporary,
            session_ids=[*nitrogen_ids, *y4n_ids],
            inventory=inventory,
            source_name="nitrogen",
        )
        _link_sessions(
            source=args.wild_root,
            destination=temporary,
            session_ids=wild_ids,
            inventory=inventory,
            source_name="wild_provisional",
        )
        _link_sessions(
            source=args.local_root,
            destination=temporary,
            session_ids=[*local_ids, *local_val_ids],
            inventory=inventory,
            source_name="local",
        )

        arm_by_name = {arm["name"]: arm for arm in contract["arms"]}
        arm_inputs = {
            "NL_90_10": {
                "nitrogen": nitrogen_ids,
                "local": local_ids,
            },
            "NLW_70_20_10": {
                "nitrogen": nitrogen_ids,
                "wild_provisional": wild_ids,
                "local": local_ids,
            },
        }
        generated: dict[str, str] = {}
        for name, sources in arm_inputs.items():
            arm = arm_by_name[name]
            session_file = temporary / f"train_{name.lower()}_sessions.txt"
            ordered = [session for values in sources.values() for session in values]
            _write_text(session_file, "\n".join(ordered) + "\n")
            config_file = temporary / f"config_{name.lower()}.json"
            _write_json(
                config_file,
                _run_config(
                    template=template,
                    contract=contract,
                    arm=arm,
                    sources=sources,
                ),
            )
            generated[session_file.name] = sha256_file(session_file)
            generated[config_file.name] = sha256_file(config_file)

        _write_text(temporary / "val_sessions.txt", "\n".join(y4n_ids) + "\n")
        _write_text(
            temporary / "later_eight_sessions.txt", "\n".join(y4n_ids[8:]) + "\n"
        )
        _write_text(
            temporary / "local_val_a_sessions.txt", "\n".join(local_val_ids) + "\n"
        )
        for name in ("val_sessions.txt", "later_eight_sessions.txt", "local_val_a_sessions.txt"):
            generated[name] = sha256_file(temporary / name)

        expected_link_count = (
            len(nitrogen_ids) + len(y4n_ids) + len(wild_ids)
            + len(local_ids) + len(local_val_ids)
        )
        if len(inventory) != expected_link_count:
            raise ValueError("hard-link inventory count changed")
        inventory_name = "hardlink_inventory.jsonl"
        inventory_sha256 = _write_hardlink_inventory(
            temporary / inventory_name, inventory
        )
        cache_name = "shard_hashes.json"
        cache_sha256 = _write_shard_hash_cache(temporary / cache_name, inventory)
        generated[inventory_name] = inventory_sha256
        generated[cache_name] = cache_sha256
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "study_id": contract["study_id"],
            "assembled_at": datetime.now(timezone.utc).isoformat(),
            "contract": {"path": str(args.contract), "sha256": contract_sha256},
            "sources": {
                "nitrogen": {
                    "train_sessions": len(nitrogen_ids),
                    "validation_report_sha256": sha256_file(
                        args.nitrogen_validation_report
                    ),
                },
                "wild_provisional": {
                    "train_sessions": len(wild_ids),
                    "admission_tier": "provisional_not_train_ready",
                    "admitted_hours": 0.0,
                    "validation_marker_sha256": sha256_file(
                        args.wild_validation_marker
                    ),
                },
                "local": {
                    "train_sessions": len(local_ids),
                    "val_a_sessions": len(local_val_ids),
                    "generation": "own-v3 corrected mask",
                    "validation_marker_sha256": sha256_file(
                        args.local_validation_marker
                    ),
                    "validation_content_sha256": local_receipt["content_sha256"],
                },
                "mapped_y4n": {"validation_sessions": len(y4n_ids)},
            },
            "hardlinks": {
                "files": len(inventory),
                "verified": True,
                "inventory_file": inventory_name,
                "inventory_sha256": inventory_sha256,
            },
            "generated_files": generated,
            "sealed_untouched_session_present": False,
            "temporary_files_present": False,
        }
        _write_json(temporary / "blend_feature_view_receipt.json", receipt)
        expected_files = {
            *(f"{row['session_id']}.npz" for row in inventory),
            *generated,
            "blend_feature_view_receipt.json",
        }
        entries = list(temporary.iterdir())
        if {path.name for path in entries} != expected_files:
            raise ValueError("assembled blend feature-view inventory changed")
        if any(path.is_symlink() or not path.is_file() for path in entries):
            raise ValueError("assembled blend feature view has non-regular entries")
        if any(path.name.startswith(".") and ".tmp" in path.name for path in entries):
            raise ValueError("temporary file remained in blend feature view")
        temporary.replace(destination)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--contract", type=Path, required=True)
    result.add_argument("--template", type=Path, required=True)
    result.add_argument("--nitrogen-root", type=Path, required=True)
    result.add_argument("--nitrogen-train", type=Path, required=True)
    result.add_argument("--nitrogen-validation-report", type=Path, required=True)
    result.add_argument("--y4n-sessions", type=Path, required=True)
    result.add_argument("--wild-root", type=Path, required=True)
    result.add_argument("--wild-train", type=Path, required=True)
    result.add_argument("--wild-validation-marker", type=Path, required=True)
    result.add_argument("--local-root", type=Path, required=True)
    result.add_argument("--local-validation-marker", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    receipt = assemble(args)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
