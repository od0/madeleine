"""Fail-closed orchestration checks for the corrected own-v3 six-run study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = "madeleine.own-v3-primary-rerun-complete.v1"
REGISTRATION_VERSION = "madeleine.own-v3-checkpoint-registration.v1"
EXPECTED_SUPPORT = {"all_frames": 29_086, "input_active_only": 25_028}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(repo), *args), text=True
    ).strip()


def _require_hash(path: Path, expected: str, description: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing {description}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{description} SHA-256 changed: expected {expected}, got {observed}"
        )


def _run_row(contract: Mapping[str, Any], family: str, seed: int) -> dict[str, Any]:
    rows = [
        row
        for row in contract.get("runs", [])
        if row.get("family") == family and row.get("seed") == seed
    ]
    if len(rows) != 1:
        raise ValueError(f"contract does not define exactly one {family} seed {seed}")
    return dict(rows[0])


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "madeleine.own-v3-primary-reruns.v1":
        raise ValueError("own-v3 rerun contract schema changed")
    runs = contract.get("runs")
    expected = {
        (family, seed)
        for family in ("scratch", "tier_b_init")
        for seed in range(3)
    }
    observed = {
        (row.get("family"), row.get("seed"))
        for row in runs if isinstance(row, Mapping)
    } if isinstance(runs, list) else set()
    if observed != expected or len(runs) != 6:
        raise ValueError("own-v3 six-run set changed")


def preflight(
    *,
    contract_path: Path,
    historical_repo: Path,
    control_repo: Path,
    results_root: Path,
    init_root: Path,
    family: str,
    seed: int,
) -> dict[str, Any]:
    contract = _json(contract_path, "own-v3 rerun contract")
    _validate_contract(contract)
    contract_hash = sha256_file(contract_path)
    row = _run_row(contract, family, seed)
    data = contract["data_contract"]
    training = contract["training_contract"]
    feature_root = Path(data["feature_root"])
    marker_path = Path(data["validation_marker"])
    if feature_root.resolve() == Path(data["rejected_legacy_feature_root"]).resolve():
        raise ValueError("legacy own-feature generation is forbidden")
    _require_hash(marker_path, data["validation_marker_sha256"], "feature receipt")
    marker = _json(marker_path, "feature receipt")
    exact_marker = {
        "schema_version": data["validation_schema"],
        "status": "complete",
        "content_sha256": data["content_sha256"],
        "session_count": 4,
        "frame_count": data["frame_count"],
        "train_sessions": data["train_sessions"],
        "validation_sessions": data["validation_sessions"],
        "published_output": str(feature_root),
    }
    for key, expected in exact_marker.items():
        if marker.get(key) != expected:
            raise ValueError(f"feature receipt changed {key}")
    implementation = marker.get("content", {}).get("implementation", {})
    if implementation.get("commit") != data["feature_generation"]["commit"]:
        raise ValueError("feature-generation commit changed")
    if implementation.get("relevant_files") != data["feature_generation"]["relevant_files"]:
        raise ValueError("feature-generation implementation hashes changed")

    expected_files = data["feature_files"]
    runtime_files = data.get("permitted_runtime_files", {})
    observed_files = {path.name for path in feature_root.iterdir() if path.is_file()}
    allowed_inventories = (set(expected_files), set(expected_files) | set(runtime_files))
    if observed_files not in allowed_inventories:
        raise ValueError("corrected feature-cache inventory changed")
    for name, receipt in expected_files.items():
        path = feature_root / name
        _require_hash(path, receipt["sha256"], f"own-v3 feature {name}")
        if path.stat().st_size != receipt["bytes"]:
            raise ValueError(f"own-v3 feature byte count changed: {name}")
    for name, receipt in runtime_files.items():
        path = feature_root / name
        if path.exists():
            _require_hash(path, receipt["sha256"], f"own-v3 runtime cache {name}")
            if path.stat().st_size != receipt["bytes"]:
                raise ValueError(f"own-v3 runtime-cache byte count changed: {name}")

    historical_commit = training["historical_commit"]
    if _git(historical_repo, "rev-parse", "HEAD") != historical_commit:
        raise ValueError("historical training repository is not at the pinned commit")
    historical_files = {
        "badeline/train.py": training["trainer_sha256"],
        "badeline/model.py": training["model_sha256"],
        "data/schema.py": training["schema_sha256"],
    }
    for relative, expected in historical_files.items():
        _require_hash(historical_repo / relative, expected, f"historical {relative}")
    family_contract = training["families"][family]
    config = historical_repo / family_contract["config"]
    _require_hash(config, family_contract["config_sha256"], f"{family} config")

    evaluator = contract["evaluation_contract"]["evaluator_relevant_files"]
    for relative, expected in evaluator.items():
        _require_hash(control_repo / relative, expected, f"evaluator {relative}")

    initializer: Path | None = None
    initializer_sha256: str | None = None
    if family == "tier_b_init":
        init = training["tier_b_initializations"][str(seed)]
        initializer = init_root / init["relative_path"]
        initializer_sha256 = init["sha256"]
        _require_hash(initializer, initializer_sha256, "Tier-B initializer")

    run_id = row["run_id"]
    report = results_root / f"{run_id}_val_a.json"
    sidecar = results_root / f"{run_id}_val_a_preds.npz"
    marker_out = results_root / f".{run_id}_val_a_done.json"
    run = results_root / run_id
    for path in (run, report, sidecar, marker_out):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite own-v3 artifact: {path}")
    return {
        "contract_sha256": contract_hash,
        "run_id": run_id,
        "family": family,
        "seed": seed,
        "config": str(config),
        "max_steps": family_contract["max_steps"],
        "initializer": str(initializer) if initializer else None,
        "initializer_sha256": initializer_sha256,
        "feature_root": str(feature_root),
        "train_sessions": str(feature_root / "train_sessions.txt"),
        "validation_sessions": str(feature_root / "val_sessions.txt"),
        "run": str(run),
        "report": str(report),
        "sidecar": str(sidecar),
        "completion_marker": str(marker_out),
    }


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"checkpoint tensor contains non-finite values: {name}")
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _states_identical(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
        for name in left
    )


def register_checkpoint(
    *,
    contract_path: Path,
    historical_repo: Path,
    run: Path,
    family: str,
    seed: int,
    registry: Path,
) -> dict[str, Any]:
    contract = _json(contract_path, "own-v3 rerun contract")
    _validate_contract(contract)
    row = _run_row(contract, family, seed)
    if run.name != row["run_id"]:
        raise ValueError("run directory identity differs from contract")
    for name in ("model.pt", "config.json", "run_meta.json", "log.jsonl"):
        path = run / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"training artifact is absent or empty: {path}")

    data = contract["data_contract"]
    training = contract["training_contract"]
    family_contract = training["families"][family]
    config = _json(run / "config.json", "resolved training config")
    source_config = json.loads(
        _git(
            historical_repo,
            "show",
            f"{training['historical_commit']}:{family_contract['config']}",
        )
    )
    source_config["seed"] = seed
    if config != source_config:
        raise ValueError("resolved training config differs from historical recipe")
    meta_path = run / "run_meta.json"
    meta = _json(meta_path, "training run metadata")
    if meta.get("seed") != seed or meta.get("config") != config:
        raise ValueError("training metadata seed/config mismatch")
    expected_split = {
        "train": data["train_sessions"],
        "val": data["validation_sessions"],
    }
    if meta.get("split") != expected_split:
        raise ValueError("training metadata split differs from own-v3 contract")
    expected_shards = {
        f"{session}.npz": data["feature_files"][f"{session}.npz"]["sha256"]
        for session in (*data["train_sessions"], *data["validation_sessions"])
    }
    observed_shards = {
        f"{session}.npz": digest
        for session, digest in meta.get("shard_sha256", {}).items()
    }
    if observed_shards != expected_shards:
        raise ValueError("training metadata shard hashes differ from own-v3 contract")
    expected_init = (
        training["tier_b_initializations"][str(seed)]["sha256"]
        if family == "tier_b_init" else None
    )
    initialized_from = meta.get("initialized_from")
    if (initialized_from is None) != (expected_init is None):
        raise ValueError("checkpoint initialization presence changed")
    if expected_init is not None:
        _require_hash(Path(initialized_from), expected_init, "recorded initializer")

    checkpoint_path = run / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("steps") != family_contract["max_steps"]:
        raise ValueError("checkpoint endpoint differs from contract")
    if checkpoint.get("config") != config:
        raise ValueError("checkpoint config differs from resolved config")
    selected = checkpoint.get("model_state_dict")
    final = checkpoint.get("final_state_dict")
    if not isinstance(selected, Mapping) or not isinstance(final, Mapping):
        raise ValueError("checkpoint lacks selected/final tensor mappings")
    selected_hash = _state_digest(selected)
    final_hash = _state_digest(final)
    checkpoint_hash = sha256_file(checkpoint_path)
    original_meta_hash = sha256_file(meta_path)
    meta["own_v3_provenance"] = {
        "contract_sha256": sha256_file(contract_path),
        "feature_validation_marker_sha256": data["validation_marker_sha256"],
        "feature_content_sha256": data["content_sha256"],
        "feature_generation": data["feature_generation"],
        "feature_files": data["feature_files"],
        "historical_training_commit": training["historical_commit"],
        "historical_trainer_sha256": training["trainer_sha256"],
        "historical_model_sha256": training["model_sha256"],
        "family": family,
        "run_id": row["run_id"],
        "initializer_sha256": expected_init,
        "original_run_meta_sha256": original_meta_hash,
    }
    _write_json(meta_path, meta)
    registration = {
        "schema_version": REGISTRATION_VERSION,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run_id": row["run_id"],
        "family": family,
        "seed": seed,
        "checkpoint": {
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_hash,
            "selected_tensor_sha256": selected_hash,
            "final_tensor_sha256": final_hash,
            "selected_final_identical": _states_identical(selected, final),
            "best_val_step": checkpoint.get("best_val_step"),
            "best_val_mean_bce": checkpoint.get("best_val_mean_bce"),
        },
        "run_meta_sha256": sha256_file(meta_path),
        "config_sha256": sha256_file(run / "config.json"),
        "contract_sha256": sha256_file(contract_path),
    }
    _write_json(run / "checkpoint-registration.json", registration)

    registry.parent.mkdir(parents=True, exist_ok=True)
    line = f"{checkpoint_hash}  own_v3/{row['run_id']}_model.pt"
    with registry.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        lines = [value.rstrip("\n") for value in handle]
        if line not in lines:
            if any(row["run_id"] in value for value in lines):
                raise ValueError("node registry already has another hash for this run")
            handle.write(line + "\n")
            handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return registration


def finalize(
    *,
    contract_path: Path,
    run: Path,
    report_path: Path,
    marker_path: Path,
    family: str,
    seed: int,
) -> dict[str, Any]:
    contract = _json(contract_path, "own-v3 rerun contract")
    row = _run_row(contract, family, seed)
    if run.name != row["run_id"]:
        raise ValueError("run identity differs from contract")
    registration = _json(run / "checkpoint-registration.json", "checkpoint registration")
    checkpoint = run / "model.pt"
    if registration["checkpoint"]["sha256"] != sha256_file(checkpoint):
        raise ValueError("checkpoint changed after registry creation")
    report = _json(report_path, "val-A report")
    if report.get("sessions") != contract["data_contract"]["validation_sessions"]:
        raise ValueError("val-A report membership changed")
    if report.get("weights") != "selected":
        raise ValueError("val-A report did not evaluate selected weights")
    required_metrics = {
        "per_key_ap", "per_key_f1", "per_key_calibration",
        "onset_timing_errors", "transition_f1_at_0.5",
        "transition_f1_oracle", "transition_f1_oracle_collars",
    }
    for population, expected_n in EXPECTED_SUPPORT.items():
        value = report.get(population)
        if not isinstance(value, Mapping) or value.get("n") != expected_n:
            raise ValueError(f"val-A support changed for {population}")
        metrics = value.get("metrics")
        if not isinstance(metrics, Mapping) or not required_metrics.issubset(metrics):
            raise ValueError(f"val-A metrics incomplete for {population}")
    sidecar = report_path.with_name(report_path.stem + "_preds.npz")
    with np.load(sidecar, allow_pickle=False) as archive:
        if set(archive.files) != {
            "y_true", "y_prob", "input_active", "session_lengths", "session_ids"
        }:
            raise ValueError("val-A prediction sidecar fields changed")
        truth = np.asarray(archive["y_true"])
        probability = np.asarray(archive["y_prob"])
        active = np.asarray(archive["input_active"])
        lengths = np.asarray(archive["session_lengths"])
        session_ids = np.asarray(archive["session_ids"])
    if truth.shape != (EXPECTED_SUPPORT["all_frames"], 7) or probability.shape != truth.shape:
        raise ValueError("val-A prediction arrays have unexpected shape")
    if active.shape != (len(truth),) or int(active.sum()) != EXPECTED_SUPPORT["input_active_only"]:
        raise ValueError("val-A input-active support changed")
    if lengths.ndim != 1 or int(lengths.sum()) != len(truth) or len(session_ids) != len(lengths):
        raise ValueError("val-A stream boundaries are unaligned")
    if not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("val-A probabilities are not finite probabilities")
    if not np.all(np.isin(truth, (0, 1))) or not np.all(np.isin(active, (0, 1))):
        raise ValueError("val-A truth or activity arrays are non-binary")
    if marker_path.exists() or marker_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite completion marker: {marker_path}")
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (
            checkpoint,
            run / "config.json",
            run / "run_meta.json",
            run / "log.jsonl",
            run / "checkpoint-registration.json",
            report_path,
            sidecar,
        )
    }
    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_id": row["run_id"],
        "family": family,
        "seed": seed,
        "contract_sha256": sha256_file(contract_path),
        "support": EXPECTED_SUPPORT,
        "weights": "selected",
        "artifacts": artifacts,
    }
    _write_json(marker_path, completion)
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--contract", required=True, type=Path)
    common.add_argument("--family", required=True, choices=("scratch", "tier_b_init"))
    common.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    pre = sub.add_parser("preflight", parents=[common])
    pre.add_argument("--historical-repo", required=True, type=Path)
    pre.add_argument("--control-repo", required=True, type=Path)
    pre.add_argument("--results-root", required=True, type=Path)
    pre.add_argument("--init-root", required=True, type=Path)
    register = sub.add_parser("register", parents=[common])
    register.add_argument("--historical-repo", required=True, type=Path)
    register.add_argument("--run", required=True, type=Path)
    register.add_argument("--registry", required=True, type=Path)
    finish = sub.add_parser("finalize", parents=[common])
    finish.add_argument("--run", required=True, type=Path)
    finish.add_argument("--report", required=True, type=Path)
    finish.add_argument("--marker", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(
            contract_path=args.contract,
            historical_repo=args.historical_repo,
            control_repo=args.control_repo,
            results_root=args.results_root,
            init_root=args.init_root,
            family=args.family,
            seed=args.seed,
        )
    elif args.command == "register":
        result = register_checkpoint(
            contract_path=args.contract,
            historical_repo=args.historical_repo,
            run=args.run,
            family=args.family,
            seed=args.seed,
            registry=args.registry,
        )
    else:
        result = finalize(
            contract_path=args.contract,
            run=args.run,
            report_path=args.report,
            marker_path=args.marker,
            family=args.family,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
