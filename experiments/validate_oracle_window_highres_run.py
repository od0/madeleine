"""Independent checkpoint replay and publication audit for frozen Study H."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.oracle_window_highres_regional import (
    ARMS,
    FullResolutionOracleDataset,
    HighResolutionRegionalLocalizer,
    arm_geometry,
    predict_model,
    state_dict_sha256,
    validate_cache,
    validate_implementation,
)
from experiments.oracle_window_localization import sha256_file
from experiments.score_oracle_window_highres_regional import (
    INDEX_FIELDS,
    _validate_run,
    score_highres,
)


FROZEN_CONFIG_SHA256 = "7144e68f65acb75a7ae5712330d162f20e16f0fa9bd9440c7f58f67e7962e75f"
AUDIT_SCHEMA = "madeleine.oracle-window-highres-audit.v1"
AUDIT_MARKER_SCHEMA = "madeleine.oracle-window-highres-audit-complete.v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_completion_marker(
    *,
    marker_path: Path,
    report_path: Path,
    runs: Mapping[str, Path],
    config_path: Path,
    cache_receipt_path: Path,
    base_manifest_path: Path,
) -> dict[str, Any]:
    marker = _json(marker_path)
    content = dict(marker)
    observed = content.pop("content_sha256", None)
    if observed != _canonical_sha256(content):
        raise ValueError("Study-H completion marker content hash changed")
    if marker.get("status") != "complete" or marker.get("execution_mode") != "production":
        raise ValueError("Study-H completion marker is not production-complete")
    bindings = {
        "report": (report_path, "report"),
        "config": (config_path, "config"),
        "cache_receipt": (cache_receipt_path, "cache_receipt"),
        "base_dataset_manifest": (base_manifest_path, "base_dataset_manifest"),
    }
    for _, (path, field) in bindings.items():
        if marker[field]["sha256"] != sha256_file(path):
            raise ValueError(f"Study-H completion binding changed: {field}")
    for arm in ARMS:
        bound = marker["runs"][arm]
        for filename, field in (
            ("run_receipt.json", "run_receipt_sha256"),
            ("model.pt", "checkpoint_sha256"),
            ("predictions.npz", "prediction_sidecar_sha256"),
        ):
            if bound[field] != sha256_file(runs[arm] / filename):
                raise ValueError(f"Study-H completion run binding changed: {arm}:{filename}")
    return marker


def audit_run(
    *,
    runs: Mapping[str, Path],
    config_path: Path,
    cache: Path,
    cache_receipt_path: Path,
    base_manifest_path: Path,
    report_path: Path,
    marker_path: Path,
    expected_source_commit: str,
    device_name: str,
) -> dict[str, Any]:
    if sha256_file(config_path) != FROZEN_CONFIG_SHA256:
        raise ValueError("Study-H frozen config hash changed")
    config = _json(config_path)
    repo = config_path.resolve().parents[2]
    implementation = validate_implementation(config, repo=repo)
    if implementation["git_head"] != expected_source_commit:
        raise ValueError("Study-H audit checkout differs from assigned source commit")
    cache_receipt = validate_cache(
        cache.resolve(),
        cache_receipt_path.resolve(),
        expected_receipt_sha256=str(config["dataset"]["cache_receipt_sha256"]),
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(device_name)
    torch.use_deterministic_algorithms(True, warn_only=(device.type != "cuda"))
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset = FullResolutionOracleDataset(cache, split="validation")
    loaded: dict[str, dict[str, Any]] = {}
    replay: dict[str, Any] = {}
    seed = int(config["training"]["seed"])
    for arm in ARMS:
        loaded[arm] = _validate_run(
            runs[arm],
            arm=arm,
            config_path=config_path,
            cache_receipt_path=cache_receipt_path,
            width=int(config["dataset"]["candidate_width"]),
            smoke=False,
        )
        receipt = loaded[arm]["receipt"]
        if receipt["implementation"]["git_head"] != expected_source_commit:
            raise ValueError(f"Study-H run source commit changed: {arm}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model_config = config["model"]
        model = HighResolutionRegionalLocalizer(
            token_dim=int(model_config["token_dim"]),
            temporal_kernel=int(model_config["temporal_kernel"]),
            imagenet_weights=False,
        )
        checkpoint = loaded[arm]["checkpoint"]
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if state_dict_sha256(model) != checkpoint["model_state_sha256"]:
            raise ValueError(f"Study-H checkpoint tensor hash changed: {arm}")
        model.to(device)
        probability, entropy, attention = predict_model(
            model,
            dataset,
            arm=arm,
            device=device,
            batch_size=int(config["training"]["eval_batch_size"]),
            cuda_bf16=bool(config["training"]["cuda_bf16"]),
        )
        expected = loaded[arm]["arrays"]
        if not np.array_equal(probability, expected["probability"]):
            raise ValueError(f"Study-H exact probability replay changed: {arm}")
        if not np.array_equal(entropy, expected["spatial_attention_entropy"]):
            raise ValueError(f"Study-H exact attention entropy replay changed: {arm}")
        if not np.array_equal(attention, expected["attention_mean_by_head"]):
            raise ValueError(f"Study-H exact attention map replay changed: {arm}")
        for name in INDEX_FIELDS:
            if not np.array_equal(dataset.metadata[name], expected[name]):
                raise ValueError(f"Study-H replay metadata changed: {arm}:{name}")
        replay[arm] = {
            "checkpoint_sha256": sha256_file(runs[arm] / "model.pt"),
            "model_state_sha256": checkpoint["model_state_sha256"],
            "prediction_sidecar_sha256": sha256_file(runs[arm] / "predictions.npz"),
            "exact_probability_replay": True,
            "exact_attention_replay": True,
        }

    regenerated = score_highres(
        runs=runs,
        config_path=config_path,
        cache_receipt_path=cache_receipt_path,
        base_manifest_path=base_manifest_path,
        smoke=False,
    )
    recorded = _json(report_path)
    if regenerated != recorded:
        raise ValueError("Study-H report does not exactly regenerate")
    completion = validate_completion_marker(
        marker_path=marker_path,
        report_path=report_path,
        runs=runs,
        config_path=config_path,
        cache_receipt_path=cache_receipt_path,
        base_manifest_path=base_manifest_path,
    )
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "complete",
        "study_id": config["study_id"],
        "source_commit": expected_source_commit,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "cache_receipt_sha256": sha256_file(cache_receipt_path),
        "cache_content_sha256": cache_receipt["content_sha256"],
        "report_sha256": sha256_file(report_path),
        "completion_marker_sha256": sha256_file(marker_path),
        "decision": recorded["decision"],
        "implementation": implementation,
        "replay": replay,
        "checks": {
            "exact_checkpoint_tensor_hashes": True,
            "exact_probability_sidecar_replay": True,
            "exact_attention_sidecar_replay": True,
            "exact_fixed_policy_report_regeneration": True,
            "content_bound_completion_marker": bool(completion["content_sha256"]),
        },
    }


def publish_audit(*, audit: Mapping[str, Any], out: Path, marker: Path) -> None:
    for path in (out, marker):
        if os.path.lexists(path):
            raise ValueError(f"refusing to overwrite Study-H audit publication: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp")
    temporary.write_text(
        json.dumps(audit, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if _json(temporary) != audit:
        raise ValueError("serialized Study-H audit changed")
    temporary.replace(out)
    marker_base = {
        "schema_version": AUDIT_MARKER_SCHEMA,
        "status": "complete",
        "audit": {"path": str(out.resolve()), "sha256": sha256_file(out)},
        "study_id": audit["study_id"],
        "source_commit": audit["source_commit"],
        "decision": audit["decision"]["status"],
    }
    marker_value = dict(marker_base)
    marker_value["content_sha256"] = _canonical_sha256(marker_base)
    marker_tmp = marker.with_name(f".{marker.name}.tmp")
    marker_tmp.write_text(
        json.dumps(marker_value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if _json(marker_tmp) != marker_value:
        raise ValueError("serialized Study-H audit marker changed")
    marker_tmp.replace(marker)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm.replace('_', '-')}-run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--cache-receipt", required=True, type=Path)
    parser.add_argument("--base-dataset-manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--completion-marker", required=True, type=Path)
    parser.add_argument("--expect-source-commit", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    runs = {arm: getattr(args, f"{arm}_run") for arm in ARMS}
    audit = audit_run(
        runs=runs,
        config_path=args.config,
        cache=args.cache,
        cache_receipt_path=args.cache_receipt,
        base_manifest_path=args.base_dataset_manifest,
        report_path=args.report,
        marker_path=args.completion_marker,
        expected_source_commit=args.expect_source_commit,
        device_name=args.device,
    )
    publish_audit(audit=audit, out=args.out, marker=args.marker)
    print(json.dumps({"status": audit["status"], "decision": audit["decision"]["status"]}))


if __name__ == "__main__":
    main()
