"""Fixed-policy scorer for the three matched Study-H arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.oracle_window_highres_regional import (
    ARMS,
    CHECKPOINT_SCHEMA,
    INDEX_FIELDS,
    RUN_SCHEMA,
    SIDECAR_SCHEMA,
)
from experiments.oracle_window_localization import HEAD_NAMES, sha256_file
from experiments.score_oracle_window_localization import (
    analytic_uniform_chance,
    arm_metrics,
    estimable_heads,
    paired_block_bootstrap,
    per_head_delta_bootstrap,
)


REPORT_SCHEMA = "madeleine.oracle-window-highres-score.v1"
MARKER_SCHEMA = "madeleine.oracle-window-highres-complete.v1"
RUN_FILES = frozenset({"model.pt", "training_log.json", "predictions.npz", "run_receipt.json"})
SIDECAR_FIELDS = frozenset(
    {
        *INDEX_FIELDS,
        "schema_version",
        "arm",
        "probability",
        "spatial_attention_entropy",
        "attention_mean_by_head",
    }
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_run(
    run: Path,
    *,
    arm: str,
    config_path: Path,
    cache_receipt_path: Path,
    width: int,
    smoke: bool,
) -> dict[str, Any]:
    if {path.name for path in run.iterdir()} != RUN_FILES:
        raise ValueError(f"Study-H run inventory changed: {arm}")
    receipt_path = run / "run_receipt.json"
    receipt = _json(receipt_path)
    if receipt.get("schema_version") != RUN_SCHEMA or receipt.get("arm") != arm:
        raise ValueError(f"Study-H run identity changed: {arm}")
    expected_status = (
        "smoke_predictions_complete_unscored"
        if smoke
        else "predictions_complete_unscored"
    )
    expected_mode = "smoke" if smoke else "production"
    if receipt.get("status") != expected_status:
        raise ValueError(f"Study-H run is incomplete: {arm}")
    if receipt.get("execution_mode") != expected_mode:
        raise ValueError(f"Study-H run mode changed: {arm}")
    if receipt.get("config_sha256") != sha256_file(config_path):
        raise ValueError(f"Study-H run config changed: {arm}")
    if receipt.get("cache_receipt_sha256") != sha256_file(cache_receipt_path):
        raise ValueError(f"Study-H run cache binding changed: {arm}")
    content = dict(receipt)
    observed_content = content.pop("content_sha256", None)
    if observed_content != _canonical_sha256(content):
        raise ValueError(f"Study-H run receipt content hash changed: {arm}")
    bindings = {
        "model.pt": "checkpoint_sha256",
        "training_log.json": "training_log_sha256",
        "predictions.npz": "prediction_sidecar_sha256",
    }
    for name, field in bindings.items():
        if receipt.get(field) != sha256_file(run / name):
            raise ValueError(f"Study-H run artifact hash changed: {arm}:{name}")

    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    expected_checkpoint_fields = {
        "schema_version", "arm", "config_sha256", "cache_receipt_sha256",
        "initial_state_sha256", "model_state_sha256", "seed", "epochs",
        "model_state_dict",
    }
    if set(checkpoint) != expected_checkpoint_fields:
        raise ValueError(f"Study-H checkpoint inventory changed: {arm}")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA or checkpoint.get("arm") != arm:
        raise ValueError(f"Study-H checkpoint identity changed: {arm}")
    for field in (
        "config_sha256",
        "cache_receipt_sha256",
        "initial_state_sha256",
        "model_state_sha256",
        "seed",
        "epochs",
    ):
        if checkpoint.get(field) != receipt.get(field):
            raise ValueError(f"Study-H checkpoint binding changed: {arm}:{field}")
    config = _json(config_path)
    if receipt.get("seed") != int(config["training"]["seed"]):
        raise ValueError(f"Study-H run seed changed: {arm}")
    if receipt.get("configured_epochs") != int(config["training"]["epochs"]):
        raise ValueError(f"Study-H configured endpoint changed: {arm}")
    if not smoke and receipt.get("epochs") != int(config["training"]["epochs"]):
        raise ValueError(f"Study-H production endpoint changed: {arm}")
    expected_support = config["dataset"]["expected_support"]
    if receipt.get("full_train_examples") != int(expected_support["training_examples"]):
        raise ValueError(f"Study-H full training support changed: {arm}")
    if receipt.get("full_validation_examples") != int(expected_support["validation_examples"]):
        raise ValueError(f"Study-H full validation support changed: {arm}")
    if not smoke and (
        receipt.get("train_examples") != int(expected_support["training_examples"])
        or receipt.get("validation_examples") != int(expected_support["validation_examples"])
    ):
        raise ValueError(f"Study-H production support changed: {arm}")

    with np.load(run / "predictions.npz", allow_pickle=False) as archive:
        if set(archive.files) != SIDECAR_FIELDS:
            raise ValueError(f"Study-H sidecar inventory changed: {arm}")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if str(arrays.pop("schema_version").reshape(()).item()) != SIDECAR_SCHEMA:
        raise ValueError(f"Study-H sidecar schema changed: {arm}")
    if str(arrays.pop("arm").reshape(()).item()) != arm:
        raise ValueError(f"Study-H sidecar arm changed: {arm}")
    probability = arrays["probability"]
    if probability.dtype != np.float32 or probability.ndim != 2 or probability.shape[1] != width:
        raise ValueError(f"Study-H probability geometry changed: {arm}")
    if not np.all(np.isfinite(probability)) or not np.allclose(probability.sum(1), 1.0, atol=1e-5):
        raise ValueError(f"Study-H probabilities are invalid: {arm}")
    entropy = arrays["spatial_attention_entropy"]
    if entropy.dtype != np.float32 or entropy.shape != (len(probability),) or not np.all(np.isfinite(entropy)):
        raise ValueError(f"Study-H attention entropy changed: {arm}")
    attention = arrays["attention_mean_by_head"]
    expected_grid = 2 if arm == "h32_q" else 8
    if attention.dtype != np.float32 or attention.shape != (len(HEAD_NAMES), expected_grid, expected_grid):
        raise ValueError(f"Study-H attention summary geometry changed: {arm}")
    if not np.all(np.isfinite(attention)):
        raise ValueError(f"Study-H attention summary is non-finite: {arm}")
    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "checkpoint": checkpoint,
        "arrays": arrays,
    }


def _same_metadata(reference: Mapping[str, np.ndarray], observed: Mapping[str, np.ndarray]) -> None:
    for name in INDEX_FIELDS:
        left = reference[name]
        right = observed[name]
        if left.dtype != right.dtype or not np.array_equal(left, right):
            raise ValueError(f"Study-H arm metadata differs: {name}")


def _macro_attention(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    heads = arrays["head_index"].astype(np.int64)
    entropy = arrays["spatial_attention_entropy"]
    per_head = {
        name: {
            "support": int((heads == index).sum()),
            "mean_spatial_attention_entropy": float(entropy[heads == index].mean()),
            "mean_spatial_attention": arrays["attention_mean_by_head"][index].tolist(),
        }
        for index, name in enumerate(HEAD_NAMES)
    }
    return {
        "pooled_mean_spatial_attention_entropy": float(entropy.mean()),
        "per_head": per_head,
    }


def _gate(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    estimable: Sequence[str],
    gate_name: str,
) -> dict[str, Any]:
    selected = config["decision_gate"][gate_name]
    candidate_macro = candidate["macro_estimable"]
    control_macro = control["macro_estimable"]
    delta = float(candidate_macro["exact"] - control_macro["exact"])
    within_delta = float(candidate_macro["within_2"] - control_macro["within_2"])
    positive = [
        name
        for name in estimable
        if candidate["per_head"][name]["exact"] > control["per_head"][name]["exact"]
    ]
    positive_keys = {name.split(":", 1)[0] for name in positive}
    positive_types = {name.split(":", 1)[1] for name in positive}
    checks = {
        "minimum_candidate_macro_exact": candidate_macro["exact"] >= float(selected["minimum_candidate_macro_exact"]),
        "candidate_ci_low_above_chance": float(bootstrap["conditional_macro_95"][0]) > float(selected["minimum_candidate_exact_ci_low"]),
        "minimum_macro_exact_delta": delta >= float(selected["minimum_macro_exact_delta"]),
        "delta_ci_low_above_zero": float(bootstrap["delta_macro_95"][0]) > float(selected["minimum_macro_exact_delta_ci_low"]),
        "minimum_positive_heads": len(positive) >= int(selected["minimum_positive_estimable_heads"]),
        "minimum_positive_keys": len(positive_keys) >= int(selected["minimum_positive_distinct_keys"]),
        "positive_both_event_types": positive_types == {"onset", "release"},
        "within_2_noninferiority": within_delta >= float(selected["minimum_macro_within_2_delta"]),
        "lower_nll": candidate_macro["nll"] < control_macro["nll"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_estimable_macro": candidate_macro,
        "control_estimable_macro": control_macro,
        "macro_exact_delta": delta,
        "macro_within_2_delta": within_delta,
        "positive_estimable_heads": positive,
        "positive_distinct_keys": sorted(positive_keys),
        "estimable_heads": list(estimable),
        "validation_blocks": int(manifest["validation_block_count"]),
    }


def score_highres(
    *,
    runs: Mapping[str, Path],
    config_path: Path,
    cache_receipt_path: Path,
    base_manifest_path: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    config = _json(config_path)
    manifest = _json(base_manifest_path)
    if sha256_file(base_manifest_path) != str(config["dataset"]["base_dataset_manifest_sha256"]):
        raise ValueError("Study-H base dataset manifest hash changed")
    width = int(config["dataset"]["candidate_width"])
    loaded = {
        arm: _validate_run(
            runs[arm], arm=arm, config_path=config_path,
            cache_receipt_path=cache_receipt_path, width=width, smoke=smoke,
        )
        for arm in ARMS
    }
    reference_arrays = loaded["h32_q"]["arrays"]
    for arm in ARMS[1:]:
        _same_metadata(reference_arrays, loaded[arm]["arrays"])
    initial_hashes = {loaded[arm]["receipt"]["initial_state_sha256"] for arm in ARMS}
    if len(initial_hashes) != 1:
        raise ValueError("Study-H arms do not share exact initialization")
    if len({loaded[arm]["receipt"]["model"]["trainable_parameters"] for arm in ARMS}) != 1:
        raise ValueError("Study-H arms are not parameter matched")

    truth = reference_arrays["true_offset"].astype(np.int64)
    heads = reference_arrays["head_index"].astype(np.int64)
    blocks = reference_arrays["block_id"]
    selected_names = estimable_heads(manifest, config["decision_gate"]["estimability"])
    selected_indices = [HEAD_NAMES.index(name) for name in selected_names]
    metrics: dict[str, Any] = {}
    for arm in ARMS:
        metrics[arm] = arm_metrics(
            loaded[arm]["arrays"]["probability"], truth, heads, width=width
        )
        metrics[arm]["macro_estimable"] = {
            key: float(np.mean([metrics[arm]["per_head"][name][key] for name in selected_names]))
            for key in ("exact", "within_1", "within_2", "nll", "entropy")
        }
        metrics[arm]["spatial_diagnostics"] = _macro_attention(loaded[arm]["arrays"])

    if smoke:
        primary_gate: dict[str, Any] = {
            "passed": False,
            "not_evaluated": "smoke support is never scientific evidence",
        }
        attribution_gate: dict[str, Any] = {
            "passed": False,
            "not_evaluated": "smoke support is never scientific evidence",
        }
        decision = "smoke_only_no_scientific_decision"
    else:
        evaluation = config["evaluation"]
        primary_correct = (loaded["h128_q"]["arrays"]["probability"].argmax(1) == truth).astype(np.float64)
        h32_correct = (loaded["h32_q"]["arrays"]["probability"].argmax(1) == truth).astype(np.float64)
        global_correct = (loaded["h128_g"]["arrays"]["probability"].argmax(1) == truth).astype(np.float64)
        primary_bootstrap = paired_block_bootstrap(
            primary_correct,
            h32_correct,
            heads,
            blocks,
            selected_heads=selected_indices,
            replicates=int(evaluation["bootstrap_replicates"]),
            seed=int(evaluation["bootstrap_seed"]),
        )
        attribution_bootstrap = paired_block_bootstrap(
            primary_correct,
            global_correct,
            heads,
            blocks,
            selected_heads=selected_indices,
            replicates=int(evaluation["bootstrap_replicates"]),
            seed=int(evaluation["bootstrap_seed"]) + 1,
        )
        primary_gate = _gate(
            config=config,
            manifest=manifest,
            candidate=metrics["h128_q"],
            control=metrics["h32_q"],
            bootstrap=primary_bootstrap,
            estimable=selected_names,
            gate_name="seed_zero_primary",
        )
        attribution_gate = _gate(
            config=config,
            manifest=manifest,
            candidate=metrics["h128_q"],
            control=metrics["h128_g"],
            bootstrap=attribution_bootstrap,
            estimable=selected_names,
            gate_name="regional_attribution",
        )
        primary_gate["paired_block_bootstrap"] = primary_bootstrap
        primary_gate["per_head_descriptive_bootstrap"] = per_head_delta_bootstrap(
            primary_correct,
            h32_correct,
            heads,
            blocks,
            replicates=int(evaluation["bootstrap_replicates"]),
            seed=int(evaluation["bootstrap_seed"]),
        )
        attribution_gate["paired_block_bootstrap"] = attribution_bootstrap
        decision = (
            "replicate_h128_q_seeds_1_and_2"
            if primary_gate["passed"]
            else "reject_study_h_primary_gate"
        )
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "smoke_complete" if smoke else "complete",
        "execution_mode": "smoke" if smoke else "production",
        "study_id": config["study_id"],
        "config_sha256": sha256_file(config_path),
        "cache_receipt_sha256": sha256_file(cache_receipt_path),
        "base_dataset_manifest_sha256": sha256_file(base_manifest_path),
        "support": {
            "validation_examples": len(truth),
            "validation_blocks": len(set(str(value) for value in blocks.tolist())),
            "estimable_heads": selected_names,
            "offset_counts": np.bincount(truth, minlength=width).tolist(),
        },
        "chance": analytic_uniform_chance(truth, width=width),
        "arms": metrics,
        "primary_comparison": primary_gate,
        "regional_attribution": attribution_gate,
        "decision": {
            "status": decision,
            "study_h_primary_passed": primary_gate["passed"],
            "regional_attribution_passed": attribution_gate["passed"],
            "study_d_requires_separate_contract_and_authorization": (
                False if smoke else not primary_gate["passed"]
            ),
        },
        "bindings": {
            arm: {
                "run_receipt_sha256": sha256_file(loaded[arm]["receipt_path"]),
                "prediction_sidecar_sha256": loaded[arm]["receipt"]["prediction_sidecar_sha256"],
                "checkpoint_sha256": loaded[arm]["receipt"]["checkpoint_sha256"],
            }
            for arm in ARMS
        },
    }


def publish_score(
    *,
    report: Mapping[str, Any],
    out: Path,
    marker: Path,
    runs: Mapping[str, Path],
    config_path: Path,
    cache_receipt_path: Path,
    base_manifest_path: Path,
) -> None:
    for path in (out, marker):
        if os.path.lexists(path):
            raise ValueError(f"refusing to overwrite Study-H publication: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    report_tmp = out.with_name(f".{out.name}.tmp")
    report_tmp.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if _json(report_tmp) != report:
        raise ValueError("serialized Study-H report changed")
    report_tmp.replace(out)
    marker_base = {
        "schema_version": MARKER_SCHEMA,
        "status": report.get("status", "complete"),
        "execution_mode": report.get("execution_mode", "production"),
        "study_id": report["study_id"],
        "decision": report["decision"]["status"],
        "report": {"path": str(out.resolve()), "sha256": sha256_file(out)},
        "config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
        "cache_receipt": {"path": str(cache_receipt_path.resolve()), "sha256": sha256_file(cache_receipt_path)},
        "base_dataset_manifest": {"path": str(base_manifest_path.resolve()), "sha256": sha256_file(base_manifest_path)},
        "runs": {
            arm: {
                "path": str(runs[arm].resolve()),
                "run_receipt_sha256": sha256_file(runs[arm] / "run_receipt.json"),
                "checkpoint_sha256": sha256_file(runs[arm] / "model.pt"),
                "prediction_sidecar_sha256": sha256_file(runs[arm] / "predictions.npz"),
            }
            for arm in ARMS
        },
    }
    marker_value = dict(marker_base)
    marker_value["content_sha256"] = _canonical_sha256(marker_base)
    marker_tmp = marker.with_name(f".{marker.name}.tmp")
    marker_tmp.write_text(
        json.dumps(marker_value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if _json(marker_tmp) != marker_value:
        raise ValueError("serialized Study-H marker changed")
    marker_tmp.replace(marker)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm.replace('_', '-')}-run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-receipt", required=True, type=Path)
    parser.add_argument("--base-dataset-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    runs = {
        "h32_q": args.h32_q_run,
        "h128_g": args.h128_g_run,
        "h128_q": args.h128_q_run,
    }
    report = score_highres(
        runs=runs,
        config_path=args.config,
        cache_receipt_path=args.cache_receipt,
        base_manifest_path=args.base_dataset_manifest,
        smoke=args.smoke,
    )
    publish_score(
        report=report,
        out=args.out,
        marker=args.marker,
        runs=runs,
        config_path=args.config,
        cache_receipt_path=args.cache_receipt,
        base_manifest_path=args.base_dataset_manifest,
    )
    print(json.dumps({"status": report["status"], "decision": report["decision"]["status"]}))


if __name__ == "__main__":
    main()
