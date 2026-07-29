"""Fixed-policy scoring and publication for the bounded differential follow-up."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.oracle_window_differential_followup import (
    METADATA_NAMES,
    DifferentialCandidateModel,
    PixelOracleDataset,
    load_pixel_split,
    predict_probabilities,
)
from experiments.oracle_window_localization import (
    HEAD_NAMES,
    _validate_implementation,
    sha256_file,
    state_dict_sha256,
)
from experiments.score_oracle_window_localization import (
    apply_seed_zero_gate,
    arm_metrics,
    estimable_heads,
    paired_block_bootstrap,
    per_head_delta_bootstrap,
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_probability(name: str, value: np.ndarray) -> None:
    if value.dtype != np.float32 or value.ndim != 2 or value.shape[1] != 16:
        raise ValueError(f"{name} must be float32 [N,16]")
    if not np.all(np.isfinite(value)) or np.any(value < 0):
        raise ValueError(f"{name} contains invalid probabilities")
    if not np.allclose(value.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{name} is not normalized")


def load_differential_sidecar(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            *METADATA_NAMES,
            "ordered_pair_prob",
            "symmetric_pair_prob",
            "feature_conditional_prob",
        }
        if set(archive.files) != required:
            raise ValueError("differential sidecar inventory changed")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    count = len(arrays["true_offset"])
    for name, value in arrays.items():
        if len(value) != count:
            raise ValueError(f"differential sidecar length changed: {name}")
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
    for name, dtype in expected_dtypes.items():
        if arrays[name].dtype != dtype:
            raise ValueError(f"differential sidecar dtype changed: {name}")
    heads = arrays["head_index"]
    truth = arrays["true_offset"]
    if np.any((heads < 0) | (heads >= len(HEAD_NAMES))):
        raise ValueError("differential head index outside frozen order")
    if np.any((truth < 0) | (truth >= 16)):
        raise ValueError("differential truth outside candidate window")
    if not np.array_equal(arrays["key_index"], heads % 7):
        raise ValueError("differential key identity changed")
    if not np.array_equal(arrays["event_type_index"], heads // 7):
        raise ValueError("differential event identity changed")
    _require_probability("ordered_pair_prob", arrays["ordered_pair_prob"])
    _require_probability("symmetric_pair_prob", arrays["symmetric_pair_prob"])
    _require_probability("feature_conditional_prob", arrays["feature_conditional_prob"])
    return arrays


def extended_uniform_chance(true_offset: np.ndarray) -> dict[str, float]:
    truth = np.asarray(true_offset, dtype=np.int64)
    if truth.ndim != 1 or not len(truth) or np.any((truth < 0) | (truth >= 16)):
        raise ValueError("uniform chance requires valid offsets")
    candidate = np.arange(16)[None]
    signed = candidate - truth[:, None]
    distance = np.abs(signed)
    return {
        "exact": 1.0 / 16.0,
        "within_1": float((distance <= 1).sum(axis=1).mean() / 16.0),
        "within_2": float((distance <= 2).sum(axis=1).mean() / 16.0),
        "nll": math.log(16.0),
        "entropy": math.log(16.0),
        "normalized_entropy": 1.0,
        "mean_signed_error": float(signed.mean(axis=1).mean()),
        "mean_absolute_error": float(distance.mean(axis=1).mean()),
        "early_rate": float((signed < 0).sum(axis=1).mean() / 16.0),
        "late_rate": float((signed > 0).sum(axis=1).mean() / 16.0),
    }


def _validate_state_dict_exact(
    observed: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor]
) -> None:
    if set(observed) != set(expected):
        raise ValueError("differential checkpoint state inventory changed")
    for name, reference in expected.items():
        value = observed[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint state is not a tensor: {name}")
        if value.dtype != reference.dtype or value.shape != reference.shape:
            raise ValueError(f"checkpoint tensor schema changed: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"checkpoint tensor is non-finite: {name}")


def validate_run_binding(
    *,
    predictions_path: Path,
    run_receipt_path: Path,
    ordered_checkpoint_path: Path,
    symmetric_pair_checkpoint_path: Path,
    training_log_path: Path,
    cache_root: Path,
    cache_receipt_path: Path,
    baseline_sidecar_path: Path,
    baseline_audit_path: Path,
    config_path: Path,
    device_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    config = _json(config_path)
    if config.get("status") != "preregistered_before_validation_inference":
        raise ValueError("differential follow-up config is not frozen")
    implementation = _validate_implementation(
        config, repo=config_path.resolve().parents[2]
    )
    run = _json(run_receipt_path)
    cache_receipt = _json(cache_receipt_path)
    arrays = load_differential_sidecar(predictions_path)
    expected = config["dataset"]["expected_support"]
    validation_cache = load_pixel_split(
        cache_root,
        cache_receipt,
        split="validation",
        expected_examples=int(expected["validation_examples"]),
    )
    for name in METADATA_NAMES:
        if not np.array_equal(arrays[name], validation_cache[name]):
            raise ValueError(f"sidecar differs from validated pixel cache: {name}")

    required_run = {
        "schema_version": "madeleine.oracle-window-differential-run.v1",
        "status": "predictions_complete_unscored",
        "config_sha256": sha256_file(config_path),
        "pixel_cache_receipt_sha256": sha256_file(cache_receipt_path),
        "baseline_sidecar_sha256": sha256_file(baseline_sidecar_path),
        "baseline_audit_sha256": sha256_file(baseline_audit_path),
        "training_log_sha256": sha256_file(training_log_path),
        "prediction_sidecar_sha256": sha256_file(predictions_path),
        "seed": int(config["training"]["seed"]),
        "epochs": int(config["training"]["epochs"]),
        "configured_epochs": int(config["training"]["epochs"]),
        "final_weights_only": True,
        "validation_used_for_training_or_selection": False,
        "matched_initialization": True,
        "matched_batch_order": True,
        "train_examples": int(expected["training_examples"]),
        "validation_examples": int(expected["validation_examples"]),
        "implementation": implementation,
    }
    for name, expected_value in required_run.items():
        if run.get(name) != expected_value:
            raise ValueError(f"differential run receipt changed: {name}")
    for name, expected_path in {
        "config_path": config_path,
        "pixel_cache_receipt_path": cache_receipt_path,
        "baseline_sidecar_path": baseline_sidecar_path,
        "baseline_audit_path": baseline_audit_path,
    }.items():
        if Path(str(run.get(name))).resolve() != expected_path.resolve():
            raise ValueError(f"differential run path changed: {name}")
    log = _json(training_log_path)
    if (
        log.get("fixed_final_epoch") != int(config["training"]["epochs"])
        or log.get("configured_final_epoch") != int(config["training"]["epochs"])
        or log.get("validation_used_for_training_or_selection") is not False
        or log.get("matched_batch_order") is not True
        or not isinstance(log.get("ordered_pair"), list)
        or not isinstance(log.get("symmetric_pair"), list)
        or len(log["ordered_pair"]) != int(config["training"]["epochs"])
        or len(log["symmetric_pair"]) != int(config["training"]["epochs"])
    ):
        raise ValueError("differential training endpoint evidence changed")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS validation requested but unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requested but unavailable")
    device = torch.device(device_name)
    checkpoint_paths = {
        "ordered_pair": ordered_checkpoint_path,
        "symmetric_pair": symmetric_pair_checkpoint_path,
    }
    probability_names = {
        "ordered_pair": "ordered_pair_prob",
        "symmetric_pair": "symmetric_pair_prob",
    }
    run_checkpoints = run.get("checkpoints")
    if not isinstance(run_checkpoints, Mapping) or set(run_checkpoints) != set(checkpoint_paths):
        raise ValueError("differential run checkpoint inventory changed")
    checkpoint_bindings: dict[str, Any] = {}
    for arm, checkpoint_path in checkpoint_paths.items():
        run_checkpoint = run_checkpoints[arm]
        if not isinstance(run_checkpoint, Mapping):
            raise ValueError(f"differential run checkpoint receipt changed: {arm}")
        if Path(str(run_checkpoint.get("path"))).resolve() != checkpoint_path.resolve():
            raise ValueError(f"differential checkpoint path changed: {arm}")
        checkpoint_sha = sha256_file(checkpoint_path)
        if run_checkpoint.get("sha256") != checkpoint_sha:
            raise ValueError(f"differential checkpoint hash changed: {arm}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for name, expected_value in {
            "schema_version": "madeleine.oracle-window-differential-checkpoint.v1",
            "config_sha256": sha256_file(config_path),
            "pixel_cache_receipt_sha256": sha256_file(cache_receipt_path),
            "baseline_sidecar_sha256": sha256_file(baseline_sidecar_path),
            "seed": int(config["training"]["seed"]),
            "epochs": int(config["training"]["epochs"]),
            "initial_state_sha256": run["initial_state_sha256"],
            "arm": arm,
        }.items():
            if checkpoint.get(name) != expected_value:
                raise ValueError(f"differential checkpoint receipt changed: {arm}:{name}")
        model = DifferentialCandidateModel(
            embedding_dim=int(config["model"]["embedding_dim"])
        )
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise ValueError(f"differential checkpoint lacks a model state: {arm}")
        _validate_state_dict_exact(state, model.state_dict())
        model.load_state_dict(state, strict=True)
        model_state_sha = state_dict_sha256(model)
        if model_state_sha != run_checkpoint.get("model_state_sha256"):
            raise ValueError(f"differential checkpoint state hash changed: {arm}")
        model.to(device)
        reproduced = predict_probabilities(
            model,
            PixelOracleDataset(validation_cache),
            device=device,
            batch_size=int(config["training"]["eval_batch_size"]),
            arm=arm,
        )
        if not np.array_equal(reproduced, arrays[probability_names[arm]]):
            raise ValueError(f"saved model did not exactly reproduce predictions: {arm}")
        checkpoint_bindings[arm] = {
            "sha256": checkpoint_sha,
            "state_dict_sha256": model_state_sha,
            "predictions_reproduced_exactly": True,
        }

    frozen_baseline = config["frozen_feature_baseline"]
    if sha256_file(baseline_sidecar_path) != str(
        frozen_baseline["prediction_sidecar_sha256"]
    ):
        raise ValueError("frozen feature sidecar hash differs from the config")
    if sha256_file(baseline_audit_path) != str(
        frozen_baseline["audit_receipt_sha256"]
    ):
        raise ValueError("frozen feature audit hash differs from the config")
    baseline_audit = _json(baseline_audit_path)
    if baseline_audit.get("status") != "complete" or baseline_audit.get(
        "content_sha256"
    ) != config["frozen_feature_baseline"]["audit_content_sha256"]:
        raise ValueError("feature-baseline audit content changed")
    from experiments.score_oracle_window_localization import load_prediction_sidecar

    baseline = load_prediction_sidecar(baseline_sidecar_path, width=16)
    for name in METADATA_NAMES:
        if not np.array_equal(baseline[name], arrays[name]):
            raise ValueError(f"feature baseline identity differs: {name}")
    if not np.array_equal(baseline["conditional_prob"], arrays["feature_conditional_prob"]):
        raise ValueError("feature-baseline probabilities changed")
    return arrays, {
        "run_receipt_sha256": sha256_file(run_receipt_path),
        "checkpoints": checkpoint_bindings,
        "training_log_sha256": sha256_file(training_log_path),
        "model_predictions_reproduced_exactly": True,
        "pixel_cache_identity_exact": True,
        "feature_baseline_identity_exact": True,
    }


def score_followup(
    *,
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    base_manifest: Mapping[str, Any],
    bindings: Mapping[str, Any],
    config_sha256: str,
    predictions_sha256: str,
    cache_receipt_sha256: str,
    base_manifest_sha256: str,
) -> dict[str, Any]:
    truth = arrays["true_offset"].astype(np.int64)
    heads = arrays["head_index"].astype(np.int64)
    blocks = arrays["block_id"]
    ordered = arm_metrics(arrays["ordered_pair_prob"], truth, heads, width=16)
    symmetric_pair = arm_metrics(arrays["symmetric_pair_prob"], truth, heads, width=16)
    feature = arm_metrics(
        arrays["feature_conditional_prob"], truth, heads, width=16
    )
    estimable = estimable_heads(
        base_manifest, config["decision_gate"]["estimability"]
    )
    selected_indices = [HEAD_NAMES.index(name) for name in estimable]
    ordered_correct = (
        arrays["ordered_pair_prob"].argmax(axis=1) == truth
    ).astype(np.float64)
    symmetric_pair_correct = (
        arrays["symmetric_pair_prob"].argmax(axis=1) == truth
    ).astype(np.float64)
    feature_correct = (
        arrays["feature_conditional_prob"].argmax(axis=1) == truth
    ).astype(np.float64)
    evaluation = config["evaluation"]
    bootstrap_seed = int(evaluation["bootstrap_seed"])

    def comparison_bootstrap(
        left: np.ndarray, right: np.ndarray, *, seed_offset: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        bootstrap = paired_block_bootstrap(
            left,
            right,
            heads,
            blocks,
            selected_heads=selected_indices,
            replicates=int(evaluation["bootstrap_replicates"]),
            seed=bootstrap_seed + seed_offset,
        )
        if bootstrap["replicates_valid"] != bootstrap["replicates_requested"]:
            raise ValueError("bounded follow-up requires every bootstrap draw to be valid")
        per_head = per_head_delta_bootstrap(
            left,
            right,
            heads,
            blocks,
            replicates=int(evaluation["bootstrap_replicates"]),
            seed=bootstrap_seed + seed_offset,
        )
        return bootstrap, per_head

    ordered_feature_bootstrap, ordered_feature_per_head = comparison_bootstrap(
        ordered_correct, feature_correct, seed_offset=0
    )
    same_feature_bootstrap, same_feature_per_head = comparison_bootstrap(
        symmetric_pair_correct, feature_correct, seed_offset=1
    )
    ordered_same_bootstrap, ordered_same_per_head = comparison_bootstrap(
        ordered_correct, symmetric_pair_correct, seed_offset=2
    )
    ordered_feature_gate = apply_seed_zero_gate(
        config=config,
        manifest=base_manifest,
        conditional_metrics=ordered,
        dense_metrics=feature,
        bootstrap=ordered_feature_bootstrap,
        estimable=estimable,
    )
    same_feature_gate = apply_seed_zero_gate(
        config=config,
        manifest=base_manifest,
        conditional_metrics=symmetric_pair,
        dense_metrics=feature,
        bootstrap=same_feature_bootstrap,
        estimable=estimable,
    )
    attribution_config = copy.deepcopy(config)
    attribution_config["decision_gate"]["seed_zero"] = config["decision_gate"][
        "differential_attribution"
    ]
    ordered_same_gate = apply_seed_zero_gate(
        config=attribution_config,
        manifest=base_manifest,
        conditional_metrics=ordered,
        dense_metrics=symmetric_pair,
        bootstrap=ordered_same_bootstrap,
        estimable=estimable,
    )
    if ordered_feature_gate["passed"]:
        if ordered_same_gate["passed"]:
            decision = "differential_specific_pixel_rescue_original_phase_2_still_rejected"
        else:
            decision = "learned_pixel_rescue_not_differentially_attributed_original_phase_2_still_rejected"
    elif same_feature_gate["passed"]:
        decision = "appearance_only_pixel_rescue_pair_hypothesis_rejected_no_phase_2"
    else:
        decision = "no_bounded_pixel_rescue_no_phase_2"
    chance_per_head = {
        name: extended_uniform_chance(truth[heads == index])
        for index, name in enumerate(HEAD_NAMES)
        if np.any(heads == index)
    }
    return {
        "schema_version": "madeleine.oracle-window-differential-score.v1",
        "status": "complete",
        "study_id": config["study_id"],
        "scope": "one bounded representation diagnostic; cannot reopen the rejected cascade",
        "config_sha256": config_sha256,
        "prediction_sidecar_sha256": predictions_sha256,
        "pixel_cache_receipt_sha256": cache_receipt_sha256,
        "base_dataset_manifest_sha256": base_manifest_sha256,
        "bindings": dict(bindings),
        "support": {
            "validation_examples": len(truth),
            "validation_blocks": len(set(str(value) for value in blocks.tolist())),
            "offset_counts": np.bincount(truth, minlength=16).tolist(),
            "head_names": list(HEAD_NAMES),
        },
        "chance": {
            "overall": extended_uniform_chance(truth),
            "per_head": chance_per_head,
        },
        "arms": {
            "ordered_pixel_pair": ordered,
            "matched_symmetric_pair": symmetric_pair,
            "frozen_feature_conditional": feature,
        },
        "primary_comparison": {
            "estimable_heads": estimable,
            "contrast": "ordered_pixel_pair_minus_frozen_feature_conditional",
            "paired_block_bootstrap": ordered_feature_bootstrap,
            "per_head_descriptive_bootstrap": ordered_feature_per_head,
        },
        "attribution_comparisons": {
            "symmetric_pair_minus_frozen_feature": {
                "paired_block_bootstrap": same_feature_bootstrap,
                "per_head_descriptive_bootstrap": same_feature_per_head,
            },
            "ordered_pair_minus_symmetric_pair": {
                "paired_block_bootstrap": ordered_same_bootstrap,
                "per_head_descriptive_bootstrap": ordered_same_per_head,
            },
        },
        "decision_gate": {
            "ordered_pair_vs_frozen_feature": ordered_feature_gate,
            "symmetric_pair_vs_frozen_feature": same_feature_gate,
            "ordered_pair_vs_symmetric_pair": ordered_same_gate,
            "original_phase_2_decision_remains_rejected": True,
            "passed_primary_pixel_rescue": ordered_feature_gate["passed"],
            "passed_differential_attribution": (
                ordered_feature_gate["passed"] and ordered_same_gate["passed"]
            ),
            "decision": decision,
        },
    }


def publish_followup(
    *,
    report: Mapping[str, Any],
    out: Path,
    marker: Path,
    config_path: Path,
    predictions_path: Path,
    run_receipt_path: Path,
    ordered_checkpoint_path: Path,
    symmetric_pair_checkpoint_path: Path,
    cache_receipt_path: Path,
) -> None:
    paths = [
        out,
        marker,
        config_path,
        predictions_path,
        run_receipt_path,
        ordered_checkpoint_path,
        symmetric_pair_checkpoint_path,
        cache_receipt_path,
    ]
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("follow-up publication paths alias")
    for path in (out, marker):
        if os.path.lexists(path):
            raise ValueError(f"refusing to overwrite follow-up artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    if report.get("config_sha256") != sha256_file(config_path):
        raise ValueError("follow-up report is not bound to the supplied config")
    if report.get("prediction_sidecar_sha256") != sha256_file(predictions_path):
        raise ValueError("follow-up report is not bound to the supplied predictions")
    temporary = out.with_name(f".{out.name}.tmp")
    if os.path.lexists(temporary):
        raise ValueError(f"stale follow-up report temporary exists: {temporary}")
    temporary.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if _json(temporary) != report:
        raise ValueError("serialized follow-up report changed")
    temporary.replace(out)
    marker_content = {
        "schema_version": "madeleine.oracle-window-differential-complete.v1",
        "status": "complete",
        "study_id": report["study_id"],
        "decision": report["decision_gate"]["decision"],
        "report": {"path": str(out), "sha256": sha256_file(out)},
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
        },
        "run_receipt": {
            "path": str(run_receipt_path),
            "sha256": sha256_file(run_receipt_path),
        },
        "checkpoints": {
            "ordered_pair": {
                "path": str(ordered_checkpoint_path),
                "sha256": sha256_file(ordered_checkpoint_path),
            },
            "symmetric_pair": {
                "path": str(symmetric_pair_checkpoint_path),
                "sha256": sha256_file(symmetric_pair_checkpoint_path),
            },
        },
        "pixel_cache_receipt": {
            "path": str(cache_receipt_path),
            "sha256": sha256_file(cache_receipt_path),
        },
    }
    marker_content["content_sha256"] = _canonical_sha256(marker_content)
    marker_tmp = marker.with_name(f".{marker.name}.tmp")
    if os.path.lexists(marker_tmp):
        raise ValueError(f"stale follow-up marker temporary exists: {marker_tmp}")
    marker_tmp.write_text(
        json.dumps(marker_content, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if _json(marker_tmp) != marker_content:
        raise ValueError("serialized follow-up marker changed")
    marker_tmp.replace(marker)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--run-receipt", required=True, type=Path)
    parser.add_argument("--ordered-checkpoint", required=True, type=Path)
    parser.add_argument("--symmetric-pair-checkpoint", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--cache-receipt", required=True, type=Path)
    parser.add_argument("--baseline-sidecar", required=True, type=Path)
    parser.add_argument("--baseline-audit", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    arrays, bindings = validate_run_binding(
        predictions_path=args.predictions,
        run_receipt_path=args.run_receipt,
        ordered_checkpoint_path=args.ordered_checkpoint,
        symmetric_pair_checkpoint_path=args.symmetric_pair_checkpoint,
        training_log_path=args.training_log,
        cache_root=args.cache,
        cache_receipt_path=args.cache_receipt,
        baseline_sidecar_path=args.baseline_sidecar,
        baseline_audit_path=args.baseline_audit,
        config_path=args.config,
        device_name=args.device,
    )
    config = _json(args.config)
    observed_manifest_sha = sha256_file(args.base_manifest)
    if observed_manifest_sha != str(
        config["dataset"]["base_dataset_manifest_sha256"]
    ):
        raise ValueError("base oracle dataset manifest hash changed")
    report = score_followup(
        arrays=arrays,
        config=config,
        base_manifest=_json(args.base_manifest),
        bindings=bindings,
        config_sha256=sha256_file(args.config),
        predictions_sha256=sha256_file(args.predictions),
        cache_receipt_sha256=sha256_file(args.cache_receipt),
        base_manifest_sha256=observed_manifest_sha,
    )
    publish_followup(
        report=report,
        out=args.out,
        marker=args.marker,
        config_path=args.config,
        predictions_path=args.predictions,
        run_receipt_path=args.run_receipt,
        ordered_checkpoint_path=args.ordered_checkpoint,
        symmetric_pair_checkpoint_path=args.symmetric_pair_checkpoint,
        cache_receipt_path=args.cache_receipt,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "decision": report["decision_gate"]["decision"],
                "report": str(args.out),
                "marker": str(args.marker),
            }
        )
    )


if __name__ == "__main__":
    main()
