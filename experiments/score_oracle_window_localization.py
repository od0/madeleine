"""Fixed-policy scoring for the oracle-window localization experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.oracle_window_localization import HEAD_NAMES, sha256_file


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_probabilities(name: str, value: np.ndarray, width: int) -> None:
    if value.dtype != np.float32 or value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must be float32 [N,{width}]")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite probabilities")
    if np.any(value < 0) or not np.allclose(value.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{name} is not a normalized distribution")


def load_prediction_sidecar(path: Path, *, width: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "session_id",
            "run_index",
            "array_index",
            "engine_frame_idx",
            "head_index",
            "key_index",
            "event_type_index",
            "true_offset",
            "crop_start",
            "block_id",
            "conditional_prob",
            "dense_prob",
            "current_dense_prob",
            "current_dense_support",
        }
        if set(archive.files) != required:
            raise ValueError(
                "prediction sidecar inventory changed: "
                f"missing={sorted(required - set(archive.files))} "
                f"extra={sorted(set(archive.files) - required)}"
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    count = len(arrays["true_offset"])
    for name, value in arrays.items():
        if len(value) != count:
            raise ValueError(f"sidecar array length changed: {name}")
    true = arrays["true_offset"]
    heads = arrays["head_index"]
    if not np.issubdtype(true.dtype, np.integer) or np.any((true < 0) | (true >= width)):
        raise ValueError("true offsets are outside the candidate region")
    if not np.issubdtype(heads.dtype, np.integer) or np.any(
        (heads < 0) | (heads >= len(HEAD_NAMES))
    ):
        raise ValueError("head indices are outside the frozen task order")
    if not np.array_equal(arrays["key_index"], heads % 7):
        raise ValueError("key indices do not match requested heads")
    if not np.array_equal(arrays["event_type_index"], heads // 7):
        raise ValueError("event-type indices do not match requested heads")
    _require_probabilities("conditional_prob", arrays["conditional_prob"], width)
    _require_probabilities("dense_prob", arrays["dense_prob"], width)
    support = arrays["current_dense_support"]
    current = arrays["current_dense_prob"]
    if support.dtype != np.bool_ or support.shape != (count,):
        raise ValueError("current_dense_support must be bool [N]")
    if current.dtype != np.float32 or current.shape != (count, width):
        raise ValueError(f"current_dense_prob must be float32 [N,{width}]")
    if np.any(support):
        _require_probabilities("current_dense_prob[support]", current[support], width)
    if np.any(~support) and not np.all(np.isnan(current[~support])):
        raise ValueError("unsupported current-dense rows must be all-NaN")
    return arrays


def analytic_uniform_chance(true_offset: np.ndarray, *, width: int) -> dict[str, float]:
    true = np.asarray(true_offset, dtype=np.int64)
    if true.ndim != 1 or not len(true) or np.any((true < 0) | (true >= width)):
        raise ValueError("chance calculation requires valid one-dimensional offsets")
    candidates = np.arange(width)[None]
    distance = np.abs(candidates - true[:, None])
    return {
        "exact": 1.0 / width,
        "within_1": float((distance <= 1).sum(axis=1).mean() / width),
        "within_2": float((distance <= 2).sum(axis=1).mean() / width),
        "nll": math.log(width),
        "entropy": math.log(width),
    }


def _entropy_bins(entropy: np.ndarray, correct: np.ndarray) -> list[dict[str, float | int]]:
    order = np.argsort(entropy, kind="stable")
    result: list[dict[str, float | int]] = []
    for bin_index, indices in enumerate(np.array_split(order, 4)):
        if not len(indices):
            continue
        result.append(
            {
                "bin": bin_index,
                "support": int(len(indices)),
                "mean_entropy": float(entropy[indices].mean()),
                "exact_accuracy": float(correct[indices].mean()),
            }
        )
    return result


def summarize_probabilities(
    probability: np.ndarray, true_offset: np.ndarray, *, width: int
) -> dict[str, Any]:
    _require_probabilities("probability", np.asarray(probability), width)
    truth = np.asarray(true_offset, dtype=np.int64)
    if truth.shape != (len(probability),):
        raise ValueError("truth does not align with probabilities")
    prediction = probability.argmax(axis=1)
    signed = prediction - truth
    distance = np.abs(signed)
    correct = distance == 0
    selected = np.clip(probability[np.arange(len(truth)), truth], 1e-12, 1.0)
    entropy = -(probability * np.log(np.clip(probability, 1e-12, 1.0))).sum(axis=1)
    confusion = np.zeros((width, width), dtype=np.int64)
    np.add.at(confusion, (truth, prediction), 1)
    return {
        "support": int(len(truth)),
        "exact": float(correct.mean()),
        "within_1": float((distance <= 1).mean()),
        "within_2": float((distance <= 2).mean()),
        "nll": float(-np.log(selected).mean()),
        "entropy": float(entropy.mean()),
        "normalized_entropy": float(entropy.mean() / math.log(width)),
        "mean_signed_error": float(signed.mean()),
        "mean_absolute_error": float(distance.mean()),
        "early_rate": float((signed < 0).mean()),
        "late_rate": float((signed > 0).mean()),
        "mean_entropy_when_exact": (
            float(entropy[correct].mean()) if np.any(correct) else None
        ),
        "mean_entropy_when_inexact": (
            float(entropy[~correct].mean()) if np.any(~correct) else None
        ),
        "entropy_quartiles": _entropy_bins(entropy, correct),
        "confusion_true_rows_predicted_columns": confusion.tolist(),
    }


def arm_metrics(
    probability: np.ndarray,
    truth: np.ndarray,
    heads: np.ndarray,
    *,
    width: int,
) -> dict[str, Any]:
    pooled = summarize_probabilities(probability, truth, width=width)
    per_head: dict[str, Any] = {}
    for head_index, name in enumerate(HEAD_NAMES):
        selected = heads == head_index
        if not np.any(selected):
            per_head[name] = {"support": 0}
        else:
            per_head[name] = summarize_probabilities(
                probability[selected], truth[selected], width=width
            )
    scalar_names = (
        "exact",
        "within_1",
        "within_2",
        "nll",
        "entropy",
        "normalized_entropy",
        "mean_signed_error",
        "mean_absolute_error",
        "early_rate",
        "late_rate",
    )
    present = [row for row in per_head.values() if row.get("support", 0)]
    macro = {
        name: float(np.mean([float(row[name]) for row in present]))
        for name in scalar_names
    }
    macro["head_count"] = len(present)
    return {"pooled": pooled, "macro_all_heads": macro, "per_head": per_head}


def macro_for_heads(
    metrics: Mapping[str, Any], head_names: Sequence[str]
) -> dict[str, float]:
    names = ("exact", "within_1", "within_2", "nll", "entropy")
    rows = [metrics["per_head"][name] for name in head_names]
    if not rows or any(not row.get("support", 0) for row in rows):
        raise ValueError("macro task list contains an unsupported head")
    return {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in names
    }


def estimable_heads(
    manifest: Mapping[str, Any], gate: Mapping[str, Any]
) -> list[str]:
    train = manifest.get("train_counts_by_session_head")
    val = manifest.get("val_counts_by_head")
    if not isinstance(train, Mapping) or not isinstance(val, Mapping):
        raise ValueError("dataset manifest lacks per-head support counts")
    minimum_val = int(gate["minimum_validation_events_per_head"])
    minimum_train = int(gate["minimum_training_events_per_head_per_session"])
    minimum_sessions = int(gate["minimum_training_sessions"])
    result = []
    for name in HEAD_NAMES:
        session_count = sum(
            int(counts[name]) >= minimum_train for counts in train.values()
        )
        if int(val[name]) >= minimum_val and session_count >= minimum_sessions:
            result.append(name)
    return result


def _bootstrap_block_table(
    value: np.ndarray,
    heads: np.ndarray,
    blocks: np.ndarray,
    selected_heads: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    unique_blocks = sorted(set(str(value) for value in blocks.tolist()))
    block_lookup = {name: index for index, name in enumerate(unique_blocks)}
    head_lookup = {head: index for index, head in enumerate(selected_heads)}
    sums = np.zeros((len(unique_blocks), len(selected_heads)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for index in range(len(value)):
        head = int(heads[index])
        if head not in head_lookup:
            continue
        row = block_lookup[str(blocks[index])]
        column = head_lookup[head]
        sums[row, column] += float(value[index])
        counts[row, column] += 1.0
    return sums, counts, unique_blocks


def paired_block_bootstrap(
    conditional_value: np.ndarray,
    dense_value: np.ndarray,
    heads: np.ndarray,
    blocks: np.ndarray,
    *,
    selected_heads: Sequence[int],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Paired resampling of whole continuity-bounded temporal blocks."""

    left = np.asarray(conditional_value, dtype=np.float64)
    right = np.asarray(dense_value, dtype=np.float64)
    if left.shape != right.shape or left.shape != heads.shape or left.shape != blocks.shape:
        raise ValueError("bootstrap arrays are not aligned")
    if replicates < 100:
        raise ValueError("bootstrap requires at least 100 replicates")
    left_sum, counts, block_names = _bootstrap_block_table(
        left, heads, blocks, selected_heads
    )
    right_sum, right_counts, right_blocks = _bootstrap_block_table(
        right, heads, blocks, selected_heads
    )
    if block_names != right_blocks or not np.array_equal(counts, right_counts):
        raise AssertionError("paired block supports differ")
    block_count = len(block_names)
    if block_count < 1:
        raise ValueError("bootstrap has no blocks")
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(
        block_count,
        np.full(block_count, 1.0 / block_count),
        size=replicates,
    )
    denominator = draws @ counts
    with np.errstate(divide="ignore", invalid="ignore"):
        left_head = (draws @ left_sum) / denominator
        right_head = (draws @ right_sum) / denominator
    valid = np.all(denominator > 0, axis=1)
    if int(valid.sum()) < int(0.95 * replicates):
        raise ValueError("too many bootstrap replicates omit an estimable head")
    left_macro = np.nanmean(left_head[valid], axis=1)
    right_macro = np.nanmean(right_head[valid], axis=1)
    delta = left_macro - right_macro

    def interval(value: np.ndarray) -> list[float]:
        return [float(x) for x in np.quantile(value, [0.025, 0.5, 0.975])]

    return {
        "replicates_requested": replicates,
        "replicates_valid": int(valid.sum()),
        "block_count": block_count,
        "conditional_macro_95": interval(left_macro),
        "dense_macro_95": interval(right_macro),
        "delta_macro_95": interval(delta),
    }


def per_head_delta_bootstrap(
    conditional_correct: np.ndarray,
    dense_correct: np.ndarray,
    heads: np.ndarray,
    blocks: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for head_index, name in enumerate(HEAD_NAMES):
        selected = heads == head_index
        if not np.any(selected):
            result[name] = {"support": 0}
            continue
        unique_blocks = sorted(set(str(value) for value in blocks[selected].tolist()))
        block_index = {value: index for index, value in enumerate(unique_blocks)}
        left_sum = np.zeros(len(unique_blocks), dtype=np.float64)
        right_sum = np.zeros_like(left_sum)
        counts = np.zeros_like(left_sum)
        for left, right, block in zip(
            conditional_correct[selected],
            dense_correct[selected],
            blocks[selected],
            strict=True,
        ):
            index = block_index[str(block)]
            left_sum[index] += float(left)
            right_sum[index] += float(right)
            counts[index] += 1
        rng = np.random.default_rng(seed + head_index)
        draws = rng.multinomial(
            len(unique_blocks),
            np.full(len(unique_blocks), 1.0 / len(unique_blocks)),
            size=replicates,
        )
        denominator = draws @ counts
        delta = (draws @ (left_sum - right_sum)) / denominator
        result[name] = {
            "support": int(selected.sum()),
            "block_count": len(unique_blocks),
            "exact_delta_95": [
                float(value) for value in np.quantile(delta, [0.025, 0.5, 0.975])
            ],
        }
    return result


def apply_seed_zero_gate(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    conditional_metrics: Mapping[str, Any],
    dense_metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    estimable: Sequence[str],
) -> dict[str, Any]:
    support_gate = config["decision_gate"]["estimability"]
    decision = config["decision_gate"]["seed_zero"]
    distinct_keys = {name.split(":", 1)[0] for name in estimable}
    event_types = {name.split(":", 1)[1] for name in estimable}
    block_count = int(manifest["validation_block_count"])
    capable_checks = {
        "minimum_estimable_heads": len(estimable)
        >= int(support_gate["minimum_estimable_heads"]),
        "minimum_distinct_keys": len(distinct_keys)
        >= int(support_gate["minimum_distinct_keys"]),
        "both_event_types": (
            event_types == {"onset", "release"}
            if bool(support_gate["require_both_event_types"])
            else True
        ),
        "minimum_validation_blocks": block_count
        >= int(support_gate["minimum_validation_blocks"]),
    }
    capable = all(capable_checks.values())
    conditional_macro = macro_for_heads(conditional_metrics, estimable)
    dense_macro = macro_for_heads(dense_metrics, estimable)
    positive = [
        name
        for name in estimable
        if float(conditional_metrics["per_head"][name]["exact"])
        > float(dense_metrics["per_head"][name]["exact"])
    ]
    positive_keys = {name.split(":", 1)[0] for name in positive}
    positive_types = {name.split(":", 1)[1] for name in positive}
    exact_delta = conditional_macro["exact"] - dense_macro["exact"]
    within_2_delta = conditional_macro["within_2"] - dense_macro["within_2"]
    conditional_ci_low = float(bootstrap["conditional_macro_95"][0])
    delta_ci_low = float(bootstrap["delta_macro_95"][0])
    checks = {
        "minimum_conditional_macro_exact": conditional_macro["exact"]
        >= float(decision["minimum_conditional_macro_exact"]),
        "conditional_exact_ci_low_above_chance": conditional_ci_low
        > float(decision["minimum_conditional_exact_ci_low"]),
        "minimum_macro_exact_delta": exact_delta
        >= float(decision["minimum_macro_exact_delta"]),
        "macro_exact_delta_ci_low_above_zero": delta_ci_low
        > float(decision["minimum_macro_exact_delta_ci_low"]),
        "minimum_positive_estimable_heads": len(positive)
        >= int(decision["minimum_positive_estimable_heads"]),
        "minimum_positive_distinct_keys": len(positive_keys)
        >= int(decision["minimum_positive_distinct_keys"]),
        "positive_both_event_types": (
            positive_types == {"onset", "release"}
            if bool(decision["require_positive_both_event_types"])
            else True
        ),
        "within_2_noninferiority": within_2_delta
        >= float(decision["minimum_macro_within_2_delta"]),
        "lower_macro_nll": (
            conditional_macro["nll"] < dense_macro["nll"]
            if bool(decision["require_lower_macro_nll"])
            else True
        ),
    }
    passed = capable and all(checks.values())
    return {
        "gate_capable": capable,
        "support_checks": capable_checks,
        "estimable_heads": list(estimable),
        "estimable_head_count": len(estimable),
        "estimable_distinct_keys": sorted(distinct_keys),
        "validation_block_count": block_count,
        "conditional_estimable_macro": conditional_macro,
        "dense_estimable_macro": dense_macro,
        "macro_exact_delta": exact_delta,
        "macro_within_2_delta": within_2_delta,
        "positive_estimable_heads": positive,
        "positive_distinct_keys": sorted(positive_keys),
        "checks": checks,
        "passed": passed,
        "decision": (
            "replicate_unchanged_seeds_1_and_2"
            if passed
            else "reject_phase_2_at_seed_zero_gate"
        ),
    }


def score_experiment(
    *,
    predictions_path: Path,
    dataset_manifest_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    config = _json(config_path)
    manifest = _json(dataset_manifest_path)
    width = int(config["dataset"]["candidate_width"])
    if width != 16:
        raise ValueError("the frozen candidate width changed")
    arrays = load_prediction_sidecar(predictions_path, width=width)
    truth = arrays["true_offset"].astype(np.int64)
    heads = arrays["head_index"].astype(np.int64)
    blocks = arrays["block_id"]
    conditional = arm_metrics(arrays["conditional_prob"], truth, heads, width=width)
    dense = arm_metrics(arrays["dense_prob"], truth, heads, width=width)
    support_gate = config["decision_gate"]["estimability"]
    selected_names = estimable_heads(manifest, support_gate)
    selected_indices = [HEAD_NAMES.index(name) for name in selected_names]
    conditional_correct = (
        arrays["conditional_prob"].argmax(axis=1) == truth
    ).astype(np.float64)
    dense_correct = (arrays["dense_prob"].argmax(axis=1) == truth).astype(np.float64)
    evaluation = config["evaluation"]
    bootstrap = paired_block_bootstrap(
        conditional_correct,
        dense_correct,
        heads,
        blocks,
        selected_heads=selected_indices,
        replicates=int(evaluation["bootstrap_replicates"]),
        seed=int(evaluation["bootstrap_seed"]),
    )
    per_head_ci = per_head_delta_bootstrap(
        conditional_correct,
        dense_correct,
        heads,
        blocks,
        replicates=int(evaluation["bootstrap_replicates"]),
        seed=int(evaluation["bootstrap_seed"]),
    )
    gate = apply_seed_zero_gate(
        config=config,
        manifest=manifest,
        conditional_metrics=conditional,
        dense_metrics=dense,
        bootstrap=bootstrap,
        estimable=selected_names,
    )
    current_support = arrays["current_dense_support"]
    current: dict[str, Any] = {
        "role": "unmatched historical reference; excluded from gate",
        "support": int(current_support.sum()),
    }
    if np.any(current_support):
        current["metrics"] = arm_metrics(
            arrays["current_dense_prob"][current_support],
            truth[current_support],
            heads[current_support],
            width=width,
        )
        current["matched_arms_on_same_support"] = {
            "conditional": arm_metrics(
                arrays["conditional_prob"][current_support],
                truth[current_support],
                heads[current_support],
                width=width,
            ),
            "dense": arm_metrics(
                arrays["dense_prob"][current_support],
                truth[current_support],
                heads[current_support],
                width=width,
            ),
        }
    return {
        "schema_version": "madeleine.oracle-window-score.v1",
        "status": "complete",
        "study_id": config["study_id"],
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "prediction_sidecar_sha256": sha256_file(predictions_path),
        "support": {
            "validation_examples": len(truth),
            "head_names": list(HEAD_NAMES),
            "offset_counts": np.bincount(truth, minlength=width).tolist(),
            "block_count": len(set(str(value) for value in blocks.tolist())),
        },
        "chance": analytic_uniform_chance(truth, width=width),
        "arms": {"conditional_softmax": conditional, "matched_dense_bce": dense},
        "primary_comparison": {
            "estimable_heads": selected_names,
            "paired_block_bootstrap": bootstrap,
            "per_head_descriptive_bootstrap": per_head_ci,
        },
        "current_dense_reference": current,
        "decision_gate": gate,
    }


def publish_score(
    *,
    report: Mapping[str, Any],
    out: Path,
    marker: Path,
    predictions_path: Path,
    dataset_manifest_path: Path,
    config_path: Path,
) -> None:
    for path in (out, marker):
        if os.path.lexists(path):
            raise ValueError(f"refusing to overwrite published artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp")
    if os.path.lexists(temporary):
        raise ValueError(f"stale report temporary exists: {temporary}")
    temporary.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reloaded = _json(temporary)
    if reloaded != report:
        raise ValueError("serialized score report changed on reload")
    temporary.replace(out)
    marker_content = {
        "schema_version": "madeleine.oracle-window-complete.v1",
        "status": "complete",
        "study_id": report["study_id"],
        "report": {"path": str(out), "sha256": sha256_file(out)},
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
        },
        "dataset_manifest": {
            "path": str(dataset_manifest_path),
            "sha256": sha256_file(dataset_manifest_path),
        },
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "decision": report["decision_gate"]["decision"],
    }
    canonical = json.dumps(
        marker_content, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    marker_content["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    marker_tmp = marker.with_name(f".{marker.name}.tmp")
    if os.path.lexists(marker_tmp):
        raise ValueError(f"stale marker temporary exists: {marker_tmp}")
    marker_tmp.write_text(
        json.dumps(marker_content, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if _json(marker_tmp) != marker_content:
        raise ValueError("serialized completion marker changed on reload")
    marker_tmp.replace(marker)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = score_experiment(
        predictions_path=args.predictions,
        dataset_manifest_path=args.dataset_manifest,
        config_path=args.config,
    )
    publish_score(
        report=report,
        out=args.out,
        marker=args.marker,
        predictions_path=args.predictions,
        dataset_manifest_path=args.dataset_manifest,
        config_path=args.config,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision_gate"]["decision"],
                "report": str(args.out),
                "marker": str(args.marker),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
