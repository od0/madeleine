"""Score the preregistered dynamics-identifiability probe artifacts.

The scorer consumes only the raw probe JSON and its aligned prediction NPZ. It
verifies both against the frozen contract, then applies paired block
uncertainty, a continuity-run-bounded circular target null, fixed-family FDR,
LOSO sign checks, and the preregistered subset/global gates. It never opens a
source image shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from data.schema import KEY_ORDER
from experiments.dynamics_offset_probe import (
    DEFAULT_OFFSETS,
    FEATURE_VARIANTS,
    OUTPUT_NAMES,
    WORST_CASE_VALIDATION_SAMPLES,
    canonical_array_sha256,
    loso_prediction_array_name,
    prediction_array_name,
    sha256_file,
    validate_prediction_sidecar,
)


PAIR_CONTROL = {
    "pooled_pair": "pooled_same_frame",
    "spatial_motion": "spatial_same_frame",
}
CANDIDATE_HORIZONS = (1, 2, 4, 8, 16)
NEGATIVE_OFFSETS = (-4, -3, -2, -1)
BOOTSTRAP_REPLICATES = 5_000
NULL_REPLICATES = 5_000
BLOCK_FRAMES = 600
MIN_NULL_SHIFT = 300
FDR_Q = 0.05
BOOTSTRAP_SEED = 2_026_072_701
CIRCULAR_NULL_SEED = 2_026_072_702
FROZEN_FDR_FAMILY_SIZE = 2 * 5 * 14
NULL_STATISTIC_CONTRACT = (
    "on retained long-run rows, the post-state-conditioned mean predicted-score "
    "contrast (event minus same-post-state non-event), after arithmetic averaging "
    "of the three seed probabilities, at candidate h minus the maximum "
    "corresponding contrast over -4..-1"
)
MAX_NULL_BENCHMARK_SECONDS = 15 * 60


def _finite_mean(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    numerator = np.where(finite, array, 0.0).sum(axis=axis)
    denominator = finite.sum(axis=axis)
    result = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _task_mask(post_state: np.ndarray, task: int) -> np.ndarray:
    key = task % len(KEY_ORDER)
    return np.asarray(post_state)[:, key] == (
        1 if task < len(KEY_ORDER) else 0
    )


def _ap_skill(truth: np.ndarray, score: np.ndarray, eligible: np.ndarray) -> float:
    mask = np.asarray(eligible, dtype=bool)
    label = np.asarray(truth, dtype=np.uint8)[mask]
    value = np.asarray(score, dtype=np.float64)[mask]
    positives = int(label.sum())
    if not len(label) or positives == 0 or positives == len(label):
        return float("nan")
    order = np.argsort(-value, kind="mergesort")
    ranked = label[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    average_precision = float(precision[ranked == 1].mean())
    prevalence = positives / len(label)
    return float((average_precision - prevalence) / (1.0 - prevalence))


def continuity_runs(
    engine_frame_idx: np.ndarray,
    session_index: np.ndarray,
    run_id: np.ndarray,
) -> list[np.ndarray]:
    engine = np.asarray(engine_frame_idx, dtype=np.int64)
    session = np.asarray(session_index)
    run = np.asarray(run_id)
    if engine.ndim != 1 or session.shape != engine.shape or run.shape != engine.shape:
        raise ValueError("continuity identifiers must be aligned vectors")
    if not len(engine):
        return []
    boundary = np.flatnonzero(
        (np.diff(engine) != 1)
        | (session[1:] != session[:-1])
        | (run[1:] != run[:-1])
    ) + 1
    starts = np.concatenate(([0], boundary))
    ends = np.concatenate((boundary, [len(engine)]))
    return [np.arange(start, end, dtype=np.int64) for start, end in zip(starts, ends, strict=True)]


def continuity_blocks(
    engine_frame_idx: np.ndarray,
    session_index: np.ndarray,
    run_id: np.ndarray,
    *,
    block_frames: int = BLOCK_FRAMES,
) -> list[np.ndarray]:
    if block_frames < 1:
        raise ValueError("block_frames must be positive")
    blocks = [
        run[start : start + block_frames]
        for run in continuity_runs(engine_frame_idx, session_index, run_id)
        for start in range(0, len(run), block_frames)
    ]
    if not blocks:
        raise ValueError("no continuity blocks")
    return blocks


def block_ap_components(
    truth: np.ndarray,
    score: np.ndarray,
    eligible: np.ndarray,
    block_index: np.ndarray,
    block_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Decompose AP skill into fixed-ranking block contributions."""

    labels = np.asarray(truth, dtype=np.uint8)
    values = np.asarray(score, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    chosen = np.flatnonzero(mask)
    ranked = chosen[np.argsort(-values[chosen], kind="mergesort")]
    ranked_labels = labels[ranked]
    positives = int(ranked_labels.sum())
    if not len(ranked) or positives == 0 or positives == len(ranked_labels):
        nan = np.full(block_count, np.nan)
        return nan, nan.copy(), nan.copy(), float("nan")
    precision = np.cumsum(ranked_labels) / np.arange(1, len(ranked) + 1)
    contribution = np.zeros(len(labels), dtype=np.float64)
    positive_rank = ranked[ranked_labels == 1]
    contribution[positive_rank] = precision[ranked_labels == 1]
    contribution_by_block = np.bincount(
        block_index, weights=contribution, minlength=block_count
    ).astype(np.float64)
    positives_by_block = np.bincount(
        block_index, weights=labels * mask, minlength=block_count
    ).astype(np.float64)
    eligible_by_block = np.bincount(
        block_index, weights=mask, minlength=block_count
    ).astype(np.float64)
    average_precision = contribution_by_block.sum() / positives_by_block.sum()
    prevalence = positives_by_block.sum() / eligible_by_block.sum()
    skill = (average_precision - prevalence) / (1.0 - prevalence)
    return contribution_by_block, positives_by_block, eligible_by_block, float(skill)


def bootstrap_skill(
    components: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    block_weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    contribution, positives, eligible, observed = components
    if not np.isfinite(observed):
        return observed, np.full(len(block_weights), np.nan)
    positive_draw = block_weights @ positives
    eligible_draw = block_weights @ eligible
    with np.errstate(divide="ignore", invalid="ignore"):
        average_precision = (block_weights @ contribution) / positive_draw
        prevalence = positive_draw / eligible_draw
        skill = (average_precision - prevalence) / (1.0 - prevalence)
    skill[(positive_draw <= 0) | (positive_draw >= eligible_draw)] = np.nan
    return observed, skill


def _bh_fdr(p_values: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """BH over the frozen family, retaining NaNs as failed hypotheses."""

    values = np.asarray(p_values, dtype=np.float64)
    flat = values.reshape(-1)
    if len(flat) != FROZEN_FDR_FAMILY_SIZE:
        raise ValueError(
            f"BH family must contain exactly {FROZEN_FDR_FAMILY_SIZE} hypotheses"
        )
    finite = np.flatnonzero(np.isfinite(flat))
    result = np.zeros(len(flat), dtype=bool)
    if not len(finite):
        return result.reshape(values.shape)
    order = finite[np.argsort(flat[finite], kind="mergesort")]
    # The denominator remains the preregistered 140 even when a null surface
    # is inestimable; missing hypotheses are never silently removed.
    threshold = q * np.arange(1, len(order) + 1) / len(flat)
    passing = np.flatnonzero(flat[order] <= threshold)
    if len(passing):
        cutoff = flat[order[passing[-1]]]
        result[finite] = flat[finite] <= cutoff
    return result.reshape(values.shape)


def _upper_tail_p_value(observed: float, null_distribution: np.ndarray) -> float:
    null = np.asarray(null_distribution, dtype=np.float64)
    if not np.isfinite(observed) or not len(null) or not np.all(np.isfinite(null)):
        return float("nan")
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def _series(
    arrays: dict[str, np.ndarray],
    *,
    variants: Sequence[str],
    offsets: Sequence[int],
    seeds: Sequence[int],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Full-support val-A effects, averaged as metrics across three seeds."""

    truth = arrays["y_true"]
    state = arrays["post_state"]
    blocks = continuity_blocks(
        arrays["target_engine_frame_idx"],
        arrays["target_session_index"],
        arrays["target_run_id"],
    )
    block_index = np.empty(len(truth), dtype=np.int32)
    for index, rows in enumerate(blocks):
        block_index[rows] = index
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(
        len(blocks),
        np.full(len(blocks), 1.0 / len(blocks)),
        size=bootstrap_replicates,
    ).astype(np.float64)
    observed: dict[str, np.ndarray] = {}
    boot: dict[str, np.ndarray] = {}
    natural_observed: dict[str, np.ndarray] = {}
    natural_boot: dict[str, np.ndarray] = {}
    for variant in variants:
        observed[variant] = np.empty((len(offsets), len(OUTPUT_NAMES)))
        boot[variant] = np.empty(
            (bootstrap_replicates, len(offsets), len(OUTPUT_NAMES))
        )
        natural_observed[variant] = np.empty_like(observed[variant])
        natural_boot[variant] = np.empty_like(boot[variant])
        for offset_index, offset in enumerate(offsets):
            probabilities = [
                arrays[prediction_array_name(variant, int(offset), int(seed))]
                for seed in seeds
            ]
            for task in range(len(OUTPUT_NAMES)):
                seed_observed: list[float] = []
                seed_bootstrap: list[np.ndarray] = []
                seed_natural_observed: list[float] = []
                seed_natural_bootstrap: list[np.ndarray] = []
                for probability in probabilities:
                    components = block_ap_components(
                        truth[:, task],
                        probability[:, task],
                        _task_mask(state, task),
                        block_index,
                        len(blocks),
                    )
                    value, distribution = bootstrap_skill(components, weights)
                    seed_observed.append(value)
                    seed_bootstrap.append(distribution)
                    natural_components = block_ap_components(
                        truth[:, task],
                        probability[:, task],
                        np.ones(len(truth), dtype=bool),
                        block_index,
                        len(blocks),
                    )
                    value, distribution = bootstrap_skill(
                        natural_components, weights
                    )
                    seed_natural_observed.append(value)
                    seed_natural_bootstrap.append(distribution)
                observed[variant][offset_index, task] = _finite_mean(
                    np.asarray(seed_observed), axis=0
                )
                boot[variant][:, offset_index, task] = _finite_mean(
                    np.stack(seed_bootstrap), axis=0
                )
                natural_observed[variant][offset_index, task] = _finite_mean(
                    np.asarray(seed_natural_observed), axis=0
                )
                natural_boot[variant][:, offset_index, task] = _finite_mean(
                    np.stack(seed_natural_bootstrap), axis=0
                )
    return {
        "observed": observed,
        "bootstrap": boot,
        "natural_observed": natural_observed,
        "natural_bootstrap": natural_boot,
        "block_count": len(blocks),
    }


def _simultaneous_lower(observed: np.ndarray, bootstrap: np.ndarray) -> np.ndarray:
    difference = np.abs(bootstrap - observed[None, ...])
    finite = np.isfinite(difference)
    per_draw = np.full(len(difference), np.nan)
    valid_draw = finite.any(axis=(1, 2))
    if valid_draw.any():
        per_draw[valid_draw] = np.max(
            np.where(finite[valid_draw], difference[valid_draw], -np.inf),
            axis=(1, 2),
        )
    critical = float(np.nanquantile(per_draw, 0.95))
    return observed - critical


def _lift_tables(series: dict[str, Any], offsets: Sequence[int]) -> dict[str, Any]:
    offset_index = {int(offset): index for index, offset in enumerate(offsets)}
    positive_indices = [offset_index[offset] for offset in range(18)]
    negative_indices = [offset_index[offset] for offset in NEGATIVE_OFFSETS]
    result: dict[str, Any] = {}
    for pair, control in PAIR_CONTROL.items():
        pair_observed = series["observed"][pair]
        pair_boot = series["bootstrap"][pair]
        control_observed = series["observed"][control]
        control_boot = series["bootstrap"][control]
        negative_baseline = np.nanmax(pair_observed[negative_indices], axis=0)
        negative_bootstrap = np.nanmax(pair_boot[:, negative_indices], axis=1)
        causal = pair_observed[positive_indices] - negative_baseline
        causal_boot = pair_boot[:, positive_indices] - negative_bootstrap[:, None]
        pair_lift = pair_observed[positive_indices] - control_observed[positive_indices]
        pair_lift_boot = pair_boot[:, positive_indices] - control_boot[:, positive_indices]
        natural = series["natural_observed"][pair]
        natural_boot = series["natural_bootstrap"][pair]
        natural_baseline = np.nanmax(natural[negative_indices], axis=0)
        natural_boot_baseline = np.nanmax(natural_boot[:, negative_indices], axis=1)
        natural_causal = natural[positive_indices] - natural_baseline
        natural_causal_boot = natural_boot[:, positive_indices] - natural_boot_baseline[:, None]
        result[pair] = {
            "causal": causal,
            "causal_bootstrap": causal_boot,
            "causal_lower": _simultaneous_lower(causal, causal_boot),
            "pair_lift": pair_lift,
            "pair_lift_bootstrap": pair_lift_boot,
            "pair_lift_lower": _simultaneous_lower(pair_lift, pair_lift_boot),
            "natural_causal": natural_causal,
            "natural_causal_bootstrap": natural_causal_boot,
            "natural_causal_lower": _simultaneous_lower(
                natural_causal, natural_causal_boot
            ),
        }
    return result


def _retained_null_runs(arrays: dict[str, np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    runs = continuity_runs(
        arrays["target_engine_frame_idx"],
        arrays["target_session_index"],
        arrays["target_run_id"],
    )
    return (
        [run for run in runs if len(run) >= 2 * MIN_NULL_SHIFT],
        [run for run in runs if len(run) < 2 * MIN_NULL_SHIFT],
    )


def circular_shift_permutations(
    retained_runs: Sequence[np.ndarray],
    *,
    replicates: int,
    seed: int = CIRCULAR_NULL_SEED,
) -> Iterator[np.ndarray]:
    """Yield one independent legal circular offset per run and replicate."""

    if replicates < 1:
        raise ValueError("null replicates must be positive")
    if not retained_runs:
        return
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        shifted: list[np.ndarray] = []
        for run in retained_runs:
            length = len(run)
            if length < 2 * MIN_NULL_SHIFT:
                raise ValueError("short run reached circular shift sampler")
            # Inclusive upper bound: a 600-row run has the one legal shift 300.
            shift = int(rng.integers(MIN_NULL_SHIFT, length - MIN_NULL_SHIFT + 1))
            shifted.append(np.roll(run, shift))
        yield np.concatenate(shifted)


def _shift_target_bundle(
    arrays: dict[str, np.ndarray], permutation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = arrays["y_true"][permutation]
    state = arrays["post_state"][permutation]
    context = arrays["key_context"][permutation]
    return truth, state, context


def _mean_seed_skill(
    arrays: dict[str, np.ndarray],
    *,
    variant: str,
    offset: int,
    task: int,
    truth: np.ndarray,
    state: np.ndarray,
    prediction_rows: np.ndarray,
    seeds: Sequence[int] = (0, 1, 2),
) -> float:
    eligible = _task_mask(state, task)
    values = np.asarray(
        [
            _ap_skill(
                truth[:, task],
                arrays[prediction_array_name(variant, offset, seed)][
                    prediction_rows, task
                ],
                eligible,
            )
            for seed in seeds
        ]
    )
    return float(_finite_mean(values, axis=0))


def _score_contrast(
    truth: np.ndarray,
    score: np.ndarray,
    eligible: np.ndarray,
) -> float:
    """Mean event score minus same-post-state non-event score."""

    label = np.asarray(truth, dtype=bool)
    value = np.asarray(score, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    if label.shape != value.shape or label.shape != mask.shape:
        raise ValueError("score-contrast inputs must be aligned vectors")
    if np.any(label & ~mask):
        raise ValueError("event rows must be a subset of post-state eligibility")
    positive = label
    negative = mask & ~label
    if not positive.any() or not negative.any():
        return float("nan")
    return float(value[positive].mean() - value[negative].mean())


def _resolve_null_device(device_name: str) -> torch.device:
    requested = str(device_name).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA null scoring requested but CUDA is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("null scoring device must be cpu, cuda, or auto")
    return device


def _null_device_identity(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


def _shift_permutation_matrix(
    retained_runs: Sequence[np.ndarray],
    *,
    replicates: int,
) -> np.ndarray:
    """Materialize deterministic shifts once as compact row indices."""

    permutations = np.stack(
        list(
            circular_shift_permutations(
                retained_runs,
                replicates=replicates,
                seed=CIRCULAR_NULL_SEED,
            )
        )
    )
    if permutations.size and int(permutations.max()) > np.iinfo(np.int32).max:
        raise ValueError("retained null row index exceeds int32 capacity")
    return permutations.astype(np.int32, copy=False)


def _batched_score_contrasts(
    *,
    shifted_truth: np.ndarray,
    shifted_eligible: np.ndarray,
    scores: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Compute all shifted score contrasts by deterministic matrix products."""

    truth = np.asarray(shifted_truth, dtype=np.uint8)
    eligible = np.asarray(shifted_eligible, dtype=np.uint8)
    value = np.asarray(scores, dtype=np.float64)
    if truth.shape != eligible.shape or truth.ndim != 2:
        raise ValueError("shifted truth and eligibility must be aligned [R,N]")
    if value.ndim != 2 or value.shape[0] != truth.shape[1]:
        raise ValueError("score matrix must be [N,M]")
    if np.any(truth > eligible):
        raise ValueError("shifted events must remain post-state eligible")

    torch.use_deterministic_algorithms(True)
    target = torch.from_numpy(truth.astype(np.float64)).to(device)
    mask = torch.from_numpy(eligible.astype(np.float64)).to(device)
    score = torch.from_numpy(value).to(device)
    negative = mask - target
    positive_count = target.sum(dim=1, keepdim=True)
    negative_count = negative.sum(dim=1, keepdim=True)
    with torch.no_grad():
        positive_mean = (target @ score) / positive_count
        negative_mean = (negative @ score) / negative_count
        contrast = positive_mean - negative_mean
        invalid = (positive_count <= 0) | (negative_count <= 0)
        contrast = torch.where(
            invalid.expand_as(contrast),
            torch.full_like(contrast, torch.nan),
            contrast,
        )
    result = contrast.cpu().numpy()
    del target, mask, score, negative, positive_count, negative_count, contrast
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_synthetic_null_benchmark(
    *,
    device_name: str,
    samples: int = WORST_CASE_VALIDATION_SAMPLES,
    replicates: int = NULL_REPLICATES,
) -> dict[str, Any]:
    """Exercise the complete vectorized null at preregistered worst-case size."""

    if samples < 2 * MIN_NULL_SHIFT:
        raise ValueError("synthetic null benchmark needs one eligible continuity run")
    if replicates < 1:
        raise ValueError("synthetic null benchmark replicates must be positive")
    device = _resolve_null_device(device_name)
    rng = np.random.default_rng(2_026_072_703)
    previous = rng.integers(
        0, 2, size=(samples, len(KEY_ORDER)), dtype=np.uint8
    )
    current = rng.integers(
        0, 2, size=(samples, len(KEY_ORDER)), dtype=np.uint8
    )
    truth = np.concatenate(
        ((previous == 0) & (current == 1), (previous == 1) & (current == 0)),
        axis=1,
    ).astype(np.uint8)
    context = np.repeat(current[:, None, :], 23, axis=1)
    context[:, 4] = previous
    arrays: dict[str, np.ndarray] = {
        "y_true": truth,
        "post_state": current,
        "key_context": context,
        "context_relative_offsets": np.arange(-5, 18, dtype=np.int16),
        "target_engine_frame_idx": np.arange(samples, dtype=np.int64),
        "target_session_index": np.zeros(samples, dtype=np.int32),
        "target_run_id": np.zeros(samples, dtype=np.int64),
    }
    for pair in PAIR_CONTROL:
        for offset in (*NEGATIVE_OFFSETS, *CANDIDATE_HORIZONS):
            for seed in (0, 1, 2):
                arrays[prediction_array_name(pair, offset, seed)] = rng.random(
                    (samples, len(OUTPUT_NAMES)), dtype=np.float32
                )
    result = _null_and_fdr(
        arrays,
        null_replicates=replicates,
        device_name=str(device),
    )
    elapsed = float(result["computation"]["elapsed_seconds"])
    allowed = (
        samples >= WORST_CASE_VALIDATION_SAMPLES
        and replicates >= NULL_REPLICATES
        and np.isfinite(elapsed)
        and elapsed <= MAX_NULL_BENCHMARK_SECONDS
    )
    return {
        "status": "pass" if allowed else "blocked_over_fifteen_minutes",
        "real_data_or_validation_opened": False,
        "samples": int(samples),
        "replicates": int(replicates),
        "device": str(device),
        "device_identity": (
            _null_device_identity(device)
        ),
        "statistic": NULL_STATISTIC_CONTRACT,
        "elapsed_seconds": elapsed,
        "hard_limit_seconds": MAX_NULL_BENCHMARK_SECONDS,
        "allowed_to_open_real_shards": bool(allowed),
    }


def _null_and_fdr(
    arrays: dict[str, np.ndarray],
    *,
    null_replicates: int,
    device_name: str = "cpu",
) -> dict[str, Any]:
    retained_runs, excluded_runs = _retained_null_runs(arrays)
    retained = (
        np.concatenate(retained_runs)
        if retained_runs
        else np.asarray([], dtype=np.int64)
    )
    observed = np.full(
        (len(PAIR_CONTROL), len(CANDIDATE_HORIZONS), len(OUTPUT_NAMES)), np.nan
    )
    null_causal = np.full(
        (
            null_replicates,
            len(PAIR_CONTROL),
            len(CANDIDATE_HORIZONS),
            len(OUTPUT_NAMES),
        ),
        np.nan,
    )
    retained_truth = arrays["y_true"][retained]
    retained_state = arrays["post_state"][retained]
    device = _resolve_null_device(device_name)
    started = time.perf_counter()
    model_keys = [
        (representation, pair, offset)
        for representation, pair in enumerate(PAIR_CONTROL)
        for offset in (*NEGATIVE_OFFSETS, *CANDIDATE_HORIZONS)
    ]
    averaged_scores = {
        (representation, offset): _finite_mean(
            np.stack(
                [
                    arrays[prediction_array_name(pair, offset, seed)][retained]
                    for seed in (0, 1, 2)
                ]
            ),
            axis=0,
        )
        for representation, pair, offset in model_keys
    }
    observed_by_model = np.full(
        (len(PAIR_CONTROL), len(NEGATIVE_OFFSETS) + len(CANDIDATE_HORIZONS), len(OUTPUT_NAMES)),
        np.nan,
    )
    ordered_offsets = (*NEGATIVE_OFFSETS, *CANDIDATE_HORIZONS)
    if len(retained):
        for representation in range(len(PAIR_CONTROL)):
            for offset_index, offset in enumerate(ordered_offsets):
                probability = averaged_scores[(representation, offset)]
                for task in range(len(OUTPUT_NAMES)):
                    observed_by_model[representation, offset_index, task] = (
                        _score_contrast(
                            retained_truth[:, task],
                            probability[:, task],
                            _task_mask(retained_state, task),
                        )
                    )
        negative = observed_by_model[:, : len(NEGATIVE_OFFSETS)]
        baseline = np.nanmax(negative, axis=1)
        observed[:] = (
            observed_by_model[:, len(NEGATIVE_OFFSETS) :] - baseline[:, None, :]
        )

        permutations = _shift_permutation_matrix(
            retained_runs, replicates=null_replicates
        )
        zero = int(np.flatnonzero(arrays["context_relative_offsets"] == 0)[0])
        if not np.array_equal(arrays["key_context"][:, zero], arrays["post_state"]):
            raise RuntimeError("target context at t is not bound to post_state")
        for task in range(len(OUTPUT_NAMES)):
            key = task % len(KEY_ORDER)
            required_state = 1 if task < len(KEY_ORDER) else 0
            # ``permutations`` contains original sidecar row IDs, not compact
            # retained-array positions. Index the full target bundle while
            # keeping score columns in the concatenated retained-run order.
            shifted_truth = arrays["y_true"][permutations, task]
            shifted_eligible = (
                arrays["post_state"][permutations, key] == required_state
            ).astype(np.uint8)
            score_matrix = np.column_stack(
                [
                    averaged_scores[(representation, offset)][:, task]
                    for representation in range(len(PAIR_CONTROL))
                    for offset in ordered_offsets
                ]
            )
            contrast = _batched_score_contrasts(
                shifted_truth=shifted_truth,
                shifted_eligible=shifted_eligible,
                scores=score_matrix,
                device=device,
            ).reshape(
                null_replicates,
                len(PAIR_CONTROL),
                len(ordered_offsets),
            )
            shifted_baseline = np.nanmax(
                contrast[:, :, : len(NEGATIVE_OFFSETS)], axis=2
            )
            null_causal[:, :, :, task] = (
                contrast[:, :, len(NEGATIVE_OFFSETS) :]
                - shifted_baseline[:, :, None]
            )
    p_value = np.full_like(observed, np.nan)
    estimable = np.isfinite(observed) & np.all(np.isfinite(null_causal), axis=0)
    for index in zip(*np.nonzero(estimable), strict=True):
        p_value[index] = _upper_tail_p_value(
            float(observed[index]), null_causal[(slice(None), *index)]
        )
    conditioned_support = np.asarray(
        [int(_task_mask(retained_state, task).sum()) for task in range(len(OUTPUT_NAMES))]
    )
    positives = retained_truth.sum(axis=0).astype(np.int64) if len(retained) else np.zeros(len(OUTPUT_NAMES), dtype=np.int64)
    return {
        "observed_retained_causal": observed,
        "p_value": p_value,
        "fdr_pass": _bh_fdr(p_value),
        "null_causal": null_causal,
        "computation": {
            "device": str(device),
            "elapsed_seconds": float(time.perf_counter() - started),
            "statistic": NULL_STATISTIC_CONTRACT,
            "seed_probability_aggregation": (
                "arithmetic mean before contrast; exactly equal to averaging "
                "the per-seed contrasts because the statistic is linear"
            ),
        },
        "support": {
            "eligible_run_count": len(retained_runs),
            "excluded_short_run_count": len(excluded_runs),
            "retained_rows": int(len(retained)),
            "excluded_rows": int(sum(len(run) for run in excluded_runs)),
            "retained_conditioned_rows_per_task": conditioned_support.tolist(),
            "retained_positives_per_task": positives.tolist(),
            "null_inestimable_tasks": [
                OUTPUT_NAMES[index]
                for index in np.flatnonzero(~np.any(estimable, axis=(0, 1)))
            ],
        },
    }


def _loso_fold_lifts(
    arrays: dict[str, np.ndarray], pair: str, control: str
) -> dict[str, np.ndarray]:
    lengths = arrays["train_session_lengths"].astype(np.int64)
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    fold_causal = np.full((len(lengths), 18, len(OUTPUT_NAMES)), np.nan)
    fold_pair = np.full_like(fold_causal, np.nan)
    for fold, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        rows = np.arange(int(start), int(start + length), dtype=np.int64)
        truth = arrays["train_y_true"][rows]
        state = arrays["train_post_state"][rows]
        variant_skill: dict[str, np.ndarray] = {}
        for variant in (pair, control):
            skill = np.full((len(DEFAULT_OFFSETS), len(OUTPUT_NAMES)), np.nan)
            for offset_index, offset in enumerate(DEFAULT_OFFSETS):
                probability = arrays[
                    loso_prediction_array_name(variant, int(offset), 0)
                ][rows]
                for task in range(len(OUTPUT_NAMES)):
                    skill[offset_index, task] = _ap_skill(
                        truth[:, task], probability[:, task], _task_mask(state, task)
                    )
            variant_skill[variant] = skill
        negative = np.nanmax(variant_skill[pair][0:4], axis=0)
        fold_causal[fold] = variant_skill[pair][4:] - negative
        fold_pair[fold] = variant_skill[pair][4:] - variant_skill[control][4:]
    return {"causal": fold_causal, "pair_lift": fold_pair}


def _context_subset_masks(
    arrays: dict[str, np.ndarray], horizon: int
) -> dict[str, np.ndarray]:
    relative = arrays["context_relative_offsets"].astype(int)
    position = {int(value): index for index, value in enumerate(relative)}
    context = arrays["key_context"]
    if horizon < 0 or horizon > 17:
        raise ValueError("subset horizon must be in 0..17")
    intervening = np.zeros(len(context), dtype=bool)
    for delta in range(1, horizon + 1):
        intervening |= np.any(
            context[:, position[delta - 1]] != context[:, position[delta]], axis=1
        )
    event_count = arrays["y_true"].sum(axis=1)
    return {
        "no_intervening_action": ~intervening,
        "solo_transition": event_count == 1,
        "cooccurring_transition": event_count > 1,
    }


def _persistent_task_mask(
    arrays: dict[str, np.ndarray], horizon: int, task: int
) -> np.ndarray:
    relative = arrays["context_relative_offsets"].astype(int)
    position = {int(value): index for index, value in enumerate(relative)}
    key = task % len(KEY_ORDER)
    state = arrays["post_state"][:, key]
    observed = arrays["key_context"][:, [position[value] for value in range(horizon + 1)], key]
    return np.all(observed == state[:, None], axis=1)


def _subset_causal_lift(
    arrays: dict[str, np.ndarray],
    *,
    pair: str,
    horizon: int,
    task: int,
    subset: np.ndarray,
) -> float:
    state = arrays["post_state"]
    truth = arrays["y_true"]
    eligible = _task_mask(state, task) & subset
    negative = []
    for offset in NEGATIVE_OFFSETS:
        values = [
            _ap_skill(
                truth[:, task],
                arrays[prediction_array_name(pair, offset, seed)][:, task],
                eligible,
            )
            for seed in (0, 1, 2)
        ]
        negative.append(float(_finite_mean(np.asarray(values), axis=0)))
    horizon_values = [
        _ap_skill(
            truth[:, task],
            arrays[prediction_array_name(pair, horizon, seed)][:, task],
            eligible,
        )
        for seed in (0, 1, 2)
    ]
    return float(_finite_mean(np.asarray(horizon_values), axis=0) - np.nanmax(negative))


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _write_json_exclusive_atomic(destination: Path, payload: dict[str, Any]) -> None:
    """Serialize fully, fsync, and publish without overwrite or partial files."""

    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite JSON artifact: {target}")
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
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "circular_null_replicates": NULL_REPLICATES,
        "circular_null_seed": CIRCULAR_NULL_SEED,
        "minimum_circular_distance_rows": MIN_NULL_SHIFT,
        "fdr_q": FDR_Q,
        "fdr_family_size": FROZEN_FDR_FAMILY_SIZE,
        "null_statistic": NULL_STATISTIC_CONTRACT,
    }
    scoring = contract.get("scoring", {})
    mismatch = {
        key: {"expected": value, "observed": scoring.get(key)}
        for key, value in expected.items()
        if scoring.get(key) != value
    }
    if mismatch:
        raise ValueError(f"contract scoring knobs do not match scorer: {mismatch}")
    return contract


def score_probe(
    *,
    raw_report_path: Path,
    prediction_sidecar_path: Path,
    contract_path: Path,
    output: Path,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    null_replicates: int = NULL_REPLICATES,
    null_device_name: str,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite score report: {destination}")
    raw_bytes = Path(raw_report_path).read_bytes()
    raw = json.loads(raw_bytes)
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    sidecar_sha256 = sha256_file(Path(prediction_sidecar_path))
    contract_sha256 = sha256_file(Path(contract_path))
    _load_contract(contract_path)
    null_device = _resolve_null_device(null_device_name)
    bound = raw.get("provenance", {}).get("prediction_sidecar", {})
    if bound.get("sha256") != sidecar_sha256:
        raise ValueError("raw report does not bind this prediction sidecar hash")
    if raw.get("provenance", {}).get("contract", {}).get("sha256") != contract_sha256:
        raise ValueError("raw report does not bind this scoring contract hash")
    benchmark = raw.get("provenance", {}).get("runtime_benchmark", {})
    null_benchmark = benchmark.get("scoring_null_benchmark", {})
    if (
        null_benchmark.get("device") != str(null_device)
        or null_benchmark.get("device_identity") != _null_device_identity(null_device)
        or null_benchmark.get("statistic") != NULL_STATISTIC_CONTRACT
        or null_benchmark.get("allowed_to_open_real_shards") is not True
    ):
        raise ValueError(
            "score device/statistic does not match the validated pre-shard "
            "null benchmark"
        )
    arrays = validate_prediction_sidecar(prediction_sidecar_path)
    if arrays["offsets"].tolist() != list(DEFAULT_OFFSETS):
        raise ValueError("sidecar offsets do not match preregistration")
    if arrays["variants"].astype(str).tolist() != list(FEATURE_VARIANTS):
        raise ValueError("sidecar variants do not match preregistration")
    if arrays["seeds"].tolist() != [0, 1, 2]:
        raise ValueError("sidecar seeds do not match preregistration")

    final_series = _series(
        arrays,
        variants=FEATURE_VARIANTS,
        offsets=DEFAULT_OFFSETS,
        seeds=(0, 1, 2),
        bootstrap_replicates=bootstrap_replicates,
    )
    final_lifts = _lift_tables(final_series, DEFAULT_OFFSETS)
    null = _null_and_fdr(
        arrays,
        null_replicates=null_replicates,
        device_name=str(null_device),
    )
    validation_truth = arrays["y_true"]
    train_truth = arrays["train_y_true"]
    validation_support = validation_truth.sum(axis=0)
    train_lengths = arrays["train_session_lengths"]
    train_starts = np.concatenate(([0], np.cumsum(train_lengths)[:-1]))
    train_support_sessions = np.stack(
        [
            train_truth[int(start) : int(start + length)].sum(axis=0)
            for start, length in zip(train_starts, train_lengths, strict=True)
        ]
    )
    supported = (validation_support >= 30) & (
        np.sum(train_support_sessions >= 20, axis=0) >= 2
    )

    decisions: dict[str, Any] = {}
    any_global = False
    for representation, (pair, control) in enumerate(PAIR_CONTROL.items()):
        table = final_lifts[pair]
        loso = _loso_fold_lifts(arrays, pair, control)
        loso_causal_counts = np.sum(loso["causal"] > 0, axis=0)
        loso_pair_counts = np.sum(loso["pair_lift"] > 0, axis=0)
        horizon_rows: dict[str, Any] = {}
        global_passes: list[int] = []
        for candidate_position, horizon in enumerate(CANDIDATE_HORIZONS):
            causal = table["causal"][horizon]
            pair_lift = table["pair_lift"][horizon]
            adjacent = horizon + 1 if horizon < 17 else None
            at_h = (
                supported
                & (causal >= 0.03)
                & (pair_lift >= 0.01)
                & (table["causal_lower"][horizon] > 0)
                & (table["pair_lift_lower"][horizon] > 0)
                & (loso_causal_counts[horizon] >= 2)
                & (loso_pair_counts[horizon] >= 2)
                & null["fdr_pass"][representation, candidate_position]
            )
            if adjacent is None:
                adjacent_gate = np.zeros(len(OUTPUT_NAMES), dtype=bool)
            else:
                adjacent_gate = (
                    (table["causal"][adjacent] >= 0.03)
                    & (table["pair_lift"][adjacent] >= 0.01)
                    & (table["causal_lower"][adjacent] > 0)
                    & (table["pair_lift_lower"][adjacent] > 0)
                    & (loso_causal_counts[adjacent] >= 2)
                    & (loso_pair_counts[adjacent] >= 2)
                )
            qualifying = at_h & adjacent_gate
            keys = {
                KEY_ORDER[task % len(KEY_ORDER)]
                for task in np.flatnonzero(qualifying)
            }
            polarities = {
                "onset" if task < len(KEY_ORDER) else "release"
                for task in np.flatnonzero(qualifying)
            }
            subsets = _context_subset_masks(arrays, horizon)
            subset_lift: dict[str, np.ndarray] = {}
            for name, mask in subsets.items():
                subset_lift[name] = np.asarray(
                    [
                        _subset_causal_lift(
                            arrays,
                            pair=pair,
                            horizon=horizon,
                            task=task,
                            subset=mask,
                        )
                        for task in range(len(OUTPUT_NAMES))
                    ]
                )
            persistent_lift = np.asarray(
                [
                    _subset_causal_lift(
                        arrays,
                        pair=pair,
                        horizon=horizon,
                        task=task,
                        subset=_persistent_task_mask(arrays, horizon, task),
                    )
                    for task in range(len(OUTPUT_NAMES))
                ]
            )
            retained_half = (
                np.isfinite(subset_lift["no_intervening_action"])
                & (causal > 0)
                & (subset_lift["no_intervening_action"] >= 0.5 * causal)
            )
            macro_causal = float(_finite_mean(causal[supported], axis=0))
            macro_pair = float(_finite_mean(pair_lift[supported], axis=0))
            macro_causal_lower = float(
                _finite_mean(table["causal_lower"][horizon][supported], axis=0)
            )
            macro_pair_lower = float(
                _finite_mean(table["pair_lift_lower"][horizon][supported], axis=0)
            )
            macro_natural_lower = float(
                _finite_mean(
                    table["natural_causal_lower"][horizon][supported], axis=0
                )
            )
            null_supported = supported & np.isfinite(
                null["observed_retained_causal"][representation, candidate_position]
            )
            null_macro_observed = float(
                _finite_mean(
                    null["observed_retained_causal"][representation, candidate_position][
                        null_supported
                    ],
                    axis=0,
                )
            )
            null_macro_distribution = _finite_mean(
                null["null_causal"][:, representation, candidate_position, :][
                    :, null_supported
                ],
                axis=1,
            )
            null_macro_p = _upper_tail_p_value(
                null_macro_observed, null_macro_distribution
            )
            adjacent_macro_gate = False
            adjacent_loso_gate = False
            if adjacent is not None:
                adjacent_macro_gate = bool(
                    float(_finite_mean(table["causal"][adjacent][supported], axis=0))
                    >= 0.03
                    and float(_finite_mean(table["causal_lower"][adjacent][supported], axis=0))
                    > 0
                    and float(_finite_mean(table["pair_lift"][adjacent][supported], axis=0))
                    >= 0.01
                    and float(_finite_mean(table["pair_lift_lower"][adjacent][supported], axis=0))
                    > 0
                )
                fold_macro_causal = _finite_mean(
                    loso["causal"][:, adjacent, supported], axis=1
                )
                fold_macro_pair = _finite_mean(
                    loso["pair_lift"][:, adjacent, supported], axis=1
                )
                adjacent_loso_gate = bool(
                    np.sum((fold_macro_causal > 0) & (fold_macro_pair > 0)) >= 2
                )
            fold_macro_causal_h = _finite_mean(
                loso["causal"][:, horizon, supported], axis=1
            )
            fold_macro_pair_h = _finite_mean(
                loso["pair_lift"][:, horizon, supported], axis=1
            )
            macro_loso_gate = bool(
                np.sum((fold_macro_causal_h > 0) & (fold_macro_pair_h > 0)) >= 2
            )
            global_gate = bool(
                adjacent is not None
                and macro_causal >= 0.03
                and macro_causal_lower > 0
                and macro_pair >= 0.01
                and macro_pair_lower > 0
                and macro_natural_lower > -0.01
                and adjacent_macro_gate
                and macro_loso_gate
                and adjacent_loso_gate
                and int(qualifying.sum()) >= 7
                and len(keys) >= 4
                and polarities == {"onset", "release"}
                and np.isfinite(null_macro_p)
                and null_macro_p <= 0.05
                and int((retained_half & supported).sum())
                >= int(np.ceil(supported.sum() / 2))
            )
            if global_gate:
                global_passes.append(horizon)
                any_global = True
            horizon_rows[str(horizon)] = {
                "adjacent_confirmation_horizon": adjacent,
                "macro_conditioned_causal_lift": macro_causal,
                "macro_conditioned_causal_simultaneous_lower": macro_causal_lower,
                "macro_pair_lift": macro_pair,
                "macro_pair_lift_simultaneous_lower": macro_pair_lower,
                "macro_natural_causal_simultaneous_lower": macro_natural_lower,
                "shifted_null_macro_p": null_macro_p,
                "qualifying_tasks": [
                    OUTPUT_NAMES[index] for index in np.flatnonzero(qualifying)
                ],
                "supported_task_count": int(supported.sum()),
                "no_intervening_half_lift_task_count": int(
                    (retained_half & supported).sum()
                ),
                "loso_positive_fold_counts": {
                    "causal": _json_value(loso_causal_counts[horizon]),
                    "pair_over_control": _json_value(loso_pair_counts[horizon]),
                },
                "subset_causal_lift": {
                    **{name: _json_value(value) for name, value in subset_lift.items()},
                    "persistent_post_state": _json_value(persistent_lift),
                },
                "global_gate": global_gate,
            }
        decisions[pair] = {
            "control": control,
            "candidate_horizons": horizon_rows,
            "earliest_global_horizon": min(global_passes) if global_passes else None,
            "causal_lift": _json_value(table["causal"]),
            "pair_lift": _json_value(table["pair_lift"]),
            "causal_simultaneous_lower": _json_value(table["causal_lower"]),
            "pair_lift_simultaneous_lower": _json_value(table["pair_lift_lower"]),
        }

    frozen_counts = (
        bootstrap_replicates == BOOTSTRAP_REPLICATES
        and null_replicates == NULL_REPLICATES
    )
    report = {
        "schema_version": "madeleine.dynamics-offset-probe-score.v1",
        "status": "decision_complete" if frozen_counts else "test_mode_not_decision_complete",
        "bindings": {
            "scorer_script_sha256": sha256_file(Path(__file__)),
            "raw_report": {"path": str(Path(raw_report_path)), "sha256": raw_sha256},
            "prediction_sidecar": {
                "path": str(Path(prediction_sidecar_path)),
                "sha256": sidecar_sha256,
            },
            "contract": {"path": str(Path(contract_path)), "sha256": contract_sha256},
            "probe_runtime_benchmark": {
                "path": benchmark.get("path"),
                "sha256": benchmark.get("sha256"),
                "scoring_null_benchmark": null_benchmark,
            },
            "validation_truth_sha256": canonical_array_sha256(
                arrays["y_true"].astype(np.uint8)
            ),
            "validation_key_context_sha256": canonical_array_sha256(
                arrays["key_context"].astype(np.uint8)
            ),
        },
        "method": {
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "block_frames": BLOCK_FRAMES,
            "bootstrap": "paired fixed-ranking AP-contribution continuity-block bootstrap with simultaneous max-absolute-deviation bands",
            "val_A_seed_aggregation": (
                "AP effect, CI, LOSO, subset, and decision surfaces use the "
                "arithmetic mean of three per-seed metric values and never "
                "ensemble probabilities; the linear circular-null score "
                "contrast averages seed probabilities, which is algebraically "
                "identical to averaging its per-seed contrasts"
            ),
            "null_replicates": null_replicates,
            "circular_null_seed": CIRCULAR_NULL_SEED,
            "null_sampling": "one legal offset independently per retained continuity run, uniformly with replacement across replicates",
            "null_shift_bundle": "truth, post_state, and t-5..t+17 key context shift together; predictions remain fixed",
            "null_statistic": NULL_STATISTIC_CONTRACT,
            "null_computation": null["computation"],
            "null_tail": "one-sided upper; count(null_statistic >= observed_statistic)",
            "null_p_value": "(exceedances + 1) / (5000 + 1)",
            "minimum_circular_distance_rows": MIN_NULL_SHIFT,
            "short_run_policy": "continuity runs shorter than 600 rows are excluded and reported; gaps are never crossed",
            "fdr": "Benjamini-Hochberg q=0.05 over the frozen 2 representations x 5 candidate horizons x 14 tasks = 140 family; inestimable hypotheses remain in the denominator and fail",
            "candidate_horizons": list(CANDIDATE_HORIZONS),
            "negative_baseline_offsets": list(NEGATIVE_OFFSETS),
            "adjacent_confirmation": "candidate h must also pass effect, CI, and LOSO gates at h+1; +17 exists only to confirm candidate +16; FDR applies at h without expanding the frozen 140 family",
            "loso_seed": 0,
        },
        "support": {
            "validation_positive_per_task": validation_support.astype(int).tolist(),
            "train_positive_per_session_task": train_support_sessions.astype(int).tolist(),
            "supported_tasks": [
                OUTPUT_NAMES[index] for index in np.flatnonzero(supported)
            ],
            "validation_block_count": final_series["block_count"],
            "circular_null": null["support"],
        },
        "multiplicity": {
            "family_size": FROZEN_FDR_FAMILY_SIZE,
            "minimum_attainable_p": 1.0 / (NULL_REPLICATES + 1),
            "p_values": _json_value(null["p_value"]),
            "fdr_pass": _json_value(null["fdr_pass"]),
            "observed_retained_support_statistic": _json_value(
                null["observed_retained_causal"]
            ),
        },
        "representations": decisions,
        "decision": {
            "dynamics_pretraining_allowed": bool(any_global and frozen_counts),
            "reason": (
                "at least one representation passed the preregistered global horizon gate"
                if any_global and frozen_counts
                else "no preregistered global horizon gate passed, or scorer ran in test mode"
            ),
            "C_and_D_remain_blocked": not bool(any_global and frozen_counts),
        },
    }
    normalized_report = _json_value(report)
    _write_json_exclusive_atomic(destination, normalized_report)
    return normalized_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--prediction-sidecar", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--null-device",
        choices=("cpu", "cuda", "auto"),
        required=True,
        help="device for the vectorized circular-null matrix products",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    score_probe(
        raw_report_path=args.raw_report,
        prediction_sidecar_path=args.prediction_sidecar,
        contract_path=args.contract,
        output=args.output,
        null_device_name=args.null_device,
    )


if __name__ == "__main__":
    main()
