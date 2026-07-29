import json

import numpy as np
import pytest
import torch

from data.schema import KEY_ORDER
from experiments.score_dynamics_offset_probe import (
    CANDIDATE_HORIZONS,
    CIRCULAR_NULL_SEED,
    FROZEN_FDR_FAMILY_SIZE,
    MIN_NULL_SHIFT,
    NULL_REPLICATES,
    NULL_STATISTIC_CONTRACT,
    _bh_fdr,
    _batched_score_contrasts,
    _context_subset_masks,
    _json_value,
    _lift_tables,
    _loso_fold_lifts,
    _null_and_fdr,
    _retained_null_runs,
    _shift_target_bundle,
    _shift_permutation_matrix,
    _upper_tail_p_value,
    _write_json_exclusive_atomic,
    _series,
    circular_shift_permutations,
    continuity_blocks,
    run_synthetic_null_benchmark,
)
from experiments.dynamics_offset_probe import (
    DEFAULT_OFFSETS,
    FEATURE_VARIANTS,
    OUTPUT_NAMES,
    loso_prediction_array_name,
    prediction_array_name,
)


def test_continuity_blocks_never_cross_gap_session_or_run() -> None:
    engine = np.asarray([0, 1, 2, 9, 10, 0, 1, 2], dtype=np.int64)
    session = np.asarray([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    run = np.asarray([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int64)

    blocks = continuity_blocks(engine, session, run, block_frames=2)

    assert [block.tolist() for block in blocks] == [
        [0, 1],
        [2],
        [3, 4],
        [5, 6],
        [7],
    ]


def test_circular_null_excludes_599_rows_and_accepts_exactly_600() -> None:
    lengths = (599, 600)
    arrays = {
        "target_engine_frame_idx": np.concatenate(
            [np.arange(length, dtype=np.int64) for length in lengths]
        ),
        "target_session_index": np.concatenate(
            [np.full(length, index, dtype=np.int32) for index, length in enumerate(lengths)]
        ),
        "target_run_id": np.concatenate(
            [np.full(length, index, dtype=np.int64) for index, length in enumerate(lengths)]
        ),
    }
    retained, excluded = _retained_null_runs(arrays)

    assert [len(run) for run in retained] == [600]
    assert [len(run) for run in excluded] == [599]
    permutation = next(
        circular_shift_permutations(
            retained, replicates=1, seed=CIRCULAR_NULL_SEED
        )
    )
    assert np.array_equal(permutation, np.roll(retained[0], MIN_NULL_SHIFT))


def test_target_bundle_uses_one_shared_circular_mapping() -> None:
    samples = 600
    context = np.zeros((samples, 23, len(KEY_ORDER)), dtype=np.uint8)
    context[:, 5] = np.arange(samples)[:, None] % 2
    truth = np.zeros((samples, 2 * len(KEY_ORDER)), dtype=np.uint8)
    arrays = {
        "y_true": truth,
        "post_state": context[:, 5].copy(),
        "key_context": context,
    }
    permutation = np.roll(np.arange(samples), 300)

    shifted_truth, shifted_state, shifted_context = _shift_target_bundle(
        arrays, permutation
    )

    assert np.array_equal(shifted_truth, truth[permutation])
    assert np.array_equal(shifted_state, arrays["post_state"][permutation])
    assert np.array_equal(shifted_context, context[permutation])
    assert np.array_equal(shifted_context[:, 5], shifted_state)


def test_p_value_floor_and_frozen_family_bh_denominator() -> None:
    assert _upper_tail_p_value(2.0, np.ones(NULL_REPLICATES)) == 1 / 5001
    values = np.full((2, 5, 14), np.nan)
    assert values.size == FROZEN_FDR_FAMILY_SIZE
    values.flat[0] = 0.0001
    values.flat[1] = 0.001

    passed = _bh_fdr(values)

    assert passed.flat[0]
    # With a denominator of only the two finite tests this would pass; with
    # the frozen family of 140 it correctly fails.
    assert not passed.flat[1]
    assert int(passed.sum()) == 1


def test_context_subsets_distinguish_intervening_and_cooccurring_events() -> None:
    context = np.zeros((3, 23, len(KEY_ORDER)), dtype=np.uint8)
    # relative zero is position 5. Row 0 changes after t; row 1 stays stable.
    context[0, 5:, 0] = 1
    context[0, 7:, 1] = 1
    context[1, 5:, 0] = 1
    context[2, 5:, :2] = 1
    truth = np.zeros((3, 2 * len(KEY_ORDER)), dtype=np.uint8)
    truth[0, 0] = 1
    truth[1, 0] = 1
    truth[2, :2] = 1
    arrays = {
        "context_relative_offsets": np.arange(-5, 18, dtype=np.int16),
        "key_context": context,
        "y_true": truth,
    }

    masks = _context_subset_masks(arrays, horizon=3)

    assert masks["no_intervening_action"].tolist() == [False, True, True]
    assert masks["solo_transition"].tolist() == [True, True, False]
    assert masks["cooccurring_transition"].tolist() == [False, False, True]


def test_small_circular_null_is_finite_and_uses_fixed_family() -> None:
    samples = 600
    rng = np.random.default_rng(17)
    previous = rng.integers(0, 2, size=(samples, len(KEY_ORDER)), dtype=np.uint8)
    current = rng.integers(0, 2, size=(samples, len(KEY_ORDER)), dtype=np.uint8)
    onset = (previous == 0) & (current == 1)
    release = (previous == 1) & (current == 0)
    truth = np.concatenate((onset, release), axis=1).astype(np.uint8)
    context = np.repeat(current[:, None, :], 23, axis=1)
    context[:, 4] = previous
    arrays = {
        "y_true": truth,
        "post_state": current,
        "key_context": context,
        "context_relative_offsets": np.arange(-5, 18, dtype=np.int16),
        "target_engine_frame_idx": np.arange(samples, dtype=np.int64),
        "target_session_index": np.zeros(samples, dtype=np.int32),
        "target_run_id": np.zeros(samples, dtype=np.int64),
    }
    for variant in ("pooled_pair", "spatial_motion"):
        for offset in (*(-4, -3, -2, -1), *CANDIDATE_HORIZONS):
            for seed in (0, 1, 2):
                arrays[prediction_array_name(variant, offset, seed)] = rng.random(
                    (samples, len(OUTPUT_NAMES)), dtype=np.float32
                )

    result = _null_and_fdr(arrays, null_replicates=2)

    assert result["p_value"].shape == (2, 5, 14)
    assert np.all(np.isfinite(result["p_value"]))
    assert set(np.unique(result["p_value"])) <= {1 / 3, 2 / 3, 1.0}
    assert result["support"]["eligible_run_count"] == 1
    assert result["support"]["excluded_short_run_count"] == 0
    assert result["computation"]["statistic"] == NULL_STATISTIC_CONTRACT


def test_vectorized_score_contrast_matches_manual_values() -> None:
    shifted_truth = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.uint8)
    shifted_eligible = np.asarray([[1, 1, 1, 0], [1, 1, 0, 1]], dtype=np.uint8)
    scores = np.asarray(
        [[0.9, 0.1], [0.2, 0.8], [0.4, 0.3], [0.6, 0.7]],
        dtype=np.float64,
    )

    contrast = _batched_score_contrasts(
        shifted_truth=shifted_truth,
        shifted_eligible=shifted_eligible,
        scores=scores,
        device=torch.device("cpu"),
    )

    expected = np.asarray(
        [
            [0.9 - np.mean([0.2, 0.4]), 0.1 - np.mean([0.8, 0.3])],
            [0.2 - np.mean([0.9, 0.6]), 0.8 - np.mean([0.1, 0.7])],
        ]
    )
    assert np.allclose(contrast, expected)


def test_interleaved_short_runs_keep_global_shift_indices_aligned() -> None:
    lengths = (7, 600, 11, 600)
    samples = sum(lengths)
    rng = np.random.default_rng(31)
    previous = rng.integers(0, 2, size=(samples, len(KEY_ORDER)), dtype=np.uint8)
    current = rng.integers(0, 2, size=(samples, len(KEY_ORDER)), dtype=np.uint8)
    truth = np.concatenate(
        ((previous == 0) & (current == 1), (previous == 1) & (current == 0)),
        axis=1,
    ).astype(np.uint8)
    context = np.repeat(current[:, None, :], 23, axis=1)
    context[:, 4] = previous
    session = np.concatenate(
        [np.full(length, index, dtype=np.int32) for index, length in enumerate(lengths)]
    )
    engine = np.concatenate([np.arange(length, dtype=np.int64) for length in lengths])
    arrays = {
        "y_true": truth,
        "post_state": current,
        "key_context": context,
        "context_relative_offsets": np.arange(-5, 18, dtype=np.int16),
        "target_engine_frame_idx": engine,
        "target_session_index": session,
        "target_run_id": session.astype(np.int64),
    }
    for variant in ("pooled_pair", "spatial_motion"):
        for offset in (*(-4, -3, -2, -1), *CANDIDATE_HORIZONS):
            for seed in (0, 1, 2):
                arrays[prediction_array_name(variant, offset, seed)] = rng.random(
                    (samples, len(OUTPUT_NAMES)), dtype=np.float32
                )

    retained, excluded = _retained_null_runs(arrays)
    permutation = _shift_permutation_matrix(retained, replicates=1)[0]
    expected = np.concatenate([np.roll(run, 300) for run in retained])
    assert [len(run) for run in excluded] == [7, 11]
    assert np.array_equal(permutation, expected)
    assert np.array_equal(arrays["y_true"][permutation], truth[expected])

    result = _null_and_fdr(arrays, null_replicates=1)
    assert result["support"]["retained_rows"] == 1200
    assert result["support"]["excluded_rows"] == 18


def test_no_retained_null_is_nan_safe_and_json_publication_is_atomic(
    tmp_path,
) -> None:
    samples = 599
    rng = np.random.default_rng(41)
    previous = rng.integers(0, 2, size=(samples, len(KEY_ORDER)), dtype=np.uint8)
    current = rng.integers(0, 2, size=(samples, len(KEY_ORDER)), dtype=np.uint8)
    truth = np.concatenate(
        ((previous == 0) & (current == 1), (previous == 1) & (current == 0)),
        axis=1,
    ).astype(np.uint8)
    context = np.repeat(current[:, None, :], 23, axis=1)
    context[:, 4] = previous
    arrays = {
        "y_true": truth,
        "post_state": current,
        "key_context": context,
        "context_relative_offsets": np.arange(-5, 18, dtype=np.int16),
        "target_engine_frame_idx": np.arange(samples, dtype=np.int64),
        "target_session_index": np.zeros(samples, dtype=np.int32),
        "target_run_id": np.zeros(samples, dtype=np.int64),
    }
    for variant in ("pooled_pair", "spatial_motion"):
        for offset in (*(-4, -3, -2, -1), *CANDIDATE_HORIZONS):
            for seed in (0, 1, 2):
                arrays[prediction_array_name(variant, offset, seed)] = rng.random(
                    (samples, len(OUTPUT_NAMES)), dtype=np.float32
                )

    result = _null_and_fdr(arrays, null_replicates=2)
    normalized = _json_value(result)
    assert normalized["support"]["retained_rows"] == 0
    assert normalized["p_value"][0][0][0] is None

    destination = tmp_path / "score.json"
    _write_json_exclusive_atomic(destination, normalized)
    assert "NaN" not in destination.read_text(encoding="utf-8")
    assert json.loads(destination.read_text(encoding="utf-8"))["p_value"][0][0][0] is None
    with pytest.raises(FileExistsError):
        _write_json_exclusive_atomic(destination, normalized)


def test_small_synthetic_null_benchmark_is_honest_nonproduction_probe() -> None:
    receipt = run_synthetic_null_benchmark(
        device_name="cpu", samples=600, replicates=2
    )
    assert receipt["status"] == "blocked_over_fifteen_minutes"
    assert receipt["real_data_or_validation_opened"] is False
    assert receipt["samples"] == 600
    assert receipt["replicates"] == 2


def test_seed_mean_series_and_loso_fold_shapes_cover_plus_17() -> None:
    rng = np.random.default_rng(23)
    validation_samples = 120
    train_session_length = 80
    train_samples = 3 * train_session_length

    def target_bundle(samples: int) -> tuple[np.ndarray, np.ndarray]:
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
        return truth, current

    validation_truth, validation_state = target_bundle(validation_samples)
    train_truth, train_state = target_bundle(train_samples)
    arrays = {
        "y_true": validation_truth,
        "post_state": validation_state,
        "target_engine_frame_idx": np.arange(validation_samples, dtype=np.int64),
        "target_session_index": np.zeros(validation_samples, dtype=np.int32),
        "target_run_id": np.zeros(validation_samples, dtype=np.int64),
        "train_y_true": train_truth,
        "train_post_state": train_state,
        "train_session_lengths": np.full(3, train_session_length, dtype=np.int64),
    }
    for variant in FEATURE_VARIANTS:
        for offset in DEFAULT_OFFSETS:
            for seed in (0, 1, 2):
                arrays[prediction_array_name(variant, offset, seed)] = rng.random(
                    (validation_samples, len(OUTPUT_NAMES)), dtype=np.float32
                )
            arrays[loso_prediction_array_name(variant, offset, 0)] = rng.random(
                (train_samples, len(OUTPUT_NAMES)), dtype=np.float32
            )

    series = _series(
        arrays,
        variants=FEATURE_VARIANTS,
        offsets=DEFAULT_OFFSETS,
        seeds=(0, 1, 2),
        bootstrap_replicates=3,
    )
    lifts = _lift_tables(series, DEFAULT_OFFSETS)
    loso = _loso_fold_lifts(arrays, "pooled_pair", "pooled_same_frame")

    assert series["observed"]["pooled_pair"].shape == (22, 14)
    assert lifts["pooled_pair"]["causal"].shape == (18, 14)
    assert loso["causal"].shape == (3, 18, 14)
    assert np.all(np.isfinite(lifts["pooled_pair"]["causal"][17]))
