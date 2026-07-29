import json
from pathlib import Path

import numpy as np
import pytest
import torch

import experiments.score_dynamics_offset_probe as dynamics_scorer

from data.schema import KEY_ORDER
from experiments import dynamics_offset_probe as dynamics_probe
from experiments.dynamics_offset_probe import (
    CandidateSet,
    DEFAULT_OFFSETS,
    EncodedCorpus,
    EncodedSession,
    EXPECTED_BACKBONE_CONTRACT,
    FEATURE_VARIANTS,
    aggregate_seed_scores,
    binary_diagnostic_metrics,
    build_candidates,
    candidate_indices_for_run,
    concatenate_encoded_sessions,
    feature_dimension,
    fit_linear_probe,
    fit_linear_probe_from_matrix,
    fit_matrix_standardizer,
    load_and_validate_contract,
    load_rgb_session,
    materialized_feature_bytes,
    materialize_feature_matrix,
    pair_features,
    parse_offsets,
    reject_forbidden_sessions,
    require_canonical_session_list_bytes,
    require_materialization_capacity,
    score_predictions,
    standardize_matrix_in_place,
    target_key_context,
    transition_targets,
    validate_split_ids,
    validate_prediction_sidecar,
    validate_prediction_sidecar_arrays,
    validate_runtime_benchmark_receipt,
    write_prediction_sidecar,
)


def _encoded_session(
    *,
    session_id: str = "train-a",
    frame_idx: np.ndarray | None = None,
    input_active: np.ndarray | None = None,
    seed: int = 5,
) -> EncodedSession:
    if frame_idx is None:
        frame_idx = np.arange(80, dtype=np.int64)
    if input_active is None:
        input_active = np.ones(len(frame_idx), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    # Independent bits yield both onset and release examples for every key.
    keys = rng.integers(0, 2, size=(len(frame_idx), len(KEY_ORDER)), dtype=np.uint8)
    return EncodedSession(
        session_id=session_id,
        path=Path(f"/{session_id}.npz"),
        shard_sha256="a" * 64,
        pooled=rng.normal(size=(len(frame_idx), 512)).astype(np.float16),
        coarse_spatial=rng.normal(size=(len(frame_idx), 256, 4, 4)).astype(
            np.float16
        ),
        keys=keys,
        engine_frame_idx=frame_idx,
        input_active=input_active,
    )


def test_candidate_indices_for_run_align_pair_and_transition() -> None:
    observed_previous, observed_current, target_previous, target_current = (
        candidate_indices_for_run(10, 20, 3)
    )

    assert target_current.tolist() == [11, 12, 13, 14, 15, 16]
    assert target_previous.tolist() == [10, 11, 12, 13, 14, 15]
    assert observed_current.tolist() == [14, 15, 16, 17, 18, 19]
    assert observed_previous.tolist() == [13, 14, 15, 16, 17, 18]


def test_common_candidates_use_identical_active_gap_bounded_support() -> None:
    frame_idx = np.asarray([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16])
    active = np.ones(len(frame_idx), dtype=np.uint8)
    corpus = concatenate_encoded_sessions(
        [_encoded_session(frame_idx=frame_idx, input_active=active)]
    )

    negative = build_candidates(corpus, -1, common_offsets=(-1, 2))
    positive = build_candidates(corpus, 2, common_offsets=(-1, 2))

    # The target indices are common across offsets and never bridge the 5->10
    # engine-frame discontinuity.
    assert negative.target_current.tolist() == positive.target_current.tolist()
    assert negative.target_current.tolist() == [2, 3, 8, 9, 10]
    assert np.all(
        corpus.engine_frame_idx[0][negative.target_current[:2]]
        == np.asarray([2, 3])
    )
    assert negative.observed_current.tolist() == [1, 2, 7, 8, 9]
    assert positive.observed_current.tolist() == [4, 5, 10, 11, 12]

    # Inactivating any row in the full common window removes that target for
    # every offset, rather than silently giving offsets unequal support.
    active[8] = 0
    inactive_corpus = concatenate_encoded_sessions(
        [_encoded_session(frame_idx=frame_idx, input_active=active)]
    )
    a = build_candidates(inactive_corpus, -1, common_offsets=(-1, 2))
    b = build_candidates(inactive_corpus, 2, common_offsets=(-1, 2))
    assert a.target_current.tolist() == b.target_current.tolist()
    assert a.target_current.tolist() == [2, 3]


def test_transition_targets_keep_onset_and_release_separate() -> None:
    keys = np.zeros((4, len(KEY_ORDER)), dtype=np.uint8)
    keys[1, 0] = 1  # left onset
    keys[2, 0] = 1  # held
    keys[3, 0] = 0  # left release
    candidates = CandidateSet(
        observed_previous=np.asarray([0, 1, 2]),
        observed_current=np.asarray([1, 2, 3]),
        target_previous=np.asarray([0, 1, 2]),
        target_current=np.asarray([1, 2, 3]),
        per_session={"s": 3},
    )

    targets = transition_targets(keys, candidates)

    assert targets[:, 0].tolist() == [1.0, 0.0, 0.0]
    assert targets[:, len(KEY_ORDER)].tolist() == [0.0, 0.0, 1.0]
    assert int(targets[:, 1:].sum()) == 1


def test_target_key_context_is_exact_t_minus_5_through_t_plus_17() -> None:
    session = _encoded_session(frame_idx=np.arange(40, dtype=np.int64))
    corpus = concatenate_encoded_sessions([session])
    candidates = build_candidates(corpus, 0, common_offsets=tuple(range(-4, 18)))

    context = target_key_context(corpus, candidates)

    assert context.shape == (len(candidates), 23, len(KEY_ORDER))
    first_target = int(candidates.target_current[0])
    assert np.array_equal(context[0], corpus.keys[first_target - 5 : first_target + 18])


def test_plus_17_confirmation_preserves_candidate_plus_16_boundary() -> None:
    corpus = concatenate_encoded_sessions(
        [_encoded_session(frame_idx=np.arange(50, dtype=np.int64))]
    )
    at_sixteen = build_candidates(corpus, 16, common_offsets=DEFAULT_OFFSETS)
    at_seventeen = build_candidates(corpus, 17, common_offsets=DEFAULT_OFFSETS)

    assert np.array_equal(at_sixteen.target_current, at_seventeen.target_current)
    assert np.array_equal(
        at_seventeen.observed_current, at_seventeen.target_current + 17
    )


def test_pair_features_include_signed_absolute_and_prepool_motion() -> None:
    session = _encoded_session(frame_idx=np.arange(3, dtype=np.int64))
    pooled = np.zeros((3, 512), dtype=np.float16)
    pooled[0, 0] = 1
    pooled[1, 0] = -1
    spatial = np.zeros((3, 256, 4, 4), dtype=np.float16)
    spatial[1, 0, 0, 0] = -2
    session = EncodedSession(
        session.session_id,
        session.path,
        session.shard_sha256,
        pooled,
        spatial,
        session.keys[:3],
        session.engine_frame_idx,
        session.input_active,
    )
    corpus = concatenate_encoded_sessions([session])
    candidates = CandidateSet(
        observed_previous=np.asarray([0]),
        observed_current=np.asarray([1]),
        target_previous=np.asarray([0]),
        target_current=np.asarray([1]),
        per_session={"train-a": 1},
    )

    same = pair_features(
        corpus,
        candidates,
        np.asarray([0]),
        variant="pooled_same_frame",
        device=torch.device("cpu"),
    )
    pooled_pair = pair_features(
        corpus,
        candidates,
        np.asarray([0]),
        variant="pooled_pair",
        device=torch.device("cpu"),
    )
    spatial_motion = pair_features(
        corpus,
        candidates,
        np.asarray([0]),
        variant="spatial_motion",
        device=torch.device("cpu"),
    )

    assert same.shape == (1, feature_dimension("pooled_same_frame"))
    assert pooled_pair.shape == (1, feature_dimension("pooled_pair"))
    assert spatial_motion.shape == (1, feature_dimension("spatial_motion"))
    # Signed and absolute pooled deltas differ in sign at the changed feature.
    assert pooled_pair[0, 2 * 512] < 0
    assert pooled_pair[0, 3 * 512] > 0
    # Spatial current/signed blocks are negative, absolute is positive.
    block = 256 * 4 * 4
    assert spatial_motion[0, 0] < 0
    assert spatial_motion[0, block] < 0
    assert spatial_motion[0, 2 * block] > 0


def test_metrics_report_natural_prevalence_ap_and_fixed_f1() -> None:
    truth = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    probability = np.asarray([0.1, 0.9, 0.2, 0.8])

    metrics = binary_diagnostic_metrics(truth, probability)

    assert metrics["prevalence_chance_ap"] == 0.5
    assert metrics["average_precision"] == 1.0
    assert metrics["fixed_threshold"] == 0.5
    assert metrics["fixed_f1"] == 1.0

    repeated_truth = np.tile(truth[:, None], (1, 2 * len(KEY_ORDER)))
    repeated_probability = np.tile(probability[:, None], (1, 2 * len(KEY_ORDER)))
    report = score_predictions(repeated_truth, repeated_probability)
    assert report["macro"]["all_event_average_precision"] == 1.0
    aggregate = aggregate_seed_scores([report, report])
    assert aggregate["all_event_average_precision"] == {
        "mean": 1.0,
        "std_population": 0.0,
    }

    post_state = np.zeros((8, len(KEY_ORDER)), dtype=np.uint8)
    post_state[:4] = 1
    conditioned_truth = np.zeros((8, 2 * len(KEY_ORDER)), dtype=np.uint8)
    conditioned_probability = np.full(conditioned_truth.shape, 0.1)
    conditioned_truth[[1, 3], : len(KEY_ORDER)] = 1
    conditioned_probability[[1, 3], : len(KEY_ORDER)] = 0.9
    conditioned_truth[[5, 7], len(KEY_ORDER) :] = 1
    conditioned_probability[[5, 7], len(KEY_ORDER) :] = 0.9
    conditioned_report = score_predictions(
        conditioned_truth,
        conditioned_probability,
        post_state=post_state,
    )
    assert (
        conditioned_report["macro"]["conditioned_all_event_average_precision"]
        == 1.0
    )


def test_one_class_metric_and_fit_are_reported_inestimable_without_abort() -> None:
    metric = binary_diagnostic_metrics(
        np.zeros(8, dtype=np.uint8), np.full(8, 0.2, dtype=np.float32)
    )
    assert metric["estimable"] is False
    assert metric["inestimable_reason"] == "one-class support"
    assert metric["average_precision"] == 0.0
    assert metric["normalized_ap_skill"] == 0.0

    features = torch.zeros((12, feature_dimension("pooled_same_frame")))
    targets = np.zeros((12, 2 * len(KEY_ORDER)), dtype=np.float32)
    targets[:, 1:] = np.tile(np.arange(12)[:, None] % 2, (1, targets.shape[1] - 1))
    _, receipt = fit_linear_probe_from_matrix(
        features,
        targets,
        np.arange(12),
        variant="pooled_same_frame",
        offset=0,
        seed=0,
        epochs=1,
        batch_size=6,
        learning_rate=1e-3,
        weight_decay=1e-4,
        positive_weight_cap=50.0,
    )
    assert receipt["per_output_estimable"][f"{KEY_ORDER[0]}:onset"] is False
    assert receipt["per_output_positive_weight"][f"{KEY_ORDER[0]}:onset"] == 1.0


def test_linear_probe_fit_is_deterministic_and_records_balancing() -> None:
    corpus = concatenate_encoded_sessions([_encoded_session()])
    candidates = build_candidates(corpus, 0, common_offsets=(0,))
    kwargs = dict(
        variant="pooled_same_frame",
        offset=0,
        device=torch.device("cpu"),
        seed=7,
        epochs=2,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_train_samples=0,
        positive_weight_cap=50.0,
    )

    _, _, first = fit_linear_probe(corpus, candidates, **kwargs)
    _, _, second = fit_linear_probe(corpus, candidates, **kwargs)

    assert first["model_state_sha256"] == second["model_state_sha256"]
    assert first["loss"].startswith("class-balanced BCE")
    assert first["positive_weight_cap"] == 50.0


def test_materialized_run_path_reuses_standardized_tensor_across_seeds() -> None:
    corpus = concatenate_encoded_sessions([_encoded_session()])
    candidates = build_candidates(corpus, 0, common_offsets=(0,))
    features = materialize_feature_matrix(
        corpus,
        candidates,
        variant="pooled_same_frame",
        device=torch.device("cpu"),
        batch_size=32,
    )
    selection = np.arange(len(candidates), dtype=np.int64)
    standardizer = fit_matrix_standardizer(features, selection, batch_size=32)
    standardize_matrix_in_place(features, standardizer, batch_size=32)
    targets = transition_targets(corpus.keys, candidates)
    kwargs = dict(
        variant="pooled_same_frame",
        offset=0,
        seed=9,
        epochs=2,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        positive_weight_cap=50.0,
    )

    _, first = fit_linear_probe_from_matrix(
        features, targets, selection, **kwargs
    )
    _, second = fit_linear_probe_from_matrix(
        features, targets, selection, **kwargs
    )

    assert features.device.type == "cpu"
    assert first["model_state_sha256"] == second["model_state_sha256"]


def test_rgb_loader_is_fail_closed_and_rejects_embargo_before_access(
    tmp_path: Path,
) -> None:
    session_id = "train-a"
    frames = np.zeros((3, 128, 128, 3), dtype=np.uint8)
    keys = np.zeros((3, len(KEY_ORDER)), dtype=np.uint8)
    np.savez_compressed(
        tmp_path / f"{session_id}.npz",
        frames=frames,
        keys=keys,
        engine_frame_idx=np.arange(3, dtype=np.int64),
        input_active=np.ones(3, dtype=np.uint8),
        session_id=np.asarray(session_id),
    )

    loaded = load_rgb_session(tmp_path, session_id)
    assert loaded.frames.shape == (3, 128, 128, 3)
    assert loaded.input_active.tolist() == [1, 1, 1]

    with pytest.raises(ValueError, match="embargoed"):
        reject_forbidden_sessions(["rec_20260727_220000_test"])
    with pytest.raises(ValueError, match="embargoed"):
        load_rgb_session(tmp_path, "rec_20260727_220000_test")


def test_split_and_offset_parsing_contracts() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_split_ids(["a"], ["a"])
    with pytest.raises(ValueError, match="embargoed"):
        validate_split_ids(["rec_20260727_220000_test"], ["val"])
    assert parse_offsets("-4:17") == tuple(range(-4, 18))
    assert parse_offsets("-4,0,16") == (-4, 0, 16)


def test_split_list_bytes_are_canonical_and_exact(tmp_path: Path) -> None:
    split = tmp_path / "train.txt"
    split.write_bytes(b"a\nb\n")
    require_canonical_session_list_bytes(split, ["a", "b"])
    split.write_bytes(b"a\nb")
    with pytest.raises(ValueError, match="bytes"):
        require_canonical_session_list_bytes(split, ["a", "b"])


def test_preregistration_contract_binds_splits_hashes_and_fit(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    contract = {
        "schema_version": "madeleine.dynamics-offset-probe.v1",
        "study_id": "fixture",
        "data": {
            "root": str(tmp_path),
            "train_sessions": ["train"],
            "validation_sessions": ["val"],
            "build_manifest_sha256": "c" * 64,
            "train_shard_sha256": {"train": "a" * 64},
            "validation_shard_sha256": {"val": "b" * 64},
        },
        "offsets_native_frames": list(range(-4, 18)),
        "targets": {"key_order": KEY_ORDER},
        "probe_surfaces": {"backbone": EXPECTED_BACKBONE_CONTRACT},
        "fit": {
            "random_seeds": [0, 1, 2],
            "epochs": 40,
            "batch_size": 2048,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "embargo": {"sealed_untouched_session": "rec_20260727_220000_test"},
    }
    contract_path.write_text(json.dumps(contract))
    kwargs = dict(
        data_dir=tmp_path,
        train_ids=["train"],
        validation_ids=["val"],
        offsets=list(range(-4, 18)),
        variants=FEATURE_VARIANTS,
        seeds=[0, 1, 2],
        epochs=40,
        batch_size=2048,
        learning_rate=0.001,
        weight_decay=0.0001,
        max_train_samples=0,
        positive_weight_cap=50.0,
    )

    loaded = load_and_validate_contract(contract_path, **kwargs)
    assert loaded["study_id"] == "fixture"
    with pytest.raises(ValueError, match="does not match"):
        load_and_validate_contract(contract_path, **dict(kwargs, epochs=39))


def test_prediction_sidecar_is_aligned_atomic_and_exclusive(tmp_path: Path) -> None:
    truth = np.zeros((4, 2 * len(KEY_ORDER)), dtype=np.uint8)
    arrays = {
        "y_true": truth,
        "post_state": np.zeros((4, len(KEY_ORDER)), dtype=np.uint8),
        "key_context": np.zeros((4, 23, len(KEY_ORDER)), dtype=np.uint8),
        "context_relative_offsets": np.arange(-5, 18, dtype=np.int16),
        "target_global_index": np.arange(4, dtype=np.int64),
        "target_engine_frame_idx": np.arange(10, 14, dtype=np.int64),
        "target_session_index": np.zeros(4, dtype=np.int32),
        "target_run_id": np.zeros(4, dtype=np.int64),
        "session_ids": np.asarray(["val"]),
        "session_lengths": np.asarray([4], dtype=np.int64),
        "offsets": np.asarray([0], dtype=np.int16),
        "variants": np.asarray(["pooled_pair"]),
        "seeds": np.asarray([0], dtype=np.int64),
        "train_y_true": truth.copy(),
        "train_post_state": np.zeros((4, len(KEY_ORDER)), dtype=np.uint8),
        "train_key_context": np.zeros((4, 23, len(KEY_ORDER)), dtype=np.uint8),
        "train_target_global_index": np.arange(4, dtype=np.int64),
        "train_target_engine_frame_idx": np.arange(20, 24, dtype=np.int64),
        "train_target_session_index": np.zeros(4, dtype=np.int32),
        "train_target_run_id": np.zeros(4, dtype=np.int64),
        "train_session_ids": np.asarray(["train"]),
        "train_session_lengths": np.asarray([4], dtype=np.int64),
        "y_prob__pooled_pair__offset_p00__seed_0": np.full(
            truth.shape, 0.25, dtype=np.float32
        ),
        "loso_prob__pooled_pair__offset_p00__seed_0": np.full(
            truth.shape, 0.5, dtype=np.float32
        ),
    }
    path = tmp_path / "predictions.npz"

    digest = write_prediction_sidecar(path, arrays)
    loaded = validate_prediction_sidecar(path)

    assert len(digest) == 64
    assert np.array_equal(loaded["y_true"], truth)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_prediction_sidecar(path, arrays)
    invalid = dict(arrays)
    invalid["y_prob__pooled_pair__offset_p00__seed_0"] = np.full(
        truth.shape, np.nan, dtype=np.float32
    )
    with pytest.raises(ValueError, match="finite"):
        validate_prediction_sidecar_arrays(invalid)

    y_name = "y_prob__pooled_pair__offset_p00__seed_0"
    loso_name = "loso_prob__pooled_pair__offset_p00__seed_0"
    malformed_members: list[tuple[dict[str, np.ndarray], str]] = []

    renamed_y = dict(arrays)
    renamed_y["y_prob__pooled_pair__offset_p01__seed_0"] = renamed_y.pop(y_name)
    malformed_members.append((renamed_y, "y_prob member names"))
    extra_y = dict(arrays)
    extra_y["y_prob__pooled_pair__offset_p00__seed_1"] = arrays[y_name]
    malformed_members.append((extra_y, "y_prob member names"))
    missing_y = dict(arrays)
    missing_y.pop(y_name)
    malformed_members.append((missing_y, "y_prob member names"))

    renamed_loso = dict(arrays)
    renamed_loso["loso_prob__pooled_pair__offset_p01__seed_0"] = (
        renamed_loso.pop(loso_name)
    )
    malformed_members.append((renamed_loso, "loso_prob member names"))
    extra_loso = dict(arrays)
    extra_loso["loso_prob__pooled_pair__offset_p00__seed_1"] = arrays[loso_name]
    malformed_members.append((extra_loso, "loso_prob member names"))
    missing_loso = dict(arrays)
    missing_loso.pop(loso_name)
    malformed_members.append((missing_loso, "loso_prob member names"))

    for malformed, message in malformed_members:
        with pytest.raises(ValueError, match=message):
            validate_prediction_sidecar_arrays(malformed)


def test_materialized_byte_estimate_is_exact() -> None:
    assert materialized_feature_bytes(10, 5, "pooled_pair") == 15 * 2048 * 4
    with pytest.raises(MemoryError, match="above configured limit"):
        require_materialization_capacity(
            required_bytes=1024,
            configured_limit_bytes=512,
            device=torch.device("cpu"),
        )


def _runtime_benchmark_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text('{"fixture": true}\n', encoding="utf-8")
    receipt_path = tmp_path / "runtime.json"
    cells = {
        variant: {
            "elapsed_seconds": 1.0,
            "feature_dimension": dynamics_probe.feature_dimension(variant),
            "capacity": {},
            "loso_fits": 3,
            "final_fits": 3,
            "epochs_per_fit": 40,
        }
        for variant in FEATURE_VARIANTS
    }
    measured = float(sum(cell["elapsed_seconds"] for cell in cells.values()))
    projected = float(
        measured
        * len(DEFAULT_OFFSETS)
        * dynamics_probe.BENCHMARK_RUNTIME_MULTIPLIER
        + dynamics_probe.BENCHMARK_FIXED_OVERHEAD_SECONDS
    )
    receipt = {
        "schema_version": "madeleine.dynamics-offset-probe-runtime.v1",
        "status": "pass",
        "real_data_or_validation_opened": False,
        "contract_sha256": dynamics_probe.sha256_file(contract_path),
        "script_sha256": dynamics_probe.sha256_file(
            Path(dynamics_probe.__file__)
        ),
        "scorer_script_sha256": dynamics_probe.sha256_file(
            Path(dynamics_scorer.__file__)
        ),
        "device": "cpu",
        "device_identity": "cpu",
        "worst_case_samples": {
            "train": dynamics_probe.WORST_CASE_TRAIN_SAMPLES,
            "validation": dynamics_probe.WORST_CASE_VALIDATION_SAMPLES,
        },
        "scientific_knobs": {
            "offsets": list(DEFAULT_OFFSETS),
            "variants": list(FEATURE_VARIANTS),
            "seeds": [0, 1, 2],
            "loso_folds": 3,
            "epochs": 40,
            "batch_size": 2048,
        },
        "cells": cells,
        "scoring_null_benchmark": {
            "schema_version": "madeleine.dynamics-offset-null-runtime.v1",
            "contract_sha256": dynamics_probe.sha256_file(contract_path),
            "scorer_script_sha256": dynamics_probe.sha256_file(
                Path(dynamics_scorer.__file__)
            ),
            "status": "pass",
            "real_data_or_validation_opened": False,
            "samples": dynamics_probe.WORST_CASE_VALIDATION_SAMPLES,
            "replicates": dynamics_scorer.NULL_REPLICATES,
            "device": "cpu",
            "device_identity": "cpu",
            "statistic": dynamics_scorer.NULL_STATISTIC_CONTRACT,
            "elapsed_seconds": 30.0,
            "hard_limit_seconds": dynamics_scorer.MAX_NULL_BENCHMARK_SECONDS,
            "allowed_to_open_real_shards": True,
        },
        "projection": {
            "measured_one_offset_all_surfaces_seconds": measured,
            "offset_count": len(DEFAULT_OFFSETS),
            "runtime_multiplier_for_host_materialization_and_variance": (
                dynamics_probe.BENCHMARK_RUNTIME_MULTIPLIER
            ),
            "fixed_encoding_compression_verification_seconds": (
                dynamics_probe.BENCHMARK_FIXED_OVERHEAD_SECONDS
            ),
            "projected_full_runtime_seconds": projected,
            "hard_limit_seconds": dynamics_probe.MAX_PROJECTED_RUNTIME_SECONDS,
            "allowed_to_open_real_shards": True,
        },
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return contract_path, receipt_path, receipt


def test_runtime_benchmark_receipt_accepts_exact_current_bindings(
    tmp_path: Path,
) -> None:
    contract_path, receipt_path, _ = _runtime_benchmark_fixture(tmp_path)

    loaded = validate_runtime_benchmark_receipt(
        receipt_path,
        contract_path=contract_path,
        device=torch.device("cpu"),
    )

    assert loaded["status"] == "pass"


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("contract", ValueError, "contract hash is stale"),
        ("script", ValueError, "script hash is stale"),
        ("scorer_script", ValueError, "scorer script hash is stale"),
        ("device", ValueError, "device does not match"),
        ("knob", ValueError, "scientific knobs do not match"),
        ("projection", ValueError, "projection arithmetic is invalid"),
        ("null", RuntimeError, "circular-null benchmark receipt is unsafe"),
        ("over_two_hours", RuntimeError, "more than two GPU-hours"),
    ],
)
def test_runtime_benchmark_receipt_rejects_stale_or_unsafe_receipts(
    tmp_path: Path,
    case: str,
    error: type[Exception],
    message: str,
) -> None:
    contract_path, receipt_path, receipt = _runtime_benchmark_fixture(tmp_path)
    if case == "contract":
        receipt["contract_sha256"] = "0" * 64
    elif case == "script":
        receipt["script_sha256"] = "0" * 64
    elif case == "scorer_script":
        receipt["scorer_script_sha256"] = "0" * 64
    elif case == "device":
        receipt["device_identity"] = "not-the-current-device"
    elif case == "knob":
        receipt["scientific_knobs"]["epochs"] = 39
    elif case == "projection":
        receipt["projection"]["projected_full_runtime_seconds"] += 1
    elif case == "null":
        receipt["scoring_null_benchmark"]["replicates"] -= 1
    elif case == "over_two_hours":
        elapsed = (
            dynamics_probe.MAX_PROJECTED_RUNTIME_SECONDS
            - dynamics_probe.BENCHMARK_FIXED_OVERHEAD_SECONDS
        ) / (
            len(DEFAULT_OFFSETS) * dynamics_probe.BENCHMARK_RUNTIME_MULTIPLIER
        ) + 1
        first = FEATURE_VARIANTS[0]
        receipt["cells"][first]["elapsed_seconds"] = elapsed
        measured = sum(
            cell["elapsed_seconds"] for cell in receipt["cells"].values()
        )
        projected = (
            measured
            * len(DEFAULT_OFFSETS)
            * dynamics_probe.BENCHMARK_RUNTIME_MULTIPLIER
            + dynamics_probe.BENCHMARK_FIXED_OVERHEAD_SECONDS
        )
        receipt["projection"]["measured_one_offset_all_surfaces_seconds"] = measured
        receipt["projection"]["projected_full_runtime_seconds"] = projected
    else:  # pragma: no cover - the parameter table is closed above.
        raise AssertionError(case)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(error, match=message):
        validate_runtime_benchmark_receipt(
            receipt_path,
            contract_path=contract_path,
            device=torch.device("cpu"),
        )
