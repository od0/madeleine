import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from data.schema import KEY_ORDER
from experiments.calibrate_keypress import (
    Sidecar,
    _equal_mass_ece,
    _probability_diagnostics,
    _state_metrics,
    apply_affine_calibrators,
    calibrate_sidecar,
    fit_affine_key,
    load_roles,
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def test_affine_fit_recovers_monotone_logit_correction() -> None:
    rng = np.random.default_rng(20260727)
    raw_logits = rng.normal(size=50_000)
    raw_probability = _sigmoid(raw_logits)
    truth = rng.binomial(1, _sigmoid(0.45 * raw_logits - 0.8))

    result = fit_affine_key(truth, raw_probability)

    assert result["converged"] is True
    assert result["scale"] == pytest.approx(0.45, abs=0.03)
    assert result["bias"] == pytest.approx(-0.8, abs=0.03)
    assert result["final_unweighted_nll"] < result["initial_unweighted_nll"]
    expected_threshold = float(_sigmoid(np.asarray([0.8 / 0.45]))[0])
    assert result["equivalent_raw_probability_threshold"] == pytest.approx(
        expected_threshold, abs=0.03
    )


def test_affine_application_preserves_per_key_ranking_and_ap() -> None:
    probability = np.linspace(0.01, 0.99, 200).reshape(20, 10)[:, : len(KEY_ORDER)]
    truth = np.asarray(
        [[(row + column) % 3 == 0 for column in range(len(KEY_ORDER))] for row in range(20)]
    )
    parameters = {
        key: {"scale": 0.2 + column / 10, "bias": -0.5 + column / 20}
        for column, key in enumerate(KEY_ORDER)
    }

    calibrated = apply_affine_calibrators(probability, parameters)

    for column in range(len(KEY_ORDER)):
        assert np.array_equal(
            np.argsort(probability[:, column]), np.argsort(calibrated[:, column])
        )
        assert average_precision_score(
            truth[:, column], calibrated[:, column]
        ) == pytest.approx(
            average_precision_score(truth[:, column], probability[:, column])
        )


def _write_sidecar(
    path: Path,
    *,
    truth: np.ndarray,
    probability: np.ndarray,
    ids: list[str],
    lengths: list[int],
) -> None:
    np.savez_compressed(
        path,
        y_true=truth.astype(np.uint8),
        y_prob=probability.astype(np.float32),
        input_active=np.ones(len(truth), dtype=np.uint8),
        session_lengths=np.asarray(lengths, dtype=np.int64),
        session_ids=np.asarray(ids),
    )


def test_calibration_report_keeps_stream_roles_and_transfer_fit_separate(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(42)
    lengths = [500, 500, 500, 500]
    ids = ["cal-a", "cal-b", "eval-a", "eval-b"]
    raw_logits = rng.normal(size=(sum(lengths), len(KEY_ORDER)))
    probability = _sigmoid(raw_logits)
    truth_probability = _sigmoid(0.55 * raw_logits - 0.75)
    truth = rng.binomial(1, truth_probability)
    fit_path = tmp_path / "fit.npz"
    _write_sidecar(
        fit_path,
        truth=truth,
        probability=probability,
        ids=ids,
        lengths=lengths,
    )

    # Deliberately unrelated transfer labels.  They must be scored but never
    # become part of the calibration objective.
    transfer_truth = rng.binomial(1, 0.2, size=(600, len(KEY_ORDER)))
    transfer_probability = rng.uniform(0.01, 0.99, size=transfer_truth.shape)
    transfer_path = tmp_path / "transfer.npz"
    _write_sidecar(
        transfer_path,
        truth=transfer_truth,
        probability=transfer_probability,
        ids=["transfer-only"],
        lengths=[600],
    )

    roles_path = tmp_path / "roles.json"
    roles_path.write_text(
        json.dumps(
            {
                "surface": "synthetic development surface",
                "policy": "whole-stream split",
                "calibration_session_ids": ["cal-a", "cal-b"],
                "evaluation_session_ids": ["eval-a", "eval-b"],
            }
        )
    )

    report = calibrate_sidecar(
        fit_path,
        roles_path,
        transfers={"transfer": transfer_path},
    )

    assert report["fit"]["active_frames"] == 1000
    assert report["fit"]["session_ids"] == ["cal-a", "cal-b"]
    assert report["provenance"]["evaluation_session_ids"] == ["eval-a", "eval-b"]
    assert report["provenance"]["b1_labels_used_for_fit"] is False
    assert report["provenance"]["untouched_test_used_for_fit_or_scoring"] is False
    evaluation = report["surfaces"]["mapped_y4n_disjoint_evaluation"]
    assert evaluation["session_ids"] == ["eval-a", "eval-b"]
    assert evaluation["labels_used_for_fit"] is False
    assert evaluation["average_precision_invariance"][
        "maximum_absolute_difference"
    ] < 1e-12
    assert report["surfaces"]["transfer"]["labels_used_for_fit"] is False
    for key in KEY_ORDER:
        assert report["fit"]["parameters"][key]["scale"] > 0


def test_role_loader_rejects_overlap(tmp_path: Path) -> None:
    path = tmp_path / "roles.json"
    path.write_text(
        json.dumps(
            {
                "calibration_session_ids": ["same"],
                "evaluation_session_ids": ["same"],
            }
        )
    )

    with pytest.raises(ValueError, match="overlap"):
        load_roles(path)


def test_equal_mass_ece_exposes_balanced_bin_counts() -> None:
    probability = np.linspace(0.001, 0.999, 103)
    truth = probability >= 0.6

    report = _equal_mass_ece(truth, probability, bin_count=15)

    assert report["kind"] == "equal_mass"
    assert report["requested_bin_count"] == 15
    assert report["nonempty_bin_count"] == 15
    assert sum(report["bin_counts"]) == 103
    assert max(report["bin_counts"]) - min(report["bin_counts"]) <= 1
    assert [entry["count"] for entry in report["bins"]] == report["bin_counts"]
    assert 0.0 <= report["ece"] <= 1.0


def test_state_metrics_include_prior_skill_and_segment_bounded_events(
    tmp_path: Path,
) -> None:
    pattern_a = np.asarray([0, 1, 1, 0, 0, 0], dtype=bool)
    pattern_b = np.asarray([0, 0, 1, 1, 0, 0], dtype=bool)
    one_key = np.concatenate((pattern_a, pattern_b))
    truth = np.repeat(one_key[:, None], len(KEY_ORDER), axis=1)
    probability = np.where(truth, 0.9, 0.1)
    sidecar = Sidecar(
        path=tmp_path / "synthetic.npz",
        truth=truth,
        probability=probability,
        active=np.ones(len(truth), dtype=bool),
        session_lengths=np.asarray([6, 6], dtype=np.int64),
        session_ids=np.asarray(["a", "b"]),
    )
    prior_rates = {key: 0.3 for key in KEY_ORDER}

    report = _state_metrics(
        sidecar,
        probability,
        prior_rates=prior_rates,
    )

    assert report["transition_event_f1"]["model"]["exact"][
        "macro_onset_plus_release_f1"
    ] == pytest.approx(1.0)
    assert report["transition_event_f1"]["persistence_baseline"]["exact"][
        "macro_onset_plus_release_f1"
    ] == pytest.approx(0.0)
    assert report["transition_event_f1"]["persistence_baseline"][
        "plus_or_minus_2_frames"
    ]["macro_onset_plus_release_f1"] == pytest.approx(1.0)
    expected_prior_brier = float(np.mean((0.3 - truth) ** 2))
    assert report["calibration_prior_rate_baseline"]["brier_score"] == pytest.approx(
        expected_prior_brier
    )
    assert report["probability_skill_scores"]["brier"] > 0
    assert report["equal_mass_expected_calibration_error"]["bin_count"] == 15
    assert sum(
        report["equal_mass_expected_calibration_error"]["per_key"]["left"][
            "bin_counts"
        ]
    ) == len(truth)


def test_probability_diagnostics_report_clipping_and_unreachable_dash(
    tmp_path: Path,
) -> None:
    raw = np.full((4, len(KEY_ORDER)), 0.5, dtype=np.float64)
    raw[0, 0] = 0.0
    raw[1, 0] = 1.0
    sidecar = Sidecar(
        path=tmp_path / "synthetic.npz",
        truth=np.zeros_like(raw, dtype=bool),
        probability=raw,
        active=np.ones(4, dtype=bool),
        session_lengths=np.asarray([4], dtype=np.int64),
        session_ids=np.asarray(["stream"]),
    )
    parameters = {
        key: {
            "scale": 1.0,
            "bias": 0.0,
            "calibrated_half_reachable_with_clipped_float64_logit": True,
            "required_raw_logit_at_calibrated_half": 0.0,
        }
        for key in KEY_ORDER
    }
    parameters["dash"] = {
        "scale": 0.01,
        "bias": -1.0,
        "calibrated_half_reachable_with_clipped_float64_logit": False,
        "required_raw_logit_at_calibrated_half": 100.0,
    }
    calibrated = apply_affine_calibrators(raw, parameters)

    report = _probability_diagnostics(sidecar, calibrated, parameters)

    assert report["per_key"]["left"]["raw_exact_zero_count"] == 1
    assert report["per_key"]["left"]["raw_exact_one_count"] == 1
    assert report["per_key"]["left"]["raw_low_logit_clip_count"] == 1
    assert report["per_key"]["left"]["raw_high_logit_clip_count"] == 1
    assert [entry["key"] for entry in report["structurally_unreachable_half_thresholds"]] == [
        "dash"
    ]


def test_tolerant_calibration_events_do_not_match_across_streams(
    tmp_path: Path,
) -> None:
    # Truth onsets at the final frame of stream A. Prediction onsets one frame
    # later at the first frame of stream B. They are close in concatenated
    # coordinates but belong to different streams and must not match.
    truth_key = np.asarray([0, 0, 1, 0, 0, 0], dtype=bool)
    predicted_key = np.asarray([0, 0, 0, 1, 0, 0], dtype=bool)
    truth = np.repeat(truth_key[:, None], len(KEY_ORDER), axis=1)
    probability = np.repeat(
        np.where(predicted_key, 0.9, 0.1)[:, None],
        len(KEY_ORDER),
        axis=1,
    )
    sidecar = Sidecar(
        path=tmp_path / "boundary.npz",
        truth=truth,
        probability=probability,
        active=np.ones(len(truth), dtype=bool),
        session_lengths=np.asarray([3, 3], dtype=np.int64),
        session_ids=np.asarray(["a", "b"]),
    )

    report = _state_metrics(
        sidecar,
        probability,
        prior_rates={key: 0.2 for key in KEY_ORDER},
    )

    onset = report["transition_event_f1"]["model"]["plus_or_minus_2_frames"]
    assert onset["macro_onset_f1"] == pytest.approx(0.0)
    assert onset["per_key"]["left"]["onset"]["n_matched"] == 0
