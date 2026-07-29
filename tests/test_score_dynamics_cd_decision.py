from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data.schema import KEY_ORDER
import experiments.score_dynamics_cd_decision as score


def _metrics(*, exact: float, plus2: float, ap: float) -> dict:
    return {
        "events_fixed_0_5": {
            "exact": {"macro_combined_f1": exact},
            "plus_minus_2": {"macro_combined_f1": plus2},
        },
        "macro_ap": ap,
    }


def test_promotion_decision_applies_all_frozen_gates() -> None:
    result = score.promotion_decision(
        _metrics(exact=0.10, plus2=0.20, ap=0.30),
        _metrics(exact=0.098, plus2=0.21, ap=0.295),
    )
    assert result["D_replication_recommended"] is True
    assert all(result["gates"].values())

    result = score.promotion_decision(
        _metrics(exact=0.10, plus2=0.20, ap=0.30),
        _metrics(exact=0.0979, plus2=0.25, ap=0.40),
    )
    assert result["D_replication_recommended"] is False
    assert result["gates"]["exact_loses_at_most_0_002"] is False


def _write_sidecar(
    path: Path,
    *,
    truth: np.ndarray,
    ids: list[str],
    lengths: list[int],
) -> None:
    np.savez_compressed(
        path,
        y_true=truth.astype(np.uint8),
        y_prob=np.full(truth.shape, 0.25, dtype=np.float32),
        input_active=np.ones(len(truth), dtype=np.uint8),
        session_lengths=np.asarray(lengths, dtype=np.int64),
        session_ids=np.asarray(ids),
    )


def test_later8_loader_rejects_superset_for_cd_and_derives_historical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact_ids = ["held__r008__stream000", "held__r009__stream000"]
    exact_lengths = [3, 2]
    later_truth = np.zeros((5, len(KEY_ORDER)), dtype=np.uint8)
    later_truth[:, 0] = [0, 1, 1, 0, 1]
    monkeypatch.setattr(score, "Y4N_STREAM_IDS", exact_ids)
    monkeypatch.setattr(score, "Y4N_STREAM_LENGTHS", exact_lengths)
    monkeypatch.setattr(score, "Y4N_FRAMES", 5)
    monkeypatch.setattr(
        score, "Y4N_TRUTH_SHA256", score._canonical_array_sha256(later_truth)
    )

    exact = tmp_path / "exact.npz"
    _write_sidecar(exact, truth=later_truth, ids=exact_ids, lengths=exact_lengths)
    loaded = score.load_later8_sidecar(exact, permit_superset=False)
    assert loaded["stream_ids"] == exact_ids
    assert loaded["support"]["rows"] == 5

    prefix = np.ones((1, len(KEY_ORDER)), dtype=np.uint8)
    superset = tmp_path / "all.npz"
    _write_sidecar(
        superset,
        truth=np.concatenate([prefix, later_truth]),
        ids=["held__r000__stream000", *exact_ids],
        lengths=[1, *exact_lengths],
    )
    with pytest.raises(ValueError, match="not exact later-eight"):
        score.load_later8_sidecar(superset, permit_superset=False)
    derived = score.load_later8_sidecar(superset, permit_superset=True)
    np.testing.assert_array_equal(derived["truth"], later_truth)


def test_promotion_decision_refuses_missing_or_nonfinite_metric() -> None:
    with pytest.raises(ValueError, match="missing decision metric"):
        score.promotion_decision({}, {})
    with pytest.raises(ValueError, match="non-finite decision metric"):
        score.promotion_decision(
            _metrics(exact=0.1, plus2=0.2, ap=0.3),
            _metrics(exact=0.1, plus2=float("nan"), ap=0.3),
        )


def test_contract_validation_binds_authorized_downstream_rule(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": score.CONTRACT_SCHEMA_VERSION,
                "study_id": score.STUDY_ID,
                "status": (
                    "owner_authorized_post_phase0_exploratory_override_before_optimizer_step_one"
                ),
                "downstream": {
                    "steps": 20_458,
                    "seed": 0,
                    "checkpoint": "final only",
                    "evaluation": "mapped y4n later-eight on identical support",
                    "threshold": 0.5,
                },
                "promotion": {
                    "D_vs_C_rule": (
                        "D plus-or-minus-2 event F1 improves by at least 0.010 "
                        "absolute, exact event F1 loses at most 0.002, and macro "
                        "AP loses at most 0.005"
                    )
                },
            }
        )
    )
    assert score._validate_contract(path)["sha256"] == score.sha256_file(path)
    payload = json.loads(path.read_text())
    payload["downstream"]["threshold"] = 0.4
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="threshold"):
        score._validate_contract(path)
