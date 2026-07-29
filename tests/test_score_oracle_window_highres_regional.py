from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.oracle_window_localization import HEAD_NAMES
from experiments.score_oracle_window_highres_regional import (
    _gate,
    _same_metadata,
    publish_score,
)


def _metrics(exact: float, within_2: float, nll: float) -> dict[str, object]:
    return {
        "macro_estimable": {
            "exact": exact,
            "within_1": within_2,
            "within_2": within_2,
            "nll": nll,
            "entropy": 1.0,
        },
        "per_head": {
            name: {"exact": exact, "within_2": within_2, "nll": nll}
            for name in HEAD_NAMES
        },
    }


def _config() -> dict[str, object]:
    rule = {
        "minimum_candidate_macro_exact": 0.125,
        "minimum_candidate_exact_ci_low": 0.0625,
        "minimum_macro_exact_delta": 0.03,
        "minimum_macro_exact_delta_ci_low": 0.0,
        "minimum_positive_estimable_heads": 7,
        "minimum_positive_distinct_keys": 4,
        "minimum_macro_within_2_delta": -0.01,
    }
    return {"decision_gate": {"seed_zero_primary": rule, "regional_attribution": rule}}


def test_metadata_must_match_every_arm_exactly() -> None:
    reference = {name: np.asarray([0]) for name in (
        "session_id", "run_index", "array_index", "engine_frame_idx", "head_index",
        "key_index", "event_type_index", "true_offset", "crop_start", "block_id",
    )}
    observed = {name: value.copy() for name, value in reference.items()}
    _same_metadata(reference, observed)
    observed["crop_start"][0] = 1
    with pytest.raises(ValueError, match="crop_start"):
        _same_metadata(reference, observed)


def test_primary_gate_requires_materiality_uncertainty_breadth_and_nll() -> None:
    estimable = [
        "left:onset", "right:onset", "up:onset", "jump:onset",
        "left:release", "right:release", "up:release", "jump:release",
    ]
    candidate = _metrics(0.16, 0.40, 2.0)
    control = _metrics(0.10, 0.39, 2.5)
    # Make all non-estimable heads irrelevant and every estimable head positive.
    result = _gate(
        config=_config(),
        manifest={"validation_block_count": 25},
        candidate=candidate,
        control=control,
        bootstrap={"conditional_macro_95": [0.13, 0.16, 0.19], "delta_macro_95": [0.02, 0.06, 0.10]},
        estimable=estimable,
        gate_name="seed_zero_primary",
    )
    assert result["passed"] is True
    failed = _gate(
        config=_config(),
        manifest={"validation_block_count": 25},
        candidate=_metrics(0.12, 0.40, 3.0),
        control=control,
        bootstrap={"conditional_macro_95": [0.05, 0.12, 0.18], "delta_macro_95": [-0.01, 0.02, 0.05]},
        estimable=estimable,
        gate_name="seed_zero_primary",
    )
    assert failed["passed"] is False
    assert failed["checks"]["minimum_candidate_macro_exact"] is False
    assert failed["checks"]["delta_ci_low_above_zero"] is False
    assert failed["checks"]["lower_nll"] is False


def test_publication_writes_marker_last_and_refuses_collision(tmp_path: Path) -> None:
    runs = {}
    for arm in ("h32_q", "h128_g", "h128_q"):
        run = tmp_path / arm
        run.mkdir()
        for name in ("run_receipt.json", "model.pt", "predictions.npz"):
            (run / name).write_bytes(name.encode())
        runs[arm] = run
    config = tmp_path / "config.json"
    cache = tmp_path / "cache.json"
    manifest = tmp_path / "manifest.json"
    for path in (config, cache, manifest):
        path.write_text("{}\n", encoding="utf-8")
    report = {
        "study_id": "test",
        "decision": {"status": "reject_study_h_primary_gate"},
    }
    out = tmp_path / "report.json"
    marker = tmp_path / "complete.json"
    publish_score(
        report=report,
        out=out,
        marker=marker,
        runs=runs,
        config_path=config,
        cache_receipt_path=cache,
        base_manifest_path=manifest,
    )
    assert out.is_file() and marker.is_file()
    with pytest.raises(ValueError, match="overwrite"):
        publish_score(
            report=report,
            out=out,
            marker=marker,
            runs=runs,
            config_path=config,
            cache_receipt_path=cache,
            base_manifest_path=manifest,
        )
