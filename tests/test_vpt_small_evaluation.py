from __future__ import annotations

import numpy as np

from badeline.metrics import summarize
from experiments.eval_vpt_small import combine_phases, flatten_center_probabilities


def _part(rows: list[int], phase: int) -> dict[str, np.ndarray]:
    count = len(rows)
    truth = np.zeros((count, 7), dtype=np.uint8)
    truth[:, 4] = np.asarray(rows) % 2
    return {
        "source_row": np.asarray(rows, dtype=np.int64),
        "engine_idx": np.asarray(rows, dtype=np.int64),
        "continuity": np.zeros(count, dtype=np.int32),
        "truth": truth,
        "active": np.ones(count, dtype=np.uint8),
        "probability": np.full((count, 7), 0.1 + phase * 0.1, dtype=np.float32),
    }


def test_three_phase_interleave_restores_native_order_and_splits_gaps() -> None:
    manifest = {
        "records": [
            {"session_id": "s", "phase": 0},
            {"session_id": "s", "phase": 1},
            {"session_id": "s", "phase": 2},
        ]
    }
    inferred = [
        _part([0, 3, 6, 12], 0),
        _part([1, 4, 7, 13], 1),
        _part([2, 5, 8, 14], 2),
    ]
    combined = combine_phases(manifest, None, inferred)  # type: ignore[arg-type]
    assert np.array_equal(combined["source_row"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14])
    assert np.array_equal(combined["stream_lengths"], [9, 3])
    assert combined["truth"].shape == combined["probability"].shape == (12, 7)


def test_fixed_only_summary_does_not_fit_or_emit_oracle_thresholds() -> None:
    truth = np.zeros((12, 7), dtype=np.uint8)
    truth[3:6, 4] = 1
    probability = np.full((12, 7), 0.1, dtype=np.float32)
    probability[3:6, 4] = 0.9
    report = summarize(
        truth,
        probability,
        boundaries=[12],
        fixed_transition_thresholds={key: 0.5 for key in (
            "left", "right", "up", "down", "jump", "dash", "grab"
        )},
        include_oracle=False,
    )
    assert "transition_f1_oracle" not in report
    assert "transition_f1_oracle_collars" not in report
    assert report["transition_f1_at_0.5_collars"]["2"]["jump"]["event"]["f1"] == 1.0


def test_dense_center_probabilities_flatten_to_one_prediction_per_row() -> None:
    first = np.arange(2 * 64 * 7, dtype=np.float32).reshape(2, 64, 7)
    second = np.arange(64 * 7, dtype=np.float32).reshape(1, 64, 7)
    flattened = flatten_center_probabilities([first, second])
    assert flattened.shape == (3 * 64, 7)
    assert np.array_equal(flattened[: 2 * 64], first.reshape(-1, 7))
    assert np.array_equal(flattened[2 * 64 :], second.reshape(-1, 7))
