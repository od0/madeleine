from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.schema import KEY_ORDER
from goldenberry.verify import (
    consistency_score,
    shift_sweep,
    swap_score,
)


def _table(frame_pos: np.ndarray, values: np.ndarray, *, probs: bool) -> pa.Table:
    columns: dict[str, object] = {"frame_pos": frame_pos}
    for index, key in enumerate(KEY_ORDER):
        columns[key] = values[:, index]
    schema = pa.schema(
        [
            pa.field("frame_pos", pa.int64()),
            *(
                pa.field(key, pa.float64() if probs else pa.bool_())
                for key in KEY_ORDER
            ),
        ]
    )
    return pa.table(columns, schema=schema)


def _scripted_labels(frame_pos: np.ndarray, variant: int = 0) -> np.ndarray:
    periods = np.array([7, 9, 11, 13, 15, 17, 19], dtype=np.int64)
    widths = np.array([3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    multipliers = np.array([1, 2, 3, 5, 7, 4, 6], dtype=np.int64)
    phases = np.arange(len(KEY_ORDER), dtype=np.int64) * (variant + 1)
    residues = (
        frame_pos[:, None] * multipliers[None, :]
        + phases[None, :]
        + variant * 5
    ) % periods[None, :]
    return residues < widths[None, :]


def _synthetic_tables() -> tuple[pa.Table, pa.Table, pa.Table]:
    frame_pos = np.arange(256, dtype=np.int64)
    labels = _scripted_labels(frame_pos)
    other_labels = _scripted_labels(frame_pos, variant=3)
    rng = np.random.default_rng(20250308)
    noise = rng.normal(0.0, 0.005, size=labels.shape)
    probabilities = np.clip(labels * 0.9 + 0.05 + noise, 0.0, 1.0)
    return (
        _table(frame_pos, probabilities.astype(np.float64), probs=True),
        _table(frame_pos, labels.astype(np.bool_), probs=False),
        _table(frame_pos, other_labels.astype(np.bool_), probs=False),
    )


def test_aligned_score_is_shift_sweep_minimum() -> None:
    probs, claimed, _ = _synthetic_tables()

    aligned = consistency_score(probs, claimed)
    sweep = shift_sweep(probs, claimed)

    assert min(sweep, key=sweep.get) == 0
    assert sweep[0] == pytest.approx(aligned["nll_mean"])
    assert sweep[-8] > sweep[0] * 1.5
    assert sweep[8] > sweep[0] * 1.5


def test_swapped_claims_score_much_worse() -> None:
    probs, claimed, claimed_other = _synthetic_tables()

    aligned = consistency_score(probs, claimed)
    swapped = swap_score(probs, claimed_other)

    assert swapped["nll_mean"] > aligned["nll_mean"] * 10


def test_hand_computed_three_frame_bce() -> None:
    frame_pos = np.arange(3, dtype=np.int64)
    one_key_probs = np.array([0.2, 0.7, 0.9], dtype=np.float64)
    one_key_claimed = np.array([False, True, False], dtype=np.bool_)
    probabilities = np.repeat(one_key_probs[:, None], len(KEY_ORDER), axis=1)
    claimed = np.repeat(one_key_claimed[:, None], len(KEY_ORDER), axis=1)

    score = consistency_score(
        (frame_pos, probabilities),
        (frame_pos.copy(), claimed),
    )
    expected = float(
        (-np.log(1.0 - 0.2) - np.log(0.7) - np.log(1.0 - 0.9)) / 3
    )

    assert score["nll_per_key"]["left"] == pytest.approx(expected, abs=1e-9)
    assert score["nll_mean"] == pytest.approx(expected, abs=1e-9)


def test_frame_pos_mismatches_are_counted_and_empty_overlap_raises() -> None:
    prob_pos = np.array([0, 1, 2], dtype=np.int64)
    claimed_pos = np.array([1, 2, 3, 8], dtype=np.int64)
    probabilities = np.full((3, len(KEY_ORDER)), 0.5, dtype=np.float64)
    labels = np.zeros((4, len(KEY_ORDER)), dtype=np.bool_)

    score = consistency_score(
        (prob_pos, probabilities),
        (claimed_pos, labels),
    )

    assert score["n_frames"] == 2
    assert score["n_unmatched"] == 3

    with pytest.raises(ValueError, match="no overlapping frames"):
        consistency_score(
            (prob_pos, probabilities),
            (
                np.array([10], dtype=np.int64),
                np.zeros((1, len(KEY_ORDER)), dtype=np.bool_),
            ),
        )


def test_cli_round_trip(tmp_path) -> None:
    probs, claimed, _ = _synthetic_tables()
    probs_path = tmp_path / "p.parquet"
    claimed_path = tmp_path / "c.parquet"
    report_path = tmp_path / "report.json"
    pq.write_table(probs, probs_path)
    pq.write_table(claimed, claimed_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "goldenberry.verify",
            "--probs",
            str(probs_path),
            "--claimed",
            str(claimed_path),
            "--out",
            str(report_path),
            "--max-shift",
            "2",
        ],
        check=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) == {"score", "shift_sweep"}
    assert set(report["shift_sweep"]) == {"-2", "-1", "0", "1", "2"}
    assert report["score"]["n_frames"] == 256
