"""it checks that a log is consistent with the video; it does not detect whether a human played. Record authenticity, not player humanity."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.schema import KEY_ORDER


EPS = 1e-6
ArrayPair: TypeAlias = tuple[np.ndarray, np.ndarray]
ResidualInput: TypeAlias = pa.Table | ArrayPair


def _table_arrays(
    table: pa.Table, *, probability: bool
) -> tuple[np.ndarray, np.ndarray]:
    expected_names = ["frame_pos", *KEY_ORDER]
    if table.column_names != expected_names:
        raise ValueError(f"expected columns {expected_names}, got {table.column_names}")

    expected_value_type = pa.float64() if probability else pa.bool_()
    expected_types = [pa.int64(), *([expected_value_type] * len(KEY_ORDER))]
    actual_types = [field.type for field in table.schema]
    if actual_types != expected_types:
        raise TypeError(f"expected column types {expected_types}, got {actual_types}")
    if any(column.null_count for column in table.columns):
        raise ValueError("residual inputs must not contain null values")

    frame_pos = table.column("frame_pos").to_numpy(zero_copy_only=False)
    values = np.column_stack(
        [
            table.column(key).to_numpy(zero_copy_only=False)
            for key in KEY_ORDER
        ]
    )
    return frame_pos, values


def _pair_arrays(
    pair: ArrayPair, *, probability: bool
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise TypeError("NumPy input must be a (frame_pos, values) tuple")

    frame_pos, values = pair
    if not isinstance(frame_pos, np.ndarray) or not isinstance(values, np.ndarray):
        raise TypeError("NumPy input pair members must be ndarrays")
    if frame_pos.dtype != np.dtype(np.int64):
        raise TypeError(f"frame_pos must have dtype int64, got {frame_pos.dtype}")

    expected_dtype = np.dtype(np.float64 if probability else np.bool_)
    if values.dtype != expected_dtype:
        kind = "probability" if probability else "claimed-label"
        raise TypeError(
            f"{kind} values must have dtype {expected_dtype}, got {values.dtype}"
        )
    if frame_pos.ndim != 1:
        raise ValueError("frame_pos must be one-dimensional")
    expected_shape = (frame_pos.size, len(KEY_ORDER))
    if values.shape != expected_shape:
        raise ValueError(f"values must have shape {expected_shape}, got {values.shape}")
    return frame_pos, values


def _as_arrays(
    data: ResidualInput, *, probability: bool
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(data, pa.Table):
        frame_pos, values = _table_arrays(data, probability=probability)
    else:
        frame_pos, values = _pair_arrays(data, probability=probability)

    if np.unique(frame_pos).size != frame_pos.size:
        raise ValueError("frame_pos values must be unique")
    if probability and (
        not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return frame_pos, values


def _score_arrays(
    prob_pos: np.ndarray,
    prob_values: np.ndarray,
    claimed_pos: np.ndarray,
    claimed_values: np.ndarray,
) -> dict[str, object]:
    _, prob_indices, claimed_indices = np.intersect1d(
        prob_pos,
        claimed_pos,
        assume_unique=True,
        return_indices=True,
    )
    n_frames = int(prob_indices.size)
    n_unmatched = int(prob_pos.size + claimed_pos.size - 2 * n_frames)
    if n_frames == 0:
        raise ValueError("probability and claimed-label inputs have no overlapping frames")

    clipped = np.clip(prob_values[prob_indices], EPS, 1.0 - EPS)
    labels = claimed_values[claimed_indices]
    losses = -(
        labels * np.log(clipped) + (~labels) * np.log1p(-clipped)
    )
    means = losses.mean(axis=0)
    return {
        "nll_per_key": {
            key: float(means[index]) for index, key in enumerate(KEY_ORDER)
        },
        "nll_mean": float(means.mean()),
        "n_frames": n_frames,
        "n_unmatched": n_unmatched,
    }


def consistency_score(
    probs: ResidualInput, claimed: ResidualInput
) -> dict[str, object]:
    """Compute per-key and aggregate binary cross-entropy on matched frames."""

    prob_pos, prob_values = _as_arrays(probs, probability=True)
    claimed_pos, claimed_values = _as_arrays(claimed, probability=False)
    return _score_arrays(prob_pos, prob_values, claimed_pos, claimed_values)


def shift_sweep(
    probs: ResidualInput, claimed: ResidualInput, max_shift: int = 8
) -> dict[int, float]:
    """Score claimed labels after shifting their frame positions by each offset."""

    if isinstance(max_shift, bool) or not isinstance(max_shift, int):
        raise TypeError("max_shift must be an integer")
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")

    prob_pos, prob_values = _as_arrays(probs, probability=True)
    claimed_pos, claimed_values = _as_arrays(claimed, probability=False)
    limits = np.iinfo(np.int64)
    if claimed_pos.size and (
        int(claimed_pos.min()) < limits.min + max_shift
        or int(claimed_pos.max()) > limits.max - max_shift
    ):
        raise ValueError("shift would overflow frame_pos int64 values")

    scores: dict[int, float] = {}
    for shift in range(-max_shift, max_shift + 1):
        score = _score_arrays(
            prob_pos,
            prob_values,
            claimed_pos + shift,
            claimed_values,
        )
        scores[shift] = float(score["nll_mean"])
    return scores


swap_score = consistency_score


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score claimed labels against probabilities")
    parser.add_argument("--probs", required=True, type=Path)
    parser.add_argument("--claimed", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-shift", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    probs = pq.read_table(args.probs)
    claimed = pq.read_table(args.claimed)
    report = {
        "score": consistency_score(probs, claimed),
        "shift_sweep": shift_sweep(probs, claimed, max_shift=args.max_shift),
    }
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
