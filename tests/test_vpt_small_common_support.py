from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.eval_gru_vpt_common_support import select_common_rows


def write_support(path: Path, rows: np.ndarray, *, truth: np.ndarray | None = None) -> None:
    if truth is None:
        truth = np.zeros((len(rows), 7), dtype=np.uint8)
    np.savez_compressed(
        path,
        y_true=truth,
        y_prob=np.zeros((len(rows), 7), dtype=np.float32),
        input_active=np.ones(len(rows), dtype=np.uint8),
        session_lengths=np.asarray([len(rows)], dtype=np.int64),
        session_ids=np.asarray(["session__run000"]),
        source_row_index=rows,
        source_engine_frame_idx=rows + 100,
    )


def native_rows() -> dict[str, np.ndarray]:
    rows = np.asarray([10, 11, 12, 13], dtype=np.int64)
    return {
        "source_row": rows,
        "probability": np.arange(28, dtype=np.float32).reshape(4, 7) / 28,
        "truth": np.zeros((4, 7), dtype=np.uint8),
        "active": np.ones(4, dtype=np.uint8),
        "engine_idx": rows + 100,
    }


def test_common_support_preserves_vpt_order_and_exact_rows(tmp_path: Path) -> None:
    support = tmp_path / "support.npz"
    write_support(support, np.asarray([13, 10, 12], dtype=np.int64))
    selected = select_common_rows(native_rows(), support)
    assert selected["source_row"].tolist() == [13, 10, 12]
    assert np.array_equal(
        selected["probability"], native_rows()["probability"][[3, 0, 2]]
    )
    assert selected["session_lengths"].tolist() == [3]


def test_common_support_fails_closed_on_missing_or_duplicate_rows(tmp_path: Path) -> None:
    missing = tmp_path / "missing.npz"
    write_support(missing, np.asarray([10, 99], dtype=np.int64))
    with pytest.raises(RuntimeError, match="cannot predict"):
        select_common_rows(native_rows(), missing)

    duplicate = tmp_path / "duplicate.npz"
    write_support(duplicate, np.asarray([10, 10], dtype=np.int64))
    with pytest.raises(RuntimeError, match="duplicate"):
        select_common_rows(native_rows(), duplicate)


def test_common_support_fails_closed_on_label_identity_mismatch(tmp_path: Path) -> None:
    support = tmp_path / "mismatch.npz"
    truth = np.zeros((2, 7), dtype=np.uint8)
    truth[0, 0] = 1
    write_support(support, np.asarray([10, 11], dtype=np.int64), truth=truth)
    with pytest.raises(RuntimeError, match="truth differs"):
        select_common_rows(native_rows(), support)
