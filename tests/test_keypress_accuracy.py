from pathlib import Path

import numpy as np
import pytest

from experiments.keypress_accuracy import score_sidecar


def _write_sidecar(path: Path) -> None:
    truth = np.asarray(
        [
            [0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    probability = np.asarray(
        [
            [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
            [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        y_true=truth,
        y_prob=probability,
        input_active=np.asarray([1, 0, 1], dtype=np.uint8),
    )


def test_score_sidecar_uses_active_rows_and_binary_argmax(tmp_path: Path) -> None:
    path = tmp_path / "preds.npz"
    _write_sidecar(path)

    report = score_sidecar(path)

    assert report["frames"] == 2
    assert report["binary_decisions"] == 14
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["key_state_micro_accuracy"] == pytest.approx(0.5)
    assert report["always_released_accuracy"] == pytest.approx(4 / 14)
    assert report["exact_seven_key_vector_accuracy"] == pytest.approx(0.5)
    assert report["joint_exact_match_accuracy"] == pytest.approx(0.5)


def test_baselines_use_same_gate_without_compressing_time(tmp_path: Path) -> None:
    path = tmp_path / "preds.npz"
    _write_sidecar(path)

    report = score_sidecar(path)

    assert report["always_released_key_state_micro_accuracy"] == pytest.approx(
        4 / 14
    )
    assert report["always_released_joint_exact_match_accuracy"] == 0.0
    # Active row 2 copies truth row 1, even though row 1 is outside the gate.
    # If gated rows had first been compressed, this would be 7/14 instead.
    assert report["persistence_key_state_micro_accuracy"] == pytest.approx(8 / 14)
    assert report["persistence_joint_exact_match_accuracy"] == 0.0


def test_score_sidecar_can_include_all_rows(tmp_path: Path) -> None:
    path = tmp_path / "preds.npz"
    _write_sidecar(path)

    report = score_sidecar(path, active_only=False)

    assert report["frames"] == 3
    assert report["accuracy"] == pytest.approx(2 / 3)


def test_persistence_resets_to_released_at_stream_boundaries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "preds.npz"
    truth = np.zeros((4, 7), dtype=np.uint8)
    truth[1, 0] = 1
    np.savez_compressed(
        path,
        y_true=truth,
        y_prob=np.zeros_like(truth, dtype=np.float32),
        input_active=np.ones(4, dtype=np.uint8),
        session_lengths=np.asarray([2, 2], dtype=np.int64),
        session_ids=np.asarray(["a", "b"]),
    )

    report = score_sidecar(path)

    assert report["streams"] == 2
    # Persistence predictions for key 0 are [0, 0, 0, 0].  In particular,
    # row 2 starts released rather than inheriting row 1 across the boundary.
    assert report["per_key_persistence_accuracy"]["left"] == pytest.approx(3 / 4)
    assert report["persistence_joint_exact_match_accuracy"] == pytest.approx(3 / 4)


@pytest.mark.parametrize(
    ("lengths", "message"),
    [
        (np.asarray([1, 1]), "must sum"),
        (np.asarray([3, 0]), "positive"),
        (np.asarray([[3]]), r"shape \[streams\]"),
        (np.asarray([1.5, 1.5]), "integers"),
    ],
)
def test_score_sidecar_validates_stream_lengths(
    tmp_path: Path, lengths: np.ndarray, message: str
) -> None:
    path = tmp_path / "preds.npz"
    np.savez_compressed(
        path,
        y_true=np.zeros((3, 7), dtype=np.uint8),
        y_prob=np.zeros((3, 7), dtype=np.float32),
        input_active=np.ones(3, dtype=np.uint8),
        session_lengths=lengths,
    )

    with pytest.raises(ValueError, match=message):
        score_sidecar(path)


def test_score_sidecar_validates_stream_id_count(tmp_path: Path) -> None:
    path = tmp_path / "preds.npz"
    np.savez_compressed(
        path,
        y_true=np.zeros((3, 7), dtype=np.uint8),
        y_prob=np.zeros((3, 7), dtype=np.float32),
        input_active=np.ones(3, dtype=np.uint8),
        session_lengths=np.asarray([1, 2], dtype=np.int64),
        session_ids=np.asarray(["only-one-id"]),
    )

    with pytest.raises(ValueError, match="one entry per stream"):
        score_sidecar(path)


def test_score_sidecar_rejects_missing_activity_gate(tmp_path: Path) -> None:
    path = tmp_path / "preds.npz"
    np.savez_compressed(
        path,
        y_true=np.zeros((2, 7), dtype=np.uint8),
        y_prob=np.zeros((2, 7), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="input_active is required"):
        score_sidecar(path)
