from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.build_dynamics_pretraining_inventory import (
    _eligible_windows,
    _required_npz_metadata,
    atomic_json,
    canonical_sha256,
    reject_forbidden_text,
)


def _write_metadata_shard(path: Path, session_id: str) -> None:
    # Deliberately include poison values in members the inventory must never
    # index.  The metadata-only validator should not care about their values.
    np.savez_compressed(
        path,
        frames=np.full((12, 2, 2, 3), 255, dtype=np.uint8),
        keys=np.full((12, 7), 255, dtype=np.uint8),
        engine_frame_idx=np.arange(100, 112, dtype=np.int64),
        input_active=np.asarray(
            [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=np.uint8
        ),
        session_id=np.asarray(session_id),
    )


def test_forbidden_identities_fail_before_inventory_publication() -> None:
    for value in (
        "rec_20260727_220000_test",
        "/tmp/untouched/session.npz",
        "/tmp/B1/features.npz",
        "y4nQHqYSObI__r000",
        "/tmp/val_b/data.npz",
    ):
        with pytest.raises(ValueError):
            reject_forbidden_text(value)


def test_metadata_reader_aligns_without_using_supervision_values(tmp_path: Path) -> None:
    path = tmp_path / "train.npz"
    _write_metadata_shard(path, "train-session")
    engine, active, active_frames = _required_npz_metadata(
        path,
        expected_session="train-session",
        expected_frames=12,
        require_all_active=False,
        require_contiguous=True,
    )
    assert engine.tolist() == list(range(100, 112))
    assert active_frames == 11
    assert _eligible_windows(engine, active, max_horizon=4) == 1


def test_eligible_windows_never_cross_inactive_boundaries() -> None:
    engine = np.arange(30, dtype=np.int64)
    active = np.asarray(
        [1] * 9 + [0] + [1] * 14 + [0] + [1] * 5, dtype=np.uint8
    )
    # For h=4, each active run of length L contributes max(0, L-5).
    assert _eligible_windows(engine, active, max_horizon=4) == 4 + 9 + 0


def test_eligible_windows_never_cross_engine_gaps() -> None:
    engine = np.asarray([0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25], dtype=np.int64)
    active = np.ones(len(engine), dtype=np.uint8)
    assert _eligible_windows(engine, active, max_horizon=4) == 1 + 1


def test_atomic_json_refuses_overwrite_and_hash_is_canonical(tmp_path: Path) -> None:
    path = tmp_path / "inventory.json"
    payload = {"b": 2, "a": 1}
    atomic_json(path, payload)
    assert json.loads(path.read_text()) == payload
    assert canonical_sha256(payload) == canonical_sha256({"a": 1, "b": 2})
    with pytest.raises(FileExistsError):
        atomic_json(path, payload)
