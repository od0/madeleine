import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.eval_vpt_paper_idm_xla import (
    CHECKPOINT_SCHEMA,
    RECEIPT_SCHEMA,
    combine_phases,
    restrict_to_reference,
    validate_checkpoint_receipt,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_receipt_binds_all_files(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_epoch_03"
    (checkpoint / "state").mkdir(parents=True)
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA,
        "completed_epoch": True,
        "epoch": 3,
        "optimizer_step": 351,
    }
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    (checkpoint / "rng.pt").write_bytes(b"rng")
    (checkpoint / "state" / ".metadata").write_bytes(b"metadata")
    (checkpoint / "state" / "__0_0.distcp").write_bytes(b"state")
    objects = []
    for relative in ("manifest.json", "rng.pt", "state/.metadata", "state/__0_0.distcp"):
        path = checkpoint / relative
        objects.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha(path)})
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": RECEIPT_SCHEMA,
                "checkpoint": checkpoint.name,
                "objects": objects,
                "total_bytes": sum(item["bytes"] for item in objects),
            }
        )
    )
    loaded, _ = validate_checkpoint_receipt(checkpoint, receipt_path)
    assert loaded["epoch"] == 3
    (checkpoint / "state" / "__0_0.distcp").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="byte count mismatch"):
        validate_checkpoint_receipt(checkpoint, receipt_path)


def test_three_phase_reconstruction_and_reference_restriction(tmp_path: Path) -> None:
    manifest = {
        "records": [
            {"session_id": "session", "phase": 0},
            {"session_id": "session", "phase": 1},
            {"session_id": "session", "phase": 2},
        ]
    }
    inferred = []
    for phase in range(3):
        rows = np.asarray([phase, phase + 3], dtype=np.int64)
        inferred.append(
            {
                "source_row": rows,
                "engine_idx": rows + 100,
                "continuity": np.zeros(2, dtype=np.int32),
                "truth": np.tile(np.asarray([[phase % 2] * 7], dtype=np.uint8), (2, 1)),
                "active": np.ones(2, dtype=np.uint8),
                "probability": np.full((2, 7), phase / 10, dtype=np.float32),
            }
        )
    combined = combine_phases(manifest, inferred)
    assert combined["source_row"].tolist() == [0, 1, 2, 3, 4, 5]
    reference = tmp_path / "reference.npz"
    np.savez_compressed(
        reference,
        y_true=combined["truth"],
        input_active=combined["active"],
        source_row_index=combined["source_row"],
        source_engine_frame_idx=combined["engine_idx"],
        session_lengths=np.asarray([6], dtype=np.int64),
        session_ids=np.asarray(["session__run000__sub000"]),
    )
    restricted = restrict_to_reference(combined, reference)
    assert restricted["probability"].shape == (6, 7)
    assert np.array_equal(restricted["source_row"], np.arange(6))
