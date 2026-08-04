from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.verify_vpt_object_inventory import verify


def test_verify_accepts_exact_inventory(tmp_path: Path) -> None:
    payload = b"pixel-data"
    object_path = tmp_path / "session" / "frames.npy"
    object_path.parent.mkdir()
    object_path.write_bytes(payload)
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        json.dumps(
            {
                "relative_path": "session/frames.npy",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    result = verify(inventory, tmp_path)
    assert result["status"] == "pass"
    assert result["objects"] == 1
    assert result["bytes"] == len(payload)


def test_verify_rejects_hash_mismatch(tmp_path: Path) -> None:
    object_path = tmp_path / "frames.npy"
    object_path.write_bytes(b"wrong")
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        json.dumps(
            {
                "relative_path": "frames.npy",
                "bytes": 5,
                "sha256": "0" * 64,
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify(inventory, tmp_path)
