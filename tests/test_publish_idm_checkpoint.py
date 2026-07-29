from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments import publish_idm_checkpoint as publisher


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(tmp_path: Path) -> tuple[Path, bytes]:
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = b"checkpoint-bytes"
    (run / "model.pt").write_bytes(checkpoint)
    (run / "config.json").write_text('{"seed":0}\n')
    (run / "run_meta.json").write_text('{"data":"own-v3"}\n')
    return run, checkpoint


def test_publish_is_content_addressed_and_marker_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, checkpoint = _run(tmp_path)
    remote: dict[str, bytes] = {}
    operations: list[str] = []

    def inventory(prefix: str) -> list[str]:
        stem = prefix.rstrip("/") + "/"
        return sorted(key.removeprefix(stem) for key in remote if key.startswith(stem))

    def copy(local: Path, destination: str) -> None:
        operations.append(local.name)
        remote[destination] = local.read_bytes()

    def remote_hash(path: str) -> tuple[str, int]:
        value = remote[path]
        return _sha(value), len(value)

    monkeypatch.setattr(publisher, "remote_inventory", inventory)
    monkeypatch.setattr(publisher, "_copy_verified", copy)
    monkeypatch.setattr(publisher, "remote_sha256", remote_hash)
    result = publisher.publish(
        run=run,
        artifact_id="own-features-v3-32nc-s0",
        role="corrected own-v3 scratch seed 0",
        remote_root="r2:bucket/runs/idm/v1",
        expected_checkpoint_sha256=_sha(checkpoint),
    )

    assert operations == [
        "model.pt",
        "checkpoint-manifest.json",
        "checkpoint_complete.json",
    ]
    assert result["checkpoint_sha256"] == _sha(checkpoint)
    manifest = json.loads((run / "checkpoint-manifest.json").read_text())
    assert manifest["metadata_hashes"]["run_meta_sha256"] == publisher.sha256_file(
        run / "run_meta.json"
    )


def test_publish_refuses_hash_mismatch(tmp_path: Path) -> None:
    run, _ = _run(tmp_path)
    with pytest.raises(ValueError, match="checkpoint SHA-256 changed"):
        publisher.publish(
            run=run,
            artifact_id="own-features-v3-32nc-s0",
            role="corrected own-v3 scratch seed 0",
            remote_root="r2:bucket/runs/idm/v1",
            expected_checkpoint_sha256="0" * 64,
        )


def test_publish_refuses_existing_receipts(tmp_path: Path) -> None:
    run, checkpoint = _run(tmp_path)
    (run / "checkpoint-manifest.json").write_text("{}")
    with pytest.raises(FileExistsError, match="already exist"):
        publisher.publish(
            run=run,
            artifact_id="own-features-v3-32nc-s0",
            role="corrected own-v3 scratch seed 0",
            remote_root="r2:bucket/runs/idm/v1",
            expected_checkpoint_sha256=_sha(checkpoint),
        )
