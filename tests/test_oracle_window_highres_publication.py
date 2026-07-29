from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


REGISTRY_PATH = Path(
    "results/idm/checkpoint-index-oracle-window-highres-s0-20260728.json"
)
BACKUP_PATH = Path(
    "results/idm/oracle_window_highres_s0_checkpoint_backup_validation.json"
)
CONTRACT_SHA256 = (
    "7144e68f65acb75a7ae5712330d162f20e16f0fa9bd9440c7f58f67e7962e75f"
)
EXPECTED_FILES = {
    "checkpoint-manifest.json",
    "checkpoint_complete.json",
    "config.json",
    "r2_publication.json",
    "run_meta.json",
}

pytestmark = pytest.mark.requires_private_artifacts(REGISTRY_PATH, BACKUP_PATH)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_study_h_checkpoint_publications_are_content_bound() -> None:
    registry = _json(REGISTRY_PATH)
    backup = _json(BACKUP_PATH)

    assert registry["format_version"] == "madeleine.idm-checkpoint-registry.v1"
    assert registry["contract_sha256"] == CONTRACT_SHA256
    assert registry["checkpoint_count"] == 3
    assert registry["checkpoint_bytes"] == 38_687_253
    assert backup["status"] == "complete"
    assert backup["checkpoint_count"] == registry["checkpoint_count"]
    assert backup["checkpoint_bytes"] == registry["checkpoint_bytes"]
    assert backup["remote_object_count"] == 9
    assert all(backup["checks"].values())

    registry_by_id = {row["artifact_id"]: row for row in registry["records"]}
    backup_by_id = {row["artifact_id"]: row for row in backup["records"]}
    assert registry_by_id.keys() == backup_by_id.keys()
    assert len({row["checkpoint_sha256"] for row in registry["records"]}) == 3

    for artifact_id, record in registry_by_id.items():
        stem = artifact_id.removeprefix("oracle-window-highres-").removesuffix(
            "-s0"
        )
        tracked = Path(
            f"results/idm/oracle_window_highres_{stem.replace('-', '_')}_s0"
        )
        assert {path.name for path in tracked.iterdir()} == EXPECTED_FILES

        manifest_path = tracked / "checkpoint-manifest.json"
        completion_path = tracked / "checkpoint_complete.json"
        config_path = tracked / "config.json"
        run_meta_path = tracked / "run_meta.json"
        publication_path = tracked / "r2_publication.json"

        manifest = _json(manifest_path)
        completion = _json(completion_path)
        publication = _json(publication_path)
        backed_up = backup_by_id[artifact_id]
        objects = {row["name"]: row for row in backed_up["objects"]}

        assert _sha256(config_path) == CONTRACT_SHA256
        assert manifest["artifact_id"] == artifact_id
        assert manifest["role"] == record["role"]
        assert manifest["metadata_hashes"] == {
            "config_sha256": _sha256(config_path),
            "run_meta_sha256": _sha256(run_meta_path),
        }
        assert manifest["checkpoint"] == {
            "bytes": record["checkpoint_bytes"],
            "filename": "model.pt",
            "sha256": record["checkpoint_sha256"],
        }

        assert record["manifest_sha256"] == _sha256(manifest_path)
        assert record["completion_sha256"] == _sha256(completion_path)
        assert completion["manifest_sha256"] == record["manifest_sha256"]
        assert completion["checkpoint_sha256"] == record["checkpoint_sha256"]
        assert completion["checkpoint_bytes"] == record["checkpoint_bytes"]
        assert completion["payload_object_count"] == 2

        assert publication["artifact_id"] == artifact_id
        assert publication["object_prefix"] == record["object_prefix"]
        assert publication["manifest_sha256"] == record["manifest_sha256"]
        assert publication["checkpoint_sha256"] == record["checkpoint_sha256"]
        assert publication["checkpoint_bytes"] == record["checkpoint_bytes"]

        assert backed_up["remote_prefix"] == record["object_prefix"]
        assert backed_up["exact_remote_inventory"] == [
            "checkpoint-manifest.json",
            "checkpoint_complete.json",
            "model.pt",
        ]
        assert objects["model.pt"] == {
            "bytes": record["checkpoint_bytes"],
            "name": "model.pt",
            "sha256": record["checkpoint_sha256"],
        }
        assert objects["checkpoint-manifest.json"]["sha256"] == record[
            "manifest_sha256"
        ]
        assert objects["checkpoint_complete.json"]["sha256"] == record[
            "completion_sha256"
        ]
        assert backed_up["tracked_metadata"]["config.json"]["sha256"] == _sha256(
            config_path
        )
        assert backed_up["tracked_metadata"]["run_meta.json"]["sha256"] == _sha256(
            run_meta_path
        )
        assert backed_up["tracked_metadata"]["r2_publication.json"][
            "sha256"
        ] == _sha256(publication_path)
