from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.validate_oracle_window_highres_run import (
    FROZEN_CONFIG_SHA256,
    _canonical_sha256,
    publish_audit,
    validate_completion_marker,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.requires_private_artifacts("experiments/configs/oracle_window_highres_regional_v1.json")
def test_validator_is_bound_to_frozen_config() -> None:
    path = Path("experiments/configs/oracle_window_highres_regional_v1.json")
    assert _sha(path) == FROZEN_CONFIG_SHA256


def test_completion_marker_rechecks_every_bound_artifact(tmp_path: Path) -> None:
    runs = {}
    for arm in ("h32_q", "h128_g", "h128_q"):
        run = tmp_path / arm
        run.mkdir()
        for name in ("run_receipt.json", "model.pt", "predictions.npz"):
            (run / name).write_bytes(f"{arm}:{name}".encode())
        runs[arm] = run
    inputs = {}
    for name in ("report", "config", "cache_receipt", "base_dataset_manifest"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        inputs[name] = path
    marker_base = {
        "schema_version": "madeleine.oracle-window-highres-complete.v1",
        "status": "complete",
        "execution_mode": "production",
        **{
            name: {"path": str(path), "sha256": _sha(path)}
            for name, path in inputs.items()
        },
        "runs": {
            arm: {
                "path": str(run),
                "run_receipt_sha256": _sha(run / "run_receipt.json"),
                "checkpoint_sha256": _sha(run / "model.pt"),
                "prediction_sidecar_sha256": _sha(run / "predictions.npz"),
            }
            for arm, run in runs.items()
        },
    }
    marker = {**marker_base, "content_sha256": _canonical_sha256(marker_base)}
    marker_path = tmp_path / "complete.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    validate_completion_marker(
        marker_path=marker_path,
        report_path=inputs["report"],
        runs=runs,
        config_path=inputs["config"],
        cache_receipt_path=inputs["cache_receipt"],
        base_manifest_path=inputs["base_dataset_manifest"],
    )
    (runs["h128_q"] / "predictions.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="h128_q:predictions"):
        validate_completion_marker(
            marker_path=marker_path,
            report_path=inputs["report"],
            runs=runs,
            config_path=inputs["config"],
            cache_receipt_path=inputs["cache_receipt"],
            base_manifest_path=inputs["base_dataset_manifest"],
        )


def test_audit_publication_is_marker_last_and_collision_safe(tmp_path: Path) -> None:
    audit = {
        "status": "complete",
        "study_id": "test",
        "source_commit": "a" * 40,
        "decision": {"status": "reject_study_h_primary_gate"},
    }
    out = tmp_path / "audit.json"
    marker = tmp_path / "audit_complete.json"
    publish_audit(audit=audit, out=out, marker=marker)
    receipt = json.loads(marker.read_text())
    assert receipt["audit"]["sha256"] == _sha(out)
    content = dict(receipt)
    digest = content.pop("content_sha256")
    assert digest == _canonical_sha256(content)
    with pytest.raises(ValueError, match="overwrite"):
        publish_audit(audit=audit, out=out, marker=marker)
