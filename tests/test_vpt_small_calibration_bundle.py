from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.schema import KEY_ORDER
from experiments.prepare_vpt_small_calibration_bundle import seal_bundle


def _component(path: Path, session: str, minutes: float, runs: int) -> Path:
    receipt = {
        "role": "c1",
        "decision": "rejected",
        "violations": [
            f"common-support active minutes {minutes:.6f} below 15.000000"
        ],
        "model_accessed": False,
        "session": {"session_id": session},
        "capture_integrity": {"validator_violations": [], "drop_rate": 0.001},
        "leak_probe": {"max_symmetric_margin_auc": 0.55},
        "support": {
            "rows": 100,
            "active_rows": 90,
            "active_minutes": minutes,
            "positive_state_runs": {key: runs for key in KEY_ORDER},
            "segments": 2,
            "support_sha256": session * 2,
        },
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_bundle_sums_support_without_bridging(tmp_path: Path) -> None:
    first = _component(tmp_path / "a.json", "a", 14.2, 20)
    second = _component(tmp_path / "b.json", "b", 2.0, 10)
    receipt = seal_bundle(
        role="c1",
        component_receipts=[first, second],
        out=tmp_path / "bundle.json",
        repo=Path("."),
    )
    assert receipt["decision"] == "accepted"
    assert receipt["support"]["active_minutes"] == pytest.approx(16.2)
    assert receipt["support"]["positive_state_runs"]["down"] == 30
    assert receipt["support"]["component_boundaries"] == 1
    assert "no inference window may bridge" in receipt["window_boundary_policy"]


def test_bundle_rejects_component_integrity_failure(tmp_path: Path) -> None:
    first = _component(tmp_path / "a.json", "a", 14.2, 30)
    second = _component(tmp_path / "b.json", "b", 2.0, 30)
    value = json.loads(second.read_text())
    value["violations"].append("input_overlay masked zone is not identically zero")
    second.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="masked zone"):
        seal_bundle(
            role="c1",
            component_receipts=[first, second],
            out=tmp_path / "bundle.json",
            repo=Path("."),
        )


def test_bundle_rejects_model_exposed_component(tmp_path: Path) -> None:
    first = _component(tmp_path / "a.json", "a", 14.2, 30)
    second = _component(tmp_path / "b.json", "b", 2.0, 30)
    value = json.loads(second.read_text())
    value["model_accessed"] = True
    second.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="exposed to a model"):
        seal_bundle(
            role="c1",
            component_receipts=[first, second],
            out=tmp_path / "bundle.json",
            repo=Path("."),
        )
