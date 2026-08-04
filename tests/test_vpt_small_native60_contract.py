from __future__ import annotations

import hashlib
import json
from pathlib import Path

import experiments.validate_vpt_small_native60_contract as contract_module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generation_contract_binds_manifests_and_endpoints(
    tmp_path: Path, monkeypatch,
) -> None:
    roots = {}
    for name in ("train", "validation"):
        root = tmp_path / name
        root.mkdir()
        (root / "build_manifest.json").write_text(name + " manifest\n")
        (root / "complete.json").write_text(name + " marker\n")
        roots[name] = root
    reports = {
        "train": {
            "ok": True,
            "manifest_sha256": _sha256(roots["train"] / "build_manifest.json"),
            "derived_streams": 148,
            "derived_rows": 2_900_000,
            "windows": 15_000,
            "window": 128,
            "stride": 64,
        },
        "validation": {
            "ok": True,
            "manifest_sha256": _sha256(roots["validation"] / "build_manifest.json"),
            "derived_streams": 1,
            "derived_rows": 18_000,
            "windows": 280,
            "window": 128,
            "stride": 64,
        },
    }
    monkeypatch.setattr(
        contract_module,
        "validate",
        lambda root: reports["train" if root == roots["train"] else "validation"],
    )
    steps = 118
    contract = {
        "schema_version": "madeleine.vpt-small-native60-generation.v1",
        "geometry": {"window": 128, "stride": 64},
        "generations": {
            name: {
                "manifest_sha256": _sha256(root / "build_manifest.json"),
                "marker_sha256": _sha256(root / "complete.json"),
                "derived_rows": reports[name]["derived_rows"],
                "windows": reports[name]["windows"],
                "derived_streams": reports[name]["derived_streams"],
            }
            for name, root in roots.items()
        },
        "training": {
            "global_batch": 128,
            "full_epochs": 20,
            "optimizer_steps_per_epoch": steps,
            "short_endpoint_optimizer_steps": 2340,
            "full_endpoint_optimizer_steps": steps * 20,
        },
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    report = contract_module.validate_contract(
        path, roots["train"], roots["validation"]
    )
    assert report["ok"] is True
    assert report["full_endpoint_optimizer_steps"] == 2360


def test_span_contract_has_no_short_endpoint(
    tmp_path: Path, monkeypatch,
) -> None:
    roots = {}
    for name in ("train", "validation"):
        root = tmp_path / name
        root.mkdir()
        (root / "build_manifest.json").write_text(name + " span manifest\n")
        (root / "complete.json").write_text(name + " span marker\n")
        roots[name] = root
    reports = {
        "train": {
            "manifest_sha256": _sha256(roots["train"] / "build_manifest.json"),
            "derived_streams": 148,
            "derived_rows": 2_900_000,
            "windows": 14_900,
            "window": 384,
            "stride": 192,
        },
        "validation": {
            "manifest_sha256": _sha256(roots["validation"] / "build_manifest.json"),
            "derived_streams": 1,
            "derived_rows": 18_000,
            "windows": 92,
            "window": 384,
            "stride": 192,
        },
    }
    monkeypatch.setattr(
        contract_module,
        "validate",
        lambda root: reports["train" if root == roots["train"] else "validation"],
    )
    steps = 117
    contract = {
        "schema_version": "madeleine.vpt-small-native60-generation.v1",
        "geometry": {"window": 384, "stride": 192},
        "generations": {
            name: {
                "manifest_sha256": _sha256(root / "build_manifest.json"),
                "marker_sha256": _sha256(root / "complete.json"),
                "derived_rows": reports[name]["derived_rows"],
                "windows": reports[name]["windows"],
                "derived_streams": reports[name]["derived_streams"],
            }
            for name, root in roots.items()
        },
        "training": {
            "global_batch": 128,
            "full_epochs": 20,
            "optimizer_steps_per_epoch": steps,
            "full_endpoint_optimizer_steps": steps * 20,
        },
    }
    path = tmp_path / "span_contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    report = contract_module.validate_contract(
        path, roots["train"], roots["validation"]
    )
    assert report["window"] == 384
    assert report["full_endpoint_optimizer_steps"] == 2340
    assert "short_endpoint_optimizer_steps" not in report
