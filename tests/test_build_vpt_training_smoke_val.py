from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments import build_vpt_training_smoke_val as smoke


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(path: Path) -> None:
    records = [
        {"session_id": "a", "phase": 0, "derived_rows": 100, "windows": 0},
        {"session_id": "b", "phase": 0, "derived_rows": 200, "windows": 20},
        {"session_id": "c", "phase": 0, "derived_rows": 300, "windows": 15},
        {"session_id": "d", "phase": 0, "derived_rows": 400, "windows": 99},
    ]
    value = {
        "schema_version": smoke.SCHEMA,
        "created_at": "frozen",
        "source_root": "/source",
        "sessions_file": {"path": "all.txt"},
        "phases": [0],
        "source_rate_hz": 60,
        "derived_rate_hz": 20,
        "window": 128,
        "stride": 64,
        "records": records,
        "totals": {
            "source_sessions": 4,
            "derived_streams": 4,
            "derived_rows": 1000,
            "windows": 134,
        },
    }
    value["content_sha256"] = smoke.canonical_sha256(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_builds_deterministic_training_only_prefix(tmp_path: Path) -> None:
    source = tmp_path / "build_manifest.json"
    _write_source(source)
    args = smoke.argparse.Namespace(
        source_manifest=source,
        source_manifest_sha256=_sha(source),
        minimum_windows=32,
        output_manifest=tmp_path / "smoke_val_manifest.json",
        output_sessions=tmp_path / "smoke_val_sessions.txt",
        output_receipt=tmp_path / "smoke_val_receipt.json",
    )
    receipt = smoke.build(args)
    manifest = json.loads(args.output_manifest.read_text())
    assert [record["session_id"] for record in manifest["records"]] == ["b", "c"]
    assert manifest["totals"] == {
        "source_sessions": 2,
        "derived_streams": 2,
        "derived_rows": 500,
        "windows": 35,
    }
    assert receipt["selected_windows"] == 35
    assert receipt["source_manifest_sha256"] == _sha(source)

    with pytest.raises(FileExistsError):
        smoke.build(args)


def test_rejects_changed_source_content_hash(tmp_path: Path) -> None:
    source = tmp_path / "build_manifest.json"
    _write_source(source)
    value = json.loads(source.read_text())
    value["records"][1]["windows"] = 21
    source.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args = smoke.argparse.Namespace(
        source_manifest=source,
        source_manifest_sha256=_sha(source),
        minimum_windows=32,
        output_manifest=tmp_path / "smoke_val_manifest.json",
        output_sessions=tmp_path / "smoke_val_sessions.txt",
        output_receipt=tmp_path / "smoke_val_receipt.json",
    )
    with pytest.raises(RuntimeError, match="content hash"):
        smoke.build(args)
