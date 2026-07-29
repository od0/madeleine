from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from data.toy_sessions import (
    FRAME_INDEX_CELL_COUNT,
    FRAME_INDEX_CELL_SIZE,
    generate_sessions,
    render_frame_index_strip,
)


VECTORS_PATH = ROOT / "specs" / "frameindex_test_vectors.json"


def _validate(session_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "data.validate_session", str(session_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _read_manifest(session_dir: Path) -> dict:
    return json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(session_dir: Path, manifest: dict) -> None:
    (session_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


@pytest.fixture(scope="module")
def toy_sessions(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    out = tmp_path_factory.mktemp("toy_sessions")
    return generate_sessions(out=out, sessions=2, seconds=1.0, seed=123)


def test_generated_toy_sessions_validate(toy_sessions: list[Path]) -> None:
    assert len(toy_sessions) == 2
    for session_dir in toy_sessions:
        result = _validate(session_dir)
        assert result.returncode == 0, result.stderr


def test_frame_index_strip_matches_frozen_vectors() -> None:
    vectors = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))["vectors"]
    assert FRAME_INDEX_CELL_COUNT == 30
    for vector in vectors:
        strip = render_frame_index_strip(vector["frame_idx"])
        center_y = FRAME_INDEX_CELL_SIZE + FRAME_INDEX_CELL_SIZE // 2
        fills = []
        for cell_idx in range(FRAME_INDEX_CELL_COUNT):
            center_x = (
                (1 + cell_idx) * FRAME_INDEX_CELL_SIZE
                + FRAME_INDEX_CELL_SIZE // 2
            )
            fills.append("1" if strip[center_y, center_x] == 255 else "0")
        assert "".join(fills) == vector["cells"]


def test_foreign_provenance_with_truth_fails(
    toy_sessions: list[Path], tmp_path: Path
) -> None:
    session_dir = tmp_path / toy_sessions[0].name
    shutil.copytree(toy_sessions[0], session_dir)
    manifest = _read_manifest(session_dir)
    manifest["provenance"]["source"] = "nitrogen"
    _write_manifest(session_dir, manifest)

    result = _validate(session_dir)
    assert result.returncode != 0
    assert "truth.parquet is forbidden" in result.stderr


def test_truth_frame_gap_fails(
    toy_sessions: list[Path], tmp_path: Path
) -> None:
    session_dir = tmp_path / toy_sessions[0].name
    shutil.copytree(toy_sessions[0], session_dir)
    truth_path = session_dir / "truth.parquet"
    table = pq.read_table(truth_path)
    keep = pa.array(
        [row for row in range(table.num_rows) if row != 10], type=pa.int64()
    )
    pq.write_table(table.take(keep), truth_path)

    manifest = _read_manifest(session_dir)
    manifest["integrity"]["sha256"]["truth.parquet"] = hashlib.sha256(
        truth_path.read_bytes()
    ).hexdigest()
    _write_manifest(session_dir, manifest)

    result = _validate(session_dir)
    assert result.returncode != 0
    assert "dense and monotonic" in result.stderr


def test_corrupt_manifest_sha256_fails(
    toy_sessions: list[Path], tmp_path: Path
) -> None:
    session_dir = tmp_path / toy_sessions[0].name
    shutil.copytree(toy_sessions[0], session_dir)
    manifest = _read_manifest(session_dir)
    manifest["integrity"]["sha256"]["truth.parquet"] = "0" * 64
    _write_manifest(session_dir, manifest)

    result = _validate(session_dir)
    assert result.returncode != 0
    assert "sha256 mismatch for truth.parquet" in result.stderr
