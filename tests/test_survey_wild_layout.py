from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harvest.survey_wild_layout import (
    contact_sheet_command,
    declared_artifact_path,
    exact_extract_command,
    sample_indices,
    sha256_file,
    validate_pts_order,
)


def test_sample_indices_are_unique_and_span_source() -> None:
    indices = sample_indices(60_000, 16)
    assert len(indices) == len(set(indices)) == 16
    assert indices == sorted(indices)
    assert indices[0] == pytest.approx(1_200, abs=1)
    assert indices[-1] == pytest.approx(58_799, abs=1)


def test_sample_indices_handles_short_sources_without_duplicates() -> None:
    assert sample_indices(3, 10) == [0, 1, 2]
    with pytest.raises(ValueError, match="positive"):
        sample_indices(0, 2)


def test_pts_order_allows_explicit_duplicates_but_not_time_reversal() -> None:
    assert validate_pts_order(np.array([0.0, 0.1, 0.1, 0.2])) == 1
    with pytest.raises(ValueError, match="nondecreasing"):
        validate_pts_order(np.array([0.0, 0.2, 0.1]))
    with pytest.raises(ValueError, match="finite"):
        validate_pts_order(np.array([0.0, np.nan]))


def test_exact_extract_command_uses_decoded_indices_and_gpu_download() -> None:
    command = exact_extract_command(
        Path("source.mp4"), Path("sample-%03d.png"), [10, 20], "cuda"
    )
    assert command[command.index("-hwaccel") + 1] == "cuda"
    assert command[command.index("-frames:v") + 1] == "2"
    filter_graph = command[command.index("-vf") + 1]
    assert "select=eq(n\\,10)+eq(n\\,20)" in filter_graph
    assert "hwdownload" in filter_graph


def test_cpu_extract_and_contact_sheet_commands_are_bounded() -> None:
    command = exact_extract_command(
        Path("source.mp4"), Path("sample-%03d.png"), [10], "none"
    )
    assert "-hwaccel" not in command
    assert "hwdownload" not in command[command.index("-vf") + 1]
    sheet = contact_sheet_command(
        Path("sample-%03d.png"), Path("contact.png"), 16
    )
    assert "tile=4x4" in sheet[sheet.index("-vf") + 1]
    assert sheet[sheet.index("-frames:v") + 1] == "1"


def test_declared_artifact_rejects_mutation_and_path_escape(tmp_path: Path) -> None:
    artifact = tmp_path / "frame.png"
    artifact.write_bytes(b"original")
    declared = {
        "path": artifact.name,
        "size_bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
    }
    assert declared_artifact_path(tmp_path, declared) == artifact.resolve()
    artifact.write_bytes(b"mutatED!")
    with pytest.raises(ValueError, match="hash differs"):
        declared_artifact_path(tmp_path, declared)
    with pytest.raises(ValueError, match="inside"):
        declared_artifact_path(tmp_path, {**declared, "path": "../frame.png"})
