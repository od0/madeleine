from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harvest.scan_wild_cells import (
    SPEC_VERSION,
    classify_cells,
    load_spec,
    ranged_score_filter_graph,
    score_decode_command,
    score_filter_graph,
    transition_stats,
)


def spec() -> dict:
    return {
        "format_version": SPEC_VERSION,
        "video_id": "video_1",
        "frame_size_wh": [1280, 720],
        "source_sha256": "a" * 64,
        "pts_sha256": "b" * 64,
        "survey_sha256": "c" * 64,
        "survey_contact_sheet_sha256": "d" * 64,
        "cells": [
            {
                "cell_id": "left",
                "physical_label": "Left",
                "sample_rect_px": [10, 20, 8, 8],
                "pressed_polarity": "high",
            },
            {
                "cell_id": "right",
                "physical_label": "Right",
                "sample_rect_px": [30, 20, 8, 8],
                "pressed_polarity": "low",
            },
        ],
        "human_reviewed": False,
        "training_admitted": False,
    }


def test_load_spec_rejects_claims_and_bad_rectangles(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec()))
    assert load_spec(path)["video_id"] == "video_1"
    claimed = spec()
    claimed["human_reviewed"] = True
    path.write_text(json.dumps(claimed))
    with pytest.raises(ValueError, match="cannot claim"):
        load_spec(path)
    outside = spec()
    outside["cells"][0]["sample_rect_px"] = [1279, 719, 8, 8]
    path.write_text(json.dumps(outside))
    with pytest.raises(ValueError, match="leaves"):
        load_spec(path)


def test_score_filter_graph_has_one_crop_per_cell() -> None:
    graph, frame_bytes = score_filter_graph(spec()["cells"])
    assert "split=2" in graph
    assert "crop=8:8:10:20" in graph
    assert "crop=8:8:30:20" in graph
    assert "hstack=inputs=2" in graph
    assert frame_bytes == 8 * 16


def test_score_filter_graph_pads_heterogeneous_cell_heights() -> None:
    cells = spec()["cells"]
    cells[0]["sample_rect_px"] = [10, 20, 8, 7]
    cells[1]["sample_rect_px"] = [30, 20, 8, 9]
    graph, frame_bytes = score_filter_graph(cells)
    assert "crop=8:7:10:20,pad=8:9:0:0:black" in graph
    assert "crop=8:9:30:20" in graph
    assert frame_bytes == 8 * 2 * 9


def test_ranged_score_filter_graph_selects_exact_source_indices() -> None:
    graph, _ = ranged_score_filter_graph(spec()["cells"], 120, 60)
    assert graph.startswith("[0:v]select=between(n\\,120\\,179),format=gray")
    command, _ = score_decode_command(
        Path("source.mp4"), spec()["cells"], hwaccel="none",
        start_index=120, count=60,
    )
    assert command[command.index("-frames:v") + 1] == "60"


def test_classify_cells_finds_high_and_low_polarities() -> None:
    high_states = np.tile(np.r_[np.zeros(40), np.ones(40)], 2)
    scores = np.column_stack((20 + 200 * high_states, 220 - 200 * high_states))
    pts = np.arange(scores.shape[0], dtype=np.float64) / 60.0
    diagnostics, evidence = classify_cells(scores, pts, spec()["cells"])
    assert [row["changing"] for row in diagnostics] == [True, True]
    assert all(row["minority_frames"] == 80 for row in diagnostics)
    assert len(evidence) >= 2


def test_transition_stats_counts_single_frame_positive_runs() -> None:
    stats = transition_stats(
        np.asarray([False, True, False, True, True, False]), 1.0
    )
    assert stats["pressed_frames"] == 3
    assert stats["positive_runs"] == 2
    assert stats["single_frame_positive_runs"] == 1


def test_decode_scores_source_requires_passthrough_vsync() -> None:
    command, _ = score_decode_command(Path("source.mp4"), spec()["cells"])
    assert command[command.index("-vsync") + 1] == "0"


def test_decode_scores_can_explicitly_use_cpu() -> None:
    command, _ = score_decode_command(
        Path("source.mp4"), spec()["cells"], hwaccel="none"
    )
    assert "-hwaccel" not in command
