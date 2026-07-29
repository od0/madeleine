from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from data.schema import KEY_ORDER, labels_native_schema
from nitrogen import map_actions
from nitrogen import slice as nitrogen_slice


CHUNK_SIZE = 600
VIDEO_BINDS = {
    "video_a": {"jump": "south", "dash": "west", "grab": "right_trigger"},
    "video_b": {"jump": "east", "dash": "south", "grab": "left_shoulder"},
}


def _metadata(video_id: str, chunk_id: int) -> dict[str, Any]:
    return {
        "chunk_id": f"{chunk_id:04d}",
        "chunk_size": CHUNK_SIZE,
        "original_video": {
            "video_id": video_id,
            "source": "youtube",
            "url": f"https://example.test/{video_id}",
            "resolution": [720, 1280],
            "duration": 20,
        },
        "game": "celeste",
        "controller_type": "xboxone",
        "bbox_controller_overlay": [10, 650, 300, 150],
    }


def _add_runs(values: np.ndarray, starts: list[int], duration: int) -> None:
    for start in starts:
        values[start : start + duration] = 1


def _raw_arrays(
    video_id: str,
    *,
    include_threshold_edges: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    buttons = {
        button: np.zeros(CHUNK_SIZE, dtype=np.int32)
        for button in map_actions.BUTTON_COLUMNS
    }
    j_left = np.zeros((CHUNK_SIZE, 2), dtype=np.float64)
    j_right = np.zeros((CHUNK_SIZE, 2), dtype=np.float64)

    if include_threshold_edges:
        j_left[0, 0] = -0.49
        j_left[1, 0] = -0.51
        j_left[2, 0] = 0.49
        j_left[3, 0] = 0.51
        j_left[4, 1] = -0.49
        j_left[5, 1] = -0.51
        j_left[6, 1] = 0.49
        j_left[7, 1] = 0.51

    if video_id in VIDEO_BINDS:
        binds = VIDEO_BINDS[video_id]
        jump_starts = list(range(20, 580, 40))
        dash_starts = list(range(30, 590, 40))
        grab_starts = [100, 280, 460]
        _add_runs(buttons[binds["jump"]], jump_starts, 4)
        _add_runs(buttons[binds["dash"]], dash_starts, 1)
        _add_runs(buttons[binds["grab"]], grab_starts, 25)
        for index, start in enumerate(dash_starts):
            j_left[start, 0] = -1.0 if index % 2 else 1.0
        for start in grab_starts:
            buttons["dpad_up"][start : start + 25] = 1
            j_left[start : start + 25, 1] = -1.0
    else:
        # Too little evidence to accept any inferred assignment.
        buttons["south"][20] = 1
        buttons["west"][30] = 1
        j_left[30, 0] = 1.0

    return buttons, j_left, j_right


def _write_raw(
    path: Path,
    video_id: str,
    *,
    stick_type: pa.DataType,
    include_threshold_edges: bool,
) -> None:
    buttons, j_left, j_right = _raw_arrays(
        video_id,
        include_threshold_edges=include_threshold_edges,
    )
    if pa.types.is_int64(stick_type):
        j_left = j_left.astype(np.int64)
        j_right = j_right.astype(np.int64)
    arrays: list[pa.Array] = [
        pa.array(buttons[button], type=pa.int32())
        for button in map_actions.BUTTON_COLUMNS
    ]
    arrays.extend(
        [
            pa.array(j_left.tolist(), type=pa.list_(stick_type)),
            pa.array(j_right.tolist(), type=pa.list_(stick_type)),
        ]
    )
    names = [*map_actions.BUTTON_COLUMNS, "j_left", "j_right"]
    pq.write_table(pa.Table.from_arrays(arrays, names=names), path)


@pytest.fixture
def action_fixture(tmp_path: Path) -> tuple[Path, Path]:
    actions_root = tmp_path / "actions"
    for video_index, video_id in enumerate(("video_a", "video_b", "video_c")):
        for chunk_id in range(2):
            chunk_dir = (
                actions_root
                / f"SHARD_{video_index:04d}"
                / video_id
                / f"{video_id}_chunk_{chunk_id:04d}"
            )
            chunk_dir.mkdir(parents=True)
            (chunk_dir / "metadata.json").write_text(
                json.dumps(_metadata(video_id, chunk_id)),
                encoding="utf-8",
            )
            # Video B contains otherwise-identical DOUBLE[] and BIGINT[] chunks.
            stick_type = (
                pa.int64()
                if video_id == "video_b" and chunk_id == 1
                else pa.float64()
            )
            _write_raw(
                chunk_dir / "actions_raw.parquet",
                video_id,
                stick_type=stick_type,
                include_threshold_edges=video_id == "video_a" and chunk_id == 0,
            )

    chunk_index = tmp_path / "chunk_index.parquet"
    rows = nitrogen_slice.discover_chunks(actions_root, "celeste")
    nitrogen_slice.write_chunk_index(rows, chunk_index)
    return actions_root, chunk_index


def _report(output: Path, video_id: str) -> dict[str, Any]:
    return json.loads(
        (output / video_id / "mapping_report.json").read_text(encoding="utf-8")
    )


def _labels(output: Path, video_id: str, chunk_id: int) -> pa.Table:
    return pq.read_table(
        output
        / video_id
        / f"{video_id}_chunk_{chunk_id:04d}"
        / "labels_native.parquet"
    )


def test_per_video_bind_inference_reports_and_prior_fallback(
    action_fixture: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    actions_root, chunk_index = action_fixture
    output = tmp_path / "mapped"

    reports = map_actions.map_actions(chunk_index, actions_root, output)

    assert len(reports) == 3
    report_a = _report(output, "video_a")
    assert report_a["bind_map"] == {
        "jump": ["south"],
        "dash": ["west"],
        "grab": ["right_trigger"],
    }
    assert report_a["confidence"] >= map_actions.FLAG_THRESHOLD
    assert report_a["flagged"] is False
    assert report_a["axis_sign_convention"] == "negative_is_up"
    assert report_a["axis_sign_indeterminate"] is False
    assert report_a["axis_sign_evidence"]["negative_is_up_votes"] > 0
    assert report_a["chunks_mapped"] == 2
    assert report_a["chunks_skipped"] == 0
    assert report_a["tool_version"] == map_actions.TOOL_VERSION

    report_b = _report(output, "video_b")
    assert report_b["bind_map"] == {
        "jump": ["east"],
        "dash": ["south"],
        "grab": ["left_shoulder"],
    }
    assert report_b["confidence"] >= map_actions.FLAG_THRESHOLD
    assert report_b["flagged"] is False

    report_c = _report(output, "video_c")
    assert report_c["flagged"] is True
    assert report_c["confidence"] < map_actions.FLAG_THRESHOLD
    assert report_c["bind_map"] == map_actions.PRIOR_BIND_MAP
    assert report_c["axis_sign_convention"] == "negative_is_up"
    assert report_c["axis_sign_indeterminate"] is True

    south_stats = report_a["evidence"]["south"]
    west_stats = report_a["evidence"]["west"]
    trigger_stats = report_a["evidence"]["right_trigger"]
    assert south_stats["press_count"] > 10
    assert south_stats["median_press_duration_frames"] == 4.0
    assert west_stats["burst_fraction_le_8_frames_60hz_equivalent"] == 1.0
    assert west_stats["direction_co_press_rate"] == 1.0
    assert trigger_stats["median_press_duration_frames"] == 25.0
    assert trigger_stats["up_co_press_rate"] == 1.0


def test_native_schema_threshold_edges_and_integer_sticks(
    action_fixture: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    actions_root, chunk_index = action_fixture
    output = tmp_path / "mapped"
    assert (
        map_actions.main(
            [
                "--chunk-index",
                str(chunk_index),
                "--actions-root",
                str(actions_root),
                "--out",
                str(output),
            ]
        )
        == 0
    )

    labels_a = _labels(output, "video_a", 0)
    assert labels_a.schema == labels_native_schema(30.0)
    assert labels_a.column_names == ["frame_idx", *KEY_ORDER]
    assert labels_a.num_rows == CHUNK_SIZE
    assert labels_a["frame_idx"].to_pylist() == list(range(CHUNK_SIZE))
    assert labels_a.schema.metadata == {b"grid_hz": b"30.0"}

    rows = labels_a.select(["left", "right", "up", "down"]).slice(0, 8).to_pylist()
    assert rows == [
        {"left": False, "right": False, "up": False, "down": False},
        {"left": True, "right": False, "up": False, "down": False},
        {"left": False, "right": False, "up": False, "down": False},
        {"left": False, "right": True, "up": False, "down": False},
        {"left": False, "right": False, "up": False, "down": False},
        {"left": False, "right": False, "up": True, "down": False},
        {"left": False, "right": False, "up": False, "down": False},
        {"left": False, "right": False, "up": False, "down": True},
    ]
    assert labels_a["jump"][20].as_py() is True
    assert labels_a["jump"][24].as_py() is False
    assert labels_a["dash"][30].as_py() is True
    assert labels_a["grab"][100].as_py() is True

    labels_b_double = _labels(output, "video_b", 0)
    labels_b_bigint = _labels(output, "video_b", 1)
    assert labels_b_double.equals(labels_b_bigint)
    assert labels_b_bigint.schema == labels_native_schema(30.0)

    assert not any(path.name == "truth.parquet" for path in output.rglob("*"))


def test_missing_chunk_is_reported_and_other_chunks_continue(
    action_fixture: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    actions_root, chunk_index = action_fixture
    missing = (
        actions_root
        / "SHARD_0002"
        / "video_c"
        / "video_c_chunk_0001"
        / "actions_raw.parquet"
    )
    missing.unlink()
    output = tmp_path / "mapped"

    map_actions.map_actions(
        chunk_index,
        actions_root,
        output,
        videos=["video_c"],
    )

    report = _report(output, "video_c")
    assert report["chunks_mapped"] == 1
    assert report["chunks_skipped"] == 1
    assert report["skipped_details"] == [
        {"chunk_id": 1, "error": "missing actions_raw.parquet"}
    ]
    assert not (output / "video_a").exists()
