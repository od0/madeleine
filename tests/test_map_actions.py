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
        grab_starts = [80, 180, 280, 380, 480]
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
    assert report_a["schema_version"] == map_actions.REPORT_SCHEMA_VERSION
    assert set(report_a["per_action"]) == set(map_actions.ACTION_NAMES)
    for action, detail in report_a["per_action"].items():
        assert detail["flagged"] is False
        assert detail["fallback_used"] is False
        assert detail["selected"] == report_a["bind_map"][action]
        assert detail["confidence"] >= map_actions.FLAG_THRESHOLD
    assert report_a["direction_rule"] == {
        "source": "NitroGen dataset coordinate contract",
        "source_revision": map_actions.NITROGEN_COORDINATE_CONTRACT_REVISION,
        "axis_threshold": 0.5,
        "left": "dpad_left OR j_left_x < -axis_threshold",
        "right": "dpad_right OR j_left_x > axis_threshold",
        "up": "dpad_up OR j_left_y < -axis_threshold",
        "down": "dpad_down OR j_left_y > axis_threshold",
        "comparisons": "strict",
    }
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


def _synthetic_evidence(
    **buttons: dict[str, float],
) -> dict[str, dict[str, float | int | str]]:
    zero = {
        "press_rate": 0.0,
        "press_count": 0,
        "presses_per_hour": 0.0,
        "median_press_duration_frames": 0.0,
        "median_press_duration_seconds": 0.0,
        "median_press_duration_frames_60hz_equivalent": 0.0,
        "burst_fraction_le_8_frames_60hz_equivalent": 0.0,
        "direction_co_press_rate": 0.0,
        "up_co_press_rate": 0.0,
        "duration_note": "synthetic",
    }
    evidence = {button: dict(zero) for button in map_actions.CANDIDATE_BUTTONS}
    for button, overrides in buttons.items():
        evidence[button].update(overrides)
    return evidence


def _button_stats(
    presses: int,
    per_hour: float,
    median_frames: float,
    *,
    burst: float,
    direction: float,
    upward: float = 0.0,
) -> dict[str, float | int]:
    return {
        "press_count": presses,
        "presses_per_hour": per_hour,
        "median_press_duration_frames_60hz_equivalent": median_frames,
        "burst_fraction_le_8_frames_60hz_equivalent": burst,
        "direction_co_press_rate": direction,
        "up_co_press_rate": upward,
    }


def test_phantom_button_cannot_win_dash() -> None:
    # Regression for the audited dash-starvation defect: a three-press
    # one-frame phantom trigger must not outscore a real held dash button.
    evidence = _synthetic_evidence(
        south=_button_stats(1400, 1500.0, 12.0, burst=0.30, direction=0.85, upward=0.20),
        west=_button_stats(1200, 1300.0, 11.0, burst=0.40, direction=0.96, upward=0.45),
        right_trigger=_button_stats(500, 550.0, 70.0, burst=0.02, direction=0.80, upward=0.30),
        left_trigger=_button_stats(3, 3.2, 1.0, burst=1.0, direction=1.0),
    )
    bind_map, _, _, _, per_action = map_actions.infer_bind_map(evidence)
    assert bind_map["dash"] == ["west"]
    assert per_action["dash"]["inferred_button"] == "west"
    assert "left_trigger" not in per_action["dash"]["eligible_candidates"]


def test_action_without_eligible_candidate_falls_back_alone() -> None:
    # An hour-scale video whose only dash-like button is far below the rate
    # floor must flag dash and fall back to the dash prior, while jump and
    # grab keep their inferred buttons.
    evidence = _synthetic_evidence(
        south=_button_stats(2000, 1800.0, 28.0, burst=0.25, direction=0.85, upward=0.20),
        right_trigger=_button_stats(600, 540.0, 70.0, burst=0.02, direction=0.80, upward=0.35),
        west=_button_stats(20, 18.0, 2.0, burst=0.95, direction=0.60),
    )
    bind_map, _, flagged, _, per_action = map_actions.infer_bind_map(evidence)
    assert flagged is True
    assert per_action["dash"]["flagged"] is True
    assert per_action["dash"]["reason"] == "no_eligible_candidate"
    assert bind_map["dash"] == map_actions.PRIOR_BIND_MAP["dash"]
    assert per_action["jump"]["flagged"] is False
    assert bind_map["jump"] == ["south"]
    assert per_action["grab"]["flagged"] is False
    assert bind_map["grab"] == ["right_trigger"]


def test_short_video_clears_floors_by_rate() -> None:
    # 18 dash presses in roughly four minutes is 240/h: above both floors.
    evidence = _synthetic_evidence(
        south=_button_stats(60, 800.0, 10.0, burst=0.40, direction=0.80, upward=0.20),
        west=_button_stats(18, 240.0, 3.0, burst=0.95, direction=0.95),
        right_trigger=_button_stats(20, 260.0, 60.0, burst=0.05, direction=0.70, upward=0.40),
    )
    bind_map, _, _, _, per_action = map_actions.infer_bind_map(evidence)
    assert bind_map["dash"] == ["west"]
    assert per_action["dash"]["flagged"] is False


def test_shared_sole_candidate_yields_assignment_conflict() -> None:
    # One button eligible for both jump and dash: the stronger action keeps
    # it and the other reports an assignment conflict with prior fallback.
    evidence = _synthetic_evidence(
        south=_button_stats(2000, 1900.0, 8.0, burst=0.60, direction=0.90, upward=0.20),
    )
    bind_map, _, _, _, per_action = map_actions.infer_bind_map(evidence)
    inferred = {
        action
        for action in map_actions.ACTION_NAMES
        if per_action[action]["inferred_button"] == "south"
    }
    assert len(inferred) == 1
    conflicted = [
        action
        for action in map_actions.ACTION_NAMES
        if per_action[action].get("reason") == "assignment_conflict"
    ]
    assert conflicted
    for action in conflicted:
        assert bind_map[action] == map_actions.PRIOR_BIND_MAP[action]


def test_prior_fallback_excludes_buttons_inferred_elsewhere() -> None:
    # south is a long-hold grab and west a low-rate dash; jump has no
    # eligible candidate, and its prior (south) is claimed by grab, so jump
    # must fall back to an explicitly empty selection instead of duplicating
    # the grab column.
    evidence = _synthetic_evidence(
        south=_button_stats(900, 900.0, 45.0, burst=0.05, direction=0.80, upward=0.40),
        west=_button_stats(90, 90.0, 2.0, burst=0.95, direction=0.90),
    )
    bind_map, _, flagged, _, per_action = map_actions.infer_bind_map(evidence)
    assert flagged is True
    assert bind_map["grab"] == ["south"]
    assert bind_map["dash"] == ["west"]
    # south is claimed by grab, so the jump prior reduces to its secondary.
    assert bind_map["jump"] == ["north"]
    assert per_action["jump"]["flagged"] is True
    assert per_action["jump"]["prior_reduced_by_inferred"] == ["south"]


def test_grab_prior_fallback_drops_inferred_shoulder() -> None:
    # jump inferred on left_shoulder; the flagged grab fallback must drop
    # left_shoulder from its four-button prior.
    evidence = _synthetic_evidence(
        left_shoulder=_button_stats(2000, 1800.0, 9.0, burst=0.45, direction=0.85, upward=0.25),
        west=_button_stats(900, 800.0, 3.0, burst=0.95, direction=0.95),
    )
    bind_map, _, _, _, per_action = map_actions.infer_bind_map(evidence)
    assert bind_map["jump"] == ["left_shoulder"]
    assert per_action["grab"]["fallback_used"] is True
    assert bind_map["grab"] == ["left_trigger", "right_trigger", "right_shoulder"]
    assert per_action["grab"]["prior_reduced_by_inferred"] == ["left_shoulder"]


def test_dual_dash_composite_prevents_jump_misroute() -> None:
    # Modeled on v2093685549: west is the dominant dash, east the secondary,
    # south the held jump. Single-bind inference misrouted jump onto west;
    # the west+east composite must win dash so south keeps jump.
    evidence = _synthetic_evidence(
        south=_button_stats(1888, 950.0, 28.0, burst=0.12, direction=0.85, upward=0.25),
        west=_button_stats(3037, 1500.0, 7.0, burst=0.76, direction=0.97),
        east=_button_stats(378, 190.0, 9.0, burst=0.42, direction=0.91),
        right_trigger=_button_stats(600, 300.0, 60.0, burst=0.03, direction=0.75, upward=0.35),
    )
    bind_map, _, _, _, per_action = map_actions.infer_bind_map(evidence)
    assert bind_map["dash"] == ["east", "west"]
    assert per_action["dash"]["composite"] is True
    assert bind_map["jump"] == ["south"]
    assert bind_map["grab"] == ["right_trigger"]


def test_mid_video_button_switch_is_covered_by_composite() -> None:
    # Modeled on v2129932598: the player switches dash from west to east
    # partway through; both buttons carry real dash mass.
    evidence = _synthetic_evidence(
        south=_button_stats(3000, 1400.0, 12.0, burst=0.30, direction=0.85, upward=0.25),
        west=_button_stats(4148, 1900.0, 6.0, burst=0.80, direction=0.95),
        east=_button_stats(963, 450.0, 5.0, burst=0.85, direction=0.71),
        left_trigger=_button_stats(700, 330.0, 55.0, burst=0.04, direction=0.70, upward=0.30),
    )
    bind_map, _, _, _, _ = map_actions.infer_bind_map(evidence)
    assert bind_map["dash"] == ["east", "west"]
    assert bind_map["jump"] == ["south"]


def test_talk_like_sibling_is_not_absorbed_into_dash() -> None:
    # A non-directional, sparse east (the Talk button) must stay out of the
    # dash selection even though west is a confident dash.
    evidence = _synthetic_evidence(
        south=_button_stats(2000, 1000.0, 12.0, burst=0.30, direction=0.85, upward=0.25),
        west=_button_stats(1500, 750.0, 5.0, burst=0.90, direction=0.96),
        east=_button_stats(120, 60.0, 7.0, burst=0.80, direction=0.05),
        right_trigger=_button_stats(500, 250.0, 60.0, burst=0.03, direction=0.75, upward=0.35),
    )
    bind_map, _, _, _, per_action = map_actions.infer_bind_map(evidence)
    assert bind_map["dash"] == ["west"]
    assert "east" not in per_action["dash"]["inferred_buttons"]


def test_default_jump_pair_forms_composite() -> None:
    evidence = _synthetic_evidence(
        south=_button_stats(2400, 1200.0, 11.0, burst=0.35, direction=0.85, upward=0.25),
        north=_button_stats(320, 160.0, 8.0, burst=0.45, direction=0.80, upward=0.20),
        west=_button_stats(1500, 750.0, 4.0, burst=0.92, direction=0.96),
        right_trigger=_button_stats(600, 300.0, 55.0, burst=0.04, direction=0.75, upward=0.35),
    )
    bind_map, _, _, _, per_action = map_actions.infer_bind_map(evidence)
    assert bind_map["jump"] == ["north", "south"]
    assert per_action["jump"]["composite"] is True
    assert bind_map["dash"] == ["west"]


def test_grab_group_composite_admits_secondary_trigger() -> None:
    evidence = _synthetic_evidence(
        south=_button_stats(2000, 1000.0, 12.0, burst=0.30, direction=0.85, upward=0.25),
        west=_button_stats(1400, 700.0, 4.0, burst=0.92, direction=0.96),
        right_trigger=_button_stats(900, 450.0, 55.0, burst=0.03, direction=0.75, upward=0.35),
        left_shoulder=_button_stats(120, 60.0, 40.0, burst=0.08, direction=0.70, upward=0.30),
    )
    bind_map, _, _, _, per_action = map_actions.infer_bind_map(evidence)
    assert bind_map["grab"] == ["left_shoulder", "right_trigger"]
    assert per_action["grab"]["composite"] is True


def test_bind_resolution_override_labels_and_report(tmp_path: Path) -> None:
    # Labels must follow the resolved button sets, including an empty jump
    # (all-negative column), while the report keeps the inference output
    # under an inference block and records the resolution provenance.
    chunk_dir = tmp_path / "actions" / "SHARD_0000" / "video_r" / "video_r_chunk_0000"
    buttons = {
        button: np.zeros(CHUNK_SIZE, dtype=np.int32)
        for button in map_actions.BUTTON_COLUMNS
    }
    _add_runs(buttons["south"], list(range(20, 580, 40)), 4)
    _add_runs(buttons["left_trigger"], [100, 220, 340, 460], 25)
    j_left = np.zeros((CHUNK_SIZE, 2), dtype=np.float64)
    arrays = [
        pa.array(buttons[button], type=pa.int32())
        for button in map_actions.BUTTON_COLUMNS
    ]
    arrays.extend(
        [
            pa.array(j_left.tolist(), type=pa.list_(pa.float64())),
            pa.array(j_left.tolist(), type=pa.list_(pa.float64())),
        ]
    )
    chunk_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_arrays(
            arrays, names=[*map_actions.BUTTON_COLUMNS, "j_left", "j_right"]
        ),
        chunk_dir / "actions_raw.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "video_id": "video_r",
                    "chunk_id": 0,
                    "chunk_size": CHUNK_SIZE,
                    "grid_hz": 60.0,
                    "controller_type": "xboxone",
                }
            ]
        ),
        tmp_path / "chunk_index.parquet",
    )
    resolution = {
        "schema_version": "madeleine.nitrogen-bind-resolution.v1",
        "created_at": "2026-08-02T00:00:00+00:00",
        "resolution": [
            {"video_id": "video_r", "action": "jump", "resolved": [],
             "review_status": "unresolved_no_legible_event"},
            {"video_id": "video_r", "action": "dash", "resolved": ["south"],
             "review_status": "pending_final_human_review"},
            {"video_id": "video_r", "action": "grab", "resolved": ["left_trigger"],
             "review_status": "cross_validated_not_human_reviewed"},
        ],
    }
    resolution_path = tmp_path / "bind_resolution.json"
    resolution_path.write_text(json.dumps(resolution))
    overrides, review, meta = map_actions.load_bind_resolution(resolution_path)
    out = tmp_path / "mapped"
    reports = map_actions.map_actions(
        tmp_path / "chunk_index.parquet",
        tmp_path / "actions",
        out,
        bind_overrides=overrides,
        bind_review_status=review,
        bind_resolution_meta=meta,
    )
    assert len(reports) == 1
    report = reports[0]
    assert report["bind_map"] == {
        "jump": [], "dash": ["south"], "grab": ["left_trigger"]
    }
    assert report["flagged"] is False
    assert report["bind_source"]["kind"] == "resolved"
    assert report["bind_source"]["resolution_sha256"]
    assert report["per_action"]["jump"]["review_status"] == "unresolved_no_legible_event"
    assert "inference" in report and "bind_map" in report["inference"]
    table = pq.read_table(
        out / "video_r" / "video_r_chunk_0000" / "labels_native.parquet"
    )
    assert not any(table["jump"].to_pylist())
    assert table["dash"].to_pylist() == [bool(v) for v in buttons["south"]]
    assert table["grab"].to_pylist() == [bool(v) for v in buttons["left_trigger"]]


def test_map_chunk_empty_bind_list_yields_all_negative_column() -> None:
    buttons = {
        button: np.zeros(4, dtype=np.bool_)
        for button in map_actions.BUTTON_COLUMNS
    }
    buttons["south"][1] = True
    raw = map_actions.RawChunk(
        buttons=buttons,
        j_left=np.zeros((4, 2), dtype=np.float64),
        j_right=np.zeros((4, 2), dtype=np.float64),
    )
    mapped = map_actions._map_chunk(
        raw,
        bind_map={"jump": [], "dash": ["south"], "grab": []},
        axis_threshold=0.5,
    )
    assert mapped["jump"].tolist() == [False, False, False, False]
    assert mapped["dash"].tolist() == [False, True, False, False]
    assert mapped["grab"].tolist() == [False, False, False, False]


def test_dpad_stick_disagreement_cannot_change_vertical_mapping() -> None:
    buttons = {
        button: np.zeros(6, dtype=np.bool_)
        for button in map_actions.BUTTON_COLUMNS
    }
    # Deliberately contradictory co-presses. D-pad is direct, while the stick
    # follows NitroGen's fixed upper-left (-1, -1) coordinate contract.
    buttons["dpad_down"][0] = True
    buttons["dpad_up"][1] = True
    j_left = np.asarray(
        [[0, -1], [0, 1], [0, -1], [0, 1], [0, -0.5], [0, 0.5]],
        dtype=np.float64,
    )
    raw = map_actions.RawChunk(
        buttons=buttons,
        j_left=j_left,
        j_right=np.zeros((6, 2), dtype=np.float64),
    )

    mapped = map_actions._directional_states(raw, 0.5)

    assert mapped["up"].tolist() == [True, True, True, False, False, False]
    assert mapped["down"].tolist() == [True, True, False, True, False, False]
