"""Contract pins from the NitroGen mapping code audit.

These tests freeze the invariants the audit verified end to end:

- the vertical direction rule follows NitroGen's fixed upper-left ``(-1,-1)``
  coordinate contract with strict comparisons, for both stick dtypes;
- the written ``labels_native.parquet`` column order equals the frozen
  ``data.schema.KEY_ORDER``, which downstream ``keys.npy`` consumers use
  positionally;
- the v2 mapping report carries the invariant direction rule and no
  per-video axis-sign state;
- contradictory d-pad/stick rows keep both OR contributions instead of
  letting one channel rewrite the other.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from data.schema import KEY_ORDER, LABELS_NATIVE_COLUMNS
from nitrogen import map_actions


CANARIES = [
    # (j_left_y, dpad_up, dpad_down) -> expected (up, down)
    ((0, -1.0), 0, 0, (True, False)),    # negative analog only: up
    ((0, 1.0), 0, 0, (False, True)),     # positive analog only: down
    ((0, 0.0), 1, 0, (True, False)),     # d-pad up only
    ((0, 0.0), 0, 1, (False, True)),     # d-pad down only
    ((0, -0.9), 1, 0, (True, False)),    # agreement
    ((0, -0.9), 0, 1, (True, True)),     # disagreement keeps both channels
    ((0, -0.5), 0, 0, (False, False)),   # exact threshold is exclusive
    ((0, 0.5), 0, 0, (False, False)),
    ((0, 0.0), 0, 0, (False, False)),    # centered stick
]


def _canary_chunk() -> map_actions.RawChunk:
    n = len(CANARIES)
    buttons = {
        button: np.zeros(n, dtype=np.bool_) for button in map_actions.BUTTON_COLUMNS
    }
    j_left = np.zeros((n, 2), dtype=np.float64)
    for row, (stick, dpad_up, dpad_down, _) in enumerate(CANARIES):
        j_left[row] = stick
        buttons["dpad_up"][row] = bool(dpad_up)
        buttons["dpad_down"][row] = bool(dpad_down)
    return map_actions.RawChunk(
        buttons=buttons, j_left=j_left, j_right=np.zeros((n, 2), dtype=np.float64)
    )


def test_vertical_canaries_follow_upper_left_contract() -> None:
    states = map_actions._directional_states(_canary_chunk(), 0.5)
    for row, (_, _, _, expected) in enumerate(CANARIES):
        assert (bool(states["up"][row]), bool(states["down"][row])) == expected, row


def test_key_order_vertical_indices_are_frozen() -> None:
    # keys.npy consumers index columns positionally; the audit verified the
    # whole chain assumes this exact layout.
    assert KEY_ORDER == ["left", "right", "up", "down", "jump", "dash", "grab"]
    assert KEY_ORDER.index("up") == 2
    assert KEY_ORDER.index("down") == 3


def _write_raw_chunk(path: Path, stick_type: pa.DataType) -> None:
    n = 600
    buttons = {
        button: np.zeros(n, dtype=np.int64) for button in map_actions.BUTTON_COLUMNS
    }
    j_left = np.zeros((n, 2), dtype=np.float64)
    # Direction canaries expressible in both integer and float stick dtypes.
    j_left[10, 1] = -1.0
    j_left[11, 1] = 1.0
    buttons["dpad_up"][12] = 1
    buttons["dpad_down"][13] = 1
    j_left[14, 1] = -1.0
    buttons["dpad_down"][14] = 1
    if pa.types.is_integer(stick_type):
        j_left_values = j_left.astype(np.int64)
        j_right_values = np.zeros((n, 2), dtype=np.int64)
    else:
        j_left_values = j_left
        j_right_values = np.zeros((n, 2), dtype=np.float64)
    arrays = [
        pa.array(buttons[button], type=pa.int64())
        for button in map_actions.BUTTON_COLUMNS
    ]
    arrays.append(pa.array(j_left_values.tolist(), type=pa.list_(stick_type)))
    arrays.append(pa.array(j_right_values.tolist(), type=pa.list_(stick_type)))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_arrays(arrays, names=[*map_actions.BUTTON_COLUMNS, "j_left", "j_right"]),
        path,
    )


def _run_mapper(tmp_path: Path, stick_type: pa.DataType, tag: str):
    root = tmp_path / tag
    chunk_dir = root / "actions" / "SHARD_0000" / "video_x" / "video_x_chunk_0000"
    _write_raw_chunk(chunk_dir / "actions_raw.parquet", stick_type)
    index_path = root / "chunk_index.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "video_id": "video_x",
                    "chunk_id": 0,
                    "chunk_size": 600,
                    "grid_hz": 60.0,
                    "controller_type": "xboxone",
                }
            ]
        ),
        index_path,
    )
    out = root / "mapped"
    reports = map_actions.map_actions(index_path, root / "actions", out)
    assert len(reports) == 1
    table = pq.read_table(out / "video_x" / "video_x_chunk_0000" / "labels_native.parquet")
    return reports[0], table


def test_end_to_end_written_labels_and_column_order(tmp_path: Path) -> None:
    report, table = _run_mapper(tmp_path, pa.float64(), "float_sticks")

    assert table.schema.names == LABELS_NATIVE_COLUMNS
    assert table.schema.names == ["frame_idx", *KEY_ORDER]

    up = table["up"].to_pylist()
    down = table["down"].to_pylist()
    assert (up[10], down[10]) == (True, False)
    assert (up[11], down[11]) == (False, True)
    assert (up[12], down[12]) == (True, False)
    assert (up[13], down[13]) == (False, True)
    assert (up[14], down[14]) == (True, True)

    rule = report["direction_rule"]
    assert rule["up"] == "dpad_up OR j_left_y < -axis_threshold"
    assert rule["down"] == "dpad_down OR j_left_y > axis_threshold"
    assert rule["source_revision"] == map_actions.NITROGEN_COORDINATE_CONTRACT_REVISION
    assert rule["comparisons"] == "strict"
    # No per-video vertical sign state may reappear in reports.
    assert not any(key.startswith("axis_sign") for key in report)


def test_integer_and_float_sticks_produce_identical_labels(tmp_path: Path) -> None:
    _, float_table = _run_mapper(tmp_path, pa.float64(), "float_sticks")
    _, int_table = _run_mapper(tmp_path, pa.int64(), "int_sticks")
    for key in KEY_ORDER:
        assert float_table[key].to_pylist() == int_table[key].to_pylist(), key
