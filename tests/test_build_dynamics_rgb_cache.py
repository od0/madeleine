from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import cv2

import experiments.build_dynamics_rgb_cache as builder


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frames(count: int, offset: int) -> np.ndarray:
    frames = np.zeros((count, 128, 128, 3), dtype=np.uint8)
    for index in range(count):
        frames[index, 8:40, 4 + index : 20 + index, 0] = offset + 3 * index
        frames[index, 60:90, 30:70, 1] = offset
    return frames


def _shard(
    root: Path,
    session_id: str,
    *,
    count: int = 14,
    offset: int = 5,
    gap_at: int | None = None,
) -> Path:
    engine = np.arange(count, dtype=np.int64)
    if gap_at is not None:
        engine[gap_at:] += 5
    path = root / f"{session_id}.npz"
    np.savez_compressed(
        path,
        frames=_frames(count, offset),
        # Deliberately toxic values: the cache builder must never request this
        # member, validate its values, or use it for selection.
        keys=np.full((count, 7), 255, dtype=np.uint8),
        engine_frame_idx=engine,
        input_active=np.ones(count, dtype=np.uint8),
        session_id=np.asarray(session_id),
    )
    return path


def _patch_npz_nitrogen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route a synthetic NitroGen row through masked NPZ pixels, not a codec."""

    original_validate = builder._validate_row_without_access

    def validate_synthetic(row):
        if row["source"] == "nitrogen" and "path" in row:
            copy = dict(row)
            copy["source"] = "wild_provisional"
            return original_validate(copy)
        return original_validate(row)

    def build_synthetic_runs(rows_arg, max_horizon):
        runs = []
        eligible = {name: 0 for name in builder.SOURCE_NAMES}
        for index, row in enumerate(rows_arg):
            row_runs = builder._npz_runs(row, index, max_horizon)
            runs.extend(row_runs)
            eligible[row["source"]] += sum(
                run.length - max_horizon - 1 for run in row_runs
            )
        return runs, eligible

    original_row_path = builder._row_path
    original_row_digest = builder._row_digest

    def row_path(row):
        return Path(row["path"]) if "path" in row else original_row_path(row)

    def row_digest(row):
        return str(row["sha256"]) if "sha256" in row else original_row_digest(row)

    monkeypatch.setattr(builder, "_validate_row_without_access", validate_synthetic)
    monkeypatch.setattr(builder, "build_runs", build_synthetic_runs)
    monkeypatch.setattr(builder, "_row_path", row_path)
    monkeypatch.setattr(builder, "_row_digest", row_digest)
    monkeypatch.setattr(
        builder,
        "_decode_nitrogen_group",
        lambda source_rows, group, span: builder._load_npz_windows(
            source_rows[group[0].row_index], group, span
        ),
    )


def _three_source_inventory(tmp_path: Path) -> Path:
    rows = []
    for source, count, offset in (
        ("nitrogen", 24, 5),
        ("wild_provisional", 20, 35),
        ("local", 18, 65),
    ):
        session = f"{source}-train"
        path = _shard(tmp_path, session, count=count, offset=offset)
        rows.append(
            {
                "source": source,
                "session_id": session,
                "video_id": f"{source}-video",
                "frame_count": count,
                "path": str(path),
                "sha256": _sha(path),
                "masked": True,
            }
        )
    inventory = tmp_path / "synthetic-inventory.json"
    inventory.write_text(
        json.dumps({"schema_version": builder.INVENTORY_SCHEMA, "rows": rows}),
        encoding="utf-8",
    )
    return inventory


def test_forbidden_identity_rejected_before_any_path_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = {
        "schema_version": builder.INVENTORY_SCHEMA,
        "study_id": builder.STUDY_ID,
        "rows": [
            {
                "source": "local",
                "session_id": "rec_20260727_220000_test",
                "frame_count": 10,
                "path": str(tmp_path / "does-not-exist.npz"),
                "sha256": "0" * 64,
                "masked": True,
            }
        ],
    }
    touched = False

    def forbidden_stat(*_args, **_kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("filesystem must not be touched")

    monkeypatch.setattr(Path, "is_file", forbidden_stat)
    with pytest.raises(ValueError, match="forbidden evaluation identity"):
        builder.validate_inventory(inventory, production=False)
    assert touched is False


def test_npz_boundary_planner_never_requests_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _shard(tmp_path, "safe-train", gap_at=7)
    row = {
        "source": "local",
        "session_id": "safe-train",
        "frame_count": 14,
        "path": str(path),
        "sha256": _sha(path),
        "masked": True,
    }
    accessed: list[str] = []
    original = np.lib.npyio.NpzFile.__getitem__

    def tracking(self, key):
        accessed.append(str(key))
        if key == "keys":
            raise AssertionError("action member was accessed")
        return original(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", tracking)
    runs = builder._npz_runs(row, 0, 2)
    assert [run.length for run in runs] == [7, 7]
    windows = [builder.Window(0, 0, 0, 2, 2)]
    loaded = builder._load_npz_windows(row, windows, 4)
    assert loaded[0].shape == (4, 128, 128, 3)
    assert "keys" not in accessed
    assert set(accessed) <= {"engine_frame_idx", "input_active", "session_id", "frames"}


def test_canonical_inventory_sessions_join_nitrogen_video_metadata(tmp_path: Path) -> None:
    inventory = {
        "schema_version": builder.INVENTORY_SCHEMA,
        "nitrogen_videos": [
            {
                "video_id": "nitrogen-train-video",
                "video_path": str(tmp_path / "train.mp4"),
                "video_sha256": "1" * 64,
                "decoder_mode": "opencv_native_60hz",
                "source_width": 160,
                "source_height": 120,
                "mask_rect_source_xyxy": [120, 80, 160, 120],
                "mask_rect_128_xyxy": [95, 84, 128, 128],
            }
        ],
        "sessions": [
            {
                "source": "nitrogen",
                "session_id": "nitrogen-train-video__r000",
                "video_id": "nitrogen-train-video",
                "reference_shard": str(tmp_path / "reference.npz"),
                "reference_shard_sha256": "2" * 64,
                "frames": 12,
                "eligible_windows": 7,
                "engine_frame_start": 30,
                "engine_frame_end_exclusive": 42,
            }
        ],
    }
    rows = builder._normalize_inventory_rows(inventory, production=False)
    assert rows == [
        {
            "source": "nitrogen",
            "session_id": "nitrogen-train-video__r000",
            "frame_count": 12,
            "declared_eligible_windows": 7,
            "engine_frame_start": 30,
            "engine_frame_end_exclusive": 42,
            "video_id": "nitrogen-train-video",
            "raw_video_path": str(tmp_path / "train.mp4"),
            "raw_video_sha256": "1" * 64,
            "decoder_mode": "opencv_native_60hz",
            "source_frame_start": 30,
            "source_frame_end": 42,
            "source_resolution_wh": [160, 120],
            "source_mask_xyxy": [120, 80, 160, 120],
            "scaled_mask_xyxy": [95, 84, 128, 128],
            "reference_shard": str(tmp_path / "reference.npz"),
            "reference_shard_sha256": "2" * 64,
        }
    ]


def test_native_nitrogen_decoder_orders_selected_ranges_and_supermasks(tmp_path: Path) -> None:
    video = tmp_path / "train.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 60.0, (160, 120)
    )
    assert writer.isOpened()
    for index in range(12):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[..., 0] = 10 + index  # B
        frame[..., 1] = 30 + index  # G
        frame[..., 2] = 80 + index  # R
        frame[80:120, 120:160] = 255
        writer.write(frame)
    writer.release()
    row = {
        "source": "nitrogen",
        "session_id": "native-train__r000",
        "video_id": "native-train",
        "frame_count": 12,
        "raw_video_path": str(video),
        "raw_video_sha256": _sha(video),
        "decoder_mode": "opencv_native_60hz",
        "source_frame_start": 0,
        "source_frame_end": 12,
        "source_resolution_wh": [160, 120],
        "source_mask_xyxy": [120, 80, 160, 120],
        "scaled_mask_xyxy": [95, 84, 128, 128],
    }
    decoded = builder._decode_native_intervals(row, [(2, 6), (8, 10)])
    assert list(decoded) == [(2, 6), (8, 10)]
    assert decoded[(2, 6)].shape == (4, 128, 128, 3)
    assert np.all(decoded[(2, 6)][:, 84:128, 95:128] == 0)
    # RGB conversion is observable away from the mask: red exceeds blue.
    assert int(decoded[(2, 6)][0, 10, 10, 0]) > int(
        decoded[(2, 6)][0, 10, 10, 2]
    )


def test_source_proportional_sampling_and_gap_safe_geometry(tmp_path: Path) -> None:
    paths = {
        source: _shard(tmp_path, f"{source}-train", count=count, offset=offset, gap_at=gap)
        for source, count, offset, gap in (
            ("nitrogen", 24, 5, None),
            ("wild_provisional", 18, 25, None),
            ("local", 16, 55, 8),
        )
    }
    rows = [
        {
            "source": source,
            "session_id": f"{source}-train",
            "video_id": f"{source}-video",
            "frame_count": 24 if source == "nitrogen" else 18 if source == "wild_provisional" else 16,
            "path": str(paths[source]),
            "sha256": _sha(paths[source]),
            "masked": True,
        }
        for source in builder.SOURCE_NAMES
    ]
    rows[0]["raw_video_path"] = rows[0]["path"]
    rows[0]["raw_video_sha256"] = rows[0]["sha256"]
    # Exercise the common run/window logic through NPZs, including a split run.
    runs = []
    eligible = {source: 0 for source in builder.SOURCE_NAMES}
    for index, row in enumerate(rows):
        row_runs = builder._npz_runs(row, index, 4)
        runs.extend(row_runs)
        eligible[row["source"]] += sum(run.length - 5 for run in row_runs)
    allocation = builder.proportional_allocation(eligible, 12)
    assert sum(allocation.values()) == 12
    windows = builder.build_windows(
        rows, runs, eligible, allocation, max_horizon=4, seed=99
    )
    assert [window.window_id for window in windows] == list(range(12))
    for window in windows:
        row = rows[window.row_index]
        with np.load(row["path"], allow_pickle=False) as archive:
            engine = archive["engine_frame_idx"]
            start = window.anchor_local - 1
            assert np.all(np.diff(engine[start : start + 6]) == 1)


def test_end_to_end_cache_is_deterministic_atomic_and_indexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Use three NPZ sources end-to-end. The tiny test replaces only the
    # production NitroGen raw decoder with the identical masked-NPZ loader.
    inventory = _three_source_inventory(tmp_path)
    _patch_npz_nitrogen(monkeypatch)

    output = tmp_path / "cache"
    manifest = builder.build_cache(
        inventory_path=inventory,
        output=output,
        horizons=(1, 2, 4),
        window_count=12,
        seed=17,
        production=False,
    )
    assert manifest["schema_version"] == builder.CACHE_SCHEMA
    assert manifest["labels"]["loaded"] is False
    assert manifest["labels"]["arrays_accessed"] == []
    assert manifest["window_span_frames"] == 6
    assert manifest["windows"] == 12
    assert manifest["tuples"] == 36
    assert not output.with_name("cache.partial").exists()

    rgb = np.load(output / "rgb.npy", mmap_mode="r")
    with np.load(output / "index.npz", allow_pickle=False) as index:
        assert rgb.shape == (72, 128, 128, 3)
        assert len(index["tuple_id"]) == 36
        assert len(np.unique(index["tuple_id"])) == 36
        assert np.all(index["online_current"] == index["online_previous"] + 1)
        assert np.all(index["target_previous"] == index["online_previous"] + index["horizon"])
        assert np.all(index["target_current"] == index["target_previous"] + 1)
        assert set(index["horizon"].tolist()) == {1, 2, 4}
        assert set(index["source_names"].tolist()) == set(builder.SOURCE_NAMES)
        first_tuple_ids = index["tuple_id"].copy()

    # Existing complete output is validated and returned, not rebuilt.
    second = builder.build_cache(
        inventory_path=inventory,
        output=output,
        horizons=(1, 2, 4),
        window_count=12,
        seed=17,
        production=False,
    )
    assert second["artifacts"] == manifest["artifacts"]
    with np.load(output / "index.npz", allow_pickle=False) as index:
        assert np.array_equal(index["tuple_id"], first_tuple_ids)

    receipt = builder.validate_cache(output)
    assert receipt["status"] == "valid"
    assert receipt["rgb_sha256"] == manifest["artifacts"]["rgb"]["sha256"]
    assert builder.main(["--validate-only", "--output", str(output)]) == 0
    assert '"status": "valid"' in capsys.readouterr().out

    # Validation-only rehashes bytes and refuses silent content drift.
    with (output / "rgb.npy").open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="rgb artifact changed"):
        builder.validate_cache(output)


def test_interrupted_build_resumes_only_the_identical_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = _three_source_inventory(tmp_path)
    _patch_npz_nitrogen(monkeypatch)
    original_atomic_json = builder._atomic_json
    interrupted = False

    def interrupt_after_progress(path, value):
        nonlocal interrupted
        original_atomic_json(path, value)
        if (
            not interrupted
            and path.name == "state.json"
            and int(value.get("completed_windows", 0)) > 0
        ):
            interrupted = True
            raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(builder, "_atomic_json", interrupt_after_progress)
    output = tmp_path / "resumed-cache"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        builder.build_cache(
            inventory_path=inventory,
            output=output,
            horizons=(1, 2, 4),
            window_count=12,
            seed=23,
            production=False,
        )
    partial = output.with_name(output.name + ".partial")
    state = json.loads((partial / "state.json").read_text())
    assert 0 < state["completed_windows"] < 12

    # The same plan resumes and publishes atomically.
    monkeypatch.setattr(builder, "_atomic_json", original_atomic_json)
    resumed = builder.build_cache(
        inventory_path=inventory,
        output=output,
        horizons=(1, 2, 4),
        window_count=12,
        seed=23,
        production=False,
    )
    assert resumed["resume"]["completed_windows"] == 12
    assert not partial.exists()

    # A partial cache is fail-closed under any changed seed/plan.
    mismatch = tmp_path / "mismatch-cache"
    mismatch_partial = mismatch.with_name(mismatch.name + ".partial")
    mismatch_partial.mkdir()
    wrong_state = {
        "schema_version": builder.STATE_SCHEMA,
        "inventory_sha256": _sha(inventory),
        "plan_sha256": "0" * 64,
        "horizons": [1, 2, 4],
        "span": 6,
        "windows": 12,
        "completed_windows": 1,
    }
    np.lib.format.open_memmap(
        mismatch_partial / "rgb.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(72, 128, 128, 3),
    ).flush()
    (mismatch_partial / "state.json").write_text(json.dumps(wrong_state))
    with pytest.raises(ValueError, match="another build contract"):
        builder.build_cache(
            inventory_path=inventory,
            output=mismatch,
            horizons=(1, 2, 4),
            window_count=12,
            seed=23,
            production=False,
        )


def test_cli_help_and_validation_only_contract() -> None:
    help_text = builder.build_parser().format_help()
    for option in (
        "--inventory",
        "--output",
        "--horizons",
        "--window-count",
        "--seed",
        "--validate-only",
    ):
        assert option in help_text


def test_production_population_requires_exact_train_only_membership(tmp_path: Path) -> None:
    path = _shard(tmp_path, "one-own")
    inventory = {
        "schema_version": builder.INVENTORY_SCHEMA,
        "study_id": builder.STUDY_ID,
        "labels_consumed": False,
        "forbidden_exclusion_proof": {
            "sealed_untouched_absent": True,
            "whole_y4n_absent": True,
            "own_val_a_absent": True,
            "B1_absent": True,
            "val_B_absent": True,
            "checked_before_cache_RGB_access": True,
        },
        "nitrogen_videos": [],
        "sessions": [
            {
                "source": "local",
                "session_id": "one-own",
                "frames": 14,
                "active_frames": 14,
                "eligible_windows": 9,
                "engine_frame_start": 0,
                "engine_frame_end_exclusive": 14,
                "shard_path": str(path),
                "shard_sha256": _sha(path),
            }
        ],
    }
    inventory["inventory_content_sha256"] = builder._canonical_sha256(inventory)
    with pytest.raises(ValueError, match="nitrogen production population changed"):
        builder.validate_inventory(inventory, production=True)
