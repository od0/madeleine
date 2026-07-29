from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from badeline.eval import select_checkpoint_state
from badeline.model import BadelineIDM
from badeline.train import (
    DeterministicSourceBatchSampler,
    SegmentSessionDataset,
    SessionArrays,
    WindowedSessionDataset,
    build_source_batch_sampler,
    contiguous_runs,
    load_session,
    resolve_positive_weight,
    run_training,
)
from data.schema import KEY_ORDER


def _scripted_keys(frame_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keys = np.zeros((frame_count, len(KEY_ORDER)), dtype=np.uint8)
    phase_shift = int(rng.integers(0, 12))
    for frame in range(frame_count):
        phase = (frame + phase_shift) % 120
        keys[frame, 1] = phase < 45
        keys[frame, 0] = 52 <= phase < 97
        keys[frame, 4] = 24 <= phase < 32 or 82 <= phase < 90
        keys[frame, 2] = 100 <= phase < 106
        keys[frame, 3] = 109 <= phase < 115
        keys[frame, 5] = 40 <= phase < 44
        keys[frame, 6] = 72 <= phase < 79
    return keys


def _make_session(path: Path, session_index: int) -> str:
    session_id = f"toy-{session_index}"
    frame_count = 600
    keys = _scripted_keys(frame_count, 10_000 + session_index)
    rng = np.random.default_rng(20_000 + session_index)
    frames = np.zeros((frame_count, 128, 128, 3), dtype=np.uint8)

    x = float(rng.integers(24, 88))
    ground_y = 88.0
    y = ground_y
    velocity_y = 0.0
    for frame in range(frame_count):
        left, right, _, _, jump, _, _ = keys[frame]
        x = float(np.clip(x + 2.0 * (int(right) - int(left)), 0, 112))
        if jump and y >= ground_y:
            velocity_y = -5.0
        velocity_y += 0.45
        y = min(ground_y, y + velocity_y)
        if y >= ground_y:
            velocity_y = 0.0

        x0 = int(round(x))
        y0 = int(round(y))
        frames[frame, y0 : y0 + 16, x0 : x0 + 16] = 255

    np.savez(
        path / f"{session_id}.npz",
        frames=frames,
        keys=keys,
        engine_frame_idx=np.arange(frame_count, dtype=np.int64),
        input_active=np.ones(frame_count, dtype=np.uint8),
        session_id=np.asarray(session_id),
    )
    return session_id


@pytest.fixture(scope="module")
def synthetic_shards(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[str]]:
    data_dir = tmp_path_factory.mktemp("badeline-shards")
    ids = [_make_session(data_dir, index) for index in range(4)]
    return data_dir, ids


def _write_lines(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def test_overlap_error(synthetic_shards: tuple[Path, list[str]], tmp_path: Path) -> None:
    data_dir, ids = synthetic_shards
    train_ids = tmp_path / "train.txt"
    val_ids = tmp_path / "val.txt"
    config = tmp_path / "config.json"
    _write_lines(train_ids, [ids[0], ids[1]])
    _write_lines(val_ids, [ids[1]])
    config.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="overlapping split"):
        run_training(
            data_dir=data_dir,
            train_sessions=train_ids,
            val_sessions=val_ids,
            config_path=config,
            out_dir=tmp_path / "out",
            max_steps=0,
            device_name="cpu",
        )


@pytest.mark.parametrize("window_mode", ["centered", "past_only"])
@pytest.mark.parametrize(
    "input_config", ["pixels", "history", "pixels_plus_history"]
)
@pytest.mark.parametrize("window", [2, 16])
def test_shapes_all_configs(
    window_mode: str, input_config: str, window: int
) -> None:
    config = {
        "window": window,
        "window_mode": window_mode,
        "input_config": input_config,
        "history_len": 8,
        "embedding_dim": 16,
        "temporal_dim": 16,
        "spatial_size": 2,
    }
    model = BadelineIDM(config).eval()
    batch_size = 1
    batch: dict[str, torch.Tensor] = {}
    if input_config in ("pixels", "pixels_plus_history"):
        batch["frames"] = torch.rand(batch_size, window, 3, 64, 64)
    if input_config in ("history", "pixels_plus_history"):
        batch["history"] = torch.rand(batch_size, 8, len(KEY_ORDER))
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (batch_size, len(KEY_ORDER))
    assert model.key_order == KEY_ORDER


def test_feature_delta_path_shape() -> None:
    config = {
        "window": 4,
        "window_mode": "centered",
        "input_config": "pixels",
        "embedding_dim": 16,
        "temporal_dim": 16,
        "feature_deltas": True,
    }
    model = BadelineIDM(config).eval()
    with torch.no_grad():
        logits = model({"frames": torch.rand(2, 4, 3, 128, 128)})
        segment_logits = model.forward_segment({
            "frames": torch.rand(2, 8, 3, 128, 128)
        })
    assert logits.shape == (2, len(KEY_ORDER))
    assert segment_logits.shape == (2, 5, len(KEY_ORDER))


def test_vpt_range_augmentation_is_temporally_consistent() -> None:
    config = {
        "window": 4,
        "input_config": "pixels",
        "pretrained_encoder": True,
        "trainable_encoder": True,
        "video_augmentation": True,
        "embedding_dim": 8,
        "temporal_dim": 8,
    }
    model = BadelineIDM(config).train()
    repeated = torch.rand(2, 1, 3, 128, 128).expand(-1, 4, -1, -1, -1)
    augmented = model._augment_frames(repeated)

    assert augmented.shape == repeated.shape
    assert torch.equal(augmented[:, 0], augmented[:, 1])
    assert any(
        parameter.requires_grad
        for parameter in model.frame_encoder.features.parameters()
    )


def test_augmentation_rejects_precomputed_features() -> None:
    with pytest.raises(ValueError, match="requires pixels"):
        BadelineIDM({
            "window": 4,
            "input_config": "pixels",
            "precomputed_features": True,
            "video_augmentation": True,
        })


def test_feature_delta_segment_matches_individual_dilated_windows() -> None:
    torch.manual_seed(3)
    config = {
        "window": 4,
        "frame_stride": 3,
        "window_mode": "centered",
        "input_config": "pixels",
        "precomputed_features": True,
        "backbone_feature_dim": 12,
        "embedding_dim": 8,
        "temporal_dim": 8,
        "feature_deltas": True,
    }
    model = BadelineIDM(config).eval()
    features = torch.rand(14, 12)  # span 10, hence five target windows

    with torch.no_grad():
        individual = torch.cat([
            model({"features": features[start : start + 10 : 3].unsqueeze(0)})
            for start in range(5)
        ])
        segment = model.forward_segment({"features": features.unsqueeze(0)})[0]

    assert segment.shape == (5, len(KEY_ORDER))
    assert torch.allclose(individual, segment, atol=1e-6)


def test_precomputed_feature_path_shapes() -> None:
    config = {
        "window": 4,
        "window_mode": "centered",
        "input_config": "pixels",
        "precomputed_features": True,
        "backbone_feature_dim": 12,
        "embedding_dim": 8,
        "temporal_dim": 8,
        "feature_deltas": True,
    }
    model = BadelineIDM(config).eval()
    with torch.no_grad():
        per_window = model({"features": torch.rand(2, 4, 12)})
        segment = model.forward_segment({"features": torch.rand(2, 8, 12)})
    assert per_window.shape == (2, len(KEY_ORDER))
    assert segment.shape == (2, 5, len(KEY_ORDER))


def test_precomputed_feature_shard_loading(tmp_path: Path) -> None:
    session_id = "feature-toy"
    frame_count = 160
    features = np.random.default_rng(4).normal(
        size=(frame_count, 12)
    ).astype(np.float16)
    keys = _scripted_keys(frame_count, 5)
    np.savez(
        tmp_path / f"{session_id}.npz",
        features=features,
        keys=keys,
        engine_frame_idx=np.arange(frame_count, dtype=np.int64),
        input_active=np.ones(frame_count, dtype=np.uint8),
        session_id=np.asarray(session_id),
    )

    session = load_session(
        tmp_path, session_id, precomputed_features=True
    )
    dataset = SegmentSessionDataset(
        [session],
        window=4,
        window_mode="centered",
        input_config="pixels",
        history_len=8,
        segment_windows=32,
        precomputed_features=True,
    )
    item = dataset[0]
    assert item["features"].shape == (35, 12)
    assert item["features"].dtype == torch.float32
    assert item["target"].shape == (32, len(KEY_ORDER))


def test_finetune_initialization_round_trip(
    synthetic_shards: tuple[Path, list[str]], tmp_path: Path
) -> None:
    data_dir, ids = synthetic_shards
    train_ids = tmp_path / "train.txt"
    val_ids = tmp_path / "val.txt"
    config_path = tmp_path / "config.json"
    _write_lines(train_ids, ids[:3])
    _write_lines(val_ids, ids[3:])
    config_path.write_text(json.dumps({
        "window": 2,
        "input_config": "history",
        "history_len": 8,
        "embedding_dim": 16,
        "temporal_dim": 16,
        "class_balance": True,
    }))

    first = run_training(
        data_dir=data_dir, train_sessions=train_ids, val_sessions=val_ids,
        config_path=config_path, out_dir=tmp_path / "first", max_steps=0,
        device_name="cpu",
    )
    second = run_training(
        data_dir=data_dir, train_sessions=train_ids, val_sessions=val_ids,
        config_path=config_path, out_dir=tmp_path / "second", max_steps=0,
        device_name="cpu", init_checkpoint=first,
    )

    first_state = torch.load(first, map_location="cpu", weights_only=True)
    second_state = torch.load(second, map_location="cpu", weights_only=True)
    for name, value in first_state["model_state_dict"].items():
        assert torch.equal(value, second_state["model_state_dict"][name])
    meta = json.loads((tmp_path / "second" / "run_meta.json").read_text())
    assert meta["initialized_from"] == str(first)
    assert meta["positive_weight"] is not None


def test_exact_source_cycle_is_deterministic_and_receipted() -> None:
    cycle = [
        {"nitrogen": 11, "wild": 3, "local": 2},
        {"nitrogen": 11, "wild": 3, "local": 2},
        {"nitrogen": 11, "wild": 3, "local": 2},
        {"nitrogen": 12, "wild": 3, "local": 1},
        {"nitrogen": 11, "wild": 4, "local": 1},
    ]
    source_items = {
        "nitrogen": list(range(0, 100)),
        "wild": list(range(100, 120)),
        "local": list(range(120, 127)),
    }

    def sample() -> tuple[list[list[int]], dict[str, object]]:
        sampler = DeterministicSourceBatchSampler(
            source_items,
            step_cycle=cycle,
            steps=10,
            seed=17,
            expected_cycle_steps=5,
            expected_cycle_items=80,
            source_session_counts={"nitrogen": 92, "wild": 2058, "local": 3},
        )
        batches = list(sampler)
        receipt = sampler.receipt(require_complete=True)
        with pytest.raises(RuntimeError, match="one-shot"):
            list(sampler)
        return batches, receipt

    first_batches, first = sample()
    second_batches, second = sample()
    assert first_batches == second_batches
    assert first == second
    assert all(len(batch) == 16 for batch in first_batches)
    sources = first["sources"]
    assert sources["nitrogen"]["actual_draws"] == 112
    assert sources["wild"]["actual_draws"] == 32
    assert sources["local"]["actual_draws"] == 16
    assert sources["local"]["unique_segment_items_drawn"] == 7
    assert sources["local"]["minimum_draws_per_item"] == 2
    assert sources["local"]["maximum_draws_per_item"] == 3
    assert first["complete"] is True


def test_exact_nitrogen_local_cycle_emits_90_10() -> None:
    sampler = DeterministicSourceBatchSampler(
        {
            "nitrogen": list(range(100)),
            "local": list(range(100, 107)),
        },
        step_cycle=[
            {"nitrogen": 14, "local": 2},
            {"nitrogen": 14, "local": 2},
            {"nitrogen": 14, "local": 2},
            {"nitrogen": 15, "local": 1},
            {"nitrogen": 15, "local": 1},
        ],
        steps=5,
        seed=0,
        expected_cycle_steps=5,
        expected_cycle_items=80,
    )
    assert len(list(sampler)) == 5
    receipt = sampler.receipt(require_complete=True)
    assert receipt["sources"]["nitrogen"]["actual_draws"] == 72
    assert receipt["sources"]["local"]["actual_draws"] == 8


def test_source_sampling_rejects_session_membership_mismatch() -> None:
    sessions = [
        SessionArrays(
            session_id,
            np.zeros((20, 4), dtype=np.float16),
            np.zeros((20, len(KEY_ORDER)), dtype=np.uint8),
        )
        for session_id in ("nitrogen-a", "local-a")
    ]
    dataset = SegmentSessionDataset(
        sessions,
        window=2,
        window_mode="centered",
        input_config="pixels",
        history_len=8,
        segment_windows=4,
        precomputed_features=True,
    )
    config = {
        "source_sampling": {
            "format_version": "madeleine.source-balanced-batch.v1",
            "expected_steps": 5,
            "cycle_steps": 5,
            "cycle_items": 10,
            "sources": {"nitrogen": ["nitrogen-a"]},
            "step_cycle": [{"nitrogen": 2}] * 5,
        }
    }
    with pytest.raises(ValueError, match="session membership mismatch"):
        build_source_batch_sampler(
            dataset,
            ["nitrogen-a", "local-a"],
            config,
            steps=5,
            seed=0,
            expected_batch_items=2,
        )


def test_frozen_positive_weight_is_exact_and_fail_closed() -> None:
    frozen = {
        "left": 6.5241737274432445,
        "right": 2.165329049329906,
        "up": 5.234555340345659,
        "down": 10.0,
        "jump": 6.478684563686479,
        "dash": 10.0,
        "grab": 1.7741724096558231,
    }
    expected = [frozen[key] for key in KEY_ORDER]
    assert resolve_positive_weight(
        {
            "class_balance": True,
            "class_balance_max": 10.0,
            "frozen_positive_weight": frozen,
        },
        [],
    ) == expected
    with pytest.raises(ValueError, match="class_balance=true"):
        resolve_positive_weight({"frozen_positive_weight": frozen}, [])
    missing = dict(frozen)
    missing.pop("dash")
    with pytest.raises(ValueError, match="canonical key set"):
        resolve_positive_weight({
            "class_balance": True,
            "frozen_positive_weight": missing,
        }, [])


def test_source_sampling_training_writes_completed_receipt(tmp_path: Path) -> None:
    data_dir = tmp_path / "features"
    data_dir.mkdir()
    train_ids = ["nitrogen-a", "local-a"]
    val_ids = ["mapped-val"]
    for offset, session_id in enumerate([*train_ids, *val_ids]):
        frame_count = 40
        np.savez(
            data_dir / f"{session_id}.npz",
            features=np.random.default_rng(offset).normal(
                size=(frame_count, 4)
            ).astype(np.float16),
            keys=_scripted_keys(frame_count, offset),
            engine_frame_idx=np.arange(frame_count, dtype=np.int64),
            input_active=np.ones(frame_count, dtype=np.uint8),
            session_id=np.asarray(session_id),
        )
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    _write_lines(train_path, train_ids)
    _write_lines(val_path, val_ids)
    config = {
        "window": 2,
        "window_mode": "centered",
        "input_config": "pixels",
        "precomputed_features": True,
        "backbone_feature_dim": 4,
        "embedding_dim": 4,
        "temporal_dim": 4,
        "segment_windows": 4,
        "batch_size": 8,
        "eval_batch_size": 8,
        "eval_interval": 5,
        "max_steps": 5,
        "initial_train_eval": False,
        "class_balance": True,
        "class_balance_max": 10.0,
        "frozen_positive_weight": {key: 2.0 for key in KEY_ORDER},
        "source_sampling": {
            "format_version": "madeleine.source-balanced-batch.v1",
            "expected_steps": 5,
            "cycle_steps": 5,
            "cycle_items": 10,
            "sources": {
                "nitrogen": ["nitrogen-a"],
                "local": ["local-a"],
            },
            "step_cycle": [
                {"nitrogen": 1, "local": 1},
                {"nitrogen": 1, "local": 1},
                {"nitrogen": 1, "local": 1},
                {"nitrogen": 1, "local": 1},
                {"nitrogen": 1, "local": 1},
            ],
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "out"
    checkpoint_path = run_training(
        data_dir=data_dir,
        train_sessions=train_path,
        val_sessions=val_path,
        config_path=config_path,
        out_dir=output,
        device_name="cpu",
    )

    receipt = json.loads((output / "source_sampling_receipt.json").read_text())
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    run_meta = json.loads((output / "run_meta.json").read_text())
    assert receipt["complete"] is True
    assert receipt["actual_steps"] == 5
    assert receipt["sources"]["local"]["actual_draws"] == 5
    assert checkpoint["positive_weight"] == [2.0] * len(KEY_ORDER)
    assert checkpoint["source_sampling_receipt"] == receipt
    assert run_meta["source_sampling"] == receipt


def test_eval_checkpoint_selection_is_explicit() -> None:
    checkpoint = {
        "model_state_dict": {"value": "selected"},
        "final_state_dict": {"value": "final"},
    }
    assert select_checkpoint_state(checkpoint, "selected") == {
        "value": "selected"
    }
    assert select_checkpoint_state(checkpoint, "final") == {"value": "final"}
    with pytest.raises(KeyError, match="final_state_dict"):
        select_checkpoint_state({"model_state_dict": {}}, "final")
    with pytest.raises(ValueError, match="unsupported"):
        select_checkpoint_state(checkpoint, "mystery")


def test_converges_mps_or_cpu(
    synthetic_shards: tuple[Path, list[str]], tmp_path: Path
) -> None:
    data_dir, ids = synthetic_shards
    train_ids = tmp_path / "train.txt"
    val_ids = tmp_path / "val.txt"
    config_path = tmp_path / "config.json"
    output = tmp_path / "out"
    _write_lines(train_ids, ids[:3])
    _write_lines(val_ids, ids[3:])
    config = {
        "window": 2,
        "window_mode": "centered",
        "input_config": "pixels",
        "history_len": 8,
        "embedding_dim": 64,
        "temporal_dim": 64,
        "batch_size": 32,
        "eval_batch_size": 64,
        "eval_interval": 50,
        "seed": 7,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    command = [
        sys.executable,
        "-m",
        "badeline.train",
        "--data",
        str(data_dir),
        "--train-sessions",
        str(train_ids),
        "--val-sessions",
        str(val_ids),
        "--config",
        str(config_path),
        "--out",
        str(output),
        "--max-steps",
        "200",
        "--device",
        device,
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr

    records = [
        json.loads(line)
        for line in (output / "log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    initial = np.mean(list(records[0]["val_bce_per_key"].values()))
    final = np.mean(list(records[-1]["val_bce_per_key"].values()))
    assert final < 0.35
    assert final < 0.60 * initial
    assert (output / "model.pt").is_file()
    assert json.loads((output / "config.json").read_text(encoding="utf-8")) == config


def test_past_only_shifts_target() -> None:
    frames = np.zeros((6, 128, 128, 3), dtype=np.uint8)
    keys = np.zeros((6, len(KEY_ORDER)), dtype=np.uint8)
    keys[3:, KEY_ORDER.index("jump")] = 1
    session = SessionArrays("flip", frames, keys)

    centered = WindowedSessionDataset(
        [session],
        window=4,
        window_mode="centered",
        input_config="pixels",
        history_len=8,
    )
    past_only = WindowedSessionDataset(
        [session],
        window=4,
        window_mode="past_only",
        input_config="pixels",
        history_len=8,
    )
    jump_index = KEY_ORDER.index("jump")
    assert centered[0]["target"][jump_index].item() == 0.0
    assert past_only[0]["target"][jump_index].item() == 1.0


def test_windows_never_cross_gaps_and_inactive_targets_are_filtered() -> None:
    frames = np.zeros((7, 128, 128, 3), dtype=np.uint8)
    keys = np.zeros((7, len(KEY_ORDER)), dtype=np.uint8)
    keys[:, KEY_ORDER.index("jump")] = np.arange(7) % 2
    engine_frame_idx = np.asarray([0, 1, 2, 10, 11, 12, 13], dtype=np.int64)
    input_active = np.asarray([1, 0, 1, 1, 1, 1, 1], dtype=np.uint8)
    session = SessionArrays(
        "gapped", frames, keys, engine_frame_idx, input_active
    )

    dataset = WindowedSessionDataset(
        [session], window=2, window_mode="centered",
        input_config="pixels", history_len=2,
    )

    # Run 0 contributes start 0 only (start 1 has an inactive target); run 1
    # contributes starts 3, 4 and 5. No start can bridge engine frames 2 -> 10.
    assert len(dataset) == 4
    starts = [location[1] for location in dataset._locations]
    assert starts == [0, 3, 4, 5]


def test_contiguous_runs_split_engine_counter_resets() -> None:
    assert contiguous_runs(np.asarray([5, 6, 7, 0, 1, 1, 2])) == [
        (0, 3), (3, 5), (5, 7)
    ]


def test_transition_weight_marks_onsets_and_offsets_per_key() -> None:
    frames = np.zeros((8, 128, 128, 3), dtype=np.uint8)
    keys = np.zeros((8, len(KEY_ORDER)), dtype=np.uint8)
    jump = KEY_ORDER.index("jump")
    keys[3:6, jump] = 1
    session = SessionArrays("events", frames, keys)
    dataset = WindowedSessionDataset(
        [session], window=2, window_mode="centered",
        input_config="pixels", history_len=2, transition_weight=7.0,
    )

    # Window-2 centered targets its first frame, so items 3 and 6 are the
    # jump onset and offset respectively.
    assert dataset[3]["loss_weight"][jump].item() == 7.0
    assert dataset[6]["loss_weight"][jump].item() == 7.0
    assert dataset[4]["loss_weight"][jump].item() == 1.0


# --- segment path (brief v3.2 shared-window-encoding) ---

from badeline.train import SegmentSessionDataset, history_block  # noqa: E402


def test_forward_segment_matches_forward_per_window() -> None:
    torch.manual_seed(0)
    config = {
        "window": 4, "window_mode": "centered",
        "input_config": "pixels_plus_history",
        "history_len": 3, "embedding_dim": 16, "temporal_dim": 16,
    }
    model = BadelineIDM(config).eval()
    frames = torch.rand(10, 3, 128, 128)          # 10 unique frames → 7 windows
    history = torch.rand(7, 3, 7)                  # one history row per window

    with torch.no_grad():
        per_window = torch.cat([
            model({
                "frames": frames[s : s + 4].unsqueeze(0),
                "history": history[s].unsqueeze(0),
            })
            for s in range(7)
        ])
        segment = model.forward_segment({
            "frames": frames.unsqueeze(0),
            "history": history.unsqueeze(0),
        })[0]

    assert segment.shape == (7, 7)
    assert torch.allclose(per_window, segment, atol=1e-5)


def test_segment_dataset_matches_windowed_items() -> None:
    from badeline.train import SessionArrays, WindowedSessionDataset

    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (20, 128, 128, 3), dtype=np.uint8)
    keys = rng.integers(0, 2, (20, 7), dtype=np.uint8)
    session = SessionArrays("toy", frames, keys)
    kwargs = dict(window=4, window_mode="centered",
                  input_config="pixels_plus_history",
                  history_len=3, history_gap=2)

    windowed = WindowedSessionDataset([session], **kwargs)
    segmented = SegmentSessionDataset([session], segment_windows=5, **kwargs)

    item = segmented[0]
    assert item["frames"].shape == (5 + 4 - 1, 3, 128, 128)
    assert item["target"].shape == (5, 7)
    assert item["history"].shape == (5, 3, 7)
    for s in range(5):
        ref = windowed[s]
        assert torch.equal(item["target"][s], ref["target"])
        assert torch.equal(item["history"][s], ref["history"])
        assert torch.equal(item["frames"][s : s + 4], ref["frames"])


def test_dilated_segment_dataset_matches_windowed_items() -> None:
    rng = np.random.default_rng(9)
    frames = rng.integers(0, 255, (30, 128, 128, 3), dtype=np.uint8)
    keys = rng.integers(0, 2, (30, 7), dtype=np.uint8)
    session = SessionArrays("dilated", frames, keys)
    kwargs = dict(
        window=4,
        frame_stride=3,
        window_mode="centered",
        input_config="pixels_plus_history",
        history_len=3,
        history_gap=2,
    )

    windowed = WindowedSessionDataset([session], **kwargs)
    segmented = SegmentSessionDataset([session], segment_windows=5, **kwargs)
    item = segmented[0]

    assert item["frames"].shape == (5 + (4 - 1) * 3, 3, 128, 128)
    for start in range(5):
        ref = windowed[start]
        assert torch.equal(item["target"][start], ref["target"])
        assert torch.equal(item["history"][start], ref["history"])
        assert torch.equal(item["frames"][start : start + 10 : 3], ref["frames"])


def test_history_block_matches_windowed_convention() -> None:
    keys = np.arange(40, dtype=np.uint8).reshape(20, 2).repeat(4, axis=1)[:, :7]
    block = history_block(keys, [0, 5], history_len=4, history_gap=2)
    assert block.shape == (2, 4, 7)
    # Target 0 with gap 2: no history available → all zeros, left-padded.
    assert block[0].sum() == 0
    # Target 5 with gap 2: rows 0..2 fill the right side of the window.
    assert np.array_equal(block[1, -3:], keys[0:3].astype(np.float32))
    assert block[1, 0].sum() == 0
