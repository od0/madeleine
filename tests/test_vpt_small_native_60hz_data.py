from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.build_vpt_small_128px_60hz import main
from experiments.validate_vpt_small_data import validate
from experiments.train_vpt_small import VPTWindowDataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_builder_retains_every_row_and_never_crosses_gap(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    session_id = "example"
    rows = 900
    frames = np.zeros((rows, 128, 128, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(rows, dtype=np.uint16) % 251
    keys = np.zeros((rows, 7), dtype=np.uint8)
    keys[:, 4] = np.arange(rows) % 2
    engine = np.concatenate((np.arange(450), np.arange(550, 1000))).astype(np.int64)
    active = np.ones(rows, dtype=np.uint8)
    shard = source / f"{session_id}.npz"
    np.savez_compressed(
        shard,
        frames=frames,
        keys=keys,
        engine_frame_idx=engine,
        input_active=active,
        session_id=np.asarray(session_id),
    )
    sessions = tmp_path / "sessions.txt"
    sessions.write_text(session_id + "\n", encoding="utf-8")
    hashes = tmp_path / "hashes.json"
    hashes.write_text(
        json.dumps({"shard_sha256": {session_id: _sha256(shard)}}), encoding="utf-8"
    )
    output = tmp_path / "derived"
    assert main(
        [
            "--source-root", str(source),
            "--sessions", str(sessions),
            "--expected-hashes", str(hashes),
            "--output-root", str(output),
        ]
    ) == 0
    directory = output / f"{session_id}__p0"
    selected = np.load(directory / "source_row_index.npy")
    starts = np.load(directory / "window_start.npy")
    continuity = np.load(directory / "continuity_id.npy")
    assert np.array_equal(selected[:4], [0, 1, 2, 3])
    assert np.array_equal(selected[450:454], [450, 451, 452, 453])
    expected_starts = np.concatenate((np.arange(0, 323, 64), np.arange(450, 773, 64)))
    assert np.array_equal(starts, expected_starts)
    assert np.all(continuity[:450] == 0)
    assert np.all(continuity[450:] == 1)
    report = validate(output)
    assert report["ok"] is True
    assert report["derived_rows"] == 900
    assert report["windows"] == 12
    dataset = VPTWindowDataset(output / "build_manifest.json")
    assert dataset.window == 128
    assert dataset[0]["frames"].shape == (128, 3, 128, 128)


def test_span_matched_builder_uses_384_window_and_192_stride(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    session_id = "span"
    rows = 1_152
    shard = source / f"{session_id}.npz"
    np.savez_compressed(
        shard,
        frames=np.zeros((rows, 128, 128, 3), dtype=np.uint8),
        keys=np.zeros((rows, 7), dtype=np.uint8),
        engine_frame_idx=np.arange(rows, dtype=np.int64),
        input_active=np.ones(rows, dtype=np.uint8),
        session_id=np.asarray(session_id),
    )
    sessions = tmp_path / "sessions.txt"
    sessions.write_text(session_id + "\n", encoding="utf-8")
    hashes = tmp_path / "hashes.json"
    hashes.write_text(
        json.dumps({"shard_sha256": {session_id: _sha256(shard)}}), encoding="utf-8"
    )
    output = tmp_path / "span-derived"
    assert main(
        [
            "--source-root", str(source),
            "--sessions", str(sessions),
            "--expected-hashes", str(hashes),
            "--output-root", str(output),
            "--window", "384", "--stride", "192",
        ]
    ) == 0
    starts = np.load(output / f"{session_id}__p0" / "window_start.npy")
    assert np.array_equal(starts, np.arange(0, 769, 192))
    report = validate(output)
    assert (report["window"], report["stride"], report["windows"]) == (384, 192, 5)
    dataset = VPTWindowDataset(output / "build_manifest.json")
    assert dataset[0]["frames"].shape == (384, 3, 128, 128)
