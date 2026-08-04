from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.build_vpt_small_128px_20hz import main
from experiments.complete_vpt_small_tail_windows import complete_tail_windows
from experiments.validate_vpt_small_data import validate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builder_subsamples_within_runs_and_never_crosses_gap(tmp_path: Path) -> None:
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
            "--phases", "0",
        ]
    ) == 0
    directory = output / f"{session_id}__p0"
    selected = np.load(directory / "source_row_index.npy")
    starts = np.load(directory / "window_start.npy")
    continuity = np.load(directory / "continuity_id.npy")
    assert np.array_equal(selected[:4], [0, 3, 6, 9])
    assert np.array_equal(selected[150:154], [450, 453, 456, 459])
    assert np.array_equal(starts, [0, 150])
    assert np.all(continuity[:150] == 0)
    assert np.all(continuity[150:] == 1)
    marker = json.loads((output / "complete.json").read_text(encoding="utf-8"))
    assert marker["derived_rows"] == 300
    assert marker["windows"] == 2
    report = validate(output)
    assert report["ok"] is True
    assert report["derived_rows"] == 300

    completed = tmp_path / "tail-complete"
    completion = complete_tail_windows(output, completed)
    completed_starts = np.load(completed / f"{session_id}__p0" / "window_start.npy")
    assert np.array_equal(completed_starts, [0, 150, 22, 172])
    assert completion["tail_windows"] == 2
    completed_manifest = json.loads(
        (completed / "build_manifest.json").read_text(encoding="utf-8")
    )
    assert completed_manifest["center_overlap_policy"] == "base-first-stable-tail-fill"
    assert validate(completed)["ok"] is True
