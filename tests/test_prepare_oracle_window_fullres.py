from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.oracle_window_localization import FeatureSession, OracleExample, sha256_file
from experiments.prepare_oracle_window_fullres import (
    CACHE_SCHEMA,
    RECEIPT_SCHEMA,
    _copy_frame_memmap,
    build_fullres_cache,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _source(path: Path, session_id: str, value: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.full((32, 128, 128, 3), value, dtype=np.uint8)
    keys = np.zeros((32, 7), dtype=np.uint8)
    engine = np.arange(32, dtype=np.int64)
    active = np.ones(32, dtype=np.uint8)
    np.savez_compressed(
        path,
        frames=frames,
        keys=keys,
        engine_frame_idx=engine,
        input_active=active,
        session_id=np.asarray(session_id),
    )
    return keys, engine, active


def _example(split: str, session_id: str) -> OracleExample:
    return OracleExample(
        split=split,
        session_id=session_id,
        run_index=0,
        array_index=8,
        engine_frame_idx=8,
        head_index=0,
        key_index=0,
        event_type_index=0,
        offset=0,
        crop_start=0,
        candidate_start=8,
        block_id=f"{session_id}:run0:block0",
    )


def test_copy_frame_memmap_round_trips_exact_uint8(tmp_path: Path) -> None:
    frames = np.arange(4 * 128 * 128 * 3, dtype=np.uint32).reshape(4, 128, 128, 3).astype(np.uint8)
    path = tmp_path / "frames.npy"
    _copy_frame_memmap(frames, path, chunk=2)
    observed = np.load(path, mmap_mode="r", allow_pickle=False)
    assert isinstance(observed, np.memmap)
    assert observed.dtype == np.uint8
    assert np.array_equal(observed, frames)


def test_copy_frame_memmap_rejects_wrong_geometry(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="128"):
        _copy_frame_memmap(np.zeros((2, 32, 32, 3), np.uint8), tmp_path / "bad.npy")


def test_fullres_builder_publishes_exact_four_session_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    feature_root = tmp_path / "features"
    source_root.mkdir()
    feature_root.mkdir()
    train_ids = ("train_a", "train_b", "train_c")
    val_ids = ("val_a",)
    sessions: dict[str, FeatureSession] = {}
    source_receipts = []
    source_manifest_rows = []
    for index, session_id in enumerate((*train_ids, *val_ids)):
        path = source_root / f"{session_id}.npz"
        keys, engine, active = _source(path, session_id, index)
        sessions[session_id] = FeatureSession(
            session_id=session_id,
            features=np.zeros((32, 512), dtype=np.float16),
            keys=keys,
            engine_frame_idx=engine,
            input_active=active,
        )
        source_receipts.append(
            {"session_id": session_id, "source_npz_sha256": sha256_file(path)}
        )
        source_manifest_rows.append(
            {"session_id": session_id, "masked_regions": ["frame_index_strip", "input_overlay"]}
        )
    source_manifest = {"frame_size": 128, "sessions": source_manifest_rows}
    _write_json(source_root / "build_manifest.json", source_manifest)
    feature_receipt = {
        "status": "complete",
        "source_root": str(source_root.resolve()),
        "content": {"source": {"sessions": source_receipts}},
    }
    feature_receipt_path = tmp_path / "feature_receipt.json"
    _write_json(feature_receipt_path, feature_receipt)
    base_config = {
        "status": "preregistered_before_validation_inference",
        "dataset": {
            "feature_receipt_sha256": sha256_file(feature_receipt_path),
            "source_build_manifest_sha256": sha256_file(source_root / "build_manifest.json"),
            "train_sessions": list(train_ids),
            "validation_sessions": list(val_ids),
            "crop_frames": 32,
            "candidate_width": 16,
            "context_halo": 8,
        },
    }
    base_config_path = tmp_path / "base_config.json"
    _write_json(base_config_path, base_config)
    base_manifest_path = tmp_path / "base_manifest.json"
    _write_json(base_manifest_path, {"exact": True})
    train_examples = (_example("train", train_ids[0]),)
    val_examples = (_example("validation", val_ids[0]),)
    monkeypatch.setattr(
        "experiments.prepare_oracle_window_fullres._prepare_data",
        lambda **_: (sessions, train_examples, val_examples, {}),
    )
    monkeypatch.setattr(
        "experiments.prepare_oracle_window_fullres._dataset_manifest",
        lambda **_: {"exact": True},
    )
    output = tmp_path / "cache"
    receipt_path = tmp_path / "cache_complete.json"
    receipt = build_fullres_cache(
        source_root=source_root,
        feature_root=feature_root,
        feature_receipt_path=feature_receipt_path,
        base_config_path=base_config_path,
        base_dataset_manifest_path=base_manifest_path,
        output=output,
        receipt_path=receipt_path,
    )
    assert receipt["schema_version"] == RECEIPT_SCHEMA
    assert receipt["status"] == "complete"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == CACHE_SCHEMA
    assert manifest["sessions"] == [*train_ids, *val_ids]
    assert manifest["train_examples"] == 1
    assert manifest["validation_examples"] == 1
    assert set(receipt["payload"]) == {
        "frames/train_a.npy", "frames/train_b.npy", "frames/train_c.npy", "frames/val_a.npy",
        "supervision/train_a.npz", "supervision/train_b.npz", "supervision/train_c.npz", "supervision/val_a.npz",
        "train_examples.npz", "validation_examples.npz", "manifest.json",
    }
    with pytest.raises(ValueError, match="overwrite"):
        build_fullres_cache(
            source_root=source_root,
            feature_root=feature_root,
            feature_receipt_path=feature_receipt_path,
            base_config_path=base_config_path,
            base_dataset_manifest_path=base_manifest_path,
            output=output,
            receipt_path=receipt_path,
        )
