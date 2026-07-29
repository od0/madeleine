from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.validate_own_v3_features import (
    FEATURE_FORMAT,
    SESSION_IDS,
    TRAIN_SESSION_IDS,
    VAL_SESSION_IDS,
    validate_features,
    validate_source,
)


def _write_source(root: Path, *, extra_session: bool = False) -> None:
    root.mkdir()
    reports = []
    for index, session_id in enumerate(SESSION_IDS):
        frames = np.full((5 + index, 128, 128, 3), index, dtype=np.uint8)
        keys = np.zeros((len(frames), 7), dtype=np.uint8)
        keys[:, index % 7] = 1
        engine = np.arange(100 * index, 100 * index + len(frames), dtype=np.int64)
        active = np.ones(len(frames), dtype=np.uint8)
        np.savez_compressed(
            root / f"{session_id}.npz",
            frames=frames,
            keys=keys,
            engine_frame_idx=engine,
            input_active=active,
            session_id=np.asarray(session_id),
        )
        reports.append(
            {
                "session_id": session_id,
                "frames": len(frames),
                "npz": f"{session_id}.npz",
            }
        )
    if extra_session:
        np.savez_compressed(root / "forbidden.npz", frames=np.zeros((1,)))
    (root / "train_sessions.txt").write_text(
        "\n".join(TRAIN_SESSION_IDS) + "\n", encoding="utf-8"
    )
    (root / "val_sessions.txt").write_text(
        "\n".join(VAL_SESSION_IDS) + "\n", encoding="utf-8"
    )
    (root / "build_manifest.json").write_text(
        json.dumps(
            {
                "built_at": "fixture",
                "frame_size": 128,
                "split": {
                    "train": list(TRAIN_SESSION_IDS),
                    "val": list(VAL_SESSION_IDS),
                    "unit": "session",
                },
                "grid": {"engine_hz": 60},
                "sessions": reports,
            }
        ),
        encoding="utf-8",
    )


def _write_features(source: Path, output: Path) -> None:
    output.mkdir()
    manifest = json.loads((source / "build_manifest.json").read_text())
    copied = {
        **manifest,
        "visual_representation": FEATURE_FORMAT,
        "backbone_feature_dim": 512,
        "source_build_manifest": str(source / "build_manifest.json"),
    }
    (output / "build_manifest.json").write_text(json.dumps(copied))
    (output / "train_sessions.txt").write_bytes(
        (source / "train_sessions.txt").read_bytes()
    )
    (output / "val_sessions.txt").write_bytes(
        (source / "val_sessions.txt").read_bytes()
    )
    rows = []
    for session_id in SESSION_IDS:
        source_path = source / f"{session_id}.npz"
        with np.load(source_path, allow_pickle=False) as archive:
            keys = archive["keys"]
            engine = archive["engine_frame_idx"]
            active = archive["input_active"]
            stored_id = archive["session_id"]
        np.savez(
            output / f"{session_id}.npz",
            features=np.zeros((len(keys), 512), dtype=np.float16),
            keys=keys,
            engine_frame_idx=engine,
            input_active=active,
            session_id=stored_id,
        )
        rows.append(
            {
                "session_id": session_id,
                "frames": len(keys),
                "source": str(source_path),
                "npz": f"{session_id}.npz",
                "resumed": False,
            }
        )
    (output / "feature_build_manifest.json").write_text(
        json.dumps(
            {
                "built_at": "fixture",
                "format": FEATURE_FORMAT,
                "backbone_feature_dim": 512,
                "frame_size": 128,
                "source_kind": "audited_rgb_shards",
                "sessions": rows,
            }
        )
    )


def _write_source_snapshot(source: Path, path: Path) -> None:
    path.write_text(json.dumps(validate_source(source)), encoding="utf-8")


def test_source_preflight_requires_exact_four_sessions(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source)

    report = validate_source(source)

    assert report["session_count"] == 4
    assert [row["session_id"] for row in report["sessions"]] == list(SESSION_IDS)


def test_source_preflight_rejects_extra_shard(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source, extra_session=True)

    with pytest.raises(ValueError, match="extra=.*forbidden.npz"):
        validate_source(source)


def test_feature_validation_is_content_bound_and_preserves_supervision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "features"
    _write_source(source)
    _write_features(source, output)
    snapshot = tmp_path / "source-snapshot.json"
    _write_source_snapshot(source, snapshot)
    monkeypatch.setattr(
        "experiments.validate_own_v3_features._git_receipt",
        lambda _repo, _commit: {
            "commit": "a" * 40,
            "relevant_files": {},
            "relevant_files_clean": True,
        },
    )

    report = validate_features(
        source_root=source,
        feature_root=output,
        published_output=tmp_path / "published",
        repo=tmp_path,
        expected_commit="a" * 40,
        source_snapshot_path=snapshot,
    )

    assert report["status"] == "complete"
    assert report["session_count"] == 4
    assert len(report["content_sha256"]) == 64
    assert report["checks"]["supervision_arrays_equal_to_source"]
    assert all(
        all(row["supervision_equal_to_source"].values())
        for row in report["content"]["sessions"]
    )


def test_feature_validation_rejects_changed_engine_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "features"
    _write_source(source)
    _write_features(source, output)
    snapshot = tmp_path / "source-snapshot.json"
    _write_source_snapshot(source, snapshot)
    target = output / f"{SESSION_IDS[0]}.npz"
    with np.load(target, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["engine_frame_idx"] = values["engine_frame_idx"].copy()
    values["engine_frame_idx"][0] += 1
    np.savez(target, **values)
    monkeypatch.setattr(
        "experiments.validate_own_v3_features._git_receipt",
        lambda _repo, _commit: {},
    )

    with pytest.raises(ValueError, match="engine_frame_idx"):
        validate_features(
            source_root=source,
            feature_root=output,
            published_output=tmp_path / "published",
            repo=tmp_path,
            expected_commit=None,
            source_snapshot_path=snapshot,
        )


def test_feature_validation_rejects_stale_temp_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "features"
    _write_source(source)
    _write_features(source, output)
    snapshot = tmp_path / "source-snapshot.json"
    _write_source_snapshot(source, snapshot)
    (output / ".stale.tmp.npz").write_bytes(b"partial")

    with pytest.raises(ValueError, match="inventory changed"):
        validate_features(
            source_root=source,
            feature_root=output,
            published_output=tmp_path / "published",
            repo=tmp_path,
            expected_commit=None,
            source_snapshot_path=snapshot,
        )


def test_feature_validation_rejects_source_change_after_preflight(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "features"
    snapshot = tmp_path / "source-snapshot.json"
    _write_source(source)
    _write_source_snapshot(source, snapshot)
    _write_features(source, output)
    target = source / f"{SESSION_IDS[0]}.npz"
    with np.load(target, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["frames"] = values["frames"].copy()
    values["frames"][0, 0, 0, 0] += 1
    np.savez_compressed(target, **values)

    with pytest.raises(ValueError, match="source changed after preflight"):
        validate_features(
            source_root=source,
            feature_root=output,
            published_output=tmp_path / "published",
            repo=tmp_path,
            expected_commit=None,
            source_snapshot_path=snapshot,
        )
def test_source_preflight_rejects_empty_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source)
    (source / "partial").mkdir()

    with pytest.raises(ValueError, match="non-regular entries"):
        validate_source(source)
