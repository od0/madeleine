from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from badeline.dynamics_pretraining import REPRESENTATION_DIM
from data.schema import KEY_ORDER
from experiments.assemble_dynamics_supervised_features import (
    assemble_supervised_features,
    validate_supervised_features,
)
from experiments.build_dynamics_feature_export_inventory import (
    build_export_inventory,
    main as build_inventory_main,
)
from experiments.build_dynamics_pretraining_inventory import (
    SCHEMA as SSL_SCHEMA,
    canonical_sha256,
)
from experiments.export_dynamics_features import (
    CHECKPOINT_SCHEMA,
    ExpectedCounts,
    load_checkpoint_contract,
    load_inventory,
    sha256_file,
)


COUNTS = ExpectedCounts(videos=2, sessions=2, frames=5, train_videos=1)


def _write_standard_shard(
    path: Path, *, session_id: str, start: int, frames: int
) -> None:
    keys = np.zeros((frames, len(KEY_ORDER)), dtype=np.uint8)
    keys[:, 0] = np.arange(frames) % 2
    np.savez(
        path,
        features=np.zeros((frames, REPRESENTATION_DIM), dtype=np.float16),
        keys=keys,
        engine_frame_idx=np.arange(start, start + frames, dtype=np.int64),
        input_active=np.ones(frames, dtype=np.uint8),
        session_id=np.asarray(session_id),
    )


def _fixture(tmp_path: Path) -> dict:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    train_video = raw_root / "trainVideo.mp4"
    y4n_video = raw_root / "y4n.mp4"
    train_video.write_bytes(b"training-video-bytes")
    y4n_video.write_bytes(b"held-out-y4n-video-bytes")
    fetch_report = raw_root / "fetch.jsonl"
    fetch_report.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "status": "ok",
                    "video_id": "trainVideo",
                    "path": train_video.name,
                    "bytes": train_video.stat().st_size,
                    "width": 320,
                    "height": 180,
                    "fps": 60.0,
                    "frames": 100,
                },
                {
                    "status": "ok",
                    "video_id": "y4nQHqYSObI",
                    "path": y4n_video.name,
                    "bytes": y4n_video.stat().st_size,
                    "width": 320,
                    "height": 180,
                    "fps": 60.0,
                    "frames": 100,
                },
            )
        )
        + "\n"
    )

    full_root = tmp_path / "full"
    full_root.mkdir()
    train_session = "trainVideo__r000"
    y4n_session = "y4nQHqYSObI__r000"
    _write_standard_shard(
        full_root / f"{train_session}.npz",
        session_id=train_session,
        start=5,
        frames=3,
    )
    _write_standard_shard(
        full_root / f"{y4n_session}.npz",
        session_id=y4n_session,
        start=20,
        frames=2,
    )
    shard_hashes = {
        session_id: {
            "sha256": sha256_file(full_root / f"{session_id}.npz"),
            "size": (full_root / f"{session_id}.npz").stat().st_size,
        }
        for session_id in (train_session, y4n_session)
    }
    (full_root / "shard_hashes.json").write_text(json.dumps(shard_hashes))
    full_manifest = {
        "format": "resnet18_imagenet_avgpool_float16_v1",
        "source_kind": "mapped_foreign_video",
        "video_count": 2,
        "session_count": 2,
        "train_frames": 5,
        "videos": [
            {
                "video_id": "trainVideo",
                "frames": 3,
                "sessions": [train_session],
                "decoder_mode": "opencv_native_60hz",
            },
            {
                "video_id": "y4nQHqYSObI",
                "frames": 2,
                "sessions": [y4n_session],
                "decoder_mode": "opencv_native_60hz",
            },
        ],
    }
    (full_root / "full_corpus_manifest.json").write_text(
        json.dumps(full_manifest)
    )
    all_ids = sorted((train_session, y4n_session))
    (full_root / "train_sessions.txt").write_text("\n".join(all_ids) + "\n")
    (full_root / "unflagged_sessions.txt").write_text(
        "\n".join(all_ids) + "\n"
    )
    (full_root / "val_sessions.txt").write_text("")

    feature_root = tmp_path / "features_by_video"
    y4n_feature_root = feature_root / "y4nQHqYSObI"
    y4n_feature_root.mkdir(parents=True)
    y4n_feature_manifest = {
        "format": "resnet18_imagenet_avgpool_float16_v1",
        "videos": [
            {
                "video_id": "y4nQHqYSObI",
                "video": {
                    "average_fps": 60.0,
                    "decoded_frames": 100,
                    "nominal_timeline_frames": 100,
                    "resolution_wh": [320, 180],
                },
                "decoder_mode": "opencv_native_60hz",
                "mask_rect_xyxy": [10, 10, 20, 20],
                "parts": [
                    {
                        "session_id": y4n_session,
                        "frames": 2,
                        "source_frame_range": [20, 22],
                    }
                ],
            }
        ],
    }
    (y4n_feature_root / "feature_build_manifest.json").write_text(
        json.dumps(y4n_feature_manifest)
    )

    validation_path = tmp_path / "full_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "ok": True,
                "deep_shards": True,
                "observed": {
                    "valid_videos": 2,
                    "sessions": 2,
                    "train_frames": 5,
                    "deep_shards_checked": 2,
                },
                "paths": {
                    "output_root": str(full_root.resolve()),
                    "raw_root": str(raw_root.resolve()),
                    "fetch_report": str(fetch_report.resolve()),
                    "feature_root": str(feature_root.resolve()),
                },
            }
        )
    )

    ssl_inventory = {
        "schema_version": SSL_SCHEMA,
        "labels_consumed": False,
        "forbidden_exclusion_proof": {"whole_y4n_absent": True},
        "nitrogen_videos": [
            {
                "source": "nitrogen",
                "video_id": "trainVideo",
                "video_path": str(train_video.resolve()),
                "video_sha256": sha256_file(train_video),
                "video_bytes": train_video.stat().st_size,
                "decoder_mode": "opencv_native_60hz",
                "source_width": 320,
                "source_height": 180,
                "source_fps": 60.0,
                "source_frames": 100,
                "mask_rect_source_xyxy": [10, 10, 20, 20],
                "mask_rect_128_xyxy": [3, 6, 9, 16],
                "sessions": [train_session],
            }
        ],
        "sessions": [
            {
                "source": "nitrogen",
                "session_id": train_session,
                "video_id": "trainVideo",
                "reference_shard": str(
                    (full_root / f"{train_session}.npz").resolve()
                ),
                "reference_shard_sha256": shard_hashes[train_session]["sha256"],
                "frames": 3,
                "active_frames": 3,
                "eligible_windows": 1,
                "engine_frame_start": 5,
                "engine_frame_end_exclusive": 8,
            }
        ],
    }
    ssl_inventory["inventory_content_sha256"] = canonical_sha256(ssl_inventory)
    ssl_path = tmp_path / "ssl_inventory.json"
    ssl_path.write_text(json.dumps(ssl_inventory))

    checkpoint_path = tmp_path / "final.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "model_state": {"placeholder": torch.tensor([1])},
            "arm": "C",
            "horizons": [1, 2, 4],
            "completed_steps": 30_000,
            "kind": "final",
            "selection_eligible": True,
            "resumable": False,
        },
        checkpoint_path,
    )
    return {
        "raw_root": raw_root,
        "fetch_report": fetch_report,
        "full_root": full_root,
        "feature_root": feature_root,
        "validation_path": validation_path,
        "ssl_path": ssl_path,
        "checkpoint_path": checkpoint_path,
        "train_session": train_session,
        "y4n_session": y4n_session,
    }


def _build_payload(fixture: dict) -> dict:
    return build_export_inventory(
        ssl_inventory_path=fixture["ssl_path"],
        ssl_inventory_sha256=sha256_file(fixture["ssl_path"]),
        full_root=fixture["full_root"],
        full_validation_path=fixture["validation_path"],
        raw_root=fixture["raw_root"],
        fetch_report=fixture["fetch_report"],
        feature_root=fixture["feature_root"],
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_sha256=sha256_file(fixture["checkpoint_path"]),
        arm="C",
        expected_completed_steps=30_000,
        expected_counts=COUNTS,
    )


def test_export_inventory_reuses_train_hash_and_hashes_y4n_post_terminal(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    payload = _build_payload(fixture)
    by_id = {row["video_id"]: row for row in payload["videos"]}
    ssl = json.loads(fixture["ssl_path"].read_text())
    assert by_id["trainVideo"]["video_sha256"] == (
        ssl["nitrogen_videos"][0]["video_sha256"]
    )
    assert by_id["y4nQHqYSObI"]["video_sha256"] == sha256_file(
        fixture["raw_root"] / "y4n.mp4"
    )
    assert payload["provenance"][
        "y4n_hashed_after_terminal_checkpoint_validation"
    ] is True
    assert payload["provenance"]["terminal_checkpoint"]["completed_steps"] == 30_000


def test_y4n_builder_path_is_unreachable_before_checkpoint_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    reached_y4n = False

    def forbidden_y4n(**_kwargs):
        nonlocal reached_y4n
        reached_y4n = True
        raise AssertionError("y4n path was reached")

    monkeypatch.setattr(
        "experiments.build_dynamics_feature_export_inventory._y4n_row",
        forbidden_y4n,
    )
    with pytest.raises(FileNotFoundError, match="checkpoint is absent"):
        build_export_inventory(
            ssl_inventory_path=fixture["ssl_path"],
            ssl_inventory_sha256=sha256_file(fixture["ssl_path"]),
            full_root=fixture["full_root"],
            full_validation_path=fixture["validation_path"],
            raw_root=fixture["raw_root"],
            fetch_report=fixture["fetch_report"],
            feature_root=fixture["feature_root"],
            checkpoint_path=tmp_path / "absent.pt",
            checkpoint_sha256="0" * 64,
            arm="C",
            expected_completed_steps=30_000,
            expected_counts=COUNTS,
        )
    assert reached_y4n is False


def test_export_inventory_cli_dispatches_and_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "inventory.json"
    observed: dict = {}

    def fake_build(**kwargs):
        observed.update(kwargs)
        return {
            "population": {"videos": 211},
            "provenance": {
                "terminal_checkpoint": {"arm": "D"},
            },
        }

    monkeypatch.setattr(
        "experiments.build_dynamics_feature_export_inventory.build_export_inventory",
        fake_build,
    )
    result = build_inventory_main(
        [
            "--ssl-inventory", str(tmp_path / "ssl.json"),
            "--ssl-inventory-sha256", "1" * 64,
            "--full-feature-root", str(tmp_path / "full"),
            "--full-validation", str(tmp_path / "validation.json"),
            "--raw-root", str(tmp_path / "raw"),
            "--fetch-report", str(tmp_path / "fetch.jsonl"),
            "--feature-root", str(tmp_path / "features"),
            "--checkpoint", str(tmp_path / "final.pt"),
            "--checkpoint-sha256", "2" * 64,
            "--arm", "D",
            "--expected-completed-steps", "30000",
            "--output", str(output),
        ]
    )
    assert result == 0
    assert observed["arm"] == "D"
    assert observed["expected_completed_steps"] == 30_000
    assert json.loads(output.read_text())["population"]["videos"] == 211


def _write_feature_export(
    root: Path, fixture: dict, payload: dict
) -> None:
    root.mkdir()
    for video in payload["videos"]:
        for session in video["sessions"]:
            frames = session["end_frame"] - session["start_frame"]
            np.savez(
                root / f"{session['session_id']}.npz",
                features=np.full(
                    (frames, REPRESENTATION_DIM), 0.5, dtype=np.float16
                ),
                engine_frame_idx=np.arange(
                    session["start_frame"], session["end_frame"], dtype=np.int64
                ),
                input_active=np.ones(frames, dtype=np.uint8),
                session_id=np.asarray(session["session_id"]),
            )
    (root / "feature_export_manifest.json").write_text(
        json.dumps({"fixture": True})
    )


def test_supervised_assembler_is_post_terminal_and_emits_standard_npzs(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    payload = _build_payload(fixture)
    inventory_path = tmp_path / "export_inventory.json"
    inventory_path.write_text(json.dumps(payload))
    inventory = load_inventory(
        inventory_path,
        sha256_file(inventory_path),
        expected_counts=COUNTS,
    )
    checkpoint = load_checkpoint_contract(
        fixture["checkpoint_path"],
        sha256_file(fixture["checkpoint_path"]),
        expected_arm="C",
        expected_completed_steps=30_000,
    )
    export_root = tmp_path / "export"
    _write_feature_export(export_root, fixture, payload)
    validation = {
        "ok": True,
        "deep_shards": True,
        "deep_references": True,
        "inventory": {"sha256": inventory.sha256},
        "checkpoint": {
            "sha256": checkpoint.sha256,
            "arm": "C",
            "completed_steps": 30_000,
        },
        "counts": {"checked_sessions": 2, "checked_frames": 5},
    }
    output = tmp_path / "assembled"
    result = assemble_supervised_features(
        inventory=inventory,
        checkpoint=checkpoint,
        feature_root=export_root,
        reference_root=fixture["full_root"],
        output_root=output,
        terminal_validation=validation,
        expected_counts=COUNTS,
    )
    assert result["counts"] == {
        "videos": 2,
        "sessions": 2,
        "frames": 5,
        "train_sessions": 1,
        "y4n_sessions": 1,
        "y4n_later8_sessions": 0,
    }
    with np.load(
        output / f"{fixture['train_session']}.npz", allow_pickle=False
    ) as archive:
        assert set(archive.files) == {
            "features", "keys", "engine_frame_idx", "input_active", "session_id"
        }
        assert archive["features"].dtype == np.float16
        assert archive["keys"].shape == (3, len(KEY_ORDER))
    assert (output / "all_sessions.txt").read_text().splitlines() == sorted(
        [fixture["train_session"], fixture["y4n_session"]]
    )
    assert (output / "train_sessions.txt").read_text().splitlines() == [
        fixture["train_session"]
    ]
    assert (output / "val_sessions.txt").read_text().splitlines() == [
        fixture["y4n_session"]
    ]
    assert (output / "y4n_later8_sessions.txt").read_text() == ""
    manifest = json.loads((output / "full_corpus_manifest.json").read_text())
    assert manifest["format"] == "dynamics_c_final_ema_raw_avgpool_float16_v1"
    assert manifest["post_ssl_supervised_assembly"] is True

    validation_result = validate_supervised_features(
        inventory=inventory,
        checkpoint=checkpoint,
        feature_root=export_root,
        reference_root=fixture["full_root"],
        output_root=output,
        terminal_validation=validation,
        expected_counts=COUNTS,
    )
    assert validation_result["ok"] is True
    assert validation_result["counts"] == {
        "expected_sessions": 2,
        "expected_frames": 5,
        "checked_sessions": 2,
        "checked_frames": 5,
    }

    (output / "train_sessions.txt").write_text(
        fixture["train_session"] + "\n" + fixture["y4n_session"] + "\n"
    )
    failed = validate_supervised_features(
        inventory=inventory,
        checkpoint=checkpoint,
        feature_root=export_root,
        reference_root=fixture["full_root"],
        output_root=output,
        terminal_validation=validation,
        expected_counts=COUNTS,
    )
    assert failed["ok"] is False
    assert "train_sessions.txt membership differs" in failed["failures"][0]


def test_supervised_assembler_refuses_before_terminal_validation_without_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    payload = _build_payload(fixture)
    inventory_path = tmp_path / "export_inventory.json"
    inventory_path.write_text(json.dumps(payload))
    inventory = load_inventory(
        inventory_path,
        sha256_file(inventory_path),
        expected_counts=COUNTS,
    )
    checkpoint = load_checkpoint_contract(
        fixture["checkpoint_path"],
        sha256_file(fixture["checkpoint_path"]),
        expected_arm="C",
        expected_completed_steps=30_000,
    )
    label_accessed = False

    def forbidden_label_access(_session):
        nonlocal label_accessed
        label_accessed = True
        raise AssertionError("keys were opened")

    monkeypatch.setattr(
        "experiments.assemble_dynamics_supervised_features._load_supervision",
        forbidden_label_access,
    )
    with pytest.raises(ValueError, match="lacks passing validation"):
        assemble_supervised_features(
            inventory=inventory,
            checkpoint=checkpoint,
            feature_root=tmp_path / "not-opened-features",
            reference_root=fixture["full_root"],
            output_root=tmp_path / "not-created-output",
            terminal_validation={"ok": False},
            expected_counts=COUNTS,
        )
    assert label_accessed is False
    assert not (tmp_path / "not-created-output").exists()
