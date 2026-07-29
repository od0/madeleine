from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from badeline.dynamics_pretraining import REPRESENTATION_DIM
from experiments.export_dynamics_features import (
    CHECKPOINT_SCHEMA,
    COMPLETION_SCHEMA,
    EXPORT_ONLY_ROLE,
    INVENTORY_SCHEMA,
    MANIFEST_SCHEMA,
    SHARD_SIDECAR_SCHEMA,
    TRAIN_ROLE,
    Y4N_VIDEO_ID,
    CheckpointContract,
    ExpectedCounts,
    SessionSpec,
    _expected_previous_index_sha256,
    array_sha256,
    encode_rgb_frames,
    load_checkpoint_contract,
    load_reference_metadata,
    sha256_file,
    validate_inventory_payload,
)
from experiments.validate_dynamics_features import validate_export


SHA = "0" * 64


class _RawFrameEncoder(nn.Module):
    def forward(self, current: torch.Tensor) -> torch.Tensor:
        value = current.mean(dim=(1, 2, 3), keepdim=False)
        return value[:, None].expand(-1, REPRESENTATION_DIM)


class _RawPairEncoder(nn.Module):
    def forward(
        self, previous: torch.Tensor, current: torch.Tensor
    ) -> torch.Tensor:
        previous_value = previous.mean(dim=(1, 2, 3), keepdim=False)
        current_value = current.mean(dim=(1, 2, 3), keepdim=False)
        value = 2.0 * previous_value + current_value
        return value[:, None].expand(-1, REPRESENTATION_DIM)


def _video_row(
    root: Path,
    *,
    video_id: str,
    role: str,
    start: int,
    end: int,
) -> dict:
    session_id = f"{video_id}__r000"
    return {
        "video_id": video_id,
        "role": role,
        "video_path": str((root / f"{video_id}.mp4").resolve()),
        "video_sha256": SHA,
        "decoder_mode": "opencv_native_60hz",
        "video": {
            "average_fps": 60.0,
            "decoded_frames": 100,
            "nominal_timeline_frames": 100,
            "resolution_wh": [320, 180],
        },
        "mask_rect_xyxy": [10, 10, 20, 20],
        "resized_mask_rect_xyxy": [3, 6, 9, 16],
        "sessions": [
            {
                "session_id": session_id,
                "start_frame": start,
                "end_frame": end,
                "reference_shard": str((root / f"{session_id}.npz").resolve()),
                "reference_shard_sha256": SHA,
            }
        ],
    }


def _inventory_payload(root: Path) -> tuple[dict, ExpectedCounts]:
    payload = {
        "schema_version": INVENTORY_SCHEMA,
        "population": {
            "videos": 2,
            "sessions": 2,
            "frames": 5,
            "train_videos": 1,
        },
        "videos": [
            _video_row(
                root, video_id="nitrogenTrain", role=TRAIN_ROLE, start=5, end=8
            ),
            _video_row(
                root,
                video_id=Y4N_VIDEO_ID,
                role=EXPORT_ONLY_ROLE,
                start=20,
                end=22,
            ),
        ],
    }
    return payload, ExpectedCounts(videos=2, sessions=2, frames=5, train_videos=1)


def test_inventory_requires_explicit_y4n_export_only_role(tmp_path: Path) -> None:
    payload, counts = _inventory_payload(tmp_path)
    inventory = validate_inventory_payload(
        payload, path=tmp_path / "inventory.json", sha256=SHA, expected_counts=counts
    )
    assert len(inventory.videos) == 2
    assert inventory.frames == 5
    assert inventory.videos[-1].video_id == Y4N_VIDEO_ID
    assert inventory.videos[-1].role == EXPORT_ONLY_ROLE

    payload["videos"][-1]["role"] = TRAIN_ROLE
    with pytest.raises(ValueError, match="y4n must be the sole"):
        validate_inventory_payload(
            payload,
            path=tmp_path / "inventory.json",
            sha256=SHA,
            expected_counts=counts,
        )


def test_inventory_rejects_embargoed_identity_before_data_access(
    tmp_path: Path,
) -> None:
    payload, counts = _inventory_payload(tmp_path)
    payload["videos"][0]["sessions"][0]["session_id"] = (
        "rec_20260727_220000_test"
    )
    with pytest.raises(ValueError, match="forbidden evaluation data"):
        validate_inventory_payload(
            payload,
            path=tmp_path / "inventory.json",
            sha256=SHA,
            expected_counts=counts,
        )


def test_arm_d_duplicates_each_session_start_and_preserves_batch_boundary() -> None:
    frames = np.zeros((3, 128, 128, 3), dtype=np.uint8)
    frames[0] = 10
    frames[1] = 20
    frames[2] = 40
    features = encode_rgb_frames(
        frames,
        target_encoder=_RawPairEncoder(),
        arm="D",
        device=torch.device("cpu"),
        batch_size=2,
    )
    expected = np.asarray(
        [30.0 / 255.0, 40.0 / 255.0, 80.0 / 255.0], dtype=np.float16
    )
    np.testing.assert_array_equal(features[:, 0], expected)
    assert features.dtype == np.float16
    # Raw features are intentionally not L2-normalized.
    assert not np.allclose(np.linalg.norm(features.astype(np.float32), axis=1), 1.0)


def test_arm_c_exports_raw_target_values() -> None:
    frames = np.full((2, 128, 128, 3), 64, dtype=np.uint8)
    features = encode_rgb_frames(
        frames,
        target_encoder=_RawFrameEncoder(),
        arm="C",
        device=torch.device("cpu"),
        batch_size=1,
    )
    assert np.all(features == np.float16(64 / 255.0))


def test_reference_metadata_loader_never_decodes_keys_or_old_features(
    tmp_path: Path,
) -> None:
    session_id = "safeVideo__r000"
    reference = tmp_path / f"{session_id}.npz"
    # Object arrays raise under allow_pickle=False if accessed.  Their
    # presence proves that only the three permitted metadata members are read.
    np.savez(
        reference,
        features=np.asarray([object()], dtype=object),
        keys=np.asarray([object()], dtype=object),
        engine_frame_idx=np.arange(3, 6, dtype=np.int64),
        input_active=np.ones(3, dtype=np.uint8),
        session_id=np.asarray(session_id),
    )
    session = SessionSpec(
        video_id="safeVideo",
        role=TRAIN_ROLE,
        session_id=session_id,
        start_frame=3,
        end_frame=6,
        reference_shard=reference,
        reference_shard_sha256=sha256_file(reference),
    )
    engine, active = load_reference_metadata(session)
    np.testing.assert_array_equal(engine, np.arange(3, 6, dtype=np.int64))
    np.testing.assert_array_equal(active, np.ones(3, dtype=np.uint8))


def test_production_checkpoint_schema_and_exact_hash_are_required(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "model_state": {"target_encoder.fake": torch.tensor([1.0])},
            "arm": "C",
            "horizons": [1, 2, 4],
            "completed_steps": 30_000,
            "kind": "final",
            "selection_eligible": True,
            "resumable": False,
        },
        checkpoint,
    )
    digest = sha256_file(checkpoint)
    contract = load_checkpoint_contract(
        checkpoint,
        digest,
        expected_arm="C",
        expected_completed_steps=30_000,
    )
    assert contract.sha256 == digest
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_checkpoint_contract(
            checkpoint,
            "f" * 64,
            expected_arm="C",
            expected_completed_steps=30_000,
        )

    payload = torch.load(checkpoint, weights_only=False)
    payload["kind"] = "resume"
    payload["selection_eligible"] = False
    payload["resumable"] = True
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="terminal final checkpoint"):
        load_checkpoint_contract(
            checkpoint,
            sha256_file(checkpoint),
            expected_arm="C",
            expected_completed_steps=30_000,
        )

    payload["kind"] = "final"
    payload["selection_eligible"] = True
    payload["resumable"] = False
    payload["schema_version"] = "madeleine.dynamics-pretraining-checkpoint.v1"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="checkpoint schema"):
        load_checkpoint_contract(
            checkpoint,
            sha256_file(checkpoint),
            expected_arm="C",
            expected_completed_steps=30_000,
        )


def _write_feature_artifacts(
    out_dir: Path,
    *,
    inventory,
    checkpoint: CheckpointContract,
) -> None:
    out_dir.mkdir()
    rows = []
    video_by_id = {item.video_id: item for item in inventory.videos}
    for session in inventory.sessions:
        features = np.full(
            (session.frames, REPRESENTATION_DIM), 0.25, dtype=np.float16
        )
        engine = np.arange(session.start_frame, session.end_frame, dtype=np.int64)
        active = np.ones(session.frames, dtype=np.uint8)
        shard = out_dir / f"{session.session_id}.npz"
        np.savez(
            shard,
            features=features,
            engine_frame_idx=engine,
            input_active=active,
            session_id=np.asarray(session.session_id),
        )
        video = video_by_id[session.video_id]
        row = {
            "schema_version": SHARD_SIDECAR_SCHEMA,
            "session_id": session.session_id,
            "video_id": session.video_id,
            "role": session.role,
            "arm": checkpoint.arm,
            "checkpoint_sha256": checkpoint.sha256,
            "inventory_sha256": inventory.sha256,
            "reference_shard": str(session.reference_shard),
            "reference_shard_sha256": session.reference_shard_sha256,
            "source_video_sha256": video.video_sha256,
            "source_frame_range": [session.start_frame, session.end_frame],
            "frames": session.frames,
            "decoder_mode": video.decoder_mode,
            "imputed_tail_frames": 0,
            "feature_format": "dynamics_d_final_ema_raw_avgpool_float16_v1",
            "feature_dim": REPRESENTATION_DIM,
            "normalization": "none_raw_target_encoder_output",
            "supervision_phase": "label_free_feature_export_no_keys_member_read_or_written",
            "D_boundary_policy": "previous_equals_current_at_explicit_session_start_then_prior_frame",
            "previous_engine_frame_idx_sha256": _expected_previous_index_sha256(engine),
            "arrays": {
                "features_sha256": array_sha256(features),
                "engine_frame_idx_sha256": array_sha256(engine),
                "input_active_sha256": array_sha256(active),
            },
            "npz": shard.name,
            "npz_sha256": sha256_file(shard),
        }
        (out_dir / f"{session.session_id}.json").write_text(
            json.dumps(row, indent=2) + "\n"
        )
        rows.append(row)
    counts = {
        "videos": 2,
        "sessions": 2,
        "frames": 5,
        "train_videos": 1,
        "downstream_export_only_videos": 1,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "built_at": "2026-07-28T00:00:00+00:00",
        "arm": "D",
        "checkpoint": {
            "path": str(checkpoint.path),
            "sha256": checkpoint.sha256,
            "schema_version": CHECKPOINT_SCHEMA,
            "completed_steps": checkpoint.completed_steps,
            "horizons": list(checkpoint.horizons),
            "selection": "final_weights_only",
            "encoder_state": "final_ema_target_only",
        },
        "inventory": {"path": str(inventory.path), "sha256": inventory.sha256},
        "feature_format": "dynamics_d_final_ema_raw_avgpool_float16_v1",
        "feature_dim": REPRESENTATION_DIM,
        "dtype": "float16",
        "normalization": "none_raw_target_encoder_output",
        "supervision_phase": "label_free_feature_export_no_keys_member_read_or_written",
        "y4n_policy": "downstream_export_only_after_terminal_ssl_never_pretraining",
        "D_boundary_policy": "previous_equals_current_at_explicit_session_start_then_prior_frame",
        "counts": counts,
        "sessions": rows,
    }
    manifest_path = out_dir / "feature_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    completion = {
        "schema_version": COMPLETION_SCHEMA,
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "inventory_sha256": inventory.sha256,
        "checkpoint_sha256": checkpoint.sha256,
        "arm": "D",
        "counts": counts,
    }
    (out_dir / "feature_export_complete.json").write_text(
        json.dumps(completion, indent=2) + "\n"
    )


def test_deep_validator_checks_feature_only_outputs_and_d_policy(
    tmp_path: Path,
) -> None:
    payload, counts = _inventory_payload(tmp_path)
    inventory = validate_inventory_payload(
        payload, path=tmp_path / "inventory.json", sha256=SHA, expected_counts=counts
    )
    checkpoint = CheckpointContract(
        path=(tmp_path / "checkpoint.pt").resolve(),
        sha256="1" * 64,
        arm="D",
        horizons=(1, 2, 4),
        completed_steps=30_000,
        model_state={"unused": torch.tensor(1)},
    )
    out_dir = tmp_path / "features"
    _write_feature_artifacts(out_dir, inventory=inventory, checkpoint=checkpoint)
    report = validate_export(
        inventory=inventory,
        checkpoint=checkpoint,
        out_dir=out_dir,
        expected_counts=counts,
        deep_shards=True,
        deep_references=False,
        strict_checkpoint_state=False,
    )
    assert report["ok"] is True
    assert report["counts"]["checked_sessions"] == 2
    assert report["counts"]["checked_frames"] == 5

    row_path = out_dir / "nitrogenTrain__r000.json"
    row = json.loads(row_path.read_text())
    row["previous_engine_frame_idx_sha256"] = "f" * 64
    row_path.write_text(json.dumps(row, indent=2) + "\n")
    manifest_path = out_dir / "feature_export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sessions"][0] = row
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    completion_path = out_dir / "feature_export_complete.json"
    completion = json.loads(completion_path.read_text())
    completion["manifest_sha256"] = sha256_file(manifest_path)
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    failed = validate_export(
        inventory=inventory,
        checkpoint=checkpoint,
        out_dir=out_dir,
        expected_counts=counts,
        deep_shards=True,
        deep_references=False,
        strict_checkpoint_state=False,
    )
    assert failed["ok"] is False
    assert "previous-index policy mismatch" in failed["failures"][0]
