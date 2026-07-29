from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import experiments.pretrain_dynamics as trainer
from badeline.dynamics_pretraining import EMADynamicsPretrainer, REPRESENTATION_DIM
from data.schema import KEY_ORDER
from experiments.pretrain_dynamics import (
    BalancedCandidateSampler,
    RGBShard,
    augment_ordered_tuple,
    build_candidate_index,
    linear_ema_momentum,
    load_explicit_rgb_shards,
    load_final_checkpoint,
    sha256_file,
    train_fixed_steps,
)


class _TinyFrameEncoder(nn.Module):
    output_dim = REPRESENTATION_DIM

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, REPRESENTATION_DIM, bias=False)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        return self.projection(current.mean(dim=(-2, -1)))


class _TinyPairEncoder(nn.Module):
    output_dim = REPRESENTATION_DIM

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(6, REPRESENTATION_DIM, bias=False)

    def forward(
        self, previous: torch.Tensor, current: torch.Tensor
    ) -> torch.Tensor:
        value = torch.cat(
            (previous.mean(dim=(-2, -1)), current.mean(dim=(-2, -1))), dim=-1
        )
        return self.projection(value)


def _frames(count: int) -> np.ndarray:
    """Masked-looking synthetic pixels with both static and motion transitions."""

    frames = np.zeros((count, 128, 128, 3), dtype=np.uint8)
    for index in range(count):
        # Alternating duplicated frames create a genuine score-zero stratum;
        # moving rectangles of different intensities create variable changes.
        phase = index // 2
        frames[index, 8 + phase : 24 + phase, 12:48, 0] = 20 + 10 * phase
        frames[index, 40:80, 30 + phase : 44 + phase, 1] = 160
    return frames


def _write_shard(
    root: Path,
    session_id: str,
    *,
    count: int = 18,
    engine: np.ndarray | None = None,
    active: np.ndarray | None = None,
    keys_fill: int = 255,
) -> Path:
    if engine is None:
        engine = np.arange(count, dtype=np.int64)
    if active is None:
        active = np.ones(count, dtype=np.uint8)
    path = root / f"{session_id}.npz"
    np.savez_compressed(
        path,
        frames=_frames(count),
        # Deliberately non-binary.  The SSL loader validates only the schema and
        # must not consume engine-truth action values.
        keys=np.full((count, len(KEY_ORDER)), keys_fill, dtype=np.uint8),
        engine_frame_idx=engine,
        input_active=active,
        session_id=np.asarray(session_id),
    )
    return path


def _load_fixture(tmp_path: Path, session_id: str = "train-a") -> list[RGBShard]:
    path = _write_shard(tmp_path, session_id)
    return load_explicit_rgb_shards(
        [path],
        allowed_session_ids=[session_id],
        expected_sha256={session_id: sha256_file(path)},
    )


def test_explicit_loader_uses_exact_allowlist_hash_and_never_validates_key_values(
    tmp_path: Path,
) -> None:
    path = _write_shard(tmp_path, "train-a", keys_fill=255)
    digest = sha256_file(path)

    shards = load_explicit_rgb_shards(
        [path],
        allowed_session_ids=["train-a"],
        expected_sha256={"train-a": digest},
    )

    assert [shard.session_id for shard in shards] == ["train-a"]
    assert shards[0].sha256 == digest
    assert not hasattr(shards[0], "keys")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_explicit_rgb_shards(
            [path],
            allowed_session_ids=["train-a"],
            expected_sha256={"train-a": "0" * 64},
        )
    with pytest.raises(ValueError, match="count"):
        load_explicit_rgb_shards(
            [path], allowed_session_ids=["train-a", "train-b"]
        )


def test_embargo_is_rejected_before_path_access(tmp_path: Path) -> None:
    forbidden = tmp_path / "rec_20260727_220000_test.npz"
    assert not forbidden.exists()
    with pytest.raises(ValueError, match="embargoed"):
        load_explicit_rgb_shards(
            [forbidden], allowed_session_ids=["rec_20260727_220000_test"]
        )


def test_schema_and_active_values_fail_closed(tmp_path: Path) -> None:
    session_id = "bad-active"
    active = np.ones(10, dtype=np.uint8)
    active[4] = 2
    path = _write_shard(tmp_path, session_id, count=10, active=active)
    with pytest.raises(ValueError, match="input_active must be binary"):
        load_explicit_rgb_shards([path], allowed_session_ids=[session_id])


def test_candidates_never_cross_engine_gaps_or_inactive_rows(tmp_path: Path) -> None:
    engine = np.asarray([0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25], dtype=np.int64)
    active = np.ones(len(engine), dtype=np.uint8)
    active[3] = 0
    path = _write_shard(
        tmp_path,
        "gapped",
        count=len(engine),
        engine=engine,
        active=active,
    )
    shard = load_explicit_rgb_shards([path], allowed_session_ids=["gapped"])[0]

    candidates = build_candidate_index([shard], arm="C", horizons=(1, 2))

    # The first row of each run follows the frozen duplicate-predecessor rule.
    first_by_run = np.flatnonzero(
        candidates.online_previous == candidates.online_current
    )
    assert len(first_by_run) >= 2
    for row in range(len(candidates)):
        start = int(candidates.online_previous[row])
        stop = int(candidates.target_current[row])
        assert np.all(shard.input_active[start : stop + 1] == 1)
        assert np.all(np.diff(shard.engine_frame_idx[start : stop + 1]) == 1)
        assert (
            shard.engine_frame_idx[candidates.target_current[row]]
            - shard.engine_frame_idx[candidates.online_current[row]]
            == candidates.horizon[row]
        )
    assert set(candidates.stratum.tolist()) == {0, 1}


def test_same_frame_control_uses_exact_future_arm_sampling_population(
    tmp_path: Path,
) -> None:
    shards = _load_fixture(tmp_path)
    same_frame = build_candidate_index(shards, arm="B", horizons=(1, 2))
    future = build_candidate_index(shards, arm="C", horizons=(1, 2))

    for name in (
        "session",
        "online_previous",
        "online_current",
        "horizon",
        "motion_score",
        "stratum",
    ):
        assert np.array_equal(getattr(same_frame, name), getattr(future, name))
    assert np.array_equal(
        same_frame.target_current, same_frame.online_current
    )
    assert np.array_equal(
        future.target_current - future.online_current,
        future.horizon.astype(np.int64),
    )


def test_sampler_has_exact_deterministic_horizon_and_stratum_quotas(
    tmp_path: Path,
) -> None:
    shards = _load_fixture(tmp_path)
    candidates = build_candidate_index(shards, arm="C", horizons=(1, 2))
    first = BalancedCandidateSampler(
        candidates, horizons=(1, 2), batch_size=8, seed=9
    )
    second = BalancedCandidateSampler(
        candidates, horizons=(1, 2), batch_size=8, seed=9
    )

    first_batch = first.next_batch()
    second_batch = second.next_batch()

    assert np.array_equal(first_batch, second_batch)
    for horizon in (1, 2):
        for stratum in (0, 1):
            assert int(
                np.sum(
                    (candidates.horizon[first_batch] == horizon)
                    & (candidates.stratum[first_batch] == stratum)
                )
            ) == 2
    assert set(first.draw_counts.values()) == {2}


def test_ordered_augmentation_is_time_consistent_and_does_not_flip() -> None:
    image = torch.linspace(0.0, 1.0, 16).view(1, 1, 1, 4, 4).repeat(2, 2, 3, 1, 1)
    augmented = augment_ordered_tuple(
        image,
        generator=torch.Generator().manual_seed(4),
        crop_scale_min=1.0,
        brightness=0.0,
        contrast=0.0,
    )

    assert torch.equal(augmented[:, 0], augmented[:, 1])
    # With a full crop and no color jitter, the asymmetric ramp retains its
    # original left-to-right orientation (there is no hidden flip branch).
    torch.testing.assert_close(augmented, image)


def test_linear_ema_hits_frozen_endpoints() -> None:
    assert linear_ema_momentum(0, 5) == 0.998
    assert linear_ema_momentum(4, 5) == 1.0
    assert linear_ema_momentum(0, 1) == 0.998
    with pytest.raises(ValueError, match="step"):
        linear_ema_momentum(5, 5)


@pytest.mark.parametrize(
    ("arm", "horizons", "encoder_factory"),
    [
        ("B", (1, 2), _TinyFrameEncoder),
        ("C", (1, 2), _TinyFrameEncoder),
        ("D", (1, 2), _TinyPairEncoder),
    ],
)
def test_tiny_cpu_smoke_is_atomic_and_checkpoint_round_trips(
    tmp_path: Path,
    arm: str,
    horizons: tuple[int, ...],
    encoder_factory: type[nn.Module],
) -> None:
    shards = _load_fixture(tmp_path)
    candidates = build_candidate_index(
        shards, arm=arm, horizons=horizons  # type: ignore[arg-type]
    )
    torch.manual_seed(17)
    model = EMADynamicsPretrainer(
        arm,  # type: ignore[arg-type]
        horizons=horizons,
        online_encoder=encoder_factory(),
    )
    output_dir = tmp_path / f"smoke-{arm}"

    result = train_fixed_steps(
        shards=shards,
        candidates=candidates,
        model=model,
        arm=arm,  # type: ignore[arg-type]
        horizons=horizons,
        output_dir=output_dir,
        steps=2,
        batch_size=4 if len(horizons) == 2 else 2,
        learning_rate=1e-3,
        weight_decay=0.01,
        seed=3,
        device=torch.device("cpu"),
        diagnostic_interval=1,
        initialization_receipt={"kind": "synthetic-test"},
    )

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "final.pt",
        "run_receipt.json",
    ]
    receipt = json.loads((output_dir / "run_receipt.json").read_text())
    assert receipt["completed"] is True
    assert receipt["data_contract"]["labels_consumed"] is False
    assert receipt["sampling"]["equal_horizon_quota"] is True
    assert receipt["sampling"]["equal_static_change_quota"] is True
    assert receipt["loss"]["steps"] == 2
    assert len(receipt["collapse_diagnostics"]) == 2
    assert receipt["artifacts"]["checkpoint_sha256"] == sha256_file(
        output_dir / "final.pt"
    )
    assert result["checkpoint"] == str(output_dir / "final.pt")

    torch.manual_seed(999)
    reloaded = EMADynamicsPretrainer(
        arm,  # type: ignore[arg-type]
        horizons=horizons,
        online_encoder=encoder_factory(),
    )
    payload = load_final_checkpoint(output_dir / "final.pt", reloaded)
    assert payload["completed_steps"] == 2
    for expected, actual in zip(
        model.state_dict().values(), reloaded.state_dict().values(), strict=True
    ):
        assert torch.equal(expected.cpu(), actual.cpu())

    with pytest.raises(FileExistsError, match="overwrite"):
        train_fixed_steps(
            shards=shards,
            candidates=candidates,
            model=model,
            arm=arm,  # type: ignore[arg-type]
            horizons=horizons,
            output_dir=output_dir,
            steps=1,
            batch_size=4 if len(horizons) == 2 else 2,
            learning_rate=1e-3,
            weight_decay=0.01,
            seed=3,
            device=torch.device("cpu"),
        )


def test_receipt_checkpoint_hash_is_full_sha256(tmp_path: Path) -> None:
    # Small independent assertion that the helper is not using torchvision's
    # abbreviated filename hash convention for released artifacts.
    path = tmp_path / "bytes"
    path.write_bytes(b"madeleine")
    assert sha256_file(path) == hashlib.sha256(b"madeleine").hexdigest()


def test_imagenet_weight_cache_is_full_hash_validated_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"exact-preregistered-weight-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    cached = checkpoints / "fixture.pth"
    cached.write_bytes(payload)

    class _Weights:
        url = "https://download.pytorch.org/models/fixture.pth"

        @staticmethod
        def get_state_dict(*, progress: bool, check_hash: bool) -> dict[str, object]:
            assert progress is True
            assert check_hash is True
            return {}

    monkeypatch.setattr(trainer, "RESNET18_WEIGHTS", _Weights())
    monkeypatch.setattr(trainer, "RESNET18_WEIGHTS_URL", _Weights.url)
    monkeypatch.setattr(trainer, "RESNET18_WEIGHTS_SHA256", digest)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))

    receipt = trainer.validate_imagenet_initialization()

    assert receipt["sha256"] == digest
    assert receipt["validated_before_step_one"] is True
    cached.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="SHA-256"):
        trainer.validate_imagenet_initialization()
