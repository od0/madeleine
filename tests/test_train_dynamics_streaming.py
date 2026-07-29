from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from badeline.dynamics_pretraining import EMADynamicsPretrainer, REPRESENTATION_DIM
import experiments.train_dynamics_streaming as trainer


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
            (
                previous.mean(dim=(-2, -1)),
                current.mean(dim=(-2, -1)),
            ),
            dim=-1,
        )
        return self.projection(value)


def _tiny_model(
    config: trainer.TrainConfig,
) -> tuple[EMADynamicsPretrainer, dict[str, object]]:
    encoder: nn.Module = (
        _TinyPairEncoder() if config.arm == "D" else _TinyFrameEncoder()
    )
    model = EMADynamicsPretrainer(
        config.arm,
        horizons=config.horizons,
        weights=None,
        online_encoder=encoder,
    )
    trainer.initialize_shared_predictor(model, config.seed)
    return model, {"kind": "synthetic-test-initialization"}


def _write_cache(root: Path) -> Path:
    root.mkdir()
    horizons = (1, 2, 4)
    span = max(horizons) + 2
    source_names = np.asarray(("nitrogen", "wild_provisional", "local"))
    windows: list[tuple[int, int, int]] = []
    for source_id in range(3):
        for stratum in (0, 1):
            for replica in range(3):
                windows.append((source_id, stratum, replica))

    rgb = np.zeros((len(windows) * span, 128, 128, 3), dtype=np.uint8)
    for window_id, (source_id, stratum, replica) in enumerate(windows):
        base = window_id * span
        for offset in range(span):
            rgb[base + offset, :, :, 0] = (
                7 + 29 * source_id + 13 * replica + offset * (1 + stratum)
            )
            rgb[base + offset, :, :, 1] = (
                17 + 11 * source_id + 19 * replica + offset * (2 + stratum)
            )
            rgb[base + offset, :, :, 2] = (
                31 + 5 * source_id + 23 * replica + offset * (3 + stratum)
            )
            rgb[base + offset, 16:48, 8 + offset : 24 + offset, source_id] += 15
    rgb_path = root / "rgb.npy"
    np.save(rgb_path, rgb)

    count = len(windows) * len(horizons)
    arrays = {
        name: np.empty(count, dtype=dtype)
        for name, dtype in trainer.INDEX_DTYPES.items()
    }
    cursor = 0
    session_ids = np.asarray([f"training-session-{index}" for index in range(len(windows))])
    for window_id, (source_id, stratum, _replica) in enumerate(windows):
        base = window_id * span
        for horizon in horizons:
            arrays["tuple_id"][cursor] = np.uint64(cursor + 1000)
            arrays["window_id"][cursor] = window_id
            arrays["source_id"][cursor] = source_id
            arrays["session_index"][cursor] = window_id
            arrays["run_id"][cursor] = 0
            arrays["anchor_engine_frame"][cursor] = 100 + window_id
            arrays["online_previous"][cursor] = base
            arrays["online_current"][cursor] = base + 1
            arrays["target_previous"][cursor] = base + horizon
            arrays["target_current"][cursor] = base + horizon + 1
            arrays["horizon"][cursor] = horizon
            arrays["motion_score"][cursor] = np.float32(stratum + 0.1)
            arrays["stratum"][cursor] = stratum
            cursor += 1
    index_path = root / "index.npz"
    np.savez(index_path, **arrays, session_ids=session_ids, source_names=source_names)

    probabilities = (0.5, 0.3, 0.2)
    manifest = {
        "schema_version": trainer.CACHE_SCHEMA,
        "status": "complete",
        "inventory": {
            "path": "/synthetic/training-inventory.json",
            "sha256": "1" * 64,
            "schema_version": "synthetic-inventory.v1",
        },
        "horizons": list(horizons),
        "window_span_frames": span,
        "windows": len(windows),
        "tuples": count,
        "sources": [
            {
                "name": str(source_names[source_id]),
                "source_id": source_id,
                "eligible_probability": probabilities[source_id],
            }
            for source_id in range(3)
        ],
        "labels": {
            "loaded": False,
            "arrays_accessed": [],
            "boundary_metadata_accessed": [
                "engine_frame_idx",
                "input_active",
                "session_id",
            ],
        },
        "exclusion_proof": {
            "validated_before_source_access": True,
            "whole_y4n_absent": True,
            "val_a_absent": True,
            "val_b_absent": True,
            "b1_absent": True,
            "sealed_untouched_absent": True,
            "forbidden_ids": sorted(
                {*trainer.FORBIDDEN_EXACT_IDS, trainer.FORBIDDEN_VIDEO_ID}
            ),
        },
        "artifacts": {
            "rgb": {
                "path": "rgb.npy",
                "sha256": trainer.sha256_file(rgb_path),
                "bytes": rgb_path.stat().st_size,
                "shape": list(rgb.shape),
                "dtype": "uint8",
                "c_order": True,
            },
            "index": {
                "path": "index.npz",
                "sha256": trainer.sha256_file(index_path),
                "bytes": index_path.stat().st_size,
                "rows": count,
                "fields": {
                    name: str(value.dtype)
                    for name, value in {
                        **arrays,
                        "session_ids": session_ids,
                        "source_names": source_names,
                    }.items()
                },
            },
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _config(*, arm: trainer.DynamicsArm = "C") -> trainer.TrainConfig:
    return trainer.TrainConfig(
        arm=arm,
        horizons=(1, 2, 4),
        max_steps=4,
        global_batch_size=12,
        microbatch_size=6,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=7,
        schedule_seed=19,
        checkpoint_interval=2,
        diagnostic_interval=2,
        panel_size=12,
        prefetch_depth=2,
        throughput_steps=2,
        max_projected_seconds=0.0,
        collapse_relative_floor=1e-9,
        collapse_effective_rank_min=1e-9,
        collapse_mean_cosine_max=0.9999999,
        collapse_nn_unique_fraction_min=1e-9,
        collapse_consecutive_failures=3,
        max_cuda_memory_gib=76.0,
        device="cpu",
        study_id="synthetic-streaming-test",
        run_id=f"synthetic-{arm.lower()}",
        implementation_commit="2" * 40,
    )


def test_cache_rejects_forbidden_source_before_numpy_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _write_cache(tmp_path / "cache")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["name"] = "rec_20260727_220000_test"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("cache bytes were opened before identity validation")

    monkeypatch.setattr(trainer.np, "load", forbidden_load)
    with pytest.raises(trainer.CacheContractError, match="forbidden identity"):
        trainer.load_cache(manifest_path, expected_horizons=(1, 2, 4))


def test_counter_schedule_is_matched_balanced_and_source_fair(tmp_path: Path) -> None:
    cache = trainer.load_cache(
        _write_cache(tmp_path / "cache"), expected_horizons=(1, 2, 4)
    )
    left = trainer.CounterSchedule(
        cache.index,
        horizons=(1, 2, 4),
        batch_size=12,
        seed=19,
        max_steps=30,
        source_probabilities=cache.source_probabilities,
    )
    right = trainer.CounterSchedule(
        cache.index,
        horizons=(1, 2, 4),
        batch_size=12,
        seed=19,
        max_steps=30,
        source_probabilities=cache.source_probabilities,
    )
    for step in range(30):
        left_rows = left.rows_for_step(step)
        assert np.array_equal(left_rows, right.rows_for_step(step))
        for horizon in (1, 2, 4):
            for stratum in (0, 1):
                mask = (
                    (cache.index["horizon"][left_rows] == horizon)
                    & (cache.index["stratum"][left_rows] == stratum)
                )
                assert int(mask.sum()) == 2
    receipt = left.receipt(cache.index["tuple_id"])
    assert receipt["canonical_plan_sha256"] == right.receipt(
        cache.index["tuple_id"]
    )["canonical_plan_sha256"]
    errors = [
        error
        for cell in receipt["cells"].values()
        for error in cell["maximum_cumulative_allocation_error_by_source"].values()
    ]
    assert max(errors) <= 1.0


def test_fixed_panel_is_exact_unique_and_nearly_cell_balanced(tmp_path: Path) -> None:
    cache = trainer.load_cache(
        _write_cache(tmp_path / "cache"), expected_horizons=(1, 2, 4)
    )
    schedule = trainer.CounterSchedule(
        cache.index,
        horizons=(1, 2, 4),
        batch_size=12,
        seed=19,
        max_steps=4,
        source_probabilities=cache.source_probabilities,
    )
    rows = trainer.fixed_panel_rows(schedule, total=13)
    assert len(rows) == len(np.unique(rows)) == 13
    cell_counts = []
    for horizon in (1, 2, 4):
        for stratum in (0, 1):
            cell_counts.append(
                int(
                    np.sum(
                        (cache.index["horizon"][rows] == horizon)
                        & (cache.index["stratum"][rows] == stratum)
                    )
                )
            )
    assert max(cell_counts) - min(cell_counts) == 1


def test_prefetch_and_temporally_consistent_augmentation(tmp_path: Path) -> None:
    cache = trainer.load_cache(
        _write_cache(tmp_path / "cache"), expected_horizons=(1, 2, 4)
    )
    schedule = trainer.CounterSchedule(
        cache.index,
        horizons=(1, 2, 4),
        batch_size=12,
        seed=19,
        max_steps=2,
        source_probabilities=cache.source_probabilities,
    )
    with trainer.BatchPrefetcher(
        cache,
        schedule,
        start_step=0,
        stop_step=2,
        depth=1,
        pin_memory=False,
    ) as batches:
        batch = next(batches)
        assert batch.step == 0
        assert batch.online_current.shape == (12, 128, 128, 3)
        assert torch.equal(batch.rows, torch.from_numpy(schedule.rows_for_step(0)))

    frame = torch.rand((4, 3, 32, 32))
    repeated = torch.stack((frame, frame), dim=1)
    generator = torch.Generator(device="cpu").manual_seed(11)
    augmented = trainer.augment_tuple_vectorized(repeated, generator=generator)
    assert torch.equal(augmented[:, 0], augmented[:, 1])


def test_collapse_gate_requires_three_consecutive_failures() -> None:
    good_metrics = {
        "per_dimension_std_median": 1.0,
        "effective_rank": 10.0,
        "mean_pairwise_cosine": 0.1,
        "nearest_neighbor_unique_fraction": 0.8,
    }
    good = {"representations": {"online": good_metrics, "target": good_metrics}}
    state, gate = trainer.apply_collapse_gate(
        good,
        None,
        relative_floor=0.25,
        effective_rank_min=8.0,
        mean_cosine_max=0.995,
        nn_unique_fraction_min=0.25,
    )
    assert gate["status"] == "pass"
    bad_metrics = {
        **good_metrics,
        "per_dimension_std_median": 0.1,
        "effective_rank": 2.0,
        "mean_pairwise_cosine": 0.999,
        "nearest_neighbor_unique_fraction": 0.1,
    }
    bad = {"representations": {"online": bad_metrics, "target": bad_metrics}}
    for expected in (1, 2, 3):
        state, gate = trainer.apply_collapse_gate(
            bad,
            state,
            relative_floor=0.25,
            effective_rank_min=8.0,
            mean_cosine_max=0.995,
            nn_unique_fraction_min=0.25,
        )
        assert gate["consecutive_failures"] == expected


def test_exact_resume_preserves_state_and_failed_forensics(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path / "cache")
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    config = _config()
    trainer.run_training(
        manifest_path=manifest,
        output_dir=uninterrupted,
        config=config,
        model_factory=_tiny_model,
    )

    def stop_after_checkpoint(step: int) -> None:
        if step == 2:
            raise RuntimeError("synthetic interruption")

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        trainer.run_training(
            manifest_path=manifest,
            output_dir=resumed,
            config=config,
            model_factory=_tiny_model,
            step_hook=stop_after_checkpoint,
        )
    checkpoint = resumed / "resume" / "step_00000002.pt"
    assert checkpoint.is_file()
    assert list(resumed.glob("failure_*.json"))
    assert list(resumed.glob("failed_state_*.pt"))
    trainer.run_training(
        manifest_path=manifest,
        output_dir=resumed,
        config=config,
        resume_checkpoint=checkpoint,
        model_factory=_tiny_model,
    )

    full_payload = torch.load(
        uninterrupted / "final.pt", map_location="cpu", weights_only=False
    )
    resumed_payload = torch.load(
        resumed / "final.pt", map_location="cpu", weights_only=False
    )
    assert full_payload["losses"] == resumed_payload["losses"]
    assert full_payload["rng_state"]["online_augmentation"].equal(
        resumed_payload["rng_state"]["online_augmentation"]
    )
    for name, value in full_payload["model_state"].items():
        assert torch.equal(value, resumed_payload["model_state"][name]), name
    audit = trainer.validate_completed_run(resumed)
    assert audit["ok"] is True
    assert audit["preserved_prior_failures"]["failure_receipts"]


def test_cli_defaults_populate_complete_train_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_training(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(trainer, "run_training", fake_run_training)
    status = trainer.main(
        [
            "train",
            "--cache-manifest",
            "cache/manifest.json",
            "--output-dir",
            "run-c",
            "--arm",
            "C",
            "--horizons",
            "1,2,4",
            "--max-steps",
            "30000",
            "--max-projected-seconds",
            "28800",
            "--study-id",
            "study",
            "--run-id",
            "arm-c",
            "--implementation-commit",
            "3" * 40,
        ]
    )
    assert status == 0
    config = captured["config"]
    assert isinstance(config, trainer.TrainConfig)
    assert config.global_batch_size == 192
    assert config.schedule_seed == 2026072802
    assert config.collapse_effective_rank_min == 8.0
    assert config.collapse_mean_cosine_max == 0.995
    assert config.collapse_nn_unique_fraction_min == 0.25
    assert config.max_cuda_memory_gib == 76.0


def test_resume_rejects_any_recipe_change(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path / "cache")
    output = tmp_path / "interrupted"

    def stop(step: int) -> None:
        if step == 2:
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        trainer.run_training(
            manifest_path=manifest,
            output_dir=output,
            config=_config(),
            model_factory=_tiny_model,
            step_hook=stop,
        )
    with pytest.raises(ValueError, match="config differs"):
        trainer.run_training(
            manifest_path=manifest,
            output_dir=output,
            config=replace(_config(), learning_rate=2e-3),
            resume_checkpoint=output / "resume" / "step_00000002.pt",
            model_factory=_tiny_model,
        )
