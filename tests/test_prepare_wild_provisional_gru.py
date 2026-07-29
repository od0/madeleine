import json
import math
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from data.schema import KEY_ORDER
from experiments import prepare_wild_provisional_gru as prepare


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_source_shard(
    path: Path,
    *,
    session_id: str,
    frames: int,
    first_frame: int,
) -> tuple[list[int], list[float]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = np.arange(first_frame, first_frame + frames, dtype=np.int64)
    pts = engine.astype(np.float64) / 60.0
    np.savez_compressed(
        path,
        frames=np.zeros((frames, 128, 128, 3), dtype=np.uint8),
        keys=np.zeros((frames, len(KEY_ORDER)), dtype=np.uint8),
        engine_frame_idx=engine,
        pts_s=pts,
        input_active=np.ones(frames, dtype=np.uint8),
        session_id=np.asarray(session_id),
    )
    return [int(engine[0]), int(engine[-1]) + 1], [float(pts[0]), float(pts[-1])]


def _write_feature_shard(
    path: Path,
    *,
    session_id: str,
    frames: int,
    source: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if source is None:
        keys = np.zeros((frames, len(KEY_ORDER)), dtype=np.uint8)
        engine = np.arange(frames, dtype=np.int64)
        active = np.ones(frames, dtype=np.uint8)
    else:
        with np.load(source, allow_pickle=False) as archive:
            keys = archive["keys"]
            engine = archive["engine_frame_idx"]
            active = archive["input_active"]
    np.savez_compressed(
        path,
        features=np.zeros((frames, prepare.FEATURE_DIM), dtype=np.float16),
        keys=keys,
        engine_frame_idx=engine,
        input_active=active,
        session_id=np.asarray(session_id),
    )


def _source_fixture(
    tmp_path: Path, frame_counts: tuple[int, ...]
) -> tuple[Path, Path, prepare.Expectations]:
    video_id = "video_a"
    source_root = tmp_path / "source"
    rows = []
    first_frame = 0
    for index, frames in enumerate(frame_counts):
        session_id = f"wild_provisional_{video_id}__r{index:03d}"
        relative = f"{video_id}/parts/{session_id}.npz"
        path = source_root / relative
        source_range, pts_range = _write_source_shard(
            path,
            session_id=session_id,
            frames=frames,
            first_frame=first_frame,
        )
        rows.append(
            {
                "session_id": session_id,
                "npz": path.name,
                "sha256": prepare.sha256_file(path),
                "frames": frames,
                "source_frame_range": source_range,
                "pts_range_s": pts_range,
                "path": relative,
                "npz_bytes": path.stat().st_size,
            }
        )
        first_frame += 1_000

    total_frames = sum(frame_counts)
    hours = total_frames / 60.0 / 3_600.0
    manifest = {
        "format_version": prepare.CORPUS_FORMAT,
        "admission_tier": prepare.ADMISSION_TIER,
        "builder": {"sha256": "fixture-builder-sha256"},
        "video_count": 1,
        "session_count": len(rows),
        "provisional_trainable_frames": total_frames,
        "provisional_trainable_hours": hours,
        "train_ready_frames": 0,
        "train_ready_hours": 0.0,
        "verification": {
            "explicit_video_set": [video_id],
            "expected_frame_shape": [128, 128, 3],
        },
        "videos": [
            {
                "video_id": video_id,
                "effective_grid_hz": 60.0,
                "part_count": len(rows),
                "provisional_trainable_frames": total_frames,
                "provisional_trainable_hours": hours,
                "parts": rows,
            }
        ],
    }
    manifest_path = tmp_path / "aggregate.json"
    _write_json(manifest_path, manifest)

    frame_span = 382
    eligible = [max(0, frames - frame_span + 1) for frames in frame_counts]
    segments = [windows // 96 for windows in eligible]
    expectations = prepare.Expectations(
        manifest_sha256=prepare.sha256_file(manifest_path),
        builder_sha256="fixture-builder-sha256",
        video_ids=(video_id,),
        video_count=1,
        session_count=len(frame_counts),
        provisional_frames=total_frames,
        provisional_hours=hours,
        expected_max_steps=math.ceil(sum(segments) / 16),
        eligible_windows=sum(eligible),
        segment_items=sum(segments),
        contributing_sessions=sum(windows > 0 for windows in eligible),
        too_short_sessions=sum(windows == 0 for windows in eligible),
        y4n_shard_sha256=(),
        y4n_later_ids=(),
        y4n_all_eval_windows=0,
        y4n_later_eval_windows=0,
        config_template_sha256=prepare.sha256_file(
            prepare.DEFAULT_CONFIG_TEMPLATE
        ),
    )
    return manifest_path, source_root, expectations


@pytest.mark.requires_private_artifacts(
    "results/wild20/provisional-broad7-af52cee/aggregate.json"
)
def test_tracked_manifest_and_one_pass_endpoint_are_frozen(tmp_path: Path) -> None:
    assert prepare.sha256_file(prepare.DEFAULT_MANIFEST) == (
        "67a95de6a4a49f504acdcfe8f316324fe55c2d7fb8e0ff55e82782f0fdecb01b"
    )
    manifest, parts = prepare.load_source_inventory(
        prepare.DEFAULT_MANIFEST,
        tmp_path / "source-not-needed-for-manifest-only-check",
        validate_files=False,
    )
    assert manifest["video_count"] == 7
    assert manifest["builder"]["sha256"] == (
        "4c08b135d81b95cee9d517943ec41021539c4502c0eb466655f19269104481d6"
    )
    assert len(parts) == 2_058
    assert sum(part.frames for part in parts) == 4_835_638
    assert manifest["provisional_trainable_hours"] == pytest.approx(
        22.387213054995033
    )
    assert manifest["train_ready_frames"] == 0
    assert manifest["train_ready_hours"] == 0.0

    config = prepare._load_gru_template(
        prepare.DEFAULT_CONFIG_TEMPLATE, prepare.FROZEN_EXPECTATIONS
    )
    endpoint = prepare._compute_endpoint(parts, config)
    assert endpoint == {
        "frame_span": 382,
        "eligible_windows": 4_076_727,
        "contributing_sessions": 1_729,
        "too_short_sessions": 329,
        "segment_windows": 96,
        "segment_items": 41_567,
        "used_training_windows": 3_990_432,
        "discarded_eligible_tail_windows": 86_295,
        "loader_batch_items": 16,
        "max_steps": 2_598,
    }


def test_plan_and_assemble_publish_exact_provisional_hardlink_view(
    tmp_path: Path,
) -> None:
    manifest_path, source_root, expectations = _source_fixture(
        tmp_path, (478, 478)
    )
    plan_dir = tmp_path / "plan"
    source_receipt = prepare.prepare_source_inputs(
        manifest_path,
        source_root,
        plan_dir,
        expectations=expectations,
    )
    assert source_receipt["admission_tier"] == prepare.ADMISSION_TIER
    assert source_receipt["train_ready_frames"] == 0
    assert [worker["sessions"] for worker in source_receipt["workers"]] == [1, 1]

    _, parts = prepare.load_source_inventory(
        manifest_path, source_root, expectations=expectations
    )
    worker_parts = prepare.partition_two_workers(parts)
    worker_roots = [tmp_path / "features_0", tmp_path / "features_1"]
    for worker_index, expected_parts in enumerate(worker_parts):
        root = worker_roots[worker_index]
        root.mkdir()
        reports = []
        for part in expected_parts:
            output = root / f"{part.session_id}.npz"
            _write_feature_shard(
                output,
                session_id=part.session_id,
                frames=part.frames,
                source=part.source_path,
            )
            reports.append(
                {
                    "session_id": part.session_id,
                    "frames": part.frames,
                    "source": str(part.source_path),
                    "npz": output.name,
                    "resumed": False,
                }
            )
        _write_json(
            root / "feature_build_manifest.json",
            {
                "format": prepare.FEATURE_FORMAT,
                "backbone_feature_dim": prepare.FEATURE_DIM,
                "frame_size": 128,
                "source_kind": "audited_rgb_shards",
                "sessions": reports,
            },
        )

    y4n_root = tmp_path / "y4n"
    y4n_ids = ("mapped_eval__r000", "mapped_eval__r001")
    y4n_rows = []
    for session_id in y4n_ids:
        path = y4n_root / f"{session_id}.npz"
        _write_feature_shard(path, session_id=session_id, frames=478)
        y4n_rows.append((session_id, prepare.sha256_file(path)))
    expectations = replace(
        expectations,
        y4n_shard_sha256=tuple(y4n_rows),
        y4n_later_ids=(y4n_ids[1],),
        y4n_all_eval_windows=194,
        y4n_later_eval_windows=97,
    )

    output = tmp_path / "assembled"
    receipt = prepare.assemble_data_view(
        manifest_path=manifest_path,
        source_root=source_root,
        plan_dir=plan_dir,
        worker_feature_roots=worker_roots,
        y4n_data=y4n_root,
        config_template=prepare.DEFAULT_CONFIG_TEMPLATE,
        output=output,
        expectations=expectations,
    )

    assert receipt["training_label_kind"] == prepare.SOURCE_LABEL_KIND
    assert receipt["admission_tier"] == prepare.ADMISSION_TIER
    assert receipt["train_ready_frames"] == 0
    assert receipt["recipe"]["max_steps"] == 1
    assert receipt["recipe"]["used_training_windows"] == 192
    assert receipt["split"]["train_validation_overlap"] is False
    assert (output / "train_sessions.txt").read_text().splitlines() == sorted(
        part.session_id for part in parts
    )
    assert (output / "val_sessions.txt").read_text().splitlines() == list(
        y4n_ids
    )
    assert (output / "later_eight_sessions.txt").read_text().splitlines() == [
        y4n_ids[1]
    ]
    config = json.loads((output / "config.json").read_text())
    assert config["max_steps"] == config["eval_interval"] == 1
    for worker_index, expected_parts in enumerate(worker_parts):
        for part in expected_parts:
            assert os.path.samefile(
                worker_roots[worker_index] / f"{part.session_id}.npz",
                output / f"{part.session_id}.npz",
            )
    for session_id in y4n_ids:
        assert os.path.samefile(
            y4n_root / f"{session_id}.npz",
            output / f"{session_id}.npz",
        )


def test_plan_rejects_source_bytes_that_no_longer_match_manifest(
    tmp_path: Path,
) -> None:
    manifest_path, source_root, expectations = _source_fixture(tmp_path, (1,))
    manifest = json.loads(manifest_path.read_text())
    relative = manifest["videos"][0]["parts"][0]["path"]
    with (source_root / relative).open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="source byte size changed"):
        prepare.prepare_source_inputs(
            manifest_path,
            source_root,
            tmp_path / "plan",
            expectations=expectations,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("extra", "source NPZ inventory changed"),
        ("missing", "source NPZ inventory changed"),
        ("symlink", "source corpus contains symlinks"),
    ),
)
def test_plan_rejects_nonexact_or_symlinked_source_inventory(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest_path, source_root, expectations = _source_fixture(tmp_path, (1,))
    manifest = json.loads(manifest_path.read_text())
    declared = source_root / manifest["videos"][0]["parts"][0]["path"]
    if mutation == "extra":
        (source_root / "unexpected.npz").write_bytes(b"unexpected")
    elif mutation == "missing":
        declared.unlink()
    else:
        target = source_root / "symlink-target.bin"
        declared.replace(target)
        declared.symlink_to(target)

    with pytest.raises(ValueError, match=message):
        prepare.prepare_source_inputs(
            manifest_path,
            source_root,
            tmp_path / "plan",
            expectations=expectations,
        )
