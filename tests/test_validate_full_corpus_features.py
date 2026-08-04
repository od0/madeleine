import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.schema import KEY_ORDER
from experiments.build_full_corpus_features import build_chunk_frames
from experiments.validate_full_corpus_features import (
    CorpusExpectations,
    EXPECTED_DIRECTION_RULE,
    MAPPING_REPORT_SCHEMA_VERSION,
    NATIVE_MODE,
    VideoMetadata,
    validate_full_corpus_features,
)


def _write_shard(path: Path, *, finite: bool = True) -> None:
    features = np.zeros((240, 512), dtype=np.float16)
    if not finite:
        features[0, 0] = np.nan
    np.savez(
        path,
        features=features,
        keys=np.zeros((240, len(KEY_ORDER)), dtype=np.uint8),
        engine_frame_idx=np.arange(240, dtype=np.int64),
        input_active=np.ones(240, dtype=np.uint8),
        session_id=np.asarray("video_a__r000"),
    )


def _fixture(tmp_path: Path) -> dict:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    video = raw_root / "video_a.mp4"
    video.write_bytes(b"fixture-video-metadata-is-injected")
    fetch_report = raw_root / "fetch60_report.jsonl"
    fetch_report.write_text(json.dumps({
        "video_id": "video_a",
        "aligned_1to1": True,
        "fps": 60.0,
        "width": 16,
        "height": 16,
    }) + "\n")

    chunk_index = tmp_path / "chunk_index.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "video_id": "video_a",
        "chunk_id": 0,
        "chunk_size": 1_200,
        "grid_hz": 60.0,
        "metadata_resolution_w": 16,
        "metadata_resolution_h": 16,
        "bbox_x": 0,
        "bbox_y": 12,
        "bbox_w": 4,
        "bbox_h": 4,
    }]), chunk_index)

    mapped_root = tmp_path / "mapped"
    chunk_frames = mapped_root / "chunk_frames.parquet"
    build_chunk_frames(chunk_index, {"video_a"}, chunk_frames)
    mapped_video = mapped_root / "video_a"
    label_dir = mapped_video / "video_a_chunk_0000"
    label_dir.mkdir(parents=True)
    label_columns = {
        "frame_idx": np.arange(1_200, dtype=np.int64),
        **{key: np.zeros(1_200, dtype=np.uint8) for key in KEY_ORDER},
    }
    pq.write_table(pa.table(label_columns), label_dir / "labels_native.parquet")
    mapping = {
        "schema_version": MAPPING_REPORT_SCHEMA_VERSION,
        "video_id": "video_a",
        "bind_map": {"jump": ["south"], "dash": ["west"], "grab": ["east"]},
        "confidence": 0.75,
        "flagged": False,
        "direction_rule": EXPECTED_DIRECTION_RULE,
        "chunks_mapped": 1,
        "chunks_skipped": 0,
    }
    (mapped_video / "mapping_report.json").write_text(json.dumps(mapping))

    feature_root = tmp_path / "features_by_video"
    feature_video = feature_root / "video_a"
    feature_video.mkdir(parents=True)
    source_shard = feature_video / "video_a__r000.npz"
    _write_shard(source_shard)
    feature_report = {
        "video_id": "video_a",
        "video": {
            "average_fps": 60.0,
            "decoded_frames": 240,
            "nominal_timeline_frames": 240,
        },
        "decoder_mode": NATIVE_MODE,
        "bind_map": mapping["bind_map"],
        "bind_confidence": 0.75,
        "runs": 1,
        "parts": [{
            "session_id": "video_a__r000",
            "frames": 240,
            "source_frame_range": [0, 240],
            "npz": "video_a__r000.npz",
            "decoder_mode": NATIVE_MODE,
            "imputed_tail_frames": 0,
        }],
        "imputed_tail_frames": 0,
        "tail_truncated_frames": 960,
        "skipped_short_frames": 0,
    }
    (feature_video / "feature_build_manifest.json").write_text(json.dumps({
        "format": "resnet18_imagenet_avgpool_float16_v1",
        "source_kind": "mapped_foreign_video",
        "videos": [feature_report],
    }))

    output_root = tmp_path / "full_corpus_features"
    output_root.mkdir()
    os.link(source_shard, output_root / source_shard.name)
    (output_root / "train_sessions.txt").write_text("video_a__r000\n")
    (output_root / "unflagged_sessions.txt").write_text("video_a__r000\n")
    (output_root / "val_sessions.txt").write_text("")
    long_context_fraction = 819 / 1_200
    manifest = {
        "format": "resnet18_imagenet_avgpool_float16_v1",
        "source_kind": "mapped_foreign_video",
        "video_count": 1,
        "session_count": 1,
        "train_frames": 240,
        "train_label_hours_at_60hz": 240 / 216_000,
        "source_label_frames": 1_200,
        "source_label_hours_at_60hz": 1_200 / 216_000,
        "train_to_source_fraction": 240 / 1_200,
        "decoder_mode_counts": {NATIVE_MODE: 1},
        "imputed_tail_frames": 0,
        "tail_truncated_frames": 960,
        "skipped_short_frames": 0,
        "unflagged_video_count": 1,
        "unflagged_session_count": 1,
        "videos": [{
            "video_id": "video_a",
            "frames": 240,
            "label_hours": 240 / 216_000,
            "source_label_frames": 1_200,
            "train_to_source_fraction": 240 / 1_200,
            "sessions": ["video_a__r000"],
            "bind_confidence": 0.75,
            "bind_flagged": False,
            "label_run_count": 1,
            "long_context_fraction": long_context_fraction,
            "decoder_mode": NATIVE_MODE,
            "source_average_fps": 60.0,
            "source_decoded_frames": 240,
            "nominal_timeline_frames": 240,
            "imputed_tail_frames": 0,
            "tail_truncated_frames": 960,
            "skipped_short_frames": 0,
        }],
    }
    (output_root / "full_corpus_manifest.json").write_text(json.dumps(manifest))

    completion_marker = tmp_path / ".done"
    completion_marker.touch()
    build_log = tmp_path / "build.log"
    build_log.write_text("[1/1] video_a ok\n")
    video_log_root = tmp_path / "video_logs"
    video_log_root.mkdir()
    (video_log_root / "video_a.log").write_text('{"video_id":"video_a"}\n')

    expectations = CorpusExpectations(
        valid_videos=1,
        rejected_videos=0,
        chunk_rows=1,
        source_label_frames=1_200,
        sessions=1,
        train_frames=240,
        native_videos=1,
        resampled_videos=0,
        native_sessions=1,
        resampled_sessions=0,
        native_frames=240,
        resampled_frames=0,
        tail_truncated_frames=960,
        skipped_short_frames=0,
    )
    return {
        "raw_root": raw_root,
        "chunk_index": chunk_index,
        "fetch_report": fetch_report,
        "mapped_root": mapped_root,
        "chunk_frames": chunk_frames,
        "feature_root": feature_root,
        "output_root": output_root,
        "completion_marker": completion_marker,
        "build_log": build_log,
        "video_log_root": video_log_root,
        "expectations": expectations,
        "metadata_reader": lambda _path: VideoMetadata(60.0, 240),
    }


def test_validator_accepts_complete_structural_fixture(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    report = validate_full_corpus_features(**fixture)

    assert report["ok"], report["errors"]
    assert report["observed"]["sessions"] == 1
    assert report["observed"]["train_frames"] == 240
    assert report["observed"]["tail_truncated_frames"] == 960
    assert report["observed"]["mapping_reports_v2"] == 1
    assert report["observed"]["direction_rule"] == EXPECTED_DIRECTION_RULE


def test_validator_rejects_legacy_mapping_report(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    mapping_path = fixture["mapped_root"] / "video_a" / "mapping_report.json"
    mapping = json.loads(mapping_path.read_text())
    mapping.pop("schema_version")
    mapping.pop("direction_rule")
    mapping_path.write_text(json.dumps(mapping))

    report = validate_full_corpus_features(**fixture)

    assert not report["ok"]
    assert any("mapping report schema" in error for error in report["errors"])
    assert any("fixed NitroGen contract" in error for error in report["errors"])


def test_validator_rejects_copy_instead_of_hard_link(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["output_root"] / "video_a__r000.npz"
    source = fixture["feature_root"] / "video_a" / "video_a__r000.npz"
    output.unlink()
    shutil.copyfile(source, output)

    report = validate_full_corpus_features(**fixture)

    assert not report["ok"]
    assert any("not source hard link" in error for error in report["errors"])


def test_deep_validator_detects_nonfinite_features(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    source = fixture["feature_root"] / "video_a" / "video_a__r000.npz"
    _write_shard(source, finite=False)

    report = validate_full_corpus_features(**fixture, deep_shards=True)

    assert not report["ok"]
    assert any("non-finite feature value" in error for error in report["errors"])
