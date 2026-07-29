from __future__ import annotations

import json
from pathlib import Path

import pytest

from harvest.merge_probe_vlm_shards import merge_prediction_shards


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def prediction(video_id: str, label: str = "non_target") -> dict:
    return {
        "video_id": video_id,
        "class": label,
        "model": "test/model",
        "resolved_model_revision": "revision-1",
        "prompt_version": "prompt-v1",
        "prompt_sha256": "abc123",
    }


def write_prediction_manifest(path: Path, **overrides) -> None:
    value = {
        "model": "test/model",
        "resolved_model_revision": "revision-1",
        "prompt_version": "prompt-v1",
        "prompt_sha256": "abc123",
        "classical_uncertain_score": 16.0,
        "classical_input_hud_uncertain": True,
        **overrides,
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps(value))


def test_merge_is_sorted_exact_and_reports_classes_and_hours(tmp_path) -> None:
    shard_b = tmp_path / "shard-b.jsonl"
    shard_a = tmp_path / "shard-a.jsonl"
    row_b = prediction("b", "uncertain")
    write_jsonl(shard_b, [row_b, prediction("c", "target_action_hud")])
    # An identical retry is harmless and is accounted for explicitly.
    write_jsonl(shard_a, [prediction("a"), row_b])
    write_prediction_manifest(shard_a)
    write_prediction_manifest(shard_b)
    scan = tmp_path / "scan.jsonl"
    write_jsonl(
        scan,
        [
            {"video_id": "a", "error": None, "duration_s": 3600},
            {"video_id": "b", "error": None, "duration_s": 7200},
            {"video_id": "c", "error": None, "duration_s": 1800},
            {"video_id": "failed", "error": "probe failed", "duration_s": 9999},
        ],
    )
    out = tmp_path / "merged.jsonl"

    manifest = merge_prediction_shards(
        [shard_b, shard_a], out, expected_scan=scan
    )

    merged = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["video_id"] for row in merged] == ["a", "b", "c"]
    assert manifest["rows"] == 3
    assert manifest["class_counts"] == {
        "non_target": 1,
        "target_action_hud": 1,
        "uncertain": 1,
    }
    assert manifest["nominal_hours"] == {
        "known_rows": 3,
        "missing_rows": 0,
        "total_known": 3.5,
        "by_class": {
            "non_target": 1.0,
            "target_action_hud": 0.5,
            "uncertain": 2.0,
        },
    }
    assert manifest["identical_duplicate_rows_deduplicated"] == 1
    assert manifest["identical_duplicate_video_ids"] == ["b"]
    assert manifest["uniform_metadata"]["classical_uncertain_score"] == 16.0
    assert manifest["expected"]["eligible_rows"] == 3
    assert manifest["expected"]["ineligible_rows"] == 1
    assert json.loads(
        out.with_suffix(".jsonl.manifest.json").read_text()
    ) == manifest


def test_merge_rejects_conflicting_duplicate_without_writing(tmp_path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    write_jsonl(left, [prediction("same", "non_target")])
    write_jsonl(right, [prediction("same", "target_action_hud")])
    out = tmp_path / "merged.jsonl"

    with pytest.raises(ValueError, match="conflicting duplicate prediction for same"):
        merge_prediction_shards([left, right], out)

    assert not out.exists()


def test_merge_requires_exact_expected_id_coverage(tmp_path) -> None:
    shard = tmp_path / "shard.jsonl"
    ids = tmp_path / "expected.ids"
    write_jsonl(shard, [prediction("a"), prediction("extra")])
    ids.write_text("a\nmissing\n")
    out = tmp_path / "merged.jsonl"

    with pytest.raises(
        ValueError,
        match=r"missing=1 \['missing'\], unexpected=1 \['extra'\]",
    ):
        merge_prediction_shards([shard], out, expected_ids_file=ids)

    assert not out.exists()


@pytest.mark.parametrize(
    ("row_override", "manifest_override", "message"),
    [
        ({"model": "other/model"}, {}, "non-uniform model"),
        ({}, {"classical_uncertain_score": 12.0}, "non-uniform classical_uncertain_score"),
    ],
)
def test_merge_rejects_nonuniform_model_or_calibration_metadata(
    tmp_path, row_override, manifest_override, message
) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    write_jsonl(left, [prediction("a")])
    write_jsonl(right, [{**prediction("b"), **row_override}])
    write_prediction_manifest(left)
    write_prediction_manifest(right, **manifest_override)

    with pytest.raises(ValueError, match=message):
        merge_prediction_shards([left, right], tmp_path / "merged.jsonl")
