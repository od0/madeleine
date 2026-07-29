import json
import math
from pathlib import Path

import pytest

from harvest.partition_layout_surveys import (
    balance_by_nominal_hours,
    build_wave,
    load_completed_ids,
    write_immutable_files,
)


def test_cli_exclude_ids_is_available() -> None:
    source = (
        Path(__file__).parents[1] / "harvest" / "partition_layout_surveys.py"
    ).read_text()
    assert '"--exclude-ids"' in source
    assert '"kind": "video_id_snapshot"' in source


def row(video_id: str, hours: float) -> dict:
    return {
        "video_id": video_id,
        "nominal_hours": hours,
        "human_reviewed": False,
        "training_admitted": False,
    }


def test_build_wave_intersects_completion_and_excludes_prior_queue():
    rows = build_wave(
        [row("b", 2.0), row("a", 1.0), row("c", 3.0)],
        {"a", "b", "not-nominated"},
        {"b"},
    )
    assert [item["video_id"] for item in rows] == ["a"]
    assert rows[0]["human_reviewed"] is False
    assert rows[0]["training_admitted"] is False
    assert "machine nominee" in rows[0]["survey_reason"]


def test_build_wave_rejects_admitted_nomination():
    candidate = row("a", 1.0)
    candidate["training_admitted"] = True
    with pytest.raises(ValueError, match="training-admitted"):
        build_wave([candidate], {"a"}, set())


def test_balance_is_deterministic_and_reasonably_even():
    rows = [row(chr(97 + index), float(index + 1)) for index in range(8)]
    shards = balance_by_nominal_hours(rows, 3)
    assert [[item["video_id"] for item in shard] for shard in shards] == [
        ["b", "c", "h"],
        ["a", "d", "g"],
        ["e", "f"],
    ]
    totals = [sum(item["nominal_hours"] for item in shard) for shard in shards]
    assert max(totals) - min(totals) <= 2.0


def test_load_completed_ids_sorts_and_rejects_duplicates(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("b\na\n")
    assert load_completed_ids(path) == ["a", "b"]
    path.write_text("a\na\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_completed_ids(path)


@pytest.mark.parametrize("hours", [-1.0, math.nan, math.inf, "1.0", None])
def test_build_wave_rejects_invalid_nominal_hours(hours):
    candidate = row("a", 1.0)
    candidate["nominal_hours"] = hours
    with pytest.raises(ValueError, match="nominal_hours"):
        build_wave([candidate], {"a"}, set())


def test_build_wave_rejects_unsafe_video_id():
    with pytest.raises(ValueError, match="unsafe"):
        build_wave([row("../escape", 1.0)], {"../escape"}, set())


def test_load_completed_ids_rejects_unsafe_id(tmp_path: Path):
    path = tmp_path / "ids.txt"
    path.write_text("../escape\n")
    with pytest.raises(ValueError, match="unsafe"):
        load_completed_ids(path)


def test_immutable_publication_is_idempotent_and_rejects_conflicts(
    tmp_path: Path,
):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.json"
    files = {first: b'{"video_id":"a"}\n', second: b'{"rows":1}\n'}
    write_immutable_files(files)
    write_immutable_files(files)
    assert first.read_bytes() == files[first]
    assert second.read_bytes() == files[second]

    second.write_bytes(b"conflict\n")
    third = tmp_path / "must-not-be-created.json"
    with pytest.raises(FileExistsError, match="immutable"):
        write_immutable_files({third: b"new\n", second: files[second]})
    assert not third.exists()
