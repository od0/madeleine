import hashlib
import json
import math
from pathlib import Path

import pytest

from harvest.repartition_fetch_queues import (
    QueueSource,
    build_artifacts,
    build_partition,
    load_id_snapshot,
    validate_queue_rows,
    write_immutable_files,
)


def row(video_id: str, hours: float) -> dict:
    return {
        "video_id": video_id,
        "nominal_hours": hours,
        "machine_nomination_only": True,
        "human_reviewed": False,
        "training_admitted": False,
    }


def source(path: Path, rows: list[dict]) -> QueueSource:
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    return QueueSource(path=path, rows=tuple(rows))


def ids(rows) -> list[str]:
    return [item["video_id"] for item in rows]


def test_partition_filters_barrier_and_exclusions_and_appends_retries(tmp_path):
    first = source(
        tmp_path / "first.jsonl",
        [row("a", 5), row("b", 4), row("c", 3), row("d", 2)],
    )
    second = source(tmp_path / "second.jsonl", [row("e", 1), row("f", 1)])
    excluded = source(tmp_path / "mac.jsonl", [row("d", 2), row("reserved", 8)])

    plan = build_partition(
        [first, second],
        {"a", "completed_elsewhere"},
        [excluded],
        ["h200", "fetch-10", "fetch-11"],
        ["e", "c"],
    )

    assert ids(plan.eligible_rows) == ["b", "c", "e", "f"]
    assert plan.completed_input_ids == {"a"}
    assert plan.excluded_input_ids == {"d"}
    assert plan.retry_ids == ("c", "e")
    flattened = [video_id for shard in plan.shards for video_id in ids(shard)]
    assert sorted(flattened) == ["b", "c", "e", "f"]
    assert len(flattened) == len(set(flattened))
    assert all(
        not any(
            item["video_id"] not in plan.retry_ids
            for item in shard[first_retry + 1 :]
        )
        for shard in plan.shards
        for first_retry in (
            next(
                (
                    index
                    for index, item in enumerate(shard)
                    if item["video_id"] in plan.retry_ids
                ),
                len(shard),
            ),
        )
    )
    retry_shards = {
        index
        for index, shard in enumerate(plan.shards)
        if any(item["video_id"] in plan.retry_ids for item in shard)
    }
    assert len(retry_shards) == 2


def test_partition_is_deterministic_with_ordered_shard_tie_break(tmp_path):
    queue = source(
        tmp_path / "queue.jsonl",
        [row("a", 4), row("b", 3), row("c", 2), row("d", 1)],
    )
    first = build_partition([queue], set(), [], ["z", "a"])
    second = build_partition([queue], set(), [], ["z", "a"])
    assert first == second
    assert [ids(shard) for shard in first.shards] == [["a", "d"], ["b", "c"]]


def test_source_filter_precedes_completion_and_exclusion(tmp_path):
    rows = [
        {**row("yt-done", 1), "source": "youtube"},
        {**row("yt-ready", 2), "source": "youtube"},
        {**row("tw-reserved", 3), "source": "twitch"},
        {**row("tw-other", 4), "source": "twitch"},
    ]
    queue = source(tmp_path / "mixed.jsonl", rows)
    reserve = source(
        tmp_path / "reserve.jsonl",
        [{**row("tw-reserved", 3), "source": "twitch"}],
    )
    plan = build_partition(
        [queue],
        {"yt-done", "tw-other"},
        [reserve],
        ["one", "two"],
        selected_sources=["youtube"],
    )

    assert plan.selected_sources == ("youtube",)
    assert ids(plan.selected_source_rows) == ["yt-done", "yt-ready"]
    assert ids(plan.filtered_out_rows) == ["tw-reserved", "tw-other"]
    assert ids(plan.eligible_rows) == ["yt-ready"]
    assert plan.completed_input_ids == {"yt-done"}
    assert plan.excluded_input_ids == set()


@pytest.mark.parametrize("value", [None, 1, "../twitch"])
def test_source_filter_requires_safe_source_on_every_input_row(tmp_path, value):
    candidate = row("a", 1)
    if value is not None:
        candidate["source"] = value
    queue = source(tmp_path / "queue.jsonl", [candidate])
    with pytest.raises(ValueError, match="missing or unsafe source"):
        build_partition([queue], set(), [], ["worker"], selected_sources=["youtube"])

    # Omitting the filter preserves the prior schema and behavior.
    assert ids(build_partition([queue], set(), [], ["worker"]).eligible_rows) == [
        "a"
    ]


def test_source_filter_rejects_unsafe_or_duplicate_selection(tmp_path):
    queue = source(
        tmp_path / "queue.jsonl", [{**row("a", 1), "source": "youtube"}]
    )
    with pytest.raises(ValueError, match="unsafe source filter"):
        build_partition([queue], set(), [], ["worker"], selected_sources=["../x"])
    with pytest.raises(ValueError, match="must be unique"):
        build_partition(
            [queue],
            set(),
            [],
            ["worker"],
            selected_sources=["youtube", "youtube"],
        )


def test_partition_rejects_overlapping_input_and_excluded_queues(tmp_path):
    first = source(tmp_path / "first.jsonl", [row("a", 1)])
    duplicate = source(tmp_path / "duplicate.jsonl", [row("a", 2)])
    with pytest.raises(ValueError, match="input queues overlap"):
        build_partition([first, duplicate], set(), [], ["worker"])

    reserve_one = source(tmp_path / "reserve-one.jsonl", [row("r", 1)])
    reserve_two = source(tmp_path / "reserve-two.jsonl", [row("r", 1)])
    with pytest.raises(ValueError, match="excluded queues overlap"):
        build_partition([first], set(), [reserve_one, reserve_two], ["worker"])


@pytest.mark.parametrize("hours", [-1, math.nan, math.inf, "1", True, None])
def test_queue_validation_rejects_invalid_hours(hours):
    with pytest.raises(ValueError, match="nominal_hours"):
        validate_queue_rows([row("a", hours)], "queue")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("human_reviewed", True, "human_reviewed=false"),
        ("human_reviewed", None, "human_reviewed=false"),
        ("training_admitted", True, "training_admitted=false"),
        ("training_admitted", 0, "training_admitted=false"),
        ("training_admitted", "false", "training_admitted=false"),
    ],
)
def test_queue_validation_requires_explicit_false_flags(field, value, message):
    candidate = row("a", 1)
    candidate[field] = value
    with pytest.raises(ValueError, match=message):
        validate_queue_rows([candidate], "queue")


def test_null_or_missing_admission_is_accepted_and_output_is_normalized(tmp_path):
    null_row = row("null", 1)
    null_row["training_admitted"] = None
    missing_row = row("missing", 2)
    del missing_row["training_admitted"]
    queue = source(tmp_path / "queue.jsonl", [null_row, missing_row])

    plan = build_partition([queue], set(), [], ["worker"])

    assert plan.source_rows[0]["training_admitted"] is None
    assert "training_admitted" not in plan.source_rows[1]
    assert all(row["training_admitted"] is False for row in plan.eligible_rows)
    assert all(row["human_reviewed"] is False for row in plan.eligible_rows)
    assert all(row["machine_nomination_only"] is True for row in plan.eligible_rows)


def test_admitted_true_is_rejected():
    candidate = row("a", 1)
    candidate["training_admitted"] = True
    with pytest.raises(ValueError, match="training_admitted=false"):
        validate_queue_rows([candidate], "queue")


def test_non_machine_nomination_is_rejected():
    candidate = row("a", 1)
    candidate["machine_nomination_only"] = False
    with pytest.raises(ValueError, match="machine_nomination_only=true"):
        validate_queue_rows([candidate], "queue")


def test_partition_rejects_unsafe_duplicate_or_unplaceable_retries(tmp_path):
    queue = source(tmp_path / "queue.jsonl", [row("a", 1), row("b", 1)])
    with pytest.raises(ValueError, match="not eligible"):
        build_partition([queue], {"a"}, [], ["worker"], ["a"])
    with pytest.raises(ValueError, match="duplicates"):
        build_partition([queue], set(), [], ["one", "two"], ["a", "a"])
    with pytest.raises(ValueError, match="distinct shard count"):
        build_partition([queue], set(), [], ["one"], ["a", "b"])
    with pytest.raises(ValueError, match="unsafe shard label"):
        build_partition([queue], set(), [], ["../worker"])

    unsafe = row("../escape", 1)
    with pytest.raises(ValueError, match="unsafe video_id"):
        validate_queue_rows([unsafe], "queue")


def test_id_snapshot_sorts_and_rejects_duplicate_or_unsafe_ids(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("b/upload_complete.json\na\n")
    assert load_id_snapshot(path) == ["a", "b"]
    path.write_text("a\na/upload_complete.json\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_id_snapshot(path)
    path.write_text("../escape\n")
    with pytest.raises(ValueError, match="unsafe"):
        load_id_snapshot(path)
    path.write_text("a/nested/upload_complete.json\n")
    with pytest.raises(ValueError, match="unsafe completion path"):
        load_id_snapshot(path)


def test_artifacts_are_hashed_idempotent_and_immutable(tmp_path):
    queue = source(tmp_path / "queue.jsonl", [row("a", 2), row("b", 1)])
    completed_path = tmp_path / "completed.txt"
    completed_path.write_text("done\n")
    plan = build_partition([queue], {"done"}, [], ["one", "two"], ["b"])
    out = tmp_path / "out"
    artifacts, manifest = build_artifacts(
        name="launch-01",
        out_dir=out,
        plan=plan,
        queue_sources=[queue],
        completed_path=completed_path,
        completed_ids=["done"],
        excluded_sources=[],
    )
    write_immutable_files(artifacts)
    write_immutable_files(artifacts)

    for shard in manifest["shards"]:
        content = (out / shard["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == shard["sha256"]
    assert manifest["partition_invariants"]["output_union_exactly_eligible"]
    assert manifest["selected_sources"] == []
    assert manifest["source_filter"] == {
        "enabled": False,
        "selected_sources": [],
        "selected_rows": 2,
        "selected_nominal_hours": 3.0,
        "filtered_out_rows": 0,
        "filtered_out_nominal_hours": 0,
    }
    assert (out / "launch-01.manifest.json").exists()

    conflict = out / "launch-01.shard-one.jsonl"
    conflict.write_text("conflict\n")
    new_path = out / "must-not-exist.json"
    with pytest.raises(FileExistsError, match="immutable"):
        write_immutable_files({new_path: b"new\n", **artifacts})
    assert not new_path.exists()


def test_manifest_records_source_filter_counts_and_invariants(tmp_path):
    queue = source(
        tmp_path / "queue.jsonl",
        [
            {**row("yt", 2), "source": "youtube"},
            {**row("tw-one", 3), "source": "twitch"},
            {**row("tw-two", 5), "source": "twitch"},
        ],
    )
    completed_path = tmp_path / "completed.txt"
    completed_path.write_text("")
    plan = build_partition(
        [queue],
        set(),
        [],
        ["worker"],
        selected_sources=["twitch"],
    )
    _, manifest = build_artifacts(
        name="twitch-only",
        out_dir=tmp_path / "out",
        plan=plan,
        queue_sources=[queue],
        completed_path=completed_path,
        completed_ids=[],
        excluded_sources=[],
    )
    assert manifest["selected_sources"] == ["twitch"]
    assert manifest["source_filter"] == {
        "enabled": True,
        "selected_sources": ["twitch"],
        "selected_rows": 2,
        "selected_nominal_hours": 8.0,
        "filtered_out_rows": 1,
        "filtered_out_nominal_hours": 2.0,
    }
    invariants = manifest["partition_invariants"]
    assert invariants["source_filtered_ids_absent"]
    assert invariants["source_filter_applied_before_completion_and_exclusion"]
