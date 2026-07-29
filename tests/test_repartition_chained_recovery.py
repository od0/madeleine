from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from harvest.repartition_chained_recovery import (
    PredecessorSource,
    build_artifacts,
    build_plan,
    parse_predecessor,
    validate_artifact_bundle,
    validate_plan,
)
from harvest.repartition_fetch_queues import QueueSource, write_immutable_files


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


def sample_plan(tmp_path: Path):
    pred_b = source(
        tmp_path / "pred-b.jsonl", [row("b1", 5), row("bdone", 2)]
    )
    pred_c = source(tmp_path / "pred-c.jsonl", [row("c1", 1)])
    recovery = source(
        tmp_path / "recovery.jsonl",
        [row("r1", 4), row("rdone", 3), row("r2", 2)],
    )
    predecessors = [
        PredecessorSource("decode-b", pred_b),
        PredecessorSource("decode-c", pred_c),
    ]
    return build_plan(predecessors, [recovery], {"bdone", "rdone"}), predecessors, recovery


def test_success_pins_predecessors_and_balances_recovery(tmp_path: Path):
    plan, _, _ = sample_plan(tmp_path)

    assert plan.labels == ("decode-b", "decode-c")
    assert [ids(rows) for rows in plan.predecessor_missing] == [["b1"], ["c1"]]
    assert ids(plan.recovery_missing) == ["r1", "r2"]
    assert [ids(shard) for shard in plan.shards] == [
        ["b1", "r2"],
        ["c1", "r1"],
    ]
    assert plan.completed_input_ids == {"bdone", "rdone"}
    assert [sum(item["nominal_hours"] for item in shard) for shard in plan.shards] == [
        7,
        5,
    ]
    validate_plan(plan)


def test_artifacts_are_hash_bound_immutable_and_revalidate(tmp_path: Path):
    plan, predecessors, recovery = sample_plan(tmp_path)
    completed = tmp_path / "completed.txt"
    completed.write_text("rdone\nbdone\n")
    completed_ids = ["bdone", "rdone"]
    out = tmp_path / "out"

    artifacts, manifest, manifest_path = build_artifacts(
        name="recovery-01",
        out_dir=out,
        plan=plan,
        completed_path=completed,
        completed_ids=completed_ids,
    )
    write_immutable_files(artifacts)
    write_immutable_files(artifacts)
    assert validate_artifact_bundle(manifest_path) == manifest

    completed_entry = manifest["completed_snapshot_artifact"]
    completed_path = out / completed_entry["path"]
    assert completed_entry["sha256"] in completed_path.name
    assert hashlib.sha256(completed_path.read_bytes()).hexdigest() == completed_entry[
        "sha256"
    ]
    assert manifest["eligible_union"]["rows"] == 4
    assert manifest["eligible_union"]["nominal_hours"] == 12
    assert manifest["predecessors"][0]["sha256"] == hashlib.sha256(
        predecessors[0].queue.path.read_bytes()
    ).hexdigest()
    assert manifest["recovery_inputs"][0]["sha256"] == hashlib.sha256(
        recovery.path.read_bytes()
    ).hexdigest()
    assert all(manifest["partition_invariants"].values())

    conflicting = out / "recovery-01.shard-decode-b.jsonl"
    conflicting.write_text("conflict\n")
    never = out / "must-not-exist"
    with pytest.raises(FileExistsError, match="immutable"):
        write_immutable_files({never: b"no\n", **artifacts})
    assert not never.exists()


def test_rejects_cross_queue_overlap(tmp_path: Path):
    pred = source(tmp_path / "pred.jsonl", [row("same", 1)])
    recovery = source(tmp_path / "recovery.jsonl", [row("same", 2)])
    with pytest.raises(ValueError, match="input queues overlap on same"):
        build_plan([PredecessorSource("worker", pred)], [recovery], set())


def test_rejects_duplicate_inside_any_queue(tmp_path: Path):
    pred = source(
        tmp_path / "pred.jsonl", [row("duplicate", 1), row("duplicate", 1)]
    )
    with pytest.raises(ValueError, match="duplicate video_id duplicate"):
        build_plan([PredecessorSource("worker", pred)], [], set())


def test_corrupted_plan_rejects_missing_output(tmp_path: Path):
    plan, _, _ = sample_plan(tmp_path)
    corrupted = replace(
        plan, shards=(plan.shards[0][:-1], plan.shards[1])
    )
    with pytest.raises(ValueError, match="output union mismatch.*missing=.*r2"):
        validate_plan(corrupted)


def test_corrupted_plan_rejects_extra_output(tmp_path: Path):
    plan, _, _ = sample_plan(tmp_path)
    extra = row("extra", 1)
    corrupted = replace(
        plan, shards=(plan.shards[0] + (extra,), plan.shards[1])
    )
    with pytest.raises(ValueError, match="output union mismatch.*extra=.*extra"):
        validate_plan(corrupted)


def test_corrupted_plan_rejects_duplicate_output(tmp_path: Path):
    plan, _, _ = sample_plan(tmp_path)
    corrupted = replace(
        plan, shards=(plan.shards[0], plan.shards[1] + (plan.shards[0][0],))
    )
    with pytest.raises(ValueError, match="output shards overlap.*b1"):
        validate_plan(corrupted)


def test_corrupted_plan_rejects_completed_row(tmp_path: Path):
    plan, _, _ = sample_plan(tmp_path)
    completed_row = row("bdone", 2)
    corrupted = replace(
        plan,
        eligible_rows=plan.eligible_rows + (completed_row,),
        shards=(plan.shards[0] + (completed_row,), plan.shards[1]),
    )
    with pytest.raises(ValueError, match="eligible union differs"):
        validate_plan(corrupted)


def test_corrupted_artifact_fails_hash_validation(tmp_path: Path):
    plan, _, _ = sample_plan(tmp_path)
    completed = tmp_path / "completed.txt"
    completed.write_text("bdone\nrdone\n")
    artifacts, _, manifest_path = build_artifacts(
        name="recovery-02",
        out_dir=tmp_path / "out",
        plan=plan,
        completed_path=completed,
        completed_ids=["bdone", "rdone"],
    )
    write_immutable_files(artifacts)
    shard = tmp_path / "out" / "recovery-02.shard-decode-c.jsonl"
    shard.write_text(shard.read_text().replace('"c1"', '"xx"', 1))
    with pytest.raises(ValueError, match="shard decode-c hash mismatch"):
        validate_artifact_bundle(manifest_path)


def test_rejects_completed_snapshot_mismatch_at_artifact_boundary(tmp_path: Path):
    plan, _, _ = sample_plan(tmp_path)
    completed = tmp_path / "completed.txt"
    completed.write_text("bdone\nrdone\n")
    with pytest.raises(ValueError, match="does not match partition barrier"):
        build_artifacts(
            name="bad",
            out_dir=tmp_path / "out",
            plan=plan,
            completed_path=completed,
            completed_ids=["bdone"],
        )


@pytest.mark.parametrize(
    "value", ["missing-equals", "=queue.jsonl", "../host=queue.jsonl"]
)
def test_parse_predecessor_rejects_malformed_or_unsafe_values(value: str):
    with pytest.raises(ValueError):
        parse_predecessor(value)
