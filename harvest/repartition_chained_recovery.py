"""Compile fail-closed successor queues for chained fetch recovery.

Each ``--predecessor LABEL=QUEUE`` binds the rows from that predecessor that
are still missing at the completion-marker barrier to LABEL's successor.  Rows
from the disjoint ``--recovery-queue`` inputs are then balanced across those
same labels, accounting for the pinned work already assigned to each label.

This is scheduling metadata only.  It never grants human review or training
admission.  Every emitted artifact is hash-bound and published immutably.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from harvest.repartition_fetch_queues import (
    QueueSource,
    jsonl_text,
    load_id_snapshot,
    load_jsonl,
    load_queue,
    nominal_hours,
    sha256_bytes,
    sha256_file,
    validate_queue_rows,
    write_immutable_files,
)


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class PredecessorSource:
    label: str
    queue: QueueSource


@dataclass(frozen=True)
class ChainedRecoveryPlan:
    predecessors: tuple[PredecessorSource, ...]
    recovery_sources: tuple[QueueSource, ...]
    completed_ids: frozenset[str]
    completed_input_ids: frozenset[str]
    eligible_rows: tuple[dict[str, Any], ...]
    predecessor_missing: tuple[tuple[dict[str, Any], ...], ...]
    recovery_missing: tuple[dict[str, Any], ...]
    shards: tuple[tuple[dict[str, Any], ...], ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(source.label for source in self.predecessors)


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["human_reviewed"] = False
    normalized["training_admitted"] = False
    normalized["machine_nomination_only"] = True
    return normalized


def _ids(rows: Iterable[dict[str, Any]]) -> list[str]:
    return [str(row["video_id"]) for row in rows]


def _hours(rows: Iterable[dict[str, Any]]) -> float:
    return sum(nominal_hours(row, str(row["video_id"])) for row in rows)


def _id_digest(ids: Iterable[str]) -> str:
    return sha256_bytes("".join(f"{value}\n" for value in sorted(ids)).encode())


def _validate_label(label: str) -> str:
    if not _SAFE_COMPONENT.fullmatch(label):
        raise ValueError(f"unsafe predecessor label: {label!r}")
    return label


def parse_predecessor(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if separator != "=" or not raw_path:
        raise ValueError("predecessor must have the form LABEL=QUEUE")
    return _validate_label(label), Path(raw_path)


def _validate_disjoint_inputs(
    predecessors: Sequence[PredecessorSource],
    recovery_sources: Sequence[QueueSource],
) -> dict[str, tuple[str, dict[str, Any]]]:
    if not predecessors:
        raise ValueError("at least one predecessor is required")
    labels = [source.label for source in predecessors]
    if len(labels) != len(set(labels)):
        raise ValueError("predecessor labels must be unique")
    owners: dict[str, tuple[str, dict[str, Any]]] = {}
    named_sources = [
        (f"predecessor {source.label} ({source.queue.path})", source.queue)
        for source in predecessors
    ] + [
        (f"recovery queue {source.path}", source) for source in recovery_sources
    ]
    for label, source in named_sources:
        validate_queue_rows(source.rows, label)
        for row in source.rows:
            video_id = str(row["video_id"])
            prior = owners.get(video_id)
            if prior is not None:
                raise ValueError(
                    f"input queues overlap on {video_id}: {prior[0]} and {label}"
                )
            owners[video_id] = (label, row)
    return owners


def build_plan(
    predecessors: Sequence[PredecessorSource],
    recovery_sources: Sequence[QueueSource],
    completed_ids: set[str],
) -> ChainedRecoveryPlan:
    """Build a deterministic partition at one frozen completion barrier."""

    for source in predecessors:
        _validate_label(source.label)
    for video_id in completed_ids:
        if not _SAFE_ID.fullmatch(video_id):
            raise ValueError(f"completed IDs contain unsafe video_id {video_id!r}")
    owners = _validate_disjoint_inputs(predecessors, recovery_sources)

    pinned: list[tuple[dict[str, Any], ...]] = []
    for source in predecessors:
        pinned.append(
            tuple(
                _normalize(row)
                for row in source.queue.rows
                if row["video_id"] not in completed_ids
            )
        )
    recovery = tuple(
        _normalize(row)
        for source in recovery_sources
        for row in source.rows
        if row["video_id"] not in completed_ids
    )
    eligible = tuple(row for rows in pinned for row in rows) + recovery

    recovery_by_label: list[list[dict[str, Any]]] = [
        [] for _ in predecessors
    ]
    shard_hours = [_hours(rows) for rows in pinned]
    for row in sorted(
        recovery,
        key=lambda item: (
            -nominal_hours(item, str(item["video_id"])),
            str(item["video_id"]),
        ),
    ):
        index = min(
            range(len(predecessors)), key=lambda item: (shard_hours[item], item)
        )
        recovery_by_label[index].append(row)
        shard_hours[index] += nominal_hours(row, str(row["video_id"]))

    recovery_position = {
        str(row["video_id"]): index for index, row in enumerate(recovery)
    }
    shards: list[tuple[dict[str, Any], ...]] = []
    for index, pinned_rows in enumerate(pinned):
        assigned = sorted(
            recovery_by_label[index],
            key=lambda row: recovery_position[str(row["video_id"])],
        )
        shards.append(pinned_rows + tuple(assigned))

    plan = ChainedRecoveryPlan(
        predecessors=tuple(predecessors),
        recovery_sources=tuple(recovery_sources),
        completed_ids=frozenset(completed_ids),
        completed_input_ids=frozenset(set(owners).intersection(completed_ids)),
        eligible_rows=eligible,
        predecessor_missing=tuple(pinned),
        recovery_missing=recovery,
        shards=tuple(shards),
    )
    validate_plan(plan)
    return plan


def _rows_by_id(
    rows: Sequence[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    try:
        validate_queue_rows(rows, label)
    except ValueError as exc:
        raise ValueError(f"invalid chained recovery plan: {exc}") from exc
    return {str(row["video_id"]): row for row in rows}


def validate_plan(plan: ChainedRecoveryPlan) -> None:
    """Recompute every assignment invariant and reject a corrupted plan."""

    owners = _validate_disjoint_inputs(plan.predecessors, plan.recovery_sources)
    labels = plan.labels
    if len(plan.predecessor_missing) != len(labels):
        raise ValueError("invalid chained recovery plan: predecessor shard count")
    if len(plan.shards) != len(labels):
        raise ValueError("invalid chained recovery plan: output shard count")

    expected_completed = frozenset(set(owners).intersection(plan.completed_ids))
    if plan.completed_input_ids != expected_completed:
        raise ValueError("invalid chained recovery plan: completed input set mismatch")

    expected_pinned = tuple(
        tuple(
            _normalize(row)
            for row in source.queue.rows
            if row["video_id"] not in plan.completed_ids
        )
        for source in plan.predecessors
    )
    if plan.predecessor_missing != expected_pinned:
        raise ValueError("invalid chained recovery plan: bound predecessor rows differ")
    expected_recovery = tuple(
        _normalize(row)
        for source in plan.recovery_sources
        for row in source.rows
        if row["video_id"] not in plan.completed_ids
    )
    if plan.recovery_missing != expected_recovery:
        raise ValueError("invalid chained recovery plan: recovery rows differ")
    expected_eligible = tuple(row for rows in expected_pinned for row in rows)
    expected_eligible += expected_recovery
    if plan.eligible_rows != expected_eligible:
        raise ValueError("invalid chained recovery plan: eligible union differs")

    eligible = _rows_by_id(plan.eligible_rows, "eligible union")
    completed_overlap = set(eligible).intersection(plan.completed_ids)
    if completed_overlap:
        raise ValueError(
            "invalid chained recovery plan: completed IDs remain eligible: "
            f"{sorted(completed_overlap)}"
        )

    output: dict[str, dict[str, Any]] = {}
    for index, (label, shard) in enumerate(zip(labels, plan.shards, strict=True)):
        indexed = _rows_by_id(shard, f"output shard {label}")
        overlap = set(output).intersection(indexed)
        if overlap:
            raise ValueError(
                "invalid chained recovery plan: output shards overlap: "
                f"{sorted(overlap)}"
            )
        output.update(indexed)
        pinned_ids = set(_ids(plan.predecessor_missing[index]))
        shard_ids = set(indexed)
        if not pinned_ids.issubset(shard_ids):
            raise ValueError(
                f"invalid chained recovery plan: predecessor {label} is not pinned"
            )
        for other_index, other_rows in enumerate(plan.predecessor_missing):
            if other_index == index:
                continue
            crossed = shard_ids.intersection(_ids(other_rows))
            if crossed:
                raise ValueError(
                    "invalid chained recovery plan: predecessor rows crossed labels: "
                    f"{sorted(crossed)}"
                )

    eligible_ids = set(eligible)
    output_ids = set(output)
    if output_ids != eligible_ids:
        raise ValueError(
            "invalid chained recovery plan: output union mismatch: "
            f"missing={sorted(eligible_ids - output_ids)} "
            f"extra={sorted(output_ids - eligible_ids)}"
        )
    for video_id, row in output.items():
        if row != eligible[video_id]:
            raise ValueError(
                f"invalid chained recovery plan: output row differs for {video_id}"
            )


def _source_manifest(source: QueueSource) -> dict[str, Any]:
    return {
        "path": str(source.path),
        "rows": len(source.rows),
        "nominal_hours": _hours(source.rows),
        "sha256": sha256_file(source.path),
    }


def build_artifacts(
    *,
    name: str,
    out_dir: Path,
    plan: ChainedRecoveryPlan,
    completed_path: Path,
    completed_ids: Sequence[str],
) -> tuple[dict[Path, bytes], dict[str, Any], Path]:
    if not _SAFE_COMPONENT.fullmatch(name):
        raise ValueError(f"unsafe output name: {name!r}")
    validate_plan(plan)
    if list(completed_ids) != sorted(completed_ids):
        raise ValueError("completed IDs supplied to artifacts must be sorted")
    if frozenset(completed_ids) != plan.completed_ids:
        raise ValueError("completed artifact does not match partition barrier")

    completed_bytes = "".join(f"{value}\n" for value in completed_ids).encode()
    completed_sha = sha256_bytes(completed_bytes)
    completed_artifact = out_dir / f"{name}.completed-ids.{completed_sha}.txt"
    eligible_bytes = jsonl_text(plan.eligible_rows).encode()
    eligible_artifact = out_dir / f"{name}.eligible.jsonl"
    artifacts: dict[Path, bytes] = {
        completed_artifact: completed_bytes,
        eligible_artifact: eligible_bytes,
    }

    shard_entries: list[dict[str, Any]] = []
    for index, (label, shard) in enumerate(
        zip(plan.labels, plan.shards, strict=True)
    ):
        content = jsonl_text(shard).encode()
        path = out_dir / f"{name}.shard-{label}.jsonl"
        artifacts[path] = content
        pinned_ids = _ids(plan.predecessor_missing[index])
        recovery_ids = _ids(shard[len(pinned_ids) :])
        shard_entries.append(
            {
                "label": label,
                "path": path.name,
                "rows": len(shard),
                "nominal_hours": _hours(shard),
                "sha256": sha256_bytes(content),
                "pinned_predecessor": {
                    "rows": len(pinned_ids),
                    "nominal_hours": _hours(plan.predecessor_missing[index]),
                    "video_ids": pinned_ids,
                    "video_ids_sha256": _id_digest(pinned_ids),
                },
                "balanced_recovery": {
                    "rows": len(recovery_ids),
                    "nominal_hours": _hours(shard[len(pinned_ids) :]),
                    "video_ids_sha256": _id_digest(recovery_ids),
                },
            }
        )

    predecessors_manifest = []
    for index, source in enumerate(plan.predecessors):
        missing = plan.predecessor_missing[index]
        source_entry = _source_manifest(source.queue)
        source_entry.update(
            {
                "label": source.label,
                "missing_rows": len(missing),
                "missing_nominal_hours": _hours(missing),
                "missing_video_ids_sha256": _id_digest(_ids(missing)),
            }
        )
        predecessors_manifest.append(source_entry)

    all_source_rows = [
        row for source in plan.predecessors for row in source.queue.rows
    ] + [row for source in plan.recovery_sources for row in source.rows]
    manifest: dict[str, Any] = {
        "format_version": "madeleine.chained-fetch-repartition.v1",
        "name": name,
        "semantics": "machine-only fetch recovery; not review or admission",
        "human_reviewed": False,
        "training_admitted": False,
        "completed_snapshot_input": {
            "path": str(completed_path),
            "ids": len(completed_ids),
            "sha256": sha256_file(completed_path),
        },
        "completed_snapshot_artifact": {
            "path": completed_artifact.name,
            "ids": len(completed_ids),
            "sha256": completed_sha,
            "content_addressed": True,
        },
        "predecessors": predecessors_manifest,
        "recovery_inputs": [
            _source_manifest(source) for source in plan.recovery_sources
        ],
        "source_union": {
            "rows": len(all_source_rows),
            "nominal_hours": _hours(all_source_rows),
            "unique_ids": len(all_source_rows),
        },
        "completed_source_rows_removed": {
            "rows": len(plan.completed_input_ids),
            "nominal_hours": _hours(
                row
                for row in all_source_rows
                if row["video_id"] in plan.completed_input_ids
            ),
        },
        "eligible_union": {
            "path": eligible_artifact.name,
            "rows": len(plan.eligible_rows),
            "nominal_hours": _hours(plan.eligible_rows),
            "sha256": sha256_bytes(eligible_bytes),
            "video_ids_sha256": _id_digest(_ids(plan.eligible_rows)),
        },
        "ordered_successor_labels": list(plan.labels),
        "partition_method": (
            "missing predecessor rows pinned to their matching successor in "
            "original queue order; disjoint recovery rows assigned by largest "
            "nominal-hours first to the lightest total successor, label-order "
            "tie-break, then restored to recovery input order"
        ),
        "partition_invariants": {
            "all_input_queues_pairwise_disjoint": True,
            "completed_ids_absent": True,
            "output_shards_disjoint": True,
            "output_union_exactly_source_minus_completed": True,
            "each_missing_predecessor_row_bound_exactly_once": True,
            "no_predecessor_cross_label_assignment": True,
        },
        "shards": shard_entries,
    }
    manifest_path = out_dir / f"{name}.manifest.json"
    artifacts[manifest_path] = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    return artifacts, manifest, manifest_path


def _bundle_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or Path(value).name != value:
        raise ValueError(f"invalid chained recovery artifacts: unsafe {label} path")
    path = root / value
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"invalid chained recovery artifacts: missing {label}")
    return path


def _check_artifact(
    root: Path, entry: dict[str, Any], label: str, *, jsonl: bool
) -> tuple[Path, list[dict[str, Any]] | list[str]]:
    path = _bundle_path(root, entry.get("path"), label)
    if sha256_file(path) != entry.get("sha256"):
        raise ValueError(f"invalid chained recovery artifacts: {label} hash mismatch")
    if jsonl:
        rows = load_jsonl(path)
        validate_queue_rows(rows, label)
        if len(rows) != entry.get("rows") or _hours(rows) != entry.get(
            "nominal_hours"
        ):
            raise ValueError(
                f"invalid chained recovery artifacts: {label} accounting mismatch"
            )
        return path, rows
    ids = load_id_snapshot(path, label)
    if len(ids) != entry.get("ids"):
        raise ValueError(f"invalid chained recovery artifacts: {label} count mismatch")
    return path, ids


def validate_artifact_bundle(manifest_path: Path) -> dict[str, Any]:
    """Validate hashes, accounting, pinning, and exact union from disk."""

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid chained recovery manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != (
        "madeleine.chained-fetch-repartition.v1"
    ):
        raise ValueError("invalid chained recovery manifest format")
    root = manifest_path.parent
    _, completed_values = _check_artifact(
        root,
        manifest.get("completed_snapshot_artifact", {}),
        "completed snapshot",
        jsonl=False,
    )
    completed = set(completed_values)
    _, eligible_values = _check_artifact(
        root, manifest.get("eligible_union", {}), "eligible union", jsonl=True
    )
    eligible_rows = list(eligible_values)
    eligible = {str(row["video_id"]): row for row in eligible_rows}
    if set(eligible).intersection(completed):
        raise ValueError("invalid chained recovery artifacts: completed ID eligible")
    eligible_entry = manifest["eligible_union"]
    if _id_digest(eligible) != eligible_entry.get("video_ids_sha256"):
        raise ValueError("invalid chained recovery artifacts: eligible ID hash mismatch")

    predecessor_entries = manifest.get("predecessors")
    shard_entries = manifest.get("shards")
    labels = manifest.get("ordered_successor_labels")
    if (
        not isinstance(predecessor_entries, list)
        or not isinstance(shard_entries, list)
        or not isinstance(labels, list)
        or len(predecessor_entries) != len(labels)
        or len(shard_entries) != len(labels)
        or len(labels) != len(set(labels))
    ):
        raise ValueError("invalid chained recovery artifacts: label structure")

    output: dict[str, dict[str, Any]] = {}
    all_pinned: set[str] = set()
    for label, predecessor, shard_entry in zip(
        labels, predecessor_entries, shard_entries, strict=True
    ):
        if predecessor.get("label") != label or shard_entry.get("label") != label:
            raise ValueError("invalid chained recovery artifacts: label binding")
        _, shard_values = _check_artifact(
            root, shard_entry, f"shard {label}", jsonl=True
        )
        shard_rows = list(shard_values)
        shard = {str(row["video_id"]): row for row in shard_rows}
        overlap = set(output).intersection(shard)
        if overlap:
            raise ValueError(
                f"invalid chained recovery artifacts: shard overlap {sorted(overlap)}"
            )
        output.update(shard)
        pinned_entry = shard_entry.get("pinned_predecessor", {})
        pinned_ids = pinned_entry.get("video_ids")
        if (
            not isinstance(pinned_ids, list)
            or len(pinned_ids) != len(set(pinned_ids))
            or any(not isinstance(item, str) or not _SAFE_ID.fullmatch(item) for item in pinned_ids)
            or pinned_entry.get("rows") != len(pinned_ids)
            or pinned_entry.get("video_ids_sha256") != _id_digest(pinned_ids)
            or predecessor.get("missing_rows") != len(pinned_ids)
            or predecessor.get("missing_video_ids_sha256") != _id_digest(pinned_ids)
        ):
            raise ValueError(
                f"invalid chained recovery artifacts: predecessor {label} pin metadata"
            )
        if shard_rows[: len(pinned_ids)] != [eligible.get(item) for item in pinned_ids]:
            raise ValueError(
                f"invalid chained recovery artifacts: predecessor {label} not pinned"
            )
        crossed = all_pinned.intersection(pinned_ids)
        if crossed:
            raise ValueError(
                f"invalid chained recovery artifacts: pinned overlap {sorted(crossed)}"
            )
        all_pinned.update(pinned_ids)

    if set(output) != set(eligible):
        raise ValueError(
            "invalid chained recovery artifacts: output union mismatch: "
            f"missing={sorted(set(eligible) - set(output))} "
            f"extra={sorted(set(output) - set(eligible))}"
        )
    if any(output[video_id] != row for video_id, row in eligible.items()):
        raise ValueError("invalid chained recovery artifacts: output row mismatch")
    if set(output).intersection(completed):
        raise ValueError("invalid chained recovery artifacts: completed ID in output")
    invariants = manifest.get("partition_invariants")
    if not isinstance(invariants, dict) or not invariants or not all(
        value is True for value in invariants.values()
    ):
        raise ValueError("invalid chained recovery artifacts: invariant declaration")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predecessor",
        action="append",
        required=True,
        metavar="LABEL=QUEUE",
    )
    parser.add_argument("--recovery-queue", type=Path, action="append", default=[])
    parser.add_argument("--completed-ids", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    predecessors = []
    for value in args.predecessor:
        try:
            label, path = parse_predecessor(value)
        except ValueError as exc:
            parser.error(str(exc))
        predecessors.append(PredecessorSource(label, load_queue(path)))
    recovery_sources = [load_queue(path) for path in args.recovery_queue]
    completed_ids = load_id_snapshot(args.completed_ids, "completed snapshot")
    plan = build_plan(predecessors, recovery_sources, set(completed_ids))
    artifacts, manifest, manifest_path = build_artifacts(
        name=args.name,
        out_dir=args.out_dir,
        plan=plan,
        completed_path=args.completed_ids,
        completed_ids=completed_ids,
    )
    write_immutable_files(artifacts)
    validate_artifact_bundle(manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "eligible_rows": manifest["eligible_union"]["rows"],
                "eligible_nominal_hours": manifest["eligible_union"][
                    "nominal_hours"
                ],
                "shards": manifest["shards"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
