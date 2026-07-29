"""Freeze and repartition fetch queues at a completion-marker barrier.

The compiler consumes disjoint machine-nomination queues, removes IDs already
present in an immutable completion snapshot or reserved by excluded queues,
and balances the remainder across an ordered set of workers.  Previously
failed rows can be named as retries; they are assigned to distinct lightest
workers and appended after ordinary work so failures cannot block a healthy
lane at startup.

Every output is content-addressed in a manifest.  Publication is atomic per
file, idempotent for identical bytes, and refuses to overwrite any conflicting
artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Sequence


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class QueueSource:
    path: Path
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PartitionPlan:
    source_rows: tuple[dict[str, Any], ...]
    selected_source_rows: tuple[dict[str, Any], ...]
    filtered_out_rows: tuple[dict[str, Any], ...]
    eligible_rows: tuple[dict[str, Any], ...]
    completed_input_ids: frozenset[str]
    excluded_input_ids: frozenset[str]
    retry_ids: tuple[str, ...]
    selected_sources: tuple[str, ...]
    shard_labels: tuple[str, ...]
    shards: tuple[tuple[dict[str, Any], ...], ...]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {line_number} is not valid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}: line {line_number} is not an object")
        rows.append(row)
    return rows


def _validate_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} contains an unsafe video_id")
    return value


def nominal_hours(row: dict[str, Any], label: str) -> float:
    value = row.get("nominal_hours")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} has missing or non-numeric nominal_hours")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} has non-finite or negative nominal_hours")
    return result


def validate_queue_rows(
    rows: Sequence[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = _validate_id(row.get("video_id"), label)
        if video_id in indexed:
            raise ValueError(f"{label} contains duplicate video_id {video_id}")
        nominal_hours(row, f"{label} row {video_id}")
        if row.get("human_reviewed") is not False:
            raise ValueError(f"{label} row {video_id} must be human_reviewed=false")
        training_admitted = row.get("training_admitted")
        if training_admitted is not None and training_admitted is not False:
            raise ValueError(
                f"{label} row {video_id} must be training_admitted=false or null"
            )
        if row.get("machine_nomination_only") is not True:
            raise ValueError(
                f"{label} row {video_id} must remain machine_nomination_only=true"
            )
        indexed[video_id] = row
    return indexed


def load_queue(path: Path) -> QueueSource:
    rows = load_jsonl(path)
    validate_queue_rows(rows, str(path))
    return QueueSource(path=path, rows=tuple(rows))


def load_id_snapshot(path: Path, label: str = "ID snapshot") -> list[str]:
    raw_values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    values: list[str] = []
    for raw_value in raw_values:
        if raw_value.endswith("/upload_complete.json"):
            video_id, separator, suffix = raw_value.partition("/")
            if separator != "/" or suffix != "upload_complete.json":
                raise ValueError(f"{label} contains an unsafe completion path")
            values.append(video_id)
        else:
            values.append(raw_value)
    seen: set[str] = set()
    for value in values:
        _validate_id(value, label)
        if value in seen:
            raise ValueError(f"{label} contains duplicate video_id {value}")
        seen.add(value)
    return sorted(values)


def _validate_components(values: Sequence[str], label: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"at least one {label} is required")
    if len(values) != len(set(values)):
        raise ValueError(f"{label}s must be unique")
    for value in values:
        if not _SAFE_COMPONENT.fullmatch(value):
            raise ValueError(f"unsafe {label}: {value!r}")
    return tuple(values)


def _validate_source_filter(values: Sequence[str]) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError("source filters must be unique")
    for value in values:
        if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
            raise ValueError(f"unsafe source filter: {value!r}")
    return tuple(sorted(values))


def _filter_source_rows(
    rows: Sequence[dict[str, Any]], selected_sources: tuple[str, ...]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_set = set(selected_sources)
    selected: list[dict[str, Any]] = []
    filtered_out: list[dict[str, Any]] = []
    for row in rows:
        video_id = str(row["video_id"])
        if selected_sources:
            source = row.get("source")
            if not isinstance(source, str) or not _SAFE_COMPONENT.fullmatch(source):
                raise ValueError(
                    f"input row {video_id} has missing or unsafe source for filtering"
                )
            if source not in selected_set:
                filtered_out.append(row)
                continue
        normalized = dict(row)
        normalized["human_reviewed"] = False
        normalized["training_admitted"] = False
        normalized["machine_nomination_only"] = True
        selected.append(normalized)
    return selected, filtered_out


def _merge_disjoint_sources(
    sources: Sequence[QueueSource], label: str
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    rows: list[dict[str, Any]] = []
    owners: dict[str, Path] = {}
    for source in sources:
        validate_queue_rows(source.rows, str(source.path))
        for row in source.rows:
            video_id = str(row["video_id"])
            prior = owners.get(video_id)
            if prior is not None:
                raise ValueError(
                    f"{label} queues overlap on {video_id}: {prior} and {source.path}"
                )
            owners[video_id] = source.path
            rows.append(row)
    return rows, owners


def _validated_retry_ids(
    retry_ids: Sequence[str], eligible_ids: set[str], shard_count: int
) -> tuple[str, ...]:
    for retry_id in retry_ids:
        _validate_id(retry_id, "retry IDs")
    if len(retry_ids) != len(set(retry_ids)):
        raise ValueError("retry IDs contain duplicates")
    ordered = tuple(sorted(retry_ids))
    missing = set(ordered).difference(eligible_ids)
    if missing:
        raise ValueError(f"retry IDs are not eligible: {sorted(missing)}")
    if len(ordered) > shard_count:
        raise ValueError("retry IDs exceed distinct shard count")
    return ordered


def build_partition(
    queue_sources: Sequence[QueueSource],
    completed_ids: set[str],
    excluded_sources: Sequence[QueueSource],
    shard_labels: Sequence[str],
    retry_ids: Sequence[str] = (),
    selected_sources: Sequence[str] = (),
) -> PartitionPlan:
    labels = _validate_components(shard_labels, "shard label")
    source_filter = _validate_source_filter(selected_sources)
    for video_id in completed_ids:
        _validate_id(video_id, "completed IDs")

    source_rows, source_owners = _merge_disjoint_sources(queue_sources, "input")
    excluded_rows, _ = _merge_disjoint_sources(excluded_sources, "excluded")
    excluded_ids = {str(row["video_id"]) for row in excluded_rows}
    selected_rows, filtered_out_rows = _filter_source_rows(source_rows, source_filter)

    eligible_rows = [
        row
        for row in selected_rows
        if row["video_id"] not in completed_ids
        and row["video_id"] not in excluded_ids
    ]
    eligible_ids = {str(row["video_id"]) for row in eligible_rows}
    retries = _validated_retry_ids(retry_ids, eligible_ids, len(labels))
    retry_set = set(retries)

    shard_members: list[list[dict[str, Any]]] = [[] for _ in labels]
    shard_hours = [0.0] * len(labels)
    ordinary = [row for row in eligible_rows if row["video_id"] not in retry_set]
    for row in sorted(
        ordinary,
        key=lambda item: (
            -nominal_hours(item, str(item["video_id"])),
            str(item["video_id"]),
        ),
    ):
        index = min(
            range(len(labels)), key=lambda value: (shard_hours[value], value)
        )
        shard_members[index].append(row)
        shard_hours[index] += nominal_hours(row, str(row["video_id"]))

    row_by_id = {str(row["video_id"]): row for row in eligible_rows}
    available = set(range(len(labels)))
    for retry_id in retries:
        index = min(available, key=lambda value: (shard_hours[value], value))
        row = row_by_id[retry_id]
        shard_members[index].append(row)
        shard_hours[index] += nominal_hours(row, retry_id)
        available.remove(index)

    input_position = {
        str(row["video_id"]): index for index, row in enumerate(eligible_rows)
    }
    retry_position = {video_id: index for index, video_id in enumerate(retries)}
    for shard in shard_members:
        shard.sort(
            key=lambda row: (
                row["video_id"] in retry_set,
                retry_position.get(
                    str(row["video_id"]), input_position[str(row["video_id"])]
                ),
            )
        )

    selected_ids = {str(row["video_id"]) for row in selected_rows}
    completed_input = frozenset(selected_ids).intersection(completed_ids)
    excluded_input = frozenset(selected_ids).intersection(excluded_ids)
    plan = PartitionPlan(
        source_rows=tuple(source_rows),
        selected_source_rows=tuple(selected_rows),
        filtered_out_rows=tuple(filtered_out_rows),
        eligible_rows=tuple(eligible_rows),
        completed_input_ids=frozenset(completed_input),
        excluded_input_ids=frozenset(excluded_input),
        retry_ids=retries,
        selected_sources=source_filter,
        shard_labels=labels,
        shards=tuple(tuple(shard) for shard in shard_members),
    )
    validate_partition(plan)
    return plan


def validate_partition(plan: PartitionPlan) -> None:
    eligible_ids = {str(row["video_id"]) for row in plan.eligible_rows}
    filtered_out_ids = {str(row["video_id"]) for row in plan.filtered_out_rows}
    if eligible_ids.intersection(filtered_out_ids):
        raise AssertionError("source-filtered rows remain eligible")
    output_ids: set[str] = set()
    retry_set = set(plan.retry_ids)
    for label, shard in zip(plan.shard_labels, plan.shards, strict=True):
        ids = [str(row["video_id"]) for row in shard]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"output shard {label} contains duplicates")
        overlap = output_ids.intersection(ids)
        if overlap:
            raise AssertionError(f"output shards overlap: {sorted(overlap)}")
        output_ids.update(ids)
        seen_retry = False
        for video_id in ids:
            if video_id in retry_set:
                seen_retry = True
            elif seen_retry:
                raise AssertionError(f"output shard {label} has work after a retry")
    if output_ids != eligible_ids:
        missing = eligible_ids.difference(output_ids)
        extra = output_ids.difference(eligible_ids)
        raise AssertionError(
            f"output union mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )


def _total_hours(rows: Iterable[dict[str, Any]]) -> float:
    return sum(nominal_hours(row, str(row["video_id"])) for row in rows)


def build_artifacts(
    *,
    name: str,
    out_dir: Path,
    plan: PartitionPlan,
    queue_sources: Sequence[QueueSource],
    completed_path: Path,
    completed_ids: Sequence[str],
    excluded_sources: Sequence[QueueSource],
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    if not _SAFE_COMPONENT.fullmatch(name):
        raise ValueError(f"unsafe output name: {name!r}")
    validate_partition(plan)

    completed_bytes = "".join(value + "\n" for value in sorted(completed_ids)).encode()
    excluded_ids = sorted(
        str(row["video_id"]) for source in excluded_sources for row in source.rows
    )
    excluded_bytes = "".join(value + "\n" for value in excluded_ids).encode()
    eligible_bytes = jsonl_text(plan.eligible_rows).encode()

    completed_artifact = out_dir / f"{name}.completed-ids.txt"
    excluded_artifact = out_dir / f"{name}.excluded-ids.txt"
    eligible_artifact = out_dir / f"{name}.eligible.jsonl"
    artifacts: dict[Path, bytes] = {
        completed_artifact: completed_bytes,
        excluded_artifact: excluded_bytes,
        eligible_artifact: eligible_bytes,
    }

    shard_manifest: list[dict[str, Any]] = []
    for label, shard in zip(plan.shard_labels, plan.shards, strict=True):
        path = out_dir / f"{name}.shard-{label}.jsonl"
        content = jsonl_text(shard).encode()
        artifacts[path] = content
        retry_rows = [
            str(row["video_id"])
            for row in shard
            if row["video_id"] in set(plan.retry_ids)
        ]
        shard_manifest.append(
            {
                "label": label,
                "path": path.name,
                "rows": len(shard),
                "nominal_hours": _total_hours(shard),
                "retry_ids_appended_last": retry_rows,
                "sha256": sha256_bytes(content),
            }
        )

    input_ids = {str(row["video_id"]) for row in plan.source_rows}
    completed_rows = [
        row
        for row in plan.selected_source_rows
        if row["video_id"] in plan.completed_input_ids
    ]
    excluded_rows = [
        row
        for row in plan.selected_source_rows
        if row["video_id"] in plan.excluded_input_ids
        and row["video_id"] not in plan.completed_input_ids
    ]
    manifest: dict[str, Any] = {
        "format_version": "madeleine.fetch-repartition.v1",
        "name": name,
        "semantics": "machine-only fetch scheduling; not review or admission",
        "human_reviewed": False,
        "training_admitted": False,
        "source_inputs": [
            {
                "path": str(source.path),
                "rows": len(source.rows),
                "nominal_hours": _total_hours(source.rows),
                "sha256": sha256_file(source.path),
            }
            for source in queue_sources
        ],
        "source_union": {
            "rows": len(plan.source_rows),
            "nominal_hours": _total_hours(plan.source_rows),
            "unique_ids": len(input_ids),
        },
        "selected_sources": list(plan.selected_sources),
        "source_filter": {
            "enabled": bool(plan.selected_sources),
            "selected_sources": list(plan.selected_sources),
            "selected_rows": len(plan.selected_source_rows),
            "selected_nominal_hours": _total_hours(plan.selected_source_rows),
            "filtered_out_rows": len(plan.filtered_out_rows),
            "filtered_out_nominal_hours": _total_hours(plan.filtered_out_rows),
        },
        "completed_snapshot_input": {
            "path": str(completed_path),
            "ids": len(completed_ids),
            "sha256": sha256_file(completed_path),
        },
        "completed_snapshot_artifact": {
            "path": completed_artifact.name,
            "ids": len(completed_ids),
            "sha256": sha256_bytes(completed_bytes),
        },
        "completed_source_rows_removed": {
            "rows": len(completed_rows),
            "nominal_hours": _total_hours(completed_rows),
        },
        "excluded_inputs": [
            {
                "path": str(source.path),
                "rows": len(source.rows),
                "nominal_hours": _total_hours(source.rows),
                "sha256": sha256_file(source.path),
            }
            for source in excluded_sources
        ],
        "excluded_snapshot_artifact": {
            "path": excluded_artifact.name,
            "ids": len(excluded_ids),
            "sha256": sha256_bytes(excluded_bytes),
        },
        "excluded_source_rows_removed_after_completion_filter": {
            "rows": len(excluded_rows),
            "nominal_hours": _total_hours(excluded_rows),
        },
        "eligible_master": {
            "path": eligible_artifact.name,
            "rows": len(plan.eligible_rows),
            "nominal_hours": _total_hours(plan.eligible_rows),
            "sha256": sha256_bytes(eligible_bytes),
        },
        "retry_ids": list(plan.retry_ids),
        "partition_method": (
            "ordinary rows: largest nominal hours first, video_id tie-break, "
            "ordered-shard lightest-first assignment; original input order within "
            "each shard; retries: sorted IDs on distinct lightest shards, appended last"
        ),
        "ordered_shard_labels": list(plan.shard_labels),
        "partition_invariants": {
            "input_queues_disjoint": True,
            "output_shards_disjoint": True,
            "output_union_exactly_eligible": True,
            "completed_ids_absent": True,
            "excluded_ids_absent": True,
            "source_filtered_ids_absent": True,
            "source_filter_applied_before_completion_and_exclusion": True,
            "retry_ids_appended_last": True,
        },
        "shards": shard_manifest,
    }
    manifest_path = out_dir / f"{name}.manifest.json"
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    artifacts[manifest_path] = manifest_bytes
    return artifacts, manifest


def write_immutable_files(files: dict[Path, bytes]) -> None:
    """Atomically publish files and reject a conflicting prior partition."""

    conflicts = [
        path
        for path, content in files.items()
        if path.exists()
        and (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != content
        )
    ]
    if conflicts:
        raise FileExistsError(
            f"refusing to overwrite immutable repartition artifact: {conflicts[0]}"
        )

    for path, content in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.read_bytes() != content
                ):
                    raise FileExistsError(
                        f"concurrent immutable repartition conflict: {path}"
                    ) from None
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, action="append", required=True)
    parser.add_argument("--completed-ids", type=Path, required=True)
    parser.add_argument("--exclude-queue", type=Path, action="append", default=[])
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--shard-label", action="append", required=True)
    parser.add_argument("--retry-id", action="append", default=[])
    parser.add_argument("--retry-ids", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    queue_sources = [load_queue(path) for path in args.queue]
    excluded_sources = [load_queue(path) for path in args.exclude_queue]
    completed_ids = load_id_snapshot(args.completed_ids, "completed snapshot")
    retry_ids = list(args.retry_id)
    if args.retry_ids is not None:
        retry_ids.extend(load_id_snapshot(args.retry_ids, "retry ID snapshot"))
    plan = build_partition(
        queue_sources,
        set(completed_ids),
        excluded_sources,
        args.shard_label,
        retry_ids,
        args.source,
    )
    artifacts, manifest = build_artifacts(
        name=args.name,
        out_dir=args.out_dir,
        plan=plan,
        queue_sources=queue_sources,
        completed_path=args.completed_ids,
        completed_ids=completed_ids,
        excluded_sources=excluded_sources,
    )
    write_immutable_files(artifacts)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
