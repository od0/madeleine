"""Merge VLM prediction shards with strict coverage and provenance checks.

The classifier writes one append-only JSONL per worker.  This utility turns a
set of completed worker files into one reproducible artifact, but only after
checking that duplicate IDs agree, classification configuration is uniform,
and (when supplied) the merged IDs exactly cover an expected scan or ID list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROW_UNIFORM_FIELDS = (
    "model",
    "resolved_model_revision",
    "prompt_version",
    "prompt_sha256",
)
MANIFEST_UNIFORM_FIELDS = (
    *ROW_UNIFORM_FIELDS,
    "classical_uncertain_score",
    "classical_input_hud_uncertain",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"expected an object in {path}:{line_number}, "
                    f"got {type(value).__name__}"
                )
            rows.append(value)
    return rows


def load_expected_ids(path: Path) -> set[str]:
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if isinstance(value, dict):
            if "video_ids" not in value:
                raise ValueError(
                    f"expected JSON object {path} to contain a video_ids list"
                )
            value = value["video_ids"]
        if not isinstance(value, list):
            raise ValueError(f"expected {path} to contain a list of video IDs")
        raw_ids: Iterable[Any] = value
    else:
        raw_ids = (line.strip() for line in path.read_text().splitlines())

    ids = [str(value) for value in raw_ids if str(value)]
    if len(ids) != len(set(ids)):
        duplicates = sorted(
            video_id for video_id, count in Counter(ids).items() if count > 1
        )
        raise ValueError(
            f"duplicate video IDs in expected IDs file: {duplicates[:10]}"
        )
    return set(ids)


def _metadata_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _record_uniform_metadata(
    observed: dict[str, dict[str, tuple[Any, list[str]]]],
    values: dict[str, Any],
    fields: Iterable[str],
    source: str,
) -> None:
    for field in fields:
        value = values.get(field)
        if value is None:
            continue
        key = _metadata_key(value)
        prior = observed[field].get(key)
        if prior is None:
            observed[field][key] = (value, [source])
        else:
            prior[1].append(source)


def _validate_uniform_metadata(
    observed: dict[str, dict[str, tuple[Any, list[str]]]],
) -> dict[str, Any]:
    uniform: dict[str, Any] = {}
    for field in sorted(observed):
        variants = observed[field]
        if len(variants) > 1:
            details = "; ".join(
                f"{key} from {sorted(sources)[:3]}"
                for key, (_, sources) in sorted(variants.items())
            )
            raise ValueError(f"non-uniform {field}: {details}")
        if variants:
            uniform[field] = next(iter(variants.values()))[0]
    return uniform


def _hours_from_row(row: dict[str, Any]) -> float | None:
    duration_s = row.get("duration_s")
    if (
        isinstance(duration_s, (int, float))
        and not isinstance(duration_s, bool)
        and math.isfinite(duration_s)
        and duration_s >= 0
    ):
        return float(duration_s) / 3600.0
    nominal_hours = row.get("nominal_hours")
    if (
        isinstance(nominal_hours, (int, float))
        and not isinstance(nominal_hours, bool)
        and math.isfinite(nominal_hours)
        and nominal_hours >= 0
    ):
        return float(nominal_hours)
    return None


def _load_expected_scan(
    path: Path,
) -> tuple[set[str], dict[str, dict[str, Any]], int]:
    eligible_rows: dict[str, dict[str, Any]] = {}
    ineligible_count = 0
    for row in load_jsonl(path):
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"expected scan row lacks a non-empty video_id: {row}")
        if row.get("error") is not None:
            ineligible_count += 1
            continue
        if video_id in eligible_rows:
            raise ValueError(f"duplicate eligible video ID in expected scan: {video_id}")
        eligible_rows[video_id] = row
    return set(eligible_rows), eligible_rows, ineligible_count


def merge_prediction_shards(
    prediction_paths: list[Path],
    out: Path,
    *,
    expected_scan: Path | None = None,
    expected_ids_file: Path | None = None,
) -> dict[str, Any]:
    """Validate and merge completed prediction shards, returning the manifest."""

    if not prediction_paths:
        raise ValueError("at least one prediction input is required")
    if expected_scan is not None and expected_ids_file is not None:
        raise ValueError("use only one of expected_scan and expected_ids_file")

    resolved_inputs = [path.resolve() for path in prediction_paths]
    if len(resolved_inputs) != len(set(resolved_inputs)):
        raise ValueError("the same prediction path was supplied more than once")
    if out.resolve() in set(resolved_inputs):
        raise ValueError("output path must not overwrite a prediction input")

    expected_ids: set[str] | None = None
    expected_rows: dict[str, dict[str, Any]] = {}
    expected_manifest: dict[str, Any] | None = None
    if expected_scan is not None:
        expected_ids, expected_rows, ineligible_count = _load_expected_scan(
            expected_scan
        )
        expected_manifest = {
            "kind": "eligible_rows_from_scan",
            "path": str(expected_scan.resolve()),
            "sha256": sha256(expected_scan),
            "eligible_rows": len(expected_ids),
            "ineligible_rows": ineligible_count,
        }
    elif expected_ids_file is not None:
        expected_ids = load_expected_ids(expected_ids_file)
        expected_manifest = {
            "kind": "video_ids",
            "path": str(expected_ids_file.resolve()),
            "sha256": sha256(expected_ids_file),
            "eligible_rows": len(expected_ids),
        }

    rows_by_id: dict[str, dict[str, Any]] = {}
    first_source_by_id: dict[str, str] = {}
    identical_duplicate_ids: list[str] = []
    input_summaries: list[dict[str, Any]] = []
    observed_metadata: dict[str, dict[str, tuple[Any, list[str]]]] = defaultdict(dict)

    for path in prediction_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = load_jsonl(path)
        sidecar = path.with_suffix(path.suffix + ".manifest.json")
        sidecar_summary: dict[str, Any] | None = None
        if sidecar.is_file():
            sidecar_value = json.loads(sidecar.read_text())
            if not isinstance(sidecar_value, dict):
                raise ValueError(f"prediction manifest is not an object: {sidecar}")
            _record_uniform_metadata(
                observed_metadata,
                sidecar_value,
                MANIFEST_UNIFORM_FIELDS,
                str(sidecar),
            )
            sidecar_summary = {
                "path": str(sidecar.resolve()),
                "sha256": sha256(sidecar),
            }

        for line_number, row in enumerate(rows, start=1):
            source = f"{path}:{line_number}"
            video_id = row.get("video_id")
            if not isinstance(video_id, str) or not video_id:
                raise ValueError(f"prediction lacks a non-empty video_id at {source}")
            label = row.get("class")
            if not isinstance(label, str) or not label:
                raise ValueError(f"prediction lacks a non-empty class at {source}")
            _record_uniform_metadata(
                observed_metadata, row, ROW_UNIFORM_FIELDS, source
            )
            prior = rows_by_id.get(video_id)
            if prior is None:
                rows_by_id[video_id] = row
                first_source_by_id[video_id] = source
            elif prior == row:
                identical_duplicate_ids.append(video_id)
            else:
                raise ValueError(
                    f"conflicting duplicate prediction for {video_id}: "
                    f"{first_source_by_id[video_id]} vs {source}"
                )

        input_summaries.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "rows": len(rows),
                "manifest": sidecar_summary,
            }
        )

    uniform_metadata = _validate_uniform_metadata(observed_metadata)
    merged_ids = set(rows_by_id)
    if expected_ids is not None:
        missing = sorted(expected_ids - merged_ids)
        unexpected = sorted(merged_ids - expected_ids)
        if missing or unexpected:
            raise ValueError(
                "merged prediction coverage does not exactly match expected IDs: "
                f"missing={len(missing)} {missing[:10]}, "
                f"unexpected={len(unexpected)} {unexpected[:10]}"
            )

    ordered_rows = [rows_by_id[video_id] for video_id in sorted(rows_by_id)]
    class_counts = Counter(str(row["class"]) for row in ordered_rows)
    hours_by_class: dict[str, float] = defaultdict(float)
    rows_with_hours = 0
    for row in ordered_rows:
        metadata_row = expected_rows.get(row["video_id"], {})
        hours = _hours_from_row(row)
        if hours is None:
            hours = _hours_from_row(metadata_row)
        if hours is not None:
            rows_with_hours += 1
            hours_by_class[str(row["class"])] += hours

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered_rows)
    )
    manifest = {
        "schema_version": 1,
        "merge_order": "video_id_lexicographic",
        "rows": len(ordered_rows),
        "unique_video_ids": len(merged_ids),
        "class_counts": dict(sorted(class_counts.items())),
        "nominal_hours": {
            "known_rows": rows_with_hours,
            "missing_rows": len(ordered_rows) - rows_with_hours,
            "total_known": sum(hours_by_class.values()),
            "by_class": dict(sorted(hours_by_class.items())),
        },
        "identical_duplicate_rows_deduplicated": len(identical_duplicate_ids),
        "identical_duplicate_video_ids": sorted(set(identical_duplicate_ids)),
        "uniform_metadata": uniform_metadata,
        "expected": expected_manifest,
        "prediction_inputs": sorted(input_summaries, key=lambda item: item["path"]),
        "output": {
            "path": str(out.resolve()),
            "sha256": sha256(out),
        },
    }
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictions",
        type=Path,
        action="append",
        required=True,
        help="prediction JSONL; repeat for each pre-shard/shard artifact",
    )
    expected = ap.add_mutually_exclusive_group()
    expected.add_argument(
        "--expected-scan",
        type=Path,
        help="JSONL whose error-free video IDs must be covered exactly",
    )
    expected.add_argument(
        "--expected-ids-file",
        type=Path,
        help="newline-delimited IDs or JSON video_ids list to cover exactly",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = merge_prediction_shards(
        args.predictions,
        args.out,
        expected_scan=args.expected_scan,
        expected_ids_file=args.expected_ids_file,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
