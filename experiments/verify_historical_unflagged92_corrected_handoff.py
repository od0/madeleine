#!/usr/bin/env python3
"""Independently verify the local historical-unflagged92 corrected handoff.

This verifier deliberately does not import the builder.  It re-derives exact
membership, VPT-v2 record and object filtering, changed-label totals, leak
exclusions, and frozen-feature/corrected-label joins from the source
authorities.  It performs no R2 I/O and cannot create a completion marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


Y4N = "y4nQHqYSObI"
KEYS = ("left", "right", "up", "down", "jump", "dash", "grab")
MEMBERSHIP_SHA = "e668ccbd0aa02fb1bda79c2da621df6ae4cb0d183280580076e20b9f00996943"
VIDEO_SHA = "2e06cab47c7921fb645097070a44497333cbd0bbe8d5f835a131cf06387cbf72"
EXPECTED = {
    "videos": 92,
    "sessions": 1062,
    "derived_rows": 7_445_200,
    "native_rows": 22_335_600,
    "windows": 114_821,
    "objects": 8_496,
    "changed_rows": 295_711,
}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value: Any) -> str:
    return bytes_sha(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    )


def load_json(path: Path, *, content: bool = False) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if content:
        payload = dict(value)
        claimed = payload.pop("content_sha256", None)
        if claimed != canonical_sha(payload):
            raise RuntimeError(f"{path}: content hash mismatch")
    return value


def sessions(path: Path) -> list[str]:
    values = path.read_text(encoding="utf-8").splitlines()
    if path.read_bytes() != ("\n".join(values) + "\n").encode():
        raise RuntimeError(f"{path}: noncanonical lines")
    if values != sorted(set(values)):
        raise RuntimeError(f"{path}: not sorted/unique")
    return values


def video(session: str) -> str:
    return session.split("__r", 1)[0]


def inventory(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows = [
        {
            "relative_path": str(row["relative_path"]),
            "bytes": int(row["bytes"]),
            "sha256": str(row["sha256"]),
        }
        for row in rows
    ]
    paths = [row["relative_path"] for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise RuntimeError(f"{path}: inventory not sorted/unique")
    return rows


def record_objects(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    sid = str(record["session_id"])
    directory = f"train-p0/{sid}__p0"
    declared = [record["metadata_file"], *record["arrays"].values()]
    return sorted(
        ({
            "relative_path": f"{directory}/{row['file']}",
            "bytes": int(row["bytes"]),
            "sha256": str(row["sha256"]),
        } for row in declared),
        key=lambda row: row["relative_path"],
    )


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def seal(value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value)
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha(value)
    return value


def verify(args: argparse.Namespace) -> dict[str, Any]:
    bundle = args.bundle
    names = (
        "train_sessions.txt", "video_ids.txt", "build_manifest.json",
        "object_inventory.jsonl", "changed_row_audit.json",
        "leak_exclusions.json", "gru_corrected_input_manifest.jsonl",
        "gru_corrected_input_proposal.json", "handoff_authority.json",
        "publication_marker_template.json",
    )
    for name in names:
        if not (bundle / name).is_file():
            raise FileNotFoundError(bundle / name)

    source_sessions = sessions(args.historical_reference / "train_sessions.txt")
    bundle_sessions = sessions(bundle / "train_sessions.txt")
    if bundle_sessions != source_sessions or file_sha(bundle / "train_sessions.txt") != MEMBERSHIP_SHA:
        raise RuntimeError("bundle membership differs from historical authority")
    videos = sorted({video(sid) for sid in bundle_sessions})
    bundle_videos = sessions(bundle / "video_ids.txt")
    if bundle_videos != videos or bytes_sha(("\n".join(videos) + "\n").encode()) != VIDEO_SHA:
        raise RuntimeError("bundle video identity mismatch")
    if len(videos) != EXPECTED["videos"] or Y4N in videos:
        raise RuntimeError("bundle video count/holdout exclusion mismatch")

    full = load_json(args.corrected_vpt_root / "build_manifest.json", content=True)
    full_by_session = {str(row["session_id"]): row for row in full["records"]}
    expected_records = [full_by_session[sid] for sid in bundle_sessions]
    exact_filter = [
        row for row in full["records"] if video(str(row["session_id"])) in set(videos)
    ]
    if exact_filter != expected_records:
        raise RuntimeError("historical sessions are not exact full-v2 video filter")
    subset = load_json(bundle / "build_manifest.json", content=True)
    if subset["records"] != expected_records:
        raise RuntimeError("subset records differ from corrected VPT-v2")
    totals = {
        "source_sessions": len(expected_records),
        "derived_streams": len(expected_records),
        "derived_rows": sum(int(row["derived_rows"]) for row in expected_records),
        "windows": sum(int(row["windows"]) for row in expected_records),
    }
    native_rows = sum(int(row["source"]["source_rows"]) for row in expected_records)
    if (
        totals != subset["totals"]
        or totals["source_sessions"] != EXPECTED["sessions"]
        or totals["derived_rows"] != EXPECTED["derived_rows"]
        or totals["windows"] != EXPECTED["windows"]
        or native_rows != EXPECTED["native_rows"]
        or native_rows != 3 * totals["derived_rows"]
    ):
        raise RuntimeError("subset geometry mismatch")

    expected_inventory = sorted(
        (obj for row in expected_records for obj in record_objects(row)),
        key=lambda row: row["relative_path"],
    )
    observed_inventory = inventory(bundle / "object_inventory.jsonl")
    if observed_inventory != expected_inventory or len(observed_inventory) != EXPECTED["objects"]:
        raise RuntimeError("bundle object inventory mismatch")
    full_inventory = inventory(args.corrected_vpt_root / "object_inventory.jsonl")
    prefix_set = {f"train-p0/{sid}__p0/" for sid in bundle_sessions}
    filtered_inventory = [
        row for row in full_inventory
        if any(row["relative_path"].startswith(prefix) for prefix in prefix_set)
    ]
    if filtered_inventory != observed_inventory:
        raise RuntimeError("inventory is not exact full-v2 session filter")

    repair = load_json(args.corrected_vpt_root / "label_repair_receipt.json", content=True)
    selected_video_rows = [repair["per_video"][vid] for vid in videos]
    changed_rows = sum(int(row["changed_rows"]) for row in selected_video_rows)
    changed_by_key = {
        key: sum(int(row["changed_by_key"][key]) for row in selected_video_rows)
        for key in KEYS
    }
    record_changed = sum(
        int(row["label_correction"]["changed_rows"]) for row in expected_records
    )
    record_by_key = {
        key: sum(int(row["label_correction"]["changed_by_key"][key]) for row in expected_records)
        for key in KEYS
    }
    audit = load_json(bundle / "changed_row_audit.json", content=True)
    if (
        changed_rows != EXPECTED["changed_rows"]
        or record_changed != changed_rows
        or record_by_key != changed_by_key
        or audit["changed_rows"] != changed_rows
        or audit["changed_by_key"] != changed_by_key
        or changed_by_key["left"] != 0
        or changed_by_key["right"] != 0
    ):
        raise RuntimeError("changed-label audit mismatch")

    corrected95_videos = sessions(args.unflagged95_videos)
    corrected95_sessions = sessions(args.unflagged95_sessions)
    corrected_only = sorted(set(corrected95_videos) - set(videos))
    if (
        set(videos) - set(corrected95_videos)
        or corrected_only != ["9MD9YUi63Ng", "aEqQWc04jIA", "v1459001667"]
        or not set(bundle_sessions).issubset(corrected95_sessions)
    ):
        raise RuntimeError("historical92/corrected95 separation mismatch")
    leak = load_json(bundle / "leak_exclusions.json", content=True)
    if not all(leak.get("proofs", {}).values()) or leak["holdout_excluded"] != Y4N:
        raise RuntimeError("leak-exclusion proof is not PASS")

    run_meta = load_json(args.gru_run_meta)
    if list(map(str, run_meta["split"]["train"])) != bundle_sessions:
        raise RuntimeError("historical GRU membership mismatch")
    feature_hashes = dict(run_meta["shard_sha256"])
    gru_rows = [json.loads(line) for line in (bundle / "gru_corrected_input_manifest.jsonl").read_text().splitlines()]
    if [row["session_id"] for row in gru_rows] != bundle_sessions:
        raise RuntimeError("GRU corrected-input manifest membership mismatch")
    for row, record in zip(gru_rows, expected_records, strict=True):
        sid = row["session_id"]
        arrays = record["arrays"]
        if (
            row["original_frozen_feature_shard"]["whole_npz_sha256"] != feature_hashes[sid]
            or row["native_rows"] != int(record["source"]["source_rows"])
            or row["corrected_vpt20_labels"]["keys_sha256"] != arrays["keys"]["sha256"]
            or row["corrected_vpt20_labels"]["source_engine_frame_idx_sha256"]
            != arrays["source_engine_frame_idx"]["sha256"]
            or row["corrected_vpt20_labels"]["source_row_index_sha256"]
            != arrays["source_row_index"]["sha256"]
        ):
            raise RuntimeError(f"{sid}: frozen-feature/corrected-label binding mismatch")
    proposal = load_json(bundle / "gru_corrected_input_proposal.json", content=True)
    if (
        proposal["status"] != "exact_input_authority_materialization_pending"
        or proposal["original_frozen_feature_authority"]["run_meta_sha256"]
        != file_sha(args.gru_run_meta)
        or proposal["proofs_already_satisfied"]["r2_not_accessed"] is not True
    ):
        raise RuntimeError("GRU corrected-input proposal mismatch")

    handoff = load_json(bundle / "handoff_authority.json", content=True)
    for name, bound in handoff["payloads"].items():
        path = bundle / name
        if file_sha(path) != bound["sha256"] or path.stat().st_size != bound["bytes"]:
            raise RuntimeError(f"handoff payload binding mismatch: {name}")
    if (
        handoff["status"] != "local_prepublication_non_marker_bundle"
        or not all(handoff["proofs"].values())
        or any(row["observed"] for row in handoff["core_completion_markers"].values())
    ):
        raise RuntimeError("handoff incorrectly claims completion/publication")
    template = load_json(bundle / "publication_marker_template.json", content=True)
    if (
        template["status"] != "template_only_not_complete"
        or template["completion_claimed"] is not False
        or template["r2_publication_claimed"] is not False
    ):
        raise RuntimeError("marker template claims completion")

    checks = {
        "membership_exact_historical1062": True,
        "video_identity_exact_historical92": True,
        "y4n_excluded": True,
        "subset_records_exact_corrected_vpt_v2_filter": True,
        "geometry_exact": True,
        "object_inventory_exact_sorted_record_expansion": True,
        "object_inventory_exact_full_v2_filter": True,
        "changed_rows_equal_session_and_video_sums": True,
        "horizontal_labels_unchanged": True,
        "historical92_distinct_from_corrected95": True,
        "original_feature_whole_npz_hashes_equal_run_meta": True,
        "corrected_label_hashes_equal_vpt_v2_records": True,
        "handoff_payload_hashes_valid": True,
        "completion_not_claimed": True,
        "r2_io_not_performed": True,
        "evaluation_outputs_not_used": True,
    }
    receipt = seal({
        "schema_version": "madeleine.nitrogen-historical-unflagged92-independent-verification.v1",
        "status": "PASS",
        "bundle": str(bundle),
        "checks": checks,
        "exact_totals": {
            **totals,
            "videos": len(videos),
            "native_rows": native_rows,
            "objects": len(observed_inventory),
            "object_bytes": sum(row["bytes"] for row in observed_inventory),
            "changed_rows": changed_rows,
            "changed_by_key": changed_by_key,
        },
        # The template binds this receipt after PASS, so including the
        # template here would create a hash cycle. All immutable payloads and
        # the handoff authority are bound; the template is separately checked
        # above and then populated with this receipt's hash.
        "output_hashes": {
            name: file_sha(bundle / name)
            for name in names
            if name != "publication_marker_template.json"
        },
    })
    receipt_path = bundle / "independent_verification.json"
    write_json(receipt_path, receipt)

    template = load_json(bundle / "publication_marker_template.json", content=True)
    template["independent_verification"] = {
        "path": "independent_verification.json",
        "status": "PASS",
        "sha256": file_sha(receipt_path),
        "content_sha256": receipt["content_sha256"],
    }
    template = seal(template)
    write_json(bundle / "publication_marker_template.json", template)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path,
        default=Path("/private/tmp/nitrogen-axis-repair-v2/historical-unflagged92-corrected-v2"),
    )
    parser.add_argument(
        "--historical-reference", type=Path,
        default=Path("results/idm/vpt_small_data_scale_v1/unflagged92_reference_v1"),
    )
    parser.add_argument(
        "--corrected-vpt-root", type=Path,
        default=Path("/private/tmp/nitrogen-axis-repair-v2/vpt-v2-fresh"),
    )
    parser.add_argument(
        "--unflagged95-sessions", type=Path,
        default=Path("/private/tmp/nitrogen-axis-repair-v2/vpt-v2-fresh/provenance/unflagged95_train_sessions.txt"),
    )
    parser.add_argument(
        "--unflagged95-videos", type=Path,
        default=Path("/private/tmp/nitrogen-axis-repair-v2/vpt-v2-fresh/provenance/unflagged95_video_ids.txt"),
    )
    parser.add_argument(
        "--gru-run-meta", type=Path,
        default=Path("results/idm/nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0/run_meta.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = verify(args)
    print(json.dumps({
        "status": receipt["status"],
        "bundle": str(args.bundle),
        "receipt_sha256": file_sha(args.bundle / "independent_verification.json"),
        "exact_totals": receipt["exact_totals"],
        "r2_io_performed": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
