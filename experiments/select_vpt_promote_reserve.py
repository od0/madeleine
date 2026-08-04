#!/usr/bin/env python3
"""Select a deterministic labeler-unseen NitroGen promotion reserve.

The selector consumes metadata only.  It draws whole videos exclusively from
the all-valid-minus-unflagged pool, stratifies by decoder path and duration,
and approximates a 25-hour target without changing the frozen unflagged arm.
No labels, pixels, calibration captures, or evaluation surfaces are opened.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "madeleine.vpt-promote-reserve.v1"
SALT = "madeleine-vpt-promote-reserve-v1"
GRID_HZ = 60
TARGET_HOURS = 25
TARGET_FRAMES = TARGET_HOURS * 60 * 60 * GRID_HZ
WINDOW = 128
STRIDE = 64
GLOBAL_BATCH = 128
EPOCHS = 20
Y4N_VIDEO_ID = "y4nQHqYSObI"
WILD_NATIVE_ROWS = 2_947_146
WILD_PHASE0_ROWS = 982_845
WILD_BASE_WINDOWS = 13_166
DURATION_BINS = (
    ("<5m", 0, 5 * 60 * GRID_HZ),
    ("5-30m", 5 * 60 * GRID_HZ, 30 * 60 * GRID_HZ),
    ("30m-2h", 30 * 60 * GRID_HZ, 2 * 60 * 60 * GRID_HZ),
    (">=2h", 2 * 60 * 60 * GRID_HZ, None),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _input_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{path}: sha256 {actual} != {expected_sha256}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
    }


def _duration_bin(frames: int) -> str:
    for name, lower, upper in DURATION_BINS:
        if frames >= lower and (upper is None or frames < upper):
            return name
    raise AssertionError(frames)


def _phase0_rows(frames: int) -> int:
    return (frames + 2) // 3


def _base_windows(frames: int) -> int:
    rows = _phase0_rows(frames)
    if rows < WINDOW:
        return 0
    return 1 + (rows - WINDOW) // STRIDE


def _video_id(session_id: str) -> str:
    if "__r" not in session_id:
        raise ValueError(f"invalid NitroGen session ID {session_id!r}")
    return session_id.rsplit("__r", 1)[0]


def _salted_order(stratum: str, video_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{SALT}\0{stratum}\0{video_id}".encode("utf-8")
    ).hexdigest()
    return digest, video_id


def _subset_digest(video_ids: Sequence[str]) -> str:
    payload = f"{SALT}\n" + "".join(f"{video_id}\n" for video_id in video_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def nearest_subset(
    candidates: Sequence[tuple[str, int]],
    target: Fraction,
    *,
    stratum: str,
) -> tuple[list[str], int, int]:
    """Return exact nearest whole-video subset and its frame-unit GCD.

    Candidates are traversed in salted-SHA order.  The first subset to reach a
    frame sum wins collisions at that sum; equal-distance frame sums are then
    resolved by the salted canonical subset digest.
    """

    if not candidates:
        return [], 0, 1
    ordered = sorted(candidates, key=lambda row: _salted_order(stratum, row[0]))
    unit = 0
    for _video, frames in ordered:
        unit = math.gcd(unit, frames)
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for video_id, frames in ordered:
        weight = frames // unit
        additions: dict[int, tuple[str, ...]] = {}
        for total, subset in reachable.items():
            new_total = total + weight
            if new_total not in reachable and new_total not in additions:
                additions[new_total] = subset + (video_id,)
        reachable.update(additions)

    def rank(item: tuple[int, tuple[str, ...]]) -> tuple[Fraction, str]:
        total_units, subset = item
        selected = sorted(subset)
        return abs(Fraction(total_units * unit) - target), _subset_digest(selected)

    selected_units, selected_tuple = min(reachable.items(), key=rank)
    selected = sorted(selected_tuple)
    return selected, selected_units * unit, unit


def _load_run_meta(path: Path) -> tuple[list[str], list[str], Mapping[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = payload.get("split")
    shard_sha256 = payload.get("shard_sha256")
    if not isinstance(split, Mapping) or not isinstance(shard_sha256, Mapping):
        raise ValueError(f"{path}: missing split or shard_sha256")
    train = [str(value) for value in split.get("train", [])]
    val = [str(value) for value in split.get("val", [])]
    if len(train) != len(set(train)) or len(val) != len(set(val)):
        raise ValueError(f"{path}: duplicate split session")
    if set(train).intersection(val):
        raise ValueError(f"{path}: train/validation overlap")
    return train, val, {str(key): str(value) for key, value in shard_sha256.items()}


def select_reserve(
    *,
    all_valid_meta: Path,
    unflagged_meta: Path,
    inventory_path: Path,
    input_contracts: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str], dict[str, Any]]:
    all_sessions, all_val, all_hashes = _load_run_meta(all_valid_meta)
    unflagged_sessions, unflagged_val, unflagged_hashes = _load_run_meta(
        unflagged_meta
    )
    if all_val != unflagged_val or not all_val:
        raise ValueError("all-valid and unflagged validation membership differs")
    if {_video_id(session_id) for session_id in all_val} != {Y4N_VIDEO_ID}:
        raise ValueError("validation membership is not exactly the whole y4n video")
    for session_id in all_val:
        if all_hashes.get(session_id) != unflagged_hashes.get(session_id):
            raise ValueError("validation shard hashes differ between run metas")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory_sessions = {
        str(row["session_id"]): row
        for row in inventory.get("sessions", [])
        if row.get("source") == "nitrogen"
    }
    inventory_videos = {
        str(row["video_id"]): row for row in inventory.get("nitrogen_videos", [])
    }
    if set(all_sessions) != set(inventory_sessions):
        raise ValueError("all-valid train membership differs from inventory")
    if not set(unflagged_sessions).issubset(all_sessions):
        raise ValueError("unflagged train membership is not an all-valid subset")
    if any(session_id.startswith("rec_") for session_id in all_sessions):
        raise ValueError("local capture entered NitroGen membership")
    for session_id in all_sessions:
        expected = str(inventory_sessions[session_id]["reference_shard_sha256"])
        if all_hashes.get(session_id) != expected:
            raise ValueError(f"{session_id}: all-valid shard hash mismatch")
    for session_id in unflagged_sessions:
        expected = str(inventory_sessions[session_id]["reference_shard_sha256"])
        if unflagged_hashes.get(session_id) != expected:
            raise ValueError(f"{session_id}: unflagged shard hash mismatch")

    all_videos = {_video_id(session_id) for session_id in all_sessions}
    unflagged_videos = {_video_id(session_id) for session_id in unflagged_sessions}
    if all_videos != set(inventory_videos):
        raise ValueError("all-valid video membership differs from inventory")
    candidate_videos = all_videos.difference(unflagged_videos)
    if not candidate_videos:
        raise ValueError("reserve candidate pool is empty")

    sessions_by_video: dict[str, list[Mapping[str, Any]]] = {
        video_id: [] for video_id in all_videos
    }
    for session_id in all_sessions:
        sessions_by_video[_video_id(session_id)].append(inventory_sessions[session_id])
    frames_by_video = {
        video_id: sum(int(row["frames"]) for row in rows)
        for video_id, rows in sessions_by_video.items()
    }
    strata: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for video_id in sorted(candidate_videos):
        decoder = str(inventory_videos[video_id]["decoder_mode"])
        duration_bin = _duration_bin(frames_by_video[video_id])
        strata.setdefault((decoder, duration_bin), []).append(
            (video_id, frames_by_video[video_id])
        )

    candidate_frames = sum(frames_by_video[video_id] for video_id in candidate_videos)
    reserve: set[str] = set()
    stratum_receipts: list[dict[str, Any]] = []
    for decoder, duration_bin in sorted(strata):
        candidates = strata[(decoder, duration_bin)]
        stratum_frames = sum(frames for _video, frames in candidates)
        target = Fraction(TARGET_FRAMES * stratum_frames, candidate_frames)
        name = f"{decoder}/{duration_bin}"
        selected, selected_frames, unit = nearest_subset(
            candidates, target, stratum=name
        )
        reserve.update(selected)
        stratum_receipts.append(
            {
                "decoder_mode": decoder,
                "duration_bin": duration_bin,
                "candidate_videos": len(candidates),
                "candidate_frames": stratum_frames,
                "target_frames_fraction": {
                    "numerator": target.numerator,
                    "denominator": target.denominator,
                },
                "selected_videos": selected,
                "selected_frames": selected_frames,
                "absolute_target_delta_frames_fraction": {
                    "numerator": abs(Fraction(selected_frames) - target).numerator,
                    "denominator": abs(Fraction(selected_frames) - target).denominator,
                },
                "subset_sum_frame_unit": unit,
            }
        )

    retained = all_videos.difference(reserve)
    reserve_frames = sum(frames_by_video[video_id] for video_id in reserve)
    largest_candidate = max(frames_by_video[video_id] for video_id in candidate_videos)
    if abs(reserve_frames - TARGET_FRAMES) > largest_candidate:
        raise ValueError("reserve misses target by more than the largest candidate")
    if reserve.intersection(unflagged_videos):
        raise ValueError("reserve overlaps the frozen unflagged arm")
    if reserve.intersection(retained) or reserve.union(retained) != all_videos:
        raise ValueError("reserve/retained partition is invalid")
    if Y4N_VIDEO_ID in reserve or Y4N_VIDEO_ID in retained:
        raise ValueError("y4n entered train/reserve membership")

    non_nitrogen_ids = {
        str(row.get("video_id"))
        for row in inventory.get("sessions", [])
        if row.get("source") != "nitrogen" and row.get("video_id") is not None
    }
    if reserve.intersection(non_nitrogen_ids):
        raise ValueError("reserve overlaps a non-NitroGen/Tier-B inventory video")

    reserve_sessions = [
        row for video_id in reserve for row in sessions_by_video[video_id]
    ]
    retained_sessions = [
        row for video_id in retained for row in sessions_by_video[video_id]
    ]
    reserve_phase0 = sum(_phase0_rows(int(row["frames"])) for row in reserve_sessions)
    reserve_windows = sum(_base_windows(int(row["frames"])) for row in reserve_sessions)
    retained_native = sum(int(row["frames"]) for row in retained_sessions)
    retained_phase0 = sum(
        _phase0_rows(int(row["frames"])) for row in retained_sessions
    )
    retained_windows = sum(
        _base_windows(int(row["frames"])) for row in retained_sessions
    )
    max_native = retained_native + WILD_NATIVE_ROWS
    max_phase0 = retained_phase0 + WILD_PHASE0_ROWS
    max_windows = retained_windows + WILD_BASE_WINDOWS
    steps_per_epoch = math.ceil(max_windows / GLOBAL_BATCH)

    reserve_list = sorted(reserve)
    retained_list = sorted(retained)
    reserve_text = "".join(f"{video_id}\n" for video_id in reserve_list)
    retained_text = "".join(f"{video_id}\n" for video_id in retained_list)
    implementation_path = Path(__file__).resolve()
    try:
        implementation_label = implementation_path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        implementation_label = implementation_path.name
    receipt = {
        "schema_version": SCHEMA,
        "implementation": {
            "path": implementation_label,
            "bytes": implementation_path.stat().st_size,
            "sha256": sha256_file(implementation_path),
        },
        "inputs": dict(input_contracts),
        "inventory_content_sha256": inventory.get("inventory_content_sha256"),
        "algorithm": {
            "salt": SALT,
            "target_hours": TARGET_HOURS,
            "target_native_frames": TARGET_FRAMES,
            "population": "all-valid train videos minus unflagged train videos",
            "strata": "decoder_mode x duration_bin",
            "duration_bins_native_frames": [
                {"name": name, "lower_inclusive": lower, "upper_exclusive": upper}
                for name, lower, upper in DURATION_BINS
            ],
            "allocation": "proportional to exact candidate train-ready native frames",
            "selection": (
                "exact nearest whole-video subset sum independently per stratum"
            ),
            "tie_break": (
                "salted-SHA candidate traversal; first reach for equal frame sum; "
                "salted canonical subset SHA for equal-distance sums"
            ),
        },
        "candidate_pool": {
            "videos": len(candidate_videos),
            "native_frames": candidate_frames,
            "hours_fraction": {
                "numerator": candidate_frames,
                "denominator": GRID_HZ * 60 * 60,
            },
        },
        "strata": stratum_receipts,
        "reserve": {
            "videos": len(reserve_list),
            "native_frames": reserve_frames,
            "hours_fraction": {
                "numerator": reserve_frames,
                "denominator": GRID_HZ * 60 * 60,
            },
            "phase0_rows": reserve_phase0,
            "base_windows": reserve_windows,
            "target_delta_native_frames": reserve_frames - TARGET_FRAMES,
            "largest_candidate_native_frames": largest_candidate,
            "video_ids": reserve_list,
            "newline_list_sha256": hashlib.sha256(
                reserve_text.encode("utf-8")
            ).hexdigest(),
        },
        "retained_nitrogen_train": {
            "videos": len(retained_list),
            "sessions": len(retained_sessions),
            "native_frames": retained_native,
            "hours_fraction": {
                "numerator": retained_native,
                "denominator": GRID_HZ * 60 * 60,
            },
            "phase0_rows": retained_phase0,
            "base_windows": retained_windows,
            "video_ids": retained_list,
            "newline_list_sha256": hashlib.sha256(
                retained_text.encode("utf-8")
            ).hexdigest(),
        },
        "retained_max_train_with_frozen_wild": {
            "wild_native_rows": WILD_NATIVE_ROWS,
            "wild_phase0_rows": WILD_PHASE0_ROWS,
            "wild_base_windows": WILD_BASE_WINDOWS,
            "native_rows": max_native,
            "hours_fraction": {
                "numerator": max_native,
                "denominator": GRID_HZ * 60 * 60,
            },
            "phase0_rows": max_phase0,
            "base_windows": max_windows,
            "global_batch": GLOBAL_BATCH,
            "steps_per_epoch": steps_per_epoch,
            "padding_windows_per_epoch": steps_per_epoch * GLOBAL_BATCH - max_windows,
            "epochs": EPOCHS,
            "optimizer_steps": steps_per_epoch * EPOCHS,
        },
        "proofs": {
            "all_valid_videos": len(all_videos),
            "unflagged_videos": len(unflagged_videos),
            "candidate_is_all_valid_minus_unflagged": candidate_videos
            == all_videos.difference(unflagged_videos),
            "unflagged_preserved_in_retained": unflagged_videos.issubset(retained),
            "reserve_unflagged_overlap_empty": not reserve.intersection(
                unflagged_videos
            ),
            "reserve_retained_overlap_empty": not reserve.intersection(retained),
            "partition_covers_all_valid": reserve.union(retained) == all_videos,
            "whole_y4n_absent": Y4N_VIDEO_ID not in reserve.union(retained),
            "local_rec_sessions_absent": not any(
                session_id.startswith("rec_") for session_id in all_sessions
            ),
            "reserve_non_nitrogen_tier_b_id_overlap_empty": not reserve.intersection(
                non_nitrogen_ids
            ),
            "duplicate_video_ids_absent": len(reserve_list) == len(reserve)
            and len(retained_list) == len(retained),
            "reserve_target_within_one_largest_candidate": abs(
                reserve_frames - TARGET_FRAMES
            )
            <= largest_candidate,
        },
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    return reserve_list, retained_list, receipt


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-valid-run-meta", type=Path, required=True)
    parser.add_argument("--all-valid-run-meta-sha256", required=True)
    parser.add_argument("--unflagged-run-meta", type=Path, required=True)
    parser.add_argument("--unflagged-run-meta-sha256", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contracts = {
        "all_valid_run_meta": _input_contract(
            args.all_valid_run_meta, args.all_valid_run_meta_sha256
        ),
        "unflagged_run_meta": _input_contract(
            args.unflagged_run_meta, args.unflagged_run_meta_sha256
        ),
        "inventory": _input_contract(args.inventory, args.inventory_sha256),
    }
    reserve, retained, receipt = select_reserve(
        all_valid_meta=args.all_valid_run_meta,
        unflagged_meta=args.unflagged_run_meta,
        inventory_path=args.inventory,
        input_contracts=contracts,
    )
    reserve_text = "".join(f"{video_id}\n" for video_id in reserve)
    retained_text = "".join(f"{video_id}\n" for video_id in retained)
    atomic_text(args.output_dir / "reserve_video_ids.txt", reserve_text)
    atomic_text(
        args.output_dir / "retained_max_training_video_ids.txt", retained_text
    )
    atomic_json(args.output_dir / "reserve_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
