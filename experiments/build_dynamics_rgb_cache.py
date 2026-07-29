#!/usr/bin/env python3
"""Build the label-free, arm-independent RGB tuple cache for Arms B/C/D.

The production command accepts one *explicit* inventory.  It never discovers
sessions by walking a data directory, and it validates the complete train-only
population before opening a source.  NitroGen rows point at raw videos and
already-validated session ranges; corrected own-v3 and provisional-wild rows
point at masked RGB NPZ shards.  NPZ action members are never requested.

Each sampled anchor stores the compact consecutive pixel span ``t-1`` through
``t+max(h)`` once.  ``index.npz`` then maps every matched ``(anchor, h)`` tuple
to global rows in the flattened ``rgb.npy`` memmap.  This lets all three arms
consume exactly the same tuple IDs without materializing the complete RGB
corpus.

Production consumes the exact output of
``experiments/build_dynamics_pretraining_inventory.py``
(``madeleine.dynamics-pretraining-inventory.v1``).  Its ``sessions`` rows are
joined to ``nitrogen_videos`` rows in memory; no second production inventory
format is accepted.

Outputs are assembled under ``OUTPUT.partial`` and atomically renamed only
after both artifacts and their hashes validate.  A matching partial build is
resumed at a whole-window boundary; a mismatched partial build is rejected.

The authorized production invocation is::

    python experiments/build_dynamics_rgb_cache.py \
      --inventory /ephemeral/data/dynamics_pretraining_inventory.json \
      --output /ephemeral/data/dynamics_rgb_cache_c_d_v1 \
      --horizons 1,2,4 --window-count 60000 --seed 2026072801

The source-independent verification invocation is::

    python experiments/build_dynamics_rgb_cache.py \
      --validate-only --output /ephemeral/data/dynamics_rgb_cache_c_d_v1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence
import zipfile

import cv2
import numpy as np
from numpy.lib.format import open_memmap, read_array_header_1_0, read_array_header_2_0

from data.precompute_features import (
    FRAME_SIZE,
    MAX_SEQUENTIAL_GAP,
    _decode_resampled_part,
)


INVENTORY_SCHEMA = "madeleine.dynamics-pretraining-inventory.v1"
CACHE_SCHEMA = "madeleine.dynamics-rgb-cache.v1"
STATE_SCHEMA = "madeleine.dynamics-rgb-cache-state.v1"
STUDY_ID = "photon_inspired_celeste_dynamics_exploratory_cd_s0_v1"
PRODUCTION_HORIZONS = (1, 2, 4)
PRODUCTION_WINDOWS = 60_000
PRODUCTION_SEED = 2026072801
SOURCE_NAMES = ("nitrogen", "wild_provisional", "local")
SOURCE_TO_ID = {name: index for index, name in enumerate(SOURCE_NAMES)}
FORBIDDEN_IDS = (
    "y4nQHqYSObI",
    "rec_20260724_171305_5min",  # own-v3 val-A
    "rec_20260725_025853",       # val-B
    "rec_20260725_160450_b1",    # B1
    "rec_20260727_220000_test",  # sealed untouched test
)
OWN_TRAIN_IDS = (
    "rec_20260724_190233",
    "rec_20260725_015612",
    "rec_20260725_021338",
)
WILD_VIDEO_IDS = (
    "6vEpVqbrvSE",
    "Y6AeZFCU4LY",
    "kdQbIoMxzZw",
    "nRMVyWdNsTo",
    "ofy37Fm6EgI",
    "v1068970940",
    "v498642684",
)
PRODUCTION_POPULATION = {
    "nitrogen": {"videos": 210, "sessions": 1538, "frames": 32_037_600},
    "wild_provisional": {"videos": 7, "sessions": 2058, "frames": 4_835_638},
    "local": {"videos": 3, "sessions": 3, "frames": 143_451},
}
INDEX_DTYPES = {
    "tuple_id": np.dtype("uint64"),
    "window_id": np.dtype("int64"),
    "source_id": np.dtype("uint8"),
    "session_index": np.dtype("int32"),
    "run_id": np.dtype("int32"),
    "anchor_engine_frame": np.dtype("int64"),
    "online_previous": np.dtype("int64"),
    "online_current": np.dtype("int64"),
    "target_previous": np.dtype("int64"),
    "target_current": np.dtype("int64"),
    "horizon": np.dtype("int16"),
    "motion_score": np.dtype("float32"),
    "stratum": np.dtype("uint8"),
}


@dataclass(frozen=True)
class Run:
    row_index: int
    run_id: int
    local_start: int
    local_stop: int
    engine_start: int

    @property
    def length(self) -> int:
        return self.local_stop - self.local_start


@dataclass(frozen=True)
class Window:
    window_id: int
    row_index: int
    run_id: int
    anchor_local: int
    anchor_engine: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_forbidden_identity(value: str) -> None:
    folded = value.casefold()
    normalized = folded.replace("\\", "/")
    components = [component for component in normalized.split("/") if component]
    if (
        "untouched" in folded
        or "val-b" in folded
        or "val_b" in folded
        or any(
            component == "b1"
            or component.startswith("b1_")
            or component.endswith("_b1")
            for component in components
        )
    ):
        raise ValueError(f"forbidden evaluation identity: {value}")
    for forbidden in FORBIDDEN_IDS:
        if forbidden.casefold() in folded:
            raise ValueError(f"forbidden evaluation identity {forbidden}: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _row_path(row: Mapping[str, Any]) -> Path:
    name = "raw_video_path" if row.get("source") == "nitrogen" else "path"
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{row.get('session_id')}: missing {name}")
    return Path(value)


def _row_digest(row: Mapping[str, Any]) -> str:
    name = "raw_video_sha256" if row.get("source") == "nitrogen" else "sha256"
    value = row.get(name)
    if not _valid_digest(value):
        raise ValueError(f"{row.get('session_id')}: invalid {name}")
    return str(value)


def _normalize_inventory_rows(
    inventory: Mapping[str, Any], *, production: bool
) -> list[dict[str, Any]]:
    """Translate the inventory producer's compact references to cache rows."""

    values = inventory.get("sessions")
    # Tests may directly provide already-normalized rows, but the production
    # CLI accepts only the canonical inventory producer's `sessions` field.
    if values is None and not production:
        values = inventory.get("rows")
    if not isinstance(values, list) or not values:
        raise ValueError("RGB inventory needs explicit sessions")
    videos_value = inventory.get("nitrogen_videos", [])
    if not isinstance(videos_value, list):
        raise ValueError("nitrogen_videos must be a list")
    videos: dict[str, Mapping[str, Any]] = {}
    for value in videos_value:
        if not isinstance(value, Mapping):
            raise ValueError("nitrogen video rows must be objects")
        video_id = value.get("video_id")
        if not isinstance(video_id, str) or video_id in videos:
            raise ValueError("duplicate/missing NitroGen video identity")
        _reject_forbidden_identity(video_id)
        video_path = value.get("video_path")
        if not isinstance(video_path, str):
            raise ValueError(f"{video_id}: missing video_path")
        _reject_forbidden_identity(video_path)
        videos[video_id] = value

    rows: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("inventory session rows must be objects")
        # Already-normalized synthetic fixture.
        if "frame_count" in value:
            rows.append(dict(value))
            continue
        source = value.get("source")
        common = {
            "source": source,
            "session_id": value.get("session_id"),
            "frame_count": value.get("frames"),
            "declared_eligible_windows": value.get("eligible_windows"),
            "engine_frame_start": value.get("engine_frame_start"),
            "engine_frame_end_exclusive": value.get("engine_frame_end_exclusive"),
        }
        if source == "nitrogen":
            video_id = value.get("video_id")
            video = videos.get(str(video_id))
            if video is None:
                raise ValueError(f"missing NitroGen video row for {video_id}")
            rows.append(
                {
                    **common,
                    "video_id": video_id,
                    "raw_video_path": video.get("video_path"),
                    "raw_video_sha256": video.get("video_sha256"),
                    "decoder_mode": video.get("decoder_mode"),
                    "source_frame_start": value.get("engine_frame_start"),
                    "source_frame_end": value.get("engine_frame_end_exclusive"),
                    "source_resolution_wh": [
                        video.get("source_width"),
                        video.get("source_height"),
                    ],
                    "source_mask_xyxy": video.get("mask_rect_source_xyxy"),
                    "scaled_mask_xyxy": video.get("mask_rect_128_xyxy"),
                    "reference_shard": value.get("reference_shard"),
                    "reference_shard_sha256": value.get("reference_shard_sha256"),
                }
            )
        elif source in ("wild_provisional", "local"):
            rows.append(
                {
                    **common,
                    "video_id": value.get("video_id", value.get("session_id")),
                    "path": value.get("shard_path"),
                    "sha256": value.get("shard_sha256"),
                    "masked": True,
                }
            )
        else:
            raise ValueError(f"unknown source: {source!r}")
    if production:
        if inventory.get("study_id") != STUDY_ID:
            raise ValueError("exploratory C/D inventory study identity changed")
        used_videos = {
            str(row["video_id"]) for row in rows if row["source"] == "nitrogen"
        }
        if set(videos) != used_videos:
            raise ValueError("NitroGen video/session inventory join is not exact")
    return rows


def _validate_row_without_access(row: Mapping[str, Any]) -> None:
    source = row.get("source")
    if source not in SOURCE_NAMES:
        raise ValueError(f"unknown source: {source!r}")
    session_id = row.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("inventory row lacks a session_id")
    _reject_forbidden_identity(session_id)
    video_id = row.get("video_id", session_id)
    if not isinstance(video_id, str) or not video_id:
        raise ValueError(f"{session_id}: invalid video_id")
    _reject_forbidden_identity(video_id)
    path = _row_path(row)
    _reject_forbidden_identity(str(path))
    _row_digest(row)
    frame_count = row.get("frame_count")
    if not isinstance(frame_count, int) or frame_count < 2:
        raise ValueError(f"{session_id}: frame_count must be >=2")
    if source == "nitrogen":
        if row.get("decoder_mode") not in (
            "opencv_native_60hz",
            "ffmpeg_timestamp_resample_60hz",
        ):
            raise ValueError(f"{session_id}: invalid NitroGen decoder_mode")
        start = row.get("source_frame_start")
        stop = row.get("source_frame_end")
        if not isinstance(start, int) or not isinstance(stop, int):
            raise ValueError(f"{session_id}: invalid source frame range")
        if start < 0 or stop - start != frame_count:
            raise ValueError(f"{session_id}: source frame range/count mismatch")
        resolution = row.get("source_resolution_wh")
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or any(not isinstance(item, int) or item < 1 for item in resolution)
        ):
            raise ValueError(f"{session_id}: invalid source_resolution_wh")
        _mask_rectangles(row)
    elif row.get("masked") is not True:
        raise ValueError(f"{session_id}: NPZ pixels are not declared masked")


def _mask_rectangles(
    row: Mapping[str, Any],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    width, height = map(int, row["source_resolution_wh"])
    source_value = row.get("source_mask_xyxy")
    if source_value is None and row.get("source_mask_xywh") is not None:
        x, y, w, h = map(int, row["source_mask_xywh"])
        source_value = [x, y, x + w, y + h]
    if not isinstance(source_value, list) or len(source_value) != 4:
        raise ValueError(f"{row.get('session_id')}: missing source mask")
    source = tuple(map(int, source_value))
    x0, y0, x1, y1 = source
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"{row.get('session_id')}: source mask is out of bounds")
    scaled_value = row.get("scaled_mask_xyxy")
    if scaled_value is None:
        scaled = (
            max(0, int(x0 / width * FRAME_SIZE) - 1),
            max(0, int(y0 / height * FRAME_SIZE) - 1),
            min(FRAME_SIZE, int(math.ceil(x1 / width * FRAME_SIZE)) + 1),
            min(FRAME_SIZE, int(math.ceil(y1 / height * FRAME_SIZE)) + 1),
        )
    else:
        if not isinstance(scaled_value, list) or len(scaled_value) != 4:
            raise ValueError(f"{row.get('session_id')}: invalid scaled mask")
        scaled = tuple(map(int, scaled_value))
    sx0, sy0, sx1, sy1 = scaled
    if not (0 <= sx0 < sx1 <= FRAME_SIZE and 0 <= sy0 < sy1 <= FRAME_SIZE):
        raise ValueError(f"{row.get('session_id')}: scaled mask is out of bounds")
    return source, scaled


def validate_inventory(
    inventory: Mapping[str, Any], *, production: bool = True
) -> list[dict[str, Any]]:
    """Validate all identities and population counts before source access."""

    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("RGB inventory schema changed")
    values = _normalize_inventory_rows(inventory, production=production)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("inventory rows must be objects")
        _validate_row_without_access(value)
        session_id = str(value["session_id"])
        if session_id in seen:
            raise ValueError(f"duplicate session ID: {session_id}")
        seen.add(session_id)
        rows.append(dict(value))

    rows.sort(
        key=lambda row: (
            SOURCE_TO_ID[str(row["source"])],
            str(row.get("video_id", row["session_id"])),
            int(row.get("source_frame_start", 0)),
            str(row["session_id"]),
        )
    )
    if production:
        if inventory.get("labels_consumed") is not False:
            raise ValueError("inventory does not prove labels were unconsumed")
        proof = inventory.get("forbidden_exclusion_proof")
        required_proof = {
            "sealed_untouched_absent",
            "whole_y4n_absent",
            "own_val_a_absent",
            "B1_absent",
            "val_B_absent",
            "checked_before_cache_RGB_access",
        }
        if not isinstance(proof, Mapping) or any(
            proof.get(name) is not True for name in required_proof
        ):
            raise ValueError("inventory forbidden-exclusion proof is incomplete")
        expected_content = inventory.get("inventory_content_sha256")
        content = dict(inventory)
        content.pop("inventory_content_sha256", None)
        if expected_content != _canonical_sha256(content):
            raise ValueError("inventory canonical content hash changed")

        observed: dict[str, dict[str, Any]] = {}
        for source in SOURCE_NAMES:
            source_rows = [row for row in rows if row["source"] == source]
            videos = {
                str(row.get("video_id", row["session_id"])) for row in source_rows
            }
            observed[source] = {
                "videos": len(videos),
                "sessions": len(source_rows),
                "frames": sum(int(row["frame_count"]) for row in source_rows),
            }
            if observed[source] != PRODUCTION_POPULATION[source]:
                raise ValueError(
                    f"{source} production population changed: {observed[source]}"
                )
        wild_videos = {
            str(row["video_id"])
            for row in rows
            if row["source"] == "wild_provisional"
        }
        if wild_videos != set(WILD_VIDEO_IDS):
            raise ValueError("provisional-wild seven-video membership changed")
        own_sessions = {
            str(row["session_id"]) for row in rows if row["source"] == "local"
        }
        if own_sessions != set(OWN_TRAIN_IDS):
            raise ValueError("corrected own-v3 train membership changed")
    return rows


def _validate_receipt_and_source_hashes(
    inventory: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, production: bool
) -> None:
    expected: dict[Path, str] = {}
    if production:
        contract = inventory.get("contract")
        if not isinstance(contract, Mapping):
            raise ValueError("inventory lacks its exploratory contract receipt")
        contract_path = contract.get("path")
        contract_sha = contract.get("sha256")
        if not isinstance(contract_path, str) or not _valid_digest(contract_sha):
            raise ValueError("inventory exploratory contract receipt is malformed")
        _reject_forbidden_identity(contract_path)
        expected[Path(contract_path)] = str(contract_sha)
    for row in rows:
        path = _row_path(row)
        digest = _row_digest(row)
        previous = expected.setdefault(path, digest)
        if previous != digest:
            raise ValueError(f"one source path has conflicting hashes: {path}")
        if row["source"] == "nitrogen" and production:
            reference = row.get("reference_shard")
            reference_sha = row.get("reference_shard_sha256")
            if not isinstance(reference, str) or not _valid_digest(reference_sha):
                raise ValueError(f"{row['session_id']}: malformed reference shard")
            _reject_forbidden_identity(reference)
            reference_path = Path(reference)
            previous_reference = expected.setdefault(reference_path, str(reference_sha))
            if previous_reference != reference_sha:
                raise ValueError("NitroGen reference-shard hash conflict")
    for path, digest in sorted(expected.items(), key=lambda item: str(item[0])):
        if not path.is_file():
            raise FileNotFoundError(f"explicit RGB source is absent: {path}")
        if sha256_file(path) != digest:
            raise ValueError(f"explicit RGB source SHA-256 changed: {path}")


def _npy_member_header(path: Path, member: str) -> tuple[tuple[int, ...], np.dtype]:
    """Read only an NPY header inside an NPZ, never its action member."""

    try:
        with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _fortran, dtype = read_array_header_1_0(stream)
            elif version in ((2, 0), (3, 0)):
                shape, _fortran, dtype = read_array_header_2_0(stream)
            else:
                raise ValueError(f"unsupported NPY version {version}: {path}")
    except KeyError as error:
        raise ValueError(f"{path}: missing {member}") from error
    return tuple(map(int, shape)), np.dtype(dtype)


def _npz_runs(row: Mapping[str, Any], row_index: int, max_horizon: int) -> list[Run]:
    path = _row_path(row)
    shape, dtype = _npy_member_header(path, "frames.npy")
    expected_shape = (int(row["frame_count"]), FRAME_SIZE, FRAME_SIZE, 3)
    if shape != expected_shape or dtype != np.dtype("uint8"):
        raise ValueError(f"{path}: frames header is not uint8 {expected_shape}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            engine = np.asarray(archive["engine_frame_idx"])
            active = np.asarray(archive["input_active"])
            stored = np.asarray(archive["session_id"])
    except KeyError as error:
        raise ValueError(f"{path}: missing label-free boundary metadata") from error
    count = int(row["frame_count"])
    if engine.dtype != np.int64 or engine.shape != (count,):
        raise ValueError(f"{path}: engine_frame_idx must be int64 [N]")
    if active.dtype != np.uint8 or active.shape != (count,):
        raise ValueError(f"{path}: input_active must be uint8 [N]")
    if not np.all(np.isin(active, (0, 1))):
        raise ValueError(f"{path}: input_active must be binary")
    if stored.size != 1 or str(stored.reshape(()).item()) != row["session_id"]:
        raise ValueError(f"{path}: stored session identity changed")
    keep = active == 1
    starts = np.flatnonzero(
        keep & np.r_[True, (~keep[:-1]) | (np.diff(engine) != 1)]
    )
    stops = np.flatnonzero(
        keep & np.r_[(~keep[1:]) | (np.diff(engine) != 1), True]
    ) + 1
    result: list[Run] = []
    for run_id, (start, stop) in enumerate(zip(starts.tolist(), stops.tolist())):
        if stop - start >= max_horizon + 2:
            result.append(
                Run(row_index, run_id, start, stop, int(engine[start]))
            )
    return result


def build_runs(
    rows: Sequence[Mapping[str, Any]], max_horizon: int
) -> tuple[list[Run], dict[str, int]]:
    runs: list[Run] = []
    eligible = {source: 0 for source in SOURCE_NAMES}
    for row_index, row in enumerate(rows):
        if row["source"] == "nitrogen":
            count = int(row["frame_count"])
            row_runs = [
                Run(
                    row_index=row_index,
                    run_id=0,
                    local_start=0,
                    local_stop=count,
                    engine_start=int(row["source_frame_start"]),
                )
            ]
        else:
            row_runs = _npz_runs(row, row_index, max_horizon)
        runs.extend(row_runs)
        row_eligible = sum(
            run.length - max_horizon - 1 for run in row_runs
        )
        declared = row.get("declared_eligible_windows")
        if declared is not None and declared != row_eligible:
            raise ValueError(
                f"{row['session_id']}: eligible-window receipt changed "
                f"({declared} != {row_eligible})"
            )
        eligible[str(row["source"])] += row_eligible
    if any(value <= 0 for value in eligible.values()):
        raise ValueError(f"every source needs eligible common anchors: {eligible}")
    return runs, eligible


def proportional_allocation(eligible: Mapping[str, int], total: int) -> dict[str, int]:
    if total < len(eligible):
        raise ValueError("window count must permit every source")
    population = sum(eligible.values())
    if total > population:
        raise ValueError("window count exceeds eligible anchor population")
    exact = {name: total * count / population for name, count in eligible.items()}
    allocated = {name: int(math.floor(value)) for name, value in exact.items()}
    for name in sorted(
        eligible, key=lambda item: (-(exact[item] - allocated[item]), item)
    )[: total - sum(allocated.values())]:
        allocated[name] += 1
    # Tiny synthetic inputs can round a source to zero. Preserve the declared
    # all-source experiment by borrowing from the largest allocation.
    for name in SOURCE_NAMES:
        if allocated[name] == 0:
            donor = max(SOURCE_NAMES, key=lambda item: (allocated[item], item))
            if allocated[donor] <= 1:
                raise ValueError("cannot allocate at least one window per source")
            allocated[donor] -= 1
            allocated[name] = 1
    if any(allocated[name] > eligible[name] for name in eligible):
        raise ValueError("source allocation exceeds eligibility")
    return allocated


def _systematic_positions(population: int, count: int, seed: int) -> np.ndarray:
    if not 0 < count <= population:
        raise ValueError("invalid systematic sample size")
    generator = np.random.Generator(np.random.PCG64(seed))
    phase = float(generator.random())
    positions = np.floor((np.arange(count, dtype=np.float64) + phase) * population / count)
    result = positions.astype(np.int64)
    if len(np.unique(result)) != count or result[0] < 0 or result[-1] >= population:
        raise AssertionError("systematic sampler did not produce unique valid rows")
    return result


def build_windows(
    rows: Sequence[Mapping[str, Any]],
    runs: Sequence[Run],
    eligible: Mapping[str, int],
    allocated: Mapping[str, int],
    *,
    max_horizon: int,
    seed: int,
) -> list[Window]:
    result: list[Window] = []
    window_id = 0
    for source_id, source in enumerate(SOURCE_NAMES):
        source_runs = [run for run in runs if rows[run.row_index]["source"] == source]
        lengths = np.asarray(
            [run.length - max_horizon - 1 for run in source_runs], dtype=np.int64
        )
        cumulative = np.cumsum(lengths)
        positions = _systematic_positions(
            int(cumulative[-1]), int(allocated[source]), seed + 104729 * source_id
        )
        run_indices = np.searchsorted(cumulative, positions, side="right")
        prior = np.r_[0, cumulative[:-1]]
        for position, run_index in zip(positions.tolist(), run_indices.tolist()):
            run = source_runs[run_index]
            offset = int(position - prior[run_index])
            anchor_local = run.local_start + 1 + offset
            anchor_engine = run.engine_start + anchor_local - run.local_start
            result.append(
                Window(window_id, run.row_index, run.run_id, anchor_local, anchor_engine)
            )
            window_id += 1
    if len(result) != sum(allocated.values()):
        raise AssertionError("window allocation mismatch")
    return result


def _plan_sha256(
    inventory_sha256: str,
    horizons: Sequence[int],
    windows: Sequence[Window],
) -> str:
    digest = hashlib.sha256()
    digest.update(inventory_sha256.encode("ascii"))
    digest.update(struct.pack("<I", len(horizons)))
    for horizon in horizons:
        digest.update(struct.pack("<H", horizon))
    for window in windows:
        digest.update(
            struct.pack(
                "<Qiiqq",
                window.window_id,
                window.row_index,
                window.run_id,
                window.anchor_local,
                window.anchor_engine,
            )
        )
    return digest.hexdigest()


def _scaled_mask_is_black(frames: np.ndarray, rect: Sequence[int]) -> bool:
    x0, y0, x1, y1 = map(int, rect)
    return int(frames[:, y0:y1, x0:x1].max(initial=0)) == 0


def _decode_native_intervals(
    row: Mapping[str, Any], intervals: Sequence[tuple[int, int]]
) -> dict[tuple[int, int], np.ndarray]:
    source_mask, scaled_mask = _mask_rectangles(row)
    rx0, ry0, rx1, ry1 = source_mask
    sx0, sy0, sx1, sy1 = scaled_mask
    cap = cv2.VideoCapture(str(_row_path(row)))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open NitroGen video: {_row_path(row)}")
    expected_width, expected_height = map(int, row["source_resolution_wh"])
    if (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != expected_width
        or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != expected_height
    ):
        cap.release()
        raise ValueError(f"{row['video_id']}: raw-video resolution changed")
    cursor = -1
    result: dict[tuple[int, int], np.ndarray] = {}
    try:
        for start, stop in intervals:
            if cursor >= 0 and 0 <= start - cursor <= MAX_SEQUENTIAL_GAP:
                while cursor < start:
                    if not cap.grab():
                        raise RuntimeError(f"decode ended before nominal frame {start}")
                    cursor += 1
            elif cursor != start:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
                    raise RuntimeError(f"failed to seek to nominal frame {start}")
                landed = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                if landed != start:
                    raise RuntimeError(f"seek landed at {landed}, expected {start}")
                cursor = start
            frames = np.empty((stop - start, FRAME_SIZE, FRAME_SIZE, 3), np.uint8)
            for index in range(stop - start):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"decode failed at source frame {cursor}")
                cursor += 1
                frame[ry0:ry1, rx0:rx1] = 0
                small = cv2.resize(
                    frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA
                )
                small[sy0:sy1, sx0:sx1] = 0
                frames[index] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            if not _scaled_mask_is_black(frames, scaled_mask):
                raise AssertionError(f"{row['video_id']}: scaled mask is not black")
            result[(start, stop)] = frames
    finally:
        cap.release()
    return result


def _merge_intervals(values: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(set(values)):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged


def _decode_nitrogen_group(
    rows: Sequence[Mapping[str, Any]],
    group: Sequence[Window],
    span: int,
) -> dict[int, np.ndarray]:
    # A group contains one raw video. Merge selected nominal-time intervals so
    # overlapping windows are decoded once and all source access stays ordered.
    requests: list[tuple[Window, int, int]] = []
    for window in group:
        row = rows[window.row_index]
        start = int(row["source_frame_start"]) + window.anchor_local - 1
        requests.append((window, start, start + span))
    first = rows[group[0].row_index]
    intervals = _merge_intervals([(start, stop) for _, start, stop in requests])
    if first["decoder_mode"] == "opencv_native_60hz":
        decoded = _decode_native_intervals(first, intervals)
    else:
        _source_mask, scaled_mask = _mask_rectangles(first)
        decoded = {}
        for interval in intervals:
            frames, _imputed = _decode_resampled_part(
                _row_path(first),
                start_frame=interval[0],
                end_frame=interval[1],
                mask_xyxy=scaled_mask,
            )
            if not _scaled_mask_is_black(frames, scaled_mask):
                raise AssertionError(f"{first['video_id']}: scaled mask is not black")
            decoded[interval] = frames
    result: dict[int, np.ndarray] = {}
    for window, start, stop in requests:
        container = next(
            interval for interval in intervals if interval[0] <= start and stop <= interval[1]
        )
        result[window.window_id] = decoded[container][
            start - container[0] : stop - container[0]
        ]
    return result


def _load_npz_windows(
    row: Mapping[str, Any], group: Sequence[Window], span: int
) -> dict[int, np.ndarray]:
    # Deliberately request frames and boundary metadata only. `keys` may exist
    # in the archive but is neither decompressed nor indexed by this program.
    with np.load(_row_path(row), allow_pickle=False) as archive:
        frames = np.asarray(archive["frames"])
        engine = np.asarray(archive["engine_frame_idx"])
        active = np.asarray(archive["input_active"])
    result: dict[int, np.ndarray] = {}
    for window in group:
        start = window.anchor_local - 1
        stop = start + span
        if stop > len(frames):
            raise ValueError(f"{row['session_id']}: selected window crosses shard end")
        if not np.all(active[start:stop] == 1):
            raise ValueError(f"{row['session_id']}: selected window crosses inactive row")
        if not np.all(np.diff(engine[start:stop]) == 1):
            raise ValueError(f"{row['session_id']}: selected window crosses engine gap")
        result[window.window_id] = np.asarray(frames[start:stop], dtype=np.uint8)
    return result


def _motion_score(previous: np.ndarray, current: np.ndarray) -> float:
    # Integer BT.601 luma avoids platform-specific floating preprocessing.
    left = (
        77 * previous[..., 0].astype(np.int32)
        + 150 * previous[..., 1].astype(np.int32)
        + 29 * previous[..., 2].astype(np.int32)
    )
    right = (
        77 * current[..., 0].astype(np.int32)
        + 150 * current[..., 1].astype(np.int32)
        + 29 * current[..., 2].astype(np.int32)
    )
    return float(np.abs(right - left).mean(dtype=np.float64) / 256.0)


def _tuple_id(
    inventory_sha256: str, source: str, session_id: str, anchor: int, horizon: int
) -> np.uint64:
    payload = (
        f"{inventory_sha256}\0{source}\0{session_id}\0{anchor}\0{horizon}"
    ).encode("utf-8")
    return np.uint64(int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little"))


def _write_index(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    windows: Sequence[Window],
    horizons: Sequence[int],
    span: int,
    inventory_sha256: str,
    rgb: np.ndarray,
    require_all_motion_cells: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    count = len(windows) * len(horizons)
    arrays = {name: np.empty(count, dtype=dtype) for name, dtype in INDEX_DTYPES.items()}
    session_ids = np.asarray([str(row["session_id"]) for row in rows])
    source_names = np.asarray(SOURCE_NAMES)
    cursor = 0
    seen_tuple_ids: set[int] = set()
    for window in windows:
        row = rows[window.row_index]
        base = window.window_id * span
        for horizon in horizons:
            value = _tuple_id(
                inventory_sha256,
                str(row["source"]),
                str(row["session_id"]),
                window.anchor_engine,
                horizon,
            )
            if int(value) in seen_tuple_ids:
                raise RuntimeError("tuple-ID collision")
            seen_tuple_ids.add(int(value))
            arrays["tuple_id"][cursor] = value
            arrays["window_id"][cursor] = window.window_id
            arrays["source_id"][cursor] = SOURCE_TO_ID[str(row["source"])]
            arrays["session_index"][cursor] = window.row_index
            arrays["run_id"][cursor] = window.run_id
            arrays["anchor_engine_frame"][cursor] = window.anchor_engine
            arrays["online_previous"][cursor] = base
            arrays["online_current"][cursor] = base + 1
            arrays["target_previous"][cursor] = base + horizon
            arrays["target_current"][cursor] = base + horizon + 1
            arrays["horizon"][cursor] = horizon
            arrays["motion_score"][cursor] = _motion_score(
                rgb[base + 1], rgb[base + horizon + 1]
            )
            cursor += 1
    strata: list[dict[str, Any]] = []
    for source_id, source in enumerate(SOURCE_NAMES):
        for horizon in horizons:
            mask = (arrays["source_id"] == source_id) & (arrays["horizon"] == horizon)
            scores = arrays["motion_score"][mask]
            if not len(scores):
                raise ValueError(f"empty source/horizon stratum: {source}/{horizon}")
            threshold = float(np.median(scores))
            arrays["stratum"][mask] = (scores > threshold).astype(np.uint8)
            static = int(np.sum(arrays["stratum"][mask] == 0))
            change = int(np.sum(arrays["stratum"][mask] == 1))
            if require_all_motion_cells and (static == 0 or change == 0):
                raise ValueError(
                    f"empty production motion cell for {source}/h={horizon}: "
                    f"static={static}, change={change}"
                )
            strata.append(
                {
                    "source": source,
                    "horizon": int(horizon),
                    "threshold": threshold,
                    "static": static,
                    "change": change,
                }
            )
    payload = {**arrays, "session_ids": session_ids, "source_names": source_names}
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return payload, strata


def _artifact_receipt(path: Path, *, shape: Sequence[int] | None = None, dtype: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if shape is not None:
        result["shape"] = list(map(int, shape))
    if dtype is not None:
        result["dtype"] = dtype
    return result


def _validate_complete_cache(output: Path) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != CACHE_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("existing cache manifest is not complete")
    labels = manifest.get("labels")
    if not isinstance(labels, Mapping) or labels.get("loaded") is not False or labels.get(
        "arrays_accessed"
    ) != []:
        raise ValueError("existing cache is not provably label-free")
    proof = manifest.get("exclusion_proof")
    proof_fields = (
        "validated_before_source_access",
        "whole_y4n_absent",
        "val_a_absent",
        "val_b_absent",
        "b1_absent",
        "sealed_untouched_absent",
    )
    if not isinstance(proof, Mapping) or any(proof.get(name) is not True for name in proof_fields):
        raise ValueError("existing cache exclusion proof is incomplete")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, Mapping) or inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("existing cache inventory receipt changed")
    expected = {"rgb.npy", "index.npz", "manifest.json"}
    entries = list(output.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("existing cache must contain regular non-symlink files")
    observed = {path.name for path in entries}
    if observed != expected:
        raise ValueError("existing cache inventory changed")
    for role in ("rgb", "index"):
        receipt = manifest["artifacts"][role]
        path = output / receipt["path"]
        if path.stat().st_size != receipt["bytes"] or sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"existing {role} artifact changed")
    rgb_receipt = manifest["artifacts"]["rgb"]
    rgb = np.load(output / "rgb.npy", mmap_mode="r", allow_pickle=False)
    if (
        rgb.dtype != np.uint8
        or rgb.ndim != 4
        or tuple(rgb.shape[1:]) != (FRAME_SIZE, FRAME_SIZE, 3)
        or list(rgb.shape) != rgb_receipt.get("shape")
        or rgb_receipt.get("dtype") != "uint8"
        or rgb_receipt.get("c_order") is not True
        or not rgb.flags.c_contiguous
    ):
        raise ValueError("existing RGB cache shape/dtype/order changed")
    expected_index_names = {*INDEX_DTYPES, "session_ids", "source_names"}
    with np.load(output / "index.npz", allow_pickle=False) as archive:
        if set(archive.files) != expected_index_names:
            raise ValueError("existing tuple-index field inventory changed")
        index = {name: np.asarray(archive[name]) for name in archive.files}
    rows = len(index["tuple_id"])
    if rows != manifest.get("tuples") or rows != manifest["artifacts"]["index"].get("rows"):
        raise ValueError("existing tuple-index row count changed")
    for name, dtype in INDEX_DTYPES.items():
        if index[name].dtype != dtype or index[name].shape != (rows,):
            raise ValueError(f"existing tuple-index {name} schema changed")
    fields = manifest["artifacts"]["index"].get("fields")
    if not isinstance(fields, Mapping) or set(fields) != expected_index_names:
        raise ValueError("existing tuple-index field receipt changed")
    if len(np.unique(index["tuple_id"])) != rows:
        raise ValueError("existing tuple IDs are not unique")
    for name in ("session_ids", "source_names"):
        if index[name].ndim != 1 or index[name].dtype.kind != "U":
            raise ValueError(f"existing tuple-index {name} schema changed")
    for identity in index["session_ids"].tolist():
        _reject_forbidden_identity(str(identity))
    if tuple(index["source_names"].tolist()) != SOURCE_NAMES:
        raise ValueError("existing source-name lookup changed")
    if np.any(index["session_index"] < 0) or np.any(
        index["session_index"] >= len(index["session_ids"])
    ):
        raise ValueError("existing session lookup is out of bounds")
    if np.any(index["source_id"] >= len(index["source_names"])):
        raise ValueError("existing source lookup is out of bounds")
    if not np.array_equal(index["online_current"], index["online_previous"] + 1):
        raise ValueError("existing online pair geometry changed")
    if not np.array_equal(index["target_current"], index["target_previous"] + 1):
        raise ValueError("existing target pair geometry changed")
    horizon = index["horizon"].astype(np.int64)
    if not np.array_equal(index["target_previous"] - index["online_previous"], horizon):
        raise ValueError("existing target horizon geometry changed")
    frame_columns = (
        "online_previous",
        "online_current",
        "target_previous",
        "target_current",
    )
    if rows and (
        min(int(index[name].min()) for name in frame_columns) < 0
        or max(int(index[name].max()) for name in frame_columns) >= len(rgb)
    ):
        raise ValueError("existing tuple frame index is out of bounds")
    if not np.isfinite(index["motion_score"]).all() or np.any(index["motion_score"] < 0):
        raise ValueError("existing motion scores are invalid")
    if set(np.unique(index["stratum"]).tolist()) != {0, 1}:
        raise ValueError("existing cache lacks both motion strata")
    if sorted(np.unique(index["horizon"]).tolist()) != sorted(manifest["horizons"]):
        raise ValueError("existing index horizons changed")
    return manifest


def validate_cache(output: Path) -> dict[str, Any]:
    """Validate a published cache and return a content-bound CLI receipt."""

    output = Path(output)
    _reject_forbidden_identity(str(output))
    manifest = _validate_complete_cache(output)
    return {
        "schema_version": CACHE_SCHEMA,
        "status": "valid",
        "cache": str(output.resolve()),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "rgb_sha256": manifest["artifacts"]["rgb"]["sha256"],
        "index_sha256": manifest["artifacts"]["index"]["sha256"],
        "windows": manifest["windows"],
        "tuples": manifest["tuples"],
    }


def build_cache(
    *,
    inventory_path: Path,
    output: Path,
    horizons: Sequence[int],
    window_count: int,
    seed: int,
    production: bool = True,
) -> dict[str, Any]:
    horizons = tuple(sorted(set(map(int, horizons))))
    if not horizons or horizons[0] < 1 or horizons[-1] > 32:
        raise ValueError("horizons must be unique positive native-frame offsets <=32")
    if window_count < len(SOURCE_NAMES):
        raise ValueError("window_count is too small")
    if production and (
        horizons != PRODUCTION_HORIZONS
        or window_count != PRODUCTION_WINDOWS
        or seed != PRODUCTION_SEED
    ):
        raise ValueError(
            "production cache must use frozen horizons=(1,2,4), "
            "window_count=60000, seed=2026072801"
        )
    _reject_forbidden_identity(str(inventory_path))
    _reject_forbidden_identity(str(output))
    if output.exists():
        manifest = _validate_complete_cache(output)
        if (
            manifest["horizons"] != list(horizons)
            or manifest["windows"] != window_count
            or manifest["sampling"]["seed"] != seed
            or manifest["inventory"]["sha256"] != sha256_file(inventory_path)
        ):
            raise ValueError("existing cache was built under another contract")
        return manifest

    inventory_sha256 = sha256_file(inventory_path)
    inventory = _load_json(inventory_path)
    if production and inventory.get("horizons_native_frames") != list(horizons):
        raise ValueError("cache horizons differ from the authorized inventory")
    rows = validate_inventory(inventory, production=production)
    # All identities and population counts have now passed. Only now may any
    # receipt/source path be opened.
    _validate_receipt_and_source_hashes(inventory, rows, production=production)
    max_horizon = max(horizons)
    span = max_horizon + 2
    runs, eligible = build_runs(rows, max_horizon)
    if production:
        summary = inventory.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError("inventory source summary is absent")
        for source in SOURCE_NAMES:
            source_summary = summary.get(source)
            if not isinstance(source_summary, Mapping) or source_summary.get(
                "eligible_windows"
            ) != eligible[source]:
                raise ValueError(f"{source} eligible-window summary changed")
    allocated = proportional_allocation(eligible, window_count)
    windows = build_windows(
        rows,
        runs,
        eligible,
        allocated,
        max_horizon=max_horizon,
        seed=seed,
    )
    plan_sha256 = _plan_sha256(inventory_sha256, horizons, windows)

    staging = output.with_name(output.name + ".partial")
    staging.mkdir(parents=True, exist_ok=True)
    state_path = staging / "state.json"
    # A crash after final-manifest publication but before the directory rename
    # leaves a complete staging directory. Validate and publish it directly.
    if (staging / "manifest.json").is_file() and not state_path.exists():
        staged_manifest = _validate_complete_cache(staging)
        if (
            staged_manifest["inventory"]["sha256"] != inventory_sha256
            or staged_manifest["horizons"] != list(horizons)
            or staged_manifest["windows"] != window_count
            or staged_manifest["sampling"]["seed"] != seed
        ):
            raise ValueError("completed partial cache belongs to another contract")
        os.replace(staging, output)
        return _validate_complete_cache(output)
    state = {
        "schema_version": STATE_SCHEMA,
        "inventory_sha256": inventory_sha256,
        "plan_sha256": plan_sha256,
        "horizons": list(horizons),
        "span": span,
        "windows": window_count,
        "completed_windows": 0,
    }
    rgb_path = staging / "rgb.npy"
    if state_path.exists():
        existing = _load_json(state_path)
        contract_fields = (
            "schema_version", "inventory_sha256", "plan_sha256", "horizons", "span", "windows"
        )
        if any(existing.get(name) != state[name] for name in contract_fields):
            raise ValueError("partial cache belongs to another build contract")
        completed = int(existing.get("completed_windows", -1))
        if not 0 <= completed <= window_count:
            raise ValueError("partial cache progress is invalid")
        rgb = open_memmap(rgb_path, mode="r+", dtype=np.uint8)
        expected_shape = (window_count * span, FRAME_SIZE, FRAME_SIZE, 3)
        if rgb.shape != expected_shape:
            raise ValueError("partial RGB memmap shape changed")
        state = existing
    else:
        entries = {path.name for path in staging.iterdir()}
        if entries not in (set(), {"rgb.npy"}):
            raise ValueError("unrecognized files in partial cache directory")
        expected_shape = (window_count * span, FRAME_SIZE, FRAME_SIZE, 3)
        if entries == {"rgb.npy"}:
            # Recognized crash point: memmap creation completed before the
            # initial state receipt. Reuse only an exact header match.
            rgb = open_memmap(rgb_path, mode="r+", dtype=np.uint8)
            if rgb.shape != expected_shape:
                raise ValueError("orphan partial RGB memmap shape changed")
        else:
            rgb = open_memmap(
                rgb_path,
                mode="w+",
                dtype=np.uint8,
                shape=expected_shape,
            )
        completed = 0
        _atomic_json(state_path, state)

    remaining = [window for window in windows if window.window_id >= completed]
    groups: list[list[Window]] = []
    for window in remaining:
        row = rows[window.row_index]
        key = (
            row["source"],
            str(_row_path(row)) if row["source"] == "nitrogen" else window.row_index,
        )
        if not groups:
            groups.append([window])
        else:
            prior = rows[groups[-1][0].row_index]
            prior_key = (
                prior["source"],
                str(_row_path(prior)) if prior["source"] == "nitrogen" else groups[-1][0].row_index,
            )
            if key == prior_key:
                groups[-1].append(window)
            else:
                groups.append([window])

    for group in groups:
        row = rows[group[0].row_index]
        if row["source"] == "nitrogen":
            decoded = _decode_nitrogen_group(rows, group, span)
        else:
            decoded = _load_npz_windows(row, group, span)
        for window in group:
            frames = decoded[window.window_id]
            if frames.dtype != np.uint8 or frames.shape != (
                span,
                FRAME_SIZE,
                FRAME_SIZE,
                3,
            ):
                raise ValueError(f"decoded window {window.window_id} has wrong shape")
            base = window.window_id * span
            rgb[base : base + span] = frames
        rgb.flush()
        completed = group[-1].window_id + 1
        state["completed_windows"] = completed
        _atomic_json(state_path, state)

    if completed != window_count:
        raise AssertionError("cache did not finish every window")
    rgb.flush()
    index_path = staging / "index.npz"
    arrays, strata = _write_index(
        index_path,
        rows=rows,
        windows=windows,
        horizons=horizons,
        span=span,
        inventory_sha256=inventory_sha256,
        rgb=rgb,
        require_all_motion_cells=production,
    )
    rgb_shape = tuple(map(int, rgb.shape))
    del rgb

    source_rows = []
    total_eligible = sum(eligible.values())
    for source in SOURCE_NAMES:
        source_rows.append(
            {
                "name": source,
                "source_id": SOURCE_TO_ID[source],
                "eligible_common_anchors": int(eligible[source]),
                "allocated_windows": int(allocated[source]),
                "eligible_probability": eligible[source] / total_eligible,
                "allocated_probability": allocated[source] / window_count,
                "sessions": sum(row["source"] == source for row in rows),
            }
        )
    manifest = {
        "schema_version": CACHE_SCHEMA,
        "status": "complete",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "inventory": {
            "path": str(inventory_path.resolve()),
            "sha256": inventory_sha256,
            "schema_version": INVENTORY_SCHEMA,
            "rows": len(rows),
        },
        "horizons": list(horizons),
        "window_span_frames": span,
        "window_layout": "flat global RGB rows; reshape to [window,offset,128,128,3]",
        "windows": window_count,
        "tuples": window_count * len(horizons),
        "sources": source_rows,
        "motion_strata": strata,
        "sampling": {
            "seed": seed,
            "method": "deterministic source-proportional systematic common-anchor sampling",
            "source_allocation": "largest remainder, then at-least-one correction",
            "source_order": list(SOURCE_NAMES),
        },
        "labels": {
            "loaded": False,
            "arrays_accessed": [],
            "boundary_metadata_accessed": ["engine_frame_idx", "input_active", "session_id"],
        },
        "exclusion_proof": {
            "validated_before_source_access": True,
            "whole_y4n_absent": True,
            "val_a_absent": True,
            "val_b_absent": True,
            "b1_absent": True,
            "sealed_untouched_absent": True,
            "forbidden_ids": list(FORBIDDEN_IDS),
            "production_population_validated": production,
        },
        "artifacts": {
            "rgb": {
                **_artifact_receipt(
                    rgb_path, shape=rgb_shape, dtype="uint8"
                ),
                "c_order": True,
            },
            "index": {
                **_artifact_receipt(index_path),
                "rows": len(arrays["tuple_id"]),
                "fields": {
                    **{name: str(dtype) for name, dtype in INDEX_DTYPES.items()},
                    "session_ids": str(arrays["session_ids"].dtype),
                    "source_names": str(arrays["source_names"].dtype),
                },
            },
        },
        "resume": {
            "plan_sha256": plan_sha256,
            "atomic_directory_publish": True,
            "completed_windows": completed,
        },
    }
    manifest_path = staging / "manifest.json"
    _atomic_json(manifest_path, manifest)
    state_path.unlink()
    os.replace(staging, output)
    return _validate_complete_cache(output)


def _parse_horizons(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("horizons must be comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("horizons cannot be empty")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=_parse_horizons, default=(1, 2, 4))
    parser.add_argument("--window-count", type=int)
    parser.add_argument("--seed", type=int, default=2026072801)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="rehash and validate an already-published cache without source access",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.validate_only:
        if args.inventory is not None or args.window_count is not None:
            parser.error("--validate-only accepts only --output")
        print(json.dumps(validate_cache(args.output), indent=2, sort_keys=True))
        return 0
    if args.inventory is None or args.window_count is None:
        parser.error("building requires --inventory and --window-count")
    manifest = build_cache(
        inventory_path=args.inventory,
        output=args.output,
        horizons=args.horizons,
        window_count=args.window_count,
        seed=args.seed,
        production=True,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
