"""Prepare the frozen broad-seven wild corpus for one matched GRU run.

This utility is intentionally preparation-only.  It never starts feature
extraction or training.  ``plan`` verifies the immutable provisional RGB
corpus and emits two deterministic input lists for independent invocations of
``data.precompute_features shards``.  ``assemble`` verifies both feature
outputs, adds the exact mapped-y4n development shards used by the reference
GRU, and publishes one hard-linked data view with a frozen config and split
receipt.

The source remains explicitly provisional throughout.  Passing these checks
does not promote any wild frame to clean or admitted training data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

import numpy as np

from data.schema import KEY_ORDER


DEFAULT_MANIFEST = (
    ROOT / "results/wild20/provisional-broad7-af52cee/aggregate.json"
)
DEFAULT_CONFIG_TEMPLATE = (
    ROOT
    / "experiments/configs/takeover_features_26m_128x3frame_full_holdout.json"
)

CORPUS_FORMAT = "madeleine.wild-provisional-corpus.v1"
SOURCE_LABEL_KIND = "wild_overlay_provisional"
ADMISSION_TIER = "provisional_not_train_ready"
FEATURE_FORMAT = "resnet18_imagenet_avgpool_float16_v1"
FEATURE_DIM = 512
SOURCE_ARRAY_MEMBERS = {
    "frames.npy",
    "keys.npy",
    "engine_frame_idx.npy",
    "pts_s.npy",
    "input_active.npy",
    "session_id.npy",
}
FEATURE_ARRAY_NAMES = {
    "features",
    "keys",
    "engine_frame_idx",
    "input_active",
    "session_id",
}
GRU_TRAINABLE_PARAMETERS = 25_719_815

Y4N_SHARD_SHA256 = (
    ("y4nQHqYSObI__r000", "492d5cace2558aded83220e339e3e5155838f415251a050b6c0388a8e7c55953"),
    ("y4nQHqYSObI__r001", "e32e360506784169b85fd7e7bceec78c8348afec4cd1bbca70b0b9c5d4e90f9f"),
    ("y4nQHqYSObI__r002", "2f8fbcdc76e4b39e1f5443844050c09b4aa6efadb313bbc117ed0f719c459ae1"),
    ("y4nQHqYSObI__r003", "f93b53f11de2ca10b6f6e72517b182f51ce65cd89a476df6695bb6d69fbfc8df"),
    ("y4nQHqYSObI__r004", "a99693f9833223fd0a3d9028fca344ff94cc37f8ecf571563402d8d7b329707e"),
    ("y4nQHqYSObI__r005", "9b596344f4fa7e39f1692da060031e3ed15ccecaa3d3ab8377de54fc44b1e1cb"),
    ("y4nQHqYSObI__r006", "baa5137abf6de1897b92180f7c5718db0f441f6c5060175004eccf183b742370"),
    ("y4nQHqYSObI__r007", "dd22111c7cd89dd341ffd68c071ce5a425f362f45bfa58178c37f28f9af415cc"),
    ("y4nQHqYSObI__r008", "06db61d0d5fce7ae02f9b766004faa521d2089fae1c163238094db4f2ac2af73"),
    ("y4nQHqYSObI__r009", "f8ad2437ff57a17ddce2cb71af89666fa519082d86342b9772fb63322e6bb0cc"),
    ("y4nQHqYSObI__r010", "4bdba7a65a4008c9eaf46873c25245af1678705b12bb0e0ad8f670f9fd65af02"),
    ("y4nQHqYSObI__r011", "09023514c2af6cf8ac3d6f506feec6062e8de70f940abfdb5a39ab9bed88a695"),
    ("y4nQHqYSObI__r012", "c02a0537d92f3f7f3a35cd2e23180de78f2fc90feccba6eea39460d4597baab8"),
    ("y4nQHqYSObI__r013", "ebf004983c264df3219368f05aafdefa6a4df5d5ea5f99e8de01c672601dd282"),
    ("y4nQHqYSObI__r014", "ba197c0d40ecc2eb89f1828b11135c266e0e0b07cc7d81feefc976b585da856f"),
    ("y4nQHqYSObI__r015", "d5817e1fa30aa63396a1f67e2070772ef0c46daacfdf4bb9f789fa4ce4c23dd1"),
)


@dataclass(frozen=True)
class Expectations:
    manifest_sha256: str
    builder_sha256: str
    video_ids: tuple[str, ...]
    video_count: int
    session_count: int
    provisional_frames: int
    provisional_hours: float
    expected_max_steps: int
    eligible_windows: int
    segment_items: int
    contributing_sessions: int
    too_short_sessions: int
    y4n_shard_sha256: tuple[tuple[str, str], ...]
    y4n_later_ids: tuple[str, ...]
    y4n_all_eval_windows: int
    y4n_later_eval_windows: int
    config_template_sha256: str


FROZEN_EXPECTATIONS = Expectations(
    manifest_sha256=(
        "67a95de6a4a49f504acdcfe8f316324fe55c2d7fb8e0ff55e82782f0fdecb01b"
    ),
    builder_sha256=(
        "4c08b135d81b95cee9d517943ec41021539c4502c0eb466655f19269104481d6"
    ),
    video_ids=(
        "6vEpVqbrvSE",
        "Y6AeZFCU4LY",
        "kdQbIoMxzZw",
        "nRMVyWdNsTo",
        "ofy37Fm6EgI",
        "v1068970940",
        "v498642684",
    ),
    video_count=7,
    session_count=2_058,
    provisional_frames=4_835_638,
    provisional_hours=22.387213054995033,
    expected_max_steps=2_598,
    eligible_windows=4_076_727,
    segment_items=41_567,
    contributing_sessions=1_729,
    too_short_sessions=329,
    y4n_shard_sha256=Y4N_SHARD_SHA256,
    y4n_later_ids=tuple(
        f"y4nQHqYSObI__r{index:03d}" for index in range(8, 16)
    ),
    y4n_all_eval_windows=554_304,
    y4n_later_eval_windows=269_352,
    config_template_sha256=(
        "9c92ee27ac37115389980490f656af1af5bf0f3389952e652b323f6b279bfb95"
    ),
)


GRU_TEMPLATE_VALUES: dict[str, object] = {
    "window": 128,
    "frame_stride": 3,
    "window_mode": "centered",
    "input_config": "pixels",
    "precomputed_features": True,
    "backbone_feature_dim": 512,
    "feature_deltas": True,
    "embedding_dim": 1024,
    "temporal_dim": 2048,
    "segment_windows": 96,
    "batch_size": 1536,
    "eval_batch_size": 3072,
    "learning_rate": 0.0003,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "linear_lr_decay": True,
    "class_balance": True,
    "class_balance_max": 10.0,
    "transition_weight": 8.0,
    "active_targets_only": True,
    "initial_train_eval": False,
    "eval_interval": 1,
    "max_steps": 1,
    "seed": 0,
}


@dataclass(frozen=True)
class SourcePart:
    video_id: str
    session_id: str
    frames: int
    grid_hz: float
    npz_bytes: int
    sha256: str
    relative_path: str
    source_path: Path
    source_frame_range: tuple[int, int]
    pts_range_s: tuple[float, float]


@dataclass(frozen=True)
class FeaturePart:
    session_id: str
    frames: int
    path: Path
    sha256: str
    worker_index: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, _json_bytes(value))


def _atomic_lines(path: Path, values: Sequence[str]) -> None:
    rendered = "".join(f"{value}\n" for value in values).encode("utf-8")
    _atomic_bytes(path, rendered)


def _npy_header(
    archive: zipfile.ZipFile, member_name: str
) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    with archive.open(member_name) as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(member)
        if version == (2, 0):
            return np.lib.format.read_array_header_2_0(member)
    raise ValueError(f"unsupported NPY header in {member_name}")


def _stored_session_id(value: np.ndarray, path: Path) -> str:
    if value.size != 1:
        raise ValueError(f"{path}: session_id must contain exactly one value")
    return str(value.reshape(()).item())


def _validate_source_npz(part: SourcePart) -> None:
    path = part.source_path
    try:
        with zipfile.ZipFile(path) as archive:
            if set(archive.namelist()) != SOURCE_ARRAY_MEMBERS:
                raise ValueError(f"{path}: source NPZ members changed")
            shape, fortran, dtype = _npy_header(archive, "frames.npy")
            if (
                shape != (part.frames, 128, 128, 3)
                or fortran
                or dtype != np.dtype(np.uint8)
            ):
                raise ValueError(f"{path}: source frame array contract changed")
        with np.load(path, allow_pickle=False) as archive:
            keys = archive["keys"]
            engine = archive["engine_frame_idx"]
            pts = archive["pts_s"]
            active = archive["input_active"]
            stored_id = _stored_session_id(archive["session_id"], path)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"invalid source shard {path}: {error}") from error

    if keys.dtype != np.uint8 or keys.shape != (part.frames, len(KEY_ORDER)):
        raise ValueError(f"{path}: source keys contract changed")
    if not np.all((keys == 0) | (keys == 1)):
        raise ValueError(f"{path}: source keys are not binary")
    if engine.dtype != np.int64 or engine.shape != (part.frames,):
        raise ValueError(f"{path}: source frame-index contract changed")
    if part.frames > 1 and not np.all(np.diff(engine) == 1):
        raise ValueError(f"{path}: source frame indices are not dense")
    if pts.dtype != np.float64 or pts.shape != (part.frames,):
        raise ValueError(f"{path}: source PTS contract changed")
    if not np.all(np.isfinite(pts)) or (
        part.frames > 1 and not np.all(np.diff(pts) > 0)
    ):
        raise ValueError(f"{path}: source PTS is not finite and increasing")
    if (
        active.dtype != np.uint8
        or active.shape != (part.frames,)
        or not np.all(active == 1)
    ):
        raise ValueError(f"{path}: source activity contract changed")
    if stored_id != part.session_id:
        raise ValueError(f"{path}: stored session ID changed")
    if (int(engine[0]), int(engine[-1]) + 1) != part.source_frame_range:
        raise ValueError(f"{path}: source frame range changed")
    if (float(pts[0]), float(pts[-1])) != part.pts_range_s:
        raise ValueError(f"{path}: source PTS range changed")


def _safe_source_path(source_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe source path in aggregate: {relative!r}")
    root = source_root.resolve()
    path = root / candidate
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"source path escapes root: {relative!r}")
    return path


def _validate_source_npz_inventory(
    source_root: Path, parts: Sequence[SourcePart]
) -> None:
    root = source_root.resolve()
    if not root.is_dir():
        raise ValueError(f"source root is not a directory: {root}")
    observed: set[Path] = set()
    symlinks: list[Path] = []
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in list(directory_names):
            path = directory_path / name
            if path.is_symlink():
                symlinks.append(path)
                directory_names.remove(name)
        for name in filenames:
            if not name.endswith(".npz"):
                continue
            path = directory_path / name
            if path.is_symlink():
                symlinks.append(path)
            else:
                observed.add(path)
    if symlinks:
        raise ValueError(f"source corpus contains symlinks: {symlinks[0]}")
    expected = {part.source_path for part in parts}
    if observed != expected:
        missing = sorted(str(path) for path in expected - observed)
        extra = sorted(str(path) for path in observed - expected)
        raise ValueError(
            "source NPZ inventory changed: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )


def load_source_inventory(
    manifest_path: Path,
    source_root: Path,
    *,
    expectations: Expectations = FROZEN_EXPECTATIONS,
    validate_files: bool = True,
) -> tuple[dict[str, Any], list[SourcePart]]:
    """Validate the frozen aggregate and return its canonical part inventory."""

    manifest_path = Path(manifest_path)
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expectations.manifest_sha256:
        raise ValueError(
            "wild aggregate SHA-256 mismatch: "
            f"{actual_manifest_sha256} != {expectations.manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != CORPUS_FORMAT:
        raise ValueError("wild aggregate format changed")
    if manifest.get("admission_tier") != ADMISSION_TIER:
        raise ValueError("wild aggregate is not explicitly provisional")
    if manifest.get("builder", {}).get("sha256") != expectations.builder_sha256:
        raise ValueError("wild aggregate builder SHA-256 changed")
    if int(manifest.get("video_count", -1)) != expectations.video_count:
        raise ValueError("wild aggregate video count changed")
    if int(manifest.get("session_count", -1)) != expectations.session_count:
        raise ValueError("wild aggregate session count changed")
    if int(manifest.get("provisional_trainable_frames", -1)) != (
        expectations.provisional_frames
    ):
        raise ValueError("wild aggregate provisional frame count changed")
    if not math.isclose(
        float(manifest.get("provisional_trainable_hours", -1.0)),
        expectations.provisional_hours,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("wild aggregate provisional hours changed")
    if int(manifest.get("train_ready_frames", -1)) != 0 or float(
        manifest.get("train_ready_hours", -1.0)
    ) != 0.0:
        raise ValueError("wild aggregate unexpectedly claims clean train-ready data")

    videos = manifest.get("videos")
    if not isinstance(videos, list):
        raise ValueError("wild aggregate videos must be a list")
    video_ids = tuple(sorted(str(row.get("video_id")) for row in videos))
    if video_ids != tuple(sorted(expectations.video_ids)):
        raise ValueError("wild aggregate video membership changed")
    explicit = manifest.get("verification", {}).get("explicit_video_set")
    if explicit != sorted(expectations.video_ids):
        raise ValueError("wild aggregate explicit video set changed")
    if manifest.get("verification", {}).get("expected_frame_shape") != [128, 128, 3]:
        raise ValueError("wild aggregate frame shape changed")

    parts: list[SourcePart] = []
    seen: set[str] = set()
    for video in videos:
        video_id = str(video["video_id"])
        grid_hz = float(video["effective_grid_hz"])
        if not math.isfinite(grid_hz) or grid_hz <= 0:
            raise ValueError(f"{video_id}: invalid effective grid")
        rows = video.get("parts")
        if not isinstance(rows, list) or len(rows) != int(video.get("part_count", -1)):
            raise ValueError(f"{video_id}: part inventory changed")
        video_frames = 0
        for part_index, row in enumerate(rows):
            session_id = str(row.get("session_id"))
            expected_id = f"wild_provisional_{video_id}__r{part_index:03d}"
            if session_id != expected_id or session_id in seen:
                raise ValueError(f"{video_id}: noncanonical or duplicate session ID")
            seen.add(session_id)
            npz_name = str(row.get("npz"))
            expected_relative = f"{video_id}/parts/{npz_name}"
            if npz_name != f"{session_id}.npz" or row.get("path") != expected_relative:
                raise ValueError(f"{session_id}: source path or filename changed")
            frames = int(row.get("frames", -1))
            npz_bytes = int(row.get("npz_bytes", -1))
            sha256 = str(row.get("sha256", ""))
            source_range = row.get("source_frame_range")
            pts_range = row.get("pts_range_s")
            if frames < 1 or npz_bytes < 1 or len(sha256) != 64:
                raise ValueError(f"{session_id}: invalid declared part metadata")
            if not isinstance(source_range, list) or len(source_range) != 2:
                raise ValueError(f"{session_id}: invalid source frame range")
            if not isinstance(pts_range, list) or len(pts_range) != 2:
                raise ValueError(f"{session_id}: invalid PTS range")
            source_path = _safe_source_path(source_root, expected_relative)
            source_part = SourcePart(
                video_id=video_id,
                session_id=session_id,
                frames=frames,
                grid_hz=grid_hz,
                npz_bytes=npz_bytes,
                sha256=sha256,
                relative_path=expected_relative,
                source_path=source_path,
                source_frame_range=(int(source_range[0]), int(source_range[1])),
                pts_range_s=(float(pts_range[0]), float(pts_range[1])),
            )
            parts.append(source_part)
            video_frames += frames
        if video_frames != int(video.get("provisional_trainable_frames", -1)):
            raise ValueError(f"{video_id}: provisional frame accounting changed")
        expected_hours = video_frames / grid_hz / 3600.0
        if not math.isclose(
            expected_hours,
            float(video.get("provisional_trainable_hours", -1.0)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{video_id}: provisional hour accounting changed")

    if len(parts) != expectations.session_count or len(seen) != len(parts):
        raise ValueError("wild source session inventory changed")
    if sum(part.frames for part in parts) != expectations.provisional_frames:
        raise ValueError("wild source frame total changed")
    if validate_files:
        _validate_source_npz_inventory(source_root, parts)
        for part in parts:
            if part.source_path.stat().st_size != part.npz_bytes:
                raise ValueError(f"{part.source_path}: source byte size changed")
            if sha256_file(part.source_path) != part.sha256:
                raise ValueError(f"{part.source_path}: source SHA-256 changed")
            _validate_source_npz(part)
    return manifest, parts


def partition_two_workers(
    parts: Sequence[SourcePart],
) -> tuple[list[SourcePart], list[SourcePart]]:
    """Return a deterministic largest-first byte-balanced two-way partition."""

    assigned: list[list[SourcePart]] = [[], []]
    totals = [0, 0]
    ordered = sorted(parts, key=lambda part: (-part.npz_bytes, part.session_id))
    for part in ordered:
        worker = min(range(2), key=lambda index: (totals[index], index))
        assigned[worker].append(part)
        totals[worker] += part.npz_bytes
    for worker_parts in assigned:
        worker_parts.sort(key=lambda part: part.session_id)
    return assigned[0], assigned[1]


def prepare_source_inputs(
    manifest_path: Path,
    source_root: Path,
    output: Path,
    *,
    expectations: Expectations = FROZEN_EXPECTATIONS,
) -> dict[str, Any]:
    """Validate every source shard and atomically write two worker lists."""

    manifest, parts = load_source_inventory(
        manifest_path,
        source_root,
        expectations=expectations,
        validate_files=True,
    )
    workers = partition_two_workers(parts)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    worker_receipts: list[dict[str, Any]] = []
    for worker_index, worker_parts in enumerate(workers):
        paths = [str(part.source_path) for part in worker_parts]
        list_path = output / f"worker_{worker_index}_inputs.txt"
        _atomic_lines(list_path, paths)
        worker_receipts.append(
            {
                "worker_index": worker_index,
                "input_list": str(list_path),
                "input_list_sha256": sha256_file(list_path),
                "sessions": len(worker_parts),
                "frames": sum(part.frames for part in worker_parts),
                "npz_bytes": sum(part.npz_bytes for part in worker_parts),
            }
        )
    receipt = {
        "format_version": "madeleine.wild-provisional-gru-source-plan.v1",
        "source_manifest": str(Path(manifest_path).resolve()),
        "source_manifest_sha256": expectations.manifest_sha256,
        "source_builder_sha256": expectations.builder_sha256,
        "source_root": str(Path(source_root).resolve()),
        "source_format": manifest["format_version"],
        "training_label_kind": SOURCE_LABEL_KIND,
        "admission_tier": ADMISSION_TIER,
        "train_ready_frames": 0,
        "train_ready_hours": 0.0,
        "provisional_trainable_frames": expectations.provisional_frames,
        "provisional_trainable_hours": expectations.provisional_hours,
        "video_count": expectations.video_count,
        "session_count": expectations.session_count,
        "source_npz_bytes": sum(part.npz_bytes for part in parts),
        "source_files_sizes_sha256_and_schema_validated": True,
        "worker_partition_policy": (
            "two-way deterministic largest-NPZ-first byte balancing; ties to "
            "lower worker index; each emitted list sorted by session ID"
        ),
        "workers": worker_receipts,
        "warning": (
            "No frame is clean/admitted training data. This plan permits only "
            "an explicitly provisional noisy-supervision diagnostic."
        ),
    }
    _atomic_json(output / "source_validation.json", receipt)
    return receipt


def _validate_feature_npz(
    path: Path,
    session_id: str,
    *,
    expected_frames: int | None = None,
    source_path: Path | None = None,
) -> tuple[int, str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing regular feature shard: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != FEATURE_ARRAY_NAMES:
                raise ValueError(f"{path}: feature NPZ members changed")
            features = archive["features"]
            keys = archive["keys"]
            engine = archive["engine_frame_idx"]
            active = archive["input_active"]
            stored_id = _stored_session_id(archive["session_id"], path)
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"invalid feature shard {path}: {error}") from error

    frames = len(features)
    if expected_frames is not None and frames != expected_frames:
        raise ValueError(f"{path}: feature frame count changed")
    if features.dtype != np.float16 or features.shape != (frames, FEATURE_DIM):
        raise ValueError(f"{path}: feature array contract changed")
    if not np.all(np.isfinite(features)):
        raise ValueError(f"{path}: feature array contains non-finite values")
    if keys.dtype != np.uint8 or keys.shape != (frames, len(KEY_ORDER)):
        raise ValueError(f"{path}: feature keys contract changed")
    if not np.all((keys == 0) | (keys == 1)):
        raise ValueError(f"{path}: feature keys are not binary")
    if engine.dtype != np.int64 or engine.shape != (frames,):
        raise ValueError(f"{path}: feature frame-index contract changed")
    if frames > 1 and not np.all(np.diff(engine) == 1):
        raise ValueError(f"{path}: feature frame indices are not dense")
    if (
        active.dtype != np.uint8
        or active.shape != (frames,)
        or not np.all(active == 1)
    ):
        raise ValueError(f"{path}: feature activity contract changed")
    if stored_id != session_id:
        raise ValueError(f"{path}: feature session ID changed")

    if source_path is not None:
        with np.load(source_path, allow_pickle=False) as source:
            for name, feature_value in (
                ("keys", keys),
                ("engine_frame_idx", engine),
                ("input_active", active),
            ):
                if not np.array_equal(source[name], feature_value):
                    raise ValueError(
                        f"{path}: feature {name} differs from provisional source"
                    )
            if _stored_session_id(source["session_id"], source_path) != session_id:
                raise ValueError(f"{source_path}: source session ID changed")
    return frames, sha256_file(path)


def _read_worker_lists(
    plan_dir: Path, parts: Sequence[SourcePart]
) -> tuple[list[SourcePart], list[SourcePart]]:
    expected_workers = partition_two_workers(parts)
    for worker_index, expected in enumerate(expected_workers):
        path = plan_dir / f"worker_{worker_index}_inputs.txt"
        if not path.is_file():
            raise ValueError(f"worker input list is missing: {path}")
        observed = path.read_text().splitlines()
        wanted = [str(part.source_path) for part in expected]
        if observed != wanted:
            raise ValueError(f"worker {worker_index} input list changed")
    return expected_workers


def _validate_worker_features(
    worker_index: int,
    root: Path,
    expected_parts: Sequence[SourcePart],
) -> tuple[list[FeaturePart], dict[str, Any]]:
    manifest_path = root / "feature_build_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"worker {worker_index} feature manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    expected_header = {
        "format": FEATURE_FORMAT,
        "backbone_feature_dim": FEATURE_DIM,
        "frame_size": 128,
        "source_kind": "audited_rgb_shards",
    }
    for key, value in expected_header.items():
        if manifest.get(key) != value:
            raise ValueError(f"worker {worker_index} feature manifest changed {key}")
    reports = manifest.get("sessions")
    if not isinstance(reports, list) or len(reports) != len(expected_parts):
        raise ValueError(f"worker {worker_index} feature session count changed")
    actual_npz = {path.name for path in root.glob("*.npz") if path.is_file()}
    expected_npz = {f"{part.session_id}.npz" for part in expected_parts}
    if actual_npz != expected_npz:
        raise ValueError(f"worker {worker_index} feature NPZ inventory changed")

    outputs: list[FeaturePart] = []
    for report, source in zip(reports, expected_parts, strict=True):
        if not isinstance(report, dict):
            raise ValueError(f"worker {worker_index} has an invalid feature report")
        if report.get("session_id") != source.session_id:
            raise ValueError(f"worker {worker_index} feature report order changed")
        if int(report.get("frames", -1)) != source.frames:
            raise ValueError(f"{source.session_id}: feature report frames changed")
        if report.get("source") != str(source.source_path):
            raise ValueError(f"{source.session_id}: feature source path changed")
        expected_name = f"{source.session_id}.npz"
        if report.get("npz") != expected_name:
            raise ValueError(f"{source.session_id}: feature filename changed")
        feature_path = root / expected_name
        frames, digest = _validate_feature_npz(
            feature_path,
            source.session_id,
            expected_frames=source.frames,
            source_path=source.source_path,
        )
        outputs.append(
            FeaturePart(
                session_id=source.session_id,
                frames=frames,
                path=feature_path,
                sha256=digest,
                worker_index=worker_index,
            )
        )
    receipt = {
        "worker_index": worker_index,
        "root": str(root.resolve()),
        "feature_manifest": str(manifest_path.resolve()),
        "feature_manifest_sha256": sha256_file(manifest_path),
        "sessions": len(outputs),
        "frames": sum(output.frames for output in outputs),
    }
    return outputs, receipt


def _load_gru_template(
    path: Path, expectations: Expectations
) -> dict[str, Any]:
    if sha256_file(path) != expectations.config_template_sha256:
        raise ValueError("matched GRU config template SHA-256 changed")
    config = json.loads(path.read_text())
    if set(config) != {*GRU_TEMPLATE_VALUES, "_note"}:
        raise ValueError("matched GRU config template fields changed")
    for key, value in GRU_TEMPLATE_VALUES.items():
        if config.get(key) != value:
            raise ValueError(f"matched GRU config template changed {key}")
    if "temporal_arch" in config:
        raise ValueError("matched reference must use the default GRU")
    return config


def _compute_endpoint(
    parts: Sequence[SourcePart], config: dict[str, Any]
) -> dict[str, int]:
    frame_span = (int(config["window"]) - 1) * int(config["frame_stride"]) + 1
    segment_windows = int(config["segment_windows"])
    loader_batch_items = max(
        1, round(int(config["batch_size"]) / segment_windows)
    )
    eligible_by_session = [max(0, part.frames - frame_span + 1) for part in parts]
    segments_by_session = [
        eligible // segment_windows for eligible in eligible_by_session
    ]
    segment_items = sum(segments_by_session)
    return {
        "frame_span": frame_span,
        "eligible_windows": sum(eligible_by_session),
        "contributing_sessions": sum(value > 0 for value in eligible_by_session),
        "too_short_sessions": sum(value == 0 for value in eligible_by_session),
        "segment_windows": segment_windows,
        "segment_items": segment_items,
        "used_training_windows": segment_items * segment_windows,
        "discarded_eligible_tail_windows": (
            sum(eligible_by_session) - segment_items * segment_windows
        ),
        "loader_batch_items": loader_batch_items,
        "max_steps": math.ceil(segment_items / loader_batch_items),
    }


def _inventory_digest(features: Sequence[FeaturePart]) -> str:
    digest = hashlib.sha256()
    for feature in sorted(features, key=lambda row: row.session_id):
        digest.update(
            f"{feature.session_id}\t{feature.frames}\t{feature.sha256}\n".encode()
        )
    return digest.hexdigest()


def _link_exact(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise ValueError(f"refusing to overwrite assembled path: {destination}")
    os.link(source, destination)
    if not os.path.samefile(source, destination):
        raise AssertionError(f"hard-link identity check failed: {destination}")


def assemble_data_view(
    *,
    manifest_path: Path,
    source_root: Path,
    plan_dir: Path,
    worker_feature_roots: Sequence[Path],
    y4n_data: Path,
    config_template: Path,
    output: Path,
    expectations: Expectations = FROZEN_EXPECTATIONS,
) -> dict[str, Any]:
    """Validate both feature workers and atomically publish one data view."""

    if len(worker_feature_roots) != 2:
        raise ValueError("exactly two worker feature roots are required")
    _, parts = load_source_inventory(
        manifest_path,
        source_root,
        expectations=expectations,
        validate_files=True,
    )
    workers = _read_worker_lists(Path(plan_dir), parts)
    feature_parts: list[FeaturePart] = []
    worker_receipts: list[dict[str, Any]] = []
    for worker_index, (root, expected_parts) in enumerate(
        zip(worker_feature_roots, workers, strict=True)
    ):
        outputs, receipt = _validate_worker_features(
            worker_index, Path(root), expected_parts
        )
        feature_parts.extend(outputs)
        worker_receipts.append(receipt)
    if len(feature_parts) != expectations.session_count:
        raise ValueError("assembled provisional feature session count changed")
    if len({row.session_id for row in feature_parts}) != len(feature_parts):
        raise ValueError("duplicate provisional feature session ID")
    if sum(row.frames for row in feature_parts) != expectations.provisional_frames:
        raise ValueError("assembled provisional feature frame count changed")

    config = _load_gru_template(Path(config_template), expectations)
    endpoint = _compute_endpoint(parts, config)
    frozen_endpoint = {
        "eligible_windows": expectations.eligible_windows,
        "segment_items": expectations.segment_items,
        "contributing_sessions": expectations.contributing_sessions,
        "too_short_sessions": expectations.too_short_sessions,
        "max_steps": expectations.expected_max_steps,
    }
    for key, value in frozen_endpoint.items():
        if endpoint[key] != value:
            raise ValueError(
                f"wild provisional one-pass endpoint changed {key}: "
                f"{endpoint[key]} != {value}"
            )
    config["max_steps"] = endpoint["max_steps"]
    config["eval_interval"] = endpoint["max_steps"]
    config["_note"] = (
        "25.7M matched GRU trained for one exact pass over the immutable "
        "broad-seven provisional wild-overlay corpus; diagnostic noisy "
        f"supervision only; final weights at {endpoint['max_steps']} steps"
    )

    y4n_rows: list[FeaturePart] = []
    y4n_hashes = dict(expectations.y4n_shard_sha256)
    y4n_ids = [session_id for session_id, _ in expectations.y4n_shard_sha256]
    for session_id in y4n_ids:
        path = Path(y4n_data) / f"{session_id}.npz"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing regular y4n feature shard: {path}")
        if sha256_file(path) != y4n_hashes[session_id]:
            raise ValueError(f"{session_id}: frozen y4n shard SHA-256 changed")
        frames, digest = _validate_feature_npz(path, session_id)
        y4n_rows.append(
            FeaturePart(
                session_id=session_id,
                frames=frames,
                path=path,
                sha256=digest,
                worker_index=-1,
            )
        )
    frame_span = endpoint["frame_span"]
    all_y4n_windows = sum(max(0, row.frames - frame_span + 1) for row in y4n_rows)
    later_set = set(expectations.y4n_later_ids)
    if not later_set.issubset(y4n_ids):
        raise ValueError("later-eight y4n membership is not a validation subset")
    later_y4n_windows = sum(
        max(0, row.frames - frame_span + 1)
        for row in y4n_rows
        if row.session_id in later_set
    )
    if all_y4n_windows != expectations.y4n_all_eval_windows:
        raise ValueError("full y4n evaluation support changed")
    if later_y4n_windows != expectations.y4n_later_eval_windows:
        raise ValueError("later-eight y4n evaluation support changed")

    output = Path(output)
    if os.path.lexists(output):
        raise ValueError(f"refusing to overwrite assembled data view: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        for feature in sorted(feature_parts, key=lambda row: row.session_id):
            _link_exact(feature.path, temporary / f"{feature.session_id}.npz")
        for feature in y4n_rows:
            _link_exact(feature.path, temporary / f"{feature.session_id}.npz")

        train_ids = sorted(row.session_id for row in feature_parts)
        later_ids = list(expectations.y4n_later_ids)
        _atomic_lines(temporary / "train_sessions.txt", train_ids)
        _atomic_lines(temporary / "val_sessions.txt", y4n_ids)
        _atomic_lines(temporary / "later_eight_sessions.txt", later_ids)
        _atomic_json(temporary / "config.json", config)
        config_sha256 = sha256_file(temporary / "config.json")

        receipt = {
            "format_version": "madeleine.wild-provisional-gru-split.v1",
            "source_manifest": str(Path(manifest_path).resolve()),
            "source_manifest_sha256": expectations.manifest_sha256,
            "source_builder_sha256": expectations.builder_sha256,
            "training_label_kind": SOURCE_LABEL_KIND,
            "admission_tier": ADMISSION_TIER,
            "train_ready_frames": 0,
            "train_ready_hours": 0.0,
            "provisional_trainable_frames": expectations.provisional_frames,
            "provisional_trainable_hours": expectations.provisional_hours,
            "provisional_sessions": expectations.session_count,
            "provisional_videos": expectations.video_count,
            "feature_format": FEATURE_FORMAT,
            "backbone_feature_dim": FEATURE_DIM,
            "feature_inventory_sha256": _inventory_digest(feature_parts),
            "worker_features": worker_receipts,
            "hard_link_policy": {
                "training_shards": len(feature_parts),
                "y4n_validation_shards": len(y4n_rows),
                "copy_or_symlink_fallback": False,
            },
            "split": {
                "train_sessions_file": "train_sessions.txt",
                "train_sessions": len(train_ids),
                "validation_sessions_file": "val_sessions.txt",
                "validation_sessions": y4n_ids,
                "later_eight_sessions_file": "later_eight_sessions.txt",
                "later_eight_sessions": later_ids,
                "train_validation_overlap": False,
            },
            "evaluation": {
                "mapped_y4n_role": (
                    "same mapped-label development surface as the matched "
                    "103.4056-hour GRU reference"
                ),
                "full_y4n_eligible_windows": all_y4n_windows,
                "later_eight_y4n_eligible_windows": later_y4n_windows,
                "y4n_shard_sha256": y4n_hashes,
            },
            "recipe": {
                "model": "BadelineIDM default GRU",
                "trainable_parameters": GRU_TRAINABLE_PARAMETERS,
                "config_template": str(Path(config_template).resolve()),
                "config_template_sha256": expectations.config_template_sha256,
                "assembled_config_sha256": config_sha256,
                "checkpoint_policy": (
                    "final_state_dict at the fixed one-pass endpoint only; "
                    "selected model_state_dict is not a decision input"
                ),
                **endpoint,
            },
            "warning": (
                "This is an explicitly provisional noisy-supervision "
                "diagnostic. Validation proves data identity and mechanical "
                "schema only; it does not admit wild labels as clean truth."
            ),
        }
        if set(train_ids) & set(y4n_ids):
            raise AssertionError("training and y4n validation sessions overlap")
        _atomic_json(temporary / "split_receipt.json", receipt)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan", help="validate RGB shards and emit exactly two worker lists"
    )
    plan.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    plan.add_argument("--source-root", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)

    assemble = subparsers.add_parser(
        "assemble", help="verify two feature workers and publish the data view"
    )
    assemble.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    assemble.add_argument("--source-root", type=Path, required=True)
    assemble.add_argument("--plan-dir", type=Path, required=True)
    assemble.add_argument(
        "--worker-features", type=Path, nargs=2, required=True,
        metavar=("WORKER_0", "WORKER_1"),
    )
    assemble.add_argument("--y4n-data", type=Path, required=True)
    assemble.add_argument(
        "--config-template", type=Path, default=DEFAULT_CONFIG_TEMPLATE
    )
    assemble.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        receipt = prepare_source_inputs(
            args.manifest, args.source_root, args.out
        )
        summary = {
            "admission_tier": receipt["admission_tier"],
            "video_count": receipt["video_count"],
            "session_count": receipt["session_count"],
            "provisional_trainable_frames": receipt[
                "provisional_trainable_frames"
            ],
            "workers": receipt["workers"],
        }
    else:
        receipt = assemble_data_view(
            manifest_path=args.manifest,
            source_root=args.source_root,
            plan_dir=args.plan_dir,
            worker_feature_roots=args.worker_features,
            y4n_data=args.y4n_data,
            config_template=args.config_template,
            output=args.out,
        )
        summary = {
            "admission_tier": receipt["admission_tier"],
            "provisional_sessions": receipt["provisional_sessions"],
            "max_steps": receipt["recipe"]["max_steps"],
            "later_eight_y4n_eligible_windows": receipt["evaluation"][
                "later_eight_y4n_eligible_windows"
            ],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
