#!/usr/bin/env python3
"""Evaluate historical Badeline GRUs on frozen Wild7 rows.

The public VPT scorecard retains the middle 64 positions from each 128-frame
20 Hz window.  A centered 128-sample GRU with native stride three needs more
context at each stream boundary than that VPT surface provides.  This driver
therefore constructs the largest natural (unpadded) subset supported by the
GRU and scores every requested checkpoint, plus existing VPT sidecars, on
that exact row identity.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterable

import numpy as np
import torch

from badeline.model import BadelineIDM
from badeline.train import contiguous_runs, target_offset
from data.precompute_features import FeatureEncoder
from data.schema import KEY_ORDER
from experiments.eval_vpt_small_native60_wild7 import (
    json_ready,
    key_metrics,
    load_manifest,
    load_reference,
    load_verification,
    sha256_file,
    source_path,
    video_id_from_session,
)


SCHEMA = "madeleine.gru-wild7-eval.v1"
SUPPORT_SCHEMA = "madeleine.gru128x3-wild7-common-support.v1"


def load_raw_session(path: Path, expected_session_id: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "frames",
            "keys",
            "engine_frame_idx",
            "input_active",
            "session_id",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"raw Wild shard lacks {sorted(missing)}: {path}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    session_id = str(arrays["session_id"].reshape(()).item())
    if session_id != expected_session_id:
        raise RuntimeError(f"stored session identity differs: {path}")
    frames = arrays["frames"]
    keys = arrays["keys"]
    engine_idx = arrays["engine_frame_idx"]
    active = arrays["input_active"]
    if frames.dtype != np.uint8 or frames.shape[1:] != (128, 128, 3):
        raise ValueError(f"unexpected frame array: {path}")
    if keys.dtype != np.uint8 or keys.shape != (len(frames), len(KEY_ORDER)):
        raise ValueError(f"unexpected key array: {path}")
    if engine_idx.dtype != np.int64 or engine_idx.shape != (len(frames),):
        raise ValueError(f"unexpected engine index array: {path}")
    if active.dtype != np.uint8 or active.shape != (len(frames),):
        raise ValueError(f"unexpected activity array: {path}")
    return arrays


def row_run_bounds(engine_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each raw row's half-open contiguous-run bounds."""

    starts = np.empty(len(engine_idx), dtype=np.int64)
    ends = np.empty(len(engine_idx), dtype=np.int64)
    for start, end in contiguous_runs(engine_idx):
        starts[start:end] = start
        ends[start:end] = end
    return starts, ends


def context_supported(
    source_rows: np.ndarray,
    engine_idx: np.ndarray,
    *,
    window: int,
    frame_stride: int,
    window_mode: str = "centered",
) -> np.ndarray:
    """Return rows with the checkpoint's full natural temporal context."""

    rows = np.asarray(source_rows, dtype=np.int64)
    if np.any(rows < 0) or np.any(rows >= len(engine_idx)):
        raise ValueError("source row lies outside its raw session")
    offset = target_offset(window, window_mode)
    past = offset * frame_stride
    future = (window - 1 - offset) * frame_stride
    run_start, run_end = row_run_bounds(engine_idx)
    return np.logical_and(rows - run_start[rows] >= past, run_end[rows] - 1 - rows >= future)


def reference_stream_slices(reference: dict[str, np.ndarray]) -> list[slice]:
    lengths = reference["session_lengths"].astype(np.int64, copy=False)
    starts = np.concatenate(([0], np.cumsum(lengths[:-1], dtype=np.int64)))
    return [slice(int(start), int(start + length)) for start, length in zip(starts, lengths, strict=True)]


def selected_stream_metadata(
    reference: dict[str, np.ndarray], selected: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lengths: list[int] = []
    ids: list[str] = []
    stream_ids = reference["session_ids"].astype(str, copy=False)
    for stream_id, stream_slice in zip(stream_ids, reference_stream_slices(reference), strict=True):
        count = int(selected[stream_slice].sum())
        if count:
            lengths.append(count)
            ids.append(str(stream_id))
    return np.asarray(lengths, dtype=np.int64), np.asarray(ids)


def build_support(args: argparse.Namespace) -> dict[str, Any]:
    manifest, records = load_manifest(args.manifest)
    load_verification(
        args.source_verification,
        manifest_path=args.manifest,
        raw_root=args.raw_root,
        records=records,
    )
    reference = load_reference(args.reference_sidecar)
    row_sessions = reference["row_session_id"].astype(str, copy=False)
    selected = np.zeros(len(row_sessions), dtype=bool)
    for index, session_id in enumerate(sorted(set(row_sessions.tolist())), start=1):
        raw = load_raw_session(source_path(args.raw_root, session_id), session_id)
        positions = np.flatnonzero(row_sessions == session_id)
        source_rows = reference["source_row_index"][positions].astype(np.int64)
        if not np.array_equal(
            raw["keys"][source_rows], reference["y_true"][positions]
        ):
            raise RuntimeError(f"raw/reference truth differs: {session_id}")
        if not np.array_equal(
            raw["input_active"][source_rows], reference["input_active"][positions]
        ):
            raise RuntimeError(f"raw/reference activity differs: {session_id}")
        if not np.array_equal(
            raw["engine_frame_idx"][source_rows],
            reference["source_engine_frame_idx"][positions],
        ):
            raise RuntimeError(f"raw/reference engine index differs: {session_id}")
        selected[positions] = context_supported(
            source_rows,
            raw["engine_frame_idx"],
            window=args.window,
            frame_stride=args.frame_stride,
        )
        if index % 100 == 0 or index == len(set(row_sessions.tolist())):
            print(f"support_sessions={index}/{len(set(row_sessions.tolist()))}", flush=True)
    selected_indices = np.flatnonzero(selected).astype(np.int64)
    if not len(selected_indices):
        raise RuntimeError("natural GRU support is empty")
    stream_lengths, stream_ids = selected_stream_metadata(reference, selected)
    if int(stream_lengths.sum()) != len(selected_indices):
        raise RuntimeError("selected stream lengths do not cover common support")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        reference_indices=selected_indices,
        y_true=reference["y_true"][selected],
        y_prob=reference["y_prob"][selected],
        input_active=reference["input_active"][selected],
        session_lengths=stream_lengths,
        session_ids=stream_ids,
        source_row_index=reference["source_row_index"][selected],
        source_engine_frame_idx=reference["source_engine_frame_idx"][selected],
    )
    gate = reference["input_active"][selected].astype(bool)
    result = {
        "schema_version": SUPPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": "full natural context only; no boundary padding",
        "window": args.window,
        "frame_stride": args.frame_stride,
        "reference_rows": int(len(reference["y_true"])),
        "selected_rows": int(len(selected_indices)),
        "selected_active_rows": int(gate.sum()),
        "selected_streams": int(len(stream_lengths)),
        "manifest_sha256": sha256_file(args.manifest),
        "reference_sidecar_sha256": sha256_file(args.reference_sidecar),
        "source_verification_sha256": sha256_file(args.source_verification),
        "support_sidecar": {
            "path": str(args.out),
            "bytes": args.out.stat().st_size,
            "sha256": sha256_file(args.out),
        },
        "reference_model_on_common_support": key_metrics(
            reference["y_true"][selected][gate], reference["y_prob"][selected][gate]
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def load_support(path: Path, reference: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = {
        "reference_indices",
        "y_true",
        "input_active",
        "session_lengths",
        "session_ids",
        "source_row_index",
        "source_engine_frame_idx",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"support sidecar lacks {sorted(missing)}")
        support = {name: np.asarray(archive[name]) for name in required}
    indices = support["reference_indices"].astype(np.int64, copy=False)
    checks = {
        "y_true": "y_true",
        "input_active": "input_active",
        "source_row_index": "source_row_index",
        "source_engine_frame_idx": "source_engine_frame_idx",
    }
    for support_name, reference_name in checks.items():
        if not np.array_equal(support[support_name], reference[reference_name][indices]):
            raise RuntimeError(f"support/reference {support_name} differs")
    if int(support["session_lengths"].sum()) != len(indices):
        raise RuntimeError("support stream lengths do not cover rows")
    return support


def checkpoint_state(checkpoint: dict[str, Any], weights: str) -> Any:
    if weights == "final":
        name = "final_state_dict"
    elif weights == "selected":
        name = "model_state_dict" if "model_state_dict" in checkpoint else "model"
    else:
        raise ValueError(f"unsupported weights: {weights}")
    if name not in checkpoint:
        raise KeyError(f"checkpoint lacks {name}")
    return checkpoint[name]


def load_model(
    path: Path, expected_sha256: str, weights: str, device: torch.device
) -> tuple[BadelineIDM, dict[str, Any], dict[str, Any], str]:
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise RuntimeError("checkpoint SHA-256 differs from expected value")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    model = BadelineIDM(config)
    model.load_state_dict(checkpoint_state(checkpoint, weights))
    model.to(device).eval()
    return model, checkpoint, config, digest


def load_or_encode_features(
    encoder: FeatureEncoder,
    frames: np.ndarray,
    cache_path: Path | None,
) -> np.ndarray:
    expected_shape = (len(frames), 512)
    if cache_path is not None and cache_path.is_file():
        cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if cached.dtype != np.float16 or cached.shape != expected_shape:
            raise RuntimeError(f"invalid frozen-feature cache: {cache_path}")
        return cached
    features = encoder.encode(frames)
    if features.dtype != np.float16 or features.shape != expected_shape:
        raise RuntimeError("frozen feature encoder changed its output contract")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp.npy")
        np.save(temporary, features, allow_pickle=False)
        temporary.replace(cache_path)
    return features


def feature_predictions(
    model: BadelineIDM,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    offset = target_offset(model.window, model.window_mode)
    starts = targets - offset * model.frame_stride
    sample = np.arange(model.window, dtype=np.int64) * model.frame_stride
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for begin in range(0, len(starts), batch_size):
            block_starts = starts[begin : begin + batch_size]
            windows = np.stack([features[start + sample] for start in block_starts])
            tensor = torch.from_numpy(windows).to(device=device, dtype=torch.float32)
            output.append(torch.sigmoid(model({"features": tensor})).float().cpu().numpy())
    return np.concatenate(output) if output else np.empty((0, len(KEY_ORDER)), dtype=np.float32)


def pixel_predictions(
    model: BadelineIDM,
    frames: np.ndarray,
    targets: np.ndarray,
    *,
    dense_target_count: int,
    device: torch.device,
) -> np.ndarray:
    """Encode each native frame once across contiguous phase-0 target blocks."""

    if dense_target_count < 1:
        raise ValueError("dense_target_count must be positive")
    offset = target_offset(model.window, model.window_mode)
    frame_span = (model.window - 1) * model.frame_stride + 1
    order = np.argsort(targets, kind="stable")
    ordered_targets = targets[order]
    ordered_probability = np.empty(
        (len(targets), len(KEY_ORDER)), dtype=np.float32
    )
    # Frozen Wild7 targets lie on phase 0 of the 60 Hz source. Splitting at
    # any larger gap avoids encoding inactive or discontinuous spans merely
    # to reach the next scored target.
    group_starts = np.concatenate(
        ([0], np.flatnonzero(np.diff(ordered_targets) != 3) + 1)
    )
    group_ends = np.concatenate((group_starts[1:], [len(ordered_targets)]))
    with torch.inference_mode():
        for group_start, group_end in zip(group_starts, group_ends, strict=True):
            for begin in range(group_start, group_end, dense_target_count):
                end = min(begin + dense_target_count, group_end)
                block_targets = ordered_targets[begin:end]
                first_target = int(block_targets[0])
                last_target = int(block_targets[-1])
                first_window = first_target - offset * model.frame_stride
                dense_windows = last_target - first_target + 1
                block = frames[
                    first_window : first_window + dense_windows + frame_span - 1
                ]
                tensor = (
                    torch.from_numpy(np.array(block, copy=True))
                    .permute(0, 3, 1, 2)
                    .to(device=device, dtype=torch.float32)
                    .div_(255.0)
                    .unsqueeze(0)
                )
                dense_logits = model.forward_segment({"frames": tensor})[0]
                selected = torch.from_numpy(
                    (block_targets - first_target).astype(np.int64)
                ).to(device)
                ordered_probability[begin:end] = (
                    torch.sigmoid(dense_logits[selected]).float().cpu().numpy()
                )
    probability = np.empty_like(ordered_probability)
    probability[order] = ordered_probability
    return probability


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    manifest, records = load_manifest(args.manifest)
    load_verification(
        args.source_verification,
        manifest_path=args.manifest,
        raw_root=args.raw_root,
        records=records,
    )
    reference = load_reference(args.reference_sidecar)
    support = load_support(args.support_sidecar, reference)
    indices = support["reference_indices"].astype(np.int64, copy=False)
    selected_sessions = reference["row_session_id"][indices].astype(str, copy=False)
    device = torch.device(args.device)
    model, checkpoint, config, checkpoint_sha = load_model(
        args.checkpoint, args.expected_checkpoint_sha256, args.weights, device
    )
    if model.input_config != "pixels":
        raise ValueError("Wild7 GRU evaluator currently supports pixel-input checkpoints")
    probability = np.full(support["y_true"].shape, np.nan, dtype=np.float32)
    encoder = (
        FeatureEncoder(args.device, args.encoder_batch_size)
        if model.precomputed_features
        else None
    )
    required_sessions = sorted(set(selected_sessions.tolist()))
    for session_number, session_id in enumerate(required_sessions, start=1):
        raw = load_raw_session(source_path(args.raw_root, session_id), session_id)
        positions = np.flatnonzero(selected_sessions == session_id)
        targets = support["source_row_index"][positions].astype(np.int64)
        if not np.all(
            context_supported(
                targets,
                raw["engine_frame_idx"],
                window=model.window,
                frame_stride=model.frame_stride,
                window_mode=model.window_mode,
            )
        ):
            raise RuntimeError(f"support lacks checkpoint context: {session_id}")
        if model.precomputed_features:
            assert encoder is not None
            cache_path = (
                args.feature_cache_root
                / video_id_from_session(session_id)
                / f"{session_id}.npy"
                if args.feature_cache_root is not None
                else None
            )
            features = load_or_encode_features(
                encoder, raw["frames"], cache_path
            )
            predicted = feature_predictions(
                model,
                features,
                targets,
                batch_size=args.model_batch_size,
                device=device,
            )
        else:
            predicted = pixel_predictions(
                model,
                raw["frames"],
                targets,
                dense_target_count=args.pixel_dense_targets,
                device=device,
            )
        probability[positions] = predicted
        if session_number % 100 == 0 or session_number == len(required_sessions):
            print(
                f"evaluated_sessions={session_number}/{len(required_sessions)}",
                flush=True,
            )
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("prediction array is incomplete or nonfinite")
    args.preds_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.preds_out,
        y_true=support["y_true"],
        y_prob=probability,
        input_active=support["input_active"],
        session_lengths=support["session_lengths"],
        session_ids=support["session_ids"],
        source_row_index=support["source_row_index"],
        source_engine_frame_idx=support["source_engine_frame_idx"],
    )
    gate = support["input_active"].astype(bool)
    metrics = key_metrics(support["y_true"][gate], probability[gate])
    report = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "surface": "Wild admitted7 natural GRU128x3 common support; no padding",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha,
            "weights": args.weights,
            "steps": checkpoint.get("steps"),
            "best_val_step": checkpoint.get("best_val_step"),
            "window": model.window,
            "frame_stride": model.frame_stride,
            "precomputed_features": model.precomputed_features,
        },
        "data": {
            "manifest_sha256": sha256_file(args.manifest),
            "reference_sidecar_sha256": sha256_file(args.reference_sidecar),
            "support_sidecar_sha256": sha256_file(args.support_sidecar),
            "source_verification_sha256": sha256_file(args.source_verification),
            "rows": int(len(probability)),
            "active_rows": int(gate.sum()),
            "streams": int(len(support["session_lengths"])),
        },
        "row_weighted": metrics,
        "micro_key_accuracy": metrics["micro_key_accuracy"],
        "prediction_sidecar": {
            "path": str(args.preds_out),
            "bytes": args.preds_out.stat().st_size,
            "sha256": sha256_file(args.preds_out),
        },
        "config": config,
        "execution": {
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "precision": "float32",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def rescore(args: argparse.Namespace) -> dict[str, Any]:
    reference = load_reference(args.reference_sidecar)
    support = load_support(args.support_sidecar, reference)
    indices = support["reference_indices"].astype(np.int64, copy=False)
    with np.load(args.prediction_sidecar, allow_pickle=False) as archive:
        required = {"y_true", "y_prob", "input_active"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"prediction sidecar lacks {sorted(missing)}")
        truth = np.asarray(archive["y_true"])
        probability = np.asarray(archive["y_prob"])
        active = np.asarray(archive["input_active"])
    if not np.array_equal(truth, reference["y_true"]):
        raise RuntimeError("prediction/reference truth differs")
    if not np.array_equal(active, reference["input_active"]):
        raise RuntimeError("prediction/reference activity differs")
    gate = active[indices].astype(bool)
    metrics = key_metrics(truth[indices][gate], probability[indices][gate])
    result = {
        "schema_version": "madeleine.wild7-common-support-rescore.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prediction_sidecar_sha256": sha256_file(args.prediction_sidecar),
        "reference_sidecar_sha256": sha256_file(args.reference_sidecar),
        "support_sidecar_sha256": sha256_file(args.support_sidecar),
        "rows": int(len(indices)),
        "active_rows": int(gate.sum()),
        "row_weighted": metrics,
        "micro_key_accuracy": metrics["micro_key_accuracy"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    support = commands.add_parser("build-support")
    support.add_argument("--manifest", type=Path, required=True)
    support.add_argument("--raw-root", type=Path, required=True)
    support.add_argument("--source-verification", type=Path, required=True)
    support.add_argument("--reference-sidecar", type=Path, required=True)
    support.add_argument("--window", type=int, default=128)
    support.add_argument("--frame-stride", type=int, default=3)
    support.add_argument("--out", type=Path, required=True)
    support.add_argument("--receipt", type=Path, required=True)

    run = commands.add_parser("evaluate")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--expected-checkpoint-sha256", required=True)
    run.add_argument("--weights", choices=("selected", "final"), default="final")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--raw-root", type=Path, required=True)
    run.add_argument("--source-verification", type=Path, required=True)
    run.add_argument("--reference-sidecar", type=Path, required=True)
    run.add_argument("--support-sidecar", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--preds-out", type=Path, required=True)
    run.add_argument("--encoder-batch-size", type=int, default=512)
    run.add_argument("--model-batch-size", type=int, default=128)
    run.add_argument("--pixel-dense-targets", type=int, default=512)
    run.add_argument("--feature-cache-root", type=Path)
    run.add_argument("--device", default="cuda")

    score = commands.add_parser("rescore")
    score.add_argument("--prediction-sidecar", type=Path, required=True)
    score.add_argument("--reference-sidecar", type=Path, required=True)
    score.add_argument("--support-sidecar", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build-support":
        result = build_support(args)
    elif args.command == "evaluate":
        result = evaluate(args)
    else:
        result = rescore(args)
    print(json.dumps(json_ready(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
