#!/usr/bin/env python3
"""Evaluate a native-60-Hz VPT-small checkpoint on frozen Wild7 rows.

The raw Wild shards stay at 60 Hz for inference. Predictions are then selected
by ``(session_id, source_row_index)`` onto an existing, immutable 20 Hz Wild7
sidecar so every model is scored on exactly the same labeled rows.
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
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

from badeline.vpt_small import VPTSmallConfig, VPTSmallIDM, maybe_autocast
from data.schema import KEY_ORDER
from experiments.build_vpt_small_128px_20hz import selected_rows


SCHEMA = "madeleine.vpt-small-native60-wild7-eval.v1"
VERIFY_SCHEMA = "madeleine.vpt-small-native60-wild7-source-verification.v1"


def key_metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    predicted = probability >= 0.5
    per_key_ap: dict[str, float | None] = {}
    prevalence: dict[str, float] = {}
    state_f1: dict[str, float] = {}
    for column, key in enumerate(KEY_ORDER):
        key_truth = truth[:, column]
        key_probability = probability[:, column]
        positives = int(key_truth.sum())
        prevalence[key] = float(key_truth.mean())
        per_key_ap[key] = (
            float(average_precision_score(key_truth, key_probability))
            if positives and positives < len(key_truth)
            else None
        )
        key_predicted = predicted[:, column]
        true_positive = int(np.logical_and(key_predicted, key_truth == 1).sum())
        false_positive = int(np.logical_and(key_predicted, key_truth == 0).sum())
        false_negative = int(np.logical_and(~key_predicted, key_truth == 1).sum())
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        state_f1[key] = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    finite_ap = [value for value in per_key_ap.values() if value is not None]
    return {
        "rows": int(len(truth)),
        "per_key_ap": per_key_ap,
        "macro_ap": float(np.mean(finite_ap)) if finite_ap else None,
        "prevalence_per_key": prevalence,
        "macro_state_f1": float(np.mean(list(state_f1.values()))),
        "state_f1_per_key": state_f1,
        "micro_key_accuracy": float((predicted == truth).mean()),
        "joint_key_accuracy": float(np.all(predicted == truth, axis=1).mean()),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def video_id_from_session(session_id: str) -> str:
    if not session_id.startswith("wild_") or "__r" not in session_id:
        raise ValueError(f"unexpected Wild session id: {session_id}")
    return session_id[len("wild_") :].split("__r", 1)[0]


def source_path(raw_root: Path, session_id: str) -> Path:
    return raw_root / video_id_from_session(session_id) / f"{session_id}.npz"


def load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("derived_rate_hz") != 20 or manifest.get("phases") != [0]:
        raise ValueError("reference manifest must be Wild7 deployment phase0 at 20 Hz")
    records = {str(record["session_id"]): record for record in manifest["records"]}
    if len(records) != len(manifest["records"]):
        raise RuntimeError("reference manifest contains duplicate sessions")
    return manifest, records


def verify_sources(manifest_path: Path, raw_root: Path, out: Path) -> dict[str, Any]:
    _, records = load_manifest(manifest_path)
    verified: list[dict[str, Any]] = []
    for index, (session_id, record) in enumerate(records.items(), start=1):
        path = source_path(raw_root, session_id)
        expected = record["source"]
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != int(expected["bytes"]):
            raise RuntimeError(f"source byte mismatch: {path}")
        digest = sha256_file(path)
        if digest != expected["sha256"]:
            raise RuntimeError(f"source SHA-256 mismatch: {path}")
        verified.append(
            {
                "session_id": session_id,
                "path": str(path),
                "bytes": size,
                "sha256": digest,
            }
        )
        if index % 100 == 0 or index == len(records):
            print(f"verified_sources={index}/{len(records)}", flush=True)
    result = {
        "schema_version": VERIFY_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "raw_root": str(raw_root),
        "objects": len(verified),
        "bytes": sum(int(item["bytes"]) for item in verified),
        "files": verified,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def load_verification(
    path: Path,
    *,
    manifest_path: Path,
    raw_root: Path,
    records: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != VERIFY_SCHEMA:
        raise ValueError("unsupported source-verification receipt")
    if receipt.get("manifest", {}).get("sha256") != sha256_file(manifest_path):
        raise RuntimeError("source verification names a different manifest")
    if Path(receipt.get("raw_root", "")) != raw_root:
        raise RuntimeError("source verification names a different raw root")
    files = {str(item["session_id"]): item for item in receipt.get("files", [])}
    if set(files) != set(records):
        raise RuntimeError("source-verification membership differs from manifest")
    for session_id, record in records.items():
        item = files[session_id]
        expected = record["source"]
        if item.get("sha256") != expected["sha256"] or int(item.get("bytes", -1)) != int(expected["bytes"]):
            raise RuntimeError(f"source-verification identity differs: {session_id}")
        if Path(item.get("path", "")) != source_path(raw_root, session_id):
            raise RuntimeError(f"source-verification path differs: {session_id}")
    return files


def load_reference(path: Path) -> dict[str, np.ndarray]:
    required = {
        "y_true",
        "y_prob",
        "input_active",
        "session_lengths",
        "session_ids",
        "source_row_index",
        "source_engine_frame_idx",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"reference sidecar lacks {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    lengths = arrays["session_lengths"].astype(np.int64, copy=False)
    stream_ids = arrays["session_ids"].astype(str, copy=False)
    if int(lengths.sum()) != len(arrays["y_true"]):
        raise RuntimeError("reference stream lengths do not cover its rows")
    arrays["row_session_id"] = np.concatenate(
        [
            np.full(int(length), stream_id.split("__run", 1)[0])
            for stream_id, length in zip(stream_ids, lengths, strict=True)
        ]
    )
    return arrays


def load_checkpoint(
    path: Path, expected_sha256: str, device: torch.device
) -> tuple[VPTSmallIDM, dict[str, Any], str]:
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise RuntimeError("checkpoint SHA-256 differs from expected value")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "madeleine.vpt-small-checkpoint.v1":
        raise ValueError("unsupported VPT-small checkpoint")
    config = VPTSmallConfig.from_dict(checkpoint["config"]["model"])
    if config.frames not in (128, 384):
        raise ValueError("native60 Wild7 evaluator supports 128 or 384 frames")
    model = VPTSmallIDM(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint, digest


def completed_window_starts(
    runs: list[tuple[int, int]], *, window: int, stride: int
) -> np.ndarray:
    """Return base starts plus one end-aligned fill window per run."""

    per_run_starts: list[np.ndarray] = []
    for start, end in runs:
        if end - start < window:
            continue
        base = np.arange(start, end - window + 1, stride, dtype=np.int64)
        tail = end - window
        if int(base[-1]) != tail:
            base = np.concatenate((base, np.asarray([tail], dtype=np.int64)))
        per_run_starts.append(base)
    return (
        np.concatenate(per_run_starts)
        if per_run_starts
        else np.empty(0, dtype=np.int64)
    )


def inference_window_slices(
    runs: list[tuple[int, int]], *, window: int, stride: int
) -> list[tuple[int, int]]:
    """Return full windows plus minimally tail-padded short-context windows.

    A phase-0 20 Hz stream can contain a complete 128-frame evaluation window
    when its native-60 Hz source has only 382 or 383 rows.  The 384-frame arm
    still needs predictions for the shared center support, so those final one
    or two context rows are repeated without inventing additional scored rows.
    """

    retain_end = (window + stride) // 2
    slices = [
        (int(start), int(start) + window)
        for start in completed_window_starts(runs, window=window, stride=stride)
    ]
    slices.extend(
        (int(start), int(end))
        for start, end in runs
        if retain_end <= end - start < window
    )
    return slices


def infer_session(
    model: VPTSmallIDM,
    path: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"frames", "keys", "engine_frame_idx", "input_active", "session_id"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"raw Wild shard lacks {sorted(missing)}: {path}")
        frames = np.asarray(archive["frames"])
        truth = np.asarray(archive["keys"])
        engine_idx = np.asarray(archive["engine_frame_idx"])
        active = np.asarray(archive["input_active"])
        session_id = str(np.asarray(archive["session_id"]).item())
    if session_id != path.stem:
        raise RuntimeError(f"session identity differs: {path}")
    if frames.dtype != np.uint8 or frames.shape[1:] != (128, 128, 3):
        raise ValueError(f"unexpected frame array: {path}")
    if truth.dtype != np.uint8 or truth.shape != (len(frames), len(KEY_ORDER)):
        raise ValueError(f"unexpected truth array: {path}")
    if engine_idx.dtype != np.int64 or active.shape != (len(frames),):
        raise ValueError(f"unexpected identity arrays: {path}")

    rows, continuity, runs = selected_rows(engine_idx, 0, row_step=1)
    window = int(model.config.frames)
    stride = window // 2
    window_slices = inference_window_slices(runs, window=window, stride=stride)
    if not window_slices:
        return {
            "source_row": np.empty(0, dtype=np.int64),
            "engine_idx": np.empty(0, dtype=np.int64),
            "truth": np.empty((0, len(KEY_ORDER)), dtype=np.uint8),
            "active": np.empty(0, dtype=np.uint8),
            "probability": np.empty((0, len(KEY_ORDER)), dtype=np.float32),
        }
    begin = (window - stride) // 2
    end = begin + stride
    probabilities: list[np.ndarray] = []
    center_positions: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(window_slices), batch_size):
            batch_slices = window_slices[offset : offset + batch_size]
            blocks = []
            for start, stop in batch_slices:
                item = frames[rows[start:stop]]
                if len(item) < window:
                    item = np.pad(
                        item,
                        ((0, window - len(item)), (0, 0), (0, 0), (0, 0)),
                        mode="edge",
                    )
                blocks.append(item)
            block = np.stack(blocks)
            tensor = (
                torch.from_numpy(block)
                .permute(0, 1, 4, 2, 3)
                .to(device=device, dtype=torch.float32)
                .div_(255.0)
            )
            with maybe_autocast(device, torch.bfloat16):
                logits = model(tensor)
            probabilities.append(
                F.softmax(logits[:, begin:end], dim=-1)[..., 1]
                .float()
                .cpu()
                .numpy()
                .reshape(-1, len(KEY_ORDER))
            )
            center_positions.extend(
                np.arange(start + begin, start + end, dtype=np.int64)
                for start, _ in batch_slices
            )
    positions = np.concatenate(center_positions)
    probability = np.concatenate(probabilities)
    if len(positions) != len(probability):
        raise RuntimeError(f"native60 center reconstruction changed: {path}")
    if len(positions) != len(np.unique(positions)):
        _, first = np.unique(positions, return_index=True)
        keep = np.sort(first)
        positions = positions[keep]
        probability = probability[keep]
    source = rows[positions]
    return {
        "source_row": source,
        "engine_idx": engine_idx[source],
        "truth": truth[source],
        "active": active[source],
        "probability": probability,
    }


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
    reference_sessions = reference["row_session_id"].astype(str)
    required_sessions = set(reference_sessions.tolist())
    if not required_sessions.issubset(records):
        raise RuntimeError("reference contains sessions outside the frozen manifest")
    device = torch.device(args.device)
    model, checkpoint, checkpoint_sha = load_checkpoint(
        args.checkpoint, args.expected_checkpoint_sha256, device
    )
    probability = np.full(reference["y_true"].shape, np.nan, dtype=np.float32)
    for index, session_id in enumerate(sorted(required_sessions), start=1):
        candidate = infer_session(
            model,
            source_path(args.raw_root, session_id),
            device=device,
            batch_size=args.batch_size,
        )
        candidate_rows = candidate["source_row"].astype(np.int64, copy=False)
        if len(candidate_rows) != len(np.unique(candidate_rows)):
            raise RuntimeError(f"candidate contains duplicate source rows: {session_id}")
        lookup = {int(row): position for position, row in enumerate(candidate_rows)}
        selected_global = np.flatnonzero(reference_sessions == session_id)
        requested_rows = reference["source_row_index"][selected_global].astype(np.int64)
        missing = [int(row) for row in requested_rows if int(row) not in lookup]
        if missing:
            raise RuntimeError(
                f"native60 candidate lacks {len(missing)} frozen rows: {session_id}"
            )
        selected_candidate = np.asarray([lookup[int(row)] for row in requested_rows])
        checks = {
            "truth": "y_true",
            "active": "input_active",
            "engine_idx": "source_engine_frame_idx",
        }
        for candidate_name, reference_name in checks.items():
            if not np.array_equal(
                candidate[candidate_name][selected_candidate],
                reference[reference_name][selected_global],
            ):
                raise RuntimeError(
                    f"native/reference {candidate_name} differs: {session_id}"
                )
        probability[selected_global] = candidate["probability"][selected_candidate]
        if index % 100 == 0 or index == len(required_sessions):
            print(f"evaluated_sessions={index}/{len(required_sessions)}", flush=True)
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("native60 prediction array is incomplete or nonfinite")

    args.preds_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.preds_out,
        y_true=reference["y_true"],
        y_prob=probability,
        input_active=reference["input_active"],
        session_lengths=reference["session_lengths"],
        session_ids=reference["session_ids"],
        source_row_index=reference["source_row_index"],
        source_engine_frame_idx=reference["source_engine_frame_idx"],
    )
    gate = reference["input_active"].astype(bool)
    metrics = key_metrics(reference["y_true"][gate], probability[gate])
    predicted = probability[gate] >= 0.5
    report = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "surface": "Wild admitted7 identical 842624-row support; native60 inference",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha,
            "epoch": checkpoint.get("epoch"),
            "optimizer_step": checkpoint.get("optimizer_step"),
            "frames": model.config.frames,
        },
        "data": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "reference_sidecar": str(args.reference_sidecar),
            "reference_sidecar_sha256": sha256_file(args.reference_sidecar),
            "source_verification": str(args.source_verification),
            "source_verification_sha256": sha256_file(args.source_verification),
            "rows": len(probability),
            "active_rows": int(gate.sum()),
            "sessions": len(required_sessions),
            "native_rate_hz": 60,
        },
        "row_weighted": metrics,
        "micro_key_accuracy": float(
            np.mean(predicted == reference["y_true"][gate].astype(bool))
        ),
        "prediction_sidecar": {
            "path": str(args.preds_out),
            "bytes": args.preds_out.stat().st_size,
            "sha256": sha256_file(args.preds_out),
        },
        "execution": {
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "precision": "torch.bfloat16",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-sources")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--raw-root", type=Path, required=True)
    verify.add_argument("--out", type=Path, required=True)
    run = subparsers.add_parser("evaluate")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--expected-checkpoint-sha256", required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--raw-root", type=Path, required=True)
    run.add_argument("--source-verification", type=Path, required=True)
    run.add_argument("--reference-sidecar", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--preds-out", type=Path, required=True)
    run.add_argument("--batch-size", type=int, default=2)
    run.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "verify-sources":
        result = verify_sources(args.manifest, args.raw_root, args.out)
    else:
        result = evaluate(args)
    print(json.dumps(json_ready(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
