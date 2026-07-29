#!/usr/bin/env python3
"""Dense center-64 evaluation for VPT-small 20 Hz phase manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from badeline.metrics import summarize
from badeline.vpt_small import VPTSmallConfig, VPTSmallIDM, maybe_autocast
from data.schema import KEY_ORDER
from experiments.keypress_accuracy import score_sidecar


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def equal_mass_ece(truth: np.ndarray, probability: np.ndarray, bins: int = 15) -> dict[str, float]:
    result: dict[str, float] = {}
    for column, key in enumerate(KEY_ORDER):
        order = np.argsort(probability[:, column], kind="stable")
        ece = 0.0
        for members in np.array_split(order, bins):
            if not len(members):
                continue
            confidence = float(probability[members, column].mean())
            frequency = float(truth[members, column].mean())
            ece += len(members) / len(order) * abs(confidence - frequency)
        result[key] = ece
    return result


def flatten_center_probabilities(blocks: list[np.ndarray]) -> np.ndarray:
    """Flatten dense center-64 window outputs into one prediction per row."""

    probability = np.concatenate(blocks, axis=0)
    expected_tail = (64, len(KEY_ORDER))
    if probability.ndim != 3 or probability.shape[1:] != expected_tail:
        raise RuntimeError(
            "center probability blocks must concatenate to "
            f"[windows,64,{len(KEY_ORDER)}], got {probability.shape}"
        )
    return probability.reshape(-1, len(KEY_ORDER))


def infer_stream(
    model: VPTSmallIDM,
    *,
    directory: Path,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    frames = np.load(directory / "frames.npy", mmap_mode="r", allow_pickle=False)
    keys = np.load(directory / "keys.npy", mmap_mode="r", allow_pickle=False)
    active = np.load(directory / "input_active.npy", mmap_mode="r", allow_pickle=False)
    engine_idx = np.load(
        directory / "source_engine_frame_idx.npy", mmap_mode="r", allow_pickle=False
    )
    source_rows = np.load(
        directory / "source_row_index.npy", mmap_mode="r", allow_pickle=False
    )
    continuity = np.load(
        directory / "continuity_id.npy", mmap_mode="r", allow_pickle=False
    )
    starts = np.load(directory / "window_start.npy", allow_pickle=False)
    probabilities: list[np.ndarray] = []
    selected_rows: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            block = np.stack(
                [np.array(frames[int(start) : int(start) + 128], copy=True) for start in batch_starts]
            )
            tensor = (
                torch.from_numpy(block)
                .permute(0, 1, 4, 2, 3)
                .to(device=device, dtype=torch.float32)
                .div_(255.0)
            )
            with maybe_autocast(device, dtype):
                logits = model(tensor)
            probabilities.append(
                F.softmax(logits[:, 32:96], dim=-1)[..., 1].float().cpu().numpy()
            )
            selected_rows.extend(
                np.arange(int(start) + 32, int(start) + 96, dtype=np.int64)
                for start in batch_starts
            )
    if not probabilities:
        raise ValueError(f"no center-supported windows in {directory}")
    rows = np.concatenate(selected_rows)
    probability = flatten_center_probabilities(probabilities)
    if len(rows) != len(np.unique(rows)):
        raise RuntimeError(f"center-64 reconstruction overlaps in {directory}")
    if len(probability) != len(rows):
        raise RuntimeError("center probability rows do not match reconstructed rows")
    return {
        "source_row": np.asarray(source_rows[rows]),
        "engine_idx": np.asarray(engine_idx[rows]),
        "continuity": np.asarray(continuity[rows]),
        "truth": np.asarray(keys[rows]),
        "active": np.asarray(active[rows]),
        "probability": probability,
    }


def combine_phases(
    manifest: dict[str, Any], root: Path, inferred: list[dict[str, np.ndarray]]
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, np.ndarray]]] = defaultdict(list)
    for record, arrays in zip(manifest["records"], inferred, strict=True):
        for continuity_id in np.unique(arrays["continuity"]):
            selected = arrays["continuity"] == continuity_id
            grouped[(str(record["session_id"]), int(continuity_id))].append(
                {key: value[selected] for key, value in arrays.items()}
            )

    truth: list[np.ndarray] = []
    probability: list[np.ndarray] = []
    active: list[np.ndarray] = []
    source_rows: list[np.ndarray] = []
    engine_idx: list[np.ndarray] = []
    stream_lengths: list[int] = []
    stream_ids: list[str] = []
    for (session_id, continuity_id), phase_parts in grouped.items():
        merged = {
            key: np.concatenate([part[key] for part in phase_parts])
            for key in phase_parts[0]
            if key != "continuity"
        }
        order = np.argsort(merged["source_row"], kind="stable")
        for key in merged:
            merged[key] = merged[key][order]
        if len(np.unique(merged["source_row"])) != len(merged["source_row"]):
            raise RuntimeError(f"duplicate interleaved target rows for {session_id}")
        boundaries = np.flatnonzero(np.diff(merged["source_row"]) != 1) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(merged["source_row"])]))
        for subrun, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if end <= start:
                continue
            truth.append(merged["truth"][start:end])
            probability.append(merged["probability"][start:end])
            active.append(merged["active"][start:end])
            source_rows.append(merged["source_row"][start:end])
            engine_idx.append(merged["engine_idx"][start:end])
            stream_lengths.append(int(end - start))
            stream_ids.append(f"{session_id}__run{continuity_id:03d}__sub{subrun:03d}")
    return {
        "truth": np.concatenate(truth),
        "probability": np.concatenate(probability),
        "active": np.concatenate(active),
        "source_row": np.concatenate(source_rows),
        "engine_idx": np.concatenate(engine_idx),
        "stream_lengths": np.asarray(stream_lengths, dtype=np.int64),
        "stream_ids": np.asarray(stream_ids),
    }


def evaluate(
    checkpoint_path: Path,
    manifest_path: Path,
    sidecar_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    checkpoint_hash = sha256_file(checkpoint_path)
    if expected_checkpoint_sha256 and checkpoint_hash != expected_checkpoint_sha256:
        raise RuntimeError("checkpoint SHA-256 does not match the frozen registry")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "madeleine.vpt-small-checkpoint.v1":
        raise ValueError("unsupported VPT-small checkpoint")
    model_config = VPTSmallConfig.from_dict(checkpoint["config"]["model"])
    model = VPTSmallIDM(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sorted(manifest["phases"]) != [0, 1, 2]:
        raise ValueError("primary native-rate evaluation requires all phases 0,1,2")
    inferred = []
    for record in manifest["records"]:
        directory = manifest_path.parent / f"{record['session_id']}__p{record['phase']}"
        inferred.append(
            infer_stream(
                model,
                directory=directory,
                batch_size=batch_size,
                device=device,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            )
        )
    combined = combine_phases(manifest, manifest_path.parent, inferred)
    truth = combined["truth"].astype(np.uint8, copy=False)
    probability = combined["probability"].astype(np.float32, copy=False)
    active = combined["active"].astype(np.uint8, copy=False)
    if truth.shape != probability.shape or truth.shape[1] != len(KEY_ORDER):
        raise RuntimeError("aligned truth/probability shapes differ")
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("prediction sidecar contains nonfinite values")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sidecar_path,
        y_true=truth,
        y_prob=probability,
        input_active=active,
        session_lengths=combined["stream_lengths"],
        session_ids=combined["stream_ids"],
        source_row_index=combined["source_row"],
        source_engine_frame_idx=combined["engine_idx"],
    )
    gate = active.astype(bool)
    gated_truth = truth[gate]
    gated_probability = probability[gate]
    eps = 1e-7
    clipped = np.clip(gated_probability, eps, 1.0 - eps)
    natural_nll = float(
        np.mean(
            -(gated_truth * np.log(clipped) + (1 - gated_truth) * np.log(1 - clipped)),
            axis=0,
        ).sum()
    )
    brier = float(np.mean((gated_probability - gated_truth) ** 2))
    metric_detail = summarize(
        truth,
        probability,
        boundaries=combined["stream_lengths"].tolist(),
        active=gate,
        fixed_transition_thresholds={key: 0.5 for key in KEY_ORDER},
        include_oracle=False,
    )
    predicted = gated_probability >= 0.5
    true_positive = np.logical_and(predicted, gated_truth == 1).sum(axis=0)
    false_positive = np.logical_and(predicted, gated_truth == 0).sum(axis=0)
    false_negative = np.logical_and(~predicted, gated_truth == 1).sum(axis=0)
    precision = true_positive / np.maximum(1, true_positive + false_positive)
    recall = true_positive / np.maximum(1, true_positive + false_negative)
    state_f1 = 2 * precision * recall / np.maximum(1e-12, precision + recall)
    aggregate = {
        "prevalence_macro": float(gated_truth.mean(axis=0).mean()),
        "prevalence_per_key": {
            key: float(gated_truth[:, column].mean())
            for column, key in enumerate(KEY_ORDER)
        },
        "predicted_positive_rate_macro": float(predicted.mean(axis=0).mean()),
        "predicted_positive_rate_per_key": {
            key: float(predicted[:, column].mean())
            for column, key in enumerate(KEY_ORDER)
        },
        "macro_ap": float(np.nanmean(list(metric_detail["per_key_ap"].values()))),
        "macro_state_f1": float(np.nanmean(state_f1)),
        "macro_state_precision": float(np.nanmean(precision)),
        "macro_state_recall": float(np.nanmean(recall)),
        "macro_event_f1_collar_0": float(np.nanmean([
            metric_detail["transition_f1_at_0.5"][key]["event"]["f1"]
            for key in KEY_ORDER
        ])),
        "macro_event_f1_collar_2_native_frames": float(np.nanmean([
            metric_detail["transition_f1_at_0.5_collars"]["2"][key]["event"]["f1"]
            for key in KEY_ORDER
        ])),
    }
    report = {
        "schema_version": "madeleine.vpt-small-eval.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "surface": "corrected own-v3 val-A, three-phase VPT common support",
        "threshold": 0.5,
        "weights": {
            "checkpoint": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "epoch": checkpoint["epoch"],
            "optimizer_step": checkpoint["optimizer_step"],
        },
        "data": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "phases": manifest["phases"],
            "rows": len(truth),
            "active_rows": int(gate.sum()),
            "streams": len(combined["stream_lengths"]),
        },
        "natural_nll": natural_nll,
        "brier": brier,
        "equal_mass_ece": equal_mass_ece(gated_truth, gated_probability),
        "aggregate": aggregate,
        "metrics": metric_detail,
        "sidecar": {
            "path": str(sidecar_path),
            "bytes": sidecar_path.stat().st_size,
            "sha256": sha256_file(sidecar_path),
        },
    }
    report["key_state_accuracy"] = score_sidecar(sidecar_path, threshold=0.5)
    return json_ready(report)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preds-out", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate(
        args.checkpoint,
        args.manifest,
        args.preds_out,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
