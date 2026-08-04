#!/usr/bin/env python3
"""Score a deterministic VPT-small prediction sidecar on the Wild holdout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from data.schema import KEY_ORDER
from experiments.summarize_vpt_small_wild_eval import (
    key_metrics,
    sha256_file,
    video_id_from_stream_id,
)


def score(
    *,
    contract_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    sidecar_path: Path,
    repeat_sidecar_path: Path,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checkpoint_sha256 = sha256_file(checkpoint_path)
    manifest_sha256 = sha256_file(manifest_path)
    sidecar_sha256 = sha256_file(sidecar_path)
    repeat_sidecar_sha256 = sha256_file(repeat_sidecar_path)
    if checkpoint_sha256 != contract["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint SHA-256 differs from the evaluation contract")
    population = contract["evaluation_population"]
    if manifest_sha256 != population["build_manifest_sha256"]:
        raise RuntimeError("Wild manifest SHA-256 differs from the evaluation contract")
    if sidecar_sha256 != repeat_sidecar_sha256:
        raise RuntimeError("repeated inference sidecars are not byte-identical")

    with np.load(sidecar_path, allow_pickle=False) as archive:
        required = {
            "y_true",
            "y_prob",
            "input_active",
            "session_lengths",
            "session_ids",
            "source_row_index",
            "source_engine_frame_idx",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"prediction sidecar lacks {sorted(missing)}")
        truth = np.asarray(archive["y_true"], dtype=np.uint8)
        probability = np.asarray(archive["y_prob"], dtype=np.float32)
        active = np.asarray(archive["input_active"], dtype=np.uint8)
        lengths = np.asarray(archive["session_lengths"], dtype=np.int64)
        stream_ids = np.asarray(archive["session_ids"]).astype(str)
        source_rows = np.asarray(archive["source_row_index"], dtype=np.int64)
        engine_rows = np.asarray(archive["source_engine_frame_idx"], dtype=np.int64)
    if truth.shape != probability.shape or truth.ndim != 2:
        raise RuntimeError("truth and probability sidecar arrays differ")
    if truth.shape[1] != len(KEY_ORDER):
        raise RuntimeError("prediction sidecar does not contain seven keys")
    if len(truth) != int(population["expected_center_supported_rows"]):
        raise RuntimeError("prediction sidecar row count differs from the contract")
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("prediction sidecar contains nonfinite probabilities")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise RuntimeError("prediction probabilities lie outside [0,1]")
    if not np.all(np.isin(truth, (0, 1))) or not np.all(np.isin(active, (0, 1))):
        raise RuntimeError("truth or active arrays are not binary")
    if int(lengths.sum()) != len(truth) or len(lengths) != len(stream_ids):
        raise RuntimeError("stream metadata does not cover the sidecar rows")
    if len(source_rows) != len(truth) or len(engine_rows) != len(truth):
        raise RuntimeError("source identity arrays do not cover the sidecar rows")

    gate = active.astype(bool)
    gated_truth = truth[gate]
    gated_probability = probability[gate]
    row_weighted = key_metrics(gated_truth, gated_probability)
    eps = 1e-7
    clipped = np.clip(gated_probability, eps, 1.0 - eps)
    natural_nll = float(
        np.mean(
            -(gated_truth * np.log(clipped) + (1 - gated_truth) * np.log(1 - clipped)),
            axis=0,
        ).sum()
    )
    brier = float(np.mean((gated_probability - gated_truth) ** 2))

    grouped: dict[str, list[np.ndarray]] = {}
    offset = 0
    for stream_id, length in zip(stream_ids, lengths, strict=True):
        stop = offset + int(length)
        grouped.setdefault(video_id_from_stream_id(stream_id), []).append(
            np.arange(offset, stop, dtype=np.int64)
        )
        offset = stop
    expected_videos = set(population["videos"])
    if set(grouped) != expected_videos:
        raise RuntimeError("evaluated Wild video membership differs from the contract")

    per_video: dict[str, Any] = {}
    for video_id, blocks in sorted(grouped.items()):
        indices = np.concatenate(blocks)
        selected = indices[gate[indices]]
        metrics = key_metrics(truth[selected], probability[selected])
        metrics["streams"] = len(blocks)
        metrics["active_hours_at_20hz"] = len(selected) / 20.0 / 3600.0
        per_video[video_id] = metrics
    video_macro_ap = [
        float(metrics["macro_ap"])
        for metrics in per_video.values()
        if metrics["macro_ap"] is not None
    ]
    equal_video_per_key: dict[str, float | None] = {}
    for key in KEY_ORDER:
        values = [
            metrics["per_key_ap"][key]
            for metrics in per_video.values()
            if metrics["per_key_ap"][key] is not None
        ]
        equal_video_per_key[key] = float(np.mean(values)) if values else None

    return {
        "schema_version": "madeleine.vpt-small-wild-holdout-score.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "surface": "admitted Wild7 public-gameplay holdout; masked controller HUD; 20 Hz phase0 deployment support",
        "threshold": 0.5,
        "threshold_tuned": False,
        "contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
        },
        "checkpoint": {
            **contract["checkpoint"],
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
        },
        "data": {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "videos": len(per_video),
            "streams": len(lengths),
            "rows": len(truth),
            "active_rows": int(gate.sum()),
            "active_hours_at_20hz": int(gate.sum()) / 20.0 / 3600.0,
            "support_mode": contract["protocol"]["support_mode"],
        },
        "prediction_sidecar": {
            "path": str(sidecar_path),
            "sha256": sidecar_sha256,
            "repeat_path": str(repeat_sidecar_path),
            "repeat_sha256": repeat_sidecar_sha256,
            "byte_identical_repeat": True,
        },
        "natural_nll": natural_nll,
        "brier": brier,
        "row_weighted": row_weighted,
        "equal_video": {
            "macro_ap": float(np.mean(video_macro_ap)),
            "per_key_ap": equal_video_per_key,
        },
        "per_video": per_video,
        "execution": {
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started,
        },
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--repeat-sidecar", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = score(
        contract_path=args.contract,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        sidecar_path=args.sidecar,
        repeat_sidecar_path=args.repeat_sidecar,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
