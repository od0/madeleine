#!/usr/bin/env python3
"""Summarize a VPT-small Wild evaluation by source video."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score

from data.schema import KEY_ORDER


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_id_from_stream_id(stream_id: str) -> str:
    session_id = stream_id.split("__run", 1)[0]
    if not session_id.startswith("wild_") or "__r" not in session_id:
        raise ValueError(f"unexpected Wild stream ID: {stream_id}")
    return session_id.removeprefix("wild_").split("__r", 1)[0]


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


def summarize(report_path: Path, sidecar_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sidecar_sha256 = sha256_file(sidecar_path)
    if report.get("sidecar", {}).get("sha256") != sidecar_sha256:
        raise RuntimeError("evaluation report does not bind the supplied sidecar")
    with np.load(sidecar_path, allow_pickle=False) as archive:
        truth = np.asarray(archive["y_true"], dtype=np.uint8)
        probability = np.asarray(archive["y_prob"], dtype=np.float32)
        active = np.asarray(archive["input_active"], dtype=np.uint8).astype(bool)
        lengths = np.asarray(archive["session_lengths"], dtype=np.int64)
        stream_ids = np.asarray(archive["session_ids"]).astype(str)
    if truth.shape != probability.shape or truth.shape[1] != len(KEY_ORDER):
        raise RuntimeError("truth and probability arrays are not aligned seven-key rows")
    if len(truth) != len(active) or int(lengths.sum()) != len(truth):
        raise RuntimeError("stream lengths do not cover the evaluation rows")
    if len(lengths) != len(stream_ids):
        raise RuntimeError("stream ID and length counts differ")

    grouped: dict[str, list[np.ndarray]] = {}
    offset = 0
    for stream_id, length in zip(stream_ids, lengths, strict=True):
        stop = offset + int(length)
        grouped.setdefault(video_id_from_stream_id(stream_id), []).append(
            np.arange(offset, stop, dtype=np.int64)
        )
        offset = stop

    per_video: dict[str, Any] = {}
    for video_id, index_blocks in sorted(grouped.items()):
        indices = np.concatenate(index_blocks)
        selected = indices[active[indices]]
        metrics = key_metrics(truth[selected], probability[selected])
        metrics["streams"] = len(index_blocks)
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
        "schema_version": "madeleine.vpt-small-wild-eval-summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
        },
        "prediction_sidecar": {
            "path": str(sidecar_path),
            "sha256": sidecar_sha256,
        },
        "videos": len(per_video),
        "row_weighted": report["aggregate"],
        "equal_video": {
            "macro_ap": float(np.mean(video_macro_ap)),
            "per_key_ap": equal_video_per_key,
        },
        "per_video": per_video,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = summarize(args.report, args.sidecar)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
