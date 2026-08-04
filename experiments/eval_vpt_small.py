#!/usr/bin/env python3
"""Dense center-region evaluation for VPT-small temporal-rate manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import time
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


def retained_positions(window: int, stride: int) -> tuple[int, int]:
    """Return the non-overlapping center region for a half-window stride."""

    if window != 2 * stride:
        raise ValueError("VPT dense reconstruction requires stride == window / 2")
    begin = (window - stride) // 2
    return begin, begin + stride


def validate_temporal_support(
    *, derived_rate_hz: int, phases: list[int], support_mode: str
) -> None:
    """Validate the temporal grid represented by an evaluation manifest."""

    if support_mode == "native-grid":
        if derived_rate_hz == 20 and phases != [0, 1, 2]:
            raise ValueError("20 Hz native-grid evaluation requires all phases 0,1,2")
        if derived_rate_hz == 60 and phases != [0]:
            raise ValueError("native 60 Hz evaluation requires exactly phase 0")
        if derived_rate_hz not in (20, 60):
            raise ValueError("unsupported derived evaluation rate")
        return
    if support_mode == "deployment-20hz-phase0":
        if derived_rate_hz != 20 or phases != [0]:
            raise ValueError(
                "20 Hz deployment evaluation requires derived_rate_hz=20 and phase 0"
            )
        return
    raise ValueError(f"unsupported evaluation support mode: {support_mode}")


def center_supported_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exclude streams that cannot contribute a complete evaluation window."""

    selected = [record for record in records if int(record.get("windows", -1)) > 0]
    if not selected:
        raise ValueError("evaluation manifest has no center-supported windows")
    return selected


def flatten_center_probabilities(
    blocks: list[np.ndarray], *, retained: int = 64
) -> np.ndarray:
    """Flatten dense center-region outputs into one prediction per row."""

    probability = np.concatenate(blocks, axis=0)
    expected_tail = (retained, len(KEY_ORDER))
    if probability.ndim != 3 or probability.shape[1:] != expected_tail:
        raise RuntimeError(
            "center probability blocks must concatenate to "
            f"[windows,{retained},{len(KEY_ORDER)}], got {probability.shape}"
        )
    return probability.reshape(-1, len(KEY_ORDER))


def resolve_center_overlap(
    rows: np.ndarray, probability: np.ndarray, *, policy: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """Keep base-window predictions and use appended tail windows only as fill."""

    if len(rows) == len(np.unique(rows)):
        return rows, probability
    if policy != "base-first-stable-tail-fill":
        raise RuntimeError("center reconstruction overlaps without a frozen policy")
    _, first = np.unique(rows, return_index=True)
    keep = np.sort(first)
    return rows[keep], probability[keep]


def infer_stream(
    model: VPTSmallIDM,
    *,
    directory: Path,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    window: int,
    stride: int,
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
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    overlap_policy = metadata.get("center_overlap_policy")
    probabilities: list[np.ndarray] = []
    selected_rows: list[np.ndarray] = []
    retain_begin, retain_end = retained_positions(window, stride)
    if model.config.frames != window:
        raise ValueError("checkpoint frame count differs from evaluation window")
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            block = np.stack(
                [
                    np.array(frames[int(start) : int(start) + window], copy=True)
                    for start in batch_starts
                ]
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
                F.softmax(logits[:, retain_begin:retain_end], dim=-1)[..., 1]
                .float()
                .cpu()
                .numpy()
            )
            selected_rows.extend(
                np.arange(
                    int(start) + retain_begin,
                    int(start) + retain_end,
                    dtype=np.int64,
                )
                for start in batch_starts
            )
    if not probabilities:
        raise ValueError(f"no center-supported windows in {directory}")
    rows = np.concatenate(selected_rows)
    probability = flatten_center_probabilities(probabilities, retained=stride)
    if len(probability) != len(rows):
        raise RuntimeError("center probability rows do not match reconstructed rows")
    rows, probability = resolve_center_overlap(
        rows, probability, policy=overlap_policy
    )
    return {
        "source_row": np.asarray(source_rows[rows]),
        "engine_idx": np.asarray(engine_idx[rows]),
        "continuity": np.asarray(continuity[rows]),
        "truth": np.asarray(keys[rows]),
        "active": np.asarray(active[rows]),
        "probability": probability,
    }


def combine_phases(
    manifest: dict[str, Any],
    root: Path,
    inferred: list[dict[str, np.ndarray]],
    *,
    expected_source_row_step: int = 1,
) -> dict[str, Any]:
    if expected_source_row_step < 1:
        raise ValueError("expected source-row step must be positive")
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
    row_session_id: list[np.ndarray] = []
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
        boundaries = (
            np.flatnonzero(
                np.diff(merged["source_row"]) != expected_source_row_step
            )
            + 1
        )
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
            row_session_id.append(np.full(end - start, session_id))
            stream_lengths.append(int(end - start))
            stream_ids.append(f"{session_id}__run{continuity_id:03d}__sub{subrun:03d}")
    return {
        "truth": np.concatenate(truth),
        "probability": np.concatenate(probability),
        "active": np.concatenate(active),
        "source_row": np.concatenate(source_rows),
        "engine_idx": np.concatenate(engine_idx),
        "row_session_id": np.concatenate(row_session_id),
        "stream_lengths": np.asarray(stream_lengths, dtype=np.int64),
        "stream_ids": np.asarray(stream_ids),
    }


def restrict_to_reference_support(
    combined: dict[str, Any], reference_path: Path
) -> dict[str, Any]:
    """Select candidate rows in the exact order and boundaries of a frozen sidecar."""

    with np.load(reference_path, allow_pickle=False) as archive:
        required = {
            "y_true", "input_active", "source_row_index",
            "source_engine_frame_idx", "session_lengths", "session_ids",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"reference sidecar lacks {sorted(missing)}")
        reference = {name: np.asarray(archive[name]) for name in required}
    lengths = reference["session_lengths"].astype(np.int64, copy=False)
    stream_ids = reference["session_ids"].astype(str, copy=False)
    if int(lengths.sum()) != len(reference["source_row_index"]):
        raise RuntimeError("reference sidecar stream lengths do not cover its rows")
    reference_sessions = np.concatenate(
        [
            np.full(int(length), stream_id.split("__run", 1)[0])
            for stream_id, length in zip(stream_ids, lengths, strict=True)
        ]
    )
    candidate_keys = list(
        zip(
            combined["row_session_id"].astype(str).tolist(),
            combined["source_row"].astype(np.int64).tolist(),
            strict=True,
        )
    )
    if len(candidate_keys) != len(set(candidate_keys)):
        raise RuntimeError("candidate native-rate support contains duplicate row keys")
    candidate_by_key = {key: index for index, key in enumerate(candidate_keys)}
    reference_keys = list(
        zip(
            reference_sessions.tolist(),
            reference["source_row_index"].astype(np.int64).tolist(),
            strict=True,
        )
    )
    if len(reference_keys) != len(set(reference_keys)):
        raise RuntimeError("reference sidecar contains duplicate row keys")
    missing_keys = [key for key in reference_keys if key not in candidate_by_key]
    if missing_keys:
        raise RuntimeError(f"candidate lacks {len(missing_keys)} frozen reference rows")
    selected = np.asarray([candidate_by_key[key] for key in reference_keys], dtype=np.int64)
    if not np.array_equal(
        combined["engine_idx"][selected], reference["source_engine_frame_idx"]
    ):
        raise RuntimeError("reference/candidate engine indices differ")
    if not np.array_equal(combined["truth"][selected], reference["y_true"]):
        raise RuntimeError("reference/candidate truth differs")
    if not np.array_equal(combined["active"][selected], reference["input_active"]):
        raise RuntimeError("reference/candidate activity mask differs")
    return {
        "truth": combined["truth"][selected],
        "probability": combined["probability"][selected],
        "active": combined["active"][selected],
        "source_row": combined["source_row"][selected],
        "engine_idx": combined["engine_idx"][selected],
        "row_session_id": combined["row_session_id"][selected],
        "stream_lengths": lengths,
        "stream_ids": stream_ids,
    }


def evaluate(
    checkpoint_path: Path,
    manifest_path: Path,
    sidecar_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    expected_checkpoint_sha256: str | None = None,
    reference_sidecar: Path | None = None,
    expected_reference_sha256: str | None = None,
    surface: str = "corrected own-v3 val-A, three-phase VPT common support",
    support_mode: str = "native-grid",
    implementation_commit: str | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
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
    derived_rate = int(manifest.get("derived_rate_hz", -1))
    phases = sorted(manifest["phases"])
    validate_temporal_support(
        derived_rate_hz=derived_rate,
        phases=phases,
        support_mode=support_mode,
    )
    window = int(manifest.get("window", -1))
    stride = int(manifest.get("stride", -1))
    if (window, stride) not in {(128, 64), (384, 192)}:
        raise ValueError("unsupported evaluation window/stride geometry")
    if model_config.frames != window:
        raise ValueError("checkpoint and evaluation manifest frame counts differ")
    manifest_records = list(manifest["records"])
    evaluation_records = center_supported_records(manifest_records)
    inferred = []
    for record_index, record in enumerate(evaluation_records, start=1):
        directory = manifest_path.parent / f"{record['session_id']}__p{record['phase']}"
        inferred.append(
            infer_stream(
                model,
                directory=directory,
                batch_size=batch_size,
                device=device,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                window=window,
                stride=stride,
            )
        )
        if record_index % 100 == 0 or record_index == len(evaluation_records):
            print(
                f"evaluated_records={record_index}/{len(evaluation_records)}",
                flush=True,
            )
    evaluation_manifest = {**manifest, "records": evaluation_records}
    combined = combine_phases(
        evaluation_manifest,
        manifest_path.parent,
        inferred,
        expected_source_row_step=(
            3 if support_mode == "deployment-20hz-phase0" else 1
        ),
    )
    reference_hash = None
    if reference_sidecar is not None:
        reference_hash = sha256_file(reference_sidecar)
        if expected_reference_sha256 and reference_hash != expected_reference_sha256:
            raise RuntimeError("reference sidecar SHA-256 differs from frozen value")
        combined = restrict_to_reference_support(combined, reference_sidecar)
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
        "surface": surface,
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
            "manifest_records": len(manifest_records),
            "evaluated_records": len(evaluation_records),
            "excluded_zero_window_records": len(manifest_records)
            - len(evaluation_records),
            "phases": manifest["phases"],
            "rows": len(truth),
            "active_rows": int(gate.sum()),
            "streams": len(combined["stream_lengths"]),
            "derived_rate_hz": derived_rate,
            "support_mode": support_mode,
            "window": window,
            "stride": stride,
            "retained_positions": list(retained_positions(window, stride)),
            "reference_sidecar": str(reference_sidecar) if reference_sidecar else None,
            "reference_sidecar_sha256": reference_hash,
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        memory_method = "torch.cuda.max_memory_allocated"
        device_name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        torch.mps.synchronize()
        peak_memory = int(torch.mps.driver_allocated_memory())
        memory_method = "torch.mps.driver_allocated_memory_at_completion"
        device_name = "mps"
    else:
        maximum_resident_set = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_memory = (
            maximum_resident_set
            if platform.system() == "Darwin"
            else maximum_resident_set * 1024
        )
        memory_method = "resource.RUSAGE_SELF.ru_maxrss"
        device_name = str(device)
    repo = Path(__file__).resolve().parents[1]
    if implementation_commit is None:
        implementation_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    report["execution"] = {
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "implementation_commit": implementation_commit,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "device": {
            "requested": str(device),
            "type": device.type,
            "name": device_name,
            "precision": str(torch.bfloat16 if device.type == "cuda" else torch.float32),
        },
        "peak_memory_bytes": peak_memory,
        "peak_memory_method": memory_method,
    }
    return json_ready(report)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preds-out", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--reference-sidecar", type=Path)
    parser.add_argument("--expected-reference-sha256")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--surface",
        default="corrected own-v3 val-A, three-phase VPT common support",
    )
    parser.add_argument(
        "--support-mode",
        choices=("native-grid", "deployment-20hz-phase0"),
        default="native-grid",
    )
    parser.add_argument("--implementation-commit")
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
        reference_sidecar=args.reference_sidecar,
        expected_reference_sha256=args.expected_reference_sha256,
        surface=args.surface,
        support_mode=args.support_mode,
        implementation_commit=args.implementation_commit,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
