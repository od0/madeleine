#!/usr/bin/env python3
"""Evaluate one distributed paper-IDM checkpoint on frozen three-phase val-A.

The evaluator runs the exact FSDPv2 graph used for training, reconstructs the
64-position center releases, and then restricts them to a SHA-bound reference
sidecar.  It intentionally supports fixed 0.5 only.
"""

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
from badeline.vpt_paper_idm import (
    VPTPaperIDM,
    VPTPaperIDMConfig,
    parameter_inventory,
)
from data.schema import KEY_ORDER
from experiments.keypress_accuracy import score_sidecar
from experiments.train_vpt_paper_idm_xla import mark_empty_parameters_replicated


SCHEMA = "madeleine.vpt-paper-idm-xla-eval.v1"
CHECKPOINT_SCHEMA = "madeleine.vpt-paper-idm-xla-checkpoint.v1"
RECEIPT_SCHEMA = "madeleine.vpt-paper-idm-xla-epoch-publication.v1"
EXPECTED_PARAMETERS = 482_133_390


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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


def equal_mass_ece(
    truth: np.ndarray, probability: np.ndarray, bins: int = 15
) -> dict[str, float]:
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
    if window != 2 * stride:
        raise ValueError("paper-IDM evaluation requires half-window stride")
    begin = (window - stride) // 2
    return begin, begin + stride


def validate_checkpoint_receipt(
    checkpoint: Path, receipt_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported paper-IDM checkpoint manifest")
    if manifest.get("completed_epoch") is not True:
        raise ValueError("evaluation requires a completed-epoch checkpoint")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("unsupported checkpoint publication receipt")
    if receipt.get("checkpoint") != checkpoint.name:
        raise ValueError("checkpoint receipt names a different checkpoint")
    objects = receipt.get("objects")
    if not isinstance(objects, list) or len(objects) != 4:
        raise ValueError("checkpoint receipt must bind exactly four objects")
    expected_paths = {"manifest.json", "rng.pt", "state/.metadata", "state/__0_0.distcp"}
    if {item.get("path") for item in objects} != expected_paths:
        raise ValueError("checkpoint receipt object inventory changed")
    total = 0
    for item in objects:
        path = checkpoint / str(item["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != int(item["bytes"]):
            raise RuntimeError(f"checkpoint byte count mismatch: {path}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"checkpoint SHA-256 mismatch: {path}")
        total += size
    if total != int(receipt.get("total_bytes", -1)):
        raise RuntimeError("checkpoint receipt total byte count mismatch")
    return manifest, receipt


def _load_arrays(directory: Path) -> dict[str, np.ndarray]:
    return {
        "frames": np.load(directory / "frames.npy", mmap_mode="r", allow_pickle=False),
        "truth": np.load(directory / "keys.npy", mmap_mode="r", allow_pickle=False),
        "active": np.load(directory / "input_active.npy", mmap_mode="r", allow_pickle=False),
        "engine_idx": np.load(
            directory / "source_engine_frame_idx.npy", mmap_mode="r", allow_pickle=False
        ),
        "source_row": np.load(
            directory / "source_row_index.npy", mmap_mode="r", allow_pickle=False
        ),
        "continuity": np.load(
            directory / "continuity_id.npy", mmap_mode="r", allow_pickle=False
        ),
        "starts": np.load(directory / "window_start.npy", allow_pickle=False),
    }


def infer_stream(
    model: torch.nn.Module,
    *,
    directory: Path,
    microbatch: int,
    device: torch.device,
    mesh: Any,
    xs: Any,
    xm: Any,
    window: int,
    stride: int,
) -> dict[str, np.ndarray]:
    arrays = _load_arrays(directory)
    starts = arrays.pop("starts")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    begin, end = retained_positions(window, stride)
    probabilities: list[np.ndarray] = []
    selected_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), microbatch):
            valid_starts = starts[offset : offset + microbatch]
            padded = valid_starts
            if len(padded) < microbatch:
                padded = np.concatenate(
                    [padded, np.repeat(padded[-1:], microbatch - len(padded))]
                )
            block = np.stack(
                [
                    np.array(arrays["frames"][int(start) : int(start) + window], copy=True)
                    for start in padded
                ]
            )
            frames = (
                torch.from_numpy(block)
                .permute(0, 1, 4, 2, 3)
                .to(device=device, dtype=torch.float32)
                .div_(255.0)
            )
            xs.mark_sharding(frames, mesh, ("fsdp", None, None, None, None))
            with torch.autocast("xla", dtype=torch.bfloat16):
                logits = model(frames)
                probability = F.softmax(logits[:, begin:end], dim=-1)[..., 1]
            xm.mark_step()
            probability_numpy = probability.float().cpu().numpy()[: len(valid_starts)]
            probabilities.append(probability_numpy)
            selected_rows.extend(
                np.arange(int(start) + begin, int(start) + end, dtype=np.int64)
                for start in valid_starts
            )
    if not probabilities:
        raise ValueError(f"no evaluation windows in {directory}")
    rows = np.concatenate(selected_rows)
    probability = np.concatenate(probabilities, axis=0).reshape(-1, len(KEY_ORDER))
    if len(rows) != len(probability):
        raise RuntimeError("prediction and center-row counts differ")
    if len(rows) != len(np.unique(rows)):
        if metadata.get("center_overlap_policy") != "base-first-stable-tail-fill":
            raise RuntimeError("overlapping centers lack the frozen resolution policy")
        _, first = np.unique(rows, return_index=True)
        keep = np.sort(first)
        rows = rows[keep]
        probability = probability[keep]
    return {
        "source_row": np.asarray(arrays["source_row"][rows]),
        "engine_idx": np.asarray(arrays["engine_idx"][rows]),
        "continuity": np.asarray(arrays["continuity"][rows]),
        "truth": np.asarray(arrays["truth"][rows]),
        "active": np.asarray(arrays["active"][rows]),
        "probability": probability,
    }


def combine_phases(
    manifest: dict[str, Any], inferred: list[dict[str, np.ndarray]]
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, np.ndarray]]] = defaultdict(list)
    for record, arrays in zip(manifest["records"], inferred, strict=True):
        for continuity_id in np.unique(arrays["continuity"]):
            selected = arrays["continuity"] == continuity_id
            grouped[(str(record["session_id"]), int(continuity_id))].append(
                {key: value[selected] for key, value in arrays.items()}
            )
    combined = {key: [] for key in ("truth", "probability", "active", "source_row", "engine_idx")}
    row_session_id: list[np.ndarray] = []
    stream_lengths: list[int] = []
    stream_ids: list[str] = []
    for (session_id, continuity_id), parts in grouped.items():
        merged = {
            key: np.concatenate([part[key] for part in parts])
            for key in parts[0]
            if key != "continuity"
        }
        order = np.argsort(merged["source_row"], kind="stable")
        for key in merged:
            merged[key] = merged[key][order]
        if len(np.unique(merged["source_row"])) != len(merged["source_row"]):
            raise RuntimeError("three-phase reconstruction has duplicate target rows")
        boundaries = np.flatnonzero(np.diff(merged["source_row"]) != 1) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(merged["source_row"])]))
        for subrun, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if end <= start:
                continue
            for key in combined:
                combined[key].append(merged[key][start:end])
            row_session_id.append(np.full(end - start, session_id))
            stream_lengths.append(int(end - start))
            stream_ids.append(f"{session_id}__run{continuity_id:03d}__sub{subrun:03d}")
    result = {key: np.concatenate(values) for key, values in combined.items()}
    result.update(
        row_session_id=np.concatenate(row_session_id),
        stream_lengths=np.asarray(stream_lengths, dtype=np.int64),
        stream_ids=np.asarray(stream_ids),
    )
    return result


def restrict_to_reference(combined: dict[str, Any], reference: Path) -> dict[str, Any]:
    with np.load(reference, allow_pickle=False) as archive:
        required = {
            "y_true", "input_active", "source_row_index",
            "source_engine_frame_idx", "session_lengths", "session_ids",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"reference sidecar lacks {sorted(missing)}")
        frozen = {name: np.asarray(archive[name]) for name in required}
    lengths = frozen["session_lengths"].astype(np.int64, copy=False)
    stream_ids = frozen["session_ids"].astype(str, copy=False)
    if int(lengths.sum()) != len(frozen["source_row_index"]):
        raise RuntimeError("reference stream lengths do not cover its rows")
    sessions = np.concatenate(
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
        raise RuntimeError("candidate support contains duplicate row keys")
    lookup = {key: index for index, key in enumerate(candidate_keys)}
    reference_keys = list(
        zip(
            sessions.tolist(),
            frozen["source_row_index"].astype(np.int64).tolist(),
            strict=True,
        )
    )
    if len(reference_keys) != len(set(reference_keys)):
        raise RuntimeError("reference support contains duplicate row keys")
    missing = [key for key in reference_keys if key not in lookup]
    if missing:
        raise RuntimeError(f"paper-IDM lacks {len(missing)} frozen support rows")
    selected = np.asarray([lookup[key] for key in reference_keys], dtype=np.int64)
    checks = {
        "truth": "y_true",
        "active": "input_active",
        "engine_idx": "source_engine_frame_idx",
    }
    for candidate_name, frozen_name in checks.items():
        if not np.array_equal(combined[candidate_name][selected], frozen[frozen_name]):
            raise RuntimeError(f"reference {candidate_name} differs from candidate")
    return {
        "truth": combined["truth"][selected],
        "probability": combined["probability"][selected],
        "active": combined["active"][selected],
        "source_row": combined["source_row"][selected],
        "engine_idx": combined["engine_idx"][selected],
        "stream_lengths": lengths,
        "stream_ids": stream_ids,
    }


def fixed_report(
    common: dict[str, Any],
    *,
    checkpoint: Path,
    checkpoint_manifest: dict[str, Any],
    checkpoint_receipt: Path,
    manifest_path: Path,
    reference_sidecar: Path,
    sidecar_path: Path,
) -> dict[str, Any]:
    truth = common["truth"].astype(np.uint8, copy=False)
    probability = common["probability"].astype(np.float32, copy=False)
    active = common["active"].astype(np.uint8, copy=False)
    if truth.shape != probability.shape or truth.shape[1] != len(KEY_ORDER):
        raise RuntimeError("truth/probability shapes differ")
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("paper-IDM probabilities are nonfinite")
    if sidecar_path.exists():
        raise FileExistsError(sidecar_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sidecar_path,
        y_true=truth,
        y_prob=probability,
        input_active=active,
        session_lengths=common["stream_lengths"],
        session_ids=common["stream_ids"],
        source_row_index=common["source_row"],
        source_engine_frame_idx=common["engine_idx"],
    )
    gate = active.astype(bool)
    gated_truth = truth[gate]
    gated_probability = probability[gate]
    clipped = np.clip(gated_probability, 1e-7, 1.0 - 1e-7)
    natural_nll = float(
        np.mean(
            -(gated_truth * np.log(clipped) + (1 - gated_truth) * np.log(1 - clipped)),
            axis=0,
        ).sum()
    )
    detail = summarize(
        truth,
        probability,
        boundaries=common["stream_lengths"].tolist(),
        active=gate,
        fixed_transition_thresholds={key: 0.5 for key in KEY_ORDER},
        include_oracle=False,
    )
    predicted = gated_probability >= 0.5
    tp = np.logical_and(predicted, gated_truth == 1).sum(axis=0)
    fp = np.logical_and(predicted, gated_truth == 0).sum(axis=0)
    fn = np.logical_and(~predicted, gated_truth == 1).sum(axis=0)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / np.maximum(1, tp + fn)
    state_f1 = 2 * precision * recall / np.maximum(1e-12, precision + recall)
    receipt = json.loads(checkpoint_receipt.read_text(encoding="utf-8"))
    state_object = next(item for item in receipt["objects"] if item["path"] == "state/__0_0.distcp")
    report = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "surface": "corrected own-v3 val-A, exact VPT three-phase common support",
        "threshold": 0.5,
        "weights": {
            "checkpoint": str(checkpoint),
            "epoch": checkpoint_manifest["epoch"],
            "optimizer_step": checkpoint_manifest["optimizer_step"],
            "state_object_sha256": state_object["sha256"],
            "checkpoint_receipt": str(checkpoint_receipt),
            "checkpoint_receipt_sha256": sha256_file(checkpoint_receipt),
        },
        "support_authority": {
            "sidecar": str(reference_sidecar),
            "sha256": sha256_file(reference_sidecar),
        },
        "data": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "rows": len(truth),
            "active_rows": int(gate.sum()),
            "streams": len(common["stream_lengths"]),
            "derived_rate_hz": 20,
            "window": 128,
            "stride": 64,
            "retained_positions": [32, 96],
        },
        "natural_nll": natural_nll,
        "brier": float(np.mean((gated_probability - gated_truth) ** 2)),
        "equal_mass_ece": equal_mass_ece(gated_truth, gated_probability),
        "aggregate": {
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
            "macro_ap": float(np.nanmean(list(detail["per_key_ap"].values()))),
            "macro_state_f1": float(np.nanmean(state_f1)),
            "macro_state_precision": float(np.nanmean(precision)),
            "macro_state_recall": float(np.nanmean(recall)),
            "macro_event_f1_collar_0": float(np.nanmean([
                detail["transition_f1_at_0.5"][key]["event"]["f1"] for key in KEY_ORDER
            ])),
            "macro_event_f1_collar_2_native_frames": float(np.nanmean([
                detail["transition_f1_at_0.5_collars"]["2"][key]["event"]["f1"]
                for key in KEY_ORDER
            ])),
        },
        "metrics": detail,
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-sidecar", type=Path, required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preds-out", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.microbatch != 4:
        raise ValueError("matched v6e-4 evaluation is frozen to microbatch 4")
    if args.out.exists() or args.preds_out.exists():
        raise FileExistsError("refusing to overwrite evaluation artifacts")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config_sha = sha256_file(args.config)
    manifest_sha = sha256_file(args.manifest)
    if manifest_sha != config["data"]["val_manifest_sha256"]:
        raise RuntimeError("validation manifest differs from frozen config")
    if sha256_file(args.reference_sidecar) != args.expected_reference_sha256:
        raise RuntimeError("reference support sidecar SHA-256 mismatch")
    checkpoint_manifest, _ = validate_checkpoint_receipt(
        args.checkpoint, args.checkpoint_receipt
    )
    if checkpoint_manifest["config_sha256"] != config_sha:
        raise RuntimeError("checkpoint/config SHA-256 mismatch")
    if checkpoint_manifest["val_manifest_sha256"] != manifest_sha:
        raise RuntimeError("checkpoint/val-manifest SHA-256 mismatch")

    import torch.distributed as dist
    import torch.distributed.checkpoint as dist_cp
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.distributed.xla_backend  # noqa: F401
    import torch_xla.runtime as xr
    from torch_xla.distributed.spmd import Mesh
    from torch_xla.experimental.distributed_checkpoint import SPMDLoadPlanner
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        SpmdFullyShardedDataParallel as FSDPv2,
    )

    xr.use_spmd()
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo", init_method="xla://", rank=xr.process_index(), world_size=xr.process_count()
        )
    device = xm.xla_device()
    device_count = xr.global_runtime_device_count()
    if device_count != 4:
        raise RuntimeError(f"matched evaluation requires four TPU devices, got {device_count}")
    mesh = Mesh(np.arange(device_count), (device_count, 1), ("fsdp", "model"))
    model = VPTPaperIDM(VPTPaperIDMConfig.from_dict(config["model"]))
    if parameter_inventory(model)["total"] != EXPECTED_PARAMETERS:
        raise RuntimeError("paper-IDM parameter inventory changed")
    model = model.to(device)
    empty = mark_empty_parameters_replicated(model, mesh=mesh, xs=xs)
    if len(empty) != 2:
        raise RuntimeError("paper-IDM empty-parameter inventory changed")
    model = FSDPv2(model, mesh=mesh)
    state = {"model": model.state_dict()}
    dist_cp.load(
        state_dict=state,
        storage_reader=dist_cp.FileSystemReader(args.checkpoint / "state"),
        planner=SPMDLoadPlanner(),
    )
    model.load_state_dict(state["model"])

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("derived_rate_hz") != 20 or sorted(manifest.get("phases", [])) != [0, 1, 2]:
        raise ValueError("paper-IDM evaluation requires the frozen three-phase 20 Hz manifest")
    if (manifest.get("window"), manifest.get("stride")) != (128, 64):
        raise ValueError("paper-IDM evaluation geometry changed")
    inferred = []
    for record in manifest["records"]:
        directory = args.manifest.parent / f"{record['session_id']}__p{record['phase']}"
        inferred.append(
            infer_stream(
                model,
                directory=directory,
                microbatch=args.microbatch,
                device=device,
                mesh=mesh,
                xs=xs,
                xm=xm,
                window=128,
                stride=64,
            )
        )
    common = restrict_to_reference(combine_phases(manifest, inferred), args.reference_sidecar)
    report = fixed_report(
        common,
        checkpoint=args.checkpoint,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_receipt=args.checkpoint_receipt,
        manifest_path=args.manifest,
        reference_sidecar=args.reference_sidecar,
        sidecar_path=args.preds_out,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
