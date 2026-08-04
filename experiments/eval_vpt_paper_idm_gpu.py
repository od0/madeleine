#!/usr/bin/env python3
"""Evaluate a TPU-trained paper-IDM checkpoint on a CUDA Wild7 surface.

The distributed checkpoint stores the FSDPv2 model under ``_orig_module``.
PyTorch DCP can restore that state into the identical unwrapped module on CPU;
inference then uses the same center-release reconstruction as VPT-small.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed.checkpoint as dist_cp

from badeline.metrics import summarize
from badeline.vpt_paper_idm import (
    VPTPaperIDM,
    VPTPaperIDMConfig,
    parameter_inventory,
)
from data.schema import KEY_ORDER
from experiments.eval_vpt_paper_idm_xla import validate_checkpoint_receipt
from experiments.eval_vpt_small import (
    center_supported_records,
    combine_phases,
    infer_stream,
    json_ready,
    sha256_file,
)
from experiments.keypress_accuracy import score_sidecar


SCHEMA = "madeleine.vpt-paper-idm-cuda-wild7-eval.v1"
EXPECTED_PARAMETERS = 482_133_390


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    checkpoint_manifest, receipt = validate_checkpoint_receipt(
        args.checkpoint, args.checkpoint_receipt
    )
    run_meta = json.loads(args.run_meta.read_text(encoding="utf-8"))
    if run_meta.get("config_sha256") != checkpoint_manifest.get("config_sha256"):
        raise RuntimeError("run metadata and checkpoint name different configs")
    config = VPTPaperIDMConfig.from_dict(run_meta["config"]["model"])
    model = VPTPaperIDM(config)
    if parameter_inventory(model)["total"] != EXPECTED_PARAMETERS:
        raise RuntimeError("paper-IDM parameter inventory changed")
    state = {"model": {"_orig_module": model.state_dict()}}
    dist_cp.load(state, checkpoint_id=args.checkpoint / "state")
    missing, unexpected = model.load_state_dict(state["model"]["_orig_module"])
    if missing or unexpected:
        raise RuntimeError("paper-IDM distributed checkpoint did not load exactly")
    device = torch.device(args.device)
    model.to(device).eval()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("derived_rate_hz") != 20 or manifest.get("phases") != [0]:
        raise ValueError("Wild7 paper-IDM evaluation requires 20 Hz phase0")
    if (manifest.get("window"), manifest.get("stride")) != (128, 64):
        raise ValueError("Wild7 paper-IDM evaluation geometry changed")
    records = center_supported_records(list(manifest["records"]))
    inferred = []
    for index, record in enumerate(records, start=1):
        directory = args.manifest.parent / f"{record['session_id']}__p{record['phase']}"
        inferred.append(
            infer_stream(
                model,
                directory=directory,
                batch_size=args.batch_size,
                device=device,
                dtype=torch.bfloat16,
                window=128,
                stride=64,
            )
        )
        if index % 100 == 0 or index == len(records):
            print(f"evaluated_records={index}/{len(records)}", flush=True)
    combined = combine_phases(
        {**manifest, "records": records},
        args.manifest.parent,
        inferred,
        expected_source_row_step=3,
    )
    truth = combined["truth"].astype(np.uint8, copy=False)
    probability = combined["probability"].astype(np.float32, copy=False)
    active = combined["active"].astype(np.uint8, copy=False)
    if truth.shape != probability.shape or truth.shape[1] != len(KEY_ORDER):
        raise RuntimeError("paper-IDM truth and probability shapes differ")
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("paper-IDM produced nonfinite probabilities")
    args.preds_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.preds_out,
        y_true=truth,
        y_prob=probability,
        input_active=active,
        session_lengths=combined["stream_lengths"],
        session_ids=combined["stream_ids"],
        source_row_index=combined["source_row"],
        source_engine_frame_idx=combined["engine_idx"],
    )
    gate = active.astype(bool)
    detail = summarize(
        truth,
        probability,
        boundaries=combined["stream_lengths"].tolist(),
        active=gate,
        fixed_transition_thresholds={key: 0.5 for key in KEY_ORDER},
        include_oracle=False,
    )
    predicted = probability[gate] >= 0.5
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "surface": "Wild admitted7 identical 842624-row support; paper-IDM CUDA inference",
        "checkpoint": {
            "path": str(args.checkpoint),
            "epoch": checkpoint_manifest["epoch"],
            "optimizer_step": checkpoint_manifest["optimizer_step"],
            "state_object_sha256": next(
                item["sha256"]
                for item in receipt["objects"]
                if item["path"] == "state/__0_0.distcp"
            ),
            "receipt": str(args.checkpoint_receipt),
            "receipt_sha256": sha256_file(args.checkpoint_receipt),
        },
        "data": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "rows": len(truth),
            "active_rows": int(gate.sum()),
            "streams": len(combined["stream_lengths"]),
            "support_mode": "deployment-20hz-phase0",
        },
        "aggregate": {
            "macro_ap": float(np.nanmean(list(detail["per_key_ap"].values()))),
            "per_key_ap": detail["per_key_ap"],
            "micro_key_accuracy": float(
                np.mean(predicted == truth[gate].astype(bool))
            ),
        },
        "metrics": detail,
        "key_state_accuracy": score_sidecar(args.preds_out, threshold=0.5),
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
            "checkpoint_conversion": "none; selective DCP restore into identical unwrapped module",
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preds-out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate(args)
    print(json.dumps(json_ready(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
