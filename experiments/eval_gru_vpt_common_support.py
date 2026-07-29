#!/usr/bin/env python3
"""Rescore an existing pixel GRU on VPT-small's exact val-A row support.

The GRU keeps its native 60 Hz, 32-frame inference path.  The VPT prediction
sidecar is used only as a frozen list of target row IDs and segment
boundaries; its probabilities are never consulted while producing the GRU
predictions.  This makes the architecture comparison share labels, masks,
row IDs, and event boundaries without changing either model's input recipe.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from badeline.eval import select_checkpoint_state
from badeline.metrics import summarize
from badeline.model import BadelineIDM
from badeline.train import contiguous_runs, history_block, load_session, target_offset
from data.schema import KEY_ORDER
from experiments.eval_vpt_small import equal_mass_ece, json_ready, sha256_file
from experiments.keypress_accuracy import score_sidecar


def infer_native_gru(
    model: BadelineIDM,
    config: dict[str, Any],
    *,
    data_dir: Path,
    session_id: str,
    device: torch.device,
    segment_span: int = 512,
) -> dict[str, np.ndarray]:
    """Return one prediction for every native target row the GRU supports."""

    model.eval().to(device)
    window = int(config.get("window", 2))
    frame_stride = int(config.get("frame_stride", 1))
    offset = target_offset(window, config.get("window_mode", "centered"))
    frame_span = (window - 1) * frame_stride + 1
    input_config = config.get("input_config", "pixels")
    if input_config not in ("pixels", "pixels_plus_history"):
        raise ValueError("common-support rescore requires a native pixel model")
    uses_history = input_config == "pixels_plus_history"
    history_len = int(config.get("history_len", 8))
    history_gap = int(config.get("history_gap", 0))
    arrays = load_session(data_dir, session_id, precomputed_features=False)
    if arrays.engine_frame_idx is None or arrays.input_active is None:
        raise ValueError("engine_frame_idx and input_active are required")

    rows: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for run_start, run_end in contiguous_runs(arrays.engine_frame_idx):
            n_windows = run_end - run_start - frame_span + 1
            if n_windows < 1:
                continue
            run_probabilities: list[np.ndarray] = []
            for relative_start in range(0, n_windows, segment_span):
                count = min(segment_span, n_windows - relative_start)
                start = run_start + relative_start
                block = arrays.frames[start : start + count + frame_span - 1]
                inputs: dict[str, torch.Tensor] = {
                    "frames": (
                        torch.from_numpy(block.copy())
                        .permute(0, 3, 1, 2)
                        .to(dtype=torch.float32)
                        .div_(255.0)
                        .unsqueeze(0)
                        .to(device)
                    )
                }
                if uses_history:
                    target_indices = [
                        start + step + offset * frame_stride
                        for step in range(count)
                    ]
                    inputs["history"] = torch.from_numpy(
                        history_block(
                            arrays.keys,
                            target_indices,
                            history_len,
                            history_gap,
                            floor=run_start,
                        )
                    ).unsqueeze(0).to(device)
                logits = model.forward_segment(inputs)
                run_probabilities.append(
                    torch.sigmoid(logits)[0].float().cpu().numpy()
                )
            probabilities.append(np.concatenate(run_probabilities))
            first_target = run_start + offset * frame_stride
            rows.append(np.arange(first_target, first_target + n_windows, dtype=np.int64))

    if not rows:
        raise ValueError("GRU has no supported native target rows")
    row = np.concatenate(rows)
    probability = np.concatenate(probabilities)
    if len(row) != len(np.unique(row)) or probability.shape != (len(row), len(KEY_ORDER)):
        raise RuntimeError("native GRU row reconstruction is not one-to-one")
    return {
        "source_row": row,
        "probability": probability.astype(np.float32, copy=False),
        "truth": arrays.keys[row].astype(np.uint8, copy=False),
        "active": arrays.input_active[row].astype(np.uint8, copy=False),
        "engine_idx": arrays.engine_frame_idx[row].astype(np.int64, copy=False),
    }


def select_common_rows(
    native: dict[str, np.ndarray], vpt_sidecar: Path
) -> dict[str, np.ndarray]:
    with np.load(vpt_sidecar, allow_pickle=False) as archive:
        required = {
            "y_true",
            "input_active",
            "session_lengths",
            "session_ids",
            "source_row_index",
            "source_engine_frame_idx",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"VPT sidecar is missing {sorted(missing)}")
        support = np.asarray(archive["source_row_index"], dtype=np.int64)
        expected_truth = np.asarray(archive["y_true"], dtype=np.uint8)
        expected_active = np.asarray(archive["input_active"], dtype=np.uint8)
        expected_engine = np.asarray(archive["source_engine_frame_idx"], dtype=np.int64)
        lengths = np.asarray(archive["session_lengths"], dtype=np.int64)
        session_ids = np.asarray(archive["session_ids"])
    if len(support) != len(np.unique(support)):
        raise RuntimeError("VPT common support contains duplicate source rows")
    lookup = {int(row): index for index, row in enumerate(native["source_row"])}
    absent = [int(row) for row in support if int(row) not in lookup]
    if absent:
        raise RuntimeError(f"GRU cannot predict {len(absent)} VPT support rows")
    indexes = np.asarray([lookup[int(row)] for row in support], dtype=np.int64)
    truth = native["truth"][indexes]
    active = native["active"][indexes]
    engine = native["engine_idx"][indexes]
    if not np.array_equal(truth, expected_truth):
        raise RuntimeError("GRU source truth differs from the VPT sidecar truth")
    if not np.array_equal(active, expected_active):
        raise RuntimeError("GRU source active mask differs from the VPT sidecar mask")
    if not np.array_equal(engine, expected_engine):
        raise RuntimeError("GRU source engine rows differ from the VPT sidecar rows")
    if int(lengths.sum()) != len(support):
        raise RuntimeError("VPT session lengths do not cover its support")
    return {
        "truth": truth,
        "active": active,
        "probability": native["probability"][indexes],
        "source_row": support,
        "engine_idx": engine,
        "session_lengths": lengths,
        "session_ids": session_ids,
    }


def fixed_report(
    common: dict[str, np.ndarray],
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    weights: str,
    sidecar: Path,
    support_sidecar: Path,
) -> dict[str, Any]:
    truth = common["truth"]
    probability = common["probability"]
    active = common["active"].astype(bool)
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("GRU probabilities contain nonfinite values")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sidecar,
        y_true=truth,
        y_prob=probability.astype(np.float32),
        input_active=active.astype(np.uint8),
        session_lengths=common["session_lengths"],
        session_ids=common["session_ids"],
        source_row_index=common["source_row"],
        source_engine_frame_idx=common["engine_idx"],
    )
    gated_truth = truth[active]
    gated_probability = probability[active]
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
        boundaries=common["session_lengths"].tolist(),
        active=active,
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
    return json_ready({
        "schema_version": "madeleine.vpt-small-gru-common-support-eval.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "surface": "corrected own-v3 val-A, exact VPT three-phase common support",
        "threshold": 0.5,
        "weights": {
            "checkpoint": str(checkpoint),
            "sha256": checkpoint_sha256,
            "population": weights,
        },
        "support_authority": {
            "sidecar": str(support_sidecar),
            "sha256": sha256_file(support_sidecar),
        },
        "data": {
            "rows": len(truth),
            "active_rows": int(active.sum()),
            "streams": len(common["session_lengths"]),
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
                detail["transition_f1_at_0.5"][key]["event"]["f1"]
                for key in KEY_ORDER
            ])),
            "macro_event_f1_collar_2_native_frames": float(np.nanmean([
                detail["transition_f1_at_0.5_collars"]["2"][key]["event"]["f1"]
                for key in KEY_ORDER
            ])),
        },
        "metrics": detail,
        "key_state_accuracy": score_sidecar(sidecar, threshold=0.5),
        "sidecar": {
            "path": str(sidecar),
            "bytes": sidecar.stat().st_size,
            "sha256": sha256_file(sidecar),
        },
    })


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--vpt-sidecar", type=Path, required=True)
    parser.add_argument("--weights", choices=("selected", "final"), required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preds-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_path = args.run / "model.pt"
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != args.expected_checkpoint_sha256:
        raise RuntimeError("GRU checkpoint SHA-256 does not match the frozen registry")
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = BadelineIDM(config)
    model.load_state_dict(select_checkpoint_state(checkpoint, args.weights))
    native = infer_native_gru(
        model,
        config,
        data_dir=args.data,
        session_id=args.session_id,
        device=torch.device(args.device),
    )
    common = select_common_rows(native, args.vpt_sidecar)
    report = fixed_report(
        common,
        checkpoint=checkpoint_path,
        checkpoint_sha256=actual_hash,
        weights=args.weights,
        sidecar=args.preds_out,
        support_sidecar=args.vpt_sidecar,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"weights": args.weights, **report["aggregate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
