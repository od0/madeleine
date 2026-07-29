#!/usr/bin/env python3
"""Validate the VPT-small release bundle and apply its frozen candidate gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from data.schema import KEY_ORDER


SUPPORT_FIELDS = (
    "y_true",
    "input_active",
    "session_lengths",
    "session_ids",
    "source_row_index",
    "source_engine_frame_idx",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    sidecar = Path(report["sidecar"]["path"]).name
    report["_path"] = str(path)
    report["_sidecar_name"] = sidecar
    return report


def load_sidecar(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(SUPPORT_FIELDS + ("y_prob",)).difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        result = {
            field: np.asarray(archive[field])
            for field in SUPPORT_FIELDS + ("y_prob",)
        }
    if result["y_true"].shape != result["y_prob"].shape:
        raise ValueError(f"{path} truth/probability shapes differ")
    if result["y_true"].ndim != 2 or result["y_true"].shape[1] != len(KEY_ORDER):
        raise ValueError(f"{path} does not contain seven-key row predictions")
    if not np.all(np.isfinite(result["y_prob"])):
        raise ValueError(f"{path} contains nonfinite probabilities")
    if int(result["session_lengths"].sum()) != len(result["y_true"]):
        raise ValueError(f"{path} session lengths do not cover support")
    if len(np.unique(result["source_row_index"])) != len(result["source_row_index"]):
        raise ValueError(f"{path} contains duplicate source rows")
    return result


def sidecar_entry(report: dict[str, Any], directory: Path) -> dict[str, Any]:
    path = directory / report["_sidecar_name"]
    arrays = load_sidecar(path)
    actual_sha = sha256_file(path)
    if actual_sha != report["sidecar"]["sha256"]:
        raise ValueError(f"{path} SHA-256 differs from its report")
    if path.stat().st_size != report["sidecar"]["bytes"]:
        raise ValueError(f"{path} byte count differs from its report")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "bytes": path.stat().st_size,
        "arrays": arrays,
    }


def equal_support(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> bool:
    return all(
        np.array_equal(reference[field], candidate[field])
        for field in SUPPORT_FIELDS
    )


def per_key_recall(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    active = arrays["input_active"].astype(bool)
    truth = arrays["y_true"][active].astype(bool)
    predicted = arrays["y_prob"][active] >= 0.5
    result: dict[str, float] = {}
    for column, key in enumerate(KEY_ORDER):
        positives = int(truth[:, column].sum())
        if positives == 0:
            raise ValueError(f"candidate support has no positive examples for {key}")
        result[key] = float(
            np.logical_and(truth[:, column], predicted[:, column]).sum() / positives
        )
    return result


def validate_receipts(run_dir: Path) -> list[dict[str, Any]]:
    receipts = []
    for epoch in range(1, 21):
        path = run_dir / f"checkpoint_epoch_{epoch:02d}.r2.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt["checkpoint"]["filename"] != f"checkpoint_epoch_{epoch:02d}.pt":
            raise ValueError(f"{path} records the wrong epoch")
        if receipt["checkpoint"]["sha256"] != receipt["r2_streamed_sha256"]:
            raise ValueError(f"{path} local and streamed R2 hashes differ")
        receipts.append(receipt)
    return receipts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vpt-eval-dir", type=Path, required=True)
    parser.add_argument("--gru-eval-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    run_id = args.run_dir.name
    complete = json.loads((args.run_dir / "complete.json").read_text(encoding="utf-8"))
    best = json.loads((args.run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    if not complete["completed_production_endpoint"]:
        raise ValueError("production completion marker is not final")
    if int(complete["epochs_recorded"]) != 20 or int(complete["optimizer_steps"]) != 2340:
        raise ValueError("production endpoint is not the frozen 20 epochs / 2,340 steps")
    receipts = validate_receipts(args.run_dir)

    report_paths = {
        "vpt_final": args.vpt_eval_dir / f"{run_id}_final_val_a.json",
        "vpt_selected": args.vpt_eval_dir / f"{run_id}_selected_val_a.json",
        "gru_112m95_final": args.gru_eval_dir / "gru_112m95_final_val_a.json",
        "gru_112m95_selected": args.gru_eval_dir / "gru_112m95_selected_val_a.json",
        "gru_36m9_final": args.gru_eval_dir / "gru_36m9_final_val_a.json",
    }
    reports = {name: load_report(path) for name, path in report_paths.items()}
    entries = {
        name: sidecar_entry(
            report,
            args.vpt_eval_dir if name.startswith("vpt_") else args.gru_eval_dir,
        )
        for name, report in reports.items()
    }
    authority = entries["vpt_final"]["arrays"]
    mismatched = [
        name
        for name, entry in entries.items()
        if not equal_support(authority, entry["arrays"])
    ]
    if mismatched:
        raise ValueError(f"common-support mismatch: {mismatched}")
    if int(reports["vpt_final"]["weights"]["epoch"]) != 20:
        raise ValueError("VPT final report is not epoch 20")
    if reports["vpt_final"]["weights"]["sha256"] != complete["final_checkpoint"]["sha256"]:
        raise ValueError("VPT final report/checkpoint marker SHA-256 mismatch")
    if reports["vpt_selected"]["weights"]["sha256"] != best["sha256"]:
        raise ValueError("VPT selected report/best-checkpoint SHA-256 mismatch")

    final_grus = ("gru_112m95_final", "gru_36m9_final")
    better_gru = max(final_grus, key=lambda name: reports[name]["aggregate"]["macro_ap"])
    vpt = reports["vpt_final"]
    guard = reports[better_gru]
    vpt_ap = vpt["metrics"]["per_key_ap"]
    gru_ap = reports["gru_112m95_final"]["metrics"]["per_key_ap"]
    improved_keys = [key for key in KEY_ORDER if vpt_ap[key] > gru_ap[key]]
    recall = per_key_recall(authority)
    prevalence = vpt["aggregate"]["prevalence_per_key"]
    ppr = vpt["aggregate"]["predicted_positive_rate_per_key"]
    ppr_ratio = {key: ppr[key] / prevalence[key] for key in KEY_ORDER}

    gates = {
        "macro_ap_margin": {
            "pass": vpt["aggregate"]["macro_ap"] >= guard["aggregate"]["macro_ap"] + 0.010,
            "vpt": vpt["aggregate"]["macro_ap"],
            "guard": guard["aggregate"]["macro_ap"],
            "required_margin": 0.010,
        },
        "state_f1_guard": {
            "pass": vpt["aggregate"]["macro_state_f1"] >= guard["aggregate"]["macro_state_f1"] - 0.010,
            "vpt": vpt["aggregate"]["macro_state_f1"],
            "guard": guard["aggregate"]["macro_state_f1"],
            "allowed_deficit": 0.010,
        },
        "event_f1_collar_2_guard": {
            "pass": vpt["aggregate"]["macro_event_f1_collar_2_native_frames"]
            >= guard["aggregate"]["macro_event_f1_collar_2_native_frames"] - 0.010,
            "vpt": vpt["aggregate"]["macro_event_f1_collar_2_native_frames"],
            "guard": guard["aggregate"]["macro_event_f1_collar_2_native_frames"],
            "allowed_deficit": 0.010,
        },
        "per_key_ap_improvements": {
            "pass": len(improved_keys) >= 4,
            "count": len(improved_keys),
            "keys": improved_keys,
            "required": 4,
        },
        "state_accuracy_baselines": {
            "pass": (
                vpt["key_state_accuracy"]["key_state_micro_accuracy"]
                > vpt["key_state_accuracy"]["always_released_key_state_micro_accuracy"]
                and vpt["key_state_accuracy"]["joint_exact_match_accuracy"]
                >= vpt["key_state_accuracy"]["always_released_joint_exact_match_accuracy"] - 0.01
            ),
            "micro": vpt["key_state_accuracy"]["key_state_micro_accuracy"],
            "always_released_micro": vpt["key_state_accuracy"]["always_released_key_state_micro_accuracy"],
            "joint": vpt["key_state_accuracy"]["joint_exact_match_accuracy"],
            "always_released_joint": vpt["key_state_accuracy"]["always_released_joint_exact_match_accuracy"],
        },
        "coverage_and_rate": {
            "pass": all(value > 0 for value in recall.values())
            and all(0.5 <= value <= 2.0 for value in ppr_ratio.values()),
            "recall_at_0.5": recall,
            "predicted_positive_rate_to_prevalence": ppr_ratio,
            "required_ratio_range": [0.5, 2.0],
        },
    }
    validation = {
        "schema_version": "madeleine.vpt-small-release-validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "implementation_success": True,
        "candidate_gate_passed": all(item["pass"] for item in gates.values()),
        "better_final_gru": better_gru,
        "support": {
            "rows": int(len(authority["y_true"])),
            "streams": int(len(authority["session_lengths"])),
            "unique_source_rows": int(len(np.unique(authority["source_row_index"]))),
            "all_five_sidecars_identical": True,
        },
        "checkpoint_receipts": {
            "count": len(receipts),
            "all_streamed_hashes_match": True,
            "final_sha256": complete["final_checkpoint"]["sha256"],
            "selected_sha256": best["sha256"],
        },
        "reports": {
            name: {
                "path": str(report_paths[name]),
                "sha256": sha256_file(report_paths[name]),
                "sidecar": {
                    key: value
                    for key, value in entries[name].items()
                    if key != "arrays"
                },
            }
            for name in reports
        },
        "gates": gates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "implementation_success": True,
        "candidate_gate_passed": validation["candidate_gate_passed"],
        "better_final_gru": better_gru,
        "gates": {name: item["pass"] for name, item in gates.items()},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
