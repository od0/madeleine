#!/usr/bin/env python3
"""Validate the matched paper-IDM release and its exact comparison support."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sidecar(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = SUPPORT_FIELDS + ("y_prob",)
        missing = set(required).difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        result = {field: np.asarray(archive[field]) for field in required}
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


def equal_support(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> bool:
    return all(
        np.array_equal(reference[field], candidate[field])
        for field in SUPPORT_FIELDS
    )


def report_entry(
    report_path: Path,
    sidecar_path: Path,
    *,
    expected_report_sha256: str | None = None,
    expected_sidecar_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    report_sha = sha256_file(report_path)
    sidecar_sha = sha256_file(sidecar_path)
    if expected_report_sha256 is not None and report_sha != expected_report_sha256:
        raise ValueError(f"{report_path} differs from its frozen SHA-256")
    if expected_sidecar_sha256 is not None and sidecar_sha != expected_sidecar_sha256:
        raise ValueError(f"{sidecar_path} differs from its frozen SHA-256")
    report = read_json(report_path)
    if sidecar_sha != report["sidecar"]["sha256"]:
        raise ValueError(f"{sidecar_path} differs from its report")
    if sidecar_path.stat().st_size != int(report["sidecar"]["bytes"]):
        raise ValueError(f"{sidecar_path} byte count differs from its report")
    return report, load_sidecar(sidecar_path), {
        "path": str(report_path),
        "sha256": report_sha,
        "sidecar": {
            "path": str(sidecar_path),
            "sha256": sidecar_sha,
            "bytes": sidecar_path.stat().st_size,
        },
    }


def metric_summary(report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report["aggregate"]
    accuracy = report["key_state_accuracy"]
    return {
        "natural_nll": report["natural_nll"],
        "brier": report["brier"],
        "macro_ap": aggregate["macro_ap"],
        "macro_state_f1": aggregate["macro_state_f1"],
        "macro_event_f1_collar_0": aggregate["macro_event_f1_collar_0"],
        "macro_event_f1_collar_2_native_frames": aggregate[
            "macro_event_f1_collar_2_native_frames"
        ],
        "key_state_micro_accuracy": accuracy["key_state_micro_accuracy"],
        "joint_exact_match_accuracy": accuracy["joint_exact_match_accuracy"],
        "predicted_positive_rate_macro": aggregate["predicted_positive_rate_macro"],
        "predicted_positive_rate_per_key": aggregate["predicted_positive_rate_per_key"],
        "prevalence_macro": aggregate["prevalence_macro"],
        "prevalence_per_key": aggregate["prevalence_per_key"],
        "per_key_ap": report["metrics"]["per_key_ap"],
    }


def metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_summary = metric_summary(left)
    right_summary = metric_summary(right)
    fields = (
        "natural_nll",
        "brier",
        "macro_ap",
        "macro_state_f1",
        "macro_event_f1_collar_0",
        "macro_event_f1_collar_2_native_frames",
        "key_state_micro_accuracy",
        "joint_exact_match_accuracy",
        "predicted_positive_rate_macro",
    )
    return {
        field: float(left_summary[field] - right_summary[field])
        for field in fields
    }


def validate_checkpoint_receipts(training_dir: Path) -> tuple[list[dict[str, Any]], dict[int, str]]:
    receipts: list[dict[str, Any]] = []
    state_hashes: dict[int, str] = {}
    for epoch in range(1, 21):
        path = training_dir / f"checkpoint_epoch_{epoch:02d}.r2.json"
        receipt = read_json(path)
        if receipt["checkpoint"] != f"checkpoint_epoch_{epoch:02d}":
            raise ValueError(f"{path} records the wrong epoch")
        if int(receipt["object_count"]) != 4 or len(receipt["objects"]) != 4:
            raise ValueError(f"{path} does not contain the four frozen checkpoint objects")
        objects = {item["path"]: item for item in receipt["objects"]}
        if set(objects) != {"manifest.json", "rng.pt", "state/.metadata", "state/__0_0.distcp"}:
            raise ValueError(f"{path} has an unexpected checkpoint object set")
        if int(receipt["total_bytes"]) != sum(int(item["bytes"]) for item in receipt["objects"]):
            raise ValueError(f"{path} total byte count is inconsistent")
        if "full independent R2 byte-stream" not in receipt["verification"]:
            raise ValueError(f"{path} lacks the required R2 stream-verification declaration")
        state_hashes[epoch] = objects["state/__0_0.distcp"]["sha256"]
        receipts.append({
            "epoch": epoch,
            "path": str(path),
            "sha256": sha256_file(path),
            "state_object_sha256": state_hashes[epoch],
            "total_bytes": receipt["total_bytes"],
            "r2_prefix": receipt["r2_prefix"],
        })
    return receipts, state_hashes


def validate_bundle(
    training_dir: Path,
    paper_eval_dir: Path,
    vpt_small_validation_path: Path,
) -> dict[str, Any]:
    complete = read_json(training_dir / "complete.json")
    if not complete["completed_endpoint"]:
        raise ValueError("paper-IDM completion marker is not final")
    if int(complete["epochs_recorded"]) != 20 or int(complete["optimizer_steps"]) != 2340:
        raise ValueError("paper-IDM endpoint is not 20 epochs / 2,340 steps")

    history = read_json(training_dir / "training_history.json")
    if len(history) != 20 or [row["epoch"] for row in history] != list(range(1, 21)):
        raise ValueError("paper-IDM training history is not the exact 20-epoch sequence")
    if any(not np.isfinite(row["train_nll_mean_microbatch"]) for row in history):
        raise ValueError("paper-IDM training history contains nonfinite loss")
    if any(not np.isfinite(row["validation"]["natural_nll"]) for row in history):
        raise ValueError("paper-IDM validation history contains nonfinite loss")
    selected = min(history, key=lambda row: row["validation"]["natural_nll"])
    if int(selected["epoch"]) != 3:
        raise ValueError("paper-IDM selected checkpoint is not epoch 3")

    receipts, state_hashes = validate_checkpoint_receipts(training_dir)
    frozen = read_json(vpt_small_validation_path)
    prior_entries = frozen["reports"]

    paper_paths = {
        "paper_final": (
            paper_eval_dir / "final_epoch20_val_a.json",
            paper_eval_dir / "final_epoch20_val_a_preds.npz",
        ),
        "paper_selected": (
            paper_eval_dir / "selected_epoch03_val_a.json",
            paper_eval_dir / "selected_epoch03_val_a_preds.npz",
        ),
    }
    prior_paths = {
        name: (Path(entry["path"]), Path(entry["sidecar"]["path"]))
        for name, entry in prior_entries.items()
    }
    reports: dict[str, dict[str, Any]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for name, (report_path, sidecar_path) in paper_paths.items():
        report, sidecar, entry = report_entry(report_path, sidecar_path)
        reports[name], arrays[name], evidence[name] = report, sidecar, entry
    for name, (report_path, sidecar_path) in prior_paths.items():
        frozen_entry = prior_entries[name]
        report, sidecar, entry = report_entry(
            report_path,
            sidecar_path,
            expected_report_sha256=frozen_entry["sha256"],
            expected_sidecar_sha256=frozen_entry["sidecar"]["sha256"],
        )
        reports[name], arrays[name], evidence[name] = report, sidecar, entry

    authority = arrays["vpt_final"]
    mismatched = [name for name, value in arrays.items() if not equal_support(authority, value)]
    if mismatched:
        raise ValueError(f"common-support mismatch: {mismatched}")

    if reports["paper_final"]["weights"]["state_object_sha256"] != state_hashes[20]:
        raise ValueError("paper final evaluation/checkpoint state SHA-256 mismatch")
    if reports["paper_selected"]["weights"]["state_object_sha256"] != state_hashes[3]:
        raise ValueError("paper selected evaluation/checkpoint state SHA-256 mismatch")
    if reports["paper_final"]["weights"]["checkpoint_receipt_sha256"] != receipts[19]["sha256"]:
        raise ValueError("paper final checkpoint receipt SHA-256 mismatch")
    if reports["paper_selected"]["weights"]["checkpoint_receipt_sha256"] != receipts[2]["sha256"]:
        raise ValueError("paper selected checkpoint receipt SHA-256 mismatch")

    xla_text = (training_dir / "xla_metrics.txt").read_text(encoding="utf-8")
    uncached_match = re.search(r"Counter: UncachedCompile\s+Value: (\d+)", xla_text)
    cached_match = re.search(r"Counter: CachedCompile\s+Value: (\d+)", xla_text)
    if uncached_match is None or cached_match is None:
        raise ValueError("paper-IDM XLA receipt lacks compile counters")
    uncached_compile = int(uncached_match.group(1))
    cached_compile = int(cached_match.group(1))
    stable_xla = uncached_compile == 11 and cached_compile > 100_000

    implementation_gates = {
        "exact_endpoint": True,
        "finite_values": True,
        "nondegenerate_loss_reduction": (
            history[-1]["train_nll_mean_microbatch"]
            < history[0]["train_nll_mean_microbatch"]
        ),
        "stable_xla_graphs": stable_xla,
        "all_twenty_checkpoints_stream_verified": len(receipts) == 20,
        "all_seven_sidecars_exact_common_support": True,
        "paper_final_ranking_above_prevalence": (
            reports["paper_final"]["aggregate"]["macro_ap"]
            > reports["paper_final"]["aggregate"]["prevalence_macro"]
        ),
    }
    implementation_success = all(implementation_gates.values())

    return {
        "schema_version": "madeleine.vpt-paper-idm-tier-b-release-validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": "vpt_paper_idm_482133390_tier_b_13p45h_s0",
        "parameter_count": 482_133_390,
        "implementation_success": implementation_success,
        "matched_tier_b_scientific_result": "negative",
        "phase_3_maximum_generation_build_eligible": implementation_success,
        "phase_4_maximum_training_authorized_by_this_receipt": False,
        "implementation_gates": implementation_gates,
        "endpoint": {
            "epochs": complete["epochs_recorded"],
            "optimizer_steps": complete["optimizer_steps"],
            "elapsed_seconds": complete["elapsed_seconds"],
            "sample_order_sha256": complete["sample_order_sha256"],
            "train_nll_epoch_1": history[0]["train_nll_mean_microbatch"],
            "train_nll_epoch_20": history[-1]["train_nll_mean_microbatch"],
            "selected_epoch": selected["epoch"],
            "selected_validation_nll": selected["validation"]["natural_nll"],
            "final_validation_nll": history[-1]["validation"]["natural_nll"],
        },
        "xla": {
            "uncached_compile": uncached_compile,
            "cached_compile": cached_compile,
        },
        "support": {
            "rows": int(len(authority["y_true"])),
            "streams": int(len(authority["session_lengths"])),
            "unique_source_rows": int(len(np.unique(authority["source_row_index"]))),
            "all_seven_sidecars_identical": True,
            "probabilities_finite": True,
        },
        "checkpoint_receipts": {
            "count": len(receipts),
            "all_full_stream_verified": True,
            "final_state_object_sha256": state_hashes[20],
            "selected_state_object_sha256": state_hashes[3],
            "receipts": receipts,
        },
        "reports": evidence,
        "metrics": {
            name: metric_summary(report)
            for name, report in reports.items()
        },
        "deltas": {
            "paper_final_minus_vpt_small_final": metric_delta(
                reports["paper_final"], reports["vpt_final"]
            ),
            "paper_final_minus_gru_112m95_final": metric_delta(
                reports["paper_final"], reports["gru_112m95_final"]
            ),
            "paper_final_minus_gru_36m9_final": metric_delta(
                reports["paper_final"], reports["gru_36m9_final"]
            ),
            "paper_selected_minus_vpt_small_selected": metric_delta(
                reports["paper_selected"], reports["vpt_selected"]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--paper-eval-dir", type=Path, required=True)
    parser.add_argument("--vpt-small-validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    validation = validate_bundle(
        args.training_dir,
        args.paper_eval_dir,
        args.vpt_small_validation,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "implementation_success": validation["implementation_success"],
        "matched_tier_b_scientific_result": validation["matched_tier_b_scientific_result"],
        "rows": validation["support"]["rows"],
        "paper_final_macro_ap": validation["metrics"]["paper_final"]["macro_ap"],
        "vpt_small_final_macro_ap": validation["metrics"]["vpt_final"]["macro_ap"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
