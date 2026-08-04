#!/usr/bin/env python3
"""Frozen positive-slope calibration and scoring for the VPT-small branch.

This module deliberately has no threshold search.  It fits seven independent
bounded affine transforms in logit space on a declared C1 sidecar and applies
them at the unchanged probability threshold of 0.5.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import platform
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import scipy
from scipy.optimize import minimize

from badeline.metrics import per_key_ap, summarize
from data.schema import KEY_ORDER
from experiments.eval_vpt_small import equal_mass_ece, json_ready, sha256_file
from experiments.keypress_accuracy import score_sidecar


SCHEMA_VERSION = "madeleine.vpt-small-positive-affine-calibration.v1"
START = np.asarray([1.0, 0.0], dtype=np.float64)
BOUNDS = ((0.05, 20.0), (-12.0, 12.0))
CLIP_EPSILON = 1e-6
OPTIMIZER_OPTIONS = {"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000, "maxls": 50}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _implementation_identity(repo: Path) -> dict[str, Any]:
    commit = subprocess_run(
        ["git", "rev-parse", "HEAD"], cwd=repo
    ).strip()
    with io.StringIO() as buffer, contextlib.redirect_stdout(buffer):
        np.show_config()
        blas_runtime = buffer.getvalue()
    return {
        "commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_runtime": blas_runtime,
        "optimizer": "scipy.optimize.minimize/L-BFGS-B",
        "options": OPTIMIZER_OPTIONS,
        "clip_epsilon": CLIP_EPSILON,
    }


def subprocess_run(command: list[str], *, cwd: Path) -> str:
    import subprocess

    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _required_sidecar(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
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
            raise ValueError(f"prediction sidecar is missing {sorted(missing)}")
        result = {name: np.asarray(archive[name]) for name in archive.files}
    truth = np.asarray(result["y_true"])
    probability = np.asarray(result["y_prob"])
    active = np.asarray(result["input_active"])
    if truth.shape != probability.shape or truth.ndim != 2:
        raise ValueError("truth and probability must share shape [N,K]")
    if truth.shape[1] != len(KEY_ORDER) or active.shape != (len(truth),):
        raise ValueError("sidecar key count or activity shape differs")
    if not np.all(np.isin(truth, (0, 1))):
        raise ValueError("truth is not binary")
    if not np.all(np.isfinite(probability)):
        raise ValueError("probability contains nonfinite values")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probability lies outside [0,1]")
    if not np.all(np.isin(active, (0, 1))):
        raise ValueError("input_active is not binary")
    if int(np.asarray(result["session_lengths"], dtype=np.int64).sum()) != len(truth):
        raise ValueError("session lengths do not cover the sidecar")
    return result


def affine_probabilities(
    probability: np.ndarray, parameters: np.ndarray
) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != len(KEY_ORDER):
        raise ValueError("probability must have shape [N,7]")
    if parameters.shape != (len(KEY_ORDER), 2):
        raise ValueError("parameters must have shape [7,2]")
    if not np.all(np.isfinite(probability)) or not np.all(np.isfinite(parameters)):
        raise ValueError("calibration inputs must be finite")
    if np.any(parameters[:, 0] <= 0):
        raise ValueError("calibration slope must remain positive")
    clipped = np.clip(probability, CLIP_EPSILON, 1.0 - CLIP_EPSILON)
    logits = np.log(clipped) - np.log1p(-clipped)
    transformed = logits * parameters[None, :, 0] + parameters[None, :, 1]
    calibrated = np.empty_like(transformed)
    positive = transformed >= 0
    calibrated[positive] = 1.0 / (1.0 + np.exp(-transformed[positive]))
    exp_value = np.exp(transformed[~positive])
    calibrated[~positive] = exp_value / (1.0 + exp_value)
    return calibrated


def _per_key_nll(truth: np.ndarray, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    truth = np.asarray(truth, dtype=np.float64)
    return np.mean(
        -(truth * np.log(clipped) + (1.0 - truth) * np.log1p(-clipped)), axis=0
    )


def _fit_one(truth: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), CLIP_EPSILON, 1.0 - CLIP_EPSILON)
    logits = np.log(clipped) - np.log1p(-clipped)
    truth = np.asarray(truth, dtype=np.float64)
    if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(truth)):
        raise ValueError("nonfinite calibration column")
    if not np.any(truth == 1.0) or not np.any(truth == 0.0):
        raise ValueError("each calibration key needs positive and negative rows")

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        scores = theta[0] * logits + theta[1]
        loss = float(np.mean(np.logaddexp(0.0, scores) - truth * scores))
        sigmoid = np.empty_like(scores)
        positive = scores >= 0
        sigmoid[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
        exp_value = np.exp(scores[~positive])
        sigmoid[~positive] = exp_value / (1.0 + exp_value)
        residual = sigmoid - truth
        gradient = np.asarray(
            [np.mean(residual * logits), np.mean(residual)], dtype=np.float64
        )
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("nonfinite L-BFGS-B objective")
        return loss, gradient

    result = minimize(
        objective,
        START.copy(),
        method="L-BFGS-B",
        jac=True,
        bounds=BOUNDS,
        options=dict(OPTIMIZER_OPTIONS),
    )
    if not result.success or not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"L-BFGS-B calibration failed: {result.message}")
    at_bound = bool(
        np.isclose(result.x[0], BOUNDS[0][0])
        or np.isclose(result.x[0], BOUNDS[0][1])
        or np.isclose(result.x[1], BOUNDS[1][0])
        or np.isclose(result.x[1], BOUNDS[1][1])
    )
    return {
        "slope": float(result.x[0]),
        "bias": float(result.x[1]),
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "gradient": np.asarray(result.jac, dtype=float).tolist(),
        "message": str(result.message),
        "success": bool(result.success),
        "at_bound": at_bound,
    }


def fit_calibrators(
    truth: np.ndarray, probability: np.ndarray, active: np.ndarray
) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.uint8)
    probability = np.asarray(probability, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if truth.shape != probability.shape or truth.shape[1] != len(KEY_ORDER):
        raise ValueError("calibration truth/probability shape mismatch")
    if active.shape != (len(truth),) or not np.any(active):
        raise ValueError("calibration activity mask is empty or misaligned")
    if not np.all(np.isfinite(probability)):
        raise ValueError("calibration probability contains nonfinite values")
    gated_truth = truth[active]
    gated_probability = probability[active]
    fits = [_fit_one(gated_truth[:, column], gated_probability[:, column]) for column in range(len(KEY_ORDER))]
    parameters = np.asarray(
        [[fit["slope"], fit["bias"]] for fit in fits], dtype=np.float64
    )
    calibrated = affine_probabilities(gated_probability, parameters)
    raw_ap = per_key_ap(gated_truth, gated_probability)
    calibrated_ap = per_key_ap(gated_truth, calibrated)
    for key in KEY_ORDER:
        if not np.isclose(raw_ap[key], calibrated_ap[key], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"positive-slope calibration changed AP for {key}")
    return {
        "parameters": {
            key: {**fits[column], "start": START.tolist(), "bounds": [list(v) for v in BOUNDS]}
            for column, key in enumerate(KEY_ORDER)
        },
        "parameter_matrix": parameters,
        "raw_nll_per_key": _per_key_nll(gated_truth, gated_probability),
        "calibrated_nll_per_key": _per_key_nll(gated_truth, calibrated),
        "raw_brier": float(np.mean((gated_probability - gated_truth) ** 2)),
        "calibrated_brier": float(np.mean((calibrated - gated_truth) ** 2)),
        "raw_ece": equal_mass_ece(gated_truth, gated_probability),
        "calibrated_ece": equal_mass_ece(gated_truth, calibrated),
        "raw_ap": raw_ap,
        "calibrated_ap": calibrated_ap,
        "rows": int(len(gated_truth)),
    }


def parameter_matrix(receipt: dict[str, Any]) -> np.ndarray:
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported calibrator receipt")
    rows = []
    for key in KEY_ORDER:
        entry = receipt["parameters"][key]
        rows.append([entry["slope"], entry["bias"]])
    return np.asarray(rows, dtype=np.float64)


def write_calibrated_sidecar(
    source: Path, destination: Path, parameters: np.ndarray
) -> dict[str, Any]:
    arrays = _required_sidecar(source)
    calibrated = affine_probabilities(arrays["y_prob"], parameters).astype(np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays["y_prob"] = calibrated
    np.savez_compressed(destination, **arrays)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "source_sha256": sha256_file(source),
    }


def score_prediction_sidecar(
    sidecar: Path, *, surface: str, variant: str
) -> dict[str, Any]:
    arrays = _required_sidecar(sidecar)
    truth = arrays["y_true"].astype(np.uint8, copy=False)
    probability = arrays["y_prob"].astype(np.float64, copy=False)
    active = arrays["input_active"].astype(bool, copy=False)
    gated_truth = truth[active]
    gated_probability = probability[active]
    detail = summarize(
        truth,
        probability,
        boundaries=np.asarray(arrays["session_lengths"], dtype=np.int64).tolist(),
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
    prevalence = gated_truth.mean(axis=0)
    ppr = predicted.mean(axis=0)
    return json_ready({
        "schema_version": "madeleine.vpt-small-calibrated-eval.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "surface": surface,
        "variant": variant,
        "threshold": 0.5,
        "sidecar": {"path": str(sidecar), "bytes": sidecar.stat().st_size, "sha256": sha256_file(sidecar)},
        "data": {"rows": len(truth), "active_rows": int(active.sum()), "streams": len(arrays["session_lengths"])},
        "natural_nll": float(_per_key_nll(gated_truth, gated_probability).sum()),
        "natural_nll_per_key": dict(zip(KEY_ORDER, _per_key_nll(gated_truth, gated_probability), strict=True)),
        "brier": float(np.mean((gated_probability - gated_truth) ** 2)),
        "equal_mass_ece": equal_mass_ece(gated_truth, gated_probability),
        "aggregate": {
            "prevalence_macro": float(prevalence.mean()),
            "prevalence_per_key": dict(zip(KEY_ORDER, prevalence, strict=True)),
            "predicted_positive_rate_macro": float(ppr.mean()),
            "predicted_positive_rate_per_key": dict(zip(KEY_ORDER, ppr, strict=True)),
            "predicted_positive_to_prevalence_per_key": {
                key: float(ppr[i] / prevalence[i]) if prevalence[i] > 0 else float("nan")
                for i, key in enumerate(KEY_ORDER)
            },
            "macro_ap": float(np.nanmean(list(detail["per_key_ap"].values()))),
            "macro_state_f1": float(np.nanmean(state_f1)),
            "macro_state_precision": float(np.nanmean(precision)),
            "macro_state_recall": float(np.nanmean(recall)),
            "macro_event_f1_collar_0": float(np.nanmean([detail["transition_f1_at_0.5"][key]["event"]["f1"] for key in KEY_ORDER])),
            "macro_event_f1_collar_2_native_frames": float(np.nanmean([detail["transition_f1_at_0.5_collars"]["2"][key]["event"]["f1"] for key in KEY_ORDER])),
        },
        "per_key": {
            key: {
                "ap": detail["per_key_ap"][key],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "state_f1": float(state_f1[i]),
                "prevalence": float(prevalence[i]),
                "predicted_positive_rate": float(ppr[i]),
            }
            for i, key in enumerate(KEY_ORDER)
        },
        "metrics": detail,
        "key_state_accuracy": score_sidecar(sidecar, threshold=0.5),
    })


def calibration_validity_guard(
    raw_report: dict[str, Any], calibrated_report: dict[str, Any]
) -> dict[str, Any]:
    raw_nll = float(raw_report["natural_nll"])
    calibrated_nll = float(calibrated_report["natural_nll"])
    raw_brier = float(raw_report["brier"])
    calibrated_brier = float(calibrated_report["brier"])
    tolerance = 1e-12
    nll_no_worse = calibrated_nll <= raw_nll + tolerance
    brier_no_worse = calibrated_brier <= raw_brier + tolerance
    one_improves = calibrated_nll < raw_nll - tolerance or calibrated_brier < raw_brier - tolerance
    return {
        "pass": bool(nll_no_worse and brier_no_worse and one_improves),
        "raw_nll": raw_nll,
        "calibrated_nll": calibrated_nll,
        "nll_no_worse": bool(nll_no_worse),
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "brier_no_worse": bool(brier_no_worse),
        "at_least_one_strictly_improves": bool(one_improves),
    }


def candidate_decision(
    *,
    raw_vpt: dict[str, Any],
    calibrated_vpt: dict[str, Any],
    gru_112m95: dict[str, Any],
    gru_36m9: dict[str, Any],
) -> dict[str, Any]:
    comparators = [gru_112m95, gru_36m9]
    better_gru = max(comparators, key=lambda report: float(report["aggregate"]["macro_ap"]))
    better_name = str(better_gru.get("variant", better_gru.get("weights", {}).get("population", "better_final_gru")))
    calibrated = calibrated_vpt["aggregate"]
    comparator = better_gru["aggregate"]
    clause_1 = float(calibrated["macro_ap"]) >= float(comparator["macro_ap"]) + 0.010
    clause_2 = float(calibrated["macro_state_f1"]) >= float(comparator["macro_state_f1"]) - 0.010
    clause_3 = float(calibrated["macro_event_f1_collar_2_native_frames"]) >= float(comparator["macro_event_f1_collar_2_native_frames"]) - 0.010
    vpt_per_key = calibrated_vpt["per_key"]
    gru_per_key = gru_112m95["per_key"]
    improved_keys = [key for key in KEY_ORDER if float(vpt_per_key[key]["ap"]) > float(gru_per_key[key]["ap"])]
    clause_4 = len(improved_keys) >= 4
    accuracy = calibrated_vpt["key_state_accuracy"]
    micro = float(accuracy["key_state_micro_accuracy"])
    joint = float(accuracy["joint_exact_match_accuracy"])
    always_micro = float(accuracy["always_released_key_state_micro_accuracy"])
    always_joint = float(accuracy["always_released_joint_exact_match_accuracy"])
    clause_5 = micro > always_micro and joint >= always_joint - 0.010
    key_coverage = {}
    for key in KEY_ORDER:
        entry = vpt_per_key[key]
        prevalence = float(entry["prevalence"])
        ppr = float(entry["predicted_positive_rate"])
        ratio = ppr / prevalence if prevalence > 0 else float("nan")
        active = prevalence > 0
        passed = (not active) or (float(entry["recall"]) > 0 and 0.5 <= ratio <= 2.0)
        key_coverage[key] = {
            "active": active,
            "recall": float(entry["recall"]),
            "ppr_prevalence_ratio": ratio,
            "pass": bool(passed),
        }
    clause_6 = all(entry["pass"] for entry in key_coverage.values())
    guard_7 = calibration_validity_guard(raw_vpt, calibrated_vpt)
    diagnosis: dict[str, Any] = {}
    for key in KEY_ORDER:
        raw_key = raw_vpt["per_key"][key]
        calibrated_key = calibrated_vpt["per_key"][key]
        prevalence = float(calibrated_key["prevalence"])
        ap = float(calibrated_key["ap"])
        recall = float(calibrated_key["recall"])
        ppr = float(calibrated_key["predicted_positive_rate"])
        ratio = ppr / prevalence if prevalence > 0 else float("nan")
        rate_recovered = prevalence > 0 and recall > 0 and 0.5 <= ratio <= 2.0
        state_preserved = float(calibrated_key["state_f1"]) >= float(raw_key["state_f1"]) - 0.010
        if ap <= prevalence or not rate_recovered:
            label = "representation-limited"
        elif state_preserved:
            label = "positioning recovered"
        else:
            label = "inconclusive"
        diagnosis[key] = {
            "label": label,
            "ap": ap,
            "prevalence": prevalence,
            "calibrated_recall": recall,
            "calibrated_ppr_prevalence_ratio": ratio,
            "raw_state_f1": float(raw_key["state_f1"]),
            "calibrated_state_f1": float(calibrated_key["state_f1"]),
            "rate_recovered": rate_recovered,
            "state_f1_preserved_within_0p010": state_preserved,
        }
    clauses = {
        "1_macro_ap": {"pass": bool(clause_1), "candidate": calibrated["macro_ap"], "comparator": comparator["macro_ap"], "required_margin": 0.010},
        "2_state_f1": {"pass": bool(clause_2), "candidate": calibrated["macro_state_f1"], "comparator": comparator["macro_state_f1"], "allowed_deficit": 0.010},
        "3_event_f1_collar_2": {"pass": bool(clause_3), "candidate": calibrated["macro_event_f1_collar_2_native_frames"], "comparator": comparator["macro_event_f1_collar_2_native_frames"], "allowed_deficit": 0.010},
        "4_per_key_ap": {"pass": bool(clause_4), "improved_keys": improved_keys, "count": len(improved_keys), "required": 4},
        "5_key_state_accuracy": {"pass": bool(clause_5), "micro": micro, "always_micro": always_micro, "joint": joint, "always_joint": always_joint},
        "6_key_coverage_rate": {"pass": bool(clause_6), "keys": key_coverage},
        "7_calibration_validity": guard_7,
    }
    return json_ready({
        "schema_version": "madeleine.vpt-small-calibration-phase-a-decision.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "better_final_gru": better_name,
        "clauses": clauses,
        "diagnosis": diagnosis,
        "pass": all(bool(entry["pass"]) for entry in clauses.values()),
    })


def fit_from_sidecar(
    c1_sidecar: Path,
    *,
    c1_receipt: Path,
    c1_eval_report: Path,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    e1_command: Path,
    out: Path,
    calibrated_sidecar: Path,
    repo: Path,
) -> dict[str, Any]:
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != expected_checkpoint_sha256:
        raise RuntimeError("parent checkpoint SHA-256 mismatch")
    capture = json.loads(c1_receipt.read_text(encoding="utf-8"))
    if capture.get("role") != "c1" or capture.get("decision") != "accepted":
        raise RuntimeError("calibration fit requires an accepted C1 receipt")
    if capture.get("model_accessed") is not False or capture.get("violations"):
        raise RuntimeError("accepted C1 receipt is not model-blind and violation-free")
    command_bytes = e1_command.read_bytes()
    if not command_bytes.strip():
        raise ValueError("frozen E1 command is empty")
    arrays = _required_sidecar(c1_sidecar)
    evaluation = json.loads(c1_eval_report.read_text(encoding="utf-8"))
    if evaluation.get("schema_version") != "madeleine.vpt-small-eval.v1":
        raise RuntimeError("C1 evaluation report has an unsupported schema")
    report_checkpoint = evaluation.get("weights", {}).get("sha256")
    if report_checkpoint != checkpoint_hash:
        raise RuntimeError("C1 evaluation report is not bound to the parent checkpoint")
    report_sidecar = evaluation.get("sidecar", {})
    if report_sidecar.get("sha256") != sha256_file(c1_sidecar):
        raise RuntimeError("C1 evaluation report is not bound to the C1 sidecar")
    capture_manifest = capture.get("derived", {}).get("build_manifest_sha256")
    report_manifest = evaluation.get("data", {}).get("manifest_sha256")
    if not capture_manifest or report_manifest != capture_manifest:
        raise RuntimeError("C1 sidecar is not bound to the accepted derived manifest")
    support = capture.get("support", {})
    if int(evaluation.get("data", {}).get("rows", -1)) != int(support.get("rows", -2)):
        raise RuntimeError("C1 sidecar row count differs from the accepted support")
    active_rows = int(np.asarray(arrays["input_active"], dtype=bool).sum())
    if active_rows != int(support.get("active_rows", -1)):
        raise RuntimeError("C1 sidecar active-row count differs from the accepted support")
    fit = fit_calibrators(arrays["y_true"], arrays["y_prob"], arrays["input_active"])
    parameters = fit.pop("parameter_matrix")
    calibrated = write_calibrated_sidecar(c1_sidecar, calibrated_sidecar, parameters)
    receipt = json_ready({
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit_role": "c1_only",
        "threshold": 0.5,
        "checkpoint": {"path": str(checkpoint), "bytes": checkpoint.stat().st_size, "sha256": checkpoint_hash},
        "c1": {
            "sidecar": str(c1_sidecar),
            "sidecar_sha256": sha256_file(c1_sidecar),
            "capture_receipt": str(c1_receipt),
            "capture_receipt_sha256": sha256_file(c1_receipt),
            "evaluation_report": str(c1_eval_report),
            "evaluation_report_sha256": sha256_file(c1_eval_report),
            "derived_manifest_sha256": report_manifest,
            "rows": len(arrays["y_true"]),
            "active_rows": active_rows,
        },
        "e1_command": {"path": str(e1_command), "bytes": len(command_bytes), "sha256": _sha256_bytes(command_bytes)},
        "implementation": _implementation_identity(repo),
        "parameters": fit["parameters"],
        "diagnostics": {key: value for key, value in fit.items() if key != "parameters"},
        "calibrated_c1_sidecar": calibrated,
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    out.write_bytes(payload)
    out.with_suffix(out.suffix + ".sha256").write_text(
        f"{_sha256_bytes(payload)}  {out.name}\n", encoding="utf-8"
    )
    return receipt


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--c1-sidecar", type=Path, required=True)
    fit.add_argument("--c1-receipt", type=Path, required=True)
    fit.add_argument("--c1-eval-report", type=Path, required=True)
    fit.add_argument("--checkpoint", type=Path, required=True)
    fit.add_argument("--expected-checkpoint-sha256", required=True)
    fit.add_argument("--e1-command", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--calibrated-sidecar", type=Path, required=True)
    fit.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    apply = sub.add_parser("apply")
    apply.add_argument("--sidecar", type=Path, required=True)
    apply.add_argument("--calibrator", type=Path, required=True)
    apply.add_argument("--out", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--sidecar", type=Path, required=True)
    score.add_argument("--surface", required=True)
    score.add_argument("--variant", required=True)
    score.add_argument("--out", type=Path, required=True)
    decide = sub.add_parser("decide")
    decide.add_argument("--raw-vpt", type=Path, required=True)
    decide.add_argument("--calibrated-vpt", type=Path, required=True)
    decide.add_argument("--gru-112m95", type=Path, required=True)
    decide.add_argument("--gru-36m9", type=Path, required=True)
    decide.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "fit":
        result = fit_from_sidecar(
            args.c1_sidecar,
            c1_receipt=args.c1_receipt,
            c1_eval_report=args.c1_eval_report,
            checkpoint=args.checkpoint,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            e1_command=args.e1_command,
            out=args.out,
            calibrated_sidecar=args.calibrated_sidecar,
            repo=args.repo,
        )
    elif args.command == "apply":
        receipt = json.loads(args.calibrator.read_text(encoding="utf-8"))
        result = write_calibrated_sidecar(args.sidecar, args.out, parameter_matrix(receipt))
    elif args.command == "score":
        result = score_prediction_sidecar(args.sidecar, surface=args.surface, variant=args.variant)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        reports = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in {
                "raw_vpt": args.raw_vpt,
                "calibrated_vpt": args.calibrated_vpt,
                "gru_112m95": args.gru_112m95,
                "gru_36m9": args.gru_36m9,
            }.items()
        }
        result = candidate_decision(**reports)
        result["inputs"] = {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in {
                "raw_vpt": args.raw_vpt,
                "calibrated_vpt": args.calibrated_vpt,
                "gru_112m95": args.gru_112m95,
                "gru_36m9": args.gru_36m9,
            }.items()
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
