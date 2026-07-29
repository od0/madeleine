#!/usr/bin/env python3
"""Freeze the matched C-versus-D downstream decision on y4n later-eight.

This scorer has no interface for B1 or the sealed untouched session.  It
validates and re-scores both final-only fixed-0.5 releases from their saved
prediction arrays, applies the predeclared D-versus-C replication rule, and
includes the historical ImageNet feature result only as explicitly confounded
context.  No threshold, calibration parameter, or checkpoint is fit or
selected here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from data.schema import KEY_ORDER
from experiments.eval_wild_provisional_gru import (
    Y4N_BASE_SESSION_IDS,
    Y4N_FRAMES,
    Y4N_STREAM_IDS,
    Y4N_STREAM_LENGTHS,
    Y4N_TRUTH_SHA256,
)


SCHEMA_VERSION = "madeleine.dynamics-cd-downstream-decision.v1"
MARKER_SCHEMA_VERSION = "madeleine.dynamics-cd-downstream-decision-complete.v1"
STUDY_ID = "photon_inspired_celeste_dynamics_exploratory_cd_s0_v1"
SURFACE = "mapped_y4n_later_eight"
REFERENCE_A_RUN_ID = "nitrogen_full_210train_y4n_holdout_26m_128x3_s0"
ARM_RUN_IDS = {
    "C": "dynamics_c_full_210train_y4n_holdout_26m_128x3_s0",
    "D": "dynamics_d_full_210train_y4n_holdout_26m_128x3_s0",
}
MINIMUM_PLUS_MINUS_2_DELTA = 0.010
MAXIMUM_EXACT_LOSS = 0.002
MAXIMUM_MACRO_AP_LOSS = 0.005
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTRACT_SCHEMA_VERSION = "madeleine.dynamics-pretraining-exploratory-cd.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _canonical_array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if os.path.lexists(path) or os.path.lexists(temporary):
        raise ValueError(f"refusing existing decision artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _nested_close(
    observed: object,
    expected: object,
    *,
    field: str,
    tolerance: float = 1e-10,
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ValueError(f"{field} object keys differ")
        for key, value in expected.items():
            _nested_close(
                observed[key], value, field=f"{field}.{key}", tolerance=tolerance
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"{field} list shape differs")
        for index, value in enumerate(expected):
            _nested_close(
                observed[index],
                value,
                field=f"{field}[{index}]",
                tolerance=tolerance,
            )
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(observed, (int, float)) or isinstance(observed, bool):
            raise ValueError(f"{field} numeric type differs")
        if not math.isclose(
            float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance
        ):
            raise ValueError(f"{field} differs")
        return
    if observed != expected:
        raise ValueError(f"{field} differs")


def _stream_slices(lengths: Sequence[int]) -> list[slice]:
    result: list[slice] = []
    start = 0
    for length in lengths:
        end = start + int(length)
        result.append(slice(start, end))
        start = end
    return result


def load_later8_sidecar(
    path: Path,
    *,
    permit_superset: bool,
) -> dict[str, Any]:
    """Load exact later-eight support, optionally deriving it from all y4n."""

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "y_true", "y_prob", "input_active", "session_lengths", "session_ids"
        }
        if set(archive.files) != required:
            raise ValueError(f"{path}: prediction-sidecar member set differs")
        truth = np.asarray(archive["y_true"])
        probability = np.asarray(archive["y_prob"])
        active = np.asarray(archive["input_active"])
        lengths = np.asarray(archive["session_lengths"])
        session_ids = np.asarray(archive["session_ids"])
    if truth.dtype != np.uint8 or probability.dtype != np.float32:
        raise ValueError(f"{path}: truth/probability dtypes differ")
    if active.dtype != np.uint8 or lengths.dtype != np.int64:
        raise ValueError(f"{path}: activity/boundary dtypes differ")
    if truth.ndim != 2 or truth.shape[1] != len(KEY_ORDER):
        raise ValueError(f"{path}: truth shape differs")
    if probability.shape != truth.shape or active.shape != (len(truth),):
        raise ValueError(f"{path}: sidecar arrays are not aligned")
    if lengths.ndim != 1 or int(lengths.sum()) != len(truth):
        raise ValueError(f"{path}: stream lengths do not cover the arrays")
    ids = session_ids.tolist()
    if len(ids) != len(lengths) or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: stream identity receipt differs")
    if not np.all(np.isin(truth, (0, 1))) or not np.all(np.isin(active, (0, 1))):
        raise ValueError(f"{path}: truth or activity mask is non-binary")
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError(f"{path}: probabilities are not finite in [0,1]")
    if ids == Y4N_STREAM_IDS:
        selected = list(range(len(ids)))
    else:
        if not permit_superset:
            raise ValueError(f"{path}: C/D release is not exact later-eight support")
        if not set(Y4N_STREAM_IDS).issubset(ids):
            raise ValueError(f"{path}: historical sidecar lacks later-eight streams")
        selected = [ids.index(value) for value in Y4N_STREAM_IDS]
    slices = _stream_slices(lengths.tolist())
    truth = np.concatenate([truth[slices[index]] for index in selected])
    probability = np.concatenate(
        [probability[slices[index]] for index in selected]
    )
    active = np.concatenate([active[slices[index]] for index in selected])
    selected_lengths = [int(lengths[index]) for index in selected]
    if truth.shape != (Y4N_FRAMES, len(KEY_ORDER)):
        raise ValueError(f"{path}: later-eight tensor support differs")
    if selected_lengths != Y4N_STREAM_LENGTHS or not np.all(active == 1):
        raise ValueError(f"{path}: later-eight boundaries/activity differ")
    truth_sha256 = _canonical_array_sha256(truth)
    if truth_sha256 != Y4N_TRUTH_SHA256:
        raise ValueError(f"{path}: later-eight mapped truth differs")
    return {
        "truth": truth,
        "probability": probability,
        "active": active,
        "lengths": selected_lengths,
        "stream_ids": list(Y4N_STREAM_IDS),
        "support": {
            "rows": Y4N_FRAMES,
            "streams": len(Y4N_STREAM_IDS),
            "stream_ids": list(Y4N_STREAM_IDS),
            "stream_lengths": list(Y4N_STREAM_LENGTHS),
            "truth_sha256": truth_sha256,
            "sidecar_sha256": sha256_file(path),
            "finite_aligned_arrays": True,
        },
    }


def promotion_decision(
    c_metrics: Mapping[str, Any], d_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the frozen D-versus-C seed-zero replication rule."""

    def number(metrics: Mapping[str, Any], *keys: str) -> float:
        value: object = metrics
        for key in keys:
            if not isinstance(value, Mapping) or key not in value:
                raise ValueError(f"missing decision metric: {'.'.join(keys)}")
            value = value[key]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"non-finite decision metric: {'.'.join(keys)}")
        return float(value)

    c_plus2 = number(
        c_metrics, "events_fixed_0_5", "plus_minus_2", "macro_combined_f1"
    )
    d_plus2 = number(
        d_metrics, "events_fixed_0_5", "plus_minus_2", "macro_combined_f1"
    )
    c_exact = number(
        c_metrics, "events_fixed_0_5", "exact", "macro_combined_f1"
    )
    d_exact = number(
        d_metrics, "events_fixed_0_5", "exact", "macro_combined_f1"
    )
    c_ap = number(c_metrics, "macro_ap")
    d_ap = number(d_metrics, "macro_ap")
    deltas = {
        "plus_minus_2_event_f1": d_plus2 - c_plus2,
        "exact_event_f1": d_exact - c_exact,
        "macro_ap": d_ap - c_ap,
    }
    gates = {
        "plus_minus_2_improves_at_least_0_010": (
            deltas["plus_minus_2_event_f1"] + 1e-12
            >= MINIMUM_PLUS_MINUS_2_DELTA
        ),
        "exact_loses_at_most_0_002": (
            deltas["exact_event_f1"] + 1e-12 >= -MAXIMUM_EXACT_LOSS
        ),
        "macro_ap_loses_at_most_0_005": (
            deltas["macro_ap"] + 1e-12 >= -MAXIMUM_MACRO_AP_LOSS
        ),
    }
    passed = all(gates.values())
    return {
        "rule": {
            "plus_minus_2_event_f1_minimum_delta": MINIMUM_PLUS_MINUS_2_DELTA,
            "exact_event_f1_minimum_delta": -MAXIMUM_EXACT_LOSS,
            "macro_ap_minimum_delta": -MAXIMUM_MACRO_AP_LOSS,
        },
        "d_minus_c": deltas,
        "gates": gates,
        "D_replication_recommended": passed,
        "next_step": (
            "request separate authorization for full C/D seeds 1 and 2"
            if passed
            else "do not replicate D under the frozen seed-zero rule"
        ),
    }


def _validate_release(
    *,
    arm: str,
    report_path: Path,
    sidecar_path: Path,
    marker_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from experiments.eval_dynamics_downstream import validate_release

    validate_release(
        report_path,
        sidecar_path,
        marker_path,
        expected_arm=arm,
    )
    report = _json_object(report_path, f"Arm {arm} downstream report")
    marker = _json_object(marker_path, f"Arm {arm} completion marker")
    expected = {
        "study_id": STUDY_ID,
        "arm": arm,
        "run_id": ARM_RUN_IDS[arm],
        "surface": SURFACE,
        "weights": "final",
    }
    for key, value in expected.items():
        if report.get(key) != value or marker.get(key) != value:
            raise ValueError(f"Arm {arm} release {key} differs")
    if marker.get("status") != "complete":
        raise ValueError(f"Arm {arm} release is not complete")
    if marker.get("report_sha256") != sha256_file(report_path):
        raise ValueError(f"Arm {arm} report hash differs")
    if marker.get("sidecar_sha256") != sha256_file(sidecar_path):
        raise ValueError(f"Arm {arm} sidecar hash differs")
    sidecar_receipt = report.get("prediction_sidecar")
    if not isinstance(sidecar_receipt, Mapping) or sidecar_receipt.get(
        "sha256"
    ) != sha256_file(sidecar_path):
        raise ValueError(f"Arm {arm} report sidecar receipt differs")
    loaded = load_later8_sidecar(sidecar_path, permit_superset=False)
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Arm {arm} report metrics are missing")
    support = report.get("support")
    if not isinstance(support, Mapping):
        raise ValueError(f"Arm {arm} support receipt is missing")
    for key, value in {
        "all_frames": Y4N_FRAMES,
        "input_active_frames": Y4N_FRAMES,
        "streams": len(Y4N_STREAM_IDS),
        "truth_sha256": Y4N_TRUTH_SHA256,
        "finite_aligned_arrays": True,
    }.items():
        if support.get(key) != value:
            raise ValueError(f"Arm {arm} support {key} differs")
    return report, dict(metrics), loaded


def _checkpoint_digest(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    parts = text.split()
    if len(parts) != 2 or parts[1] != "model.pt" or not SHA256_RE.fullmatch(parts[0]):
        raise ValueError("historical Arm A checkpoint record is malformed")
    return parts[0]


def _validate_contract(path: Path) -> dict[str, Any]:
    contract = _json_object(path, "exploratory C/D contract")
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("exploratory C/D contract schema differs")
    if contract.get("study_id") != STUDY_ID:
        raise ValueError("exploratory C/D contract study identity differs")
    if contract.get("status") != (
        "owner_authorized_post_phase0_exploratory_override_before_optimizer_step_one"
    ):
        raise ValueError("exploratory C/D authorization status differs")
    downstream = contract.get("downstream")
    promotion = contract.get("promotion")
    if not isinstance(downstream, Mapping) or not isinstance(promotion, Mapping):
        raise ValueError("exploratory C/D downstream or promotion contract is missing")
    expected_downstream = {
        "steps": 20_458,
        "seed": 0,
        "checkpoint": "final only",
        "evaluation": "mapped y4n later-eight on identical support",
        "threshold": 0.5,
    }
    for key, expected in expected_downstream.items():
        if downstream.get(key) != expected:
            raise ValueError(f"exploratory C/D downstream contract changed {key}")
    if promotion.get("D_vs_C_rule") != (
        "D plus-or-minus-2 event F1 improves by at least 0.010 absolute, "
        "exact event F1 loses at most 0.002, and macro AP loses at most 0.005"
    ):
        raise ValueError("exploratory C/D replication rule differs")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "study_id": STUDY_ID,
    }


def _historical_a_context(
    report_path: Path,
    sidecar_path: Path,
    checkpoint_record: Path,
) -> dict[str, Any]:
    from experiments.eval_dynamics_downstream import score_fixed_surface

    report = _json_object(report_path, "historical Arm A report")
    if report.get("weights") != "final" or report.get("label_kind") != (
        "mapped_foreign_nitrogen"
    ):
        raise ValueError("historical Arm A report is not final mapped-y4n evidence")
    if Path(str(report.get("run", ""))).name != REFERENCE_A_RUN_ID:
        raise ValueError("historical Arm A run identity differs")
    if report.get("sessions") != [
        f"y4nQHqYSObI__r{index:03d}" for index in range(16)
    ]:
        raise ValueError("historical Arm A report does not cover exact all-y4n")
    loaded = load_later8_sidecar(sidecar_path, permit_superset=True)
    metrics = score_fixed_surface(
        loaded["truth"],
        loaded["probability"],
        loaded["active"],
        loaded["lengths"],
    )
    return {
        "run_id": REFERENCE_A_RUN_ID,
        "checkpoint_sha256": _checkpoint_digest(checkpoint_record),
        "source_report": str(report_path),
        "source_report_sha256": sha256_file(report_path),
        "source_sidecar": str(sidecar_path),
        "source_sidecar_sha256": sha256_file(sidecar_path),
        "derived_surface": SURFACE,
        "support": loaded["support"],
        "metrics": metrics,
        "interpretation": (
            "Historical ImageNet-feature context only. Comparisons to C or D "
            "are not an isolated causal contrast because representation "
            "generation and execution chronology differ; D versus C is the "
            "matched contrast."
        ),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    for arm in ("c", "d"):
        value.add_argument(f"--{arm}-report", type=Path, required=True)
        value.add_argument(f"--{arm}-sidecar", type=Path, required=True)
        value.add_argument(f"--{arm}-marker", type=Path, required=True)
    value.add_argument("--reference-a-report", type=Path, required=True)
    value.add_argument("--reference-a-sidecar", type=Path, required=True)
    value.add_argument("--reference-a-checkpoint-record", type=Path, required=True)
    value.add_argument("--out", type=Path, required=True)
    value.add_argument("--completion-marker", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    contract = _validate_contract(args.contract)
    c_report, c_metrics, _ = _validate_release(
        arm="C",
        report_path=args.c_report,
        sidecar_path=args.c_sidecar,
        marker_path=args.c_marker,
    )
    d_report, d_metrics, _ = _validate_release(
        arm="D",
        report_path=args.d_report,
        sidecar_path=args.d_sidecar,
        marker_path=args.d_marker,
    )
    historical_a = _historical_a_context(
        args.reference_a_report,
        args.reference_a_sidecar,
        args.reference_a_checkpoint_record,
    )
    decision = promotion_decision(c_metrics, d_metrics)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "surface": SURFACE,
        "labels": "noisy mapped NitroGen development labels",
        "weights": "final",
        "threshold": 0.5,
        "threshold_or_calibration_parameters_fitted": False,
        "checkpoint_reselection": False,
        "B1_accessed": False,
        "sealed_untouched_session_accessed": False,
        "contract": contract,
        "matched_arms": {
            "C": {
                "run_id": ARM_RUN_IDS["C"],
                "report_sha256": sha256_file(args.c_report),
                "sidecar_sha256": sha256_file(args.c_sidecar),
                "marker_sha256": sha256_file(args.c_marker),
                "checkpoint_sha256": c_report["run_receipt"]["checkpoint_sha256"],
                "metrics": c_metrics,
            },
            "D": {
                "run_id": ARM_RUN_IDS["D"],
                "report_sha256": sha256_file(args.d_report),
                "sidecar_sha256": sha256_file(args.d_sidecar),
                "marker_sha256": sha256_file(args.d_marker),
                "checkpoint_sha256": d_report["run_receipt"]["checkpoint_sha256"],
                "metrics": d_metrics,
            },
        },
        "decision": decision,
        "historical_A_context": historical_a,
        "interpretation_boundary": (
            "This seed-zero exploratory result tests the bounded C/D recipe. "
            "It is not confirmatory, not a Photon-1 reproduction, and without "
            "Arm B cannot isolate future prediction from ordinary Celeste "
            "image-domain adaptation."
        ),
    }
    _atomic_json(args.out, payload)
    marker = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "status": "complete",
        "study_id": STUDY_ID,
        "surface": SURFACE,
        "decision_sha256": sha256_file(args.out),
        "contract_sha256": contract["sha256"],
        "C_report_sha256": sha256_file(args.c_report),
        "D_report_sha256": sha256_file(args.d_report),
        "C_checkpoint_sha256": c_report["run_receipt"]["checkpoint_sha256"],
        "D_checkpoint_sha256": d_report["run_receipt"]["checkpoint_sha256"],
        "D_replication_recommended": decision["D_replication_recommended"],
    }
    try:
        _atomic_json(args.completion_marker, marker)
    except BaseException:
        args.out.unlink(missing_ok=True)
        raise
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
