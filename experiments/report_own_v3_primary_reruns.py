"""Validate and summarize the six corrected own-v3 primary reruns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from data.schema import KEY_ORDER


# Bucket identity is operational and stays out of Git (the contributor guide
# requires placeholders or environment variables for host and account
# identifiers). Rerun validation against real publication records sets
# MADELEINE_R2_BUCKET_URI; the placeholder default fails closed on the
# object-prefix comparison rather than leaking the bucket into Git.
R2_BUCKET_URI = os.environ.get("MADELEINE_R2_BUCKET_URI", "r2:<bucket>")

FAMILIES = {
    "scratch": {
        "old_prefix": "own_features_32nc_s",
        "new_prefix": "own_features_v3_32nc_s",
    },
    "tier_b_init": {
        "old_prefix": "foreign_tier_b_13p45h_32nc_finetune_s",
        "new_prefix": "own_features_v3_tier_b_init_32nc_s",
    },
}
POPULATIONS = ("all_frames", "input_active_only")
SUPPORT = {"all_frames": 29_086, "input_active_only": 25_028}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mean(values: Mapping[str, Any]) -> float:
    return float(np.mean([float(values[key]) for key in KEY_ORDER]))


def _event_mean(
    metrics: Mapping[str, Any], field: str, collar: int | None = None
) -> float:
    rows = metrics[field] if collar is None else metrics[field][str(collar)]
    return float(np.mean([float(rows[key]["event"]["f1"]) for key in KEY_ORDER]))


def summarize_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "macro_ap": _mean(metrics["per_key_ap"]),
        "per_key_ap": {key: float(metrics["per_key_ap"][key]) for key in KEY_ORDER},
        "macro_state_f1_at_0.5": _mean(metrics["per_key_f1"]),
        "macro_event_f1_at_0.5_collar0": _event_mean(
            metrics, "transition_f1_at_0.5"
        ),
        "macro_event_f1_oracle_collar0": _event_mean(
            metrics, "transition_f1_oracle"
        ),
        "macro_event_f1_oracle_collar2": _event_mean(
            metrics, "transition_f1_oracle_collars", 2
        ),
    }


def _subtract(new: Mapping[str, Any], old: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in new.items():
        if isinstance(value, Mapping):
            result[key] = {
                nested: float(value[nested]) - float(old[key][nested])
                for nested in value
            }
        else:
            result[key] = float(value) - float(old[key])
    return result


def _average(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in rows[0]:
        if isinstance(rows[0][key], Mapping):
            result[key] = {
                nested: float(np.mean([float(row[key][nested]) for row in rows]))
                for nested in rows[0][key]
            }
        else:
            result[key] = float(np.mean([float(row[key]) for row in rows]))
    return result


def _validate_sidecar(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            "y_true", "y_prob", "input_active", "session_lengths", "session_ids"
        }
        if set(archive.files) != expected:
            raise ValueError(f"prediction fields changed: {path}")
        truth = np.asarray(archive["y_true"])
        probability = np.asarray(archive["y_prob"])
        active = np.asarray(archive["input_active"])
        lengths = np.asarray(archive["session_lengths"])
        ids = np.asarray(archive["session_ids"])
    if truth.shape != (SUPPORT["all_frames"], len(KEY_ORDER)):
        raise ValueError(f"prediction truth shape changed: {path}")
    if probability.shape != truth.shape or not np.all(np.isfinite(probability)):
        raise ValueError(f"prediction probabilities are unaligned/non-finite: {path}")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError(f"prediction probabilities are out of range: {path}")
    if active.shape != (len(truth),) or int(active.sum()) != SUPPORT["input_active_only"]:
        raise ValueError(f"prediction active support changed: {path}")
    if int(lengths.sum()) != len(truth) or len(lengths) != len(ids):
        raise ValueError(f"prediction stream boundaries are unaligned: {path}")
    return {
        "rows": len(truth),
        "active_rows": int(active.sum()),
        "streams": len(lengths),
        "sha256": sha256_file(path),
    }


def build_report(results: Path, contract_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_hash = sha256_file(contract_path)
    contract = _json(contract_path)
    families: dict[str, Any] = {}
    registry_records: list[dict[str, Any]] = []
    for family, names in FAMILIES.items():
        seed_rows: list[dict[str, Any]] = []
        for seed in range(3):
            old_id = f"{names['old_prefix']}{seed}"
            new_id = f"{names['new_prefix']}{seed}"
            run = results / new_id
            registration = _json(run / "checkpoint-registration.json")
            manifest_path = run / "checkpoint-manifest.json"
            manifest = _json(manifest_path)
            completion_path = run / "checkpoint_complete.json"
            completion = _json(completion_path)
            publication = _json(run / "r2_publication.json")
            val_completion = _json(run / "val_a_complete.json")
            meta = _json(run / "run_meta.json")
            checkpoint_hash = registration["checkpoint"]["sha256"]
            identities = {
                registration["run_id"], val_completion["run_id"], new_id
            }
            if identities != {new_id}:
                raise ValueError(f"run identity mismatch: {new_id}")
            if meta.get("own_v3_provenance", {}).get("contract_sha256") != contract_hash:
                raise ValueError(f"run metadata contract mismatch: {new_id}")
            if val_completion.get("contract_sha256") != contract_hash:
                raise ValueError(f"val completion contract mismatch: {new_id}")
            if not all(
                value == checkpoint_hash
                for value in (
                    manifest["checkpoint"]["sha256"],
                    completion["checkpoint_sha256"],
                    publication["checkpoint_sha256"],
                    val_completion["artifacts"]["model.pt"]["sha256"],
                )
            ):
                raise ValueError(f"checkpoint hash disagreement: {new_id}")
            if completion["manifest_sha256"] != sha256_file(manifest_path):
                raise ValueError(f"checkpoint manifest hash mismatch: {new_id}")
            if publication["manifest_sha256"] != completion["manifest_sha256"]:
                raise ValueError(f"R2 manifest hash mismatch: {new_id}")
            expected_prefix = (
                f"{R2_BUCKET_URI}/runs/idm/v1/{manifest['artifact_id']}/{checkpoint_hash}"
            )
            if publication["object_prefix"] != expected_prefix:
                raise ValueError(f"R2 object prefix mismatch: {new_id}")

            new_report_path = results / f"{new_id}_val_a.json"
            old_report_path = results / f"{old_id}_val_a.json"
            new_report = _json(new_report_path)
            old_report = _json(old_report_path)
            if new_report.get("sessions") != contract["data_contract"]["validation_sessions"]:
                raise ValueError(f"new val-A membership mismatch: {new_id}")
            if old_report.get("sessions") != contract["data_contract"]["validation_sessions"]:
                raise ValueError(f"old val-A membership mismatch: {old_id}")
            if new_report.get("weights") != "selected":
                raise ValueError(f"new report is not selected weights: {new_id}")
            metrics: dict[str, Any] = {}
            for population in POPULATIONS:
                if new_report[population]["n"] != SUPPORT[population]:
                    raise ValueError(f"new support mismatch: {new_id}/{population}")
                if old_report[population]["n"] != SUPPORT[population]:
                    raise ValueError(f"old support mismatch: {old_id}/{population}")
                old_metrics = summarize_metrics(old_report[population]["metrics"])
                new_metrics = summarize_metrics(new_report[population]["metrics"])
                metrics[population] = {
                    "mask_era": old_metrics,
                    "own_v3": new_metrics,
                    "delta": _subtract(new_metrics, old_metrics),
                }
            sidecar = _validate_sidecar(results / f"{new_id}_val_a_preds.npz")
            expected_sidecar_hash = val_completion["artifacts"][f"{new_id}_val_a_preds.npz"]["sha256"]
            if sidecar["sha256"] != expected_sidecar_hash:
                raise ValueError(f"sidecar hash mismatch: {new_id}")
            seed_rows.append({
                "seed": seed,
                "old_run_id": old_id,
                "new_run_id": new_id,
                "checkpoint": registration["checkpoint"],
                "checkpoint_sha256": checkpoint_hash,
                "sidecar": sidecar,
                "metrics": metrics,
            })
            registry_records.append({
                "artifact_id": manifest["artifact_id"],
                "checkpoint_bytes": manifest["checkpoint"]["bytes"],
                "checkpoint_sha256": checkpoint_hash,
                "completion_sha256": sha256_file(completion_path),
                "manifest_sha256": sha256_file(manifest_path),
                "object_prefix": publication["object_prefix"],
                "role": manifest["role"],
                "run_id": new_id,
            })
        means: dict[str, Any] = {}
        for population in POPULATIONS:
            old_rows = [row["metrics"][population]["mask_era"] for row in seed_rows]
            new_rows = [row["metrics"][population]["own_v3"] for row in seed_rows]
            old_mean = _average(old_rows)
            new_mean = _average(new_rows)
            means[population] = {
                "mask_era_mean": old_mean,
                "own_v3_mean": new_mean,
                "paired_mean_delta": _subtract(new_mean, old_mean),
            }
        families[family] = {"seeds": seed_rows, "three_seed_means": means}

    report = {
        "schema_version": "madeleine.own-v3-primary-rerun-delta.v1",
        "contract_sha256": contract_hash,
        "surface": "own-v3 val-A",
        "support": SUPPORT,
        "weights": "selected checkpoints",
        "threshold_provenance": {
            "state_f1": "fixed probability threshold 0.5",
            "fixed_event_collar0": "fixed probability threshold 0.5",
            "oracle_event_collar0_and_collar2": (
                "per-key thresholds fit separately within each report population "
                "on this same val-A surface; diagnostic only"
            ),
        },
        "families": families,
    }
    registry_records.sort(key=lambda row: row["artifact_id"])
    registry = {
        "format_version": "madeleine.idm-checkpoint-registry.v1",
        "registry_id": "own-v3-primary-reruns-20260728-v1",
        "checkpoint_count": len(registry_records),
        "checkpoint_bytes": sum(row["checkpoint_bytes"] for row in registry_records),
        "contract_sha256": contract_hash,
        "records": registry_records,
        "scope": "all six corrected own-v3 primary rerun checkpoints",
    }
    return report, registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--registry-out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report, registry = build_report(args.results, args.contract)
    _write_json(args.out, report)
    _write_json(args.registry_out, registry)
    print(json.dumps({
        "report_sha256": sha256_file(args.out),
        "registry_sha256": sha256_file(args.registry_out),
        "checkpoint_count": registry["checkpoint_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
