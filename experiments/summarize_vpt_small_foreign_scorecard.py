#!/usr/bin/env python3
"""Validate and summarize the separate VPT-small foreign scorecard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from badeline.metrics import per_key_ap, summarize
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


def parse_named_paths(values: Iterable[str], *, argument: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"{argument} values must use NAME=PATH")
        if name in result:
            raise ValueError(f"duplicate {argument} name: {name}")
        result[name] = Path(raw_path)
    return result


def load_sidecar(path: Path, *, require_source_support: bool) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"y_true", "y_prob", "input_active", "session_lengths", "session_ids"}
        if require_source_support:
            required.update(SUPPORT_FIELDS)
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    truth = arrays["y_true"]
    probability = arrays["y_prob"]
    active = arrays["input_active"]
    lengths = arrays["session_lengths"]
    if truth.shape != probability.shape or truth.ndim != 2:
        raise ValueError(f"{path}: truth and probability shapes differ")
    if truth.shape[1] != len(KEY_ORDER) or active.shape != (len(truth),):
        raise ValueError(f"{path}: invalid action or active-mask shape")
    if int(lengths.sum()) != len(truth):
        raise ValueError(f"{path}: session lengths do not cover rows")
    if not np.all(np.isfinite(probability)):
        raise ValueError(f"{path}: nonfinite probabilities")
    return arrays


def support_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in SUPPORT_FIELDS:
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def assert_same_support(
    authority: dict[str, np.ndarray], candidate: dict[str, np.ndarray], *, name: str
) -> None:
    for field in SUPPORT_FIELDS:
        if not np.array_equal(authority[field], candidate[field]):
            raise RuntimeError(f"{name}: VPT support differs at {field}")


def score(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    truth = arrays["y_true"].astype(np.uint8, copy=False)
    probability = arrays["y_prob"].astype(np.float64, copy=False)
    active = arrays["input_active"].astype(bool, copy=False)
    gated_truth = truth[active]
    gated_probability = probability[active]
    predicted = gated_probability >= 0.5
    tp = np.logical_and(predicted, gated_truth == 1).sum(axis=0)
    fp = np.logical_and(predicted, gated_truth == 0).sum(axis=0)
    fn = np.logical_and(~predicted, gated_truth == 1).sum(axis=0)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / np.maximum(1, tp + fn)
    state_f1 = 2 * precision * recall / np.maximum(1e-12, precision + recall)
    prevalence = gated_truth.mean(axis=0)
    ppr = predicted.mean(axis=0)
    ppr_ratio = ppr / np.maximum(1e-12, prevalence)
    aps = per_key_ap(gated_truth, gated_probability)
    detailed = summarize(
        truth,
        probability,
        boundaries=arrays["session_lengths"].tolist(),
        active=active,
        fixed_transition_thresholds={key: 0.5 for key in KEY_ORDER},
        include_oracle=False,
    )
    clipped = np.clip(gated_probability, 1e-7, 1.0 - 1e-7)
    natural_nll = float(
        np.mean(
            -(gated_truth * np.log(clipped) + (1 - gated_truth) * np.log(1 - clipped)),
            axis=0,
        ).sum()
    )
    per_key = {
        key: {
            "ap": float(aps[key]),
            "precision": float(precision[column]),
            "recall": float(recall[column]),
            "state_f1": float(state_f1[column]),
            "prevalence": float(prevalence[column]),
            "predicted_positive_rate": float(ppr[column]),
            "ppr_prevalence_ratio": float(ppr_ratio[column]),
            "event_f1_collar_0": float(
                detailed["transition_f1_at_0.5"][key]["event"]["f1"]
            ),
            "event_f1_collar_2_native_frames": float(
                detailed["transition_f1_at_0.5_collars"]["2"][key]["event"]["f1"]
            ),
        }
        for column, key in enumerate(KEY_ORDER)
    }
    return {
        "rows": int(len(truth)),
        "active_rows": int(active.sum()),
        "streams": int(len(arrays["session_lengths"])),
        "natural_nll": natural_nll,
        "macro_ap": float(np.nanmean(list(aps.values()))),
        "macro_state_f1": float(np.nanmean(state_f1)),
        "all_keys_nonzero_recall_at_0_5": bool(np.all(recall > 0)),
        "all_keys_ppr_prevalence_ratio_in_0_5_to_2_0": bool(
            np.all((ppr_ratio >= 0.5) & (ppr_ratio <= 2.0))
        ),
        "per_key": per_key,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# VPT-small foreign gameplay scorecard",
        "",
        "This is a mapped-foreign **development** scorecard. It does not replace or blend with the native-keyboard eligibility gates. The y4n vertical-axis sign is indeterminate, so up/down results are label-noise-sensitive.",
        "",
        "| Family | Endpoint | Rows | Macro AP | Macro state F1 | Down AP | Down recall | Down PPR / prevalence | All-key nonzero recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for name, model in result["models"].items():
        metrics = model["metrics"]
        down = metrics["per_key"]["down"]
        lines.append(
            "| {family} | `{name}` | {rows:,} | {macro_ap:.4f} | {macro_f1:.4f} | "
            "{down_ap:.4f} | {down_recall:.4f} | {down_ratio:.3f} | {all_key} |".format(
                family=model["family"],
                name=name,
                rows=metrics["rows"],
                macro_ap=metrics["macro_ap"],
                macro_f1=metrics["macro_state_f1"],
                down_ap=down["ap"],
                down_recall=down["recall"],
                down_ratio=down["ppr_prevalence_ratio"],
                all_key="yes" if metrics["all_keys_nonzero_recall_at_0_5"] else "no",
            )
        )
    lines.extend([
        "",
        "Historical GRU rows are shown as contextual evidence on their native full-y4n support. Only the VPT-small rows are guaranteed identical across endpoints.",
        "",
    ])
    return "\n".join(lines)


def build_scorecard(
    contract_path: Path,
    reports: dict[str, Path],
    vpt_sidecars: dict[str, Path],
    gru_sidecars: dict[str, Path],
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    endpoints = {item["name"]: item for item in contract["vpt_endpoints"]}
    if set(reports) != set(endpoints) or set(vpt_sidecars) != set(endpoints):
        raise ValueError("VPT report and sidecar names must exactly match frozen endpoints")
    models: dict[str, Any] = {}
    authority_arrays: dict[str, np.ndarray] | None = None
    authority_hash: str | None = None
    manifest_hash: str | None = None
    for name in endpoints:
        report = json.loads(reports[name].read_text(encoding="utf-8"))
        endpoint = endpoints[name]
        if report.get("threshold") != 0.5:
            raise RuntimeError(f"{name}: report threshold is not 0.5")
        if report["weights"]["sha256"] != endpoint["sha256"]:
            raise RuntimeError(f"{name}: checkpoint hash differs from contract")
        candidate_manifest_hash = report["data"]["manifest_sha256"]
        if manifest_hash is None:
            manifest_hash = candidate_manifest_hash
        elif manifest_hash != candidate_manifest_hash:
            raise RuntimeError(f"{name}: evaluation manifest differs")
        arrays = load_sidecar(vpt_sidecars[name], require_source_support=True)
        if authority_arrays is None:
            authority_arrays = arrays
            authority_hash = support_sha256(arrays)
        else:
            assert_same_support(authority_arrays, arrays, name=name)
        models[name] = {
            "family": "VPT-small",
            "support": "identical VPT three-phase center support",
            "checkpoint_sha256": endpoint["sha256"],
            "report": str(reports[name]),
            "report_sha256": sha256_file(reports[name]),
            "sidecar": str(vpt_sidecars[name]),
            "sidecar_sha256": sha256_file(vpt_sidecars[name]),
            "metrics": score(arrays),
        }
    for name, path in gru_sidecars.items():
        arrays = load_sidecar(path, require_source_support=False)
        models[name] = {
            "family": "pixel GRU",
            "support": "historical GRU-native full-y4n support",
            "sidecar": str(path),
            "sidecar_sha256": sha256_file(path),
            "metrics": score(arrays),
        }
    return {
        "schema_version": "madeleine.vpt-small-foreign-scorecard.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "surface": contract["surface"]["name"],
        "role": contract["surface"]["role"],
        "threshold": 0.5,
        "vpt_manifest_sha256": manifest_hash,
        "vpt_support_sha256": authority_hash,
        "models": models,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--vpt-report", action="append", default=[])
    parser.add_argument("--vpt-sidecar", action="append", default=[])
    parser.add_argument("--gru-sidecar", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_scorecard(
        args.contract,
        parse_named_paths(args.vpt_report, argument="--vpt-report"),
        parse_named_paths(args.vpt_sidecar, argument="--vpt-sidecar"),
        parse_named_paths(args.gru_sidecar, argument="--gru-sidecar"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_out.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"models": list(result["models"]), "support": result["vpt_support_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
