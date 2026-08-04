#!/usr/bin/env python3
"""Assign a calibration role to independently sealed capture components."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

from data.schema import KEY_ORDER
from experiments.eval_vpt_small import sha256_file
from experiments.prepare_vpt_small_calibration_capture import (
    MAX_DROP_RATE,
    MAX_MARGIN_AUC,
    MIN_ACTIVE_MINUTES,
    MIN_RUNS_PER_KEY,
)


SCHEMA_VERSION = "madeleine.vpt-small-calibration-capture-bundle-receipt.v1"
_SUPPORT_VIOLATION = re.compile(
    r"^(?:common-support active minutes|(?:"
    + "|".join(re.escape(key) for key in KEY_ORDER)
    + r") positive state runs)"
)


def _source_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def seal_bundle(
    *,
    role: str,
    component_receipts: list[Path],
    out: Path,
    repo: Path,
) -> dict[str, Any]:
    if len(component_receipts) != 2:
        raise ValueError("the frozen bundle requires exactly two components")

    violations: list[str] = []
    components: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()
    active_minutes = 0.0
    rows = 0
    active_rows = 0
    segments = 0
    run_counts = {key: 0 for key in KEY_ORDER}

    for path in component_receipts:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("role") != role:
            violations.append(f"{path}: component role differs from {role}")
        if receipt.get("model_accessed") is not False:
            violations.append(f"{path}: component was exposed to a model")
        session_id = receipt.get("session", {}).get("session_id")
        if not isinstance(session_id, str) or not session_id:
            violations.append(f"{path}: component lacks session identity")
            session_id = str(session_id)
        if session_id in seen_sessions:
            violations.append(f"{path}: duplicate component session {session_id}")
        seen_sessions.add(session_id)

        component_violations = [str(value) for value in receipt.get("violations", [])]
        integrity_violations = [
            value for value in component_violations if not _SUPPORT_VIOLATION.match(value)
        ]
        if integrity_violations:
            violations.extend(f"{session_id}: {value}" for value in integrity_violations)

        integrity = receipt.get("capture_integrity", {})
        if integrity.get("validator_violations"):
            violations.append(f"{session_id}: session validator failed")
        if float(integrity.get("drop_rate", float("inf"))) > MAX_DROP_RATE:
            violations.append(f"{session_id}: drop-rate gate failed")
        leak = receipt.get("leak_probe", {})
        if float(leak.get("max_symmetric_margin_auc", float("inf"))) >= MAX_MARGIN_AUC:
            violations.append(f"{session_id}: adjacent-band leak gate failed")

        support = receipt.get("support", {})
        active_minutes += float(support.get("active_minutes", 0.0))
        rows += int(support.get("rows", 0))
        active_rows += int(support.get("active_rows", 0))
        segments += int(support.get("segments", 0))
        component_runs = support.get("positive_state_runs", {})
        for key in KEY_ORDER:
            run_counts[key] += int(component_runs.get(key, 0))

        components.append(
            {
                "receipt": str(path),
                "receipt_sha256": sha256_file(path),
                "session_id": session_id,
                "standalone_decision": receipt.get("decision"),
                "standalone_support_violations": [
                    value for value in component_violations if _SUPPORT_VIOLATION.match(value)
                ],
                "support_sha256": support.get("support_sha256"),
            }
        )

    if active_minutes < MIN_ACTIVE_MINUTES:
        violations.append(
            f"bundle active minutes {active_minutes:.6f} below {MIN_ACTIVE_MINUTES:.6f}"
        )
    for key, count in run_counts.items():
        if count < MIN_RUNS_PER_KEY:
            violations.append(f"bundle {key} runs {count} below {MIN_RUNS_PER_KEY}")

    support_identity = {
        "component_support_sha256": [value["support_sha256"] for value in components],
        "boundary_policy": "hard component boundary; no inference window may bridge",
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": _source_commit(repo),
        "role": role,
        "decision": "accepted" if not violations else "rejected",
        "violations": violations,
        "components": components,
        "support": {
            "rows": rows,
            "active_rows": active_rows,
            "active_minutes": active_minutes,
            "positive_state_runs": run_counts,
            "segments": segments,
            "component_boundaries": len(components) - 1,
            "support_sha256": _canonical_sha256(support_identity),
        },
        "window_boundary_policy": support_identity["boundary_policy"],
        "model_accessed": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if violations:
        raise RuntimeError(f"capture bundle rejected: {violations}")
    return receipt


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("c1", "e1", "e2"), required=True)
    parser.add_argument("--component-receipt", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("."))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = seal_bundle(
        role=args.role,
        component_receipts=args.component_receipt,
        out=args.out,
        repo=args.repo,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
