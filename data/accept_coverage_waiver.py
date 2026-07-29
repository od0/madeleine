"""Record a human ruling that a session's mask-coverage failure may be waived.

``data.mask_coverage`` fails a session when the union of key-correlated
pixels in the band just outside a declared key-driven rect exceeds the gate
floor.  The statistic cannot distinguish widget bleed from gameplay content
whose brightness correlates with key state; when an owner-reviewed evidence
packet establishes that a specific session's failure is content correlation
and not overlay bleed, that ruling must itself be a recorded artifact, not a
policy edit or a bypass flag.

This module creates that artifact.  It binds the exact session manifest
bytes, the exact evidence-packet decision bytes, the measured per-key band
fractions being waived (re-measured at creation, or cross-checked against a
supplied ``--measured`` report), and the enforced gate floor, together with
a human reviewer identity and a required free-text rationale.
``data.build_dataset`` verifies the artifact and then suppresses only the
coverage rejection; the build manifest records that the gate was
overridden, never that it passed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from data.mask_coverage import BAND_FRAC_MAX, measure_mask_coverage


WAIVER_VERSION = "madeleine.mask-coverage-waiver.v1"
WAIVED_GATE = "mask coverage band fraction above the gate floor"
REVIEWER_KINDS = ("human", "human_with_ai_assistance")


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _sha256(value: Any, field: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return result


def _local_name(value: Any, field: str) -> str:
    result = str(value).strip()
    if not result or result in (".", "..") or Path(result).name != result:
        raise ValueError(f"{field} must be one local file name")
    return result


def _same_float(actual: float, expected: float, field: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{field} is inconsistent with the bound evidence")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_new(path: Path, payload: bytes) -> None:
    """Atomically create ``path`` and fail rather than replace an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _failing_regions(report: dict[str, Any], band_frac_max: float) -> list[dict[str, Any]]:
    return [
        region
        for region in report["regions"]
        if region["band_any_key_fraction"] > band_frac_max
    ]


def _check_measured(measured: dict[str, Any], report: dict[str, Any]) -> None:
    """Require a supplied measurement to match a fresh one exactly.

    ``--measured`` never replaces measurement; it asserts that the numbers
    the reviewer waived are the numbers the verifier reproduces.
    """

    measured = _mapping(measured, "measured coverage report")
    if measured.get("session_id") != report["session_id"]:
        raise ValueError("measured coverage report names a different session")
    rows = measured.get("regions")
    if not isinstance(rows, list):
        raise ValueError("measured coverage report.regions must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        row = _mapping(row, "measured coverage report.regions[]")
        name = str(row.get("name", "")).strip()
        if not name or name in by_name:
            raise ValueError("measured coverage report regions must be uniquely named")
        by_name[name] = row
    fresh = {region["name"]: region for region in report["regions"]}
    if set(by_name) != set(fresh):
        raise ValueError(
            "measured coverage report regions differ from a fresh "
            "data.mask_coverage measurement"
        )
    for name, fresh_region in fresh.items():
        row = by_name[name]
        _same_float(
            _number(
                row.get("band_any_key_fraction"),
                f"measured {name} band_any_key_fraction",
            ),
            fresh_region["band_any_key_fraction"],
            f"measured {name} band_any_key_fraction",
        )
        per_key = _mapping(
            row.get("band_key_correlated_fraction"),
            f"measured {name} band_key_correlated_fraction",
        )
        if set(per_key) != set(fresh_region["band_key_correlated_fraction"]):
            raise ValueError(
                f"measured {name} per-key fractions differ from a fresh measurement"
            )
        for key, fresh_value in fresh_region["band_key_correlated_fraction"].items():
            _same_float(
                _number(per_key[key], f"measured {name} fraction for {key!r}"),
                fresh_value,
                f"measured {name} fraction for {key!r}",
            )


def _region_record(region: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": region["name"],
        "band_any_key_fraction": region["band_any_key_fraction"],
        "band_key_correlated_fraction": {
            key: region["band_key_correlated_fraction"][key]
            for key in sorted(region["band_key_correlated_fraction"])
        },
    }


def accept_coverage_waiver(
    session_dir: str | Path,
    evidence_path: str | Path,
    waiver_path: str | Path,
    *,
    reviewer_identity: str,
    reviewer_kind: str,
    rationale: str,
    approved: bool,
    band_frac_max: float = BAND_FRAC_MAX,
    measured: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an immutable, hash-bound mask-coverage waiver ruling."""

    session_root = Path(session_dir)
    evidence_file = Path(evidence_path)
    waiver_file = Path(waiver_path)
    identity = reviewer_identity.strip()
    if not identity:
        raise ValueError("reviewer_identity is required")
    if reviewer_kind not in REVIEWER_KINDS:
        raise ValueError(f"reviewer_kind must be one of {REVIEWER_KINDS}")
    reason = rationale.strip()
    if not reason:
        raise ValueError("a non-empty free-text rationale is required")
    if approved is not True:
        raise ValueError("an explicit waiver approval is required")
    if waiver_file.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {waiver_file}")

    floor = _number(band_frac_max, "band_frac_max")
    if not 0.0 < floor < 1.0:
        raise ValueError("band_frac_max must lie in (0, 1)")

    manifest_file = session_root / "manifest.json"
    if not manifest_file.is_file():
        raise ValueError(f"no session manifest at {manifest_file}")
    manifest = _mapping(
        json.loads(manifest_file.read_text(encoding="utf-8")), "session manifest"
    )
    session_id = str(manifest.get("session_id", "")).strip()
    if not session_id or session_id != session_root.name:
        raise ValueError(
            "session manifest session_id does not match the session directory"
        )
    if not evidence_file.is_file():
        raise ValueError(f"no evidence file at {evidence_file}")

    # Always re-measure; a supplied report can only confirm, never substitute.
    report = measure_mask_coverage(session_root)
    if measured is not None:
        _check_measured(measured, report)
    failing = _failing_regions(report, floor)
    if not failing:
        raise ValueError(
            f"session {session_id} passes the mask-coverage gate at floor "
            f"{floor:.4f}; there is nothing to waive"
        )

    waiver = {
        "format_version": WAIVER_VERSION,
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": {
            "approved": True,
            "reviewer_identity": identity,
            "reviewer_kind": reviewer_kind,
            "rationale": reason,
        },
        "session_manifest": {
            "path": manifest_file.name,
            "sha256": sha256_file(manifest_file),
        },
        "evidence": {
            "path": evidence_file.name,
            "sha256": sha256_file(evidence_file),
        },
        "waived_gate": {
            "gate": WAIVED_GATE,
            "band_frac_max": floor,
            "regions": [_region_record(region) for region in failing],
        },
    }
    payload = (json.dumps(waiver, indent=2) + "\n").encode("utf-8")
    _atomic_write_new(waiver_file, payload)
    return waiver


def verify_coverage_waiver(
    session_dir: str | Path,
    waiver_path: str | Path,
    *,
    report: dict[str, Any],
    band_frac_max: float = BAND_FRAC_MAX,
) -> dict[str, Any]:
    """Verify a waiver against the session, measurement, and floor in use.

    ``report`` is the result of :func:`data.mask_coverage.measure_mask_coverage`
    for the same session being built.  Any mismatch is a hard error, never a
    silent fallthrough.
    """

    session_root = Path(session_dir)
    waiver_file = Path(waiver_path)
    waiver = _mapping(
        json.loads(waiver_file.read_text(encoding="utf-8")), "coverage waiver"
    )
    if waiver.get("format_version") != WAIVER_VERSION:
        raise ValueError("unsupported coverage waiver format_version")
    session_id = session_root.name
    if waiver.get("session_id") != session_id or report["session_id"] != session_id:
        raise ValueError("coverage waiver session_id differs from the session being built")

    decision = _mapping(waiver.get("decision"), "coverage waiver.decision")
    if decision.get("approved") is not True:
        raise ValueError("coverage waiver lacks explicit approval")
    identity = str(decision.get("reviewer_identity", "")).strip()
    kind = str(decision.get("reviewer_kind", "")).strip()
    if not identity or kind not in REVIEWER_KINDS:
        raise ValueError(
            "coverage waiver reviewer identity/kind is not a human provenance"
        )
    reason = str(decision.get("rationale", "")).strip()
    if not reason:
        raise ValueError("coverage waiver lacks a non-empty rationale")

    manifest_row = _mapping(
        waiver.get("session_manifest"), "coverage waiver.session_manifest"
    )
    if _local_name(
        manifest_row.get("path"), "coverage waiver.session_manifest.path"
    ) != "manifest.json":
        raise ValueError("coverage waiver names a different session manifest")
    if _sha256(
        manifest_row.get("sha256"), "coverage waiver.session_manifest.sha256"
    ) != sha256_file(session_root / "manifest.json"):
        raise ValueError("coverage waiver is bound to different session manifest bytes")

    evidence_row = _mapping(waiver.get("evidence"), "coverage waiver.evidence")
    _local_name(evidence_row.get("path"), "coverage waiver.evidence.path")
    evidence_sha256 = _sha256(
        evidence_row.get("sha256"), "coverage waiver.evidence.sha256"
    )

    gate_row = _mapping(waiver.get("waived_gate"), "coverage waiver.waived_gate")
    if gate_row.get("gate") != WAIVED_GATE:
        raise ValueError("coverage waiver names a different admission gate")
    recorded_floor = _number(
        gate_row.get("band_frac_max"), "coverage waiver.waived_gate.band_frac_max"
    )
    _same_float(
        recorded_floor,
        _number(band_frac_max, "band_frac_max"),
        "coverage waiver gate floor",
    )

    rows = gate_row.get("regions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("coverage waiver records no waived band fractions")
    recorded: dict[str, dict[str, Any]] = {}
    for row in rows:
        row = _mapping(row, "coverage waiver.waived_gate.regions[]")
        name = str(row.get("name", "")).strip()
        if not name or name in recorded:
            raise ValueError("coverage waiver regions must be uniquely named")
        recorded[name] = row
    fresh_failing = {
        region["name"]: region for region in _failing_regions(report, recorded_floor)
    }
    if not fresh_failing:
        raise ValueError(
            "coverage waiver applies to a session that passes the gate; "
            "there is nothing to waive"
        )
    if set(recorded) != set(fresh_failing):
        raise ValueError(
            "coverage waiver band fractions do not name the measured coverage failures"
        )

    band_fractions: dict[str, dict[str, Any]] = {}
    for name, row in recorded.items():
        fresh_region = fresh_failing[name]
        union = _number(
            row.get("band_any_key_fraction"),
            f"coverage waiver {name} band_any_key_fraction",
        )
        _same_float(
            union,
            fresh_region["band_any_key_fraction"],
            f"coverage waiver {name} band_any_key_fraction",
        )
        if not union > recorded_floor:
            raise ValueError(
                "coverage waiver records a band fraction that is not above its floor"
            )
        per_key = _mapping(
            row.get("band_key_correlated_fraction"),
            f"coverage waiver {name} band_key_correlated_fraction",
        )
        if set(per_key) != set(fresh_region["band_key_correlated_fraction"]):
            raise ValueError(
                f"coverage waiver {name} per-key fractions differ from the measurement"
            )
        for key, fresh_value in fresh_region["band_key_correlated_fraction"].items():
            _same_float(
                _number(per_key[key], f"coverage waiver {name} fraction for {key!r}"),
                fresh_value,
                f"coverage waiver {name} fraction for {key!r}",
            )
        band_fractions[name] = {
            "band_any_key_fraction": union,
            "band_key_correlated_fraction": {
                key: per_key[key] for key in sorted(per_key)
            },
        }

    return {
        "path": str(waiver_file),
        "sha256": sha256_file(waiver_file),
        "format_version": WAIVER_VERSION,
        "session_id": session_id,
        "reviewer_identity": identity,
        "reviewer_kind": kind,
        "human_reviewed": True,
        "rationale": reason,
        "waived_gate": WAIVED_GATE,
        "band_frac_max": recorded_floor,
        "band_fractions": band_fractions,
        "session_manifest_sha256": manifest_row["sha256"],
        "evidence_sha256": evidence_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True,
                        help="frozen-format session directory")
    parser.add_argument("--evidence", type=Path, required=True,
                        help="the owner evidence packet decision file being bound")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewer-kind", choices=REVIEWER_KINDS, required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--approve", action="store_true", required=True)
    parser.add_argument(
        "--measured", type=Path, default=None,
        help="optional data.mask_coverage --report JSON; cross-checked against "
             "a fresh measurement, never a substitute for one",
    )
    parser.add_argument(
        "--floor", type=float, default=BAND_FRAC_MAX,
        help="the enforced coverage gate floor being waived",
    )
    args = parser.parse_args()
    measured = (
        json.loads(args.measured.read_text(encoding="utf-8"))
        if args.measured is not None
        else None
    )
    result = accept_coverage_waiver(
        args.session,
        args.evidence,
        args.out,
        reviewer_identity=args.reviewer,
        reviewer_kind=args.reviewer_kind,
        rationale=args.rationale,
        approved=args.approve,
        band_frac_max=args.floor,
        measured=measured,
    )
    print(json.dumps({
        "session_id": result["session_id"],
        "band_frac_max": result["waived_gate"]["band_frac_max"],
        "regions": [
            {
                "name": region["name"],
                "band_any_key_fraction": region["band_any_key_fraction"],
            }
            for region in result["waived_gate"]["regions"]
        ],
        "reviewer_identity": result["decision"]["reviewer_identity"],
        "reviewer_kind": result["decision"]["reviewer_kind"],
        "waiver": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
