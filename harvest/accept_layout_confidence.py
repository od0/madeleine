"""Record a human ruling that a layout's inference confidence may stand.

Decode QC rejects any layout whose recorded ``inference_confidence`` sits
below the admission floor, even when a human has accepted the layout through
the hash-bound review packet.  When a reviewer consciously decides that the
recorded confidence should stand — for example because the mapping is
fail-closed and minimal and the packet evidence supports every bound cell —
that ruling must itself be a recorded artifact, not a policy edit.

This module creates that artifact.  It binds the exact layout bytes in use,
the exact human layout-acceptance bytes, the recorded confidence being
overridden, and the enforced floor, together with a human reviewer identity
and a required free-text rationale.  ``decode_wild`` verifies the artifact
and then suppresses only the confidence rejection; the decode report records
that the gate was overridden, never that it passed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from harvest.accept_wild_layout import (
    LAYOUT_ACCEPTANCE_VERSION,
    verify_layout_acceptance,
)
from harvest.fetch_wild import sha256_file
from harvest.wild_layout import WildLayout


OVERRIDE_VERSION = "madeleine.wild-layout-confidence-override.v1"
OVERRIDDEN_GATE = "layout inference confidence below admission threshold"
REVIEWER_KINDS = ("human", "human_with_ai_assistance")
# Mirrors QCPolicy.min_layout_confidence in harvest.decode_wild; the decode
# verifier rechecks equality against the enforced policy floor, so drift
# between the two constants fails closed instead of silently widening a gate.
DEFAULT_MIN_LAYOUT_CONFIDENCE = 0.80


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


def accept_confidence_override(
    layout_path: str | Path,
    layout_acceptance_path: str | Path,
    override_path: str | Path,
    *,
    reviewer_identity: str,
    reviewer_kind: str,
    rationale: str,
    approved: bool,
    min_layout_confidence: float = DEFAULT_MIN_LAYOUT_CONFIDENCE,
) -> dict[str, Any]:
    """Create an immutable, hash-bound confidence-override ruling."""

    layout_file = Path(layout_path)
    layout_acceptance_file = Path(layout_acceptance_path)
    override_file = Path(override_path)
    identity = reviewer_identity.strip()
    if not identity:
        raise ValueError("reviewer_identity is required")
    if reviewer_kind not in REVIEWER_KINDS:
        raise ValueError(f"reviewer_kind must be one of {REVIEWER_KINDS}")
    reason = rationale.strip()
    if not reason:
        raise ValueError("a non-empty free-text rationale is required")
    if approved is not True:
        raise ValueError("an explicit override approval is required")
    if override_file.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {override_file}")

    layout = WildLayout.load(layout_file)
    floor = _number(min_layout_confidence, "min_layout_confidence")
    if not 0.0 < floor <= 1.0:
        raise ValueError("min_layout_confidence must lie in (0, 1]")
    if not layout.inference_confidence < floor:
        raise ValueError(
            f"layout inference confidence {layout.inference_confidence:.4f} is not "
            f"below the admission floor {floor:.4f}; there is nothing to override"
        )
    layout_hash = sha256_file(layout_file)

    # Fully re-verify the human layout acceptance and its review packet before
    # binding it; the acceptance's own recorded source hash names the video,
    # and decode-time verification rechecks it against the fetch report.
    acceptance_raw = _mapping(
        json.loads(layout_acceptance_file.read_text()), "layout acceptance"
    )
    layout_review = verify_layout_acceptance(
        layout_file,
        layout,
        layout_acceptance_file,
        source_sha256=_sha256(
            acceptance_raw.get("source_video_sha256"),
            "layout acceptance.source_video_sha256",
        ),
        allow_timing_derivative=layout.temporal_offset_source != "unmeasured",
    )
    if not layout_review["human_reviewed"]:
        raise ValueError("a confidence override requires a human-reviewed layout acceptance")

    override = {
        "format_version": OVERRIDE_VERSION,
        "video_id": layout.video_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": {
            "approved": True,
            "reviewer_identity": identity,
            "reviewer_kind": reviewer_kind,
            "rationale": reason,
        },
        "layout": {
            "path": layout_file.name,
            "sha256": layout_hash,
        },
        "layout_acceptance": {
            "path": layout_acceptance_file.name,
            "sha256": layout_review["sha256"],
            "format_version": LAYOUT_ACCEPTANCE_VERSION,
            "reviewer_kind": layout_review["reviewer_kind"],
            "human_reviewed": layout_review["human_reviewed"],
        },
        "overridden_gate": {
            "gate": OVERRIDDEN_GATE,
            "inference_confidence": layout.inference_confidence,
            "min_layout_confidence": floor,
        },
        "source_video_sha256": layout_review["source_video_sha256"],
    }
    payload = (json.dumps(override, indent=2) + "\n").encode("utf-8")
    _atomic_write_new(override_file, payload)
    return override


def verify_confidence_override(
    layout_path: str | Path,
    layout: WildLayout,
    override_path: str | Path,
    *,
    layout_acceptance: dict[str, Any],
    source_sha256: str,
    min_layout_confidence: float,
) -> dict[str, Any]:
    """Verify an override against the layout, acceptance, and policy in use.

    ``layout_acceptance`` is the verified result of
    :func:`harvest.accept_wild_layout.verify_layout_acceptance` for the same
    decode.  Any mismatch is a hard error, never a silent fallthrough.
    """

    layout_file = Path(layout_path)
    override_file = Path(override_path)
    override = _mapping(json.loads(override_file.read_text()), "confidence override")
    if override.get("format_version") != OVERRIDE_VERSION:
        raise ValueError("unsupported confidence override format_version")
    if override.get("video_id") != layout.video_id:
        raise ValueError("confidence override video_id differs from layout")

    decision = _mapping(override.get("decision"), "confidence override.decision")
    if decision.get("approved") is not True:
        raise ValueError("confidence override lacks explicit approval")
    identity = str(decision.get("reviewer_identity", "")).strip()
    kind = str(decision.get("reviewer_kind", "")).strip()
    if not identity or kind not in REVIEWER_KINDS:
        raise ValueError(
            "confidence override reviewer identity/kind is not a human provenance"
        )
    reason = str(decision.get("rationale", "")).strip()
    if not reason:
        raise ValueError("confidence override lacks a non-empty rationale")

    layout_row = _mapping(override.get("layout"), "confidence override.layout")
    if _local_name(layout_row.get("path"), "confidence override.layout.path") != (
        layout_file.name
    ):
        raise ValueError("confidence override names a different layout")
    if _sha256(
        layout_row.get("sha256"), "confidence override.layout.sha256"
    ) != sha256_file(layout_file):
        raise ValueError("confidence override is bound to different layout bytes")

    acceptance_row = _mapping(
        override.get("layout_acceptance"), "confidence override.layout_acceptance"
    )
    if acceptance_row.get("format_version") != LAYOUT_ACCEPTANCE_VERSION:
        raise ValueError("confidence override names an unsupported layout acceptance")
    if _local_name(
        acceptance_row.get("path"), "confidence override.layout_acceptance.path"
    ) != Path(str(layout_acceptance["path"])).name:
        raise ValueError("confidence override names a different layout acceptance")
    if _sha256(
        acceptance_row.get("sha256"), "confidence override.layout_acceptance.sha256"
    ) != layout_acceptance["sha256"]:
        raise ValueError(
            "confidence override is bound to different layout-acceptance bytes"
        )
    if acceptance_row.get("reviewer_kind") != layout_acceptance["reviewer_kind"]:
        raise ValueError("confidence override layout reviewer kind is inconsistent")
    if (
        acceptance_row.get("human_reviewed") is not True
        or layout_acceptance["human_reviewed"] is not True
    ):
        raise ValueError("confidence override requires human-reviewed layout provenance")

    gate_row = _mapping(
        override.get("overridden_gate"), "confidence override.overridden_gate"
    )
    if gate_row.get("gate") != OVERRIDDEN_GATE:
        raise ValueError("confidence override names a different admission gate")
    recorded_confidence = _number(
        gate_row.get("inference_confidence"),
        "confidence override.overridden_gate.inference_confidence",
    )
    recorded_floor = _number(
        gate_row.get("min_layout_confidence"),
        "confidence override.overridden_gate.min_layout_confidence",
    )
    _same_float(
        recorded_confidence,
        layout.inference_confidence,
        "confidence override inference_confidence",
    )
    _same_float(
        recorded_floor,
        _number(min_layout_confidence, "min_layout_confidence"),
        "confidence override admission floor",
    )
    if not recorded_confidence < recorded_floor:
        raise ValueError(
            "confidence override records a confidence that is not below its floor"
        )

    if _sha256(
        override.get("source_video_sha256"), "confidence override.source_video_sha256"
    ) != _sha256(source_sha256, "source_sha256"):
        raise ValueError("confidence override is bound to a different source video")

    return {
        "path": str(override_file),
        "sha256": sha256_file(override_file),
        "format_version": OVERRIDE_VERSION,
        "reviewer_identity": identity,
        "reviewer_kind": kind,
        "human_reviewed": True,
        "rationale": reason,
        "overridden_gate": OVERRIDDEN_GATE,
        "inference_confidence": recorded_confidence,
        "min_layout_confidence": recorded_floor,
        "layout_sha256": layout_row["sha256"],
        "layout_acceptance_sha256": acceptance_row["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--layout-acceptance", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewer-kind", choices=REVIEWER_KINDS, required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--approve", action="store_true", required=True)
    parser.add_argument(
        "--floor", type=float, default=DEFAULT_MIN_LAYOUT_CONFIDENCE,
        help="the enforced decode admission floor being overridden",
    )
    args = parser.parse_args()
    result = accept_confidence_override(
        args.layout,
        args.layout_acceptance,
        args.out,
        reviewer_identity=args.reviewer,
        reviewer_kind=args.reviewer_kind,
        rationale=args.rationale,
        approved=args.approve,
        min_layout_confidence=args.floor,
    )
    print(json.dumps({
        "video_id": result["video_id"],
        "inference_confidence": result["overridden_gate"]["inference_confidence"],
        "min_layout_confidence": result["overridden_gate"]["min_layout_confidence"],
        "reviewer_identity": result["decision"]["reviewer_identity"],
        "reviewer_kind": result["decision"]["reviewer_kind"],
        "override": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
