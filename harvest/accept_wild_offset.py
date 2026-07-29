"""Accept a measured wild-HUD offset through a hash-bound review artifact.

The dash-hitstop calibrator deliberately leaves its result pending.  This
module is the only supported bridge from that pending evidence to a measured
layout: it rechecks every serialized automatic gate, verifies the calibration
and contact-sheet bytes, records an explicit reviewer and approval, and writes
both a new layout and an acceptance artifact without overwriting either path.

Only v3 calibrations are acceptable.  A ``pass`` verdict is accepted normally;
an ``uncertain_adjacent`` verdict (margin below floor, winner decisive by
bootstrap and unanimous temporal blocks within the ±1 collar) additionally
requires the explicit ``--accept-uncertain-tier`` acknowledgement, and the
acceptance artifact records the tier and the ±1-frame offset uncertainty.

Final decoding verifies the acceptance artifact and its referenced evidence;
editable offset fields in a layout are not sufficient for admission.
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

from harvest.accept_wild_layout import (
    LAYOUT_ACCEPTANCE_VERSION,
    verify_layout_acceptance,
)
from harvest.fetch_wild import sha256_file
from harvest.wild_layout import WildLayout


ACCEPTANCE_VERSION = "madeleine.wild-offset-acceptance.v3"
CALIBRATION_VERSION = "madeleine.dash-hitstop-offset.v3"
# v2 calibration artifacts remain readable evidence records, but acceptance
# requires a v3 calibration (produce one with harvest.reverdict_offset_v3
# from the immutable v2 record, or recalibrate).
VERDICT_PASS = "pass"
VERDICT_UNCERTAIN_ADJACENT = "uncertain_adjacent"
ACCEPTABLE_VERDICTS = (VERDICT_PASS, VERDICT_UNCERTAIN_ADJACENT)
MARGIN_FAILURE_PREFIX = "non-adjacent median margin"
REVIEWER_KINDS = ("human", "human_with_ai_assistance", "ai_agent")
HUMAN_REVIEWER_KINDS = frozenset(("human", "human_with_ai_assistance"))


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
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
        raise ValueError(f"{field} is inconsistent with serialized gate evidence")


def _validate_automatic_gates(
    calibration: dict[str, Any], *, video_id: str, input_layout_sha256: str
) -> dict[str, Any]:
    """Recheck every serialized calibration gate instead of trusting one bool."""

    if calibration.get("format_version") != CALIBRATION_VERSION:
        raise ValueError(f"unsupported calibration format_version; expected {CALIBRATION_VERSION}")
    if calibration.get("video_id") != video_id:
        raise ValueError("calibration video_id differs from layout")
    inputs = _mapping(calibration.get("inputs"), "calibration.inputs")
    if (
        _sha256(inputs.get("layout_sha256"), "calibration.inputs.layout_sha256")
        != input_layout_sha256
    ):
        raise ValueError("calibration is bound to a different input layout")
    source_sha256 = _sha256(
        inputs.get("video_sha256"), "calibration.inputs.video_sha256"
    )
    verdict = calibration.get("verdict")
    if verdict not in ACCEPTABLE_VERDICTS:
        raise ValueError(
            "calibration verdict must be pass or uncertain_adjacent; "
            f"got {verdict!r}"
        )
    failures = _sequence(
        calibration.get("automatic_failure_reasons"),
        "calibration.automatic_failure_reasons",
    )
    if verdict == VERDICT_PASS:
        if calibration.get("automatic_gates_passed") is not True:
            raise ValueError("pass verdict requires automatic_gates_passed true")
        if failures:
            raise ValueError("pass verdict may not record gate failures")
    else:
        if calibration.get("automatic_gates_passed") is not False:
            raise ValueError(
                "uncertain_adjacent verdict may not claim automatic_gates_passed"
            )
        if not failures or any(
            not str(reason).startswith(MARGIN_FAILURE_PREFIX) for reason in failures
        ):
            raise ValueError(
                "uncertain_adjacent verdict requires margin-only failure reasons"
            )
    if calibration.get("human_contact_sheet_review") != "pending":
        raise ValueError("calibration must be the immutable pending-review handoff")
    if calibration.get("calibration_accepted") is not False:
        raise ValueError("calibration must not claim its own acceptance")
    if calibration.get("layout_was_modified") is not False:
        raise ValueError("calibration must not claim it modified the input layout")

    policy = _mapping(calibration.get("policy"), "calibration.policy")
    min_lag = _integer(policy.get("min_lag"), "calibration.policy.min_lag")
    max_lag = _integer(policy.get("max_lag"), "calibration.policy.max_lag")
    if min_lag >= max_lag:
        raise ValueError("calibration lag range is invalid")
    winner = _integer(
        calibration.get("best_candidate_offset_frames"),
        "calibration.best_candidate_offset_frames",
    )
    if not min_lag < winner < max_lag:
        raise ValueError("winning offset lies outside or on the search boundary")

    events = _mapping(calibration.get("events"), "calibration.events")
    usable_events = _integer(
        events.get("usable_quality_matches"),
        "calibration.events.usable_quality_matches",
    )
    min_events = _integer(policy.get("min_events"), "calibration.policy.min_events")
    if min_events < 20:
        raise ValueError("calibration policy weakens the 20-event minimum")
    if usable_events < min_events:
        raise ValueError("calibration has too few usable events")

    candidates = _sequence(calibration.get("candidates"), "calibration.candidates")
    expected_lags = set(range(min_lag, max_lag + 1))
    by_lag: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(candidates):
        row = _mapping(raw, f"calibration.candidates[{index}]")
        lag = _integer(row.get("offset_frames"), f"calibration.candidates[{index}].offset_frames")
        if lag in by_lag:
            raise ValueError("calibration candidate offsets are duplicated")
        _number(row.get("median_score"), f"calibration.candidates[{index}].median_score")
        _number(row.get("mean_score"), f"calibration.candidates[{index}].mean_score")
        _integer(row.get("event_wins"), f"calibration.candidates[{index}].event_wins")
        _integer(row.get("bootstrap_wins"), f"calibration.candidates[{index}].bootstrap_wins")
        _number(
            row.get("bootstrap_fraction"),
            f"calibration.candidates[{index}].bootstrap_fraction",
        )
        by_lag[lag] = row
    if set(by_lag) != expected_lags:
        raise ValueError("calibration candidates do not exactly cover the declared lag search")

    winner_median = _number(by_lag[winner]["median_score"], "winner median_score")
    if winner_median != max(
        _number(row["median_score"], "candidate median_score")
        for row in by_lag.values()
    ):
        raise ValueError("best offset is not the highest-median candidate")
    runner = _integer(calibration.get("runner_up_offset_frames"), "runner_up_offset_frames")
    if runner == winner or runner not in by_lag:
        raise ValueError("runner-up offset is invalid")
    runner_median = _number(by_lag[runner]["median_score"], "runner median_score")
    expected_runner_median = max(
        _number(row["median_score"], "candidate median_score")
        for lag, row in by_lag.items()
        if lag != winner
    )
    if runner_median != expected_runner_median:
        raise ValueError("runner-up offset is inconsistent with candidates")
    collar = _integer(policy.get("mode_lag_collar"), "policy.mode_lag_collar")
    if not 0 <= collar <= 1:
        raise ValueError("calibration policy widens the winner collar")
    nonadjacent_gap = _integer(
        policy.get("margin_nonadjacent_gap"), "policy.margin_nonadjacent_gap"
    )
    if nonadjacent_gap < 2 or nonadjacent_gap <= collar:
        raise ValueError("calibration policy weakens the non-adjacent margin gap")
    uncertainty = _integer(
        calibration.get("offset_uncertainty_frames"), "offset_uncertainty_frames"
    )
    if uncertainty != collar:
        raise ValueError("offset uncertainty differs from the policy collar")

    # The margin is winner median minus the best non-adjacent median, matching
    # the calibrator; the runner-up (best non-winner overall) may legitimately
    # sit inside the collar.
    computed_margin = winner_median - max(
        _number(row["median_score"], "candidate median_score")
        for lag, row in by_lag.items()
        if abs(lag - winner) >= nonadjacent_gap
    )
    margin = _number(calibration.get("median_score_margin"), "median_score_margin")
    _same_float(margin, computed_margin, "median_score_margin")
    min_margin = _number(policy.get("min_median_margin"), "policy.min_median_margin")
    if min_margin < 2.0:
        raise ValueError("calibration policy weakens the median-score margin")
    if verdict == VERDICT_PASS and margin < min_margin:
        raise ValueError("calibration median-score margin is below policy")
    if verdict == VERDICT_UNCERTAIN_ADJACENT and margin >= min_margin:
        raise ValueError(
            "uncertain_adjacent verdict is inconsistent with a passing margin"
        )

    # Mode and collar fractions are recorded SNR indicators (see the
    # 2026-07-28 ground-truth diagnostic); they are validated for internal
    # consistency but are not gates.
    modal_lag = _integer(
        calibration.get("per_event_modal_offset_frames"),
        "per_event_modal_offset_frames",
    )
    mode_fraction = _number(calibration.get("per_event_mode_fraction"), "per_event_mode_fraction")
    modal_wins = _integer(by_lag[modal_lag]["event_wins"], "modal event_wins")
    if modal_wins != max(
        _integer(row["event_wins"], "candidate event_wins") for row in by_lag.values()
    ):
        raise ValueError("per-event modal offset is not the most-winning lag")
    _same_float(mode_fraction, modal_wins / usable_events, "per_event_mode_fraction")
    collar_fraction = _number(
        calibration.get("per_event_collar_fraction"), "per_event_collar_fraction"
    )
    _same_float(
        collar_fraction,
        sum(
            _integer(row["event_wins"], "candidate event_wins")
            for lag, row in by_lag.items()
            if abs(lag - winner) <= collar
        ) / usable_events,
        "per_event_collar_fraction",
    )
    winner_row = by_lag[winner]

    bootstrap_samples = _integer(
        policy.get("bootstrap_samples"), "calibration.policy.bootstrap_samples"
    )
    if bootstrap_samples < 2_000:
        raise ValueError("calibration policy weakens the bootstrap sample count")
    bootstrap_fraction = _number(
        calibration.get("bootstrap_win_fraction"), "bootstrap_win_fraction"
    )
    min_bootstrap_fraction = _number(
        policy.get("min_bootstrap_win_fraction"), "policy.min_bootstrap_win_fraction"
    )
    if min_bootstrap_fraction < 0.95:
        raise ValueError("calibration policy weakens the bootstrap win fraction")
    if bootstrap_fraction < min_bootstrap_fraction:
        raise ValueError("bootstrap winner fraction is below policy")
    # The serialized fraction counts bootstrap winners within winner±collar,
    # matching the calibrator.
    _same_float(
        bootstrap_fraction,
        sum(
            _integer(row["bootstrap_wins"], "candidate bootstrap_wins")
            for lag, row in by_lag.items()
            if abs(lag - winner) <= collar
        ) / bootstrap_samples,
        "bootstrap_win_fraction",
    )
    _same_float(
        _number(winner_row["bootstrap_fraction"], "winner bootstrap_fraction"),
        _integer(winner_row["bootstrap_wins"], "winner bootstrap_wins")
        / bootstrap_samples,
        "winner bootstrap_fraction",
    )

    blocks = _sequence(calibration.get("temporal_blocks"), "calibration.temporal_blocks")
    block_count = _integer(policy.get("temporal_blocks"), "policy.temporal_blocks")
    minimum_per_block = _integer(
        policy.get("min_events_per_block"), "policy.min_events_per_block"
    )
    if block_count < 3 or minimum_per_block < 4:
        raise ValueError("calibration policy weakens temporal-block evidence")
    if len(blocks) != block_count:
        raise ValueError("calibration temporal-block count differs from policy")
    for index, raw in enumerate(blocks):
        block = _mapping(raw, f"calibration.temporal_blocks[{index}]")
        if _integer(block.get("block"), f"temporal_blocks[{index}].block") != index:
            raise ValueError("calibration temporal blocks are not canonical")
        if _integer(block.get("events"), f"temporal_blocks[{index}].events") < minimum_per_block:
            raise ValueError("calibration temporal block has too few events")
        if abs(
            _integer(
                block.get("winner_offset_frames"),
                f"temporal_blocks[{index}].winner_offset_frames",
            )
            - winner
        ) > collar:
            raise ValueError(
                "calibration temporal blocks disagree with the winner beyond the collar"
            )

    if sum(
        _integer(row["event_wins"], "candidate event_wins")
        for row in by_lag.values()
    ) != usable_events:
        raise ValueError("candidate event-win counts do not sum to usable events")
    if sum(
        _integer(row["bootstrap_wins"], "candidate bootstrap_wins")
        for row in by_lag.values()
    ) != bootstrap_samples:
        raise ValueError("candidate bootstrap-win counts do not sum to bootstrap samples")
    if sum(
        _integer(block["events"], "temporal block events") for block in blocks
    ) != usable_events:
        raise ValueError("temporal-block event counts do not sum to usable events")

    # The report version identifies the estimator, while these floors prevent a
    # programmatic caller from weakening its physical/cadence gates and still
    # producing an apparently successful acceptance.
    lower_bounds = {
        "min_effective_fps": 59.0,
        "min_local_motion_range": 0.50,
        "min_event_score": 3.0,
    }
    upper_bounds = {
        "max_effective_fps": 61.0,
        "max_vfr_ratio_p99_p01": 1.10,
    }
    for field, lower in lower_bounds.items():
        if _number(policy.get(field), f"policy.{field}") < lower:
            raise ValueError(f"calibration policy weakens {field}")
    for field, upper in upper_bounds.items():
        if _number(policy.get(field), f"policy.{field}") > upper:
            raise ValueError(f"calibration policy weakens {field}")

    handoff = _mapping(calibration.get("human_handoff"), "calibration.human_handoff")
    contact_name = _local_name(handoff.get("contact_sheet"), "human_handoff.contact_sheet")
    contact_hash = _sha256(
        handoff.get("contact_sheet_sha256"), "human_handoff.contact_sheet_sha256"
    )
    return {
        "winner": winner,
        "verdict": verdict,
        "uncertainty": uncertainty,
        "confidence": bootstrap_fraction,
        "contact_name": contact_name,
        "contact_sha256": contact_hash,
        "source_sha256": source_sha256,
    }


def _load_and_validate_calibration(
    calibration_path: Path, *, video_id: str, input_layout_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    calibration = json.loads(calibration_path.read_text())
    checked = _validate_automatic_gates(
        calibration, video_id=video_id, input_layout_sha256=input_layout_sha256
    )
    sidecar = calibration_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError("calibration SHA-256 sidecar is missing")
    calibration_hash = sha256_file(calibration_path)
    if sidecar.read_text().strip() != f"{calibration_hash}  {calibration_path.name}":
        raise ValueError("calibration SHA-256 sidecar does not match calibration bytes")
    contact_path = calibration_path.parent / checked["contact_name"]
    if not contact_path.is_file():
        raise FileNotFoundError("calibration contact sheet is missing")
    if sha256_file(contact_path) != checked["contact_sha256"]:
        raise ValueError("calibration contact-sheet hash mismatch")
    checked["calibration_sha256"] = calibration_hash
    return calibration, checked, contact_path


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


def accept_offset(
    calibration_path: str | Path,
    input_layout_path: str | Path,
    layout_acceptance_path: str | Path,
    output_layout_path: str | Path,
    acceptance_path: str | Path,
    *,
    reviewer_identity: str,
    reviewer_kind: str,
    approved: bool,
    accept_uncertain_tier: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    """Create a reviewed measured layout and its immutable acceptance contract."""

    calibration_file = Path(calibration_path)
    input_layout_file = Path(input_layout_path)
    layout_acceptance_file = Path(layout_acceptance_path)
    output_layout_file = Path(output_layout_path)
    acceptance_file = Path(acceptance_path)
    identity = reviewer_identity.strip()
    if not identity:
        raise ValueError("reviewer_identity is required")
    if reviewer_kind not in REVIEWER_KINDS:
        raise ValueError(f"reviewer_kind must be one of {REVIEWER_KINDS}")
    if approved is not True:
        raise ValueError("explicit contact-sheet approval is required")
    if output_layout_file == input_layout_file:
        raise ValueError("output layout must be a new path")
    if acceptance_file.parent.resolve() != calibration_file.parent.resolve():
        raise ValueError("acceptance artifact must live beside its calibration evidence")
    if output_layout_file.exists() or acceptance_file.exists():
        raise FileExistsError("refusing to overwrite an existing output layout or acceptance")

    input_raw = json.loads(input_layout_file.read_text())
    input_layout = WildLayout.from_dict(input_raw)
    if input_layout.temporal_offset_source != "unmeasured":
        raise ValueError("input layout temporal offset must be explicitly unmeasured")
    input_hash = sha256_file(input_layout_file)
    calibration, checked, contact_file = _load_and_validate_calibration(
        calibration_file,
        video_id=input_layout.video_id,
        input_layout_sha256=input_hash,
    )
    if checked["verdict"] == VERDICT_UNCERTAIN_ADJACENT and not accept_uncertain_tier:
        raise ValueError(
            "calibration verdict is uncertain_adjacent (median margin below "
            "floor with a bootstrap- and block-decisive winner); acceptance "
            "requires explicit --accept-uncertain-tier acknowledgement"
        )
    if checked["verdict"] == VERDICT_PASS and accept_uncertain_tier:
        raise ValueError(
            "--accept-uncertain-tier applies only to uncertain_adjacent verdicts"
        )
    layout_review = verify_layout_acceptance(
        input_layout_file,
        input_layout,
        layout_acceptance_file,
        source_sha256=checked["source_sha256"],
    )
    if not layout_review["human_reviewed"]:
        raise ValueError("input layout acceptance was not reviewed by a human")
    calibration_hash = checked["calibration_sha256"]
    offset_source = f"dash_hitstop_v1+reviewed_acceptance:{calibration_hash}"

    output_raw = dict(input_raw)
    output_raw.update({
        "temporal_offset_frames": checked["winner"],
        "temporal_offset_source": offset_source,
        "temporal_offset_confidence": checked["confidence"],
        "temporal_offset_acceptance": {
            "format_version": ACCEPTANCE_VERSION,
            "artifact": acceptance_file.name,
            "calibration_sha256": calibration_hash,
            "tier": checked["verdict"],
            "offset_uncertainty_frames": checked["uncertainty"],
        },
    })
    WildLayout.from_dict(output_raw)
    layout_bytes = (json.dumps(output_raw, indent=2) + "\n").encode("utf-8")

    # Hash the exact bytes before publishing either file.  The acceptance binds
    # the generated layout; the layout embeds the immutable calibration hash.
    output_layout_hash = hashlib.sha256(layout_bytes).hexdigest()
    created_at = datetime.now(timezone.utc).isoformat()
    acceptance = {
        "format_version": ACCEPTANCE_VERSION,
        "video_id": input_layout.video_id,
        "created_at": created_at,
        "decision": {
            "approved": True,
            "contact_sheet_reviewed": True,
            "reviewer_identity": identity,
            "reviewer_kind": reviewer_kind,
            "uncertain_tier_acknowledged": (
                checked["verdict"] == VERDICT_UNCERTAIN_ADJACENT
            ),
            "notes": notes,
        },
        "calibration": {
            "path": calibration_file.name,
            "sha256": calibration_hash,
            "format_version": calibration["format_version"],
            "contact_sheet": {
                "path": contact_file.name,
                "sha256": checked["contact_sha256"],
            },
        },
        "input_layout": {
            "path": input_layout_file.name,
            "sha256": input_hash,
        },
        "layout_review_acceptance": {
            "path": layout_acceptance_file.name,
            "sha256": layout_review["sha256"],
            "format_version": LAYOUT_ACCEPTANCE_VERSION,
            "review_manifest_sha256": layout_review["review_manifest_sha256"],
            "reviewed_layout_sha256": layout_review["reviewed_layout_sha256"],
            "reviewer_kind": layout_review["reviewer_kind"],
            "human_reviewed": layout_review["human_reviewed"],
        },
        "output_layout": {
            "path": output_layout_file.name,
            "sha256": output_layout_hash,
        },
        "accepted_offset_frames": checked["winner"],
        "accepted_offset_confidence": checked["confidence"],
        "accepted_offset_source": offset_source,
        "accepted_tier": checked["verdict"],
        "accepted_offset_uncertainty_frames": checked["uncertainty"],
        "source_video_sha256": checked["source_sha256"],
    }
    acceptance_bytes = (json.dumps(acceptance, indent=2) + "\n").encode("utf-8")

    # Either half of an interrupted two-file handoff remains unusable: final
    # decoding requires both and verifies their hashes.  Neither can overwrite.
    _atomic_write_new(output_layout_file, layout_bytes)
    _atomic_write_new(acceptance_file, acceptance_bytes)
    return acceptance


def verify_offset_acceptance(
    layout_path: str | Path,
    layout: WildLayout,
    acceptance_path: str | Path,
    *,
    source_sha256: str,
    layout_acceptance_path: str | Path,
) -> dict[str, Any]:
    """Verify a measured layout against acceptance and original evidence bytes."""

    layout_file = Path(layout_path)
    acceptance_file = Path(acceptance_path)
    acceptance = json.loads(acceptance_file.read_text())
    if acceptance.get("format_version") != ACCEPTANCE_VERSION:
        raise ValueError("unsupported offset acceptance format_version")
    if acceptance.get("video_id") != layout.video_id:
        raise ValueError("offset acceptance video_id differs from layout")
    decision = _mapping(acceptance.get("decision"), "acceptance.decision")
    if decision.get("approved") is not True or decision.get("contact_sheet_reviewed") is not True:
        raise ValueError("offset acceptance lacks explicit contact-sheet approval")
    identity = str(decision.get("reviewer_identity", "")).strip()
    kind = str(decision.get("reviewer_kind", "")).strip()
    if not identity or kind not in REVIEWER_KINDS:
        raise ValueError("offset acceptance reviewer identity/kind is invalid")
    tier = str(acceptance.get("accepted_tier", "")).strip()
    if tier not in ACCEPTABLE_VERDICTS:
        raise ValueError("offset acceptance tier is missing or invalid")
    accepted_uncertainty = _integer(
        acceptance.get("accepted_offset_uncertainty_frames"),
        "accepted_offset_uncertainty_frames",
    )
    acknowledged = decision.get("uncertain_tier_acknowledged")
    if not isinstance(acknowledged, bool):
        raise ValueError("offset acceptance must record uncertain_tier_acknowledged")
    if (tier == VERDICT_UNCERTAIN_ADJACENT) != acknowledged:
        raise ValueError(
            "uncertain_tier_acknowledged is inconsistent with the accepted tier"
        )

    output = _mapping(acceptance.get("output_layout"), "acceptance.output_layout")
    if _local_name(output.get("path"), "output_layout.path") != layout_file.name:
        raise ValueError("offset acceptance names a different output layout")
    if _sha256(output.get("sha256"), "output_layout.sha256") != sha256_file(layout_file):
        raise ValueError("offset acceptance is bound to different output-layout bytes")
    input_row = _mapping(acceptance.get("input_layout"), "acceptance.input_layout")
    input_hash = _sha256(input_row.get("sha256"), "input_layout.sha256")
    accepted_offset = _integer(
        acceptance.get("accepted_offset_frames"), "accepted_offset_frames"
    )
    accepted_confidence = _number(
        acceptance.get("accepted_offset_confidence"), "accepted_offset_confidence"
    )
    accepted_source = str(acceptance.get("accepted_offset_source", "")).strip()
    accepted_source_video_hash = _sha256(
        acceptance.get("source_video_sha256"), "source_video_sha256"
    )
    if accepted_source_video_hash != _sha256(source_sha256, "source_sha256"):
        raise ValueError("offset acceptance is bound to a different source video")
    layout_acceptance_file = Path(layout_acceptance_path)
    layout_review = verify_layout_acceptance(
        layout_file,
        layout,
        layout_acceptance_file,
        source_sha256=accepted_source_video_hash,
        allow_timing_derivative=True,
    )
    review_row = _mapping(
        acceptance.get("layout_review_acceptance"),
        "acceptance.layout_review_acceptance",
    )
    if review_row.get("format_version") != LAYOUT_ACCEPTANCE_VERSION:
        raise ValueError("offset acceptance names an unsupported layout acceptance")
    if _local_name(review_row.get("path"), "layout_review_acceptance.path") != (
        layout_acceptance_file.name
    ):
        raise ValueError("offset acceptance names a different layout acceptance")
    if _sha256(
        review_row.get("sha256"), "layout_review_acceptance.sha256"
    ) != layout_review["sha256"]:
        raise ValueError("offset acceptance layout-review hash mismatch")
    if _sha256(
        review_row.get("review_manifest_sha256"),
        "layout_review_acceptance.review_manifest_sha256",
    ) != layout_review["review_manifest_sha256"]:
        raise ValueError("offset acceptance review-manifest hash mismatch")
    if _sha256(
        review_row.get("reviewed_layout_sha256"),
        "layout_review_acceptance.reviewed_layout_sha256",
    ) != layout_review["reviewed_layout_sha256"]:
        raise ValueError("offset acceptance reviewed-layout hash mismatch")
    if input_hash != layout_review["reviewed_layout_sha256"]:
        raise ValueError(
            "offset calibration input-layout hash differs from layout review acceptance"
        )
    if review_row.get("reviewer_kind") != layout_review["reviewer_kind"]:
        raise ValueError("offset acceptance layout reviewer kind is inconsistent")
    if review_row.get("human_reviewed") is not layout_review["human_reviewed"]:
        raise ValueError("offset acceptance layout human gate is inconsistent")
    if not layout_review["human_reviewed"]:
        raise ValueError("offset acceptance requires human-reviewed layout provenance")
    if accepted_offset != layout.temporal_offset_frames:
        raise ValueError("layout offset differs from accepted offset")
    _same_float(
        accepted_confidence,
        layout.temporal_offset_confidence,
        "layout temporal_offset_confidence",
    )
    if accepted_source != layout.temporal_offset_source:
        raise ValueError("layout temporal_offset_source differs from acceptance")

    calibration_row = _mapping(acceptance.get("calibration"), "acceptance.calibration")
    if calibration_row.get("format_version") != CALIBRATION_VERSION:
        raise ValueError("offset acceptance names an unsupported calibration format")
    calibration_name = _local_name(calibration_row.get("path"), "calibration.path")
    calibration_file = acceptance_file.parent / calibration_name
    calibration_hash = _sha256(calibration_row.get("sha256"), "calibration.sha256")
    if sha256_file(calibration_file) != calibration_hash:
        raise ValueError("offset acceptance calibration hash mismatch")
    _, checked, contact_file = _load_and_validate_calibration(
        calibration_file,
        video_id=layout.video_id,
        input_layout_sha256=input_hash,
    )
    if checked["calibration_sha256"] != calibration_hash:
        raise ValueError("offset acceptance names inconsistent calibration hashes")
    if checked["source_sha256"] != accepted_source_video_hash:
        raise ValueError("offset acceptance and calibration name different source videos")
    contact_row = _mapping(calibration_row.get("contact_sheet"), "calibration.contact_sheet")
    if _local_name(contact_row.get("path"), "contact_sheet.path") != contact_file.name:
        raise ValueError("offset acceptance names a different contact sheet")
    if _sha256(contact_row.get("sha256"), "contact_sheet.sha256") != checked["contact_sha256"]:
        raise ValueError("offset acceptance contact-sheet hash differs from calibration")
    if checked["winner"] != accepted_offset:
        raise ValueError("accepted offset differs from calibrated winner")
    if checked["verdict"] != tier:
        raise ValueError("accepted tier differs from the calibration verdict")
    if checked["uncertainty"] != accepted_uncertainty:
        raise ValueError("accepted offset uncertainty differs from the calibration")
    _same_float(checked["confidence"], accepted_confidence, "accepted offset confidence")
    expected_source = f"dash_hitstop_v1+reviewed_acceptance:{calibration_hash}"
    if accepted_source != expected_source:
        raise ValueError("accepted offset source does not bind the calibration hash")

    layout_raw = json.loads(layout_file.read_text())
    embedded = _mapping(
        layout_raw.get("temporal_offset_acceptance"),
        "layout.temporal_offset_acceptance",
    )
    if embedded.get("format_version") != ACCEPTANCE_VERSION:
        raise ValueError("layout offset-acceptance format is missing or unsupported")
    if embedded.get("artifact") != acceptance_file.name:
        raise ValueError("layout names a different offset-acceptance artifact")
    if _sha256(
        embedded.get("calibration_sha256"),
        "layout.temporal_offset_acceptance.calibration_sha256",
    ) != calibration_hash:
        raise ValueError("layout embeds a different calibration hash")
    if embedded.get("tier") != tier:
        raise ValueError("layout embeds a different acceptance tier")
    if _integer(
        embedded.get("offset_uncertainty_frames"),
        "layout.temporal_offset_acceptance.offset_uncertainty_frames",
    ) != accepted_uncertainty:
        raise ValueError("layout embeds a different offset uncertainty")

    return {
        "path": str(acceptance_file),
        "sha256": sha256_file(acceptance_file),
        "format_version": ACCEPTANCE_VERSION,
        "reviewer_identity": identity,
        "reviewer_kind": kind,
        "human_reviewed": kind in HUMAN_REVIEWER_KINDS,
        "tier": tier,
        "offset_uncertainty_frames": accepted_uncertainty,
        "calibration_sha256": calibration_hash,
        "contact_sheet_sha256": checked["contact_sha256"],
        "layout_review_acceptance_sha256": layout_review["sha256"],
        "layout_review_manifest_sha256": layout_review["review_manifest_sha256"],
        "reviewed_layout_sha256": layout_review["reviewed_layout_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--input-layout", type=Path, required=True)
    parser.add_argument("--layout-acceptance", type=Path, required=True)
    parser.add_argument("--output-layout", type=Path, required=True)
    parser.add_argument("--acceptance-out", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewer-kind", choices=REVIEWER_KINDS, required=True)
    parser.add_argument("--approve-contact-sheet", action="store_true", required=True)
    parser.add_argument(
        "--accept-uncertain-tier",
        action="store_true",
        help=(
            "explicitly acknowledge an uncertain_adjacent verdict (median "
            "margin below floor; winner decisive by bootstrap and temporal "
            "blocks; offset uncertainty ±1 frame)"
        ),
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    result = accept_offset(
        args.calibration,
        args.input_layout,
        args.layout_acceptance,
        args.output_layout,
        args.acceptance_out,
        reviewer_identity=args.reviewer,
        reviewer_kind=args.reviewer_kind,
        approved=args.approve_contact_sheet,
        accept_uncertain_tier=args.accept_uncertain_tier,
        notes=args.notes,
    )
    print(json.dumps({
        "video_id": result["video_id"],
        "accepted_offset_frames": result["accepted_offset_frames"],
        "accepted_offset_confidence": result["accepted_offset_confidence"],
        "accepted_tier": result["accepted_tier"],
        "accepted_offset_uncertainty_frames": result["accepted_offset_uncertainty_frames"],
        "reviewer_identity": result["decision"]["reviewer_identity"],
        "reviewer_kind": result["decision"]["reviewer_kind"],
        "output_layout": str(args.output_layout),
        "acceptance": str(args.acceptance_out),
    }, indent=2))


if __name__ == "__main__":
    main()
