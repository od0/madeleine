"""Re-verdict an immutable v2 offset calibration under OffsetPolicy v3.

The 2026-07-28 ground-truth diagnostic
(``results/wild20/offset-gate-groundtruth-diagnostic/``) showed the v2
per-event mode/collar fraction measures footage SNR, not offset correctness,
so OffsetPolicy v3 removed it as a blocking gate and added the
``uncertain_adjacent`` tier.  The v2 gate statistics are all serialized in
``offset_calibration.json`` per candidate lag (median scores, per-event wins,
bootstrap wins, temporal-block winners), which is a sufficient record to
recompute every v3 gate.  This tool therefore re-verdicts an existing v2
output directory without reprocessing the source video:

1. verify the v2 calibration bytes against their SHA-256 sidecar and the
   contact sheet against the hash the record binds;
2. recompute the v3 gates from the serialized per-lag statistics,
   cross-checking every serialized aggregate against its recomputation;
3. write a fresh v3 ``offset_calibration.json`` plus sidecar into an
   ``offset-v3/`` directory, copying the existing contact sheet beside it
   with its hash re-bound and recording the v2 provenance.

The v3 record it writes is a normal pending-review handoff: acceptance still
happens only through ``harvest.accept_wild_offset``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
from typing import Any

from harvest.calibrate_offset import (
    CALIBRATION_VERSION,
    HANDOFF_INSTRUCTION,
    MARGIN_FAILURE_PREFIX,
    OffsetPolicy,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_UNCERTAIN_ADJACENT,
)
from harvest.fetch_wild import sha256_file


V2_CALIBRATION_VERSION = "madeleine.dash-hitstop-offset.v2"
REVERDICT_VERSION = "madeleine.offset-reverdict.v3"


def _check_close(actual: float, expected: float, field: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"serialized {field} disagrees with its recomputation from the "
            f"per-lag record ({actual!r} vs {expected!r})"
        )


def _verified_v2_record(v2_dir: Path) -> tuple[dict[str, Any], str, Path]:
    calibration_path = v2_dir / "offset_calibration.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"missing {calibration_path}")
    calibration = json.loads(calibration_path.read_text())
    if calibration.get("format_version") != V2_CALIBRATION_VERSION:
        raise ValueError(
            f"expected a {V2_CALIBRATION_VERSION} record; "
            f"got {calibration.get('format_version')!r}"
        )
    sidecar = calibration_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError("v2 calibration SHA-256 sidecar is missing")
    calibration_hash = sha256_file(calibration_path)
    if sidecar.read_text().strip() != f"{calibration_hash}  {calibration_path.name}":
        raise ValueError("v2 calibration SHA-256 sidecar does not match its bytes")

    handoff = calibration.get("human_handoff") or {}
    contact_name = str(handoff.get("contact_sheet", "")).strip()
    contact_path = v2_dir / contact_name
    if not contact_name or not contact_path.is_file():
        raise FileNotFoundError("v2 contact sheet is missing")
    if sha256_file(contact_path) != handoff.get("contact_sheet_sha256"):
        raise ValueError("v2 contact-sheet bytes differ from the recorded hash")

    # v2 never serialized a score matrix; if one is recorded, its hash must
    # match, and an unrecorded matrix lying beside the record is rejected as
    # unbound evidence.
    recorded_matrix = calibration.get("score_matrix")
    matrix_path = v2_dir / "score_matrix.npz"
    if recorded_matrix is not None:
        bound = v2_dir / str(recorded_matrix.get("path", "score_matrix.npz"))
        if not bound.is_file():
            raise FileNotFoundError("recorded score matrix is missing")
        if sha256_file(bound) != recorded_matrix.get("sha256"):
            raise ValueError("score matrix bytes differ from the v2 record")
    elif matrix_path.exists():
        raise ValueError("score_matrix.npz exists but is not bound by the v2 record")
    return calibration, calibration_hash, contact_path


def _v3_policy(v2_policy: dict[str, Any]) -> OffsetPolicy:
    fields = dict(v2_policy)
    fields.pop("min_mode_fraction", None)
    policy = OffsetPolicy(**fields)
    policy.lags()  # runs the structural validation
    return policy


def reverdict_gates(
    calibration: dict[str, Any], policy: OffsetPolicy
) -> dict[str, Any]:
    """Recompute the v3 gates from serialized per-lag sufficient statistics."""

    winner = calibration.get("best_candidate_offset_frames")
    if not isinstance(winner, int) or isinstance(winner, bool):
        raise ValueError("v2 record has no integer winning offset to re-verdict")
    by_lag: dict[int, dict[str, Any]] = {}
    for row in calibration["candidates"]:
        by_lag[int(row["offset_frames"])] = row
    expected_lags = set(range(policy.min_lag, policy.max_lag + 1))
    if set(by_lag) != expected_lags:
        raise ValueError("v2 candidates do not exactly cover the lag search")
    usable = int(calibration["events"]["usable_quality_matches"])

    hard_failures: list[str] = []
    margin_failures: list[str] = []
    if usable < policy.min_events:
        hard_failures.append(f"usable events {usable} < required {policy.min_events}")
    if winner in (policy.min_lag, policy.max_lag):
        hard_failures.append("winning lag lies on search boundary")

    winner_median = float(by_lag[winner]["median_score"])
    if winner_median != max(float(row["median_score"]) for row in by_lag.values()):
        raise ValueError("v2 winner is not the highest-median candidate")
    margin = winner_median - max(
        float(row["median_score"])
        for lag, row in by_lag.items()
        if abs(lag - winner) >= policy.margin_nonadjacent_gap
    )
    _check_close(
        float(calibration["median_score_margin"]), margin, "median_score_margin"
    )
    if margin < policy.min_median_margin:
        margin_failures.append(
            f"{MARGIN_FAILURE_PREFIX} {margin:.4f} < required "
            f"{policy.min_median_margin:.4f}"
        )

    for block in calibration["temporal_blocks"]:
        number = int(block["block"])
        events = int(block["events"])
        block_winner = block["winner_offset_frames"]
        if events < policy.min_events_per_block or block_winner is None:
            hard_failures.append(
                f"temporal block {number} has {events} events; "
                f"need {policy.min_events_per_block}"
            )
        elif abs(int(block_winner) - winner) > policy.mode_lag_collar:
            hard_failures.append(
                f"temporal block {number} winner {int(block_winner)} outside "
                f"winner±{policy.mode_lag_collar} of {winner}"
            )

    bootstrap_fraction = sum(
        int(row["bootstrap_wins"])
        for lag, row in by_lag.items()
        if abs(lag - winner) <= policy.mode_lag_collar
    ) / policy.bootstrap_samples
    _check_close(
        float(calibration["bootstrap_win_fraction"]),
        bootstrap_fraction,
        "bootstrap_win_fraction",
    )
    if bootstrap_fraction < policy.min_bootstrap_win_fraction:
        hard_failures.append(
            f"bootstrap winner±{policy.mode_lag_collar} fraction "
            f"{bootstrap_fraction:.3f} < required "
            f"{policy.min_bootstrap_win_fraction:.3f}"
        )

    collar_fraction = sum(
        int(row["event_wins"])
        for lag, row in by_lag.items()
        if abs(lag - winner) <= policy.mode_lag_collar
    ) / usable
    modal_lag = int(calibration["per_event_modal_offset_frames"])
    _check_close(
        float(calibration["per_event_mode_fraction"]),
        int(by_lag[modal_lag]["event_wins"]) / usable,
        "per_event_mode_fraction",
    )

    if not hard_failures and not margin_failures:
        verdict = VERDICT_PASS
    elif not hard_failures:
        verdict = VERDICT_UNCERTAIN_ADJACENT
    else:
        verdict = VERDICT_FAIL
    return {
        "verdict": verdict,
        "failure_reasons": hard_failures + margin_failures,
        "per_event_collar_fraction": collar_fraction,
        "median_score_margin": margin,
        "bootstrap_win_fraction": bootstrap_fraction,
    }


def reverdict_calibration_dir(
    v2_dir: str | Path, out_dir: str | Path | None = None
) -> dict[str, Any]:
    source_dir = Path(v2_dir)
    destination = Path(out_dir) if out_dir is not None else (
        source_dir.parent / "offset-v3"
    )
    calibration, v2_hash, contact_path = _verified_v2_record(source_dir)
    policy = _v3_policy(calibration["policy"])
    gates = reverdict_gates(calibration, policy)

    report_path = destination / "offset_calibration.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    destination.mkdir(parents=True, exist_ok=True)
    new_contact = destination / contact_path.name
    if not new_contact.exists():
        shutil.copyfile(contact_path, new_contact)
    contact_hash = sha256_file(new_contact)
    if contact_hash != calibration["human_handoff"]["contact_sheet_sha256"]:
        raise ValueError("copied contact-sheet bytes differ from the v2 record")

    result = dict(calibration)
    result.update({
        "format_version": CALIBRATION_VERSION,
        "policy": asdict(policy),
        "per_event_collar_fraction": gates["per_event_collar_fraction"],
        "offset_uncertainty_frames": int(policy.mode_lag_collar),
        "verdict": gates["verdict"],
        "automatic_gates_passed": gates["verdict"] == VERDICT_PASS,
        "automatic_failure_reasons": gates["failure_reasons"],
        "human_contact_sheet_review": "pending",
        "calibration_accepted": False,
        "layout_was_modified": False,
        "reverdict": {
            "format_version": REVERDICT_VERSION,
            "from_calibration": str(source_dir / "offset_calibration.json"),
            "from_calibration_sha256": v2_hash,
            "from_format_version": V2_CALIBRATION_VERSION,
            "method": (
                "v3 gates recomputed from the serialized per-lag statistics of "
                "the hash-verified v2 record; the source video was not "
                "reprocessed and the contact sheet is byte-identical"
            ),
        },
    })
    handoff = dict(result["human_handoff"])
    handoff.update({
        "contact_sheet": new_contact.name,
        "contact_sheet_sha256": contact_hash,
        "instruction": HANDOFF_INSTRUCTION,
    })
    result["human_handoff"] = handoff
    # A reverdict record carries no score matrix of its own; drop any stale
    # binding rather than pointing at a file this directory does not contain.
    result.pop("score_matrix", None)

    report_path.write_text(json.dumps(result, indent=2) + "\n")
    report_hash = sha256_file(report_path)
    (destination / "offset_calibration.sha256").write_text(
        f"{report_hash}  {report_path.name}\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = reverdict_calibration_dir(args.v2_dir, args.out)
    print(json.dumps({
        "video_id": result["video_id"],
        "best_candidate_offset_frames": result["best_candidate_offset_frames"],
        "verdict": result["verdict"],
        "median_score_margin": result["median_score_margin"],
        "per_event_collar_fraction": result["per_event_collar_fraction"],
        "bootstrap_win_fraction": result["bootstrap_win_fraction"],
        "automatic_failure_reasons": result["automatic_failure_reasons"],
    }, indent=2))


if __name__ == "__main__":
    main()
