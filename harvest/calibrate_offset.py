"""Bounded dash-hitstop evidence for a wild HUD's temporal offset.

Celeste freezes gameplay motion for the three frames after a true dash and
rebounds on frame +4.  For an observed HUD dash onset ``o`` and candidate
offset ``h`` (gameplay frame ``g = o + h``), the measured v1 fingerprint is::

    maxfreeze = max(motion[g+1], motion[g+2], motion[g+3])
    score = log((baseline + eps) / (maxfreeze + eps))
          + log((motion[g+4] + eps) / (maxfreeze + eps))

Using MAX for the freeze term is load-bearing: a median lets a one-frame shift
hide the +4 launch.  The helper searches integer offsets only on near-CFR
59--61 Hz footage, processes a bounded event sample, and writes diagnostics plus
a masked-gameplay contact sheet.  It never edits a layout or marks review as
complete; even an automatic pass requires a human contact-sheet review before
the offset may enter a layout.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq

from harvest.decode_wild import masked_resize
from harvest.fetch_wild import sha256_file
from harvest.wild_layout import WildLayout


CALIBRATION_VERSION = "madeleine.dash-hitstop-offset.v3"
OFFSET_ONLY_REJECTIONS = {
    "HUD compositor offset is unmeasured",
    "HUD compositor offset confidence below admission threshold",
}
# v3 verdicts.  "pass": every blocking gate holds.  "uncertain_adjacent": the
# winner is decisive by bootstrap and temporal-block unanimity within the ±1
# collar, but the non-adjacent median margin is below the floor; the offset is
# admission-eligible only through a human acceptance that explicitly
# acknowledges the tier and its ±1-frame uncertainty.  "fail": anything else.
VERDICT_PASS = "pass"
VERDICT_UNCERTAIN_ADJACENT = "uncertain_adjacent"
VERDICT_FAIL = "fail"
MARGIN_FAILURE_PREFIX = "non-adjacent median margin"
HANDOFF_INSTRUCTION = (
    "If and only if the verdict is pass or uncertain_adjacent and a human verifies "
    "+1..+3 freeze/+4 rebound across the sheet, accept through harvest.accept_wild_offset "
    "(an uncertain_adjacent verdict additionally requires --accept-uncertain-tier). "
    "The acceptance binds this calibration by SHA-256; never edit a layout offset by hand."
)


@dataclass(frozen=True)
class OffsetPolicy:
    min_lag: int = -12
    max_lag: int = 12
    max_events: int = 256
    min_events: int = 20
    temporal_blocks: int = 3
    min_events_per_block: int = 4
    min_effective_fps: float = 59.0
    max_effective_fps: float = 61.0
    max_vfr_ratio_p99_p01: float = 1.10
    min_local_motion_range: float = 0.50
    min_event_score: float = 3.0
    min_median_margin: float = 2.0
    mode_lag_collar: int = 1
    margin_nonadjacent_gap: int = 2
    bootstrap_samples: int = 2_000
    min_bootstrap_win_fraction: float = 0.95
    bootstrap_seed: int = 20_260_726
    epsilon: float = 0.25
    motion_frame_size: int = 160
    contact_events: int = 12
    contact_frame_size: int = 192

    def lags(self) -> np.ndarray:
        if self.min_lag >= self.max_lag:
            raise ValueError("min_lag must be less than max_lag")
        if self.min_events < 20:
            raise ValueError("min_events may not weaken the 20-event fail-closed gate")
        if not 0 <= self.mode_lag_collar <= 2:
            raise ValueError("mode_lag_collar must be 0, 1, or 2")
        if self.margin_nonadjacent_gap <= self.mode_lag_collar:
            raise ValueError("margin_nonadjacent_gap must exceed mode_lag_collar")
        return np.arange(self.min_lag, self.max_lag + 1, dtype=np.int64)


def fingerprint_scores(
    motions: np.ndarray,
    relative_start: int,
    lags: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Return [events, lags] grounded hitstop scores."""

    if motions.ndim != 2:
        raise ValueError("motions must have shape [events, relative_frame]")
    scores = np.empty((motions.shape[0], lags.size), dtype=np.float64)
    for column, lag in enumerate(lags.tolist()):
        baseline_indices = [lag - 3 - relative_start, lag - 2 - relative_start,
                            lag - 1 - relative_start]
        freeze_indices = [lag + 1 - relative_start, lag + 2 - relative_start,
                          lag + 3 - relative_start]
        rebound_index = lag + 4 - relative_start
        required = baseline_indices + freeze_indices + [rebound_index]
        if min(required) < 0 or max(required) >= motions.shape[1]:
            raise ValueError("motion window does not cover requested lag range")
        baseline = np.median(motions[:, baseline_indices], axis=1)
        maxfreeze = np.max(motions[:, freeze_indices], axis=1)
        rebound = motions[:, rebound_index]
        scores[:, column] = (
            np.log((baseline + epsilon) / (maxfreeze + epsilon))
            + np.log((rebound + epsilon) / (maxfreeze + epsilon))
        )
    return scores


def evaluate_offset_evidence(
    motions: np.ndarray,
    onsets: np.ndarray,
    relative_start: int,
    policy: OffsetPolicy = OffsetPolicy(),
) -> dict[str, Any]:
    """Apply all automatic gates to precomputed per-event motion evidence."""

    lags = policy.lags()
    if onsets.ndim != 1 or onsets.size != motions.shape[0]:
        raise ValueError("one onset is required for every motion row")
    # A true 60 Hz dash contributes only three frozen transitions to this
    # 32-frame window.  A 10th percentile can interpolate wholly outside those
    # three samples (3/32 < 10%), incorrectly classifying a textbook hitstop as
    # static.  Five percent is still robust to a single bad transition while
    # retaining the three-frame physical signal.
    local_range = np.percentile(motions, 95, axis=1) - np.percentile(motions, 5, axis=1)
    raw_scores = fingerprint_scores(motions, relative_start, lags, policy.epsilon)
    event_best_index = np.argmax(raw_scores, axis=1)
    event_best_score = raw_scores[np.arange(raw_scores.shape[0]), event_best_index]
    usable = (
        np.all(np.isfinite(raw_scores), axis=1)
        & (local_range >= policy.min_local_motion_range)
        & (event_best_score >= policy.min_event_score)
    )
    scores = raw_scores[usable]
    usable_onsets = onsets[usable]
    # Blocking gates and the margin gate are tracked separately: a margin-only
    # shortfall with decisive bootstrap and unanimous temporal blocks yields the
    # uncertain_adjacent tier instead of a hard failure.
    hard_failures: list[str] = []
    margin_failures: list[str] = []
    if scores.shape[0] < policy.min_events:
        hard_failures.append(
            f"usable events {scores.shape[0]} < required {policy.min_events}"
        )

    # Diagnostics remain available on failure, but no offset is recommended.
    if scores.shape[0] == 0:
        return {
            "automatic_gates_passed": False,
            "verdict": VERDICT_FAIL,
            "failure_reasons": hard_failures or ["no usable dash-hitstop events"],
            "total_events": int(motions.shape[0]),
            "usable_events": 0,
            "usable_mask": usable,
            "lags": lags,
            "offset_uncertainty_frames": int(policy.mode_lag_collar),
            "score_matrix": scores,
            "candidate_rows": [],
            "best_candidate_offset_frames": None,
            "winning_usable_event_rows": np.asarray([], dtype=np.int64),
        }

    aggregate = np.median(scores, axis=0)
    order = np.argsort(aggregate)[::-1]
    winner_index, runner_index = int(order[0]), int(order[1])
    winner, runner = int(lags[winner_index]), int(lags[runner_index])
    nonadjacent = np.abs(lags - winner) >= policy.margin_nonadjacent_gap
    margin = float(aggregate[winner_index] - float(np.max(aggregate[nonadjacent])))
    if winner_index in (0, lags.size - 1):
        hard_failures.append("winning lag lies on search boundary")
    if margin < policy.min_median_margin:
        margin_failures.append(
            f"{MARGIN_FAILURE_PREFIX} {margin:.4f} < required "
            f"{policy.min_median_margin:.4f}"
        )

    # Per-event mode and collar fractions are recorded as per-event motion SNR
    # indicators, not blocking gates: the 2026-07-28 ground-truth diagnostic
    # (results/wild20/offset-gate-groundtruth-diagnostic/) measured collar
    # fractions of 0.63-0.93 on sessions whose true offset is 0 by
    # construction, so the statistic tracks footage quality, not correctness.
    per_event_lags = lags[np.argmax(scores, axis=1)]
    values, counts = np.unique(per_event_lags, return_counts=True)
    mode_index = int(np.argmax(counts))
    event_mode = int(values[mode_index])
    mode_fraction = float(counts[mode_index] / scores.shape[0])
    collar_fraction = float(
        np.count_nonzero(np.abs(per_event_lags - winner) <= policy.mode_lag_collar)
        / scores.shape[0]
    )

    # Temporal blocks are defined by wall-clock order, not arbitrary row order.
    temporal_order = np.argsort(usable_onsets)
    blocks = np.array_split(temporal_order, policy.temporal_blocks)
    block_rows = []
    for block_number, indices in enumerate(blocks):
        if indices.size < policy.min_events_per_block:
            block_winner = None
            hard_failures.append(
                f"temporal block {block_number} has {indices.size} events; "
                f"need {policy.min_events_per_block}"
            )
        else:
            block_median = np.median(scores[indices], axis=0)
            block_winner = int(lags[int(np.argmax(block_median))])
            if abs(block_winner - winner) > policy.mode_lag_collar:
                hard_failures.append(
                    f"temporal block {block_number} winner {block_winner} outside "
                    f"winner±{policy.mode_lag_collar} of {winner}"
                )
        block_rows.append({
            "block": block_number,
            "events": int(indices.size),
            "first_onset_frame": int(usable_onsets[indices].min()) if indices.size else None,
            "last_onset_frame": int(usable_onsets[indices].max()) if indices.size else None,
            "winner_offset_frames": block_winner,
        })

    rng = np.random.default_rng(policy.bootstrap_seed)
    bootstrap_wins = np.zeros(lags.size, dtype=np.int64)
    for _ in range(policy.bootstrap_samples):
        sample = rng.integers(0, scores.shape[0], size=scores.shape[0])
        chosen = int(np.argmax(np.median(scores[sample], axis=0)))
        bootstrap_wins[chosen] += 1
    collar_indices = np.flatnonzero(np.abs(lags - winner) <= policy.mode_lag_collar)
    bootstrap_fraction = float(
        bootstrap_wins[collar_indices].sum() / policy.bootstrap_samples
    )
    if bootstrap_fraction < policy.min_bootstrap_win_fraction:
        hard_failures.append(
            f"bootstrap winner±{policy.mode_lag_collar} fraction "
            f"{bootstrap_fraction:.3f} < required "
            f"{policy.min_bootstrap_win_fraction:.3f}"
        )

    candidate_rows = [
        {
            "offset_frames": int(lag),
            "median_score": float(aggregate[index]),
            "mean_score": float(np.mean(scores[:, index])),
            "event_wins": int(np.count_nonzero(per_event_lags == lag)),
            "bootstrap_wins": int(bootstrap_wins[index]),
            "bootstrap_fraction": float(bootstrap_wins[index] / policy.bootstrap_samples),
        }
        for index, lag in enumerate(lags.tolist())
    ]
    winning_usable_rows = np.flatnonzero(per_event_lags == winner)
    if not hard_failures and not margin_failures:
        verdict = VERDICT_PASS
    elif not hard_failures:
        # Margin below floor, but the winner is decisive by bootstrap and the
        # temporal blocks are unanimous within the collar: adjacent-lag
        # uncertainty, not a wrong offset.
        verdict = VERDICT_UNCERTAIN_ADJACENT
    else:
        verdict = VERDICT_FAIL
    return {
        "automatic_gates_passed": verdict == VERDICT_PASS,
        "verdict": verdict,
        "failure_reasons": hard_failures + margin_failures,
        "total_events": int(motions.shape[0]),
        "usable_events": int(scores.shape[0]),
        "usable_mask": usable,
        "lags": lags,
        "offset_uncertainty_frames": int(policy.mode_lag_collar),
        "score_matrix": scores,
        "candidate_rows": candidate_rows,
        "best_candidate_offset_frames": winner,
        "runner_up_offset_frames": runner,
        "median_score_margin": margin,
        "per_event_modal_offset_frames": event_mode,
        "per_event_mode_fraction": mode_fraction,
        "per_event_collar_fraction": collar_fraction,
        "bootstrap_win_fraction": bootstrap_fraction,
        "temporal_blocks": block_rows,
        # Row indices into score_matrix / the usable-onset array.  Keeping this
        # coordinate system explicit avoids accidentally applying original
        # motion-row indices to an already-filtered array.
        "winning_usable_event_rows": winning_usable_rows,
    }


def _dash_onsets(table: dict[str, list[Any]]) -> tuple[np.ndarray, np.ndarray]:
    dash = np.asarray(table["dash"], dtype=bool)
    frames = np.asarray(table["video_frame_idx"], dtype=np.int64)
    allowed = np.asarray(table.get("gameplay_allowed", np.ones(dash.size)), dtype=bool)
    onset = dash & ~np.r_[False, dash[:-1]] & allowed
    return frames[onset], np.flatnonzero(onset)


def _event_motion(
    capture: cv2.VideoCapture,
    onset: int,
    relative_start: int,
    relative_end: int,
    layout: WildLayout,
    size: int,
) -> np.ndarray | None:
    first_frame = onset + relative_start - 1
    last_frame = onset + relative_end
    if first_frame < 0:
        return None
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame):
        return None
    ok, frame = capture.read()
    if not ok:
        return None
    previous = cv2.cvtColor(masked_resize(frame, layout, size), cv2.COLOR_BGR2GRAY)
    motion = []
    for _ in range(first_frame + 1, last_frame + 1):
        ok, frame = capture.read()
        if not ok:
            return None
        current = cv2.cvtColor(masked_resize(frame, layout, size), cv2.COLOR_BGR2GRAY)
        motion.append(float(np.mean(cv2.absdiff(current, previous))))
        previous = current
    return np.asarray(motion, dtype=np.float64)


def collect_motion_evidence(
    video: str | Path,
    layout: WildLayout,
    onsets: np.ndarray,
    policy: OffsetPolicy,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Seek only bounded local windows around a deterministic event sample."""

    relative_start = policy.min_lag - 3
    relative_end = policy.max_lag + 4
    if onsets.size > policy.max_events:
        positions = np.linspace(0, onsets.size - 1, policy.max_events, dtype=np.int64)
        onsets = onsets[positions]
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open {video}")
    rows, kept = [], []
    try:
        for onset in onsets.tolist():
            motion = _event_motion(
                capture, int(onset), relative_start, relative_end,
                layout, policy.motion_frame_size,
            )
            if motion is not None:
                rows.append(motion)
                kept.append(onset)
    finally:
        capture.release()
    width = relative_end - relative_start + 1
    return (
        np.stack(rows) if rows else np.empty((0, width), dtype=np.float64),
        np.asarray(kept, dtype=np.int64),
        relative_start,
    )


def _read_contact_frames(
    capture: cv2.VideoCapture,
    game_frame: int,
    layout: WildLayout,
    size: int,
) -> list[np.ndarray] | None:
    start, end = game_frame - 1, game_frame + 4
    if start < 0 or not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
        return None
    frames = []
    for _ in range(start, end + 1):
        ok, frame = capture.read()
        if not ok:
            return None
        frames.append(masked_resize(frame, layout, size))
    return frames


def write_contact_sheet(
    video: str | Path,
    layout: WildLayout,
    onsets: np.ndarray,
    scores: np.ndarray,
    winner: int | None,
    out_path: str | Path,
    policy: OffsetPolicy,
) -> list[dict[str, Any]]:
    """Write masked-gameplay evidence; it is evidence, not human approval."""

    labels = ["g-1", "g", "g+1 freeze", "g+2 freeze", "g+3 freeze", "g+4 rebound"]
    if winner is None or onsets.size == 0:
        sheet = np.full((160, 960, 3), 245, dtype=np.uint8)
        cv2.putText(sheet, "No usable dash-hitstop evidence", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 180), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_path), sheet)
        return []
    order = np.argsort(scores)[::-1][:policy.contact_events]
    capture = cv2.VideoCapture(str(video))
    rendered, evidence = [], []
    try:
        for index in order.tolist():
            onset = int(onsets[index])
            game_frame = onset + int(winner)
            frames = _read_contact_frames(
                capture, game_frame, layout, policy.contact_frame_size
            )
            if frames is None:
                continue
            header = np.full((28, policy.contact_frame_size * len(frames), 3), 250, np.uint8)
            cv2.putText(
                header, f"HUD onset {onset}  candidate g={game_frame}  score={scores[index]:.2f}",
                (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA,
            )
            row_frames = []
            for label, frame in zip(labels, frames, strict=True):
                tile = frame.copy()
                cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1),
                              (0, 180, 0), 1)
                cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0, 0, 0), 1, cv2.LINE_AA)
                row_frames.append(tile)
            rendered.append(np.vstack([header, np.hstack(row_frames)]))
            evidence.append({
                "hud_onset_frame": onset,
                "candidate_game_frame": game_frame,
                "event_score": float(scores[index]),
            })
    finally:
        capture.release()
    if not rendered:
        sheet = np.full((160, 960, 3), 245, dtype=np.uint8)
        cv2.putText(sheet, "Contact frames could not be decoded", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 180), 2, cv2.LINE_AA)
    else:
        sheet = np.vstack(rendered)
    cv2.imwrite(str(out_path), sheet)
    return evidence


def calibrate_offset(
    video_path: str | Path,
    layout_path: str | Path,
    labels_path: str | Path,
    decode_report_path: str | Path,
    out_dir: str | Path,
    *,
    policy: OffsetPolicy = OffsetPolicy(),
) -> dict[str, Any]:
    video = Path(video_path)
    layout_file, labels_file = Path(layout_path), Path(labels_path)
    decode_file = Path(decode_report_path)
    layout = WildLayout.load(layout_file)
    decode = json.loads(decode_file.read_text())
    if decode.get("video_id") != layout.video_id:
        raise ValueError("decode report and layout video IDs differ")
    if not layout.human_reviewed:
        raise ValueError("draft layout must already be human-reviewed")
    if decode.get("layout", {}).get("sha256") != sha256_file(layout_file):
        raise ValueError("layout changed after label decoding")
    if decode.get("source_video", {}).get("sha256") != sha256_file(video):
        raise ValueError("source video hash differs from decode report")
    rejections = set(decode.get("rejection_reasons") or [])
    if rejections - OFFSET_ONLY_REJECTIONS:
        raise ValueError(
            "decode has non-offset admission failures: "
            + ", ".join(sorted(rejections - OFFSET_ONLY_REJECTIONS))
        )
    fps = float(decode["timing"]["pts"]["effective_fps"])
    vfr_ratio = float(decode["timing"]["pts"]["vfr_ratio_p99_p01"])
    if not policy.min_effective_fps <= fps <= policy.max_effective_fps:
        raise ValueError("dash fixed-frame fingerprint requires 59--61 Hz footage")
    if vfr_ratio > policy.max_vfr_ratio_p99_p01:
        raise ValueError("dash fixed-frame fingerprint requires near-CFR footage")

    labels_hash = sha256_file(labels_file)
    if labels_hash == decode.get("raw_labels_sha256"):
        label_kind = "raw_observed_overlay"
        provisional_offset = 0
    elif labels_hash == decode.get("labels_sha256"):
        label_kind = "aligned_labels_inverted_to_observation"
        provisional_offset = int(layout.temporal_offset_frames)
    else:
        raise ValueError("label parquet hash is not named by decode report")
    table = pq.read_table(labels_file).to_pydict()
    onsets, _ = _dash_onsets(table)
    onsets = onsets - provisional_offset
    motions, kept_onsets, relative_start = collect_motion_evidence(
        video, layout, onsets, policy
    )
    evaluation = evaluate_offset_evidence(
        motions, kept_onsets, relative_start, policy
    )

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    contact_path = destination / "dash_offset_contact.png"
    winner = evaluation["best_candidate_offset_frames"]
    winning_rows = evaluation["winning_usable_event_rows"]
    usable_mask = evaluation["usable_mask"]
    usable_onsets = kept_onsets[usable_mask]
    score_matrix = evaluation["score_matrix"]
    if winner is not None and score_matrix.size:
        winner_column = int(np.where(evaluation["lags"] == winner)[0][0])
        contact_onsets = usable_onsets[winning_rows]
        contact_scores = score_matrix[winning_rows, winner_column]
    else:
        contact_onsets = np.asarray([], dtype=np.int64)
        contact_scores = np.asarray([], dtype=np.float64)
    contact_events = write_contact_sheet(
        video, layout, contact_onsets, contact_scores, winner,
        contact_path, policy,
    )

    # Persist the per-event score matrix so a later policy revision can
    # re-verdict this calibration from hash-bound evidence without decoding
    # the source video again.
    score_matrix_path = destination / "score_matrix.npz"
    np.savez_compressed(
        score_matrix_path,
        lags=evaluation["lags"],
        score_matrix=score_matrix,
        usable_onsets=usable_onsets,
    )

    automatic_pass = bool(evaluation["automatic_gates_passed"])
    serializable_candidates = evaluation["candidate_rows"]
    result = {
        "format_version": CALIBRATION_VERSION,
        "video_id": layout.video_id,
        "inputs": {
            "video": str(video),
            "video_sha256": sha256_file(video),
            "layout": str(layout_file),
            "layout_sha256": sha256_file(layout_file),
            "labels": str(labels_file),
            "labels_sha256": labels_hash,
            "label_kind": label_kind,
            "decode_report": str(decode_file),
            "decode_report_sha256": sha256_file(decode_file),
        },
        "offset_semantics": "observed HUD frame o maps to gameplay frame g=o+offset; late HUD is negative",
        "fingerprint": {
            "name": "dash_hitstop_v1",
            "freeze_term": "max motion over gameplay frames g+1..g+3",
            "rebound_frame": "g+4",
            "baseline_frames": "g-3..g-1 median",
        },
        "policy": asdict(policy),
        "events": {
            "dash_onsets_in_labels": int(onsets.size),
            "motion_windows_decoded": int(kept_onsets.size),
            "usable_quality_matches": int(evaluation["usable_events"]),
        },
        "candidates": serializable_candidates,
        "best_candidate_offset_frames": winner,
        "runner_up_offset_frames": evaluation.get("runner_up_offset_frames"),
        "median_score_margin": evaluation.get("median_score_margin"),
        "per_event_modal_offset_frames": evaluation.get("per_event_modal_offset_frames"),
        "per_event_mode_fraction": evaluation.get("per_event_mode_fraction"),
        "per_event_collar_fraction": evaluation.get("per_event_collar_fraction"),
        "bootstrap_win_fraction": evaluation.get("bootstrap_win_fraction"),
        "temporal_blocks": evaluation.get("temporal_blocks", []),
        "offset_uncertainty_frames": evaluation.get(
            "offset_uncertainty_frames", int(policy.mode_lag_collar)
        ),
        "verdict": evaluation["verdict"],
        "automatic_gates_passed": automatic_pass,
        "automatic_failure_reasons": evaluation["failure_reasons"],
        "score_matrix": {
            "path": score_matrix_path.name,
            "sha256": sha256_file(score_matrix_path),
        },
        "human_contact_sheet_review": "pending",
        "calibration_accepted": False,
        "layout_was_modified": False,
        "human_handoff": {
            "contact_sheet": contact_path.name,
            "contact_sheet_sha256": sha256_file(contact_path),
            "events": contact_events,
            "instruction": HANDOFF_INSTRUCTION,
        },
    }
    report_path = destination / "offset_calibration.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    report_hash = sha256_file(report_path)
    (destination / "offset_calibration.sha256").write_text(
        f"{report_hash}  {report_path.name}\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--decode-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-lag", type=int, default=-12)
    parser.add_argument("--max-lag", type=int, default=12)
    parser.add_argument("--max-events", type=int, default=256)
    args = parser.parse_args()
    policy = OffsetPolicy(
        min_lag=args.min_lag, max_lag=args.max_lag, max_events=args.max_events
    )
    report = calibrate_offset(
        args.video, args.layout, args.labels, args.decode_report, args.out,
        policy=policy,
    )
    print(json.dumps({
        "video_id": report["video_id"],
        "best_candidate_offset_frames": report["best_candidate_offset_frames"],
        "verdict": report["verdict"],
        "automatic_gates_passed": report["automatic_gates_passed"],
        "human_contact_sheet_review": report["human_contact_sheet_review"],
        "calibration_accepted": report["calibration_accepted"],
    }, indent=2))


if __name__ == "__main__":
    main()
