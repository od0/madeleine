"""Map NitroGen controller rows to MADELEINE's seven-key native-grid labels.

Every output row corresponds one-for-one with an ``actions_raw.parquet`` row.
The mapper never changes a chunk's rate or row count; it declares the measured
``grid_hz`` through :func:`data.schema.labels_native_schema`.
The resulting noisy labels are training fuel and an audit target, never
evaluation ground truth.

Bind inference is performed independently per video.  Each candidate button is
described by its pressed-frame rate, press count, presses per hour, median run
duration, fraction of runs no longer than 8/60 second, directional overlap,
and upward-direction overlap.  A button is eligible for an action only when it
clears that action's absolute press floor, presses-per-hour floor, and a
coarse shape gate (dash demands a directional short-to-held profile, grab a
held profile, jump a non-held profile); the floors exist because the
2026-08-02 mapping audit showed a three-press phantom button outscoring a
real dash button in 13 unflagged videos once shape terms saturate on tiny
samples.  Because Celeste's default layout is itself multi-bound, candidate
units include sibling composites (south+north jump, west+east dash) and a
shoulder/trigger grab group whenever every member is eligible and minor
members carry a meaningful press fraction; a joint assignment then selects
one unit per action with pairwise-disjoint buttons, maximizing mean unit
score minus an orphan-coverage penalty so an assignment cannot win by
leaving a heavily used button unexplained.  Jump rewards frequent
short-to-medium runs, dash rewards frequent short-to-held directional runs
(full credit through 12-frame medians, tapering to the hold-tolerant bound),
and grab rewards long upward-direction runs.  Celeste's default binds provide
only a small tie-breaking prior.

Confidence, flagging, and fallback are per action.  Each selected action's
confidence is ``support * (0.65 * score + 0.35 * margin)`` in ``[0, 1]``,
where ``support`` saturates at twice the action's absolute floor and
``margin`` is the gap to the best unassigned eligible alternative for that
action scaled by ``InferenceParameters.assignment_margin_scale``.  An action
with no eligible candidate, or with confidence below the flag threshold, is
flagged and falls back to its own entry in the default bind prior with any
buttons inferred for other actions removed; a fully-conflicted prior yields
an explicitly empty selection whose label column is all-negative rather than
a duplicate of another action's column.  Unflagged actions keep their
inferred button set.  The report's top-level ``confidence`` is the minimum
per-action confidence and ``flagged`` is true when any action is flagged, so
consumers that gate on the video level remain conservative.

NitroGen's published coordinate contract defines the upper-left corner as
``(-1, -1)``.  Therefore left-stick Y is mapped directly and invariantly:
values below ``-axis_threshold`` mean up and values above ``axis_threshold``
mean down.  D-pad state is direct and never used to infer or alter this rule.
Stick arrays are explicitly converted element-by-element to float, covering
both DOUBLE[] and BIGINT[] source columns.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.schema import KEY_ORDER, labels_native_schema


DEFAULT_AXIS_THRESHOLD = 0.5
FLAG_THRESHOLD = 0.5
TOOL_VERSION = "nitrogen.map_actions/3.1"
REPORT_SCHEMA_VERSION = "madeleine.nitrogen.mapping-report.v3"
NITROGEN_COORDINATE_CONTRACT_REVISION = (
    "b171bc8ed2e3c311e9305ebb993c56ef565ab509"
)

BUTTON_COLUMNS = (
    "back",
    "dpad_down",
    "dpad_left",
    "dpad_right",
    "dpad_up",
    "east",
    "guide",
    "left_shoulder",
    "left_thumb",
    "left_trigger",
    "north",
    "right_shoulder",
    "right_thumb",
    "right_trigger",
    "south",
    "start",
    "west",
)
CANDIDATE_BUTTONS = (
    "south",
    "east",
    "west",
    "north",
    "left_shoulder",
    "right_shoulder",
    "left_trigger",
    "right_trigger",
)
# Celeste's default layout is itself multi-bound: jump on A and Y, dash on
# X and B, grab on both triggers and both bumpers.  The priors mirror that.
PRIOR_BIND_MAP = {
    "jump": ["south", "north"],
    "dash": ["west", "east"],
    "grab": [
        "left_trigger",
        "right_trigger",
        "left_shoulder",
        "right_shoulder",
    ],
}
SIBLING_COMPOSITES = {"jump": ("south", "north"), "dash": ("west", "east")}
GRAB_GROUP = ("left_trigger", "right_trigger", "left_shoulder", "right_shoulder")
ACTION_NAMES = ("jump", "dash", "grab")


@dataclass(frozen=True)
class InferenceParameters:
    """Thresholds and score-shape values intended for orchestration tuning."""

    min_presses: int = 3
    jump_target_frames_60hz_equivalent: float = 8.0
    dash_full_credit_frames_60hz_equivalent: float = 12.0
    dash_hold_tolerant_frames_60hz_equivalent: float = 24.0
    grab_long_frames_60hz_equivalent: float = 18.0
    assignment_margin_scale: float = 0.15
    prior_bonus: float = 0.08
    min_action_presses: Mapping[str, int] = field(
        default_factory=lambda: {"jump": 20, "dash": 15, "grab": 8}
    )
    min_action_presses_per_hour: Mapping[str, float] = field(
        default_factory=lambda: {"jump": 150.0, "dash": 60.0, "grab": 15.0}
    )
    dash_min_direction_co_press: float = 0.3
    grab_min_median_frames_60hz_equivalent: float = 10.0
    jump_max_median_frames_60hz_equivalent: float = 40.0
    secondary_min_presses: int = 10
    secondary_min_fraction_of_primary: float = 0.05
    orphan_coverage_weight: float = 0.5


DEFAULT_INFERENCE_PARAMETERS = InferenceParameters()


@dataclass
class RawChunk:
    buttons: dict[str, np.ndarray]
    j_left: np.ndarray
    j_right: np.ndarray

    @property
    def row_count(self) -> int:
        return len(self.j_left)


@dataclass
class _ButtonAccumulator:
    active_frames: int = 0
    run_lengths_native: list[int] = field(default_factory=list)
    run_durations_seconds: list[float] = field(default_factory=list)
    direction_overlap: int = 0
    up_overlap: int = 0


@dataclass
class _EvidenceAccumulator:
    total_frames: int = 0
    total_seconds: float = 0.0
    buttons: dict[str, _ButtonAccumulator] = field(
        default_factory=lambda: {
            button: _ButtonAccumulator() for button in CANDIDATE_BUTTONS
        }
    )
    def update(
        self,
        raw: RawChunk,
        grid_hz: float,
        axis_threshold: float,
    ) -> None:
        if grid_hz <= 0:
            raise ValueError("grid_hz must be positive")
        self.total_frames += raw.row_count
        self.total_seconds += raw.row_count / grid_hz
        directions = _directional_states(raw, axis_threshold)
        any_direction = np.logical_or.reduce(list(directions.values()))

        for button, accumulator in self.buttons.items():
            pressed = raw.buttons[button]
            accumulator.active_frames += int(np.count_nonzero(pressed))
            starts, ends = _true_runs(pressed)
            run_lengths = (ends - starts).astype(np.int64)
            accumulator.run_lengths_native.extend(int(length) for length in run_lengths)
            accumulator.run_durations_seconds.extend(
                float(length) / grid_hz for length in run_lengths
            )
            accumulator.direction_overlap += int(
                np.count_nonzero(pressed & any_direction)
            )
            accumulator.up_overlap += int(
                np.count_nonzero(pressed & directions["up"])
            )


def _true_runs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate(
        (
            np.array([False], dtype=np.bool_),
            values.astype(np.bool_, copy=False),
            np.array([False], dtype=np.bool_),
        )
    )
    transitions = np.diff(padded.astype(np.int8, copy=False))
    return np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)


def _float_sticks(column: pa.ChunkedArray, name: str) -> np.ndarray:
    converted: list[tuple[float, float]] = []
    for row_index, pair in enumerate(column.to_pylist()):
        if pair is None or len(pair) != 2:
            raise ValueError(f"{name}[{row_index}] is not a length-two stick array")
        converted.append((float(pair[0]), float(pair[1])))
    return np.asarray(converted, dtype=np.float64).reshape((-1, 2))


def read_raw_chunk(path: str | Path) -> RawChunk:
    """Read the real 17-button/two-stick schema, casting both sticks to float."""

    table = pq.read_table(
        path,
        columns=[*BUTTON_COLUMNS, "j_left", "j_right"],
    )
    buttons = {
        button: np.asarray(table[button].to_pylist(), dtype=np.int64).astype(
            np.bool_, copy=False
        )
        for button in BUTTON_COLUMNS
    }
    j_left = _float_sticks(table["j_left"], "j_left")
    j_right = _float_sticks(table["j_right"], "j_right")
    lengths = {len(values) for values in [*buttons.values(), j_left, j_right]}
    if len(lengths) != 1:
        raise ValueError(f"columns have inconsistent row counts in {path}")
    return RawChunk(buttons=buttons, j_left=j_left, j_right=j_right)


def _directional_states(
    raw: RawChunk,
    axis_threshold: float,
) -> dict[str, np.ndarray]:
    x_axis = raw.j_left[:, 0]
    y_axis = raw.j_left[:, 1]
    return {
        "left": raw.buttons["dpad_left"] | (x_axis < -axis_threshold),
        "right": raw.buttons["dpad_right"] | (x_axis > axis_threshold),
        "up": raw.buttons["dpad_up"] | (y_axis < -axis_threshold),
        "down": raw.buttons["dpad_down"] | (y_axis > axis_threshold),
    }


def _evidence_stats(
    accumulator: _EvidenceAccumulator,
) -> dict[str, dict[str, int | float | str]]:
    stats: dict[str, dict[str, int | float | str]] = {}
    hours = accumulator.total_seconds / 3600.0
    for button, values in accumulator.buttons.items():
        press_count = len(values.run_durations_seconds)
        if press_count:
            median_native = float(np.median(values.run_lengths_native))
            median_seconds = float(np.median(values.run_durations_seconds))
            burst_fraction = sum(
                duration <= 8.0 / 60.0 for duration in values.run_durations_seconds
            ) / press_count
        else:
            median_native = 0.0
            median_seconds = 0.0
            burst_fraction = 0.0
        direction_overlap = values.direction_overlap
        up_overlap = values.up_overlap
        active_frames = values.active_frames
        stats[button] = {
            "press_rate": (
                active_frames / accumulator.total_frames
                if accumulator.total_frames
                else 0.0
            ),
            "press_count": press_count,
            "presses_per_hour": press_count / hours if hours > 0 else 0.0,
            "median_press_duration_frames": median_native,
            "median_press_duration_seconds": median_seconds,
            "median_press_duration_frames_60hz_equivalent": median_seconds * 60.0,
            "burst_fraction_le_8_frames_60hz_equivalent": burst_fraction,
            "direction_co_press_rate": (
                direction_overlap / active_frames if active_frames else 0.0
            ),
            "up_co_press_rate": up_overlap / active_frames if active_frames else 0.0,
            "duration_note": (
                "native-frame median plus time-equivalent cross-grid statistic; "
                "no label rows converted"
            ),
        }
    return stats


def _prior_contains(action: str, button: str) -> bool:
    return button in PRIOR_BIND_MAP[action]


def _candidate_scores(
    evidence: Mapping[str, Mapping[str, int | float | str]],
    parameters: InferenceParameters,
) -> dict[str, dict[str, float]]:
    max_press_count = max(
        (int(evidence[button]["press_count"]) for button in CANDIDATE_BUTTONS),
        default=0,
    )
    scores = {action: {} for action in ACTION_NAMES}
    for button in CANDIDATE_BUTTONS:
        button_stats = evidence[button]
        press_count = int(button_stats["press_count"])
        frequency = press_count / max_press_count if max_press_count else 0.0
        support = min(1.0, press_count / max(parameters.min_presses, 1))
        duration_frames = float(
            button_stats["median_press_duration_frames_60hz_equivalent"]
        )
        burst = float(button_stats["burst_fraction_le_8_frames_60hz_equivalent"])
        direction = float(button_stats["direction_co_press_rate"])
        upward = float(button_stats["up_co_press_rate"])

        if duration_frames > 0:
            jump_fit = math.exp(
                -abs(
                    math.log(
                        duration_frames
                        / parameters.jump_target_frames_60hz_equivalent
                    )
                )
            )
            full_credit = parameters.dash_full_credit_frames_60hz_equivalent
            hold_tolerant = parameters.dash_hold_tolerant_frames_60hz_equivalent
            if duration_frames <= full_credit:
                dash_fit = 1.0
            elif duration_frames <= hold_tolerant:
                dash_fit = 1.0 - 0.6 * (duration_frames - full_credit) / max(
                    hold_tolerant - full_credit, 1e-9
                )
            else:
                dash_fit = 0.4 * math.exp(-(duration_frames - hold_tolerant) / 6.0)
            grab_fit = min(
                1.0,
                duration_frames / parameters.grab_long_frames_60hz_equivalent,
            )
        else:
            jump_fit = dash_fit = grab_fit = 0.0

        raw_scores = {
            "jump": 0.50 * frequency + 0.40 * jump_fit + 0.10 * burst,
            "dash": (
                0.25 * frequency
                + 0.30 * dash_fit
                + 0.40 * direction
                + 0.05 * burst
            ),
            "grab": (
                0.15 * frequency
                + 0.45 * grab_fit
                + 0.30 * upward
                + 0.10 * (1.0 - burst)
            ),
        }
        for action, raw_score in raw_scores.items():
            prior = parameters.prior_bonus if _prior_contains(action, button) else 0.0
            scores[action][button] = min(1.0, support * raw_score + prior)
    return scores


def _action_shape_admits(
    action: str,
    stats: Mapping[str, int | float | str],
    parameters: InferenceParameters,
) -> bool:
    median = float(stats["median_press_duration_frames_60hz_equivalent"])
    if action == "dash":
        return (
            median <= parameters.dash_hold_tolerant_frames_60hz_equivalent
            and float(stats["direction_co_press_rate"])
            >= parameters.dash_min_direction_co_press
        )
    if action == "grab":
        return median >= parameters.grab_min_median_frames_60hz_equivalent
    if action == "jump":
        return median <= parameters.jump_max_median_frames_60hz_equivalent
    return True


def _eligible_candidates(
    evidence: Mapping[str, Mapping[str, int | float | str]],
    parameters: InferenceParameters,
) -> dict[str, list[str]]:
    eligible: dict[str, list[str]] = {}
    for action in ACTION_NAMES:
        min_abs = int(parameters.min_action_presses.get(action, 0))
        min_rate = float(parameters.min_action_presses_per_hour.get(action, 0.0))
        eligible[action] = [
            button
            for button in CANDIDATE_BUTTONS
            if int(evidence[button]["press_count"]) >= min_abs
            and float(evidence[button]["presses_per_hour"]) >= min_rate
            and _action_shape_admits(action, evidence[button], parameters)
        ]
    return eligible


def _stats_scores(
    stats: Mapping[str, int | float | str],
    parameters: InferenceParameters,
    max_press_count: int,
) -> dict[str, float]:
    """Support-scaled raw action scores for one button or composite unit."""

    press_count = int(stats["press_count"])
    frequency = min(1.0, press_count / max_press_count) if max_press_count else 0.0
    support = min(1.0, press_count / max(parameters.min_presses, 1))
    duration_frames = float(stats["median_press_duration_frames_60hz_equivalent"])
    burst = float(stats["burst_fraction_le_8_frames_60hz_equivalent"])
    direction = float(stats["direction_co_press_rate"])
    upward = float(stats["up_co_press_rate"])

    if duration_frames > 0:
        jump_fit = math.exp(
            -abs(
                math.log(
                    duration_frames / parameters.jump_target_frames_60hz_equivalent
                )
            )
        )
        full_credit = parameters.dash_full_credit_frames_60hz_equivalent
        hold_tolerant = parameters.dash_hold_tolerant_frames_60hz_equivalent
        if duration_frames <= full_credit:
            dash_fit = 1.0
        elif duration_frames <= hold_tolerant:
            dash_fit = 1.0 - 0.6 * (duration_frames - full_credit) / max(
                hold_tolerant - full_credit, 1e-9
            )
        else:
            dash_fit = 0.4 * math.exp(-(duration_frames - hold_tolerant) / 6.0)
        grab_fit = min(
            1.0, duration_frames / parameters.grab_long_frames_60hz_equivalent
        )
    else:
        jump_fit = dash_fit = grab_fit = 0.0

    raw_scores = {
        "jump": 0.50 * frequency + 0.40 * jump_fit + 0.10 * burst,
        "dash": 0.25 * frequency + 0.30 * dash_fit + 0.40 * direction + 0.05 * burst,
        "grab": (
            0.15 * frequency
            + 0.45 * grab_fit
            + 0.30 * upward
            + 0.10 * (1.0 - burst)
        ),
    }
    return {action: support * raw for action, raw in raw_scores.items()}


def _composite_stats(
    evidence: Mapping[str, Mapping[str, int | float | str]],
    members: Sequence[str],
) -> dict[str, int | float]:
    counts = {b: int(evidence[b]["press_count"]) for b in members}
    total = sum(counts.values())
    major = max(members, key=lambda b: counts[b])

    def weighted(key: str) -> float:
        return sum(float(evidence[b][key]) * counts[b] for b in members) / total

    return {
        "press_count": total,
        "presses_per_hour": sum(
            float(evidence[b]["presses_per_hour"]) for b in members
        ),
        "median_press_duration_frames_60hz_equivalent": float(
            evidence[major]["median_press_duration_frames_60hz_equivalent"]
        ),
        "burst_fraction_le_8_frames_60hz_equivalent": weighted(
            "burst_fraction_le_8_frames_60hz_equivalent"
        ),
        "direction_co_press_rate": weighted("direction_co_press_rate"),
        "up_co_press_rate": weighted("up_co_press_rate"),
    }


def _candidate_units(
    evidence: Mapping[str, Mapping[str, int | float | str]],
    eligible: Mapping[str, Sequence[str]],
    scores: Mapping[str, Mapping[str, float]],
    parameters: InferenceParameters,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per-action candidate units: single buttons plus multi-bind composites.

    Composites model Celeste's multi-bound defaults: the sibling face pairs
    (south+north jump, west+east dash) and the shoulder/trigger grab group.
    A composite exists only when every member is eligible for the action and
    each minor member carries a meaningful fraction of the major's presses,
    so a Talk button or a phantom cannot ride along.
    """

    max_press_count = max(
        (int(evidence[button]["press_count"]) for button in CANDIDATE_BUTTONS),
        default=0,
    )

    def unit_from(members: Sequence[str], action: str) -> dict[str, Any]:
        members = tuple(sorted(members))
        if len(members) == 1:
            score = float(scores[action][members[0]])
            press_count = int(evidence[members[0]]["press_count"])
        else:
            stats = _composite_stats(evidence, members)
            base = _stats_scores(stats, parameters, max_press_count)[action]
            prior = (
                parameters.prior_bonus
                if any(_prior_contains(action, b) for b in members)
                else 0.0
            )
            score = min(1.0, base + prior)
            press_count = int(stats["press_count"])
        return {
            "buttons": members,
            "score": score,
            "press_count": press_count,
            "composite": len(members) > 1,
        }

    def fraction_ok(members: Sequence[str]) -> bool:
        counts = sorted(int(evidence[b]["press_count"]) for b in members)
        floor = max(
            parameters.secondary_min_presses,
            parameters.secondary_min_fraction_of_primary * counts[-1],
        )
        return all(count >= floor for count in counts[:-1])

    units: dict[str, dict[str, dict[str, Any]]] = {}
    for action in ACTION_NAMES:
        pool: dict[str, dict[str, Any]] = {}
        for button in eligible[action]:
            pool[button] = unit_from((button,), action)
        pair = SIBLING_COMPOSITES.get(action)
        if pair and all(b in eligible[action] for b in pair) and fraction_ok(pair):
            unit = unit_from(pair, action)
            pool["+".join(unit["buttons"])] = unit
        if action == "grab":
            members = [b for b in GRAB_GROUP if b in eligible["grab"]]
            if len(members) >= 2:
                major = max(int(evidence[b]["press_count"]) for b in members)
                floor = max(
                    parameters.secondary_min_presses,
                    parameters.secondary_min_fraction_of_primary * major,
                )
                kept = [
                    b for b in members if int(evidence[b]["press_count"]) >= floor
                ]
                if len(kept) >= 2:
                    unit = unit_from(kept, "grab")
                    pool["+".join(unit["buttons"])] = unit
        units[action] = pool
    return units


def _best_unit_assignment(
    units: Mapping[str, Mapping[str, dict[str, Any]]],
    included: Sequence[str],
    evidence: Mapping[str, Mapping[str, int | float | str]],
    parameters: InferenceParameters,
) -> dict[str, str] | None:
    """Best one-unit-per-action assignment with disjoint buttons.

    The objective is the mean unit score minus an orphan-coverage penalty:
    the press-mass fraction of candidate buttons left unexplained by the
    assignment.  Real layouts explain the heavily used buttons, and without
    this term a frequent dash button can be misrouted to jump while the true
    jump button is orphaned entirely.
    """

    total_presses = sum(
        int(evidence[button]["press_count"]) for button in CANDIDATE_BUTTONS
    )
    best: tuple[float, tuple[str, ...]] | None = None

    def search(
        index: int,
        used: frozenset,
        total: float,
        covered: int,
        chosen: tuple[str, ...],
    ):
        nonlocal best
        if index == len(included):
            orphan_fraction = (
                1.0 - covered / total_presses if total_presses else 0.0
            )
            objective = (
                total / len(included)
                - parameters.orphan_coverage_weight * orphan_fraction
            )
            key = (objective, chosen)
            if best is None or (-key[0], key[1]) < (-best[0], best[1]):
                best = key
            return
        action = included[index]
        for unit_id in sorted(units[action]):
            unit = units[action][unit_id]
            if used & frozenset(unit["buttons"]):
                continue
            search(
                index + 1,
                used | frozenset(unit["buttons"]),
                total + unit["score"],
                covered + unit["press_count"],
                chosen + (unit_id,),
            )

    search(0, frozenset(), 0.0, 0, ())
    if best is None:
        return None
    return dict(zip(included, best[1]))


def infer_bind_map(
    evidence: Mapping[str, Mapping[str, int | float | str]],
    *,
    flag_threshold: float = FLAG_THRESHOLD,
    parameters: InferenceParameters = DEFAULT_INFERENCE_PARAMETERS,
) -> tuple[
    dict[str, list[str]],
    float,
    bool,
    dict[str, dict[str, float]],
    dict[str, dict[str, Any]],
]:
    """Infer per-action button sets with per-action confidence and fallback."""

    if not 0.0 <= flag_threshold <= 1.0:
        raise ValueError("flag_threshold must be in [0, 1]")
    scores = _candidate_scores(evidence, parameters)
    eligible = _eligible_candidates(evidence, parameters)
    units = _candidate_units(evidence, eligible, scores, parameters)

    included = [action for action in ACTION_NAMES if units[action]]
    assignment: dict[str, str] = {}
    while included:
        result = _best_unit_assignment(units, included, evidence, parameters)
        if result is not None:
            assignment = result
            break
        # A shared sole candidate can make a disjoint assignment impossible;
        # drop the action whose best unit score is weakest and retry.
        weakest = min(
            included,
            key=lambda action: (
                max(unit["score"] for unit in units[action].values()),
                action,
            ),
        )
        included = [action for action in included if action != weakest]

    # Selections are exactly the assigned units' button sets; multi-bind
    # arrives through the sibling and grab-group composites, so a Talk or
    # menu button can never be appended to another action's selection.
    selections: dict[str, list[str]] = {}
    secondaries: dict[str, list[str]] = {}
    for action, unit_id in assignment.items():
        unit = units[action][unit_id]
        selections[action] = sorted(unit["buttons"])
        major = max(
            unit["buttons"], key=lambda b: int(evidence[b]["press_count"])
        )
        secondaries[action] = sorted(b for b in unit["buttons"] if b != major)

    bind_map: dict[str, list[str]] = {}
    per_action: dict[str, dict[str, Any]] = {}
    confidences: list[float] = []
    any_flagged = False
    for action in ACTION_NAMES:
        min_abs = int(parameters.min_action_presses.get(action, 0))
        detail: dict[str, Any] = {
            "eligible_candidates": list(eligible[action]),
            "floors": {
                "min_presses": min_abs,
                "min_presses_per_hour": float(
                    parameters.min_action_presses_per_hour.get(action, 0.0)
                ),
            },
        }
        unit_id = assignment.get(action)
        if unit_id is None:
            detail.update(
                {
                    "inferred_button": None,
                    "inferred_buttons": [],
                    "confidence": 0.0,
                    "flagged": True,
                    "fallback_used": True,
                    "reason": (
                        "no_eligible_candidate"
                        if not units[action]
                        else "assignment_conflict"
                    ),
                }
            )
            flagged_action = True
            confidence_action = 0.0
        else:
            unit = units[action][unit_id]
            selected = selections[action]
            score = float(unit["score"])
            other_buttons = {
                b
                for other, sel in selections.items()
                if other != action
                for b in sel
            }
            alternatives = [
                float(candidate["score"])
                for candidate_id, candidate in units[action].items()
                if candidate_id != unit_id
                and not (set(candidate["buttons"]) & other_buttons)
                and not (set(candidate["buttons"]) & set(selected))
            ]
            margin_raw = score - max(alternatives) if alternatives else float("inf")
            margin = min(
                1.0,
                max(0.0, margin_raw)
                / max(parameters.assignment_margin_scale, 1e-9),
            )
            total_presses = sum(
                int(evidence[b]["press_count"]) for b in selected
            )
            support = min(1.0, total_presses / max(1, 2 * min_abs))
            confidence_action = min(
                1.0, max(0.0, support * (0.65 * score + 0.35 * margin))
            )
            flagged_action = confidence_action < flag_threshold
            major = max(
                unit["buttons"],
                key=lambda b: int(evidence[b]["press_count"]),
            )
            detail.update(
                {
                    "inferred_button": major,
                    "inferred_buttons": list(selected),
                    "composite": bool(unit["composite"]),
                    "secondary_buttons": list(secondaries[action]),
                    "score": score,
                    "margin": margin,
                    "support": support,
                    "confidence": confidence_action,
                    "flagged": flagged_action,
                    "fallback_used": flagged_action,
                }
            )
        per_action[action] = detail
        confidences.append(confidence_action)
        any_flagged = any_flagged or flagged_action

    # Resolve fallback selections with knowledge of the inferred buttons so a
    # prior fallback can never duplicate another action's inferred column.  A
    # fully-conflicted prior yields an explicitly empty selection: the action
    # maps to an all-negative column instead of copying another action's.
    inferred_buttons = {
        button
        for action, detail in per_action.items()
        if not detail["fallback_used"]
        for button in detail["inferred_buttons"]
    }
    for action, detail in per_action.items():
        if not detail["fallback_used"]:
            bind_map[action] = list(detail["inferred_buttons"])
        else:
            reduced = [
                candidate
                for candidate in PRIOR_BIND_MAP[action]
                if candidate not in inferred_buttons
            ]
            removed = [
                candidate
                for candidate in PRIOR_BIND_MAP[action]
                if candidate in inferred_buttons
            ]
            if removed:
                detail["prior_reduced_by_inferred"] = removed
            if not reduced:
                detail["prior_conflicts_with_inferred"] = True
                detail.setdefault("reason", "prior_conflicts_with_inferred")
            bind_map[action] = reduced
        detail["selected"] = list(bind_map[action])

    confidence = min(confidences) if confidences else 0.0
    return bind_map, confidence, any_flagged, scores, per_action


def load_bind_resolution(
    path: str | Path,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, str]], dict[str, Any]]:
    """Load a bind-resolution record as per-video overrides.

    Returns (overrides, review_status, meta): overrides maps video id to
    action button lists (possibly empty, meaning an all-negative column),
    review_status carries each entry's review state for the report's
    bind_source block, and meta identifies the resolution artifact.
    """

    path = Path(path)
    encoded = path.read_bytes()
    record = json.loads(encoded)
    if record.get("schema_version") != "madeleine.nitrogen-bind-resolution.v1":
        raise ValueError(f"{path}: unexpected bind-resolution schema")
    overrides: dict[str, dict[str, list[str]]] = {}
    review: dict[str, dict[str, str]] = {}
    for entry in record["resolution"]:
        video_id = str(entry["video_id"])
        action = str(entry["action"])
        if action not in ACTION_NAMES:
            raise ValueError(f"{path}: unknown action {action!r}")
        buttons = [str(b) for b in entry["resolved"]]
        bad = [b for b in buttons if b not in BUTTON_COLUMNS]
        if bad:
            raise ValueError(f"{path}: {video_id}/{action} has unknown buttons {bad}")
        overrides.setdefault(video_id, {})[action] = buttons
        review.setdefault(video_id, {})[action] = str(entry["review_status"])
    for video_id, actions in overrides.items():
        if set(actions) != set(ACTION_NAMES):
            raise ValueError(f"{path}: {video_id} does not resolve all actions")
    meta = {
        "kind": "resolved",
        "resolution_schema_version": record["schema_version"],
        "resolution_sha256": hashlib.sha256(encoded).hexdigest(),
        "resolution_created_at": record.get("created_at"),
    }
    return overrides, review, meta


def _discover_chunk_paths(
    actions_root: Path,
    selected_video_ids: set[str],
) -> dict[tuple[str, int], Path]:
    paths: dict[tuple[str, int], Path] = {}
    for shard_dir in sorted(path for path in actions_root.glob("SHARD_*") if path.is_dir()):
        for video_dir in sorted(path for path in shard_dir.iterdir() if path.is_dir()):
            video_id = video_dir.name
            if video_id not in selected_video_ids:
                continue
            prefix = f"{video_id}_chunk_"
            with os.scandir(video_dir) as entries:
                names = sorted(
                    entry.name
                    for entry in entries
                    if entry.is_dir() and entry.name.startswith(prefix)
                )
            for name in names:
                try:
                    chunk_id = int(name.removeprefix(prefix))
                except ValueError:
                    continue
                key = (video_id, chunk_id)
                if key in paths:
                    raise ValueError(f"duplicate chunk path for {video_id} chunk {chunk_id}")
                paths[key] = video_dir / name
    return paths


def _validate_index_row(row: Mapping[str, Any]) -> None:
    required = {"video_id", "chunk_id", "chunk_size", "grid_hz", "controller_type"}
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"chunk index is missing columns: {', '.join(missing)}")
    if int(row["chunk_size"]) <= 0 or float(row["grid_hz"]) <= 0:
        raise ValueError("chunk_size and grid_hz must be positive")


def _map_chunk(
    raw: RawChunk,
    *,
    bind_map: Mapping[str, Sequence[str]],
    axis_threshold: float,
) -> dict[str, np.ndarray]:
    directions = _directional_states(raw, axis_threshold)
    mapped = dict(directions)
    for action in ACTION_NAMES:
        selected = [raw.buttons[button] for button in bind_map[action]]
        if selected:
            mapped[action] = np.logical_or.reduce(selected)
        else:
            mapped[action] = np.zeros(raw.row_count, dtype=np.bool_)
    return mapped


def _write_labels(
    destination: Path,
    mapped: Mapping[str, np.ndarray],
    row_count: int,
    grid_hz: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    schema = labels_native_schema(grid_hz)
    arrays = [pa.array(np.arange(row_count, dtype=np.int64), type=pa.int64())]
    arrays.extend(pa.array(mapped[key], type=pa.bool_()) for key in KEY_ORDER)
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), destination)


def _video_controller_type(rows: Sequence[Mapping[str, Any]]) -> str:
    controller_types = sorted({str(row["controller_type"]) for row in rows})
    return controller_types[0] if len(controller_types) == 1 else "mixed:" + ",".join(
        controller_types
    )


def map_actions(
    chunk_index: str | Path,
    actions_root: str | Path,
    out: str | Path,
    *,
    videos: Sequence[str] | None = None,
    axis_threshold: float = DEFAULT_AXIS_THRESHOLD,
    flag_threshold: float = FLAG_THRESHOLD,
    inference_parameters: InferenceParameters = DEFAULT_INFERENCE_PARAMETERS,
    bind_overrides: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    bind_review_status: Mapping[str, Mapping[str, str]] | None = None,
    bind_resolution_meta: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Map selected videos and return their JSON-serializable reports.

    When ``bind_overrides`` supplies a video's resolved button sets, the
    labels are built from those sets instead of the inferred assignment;
    the report keeps the inference output for transparency under an
    ``inference`` block and records the resolution provenance under
    ``bind_source``. Every overridden video must resolve all three
    actions, and an empty list yields an all-negative action column.
    """

    if axis_threshold < 0:
        raise ValueError("axis_threshold must be non-negative")
    if bind_overrides and bind_resolution_meta is None:
        raise ValueError("bind_overrides require bind_resolution_meta")

    index_rows = pq.read_table(chunk_index).to_pylist()
    for row in index_rows:
        _validate_index_row(row)
    selected = set(videos) if videos is not None else None
    if selected is not None:
        index_rows = [row for row in index_rows if row["video_id"] in selected]

    rows_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index_rows:
        rows_by_video[str(row["video_id"])].append(row)
    for rows in rows_by_video.values():
        rows.sort(key=lambda row: int(row["chunk_id"]))

    path_by_chunk = _discover_chunk_paths(Path(actions_root), set(rows_by_video))
    destination_root = Path(out)
    reports: list[dict[str, Any]] = []

    for video_id in sorted(rows_by_video):
        rows = rows_by_video[video_id]
        accumulator = _EvidenceAccumulator()
        readable: dict[int, Path] = {}
        skipped_details: list[dict[str, Any]] = []

        for row in rows:
            chunk_id = int(row["chunk_id"])
            chunk_dir = path_by_chunk.get((video_id, chunk_id))
            raw_path = chunk_dir / "actions_raw.parquet" if chunk_dir else None
            if raw_path is None or not raw_path.is_file():
                skipped_details.append(
                    {"chunk_id": chunk_id, "error": "missing actions_raw.parquet"}
                )
                continue
            try:
                raw = read_raw_chunk(raw_path)
                if raw.row_count != int(row["chunk_size"]):
                    raise ValueError(
                        f"row count {raw.row_count} != chunk_size {row['chunk_size']}"
                    )
                accumulator.update(raw, float(row["grid_hz"]), axis_threshold)
                readable[chunk_id] = chunk_dir
            except Exception as error:
                skipped_details.append(
                    {
                        "chunk_id": chunk_id,
                        "error": str(error) or type(error).__name__,
                    }
                )

        evidence = _evidence_stats(accumulator)
        bind_map, confidence, flagged, candidate_scores, per_action = infer_bind_map(
            evidence,
            flag_threshold=flag_threshold,
            parameters=inference_parameters,
        )

        bind_source: dict[str, Any] | None = None
        inference_block: dict[str, Any] | None = None
        override = bind_overrides.get(video_id) if bind_overrides else None
        if override is not None:
            inference_block = {
                "bind_map": bind_map,
                "confidence": confidence,
                "flagged": flagged,
                "per_action": per_action,
            }
            bind_map = {action: list(override[action]) for action in ACTION_NAMES}
            review = (
                dict(bind_review_status.get(video_id, {}))
                if bind_review_status
                else {}
            )
            bind_source = dict(bind_resolution_meta or {})
            per_action = {
                action: {
                    "selected": list(bind_map[action]),
                    "source": "resolved",
                    "review_status": review.get(action),
                }
                for action in ACTION_NAMES
            }
            flagged = False

        mapped_count = 0
        for row in rows:
            chunk_id = int(row["chunk_id"])
            chunk_dir = readable.get(chunk_id)
            if chunk_dir is None:
                continue
            try:
                raw = read_raw_chunk(chunk_dir / "actions_raw.parquet")
                mapped = _map_chunk(
                    raw,
                    bind_map=bind_map,
                    axis_threshold=axis_threshold,
                )
                output_path = (
                    destination_root
                    / video_id
                    / chunk_dir.name
                    / "labels_native.parquet"
                )
                _write_labels(
                    output_path,
                    mapped,
                    raw.row_count,
                    float(row["grid_hz"]),
                )
                mapped_count += 1
            except Exception as error:
                skipped_details.append(
                    {
                        "chunk_id": chunk_id,
                        "error": str(error) or type(error).__name__,
                    }
                )

        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "video_id": video_id,
            "controller_type": _video_controller_type(rows),
            "bind_map": bind_map,
            "confidence": confidence,
            "per_action": per_action,
            "evidence": evidence,
            "direction_rule": {
                "source": "NitroGen dataset coordinate contract",
                "source_revision": NITROGEN_COORDINATE_CONTRACT_REVISION,
                "axis_threshold": axis_threshold,
                "left": "dpad_left OR j_left_x < -axis_threshold",
                "right": "dpad_right OR j_left_x > axis_threshold",
                "up": "dpad_up OR j_left_y < -axis_threshold",
                "down": "dpad_down OR j_left_y > axis_threshold",
                "comparisons": "strict",
            },
            "flagged": flagged,
            "chunks_mapped": mapped_count,
            "chunks_skipped": len(rows) - mapped_count,
            "skipped_details": skipped_details,
            "candidate_scores": candidate_scores,
            "tool_version": TOOL_VERSION,
        }
        if bind_source is not None:
            report["bind_source"] = bind_source
            report["inference"] = inference_block
        report_dir = destination_root / video_id
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "mapping_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reports.append(report)
    return reports


def _parse_videos(value: str | None) -> list[str] | None:
    if value is None:
        return None
    videos = [part.strip() for part in value.split(",") if part.strip()]
    if not videos:
        raise argparse.ArgumentTypeError("--videos must contain at least one id")
    return videos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map NitroGen actions on each chunk's native row grid."
    )
    parser.add_argument("--chunk-index", required=True, type=Path)
    parser.add_argument("--actions-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--videos", type=_parse_videos)
    parser.add_argument(
        "--bind-resolution",
        type=Path,
        help="bind-resolution record; label actions from its resolved button sets",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = review = meta = None
    if args.bind_resolution is not None:
        overrides, review, meta = load_bind_resolution(args.bind_resolution)
    reports = map_actions(
        args.chunk_index,
        args.actions_root,
        args.out,
        videos=args.videos,
        bind_overrides=overrides,
        bind_review_status=review,
        bind_resolution_meta=meta,
    )
    print(
        f"videos={len(reports)} "
        f"chunks-mapped={sum(report['chunks_mapped'] for report in reports)} "
        f"chunks-skipped={sum(report['chunks_skipped'] for report in reports)} "
        f"flagged={sum(bool(report['flagged']) for report in reports)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
