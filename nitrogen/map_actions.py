"""Map NitroGen controller rows to MADELEINE's seven-key native-grid labels.

Every output row corresponds one-for-one with an ``actions_raw.parquet`` row.
The mapper never changes a chunk's rate or row count; it declares the measured
``grid_hz`` through :func:`data.schema.labels_native_schema`.
The resulting noisy labels are training fuel and an audit target, never
evaluation ground truth.

Bind inference is performed independently per video.  Each candidate button is
described by its pressed-frame rate, run count, median run duration, fraction
of runs no longer than 8/60 second, directional overlap, and upward-direction
overlap.  A joint, one-button-per-action assignment scores these signatures:
jump rewards frequent short-to-medium runs, dash rewards frequent very short
directional runs, and grab rewards long upward-direction runs.  Celeste's
default binds provide only a small tie-breaking prior.

Confidence is ``support * (0.65 * quality + 0.35 * margin)`` in ``[0, 1]``.
``support`` is the least-supported selected action relative to
``InferenceParameters.min_presses``; ``quality`` is the mean selected signature
score; and ``margin`` is the best-versus-second-best joint-assignment gap scaled
by ``InferenceParameters.assignment_margin_scale``.  Below ``FLAG_THRESHOLD``
the report is flagged and mapping deliberately falls back to the complete
default bind prior.

Joystick Y sign is inferred per video by comparing stick deflections on frames
where d-pad up/down is simultaneously active.  Indeterminate videos are
reported as such and use the dataset-observed/controller-standard negative-Y-up
fallback.  Stick arrays are explicitly converted element-by-element to float,
covering both DOUBLE[] and BIGINT[] source columns.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
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
DEFAULT_MIN_AXIS_VOTES = 3
DEFAULT_AXIS_DOMINANCE = 0.6
TOOL_VERSION = "nitrogen.map_actions/1.0"

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
PRIOR_BIND_MAP = {
    "jump": ["south"],
    "dash": ["west", "east"],
    "grab": [
        "left_trigger",
        "right_trigger",
        "left_shoulder",
        "right_shoulder",
    ],
}
ACTION_NAMES = ("jump", "dash", "grab")


@dataclass(frozen=True)
class InferenceParameters:
    """Thresholds and score-shape values intended for orchestration tuning."""

    min_presses: int = 3
    jump_target_frames_60hz_equivalent: float = 8.0
    dash_short_frames_60hz_equivalent: float = 3.0
    grab_long_frames_60hz_equivalent: float = 18.0
    assignment_margin_scale: float = 0.15
    prior_bonus: float = 0.08


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
    direction_overlap_negative_up: int = 0
    direction_overlap_positive_up: int = 0
    up_overlap_negative_up: int = 0
    up_overlap_positive_up: int = 0


@dataclass
class _EvidenceAccumulator:
    total_frames: int = 0
    buttons: dict[str, _ButtonAccumulator] = field(
        default_factory=lambda: {
            button: _ButtonAccumulator() for button in CANDIDATE_BUTTONS
        }
    )
    up_with_negative_stick: int = 0
    up_with_positive_stick: int = 0
    down_with_negative_stick: int = 0
    down_with_positive_stick: int = 0

    def update(
        self,
        raw: RawChunk,
        grid_hz: float,
        axis_threshold: float,
    ) -> None:
        if grid_hz <= 0:
            raise ValueError("grid_hz must be positive")
        self.total_frames += raw.row_count
        y_axis = raw.j_left[:, 1]
        stick_negative = y_axis < -axis_threshold
        stick_positive = y_axis > axis_threshold
        dpad_up = raw.buttons["dpad_up"]
        dpad_down = raw.buttons["dpad_down"]
        self.up_with_negative_stick += int(np.count_nonzero(dpad_up & stick_negative))
        self.up_with_positive_stick += int(np.count_nonzero(dpad_up & stick_positive))
        self.down_with_negative_stick += int(
            np.count_nonzero(dpad_down & stick_negative)
        )
        self.down_with_positive_stick += int(
            np.count_nonzero(dpad_down & stick_positive)
        )

        negative_up = _directional_states(raw, axis_threshold, "negative_is_up")
        positive_up = _directional_states(raw, axis_threshold, "positive_is_up")
        any_negative_up = np.logical_or.reduce(list(negative_up.values()))
        any_positive_up = np.logical_or.reduce(list(positive_up.values()))

        for button, accumulator in self.buttons.items():
            pressed = raw.buttons[button]
            accumulator.active_frames += int(np.count_nonzero(pressed))
            starts, ends = _true_runs(pressed)
            run_lengths = (ends - starts).astype(np.int64)
            accumulator.run_lengths_native.extend(int(length) for length in run_lengths)
            accumulator.run_durations_seconds.extend(
                float(length) / grid_hz for length in run_lengths
            )
            accumulator.direction_overlap_negative_up += int(
                np.count_nonzero(pressed & any_negative_up)
            )
            accumulator.direction_overlap_positive_up += int(
                np.count_nonzero(pressed & any_positive_up)
            )
            accumulator.up_overlap_negative_up += int(
                np.count_nonzero(pressed & negative_up["up"])
            )
            accumulator.up_overlap_positive_up += int(
                np.count_nonzero(pressed & positive_up["up"])
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
    axis_sign_convention: str,
) -> dict[str, np.ndarray]:
    x_axis = raw.j_left[:, 0]
    y_axis = raw.j_left[:, 1]
    if axis_sign_convention == "negative_is_up":
        stick_up = y_axis < -axis_threshold
        stick_down = y_axis > axis_threshold
    elif axis_sign_convention == "positive_is_up":
        stick_up = y_axis > axis_threshold
        stick_down = y_axis < -axis_threshold
    else:
        raise ValueError(f"unknown axis sign convention: {axis_sign_convention}")
    return {
        "left": raw.buttons["dpad_left"] | (x_axis < -axis_threshold),
        "right": raw.buttons["dpad_right"] | (x_axis > axis_threshold),
        "up": raw.buttons["dpad_up"] | stick_up,
        "down": raw.buttons["dpad_down"] | stick_down,
    }


def _infer_axis_sign(
    accumulator: _EvidenceAccumulator,
    *,
    min_axis_votes: int,
    axis_dominance: float,
) -> tuple[str, bool, dict[str, int | float]]:
    negative_votes = (
        accumulator.up_with_negative_stick
        + accumulator.down_with_positive_stick
    )
    positive_votes = (
        accumulator.up_with_positive_stick
        + accumulator.down_with_negative_stick
    )
    total_votes = negative_votes + positive_votes
    winning_share = (
        max(negative_votes, positive_votes) / total_votes if total_votes else 0.0
    )
    indeterminate = (
        total_votes < min_axis_votes
        or winning_share < axis_dominance
        or negative_votes == positive_votes
    )
    if indeterminate or negative_votes > positive_votes:
        convention = "negative_is_up"
    else:
        convention = "positive_is_up"
    evidence: dict[str, int | float] = {
        "dpad_up_with_negative_stick": accumulator.up_with_negative_stick,
        "dpad_up_with_positive_stick": accumulator.up_with_positive_stick,
        "dpad_down_with_negative_stick": accumulator.down_with_negative_stick,
        "dpad_down_with_positive_stick": accumulator.down_with_positive_stick,
        "negative_is_up_votes": negative_votes,
        "positive_is_up_votes": positive_votes,
        "winning_share": winning_share,
    }
    return convention, indeterminate, evidence


def _evidence_stats(
    accumulator: _EvidenceAccumulator,
    axis_sign_convention: str,
) -> dict[str, dict[str, int | float | str]]:
    stats: dict[str, dict[str, int | float | str]] = {}
    negative_up = axis_sign_convention == "negative_is_up"
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
        direction_overlap = (
            values.direction_overlap_negative_up
            if negative_up
            else values.direction_overlap_positive_up
        )
        up_overlap = (
            values.up_overlap_negative_up
            if negative_up
            else values.up_overlap_positive_up
        )
        active_frames = values.active_frames
        stats[button] = {
            "press_rate": (
                active_frames / accumulator.total_frames
                if accumulator.total_frames
                else 0.0
            ),
            "press_count": press_count,
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
            if duration_frames <= parameters.dash_short_frames_60hz_equivalent:
                dash_fit = 1.0
            else:
                dash_fit = math.exp(
                    -(duration_frames - parameters.dash_short_frames_60hz_equivalent)
                    / 4.0
                )
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


def infer_bind_map(
    evidence: Mapping[str, Mapping[str, int | float | str]],
    *,
    flag_threshold: float = FLAG_THRESHOLD,
    parameters: InferenceParameters = DEFAULT_INFERENCE_PARAMETERS,
) -> tuple[dict[str, list[str]], float, bool, dict[str, dict[str, float]]]:
    """Infer a unique three-button assignment, with low-confidence prior fallback."""

    if not 0.0 <= flag_threshold <= 1.0:
        raise ValueError("flag_threshold must be in [0, 1]")
    scores = _candidate_scores(evidence, parameters)
    assignments: list[tuple[float, tuple[str, str, str]]] = []
    for buttons in itertools.permutations(CANDIDATE_BUTTONS, len(ACTION_NAMES)):
        mean_score = sum(
            scores[action][button] for action, button in zip(ACTION_NAMES, buttons)
        ) / len(ACTION_NAMES)
        assignments.append((mean_score, buttons))
    assignments.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_buttons = assignments[0]
    second_score = assignments[1][0]
    margin = min(
        1.0,
        max(0.0, best_score - second_score)
        / max(parameters.assignment_margin_scale, 1e-9),
    )
    selected_press_counts = [
        int(evidence[button]["press_count"]) for button in best_buttons
    ]
    support = min(
        1.0,
        min(selected_press_counts, default=0) / max(parameters.min_presses, 1),
    )
    confidence = min(1.0, max(0.0, support * (0.65 * best_score + 0.35 * margin)))
    flagged = confidence < flag_threshold
    if flagged:
        bind_map = {action: list(buttons) for action, buttons in PRIOR_BIND_MAP.items()}
    else:
        bind_map = {
            action: [button] for action, button in zip(ACTION_NAMES, best_buttons)
        }
    return bind_map, confidence, flagged, scores


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
    axis_sign_convention: str,
) -> dict[str, np.ndarray]:
    directions = _directional_states(raw, axis_threshold, axis_sign_convention)
    mapped = dict(directions)
    for action in ACTION_NAMES:
        selected = [raw.buttons[button] for button in bind_map[action]]
        mapped[action] = np.logical_or.reduce(selected)
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
    min_axis_votes: int = DEFAULT_MIN_AXIS_VOTES,
    axis_dominance: float = DEFAULT_AXIS_DOMINANCE,
) -> list[dict[str, Any]]:
    """Map selected videos and return their JSON-serializable reports."""

    if axis_threshold < 0:
        raise ValueError("axis_threshold must be non-negative")
    if min_axis_votes < 0:
        raise ValueError("min_axis_votes must be non-negative")
    if not 0.5 <= axis_dominance <= 1.0:
        raise ValueError("axis_dominance must be in [0.5, 1]")

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

        convention, axis_indeterminate, axis_evidence = _infer_axis_sign(
            accumulator,
            min_axis_votes=min_axis_votes,
            axis_dominance=axis_dominance,
        )
        evidence = _evidence_stats(accumulator, convention)
        bind_map, confidence, flagged, candidate_scores = infer_bind_map(
            evidence,
            flag_threshold=flag_threshold,
            parameters=inference_parameters,
        )

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
                    axis_sign_convention=convention,
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
            "video_id": video_id,
            "controller_type": _video_controller_type(rows),
            "bind_map": bind_map,
            "confidence": confidence,
            "evidence": evidence,
            "axis_sign_convention": convention,
            "axis_sign_indeterminate": axis_indeterminate,
            "axis_sign_evidence": axis_evidence,
            "flagged": flagged,
            "chunks_mapped": mapped_count,
            "chunks_skipped": len(rows) - mapped_count,
            "skipped_details": skipped_details,
            "candidate_scores": candidate_scores,
            "tool_version": TOOL_VERSION,
        }
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = map_actions(
        args.chunk_index,
        args.actions_root,
        args.out,
        videos=args.videos,
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
