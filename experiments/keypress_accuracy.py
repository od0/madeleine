"""Score per-frame key-state accuracy from prediction sidecars.

Each frame/key pair is one binary decision.  The default decision rule is the
natural binary-head argmax, ``probability >= 0.5``.  By default, scoring is
restricted to rows whose ``input_active`` flag is true, matching the project's
normal valid-gameplay evaluation surface.

Both common readings of "keypress accuracy" are reported:

* micro accuracy treats every frame/key pair as one binary decision; and
* joint accuracy requires the complete seven-key vector to match on a frame.

Always-released and one-frame persistence baselines use exactly the same rows
as the model.  Persistence is constructed on the original streams before the
activity gate is applied, so an inactive row still supplies the preceding
state and state is never copied across a stream boundary.  At the first frame
of each stream the baseline starts from the all-released state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.schema import KEY_ORDER


def _stream_starts(lengths: np.ndarray, frame_count: int) -> np.ndarray:
    """Validate stream lengths and return their zero-based start offsets."""

    if lengths.ndim != 1:
        raise ValueError("session_lengths must have shape [streams]")
    if not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError("session_lengths must contain integers")
    if not len(lengths):
        raise ValueError("session_lengths must contain at least one stream")
    if np.any(lengths <= 0):
        raise ValueError("session_lengths must contain only positive values")
    if int(lengths.sum()) != frame_count:
        raise ValueError("session_lengths must sum to the y_true frame count")
    return np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64))
    )


def _accuracy(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """Return (frame/key micro accuracy, per-frame joint accuracy)."""

    correct = predicted == truth
    return float(correct.mean()), float(correct.all(axis=1).mean())


def score_sidecar(
    path: Path,
    *,
    threshold: float = 0.5,
    active_only: bool = True,
) -> dict[str, object]:
    """Return aggregate and per-key binary state accuracy for one sidecar."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")

    with np.load(path, allow_pickle=False) as archive:
        required = {"y_true", "y_prob"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing arrays: {sorted(missing)}")

        truth = np.asarray(archive["y_true"])
        probability = np.asarray(archive["y_prob"])
        expected_shape = (truth.shape[0], len(KEY_ORDER))
        if truth.ndim != 2 or truth.shape != expected_shape:
            raise ValueError(
                f"{path}: y_true must have shape [N,{len(KEY_ORDER)}]"
            )
        if probability.shape != truth.shape:
            raise ValueError(f"{path}: y_prob shape does not match y_true")
        if not np.all(np.isin(truth, (0, 1))):
            raise ValueError(f"{path}: y_true is not binary")
        if not np.all(np.isfinite(probability)):
            raise ValueError(f"{path}: y_prob contains non-finite values")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError(f"{path}: y_prob lies outside [0, 1]")

        frame_count = truth.shape[0]
        if "session_lengths" in archive.files:
            stream_lengths = np.asarray(archive["session_lengths"])
        else:
            # Older/local sidecars may omit stream metadata.  Preserve their
            # historical single-stream interpretation.
            stream_lengths = np.asarray([frame_count], dtype=np.int64)
        stream_starts = _stream_starts(stream_lengths, frame_count)
        if "session_ids" in archive.files:
            session_ids = np.asarray(archive["session_ids"])
            if session_ids.ndim != 1 or len(session_ids) != len(stream_lengths):
                raise ValueError(
                    f"{path}: session_ids must have one entry per stream"
                )

        gate = np.ones(frame_count, dtype=bool)
        if active_only:
            if "input_active" not in archive.files:
                raise ValueError(f"{path}: input_active is required")
            active = np.asarray(archive["input_active"])
            if active.shape != (frame_count,):
                raise ValueError(f"{path}: input_active must have shape [N]")
            if not np.all(np.isin(active, (0, 1))):
                raise ValueError(f"{path}: input_active is not binary")
            gate = active.astype(bool)

    truth = truth.astype(bool, copy=False)
    predicted = probability >= threshold

    always_released = np.zeros_like(truth, dtype=bool)
    persistence = np.zeros_like(truth, dtype=bool)
    stream_ends = np.concatenate((stream_starts[1:], [frame_count]))
    for start, end in zip(stream_starts, stream_ends, strict=True):
        # The first row remains all-released; every later row copies only the
        # preceding state in this stream.
        persistence[start + 1 : end] = truth[start : end - 1]

    truth = truth[gate]
    predicted = predicted[gate]
    always_released = always_released[gate]
    persistence = persistence[gate]
    if not len(truth):
        raise ValueError(f"{path}: selected evaluation surface is empty")

    correct = predicted == truth
    micro_accuracy, joint_accuracy = _accuracy(predicted, truth)
    released_micro, released_joint = _accuracy(always_released, truth)
    persistence_micro, persistence_joint = _accuracy(persistence, truth)
    per_key = {
        key: float(correct[:, column].mean())
        for column, key in enumerate(KEY_ORDER)
    }
    per_key_baseline = {
        key: float((~truth[:, column]).mean())
        for column, key in enumerate(KEY_ORDER)
    }
    per_key_persistence = {
        key: float((persistence[:, column] == truth[:, column]).mean())
        for column, key in enumerate(KEY_ORDER)
    }
    return {
        "path": str(path),
        "threshold": threshold,
        "active_only": active_only,
        "frames": int(len(truth)),
        "binary_decisions": int(truth.size),
        "streams": int(len(stream_lengths)),
        # Explicit names are the preferred interface.  The shorter historical
        # names below remain aliases so existing report consumers keep working.
        "key_state_micro_accuracy": micro_accuracy,
        "joint_exact_match_accuracy": joint_accuracy,
        "always_released_key_state_micro_accuracy": released_micro,
        "always_released_joint_exact_match_accuracy": released_joint,
        "persistence_key_state_micro_accuracy": persistence_micro,
        "persistence_joint_exact_match_accuracy": persistence_joint,
        "accuracy": micro_accuracy,
        "always_released_accuracy": released_micro,
        "exact_seven_key_vector_accuracy": joint_accuracy,
        "per_key_accuracy": per_key,
        "per_key_always_released_accuracy": per_key_baseline,
        "per_key_persistence_accuracy": per_key_persistence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sidecars", nargs="+", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="include rows whose input_active flag is false",
    )
    args = parser.parse_args()

    reports = [
        score_sidecar(
            path,
            threshold=args.threshold,
            active_only=not args.all_frames,
        )
        for path in args.sidecars
    ]
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
