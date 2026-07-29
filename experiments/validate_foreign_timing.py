"""Audit one foreign shard against the visible controller overlay.

The mapped labels ultimately came from this overlay, so raw overlay-pixel
motion should peak on the same source frame as a mapped key transition. This
does not prove every button classification, but it directly detects the
constant ±frame timing offsets that would invalidate an IDM experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from data.schema import KEY_ORDER
from nitrogen.mask import video_mask_rect


def transition_lag_scores(
    keys: np.ndarray, overlay_change: np.ndarray, radius: int = 4
) -> tuple[dict[int, float], np.ndarray]:
    """Mean overlay-change energy around every mapped key transition."""

    events = np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1
    if not len(events):
        raise ValueError("shard contains no mapped key transitions")
    scores = {
        lag: float(overlay_change[events + lag].mean())
        for lag in range(-radius, radius + 1)
        if int((events + lag).min()) >= 0
        and int((events + lag).max()) < len(overlay_change)
    }
    return scores, events


def audit(
    video: Path,
    video_id: str,
    shard: Path,
    chunk_index: Path,
    lag_radius: int = 4,
) -> dict:
    with np.load(shard, allow_pickle=False) as archive:
        frames = archive["frames"]
        keys = archive["keys"]
        frame_idx = archive["engine_frame_idx"]
        active = archive["input_active"]

    if not np.all(np.diff(frame_idx) == 1):
        raise ValueError("audit requires one contiguous shard")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rect = video_mask_rect(chunk_index, video_id, (width, height))
    x0, y0, x1, y1 = rect
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx[0]))

    overlay_change: list[float] = []
    previous: np.ndarray | None = None
    for source_frame in frame_idx:
        position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if position != int(source_frame):
            raise ValueError(
                f"decoder at source frame {position}, expected {source_frame}"
            )
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"decode failed at source frame {source_frame}")
        roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        change = (
            0.0 if previous is None
            else float(np.abs(roi.astype(np.int16) - previous.astype(np.int16)).mean())
        )
        overlay_change.append(change)
        previous = roi
    cap.release()

    change_array = np.asarray(overlay_change)
    lag_scores, events = transition_lag_scores(keys, change_array, lag_radius)
    best_lag = max(lag_scores, key=lag_scores.get)

    frame_size = frames.shape[1]
    sx0 = max(0, int(x0 / width * frame_size) - 1)
    sy0 = max(0, int(y0 / height * frame_size) - 1)
    sx1 = min(frame_size, int(np.ceil(x1 / width * frame_size)) + 1)
    sy1 = min(frame_size, int(np.ceil(y1 / height * frame_size)) + 1)
    transitions = np.abs(np.diff(keys.astype(np.int8), axis=0)).sum(axis=0)

    return {
        "video_id": video_id,
        "video": str(video),
        "shard": str(shard),
        "frames": int(len(frame_idx)),
        "source_frame_range_inclusive": [
            int(frame_idx[0]), int(frame_idx[-1])
        ],
        "contiguous": True,
        "input_active_all": bool(active.all()),
        "stored_mask_max": int(frames[:, sy0:sy1, sx0:sx1].max()),
        "transition_events_any_key": int(len(events)),
        "transitions_per_key": {
            key: int(value)
            for key, value in zip(KEY_ORDER, transitions, strict=True)
        },
        "key_positive_rate": {
            key: float(value)
            for key, value in zip(KEY_ORDER, keys.mean(axis=0), strict=True)
        },
        "overlay_change_by_label_lag": {
            str(lag): value for lag, value in lag_scores.items()
        },
        "best_lag_frames": int(best_lag),
        "lag_zero_vs_background_ratio": float(
            lag_scores[0] / max(float(change_array.mean()), 1e-12)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--chunk-index", type=Path, required=True)
    parser.add_argument("--lag-radius", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = audit(
        args.video, args.video_id, args.shard, args.chunk_index,
        args.lag_radius,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
