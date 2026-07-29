"""Generate deterministic, visually inspectable frozen-format toy sessions."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.schema import ALIGNMENT_SCHEMA, KEY_ORDER, TRUTH_SCHEMA


FPS = 60
FRAME_WIDTH = 320
FRAME_HEIGHT = 180
SQUARE_SIZE = 16
FRAME_INDEX_CELL_SIZE = 8
FRAME_INDEX_CELL_COUNT = 30
FRAME_INDEX_QUIET_CELLS = 1
FRAME_INDEX_STRIP_WIDTH = 32 * FRAME_INDEX_CELL_SIZE
FRAME_INDEX_STRIP_HEIGHT = 3 * FRAME_INDEX_CELL_SIZE
FRAME_INDEX_MODULUS = 1 << 24


def encode_frame_index_cells(frame_idx: int) -> tuple[int, ...]:
    """Encode a frame index as S1 S0, 24 data bits, and four checksum bits."""
    if frame_idx < 0:
        raise ValueError("frame_idx must be non-negative")
    value = frame_idx & (FRAME_INDEX_MODULUS - 1)
    checksum = 0
    for shift in range(20, -1, -4):
        checksum ^= (value >> shift) & 0xF
    data = tuple((value >> shift) & 1 for shift in range(23, -1, -1))
    check = tuple((checksum >> shift) & 1 for shift in range(3, -1, -1))
    return (1, 0, *data, *check)


def render_frame_index_strip(
    frame_idx: int, cell_size: int = FRAME_INDEX_CELL_SIZE
) -> np.ndarray:
    """Render the scaled backing bar as a hard-edged uint8 luma image."""
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    strip = np.zeros((3 * cell_size, 32 * cell_size), dtype=np.uint8)
    for cell_index, bit in enumerate(encode_frame_index_cells(frame_idx)):
        if bit:
            x0 = (FRAME_INDEX_QUIET_CELLS + cell_index) * cell_size
            strip[cell_size : 2 * cell_size, x0 : x0 + cell_size] = 255
    return strip


def _scripted_keys(frame_count: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    keys = {key: np.zeros(frame_count, dtype=np.bool_) for key in KEY_ORDER}
    swap_horizontal = bool(rng.integers(0, 2))
    first, second = ("right", "left") if swap_horizontal else ("left", "right")
    for frame in range(frame_count):
        phase = frame % 90
        keys[first][frame] = 0 <= phase < 20
        keys[second][frame] = 25 <= phase < 55
        keys["up"][frame] = 6 <= phase < 12
        keys["down"][frame] = phase == 31
        keys["jump"][frame] = phase == 8
        keys["dash"][frame] = phase in {8, 52}
        keys["grab"][frame] = 30 <= phase < 45

    # Seeded extra taps keep the fixture deterministic while making seeds differ.
    if frame_count:
        for key in ("jump", "dash", "up", "down"):
            frame = int(rng.integers(0, frame_count))
            keys[key][frame] = True
    return keys


def _table_from_columns(schema: pa.Schema, columns: dict[str, Any]) -> pa.Table:
    arrays = [pa.array(columns[field.name], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_ffmpeg(path: Path) -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "cfr",
        "-r",
        str(FPS),
        str(path),
    ]
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise RuntimeError(f"could not start ffmpeg: {exc}") from exc


def _render_game_frame(
    frame_idx: int,
    x: float,
    y: float,
    active: bool,
    dash: bool,
    death: bool,
) -> np.ndarray:
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (30, 24, 20), dtype=np.uint8)
    cv2.line(frame, (0, 172), (FRAME_WIDTH - 1, 172), (80, 80, 80), 2)
    for platform_x in range(16, FRAME_WIDTH, 48):
        cv2.rectangle(
            frame,
            (platform_x, 144),
            (platform_x + 27, 148),
            (55, 55, 65),
            -1,
        )

    if active:
        x0 = int(round(x))
        y0 = int(round(y))
        if dash:
            cv2.rectangle(
                frame,
                (x0 - 4, y0 - 4),
                (x0 + SQUARE_SIZE + 3, y0 + SQUARE_SIZE + 3),
                (255, 220, 80),
                2,
            )
        square_color = (210, 210, 255) if death else (255, 255, 255)
        cv2.rectangle(
            frame,
            (x0, y0),
            (x0 + SQUARE_SIZE - 1, y0 + SQUARE_SIZE - 1),
            square_color,
            -1,
        )

    strip = render_frame_index_strip(frame_idx)
    frame[: strip.shape[0], : strip.shape[1]] = strip[:, :, None]
    return frame


def _write_one_session(
    session_dir: Path,
    frame_count: int,
    seed: int,
    session_number: int,
) -> None:
    session_id = session_dir.name
    rng = np.random.default_rng(
        np.random.SeedSequence([seed & 0xFFFFFFFF, session_number])
    )
    keys = _scripted_keys(frame_count, rng)
    max_start = FRAME_INDEX_MODULUS - frame_count
    if max_start <= 0:
        raise ValueError("session is too long for the 24-bit frame-index payload")
    engine_start = int(
        ((seed & 0xFFFFFFFF) * 97 + session_number * 1009) % max_start
    )

    columns: dict[str, list[Any] | np.ndarray] = {
        "frame_idx": np.arange(
            engine_start, engine_start + frame_count, dtype=np.int64
        ),
        **keys,
        "input_active": [],
        "room_id": [],
        "pos_x": [],
        "pos_y": [],
        "speed_x": [],
        "speed_y": [],
        "dash_count": [],
        "stamina": [],
        "on_ground": [],
        "death": [],
        "session_id": [session_id] * frame_count,
    }

    x = float(rng.uniform(72.0, 232.0))
    ground_y = 156.0
    y = ground_y
    speed_x = 0.0
    speed_y = 0.0
    on_ground = True
    dash_available = 1
    dash_cooldown = 0
    stamina = 110.0

    video_path = session_dir / "video.mkv"
    encoder = _open_ffmpeg(video_path)
    assert encoder.stdin is not None
    try:
        for video_frame_idx in range(frame_count):
            phase = video_frame_idx % 97
            player_absent = phase in {15, 16, 17}
            death = phase == 14
            active = not player_absent

            left = bool(keys["left"][video_frame_idx])
            right = bool(keys["right"][video_frame_idx])
            up = bool(keys["up"][video_frame_idx])
            down = bool(keys["down"][video_frame_idx])
            jump = bool(keys["jump"][video_frame_idx])
            dash = bool(keys["dash"][video_frame_idx])
            grab = bool(keys["grab"][video_frame_idx])

            if active:
                direction = int(right) - int(left)
                target_speed_x = 96.0 * direction
                speed_x += np.clip(target_speed_x - speed_x, -18.0, 18.0)
                if direction == 0:
                    speed_x *= 0.82
                if jump and on_ground:
                    speed_y = -215.0
                    on_ground = False
                if dash and dash_available:
                    dash_direction = direction or (1 if speed_x >= 0 else -1)
                    speed_x = 220.0 * dash_direction
                    dash_available = 0
                    dash_cooldown = 14
                if not on_ground:
                    speed_y += 510.0 / FPS
                    if up:
                        speed_y -= 8.0
                    if down:
                        speed_y += 10.0
                    if grab and speed_y > 45.0:
                        speed_y = 45.0
                x += speed_x / FPS
                y += speed_y / FPS
                x = float(np.clip(x, 0.0, FRAME_WIDTH - SQUARE_SIZE))
                if y >= ground_y:
                    y = ground_y
                    speed_y = 0.0
                    on_ground = True
                dash_cooldown = max(0, dash_cooldown - 1)
                if on_ground and dash_cooldown == 0:
                    dash_available = 1
                stamina = float(
                    np.clip(stamina + (-2.5 if grab else 1.0), 0.0, 110.0)
                )

                columns["pos_x"].append(x)
                columns["pos_y"].append(y)
                columns["speed_x"].append(speed_x)
                columns["speed_y"].append(speed_y)
                columns["dash_count"].append(dash_available)
                columns["stamina"].append(stamina)
                columns["on_ground"].append(on_ground)
            else:
                columns["pos_x"].append(np.nan)
                columns["pos_y"].append(np.nan)
                columns["speed_x"].append(np.nan)
                columns["speed_y"].append(np.nan)
                columns["dash_count"].append(-1)
                columns["stamina"].append(np.nan)
                columns["on_ground"].append(False)

            columns["input_active"].append(active)
            columns["room_id"].append("toy_a")
            columns["death"].append(death)
            frame = _render_game_frame(
                engine_start + video_frame_idx, x, y, active, dash, death
            )
            encoder.stdin.write(frame.tobytes())
    except (BrokenPipeError, OSError) as exc:
        raise RuntimeError(f"ffmpeg stopped while encoding {video_path}") from exc
    finally:
        encoder.stdin.close()
    stderr = (
        encoder.stderr.read().decode("utf-8", errors="replace")
        if encoder.stderr
        else ""
    )
    return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path}: {stderr.strip() or 'unknown error'}"
        )

    truth = _table_from_columns(TRUTH_SCHEMA, columns)
    truth_path = session_dir / "truth.parquet"
    pq.write_table(truth, truth_path)

    alignment_columns = {
        "video_frame_idx": np.arange(frame_count, dtype=np.int64),
        "engine_frame_idx": np.arange(
            engine_start, engine_start + frame_count, dtype=np.int64
        ),
        "decode_status": ["ok"] * frame_count,
        "is_duplicate": np.zeros(frame_count, dtype=np.bool_),
        "preceded_by_drop_count": np.zeros(frame_count, dtype=np.int32),
    }
    alignment = _table_from_columns(ALIGNMENT_SCHEMA, alignment_columns)
    alignment_path = session_dir / "alignment.parquet"
    pq.write_table(alignment, alignment_path)

    seed_tag = seed & 0xFFFFFFFF
    created_at = (
        datetime(2025, 1, 1, tzinfo=timezone.utc)
        + timedelta(seconds=seed_tag + session_number)
    ).isoformat().replace("+00:00", "Z")
    stream_hashes = {
        path.name: _sha256(path)
        for path in (video_path, truth_path, alignment_path)
    }
    manifest = {
        "format_version": "1",
        "session_id": session_id,
        "created_at": created_at,
        "env": {
            "game": "madeleine-toy",
            "everest": "not-applicable",
            "mod": "InputTruth toy-v1",
        },
        "capture": {
            "tool": "ffmpeg-rawvideo",
            "requested_fps": FPS,
            "achieved_fps": 60.0,
            "encode": "libx264 crf16 veryfast yuv420p",
            "resolution": [FRAME_WIDTH, FRAME_HEIGHT],
        },
        "streams": {
            "video": "video.mkv",
            "truth": "truth.parquet",
            "alignment": "alignment.parquet",
            "overlay_style": "none",
        },
        "grid": {"engine_hz": FPS},
        "label_kind": "engine_truth",
        "masked_regions": [
            {
                "name": "frame_index_strip",
                "space": "capture_pixels",
                "applied": "post_crop",
                "rect_px": [
                    0,
                    0,
                    FRAME_INDEX_STRIP_WIDTH,
                    FRAME_INDEX_STRIP_HEIGHT,
                ],
                "rect_norm": [
                    0.0,
                    0.0,
                    FRAME_INDEX_STRIP_WIDTH / FRAME_WIDTH,
                    FRAME_INDEX_STRIP_HEIGHT / FRAME_HEIGHT,
                ],
            }
        ],
        "integrity": {
            "video_frames": frame_count,
            "duplicates": 0,
            "drops": 0,
            "sha256": stream_hashes,
        },
        "actions": {"keys": KEY_ORDER},
        "provenance": {
            "source": "recorded",
            "origin_url": None,
            "mapping_report": None,
        },
    }
    (session_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def generate_sessions(
    out: str | Path, sessions: int, seconds: float, seed: int
) -> list[Path]:
    if sessions <= 0:
        raise ValueError("sessions must be positive")
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    frame_count = int(round(seconds * FPS))
    if frame_count <= 0:
        raise ValueError("seconds must produce at least one 60 Hz frame")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = seed & 0xFFFFFFFF
    generated: list[Path] = []
    for session_number in range(sessions):
        session_id = f"toy_{seed_tag:010d}_{session_number:03d}"
        session_dir = out_dir / session_id
        session_dir.mkdir()
        _write_one_session(session_dir, frame_count, seed, session_number)
        generated.append(session_dir)
    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--sessions", required=True, type=int)
    parser.add_argument("--seconds", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args(argv)
    generated = generate_sessions(
        out=args.out,
        sessions=args.sessions,
        seconds=args.seconds,
        seed=args.seed,
    )
    for session_dir in generated:
        print(session_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
