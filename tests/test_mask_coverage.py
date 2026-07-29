"""Mask-coverage check: an undershooting rect must fail, a covering one pass.

Regression tests for the 2026-07-26 masking-audit finding: the declared
input_overlay rect missed the top of the rendered key cells and the leak
survived into training shards because the builder only verified that the
declared rect was zeroed. These tests paint a key-driven overlay widget onto
a toy session and assert that data.mask_coverage rejects a declared rect
that does not cover it — including through data.build_dataset.build_session.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from data.build_dataset import build_session
from data.mask_coverage import verify_mask_coverage
from data.schema import KEY_ORDER
from data.toy_sessions import _open_ffmpeg, generate_sessions

# Widget geometry, top-right so it never overlaps the toy player's range
# (the player square is key-correlated by construction; the widget must be
# the only key-correlated content near its own rect) and sits fully clear
# of the frame-index strip's columns (x < 256), whose rect the band
# correctly excludes.
BAR_X, BAR_Y, BAR_W, BAR_H = 256, 0, 64, 20
CELL_Y, CELL_H, CELL_W, CELL_PITCH = 4, 12, 6, 8


@pytest.fixture(scope="module")
def toy_session(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("mask_coverage_toy")
    return generate_sessions(out=out, sessions=1, seconds=6.0, seed=20260726)[0]


def _paint_overlay(frame: np.ndarray, keys_down: np.ndarray) -> None:
    frame[BAR_Y : BAR_Y + BAR_H, BAR_X : BAR_X + BAR_W] = (20, 20, 20)
    for cell, down in enumerate(keys_down):
        x0 = BAR_X + 2 + cell * CELL_PITCH
        color = (255, 255, 255) if down else (40, 40, 40)
        frame[CELL_Y : CELL_Y + CELL_H, x0 : x0 + CELL_W] = color


def _with_overlay(source: Path, dest: Path, declared_y: int) -> Path:
    """Copy a toy session, paint a key-driven widget, declare its rect.

    ``declared_y`` is the declared rect's top; the widget itself is always
    rendered at BAR_Y, so declared_y > BAR_Y reproduces the undershoot.
    """
    shutil.copytree(source, dest)
    truth = pq.read_table(dest / "truth.parquet")
    keys = np.stack(
        [np.asarray(truth[k].to_pylist(), dtype=bool) for k in KEY_ORDER],
        axis=1,
    )

    cap = cv2.VideoCapture(str(dest / "video.mkv"))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) == len(keys)

    video_path = dest / "video.mkv"
    video_path.unlink()
    encoder = _open_ffmpeg(video_path)
    assert encoder.stdin is not None
    for frame, keys_down in zip(frames, keys):
        _paint_overlay(frame, keys_down)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    assert encoder.wait() == 0

    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    frame_h, frame_w = frames[0].shape[:2]
    manifest["masked_regions"].append(
        {
            "name": "input_overlay",
            "space": "capture_pixels",
            "applied": "post_crop",
            "rect_px": [BAR_X, declared_y, BAR_W, BAR_H],
            "rect_norm": [
                BAR_X / frame_w,
                declared_y / frame_h,
                (BAR_X + BAR_W) / frame_w,
                (declared_y + BAR_H) / frame_h,
            ],
        }
    )
    digest = hashlib.sha256()
    with video_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest["integrity"]["sha256"]["video.mkv"] = digest.hexdigest()
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return dest


def test_plain_toy_session_passes(toy_session: Path) -> None:
    assert verify_mask_coverage(toy_session) == []


def test_covering_rect_passes(toy_session: Path, tmp_path: Path) -> None:
    session = _with_overlay(
        toy_session, tmp_path / toy_session.name, declared_y=BAR_Y
    )
    assert verify_mask_coverage(session) == []


def test_undershooting_rect_fails(toy_session: Path, tmp_path: Path) -> None:
    session = _with_overlay(
        toy_session, tmp_path / toy_session.name, declared_y=BAR_Y + 12
    )
    violations = verify_mask_coverage(session)
    assert len(violations) == 1
    assert "input_overlay" in violations[0]
    assert "does not cover" in violations[0]


def test_build_refuses_undershooting_session(
    toy_session: Path, tmp_path: Path
) -> None:
    session = _with_overlay(
        toy_session, tmp_path / toy_session.name, declared_y=BAR_Y + 12
    )
    with pytest.raises(SystemExit, match="mask coverage failed"):
        build_session(session, tmp_path / "shards")
