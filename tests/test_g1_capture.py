from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from theo import g1_capture


def test_capture_snapshot_reports_wall_time_video_time_and_effective_fps() -> None:
    meta = g1_capture._capture_snapshot(
        {"requested_frames": 600},
        status="recording",
        frames_written=300,
        wall_clock_seconds=10.0,
        grab_times=[0.0, 0.04, 0.08],
    )

    assert meta["status"] == "recording"
    assert meta["frames_written"] == 300
    assert meta["wall_clock_seconds"] == 10.0
    assert meta["video_duration_seconds"] == 5.0
    assert meta["effective_fps"] == 30.0
    assert meta["achieved_fps"] == 30.0
    assert meta["wall_minus_video_seconds"] == 5.0
    assert meta["tick_jitter_ms_p99"] == pytest.approx(40.0)


def test_capture_metadata_is_atomically_replaced(tmp_path: Path) -> None:
    path = tmp_path / "capture_meta.json"
    g1_capture._write_capture_meta(path, {"status": "recording", "frames": 1})
    g1_capture._write_capture_meta(path, {"status": "recording", "frames": 2})

    assert json.loads(path.read_text()) == {"status": "recording", "frames": 2}
    assert not (tmp_path / ".capture_meta.json.tmp").exists()


def test_interrupted_recording_keeps_final_incremental_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeFrame:
        raw = b"\x00" * 16

        def __array__(self, dtype=None, copy=None):  # noqa: ANN001
            return np.zeros((2, 2, 4), dtype=dtype or np.uint8)

    class FakeCapture:
        grabs = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def grab(self, _rect):
            FakeCapture.grabs += 1
            # One sizing grab, one successfully written frame, then Ctrl-C.
            if FakeCapture.grabs == 3:
                raise KeyboardInterrupt
            return FakeFrame()

    class FakeStdin:
        def __init__(self) -> None:
            self.closed = False
            self.writes = 0

        def write(self, _raw: bytes) -> None:
            self.writes += 1

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = FakeStdin()
            self.args = ["ffmpeg"]

        def wait(self) -> int:
            return 0

    process = FakeProcess()
    metadata_updates: list[dict[str, object]] = []
    original_write_capture_meta = g1_capture._write_capture_meta

    def track_metadata(path: Path, meta: dict[str, object]) -> None:
        metadata_updates.append(dict(meta))
        original_write_capture_meta(path, meta)

    monkeypatch.setattr(g1_capture, "capture_rect", lambda *_args: {
        "left": 0, "top": 0, "width": 2, "height": 2
    })
    monkeypatch.setattr(g1_capture.mss, "mss", FakeCapture)
    monkeypatch.setattr(g1_capture.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(g1_capture.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(g1_capture, "PROGRESS_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(g1_capture, "_write_capture_meta", track_metadata)

    with pytest.raises(KeyboardInterrupt):
        g1_capture.record(1.0, tmp_path)

    meta = json.loads((tmp_path / "capture_meta.json").read_text())
    assert meta["status"] == "interrupted"
    assert meta["frames_written"] == 1
    assert meta["video_duration_seconds"] == pytest.approx(1 / 60, abs=0.001)
    assert meta["wall_clock_seconds"] >= 0.0
    assert meta["error"] == "KeyboardInterrupt"
    assert meta["ffmpeg_return_code"] == 0
    assert process.stdin.closed
    assert any(
        update["status"] == "recording" and update["frames_written"] == 1
        for update in metadata_updates
    )
