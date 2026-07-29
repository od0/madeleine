import io
from pathlib import Path

import numpy as np
import pytest

import data.precompute_features as features


def test_nominal_timeline_preserves_cfr_and_expands_vfr() -> None:
    assert features._nominal_timeline_frames(6_000, 60.0) == (False, 6_000)
    assert features._nominal_timeline_frames(3_389, 33.89) == (
        True,
        round(3_389 / 33.89 * 60.0),
    )


def test_timestamp_resample_repeats_only_a_short_tail(monkeypatch) -> None:
    frame = np.full(
        (features.FRAME_SIZE, features.FRAME_SIZE, 3), 17, dtype=np.uint8
    )
    commands: list[list[str]] = []

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO(frame.tobytes())
            self.stderr = io.BytesIO(b"non-fatal diagnostic\n")

        @staticmethod
        def wait() -> int:
            return 0

    def fake_popen(command, **_kwargs):
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr(features.subprocess, "Popen", fake_popen)
    decoded, imputed = features._decode_resampled_part(
        Path("variable.mp4"),
        start_frame=1_200,
        end_frame=1_202,
        mask_xyxy=(3, 4, 10, 12),
    )

    assert decoded.shape == (2, features.FRAME_SIZE, features.FRAME_SIZE, 3)
    assert np.array_equal(decoded[0], frame)
    assert np.array_equal(decoded[1], frame)
    assert imputed == 1
    command = commands[0]
    assert command[command.index("-ss") + 1] == "20.000000000"
    assert command[command.index("-frames:v") + 1] == "2"
    filter_graph = command[command.index("-vf") + 1]
    assert "fps=fps=60:round=near" in filter_graph
    assert "drawbox=x=3:y=4:w=7:h=8" in filter_graph


def test_timestamp_resample_rejects_a_large_missing_tail(monkeypatch) -> None:
    frame = np.zeros(
        (features.FRAME_SIZE, features.FRAME_SIZE, 3), dtype=np.uint8
    )

    class FakeProcess:
        stdout = io.BytesIO(frame.tobytes())
        stderr = io.BytesIO()

        @staticmethod
        def wait() -> int:
            return 0

    monkeypatch.setattr(
        features.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess()
    )

    with pytest.raises(RuntimeError, match="produced 1/5 frames"):
        features._decode_resampled_part(
            Path("variable.mp4"),
            start_frame=0,
            end_frame=5,
            mask_xyxy=(0, 0, 1, 1),
        )


def test_timestamp_resample_drains_and_rejects_extra_output(monkeypatch) -> None:
    frame = np.zeros(
        (features.FRAME_SIZE, features.FRAME_SIZE, 3), dtype=np.uint8
    )

    class FakeProcess:
        stdout = io.BytesIO(frame.tobytes() * 3)
        stderr = io.BytesIO()

        @staticmethod
        def wait() -> int:
            return 0

    monkeypatch.setattr(
        features.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess()
    )

    with pytest.raises(RuntimeError, match="extra raw-video bytes"):
        features._decode_resampled_part(
            Path("variable.mp4"),
            start_frame=0,
            end_frame=2,
            mask_xyxy=(0, 0, 1, 1),
        )
