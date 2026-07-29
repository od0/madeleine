import threading

import experiments.audit_corpus_contiguity as audit
from experiments.audit_corpus_contiguity import LabelRun, VideoPlan, _label_runs


def _row(chunk_id: int, *, grid_hz: float = 60.0, size: int = 1200) -> dict:
    return {"chunk_id": chunk_id, "grid_hz": grid_hz, "chunk_size": size}


def test_label_runs_join_consecutive_chunks_and_break_on_gap() -> None:
    runs = _label_runs([
        _row(4),
        _row(2),
        _row(3),
        _row(7, size=600),
    ])

    assert [(run.start_s, run.duration_s, run.expected_frames) for run in runs] == [
        (40.0, 60.0, 3600),
        (140.0, 10.0, 600),
    ]


def test_label_runs_exclude_non_60hz_groups() -> None:
    runs = _label_runs([_row(0, grid_hz=30.0), _row(1, grid_hz=30.0)])

    assert runs == []


def test_scan_run_starts_stderr_drain_before_waiting_on_stdout(monkeypatch) -> None:
    stderr_started = threading.Event()

    class FakeStderr:
        exhausted = False

        def read(self, _size: int) -> bytes:
            if self.exhausted:
                return b""
            self.exhausted = True
            stderr_started.set()
            return b"repeated decoder diagnostic\n"

    class FakeStdout:
        exhausted = False

        def read(self, _size: int) -> bytes:
            assert stderr_started.wait(1.0), "stderr was not drained concurrently"
            if self.exhausted:
                return b""
            self.exhausted = True
            return b"\x00" * 4

    class FakeProcess:
        stdout = FakeStdout()
        stderr = FakeStderr()

        @staticmethod
        def wait() -> int:
            return 0

    monkeypatch.setattr(audit.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    plan = VideoPlan(
        video_id="test",
        path="unused.mp4",
        bytes=0,
        width=2,
        height=2,
        fps=60.0,
        label_frames=1,
        label_hours=1 / 216_000.0,
        label_run_count=1,
        long_context_targets=0,
        long_context_fraction=0.0,
        mask_rect=None,
        runs=(),
    )

    result = audit._scan_run(
        plan,
        LabelRun(start_s=0.0, duration_s=1 / 60.0, expected_frames=1),
        gpu=0,
        width=2,
        height=2,
        near_threshold=0.5,
    )

    assert result["decoded_frames"] == 1
