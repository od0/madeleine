from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import harvest.fetch_fleet_worker as fleet
from harvest.fetch_wild import FetchPolicy


def candidate(video_id: str) -> dict:
    return {
        "video_id": video_id,
        "url": f"https://example.invalid/{video_id}",
        "duration_s": 60.0,
    }


def test_queue_rejects_duplicates_and_incomplete_rows(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        json.dumps(candidate("same")) + "\n" + json.dumps(candidate("same")) + "\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        fleet.load_queue(duplicate)

    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text(json.dumps({"video_id": "missing"}) + "\n")
    with pytest.raises(ValueError, match="lacks URL or duration"):
        fleet.load_queue(incomplete)


def test_worker_skips_remote_completion_and_continues_after_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [candidate("done"), candidate("good"), candidate("bad")]
    monkeypatch.setattr(
        fleet, "remote_complete", lambda remote: remote.endswith("/done")
    )
    monkeypatch.setattr(
        fleet.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3)
    )

    def run(candidate_row, out_root, remote_root, policy):
        assert isinstance(policy, FetchPolicy)
        if candidate_row["video_id"] == "bad":
            raise RuntimeError("source unavailable")
        return {"total_bytes": 123}

    monkeypatch.setattr(fleet, "run_worker", run)
    progress = tmp_path / "progress.jsonl"
    counts = fleet.run_queue(
        rows, tmp_path / "raw", "r2:bucket/raw", progress, FetchPolicy(), 20
    )
    assert counts == {"queued": 3, "ok": 1, "error": 1, "skipped": 1}
    saved = [json.loads(line) for line in progress.read_text().splitlines()]
    assert [row["video_id"] for row in saved] == ["good", "bad"]
    assert saved[0]["published_bytes"] == 123
    assert saved[1]["status"] == "error"
    assert "source unavailable" in saved[1]["error"]
    assert saved[1]["source_blocked"] is False


def test_worker_stops_on_ip_wide_source_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [candidate("blocked"), candidate("preserved")]
    monkeypatch.setattr(fleet, "remote_complete", lambda _: False)
    monkeypatch.setattr(
        fleet.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3)
    )

    def run(*_args, **_kwargs):
        raise fleet.subprocess.CalledProcessError(
            1,
            ["yt-dlp"],
            stderr="ERROR: Sign in to confirm you're not a bot",
        )

    monkeypatch.setattr(fleet, "run_worker", run)
    progress = tmp_path / "progress.jsonl"
    counts = fleet.run_queue(
        rows, tmp_path / "raw", "r2:bucket/raw", progress, FetchPolicy(), 20
    )
    assert counts == {"queued": 2, "ok": 0, "error": 1, "skipped": 0}
    saved = [json.loads(line) for line in progress.read_text().splitlines()]
    assert [row["video_id"] for row in saved] == ["blocked"]
    assert saved[0]["source_blocked"] is True
    assert "not a bot" in saved[0]["error"]


def test_verified_worker_can_remove_only_its_local_video_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [candidate("cleaned")]
    raw = tmp_path / "raw"
    local = raw / "cleaned"
    local.mkdir(parents=True)
    (local / "source.mp4").write_bytes(b"source")
    monkeypatch.setattr(fleet, "remote_complete", lambda _: False)
    monkeypatch.setattr(
        fleet.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3)
    )
    monkeypatch.setattr(
        fleet, "run_worker", lambda *_args, **_kwargs: {"total_bytes": 6}
    )

    counts = fleet.run_queue(
        rows,
        raw,
        "r2:bucket/raw",
        tmp_path / "progress.jsonl",
        FetchPolicy(),
        20,
        delete_local_after_verified=True,
    )
    assert counts["ok"] == 1
    assert not local.exists()
    assert raw.is_dir()


def test_remote_audit_fails_closed_on_rclone_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fleet.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="permission denied"
        ),
    )
    with pytest.raises(RuntimeError, match="permission denied"):
        fleet.remote_complete("r2:bucket/raw/video")
