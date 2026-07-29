from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from nitrogen import refetch


VIDEO_SPECS = [
    ("SHARD_0000", "yt_ok", "celeste", "youtube", [1080, 1920], 2),
    ("SHARD_0000", "hk_youtube", "hollow_knight", "youtube", [720, 1280], 3),
    ("SHARD_0001", "yt_fail", "celeste", "youtube", [720, 1280], 3),
    ("SHARD_0001", "tw_one", "celeste", "twitch", [900, 1600], 2),
    ("SHARD_0002", "tw_two", "celeste", "twitch", [1080, 1920], 3),
    ("SHARD_0002", "hk_twitch", "hollow_knight", "twitch", [900, 1600], 2),
]


def _metadata(
    video_id: str,
    chunk_index: int,
    game: str,
    source: str,
    resolution: list[int],
) -> dict[str, Any]:
    start_time = chunk_index * 20
    return {
        "uuid": f"{video_id}_chunk_{chunk_index:04d}_actions",
        "chunk_id": f"{chunk_index:04d}",
        "chunk_size": 1200,
        "original_video": {
            "resolution": resolution,
            "video_id": video_id,
            "start_time": start_time,
            "end_time": start_time + 20,
            "duration": 20,
            "start_frame": chunk_index * 1200,
            "end_frame": (chunk_index + 1) * 1200 - 1,
            "source": source,
            "url": f"https://example.test/{source}/{video_id}",
        },
        "game": game,
        "controller_type": "xboxone",
        "bbox_controller_overlay": [10, 500, 320, 240],
    }


@pytest.fixture
def actions_root(tmp_path: Path) -> Path:
    root = tmp_path / "actions"
    for shard, video_id, game, source, resolution, n_chunks in VIDEO_SPECS:
        for chunk_index in range(n_chunks):
            chunk_dir = (
                root
                / shard
                / video_id
                / f"{video_id}_chunk_{chunk_index:04d}"
            )
            chunk_dir.mkdir(parents=True)
            metadata = _metadata(
                video_id, chunk_index, game, source, resolution
            )
            (chunk_dir / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
    return root


def test_discovery_is_per_video_and_preserves_height_width(
    actions_root: Path,
) -> None:
    records = refetch.discover_videos(actions_root, "celeste")

    assert {record["video_id"] for record in records} == {
        "yt_ok",
        "yt_fail",
        "tw_one",
        "tw_two",
    }
    by_id = {record["video_id"]: record for record in records}
    assert by_id["yt_ok"] == {
        "video_id": "yt_ok",
        "source": "youtube",
        "url": "https://example.test/youtube/yt_ok",
        "metadata_resolution": [1080, 1920],
        "n_chunks": 2,
        "chunk_hours": 2 * 20 / 3600,
    }
    assert by_id["yt_fail"]["n_chunks"] == 3
    assert by_id["tw_one"]["metadata_resolution"] == [900, 1600]
    assert all(record["source"] in {"youtube", "twitch"} for record in records)


def test_sampling_is_seeded_source_priority_and_video_level(
    actions_root: Path,
) -> None:
    records = refetch.discover_videos(actions_root, "celeste")

    first = refetch.sample_videos(records, 3, "youtube,twitch", seed=7)
    second = refetch.sample_videos(records, 3, "youtube,twitch", seed=7)

    assert first == second
    assert len(first) == 3
    assert [record["source"] for record in first] == [
        "youtube",
        "youtube",
        "twitch",
    ]
    assert len({record["video_id"] for record in first}) == 3


def _write_tiny_mp4(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (32, 24),
    )
    assert writer.isOpened()
    try:
        for value in (0, 80, 160):
            writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    finally:
        writer.release()


def _install_fake_downloader(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    calls = {"yt_ok": 0, "yt_fail": 0}
    options_seen: list[dict[str, Any]] = []

    class FakeYoutubeDL:
        def __init__(self, options: dict[str, Any]) -> None:
            self.options = options
            options_seen.append(options)

        def __enter__(self) -> FakeYoutubeDL:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, url: str, download: bool) -> dict[str, str]:
            assert download is True
            video_id = url.rsplit("/", maxsplit=1)[-1]
            calls[video_id] += 1
            if video_id == "yt_fail":
                raise RuntimeError("mock dead link")
            output = Path(self.options["outtmpl"].replace("%(ext)s", "mp4"))
            _write_tiny_mp4(output)
            return {"id": video_id, "ext": "mp4"}

    monkeypatch.setattr(refetch.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    return calls, options_seen


def _fetch_two_youtube(
    actions_root: Path,
    out_dir: Path,
) -> list[str]:
    return [
        "--actions-root",
        str(actions_root),
        "--game",
        "celeste",
        "--out",
        str(out_dir),
        "--n-videos",
        "2",
        "--source-priority",
        "youtube,twitch",
        "--seed",
        "7",
    ]


def _read_report(out_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (out_dir / refetch.REPORT_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_fetch_records_success_and_failure_without_aborting(
    actions_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "videos"
    calls, options_seen = _install_fake_downloader(monkeypatch)

    assert refetch.main(_fetch_two_youtube(actions_root, out_dir)) == 0

    reports = _read_report(out_dir)
    assert {report["status"] for report in reports} == {"ok", "failed"}
    by_status = {report["status"]: report for report in reports}
    assert "mock dead link" in by_status["failed"]["error"]
    assert by_status["failed"]["fetched_width"] is None
    assert by_status["failed"]["fetched_height"] is None
    assert by_status["failed"]["fetched_fps"] is None
    assert by_status["failed"]["fetched_frames"] is None
    assert by_status["failed"]["metadata_resolution"] == [720, 1280]
    assert by_status["ok"]["error"] is None
    assert by_status["ok"]["metadata_resolution"] == [1080, 1920]
    assert by_status["ok"]["fetched_width"] == 32
    assert by_status["ok"]["fetched_height"] == 24
    assert by_status["ok"]["fetched_fps"] == pytest.approx(12.0)
    assert by_status["ok"]["fetched_frames"] == 3
    assert calls == {"yt_ok": 1, "yt_fail": 1}
    assert all(option["format"] == refetch.DOWNLOAD_FORMAT for option in options_seen)
    assert all(option["retries"] == 2 for option in options_seen)
    assert all(
        option["retry_sleep_functions"]["http"](0) == 5
        and option["retry_sleep_functions"]["http"](1) == 15
        for option in options_seen
    )
    assert all(
        option["socket_timeout"] == refetch.SOCKET_TIMEOUT_SECONDS
        for option in options_seen
    )


def test_idempotent_rerun_skips_ok_and_retries_failed(
    actions_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "videos"
    calls, _ = _install_fake_downloader(monkeypatch)
    arguments = _fetch_two_youtube(actions_root, out_dir)

    assert refetch.main(arguments) == 0
    assert refetch.main(arguments) == 0

    reports = _read_report(out_dir)
    assert calls == {"yt_ok": 1, "yt_fail": 2}
    assert [report["video_id"] for report in reports].count("yt_ok") == 1
    assert [report["video_id"] for report in reports].count("yt_fail") == 2
    assert [report["status"] for report in reports].count("ok") == 1
    assert [report["status"] for report in reports].count("failed") == 2


def test_dry_run_prints_sample_and_writes_nothing(
    actions_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "videos"

    def unexpected_downloader(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run must not construct a downloader")

    monkeypatch.setattr(refetch.yt_dlp, "YoutubeDL", unexpected_downloader)
    arguments = _fetch_two_youtube(actions_root, out_dir) + ["--dry-run"]

    assert refetch.main(arguments) == 0

    output = capsys.readouterr().out
    assert "yt_ok" in output
    assert "yt_fail" in output
    assert not out_dir.exists()
