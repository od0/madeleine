from __future__ import annotations

import subprocess

from harvest.scan_corpus import format_probe_error


def test_probe_error_preserves_captured_platform_reason() -> None:
    error = subprocess.CalledProcessError(
        1,
        ["yt-dlp", "https://example.invalid/video"],
        output=b"",
        stderr=b"ERROR: Video unavailable. It was blocked due to a copyright claim.\n",
    )

    detail = format_probe_error(error)

    assert detail.startswith("CalledProcessError:")
    assert "Video unavailable" in detail
    assert "copyright claim" in detail


def test_probe_error_is_bounded() -> None:
    error = subprocess.CalledProcessError(
        1, ["yt-dlp"], stderr=("x" * 5000).encode()
    )

    assert len(format_probe_error(error, limit=300)) == 300
