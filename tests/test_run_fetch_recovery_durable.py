from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = (
    Path(__file__).parents[1] / "harvest" / "run_fetch_recovery_durable.sh"
)
pytestmark = pytest.mark.requires_private_artifacts(
    "harvest/run_fetch_recovery_durable.sh"
)


def make_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def prepare_runtime(tmp_path: Path) -> dict[str, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make_executable(
        fake_bin / "realpath",
        """#!/usr/bin/env bash
if [[ "${1:-}" == "-m" ]]; then
  shift
fi
printf '%s\n' "$1"
""",
    )
    make_executable(
        fake_bin / "flock",
        "#!/usr/bin/env bash\nexit \"${FAKE_FLOCK_RC:-0}\"\n",
    )
    for command in ("rclone", "ffprobe"):
        make_executable(fake_bin / command, "#!/usr/bin/env bash\nexit 0\n")
    make_executable(
        fake_bin / "sha256sum",
        "#!/usr/bin/env bash\nprintf '%064d  %s\\n' 0 \"$1\"\n",
    )

    base_python = fake_bin / "base-python"
    make_executable(
        base_python,
        """#!/usr/bin/env bash
printf '%s\n' "$0" > "$FAKE_INVOCATION_FILE"
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--progress" ]]; then
    progress=$2
    shift 2
  else
    shift
  fi
done
mkdir -p "$(dirname "$progress")"
printf '{"status":"ok"}\n' > "$progress"
""",
    )
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.symlink_to(base_python)

    yt_dlp = fake_bin / "yt-dlp"
    deno = fake_bin / "deno"
    make_executable(yt_dlp, "#!/usr/bin/env bash\nexit 0\n")
    make_executable(deno, "#!/usr/bin/env bash\nexit 0\n")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    return {
        "fake_bin": fake_bin,
        "python": venv_python,
        "yt_dlp": yt_dlp,
        "deno": deno,
        "code_dir": code_dir,
    }


def write_queue(path: Path, *sources: str) -> None:
    path.write_text(
        "".join(
            json.dumps({"video_id": f"video_{index}", "source": source}) + "\n"
            for index, source in enumerate(sources)
        )
    )


def run_wrapper(
    tmp_path: Path,
    runtime: dict[str, Path],
    queue: Path,
    *,
    source: str = "youtube",
    python: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{runtime['fake_bin']}:{environment['PATH']}",
            "WILD20_CODE_DIR": str(runtime["code_dir"]),
            "WILD20_RAW_ROOT": str(tmp_path / "raw"),
            "WILD20_REMOTE_RAW": "test-remote:raw",
            "WILD20_WATCHDOG_SECONDS": "1",
            "FAKE_INVOCATION_FILE": str(tmp_path / "invoked-python.txt"),
        }
    )
    if extra_env:
        environment.update(extra_env)
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(queue),
            str(tmp_path / "work"),
            source,
            str(python or runtime["python"]),
            str(runtime["yt_dlp"]),
            str(runtime["deno"]),
        ],
        text=True,
        capture_output=True,
        env=environment,
        timeout=10,
        check=False,
    )


def test_preserves_venv_python_symlink_entrypoint(tmp_path: Path) -> None:
    runtime = prepare_runtime(tmp_path)
    queue = tmp_path / "queue.jsonl"
    write_queue(queue, "youtube")

    result = run_wrapper(tmp_path, runtime, queue)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "invoked-python.txt").read_text().strip() == str(
        runtime["python"]
    )
    finished = json.loads(
        (tmp_path / "work" / "status" / "finished.json").read_text()
    )
    assert finished["worker_rc"] == 0
    assert finished["progress_counts"] == {"ok": 1}


def test_rejects_mixed_source_queue_before_starting_worker(tmp_path: Path) -> None:
    runtime = prepare_runtime(tmp_path)
    queue = tmp_path / "queue.jsonl"
    write_queue(queue, "youtube", "twitch")

    result = run_wrapper(tmp_path, runtime, queue)

    assert result.returncode == 2
    assert "queue is not exclusively source=youtube" in result.stderr
    assert not (tmp_path / "invoked-python.txt").exists()
    assert not (tmp_path / "work" / "status" / "started.json").exists()


def test_rejects_missing_executable_during_preflight(tmp_path: Path) -> None:
    runtime = prepare_runtime(tmp_path)
    queue = tmp_path / "queue.jsonl"
    write_queue(queue, "youtube")
    missing_python = tmp_path / "missing-venv" / "bin" / "python"

    result = run_wrapper(tmp_path, runtime, queue, python=missing_python)

    assert result.returncode == 2
    assert f"required executable is unavailable: {missing_python}" in result.stderr
    assert not (tmp_path / "work").exists()


def test_rejects_queue_hash_that_differs_from_expected(tmp_path: Path) -> None:
    runtime = prepare_runtime(tmp_path)
    queue = tmp_path / "queue.jsonl"
    write_queue(queue, "youtube")

    result = run_wrapper(
        tmp_path,
        runtime,
        queue,
        extra_env={"WILD20_EXPECTED_QUEUE_SHA256": "f" * 64},
    )

    assert result.returncode == 2
    assert "queue SHA-256 differs" in result.stderr
    assert not (tmp_path / "invoked-python.txt").exists()
    assert not (tmp_path / "work").exists()


def test_contended_host_lock_fails_without_starting_worker(tmp_path: Path) -> None:
    runtime = prepare_runtime(tmp_path)
    queue = tmp_path / "queue.jsonl"
    write_queue(queue, "youtube")

    result = run_wrapper(
        tmp_path,
        runtime,
        queue,
        extra_env={"FAKE_FLOCK_RC": "1"},
    )

    assert result.returncode == 3
    assert "another fetch recovery worker owns this public-IP host" in result.stderr
    assert not (tmp_path / "invoked-python.txt").exists()
    assert not (tmp_path / "work" / "status" / "started.json").exists()
