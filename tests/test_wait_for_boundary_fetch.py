from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "results"
    / "wild"
    / "general-harvest-twitch-fanout-20260727T233330Z"
    / "ops"
    / "wait_for_boundary_fetch.sh"
)
pytestmark = pytest.mark.requires_private_artifacts(
    "results/wild/general-harvest-twitch-fanout-20260727T233330Z/ops/"
    "wait_for_boundary_fetch.sh"
)


def make_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def prepare_runtime(tmp_path: Path) -> dict[str, Path | str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make_executable(
        fake_bin / "realpath",
        """#!/usr/bin/env bash
if [[ "${1:-}" == "-m" ]]; then
  shift
fi
if [[ "$1" = /* ]]; then
  printf '%s\n' "$1"
else
  printf '%s/%s\n' "$PWD" "$1"
fi
""",
    )
    make_executable(fake_bin / "flock", "#!/usr/bin/env bash\nexit 0\n")
    make_executable(
        fake_bin / "sha256sum",
        "#!/usr/bin/env bash\nexec /usr/bin/shasum -a 256 \"$1\"\n",
    )
    make_executable(
        fake_bin / "rclone",
        """#!/usr/bin/env bash
if [[ "$1" != "cat" ]]; then
  exit 2
fi
# Reproduce the backend edge case: absent object, exit zero, zero bytes.
if [[ -s "${FAKE_REMOTE_MARKER:-}" ]]; then
  exec /bin/cat "$FAKE_REMOTE_MARKER"
fi
exit 0
""",
    )

    code = tmp_path / "code"
    ops = code / "ops"
    harvest = code / "harvest"
    ops.mkdir(parents=True)
    harvest.mkdir()
    copied_script = ops / SCRIPT.name
    shutil.copy2(SCRIPT, copied_script)
    copied_script.chmod(0o755)
    invoked = tmp_path / "wrapper-invoked.txt"
    make_executable(
        harvest / "run_fetch_recovery_durable.sh",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" > \"$FAKE_INVOKED\"\n",
    )
    executables = []
    for name in ("python", "yt-dlp", "deno"):
        path = fake_bin / name
        make_executable(path, "#!/usr/bin/env bash\nexit 0\n")
        executables.append(path)

    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps(
            {
                "video_id": "twitch_tail",
                "source": "twitch",
                "nominal_hours": 1.0,
            }
        )
        + "\n"
    )
    queue_sha = hashlib.sha256(queue.read_bytes()).hexdigest()
    marker = tmp_path / "remote-marker.json"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "FAKE_INVOKED": str(invoked),
            "FAKE_REMOTE_MARKER": str(marker),
            "WILD20_BOUNDARY_POLL_SECONDS": "1",
            "WILD20_CODE_DIR": str(code),
        }
    )
    command = [
        "bash",
        str(copied_script),
        "fake:boundary.json",
        "a" * 64,
        "v1850996198",
        str(queue),
        str(tmp_path / "work"),
        "twitch",
        *(str(path) for path in executables),
        queue_sha,
    ]
    return {
        "command": command,
        "environment": environment,
        "invoked": invoked,
        "marker": marker,
        "work": tmp_path / "work",
        "queue_sha": queue_sha,
    }


def test_empty_successful_rclone_cat_does_not_open_gate(tmp_path: Path) -> None:
    runtime = prepare_runtime(tmp_path)
    process = subprocess.Popen(
        runtime["command"],
        env=runtime["environment"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(1.3)
        assert process.poll() is None
        assert not Path(runtime["invoked"]).exists()
        watchdog = json.loads(
            (Path(runtime["work"]) / "gate/status/watchdog.json").read_text()
        )
        assert watchdog["phase"] == "waiting_for_boundary_marker"
        assert not (
            Path(runtime["work"]) / "gate/status/boundary_observed.json"
        ).exists()
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_nonempty_provenance_bound_marker_opens_gate(tmp_path: Path) -> None:
    runtime = prepare_runtime(tmp_path)
    Path(runtime["marker"]).write_text(
        json.dumps(
            {
                "format_version": "madeleine.fetch-boundary.v1",
                "queue_sha256": "a" * 64,
                "video_id": "v1850996198",
                "progress_index": 4,
                "predecessor_finished_sha256": "b" * 64,
            }
        )
        + "\n"
    )

    result = subprocess.run(
        runtime["command"],
        env=runtime["environment"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert Path(runtime["invoked"]).is_file()
    gate_status = Path(runtime["work"]) / "gate/status"
    assert (gate_status / "boundary_observed.json").is_file()
    assert (gate_status / "launched.json").is_file()
    assert json.loads((gate_status / "finished.json").read_text())["fetch_rc"] == 0
