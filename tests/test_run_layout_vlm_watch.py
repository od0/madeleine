from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).parents[1] / "harvest" / "run_layout_vlm_watch.sh"
pytestmark = pytest.mark.requires_private_artifacts(
    "harvest/run_layout_vlm_watch.sh"
)


def make_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def run_watcher(tmp_path: Path, *, rows: int, unique: int) -> subprocess.CompletedProcess:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        '{"video_id":"video_1","human_reviewed":false,'
        '"training_admitted":false}\n'
    )
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make_executable(
        fake_bin / "rclone",
        """#!/usr/bin/env bash
if [[ "$1" == "copy" ]]; then
  exit 0
fi
if [[ "$1" == "lsf" ]]; then
  printf 'video_1/survey_complete.json\\n'
  exit 0
fi
exit 2
""",
    )
    fake_python = fake_bin / "python"
    make_executable(
        fake_python,
        """#!/usr/bin/env bash
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--out" ]]; then
    out=$2
    shift 2
  else
    shift
  fi
done
mkdir -p "$(dirname "$out")"
: > "$out"
printf '{"predictions":{"rows":%s,"unique_video_ids":%s}}\\n' \
  "$FAKE_ROWS" "$FAKE_UNIQUE" > "$out.manifest.json"
""",
    )
    environment = os.environ.copy()
    environment.update({
        "PATH": f"{fake_bin}:{environment['PATH']}",
        "WILD20_VLM_PYTHON": str(fake_python),
        "WILD20_VLM_WATCH_PASSES": "1",
        "WILD20_VLM_WATCH_SLEEP_S": "1",
        "FAKE_ROWS": str(rows),
        "FAKE_UNIQUE": str(unique),
    })
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(queue),
            str(tmp_path / "work"),
            str(code_dir),
            "1",
        ],
        text=True,
        capture_output=True,
        env=environment,
        timeout=10,
        check=False,
    )


def test_watcher_uses_validated_unique_prediction_count(tmp_path: Path) -> None:
    result = run_watcher(tmp_path, rows=1, unique=1)
    assert result.returncode == 0, result.stderr
    assert "predictions=1 target=1" in result.stdout


def test_watcher_rejects_nonunique_prediction_count(tmp_path: Path) -> None:
    result = run_watcher(tmp_path, rows=2, unique=1)
    assert result.returncode == 2
    assert "validated prediction count is unavailable" in result.stderr


def test_watcher_preflights_runtime_dependencies() -> None:
    script = SCRIPT.read_text()
    assert "WILD20_VLM_WATCH_PASSES:-720" in script
    assert "required command is unavailable" in script
    assert '[[ ! -x "$python_bin" ]]' in script
    assert "WILD20_VLM_WATCH_PASSES must be a positive integer" in script
    assert "WILD20_VLM_WATCH_SLEEP_S must be a positive integer" in script
