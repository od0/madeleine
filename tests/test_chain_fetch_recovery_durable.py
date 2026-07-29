from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = (
    Path(__file__).parents[1] / "harvest" / "chain_fetch_recovery_durable.sh"
)
requires_chain_script = pytest.mark.requires_private_artifacts(
    "harvest/chain_fetch_recovery_durable.sh"
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
        """#!/usr/bin/env bash
if [[ "$*" == "-n 9" ]]; then
  exit "${FAKE_HOST_FLOCK_RC:-0}"
fi
exit "${FAKE_CHAIN_FLOCK_RC:-0}"
""",
    )
    make_executable(
        fake_bin / "rclone",
        """#!/usr/bin/env bash
if [[ "$1" != "lsf" ]]; then
  exit 0
fi
video_id=${2##*/}
case ",${FAKE_R2_MISSING_IDS:-}," in
  *",$video_id,"*) exit 0 ;;
esac
printf 'upload_complete.json\n'
""",
    )
    make_executable(fake_bin / "ffprobe", "#!/usr/bin/env bash\nexit 0\n")
    make_executable(
        fake_bin / "sha256sum",
        "#!/usr/bin/env bash\nLC_ALL=C exec /usr/bin/shasum -a 256 \"$1\"\n",
    )

    base_python = fake_bin / "base-python"
    make_executable(
        base_python,
        """#!/usr/bin/env bash
printf '%s\n' "$0" > "$FAKE_INVOCATION_FILE"
printf '%s\n' "$@" > "$FAKE_ARGUMENT_FILE"
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


def write_queue(path: Path, source: str = "youtube") -> str:
    path.write_text(
        json.dumps(
            {
                "video_id": "video_0",
                "source": source,
                "url": "https://example.invalid/video_0",
                "duration_s": 60.0,
            }
        )
        + "\n"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_started(
    path: Path,
    pid: int,
    video_ids: tuple[str, ...] = ("predecessor_0",),
) -> None:
    status = path / "status"
    status.mkdir(parents=True, exist_ok=True)
    queue = path / "queue.jsonl"
    queue.write_text(
        "".join(
            json.dumps({"video_id": video_id, "source": "youtube"}) + "\n"
            for video_id in video_ids
        )
    )
    (status / "started.json").write_text(
        json.dumps(
            {
                "supervisor_pid": pid,
                "queue": str(queue),
                "queue_sha256": hashlib.sha256(queue.read_bytes()).hexdigest(),
            }
        )
    )


def environment(tmp_path: Path, runtime: dict[str, Path]) -> dict[str, str]:
    result = os.environ.copy()
    result.update(
        {
            "PATH": f"{runtime['fake_bin']}:{result['PATH']}",
            "WILD20_CODE_DIR": str(runtime["code_dir"]),
            "WILD20_RAW_ROOT": str(tmp_path / "raw"),
            "WILD20_REMOTE_RAW": "test-remote:raw",
            "WILD20_WATCHDOG_SECONDS": "1",
            "WILD20_CHAIN_POLL_SECONDS": "1",
            "WILD20_CHAIN_MAX_WAIT_SECONDS": "5",
            "FAKE_INVOCATION_FILE": str(tmp_path / "invoked-python.txt"),
            "FAKE_ARGUMENT_FILE": str(tmp_path / "python-arguments.txt"),
        }
    )
    return result


def chain_command(
    tmp_path: Path,
    runtime: dict[str, Path],
    predecessor: Path,
    queue: Path,
    digest: str,
) -> list[str]:
    return [
        "bash",
        str(SCRIPT),
        str(predecessor),
        str(queue),
        str(tmp_path / "successor"),
        "youtube",
        str(runtime["python"]),
        str(runtime["yt_dlp"]),
        str(runtime["deno"]),
        digest,
    ]


@requires_chain_script
def test_waits_for_predecessor_and_preserves_venv_entrypoint(
    tmp_path: Path,
) -> None:
    runtime = prepare_runtime(tmp_path)
    queue = tmp_path / "queue.jsonl"
    digest = write_queue(queue)
    predecessor = tmp_path / "predecessor"
    status = predecessor / "status"
    status.mkdir(parents=True)
    predecessor_process = subprocess.Popen(
        [
            "bash",
            "-c",
            "sleep 0.2; printf '{\"worker_rc\":0}\\n' > \"$1\"",
            "predecessor",
            str(status / "finished.json"),
        ]
    )
    write_started(predecessor, predecessor_process.pid)

    result = subprocess.run(
        chain_command(tmp_path, runtime, predecessor, queue, digest),
        text=True,
        capture_output=True,
        env=environment(tmp_path, runtime),
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "invoked-python.txt").read_text().strip() == str(
        runtime["python"]
    )
    arguments = (tmp_path / "python-arguments.txt").read_text().splitlines()
    assert arguments[:2] == ["-m", "harvest.fetch_fleet_worker"]
    started = json.loads(
        (tmp_path / "successor" / "status" / "chain_started.json").read_text()
    )
    assert started["queue_sha256"] == digest
    assert started["python"] == str(runtime["python"])
    assert started["semantics"] == (
        "R2-completion-marker missing-only via durable wrapper"
    )
    assert (tmp_path / "successor" / "status" / "started.json").is_file()
    finished = json.loads(
        (tmp_path / "successor" / "status" / "chain_finished.json").read_text()
    )
    assert finished["status"] == "successor_finished"
    assert finished["successor_rc"] == 0


@requires_chain_script
def test_bounded_wait_never_launches_while_predecessor_is_live(
    tmp_path: Path,
) -> None:
    runtime = prepare_runtime(tmp_path)
    queue = tmp_path / "queue.jsonl"
    digest = write_queue(queue)
    predecessor = tmp_path / "predecessor"
    predecessor_process = subprocess.Popen(["sleep", "10"])
    write_started(predecessor, predecessor_process.pid)
    env = environment(tmp_path, runtime)
    env["WILD20_CHAIN_MAX_WAIT_SECONDS"] = "1"
    try:
        result = subprocess.run(
            chain_command(tmp_path, runtime, predecessor, queue, digest),
            text=True,
            capture_output=True,
            env=env,
            timeout=5,
            check=False,
        )
    finally:
        predecessor_process.terminate()
        predecessor_process.wait(timeout=5)

    assert result.returncode == 4
    assert "timed out waiting for predecessor to exit" in result.stderr
    assert not (tmp_path / "invoked-python.txt").exists()
    assert not (
        tmp_path / "successor" / "status" / "chain_launched.json"
    ).exists()
    finished = json.loads(
        (tmp_path / "successor" / "status" / "chain_finished.json").read_text()
    )
    assert finished["status"] == "wait_timeout"


@requires_chain_script
def test_rejects_changed_queue_before_waiting_or_launching(tmp_path: Path) -> None:
    runtime = prepare_runtime(tmp_path)
    queue = tmp_path / "queue.jsonl"
    digest = write_queue(queue)
    queue.write_text(queue.read_text() + "\n")
    predecessor = tmp_path / "predecessor"
    write_started(predecessor, os.getpid())

    result = subprocess.run(
        chain_command(tmp_path, runtime, predecessor, queue, digest),
        text=True,
        capture_output=True,
        env=environment(tmp_path, runtime),
        timeout=5,
        check=False,
    )

    assert result.returncode == 2
    assert "queue copy SHA-256 differs" in result.stderr
    assert not (tmp_path / "invoked-python.txt").exists()
    finished = json.loads(
        (tmp_path / "successor" / "status" / "chain_finished.json").read_text()
    )
    assert finished["status"] == "queue_hash_mismatch"


@requires_chain_script
def test_existing_wrapper_host_lock_fails_closed_without_fetch_worker(
    tmp_path: Path,
) -> None:
    runtime = prepare_runtime(tmp_path)
    queue = tmp_path / "queue.jsonl"
    digest = write_queue(queue)
    predecessor = tmp_path / "predecessor"
    finished_process = subprocess.run(["true"], check=True)
    del finished_process
    write_started(predecessor, 99999999)
    (predecessor / "status" / "finished.json").write_text(
        json.dumps({"worker_rc": 0})
    )
    env = environment(tmp_path, runtime)
    env["FAKE_HOST_FLOCK_RC"] = "1"

    result = subprocess.run(
        chain_command(tmp_path, runtime, predecessor, queue, digest),
        text=True,
        capture_output=True,
        env=env,
        timeout=5,
        check=False,
    )

    assert result.returncode == 3
    assert not (tmp_path / "invoked-python.txt").exists()
    worker_log = (
        tmp_path / "successor" / "logs" / "chained_worker.log"
    ).read_text()
    assert "another fetch recovery worker owns this public-IP host" in worker_log
    finished = json.loads(
        (tmp_path / "successor" / "status" / "chain_finished.json").read_text()
    )
    assert finished["successor_rc"] == 3


@requires_chain_script
def test_source_blocked_predecessor_tail_must_be_complete_in_r2(
    tmp_path: Path,
) -> None:
    runtime = prepare_runtime(tmp_path)
    queue = tmp_path / "queue.jsonl"
    digest = write_queue(queue)
    predecessor = tmp_path / "predecessor"
    write_started(
        predecessor,
        99999999,
        video_ids=("predecessor_done", "predecessor_tail"),
    )
    # The durable wrapper can exit rc=0 after fetch_fleet_worker trips its
    # source-block gate, leaving the remainder deliberately unfinished.
    (predecessor / "status" / "finished.json").write_text(
        json.dumps({"worker_rc": 0, "progress_counts": {"error": 1}})
    )
    env = environment(tmp_path, runtime)
    env["FAKE_R2_MISSING_IDS"] = "predecessor_tail"

    result = subprocess.run(
        chain_command(tmp_path, runtime, predecessor, queue, digest),
        text=True,
        capture_output=True,
        env=env,
        timeout=5,
        check=False,
    )

    assert result.returncode == 4
    assert not (tmp_path / "invoked-python.txt").exists()
    assert not (
        tmp_path / "successor" / "status" / "chain_launched.json"
    ).exists()
    audit = json.loads(
        (
            tmp_path
            / "successor"
            / "status"
            / "predecessor_completion_audit.json"
        ).read_text()
    )
    assert audit["predecessor_rows"] == 2
    assert audit["missing_count"] == 1
    assert audit["missing_video_ids"] == ["predecessor_tail"]
    assert audit["audit_error_count"] == 0
    finished = json.loads(
        (tmp_path / "successor" / "status" / "chain_finished.json").read_text()
    )
    assert finished == {
        "finished_at": finished["finished_at"],
        "status": "predecessor_queue_incomplete",
        "queue_sha256": digest,
        "chain_rc": 4,
        "missing_count": 1,
        "missing_video_ids": ["predecessor_tail"],
    }


@requires_chain_script
def test_successor_owned_policy_launches_when_every_predecessor_gap_is_owned(
    tmp_path: Path,
) -> None:
    runtime = prepare_runtime(tmp_path)
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps(
            {
                "video_id": "predecessor_tail",
                "source": "youtube",
                "url": "https://example.invalid/predecessor_tail",
                "duration_s": 60.0,
            }
        )
        + "\n"
    )
    digest = hashlib.sha256(queue.read_bytes()).hexdigest()
    predecessor = tmp_path / "predecessor"
    write_started(
        predecessor,
        99999999,
        video_ids=("predecessor_done", "predecessor_tail"),
    )
    (predecessor / "status" / "finished.json").write_text(
        json.dumps({"worker_rc": 0, "progress_counts": {"error": 1}})
    )
    env = environment(tmp_path, runtime)
    env["FAKE_R2_MISSING_IDS"] = "predecessor_tail"
    env["WILD20_PREDECESSOR_GAP_POLICY"] = "successor_owned"

    result = subprocess.run(
        chain_command(tmp_path, runtime, predecessor, queue, digest),
        text=True,
        capture_output=True,
        env=env,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "invoked-python.txt").is_file()
    ownership = json.loads(
        (
            tmp_path
            / "successor"
            / "status"
            / "predecessor_gap_ownership.json"
        ).read_text()
    )
    assert ownership["policy"] == "successor_owned"
    assert ownership["predecessor_missing_ids"] == ["predecessor_tail"]
    assert ownership["unowned_count"] == 0
    assert ownership["unowned_ids"] == []


@requires_chain_script
def test_successor_owned_policy_rejects_an_unowned_predecessor_gap(
    tmp_path: Path,
) -> None:
    runtime = prepare_runtime(tmp_path)
    queue = tmp_path / "queue.jsonl"
    digest = write_queue(queue)
    predecessor = tmp_path / "predecessor"
    write_started(
        predecessor,
        99999999,
        video_ids=("predecessor_done", "predecessor_tail"),
    )
    (predecessor / "status" / "finished.json").write_text(
        json.dumps({"worker_rc": 0, "progress_counts": {"error": 1}})
    )
    env = environment(tmp_path, runtime)
    env["FAKE_R2_MISSING_IDS"] = "predecessor_tail"
    env["WILD20_PREDECESSOR_GAP_POLICY"] = "successor_owned"

    result = subprocess.run(
        chain_command(tmp_path, runtime, predecessor, queue, digest),
        text=True,
        capture_output=True,
        env=env,
        timeout=5,
        check=False,
    )

    assert result.returncode == 4
    assert "predecessor gaps are absent" in result.stderr
    assert not (tmp_path / "invoked-python.txt").exists()
    ownership = json.loads(
        (
            tmp_path
            / "successor"
            / "status"
            / "predecessor_gap_ownership.json"
        ).read_text()
    )
    assert ownership["unowned_count"] == 1
    assert ownership["unowned_ids"] == ["predecessor_tail"]


@requires_chain_script
def test_rejects_duplicate_successor_queue_before_launch(tmp_path: Path) -> None:
    runtime = prepare_runtime(tmp_path)
    queue = tmp_path / "queue.jsonl"
    write_queue(queue)
    queue.write_text(queue.read_text() * 2)
    digest = hashlib.sha256(queue.read_bytes()).hexdigest()
    predecessor = tmp_path / "predecessor"
    write_started(predecessor, 99999999)
    (predecessor / "status" / "finished.json").write_text(
        json.dumps({"worker_rc": 0})
    )

    result = subprocess.run(
        chain_command(tmp_path, runtime, predecessor, queue, digest),
        text=True,
        capture_output=True,
        env=environment(tmp_path, runtime),
        timeout=5,
        check=False,
    )

    assert result.returncode == 2
    assert "queue is duplicated, unsafe" in result.stderr
    assert not (tmp_path / "invoked-python.txt").exists()
