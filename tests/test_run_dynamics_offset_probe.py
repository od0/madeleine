from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "experiments/run_dynamics_offset_probe.sh"
pytestmark = pytest.mark.requires_private_artifacts(
    "experiments/run_dynamics_offset_probe.sh"
)


def test_dynamics_offset_launcher_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_dynamics_offset_launcher_is_fail_closed_and_ordered() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for variable in (
        "OWN_V3_ROOT",
        "TRAIN_LIST",
        "VALIDATION_LIST",
        "CONTRACT",
        "OUTPUT_DIR",
        "CUDA_DEVICE",
        "MADELEINE_SOURCE_COMMIT",
    ):
        assert f"${{{variable}:?" in text

    assert "status --porcelain=v1 --untracked-files=all" in text
    assert "ls-files --error-unmatch" in text
    assert "HEAD does not match MADELEINE_SOURCE_COMMIT" in text
    assert "refusing to reuse output directory" in text
    assert 'mkdir -- "${output_dir}"' in text
    assert 'mkdir -p "${output_dir}"' not in text

    benchmark = text.index("--benchmark-only")
    benchmark_validation = text.index("validate_runtime_benchmark_receipt")
    real_probe = text.index('--data-dir "${own_v3_root}"')
    scorer = text.index("-m experiments.score_dynamics_offset_probe")
    assert benchmark < benchmark_validation < real_probe < scorer
    assert "--device cuda" in text
    assert "--null-device cuda" in text
    assert "CUDA_VISIBLE_DEVICES=${cuda_device}" in text
    assert "one nonnegative physical GPU index" in text
    assert "scoring_null_benchmark" not in text  # exact validator owns this detail

    forbidden_session = "rec_" + "20260727_220000_test"
    forbidden_store = "test-" + "untouched-v1"
    assert forbidden_session not in text
    assert forbidden_store not in text
    for forbidden_capability in ("ssh ", "rclone ", "aws ", "token", "secret"):
        assert forbidden_capability not in text.casefold()


def test_dynamics_offset_launcher_requires_explicit_paths_before_git_or_data() -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=False,
        capture_output=True,
        env={"PATH": os.environ["PATH"]},
        text=True,
    )
    assert result.returncode != 0
    assert "OWN_V3_ROOT must name the corrected own-v3 root" in result.stderr
