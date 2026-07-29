from __future__ import annotations

from pathlib import Path

import pytest

from badeline.train import _git_describe


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "experiments/run_tcn_control_lr_study.sh"
requires_scheduler = pytest.mark.requires_private_artifacts(
    "experiments/run_tcn_control_lr_study.sh"
)


@requires_scheduler
def test_scheduler_runs_one_job_per_gpu_and_queues_third() -> None:
    text = SCHEDULER.read_text()

    assert "GPU=0" in text
    assert "run_lr 1 1e-4" in text
    assert "wait -n -p finished_pid" in text
    assert 'run_lr "${free_gpu}" 1e-3' in text
    assert "run_vptlite_tcn_natural_holdout.sh" in text
    assert "run_vptlite_tcn_lr_sensitivity.sh" in text
    assert "eval_b1" not in text.lower()
    assert "b1_eval" not in text.lower()


@requires_scheduler
def test_scheduler_marks_success_only_after_all_children() -> None:
    text = SCHEDULER.read_text()

    failure_gate = text.index("if (( first_status != 0")
    marker = text.index('touch "${study_marker}"')
    assert failure_gate < marker
    assert ".tcn_control_lr_y4n_study_done" in text


def test_declared_source_commit_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    commit = "a" * 40
    monkeypatch.setenv("MADELEINE_SOURCE_COMMIT", commit)
    assert _git_describe() == f"{commit}-declared"

    monkeypatch.setenv("MADELEINE_SOURCE_COMMIT", "abc")
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        _git_describe()
