from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT / "experiments/configs/vptlite_tcn_26m_128x3frame_full_holdout.json"
)
CONTROL_CONFIG = (
    ROOT
    / "experiments/configs/vptlite_tcn_natural_26m_128x3frame_full_holdout.json"
)
CONTROL_LAUNCHER = ROOT / "experiments/run_vptlite_tcn_natural_holdout.sh"


def test_natural_control_changes_only_the_state_objective() -> None:
    baseline = json.loads(BASE_CONFIG.read_text())
    control = json.loads(CONTROL_CONFIG.read_text())

    differing = {
        key
        for key in baseline.keys() | control.keys()
        if baseline.get(key) != control.get(key)
    }
    assert differing == {"_note", "class_balance", "transition_weight"}
    assert control["class_balance"] is False
    assert control["transition_weight"] == 1.0
    assert control["learning_rate"] == 3e-4
    assert control["seed"] == 0
    assert control["temporal_arch"] == "aligned_tcn"
    assert "event_latch" not in control
    assert not any(key.startswith("event_") for key in control)
    assert not any(key.endswith("_loss_weight") for key in control)


@pytest.mark.requires_private_artifacts(
    "experiments/run_vptlite_tcn_natural_holdout.sh"
)
def test_natural_control_launcher_is_final_y4n_only() -> None:
    launcher = CONTROL_LAUNCHER.read_text()

    assert "GPU:-0" in launcher
    assert "nitrogen_unflagged_92train_y4n_holdout_26m_vptlite_tcn_natural_s0" in launcher
    assert "max_steps != 14265" in launcher
    assert "-m badeline.train" in launcher
    assert "eval_mapped_foreign.py" in launcher
    assert "--weights final" in launcher
    assert "--weights selected" not in launcher
    assert "eval_event" not in launcher
    assert "run_event" not in launcher
    assert "b1_used\": False" in launcher
    assert "VALIDATION_ONLY" in launcher
    assert 'f"{session_id}__stream000"' in launcher
