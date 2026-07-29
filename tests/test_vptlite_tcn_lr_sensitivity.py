from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
BASE = ROOT / "experiments/configs/vptlite_tcn_26m_128x3frame_full_holdout.json"
VARIANTS = {
    "vptlite_tcn_26m_128x3frame_full_holdout_lr1e4.json": 1e-4,
    "vptlite_tcn_26m_128x3frame_full_holdout_lr1e3.json": 1e-3,
}


@pytest.mark.parametrize(("filename", "learning_rate"), VARIANTS.items())
def test_lr_sensitivity_config_changes_only_learning_rate_and_note(
    filename: str,
    learning_rate: float,
) -> None:
    base = json.loads(BASE.read_text())
    variant = json.loads((BASE.parent / filename).read_text())

    changed = {
        key for key in base.keys() | variant.keys()
        if base.get(key) != variant.get(key)
    }
    assert changed == {"_note", "learning_rate"}
    assert variant["learning_rate"] == learning_rate
    assert "Sensitivity-only" in variant["_note"]


@pytest.mark.requires_private_artifacts(
    "experiments/run_vptlite_tcn_lr_sensitivity.sh"
)
def test_lr_sensitivity_launcher_rejects_unregistered_rate() -> None:
    launcher = ROOT / "experiments/run_vptlite_tcn_lr_sensitivity.sh"
    result = subprocess.run(
        ["bash", str(launcher)],
        check=False,
        capture_output=True,
        env={"SENSITIVITY_LR": "0.0002"},
        text=True,
    )

    assert result.returncode != 0
    assert "SENSITIVITY_LR must be one of" in result.stderr


@pytest.mark.requires_private_artifacts(
    "experiments/run_vptlite_tcn_lr_sensitivity.sh"
)
@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"SENSITIVITY_LR": "1e-4", "COHORT": "all"}, "frozen to COHORT"),
        ({"SENSITIVITY_LR": "1e-4", "SEED": "1"}, "frozen to SEED"),
        (
            {"SENSITIVITY_LR": "1e-4", "HOLDOUT": "another-video"},
            "frozen to HOLDOUT",
        ),
    ],
)
def test_lr_sensitivity_launcher_rejects_recipe_drift(
    environment: dict[str, str],
    message: str,
) -> None:
    launcher = ROOT / "experiments/run_vptlite_tcn_lr_sensitivity.sh"
    result = subprocess.run(
        ["bash", str(launcher)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode != 0
    assert message in result.stderr
