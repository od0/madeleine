from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "harvest" / "run_layout_survey_watch.sh"
pytestmark = pytest.mark.requires_private_artifacts(
    "harvest/run_layout_survey_watch.sh"
)


def test_watcher_is_r2_gated_and_single_owner() -> None:
    source = SCRIPT.read_text()
    assert "upload_complete.json" in source
    assert "survey_complete.json" in source
    assert "flock -n 9" in source
    assert 'pids+=("$!")' in source
    assert "--exclude-ids" in source
