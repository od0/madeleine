from pathlib import Path

import pytest


pytestmark = pytest.mark.requires_private_artifacts(
    "harvest/run_layout_survey_queue.sh"
)


def test_remote_completion_rejects_empty_rclone_success() -> None:
    script = (
        Path(__file__).parents[1] / "harvest" / "run_layout_survey_queue.sh"
    ).read_text()
    assert '[[ -n "$marker" ]] || return 1' in script
    assert 'type == "object"' in script
