from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ARTIFACT_REASON = "requires private data artifacts — see 'What is public'"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_private_artifacts(*paths): skip when any repository-relative "
        "private artifact path is absent",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    missing: list[str] = []
    for marker in item.iter_markers(name="requires_private_artifacts"):
        for value in marker.args:
            path = Path(str(value))
            resolved = path if path.is_absolute() else REPO_ROOT / path
            if not resolved.exists():
                missing.append(path.as_posix())
    if missing:
        pytest.skip(PRIVATE_ARTIFACT_REASON)
