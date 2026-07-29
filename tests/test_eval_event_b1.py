from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.eval_event_b1 import B1_SESSION_ID, validate_b1_manifest


def _manifest() -> dict[str, object]:
    return {
        "grid": {"engine_hz": 60},
        "split": {"unit": "session", "train": [], "val": [B1_SESSION_ID]},
        "sessions": [
            {
                "session_id": B1_SESSION_ID,
                "frames": 53762,
                "input_active_frames": 37898,
            }
        ],
        "visual_representation": "resnet18_imagenet_avgpool_float16_v1",
        "backbone_feature_dim": 512,
    }


def test_b1_manifest_accepts_only_frozen_engine_truth_surface(
    tmp_path: Path,
) -> None:
    (tmp_path / "build_manifest.json").write_text(json.dumps(_manifest()))

    loaded = validate_b1_manifest(tmp_path, [B1_SESSION_ID])

    assert loaded["grid"] == {"engine_hz": 60}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("grid", "engine_hz", 30), "60-Hz engine truth"),
        (("split", "unit", "frame"), "validation-only session split"),
        (("split", "train", ["leak"]), "validation-only session split"),
        (("split", "val", ["other"]), "only the frozen B1 session"),
        (("sessions", 0, {"session_id": B1_SESSION_ID, "frames": 1, "input_active_frames": 37898}), "frame support changed"),
    ],
)
def test_b1_manifest_fails_closed(
    tmp_path: Path,
    mutation: tuple[str, str | int, object],
    message: str,
) -> None:
    manifest = _manifest()
    outer, inner, value = mutation
    manifest[outer][inner] = value  # type: ignore[index]
    (tmp_path / "build_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        validate_b1_manifest(tmp_path, [B1_SESSION_ID])


def test_b1_manifest_rejects_different_requested_session(tmp_path: Path) -> None:
    (tmp_path / "build_manifest.json").write_text(json.dumps(_manifest()))
    with pytest.raises(ValueError, match="only the frozen B1 session"):
        validate_b1_manifest(tmp_path, ["other"])
