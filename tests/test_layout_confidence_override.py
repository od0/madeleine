"""Gate semantics for the human layout-confidence override.

Encodes the 2026-07-27 owner ruling shape (ofy37Fm6EgI): a layout whose
recorded ``inference_confidence`` sits below the 0.80 decode admission floor
was nevertheless accepted by a human through the hash-bound review packet.
The confidence rejection may be suppressed only by a verified, hash-bound,
human-recorded override artifact, and the decode report must show the gate
as overridden, never as passed.  Every binding mismatch is a hard error and
the no-flag decode is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harvest.accept_layout_confidence import (
    OVERRIDE_VERSION,
    accept_confidence_override,
)
from harvest.accept_wild_offset import accept_offset
from harvest.decode_wild import decode_video
from harvest.fetch_wild import probe_media, sha256_file
from harvest.wild_boundaries import BOUNDARIES_VERSION
from harvest.wild_layout import WildLayout
from tests.test_layout_acceptance import _layout_review_fixture
from tests.test_offset_acceptance import _write_calibration
from tests.test_wild_pipeline import FPS, N, _layout_dict, _write_video


CONFIDENCE_REJECTION = "layout inference confidence below admission threshold"
LOW_CONFIDENCE = 0.78


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One decodable video with a human-accepted 0.78-confidence layout."""

    root = tmp_path_factory.mktemp("confidence-override")
    video_id = "confidence_override_test"
    layout_raw = _layout_dict(video_id)
    layout_raw["inference_confidence"] = LOW_CONFIDENCE
    layout_raw["human_reviewed"] = False
    render_layout_path = root / "layout.render.json"
    render_layout_path.write_text(json.dumps(layout_raw))
    video = root / f"{video_id}.mp4"
    _write_video(video, WildLayout.load(render_layout_path))

    media = probe_media(video, scan_pts=True)
    fetch = {
        "format_version": "madeleine.wild-fetch.v1",
        "video_id": video_id,
        "source": "youtube",
        "origin_url": "https://youtu.be/confidence_override_test",
        "source_file": video.name,
        "sha256": sha256_file(video),
        "candidate": {"duration_s": N / FPS},
        "media": media,
        "run_window": {
            "resolved": True,
            "start_s": 0.0,
            "end_s": N / FPS,
            "duration_s": N / FPS,
            "source": "synthetic",
            "reason": None,
        },
    }
    fetch_path = root / "fetch.json"
    fetch_path.write_text(json.dumps(fetch))

    layout_path, layout_acceptance, _, _ = _layout_review_fixture(
        root / "review-fixture",
        video_id=video_id,
        source_sha256=fetch["sha256"],
        layout_raw=layout_raw,
    )
    boundaries_path = root / "boundaries.json"
    boundaries_path.write_text(json.dumps({
        "format_version": BOUNDARIES_VERSION,
        "video_id": video_id,
        "source_sha256": fetch["sha256"],
        "wall_clock_range_s": [0.0, N / FPS],
        "excluded_ranges_s": [[4.0, 5.0]],
        "human_reviewed": True,
        "reviewer": "synthetic-test",
        "reviewer_kind": "human",
        "evidence": ["fixture"],
    }))

    calibration, _ = _write_calibration(
        root / "calibration",
        layout_path,
        winner=0,
        source_sha256=fetch["sha256"],
    )
    final_layout_path = root / "layout.final.json"
    offset_acceptance = calibration.parent / "offset_acceptance.json"
    accept_offset(
        calibration,
        layout_path,
        layout_acceptance,
        final_layout_path,
        offset_acceptance,
        reviewer_identity="synthetic-test-reviewer",
        reviewer_kind="human",
        approved=True,
    )

    override_path = root / "layout_confidence_override.json"
    accept_confidence_override(
        final_layout_path,
        layout_acceptance,
        override_path,
        reviewer_identity="Override Reviewer",
        reviewer_kind="human_with_ai_assistance",
        rationale="recorded 0.78 stands: the mapping is fail-closed and minimal",
        approved=True,
    )
    return {
        "root": root,
        "fetch_path": fetch_path,
        "layout_path": layout_path,
        "final_layout_path": final_layout_path,
        "layout_acceptance": layout_acceptance,
        "offset_acceptance": offset_acceptance,
        "boundaries_path": boundaries_path,
        "override_path": override_path,
    }


def _decode(env: dict[str, Path], out_name: str, **kwargs) -> dict:
    return decode_video(
        env["fetch_path"],
        env["final_layout_path"],
        env["boundaries_path"],
        env["root"] / out_name,
        layout_acceptance_path=env["layout_acceptance"],
        offset_acceptance_path=env["offset_acceptance"],
        **kwargs,
    )


def _tampered_override(env: dict[str, Path], tmp_path: Path, mutate) -> Path:
    override = json.loads(env["override_path"].read_text())
    mutate(override)
    tampered = tmp_path / "override.tampered.json"
    tampered.write_text(json.dumps(override, indent=2) + "\n")
    return tampered


def test_no_flag_confidence_rejection_is_unchanged(env: dict[str, Path]) -> None:
    report = _decode(env, "decoded-no-flag")
    assert not report["admitted"]
    assert report["rejection_reasons"] == [CONFIDENCE_REJECTION]
    assert "admission_overrides" not in report
    assert "confidence_override" not in report["layout"]


def test_override_admits_and_report_shows_the_gate_as_overridden(
    env: dict[str, Path],
) -> None:
    report = _decode(
        env, "decoded-override", confidence_override_path=env["override_path"]
    )
    assert report["admitted"], report["rejection_reasons"]
    assert CONFIDENCE_REJECTION not in report["rejection_reasons"]

    note = report["admission_overrides"]
    assert len(note) == 1
    assert note[0]["gate"] == CONFIDENCE_REJECTION
    assert note[0]["outcome"] == "overridden_by_recorded_human_ruling"
    assert note[0]["inference_confidence"] == LOW_CONFIDENCE
    assert note[0]["min_layout_confidence"] == 0.80
    assert note[0]["override_sha256"] == sha256_file(env["override_path"])
    assert note[0]["reviewer_identity"] == "Override Reviewer"
    assert note[0]["reviewer_kind"] == "human_with_ai_assistance"
    assert "0.78 stands" in note[0]["rationale"]

    embedded = report["layout"]["confidence_override"]
    assert embedded["format_version"] == OVERRIDE_VERSION
    assert embedded["sha256"] == sha256_file(env["override_path"])
    assert embedded["human_reviewed"] is True
    assert embedded["layout_sha256"] == sha256_file(env["final_layout_path"])
    assert embedded["layout_acceptance_sha256"] == sha256_file(
        env["layout_acceptance"]
    )


def test_wrong_layout_hash_is_a_hard_error(
    env: dict[str, Path], tmp_path: Path
) -> None:
    def mutate(override: dict) -> None:
        override["layout"]["sha256"] = "0" * 64

    tampered = _tampered_override(env, tmp_path, mutate)
    with pytest.raises(ValueError, match="different layout bytes"):
        _decode(env, "decoded-wrong-layout-hash", confidence_override_path=tampered)


def test_wrong_acceptance_hash_is_a_hard_error(
    env: dict[str, Path], tmp_path: Path
) -> None:
    def mutate(override: dict) -> None:
        override["layout_acceptance"]["sha256"] = "0" * 64

    tampered = _tampered_override(env, tmp_path, mutate)
    with pytest.raises(ValueError, match="different layout-acceptance bytes"):
        _decode(env, "decoded-wrong-acceptance-hash", confidence_override_path=tampered)


def test_ai_agent_reviewer_is_refused_at_creation_and_at_decode(
    env: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="reviewer_kind must be one of"):
        accept_confidence_override(
            env["final_layout_path"],
            env["layout_acceptance"],
            tmp_path / "override.ai.json",
            reviewer_identity="Synthetic AI Reviewer",
            reviewer_kind="ai_agent",
            rationale="an agent may not waive the confidence floor",
            approved=True,
        )

    def mutate(override: dict) -> None:
        override["decision"]["reviewer_kind"] = "ai_agent"

    tampered = _tampered_override(env, tmp_path, mutate)
    with pytest.raises(ValueError, match="not a human provenance"):
        _decode(env, "decoded-ai-reviewer", confidence_override_path=tampered)


def test_missing_rationale_is_refused_at_creation_and_at_decode(
    env: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="rationale"):
        accept_confidence_override(
            env["final_layout_path"],
            env["layout_acceptance"],
            tmp_path / "override.no-rationale.json",
            reviewer_identity="Override Reviewer",
            reviewer_kind="human",
            rationale="   ",
            approved=True,
        )

    def mutate(override: dict) -> None:
        override["decision"]["rationale"] = ""

    tampered = _tampered_override(env, tmp_path, mutate)
    with pytest.raises(ValueError, match="rationale"):
        _decode(env, "decoded-no-rationale", confidence_override_path=tampered)


def test_override_without_layout_acceptance_is_a_hard_error(
    env: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="requires a verified hash-bound layout acceptance"):
        decode_video(
            env["fetch_path"],
            env["layout_path"],
            env["boundaries_path"],
            env["root"] / "decoded-override-without-acceptance",
            confidence_override_path=env["override_path"],
        )


def test_confidence_at_or_above_the_floor_has_nothing_to_override(
    tmp_path: Path,
) -> None:
    layout_path, layout_acceptance, _, _ = _layout_review_fixture(
        tmp_path, video_id="confident_layout"
    )
    assert WildLayout.load(layout_path).inference_confidence >= 0.80
    with pytest.raises(ValueError, match="nothing to override"):
        accept_confidence_override(
            layout_path,
            layout_acceptance,
            tmp_path / "override.unneeded.json",
            reviewer_identity="Override Reviewer",
            reviewer_kind="human",
            rationale="this layout does not need an override",
            approved=True,
        )
