from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.schema import KEY_ORDER
from harvest.accept_wild_layout import (
    LAYOUT_ACCEPTANCE_VERSION,
    REVIEW_MANIFEST_VERSION,
    accept_layout,
    validate_review_manifest,
    verify_layout_acceptance,
    write_review_manifest,
)
from harvest.fetch_wild import sha256_file
from harvest.wild_layout import SCHEMA_VERSION, WildLayout


SOURCE_SHA256 = "a" * 64


def _draft_layout_dict(video_id: str = "layout_acceptance_test") -> dict:
    cells = []
    for index, action in enumerate(KEY_ORDER):
        x = 0.05 + index * 0.085
        cells.append({
            "cell_id": f"cell_{action}",
            "action": action,
            "sample_rect": [x, 0.82, 0.06, 0.08],
            "reference_rect": [x, 0.92, 0.06, 0.03],
            "decoder": "local_contrast",
            "pressed_polarity": "high",
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "overlay_style": "synthetic_action_grid",
        "gameplay_rect": [0.0, 0.0, 1.0, 1.0],
        "gameplay_rect_source": "ai_geometry_proposal",
        "gameplay_rect_confidence": 1.0,
        "mask_rects": [[0.02, 0.78, 0.66, 0.20]],
        "cells": cells,
        "inference_source": "ai_visual_draft",
        "inference_confidence": 1.0,
        "human_reviewed": False,
        "evidence_frames_s": [1.0, 5.0],
        "temporal_offset_frames": 0,
        "temporal_offset_source": "unmeasured",
        "temporal_offset_confidence": 0.0,
    }


def _layout_review_fixture(
    tmp_path: Path,
    *,
    video_id: str = "layout_acceptance_test",
    source_sha256: str = SOURCE_SHA256,
    reviewer_kind: str = "human_with_ai_assistance",
    layout_raw: dict | None = None,
) -> tuple[Path, Path, Path, Path]:
    packet = tmp_path / "layout-review"
    frames = packet / "frames"
    frames.mkdir(parents=True)
    draft = packet / "layout.draft.json"
    raw = dict(layout_raw) if layout_raw is not None else _draft_layout_dict(video_id)
    raw["video_id"] = video_id
    raw["human_reviewed"] = False
    draft.write_text(json.dumps(raw, indent=2) + "\n")
    frame_paths = []
    for index, time_s in enumerate(raw["evidence_frames_s"]):
        name = ("released.jpg", "pressed.jpg")[index] if index < 2 else f"evidence_{index:03d}.jpg"
        frame = frames / name
        frame.write_bytes(f"exact source frame at {time_s}".encode("ascii"))
        frame_paths.append(frame)
    if len(frame_paths) < 2:
        raise ValueError("test layout fixture needs at least two evidence frames")
    released, pressed = frame_paths[:2]
    geometry = packet / "geometry.png"
    contact = packet / "cell_states.png"
    geometry.write_bytes(b"geometry review image")
    contact.write_bytes(b"cell state contact sheet")
    cell_evidence = packet / "cell_states.json"
    cell_evidence.write_text(json.dumps({
        "format_version": "madeleine.wild-layout-cell-review.v1",
        "video_id": video_id,
        "cells": [
            {
                "cell_id": f"cell_{action}",
                "action": action,
                "released": {
                    "path": "frames/released.jpg",
                    "sha256": sha256_file(released),
                },
                "pressed": {
                    "path": "frames/pressed.jpg",
                    "sha256": sha256_file(pressed),
                },
            }
            for action in KEY_ORDER
        ],
    }, indent=2) + "\n")
    manifest = packet / "review_manifest.json"
    write_review_manifest(
        manifest,
        draft,
        source_sha256=source_sha256,
        artifacts={
            "geometry_overlay": geometry,
            "cell_state_evidence": cell_evidence,
            "cell_state_contact_sheet": contact,
        },
        evidence_frames=list(zip(raw["evidence_frames_s"], frame_paths, strict=True)),
    )
    reviewed = tmp_path / "layout.reviewed.json"
    acceptance = packet / "layout_acceptance.json"
    accept_layout(
        manifest,
        draft,
        reviewed,
        acceptance,
        reviewer_identity="Test Reviewer",
        reviewer_kind=reviewer_kind,
        approved=True,
    )
    return reviewed, acceptance, manifest, draft


def test_layout_acceptance_binds_portable_packet_and_derives_human_gate(
    tmp_path: Path,
) -> None:
    reviewed, acceptance_path, manifest_path, draft = _layout_review_fixture(tmp_path)
    layout = WildLayout.load(reviewed)
    acceptance = json.loads(acceptance_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    assert manifest["format_version"] == REVIEW_MANIFEST_VERSION
    assert set(manifest) == {
        "format_version",
        "video_id",
        "source_video_sha256",
        "draft_layout",
        "review_artifacts",
        "evidence_frames",
    }
    assert not Path(manifest["draft_layout"]["path"]).is_absolute()
    assert acceptance["format_version"] == LAYOUT_ACCEPTANCE_VERSION
    assert acceptance["draft_layout"]["sha256"] == sha256_file(draft)
    assert acceptance["decision"]["human_reviewed"] is True
    assert layout.human_reviewed is True
    assert layout.to_dict()["human_reviewed"] is True
    embedded = json.loads(reviewed.read_text())["layout_review_acceptance"]
    assert embedded["artifact"] == acceptance_path.name
    assert embedded["review_manifest_sha256"] == sha256_file(manifest_path)

    verified = verify_layout_acceptance(
        reviewed, layout, acceptance_path, source_sha256=SOURCE_SHA256
    )
    assert verified["reviewer_identity"] == "Test Reviewer"
    assert verified["reviewer_kind"] == "human_with_ai_assistance"
    assert verified["human_reviewed"] is True
    assert len(verified["review_artifacts"]) == 3
    assert len(verified["evidence_frames"]) == 2


def test_ai_layout_acceptance_is_explicit_but_never_human_review(
    tmp_path: Path,
) -> None:
    reviewed, acceptance, _, _ = _layout_review_fixture(
        tmp_path, reviewer_kind="ai_agent"
    )
    layout = WildLayout.load(reviewed)
    verified = verify_layout_acceptance(
        reviewed, layout, acceptance, source_sha256=SOURCE_SHA256
    )
    assert layout.human_reviewed is False
    assert verified["reviewer_kind"] == "ai_agent"
    assert verified["human_reviewed"] is False


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("geometry.png", "(size|hash) does not match"),
        ("frames/released.jpg", "(size|hash) does not match"),
        ("layout.draft.json", "(size|hash) does not match"),
    ],
)
def test_layout_verification_fails_on_review_packet_tamper(
    tmp_path: Path, target: str, message: str,
) -> None:
    reviewed, acceptance, manifest, _ = _layout_review_fixture(tmp_path)
    (manifest.parent / target).write_bytes(b"tampered bytes")
    with pytest.raises(ValueError, match=message):
        verify_layout_acceptance(
            reviewed,
            WildLayout.load(reviewed),
            acceptance,
            source_sha256=SOURCE_SHA256,
        )


def test_layout_verification_fails_on_missing_frame_and_wrong_source(
    tmp_path: Path,
) -> None:
    reviewed, acceptance, manifest, _ = _layout_review_fixture(tmp_path)
    (manifest.parent / "frames" / "pressed.jpg").unlink()
    with pytest.raises(ValueError, match="existing non-symlink regular file"):
        verify_layout_acceptance(
            reviewed,
            WildLayout.load(reviewed),
            acceptance,
            source_sha256=SOURCE_SHA256,
        )

    other = tmp_path / "other"
    reviewed, acceptance, _, _ = _layout_review_fixture(other)
    with pytest.raises(ValueError, match="different source video"):
        verify_layout_acceptance(
            reviewed,
            WildLayout.load(reviewed),
            acceptance,
            source_sha256="b" * 64,
        )


def test_layout_verification_fails_on_missing_required_artifact(tmp_path: Path) -> None:
    reviewed, acceptance, manifest, _ = _layout_review_fixture(tmp_path)
    (manifest.parent / "cell_states.png").unlink()
    with pytest.raises(ValueError, match="existing non-symlink regular file"):
        verify_layout_acceptance(
            reviewed,
            WildLayout.load(reviewed),
            acceptance,
            source_sha256=SOURCE_SHA256,
        )


def test_v2_manifest_rejects_sensitive_or_machine_local_metadata(
    tmp_path: Path,
) -> None:
    _, _, manifest_path, _ = _layout_review_fixture(tmp_path)
    original = json.loads(manifest_path.read_text())
    serialized = manifest_path.read_text().lower()
    assert "reviewer" not in serialized
    assert "http" not in serialized
    assert "remote_host" not in serialized
    assert "/users/" not in serialized
    assert "/home/" not in serialized

    original["origin_url"] = "https://example.invalid/private"
    manifest_path.write_text(json.dumps(original, indent=2) + "\n")
    with pytest.raises(ValueError, match="privacy whitelist"):
        validate_review_manifest(manifest_path)


def test_legacy_manifest_and_bare_boolean_fail_closed(tmp_path: Path) -> None:
    reviewed, acceptance, manifest_path, _ = _layout_review_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = "madeleine.wild20-layout-evidence.v1"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="unsupported review manifest format"):
        validate_review_manifest(manifest_path)

    other = tmp_path / "bare-fixture"
    _, bare_acceptance, _, _ = _layout_review_fixture(other)
    bare = _draft_layout_dict()
    bare["human_reviewed"] = True
    bare_path = other / "bare-human-reviewed.json"
    bare_path.write_text(json.dumps(bare, indent=2) + "\n")
    with pytest.raises(ValueError, match="layout_review_acceptance"):
        verify_layout_acceptance(
            bare_path,
            WildLayout.load(bare_path),
            bare_acceptance,
            source_sha256=SOURCE_SHA256,
        )


def test_legacy_layout_acceptance_fails_closed(tmp_path: Path) -> None:
    reviewed, acceptance_path, _, _ = _layout_review_fixture(tmp_path)
    acceptance = json.loads(acceptance_path.read_text())
    acceptance["format_version"] = "madeleine.wild-layout-acceptance.v0"
    acceptance_path.write_text(json.dumps(acceptance, indent=2) + "\n")
    with pytest.raises(ValueError, match="unsupported or legacy"):
        verify_layout_acceptance(
            reviewed,
            WildLayout.load(reviewed),
            acceptance_path,
            source_sha256=SOURCE_SHA256,
        )


def test_only_timing_fields_may_change_in_a_verified_derivative(tmp_path: Path) -> None:
    reviewed, acceptance, _, _ = _layout_review_fixture(tmp_path)
    raw = json.loads(reviewed.read_text())
    raw.update({
        "temporal_offset_frames": -3,
        "temporal_offset_source": "measured fixture",
        "temporal_offset_confidence": 1.0,
        "temporal_offset_acceptance": {"artifact": "offset.json"},
    })
    timed = tmp_path / "layout.final.json"
    timed.write_text(json.dumps(raw, indent=2) + "\n")
    verified = verify_layout_acceptance(
        timed,
        WildLayout.load(timed),
        acceptance,
        source_sha256=SOURCE_SHA256,
        allow_timing_derivative=True,
    )
    assert verified["human_reviewed"] is True

    raw["gameplay_rect_confidence"] = 0.99
    timed.write_text(json.dumps(raw, indent=2) + "\n")
    with pytest.raises(ValueError, match="outside the accepted review core"):
        verify_layout_acceptance(
            timed,
            WildLayout.load(timed),
            acceptance,
            source_sha256=SOURCE_SHA256,
            allow_timing_derivative=True,
        )


def test_layout_and_manifest_outputs_are_immutable(tmp_path: Path) -> None:
    reviewed, acceptance, manifest, draft = _layout_review_fixture(tmp_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        accept_layout(
            manifest,
            draft,
            reviewed,
            acceptance,
            reviewer_identity="Test Reviewer",
            reviewer_kind="human",
            approved=True,
        )
    with pytest.raises(FileExistsError, match="overwrite"):
        write_review_manifest(
            manifest,
            draft,
            source_sha256=SOURCE_SHA256,
            artifacts={},
            evidence_frames=[],
        )


def test_layout_human_review_field_must_be_boolean() -> None:
    raw = _draft_layout_dict()
    raw["human_reviewed"] = "false"
    with pytest.raises(ValueError, match="explicit boolean"):
        WildLayout.from_dict(raw)
