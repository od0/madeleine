"""Gate semantics for the human mask-coverage waiver.

Encodes the 2026-07-28 owner ruling shape (battery sessions): a session whose
band statistic fails through key-correlated gameplay CONTENT in the band —
not overlay bleed; the declared rect fully covers the rendered widget — may
be built only under a verified, hash-bound, human-recorded waiver artifact
bound to the session manifest bytes, the evidence-packet bytes, and the
measured per-key band fractions.  The build manifest must show the gate as
overridden, never as passed.  Every binding mismatch is a hard error and the
no-flag build is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

from data.accept_coverage_waiver import (
    WAIVER_VERSION,
    accept_coverage_waiver,
    sha256_file,
)
from data.build_dataset import build_session, main as build_main
from data.mask_coverage import measure_mask_coverage, verify_mask_coverage
from data.schema import KEY_ORDER
from data.toy_sessions import _open_ffmpeg, generate_sessions
from tests.test_mask_coverage import BAR_H, BAR_W, BAR_X, BAR_Y, _paint_overlay


# A "terrain" stripe inside the coverage band below the declared rect: it is
# gameplay content whose brightness follows the grab key, the exact failure
# signature the owner packet diagnosed (content correlation, no bleed).
STRIPE_Y0, STRIPE_Y1 = 22, 32


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One content-correlated (failing) and one clean (passing) session."""

    root = tmp_path_factory.mktemp("coverage-waiver")
    toy_a, toy_b = generate_sessions(
        out=root / "toy", sessions=2, seconds=6.0, seed=20260728
    )

    failing = _overlay_session(toy_a, root / "failing" / toy_a.name, correlated=True)
    passing = _overlay_session(toy_b, root / "passing" / toy_b.name, correlated=False)
    assert verify_mask_coverage(failing) != []
    assert verify_mask_coverage(passing) == []

    evidence = root / "DECISION.md"
    evidence.write_text(
        "# Owner decision packet stand-in\n\n"
        "The band failure is chapter-content correlation, not overlay bleed.\n",
        encoding="utf-8",
    )

    waiver = root / "coverage_waiver.json"
    accept_coverage_waiver(
        failing,
        evidence,
        waiver,
        reviewer_identity="Waiver Reviewer",
        reviewer_kind="human",
        rationale="content correlation, not overlay bleed; evidence packet bound",
        approved=True,
    )
    return {
        "root": root,
        "failing": failing,
        "passing": passing,
        "evidence": evidence,
        "waiver": waiver,
    }


def _overlay_session(source: Path, dest: Path, correlated: bool) -> Path:
    """Copy a toy session, add a correctly covered widget, optionally add
    grab-correlated content in the coverage band, and declare the rect."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest)
    truth = pq.read_table(dest / "truth.parquet")
    keys = np.stack(
        [np.asarray(truth[k].to_pylist(), dtype=bool) for k in KEY_ORDER],
        axis=1,
    )
    grab = keys[:, KEY_ORDER.index("grab")]

    cap = cv2.VideoCapture(str(dest / "video.mkv"))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) == len(keys)

    video_path = dest / "video.mkv"
    video_path.unlink()
    encoder = _open_ffmpeg(video_path)
    assert encoder.stdin is not None
    for frame, keys_down, grab_down in zip(frames, keys, grab):
        _paint_overlay(frame, keys_down)  # rendered strictly inside the rect
        if correlated:
            frame[STRIPE_Y0:STRIPE_Y1, BAR_X:] = (
                (200, 200, 200) if grab_down else (40, 40, 40)
            )
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    assert encoder.wait() == 0

    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    frame_h, frame_w = frames[0].shape[:2]
    manifest["masked_regions"].append(
        {
            "name": "input_overlay",
            "space": "capture_pixels",
            "applied": "post_crop",
            "rect_px": [BAR_X, BAR_Y, BAR_W, BAR_H],
            "rect_norm": [
                BAR_X / frame_w,
                BAR_Y / frame_h,
                (BAR_X + BAR_W) / frame_w,
                (BAR_Y + BAR_H) / frame_h,
            ],
        }
    )
    digest = hashlib.sha256()
    with video_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest["integrity"]["sha256"]["video.mkv"] = digest.hexdigest()
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return dest


def _tampered_waiver(env: dict[str, Path], tmp_path: Path, mutate) -> Path:
    waiver = json.loads(env["waiver"].read_text(encoding="utf-8"))
    mutate(waiver)
    tampered = tmp_path / "waiver.tampered.json"
    tampered.write_text(json.dumps(waiver, indent=2) + "\n", encoding="utf-8")
    return tampered


def test_no_flag_coverage_rejection_is_unchanged(
    env: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="mask coverage failed"):
        build_session(env["failing"], tmp_path / "shards")


def test_no_flag_build_of_a_passing_session_carries_no_note(
    env: dict[str, Path], tmp_path: Path
) -> None:
    report = build_session(env["passing"], tmp_path / "shards")
    assert "mask_coverage" not in report
    assert (tmp_path / "shards" / f"{env['passing'].name}.npz").is_file()


def test_waiver_admits_and_manifest_shows_the_gate_as_overridden(
    env: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "shards"
    build_main([
        "--sessions", str(env["failing"]),
        "--out", str(out),
        "--coverage-waiver", str(env["waiver"]),
    ])
    manifest = json.loads((out / "build_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["sessions"]) == 1
    note = manifest["sessions"][0]["mask_coverage"]
    assert note["outcome"] == "overridden_by_recorded_human_waiver"
    assert note["waiver_sha256"] == sha256_file(env["waiver"])
    fractions = note["band_fractions"]["input_overlay"]
    assert fractions["band_any_key_fraction"] > 0.05
    assert fractions["band_key_correlated_fraction"]["grab"] > 0.05
    assert (out / f"{env['failing'].name}.npz").is_file()

    waiver = json.loads(env["waiver"].read_text(encoding="utf-8"))
    assert waiver["format_version"] == WAIVER_VERSION
    assert waiver["session_id"] == env["failing"].name
    assert waiver["session_manifest"]["sha256"] == sha256_file(
        env["failing"] / "manifest.json"
    )
    assert waiver["evidence"]["sha256"] == sha256_file(env["evidence"])
    assert waiver["waived_gate"]["band_frac_max"] == 0.05


def test_wrong_manifest_hash_is_a_hard_error(
    env: dict[str, Path], tmp_path: Path
) -> None:
    def mutate(waiver: dict) -> None:
        waiver["session_manifest"]["sha256"] = "0" * 64

    tampered = _tampered_waiver(env, tmp_path, mutate)
    with pytest.raises(ValueError, match="different session manifest bytes"):
        build_session(env["failing"], tmp_path / "shards", coverage_waiver=tampered)


def test_wrong_session_is_a_hard_error(
    env: dict[str, Path], tmp_path: Path
) -> None:
    def mutate(waiver: dict) -> None:
        waiver["session_id"] = "rec_20990101_000000_other"

    tampered = _tampered_waiver(env, tmp_path, mutate)
    with pytest.raises(ValueError, match="session_id differs"):
        build_session(env["failing"], tmp_path / "shards", coverage_waiver=tampered)


def test_ai_agent_reviewer_is_refused_at_creation_and_at_build(
    env: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="reviewer_kind must be one of"):
        accept_coverage_waiver(
            env["failing"],
            env["evidence"],
            tmp_path / "waiver.ai.json",
            reviewer_identity="Synthetic AI Reviewer",
            reviewer_kind="ai_agent",
            rationale="an agent may not waive the coverage gate",
            approved=True,
        )

    def mutate(waiver: dict) -> None:
        waiver["decision"]["reviewer_kind"] = "ai_agent"

    tampered = _tampered_waiver(env, tmp_path, mutate)
    with pytest.raises(ValueError, match="not a human provenance"):
        build_session(env["failing"], tmp_path / "shards", coverage_waiver=tampered)


def test_missing_rationale_is_refused_at_creation_and_at_build(
    env: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="rationale"):
        accept_coverage_waiver(
            env["failing"],
            env["evidence"],
            tmp_path / "waiver.no-rationale.json",
            reviewer_identity="Waiver Reviewer",
            reviewer_kind="human",
            rationale="   ",
            approved=True,
        )

    def mutate(waiver: dict) -> None:
        waiver["decision"]["rationale"] = ""

    tampered = _tampered_waiver(env, tmp_path, mutate)
    with pytest.raises(ValueError, match="rationale"):
        build_session(env["failing"], tmp_path / "shards", coverage_waiver=tampered)


def test_passing_session_has_nothing_to_waive(
    env: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="nothing to waive"):
        accept_coverage_waiver(
            env["passing"],
            env["evidence"],
            tmp_path / "waiver.unneeded.json",
            reviewer_identity="Waiver Reviewer",
            reviewer_kind="human",
            rationale="this session does not need a waiver",
            approved=True,
        )
    # And a build of a passing session refuses to consume any waiver at all.
    with pytest.raises(SystemExit, match="nothing to waive"):
        build_session(
            env["passing"], tmp_path / "shards", coverage_waiver=env["waiver"]
        )


def test_measured_json_is_cross_checked_against_a_fresh_measurement(
    env: dict[str, Path], tmp_path: Path
) -> None:
    measured = measure_mask_coverage(env["failing"])
    good = accept_coverage_waiver(
        env["failing"],
        env["evidence"],
        tmp_path / "waiver.measured.json",
        reviewer_identity="Waiver Reviewer",
        reviewer_kind="human_with_ai_assistance",
        rationale="measured report supplied and re-verified",
        approved=True,
        measured=measured,
    )
    assert good["waived_gate"]["regions"][0]["name"] == "input_overlay"

    tampered = json.loads(json.dumps(measured))
    tampered["regions"][0]["band_any_key_fraction"] = 0.5
    with pytest.raises(ValueError, match="band_any_key_fraction"):
        accept_coverage_waiver(
            env["failing"],
            env["evidence"],
            tmp_path / "waiver.measured-tampered.json",
            reviewer_identity="Waiver Reviewer",
            reviewer_kind="human",
            rationale="tampered measured report must be refused",
            approved=True,
            measured=tampered,
        )


def test_waiver_for_a_session_not_being_built_is_a_hard_error(
    env: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="not\\s+among --sessions"):
        build_main([
            "--sessions", str(env["passing"]),
            "--out", str(tmp_path / "shards"),
            "--coverage-waiver", str(env["waiver"]),
        ])
