"""The v2-to-v3 offset re-verdict works from hash-bound serialized evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harvest.accept_wild_offset import accept_offset, verify_offset_acceptance
from harvest.calibrate_offset import (
    CALIBRATION_VERSION,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_UNCERTAIN_ADJACENT,
)
from harvest.fetch_wild import sha256_file
from harvest.reverdict_offset_v3 import (
    V2_CALIBRATION_VERSION,
    reverdict_calibration_dir,
)
from harvest.wild_layout import WildLayout
from tests.test_layout_acceptance import _layout_review_fixture
from tests.test_offset_acceptance import _write_calibration


def _downgrade_to_v2(calibration_path: Path) -> None:
    """Rewrite a v3 fixture as the equivalent immutable v2 evidence record."""

    raw = json.loads(calibration_path.read_text())
    raw["format_version"] = V2_CALIBRATION_VERSION
    for key in ("verdict", "per_event_collar_fraction", "offset_uncertainty_frames"):
        raw.pop(key, None)
    raw["policy"]["min_mode_fraction"] = 0.80
    raw["automatic_gates_passed"] = False
    raw["automatic_failure_reasons"] = [
        "per-event lag within winner±1 at 0.500; need >= 0.800 (exact mode -3 at 0.500)"
    ]
    _rewrite(calibration_path, raw)


def _rewrite(calibration_path: Path, raw: dict) -> None:
    calibration_path.write_text(json.dumps(raw, indent=2) + "\n")
    calibration_path.with_suffix(".sha256").write_text(
        f"{sha256_file(calibration_path)}  {calibration_path.name}\n"
    )


def _v2_dir(tmp_path: Path, layout_path: Path, *, verdict: str = "pass") -> Path:
    calibration, _ = _write_calibration(
        tmp_path / "offset-v2", layout_path, verdict=verdict
    )
    _downgrade_to_v2(calibration)
    return calibration.parent


def test_mode_fraction_only_v2_failure_becomes_a_v3_pass(tmp_path: Path) -> None:
    layout_path, _, _, _ = _layout_review_fixture(tmp_path, video_id="acceptance_test")
    v2_dir = _v2_dir(tmp_path, layout_path)
    v2_hash = sha256_file(v2_dir / "offset_calibration.json")

    result = reverdict_calibration_dir(v2_dir)
    out_dir = tmp_path / "offset-v3"
    assert result["format_version"] == CALIBRATION_VERSION
    assert result["verdict"] == VERDICT_PASS
    assert result["automatic_gates_passed"] is True
    assert result["automatic_failure_reasons"] == []
    assert result["per_event_collar_fraction"] == 1.0
    assert result["offset_uncertainty_frames"] == 1
    assert "min_mode_fraction" not in result["policy"]
    assert result["reverdict"]["from_calibration_sha256"] == v2_hash

    written = json.loads((out_dir / "offset_calibration.json").read_text())
    assert written == json.loads(json.dumps(result))
    sidecar = (out_dir / "offset_calibration.sha256").read_text().strip()
    assert sidecar == (
        f"{sha256_file(out_dir / 'offset_calibration.json')}  offset_calibration.json"
    )
    # The contact sheet is reused byte-for-byte with its hash re-bound.
    assert (
        sha256_file(out_dir / "dash_offset_contact.png")
        == result["human_handoff"]["contact_sheet_sha256"]
        == sha256_file(v2_dir / "dash_offset_contact.png")
    )


def test_margin_shortfall_reverdicts_to_uncertain_and_accepts_with_flag(
    tmp_path: Path,
) -> None:
    layout_path, layout_acceptance, _, _ = _layout_review_fixture(
        tmp_path, video_id="acceptance_test"
    )
    v2_dir = _v2_dir(tmp_path, layout_path, verdict="uncertain_adjacent")
    result = reverdict_calibration_dir(v2_dir)
    assert result["verdict"] == VERDICT_UNCERTAIN_ADJACENT
    assert result["automatic_gates_passed"] is False
    assert all(
        reason.startswith("non-adjacent median margin")
        for reason in result["automatic_failure_reasons"]
    )

    out_dir = tmp_path / "offset-v3"
    output_layout = tmp_path / "layout.final.json"
    acceptance_path = out_dir / "offset_acceptance.json"
    acceptance = accept_offset(
        out_dir / "offset_calibration.json",
        layout_path,
        layout_acceptance,
        output_layout,
        acceptance_path,
        reviewer_identity="Reviewer",
        reviewer_kind="human_with_ai_assistance",
        approved=True,
        accept_uncertain_tier=True,
    )
    assert acceptance["accepted_tier"] == VERDICT_UNCERTAIN_ADJACENT
    verified = verify_offset_acceptance(
        output_layout,
        WildLayout.load(output_layout),
        acceptance_path,
        source_sha256="a" * 64,
        layout_acceptance_path=layout_acceptance,
    )
    assert verified["tier"] == VERDICT_UNCERTAIN_ADJACENT
    assert verified["offset_uncertainty_frames"] == 1


def test_block_or_bootstrap_failures_remain_hard_fails(tmp_path: Path) -> None:
    layout_path, _, _, _ = _layout_review_fixture(tmp_path, video_id="acceptance_test")

    blocks_dir = _v2_dir(tmp_path, layout_path)
    raw = json.loads((blocks_dir / "offset_calibration.json").read_text())
    raw["temporal_blocks"][2]["winner_offset_frames"] = (
        raw["best_candidate_offset_frames"] + 3
    )
    _rewrite(blocks_dir / "offset_calibration.json", raw)
    result = reverdict_calibration_dir(blocks_dir, tmp_path / "v3-blocks")
    assert result["verdict"] == VERDICT_FAIL
    assert any("temporal block" in r for r in result["automatic_failure_reasons"])

    (tmp_path / "b").mkdir()
    bootstrap_dir = _v2_dir(tmp_path / "b", layout_path)
    raw = json.loads((bootstrap_dir / "offset_calibration.json").read_text())
    winner = raw["best_candidate_offset_frames"]
    runner = raw["runner_up_offset_frames"]
    for row in raw["candidates"]:
        if row["offset_frames"] == winner:
            row["bootstrap_wins"], row["bootstrap_fraction"] = 1_600, 0.8
        elif row["offset_frames"] == runner:
            row["bootstrap_wins"], row["bootstrap_fraction"] = 400, 0.2
    raw["bootstrap_win_fraction"] = 0.8
    _rewrite(bootstrap_dir / "offset_calibration.json", raw)
    result = reverdict_calibration_dir(bootstrap_dir, tmp_path / "v3-bootstrap")
    assert result["verdict"] == VERDICT_FAIL
    assert any("bootstrap" in r for r in result["automatic_failure_reasons"])


def test_reverdict_fails_closed_on_tampered_or_inconsistent_evidence(
    tmp_path: Path,
) -> None:
    layout_path, _, _, _ = _layout_review_fixture(tmp_path, video_id="acceptance_test")
    v2_dir = _v2_dir(tmp_path, layout_path)
    calibration_path = v2_dir / "offset_calibration.json"

    raw = json.loads(calibration_path.read_text())
    raw["median_score_margin"] = raw["median_score_margin"] + 0.5
    _rewrite(calibration_path, raw)
    with pytest.raises(ValueError, match="disagrees with its recomputation"):
        reverdict_calibration_dir(v2_dir, tmp_path / "v3-a")

    # Tampering without updating the sidecar fails the byte check.
    raw["median_score_margin"] = raw["median_score_margin"] - 0.5
    calibration_path.write_text(json.dumps(raw, indent=2) + "\n")
    with pytest.raises(ValueError, match="sidecar does not match"):
        reverdict_calibration_dir(v2_dir, tmp_path / "v3-b")
    _rewrite(calibration_path, raw)

    # An unrecorded score matrix beside the record is unbound evidence.
    (v2_dir / "score_matrix.npz").write_bytes(b"stray")
    with pytest.raises(ValueError, match="not bound by the v2 record"):
        reverdict_calibration_dir(v2_dir, tmp_path / "v3-c")
