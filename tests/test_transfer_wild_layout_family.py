from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from data.schema import KEY_ORDER
from harvest.fetch_wild import sha256_file
from harvest.transfer_wild_layout_family import prepare_transfer
from harvest.wild_layout import SCHEMA_VERSION


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    video_id = "target_video"
    source_dir = root / "source" / video_id
    survey_dir = root / "surveys" / video_id
    source_dir.mkdir(parents=True)
    survey_dir.mkdir(parents=True)

    video = source_dir / f"{video_id}.mp4"
    video.write_bytes(b"immutable synthetic video")
    source_sha = sha256_file(video)
    pts = np.arange(16, dtype=np.float64) / 60.0
    pts_path = source_dir / "frame_pts.npy"
    np.save(pts_path, pts, allow_pickle=False)
    pts_sha = sha256_file(pts_path)
    fetch_path = _write_json(
        source_dir / "fetch.json",
        {
            "video_id": video_id,
            "source_file": video.name,
            "sha256": source_sha,
            "media": {"resolution_wh": [160, 90]},
        },
    )
    pts_manifest_path = _write_json(
        source_dir / "frame_pts.json",
        {
            "source_sha256": source_sha,
            "sha256": pts_sha,
            "frames": int(pts.size),
        },
    )
    _write_json(
        source_dir / "upload_complete.json",
        {
            "video_id": video_id,
            "objects": [
                {"name": path.name, "sha256": sha256_file(path)}
                for path in (fetch_path, pts_path, pts_manifest_path, video)
            ],
        },
    )

    frame_rows = []
    for index in range(8):
        frame = np.full((90, 160, 3), index * 20, dtype=np.uint8)
        path = survey_dir / f"sample-{index:02d}.png"
        assert cv2.imwrite(str(path), frame)
        frame_rows.append(
            {
                "exact_pts_s": float(pts[index * 2]),
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    contact = survey_dir / "contact-sheet.png"
    assert cv2.imwrite(str(contact), np.zeros((90, 160, 3), dtype=np.uint8))
    survey_path = _write_json(
        survey_dir / "survey.json",
        {
            "video_id": video_id,
            "source": {"sha256": source_sha},
            "pts": {"sha256": pts_sha},
            "frames": frame_rows,
            "contact_sheet": {
                "path": contact.name,
                "size_bytes": contact.stat().st_size,
                "sha256": sha256_file(contact),
            },
            "human_reviewed": False,
            "training_admitted": False,
        },
    )
    _write_json(
        survey_dir / "survey_complete.json",
        {
            "video_id": video_id,
            "source_sha256": source_sha,
            "survey_sha256": sha256_file(survey_path),
            "human_reviewed": False,
            "training_admitted": False,
        },
    )

    reference_id = "reference_video"
    reference_layout = _write_json(
        root / "reference-layout.json",
        {
            "schema_version": SCHEMA_VERSION,
            "video_id": reference_id,
            "overlay_style": "synthetic_direct_grid",
            "gameplay_rect": [0.0, 0.0, 1.0, 0.75],
            "gameplay_rect_source": "source-bound synthetic fixture",
            "gameplay_rect_confidence": 0.9,
            "mask_rects": [[0.0, 0.75, 1.0, 0.25]],
            "cells": [
                {
                    "cell_id": f"printed_{action}",
                    "action": action,
                    "sample_rect": [0.02 + index * 0.13, 0.8, 0.05, 0.1],
                    "decoder": "luma",
                    "pressed_polarity": "high",
                }
                for index, action in enumerate(KEY_ORDER)
            ],
            "inference_source": "synthetic fixture",
            "inference_confidence": 0.9,
            "human_reviewed": False,
            "evidence_frames_s": [0.0],
            "temporal_offset_frames": 0,
            "temporal_offset_source": "unmeasured",
            "temporal_offset_confidence": 0.0,
        },
    )
    return source_dir, survey_dir, reference_layout


def test_prepare_is_deterministic_across_fresh_preflight_directories(
    tmp_path: Path,
) -> None:
    source_dir, survey_dir, reference_layout = _fixture(tmp_path)
    outputs = []
    for name in ("preflight", "worker"):
        out = tmp_path / name
        paths = prepare_transfer(
            source_dir=source_dir,
            survey_dir=survey_dir,
            reference_layout_path=reference_layout,
            reference_video_id="reference_video",
            out_dir=out,
        )
        outputs.append({key: path.read_bytes() for key, path in paths.items()})

    assert outputs[0] == outputs[1]
    assert "created_at" not in json.loads(outputs[0]["evidence"])


@pytest.mark.requires_private_artifacts(
    "harvest/run_validated_scan_assignment.sh",
    "harvest/run_layout_family_worker.sh",
)
def test_validated_preflight_and_inner_worker_share_canonical_evidence(
    tmp_path: Path,
) -> None:
    source_dir, survey_dir, reference_layout = _fixture(tmp_path)
    root = Path(__file__).resolve().parents[1]
    launcher = root / "harvest/run_validated_scan_assignment.sh"
    worker = root / "harvest/run_layout_family_worker.sh"

    # The Python producer—not either shell caller—owns the canonical note.
    explicit_note_option = re.compile(r"^\s+--assessment-note(?:\s|$)", re.M)
    assert explicit_note_option.search(launcher.read_text()) is None
    assert explicit_note_option.search(worker.read_text()) is None

    evidence = []
    for caller in ("validated-preflight", "inner-worker"):
        out = tmp_path / caller
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "harvest.transfer_wild_layout_family",
                "prepare",
                "--source-dir",
                str(source_dir),
                "--survey-dir",
                str(survey_dir),
                "--reference-layout",
                str(reference_layout),
                "--reference-video-id",
                "reference_video",
                "--out",
                str(out),
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        evidence.append((out / "transfer_evidence.json").read_bytes())

    assert evidence[0] == evidence[1]


@pytest.mark.requires_private_artifacts(
    "harvest/run_validated_scan_assignment.sh",
    "harvest/run_layout_family_worker.sh",
)
def test_validated_launcher_code_hashes_match_tracked_closure() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "harvest/run_validated_scan_assignment.sh").read_text()
    expected = {
        "worker": root / "harvest/run_layout_family_worker.sh",
        "transfer": root / "harvest/transfer_wild_layout_family.py",
        "builder": root / "harvest/build_wild.py",
    }
    for name, path in expected.items():
        match = re.search(
            rf"^expected_{name}_sha=([0-9a-f]{{64}})$", launcher, re.M
        )
        assert match is not None
        assert match.group(1) == sha256_file(path)


def test_prepare_preserves_legacy_timestamp_and_rejects_changed_assessment(
    tmp_path: Path,
) -> None:
    source_dir, survey_dir, reference_layout = _fixture(tmp_path)
    out = tmp_path / "transfer"
    kwargs = {
        "source_dir": source_dir,
        "survey_dir": survey_dir,
        "reference_layout_path": reference_layout,
        "reference_video_id": "reference_video",
        "out_dir": out,
    }
    paths = prepare_transfer(**kwargs)
    evidence = json.loads(paths["evidence"].read_text())
    evidence = {
        "format_version": evidence.pop("format_version"),
        "created_at": "2026-07-27T00:00:00+00:00",
        **evidence,
    }
    _write_json(paths["evidence"], evidence)

    prepare_transfer(**kwargs)
    assert json.loads(paths["evidence"].read_text())["created_at"] == (
        "2026-07-27T00:00:00+00:00"
    )
    with pytest.raises(FileExistsError, match="transfer_evidence.json"):
        prepare_transfer(**kwargs, assessment_note="a different provenance claim")
