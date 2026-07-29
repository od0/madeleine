from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harvest.build_wild import PART_COMPLETION_VERSION, PROVISIONAL_BUILD_VERSION
from harvest.fetch_wild import sha256_file
import harvest.publish_provisional_existing_layout as publisher
import harvest.publish_provisional_family_transfer as transport
from tests.test_publish_provisional_family_transfer import _FakeRclone


VIDEO_ID = "synthetic-existing-layout"
LAYOUT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "provisional_existing_layout.json"
)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    work = tmp_path / "work"
    raw = tmp_path / "raw" / VIDEO_ID
    derived = tmp_path / "derived" / VIDEO_ID
    decode_dir = work / "decode"
    parts_dir = work / "shards" / "parts"
    for directory in (raw, derived, decode_dir, parts_dir):
        directory.mkdir(parents=True)

    source = raw / f"{VIDEO_ID}.mp4"
    source.write_bytes(b"exact source video fixture")
    source_sha = sha256_file(source)
    fetch = _write(
        raw / "fetch.json",
        {
            "format_version": "madeleine.wild-fetch.v1",
            "video_id": VIDEO_ID,
            "source_file": source.name,
            "sha256": source_sha,
        },
    )
    pts_path = raw / "frame_pts.npy"
    np.save(pts_path, np.asarray([0.0, 1 / 30, 2 / 30], dtype=np.float64))
    pts_manifest = _write(
        raw / "frame_pts.json",
        {
            "format_version": "madeleine.wild-pts.v1",
            "source_file": source.name,
            "source_sha256": source_sha,
            "path": pts_path.name,
            "sha256": sha256_file(pts_path),
            "frames": 3,
        },
    )
    layout_value = json.loads(LAYOUT_FIXTURE.read_text())
    layout = _write(derived / "layout.draft.json", layout_value)
    boundaries = _write(
        derived / "boundaries.outer-ai.json",
        {
            "format_version": "madeleine.wild-boundaries.v2",
            "video_id": VIDEO_ID,
            "source_sha256": source_sha,
            "wall_clock_range_s": [0.0, 0.1],
            "excluded_ranges_s": [],
            "human_reviewed": False,
            "reviewer": "fixture AI",
            "reviewer_kind": "ai_agent",
            "evidence": ["existing AI layout"],
        },
    )
    raw_labels = decode_dir / "labels_raw.parquet"
    labels = decode_dir / "labels_native.parquet"
    raw_labels.write_bytes(b"raw labels")
    labels.write_bytes(b"native labels")
    decoded_hours = 3 / 30 / 3600
    reasons = [
        "HUD compositor offset confidence below admission threshold",
        "HUD compositor offset is unmeasured",
        "gameplay boundaries were not reviewed by a human",
        "layout lacks a verified hash-bound review acceptance",
    ]
    decode = _write(
        decode_dir / "decode_report.json",
        {
            "format_version": publisher.DECODE_VERSION,
            "video_id": VIDEO_ID,
            "source_video": {"path": str(source), "sha256": source_sha},
            "boundaries": {
                "path": str(boundaries),
                "sha256": sha256_file(boundaries),
                "human_reviewed": False,
            },
            "layout": {
                "path": str(layout),
                "sha256": sha256_file(layout),
                "human_reviewed": False,
            },
            "timing": {
                "authority": "presentation_timestamp",
                "pts_evidence": {
                    "manifest": str(pts_manifest),
                    "sha256": sha256_file(pts_path),
                    "frames": 3,
                },
            },
            "score_source": None,
            "decoded_frames": 3,
            "decoded_hours": decoded_hours,
            "raw_labels": raw_labels.name,
            "raw_labels_sha256": sha256_file(raw_labels),
            "labels": labels.name,
            "labels_sha256": sha256_file(labels),
            "admitted": False,
            "rejection_reasons": reasons,
        },
    )
    part = parts_dir / f"wild_provisional_{VIDEO_ID}__r000.npz"
    part.write_bytes(b"NPZ fixture")
    part_row = {
        "session_id": f"wild_provisional_{VIDEO_ID}__r000",
        "npz": part.name,
        "frames": 3,
        "source_frame_range": [0, 3],
        "pts_range_s": [0.0, 2 / 30],
        "sha256": sha256_file(part),
    }
    implementation_sha = "f" * 64
    _write(
        part.with_name(part.name + ".complete.json"),
        {
            "format_version": PART_COMPLETION_VERSION,
            "row": part_row,
            "npz_bytes": part.stat().st_size,
            "bindings": {
                "implementation_sha256": implementation_sha,
                "source_video_sha256": source_sha,
                "labels_sha256": sha256_file(labels),
                "layout_sha256": sha256_file(layout),
                "boundaries_sha256": sha256_file(boundaries),
            },
            "arrays": {},
        },
    )
    build = {
        "format_version": PROVISIONAL_BUILD_VERSION,
        "video_id": VIDEO_ID,
        "label_kind": "wild_overlay_provisional",
        "admission_tier": "provisional_not_train_ready",
        "timing_authority": "presentation_timestamp",
        "implementation": {
            "module": "harvest/build_wild.py",
            "sha256": implementation_sha,
        },
        "effective_grid_hz": 30.0,
        "decoded_frames": 3,
        "decoded_hours": decoded_hours,
        "train_ready_frames": 0,
        "train_ready_hours": 0.0,
        "provisional_trainable_frames": 3,
        "provisional_trainable_hours": decoded_hours,
        "inputs": {
            "decode_report": {"path": decode.name, "sha256": sha256_file(decode)},
            "labels": {"path": labels.name, "sha256": sha256_file(labels)},
            "layout": {"path": layout.name, "sha256": sha256_file(layout)},
            "boundaries_sha256": sha256_file(boundaries),
            "source_video_sha256": source_sha,
        },
        "parts": [part_row],
        "unresolved_admission_reasons": reasons,
    }
    _write(parts_dir / "wild_provisional_build_report.json", build)
    _write(
        parts_dir.parent / "wild_provisional_corpus_manifest.json",
        {
            "format_version": PROVISIONAL_BUILD_VERSION,
            "admission_tier": "provisional_not_train_ready",
            "videos": [build],
            "video_count": 1,
            "train_ready_hours": 0.0,
            "provisional_trainable_hours": decoded_hours,
        },
    )
    return {
        "work_root": work,
        "raw_dir": raw,
        "layout_path": layout,
        "boundaries_path": boundaries,
    }


def test_existing_layout_publication_is_hash_verified_marker_last_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    remote = _FakeRclone()
    monkeypatch.setattr(transport.subprocess, "run", remote.run)
    monkeypatch.setattr(transport.subprocess, "Popen", remote.popen)
    kwargs = {
        **paths,
        "video_id": VIDEO_ID,
        "state_dir": tmp_path / "state",
        "remote_root": "r2:testbucket/wild/v1/provisional-existing-layout",
        "cadence_tier": "native30",
    }

    result = publisher.publish(**kwargs)

    prefix = (
        "r2:testbucket/wild/v1/provisional-existing-layout/native30/"
        + VIDEO_ID
    )
    assert result["publication_status"] == "published"
    assert remote.copies[-1] == f"{prefix}/{publisher.COMPLETION_NAME}"
    manifest = json.loads((tmp_path / "state" / publisher.MANIFEST_NAME).read_text())
    assert manifest["human_reviewed"] is False
    assert manifest["training_admitted"] is False
    assert manifest["source"]["source_video_uploaded_here"] is False
    copy_count = len(remote.copies)

    repeated = publisher.publish(**kwargs)

    assert repeated["publication_status"] == "already_complete_validated"
    assert len(remote.copies) == copy_count


def test_existing_layout_publication_rejects_pts_and_part_binding_tamper(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    pts = paths["raw_dir"] / "frame_pts.npy"
    pts.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="PTS array SHA-256 mismatch"):
        publisher.collect_artifacts(
            **paths, video_id=VIDEO_ID, cadence_tier="native30"
        )

    paths = _fixture(tmp_path / "fresh")
    sidecar_path = next(
        (paths["work_root"] / "shards" / "parts").glob("*.npz.complete.json")
    )
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["bindings"]["layout_sha256"] = "0" * 64
    _write(sidecar_path, sidecar)
    with pytest.raises(ValueError, match="sidecar bindings mismatch"):
        publisher.collect_artifacts(
            **paths, video_id=VIDEO_ID, cadence_tier="native30"
        )
