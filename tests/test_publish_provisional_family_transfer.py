from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from harvest.build_wild import PART_COMPLETION_VERSION, PROVISIONAL_BUILD_VERSION
from harvest.fetch_wild import sha256_file
import harvest.publish_provisional_family_transfer as publisher
from harvest.transfer_wild_layout_family import EVIDENCE_VERSION


VIDEO_ID = "video_family_1"
SOURCE_SHA = "a" * 64
PTS_SHA = "b" * 64


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def _fixture(tmp_path: Path, *, effective_fps: float = 60.0) -> Path:
    root = tmp_path / "job"
    transfer = root / "transfer"
    scan = root / "scan" / VIDEO_ID
    decode = root / "decode"
    parts = root / "shards" / "parts"
    for directory in (transfer, scan, decode, parts):
        directory.mkdir(parents=True)

    actions = ("left", "right", "up", "down", "jump", "dash", "grab")
    layout = _write_json(
        transfer / "layout.family-transfer-ai.json",
        {
            "schema_version": "madeleine.wild-layout.v1",
            "video_id": VIDEO_ID,
            "overlay_style": "fixture",
            "gameplay_rect": [0.0, 0.0, 1.0, 0.8],
            "gameplay_rect_source": "AI-only family fixture",
            "gameplay_rect_confidence": 0.85,
            "mask_rects": [[0.0, 0.8, 1.0, 0.2]],
            "cells": [
                {
                    "cell_id": action,
                    "action": action,
                    "sample_rect": [index / 10, 0.85, 0.05, 0.05],
                    "decoder": "luma",
                    "pressed_polarity": "high",
                }
                for index, action in enumerate(actions)
            ],
            "inference_source": "AI family transfer fixture",
            "inference_confidence": 0.85,
            "human_reviewed": False,
            "evidence_frames_s": [1.0],
            "temporal_offset_frames": 0,
            "temporal_offset_source": "unmeasured",
            "temporal_offset_confidence": 0.0,
        },
    )
    reference_layout = transfer / "reference-layout.source.json"
    reference_layout.write_bytes(layout.read_bytes())
    spec_value = {
        "format_version": "madeleine.wild-cell-scan-spec.v1",
        "video_id": VIDEO_ID,
        "source_sha256": SOURCE_SHA,
        "pts_sha256": PTS_SHA,
        "survey_sha256": "c" * 64,
        "survey_contact_sheet_sha256": "d" * 64,
        "frame_size_wh": [1280, 720],
        "cells": [],
        "human_reviewed": False,
        "training_admitted": False,
    }
    spec = _write_json(
        transfer / "cell-scan-spec.family-transfer-ai.json", spec_value
    )
    boundaries = _write_json(
        transfer / "boundaries.outer-ai.json",
        {
            "format_version": "madeleine.wild-boundaries.v2",
            "video_id": VIDEO_ID,
            "source_sha256": SOURCE_SHA,
            "wall_clock_range_s": [0.0, 1.0],
            "allowed_ranges_s": [[0.0, 1.0]],
            "human_reviewed": False,
            "reviewer": "fixture AI",
            "reviewer_kind": "ai_agent",
            "evidence": ["fixture"],
        },
    )
    evidence = _write_json(
        transfer / "transfer_evidence.json",
        {
            "format_version": EVIDENCE_VERSION,
            "video_id": VIDEO_ID,
            "reference_video_id": "reference_1",
            "bindings": {
                "target_source_sha256": SOURCE_SHA,
                "target_pts_sha256": PTS_SHA,
                "target_survey_sha256": "c" * 64,
                "target_contact_sheet_sha256": "d" * 64,
                "reference_layout_sha256": sha256_file(reference_layout),
                "generated_layout_sha256": sha256_file(layout),
                "generated_scan_spec_sha256": sha256_file(spec),
                "generated_boundaries_sha256": sha256_file(boundaries),
            },
            "human_reviewed": False,
            "training_admitted": False,
        },
    )
    assert evidence.is_file()

    copied_spec = scan / spec.name
    copied_spec.write_bytes(spec.read_bytes())
    scores = scan / "cell_scores.f32"
    scores.write_bytes(b"score-bytes-bound-to-report")
    evidence_frame = scan / "evidence" / "frame-000.png"
    evidence_frame.parent.mkdir()
    evidence_frame.write_bytes(b"image evidence")
    contact = scan / "evidence-contact-sheet.png"
    contact.write_bytes(b"contact sheet")
    scan_report = _write_json(
        scan / "cell_activity_scan.json",
        {
            "format_version": publisher.SCAN_VERSION,
            "video_id": VIDEO_ID,
            "source": {"sha256": SOURCE_SHA, "frames": 10},
            "pts": {"sha256": PTS_SHA},
            "spec": {
                "path": copied_spec.name,
                "size_bytes": copied_spec.stat().st_size,
                "sha256": sha256_file(copied_spec),
            },
            "scores": {
                "path": scores.name,
                "size_bytes": scores.stat().st_size,
                "sha256": sha256_file(scores),
                "shape": [10, 7],
                "dtype": "float32",
            },
            "cells": [
                {
                    "cell_id": action,
                    "changing": True,
                    "cluster_separation_luma": 100.0,
                }
                for action in actions
            ],
            "evidence": [
                {
                    "path": "evidence/frame-000.png",
                    "size_bytes": evidence_frame.stat().st_size,
                    "sha256": sha256_file(evidence_frame),
                }
            ],
            "evidence_contact_sheet": {
                "path": contact.name,
                "size_bytes": contact.stat().st_size,
                "sha256": sha256_file(contact),
            },
            "human_reviewed": False,
            "training_admitted": False,
        },
    )
    _write_json(
        scan / "family_transfer_scan_validation.json",
        {
            "format_version": publisher.SCAN_VALIDATION_VERSION,
            "video_id": VIDEO_ID,
            "scan_report_sha256": sha256_file(scan_report),
            "layout_sha256": sha256_file(layout),
            "validated_cells": 7,
            "validated_actions": sorted(actions),
            "minimum_cluster_separation_luma": 100.0,
            "human_reviewed": False,
            "training_admitted": False,
        },
    )
    _write_json(
        scan / "cell_activity_complete.json",
        {
            "format_version": "madeleine.wild-cell-activity-publication.v1",
            "video_id": VIDEO_ID,
            "source_sha256": SOURCE_SHA,
            "report_sha256": sha256_file(scan_report),
            "human_reviewed": False,
            "training_admitted": False,
        },
    )

    raw_labels = decode / "labels_raw.parquet"
    labels = decode / "labels_native.parquet"
    raw_labels.write_bytes(b"raw label fixture")
    labels.write_bytes(b"native label fixture")
    decode_report = _write_json(
        decode / "decode_report.json",
        {
            "format_version": publisher.DECODE_VERSION,
            "video_id": VIDEO_ID,
            "source_video": {"sha256": SOURCE_SHA, "path": "/source.mp4"},
            "boundaries": {
                "sha256": sha256_file(boundaries),
                "human_reviewed": False,
            },
            "layout": {
                "sha256": sha256_file(layout),
                "human_reviewed": False,
            },
            "score_source": {
                "kind": "hash_bound_full_cell_scan",
                "report_sha256": sha256_file(scan_report),
                "spec_sha256": sha256_file(copied_spec),
                "scores_sha256": sha256_file(scores),
            },
            "decoded_frames": 10,
            "decoded_hours": 10 / effective_fps / 3600,
            "raw_labels": raw_labels.name,
            "raw_labels_sha256": sha256_file(raw_labels),
            "labels": labels.name,
            "labels_sha256": sha256_file(labels),
            "admitted": False,
            "rejection_reasons": [
                "gameplay boundaries were not reviewed by a human",
                "layout lacks a verified hash-bound review acceptance",
            ],
        },
    )
    _write_json(
        decode / publisher.DECODE_COMPLETION_NAME,
        {
            "format_version": publisher.DECODE_COMPLETION_VERSION,
            "video_id": VIDEO_ID,
            "admitted": False,
            "report": {
                "path": decode_report.name,
                "size_bytes": decode_report.stat().st_size,
                "sha256": sha256_file(decode_report),
            },
            "artifacts": {
                "raw_labels": {
                    "path": raw_labels.name,
                    "size_bytes": raw_labels.stat().st_size,
                    "sha256": sha256_file(raw_labels),
                },
                "labels": {
                    "path": labels.name,
                    "size_bytes": labels.stat().st_size,
                    "sha256": sha256_file(labels),
                },
            },
            "bindings": {
                "source_sha256": SOURCE_SHA,
                "layout_sha256": sha256_file(layout),
                "boundaries_sha256": sha256_file(boundaries),
                "scan_report_sha256": sha256_file(scan_report),
            },
        },
    )

    part_path = parts / f"wild_provisional_{VIDEO_ID}__r000.npz"
    part_path.write_bytes(b"NPZ fixture bytes")
    part_row = {
        "session_id": f"wild_provisional_{VIDEO_ID}__r000",
        "npz": part_path.name,
        "frames": 10,
        "source_frame_range": [0, 10],
        "pts_range_s": [0.0, 0.15],
        "sha256": sha256_file(part_path),
    }
    implementation_sha = "f" * 64
    _write_json(
        part_path.with_name(part_path.name + ".complete.json"),
        {
            "format_version": PART_COMPLETION_VERSION,
            "row": part_row,
            "npz_bytes": part_path.stat().st_size,
            "bindings": {
                "implementation_sha256": implementation_sha,
                "source_video_sha256": SOURCE_SHA,
                "labels_sha256": sha256_file(labels),
                "layout_sha256": sha256_file(layout),
                "boundaries_sha256": sha256_file(boundaries),
            },
            "arrays": {
                "frames": {"shape": [10, 128, 128, 3], "dtype": "|u1"},
                "keys": {"shape": [10, 7], "dtype": "|u1"},
            },
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
        "effective_grid_hz": effective_fps,
        "decoded_frames": 10,
        "decoded_hours": 10 / effective_fps / 3600,
        "train_ready_frames": 0,
        "train_ready_hours": 0.0,
        "provisional_trainable_frames": 10,
        "provisional_trainable_hours": 10 / effective_fps / 3600,
        "inputs": {
            "decode_report": {
                "path": decode_report.name,
                "sha256": sha256_file(decode_report),
            },
            "labels": {"path": labels.name, "sha256": sha256_file(labels)},
            "layout": {"path": layout.name, "sha256": sha256_file(layout)},
            "boundaries_sha256": sha256_file(boundaries),
            "source_video_sha256": SOURCE_SHA,
        },
        "parts": [part_row],
        "unresolved_admission_reasons": [
            "gameplay boundaries were not reviewed by a human",
            "layout lacks a verified hash-bound review acceptance",
        ],
    }
    _write_json(parts / "wild_provisional_build_report.json", build)
    _write_json(
        parts.parent / "wild_provisional_corpus_manifest.json",
        {
            "format_version": PROVISIONAL_BUILD_VERSION,
            "admission_tier": "provisional_not_train_ready",
            "videos": [build],
            "video_count": 1,
            "train_ready_hours": 0.0,
            "provisional_trainable_hours": 10 / effective_fps / 3600,
            "warning": "This manifest is not an admitted Wild20 corpus.",
        },
    )
    return root


def _set_scan_validation_policy(
    root: Path, policy: str, validation_mode: str
) -> None:
    validation_path = (
        root / "scan" / VIDEO_ID / "family_transfer_scan_validation.json"
    )
    validation = json.loads(validation_path.read_text())
    validation["validation_policy"] = policy
    validation["cell_validation"] = [
        {"cell_id": action, "validation_mode": validation_mode}
        for action in ("left", "right", "up", "down", "jump", "dash", "grab")
    ]
    _write_json(validation_path, validation)

    build_path = root / "shards" / "parts" / "wild_provisional_build_report.json"
    build = json.loads(build_path.read_text())
    build["inputs"]["scan_validation"] = {
        "path": validation_path.name,
        "sha256": sha256_file(validation_path),
    }
    _write_json(build_path, build)

    corpus_path = root / "shards" / "wild_provisional_corpus_manifest.json"
    corpus = json.loads(corpus_path.read_text())
    corpus["videos"] = [build]
    _write_json(corpus_path, corpus)


class _FakeRclone:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.copies: list[str] = []

    def run(self, command, **_kwargs):
        if command[1] == "size":
            payload = self.objects[command[2]]
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"count": 1, "bytes": len(payload)}),
            )
        if command[1] == "lsjson":
            if "--stat" in command:
                payload = self.objects[command[-1]]
                return SimpleNamespace(
                    returncode=0, stdout=json.dumps({"Size": len(payload)})
                )
            prefix = command[-1].rstrip("/") + "/"
            rows = [
                {"Path": path.removeprefix(prefix), "Size": len(payload)}
                for path, payload in sorted(self.objects.items())
                if path.startswith(prefix)
            ]
            return SimpleNamespace(returncode=0, stdout=json.dumps(rows))
        assert command[1] == "copyto"
        local = Path(command[2])
        remote = command[3]
        payload = local.read_bytes()
        if remote in self.objects and self.objects[remote] != payload:
            return SimpleNamespace(returncode=1, stdout="")
        self.objects[remote] = payload
        self.copies.append(remote)
        return SimpleNamespace(returncode=0, stdout="")

    def popen(self, command, **_kwargs):
        payload = self.objects[command[2]]

        class Process:
            stdout = io.BytesIO(payload)

            @staticmethod
            def wait() -> int:
                return 0

        return Process()


def test_publish_is_complete_marker_last_and_completed_rerun_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture(tmp_path)
    remote = _FakeRclone()
    monkeypatch.setattr(publisher.subprocess, "run", remote.run)
    monkeypatch.setattr(publisher.subprocess, "Popen", remote.popen)
    state = tmp_path / "state"

    result = publisher.publish(
        root, VIDEO_ID, state, "r2:testbucket/wild/provisional", "native60"
    )

    assert result["publication_status"] == "published"
    assert result["format_version"].endswith(".v2")
    assert remote.copies[-1].endswith("/publication-complete.json")
    manifest = json.loads((state / publisher.MANIFEST_NAME).read_text())
    assert manifest["format_version"].endswith(".v2")
    paths = {row["path"] for row in manifest["objects"]}
    assert "scan/cell_scores.f32" in paths
    assert "transfer/reference-layout.source.json" in paths
    assert "scan/evidence/frame-000.png" in paths
    assert "shards/wild_provisional_corpus_manifest.json" in paths
    assert any(path.endswith(".npz.complete.json") for path in paths)
    assert manifest["reconstruction_sources"]["raw_r2_prefix"].endswith(
        f"/raw/{VIDEO_ID}"
    )
    first_copy_count = len(remote.copies)

    repeated = publisher.publish(
        root, VIDEO_ID, state, "r2:testbucket/wild/provisional", "native60"
    )

    assert repeated["publication_status"] == "already_complete_validated"
    assert len(remote.copies) == first_copy_count


def test_partial_retry_skips_existing_verified_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture(tmp_path)
    remote = _FakeRclone()
    monkeypatch.setattr(publisher.subprocess, "run", remote.run)
    monkeypatch.setattr(publisher.subprocess, "Popen", remote.popen)
    artifacts, _ = publisher.collect_artifacts(root, VIDEO_ID, "native60")
    remote_dir = (
        "r2:testbucket/wild/provisional/native60/" + VIDEO_ID
    )
    existing = artifacts[0]
    remote.objects[f"{remote_dir}/{existing.relative_path}"] = existing.local.read_bytes()

    result = publisher.publish(
        root,
        VIDEO_ID,
        tmp_path / "retry-state",
        "r2:testbucket/wild/provisional",
        "native60",
        npz_size_only=True,
    )

    assert result["publication_status"] == "published"
    assert result["npz_verification"].startswith("remote_size_only;")
    manifest = json.loads(
        (tmp_path / "retry-state" / publisher.MANIFEST_NAME).read_text()
    )
    npz_rows = [row for row in manifest["objects"] if row["path"].endswith(".npz")]
    assert npz_rows
    assert all(
        row["remote_verification"].startswith("remote_size_only;")
        for row in npz_rows
    )
    assert f"{remote_dir}/{existing.relative_path}" not in remote.copies
    assert remote.copies[-1] == f"{remote_dir}/{publisher.COMPLETION_NAME}"


def test_collect_rejects_tampered_sidecar_score_and_wrong_cadence(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    sidecar_path = next((root / "shards" / "parts").glob("*.npz.complete.json"))
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["bindings"]["source_video_sha256"] = "0" * 64
    _write_json(sidecar_path, sidecar)
    with pytest.raises(ValueError, match="sidecar bindings"):
        publisher.collect_artifacts(root, VIDEO_ID, "native60")

    root = _fixture(tmp_path / "fresh")
    scores = root / "scan" / VIDEO_ID / "cell_scores.f32"
    scores.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cell scores (SHA-256|size) mismatch"):
        publisher.collect_artifacts(root, VIDEO_ID, "native60")

    root = _fixture(tmp_path / "cadence")
    with pytest.raises(ValueError, match="does not match native30"):
        publisher.collect_artifacts(root, VIDEO_ID, "native30")


@pytest.mark.parametrize(
    ("effective_fps", "cadence_tier"),
    ((23.976, "native24"), (29.97, "native30"), (60.0, "native60")),
)
def test_collect_accepts_each_explicit_cadence_tier(
    tmp_path: Path, effective_fps: float, cadence_tier: str
) -> None:
    root = _fixture(tmp_path, effective_fps=effective_fps)

    artifacts, build = publisher.collect_artifacts(root, VIDEO_ID, cadence_tier)

    assert artifacts
    assert build["effective_grid_hz"] == effective_fps


@pytest.mark.parametrize(
    ("fps_min", "fps_max", "cadence_tier"),
    (("23", "25", "native24"), ("29", "31", "native30"), ("50", "61", "native60")),
)
@pytest.mark.requires_private_artifacts("harvest/run_layout_family_worker.sh")
def test_family_worker_accepts_only_declared_cadence_bounds(
    tmp_path: Path, fps_min: str, fps_max: str, cadence_tier: str
) -> None:
    script = Path(__file__).resolve().parents[1] / "harvest/run_layout_family_worker.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            VIDEO_ID,
            "reference",
            "unused-layout.json",
            str(tmp_path),
            fps_min,
            fps_max,
            cadence_tier,
            "full-sha256",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 3
    assert result.stderr.strip() == "materialization marker missing"


@pytest.mark.requires_private_artifacts("harvest/run_layout_family_worker.sh")
def test_family_worker_rejects_loose_native24_bounds(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "harvest/run_layout_family_worker.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            VIDEO_ID,
            "reference",
            "unused-layout.json",
            str(tmp_path),
            "22",
            "26",
            "native24",
            "full-sha256",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 4
    assert "native24 requires validated 23..25 Hz bounds" in result.stderr


@pytest.mark.parametrize("cadence_tier", ("native24", "native30", "native60"))
@pytest.mark.requires_private_artifacts("harvest/run_family_publication_watch.sh")
def test_publication_watcher_accepts_each_explicit_cadence_tier(
    tmp_path: Path, cadence_tier: str
) -> None:
    script = Path(__file__).resolve().parents[1] / "harvest/run_family_publication_watch.sh"
    result = subprocess.run(
        ["bash", str(script), VIDEO_ID, str(tmp_path), cadence_tier, "invalid-mode"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 4
    assert "verify mode must be full-sha256 or npz-size-only" in result.stderr


def test_collect_requires_transfer_and_scan_survey_bindings_to_match(
    tmp_path: Path,
) -> None:
    root = _fixture(tmp_path)
    evidence_path = root / "transfer" / "transfer_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["bindings"]["target_survey_sha256"] = "9" * 64
    _write_json(evidence_path, evidence)
    with pytest.raises(ValueError, match="survey binding differs"):
        publisher.collect_artifacts(root, VIDEO_ID, "native60")


def test_collect_requires_decode_completion_admission_and_artifact_paths(
    tmp_path: Path,
) -> None:
    root = _fixture(tmp_path)
    completion_path = root / "decode" / publisher.DECODE_COMPLETION_NAME
    completion = json.loads(completion_path.read_text())
    completion["admitted"] = True
    _write_json(completion_path, completion)
    with pytest.raises(ValueError, match="decode completion identity/version"):
        publisher.collect_artifacts(root, VIDEO_ID, "native60")

    root = _fixture(tmp_path / "wrong-path")
    completion_path = root / "decode" / publisher.DECODE_COMPLETION_NAME
    completion = json.loads(completion_path.read_text())
    completion["artifacts"]["raw_labels"]["path"] = "wrong.parquet"
    _write_json(completion_path, completion)
    with pytest.raises(ValueError, match="decode completion label binding"):
        publisher.collect_artifacts(root, VIDEO_ID, "native60")


def test_v2_publisher_recomputes_legacy_scan_strength(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    validation_path = (
        root / "scan" / VIDEO_ID / "family_transfer_scan_validation.json"
    )
    validation = json.loads(validation_path.read_text())
    validation["minimum_cluster_separation_luma"] = 10.0
    _write_json(validation_path, validation)
    with pytest.raises(ValueError, match="independently strong absolute gaps"):
        publisher.collect_artifacts(root, VIDEO_ID, "native60")


@pytest.mark.parametrize(
    ("policy", "validation_mode"),
    (
        ("absolute_luma_or_low_dynamic_binary_v1", "absolute_luma_gap"),
        ("absolute_luma_or_low_dynamic_binary_v1", "low_dynamic_binary"),
        (
            "absolute_luma_or_disjoint_stable_pressed_or_low_dynamic_binary_v2",
            "absolute_luma_gap",
        ),
        (
            "absolute_luma_or_disjoint_stable_pressed_or_low_dynamic_binary_v2",
            "disjoint_stable_pressed_state",
        ),
        (
            "absolute_luma_or_disjoint_stable_pressed_or_low_dynamic_binary_v2",
            "low_dynamic_binary",
        ),
    ),
)
def test_collect_accepts_exact_modes_for_each_scan_validation_policy(
    tmp_path: Path, policy: str, validation_mode: str
) -> None:
    root = _fixture(tmp_path)
    _set_scan_validation_policy(root, policy, validation_mode)

    artifacts, _ = publisher.collect_artifacts(root, VIDEO_ID, "native60")

    assert artifacts


@pytest.mark.parametrize(
    ("policy", "validation_mode", "error"),
    (
        (
            "absolute_luma_or_low_dynamic_binary_v1",
            "disjoint_stable_pressed_state",
            "lacks every accepted cell",
        ),
        (
            "absolute_luma_or_low_dynamic_binary_v1",
            "unknown_mode",
            "lacks every accepted cell",
        ),
        (
            "absolute_luma_or_disjoint_stable_pressed_or_low_dynamic_binary_v2",
            "unknown_mode",
            "lacks every accepted cell",
        ),
        ("unknown_policy", "absolute_luma_gap", "unsupported scan-validation policy"),
    ),
)
def test_collect_rejects_modes_outside_the_declared_scan_validation_policy(
    tmp_path: Path, policy: str, validation_mode: str, error: str
) -> None:
    root = _fixture(tmp_path)
    _set_scan_validation_policy(root, policy, validation_mode)

    with pytest.raises(ValueError, match=error):
        publisher.collect_artifacts(root, VIDEO_ID, "native60")


def test_remote_inventory_failure_is_not_treated_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    with pytest.raises(RuntimeError, match="inventory failed"):
        publisher._remote_inventory("r2:bucket/prefix")
