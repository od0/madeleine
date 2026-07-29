from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from harvest.fetch_wild import sha256_file
import harvest.publish_wild_derived as publisher
from harvest.publish_wild_derived import collect_artifacts, publish_derived
from harvest.wild_boundaries import BOUNDARIES_VERSION
from tests.test_offset_acceptance import _accepted_fixture


def _publisher_fixture(tmp_path: Path) -> dict[str, Path | str]:
    layout, acceptance, calibration, contact = _accepted_fixture(tmp_path)
    layout_acceptance = tmp_path / "layout-review" / "layout_acceptance.json"
    source_hash = "a" * 64
    boundaries = tmp_path / "boundaries.json"
    boundaries.write_text(json.dumps({
        "format_version": BOUNDARIES_VERSION,
        "video_id": "acceptance_test",
        "source_sha256": source_hash,
        "wall_clock_range_s": [0.0, 10.0],
        "allowed_ranges_s": [[0.0, 10.0]],
        "human_reviewed": True,
        "reviewer": "Test Reviewer",
        "reviewer_kind": "human_with_ai_assistance",
        "evidence": ["fixture"],
    }, indent=2) + "\n")

    decoded = tmp_path / "decoded"
    decoded.mkdir()
    raw_labels = decoded / "labels_raw.parquet"
    native_labels = decoded / "labels_native.parquet"
    raw_labels.write_bytes(b"raw parquet fixture")
    native_labels.write_bytes(b"native parquet fixture")
    decode_report = decoded / "decode_report.json"
    decode_report.write_text(json.dumps({
        "format_version": "madeleine.wild-decode.v1",
        "video_id": "acceptance_test",
        "source_video": {"path": "/source.mp4", "sha256": source_hash},
        "boundaries": {
            "path": boundaries.name,
            "sha256": sha256_file(boundaries),
        },
        "layout": {
            "path": layout.name,
            "sha256": sha256_file(layout),
            "review_acceptance": {
                "sha256": sha256_file(layout_acceptance),
                "reviewer_kind": "human_with_ai_assistance",
                "human_reviewed": True,
            },
        },
        "timing": {
            "pts": {"effective_fps": 60.0},
            "offset_acceptance": {
                "sha256": sha256_file(acceptance),
                "reviewer_kind": "human_with_ai_assistance",
                "human_reviewed": True,
            },
        },
        "decoded_hours": 1.0,
        "raw_labels": raw_labels.name,
        "raw_labels_sha256": sha256_file(raw_labels),
        "labels": native_labels.name,
        "labels_sha256": sha256_file(native_labels),
        "admitted": True,
        "rejection_reasons": [],
    }, indent=2) + "\n")

    shards = tmp_path / "shards"
    shards.mkdir()
    session_id = "wild_acceptance_test__r000"
    shard = shards / f"{session_id}.npz"
    shard.write_bytes(b"npz fixture")
    build_report = shards / "wild_build_report.json"
    build_report.write_text(json.dumps({
        "format_version": "madeleine.wild-shards.v1",
        "video_id": "acceptance_test",
        "effective_grid_hz": 60.0,
        "decoded_hours": 1.0,
        "train_ready_frames": 10,
        "train_ready_hours": 10 / 60 / 3600,
        "inputs": {
            "decode_report": {
                "path": decode_report.name,
                "sha256": sha256_file(decode_report),
            },
            "labels": {
                "path": native_labels.name,
                "sha256": sha256_file(native_labels),
            },
            "layout": {
                "path": layout.name,
                "sha256": sha256_file(layout),
            },
            "boundaries_sha256": sha256_file(boundaries),
            "source_video_sha256": source_hash,
        },
        "parts": [{
            "session_id": session_id,
            "npz": shard.name,
            "sha256": sha256_file(shard),
            "frames": 10,
            "source_frame_range": [0, 10],
            "pts_range_s": [0.0, 0.15],
        }],
    }, indent=2) + "\n")

    return {
        "video_id": "acceptance_test",
        "input_layout_path": tmp_path / "layout.reviewed.json",
        "layout_acceptance_path": layout_acceptance,
        "layout_path": layout,
        "boundaries_path": boundaries,
        "calibration_path": calibration,
        "calibration_sha256_path": calibration.with_suffix(".sha256"),
        "contact_sheet_path": contact,
        "acceptance_path": acceptance,
        "decode_report_path": decode_report,
        "labels_raw_path": raw_labels,
        "labels_native_path": native_labels,
        "build_report_path": build_report,
        "shard_dir": shards,
    }


def test_publish_uses_only_allowlist_and_completion_is_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _publisher_fixture(tmp_path)
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(publisher, "_remote_files", lambda remote: set())

    def fake_copy(local, remote, *, expected_sha256, expected_size):
        local = Path(local)
        assert sha256_file(local) == expected_sha256
        assert local.stat().st_size == expected_size
        calls.append((local, remote))
        return {
            "relative_path": remote.rsplit("/", 1)[-1],
            "size_bytes": expected_size,
            "sha256": expected_sha256,
            "verified": "sha256_readback",
        }

    monkeypatch.setattr(publisher, "_copy_verified", fake_copy)
    state = tmp_path / "publication-state"
    result = publish_derived(
        **inputs,
        state_dir=state,
        remote_root="object-store:example-bucket/wild/v1/derived",
    )

    expected_relative = {
        "layout/review_packet/layout.draft.json",
        "layout/review_packet/review_manifest.json",
        "layout/review_packet/layout_acceptance.json",
        "layout/review_packet/geometry.png",
        "layout/review_packet/cell_states.json",
        "layout/review_packet/cell_states.png",
        "layout/review_packet/frames/released.jpg",
        "layout/review_packet/frames/pressed.jpg",
        "layout/reviewed_unmeasured.json",
        "layout/final.json",
        "boundaries/boundaries.json",
        "calibration/offset_calibration.json",
        "calibration/offset_calibration.sha256",
        "calibration/dash_offset_contact.png",
        "calibration/offset_acceptance.json",
        "decoded/decode_report.json",
        "decoded/labels_raw.parquet",
        "decoded/labels_native.parquet",
        "shards/wild_build_report.json",
        "shards/wild_acceptance_test__r000.npz",
        "derived_objects.json",
    }
    uploaded = {
        remote.split("/acceptance_test/", 1)[1] for _, remote in calls[:-1]
    }
    assert uploaded == expected_relative
    assert calls[-1][1].endswith("/acceptance_test/derived_complete.json")
    assert result["object_count"] == len(expected_relative)
    assert all(row["verified"] == "sha256_readback" for row in result["objects"])
    manifest = json.loads((state / "derived_objects.json").read_text())
    assert {row["relative_path"] for row in manifest["objects"]} == (
        expected_relative - {"derived_objects.json"}
    )


def test_collect_refuses_stale_local_shard_and_tampered_label(tmp_path: Path) -> None:
    inputs = _publisher_fixture(tmp_path)
    shard_dir = Path(inputs["shard_dir"])
    (shard_dir / "stale.npz").write_bytes(b"stale")
    with pytest.raises(ValueError, match="stale=.*stale.npz"):
        collect_artifacts(**inputs)
    (shard_dir / "stale.npz").unlink()
    Path(inputs["labels_native_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="native-label bytes"):
        collect_artifacts(**inputs)


def test_publish_refuses_remote_completion_stale_objects_and_state_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _publisher_fixture(tmp_path)
    monkeypatch.setattr(
        publisher, "_remote_files", lambda remote: {"derived_complete.json"}
    )
    with pytest.raises(FileExistsError, match="already complete"):
        publish_derived(
            **inputs,
            state_dir=tmp_path / "state-complete",
            remote_root="object-store:example-bucket/wild/v1/derived",
        )

    monkeypatch.setattr(
        publisher, "_remote_files", lambda remote: {"shards/stale.npz"}
    )
    with pytest.raises(ValueError, match="stale objects"):
        publish_derived(
            **inputs,
            state_dir=tmp_path / "state-stale",
            remote_root="object-store:example-bucket/wild/v1/derived",
        )

    state = tmp_path / "existing-state"
    state.mkdir()
    with pytest.raises(FileExistsError, match="reuse publication state_dir"):
        publish_derived(
            **inputs,
            state_dir=state,
            remote_root="object-store:example-bucket/wild/v1/derived",
        )


def test_copy_is_immutable_and_sha256_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "one.bin"
    local.write_bytes(b"one exact object")
    commands: list[list[str]] = []

    class Result:
        returncode = 0

    def fake_run(command, **kwargs):
        commands.append(command)
        return Result()

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)
    monkeypatch.setattr(
        publisher,
        "_remote_sha256",
        lambda remote: (sha256_file(local), local.stat().st_size),
    )
    result = publisher._copy_verified(
        local,
        "object-store:example-bucket/wild/v1/derived/id/one.bin",
        expected_sha256=sha256_file(local),
        expected_size=local.stat().st_size,
    )
    assert "--immutable" in commands[0]
    assert commands[0][0:2] == ["rclone", "copyto"]
    assert result["verified"] == "sha256_readback"


def test_remote_sha256_counts_each_streamed_byte_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"remote bytes streamed exactly once"

    class Process:
        stdout = io.BytesIO(payload)

        @staticmethod
        def wait() -> int:
            return 0

    monkeypatch.setattr(
        publisher.subprocess,
        "Popen",
        lambda *args, **kwargs: Process(),
    )

    digest, size = publisher._remote_sha256(
        "object-store:example-bucket/wild/v1/derived/id/object.bin"
    )

    assert digest == hashlib.sha256(payload).hexdigest()
    assert size == len(payload)


@pytest.mark.parametrize(
    "video_id,remote_root",
    [
        ("../escape", "object-store:example-bucket/wild/v1/derived"),
        ("acceptance_test", "bad root"),
    ],
)
def test_publisher_rejects_unsafe_identifiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    video_id: str,
    remote_root: str,
) -> None:
    inputs = _publisher_fixture(tmp_path)
    inputs["video_id"] = video_id
    monkeypatch.setattr(publisher, "_remote_files", lambda remote: set())
    with pytest.raises(ValueError):
        publish_derived(
            **inputs,
            state_dir=tmp_path / "state",
            remote_root=remote_root,
        )
