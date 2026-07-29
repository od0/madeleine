from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

from data.schema import KEY_ORDER
from harvest.accept_wild_offset import accept_offset
from harvest.build_wild import (
    _validate_low_dynamic_scan,
    _validate_provisional_decode,
    activity_mask,
    build_wild_video,
    contiguous_true_runs,
    update_corpus_manifest,
    update_provisional_corpus_manifest,
)
from harvest.decode_wild import (
    apply_temporal_offset,
    decode_video,
    mask_frame,
    mask_rects_in_gameplay,
    masked_resize,
)
from harvest.fetch_wild import (
    build_fetch_command,
    load_candidate,
    parse_timestamp,
    probe_media,
    resolve_run_window,
    sha256_file,
    summarize_pts,
    url_start_time,
)
from harvest.finalize_provisional_shards import finalize_provisional_corpus
from harvest.transfer_wild_layout_family import _sample_diagnostics, validate_full_scan
from harvest.wild_layout import SCHEMA_VERSION, WildLayout, rect_to_pixels
from harvest.wild_boundaries import BOUNDARIES_VERSION, WildBoundaries
import harvest.worker_wild as wild_worker
from tests.test_offset_acceptance import _write_calibration
from tests.test_layout_acceptance import _layout_review_fixture


W, H, FPS, N = 320, 180, 60.0, 600


def test_family_transfer_four_frame_evidence_is_bounded_only(tmp_path: Path) -> None:
    layout = WildLayout.from_dict({**_layout_dict(), "human_reviewed": False})
    frames = []
    for index in range(4):
        path = tmp_path / f"sample-{index}.png"
        image = np.full((H, W), 20 + index * 30, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        frames.append({"path": path.name, "sha256": sha256_file(path)})
    survey = {"frames": frames}

    with pytest.raises(ValueError, match="at least 8 exact survey frames"):
        _sample_diagnostics(tmp_path, survey, layout, (W, H))

    bounded = _sample_diagnostics(
        tmp_path, survey, layout, (W, H), bounded=True
    )
    assert len(bounded) == len(KEY_ORDER)


def _provisional_policy_fixture(fps: float, single_frame_runs: int) -> dict:
    return {
        "admitted": False,
        "rejection_reasons": ["HUD compositor offset is unmeasured"],
        "layout": {"inference_confidence": 0.9},
        "timing": {
            "pts": {
                "effective_fps": fps,
                "nonmonotonic_intervals": 0,
                "large_gap_intervals": 0,
            }
        },
        "cell_qc": [{"cell_id": "cell", "cluster_separation": 100.0}],
        "action_qc": {
            action: {"transitions": 200, "single_frame_runs": single_frame_runs}
            for action in KEY_ORDER
        },
    }


def test_provisional_flicker_policy_is_explicitly_cadence_aware() -> None:
    native60 = _validate_provisional_decode(
        _provisional_policy_fixture(60.0, single_frame_runs=5)
    )
    assert native60["cadence_tier"] == "native60"
    assert native60["max_single_frame_run_fraction"] == 0.05
    native30 = _validate_provisional_decode(
        _provisional_policy_fixture(29.97, single_frame_runs=8)
    )
    assert native30["cadence_tier"] == "native30"
    assert native30["max_single_frame_run_fraction"] == 0.10
    native24 = _validate_provisional_decode(
        _provisional_policy_fixture(23.976, single_frame_runs=8)
    )
    assert native24["cadence_tier"] == "native24"
    assert native24["max_single_frame_run_fraction"] == 0.10
    with pytest.raises(ValueError, match="single-frame flicker"):
        _validate_provisional_decode(
            _provisional_policy_fixture(60.0, single_frame_runs=8)
        )
    with pytest.raises(ValueError, match="single-frame flicker"):
        _validate_provisional_decode(
            _provisional_policy_fixture(30.0, single_frame_runs=11)
        )
    with pytest.raises(ValueError, match="single-frame flicker"):
        _validate_provisional_decode(
            _provisional_policy_fixture(24.0, single_frame_runs=11)
        )


def test_low_dynamic_builder_exception_is_explicit_and_cell_scoped() -> None:
    decode = _provisional_policy_fixture(60.0, single_frame_runs=0)
    decode["cell_qc"][0]["cluster_separation"] = 15.0
    with pytest.raises(ValueError, match="weak state separation"):
        _validate_provisional_decode(decode)
    accepted = _validate_provisional_decode(
        decode, low_dynamic_cells={"cell"}
    )
    assert accepted["min_cell_separation"] == 20.0


def test_builder_accepts_hash_bound_v2_scan_validation(tmp_path: Path) -> None:
    validation_path = tmp_path / "family_transfer_scan_validation.json"
    validation_path.write_text(json.dumps({
        "format_version": "madeleine.wild-layout-family-scan-validation.v1",
        "video_id": "v183",
        "validation_policy": (
            "absolute_luma_or_disjoint_stable_pressed_or_"
            "low_dynamic_binary_v2"
        ),
        "scan_report_sha256": "a" * 64,
        "layout_sha256": "b" * 64,
        "cell_validation": [{
            "cell_id": "printed_up",
            "validation_mode": "disjoint_stable_pressed_state",
            "absolute_gap_luma": 225.0,
            "inter_cluster_support_gap_luma": 177.0,
            "pressed_state_mad_luma": 0.0,
            "pressed_state_range_luma": 2.0,
            "minority_frames": 40_000,
            "single_frame_positive_run_fraction": 0.0,
        }],
    }))
    decode = {
        "video_id": "v183",
        "score_source": {
            "kind": "hash_bound_full_cell_scan",
            "report_sha256": "a" * 64,
        },
        "layout": {"sha256": "b" * 64},
    }
    allowed, summary = _validate_low_dynamic_scan(decode, validation_path)
    assert allowed == {"printed_up"}
    assert summary is not None
    assert summary["policy"].endswith("_v2")


def test_full_scan_records_strict_low_dynamic_binary_evidence(
    tmp_path: Path,
) -> None:
    raw_layout = {**_layout_dict("low_dynamic"), "human_reviewed": False}
    raw_layout["cells"] = [
        {**cell, "decoder": "luma", "reference_rect": None}
        for cell in raw_layout["cells"]
    ]
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(raw_layout))
    layout = WildLayout.load(layout_path)
    frame_count = 2_000
    scores = np.empty((frame_count, len(layout.cells)), dtype=np.float32)
    scores[:1_000] = 8.0
    scores[1_000:] = 23.0
    score_path = tmp_path / "cell_scores.f32"
    score_path.write_bytes(scores.tobytes())
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"video_id": layout.video_id}))
    report_path = tmp_path / "cell_activity_scan.json"
    report_path.write_text(json.dumps({
        "video_id": layout.video_id,
        "spec": {"path": spec_path.name, "sha256": sha256_file(spec_path)},
        "scores": {
            "path": score_path.name,
            "sha256": sha256_file(score_path),
            "shape": list(scores.shape),
            "dtype": "float32",
        },
        "cells": [
            {
                "cell_id": cell.cell_id,
                "changing": True,
                "threshold": 15.5,
                "cluster_separation_luma": 15.0,
                "minority_frames": 1_000,
                "positive_runs": 100,
                "single_frame_positive_runs": 0,
            }
            for cell in layout.cells
        ],
    }))
    result = validate_full_scan(report_path, layout_path)
    assert {
        row["validation_mode"] for row in result["cell_validation"]
    } == {"low_dynamic_binary"}


def test_full_scan_rejects_large_absolute_gap_with_noisy_state_clusters(
    tmp_path: Path,
) -> None:
    raw_layout = {**_layout_dict("noisy_absolute"), "human_reviewed": False}
    raw_layout["cells"] = [
        {**cell, "decoder": "luma", "reference_rect": None}
        for cell in raw_layout["cells"]
    ]
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(raw_layout))
    layout = WildLayout.load(layout_path)
    frame_count = 2_000
    # The state medians differ by 100 luma points, but each state is broad
    # enough that the robust decoder separation is far below the builder's
    # required 20.  Absolute gap alone must not let this through preflight.
    low = np.tile(np.linspace(20.0, 80.0, 1_000, dtype=np.float32)[:, None],
                  (1, len(layout.cells)))
    high = np.tile(np.linspace(120.0, 180.0, 1_000, dtype=np.float32)[:, None],
                   (1, len(layout.cells)))
    scores = np.concatenate([low, high], axis=0)
    score_path = tmp_path / "cell_scores.f32"
    score_path.write_bytes(scores.tobytes())
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"video_id": layout.video_id}))
    report_path = tmp_path / "cell_activity_scan.json"
    report_path.write_text(json.dumps({
        "video_id": layout.video_id,
        "spec": {"path": spec_path.name, "sha256": sha256_file(spec_path)},
        "scores": {
            "path": score_path.name,
            "sha256": sha256_file(score_path),
            "shape": list(scores.shape),
            "dtype": "float32",
        },
        "cells": [
            {
                "cell_id": cell.cell_id,
                "changing": True,
                "threshold": 100.0,
                "cluster_separation_luma": 100.0,
                "minority_frames": 1_000,
                "positive_runs": 100,
                "single_frame_positive_runs": 0,
            }
            for cell in layout.cells
        ],
    }))
    with pytest.raises(ValueError, match="no-valid-separation-policy"):
        validate_full_scan(report_path, layout_path)


def test_full_scan_accepts_disjoint_support_with_stable_pressed_state(
    tmp_path: Path,
) -> None:
    raw_layout = {**_layout_dict("stable_pressed"), "human_reviewed": False}
    raw_layout["cells"] = [
        {**cell, "decoder": "luma", "reference_rect": None}
        for cell in raw_layout["cells"]
    ]
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(raw_layout))
    layout = WildLayout.load(layout_path)
    frame_count = 2_000
    # This reproduces the important v183 scan shape: the translucent released
    # state is broad, but every observed pressed-state value is an opaque,
    # saturated fill with a large empty interval between the two supports.
    low = np.tile(np.linspace(16.0, 76.0, 1_000, dtype=np.float32)[:, None],
                  (1, len(layout.cells)))
    high = np.full((1_000, len(layout.cells)), 253.0, dtype=np.float32)
    scores = np.concatenate([low, high], axis=0)
    score_path = tmp_path / "cell_scores.f32"
    score_path.write_bytes(scores.tobytes())
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"video_id": layout.video_id}))
    report_path = tmp_path / "cell_activity_scan.json"
    report_path.write_text(json.dumps({
        "video_id": layout.video_id,
        "spec": {"path": spec_path.name, "sha256": sha256_file(spec_path)},
        "scores": {
            "path": score_path.name,
            "sha256": sha256_file(score_path),
            "shape": list(scores.shape),
            "dtype": "float32",
        },
        "cells": [
            {
                "cell_id": cell.cell_id,
                "changing": True,
                "threshold": 149.5,
                "cluster_separation_luma": 207.0,
                "minority_frames": 1_000,
                "positive_runs": 100,
                "single_frame_positive_runs": 0,
            }
            for cell in layout.cells
        ],
    }))
    result = validate_full_scan(report_path, layout_path)
    assert result["validation_policy"].endswith("_v2")
    assert {
        row["validation_mode"] for row in result["cell_validation"]
    } == {"disjoint_stable_pressed_state"}
    for row in result["cell_validation"]:
        assert row["inter_cluster_support_gap_luma"] == pytest.approx(177.0)
        assert row["pressed_state_range_luma"] == 0.0
        # The old symmetric-MAD rule rejected this legitimately asymmetric
        # state shape, which is why the disjoint-support rule is needed.
        assert row["decoder_cluster_separation_floor1"] < 20.0


def _layout_dict(video_id: str = "wild_test") -> dict:
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
        "gameplay_rect": [0.0, 0.0, 0.85, 1.0],
        "gameplay_rect_source": "synthetic_review",
        "gameplay_rect_confidence": 1.0,
        "mask_rects": [[0.02, 0.78, 0.66, 0.20]],
        "cells": cells,
        "inference_source": "synthetic_fixture",
        "inference_confidence": 1.0,
        "human_reviewed": True,
        "evidence_frames_s": [1.0, 5.0, 9.0],
        "temporal_offset_frames": 0,
        "temporal_offset_source": "unmeasured",
        "temporal_offset_confidence": 0.0,
    }


def _write_video(path: Path, layout: WildLayout) -> np.ndarray:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )
    assert writer.isOpened()
    truth = np.zeros((N, len(KEY_ORDER)), dtype=bool)
    for frame_index in range(N):
        # Moving background establishes that local contrast—not absolute
        # brightness—is doing the decoding.
        base = 55 + (frame_index * 3) % 90
        frame = np.full((H, W, 3), base, dtype=np.uint8)
        for key_index, cell in enumerate(layout.cells):
            period = 32 + key_index * 5
            pressed = (frame_index % period) < (7 + key_index)
            truth[frame_index, key_index] = pressed
            sx0, sy0, sx1, sy1 = rect_to_pixels(cell.sample_rect, W, H)
            rx0, ry0, rx1, ry1 = rect_to_pixels(cell.reference_rect, W, H)  # type: ignore[arg-type]
            frame[ry0:ry1, rx0:rx1] = 35
            frame[sy0:sy1, sx0:sx1] = 220 if pressed else 35
        writer.write(frame)
    writer.release()
    return truth


def test_layout_requires_complete_observable_schema_and_mask_coverage() -> None:
    layout = WildLayout.from_dict(_layout_dict())
    assert {cell.action for cell in layout.cells} == set(KEY_ORDER)
    missing = _layout_dict()
    missing["cells"] = missing["cells"][:-1]
    with pytest.raises(ValueError, match="missing grab"):
        WildLayout.from_dict(missing)
    uncovered = _layout_dict()
    uncovered["mask_rects"] = [[0.0, 0.0, 0.2, 0.2]]
    with pytest.raises(ValueError, match="not covered"):
        WildLayout.from_dict(uncovered)


def test_mask_frame_zeros_the_answer_key() -> None:
    layout = WildLayout.from_dict(_layout_dict())
    frame = np.full((H, W, 3), 255, dtype=np.uint8)
    masked = mask_frame(frame, layout)
    x0, y0, x1, y1 = rect_to_pixels(layout.mask_rects[0], W, H)
    assert masked[y0:y1, x0:x1].max() == 0
    assert frame[y0:y1, x0:x1].max() == 255  # copy by default


def test_gameplay_crop_excludes_external_hud_before_resize() -> None:
    raw = _layout_dict()
    raw["gameplay_rect"] = [0.0, 0.0, 0.85, 0.70]
    layout = WildLayout.from_dict(raw)
    frame = np.full((H, W, 3), 20, dtype=np.uint8)
    # Bright answer pixels live wholly below the reviewed gameplay crop.
    for cell in layout.cells:
        x0, y0, x1, y1 = rect_to_pixels(cell.sample_rect, W, H)
        frame[y0:y1, x0:x1] = 255
    assert mask_rects_in_gameplay(layout) == []
    cropped = masked_resize(frame, layout, 64)
    assert cropped.max() == 20


def test_fetch_command_is_single_fragment_and_uses_deno(tmp_path: Path) -> None:
    command = build_fetch_command("https://youtu.be/abc", tmp_path / "x.%(ext)s")
    assert command[command.index("--js-runtimes") + 1] == "deno:deno"
    assert command[command.index("--concurrent-fragments") + 1] == "1"
    assert "--no-playlist" in command
    assert "--no-progress" in command
    selector = command[command.index("-f") + 1]
    assert selector == (
        "bv*[height<=720][fps>=50]/bv*[height<=720]/b[height<=720]/b"
    )


def test_candidate_loader_accepts_pretty_json_and_rejects_multiple_rows(
    tmp_path: Path,
) -> None:
    candidate = {
        "video_id": "abc",
        "url": "https://youtu.be/abc",
        "duration_s": 1.0,
    }
    pretty = tmp_path / "candidate.json"
    pretty.write_text(json.dumps(candidate, indent=2) + "\n")
    assert load_candidate(pretty) == candidate

    multiple = tmp_path / "multiple.jsonl"
    multiple.write_text(json.dumps(candidate) + "\n" + json.dumps(candidate) + "\n")
    with pytest.raises(ValueError, match="exactly one valid JSON object"):
        load_candidate(multiple)


def test_upload_completion_is_published_after_sha256_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "abc.mp4"
    video.write_bytes(b"video bytes")
    (tmp_path / "abc.info.json").write_text("{}")
    fetch = {"video_id": "abc", "source_file": video.name}
    (tmp_path / "fetch.json").write_text(json.dumps(fetch))
    commands: list[list[str]] = []
    monkeypatch.setattr(wild_worker, "_run", lambda command: commands.append(command))

    def readback(remote: str) -> tuple[str, int]:
        local = tmp_path / remote.rsplit("/", 1)[-1]
        return sha256_file(local), local.stat().st_size

    monkeypatch.setattr(wild_worker, "_remote_sha256", readback)
    report = wild_worker.upload_verified(
        tmp_path, fetch, "object-store:example-bucket/wild/v1/raw"
    )
    assert report["video_id"] == "abc"
    assert all(row["verified"] == "sha256_readback" for row in report["objects"])
    assert commands[-1][3].endswith("/abc/upload_complete.json")
    assert (tmp_path / "upload_complete.json").is_file()


def test_legacy_pts_publication_uses_separate_evidence_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "abc.mp4"
    video.write_bytes(b"legacy video")
    fetch = {
        "format_version": "madeleine.wild-fetch.v1",
        "video_id": "abc",
        "source_file": video.name,
        "sha256": sha256_file(video),
    }
    np.save(tmp_path / "frame_pts.npy", np.arange(10) / 60.0)
    pts_path = tmp_path / "frame_pts.npy"
    (tmp_path / "frame_pts.json").write_text(json.dumps({
        "format_version": "madeleine.wild-pts.v1",
        "source_sha256": fetch["sha256"],
        "sha256": sha256_file(pts_path),
        "frames": 10,
    }))
    commands: list[list[str]] = []
    monkeypatch.setattr(wild_worker, "_run", lambda command: commands.append(command))

    def readback(remote: str) -> tuple[str, int]:
        local = tmp_path / remote.rsplit("/", 1)[-1]
        return sha256_file(local), local.stat().st_size

    monkeypatch.setattr(wild_worker, "_remote_sha256", readback)
    report = wild_worker.publish_pts_evidence(
        tmp_path, fetch, "object-store:example-bucket/wild/v1/evidence"
    )
    assert report["remote_dir"].endswith("/evidence/abc")
    assert all("/raw/" not in command[3] for command in commands)
    assert commands[-1][3].endswith("pts_evidence_complete.json")


def test_boundaries_require_one_reviewed_gameplay_range_policy() -> None:
    base = {
        "format_version": BOUNDARIES_VERSION,
        "video_id": "abc",
        "source_sha256": "a" * 64,
        "wall_clock_range_s": [10.0, 100.0],
        "human_reviewed": True,
        "reviewer": "reviewer",
        "reviewer_kind": "human",
    }
    allowed = WildBoundaries.from_dict({**base, "allowed_ranges_s": [[20, 40]]})
    assert allowed.gameplay_mask(np.asarray([15, 25, 50])).tolist() == [False, True, False]
    with pytest.raises(ValueError, match="exactly one"):
        WildBoundaries.from_dict({
            **base, "allowed_ranges_s": [[20, 40]], "excluded_ranges_s": []
        })


def test_boundary_human_gate_is_derived_from_explicit_reviewer_kind() -> None:
    base = {
        "format_version": BOUNDARIES_VERSION,
        "video_id": "abc",
        "source_sha256": "a" * 64,
        "wall_clock_range_s": [10.0, 100.0],
        "allowed_ranges_s": [[20.0, 40.0]],
        "reviewer": "OpenAI Codex visual draft",
        "reviewer_kind": "ai_agent",
    }
    ai_reviewed = WildBoundaries.from_dict({**base, "human_reviewed": False})
    assert ai_reviewed.reviewer_kind == "ai_agent"
    assert ai_reviewed.human_reviewed is False
    with pytest.raises(ValueError, match="derived from reviewer_kind"):
        WildBoundaries.from_dict({**base, "human_reviewed": True})


def test_legacy_boundaries_without_reviewer_kind_fail_closed() -> None:
    with pytest.raises(ValueError, match="lacks reviewer_kind provenance"):
        WildBoundaries.from_dict({
            "format_version": "madeleine.wild-boundaries.v1",
            "video_id": "abc",
            "source_sha256": "a" * 64,
            "wall_clock_range_s": [10.0, 100.0],
            "allowed_ranges_s": [[20.0, 40.0]],
            "human_reviewed": True,
            "reviewer": "unknown legacy reviewer",
        })


def test_timestamp_and_run_window_resolution() -> None:
    assert parse_timestamp("1h02m03.5s") == pytest.approx(3723.5)
    assert url_start_time("https://youtu.be/x?t=2m3s") == pytest.approx(123.0)
    assert url_start_time("https://twitch.tv/videos/1?t=1h2m") == pytest.approx(3720.0)
    start_only = resolve_run_window("https://youtu.be/x?t=30", 100, 150)
    assert not start_only["resolved"]
    assert start_only["start_s"] == 30 and start_only["end_s"] is None
    unresolved = resolve_run_window("https://youtu.be/x", 100, 300)
    assert not unresolved["resolved"] and unresolved["start_s"] is None
    matched = resolve_run_window("https://youtu.be/x", 100, 108)
    assert matched["resolved"] and matched["start_s"] == 0
    assert matched["end_s"] == 108


def test_loadless_nominal_duration_never_infers_wall_clock_end() -> None:
    # Measured ofy shape: URL identifies a plausible start, but gameplay/HUD
    # continues well beyond start + speedrun.com's loadless run duration.
    window = resolve_run_window(
        "https://youtu.be/ofy37Fm6EgI?t=159",
        nominal_duration_s=15_942.753,
        media_duration_s=18_000.0,
    )
    assert window["start_resolved"] and window["start_s"] == 159
    assert not window["end_resolved"] and window["end_s"] is None
    reviewed = resolve_run_window(
        "https://youtu.be/ofy37Fm6EgI?t=159",
        nominal_duration_s=15_942.753,
        media_duration_s=18_000.0,
        explicit_end_s=17_950.0,
    )
    assert reviewed["resolved"] and reviewed["end_s"] == 17_950.0


def test_pts_summary_detects_gaps() -> None:
    pts = np.arange(120, dtype=float) / 60.0
    pts[80:] += 0.2
    report = summarize_pts(pts)
    assert report["effective_fps"] == pytest.approx(60.0)
    assert report["large_gap_intervals"] == 1


def test_pts_summary_uses_mean_cadence_for_millisecond_quantized_60hz() -> None:
    # Twitch VOD timestamps commonly quantize true 1/60-second cadence into
    # two 0.017-second intervals and one 0.016-second interval.  The median is
    # 0.017 (58.82 Hz), while the inlier mean correctly recovers 60 Hz.
    intervals = np.tile(np.asarray([0.016, 0.017, 0.017]), 400)
    pts = np.r_[0.0, np.cumsum(intervals)]
    report = summarize_pts(pts)
    assert report["median_dt_s"] == pytest.approx(0.017)
    assert report["mean_cadence_dt_s"] == pytest.approx(1 / 60)
    assert report["effective_fps"] == pytest.approx(60.0)
    assert report["span_fps"] == pytest.approx(60.0)
    assert report["large_gap_intervals"] == 0


def test_temporal_offset_semantics() -> None:
    observed = np.arange(6)[:, None] >= 3
    source = np.arange(10, 16)
    pts = np.arange(6) / 60
    aligned, aligned_source, _ = apply_temporal_offset(observed, source, pts, -2)
    # Overlay frame 2 labels gameplay frame 0 when the overlay is two frames late.
    assert aligned.shape[0] == 4
    assert aligned_source.tolist() == [10, 11, 12, 13]
    assert aligned[:, 0].tolist() == [False, True, True, True]


def test_activity_runs_do_not_bridge_long_idle_spans() -> None:
    keys = np.zeros((30, len(KEY_ORDER)), np.uint8)
    keys[5, 0] = 1
    keys[24, 1] = 1
    active = activity_mask(keys, radius_frames=2)
    assert contiguous_true_runs(active, min_frames=1) == [(3, 8), (22, 27)]


def test_decode_and_build_train_ready_wild_shards(tmp_path: Path) -> None:
    video_id = "wild_test"
    source_dir = tmp_path / video_id
    source_dir.mkdir()
    layout_raw = _layout_dict(video_id)
    draft_render_layout_path = source_dir / "layout.render.json"
    draft_render_layout_path.write_text(json.dumps(layout_raw))
    layout = WildLayout.load(draft_render_layout_path)
    video = source_dir / f"{video_id}.mp4"
    truth = _write_video(video, layout)

    media = probe_media(video, scan_pts=True)
    assert media["pts"]["frames"] == N
    fetch = {
        "format_version": "madeleine.wild-fetch.v1",
        "video_id": video_id,
        "source": "youtube",
        "origin_url": "https://youtu.be/wild_test",
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
    fetch_path = source_dir / "fetch.json"
    fetch_path.write_text(json.dumps(fetch))
    layout_path, layout_acceptance, _, _ = _layout_review_fixture(
        source_dir / "review-fixture",
        video_id=video_id,
        source_sha256=fetch["sha256"],
        layout_raw=layout_raw,
    )
    boundaries_path = source_dir / "boundaries.json"
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
    provisional_dir = source_dir / "decoded-provisional"
    provisional = decode_video(
        fetch_path,
        layout_path,
        boundaries_path,
        provisional_dir,
        layout_acceptance_path=layout_acceptance,
    )
    assert not provisional["admitted"]
    assert set(provisional["rejection_reasons"]) == {
        "HUD compositor offset is unmeasured",
        "HUD compositor offset confidence below admission threshold",
    }
    with pytest.raises(ValueError, match="not admitted"):
        build_wild_video(
            provisional_dir / "decode_report.json",
            layout_path,
            source_dir / "shards-provisional-without-opt-in",
            frame_size=64,
        )
    provisional_jobs_root = source_dir / "aggregate-jobs"
    provisional_shards = provisional_jobs_root / video_id / "parts"
    provisional_build = build_wild_video(
        provisional_dir / "decode_report.json",
        layout_path,
        provisional_shards,
        frame_size=64,
        provisional=True,
    )
    assert provisional_build["admission_tier"] == "provisional_not_train_ready"
    assert provisional_build["train_ready_frames"] == 0
    assert provisional_build["train_ready_hours"] == 0
    assert provisional_build["provisional_trainable_frames"] == N - int(FPS)
    assert provisional_build["provisional_trainable_hours"] > 0
    assert provisional_build["implementation"]["sha256"] == sha256_file(
        Path(build_wild_video.__code__.co_filename)
    )
    assert (provisional_shards / "wild_provisional_build_report.json").is_file()
    provisional_manifest = update_provisional_corpus_manifest(
        source_dir,
        provisional_build,
    )
    manifest_value = json.loads(provisional_manifest.read_text())
    assert manifest_value["train_ready_hours"] == 0
    assert manifest_value["provisional_trainable_hours"] > 0
    with pytest.raises(ValueError, match="admitted build reports"):
        update_corpus_manifest(source_dir, provisional_build)

    # Concurrent builders write isolated video directories. The finalizer
    # consumes an explicit set, rehashes every bound input and part, validates
    # the array contract, and atomically publishes a non-train-ready aggregate.
    aggregate_decode_root = source_dir / "aggregate-decodes"
    aggregate_decode_dir = aggregate_decode_root / video_id
    aggregate_decode_dir.mkdir(parents=True)
    aggregate_decode_report = aggregate_decode_dir / "decode_report.json"
    aggregate_labels = aggregate_decode_dir / provisional["labels"]
    aggregate_decode_report.write_bytes(
        (provisional_dir / "decode_report.json").read_bytes()
    )
    aggregate_labels.write_bytes((provisional_dir / provisional["labels"]).read_bytes())
    aggregate_boundary_store = source_dir / "aggregate-boundaries"
    aggregate_boundary_store.mkdir()
    boundary_sha256 = provisional["boundaries"]["sha256"]
    (aggregate_boundary_store / f"{boundary_sha256}.json").write_bytes(
        boundaries_path.read_bytes()
    )
    aggregate_path = source_dir / "wild_provisional_aggregate.json"
    aggregate = finalize_provisional_corpus(
        jobs_root=provisional_jobs_root,
        decode_root=aggregate_decode_root,
        layout_root=layout_path.parent,
        boundary_store=aggregate_boundary_store,
        builder_file=Path(build_wild_video.__code__.co_filename),
        video_ids=[video_id],
        output=aggregate_path,
        expected_frame_size=64,
    )
    assert aggregate["video_count"] == 1
    assert aggregate["session_count"] == len(provisional_build["parts"])
    assert aggregate["train_ready_hours"] == 0
    assert aggregate["provisional_trainable_frames"] == (
        provisional_build["provisional_trainable_frames"]
    )
    assert aggregate["verification"]["expected_frame_shape"] == [64, 64, 3]
    with pytest.raises(FileExistsError):
        finalize_provisional_corpus(
            jobs_root=provisional_jobs_root,
            decode_root=aggregate_decode_root,
            layout_root=layout_path.parent,
            boundary_store=aggregate_boundary_store,
            builder_file=Path(build_wild_video.__code__.co_filename),
            video_ids=[video_id],
            output=aggregate_path,
            expected_frame_size=64,
        )
    with pytest.raises(ValueError, match="video ID must be a relative basename"):
        finalize_provisional_corpus(
            jobs_root=provisional_jobs_root,
            decode_root=aggregate_decode_root,
            layout_root=layout_path.parent,
            boundary_store=aggregate_boundary_store,
            builder_file=Path(build_wild_video.__code__.co_filename),
            video_ids=["../escape"],
            output=source_dir / "escape.json",
            expected_frame_size=64,
        )
    tampered_report_path = provisional_shards / "wild_provisional_build_report.json"
    tampered_report = json.loads(tampered_report_path.read_text())
    tampered_report["parts"][0]["sha256"] = "0" * 64
    tampered_report_path.write_text(json.dumps(tampered_report))
    with pytest.raises(ValueError, match="part SHA-256 mismatch"):
        finalize_provisional_corpus(
            jobs_root=provisional_jobs_root,
            decode_root=aggregate_decode_root,
            layout_root=layout_path.parent,
            boundary_store=aggregate_boundary_store,
            builder_file=Path(build_wild_video.__code__.co_filename),
            video_ids=[video_id],
            output=source_dir / "tampered.json",
            expected_frame_size=64,
        )

    rejected_report = json.loads(
        (provisional_dir / "decode_report.json").read_text()
    )
    rejected_report["rejection_reasons"].append("cell fixture is weak")
    rejected_path = provisional_dir / "decode_report.rejected.json"
    rejected_path.write_text(json.dumps(rejected_report))
    with pytest.raises(ValueError, match="non-provenance QC"):
        build_wild_video(
            rejected_path,
            layout_path,
            source_dir / "shards-rejected",
            frame_size=64,
            provisional=True,
        )

    low_confidence_report = json.loads(
        (provisional_dir / "decode_report.json").read_text()
    )
    low_confidence_report["layout"]["inference_confidence"] = 0.74
    low_confidence_path = provisional_dir / "decode_report.low-confidence.json"
    low_confidence_path.write_text(json.dumps(low_confidence_report))
    with pytest.raises(ValueError, match="layout confidence"):
        build_wild_video(
            low_confidence_path,
            layout_path,
            source_dir / "shards-low-confidence",
            frame_size=64,
            provisional=True,
        )

    bare_layout_raw = json.loads(layout_path.read_text())
    bare_layout_raw.pop("layout_review_acceptance")
    bare_layout_path = source_dir / "layout.bare-human-boolean.json"
    bare_layout_path.write_text(json.dumps(bare_layout_raw, indent=2) + "\n")
    bare_report = decode_video(
        fetch_path,
        bare_layout_path,
        boundaries_path,
        source_dir / "decoded-bare-layout",
    )
    assert not bare_report["admitted"]
    assert "layout lacks a verified hash-bound review acceptance" in (
        bare_report["rejection_reasons"]
    )

    ai_boundaries_path = source_dir / "boundaries.ai.json"
    ai_boundaries = json.loads(boundaries_path.read_text())
    ai_boundaries.update({
        "human_reviewed": False,
        "reviewer": "Synthetic AI Reviewer",
        "reviewer_kind": "ai_agent",
    })
    ai_boundaries_path.write_text(json.dumps(ai_boundaries))
    ai_boundary_report = decode_video(
        fetch_path,
        layout_path,
        ai_boundaries_path,
        source_dir / "decoded-ai-boundaries",
        layout_acceptance_path=layout_acceptance,
    )
    assert not ai_boundary_report["admitted"]
    assert "gameplay boundaries were not reviewed by a human" in (
        ai_boundary_report["rejection_reasons"]
    )

    calibration, _ = _write_calibration(
        source_dir / "calibration",
        layout_path,
        winner=0,
        source_sha256=fetch["sha256"],
    )
    final_layout_path = source_dir / "layout.final.json"
    acceptance_path = calibration.parent / "offset_acceptance.json"
    accept_offset(
        calibration,
        layout_path,
        layout_acceptance,
        final_layout_path,
        acceptance_path,
        reviewer_identity="synthetic-test-reviewer",
        reviewer_kind="human",
        approved=True,
    )
    decoded_dir = source_dir / "decoded-final"
    report = decode_video(
        fetch_path,
        final_layout_path,
        boundaries_path,
        decoded_dir,
        layout_acceptance_path=layout_acceptance,
        offset_acceptance_path=acceptance_path,
    )
    assert report["admitted"], report["rejection_reasons"]
    labels = pq.read_table(decoded_dir / "labels_native.parquet").to_pydict()
    predicted = np.stack([np.asarray(labels[key]) for key in KEY_ORDER], axis=1)
    assert predicted.shape == truth.shape
    assert np.mean(predicted == truth) > 0.995

    shards = source_dir / "shards"
    build = build_wild_video(
        decoded_dir / "decode_report.json", final_layout_path, shards,
        frame_size=64, idle_context_s=3.0, workers=2,
    )
    assert build["train_ready_frames"] == N - int(FPS)
    assert build["excluded_by_reviewed_ranges"] == int(FPS)
    assert len(build["parts"]) == 2
    frames_parts, key_parts, index_parts = [], [], []
    for row in build["parts"]:
        assert (shards / f"{row['npz']}.complete.json").is_file()
        with np.load(shards / row["npz"]) as part:
            frames_parts.append(part["frames"])
            key_parts.append(part["keys"])
            index_parts.append(part["engine_frame_idx"])
    frames = np.concatenate(frames_parts)
    keys = np.concatenate(key_parts)
    indices = np.concatenate(index_parts)
    assert not np.any((indices >= 4 * FPS) & (indices < 5 * FPS))
    assert np.mean(keys == truth[indices]) > 0.995
    crop_masks = mask_rects_in_gameplay(layout)
    assert len(crop_masks) == 1
    x0, y0, x1, y1 = rect_to_pixels(crop_masks[0], 64, 64)
    x0, y0 = max(0, x0 - 1), max(0, y0 - 1)
    x1, y1 = min(64, x1 + 1), min(64, y1 + 1)
    assert frames[:, y0:y1, x0:x1].max() == 0

    resumed = build_wild_video(
        decoded_dir / "decode_report.json", final_layout_path, shards,
        frame_size=64, idle_context_s=3.0, workers=2, resume=True,
    )
    assert resumed["resumed_parts"] == len(build["parts"])
    assert [row["sha256"] for row in resumed["parts"]] == [
        row["sha256"] for row in build["parts"]
    ]

    # A completion marker whose exact implementation/source bindings change is
    # never trusted. Only that part is rebuilt; the other part still resumes.
    first_marker = shards / f"{build['parts'][0]['npz']}.complete.json"
    tampered = json.loads(first_marker.read_text())
    tampered["bindings"]["source_video_sha256"] = "0" * 64
    first_marker.write_text(json.dumps(tampered))
    after_tamper = build_wild_video(
        decoded_dir / "decode_report.json", final_layout_path, shards,
        frame_size=64, idle_context_s=3.0, workers=2, resume=True,
    )
    assert after_tamper["resumed_parts"] == len(build["parts"]) - 1
