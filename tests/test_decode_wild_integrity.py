from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import harvest.decode_wild as decode_module
from data.schema import KEY_ORDER
from harvest.decode_wild import (
    DECODE_COMPLETION_NAME,
    DECODE_VERSION,
    WILD_LABEL_SCHEMA,
    _load_precomputed_scan_scores,
    decode_video,
    validate_decode_output,
)
from harvest.fetch_wild import PTS_SIDECAR_VERSION, sha256_file
from harvest.wild_boundaries import BOUNDARIES_VERSION
from harvest.wild_layout import CellSpec, WildLayout, rect_to_pixels


def _layout(video_id: str, *, decoder: str = "luma") -> WildLayout:
    cells = []
    for index, action in enumerate(KEY_ORDER):
        sample = (0.01 + 0.1 * index, 0.8, 0.04, 0.04)
        cells.append(CellSpec(
            cell_id=f"cell_{action}",
            action=action,
            sample_rect=sample,
            decoder=decoder,  # type: ignore[arg-type]
            pressed_polarity="high",
            reference_rect=(sample[0], 0.86, 0.04, 0.04)
            if decoder == "local_contrast"
            else None,
        ))
    return WildLayout(
        video_id=video_id,
        overlay_style="fixture",
        gameplay_rect=(0.0, 0.0, 1.0, 1.0),
        gameplay_rect_source="fixture",
        gameplay_rect_confidence=0.9,
        mask_rects=((0.0, 0.0, 1.0, 1.0),),
        cells=tuple(cells),
        inference_source="fixture",
        inference_confidence=0.9,
        human_reviewed=False,
        evidence_frames=(0.0,),
        temporal_offset_frames=0,
        temporal_offset_source="unmeasured",
        temporal_offset_confidence=0.0,
    )


def _scan_fixture(
    directory: Path,
    layout: WildLayout,
    *,
    source_sha256: str,
    pts_sha256: str,
    decoder: str | None = None,
) -> Path:
    frame_count = 4
    resolution = [100, 100]
    scores = np.arange(frame_count * len(layout.cells), dtype=np.float32)
    score_path = directory / "cell_scores.f32"
    scores.tofile(score_path)
    spec_cells = []
    for cell in layout.cells:
        x0, y0, x1, y1 = rect_to_pixels(cell.sample_rect, *resolution)
        row = {
            "cell_id": cell.cell_id,
            "sample_rect_px": [x0, y0, x1 - x0, y1 - y0],
            "pressed_polarity": cell.pressed_polarity,
            "semantic_action_from_reference": cell.action,
        }
        if decoder is not None:
            row["decoder"] = decoder
        spec_cells.append(row)
    spec = {
        "video_id": layout.video_id,
        "source_sha256": source_sha256,
        "pts_sha256": pts_sha256,
        "frame_size_wh": resolution,
        "cells": spec_cells,
    }
    spec_path = directory / "spec.json"
    spec_path.write_text(json.dumps(spec))
    report = {
        "format_version": "madeleine.wild-cell-activity-scan.v1",
        "video_id": layout.video_id,
        "source": {"sha256": source_sha256, "frames": frame_count},
        "pts": {"sha256": pts_sha256},
        "spec": {"path": spec_path.name, "sha256": sha256_file(spec_path)},
        "scores": {
            "path": score_path.name,
            "sha256": sha256_file(score_path),
            "dtype": "float32",
            "shape": [frame_count, len(layout.cells)],
        },
        "human_reviewed": False,
        "training_admitted": False,
    }
    report_path = directory / "cell_activity_scan.json"
    report_path.write_text(json.dumps(report))
    return report_path


def test_precomputed_scan_fails_closed_for_non_luma_layout(tmp_path: Path) -> None:
    layout = _layout("fixture", decoder="local_contrast")
    with pytest.raises(ValueError, match="only plain luma"):
        _load_precomputed_scan_scores(
            tmp_path / "missing.json",
            layout,
            {"sha256": "a" * 64},
            {"sha256": "b" * 64},
            4,
        )


def test_precomputed_scan_checks_declared_decoder_when_present(
    tmp_path: Path,
) -> None:
    layout = _layout("fixture")
    source_hash, pts_hash = "a" * 64, "b" * 64
    report_path = _scan_fixture(
        tmp_path,
        layout,
        source_sha256=source_hash,
        pts_sha256=pts_hash,
        decoder="local_contrast",
    )
    with pytest.raises(ValueError, match="differs from target layout"):
        _load_precomputed_scan_scores(
            report_path,
            layout,
            {"sha256": source_hash, "media": {"resolution_wh": [100, 100]}},
            {"sha256": pts_hash},
            4,
        )


def test_precomputed_scan_accepts_legacy_spec_without_decoder(tmp_path: Path) -> None:
    layout = _layout("fixture")
    source_hash, pts_hash = "a" * 64, "b" * 64
    report_path = _scan_fixture(
        tmp_path,
        layout,
        source_sha256=source_hash,
        pts_sha256=pts_hash,
    )
    scores, provenance = _load_precomputed_scan_scores(
        report_path,
        layout,
        {"sha256": source_hash, "media": {"resolution_wh": [100, 100]}},
        {"sha256": pts_hash},
        4,
    )
    assert scores.shape == (4, len(KEY_ORDER))
    assert provenance["kind"] == "hash_bound_full_cell_scan"


def _decode_chain(root: Path) -> tuple[Path, Path, Path, Path]:
    video_id = "fixture"
    source_dir = root / "source"
    source_dir.mkdir()
    video = source_dir / "fixture.mp4"
    video.write_bytes(b"immutable source fixture")
    source_hash = sha256_file(video)
    pts = 10.0 + np.arange(8, dtype=np.float64) / 60.0
    pts_path = source_dir / "frame_pts.npy"
    np.save(pts_path, pts, allow_pickle=False)
    pts_manifest = {
        "format_version": PTS_SIDECAR_VERSION,
        "source_file": video.name,
        "source_sha256": source_hash,
        "path": pts_path.name,
        "sha256": sha256_file(pts_path),
        "frames": int(pts.size),
    }
    (source_dir / "frame_pts.json").write_text(json.dumps(pts_manifest))
    fetch = {
        "video_id": video_id,
        "source_file": video.name,
        "sha256": source_hash,
        "media": {"pts": {"frames": int(pts.size)}},
    }
    fetch_path = source_dir / "fetch.json"
    fetch_path.write_text(json.dumps(fetch))

    layout_path = root / "layout.json"
    layout_path.write_text(json.dumps(_layout(video_id).to_dict()))
    boundaries = {
        "format_version": BOUNDARIES_VERSION,
        "video_id": video_id,
        "source_sha256": source_hash,
        "wall_clock_range_s": [0.0, 8.0 / 60.0],
        "excluded_ranges_s": [],
        "human_reviewed": False,
        "reviewer": "fixture AI",
        "reviewer_kind": "ai_agent",
        "evidence": ["fixture"],
    }
    boundaries_path = root / "boundaries.json"
    boundaries_path.write_text(json.dumps(boundaries))

    decoded = root / "decoded"
    decoded.mkdir()
    indices = np.arange(8, dtype=np.int64)
    allowed = np.ones(8, dtype=bool)
    columns: dict[str, object] = {
        "video_frame_idx": indices,
        "pts_s": pts - pts[0],
        **{key: (indices + index) % 2 == 0 for index, key in enumerate(KEY_ORDER)},
        "gameplay_allowed": allowed,
    }
    table = pa.Table.from_pydict(columns, schema=WILD_LABEL_SCHEMA)
    raw_path = decoded / "labels_raw.parquet"
    labels_path = decoded / "labels_native.parquet"
    pq.write_table(table, raw_path)
    pq.write_table(table, labels_path)
    report = {
        "format_version": DECODE_VERSION,
        "video_id": video_id,
        "source_video": {
            "sha256": source_hash,
            "source_frame_range": [0, 8],
        },
        "layout": {"sha256": sha256_file(layout_path)},
        "boundaries": {"sha256": sha256_file(boundaries_path)},
        "timing": {
            "pts_evidence": {
                "sha256": pts_manifest["sha256"],
                "frames": int(pts.size),
            }
        },
        "score_source": {"kind": "decoded_from_source_video"},
        "decoded_frames": 8,
        "gameplay_allowed_frames": 8,
        "raw_labels": raw_path.name,
        "raw_labels_sha256": sha256_file(raw_path),
        "labels": labels_path.name,
        "labels_sha256": sha256_file(labels_path),
        "admitted": False,
    }
    (decoded / "decode_report.json").write_text(json.dumps(report))
    return decoded, fetch_path, layout_path, boundaries_path


def test_completion_backfill_requires_and_binds_full_existing_chain(
    tmp_path: Path,
) -> None:
    decoded, fetch, layout, boundaries = _decode_chain(tmp_path)
    completion = validate_decode_output(
        decoded, fetch, layout, boundaries, backfill_completion=True
    )
    assert (decoded / DECODE_COMPLETION_NAME).is_file()
    assert completion["bindings"]["fetch_report_sha256"] == sha256_file(fetch)
    assert completion["bindings"]["pts_manifest_sha256"] == sha256_file(
        fetch.parent / "frame_pts.json"
    )
    assert completion["bindings"]["pts_sha256"] == sha256_file(
        fetch.parent / "frame_pts.npy"
    )
    assert validate_decode_output(decoded, fetch, layout, boundaries) == completion
    assert not list(decoded.glob(".decode_complete.json.*"))


def test_invalid_existing_chain_is_never_backfilled(tmp_path: Path) -> None:
    decoded, fetch, layout, boundaries = _decode_chain(tmp_path)
    labels = decoded / "labels_native.parquet"
    labels.write_bytes(labels.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="labels bytes differ"):
        validate_decode_output(
            decoded, fetch, layout, boundaries, backfill_completion=True
        )
    assert not (decoded / DECODE_COMPLETION_NAME).exists()


def test_decode_publishes_report_then_hash_bound_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_id = "atomic_fixture"
    source = tmp_path / "source"
    source.mkdir()
    layout = _layout(video_id)
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(layout.to_dict()))
    video = source / "atomic_fixture.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 60.0, (100, 100)
    )
    assert writer.isOpened()
    for frame_index in range(120):
        frame = np.full((100, 100, 3), 20, dtype=np.uint8)
        for cell_index, cell in enumerate(layout.cells):
            if (frame_index // 10 + cell_index) % 2:
                x0, y0, x1, y1 = rect_to_pixels(cell.sample_rect, 100, 100)
                frame[y0:y1, x0:x1] = 240
        writer.write(frame)
    writer.release()
    source_hash = sha256_file(video)
    pts = np.arange(120, dtype=np.float64) / 60.0
    pts_path = source / "frame_pts.npy"
    np.save(pts_path, pts, allow_pickle=False)
    pts_manifest = {
        "format_version": PTS_SIDECAR_VERSION,
        "source_file": video.name,
        "source_sha256": source_hash,
        "path": pts_path.name,
        "sha256": sha256_file(pts_path),
        "frames": int(pts.size),
    }
    (source / "frame_pts.json").write_text(json.dumps(pts_manifest))
    fetch_path = source / "fetch.json"
    fetch_path.write_text(json.dumps({
        "video_id": video_id,
        "source_file": video.name,
        "sha256": source_hash,
        "media": {
            "resolution_wh": [100, 100],
            "pts": {"frames": int(pts.size)},
        },
    }))
    boundaries_path = tmp_path / "boundaries.json"
    boundaries_path.write_text(json.dumps({
        "format_version": BOUNDARIES_VERSION,
        "video_id": video_id,
        "source_sha256": source_hash,
        "wall_clock_range_s": [0.0, 2.0],
        "excluded_ranges_s": [],
        "human_reviewed": False,
        "reviewer": "fixture AI",
        "reviewer_kind": "ai_agent",
        "evidence": ["fixture"],
    }))
    decoded = tmp_path / "decoded"
    publication_order: list[str] = []
    atomic_json = decode_module._atomic_json

    def observe_publication(
        path: Path, value: dict[str, object], *, replace: bool
    ) -> None:
        publication_order.append(path.name)
        if path.name == "decode_report.json":
            assert pq.read_table(decoded / "labels_raw.parquet").num_rows == 120
            assert pq.read_table(decoded / "labels_native.parquet").num_rows == 120
            assert not (decoded / DECODE_COMPLETION_NAME).exists()
        elif path.name == DECODE_COMPLETION_NAME:
            assert (decoded / "decode_report.json").is_file()
        atomic_json(path, value, replace=replace)  # type: ignore[arg-type]

    monkeypatch.setattr(decode_module, "_atomic_json", observe_publication)
    decode_video(fetch_path, layout_path, boundaries_path, decoded)
    assert publication_order == ["decode_report.json", DECODE_COMPLETION_NAME]
    validate_decode_output(decoded, fetch_path, layout_path, boundaries_path)
    assert not list(decoded.glob(".*.tmp"))
