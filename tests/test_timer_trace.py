from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from harvest.extract_timer_trace import (
    MANIFEST_FILE,
    PROPOSAL_FILE,
    TRACE_FILE,
    TRACE_VERSION,
    FFMPEG_BACKEND_VERSION,
    _read_exact,
    extract_timer_trace,
)
from harvest.fetch_wild import (
    PTS_SIDECAR_VERSION,
    frame_pts,
    sha256_file,
    summarize_pts,
)
from harvest.timer_activity import timer_change_scores, timer_presence_scores
from harvest.wild_layout import rect_to_pixels


W, H, FPS, N = 96, 64, 60.0, 360
TIMER_ROI = (0.5, 0.0, 0.5, 0.5)
ODD_TIMER_ROI = (20 / W, 10 / H, 25 / W, 17 / H)
EVIDENCE = {
    "timer_roi": "review/timer-roi-contact-sheet.png",
    "wall_clock_bounds": "review/run-envelope.json",
}
HUMAN_REVIEW = {
    "reviewer_identity": "Synthetic Human Reviewer",
    "reviewer_kind": "human",
}


def _write_timer_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )
    assert writer.isOpened()
    x0, y0, x1, y1 = rect_to_pixels(TIMER_ROI, W, H)
    for index in range(N):
        frame = np.full((H, W, 3), 80 + index % 40, dtype=np.uint8)
        frame[y0:y1, x0:x1] = 0
        phase = (index // 2) % 12
        first_x = x0 + 2 + phase * 3
        second_x = x0 + 2 + ((phase + 5) % 12) * 3
        frame[y0 + 4:y0 + 12, first_x:first_x + 8] = 255
        frame[y0 + 18:y0 + 26, second_x:second_x + 8] = 255
        writer.write(frame)
    writer.release()


@pytest.fixture
def fetched_video(tmp_path: Path) -> dict[str, object]:
    video_id = "timer_trace_fixture"
    source = tmp_path / f"{video_id}.mp4"
    _write_timer_video(source)
    source_hash = sha256_file(source)
    pts = frame_pts(source)
    assert pts.size == N
    vector = tmp_path / "frame_pts.npy"
    np.save(vector, pts, allow_pickle=False)
    vector_hash = sha256_file(vector)
    pts_manifest = {
        "format_version": PTS_SIDECAR_VERSION,
        "source_file": source.name,
        "source_sha256": source_hash,
        "path": vector.name,
        "sha256": vector_hash,
        "frames": int(pts.size),
        "summary": summarize_pts(pts),
    }
    manifest_path = tmp_path / "frame_pts.json"
    manifest_path.write_text(json.dumps(pts_manifest, indent=2) + "\n")
    fetch = {
        "format_version": "madeleine.wild-fetch.v2",
        "video_id": video_id,
        "source_file": source.name,
        "sha256": source_hash,
        "candidate": {"duration_s": N / FPS},
        "pts_sidecar": {
            "path": vector.name,
            "manifest": manifest_path.name,
            "sha256": vector_hash,
            "frames": int(pts.size),
        },
        "media": {"pts": {"frames": int(pts.size)}},
    }
    fetch_path = tmp_path / "fetch.json"
    fetch_path.write_text(json.dumps(fetch, indent=2) + "\n")
    relative = pts - pts[0]
    bounds = (float(relative[60]), float(relative[300]))
    return {
        "source": source,
        "pts": pts,
        "vector": vector,
        "pts_manifest": manifest_path,
        "fetch": fetch,
        "fetch_path": fetch_path,
        "bounds": bounds,
    }


def _decode_reviewed_rois(
    video: Path, start_index: int, end_index: int
) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened()
    assert capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_index))
    crops = []
    try:
        for _ in range(start_index, end_index):
            ok, frame = capture.read()
            assert ok
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x0, y0, x1, y1 = rect_to_pixels(TIMER_ROI, W, H)
            crops.append(gray[y0:y1, x0:x1].copy())
    finally:
        capture.release()
    return np.stack(crops)


def _rewrite_pts_binding(
    fixture: dict[str, object], pts: np.ndarray
) -> None:
    vector = fixture["vector"]
    assert isinstance(vector, Path)
    np.save(vector, pts, allow_pickle=False)
    vector_hash = sha256_file(vector)
    manifest_path = fixture["pts_manifest"]
    assert isinstance(manifest_path, Path)
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "sha256": vector_hash,
        "frames": int(pts.size),
        "summary": summarize_pts(pts),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    fetch_path = fixture["fetch_path"]
    assert isinstance(fetch_path, Path)
    fetch = json.loads(fetch_path.read_text())
    fetch["pts_sidecar"].update({
        "sha256": vector_hash,
        "frames": int(pts.size),
    })
    fetch["media"]["pts"]["frames"] = int(pts.size)
    fetch_path.write_text(json.dumps(fetch, indent=2) + "\n")


def test_streamed_trace_matches_in_memory_scalars_and_binds_every_hash(
    fetched_video: dict[str, object], tmp_path: Path
) -> None:
    fetch_path = fetched_video["fetch_path"]
    source = fetched_video["source"]
    bounds = fetched_video["bounds"]
    assert isinstance(fetch_path, Path)
    assert isinstance(source, Path)
    assert isinstance(bounds, tuple)
    out = tmp_path / "trace"
    manifest = extract_timer_trace(
        fetch_path, TIMER_ROI, bounds, EVIDENCE, out, **HUMAN_REVIEW
    )

    with np.load(out / TRACE_FILE, allow_pickle=False) as trace:
        assert trace["format_version"].item() == TRACE_VERSION
        assert trace["source_frame_idx"].tolist() == list(range(60, 300))
        crops = _decode_reviewed_rois(source, 60, 300)
        expected_change = timer_change_scores(crops)
        expected_bright, expected_dark = timer_presence_scores(crops)
        np.testing.assert_allclose(trace["change_score"], expected_change, rtol=0, atol=0)
        np.testing.assert_allclose(
            trace["bright_mask_mean"], expected_bright, rtol=0, atol=0
        )
        np.testing.assert_allclose(
            trace["dark_mask_mean"], expected_dark, rtol=0, atol=0
        )

    persisted = json.loads((out / MANIFEST_FILE).read_text())
    assert persisted == manifest
    proposal = json.loads((out / PROPOSAL_FILE).read_text())
    assert manifest["source_video"]["sha256"] == sha256_file(source)
    assert manifest["source_video"]["hash_verified"] is True
    assert manifest["fetch_report"]["sha256"] == sha256_file(fetch_path)
    assert manifest["pts_evidence"]["vector_sha256"] == sha256_file(
        fetched_video["vector"]
    )
    assert manifest["trace"]["sha256"] == sha256_file(out / TRACE_FILE)
    assert manifest["proposal"]["sha256"] == sha256_file(out / PROPOSAL_FILE)
    assert proposal["trace_binding"]["trace_npz_sha256"] == manifest["trace"]["sha256"]
    assert proposal["trace_binding"]["source_sha256"] == manifest["source_video"]["sha256"]
    assert (
        proposal["trace_binding"]["pts_vector_sha256"]
        == manifest["pts_evidence"]["vector_sha256"]
    )
    assert proposal["auto_admitted"] is False
    assert proposal["proposal_quality"]["nominal_coverage_check"] == "passed"
    assert manifest["review"]["nominal_loadless_duration_s"] == N / FPS
    assert manifest["admission"]["wild_boundaries_created"] is False
    assert not (out / "boundaries.json").exists()

    second_out = tmp_path / "trace-repeat"
    second = extract_timer_trace(
        fetch_path, TIMER_ROI, bounds, EVIDENCE, second_out, **HUMAN_REVIEW
    )
    assert second["trace"]["sha256"] == manifest["trace"]["sha256"]
    assert (second_out / PROPOSAL_FILE).read_bytes() == (out / PROPOSAL_FILE).read_bytes()


class _ShortReadStream:
    def __init__(self, payload: bytes, chunk_size: int) -> None:
        self.payload = payload
        self.chunk_size = chunk_size
        self.offset = 0

    def readinto(self, destination: memoryview) -> int:
        count = min(self.chunk_size, len(destination), len(self.payload) - self.offset)
        if count <= 0:
            return 0
        destination[:count] = self.payload[self.offset:self.offset + count]
        self.offset += count
        return count


def test_raw_pipe_reader_accumulates_partial_reads() -> None:
    payload = bytes(range(37))
    target = bytearray(len(payload))
    received = _read_exact(_ShortReadStream(payload, 3), memoryview(target))
    assert received == len(payload)
    assert bytes(target) == payload


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for the fallback backend",
)
def test_ffmpeg_backend_preserves_odd_crop_and_exact_pipe_byte_count(
    fetched_video: dict[str, object], tmp_path: Path
) -> None:
    fetch_path = fetched_video["fetch_path"]
    bounds = fetched_video["bounds"]
    assert isinstance(fetch_path, Path)
    assert isinstance(bounds, tuple)
    out = tmp_path / "ffmpeg-trace"
    manifest = extract_timer_trace(
        fetch_path,
        ODD_TIMER_ROI,
        bounds,
        EVIDENCE,
        out,
        decode_backend="ffmpeg",
        **HUMAN_REVIEW,
    )

    selection = manifest["selection"]
    x0, y0, x1, y1 = rect_to_pixels(ODD_TIMER_ROI, W, H)
    assert (x1 - x0, y1 - y0) == (25, 17)
    assert selection["decode_backend"] == FFMPEG_BACKEND_VERSION
    assert selection["ffmpeg_crop_exact"] is True
    assert selection["timer_roi_pixels_xyxy"] == [x0, y0, x1, y1]
    assert selection["raw_frame_bytes"] == 25 * 17
    assert selection["decoded_frames"] == 240
    assert selection["raw_bytes_verified"] == 240 * 25 * 17
    assert selection["exact_row_count_verified"] is True
    assert selection["pipe_eof_verified"] is True
    assert "exact=1" in selection["ffmpeg_filter"]

    with np.load(out / TRACE_FILE, allow_pickle=False) as trace:
        assert trace["source_frame_idx"].tolist() == list(range(60, 300))
        assert trace["pts_s"].shape == (240,)
        assert trace["change_score"].shape == (240,)
    proposal = json.loads((out / PROPOSAL_FILE).read_text())
    assert proposal["trace_binding"]["trace_npz_sha256"] == sha256_file(
        out / TRACE_FILE
    )


def test_unknown_decode_backend_fails_before_output(
    fetched_video: dict[str, object], tmp_path: Path
) -> None:
    fetch_path = fetched_video["fetch_path"]
    bounds = fetched_video["bounds"]
    assert isinstance(fetch_path, Path)
    assert isinstance(bounds, tuple)
    out = tmp_path / "unknown-backend"
    with pytest.raises(ValueError, match="decode_backend"):
        extract_timer_trace(
            fetch_path,
            TIMER_ROI,
            bounds,
            EVIDENCE,
            out,
            decode_backend="mystery",
            **HUMAN_REVIEW,
        )
    assert not out.exists()


def test_ai_reviewer_trace_is_preserved_but_proposal_abstains(
    fetched_video: dict[str, object], tmp_path: Path
) -> None:
    fetch_path = fetched_video["fetch_path"]
    bounds = fetched_video["bounds"]
    assert isinstance(fetch_path, Path)
    assert isinstance(bounds, tuple)
    out = tmp_path / "ai-reviewed-trace"
    manifest = extract_timer_trace(
        fetch_path,
        TIMER_ROI,
        bounds,
        EVIDENCE,
        out,
        reviewer_identity="OpenAI Codex visual draft",
        reviewer_kind="ai_agent",
    )
    proposal = json.loads((out / PROPOSAL_FILE).read_text())
    assert (out / TRACE_FILE).is_file()
    assert proposal["signal_quality_gates_passed"] is True
    assert proposal["review_provenance_gate_passed"] is False
    assert proposal["automatic_gates_passed"] is False
    assert proposal["suggested_allowed_ranges_s"] == []
    assert proposal["activity"]["candidate_range_count_before_gates"] == 1
    assert manifest["review"]["reviewer_kind"] == "ai_agent"
    assert manifest["review"]["human_reviewed"] is False
    assert manifest["proposal"]["status"] == "abstained"


def test_source_and_pts_hashes_fail_closed_by_default(
    fetched_video: dict[str, object], tmp_path: Path
) -> None:
    fetch_path = fetched_video["fetch_path"]
    bounds = fetched_video["bounds"]
    vector = fetched_video["vector"]
    assert isinstance(fetch_path, Path)
    assert isinstance(bounds, tuple)
    assert isinstance(vector, Path)

    vector.write_bytes(vector.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="PTS sidecar hash mismatch"):
        extract_timer_trace(
            fetch_path, TIMER_ROI, bounds, EVIDENCE, tmp_path / "bad-pts",
            **HUMAN_REVIEW,
        )
    assert not (tmp_path / "bad-pts").exists()


def test_source_hash_mismatch_fails_before_trace_output(
    fetched_video: dict[str, object], tmp_path: Path
) -> None:
    fetch_path = fetched_video["fetch_path"]
    bounds = fetched_video["bounds"]
    assert isinstance(fetch_path, Path)
    assert isinstance(bounds, tuple)
    fetch = json.loads(fetch_path.read_text())
    fetch["sha256"] = "b" * 64
    fetch_path.write_text(json.dumps(fetch, indent=2) + "\n")
    with pytest.raises(ValueError, match="source video hash"):
        extract_timer_trace(
            fetch_path, TIMER_ROI, bounds, EVIDENCE, tmp_path / "bad-source",
            **HUMAN_REVIEW,
        )
    assert not (tmp_path / "bad-source").exists()


def test_semantically_gapped_but_rehashed_pts_fails_closed(
    fetched_video: dict[str, object], tmp_path: Path
) -> None:
    pts = np.asarray(fetched_video["pts"]).copy()
    pts[180:] += 0.25
    _rewrite_pts_binding(fetched_video, pts)
    fetch_path = fetched_video["fetch_path"]
    bounds = fetched_video["bounds"]
    assert isinstance(fetch_path, Path)
    assert isinstance(bounds, tuple)
    with pytest.raises(ValueError, match="contains large gaps"):
        extract_timer_trace(
            fetch_path, TIMER_ROI, bounds, EVIDENCE, tmp_path / "gapped",
            **HUMAN_REVIEW,
        )


def test_opencv_and_pts_frame_count_mismatch_fails_closed(
    fetched_video: dict[str, object], tmp_path: Path
) -> None:
    pts = np.asarray(fetched_video["pts"])[:-1].copy()
    _rewrite_pts_binding(fetched_video, pts)
    fetch_path = fetched_video["fetch_path"]
    bounds = fetched_video["bounds"]
    assert isinstance(fetch_path, Path)
    assert isinstance(bounds, tuple)
    with pytest.raises(ValueError, match="decode count mismatch"):
        extract_timer_trace(
            fetch_path, TIMER_ROI, bounds, EVIDENCE, tmp_path / "count-mismatch",
            **HUMAN_REVIEW,
        )


def test_quantized_twitch_pts_use_span_cadence_in_streaming_bridge(
    fetched_video: dict[str, object], tmp_path: Path
) -> None:
    intervals = np.resize(np.asarray([0.017, 0.017, 0.016]), N - 1)
    pts = np.r_[0.0, np.cumsum(intervals)]
    _rewrite_pts_binding(fetched_video, pts)
    fetch_path = fetched_video["fetch_path"]
    assert isinstance(fetch_path, Path)
    bounds = (float(pts[60]), float(pts[300]))
    manifest = extract_timer_trace(
        fetch_path,
        TIMER_ROI,
        bounds,
        EVIDENCE,
        tmp_path / "quantized-pts",
        **HUMAN_REVIEW,
    )
    cadence = manifest["selection"]["pts_summary"]
    assert cadence["span_effective_fps"] == pytest.approx(60.0, abs=0.02)
    assert cadence["median_interval_fps"] == pytest.approx(1 / 0.017)
    assert cadence["vfr_ratio_p99_p01"] > 1.05
    assert cadence["quantization_adjusted_vfr_ratio_p99_p01"] == pytest.approx(1.0)
    assert manifest["proposal"]["status"] == "review_required"


@pytest.mark.parametrize(
    ("roi", "evidence", "message"),
    [
        ((0.9, 0.0, 0.2, 0.2), EVIDENCE, "timer ROI"),
        (TIMER_ROI, {"timer_roi": "review.png"}, "wall_clock_bounds"),
    ],
)
def test_invalid_review_inputs_fail_before_decode(
    fetched_video: dict[str, object],
    tmp_path: Path,
    roi: tuple[float, float, float, float],
    evidence: dict[str, str],
    message: str,
) -> None:
    fetch_path = fetched_video["fetch_path"]
    bounds = fetched_video["bounds"]
    assert isinstance(fetch_path, Path)
    assert isinstance(bounds, tuple)
    with pytest.raises(ValueError, match=message):
        extract_timer_trace(
            fetch_path, roi, bounds, evidence, tmp_path / "invalid-review",
            **HUMAN_REVIEW,
        )
