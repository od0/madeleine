from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments.summarize_vpt_small_wild_eval import (
    summarize,
    video_id_from_stream_id,
)
from experiments.score_vpt_small_wild_holdout import score
from experiments.normalize_vpt_phase0_sidecar import normalize


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_video_id_from_stream_id() -> None:
    assert (
        video_id_from_stream_id("wild_abc-123__r004__run000__sub000")
        == "abc-123"
    )


def test_summary_groups_streams_and_equal_weights_videos(tmp_path: Path) -> None:
    sidecar = tmp_path / "preds.npz"
    truth = np.zeros((6, 7), dtype=np.uint8)
    truth[[0, 2, 3, 5], :] = 1
    probability = np.where(truth, 0.9, 0.1).astype(np.float32)
    np.savez_compressed(
        sidecar,
        y_true=truth,
        y_prob=probability,
        input_active=np.ones(6, dtype=np.uint8),
        session_lengths=np.asarray([2, 1, 3], dtype=np.int64),
        session_ids=np.asarray(
            [
                "wild_A__r000__run000__sub000",
                "wild_A__r001__run000__sub000",
                "wild_B__r000__run000__sub000",
            ]
        ),
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "sidecar": {"sha256": _sha256(sidecar)},
                "aggregate": {"macro_ap": 1.0},
            }
        ),
        encoding="utf-8",
    )
    result = summarize(report, sidecar)
    assert result["videos"] == 2
    assert result["per_video"]["A"]["streams"] == 2
    assert result["per_video"]["B"]["streams"] == 1
    assert result["equal_video"]["macro_ap"] == pytest.approx(1.0)


def test_summary_rejects_unbound_sidecar(tmp_path: Path) -> None:
    sidecar = tmp_path / "preds.npz"
    np.savez_compressed(sidecar, y_true=np.zeros((1, 7), dtype=np.uint8))
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"sidecar": {"sha256": "0" * 64}, "aggregate": {}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not bind"):
        summarize(report, sidecar)


def test_holdout_scorer_binds_repeated_inference_and_video_membership(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    sidecar = tmp_path / "preds-a.npz"
    truth = np.zeros((6, 7), dtype=np.uint8)
    truth[[0, 2, 3, 5], :] = 1
    probability = np.where(truth, 0.9, 0.1).astype(np.float32)
    np.savez_compressed(
        sidecar,
        y_true=truth,
        y_prob=probability,
        input_active=np.ones(6, dtype=np.uint8),
        session_lengths=np.asarray([3, 3], dtype=np.int64),
        session_ids=np.asarray(
            ["wild_A__r000__run000__sub000", "wild_B__r000__run000__sub000"]
        ),
        source_row_index=np.arange(6, dtype=np.int64),
        source_engine_frame_idx=np.arange(6, dtype=np.int64) * 3,
    )
    repeat = tmp_path / "preds-b.npz"
    shutil.copyfile(sidecar, repeat)
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "checkpoint": {"sha256": _sha256(checkpoint), "epoch": 20},
                "evaluation_population": {
                    "build_manifest_sha256": _sha256(manifest),
                    "expected_center_supported_rows": 6,
                    "videos": ["A", "B"],
                },
                "protocol": {"support_mode": "deployment-20hz-phase0"},
            }
        ),
        encoding="utf-8",
    )
    result = score(
        contract_path=contract,
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        sidecar_path=sidecar,
        repeat_sidecar_path=repeat,
    )
    assert result["prediction_sidecar"]["byte_identical_repeat"] is True
    assert result["row_weighted"]["macro_ap"] == pytest.approx(1.0)
    assert result["equal_video"]["macro_ap"] == pytest.approx(1.0)


def test_phase0_sidecar_normalization_merges_three_step_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    truth = np.zeros((5, 7), dtype=np.uint8)
    np.savez_compressed(
        source,
        y_true=truth,
        y_prob=np.zeros((5, 7), dtype=np.float32),
        input_active=np.ones(5, dtype=np.uint8),
        session_lengths=np.ones(5, dtype=np.int64),
        session_ids=np.asarray(
            [
                "wild_A__r000__run000__sub000",
                "wild_A__r000__run000__sub001",
                "wild_A__r000__run000__sub002",
                "wild_B__r000__run000__sub000",
                "wild_B__r000__run000__sub001",
            ]
        ),
        source_row_index=np.asarray([0, 3, 9, 0, 3], dtype=np.int64),
        source_engine_frame_idx=np.asarray([0, 3, 9, 0, 3], dtype=np.int64),
    )
    output = tmp_path / "normalized.npz"
    receipt = normalize(source, output)
    with np.load(output, allow_pickle=False) as archive:
        assert archive["session_lengths"].tolist() == [2, 1, 2]
        assert archive["session_ids"].astype(str).tolist() == [
            "wild_A__r000__run000__sub000",
            "wild_A__r000__run000__sub001",
            "wild_B__r000__run000__sub000",
        ]
    assert receipt["truth_probability_rows_changed"] == 0
