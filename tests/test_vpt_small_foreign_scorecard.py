import json

import numpy as np
import pytest

from experiments.summarize_vpt_small_foreign_scorecard import build_scorecard


NAMES = ("tier_b_13p45h_epoch20", "unflagged92_103p4056h_epoch20", "unflagged92_down_ridge5pct_ft5e_epoch5")


def _sidecar(path, *, rows=(4, 5, 6, 7)):
    truth = np.zeros((4, 7), dtype=np.uint8)
    truth[1:3, 3] = 1
    probability = np.full((4, 7), 0.1, dtype=np.float32)
    probability[1:3, 3] = 0.9
    np.savez_compressed(
        path,
        y_true=truth,
        y_prob=probability,
        input_active=np.ones(4, dtype=np.uint8),
        session_lengths=np.asarray([4], dtype=np.int64),
        session_ids=np.asarray(["session__run000__sub000"]),
        source_row_index=np.asarray(rows, dtype=np.int64),
        source_engine_frame_idx=np.asarray(rows, dtype=np.int64) * 3,
    )


def _fixture(tmp_path):
    contract = {
        "surface": {"name": "fixture", "role": "development"},
        "vpt_endpoints": [
            {"name": name, "sha256": f"hash-{index}"}
            for index, name in enumerate(NAMES)
        ],
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    reports = {}
    sidecars = {}
    for index, name in enumerate(NAMES):
        report = tmp_path / f"{name}.json"
        report.write_text(json.dumps({
            "threshold": 0.5,
            "weights": {"sha256": f"hash-{index}"},
            "data": {"manifest_sha256": "manifest"},
        }), encoding="utf-8")
        sidecar = tmp_path / f"{name}.npz"
        _sidecar(sidecar)
        reports[name] = report
        sidecars[name] = sidecar
    return contract_path, reports, sidecars


def test_scorecard_requires_identical_vpt_support(tmp_path):
    contract, reports, sidecars = _fixture(tmp_path)
    result = build_scorecard(contract, reports, sidecars, {})
    assert result["threshold"] == 0.5
    assert result["models"][NAMES[0]]["metrics"]["per_key"]["down"]["recall"] == 1.0
    assert result["models"][NAMES[0]]["metrics"]["all_keys_nonzero_recall_at_0_5"] is False


def test_scorecard_fails_closed_on_row_drift(tmp_path):
    contract, reports, sidecars = _fixture(tmp_path)
    _sidecar(sidecars[NAMES[-1]], rows=(4, 5, 6, 8))
    with pytest.raises(RuntimeError, match="support differs"):
        build_scorecard(contract, reports, sidecars, {})
