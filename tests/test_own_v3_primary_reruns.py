from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import own_v3_primary_reruns as reruns


CONTRACT = Path("experiments/configs/own_v3_primary_reruns.json")


def test_contract_is_exactly_two_families_by_three_seeds() -> None:
    contract = json.loads(CONTRACT.read_text())
    reruns._validate_contract(contract)
    assert {
        (row["family"], row["seed"], row["run_id"])
        for row in contract["runs"]
    } == {
        (family, seed, f"own_features_v3_32nc_s{seed}" if family == "scratch"
         else f"own_features_v3_tier_b_init_32nc_s{seed}")
        for family in ("scratch", "tier_b_init")
        for seed in range(3)
    }


def test_contract_rejects_missing_seed() -> None:
    contract = copy.deepcopy(json.loads(CONTRACT.read_text()))
    contract["runs"].pop()
    with pytest.raises(ValueError, match="six-run set changed"):
        reruns._validate_contract(contract)


def test_state_digest_rejects_nonfinite_tensor() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        reruns._state_digest({"weight": torch.tensor([float("nan")])})


def test_finalize_validates_support_and_writes_marker(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text())
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    run = tmp_path / "own_features_v3_32nc_s0"
    run.mkdir()
    artifacts = {
        "model.pt": b"model",
        "config.json": b"{}",
        "run_meta.json": b"{}",
        "log.jsonl": b"{}\n",
    }
    for name, value in artifacts.items():
        (run / name).write_bytes(value)
    registration = {
        "checkpoint": {"sha256": reruns.sha256_file(run / "model.pt")}
    }
    (run / "checkpoint-registration.json").write_text(json.dumps(registration))
    metrics = {
        "per_key_ap": {},
        "per_key_f1": {},
        "per_key_calibration": {},
        "onset_timing_errors": {},
        "transition_f1_at_0.5": {},
        "transition_f1_oracle": {},
        "transition_f1_oracle_collars": {},
    }
    report = tmp_path / "own_features_v3_32nc_s0_val_a.json"
    report.write_text(json.dumps({
        "sessions": contract["data_contract"]["validation_sessions"],
        "weights": "selected",
        "all_frames": {"n": 29_086, "metrics": metrics},
        "input_active_only": {"n": 25_028, "metrics": metrics},
    }))
    truth = np.zeros((29_086, 7), dtype=np.uint8)
    active = np.zeros(29_086, dtype=np.uint8)
    active[:25_028] = 1
    np.savez_compressed(
        report.with_name(report.stem + "_preds.npz"),
        y_true=truth,
        y_prob=np.zeros_like(truth, dtype=np.float32),
        input_active=active,
        session_lengths=np.asarray([29_086], dtype=np.int64),
        session_ids=np.asarray(["rec_20260724_171305_5min__stream000"]),
    )
    marker = tmp_path / ".done.json"
    result = reruns.finalize(
        contract_path=contract_path,
        run=run,
        report_path=report,
        marker_path=marker,
        family="scratch",
        seed=0,
    )
    assert result["status"] == "complete"
    assert marker.is_file()
