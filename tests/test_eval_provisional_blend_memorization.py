from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from data.schema import KEY_ORDER
import experiments.eval_provisional_blend_memorization as evaluator


class _FrameLogitModel(torch.nn.Module):
    """Use the target-aligned synthetic feature rows as seven logits."""

    def forward_segment(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        features = batch["features"]
        windows = features.shape[1] - 3 + 1
        return features[:, 1 : 1 + windows, : len(KEY_ORDER)]


def _config() -> dict[str, object]:
    return {
        "window": 3,
        "window_mode": "centered",
        "input_config": "pixels",
        "history_len": 2,
        "history_gap": 0,
        "segment_windows": 2,
        "active_targets_only": True,
        "transition_weight": 8.0,
        "precomputed_features": True,
        "frame_stride": 1,
    }


def _write_feature_shard(
    root: Path,
    session_id: str,
    *,
    input_active: np.ndarray | None = None,
) -> np.ndarray:
    root.mkdir(parents=True, exist_ok=True)
    target_truth = np.asarray(
        [
            [(row + column) % 2 for column in range(len(KEY_ORDER))]
            for row in range(6)
        ],
        dtype=np.uint8,
    )
    keys = np.zeros((8, len(KEY_ORDER)), dtype=np.uint8)
    keys[1:7] = target_truth
    features = np.zeros((8, len(KEY_ORDER)), dtype=np.float32)
    features[1:7] = np.where(target_truth == 1, 6.0, -6.0)
    if input_active is None:
        input_active = np.ones(8, dtype=np.uint8)
    np.savez_compressed(
        root / f"{session_id}.npz",
        features=features,
        keys=keys,
        engine_frame_idx=np.arange(8, dtype=np.int64),
        input_active=np.asarray(input_active, dtype=np.uint8),
        session_id=np.asarray(session_id),
    )
    return target_truth


def test_complete_segment_pool_is_unique_unweighted_and_fixed(
    tmp_path: Path,
) -> None:
    data = tmp_path / "features"
    truth = _write_feature_shard(data, "train")

    result = evaluator.evaluate_complete_segment_pool(
        _FrameLogitModel(),
        _config(),
        data,
        ["train"],
        expected_segment_items=3,
        device="cpu",
        batch_segments=2,
        scratch_dir=tmp_path / "scratch",
    )

    assert result["support"]["segment_items"] == 3
    assert result["support"]["target_frames"] == 6
    assert result["support"]["session_segment_items"] == {"train": 3}
    assert result["metrics"]["average_precision"]["macro"] == 1.0
    assert result["metrics"]["state_f1_fixed_0_5"]["macro"] == 1.0
    expected_bce = float(np.logaddexp(0.0, -6.0))
    assert result["metrics"]["unweighted_bce"]["macro"] == pytest.approx(
        expected_bce, abs=1e-7
    )
    assert result["metrics"]["prevalence"]["per_key"] == {
        key: float(truth[:, index].mean())
        for index, key in enumerate(KEY_ORDER)
    }
    assert not list((tmp_path / "scratch").iterdir())


def test_complete_segment_pool_preserves_active_gap_boundaries(
    tmp_path: Path,
) -> None:
    data = tmp_path / "features"
    # Target indices are 1..6. Making index 3 inactive splits eligible starts
    # into runs of two and three; only one full two-window item survives each.
    active = np.ones(8, dtype=np.uint8)
    active[3] = 0
    _write_feature_shard(data, "gapped", input_active=active)

    count, per_session = evaluator.count_complete_segment_items(
        data, ["gapped"], _config()
    )

    assert count == 2
    assert per_session == {"gapped": 2}


@pytest.mark.parametrize(
    "session_id",
    [
        evaluator.UNTOUCHED_SESSION_ID,
        evaluator.B1_ID,
        evaluator.VAL_B_ID,
    ],
)
def test_embargoed_session_is_rejected_before_npz_access(
    session_id: str,
) -> None:
    with pytest.raises(ValueError, match="embargoed session"):
        evaluator._require_allowed_session_ids([session_id], "test membership")


def _write_feature_view(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "view"
    root.mkdir()
    rows = [
        {
            "session_id": "n1",
            "source": "nitrogen",
            "bytes": 11,
            "sha256": "a" * 64,
        },
        {
            "session_id": "l1",
            "source": "local",
            "bytes": 12,
            "sha256": "b" * 64,
        },
        {
            "session_id": evaluator.LOCAL_VAL_A_ID,
            "source": "local",
            "bytes": 13,
            "sha256": "c" * 64,
        },
    ]
    inventory = root / evaluator.HARDLINK_INVENTORY
    inventory.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    local_val = root / evaluator.LOCAL_VAL_A_LIST
    local_val.write_text(evaluator.LOCAL_VAL_A_ID + "\n", encoding="utf-8")
    generated = {
        inventory.name: evaluator.sha256_file(inventory),
        local_val.name: evaluator.sha256_file(local_val),
    }
    contract_sha256 = "d" * 64
    receipt = {
        "schema_version": evaluator.FEATURE_VIEW_SCHEMA_VERSION,
        "study_id": evaluator.STUDY_ID,
        "contract": {"sha256": contract_sha256},
        "sealed_untouched_session_present": False,
        "temporary_files_present": False,
        "hardlinks": {
            "verified": True,
            "inventory_file": inventory.name,
            "inventory_sha256": evaluator.sha256_file(inventory),
            "files": len(rows),
        },
        "generated_files": generated,
    }
    (root / evaluator.FEATURE_VIEW_RECEIPT).write_text(
        json.dumps(receipt) + "\n", encoding="utf-8"
    )
    return root, contract_sha256


def test_feature_view_binds_contract_inventory_and_source_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, contract_sha256 = _write_feature_view(tmp_path)
    monkeypatch.setattr(
        evaluator,
        "EXPECTED_FEATURE_VIEW_SOURCE_COUNTS",
        {"nitrogen": 1, "wild_provisional": 0, "local": 2},
    )

    receipt = evaluator.validate_feature_view(
        root,
        contract_sha256=contract_sha256,
        source_sessions={"nitrogen": ["n1"], "local": ["l1"]},
        local_val_a_sessions=[evaluator.LOCAL_VAL_A_ID],
    )

    assert receipt["hardlink_inventory_rows"] == 3
    assert receipt["source_sessions"] == {
        "nitrogen": ["n1"],
        "local": ["l1"],
    }
    with pytest.raises(ValueError, match="another contract"):
        evaluator.validate_feature_view(
            root,
            contract_sha256="e" * 64,
            source_sessions={"nitrogen": ["n1"], "local": ["l1"]},
            local_val_a_sessions=[evaluator.LOCAL_VAL_A_ID],
        )


def test_selected_shards_are_rehashed_and_bound_to_training_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "view"
    source = tmp_path / "source.npz"
    root.mkdir()
    source.write_bytes(b"immutable feature bytes")
    target = root / "n1.npz"
    os.link(source, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    inventory = {
        "n1": {
            "source": "nitrogen",
            "bytes": target.stat().st_size,
            "sha256": digest,
        }
    }

    receipt = evaluator.validate_selected_shards(
        root,
        source="nitrogen",
        session_ids=["n1"],
        inventory=inventory,
        run_meta_shards={"n1": digest},
    )

    assert receipt["sessions"] == 1
    assert receipt["shards"]["n1"]["sha256"] == digest
    with pytest.raises(ValueError, match="training-time shard hash"):
        evaluator.validate_selected_shards(
            root,
            source="nitrogen",
            session_ids=["n1"],
            inventory=inventory,
            run_meta_shards={"n1": "0" * 64},
        )


def _surface(bce: float, ap: float, f1: float) -> dict[str, object]:
    def metric(value: float) -> dict[str, object]:
        return {
            "per_key": {key: value for key in KEY_ORDER},
            "macro": value,
        }

    return {
        "metrics": {
            "unweighted_bce": metric(bce),
            "average_precision": metric(ap),
            "state_f1_fixed_0_5": metric(f1),
        }
    }


def test_local_gap_names_its_signed_direction() -> None:
    gap = evaluator.local_generalization_gap(
        _surface(0.1, 0.9, 0.8),
        _surface(0.4, 0.3, 0.2),
    )

    assert gap["unweighted_bce"]["direction"] == "val_a_minus_train"
    assert gap["unweighted_bce"]["macro"] == pytest.approx(0.3)
    assert gap["average_precision"]["direction"] == "train_minus_val_a"
    assert gap["average_precision"]["macro"] == pytest.approx(0.6)
    assert gap["state_f1_fixed_0_5"]["macro"] == pytest.approx(0.6)


def test_atomic_report_marker_is_content_bound_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "diagnostic.json"
    marker_path = tmp_path / ".diagnostic.done.json"
    report = {"schema_version": evaluator.REPORT_SCHEMA_VERSION, "value": 7}

    marker = evaluator.publish_atomic_report(
        report_path,
        marker_path,
        report,
        {"run_id": "run", "checkpoint_sha256": "a" * 64},
    )

    assert marker["status"] == "complete"
    assert marker["report_sha256"] == evaluator.sha256_file(report_path)
    assert json.loads(marker_path.read_text())["report_sha256"] == marker[
        "report_sha256"
    ]
    with pytest.raises(ValueError, match="refusing to overwrite"):
        evaluator.publish_atomic_report(
            report_path,
            marker_path,
            report,
            {"run_id": "run"},
        )


def _git(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(tmp_path), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def test_diagnostic_source_is_clean_committed_and_content_bound(
    tmp_path: Path,
) -> None:
    module = tmp_path / evaluator.DIAGNOSTIC_RELATIVE_PATH
    module.parent.mkdir(parents=True)
    module.write_text("diagnostic = 'exact'\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "add", evaluator.DIAGNOSTIC_RELATIVE_PATH.as_posix())
    _git(tmp_path, "commit", "-qm", "diagnostic")
    commit = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()

    receipt = evaluator.bind_diagnostic_source(
        tmp_path,
        commit,
        loaded_module_path=module,
    )

    assert receipt == {
        "git_commit": commit,
        "relative_path": evaluator.DIAGNOSTIC_RELATIVE_PATH.as_posix(),
        "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
    }
    module.write_text("diagnostic = 'dirty'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from declared commit"):
        evaluator.bind_diagnostic_source(
            tmp_path,
            commit,
            loaded_module_path=module,
        )


def test_diagnostic_source_refuses_uncommitted_or_wrong_checkout(
    tmp_path: Path,
) -> None:
    module = tmp_path / evaluator.DIAGNOSTIC_RELATIVE_PATH
    module.parent.mkdir(parents=True)
    module.write_text("diagnostic = True\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "add", evaluator.DIAGNOSTIC_RELATIVE_PATH.as_posix())
    _git(tmp_path, "commit", "-qm", "diagnostic")
    commit = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()

    with pytest.raises(ValueError, match="outside --repo"):
        evaluator.bind_diagnostic_source(
            tmp_path,
            commit,
            loaded_module_path=tmp_path / "other.py",
        )
    with pytest.raises(ValueError, match="40 lowercase hex"):
        evaluator.bind_diagnostic_source(
            tmp_path,
            "HEAD",
            loaded_module_path=module,
        )
