from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from badeline.train import SegmentSessionDataset, build_source_batch_sampler
from experiments.prepare_provisional_blend_gru import (
    EXPECTED_ARMS,
    LOCAL_FEATURE_FILES,
    LOCAL_TRAIN_IDS,
    LOCAL_VAL_IDS,
    MAX_STEPS,
    UNTOUCHED_SESSION,
    _canonical_sha256,
    _link_sessions,
    _run_config,
    _sessions,
    _validate_contract,
    _validate_local_feature_receipt,
    _validate_nitrogen_receipt,
    _validate_wild_receipt,
    sha256_file,
)


REPO = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO / "experiments/configs/provisional_blend_gru_decision.json"
TEMPLATE_PATH = (
    REPO / "experiments/configs/takeover_features_26m_128x3frame_full_holdout.json"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _local_contract(root: Path) -> tuple[dict, Path]:
    root.mkdir()
    for index, name in enumerate(sorted(LOCAL_FEATURE_FILES)):
        (root / name).write_bytes(f"feature-fixture-{index}:{name}\n".encode())
    source_hashes = {
        session_id: f"{index + 1:064x}"
        for index, session_id in enumerate((*LOCAL_TRAIN_IDS, *LOCAL_VAL_IDS))
    }
    contract = {
        "build_manifest_sha256": "a" * 64,
        "train_sessions_sha256": "b" * 64,
        "val_sessions_sha256": "c" * 64,
        "train_shard_sha256": {
            session_id: source_hashes[session_id] for session_id in LOCAL_TRAIN_IDS
        },
        "val_a_shard_sha256": {
            session_id: source_hashes[session_id] for session_id in LOCAL_VAL_IDS
        },
        "feature_validation_schema": "madeleine.own-v3-features-validation.v1",
    }
    inventory = {
        name: {
            "bytes": (root / name).stat().st_size,
            "sha256": sha256_file(root / name),
        }
        for name in sorted(LOCAL_FEATURE_FILES)
    }
    content = {
        "source": {
            "build_manifest_sha256": contract["build_manifest_sha256"],
            "split": {
                "train_sha256": contract["train_sessions_sha256"],
                "validation_sha256": contract["val_sessions_sha256"],
            },
            "sessions": [
                {
                    "session_id": session_id,
                    "source_npz_sha256": source_hashes[session_id],
                }
                for session_id in (*LOCAL_TRAIN_IDS, *LOCAL_VAL_IDS)
            ],
        },
        "feature_inventory": inventory,
        "sessions": [
            {
                "session_id": session_id,
                "source_npz_sha256": source_hashes[session_id],
                "feature_npz_sha256": inventory[f"{session_id}.npz"]["sha256"],
                "supervision_equal_to_source": {
                    "keys": True,
                    "engine_frame_idx": True,
                    "input_active": True,
                    "session_id": True,
                },
            }
            for session_id in (*LOCAL_TRAIN_IDS, *LOCAL_VAL_IDS)
        ],
    }
    marker = root.parent / "own-v3-marker.json"
    _write_json(
        marker,
        {
            "schema_version": contract["feature_validation_schema"],
            "status": "complete",
            "published_output": str(root),
            "train_sessions": list(LOCAL_TRAIN_IDS),
            "validation_sessions": list(LOCAL_VAL_IDS),
            "checks": {
                "exact_four_source_sessions": True,
                "exact_four_feature_sessions": True,
                "supervision_arrays_equal_to_source": True,
            },
            "content_sha256": _canonical_sha256(content),
            "content": content,
        },
    )
    return contract, marker


def test_contract_fixes_exact_two_arms_and_template() -> None:
    contract = _json(CONTRACT_PATH)

    _validate_contract(contract)

    changed = deepcopy(contract)
    changed["arms"][0]["five_step_cycle"][0]["nitrogen"] -= 1
    with pytest.raises(ValueError, match="NL_90_10 changed five_step_cycle"):
        _validate_contract(changed)


def test_generated_source_sampling_matches_train_sampler_api() -> None:
    contract = _json(CONTRACT_PATH)
    template = _json(TEMPLATE_PATH)
    arm = next(row for row in contract["arms"] if row["name"] == "NL_90_10")
    sources = {"nitrogen": ["n0", "n1"], "local": ["l0"]}
    config = _run_config(
        template=template,
        contract=contract,
        arm=arm,
        sources=sources,
    )
    dataset = object.__new__(SegmentSessionDataset)
    dataset.sessions = [
        SimpleNamespace(session_id="n0"),
        SimpleNamespace(session_id="n1"),
        SimpleNamespace(session_id="l0"),
    ]
    dataset._locations = [(0, 0, 0), (0, 96, 0), (1, 0, 0), (2, 0, 0)]

    sampler = build_source_batch_sampler(
        dataset,
        ["n0", "n1", "l0"],
        config,
        steps=MAX_STEPS,
        seed=0,
        expected_batch_items=16,
    )

    assert sampler is not None
    assert sampler.scheduled_draws == EXPECTED_ARMS["NL_90_10"]["expected_draws"]
    assert sampler.source_session_counts == {"local": 1, "nitrogen": 2}


def test_local_feature_receipt_binds_features_to_corrected_pixels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "own-v3-features"
    contract, marker = _local_contract(root)

    report = _validate_local_feature_receipt(marker, root=root, contract=contract)

    assert report["content_sha256"] == _canonical_sha256(report["content"])
    target = root / f"{LOCAL_TRAIN_IDS[0]}.npz"
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="byte count changed|SHA-256 changed"):
        _validate_local_feature_receipt(marker, root=root, contract=contract)


def test_hardlink_inventory_hashes_exact_source_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    shard = source / "s0.npz"
    shard.write_bytes(b"exact-feature-bytes")
    inventory: list[dict] = []

    _link_sessions(
        source=source,
        destination=destination,
        session_ids=["s0"],
        inventory=inventory,
        source_name="nitrogen",
    )

    assert os.stat(shard).st_ino == os.stat(destination / "s0.npz").st_ino
    assert inventory == [
        {
            "session_id": "s0",
            "source": "nitrogen",
            "bytes": len(b"exact-feature-bytes"),
            "mtime_ns": shard.stat().st_mtime_ns,
            "sha256": sha256_file(shard),
        }
    ]


def test_sealed_session_is_rejected_from_every_input_list(tmp_path: Path) -> None:
    path = tmp_path / "train.txt"
    path.write_text(f"safe\n{UNTOUCHED_SESSION}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sealed untouched test"):
        _sessions(path)


def test_wild_receipt_requires_exact_npz_membership(tmp_path: Path) -> None:
    root = tmp_path / "wild"
    root.mkdir()
    wild_ids = ["wild0", "wild1"]
    y4n_ids = ["y4nQHqYSObI__r000"]
    for session_id in (*wild_ids, *y4n_ids):
        (root / f"{session_id}.npz").write_bytes(session_id.encode())
    (root / "train_sessions.txt").write_text("\n".join(wild_ids) + "\n")
    (root / "val_sessions.txt").write_text("\n".join(y4n_ids) + "\n")
    source_sha = "d" * 64
    marker = tmp_path / "wild-marker.json"
    _write_json(
        marker,
        {
            "format_version": "madeleine.wild-provisional-gru-features-validated.v1",
            "source_manifest_sha256": source_sha,
            "training_shards": len(wild_ids),
            "validation_shards": len(y4n_ids),
            "total_hardlinks": len(wild_ids) + len(y4n_ids),
            "checks": {"exact_npz_inventory": True, "hardlinks": True},
            "files": {
                name: sha256_file(root / name)
                for name in ("train_sessions.txt", "val_sessions.txt")
            },
        },
    )
    marker_sha = sha256_file(marker)

    _validate_wild_receipt(
        marker,
        root=root,
        expected_sha256=marker_sha,
        expected_source_sha256=source_sha,
        wild_ids=wild_ids,
        y4n_ids=y4n_ids,
    )
    (root / f"{UNTOUCHED_SESSION}.npz").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="NPZ inventory changed"):
        _validate_wild_receipt(
            marker,
            root=root,
            expected_sha256=marker_sha,
            expected_source_sha256=source_sha,
            wild_ids=wild_ids,
            y4n_ids=y4n_ids,
        )


def test_nitrogen_receipt_requires_deep_clean_bound_root(tmp_path: Path) -> None:
    root = tmp_path / "nitrogen"
    root.mkdir()
    marker = tmp_path / "nitrogen-validation.json"
    _write_json(
        marker,
        {
            "ok": True,
            "errors": [],
            "deep_shards": True,
            "observed": {
                "sessions": 1554,
                "unflagged_sessions": 1078,
                "deep_shards_checked": 1554,
                "hardlinks_checked": 1554,
                "skipped_short_frames": 0,
                "tail_truncated_frames": 0,
                "imputed_tail_frames": 0,
            },
            "paths": {"output_root": str(root)},
        },
    )

    _validate_nitrogen_receipt(
        marker, root=root, expected_sha256=sha256_file(marker)
    )
    changed = _json(marker)
    changed["observed"]["deep_shards_checked"] -= 1
    _write_json(marker, changed)
    with pytest.raises(ValueError, match="deep_shards_checked"):
        _validate_nitrogen_receipt(
            marker, root=root, expected_sha256=sha256_file(marker)
        )
