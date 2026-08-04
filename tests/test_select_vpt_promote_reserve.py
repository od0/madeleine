import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

import experiments.select_vpt_promote_reserve as selector


REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_VALID = Path(
    "results/idm/nitrogen_full_210train_y4n_holdout_26m_128x3_s0/run_meta.json"
)
UNFLAGGED = Path(
    "results/idm/nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0/run_meta.json"
)
INVENTORY = Path("results/idm/dynamics_pretraining_cd_s0/inventory.json")
ALL_VALID_SHA = "8c7d104d4a0072385b4513fcf8c8525ba3bc3af3cd12c74066984d7c149a18c7"
UNFLAGGED_SHA = "434319316d3ddcae975ab50e96837e1e179762af3ce87fc99e46dd65517f3a67"
INVENTORY_SHA = "3197ccd684f67d5d45b2bbbc5a05eaa83816b18f9fa315459476aa16639ec9f8"


def _args(output_dir: Path) -> list[str]:
    return [
        "--all-valid-run-meta",
        str(ALL_VALID),
        "--all-valid-run-meta-sha256",
        ALL_VALID_SHA,
        "--unflagged-run-meta",
        str(UNFLAGGED),
        "--unflagged-run-meta-sha256",
        UNFLAGGED_SHA,
        "--inventory",
        str(INVENTORY),
        "--inventory-sha256",
        INVENTORY_SHA,
        "--output-dir",
        str(output_dir),
    ]


def test_nearest_subset_is_order_independent_and_exact() -> None:
    candidates = [("v3", 3_600), ("v1", 1_200), ("v2", 2_400)]
    first = selector.nearest_subset(
        candidates, Fraction(4_800), stratum="opencv_native_60hz/<5m"
    )
    second = selector.nearest_subset(
        list(reversed(candidates)),
        Fraction(4_800),
        stratum="opencv_native_60hz/<5m",
    )
    assert first == second
    assert first[1:] == (4_800, 1_200)
    assert sum(dict(candidates)[video_id] for video_id in first[0]) == 4_800


@pytest.mark.requires_private_artifacts(
    "results/idm/nitrogen_full_210train_y4n_holdout_26m_128x3_s0/run_meta.json",
    "results/idm/dynamics_pretraining_cd_s0/inventory.json",
)
def test_frozen_metadata_selects_exact_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    output = tmp_path / "reserve"
    assert selector.main(_args(output)) == 0
    receipt = json.loads((output / "reserve_receipt.json").read_text())
    reserve_text = (output / "reserve_video_ids.txt").read_text()
    retained_text = (output / "retained_max_training_video_ids.txt").read_text()

    assert receipt["candidate_pool"] == {
        "videos": 118,
        "native_frames": 9_702_000,
        "hours_fraction": {"numerator": 9_702_000, "denominator": 216_000},
    }
    assert receipt["reserve"]["videos"] == 66
    assert receipt["reserve"]["native_frames"] == 5_816_400
    assert receipt["reserve"]["phase0_rows"] == 1_938_800
    assert receipt["reserve"]["base_windows"] == 29_909
    assert receipt["retained_nitrogen_train"]["videos"] == 144
    assert receipt["retained_nitrogen_train"]["sessions"] == 1_272
    assert receipt["retained_nitrogen_train"]["native_frames"] == 26_221_200
    assert receipt["retained_nitrogen_train"]["phase0_rows"] == 8_740_400
    assert receipt["retained_nitrogen_train"]["base_windows"] == 134_765
    assert receipt["retained_max_train_with_frozen_wild"]["base_windows"] == 147_931
    assert receipt["retained_max_train_with_frozen_wild"]["steps_per_epoch"] == 1_156
    assert receipt["retained_max_train_with_frozen_wild"]["optimizer_steps"] == 23_120
    assert all(receipt["proofs"].values())
    assert hashlib.sha256(reserve_text.encode()).hexdigest() == (
        "30cd62dcbeffb9aac217b22a97f6a38554bb3e818481fa9d070662af12fc0a20"
    )
    assert hashlib.sha256(retained_text.encode()).hexdigest() == (
        "e38c5b316fc0180ed6b0979b275baaf1b76c7fa8d3f315b48da6e25ca7cd1522"
    )


@pytest.mark.requires_private_artifacts(
    "results/idm/nitrogen_full_210train_y4n_holdout_26m_128x3_s0/run_meta.json",
    "results/idm/dynamics_pretraining_cd_s0/inventory.json",
)
def test_input_hash_mismatch_fails_before_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    args = _args(tmp_path / "reserve")
    index = args.index("--inventory-sha256") + 1
    args[index] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        selector.main(args)
