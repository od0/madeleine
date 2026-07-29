from __future__ import annotations

from pathlib import Path
import re
import subprocess
import pytest

pytestmark = pytest.mark.requires_private_artifacts(
    "experiments/run_dynamics_downstream.sh"
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run_dynamics_downstream.sh"


def _source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_runner_has_valid_shell_syntax_and_frozen_arm_ids() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    source = _source()
    assert "dynamics_c_full_210train_y4n_holdout_26m_128x3_s0" in source
    assert "dynamics_d_full_210train_y4n_holdout_26m_128x3_s0" in source
    assert 'case "${arm}" in' in source
    assert "ARM must be exactly C or D" in source


def test_runner_freezes_existing_gru_recipe_and_exact_endpoint() -> None:
    source = _source()
    assert "takeover_features_26m_128x3frame_full_holdout.json" in source
    assert "expected_steps=20458" in source
    assert '--max-steps "${expected_steps}"' in source
    assert "--seed 0" in source
    assert 'config["max_steps"] = steps' in source
    assert 'config["eval_interval"] = steps' in source
    assert '"model": "BadelineIDM default GRU"' in source
    assert '"checkpoint_reselection": False' in source
    assert '"weights_for_release": "final_state_dict"' in source
    assert 'checkpoint.get("final_state_dict")' in source
    assert '[row.get("step") for row in lines] != [0, steps]' in source


def test_runner_binds_deep_assembly_checkpoint_inventory_and_exact_splits() -> None:
    source = _source()
    for token in (
        "madeleine.dynamics-supervised-feature-assembly.v1",
        "madeleine.dynamics-supervised-feature-assembly-complete.v1",
        "madeleine.dynamics-supervised-feature-assembly-validation.v1",
        'validation.get("deep_shards") is not True',
        'validation.get("deep_sources") is not True',
        'validation.get("checkpoint_sha256") != checkpoint_sha',
        'validation.get("inventory_sha256") != inventory_sha',
        'completion.get("assembly_manifest_sha256") != sha256(assembly_path)',
        'completion.get("shard_hashes_sha256") != sha256(hashes_path)',
        '"all_sessions": 1554',
        '"train_sessions": 1538',
        '"train_videos": 210',
        '"validation_sessions": 16',
        '"later_eight_sessions": 8',
        '"frames": 32598000',
        'range(16)',
        'expected_val[8:]',
    ):
        assert token in source
    assert "deep supervised validation receipt is stale" in source


def test_evaluator_call_is_final_fixed_later8_only() -> None:
    source = _source()
    match = re.search(
        r'-m experiments\.eval_dynamics_downstream \\\n+(.*?)>"\$\{logs\}/eval_y4n_later8\.log"',
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    call = match.group(1)
    assert '--sessions "${later8_sessions}"' in call
    assert '--assembly-validation "${assembly_validation}"' in call
    assert '--split-receipt "${split_receipt}"' in call
    assert '--ssl-checkpoint-sha256 "${ssl_checkpoint_sha256}"' in call
    assert '--inventory-sha256 "${inventory_sha256}"' in call
    assert "--weights" not in call
    assert "oracle" not in call.lower()
    assert "calibr" not in call.lower()
    assert "b1" not in call.lower()


def test_runner_refuses_overwrite_and_atomically_closes_receipts() -> None:
    source = _source()
    assert 'if [[ -e ${path} || -L ${path} ]]' in source
    assert "refusing to overwrite dynamics downstream artifact" in source
    assert 'run / "dynamics_downstream_meta.json"' in source
    assert 'atomic_json(meta_path, meta, replace=True)' in source
    assert source.count("temporary.replace(") >= 3
    assert "madeleine.dynamics-downstream-wrapper-complete.v1" in source
    assert '"status": "complete"' in source


def test_runner_has_no_auxiliary_evaluation_or_training_surface() -> None:
    source = _source().lower()
    assert "eval_event_b1" not in source
    assert "eval_wild_provisional" not in source
    assert "--init " not in source
    assert '"b1_accessed": false' in source
    assert '"oracle_thresholds": false' in source
    assert '"calibration": false' in source
    assert 'lowered.startswith("rec_")' in source
    assert '"sealed" in lowered' in source
    assert '"untouched" in lowered' in source
