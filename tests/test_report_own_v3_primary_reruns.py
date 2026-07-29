from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.report_own_v3_primary_reruns import build_report

pytestmark = pytest.mark.requires_private_artifacts(
    "results/idm/checkpoint-index-own-v3-primary-20260728.json"
)


def test_tracked_own_v3_evidence_validates_and_matches_key_deltas() -> None:
    report, registry = build_report(
        Path("results/idm"),
        Path("experiments/configs/own_v3_primary_reruns.json"),
    )
    assert registry["checkpoint_count"] == 6
    assert len({row["checkpoint_sha256"] for row in registry["records"]}) == 6

    scratch = report["families"]["scratch"]["three_seed_means"][
        "input_active_only"
    ]["paired_mean_delta"]
    tier_b_init = report["families"]["tier_b_init"]["three_seed_means"][
        "input_active_only"
    ]["paired_mean_delta"]
    assert scratch["macro_ap"] == pytest.approx(-0.0034611, abs=1e-6)
    assert scratch["macro_state_f1_at_0.5"] == pytest.approx(0.0146286, abs=1e-6)
    assert scratch["macro_event_f1_oracle_collar2"] == pytest.approx(
        -0.0057519, abs=1e-6
    )
    assert tier_b_init["macro_ap"] == pytest.approx(0.0060213, abs=1e-6)
    assert tier_b_init["macro_state_f1_at_0.5"] == pytest.approx(
        0.0260062, abs=1e-6
    )
    assert tier_b_init["macro_event_f1_oracle_collar2"] == pytest.approx(
        -0.0017368, abs=1e-6
    )

    selected_steps = [
        row["checkpoint"]["best_val_step"]
        for row in report["families"]["tier_b_init"]["seeds"]
    ]
    assert selected_steps == [200, 450, 400]

    tracked_report = json.loads(
        Path("results/idm/own_v3_primary_reruns_delta.json").read_text()
    )
    tracked_registry = json.loads(
        Path("results/idm/checkpoint-index-own-v3-primary-20260728.json").read_text()
    )
    assert tracked_report == report
    assert tracked_registry == registry

    hash_registry = Path("results/idm/checkpoint_sha256.txt").read_text().splitlines()
    for row in registry["records"]:
        expected = (
            f"{row['checkpoint_sha256']}  own_v3/{row['run_id']}_model.pt"
        )
        assert hash_registry.count(expected) == 1
