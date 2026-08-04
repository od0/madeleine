from __future__ import annotations

from pathlib import Path

import pytest

from experiments.validate_vpt_paper_idm_release import validate_bundle


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/idm/vpt_paper_idm_482133390_tier_b_13p45h_s0_evidence"


@pytest.mark.requires_private_artifacts(
    "results/idm/vpt_paper_idm_482133390_tier_b_13p45h_s0_evidence/training/run_meta.json",
    "results/idm/VPT_SMALL_113M_RELEASE_VALIDATION.json",
)
def test_matched_paper_idm_release_is_exact_and_support_aligned() -> None:
    validation = validate_bundle(
        EVIDENCE / "training",
        EVIDENCE / "evaluation/vpt_paper_idm_482133390_tier_b_13p45h_s0_eval",
        ROOT / "results/idm/VPT_SMALL_113M_RELEASE_VALIDATION.json",
    )

    assert validation["implementation_success"] is True
    assert validation["matched_tier_b_scientific_result"] == "negative"
    assert validation["phase_3_maximum_generation_build_eligible"] is True
    assert validation["phase_4_maximum_training_authorized_by_this_receipt"] is False
    assert validation["support"] == {
        "rows": 4224,
        "streams": 21,
        "unique_source_rows": 4224,
        "all_seven_sidecars_identical": True,
        "probabilities_finite": True,
    }
    assert validation["checkpoint_receipts"]["count"] == 20
    assert validation["endpoint"]["selected_epoch"] == 3
    assert validation["metrics"]["paper_final"]["macro_ap"] == pytest.approx(
        0.18435860179791225
    )
    assert validation["deltas"]["paper_final_minus_vpt_small_final"][
        "macro_ap"
    ] == pytest.approx(-0.17591378203854768)
