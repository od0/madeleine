"""Contract tests for fixed-crop, pairwise HUD-change verification.

These tests cover only pure geometry, pairing, aggregation, response
validation, and retry bookkeeping.  They do not load a model, GPU, or image.
"""

import json
import math

import pytest

from harvest.verify_hud_change_vlm import (
    SWEEP_REGIONS,
    aggregate_pair_verdicts,
    aggregate_regions,
    construct_pairs,
    finalize_response_attempts,
    parse_pair_response,
)


VALID_DIFFERENT = {
    "verdict": "different_state",
    "differing_controls": ["D-pad Left"],
    "evidence": "The left d-pad arm is filled only in the second crop.",
}


def response(**overrides):
    return json.dumps({**VALID_DIFFERENT, **overrides})


def region_verdicts(**by_region):
    """Build pair-verdict dicts labeled with their sweep region."""

    output = []
    for region, verdicts in by_region.items():
        for verdict in verdicts:
            output.append(
                {"region": region, "verdict": verdict, "validation_errors": []}
            )
    return output


class TestRegionReduction:
    def test_change_in_one_quadrant_confirms_the_video(self):
        verdicts = region_verdicts(
            top_left=["no_input_overlay"] * 6,
            top_right=["no_input_overlay"] * 6,
            bottom_left=["different_state"] * 2 + ["same_state"] * 4,
            bottom_right=["no_input_overlay"] * 6,
        )
        assert aggregate_regions(verdicts)["result"] == "changing_overlay_confirmed"

    def test_static_overlay_in_one_quadrant_beats_no_overlay_elsewhere(self):
        verdicts = region_verdicts(
            top_left=["no_input_overlay"] * 6,
            top_right=["same_state"] * 6,
            bottom_left=["no_input_overlay"] * 6,
            bottom_right=["no_input_overlay"] * 6,
        )
        assert aggregate_regions(verdicts)["result"] == "static_overlay"

    def test_no_overlay_requires_every_quadrant_to_agree(self):
        all_none = region_verdicts(
            **{region: ["no_input_overlay"] * 6 for region in SWEEP_REGIONS}
        )
        assert aggregate_regions(all_none)["result"] == "no_overlay"
        one_insufficient = region_verdicts(
            top_left=["no_input_overlay"] * 6,
            top_right=["no_input_overlay"] * 6,
            bottom_left=["no_input_overlay"] * 6,
            bottom_right=["illegible"] * 6,
        )
        assert (
            aggregate_regions(one_insufficient)["result"]
            == "insufficient_evidence"
        )

    def test_single_diff_pair_is_not_confirmation(self):
        verdicts = region_verdicts(
            top_left=["different_state"] + ["same_state"] * 5,
            top_right=["no_input_overlay"] * 6,
            bottom_left=["no_input_overlay"] * 6,
            bottom_right=["no_input_overlay"] * 6,
        )
        assert (
            aggregate_regions(verdicts)["result"] == "insufficient_evidence"
        )

    def test_empty_input_is_insufficient(self):
        assert aggregate_regions([])["result"] == "insufficient_evidence"


class TestStackedComposite:
    def test_composite_stacks_with_red_divider(self):
        PIL = pytest.importorskip("PIL")
        from PIL import Image

        from harvest.verify_hud_change_vlm import STACK_DIVIDER_PX, stack_pair

        top = Image.new("RGB", (40, 30), (0, 255, 0))
        bottom = Image.new("RGB", (40, 30), (0, 0, 255))
        composite = stack_pair(top, bottom)
        assert composite.size == (40, 60 + STACK_DIVIDER_PX)
        assert composite.getpixel((20, 10)) == (0, 255, 0)
        assert composite.getpixel((20, 30 + STACK_DIVIDER_PX // 2)) == (255, 0, 0)
        assert composite.getpixel((20, 40 + STACK_DIVIDER_PX)) == (0, 0, 255)

    def test_composite_rejects_mismatched_sizes(self):
        PIL = pytest.importorskip("PIL")
        from PIL import Image

        from harvest.verify_hud_change_vlm import stack_pair

        with pytest.raises(ValueError):
            stack_pair(
                Image.new("RGB", (40, 30)), Image.new("RGB", (41, 30))
            )


class TestPairConstruction:
    @pytest.mark.parametrize(
        "frame_count,expected",
        [
            (
                6,
                [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)],
            ),
            (3, [(0, 1), (1, 2), (0, 2)]),
            (2, [(0, 1)]),
        ],
    )
    def test_pairs_are_ordered_and_deduplicated(self, frame_count, expected):
        assert construct_pairs(frame_count) == expected


class TestAggregation:
    @pytest.mark.parametrize(
        "verdicts,expected",
        [
            (["same_state"] * 6, "static_overlay"),
            (
                ["different_state", "same_state", "different_state"],
                "changing_overlay_confirmed",
            ),
            (
                ["different_state", "same_state", "same_state", "same_state"],
                "insufficient_evidence",
            ),
            (
                ["no_input_overlay"] * 4 + ["same_state"],
                "no_overlay",
            ),
            (["illegible"] * 6, "insufficient_evidence"),
            ([], "insufficient_evidence"),
        ],
    )
    def test_truth_table(self, verdicts, expected):
        assert aggregate_pair_verdicts(verdicts)["result"] == expected

    def test_static_threshold_uses_ceil_of_valid_state_pairs(self):
        verdicts = ["same_state"] * 3 + ["no_input_overlay"] * 3
        aggregate = aggregate_pair_verdicts(verdicts)
        assert aggregate["n_same"] == math.ceil(aggregate["valid"] / 2)
        assert aggregate["result"] == "static_overlay"

    def test_failed_validation_is_not_counted_as_illegible(self):
        aggregate = aggregate_pair_verdicts(
            [
                {
                    "verdict": "illegible",
                    "validation_errors": ["invalid_json"],
                }
            ]
        )
        assert aggregate["n_illeg"] == 0
        assert aggregate["invalid_pairs"] == 1


class TestPairResponseParsing:
    def test_valid_different_state(self):
        parsed = parse_pair_response(response())
        assert parsed["verdict"] == "different_state"
        assert parsed["differing_controls"] == ["D-pad Left"]
        assert parsed["validation_errors"] == []

    def test_different_state_requires_control(self):
        parsed = parse_pair_response(response(differing_controls=[]))
        assert parsed["verdict"] == "illegible"
        assert (
            "different_state_lacks_differing_controls"
            in parsed["validation_errors"]
        )

    def test_same_state_forbids_control(self):
        parsed = parse_pair_response(response(verdict="same_state"))
        assert parsed["verdict"] == "illegible"
        assert (
            "non_different_state_has_differing_controls"
            in parsed["validation_errors"]
        )

    def test_unknown_verdict_fails_closed(self):
        parsed = parse_pair_response(response(verdict="probably_different"))
        assert parsed["verdict"] == "illegible"
        assert "invalid_verdict" in parsed["validation_errors"]

    def test_extraneous_text_fails_closed(self):
        parsed = parse_pair_response(response() + "\nThis looks different.")
        assert parsed["verdict"] == "illegible"
        assert "extraneous_text_after_json" in parsed["validation_errors"]


class TestRetryBookkeeping:
    def test_valid_first_response_has_no_retry_response_key(self):
        first_raw = response()
        finalized = finalize_response_attempts(
            first_raw, parse_pair_response(first_raw)
        )
        assert finalized["retry_count"] == 0
        assert "raw_response_retry" not in finalized
        assert "retry_validation_errors" not in finalized

    def test_valid_retry_replaces_first_fail_closed_parse(self):
        first_raw = "not json"
        retry_raw = response()
        finalized = finalize_response_attempts(
            first_raw,
            parse_pair_response(first_raw),
            retry_raw=retry_raw,
            retry_parsed=parse_pair_response(retry_raw),
        )
        assert finalized["verdict"] == "different_state"
        assert finalized["retry_count"] == 1
        assert finalized["raw_response"] == first_raw
        assert finalized["raw_response_retry"] == retry_raw
        assert "retry_validation_errors" not in finalized

    def test_second_failure_preserves_first_errors(self):
        first_raw = "not json"
        retry_raw = response(differing_controls=[])
        first = parse_pair_response(first_raw)
        retry = parse_pair_response(retry_raw)
        finalized = finalize_response_attempts(
            first_raw,
            first,
            retry_raw=retry_raw,
            retry_parsed=retry,
        )
        assert finalized["validation_errors"] == first["validation_errors"]
        assert finalized["retry_validation_errors"] == retry["validation_errors"]
        assert finalized["retry_count"] == 1
