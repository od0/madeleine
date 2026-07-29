from __future__ import annotations

import json

from harvest.classify_layout_families_vlm import (
    ACTIONS,
    combine_order_checks,
    parse_response,
)


def test_parse_reference_match_requires_visible_reason() -> None:
    result = parse_response(json.dumps({
        "decision": "same_normalized_layout",
        "reference_video_id": "0H53eYMpGsg",
        "confidence": "high",
        "centers_normalized": None,
        "reason": "same seven labeled cells at the same normalized centers",
    }))

    assert result == {
        "decision": "same_normalized_layout",
        "reference_video_id": "0H53eYMpGsg",
        "confidence": "high",
        "centers_normalized": None,
        "reason": "same seven labeled cells at the same normalized centers",
    }


def test_parse_reference_match_without_reason_fails_closed() -> None:
    result = parse_response(json.dumps({
        "decision": "same_normalized_layout",
        "reference_video_id": "0H53eYMpGsg",
        "confidence": "high",
        "centers_normalized": None,
    }))

    assert result["decision"] == "unknown"
    assert result["reference_video_id"] is None
    assert result["confidence"] == "low"
    assert result["parse_error"] == "missing_reason"


def test_parse_complete_normalized_geometry() -> None:
    centers = {
        action: [0.1 + index * 0.1, 0.25]
        for index, action in enumerate(ACTIONS)
    }
    result = parse_response(json.dumps({
        "decision": "explicit_geometry",
        "reference_video_id": None,
        "confidence": "medium",
        "centers_normalized": centers,
        "reason": "all seven labels are directly legible",
    }))

    assert result["decision"] == "explicit_geometry"
    assert result["reference_video_id"] is None
    assert result["centers_normalized"] == centers


def test_parse_incomplete_geometry_fails_closed() -> None:
    result = parse_response(json.dumps({
        "decision": "explicit_geometry",
        "reference_video_id": None,
        "confidence": "high",
        "centers_normalized": {"grab": [0.2, 0.3]},
        "reason": "only one label is visible",
    }))

    assert result["decision"] == "unknown"
    assert result["centers_normalized"] is None
    assert result["parse_error"] == "incomplete_geometry"


def test_reference_match_requires_cross_order_agreement() -> None:
    first = {
        "decision": "same_normalized_layout",
        "reference_video_id": "0H53eYMpGsg",
        "confidence": "high",
        "centers_normalized": None,
        "reason": "same grid",
    }
    second = {**first, "confidence": "medium", "reason": "same centers"}

    result = combine_order_checks(first, second)

    assert result["decision"] == "same_normalized_layout"
    assert result["reference_video_id"] == "0H53eYMpGsg"
    assert result["confidence"] == "medium"
    assert result["order_consistent"] is True


def test_reference_order_disagreement_fails_closed() -> None:
    first = {
        "decision": "same_normalized_layout",
        "reference_video_id": "0H53eYMpGsg",
        "confidence": "high",
        "centers_normalized": None,
        "reason": "first position",
    }
    second = {
        **first,
        "reference_video_id": "wHrVwjK7dDw",
        "reason": "reversed first position",
    }

    result = combine_order_checks(first, second)

    assert result["decision"] == "unknown"
    assert result["reference_video_id"] is None
    assert result["parse_error"] == "order_inconsistent"
    assert result["order_consistent"] is False


def test_explicit_geometry_requires_cross_order_coordinate_stability() -> None:
    first_centers = {
        action: [0.1 + index * 0.1, 0.25]
        for index, action in enumerate(ACTIONS)
    }
    second_centers = {
        action: [center[0] + 0.01, center[1] - 0.01]
        for action, center in first_centers.items()
    }
    first = {
        "decision": "explicit_geometry",
        "reference_video_id": None,
        "confidence": "high",
        "centers_normalized": first_centers,
        "reason": "seven visible centers",
    }
    second = {
        **first,
        "centers_normalized": second_centers,
        "reason": "same seven centers after reordering",
    }

    result = combine_order_checks(first, second)

    assert result["decision"] == "explicit_geometry"
    assert result["order_consistent"] is True
    assert result["maximum_center_delta"] < 0.03
