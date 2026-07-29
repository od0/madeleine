from __future__ import annotations

import json

from harvest.verify_layout_pairs_vlm import combine_order_checks, parse_response


def test_pair_confirmation_requires_two_medium_or_better_same_decisions() -> None:
    first = {"decision": "same", "confidence": "high", "evidence": "same cells"}
    second = {"decision": "same", "confidence": "medium", "evidence": "same position"}

    result = combine_order_checks(first, second)

    assert result["pair_confirmed"] is True
    assert result["confidence"] == "medium"
    assert result["order_consistent"] is True


def test_low_confidence_same_fails_closed() -> None:
    first = {"decision": "same", "confidence": "high", "evidence": "same cells"}
    second = {"decision": "same", "confidence": "low", "evidence": "unclear"}

    result = combine_order_checks(first, second)

    assert result["pair_confirmed"] is False
    assert result["decision"] == "unknown"


def test_order_disagreement_fails_closed() -> None:
    first = {"decision": "same", "confidence": "high", "evidence": "same cells"}
    second = {"decision": "different", "confidence": "high", "evidence": "different scale"}

    result = combine_order_checks(first, second)

    assert result["pair_confirmed"] is False
    assert result["order_consistent"] is False


def test_parse_response_rejects_missing_evidence() -> None:
    result = parse_response(json.dumps({"decision": "same", "confidence": "high"}))

    assert result["decision"] == "unknown"
    assert result["confidence"] == "low"
    assert result["parse_error"] == "missing_specific_evidence"


def test_parse_response_rejects_schema_placeholder() -> None:
    result = parse_response(json.dumps({
        "decision": "same",
        "confidence": "high",
        "evidence": "short visible comparison",
    }))

    assert result["decision"] == "unknown"
    assert result["confidence"] == "low"
    assert result["parse_error"] == "missing_specific_evidence"
