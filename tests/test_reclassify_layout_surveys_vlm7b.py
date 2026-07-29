"""Contract tests for the crop-first Qwen 7B survey reclassifier.

These tests exercise the pure validation surfaces: strict response parsing,
review-candidate consistency, evidence tiling, queue loading, and resume
manifest binding. No model, GPU, or PIL image is required.
"""

import hashlib
import json
from pathlib import Path

import pytest

from harvest.reclassify_layout_surveys_vlm7b import (
    ALLOWED_CLASSES,
    EvidenceBundle,
    PROMPT_TEMPLATE,
    PROMPT_VERSION,
    RECORD_SCHEMA_VERSION,
    REVIEW_CANDIDATE_CLASSES,
    _sha256_line_prefix,
    choose_evidence_frames,
    load_queue_rows,
    parse_response,
    quadrant_boxes,
    sha256_text,
    validate_record_identity,
    validate_resume_manifest,
)


VALID_RESPONSE = {
    "class": "decodable_input_hud",
    "modality": "controller",
    "confidence": "high",
    "evidence": "D-pad Left fills in samples 3 and 9 while button A stays hollow",
    "location": "bottom_left",
    "physical_control_labels": ["D-pad Left", "button A"],
    "changing_controls": ["D-pad Left"],
    "layout_description": "grey gamepad diagram with d-pad left of two face buttons",
    "evidence_sample_orders": [3, 9],
    "non_decodable_reason": "none",
}


def response(**overrides):
    value = {**VALID_RESPONSE, **overrides}
    return json.dumps(value)


class TestParseResponse:
    def test_valid_controller_positive(self):
        parsed = parse_response(response(), frozenset(range(16)))
        assert parsed["class"] == "decodable_input_hud"
        assert parsed["modality"] == "controller"
        assert parsed["validation_errors"] == []

    def test_obsolete_keyboard_action_hud_class_fails_closed(self):
        parsed = parse_response(
            response(**{"class": "keyboard_action_hud"}), frozenset(range(16))
        )
        assert parsed["class"] == "uncertain"
        assert "invalid_class" in parsed["validation_errors"]

    def test_valid_negative(self):
        parsed = parse_response(
            response(
                **{
                    "class": "non_decodable",
                    "modality": "none",
                    "location": "none",
                    "physical_control_labels": [],
                    "changing_controls": [],
                    "evidence_sample_orders": [],
                    "non_decodable_reason": "no_input_hud",
                    "evidence": "only a timer panel and gameplay are visible",
                }
            ),
            frozenset(range(16)),
        )
        assert parsed["class"] == "non_decodable"
        assert parsed["validation_errors"] == []

    def test_decodable_without_changing_controls_fails_closed(self):
        parsed = parse_response(
            response(changing_controls=[]), frozenset(range(16))
        )
        assert parsed["class"] == "uncertain"
        assert (
            "decodable_hud_lacks_visible_changing_controls"
            in parsed["validation_errors"]
        )
        assert parsed["model_claimed_class"] == "decodable_input_hud"

    def test_decodable_with_single_sample_fails_closed(self):
        parsed = parse_response(
            response(evidence_sample_orders=[3]), frozenset(range(16))
        )
        assert "decodable_hud_lacks_cross_frame_evidence" in parsed["validation_errors"]

    def test_non_decodable_claiming_changes_fails_closed(self):
        parsed = parse_response(
            response(
                **{
                    "class": "non_decodable",
                    "non_decodable_reason": "static_or_frozen",
                }
            ),
            frozenset(range(16)),
        )
        assert "non_decodable_claims_changing_controls" in parsed["validation_errors"]

    def test_unknown_sample_orders_fail_closed(self):
        parsed = parse_response(
            response(evidence_sample_orders=[3, 99]), frozenset(range(16))
        )
        assert "invalid_evidence_sample_orders" in parsed["validation_errors"]

    def test_extraneous_text_fails_closed(self):
        parsed = parse_response(
            response() + "\nSure, here is the JSON!", frozenset(range(16))
        )
        assert parsed["class"] == "uncertain"
        assert "extraneous_text_after_json" in parsed["validation_errors"]

    def test_non_json_fails_closed(self):
        parsed = parse_response("I could not find a HUD.", frozenset(range(16)))
        assert parsed["class"] == "uncertain"
        assert "response_not_bare_json_object" in parsed["validation_errors"]

    def test_extra_keys_fail_closed(self):
        raw = json.dumps({**VALID_RESPONSE, "celeste_binding": "jump"})
        parsed = parse_response(raw, frozenset(range(16)))
        assert "extra_keys:celeste_binding" in parsed["validation_errors"]


class TestReviewCandidateContract:
    def test_candidate_classes_use_current_schema(self):
        assert REVIEW_CANDIDATE_CLASSES == {"decodable_input_hud", "uncertain"}
        assert REVIEW_CANDIDATE_CLASSES < ALLOWED_CLASSES

    def test_obsolete_class_absent_from_module(self):
        source = Path("harvest/reclassify_layout_surveys_vlm7b.py").read_text()
        assert "keyboard_action_hud" not in source

    def _record_and_context(self, label, review_candidate):
        row = {
            "video_id": "vidA",
            "survey_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "contact_sha256": "c" * 64,
        }
        bundle = EvidenceBundle(
            images=[],
            descriptors=[{"position": 1, "kind": "contact_sheet"}],
            prompt="p",
            prompt_sha256=sha256_text("p"),
            image_bundle_sha256="d" * 64,
            all_sample_orders=frozenset(range(16)),
        )
        record = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "video_id": "vidA",
            "queue_sha256": "e" * 64,
            "survey_sha256": row["survey_sha256"],
            "source_sha256": row["source_sha256"],
            "contact_sha256": row["contact_sha256"],
            "image_bundle_sha256": bundle.image_bundle_sha256,
            "input_images": bundle.descriptors,
            "model": "m",
            "resolved_model_revision": "f" * 40,
            "prompt_version": PROMPT_VERSION,
            "prompt_template_sha256": sha256_text(PROMPT_TEMPLATE),
            "prompt_sha256": bundle.prompt_sha256,
            "evidence_spec_version": "crop-first-overlap-quadrants-v1",
            "max_new_tokens": 320,
            "human_reviewed": False,
            "training_admitted": False,
            "class": label,
            "confidence": "high",
            "review_candidate": review_candidate,
            "retry_count": 0,
        }
        return record, row, bundle

    @pytest.mark.parametrize(
        "label,candidate",
        [
            ("decodable_input_hud", True),
            ("uncertain", True),
            ("non_decodable", False),
        ],
    )
    def test_consistent_candidacy_accepted(self, label, candidate):
        record, row, bundle = self._record_and_context(label, candidate)
        validate_record_identity(
            record,
            row,
            bundle,
            queue_sha256="e" * 64,
            model="m",
            resolved_model_revision="f" * 40,
            max_new_tokens=320,
        )

    @pytest.mark.parametrize(
        "label,candidate",
        [
            ("decodable_input_hud", False),
            ("uncertain", False),
            ("non_decodable", True),
        ],
    )
    def test_inconsistent_candidacy_rejected(self, label, candidate):
        record, row, bundle = self._record_and_context(label, candidate)
        with pytest.raises(ValueError, match="review candidacy"):
            validate_record_identity(
                record,
                row,
                bundle,
                queue_sha256="e" * 64,
                model="m",
                resolved_model_revision="f" * 40,
                max_new_tokens=320,
            )

    @pytest.mark.parametrize("retry_count", [0, 1])
    def test_valid_retry_counts_accepted(self, retry_count):
        record, row, bundle = self._record_and_context("uncertain", True)
        record["retry_count"] = retry_count
        if retry_count == 1:
            record["raw_response_retry"] = '{"class":"uncertain"}'
        validate_record_identity(
            record,
            row,
            bundle,
            queue_sha256="e" * 64,
            model="m",
            resolved_model_revision="f" * 40,
            max_new_tokens=320,
        )

    def test_retry_count_two_rejected(self):
        record, row, bundle = self._record_and_context("uncertain", True)
        record["retry_count"] = 2
        with pytest.raises(ValueError, match="retry_count"):
            validate_record_identity(
                record,
                row,
                bundle,
                queue_sha256="e" * 64,
                model="m",
                resolved_model_revision="f" * 40,
                max_new_tokens=320,
            )


class TestPromptTemplate:
    def test_placeholder_substitution_survives_json_braces(self):
        # The template contains literal JSON braces; building the prompt must
        # not treat them as format fields.
        evidence_map = json.dumps({"position": 1, "kind": "tile"})
        prompt = PROMPT_TEMPLATE.replace("{evidence_map}", evidence_map)
        assert evidence_map in prompt
        assert "{evidence_map}" not in prompt
        assert '"class":"decodable_input_hud|non_decodable|uncertain"' in prompt


class TestEvidenceGeometry:
    def test_quadrants_cover_frame_with_overlap(self):
        width, height = 640, 360
        boxes = quadrant_boxes(width, height)
        assert set(boxes) == {"top_left", "top_right", "bottom_left", "bottom_right"}
        for x0, y0, x1, y1 in boxes.values():
            assert 0 <= x0 < x1 <= width
            assert 0 <= y0 < y1 <= height
        # The union covers the frame and adjacent quadrants overlap.
        assert boxes["top_left"][2] > boxes["top_right"][0]
        assert boxes["top_left"][3] > boxes["bottom_left"][1]

    def test_tiny_frame_rejected(self):
        with pytest.raises(ValueError):
            quadrant_boxes(1, 1)

    def test_choose_evidence_frames_deterministic_span(self):
        frames = [{"sample_order": index} for index in range(16)]
        chosen = choose_evidence_frames(frames, 4)
        assert [frame["sample_order"] for frame in chosen] == [0, 5, 10, 15]

    def test_choose_evidence_frames_small_survey(self):
        frames = [{"sample_order": index} for index in range(3)]
        assert choose_evidence_frames(frames, 4) == frames

    def test_choose_evidence_frames_minimum(self):
        with pytest.raises(ValueError):
            choose_evidence_frames([{"sample_order": 0}], 1)


class TestQueueLoading:
    def _write_queue(self, tmp_path, rows):
        path = tmp_path / "queue.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_valid_queue(self, tmp_path):
        rows = [
            {"video_id": "abc-123", "human_reviewed": False},
            {"video_id": "v456"},
        ]
        assert len(load_queue_rows(self._write_queue(tmp_path, rows))) == 2

    def test_duplicate_video_id_rejected(self, tmp_path):
        rows = [{"video_id": "abc"}, {"video_id": "abc"}]
        with pytest.raises(ValueError, match="duplicate"):
            load_queue_rows(self._write_queue(tmp_path, rows))

    def test_human_review_claim_rejected(self, tmp_path):
        rows = [{"video_id": "abc", "human_reviewed": True}]
        with pytest.raises(ValueError, match="human review"):
            load_queue_rows(self._write_queue(tmp_path, rows))

    def test_admission_claim_rejected(self, tmp_path):
        rows = [{"video_id": "abc", "training_admitted": True}]
        with pytest.raises(ValueError, match="admission"):
            load_queue_rows(self._write_queue(tmp_path, rows))

    def test_unsafe_video_id_rejected(self, tmp_path):
        rows = [{"video_id": "../escape"}]
        with pytest.raises(ValueError, match="unsafe"):
            load_queue_rows(self._write_queue(tmp_path, rows))


class TestResumeBinding:
    def test_prefix_hash_matches_line_boundaries(self, tmp_path):
        path = tmp_path / "predictions.jsonl"
        lines = [b'{"video_id":"a"}\n', b'{"video_id":"b"}\n']
        path.write_bytes(b"".join(lines))
        expected = hashlib.sha256(lines[0]).hexdigest()
        assert _sha256_line_prefix(path, 1) == expected
        with pytest.raises(ValueError, match="shorter"):
            _sha256_line_prefix(path, 3)

    def test_missing_manifest_with_rows_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="without its manifest"):
            validate_resume_manifest(
                tmp_path / "absent.manifest.json", {}, tmp_path / "out.jsonl", 5
            )

    def test_stale_identity_rejected(self, tmp_path):
        manifest_path = tmp_path / "out.manifest.json"
        manifest_path.write_text(json.dumps({"queue_sha256": "old"}))
        with pytest.raises(ValueError, match="stale or mismatched queue_sha256"):
            validate_resume_manifest(
                manifest_path,
                {"queue_sha256": "new"},
                tmp_path / "out.jsonl",
                0,
            )

    def test_valid_resume(self, tmp_path):
        out = tmp_path / "out.jsonl"
        line = b'{"video_id":"a"}\n'
        out.write_bytes(line)
        manifest_path = tmp_path / "out.manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "queue_sha256": "q",
                    "created_at": "2026-07-28T00:00:00+00:00",
                    "predictions": {
                        "rows": 1,
                        "sha256": hashlib.sha256(line).hexdigest(),
                    },
                }
            )
        )
        created = validate_resume_manifest(
            manifest_path, {"queue_sha256": "q"}, out, 1
        )
        assert created == "2026-07-28T00:00:00+00:00"
