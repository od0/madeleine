from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import harvest.classify_layout_surveys_vlm as layout_vlm
from harvest.classify_layout_surveys_vlm import (
    PROMPT_VERSION,
    load_existing_predictions,
    load_queue_rows,
    parse_response,
    validate_resume_manifest,
)


def test_parse_response_accepts_stable_active_candidate() -> None:
    parsed = parse_response(
        'prefix {"class":"stable_active_candidate","confidence":"high",'
        '"evidence":"bottom HUD changes from left to right"}'
    )
    assert parsed == {
        "class": "stable_active_candidate",
        "confidence": "high",
        "evidence": "bottom HUD changes from left to right",
    }


def test_parse_response_fails_closed_to_mechanical_scan() -> None:
    parsed = parse_response("not json")
    assert parsed["class"] == "stable_activity_uncertain"
    assert parsed["confidence"] == "low"
    assert parsed["parse_error"] == "missing_json_object"


def test_parse_response_rejects_unknown_class() -> None:
    parsed = parse_response(
        '{"class":"excellent","confidence":"certain","evidence":"looks good"}'
    )
    assert parsed["class"] == "stable_activity_uncertain"
    assert parsed["confidence"] == "low"
    assert "invalid_class" in parsed["parse_error"]


def queue_row(video_id: str = "video_1") -> dict:
    return {
        "video_id": video_id,
        "source": "youtube",
        "nominal_hours": 1.0,
        "human_reviewed": False,
        "training_admitted": False,
    }


def available_row(video_id: str = "video_1") -> dict:
    return {
        **queue_row(video_id),
        "survey_sha256": "a" * 64,
        "contact_sha256": "b" * 64,
        "survey_path": f"/{video_id}/survey.json",
        "contact_path": f"/{video_id}/contact-sheet.png",
    }


def prediction(video_id: str = "video_1") -> dict:
    return {
        "schema_version": 1,
        "video_id": video_id,
        "survey_sha256": "a" * 64,
        "contact_sha256": "b" * 64,
        "class": "stable_active_candidate",
        "confidence": "high",
        "full_scan_candidate": True,
        "human_reviewed": False,
        "training_admitted": False,
        "model": "model",
        "resolved_model_revision": "c" * 40,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": "d" * 64,
    }


def test_queue_validation_rejects_duplicate_and_unsafe_ids(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(queue_row("../escape")) + "\n")
    with pytest.raises(ValueError, match="unsafe"):
        load_queue_rows(queue)

    row = queue_row()
    queue.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_queue_rows(queue)


def test_prediction_resume_requires_full_identity_and_unique_ids(
    tmp_path: Path,
) -> None:
    output = tmp_path / "predictions.jsonl"
    record = prediction()
    output.write_text(json.dumps(record) + "\n")
    loaded = load_existing_predictions(
        output,
        [available_row()],
        model="model",
        resolved_model_revision="c" * 40,
        prompt_sha256="d" * 64,
    )
    assert loaded == [record]

    stale = dict(record, contact_sha256="e" * 64)
    output.write_text(json.dumps(stale) + "\n")
    with pytest.raises(ValueError, match="contact_sha256"):
        load_existing_predictions(
            output,
            [available_row()],
            model="model",
            resolved_model_revision="c" * 40,
            prompt_sha256="d" * 64,
        )

    output.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_existing_predictions(
            output,
            [available_row()],
            model="model",
            resolved_model_revision="c" * 40,
            prompt_sha256="d" * 64,
        )


def test_resume_manifest_binds_completed_prediction_prefix(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    first = (json.dumps(prediction()) + "\n").encode()
    second = (json.dumps(prediction("video_2")) + "\n").encode()
    output.write_bytes(first + second)
    identity = {
        "task": "celeste_layout_stability_nomination",
        "classification_is_human_review": False,
        "training_admission": False,
        "queue_sha256": "e" * 64,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": "d" * 64,
        "model": "model",
        "resolved_model_revision": "c" * 40,
        "max_new_tokens": 160,
    }
    manifest = {
        "schema_version": 2,
        **identity,
        "created_at": "2026-07-27T00:00:00+00:00",
        "predictions": {
            "rows": 1,
            "unique_video_ids": 1,
            "sha256": hashlib.sha256(first).hexdigest(),
        },
    }
    manifest_path = tmp_path / "predictions.jsonl.manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assert validate_resume_manifest(manifest_path, identity, output, 2) == manifest[
        "created_at"
    ]

    legacy = dict(manifest)
    legacy["schema_version"] = 1
    legacy.pop("predictions")
    legacy.pop("max_new_tokens")
    manifest_path.write_text(json.dumps(legacy))
    assert validate_resume_manifest(manifest_path, identity, output, 2) == legacy[
        "created_at"
    ]

    manifest_path.write_text(json.dumps(manifest))
    output.write_bytes(b"{}\n" + second)
    with pytest.raises(ValueError, match="prefix differs"):
        validate_resume_manifest(manifest_path, identity, output, 2)


def test_main_skips_model_weight_load_when_no_surveys_are_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text("{}\n")
    survey_root = tmp_path / "surveys"
    output = tmp_path / "predictions.jsonl"
    fake_transformers = SimpleNamespace(
        __version__="test",
        AutoConfig=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: SimpleNamespace(
                _commit_hash="c" * 40
            )
        ),
    )
    fake_torch = SimpleNamespace(__version__="test")
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(layout_vlm, "available_surveys", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify_layout_surveys_vlm",
            "--queue",
            str(queue),
            "--survey-root",
            str(survey_root),
            "--out",
            str(output),
            "--available-only",
        ],
    )
    layout_vlm.main()
    manifest = json.loads(
        output.with_suffix(".jsonl.manifest.json").read_text()
    )
    assert manifest["predictions"] == {
        "rows": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
        "unique_video_ids": 0,
    }
    assert manifest["state"] == "complete"
