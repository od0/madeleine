from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from harvest.classify_probe_frames_vlm import parse_response
from harvest.evaluate_probe_vlm import build_gold, evaluate
from harvest.index_probe_campaign import main as index_campaign_main


def test_vlm_parser_fails_closed_on_malformed_or_unknown_output() -> None:
    assert parse_response("not json")["class"] == "uncertain"
    parsed = parse_response(
        '{"class":"maybe","confidence":"certain","evidence":""}'
    )
    assert parsed["class"] == "uncertain"
    assert parsed["confidence"] == "low"
    assert "parse_error" in parsed


@pytest.mark.requires_private_artifacts(
    "results/wild/hand_labels.json",
    "results/wild/style_labels.json",
    "results/wild/style_survey.jsonl",
)
def test_human_visual_gold_union_has_expected_counts() -> None:
    root = Path(__file__).parents[1]
    gold, provenance = build_gold(
        root / "results/wild/hand_labels.json",
        root / "results/wild/style_labels.json",
        root / "results/wild/style_survey.jsonl",
    )
    assert len(gold) == 86
    assert provenance["class_counts"] == {
        "non_target": 68,
        "target_action_hud": 12,
        "uncertain": 6,
    }
    assert len(provenance["hand_precedence_conflicts"]) == 2


def test_classical_calibration_converts_only_high_score_negatives(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"video_id": "positive", "class": "target_action_hud"}) + "\n"
        + json.dumps({"video_id": "high", "class": "non_target"}) + "\n"
        + json.dumps({"video_id": "low", "class": "non_target"}) + "\n"
    )
    scan = tmp_path / "scan.jsonl"
    scan.write_text(
        json.dumps({"video_id": "positive", "score": 1, "error": None}) + "\n"
        + json.dumps({"video_id": "high", "score": 16, "error": None}) + "\n"
        + json.dumps({"video_id": "low", "score": 15.999, "error": None}) + "\n"
    )
    gold = {
        "positive": "target_action_hud",
        "high": "target_action_hud",
        "low": "non_target",
    }
    report = evaluate(gold, predictions, scan, classical_uncertain_score=16)
    assert report["review_nomination"]["human_target_recall"] == 1.0
    assert report["calibration"]["changed_video_ids"] == ["high"]


def test_campaign_index_is_completion_gated_and_deduplicated(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "campaign"

    def make_attempt(worker: str, video_id: str, status: str) -> None:
        attempt = root / worker / video_id
        (attempt / "frames").mkdir(parents=True)
        (attempt / "crops").mkdir()
        frame = attempt / "frames" / f"{video_id}.png"
        frame.write_bytes(b"frame")
        probe = attempt / "probe.json"
        probe.write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "url": f"https://example.invalid/{video_id}",
                    "error": None if status == "ok" else "failed",
                    "crops": [],
                }
            )
        )
        marker = {
            "video_id": video_id,
            "status": status,
            "objects": [],
        }
        (attempt / "probe_complete.json").write_text(json.dumps(marker))

    make_attempt("worker-b", "same", "ok")
    make_attempt("worker-a", "same", "ok")
    make_attempt("worker-a", "failed", "error")
    out = tmp_path / "index.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "index_probe_campaign.py",
            "--campaign-root",
            str(root),
            "--campaign-id",
            "test-campaign",
            "--out",
            str(out),
        ],
    )
    index_campaign_main()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["attempt_path"] == "worker-a/same"
    assert rows[0]["duplicate_attempt_paths"] == ["worker-b/same"]
    manifest = json.loads(out.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["unique_successful_videos"] == 1
    assert manifest["duplicate_successful_attempts"] == 1
    assert len(manifest["rejected_attempts"]) == 1
