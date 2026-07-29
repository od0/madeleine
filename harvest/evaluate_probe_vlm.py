"""Build the 86-frame human gold set and evaluate VLM probe triage.

The two legacy visual-label files contain ten transcription mistakes in video
IDs.  They are corrected through the explicit alias table below rather than a
fuzzy match.  The 48-row hand-label artifact takes precedence over the older
55-row style survey on their 17 overlapping videos.  This resolves two label
conflicts and yields 86 unique human-labeled frames.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ALIASES = {
    "3SCm-y__Zc-0": "3SCm-y_Zc-0",
    "BlmNABjKG08": "BImNABjKG08",
    "MaqrzKSF06A": "MaqrzKSFO6A",
    "TwW08kqStrE": "TWWO8kqStrE",
    "dQNwD3BCO__w": "dQNwD3BCO_w",
    "eIDsFg-S8YA": "elDsFg-S8YA",
    "odilYNqjL9Y": "odiIYNqjL9Y",
    "ofy37Fm6Egl": "ofy37Fm6EgI",
    "ubLdiTl1jJo": "ubLdiTI1jJo",
    "w-__-We2k_Vk": "w-_-We2k_Vk",
}

HAND_CLASS_MAP = {
    "keyboard_or_action_hud": "target_action_hud",
    "gamepad_display": "non_target",
    "none": "non_target",
    "non_gameplay": "uncertain",
}
CLASSES = ("target_action_hud", "non_target", "uncertain")


def canonical(video_id: str) -> str:
    return ALIASES.get(video_id, video_id)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_gold(
    hand_labels_path: Path,
    style_labels_path: Path,
    style_survey_path: Path,
) -> tuple[dict[str, str], dict]:
    hand_raw = json.loads(hand_labels_path.read_text())["labels"]
    hand = {canonical(video_id): HAND_CLASS_MAP[label]
            for video_id, label in hand_raw.items()}

    style_summary = json.loads(style_labels_path.read_text())
    successful_style_ids = {
        row["video_id"]
        for row in load_jsonl(style_survey_path)
        if row.get("error") is None
    }
    target_style_ids: set[str] = set()
    for family, video_ids in style_summary["has_input_overlay"].items():
        if family != "gamepad_display":
            target_style_ids.update(canonical(video_id) for video_id in video_ids)
    uncertain_style_ids = {
        canonical(video_id)
        for video_id in style_summary["unclassifiable_probe_hit_non_gameplay"]
    }
    style = {
        video_id: (
            "target_action_hud"
            if video_id in target_style_ids
            else "uncertain"
            if video_id in uncertain_style_ids
            else "non_target"
        )
        for video_id in successful_style_ids
    }
    conflicts = {
        video_id: {"style": style[video_id], "hand": hand[video_id]}
        for video_id in hand.keys() & style.keys()
        if style[video_id] != hand[video_id]
    }
    gold = dict(style)
    gold.update(hand)  # the dedicated hand-label pass is authoritative
    provenance = {
        "schema_version": 1,
        "classification_is_human_review": True,
        "human_gold_rows": len(gold),
        "hand_rows": len(hand),
        "style_rows": len(style),
        "overlap_rows": len(hand.keys() & style.keys()),
        "hand_precedence_conflicts": conflicts,
        "class_counts": dict(Counter(gold.values())),
        "explicit_video_id_aliases": ALIASES,
    }
    if len(gold) != 86:
        raise ValueError(f"expected 86 unique human labels, found {len(gold)}")
    return gold, provenance


def evaluate(
    gold: dict[str, str],
    predictions_path: Path,
    scan_path: Path | None = None,
    classical_uncertain_score: float = 0.0,
) -> dict:
    rows = load_jsonl(predictions_path)
    predictions = {row["video_id"]: row["class"] for row in rows}
    calibrated_ids: list[str] = []
    scan_rows = (
        {row["video_id"]: row for row in load_jsonl(scan_path)}
        if scan_path is not None
        else {}
    )
    if classical_uncertain_score > 0:
        for video_id, label in list(predictions.items()):
            score = scan_rows.get(video_id, {}).get("score")
            if (
                label == "non_target"
                and isinstance(score, (int, float))
                and score >= classical_uncertain_score
            ):
                predictions[video_id] = "uncertain"
                calibrated_ids.append(video_id)
    missing = sorted(set(gold) - set(predictions))
    extra = sorted(set(predictions) - set(gold))
    matrix = {
        actual: {predicted: 0 for predicted in CLASSES}
        for actual in CLASSES
    }
    for video_id, actual in gold.items():
        predicted = predictions.get(video_id)
        if predicted in CLASSES:
            matrix[actual][predicted] += 1
    evaluable_binary = [
        video_id for video_id, label in gold.items()
        if label != "uncertain" and video_id in predictions
    ]
    human_targets = [
        video_id for video_id in evaluable_binary
        if gold[video_id] == "target_action_hud"
    ]
    review_nominees = {
        video_id for video_id, label in predictions.items()
        if label in {"target_action_hud", "uncertain"}
    }
    target_recalled = sum(video_id in review_nominees for video_id in human_targets)
    exact = sum(predictions.get(video_id) == label for video_id, label in gold.items())
    successful_scan_gold = [
        video_id
        for video_id in gold
        if scan_rows.get(video_id, {}).get("error") is None
        and video_id in predictions
    ]
    successful_scan_targets = [
        video_id
        for video_id in successful_scan_gold
        if gold[video_id] == "target_action_hud"
    ]
    successful_scan_target_hits = [
        video_id
        for video_id in successful_scan_targets
        if video_id in review_nominees
    ]
    return {
        "schema_version": 1,
        "classification_is_human_review": False,
        "human_gold_rows": len(gold),
        "prediction_rows": len(predictions),
        "calibration": {
            "classical_uncertain_score": classical_uncertain_score,
            "rows_changed_non_target_to_uncertain": len(calibrated_ids),
            "changed_video_ids": sorted(calibrated_ids),
        },
        "missing_prediction_ids": missing,
        "extra_prediction_ids": extra,
        "confusion_matrix_actual_by_predicted": matrix,
        "exact_three_class_accuracy": exact / len(gold),
        "review_nomination": {
            "definition": "machine target_action_hud OR uncertain",
            "human_target_count": len(human_targets),
            "human_targets_nominated": target_recalled,
            "human_target_recall": (
                target_recalled / len(human_targets) if human_targets else None
            ),
            "human_targets_missed_as_non_target": sorted(
                video_id for video_id in human_targets
                if predictions.get(video_id) == "non_target"
            ),
            "nominees_in_gold_set": len(review_nominees & set(gold)),
        },
        "successful_initial_scan_cohort": {
            "human_gold_rows": len(successful_scan_gold),
            "human_target_count": len(successful_scan_targets),
            "review_nominees": len(review_nominees & set(successful_scan_gold)),
            "human_targets_nominated": len(successful_scan_target_hits),
            "human_target_recall": (
                len(successful_scan_target_hits) / len(successful_scan_targets)
                if successful_scan_targets
                else None
            ),
            "human_targets_missed": sorted(
                set(successful_scan_targets) - set(successful_scan_target_hits)
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hand-labels", type=Path, required=True)
    ap.add_argument("--style-labels", type=Path, required=True)
    ap.add_argument("--style-survey", type=Path, required=True)
    ap.add_argument("--gold-out", type=Path, required=True)
    ap.add_argument("--ids-out", type=Path, required=True)
    ap.add_argument("--predictions", type=Path)
    ap.add_argument("--evaluation-out", type=Path)
    ap.add_argument("--scan", type=Path)
    ap.add_argument("--classical-uncertain-score", type=float, default=0.0)
    args = ap.parse_args()

    gold, provenance = build_gold(
        args.hand_labels, args.style_labels, args.style_survey
    )
    args.gold_out.parent.mkdir(parents=True, exist_ok=True)
    gold_rows = [
        {"video_id": video_id, "human_class": gold[video_id]}
        for video_id in sorted(gold)
    ]
    args.gold_out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in gold_rows)
    )
    args.gold_out.with_suffix(args.gold_out.suffix + ".manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    args.ids_out.write_text("".join(video_id + "\n" for video_id in sorted(gold)))

    if args.predictions is not None:
        if args.evaluation_out is None:
            raise ValueError("--evaluation-out is required with --predictions")
        report = evaluate(
            gold,
            args.predictions,
            scan_path=args.scan,
            classical_uncertain_score=args.classical_uncertain_score,
        )
        args.evaluation_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
