"""Append raw VLM target nominations to an auditable fetch-review queue.

Calibrated ``uncertain`` rows are intentionally excluded. Every emitted row is
still machine nomination only; this queue prepares work and does not authorize
fetch, review, decoding, or corpus admission by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def collect_video_ids(value) -> set[str]:
    if isinstance(value, dict):
        ids = {str(value["video_id"])} if "video_id" in value else set()
        for child in value.values():
            ids.update(collect_video_ids(child))
        return ids
    if isinstance(value, list):
        ids: set[str] = set()
        for child in value:
            ids.update(collect_video_ids(child))
        return ids
    return set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, action="append", required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--tranche", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    candidates = {row["video_id"]: row for row in load_jsonl(args.candidates)}
    tranche_ids = (
        collect_video_ids(json.loads(args.tranche.read_text()))
        if args.tranche is not None
        else set()
    )
    predictions: dict[str, tuple[dict, Path]] = {}
    for path in args.predictions:
        for row in load_jsonl(path):
            video_id = row["video_id"]
            prior = predictions.get(video_id)
            raw_class = row.get("vlm_class_before_calibration", row["class"])
            if prior is not None:
                prior_class = prior[0].get(
                    "vlm_class_before_calibration", prior[0]["class"]
                )
                if prior_class != raw_class:
                    raise ValueError(
                        f"conflicting raw classes for {video_id}: "
                        f"{prior_class} vs {raw_class}"
                    )
                continue
            predictions[video_id] = (row, path)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = load_jsonl(args.out) if args.out.exists() else []
    existing_ids = {row["video_id"] for row in existing_rows}
    new_rows = []
    for video_id in sorted(predictions):
        prediction, source_path = predictions[video_id]
        raw_class = prediction.get(
            "vlm_class_before_calibration", prediction["class"]
        )
        if raw_class != "target_action_hud" or video_id in existing_ids:
            continue
        candidate = candidates.get(video_id, {})
        duration_s = candidate.get("duration_s")
        new_rows.append({
            "schema_version": 1,
            "video_id": video_id,
            "url": candidate.get("url", prediction.get("url")),
            "source": candidate.get("source"),
            "category": candidate.get("category"),
            "place": candidate.get("place"),
            "duration_s": duration_s,
            "nominal_hours": duration_s / 3600 if duration_s is not None else None,
            "raw_vlm_class": raw_class,
            "calibrated_class": prediction["class"],
            "confidence": prediction.get("confidence"),
            "evidence": prediction.get("evidence"),
            "model": prediction.get("model"),
            "resolved_model_revision": prediction.get("resolved_model_revision"),
            "prompt_version": prediction.get("prompt_version"),
            "prompt_sha256": prediction.get("prompt_sha256"),
            "input_campaign_id": prediction.get("input_campaign_id"),
            "input_worker_id": prediction.get("input_worker_id"),
            "input_attempt_path": prediction.get("input_attempt_path"),
            "prediction_artifact": source_path.name,
            "already_in_frozen_wild20_tranche": video_id in tranche_ids,
            "machine_nomination_only": True,
            "human_reviewed": False,
            "fetch_authorized_by_this_record": False,
            "queued_at": datetime.now(timezone.utc).isoformat(),
        })
    with args.out.open("a") as fh:
        for row in new_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()

    all_rows = existing_rows + new_rows
    manifest = {
        "schema_version": 1,
        "queue_semantics": "raw_vlm_target_machine_nominations_only",
        "human_reviewed": False,
        "fetch_authorized_by_queue": False,
        "rows": len(all_rows),
        "new_rows_this_run": len(new_rows),
        "nominal_hours_known": sum(
            row["nominal_hours"] for row in all_rows
            if row.get("nominal_hours") is not None
        ),
        "already_in_frozen_wild20_tranche": sum(
            bool(row.get("already_in_frozen_wild20_tranche")) for row in all_rows
        ),
        "queue_sha256": sha256(args.out),
        "prediction_inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in args.predictions
        ],
        "candidates": {
            "path": str(args.candidates.resolve()),
            "sha256": sha256(args.candidates),
        },
    }
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
