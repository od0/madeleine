#!/usr/bin/env python3
"""Replay mapper-v3 bind inference over published v2 mapping reports.

The v3 inference in :mod:`nitrogen.map_actions` consumes only per-button
evidence statistics, so its behavior on the full corpus can be validated
offline against the published v2 reports without touching raw chunks.  The
v2 evidence blobs lack the ``presses_per_hour`` statistic v3 adds, so it is
reconstructed here from press counts and per-video hours.

For every video this tool records the v2 bind map and flag, the v3
per-action assignment, confidence, flag, and fallback, and emits a summary:
per-action flag counts, fully-unflagged counts, dash reassignments for the
audit's dash-starved cohort, and every previously-unflagged video whose
inferred buttons changed.  No published artifact is modified; adopting v3
for a corpus remap remains a separately authorized step.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from nitrogen import map_actions

ACTIONS = ("jump", "dash", "grab")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-reports-root", type=Path, required=True,
                        help="tree containing <video_id>/mapping_report.json")
    parser.add_argument("--per-video-mapping-audit", type=Path, required=True,
                        help="audit JSON supplying per-video hours and the dash-starved list")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    audit = json.loads(args.per_video_mapping_audit.read_text())
    hours = {record["video_id"]: float(record["hours"]) for record in audit["records"]}
    starved = set(audit.get("dash_starved_unflagged_videos", []))

    records = []
    for report_path in sorted(args.v2_reports_root.glob("*/mapping_report.json")):
        report = json.loads(report_path.read_text())
        video_id = report["video_id"]
        evidence = {button: dict(stats) for button, stats in report["evidence"].items()}
        video_hours = hours[video_id]
        for stats in evidence.values():
            stats["presses_per_hour"] = (
                int(stats["press_count"]) / video_hours if video_hours > 0 else 0.0
            )
        bind_v3, confidence, flagged, _, per_action = map_actions.infer_bind_map(evidence)
        records.append(
            {
                "video_id": video_id,
                "hours": video_hours,
                "dash_starved_in_v2": video_id in starved,
                "v2": {"bind_map": report["bind_map"], "flagged": report["flagged"],
                        "confidence": report["confidence"]},
                "v3": {
                    "bind_map": bind_v3,
                    "flagged": flagged,
                    "confidence": confidence,
                    "per_action": {
                        action: {
                            "inferred_button": per_action[action]["inferred_button"],
                            "selected": per_action[action]["selected"],
                            "confidence": per_action[action]["confidence"],
                            "flagged": per_action[action]["flagged"],
                            "fallback_used": per_action[action]["fallback_used"],
                            "reason": per_action[action].get("reason"),
                        }
                        for action in ACTIONS
                    },
                },
            }
        )

    flag_counts = Counter()
    for record in records:
        for action in ACTIONS:
            if record["v3"]["per_action"][action]["flagged"]:
                flag_counts[action] += 1
    changed_unflagged = [
        {
            "video_id": record["video_id"],
            "action": action,
            "v2": record["v2"]["bind_map"][action],
            "v3": record["v3"]["per_action"][action]["selected"],
        }
        for record in records
        if not record["v2"]["flagged"]
        for action in ACTIONS
        if record["v3"]["per_action"][action]["selected"] != record["v2"]["bind_map"][action]
    ]
    summary = {
        "schema_version": "madeleine.nitrogen-mapper-v3-offline-replay.v1",
        "mapper_tool_version": map_actions.TOOL_VERSION,
        "videos": len(records),
        "v3_per_action_flag_counts": dict(flag_counts),
        "v3_fully_unflagged_videos": sum(
            1 for record in records if not record["v3"]["flagged"]
        ),
        "v2_unflagged_videos": sum(1 for record in records if not record["v2"]["flagged"]),
        "dash_starved_resolution": [
            {
                "video_id": record["video_id"],
                "v2_dash": record["v2"]["bind_map"]["dash"],
                "v3_dash": record["v3"]["per_action"]["dash"]["selected"],
                "v3_dash_flagged": record["v3"]["per_action"]["dash"]["flagged"],
                "v3_dash_confidence": record["v3"]["per_action"]["dash"]["confidence"],
            }
            for record in records
            if record["dash_starved_in_v2"]
        ],
        "changed_selections_on_v2_unflagged_videos": changed_unflagged,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    compact = {k: summary[k] for k in (
        "videos", "v3_per_action_flag_counts", "v3_fully_unflagged_videos", "v2_unflagged_videos"
    )}
    print(json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
