"""Reviewed wall-clock and gameplay-range contract for wild videos.

Speedrun.com durations are loadless game time, so they cannot select a video
interval.  This separate artifact preserves immutable raw fetch evidence while
recording evidence-reviewed wall-clock boundaries and either explicit allowed
gameplay ranges or explicit exclusions.  Reviewer identity and kind are
explicit; only a human or human-with-AI-assistance review can admit training
data.  All times are relative to the first decoded video PTS.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np


BOUNDARIES_VERSION = "madeleine.wild-boundaries.v2"
LEGACY_BOUNDARIES_VERSION = "madeleine.wild-boundaries.v1"
REVIEWER_KINDS = ("human", "human_with_ai_assistance", "ai_agent")
HUMAN_REVIEWER_KINDS = frozenset(("human", "human_with_ai_assistance"))
Range = tuple[float, float]
PolicyMode = Literal["allowed_ranges", "excluded_ranges"]


def _range(value: Any, field: str) -> Range:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must be [start_s, end_s]")
    start, end = float(value[0]), float(value[1])
    if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
        raise ValueError(f"{field} must be a finite positive-duration range")
    return start, end


def _validated_ranges(values: Any, field: str, wall: Range, allow_empty: bool) -> tuple[Range, ...]:
    if not isinstance(values, list) or (not values and not allow_empty):
        suffix = " (an empty reviewed exclusion list is allowed)" if allow_empty else ""
        raise ValueError(f"{field} must be a list of ranges{suffix}")
    ranges = tuple(_range(value, f"{field}[{i}]") for i, value in enumerate(values))
    prior_end = wall[0]
    for index, (start, end) in enumerate(ranges):
        if start < wall[0] or end > wall[1]:
            raise ValueError(f"{field}[{index}] lies outside wall_clock_range_s")
        if start < prior_end:
            raise ValueError(f"{field} must be sorted and non-overlapping")
        prior_end = end
    return ranges


@dataclass(frozen=True)
class WildBoundaries:
    video_id: str
    source_sha256: str
    wall_clock_range_s: Range
    policy_mode: PolicyMode
    ranges_s: tuple[Range, ...]
    human_reviewed: bool
    reviewer: str
    reviewer_kind: str
    evidence: tuple[str, ...]
    notes: str
    format_version: str = BOUNDARIES_VERSION

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WildBoundaries":
        if raw.get("format_version") == LEGACY_BOUNDARIES_VERSION:
            raise ValueError(
                "legacy wild-boundaries.v1 lacks reviewer_kind provenance; "
                "recreate it explicitly as v2"
            )
        if raw.get("format_version") != BOUNDARIES_VERSION:
            raise ValueError(f"format_version must equal {BOUNDARIES_VERSION!r}")
        video_id = str(raw.get("video_id", "")).strip()
        source_hash = str(raw.get("source_sha256", "")).strip().lower()
        if not video_id or len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
            raise ValueError("video_id and a lowercase SHA-256 source hash are required")
        wall = _range(raw.get("wall_clock_range_s"), "wall_clock_range_s")
        has_allowed = "allowed_ranges_s" in raw
        has_excluded = "excluded_ranges_s" in raw
        if has_allowed == has_excluded:
            raise ValueError("provide exactly one of allowed_ranges_s or excluded_ranges_s")
        if has_allowed:
            mode: PolicyMode = "allowed_ranges"
            ranges = _validated_ranges(raw["allowed_ranges_s"], "allowed_ranges_s", wall, False)
        else:
            mode = "excluded_ranges"
            ranges = _validated_ranges(raw["excluded_ranges_s"], "excluded_ranges_s", wall, True)
        reviewer = str(raw.get("reviewer", "")).strip()
        reviewer_kind = str(raw.get("reviewer_kind", "")).strip()
        if not reviewer:
            raise ValueError("reviewer must be named")
        if reviewer_kind not in REVIEWER_KINDS:
            raise ValueError(f"reviewer_kind must be one of {REVIEWER_KINDS}")
        human_reviewed = reviewer_kind in HUMAN_REVIEWER_KINDS
        if raw.get("human_reviewed") is not human_reviewed:
            raise ValueError(
                "human_reviewed must be derived from reviewer_kind; AI-agent "
                "review cannot satisfy the human gate"
            )
        evidence_raw = raw.get("evidence", [])
        if not isinstance(evidence_raw, list):
            raise ValueError("evidence must be a list of artifact references/notes")
        return cls(
            video_id=video_id,
            source_sha256=source_hash,
            wall_clock_range_s=wall,
            policy_mode=mode,
            ranges_s=ranges,
            human_reviewed=human_reviewed,
            reviewer=reviewer,
            reviewer_kind=reviewer_kind,
            evidence=tuple(str(value) for value in evidence_raw),
            notes=str(raw.get("notes", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "WildBoundaries":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "format_version": self.format_version,
            "video_id": self.video_id,
            "source_sha256": self.source_sha256,
            "timeline": "seconds_relative_to_first_decoded_video_pts",
            "wall_clock_range_s": list(self.wall_clock_range_s),
            "human_reviewed": self.human_reviewed,
            "reviewer": self.reviewer,
            "reviewer_kind": self.reviewer_kind,
            "evidence": list(self.evidence),
            "notes": self.notes,
        }
        result[f"{self.policy_mode}_s"] = [list(value) for value in self.ranges_s]
        return result

    def gameplay_mask(self, pts_s: np.ndarray) -> np.ndarray:
        if pts_s.ndim != 1:
            raise ValueError("PTS must be one-dimensional")
        start, end = self.wall_clock_range_s
        base = (pts_s >= start) & (pts_s < end)
        covered = np.zeros(pts_s.shape, dtype=bool)
        for range_start, range_end in self.ranges_s:
            covered |= (pts_s >= range_start) & (pts_s < range_end)
        return base & covered if self.policy_mode == "allowed_ranges" else base & ~covered


def _parse_cli_range(value: str) -> Range:
    try:
        start, end = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("range must be START:END seconds") from exc
    try:
        return _range([float(start), float(end)], "range")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--end-s", type=float, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--allow", action="append", type=_parse_cli_range)
    group.add_argument("--exclude", action="append", type=_parse_cli_range)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewer-kind", choices=REVIEWER_KINDS, required=True)
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--notes", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw: dict[str, Any] = {
        "format_version": BOUNDARIES_VERSION,
        "video_id": args.video_id,
        "source_sha256": args.source_sha256,
        "wall_clock_range_s": [args.start_s, args.end_s],
        "human_reviewed": args.reviewer_kind in HUMAN_REVIEWER_KINDS,
        "reviewer": args.reviewer,
        "reviewer_kind": args.reviewer_kind,
        "evidence": args.evidence,
        "notes": args.notes,
    }
    raw["allowed_ranges_s" if args.allow is not None else "excluded_ranges_s"] = (
        args.allow if args.allow is not None else args.exclude
    )
    boundaries = WildBoundaries.from_dict(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(boundaries.to_dict(), indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
