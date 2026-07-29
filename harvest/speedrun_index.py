"""Enumerate speedrun.com Celeste runs into a fetch-candidate list.

The leaderboards link roughly 6,400 PC runs across categories. This module
pulls them through the public API, extracts the linked video ids, and applies
the two filters that must happen BEFORE anything is downloaded:

1. **PC only.** Console runs use a gamepad, so they carry no keyboard overlay
   and belong to NitroGen's channel, not this one.
2. **Not already in the NitroGen slice.** A video present in both corpora would
   be counted twice in any mix, and would silently destroy the premise of the
   matched-hours experiment — which compares two INDEPENDENT label channels.
   Overlap is checked against the Celeste chunk index by video id and reported
   explicitly rather than dropped in silence, because an overlap is itself a
   finding (the same run, labelled two ways, is a direct label-quality probe).

Ranking is retained per run, because run length is inversely related to rank:
world-record Any% runs are ~25 minutes while lower-ranked runs reach 2-3 hours,
and slower play also sits closer to our own recorded distribution (the
novice-expert gap measured 4-15x input density against expert footage). Both
effects favour sampling down the leaderboard, not just off the top.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://www.speedrun.com/api/v1"
GAME_ID = "o1y9j9v6"

# Full-game categories worth harvesting, longest routes first.
CATEGORIES = {
    "100%": "xk9ry6xk",
    "All Red Berries": "jdz8oprd",
    "All Chapters": "z27rpe5d",
    "True Ending": "zdn0m372",
    "All Cassettes": "q2517ggd",
    "All Hearts": "xd1718wd",
    "Any%": "7kjpl1gk",
}

YOUTUBE_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{11})"
)
TWITCH_RE = re.compile(r"twitch\.tv/videos/(\d+)")


def _get(url: str, retries: int = 3) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": "madeleine-research/0.1"}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def parse_video_ids(uri: str) -> tuple[str, str] | None:
    """Return (source, video_id) for a supported host, else None."""

    match = YOUTUBE_RE.search(uri)
    if match:
        return "youtube", match.group(1)
    match = TWITCH_RE.search(uri)
    if match:
        return "twitch", f"v{match.group(1)}"
    return None


def leaderboard_runs(category_id: str, category_name: str) -> list[dict]:
    payload = _get(
        f"{API}/leaderboards/{GAME_ID}/category/{category_id}?embed=platforms"
    )["data"]
    platforms = {p["id"]: p["name"] for p in payload.get("platforms", {}).get("data", [])}

    rows: list[dict] = []
    for entry in payload.get("runs", []):
        run = entry["run"]
        platform = platforms.get(run.get("system", {}).get("platform"), "")
        links = run.get("videos") or {}
        for link in (links.get("links") or []):
            parsed = parse_video_ids(link.get("uri", ""))
            if not parsed:
                continue
            source, video_id = parsed
            rows.append({
                "video_id": video_id,
                "source": source,
                "url": link["uri"],
                "category": category_name,
                "place": entry.get("place"),
                "duration_s": run.get("times", {}).get("primary_t"),
                "platform": platform,
                "run_id": run.get("id"),
            })
            break                       # one video per run is enough
    return rows


def nitrogen_video_ids(chunk_index: Path) -> set[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(chunk_index, columns=["video_id"])
    return set(table["video_id"].to_pylist())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk-index", type=Path,
                    default=Path("data/celeste_chunk_index.parquet"))
    ap.add_argument("--categories", default=",".join(CATEGORIES))
    ap.add_argument("--pc-only", action="store_true", default=True)
    args = ap.parse_args()

    wanted = [c.strip() for c in args.categories.split(",") if c.strip()]
    rows: list[dict] = []
    for name in wanted:
        cid = CATEGORIES.get(name)
        if not cid:
            print(f"skip unknown category {name!r}")
            continue
        found = leaderboard_runs(cid, name)
        rows.extend(found)
        print(f"{name}: {len(found)} runs with video links")

    # One row per video id; keep the longest run if a video appears twice.
    by_id: dict[str, dict] = {}
    for row in rows:
        prior = by_id.get(row["video_id"])
        if prior is None or (row["duration_s"] or 0) > (prior["duration_s"] or 0):
            by_id[row["video_id"]] = row

    pc = {k: v for k, v in by_id.items() if not args.pc_only or v["platform"] == "PC"}
    dropped_console = len(by_id) - len(pc)

    known = nitrogen_video_ids(args.chunk_index)
    overlap = sorted(set(pc) & known)
    fresh = {k: v for k, v in pc.items() if k not in known}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for row in sorted(fresh.values(),
                          key=lambda r: -(r["duration_s"] or 0)):
            fh.write(json.dumps(row) + "\n")

    hours = sum((r["duration_s"] or 0) for r in fresh.values()) / 3600.0
    print(f"\nruns with videos: {len(rows)}  unique videos: {len(by_id)}")
    print(f"dropped non-PC: {dropped_console}")
    print(f"ALREADY IN NITROGEN (excluded): {len(overlap)}"
          + (f" -> {overlap[:10]}" if overlap else ""))
    print(f"fresh candidates: {len(fresh)}  ({hours:.1f} video-hours)")
    print(f"wrote {args.out}")
    if overlap:
        (args.out.parent / "speedrun_nitrogen_overlap.json").write_text(
            json.dumps(overlap, indent=2)
        )


if __name__ == "__main__":
    main()
