"""Build a deterministic classifier index from a mirrored probe campaign.

Workers publish one immutable directory per attempt.  This indexer only admits
attempts whose completion marker says ``status=ok`` and whose probe frame and
declared crops are present. Duplicate video attempts are retained as provenance
but one is selected deterministically by lexicographic attempt path.  Machine
classification downstream remains a nomination and does not turn these rows
into human-reviewed data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-root", type=Path, required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verify-sha256", action="store_true")
    args = ap.parse_args()

    attempts: dict[str, list[dict]] = defaultdict(list)
    rejected: list[dict] = []
    for marker in sorted(args.campaign_root.glob("*/*/probe_complete.json")):
        attempt_dir = marker.parent
        worker_id = attempt_dir.parent.name
        marker_data = load_json(marker)
        video_id = str(marker_data.get("video_id", ""))
        problems: list[str] = []
        if marker_data.get("status") != "ok":
            problems.append(f"completion_status:{marker_data.get('status')}")
        if video_id != attempt_dir.name:
            problems.append("video_id_directory_mismatch")
        probe_path = attempt_dir / "probe.json"
        if not probe_path.is_file():
            problems.append("missing_probe_json")
            probe = {}
        else:
            probe = load_json(probe_path)
            if probe.get("error") is not None:
                problems.append("probe_has_error")
            if probe.get("video_id") != video_id:
                problems.append("probe_video_id_mismatch")
        frame_path = attempt_dir / "frames" / f"{video_id}.png"
        if not frame_path.is_file():
            problems.append("missing_frame")
        crop_paths = [attempt_dir / "crops" / name for name in probe.get("crops", [])]
        if any(not path.is_file() for path in crop_paths):
            problems.append("missing_declared_crop")

        object_by_name = {
            obj.get("name"): obj for obj in marker_data.get("objects", [])
        }
        evidence_paths = [probe_path, frame_path, *crop_paths]
        if args.verify_sha256 and not problems:
            for evidence_path in evidence_paths:
                obj = object_by_name.get(evidence_path.name)
                if obj is None:
                    problems.append(f"undeclared_object:{evidence_path.name}")
                elif evidence_path.stat().st_size != obj.get("size_bytes"):
                    problems.append(f"size_mismatch:{evidence_path.name}")
                elif sha256(evidence_path) != obj.get("sha256"):
                    problems.append(f"sha256_mismatch:{evidence_path.name}")

        relative_attempt = str(attempt_dir.relative_to(args.campaign_root))
        if problems:
            rejected.append({
                "attempt_path": relative_attempt,
                "video_id": video_id,
                "problems": problems,
            })
            continue
        attempts[video_id].append({
            **probe,
            "campaign_id": args.campaign_id,
            "worker_id": worker_id,
            "attempt_path": relative_attempt,
            "frame_path": str(frame_path.resolve()),
            "crop_paths": [str(path.resolve()) for path in crop_paths],
            "probe_complete_sha256": sha256(marker),
            "probe_sha256": sha256(probe_path),
        })

    rows = []
    for video_id in sorted(attempts):
        candidates = sorted(attempts[video_id], key=lambda row: row["attempt_path"])
        selected = dict(candidates[0])
        selected["duplicate_attempt_paths"] = [
            row["attempt_path"] for row in candidates[1:]
        ]
        rows.append(selected)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    manifest = {
        "schema_version": 1,
        "campaign_id": args.campaign_id,
        "campaign_root": str(args.campaign_root.resolve()),
        "verify_sha256": args.verify_sha256,
        "completion_markers_seen": sum(len(value) for value in attempts.values())
        + len(rejected),
        "unique_successful_videos": len(rows),
        "duplicate_successful_attempts": sum(
            max(0, len(value) - 1) for value in attempts.values()
        ),
        "rejected_attempts": rejected,
        "index_sha256": sha256(args.out),
    }
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
