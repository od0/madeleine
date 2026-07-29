"""Build a resumable, all-video frozen-feature NitroGen corpus.

The source corpus being present is not enough for ``badeline.train``.  This
driver turns the metadata-valid raw videos and mapped action tables into the
exact FP16 ResNet-18 feature shards consumed by the current trainer, then
assembles one validated hard-link directory and manifest.

No video is silently filtered by bind confidence.  ``train_sessions.txt``
contains every successfully built metadata-valid video; an additional
``unflagged_sessions.txt`` is emitted for a future quality ablation.

Known missing chunks are already separate runs, so the exhaustive pixel-level
continuity audit is deliberately not an input or prerequisite.  Visual quality
rankings can be joined to the manifest later without changing corpus identity.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.schema import KEY_ORDER
from experiments.audit_corpus_contiguity import VideoPlan, build_plans


FEATURE_DIM = 512
CHUNK_SECONDS = 20.0
CHUNK_FRAME_SCHEMA = pa.schema([
    pa.field("video_id", pa.string()),
    pa.field("chunk_id", pa.string()),
    pa.field("start_frame", pa.int64()),
    pa.field("end_frame", pa.int64()),
    pa.field("start_time", pa.float64()),
    pa.field("end_time", pa.float64()),
    pa.field("grid_hz", pa.float64()),
    pa.field("n_rows", pa.int64()),
])


def build_chunk_frames(
    chunk_index: Path,
    video_ids: set[str],
    destination: Path,
    *,
    reference: Path | None = None,
) -> list[dict[str, Any]]:
    """Write the exclusive-frame chunk table for selected native-60 videos."""

    source_rows = pq.read_table(chunk_index).to_pylist()
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        video_id = str(source["video_id"])
        if video_id not in video_ids:
            continue
        grid_hz = float(source["grid_hz"])
        count = int(source["chunk_size"])
        if grid_hz != 60.0 or count != 1200:
            raise ValueError(
                f"{video_id}/{source['chunk_id']}: expected 1,200 rows at "
                f"60 Hz, found {count} at {grid_hz}"
            )
        chunk_number = int(source["chunk_id"])
        start_frame = chunk_number * count
        rows.append({
            "video_id": video_id,
            "chunk_id": f"{video_id}_chunk_{chunk_number:04d}",
            "start_frame": start_frame,
            "end_frame": start_frame + count,
            "start_time": chunk_number * CHUNK_SECONDS,
            "end_time": (chunk_number + 1) * CHUNK_SECONDS,
            "grid_hz": grid_hz,
            "n_rows": count,
        })
    rows.sort(key=lambda row: (row["video_id"], row["start_frame"]))
    observed = {str(row["video_id"]) for row in rows}
    if observed != video_ids:
        raise ValueError(
            f"chunk index video mismatch: missing={sorted(video_ids-observed)}, "
            f"extra={sorted(observed-video_ids)}"
        )

    if reference is not None and reference.is_file():
        reference_rows = {
            (str(row["video_id"]), str(row["chunk_id"])): row
            for row in pq.read_table(reference).to_pylist()
            if str(row["video_id"]) in video_ids
        }
        generated = {
            (str(row["video_id"]), str(row["chunk_id"])): row for row in rows
        }
        for key in sorted(reference_rows.keys() & generated.keys()):
            expected = reference_rows[key]
            actual = generated[key]
            for field in CHUNK_FRAME_SCHEMA.names:
                if expected[field] != actual[field]:
                    raise ValueError(
                        f"generated {key} field {field}={actual[field]!r}, "
                        f"reference has {expected[field]!r}"
                    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pylist(rows, schema=CHUNK_FRAME_SCHEMA), temporary)
    temporary.replace(destination)
    return rows


def _run_video(
    plan: VideoPlan,
    *,
    repo: Path,
    mapped_root: Path,
    chunk_frames: Path,
    chunk_index: Path,
    feature_root: Path,
    log_root: Path,
    batch_size: int,
) -> dict[str, Any]:
    output = feature_root / plan.video_id
    log_path = log_root / f"{plan.video_id}.log"
    command = [
        sys.executable,
        "-m", "data.precompute_features", "foreign",
        "--video", plan.path,
        "--video-id", plan.video_id,
        "--mapped-root", str(mapped_root),
        "--chunk-frames", str(chunk_frames),
        "--chunk-index", str(chunk_index),
        "--out", str(output),
        "--device", "cuda",
        "--batch-size", str(batch_size),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo)
    started = datetime.now(timezone.utc)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "video_id": plan.video_id,
        "returncode": result.returncode,
        "log": str(log_path),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def _validate_feature_shard(path: Path, session_id: str) -> int:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "features", "keys", "engine_frame_idx", "input_active", "session_id"
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        features = archive["features"]
        keys = archive["keys"]
        engine = archive["engine_frame_idx"]
        active = archive["input_active"]
        stored_id = str(archive["session_id"].reshape(()).item())
        frames = len(features)
        if features.dtype != np.float16 or features.shape != (frames, FEATURE_DIM):
            raise ValueError(f"{path}: invalid feature array")
        if keys.dtype != np.uint8 or keys.shape != (frames, len(KEY_ORDER)):
            raise ValueError(f"{path}: invalid keys")
        if engine.shape != (frames,) or active.shape != (frames,):
            raise ValueError(f"{path}: supervision arrays do not align")
        if stored_id != session_id:
            raise ValueError(f"{path}: session id {stored_id!r} != {session_id!r}")
    return frames


def assemble(
    plans: list[VideoPlan],
    *,
    mapped_root: Path,
    feature_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate every feature shard and assemble all/quality manifests."""

    output.mkdir(parents=True, exist_ok=True)
    sessions: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    seen: set[str] = set()
    for plan in sorted(plans, key=lambda item: item.video_id):
        feature_manifest_path = feature_root / plan.video_id / "feature_build_manifest.json"
        if not feature_manifest_path.is_file():
            raise ValueError(f"{plan.video_id}: feature manifest missing")
        feature_manifest = json.loads(feature_manifest_path.read_text())
        report = feature_manifest["videos"][0]
        if report["video_id"] != plan.video_id:
            raise ValueError(f"{plan.video_id}: feature manifest identity mismatch")
        mapping = json.loads(
            (mapped_root / plan.video_id / "mapping_report.json").read_text()
        )
        video_frames = 0
        video_sessions: list[str] = []
        for part in report["parts"]:
            session_id = str(part["session_id"])
            if session_id in seen:
                raise ValueError(f"duplicate session id {session_id}")
            seen.add(session_id)
            source = feature_root / plan.video_id / str(part["npz"])
            frames = _validate_feature_shard(source, session_id)
            if frames != int(part["frames"]):
                raise ValueError(f"{source}: frame count disagrees with manifest")
            destination = output / f"{session_id}.npz"
            if destination.exists():
                if not os.path.samefile(source, destination):
                    raise ValueError(f"{destination}: exists but is not source hard link")
            else:
                os.link(source, destination)
            sessions.append({
                "session_id": session_id,
                "video_id": plan.video_id,
                "frames": frames,
                "source": str(source),
                "bind_confidence": float(mapping["confidence"]),
                "bind_flagged": bool(mapping["flagged"]),
                "decoder_mode": str(
                    part.get("decoder_mode", report["decoder_mode"])
                ),
                "imputed_tail_frames": int(
                    part.get("imputed_tail_frames", 0)
                ),
            })
            video_sessions.append(session_id)
            video_frames += frames
        videos.append({
            "video_id": plan.video_id,
            "frames": video_frames,
            "label_hours": video_frames / 216_000.0,
            "source_label_frames": plan.label_frames,
            "train_to_source_fraction": video_frames / plan.label_frames,
            "sessions": video_sessions,
            "bind_confidence": float(mapping["confidence"]),
            "bind_flagged": bool(mapping["flagged"]),
            "label_run_count": plan.label_run_count,
            "long_context_fraction": plan.long_context_fraction,
            "decoder_mode": str(report["decoder_mode"]),
            "source_average_fps": float(report["video"]["average_fps"]),
            "source_decoded_frames": int(report["video"]["decoded_frames"]),
            "nominal_timeline_frames": int(
                report["video"]["nominal_timeline_frames"]
            ),
            "imputed_tail_frames": int(report["imputed_tail_frames"]),
            "tail_truncated_frames": int(report["tail_truncated_frames"]),
            "skipped_short_frames": int(report["skipped_short_frames"]),
        })

    all_ids = sorted(row["session_id"] for row in sessions)
    unflagged_ids = sorted(
        row["session_id"] for row in sessions if not row["bind_flagged"]
    )
    (output / "train_sessions.txt").write_text("\n".join(all_ids) + "\n")
    (output / "unflagged_sessions.txt").write_text(
        "\n".join(unflagged_ids) + ("\n" if unflagged_ids else "")
    )
    (output / "val_sessions.txt").write_text("")
    total_frames = sum(row["frames"] for row in sessions)
    source_label_frames = sum(plan.label_frames for plan in plans)
    decoder_mode_counts = {
        mode: sum(row["decoder_mode"] == mode for row in videos)
        for mode in sorted({row["decoder_mode"] for row in videos})
    }
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "format": "resnet18_imagenet_avgpool_float16_v1",
        "source_kind": "mapped_foreign_video",
        "selection": (
            "all metadata-valid nominal-60 videos; variable-rate decoded "
            "sources are timestamp-resampled to the label grid; bind "
            "confidence is recorded but does not filter train_sessions.txt"
        ),
        "video_count": len(videos),
        "session_count": len(sessions),
        "train_frames": total_frames,
        "train_label_hours_at_60hz": total_frames / 216_000.0,
        "source_label_frames": source_label_frames,
        "source_label_hours_at_60hz": source_label_frames / 216_000.0,
        "train_to_source_fraction": total_frames / source_label_frames,
        "decoder_mode_counts": decoder_mode_counts,
        "imputed_tail_frames": sum(
            int(row["imputed_tail_frames"]) for row in videos
        ),
        "tail_truncated_frames": sum(
            int(row["tail_truncated_frames"]) for row in videos
        ),
        "skipped_short_frames": sum(
            int(row["skipped_short_frames"]) for row in videos
        ),
        "unflagged_video_count": sum(not row["bind_flagged"] for row in videos),
        "unflagged_session_count": len(unflagged_ids),
        "videos": videos,
    }
    (output / "full_corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--chunk-index", type=Path, required=True)
    parser.add_argument("--fetch-report", type=Path, required=True)
    parser.add_argument("--mapped-root", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=Path, required=True)
    parser.add_argument("--reference-chunk-frames", type=Path)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    plans, rejected = build_plans(
        args.raw_root, args.chunk_index, args.fetch_report
    )
    if len(plans) != 211 or len(rejected) != 21:
        raise ValueError(
            "expected 211 valid plans and 21 explicit rejections, got "
            f"{len(plans)} valid / {len(rejected)} rejected"
        )
    video_ids = {plan.video_id for plan in plans}
    build_chunk_frames(
        args.chunk_index,
        video_ids,
        args.chunk_frames,
        reference=args.reference_chunk_frames,
    )
    missing_mapping = [
        plan.video_id for plan in plans
        if not (args.mapped_root / plan.video_id / "mapping_report.json").is_file()
    ]
    if missing_mapping:
        raise ValueError(f"missing mapping reports: {missing_mapping[:10]}")

    args.feature_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    work = sorted(plans, key=lambda plan: plan.label_frames, reverse=True)
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_video,
                plan,
                repo=args.repo,
                mapped_root=args.mapped_root,
                chunk_frames=args.chunk_frames,
                chunk_index=args.chunk_index,
                feature_root=args.feature_root,
                log_root=args.log_root,
                batch_size=args.batch_size,
            ): plan
            for plan in work
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), 1
        ):
            result = future.result()
            status = "ok" if result["returncode"] == 0 else "ERROR"
            print(
                f"[{completed}/{len(work)}] {result['video_id']} {status}",
                flush=True,
            )
            if result["returncode"] != 0:
                failures.append(result)
    if failures:
        raise SystemExit(
            "feature generation failed: "
            + ", ".join(row["video_id"] for row in failures)
        )

    manifest = assemble(
        plans,
        mapped_root=args.mapped_root,
        feature_root=args.feature_root,
        output=args.output,
    )
    print(json.dumps({
        "video_count": manifest["video_count"],
        "session_count": manifest["session_count"],
        "train_frames": manifest["train_frames"],
        "train_label_hours_at_60hz": manifest["train_label_hours_at_60hz"],
        "unflagged_video_count": manifest["unflagged_video_count"],
        "decoder_mode_counts": manifest["decoder_mode_counts"],
        "imputed_tail_frames": manifest["imputed_tail_frames"],
    }, indent=2))


if __name__ == "__main__":
    main()
