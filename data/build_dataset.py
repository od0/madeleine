"""Build badeline training shards from frozen-format sessions.

Orchestrator-owned. This file holds the three decisions that must never be
delegated or implicit:

1. Splits are SESSION-level and explicit: the caller names validation
   sessions; nothing here ever partitions frames, and a session appearing on
   both sides is a hard error (mirrored by badeline.train's own check).
2. Masking is applied HERE, before any resize, from the session manifest's
   masked_regions (rect_norm x frame dims). badeline never opens video.mkv,
   so a frame that leaves this builder unmasked would poison every
   downstream number — the masked-pixel check in the output manifest exists
   for that reason.
3. Grids: recorded sessions only for now (engine_hz 60, label_kind
   engine_truth, enforced). Foreign (mapped, native-grid) sessions enter
   only when the single declared resample policy lands here — nowhere else.

Output per session: <out>/<session_id>.npz with
  frames        uint8 [N,S,S,3] RGB, masked then resized
  keys          uint8 [N,7]     truth keys (KEY_ORDER) at the frame's engine index
  engine_frame_idx int64 [N]
  input_active  uint8 [N]       from truth (placeholder-true for 0.1.0 sessions)
  session_id    str
plus <out>/build_manifest.json describing exactly what was done.

Frame selection: alignment rows with decode_status == "ok" and
is_duplicate == False. Unreadable/out-of-session frames never become
training data; duplicate video frames of one engine frame are kept once.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from data.accept_coverage_waiver import verify_coverage_waiver
from data.mask_coverage import coverage_violations, measure_mask_coverage
from data.schema import KEY_ORDER
from data.validate_session import validate_session

FRAME_SIZE = 128


def build_session(
    session_dir: Path,
    out_dir: Path,
    frame_size: int = FRAME_SIZE,
    coverage_waiver: Path | None = None,
) -> dict:
    session_dir = Path(session_dir)
    session_id = session_dir.name

    violations = validate_session(session_dir)
    if violations:
        raise SystemExit(f"{session_id}: invalid session: {violations[:3]}")
    # Declared rects zeroing out is checked below; that the rects actually
    # COVER the rendered widgets is a separate contract, learned the hard way
    # (2026-07-26 undershoot, report/findings_log.md). No bypass flag: the
    # only way past a coverage failure is a verified, hash-bound, human
    # mask-coverage waiver (data.accept_coverage_waiver, 2026-07-28 owner
    # ruling on chapter-content correlation), and the build manifest then
    # records the gate as overridden, never as passed.
    coverage_report = measure_mask_coverage(session_dir)
    found_coverage_violations = coverage_violations(coverage_report)
    coverage_note = None
    if found_coverage_violations:
        if coverage_waiver is None:
            raise SystemExit(
                f"{session_id}: mask coverage failed: {found_coverage_violations}"
            )
        waiver = verify_coverage_waiver(
            session_dir, coverage_waiver, report=coverage_report
        )
        coverage_note = {
            "outcome": "overridden_by_recorded_human_waiver",
            "waiver_sha256": waiver["sha256"],
            "band_fractions": waiver["band_fractions"],
        }
    elif coverage_waiver is not None:
        raise SystemExit(
            f"{session_id}: mask coverage passed; refusing a coverage waiver "
            "with nothing to waive"
        )

    manifest = json.loads((session_dir / "manifest.json").read_text())
    if manifest["provenance"]["source"] != "recorded":
        raise SystemExit(f"{session_id}: foreign sessions need the resample policy; not built yet")
    if manifest["grid"].get("engine_hz") != 60:
        raise SystemExit(f"{session_id}: expected engine_hz 60")
    if manifest["label_kind"] != "engine_truth":
        raise SystemExit(f"{session_id}: label_kind must be engine_truth")

    alignment = pq.read_table(session_dir / "alignment.parquet")
    truth = pq.read_table(session_dir / "truth.parquet")

    status = np.asarray(alignment["decode_status"].to_pylist())
    dup = np.asarray(alignment["is_duplicate"].to_pylist(), dtype=bool)
    engine_idx = np.asarray(alignment["engine_frame_idx"].to_pylist(), dtype=np.int64)
    keep = (status == "ok") & (~dup)

    truth_frame_idx = np.asarray(truth["frame_idx"].to_pylist(), dtype=np.int64)
    truth_base = truth_frame_idx[0]
    # truth frame_idx is validated dense, so engine index -> row is arithmetic.
    keys_all = np.stack(
        [np.asarray(truth[k].to_pylist(), dtype=np.uint8) for k in KEY_ORDER], axis=1
    )
    active_all = np.asarray(truth["input_active"].to_pylist(), dtype=np.uint8)

    regions = manifest["masked_regions"]
    if not regions:
        raise SystemExit(f"{session_id}: no masked_regions in manifest; refusing to build")

    cap = cv2.VideoCapture(str(session_dir / "video.mkv"))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_out, keys_out, engine_out, active_out = [], [], [], []
    masked_px_checked = False
    for video_frame in range(n_video):
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"{session_id}: video ended early at frame {video_frame}")
        if not keep[video_frame]:
            continue
        e = engine_idx[video_frame]
        row = e - truth_base
        if row < 0 or row >= len(keys_all):
            continue
        fh, fw = frame.shape[:2]
        # Mask at full-res (removes the answer-key cells so they can't blend
        # into neighbours during downscale).
        for region in regions:
            x0, y0, x1, y1 = region["rect_norm"]
            frame[int(y0 * fh) : int(np.ceil(y1 * fh)), int(x0 * fw) : int(np.ceil(x1 * fw))] = 0
        small = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        # Re-mask at output resolution with a 1px dilation: INTER_AREA blends a
        # 1px boundary from adjacent game pixels, so zero it too. This makes the
        # masked region provably exactly 0 (answer-key trap: no doubt allowed).
        for region in regions:
            x0, y0, x1, y1 = region["rect_norm"]
            mx0 = max(0, int(x0 * frame_size) - 1)
            my0 = max(0, int(y0 * frame_size) - 1)
            mx1 = min(frame_size, int(np.ceil(x1 * frame_size)) + 1)
            my1 = min(frame_size, int(np.ceil(y1 * frame_size)) + 1)
            small[my0:my1, mx0:mx1] = 0
        if not masked_px_checked:
            for region in regions:
                x0, y0, x1, y1 = region["rect_norm"]
                mx0 = max(0, int(x0 * frame_size) - 1)
                my0 = max(0, int(y0 * frame_size) - 1)
                mx1 = min(frame_size, int(np.ceil(x1 * frame_size)) + 1)
                my1 = min(frame_size, int(np.ceil(y1 * frame_size)) + 1)
                assert int(small[my0:my1, mx0:mx1].max()) == 0, region["name"]
            masked_px_checked = True
        frames_out.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        keys_out.append(keys_all[row])
        engine_out.append(e)
        active_out.append(active_all[row])
    cap.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}.npz"
    np.savez_compressed(
        out_path,
        frames=np.stack(frames_out),
        keys=np.stack(keys_out),
        engine_frame_idx=np.asarray(engine_out, dtype=np.int64),
        input_active=np.asarray(active_out, dtype=np.uint8),
        session_id=session_id,
    )
    report = {
        "session_id": session_id,
        "frames": len(frames_out),
        "video_frames": n_video,
        "excluded": int(n_video - len(frames_out)),
        "input_active_frames": int(np.sum(active_out)),
        "masked_regions": [r["name"] for r in regions],
        "npz": out_path.name,
    }
    if coverage_note is not None:
        report["mask_coverage"] = coverage_note
    return report


# --------------------------------------------------------------------------
# Foreign (mapped NitroGen) sessions.
#
# This is the brief's single declared entry point for foreign data, and for
# now the declared resample policy is the identity: only videos whose every
# chunk runs at grid_hz == 60 with one label row per source video frame are
# accepted, decoded at their native 60 fps, and paired row-for-frame. Any
# other rate is refused here — resampling to the engine grid is a measured,
# explicit decision (E4 priced timing error at 4.5% macro-F1 per frame) and
# it does not exist until it is implemented HERE, nowhere else.
#
# Unit of output: the maximal contiguous run of retained chunks (NitroGen's
# filtering drops chunks, so a video is runs-with-gaps, not one stream). One
# NPZ per run part, session_id "<video_id>__rNNN", so the training dataset's
# windows never silently span a gap. Split discipline for foreign data is
# VIDEO-level: every run of a video goes to the same side, enforced by the
# split lists that name them.
# --------------------------------------------------------------------------

FOREIGN_GRID_HZ = 60.0
MIN_RUN_FRAMES = 240          # runs under 4 s are clutter, skipped + counted
MAX_PART_FRAMES = 36_000      # split runs at 10 min to bound memory


def _foreign_runs(
    chunk_frames: Path, mapped_video_dir: Path, video_id: str
) -> list[list[dict]]:
    """Contiguous runs of retained, 60 Hz, label-complete chunks."""

    table = pq.read_table(chunk_frames)
    rows = [
        {k: table[k][i].as_py() for k in table.column_names}
        for i in range(table.num_rows)
        if table["video_id"][i].as_py() == video_id
    ]
    if not rows:
        raise SystemExit(f"{video_id}: no chunks in {chunk_frames}")
    rows.sort(key=lambda r: r["start_frame"])

    runs: list[list[dict]] = []
    for row in rows:
        declared_end = row["end_frame"]
        exclusive_span = declared_end - row["start_frame"]
        inclusive_span = exclusive_span + 1
        if row["n_rows"] == exclusive_span:
            row["end_convention"] = "exclusive"
        elif row["n_rows"] == inclusive_span:
            # NitroGen extraction metadata uses an inclusive source end-frame
            # in the current Celeste slice. Normalize once at this boundary;
            # all downstream ranges remain ordinary Python half-open ranges.
            row["end_frame"] = declared_end + 1
            row["end_convention"] = "inclusive"
        else:
            raise SystemExit(
                f"{video_id}/{row['chunk_id']}: grid_hz {row['grid_hz']:.2f} "
                f"or n_rows {row['n_rows']} matches neither exclusive span "
                f"{exclusive_span} nor inclusive span {inclusive_span}; only "
                "1:1 60 Hz chunks are accepted"
            )
        if abs(row["grid_hz"] - FOREIGN_GRID_HZ) > 0.01:
            raise SystemExit(
                f"{video_id}/{row['chunk_id']}: grid_hz "
                f"{row['grid_hz']:.2f}; only 60 Hz chunks are accepted "
                "(resample policy not implemented)"
            )
        labels = mapped_video_dir / row["chunk_id"] / "labels_native.parquet"
        if not labels.is_file():
            raise SystemExit(f"{video_id}/{row['chunk_id']}: mapped labels missing")
        row["labels_path"] = labels
        if runs and runs[-1][-1]["end_frame"] == row["start_frame"]:
            runs[-1].append(row)
        else:
            runs.append([row])
    return runs


def _run_keys(run: list[dict]) -> np.ndarray:
    """Concatenate a run's chunk labels into one [N,7] uint8 array."""

    parts = []
    for row in run:
        table = pq.read_table(row["labels_path"])
        if table.num_rows != row["n_rows"]:
            raise SystemExit(
                f"{row['chunk_id']}: labels rows {table.num_rows} != "
                f"declared {row['n_rows']}"
            )
        idx = np.asarray(table["frame_idx"].to_pylist())
        if not np.array_equal(idx, np.arange(row["n_rows"])):
            raise SystemExit(f"{row['chunk_id']}: frame_idx not dense from 0")
        parts.append(np.stack(
            [np.asarray(table[k].to_pylist(), dtype=np.uint8) for k in KEY_ORDER],
            axis=1,
        ))
    return np.concatenate(parts)


def build_foreign_video(
    video_path: Path,
    video_id: str,
    mapped_root: Path,
    chunk_frames: Path,
    chunk_index: Path,
    out_dir: Path,
    frame_size: int = FRAME_SIZE,
) -> dict:
    from nitrogen.mask import video_mask_rect

    mapped_video_dir = Path(mapped_root) / video_id
    if not mapped_video_dir.is_dir():
        raise SystemExit(f"{video_id}: no mapped labels at {mapped_video_dir}")
    mapping_report = json.loads(
        (mapped_video_dir / "mapping_report.json").read_text()
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"{video_id}: cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if abs(fps - FOREIGN_GRID_HZ) > 0.1:
        raise SystemExit(f"{video_id}: video fps {fps:.2f}, need {FOREIGN_GRID_HZ}")

    rect = video_mask_rect(chunk_index, video_id, (width, height))
    rx0, ry0, rx1, ry1 = rect
    # The same rect at output resolution, dilated 1 px (INTER_AREA blends a
    # 1 px boundary), mirroring the recorded-session discipline above.
    sx0 = max(0, int(rx0 / width * frame_size) - 1)
    sy0 = max(0, int(ry0 / height * frame_size) - 1)
    sx1 = min(frame_size, int(np.ceil(rx1 / width * frame_size)) + 1)
    sy1 = min(frame_size, int(np.ceil(ry1 / height * frame_size)) + 1)

    runs = _foreign_runs(chunk_frames, mapped_video_dir, video_id)
    # Split long runs into bounded parts; drop sub-window fragments.
    parts: list[tuple[int, int, np.ndarray]] = []   # (start_frame, end_frame, keys)
    skipped_short = truncated = 0
    for run in runs:
        keys = _run_keys(run)
        start, end = run[0]["start_frame"], run[-1]["end_frame"]
        if end > n_video:
            # Tail-trimmed VOD: keep what exists, declare the loss.
            truncated += end - n_video
            end = n_video
            keys = keys[: max(0, end - start)]
        for part_start in range(start, end, MAX_PART_FRAMES):
            part_end = min(part_start + MAX_PART_FRAMES, end)
            if part_end - part_start < MIN_RUN_FRAMES:
                skipped_short += part_end - part_start
                continue
            parts.append((
                part_start, part_end,
                keys[part_start - start : part_end - start],
            ))

    out_dir.mkdir(parents=True, exist_ok=True)
    session_rows = []
    masked_px_checked = False
    cursor = 0
    for part_index, (start, end, keys) in enumerate(parts):
        frames_out = np.empty((end - start, frame_size, frame_size, 3), np.uint8)
        while cursor < start:
            if not cap.grab():
                raise SystemExit(f"{video_id}: video ended at {cursor} before part start {start}")
            cursor += 1
        for i in range(end - start):
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"{video_id}: decode failed at frame {cursor}")
            cursor += 1
            frame[ry0:ry1, rx0:rx1] = 0
            small = cv2.resize(frame, (frame_size, frame_size),
                               interpolation=cv2.INTER_AREA)
            small[sy0:sy1, sx0:sx1] = 0
            if not masked_px_checked:
                assert int(small[sy0:sy1, sx0:sx1].max()) == 0
                masked_px_checked = True
            frames_out[i] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        session_id = f"{video_id}__r{part_index:03d}"
        np.savez_compressed(
            out_dir / f"{session_id}.npz",
            frames=frames_out,
            keys=keys,
            engine_frame_idx=np.arange(start, end, dtype=np.int64),
            input_active=np.ones(end - start, dtype=np.uint8),
            session_id=session_id,
        )
        session_rows.append({
            "session_id": session_id,
            "frames": int(end - start),
            "source_frame_range": [int(start), int(end)],
            "npz": f"{session_id}.npz",
        })
    cap.release()

    return {
        "video_id": video_id,
        "label_kind": "mapped",
        "grid_hz": FOREIGN_GRID_HZ,
        "video": {"path": str(video_path), "fps": fps,
                  "resolution_wh": [width, height], "frames": n_video},
        "mask_rect_xyxy": [int(v) for v in rect],
        "bind_confidence": mapping_report["confidence"],
        "bind_map": mapping_report["bind_map"],
        "end_frame_conventions": sorted({
            row["end_convention"] for run in runs for row in run
        }),
        "runs": len(runs),
        "parts": session_rows,
        "skipped_short_frames": int(skipped_short),
        "tail_truncated_frames": int(truncated),
    }


def _main_foreign(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        description="Build mapped-label shards from a re-fetched video."
    )
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--mapped-root", type=Path, required=True)
    ap.add_argument("--chunk-frames", type=Path, required=True)
    ap.add_argument("--chunk-index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frame-size", type=int, default=FRAME_SIZE)
    args = ap.parse_args(argv)

    report = build_foreign_video(
        args.video, args.video_id, args.mapped_root, args.chunk_frames,
        args.chunk_index, args.out, args.frame_size,
    )
    manifest_path = args.out / "foreign_build_manifest.json"
    existing = (
        json.loads(manifest_path.read_text()) if manifest_path.exists()
        else {"built_at": None, "frame_size": args.frame_size,
              "grid": {"grid_hz": FOREIGN_GRID_HZ, "label_kind": "mapped"},
              "videos": []}
    )
    existing["built_at"] = datetime.now(timezone.utc).isoformat()
    existing["videos"] = [
        v for v in existing["videos"] if v["video_id"] != report["video_id"]
    ] + [report]
    manifest_path.write_text(json.dumps(existing, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("video_id", "runs", "skipped_short_frames",
                       "tail_truncated_frames")}, indent=2))
    print(f"parts: {len(report['parts'])}")


def main(argv: list[str] | None = None) -> None:
    import sys
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "foreign":
        _main_foreign(argv[1:])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", type=Path, required=True)
    ap.add_argument("--val-sessions", nargs="*", default=[],
                    help="session_ids held out for validation (explicit, never inferred)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frame-size", type=int, default=FRAME_SIZE)
    ap.add_argument("--coverage-waiver", dest="coverage_waivers", action="append",
                    type=Path, default=[],
                    help="hash-bound human mask-coverage waiver JSON "
                         "(data.accept_coverage_waiver; repeatable, one per session)")
    args = ap.parse_args(argv)

    all_ids = [p.name for p in args.sessions]
    val_ids = list(args.val_sessions)
    unknown = set(val_ids) - set(all_ids)
    if unknown:
        raise SystemExit(f"--val-sessions not among --sessions: {sorted(unknown)}")
    train_ids = [s for s in all_ids if s not in set(val_ids)]
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise SystemExit(f"overlapping split: {sorted(overlap)}")

    waivers: dict[str, Path] = {}
    for waiver_path in args.coverage_waivers:
        entry = json.loads(Path(waiver_path).read_text(encoding="utf-8"))
        waived_id = entry.get("session_id") if isinstance(entry, dict) else None
        if not isinstance(waived_id, str) or not waived_id:
            raise SystemExit(f"--coverage-waiver {waiver_path}: no session_id")
        if waived_id in waivers:
            raise SystemExit(
                f"--coverage-waiver: duplicate waiver for session {waived_id}"
            )
        if waived_id not in set(all_ids):
            raise SystemExit(
                f"--coverage-waiver {waiver_path}: session {waived_id} is not "
                "among --sessions"
            )
        waivers[waived_id] = Path(waiver_path)

    reports = [
        build_session(s, args.out, args.frame_size,
                      coverage_waiver=waivers.get(s.name))
        for s in args.sessions
    ]

    (args.out / "train_sessions.txt").write_text("\n".join(train_ids) + "\n")
    (args.out / "val_sessions.txt").write_text("\n".join(val_ids) + ("\n" if val_ids else ""))
    (args.out / "build_manifest.json").write_text(json.dumps({
        "built_at": datetime.now(timezone.utc).isoformat(),
        "frame_size": args.frame_size,
        "split": {"train": train_ids, "val": val_ids, "unit": "session"},
        "grid": {"engine_hz": 60},
        "sessions": reports,
    }, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
