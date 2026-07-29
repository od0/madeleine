"""Assemble the G1 session: video + mod CSV -> frozen-format session dir.

Orchestrator-owned day-0 glue. Measures the strip rect from the video itself,
decodes every frame, builds truth/alignment parquets and the manifest, then
runs the P0 validator. The skeleton mod (0.1.0) logs only the 7 keys; state
fields land with packet A4, so G1 truth carries documented placeholders for
them (input_active=True, empty room, zeroed state) — the gate measures
synchronization, not state.

Usage: uv run python -m theo.g1_assemble --video-dir sessions/_g1_capture \
           --truth-dir ~/madeleine_sessions/inputtruth/<newest> \
           --out sessions/<session_id>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.schema import ALIGNMENT_SCHEMA, KEY_ORDER, TRUTH_SCHEMA
from theo.frameindex import decode_strip

CANON_STRIP = (1024, 96)  # exact 32:3 target for decode after resize


def measure_strip_rect(video: Path) -> tuple[int, int, int, int]:
    """Find the black backing bar's pixel bbox from sampled frames."""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best = None
    for probe_at in (n // 2, n // 3, 2 * n // 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, probe_at)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # The bar lives in the top-left quadrant; it is the widest run of
        # near-black rows starting at y=0.
        region = gray[: gray.shape[0] // 4, : gray.shape[1] // 2]
        dark = region < 40
        col_runs = dark[0].nonzero()[0]
        if len(col_runs) < 50:
            continue
        width = int(col_runs.max()) + 1
        rows = [y for y in range(region.shape[0]) if dark[y, : width].mean() > 0.55]
        if not rows:
            continue
        height = int(max(rows)) + 1
        best = (0, 0, width, height)
        break
    cap.release()
    if best is None:
        raise SystemExit("could not measure strip rect from video")
    return best


def locate_strip_rect(video: Path) -> tuple[int, int, int, int] | None:
    """Find the strip rect by search, validated by TEMPORAL CONSISTENCY.

    The 24-bit payload carries only a 4-bit checksum, so a random rect passes
    it ~1/16 of the time; across a large candidate space, checksum agreement
    on a couple of frames yields false positives (observed: a bogus rect that
    decoded 5% of frames). A genuine rect must instead (a) decode on most
    sampled frames and (b) advance its engine index in step with the video
    frame index (~1:1 at 60 fps). Coincidence cannot satisfy that.
    """
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = [int(v) for v in np.linspace(n * 0.15, n * 0.85, 12)]
    samples = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, f = cap.read()
        if ok:
            samples.append((pos, f))
    cap.release()
    if len(samples) < 4:
        return None

    def try_rect(f, x0, y0, sw):
        sh = max(8, round(sw * 48 / 512))
        crop = f[y0:y0 + sh, x0:x0 + sw]
        if crop.shape[0] < sh or crop.shape[1] < sw:
            return None
        canon = cv2.resize(crop, CANON_STRIP, interpolation=cv2.INTER_AREA)
        return decode_strip(canon)

    def score(x0, y0, sw):
        """(decode_rate, slope_error) over all sampled frames."""
        pts = []
        for pos, f in samples:
            v = try_rect(f, x0, y0, sw)
            if v is not None:
                pts.append((pos, v))
        rate = len(pts) / len(samples)
        if len(pts) < 4:
            return rate, 9.9
        dv = pts[-1][0] - pts[0][0]
        de = pts[-1][1] - pts[0][1]
        slope = de / dv if dv else 0.0
        # engine advances ~1:1 with video frames (drops make it <=1)
        return rate, abs(slope - 1.0)

    ref = samples[len(samples) // 2][1]
    best = None
    for sw in range(360, 549, 4):
        for y0 in range(0, 221, 4):
            for x0 in range(0, 25, 4):
                if try_rect(ref, x0, y0, sw) is None:
                    continue
                rate, slope_err = score(x0, y0, sw)
                if rate < 0.7 or slope_err > 0.15:
                    continue          # false positive: inconsistent over time
                cand = (rate, -slope_err, x0, y0, sw)
                if best is None or cand > best:
                    best = cand
    if best is None:
        return None
    _, _, x0, y0, sw = best
    # fine pass: nudge origin/scale, keep the best temporally-consistent rect
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            for dw in range(-3, 4):
                cx, cy, cw = x0 + dx, y0 + dy, sw + dw
                if cx < 0 or cy < 0 or cw < 8:
                    continue
                if try_rect(ref, cx, cy, cw) is None:
                    continue
                rate, slope_err = score(cx, cy, cw)
                if rate < 0.7 or slope_err > 0.15:
                    continue
                cand = (rate, -slope_err, cx, cy, cw)
                if cand > best:
                    best = cand
    rate, neg_err, x0, y0, sw = best
    print(f"strip search: rect=({x0},{y0},{sw}) decode_rate={rate:.2f} "
          f"slope_err={-neg_err:.3f}")
    return (x0, y0, sw, max(8, round(sw * 48 / 512)))


def decode_video(video: Path, rect: tuple[int, int, int, int]) -> list[int | None]:
    x, y, w, h = rect
    cap = cv2.VideoCapture(str(video))
    out: list[int | None] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y : y + h, x : x + w]
        canon = cv2.resize(crop, CANON_STRIP, interpolation=cv2.INTER_AREA)
        out.append(decode_strip(canon))
    cap.release()
    return out


def build_alignment(decoded: list[int | None]) -> pa.Table:
    readable = [i for i, v in enumerate(decoded) if v is not None]
    first = readable[0] if readable else len(decoded)
    last = readable[-1] if readable else -1
    engine, status, dup, drop = [], [], [], []
    prev: int | None = None
    for i, v in enumerate(decoded):
        if v is None:
            engine.append(-1)
            status.append("out_of_session" if (i < first or i > last) else "unreadable")
            dup.append(False)
            drop.append(0)
            continue
        engine.append(v)
        status.append("ok")
        if prev is None:
            dup.append(False)
            drop.append(0)
        else:
            dup.append(v == prev)
            drop.append(max(0, v - prev - 1))
        prev = v
    return pa.table(
        {
            "video_frame_idx": pa.array(range(len(decoded)), pa.int64()),
            "engine_frame_idx": pa.array(engine, pa.int64()),
            "decode_status": pa.array(status, pa.string()),
            "is_duplicate": pa.array(dup, pa.bool_()),
            "preceded_by_drop_count": pa.array(drop, pa.int32()),
        },
        schema=ALIGNMENT_SCHEMA,
    )


def build_truth(csv_path: Path, session_id: str) -> pa.Table:
    reader = csv.DictReader(csv_path.open())
    rows = list(reader)
    n = len(rows)
    has_state = "input_active" in (reader.fieldnames or [])

    cols: dict[str, list] = {"frame_idx": [int(r["frame_idx"]) for r in rows]}
    for key in KEY_ORDER:
        cols[key] = [r[key] == "1" for r in rows]
    if has_state:
        # Mod >= 0.2.0: real state fields per specs/field_semantics.md
        # (NaN / -1 are the player-absent sentinels, parsed as written).
        cols["input_active"] = [r["input_active"] == "1" for r in rows]
        cols["room_id"] = [r["room_id"] for r in rows]
        for f in ("pos_x", "pos_y", "speed_x", "speed_y", "stamina"):
            cols[f] = [float(r[f]) for r in rows]
        cols["dash_count"] = [int(r["dash_count"]) for r in rows]
        cols["on_ground"] = [r["on_ground"] == "1" for r in rows]
        cols["death"] = [r["death"] == "1" for r in rows]
    else:
        # Skeleton-mod (0.1.0) sessions: documented placeholders.
        cols["input_active"] = [True] * n
        cols["room_id"] = [""] * n
        for f in ("pos_x", "pos_y", "speed_x", "speed_y", "stamina"):
            cols[f] = [0.0] * n
        cols["dash_count"] = [0] * n
        cols["on_ground"] = [False] * n
        cols["death"] = [False] * n
    cols["session_id"] = [session_id] * n
    arrays = [pa.array(cols[f.name], f.type) for f in TRUTH_SCHEMA]
    return pa.Table.from_arrays(arrays, schema=TRUTH_SCHEMA)


def _masked_regions(
    strip_rect: tuple[int, int, int, int],
    capture_px: tuple[int, int],
    mod_meta: dict,
) -> list[dict]:
    """Every rendered answer key becomes a masked region, or downstream
    training is fiction. Mod >= 0.2.0 renders the input overlay too."""
    cw, ch = capture_px
    x, y, w, h = strip_rect

    def region(name: str, rx: int, ry: int, rw: int, rh: int) -> dict:
        return {
            "name": name,
            "space": "capture-pixel (post-encode frame)",
            "applied": "not-applied (masking happens in build_dataset)",
            "rect_px": [rx, ry, rw, rh],
            "rect_norm": [rx / cw, ry / ch, (rx + rw) / cw, (ry + rh) / ch],
        }

    regions = [region("frame_index_strip", x, y, w, h)]
    if mod_meta.get("overlay_style") == "inputtruth-v1":
        # The overlay's DRAW constants are (0,1032,416,48) in 1920x1080
        # logical (specs/overlay_spec.md), but the deployed render pass does
        # not land them at logical*canvas-scale: on the built-in-display rigs
        # the measured vertical transform is y ~ 0.856*logical + 5 against an
        # assumed 0.891*logical, putting the rendered bar ~33 logical px
        # higher and leaving the top of the key cells outside a rect derived
        # from the constants (2026-07-26 masking audit, report/findings_log.md).
        # So the derived rect is a deliberate SUPERSET: from logical y 984
        # down to the bottom of the capture (the bar is bottom-anchored and
        # anything below the canvas is letterbox), 436 logical wide. Actual
        # coverage is verified per session by data.mask_coverage, which
        # build_dataset runs before any shard is written.
        s = w / 512.0
        overlay_y = y + round(984 * s)
        regions.append(region(
            "input_overlay",
            x, overlay_y, round(436 * s), ch - overlay_y,
        ))

        # Calibration sessions add a translucent wild-style overlay. Its rect
        # is READ FROM THE SESSION rather than hardcoded here, because it is
        # optional and its geometry belongs to the mod that drew it. Same
        # canvas-scale derivation, so it is correct on any display geometry.
        if mod_meta.get("wild_overlay_toggled_mid_session"):
            raise SystemExit(
                "meta.json reports the wild overlay was toggled mid-session; "
                "the region was drawn on only some frames, so a single mask "
                "rect is wrong either way. Re-record with the setting fixed."
            )
        wild = mod_meta.get("wild_overlay_rect_logical")
        if mod_meta.get("wild_overlay") and wild:
            wx, wy, ww, wh = (float(v) for v in wild)
            # Same superset discipline as input_overlay: the render pass
            # lands the panel higher than logical*canvas-scale (~28 logical
            # px measured), so pad the derived rect and let
            # data.mask_coverage verify actual coverage per session.
            wx0 = x + round((wx - 16) * s)
            wy0 = y + round((wy - 40) * s)
            regions.append(region(
                "wild_overlay",
                max(0, wx0), max(0, wy0),
                min(cw - max(0, wx0), round((ww + 32) * s)),
                min(ch - max(0, wy0), round((wh + 56) * s)),
            ))
        elif mod_meta.get("wild_overlay"):
            # Declared on but geometry missing: refuse rather than ship a
            # session whose third answer key would go unmasked.
            raise SystemExit(
                "meta.json declares wild_overlay but no "
                "wild_overlay_rect_logical; refusing to assemble an "
                "unmaskable session"
            )
    return regions


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def trim_video(src: Path, dst: Path, a: int, b: int, fps: int = 60) -> None:
    """Re-encode frames [a, b] inclusive of src into dst, CFR-preserving."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vf", f"select='between(n\\,{a}\\,{b})',setpts=N/{fps}/TB",
         "-fps_mode", "cfr", "-r", str(fps),
         "-c:v", "h264_videotoolbox", "-b:v", "20M", "-pix_fmt", "yuv420p",
         str(dst)],
        check=True,
    )


def _verify_trim(dst: Path, expected: list[int | None],
                 rect: tuple[int, int, int, int]) -> None:
    """Assert the trimmed video's frame k decodes to expected[k] (guards
    against any ffmpeg off-by-one, which would silently misalign labels)."""
    cap = cv2.VideoCapture(str(dst))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n != len(expected):
        cap.release()
        raise SystemExit(f"trim frame count {n} != expected {len(expected)}")
    # Use the rect the caller located — recomputing it here would re-apply
    # the fullscreen assumption and look in the wrong place on other geometry.
    x, y, w, h = rect
    for i in sorted({int(v) for v in np.linspace(0, n - 1, 8)}):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        canon = cv2.resize(fr[y:y + h, x:x + w], CANON_STRIP, interpolation=cv2.INTER_AREA)
        got = decode_strip(canon)
        if got != expected[i]:
            cap.release()
            raise SystemExit(f"trim verify FAILED frame {i}: got {got}, expected {expected[i]}")
    cap.release()
    print("trim verified at 8 sample frames")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--truth-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trim-lead-seconds", type=float, default=0.0,
                    help="force-drop at least this many leading seconds")
    ap.add_argument("--trim-tail-seconds", type=float, default=0.0,
                    help="force-drop at least this many trailing seconds")
    args = ap.parse_args()

    video = args.video_dir / "video.mkv"
    capture_meta = json.loads((args.video_dir / "capture_meta.json").read_text())
    mod_meta = json.loads((args.truth_dir / "meta.json").read_text())
    session_id = args.out.name

    cap = cv2.VideoCapture(str(video))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Candidate: fullscreen 1920x1080 canvas scaled to the display. Verify on
    # sampled frames; if the display geometry differs (built-in screen,
    # letterbox, menu bar), fall back to checksum-validated search.
    rect = (0, 0, round(512 * vw / 1920), round(48 * vh / 1080))
    verified = 0
    for pos in np.linspace(nf * 0.2, nf * 0.8, 5, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, f = cap.read()
        if not ok:
            continue
        canon = cv2.resize(f[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]],
                           CANON_STRIP, interpolation=cv2.INTER_AREA)
        if decode_strip(canon) is not None:
            verified += 1
    cap.release()
    if verified >= 2:
        print("computed strip rect verified:", rect)
    else:
        print("computed rect failed verification; searching (checksum-validated)...")
        found = locate_strip_rect(video)
        if found is None:
            raise SystemExit("strip not found by search — is the mod overlay visible?")
        rect = found
        print("located strip rect:", rect)
    decoded = decode_video(video, rect)
    n_ok = sum(1 for v in decoded if v is not None)
    print(f"decoded {n_ok}/{len(decoded)} frames ok")

    args.out.mkdir(parents=True, exist_ok=True)
    out_video = args.out / "video.mkv"

    # Leading/trailing frames with no decodable strip are not part of the
    # session (startup Space-switch, tail) — trim them from the video itself.
    # --trim-lead-seconds forces extra head removal beyond the auto boundary.
    ok_idx = [i for i, v in enumerate(decoded) if v is not None]
    if not ok_idx:
        raise SystemExit("no decodable frames in video")
    lead = max(ok_idx[0], round(args.trim_lead_seconds * 60))
    tail = min(ok_idx[-1], len(decoded) - 1 - round(args.trim_tail_seconds * 60))
    if tail <= lead:
        raise SystemExit("trim settings leave no frames")
    if lead > 0 or tail < len(decoded) - 1:
        print(f"trimming to frames [{lead}, {tail}] "
              f"(dropped {lead} head + {len(decoded) - 1 - tail} tail)")
        trim_video(video, out_video, lead, tail)
        decoded = decoded[lead:tail + 1]
        _verify_trim(out_video, decoded, rect)
    else:
        import shutil
        shutil.copy2(video, out_video)

    alignment = build_alignment(decoded)
    truth = build_truth(args.truth_dir / "truth_raw.csv", session_id)
    pq.write_table(truth, args.out / "truth.parquet")
    pq.write_table(alignment, args.out / "alignment.parquet")

    dup_count = int(sum(alignment.column("is_duplicate").to_pylist()))
    drop_count = int(sum(alignment.column("preceded_by_drop_count").to_pylist()))
    cw, ch = capture_meta["captured_px"]
    x, y, w, h = rect
    manifest = {
        "format_version": "1",
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "env": {
            "game": mod_meta["celeste_version"],
            "everest": mod_meta["everest_version"],
            "mod": f"InputTruth {mod_meta['mod_version']}"
                   + (f" (overlay {mod_meta['overlay_style']})"
                      if mod_meta.get("overlay_style") else " (skeleton: state placeholders)"),
        },
        "capture": {
            "tool": capture_meta["tool"],
            "requested_fps": capture_meta["requested_fps"],
            "achieved_fps": capture_meta["achieved_fps"],
            "encode": capture_meta["encode"],
            "resolution": [cw, ch],
        },
        "streams": {
            "video": "video.mkv",
            "truth": "truth.parquet",
            "alignment": "alignment.parquet",
            "overlay_style": "none",
        },
        "grid": {"engine_hz": 60},
        "label_kind": "engine_truth",
        "masked_regions": _masked_regions(rect, (cw, ch), mod_meta),
        "integrity": {
            "video_frames": len(decoded),
            "duplicates": dup_count,
            "drops": drop_count,
            "sha256": {
                "video.mkv": sha256(args.out / "video.mkv"),
                "truth.parquet": sha256(args.out / "truth.parquet"),
                "alignment.parquet": sha256(args.out / "alignment.parquet"),
            },
        },
        "actions": {"keys": KEY_ORDER},
        "provenance": {
            "source": "recorded",
            "origin_url": None,
            "mapping_report": None,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    ok_idx = [v for v in decoded if v is not None]
    truth_min = truth.column("frame_idx")[0].as_py()
    truth_max = truth.column("frame_idx")[-1].as_py()
    in_range = all(truth_min <= v <= truth_max for v in ok_idx)
    print(
        json.dumps(
            {
                "session": session_id,
                "video_frames": len(decoded),
                "decoded_ok": n_ok,
                "unreadable": sum(
                    1 for i, v in enumerate(decoded) if v is None
                ) - (len(decoded) - n_ok - 0),
                "duplicates": dup_count,
                "drops": drop_count,
                "engine_span": [min(ok_idx), max(ok_idx)] if ok_idx else None,
                "truth_span": [truth_min, truth_max],
                "decoded_within_truth": in_range,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
