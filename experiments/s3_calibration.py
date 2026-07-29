"""Phase A: score the translucent decoder against engine truth.

The gate that decides whether the speedrun harvest can produce labels at all.
A calibration session carries three overlays at once: the frame-index strip
(alignment), the opaque input overlay (the E4 instrument, already validated at
macro-F1 1.0), and the translucent wild-style overlay this experiment decodes.
Because all three sit in one recording alongside the engine-truth CSV, the
translucent decoder can be scored exactly the way E4 scored the opaque one.

Reported per key: precision, recall, F1, and onset timing offsets — plus the
offset sweep, because a decoder that is accurate but systematically late is
usable (shift the labels) while one that is accurate only at a jitter of zero
is not. Reuses experiments.e4_metrology's scorers so the numbers are directly
comparable to the opaque baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from data.schema import KEY_ORDER
from experiments.e4_metrology import _macro_f1, onset_err, prf
from harvest.translucent_parser import (
    calibrate_threshold,
    cell_deltas,
    decode,
    scale_geometry,
)


def load_session(session: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    manifest = json.loads((session / "manifest.json").read_text())
    rects = {r["name"]: r for r in manifest["masked_regions"]}
    if "wild_overlay" not in rects:
        raise SystemExit(
            f"{session}: no wild_overlay masked region — this is not a "
            f"calibration session"
        )
    # The decoder was validated with cell geometry scaled from the panel
    # rect as originally declared. After the 2026-07-26 masking audit,
    # rect_px is the (larger) measured MASK rect; panel_rect_px preserves
    # the decoder's validated anchor. Masking and instrument geometry are
    # separate duties — do not scale cells from the mask rect.
    wild_region = rects["wild_overlay"]
    panel = tuple(wild_region.get("panel_rect_px", wild_region["rect_px"]))

    alignment = pq.read_table(session / "alignment.parquet")
    truth = pq.read_table(session / "truth.parquet")
    status = np.asarray(alignment["decode_status"].to_pylist())
    dup = np.asarray(alignment["is_duplicate"].to_pylist(), dtype=bool)
    engine_idx = np.asarray(alignment["engine_frame_idx"].to_pylist(), dtype=np.int64)

    truth_idx = np.asarray(truth["frame_idx"].to_pylist(), dtype=np.int64)
    base = truth_idx[0]
    keys = np.stack(
        [np.asarray(truth[k].to_pylist(), dtype=bool) for k in KEY_ORDER], axis=1
    )

    cap = cv2.VideoCapture(str(session / "video.mkv"))
    geometry = scale_geometry(panel)
    deltas, y_true = [], []
    for video_frame in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ok, frame = cap.read()
        if not ok:
            break
        if status[video_frame] != "ok" or dup[video_frame]:
            continue
        row = engine_idx[video_frame] - base
        if row < 0 or row >= len(keys):
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        deltas.append(cell_deltas(gray, geometry))
        y_true.append(keys[row])
    cap.release()
    return np.stack(deltas), np.stack(y_true), panel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    deltas, y_true, panel = load_session(args.session)
    thresholds = calibrate_threshold(deltas)
    y_pred = decode(deltas, thresholds)

    # A constant lag is correctable; jitter is not. Sweep to find the best
    # shift and report the curve so the choice is visible, not buried.
    sweep = {}
    for shift in range(-4, 5):
        if shift >= 0:
            a, b = y_true[shift:], y_pred[: len(y_pred) - shift or None]
        else:
            a, b = y_true[:shift], y_pred[-shift:]
        sweep[str(shift)] = round(_macro_f1(a, b), 4)
    best_shift = int(max(sweep, key=lambda k: sweep[k]))

    if best_shift >= 0:
        ta, pa = y_true[best_shift:], y_pred[: len(y_pred) - best_shift or None]
    else:
        ta, pa = y_true[:best_shift], y_pred[-best_shift:]

    report = {
        "session": args.session.name,
        "frames_scored": int(len(y_true)),
        "panel_rect_px": list(panel),
        "per_key_delta_threshold": {
            k: round(float(t), 3) for k, t in zip(KEY_ORDER, thresholds)
        },
        "macro_f1_at_zero_shift": round(_macro_f1(y_true, y_pred), 4),
        "shift_sweep_macro_f1": sweep,
        "best_shift_frames": best_shift,
        "macro_f1_at_best_shift": sweep[str(best_shift)],
        "per_key_prf_at_best_shift": prf(ta, pa),
        "onset_timing": onset_err(y_true, y_pred),
        "key_base_rates": {
            k: round(float(y_true[:, i].mean()), 4) for i, k in enumerate(KEY_ORDER)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float))

    print(f"frames scored: {report['frames_scored']}")
    print(f"macro-F1 @0: {report['macro_f1_at_zero_shift']}  "
          f"@best shift {best_shift}: {report['macro_f1_at_best_shift']}")
    for key in KEY_ORDER:
        row = report["per_key_prf_at_best_shift"][key]
        print(f"  {key:6s} P={row['precision']:.3f} R={row['recall']:.3f} "
              f"F1={row['f1']:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
