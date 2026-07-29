"""E4 — metrology of overlay-recovered labels vs engine truth.

Leg 1 (this file, so far): validate the classical overlay parser against
engine truth on a recorded session. The parser reads the mod's machine-readable
input overlay off the (unmasked) video; engine truth is the mod's per-frame
log. Reports per-key precision/recall/F1 and onset-timing error, plus the
parser's abstention rate (None on fully-idle frames — it needs a white/pressed
reference to set its threshold; None decodes as all-released by convention).

Legs 2 (label jitter ±1/2/4) and 3 (internet-grade transcode + re-parse) build
on this baseline and are added as separate entry points.

Usage: uv run python -m experiments.e4_metrology --session sessions/<id> --out results/e4_<id>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from data.schema import KEY_ORDER
from harvest.overlay_parser import parse_overlay


def parse_and_align(session: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred, abstained) over ok non-duplicate video frames.

    y_pred rows where the parser abstained (None) are filled all-False and
    flagged in `abstained`.
    """
    al = pq.read_table(session / "alignment.parquet").to_pydict()
    tr = pq.read_table(session / "truth.parquet").to_pydict()
    truth_row = {f: i for i, f in enumerate(tr["frame_idx"])}
    truth_cols = {k: tr[k] for k in KEY_ORDER}

    cap = cv2.VideoCapture(str(session / "video.mkv"))
    vw, vh = int(cap.get(3)), int(cap.get(4))
    y_true, y_pred, abst = [], [], []
    status, dup = al["decode_status"], al["is_duplicate"]
    eng = al["engine_frame_idx"]
    vf = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if status[vf] == "ok" and not dup[vf]:
            ti = truth_row.get(eng[vf])
            if ti is not None:
                pred = parse_overlay(frame, (vw, vh))
                abstained = all(pred[k] is None for k in KEY_ORDER)
                y_pred.append([bool(pred[k]) for k in KEY_ORDER])
                y_true.append([bool(truth_cols[k][ti]) for k in KEY_ORDER])
                abst.append(abstained)
        vf += 1
    cap.release()
    return np.array(y_true), np.array(y_pred), np.array(abst)


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for i, k in enumerate(KEY_ORDER):
        t, p = y_true[:, i], y_pred[:, i]
        tp = int((t & p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if prec == prec and rec == rec and prec + rec else float("nan"))
        out[k] = {"precision": prec, "recall": rec, "f1": f1,
                  "tp": tp, "fp": fp, "fn": fn}
    return out


def onset_err(y_true: np.ndarray, y_pred: np.ndarray, max_lag: int = 8) -> dict:
    """Signed frame offset (pred_onset − true_onset) for matched onsets."""
    out = {}
    for i, k in enumerate(KEY_ORDER):
        t, p = y_true[:, i].astype(np.int8), y_pred[:, i].astype(np.int8)
        t_on = np.flatnonzero(np.diff(t) == 1) + 1
        p_on = np.flatnonzero(np.diff(p) == 1) + 1
        offs = []
        for s in t_on:
            near = p_on[np.abs(p_on - s) <= max_lag]
            if len(near):
                offs.append(int(near[np.argmin(np.abs(near - s))] - s))
        out[k] = {"n_true_onsets": int(len(t_on)), "n_matched": len(offs),
                  "median_offset": float(np.median(offs)) if offs else None,
                  "mean_abs_offset": float(np.mean(np.abs(offs))) if offs else None}
    return out


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = prf(y_true, y_pred)
    vals = [p[k]["f1"] for k in KEY_ORDER if p[k]["f1"] == p[k]["f1"]]
    return float(np.mean(vals)) if vals else float("nan")


def jitter_leg(y_true: np.ndarray) -> dict:
    """Cost of mistimed labels: shift truth by ±k frames, measure agreement
    vs the unshifted truth. Isolates what pure timing error (a known unknown
    in harvested labels) costs, independent of any parser."""
    out = {}
    for k in (1, 2, 4):
        # shift forward by k: labels[t] compared against truth[t-k]
        a, b = y_true[k:], y_true[:-k]
        out[f"shift_{k}"] = round(_macro_f1(b, a), 4)
    return out


def transcode_leg(session: Path, fps: int, bitrate: str, seconds: int,
                  scale_h: int | None = None) -> dict:
    """Internet-grade transcode (low fps + streaming bitrate, optional
    downscale to scale_h — NitroGen refetches at 480p), then re-decode strip
    (for alignment) and re-parse overlay from the degraded video."""
    import subprocess, tempfile
    from theo.frameindex import decode_strip
    src = session / "video.mkv"
    vf = f"scale=-2:{scale_h}" if scale_h else "null"
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / "transcoded.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-t", str(seconds), "-i", str(src),
             "-r", str(fps), "-vf", vf, "-c:v", "libx264", "-b:v", bitrate,
             "-maxrate", bitrate, "-bufsize", bitrate, "-pix_fmt", "yuv420p", str(dst)],
            check=True)
        tr = pq.read_table(session / "truth.parquet").to_pydict()
        truth_row = {f: i for i, f in enumerate(tr["frame_idx"])}
        cols = {k: tr[k] for k in KEY_ORDER}
        cap = cv2.VideoCapture(str(dst))
        vw, vh = int(cap.get(3)), int(cap.get(4))
        rx = (0, 0, round(512 * vw / 1920), round(48 * vh / 1080))
        y_true, y_pred = [], []
        strip_ok = 0; nframes = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            nframes += 1
            canon = cv2.resize(frame[rx[1]:rx[1]+rx[3], rx[0]:rx[0]+rx[2]],
                               (1024, 96), interpolation=cv2.INTER_AREA)
            e = decode_strip(canon)
            if e is None:
                continue
            strip_ok += 1
            ti = truth_row.get(e)
            if ti is None:
                continue
            pred = parse_overlay(frame, (vw, vh))
            y_pred.append([bool(pred[k]) for k in KEY_ORDER])
            y_true.append([bool(cols[k][ti]) for k in KEY_ORDER])
        cap.release()
    yt, yp = np.array(y_true), np.array(y_pred)
    return {
        "profile": f"{fps}fps_{bitrate}" + (f"_{scale_h}p" if scale_h else "_fullres"),
        "transcoded_frames": nframes,
        "strip_decode_rate": round(strip_ok / nframes, 4) if nframes else 0,
        "aligned_frames": len(yt),
        "macro_f1": round(_macro_f1(yt, yp), 4) if len(yt) else None,
        "per_key_recall": {k: round(prf(yt, yp)[k]["recall"], 3) for k in KEY_ORDER} if len(yt) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--transcode-seconds", type=int, default=300)
    args = ap.parse_args()

    y_true, y_pred, abst = parse_and_align(args.session)
    n = len(y_true)
    active = y_true.any(axis=1)
    # exactness on frames where at least one key is truly pressed
    exact_active = float((y_pred[active] == y_true[active]).all(axis=1).mean())
    report = {
        "session": args.session.name,
        "leg1_clean": {
            "frames_evaluated": n,
            "abstention_rate": float(abst.mean()),
            "abstained_frames_that_were_idle": float((~y_true[abst].any(axis=1)).mean()) if abst.any() else None,
            "per_key_prf": prf(y_true, y_pred),
            "onset_timing": onset_err(y_true, y_pred),
            "macro_f1": _macro_f1(y_true, y_pred),
            "exact_match_rate_active_frames": exact_active,
        },
        "leg2_jitter": jitter_leg(y_true),
        "leg3_transcode": [
            transcode_leg(args.session, fps=30, bitrate="1M", seconds=args.transcode_seconds),
            transcode_leg(args.session, fps=30, bitrate="500k", seconds=args.transcode_seconds),
            # downscale sweep: NitroGen refetches source video at 480p, so the
            # overlay occupies far fewer pixels — the real harvested scenario.
            transcode_leg(args.session, fps=30, bitrate="1M", seconds=args.transcode_seconds, scale_h=720),
            transcode_leg(args.session, fps=30, bitrate="1M", seconds=args.transcode_seconds, scale_h=480),
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "leg1_clean_macro_F1": round(report["leg1_clean"]["macro_f1"], 4),
        "leg1_abstention": round(report["leg1_clean"]["abstention_rate"], 4),
        "leg2_jitter_macroF1_by_shift": report["leg2_jitter"],
        "leg3_transcode": [{"profile": t["profile"], "strip_decode_rate": t["strip_decode_rate"],
                            "macro_f1": t["macro_f1"]} for t in report["leg3_transcode"]],
    }, indent=2))


if __name__ == "__main__":
    main()
