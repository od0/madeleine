"""Pixels-only diagnostic: pipeline fault vs distribution shift vs memorization.

Grid v1/v2 read pixels-only arms at ≈ chance per-frame AP on val-A. Direction
keys are visually inferable from sprite motion in any room, so chance-level
pixels demands a diagnosis before any E1/E2 conclusion. Orchestrator-owned:
this script exists to attribute blame, and blame attribution is a judgment
call.

Two modes:

data — model-free pipeline probes over shards:
  * alignment: phase-correlation global dx per frame pair, cross-correlated
    against (right − left) at lags −8..+8 (peak far from 0 ⇒ frame↔label
    misalignment); motion-energy event-triggered average around jump onsets.
  * contiguity: fraction of 16-frame windows whose engine_frame_idx is
    strictly consecutive (drop-gap contamination rate for windowed configs).
  * masks: bounding boxes of pixels that are zero across every frame.
  * geometry: per-session mean image montage (the mid-project capture
    geometry change made 2560×1440, 1710×1112 and 1710×962 sessions; all are
    squashed to 128×128, so the game canvas lands differently per group).

probe — model evals per session (never pooled):
  * evaluates a checkpoint state (final = memorization story, best = the
    selected model) on named sessions one at a time via badeline.eval;
  * writes results/diag/<run>__<state>__<session>.json;
  * renders prediction-vs-truth spot-check overlays for eyeballing.

Diagnostic fork (from the session-5 boot mandate): good on train sessions but
chance across sessions ⇒ memorization / distribution shift; chance everywhere
⇒ pipeline fault, suspects in order: frame↔label alignment, masking rects,
transforms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from badeline.eval import evaluate
from badeline.model import BadelineIDM
from data.schema import KEY_ORDER

LAGS = range(-8, 9)


def _load_shard(data_dir: Path, sid: str) -> dict[str, np.ndarray]:
    with np.load(data_dir / f"{sid}.npz") as z:
        return {name: z[name] for name in z.files}


# ---------------- data mode ----------------

def _phase_dx(frames: np.ndarray) -> np.ndarray:
    """Global horizontal shift between consecutive frames (camera scroll)."""
    dx = np.zeros(len(frames), dtype=np.float32)
    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY).astype(np.float32)
    for i in range(1, len(frames)):
        cur = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        (shift_x, _), _ = cv2.phaseCorrelate(prev, cur)
        dx[i] = shift_x
        prev = cur
    return dx


def _lag_correlation(signal: np.ndarray, reference: np.ndarray) -> dict:
    """Pearson r of signal[t] vs reference[t+lag]; both zero-meaned."""
    s = signal - signal.mean()
    out = {}
    for lag in LAGS:
        if lag >= 0:
            a, b = s[: len(s) - lag or None], reference[lag:]
        else:
            a, b = s[-lag:], reference[: lag]
        b = b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        out[str(lag)] = float((a * b).sum() / denom) if denom > 0 else 0.0
    return out


def _event_triggered(signal: np.ndarray, events: np.ndarray, half: int = 8) -> list[float]:
    rows = [signal[e - half : e + half + 1] for e in events
            if e - half >= 0 and e + half + 1 <= len(signal)]
    return np.mean(rows, axis=0).tolist() if rows else []


def _contiguity(engine_idx: np.ndarray, window: int = 16) -> float:
    d = np.diff(engine_idx) == 1
    ok = np.convolve(d.astype(int), np.ones(window - 1, dtype=int), "valid")
    return float((ok == window - 1).mean()) if len(ok) else float("nan")


def _zero_regions(frames: np.ndarray) -> list[list[int]]:
    """Bounding boxes of pixels that are zero in every sampled frame."""
    always_zero = (frames[:: max(1, len(frames) // 200)].max(axis=(0, 3)) == 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        always_zero.astype(np.uint8), connectivity=4
    )
    boxes = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area >= 16:  # ignore stray dark pixels
            boxes.append([int(x), int(y), int(w), int(h), int(area)])
    return boxes


def run_data(args: argparse.Namespace) -> None:
    report, means = {}, []
    for sid in args.sessions:
        shard = _load_shard(args.data, sid)
        frames, keys = shard["frames"], shard["keys"].astype(bool)
        engine_idx = shard["engine_frame_idx"]
        left = keys[:, KEY_ORDER.index("left")].astype(np.float32)
        right = keys[:, KEY_ORDER.index("right")].astype(np.float32)
        jump = keys[:, KEY_ORDER.index("jump")]

        dx = _phase_dx(frames)
        lr = right - left
        motion = np.zeros(len(frames), dtype=np.float32)
        for start in range(1, len(frames), 2000):
            stop = min(start + 2000, len(frames))
            block = frames[start:stop].astype(np.int16)
            prev_block = frames[start - 1 : stop - 1].astype(np.int16)
            motion[start:stop] = np.abs(block - prev_block).mean(axis=(1, 2, 3))

        jump_prev = np.empty_like(jump)
        jump_prev[0] = False
        jump_prev[1:] = jump[:-1]
        jump_onsets = np.flatnonzero(jump & ~jump_prev)

        corr = _lag_correlation(dx, lr)
        peak = max(corr, key=lambda k: abs(corr[k]))
        report[sid] = {
            "n_frames": int(len(frames)),
            "dx_vs_rightminusleft_lag_r": corr,
            "dx_peak_lag": int(peak),
            "dx_peak_r": corr[peak],
            "motion_around_jump_onset_lag-8..8": _event_triggered(motion, jump_onsets),
            "n_jump_onsets": int(len(jump_onsets)),
            "window16_contiguous_fraction": _contiguity(engine_idx),
            "zero_region_boxes_xywh_area": _zero_regions(frames),
        }
        mean_img = frames[:: max(1, len(frames) // 300)].mean(axis=0).astype(np.uint8)
        means.append((sid, mean_img))
        print(f"{sid}: dx↔(R−L) peak r={corr[peak]:+.3f} at lag {peak}; "
              f"16f-contiguous {report[sid]['window16_contiguous_fraction']:.1%}; "
              f"masks {len(report[sid]['zero_region_boxes_xywh_area'])} regions")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    tile_h, tile_w, pad = 160, 160, 24
    canvas = np.full((tile_h + 2 * pad, len(means) * (tile_w + pad) + pad, 3),
                     255, dtype=np.uint8)
    for i, (sid, img) in enumerate(means):
        x0 = pad + i * (tile_w + pad)
        canvas[pad : pad + tile_h, x0 : x0 + tile_w] = cv2.resize(
            img, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST
        )
        cv2.putText(canvas, sid[4:], (x0, pad - 8), cv2.FONT_HERSHEY_PLAIN,
                    0.9, (0, 0, 0), 1, cv2.LINE_AA)
    fig = Path("results/figures/diag_mean_frames.png")
    fig.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(fig), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"wrote {out} and {fig}")


# ---------------- probe mode ----------------

def _load_model(run_dir: Path, state: str) -> tuple[BadelineIDM, dict]:
    config = json.loads((run_dir / "config.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    key = "final_state_dict" if state == "final" else "model_state_dict"
    model = BadelineIDM(config)
    model.load_state_dict(ckpt[key])
    return model, config


def _overlay(frame: np.ndarray, truth: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """One spot-check tile: frame upscaled, per-key truth box + prob bar."""
    tile = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
    strip = np.full((64, 256, 3), 24, dtype=np.uint8)
    cell = 256 // len(KEY_ORDER)
    for i, key in enumerate(KEY_ORDER):
        x0 = i * cell
        color = (80, 220, 80) if truth[i] else (70, 70, 70)
        cv2.rectangle(strip, (x0 + 2, 2), (x0 + cell - 3, 18), color,
                      -1 if truth[i] else 1)
        bar = int(prob[i] * 40)
        cv2.rectangle(strip, (x0 + 2, 62 - bar), (x0 + cell - 3, 62),
                      (60, 140, 255), -1)
        cv2.putText(strip, key[:2], (x0 + 3, 32), cv2.FONT_HERSHEY_PLAIN,
                    0.8, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([tile, strip])


def run_probe(args: argparse.Namespace) -> None:
    model, config = _load_model(args.run, args.state)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path("results/diag")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run.name

    window = int(config.get("window", 2))
    batch = 256 if window <= 2 else 64
    for sid in args.sessions:
        preds_path = out_dir / f"{run_name}__{args.state}__{sid}_preds.npz"
        report = evaluate(model, config, args.data, [sid], device,
                          batch_size=batch, preds_out=preds_path)
        report["run"], report["state"], report["session"] = run_name, args.state, sid
        out = out_dir / f"{run_name}__{args.state}__{sid}.json"
        out.write_text(json.dumps(report, indent=2))
        m = report["input_active_only"]["metrics"]
        ap = {k: round(v, 3) for k, v in m["per_key_ap"].items()}
        tf1 = {k: round(r["event"]["f1"], 3)
               for k, r in m["transition_f1_oracle"].items()}
        print(f"{run_name}[{args.state}] on {sid}:")
        print(f"  AP        {ap}")
        print(f"  eventF1@0 {tf1}")

        if args.overlays and sid == args.sessions[-1]:
            with np.load(preds_path) as z:
                y_true, y_prob = z["y_true"].astype(bool), z["y_prob"]
                active = z["input_active"].astype(bool)
            shard = _load_shard(args.data, sid)
            offset = (int(config.get("window", 2)) - 1) // 2 \
                if config.get("window_mode", "centered") == "centered" \
                else int(config.get("window", 2)) - 1
            rng = np.random.default_rng(0)
            dir_cols = [KEY_ORDER.index(k) for k in ("left", "right", "up", "down")]
            onset_pool = np.flatnonzero(
                (y_true[1:, dir_cols] & ~y_true[:-1, dir_cols]).any(axis=1)
                & active[1:]) + 1
            act_pool = np.flatnonzero(y_true.any(axis=1) & active)
            picks = np.concatenate([
                rng.choice(onset_pool, min(12, len(onset_pool)), replace=False),
                rng.choice(act_pool, min(8, len(act_pool)), replace=False),
            ])
            tiles = [_overlay(shard["frames"][p + offset], y_true[p], y_prob[p])
                     for p in sorted(picks)]
            rows = [np.hstack(tiles[i : i + 5]) for i in range(0, len(tiles) - 4, 5)]
            fig = Path(f"results/figures/diag_overlay_{run_name}_{args.state}_{sid}.png")
            cv2.imwrite(str(fig), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
            print(f"  wrote {fig} ({len(tiles)} tiles)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    d = sub.add_parser("data", help="model-free pipeline probes")
    d.add_argument("--data", type=Path, required=True)
    d.add_argument("--sessions", nargs="+", required=True)
    d.add_argument("--out", type=Path, default=Path("results/diag/data_probes.json"))
    d.set_defaults(func=run_data)

    p = sub.add_parser("probe", help="model evals per session + overlays")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--state", choices=("final", "best"), default="final")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--sessions", nargs="+", required=True)
    p.add_argument("--overlays", action="store_true")
    p.set_defaults(func=run_probe)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
