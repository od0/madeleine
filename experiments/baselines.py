"""Reference baselines — no score in this project is reported without them.

Per-frame AP:
  chance: per-key prevalence (an uninformative constant-score classifier's AP).
  persistence: predict keys[t] with keys[t-1]. Celeste inputs are held for
  16-40 frames at 60Hz, so this trivial rule is very strong (0.91 macro-AP);
  any model given true action history is competing with it, not with chance.

Transition-event F1 (primary metric per brief v3.1):
  persistence: its every event is an echo one frame late — exactly 0 at
  collar 0, saturating toward 1 at collar ≥ 1. That contrast is why collar 0
  is the primary setting: loose collars re-admit the autocorrelation shortcut.
  chance: a constant-probability stream never crosses a threshold, so it
  produces no events at all — event F1 is 0 by construction.
  shuffled: the true number of events placed uniformly at random over active
  frames (10 seeds, mean) — the luck anchor for a given event rate.

Per-frame rows are evaluated on input_active frames only; event rows are
computed on the contiguous stream with active-gated event times, matching
badeline.eval's surfaces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from badeline.metrics import (
    match_event_counts,
    per_key_transition_f1,
    score_events,
    transition_events,
)
from data.schema import KEY_ORDER

EVENT_COLLARS = (0, 1, 2)
SHUFFLE_SEEDS = 10


def _persistence_probs(keys: np.ndarray) -> np.ndarray:
    probs = np.zeros_like(keys, dtype=float)
    probs[1:] = keys[:-1].astype(float)
    return probs


def _shuffled_event_f1(
    keys: np.ndarray, active: np.ndarray, collar: int, rng: np.random.Generator
) -> dict[str, float]:
    """Mean event F1 of uniformly-placed events at the true per-key rate."""
    active_frames = np.flatnonzero(active)
    out: dict[str, float] = {}
    for i, key in enumerate(KEY_ORDER):
        true_on, true_off = transition_events(keys[:, i])
        true_on, true_off = true_on[active[true_on]], true_off[active[true_off]]
        n_on, n_off = len(true_on), len(true_off)
        if n_on + n_off == 0 or len(active_frames) == 0:
            out[key] = float("nan")
            continue
        scores = []
        for _ in range(SHUFFLE_SEEDS):
            rand_on = np.sort(rng.choice(active_frames, size=n_on, replace=False))
            rand_off = np.sort(rng.choice(active_frames, size=n_off, replace=False))
            matched = (match_event_counts(true_on, rand_on, collar)
                       + match_event_counts(true_off, rand_off, collar))
            scores.append(
                score_events(n_on + n_off, n_on + n_off, matched)["f1"]
            )
        out[key] = float(np.mean(scores))
    return out


def baselines(shard: Path) -> dict:
    with np.load(shard) as z:
        keys = z["keys"].astype(bool)
        active = (z["input_active"].astype(bool) if "input_active" in z
                  else np.ones(len(keys), dtype=bool))
    k = keys[active]
    chance, pers = {}, {}
    for i, key in enumerate(KEY_ORDER):
        y = k[:, i]
        chance[key] = float(y.mean())
        pers[key] = (float(average_precision_score(y[1:], y[:-1].astype(float)))
                     if y[1:].any() else float("nan"))

    pers_probs = _persistence_probs(keys)
    pers_events = {
        str(collar): {
            key: r["event"]["f1"]
            for key, r in per_key_transition_f1(
                keys, pers_probs, threshold=0.5, collar=collar, active=active
            ).items()
        }
        for collar in EVENT_COLLARS
    }
    rng = np.random.default_rng(0)
    shuffled_events = {
        str(collar): _shuffled_event_f1(keys, active, collar, rng)
        for collar in EVENT_COLLARS
    }

    def macro(d: dict[str, float]) -> float:
        return float(np.nanmean(list(d.values())))

    return {
        "shard": shard.name, "active_frames": int(active.sum()),
        "chance_ap_per_key": chance,
        "chance_ap_macro": float(np.mean(list(chance.values()))),
        "persistence_ap_per_key": pers,
        "persistence_ap_macro": float(np.nanmean(list(pers.values()))),
        "chance_event_f1": 0.0,
        "persistence_event_f1_per_key": pers_events,
        "persistence_event_f1_macro": {
            c: macro(d) for c, d in pers_events.items()
        },
        "shuffled_event_f1_per_key": shuffled_events,
        "shuffled_event_f1_macro": {
            c: macro(d) for c, d in shuffled_events.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = [baselines(s) for s in args.shards]
    for b in out:
        pers_ev = b["persistence_event_f1_macro"]
        print(f"{b['shard']}: n={b['active_frames']} "
              f"chance_ap={b['chance_ap_macro']:.3f} "
              f"persistence_ap={b['persistence_ap_macro']:.3f} | "
              f"event_f1: chance=0.000 "
              f"persistence@c0={pers_ev['0']:.3f} @c1={pers_ev['1']:.3f} "
              f"shuffled@c0={b['shuffled_event_f1_macro']['0']:.3f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
