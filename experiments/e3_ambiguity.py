"""E3 — state ambiguity and the observation horizon (the deep result).

Question: how often does the same game state correspond to *different* actions,
and how much does future context resolve that ambiguity? Engine truth makes
this measurable: state (position, speed, dash count, stamina, on-ground) and
action are both exact.

Method, within each room (position is only comparable within a room):
1. Normalize the state vector; build a neighbour index over input_active frames.
2. For each frame, find frames within tolerance eps whose action DIFFERS.
   Ambiguity rate(eps) = fraction of frames with >=1 such near-state, diff-action
   neighbour. This is irreducible uncertainty from the present frame alone.
3. For ambiguous pairs, measure how their FUTURES diverge at horizon H — if the
   future state separates them, the action was recoverable from future frames.
   This is the mechanism behind E1's causal gap.

Usage: uv run python -m experiments.e3_ambiguity --sessions sessions/A sessions/B --out results/e3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from data.schema import KEY_ORDER

# State dims used for "same state"; position/speed continuous, dash/stamina/ground.
STATE = ["pos_x", "pos_y", "speed_x", "speed_y", "dash_count", "stamina"]
# Characteristic scales (Celeste units) to normalize each dim to ~O(1).
SCALE = {"pos_x": 8.0, "pos_y": 8.0, "speed_x": 60.0, "speed_y": 60.0,
         "dash_count": 1.0, "stamina": 20.0}  # 8px ~ one tile eighth; 60 ~ run speed


def load(sessions: list[Path]):
    rooms: dict[str, list] = {}
    for s in sessions:
        t = pq.read_table(s / "truth.parquet").to_pydict()
        n = len(t["frame_idx"])
        for i in range(n):
            if not t["input_active"][i] or t["on_ground"][i] is None:
                continue
            px = t["pos_x"][i]
            if px is None or px != px:  # NaN (player absent)
                continue
            rid = t["room_id"][i]
            key = (s.name, rid)
            rooms.setdefault(key, {"state": [], "action": [], "ground": [], "seq": []})
            rooms[key]["state"].append([t[d][i] for d in STATE])
            rooms[key]["action"].append(tuple(int(t[k][i]) for k in KEY_ORDER))
            rooms[key]["ground"].append(int(t["on_ground"][i]))
            rooms[key]["seq"].append((s.name, t["frame_idx"][i]))
    return rooms


def future_state_lookup(sessions: list[Path]):
    """(session, frame_idx) -> normalized state vector, for future-divergence."""
    table = {}
    for s in sessions:
        t = pq.read_table(s / "truth.parquet").to_pydict()
        for i, f in enumerate(t["frame_idx"]):
            px = t["pos_x"][i]
            if px is None or px != px:
                table[(s.name, f)] = None
            else:
                table[(s.name, f)] = np.array([t[d][i] / SCALE[d] for d in STATE])
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-gap", type=int, default=60,
                    help="min frame-index gap between paired frames (exclude "
                         "same-trajectory adjacency; 60 = 1s at 60Hz)")
    args = ap.parse_args()

    rooms = load(args.sessions)
    futures = future_state_lookup(args.sessions)
    eps_grid = [0.25, 0.5, 1.0, 2.0, 4.0]  # normalized-state radius
    horizons = [1, 2, 4, 8, 16]

    total = sum(len(r["state"]) for r in rooms.values())
    amb_counts = {e: 0 for e in eps_grid}
    # future divergence: for ambiguous pairs at eps=1.0, mean future L2 vs horizon
    div_by_h = {h: [] for h in horizons}
    n_pairs = 0

    for key, r in rooms.items():
        if len(r["state"]) < 20:
            continue
        X = np.array([[v / SCALE[d] for v, d in zip(s, STATE)] for s in r["state"]])
        # on_ground must match exactly (different ground state = different regime)
        ground = np.array(r["ground"])
        actions = r["action"]
        tree = cKDTree(X)
        for e in eps_grid:
            pairs = tree.query_pairs(e, output_type="ndarray")
            if len(pairs) == 0:
                continue
            seen_ambiguous = set()
            fidx = [sq[1] for sq in r["seq"]]
            for a, b in pairs:
                if ground[a] != ground[b]:
                    continue
                # exclude same-trajectory adjacency: require a real time gap so
                # pairs are independent visits to a similar state.
                if abs(fidx[a] - fidx[b]) < args.min_gap:
                    continue
                if actions[a] != actions[b]:
                    seen_ambiguous.add(a); seen_ambiguous.add(b)
                    if e == 1.0 and n_pairs < 200000:
                        n_pairs += 1
                        # future divergence at horizons
                        for h in horizons:
                            fa = futures.get((r["seq"][a][0], r["seq"][a][1] + h))
                            fb = futures.get((r["seq"][b][0], r["seq"][b][1] + h))
                            if fa is not None and fb is not None:
                                div_by_h[h].append(float(np.linalg.norm(fa - fb)))
            amb_counts[e] += len(seen_ambiguous)

    report = {
        "sessions": [s.name for s in args.sessions],
        "total_active_frames": total,
        "rooms_analyzed": len([k for k, r in rooms.items() if len(r["state"]) >= 20]),
        "ambiguity_rate_vs_eps": {
            f"eps_{e}": round(amb_counts[e] / total, 4) for e in eps_grid
        },
        "ambiguous_pairs_sampled": n_pairs,
        "future_divergence_vs_horizon": {
            f"h_{h}": {"mean_L2": round(float(np.mean(div_by_h[h])), 3),
                       "median_L2": round(float(np.median(div_by_h[h])), 3),
                       "n": len(div_by_h[h])}
            for h in horizons if div_by_h[h]
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "total_active_frames": total,
        "rooms": report["rooms_analyzed"],
        "ambiguity_rate_vs_eps": report["ambiguity_rate_vs_eps"],
        "future_divergence_vs_horizon": {
            k: v["median_L2"] for k, v in report["future_divergence_vs_horizon"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
