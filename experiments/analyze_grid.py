"""Aggregate the E1/E2 grid: per-config mean±std over seeds, headline gaps.

Reads results/grid/<config>_s<seed>.json (written by badeline.eval). Primary
metric per brief v3.1: transition-event (onset+offset) F1 at collar 0 with
per-key oracle thresholds — per-frame scores at 60Hz are autocorrelation-
dominated (persistence: 0.912 per-frame AP, exactly 0 event F1 at collar 0).
Per-key AP is the secondary table. Both read from the input_active_only
variant (the honest evaluation surface). val-B evals (<name>__valB.json) are
aggregated separately and never pooled with val-A.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from data.schema import KEY_ORDER

GRID = Path(sys.argv[1] if len(sys.argv) > 1 else "results/grid")
E1 = [("2-frame", "e1_2frame"), ("16-noncausal", "e1_16frame_noncausal"),
      ("16-past-only", "e1_16frame_pastonly")]
# E2 pixels bar == e1_16frame_noncausal (same config); reuse it.
E2 = [("pixels", "e1_16frame_noncausal"), ("history", "e2_history"),
      ("history-gap30", "e2_history_gap30"),
      ("pixels+history", "e2_pixels_history"),
      ("px+hist-gap30", "e2_pixels_history_gap30")]


def _macro(per_key: dict[str, float]) -> float:
    vals = [v for v in per_key.values() if v is not None and v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def _extract(report: dict) -> dict:
    """Pull the two metric families for one run, or None when absent."""
    m = report["input_active_only"]["metrics"]
    out = {"ap": m["per_key_ap"]}
    if "transition_f1_oracle" in m:
        out["tf1"] = {
            k: r["event"]["f1"] for k, r in m["transition_f1_oracle"].items()
        }
        out["tf1_thresholds"] = {
            k: r["threshold"] for k, r in m["transition_f1_oracle"].items()
        }
    return out


def load() -> tuple[dict, dict]:
    """Group runs by config, val-A and val-B separately."""
    val_a: dict[str, list[dict]] = defaultdict(list)
    val_b: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(GRID.glob("*.json")):
        if p.stem.endswith("_preds"):
            continue
        stem, _, suffix = p.stem.partition("__")
        cfg = stem.rsplit("_s", 1)[0]
        rep = json.loads(p.read_text())
        (val_b if suffix == "valB" else val_a)[cfg].append(_extract(rep))
    return val_a, val_b


def agg(runs: list[dict], family: str) -> dict:
    """mean±std per key and macro over seed replicates, for one metric family."""
    per_key_dicts = [r[family] for r in runs if family in r]
    if not per_key_dicts:
        return {}
    out = {}
    for k in KEY_ORDER:
        vals = [d[k] for d in per_key_dicts if d.get(k) is not None and d[k] == d[k]]
        if vals:
            out[k] = (statistics.mean(vals),
                      statistics.stdev(vals) if len(vals) > 1 else 0.0)
    macros = [_macro(d) for d in per_key_dicts]
    out["_macro"] = (statistics.mean(macros),
                     statistics.stdev(macros) if len(macros) > 1 else 0.0)
    out["_n"] = len(per_key_dicts)
    return out


def fmt(cell) -> str:
    return f"{cell[0]:.3f}±{cell[1]:.3f}" if cell else "   —   "


def table(title: str, cfgs: list[tuple[str, str]], agged: dict) -> None:
    present = [(name, cfg) for name, cfg in cfgs if agged.get(cfg)]
    if not present:
        return
    print(f"\n=== {title} ===")
    print("key".ljust(8) + "".join(name.ljust(16) for name, _ in present))
    for k in KEY_ORDER + ["_macro"]:
        row = k.ljust(8)
        for _, cfg in present:
            row += fmt(agged.get(cfg, {}).get(k)).ljust(16)
        print(row)


def analyze(surface: str, runs: dict[str, list[dict]]) -> dict:
    if not runs:
        return {}
    print(f"\n################ {surface} ################")
    print("configs: " + ", ".join(f"{c}(n={len(runs[c])})" for c in sorted(runs)))

    tf1 = {cfg: agg(lst, "tf1") for cfg, lst in runs.items()}
    ap = {cfg: agg(lst, "ap") for cfg, lst in runs.items()}

    table("E1 causality — transition-event F1, collar 0, oracle thresholds "
          "(PRIMARY)", E1, tf1)
    table("E2 blindfold — transition-event F1, collar 0, oracle thresholds "
          "(PRIMARY)", E2, tf1)
    table("E1 causality — per-key AP (secondary)", E1, ap)
    table("E2 blindfold — per-key AP (secondary)", E2, ap)

    def macro(store, cfg):
        return store.get(cfg, {}).get("_macro", (float("nan"), 0))[0]

    if any(tf1.get(cfg) for _, cfg in E1):
        nc, twof, past = (macro(tf1, "e1_16frame_noncausal"),
                          macro(tf1, "e1_2frame"),
                          macro(tf1, "e1_16frame_pastonly"))
        print("\n--- headline (transition-event F1) ---")
        print(f"E1 causal gap (16-noncausal − 16-past-only): {nc - past:+.3f}")
        print(f"E1 window gap  (16-noncausal − 2-frame):     {nc - twof:+.3f}")
    return {"transition_f1_oracle_collar0": tf1, "ap": ap}


def main() -> None:
    val_a, val_b = load()
    if not val_a and not val_b:
        print("no grid results yet at", GRID)
        return
    missing_tf1 = sorted(
        cfg for cfg, lst in {**val_a, **val_b}.items()
        if any("tf1" not in r for r in lst)
    )
    if missing_tf1:
        print("WARNING: runs missing transition metrics (old eval format, "
              f"re-eval needed): {', '.join(missing_tf1)}")
    summary = {
        "val_a": analyze("val-A", val_a),
        "val_b": analyze("val-B (unseen chapter; 33% render drops — "
                         "windowed comparisons caveated)", val_b),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/grid_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nwrote results/grid_summary.json")


if __name__ == "__main__":
    main()
