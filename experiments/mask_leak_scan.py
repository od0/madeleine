"""Scan shard frames for key-correlated luminance outside the declared masks.

Semantics: gameplay pixels legitimately correlate with keys (motion is the
signal a model is supposed to learn), so a whole-frame correlation scan would
always fire. Overlay leakage has a specific signature instead: a *static*
region adjacent to a declared mask rect whose brightness tracks a key's state
almost perfectly. The scan therefore checks, for every declared mask rect,
a margin band around it (outside the rect, scaled to shard resolution), in
small patches, and reports the maximum per-key AUC found in any patch.

Pass criterion: no patch AUC >= 0.90 in any margin band (the 2026-07-26 leak
measured AUC 1.000 at the old rect boundary). In-rect pixels are additionally
asserted to be exactly zero.

Run: uv run python experiments/mask_leak_scan.py
Writes: results/mask_leak_scan.json
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "data" / "shards_v2"
SESSIONS = ROOT / "sessions"
MARGIN = 12          # shard-pixel band width around each rect
PATCH = 4            # patch size inside the band
FAIL_AUC = 0.90


def _auc(values: np.ndarray, labels: np.ndarray) -> float:
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values))
    pos = labels.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[pos].mean() - (n_pos - 1) / 2) / n_neg)


def scan_session(shard_path: Path) -> dict:
    session_id = shard_path.stem
    manifest = json.loads((SESSIONS / session_id / "manifest.json").read_text())
    with np.load(shard_path, allow_pickle=False) as d:
        frames = d["frames"]
        keys = d["keys"].astype(bool)
    side = frames.shape[1]
    res = manifest["capture"]["resolution"]  # [width, height]
    sx, sy = side / res[0], side / res[1]
    gray = frames.mean(axis=3) if frames.ndim == 4 else frames

    report = {"session": session_id, "rects": [], "max_auc": 0.5, "in_rect_max": 0}
    for region in manifest.get("masked_regions", []):
        x, y, w, h = region["rect_px"]
        x0, y0 = int(np.floor(x * sx)), int(np.floor(y * sy))
        x1, y1 = int(np.ceil((x + w) * sx)), int(np.ceil((y + h) * sy))
        x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(side, x1), min(side, y1)
        in_rect_max = int(frames[:, y0c:y1c, x0c:x1c].max()) if (y1c > y0c and x1c > x0c) else 0
        report["in_rect_max"] = max(report["in_rect_max"], in_rect_max)

        bx0, by0 = max(0, x0 - MARGIN), max(0, y0 - MARGIN)
        bx1, by1 = min(side, x1 + MARGIN), min(side, y1 + MARGIN)
        worst = {"auc": 0.5, "key": None, "patch": None}
        for py in range(by0, by1 - PATCH + 1, PATCH):
            for px in range(bx0, bx1 - PATCH + 1, PATCH):
                if x0c <= px and px + PATCH <= x1c and y0c <= py and py + PATCH <= y1c:
                    continue  # inside the rect: covered by the zero assertion
                vals = gray[:, py:py + PATCH, px:px + PATCH].mean(axis=(1, 2))
                for k in range(7):
                    a = _auc(vals, keys[:, k])
                    a = max(a, 1.0 - a)
                    if a > worst["auc"]:
                        worst = {"auc": round(a, 4), "key": int(k),
                                 "patch": [px, py, PATCH, PATCH]}
        report["rects"].append({"name": region.get("name"),
                                "rect_shard_px": [x0c, y0c, x1c - x0c, y1c - y0c],
                                "worst_margin_patch": worst})
        report["max_auc"] = max(report["max_auc"], worst["auc"])
    return report


def main() -> None:
    reports = [scan_session(p) for p in sorted(SHARDS.glob("rec_*.npz"))]
    result = {
        "margin_px": MARGIN, "patch_px": PATCH, "fail_auc": FAIL_AUC,
        "sessions": reports,
        "max_auc_overall": max(r["max_auc"] for r in reports),
        "in_rect_max_overall": max(r["in_rect_max"] for r in reports),
        "pass": all(r["max_auc"] < FAIL_AUC and r["in_rect_max"] == 0
                    for r in reports),
    }
    out = ROOT / "results" / "mask_leak_scan.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in
                      ("max_auc_overall", "in_rect_max_overall", "pass")}))
    if not result["pass"]:
        raise SystemExit("mask leak scan FAILED")


if __name__ == "__main__":
    main()
