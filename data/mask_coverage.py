"""Verify that declared mask rects actually cover the rendered answer keys.

Why this exists: the 2026-07-26 masking audit (report/findings_log.md) found
the input_overlay rect missing the top ~23 px of the rendered key cells. The
builder's masking assertion verified that the *declared* rect was zeroed —
it policed the mask, not the overlay. This module polices the overlay: it
correlates per-pixel brightness with engine-truth key state in a band just
OUTSIDE each declared rect. A rendered widget that leaks past its rect
produces dense key-correlated pixels there; a correct rect leaves only
ordinary gameplay correlation, which is sparse.

The statistic, per masked region: the fraction of band pixels whose
|mean(key down) - mean(key up)| exceeds ``state_thresh`` for ANY key with
enough samples on both sides (the union, not the per-key worst: each key
lights only its own cell, so a leaked cell row moves the union roughly
7x further than any single key). The band is ``max(12, 0.6 * rect_height)``
px wide on every side, clipped to the frame and excluding every declared
rect. The session fails when the union fraction exceeds ``band_frac_max``.

Scope and limits, stated plainly:
- Only KEY-DRIVEN regions (input_overlay, wild_overlay) are checked: the
  mechanism is key correlation, so it can only see widgets that light up
  with keys. The frame-index strip is deliberately NOT checked — its cells
  encode the frame index, which is key-uncorrelated, so a band around it
  measures pure gameplay correlation (measured up to 0.16 on a legitimate
  train session) and would only produce false refusals; the strip's
  coverage evidence is that frame-index decode succeeds inside the
  declared rect. Undeclared widgets far from every rect are also invisible
  to this check.
- An undershoot of 1-2 px can sit under the area threshold; manifests are
  therefore required to carry measured geometry plus margin, and
  build_dataset re-masks with a 1 px dilation at output resolution.
- Gameplay pixels do correlate with keys (that is the research premise), so
  the check is local and area-thresholded rather than "zero correlation
  anywhere". Calibration on the 2026-07 corpus with corrected rects: the
  leaked geometry produces band fractions ~0.3-0.5; every >= 5-minute
  session reads <= 0.031 (worst: rec_20260724_190233, level-geometry
  correlation); the 1-minute fixture reads ~0.10 and is OUT OF ENVELOPE —
  the check needs enough visual diversity to average gameplay correlation
  down, which the >= 5-minute sessions the corpus admits for training
  always have.

Recorded engine-truth sessions only: foreign (mapped) video is masked by
nitrogen's own geometry path and has no per-frame engine truth to correlate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from data.schema import KEY_ORDER

PER_BUCKET = 60
STATE_THRESH = 60.0
BAND_FRAC_MAX = 0.05
MIN_SAMPLES = 20
MIN_BAND_PX = 12
BAND_SCALE = 0.6
KEY_DRIVEN_REGIONS = frozenset({"input_overlay", "wild_overlay"})


def _spread(indices: np.ndarray, count: int) -> np.ndarray:
    if len(indices) <= count:
        return indices
    pick = np.linspace(0, len(indices) - 1, count).round().astype(int)
    return indices[np.unique(pick)]


def _band_mask(
    rects_px: list[tuple[int, int, int, int]],
    rect_index: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Pixels just outside rect ``rect_index``, excluding every declared rect."""
    x0, y0, x1, y1 = rects_px[rect_index]
    reach = max(MIN_BAND_PX, round(BAND_SCALE * (y1 - y0)))
    band = np.zeros((height, width), bool)
    band[
        max(0, y0 - reach) : min(height, y1 + reach),
        max(0, x0 - reach) : min(width, x1 + reach),
    ] = True
    for rx0, ry0, rx1, ry1 in rects_px:
        band[ry0:ry1, rx0:rx1] = False
    return band


def measure_mask_coverage(
    session_dir: str | Path,
    per_bucket: int = PER_BUCKET,
    state_thresh: float = STATE_THRESH,
) -> dict:
    """Sample the session video and measure band statistics per region.

    Returns a report dict; use :func:`verify_mask_coverage` for the
    pass/fail contract.
    """
    root = Path(session_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    regions = manifest["masked_regions"]
    if not regions:
        raise ValueError(f"{root.name}: no masked_regions in manifest")

    truth = pq.read_table(root / "truth.parquet")
    alignment = pq.read_table(root / "alignment.parquet")
    status = np.asarray(alignment["decode_status"].to_pylist())
    dup = np.asarray(alignment["is_duplicate"].to_pylist(), dtype=bool)
    engine_idx = np.asarray(
        alignment["engine_frame_idx"].to_pylist(), dtype=np.int64
    )
    truth_base = int(truth["frame_idx"][0].as_py())
    keys_all = np.stack(
        [np.asarray(truth[k].to_pylist(), dtype=bool) for k in KEY_ORDER],
        axis=1,
    )
    active_all = np.asarray(truth["input_active"].to_pylist(), dtype=bool)

    rows = engine_idx - truth_base
    in_range = (rows >= 0) & (rows < len(keys_all)) & (status == "ok") & (~dup)
    rows_safe = np.clip(rows, 0, len(keys_all) - 1)
    frame_keys = keys_all[rows_safe] & in_range[:, None]
    eligible = active_all[rows_safe] & in_range

    chosen: set[int] = set()
    for k in range(len(KEY_ORDER)):
        chosen.update(
            _spread(np.nonzero(eligible & frame_keys[:, k])[0], per_bucket).tolist()
        )
    chosen.update(
        _spread(
            np.nonzero(eligible & ~frame_keys.any(axis=1))[0], per_bucket
        ).tolist()
    )
    if not chosen:
        raise ValueError(f"{root.name}: no eligible frames to sample")

    cap = cv2.VideoCapture(str(root / "video.mkv"))
    width = height = None
    sum_down = sum_up = None
    n_down = np.zeros(len(KEY_ORDER), np.int64)
    n_up = np.zeros(len(KEY_ORDER), np.int64)
    for video_frame in range(max(chosen) + 1):
        if not cap.grab():
            raise ValueError(f"{root.name}: video ended at frame {video_frame}")
        if video_frame not in chosen:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            raise ValueError(f"{root.name}: retrieve failed at {video_frame}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if sum_down is None:
            height, width = gray.shape
            sum_down = np.zeros((len(KEY_ORDER), height, width), np.float32)
            sum_up = np.zeros((len(KEY_ORDER), height, width), np.float32)
        kv = frame_keys[video_frame]
        for k in range(len(KEY_ORDER)):
            if kv[k]:
                sum_down[k] += gray
                n_down[k] += 1
            else:
                sum_up[k] += gray
                n_up[k] += 1
    cap.release()
    assert sum_down is not None and width is not None and height is not None

    # Masking uses rect_norm (build_dataset applies rect_norm x frame dims),
    # so coverage must be judged on the same pixels the mask will remove.
    rects_px = []
    for region in regions:
        nx0, ny0, nx1, ny1 = region["rect_norm"]
        rects_px.append(
            (
                int(nx0 * width),
                int(ny0 * height),
                int(np.ceil(nx1 * width)),
                int(np.ceil(ny1 * height)),
            )
        )

    report: dict = {
        "session_id": root.name,
        "frame_px": [width, height],
        "state_thresh": state_thresh,
        "samples_per_key_down": {
            name: int(n_down[k]) for k, name in enumerate(KEY_ORDER)
        },
        "regions": [],
    }
    correlated = {}
    for k, name in enumerate(KEY_ORDER):
        if n_down[k] < MIN_SAMPLES or n_up[k] < MIN_SAMPLES:
            continue
        diff = np.abs(sum_down[k] / n_down[k] - sum_up[k] / n_up[k])
        correlated[name] = diff > state_thresh
    report["keys_with_enough_samples"] = sorted(correlated)
    any_key = np.zeros((height, width), bool)
    for mask in correlated.values():
        any_key |= mask

    for index, region in enumerate(regions):
        if region["name"] not in KEY_DRIVEN_REGIONS:
            continue
        band = _band_mask(rects_px, index, width, height)
        band_px = int(band.sum())
        per_key = {
            name: round(float(mask[band].mean()), 5) if band_px else 0.0
            for name, mask in correlated.items()
        }
        union = float(any_key[band].mean()) if band_px else 0.0
        report["regions"].append(
            {
                "name": region["name"],
                "rect_px_from_norm": list(rects_px[index]),
                "band_px": band_px,
                "band_key_correlated_fraction": per_key,
                "band_any_key_fraction": union,
            }
        )
    return report


def coverage_violations(
    report: dict, band_frac_max: float = BAND_FRAC_MAX
) -> list[str]:
    """Derive the pass/fail contract from a measured report.

    Split out from :func:`verify_mask_coverage` so a caller that already
    holds the (expensive) measurement can judge it without re-sampling.
    """
    violations = []
    for region in report["regions"]:
        union = region["band_any_key_fraction"]
        if union > band_frac_max:
            violations.append(
                f"masked region {region['name']!r} does not cover the rendered "
                f"widget: {union:.1%} of the pixels in the band just outside "
                f"the declared rect correlate with key state (threshold "
                f"{band_frac_max:.1%}); the rect geometry is wrong or a "
                "widget leaks past it"
            )
    return violations


def verify_mask_coverage(
    session_dir: str | Path,
    per_bucket: int = PER_BUCKET,
    state_thresh: float = STATE_THRESH,
    band_frac_max: float = BAND_FRAC_MAX,
) -> list[str]:
    """Return one string per coverage violation; an empty list means covered."""
    report = measure_mask_coverage(session_dir, per_bucket, state_thresh)
    return coverage_violations(report, band_frac_max)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_dir")
    parser.add_argument("--report", action="store_true",
                        help="print the full measurement report as JSON")
    args = parser.parse_args(argv)
    if args.report:
        print(json.dumps(measure_mask_coverage(args.session_dir), indent=2))
        return 0
    violations = verify_mask_coverage(args.session_dir)
    for violation in violations:
        print(violation, file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
