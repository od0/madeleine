# Input overlay spec (frozen v1)

The mod renders a machine-readable 7-key input overlay so an overlay parser
can be validated against engine truth (that validation is recorded in
`results/e4_5min_full.json`). Like the frame-index strip, it is a masked
region: no model may ever see it.

Implementing code: renderer `granny/InputTruth/Source/InputOverlay.cs` with
geometry constants in `granny/InputTruth/Source/InputOverlayGeometry.cs`;
parser `harvest/overlay_parser.py`; tests `tests/test_overlay_parser.py`.

## Geometry (logical 1920x1080 game-window coordinates)

- Backing bar: solid black, at bottom-left, rect (0, 1032, 416, 48)
  (x, y, w, h). That is: height 48, flush with the bottom edge.
- Quiet zone: 8 px inside the bar on all sides.
- Seven cells, one per key, in the canonical order left, right, up, down,
  jump, dash, grab ([specs/session_format.md](session_format.md) canonical
  key order, `data/schema.py` `KEY_ORDER`). Cell i occupies
  rect (8 + i*56 + 8, 1040, 40, 32): 56 px pitch, 40 px wide, 32 px tall,
  vertically centered in the bar.
- Cell fill: WHITE when the key is down this engine frame, DARK GRAY
  (0x28,0x28,0x28) when up. Hard edges, no anti-aliasing, no text.
- The same engine-frame state that is written to the CSV row must drive the
  cells — one source of truth per frame, same as the strip.

## Mask rect

Exposed by the module as constants (like the strip's,
`InputTruthModule.InputOverlayMaskX/Y/Width/Height`): (0, 1032, 416, 48)
logical. Manifests record it as masked region name "input_overlay".

**Correction (2026-07-27).** The constants above are the DRAW-call
coordinates, and they are what the mod submits. But the deployed render
pass does not place them at logical x canvas-scale on every rig: on the
built-in-display captures the measured vertical mapping is
`y_px ~ 0.856*y_logical + 5` against the assumed `0.891*y_logical`,
so the rendered bar sits ~33 logical px higher than the constants and is
correspondingly taller; horizontal placement matches exactly. On the
external display the render matches the constants. Consequences:

- Manifest mask rects are no longer derived from these constants alone.
  The assembler declares a superset rect (logical y 984 to the capture
  bottom, 436 wide) and `data.mask_coverage` verifies actual coverage
  against engine truth before any shard is built.
- The parser contract below is valid only where the render matches the
  constants (validated on the external-display rig,
  `results/e4_5min_full.json` and `results/e4_15min.json`). Decoding
  built-in-display captures requires measured geometry
  (experiments/measure_overlay_geometry.py).

## Parser contract (harvest/overlay_parser.py)

Per-key ROI = the cell rect scaled by (capture_width / 1920,
capture_height / 1080); the implementation scales both boundary edges and
rounds, so adjacent rects stay adjacent at any capture size. Per frame:

- Dark reference: mean gray level over the four non-overlapping bands that
  form the bar's 8 px quiet zone (`QUIET_ZONE_RECTS`).
- White reference: maximum gray level within the whole bar rect.
- Contrast abstention: when `white − dark < MIN_CONTRAST` (80.0 in
  `harvest/overlay_parser.py`), the parser returns null for **all seven keys**
  on that frame instead of guessing. Black-to-white contrast is nominally 255
  and black-to-up-cell contrast is 40 (0x28), so a frame below the gate cannot
  supply a trustworthy white reference.
- Otherwise: threshold at the midpoint of the dark and white references; each
  key's sample is the mean of the central 50% of its scaled cell rect;
  mean ≥ threshold ⇒ pressed. No OCR.

`parse_video` writes one row per video frame with nullable boolean key columns
(`PARSED_OVERLAY_SCHEMA`) and refuses to write a file named `truth.parquet`:
parsed-overlay output is inferred supervision, never engine truth.

`tests/test_overlay_parser.py` pins the geometry constants, exact decode on
clean frames, precision/recall through a transcode, the low-contrast null
path, and the nullable output schema.

## Related overlay, out of scope here: translucent calibration HUD

The mod can additionally render a translucent, labelled, wild-style HUD
(`granny/InputTruth/Source/WildOverlay.cs`) for calibrating a decoder for
alpha-blended third-party overlays. It is not part of this spec: it is off by
default, declared per session in the mod's `meta.json`
(`wild_overlay_rect_logical`), masked under the region name `wild_overlay`,
and decoded by `harvest/translucent_parser.py` using local cell-minus-ring
contrast with per-key thresholds.

## Note (2026-07-26): mask-coverage defect in session manifests

A masking defect disclosed in `report/findings_log.md` (2026-07-26 entry)
involves this overlay: the capture-pixel `input_overlay` mask rectangles
recorded in session manifests undershot the overlay cells actually rendered in
captured frames by a static ~23 px vertically, leaving a readable sliver of
the key cells in own-data training shards (the `wild_overlay` rect similarly
missed the top of the translucent HUD on the calibration session). Per that
entry, this was a mask-coverage failure in manifest geometry — how
capture-pixel rects were derived for the manifests — not an error in the
logical overlay-rectangle definition in this spec, and the builder's zero
check verified only the declared rects. No transferable benefit was observed
on held-out sessions, but this does not rule out training distortion.
Resolved 2026-07-27 (findings log, "The overlay mask geometry is fixed"):
manifests carry measured per-family rects, `data.mask_coverage` scans the
band outside each key-driven rect for key-state-dependent luminance before
any shard is built, and the own-data shards are rebuilt and probed clean;
own-data model re-runs remain queued. The render-transform measurement
behind the corrected rects is in the "Correction (2026-07-27)" section
above.
