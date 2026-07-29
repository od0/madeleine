# Frame-Index Strip — Encoding Spec v1

The rendered frame index is the alignment substrate for all evaluation ground
truth. Both the renderer (granny/InputTruth) and the decoder (theo/frameindex.py)
conform to this document; neither defines the format. Conformance is checked
against `specs/frameindex_test_vectors.json` on both sides.

## Payload

- 24-bit unsigned frame index. The renderer masks the mod's 64-bit engine
  counter to its low 24 bits (`FrameIndexStrip.Pattern`), so the rendered value
  wraps at 2^24 (~77.7 h at 60 Hz) while `truth.parquet` `frame_idx` keeps
  counting. No component handles a wrap: the alignment builders would treat the
  post-wrap index as a backward jump (`is_duplicate` false,
  `preceded_by_drop_count` clamped to 0) and silently misalign, so sessions
  must stay well below 2^24 engine frames.
- 4-bit checksum: XOR of the six 4-bit nibbles of the 24-bit value, MSB-nibble
  first. `checksum = n5 ^ n4 ^ n3 ^ n2 ^ n1 ^ n0` where n5 is bits 23–20.

## Cell layout (left → right)

```
[S1][S0] [D23 … D0] [C3 C2 C1 C0]        30 cells total
```

- S1 S0: sync cells, constant `1 0` (white, black). Wrong sync ⇒ frame is
  `unreadable` (guards polarity, position, and stale-rect errors).
- D23…D0: data bits, MSB first. White = 1, black = 0.
- C3…C0: checksum bits, MSB first.

## Geometry (logical pixels; multiply by the backing scale for device pixels)

- Cell: 16×16.
- Strip: 30 cells in one row = 480×16.
- Backing bar: solid black, one-cell quiet zone on all sides ⇒ 512×48, drawn at
  the **top-left corner of the game window** at offset (0, 0).
- Rendered above all gameplay and HUD, every frame, from the same counter the
  truth log writes.

Luma only: information is carried entirely in black/white. No color, no
anti-aliasing, no alpha — cells are hard-edged filled rectangles.

## Decode procedure (reference; theo/frameindex.py implements exactly this)

1. Strip rect comes from the session manifest (`masked_regions`,
   name `frame_index_strip`) — never searched per frame. The assembler
   (`theo/g1_assemble.py`) locates the rect once per session: a scale-derived
   candidate verified by decoding sampled frames, falling back to a search
   validated by both the checksum and temporal consistency (the decoded index
   must advance ~1:1 with the video frame index). The found rect is recorded in
   the manifest and every frame then uses it unchanged.
2. Convert the rect region to grayscale. Threshold at the midpoint between the
   5th and 95th intensity percentiles within the backing bar. Fallback
   (`theo/frameindex.py`, `_backing_bar_threshold`): when the 95th percentile
   is not above the 5th — a nearly all-black bar, e.g. a small index whose few
   white cells cover less than 5% of the bar — the high reference becomes the
   maximum intensity within the same fixed rect, and the threshold is the
   midpoint of the 5th percentile and that maximum. The decoder never inspects
   pixels outside the rect or another video frame. A fully uniform bar
   thresholds at its own level and reads as all-ones, which fails the sync
   check below and yields `unreadable` rather than a spurious decode.
3. Sample each cell as the mean of the 5×5 patch at the cell center;
   ≥ threshold ⇒ 1.
4. Verify sync = `1 0`; recompute checksum. Either failing ⇒
   `decode_status = "unreadable"`. **An unreadable frame is never inferred,
   interpolated, or smoothed.**

`decode_strip` returns the integer index for a readable frame and `None`
otherwise. In `alignment.parquet`, the builders (`theo/frameindex.py`
`_alignment_table` and `theo/g1_assemble.py` `build_alignment`) store
`engine_frame_idx = -1` for every row whose `decode_status` is not `"ok"` —
both `unreadable` rows inside the readable span and `out_of_session` rows
before the first or after the last readable frame. `-1` never appears on a row
with `decode_status == "ok"`.

## Test vectors

`specs/frameindex_test_vectors.json` maps frame indices (including 0, 1,
boundary and wrap-adjacent values) to their exact 30-cell patterns. The
renderer's headless output and the decoder's synthetic-fixture input must both
match these patterns bit for bit.

- Decoder side: `tests/test_frameindex.py`
  (`test_cell_extraction_conforms_to_frozen_vectors`) checks every vector
  against `theo/frameindex.py`; the same file covers corrupted-strip
  unreadability, duplicate/drop accounting, and 30 fps transcode decimation.
- Renderer side: the patterns print headlessly with
  `dotnet run --project granny/InputTruth -- --print-vectors N…`
  (`granny/InputTruth/Source/Program.cs`), which formats
  `FrameIndexStrip.Pattern` output for comparison against the vectors.
