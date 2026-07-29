# Session index

Every recorded session, its measured capture quality, and its role. Numbers come
from each session's `manifest.json` (integrity block) and the alignment table —
not from capture-time intent. Sessions live at `sessions/<id>/`; video files are
git-excluded and hash-pinned in their manifests.

Drop rate is the fraction of engine frames with no unique video frame. It is the
figure that decides whether a session can serve windowed experiments: labels stay
exact regardless (the frame-index strip pins every captured frame), but drops make
a fixed count of *video* frames span a variable count of *engine* frames.

| session id | mod | capture px | fps | video frames | dups | drops | drop rate | role |
|---|---|---|---|---|---|---|---|---|
| `rec_20260724_031839_take2` | 0.1.0 | 2560×1440 | 60.0 | 9,000 | 897 | 2,604 | 29% | gap fixture (goldenberry calibration) |
| `rec_20260724_031839_take3` | 0.1.0 | 2560×1440 | 60.0 | 9,000 | 43 | 46 | 0.5% | initial exemplar; pre-`input_active` |
| `rec_20260724_171305_5min` | 0.2.0 | 2560×1440 | 60.0 | 35,400 | 326 | 324 | 0.9% | **development** (`val-A`; used for model and threshold selection) |
| `rec_20260724_180100` | 0.2.0 | 2560×1440 | 56.7 | 90,000 | 17,224 | 22,531 | 23.6% | E3 state mining + metrology only |
| `rec_20260724_190233` | 0.2.0 | 2560×1440 | 58.2 | 54,000 | 631 | 2,307 | 4.1% | **train** |
| `rec_20260725_015612` | 0.2.0 | 1710×1112 | 60.0 | 52,091 | 12,916 | 15,495 | 26% | **train** (see note) |
| `rec_20260725_021338` | 0.2.0 | 1710×962 | 60.0 | 54,000 | 219 | 220 | 0.4% | **train** (cleanest) |
| `rec_20260725_025853` | 0.2.0 | 1710×962 | 60.0 | 54,000 | 18,115 | 18,114 | 33% | **drop diagnostic** (`val-B`; unseen chapter, no strict long-context support) |
| `rec_20260725_160450_b1` | 0.2.0 | 1710×962 | 60.0 | 54,000 | 238 | 239 | 0.44% | **development diagnostic** (frozen before inference; never used in existing training runs) |
| `rec_20260726_020745_calib` | 0.2.0 | 1710×962 | 60.0 | 36,000 | 881 | 881 | 2.45% | **calibration** (Phase A: translucent wild overlay + opaque overlay + strip, all three layers; scored vs engine truth at macro-F1 0.9977; never train data) |
| `rec_20260725_192824` | 0.2.0 | 1710×962 | 60.0 | 54,000 | 2,340 | 2,339 | **4.3%** | **FAILED** — block B1, over the 2% gate; re-record |
| `rec_20260727_220000_test` | 0.2.0 | 1710×962 | 60.0 | 53,280 | 183 | 183 | 0.34% | **UNTOUCHED TEST** — sealed 2026-07-27; never for training, tuning, or selection |
| `rec_20260728_164723_battery_ch1` | 0.2.0 | 1710×962 | — (see note) | 24,502 | 102 | 103 | 0.42% | **UNTOUCHED BATTERY ch1 (anchor)** — sealed 2026-07-28; never for training, tuning, or selection |
| `rec_20260728_172310_battery_ch4` | 0.2.0 | 1710×962 | 60.0 | 36,000 | 129 | 129 | 0.36% | **UNTOUCHED BATTERY ch4** — sealed 2026-07-28; never for training, tuning, or selection |
| `rec_20260728_173748_battery_ch2` | 0.2.0 | 1710×962 | 60.0 | 36,000 | 95 | 95 | 0.26% | **UNTOUCHED BATTERY ch2** — sealed 2026-07-28; never for training, tuning, or selection |
| `rec_20260728_174924_battery_ch3` | 0.2.0 | 1710×962 | 60.0 | 36,000 | 89 | 89 | 0.25% | **UNTOUCHED BATTERY ch3** — sealed 2026-07-28; never for training, tuning, or selection |

`rec_20260727_220000_test` is the untouched test asset: recorded 2026-07-27
after all model recipes, checkpoints, and thresholds in the current results
were frozen; assembled with a deliberate 12-second tail trim (a Spotlight
search overlay appeared during the final seconds while keys were being
pressed outside the game — both the visible overlay and the video/label
disagreement it causes are removed; the trim is recorded in the manifest).
Content note: play is Chapter 6 (Reflection), which contains the launch-orb
mechanic absent from every other own session — a known distribution novelty
for own-data-trained models, unproblematic for corpus-trained ones. Before
this session, no index entry was an untouched test. As of 2026-07-26 every
recorded session has served training, development, diagnostics, or
calibration; the untouched engine-truth test capture required by the frozen
evaluation protocol has not yet been made (status in
[../../PROGRESS.md](../../PROGRESS.md)).

A first session (`rec_20260724_031839`, the initial rig-validation artifact) was
recorded and then excluded entirely: roughly half of it was menu navigation, and it predated
`input_active`, so its menu frames could not be filtered honestly. Its gate
evidence is retained in the working research record; no data from it enters any
experiment.

The four `_battery_ch*` sessions are the pre-registered untouched
engine-truth battery
(`results/idm/UNTOUCHED_BATTERY_PREREGISTRATION.md`), recorded 2026-07-28
after the pre-registration was committed. Each is a final-test surface from
the moment recording started: assembled, validated, gated, shard-built under
the corrected fail-closed mask geometry, leak-scanned, and staged to the
durable object store under `shards/test-battery-v1/` — and nothing else. All
four passed the 2% capture drop gate (worst 0.42%) with strictly 1:1 engine
spans; zero failed captures in this battery. Per-session gate statistics,
shard hashes, and staging receipts are in
`results/idm/untouched_battery_seal/SEAL_RECORD.md`. Chapter-specific notes:

- **ch1 (anchor, `rec_20260728_164723_battery_ch1`).** The recorder was
  interrupted with Ctrl+C before writing `capture_meta.json`, so the meta was
  reconstructed from verified geometry evidence only, with
  `"reconstructed": true` and `achieved_fps`/jitter/frames-written
  deliberately null rather than fabricated (hence "—" in the fps column; the
  video stream itself is CFR 60). The session validator therefore reports
  exactly one expected violation (`manifest.capture.achieved_fps must be a
  positive number`); the deviation, its audit, and the single-violation
  build allowance are recorded in the seal record. The capture tail
  contained a tab-back before the interrupt; the last 181 video frames
  (3.0 s, `--trim-tail-seconds 3.0`) were trimmed so that no frame showing
  the fading terminal overlay — 13 of which still carried a decodable
  frame-index strip — survives into the session.
- **Lost first takes (ch2/ch3).** The first Chapter 2 and Chapter 3 capture
  videos (truth dirs `rec_20260728_170009`, `rec_20260728_171156`) were
  overwritten by their re-takes before preservation. The orphaned truth
  directories are archived as evidence at
  `sessions/_battery_staging/lost_takes/` with a note; no data from them
  enters anything. Per the pre-registration each chapter may be re-captured
  exactly once; ch2 and ch3 have consumed that allowance.
- **ch4 (`rec_20260728_172310_battery_ch4`)** has a single mid-session
  unreadable strip frame (excluded row; window contiguity handles it like
  any missing chunk boundary).

## Notes

- **Calibration-session mod version (corrected 2026-07-26).** An earlier
  revision of this index listed `rec_20260726_020745_calib` under mod 0.2.1.
  The session's `manifest.json` records `InputTruth 0.2.0 (overlay
  inputtruth-v1)`, and `granny/InputTruth/everest.yaml` and the module source
  both declare 0.2.0: the overlay-capable build did not bump the version
  constant. The row now matches the manifest.
- **"B1" names two different captures.** The failed block-B first take is
  `rec_20260725_192824` (called "block B1" in its capture-time note below);
  its passing re-record is `rec_20260725_160450_b1`. The session that
  `PROGRESS.md` and the `results/idm/` reports call "B1" is
  `rec_20260725_160450_b1`, the development-only diagnostic. The failed take
  contributes no data anywhere.
- **Overlay-mask undershoot (found 2026-07-26; geometry, shards, and guard
  fixed 2026-07-27).** The declared `input_overlay` mask rect undershot the
  rendered overlay cells on the 1710-px families (entirely missing the cell
  row on the pre-crop 1710×1112 session), and the calibration session's
  `wild_overlay` rect missed the top of the translucent panel; a readable
  answer-key sliver survived in the 1710-family shards (015612, 021338,
  025853, the B1 feature build, and the calibration session). Measurement
  against engine truth showed the 2560×1440 sessions were covered all
  along. Every session manifest now carries measured per-family rects with
  the superseded rect preserved in-place, `data.mask_coverage` fail-closes
  any build whose rect does not cover its rendered widget, and
  `data/shards_v2/` was rebuilt and probed clean (the leaked build is
  parked at `data/shards_v2_leaked_20260726/` until reruns land). Blast
  radius and the full fix record are in the findings log: no transferable
  benefit was observed on held-out sessions, but this does not rule out
  training distortion; own-data model reruns are queued.
- **Mod 0.1.0 vs 0.2.0.** 0.1.0 sessions log the 7 keys only; the assembler writes
  documented placeholders for state fields and `input_active=true`. Only 0.2.0
  sessions carry real position, speed, dash count, stamina, on-ground, death, and
  `input_active`. E3 uses 0.2.0 sessions exclusively.
- **Capture geometry changed mid-project.** The 2560×1440 sessions were captured on
  an external display; the 1710×* sessions on the built-in screen, where Celeste
  runs windowed and pads its own 16:9 canvas. `rec_20260725_015612` was captured
  before letterbox cropping landed, so its strip sits at y≈121 rather than the
  canvas origin, and its masked-region rects differ accordingly — handled per
  session via the manifest, never assumed.
- **`rec_20260725_015612`'s 26% drop rate** rests on alignment rows that
  include an engine-counter reset the validator did not flag; the alignment
  needs repair-or-reject and the denominator recomputing (queued in
  PROGRESS.md). It was measured only after assembly; a
  three-point manual spot-check earlier in that video showed 1:1 correspondence,
  so its drops are concentrated rather than uniform. It is retained in training
  only through gap-aware loaders that reject non-consecutive windows. It offers
  little strict long-context support and should not drive a temporal comparison.
- **`rec_20260725_192824` (block B1, FAILED).** 2026-07-25, 15:00 duration, rooms
  `06-b`, `06-a`, `01-b`, `03-a`, `00-b`, `02-b`, `05-a`, `02-a`; 81% input-active;
  rare keys in band (grab 24.2%, dash 7.7%). Capture side was clean — 60.0 fps
  achieved, 18.02 ms p99 tick jitter, 54,000/54,000 frames decoded, strip
  continuous, engine span 1:1 overall — so the 4.3% is the game's render rate
  again, not the grab loop. Manifest at `sessions/rec_20260725_192824/manifest.json`.
  No split assigned: a failed session is not train data. The directory is kept as
  evidence rather than deleted, and the video is untouched — not trimmed, not
  patched, not retried in place.
- **Three distinct capture failure modes** were separated across these sessions —
  contention jitter (drops), thermal creep (mean fps), and the game's own render
  rate (duplicates). See the
  [curated engineering lessons](../../docs/engineering-lessons.md).
- **Shards** are built to `data/shards_v2/` (git-ignored) with measured mask
  rects applied, verified zero, and gated by the mask-coverage check
  (rebuilt 2026-07-27 after the overlay-mask undershoot; see the
  overlay-mask note above). The split lists there
  (`train_sessions.txt`, `val_a.txt`, `val_b.txt`, `train_scale{1,2,3}.txt`) are
  the authoritative record of which session served which role in which run.
  These shards, split lists, and per-session manifests exist only in the private
  working repository: `data/shards_v2/` is git-ignored and the public-export
  public repository does not include them.
