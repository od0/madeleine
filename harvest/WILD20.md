# Wild-20 production path

This document records the methodology of the wild-overlay production path:
how one candidate video moves from fetch through layout review, boundary
review, offset calibration, decoding, masking, and durable publication. In
the public export of this repository it is methodology documentation rather
than a runnable procedure: the tranche queue file
(`harvest/wild20_tranche.json`), the hand-label and candidate files under
`results/wild/`, the per-video packets under `results/wild20/`, and the
review-packet directories exist only in the private working repository and
are not in the public repository. The pipeline code
under `harvest/` is exported and is the executable contract.

Source videos are publicly posted gameplay on third-party platforms.
Project policy treats acquisition, review, and any redistribution as subject
to the source platform's terms of service and the creators' rights; the
repository publishes only derived review evidence and hashes through the
fail-closed admission path, and clearance for any further use of source
media is an owner decision, not a conclusion this document makes.

## Queue snapshot (historical)

Status note, 2026-07-26: the first target is 20 train-ready hours, not 20
nominal leaderboard hours. At freeze the queue was 11 videos / 27.401601
nominal hours: seven YouTube videos (15.437908 h) and four Twitch videos
(11.963693 h). Primary long semantic-HUD sources and dominant
translucent-style reserves are tracked separately in
`harvest/wild20_tranche.json` (private working repository); v498 is a final
physical-key fallback and the harder compact-keyboard odi source is
deliberately deferred. Live stage state is tracked in
[PROGRESS.md](../PROGRESS.md) under "Wild input overlays", not in this
document.

Commands below use an environment-neutral workspace root:

```bash
export WORK_ROOT=${WORK_ROOT:-/tmp/madeleine-wild}
mkdir -p "${WORK_ROOT}"
```

## Measured operating lessons

- The original worker host probed 14 videos concurrently and triggered
  platform blocking. A replacement worker address must process **one video at
  a time**; installing the Deno runtime does not repair an address that is
  already blocked.
- Current yt-dlp extraction needs its EJS support plus an explicitly selected
  Deno runtime. Production workers pass absolute executable paths through the
  worker's `--yt-dlp` and `--deno` options instead of relying on an ambient
  `PATH`.
- Installing NumPy with `sudo` after the first permission pass created
  `root:root` mode-750 package directories. The unprivileged worker account
  imported an empty namespace (`np.__file__ is None`) and failed only after a
  full PTS scan. Apply permissions after **all** installs and smoke-test as
  the worker account with
  `python -c "import numpy as np; assert np.asarray([1]).item()==1"`.
  Completed videos are preserved; a corrected v2 restart verifies and reuses
  them without downloading again.
- Twitch VOD timestamps are commonly quantized to whole milliseconds, so a
  true 60 Hz stream alternates 16 and 17 ms frame intervals. `1 / median_dt`
  therefore reports a misleading 58.82 fps. Cadence gates use the full-span
  or inlier-mean rate, retain the median as a diagnostic, and tolerate only
  the measured 1.1 ms quantization pattern. Genuine 50/60 Hz alternation and
  large gaps still fail closed. Previously published raw reports remain
  immutable; corrected cadence is recomputed in derived evidence.
- The classical detector's proposed `panel_rect` was wrong for **4 of the 5**
  long-duration positives. It can nominate candidates, but layout inference
  must inspect the full frame and produce reviewed per-video geometry.
- Hours are four different quantities and must remain separate: leaderboard
  **nominal** hours, successfully **decoded** hours, QC-**admitted** hours, and
  post-mask/post-activity **train-ready** hours. Only the last number counts
  toward the 20-hour target.

Two IDs in the original `results/wild/hand_labels.json` (private working
repository) were manually transcribed
incorrectly (`ofy37Fm6Egl` and `odilYNqjL9Y`, where a lowercase
`l` replaced an uppercase `I`). The tranche file uses canonical IDs from
`results/wild/label_batch.json` and `results/wild/candidates.jsonl`:
`ofy37Fm6EgI` and `odiIYNqjL9Y`.

## 1. Fetch one candidate and publish raw evidence

Give each rate-limited worker one JSON object from the tranche file. The
worker has no fleet or credential-management behavior; rclone must already
be configured by the operator.

```bash
python -m harvest.worker_wild \
  --candidate ${WORK_ROOT}/candidate.json \
  --out ${WORK_ROOT}/raw \
  --remote-root object-store:wild/v1/raw
```

Production runs pass explicit absolute paths through `--yt-dlp` and `--deno`
rather than relying on `PATH`. A worker downloads one video with one fragment, writes
yt-dlp metadata, scans every decoded PTS with ffprobe once, persists that exact
vector as hashed `frame_pts.npy` evidence, hashes each local file, uploads it,
reads every remote object back through rclone, and publishes
`upload_complete.json` last. A remote prefix without that marker is incomplete.

Leaderboard duration is loadless game time, not wall-clock video duration. It
is provenance only and never determines an end boundary. URL timestamps may
resolve a start; material excess leaves the end unresolved until reviewed
wall-clock evidence supplies `--end-s`. Only a near duration match permits the
safe inference `[0, media_end]`. The two initial canaries retain their immutable
v1 `fetch.json`; reviewed boundaries are a separate versioned artifact rather
than a destructive rewrite of otherwise-valid raw evidence.

For a legacy v1 canary, publish regenerated PTS evidence to a derived prefix;
do not rerun the raw publisher against its immutable completion manifest:

```bash
python -m harvest.worker_wild \
  --candidate ${WORK_ROOT}/candidate.json \
  --out ${WORK_ROOT}/raw \
  --pts-evidence-remote-root object-store:wild/v1/evidence
```

This writes `frame_pts.npy`, its source-bound hash manifest, and
`pts_evidence_complete.json` under `evidence/<video_id>/`; it never touches the
completed `raw/<video_id>/` object set.

## 2. Create and review one layout per video

`harvest.wild_layout.WildLayout` is the executable schema. Coordinates are
normalized to the encoded video frame. A reviewed `gameplay_rect` is required:
the model receives that crop, not the full stream composite. This is
load-bearing for layouts such as ofy (gameplay in the upper-right beside mascot
and pronoun panels) and nRM (splits/chat/branding around gameplay). Each
physical cell declares:

- its canonical action;
- a glyph-free sample rectangle;
- `luma` or `local_contrast` decoding;
- high- or low-is-pressed polarity; and
- for local contrast, a nearby reference rectangle.

The mask must enclose every sample and reference region, and all seven actions
must be observable. Multiple physical bindings for one action are ORed. Missing
keys are never treated as released. The builder intersects each HUD mask with
the gameplay crop, masks before resize, transforms it into crop coordinates,
and zeros a dilated output rectangle after resize; a HUD outside the crop is
removed by cropping. Layout inference, crop evidence, human review, and the
measured compositor offset all remain hash-bound across the review chain.

### Accept layout geometry through a portable review packet

Do not create a production layout by changing `human_reviewed` to `true`.
First stage one self-contained review-packet directory containing the
unreviewed draft, a geometry overlay, machine-readable cell-state evidence, a
cell-state contact sheet, and the exact source-frame images named by
`evidence_frames_s`. The cell-state JSON must give released and pressed image
path/hash pairs for every executable layout cell.

Create the privacy-whitelisted v2 manifest. Every named input must live below
the manifest directory so its path remains valid in a clean clone:

```bash
python -m harvest.accept_wild_layout manifest \
  --draft-layout REVIEW_PACKET/layout.draft.json \
  --source-sha256 SOURCE_VIDEO_SHA256 \
  --artifact geometry_overlay=REVIEW_PACKET/geometry.png \
  --artifact cell_state_evidence=REVIEW_PACKET/cell_states.json \
  --artifact cell_state_contact_sheet=REVIEW_PACKET/cell_states.png \
  --frame 1.000=REVIEW_PACKET/frames/000001000.jpg \
  --frame 5.000=REVIEW_PACKET/frames/000005000.jpg \
  --out REVIEW_PACKET/review_manifest.json
```

Supply one `--frame` for every time in the draft's `evidence_frames_s`; extra
or missing times fail. The v2 manifest has an exact field whitelist: relative
path, size, SHA-256, frame time, artifact role, video ID, and source SHA-256.
It deliberately excludes source URLs, raw-video paths, machine/host names,
credentials, request headers, free-form notes, and reviewer identity.

After visually reviewing those exact bytes, create two new immutable outputs:

```bash
python -m harvest.accept_wild_layout accept \
  --review-manifest REVIEW_PACKET/review_manifest.json \
  --draft-layout REVIEW_PACKET/layout.draft.json \
  --output-layout ${WORK_ROOT}/layouts/VIDEO_ID.reviewed-unmeasured.json \
  --acceptance-out REVIEW_PACKET/layout_acceptance.json \
  --reviewer "Reviewer Name" \
  --reviewer-kind human_with_ai_assistance \
  --approve
```

The acceptance lives beside the manifest, hashes the draft, manifest, required
artifacts, every evidence frame, source identity, and exact reviewed output,
and records reviewer identity/kind. `human_reviewed` is derived from kind; an
`ai_agent` acceptance remains useful diagnostic provenance but cannot admit,
calibrate, or publish training data. The reviewed layout embeds the acceptance
name plus manifest/source hashes. Both outputs refuse overwrite.

Verification needs no raw video or cloud credentials, but it does require all
manifest-named packet files. It proves byte identity and provenance, not that a
reviewer's visual judgment was correct. Legacy evidence manifests and a bare
review boolean fail closed. This is a hash chain, not a digital signature: the
reviewer identity is an explicit audit assertion whose authenticity still
depends on normal code-review/access controls and anchoring the acceptance hash
in git or an immutable completion manifest. A coordinated rewrite of every
unanchored file cannot be detected by hashes alone.

### Timer proposals are diagnostics, not boundaries

Timer extraction requires explicit reviewer identity and kind. Use `ai_agent`
for AI-selected ROI/bounds evidence; doing so preserves the source-bound scalar
trace and candidate diagnostics but necessarily returns an abstained proposal.
Never label an AI visual draft as human review.

```bash
python -m harvest.extract_timer_trace \
  --fetch-report ${WORK_ROOT}/raw/VIDEO_ID/fetch.json \
  --timer-roi X Y W H \
  --start-s START --end-s END \
  --evidence-ref timer_roi=PATH \
  --evidence-ref wall_clock_bounds=PATH \
  --reviewer "REVIEWER IDENTITY" \
  --reviewer-kind ai_agent \
  --out ${WORK_ROOT}/timer-diagnostics/VIDEO_ID-v3
```

The default frame reader is OpenCV. For a codec/runtime combination that
cannot report or seek exact frame indices (observed with AV1), add
`--decode-backend ffmpeg`. That fallback deliberately decodes from source frame
zero and uses an exact source-index `select` filter; it does not perform a
timestamp/keyframe seek. It streams only the reviewed gray timer ROI and
requires the exact row count, exact `frames × width × height` byte count, and
pipe EOF. Its crop uses ffmpeg's `exact=1`: on 4:2:0 input an odd requested
dimension can otherwise be silently rounded (for example 200×47 to 200×46),
which destroys raw-frame boundaries even when the decoded source frames are
correct. The backend and all byte-count safeguards are recorded in the trace
manifest.

The v3 proposal also fails closed when candidate activity covers less than 25%
of the reviewed envelope or, when leaderboard provenance is available, less
than 50% of nominal loadless duration. On envelopes of at least five minutes,
segment shape must have a median range of at least 5 s and a p90 of at least
15 s. Raw range count remains a diagnostic because real room/load segmentation
can legitimately produce hundreds of ranges. Rejected candidate ranges remain
bounded diagnostics; `suggested_allowed_ranges_s` is empty. Do not weaken the
gate to rescue a trace: fix the timer evidence or detector and write the new
diagnostic packet to a new directory.

The v150 legacy packet demonstrates why both gates matter. Its hashes, cadence,
and 172,800-frame decode are valid, but it proposed 110 fragments totaling
429.696 s from 2,880 s (14.92% envelope coverage and 17.14% of nominal
loadless duration), with a 3.425 s median, roughly 6.24 s p90, and timer
presence only 0.4952. Its
layout evidence was AI-only even though the old proposal set anonymous
`*_reviewed` booleans. The immutable packet is superseded and non-admissible;
see `results/wild20/v1509603803/TIMER_PACKET_INVALIDATION.md` (private
working repository).

## 3. Decode and admit

Before decoding, a reviewer creates `boundaries.json` with
`python -m harvest.wild_boundaries`. Version 2 requires `--reviewer` and
`--reviewer-kind`; `human_reviewed` is derived from the kind, and an
`ai_agent` artifact cannot admit data. Legacy v1 boundaries lack this
provenance and fail closed. The artifact records an explicit wall-clock start
and end plus exactly one of `allowed_ranges_s` or `excluded_ranges_s`, all on
the persisted PTS timeline. This is a hard admission contract: action-radius
filtering can remove more frames later, but directional menu input or held keys
can never re-admit a reviewed exclusion.

Measured half-open wall-clock envelopes for the first two canaries are:

| video | reviewed PTS range | boundary evidence |
|---|---:|---|
| `ofy37Fm6EgI` | `[175.316667, 20963.033333)` | official timer zero at 175.316667; first frozen official finish at 20963.016667 |
| `nRMVyWdNsTo` | `[45.450000, 16767.150000)` | timer zero 45.435 ± 0.001; first frozen finish at 16767.133333 |

These envelopes are necessary but not sufficient: both HUDs keep toggling
through loads and menus. Their admitted `allowed_ranges_s` must come from a
reviewed official-timer trace—keep intervals where the timer advances, close
only brief hitstop freezes, and exclude long frozen or absent-timer spans.

```bash
python -m harvest.decode_wild \
  --fetch-report ${WORK_ROOT}/raw/VIDEO_ID/fetch.json \
  --layout ${WORK_ROOT}/layouts/VIDEO_ID.reviewed-unmeasured.json \
  --layout-acceptance REVIEW_PACKET/layout_acceptance.json \
  --boundaries ${WORK_ROOT}/boundaries/VIDEO_ID.json \
  --out ${WORK_ROOT}/decoded/VIDEO_ID
```

The output label parquet retains source frame indices and presentation
timestamps. The sidecar reports per-cell threshold/separation/duty,
per-action transition rates, PTS gaps, and the temporal correction. Decoding is
not admission: `admitted` becomes true only after geometry, cell separation,
rate, timestamp continuity, human review, and compositor-offset gates pass.
`labels_raw.parquet` preserves every decoded label inside the reviewed
wall-clock envelope for QC; `gameplay_allowed` carries the reviewed range gate
into the aligned labels and shard builder.

### Measure the HUD compositor offset

A first decode may fail **only** the unmeasured-offset gate. In that case, use
the observed `labels_raw.parquet` dash onsets to measure the layout's integer
frame offset before rerunning the decoder:

```bash
python -m harvest.calibrate_offset \
  --video ${WORK_ROOT}/raw/VIDEO_ID/SOURCE.mp4 \
  --layout ${WORK_ROOT}/layouts/VIDEO_ID.draft.json \
  --labels ${WORK_ROOT}/decoded/VIDEO_ID/labels_raw.parquet \
  --decode-report ${WORK_ROOT}/decoded/VIDEO_ID/decode_report.json \
  --out ${WORK_ROOT}/calibration/VIDEO_ID
```

Offset signs are explicit: an observed HUD frame `o` maps to gameplay frame
`g = o + offset`, so a late-rendered HUD has a negative offset. The bounded
`dash_hitstop_v1` measurement searches `[-12, +12]` only on near-CFR 59--61 Hz
video. For each candidate gameplay dash it compares the median pre-dash motion
at `g-3..g-1` with the **maximum** motion at the three frozen transitions
`g+1..g+3`, then requires the motion rebound at `g+4`. Maximum is deliberate:
using the median can let a one-frame-shifted candidate hide the launch frame.

The automatic check (OffsetPolicy v3) is fail-closed. It filters to strong
events (per-event best score at least 3.0) and needs at least 20 of them, a
winner off the search boundary, a median-score lead of at least 2.0 over the
best non-adjacent (|Δ| ≥ 2) lag, early/middle/late temporal-block winners
unanimous within the winner±1 collar, and a winner±1 bootstrap fraction of at
least 0.95 over 2,000 deterministic resamples, all on near-CFR 59–61 Hz
footage. The per-event mode and collar fractions are recorded as SNR
indicators but do not block: the ground-truth diagnostic (2026-07-28, private
working repository, `offset-gate-groundtruth-diagnostic/`) ran the identical
pipeline on engine-truth sessions whose true offset is 0 by construction and
measured collar fractions from 0.63 to 0.93 while the winner, margin,
bootstrap, and block gates discriminated correctly in every condition — the
fraction tracks footage SNR, not offset correctness, so v3 removed the v2
0.80 floor. Static, multimodal, drifting, boundary-winning, and weakly
separated evidence is rejected; offsets are never averaged. At most 256
evenly spaced onsets are decoded, so this remains bounded on multi-hour
videos.

Each calibration carries a `verdict`. `pass` means every blocking gate holds.
`uncertain_adjacent` means the winner is decisive by bootstrap and unanimous
temporal blocks within the ±1 collar but the non-adjacent median margin is
below 2.0: the offset is the winning lag with a recorded ±1-frame
uncertainty, and it is admission-eligible only through a human acceptance
that passes `--accept-uncertain-tier` explicitly. Anything failing bootstrap,
block unanimity, the event floor, or the boundary check is `fail` with no
tier.

Even a `pass` verdict is **not acceptance**. The tool writes
`offset_calibration.json`, a hash sidecar, the per-event `score_matrix.npz`
(so a later policy revision can re-verdict without re-decoding the video),
and a ranked masked-gameplay contact sheet showing `g-1`, `g`, the three
expected freeze frames, and the `g+4` rebound. It always leaves
`human_contact_sheet_review: pending`, `calibration_accepted: false`, and the
input layout untouched. Existing immutable v2 calibrations are re-verdicted
without video reprocessing by `python -m harvest.reverdict_offset_v3
--v2-dir CALIBRATION_DIR`, which hash-verifies the v2 record and recomputes
the v3 gates from its serialized per-lag statistics into an `offset-v3/`
directory.

After inspecting repeated freeze/rebound evidence, create the measured layout
only through the acceptance command:

```bash
python -m harvest.accept_wild_offset \
  --calibration ${WORK_ROOT}/calibration/VIDEO_ID/offset_calibration.json \
  --input-layout ${WORK_ROOT}/layouts/VIDEO_ID.reviewed-unmeasured.json \
  --layout-acceptance REVIEW_PACKET/layout_acceptance.json \
  --output-layout ${WORK_ROOT}/layouts/VIDEO_ID.final.json \
  --acceptance-out ${WORK_ROOT}/calibration/VIDEO_ID/offset_acceptance.json \
  --reviewer "Reviewer Name" \
  --reviewer-kind human_with_ai_assistance \
  --approve-contact-sheet
```

The command rechecks every serialized automatic gate, verifies the calibration
sidecar and contact-sheet bytes, transitively verifies the layout-review
packet, requires its exact reviewed-layout hash to equal the calibration input
hash, and hash-binds the source video, calibration, contact sheet, generated
layout, reviewer identity, and explicit approval. Offset acceptance v3
requires a v3 calibration; v1 and v2 records remain readable evidence but are
not acceptable. An `uncertain_adjacent` verdict additionally requires
`--accept-uncertain-tier`, and the acceptance artifact and generated layout
record the tier and the ±1-frame offset uncertainty. The command creates new
files atomically and refuses overwrite. Reviewer kind is one of `human`,
`human_with_ai_assistance`, or `ai_agent`; an AI-agent review remains auditable
but explicitly does **not** satisfy the human admission gate.

Final decoding must supply the acceptance artifact as well as the generated
layout:

```bash
python -m harvest.decode_wild \
  --fetch-report ${WORK_ROOT}/raw/VIDEO_ID/fetch.json \
  --layout ${WORK_ROOT}/layouts/VIDEO_ID.final.json \
  --layout-acceptance REVIEW_PACKET/layout_acceptance.json \
  --boundaries ${WORK_ROOT}/boundaries/VIDEO_ID.json \
  --offset-acceptance ${WORK_ROOT}/calibration/VIDEO_ID/offset_acceptance.json \
  --out ${WORK_ROOT}/decoded-final/VIDEO_ID
```

Editable offset fields alone can no longer admit a source. If any automatic or
human gate fails, the video remains unadmitted; do not substitute a guessed
zero or an averaged lag.

## 4. Mask and build model shards

```bash
python -m harvest.build_wild \
  --decode-report ${WORK_ROOT}/decoded-final/VIDEO_ID/decode_report.json \
  --layout ${WORK_ROOT}/layouts/VIDEO_ID.final.json \
  --out ${WORK_ROOT}/shards/VIDEO_ID
```

The builder refuses unadmitted reports. It masks the answer key before resize,
zeros a dilated output rectangle after interpolation, splits at long inactive
spans, and records decoded hours separately from train-ready hours. Its report
hash-binds the final decode report, native labels, layout, boundaries, and
source video. The final corpus target is the sum of `train_ready_hours`, not
source or decoded hours.

When elapsed time matters, mechanically clean AI-only decodes may be converted
in parallel under an explicitly separate noisy-supervision tier:

```bash
python -m harvest.build_wild \
  --provisional \
  --decode-report /ephemeral/wild/decoded-provisional/VIDEO_ID/decode_report.json \
  --layout /ephemeral/wild/layouts/VIDEO_ID.draft.json \
  --out /ephemeral/wild/shards-provisional/VIDEO_ID
```

This mode does not weaken canonical admission. It accepts only known
review/offset rejections plus a disclosed provisional layout-confidence
rejection when confidence remains at least 0.75, requires contiguous PTS, cell
separation of at least 20 luma, and no action with more than 5% single-frame
positive runs. Its
session IDs, report filename, schema version, and corpus manifest are all
provisional-specific. It records `train_ready_hours: 0` and the usable volume
only as `provisional_trainable_hours`; the canonical derived publisher cannot
consume its report. This tier is suitable for a noisy-data/pretraining
ablation, not for reporting clean Wild20 yield.

## 5. Publish one derived video durably

Use a new state directory outside the calibration, decoded, and shard
directories. Publish only after final decode reports `admitted: true` with no
rejections and the build report contains positive train-ready hours:

```bash
python -m harvest.publish_wild_derived \
  --video-id VIDEO_ID \
  --input-layout ${WORK_ROOT}/layouts/VIDEO_ID.reviewed-unmeasured.json \
  --layout-acceptance REVIEW_PACKET/layout_acceptance.json \
  --layout ${WORK_ROOT}/layouts/VIDEO_ID.final.json \
  --boundaries ${WORK_ROOT}/boundaries/VIDEO_ID.json \
  --offset-calibration ${WORK_ROOT}/calibration/VIDEO_ID/offset_calibration.json \
  --offset-calibration-sha256 ${WORK_ROOT}/calibration/VIDEO_ID/offset_calibration.sha256 \
  --offset-contact-sheet ${WORK_ROOT}/calibration/VIDEO_ID/dash_offset_contact.png \
  --offset-acceptance ${WORK_ROOT}/calibration/VIDEO_ID/offset_acceptance.json \
  --decode-report ${WORK_ROOT}/decoded-final/VIDEO_ID/decode_report.json \
  --labels-raw ${WORK_ROOT}/decoded-final/VIDEO_ID/labels_raw.parquet \
  --labels-native ${WORK_ROOT}/decoded-final/VIDEO_ID/labels_native.parquet \
  --build-report ${WORK_ROOT}/shards/VIDEO_ID/wild_build_report.json \
  --shard-dir ${WORK_ROOT}/shards/VIDEO_ID \
  --state-dir ${WORK_ROOT}/publication/VIDEO_ID \
  --remote-root object-store:wild/v1/derived
```

The publisher has a typed per-video allowlist; it never uploads a directory
glob. It includes the v2 layout manifest, acceptance, draft, required review
artifacts, and exact frame evidence under `layout/review_packet/`, preserving
their manifest-relative paths for verification after download. Shard paths
come only from `wild_build_report.json`, and dedicated
calibration, decoded, and shard directories must contain exactly their named
artifacts—unexpected local files fail as stale. The publisher also rejects
unsafe IDs or relative paths, symlinks, mixed remote objects, an existing
completion marker, and reuse or overlap of its state directory.

It writes a deterministic local SHA-256/size manifest, copies every named
object with `rclone --immutable`, streams every object back through
`rclone cat`, and publishes `derived_complete.json` last. A remote video prefix
without that marker is incomplete. A completed prefix is immutable; never
repair it in place. This command deliberately does not create or publish an
aggregate corpus manifest. Build that centrally from independently verified
per-video completion markers after the tranche is complete.
