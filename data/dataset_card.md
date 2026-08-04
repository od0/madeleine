# Dataset card

Card date: 2026-07-26, mapping and evaluation sections revised 2026-08-03.
This card describes the private working corpus behind the MADELEINE
experiments; it is not a redistributable dataset release. Volatile job
state lives in [../PROGRESS.md](../PROGRESS.md). Two NitroGen mapping
defects were found and repaired after the original card date; the current
label authority is the resolved-v3 corpus described under Controller
mapping, and the incident record is
[../results/idm/NITROGEN_LABEL_INCIDENTS.md](../results/idm/NITROGEN_LABEL_INCIDENTS.md).

Intended use: research on inverse dynamics models for *Celeste* — when
actions are recoverable from visual evidence and how supervision quality
affects the answer. NitroGen-derived labels carry a CC BY-NC 4.0
non-commercial encumbrance (see
[../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).

MADELEINE combines three kinds of supervision for *Celeste* inverse dynamics:

1. engine-truth local captures;
2. mapped gamepad labels from the
   [NitroGen dataset](https://huggingface.co/datasets/nvidia/NitroGen);
3. visually decoded keyboard overlays from public gameplay videos.

They have different semantics and are never merged under a generic “ground
truth” label.

## Rights and distribution

The project does not distribute source videos or the NitroGen label corpus,
and the public export ships no derived label or feature files either: the
repository excludes video,
parquet, and array formats. What ships is code, compact aggregate reports,
source hashes, and a small set of credited figure exhibits. NitroGen's
dataset card identifies its annotations as **CC BY-NC 4.0** and intended for
research and development. Source videos remain subject to their original
owners' and platforms' terms.

Locally captured gameplay, exact engine logs, and large derived feature shards
are also kept outside ordinary Git history. Anyone publishing a derived dataset
must make a separate rights and consent decision.

## NitroGen Celeste funnel

Numbers below are measured from the extracted label slice and a source-video
census performed on 2026-07-23 through 2026-07-26.

| Stage | Videos | Chunks / frames | Hours | Meaning |
|---|---:|---:|---:|---|
| Extracted Celeste labels | 411 | 123,111 chunks | 684 nominal | Every metadata entry in the local NitroGen slice |
| Availability census | 411 censused; 245 available | — | — | 231 Twitch and 14 YouTube sources entered recovery |
| Historical successful downloads | 244 | — | 213.0889 label-hours | One available source failed recovery |
| Historically eligible 60-Hz chunks | 221 | — | 192.9 | At least one chunk met the per-chunk rate rule |
| Durably preserved media | 232 | — | 164.4222 label-hours | Current object-store and training-cache video population |
| Strict feature-eligible corpus | 211 | 27,165 chunks / 32,598,000 rows | 150.9167 | Durably present, whole-video metadata-valid membership on the nominal 60-Hz label grid |
| Higher-confidence bindings | 93 | — | 106.00 | Strict cohort without the broad bind fallback |
| Broad-fallback bindings | 118 | — | 44.92 | Strict cohort with materially noisier action semantics |

The 192.9-hour and 150.9167-hour numbers describe different inventories as
well as different gates. Twelve historical downloads were not in the durable
media archive; ten of those represented 41.9833 hours in the 221-video
eligible-chunk pool. The tracked artifacts establish the loss but not its
cause. Subtracting those ten gives the 211-video, 150.9167-hour population.
The all-video feature build over this population is complete: 1,554 FP16
feature shards covering all 32,598,000 frames (194 videos decoded natively at
60 Hz, 17 timestamp-resampled), with deep validation recorded in
[../results/idm/CORPUS_AUDIT.md](../results/idm/CORPUS_AUDIT.md).

The broader extracted games are:

| Game | Videos | Chunks | Nominal hours |
|---|---:|---:|---:|
| Celeste | 411 | 123,111 | 684 |
| Hollow Knight | 644 | 265,070 | 1,473 |
| Rocket League | 1,283 | 343,878 | 1,910 |

Silksong and Cuphead are absent from this extraction.

## NitroGen source schema

Annotations are arranged as 20-second chunks under one directory per source
video. Each chunk contains `actions_raw.parquet`, `metadata.json`, and
optionally `actions_processed.parquet`.

Measured conventions:

- `actions_raw.parquet` and metadata are present for every extracted chunk;
  processed actions are absent for about 31%, so raw actions are authoritative.
- The schema contains 17 binary button columns and two length-two joystick
  arrays, `j_left` and `j_right`.
- Joystick array dtypes vary between integer and floating point and are cast to
  float at ingestion.
- `original_video.resolution` is `[height, width]`.
- `bbox_controller_overlay` is `[x, y, width, height]` in source pixels and may
  extend beyond the image; it is clamped before rescaling.
- Chunk endpoints are normalized to half-open intervals at ingestion.
- A chunk's label grid is `chunk_size / 20`, not a global dataset constant.

## Source availability and recovery

The Celeste slice contains 14 YouTube and 397 Twitch source IDs. At the census:

- all 14 YouTube sources remained available;
- 231 of 397 Twitch VODs remained available;
- 166 unavailable sources were Twitch VODs;
- 244 videos were fetched successfully at least once, totaling 314.9 GB and
  238.88 source-video hours;
- 213.1 label-hours remained represented because label chunks omit loading and
  other portions of source video.

The durable archive is smaller: 232 media files total 237,033,179,956 bytes
and carry 164.4222 label-hours. Its `corpus/video` prefix contains 234 objects
because it also stores the fetch and availability provenance reports; those
two reports account exactly for the extra 147,046 bytes. Object count must not
be reported as video count.

Availability is time-dependent and should not be extrapolated beyond the
census date. Source sampling and train/test splits occur at the whole-video
level because adjacent 20-second chunks from one VOD are temporally dependent.

## Video rate and temporal alignment

NitroGen provides one label row per declared source frame. The first 480p
recovery pass often returned 30-fps renditions for sources labeled at 60 Hz,
which could not support a verified one-to-one mapping. The production recovery
path therefore prefers source renditions at or near 60 fps.

Rate policy:

- eligibility is checked per chunk, not only per video;
- known missing 20-second chunks create sequence boundaries;
- windows may use footage on either side but never cross a gap;
- nominal 60-fps metadata is checked against decoded average cadence;
- sources materially different from 60 fps are sampled by timestamp onto the
  nominal 60-Hz label grid;
- decoder mode, repeated-frame policy, tail fill, truncation, and skipped rows
  are recorded in manifests.

A metadata audit found 17 nominal-60 sources more than 0.1 fps from 60 when
decoded. The worst averaged 33.8854 fps. Its complete timestamp-resampled build
produced 414,000 train-ready 60-Hz frames with zero tail fill, truncation, or
skipped-short frames.

The remaining fundamental risk is constant label/video offset. Unlike local
captures, internet sources have no engine frame-index strip. A 60-Hz grid makes
one-to-one correspondence possible; it does not independently prove zero
offset.

## Controller mapping

NitroGen actions use positional gamepad controls. MADELEINE maps them to:

```text
left, right, up, down, jump, dash, grab
```

Directions combine d-pad state and joystick sign under NitroGen's
dataset-wide coordinate contract: negative Y is up. The sign is never
inferred per video. An early mapper version did infer it per video and
deterministically inverted analog-derived up/down labels in 22 of 210
training videos; the repair rebuilt directions from the raw controller
arrays and was independently verified over all 32,037,600 rows. The full
account is in
[../results/idm/NITROGEN_LABEL_INCIDENTS.md](../results/idm/NITROGEN_LABEL_INCIDENTS.md).

Jump, dash, and grab use one resolved button set per (video, action)
across all 210 training videos — upstream `actions_processed` evidence
preferred where it exists, per-action behavioral inference elsewhere,
multi-bind aware. This replaced an earlier video-wide confidence flag
whose broad fallback OR-ed multiple plausible buttons and whose inference
could starve dash (same incident record). Of the 630 resolved entries,
227 are policy-resolved pending final human review, so labels from this
corpus are described as resolved mapped labels under an
upstream-preferred policy, not as individually human-verified bindings.
The current training population is 210 videos / 148.3222 hours; the
historical 92-video / 103.4056-hour higher-confidence cohort remains
frozen for matched comparisons.

## Masking and layout

The controller widget is an answer key and is always masked. Geometry from
NitroGen metadata is clamped and rescaled to the fetched rendition. The strict
feature builder asserts that masked controller pixels are black.

The source census also records viewport layout. Fullscreen and stream-layout
videos are distinct quality categories; a controller mask does not remove
facecams, timers, or other route-correlated overlays. These are nuisance
covariates and limitations, not action ground truth.

## Engine-truth captures

Local sessions use a 60-Hz engine grid. The `granny` instrumentation
(`InputTruth` is the compatibility assembly name) records actions,
player state, room identity, and a frame counter. Capture video contains a
machine-readable frame-index strip, enabling exact video-to-engine alignment
and explicit accounting for duplicates and missing frames.

Local session roles are immutable once used:

- training and fine-tuning;
- development and threshold selection;
- drop diagnostics;
- untouched final evaluation.

Mask-coverage defect (found 2026-07-26, fixed 2026-07-27): the declared
input-overlay mask rectangle undershot the rendered overlay cells; later
per-session measurement confined the readable sliver to the 1710-px-family
sessions, and all own-data shards were rebuilt from measured geometry behind
a fail-closed coverage check (leak scan clean). No transferable benefit was observed on held-out
sessions, but this does not rule out training distortion; corrected geometry,
rebuilt shards, and own-data reruns are queued (status in
[../PROGRESS.md](../PROGRESS.md), measurement in the findings log).

The canonical schema is [../specs/session_format.md](../specs/session_format.md)
and the human-readable ledger is [sessions/INDEX.md](sessions/INDEX.md).

## Wild input-overlay labels

Wild labels are recovered from visible keyboard overlays. They are inferred
supervision, even when a pressed key is visually unambiguous. Admission
requires source-bound evidence and separate acceptance of:

- gameplay viewport and overlay geometry;
- gameplay boundaries;
- pressed/released visual states;
- video-to-input compositor offset;
- final decode quality and masked shard integrity.

AI-only proposals cannot admit data. Raw acquired hours, proposed gameplay
windows, decoded labels, and train-ready shards remain separate counters. The
production contract is [../harvest/WILD20.md](../harvest/WILD20.md).

## Splits and evaluation

- No session or source video crosses train and evaluation splits.
- The primary deployment benchmark is the seven-video admitted-wild
  holdout (HUD-decoded truth, zero training overlap); local engine truth
  is the timing-metrology and regression surface. The gate policy is
  [../results/idm/VPT_SMALL_WILD_ADMITTED7_PRIMARY_GATE.md](../results/idm/VPT_SMALL_WILD_ADMITTED7_PRIMARY_GATE.md).
- NitroGen holdouts are mapped-label diagnostics and are named as such.
- Thresholds selected on a development split are frozen before untouched
  engine-truth evaluation.
- Local results outside the untouched-test records are development-set
  results. The sealed untouched engine-truth session was captured, sealed,
  and scored in exactly one pre-registered pass on 2026-07-28; a
  pre-registered multi-chapter battery follows it.
- Prediction support, label prevalence, continuity runs, and excluded targets
  accompany every score.

## Known limitations

- NitroGen mappings contain binding, axis-sign, and timing uncertainty.
- Source video availability is incomplete and decays over time.
- Some videos contain timers, facecams, borders, or editing artifacts that can
  correlate with player route.
- Timestamp resampling preserves nominal time but cannot recreate missing
  visual information.
- Local development sessions have influenced model and threshold decisions and
  do not estimate untouched performance.
- Own-data shards were rebuilt on the corrected mask geometry, and the
  primary own-data trainings were re-run on them 2026-07-28; the sliver did
  not explain the own-only ranking (see results/idm/OWN_V3_RERUN.md).
- Third-party video rights limit what can be redistributed, even when action
  annotations or compact aggregate results can be shared.
