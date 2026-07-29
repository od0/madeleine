# Project review snapshot — 2026-07-25

> **Historical document.** This snapshot reflects the project as of
> 2026-07-25 and is preserved as written; it is superseded on execution
> matters by [PROGRESS.md](../../PROGRESS.md). It also predates the
> 2026-07-26 own-data overlay-mask coverage finding (see the
> ["Mask the measured leak surface" lesson](../engineering-lessons.md) and
> the PROGRESS.md known-issues entry), which affects own-data shards
> discussed below. Banner added 2026-07-26; the review text is otherwise
> unchanged.

This is a condensed historical review of MADELEINE before the first positive
mapped-video transfer result. It is retained because several diagnoses shaped
the current data and model contracts. Current metrics are in
[../../results/idm/SUMMARY.md](../../results/idm/SUMMARY.md).

## Assessment at the time

The review's original verdict, preserved verbatim as historical language, was:

> MADELEINE is conditionally on track for a strong take-home, but not for the
> project's current full scope.

At that point, the instrumentation and acquisition work were unusually strong,
but the IDM evidence was weak, the main foreign-data scaling experiment had not
started, and the report and presentation were unwritten. The review recommended
narrowing around one research question:

> What does trustworthy supervision buy an IDM, and how much data is required
> before it generalizes across sessions?

The acquisition and metrology work was substantially stronger than the first
model results. The repository already had:

- exact engine-frame input logging;
- clock-free video alignment through a rendered frame index;
- explicit answer-key masking;
- a session and provenance contract;
- controlled label-jitter and overlay-degradation experiments;
- a mapped NitroGen data path;
- honest records of failed assumptions and corrected measurements.

The model, however, fit training sessions much better than it transferred to a
different capture. Small engine-truth-only scaling was flat, causal and
centered windows were close, and development metrics were sensitive to the
choice of threshold and event collar.

The review's conclusion was that the project was not data-starved in the simple
sense. The next gains were more likely to come from data diversity, visual
learning, transition-aware objectives, and correct temporal loading than from
recording more minutes of the same local distribution.

## Main risks identified

### Development data was doing too many jobs

The same local session had influenced debugging, checkpoint selection,
threshold selection, and reporting. A separate low-quality split had too many
capture gaps for long-context evaluation. The recommendation was to keep the
existing surface as development evidence and record a genuinely untouched
engine-truth session only after the recipe was frozen.

### Temporal loaders could hide discontinuities

Early shards retained engine-frame indices, but the original window loader
treated adjacent stored rows as consecutive. A window could therefore bridge a
missing frame or reset. Training also failed to apply the documented
`input_active` target policy.

Both issues were subsequently fixed. Window construction now rejects index
gaps and capture resets, and inactive targets are filtered while preserving
history boundaries.

### Foreign-data timing needed an independent check

Matching labels and video at 60 Hz does not prove that their temporal offset is
zero. The controller widget is visible before masking and can provide a limited
cross-correlation diagnostic. Mapped-label evaluation must still retain an
offset/noise caveat because internet video lacks the local frame-index strip.

### The first model was too weak to settle the question

A from-scratch ResNet plus a small GRU could memorize but did not transfer
reliably. The review recommended one focused rescue rather than a broad sweep:

- pretrained or frozen visual features;
- early temporal differences or fusion;
- temporally consistent augmentation;
- transition- and class-aware loss or sampling;
- a clean causal/non-causal comparison on identical support.

### State and event metrics needed equal billing

Per-frame accuracy and F1 reward persistence. Exact event F1 is appropriately
strict but brittle. The recommended evaluation surface became average
precision plus state F1 and onset/release F1 at exact and tolerant frame
collars, with threshold provenance attached.

## Recommended experimental spine

Separately from its six numbered recommendations, the review reduced the
experimental plan to six linked comparisons:

1. engine-truth-only baseline;
2. curated mapped NitroGen pretraining;
3. optional engine-truth fine-tuning;
4. past-only versus centered context;
5. frozen-feature versus end-to-end vision;
6. development evaluation followed by one untouched engine-truth capture.

For mapped data, the recommended storage was compressed pixels or frozen FP16
features rather than duplicating uncompressed frames. Sampling should preserve
whole-video splits, cap creator dominance, emphasize transitions without
discarding ordinary holds, and never cross chunk gaps.

## Begin the report and presentation immediately

The review's sixth recommendation was to reserve several hours for:

- a 60–90 second qualitative prediction video;
- one acquisition diagram;
- the E3 ambiguity/horizon figure;
- the E4 degradation figure;
- own-only versus foreign-data scaling;
- honest failure examples.

## Claims that required softer wording

- “The future resolves the action” became: different-action revisits become
  more distinguishable when future context is available.
- “Compression costs nothing” became: the retained-frame overlay remained
  legible under the tested compression and downscaling.
- Action-history results became an oracle policy-prior diagnostic rather than
  an input available for labeling unlabeled video.
- Exact event F1 became one metric among AP, state F1, and tolerant event F1,
  not a standalone score.

## What happened next

The follow-up work validated most of the review's recommendations:

- curated 13.45-hour NitroGen pretraining improved development macro AP from
  0.1735 to 0.1941 and exact event F1 from 0.0792 to 0.0920 across three seeds;
- a confounded frozen-feature capacity diagnostic improved state F1 but not
  exact timing; a corrected matched baseline remains necessary;
- end-to-end visual learning raised macro AP to 0.2461 while exact event F1
  fell;
- a long-context NitroGen-only model beat prevalence on an unseen video but
  still had weak onset/release recovery;
- a full strict corpus of 211 videos and 150.9167 label-hours was prepared with
  explicit binding and cadence metadata.

The review therefore aged into the project's central finding: state recognition
and transition timing are separable engineering problems. More data and better
vision help the first; the second still needs a better objective, checkpoint
policy, and untouched engine-truth evaluation.
