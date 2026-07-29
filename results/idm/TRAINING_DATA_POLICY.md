# NitroGen Training-Data Policy

Status: adopted 2026-07-26 for full-corpus IDM work.

The private working tree retains the chronological implementation record;
durable failure lessons are published in
[the curated engineering notes](../../docs/engineering-lessons.md).

## Decision

The adopted policy uses every metadata-valid NitroGen video.  The exhaustive
pixel-continuity audit was not made a prerequisite for feature generation,
architecture prototyping, or training.

Known missing chunks remain sequence boundaries: training windows may use all
footage before and after a gap, but may not join the two sides as though no time
passed.  Isolated duplicated or repeated CFR frames remain ordinary training
examples.  No video is excluded merely because an automated scan finds a high
duplicate-frame rate.

This was a scheduling decision about where to spend GPU and review time. It is
not a data-quality proof: it does not establish that every retained frame is
sound, and it is not a claim that frame integrity is theoretically irrelevant.

## Evidence behind the decision

The accepted corpus contains 211 videos, 27,165 labeled 20-second chunks, and
32,598,000 rows on the nominal 60-Hz label grid (150.9167 hours). For a 128-sample context
at stride three, the known contiguous label runs retain 149.2322 target-hours,
or 98.8838 percent of all possible targets.  Avoiding windows that cross gaps
therefore costs roughly 1.1 percent of potential examples; it does not reduce
training to a small clean subset.

The visual scan was paused after 49 of 211 videos, all without decoder-process
failure.  Forty-eight decoded at least 99.43 percent of their expected labeled
frames; `v1097557936` decoded 235,382 of 414,000 expected frames (56.86
percent).  A follow-up metadata pass found that its nominal stream rate is 60
fps but its decoded average is 33.89 fps.  Sixteen other nominal-60 sources
differ from 60 by more than 0.1 fps, mostly slightly and three at roughly
53.8–55.4 fps.  This is useful cadence evidence, but it does not justify
blocking the ordinary sources or discarding every variable-rate video.

Mapping semantics are the larger known data risk.  The unflagged bind cohort
contains 93 videos and 106.00 hours; the other 118 videos and 44.92 hours use a
broader fallback mapping.  Both cohorts were featurized and retained (the
feature build completed and passed deep validation on 2026-07-26; see
`CORPUS_AUDIT.md`) so the
effect of label volume versus bind noise could be measured explicitly. The
completed matched arms favor all-valid in macro AP on mapped `y4n` (0.2723
versus 0.2693) and B1 (0.2713 versus 0.2603), so all-valid remains the default
and unflagged remains the quality diagnostic. This is a net-utility result,
not evidence that fallback mappings are correct: corpus size changes at the
same time and timing/per-key effects are mixed.

## Operational policy

1. **Preserve all valid videos.** Build features and manifests for all 211
   metadata-valid sources on the nominal 60-Hz label grid (completed and
   deep-validated 2026-07-26).  Keep an all-video
   session list and an unflagged-bind session list; never silently replace the
   former with a curated tier.
2. **Split only at known long gaps.** Missing 20-second chunks create separate
   run/session IDs.  Window sampling cannot cross a run boundary.  A gap does
   not disqualify the surrounding video.
3. **Tolerate and normalize visual irregularities.** Repeated frames already
   emitted by the CFR decoder are accepted.  Sources whose decoded average rate
   differs materially from 60 are sampled by timestamp onto the nominal 60-Hz
   label grid with FFmpeg's nearest-frame repetition/drop policy.  A missing
   output tail may repeat at most three frames; larger failures remain explicit
   rather than silently stretching the last image.
4. **Keep QA non-blocking.** The masked visual scan may resume when a GPU would
   otherwise be idle.  It produces rankings for interpretation and later
   review, not an automatic admission threshold.  Model or feature work has
   priority while the scale experiments are active.
5. **Report provenance and noise separately.** Corpus bytes present,
   train-ready hours, bind confidence, run continuity, and any decoder anomaly
   remain distinct manifest fields.  “All data” does not imply “all labels are
   equally trustworthy.”

## Assumptions

- A small number of repeated or missing individual images is negligible for
  state recognition relative to six seconds of context and existing label
  noise.
- Exact transition timing may be more sensitive than state recognition.  That
  risk is handled through transition-weighted training, tolerant event metrics,
  and an untouched local evaluation—not by removing whole NitroGen videos.
- A multi-second or 20-second discontinuity is qualitatively different from a
  one-frame defect and must remain a hard temporal boundary.
- Partial audit results and the 98.8838-percent continuity calculation are
  sufficient to make the scheduling decision; they are not proof that every
  source frame is pristine.
- The 150.9167-hour figure is source-label inventory.  Exact train-ready hours
  may differ after explicit short-run/tail handling and must be reported
  separately from source bytes and nominal labels.

## Tradeoffs accepted

| Choice | Benefit | Accepted cost or risk |
|---|---|---|
| Retain isolated repeats | Maximum coverage and simple, reproducible loading | A few contexts contain redundant visual evidence |
| Timestamp-resample 17 VFR sources | Keeps nominal-time action alignment and all-video membership | Repeated images add less information than true 60-fps captures |
| Split at missing chunks | Prevents silent time warps | Loses about 1.1% of potential long-context targets |
| Keep all 211 videos | Maximum behavioral and visual diversity | The 44.92-hour fallback-bind cohort adds label noise |
| Make visual QA optional | GPU time moves immediately to train-ready data and models | Some source anomalies may be characterized after training starts |
| Preserve clean and all-data manifests | Enables a matched noise/volume comparison | Requires reporting two cohorts rather than one headline number |

## Implementation priority at adoption

The adopted schedule assigned one GPU to the all-211 frozen-feature corpus and
one to the pre-specified larger end-to-end diagnostic after its real-data smoke
passed. Native-CFR and timestamp-resampled decoder modes are recorded
separately. The visual scan was preserved at 49/211 and may continue only when
it does not delay feature generation or higher-value model training; its
completion is not a feature or training gate. Current job state is maintained
only in `../../PROGRESS.md`.
