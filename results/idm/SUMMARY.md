# IDM results summary

Status as of 2026-07-29: curated 13.45-hour recipe frozen on local development
validation; the 40.61-hour diagnostic, NitroGen-only training-held-out-video
pilot, and both full-corpus scale arms are complete;
the matched wild-provisional diagnostic and both source-balanced provisional
blend diagnostics are complete; the six corrected own-v3 primary reruns are
also complete and independently validated. The exact 105.7M VPT-small
two-H100 run is complete: it beat both final GRUs in macro AP on exact
common support but failed its all-clauses candidate gate on rare-key
coverage/rate calibration. Neither blend nor VPT-small qualified for B1 or a
new fresh capture.
Local-development event metrics below use per-key oracle thresholds selected
on the local development session (`val-A`) and must not be presented as locked
final-test numbers. The NitroGen-only section instead selects oracle thresholds
on its same mapped-label validation video and states that caveat separately.
The sealed untouched engine-truth session has now been scored, in exactly one
pass under a written owner resolution of the preflight's three blocking input
gaps (2026-07-28; no substitution or post-seal fitting anywhere). Every table
in this report except the final untouched-test section is a
development-surface result; the untouched-test section at the end is the only
final-test evidence, and its numbers are substantially below the
development-surface tables.

Mask-coverage defect disclosure (2026-07-26; geometry, shards, and guard
fixed 2026-07-27): the declared input-overlay mask rect undershot the
rendered overlay cells in the engine-truth (own-data) shards these arms
trained on, so a readable keyboard-overlay sliver survived in some own-data
training and evaluation frames (verified first-hand: one leaked pixel region
separates `left` at AUC 1.000 in a training shard). Per-session measurement
against engine truth then narrowed the scope: the leak was confined to the
1710-px-family sessions (`rec_20260725_015612` and `rec_20260725_021338` in
training, `rec_20260725_025853` = val-B, and the B1 diagnostic frames); the
2560×1440 sessions — including `val-A` — were covered all along, and models
trained only on mapped-video pixels never saw this overlay. No transferable
benefit was observed on held-out sessions, but this does not rule out
training distortion in the arms that consumed leaked frames. The shards are
rebuilt from measured mask geometry behind a fail-closed coverage check and
probed clean (masked zones identically zero; adjacent-band AUC at gameplay
level). The six primary own-data reruns on those corrected shards and their
content-validated feature cache are now complete. Scratch macro AP changed
0.1735 -> 0.1700, while Tier-B-initialized fine-tuning changed 0.1938 ->
0.1998; neither arm improved oracle +/-2-frame timing. Legacy mask-era rows
are retained below only where explicitly labeled. The direct attribution,
run hashes, and exact seed deltas are in
[`OWN_V3_RERUN.md`](OWN_V3_RERUN.md); the tracked board summary is in
`../../PROGRESS.md`.

The key-state-accuracy inventory is tracked separately in
[`KEYPRESS_ACCURACY.md`](KEYPRESS_ACCURACY.md). The selected 36.9M and
112.95M end-to-end models reach 67.37% and 67.62% per-key micro accuracy, but
only 13.90% and 12.39% seven-key joint exact-match accuracy on val-A at the
natural 0.5 binary-head threshold. Always released reaches 82.85%/33.60% and
one-frame persistence 98.95%/93.32% on the same support. VPT never defines
which aggregation produces its published 90.6%, so the earlier claim that
MADELEINE matched VPT's ten-hour curve is withdrawn. The completed 103.41-hour
unflagged and 148.32-hour all-valid arms are useful within-project scaling
tests, not guaranteed apples-to-apples VPT rungs.

## Data and model

- Curated mapped NitroGen set: three source videos, 2,905,200 labeled frames,
  13.45 hours. The curation notes described these as three distinct creators;
  uploader identity is not recorded in the tracked corpus metadata, so that
  description rests on the original manual review rather than a
  machine-checkable field.
- Expanded mapped NitroGen set: ten videos, 8,770,800 labeled frames,
  40.61 hours.
- Historical shorthand: Tier B is the curated 13.45-hour corpus; Tier C is the
  curated 40.61-hour corpus. New prose uses the measured hours where practical.
- Storage: 2.9 GB for the 13.45-hour corpus and 5.7 GB cumulative for the
  40.61-hour corpus, using
  512-dimensional FP16 frozen ResNet-18 features.
- Engine-truth data: three training sessions and one development session.
- Model: trainable 512→256 projection, projected-feature deltas, 256-unit GRU,
  seven binary heads, 32 centered frames (15 past, target, 16 future).
- Loss: class-balanced BCE with 8× weight on onset/release labels.

## VPT-small direct comparison

The first genuine VPT-topology run is complete. Unlike the historical GRUs,
it consumes 128 raw 128x128 frames at 20 Hz, applies the noncausal 5x1x1
Conv3D before the Appendix-D spatial stack, uses four unmasked Transformer
blocks, predicts all 128 positions, and trains for 20 epochs with natural
per-key NLL. The width-only reduction yields 105,696,398 parameters, directly
comparable in scale to the 112.95M GRU.

On 4,224 identical active rows from corrected own-v3 val-A, final VPT-small
reached 0.3603 macro AP, 0.2504 state F1, 0.0351 exact event F1, 0.1005
+/-2-native-frame event F1, and 84.89%/38.35% fixed-0.5 micro/joint key-state
accuracy. The stronger final GRU on this support was the 36.9M model at 0.2622
AP, 0.2491 state F1, 0.0378 exact, 0.0810 +/-2, and 64.29%/10.91% accuracy.
Always released was 84.19%/35.58%.

Five of six preregistered clauses passed. The candidate gate failed because
`down` recall was zero and `dash`, `down`, `left`, and `up` predicted-positive
rates fell outside the required 0.5x--2.0x prevalence band. The lowest-NLL
epoch-2 diagnostic predicted every key released, so final epoch-20 weights
remain the preregistered headline. This is an implementation and architecture
success, but not authorization for more seeds or sealed evaluation. Full
metrics and immutable checkpoint provenance are in
[`VPT_SMALL_113M_RESULTS.md`](VPT_SMALL_113M_RESULTS.md).

The follow-up decision tree is frozen before new data in
[`VPT_SMALL_CALIBRATION_RETRAIN_PREREG.md`](VPT_SMALL_CALIBRATION_RETRAIN_PREREG.md).
Calibration is expected to diagnose or repair the badly positioned dash/left
scores and perhaps up, but `down` AP (0.0102) is below its 0.0137 prevalence
anchor, so a monotonic scalar transform is not expected to recover that key.
The original 5/6 result remains immutable; any calibrated system is a new
prospectively evaluated variant.

## Primary three-seed result

Evaluation surface: the local development session (`val-A`). AP is
threshold-free; event F1 uses per-key oracle thresholds selected on `val-A`
itself, so the event columns are same-surface ceilings, not calibrated
results.

The current engine-truth-only column is the corrected own-v3 rerun. The
mapped-pretrained column is the original frozen zero-shot Tier-B family: it
consumed no own-data training shards, so the mask repair did not create a
corresponding rerun for that arm.

| Seed | Clean own-v3 AP | Mapped-pretrained AP | Δ AP | Clean own-v3 exact F1 | Mapped-pretrained exact F1 | Δ exact F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1684 | 0.1935 | +0.0251 | 0.0788 | 0.0911 | +0.0123 |
| 1 | 0.1740 | 0.1968 | +0.0227 | 0.0781 | 0.0942 | +0.0161 |
| 2 | 0.1677 | 0.1920 | +0.0244 | 0.0783 | 0.0907 | +0.0124 |
| Mean | 0.1700 | 0.1941 | **+0.0241** | 0.0784 | 0.0920 | **+0.0136** |

Mean ± sample SD:

- Clean own-v3: 0.1700 ± 0.0035 macro AP; 0.0784 ± 0.0004 exact F1;
  0.0902 ± 0.0029 at ±2 frames.
- Mapped-pretrained: 0.1941 ± 0.0025 macro AP; 0.0920 ± 0.0019 exact F1;
  0.1076 ± 0.0029 at ±2 frames.
- Mean paired gain: +0.0241 AP, +0.0136 exact F1, +0.0174 ±2-frame F1.

Curated mapped NitroGen pretraining improves local development-set AP, state
F1, and exact event recovery in every seed under this corrected comparison.

For provenance, the superseded mask-era own-only rows are preserved here:

| Seed | Mask-era own-only AP | Mapped-pretrained AP | Δ AP | Mask-era own-only exact F1 | Mapped-pretrained exact F1 | Δ exact F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1730 | 0.1935 | +0.0205 | 0.0801 | 0.0911 | +0.0111 |
| 1 | 0.1761 | 0.1968 | +0.0207 | 0.0789 | 0.0942 | +0.0152 |
| 2 | 0.1714 | 0.1920 | +0.0206 | 0.0786 | 0.0907 | +0.0121 |
| Mean | 0.1735 | 0.1941 | **+0.0206** | 0.0792 | 0.0920 | **+0.0128** |

The six-run mask-attribution summary is:

| Family | Macro AP old -> v3 | State F1@0.5 old -> v3 | Fixed exact F1 old -> v3 | Oracle exact F1 old -> v3 | Oracle +/-2 F1 old -> v3 |
|---|---:|---:|---:|---:|---:|
| Scratch own-only | 0.1735 -> 0.1700 (-0.0035) | 0.0713 -> 0.0860 (+0.0146) | 0.0065 -> 0.0095 (+0.0029) | 0.0792 -> 0.0784 (-0.0008) | 0.0960 -> 0.0902 (-0.0058) |
| Tier-B init fine-tune | 0.1938 -> 0.1998 (+0.0060) | 0.0635 -> 0.0895 (+0.0260) | 0.0090 -> 0.0099 (+0.0009) | 0.0866 -> 0.0873 (+0.0008) | 0.1003 -> 0.0986 (-0.0017) |

The overlay sliver therefore does not explain the own-only model's near-chance
ranking. Full all-frame, per-key, seed-level, and checkpoint details are in
[`OWN_V3_RERUN.md`](OWN_V3_RERUN.md).

## 40.61-hour scale diagnostic

The expanded set triples mapped NitroGen data from 13.45 to 40.61 hours while
holding architecture, batch size, learning rate, step count, and the local
development session fixed.  Evaluation surface: `val-A`, with the same
same-surface oracle thresholds as above.  The normal lowest-validation-BCE
checkpoint rule produced:

| Seed | Tier B AP | Tier C AP | Delta AP | Tier B exact F1 | Tier C exact F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.1935 | 0.2182 | +0.0247 | 0.0911 | 0.0910 |
| 1 | 0.1968 | 0.1753 | -0.0215 | 0.0942 | 0.0776 |
| 2 | 0.1920 | 0.2157 | +0.0237 | 0.0907 | 0.0916 |
| Mean | 0.1941 | 0.2030 | +0.0090 | 0.0920 | 0.0867 |

Tier-C seed 1 selected step 0: an untrained initialization.  The trainer
optimizes class-balanced BCE with 8x transition weighting, but checkpoint
selection minimizes plain BCE.  Those objectives can disagree enough to
discard a trained model, so the selected-checkpoint Tier-C mean is not a
trustworthy scale endpoint.

As a diagnostic, evaluating the pre-specified final step (2,000) for both
tiers removes that selector mismatch:

| Fixed step-2,000 endpoint | Macro AP | Exact F1 | ±2-frame F1 |
|---|---:|---:|---:|
| Tier B (mean ± SD) | 0.1923 ± 0.0142 | 0.0919 ± 0.0004 | 0.1045 ± 0.0004 |
| Tier C (mean ± SD) | 0.2079 ± 0.0067 | 0.0921 ± 0.0004 | 0.1045 ± 0.0025 |
| Mean paired change | **+0.0156** | +0.0002 | +0.0000 |

The fixed-endpoint AP change is positive in two of three paired seeds
(-0.0042, +0.0150, +0.0362), driven mostly by `up`, `dash`, and `jump`.
Exact and tolerant transition F1 are flat.  The defensible conclusion is that
additional curated diversity probably improves held-action state recognition,
but this experiment does not show that scale improves transition timing.
Because the fixed-endpoint analysis was prompted by the selector failure, it
is diagnostic rather than a new frozen headline result.

## Context and scale ablations (seed 0)

Evaluation surface: `val-A`, same-surface oracle event thresholds.

| Training/architecture | Macro AP | Exact F1 | ±2-frame F1 |
|---|---:|---:|---:|
| Engine-truth only, 32 centered (mask-era; legacy) | 0.1730 | 0.0801 | 0.0990 |
| Mapped NitroGen 5.9 h, 16 past-only | 0.1643 | 0.0897 | 0.1006 |
| Mapped NitroGen 5.9 h, 32 centered | 0.1731 | 0.0934 | 0.1055 |
| Mapped NitroGen 5.9 h, 32 past-only | 0.1721 | 0.0975 | 0.1132 |
| Mapped NitroGen 13.45 h, 32 centered | 0.1935 | 0.0911 | 0.1089 |
| Mapped NitroGen 13.45 h, 32 past-only | 0.1788 | 0.0981 | 0.1215 |
| Mapped NitroGen 40.61 h, 32 centered | 0.2182 | 0.0910 | 0.1018 |

Adding 16 future frames to the same 15-frame history improves every metric
(16 past-only versus 32 centered).  A longer, 31-frame past gives the sharpest
events, while centered future context and full data give the best state AP.

## Capacity, context, and end-to-end study

Four pre-specified seed-0 runs completed on the 2×A100 training node. The
important result is a decomposition rather than one universal winner.

### Capacity on frozen visual features

At the fixed step-2,000 endpoint, increasing the trainable temporal model from
0.725M to 25.7M parameters changed the full input-active local development
(`val-A`) result as follows:

| Tier-C fixed endpoint | Macro AP | State F1 | Exact event F1 | ±2-frame F1 |
|---|---:|---:|---:|---:|
| 0.725M, 32 frames (three-seed mean) | 0.2079 | 0.2486 | 0.0921 | 0.1045 |
| 25.7M, 32 frames (seed 0) | **0.2140** | **0.2944** | 0.0903 | 0.1022 |

The larger head improves held-action state recognition, especially state F1,
but does not improve transition timing.  Its best-BCE checkpoint reached
0.2350 AP, but the fixed endpoint is the cleaner pre-specified comparison.
The old 0.725M runs used the pre-fix segment-delta convention while all new
runs reset deltas within each window; a corrected 0.725M repeat is still
needed before presenting the capacity delta as a perfectly isolated ablation.

### End-to-end visual fine-tuning

The 36.9M-parameter model fine-tuned an ImageNet ResNet-18 and used VPT-range,
temporally consistent image augmentation, AdamW, 0.01 weight decay, and linear
learning-rate decay.  It trained on the 13.45-hour Tier-B pixel corpus in about
35 minutes. On the full 25,028 input-active local development (`val-A`)
frames:

| Tier-B recipe | Macro AP | State F1 | Exact event F1 | ±2-frame F1 |
|---|---:|---:|---:|---:|
| Frozen features, 0.725M (three-seed mean) | 0.1941 | 0.2068 | **0.0920** | **0.1076** |
| End-to-end, 36.9M (seed 0, best step 6,750) | **0.2461** | **0.2510** | 0.0764 | 0.0879 |

Macro AP rises by 0.0520 absolute (27% relative), with large gains for right,
jump, and dash.  Exact event F1 falls by 0.0156 and grab AP falls by 0.0615.
The final step is nearly identical (0.2471 AP / 0.0781 exact F1), so this is not
a lucky single checkpoint.  The clean-session prevalence baseline AP is
0.1715; the new AP
is 43% above that baseline, versus 13% for the frozen Tier-B model.  The honest
interpretation is that trainable vision adds substantial state-recognition
capacity but the present objective still blurs exact action timing.

### Wider end-to-end temporal model

A matched seed-0 follow-up widened only the end-to-end model's visual
projection and GRU, increasing the parameter count from 36.9M to 112.95M.
It kept the same 13.45-hour Tier-B pixels, 32-frame centered context,
augmentation, optimizer, batch, and 7,500-step endpoint.  The selected
checkpoint is the minimum-validation-BCE checkpoint; thresholds are still
oracle-selected on local development data.

| End-to-end Tier-B model | Macro AP | State F1 | Exact event F1 | ±2-frame F1 |
|---|---:|---:|---:|---:|
| 36.9M, selected step 6,750 | **0.2461** | **0.2510** | 0.0764 | 0.0879 |
| 112.95M, selected checkpoint | 0.2318 | 0.2194 | **0.0810** | **0.0919** |
| 112.95M, fixed final step | 0.2300 | 0.2296 | 0.0776 | 0.0884 |

The wider model is not a monotonic improvement: selected AP falls by 0.0143
and state F1 by 0.0316 relative to 36.9M, while exact and ±2-frame event F1
rise by only 0.0046 and 0.0040.  Its selected per-key AP versus local
input-active prevalence is:

| Key | Prevalence | AP |
|---|---:|---:|
| left | 0.1012 | 0.1427 |
| right | 0.3090 | 0.5160 |
| up | 0.1925 | 0.1718 |
| down | 0.0136 | 0.0121 |
| jump | 0.1157 | 0.2600 |
| dash | 0.0855 | 0.1003 |
| grab | 0.3832 | 0.4195 |

The selected macro AP remains above the 0.1715 prevalence baseline, but `up`
and `down` are below their individual prevalence.  End-to-end capacity alone
therefore did not solve the transition objective or consistently improve
state ranking.  Wall time from wrapper launch through both evaluations was
about 100.5 minutes, versus about 35 minutes for 36.9M, close to the 3.06×
parameter increase despite low steady-state VRAM use.

### B1 engine-truth capacity diagnostic

After the capacity recipe was fixed, both models were evaluated on B1, a
cleaner engine-truth capture not used for gradient training.  B1 remains a
development surface: thresholds below are oracle-selected on B1 itself, so
these numbers do not replace the future untouched P0 evaluation.

| B1 model | Macro AP | State F1 | Exact event F1 | ±2-frame F1 |
|---|---:|---:|---:|---:|
| 36.9M, selected | **0.2422** | **0.3091** | 0.0610 | 0.0718 |
| 112.95M, selected | 0.2388 | 0.2958 | **0.0613** | **0.0754** |
| 36.9M, fixed final | **0.2505** | **0.3085** | 0.0602 | 0.0705 |
| 112.95M, fixed final | 0.2398 | 0.3069 | **0.0625** | **0.0829** |

B1 macro prevalence is 0.1547.  The selected per-key ranking comparison is:

| Key | Prevalence | 36.9M AP | 112.95M AP |
|---|---:|---:|---:|
| left | 0.1603 | **0.2360** | 0.2224 |
| right | 0.3082 | **0.4744** | 0.4378 |
| up | 0.0987 | **0.2329** | 0.1551 |
| down | 0.0311 | 0.0353 | **0.0460** |
| jump | 0.1039 | **0.1690** | 0.1499 |
| dash | 0.0583 | **0.1729** | 0.1512 |
| grab | 0.3223 | 0.3749 | **0.5096** |

Every selected 112.95M key exceeds its B1 prevalence, but the matched macro
result corroborates local val-A: width does not improve overall AP or state
F1.  Its benefit is limited to small exact/±2-frame timing changes and a large
grab-AP gain that is offset by regressions on five other keys.

### Long context, on common support only

The raw 6.37-second reports cover only 1,125 input-active frames in 21 streams,
whereas 32-frame evaluation covers 29,086 frames. The development capture's gaps make
most runs too short for a 382-raw-frame context span.  Comparing the raw JSON
scores would therefore be selection-biased.  Saved 32-frame predictions were
cropped by 174 leading predictions per matching stream so every model below is
scored on exactly the same targets and stream boundaries:

| Common 1,125-frame support | Macro AP | State F1 | Exact event F1 | ±2-frame F1 |
|---|---:|---:|---:|---:|
| 0.725M / 32-frame fixed endpoint (three-seed mean) | **0.3038** | 0.2784 | 0.1658 | 0.1977 |
| 0.725M / 6.37 s fixed endpoint | 0.2852 | 0.2856 | 0.1679 | **0.2481** |
| 25.7M / 6.37 s best step 3,420 | 0.3036 | **0.3007** | **0.1725** | 0.2475 |

Long context leaves exact timing nearly flat but improves ±2-frame event F1
by about 0.050 absolute.  Combining capacity and context retains baseline AP
and adds 0.0067 exact F1 at the selected checkpoint.  The combined final step
regresses to 0.2594 AP / 0.1359 exact F1, so checkpoint choice matters.  These
are promising diagnostics, not final claims: the common support is
small, easier than the complete session, and was also used for checkpoint
selection.  A fresh low-drop, uninterrupted 10–15-minute capture is the
decisive validation.

### Recorded runtime and artifacts

All four runs and both selected/final evaluations completed without failure.
The 25.7M long-context run took 3 hours 53 minutes; the end-to-end run took 35
minutes. The four checkpoints, configs, logs, prediction sidecars, and reports
were archived with hashes below. Current execution state belongs in
`../../PROGRESS.md`, not this completed-results report.

## Full-corpus execution policy — 2026-07-26

This section supersedes the earlier recommendation to block the full corpus on
a pixel-continuity audit. The completed metadata and mapping pass found 211
metadata-valid videos on the nominal 60-Hz label grid, with 32,598,000 label
rows (150.9167 hours).
Known contiguous runs retain 98.8838 percent of potential 382-raw-frame
targets, so windows split at missing 20-second chunks while retaining the rest
of every video.

The masked visual scan was paused, resumably, after 49/211 successful videos
with zero errors.  It is now optional background QA: decoded repeated frames
are tolerated, no visual-duplicate threshold filters training, and completion
is not required for feature generation or the next model.  The larger known
quality risk is action binding: 106.00 hours are unflagged and 44.92 hours use
the broad fallback mapping.  The all-211 corpus is the default scale dataset;
the unflagged list remains a preserved diagnostic rather than a hidden filter.

The partial scan also found one real cadence anomaly: `v1097557936` decoded
only 235,382 of 414,000 expected labeled frames, while the other 48 sampled
videos were at least 99.43-percent covered.  Its nominal stream rate is 60 fps
but its decoded average is 33.89 fps.  A full metadata pass found 17 nominal-60
sources outside a 0.1-fps decoded tolerance.  They now use timestamp-aware
resampling onto the 60-Hz label grid; bounded smoke tests on the worst source at
0 and 6,000 seconds each produced exactly 1,200 frames with zero tail fill.
The final manifest records native/resampled modes and exact train-ready hours.

The adopted feature-generation policy handles native-CFR and timestamp-
resampled sources separately. Exact assumptions, tradeoffs, and failure
handling are recorded in `TRAINING_DATA_POLICY.md`; current job state is in
`../../PROGRESS.md`.

## NitroGen-only training-held-out-video validation

The seed-0 25.7M-parameter model trained on nine NitroGen videos from the
40.61-hour curated set and held out all 16 runs from `y4nQHqYSObI` from
gradient training. No local capture was used. The
step-5,250 endpoint was also the lowest-validation-BCE checkpoint, so selected
and final weights and predictions are identical.

| 554,304 mapped-label holdout frames | Macro AP | Prevalence baseline AP | State F1 | Exact event F1 | ±2-frame event F1 |
|---|---:|---:|---:|---:|---:|
| Selected/final | **0.2435** | 0.1924 | 0.2745 | 0.0127 | 0.0395 |

AP is +0.0512 absolute and 26.6 percent relative above prevalence, with every
key above its own prevalence baseline.  This is useful cross-video evidence
that the mapped NitroGen signal and long-context model learn held-action state.
It is not a timing success: exact event recovery remains weak even with
per-key oracle thresholds selected on the same holdout.  This is a single-seed
run with agreement
scores against noisy mapped labels, not engine truth, not local transfer, and
not a locked final-test estimate.  Full setup, per-key results, validation, and
provenance are in `NITROGEN_HOLDOUT.md`.

### Full-corpus scale result

The matched 25.7M-parameter arms use the same mapped `y4n` holdout, seed,
context, and one-pass endpoint policy as the nine-video pilot. Unflagged uses
92 videos and 103.4056 hours; all-valid uses 210 videos and 148.3222 hours.
Selected and final weights are identical within each run.

| Same 554,304-frame mapped holdout | Macro AP | Prevalence AP | State F1 | Exact event F1 | ±2-frame event F1 |
|---|---:|---:|---:|---:|---:|
| Nine-video pilot | 0.2435 | 0.1924 | 0.2745 | 0.0127 | **0.0395** |
| 103.41 h unflagged | 0.2693 | 0.1924 | 0.2888 | 0.0128 | 0.0393 |
| 148.32 h all-valid | **0.2723** | 0.1924 | **0.2986** | **0.0141** | **0.0425** |

All-valid improves over the pilot by 0.0288 AP (11.8 percent relative) and
0.0241 state F1, but exact and two-frame timing move only 0.0014 and 0.0030.
On B1 engine-truth development data, unflagged/all-valid reach 0.2603/0.2713
AP versus 0.1448 prevalence and 0.2788/0.2773 state F1. Their oracle exact
event F1 is 0.1228/0.1167 and ±2-frame F1 is 0.1448/0.1356. B1 was not used
for training or checkpoint selection, but its event thresholds are selected
on B1 itself and remain oracle ceilings.

At threshold 0.5, unflagged/all-valid mapped-holdout accuracy is
68.48/66.14 percent micro and 11.34/10.19 percent joint, below always released
(80.76/19.21 percent) and one-frame persistence (98.73/91.45 percent). The
scale result is positive for state ranking, not timing or calibrated accuracy.
All-valid also slightly exceeds unflagged AP on both mapped `y4n` and B1, so
the present single-seed evidence does not justify discarding fallback-bound
data. Data volume and cohort membership change together, however, so this is
not a controlled label-quality conclusion.

### Temporal loss and optimizer follow-up

A preregistered aligned-TCN follow-up held the 103.41-hour cohort, model,
seed, and 14,265-step endpoint fixed while testing natural BCE and weighted
learning rates `1e-4` and `1e-3`. The formal decision used final weights on the
same later-eight mapped `y4n` streams and was committed before any new B1
inference.

| Later-eight mapped `y4n` arm | Macro AP | State F1 | Micro accuracy | Exact / +/-2 fixed event F1 |
|---|---:|---:|---:|---:|
| Weighted TCN, `3e-4` headline | 0.2795 | **0.3236** | 66.25% | **0.0073 / 0.0373** |
| Natural BCE control, `3e-4` | **0.2810** | 0.1389 | **81.02%** | 0.0027 / 0.0108 |
| Weighted TCN, `1e-4` | 0.2690 | 0.2972 | 65.41% | 0.0072 / 0.0336 |
| Weighted TCN, `1e-3` | 0.2450 | 0.2558 | 65.66% | 0.0082 / 0.0350 |

Natural BCE is tied on AP at the declared effect size but fails both timing
guards; its higher accuracy comes from predicting many fewer positive states.
The `1e-4` sensitivity does not improve the frozen decision and `1e-3` is
materially worse. The existing weighted `3e-4` run therefore remains the
matched TCN headline. This bounded screen argues against a simple LR artifact,
not against every possible TCN recipe.

Post-decision B1 transfer is mixed: natural BCE reaches 0.2478 AP and 85.81%
micro accuracy, versus 0.1926/50.99% for weighted `3e-4` and an 85.52%
always-released baseline. Its fixed state F1 is only 0.1006 and fixed exact
event F1 0.0169. Weighted `1e-4` improves B1 AP/state F1/fixed exact event F1
to 0.2143/0.2429/0.0523, but cannot revise the already-frozen mapped decision.
All are single-seed development results. Full tables and receipts are in
`TEMPORAL_REDESIGN_RESULTS.md` in the private working repository.

### Wild-provisional supervision diagnostic

An exact one-pass diagnostic trained the matched 25.7M GRU on 22.387 hours of
provisional overlay-decoded labels from seven public videos. The 27.47-hour
decoded-video envelope is not the labeled or train-ready duration: complete
training segments contribute about 18.474 target-hours, and **zero hours is
admitted or train-ready**.

Final weights were scored at fixed 0.5 with no fitted threshold or calibration.
On the identical 269,352-frame later-eight mapped `y4n` support, wild versus
the 103.41-hour unflagged NitroGen reference reached 0.2316 versus 0.2845 macro
AP, 0.2107 versus 0.2985 state F1, and 0.0052/0.0205 versus
0.0117/0.0385 exact/+/-2 event F1. Wild AP is 0.0354 above its 0.1962
prevalence baseline, so the noisy labels contain transferable signal, but the
wild-only model is worse on every primary mapped metric.

Post-release B1 is mixed. Wild loses AP/state F1 (0.2022/0.2515 versus
0.2603/0.2788) but improves fixed exact/+/-2 event F1
(0.0460/0.0595 versus 0.0282/0.0468). It also predicts positive on 66.82% of
B1 key decisions versus 42.00% for the reference, so the timing gain may partly
reflect many more candidate transitions. This single-seed, repeatedly
consulted development reversal motivates a source-balanced blend with a
shuffled-label control; it does not promote the wild-only checkpoint. Full
setup, exact receipts, and limitations are in
[`WILD_PROVISIONAL_GRU.md`](WILD_PROVISIONAL_GRU.md).

The two fixed-compute source-balanced follow-ups are complete. On the identical
later-eight support, pure NitroGen, NitroGen/local 90/10, and
NitroGen/provisional-wild/local 70/20/10 reached macro AP
0.2845/0.2738/0.2771, state F1 0.2985/0.2845/0.2788, and +/-2 event F1
0.0385/0.0384/0.0358. Local's marginal effect was -0.0106 AP; provisional
wild recovered +0.0033 versus the local blend but remained -0.0073 below pure
NitroGen. Neither arm cleared the preregistered mapped-`y4n` gate, so neither
was opened on B1 or promoted to a fresh-capture follow-up.

The local pool contained only 159 complete segment items and was cycled 143.55
times in each blend. Posthoc final-weight scoring showed near-perfect local
train memorization (AP about 1.000, F1 about 0.994–0.995, BCE below 0.0062) and
no gain on the 5-item, 480-target-frame corrected local val-A probe (AP
0.223–0.249, F1 0.205–0.280, BCE 0.788–0.815); broader local-domain transfer
was not tested. The provisional wild source remains 22.387 labeled hours but
zero admitted/train-ready hours. Both final checkpoints have independently
verified, content-addressed Cloudflare R2 receipts; compact evaluation evidence
is tracked in this repository. The sealed untouched session remained
embargoed. Full per-key, accuracy, event, baseline, sampling, provenance, and
memorization evidence is in
[`PROVISIONAL_BLEND_GRU.md`](PROVISIONAL_BLEND_GRU.md).

## Exploratory future-latent pretraining

The owner-authorized seed-zero future-latent Arms C and D both launched on the
shared 60,000-window, `+1,+2,+4` unlabeled-video cache and passed exact resume,
runtime, and memory gates. Both then hit the frozen representation-collapse
stop: single-frame C at step 6,500/30,000 and ordered-pair D at step
6,250/30,000. Terminal online effective rank fell to 22.36% of initialization
for C and 20.33% for D; D's absolute online rank also fell below the minimum of
8. Losses and weights stayed finite, so this was a scientific stability null,
not an OOM or corrupt run.

Neither arm produced a selectable final encoder. Therefore no y4n frame was
opened, no downstream GRU was trained, and no AP/state/event comparison exists
for C or D; B1 and the sealed surfaces remained untouched. The result rejects
this bounded seed-zero normalized-L1/EMA globally pooled recipe, not dynamics
pretraining in general or Photon-1. Exact evidence and durability receipts are
in `DYNAMICS_PRETRAINING_EXPLORATORY_CD.md` and its artifact directories in
the private working repository (an exploratory lane; its receipts carry
storage coordinates the public export excludes).

## Oracle-window exact localization diagnostic

A preregistered own-v3 val-A diagnostic asked whether giving a model the
requested key/event type and guaranteeing one event inside a balanced
16-frame region makes the exact engine-truth frame recoverable. It did not.
Over 12 estimable heads, the conditional 512-D-feature localizer reached
0.07369 exact accuracy versus 0.07534 for its matched dense-BCE control and
0.0625 chance; the paired delta was -0.00165 with 95% interval
[-0.02214, 0.01918]. The full coarse-to-fine cascade therefore failed its
seed-zero gate.

The required bounded pixel follow-up also failed. An ordered native-rate
32x32 frame-pair adapter reached 0.07862 macro exact versus 0.07369 for the
frozen-feature localizer, a +0.00493 delta with 95% interval
[-0.01970, 0.02925], and was essentially tied with a matched symmetric
two-frame control (delta +0.00024, 95% interval [-0.02213, 0.02162]). Its
within-two score rose descriptively to 0.31430, but the central exact-timing
materiality, confidence, breadth, and attribution checks failed. No further
seeds or cascade were run. This null applies to the verified pooled features
and the fixed small 32x32 adapter, not to all source-pixel representations.
Full support, exclusions, calibration, timing-error analysis, commands, and
artifact bindings are in
[`ORACLE_WINDOW_LOCALIZATION.md`](ORACLE_WINDOW_LOCALIZATION.md).

## Study H: high-resolution regional oracle localization

The preregistered seed-zero high-resolution follow-up recovered a real
argmax-timing signal but still failed its full primary gate. On the same 1,150
validation examples and 12 estimable key/polarity heads, the 128x128
task-query regional model reached 20.736% exact accuracy versus 15.583% for
the matched 32x32 task-query control: +5.153 percentage points, with paired
bootstrap 95% interval [+2.754, +7.439]. However, its NLL worsened from
3.4674 to 3.5458, so the registered proper-score clause failed. Both learned
models were also worse than the 2.7726 uniform-chance NLL despite beating
6.25% chance exact accuracy.

The task-query mechanism was not established as the cause of the resolution
gain. The 128x128 global-pooling control reached 20.47% exact, leaving the
task-query delta at only +0.261 points with 95% interval
[-2.018, +2.479]. The gain was also strongly polarity-asymmetric: equal-head
onset exact rose from 23.94% to 34.05%, while release exact moved only from
7.22% to 7.42%. The decision is therefore to reject the Study-H primary gate;
no confirmation seeds and no full coarse-to-fine cascade were run. Full
support, calibration, timing, per-head, provenance, audit, and durability
evidence is in
[`ORACLE_WINDOW_HIGHRES_REGIONAL.md`](ORACLE_WINDOW_HIGHRES_REGIONAL.md).

## Recipe selection

Selected: the three Tier-B pretrained centered checkpoints, evaluated as a
three-seed mean.  Each seed's transition thresholds will be frozen from its
development report and applied unchanged on final test.

Tier C does not replace the frozen endpoint. Both fixed-endpoint full-corpus
arms show that more mapped supervision improves state ranking on the same
held-out video without materially improving exact timing. All-valid slightly
outperforms unflagged in AP, but the single-seed comparison changes data volume
and cohort membership together and is not evidence of statistically secure
monotonic scaling or binding-map quality.

Not selected: corrected own-v3 Tier-B-initialized fine-tuning. Its AP across
seeds is 0.2174, 0.1715, and 0.2105 (mean 0.1998), so the apparent gain over
the mask-era mean of 0.1938 remains seed-sensitive. Oracle +/-2-frame event
F1 fell from 0.1003 to 0.0986. A ten-times-gentler calibration is also
seed-sensitive; selecting one favorable seed would overstate the result.

Not selected: any aligned-TCN variant. The target-aligned implementation is a
useful fast probe, but the matched arm trails the GRU in AP, the natural loss
control trades timing for majority-state accuracy, and neither tested LR
improves the frozen mapped decision. The B1 differences remain
post-decision, repeatedly consulted development evidence.

Not selected: the wild-provisional-only GRU. It learns above-chance ranking on
both frozen surfaces and shows an exploratory B1 timing gain, but trails the
matched NitroGen reference in mapped AP, state F1, accuracy, and event timing,
and severely overpredicts on B1.

## Artifact locations

- Tracked configs, logs, metadata, and development reports: this directory in
  the private working repository.
- Corrected own-v3 attribution report:
  [`OWN_V3_RERUN.md`](OWN_V3_RERUN.md); its six-checkpoint registry
  `checkpoint-index-own-v3-primary-20260728.json` remains in the private
  working repository (storage coordinates).
- In the public repository: this directory's result documents (this
  summary and the reports it links, including the untouched-test records
  under `untouched_test/`, the threshold-freeze records under
  `untouched_prep/`, the blend decision records, and
  `checkpoint_sha256.txt`) are exported exactly where a working link
  appears. Per-run configs, training logs, prediction sidecars,
  `tier_c_manifest.json` (referenced below), `checkpoint_backup_validation.json`
  (referenced below), the engineering log, and the storage-coordinate
  registries remain in the private working repository; references to them
  here say so.
- Local checkpoints (gitignored): `../../checkpoints/tier_b/`.
- Tier-C checkpoints, including selected and final weights (gitignored):
  `../../checkpoints/tier_c/`.
- Overnight checkpoints, each containing selected and final weights
  (gitignored): `../../checkpoints/overnight/`.
- NitroGen-only holdout checkpoint, containing tensor-identical selected and
  final weights (gitignored): `../../checkpoints/nitrogen_holdout/`.
- Tier-C corpus manifest: `tier_c_manifest.json` (private working repository).
- Checkpoint hashes: `checkpoint_sha256.txt`.
- The historical retained checkpoints have private durable, immutable backups;
  their full-byte SHA-256 validation receipt is
  `checkpoint_backup_validation.json` (private working repository). The six corrected own-v3 checkpoints
  were published under content-addressed Cloudflare R2 prefixes with
  completion-last markers and verified by their per-run checkpoint manifests,
  completion receipts, and publication receipts. This does not make weights
  part of the public export.
- Large data, checkpoints, and active run directories are private operational
  artifacts and are not published in ordinary Git history.
- The durable corpus backup is an S3-compatible object store; public
  infrastructure documentation intentionally omits access coordinates.

## Untouched engine-truth test — executed 2026-07-28

The sealed session was scored in exactly one pass under the written private
owner resolution:
ten models, one inference each, frozen val-A thresholds, per-model
training-era inference code, one hash-pinned metric implementation, all
integrity checks passing, no retries and no strikes. The per-model reports
and aggregate tables are in [`untouched_test/`](untouched_test/),
summarized in
[`untouched_test/UNTOUCHED_TEST.md`](untouched_test/UNTOUCHED_TEST.md);
the stored predictions, execution logs, and full hash manifest remain in
the private working repository.

The transfer to genuinely untouched engine-truth content (Chapter 6,
53,097 rows) is far below every development-split number. Input-active
maximal support: the best macro AP is 0.2377 (36.9M end-to-end) against a
0.1515 prevalence-chance anchor; the mapped-supervision families (Tier B
frozen 0.19–0.22, end-to-end 0.23–0.24, full-corpus 0.18–0.20) sit above
chance, while the three own-data seeds (0.166–0.170) are close to it.
Collar-0 event F1 is 0.020–0.034 versus a 0.0054 shuffled-event anchor
(4–15× luck, small in absolute terms; 0.063–0.081 on the intersection
support). Micro key-state accuracy at the fixed 0.5 rule (0.57–0.74) is
below the 0.85 always-released baseline for every model, and persistence
remains untouchable on state metrics (0.989 micro) while scoring exactly
zero on collar-0 events — the metric-family separation behaving as
designed on a fresh session.

Two facts recorded before the pass bound the interpretation: the own-data
models in that frozen battery trained on the pre-fix mask-era shards, and the
development sessions come from different chapters than the test's
launch-orb-heavy Chapter 6, so this measures transfer to new content, not
repetition of the development setting. The direct own-v3 reruns are now
complete on val-A and show that repairing the training pixels does not rescue
own-only ranking or timing; their new checkpoints were correctly not applied
retroactively to the spent session. No model, threshold, or metric was
adjusted after seeing the final-test numbers, and the session is never
evaluated again.

## Untouched battery — executed 2026-07-28 (evening)

The pre-registered four-chapter battery
([`UNTOUCHED_BATTERY_PREREGISTRATION.md`](UNTOUCHED_BATTERY_PREREGISTRATION.md),
frozen before its sessions were recorded) was sealed and scored the same
day: sixteen models (the ten frozen checkpoints above plus the six
corrected own-v3 reruns, all eligible under the pre-registered rule), one
pass per model per session; all pre-scoring acceptance conditions were
satisfied, with the documented capture deviations in the sealing record,
and all 64 inference cells completed on the first attempt. The six
own-v3 rows are protocol-locked diagnostics — their frozen thresholds came
from the contract evaluator while battery inference ran the pinned
training-era evaluator, whose probabilities differ (up to 0.1009 absolute
for the Tier-B-initialized family) — so their state/event metrics do not
enter headline comparisons; the primary battery conclusions rest on the
ten original frozen models, whose protocol is unaffected (details in the
battery report's eligibility section). Full tables, the per-chapter gradient, baselines, and
the execution record are in
[`untouched_battery/UNTOUCHED_BATTERY.md`](untouched_battery/UNTOUCHED_BATTERY.md).

Headlines as found: pooled best macro AP 0.2667 (36.9M end-to-end) against
~0.16 pooled prevalence chance, above prevalence on all seven controls and
best on every chapter. Every battery chapter shows above-prevalence action
signal, so the Chapter 6 result was neither an isolated success nor an
isolated failure; per-chapter difficulty varies (14 of 40 comparable
legacy model/chapter AP cells fall below their Chapter 6 counterpart —
Chapter 1 is slightly harder than Chapter 6 for the best model, Chapters
2–4 mostly easier). The own-data families
(mask-era and corrected v3 alike) occupy a narrow 0.15–0.20 per-chapter
band on all chapters including the training-adjacent Chapter 1 anchor —
their limitation is generalization, not content distance. Collar-0 event
F1 is 0.031–0.048 pooled (4–8× the 0.005 shuffled anchor, higher on the
denser intersection support) for every family: the recognition-versus-
timing separation reproduces on all four fresh chapters. The four battery
sessions are spent as untouched surfaces under the one-pass rule.

## Object-store rehydration verification

Object-store authentication was installed on the training node without
printing or modifying credential contents. On 2026-07-26, `rclone check`
from each authoritative S3-compatible object-store prefix to its existing
local destination, using
`--size-only --one-way`, reported zero differences:

| Prefix | Local destination | Matching files | Exact local bytes |
|---|---|---:|---:|
| `corpus/video/` | source-video cache | 234 | 237,033,327,002 |
| `mapped-labels/` | mapped-label cache | 13,772 | 119,259,650 |
| `shards/foreign/` | mapped-video pixel shards | 149 | 97,237,635,963 |
| `shards/own-v2/` | engine-truth shards | 14 | 6,321,205,696 |

The `foreign` and `own-v2` names are legacy storage keys retained for artifact
identity; current prose uses mapped-video and engine-truth terminology.

The complete **archived source corpus** had already reached the training cache
during an earlier multi-GPU-node migration: raw videos, the previously mapped
labels, and the previously built shards all match the object store. This does
**not** mean every raw video was train-ready. At this verification point, only
the ten-video, 40.61-hour curated subset had frozen ResNet features consumable
by the trainer. Full action mapping was complete and validated for all 211
accepted videos, while all-video feature generation still required a final
manifest and completion marker. The object store remained the verified durable
source of truth and recovery path, so no redundant 341 GB byte-for-byte
download was performed.
