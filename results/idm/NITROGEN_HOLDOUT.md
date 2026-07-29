# NitroGen-Only Training-Held-Out Video Validation

Status: nine-video pilot and both matched full-corpus scale arms completed and
independently validated.

## Question

Can the 25.7M-parameter long-context IDM learn from mapped NitroGen alone on a
whole-video mapped holdout, then transfer diagnostically to locally captured
engine truth without using local data for training or checkpoint selection?

This is an architecture and data-pipeline diagnostic, not the project's
engine-truth endpoint. The mapped holdout measures cross-video agreement with
noisy foreign labels; B1 supplies preliminary local-transfer evidence but is
development-only and uses same-surface oracle event thresholds.

## Frozen setup: nine-video pilot

- Training: nine Tier-C NitroGen videos, represented as 376 contiguous runs.
- Holdout: all 16 contiguous runs from the training-held-out video
  `y4nQHqYSObI`.
- Holdout support: 554,304 mapped-label frames for all seven keys.
- Inputs: frozen 512-dimensional ResNet-18 features.
- Temporal model: 1,024-dimensional projection, 2,048-unit GRU, 25.7M
  trainable parameters.
- Context: 128 samples at stride three, centered around the target, spanning
  382 raw frames or about 6.37 seconds.
- Optimization: AdamW, batch 1,536, learning rate 0.0003 with linear decay,
  weight decay 0.01, class-balanced BCE, and 8x transition weight.
- Run: seed 0, fixed endpoint at step 5,250.  Checkpoint selection minimized
  mean validation BCE and evaluated only steps 0 and 5,250.

No locally captured frames or engine-truth labels were used in this run.

Holdout scope: the holdout is at whole-video level. Whether it is also
creator-held-out cannot be established from tracked metadata: the curation
record (`results/rung2_curation.json`, private working repository) and the
fetch provenance record platform and video ID but no uploader identity. The
holdout `y4nQHqYSObI` is a YouTube source; the nine training videos are seven
YouTube and two Twitch sources. The holdout's inferred controller binding
differs from all nine training videos (grab on the left trigger rather than
the right trigger, bind confidence 0.822, the highest of the ten), which is
consistent with a different player but is not proof of a distinct uploader.

## Result

The final endpoint improved mean validation BCE from 0.7305 at initialization
to 0.6454 and was selected as the best checkpoint.  The selected and final
state dictionaries are tensor-identical, so the two evaluation surfaces
produce identical predictions.

| Training-held-out NitroGen video | Macro AP | Prevalence baseline AP | State F1 | Exact event F1 | Event F1 at ±2 frames |
|---|---:|---:|---:|---:|---:|
| Selected/final step 5,250 | **0.2435** | 0.1924 | 0.2745 | 0.0127 | 0.0395 |

Macro AP is 0.0512 absolute, or 26.6 percent relative, above the label-
prevalence baseline.  Every key is above its own prevalence baseline:

| Key | AP | Prevalence baseline | Absolute lift | Exact event F1 | ±2-frame event F1 |
|---|---:|---:|---:|---:|---:|
| Left | 0.2738 | 0.1659 | +0.1079 | 0.0115 | 0.0349 |
| Right | 0.5216 | 0.4058 | +0.1159 | 0.0133 | 0.0356 |
| Up | 0.2949 | 0.2367 | +0.0582 | 0.0123 | 0.0399 |
| Down | 0.0333 | 0.0239 | +0.0093 | 0.0048 | 0.0120 |
| Jump | 0.1679 | 0.1585 | +0.0093 | 0.0204 | 0.0677 |
| Dash | 0.1292 | 0.0943 | +0.0349 | 0.0192 | 0.0637 |
| Grab | 0.2841 | 0.2614 | +0.0227 | 0.0077 | 0.0226 |

The event scores use per-key oracle thresholds chosen on this same holdout.
They are diagnostics, not locked-test estimates.  The ±2 result reuses those
thresholds and only relaxes event matching by two frames.

Two design limits apply to every number above: this is a single seed with no
replication, and every event threshold is selected on the same surface it is
scored on.  The state-AP margin over the prevalence baseline is
threshold-free; the event F1 values are same-surface oracle ceilings.

## Interpretation

This is a positive result for cross-video state recognition: a long-context,
higher-capacity model trained only on mapped NitroGen labels separates actions
better than prevalence on a video excluded from gradient training. It also
shows that the
NitroGen-only loader, run splitting, checkpointing, and evaluation path work
end to end.

It is not yet a strong timing result.  Exact event F1 is 0.0127 even with
oracle thresholds, and allowing two frames raises it only to 0.0395.  Some of
that may be action-label offset or bind noise, but the safe conclusion is that
the current objective learns held-action state much more readily than exact
onset/release timing.  More corpus alone should not be assumed to fix this;
future model selection must explicitly reward transition quality.

The label notice is fundamental: these numbers measure agreement with noisy,
mapped NitroGen labels.  They are neither engine truth, local-transfer
performance, nor a locked final-test result.  A fresh local engine-truth
session remains the decisive endpoint.

## Full-corpus scale arms

The matched scale arms kept the same model, seed, context, held-out `y4n`
video, and exact one-pass endpoint policy. The higher-confidence arm uses 92
videos, 1,062 sessions, and 103.4056 hours; the default all-valid arm uses 210
videos, 1,538 sessions, and 148.3222 hours. Every arm's selected and
fixed-final state dictionaries are tensor-identical, as are their prediction
sidecars. The unflagged endpoint is step 14,265; all-valid contains 32,037,600
training frames and ends at step 20,458.

| Training set | Macro AP | Prevalence AP | State F1 | Exact event F1 | ±2-frame event F1 |
|---|---:|---:|---:|---:|---:|
| Nine-video pilot | 0.2435 | 0.1924 | 0.2745 | 0.0127 | **0.0395** |
| 103.41 h unflagged | 0.2693 | 0.1924 | 0.2888 | 0.0128 | 0.0393 |
| 148.32 h all-valid | **0.2723** | 0.1924 | **0.2986** | **0.0141** | **0.0425** |

| Key | Prevalence | Nine-video AP | Unflagged AP | All-valid AP |
|---|---:|---:|---:|---:|
| Left | 0.1659 | **0.2738** | 0.2523 | 0.2666 |
| Right | 0.4058 | 0.5216 | 0.5156 | **0.5454** |
| Up | 0.2367 | 0.2949 | 0.3642 | **0.3713** |
| Down | 0.0239 | 0.0333 | **0.0373** | 0.0342 |
| Jump | 0.1585 | 0.1679 | 0.1699 | **0.1700** |
| Dash | 0.0943 | 0.1292 | 0.1189 | **0.1354** |
| Grab | 0.2614 | 0.2841 | **0.4267** | 0.3831 |

Both full-corpus arms improve held-state ranking on exactly the same
mapped-label support. Relative to the pilot, all-valid adds 0.0288 AP (11.8
percent) and 0.0241 state F1. Timing changes remain small even for all-valid:
exact event F1 adds 0.0014 and the two-frame score adds 0.0030. This is a
useful positive scaling result with a precise limit—not evidence that corpus
scale alone solves onset/release timing.

At the natural 0.5 threshold, unflagged reaches 68.48/11.34 percent per-key
micro/joint accuracy and all-valid reaches 66.14/10.19 percent. Always
released scores 80.76/19.21 percent and one-frame persistence scores
98.73/91.45 percent on identical support. The higher-AP all-valid model has
lower natural-threshold accuracy, another reason not to treat sparse-state
accuracy as a complete performance measure.

The full-corpus checkpoints were also scored on B1 without using B1 for
gradient training or checkpoint selection. B1 is engine truth but remains a development
surface, and its event thresholds are oracle-selected on B1 itself:

| B1 development result | Macro AP | Prevalence AP | State F1 | Exact event F1 | ±2-frame event F1 |
|---|---:|---:|---:|---:|---:|
| 103.41 h unflagged | 0.2603 | 0.1448 | **0.2788** | **0.1228** | **0.1448** |
| 148.32 h all-valid | **0.2713** | 0.1448 | 0.2773 | 0.1167 | 0.1356 |

| B1 key | Prevalence | Unflagged AP | All-valid AP |
|---|---:|---:|---:|
| Left | 0.1498 | 0.3645 | **0.3960** |
| Right | 0.2784 | **0.5080** | 0.4790 |
| Up | 0.0857 | 0.1109 | **0.1609** |
| Down | 0.0499 | **0.1397** | 0.0622 |
| Jump | 0.1101 | 0.1952 | **0.1974** |
| Dash | 0.0512 | **0.0805** | 0.0786 |
| Grab | 0.2887 | 0.4235 | **0.5247** |

The all-valid arm improves B1 AP by 0.0109 over unflagged while state F1 is
flat and oracle timing is slightly lower. This does not support discarding the
fallback-bound data as a default policy, but corpus size and binding cohort
change together, so it is not a controlled proof that those labels are clean.
At 0.5, all-valid scores 60.67/15.36 percent micro/joint, versus
85.52/48.67 percent for always released and 99.04/94.15 percent for
persistence. AP is threshold-free; event scores are same-surface oracle
ceilings and must not be presented as locked local-transfer estimates.

## Wild-provisional comparison on frozen support

A separate single-seed diagnostic trained the same 25.7M GRU recipe on 22.387
hours of provisional, automatically decoded wild-overlay labels. No wild hour
was human-admitted or train-ready. Unlike the oracle event rows above, this
comparison uses final weights and a fixed 0.5 threshold for state and event
decisions.

On the temporally later eight mapped `y4n` streams (269,352 identical frames),
wild versus the 103.41-hour NitroGen reference reaches 0.2316 versus 0.2845 AP,
0.2107 versus 0.2985 state F1, and 0.0052/0.0205 versus 0.0117/0.0385
exact/+/-2 event F1. On B1 active frames, wild loses AP and state F1 but
improves fixed event timing: 0.0460/0.0595 versus 0.0282/0.0468. Its B1
predicted-positive rate is 66.82%, however, versus 42.00% for NitroGen, so the
timing reversal cannot be separated from much more frequent firing. The
wild-only checkpoint is not promoted. See
[`WILD_PROVISIONAL_GRU.md`](WILD_PROVISIONAL_GRU.md) for exact data-hour
boundaries, metrics, and provenance.

## Validation and provenance

- The completion marker exists only after both selected and final evaluator
  runs completed.
- Both reports parse and cover the same 16 runs and 554,304 frames.
- Both prediction sidecars contain finite `float32` probabilities with shape
  `[554304, 7]`, matching `uint8` labels with the same shape.
- Session IDs, lengths, labels, and input-active masks match exactly between
  selected and final; their prediction sidecars are byte-identical.
- The checkpoint records step 5,250 as both the best validation step and final
  endpoint; selected and final tensors compare equal.
- The 205 MB checkpoint is archived locally under
  `checkpoints/nitrogen_holdout/` and is intentionally gitignored.
- Checkpoint SHA-256:
  `359d2bf41e95b51baead7e9d996039f77b5a0c1cf71eb5708e9683a9713a82b1`.
- Reports, prediction sidecars, config, run metadata, and evaluator logs are
  preserved in this results directory in the private working repository. The
  public export includes this document and `checkpoint_sha256.txt` (which
  records the checkpoint hash above) but excludes the per-run JSON reports,
  prediction sidecars, configs, and logs.
- The unflagged scale arm checkpoint SHA-256 is
  `cf55f612382bfa7b9a1b67038b5223a1629782782995f0b008311ba380b34f94`.
  Its frozen split contains the same 16 holdout sessions and 554,304 frames as
  the pilot. Selected/final mapped and B1 prediction arrays are independently
  identical; `nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0/validation.json`
  records the validation receipt and artifact hashes.
- The all-valid checkpoint SHA-256 is
  `297c6a512914946f9d836467b31afa5b84b74e856ad3fb7b2f7326284161fd09`.
  It records step 20,458 as both selected and final, with tensor-identical
  state dictionaries. Its mapped and B1 sidecars contain finite aligned
  arrays, match selected/final byte-for-byte, and preserve the same labels,
  activity gates, and streams as the unflagged evaluations. The validation
  receipt is
  `nitrogen_full_210train_y4n_holdout_26m_128x3_s0/validation.json`.
  The mapped and B1 sidecar SHA-256 values are respectively
  `e6f81dba9025a9ee9c073979a37fdfd63df34b6a4abfc4fad079cccd014345f0`
  and `663b8185475900d6a0caaca1646b6d60e1d6e4de600366f4d5e8db633bba95de`.
