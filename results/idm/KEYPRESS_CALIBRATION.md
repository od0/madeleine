# Per-key affine calibration diagnostic

Status: complete development-only diagnostic for the two matched full-corpus
IDMs. This is a probability-calibration result, not an untouched-test result
and not a model-quality improvement.

## Question

The full-corpus models were trained with per-key positive weighting and an
additional transition weight. Is part of their poor accuracy at the raw
`probability >= 0.5` decision rule a calibration problem, and can a calibrator
be fit without consulting B1 or a future untouched test?

## Frozen protocol

The held-out mapped NitroGen video `y4nQHqYSObI` was divided before fitting at
whole contiguous-stream boundaries:

- calibration-only: runs `r000` through `r007`, 284,952 active frames;
- development evaluation: runs `r008` through `r015`, 269,352 active frames.

This temporally separated first-half/second-half policy is pinned in
`experiments/splits/y4n_keypress_calibration_roles.json`. It is deliberately
stricter than fitting and scoring on all 554,304 rows. Both halves remain one
mapped-label development video, so this is not a cross-video generalization
claim.

For each key independently, the calibration-only rows fit the monotone map

```text
calibrated_probability = sigmoid(scale * logit(raw_probability) + bias)
```

with `scale > 0` and unweighted binary negative log likelihood. Evaluation
uses the fixed rule `calibrated_probability >= 0.5`; no F1- or
accuracy-maximizing threshold is selected. B1 is scored only after parameters
are frozen. No B1 label enters fitting, and the future untouched engine-truth
test is neither fit nor scored.

Calibration quality uses 15 equal-mass bins per key, with every bin count
retained in the machine reports. Probability skill is measured against seven
constant Bernoulli predictors whose per-key rates come only from the
calibration streams. Transition events are formed within the stored
`session_lengths`, then gated by `input_active`; neither exact nor tolerant
matching may cross a stream boundary. The reports were regenerated with the
boundary-safe tolerant matcher introduced in commit `98f5a42`; every source
sidecar hash and ordered session-ID list is embedded beside its scores.

## Result on the disjoint mapped-label half

The raw values in this table cover only the temporally later eight streams, so
they need not equal the full-video values in `KEYPRESS_ACCURACY.md`.

| Model | Micro accuracy raw → calibrated | Joint accuracy raw → calibrated | Always released micro / joint | Persistence micro / joint |
|---|---:|---:|---:|---:|
| 103.41 h unflagged | 69.60% → **80.79%** | 12.33% → **21.74%** | 80.38% / 19.42% | 98.83% / 92.18% |
| 148.32 h all-valid | 66.92% → **81.01%** | 10.75% → **22.53%** | 80.38% / 19.42% | 98.83% / 92.18% |

The calibrated models narrowly exceed always released on both accuracy
readings. Their probability losses also improve out of sample:

| Model | Binary cross-entropy raw → calibrated | Brier score raw → calibrated |
|---|---:|---:|
| 103.41 h unflagged | 0.5620 → **0.4270** | 0.1894 → **0.1359** |
| 148.32 h all-valid | 0.5797 → **0.4254** | 0.1986 → **0.1357** |

The calibration-prior baseline on this later half scores 0.4482 BCE and
0.1436 Brier. Negative skill means worse than those seven constant
calibration-only priors; positive skill means better:

| Model | Equal-mass macro ECE raw → calibrated | BCE skill raw → calibrated | Brier skill raw → calibrated |
|---|---:|---:|---:|
| 103.41 h unflagged | 0.1816 → **0.0405** | −25.39% → **+4.74%** | −31.94% → **+5.37%** |
| 148.32 h all-valid | 0.2021 → **0.0337** | −29.35% → **+5.08%** | −38.32% → **+5.46%** |

Thus the calibrator improves reliability and proper scoring rules on unseen
streams, not merely thresholded accuracy. The gain over the prior is modest,
which is consistent with weak—but nonzero—visual signal.

But calibration does not create action information. Because every scale is
positive, per-key ranking and average precision are unchanged to machine
precision:

| Model | Macro AP raw = calibrated | State F1 raw → calibrated | Precision raw → calibrated | Recall raw → calibrated | Predicted positive rate raw → calibrated | Truth positive rate |
|---|---:|---:|---:|---:|---:|---:|
| 103.41 h unflagged | 0.2845 | 0.2985 → **0.0555** | 0.3046 → 0.1863 | 0.3799 → **0.0375** | 27.18% → **2.73%** | 19.62% |
| 148.32 h all-valid | 0.2770 | 0.3008 → **0.0652** | 0.2955 → 0.1910 | 0.4413 → **0.0518** | 32.56% → **3.88%** | 19.62% |

Most of the accuracy gain therefore comes from suppressing press predictions.
The natural calibrated 0.5 rule is conservative enough to miss about 95% of
held-key frames. It repairs the interpretation of the output probability, but
it is not a satisfactory action decoder.

At the same fixed 0.5 rule, transition timing also degrades. These are pooled
onset-plus-release macro F1 values, not the oracle-threshold event values in
the training reports:

| Model | Exact event F1 raw → calibrated | ±2-frame event F1 raw → calibrated | Persistence exact / ±2 |
|---|---:|---:|---:|
| 103.41 h unflagged | 0.0117 → **0.0027** | 0.0385 → **0.0082** | 0.0000 / 1.0000 |
| 148.32 h all-valid | 0.0107 → **0.0026** | 0.0371 → **0.0080** | 0.0000 / 1.0000 |

The persistence contrast is the expected warning: every transition is one
frame late, so exact F1 is zero while a ±2 collar makes it perfect.

## Frozen-parameter transfer to B1

B1 remains a repeatedly consulted engine-truth development surface. Its
labels were used only to score the already-frozen calibrators.

| Model | Micro accuracy raw → calibrated | Joint accuracy raw → calibrated | Always released micro / joint | Persistence micro / joint |
|---|---:|---:|---:|---:|
| 103.41 h unflagged | 61.52% → **85.81%** | 13.30% → **49.38%** | 85.52% / 48.67% | 99.04% / 94.15% |
| 148.32 h all-valid | 60.67% → **86.06%** | 15.36% → **50.76%** | 85.52% / 48.67% | 99.04% / 94.15% |

| Model | Macro AP raw = calibrated | State F1 raw → calibrated | Recall raw → calibrated | Predicted positive rate raw → calibrated | Truth positive rate |
|---|---:|---:|---:|---:|---:|
| 103.41 h unflagged | 0.2603 | 0.2788 → **0.0640** | 0.5992 → **0.0476** | 42.00% → **2.36%** | 14.48% |
| 148.32 h all-valid | 0.2713 | 0.2773 → **0.0559** | 0.6007 → **0.0360** | 43.38% → **1.38%** | 14.48% |

| Model | Equal-mass macro ECE raw → calibrated | BCE skill raw → calibrated | Brier skill raw → calibrated |
|---|---:|---:|---:|
| 103.41 h unflagged | 0.3013 → **0.0974** | −57.57% → **+2.06%** | −84.71% → **+1.69%** |
| 148.32 h all-valid | 0.2951 → **0.0960** | −53.84% → **+2.75%** | −79.29% → **+3.19%** |

The frozen calibration-prior baseline on B1 is 0.3966 BCE and 0.1199 Brier.
Its rates still come from mapped `y4n`, not B1 labels.

| Model | Exact event F1 raw → calibrated | ±2-frame event F1 raw → calibrated | Persistence exact / ±2 |
|---|---:|---:|---:|
| 103.41 h unflagged | 0.0282 → **0.0144** | 0.0468 → **0.0189** | 0.0000 / 0.9895 |
| 148.32 h all-valid | 0.0261 → **0.0093** | 0.0416 → **0.0143** | 0.0000 / 0.9895 |

The accuracy improvement transfers, as do lower binary cross-entropy and
Brier scores, but the same recall collapse transfers too. The calibrated
all-valid model exceeds always released by 0.55 percentage point micro and
2.09 points joint; this is real but small compared with persistence and does
not justify calling the action-recovery problem solved.

## Threshold provenance

At a calibrated probability threshold of 0.5, the equivalent thresholds on
the original model probabilities are:

| Key | 103.41 h unflagged | 148.32 h all-valid |
|---|---:|---:|
| Left | 0.977000 | 0.916020 |
| Right | 0.658922 | 0.679176 |
| Up | 0.958176 | 0.960246 |
| Down | 0.999996 | 1.000000 |
| Jump | 0.999443 | 0.992774 |
| Dash | 1.000000 | 0.999597 |
| Grab | 0.717230 | 0.807691 |

These are not accuracy-oracle thresholds. They are the algebraic consequence
of unweighted NLL calibration on the named first-half streams. Their extreme
values quantify how strongly the weighted training objective shifted the raw
logits.

## Numerical diagnostics

The transform clips raw probabilities to `[eps, 1-eps]` before taking a
float64 logit, where `eps = 2.220446049250313e-16`. Across both evaluation
surfaces and models there are:

- zero raw probabilities equal to zero or one;
- zero raw values clipped at either endpoint;
- zero calibrated probabilities saturated exactly to zero or one.

One threshold is nonetheless structurally unreachable. The unflagged model's
dash calibrator has scale 0.045917 and bias −2.319351, so reaching calibrated
0.5 requires raw logit 50.512. The largest logit representable after the
declared clipping policy is 36.044. Therefore even a source probability of
exactly one cannot produce a dash-positive decision under this calibrator; its
reported raw-equivalent threshold rounds to 1.0. This is not a numerical
overflow—the output probabilities remain finite and unsaturated—but it makes
the natural 0.5 dash decision structurally impossible and is another reason
not to treat calibrated accuracy as an action-decoding solution.

The all-valid dash threshold is theoretically reachable (required raw logit
7.816), but neither evaluation surface approaches it: observed calibrated
dash maxima are 0.152 on later `y4n` and 0.118 on B1, producing zero dash
positives there as well.

## Conclusion

Per-key affine calibration explains much of the alarming raw-0.5 accuracy
number: the uncalibrated scores are not posterior probabilities, and a frozen
calibrator improves probability loss and puts both learned models narrowly
above always released on two disjoint development surfaces. It also shows why
calibration cannot be the main solution. Ranking/AP is unchanged, recall and
state F1 collapse, and persistence remains far ahead.

Keep the calibrator as the probability layer and freeze it before final test,
but select any operational decision policy against the metric the application
actually needs. The next architecture must improve ranking and temporal
evidence—especially onset/release recognition—rather than trying to recover
useful actions by moving thresholds alone.

## Reproduction and artifacts

The reusable CLI is `experiments/calibrate_keypress.py`; its focused tests are
in `tests/test_calibrate_keypress.py`. For example:

```bash
uv run python -m experiments.calibrate_keypress \
  results/idm/nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0_selected_nitrogen_val_preds.npz \
  --roles experiments/splits/y4n_keypress_calibration_roles.json \
  --transfer b1_frozen_engine_truth=results/idm/nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0_selected_b1_preds.npz \
  --output results/idm/nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0_keypress_calibration.json
```

The two machine-readable reports retain sidecar hashes, exact stream roles,
fit parameters and convergence, raw-equivalent thresholds, all pre/post
metrics, explicit equal-mass bin counts, calibration-prior BCE/Brier and skill,
segment-bounded exact/±2 event scores, probability clipping/saturation counts,
baseline metrics, and explicit declarations that B1/test labels were not used
for fitting.
