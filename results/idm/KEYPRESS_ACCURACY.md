# Key-state accuracy and its baselines

Status: retrospective scoring of every completed current-generation training
run for which a raw prediction sidecar was retained. Both matched full-corpus
arms—the 103.41-hour unflagged and 148.32-hour all-valid cohorts—are included.

Corpus shorthand used below: Tier A is the early manually inspected 1–2-hour
mapped pilot, Tier B is the curated 13.45-hour mapped corpus, and Tier C is
the curated 40.61-hour mapped corpus. Every scored surface in this report
(`val-A`, B1, and the held-out mapped video) is a development surface. A sealed
untouched engine-truth test now exists, but the provisional-blend study below
did not access or score it; its separate status is recorded in
[`UNTOUCHED_TEST_PREFLIGHT.md`](UNTOUCHED_TEST_PREFLIGHT.md).

## Two accuracy readings

The VPT paper ([Baker et al., 2022](https://cdn.openai.com/vpt/Paper.pdf))
reports “keypress accuracy” but does not define its aggregation,
threshold, or denominator. Its appendix defines independent two-class on/off
heads and their summed training loss; that does not determine whether the
reported accuracy is averaged over heads or requires the entire key vector to
match. The official release does not contain the paper's evaluator. This report
therefore computes both defensible readings and does not claim either is VPT's:

```text
per-key micro = mean((y_prob >= 0.5) == y_true)
joint exact   = mean(all((y_prob >= 0.5) == y_true, axis=keys))
```

Scoring uses all seven keys on the same `input_active` gameplay rows. No
threshold is fit on evaluation data. Every primary table also carries two
same-support baselines:

- **always released:** predict no key is held;
- **one-frame persistence:** copy the previous true key vector within each
  contiguous stream, resetting to released at stream boundaries.

Persistence is constructed before the activity gate is applied, so an inactive
row can supply the preceding state and time is never compressed. This matters:
excluding stream-start rows only for persistence would inflate the 112.95M
val-A baseline from 98.95%/93.32% to roughly 99.03%/93.64%.

The scorer is `experiments/keypress_accuracy.py`. For example:

```bash
uv run python -m experiments.keypress_accuracy \
  results/idm/foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0_val_a_preds.npz
```

The scorer is part of the public export; the example prediction sidecar is
not (the export excludes `.npz` files), so this command runs as written only
in the private working repository.

## What can and cannot be compared with VPT

[VPT's Figure 3](https://cdn.openai.com/vpt/Paper.pdf#page=5) gives the
following approximate curve, but the unpublished aggregation rule applies to
every point:

| VPT contractor data | Reported “keypress accuracy” |
|---:|---:|
| 1 hour | about 59–60% |
| 10 hours | about 66–67% |
| 100 hours | about 72–73% |
| 1,000 hours | about 87% |
| 1,962 hours | **90.6%** |

The earlier claim that MADELEINE's 67.62% micro result sat on VPT's ten-hour
curve is withdrawn: it assumed, without evidence, that VPT used micro
frame/key accuracy. Joint exact-match is more plausible—among other clues, an
always-released predictor reaches 95.08% micro accuracy on the
[public 6,000-row VPT contractor example](https://github.com/openai/Video-Pre-Training/blob/095519fbd4ee0e9281d19f19601e45629de9ac3f/README.md#L81-L99)—but
that example is not the hidden validation set, so this remains an inference,
not a recovered definition.
[VPT also predicts 20 binary buttons](https://github.com/openai/Video-Pre-Training/blob/095519fbd4ee0e9281d19f19601e45629de9ac3f/lib/actions.py#L8-L33)
while MADELEINE predicts seven controls, so even a confirmed joint exact-match
definition would compare vectors of different dimensionality and prevalence.
Neither reading below is presented as numerically interchangeable with VPT's
curve.

| MADELEINE result at threshold 0.5 | Surface | Model micro | Model joint | Always released micro / joint | Persistence micro / joint |
|---|---|---:|---:|---:|---:|
| Frozen 0.725M, 13.45 h, 3-seed mean | val-A | 60.43% | 2.24% | 82.85% / 33.60% | 98.95% / 93.32% |
| End-to-end 36.9M, 13.45 h | val-A | 67.37% | 13.90% | 82.85% / 33.60% | 98.95% / 93.32% |
| End-to-end 112.95M, 13.45 h | val-A | 67.62% | 12.39% | 82.85% / 33.60% | 98.95% / 93.32% |
| End-to-end 36.9M, 13.45 h | B1 | 58.76% | 7.51% | 84.53% / 42.51% | 99.06% / 94.01% |
| End-to-end 112.95M, 13.45 h | B1 | 58.80% | 7.85% | 84.53% / 42.51% | 99.06% / 94.01% |
| Frozen 25.7M, about 38.01 h | held-out mapped `y4n` | 63.28% | 4.25% | 80.76% / 19.21% | 98.73% / 91.45% |
| Frozen 25.7M, 103.41 h unflagged | held-out mapped `y4n` | **68.48%** | **11.34%** | 80.76% / 19.21% | 98.73% / 91.45% |
| Frozen 25.7M, 103.41 h unflagged | B1 | 61.52% | 13.30% | 85.52% / 48.67% | 99.04% / 94.15% |
| Frozen 25.7M, 148.32 h all-valid | held-out mapped `y4n` | 66.14% | 10.19% | 80.76% / 19.21% | 98.73% / 91.45% |
| Frozen 25.7M, 148.32 h all-valid | B1 | 60.67% | **15.36%** | 85.52% / 48.67% | 99.04% / 94.15% |

### VPT-small exact common support

The 105.7M VPT-small experiment uses a different, explicitly frozen support:
4,224 active corrected own-v3 val-A rows that all compared models can predict.
The VPT final checkpoint is the first model in its direct table to beat always
released on both accuracy readings at threshold 0.5.

| Weights | Micro | Joint | Always released micro / joint | Persistence micro / joint |
|---|---:|---:|---:|---:|
| **VPT-small final, epoch 20** | **84.89%** | **38.35%** | 84.19% / 35.58% | 98.98% / 93.51% |
| VPT-small selected, epoch 2 | 84.19% | 35.58% | 84.19% / 35.58% | 98.98% / 93.51% |
| Native60 VPT-small, 128f, 7,060 steps | **85.21%** | 36.34% | 84.19% / 35.58% | 98.98% / 93.51% |
| Native60 VPT-small, 128f, 2,340 steps | 82.64% | 19.32% | 84.19% / 35.58% | 98.98% / 93.51% |
| Native60 VPT-small, 384f, 2,340 steps | 80.98% | 27.27% | 84.19% / 35.58% | 98.98% / 93.51% |
| 482.13M paper-IDM final, epoch 20 | 80.50% | 20.29% | 84.19% / 35.58% | 98.98% / 93.51% |
| 482.13M paper-IDM selected, epoch 3 | 84.19% | 35.58% | 84.19% / 35.58% | 98.98% / 93.51% |
| 112.95M GRU final | 62.28% | 9.71% | 84.19% / 35.58% | 98.98% / 93.51% |
| 112.95M GRU selected | 65.91% | 8.59% | 84.19% / 35.58% | 98.98% / 93.51% |
| 36.9M GRU final | 64.29% | 10.91% | 84.19% / 35.58% | 98.98% / 93.51% |

This is a real improvement over the majority-state baseline, but not a solved
IDM. VPT-small exceeds always released by only 0.70 micro points, rare-key
coverage is poor, and persistence remains 14.09 micro and 55.16 joint points
higher. See [`VPT_SMALL_113M_RESULTS.md`](VPT_SMALL_113M_RESULTS.md) for AP,
state/event F1, fixed predicted-positive rates, and the failed coverage gate.

The native60 full arm posts the highest micro accuracy in this table, but not
the strongest IDM. It trails old VPT-small by 2.01 joint points, 0.0592 macro
AP, 0.0537 state F1, and 0.0153 tolerant-event F1, while emitting no `dash` or
`down` positives at 0.5. Sparse-state micro accuracy again rewards the
majority-state operating point; see
[`VPT_NATIVE60_THREE_ARM_RESULTS.md`](VPT_NATIVE60_THREE_ARM_RESULTS.md).

The matched 482.13M public-artifact paper-IDM makes the imbalance problem more
obvious. Its lowest-validation-NLL checkpoint is exactly always released. The
final checkpoint predicts positives only for `grab` and `right`, lowering
micro accuracy by 3.69 points and joint accuracy by 15.29 points versus always
released while reaching only 0.1844 macro AP. More parameters on the same
13.45-hour population did not improve either accuracy reading; see
[`VPT_PAPER_IDM_TIER_B_RESULTS.md`](VPT_PAPER_IDM_TIER_B_RESULTS.md).

Neither reading is a flattering headline: the learned models lose to trivial
baselines, while persistence reaches 93–94% joint accuracy despite placing
every transition one frame late and therefore scoring exactly zero at collar-0
event F1. The useful result is the disagreement between metrics: the end-to-end
models show skill in macro AP and transition-event F1 that sparse-state
accuracy hides. B1 also exposes a micro-accuracy generalization drop from
67.62% to 58.80% for the 112.95M model.

A per-key threshold fit on the same evaluation surface remains an oracle
ceiling, not a reportable calibrated result. A later diagnostic predeclared
the first eight whole `y4n` streams as calibration-only and evaluated on the
temporally later eight streams, then transferred the frozen parameters to B1.
Per-key affine calibration puts both full-corpus models narrowly above always
released in accuracy, but suppresses almost every press and collapses macro
state F1 from about 0.30 to 0.06. It does improve equal-mass ECE and achieves
small positive BCE/Brier skill against calibration-only per-key priors, while
fixed-0.5 exact and ±2-frame event F1 both decline. The unflagged dash
calibrator cannot reach 0.5 under its declared logit-clipping domain. The full
protocol, thresholds, precision, recall, positive rates, reliability bins,
proper scoring rules, event scores, and numerical diagnostics are in
[`KEYPRESS_CALIBRATION.md`](KEYPRESS_CALIBRATION.md). Event thresholds in the
training reports remain same-surface oracle diagnostics.

The previously discussed approximately 6% result is exact action-transition
F1, not state accuracy. It measures whether onset and release boundaries land
on the exact frame and must not be compared numerically with either accuracy
reading.

### Aligned-TCN control and LR screen

The authorized follow-up uses final weights and the natural 0.5 threshold. Its
formal comparison was frozen on the later eight mapped `y4n` streams before
the three new checkpoints were scored on B1.

| Output | Surface | Micro / joint accuracy | Always released micro / joint | Persistence micro / joint |
|---|---|---:|---:|---:|
| Weighted TCN, `3e-4` | mapped `y4n` later eight | 66.25% / 9.18% | 80.38% / 19.42% | 98.83% / 92.18% |
| Natural BCE control, `3e-4` | mapped `y4n` later eight | **81.02% / 22.64%** | 80.38% / 19.42% | 98.83% / 92.18% |
| Weighted TCN, `1e-4` | mapped `y4n` later eight | 65.41% / 8.18% | 80.38% / 19.42% | 98.83% / 92.18% |
| Weighted TCN, `1e-3` | mapped `y4n` later eight | 65.66% / 8.49% | 80.38% / 19.42% | 98.83% / 92.18% |
| Weighted TCN, `3e-4` | B1 active only | 50.99% / 3.69% | 85.52% / 48.67% | 99.04% / 94.15% |
| Natural BCE control, `3e-4` | B1 active only | **85.81% / 46.19%** | 85.52% / 48.67% | 99.04% / 94.15% |
| Weighted TCN, `1e-4` | B1 active only | 54.16% / 4.42% | 85.52% / 48.67% | 99.04% / 94.15% |
| Weighted TCN, `1e-3` | B1 active only | 51.27% / 6.61% | 85.52% / 48.67% | 99.04% / 94.15% |

Natural BCE crosses the always-released micro baseline by only 0.64 percentage
points on mapped `y4n` and 0.30 points on B1; it remains below the B1 joint
baseline and has weak fixed state/event F1. The weighted models intentionally
predict many more presses, which improves positive-state and event F1 but makes
raw sparse-state accuracy poor. The accuracy table therefore confirms the
objective tradeoff; it does not select a winner. AP and boundary-safe timing
metrics are in the temporal-redesign results report, which lives in the
private working repository while that comparison is still being finalized.

### Wild-provisional GRU

The wild-only diagnostic used final weights and fixed 0.5 decisions. The
mapped row uses the same later-eight support as the TCN screen and matched
NitroGen reference; B1 is the same frozen active-only support used elsewhere.

| Output | Surface | Micro / joint accuracy | Always released micro / joint | Persistence micro / joint |
|---|---|---:|---:|---:|
| Wild provisional, 22.39 labeled h | mapped `y4n` later eight | 58.41% / 6.87% | 80.38% / 19.42% | 98.83% / 92.18% |
| Matched NitroGen, 103.41 h | mapped `y4n` later eight | **69.60% / 12.33%** | 80.38% / 19.42% | 98.83% / 92.18% |
| Wild provisional, 22.39 labeled h | B1 active only | 38.52% / 0.86% | 85.52% / 48.67% | 99.04% / 94.15% |
| Matched NitroGen, 103.41 h | B1 active only | **61.52% / 13.30%** | 85.52% / 48.67% | 99.04% / 94.15% |

The wild model predicts positive on 35.77% of mapped decisions and 66.82% on
B1, versus 27.18% and 42.00% for the NitroGen reference. That operating-point
shift explains much of its poor sparse-state accuracy and qualifies its B1
event-timing gain. See [`WILD_PROVISIONAL_GRU.md`](WILD_PROVISIONAL_GRU.md)
for the complete multi-metric comparison. The 22.39 hours is provisional
labeled duration; admitted/train-ready wild duration remains zero.

### Provisional source-balanced blends

Two fixed-compute GRUs tested corrected local engine truth at a 10% draw share
and provisional wild labels at a further 20% share. The comparison was frozen
on the identical later-eight mapped-`y4n` support before B1; neither arm met
the preregistered candidate rule, so B1 was not opened for either model.

| Output | Micro / joint accuracy | Always released micro / joint | Persistence micro / joint |
|---|---:|---:|---:|
| Pure NitroGen reference | **69.60% / 12.33%** | 80.38% / 19.42% | 98.83% / 92.18% |
| NitroGen/local 90/10 | 68.30% / 9.76% | 80.38% / 19.42% | 98.83% / 92.18% |
| NitroGen/provisional-wild/local 70/20/10 | 68.92% / 10.68% | 80.38% / 19.42% | 98.83% / 92.18% |

Local exposure reduced macro AP by 0.010609, micro accuracy by 1.30 percentage
points, and joint accuracy by 2.57 points versus pure NitroGen. Adding
provisional wild partially recovered micro/joint accuracy but remained below
pure NitroGen and reduced +/-2 event F1. The local pool had only 159 complete
segment items and was cycled 143.55 times. Posthoc scoring confirmed severe
in-sample memorization and no gain on the 5-item, 480-target-frame corrected
local val-A probe: local-train AP/F1 were about 1.000/0.994–0.995 with BCE
below 0.0062, while that probe reached only 0.223–0.249 AP and 0.205–0.280 F1
with BCE 0.788–0.815. Broader local-domain transfer was not tested. Full AP,
state/event, sampling, baseline, provenance, and decision details are in
[`PROVISIONAL_BLEND_GRU.md`](PROVISIONAL_BLEND_GRU.md).

## Current capacity and context runs

Repeated seeds are reported as mean ± sample standard deviation. Selected
and final are checkpoints from the same training run, not separate models.

| Visual IDM | Surface | Selected micro / joint | Final micro / joint | Always released micro / joint | Persistence micro / joint |
|---|---|---:|---:|---:|---:|
| 0.725M frozen, Tier B 13.45 h, 32 centered, 3 seeds | val-A | 60.43 ± 1.79% / 2.24 ± 0.89% | 46.17 ± 3.37% / 0.95 ± 0.45% | 82.85% / 33.60% | 98.95% / 93.32% |
| 0.725M frozen, Tier C 40.61 h, 32 centered, 3 seeds | val-A | 54.56 ± 4.67% / 1.60 ± 1.39% | 47.83 ± 9.79% / 1.51 ± 0.90% | 82.85% / 33.60% | 98.95% / 93.32% |
| 25.7M frozen, Tier C, 32 centered | val-A | 51.27% / 2.07% | 37.60% / 0.74% | 82.85% / 33.60% | 98.95% / 93.32% |
| 0.725M frozen, Tier C, 128x3 context | val-A common support | 49.05% / 0.00% | 32.97% / 0.00% | 81.77% / 30.67% | 98.68% / 92.00% |
| 25.7M frozen, Tier C, 128x3 context | val-A common support | 57.32% / 2.93% | 50.51% / 0.00% | 81.77% / 30.67% | 98.68% / 92.00% |
| 36.9M end-to-end, Tier B, 32 centered | val-A | **67.37% / 13.90%** | 66.08% / 14.47% | 82.85% / 33.60% | 98.95% / 93.32% |
| 112.95M end-to-end, Tier B, 32 centered | val-A | **67.62% / 12.39%** | 64.35% / 12.60% | 82.85% / 33.60% | 98.95% / 93.32% |
| 36.9M end-to-end, Tier B, 32 centered | B1 | **58.76% / 7.51%** | 57.07% / 5.52% | 84.53% / 42.51% | 99.06% / 94.01% |
| 112.95M end-to-end, Tier B, 32 centered | B1 | **58.80% / 7.85%** | 62.33% / 7.47% | 84.53% / 42.51% | 99.06% / 94.01% |
| 25.7M frozen, 9-train-video NitroGen-only, 128x3 | held-out mapped `y4n` | **63.28% / 4.25%** | 63.28% / 4.25% | 80.76% / 19.21% | 98.73% / 91.45% |
| 25.7M frozen, 103.41 h unflagged, 128x3 | held-out mapped `y4n` | **68.48% / 11.34%** | 68.48% / 11.34% | 80.76% / 19.21% | 98.73% / 91.45% |
| 25.7M frozen, 103.41 h unflagged, 128x3 | B1 | **61.52% / 13.30%** | 61.52% / 13.30% | 85.52% / 48.67% | 99.04% / 94.15% |
| 25.7M frozen, 148.32 h all-valid, 128x3 | held-out mapped `y4n` | **66.14% / 10.19%** | 66.14% / 10.19% | 80.76% / 19.21% | 98.73% / 91.45% |
| 25.7M frozen, 148.32 h all-valid, 128x3 | B1 | **60.67% / 15.36%** | 60.67% / 15.36% | 85.52% / 48.67% | 99.04% / 94.15% |

The selected 0.725M Tier-B micro values are 61.53%, 61.41%, and 58.36%;
their fixed-final values are 49.76%, 43.08%, and 45.67%.  The selected 0.725M
Tier-C micro values are 57.69%, 49.20%, and 56.79%; their fixed-final values are
37.38%, 49.32%, and 56.79%.

## Earlier visual-only runs

Each row is scored at threshold 0.5 on that run's retained development-split
prediction sidecar; the baseline columns are computed on the same support as
the row, which is why they differ slightly across window constructions.

Mask-coverage note (updated 2026-07-28): the own-data-trained legacy rows in
this section used the mask-era feature cache, which retained a readable
input-overlay sliver in two 1710-px-family training sessions. `val-A` itself
was covered correctly, and models trained only on mapped-video pixels never
saw the overlay. Those historical rows remain below and are explicitly
treated as mask-era evidence. The six corrected own-v3 primary reruns are now
complete; their current accuracy rows are bolded below, and their exact
ranking/timing attribution is in [`OWN_V3_RERUN.md`](OWN_V3_RERUN.md).

| Model family | Model micro / joint | Always released micro / joint | Persistence micro / joint |
|---|---:|---:|---:|
| Corrected-grid 2-frame pixels, 3 seeds | 83.32 ± 0.03% / 35.32 ± 0.11% | 83.30% / 35.38% | 99.06% / 93.82% |
| Corrected-grid 16-frame centered pixels, 3 seeds | 83.30 ± 0.00% / 35.39 ± 0.00% | 83.30% / 35.39% | 99.06% / 93.83% |
| Corrected-grid 16-frame past-only pixels, 3 seeds | 83.14 ± 0.26% / 35.34 ± 0.04% | 83.29% / 35.37% | 99.06% / 93.82% |
| Corrected-grid 16-frame data scale: 15 / 25 / 40 min | 83.30% / 35.39% at every scale | 83.30% / 35.39% | 99.06% / 93.83% |
| Later own-data 2-frame | 79.81% / 23.79% | 83.27% / 35.25% | 98.98% / 93.54% |
| Later own-data 32-frame centered | 80.64% / 26.10% | 82.85% / 33.60% | 98.95% / 93.32% |
| Later own-data 32-frame past-only | 80.48% / 26.03% | 82.46% / 32.42% | 98.91% / 93.09% |
| **Corrected own-v3 scratch frozen features, 32 centered, 3 seeds** | **75.52% / 14.73%** | 82.85% / 33.60% | 98.95% / 93.32% |
| **Corrected own-v3 Tier-B-init fine-tune, 3 seeds** | **76.45% / 22.77%** | 82.85% / 33.60% | 98.95% / 93.32% |
| Matched own-only frozen features, 32 centered, 3 seeds (mask-era; legacy) | 76.34 ± 2.47% / 17.25 ± 5.26% | 82.85% / 33.60% | 98.95% / 93.32% |
| Own-data transfer initialization, 32 centered | 75.66% / 16.15% | 82.85% / 33.60% | 98.95% / 93.32% |
| Tier-A pilot, 32 centered, 3 seeds | 53.35 ± 4.85% / 0.67 ± 0.33% | 82.85% / 33.60% | 98.95% / 93.32% |
| Tier-A pilot, 32 past-only | 47.84% / 0.10% | 82.46% / 32.42% | 98.91% / 93.09% |
| Tier-A local fine-tune, 3 seeds | 76.83 ± 0.35% / 14.09 ± 2.36% | 82.85% / 33.60% | 98.95% / 93.32% |
| Mapped 5.9 h, 16 past-only | 56.99% / 1.36% | 82.88% / 33.72% | 98.95% / 93.34% |
| Mapped 5.9 h, 32 centered | 55.43% / 0.76% | 82.85% / 33.60% | 98.95% / 93.32% |
| Mapped 5.9 h, 32 past-only | 58.47% / 1.59% | 82.46% / 32.42% | 98.91% / 93.09% |
| Mapped 5.9 h, centered local fine-tune | 68.27% / 2.17% | 82.85% / 33.60% | 98.95% / 93.32% |
| Mapped 5.9 h, past-only local fine-tune | 67.63% / 2.69% | 82.46% / 32.42% | 98.91% / 93.09% |
| Tier-B centered local fine-tune, 3 seeds (mask-era; legacy) | 77.51 ± 3.45% / 23.32 ± 2.62% | 82.85% / 33.60% | 98.95% / 93.32% |
| Tier-B gentle local fine-tune, 3 seeds | 69.92 ± 6.01% / 9.61 ± 7.92% | 82.85% / 33.60% | 98.95% / 93.32% |
| Tier-B past-only | 68.82% / 1.98% | 82.46% / 32.42% | 98.91% / 93.09% |
| Tier-B past-only local fine-tune | 78.30% / 22.02% | 82.46% / 32.42% | 98.91% / 93.09% |

Exact corrected-grid micro-accuracy seed values:

- 2-frame pixels: 83.30%, 83.30%, 83.36%;
- 16-frame centered pixels: 83.30%, 83.30%, 83.30%;
- 16-frame past-only pixels: 83.29%, 83.29%, 82.84%;
- matched clean-only frozen features: 73.80%, 76.48%, 78.73%;
- Tier-A centered: 52.94%, 58.40%, 48.72%;
- Tier-A fine-tune: 76.65%, 77.23%, 76.61%;
- Tier-B centered fine-tune: 80.61%, 73.80%, 78.14%;
- Tier-B gentle fine-tune: 76.74%, 65.40%, 67.61%.

Exact corrected own-v3 primary seed values (micro / joint):

- scratch: 73.62% / 11.79%, 76.14% / 16.13%, 76.79% / 16.26%;
- Tier-B init: 79.55% / 25.40%, 71.99% / 18.99%, 77.80% / 23.93%.

The clean reruns remain below the common 82.85% / 33.60% always-released
baseline. Their usefulness must therefore be judged with AP, state F1, and
transition metrics rather than sparse-state accuracy alone.

The early approximately 83.3% models numerically match the always-released
baseline: they mostly learned the class majority rather than useful action
recognition.  Conversely, the transition-weighted and class-balanced takeover
models deliberately penalize missed presses more heavily and produce too many
positive decisions for an untuned 0.5 cutoff.  Under the requested raw metric,
that is still a benchmark failure; it is not replaced here with an
accuracy-optimized threshold.

Two separately retained early final-checkpoint diagnostics scored 80.24% for
the 2-frame seed-0 model and 79.95% for the 16-frame centered seed-0 model on
val-A.  The same weights scored 99.88% and 98.75% on their training session,
which is consistent with the memorization diagnosis.

## Action-history diagnostics

These runs are included for inventory completeness but are not VPT-like
visual IDMs: their inputs contain ground-truth prior actions.

| Input | Model micro / joint, 3-seed mean | Always released micro / joint | Persistence micro / joint |
|---|---:|---:|---:|
| Action history, no gap | 99.02 ± 0.02% / 93.59 ± 0.09% | 83.30% / 35.39% | 99.06% / 93.83% |
| Pixels plus action history, no gap | 98.61 ± 0.08% / 91.14 ± 0.47% | 83.30% / 35.39% | 99.06% / 93.83% |
| Action history ending 30 frames early | 86.75 ± 0.10% / 44.35 ± 0.27% | 83.30% / 35.39% | 99.06% / 93.83% |
| Pixels plus 30-frame-gapped history | 86.54 ± 0.36% / 45.41 ± 0.03% | 83.30% / 35.39% | 99.06% / 93.83% |

The ungapped seed values are 99.00%, 99.03%, and 99.02% for history-only and
98.63%, 98.52%, and 98.68% for pixels plus history.  The gapped values are
86.81%, 86.81%, and 86.63% for history-only and 86.78%, 86.12%, and 86.72%
for pixels plus history.

## Coverage and interpretation

The exact inventory covers 63 unique completed, current-generation training
runs with retained raw predictions: 24 corrected grid-v3 runs and 39 takeover
runs.  Re-evaluating one checkpoint on val-A, val-B, or B1 does not create a
new training run, and selected/final checkpoints are deduplicated accordingly.
Smoke tests are excluded.

Earlier grid-v1/v2 weights were trained before the corrected loader and
evaluation pipeline.  Their raw prediction arrays were not retained, so an
exact post-hoc accuracy cannot be recovered from the aggregate AP/F1 reports
without rerunning obsolete inference.  They are not silently represented by
the corrected grid-v3 scores.

The direct conclusion under these benchmarks is:

- the approximately 6% number was the wrong quantity to compare with an
  accuracy number;
- current visual models score approximately 59–68% micro and 2–14% joint on
  their principal cross-session surfaces at the natural 0.5 decision rule;
- both figures lose to always released, while persistence reaches 93–94%
  joint accuracy and zero exact transition F1;
- VPT's published aggregation cannot be recovered from the paper or official
  release, so neither MADELEINE reading is labeled apples-to-apples;
- the completed 103-hour arm improves mapped-holdout micro accuracy from
  63.28% to 68.48% and joint accuracy from 4.25% to 11.34%; the 148-hour arm
  reaches 66.14%/10.19%, so more data does not monotonically improve this
  uncalibrated threshold metric;
- both full-corpus arms still lose to always released and persistence despite
  improving threshold-free AP, which is why every metric retains baselines;
- the full-corpus comparison is a within-project scale test, not movement
  along a supposedly matched VPT curve;
- every future evaluation should report micro and joint accuracy, always-
  released and persistence baselines, AP, state F1, and transition F1.

## VPT-small `down` diagnosis on separate foreign support

The frozen y4n mapped-foreign development scorecard contains 555,840 active
rows with 2.3901% `down` prevalence. At the unchanged 0.5 threshold:

| Endpoint | Down AP | Down recall | Down PPR | PPR / prevalence |
|---|---:|---:|---:|---:|
| Tier-B/Wild 13.45h | **0.4591** | **0.2116** | 0.6264% | 0.262 |
| NitroGen unflagged 103.41h | 0.0441 | 0.0044 | 0.2432% | 0.102 |
| 103.41h + 5% Ridge fine-tune | 0.0429 | 0.0233 | 0.7929% | 0.332 |

All three endpoints make nonzero `down` predictions on this foreign support,
whereas all three fixed native val-A reports have zero `down` recall at 0.5.
The smaller Tier-B/Wild endpoint's strong 0.4591 foreign down AP demonstrates
that the seven independent binary heads can represent down jointly with other
keys. The 103-hour result shows that raw positive exposure is not sufficient:
heterogeneous mapping quality and population shift can overwhelm additional
hours. The Ridge fine-tune changes the operating point but does not improve
ranking (down AP 0.0441 -> 0.0429), so it is not a repair.

None of the three passes the 0.5--2.0 PPR/prevalence condition for `down` on
y4n. This remains a development capability measurement, not a gate pass, and
the y4n vertical-axis sign is indeterminate. See the full separate scorecard
at [`vpt_small_foreign_scorecard_v1/SCORECARD.md`](vpt_small_foreign_scorecard_v1/SCORECARD.md).
