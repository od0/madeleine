# Wild-provisional GRU diagnostic

Status: completed and independently validated. This is a noisy-supervision
diagnostic, not an admitted-data result and not a replacement for the
NitroGen-trained checkpoint.

## Question and data boundary

Can labels decoded automatically from public gameplay overlays teach the
same 25.7M-parameter frozen-feature GRU anything that transfers to the frozen
mapped-video holdout and to locally captured engine truth?

The distinction between three hour counts is load-bearing:

- 27.47 hours is the decoded-video envelope present across seven videos;
- 22.387 hours has provisional seven-key label grids (4,835,638 rows in
  2,058 contiguous shards);
- about 18.474 target-hours enters complete 96-window training segments;
- **zero hours is admitted or train-ready**. No human signed these labels.

The run therefore answers a bounded research question about noisy public-data
supervision. It does not promote the source corpus or weaken the admission
policy.

## Frozen experiment

- Model: the matched 25,719,815-parameter GRU over frozen 512-dimensional
  ResNet-18 features.
- Context: 128 samples at stride three, centered on the target.
- Optimization: AdamW, class-balanced BCE, 8x transition weight, seed 0, and
  one fixed 2,598-step pass.
- Selection: final weights only. The final endpoint was also the best
  validation step, and the selected/final state dictionaries are
  tensor-identical.
- Evaluation: raw sigmoid probabilities with state and event thresholds fixed
  at 0.5. No threshold, calibration parameter, or checkpoint was fitted on an
  evaluation surface.
- Order: the temporally later eight mapped `y4n` streams were released first;
  B1 was opened afterward and remained development-only.

## Primary matched result

The matched reference is the 103.41-hour unflagged NitroGen GRU, evaluated on
the identical 269,352-frame later-eight support.

| Final weights, fixed 0.5 | Wild provisional | NitroGen reference | Delta |
|---|---:|---:|---:|
| Macro AP | 0.2316 | **0.2845** | -0.0529 |
| Prevalence chance | 0.1962 | 0.1962 | 0.0000 |
| State F1 | 0.2107 | **0.2985** | -0.0878 |
| Micro key-state accuracy | 58.41% | **69.60%** | -11.18 pp |
| Joint seven-key accuracy | 6.87% | **12.33%** | -5.46 pp |
| Exact event F1 | 0.0052 | **0.0117** | -0.0065 |
| Event F1 at +/-2 frames | 0.0205 | **0.0385** | -0.0180 |
| Mean predicted-positive rate | 35.77% | 27.18% | +8.60 pp |

The wild-trained model is above prevalence chance by 0.0354 AP, so the
decoded labels contain transferable action signal. It nevertheless loses to
the NitroGen reference on every primary mapped metric and overpredicts key
states. It is not promoted.

## B1 engine-truth diagnostic

B1 contains 9,202 active frames in 73 frozen streams and was not used for
training, fitting, selection, or the mapped decision.

| Final weights, fixed 0.5 | Wild provisional | NitroGen reference | Delta |
|---|---:|---:|---:|
| Macro AP | 0.2022 | **0.2603** | -0.0581 |
| Prevalence chance | 0.1448 | 0.1448 | 0.0000 |
| State F1 | 0.2515 | **0.2788** | -0.0274 |
| Micro key-state accuracy | 38.52% | **61.52%** | -22.99 pp |
| Joint seven-key accuracy | 0.86% | **13.30%** | -12.44 pp |
| Exact event F1 | **0.0460** | 0.0282 | +0.0179 |
| Event F1 at +/-2 frames | **0.0595** | 0.0468 | +0.0127 |
| Mean predicted-positive rate | 66.82% | 42.00% | +24.81 pp |

The timing reversal is interesting but not a model win. Wild supervision
raises exact event F1 by about 63% and tolerant event F1 by about 27%, while
substantially degrading ranking, state classification, and joint accuracy.
Because it fires much more often, some timing recall can come simply from
creating many more candidate transitions. This is single-seed evidence on a
repeatedly consulted development set.

## Conclusion and next experiment

Wild labels are not useless: they transfer above chance and expose a possible
timing benefit. Wild-only training is also clearly inferior to the matched
NitroGen reference. The next defensible test is a source-balanced blend, not
naive concatenation: retain a canonical NitroGen/local output head, let wild
labels update a shared temporal representation through a source-specific
head, match compute and local exposure, and compare real aligned wild labels
with a temporally shuffled wild-label control. A fresh sealed engine-truth
capture should adjudicate that experiment; B1 cannot.

The first lower-complexity source-balanced diagnostic is now active before the
auxiliary-head design: an exact NitroGen/local 90/10 arm and an exact
NitroGen/provisional-wild/local 70/20/10 arm, both using the existing GRU and
fixed compute. See [`PROVISIONAL_BLEND_GRU.md`](PROVISIONAL_BLEND_GRU.md). The
already staged sealed session is explicitly embargoed from those arms; any
winner must be frozen before a different fresh session is recorded.

Private receipts include the exact reports, prediction sidecars, checkpoint
hash, feature/split validation, logs, and an independent `validation.json`.
The checkpoint SHA-256 is
`bcdaa1f409cec17a4d6c647bcf251b6d435bc7b323933e8b52e7103ddc9c0ef1`.
