# VPT-small versus the 112.95M GRU

Status: **complete; implementation success, scientific candidate gate failed**.

This is the affordable VPT-topology experiment frozen in
[`VPT_SMALL_113M_PLAN.md`](VPT_SMALL_113M_PLAN.md). It is not another
post-pooling temporal head. The 105,696,398-parameter model consumes 128 raw
128x128 frames at 20 Hz, begins with the paper's noncausal 5x1x1 Conv3D,
uses the Appendix-D spatial stack and four fully unmasked Transformer blocks,
predicts all seven keys densely at every position, and trains with natural
per-key NLL for exactly 20 epochs.

## Outcome

The final VPT-small checkpoint materially beats both final GRUs in ranking and
beats the stronger 36.9M final GRU in tolerant event timing while matching its
state F1. It also becomes the first model in this direct table to beat the
always-released baseline on both fixed-0.5 micro and joint key-state accuracy.

It nevertheless fails the preregistered candidate gate because its operating
point is too conservative and uneven: `down` recall is zero, and the
predicted-positive/prevalence ratio is outside the required `[0.5, 2.0]`
range for `dash`, `down`, `left`, and `up`. Five of the six clauses pass; the
result lands as a successful architecture/recipe experiment, not a candidate
authorized for new seeds or untouched evaluation.

`Down` AP is 0.0102 against 0.0137 prevalence, so its ordering is already at
or below the chance anchor. A positive-slope scalar calibration can reposition
the other conservative keys, but it cannot manufacture the missing `down`
ranking signal.

All numbers below are development-only, fixed-threshold results on 4,224
identical active rows from 21 corrected own-v3 val-A phase streams. Tier-B
training labels remain noisy mapped labels. No B1, val-B, wild, spent
untouched, sealed-battery, fitted-threshold, or oracle-threshold data was
accessed.

## Exact common-support table

| Weights | Natural NLL | Brier | Macro AP | State F1 | Event F1 exact | Event F1 +/-2 native | Micro accuracy | Joint exact | PPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **VPT-small final, epoch 20** | **3.4139** | **0.1240** | **0.3603** | **0.2504** | 0.0351 | **0.1005** | **84.89%** | **38.35%** | 9.23% |
| VPT-small selected, epoch 2 | 2.6885 | 0.1193 | 0.2080 | 0.0000 | 0.0000 | 0.0000 | 84.19% | 35.58% | 0.00% |
| 112.95M GRU final | 4.6135 | 0.2348 | 0.1990 | 0.1970 | 0.0232 | 0.0615 | 62.28% | 9.71% | 41.03% |
| 112.95M GRU selected | 4.4120 | 0.2219 | 0.1990 | 0.1896 | 0.0139 | 0.0308 | 65.91% | 8.59% | 35.12% |
| 36.9M GRU final | 4.3033 | 0.2170 | 0.2622 | 0.2491 | **0.0378** | 0.0810 | 64.29% | 10.91% | 34.99% |
| Always released | — | — | prevalence 0.1581 | 0.0000 | 0.0000 | 0.0000 | 84.19% | 35.58% | 0.00% |
| One-frame persistence | — | — | — | — | 0.0000 | — | 98.98% | 93.51% | — |

The final VPT beats the better final GRU by +0.0981 macro AP, +0.0013 state
F1, and +0.0196 +/-2-frame event F1. Its exact event F1 is 0.0027 below the
36.9M GRU. Fixed-0.5 micro accuracy is +0.70 percentage points above always
released and joint exact is +2.77 points above it.

The lowest-validation-NLL checkpoint is a useful warning rather than the
headline: epoch 2 minimizes natural NLL by predicting every key released at
0.5. The preregistration made final weights authoritative, preventing that
majority-state solution from replacing the trained endpoint. Training NLL
fell from 3.8150 at epoch 1 to 1.3173 at epoch 20 while whole-window validation
NLL reached its minimum at epoch 2 and rose to 3.4676, so the run also shows
substantial late overfitting.

## Per-key ranking

| Key | VPT final AP | 112.95M final AP | 36.9M final AP |
|---|---:|---:|---:|
| Dash | 0.3775 | 0.1000 | **0.3860** |
| Down | 0.0102 | **0.0242** | 0.0113 |
| Grab | 0.5077 | **0.5389** | 0.4230 |
| Jump | **0.3592** | 0.1526 | 0.2261 |
| Left | **0.3078** | 0.1286 | 0.1837 |
| Right | **0.7392** | 0.2986 | 0.4201 |
| Up | **0.2202** | 0.1501 | 0.1850 |

VPT improves five of seven keys over the named 112.95M capacity comparison.
The aggregate gain is not uniform: `down` remains effectively unsolved, and
`grab` ranking is slightly worse than the 112.95M GRU.

## Frozen six-part decision

| Clause | Result | Evidence |
|---|---|---|
| Macro AP at least +0.010 over better final GRU | Pass | 0.3603 vs 0.2622 |
| State F1 no more than 0.010 below better final GRU | Pass | 0.2504 vs 0.2491 |
| +/-2 event F1 no more than 0.010 below better final GRU | Pass | 0.1005 vs 0.0810 |
| At least four per-key AP improvements over 112.95M | Pass | 5/7 |
| Micro beats always released; joint within one point | Pass | 84.89%/38.35% vs 84.19%/35.58% |
| Nonzero recall and PPR/prevalence in `[0.5,2.0]` for every key | **Fail** | `down` recall 0; four keys outside rate band |

The complete machine-readable calculation is
[`VPT_SMALL_113M_RELEASE_VALIDATION.json`](VPT_SMALL_113M_RELEASE_VALIDATION.json).
It also proves that all five prediction sidecars have identical labels, masks,
row IDs, engine indices, stream boundaries, finite arrays, and 4,224 unique
source rows.

## Training and provenance

- Run: `vpt_small_105696398_tier_b_13p45h_s0`
- Source commit at training: `909489bacbf11925c36a8d351e2517b1476f41ee`
- Hardware: 2x H100 80GB PCIe, DDP world size 2
- Global batch: 128 (`4` sequences/rank/microbatch, accumulation `16`)
- Endpoint: 20 epochs, 2,340 optimizer steps
- Training time: 8,563.09 seconds (2.3786 hours)
- Mean throughput: 34.978 sequences/second
- Peak allocated CUDA memory: 22,941,905,920 bytes
- Final checkpoint: epoch 20, 1,268,535,851 bytes,
  `38303d995e60495fc50bc9f1dfc0cc7f518c5a82dbefaa5da01f030ab3c27c7b`
- Selected checkpoint: epoch 2, 1,268,535,851 bytes,
  `d21ba286b2c89bf1cd86d654a375e263c0283cd8b0f9284ac7f0756ef7865fcf`

All 20 epoch checkpoints were uploaded immutably to the frozen Cloudflare R2
run prefix as they completed. Every receipt records equality between the local
SHA-256 and an independent streamed R2 SHA-256. The local repository retains
the compact run metadata, history, reports, sidecars, validation, and all 20
publication receipts; the 25.4 GB of checkpoint payloads remain in R2.

## Interpretation

The central architecture claim is positive: the actual small VPT topology and
multi-epoch recipe produce much better action ranking and a far saner fixed
operating point than the similarly sized GRU on exact support. This directly
answers the criticism that the project had only diagnosed timing with a
conventional ResNet/GRU.

The model is not yet a release-quality labeler. It still misses rare controls,
its exact event score remains only 0.035, one-frame persistence is far above
all learned models on state accuracy, and one development session cannot
establish generalization. The preregistered response is therefore to stop and
preserve the result. The next branch is frozen in
[`VPT_SMALL_CALIBRATION_RETRAIN_PREREG.md`](VPT_SMALL_CALIBRATION_RETRAIN_PREREG.md):
fit scalar calibration on a new calibration capture, evaluate prospectively
on a second new development capture, and trigger one matched
rare-positive-sampling retrain only if calibration does not satisfy every
guard. It does not tune on val-A or spend an existing untouched surface.
