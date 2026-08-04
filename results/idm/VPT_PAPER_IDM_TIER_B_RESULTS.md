# VPT paper-IDM matched Tier-B result

Status: **complete; implementation gate passed, matched-data scientific result
negative.** The exact public-artifact graph trained to the frozen 20-epoch /
2,340-step endpoint and was evaluated at fixed 0.5 on exact corrected own-v3
val-A common support. It substantially underperformed the 105.7M VPT-small
model and both final GRUs in macro AP and event/state F1. This result is not a
candidate model and does not authorize sealed evaluation.

## Frozen run

- run: `vpt_paper_idm_482133390_tier_b_13p45h_s0`
- model: 482,133,390 trainable parameters; released OpenAI 4x-IDM dimensions,
  adapted only to seven Celeste two-class heads
- data: exact 13.45-hour Tier-B generation, 14,921 windows
- recipe: 128 raw 128x128 frames at 20 Hz, natural factored NLL, Adam with
  coupled 0.01 weight decay, initial LR 0.003 linearly decayed to zero, global
  batch 128, seed 0
- hardware: one Flex-start `v6e-4` in `us-east5-a`; microbatch 4 and gradient
  accumulation 32
- endpoint: 20 epochs / 2,340 optimizer steps in 39,900.60 seconds
  (11:05:00.6)
- train NLL: 4.2435 at epoch 1 to 2.7019 at epoch 20
- selection: epoch 3, the lowest training-validation NLL at 2.7298; final
  training-validation NLL was 2.9773

The first production attempt reached step 115 with finite loss but failed
before the first checkpoint because `torch.inference_mode` is incompatible
with XLA FSDPv2 validation parameter materialization. It produced no epoch
checkpoint and no R2 object. The preserved, content-bound mechanical repair
replaced that context with `torch.no_grad`, changed no scientific field, and
restarted production from zero.

## Exact common-support result

All seven sidecars below have identical truth, activity masks, stream lengths,
session IDs, source rows, and source engine-frame indices: 4,224 active rows
from 21 streams, with 4,224 unique source rows. All probabilities are finite.
No threshold was fitted; all state and event decisions use 0.5.

| Model / weights | Natural NLL | Macro AP | State F1 | Exact event F1 | +/-2 native event F1 | Micro / joint accuracy | PPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Paper-IDM final, epoch 20** | **2.9385** | **0.1844** | **0.0840** | **0.0077** | **0.0191** | **80.50% / 20.29%** | **10.66%** |
| Paper-IDM selected, epoch 3 | 2.7238 | 0.1549 | 0.0000 | 0.0000 | 0.0000 | 84.19% / 35.58% | 0.00% |
| VPT-small final, epoch 20 | 3.4139 | 0.3603 | 0.2504 | 0.0351 | 0.1005 | 84.89% / 38.35% | 9.23% |
| VPT-small selected, epoch 2 | 2.6885 | 0.2080 | 0.0000 | 0.0000 | 0.0000 | 84.19% / 35.58% | 0.00% |
| 112.95M GRU final | 4.6135 | 0.1990 | 0.1970 | 0.0232 | 0.0615 | 62.28% / 9.71% | 41.03% |
| 112.95M GRU selected | 4.4120 | 0.1990 | 0.1896 | 0.0139 | 0.0308 | 65.91% / 8.59% | 35.12% |
| 36.9M GRU final | 4.3033 | 0.2622 | 0.2491 | 0.0378 | 0.0810 | 64.29% / 10.91% | 34.99% |

Same-support baselines are 0.1581 macro prevalence AP, 84.19% micro and
35.58% joint for always released, and 98.98% micro and 93.51% joint for
one-frame truth persistence. The paper-IDM final has threshold-free ranking
above prevalence, but its decisions lose to always released on both accuracy
readings and remain far below persistence.

Final paper-IDM minus final VPT-small:

- macro AP: -0.1759
- state F1: -0.1665
- exact event F1: -0.0275
- +/-2-native event F1: -0.0815
- micro accuracy: -4.38 percentage points
- joint accuracy: -18.06 percentage points
- natural NLL: -0.4754, demonstrating that the better proper score did not
  imply better ranking or usable fixed-threshold decisions

Final paper-IDM minus final 36.9M GRU was -0.0778 AP, -0.1651 state F1,
-0.0301 exact event F1, and -0.0619 tolerant event F1. Its higher accuracy is
mostly the consequence of predicting fewer positives, not better action
recovery.

## Per-key ranking and decision rate

| Key | Prevalence | Paper final AP | VPT-small final AP | 36.9M GRU final AP | 112.95M GRU final AP |
|---|---:|---:|---:|---:|---:|
| left | 0.1058 | 0.0926 | 0.3078 | 0.1837 | 0.1286 |
| right | 0.2467 | 0.2315 | 0.7392 | 0.4201 | 0.2986 |
| up | 0.1375 | 0.0951 | 0.2202 | 0.1850 | 0.1501 |
| down | 0.0137 | 0.0288 | 0.0102 | 0.0113 | 0.0242 |
| jump | 0.0821 | 0.1147 | 0.3592 | 0.2261 | 0.1526 |
| dash | 0.0779 | 0.2020 | 0.3775 | 0.3860 | 0.1000 |
| grab | 0.4429 | 0.5259 | 0.5077 | 0.4230 | 0.5389 |

| Key | Prevalence | Paper final PPR | VPT-small final PPR |
|---|---:|---:|---:|
| left | 10.58% | 0.00% | 2.51% |
| right | 24.67% | 20.31% | 26.18% |
| up | 13.75% | 0.00% | 0.99% |
| down | 1.37% | 0.00% | 0.57% |
| jump | 8.21% | 0.00% | 8.12% |
| dash | 7.79% | 0.00% | 0.66% |
| grab | 44.29% | 54.31% | 25.59% |

Only `grab` and `right` cross 0.5 in the paper-IDM final. The selected epoch-3
checkpoint predicts every key released. The final model exceeds prevalence AP
for `down`, `jump`, `dash`, and `grab`; it is below prevalence for `left`,
`right`, and `up`. This is weak, uneven ranking plus an unusable operating
point, not a pure calibration failure.

## Gate and interpretation

The preregistered Phase-2 implementation gate passed:

- exact endpoint and finite values;
- train NLL fell materially;
- XLA completed with the frozen graph set (11 uncached compiles, 152,469
  cached compiles);
- every one of 20 epoch checkpoints was published immutably and independently
  byte-stream SHA-verified in R2;
- final and selected checkpoint state hashes match their evaluation reports;
- all seven comparison sidecars have exact common support.

The matched-data scientific result is nevertheless strongly negative. At
4.56x VPT-small's parameters on the same 13.45-hour population, the public
4x-IDM topology did not extract more transferable action signal. Validation
NLL bottomed at epoch 3 and the final endpoint recovered some ranking while
losing badly on state and event metrics. This is consistent with severe data
starvation and/or optimization mismatch at this scale; one run cannot
separate them.

Under the frozen plan, losing to VPT-small does not itself cancel the
maximum-data experiment. The run did not show catastrophic nonlearning or an
unresolved fidelity/XLA mismatch, so building and validating the maximum
foreign-data generation may proceed. This receipt does **not** authorize the
later multi-host training launch: Phase 3 data receipts and the Phase 4
`v6e-16` versus `v6e-4` topology gate remain mandatory.

## Provenance

- base scientific source: `be16be904ff5488874fa639fed039781a132a77f`
- validation-context repair: `3f31ed19f009e9297ea93ec88d9edb1a61d2cc1a`
- evaluator: `b8e7fd461212e55c746fbb4ada5660e31f30345e`
- sample-order SHA-256:
  `35a50ac79d9513efc17437c76802c0a727e3cd7e39d854bcaf252ce925529fb9`
- final epoch-20 state SHA-256:
  `820804ca56a732b97d1c126d7e86a4a6e48b0503c7b628f5c40863c5d3abdb82`
- selected epoch-3 state SHA-256:
  `a9fab96e2954a8cab5d22c41c8f89b4d0b5dd563dd6b3ee695b847283e0122ad`
- immutable run prefix:
  `private:runs/idm/v1/vpt-paper-idm-tier-b-v1/vpt_paper_idm_482133390_tier_b_13p45h_s0/`
- immutable result prefix:
  `private:results/idm/v1/vpt-paper-idm-tier-b-v1/vpt_paper_idm_482133390_tier_b_13p45h_s0/`

The machine-readable release proof
(`VPT_PAPER_IDM_TIER_B_RELEASE_VALIDATION.json`), the registered hashes
(`checkpoint-index-vpt-paper-idm-tier-b-20260731.json`), and the exact
small receipts and prediction sidecars
(`vpt_paper_idm_482133390_tier_b_13p45h_s0_evidence/`) live in the
private working repository.

TPU training itself cost $59.85. The node existed for 14:59:30.6 and cost
$80.96 at $5.40/hour; the input Hyperdisk ML cost $6.21 and checkpoint
Hyperdisk Balanced cost $0.91. Total direct GCP list cost was **$88.07**. The
exact timestamps, rates, derivations, and verified absence of every named
resource are in the private `lifecycle_receipt.json`.
