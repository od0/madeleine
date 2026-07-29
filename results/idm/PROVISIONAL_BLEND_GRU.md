# Provisional source-balanced GRU study

Status: complete. The frozen mapped-`y4n` decision is **no arm eligible**.
Neither blend earned B1 evaluation or a fresh-capture follow-up. Exact values
and hashes live in
[`provisional_blend_y4n_decision.json`](provisional_blend_y4n_decision.json);
rounded tables below are explanatory views of that record.

## Verdict

The three-run comparison was frozen before B1 at canonical decision-record
self-hash
`8d87cf91aa0a91415cf034a296694d2daa6694ce47b3805d35b4c076b2b94c2e`.
It uses final weights, raw fixed threshold 0.5, and the identical 269,352-frame
support from the eight named later streams enumerated in the decision record;
the truth SHA-256 is
`f61a0de4076f4683f01494837f01c3e314873ab0d78ee131b43e8e9f6e576a01`.
No fitted threshold, calibration, B1 result, or checkpoint reselection entered
the decision. The literal decision-file SHA-256 is
`b927688b403decb3440e0d30dfa4bcece7f5cad7c8f4930bea22ac14601bb970`.

| Arm | Macro AP | State F1 | Micro acc. | 7-key joint | Pred. positive | Event exact | Event +/-2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pure N | 0.284459 | 0.298497 | 0.695969 | 0.123318 | 0.271757 | 0.011700 | 0.038534 |
| NL 90/10 | 0.273850 | 0.284451 | 0.682976 | 0.097634 | 0.293488 | 0.012184 | 0.038431 |
| NLW 70/20/10 | 0.277124 | 0.278784 | 0.689210 | 0.106775 | 0.283046 | 0.010312 | 0.035751 |

The preregistered gate required all three conditions: macro AP at least 0.005
above pure N, AP improvement on at least four of seven keys, and no more than
0.005 loss in +/-2 event F1.

| Arm versus pure N | Delta macro AP | Per-key AP wins | +/-2 loss | Eligible |
|---|---:|---:|---:|---|
| NL | -0.010609 (fail) | 4/7: left, right, jump, dash (pass) | 0.000103 (pass) | No |
| NLW | -0.007335 (fail) | 2/7: left, right (fail) | 0.002784 (pass) | No |

## Per-key AP

| Row | Left | Right | Up | Down | Jump | Dash | Grab | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prevalence chance | 0.126325 | 0.439065 | 0.268853 | 0.025287 | 0.144777 | 0.104933 | 0.263989 | 0.196176 |
| Pure N | 0.177986 | 0.537660 | 0.451250 | 0.055150 | 0.156063 | 0.144849 | 0.468256 | 0.284459 |
| NL | 0.203125 | 0.541725 | 0.444696 | 0.038232 | 0.162512 | 0.146945 | 0.379715 | 0.273850 |
| NLW | 0.179142 | 0.556267 | 0.435078 | 0.049745 | 0.152561 | 0.139468 | 0.427604 | 0.277124 |

At threshold 0.5, per-key state F1 was:

| Arm | Left | Right | Up | Down | Jump | Dash | Grab | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pure N | 0.211303 | 0.495710 | 0.475743 | 0.091171 | 0.250265 | 0.203321 | 0.361968 | 0.298497 |
| NL | 0.233539 | 0.555471 | 0.463146 | 0.076413 | 0.256611 | 0.213570 | 0.192407 | 0.284451 |
| NLW | 0.160571 | 0.560849 | 0.447921 | 0.074335 | 0.256725 | 0.205936 | 0.245153 | 0.278784 |

## Scientific contrasts

These are fixed-compute, single-seed contrasts, not an additive-hours study or
a full factorial.

| Contrast | Delta AP | Delta state F1 | Delta micro | Delta joint | Delta pred.+ | Delta event exact | Delta event +/-2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Local marginal, NL - N | -0.010609 | -0.014046 | -0.012993 | -0.025684 | +0.021731 | +0.000484 | -0.000103 |
| Provisional-wild marginal, NLW - NL | +0.003274 | -0.005667 | +0.006234 | +0.009140 | -0.010442 | -0.001872 | -0.002680 |

Local exposure improved four APs but sharply reduced grab AP (-0.088541) and
the aggregate. Provisional wild partly recovered AP and grab (+0.047890 versus
NL) and reduced overprediction, but worsened state and event metrics and still
trailed pure N. These conclusions are about noisy mapped labels, not engine
truth.

## Shared baselines

| Baseline | Micro acc. | Joint exact | Event exact | Event +/-2 | Macro AP |
|---|---:|---:|---:|---:|---:|
| Always released | 0.803824 | 0.194177 | - | - | - |
| One-frame persistence | 0.988347 | 0.921846 | 0 | 1 | - |
| Shuffled-event luck, 10 seeds | - | - | 0.006106 | 0.030333 | - |
| Prevalence chance | - | - | - | - | 0.196176 |

Persistence is constructed by delaying truth one frame, so its perfect +/-2
score is a timing-baseline artifact rather than evidence of useful inference.
The shuffled-event anchor preserves onset and release counts independently per
key and stream, samples uniformly without replacement within stream, and uses
one-to-one segment-bounded matching for seeds 0 through 9.

## Sampling and memorization receipts

| Arm/source | Draws | Pool | Unique | Repeats | Effective passes | Draws/item |
|---|---:|---:|---:|---:|---:|---:|
| NL / N | 205,416 | 228,237 | 205,416 | 0 | 0.9000 | 0-1 |
| NL / local own-v3 | 22,824 | 159 | 159 | 22,665 | 143.5472 | 143-144 |
| NLW / N | 159,768 | 228,237 | 159,768 | 0 | 0.7000 | 0-1 |
| NLW / local own-v3 | 22,824 | 159 | 159 | 22,665 | 143.5472 | 143-144 |
| NLW / provisional wild | 45,648 | 41,567 | 41,567 | 4,081 | 1.0982 | 1-2 |

NL's final mixed-batch train BCE was 0.548761 and its all-16 `y4n` validation
BCE was 0.590516. NLW's values were 0.601941 and 0.577884. The tiny local pool's
143.55 effective passes are the central memorization caveat.

Posthoc, decision-inert diagnostics scored every unique complete segment item
once with final weights and no class, transition, repetition, or draw weights:

| Arm / source | Items | Target frames | BCE | AP | State F1 | Prevalence | Pred. positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| NL / NitroGen train | 228,237 | 21,910,752 | 0.524380 | 0.522986 | 0.467178 | 0.180475 | 0.446357 |
| NL / local train | 159 | 15,264 | 0.005640 | 0.999983 | 0.995144 | 0.155473 | 0.156269 |
| NL / local val-A | 5 | 480 | 0.787551 | 0.222605 | 0.204656 | 0.212798 | 0.484821 |
| NLW / NitroGen train | 228,237 | 21,910,752 | 0.541134 | 0.497915 | 0.455951 | 0.180475 | 0.455322 |
| NLW / provisional wild train | 41,567 | 3,990,432 | 0.730431 | 0.527561 | 0.488526 | 0.255899 | 0.653439 |
| NLW / local train | 159 | 15,264 | 0.006200 | 0.999973 | 0.993791 | 0.155473 | 0.156503 |
| NLW / local val-A | 5 | 480 | 0.814705 | 0.248883 | 0.279716 | 0.212798 | 0.640774 |

The local train-versus-val-A gaps provide strong evidence of memorization and
no gain on this 5-item, 480-target-frame corrected local val-A probe. Broader
local-domain transfer was not tested:

| Arm | Val minus train BCE | Train minus val AP | Train minus val F1 |
|---|---:|---:|---:|
| NL | +0.781911 | +0.777378 | +0.790488 |
| NLW | +0.808505 | +0.751090 | +0.714075 |

The wild training pool showed in-sample rank/state signal, but the NLW model
predicted positive on 65.34% of wild key decisions against 25.59% prevalence.
This overprediction is consistent with, but does not by itself explain, the
blend's failure to beat pure NitroGen on the mapped holdout. These posthoc
values were not candidate-rule inputs and cannot alter the frozen result.

## Artifact durability

The two final checkpoints were uploaded to content-addressed Cloudflare R2
prefixes with model, manifest, and marker-last completion receipts. Independent
readback streamed and SHA-256/size-verified all 23 retained checkpoints:
3,933,139,829 bytes and exactly 71 objects. The retained snapshot contains the
previously validated 21 checkpoints plus these two additions:

| Arm | Checkpoint SHA-256 | Bytes | Manifest SHA-256 | R2 completion SHA-256 |
|---|---|---:|---|---|
| NL | `e5e194172a31e3c6a14a2ba9d1c5233c1b37a112504b3275c2e2e1d1a55e7bf9` | 205,802,555 | `1284b5de48f53a1fe7edf67461b1e069e98ab151e7a8ce1238757fe85922acc3` | `a49883dd5417a7df1cdee623fe5fdc125d6af5348067f6a08338969a6b83b8f2` |
| NLW | `b4d7a677fa8cfc981f043f79da7f87dc01536a2a9a942ecb26652d9aa96c7cfd` | 205,893,243 | `1e03bed66bcb13e1c7eb55a1277e2012270716d515c63297c2c3067b6cf91d6c` | `89e1bd62db69efc14ea6a7c5dd45353901f6aefe3a447d19eed67ea15f4ea778` |

The retained-20260728-v1 aggregate index SHA-256 is
`cd62399cddd67f58954abca2fb8436025536f5ad9fb1c20c87bc3047fa9c12d8`;
its marker SHA-256 is
`9d3b0063b758ce078875a7e729a714ba5e84deac07fba1f82f673cfd95f16727`.
The exact artifact IDs, remote prefixes, and validation checks are in
`provisional_blend_checkpoint_backup_validation.json` (private working
repository; it records storage coordinates the public export excludes).

## Evaluation artifact receipts

The frozen run receipts record selected/final checkpoint tensor identity in
each arm. Each final-only evaluation released one validated sidecar.

| Arm | Report SHA-256 | Sidecar SHA-256 | Eval marker SHA-256 | Wrapper SHA-256 |
|---|---|---|---|---|
| NL | `72facf8b4179907d24fd4043dc5e46a13fe85df8d623bc9457aa64a435d66b08` | `17061acac82b568281c7c30f3208887c124d9272e859320b216eeab359bec6c6` | `e32335819ef447bb680c08c057eeed955d9b5ab0ac9d77e7a4e875bd2ccb94be` | `7368cd6b97d3186bec53a1a42827f05d8e3e08c2dae094dd76ac1a2191645d0a` |
| NLW | `68e12f4a9ad15a55af4047647db3ad2ab92875a017e373d0a9e6ba7c1dd072f6` | `d805be6a69528cd1e2c827138d97be52b58fea4aabaceec72774018b300104e3` | `7ddb5d4e922ef4499c2f781ec304c3c3f50a0d0cc03b3c2d44cfb2fa1027ddd5` | `102ba0db14d1b8481af214086f5f5809ae4cf3751d9259aa87490306f7ba4687` |

The report and sidecar paths are linked from the canonical decision record.
The read-only validator ran against the original absolute-path-bound artifacts
on the producing node before sync. It checked the content-bound report/marker
pair, frozen Git/config/run identities, embedded membership/support/sampling
arithmetic, metric structure/count feasibility, local-gap arithmetic, and
forbidden-ID declarations. The tracked validation receipts bind the validator
commit and source hash:

| Arm | Memorization report SHA-256 | Completion marker SHA-256 |
|---|---|---|
| NL | `60364800dc66e389d3373634b941360d28c5fddcede21306a3301c7cae9079e6` | `47721077dd13d345eae125fe59e3873c095c6942815433772bd5f344cbca1e79` |
| NLW | `1a13ba6e113cffc0b75bd9f4e6a6de90d3380197e239d881739823debb159668` | `c754d731dc81c0e129852874829d446b56ba2828312574edf15432ba352aec67` |

Receipts: [`NL validation`](blend_provisional_nl90_10_92train_y4n_holdout_26m_128x3_s0_final_memorization_validation.json)
and [`NLW validation`](blend_provisional_nlw70_20_10_92train_y4n_holdout_26m_128x3_s0_final_memorization_validation.json).
The producer intentionally retained hashes rather than large prediction arrays
for these diagnostics. The validator did not reopen model/data inputs,
independently replay AP/BCE, or audit operating-system access history. The
receipts record no forbidden-ID access; this is not an OS-level access proof.

## Question

Two fixed-compute arms test the marginal value of corrected local engine truth
and provisional wild-overlay labels without changing the existing 25.7M GRU
architecture or training endpoint:

| Arm | Scheduled segment-item mix | Scientific contrast |
|---|---:|---|
| Existing pure NitroGen reference | N 100% | zero-local, zero-wild baseline |
| NL 90/10 | N 90%, local 10% | local's marginal effect versus pure N |
| NLW 70/20/10 | N 70%, wild 20%, local 10% | wild's marginal effect versus NL |

This is a provisional diagnostic, not an admitted-wild-data result. The wild
source contains 22.387 labeled hours in 2,058 shards, but **zero hours have
cleared the human admission gates**. Any later admitted-only blend requires a
new preregistration.

## Frozen recipe

- Existing reference:
  `nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0`, checkpoint SHA-256
  `cf55f612382bfa7b9a1b67038b5223a1629782782995f0b008311ba380b34f94`.
- New arms:
  `blend_provisional_nl90_10_92train_y4n_holdout_26m_128x3_s0` and
  `blend_provisional_nlw70_20_10_92train_y4n_holdout_26m_128x3_s0`.
- Model: the same 25,719,815-parameter centered GRU over frozen 512-D
  ResNet-18 features, 128 samples at stride three, seed 0, from scratch.
- Optimization: AdamW at `3e-4`, weight decay `0.01`, linear decay, transition
  weight 8, exact 14,265-step endpoint, final weights only, and no checkpoint
  selection.
- Positive-class weights: the pure-103h reference vector is frozen unchanged;
  source prevalence cannot silently change the loss scale.
- Sampling: exact deterministic five-step source-quota cycles. Every step has
  16 segment items; each source has an independently shuffled cycling pool.
  The trainer writes scheduled and actual draws, unique items, repeats, and
  effective pool passes into the checkpoint and a dedicated receipt.

NL schedules 205,416 NitroGen and 22,824 local draws. NLW schedules 159,768
NitroGen, 45,648 provisional-wild, and 22,824 local draws. The corrected local
train pool contains only 159 complete segment items, so 22,824 scheduled draws
amount to 143.55 pool passes. That exposure is intentionally matched between
the arms and is also a serious memorization risk that the result must report.

## Data and assembly receipts

Local features were rebuilt exclusively from the corrected own-v3 pixel
generation. The old pre-mask-fix feature cache is forbidden. The content-bound
own-v3 receipt reports four sessions, 178,525 total frames, exact supervision
equality, finite float16 512-D features, byte-preserved split lists, and no
extra or temporary files. Training uses three of those sessions: 143,451
source frames / 0.664125 hours, including 119,859 input-active rows.

The immutable mixed feature view contains 3,140 verified hardlinks:

- 1,062 unflagged NitroGen training sessions;
- 2,058 provisional wild training sessions;
- three corrected own-v3 training sessions plus one local val-A session;
- the 16 frozen mapped-`y4n` validation streams.

Its assembly receipt is `madeleine.provisional-blend-feature-view.v1`, binds
all source validation receipts and generated config/list hashes, reports no
temporary files, and explicitly confirms that the sealed untouched session is
absent.

## Evaluation order and embargo

Both arms first release final-weight predictions on the identical 269,352-row
temporally later-eight mapped-`y4n` support. Decisions are fixed at 0.5; no
calibration, fitted/oracle threshold, or checkpoint reselection is allowed.
The three-run comparison must be committed before B1 can be opened. B1 is
fixed-only, post-decision development evidence and cannot select or promote an
arm.

The sealed session `rec_20260727_220000_test` is forbidden to this study for
training, validation, inference, thresholding, calibration, diagnostics, and
selection. If a blend arm clears the frozen mapped-`y4n` candidate rule, its
recipe is frozen first and Bryan records a different fresh 15-minute session.
Val-B is also excluded until its counter-reset alignment is repaired.

Required primary reporting includes macro/per-key AP and prevalence, state F1,
micro and seven-key joint accuracy, segment-bounded exact and +/-2-frame event
F1, predicted-positive rate, always-released/persistence/shuffled-event
baselines, and the full source-sampling/memorization receipts.

## Execution receipt

The mixed view completed at 2026-07-28 00:04 UTC. NL launched on GPU 0 and NLW
on GPU 1 from the same clean frozen source tree. Both exact trainers reached
sustained 100% GPU utilization; allocation grew from about 8.5 GiB during
startup to 26.1 GiB once training batches were fully resident. A recurring
monitor checked exact processes, RAM/GPU health, output markers, and evaluation
order through completion. Both trainers reached the exact 14,265-step endpoint,
and both wrappers exited cleanly. The frozen run receipts record
selected/final tensor identity; each final-only evaluation emitted one
validated sidecar.

The canonical comparison and exact compact evidence were committed as
`de11377` before the posthoc diagnostic implementation commit `d86cba9` and
its later execution. B1 remained unopened because neither arm met the candidate
rule. The machine-local feature view is an operational artifact; checkpoint
durability is covered by the independently verified R2 receipts above.
