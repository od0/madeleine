# Oracle-window transition localization

## Status and scientific question

This study tests whether knowing that one requested key onset or release
occurred in a short region makes its exact engine-truth frame more recoverable
than the repository's dense independent event formulation. Phase 1 is the
decision experiment. Phase 2, the full coarse-to-fine cascade, is forbidden
unless the preregistered Phase-1 gate passes.

The decision contract was frozen before any final val-A model predictions or
metrics were examined. Dataset support and exclusion counts were inspected
before the freeze because they determine whether the experiment is estimable;
they are not model outcomes. Val-A is development evidence, not an untouched
test. Val-B, B1, and the sealed engine-truth session are excluded.

## Frozen Phase-1 design

Each example names one of fourteen tasks (seven keys times onset/release) and
contains a 16-frame candidate region with an eight-frame visual halo on both
sides. For an event at frame `t`, eligibility is decided before assigning its
offset: the full union `t-23..t+23` must be consecutive and input-active, and
`t-15..t+15` may contain no second event for the requested key and polarity.
The retained event is then assigned exactly one pseudorandom offset, balanced
within session/task to a maximum bin-count difference of one. The model sees
only the verified features and requested task, never the session, absolute
frame, crop start, or target offset.

The corrected own-v3 split contributes 4,554 training examples and 1,150
val-A examples under this strict all-offset-safe rule. The known engine-counter
fragmentation in `rec_20260725_015612` leaves it with zero eligible 32-frame
crops, so effective training support comes from two sessions. All fourteen
tasks remain represented, but val-A has only 11 `down:onset` and 10
`down:release` examples; non-estimable tasks remain visible in descriptive
tables but cannot help the decision gate.

The matched arms share exact initialization, input crops, 512-dimensional
masked ResNet-18 features, `512 -> 128` projection, existing aligned TCN with
dilations `[1, 2, 3]` and exact radius-eight support, fourteen linear event
heads, minibatch order, AdamW optimizer, task-normalized example weights, and
fixed 40-epoch endpoint. They differ only in objective:

- dense control: independent BCE over the 16 requested-head logits, with one
  positive, fifteen negatives, and fixed positive weight 15;
- conditional localizer: 16-way softmax cross-entropy for the exact offset.

The retained current event-latch checkpoint is also restricted to each oracle
region on the subset with its full 189-frame past and 192-frame future support.
It is an unmatched historical reference—it was trained on mapped NitroGen with
much longer context—and is excluded from the causal decision gate.

## Preregistered metrics and gate

Chance at width 16 is 0.0625 exact, 0.1796875 within one frame, 0.2890625
within two frames, and `ln(16) = 2.772589` for both uniform NLL and entropy.
The report includes pooled and equal-task-macro exact/within-one/within-two
accuracy, NLL, entropy, entropy versus correctness, signed early/late error,
16-by-16 confusion, and every key/polarity task.

Confidence intervals use 5,000 deterministic paired resamples of 600-engine-
frame blocks, never crossing a session or continuity boundary. The sole
primary comparison is conditional softmax minus matched dense BCE in macro
exact accuracy; per-task intervals are descriptive.

A task is estimable with at least 30 validation events and at least 20
training events in two sessions. The study must have at least seven estimable
tasks spanning four keys and both polarities, plus at least 20 validation
blocks. Seed zero earns confirmation seeds 1 and 2 only if all of these hold:

1. conditional macro exact accuracy is at least 0.125 and its 95% lower bound
   is above 0.0625;
2. conditional-minus-dense macro exact improvement is at least 0.03 and its
   paired 95% lower bound is above zero;
3. at least seven estimable tasks improve, spanning four keys and both event
   types;
4. conditional macro within-two accuracy is no more than 0.01 below dense;
5. conditional macro NLL is lower than dense.

Phase 2 advances only if unchanged seeds 1 and 2 reproduce the gate on the
three-seed arithmetic mean, exact delta is positive in at least two seeds, and
the paired block-bootstrap lower bound remains positive.

## Reproduction

The machine-readable authority is
`experiments/configs/oracle_window_localization_decision.json`. The launcher is
`experiments/run_oracle_window_localization.sh`; the trainer and fixed-policy
scorer are `experiments/oracle_window_localization.py` and
`experiments/score_oracle_window_localization.py`. The feature-generation
receipt binds the exact corrected supervision and the locally generated MPS
feature bytes. No checkpoint is selected on val-A; only final weights are
scored.

## Results

Seed zero completed locally on MPS in 174.71 seconds with a 1.09 GB maximum
resident set. The support was gate-capable: 1,150 val-A transitions in 122
continuity-bounded blocks, with 12 estimable key/polarity tasks spanning six
keys and both event types.

The conditional objective did not improve exact timing. On estimable tasks its
macro exact accuracy was 0.07369 versus 0.07534 for matched dense BCE. The
paired difference was -0.00165 with 95% interval [-0.02214, 0.01918]. The
conditional exact interval [0.05862, 0.08945] crossed chance and its point
estimate missed the required 0.125. Only five of twelve estimable tasks
improved. Macro within-two fell from 0.32710 to 0.29852, and NLL worsened from
5.195 to 6.393 (uniform chance: 2.773). Every joint gate check therefore
failed except breadth by distinct keys and event types. Seeds 1 and 2 and the
full cascade are rejected.

The conditional model was sharper without being better calibrated: pooled
entropy was 0.981 nats versus 1.315 for dense BCE, while its pooled validation
NLL was 6.426. Training CE reached 0.000544, consistent with cross-session
memorization and overconfidence. Conditional entropy quartiles had exact
accuracies 0.0625, 0.0694, 0.0871, and 0.0801, so low entropy did not identify
correct predictions. Pooled early and late error rates were both 0.4626 and
mean signed error was +0.077 frame; there is no coherent early/late latency
signal. The unmatched historical checkpoint covered only 46 examples and is
excluded from the decision.

The original completion marker is supplemented by a post-run audit that binds
the run receipt and both checkpoint hashes. It exactly reconstructed every
example identity, replayed both saved models and the retained reference on MPS,
and regenerated the 5,000-replicate report without changing the rejection.

## Preregistered bounded pixel-differential follow-up

Because the pooled-feature localizer did not clearly exceed chance, the brief
requires one bounded learned native-rate frame-pair diagnostic before treating
the null as evidence about pixels. This follow-up was frozen at 2026-07-28
05:12:41 UTC, before any follow-up validation inference. It cannot reopen
Phase 2 or authorize more seeds regardless of its result.

The exact 4,554/1,150 oracle examples, offsets, split, task weights, and block
IDs are unchanged. Exact already-masked 128x128 source pixels are reduced to
32x32 by a deterministic per-channel 4x4 integer area mean. The resulting
content-bound cache is 184 MB compressed (560,726,016 raw pixel bytes), and
its validator rechecked the source hashes, both answer-key masks, supervision,
forbidden-session absence, reconstructed manifest, and every crop identity.
A null is explicitly scoped to this fixed spatial reduction and small adapter.

For each of 31 adjacent frame pairs, the primary arm receives ordered
`[previous RGB, current RGB, current - previous]`. A matched symmetric-pair arm
receives `[mean(previous,current), mean(previous,current), zero]`, preserving
exact two-frame support and symmetric spatial appearance while removing order
and signed motion. Both use the same approximately 0.1M-
parameter pair encoder and a shared valid width-16 temporal convolution that
maps 31 pairs to 16 offset logits without padding or positional inputs. They
share exact initialization, examples, task weights, epoch permutations,
optimizer, cross-entropy objective, and fixed endpoint: seed zero, 20 epochs,
batch 128, AdamW at 1e-3 with weight decay 1e-4, final weights only, no
augmentation, scheduler, early stopping, or validation selection.

The primary rescue gate compares ordered pairs against the already frozen
feature-conditional probabilities on identical rows. It reuses the original
materiality rule: macro exact at least 0.125 with 95% lower bound above 0.0625;
macro exact improvement at least 0.03 with paired lower bound above zero; at
least seven improving estimable tasks spanning four keys and both event types;
within-two degradation no worse than 0.01; and lower NLL. Only after that gate
passes can ordered-pair minus symmetric-pair establish differential attribution,
using the same delta, breadth, within-two, and NLL requirements. A symmetric-pair
rescue without ordered-pair rescue rejects the pair hypothesis; neither arm
passing is reported as no bounded pixel rescue, not proof of intrinsic visual
ambiguity. All intervals again use 5,000 paired whole-block resamples.

The machine authority is
`experiments/configs/oracle_window_differential_followup_decision.json` and its
amended pre-inference SHA-256 is
`cac63698326a3dbe1d64b8158a25a563aeb514294ec06ef295e6321d9fe338c8`.
An initial production attempt was interrupted after 241 seconds, before output
creation or validation inference, when an independent audit found that the
former current/current/zero control omitted one boundary frame available to
the ordered arm. The amendment above fixes support parity and adds exact hashes
for every imported construction, metric, bootstrap, and gate dependency. The
aborted attempt is not evidence and its path was never created.
The smoke may proceed only if it projects the unchanged endpoint below 30
minutes and all new cache/run artifacts below 600 MB; otherwise the follow-up
is runtime-inconclusive rather than adapted.

## Bounded follow-up result and final decision

The corrected seed-zero production run completed locally on MPS in 419.52
seconds with a 1.04 GB maximum resident set. It scored the same 1,150 val-A
examples, 122 continuity-bounded bootstrap blocks, and 12 estimable heads as
Phase 1. Validation offset counts ranged from 69 to 74 across all 16 offsets;
the construction therefore did not encode a useful fixed-position guess.

Across the training split, 9,594 raw key transitions yielded 4,554 retained
examples: 2,697 lacked a valid predecessor, 2,013 failed the all-offset-safe
boundary/gap/activity rule, and 330 had another same-head event in the
ambiguity region. Val-A had 2,228 raw transitions, with 288, 528, and 262
excluded for those respective reasons, leaving 1,150. The fragmented training
session `rec_20260725_015612` supplied no eligible example; this limitation was
known before either model outcome was inspected.

The primary table is the equal-head macro over the 12 preregistered estimable
key/event heads. The empirical uniform row uses the held-out offset histogram;
its exact and NLL values are the theoretical width-16 values.

| Oracle-window predictor | Exact | Within 1 | Within 2 | NLL | Entropy |
|---|---:|---:|---:|---:|---:|
| Uniform chance | 0.06250 | 0.17984 | 0.28924 | 2.77259 | 2.77259 |
| Frozen-feature conditional | 0.07369 | 0.19830 | 0.29852 | 6.39319 | 0.98018 |
| Symmetric two-frame pixels | 0.07837 | 0.19870 | 0.30524 | 3.83817 | 2.08743 |
| Ordered pixel pairs + signed difference | **0.07862** | **0.20276** | **0.31430** | 5.29163 | 1.43417 |

Per-head ordered-arm metrics and the two exact-accuracy controls are below.
The machine report additionally retains every control's within-one, within-two,
NLL, entropy, confusion matrix, and block-bootstrap comparison. Down onset and
release have only 11 and 10 examples and remain explicitly non-estimable.

| Estimable head | n | Frozen exact | Symmetric exact | Ordered exact | Ordered ±1 | Ordered ±2 | Ordered NLL | Ordered H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| left onset | 79 | 0.0506 | 0.0633 | 0.0633 | 0.1266 | 0.2405 | 5.906 | 1.580 |
| right onset | 149 | 0.0537 | 0.0537 | 0.0470 | 0.1611 | 0.2752 | 5.092 | 1.655 |
| up onset | 102 | 0.0686 | 0.0784 | 0.0980 | 0.2353 | 0.3922 | 5.007 | 1.405 |
| jump onset | 108 | 0.0926 | 0.0648 | 0.1019 | 0.1944 | 0.2593 | 5.451 | 1.488 |
| dash onset | 87 | 0.1609 | 0.2069 | 0.1494 | 0.4023 | 0.5747 | 5.586 | 0.842 |
| grab onset | 82 | 0.0610 | 0.0610 | 0.0488 | 0.2073 | 0.2683 | 5.161 | 1.370 |
| left release | 61 | 0.0328 | 0.0656 | 0.0820 | 0.1311 | 0.1967 | 4.320 | 1.653 |
| right release | 146 | 0.0822 | 0.0616 | 0.0616 | 0.1986 | 0.3151 | 4.410 | 1.642 |
| up release | 82 | 0.0488 | 0.1098 | 0.0854 | 0.1951 | 0.3171 | 5.173 | 1.467 |
| jump release | 94 | 0.0532 | 0.0319 | 0.0745 | 0.2340 | 0.3511 | 5.707 | 1.383 |
| dash release | 84 | 0.1071 | 0.1071 | 0.0952 | 0.2381 | 0.3452 | 6.085 | 1.204 |
| grab release | 55 | 0.0727 | 0.0364 | 0.0364 | 0.1091 | 0.2364 | 5.602 | 1.519 |

Ordered pixels improved macro exact accuracy over the frozen-feature localizer
by only 0.00493, versus the required 0.03. Its paired block-bootstrap 95%
interval was [-0.01970, 0.02925], and its own exact interval was [0.06271,
0.09547], far below the required 0.125 point estimate. Six of 12 estimable
heads improved across only left, up, and jump, versus the required seven heads
and four keys. Onsets were 0.08473/0.08802/0.08124 exact for
ordered/symmetric/frozen representations; releases were
0.07251/0.06873/0.06614. The small gain was not a broad event-type effect.

The attribution control was decisive. Ordered and symmetric pairs were almost
tied in exact accuracy: delta 0.00024, 95% interval [-0.02213, 0.02162]. The
ordered arm also had worse NLL than the symmetric arm (5.292 versus 3.838).
Thus the experiment cannot attribute even its small descriptive gain to frame
order within each pair or the signed difference channel; shared two-frame
appearance, sampling noise, or both are sufficient explanations. The shared
temporal convolution still consumes the sequence of pair embeddings in order,
so this control does not remove all longer-range temporal order.

The ordered model fit the training endpoint strongly (cross-entropy 2.7585 to
0.0559) without a commensurate held-out benefit; the symmetric arm ended at
0.4859. Pooled ordered entropy was 1.463 nats. Its lowest-entropy quartile was
10.07% exact and the other quartiles were 6.94%, 6.97%, and 7.67%, a weak
confidence relationship accompanied by poor NLL. Pooled errors were 43.57%
early and 48.52% late with mean signed error +0.313 frame. That small late
tendency is not a stable per-head action-to-visual latency estimate.

A distinct post-run validator then required the exact five-file run inventory,
frozen config and dependency hashes, MPS device, seed/endpoint/support, cache
and baseline closure, checkpoint schemas and state hashes, and content-bound
report/marker publication. It exactly replayed both checkpoint probability
arrays on MPS and regenerated the frozen score. All 30 recorded checks passed;
the audit file SHA-256 is
`1fb8a8a4124a2d46a9898ca9dbee126facc2de80523f5f3e0219a1c540e51af8`
and its canonical content SHA-256 is
`2a1359c708a6b7c6fdcce6b13ced7868c247b6c8fb3d126c11e611fef44c7424`.

Final decision: **reject the hierarchical cascade and stop after the bounded
pixel diagnostic**. Neither confirmation seeds nor Phase 2 are authorized.
This is evidence that the verified 512-D representation and this fixed
32x32, approximately 0.1M-parameter native-rate adapter do not materially
recover exact timing. It is not proof that timing is intrinsically absent from
the source pixels. The next discriminating experiment should preserve more
spatial detail around action-relevant regions or condition a forward-dynamics
model on candidate actions. That materially larger experiment would benefit
from a remote GPU, but no remote machine was provisioned or contacted here.
The proposed build order, controls, gates, infrastructure, and artifact plan
are in `EXACT_LOCALIZATION_NEXT_STUDY_PLAN.md`.

## Exact commands and artifacts

The production decision paths refuse overwrite. With the validated caches and
retained checkpoint in their bound locations, the two fixed launch commands
are:

```bash
DEVICE=mps MODE=production experiments/run_oracle_window_localization.sh
DEVICE=mps MODE=production experiments/run_oracle_window_differential_followup.sh
```

The machine-readable reports are
`results/idm/oracle_window_localization_s0_decision.json` and
`results/idm/oracle_window_differential_s0_decision.json`; their completion
markers and independent audit receipts sit beside them. Large caches,
checkpoints, prediction arrays, and run directories remain local operational
artifacts and are excluded from Git. The final repository-wide test command was
`uv run --frozen pytest -q`: 45 focused experiment/audit tests and all 715
repository tests passed.
