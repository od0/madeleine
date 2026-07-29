# Oracle-window high-resolution regional localization

Status: **executed 2026-07-28; formal primary gate FAILED
(`reject_study_h_primary_gate`)**. H128-Q improved estimable-head macro exact
localization over the matched H32-Q control by **+5.153 percentage points**
(paired whole-block 95% interval **+2.754 to +7.439**), but its macro NLL was
`3.54580` versus `3.46744` for H32-Q. Because the preregistered gate was
conjunctive, Study H stops at seed zero. H128-Q also failed to distinguish
task-query regional attention from global pooling at 128x128. The result is
therefore: **higher spatial resolution improved exact localization, calibration
failed the primary gate, and regional attribution was not established**. The
improvement is onset-specific: onset macro exact rose 23.94% to 34.05% while
release localization stayed flat (7.22% to 7.42%), so the recovered evidence
concentrates where dash-freeze and jump mechanics make presses visible —
consistent with the effect-locking diagnostics.

The authoritative evidence is
[`oracle_window_highres_s0_decision.json`](oracle_window_highres_s0_decision.json)
and
[`oracle_window_highres_s0_audit.json`](oracle_window_highres_s0_audit.json).
The frozen design and gate are retained below as the preregistered contract.

## Question and prior evidence

This study asks whether spatial detail discarded by the completed 32x32 pixel
adapter remains sufficient in the verified 128x128 masked pixels to localize a
known key onset or release to its exact frame. The earlier conditional-feature
and 32x32 differential study is closed: its best exact score was `0.07862`
versus `0.0625` chance, and its primary paired interval crossed zero. Study H
does not reinterpret or rerun that result.

## Frozen population and support

The study reuses the exact completed oracle identities: 32 native-rate frames
per crop, a 16-frame candidate region, an eight-frame halo on each side, and
one requested key/event identity. Target offsets remained nearly uniform:
each of the 16 val-A offsets had 69--74 examples. Windows never cross source,
continuity, validity, or label boundaries.

| Split | Raw target events | Eligible | Invalid predecessor | Boundary/gap/inactive | Ambiguous target |
|---|---:|---:|---:|---:|---:|
| Train | 9,594 | 4,554 | 2,697 | 2,013 | 330 |
| Val-A | 2,228 | 1,150 | 288 | 528 | 262 |

Val-A contains 1,150 examples in 122 continuity blocks. Twelve heads are
estimable: six keys (`left`, `right`, `up`, `jump`, `dash`, `grab`) across both
onset and release. `down:onset` (11 examples) and `down:release` (10) remain
descriptive only. Session `rec_20260725_015612` yielded no eligible examples,
but the frozen two-session minimum still passed. No overlapping source crossed
the train/validation split.

The model receives only masked RGB crops and the requested key/event identity.
Val-B, B1, `y4n`, the sealed session, action/controller state, filenames, frame
indices, crop positions, offsets, padding cues, and answer-key pixels are
forbidden.

The completed full-resolution cache is 8,779,169,914 bytes and is bound by:

- source commit
  `546933bb010dc48d21592546096eec885a84bad4`;
- decision-config SHA-256
  `7144e68f65acb75a7ae5712330d162f20e16f0fa9bd9440c7f58f67e7962e75f`;
- base-manifest SHA-256
  `212f7ed619deb7171c0c827c634c5d99babbf091696f18bfa1de2cb6c5ff23c7`;
- cache-receipt SHA-256
  `55f9a96cec4c42ba6a5e4713b9cc35cce09b90974fb8a7127520963a75679e98`;
- validated cache-content SHA-256
  `a1c0d38fcac8af5fbfe79ff59b09282914d3d6e4d7d385403659fe4b96b364dc`.

The cache was built in 139.95 seconds with about 5.03 GiB maximum RSS. Source
pixels and the completed oracle artifacts were not modified. A relocation-only
validator amendment preserves the original publication path as provenance
while requiring the same receipt, inventory, byte counts, and payload hashes;
it changed no data, model, training, evaluation, or gate decision.

## Matched arms and frozen training

| Arm | Input | Spatial readout | Role |
|---|---:|---|---|
| H32-Q | 32x32 | Task-query attention over 2x2 maps | Resolution control |
| H128-G | 128x128 | Uniform global mean plus matched task MLP | Nonregional control |
| H128-Q | 128x128 | Task-query attention over 8x8 maps | Primary arm |

All arms use the same 3,211,713-parameter ResNet-18-through-layer-3 model and
the same previous/current/difference/absolute-difference frame-pair channels,
31 aligned pair steps, temporal kernel 16, and 16 candidate logits. Initial
tensors, seed-zero batch order, 20-epoch endpoint, AdamW schedule, effective
batch 32, and inverse-task-count-weighted exact-offset cross-entropy are
matched. CUDA bfloat16 is used with float32 loss. There is no augmentation,
scheduler, early stopping, validation selection, or hyperparameter adaptation;
only the final fixed endpoint is scored.

## Frozen decision gates

H128-Q advances to seeds 1 and 2 only if **every** primary clause passes:

- estimable-head macro exact is at least `0.125` and its 95% lower bound is
  above `0.0625`;
- H128-Q minus H32-Q macro exact is at least `+0.03` and its paired 95% lower
  bound is above zero;
- at least seven estimable heads improve across four keys and both event types;
- macro within-two is no more than `0.01` below H32-Q; and
- macro NLL is lower than H32-Q.

The separate regional-attribution claim requires H128-Q minus H128-G macro
exact of at least `+0.015` with a positive paired lower bound, the same breadth,
within-two noninferiority, and lower NLL. A failed primary gate stops Study H.
Study D requires its own frozen contract before implementation or inference.

## Reproduction paths

- Decision contract: `experiments/configs/oracle_window_highres_regional_v1.json`
- Cache builder: `experiments/prepare_oracle_window_fullres.py`
- Trainer/model: `experiments/oracle_window_highres_regional.py`
- Fixed scorer: `experiments/score_oracle_window_highres_regional.py`
- Independent replay validator:
  `experiments/validate_oracle_window_highres_run.py`
- Launcher: `experiments/run_oracle_window_highres_regional.sh`

The launcher requires the exact source SHA. Smoke output is explicitly marked,
bounded by train/validation limits, and rejected by the production scorer as
scientific evidence.

## Completed seed-zero result

The three CUDA arms reached their fixed 20-epoch endpoints without validation
selection. The following are equal-head macro metrics over the 12 estimable
heads; chance within-one and within-two account for candidate-window edges.

| Arm | Exact | Within one | Within two | NLL | Entropy |
|---|---:|---:|---:|---:|---:|
| Uniform chance | 6.250% | 17.984% | 28.924% | 2.77259 | 2.77259 |
| H32-Q | 15.583% | 25.757% | 35.818% | 3.46744 | 1.99602 |
| H128-G | 20.475% | 31.418% | 41.382% | 3.88217 | 1.33362 |
| H128-Q | 20.736% | 31.518% | 41.692% | 3.54580 | 1.58038 |

### Primary gate: H128-Q versus H32-Q

| Conjunctive clause | Frozen requirement | Observed | Result |
|---|---|---|---|
| Candidate exact | >=12.5%; 95% lower bound >6.25% | 20.736%; 95% interval 18.917--22.632% | Pass |
| Exact improvement | >=+3 pp; paired lower bound >0 | +5.153 pp; 95% interval +2.754--+7.439 pp | Pass |
| Breadth | >=7 heads, >=4 keys, both event types | 7 heads, 5 keys, onset and release | Pass |
| Within-two | No worse than H32-Q by >1 pp | +5.874 pp | Pass |
| NLL | H128-Q < H32-Q | 3.54580 versus 3.46744 | **Fail** |

The exact-accuracy bootstrap used all 122 continuity blocks and 5,000 valid
paired replicates. The sole failed primary clause is NLL, but it is decisive:
the registered status is `reject_study_h_primary_gate`.

### Regional attribution: H128-Q versus H128-G

H128-Q improved exact by only `+0.261` pp, with paired 95% interval
`[-2.018, +2.479]` pp, against the required `+1.5` pp and positive lower
bound. Only five heads across three keys improved, below the seven-head,
four-key breadth requirement. Within-two (`+0.310` pp), polarity breadth, and
NLL (`3.54580 < 3.88217`) passed. The regional-attribution gate nevertheless
failed: high resolution helped the primary comparison, but task-query spatial
attention was not shown to cause that gain.

## Where the exact-frame gain occurred

The equal-head polarity split shows that the gain was concentrated in onsets.

| Event type | H32-Q exact | H128-G exact | H128-Q exact | H128-Q - H32-Q |
|---|---:|---:|---:|---:|
| Onset | 23.943% | 30.372% | 34.048% | +10.105 pp |
| Release | 7.223% | 10.577% | 7.424% | +0.201 pp |

Formal breadth still covered both polarities because three release heads had
positive deltas, but the release macro gain was only `+0.201` pp. Per-head
results make the heterogeneity explicit:

| Requested head | n | H128-Q exact | vs H32-Q | vs H128-G |
|---|---:|---:|---:|---:|
| left onset | 79 | 16.46% | +7.59 pp | +0.00 pp |
| right onset | 149 | 32.89% | +22.15 pp | +12.75 pp |
| up onset | 102 | 2.94% | -8.82 pp | -4.90 pp |
| jump onset | 108 | 66.67% | +35.19 pp | +20.37 pp |
| dash onset | 87 | 80.46% | +5.75 pp | +1.15 pp |
| grab onset | 82 | 4.88% | -1.22 pp | -7.32 pp |
| left release | 61 | 9.84% | +1.64 pp | +0.00 pp |
| right release | 146 | 7.53% | +2.74 pp | +0.68 pp |
| up release | 82 | 9.76% | +6.10 pp | -4.88 pp |
| jump release | 94 | 4.26% | -2.13 pp | -3.19 pp |
| dash release | 84 | 9.52% | -7.14 pp | +1.19 pp |
| grab release | 55 | 3.64% | +0.00 pp | -12.73 pp |

The strongest high-resolution gains were `jump:onset` and `right:onset`; the
model regressed on `up:onset` and did not establish a general release benefit.
No positive claim should be generalized from the aggregate exact improvement
to every key or event type.

## Calibration, entropy, attention, and timing

Every learned arm had worse NLL than uniform chance despite better exact
accuracy. H128-Q was sharply confident on many correct examples but also on
enough wrong examples to fail the reliability gate: pooled mean entropy was
`0.678` when exact versus `1.850` when wrong. Exact accuracy by increasing
entropy quartile was `59.0%`, `11.8%`, `7.3%`, and `9.8%`. Low entropy is thus
informative, but the predicted distribution is not calibrated enough to use
its NLL as reliable uncertainty.

H128-Q spatial-attention entropy was `2.291`, versus the H128-G uniform-grid
entropy `4.159`, corresponding to roughly 9.9 effective cells out of 64.
Attention concentrated spatially, but H128-Q did not beat H128-G, so
concentration is a diagnostic rather than evidence of useful regional causal
selection.

H128-Q made `39.83%` early and `38.17%` late predictions, with mean signed
error `-0.058` frames and mean absolute error `4.399` frames. There is no
meaningful global early/late shift to correct; errors are primarily
head-specific and multimodal rather than a single action-to-visual latency.

## Runtime, memory, and disk

Production used three independent L40S CUDA lanes; the fourth GPU remained
unused. The arms launched once and ran concurrently.

| Arm | Fixed-endpoint wall time | Peak CUDA memory | Peak process RSS |
|---|---:|---:|---:|
| H32-Q | 44.46 min | 0.167 GiB | 5.130 GiB |
| H128-G | 44.27 min | 0.373 GiB | 5.139 GiB |
| H128-Q | 45.16 min | 0.373 GiB | 5.173 GiB |

Launch-to-score took about 45 minutes 20 seconds; scoring and independent
replay audit were complete by 45 minutes 47 seconds. The 13-minute-49-second
smoke projection underestimated concurrent production wall time by about
3.3x because CPU-side preprocessing contended across the three lanes.

The validated cache occupied 8,779,169,914 bytes. The exact 12-file production
run inventory occupied 38,954,568 bytes; the compact control evidence occupied
349,324 bytes and the four top-level decision/audit artifacts 372,397 bytes.
The three retained checkpoints total 38,687,253 bytes.

## Validation and immutable artifacts

The supervisor completed at `2026-07-28T19:55:38Z` after one frozen score and
one independent replay audit. The audit passed all five content checks:

- exact checkpoint tensor hashes;
- exact probability-sidecar replay;
- exact attention-sidecar replay;
- exact fixed-policy report regeneration; and
- a valid content-bound completion marker.

The report, marker, audit, and audit-marker SHA-256 values are respectively:

- `2aef0bcc2f05cd5fb6bb0f0f3c08b8184b42745862bcaafe9739f50c73baf3df`;
- `04e411d43ae4544c81d8913361d2301036d8599648f4d3cd4d4eec0f5c40052c`;
- `9e7e348c3c7d2343ad4fc26f0ea1983effafe3ed3c682fbfc103f6882585bdc4`;
- `48d355319d705ddd08ebb1f27e0a7f66e62792aab5d6045cc424e09521d6d62d`.

An independent Mac regeneration preserved all structure, nonfloating fields,
and the decision; the largest cross-platform floating aggregation difference
was `4.76837158203125e-7`, inside the fixed `1e-6` validation tolerance.

Each final checkpoint was published with only `model.pt`,
`checkpoint-manifest.json`, and `checkpoint_complete.json`. Streamed SHA/byte
readback, exact three-object inventory, marker-last ordering, and an independent
download check all passed:

| Arm | Checkpoint SHA-256 | Immutable durable-store prefix |
|---|---|---|
| H32-Q | `1c9685e2eafa5f5fbc79b4df2a007088a3ad20d7927fa2462ed4756766f1f9e4` | `runs/idm/v1/oracle-window-highres-h32-q-s0/1c9685e2eafa5f5fbc79b4df2a007088a3ad20d7927fa2462ed4756766f1f9e4/` |
| H128-G | `62c0fe4f0d49dfc88469733356a3d5b2e20692945dd8b31be743a5bf094d6eef` | `runs/idm/v1/oracle-window-highres-h128-g-s0/62c0fe4f0d49dfc88469733356a3d5b2e20692945dd8b31be743a5bf094d6eef/` |
| H128-Q | `753c5ddc6edf7b733ad95e2eeea886435947e21011e0eac4dcfe8c8f21303f10` | `runs/idm/v1/oracle-window-highres-h128-q-s0/753c5ddc6edf7b733ad95e2eeea886435947e21011e0eac4dcfe8c8f21303f10/` |

The private working repository retains the three-record checkpoint registry
and independent backup-validation receipt. Checkpoint bytes and storage access
coordinates are not part of the public export.

Before production, all 35 focused checks and the then-current full suite of 865
tests passed under the exact source checkout. The full suite took 275.26
seconds. The checkpoint publisher's three focused tests also passed. The
production replay audit and the independent local regeneration provide the
artifact-level validation described above.

## Decision and next boundary

Study H is **rejected at its primary seed-zero gate**. At this fixed seed, the
result supports the narrow claim that native 128x128 inputs produced more
exact-onset timing signal than the matched 32x32 control. It does not establish
that effect across seeds, nor does it support a calibrated conditional
distribution, a general release-localization improvement, or the registered
regional-attention mechanism.

Seeds 1 and 2 were **not run**, because seed confirmation was gated on all
seed-zero clauses. The full coarse-to-fine cascade was **not implemented or
launched**. Any multi-seed follow-up needs a separately frozen scorer, and any
Study D higher-resolution/action-conditioned experiment needs its own committed
contract and authorization. Neither is implied by this result.

## Preflight history

The exact committed source completed the full disposable CPU path in 98.17
seconds: real cache load, forward/backward for all three arms, final short
batch, checkpoint and sidecar serialization, fixed scorer, and content-bound
smoke marker. Peak process RSS was 1,160,429,568 bytes. The scorer emitted
`smoke_only_no_scientific_decision`, as required.

Two preceding MPS attempts were stopped before artifact publication by the
non-finite-probability guard in H32-Q. Their empty attempt paths are preserved.
This backend discrepancy did not change the CUDA production recipe. The
production machine then passed the frozen 100-update CUDA profiling smoke
before the completed runs above. Compact preflight details are in
[`oracle_window_highres_smoke_preflight.json`](oracle_window_highres_smoke_preflight.json).
