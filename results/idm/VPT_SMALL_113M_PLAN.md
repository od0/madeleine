# VPT-small: paper-topology IDM at approximately 105M parameters

Status: **completed 2026-07-29; implementation passed, but the six-part
scientific candidate gate did not**. The immutable outcome is in
[`VPT_SMALL_113M_RESULTS.md`](VPT_SMALL_113M_RESULTS.md), and the
machine-checked release decision is in
[`VPT_SMALL_113M_RELEASE_VALIDATION.json`](VPT_SMALL_113M_RELEASE_VALIDATION.json).
This document superseded the earlier proposal that reduced the
experiment to 32 frames, a stock ResNet-18 path, a weighted objective, and one pass. That
proposal was not VPT-small and must not be executed.

The purpose of this experiment is narrow: build the actual VPT IDM data path
and topology at a width that can be trained affordably on one GPU, then compare
it directly with the existing 112.95M GRU on identical evaluation target rows.
“Small” changes only the Transformer width. It does **not** change the VPT
input resolution, temporal extent, early temporal mixing, spatial stack,
number of Transformer blocks, dense prediction scheme, loss, augmentation,
optimizer family, or epoch schedule.

The implementation authority is Appendix D of the
[VPT paper](https://cdn.openai.com/vpt/Paper.pdf). The public
[VPT repository](https://github.com/openai/Video-Pre-Training) is useful as a
code-level cross-check, but its README explicitly calls the released training
code a rough demonstration rather than an exact recreation of the paper.

## Frozen experiment in one table

| Item | Frozen value |
| --- | --- |
| Training corpus | The exact 13.45-hour Tier-B membership used by the existing 112.95M end-to-end GRU |
| Input | 128 consecutive raw RGB frames, 128x128, sampled at 20 Hz |
| Early temporal layer | Noncausal Conv3D, 3 -> 128 channels, kernel 5x1x1, temporal padding 2 |
| Spatial backbone | Appendix-D three-stack ResNet layout, widths 64/128/128, two classic residual blocks per stack |
| Temporal backbone | Four residual, fully unmasked Transformer blocks |
| Small-model width | `d_model=1408`, 11 attention heads of width 128, MLP width 5632 |
| Output | Seven independent two-class softmax heads at every one of 128 positions |
| Training loss | Natural per-key negative log-likelihood on all label-supported positions; no class or event weighting |
| Primary inference | Sliding 128-frame windows, stride 64, retain positions 32..95 |
| Augmentation | VPT's temporally consistent color and affine ranges, sampled once per 128-frame sequence |
| Optimizer | Adam, LR 0.003, weight decay 0.01, linear decay to zero |
| Batch and schedule | Global batch 128 sequences, exactly 20 epochs |
| Initialization | Fan-in weights and zero biases; train from scratch |
| Scientific seed | Seed 0 only |
| Primary comparison | Final VPT-small versus final 112.95M GRU on the same corrected val-A target rows |
| Performance guard | The final 36.9M GRU is also shown because it currently beats the 112.95M GRU |
| Machine | **Completed:** one Massed Compute pod with 2x H100 80GB PCIe, 256GB RAM, and 2.5TB disk; DDP world size 2 |
| Forbidden before decision | B1, val-B, spent untouched sessions, sealed battery sessions, new threshold fitting, extra seeds |

There is no 32-frame fallback, reduced-resolution fallback, weighted-loss
fallback, GRU-recipe fallback, or truncated “one-epoch VPT” fallback in this
study. If the frozen model or 20-epoch run does not fit its approved compute
cap, the run stops for a decision instead of silently becoming a different
experiment.

## Exact model

### Input and early temporal mixing

The model accepts `uint8[B, 128, 128, 128, 3]`, converts it to floating point,
and divides by 255.0. All 128 frames are raw spatial inputs; no ResNet feature
cache or post-pooling temporal layer is permitted.

The first learned operation is a 3-D convolution with:

```text
in_channels=3
out_channels=128
kernel_size=(5, 1, 1)
padding=(2, 0, 0)
stride=(1, 1, 1)
```

Therefore output position `t` can depend on input frames `t-2..t+2`, including
future frames. Temporal padding preserves all 128 output positions. A unit
test must prove this receptive field and prove that the layer does not access
positions outside it.

### Appendix-D spatial stack

After the Conv3D, every time position passes through the same framewise
ResNet weights. The three stacks have widths `64, 128, 128`. Each stack is, in
order:

1. a 3x3 convolution with one-pixel padding and the stack's output width;
2. a 3x3 max pool with stride 2 and padding 1;
3. two classic two-convolution ResNet blocks at that width.

This is the layout the paper calls “ResNet 62”; `62` is the citation to He et
al., not a request to substitute a stock 62-layer torchvision model. No
ResNet-18, ImageNet initialization, global average pooling, or frozen spatial
features are allowed.

Appendix D contains an internal arithmetic inconsistency: three halving pools
from 128x128 at final width 128 produce `128*16*16 = 32,768` activations, while
the prose says the flattened vector is 131,072. This plan freezes the explicit
layer sequence as authoritative: all three stated stacks pool, and the
flattened size is computed from the actual tensor as 32,768. The architecture
receipt must call out this signed paper discrepancy; it must not add an
unreported resize or omit a pool merely to force the prose's number.

The flattened vector is processed by shared framewise dense layers:

```text
32,768 -> 256 -> 1,408
```

### Four unmasked residual Transformer blocks

The sequence remains length 128. It passes through four blocks, each with:

- pre-normalized, fully bidirectional self-attention;
- 11 heads, each with dimension 128 (`11*128 = 1,408`);
- one residual connection around attention;
- a framewise `1,408 -> 5,632 -> 1,408` MLP;
- one residual connection around the two-layer MLP as a pair;
- no causal mask, Transformer-XL memory, recurrence, GRU, or temporal pooling.

The raw normalized pixels feed the initial Conv3D directly, consistent with
the paper calling it the first layer and with the released VPT model code. For
subsequent convolutional tensors, `GroupNorm(1, C)` supplies the paper's
LayerNorm-equivalent pre-normalization; framewise dense transforms use
`LayerNorm(D)`. ReLU follows feature convolutions and feature dense layers, but
not the final action logits. Fan-in initialization and zero biases are used.
Dropout, stochastic depth, gradient clipping, warm-up, and pretrained weights
are disabled because the paper does not specify them. The paper's broader
sentence saying every convolution is pre-normalized conflicts with its
first-layer description and released Conv3D path; that signed interpretation
is recorded before the real-data smoke and is not chosen from validation
results.

### Dense action heads

The final tensor has shape `[B, 128, 1,408]`. Seven independent linear heads
produce `[B, 128, 7, 2]` logits for the Celeste keys. A two-class softmax is
used for each key, matching VPT's factored key heads. There is no latch,
transition head, structured decoder, threshold layer, or calibration layer.

The first parameter-count test instantiates the complete graph and records
every module's trainable parameter count. The implemented graph contains
**105,696,398** trainable parameters. The hard acceptance range is 70M through
120M; the exact measured count,
not the nickname, goes in the run ID and report. If the implementation falls
outside that range, training stops. Width may not be changed after any model
metric is seen.

## Data build: the same Tier-B cohort, at VPT rate

The existing end-to-end Tier-B shards are already canonical `uint8`
`[N,128,128,3]` RGB. The `32nc` token in the historical run name means a
32-frame noncausal window, **not** 32x32 pixels. Derive the 20 Hz generation
directly from those hash-recorded 128px rows, using the **exact** training
membership, mapped label rows, active masks, and exclusion decisions recorded
for the 112.95M Tier-B run. Do not decode source video again, approximate
membership from filenames, or rebuild labels.

The new generation is:

```text
r2:<bucket>/shards/vpt-small-tier-b-128px-20hz-v1/
```

For each declared segment:

1. Verify the source NPZ SHA-256 against the 112.95M run record and require
   `uint8 [N,128,128,3]` pixels plus aligned keys, engine indices, and masks.
2. Anchor sampling to the existing canonical 60 Hz aligned row grid and select
   phase-zero rows `0, 3, 6, ...` within each segment. Preserve the frozen
   source-frame/resample mapping for native and resampled videos.
3. Bind every output pixel row to its source video hash, decoded-frame index,
   canonical label-row index, timestamp, segment ID, seven labels, and active
   masks.
4. Form full 128-frame windows with stride 64 within a segment. Never cross a
   segment, session, room, or source-video boundary.
5. Record short fragments that cannot form a full window as excluded support;
   do not pad them into training examples.

The manifest must report videos, segments, unique frames, windows, mapped
hours, native/resampled counts, per-key prevalence, active support, and all
exclusions. It must prove that train and corrected val-A membership are
disjoint. Deep validation checks exact counts and hashes, finite arrays, RGB
range, time monotonicity, 20 Hz spacing, label/pixel alignment, masks, and
boundary safety. A content-bound completion marker is published last.

This remains a mapped-label experiment. Tier-B's labels are noisy; 20 epochs
repeat those labels rather than turning them into engine truth. The claim is
about the missing VPT architecture and recipe on the existing comparison
corpus, not about clean-label scaling.

## Training recipe

One training item is a 128-frame sequence. Sequence order is deterministically
shuffled per epoch from the frozen seed. The sampler pads each epoch to a
multiple of global batch 128 by wrapping the beginning of that epoch's
permutation; it records the at-most-127 repeated examples separately so every
optimizer update has the paper's global batch size.

The logits are trained at all 128 positions, as in the paper. For each key,
ordinary two-class negative log-likelihood is averaged over its label-supported
batch/time entries; the seven per-key losses are summed. Positive and negative
examples retain their natural frequency. There is:

- no positive-class weighting;
- no capped class balance;
- no transition or event multiplier;
- no focal loss;
- no center-only training loss;
- no fitted threshold in the loss.

The optimizer is `torch.optim.Adam` with the following frozen values:

```text
learning_rate = 0.003
betas = (0.9, 0.999)       # declared PyTorch default; Appendix D is silent
epsilon = 1e-8             # declared PyTorch default; Appendix D is silent
weight_decay = 0.01        # coupled Adam L2, not AdamW
schedule = linear to zero over all 20 epochs
global_batch = 128 sequences
epochs = 20
```

Every parameter is subject to the same declared weight decay. There is no
warm-up or post-hoc schedule extension. The total optimizer-step endpoint is
derived once from the validated window count, padded sampler, batch 128, and
20 epochs, then committed before the production launch.

Training uses BF16, activation checkpointing, and gradient accumulation only
as numerical/throughput mechanisms. A hardware smoke chooses the largest
microbatch in the fixed search order `4, 2, 1` that stays below the VRAM safety
limit; accumulation is then set so the effective batch remains exactly 128.
The chosen microbatch, accumulation count, math backend, peak VRAM, and a
single-GPU numerical receipt are frozen before production. OOM is not
permission to change frames, pixels, width, blocks, loss, or epochs.

Checkpoints are written at every epoch boundary with model, Adam state,
scheduler, sampler epoch, RNG states, config, source commit, data-manifest
hash, and architecture receipt. Final weights are the headline. The epoch with
lowest corrected-val-A natural NLL is retained as a clearly labeled selected
diagnostic; selection never replaces or hides the final result, and all 20
validation consultations are reported.

### VPT augmentation, exactly once per sequence

Each transform draw is sampled once and applied identically to all 128 frames:

| Transform | Frozen range |
| --- | --- |
| Hue factor | `[-0.2, 0.2]` |
| Saturation | `[0.8, 1.2]` |
| Brightness | `[0.8, 1.2]` |
| Contrast | `[0.8, 1.2]` |
| Rotation | `[-2, 2]` degrees |
| Scale | `[0.98, 1.02]` |
| Shear | `[-2, 2]` degrees |
| Translation x/y | `[-2, 2]` pixels independently |

The implementation freezes transform order, interpolation, fill, library
version, and sampled parameters in its test receipt. A test must detect the
common error of sampling different affine/color parameters per frame.

## Direct comparison with the 112.95M GRU

The VPT-small recipe is intentionally not hyperparameter-matched to the GRU.
The scientific question is whether the affordable VPT system is better, not
whether a Transformer wins after being forced into a 32-frame GRU recipe.
Consequently this is a **system-level comparison** changing architecture,
resolution, context, objective, optimizer, and exposure together. If it wins,
later controls can isolate which change mattered.

### Common corrected val-A support

Build a 20 Hz derived view directly from the corrected own-v3 val-A 128x128
shard and its existing clean manifest. Do not use own-v2 or re-decode the
capture. The VPT model is
evaluated on three deterministic 20 Hz phase grids:

```text
phase 0: native rows 0, 3, 6, ...
phase 1: native rows 1, 4, 7, ...
phase 2: native rows 2, 5, 8, ...
```

Within each phase, slide by 64 sampled frames and retain predictions only for
positions 32..95. Interleave the three phase outputs without averaging so each
native 60 Hz target row receives exactly one VPT prediction. Boundary rows
without a center-supported VPT prediction are excluded from **all** compared
models by one frozen common-support mask.

Rescore the final and selected 112.95M GRU checkpoints, plus the final 36.9M
performance guard, through their native 128x128/60 Hz input paths, then select
their predictions on that exact common-support mask. This gives every model
identical target row IDs, labels, masks, and segment boundaries while allowing
each model to use its intended input recipe.

A secondary paper-native table uses phase 0 only at 20 Hz. Its `+/-2` event
collar is explicitly labeled `+/-2 sampled frames = +/-100 ms`; it is never
mixed with the repository's historical `+/-2 native frames` metric.

### Metrics and thresholds

The primary table uses fixed 0.5 probabilities for every model and reports:

- natural NLL, Brier score, and equal-mass per-key ECE;
- prevalence, macro AP, and all seven per-key AP values;
- precision, recall, state F1, and predicted-positive rate;
- segment-bounded collar-0 and +/-2-native-frame onset/release event F1;
- per-key micro key-state accuracy and seven-key joint exact match;
- always-released, one-frame persistence, prevalence, and shuffled-event luck
  anchors on the exact common support;
- final/selected identity, SHA-256, parameter count, training exposure,
  throughput, peak memory, and cost.

No per-key threshold is fitted for the primary decision. Same-surface oracle
thresholds may be emitted only in a separately labeled diagnostic appendix
after the fixed table is immutable. B1, val-B, the spent untouched session,
and sealed battery sessions remain unopened by this study.

Historical 113M numbers are context, not the direct comparison, because they
used a different data generation/support. For orientation, the original
selected 112.95M run reported 0.2318 macro AP, 0.2194 state F1, 0.0810 exact
event F1, and 0.0919 +/-2 event F1 on its original val-A surface. The direct
table is populated only by the new common-support rescore.

## Preregistered interpretation

The run is an implementation success only if the paper-to-code receipt passes,
all 20 epochs complete, the final checkpoint is hash-registered, and every
common-support sidecar validates.

VPT-small is a scientific candidate for confirmation only if its **final**
weights satisfy all of the following on the common corrected val-A support:

1. macro AP is at least 0.010 above the better final GRU;
2. fixed-0.5 state F1 is no more than 0.010 below that GRU;
3. +/-2-native-frame event F1 is no more than 0.010 below that GRU;
4. at least four of seven per-key AP values improve over the 112.95M GRU;
5. micro key-state accuracy beats always released and joint exact match is no
   more than one percentage point below it;
6. no active key has zero recall, and each predicted-positive rate lies within
   `[0.5x, 2.0x]` of its truth prevalence.

The better final GRU is the denominator for the aggregate guards; the 112.95M
GRU remains the named capacity comparison. The complete result lands even if
the candidate rule fails. If loss is still improving at epoch 20, the report
may diagnose likely undertraining, but the run is not extended post hoc.

Crossing the gate authorizes a proposal, not automatic spend: seeds 1 and 2,
fresh untouched evaluation, larger mapped corpora, and any clean-data run each
require a separately frozen protocol and Bryan's approval.

## Implementation and test deliverables

Add a separate path rather than mutating the existing GRU trainer:

```text
badeline/vpt_small.py
experiments/build_vpt_small_128px_20hz.py
experiments/validate_vpt_small_data.py
experiments/train_vpt_small.py
experiments/eval_vpt_small.py
experiments/run_vpt_small.sh
experiments/configs/vpt_small_105m_128x128_128f_20hz.json
experiments/configs/vpt_small_105m_decision.json
tests/test_vpt_small_architecture.py
tests/test_vpt_small_data.py
tests/test_vpt_small_training.py
tests/test_vpt_small_evaluation.py
```

Required automated evidence:

- exact shapes and parameter inventory for every module;
- Conv3D temporal receptive field `t-2..t+2`;
- three ResNet stacks and six residual blocks with their required max pools
  and no global-average-pooling substitute;
- four bidirectional Transformer blocks with no causal mask;
- `[B,128,7,2]` logits and nonzero loss gradients at edge and center positions;
- natural NLL equivalence to a hand-computed seven-head example;
- no class/event weights in the optimizer graph or config;
- identical augmentation parameters across all frames and different draws
  across independently sampled sequences;
- exact 20-epoch step arithmetic, LR start/end, and resume equivalence;
- center-64 sliding-window reconstruction and three-phase 60 Hz interleaving;
- segment-bounded event matching and exact common target-row identity;
- save/reload logits within the declared BF16/FP32 tolerance;
- content-bound checkpoint and R2 completion-marker validation.

Before production: run unit tests, a synthetic forward/backward smoke, a
ten-step real-shard correctness smoke, and a 200-step throughput/resume smoke.
Smoke outputs live under disposable IDs and are never reported as model
results. Config, decision contract, source commit, data manifest, expected
steps, and fixed decision fields are committed before the full run launches.

## Prime Intellect request and hard cost gate

Bryan authorized the two-H100 Massed Compute option. Request exactly one pod:

```text
provider:   Massed Compute through Prime Intellect
GPU:        2x NVIDIA H100 80GB PCIe
host RAM:   256GB
local disk: 2.5TB
DDP:        one process per GPU, world size 2
```

The selection-time live inventory showed this US two-H100 shape at
`$4.70/hour` total, with 40 vCPUs. Availability, price, non-spot status, RAM,
and disk must be queried again immediately before provisioning; a reused offer
ID is not trusted. No different GPU class, provider, second pod, or spot
substitution is authorized. Global batch remains exactly 128, partitioned as
64 sequences per rank before microbatching and accumulation. A focused test
must show that the two-rank averaged natural-NLL gradient matches the
single-process global-batch gradient within declared tolerance.

The 200-step real throughput smoke produces the production estimate. The run
may launch only if all 20 epochs project to no more than 24 node-hours and
`$105`, with a hard lifecycle ceiling of 28 node-hours and `$125` including
data build, evaluation, and R2 publication. If the estimate exceeds the launch
gate, stop and return the measurement; do not reduce epochs or model fidelity.

Bryan's instruction, “Please go ahead with Two Massed H100s,” authorizes
provisioning this exact selected shape and terminating it after the workload.
Data build, smoke, training, evaluation, and artifact publication are one
continuous owned workload. The node is terminated after
independent R2 readback or a stop condition; it is never retained merely for
monitoring.

## Cloudflare R2 and artifact contract

Cloudflare R2 is durable truth and `/ephemeral` is a replaceable cache. Use:

```text
r2:<bucket>/shards/vpt-small-tier-b-128px-20hz-v1/
r2:<bucket>/contracts/idm/vpt-small-tier-b-v1/
r2:<bucket>/runs/idm/v2/vpt-small-tier-b-v1/<run-id>/
r2:<bucket>/results/idm/v2/vpt-small-tier-b-v1/<run-id>/
r2:<bucket>/provenance/idm/vpt-small-tier-b-v1/
```

Follow the Prime + R2 execution handoff and machine notes, which live in the
private working repository: use the configured `r2:`
remote without reading credentials; copy only exact allowlisted prefixes;
validate existing caches before transfer; never trust multipart ETags as
SHA-256; write through staging; publish immutable artifacts; stream-read or
download-check checkpoints; and upload a nonempty, content-bound completion
marker last.

At creation time, every final/selected checkpoint is entered in the tracked
checkpoint registry with SHA-256, bytes, architecture/config hash, data
manifest hash, source commit, seed, epoch, and R2 object path. A checkpoint
that exists only on the Prime disk is not a completed result.

## Stop conditions

Stop and report rather than adapt if:

- the exact historical Tier-B membership or source-to-label mapping cannot be
  recovered and hash-bound;
- 128x128 pixels, 20 Hz rows, labels, masks, or boundaries fail validation;
- the graph violates any frozen architecture item or falls outside 70–120M;
- the global batch 128 or exact 20-epoch endpoint cannot be maintained;
- training produces nonfinite loss/gradients and the cause is not an
  implementation or infrastructure defect;
- the 200-step estimate exceeds the approved launch gate;
- final/selected sidecars do not share identical, finite common support;
- any forbidden evaluation population is accessed;
- R2 checkpoint or marker readback fails;
- the node would otherwise idle without an authorized workload.

## Explicitly deferred

- the approximately 500M VPT configuration and any eight-GPU run;
- 25/100/250-hour clean-data scaling;
- additional seeds;
- a 32-frame or lower-resolution “matched recipe” Transformer;
- stem, objective, decoder, timing-target, or optimizer ablations;
- Wild/provisional data and blend arms;
- B1 or any untouched/battery evaluation;
- downstream behavioral-cloning utility tests.

Those remain possible follow-ups. None is required to answer this first,
affordable question: does a genuine VPT data path and topology, reduced only
in Transformer width, beat our similarly sized GRU?

## Related evidence

- Existing 112.95M config:
  [`takeover_pixels_113m_e2e_aug_32frame_noncausal.json`](../../experiments/configs/takeover_pixels_113m_e2e_aug_32frame_noncausal.json)
- Current consolidated results: [`SUMMARY.md`](SUMMARY.md)
- Full-scale VPT program, explicitly not authorized here: the faithful
  replication plan in the private working repository.
- Private Prime + R2 execution handoff: in the private working repository.
