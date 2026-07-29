# VPT-small calibration-to-retrain branch preregistration

Status: **frozen before any new capture, calibration fit, retraining, or new
model evaluation**.

This protocol follows the completed 105.7M VPT-small result in
[`VPT_SMALL_113M_RESULTS.md`](VPT_SMALL_113M_RESULTS.md). That result is
immutable: raw final weights passed five of six candidate clauses on corrected
own-v3 val-A and failed the rare-key coverage/rate clause. A calibrated model
is a new declared variant; it cannot retroactively change the parent result.

The purpose of this branch is first diagnostic and only then promotional:

1. determine prospectively which key failures are probability-positioning
   failures that scalar calibration can repair;
2. if calibration does not recover every key without damaging the aggregate
   guards, test one matched rare-positive-sampling retrain;
3. publish a promoted model only if a prospectively evaluated variant passes
   the complete frozen gate.

No compute launch is authorized by this document. Crossing the retrain trigger
authorizes implementation review and a cost proposal, not provisioning.

## Frozen parent and factual prior

Parent checkpoint:

```text
run: vpt_small_105696398_tier_b_13p45h_s0
epoch: 20 final
sha256: 38303d995e60495fc50bc9f1dfc0cc7f518c5a82dbefaa5da01f030ab3c27c7b
```

The parent model's fixed val-A evidence motivates, but does not score, this
protocol:

| Key | AP | Prevalence | PPR / prevalence | Frozen prediction |
|---|---:|---:|---:|---|
| Dash | 0.3775 | 0.0779 | 0.085x | calibration likely recovers positioning |
| Left | 0.3078 | 0.1058 | 0.237x | calibration likely recovers positioning |
| Up | 0.2202 | 0.1375 | 0.072x | calibration outcome uncertain |
| Down | 0.0102 | 0.0137 | 0.414x | scalar calibration is not expected to recover useful signal |

AP below prevalence is the critical tripwire for `down`: a positive-slope
temperature/bias transform preserves ranking and therefore cannot create
ordering information absent from the raw scores. It may produce nonzero
recall only by accepting additional false positives. The predicted branch
outcome is consequently **partial calibration success followed by the
rare-positive retrain**. A different result lands without reinterpretation.

### Execution footprint

The calibrator fit is cheap, but Phase A is not a zero-compute operation.
Probabilities do not yet exist for either prospective capture. Before C1
inference, rehydrate the exact 1,268,535,851-byte epoch-20 checkpoint from its
content-addressed Cloudflare R2 artifact and verify the streamed SHA-256 above;
never substitute a local checkpoint with an unverified identity. Run one raw
VPT-small inference pass on C1 to create the calibration inputs. After the
calibrator receipt and E1 command are committed, run the corresponding
VPT-small and registered-GRU inference passes on E1. The same rule applies to
E2 only if Phase B is authorized and completed.

These are local-workstation or single-GPU inference jobs, not a new training
pod request. Record hardware, wall time, checkpoint-transfer bytes, prediction
hashes, and peak memory in the phase receipt. The calibration optimizer itself
should take minutes; no runtime estimate for evidence generation is promoted
until the exact accepted capture lengths and inference hardware are recorded.

## Populations and embargoes

Two new engine-truth development captures are required for Phase A:

- **C1 calibration:** fit seven scalar calibrators and nothing else.
- **E1 prospective evaluation:** remain unopened until C1 parameters, hashes,
  and the exact evaluation command are committed.

If Phase B is triggered, a third new engine-truth development capture is
required:

- **E2 retrain evaluation:** recorded only after the retrain recipe, source
  commit, checkpoint-selection rule, and final checkpoint hash are frozen.

Each role is assigned by a marker-last capture receipt before any model
prediction is produced. A rejected capture is recorded with its label-only
integrity reason and may not be reinstated after model metrics are visible.
C1 and E1 must each contain at least 15 input-active minutes, at least 25
positive state runs for every key, verified fail-closed overlay-mask geometry,
no more than 2% capture drops, monotonic engine indices, no inference window
bridging a dropped-frame gap, and clean adjacent-band leak probes. The capture
may continue to the predeclared 30-minute cap to satisfy
the label-only support minimum; if it still fails, the protocol stops for a
new decision rather than relaxing support.

These sessions are prospective development data, not new final-test surfaces.
They never become training examples. The following remain forbidden throughout
this branch:

- val-A fitting or refitting, including use of its stored per-row predictions;
- B1, val-B, the spent untouched session, or any spent/sealed battery session;
- threshold, temperature, bias, checkpoint, or recipe selection on E1 or E2;
- Wild/provisional data;
- oracle thresholds in a primary table.

The existing val-A result may be cited only as the frozen motivation above.

## Phase A: calibration diagnosis

### Frozen transformation

Load the SHA-verified parent final checkpoint once on C1. For each key `k`,
fit exactly two scalar parameters to input-active C1 rows:

```text
z_k = logit(clip(p_k, 1e-6, 1 - 1e-6))
p'_k = sigmoid(a_k * z_k + b_k)
0.05 <= a_k <= 20
-12 <= b_k <= 12
```

`a_k` is constrained positive, so AP and all score ordering remain unchanged.
Fit `(a_k, b_k)` independently per key with deterministic L-BFGS-B, starting
from `(1, 0)`, minimizing ordinary Bernoulli NLL. No class weights, event
weights, prevalence penalty, target decision rate, threshold search, isotonic
fit, or manual parameter adjustment is allowed. The primary decision remains
`p'_k >= 0.5`.

The calibrator receipt records C1 identity and content hashes, parent
checkpoint hash, implementation commit, optimizer/library versions, starting
point, bounds, convergence state, parameters, fit NLL/Brier/ECE, and its own
SHA-256. Parameters and the E1 evaluator command are committed before E1
arrays are opened.

### One-pass E1 evaluation

In one declared pass, score these frozen variants on exact common E1 support:

1. VPT-small parent raw final probabilities;
2. VPT-small parent with the C1 calibrator;
3. 112.95M GRU final;
4. 36.9M GRU final;
5. always released and one-frame persistence.

The two GRU checkpoints are SHA-verified against the existing registry and use
their native inference recipes. Common-support reconstruction must prove
identical labels, masks, source rows, engine indices, and segment boundaries
for all learned models. Fail closed on a missing row, hash mismatch, nonfinite
array, or shape mismatch.

Report raw and calibrated natural NLL, Brier score, equal-mass per-key ECE,
prevalence, macro/per-key AP, precision, recall, state F1, predicted-positive
rate, collar-0 and +/-2-native-frame event F1, micro key-state accuracy, joint
exact match, and the two trivial baselines. No metric is omitted because it
makes calibration look worse.

### Phase-A gate and diagnosis certificate

The calibrated parent is a confirmation candidate only if it passes the same
six clauses as the original plan, recomputed prospectively on E1:

1. macro AP at least 0.010 above the better final GRU;
2. fixed-0.5 state F1 no more than 0.010 below that GRU;
3. +/-2-native-frame event F1 no more than 0.010 below that GRU;
4. at least four of seven per-key AP values above the 112.95M GRU;
5. micro accuracy above always released and joint exact no more than one point
   below it;
6. nonzero recall for every active key and per-key PPR/prevalence in
   `[0.5, 2.0]`.

Calibration adds a seventh validity guard: on E1, aggregate natural NLL and
Brier must each be no worse than the raw parent, and at least one must improve.
This prevents a rate-band pass produced by dishonest probabilities.

The immutable Phase-A decision labels each key:

- **positioning recovered:** calibrated recall is nonzero, the rate ratio is in
  band, and per-key state F1 is not more than 0.010 below raw;
- **representation-limited:** AP is at or below prevalence, or calibration
  fails the key's recall/rate condition;
- **inconclusive:** integrity/support is valid but neither definition applies.

This diagnosis certificate is Phase A's main product. If all seven guards pass,
the branch stops and proposes calibrated-parent confirmation; Phase B does not
run. If integrity fails, the study stops without triggering Phase B. If
integrity passes but any scientific/validity guard fails—expected primarily
for `down`—Phase B becomes eligible.

## Phase B: matched rare-positive-sampling retrain

Phase B preserves the complete parent system except for its deterministic
training sampler:

- identical 105,696,398-parameter architecture and initialization seed;
- identical Tier-B membership, 128x128/20 Hz generation, labels, masks, and
  augmentation;
- identical natural per-key NLL with no positive/event/focal weights;
- identical Adam hyperparameters, linear schedule, global batch 128, 20
  matched exposure cycles, and exactly 2,340 optimizer steps;
- final epoch-20 weights remain the headline; lowest-NLL weights are diagnostic
  only.

The failed-rate key set is frozen now as `dash, down, left, up`. Every global
batch contains exactly:

```text
64 windows from the parent's uniform window sampler
16 windows containing at least one active dash-positive row
16 windows containing at least one active down-positive row
16 windows containing at least one active left-positive row
16 windows containing at least one active up-positive row
```

Positive buckets are constructed from the same mapped Tier-B training labels.
Within each bucket, deterministic epoch/seed permutations cycle independently;
a multi-key window may belong to multiple buckets and every repeated draw is
counted. No inverse-probability correction is applied: the changed exposure is
the single intended intervention. The run receipt reports unique windows,
draws, repeats, bucket passes, per-key exposure, and realized global-batch
quotas. A real-shard smoke must prove exact two-rank quotas and resume identity
before any production launch.

The parent run is the fixed uniform-sampling control. No LR search, extra seed,
architecture change, loss change, extra epoch, threshold fit, or checkpoint
reselection is allowed.

### E2 fit and evaluation

After the retrain final checkpoint is hash-registered, fit a new seven-key
Platt calibrator for it on C1 using the exact Phase-A code and bounds. C1
remains calibration-only. Before opening E2, commit both parent and retrain
calibrator hashes and the evaluator command.

Score in one pass on exact E2 common support:

1. parent raw and parent+C1 calibration;
2. retrain raw and retrain+C1 calibration;
3. both final GRUs;
4. always released and persistence.

Raw retrain minus raw parent isolates the sampler intervention. Calibrated
retrain minus calibrated parent measures the candidate system effect. The
same seven-clause calibrated-candidate gate applies on E2. Promotion requires
every clause; a favorable seed-0 result authorizes a separate replication
proposal, not automatic seeds or sealed evaluation.

## Artifacts and publication cut line

Each phase releases privately and immutably:

- capture acceptance/sealing receipts;
- configs, source commits, commands, environment receipt, and checkpoint
  registry records;
- raw/calibrated prediction sidecars and complete fixed-threshold reports;
- calibration parameters and fit diagnostics;
- exact common-support validation;
- decision JSON containing every clause and every consulted run;
- marker-last Cloudflare R2 publication with streamed hash readback.

The original parent failure, Phase-A diagnosis, and any Phase-B failure remain
in the lab record permanently. Public milestone language waits for a variant
to pass its prospective gate; intermediate evidence is not rounded up to a
model-success claim.

Frozen external one-sentence framing:

> Our best model ranks most actions well but failed the frozen gate on rare-control coverage—most of that looks like an operating-point problem calibration will fix, except `down`, where below-chance AP says the model never learned it, so that key needs a training change. Both follow-ups are preregistered.

If an external collaborator asks during the private-work window, report the
current branch literally—for example, “calibration recovered the positioned
keys as predicted; down needs the frozen retrain”—only if that is what the
prospective evidence actually shows.

## Stop conditions

Stop and report rather than adapt if:

- a capture role, support minimum, mask/leak check, or content hash fails;
- C1 parameters or commands are not frozen before E1/E2 access;
- calibration optimization is nonfinite or hits a bound without a recorded
  convergence diagnosis;
- any forbidden population is accessed;
- common support or checkpoint identity differs;
- Phase B's exact sampler quotas, global batch, schedule, or resume receipt
  differs from this document;
- a costed Phase-B launch lacks Bryan's explicit authorization;
- an R2 immutable copy or streamed readback fails.

This branch is deliberately capable of concluding that calibration fixes
three keys, sampling still does not fix `down`, and no model advances. The
decision tree is frozen to make that mixed result as reportable as a win.
