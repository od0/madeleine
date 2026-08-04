# Pseudo-labeling and behavior-cloning program plan

Status: **draft execution plan; no GPU provisioning, fleet acquisition, data
admission, or production training is authorized by this document**. Every
provision, termination, corpus tranche, and training rung still requires
Bryan's explicit approval, and every paid run passes the smoke, projection,
and cap protocol before launch. This plan extends the scope boundary in
[PLAN.md](../../PLAN.md); the IDM study's claims and completion criteria are
unchanged.

## Purpose

The VPT paper's central move was to use a good inverse dynamics model to
pseudo-label a large corpus of unlabeled gameplay video, then train a causal
behavior-cloning model on those labels. This plan is MADELEINE's version of
that move, sized for Celeste and for a single operator. It covers both steps:

1. label a much larger video corpus with a qualified IDM, and
2. train a causal behavior-cloning model (called the BC model below) on the
   pseudo-labels, evaluate it in deterministic rollouts, and release a public
   report and weights.

The chosen ambitions, recorded 2026-07-30: the corpus target is census scale
(2,000–4,000 train-ready hours), the end state is a playable real-time agent
rather than an offline study only, spending is governed per run rather than by
a program ceiling, and the intended release is a public report plus weights.

No current model qualifies as the labeler. The strongest candidate, the
105.7M VPT-small IDM, fails clause 6 of its own preregistered gate (`down`
recall 0; four keys outside the predicted-positive-rate band —
[VPT_SMALL_113M_RESULTS.md](VPT_SMALL_113M_RESULTS.md)). The program is
therefore written so that everything expensive sits behind an entry gate,
while the two wall-clock long poles (the rollout harness and census
acquisition) can start immediately because neither needs a qualified IDM.

Numbers below are tagged **measured** (committed in this repository) or
**estimate** (derived or assumed; must be replaced by a receipt before the
figure becomes load-bearing).

## Relationship to existing documents

- `VPT_FAITHFUL_REPLICATION_PLAN.md` (private working repository):
  Phase 6 defines the promote study this plan operationalizes. Its Phase 3
  utility gate and cost-envelope conventions are reused. Phases 0–5 of that
  plan (the IDM ladder itself) remain a separate track; this plan consumes
  whichever IDM that track qualifies.
- `VPT_TEMPORAL_RATE_ENGINEERING_NOTE.md` (private working repository)
  fixes the inference geometry facts used in the labeling cost model (center
  retention, three-phase interleave for a 20 Hz labeler, one pass for a
  native-60 labeler).
- [TRAINING_DATA_POLICY.md](TRAINING_DATA_POLICY.md) gains the `pseudo_v1`
  admission tier for label-free video.
- [../../specs/pseudo_labels.md](../../specs/pseudo_labels.md) is the label
  artifact and provenance contract.

## Entry gate: when an IDM is good enough to label

The gate has two tiers. Both are composed from gates that already exist;
nothing is invented and nothing is relaxed.

### Tier E1: labeler eligibility (offline prerequisite)

An IDM checkpoint may produce pseudo-labels for the promote study only if all
of the following hold:

1. It passes the full six-clause candidate gate of
   [VPT_SMALL_113M_RESULTS.md](VPT_SMALL_113M_RESULTS.md) on a prospective
   development surface, not a refit of a spent one. Clause 6 (nonzero recall
   for every key; predicted-positive rate within 0.5–2.0 times prevalence per
   key) is not waivable: a labeler that never emits `down` produces a corpus
   in which `down` never occurs, and a BC model trained on that corpus cannot
   learn the key. The promote study would spend money proving that.
2. Macro AP is at least prevalence + 0.10 on the scoring surface, adopted
   from the Phase 3 utility gate of the faithful replication plan.
3. The checkpoint is SHA-registered and rehydratable from durable storage,
   and the inference recipe (window, stride, retained positions, phase
   handling, mask configuration, thresholds) is frozen before the first
   pseudo-label is written.

A calibrated or retrained variant counts as a new declared model and may pass
on its own evidence. The rest of the Phase 3 gate (the monotone native-hours
ladder) remains a gate inside the replication plan for the 0.5B IDM spend; it
is not a prerequisite here, so a mapped-trained model that passes E1 may
enter E2.

### Tier E2: sufficiency (the promote study, which is the experiment)

Sufficiency is defined as passing the Phase 6 promote rule. The candidate's
pseudo-label policy — the fixed small causal pilot policy trained on the
candidate's labels over a fixed labeler-unseen admitted subset — must
materially beat both the shuffled/prevalence-matched control policy and the
mapped-NitroGen-label policy trained on the same video hours, across at least
three seeds, on the predeclared deterministic rollout metrics and on held-out
engine-truth action NLL. Exact margins are frozen in the promote
preregistration before any arm trains. The true-native-label policy is the
upper reference and is reported, not gated, because its support (a fraction
of one hour of corrected own-truth data) is far smaller; the support
disparity is disclosed in every table.

The fixed labeler-unseen admitted subset is preregistered before the current
VPT-small data-scale optimizers start. It is the exact 66-video / 26.9277778 h
NitroGen promote reserve selected only from all-valid videos outside the
92-video unflagged cohort. Selection is metadata-only, stratified by frozen
decoder mode and duration bin, and deterministically tie-broken with salt
`madeleine-vpt-promote-reserve-v1`; reserve-list SHA-256 is
`30cd62dcbeffb9aac217b22a97f6a38554bb3e818481fa9d070662af12fc0a20`.
No current labeler candidate may train on these videos. Pixel and mapped-label
recovery may process them as shared infrastructure, but optimizer staging must
fail closed on any reserve member. E2 later trains the pseudo-label arm and
mapped-NitroGen-label control on these same hours.

The promote study doubles as the final IDM quality test. An E2 failure is a
publishable Phase 6 outcome, not a process failure; total program exposure
before this gate is under about $600.

### Feeders and the `down` contingency

Candidates arrive from the in-flight native-60 arms (a native-60 pass is the
preferred labeler because dense 60 Hz labels need one pass instead of three
phase passes), from the calibration and rare-positive retrain branch, and
from the 482M faithful IDM track. If every candidate keeps failing on clause
6 for `down` (prevalence 0.0137; current AP 0.0102, below chance ordering),
the preregistered contingency ladder is, in order:

1. the rare-positive retrain already preregistered in the calibration branch;
2. a targeted `down`-heavy engine-truth capture (down-dash and crouch-heavy
   rooms) added to the labeler's training data as a declared variant;
3. a six-key diagnostic exemption: run the promote pilot with `down` excluded
   from the action space, labeled as such everywhere; a promoted six-key
   labeler caps agent capability and can never be reported as a seven-key
   result;
4. stop, and record the same promote / collect more clean data / stop
   trichotomy the replication plan ends with.

The gate itself is never lowered.

## Program stages

Stage 0, census acquisition, and the IDM ladder run concurrently. Nothing at
scale is spent before the promote pilot passes.

| Stage | Content | Cost | Wall clock | Gate to next stage |
| --- | --- | --- | --- | --- |
| 0 | Benchmarks, rollout harness, pilot policy, governance freeze, promote study | $250–600 (estimate) | 3–6 weeks, harness-dominated | one IDM passes E1 and its pilot passes E2 |
| 1 | Corpus pipeline tooling and tier decision | operator time only | 1–2 weeks, overlaps stage 0 | tier spec signed; QC machinery proven on tier A |
| 2 | Label at scale, tier A then B then C | $700–2,000 plus about $100/month storage (estimate) | 4–10 weeks, acquisition-dominated | sidecar validation passes on all admitted videos; QC audits clean |
| 3 | BC training ladder, pilot to mid to headline | $1,000–8,500 by headline choice (estimate) | 1–3 weeks GPU wall | monotone data-scaling checkpoints; fresh authorization per rung |
| 4 | Evaluation batteries and the live agent | under $200 plus about 2 weeks build | 2–3 weeks | frozen policies scored once; latency benchmark met |
| 5 | Public release | operator time only | 2–4 days | provenance manifest and license posture recorded |

Program envelope: roughly $2,500–12,000 and three to six calendar months at
solo pace (estimate). The budget is dominated by the harness build, census
acquisition wall time, and review latency, not GPU dollars.

## Step 1: pseudo-label a larger corpus

### What changes relative to wild admission

Pseudo-labels come from the IDM's own view of the gameplay pixels, so the
entire overlay-decoding half of the wild pipeline is unnecessary: no layout
cell decoding, no compositor-offset calibration, no bind-confidence gates.
Everything that protects the pixels and the timeline survives unchanged:
masking, viewport, measured cadence, temporal boundaries, dedup, provenance.
The masking rule applies to the BC model with full force — a policy network
can read an input overlay exactly as an IDM can, so a video whose overlay
cannot be masked is excluded, never passed through unmasked.

Two geometry facts drive the cost model (fixed in the private
`VPT_TEMPORAL_RATE_ENGINEERING_NOTE.md`):
each retained sliding window labels exactly `stride` rows (64 of 128), so
dense 60 Hz labels cost 3,375 windows per source hour whether produced by a
native-60 labeler in one pass or a 20 Hz labeler in three phase passes; a
20 Hz training surface needs only the phase-0 third. Center retention leaves
about one second unlabeled at each end of every contiguous run; uncovered
rows are dropped and per-session coverage is reported, windows are never
extended across gaps.

### Corpus ladder

Census scale is the target; it is reached through tiers A and B because their
measured yields replace the estimates that size tier C.

| Tier | Population | Train-ready yield | Label cost (GPU) | Other cost | Review load | Wall clock |
| --- | --- | --- | --- | --- | --- | --- |
| A: admitted | 161.9664167 h before promote reserve (NitroGen all-valid 148.3222222 h + wild admitted 13.6441944 h); 135.0386389 h remains eligible for the current maximum-data labeler candidate | ~100% minus 1–2% run-edge loss (estimate) | $9–36 (estimate) | decode $2–4 (estimate) | 2–4 h | days once tooling exists |
| B: raw-complete | 628 videos / 613 nominal h, SHA-verified in durable storage (measured) | 240–400 h (estimate: 0.60–0.85 fullscreen × ~0.90 cadence × 0.73–0.85 boundary/activity) | $13–90 (estimate) | decode $6–18 (estimate) | 8–14 h batched | 2–3 weeks |
| C: census | 7,071 candidates / 6,757.5 nominal h; 4,396.8 h not yet probed (measured) | 2,000–4,000 h (estimate) | $105–885 dense 60 Hz, one third at 20 Hz (estimate) | acquisition $400–800 all-in; decode $44–140; storage +6.5 TB ≈ $97/month (estimate) | 15–25 h batched | 4–10 weeks; per-IP YouTube rate limits are the wall |

Tier A is labeled first regardless of ambition: it is the promote-study
substrate, and relabeling videos that already carry mapped or decoded labels
is the cheapest label-agreement audit available (IDM against NitroGen mapped
against overlay-decoded on the same frames; disagreement becomes a QC
statistic, never a training signal). Tier B's four hour-counters (nominal,
decoded, boundary-allowed, train-ready) are reported separately and its
measured yield fractions replace the tier C estimates before census
processing money is spent. Census acquisition starts early in capped tranches
because it is cheap CPU work and the longest pole; census processing and
labeling wait for a promoted labeler and tier B yields.

At census scale, only labels are persisted durably. Derived pixel arrays
(about 10.6 GB per hour at 60 Hz, measured shape) would cost roughly $5,000
per year to store; re-decoding from stored source video at training time
costs on the order of $100 per pass (estimate). Source video itself stays
durable at 1.467 GB per hour (measured).

### Pipeline work items

About one to two weeks of tooling, all forks or generalizations of existing
code:

- a pseudo-corpus shard builder forked from `harvest/build_wild.py` with the
  label-decode path removed, inheriting the mask-before-resize, dilated
  re-mask, and zero-verification discipline of `data/build_dataset.py`;
- a production labeler lifted from the sliding-window inference path of
  `experiments/eval_vpt_small.py`, checkpoint-SHA-bound, writing the sidecar
  of [../../specs/pseudo_labels.md](../../specs/pseudo_labels.md) with a
  completion marker last;
- a standalone cadence gate over fetch-packet PTS evidence, a batch runner
  for the existing per-video viewport classifier, a corpus dedup registry
  (ID plus media SHA-256 plus sampled perceptual hash) enforcing the
  eval-asset denylist inside the builder, and a batch acceptance tool modeled
  on the wild layout acceptance artifacts;
- the `pseudo_v1` admission tier in
  [TRAINING_DATA_POLICY.md](TRAINING_DATA_POLICY.md).

The first blocking action of the whole program is a GPU inference benchmark:
`experiments/benchmark_vpt_small_inference.py` on one H100 with a batch
sweep over both window geometries, committed as a receipt. The only committed
inference number today is CPU-only (0.1986 seq/s, measured). Every labeling
cost above assumes 20–80 seq/s forward-only on one H100 (estimate bracketed
from the measured 34.978 seq/s training forward+backward rate on two H100s)
and must be replaced by the receipt. Cost under $30, half a day.

## Step 2: train the BC model

### Model design

The BC model is a causal variant of the existing VPT-small graph, implemented
as a new module (the frozen IDM graphs are never mutated) that reuses the
spatial stack, projections, clip-consistent augmentation, and optimizer
recipe. Two changes make it causal, and one test proves it:

1. The 5×1×1 temporal convolution stem is centered and therefore leaks two
   future frames; the BC stem pads on the left only, so frame `t` sees
   `t-4..t`. Shifting targets instead is rejected because a two-frame
   decision delay changes the policy's semantics at 60 Hz.
2. Attention blocks run with a causal mask (marginally cheaper than the
   IDM's unmasked attention).
3. A causality test proves no output at position `t` depends on any pixel
   after `t`.

Heads: the primary head is a single 128-way softmax over the joint key state
per frame (2^7 combinations), which models the joint distribution exactly,
captures mutual exclusions, and matches VPT's joint-action convention. The
IDM's seven factored two-class heads run as a control arm at pilot scale. The
policy is observation-only; action-history conditioning is excluded from the
first system because one-frame persistence achieves 98.98% micro accuracy
(measured) and copying the previous action is a documented degenerate
attractor. The graph keeps VPT's convention of no positional encoding; adding
a learned positional embedding is a preregistered ablation, not a silent
change. Offline evaluation retains the causal tail 64 positions of each
window rather than the IDM's center 64.

Three sizes:

- a ~32M pilot (reduced transformer width/depth on the unchanged spatial
  stack) that serves as the frozen Phase 6 instrument — its config, endpoint,
  and seed set are frozen once and reused for every labeler comparison so
  comparisons stay matched;
- the ~105M workhorse, the causal twin of VPT-small;
- a ~0.5B causal fork of the faithful paper-IDM graph, justified only at
  roughly 450 train-ready hours or more.

Rate: the pilot and first mid-rung train at 20 Hz (VPT's own BC rate, three
times cheaper; a 50 ms actuation grid against measured 217–300 ms press
medians is playable but excludes frame-perfect technique). Native-60 is the
declared upgrade once the in-flight native-60 IDM arms report, and is the
eventual target for the playable agent. Labeler rate is paired to BC rate;
rates are never interpolated.

### Training cost matrix

Epochs decline as data grows so that total examples seen stays roughly
constant across scales; cells then differ mainly in unique data, which is the
scientific question. The expensive axes are rate (×3) and model size, not
data hours. Throughput anchors: 34.978 seq/s for the 105M graph on two H100
PCIe (measured), 20.98 seq/s per H100 SXM5 at native-60 (measured), the TPU
v6e flex envelope for the 482M graph (measured qualification throughput,
planned envelope), and the 8×A100 planning rows of the faithful replication
plan. All other cells are estimates until their smoke runs.

| Rung | Model | Data / epochs | Rate | Hardware | Wall | Cost |
| --- | --- | --- | --- | --- | --- | --- |
| Promote pilot (stage 0) | 32M, ~12 runs | 13–25 h / 20 | 20 Hz | 2× H100 PCIe $4.70/h | ~1 h per arm | $60–100 total |
| Mid | 105M | 162 h / 20 | 20 Hz | 2× H100 PCIe | ~29 h | $140–300 |
| Mid, native rate | 105M | 162 h / 20 | 60 Hz | 2× H100 SXM5 $8.38/h | ~73 h | $600–750 |
| H1 | 105M | ~450 h / 8 | 60 Hz | 8× A100 $22.32/h | 40–60 node-h | $900–1,400 |
| H2, capacity probe | 0.5B | 162 h / 20 | 20 Hz | TPU v6e-16 flex | 10–39 h | $216–842 |
| H3 | 0.5B | 162 h / 20 | 60 Hz | 8× A100 | 85–130 node-h | $2,100–2,900 |
| H4 | 0.5B | ~450 h / 8 | 60 Hz | 8× A100 | 109–154 node-h | $2,400–3,400 |
| H5, census headline | 0.5B | 2,000–4,000 h / 2–3 | 60 Hz | 8× A100 | 115–250 node-h | $2,600–5,500 |

Twenty percent contingency is held on every approved rung. Every run passes
the smoke, projection, and cap protocol (the completed VPT-small run's
projection error was 0.4%, measured). Nested data-scaling checkpoints
(25/100/162 hours inside tier A, then 450, then census) sit between rungs: a
rung is funded only if the previous scale step improved without reversal.
The recommended path to the headline is H2 as a cheap capacity probe, then
H3 or H4, then H5, each on a fresh authorization.

Endpoints are fixed in advance and final weights are authoritative; the
lowest held-out pseudo-label-NLL checkpoint is retained as a labeled
diagnostic only. The epoch-2 all-released checkpoint of the completed
VPT-small run is the named precedent for why selection is not trusted.
Engine-truth surfaces and rollout metrics never select checkpoints.

Training data pipeline: the native-60 generation-contract pattern
generalizes directly (manifest-hashed mmap streams, frozen window geometry,
boundary enforcement, deterministic on-node rebuild from SHA-verified source
shards). A BC shard schema adds a mandatory label-kind field carrying the
labeler run ID and checkpoint SHA per stream. Pixel arrays live on node disk
for the duration of a run (about 5.0 TB for 450 hours at 60 Hz, estimate)
and are not stored durably.

## Evaluation and the playable agent

### Rollout harness (stage 0 critical path)

No input-injection or rollout capability exists anywhere in the repository;
the engine-truth mod observes only. The harness is therefore the long pole
and starts first. Design and effort (estimates; the Celeste TAS ecosystem
demonstrates each mechanism is feasible):

1. input injection in the engine-truth mod via a TAS-style virtual-input
   override (3–5 days);
2. a lockstep frame-step bridge — inject action, advance exactly one frame,
   read observation — which removes any real-time requirement from
   evaluation (3–5 days);
3. in-loop observation capture: backbuffer readback, downsample to the
   training resolution, apply the identical static answer-key mask geometry
   used in training (2–4 days);
4. a deterministic scenario loader with fixed game/mod versions, fixed
   spawns, fixed RNG, validated by running identical action sequences twice
   and diffing the per-frame engine-truth log (3–5 days plus 2–3 days
   validation);
5. a scenario battery and metrics runner over a predeclared room and chapter
   list, scoring completion, deaths, room progress, time, and action
   legality, with hash-bound receipts (4–6 days).

Total roughly three to four and a half focused weeks. If the harness slips,
offline-only pilot results are provisional by preregistration; the promote
decision requires rollouts.

### Offline battery

Teacher-forced action NLL and the standard reporting battery against engine
truth (macro and per-key AP beside prevalence, state F1, onset and release F1
at exact and ±2-frame collars, joint exact match, support disclosure), plus
an event-boundary slice restricted to ±2 frames around true transitions —
the one offline surface the persistence baseline cannot win. Every BC table
carries: always-released, one-frame persistence, random-legal rollout,
mapped-label policy, shuffled-label control, and the native-truth upper
reference, with per-key support and labeler identity.

Eval assets: one or two new engine-truth captures are designated BC
development sessions now; one sealed BC test session is captured only after
the BC recipe, thresholds, and evaluation code are frozen. Spent IDM sessions
and batteries are never reused.

### Live agent (stage 4)

A real-time socket bridge between the policy process and the engine-truth
mod at 60 Hz, with incremental KV-cache inference (a rolling cache is simple
here: no positional embeddings, and the stem needs a five-frame pixel
buffer). The per-frame budget is 16.7 ms. An H100-class GPU plausibly runs
the 105M model in 2–8 ms per frame (estimate); no GPU inference benchmark
exists yet, and the local Apple-silicon number is unmeasured. A local
incremental-latency benchmark is the stage 4 entry item; fallbacks are the
32M policy, a 20 Hz actuation grid, or running the policy on a GPU box
beside the game machine. Estimated additional build time one to two weeks.

## Splits, controls, and label hygiene

- Splits are whole-video and whole-session, extended with uploader-level
  separation: one creator never straddles train and validation.
- Dedup runs before admission against every evaluation asset — own captures,
  sealed sessions and batteries, the NitroGen holdout video, wild evaluation
  videos, and the future sealed BC test — by video ID, media SHA-256, and
  sampled content hashing, because the same speedrun is routinely mirrored
  under different IDs and re-encodes.
- Labeler-seen rule: pseudo-labels on a video the labeler trained on are
  partially memorized copies of its training labels, so promote-study arms
  and all scaling comparisons use labeler-unseen video only. Headline corpora
  may include seen video, with seen and unseen hours reported separately per
  labeler.
- Promote-reserve rule: the hash-bound 66-video reserve above is excluded from
  every current data-scale labeler training manifest and may not be swapped,
  shrunk, or reselected after label recovery, rare-key audit, or model output
  inspection. Its 26.9277778 h overshoots the nominal 25 h target by 1.9277778
  h solely because selection is whole-video within frozen strata.
- Pseudo-label hours are inferred supervision. They are reported beside, and
  never summed with, mapped-label and engine-truth hours.
- Per-video automatic QC before admission: predicted-positive rate within a
  preregistered band of corpus prevalence per key, entropy and flicker caps,
  blip-rate audit against the measured NitroGen 18.7% anchor, quarantine
  rather than silent deletion on anomaly.
- Any suspiciously large BC gain triggers a masking and dedup audit before
  any other interpretation, per the standing repository rule.

## Licensing and release

NitroGen labels are CC BY-NC 4.0. The encumbrance is conservatively assumed
to propagate along the whole chain: NitroGen labels trained the IDM, the IDM
writes the pseudo-labels, the pseudo-labels train the BC model. The release
posture is therefore a public report plus weights under a non-commercial
license, with the full provenance chain documented in the model card, the
dataset card, and the third-party-notices lineage. Only derived sidecars and
hashes are ever published, never third-party media; wild and census sources
are listed with provenance statements. A future unencumbered release would
require a labeler trained only on own engine-truth data, which no current
candidate satisfies; that path stays open and is not this program.

## Governance

Preregistered before the promote pilot: the entry gate, promote margins and
seed counts, the arm list and baselines, the ladder rungs and their go/no-go
clauses, the corpus decision rules, and the sealed BC test protocol.
Adaptive with documentation: engineering details, pilot-rung hyperparameters,
admission tooling, acquisition scheduling.

Per-run cost protocol, unchanged from current practice: focused tests, then
a real-data smoke through load, forward, backward, checkpoint, and
evaluation with recorded RAM, VRAM, and throughput; a dollar and wall-clock
projection; a frozen cap; explicit approval per provisioning; non-spot
capacity only; checkpoints at least every 30 minutes published to durable
storage; SHA-verified readback; an atomic completion marker; teardown.
Fleet work additionally requires session cards, teardown triggers, and
CPU-side idle monitoring; the prior wild-fleet episode (about $400–550 with
45 idle pods found in one audit) is the named failure mode this exists to
prevent.

## Risk register

| # | Risk | Detection | Mitigation |
| --- | --- | --- | --- |
| 1 | No IDM ever passes E1 on `down` | calibration, native-60, and 482M candidates all fail clause 6 | contingency ladder above; pre-gate exposure under $600; a failed E2 is the publishable Phase 6 outcome |
| 2 | Harness slips and blocks E2 | milestone dates in PROGRESS.md | harness starts first; lockstep design removes real-time pressure; offline-only results stay provisional by preregistration |
| 3 | Silent corpus-quality tail at tiers B/C | per-video label-distribution audits against corpus priors; sampled spot checks | fail-closed tier gates; fullscreen filter; measured-cadence rule; quarantine, never silent deletion |
| 4 | Evaluation contamination through mirrored or re-encoded videos | content-hash hits; suspiciously strong offline results | mandatory content-level dedup before admission; audit-before-interpretation rule; rollouts as the contamination-proof primary surface |
| 5 | Labeler-seen circularity inflates the promote comparison | seen-flag audit of arm manifests | unseen-only rule for comparison arms; seen/unseen hours reported separately everywhere |
| 6 | Cost blowout at census scale or the headline rung | per-run caps tripped; fleet accounting deltas | smoke-projection-cap on every run; census behind two decision gates; each headline rung needs fresh authorization |
| 7 | Degenerate BC policy (always released or pure persistence) | event-boundary metrics; predicted event rate outside the prevalence band; visible in rollouts immediately | event metrics co-primary; rate band reused as a policy gate; rollout battery is the primary surface |
| 8 | Single-reviewer bottleneck | review queue depth; stages idle awaiting approval | batch-sampled QC; decisions concentrated at stage boundaries; budget envelopes approved per stage so approvals are per provision, not per job |
| 9 | Endpoint-selection trap recurs | selected-versus-final divergence in reports | fixed endpoints authoritative; the VPT-small epoch-2 precedent cited in the preregistration |

## Bryan's action items

Stated plainly, in order:

1. Approve or amend this plan and the four companion document changes
   (spec, policy tier, PLAN.md scope, PROGRESS.md sequence).
2. Approve one H100 rental (under $30) for the GPU inference benchmark when
   convenient; it unblocks every labeling cost figure.
3. Decide when to start the census acquisition tranches (cheap, slow,
   capped; can begin before any IDM qualifies).
4. Expect the promote-study preregistration (margins, seeds, arm list) as
   the next document in this series; it must be frozen before any pilot arm
   trains.
5. Review one sampled QC batch when tier A labeling runs, and sign the
   `pseudo_v1` tier spec before any tier B video is admitted.
