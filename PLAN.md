# MADELEINE research roadmap

This is the durable research plan. It describes the questions, experimental
comparisons, and evidence required for a result. Current execution state lives
in [PROGRESS.md](PROGRESS.md); measured findings live in
[results/idm/SUMMARY.md](results/idm/SUMMARY.md) and
[the technical report](report/README.md).

## Thesis

An inverse dynamics model should recover actions from visual evidence rather
than copy action persistence, read a visible controller overlay, or exploit a
single player's route. MADELEINE uses *Celeste* to separate four effects:

1. whether future frames make an action more identifiable;
2. whether visual capacity or temporal capacity limits the model;
3. how clean engine truth and noisy mapped internet labels complement each
   other;
4. how alignment, dropped frames, binding uncertainty, and label timing affect
   state recognition and exact action transitions.

The central working hypothesis is now narrower than when the project began:
more data and better visual learning improve recognition of held action state,
but exact onset and release timing require an objective and checkpoint policy
that explicitly reward transitions.

## Data funnel

The project reports each stage separately. These numbers are not competing
estimates of one corpus; they are successive admission gates.

| Stage | Videos | Hours | Meaning |
|---|---:|---:|---|
| NitroGen Celeste metadata | 411 | 684 nominal label-hours | All labeled chunks in the extracted slice |
| Source-available fetch candidates | 245 of 411 censused | — | 231 Twitch and 14 YouTube sources entered recovery |
| Historical successful downloads | 244 | 213.0889 label-hours | Source video was recovered at least once |
| Historically eligible 60-Hz chunks | 221 | 192.9 label-hours | At least some chunks met the per-chunk rate rule |
| Durably preserved media | 232 | 164.4222 label-hours | Current object-store and training-cache video population |
| Strict whole-video feature-eligible corpus | 211 | 150.9167 label-hours | Durably present, metadata-valid membership on the nominal 60-Hz label grid |
| Higher-confidence binding cohort | 93 | 106.00 label-hours | Strict corpus without the broad binding fallback |

The repository does not distribute these videos or the NitroGen label corpus.
The exact census and known caveats are in
[data/dataset_card.md](data/dataset_card.md).

Twelve historical downloads are absent from the durable archive. Ten belonged
to the historical eligible pool, so 221 − 10 = 211 and 192.9 − 41.9833 =
150.9167 hours. The evidence records the loss but not its cause.

Engine-truth captures are tracked by session rather than by a single aggregate
hour count. Their roles are frozen before inference:

- training/fine-tuning sessions;
- development sessions used for checkpoint or threshold selection;
- drop-diagnostic sessions unsuitable for long-context reporting;
- newly captured untouched sessions reserved for final evaluation.

## Workstreams

### 1. Engine-truth acquisition

The `granny` instrumentation, whose Everest assembly retains the compatibility
name `InputTruth`, logs the seven controls and player state once per engine frame. A
rendered frame-index strip lets the video be aligned without relying on two
machine clocks. The acquisition result is complete only when:

- the mod log, video, alignment table, and manifest agree;
- every mask rectangle is declared in the correct coordinate space;
- duplicate and missing frames are counted;
- action labels remain binary and use the canonical key order;
- the session role is recorded in `data/sessions/INDEX.md`.

This stream supports clean evaluation, temporal-ambiguity analysis, and
controlled degradation experiments.

Known defect (found 2026-07-26): the declared input-overlay mask rectangle
undershoots the rendered overlay cells, so a readable answer-key sliver
survives in own-data shards built before the fix (a single leaked pixel
feature separates `left` at AUC 1.000 in a training shard). Own-data
generation is affected until the correction lands: no transferable benefit
was observed on held-out sessions, but this does not rule out training
distortion; corrected manifest geometry, shard rebuilds, an outside-rectangle
leak scan, and own-data reruns are queued. Fresh engine-truth captures and
any new own-data training wait on this fix. Current status is in the
[PROGRESS.md known issues](PROGRESS.md).

### 2. Mapped NitroGen supervision

The NitroGen pipeline recovers source video, clamps and rescales controller
geometry, masks the controller widget, maps gamepad controls to the seven-key
schema, and splits at missing chunks. Quality dimensions remain independent:

- source availability;
- native grid and decoded cadence;
- controller binding confidence;
- joystick-axis sign confidence;
- viewport layout;
- chunk continuity;
- repeated or imputed images.

The default full-corpus experiment retains every strict metadata-valid video.
A matched higher-confidence cohort measures whether the additional, noisier
44.92 hours help or hurt.

### 3. Wild input-overlay supervision

The Wild20 path recovers keyboard labels from visible input displays. It is a
rolling, per-video evidence pipeline:

1. acquire source video and exact timestamps;
2. propose layout and gameplay boundaries;
3. generate source-bound review packets;
4. accept geometry and boundaries through immutable human-review artifacts;
5. decode actions and measure visual/input offset;
6. accept the offset and rerun the final decode;
7. build masked shards and publish a content-addressed derived record.

AI-only proposals may be used for diagnostics but never for admission. See
[harvest/WILD20.md](harvest/WILD20.md).

### 4. Modeling

The model family is deliberately incremental:

- 2-frame inverse-dynamics baseline;
- past-only and centered temporal controls;
- frozen ResNet-18 feature models;
- larger recurrent temporal heads;
- end-to-end visual fine-tuning with temporally consistent augmentation;
- long-context models spanning 128 samples at stride three;
- transition-aware objectives and fixed-endpoint selection.

The current full-corpus reference recipe is a 25.7M-parameter temporal model
over frozen features with 128 samples, stride three, class-balanced BCE, and
8× transition weighting. The larger end-to-end path tests whether visual
learning changes the state/timing tradeoff before spending on substantially
larger models.

### 5. Verification and reporting

`goldenberry` and the dataset validators answer whether a record is internally
consistent. They do not prove that a human played the video. Every public
result must map to:

- exact data membership and split lists;
- a committed configuration and seed;
- selected and/or fixed endpoint semantics;
- a checkpoint hash;
- an evaluation report and prediction sidecar;
- the label source and threshold-selection surface;
- support counts and continuity coverage.

## Experimental questions

### Q1 — Does future context help?

Compare past-only and centered windows with equal history and capacity. Also
compare a longer past-only window so “more context” is not conflated with
“future context.” Report on identical target support.

### Q2 — What does the model learn from noisy internet labels?

Compare engine-truth-only training with curated mapped NitroGen pretraining and
with larger mapped-label pools. Evaluate transfer on engine truth, while using
an unseen NitroGen video only as a mapped-label architecture diagnostic.

### Q3 — Is capacity spent better on vision or temporal modeling?

Compare:

- a small frozen-feature head;
- a larger frozen-feature temporal model;
- an end-to-end visual model at a similar total scale;
- a larger end-to-end model only after a real-data smoke establishes memory and
  throughput.

The comparison must track trainable parameters, total parameters, input
context, target support, optimizer steps, and checkpoint rule.

### Q4 — Why does exact timing lag state recognition?

Measure onset and release separately, inspect prediction traces around events,
and compare plain state loss with transition-aware sampling or auxiliary event
heads. Checkpoint selection should optimize the metric family used in the
claim, or use a fixed endpoint chosen before training.

### Q5 — What does label quality cost?

Use engine truth to price:

- controlled ±1/±2/±4-frame label jitter;
- dropped and repeated video frames;
- spatial compression and lower resolution;
- mapped-binding uncertainty;
- decoded-overlay offset error.

These experiments explain why a model can improve AP while failing at event
timing.

## Evaluation protocol

Every comparison reports:

- macro and per-key average precision;
- label prevalence as the chance AP baseline;
- state F1 with threshold provenance;
- onset/release F1 at exact and ±2-frame collars;
- number of frames, sessions, videos, and continuity runs evaluated;
- mean and paired deltas across seeds when replication exists.

Oracle thresholds chosen on the reported evaluation set are labeled as
diagnostic upper bounds. Development thresholds are frozen before untouched
testing. Persistence and shuffled-event baselines accompany engine-truth
results where applicable.

## Evidence already established

- Clock-free frame-index alignment works end to end on local captures.
- The overlay decoder can recover clean engine-truth actions with near-perfect
  accuracy when its geometry is known.
- Small engine-truth-only datasets fit training data but generalize poorly
  across sessions.
- Curated mapped NitroGen pretraining improves local development AP and exact
  transition F1 across three paired seeds.
- End-to-end visual learning substantially improves state AP but not exact
  transition timing.
- A long-context model trained only on mapped NitroGen labels beats prevalence
  on an unseen NitroGen video, while event timing remains weak.
- The strict full corpus contains 211 videos and 150.9167 label-hours; binding
  confidence is a larger known quality issue than sparse chunk gaps.
- Wild input-overlay admission now uses source-bound, fail-closed review
  artifacts rather than editable flags.

## Near-term sequence

This section is roadmap-level ordering; live execution state is tracked in
[PROGRESS.md](PROGRESS.md). Status markers below are as of 2026-07-26.

1. Complete and validate the all-video feature build. Done: all 211 videos
   built and deep-validated (1,554 shards, 32,598,000 train-ready frames,
   zero failures).
2. Finish the larger end-to-end diagnostic already in flight. Done: the
   112.95M end-to-end run completed with archived sidecars and checkpoint
   hashes, and the matched 36.9M/112.95M development evaluation on B1 (a
   development-only engine-truth capture) is archived in `results/idm/`.
3. Train matched all-valid and higher-confidence full-corpus models against
   the same unseen-video holdout. Active.
4. Score the completed full-corpus checkpoints on B1 as a development
   diagnostic. Queued.
5. Capture and lock a fresh engine-truth session for final transfer evidence.
   Queued; blocked on the 2026-07-26 mask-coverage fix above.
6. Test a transition-aligned objective and checkpoint rule. Queued.
7. Consolidate figures, qualitative prediction traces, and artifact links into
   a technical report. Queued.

## Scope boundaries

The repository began as an IDM and data-quality study, and those claims are
complete on their own terms: that work is considered research-complete when
the frozen engine-truth test has been evaluated, all reported results have
reproducible artifact links, known selection contamination is explicit, and
the central state-versus-timing finding survives that untouched test. The
Wild20 corpus is useful only to the extent that its review and provenance
gates remain auditable; raw hours are not counted as training yield.

As of 2026-07-30 the repository additionally carries a second, separately
gated claims section: the pseudo-labeling and behavior-cloning program of
[results/idm/PSEUDO_LABEL_BC_PLAN.md](results/idm/PSEUDO_LABEL_BC_PLAN.md).
Its research question is whether an IDM that passes its own quality gates can
label a much larger Celeste video corpus well enough that a causal
behavior-cloning model trained on those labels becomes a more capable
behavioral prior than mapped-label and control-label policies, measured in
deterministic rollouts and on held-out engine-truth action agreement. That
program has its own entry gate, promote rule, and completion criteria; its
outcomes do not amend the IDM study's conclusions, and reinforcement
learning remains outside the repository's claims.

Adopting the Wild7 public-video holdout as the primary deployment gate
(2026-08-02) does not change the project's thesis about engine truth; it
sharpens the instrument roles. An evaluation surface has two independent
qualities: label fidelity and behavioral support for the quantity being
measured. Engine-truth capture remains the only channel whose labels are
definitionally correct, and it keeps sole authority over capture-contract
regression and all exact-timing claims on the 60 Hz grid. But the local
val-A session's near-absent `down` coverage (58 positive rows at 1.37
percent prevalence) made it unable to measure that key, which is an
evaluation-support failure, not a truth failure. The Wild7 surface has
the opposite profile — decoded rather than engine-recorded truth, but
deployment-distribution behavior at scale — and it is usable as truth at
all only because the fail-closed admission machinery built for the
ground-truth discipline certified its HUD decodes. Deployment readiness
for labeling public video is therefore gated on the deployment
distribution, while engine truth remains the metrology and timing
channel; neither surface substitutes for the other.
