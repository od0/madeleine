# Counterfactual input-timing identifiability plan

Status: **design and execution plan only; no production collection or result
claim is authorized by this document**. The determinism pilot, production case
manifest, scoring policy, dependency hashes, and exact source commit must be
frozen before any production counterfactual outcome is inspected. Large game
captures, TAS files, and branch traces remain outside Git; only configs,
manifests, aggregate reports, and audit receipts enter the repository.

## Motivating evidence chain

Three committed measurements motivate this study and predict its outcome
shape. E3 measured 44.8% same-pixels/different-action state ambiguity with an
8–16-frame divergence horizon — the observational-equivalence phenomenon this
study formalizes per press. E4 measured that ±1 frame of label timing error
costs 4.5% macro event F1 while heavy transcoding costs nothing, so frame-exact
timing is where label value concentrates. The 2026-07-28 lag-deconvolution
diagnostic (`lag_deconvolution/`, private working repository) found the constant-lag mechanism unsupported
— dash probability rises systematically early (median −7 frames) with no
per-key shift recovering event F1 — establishing that the timing failure is
dispersion, not offset, exactly what effect-locked evidence in a centered
window predicts. This study decides whether that dispersion is a modeling
failure or an identifiability ceiling.

## Recommendation and ordering

Measure the pixels-only identifiability ceiling before spending more compute on
exact-frame localization. The experiment asks:

> Given the key identity and a 16-frame candidate window for free, which press
> frames are observationally distinguishable, and how much ambiguity remains
> even under deterministic engine-truth replay?

This study precedes new exact-localization production work because it determines
what collar-zero press recovery can mean. It does not modify or invalidate the
already-frozen Study-H contract in
[`ORACLE_WINDOW_HIGHRES_REGIONAL.md`](ORACLE_WINDOW_HIGHRES_REGIONAL.md).
Study H has since executed (2026-07-28) and failed its frozen primary gate:
128×128 inputs improved exact-frame localization by +5.15 points over the
32×32 control, but calibration worsened and the conjunctive NLL requirement
failed, so no candidate advanced scientifically to seed confirmation. That
outcome sharpens this study's question rather than answering it: resolution
recovers part of the timing evidence, and this metrology measures whether the
remainder is recoverable from pixels at all.

The execution-aware order is:

0. the inexpensive soft-onset/event-aware engine-truth training arm proceeds
   first or concurrently under its own contract — it can improve the model,
   while this study bounds how much improvement is measurable;
1. deterministic counterfactual input-timing metrology (this plan);
2. retain completed Study H as evidence that resolution helps localization but
   does not, under its frozen calibration gate, authorize confirmation;
3. separately preregister action-conditioned forward-dynamics localization or
   a calibration-aware high-resolution follow-up if warranted;
4. calibrated audio evidence and mapped-label alignment work as separate,
   source-specific studies.

## Claim boundaries

The primary estimand is **press-onset identifiability from rendered pixels**,
not general key-state accuracy and not an unrestricted dense-event F1 ceiling.

- Primary mechanism keys: `jump` and `dash`, where Celeste's buffered virtual
  buttons make delayed execution plausible.
- Negative/control keys: the other five keys, which are retained to distinguish
  buffering from generic delayed or invisible consequences.
- Primary polarity: onset. Releases are a separately contracted extension; they
  have different mechanics (for example variable jump height and grab release)
  and must not be pooled into the buffering claim.
- Primary surface: engine-truth-only, deterministic, room-scoped replay. No
  NitroGen or wild mapped labels enter case construction, fitting, or scoring.
- Primary observation: the exact model-facing 128x128 masked RGB sequence used
  by MADELEINE, produced from a lossless deterministic render. Full-resolution
  pre-mask renders and declared engine probes are diagnostic surfaces.
- Primary oracle: the event key and one 16-frame candidate window are supplied.
  Exactly one onset candidate is emitted, so oracle-window exact accuracy is
  the relevant ceiling. Dense stream event F1 can only be lower and is not
  numerically equated with this result.
- The first production corpus is newly captured and replayable. Existing
  engine-truth sessions did not record a deterministic launch recipe, TAS RNG
  seed, or replay checkpoint and therefore cannot silently be presented as
  counterfactually reproduced.

The study may establish a ceiling for its frozen rooms, states, and input prior.
It cannot establish a universal Celeste ceiling from one room, one route, or a
balanced synthetic offset distribution.

## Existing repository facts and feasibility gap

The current `granny/InputTruth` mod is an accurate observer, not a replay
harness:

- `InputTruthModule.OnEngineUpdate` calls the engine update first and then
  records `Input.*.Check`, player position/speed, dashes, stamina, ground state,
  room, activity, and death.
- It does not record pre-update raw edges, virtual-button buffer counters,
  buffer-consumption events, player state-machine transitions, RNG state, or a
  full engine-state hash.
- It has no input injection, TAS-file playback control, savestate creation, or
  branch orchestration.
- The frozen v1 session schema must not be expanded in place for this study.

The local Celeste installation currently contains a cached
`CelesteTAS-EverestInterop.dll`, but no active CelesteTAS mod directory or
SpeedrunTool installation was found at planning time. A cache entry is not a
reproducible dependency. The execution contract must install and pin an active
CelesteTAS release, record every dependency version and SHA-256, and prove that
the loaded modules match the receipt. CelesteTAS provides deterministic input
files and RNG control; SpeedrunTool savestates are an optional optimization,
not part of the scientific definition.

Use a new sibling experimental mod, `granny/CounterfactualLab`, for mechanics
probes and orchestration. Keep `granny/InputTruth` unchanged as the independent
frame-index, input-state, and player-telemetry instrument. Loading both mods
lets the new harness fail without changing the established truth channel.

## Formal definitions

For one factual onset case `i`:

- `t_i`: physical sampled key-down frame in the factual input trace;
- `W_i`: the 16 legal candidate onset frames;
- `X_i[tau]`: the model-facing rendered clip when the same episode is replayed
  with the target onset moved to candidate frame `tau`, with every non-target
  input fixed;
- `G_i[tau]`: the declared engine-state trace for that branch;
- `O_i[tau]`: the declared downstream outcome trace;
- `t_accept`: the frame where the game consumes/executes the input, if a
  validated mechanics hook exposes it;
- `t_visible`: the first model-facing frame that differs from the ablated
  branch after deterministic-repeat noise is excluded.

The exact observational equivalence set containing the factual press is:

```text
S_i = {tau in W_i : X_i^tau is byte-equal to X_i^t_i over the declared clip}
```

`S_i` is computed only from the 16 primary fixed-release candidates. Diagnostic
duration-preserving and effect-anchor branches never change class membership.

Exact byte equality is the only primary equivalence relation. It is reflexive,
symmetric, and transitive, which makes connected equivalence classes and their
ceilings well-defined. Pixel distances, perceptual distances, and feature
distances are reported as sensitivity analyses; they must not be substituted
post hoc for equality in the primary ceiling.

For a balanced uniform prior over candidates, the per-case oracle exact ceiling
is `1 / |S_i|`. The aggregate uniform ceiling is the mean of those values over
the frozen case population. An empirical-prior Bayes ceiling may be reported
only from a disjoint human-timing population and a prior rule frozen before
production scoring. It is never estimated and evaluated on the same episodes.

Three notions of equality remain separate:

1. **local observational equality:** model-facing clips are equal over the same
   32-frame support used by the oracle localizer;
2. **engine equality:** declared engine-state traces are equal over the
   consequence horizon;
3. **outcome equality:** the branch reaches the same declared downstream state
   after the consequence horizon without an intervening death/room transition.

Hidden buffer-state changes can break engine equality before pixels diverge.
Conversely, particles or capture artifacts can break raw pixel equality without
a gameplay consequence. Reporting all three is mandatory.

## Phase 0: dependency and determinism gate

No production case generation begins until all gates below pass on development
fixtures that are permanently excluded from the production rooms and episodes.

### 0A. Pin the runtime

Create a machine-readable dependency receipt containing:

- Celeste, Everest, `Celeste.dll`, `MMHOOK_Celeste.dll`, and `FNA.dll` versions
  and SHA-256 values;
- active CelesteTAS version, source/release identifier, DLL hash, and TAS input
  syntax version;
- SpeedrunTool and TAS Recorder versions/hashes if either is used;
- Granny and CounterfactualLab source commit and built DLL hashes;
- all enabled mods and settings, including variants, assists, simplified
  graphics, input bindings, rumble, and screen/capture settings;
- platform, runtime, display scale, frame rate, and fixed RNG seed policy.

The launcher refuses unknown or extra gameplay-affecting mods. Third-party
binaries are installed locally and recorded, not committed to this repository.

### 0B. Full-replay determinism fixture (the critical stop-gate)

This is the gate the entire primary result rests on. If byte determinism of
model-facing frames cannot be established, the study reports visibility
curves only — never an exact identifiability ceiling.

Build a short TAS fixture from a stable room start with at least one movement,
jump, dash, landing, and idle interval. Execute it ten times from the same
launch recipe without savestates.

Required pass conditions:

- identical engine-frame counts and no gaps/resets inside the fixture;
- exact equality of the complete declared engine trace across all ten runs;
- exact equality of decoded lossless frames after both answer-key masks and the
  model-facing crop/resize pipeline;
- identical frame-index alignment and no duplicate/drop ambiguity;
- identical emitted mechanics events and input edges.

If raw decorative pixels differ while the declared engine trace is exact, first
try the pinned CelesteTAS RNG and graphics controls. If exact model-facing pixels
remain nondeterministic, stop: a distance threshold learned from repeats is a
useful visibility diagnostic but does not support the primary byte-equivalence
ceiling. The plan must be amended before proceeding.

### 0C. Human-trace-to-TAS replay fixture

Record at least ten short, fresh, room-scoped human episodes from a deterministic
room entry. Convert Granny's 60 Hz held-state rows into TAS held-input files and
replay each episode from the same room-start recipe.

Pass requires exact factual reproduction through the complete local observation
window for at least 9/10 episodes and no unexplained engine divergence before
the first selected target event. Any one-frame convention between Granny's
post-update record and TAS input application must be resolved by a fixed,
fixture-proven conversion rule, never fitted per episode.

The scripted controlled-mechanics result is the primary result of this study
regardless of this gate; human-trace replay is an optional extension that
adds natural-timing priors. If this gate fails, the extension is dropped and
all human-prior or natural-timing ceiling claims are removed, with no effect
on the primary scripted scope.

### 0D. Savestate parity fixture (optional)

If savestates are proposed for speed, compare each savestate branch with a full
room-start replay of the identical branch. Require exact engine and model-facing
pixel equality over the entire scored horizon for ten fixtures including
landing, dash freeze, wall contact, and a room with active particles.

Any failure forbids savestates in production. Full replay remains the reference
path and a fixed fraction of production branches must be redundantly rerun from
room start even after the parity gate passes.

## Phase 1: mechanics instrumentation

### Separate experimental module

Add a new Everest module rather than changing frozen truth semantics:

```text
granny/CounterfactualLab/
  CounterfactualLab.csproj
  everest.yaml
  Source/CounterfactualLabModule.cs
  Source/InputEdgeProbe.cs
  Source/MechanicsProbe.cs
  Source/StateTrace.cs
  Source/TraceWriter.cs
  Source/RunContract.cs
```

The module is allowed to observe private fields or hook action methods only
after a reflection/IL inventory records the exact Celeste assembly hash and the
resolved members. It must fail at load time if a required field or method is
missing. Mechanics probes are labels and diagnostics; they are never model
inputs.

Record at least:

- pre-update and post-update held input vectors;
- sampled press/release edges before the player's update;
- virtual-button `Pressed`/buffer state where exposed;
- explicit buffer-consume/action-execution hooks for jump and dash where a
  fixture can validate their meaning;
- player state-machine id/name, position, exact/subpixel position, speed,
  dashes, stamina, ground/wall contact, dash cooldown, jump-grace/coyote timer,
  variable-jump timer, and other resolved action timers;
- room/session identity, input-active status, death, transition, freeze/hitstop,
  time rate, and RNG/sync hash if exposed;
- pre-mask render hash, post-mask model-facing frame hash, and relative frame.

The exact field list is frozen only after the assembly inventory. Names above
describe required semantics, not permission to guess private member names.

### Mechanism validation fixtures

Create hand-auditable fixtures for:

- grounded immediate jump;
- coyote/grace jump;
- jump pressed shortly before landing and executed on eligibility;
- ignored airborne jump outside the buffer window;
- immediately available dash;
- dash pressed while unavailable and later accepted, if the pinned game build
  supports that behavior;
- ignored/unconsumed dash;
- a directional onset with immediate motion as a negative control;
- a blocked directional onset with no short-horizon motion.

Each fixture states the expected physical edge, buffer interval, acceptance
event, first engine consequence, and first possible rendered consequence before
it is run. A mechanics field or hook enters production only if these fixtures
support its interpretation.

## Phase 2: case population

### Development pilot

Use rooms/templates excluded from production. Start with 48 factual cases:

- 8 immediate jump;
- 8 coyote/landing-buffered jump;
- 8 ignored or unavailable jump;
- 8 immediate dash;
- 8 delayed/blocked or unavailable dash;
- 8 directional negative controls split between visible and blocked movement.

Every case receives all 16 fixed-release candidate onset branches, one ablated
branch, one factual repeat, and both shift policies below. This pilot chooses
only operational quantities: capture backend, full-replay throughput, safe
horizons, storage, and whether a private mechanics probe is valid. Pilot
outcomes cannot enter the production ceiling or choose production rooms.

### Production population

Freeze exact room starts, episode ids, case ids, state strata, exclusions, and
candidate offsets before branch outcomes are generated. The initial target is:

- at least 50 estimable onsets for each of jump and dash in immediate and
  buffered/delayed strata;
- at least 30 estimable onsets per remaining key across visible and
  blocked/no-short-effect strata;
- at least three room starts and two independently recorded episodes/templates
  per estimable key/state stratum;
- a balanced scripted population that supplies the minimum controlled support;
- only if Phase 0C passes, additional disjoint fresh-human timing-prior and
  evaluation episodes within the frozen branch budget. Human cases need not
  meet the controlled stratum targets unless a human-specific ceiling is
  claimed.

Case selection may inspect factual input and pre-event engine state, but not
counterfactual pixels or branch outcomes. A case is eligible only when:

- the full pre-context, 16-frame candidate window, 32-frame local observation,
  and consequence horizon are within one gap-free, input-active room segment;
- exactly one onset for the requested key occurs in the candidate window;
- the factual release follows the last candidate frame, so every fixed-release
  onset candidate is a legal non-negative hold; cases that fail this condition
  are excluded from the primary population rather than silently dropping
  candidates;
- no second same-key onset enters support;
- no pause, death, cutscene, room transition, or capture discontinuity enters
  support;
- the pre-event checkpoint/room-start replay hash matches the factual receipt;
- all non-target input states are available for exact replay.

Report a clean subset with no other key transition in a frozen neighborhood of
the target and a natural subset that retains other-key transitions while fixing
them identically across branches. The clean subset is primary for causal
attribution; the natural subset measures applicability.

Additionally preregister a secondary **high-input-density cohort**: at least
20 cases per mechanism key whose candidate windows sit inside speedrun-density
input (multiple other-key transitions within the local observation window,
fixed identically across branches). This cohort exists to measure the scope
limitation of the isolated-press eligibility rules — wild-corpus play is
dense, and the identifiability answer may differ there. It reports the same
metrics, is never pooled with the primary cohort, and supports no headline on
its own.

### State strata

At minimum preserve, without pooling away:

- grounded, airborne, landing-approach, coyote/grace, wall contact/climb,
  dash-available, dash-unavailable/cooldown, and input-ignored;
- moving left/right, near-zero horizontal speed, and direction blocked by
  collision;
- accepted immediately, accepted after delay, and never accepted within the
  consequence horizon;
- room and episode/template identity.

Sparse strata remain visible and inestimable rather than being merged after
outcomes are known.

## Phase 3: counterfactual branch construction

For each eligible factual case, generate:

1. **fixed-release candidates:** move only the onset to every candidate in
   `W_i`, leaving the factual release at its absolute frame. The `tau = t_i`
   candidate is the factual branch; it is not stored a second time;
2. **repeat:** a byte-identical factual rerun for ongoing determinism checks;
3. **ablated:** remove the target onset and keep the key released until the
   factual release;
4. **duration-preserving candidates (diagnostic subset):** move both onset and
   release by the same offset, preserving pulse/hold duration;
5. **effect-anchor branches (accepted buffered cases):** move the complete
   target pulse to `t_accept` and `t_visible`, preserving its duration and
   deduplicating anchors that coincide. These branches score behavioral safety;
   they do not enter `S_i` or the press-onset ceiling.

Fixed-release shifts are primary because they isolate onset time while keeping
the later held-state endpoint fixed. They are generated for every production
case. Duration-preserving shifts answer the different behavioral question of
whether the entire input pulse can move. Generate them for a case-id-hash-based,
state-stratified subset whose size is frozen after the pilot and before any
production branch is inspected. The default cap is 20% of cases, reducible only
before production to satisfy the 10,000-branch gate. This secondary policy
cannot replace or alter the primary fixed-release result.

Effect-anchor branches use duration preservation so the proposed delayed action
remains a legal pulse even when the factual release precedes the anchor. Their
outcome horizon and equivalence rule are frozen with the case manifest.

The branch generator must prove mechanically that:

- only the requested key differs from factual;
- the declared onset/release edits match the branch policy exactly;
- all other keys and all inputs outside the edited interval are byte-identical;
- every branch starts from the same room-start/checkpoint hash and RNG state;
- candidate offsets remain balanced and inside the frozen window;
- generated TAS input expands back to the expected per-frame key matrix.

Use full room-start replay as the reference. If validated savestates accelerate
production, rerun at least 10% of branches and every surprising equivalence
class from full room start before admission.

## Phase 4: render and trace pipeline

Use lossless fixed-rate capture. Do not measure bitwise identifiability through
the current lossy x264 session default.

For every branch retain or derive:

- 60 Hz engine trace and mechanics events;
- lossless render frames spanning pre-context through consequence horizon;
- the exact corrected masks for frame strip, opaque input overlay, and any
  enabled calibration overlay;
- canonical model-facing 128x128 uint8 RGB frames produced by the same crop,
  area/interpolation, color, and masking code as the IDM data path;
- optional 32x32 frames matching the bounded pixel study;
- SHA-256 for input file, trace, raw frame payload, model-facing array, and
  config/dependency receipt.

The canonical array, not the video container bytes, defines pixel equality.
The validator must rescan outside all declared answer-key masks for
key-correlated leakage, reusing the hardened coverage logic that caught the
historical overlay sliver.

Use a separate, content-bound artifact format rather than changing frozen
session format v1:

```text
/ephemeral/data/counterfactual-identifiability-v1/
  dependency_receipt.json
  case_manifest.json
  branches/<case_id>/<branch_id>/
    input.tas
    branch_manifest.json
    trace.parquet
    model_frames.npy
    raw_capture_receipt.json
  dataset_manifest.json
  counterfactual_dataset_complete.json
```

Write every branch into staging, validate it, rename it atomically, and publish
the dataset completion marker last. No branch or marker is overwritten.

## Phase 5: scoring

### Primary metrics

Report overall, per key, per state stratum, and per room/episode:

- fraction with `|S_i|` = 1, 2, 3-4, 5-8, and 9-16;
- ambiguous fraction, defined as `|S_i| > 1`;
- mean/median observational equivalence width;
- uniform-prior oracle exact ceiling, mean `1 / |S_i|`;
- press-to-accept, press-to-first-engine-divergence, and
  press-to-first-visible-effect lags, with censored counts;
- first-visible-effect frame agreement at raw, 128x128, and 32x32 resolution;
- fixed-release versus duration-preserving equivalence;
- local observational, engine-trace, and downstream-outcome equivalence;
- fraction where pressing at `t_accept` or `t_visible` is
  outcome-equivalent to the factual buffered press.

The last metric adjudicates whether effect-anchored pseudo-labels are safe for
behavior cloning. Effect-time relabeling is not recommended merely because the
original press was visually ambiguous.

### Empirical-prior ceiling

If the human replay gate passes, split human episodes by room/template before
any prior fitting. Fit `P(t_i | key, frozen state stratum)` on the prior split
only, choose the most likely candidate within each production equivalence
class, and score on disjoint episodes. Do not condition the prior on private
mechanics probes unavailable to a video model.

Report both the balanced uniform ceiling and the disjoint empirical-prior
ceiling. The empirical number is secondary because it can exploit stereotyped
human timing even when pixels are uninformative.

### Uncertainty

Use 5,000 deterministic bootstrap resamples of whole room episodes/templates,
never individual branches or frames. Keep all branches of a factual case in the
same resampling unit. Report percentile intervals for:

- ambiguous fraction;
- uniform and empirical-prior ceilings;
- buffered-minus-immediate ambiguity;
- jump-minus-control and dash-minus-control ambiguity;
- effect-time outcome-equivalence rate.

Per-key/state strata with fewer than 30 factual cases or fewer than two
episodes/templates are descriptive and cannot support a headline.

### Secondary visibility curves

Pixel L1/L2, changed-pixel fraction, SSIM, learned-feature distance, and motion
energy may describe how evidence emerges after a press. Thresholds for those
curves are fixed from factual-repeat noise on development fixtures. They do not
change primary byte-equivalence classes or their ceiling.

## Preregistered interpretation gates

This is primarily metrology, but the following gates prevent rhetorical
promotion after the result.

### Material identifiability limit

Claim a material pixels-only limitation for a key/state stratum only if:

1. at least 30 factual cases from at least two episodes/templates are
   estimable;
2. the ambiguous fraction is at least 0.20 and its whole-episode 95% lower
   bound exceeds 0.10;
3. the uniform-prior exact ceiling is below 0.90 and its 95% upper bound is
   below 0.95;
4. factual-repeat determinism is 100% on admitted model-facing arrays.

Failure does not prove full identifiability; it means this corpus does not
measure a material ceiling under the frozen definition.

### Buffer attribution

Attribute ambiguity specifically to buffering only if:

1. the validated mechanics probe records a physical edge followed by delayed
   consumption/execution;
2. buffered cases exceed matched immediate cases in ambiguous fraction by at
   least 0.20 with a positive paired/block-bootstrap lower bound;
3. at least one non-buffered control key/state does not show the same ambiguity
   pattern;
4. the result reproduces across at least two rooms/templates.

Otherwise report delayed/invisible action consequence without calling its
mechanism buffering.

### Effect-anchored target

Call an effect-anchored label behaviorally safe only if pressing at the proposed
effect frame reproduces the factual downstream outcome in at least 95% of
estimable cases and the 95% lower bound exceeds 90%, separately for each key
being promoted. No pooled jump/dash average can hide failure for one key.

### Roadmap consequences

- **Material ambiguity measured:** report collar-zero results relative to the
  counterfactual ceiling; add interval/posterior or effect-event targets; retain
  physical press onset as censored where needed.
- **Little ambiguity, full-resolution pixels differ:** interpret the result
  alongside completed Study H, which found a robust high-resolution exactness
  gain but failed its calibration gate; any next encoder test must be a new
  frozen follow-up rather than a rerun or continuation of Study H.
- **Pixels remain equal but engine branches differ:** direct pixels-only timing
  is structurally limited; prioritize action-conditioned dynamics and explicit
  uncertainty.
- **Effect-time shifts are outcome-equivalent:** effect-anchored pseudo-labels
  may be evaluated for BC under a separate frozen downstream contract.
- **Effect-time shifts change outcome:** do not relabel to execution/visibility
  time; use intervals/posteriors or accept irreducible label uncertainty.
- **Human replay fails but scripted replay passes:** scope every conclusion to
  controlled mechanics; do not publish a natural-play ceiling.
- **Determinism fails:** stop before ceiling scoring and report the capture/replay
  limitation.

## Required repository surfaces

Implementation should add new files rather than altering completed experiment
artifacts:

```text
specs/counterfactual_trace_format.md
granny/CounterfactualLab/CounterfactualLab.csproj
granny/CounterfactualLab/everest.yaml
granny/CounterfactualLab/Source/*.cs
experiments/configs/counterfactual_input_identifiability_v1.json
experiments/build_counterfactual_cases.py
experiments/expand_counterfactual_tas.py
experiments/prepare_counterfactual_identifiability.py
experiments/score_counterfactual_identifiability.py
experiments/validate_counterfactual_identifiability_run.py
experiments/run_counterfactual_identifiability.sh
tests/test_build_counterfactual_cases.py
tests/test_expand_counterfactual_tas.py
tests/test_prepare_counterfactual_identifiability.py
tests/test_score_counterfactual_identifiability.py
tests/test_validate_counterfactual_identifiability_run.py
results/idm/COUNTERFACTUAL_INPUT_IDENTIFIABILITY.md
```

If game automation requires a small external controller, place it under
`experiments/` and communicate through content-bound command/result files.
Do not use UI timing, wall-clock sleeps, or a mutable Studio document as the
scientific record.

## Focused tests and validators

At minimum cover:

- dependency receipt rejects a missing, extra, or hash-mismatched mod;
- room-start recipe and RNG seed are present and immutable;
- Granny-to-TAS conversion expands to the exact original held-state matrix;
- the fixed one-frame convention reproduces development fixtures;
- branch edits touch only the requested key and declared interval;
- fixed-release and duration-preserving policies are distinct and exact;
- effect-anchor branches preserve duration, deduplicate equal anchors, and are
  excluded from onset equivalence classes;
- candidate offsets are balanced and all windows/horizons are gap-safe;
- case selection cannot read counterfactual outcomes;
- masks cover every answer-key pixel and no correlated sliver remains;
- canonical RGB dtype, shape, color order, crop, resize, and hashes are exact;
- repeated factual branches are byte-identical;
- savestate and full-replay branches are identical when savestates are enabled;
- equivalence classes are reflexive, symmetric, and transitive;
- toy ceilings are exact for singleton, paired, and fully ambiguous classes;
- empirical-prior fitting and scoring episodes are disjoint;
- bootstrap units never split a case or episode;
- censored/no-effect cases remain in denominators under the frozen policy;
- report, sidecar, and completion paths refuse collision/overwrite;
- independent validation replays a fixed branch subset and regenerates the
  aggregate report from primary arrays.

## Runtime and storage gates

This is a game-simulation and data-integrity study, not a GPU-training study.
Run the determinism and development pilot locally. No Prime GPU is needed.

Before production, benchmark at least 100 full branch replays and record:

- wall time per room-start and per scored branch;
- fast-forward multiplier and whether it changes output;
- peak host RAM/GPU memory;
- raw and canonical bytes per branch;
- projected branch count and total disk.

The branch projection must count 16 fixed-release candidates (including the
factual onset candidate), one independent factual repeat, one ablation, all
duration-preserving diagnostic branches, and all deduplicated effect-anchor
branches. If the frozen case targets and the diagnostic-subset cap do not fit
below the limit, reduce the diagnostic subset first and then reduce factual
case counts while preserving the minimum per-stratum support required for any
headline.

With seven current input keys, the minimum controlled target is 350 factual
cases: 200 jump/dash state-stratum cases plus 150 controls. That costs 6,300
primary/repeat/ablation branches. A 20% duration-preserving subset adds 1,120;
two effect anchors for each of the 100 minimum buffered jump/dash cases add at
most 200 before deduplication, for a planning total of 7,620. The remaining
budget can admit human-prior cases if Phase 0C passes.

Initial production limits:

- at most 10,000 admitted branches;
- at most 100 GB of new local/ephemeral artifacts;
- at most 12 hours projected collection wall time on the assigned local
  machine;
- no remote provisioning or paid compute without separate authorization.

If the pilot exceeds a limit, reduce the number of frozen factual cases before
production or validate savestate acceleration. Do not shorten observation or
consequence horizons after seeing counterfactual outcomes.

## Execution sequence

1. **Inventory:** inspect the pinned Celeste/CelesteTAS assemblies and enumerate
   the exact buffer/action fields and hook points available to
   CounterfactualLab.
2. **Dependency contract:** install active pinned dependencies and write the
   dependency-receipt builder/validator.
3. **Trace spec:** freeze the counterfactual trace schema without modifying
   session format v1.
4. **Determinism fixture:** implement room-start TAS playback, lossless capture,
   canonical masking/resizing, and ten-run equality checks.
5. **Human replay fixture:** convert fresh Granny input traces to TAS and freeze
   the single global frame convention; fall back to scripted-only scope if it
   fails.
6. **Mechanics probes:** implement and validate physical-edge, buffer, accept,
   state, and render probes on hand-auditable fixtures.
7. **Branch generator:** implement ablation, fixed-release,
   duration-preserving, and effect-anchor shifts with exact input-delta tests.
8. **Development pilot:** collect the excluded 48-case pilot and benchmark
   runtime/storage; do not admit its ceiling.
9. **Production freeze:** commit the exact case manifest, rooms, strata,
   exclusions, horizons, score policy, bootstrap seed, dependency hashes, and
   source SHA before production counterfactual inference.
10. **Production collection:** generate each branch once into unique staging,
    validate, publish atomically, and retain factual/full-replay controls.
11. **Score and audit:** compute byte-equivalence classes, ceilings, lag and
    outcome metrics, whole-episode intervals, and an independent replay audit.
12. **Roadmap update:** publish one bounded verdict and only then decide whether
    a separately contracted calibration-aware high-resolution follow-up or
    interval/effect/dynamics targets should take precedence. Study H itself
    remains closed.

## Final deliverables

The final repository result should contain:

- exact dependency, case-population, and source-code receipts;
- determinism and human-replay gate outcomes;
- case flow from factual onsets through exclusions and admitted branches;
- per-key/state equivalence-width and lag tables;
- uniform and, if authorized by the replay gate, disjoint empirical-prior
  ceilings with whole-episode intervals;
- effect-time behavioral-equivalence results;
- representative masked frame/trace examples that expose immediate, buffered,
  invisible, and ignored cases without including answer-key overlays;
- one plain conclusion: material visual ambiguity measured, no material
  ambiguity measured under this scope, or determinism/replay inconclusive.

The presentation headline should remain:

> Given the event identity and a candidate window for free, what pins the exact
> frame, and how much of the residual is unidentifiable by construction?
