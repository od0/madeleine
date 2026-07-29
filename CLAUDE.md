# MADELEINE contributor guide

This file is the operating contract for humans and coding agents working in the
repository. The public overview is [README.md](README.md), the measured status
board is [PROGRESS.md](PROGRESS.md), and the research roadmap is
[PLAN.md](PLAN.md).

## Project objective

MADELEINE studies inverse dynamics in *Celeste*: predict the seven player
controls from a temporal window of gameplay frames. The research question is
not merely whether a network can fit action labels. It is when an action is
recoverable from visual evidence, how future context helps, and how alignment,
masking, label noise, distribution shift, and model capacity affect the answer.

The project has three supervision channels:

1. locally captured gameplay with exact engine-truth actions;
2. NitroGen source videos with mapped gamepad labels;
3. public gameplay videos whose visible keyboard overlays can be decoded only
   after explicit human review and provenance checks.

## Naming

The name is MADELEINE. A sensory trace recovers a lost past; an inverse
dynamics model recovers the action that produced a visual trace. Components
follow the theme:

- `theo` owns capture and alignment because Theo is a photographer: the camera
  is his natural part of the system.
- `badeline` reconstructs the action hidden behind visible motion. Badeline is
  Madeline's shadow counterpart, which makes her a fitting name for the model
  that infers the unseen side of play.
- `goldenberry` applies strict record-consistency checks. A golden strawberry
  certifies an exacting, failure-free run, matching a verifier that admits a
  session only when every contract passes.
- `granny` is the privileged engine observer. Granny sees through Madeline's
  surface story and understands what the mountain is doing, so she represents
  engine truth. The existing Everest assembly and directory retain the
  compatibility name `InputTruth` under `granny/InputTruth/` so recorded session
  metadata and installed capture setups do not break.

## Non-negotiable research rules

### Mask every answer key

Frame-index strips, controller widgets, and keyboard overlays must be masked
before a frame reaches a model. A suspiciously large gain triggers a masking
audit before any other interpretation.

### Keep label kinds explicit

- `truth.parquet` is reserved for engine truth.
- NitroGen and decoded-overlay labels are mapped or inferred supervision.
- Evaluation against mapped labels measures agreement with that label source;
  it is not engine-truth performance.
- Never collapse source availability, mapped-label hours, train-ready hours,
  and evaluation support into one “data size” number.

### Preserve temporal boundaries

Training and evaluation windows may not cross a missing chunk, non-consecutive
engine-frame index, capture reset, or declared sequence boundary. Isolated
repeated frames can be retained when policy permits, but the decision and count
must remain visible in manifests. Variable-rate sources are sampled by
timestamp onto the declared label grid; they are never treated as 60 Hz merely
because container metadata says 60.

### Freeze evaluation roles

A session used for debugging, threshold selection, or checkpoint selection is a
development asset. It cannot later become an untouched test. Capture final
engine-truth sessions only after the model recipe, checkpoint rule, thresholds,
and evaluation code are frozen.

### Report state and events separately

Per-frame accuracy is not a headline metric. Reports include:

- macro and per-key average precision with label prevalence;
- state F1 with the threshold-selection surface named;
- onset and release F1 at exact and ±2-frame collars;
- prediction support and continuity coverage;
- checkpoint identity and whether the endpoint was selected or fixed.

State recognition and transition timing are different objectives. Improvement
in one does not imply improvement in the other.

### Fail closed on wild data

AI-generated layout, boundary, or offset proposals are diagnostics only.
Wild-video admission requires source-bound evidence, immutable acceptance
artifacts, an allowed reviewer provenance, final decode QC, and a complete
publication manifest. Editable booleans are not review records.

## Data contracts

The canonical local session format is
[specs/session_format.md](specs/session_format.md). Supporting contracts live in
`specs/`.

- Own data uses the 60 Hz engine grid and exact frame-index alignment.
- NitroGen labels are one row per declared source frame. Chunk endpoints are
  normalized to half-open intervals at ingestion.
- Missing 20-second NitroGen chunks split runs but do not discard the rest of a
  video.
- Controller bindings, axis sign, viewport, mask geometry, decoder mode, and
  continuity are independent quality dimensions.
- All train/validation/test splits occur at whole-session or whole-video level.
- Large videos, feature arrays, raw session captures, and model checkpoints do
  not enter ordinary Git history.

The measured corpus facts and current admission policy are in
[data/dataset_card.md](data/dataset_card.md),
[results/idm/CORPUS_AUDIT.md](results/idm/CORPUS_AUDIT.md), and
[results/idm/TRAINING_DATA_POLICY.md](results/idm/TRAINING_DATA_POLICY.md).

## Model and training contract

`badeline` supports pixel and precomputed-feature inputs, causal and centered
windows, feature deltas, configurable projection/temporal capacity, class
balancing, transition weighting, temporally consistent augmentation, and fixed
or selected endpoints.

Before a long run:

1. run the relevant focused tests;
2. run a real-data smoke through load, forward, backward, checkpoint, and
   evaluation;
3. record observed host RAM, device memory, and throughput;
4. freeze the config, split lists, seed, endpoint, and output path;
5. use an atomic completion marker only after all required reports and
   sidecars exist.

The optimized training objective and checkpoint-selection objective must be
aligned or the endpoint must be fixed in advance. The repository retains both
selected and final weights when they differ.

## Repository map

```text
granny/InputTruth/        granny engine-truth mod (compatibility assembly name)
theo/                   capture and alignment tools
data/                   session validation and shard construction
nitrogen/               NitroGen recovery, mapping, masking, and slicing
harvest/                wild input-overlay acquisition and review pipeline
badeline/               model, training, metrics, and evaluation
goldenberry/             record-consistency verifier
experiments/             frozen configs, launchers, and analysis scripts
specs/                   data and overlay contracts
results/                 compact machine-readable results and reports
report/                  chronological findings and engineering records
infra/                   provider-neutral compute and storage notes
tests/                   synthetic and contract tests
```

## Public and internal documents

The public repository is a curated subset of this working repository; the curation record is
the single authority on what leaves the working repository. In practice:

- Public-safe documents: `README.md`, `CLAUDE.md`, `PLAN.md`, `PROGRESS.md`,
  the pages under `docs/` (engineering lessons and history), `specs/`,
  `report/README.md`, and the result reports and figures included in the
  public repository.
- Internal working logs, not exported: `report/findings_log.md`,
  `report/engineering_log.md`, `results/idm/ENGINEERING_LOG.md`,
  `LESSONS.md`, `NOTES.md`, raw session captures, and run directories.

When a public document needs to reference an internal artifact, describe it as
living in the private working repository instead of linking to a path the
export does not contain.

## Working protocol

- The project was built with direct commits to `main` by a small set of
  coordinated internal sessions; temporary worktrees were used for bounded
  isolation and merged promptly rather than becoming permanent forks. External
  contributors should instead work on feature branches and open pull requests
  for review.
- Read `PROGRESS.md` and the relevant result report before starting.
- Preserve unrelated user changes. Never discard a dirty worktree to make an
  integration easier.
- Make small, coherent commits. Stage only files belonging to the current unit
  of work.
- Update measured status when a state changes, not from memory at the end.
- Append material failures, diagnoses, fixes, and resource tradeoffs to the
  relevant internal engineering log. Promote durable, sanitized principles to
  [docs/engineering-lessons.md](docs/engineering-lessons.md).
- Routine unchanged monitoring snapshots do not belong in documentation.
- Do not put credentials, active host addresses, account identifiers, or local
  personal paths in Git. Operational commands use placeholders or environment
  variables.
- Do not describe a process as complete until its outputs, reports, hashes, and
  completion marker have been checked.

## Verification

The default local checks are:

```bash
uv sync
uv run pytest -q
git diff --check
```

Use focused tests while iterating, then the broad suite before integration.
Expensive subprocess-backed tests have one owner at a time. Cloud launchers and
large-data validators have additional acceptance commands in their adjacent
reports and scripts.

## Research priorities

Execution priorities and the current next sequence are maintained in
[PROGRESS.md](PROGRESS.md); the durable research questions are in
[PLAN.md](PLAN.md). This file does not duplicate either list. The latest
numerical interpretation is in
[results/idm/SUMMARY.md](results/idm/SUMMARY.md). If any document and the
status board disagree, measured process state and committed artifacts win; fix
the stale document in the same change.
