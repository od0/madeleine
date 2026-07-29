# Untouched engine-truth battery: pre-registration

Status: **registered before capture; no session in this battery exists yet.**
This document freezes the design of the second untouched engine-truth
evaluation before any recording is made, so that no choice about sessions,
models, or reporting can be influenced by observed outcomes. It extends the
executed single-session test (see `UNTOUCHED_TEST_RESOLUTION.md` and
`untouched_test/`) whose Chapter 6 result could not separate recording
transfer from content shift. Registered 2026-07-28, owner Bryan Russett.

## Sessions

- Four capture sessions, one per chapter: Chapter 1, Chapter 2, Chapter 3,
  Chapter 4 — 5 to 10 minutes of play each, captured with the documented
  rig profile, each as its own session with its own capture-quality gate.
- **Chapter 1 is the designated anchor**: its content family overlaps the
  own-data training rooms, so it isolates pure recording-transfer. Chapters
  2–4 are content the local rig has never recorded but the mapped corpus
  covers, forming a content-familiarity gradient between the anchor and the
  already-scored Chapter 6 session.
- Optionally, a fifth session in a chapter unseen by all own-data capture
  may be recorded under the same rules; its absence changes nothing in this
  registration.
- A session that fails the 2% capture drop gate is recorded as a failed
  capture (itself a per-chapter finding, given the known game-render-rate
  behavior in heavier chapters), is never scored, and may be re-captured
  exactly once under a fresh session id. Failure counts are reported.
- Every captured session is a final-test surface from the moment recording
  starts: no debugging, threshold selection, or model development may touch
  it. Sealing follows the executed test's procedure: session validation,
  shard build under the corrected fail-closed mask geometry, mask-leak scan,
  staging to the durable object store under `shards/test-battery-v1/`, and
  hash receipts recorded before any scoring.

## Models

- The same ten frozen checkpoints scored in the executed untouched test,
  under the same frozen hashes, thresholds, per-model training-era inference
  code, and hash-pinned uniform metric layer fixed by
  `UNTOUCHED_TEST_RESOLUTION.md`.
- Additionally, if and only if the own-v3 rerun models (three engine-truth
  seeds and three mapped-pretrained seeds on the mask-corrected shards) have
  their checkpoint hashes registered in tracked records and their val-A
  threshold reports committed **before the first battery session is
  scored**, all six are added to the scored set. This is a fixed eligibility
  rule, not a choice made later; partial inclusion is not permitted.

## Scoring

- One inference pass per model per session; stored predictions; every
  reported metric computed from stored predictions by the same hash-pinned
  metric implementation; sessions are never re-inferred or re-scored.
- Metric families and baselines exactly as in the executed test: macro and
  per-key AP with prevalence, state F1 at the frozen thresholds, collar-0
  and ±2 transition F1, both key-state accuracy readings, and the label-only
  trivial baselines per session.
- Support: per-model maximal support and cross-model intersection support,
  both per session and pooled.

## Reporting rules

- The pooled battery aggregate (all admitted battery sessions) is the
  primary result.
- Per-chapter rows form the content-familiarity gradient and are reported
  in full alongside the existing Chapter 6 result.
- Per-chapter per-rare-key event rows are descriptive only and support no
  headline.
- Results are reported whatever they say; no session, model, threshold, or
  support definition may be added, removed, or redefined after the first
  battery number is seen.
