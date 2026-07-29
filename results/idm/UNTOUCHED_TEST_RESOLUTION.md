# Untouched engine-truth test: owner resolution

Status: **approved — scoring proceeds mechanically once the preconditions
below are met.** This document is the written owner resolution that
[`UNTOUCHED_TEST_PREFLIGHT.md`](UNTOUCHED_TEST_PREFLIGHT.md) (commit
`f8a51d7`) requires before any test prediction is created. Rulings were made
explicitly by the repository owner on 2026-07-28 in the working session and
are recorded here verbatim in effect. The sealed session remains untouched as
of this commit: no prediction has been produced, and the NPZ has not been
opened.

## Rulings

1. **Own-model checkpoint hashes (preflight blocker 1).** The three
   `own_features_32nc_s0..2` files at their canonical node paths are frozen
   now at the hashes the preflight observed:
   - `own_features_32nc_s0`: `98a0420f638f7896a492d6994f09fd5814d654d979ac1a0c91b3396f3dbece9d`
   - `own_features_32nc_s1`: `3d805c35348587dc9e25e30f9d754fcee4a997f17676a700c6eeebca70eaedcc`
   - `own_features_32nc_s2`: `ea9f976677e3480673e0eee85d0a1390de385ed4ffeb7eb53620ca609b1e3600`

   Because these values were first recorded after the test session was
   sealed, the freeze carries a binding condition: before test contact, each
   checkpoint must exactly reproduce its committed val-A development sidecar
   metrics under the training-era code named in ruling 4. Exact reproduction
   proves the file is the training-time artifact. A checkpoint that fails
   reproduction is struck from the pass and reported as struck; it is never
   substituted, retrained, or "closest-matched".

2. **Full-corpus thresholds (preflight blocker 2).** The two full-corpus
   models (`nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0`,
   `nitrogen_full_210train_y4n_holdout_26m_128x3_s0`; final endpoints 14,265
   and 20,458) receive frozen thresholds from a standard val-A development
   evaluation run before any test contact, under the models' training-time
   tree. val-A is a development surface; fitting thresholds there is the same
   operation the eight legacy models already had. The resulting report JSONs
   are hashed and recorded in the execution addendum before the test pass.

3. **Threshold population (preflight blocker 3).** The frozen vector for
   every model is
   `/input_active_only/metrics/transition_f1_oracle/<key>/threshold`, the
   population `SUMMARY.md` reports and the committed fixed-threshold eval
   path (`--transition-thresholds-from`) reads. All-frame vectors are not
   used anywhere in the test pass.

4. **Inference semantics.** Each model is scored under the code it trained
   under. The six frozen-feature `32nc` models (trained at clean `a9a4144`,
   segment-global feature deltas) are scored from a checkout pinned at
   `a9a4144`. The two end-to-end models and both full-corpus models (trained
   under the preserved `d13ef3e`-dirty tree on the GPU node) are scored from
   that preserved tree, read-only. Validation: every model with a committed
   val-A sidecar must exactly reproduce it under its assigned code tree
   before test contact (the same check as ruling 1's binding condition, run
   for all models); the two full-corpus val-A evals of ruling 2 are fresh
   development records and self-validate by construction.

5. **Support definition.** The single pass computes, from the same stored
   predictions, each model's metrics on its own maximal valid support (its
   window geometry over the session's continuity structure) and on the
   intersection support common to all scored models. Both are reported;
   neither is chosen after seeing results.

6. **One-pass definition.** One inference pass per model. Predicted
   probabilities are stored (`*_preds.npz`); every reported metric family —
   macro/per-key AP with prevalence, state F1 at frozen thresholds, collar-0
   and ±2 transition F1, and both key-state accuracy readings — is computed
   from the stored predictions. The session is never re-inferred. Trivial
   baselines (always-released, persistence, shuffled-event luck anchor,
   per-key prevalence) are label-only computations on the shard, part of the
   same report.

## Preconditions for execution (mechanical; no further owner action)

1. The preparation record (`PREP_RECORD.json`) exists with per-model
   reproduction verdicts under the assigned code trees; any non-reproducing
   model is struck per ruling 1.
2. The two full-corpus val-A threshold reports exist and their sha256 values
   are committed in an execution addendum to this file.
3. The staged shard's integrity receipt (size and provider hash equal to the
   local sealed copy) is re-confirmed at execution time.

When all three hold, the embargo recorded in the engineering log
(2026-07-28 00:06 UTC entry) is lifted for exactly one ten-model pass
executed as pre-registered in the status board (commit `4742f99`). Results
are reported whatever they say.

## Execution addendum (2026-07-28, before test contact)

Preconditions 1 and 2 are met; this addendum records the evidence and
freezes the two execution details the rulings left implicit. No test
prediction exists at the time of this commit.

**Precondition 1 — reproduction record.**
[`untouched_prep/PREP_RECORD.json`](untouched_prep/PREP_RECORD.json)
(sha256 `066201c0383bb4c5941538af0a7ce5f9fa44890b5085d25fc2f9d41ad67d23a0`)
holds the per-model verdicts. The six frozen-feature `32nc` models
reproduce their committed val-A sidecars exactly under the `a9a4144`
pinned checkout. The two end-to-end models reproduce their stored val-A
predictions bitwise under the preserved training tree; their sidecar JSONs
differ only in collar-≥1 transition-matching leaves, fully attributed to
an uncommitted metrics-layer change in that tree, with all headline
metrics identical. Ruling 4's validation is read accordingly, fixed here
before any test contact: prediction-level bitwise identity is the
operative bind; a metric-layer difference with identical predictions does
not strike a model. All ten models stand; none are struck. The ten
checkpoint hashes were re-verified against the frozen values.

**Precondition 2 — full-corpus thresholds.** The two fresh val-A
development reports are committed as
`untouched_prep/nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0_final_val_a.json`
(sha256 `2e39565db70f80c8a142848de74f193c933b8bc08c92ee4254688c92b7c91bbb`)
and
`untouched_prep/nitrogen_full_210train_y4n_holdout_26m_128x3_s0_final_val_a.json`
(sha256 `1cbc4c779a9a45a84f996b5e0657ba2fe9f34dc9450e6a7214072b38ad799a5e`).
For both, the input-active-only and all-frame vectors are identical (the
128×3 val-A support is entirely input-active), dissolving the population
ambiguity for these two models; the ruling-3 pointer applies unchanged.

**Metric-layer freeze.** Inference runs under the per-model code trees of
ruling 4. Every reported metric, for all ten models, is then computed from
the stored predictions by one uniform metric implementation: the committed
`badeline/metrics.py` whose blob sha256 is
`ce3947b199a5e98dcc0af6de2e3eefe8c6967ecb07c5b636319950a67d301434`
(current `main`; the executing checkout must verify this hash before
computing). Per-tree sidecars emitted during inference are retained as raw
execution records but are not the reported numbers.

**Retry discipline.** A mechanical failure before a model emits
predictions (crash, out-of-memory) may be retried; it is not an
evaluation. Once predictions exist, nothing about that model is re-run or
re-inferred; anomalies are reported as found.

**Precondition 3** (staged-shard integrity re-confirmation) executes at
scoring time. With this commit, the embargo is lifted for the single
ten-model pass.
