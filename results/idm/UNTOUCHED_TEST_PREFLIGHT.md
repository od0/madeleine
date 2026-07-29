# Untouched engine-truth test preflight

Status: **blocked; do not score**.

This preflight audits the inputs to the one-pass protocol posted in commit
`4742f995a7f7e8cc414d868048105d4d96c1f5ca`. It deliberately does not contain
test results. The sealed `rec_20260727_220000_test.npz` was not downloaded,
opened, read, or scored, and no predictions were emitted.

The machine-readable companion, `UNTOUCHED_TEST_PREFLIGHT.json`, remains in
the private working repository (its bytes are kept exactly as produced, and
they include storage coordinates the public export excludes).

## Sealed-surface metadata receipt

The user-supplied order and the preregistration declare session
`rec_20260727_220000_test` under object-store prefix
`shards/test-untouched-v1/`: 53,280 frames, 0.34% drops, Chapter 6,
super-covering mask geometry verified, and a clean margin-band leak scan. The
request names an NPZ, `build_manifest.json`, and split lists.

This receipt is intentionally metadata-only. The object-store listing,
`build_manifest.json`, and the two split lists were read without fetching the
NPZ. They bind the staged object to 53,280 source-video frames, 183 excluded
rows, 53,097 shard rows, 46,162 input-active rows, an empty training split,
and this one session in the validation split. The sealed NPZ itself was not
downloaded, opened, or read and is absent from both the local workspace and
the GPU node. Its object-store size is 1,393,997,902 bytes and its provider
MD5 receipt is `54880fb8cf108a2e27e0353c3484f7c4`; content and array hashes
remain intentionally uncomputed until the one authorized inference pass.

## Checkpoint audit

All ten canonical node files exist and were SHA-256 hashed without loading the
test shard. Seven match a tracked expected hash. Three engine-truth-only files
have no tracked expected hash, which is a hard failure under the written
protocol.

All ten files also deserialize successfully with the node's pinned Torch
environment using CPU placement. Each contains both `model_state_dict`
(selected) and `final_state_dict`. Recorded selected steps are 250/250/250 for
the own seeds, 1,250/250/1,250 for Tier B, 6,750 and 6,000 for the two
end-to-end models, and the fixed final endpoints 14,265 and 20,458 for the two
full-corpus models. This loadability check did not access the test shard.

| Model | Canonical node path | Observed SHA-256 | Tracked receipt | Status |
|---|---|---|---|---|
| `own_features_32nc_s0` | `/ephemeral/results/takeover/own_features_32nc_s0/model.pt` | `98a0420f638f7896a492d6994f09fd5814d654d979ac1a0c91b3396f3dbece9d` | none | **blocked** |
| `own_features_32nc_s1` | `/ephemeral/results/takeover/own_features_32nc_s1/model.pt` | `3d805c35348587dc9e25e30f9d754fcee4a997f17676a700c6eeebca70eaedcc` | none | **blocked** |
| `own_features_32nc_s2` | `/ephemeral/results/takeover/own_features_32nc_s2/model.pt` | `ea9f976677e3480673e0eee85d0a1390de385ed4ffeb7eb53620ca609b1e3600` | none | **blocked** |
| `foreign_tier_b_13p45h_32nc_s0` | `/ephemeral/results/takeover/foreign_tier_b_13p45h_32nc_s0/model.pt` | `f4294f31e6f4e84cf5dbbdf0ab0ca836fed931e08bc3b0cd886556749753231f` | checkpoint registry line 1 | pass |
| `foreign_tier_b_13p45h_32nc_s1` | `/ephemeral/results/takeover/foreign_tier_b_13p45h_32nc_s1/model.pt` | `49eeab2afdc35b129e17dd599dcb5c006141cc87c6719618257054bde3d4d155` | checkpoint registry line 2 | pass |
| `foreign_tier_b_13p45h_32nc_s2` | `/ephemeral/results/takeover/foreign_tier_b_13p45h_32nc_s2/model.pt` | `cf92769e338fc982897a64588f947097551b43307decc4c9bc645e1846044598` | checkpoint registry line 3 | pass |
| `foreign_tier_b_13p45h_37m_e2e_aug_32nc_s0` | `/ephemeral/results/takeover/foreign_tier_b_13p45h_37m_e2e_aug_32nc_s0/model.pt` | `1d780811d020cde4a9c28f31e38cba69706ef0bd08ab31666fde9f214f8119dc` | checkpoint registry line 11 | pass |
| `foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0` | `/ephemeral/results/takeover/foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0/model.pt` | `4cfb619be79f0f8626dedc3f07b7c095b1048d4144ee67d3fb3a155cf862748f` | checkpoint registry line 12 | pass |
| `nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0` | `/ephemeral/results/takeover/nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0/model.pt` | `cf55f612382bfa7b9a1b67038b5223a1629782782995f0b008311ba380b34f94` | registry line 14 plus per-run receipt | pass |
| `nitrogen_full_210train_y4n_holdout_26m_128x3_s0` | `/ephemeral/results/takeover/nitrogen_full_210train_y4n_holdout_26m_128x3_s0/model.pt` | `297c6a512914946f9d836467b31afa5b84b74e856ad3fb7b2f7326284161fd09` | registry line 15 plus per-run receipt | pass |

The observed own-model hashes are not substitutes for tracked, pre-seal
expected values. Repository history contains no occurrence of any of the
three hashes.

## Threshold audit

Eight selected-checkpoint val-A reports exist. Each contains two distinct
per-key threshold vectors:

- `/input_active_only/metrics/transition_f1_oracle/<key>/threshold`
- `/all_frames/metrics/transition_f1_oracle/<key>/threshold`

The result tables in `SUMMARY.md` use the input-active-only event surface, so
that vector is the likely intended convention. The preregistration, however,
does not name this JSON pointer, and several all-frame values differ
materially. Choosing a vector after seeing sealed predictions would be a
second analysis setting, so the exact population must be frozen first.

The two full-corpus models have no val-A threshold report in tracked results or
on the GPU node. Their keypress-calibration JSON files were fitted on the first
eight streams of mapped `y4n`, not val-A, and encode a different calibration
object. They cannot silently stand in for the requested val-A transition
thresholds.

All eight available vectors, both population variants, exact artifact hashes,
and JSON pointers are preserved in the companion JSON.

## Blocking findings

1. **Missing tracked own-model hashes.** The three `own_features` files have
   current node hashes but no immutable expected hashes in tracked records.
2. **Missing full-corpus val-A thresholds.** Neither the 103.41-hour nor the
   148.32-hour model has the requested vector.
3. **Threshold-population ambiguity.** Existing legacy val-A reports record
   both input-active-only and all-frame vectors; the protocol does not freeze
   which one applies.

The owner must resolve all three in writing before any test prediction is
created. No threshold may be fitted on this test, and no model may be silently
substituted or reselected.
