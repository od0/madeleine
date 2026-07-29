# Untouched engine-truth test: single authorized pass

Session `rec_20260727_220000_test`, scored 2026-07-28 04:21-04:26 UTC on
`madeleine-codex`, one inference pass per model, exactly as pre-registered in
`PROGRESS.md` next-sequence item 2 (protocol commit `4742f99`) and resolved in
`results/idm/UNTOUCHED_TEST_RESOLUTION.md` (rulings + execution addendum,
commit `b552851`). Results are reported as found. No git commit was made from
this pass; every artifact lives under `/ephemeral/results/untouched_test/`
with a copy on the workstation.

## Integrity chain (all checks passed before any prediction)

- **Sealed shard (precondition 3).** Object-store receipt re-confirmed at
  execution time via `rclone lsjson --hash` on
  the durable object store's `shards/test-untouched-v1/` prefix: size `1,393,997,902`, provider
  MD5 `54880fb8cf108a2e27e0353c3484f7c4`, equal to the local sealed copy
  (`data/shards_test/rec_20260727_220000_test.npz`, local MD5 identical,
  local sha256 `0c9f939709ff446a0f99aa789c65abbd3df59545e931429df0cd7a9abef4fd6f`).
  Staged to `/ephemeral/data/test_untouched/`; staged sha256 matches the local
  sealed copy exactly. `build_manifest.json`, `train_sessions.txt` (empty
  split), and `val_sessions.txt` MD5-match the remote receipts and the local
  copies. 53,097 shard rows, 46,162 input-active, 183 excluded of 53,280
  source-video frames, engine grid 60 Hz, session-unit split — all equal to
  the sealed metadata receipt in `UNTOUCHED_TEST_PREFLIGHT.md`.
- **Checkpoints (step 2).** All ten `/ephemeral/results/takeover/<run>/model.pt`
  sha256 values re-verified equal to the frozen values in
  `untouched_prep/PREP_RECORD.json` (itself sha256
  `066201c0383bb4c5941538af0a7ce5f9fa44890b5085d25fc2f9d41ad67d23a0`,
  matching the execution addendum). No model was struck; none required retry;
  all ten inferences returned rc=0 on first attempt.
- **Metrics checkout (step 3).** `/ephemeral/madeleine-metrics` was created by
  streaming `git archive` of commit
  `b5528513316031fb23da8757fc77559e99d73fd9` (current `main`) from the
  workstation repository over ssh. Verified before any reported metric:
  `badeline/metrics.py` blob sha256
  `ce3947b199a5e98dcc0af6de2e3eefe8c6967ecb07c5b636319950a67d301434` (the
  frozen value), and the imported module path resolved to this checkout.
- **Frozen thresholds (rulings 2-3).** For every model the
  `/input_active_only/metrics/transition_f1_oracle/<key>/threshold` vector of
  its designated val-A development report, read by the committed
  `--transition-thresholds-from` path at inference and re-read independently
  by the uniform scorer. Sources and sha256:
  - `own_features_32nc_s0_val_a.json` `1cb0dcf17538c9748e53e2409106cef964c255954328d9e0a2a4f329d3257f30`
  - `own_features_32nc_s1_val_a.json` `82766b8896953f516444b1bdee92c2b8aba9b9131e38eb6636846d0f6c63bc0a`
  - `own_features_32nc_s2_val_a.json` `667185aa4bcac16358e1a156b7122cbc3ec1e9b6da52a397d362b6b68845ba59`
  - `foreign_tier_b_13p45h_32nc_s0_val_a.json` `1b9c51a6a37d13c989d772d2841c8a0a0c245ffd52ae2f2812492843083c63a4`
  - `foreign_tier_b_13p45h_32nc_s1_val_a.json` `3ff1435b1f0fd1b0b1fc29e02df09ae9177d17f46200e75e2b0f448da1da610d`
  - `foreign_tier_b_13p45h_32nc_s2_val_a.json` `ae7ed6a38b011f4af84587993992e85f053ba60f1ba6bd558644fb8271a456af`
  - `foreign_tier_b_13p45h_37m_e2e_aug_32nc_s0_val_a.json` `bf0523fb0cee55c917574b5f039e2846a0ccdb02db6cdcef16e8e07ab0d3177d`
  - `foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0_val_a.json` `7be2e2b0b6a9c8b7922dee38e27627f27ab358f6fe31e554f08a3fee0a6dac4b`
  - `untouched_prep/nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0_final_val_a.json` `2e39565db70f80c8a142848de74f193c933b8bc08c92ee4254688c92b7c91bbb` (addendum-verified)
  - `untouched_prep/nitrogen_full_210train_y4n_holdout_26m_128x3_s0_final_val_a.json` `1cbc4c779a9a45a84f996b5e0657ba2fe9f34dc9450e6a7214072b38ad799a5e` (addendum-verified)
- **Test features (step 4).** Computed once with the committed converter of
  the metrics checkout (`data.precompute_features shards` — this code path is
  byte-identical between training-era `a9a4144` and current `main`; the only
  converter changes since `a9a4144` are in the foreign-video path), mirroring
  the command that produced `/ephemeral/data/own_features`:
  `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/ephemeral/madeleine-metrics
  .venv/bin/python -m data.precompute_features shards --inputs
  /ephemeral/data/test_untouched/rec_20260727_220000_test.npz --out
  /ephemeral/data/test_untouched_features --device cuda --batch-size 512`.
  Output: `(53,097, 512)` float16,
  `resnet18_imagenet_avgpool_float16_v1`, feature-shard sha256
  `d13baa44775747a0148df9e08f95bf79edb0db3112ee52be42e9f80bfe1a0264`; keys,
  engine indices, activity flags, and session id verified bitwise identical
  to the pixel shard.
- **Per-model code trees (ruling 4).** Six frozen-feature `32nc` models under
  `/ephemeral/madeleine-pinned` @ `a9a414452c07fb101d26faee9fa13864dde922b4`
  (training-era segment-global feature-delta semantics; loads the selected
  `model_state_dict` by construction). Two end-to-end and two full-corpus
  models under the preserved read-only dirty tree `/ephemeral/madeleine` @
  `d13ef3e` (porcelain-status sha256 re-confirmed equal to PREP_RECORD:
  `3e4d1a333a391eb04a87e092822577e78838024caebb93b4004460eed0dbc56e`),
  with `--weights selected` (e2e) and `--weights final` (full-corpus, fixed
  endpoints 14,265 / 20,458). Interpreter: the node training venv
  (Python 3.12.13, torch 2.13.0+cu129, cuDNN 92000, A100 80GB PCIe,
  driver 565.57.01) with `PYTHONPATH` pinning each tree.
- **Uniform metric layer.** Every reported number below was computed from the
  ten stored prediction sidecars by `score_untouched.py` (archived beside this
  report) importing `badeline/metrics.py`, `badeline/train.py` helpers, and
  the committed `experiments/keypress_accuracy.py` from the metrics checkout
  only; the metrics blob hash was asserted at scorer start. Before scoring,
  every stored prediction row was bound to its absolute shard row: stream ids,
  stream lengths, `y_true`, and `input_active` reconstructed from each model's
  window geometry over the shard's contiguity structure matched the sidecars
  bitwise for all ten models. Per-tree eval JSONs are retained under
  `raw_a9a4144/` and `raw_dirty/` as raw execution records; they are not the
  reported numbers.
- **GPU coexistence.** The two takeover trainings (PIDs 263629/263630) ran
  undisturbed throughout: 26,137 MiB per GPU before the pass, 26,137 MiB
  after; the 10-second monitor recorded a peak of 29,051 MiB on GPU 1 (the
  inference GPU) and GPU 0 never left 26,137 MiB. Both training processes
  were alive after the pass.

## Results

All headline numbers are on the input-active event/frame surface with the frozen per-key development thresholds; per-key detail, all-frame variants, calibration, onset timing, oracle-on-test diagnostics, and collar 1/4 sensitivity are in each `<run>_untouched.json`.

### Per-model maximal valid support

Eight 32-frame-window models: 132 contiguous streams, 48,944 predicted rows (43,186 input-active). Two 128x3 models: 53 streams, 16,805 rows (16,094 input-active).

| model | weights | macro AP | state F1 (frozen thr, macro) | event F1 c0 (macro) | event F1 +/-2 (macro) | key-state micro acc | joint exact-match acc |
|---|---|---|---|---|---|---|---|
| own-32nc-s0 | selected | 0.1695 | 0.2692 | 0.0259 | 0.0439 | 0.7350 | 0.1736 |
| own-32nc-s1 | selected | 0.1659 | 0.2682 | 0.0286 | 0.0421 | 0.7424 | 0.2026 |
| own-32nc-s2 | selected | 0.1656 | 0.2680 | 0.0283 | 0.0491 | 0.7095 | 0.1721 |
| tierB-32nc-s0 | selected | 0.2247 | 0.2667 | 0.0338 | 0.0401 | 0.5984 | 0.0343 |
| tierB-32nc-s1 | selected | 0.1943 | 0.2494 | 0.0312 | 0.0354 | 0.5960 | 0.0164 |
| tierB-32nc-s2 | selected | 0.2215 | 0.2495 | 0.0322 | 0.0377 | 0.6570 | 0.0477 |
| tierB-e2e-36.9M | selected | 0.2377 | 0.2662 | 0.0302 | 0.0362 | 0.6465 | 0.1457 |
| tierB-e2e-112.95M | selected | 0.2261 | 0.2636 | 0.0335 | 0.0379 | 0.6059 | 0.0509 |
| full-corpus-103.41h | final | 0.1986 | 0.2117 | 0.0201 | 0.0424 | 0.5923 | 0.1145 |
| full-corpus-148.32h | final | 0.1777 | 0.2435 | 0.0253 | 0.0439 | 0.5689 | 0.1376 |

### Cross-model intersection support (53 streams, 16,805 rows, 16,094 input-active)

The intersection equals the 128x3 support: within every shared stream the 128x3 window coverage is strictly inside the 32-frame coverage, so the two full-corpus rows repeat their maximal-support values by construction.

| model | weights | macro AP | state F1 (frozen thr, macro) | event F1 c0 (macro) | event F1 +/-2 (macro) | key-state micro acc | joint exact-match acc |
|---|---|---|---|---|---|---|---|
| own-32nc-s0 | selected | 0.1715 | 0.2556 | 0.0630 | 0.0810 | 0.7491 | 0.1849 |
| own-32nc-s1 | selected | 0.1628 | 0.2526 | 0.0671 | 0.0797 | 0.7560 | 0.2234 |
| own-32nc-s2 | selected | 0.1524 | 0.2546 | 0.0646 | 0.0842 | 0.7189 | 0.1583 |
| tierB-32nc-s0 | selected | 0.2178 | 0.2599 | 0.0776 | 0.0808 | 0.6068 | 0.0449 |
| tierB-32nc-s1 | selected | 0.1911 | 0.2404 | 0.0738 | 0.0760 | 0.5952 | 0.0095 |
| tierB-32nc-s2 | selected | 0.2198 | 0.2407 | 0.0756 | 0.0786 | 0.6571 | 0.0493 |
| tierB-e2e-36.9M | selected | 0.2235 | 0.2518 | 0.0759 | 0.0802 | 0.6453 | 0.1551 |
| tierB-e2e-112.95M | selected | 0.2117 | 0.2503 | 0.0812 | 0.0823 | 0.6128 | 0.0452 |
| full-corpus-103.41h | final | 0.1986 | 0.2117 | 0.0201 | 0.0424 | 0.5923 | 0.1145 |
| full-corpus-148.32h | final | 0.1777 | 0.2435 | 0.0253 | 0.0439 | 0.5689 | 0.1376 |

### Per-key AP with prevalence (maximal support, input-active)

| model | left | right | up | down | jump | dash | grab |
|---|---|---|---|---|---|---|---|
| prevalence (8x32-window support) | 0.1811 | 0.2257 | 0.1829 | 0.0509 | 0.0986 | 0.1421 | 0.2002 |
| prevalence (128x3 support) | 0.1639 | 0.1967 | 0.1816 | 0.0348 | 0.0900 | 0.1302 | 0.2278 |
| own-32nc-s0 | 0.2227 | 0.2525 | 0.1861 | 0.0698 | 0.1138 | 0.1627 | 0.1785 |
| own-32nc-s1 | 0.2032 | 0.2641 | 0.2093 | 0.0693 | 0.1068 | 0.1510 | 0.1576 |
| own-32nc-s2 | 0.2186 | 0.2462 | 0.1841 | 0.1161 | 0.0897 | 0.1481 | 0.1565 |
| tierB-32nc-s0 | 0.2021 | 0.3161 | 0.2195 | 0.0929 | 0.1552 | 0.1863 | 0.4007 |
| tierB-32nc-s1 | 0.1774 | 0.2457 | 0.2144 | 0.0745 | 0.1161 | 0.1698 | 0.3620 |
| tierB-32nc-s2 | 0.2066 | 0.2994 | 0.2405 | 0.0795 | 0.1089 | 0.2036 | 0.4119 |
| tierB-e2e-36.9M | 0.1993 | 0.3327 | 0.3474 | 0.1023 | 0.2055 | 0.2552 | 0.2215 |
| tierB-e2e-112.95M | 0.1945 | 0.2717 | 0.2987 | 0.0821 | 0.1909 | 0.2285 | 0.3162 |
| full-corpus-103.41h | 0.2419 | 0.3050 | 0.2537 | 0.0570 | 0.1259 | 0.1768 | 0.2299 |
| full-corpus-148.32h | 0.1809 | 0.2736 | 0.2240 | 0.0375 | 0.1318 | 0.1798 | 0.2161 |

### Trivial baselines

Label-only computations on the shard (committed `experiments/baselines.py`, shuffle seed `np.random.default_rng(0)`, 10 shuffles), input-active surface (n=46162):

- Per-key prevalence (chance AP): left 0.1788, right 0.2218, up 0.1781, down 0.0508, jump 0.0952, dash 0.1405, grab 0.1954; macro 0.1515.
- Persistence AP macro 0.9235; persistence event F1 exactly 0 at collar 0 and 0.9987 macro at collar 1 (the autocorrelation shortcut).
- Shuffled-event luck anchor: macro event F1 0.0054 at collar 0, 0.0260 at +/-2.
- Constant-probability chance produces no events: event F1 0 by construction.
- Key-state accuracy baselines on the 32-window maximal support: always-released micro 0.8455 / joint 0.4010; label-persistence micro 0.9891 / joint 0.9310.
- Same baselines on the intersection support: always-released micro 0.8536 / joint 0.4185; persistence micro 0.9888 / joint 0.9317. (Model-row baselines from the committed `experiments/keypress_accuracy.py`, decision rule probability >= 0.5.)

## Checkpoint identities

| run | endpoint | model.pt sha256 |
|---|---|---|
| own_features_32nc_s0 | selected (step 250) | 98a0420f638f7896a492d6994f09fd5814d654d979ac1a0c91b3396f3dbece9d |
| own_features_32nc_s1 | selected (step 250) | 3d805c35348587dc9e25e30f9d754fcee4a997f17676a700c6eeebca70eaedcc |
| own_features_32nc_s2 | selected (step 250) | ea9f976677e3480673e0eee85d0a1390de385ed4ffeb7eb53620ca609b1e3600 |
| foreign_tier_b_13p45h_32nc_s0 | selected (step 1,250) | f4294f31e6f4e84cf5dbbdf0ab0ca836fed931e08bc3b0cd886556749753231f |
| foreign_tier_b_13p45h_32nc_s1 | selected (step 250) | 49eeab2afdc35b129e17dd599dcb5c006141cc87c6719618257054bde3d4d155 |
| foreign_tier_b_13p45h_32nc_s2 | selected (step 1,250) | cf92769e338fc982897a64588f947097551b43307decc4c9bc645e1846044598 |
| foreign_tier_b_13p45h_37m_e2e_aug_32nc_s0 | selected (step 6,750) | 1d780811d020cde4a9c28f31e38cba69706ef0bd08ab31666fde9f214f8119dc |
| foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0 | selected (step 6,000) | 4cfb619be79f0f8626dedc3f07b7c095b1048d4144ee67d3fb3a155cf862748f |
| nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0 | final (fixed, step 14,265) | cf55f612382bfa7b9a1b67038b5223a1629782782995f0b008311ba380b34f94 |
| nitrogen_full_210train_y4n_holdout_26m_128x3_s0 | final (fixed, step 20,458) | 297c6a512914946f9d836467b31afa5b84b74e856ad3fb7b2f7326284161fd09 |

Endpoint identities are the frozen ones in `PREP_RECORD.json` and the
preflight loadability audit; no endpoint was selected or changed in this pass.

## Commands and execution record

- Staging: `rclone copy "$MADELEINE_R2_BUCKET_URI/shards/test-untouched-v1/"
  /ephemeral/data/test_untouched/ --checksum`, then sha256 verification
  against the local sealed copy.
- Inference driver: `run_untouched_pass.sh` (archived beside this report).
  One `badeline.eval` invocation per model on `CUDA_VISIBLE_DEVICES=1`,
  `--sessions val_sessions.txt` (the single test session),
  `--transition-thresholds-from` the model's designated frozen-threshold
  JSON, `--data` the feature directory for the eight frozen-feature models
  and the pixel shard directory for the two end-to-end models. Per-model
  stdout logs are under `logs/`.
- All ten inferences completed rc=0 on first attempt between 04:21:05 and
  04:25:50 UTC; no retry was needed and no model was stopped or struck.
- Uniform scoring: `score_untouched.py` (archived), which asserts the frozen
  metrics blob hash, binds every stored prediction row to its absolute shard
  row, computes both supports, and emits the ten `<run>_untouched.json`
  reports plus `intersection_support.json` and the derived intersection
  sidecars under `derived_intersection/`.
- Summary extraction: `gen_report_tables.py` (archived) producing
  `summary_tables.json`; this report's tables are rendered from that file.

## Artifact inventory

Under `/ephemeral/results/untouched_test/` (mirrored to the workstation):

- `<run>_untouched.json` — the reported per-model results (both supports),
  ten files.
- `<run>_untouched_preds.npz` — the stored predictions of the single
  inference pass (hard links of the raw per-tree sidecars; identical bytes),
  ten files.
- `raw_a9a4144/`, `raw_dirty/` — per-tree eval JSONs and prediction sidecars
  exactly as emitted by each model's assigned code tree (raw execution
  records, not the reported numbers).
- `derived_intersection/` — intersection-support slices of the stored
  predictions (derived, no re-inference).
- `intersection_support.json`, `trivial_baselines.json`,
  `summary_tables.json`, `gpu_monitor_pass.csv`, `logs/`,
  `run_untouched_pass.sh`, `score_untouched.py`, `gen_report_tables.py`,
  `UNTOUCHED_TEST.md`, and `MANIFEST.sha256` covering every file above.

No git commit was made by this pass; the main session verifies and commits.
The sealed session has now been evaluated exactly once; under the one-pass
rule it is spent as an untouched surface and is never re-inferred under
different settings.

Sanitization note: the object-store bucket URI in this file was replaced
with the provider-neutral form after the pass; content is otherwise the
node-produced report, whose unmodified original remains in the node's
results directory under the manifest hash recorded there.
