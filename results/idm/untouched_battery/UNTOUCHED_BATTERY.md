# Untouched engine-truth battery: single authorized sixteen-model pass

Four sealed battery sessions (`test-battery-v1`), scored 2026-07-28 23:06-23:47 UTC
on `madeleine-codex`, one inference pass per model per session, exactly as
pre-registered in `results/idm/UNTOUCHED_BATTERY_PREREGISTRATION.md` with the
frozen protocol inherited from `results/idm/UNTOUCHED_TEST_RESOLUTION.md`
(per-model code trees, threshold pointer, uniform metric layer, one-pass and
retry discipline). Results are reported as found. No git commit was made from
this pass; every artifact lives under `/ephemeral/results/untouched_battery/`
with a full mirror on the workstation.

## Model-set eligibility (checked before any session was scored)

The pre-registration adds the six own-v3 rerun models if and only if, before
the first battery session is scored, (a) all six checkpoint sha256 values are
registered in tracked records and (b) their val-A threshold reports are
committed. Both held: the six hashes appear in `results/idm/OWN_V3_RERUN.md`
and `results/idm/checkpoint_sha256.txt`, and the six committed
`results/idm/own_features_v3_*_val_a.json` reports carry the ruling-3 pointer
(`/input_active_only/metrics/transition_f1_oracle/<key>/threshold`).
**All six were added; the scored set is sixteen models.** No partial
inclusion question arose.

Validation before battery contact (inherited ruling 4): each of the six
checkpoints on the node hash-matches its tracked value, and each reproduces
its committed val-A sidecar **exactly** (bitwise-identical prediction arrays;
zero differing report leaves) under the evaluator the frozen study contract
names (`experiments/configs/own_v3_primary_reruns.json`,
`evaluator_relevant_files`), whose file hashes are byte-identical to the
current-main metrics checkout used in this pass. A first comparison run under
the `a9a4144` pinned tree instead showed probability-level differences
(max |dp| 0.0030 scratch / 0.1009 tier-b-init on val-A, labels and support
bitwise identical): the two evaluator generations differ in feature-delta
semantics. That record is retained as
`v3_repro_pinned_diagnostic/` — it is an evaluator-semantics difference, not
a checkpoint-identity failure; the contract-evaluator reproduction is the
operative validation. Battery inference for these six models runs from the
pinned training-era tree, matching the original `own_features_32nc` flow, as
the pre-registration and the battery instruction fix in advance; their frozen
thresholds come from the committed val-A reports per ruling 3.

Consequence for interpretation, fixed at scoring time: because those frozen
thresholds were fitted under the contract evaluator while battery inference
ran the pinned training-era evaluator, the six own-v3 rows'
threshold-dependent state and event metrics are not like-for-like and are
**protocol-locked diagnostics** — retained unmodified for provenance,
excluded from headline comparisons. Primary battery conclusions rest on the
ten original frozen models, whose threshold/evaluator pairing is internally
consistent. The battery sessions are spent; nothing is re-inferred.

## Integrity chain (all checks passed before any prediction)

- **Sealed shards (seal re-confirmation).** Each of the four session
  directories staged from the durable object store
  (`shards/test-battery-v1/<sid>/`, `rclone copy --checksum`) and the staged
  NPZ sha256 verified equal to the sealing record
  (`untouched_battery_seal/SEAL_RECORD.md`):
  - `rec_20260728_164723_battery_ch1` staged and hash-verified (SEAL-OK).
  - `rec_20260728_173748_battery_ch2` staged and hash-verified (SEAL-OK).
  - `rec_20260728_174924_battery_ch3` staged and hash-verified (SEAL-OK).
  - `rec_20260728_172310_battery_ch4` staged and hash-verified (SEAL-OK).
- **Checkpoints.** All sixteen `/ephemeral/results/takeover/<run>/model.pt`
  sha256 values re-verified against the frozen PREP_RECORD values (ten) and
  the tracked own-v3 registry values (six); recorded in
  `BATTERY_PREP_RECORD.json`. No model was struck; none required retry; all
  64 session-model inferences returned rc=0 on first attempt.
- **Metrics checkout.** `/ephemeral/madeleine-metrics` refreshed to an
  archive of workstation `main` HEAD `4a8efee22de776b19d8533133747b8c8dd9cb8b6`
  and verified blob-by-blob against `git ls-tree -r HEAD` (9,952 files, zero
  mismatches). `badeline/metrics.py` blob sha256 equals the frozen
  `ce3947b199a5e98dcc0af6de2e3eefe8c6967ecb07c5b636319950a67d301434`;
  asserted again at scorer start, and the imported module path resolved to
  this checkout.
- **Frozen thresholds.** For every model the
  `/input_active_only/metrics/transition_f1_oracle/<key>/threshold` vector of
  its designated committed val-A development report, read by
  `--transition-thresholds-from` at inference and re-read independently by the
  uniform scorer. All sixteen source files hash-verified against their
  committed values before the pass (`BATTERY_PREP_RECORD.json`).
- **Battery features.** One extraction per session with the committed
  converter of the metrics checkout (`data.precompute_features shards`,
  batch 512), byte-identical converter to the corrected own-v3 generation
  commit `f8a51d7` (`data/precompute_features.py` sha256 `be89d537...`,
  `data/schema.py` `3712ec20...`) — the corrected-geometry pixels live in the
  sealed shards themselves, so this reproduces the corrected own-v3 flow on
  the battery sessions. Keys, engine indices, activity flags, and session id
  verified bitwise identical to each pixel shard; features (N,512) float16
  (`feature_verification.json`).
- **Per-model code trees (inherited ruling 4).** Twelve frozen-feature 32nc
  models (six legacy, six own-v3) under `/ephemeral/madeleine-pinned` @
  `a9a414452c07fb101d26faee9fa13864dde922b4`; two end-to-end and two
  full-corpus models under the preserved read-only dirty tree
  `/ephemeral/madeleine` @ `d13ef3e` (porcelain-status sha256 re-confirmed
  equal to PREP_RECORD: `3e4d1a33...`), with `--weights selected` (e2e) and
  `--weights final` (full-corpus, fixed endpoints 14,265 / 20,458).
  Interpreter: the node training venv (Python 3.12.13, torch 2.13.0+cu129,
  cuDNN 92000, A100 80GB PCIe, driver 565.57.01).
- **Uniform metric layer.** Every reported number below was computed from the
  64 stored prediction sidecars by `score_battery.py` (archived under
  `scripts/`) importing `badeline/metrics.py`, `badeline/train.py` helpers,
  and the committed `experiments/keypress_accuracy.py` from the metrics
  checkout only. Before scoring, every stored prediction row was bound to its
  absolute shard row: stream ids, stream lengths, `y_true`, and
  `input_active` reconstructed from each model's window geometry over each
  shard's contiguity structure matched the sidecars bitwise for all 64
  session-model cells. Per-tree eval JSONs are retained under
  `raw_a9a4144/<sid>/` and `raw_dirty/<sid>/` as raw execution records; they
  are not the reported numbers.
- **GPU coexistence.** Both A100s were idle before the pass (no compute
  processes; the takeover trainings had completed). All battery inference ran
  sequentially on GPU 1; the 10-second monitor recorded a peak of 3,323 MiB
  on GPU 1 and GPU 0 never left 1 MiB (`gpu_monitor_pass.csv`).

## Support surfaces

Pooled (four sessions, registered order ch1,ch2,ch3,ch4): fourteen
32-frame-window models predict 309 contiguous streams,
122,128 rows (109,950 input-active). The two 128x3
models predict 184 streams, 35,697 rows
(33,454 input-active). Within every shared stream the 128x3
coverage is strictly inside the 32-frame coverage, so the cross-model
intersection equals the 128x3 support (184 streams,
35,697 rows, 33,454 input-active) and the two
full-corpus rows repeat their maximal-support values by construction.
Per-session supports:

| session | chapter | shard rows | input-active | 32w streams/rows/active | intersection streams/rows/active |
|---|---|---:|---:|---|---|
| `rec_20260728_164723_battery_ch1` | ch1 | 24,400 | 21,321 | 63/22,384/19,682 | 34/6,250/5,540 |
| `rec_20260728_173748_battery_ch2` | ch2 | 35,905 | 31,728 | 78/33,383/29,703 | 51/9,985/9,118 |
| `rec_20260728_174924_battery_ch3` | ch3 | 35,911 | 32,447 | 82/33,317/30,382 | 48/10,407/10,123 |
| `rec_20260728_172310_battery_ch4` | ch4 | 35,870 | 32,360 | 86/33,044/30,183 | 51/9,055/8,673 |

## Pooled battery aggregate (primary result)

All headline numbers are on the input-active surface with the frozen per-key
development thresholds; per-key detail, all-frame variants, calibration,
onset timing, and collar sensitivity are in each per-model JSON.

### Per-model maximal valid support (pooled)

| model | weights | macro AP | state F1 (frozen thr, macro) | event F1 c0 (macro) | event F1 +/-2 (macro) | key-state micro acc | joint exact-match acc |
|---|---|---|---|---|---|---|---|
| own-32nc-s0 | selected | 0.1827 | 0.2698 | 0.0359 | 0.0455 | 0.7171 | 0.1206 |
| own-32nc-s1 | selected | 0.1749 | 0.2697 | 0.0405 | 0.0479 | 0.7406 | 0.1576 |
| own-32nc-s2 | selected | 0.1861 | 0.2712 | 0.0397 | 0.0482 | 0.6686 | 0.0726 |
| own-v3-32nc-s0 | selected | 0.1873 | 0.2691 | 0.0390 | 0.0507 | 0.7271 | 0.1360 |
| own-v3-32nc-s1 | selected | 0.1773 | 0.2702 | 0.0392 | 0.0480 | 0.7293 | 0.1412 |
| own-v3-32nc-s2 | selected | 0.1899 | 0.2669 | 0.0392 | 0.0471 | 0.6565 | 0.0729 |
| own-v3-tierBinit-s0 | selected | 0.2138 | 0.2676 | 0.0484 | 0.0523 | 0.5560 | 0.0619 |
| own-v3-tierBinit-s1 | selected | 0.1930 | 0.2717 | 0.0446 | 0.0509 | 0.6130 | 0.0800 |
| own-v3-tierBinit-s2 | selected | 0.2169 | 0.2706 | 0.0458 | 0.0526 | 0.5357 | 0.0575 |
| tierB-32nc-s0 | selected | 0.2150 | 0.2677 | 0.0459 | 0.0496 | 0.5311 | 0.0430 |
| tierB-32nc-s1 | selected | 0.1875 | 0.2638 | 0.0424 | 0.0488 | 0.5634 | 0.0330 |
| tierB-32nc-s2 | selected | 0.2221 | 0.2646 | 0.0452 | 0.0491 | 0.5452 | 0.0381 |
| tierB-e2e-36.9M | selected | 0.2667 | 0.2690 | 0.0451 | 0.0478 | 0.5706 | 0.0918 |
| tierB-e2e-112.95M | selected | 0.2516 | 0.2684 | 0.0479 | 0.0511 | 0.5169 | 0.0462 |
| full-corpus-103.41h | final | 0.2262 | 0.2725 | 0.0308 | 0.0528 | 0.5331 | 0.0464 |
| full-corpus-148.32h | final | 0.2183 | 0.2813 | 0.0368 | 0.0532 | 0.5408 | 0.0669 |

### Cross-model intersection support (pooled)

| model | weights | macro AP | state F1 (frozen thr, macro) | event F1 c0 (macro) | event F1 +/-2 (macro) | key-state micro acc | joint exact-match acc |
|---|---|---|---|---|---|---|---|
| own-32nc-s0 | selected | 0.1853 | 0.2730 | 0.0791 | 0.0877 | 0.7169 | 0.1203 |
| own-32nc-s1 | selected | 0.1813 | 0.2730 | 0.0864 | 0.0959 | 0.7396 | 0.1618 |
| own-32nc-s2 | selected | 0.1908 | 0.2766 | 0.0877 | 0.1005 | 0.6662 | 0.0705 |
| own-v3-32nc-s0 | selected | 0.1904 | 0.2730 | 0.0836 | 0.0972 | 0.7261 | 0.1332 |
| own-v3-32nc-s1 | selected | 0.1855 | 0.2739 | 0.0850 | 0.0957 | 0.7270 | 0.1426 |
| own-v3-32nc-s2 | selected | 0.1959 | 0.2710 | 0.0850 | 0.0938 | 0.6533 | 0.0764 |
| own-v3-tierBinit-s0 | selected | 0.2099 | 0.2697 | 0.0967 | 0.1054 | 0.5402 | 0.0584 |
| own-v3-tierBinit-s1 | selected | 0.1946 | 0.2755 | 0.0943 | 0.1056 | 0.6063 | 0.0816 |
| own-v3-tierBinit-s2 | selected | 0.2125 | 0.2736 | 0.0939 | 0.1021 | 0.5269 | 0.0556 |
| tierB-32nc-s0 | selected | 0.2153 | 0.2676 | 0.0941 | 0.1023 | 0.5117 | 0.0358 |
| tierB-32nc-s1 | selected | 0.1863 | 0.2635 | 0.0906 | 0.1011 | 0.5533 | 0.0280 |
| tierB-32nc-s2 | selected | 0.2174 | 0.2635 | 0.0955 | 0.1035 | 0.5327 | 0.0354 |
| tierB-e2e-36.9M | selected | 0.2617 | 0.2709 | 0.0969 | 0.1041 | 0.5538 | 0.0867 |
| tierB-e2e-112.95M | selected | 0.2412 | 0.2707 | 0.0986 | 0.1053 | 0.4991 | 0.0327 |
| full-corpus-103.41h | final | 0.2262 | 0.2725 | 0.0308 | 0.0528 | 0.5331 | 0.0464 |
| full-corpus-148.32h | final | 0.2183 | 0.2813 | 0.0368 | 0.0532 | 0.5408 | 0.0669 |

### Per-key AP with prevalence (pooled, maximal support, input-active)

| model | left | right | up | down | jump | dash | grab |
|---|---|---|---|---|---|---|---|
| prevalence (14x32-window support) | 0.1510 | 0.3300 | 0.1543 | 0.0186 | 0.1410 | 0.0817 | 0.2588 |
| prevalence (128x3 support) | 0.1415 | 0.2985 | 0.1556 | 0.0276 | 0.1372 | 0.0830 | 0.3029 |
| own-32nc-s0 | 0.1818 | 0.3668 | 0.1963 | 0.0176 | 0.1422 | 0.0853 | 0.2892 |
| own-32nc-s1 | 0.1734 | 0.3431 | 0.1615 | 0.0190 | 0.1385 | 0.0802 | 0.3089 |
| own-32nc-s2 | 0.1849 | 0.3482 | 0.1977 | 0.0222 | 0.1487 | 0.0889 | 0.3123 |
| own-v3-32nc-s0 | 0.1873 | 0.3789 | 0.1925 | 0.0174 | 0.1427 | 0.0909 | 0.3012 |
| own-v3-32nc-s1 | 0.1733 | 0.3433 | 0.1675 | 0.0200 | 0.1395 | 0.0825 | 0.3149 |
| own-v3-32nc-s2 | 0.1898 | 0.3566 | 0.2006 | 0.0219 | 0.1537 | 0.0910 | 0.3159 |
| own-v3-tierBinit-s0 | 0.1882 | 0.4310 | 0.2491 | 0.0186 | 0.1348 | 0.1019 | 0.3727 |
| own-v3-tierBinit-s1 | 0.1713 | 0.3664 | 0.2077 | 0.0331 | 0.1459 | 0.0902 | 0.3362 |
| own-v3-tierBinit-s2 | 0.1764 | 0.4435 | 0.2359 | 0.0303 | 0.1553 | 0.1112 | 0.3656 |
| tierB-32nc-s0 | 0.1902 | 0.4630 | 0.2440 | 0.0179 | 0.1457 | 0.1076 | 0.3368 |
| tierB-32nc-s1 | 0.1621 | 0.3807 | 0.1986 | 0.0264 | 0.1389 | 0.0939 | 0.3116 |
| tierB-32nc-s2 | 0.1803 | 0.4580 | 0.2335 | 0.0167 | 0.1584 | 0.1116 | 0.3959 |
| tierB-e2e-36.9M | 0.2188 | 0.4846 | 0.2732 | 0.0280 | 0.2187 | 0.2522 | 0.3913 |
| tierB-e2e-112.95M | 0.2229 | 0.4952 | 0.2390 | 0.0297 | 0.1925 | 0.1766 | 0.4052 |
| full-corpus-103.41h | 0.2269 | 0.4401 | 0.2204 | 0.0354 | 0.1696 | 0.1068 | 0.3841 |
| full-corpus-148.32h | 0.2223 | 0.3898 | 0.2250 | 0.0454 | 0.1583 | 0.1161 | 0.3710 |

## Content-familiarity gradient (per-chapter rows)

Chapter 1 is the pre-registered anchor (content family overlaps the own-data
training rooms; isolates recording transfer). Chapters 2-4 are content the
local rig never recorded but the mapped corpus covers. The committed
Chapter 6 single-session result (`untouched_test/UNTOUCHED_TEST.md`,
n_active=43,186 for 32-window models / 16,094 for 128x3) is shown for
context; the ten legacy models' ch6 numbers are the committed ones, and the
six own-v3 models were not part of that pass (--).

### Macro AP (maximal support, input-active)

| model | ch1 (anchor) | ch2 | ch3 | ch4 | pooled | ch6 (committed) |
|---|---|---|---|---|---|---|
| own-32nc-s0 | 0.1701 | 0.1838 | 0.1593 | 0.1946 | 0.1827 | 0.1695 |
| own-32nc-s1 | 0.1762 | 0.1798 | 0.1612 | 0.1868 | 0.1749 | 0.1659 |
| own-32nc-s2 | 0.1679 | 0.1939 | 0.1677 | 0.1892 | 0.1861 | 0.1656 |
| own-v3-32nc-s0 | 0.1776 | 0.1840 | 0.1649 | 0.2023 | 0.1873 | -- |
| own-v3-32nc-s1 | 0.1759 | 0.1792 | 0.1603 | 0.1889 | 0.1773 | -- |
| own-v3-32nc-s2 | 0.1691 | 0.1964 | 0.1738 | 0.1954 | 0.1899 | -- |
| own-v3-tierBinit-s0 | 0.1749 | 0.1844 | 0.1908 | 0.2573 | 0.2138 | -- |
| own-v3-tierBinit-s1 | 0.1750 | 0.1896 | 0.1769 | 0.2148 | 0.1930 | -- |
| own-v3-tierBinit-s2 | 0.1501 | 0.1874 | 0.1953 | 0.2687 | 0.2169 | -- |
| tierB-32nc-s0 | 0.1810 | 0.1924 | 0.1847 | 0.2843 | 0.2150 | 0.2247 |
| tierB-32nc-s1 | 0.1725 | 0.1758 | 0.1545 | 0.2314 | 0.1875 | 0.1943 |
| tierB-32nc-s2 | 0.1709 | 0.1920 | 0.1917 | 0.2869 | 0.2221 | 0.2215 |
| tierB-e2e-36.9M | 0.2312 | 0.2753 | 0.2566 | 0.2882 | 0.2667 | 0.2377 |
| tierB-e2e-112.95M | 0.2283 | 0.2573 | 0.2203 | 0.2718 | 0.2516 | 0.2261 |
| full-corpus-103.41h | 0.2219 | 0.2225 | 0.1827 | 0.2688 | 0.2262 | 0.1986 |
| full-corpus-148.32h | 0.2128 | 0.2058 | 0.2118 | 0.2472 | 0.2183 | 0.1777 |

### Event F1 collar 0 (maximal support, input-active, frozen thresholds)

| model | ch1 (anchor) | ch2 | ch3 | ch4 | pooled | ch6 (committed) |
|---|---|---|---|---|---|---|
| own-32nc-s0 | 0.0361 | 0.0382 | 0.0296 | 0.0547 | 0.0359 | 0.0259 |
| own-32nc-s1 | 0.0420 | 0.0446 | 0.0310 | 0.0538 | 0.0405 | 0.0286 |
| own-32nc-s2 | 0.0370 | 0.0415 | 0.0352 | 0.0532 | 0.0397 | 0.0283 |
| own-v3-32nc-s0 | 0.0395 | 0.0377 | 0.0344 | 0.0533 | 0.0390 | -- |
| own-v3-32nc-s1 | 0.0378 | 0.0423 | 0.0304 | 0.0548 | 0.0392 | -- |
| own-v3-32nc-s2 | 0.0403 | 0.0426 | 0.0369 | 0.0523 | 0.0392 | -- |
| own-v3-tierBinit-s0 | 0.0453 | 0.0486 | 0.0406 | 0.0597 | 0.0484 | -- |
| own-v3-tierBinit-s1 | 0.0425 | 0.0447 | 0.0395 | 0.0568 | 0.0446 | -- |
| own-v3-tierBinit-s2 | 0.0447 | 0.0487 | 0.0398 | 0.0556 | 0.0458 | -- |
| tierB-32nc-s0 | 0.0426 | 0.0494 | 0.0369 | 0.0559 | 0.0459 | 0.0338 |
| tierB-32nc-s1 | 0.0379 | 0.0457 | 0.0315 | 0.0572 | 0.0424 | 0.0312 |
| tierB-32nc-s2 | 0.0384 | 0.0550 | 0.0328 | 0.0576 | 0.0452 | 0.0322 |
| tierB-e2e-36.9M | 0.0415 | 0.0456 | 0.0377 | 0.0558 | 0.0451 | 0.0302 |
| tierB-e2e-112.95M | 0.0427 | 0.0517 | 0.0397 | 0.0560 | 0.0479 | 0.0335 |
| full-corpus-103.41h | 0.0223 | 0.0292 | 0.0338 | 0.0378 | 0.0308 | 0.0201 |
| full-corpus-148.32h | 0.0224 | 0.0303 | 0.0555 | 0.0465 | 0.0368 | 0.0253 |

### State F1 at frozen thresholds (maximal support, input-active, macro)

| model | ch1 (anchor) | ch2 | ch3 | ch4 | pooled | ch6 (committed) |
|---|---|---|---|---|---|---|
| own-32nc-s0 | 0.2598 | 0.2686 | 0.2373 | 0.2902 | 0.2698 | 0.2692 |
| own-32nc-s1 | 0.2594 | 0.2708 | 0.2364 | 0.2921 | 0.2697 | 0.2682 |
| own-32nc-s2 | 0.2596 | 0.2612 | 0.2439 | 0.2935 | 0.2712 | 0.2680 |
| own-v3-32nc-s0 | 0.2578 | 0.2682 | 0.2360 | 0.2865 | 0.2691 | -- |
| own-v3-32nc-s1 | 0.2632 | 0.2705 | 0.2374 | 0.2908 | 0.2702 | -- |
| own-v3-32nc-s2 | 0.2609 | 0.2642 | 0.2363 | 0.2913 | 0.2669 | -- |
| own-v3-tierBinit-s0 | 0.2620 | 0.2677 | 0.2397 | 0.2944 | 0.2676 | -- |
| own-v3-tierBinit-s1 | 0.2584 | 0.2674 | 0.2444 | 0.2964 | 0.2717 | -- |
| own-v3-tierBinit-s2 | 0.2625 | 0.2705 | 0.2390 | 0.2991 | 0.2706 | -- |
| tierB-32nc-s0 | 0.2661 | 0.2661 | 0.2324 | 0.2935 | 0.2677 | 0.2667 |
| tierB-32nc-s1 | 0.2614 | 0.2626 | 0.2326 | 0.2923 | 0.2638 | 0.2494 |
| tierB-32nc-s2 | 0.2615 | 0.2642 | 0.2333 | 0.2923 | 0.2646 | 0.2495 |
| tierB-e2e-36.9M | 0.2624 | 0.2733 | 0.2383 | 0.2946 | 0.2690 | 0.2662 |
| tierB-e2e-112.95M | 0.2606 | 0.2724 | 0.2378 | 0.2949 | 0.2684 | 0.2636 |
| full-corpus-103.41h | 0.2571 | 0.2542 | 0.2578 | 0.2639 | 0.2725 | 0.2117 |
| full-corpus-148.32h | 0.2666 | 0.2820 | 0.2546 | 0.2672 | 0.2813 | 0.2435 |

### Key-state micro accuracy / joint exact-match (maximal support, input-active)

| model | ch1 | ch2 | ch3 | ch4 | pooled |
|---|---|---|---|---|---|
| own-32nc-s0 | 0.7526/0.1376 | 0.7398/0.1657 | 0.7008/0.0901 | 0.6880/0.0958 | 0.7171/0.1206 |
| own-32nc-s1 | 0.7638/0.2081 | 0.7254/0.1472 | 0.6898/0.0871 | 0.7916/0.2060 | 0.7406/0.1576 |
| own-32nc-s2 | 0.7233/0.1313 | 0.6861/0.1083 | 0.6372/0.0392 | 0.6474/0.0329 | 0.6686/0.0726 |
| own-v3-32nc-s0 | 0.7489/0.1399 | 0.7421/0.1733 | 0.7065/0.1077 | 0.7189/0.1250 | 0.7271/0.1360 |
| own-v3-32nc-s1 | 0.7441/0.1922 | 0.7253/0.1416 | 0.6784/0.0728 | 0.7748/0.1763 | 0.7293/0.1412 |
| own-v3-32nc-s2 | 0.7163/0.1368 | 0.6769/0.1055 | 0.6133/0.0317 | 0.6409/0.0406 | 0.6565/0.0729 |
| own-v3-tierBinit-s0 | 0.6526/0.1057 | 0.6153/0.1146 | 0.4969/0.0230 | 0.4942/0.0207 | 0.5560/0.0619 |
| own-v3-tierBinit-s1 | 0.7066/0.1246 | 0.7064/0.1589 | 0.5384/0.0289 | 0.5350/0.0246 | 0.6130/0.0800 |
| own-v3-tierBinit-s2 | 0.6139/0.1139 | 0.6302/0.1009 | 0.4702/0.0249 | 0.4578/0.0108 | 0.5357/0.0575 |
| tierB-32nc-s0 | 0.5625/0.0621 | 0.5609/0.0668 | 0.4657/0.0220 | 0.5473/0.0282 | 0.5311/0.0430 |
| tierB-32nc-s1 | 0.5989/0.0422 | 0.5974/0.0291 | 0.5631/0.0524 | 0.5073/0.0113 | 0.5634/0.0330 |
| tierB-32nc-s2 | 0.5603/0.0652 | 0.5595/0.0342 | 0.6184/0.0531 | 0.4476/0.0091 | 0.5452/0.0381 |
| tierB-e2e-36.9M | 0.6086/0.1350 | 0.6117/0.1326 | 0.6087/0.0978 | 0.4671/0.0176 | 0.5706/0.0918 |
| tierB-e2e-112.95M | 0.5811/0.0529 | 0.5365/0.0390 | 0.5663/0.0787 | 0.4061/0.0161 | 0.5169/0.0462 |
| full-corpus-103.41h | 0.5534/0.0478 | 0.4905/0.0285 | 0.4962/0.0375 | 0.6080/0.0747 | 0.5331/0.0464 |
| full-corpus-148.32h | 0.5460/0.1000 | 0.5205/0.0545 | 0.5054/0.0440 | 0.6001/0.0856 | 0.5408/0.0669 |

## Per-chapter rare-key event rows (descriptive only; no headline)

Per-key event F1 at collar 0 (maximal support, frozen thresholds) for the
three lowest-prevalence keys. Values are descriptive per the
pre-registration.

| model | chapter | down | dash | grab |
|---|---|---|---|---|
| own-32nc-s0 | ch1 | 0.0000 | 0.0213 | 0.0270 |
| own-32nc-s0 | ch2 | 0.0000 | 0.0065 | 0.0794 |
| own-32nc-s0 | ch3 | 0.0000 | 0.0166 | 0.0615 |
| own-32nc-s0 | ch4 | 0.0000 | 0.0254 | 0.1600 |
| own-32nc-s1 | ch1 | 0.0000 | 0.0350 | 0.0476 |
| own-32nc-s1 | ch2 | 0.0016 | 0.0083 | 0.0914 |
| own-32nc-s1 | ch3 | 0.0000 | 0.0185 | 0.0759 |
| own-32nc-s1 | ch4 | 0.0000 | 0.0235 | 0.1600 |
| own-32nc-s2 | ch1 | 0.0000 | 0.0127 | 0.0450 |
| own-32nc-s2 | ch2 | 0.0000 | 0.0076 | 0.0691 |
| own-32nc-s2 | ch3 | 0.0076 | 0.0114 | 0.0934 |
| own-32nc-s2 | ch4 | 0.0000 | 0.0167 | 0.1558 |
| own-v3-32nc-s0 | ch1 | 0.0000 | 0.0113 | 0.0513 |
| own-v3-32nc-s0 | ch2 | 0.0000 | 0.0085 | 0.0857 |
| own-v3-32nc-s0 | ch3 | 0.0000 | 0.0136 | 0.0909 |
| own-v3-32nc-s0 | ch4 | 0.0000 | 0.0245 | 0.1600 |
| own-v3-32nc-s1 | ch1 | 0.0000 | 0.0193 | 0.0498 |
| own-v3-32nc-s1 | ch2 | 0.0022 | 0.0084 | 0.0842 |
| own-v3-32nc-s1 | ch3 | 0.0000 | 0.0174 | 0.0757 |
| own-v3-32nc-s1 | ch4 | 0.0000 | 0.0259 | 0.1586 |
| own-v3-32nc-s2 | ch1 | 0.0000 | 0.0137 | 0.0680 |
| own-v3-32nc-s2 | ch2 | 0.0000 | 0.0074 | 0.0670 |
| own-v3-32nc-s2 | ch3 | 0.0000 | 0.0122 | 0.0930 |
| own-v3-32nc-s2 | ch4 | 0.0000 | 0.0130 | 0.1545 |
| own-v3-tierBinit-s0 | ch1 | 0.0000 | 0.0172 | 0.0773 |
| own-v3-tierBinit-s0 | ch2 | 0.0017 | 0.0086 | 0.0909 |
| own-v3-tierBinit-s0 | ch3 | 0.0021 | 0.0208 | 0.0923 |
| own-v3-tierBinit-s0 | ch4 | 0.0000 | 0.0260 | 0.1600 |
| own-v3-tierBinit-s1 | ch1 | 0.0000 | 0.0172 | 0.0603 |
| own-v3-tierBinit-s1 | ch2 | 0.0096 | 0.0086 | 0.0767 |
| own-v3-tierBinit-s1 | ch3 | 0.0000 | 0.0208 | 0.0916 |
| own-v3-tierBinit-s1 | ch4 | 0.0000 | 0.0260 | 0.1600 |
| own-v3-tierBinit-s2 | ch1 | 0.0000 | 0.0172 | 0.0892 |
| own-v3-tierBinit-s2 | ch2 | 0.0098 | 0.0081 | 0.1132 |
| own-v3-tierBinit-s2 | ch3 | 0.0000 | 0.0208 | 0.0972 |
| own-v3-tierBinit-s2 | ch4 | 0.0000 | 0.0260 | 0.1600 |
| tierB-32nc-s0 | ch1 | 0.0000 | 0.0172 | 0.0773 |
| tierB-32nc-s0 | ch2 | 0.0000 | 0.0085 | 0.1069 |
| tierB-32nc-s0 | ch3 | 0.0000 | 0.0208 | 0.0902 |
| tierB-32nc-s0 | ch4 | 0.0000 | 0.0260 | 0.1513 |
| tierB-32nc-s1 | ch1 | 0.0000 | 0.0172 | 0.0690 |
| tierB-32nc-s1 | ch2 | 0.0000 | 0.0086 | 0.0959 |
| tierB-32nc-s1 | ch3 | 0.0000 | 0.0208 | 0.0615 |
| tierB-32nc-s1 | ch4 | 0.0000 | 0.0260 | 0.1338 |
| tierB-32nc-s2 | ch1 | 0.0000 | 0.0163 | 0.0654 |
| tierB-32nc-s2 | ch2 | 0.0078 | 0.0085 | 0.1364 |
| tierB-32nc-s2 | ch3 | 0.0000 | 0.0208 | 0.0650 |
| tierB-32nc-s2 | ch4 | 0.0000 | 0.0260 | 0.1552 |
| tierB-e2e-36.9M | ch1 | 0.0000 | 0.0161 | 0.0700 |
| tierB-e2e-36.9M | ch2 | 0.0025 | 0.0080 | 0.0877 |
| tierB-e2e-36.9M | ch3 | 0.0000 | 0.0200 | 0.0805 |
| tierB-e2e-36.9M | ch4 | 0.0000 | 0.0260 | 0.1434 |
| tierB-e2e-112.95M | ch1 | 0.0000 | 0.0169 | 0.0733 |
| tierB-e2e-112.95M | ch2 | 0.0109 | 0.0084 | 0.1075 |
| tierB-e2e-112.95M | ch3 | 0.0000 | 0.0208 | 0.0941 |
| tierB-e2e-112.95M | ch4 | 0.0000 | 0.0260 | 0.1395 |
| full-corpus-103.41h | ch1 | 0.0114 | 0.0000 | 0.0449 |
| full-corpus-103.41h | ch2 | 0.0000 | 0.0148 | 0.0526 |
| full-corpus-103.41h | ch3 | 0.0132 | 0.0171 | 0.0423 |
| full-corpus-103.41h | ch4 | 0.0000 | 0.0100 | 0.0872 |
| full-corpus-148.32h | ch1 | 0.0000 | 0.0182 | 0.0520 |
| full-corpus-148.32h | ch2 | 0.0000 | 0.0360 | 0.0759 |
| full-corpus-148.32h | ch3 | 0.0162 | 0.0755 | 0.1181 |
| full-corpus-148.32h | ch4 | 0.0000 | 0.0137 | 0.1200 |

## Trivial baselines (label-only, per session)

Committed `experiments/baselines.py`, shuffle rng `np.random.default_rng(0)`,
10 shuffles, input-active surface:

| session | n active | chance AP macro | persistence AP macro | persistence event F1 c0 / c1 | shuffled event F1 c0 / c2 |
|---|---:|---|---|---|---|
| ch1 | 21,321 | 0.1554 | 0.9043 | 0.0000 / 0.9984 | 0.0052 / 0.0277 |
| ch2 | 31,728 | 0.1624 | 0.9194 | 0.0000 / 0.9984 | 0.0049 / 0.0227 |
| ch3 | 32,447 | 0.1395 | 0.9225 | 0.0000 / 0.9989 | 0.0054 / 0.0233 |
| ch4 | 32,360 | 0.1844 | 0.9348 | 0.0000 / 0.9985 | 0.0056 / 0.0241 |

Constant-probability chance produces no events (event F1 0 by construction).
Key-state accuracy baselines from the committed
`experiments/keypress_accuracy.py` (decision rule probability >= 0.5), pooled
32-window maximal support: always-released micro 0.8378 / joint 0.3641; label-persistence micro 0.9897 / joint 0.9336.
Same baselines on the pooled intersection support: always-released micro 0.8362 / joint 0.3644; persistence micro 0.9889 / joint 0.9292.
Per-session values are embedded in each per-model session report.

## Checkpoint identities

| run | endpoint | model.pt sha256 |
|---|---|---|
| own_features_32nc_s0 | selected (step 250) | 98a0420f638f7896a492d6994f09fd5814d654d979ac1a0c91b3396f3dbece9d |
| own_features_32nc_s1 | selected (step 250) | 3d805c35348587dc9e25e30f9d754fcee4a997f17676a700c6eeebca70eaedcc |
| own_features_32nc_s2 | selected (step 250) | ea9f976677e3480673e0eee85d0a1390de385ed4ffeb7eb53620ca609b1e3600 |
| own_features_v3_32nc_s0 | selected (step 250) | 5b2132f228664d2e0de4da2e15961b7d735ff7146dafb7338d8d9123cab8d6a3 |
| own_features_v3_32nc_s1 | selected (step 250) | 05f80e665eb87c84adbc22628ef95db88f9263661b07ef7ca4374d3713bfaf72 |
| own_features_v3_32nc_s2 | selected (step 250) | 525ce4423d15cd7eff23d0c3088318d6f546e363afc289a2cf1f0460ffb98ce5 |
| own_features_v3_tier_b_init_32nc_s0 | selected (step 200) | 4c7308616fc9a5c64489ec872060c27ec726abe74a2a013d25f0d421b602e181 |
| own_features_v3_tier_b_init_32nc_s1 | selected (step 450) | 35cd8efe9d420d82bbd5619bdb8e617ef7b9c009961c8e25bbd3594ffd583605 |
| own_features_v3_tier_b_init_32nc_s2 | selected (step 400) | 8fcf4e5cd2b813855aa67822dc865d5b5b3924eae587f25b0529027a36ebf13b |
| foreign_tier_b_13p45h_32nc_s0 | selected (step 1,250) | f4294f31e6f4e84cf5dbbdf0ab0ca836fed931e08bc3b0cd886556749753231f |
| foreign_tier_b_13p45h_32nc_s1 | selected (step 250) | 49eeab2afdc35b129e17dd599dcb5c006141cc87c6719618257054bde3d4d155 |
| foreign_tier_b_13p45h_32nc_s2 | selected (step 1,250) | cf92769e338fc982897a64588f947097551b43307decc4c9bc645e1846044598 |
| foreign_tier_b_13p45h_37m_e2e_aug_32nc_s0 | selected (step 6,750) | 1d780811d020cde4a9c28f31e38cba69706ef0bd08ab31666fde9f214f8119dc |
| foreign_tier_b_13p45h_113m_e2e_aug_32nc_s0 | selected (step 6,000) | 4cfb619be79f0f8626dedc3f07b7c095b1048d4144ee67d3fb3a155cf862748f |
| nitrogen_unflagged_92train_y4n_holdout_26m_128x3_s0 | final (fixed, step 14,265) | cf55f612382bfa7b9a1b67038b5223a1629782782995f0b008311ba380b34f94 |
| nitrogen_full_210train_y4n_holdout_26m_128x3_s0 | final (fixed, step 20,458) | 297c6a512914946f9d836467b31afa5b84b74e856ad3fb7b2f7326284161fd09 |

Endpoint identities are the frozen ones; no endpoint was selected or changed
in this pass.

## Commands and execution record

- Staging: `rclone copy "r2:<bucket>/shards/test-battery-v1/<sid>/"
  /ephemeral/data/battery/<sid>/ --checksum` then sha256 verification against
  the sealing record (`scripts/stage_battery.sh`; all four SEAL-OK,
  staging done 23:01:02Z).
- own-v3 validation: `scripts/repro_v3_val_a.sh` variant under the contract
  evaluator plus `scripts/compare_v3_repro.py`; verdicts in
  `v3_repro/V3_REPRO_VERDICTS.json` (six exact), pinned-tree diagnostic in
  `v3_repro_pinned_diagnostic/`.
- Features: `scripts/build_battery_features.sh` (per-session
  `python -m data.precompute_features shards --inputs <sid>.npz --out
  /ephemeral/data/battery_features/<sid> --device cuda --batch-size 512`),
  verified by `scripts/check_battery_features.py`.
- Inference driver: `scripts/run_battery_pass.sh`. One `badeline.eval`
  invocation per model per session on `CUDA_VISIBLE_DEVICES=1`,
  `--sessions <sid>/val_sessions.txt`, `--transition-thresholds-from` the
  model's designated frozen-threshold JSON, `--data` the session feature
  directory for the fourteen frozen-feature models and the session pixel
  directory for the two end-to-end models. All 64 inferences completed rc=0
  on first attempt between 23:06:35 and 23:20:29 UTC; no retry was needed and
  no model was stopped or struck. Per-cell stdout logs are under `logs/`.
- Uniform scoring: `scripts/score_battery.py` (asserts the frozen metrics
  blob hash, binds every stored prediction row to its absolute shard row,
  computes per-session maximal and cross-model intersection supports, then
  the pooled battery aggregate from the same stored predictions; emits the 64
  per-session reports, four `intersection_support.json`, sixteen pooled
  reports, and derived sidecars under `sessions/<sid>/derived_intersection/`
  and `pooled/derived/`).
- Baselines: `python -m experiments.baselines --shards <four pixel shards>
  --out trivial_baselines.json`.
- Summary extraction: `scripts/gen_battery_tables.py` producing
  `summary_tables.json`; this report's tables are rendered from that file by
  `scripts/gen_battery_report.py`.

## Artifact inventory

Under `/ephemeral/results/untouched_battery/` (mirrored to the workstation):

- `BATTERY_PREP_RECORD.json` — preflight identity record (16 checkpoints,
  16 threshold sources, metrics blob, both code trees).
- `feature_verification.json` — per-session feature/pixel bitwise binding.
- `v3_repro/` (contract-evaluator reproduction, six exact verdicts) and
  `v3_repro_pinned_diagnostic/` (pinned-tree diagnostic).
- `raw_a9a4144/<sid>/`, `raw_dirty/<sid>/` — per-tree eval JSONs and stored
  prediction sidecars exactly as emitted (raw execution records).
- `sessions/<sid>/<run>_battery.json` — the reported per-session results
  (both supports), 64 files, plus per-session `intersection_support.json`
  and `derived_intersection/` sidecars.
- `pooled/<run>_battery_pooled.json` — the sixteen pooled reports (primary
  result), `pooled_intersection_support.json`, and `pooled/derived/`
  sidecars.
- `trivial_baselines.json`, `summary_tables.json`, `gpu_monitor_pass.csv`,
  `logs/`, `scripts/`, this report, and `MANIFEST.sha256` covering every
  file above.

No git commit was made by this pass; the main session verifies and commits.
The four sealed battery sessions have now been evaluated exactly once each;
under the one-pass rule they are spent as untouched surfaces and are never
re-inferred under different settings.
