# Corrected own-v3 primary reruns

Status: complete and independently validated, 2026-07-28.

## Verdict

The corrected overlay-mask geometry did **not** rescue the scratch own-data
model. On the 25,028-row input-active val-A population, its three-seed macro AP
changed from 0.1735 to 0.1700 (paired mean delta -0.0035) against 0.1715 macro
prevalence. Same-surface oracle event F1 also fell slightly. The old overlay
sliver therefore does not explain the scratch model's near-chance ranking.

The Tier-B-initialized own-data fine-tunes improved modestly in ranking and at
the natural 0.5 state threshold: macro AP rose from 0.1938 to 0.1998 and state
F1 from 0.0635 to 0.0895. The AP gain was seed-variable and concentrated in
`grab` and `dash`; oracle +/-2-frame event F1 fell from 0.1003 to 0.0986. This
does not overturn the earlier decision to reject local fine-tuning as a stable
recipe.

This is the clean attribution result: corrected pixels/features alter some
fixed-threshold behavior, but they do not reveal a hidden timing breakthrough
or explain the basic own-only generalization failure.

## Exact comparison

All headline rows below use selected checkpoints and the input-active val-A
population. AP is threshold-free. State F1 and `fixed exact` use probability
threshold 0.5. `Oracle exact` and `oracle +/-2` use per-key thresholds fitted
separately within that same report population on val-A; they are diagnostic
development ceilings, not frozen-test estimates.

### Scratch own-only

| Seed | AP old | AP v3 | Delta | State F1 old | State F1 v3 | Delta | Oracle exact old | Oracle exact v3 | Oracle +/-2 old | Oracle +/-2 v3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1730 | 0.1684 | -0.0046 | 0.0487 | 0.0535 | +0.0048 | 0.0801 | 0.0788 | 0.0990 | 0.0934 |
| 1 | 0.1761 | 0.1740 | -0.0020 | 0.1147 | 0.1279 | +0.0132 | 0.0789 | 0.0781 | 0.0963 | 0.0892 |
| 2 | 0.1714 | 0.1677 | -0.0038 | 0.0505 | 0.0764 | +0.0259 | 0.0786 | 0.0783 | 0.0925 | 0.0880 |
| **Mean** | **0.1735** | **0.1700** | **-0.0035** | **0.0713** | **0.0860** | **+0.0146** | **0.0792** | **0.0784** | **0.0960** | **0.0902** |

Fixed-0.5 exact event F1 changed from 0.0065 to 0.0095 (+0.0029).
That improvement at one operating point coexists with worse threshold-free AP
and worse oracle +/-2 timing, so it is not evidence that the corrected geometry
made the representation generally better.

### Tier-B-initialized own-data fine-tune

| Seed | AP old | AP v3 | Delta | State F1 old | State F1 v3 | Delta | Oracle exact old | Oracle exact v3 | Oracle +/-2 old | Oracle +/-2 v3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.2080 | 0.2174 | +0.0094 | 0.0458 | 0.0923 | +0.0466 | 0.0887 | 0.0895 | 0.1009 | 0.0999 |
| 1 | 0.1742 | 0.1715 | -0.0027 | 0.0827 | 0.0962 | +0.0135 | 0.0841 | 0.0838 | 0.0986 | 0.0952 |
| 2 | 0.1991 | 0.2105 | +0.0113 | 0.0619 | 0.0798 | +0.0179 | 0.0870 | 0.0888 | 0.1015 | 0.1007 |
| **Mean** | **0.1938** | **0.1998** | **+0.0060** | **0.0635** | **0.0895** | **+0.0260** | **0.0866** | **0.0873** | **0.1003** | **0.0986** |

Fixed-0.5 exact event F1 changed from 0.0090 to 0.0099 (+0.0009).

### Mean per-key AP deltas

| Family | Left | Right | Up | Down | Jump | Dash | Grab |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scratch own-only | +0.0046 | +0.0047 | -0.0147 | +0.0020 | +0.0045 | -0.0106 | -0.0147 |
| Tier-B init | -0.0020 | -0.0028 | +0.0014 | +0.0009 | +0.0013 | +0.0094 | +0.0340 |

The all-frame population agrees qualitatively. Scratch macro AP changes
0.1620 -> 0.1594 (-0.0026), while Tier-B-init changes 0.1776 -> 0.1819
(+0.0043). Full seed, population, metric, and per-key values are in
[`own_v3_primary_reruns_delta.json`](own_v3_primary_reruns_delta.json).

At threshold 0.5, the clean scratch seeds reach 73.62%, 76.14%, and 76.79%
per-key micro accuracy (75.52% mean) and 11.79%, 16.13%, and 16.26% joint
exact-match (14.73% mean). The clean Tier-B-init seeds reach 79.55%, 71.99%,
and 77.80% micro (76.45% mean) and 25.40%, 18.99%, and 23.93% joint (22.77%
mean). The common truth-only baselines are 82.85%/33.60% always released and
98.95%/93.32% one-frame persistence. Accuracy remains dominated by released
states and is not the selection metric.

## What was held fixed

- Data changed only from the mask-era own feature cache to the corrected
  `own-v3` generation. The three train sessions and sole val-A session are
  identical in identity, supervision, and engine-frame indexing.
- The feature receipt is SHA-256
  `3a3920166f620b571b31b30e1d755cbac79c51d524cfda1b1abfe85fc3812223`;
  it binds 178,525 frames, all four feature NPZs, the corrected RGB source,
  generation commit `f8a51d7e146904500a141f5a7876f40e435f1e16`, and split hashes.
- Training used the historical implementation at
  `a9a414452c07fb101d26faee9fa13864dde922b4`, matching the original relevant
  trainer/model/schema bytes and the original Torch 2.13.0+cu129 runtime.
- Scratch used 2,000 steps, Adam at 3e-4, and eval every 250. Tier-B-init used
  600 steps, Adam at 1e-4, eval every 50, with each initializer hash verified
  against the tracked Tier-B checkpoint.
- Selection remained minimum arithmetic mean of the seven plain val BCEs,
  including step zero, with strict-improvement tie behavior. Class weights
  were recomputed from corrected supervision as in the original recipe; they
  were not copied from the mask-era runs.
- Evaluation used the one explicit val-A session and produced both all-frame
  and input-active report populations plus finite aligned prediction sidecars.
  No other evaluation surface was read or scored by this study.

The phrase `Tier-B init` here means the historical 600-step local fine-tune
arm. It is not the zero-shot Tier-B family in the original primary headline:
those models consumed no own training shards and therefore had nothing to
rerun for this attribution question.

## Checkpoints and selection

| Run | Endpoint | Selected step | Selected val BCE | Selected/final | Checkpoint SHA-256 |
|---|---:|---:|---:|---|---|
| `own_features_v3_32nc_s0` | 2,000 | 250 | 0.538801 | different | `5b2132f228664d2e0de4da2e15961b7d735ff7146dafb7338d8d9123cab8d6a3` |
| `own_features_v3_32nc_s1` | 2,000 | 250 | 0.499741 | different | `05f80e665eb87c84adbc22628ef95db88f9263661b07ef7ca4374d3713bfaf72` |
| `own_features_v3_32nc_s2` | 2,000 | 250 | 0.540591 | different | `525ce4423d15cd7eff23d0c3088318d6f546e363afc289a2cf1f0460ffb98ce5` |
| `own_features_v3_tier_b_init_32nc_s0` | 600 | 200 | 0.503209 | different | `4c7308616fc9a5c64489ec872060c27ec726abe74a2a013d25f0d421b602e181` |
| `own_features_v3_tier_b_init_32nc_s1` | 600 | 450 | 0.559137 | different | `35cd8efe9d420d82bbd5619bdb8e617ef7b9c009961c8e25bbd3594ffd583605` |
| `own_features_v3_tier_b_init_32nc_s2` | 600 | 400 | 0.526485 | different | `8fcf4e5cd2b813855aa67822dc865d5b5b3924eae587f25b0529027a36ebf13b` |

Every selected and final tensor is finite. Every selected state differs from
its final endpoint, as expected for the small-data memorization regime.

## Durability and validation

Each checkpoint was registered immediately after training and before val-A
inference. All six hashes are now in `checkpoint_sha256.txt` and the dedicated
six-record registry `checkpoint-index-own-v3-primary-20260728.json`, which
remains in the private working repository because it records storage
coordinates the public export excludes; the hashes above are the complete
public record.

Each model was uploaded under a new content-addressed Cloudflare R2 prefix
the durable object store under `runs/idm/v1/<artifact-id>/<checkpoint-sha256>/`. Publication
allowed exactly `model.pt`, `checkpoint-manifest.json`, and
`checkpoint_complete.json`; checkpoint and manifest were streamed back before
the marker was uploaded last, then all three were streamed back again. The
tracked per-run manifests, completion receipts, and publication receipts bind
those remote bytes without committing model weights to Git.

Independent validation checked all six checkpoint registrations, finite
selected/final states, selected log minima, source/feature/split/init hashes,
report membership and support, 177 segment boundaries, finite aligned
probability arrays, marker-bound artifact hashes, tracked registry entries,
and R2 receipt agreement. All checks passed.

The first two scratch trainings finished and registered correctly, after which
their isolated evaluator failed to import a current dependency. The dependency
was added to the contract and evaluation resumed from the already registered
checkpoints; neither model was retrained. The historical trainer also created
its documented 678-byte four-shard hash memo after the first run; that exact
memo was bound as runtime cache rather than treated as source data.

Compute plus val-A inference took about 7 minutes 36 seconds wall-clock across
the two A100 lanes, including the evaluator-deployment recovery. R2 publication
and final validation added roughly another minute.

## Canonical artifacts

- Frozen contract: [`../../experiments/configs/own_v3_primary_reruns.json`](../../experiments/configs/own_v3_primary_reruns.json)
- Machine-readable deltas: [`own_v3_primary_reruns_delta.json`](own_v3_primary_reruns_delta.json)
- Six-checkpoint registry: `checkpoint-index-own-v3-primary-20260728.json`
  (private working repository; carries storage coordinates)
- Per-run config, run metadata, log, registration, R2 manifest/completion, and
  val-A completion receipts: the six `own_features_v3_*` directories here
- Standard reports and prediction sidecars: the six
  `own_features_v3_*_val_a.json` / `_preds.npz` pairs here
