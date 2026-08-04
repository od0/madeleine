# Native-60-Hz VPT-small three-arm result

Status: complete on 2026-07-31. All three frozen final endpoints were trained,
published marker-last to immutable R2 prefixes, independently stream-read for
checkpoint identity, evaluated on the exact frozen common support, handed off
locally as compact evidence, and terminated. No node remains running.

## Bottom line

Direct native-60-Hz training did **not** improve VPT-small under any tested
recipe. The strongest native-rate arm was the 128-frame, 20-epoch arm. It
recovered some of the short arm's lost state accuracy and tolerant timing, but
still trailed the old 20-Hz model by 0.0592 macro AP, 0.0537 state F1, 0.0180
exact-event F1, and 0.0153 tolerant-event F1. The 384-frame span-matched arm
was worse still. These results reject the simple theory that the old model was
accuracy-limited mainly by 20-Hz sampling.

The result does not say that 60-Hz information is useless. It says that making
the full VPT-small graph operate densely at 60 Hz, either with one-third the
physical context or with a three-times-longer attention sequence, was not an
effective use of this data and optimizer recipe. A multirate design remains a
different, still-open hypothesis.

## Frozen comparison

Every row below uses the final endpoint only, threshold 0.5, and the same
4,224 active rows / 21 streams from corrected own-v3 val-A. The native-rate
evaluations were restricted to the old VPT-small final sidecar SHA-256
`4039356c349c7d44add3a700a0c32fdc3a1019c339af1a308e57274cdfef546a`.

| Model | Input cadence / context | Updates | Macro AP | State F1 | Exact event F1 | +/-2 native-frame F1 | Micro / joint accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Old VPT-small | 20 Hz, 128 frames = 6.4 s; three-phase score grid | 2,340 | **0.3603** | **0.2504** | **0.0351** | **0.1005** | 84.89% / **38.35%** |
| Native60 short | 60 Hz, 128 frames = 2.13 s | 2,340 | 0.2890 | 0.1782 | 0.0218 | 0.0600 | 82.64% / 19.32% |
| Native60 full | 60 Hz, 128 frames = 2.13 s | 7,060 | 0.3010 | 0.1968 | 0.0172 | 0.0852 | **85.21%** / 36.34% |
| Native60 span | 60 Hz, 384 frames = 6.4 s | 2,340 | 0.2071 | 0.1478 | 0.0127 | 0.0234 | 80.98% / 27.27% |

Always released is 84.19% micro / 35.58% joint on this support. Persistence is
98.98% / 93.51%. The full native-rate arm narrowly exceeds the old model in
micro accuracy (+0.33 point), but that aggregate gain coexists with worse
ranking, state F1, event localization, joint accuracy, and rare-key coverage.
It is not the more accurate IDM in the metrics that expose action structure.

## What each comparison answers

### Rate at matched 6.4-second span

Old 20-Hz VPT-small versus native60 span384 holds physical context at 6.4
seconds. The native-rate arm loses 0.1532 macro AP, 0.1026 state F1, 0.0225
exact-event F1, and 0.0771 tolerant-event F1. Merely supplying every native
frame did not overcome the harder 384-token optimization problem. The
three-times-longer sequence also changes window population, retained center,
and attention workload, so this is a system-level rate control rather than a
single-component attribution.

### Exposure within the 128-frame native-rate recipe

Extending native60 128-frame training from 2,340 to 7,060 updates raises macro
AP by 0.0120, state F1 by 0.0186, tolerant-event F1 by 0.0252, micro accuracy
by 2.57 points, and joint accuracy by 17.02 points. Exact-event F1 falls by
0.0047. More exposure clearly helps this recipe, but it does not close the gap
to the old 20-Hz final endpoint.

### Physical context at matched 2,340 updates

At the same 2,340 updates, replacing the 128-frame / 2.13-second native-rate
window with the 384-frame / 6.4-second window loses 0.0819 macro AP, 0.0304
state F1, 0.0092 exact-event F1, and 0.0366 tolerant-event F1. Longer physical
context is not free when it is bought by tripling the dense token sequence.

## Per-key ranking

| Key | Old 20 Hz AP | Native60 short AP | Native60 full AP | Native60 span AP |
|---|---:|---:|---:|---:|
| dash | **0.3775** | 0.1032 | 0.0831 | 0.1344 |
| down | 0.0102 | 0.0121 | 0.0105 | **0.0386** |
| grab | 0.5077 | 0.3869 | 0.4490 | **0.6688** |
| jump | **0.3592** | 0.3197 | 0.3073 | 0.0999 |
| left | **0.3078** | 0.2243 | 0.2793 | 0.1393 |
| right | 0.7392 | 0.6837 | **0.7978** | 0.2403 |
| up | 0.2202 | **0.2930** | 0.1801 | 0.1284 |

The span arm's improvements on `down` and `grab` do not compensate for large
losses on the other keys. Both 128-frame native-rate endpoints emit no `dash`
or `down` positives at 0.5; the short arm also emits none for `up`, while the
full arm emits only 0.95% `up` positives against 13.75% prevalence. Native-rate
training therefore does not clear the old candidate's rare-key coverage
failure.

## Proper scores and endpoint discipline

| Model | Natural NLL | Brier | Macro PPR / prevalence |
|---|---:|---:|---:|
| Old VPT-small | 3.4139 | 0.1240 | 0.0923 / 0.1581 |
| Native60 short | **3.1353** | 0.1341 | 0.1033 / 0.1581 |
| Native60 full | 3.1801 | **0.1211** | 0.0629 / 0.1581 |
| Native60 span | 3.7739 | 0.1499 | 0.1384 / 0.1581 |

Validation-loss metadata was retained for lifecycle evidence, but the frozen
study permitted evaluation of final endpoints only. No best-NLL checkpoint
was scored or selected after training.

## Training and durability

| Arm | Training time | Throughput | Peak VRAM | Final checkpoint SHA-256 |
|---|---:|---:|---:|---|
| Native60 short | 12,231 s (3:23:51) | 24.49 seq/s | 22.52 GB | `f97d38facdae590d6f0b6e69ef5d1fce0b22509359f667ae7db6ff769c9ef8fb` |
| Native60 full | 50,825 s (14:07:05) | 17.78 seq/s | 22.52 GB | `43733c2e1804fdb6baab477ddfe58113342b459b7c761a199f9cd7873f0fe947` |
| Native60 span | 46,911 s (13:01:51) | 6.38 seq/s | 63.93 GB | `52251fa10f6a0d03e786b71afcf44d85963cf32676496e9e580069be60959e5f` |

All generated checkpoints have matching immutable R2 streamed-hash receipts.
The three final objects were independently streamed back and matched these
hashes. Compact run/evaluation evidence is under
`vpt_small_native60_run_evidence/`; smoke evidence and exact generation
contracts are also tracked. The 128-frame contract SHA-256 is
`efae92a5e421be5c40f82869547360cbed980044ea531db4ad692b400507a32e`;
the corrected relative-path 384-frame contract SHA-256 is
`74cd3725073e3f6f1c9b59cea87fc80243f02d0ac1db220876e64e940f3cb951`.
Total node-lifecycle cost was $111.00. See
[`VPT_NATIVE60_THREE_ARM_LIFECYCLE.json`](VPT_NATIVE60_THREE_ARM_LIFECYCLE.json)
for node-level timing, price, cost, readback, and termination receipts.

Focused native60 architecture/data/contract/training/evaluation/watcher tests
passed 27/27. The full repository suite passed 953 tests and had one unrelated
pre-existing own-v3 evidence failure: its tracked R2 publication prefix does
not match the artifact-id-derived prefix expected by
`test_report_own_v3_primary_reruns.py`. No native60 test failed.
The machine-readable metric, hash, durability, and test audit is
[`VPT_NATIVE60_THREE_ARM_RELEASE_VALIDATION.json`](VPT_NATIVE60_THREE_ARM_RELEASE_VALIDATION.json).

## Decision

- Keep the old 20-Hz VPT-small final as the strongest tested 105.7M endpoint.
- Do not promote or calibrate a native-rate arm as though it passed the old
  candidate gate; all three native recipes remain development diagnostics.
- Treat more native-rate exposure as a real but insufficient recovery, not as
  evidence that simply running longer will surpass the old model.
- If timing remains the objective, test a multirate refinement path that keeps
  the old model's efficient long-span representation and spends 60-Hz compute
  locally, rather than repeating dense full-graph native-rate training.
