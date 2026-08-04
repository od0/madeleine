# VPT-small NitroGen full-210 v3 seed 0

`vpt_small_105696398_nitrogen210_resolved_v3_s0` completed 20 epochs and
25,740 optimizer steps from scratch on all 210 NitroGen videos (148.3222
hours). The epoch-20 checkpoint SHA-256 is
`c0371c1afdf5bf835f0216099656f939f5940b0ad5ad3a51cb445fa34f6fa483`.

## Wild admitted7 primary evaluation

The fixed, untuned evaluation contains 842,624 active rows from seven unseen
public-gameplay videos. Row-weighted macro AP is **0.633425** and micro key
accuracy is **79.88%**.

| Key | AP |
|---|---:|
| left | 0.7611 |
| right | 0.9023 |
| up | 0.6095 |
| down | 0.3953 |
| jump | 0.5675 |
| dash | 0.5630 |
| grab | 0.6354 |

The Wild7 macro AP is +0.1507 over corrected-v2 unflagged92 and +0.1127 over
the pre-correction unflagged92 result. Dash AP is +0.3239 over corrected-v2
and +0.1775 over the registered pre-correction dash bar. Two independent
inference passes produced byte-identical prediction sidecars.

## y4n agreement evaluation

The y4n agreement surface scores **0.529862 macro AP** and **85.04% micro key
accuracy** across 555,840 native-grid rows.

| Key | AP |
|---|---:|
| left | 0.7271 |
| right | 0.8416 |
| up | 0.5548 |
| down | 0.0525 |
| jump | 0.4873 |
| dash | 0.5338 |
| grab | 0.5119 |

## Fixed val-A legacy reference

On the historical 4,224-row, 21-stream common support, the final checkpoint
scores **0.474843 macro AP** and **87.21% micro key accuracy** at the fixed
raw 0.5 threshold. This is retained as a legacy reference; Wild admitted7
remains the primary evaluation surface.

| Key | AP |
|---|---:|
| left | 0.4416 |
| right | 0.7033 |
| up | 0.4061 |
| down | 0.0114 |
| jump | 0.4916 |
| dash | 0.6240 |
| grab | 0.6459 |

Training and evaluation used PyTorch `2.7.0+cu126`, CUDA 12.6, BF16, and
eight NVIDIA H100 80GB GPUs. A corrected-v2 full-210 checkpoint does not
exist, so no same-population v2-v3 comparison is available.

The exact training configuration is [config.json](config.json), SHA-256
`d4aeed09229fbd666e1fb2b928e0b766d0195ff3a7fc620ce01213f74f1f0c1a`,
byte-identical to the preregistered copy sealed before launch. The public
export replaces two internal storage prefixes in it with `private:`
placeholders, so the exported copy hashes differently; every model,
optimizer, and data field is unchanged.
