# Historical GRU parity on Wild admitted7

Five historical GRU checkpoints were rescored on one frozen, natural Wild admitted7 support: 793,716 active rows across 1,234 streams. The support uses each GRU's common 128-frame, stride-3 context without boundary padding, so these numbers are directly comparable within this table.

| Checkpoint | Training data | Macro AP | Micro key acc. | Left | Right | Up | Down | Jump | Dash | Grab |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen-feature GRU 25.7M, all-valid | 148.32h | 0.3871 | 55.33% | 0.3427 | 0.7246 | 0.3467 | 0.1980 | 0.3548 | 0.1974 | 0.5452 |
| Frozen-feature GRU 25.7M, unflagged92 | 103.41h | 0.3858 | 55.48% | 0.3499 | 0.7191 | 0.3364 | 0.2002 | 0.3498 | 0.1967 | 0.5484 |
| Frozen-feature GRU 25.7M, nine-video | 38.01h | 0.3750 | 52.19% | 0.3257 | 0.6994 | 0.3155 | 0.1902 | 0.3541 | 0.2046 | 0.5354 |
| End-to-end pixel GRU 36.9M, Tier-B | 13.45h | 0.3697 | 56.58% | 0.3115 | 0.6895 | 0.2948 | 0.1770 | 0.3412 | 0.2514 | 0.5227 |
| End-to-end pixel GRU 112.95M, Tier-B | 13.45h | 0.3632 | 56.19% | 0.3068 | 0.6871 | 0.2878 | 0.1829 | 0.3386 | 0.2210 | 0.5186 |

Reference models rescored on the identical 793,716-row subset:

| Checkpoint | Macro AP | Micro key acc. | Left | Right | Up | Down | Jump | Dash | Grab |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VPT-small full-210 resolved-v3 | 0.6330 | 79.86% | 0.7621 | 0.9019 | 0.6066 | 0.3924 | 0.5675 | 0.5617 | 0.6385 |
| VPT-small Tier-B | 0.4613 | 75.29% | 0.5668 | 0.8445 | 0.2596 | 0.1575 | 0.4723 | 0.3357 | 0.5923 |

Each GRU was evaluated twice on one H100 using float32 inference. Publication is accepted only if the two lanes produce byte-identical probability arrays. The complete sidecars and receipts are archived at `private:results/idm/v1/gru-wild7-checkpoint-parity-v1`.

These scores use the largest natural support shared by the historical GRU context rules. They are not the full 842,624-row VPT Wild7 surface, so the VPT comparator values here intentionally differ slightly from the full-surface headline values.
