# VPT checkpoint parity on Wild admitted7

All listed checkpoints were evaluated on the same 842,624 active Wild admitted7 rows at the raw 0.5 threshold. The historical macro-AP values came from earlier evaluation surfaces and remain context only; the Wild7 columns are the apples-to-apples comparison.

| Model | Historical macro AP | Wild7 macro AP | Micro key accuracy | Relationship to Wild7 |
|---|---:|---:|---:|---|
| Paper-IDM 482M, Tier-B | 0.1844 | 0.3035 | 0.7071 | unseen |
| VPT-small native60 short | 0.2890 | 0.4047 | 0.7404 | unseen |
| VPT-small native60 full | 0.3010 | 0.4155 | 0.7385 | unseen |
| VPT-small native60 span384 | 0.2071 | 0.3595 | 0.7240 | unseen |
| VPT-small unflagged92, 103.4h | 0.4209 | 0.5207 | 0.7728 | unseen |
| VPT-small down-ridge fine-tune | 0.4030 | 0.5141 | 0.7689 | unseen |
| VPT-small full-foreign, 161.97h | 0.4235 | 0.6085 | 0.7930 | training overlap |
| VPT-small corrected-v2 unflagged92 | 0.2715 | 0.4827 | 0.7599 | unseen |
| VPT-small full-210 resolved-v3 | — | **0.6334** | **0.7988** | unseen |

The full-foreign score is not a generalization result because the seven Wild videos were training members. The span384 evaluator repeated the final two context frames for one 382-row continuity run; this preserved the identical scored center rows and affected no labels or row membership.

Exact per-key AP, checkpoint hashes, prediction hashes, and R2 report locations are in `scorecard.json`.
