# VPT-small public-video deployment gate

Status: adopted 2026-08-02 as the primary evaluation surface for deciding
whether an IDM is ready to label additional public Celeste footage.

## Decision

The admitted Wild7 public-video holdout is the primary deployment gate for
future IDM checkpoints. Corrected own-v3 val-A remains useful as a secondary
engine-truth regression and timing-metrology surface, but it no longer decides
whether a model is suitable for labeling Internet video.

Wild7 stays holdout-only. Its seven videos and 1,455 sessions must not enter
training, calibration, threshold selection, or pseudo-label training. Future
public-video harvesting must use disjoint sources.

## Baseline result

The evaluated checkpoint is epoch 20 of
`vpt_small_105696398_unflagged92_103p4056h_axisfix_corrected_v2_s0`, trained
from scratch on the exact historical NitroGen unflagged92 population. Its
checkpoint SHA-256 is
`628a3311a65a9e58fd30cbafcff1fe97311f4af885d5db8566cff4363050e4c5`.

The gate contains seven public videos, 1,455 source sessions, 13.650625 phase-0
hours, and 13,166 possible windows. The fixed 128-frame, stride-64 evaluation
retains 842,624 center-supported rows in 1,234 streams, or 11.703111 hours at
20 Hz. Truth comes from the videos' visible seven-action HUDs. The complete HUD
is masked before model input. No local engine truth or NitroGen analog mapping
is used, and the evaluation has zero video or session overlap with training.

Two independent inference passes produced byte-identical prediction sidecars.

| Key | Row-weighted AP | Equal-video AP |
|---|---:|---:|
| left | 0.5660 | 0.5688 |
| right | 0.8296 | 0.8415 |
| up | 0.4495 | 0.4549 |
| down | 0.3245 | 0.3634 |
| jump | 0.3863 | 0.3870 |
| dash | 0.2391 | 0.2534 |
| grab | 0.5842 | 0.4557 |
| **Macro** | **0.4827** | **0.4750** |

At the fixed, untuned 0.5 threshold, macro state F1 is 0.3025, micro key
accuracy is 75.99 percent, and exact seven-key accuracy is 15.37 percent.
Average precision is the primary measure because it evaluates ranking without
pretending that a threshold learned on another label channel is calibrated for
these videos. Equal-video macro AP is the headline comparison so a long source
cannot dominate the result; row-weighted AP and every per-key AP remain
mandatory secondary reporting.

The complete result is in
[`vpt_small_unflagged92_axisfix_corrected_v2_v1/wild_admitted7_eval_v1/evaluation.json`](vpt_small_unflagged92_axisfix_corrected_v2_v1/wild_admitted7_eval_v1/evaluation.json).
The frozen protocol is in
[`wild_admitted7_eval_contract.json`](vpt_small_unflagged92_axisfix_corrected_v2_v1/wild_admitted7_eval_v1/wild_admitted7_eval_contract.json).

## Why val-A `down` is not the primary gate

The same checkpoint scores only 0.01189 `down` AP on fixed val-A common
support. That discrepancy does not currently implicate an inverted or corrupt
val-A ground-truth channel:

- The common-support evaluation contains only 58 positive `down` rows and 10
  `down` onsets among 4,224 rows: 1.37 percent prevalence and roughly 2.9
  seconds of positive state. Wild7 contains 18.51 percent `down` prevalence
  across 842,624 rows.
- The exact val-A source shard used by the evaluation has SHA-256
  `6abde83a24da4202e7c148722f2ac0ceaa0f204e13905795c5662c1d267b3d62`.
  It is derived from InputTruth 0.2.0, which records `up = MoveY < 0` and
  `down = MoveY > 0`.
- An independent pixel-to-log metrology pass decoded the machine-readable
  seven-key overlay from this same capture and matched the engine log at
  macro-F1 1.0. For `down`, it matched all 457 positive frames with zero false
  positives or false negatives and all 44 onsets at median offset zero.
- Swapping the current model's vertical heads does not rescue val-A `down`:
  the `up` head scores 0.00980 against `down`, below the direct `down` head's
  0.01189.
- The two surfaces also observe slightly different semantics. Val-A records
  Celeste's resolved bound `MoveY` state, while Wild7 decodes the state shown
  by a public input HUD. Simultaneous or otherwise conflicting physical inputs
  can therefore differ even when both label channels are functioning as
  designed.

The supported conclusion is narrower: val-A is a valid but small local
engine-truth regression set whose behavior, rooms, visual distribution, and
rare-key prevalence poorly represent harvested public footage. It should catch
capture-contract or catastrophic model regressions, not veto an otherwise
strong public-video IDM.

## Gate policy for the next corrected-label run

The next checkpoint must be trained from scratch on its newly corrected frozen
label population, then evaluated once on the unchanged Wild7 contract. Model
selection, threshold selection, and early stopping must not consult Wild7.

Report, in order:

1. equal-video macro AP;
2. row-weighted macro AP;
3. all seven per-key AP values, with `down` and `dash` called out rather than
   hidden by the macro;
4. fixed-0.5 state metrics as calibration diagnostics;
5. the secondary val-A engine-truth regression result.

The result above is the frozen comparison baseline for the next run. A formal
go/no-go threshold for fully automatic thousand-hour pseudo-label admission is
not claimed from seven videos. Scaling should begin with confidence-filtered
labels and audit sampling until a larger, source-diverse public holdout exists.

## Resolved-v3 result under this gate (2026-08-03)

The next checkpoint arrived as
`vpt_small_105696398_nitrogen210_resolved_v3_s0` (all 210 NitroGen videos,
148.3222 hours, resolved-v3 labels, from scratch, checkpoint SHA-256
`c0371c1afdf5bf835f0216099656f939f5940b0ad5ad3a51cb445fa34f6fa483`), and it
was evaluated once under the unchanged contract, in the required order:

1. equal-video macro AP **0.6165**;
2. row-weighted macro AP **0.6334**;
3. per-key AP 0.7611 left, 0.9023 right, 0.6095 up, **0.3953 down**,
   0.5675 jump, **0.5630 dash**, 0.6354 grab — down and dash both at their
   best held-out values, and dash above the preregistered 0.3855 bar;
4. fixed-0.5 micro key accuracy 79.88 percent;
5. secondary val-A engine-truth regression: 0.4748 macro AP and 87.21
   percent micro key accuracy on the identical 4,224-row support, down AP
   0.0114 — unchanged in character, consistent with the support diagnosis
   above.

Two independent inference passes were byte-identical. This result is the
new frozen comparison baseline for future checkpoints. The training
handoff's registered expectation said down should stay flat between
corrected-v2 and v3 because no down label changed; observed down moved
+0.0708 on this gate, which by that registration's own terms implicates
training dynamics and the enlarged population rather than labels. The
comparison caveats — single seed, no corrected-v2 full-210 control, no
causal decomposition — are recorded in
[`NITROGEN_LABEL_INCIDENTS.md`](NITROGEN_LABEL_INCIDENTS.md), and the
nine-checkpoint parity table is in
[`vpt_wild7_checkpoint_parity_v1/scorecard.json`](vpt_wild7_checkpoint_parity_v1/scorecard.json).

## Public-push plan

Prepare the public update only after the next corrected-label checkpoint and
its Wild7 evaluation are complete. The release should contain:

1. the clean training configuration, exact population summary, checkpoint
   identity, and corrected-label provenance;
2. a matched table comparing this baseline with the new checkpoint on the
   unchanged Wild7 gate;
3. the secondary val-A result, clearly labeled as local engine-truth
   regression evidence rather than the deployment headline;
4. the evaluation contract, compact result JSON, checkpoint and artifact
   hashes, and reproducible scoring commands;
5. a concise README/results narrative focused on the model, data correction,
   and measured result, excluding private operational logs and abandoned
   attempts.

No public commit or push is part of adopting this gate. Those changes should
be reviewed together after the next retraining result is available.
