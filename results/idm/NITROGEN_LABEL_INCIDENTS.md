# NitroGen label incidents and the resolved-v3 result

Status: written 2026-08-03, after the resolved-v3 production run completed.
This is the consolidated record of two independent defects in the NitroGen
action-mapping pipeline, the corrections that repaired them, and the
retraining sequence that measured the outcome. It supersedes no incident
artifact; every claim below is backed by the machine-readable authorities
listed at the end.

These were two separate failures with separate mechanisms, found in
sequence. The v2 correction repaired a vertical-axis coordinate-contract
violation. The v3 correction subsequently repaired jump/dash/grab button
binding inference, of which dash starvation was the clearest failure. The
labels in both cases were wrong because of our mapping code, not because of
noise in the NitroGen source data.

## Incident 1: the vertical-axis contract was violated

NitroGen declares one dataset-wide joystick convention: negative X is left,
positive X is right, negative Y is up, positive Y is down. The original
mapper (v1.0) instead inferred the Y-axis sign separately for every video
from sparse D-pad/stick coincidence votes. That heuristic assigned
`positive_is_up` — a property the mapper invented, present nowhere in
NitroGen metadata — to exactly 22 of the 210 training videos, and the vote
was structurally biased: D-pad taps during stick-primary play vote against
the majority stick direction, so the same heuristic would have "inverted"
the X axis had it been applied there.

The result was deterministic inversion of analog-derived up/down labels in
22 of 210 training videos: 368 of 1,538 sessions, 2,892,400 of 10,679,200
phase-0 rows, 44,666 of 164,674 training windows, 40.1722 gameplay hours.
Those videos supplied 48.15 percent of all mapped down-positive phase-0
rows, 48.19 percent of natural-window down target exposure, and 37.59
percent of windows containing any down-positive target. Seven of the 22
belonged to the historical unflagged92 training arm, which is why the
matched corrected-v2 rerun changed seven training videos while the
full-corpus incident affected 22. The defect was found by human review of
effect-locked visual evidence — a directional indicator visibly moving up
while the mapped labels read down — after paid training had completed; no
automated admission gate checked agreement with the source's declared
coordinate convention, because the mapper was allowed to infer something
the contract already specified.

Direct D-pad up/down labels were never inverted, so a blind column swap
would have corrupted valid D-pad contributions. The v2 repair instead
rematerialized the vertical labels from the raw controller arrays:
`up = dpad_up OR j_left_y < -0.5` and `down = dpad_down OR j_left_y >
+0.5`, preserving membership, rows, timestamps, pixels, features, session
boundaries, engine indices, masks, left/right, jump, dash, grab, and
direct D-pad semantics. An independent audit reconstructed all 32,037,600
raw rows and established zero remaining `positive_is_up` reports, all
corrected vertical labels formula-exact, exactly the same 22 affected
videos re-derived from data, 188 videos vertically label-identical, and no
unresolved vertical-axis ambiguity.

## Incident 2: bind inference starved dash and inflated grab

The v2 correction deliberately left jump/dash/grab bindings untouched so
the axis repair could be isolated. The subsequent all-seven-key audit of
the mapping code and raw corpus found an independent structural failure in
how buttons were assigned to actions.

In 13 of the 95 corrected unflagged videos — about 18.8 hours — the
selected dash button fired only 2 to 47 times per hour while an unselected
candidate with a far more plausible dash signature fired roughly 700 to
2,400 times per hour. In `v2136986189`, dash was assigned to
`right_shoulder` at about 3 presses per hour while the unselected `west`
button — Celeste's common default dash binding — fired about 1,454 times
per hour with direction co-press 1.00. The inferred dash label in these
videos was effectively empty of genuine dashes.

The mechanism was the interaction of several design choices: the support
term saturated at three presses; the dash shape score awarded its best
value to presses of three frames or fewer, while real players often hold
dash 8 to 24 frames, so genuine dash buttons scored around 0.1 to 0.3; a
three-press phantom button could therefore receive full support, a perfect
short-press shape, and perfect direction co-press on a tiny sample; and
the one-button-per-action assignment forced dash onto the phantom rather
than leaving the action unresolved. In many of these videos the
supposedly lower-confidence broad fallback would have produced better dash
labels than the "unflagged" inferred binding.

The video-wide confidence flag compounded this. One scalar decided whether
all three actions used inferred bindings or the broad fallback; 86 of 210
videos sat within ±0.03 of the 0.5 threshold and the median confidence was
about 0.488. A reliable jump inference could be discarded because dash was
ambiguous, and a video could be declared unflagged with an indefensible
dash assignment. Correcting the axis alone moved three videos across the
flag boundary (118 flagged / 92 unflagged became 115 / 95), and
`v1459001667` entered the unflagged cohort while remaining on the
dash-starvation list. "Flagged" never meant corrupt or excluded — it meant
the broad fallback was used — but the flag could not express per-action
quality. Separately, grab's fallback was the OR of all four
shoulders/triggers, which admitted resting-finger holds and phantom
one-frame presses and left grab occupying roughly 37 percent of all mapped
raw rows. Left/right were independently verified correct, and corrected-v2
up/down were independently verified correct; the remaining material
defects were confined to jump/dash/grab binding selection.

## The v3 resolution

V3 replaced the video-wide heuristic with one resolved button set per
(video, action) across all 210 videos, combining upstream
`actions_processed` evidence where available, v3.1 per-action inference
with multi-bind support (Celeste's defaults bind two jump, two dash, and
four grab buttons), per-action fallback instead of video-wide fallback,
and Bryan's recorded review decisions under his upstream-preference
policy. One explicit multi-bind retention is `v1451209819:dash`, where
both candidate selections were kept.

The machine-readable authority is
[`nitrogen_bind_resolution_v1/bind_resolution.json`](nitrogen_bind_resolution_v1/bind_resolution.json).
Its 630 video/action entries resolve as 25 agreed, 158 upstream-adopted,
364 v3.1-inferred, 82 v3.1 fallback, and one recorded both-right
multi-bind verdict; two actions had no legible press event and remain
unresolved. 227 entries are policy-resolved but pending final human
review, so results trained on this corpus are described as **resolved
mapped labels (upstream-preferred policy)**, not as individually
human-verified bindings.

V3 leaves left, right, up, and down byte-identical to corrected-v2. At
native 60 Hz:

| Key | v2 positives | v3 positives | Rows changed |
|---|---:|---:|---:|
| jump | 4,930,143 | 6,093,581 | 2,372,372 |
| dash | 2,300,586 | 2,839,622 | 1,831,182 |
| grab | 11,924,894 | 11,305,029 | 1,665,227 |
| left/right/up/down | unchanged | unchanged | 0 |

Dash-positive supervision increased by 539,036 native rows, about 23.4
percent. At the 20 Hz phase-0 training grid, 1,288,235 rows changed in at
least one action (jump 790,636, dash 610,182, grab 555,150, directions
zero), with row effect in 151 videos.

## Training sequence and measured results

Three from-scratch checkpoints, all scored on the identical held-out Wild7
support (seven admitted public videos, HUD-decoded truth masked from
input, 842,624 center-supported rows, zero training overlap; the
nine-checkpoint table is in
[`vpt_wild7_checkpoint_parity_v1/scorecard.json`](vpt_wild7_checkpoint_parity_v1/scorecard.json)):

| Checkpoint | Labels | Population | Macro AP | down | dash |
|---|---|---|---:|---:|---:|
| unflagged92 (historical) | pre-correction | 92 videos / 103.4 h | 0.5207 | 0.2647 | 0.3855 |
| unflagged92 corrected-v2 | axis-corrected | 92 videos / 103.4 h | 0.4827 | 0.3245 | 0.2391 |
| full-210 resolved-v3 | axis-corrected + resolved binds | 210 videos / 148.3 h | **0.6334** | **0.3953** | **0.5630** |

The resolved-v3 run (`vpt_small_105696398_nitrogen210_resolved_v3_s0`,
checkpoint SHA-256
`c0371c1afdf5bf835f0216099656f939f5940b0ad5ad3a51cb445fa34f6fa483`)
reaches 0.6165 equal-video and 0.6334 row-weighted Wild7 macro AP at
79.88 percent micro key accuracy, with two independent inference passes
byte-identical. Its dash AP clears the preregistered historical bar of
0.3855. Secondary surfaces: y4n NitroGen-agreement 0.5299 macro AP with
dash 0.5338, and the fixed val-A legacy engine-truth reference scores
dash 0.6240.

Two comparison caveats are part of the record. First, between the two
unflagged92 arms — where no dash training label changed — dash AP moved
from 0.3855 to 0.2391. The pair is confounded (one seed per condition,
different CUDA builds, vertical labels changed in seven videos), so
shared-backbone coupling, environment differences, and optimization
variance cannot be separated; the observation stands only as proof that a
single retrain can move unaffected outputs materially, which is why no
single-seed causal claim is made below. It is not evidence that the old
vertical labels were acceptable; they were objectively wrong. Second, the
historical 161.97-hour full-foreign checkpoint scores 0.6085 Wild7 macro
AP and 0.5421 dash, but its training population included the seven Wild
evaluation videos; it is not held-out performance and is excluded from
generalization comparisons. The resolved-v3 result exceeds it while
remaining zero-overlap.

## What is claimed and what is not

Supported: the resolved-v3 corpus removed the identified dash-starvation
mechanism, increased usable dash supervision by about 23.4 percent, and
produced the strongest held-out Wild7 result — overall and for dash —
among the comparable unseen-data checkpoints.

Not claimed: that the +0.3239 dash improvement over corrected-v2 was
caused solely by the dash label correction. V3 changed the training
population (210 videos and 148.3 hours versus 92 and 103.4) at the same
time as the labels, jump and grab labels also changed, no corrected-v2
full-210 control was trained, and the comparison is single-seed.
Attribution of the improvement specifically to bind resolution would
require a matched corrected-v2 full-210 control and replication across
another seed; until then the population and label-policy changes are
reported together and causal decomposition is avoided.

## Authorities

1. [`nitrogen_bind_resolution_v1/bind_resolution.json`](nitrogen_bind_resolution_v1/bind_resolution.json) — the 630 resolved bind entries and their review statuses
2. [`vpt_small_nitrogen210_resolved_v3_v1/`](vpt_small_nitrogen210_resolved_v3_v1/RESULTS.md) — v3 production run and evaluations
3. [`vpt_wild7_checkpoint_parity_v1/scorecard.json`](vpt_wild7_checkpoint_parity_v1/scorecard.json) — nine checkpoints on identical Wild7 support
4. [`VPT_SMALL_WILD_ADMITTED7_PRIMARY_GATE.md`](VPT_SMALL_WILD_ADMITTED7_PRIMARY_GATE.md) — the Wild7 gate adoption and val-A role

The remaining machine-readable authorities — the sealed incident record
and blast-radius measurement, the v2 correction contract and independent
verification, the four-evidence-class axis verdict, the all-seven-key
audit, and the bind-resolution build contract and training handoff — live
in the private working repository under `results/idm/`, bound by the
hashes cited above.
