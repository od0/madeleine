# MADELEINE

Measurement And Decoding of Evidence-Linked Environment Inputs via Neural
Estimation: recovering player actions from *Celeste* gameplay video.

Given a window of *Celeste* frames before and after a moment, the model
predicts the seven controls that produced the motion: left, right, up,
down, jump, dash, and grab. The starting point is
[Video PreTraining (VPT)](https://arxiv.org/abs/2206.11795); the emphasis
here is measurement — aligning video to engine frames exactly, masking
every on-screen answer key, scoring held state separately from press
timing, and testing whether noisy internet labels transfer to
engine-recorded actions. The model is offline — it reads frames on both
sides of the moment — and the point is what that enables: turning public
gameplay that has no action log into action-labeled training data for
video-pretrained agents.

<img src="results/figures/fig_pred_overlay.webp" width="1200" alt="Thirty seconds of gameplay scored frame by frame against engine truth">

*Thirty seconds of a self-recorded session
([higher-quality mp4](results/figures/fig_pred_overlay.mp4)), scored frame
by frame against engine truth by a model trained only on mapped NitroGen
labels — it never saw an engine-truth frame in training. Left: the
capture, masked regions dashed. Right: the literal 128×128 input the
network consumes, a gradient-saliency view of it, and the seven per-key
probabilities against engine truth. The bottom strip is the model's
actual input window — 32 consecutive frames centered on the current
moment — shaded by how much each frame influenced the prediction. The
running tally is per-frame accuracy, which flatters trivial predictors
and is [not the headline metric here](results/idm/KEYPRESS_ACCURACY.md).*

In Proust, a madeleine is the sensory trace that recovers a lost past; an
inverse dynamics model does the same, recovering the action from the
frames it left behind. In *Celeste*, the protagonist is Madeline. The
recorder is `theo`, the character who documents everything; the verifier
is `goldenberry`, after the golden strawberry that certifies a deathless
run; and the model itself is `badeline`, the reflection who knows your
every move. Engine truth comes from `granny`, the observer who is never
fooled.

**Reading paths:** [the results](#results) ·
[the technical report](report/README.md) ·
[relation to prior work](report/README.md#relation-to-prior-work) ·
[how this was built: a three-day agent-orchestrated sprint](docs/history/how-this-was-built.md) ·
[curated engineering lessons](docs/engineering-lessons.md)

## Results

Thirteen models spanning the project's architectures — recurrent
baselines through the VPT topology — scored on the same test: seven
public gameplay videos none of them ever trained on — real players, real
recording conditions, nearly twelve hours of scored play. Every video
shows an on-screen input display; it is masked from the model's view and
used as the answer key. (How these videos were found, verified, and
admitted is covered in
[the wild-overlay findings](#a-new-label-channel-wild-keyboard-overlays).)

| Model | Parameters | Training data | Macro AP | Key accuracy |
|---|---:|---|---:|---:|
| GRU, frozen features, 38 h | 26M | 38 h | 0.38 | 52% |
| GRU, frozen features, 103 h | 26M | 103 h | 0.39 | 55% |
| GRU, frozen features, 148 h | 26M | 148 h | 0.39 | 55% |
| GRU, pixels end-to-end, 37M | 37M | 13.5 h | 0.37 | 57% |
| GRU, pixels end-to-end, 113M | 113M | 13.5 h | 0.36 | 56% |
| VPT paper architecture | 482M | 13.5 h | 0.30 | 71% |
| VPT-small, 60 Hz, short training | 106M | 13.5 h | 0.40 | 74% |
| VPT-small, 60 Hz | 106M | 13.5 h | 0.42 | 74% |
| VPT-small, 60 Hz, 384-frame window | 106M | 13.5 h | 0.36 | 72% |
| VPT-small | 106M | 103 h | 0.52 | 77% |
| VPT-small, down-focused fine-tune | 106M | 103 h | 0.51 | 77% |
| VPT-small, corrected labels | 106M | 103 h | 0.48 | 76% |
| **VPT-small, full corrected corpus** | **106M** | **148 h** | **0.63** | **80%** |

Average precision per key on the same test:

| Model | Left | Right | Up | Down | Jump | Dash | Grab |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRU, frozen features, 38 h | 0.33 | 0.70 | 0.32 | 0.19 | 0.35 | 0.20 | 0.54 |
| GRU, frozen features, 103 h | 0.35 | 0.72 | 0.34 | 0.20 | 0.35 | 0.20 | 0.55 |
| GRU, frozen features, 148 h | 0.34 | 0.72 | 0.35 | 0.20 | 0.35 | 0.20 | 0.55 |
| GRU, pixels end-to-end, 37M | 0.31 | 0.69 | 0.29 | 0.18 | 0.34 | 0.25 | 0.52 |
| GRU, pixels end-to-end, 113M | 0.31 | 0.69 | 0.29 | 0.18 | 0.34 | 0.22 | 0.52 |
| VPT paper architecture | 0.21 | 0.54 | 0.31 | 0.18 | 0.28 | 0.17 | 0.44 |
| VPT-small, 60 Hz, short training | 0.49 | 0.81 | 0.25 | 0.17 | 0.36 | 0.20 | 0.55 |
| VPT-small, 60 Hz | 0.50 | 0.84 | 0.23 | 0.16 | 0.40 | 0.23 | 0.54 |
| VPT-small, 60 Hz, 384-frame window | 0.33 | 0.71 | 0.27 | 0.17 | 0.32 | 0.19 | 0.54 |
| VPT-small | 0.66 | 0.86 | 0.45 | 0.26 | 0.45 | 0.39 | 0.58 |
| VPT-small, down-focused fine-tune | 0.65 | 0.86 | 0.44 | 0.31 | 0.43 | 0.35 | 0.56 |
| VPT-small, corrected labels | 0.57 | 0.83 | 0.45 | 0.32 | 0.39 | 0.24 | 0.58 |
| **VPT-small, full corrected corpus** | **0.76** | **0.90** | **0.61** | **0.40** | **0.57** | **0.56** | **0.64** |

Training data beats parameter count here: the 482M build of the original
VPT paper's architecture, trained on the same 13.5 hours as the 60 Hz
small models, finishes last on ranking. Architecture matters just as
much: on the identical 148-hour corpus, the VPT topology beats the best
recurrent model by 24 points of macro AP (the GRU rows come from
[a dual-lane rescore](results/idm/gru_wild7_checkpoint_parity_v1/README.md)
on the same seven videos). Right is easiest for every model, down and
dash sit at the bottom for nearly every one, and the final model —
trained from scratch on the full corrected corpus — is the strongest on
every key. Two reference points: weighting each video equally instead of
each frame gives the final model 0.62 rather than 0.63, and the
do-nothing baseline for key accuracy — always predicting released —
scores 71%. It is the model the project carries forward.

The project's earlier model families also faced sealed engine-truth
tests: sessions recorded after checkpoints, thresholds, and evaluation
code were frozen, scored in exactly one pass each — an unseen chapter of
the game first (best macro AP 0.24 against 0.15 chance), then a
pre-registered four-chapter battery (pooled best 0.27 against roughly
0.16). Every chapter showed above-chance action signal, models trained
only on local capture sat near chance, and nothing was tuned afterward
([battery report](results/idm/untouched_battery/UNTOUCHED_BATTERY.md)).

Exact press timing stayed weak in every sealed test — event F1 topped
out near 0.05, several times better than shuffled chance but far from
usable — and the public-video test above measures held state, not
timing. That wall is what the next experiment attacks: a
[counterfactual identifiability study](results/idm/COUNTERFACTUAL_INPUT_IDENTIFIABILITY_PLAN.md)
will replay sessions deterministically with each press moved frame by
frame, to measure whether exact press timing is visible in pixels at all.

## Architecture

![Research architecture: evidence channels, trust gates, durable state, and execution resources](results/figures/fig_research_architecture.png)

Three evidence channels of different trust — engine-truth capture from a
local rig, NitroGen's mapped gamepad labels, and keyboard overlays
decoded from public videos — converge through explicit review gates into
one training and evaluation stack.

## Findings so far

### Measurements against engine truth

![The rig at a glance](results/figures/fig_rig_frame.png)

*One frame from the calibration session. The rig renders a frame index
(top left) for clock-free alignment, an opaque engine-truth overlay
(bottom left), and a translucent HUD (bottom right) in the style public
videos show — later used to calibrate the overlay decoder. Dashed
rectangles mark the regions masked from the model's view.*

Everything in this list is measured against exact per-frame engine truth
from the local capture rig.

- Alignment requires no clock synchronization: a frame index rendered
  into the game pins every video frame to its engine frame. The chain was
  verified end to end when a classical parser read the input overlay off
  every frame of a 15-minute session and matched the engine log at
  macro-F1 1.0.
- The current frame often does not determine the action. Nearly half of
  active frames (44.8%) have a near-identical engine-state twin at
  another moment that took a different action, and the two trajectories
  only separate over the following 8–16 frames. That is the empirical
  case for giving an inverse dynamics model future context.
- Label timing error costs far more than video quality. Shifting labels
  by a single frame costs 4.5% macro-F1 — four frames costs 17.3% —
  while re-encoding video down to internet-grade 480p produced no
  measurable loss.
- Per-frame metrics reward trivial predictors. Copying the previous
  frame's keys scores 99% per-key accuracy and 0.91 AP — and an
  exact-transition event F1 of zero. That is why transition timing, not
  accuracy, is the headline metric here, and why every accuracy table
  carries the trivial baselines.
- Forty minutes of engine-truth data was not enough to train on. Every
  pixels-only model memorized its training sessions and fell to chance on
  held-out ones, and recording 2.7× more data changed nothing. That
  failure is what justified turning to internet-scale labels.

### What the training experiments showed

- A model trained on 13.5 hours of mapped NitroGen labels — not a single
  engine-truth frame — beat matched engine-truth training on engine-truth
  evaluation, zero-shot, in every paired seed. Fine-tuning on the local
  captures was tried and rejected: it degraded exact timing every time.
- End-to-end vision, more capacity, and longer context each improved
  state recognition but not exact timing: the best of these models
  reached 0.25 macro AP against a 0.17 chance baseline while its exact
  event F1 fell. The likely mechanism, echoing LAPO, is that pixels
  reveal when an action takes effect, not when the key went down.
- Cross-video transfer is real, and it scales. Trained on nine NitroGen
  videos (38 h) and tested on a tenth, a 26M model put every key above
  its own base rate; scaling the same recipe to the full 148-hour corpus
  raised held-out AP from 0.24 to 0.27 and joint exact-match accuracy
  from 4% to 11%. On engine-truth capture, that 148-hour model posts the
  project's best exact transition matching yet.
- Implementing the actual VPT topology at 106M parameters — VPT-small in
  the tables above: raw 128×128 pixels, noncausal Conv3D front end,
  bidirectional Transformer — produced the strongest development result
  yet: 0.36 macro AP against 0.26 for the best similarly sized GRU, with
  five of seven keys improved. Its one blind spot — it never predicted
  the rare down key — is the gap the down-focused fine-tune in the tables
  was built to close
  ([full report](results/idm/VPT_SMALL_113M_RESULTS.md)).

### A new label channel: wild keyboard overlays

Existing harvesting reads gamepad widgets, so it cannot see the input
displays that keyboard players publish — including the runners at the top
of the game's leaderboards. Opening that channel meant building it end to
end: enumeration, a style survey, a decoder, calibration against engine
truth, and admission control.

- Enumerating the speedrun.com *Celeste* leaderboards found 7,071 fresh
  PC videos carrying 6,757 hours of footage. A stratified survey found
  about 15% show an on-screen input display — and the dominant style is a
  translucent action HUD, not the opaque key grid prior work assumes.
- Translucent overlays are decodable. Reading each key cell's contrast
  against its local background recovered engine truth without any
  supervision, matching 873 of 874 press onsets at a median offset of
  zero frames.
- Seven videos — just under 14 hours — have passed every admission gate
  so far, out of 33 media hours fetched. In a controlled test run before
  any hour was admitted, substituting decoded wild labels for a fifth of
  the training draws cost no measurable AP: the decoded labels are clean
  enough to train on.

The seven admitted videos now do double duty: they are the held-out test
set behind [the model comparison](#results), and that test has replaced
our small local capture session as the deciding measure of whether a
model is ready to label more Internet video
([the deployment-gate report](results/idm/VPT_SMALL_WILD_ADMITTED7_PRIMARY_GATE.md)).

#### What the human review looks like

![One row of a dash-offset review contact sheet](results/figures/fig_offset_review_row.png)

*Six consecutive masked gameplay frames around one claimed dash press,
from the timing-offset review of one harvested video. Celeste freezes the
engine for several frames when a dash starts, then motion bursts outward.
If the calibrated video-to-input offset is correct, the freeze sits
exactly where the labels claim — the three frames after the claimed press
identical — and motion returns on the fourth, here with the dash trail
visible. The reviewer scans twelve such rows per video and either sees
the pattern at the claimed offset or rejects the calibration.*

This contact sheet is the third of three human gates every video passes
before its labels can enter training: layout review (do the proposed
rectangles actually read the input display?), boundary review (are the
proposed ranges real gameplay?), and offset review (does the freeze
evidence support the measured video-to-input lag?). The layout gate's
evidence looks like this:

![An annotated layout-review frame](results/figures/fig_layout_review_geometry.png)

*The layout gate's geometry overlay: proposed gameplay viewport, timer
region, and per-key HUD cells drawn over the original stream layout.*

Every decision becomes an acceptance artifact that hash-binds the source
video, the evidence reviewed, and a named reviewer; automated stages
propose everything and can admit nothing. A
[complete boundary-review packet](results/wild20/ss3nhAUaScE/review_packet_boundaries/REVIEW.md)
is published so you can see exactly what a reviewer sees.

The gates also refuse. The largest reviewed video — nearly six cleanly
decoded hours, with layout and boundaries both human-approved — failed
the timing-offset gate on mixed freeze evidence, and no reviewer can
waive that: its hours stay out until an independent offset measurement
exists. The review sessions themselves are recorded in
[a start-to-finish walkthrough](docs/wild-review-walkthrough.md), and the
pipeline map is [harvest/README.md](harvest/README.md).

There is also a preservation argument for harvesting now: internet label
sources rot. At census time only 31% of the label-hours in the public
NitroGen *Celeste* release could still be retrieved from source. Every
unretrievable source was Twitch-hosted while all YouTube sources remained
live — consistent with
[Twitch's documented VOD retention windows](https://help.twitch.tv/s/article/video-on-demand)
— and the eight largest dead videos alone account for about 54 hours.

Complete metrics and caveats live in
[the technical report](report/README.md),
[the results summary](results/idm/SUMMARY.md), and
[the corpus audit](results/idm/CORPUS_AUDIT.md).

## Research principles

1. **Mask the answer key.** Anything on screen that reveals the inputs —
   frame counters, controller widgets, keyboard overlays — is removed
   before a frame reaches a model.
2. **Keep label kinds distinct.** Engine truth, mapped controller labels,
   and decoded overlay labels are never presented as equivalent
   supervision.
3. **Freeze roles before inference.** A session used for development
   never becomes a test; using an asset changes what can be claimed from
   it.
4. **Measure transitions, not just states.** Per-frame accuracy rewards
   copying the previous frame, so onsets and releases are scored
   directly.

## Quick start

The Python project is pinned to Python 3.12 and uses
[`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q
uv run python -m badeline.train --help
uv run python -m badeline.eval --help
```

The test suite passes on a fresh clone (808 passed, 59 skipped — the
skips need private artifacts). Reproducing the experiments end to end
requires data the repository does not distribute: a local copy of
*Celeste* with the Everest mod loader for capture, and the NitroGen
labels and source videos, obtained under their own terms, for the corpus
work.

The research roadmap is [PLAN.md](PLAN.md), measured status is
[PROGRESS.md](PROGRESS.md), and the full documentation index is
[docs/README.md](docs/README.md).

## Repository map

- `granny/` — the engine-truth instrumentation, an
  [Everest](https://everestapi.github.io/) mod that records the seven
  controls and player state per engine frame at 60 Hz and renders a
  machine-readable frame index for clock-free video alignment.
- `theo/` and `data/` — screen capture, frame-index decoding, session
  assembly and validation, answer-key masking, and shard construction for
  locally recorded sessions.
- `badeline/` — the models: visual encoders, GRU and VPT-style
  Transformer temporal stacks, training, and transition-aware evaluation.
- `nitrogen/` — source-video recovery, controller masking, action
  mapping, and quality metadata for the
  [NitroGen](https://huggingface.co/datasets/nvidia/NitroGen)
  action-label corpus.
- `harvest/` — the wild-overlay pipeline: recovering keyboard actions
  from the input displays speedrunners publish, behind human review gates
  that fail closed.
- `goldenberry/` — consistency checks for deciding whether a video and
  action record agree.
- `experiments/` and `results/` — frozen configurations, evaluation
  reports, hashes, and measured results.

## What is public

The public export contains the code, tests, specifications,
documentation, figures, and compact result reports — aggregate metrics,
run configurations, checkpoint hashes. Gameplay video, the NitroGen label
corpus, feature shards, model checkpoints, prediction arrays, session
manifests, and split lists remain in the private working repository.
Model weights are available on request against their tracked SHA-256
hashes.

Reruns are pinned end to end: `uv.lock` fixes the Python environment,
experiment JSON files fix architectures and optimization settings, run
directories retain configuration, logs, and evaluation sidecars, and
dataset membership and label provenance are explicit manifest fields. The
repository is published as a single squashed commit; commit identifiers
cited in pre-registration records refer to the private working history,
summarized in the
[build retrospective](docs/history/how-this-was-built.md).
