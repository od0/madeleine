# Why NitroGen controller data needs action mapping

MADELEINE's second supervision channel is NVIDIA's NitroGen dataset, which
annotates public gameplay videos with per-frame gamepad state. A natural
first question — and a fair challenge — is why any inference is needed at
all: does the dataset not say which key means what? This page explains what
the data actually contains, why a button-to-action dictionary cannot exist
at the dataset level, and how MADELEINE bridges the gap in a way that is
measured, versioned, and independently verified.

## What the dataset provides

NitroGen's schema is documented and precise: seventeen boolean columns
naming positions on a standard gamepad (`south`, `west`, `north`, `east`,
the d-pad, shoulders, triggers, thumbs, start/back/guide) plus two sticks
as `x,y` pairs in `[-1, 1]`, one row per video frame. Here is half a second
of real data from one Celeste video in the training corpus:

```text
  row south  west r_trig dp_up dp_dn  j_left_x  j_left_y
   16     1     0      1     0     0     0.683     0.005
   17     1     0      1     0     0     0.685     0.004
   18     1     0      1     0     0     0.682     0.004
   19     0     0      1     0     0     0.683     0.005
   ...
   26     0     0      1     0     0     0.685     0.009
   27     0     1      1     0     0     0.686     0.007
   ...
   33     0     1      1     0     0     0.683     0.006
```

A Celeste player reads this instantly: the stick is held right, grab is
held on the right trigger, a three-frame `south` tap is a jump, and the
`west` press at row 27 is a dash. Nothing in the data says any of that.
The columns record which physical buttons are down, never what they do.

## Why no semantic dictionary can ship with the data

Three independent reasons, in increasing order of subtlety.

First, NitroGen spans more than a thousand games. The same physical button
is jump in one game, roll in another, brake in a third. Button-to-action
semantics are per-game knowledge the dataset never claims to carry.

Second, even within one game the mapping is a player setting, and players
genuinely differ. Among the 210-video Celeste training population, the
videos whose bindings can be inferred confidently span roughly two dozen
distinct layouts: the plurality use the default jump-on-south,
dash-on-west, grab-on-a-trigger arrangement, but there are clusters with
jump on a shoulder and grab on `south` (a common speedrunner claw grip),
jump on `north`, and dash on a trigger. The same physical button flatly
changes meaning across videos: in one video `south` is pressed ten
thousand times in twelve-frame taps (a jump); in another it is held a
third of the entire video (a grab). Any fixed dictionary is guaranteed
wrong for a large minority of the corpus.

Worse for simple inference, Celeste's default layout is itself
multi-bound: two buttons for jump, two for dash, and four for grab are
live simultaneously, and players drift between them. One corpus video
switches its dominant dash button from `west` to `east` partway through.
An assumption of one button per action is therefore wrong even for
default-settings players, and the mapper models button sets, not single
buttons.

Third, the rows are not controller telemetry. NitroGen decodes gamepad
state from the controller overlay rendered on the player's stream, using a
learned model whose published accuracy is roughly 0.96 per button-frame.
At that rate a ninety-minute video carries tens of thousands of wrong
button-frames, and the corpus shows the expected impossibilities: rows
where d-pad up and down are pressed simultaneously, and one-frame phantom
presses on buttons the player never used. Semantic mapping therefore has
to be statistical inference over noisy evidence, not a lookup.

## What upstream does provide

Two things, and both are used.

For the four directions the dataset carries real semantics: the d-pad
columns name their own meaning, and the card fixes the stick coordinate
contract — `(-1, -1)` is the upper-left corner, so negative Y means up.
MADELEINE maps directions directly and invariantly under that contract,
which was independently re-verified against NVIDIA's own execution code
and against camera motion in the gameplay pixels after an early mapper
version wrongly allowed sparse per-video evidence to override it.

For the three bound actions, about half the chunks ship an optional
`actions_processed.parquet` in which upstream has moved each video's
buttons onto canonical default-layout positions, merged multi-bound
siblings, and dropped presses its quality filter rejects — including the
one-frame phantoms. This is the closest thing to an upstream answer key,
and it is valuable, but it is an inference on their side too: it covers
only 63 of the 210 training videos (coverage is all-or-nothing per video),
and on some videos it disagrees with MADELEINE's inference. It began as an
independent cross-check oracle — agreement as mutual confirmation,
disagreement queueing a video for visual review. After that review round,
the resolved-v3 bind sets adopt the upstream-implied binding wherever the
two disagree (the owner's upstream-preference policy, with those entries
marked pending final human review); everywhere upstream has no coverage,
MADELEINE's per-action inference remains the source.

## How MADELEINE maps jump, dash, and grab

Each video's bindings are inferred from behavioral signatures measured
over the whole video: press frequency, presses per hour, median hold
duration, the fraction of very short presses, and co-press rates with
directions. Jump favors frequent short-to-medium taps; dash favors
frequent, directional, short-to-held presses; grab favors long holds that
co-occur with upward input.

The inference is deliberately conservative. A button competes for an
action only after clearing an absolute press floor, a presses-per-hour
floor, and a coarse shape gate, so a handful of decoder phantoms can never
outscore a real button. Multi-bound defaults are modeled by composite
candidates — the sibling pairs for jump and dash and the shoulder/trigger
group for grab — and the joint assignment keeps button sets disjoint while
maximizing evidence explained, with a penalty for leaving a heavily used
button unaccounted for. Every action gets its own confidence and flag; a
flagged action falls back to the game's default binding for that action
alone, with buttons already claimed by another action removed. Mapping
reports publish the full evidence, the per-action decisions, and the
direction-rule contract, so every label is traceable to its justification.

## Why the result can be trusted, and how far

Mapped labels are a declared supervision tier: they measure agreement with
an inferred per-video mapping and are never presented as engine truth.
Within that tier, the mapping has been defended three ways. The direction
convention is pinned to the upstream contract and was confirmed by four
independent evidence classes, including frame-level motion checks of the
training pixels. The bind inference is validated by replaying it against
every video's published evidence and by onset-triggered visual review:
strips of gameplay around actual presses of a claimed dash button show
dashes. And the upstream canonicalization, where it exists, provides a
second independent opinion; the videos where the two inferences conflict
are enumerated and held for review rather than silently trusted.

The failure modes found along the way — a vertical-axis inversion in 22
videos, a dash-assignment defect that starved 13 videos of dash labels,
and the single-bind assumption itself — were each caught by audit,
quantified against the raw data, fixed in a versioned mapper, and
re-verified before any corrected corpus was published. The measured
records behind every number in this page live in the working repository's
results tree alongside the mapper in
[nitrogen/map_actions.py](../nitrogen/map_actions.py) and its tests.
