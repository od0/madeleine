# ss3nhAUaScE gameplay-boundary human review

Status: **awaiting a named human decision**. Nothing in this packet is
an acceptance, a reviewed boundary set, or train-ready data.

## What you are deciding

Whether to adopt the 106 AI-proposed half-open gameplay
ranges in `source_artifacts/boundaries.v3-ai.json` (byte-identical
copy of `results/wild20/ss3nhAUaScE/boundaries.v3-ai.json`) as
human-reviewed boundaries:

- 6,035.4 s = 1.6765 h of proposed gameplay
  inside the reviewed wall-clock envelope [12.233333,
  6867.083333) s;
- the proposal's signal-quality gates all PASSED; it abstained only
  because its ROI and wall-clock inputs were reviewed by an AI agent,
  not a human (`review_provenance_gate_passed: false`) — exactly the
  gap this review closes;
- ranges come from official in-game total-timer activity: the timer
  advances during play and freezes/disappears on loads, maps, menus,
  and after the finish.

## Review checklist (from the recorded proposal)

1. Confirm the normalized ROI contains only the official advancing timer.
2. Inspect every suggested start and end against raw source frames.
3. Confirm hitstop/noise bridges are brief gameplay freezes, not loads or menus.
4. Confirm every long frozen or absent-timer range remains excluded.
5. Copy ranges into WildBoundaries only after naming a human reviewer.

The wall-clock envelope edges themselves (LiveSplit 0.00 -> 0.01 at
12.233/12.250 s; final official value first shown at 6867.083 s) are
evidenced by the four exact frames in `source_artifacts/wall_clock/`,
hash-bound in the layout evidence manifest.

## Completeness assertions (all enforced by the generating scripts)

- `boundaries.v3-ai.json` holds exactly 106 ranges; the
  proposal's candidate accounting records the same total and is NOT
  truncated (`candidate_ranges_before_gates: rows 106, total 106,
  truncated false`).
- `boundary_review_full.png` asserts `len(ranges) == 106` against the
  authoritative artifact at render time and draws every range twice:
  as a shaded span on the full-video trace and as one bar per range
  in the indexed duration panel.
- The full-resolution scalar trace behind the image was
  independently re-decoded locally from the hash-verified source
  video with the same OpenCV pipeline (decoder builds are not
  bit-portable across hosts, so agreement is tolerance-bounded and
  the measured deviations are recorded): all 512 recorded diagnostic
  samples agree within max change-score deviation 0.0024 and
  max mask-mean deviation 5.74 of 255, and an independent
  re-segmentation reproduces the authoritative range set:
  104/106 ranges float-exact, max edge deviation
  0.0167 s, and 1 near-threshold split/merge
  flip (each recorded in `inputs/trace_verification.json`).
- The proposal's bridge diagnostic list IS truncated (256 of its
  recorded 52,898 rows; the independent re-decode finds
  52,804), so the complete bridge population was recomputed; the
  99 risky bridges
  (absent_timer_frames > 0 or duration_s >= 0.35) are bound in
  `inputs/bridge_risk_population.jsonl`.

## Spot-check evidence

Sampled ranges (deterministic: earliest, latest, longest, shortest,
second-longest, plus 8 random with seed
20260728): every page shows six exact frames — three at the
start edge (timer must begin advancing) and three at the end edge
(timer must freeze or disappear). Every displayed edge frame's
timer-ROI scalars were re-measured and equal the verified trace, so
the pages cannot silently show the wrong frame.

| page | range | why | interval (video time) | duration | online check |
|---|---|---|---|---|---|
| `spot_checks/range-000.jpg` | 0 | earliest range | 0:00:13.867 .. 0:00:39.617 | 25.75 s | [start](https://youtu.be/ss3nhAUaScE?t=13) [end](https://youtu.be/ss3nhAUaScE?t=39) |
| `spot_checks/range-001.jpg` | 1 | random sample | 0:00:54.717 .. 0:01:35.183 | 40.47 s | [start](https://youtu.be/ss3nhAUaScE?t=54) [end](https://youtu.be/ss3nhAUaScE?t=95) |
| `spot_checks/range-015.jpg` | 15 | random sample | 0:08:36.833 .. 0:09:05.367 | 28.53 s | [start](https://youtu.be/ss3nhAUaScE?t=516) [end](https://youtu.be/ss3nhAUaScE?t=545) |
| `spot_checks/range-049.jpg` | 49 | second-longest range | 0:39:02.000 .. 0:43:06.483 | 244.48 s | [start](https://youtu.be/ss3nhAUaScE?t=2342) [end](https://youtu.be/ss3nhAUaScE?t=2586) |
| `spot_checks/range-060.jpg` | 60 | random sample | 0:58:06.950 .. 0:59:15.867 | 68.92 s | [start](https://youtu.be/ss3nhAUaScE?t=3486) [end](https://youtu.be/ss3nhAUaScE?t=3555) |
| `spot_checks/range-062.jpg` | 62 | random sample | 1:01:47.883 .. 1:03:40.450 | 112.57 s | [start](https://youtu.be/ss3nhAUaScE?t=3707) [end](https://youtu.be/ss3nhAUaScE?t=3820) |
| `spot_checks/range-063.jpg` | 63 | random sample | 1:03:41.350 .. 1:04:10.167 | 28.82 s | [start](https://youtu.be/ss3nhAUaScE?t=3821) [end](https://youtu.be/ss3nhAUaScE?t=3850) |
| `spot_checks/range-082.jpg` | 82 | shortest range | 1:21:24.917 .. 1:21:27.050 | 2.13 s | [start](https://youtu.be/ss3nhAUaScE?t=4884) [end](https://youtu.be/ss3nhAUaScE?t=4887) |
| `spot_checks/range-094.jpg` | 94 | random sample | 1:34:43.100 .. 1:35:00.283 | 17.18 s | [start](https://youtu.be/ss3nhAUaScE?t=5683) [end](https://youtu.be/ss3nhAUaScE?t=5700) |
| `spot_checks/range-096.jpg` | 96 | random sample | 1:35:18.650 .. 1:35:29.183 | 10.53 s | [start](https://youtu.be/ss3nhAUaScE?t=5718) [end](https://youtu.be/ss3nhAUaScE?t=5729) |
| `spot_checks/range-099.jpg` | 99 | random sample | 1:37:20.233 .. 1:37:28.433 | 8.20 s | [start](https://youtu.be/ss3nhAUaScE?t=5840) [end](https://youtu.be/ss3nhAUaScE?t=5848) |
| `spot_checks/range-105.jpg` | 105 | latest range (also the longest) | 1:47:06.883 .. 1:54:27.083 | 440.20 s | [start](https://youtu.be/ss3nhAUaScE?t=6426) [end](https://youtu.be/ss3nhAUaScE?t=6867) |

Excluded-gap checks (timer must NOT advance anywhere inside):

| page | between ranges | interval | duration |
|---|---|---|---|
| `spot_checks/gap-after-range-085.jpg` | 85 and 86 | 1:28:05.717 .. 1:31:05.317 | 179.60 s |
| `spot_checks/gap-after-range-076.jpg` | 76 and 77 | 1:15:57.650 .. 1:16:50.283 | 52.63 s |

Included-bridge checks (frames inside proposed ranges where the timer
was briefly not advancing; must be gameplay freezes, not loads or
menus):

| page | inside range | at | duration | absent-timer frames |
|---|---|---|---|---|
| `spot_checks/bridge-001835500ms.jpg` | 38 | 0:30:35.500 | 500 ms | 29 |
| `spot_checks/bridge-002374783ms.jpg` | 49 | 0:39:34.783 | 500 ms | 29 |
| `spot_checks/bridge-003144566ms.jpg` | 56 | 0:52:24.567 | 500 ms | 29 |

## Flagged for attention

- **Range 82 (shortest, 2.13 s at 1:21:24.9)**: its frames sit
  inside a black screen-wipe transition — the gameplay viewport is
  mostly covered, and the file timer is only partially revealed
  (`47.926` with the hour/minute glyphs still occluded) while its
  visible digits advance. Look at `spot_checks/range-082.jpg` and
  decide whether this 2-second transition sliver is gameplay worth
  keeping; rejecting it costs 2.1 s.
- **30 excluded gaps are barely over the 0.5 s
  bridge limit** (typically 0.517 s): freezes of this length are
  bridged when <= 0.5 s but split ranges when slightly longer, so
  several adjacent ranges (for example 81/82/83) are conservative
  splits of continuous play, not content changes. Approving the
  split ranges as proposed only drops the frozen frames themselves.
- **Range 105 is both the latest and the longest (440.2 s)** and
  ends exactly at the reviewed wall end: its end edge is the
  final-timer-value frame evidenced in `source_artifacts/wall_clock/`.
- **The first ~1.6 s of play (12.25-13.87 s) are excluded** because
  the official timer is not yet visible right after the LiveSplit
  start; the proposal is conservative at the wall start.
- **12 ranges are shorter than 6 s** (minimum 2.13 s;
  policy floor 2.0 s) — brief gameplay slivers between deaths,
  menus, or transitions. The shortest is spot-checked above.
- The very large bridge population (52,804 gaps) is the normal
  signature of this change-score detector (kdQbIoMxzZw showed the
  same raw-to-bridged pattern); only
  78 bridges
  contain any timer-presence dropout. **The riskiest bridges are
  full-screen wipe transitions (death/room changes) of exactly
  500 ms with the timer absent for ~29 frames** — included inside
  ranges by the bridging rule, as in the previously approved
  videos; the top three are rendered above.

## What approval unlocks

Adopting the ranges closes the boundary gate for 1.68
proposed gameplay hours: it authorizes decoding labels inside these
ranges and running the dash-hitstop offset calibration for this
video. It does NOT admit training data by itself — the offset gate
and decode-time QC remain, and this video's current decode fails
action QC (8-19% single-frame-run rates), which no boundary approval
can waive.

## Recording a decision

Do not approve by implication and never edit `human_reviewed` flags.
Either list corrections/rejected range indices, or approve with an
explicit statement naming both hashes, for example:

> I approve all 106 ss3nhAUaScE candidate half-open ranges from
> proposal `d242bdc81a5101d9737e78395433501ba86e19f8bf220647f9d281616dd2ff6e`
> after reviewing packet manifest `<review_manifest.json sha256>`.

A separate step then materializes
`boundaries.human-reviewed-<date>.json` with `reviewer_kind`
`human_with_ai_assistance`, binding the reviewer name, the source
hash, and the evidence consulted (as in the seven previously
reviewed videos). This packet itself records no decision.
