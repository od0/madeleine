# Wild-harvest human review guide

Wild-decoded labels enter training only through human-signed, hash-bound
acceptance artifacts. This document describes what a reviewer looks at, how a
decision is recorded, and the packet rules the tooling enforces — with the
2026-07-27 review session as the worked example.

## Why the gate exists

Every automatic stage of the wild pipeline (layout inference, timer-boundary
proposal, offset calibration, decode QC) can propose but never admit.
`reviewer_kind` is recorded on every acceptance; only `human` or
`human_with_ai_assistance` satisfies the admission gates, and a hand-edited
`human_reviewed` flag inconsistent with `reviewer_kind` is rejected at load.
The point is that a claim about someone else's video becomes training data
only after a person has looked at the evidence that supports it.

## The three human decisions per video

1. **Layout acceptance** — do the proposed cell rectangles read the input
   display, and does each cell's pressed/released evidence look right?
   Evidence: `geometry.png` (rects over a real frame), `cell_states.png`
   (per-cell pressed/released strips), and exact per-frame exemplars under
   `frames/`. Recorded by `accept_wild_layout accept`.
2. **Boundary acceptance** — are the proposed gameplay ranges (from official
   run-timer activity) believable? Evidence: `timer_boundaries.png`,
   `timer_review.json`, and for large range sets a dedicated range-review
   packet with edge and bridge pages. Recorded via `wild_boundaries` with a
   human `reviewer_kind`.
3. **Offset acceptance** — does the dash-hitstop contact sheet support the
   measured video-to-input lag? Evidence: the calibration's contact sheet
   and per-event table. Recorded by `accept_wild_offset` (requires a passing
   production calibration; the automatic physics gates are re-checked at
   acceptance time and cannot be waived by the reviewer).

A reviewer can only unlock what the mechanical gates already tolerate:
decode-time QC (cell separation, transition-rate, cadence) still rejects
independently of any approval, which is what caught the stuck
`bottom_dash` cell on `v1509603803` after its layout looked plausible.

## Review packets are self-contained

`accept_wild_layout manifest` enforces packet hygiene, learned the hard way:

- every referenced file (draft layout, three artifact roles, all evidence
  frames) must live **under the manifest's directory**;
- manifest entries store **packet-relative paths**, so the packet verifies
  from a clean clone with no repository context;
- cell-state evidence must be the v2 schema — `cells[]` of
  `{cell_id, action, pressed: {path, sha256}, released: {path, sha256}}` —
  and each referenced frame must be hash-bound among `evidence_frames`
  (path *and* sha must match);
- required artifact roles: `geometry_overlay`, `cell_state_evidence`,
  `cell_state_contact_sheet`;
- `accept` refuses to overwrite outputs, refuses an output layout at the
  draft's path, and writes the acceptance beside its manifest.

## Worked example: the 2026-07-27 session

Four packet-ready layouts were reviewed in one sitting (reviewer: Bryan,
`human_with_ai_assistance`; the assistant surfaced evidence and recorded
decisions, the human decided):

| video | decision | note |
|---|---|---|
| `nRMVyWdNsTo` | approved, **with-Demo-as-dash variant** | the packet offered two mutually exclusive layouts; a demodash is a dash, and the semantic-context sheet supported binding the `Demo` cell to dash |
| `Y6AeZFCU4LY` | approved | |
| `6vEpVqbrvSE` | approved | |
| `b43KAaem61g` | approved | packet assembled at review time from existing evidence (see hygiene rules above); its decode still fails action QC, so approval unlocks a re-decode, not hours |

The variant question is the template for future ambiguous bindings: put both
layouts in the packet, name the question in `REVIEW.md`, and let the human
pick exactly one.

## Recording a decision

```bash
uv run python -m harvest.accept_wild_layout accept \
  --review-manifest <packet>/review_manifest.json \
  --draft-layout <packet>/layout.draft.json \
  --output-layout <packet>/layout.reviewed-unmeasured.json \
  --acceptance-out <packet>/layout_acceptance.json \
  --reviewer "<name>" --reviewer-kind human_with_ai_assistance --approve
```

The acceptance binds the source-video hash, the manifest hash, the draft
hash, every artifact and frame hash, and the output layout's hash, so any
later mutation of any input is detectable.
