# A wild-data review session, start to finish

This is the record of one working session (2026-07-27) in which the wild
keyboard-overlay harvest moved from one human-accepted layout to seven, with
human-reviewed gameplay boundaries for four videos and a resolved diagnosis
of the last admission gate. It is kept as a walkthrough because the process
is the point: every claim about someone else's video crosses a human gate
before it can become training data, and this is what that looks like in
practice.

*Screenshots referenced below are curated single-frame exhibits from public
speedrun videos, credited per the third-party notices; the full evidence
packets live in the working repository.*

## The setup

Eleven speedrun videos (27.4 nominal hours) were fetched and byte-verified.
Admission requires three human-signed, hash-bound artifacts per video —
layout acceptance (do the proposed cell rectangles read the input display?),
boundary acceptance (are the proposed gameplay ranges believable?), and
offset acceptance (is the video-to-input lag measured and supported?) — on
top of mechanical QC that no reviewer can waive. At session start: one
layout accepted, zero boundaries, zero offsets, zero admitted hours.

## Round one — four layout decisions (~5 minutes of reviewer time)

The assistant surfaced each packet's geometry overlay and per-cell
pressed/released evidence; the reviewer decided:

- `nRMVyWdNsTo` — approved, choosing between two prepared layout variants:
  the HUD shows a `Demo` cell, and the packet asked whether Demo is an
  alternate dash binding. It is (a demodash is a dash), and the packet's
  semantic-context sheet supported it. Ambiguity resolved by construction:
  both variants were prepared, the human picked exactly one.
- `Y6AeZFCU4LY`, `6vEpVqbrvSE` — approved.
- `b43KAaem61g` — approved; its packet was assembled during the session
  from existing evidence, which exercised the packet rules (self-contained
  directories, packet-relative paths, hash-bound exemplar frames — three
  schema corrections before the manifest validated, each a fail-closed
  refusal doing its job).

## Round two — boundaries (four approvals)

Gameplay ranges come from official run-timer activity; proposals abstain
until a human reviews them. Approved: nRM (669 ranges, 80.7% coverage,
corrected start edge), Y6 (111 ranges, LiveSplit-corroborated), 6vE (150
ranges), and v498's separate 212-range packet — the last approved
explicitly **on sampled diagnostics**, with the sampling basis written into
the approval statement naming the proposal and packet hashes. Each approval
was materialized as human-reviewed boundaries binding reviewer identity,
source hash, and the evidence consulted.

## Round three — two more layouts

Packets for `kdQbIoMxzZw` and `ss3nhAUaScE` were assembled mechanically to
the documented rules and approved. The ss3 approval illustrates the gate
separation: its layout is correct, but its decode shows 8–19%
single-frame-run rates on every action — a decoder-quality question that
layout approval deliberately cannot bypass; mechanical QC still holds its
data out.

## Round four — boundaries again, and the reviewer pushes back

Two more boundary sets: b43 approved from its annotated trace, but for kd
the reviewer rejected the first evidence outright — a raw timer crop with
no annotation — and asked for the source video link and explicit
instructions. The re-review worked from an annotated activity trace plus
five timestamped spot-checks in the actual video (three proposed ranges,
two excluded gaps). All checks matched. Then a second correction, this
time caught by the assistant: the reviewed visualization had drawn only
256 of 325 ranges (a truncated diagnostic list), so the complete set was
re-rendered with the omitted 69 highlighted and the approval reconfirmed
against it. The recorded acceptance carries both corrections in its notes.
Two lessons entered the guide: review images must be annotated evidence,
not raw crops, and any visualization drawn from a diagnostic list must
assert it is complete against the authoritative count.

## Round five — the largest video, unblocked then approved

`ofy37Fm6EgI` (5.77 decoded hours, the old corpus's largest) began the day
unreviewable: no exemplar frames had ever been extracted, and its layout
carries an AI inference confidence of 0.78 against the 0.80 admission
threshold — both facts stated at the top of its packet. Unblocking meant
re-fetching the 5.5 GB source from durable storage (hash-verified),
extracting thirteen exact frames at evidenced timestamps, and verifying
each frame's cell luminance against the recorded evidence (max deviation
0.17 of 255). The reviewer approved the layout with the Demo cell bound as
dash — the same explicit decision the nRM packet required — and then the
910-range boundary set against source-video spot checks, with the
complete-set count asserted at render time (the lesson from round four,
now a rule).

The 0.78-confidence flag needed one more step. Layout approval alone does
not clear it: the decode QC policy enforces the 0.80 floor unconditionally,
and the packet required a conscious recorded decision rather than a value
discovered at decode time. The owner ruled that a hash-bound human layout
acceptance supersedes the automatic confidence estimate, and the ruling is
itself an artifact: `layout_confidence_override.json` in the packet,
sha-bound to the exact layout and acceptance files, human-reviewer-only,
with the rationale in the record. The decode honors it through an explicit
override note in the report — the gate shows as overridden, never as
passed.

## The offset gate: a diagnosis instead of a rubber stamp

With layouts and boundaries accumulating, the remaining blocker was the
compositor-offset calibration — the dash-hitstop fingerprint had failed its
automatic physics gates on every video tried. The investigation's decisive
step was running the unmodified production scorer on a local
engine-instrumented session where the offset is zero by construction. It
recovered zero exactly — and still failed the gates, which demanded
per-event exact-frame agreement that capture sampling makes physically
unattainable (a 3-engine-frame freeze lands on 3 video frames only 42% of
the time), over an event population diluted by dash-key presses that
trigger no dash (22% even in calm play; far more at speedrun press rates).

The recommended revision — count only strong events, score agreement within
±1 frame, record ±1 frame as the offset uncertainty — passes ground truth
cleanly and was validated there before being proposed. The reviewer
approved it after a plain-language walkthrough of the evidence; it was then
implemented as an explicitly versioned policy (v2), with the ground-truth
jitter and contamination findings encoded as regression tests — including a
test that a genuinely split cohort still fails, because a gate revision
that cannot fail is not a gate. One wild video is expected to remain
genuinely uncertain under the revised gates, which is the correct outcome
if its evidence is genuinely mixed.

## Scoreboard after the session

| gate | before | after |
|---|---|---|
| layouts human-accepted | 1 / 11 | 8 / 11 |
| boundaries human-reviewed | 0 | 7 videos |
| offsets measured/accepted | 0 | 0 (gate revision proposed, validated on ground truth) |
| admitted hours | 0 | 0 — by design, until all three gates close |

The reviewer's total time across all rounds was under twenty minutes; the
assistant's job was to make each decision a one-look judgment with the
evidence already assembled, and to record every decision as an artifact
that verifies from a clean clone.

## Round six — the offset gate closes (2026-07-28)

The next day the proposed gate revision landed as an explicitly versioned
OffsetPolicy v3, its one substantive change justified by the ground-truth
diagnostic: on engine-truth sessions whose true offset is zero by
construction, the per-event collar fraction swung 0.63–0.93 — it measures
footage SNR, not offset correctness — so v3 records it but no longer blocks
on it, while the winner, margin, bootstrap, and temporal-block gates are
unchanged. The six production calibrations were re-verdicted from their
hash-verified serialized statistics without reprocessing video, and the
owner then reviewed the six ranked contact sheets, one per video, under one
instruction: each row shows six masked gameplay crops around one claimed
dash press — `g-1`, `g`, three frames that must be frozen, and a `g+4`
rebound — approve only if the freeze sits where labeled, row after row.

The outcome is two-tier by design. `kdQbIoMxzZw` and `Y6AeZFCU4LY` passed
every blocking gate; `nRMVyWdNsTo`, `6vEpVqbrvSE`, `v498642684`, and
`b43KAaem61g` landed `uncertain_adjacent` — bootstrap-decisive,
block-unanimous winners with a thin non-adjacent margin — so their
acceptances required the explicit uncertain-tier flag and carry a recorded
±1-frame offset uncertainty into the generated layouts. All six acceptances
were recorded 2026-07-28, hash-bound to the calibration, the contact-sheet
bytes, the layout-review packet, and the reviewer identity. The offset row
of the scoreboard above now reads six of six reviewed; admitted hours
remain zero until decode QC and the publication manifest close.
