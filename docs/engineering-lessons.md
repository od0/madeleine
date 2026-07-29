# Engineering lessons

MADELEINE's working repository keeps detailed chronological notebooks. This
page is the smaller public record: durable rules that changed how the system is
built or how its results are interpreted, without machine coordinates, raw
operational narration, or private data identities. Each lesson condenses one
or more dated incidents; the chronological findings and engineering logs that
record them remain in the private working repository and are not part of the
public export.

## Evidence must survive a spot check

An early delegated reconnaissance report (2026-07-23) contained plausible but
false example IDs and cited tooling absent from the machine. Direct inspection
rejected its load-bearing claims. Since then, no delegated measurement enters
a dataset card or result until its underlying metadata, pixels, or arrays have
been checked directly.

## A queued job is not a monitored job

Durable processes are not the same as a promised result. Every unattended run
needs a completion marker, failure log, expected duration, monitoring mechanism,
and an explicit owner for the next wake-up and analysis.

## Compare support before scores

A long-context model once appeared dramatically better because its continuity
requirements left only 1,125 easy development frames, versus 29,086 for the
shorter model. Model comparisons now begin with frame, stream, and session
support and are rescored on identical targets whenever input geometry changes
eligibility.

## Semantic fixes create new experimental generations

Changes to target alignment, window assembly, feature deltas, masking, loss, or
evaluation make old and new checkpoints non-identical experimental generations.
A causal comparison requires rerunning the matched baseline; until then, the
result is labeled diagnostic and the confound is named.

## State recognition and transition timing are separate objectives

End-to-end visual learning improved action-state average precision while exact
onset/release F1 fell. Every model report therefore pairs state AP/F1 with exact
and tolerant transition metrics, and checkpoint selection must name which
metric family it optimizes.

The matched corpus-scale result reinforces the split: increasing mapped-video
training from about 38 to 103 and 148 hours raised held-out AP from 0.2435 to
0.2693 and 0.2723, while exact event F1 stayed near 0.01. More supervision
improved state ranking; it did not repair action-boundary timing.

## Temporal continuity belongs in the data contract

Adjacent stored rows are not necessarily adjacent game frames. Windows may not
cross missing chunks, engine-frame gaps, capture resets, or declared sequence
boundaries. Variable-rate video is sampled by timestamp onto the label grid;
container metadata alone never establishes a 60 Hz timeline.

## Verify conventions on pixels

Plausible metadata can mix coordinate conventions. Controller rectangles,
viewport transforms, mask geometry, and axis signs are checked against decoded
frames before they can authorize training data. A validator should fail closed
rather than repair an uncertain record silently.

## Mask the measured leak surface

One mask passed its own assertions while leaving a readable strip outside the
declared rectangle (found 2026-07-26 in the own-data input-overlay masks): the
check enforced the wrong boundary. The corrected policy is that mask
verification requires an outside-rectangle leak scan and a downstream
exploitability check, not merely a zero-valued declared region. As of
2026-07-27 the scan, corrected manifest geometry, fail-closed guard, and shard
rebuild are complete. Earlier own-data shards contain the leak; no transferable
benefit was observed on held-out sessions, but this does not rule out training
distortion, and corrected model reruns remain queued.
Status is tracked in [PROGRESS.md](../PROGRESS.md).

## Separate data-volume counters

Source bytes, nominal label-hours, decoded media hours, metadata-valid hours,
and train-ready hours answer different questions. The pipeline records each
stage independently; only artifacts that pass the complete admission contract
count as training data.
