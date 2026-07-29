# harvest module map

`harvest/` is the wild input-overlay acquisition and review pipeline: it turns
public speedrun videos with visible input HUDs into masked, train-ready label
shards. Every automatic stage — probe triage, layout inference, timer-boundary
proposal, offset calibration, decode QC — can propose but never admit.
Training admission requires source-bound evidence, hash-bound acceptance
artifacts with an allowed human `reviewer_kind`, passing mechanical decode QC,
and a complete publication manifest; an editable boolean is never a review
record. The directory accumulated many scripts because each acquisition
campaign froze its scheduling and diagnostic tooling in place for provenance.
This map separates the pipeline a reader needs from the tooling that supports
it. The reviewer workflow is documented in [review/README.md](review/README.md);
the own-capture overlay contract is [specs/overlay_spec.md](../specs/overlay_spec.md).

## The load-bearing pipeline

These files, in order, are the v1 wild pipeline from candidate discovery to
published shards. Each stage's output is the next stage's hash-bound input.

1. `speedrun_index.py` — enumerates speedrun.com Celeste runs into a
   PC-only fetch-candidate list, before anything is downloaded.
2. `scan_corpus.py` — probes every candidate with a cheap low-resolution
   section (`overlay_probe.py`) and saves panel crops; `classify_panels.py`
   OCR-classifies which crops are decodable action HUDs.
3. `fetch_wild.py` — polite, idempotent full-video fetch with a PTS/timestamp
   audit; the media timeline, not container metadata, is the timing authority.
4. `survey_wild_layout.py` — sparse 16-frame layout survey and contact sheet,
   bound to the immutable fetch and PTS evidence.
5. `accept_wild_layout.py` — hash-bound review packet manifest and the human
   layout acceptance; the layout contract itself is `wild_layout.py`
   (normalized, versioned geometry as data, not code).
6. `extract_timer_trace.py` — streams a reviewed run-timer ROI into scalar
   activity evidence; `timer_activity.py` turns the trace into suggested
   gameplay ranges.
7. `wild_boundaries.py` — the reviewed wall-clock and gameplay-range contract;
   records the human boundary decision with explicit reviewer kind.
8. `decode_wild.py` — decodes, PTS-aligns, quality-scores, and masks the HUD;
   per-cell decoding of alpha-blended overlays lives in
   `translucent_parser.py`.
9. `calibrate_offset.py` — measures the video-to-input temporal offset from
   dash-hitstop physics; the result is deliberately left pending review.
10. `accept_wild_offset.py` — the human offset acceptance; the automatic
    physics gates are re-checked at acceptance time and cannot be waived.
11. `build_wild.py` — builds masked, train-ready shards from admitted labels,
    splitting at activity boundaries so windows cannot bridge menus or pauses.
12. `publish_wild_derived.py` — publishes one video's derived artifacts with
    SHA-256 readback, completion marker last.

## Review-round tooling

Human decisions are recorded through three CLIs, described with a worked
example in [review/README.md](review/README.md):

- `accept_wild_layout.py` — `manifest` assembles and verifies a
  self-contained, packet-relative review packet; `accept` writes the reviewed
  layout and its acceptance without overwriting either input.
- `accept_wild_offset.py` — bridges a pending offset calibration to a measured
  layout through a hash-bound acceptance artifact.
- `accept_layout_confidence.py` — records an explicit human ruling that a
  layout's below-floor inference confidence may stand, as an artifact rather
  than a policy edit.

## Batch orchestration and recovery

The shell runners and queue compilers in this section are campaign
orchestration retained for provenance; they remain in the private working
repository and are not part of the public export. Their contracts are
described here because the committed manifests and logs reference them.

Shell runners and queue compilers for the serial acquisition lanes. All of
them respect the same durability rule: completion is the immutable R2
`upload_complete.json` marker, never a local file's presence.

Fetch lanes:

- `worker_wild.py` — one-candidate fetch worker with verified R2 handoff and
  SHA-256 readback.
- `fetch_fleet_worker.py` — serial consumer of a completion-gated fleet queue.
- `repartition_fetch_queues.py` — freezes and rebalances fetch queues at a
  completion-marker barrier.
- `repartition_chained_recovery.py` — compiles fail-closed successor queues
  for chained recovery.
- `run_fetch_recovery_durable.sh` — runs one recovery queue under a host lock
  with durable start/finish markers.
- `chain_fetch_recovery_durable.sh` — chains a missing-only queue after a
  durable predecessor exits.
- `run_fetch_after_pid.sh` — starts a queue when the current serial lane's
  process exits.

Layout-survey lanes:

- `partition_layout_surveys.py` — freezes and balances a survey wave from
  raw-complete nominations.
- `run_layout_survey_queue.sh` — consumes one immutable survey queue and
  publishes with SHA readback.
- `run_layout_survey_watch.sh` — continuously surveys raw-complete videos not
  yet complete in R2.
- `run_layout_survey_after_pid.sh` — runs survey generation after an
  acquisition lane exits.

Probe and VLM classification lanes:

- `probe_fleet_worker.py` — serial probe worker publishing immutable per-attempt
  R2 checkpoints.
- `run_layout_vlm_watch.sh` — long-running watch that mirrors completed
  surveys and classifies each source exactly once.
- `launch_probe_vlm_shard.sh` — gates one probe-classifier shard on fully
  mirrored inputs.
- `launch_campaign_vlm_shard.sh` — runs campaign classifier shards
  back-to-back without overlapping the frozen legacy shard.
- `launch_campaign_followup_vlm.sh` — chains a follow-up classifier shard on a
  predecessor's completed output.

Family-transfer and publication lanes:

- `run_layout_family_worker.sh` — end-to-end AI-only family transfer, cell
  scan, and decode for one video.
- `run_validated_scan_assignment.sh` — hash-validated full-cell scan
  assignment for one video.
- `run_family_publication_watch.sh` — watches one family-transfer build and
  publishes it on completion.

## One-shot and diagnostic scripts

The remaining scripts are point-in-time tools from dated acquisition
campaigns — probe triage, VLM layout nomination, the AI-only provisional
track — plus decoders and diagnostics. They are retained because committed
manifests, logs, and reports reference them by name; none of them can mark
human review or admit training data.

| Script | Role |
|---|---|
| `classify_probe_frames_vlm.py` | resumable VLM triage of saved probe frames; malformed responses fail closed as uncertain |
| `partition_probe_vlm.py` | deterministic ID shards of unfinished probe classifications |
| `merge_probe_vlm_shards.py` | merges per-worker prediction shards with coverage and provenance checks |
| `evaluate_probe_vlm.py` | builds the 86-frame human gold set and evaluates VLM probe triage against it |
| `index_probe_campaign.py` | deterministic classifier index over a mirrored probe campaign's completed attempts |
| `build_vlm_fetch_queue.py` | appends raw VLM target nominations to an auditable fetch-review queue |
| `classify_layout_surveys_vlm.py` | VLM nomination of full-cell scans from survey contact sheets |
| `reclassify_layout_surveys_vlm7b.py` | crop-first Qwen 7B reclassification of full-source layout surveys |
| `classify_layout_families_vlm.py` | VLM matching of surveyed HUDs against source-bound reference layouts |
| `propose_layout_family_pairs.py` | deterministic edge-structure proposals of exact layout pairs for review |
| `verify_layout_pairs_vlm.py` | order-invariant pairwise VLM verification of proposed layout matches |
| `transfer_wild_layout_family.py` | binds a proven family template to a new video as unreviewed provisional inputs |
| `scan_wild_cells.py` | full-video physical cell-activity scan under a nominated layout; rejects frozen or unstable overlays |
| `materialize_ai_boundaries.py` | converts AI timer diagnostics into a boundaries artifact whose reviewer kind is permanently `ai_agent` |
| `resegment_timer_trace.py` | policy-ablation resegmentation of a hash-bound AI timer trace without re-decoding |
| `publish_provisional_existing_layout.py` | publishes one AI-only existing-layout provisional build, marker last |
| `publish_provisional_family_transfer.py` | publishes one completed AI-only family-transfer build, marker last |
| `finalize_provisional_shards.py` | verifies concurrent provisional shard jobs and writes one immutable aggregate manifest |
| `overlay_parser.py` | opaque own-capture overlay decoder (frozen v1 spec); baseline for `translucent_parser.py` |

The dated Markdown files beside the scripts (`WILD20.md`,
`HARVEST_RECOVERY_20260727.md`, `WORKTREE_RECONCILIATION_20260727.md`, and the
engineering logs) are campaign records, and `wild20_tranche.json` is a
frozen campaign tranche definition; apart from `WILD20.md`, these also
remain in the private working repository.
