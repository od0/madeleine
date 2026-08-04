# MADELEINE status

Measured status as of 2026-08-03. Stable findings and engineering conclusions
are preserved in [the technical report](report/README.md), the
[build retrospective](docs/history/how-this-was-built.md), and the
[curated lessons](docs/engineering-lessons.md).

`[x]` complete · `[~]` active · `[ ]` queued · `[-]` intentionally paused

## Research summary

The project has a functioning engine-truth acquisition rig, validated local
sessions, mapped NitroGen supervision, multiple trained IDMs, and a fail-closed
wild input-overlay pipeline. The strongest stable conclusion is that additional
supervision and end-to-end visual learning improve held-action recognition,
while exact onset/release timing remains substantially harder.

The complete 211-video frozen-feature corpus has passed deep validation. Both
matched scale jobs on the 2×A100 node are complete:

- `[x]` all-valid cohort: 210 training videos, 148.3222 trainable hours;
- `[x]` unflagged-bind cohort: 92 training videos, 103.4056 trainable hours.

Both hold out the same mapped NitroGen video, use the same 25.7M-parameter
128-sample/stride-three model, and train to an exact one-pass endpoint. The
unflagged/all-valid reach 0.2693/0.2723 AP versus 0.1924 prevalence on that
holdout and 0.2603/0.2713 versus 0.1448 on B1. State ranking improves with
scale while exact mapped-label timing remains nearly flat relative to the
nine-video pilot. The 112.95M run, B1 frozen-feature conversion, and matched
B1 evaluations are also complete.

The six primary own-data models have now been rerun on the corrected own-v3
shards and content-validated feature cache. Scratch macro AP changed 0.1735
-> 0.1700; Tier-B-initialized fine-tuning changed 0.1938 -> 0.1998 but remained
seed-variable, and neither family improved oracle +/-2-frame timing. The mask
sliver did not explain the own-only ranking failure. Exact attribution and
checkpoint hashes are in
[the corrected own-v3 report](results/idm/OWN_V3_RERUN.md).

Two independent NitroGen label defects were found and repaired
(2026-08-01 through 2026-08-02): a vertical-axis contract violation that
inverted analog up/down labels in 22 of 210 videos, and a bind-inference
defect that starved dash in 13 videos; the consolidated record is
[results/idm/NITROGEN_LABEL_INCIDENTS.md](results/idm/NITROGEN_LABEL_INCIDENTS.md).
A 105.7M VPT-topology model trained from scratch on the resolved-v3
corpus (210 videos, 148.3 hours) is the current strongest checkpoint:
0.6165 equal-video / 0.6334 row-weighted macro AP on the held-out Wild7
deployment gate, the strongest held-out result observed, single-seed,
reported without causal decomposition. Wild7 (seven admitted public
videos, HUD-decoded truth) is now the primary deployment gate per
[results/idm/VPT_SMALL_WILD_ADMITTED7_PRIMARY_GATE.md](results/idm/VPT_SMALL_WILD_ADMITTED7_PRIMARY_GATE.md);
val-A is a secondary engine-truth regression surface.

The node, credentials, and live process coordinates are intentionally not part
of this public status file. Completion is defined by validated artifacts and
markers, not by a process merely appearing in a process list.

## Flags
- **Codex API instability** — 503/500 outages truncated an infra packet mid-run; pod jobs survived only because they run under `nohup`. Keep using nohup for anything long.
- **Grid v1 vs v2 must not be pooled** — different training volumes (15 vs 40 min) and v1 lacks best-val checkpointing. Compare, never merge.
- **E2 gapped configs unrun** — any E2 claim from v1/v2 output is the persistence baseline, not a policy prior. See findings log.
- **A JS runtime (deno) is required for YouTube extraction** — the lane node lacks one, so yt-dlp reports degraded formats for YouTube URLs there. Verified harmless for this corpus (0 degraded fetches; the YouTube videos came down on the old pod, which had deno), but any fresh re-fetch box needs deno installed or YouTube pulls silently downgrade. Belongs in the reproducibility section.
- **Viewport measured: 193 fullscreen / 39 layout / 2 undecodable (82.5% clean)** — the layout problem is a tail, not the corpus; policy is FILTER to fullscreen, not build a cropper. Rung 2 drops `y4nQHqYSObI` (layout) and keeps `dH6VsMUNRFo` + `v2033463254` (both fullscreen).
- **Foreign frames are stream layouts, not game frames** — sampled corpus videos show Twitch VODs (97% of the slice) rendering the game in a sub-rectangle beside a facecam, chat and splits, while YouTube uploads are mostly full-screen. Resized to 128px, a layout spends most of the tensor on furniture and shrinks gameplay well below our own data's effective resolution — a domain shift AND a resolution loss that would surface only as "foreign data does not help". NitroGen ships no game-area bbox (verified: metadata has `bbox_controller_overlay` only). `nitrogen/viewport.py` classifies fullscreen-vs-layout; rung 2 should prefer fullscreen videos until a viewport crop exists.
- **Scale opportunity (bryan)** — speedrun.com lists ~6,400 PC Celeste runs; lower-ranked runs are LONGER (2–3 h vs 25 min) *and* closer to our own play distribution, so descending the leaderboard raises yield and cuts the 4–15x covariate shift at once. Gated on S3 Phase A proving the decode.
- **Corpus rate diversity is real** — of 244 fetched videos, 233 are 60 fps, 10 are natively 30 Hz, and one is **59.94** (NTSC: drifts ~1 frame per 1000 against a 60 Hz label grid). 23 videos / 20.2 h excluded from training by the per-chunk rule; the rule must stay per-chunk, since some single video IDs carry mixed rates internally.
- **val-B windowed comparisons are not apples-to-apples** (33% render drops).
- **`aligned_1to1: True` does not imply 60fps** — `BJ7ymU_EbE4` is natively 30Hz (grid_hz 30 in the chunk index), fetched at 30fps, and correctly stamped aligned. Corpus consumers filter on BOTH `aligned_1to1` AND `grid_hz == 60` (rung-2 curation rule); neither alone is sufficient. (Corrects this session's earlier misread of the fetch as a fallback bug.)
- **Phase-1 YouTube slice is one-channel-dominated** — 7 of 10 rung-2-eligible videos are uploader Peppy29 (measured via yt-dlp). Rung-2 pick works around it: `dH6VsMUNRFo` (Peppy29 7.1h) + `v2033463254` (nalalath 3.7h) + `y4nQHqYSObI` (CostilerAphid18 2.6h) ≈ 13.4h, 3 creators. Any phase-one-wide claim (rung 3) must state the channel skew; belongs in the dataset card.
- **WORKING CAPTURE PROFILE (passes the 2% gate, 2026-07-25)** — `g1_capture --display --delay 10`, unchanged from the documented profile: OS-queried Celeste window, 16:9 canvas crop, capture 1710×962 (canvas_scale 0.890625) off the built-in 1710×1112 screen, fullscreen game on its own Space, libx264 crf16 veryfast yuv420p CFR-60. Measured: achieved 60.0 fps, 17.94 ms p99 tick jitter, **46 drops + 45 dups in 7800 frames = 0.59%**, engine span exactly 1:1, 94% input-active. No setting needed changing. **Hold-lift is the orchestrator's call.**
- **The passing test did not cover the chapter that failed** — test A ran in rooms `s0`/`s1`/`s2`; the 33% session (`rec_20260725_025853`) was rooms `start`/`00`–`04`. Engineering-log #19 already attributes that 33% to the game's own render rate in heavier scenes, resolved-with-limitation, so this is a scope note on the pass, not a reopened defect: the profile is proven for `s`-room content and untested against the `00`–`04` chapter.
- **[RESOLVED 2026-07-26] `data/sessions/INDEX.md` is tracked** — orchestrator force-added it in `c906e35`; `git ls-files` confirms. The underlying `.gitignore` unanchored `sessions/` rule still deserves the `/sessions/` fix so future files don't need `-f`. Original flag text kept below for the record:
  `data/sessions/INDEX.md` HAS NEVER BEEN IN THE REPO — `git ls-files data/sessions/` is empty. `.gitignore:6` is `sessions/` with no leading slash, so it matches at any depth and swallows `data/sessions/` along with the intended top-level video dir. CLAUDE.md designates the index a coordination surface with "git is the arbiter"; the board marks it `[x]` shipped and links to it. Both describe a file no other session can see. Capture wrote the B1 row locally and did **not** `git add -f` — the fix is a `.gitignore` change (`/sessions/`, or a negation for the index), which is repo structure and not capture's to make. Until then every session-ledger commit in the log is PROGRESS.md only.
- **Block B1 FAILED the 2% gate at 4.3% drops** (`rec_20260725_192824`, 2,339/54,000) — capture loop clean (60.0 fps, 18.02 ms p99, 54,000/54,000 decoded), content good (81% input-active, grab 24.2%, dash 7.7%), so this is the game's render rate, not the rig. **The 0.59% profile pass therefore does not generalize across chapters**, which is the scope caveat landing exactly as written. Session kept as evidence, no split, video untouched; re-record per the block's rule. One more consecutive failure stops the block.
- **B1's video is 352 MB where prior 15-min sessions ran ~2.1 GB** at identical encoder settings (crf16 veryfast) and comparable input density — a 6× bitrate drop nobody asked for. Unexplained; may mean softer frames reaching the model. Capture is not chasing it.
- **Failed sessions live in `sessions/rec_*` like passing ones** — anything globbing that pattern instead of reading explicit split lists would pick up `rec_20260725_192824`. Index marks it FAILED; consumers should not be trusting the glob.
- **InputTruth logs `room_id` but not the area/chapter SID** — so no session can be attributed to a named chapter from truth data alone, and `INDEX.md`'s "chapters covered" column cannot be filled from manifests. Room IDs are ambiguous across maps (`00`–`04` matches Summit, Core and Farewell partially; `start`+`00`–`02` matches Reflection), which is why block B's heavy-chapter target had to be written as "`00`–`04`-style". Identified by cross-checking shipped map binaries; capture cannot fix it, a one-field mod addition would. Test A's chapter *was* resolvable — Chapter 3: Celestial Resort, confirmed by save `LastArea` SID **and** a unique 3/3 `s0`/`s1`/`s2` map match.
- **Boot disk at 98% (25 GiB free)** — a 15-min session costs ~2.1 GB, so headroom is ~10 sessions; APFS this full is also a write-stall risk mid-capture. Capture session is not clearing anything; **bryan** decides what goes.
- **[RESOLVED] Stale orchestrator rsync** — killed same day; its payload had already landed (grid v3 ran the shipped code).
- **[RESOLVED] Mystery mirror logs on the lane node** (`pixel_mirror.log`, `preservation_to_2x.log`) were Codex's preservation rsyncs to its 2×A100 node — byte counts match its notes exactly. Lesson on the board: parallel orchestrators announce jobs where the other reads. (engineering log #39)
- **Two recordings named "B1"** — orchestrator's `rec_20260725_160450_b1` PASSED (0.44%); capture session's `rec_20260725_192824` FAILED (4.3%, game render rate in a heavy chapter). Distinct sessions, both in INDEX; naming collision only. Future blocks: capture owns the B-numbering.
- **YouTube bot-block is RATE-dependent, per-IP** — 14 workers from one
  datacenter IP earned it; Deno does not lift an already blocked IP. The
  initial scan missed 5,090 YouTube probes; targeted style-survey retries have
  since recovered 13, leaving **5,077** outstanding. Distributed low-rate
  fetching is the design. (engineering log #38 and the 20:23 EDT audit)
- **R2 is the durable data home** — 369.94 GB / 29,113 objects at the close of the build window, ~1.56 TB / ~89,000 objects after the wild harvest (growth almost entirely fetched wild media; every upload is SHA-256-and-size readback-verified on write), egress free. Layout + rehydration in infra/MACHINES.md.
- **VFR VODs defeat the video-level fps stamp** — `v1097553480`'s per-chunk grid_hz sweeps 47.5–60.0 (variable-frame-rate source) yet the 60fps fetch stamps it aligned. Curation rule hardened: a video enters training only if EVERY chunk's grid_hz == 60. Excluded from rung 2 along with `BJ7ymU_EbE4`.

## Acquisition and data

### Engine-truth capture

- `[x]` `granny` (`InputTruth` compatibility assembly) logs the seven actions
  and player state at 60 Hz.
- `[x]` Machine-readable frame-index rendering supports clock-free alignment.
- `[x]` Frame-index and input-overlay regions are masked before training.
- `[x]` Session validation checks hashes, frame continuity, drops, duplicates,
  masks, and canonical key order.
- `[x]` Multiple training and development captures have been assembled.
- `[x]` B1 is frozen as a cleaner development-only diagnostic: 53,762 aligned
  frames, 37,898 active targets, and 9,202 strict long-context targets.
- `[x]` Rebuild own-data shards with corrected, measured overlay-mask
  geometry, gated by the mask-coverage check, and verify the leak is gone
  (2026-07-27; see Known issues).
- `[x]` Rerun the six primary own-data models on the rebuilt shards using the
  content-validated corrected own-v3 feature cache; validate val-A sidecars
  and register/publish all checkpoint hashes (2026-07-28; see
  [the corrected own-v3 report](results/idm/OWN_V3_RERUN.md)).
- `[x]` Capture a new uninterrupted engine-truth session after the model and
  threshold policy are frozen; use it once for final evaluation (executed
  2026-07-28; the pre-registered battery extends it).

### NitroGen

- `[x]` Full Celeste metadata census: 411 videos and 684 nominal label-hours.
- `[x]` Source recovery census: 411 sources checked, 245 available candidates,
  and 244 historical successful downloads carrying 213.0889 label-hours.
- `[x]` Historical per-chunk eligibility: 221 videos and 192.9 label-hours.
- `[x]` Durable media inventory: 232 videos and 164.4222 label-hours; twelve
  historical downloads were not preserved.
- `[x]` Strict whole-video feature-eligibility gate: 211 videos, 27,165 chunks, 32,598,000 rows,
  and 150.9167 label-hours.
- `[x]` Action mapping validated across every strict-corpus chunk with zero
  skipped rows.
- `[x]` Higher-confidence binding cohort preserved separately: 93 videos and
  106.00 hours.
- `[x]` Timestamp-resampling path validated on the worst 33.89-fps decoded
  source with zero tail fill, truncation, or skipped frames.
- `[x]` Build frozen features for all 211 videos atomically and resumably.
- `[x]` Deep-validate all 1,554 feature shards and manifests: 32,598,000
  train-ready frames (150.9167 hours), 194 native and 17 timestamp-resampled
  videos, zero failures, skipped/truncated frames, tail imputation, or temporary
  artifacts.

### Wild input overlays

- `[x]` Source acquisition and immutable raw-publication contracts implemented.
- `[x]` Layout, boundary, offset, decode, and derived-publication stages fail
  closed on missing or AI-only review provenance.
- `[x]` Source-bound review packets and focused acceptance tests implemented.
- `[~]` Convert reviewed videos through the rolling decode/calibration/shard
  pipeline.
- `[ ]` Count only accepted, shard-built output as train-ready yield; raw video
  hours and provisional gameplay windows remain separate counters.

#### Scale campaign snapshot

(Evidence paths cited below resolve in the private working repository;
the public export carries the summarized state only.)

**Track 2 — wild harvest at scale (CPU fleet) · owner: orch · serves: corpus,
W1 label-quality experiment, S3**
1. **Exact discovery snapshot (2026-07-26 20:23 EDT):** 7,071 PC candidates /
   6,757.525542 nominal hours. The initial scan successfully probed 1,768 /
   2,341.475645 h; targeted style-survey retries recovered 13 more YouTube
   candidates / 19.217046 h, so current unique probe coverage is **1,781 /
   2,360.692691 h**. The remaining **5,290 / 4,396.832851 h** are 5,077
   YouTube / 4,160.207735 h and 213 Twitch / 236.625116 h. Evidence:
   `results/wild/candidates.jsonl`, `results/wild/scan.jsonl`, and
   `results/wild/style_survey.jsonl`.
2. **Classification snapshot:** reliable visual labels cover 86 unique
   candidates / 301.109426 nominal h. They confirm 13 keyboard/action-HUD
   videos / 32.175702 h; 11 of those / 27.401601 h are already raw-published.
   The immediately fetchable confirmed remainder is `odiIYNqjL9Y`
   (4.077408 h) plus `elDsFg-S8YA` (0.696693 h). A further 1,695 successfully
   probed candidates / 2,059.583265 h remain unclassified; the classical
   shortlist alone has 1,210 unclassified nominees / 1,441.779280 h. Classical
   `has_overlay` is nomination, not confirmation. Evidence:
   `results/wild/hand_labels.json`, `results/wild/style_labels.json`, and
   `results/wild/hud_shortlist.json`.
3. **Conversion snapshot:** all 11 tranche videos are present in the raw R2
   prefix (75 objects / 44,657,490,085 bytes). Ten videos have provisional
   decode reports: 30.700426 decoded-envelope h and 26.144301 AI-allowed h.
   The immutable broad-seven aggregate contains 4,835,638 post-activity frames
   / **22.387213 provisional hours** in 2,058 parts; its R2 checkpoint is 2,072
   objects / 169,815,744,637 bytes. **Admission opened 2026-07-28: first publications complete (b43KAaem61g 0.3848 h, kdQbIoMxzZw 1.1246 h), remaining accepted videos mid-publication**;
   provisional is not clean/train-ready. Evidence: `results/wild20/raw/`,
   `results/wild20/*/*/decode_report.json`, and the 19:31 entry in
   `harvest/WILD20_ENGINEERING_LOG.md`.
4. **Idle finding resolved at 20:28 EDT:** commit `3f92db1` hardened the probe
   path with pinned Deno/yt-dlp, one fragment, request pacing, deterministic
   per-IP queues, and immutable per-attempt R2 checkpoints with SHA-256
   readback/completion-last. As of 20:58 EDT, **all 5,303 initial failed rows
   are assigned with zero duplicate queue IDs across 14 active public IPs**
   (13 serial YouTube workers plus one Twitch worker). The live campaign has
   already completed 1,218 attempts, including 785 newly usable probe frames.
   Never return to 14 concurrent requests on one IP. In parallel, the saved
   1,768-frame legacy evidence set was moved
   server-side to R2 and an open-source VLM triage run was started on the idle
   A10; its output is machine nomination, never human review.
5. The two already-confirmed positives are no longer waiting behind discovery:
   `elDsFg-S8YA` is fully raw-published and SHA-256 readback verified (685 MB,
   2,933.133 s, native 30 fps), while the 4.077-hour `odiIYNqjL9Y` fetch and
   exact PTS scan are active on a separate fresh IP. Continue the downstream
   stages concurrently as positives arrive: fetch,
   layout/timer evidence, provisional decode, named human review, offset
   calibration, then fail-closed admission. W1 remains matched-hours
   wild-keyboard versus NitroGen labels on the same holdout.
6. **Raw-acquisition completion (2026-07-28 ~11:30 UTC):** the frozen 631-row
   fast-lane queue is **628/631 raw-complete / 612.940676 nominal hours**,
   every completion marker byte-validated with SHA-256 readback. The Twitch
   portion finished via a missing-only restart on the original host; the
   125-row YouTube tail was fetched by an owner-approved fleet of fresh
   non-spot CPU IPs with per-IP real-media health gates (one US IP failed its
   gate and got nothing; one lane was source-blocked mid-run and its whole
   queue recovered onto a fresh gated IP). The terminal residual is 3 videos
   / 2.534525 h: one persistent per-video HTTP 403 (`vaonu_vOyTQ`, failed on
   two distinct IPs) and two yt-dlp JS-challenge-solver rows pending an owner
   tooling decision. Raw-complete is acquisition only: surveyed, VLM-triaged,
   strict-scan, decoded, provisional, human-reviewed, and training-admitted
   remain separate counters; human-admitted wild data reached its first nonzero hours on 2026-07-28 (see the wild snapshot above).
   Evidence: `results/wild/general-harvest-ytr-20260728T0530Z/` and
   `harvest/WILD_HARVEST_ENGINEERING_LOG.md`.
7. **Two-stage modality reclassification (2026-07-29 06:48 UTC):** stages A and
   B both ran to `state: complete` over the 627 fast-lane rows that have a
   survey artifact. Stage A nominated 567 `decodable_input_hud` / 60
   `uncertain` / **0 negatives**, reproducing at corpus scale the calibration
   finding that it cannot discriminate a negative. Stage B, the pairwise
   change verifier, split those into 223 confirmed (213.7 nominal h), 42
   uncertain (54.2 h), and 362 deprioritized (339.1 h; 331 static-or-frozen,
   31 no-HUD), moving 316 stage-A positives into the deprioritized bucket.
   These are AI diagnostics: every row is `human_reviewed=false` and
   `training_admitted=false`, and stage B orders a review queue rather than
   rejecting anything — the calibrated false-static rate on clear positives
   was 3 of 10, so the deprioritized bucket must be human-sampled before any
   durable rejection. The four unclassified rows are the three terminal raw
   residuals plus `v1068970940`, which is raw-complete but unsurveyed.
   Evidence: `results/wild/reclassify-fleet-20260728T1500Z/` and its
   calibration record `results/wild/reclassify-calib-20260728T0500Z/`.
   A 48-video seeded stratified AI-diagnostic sample of the deprioritized
   bucket (2026-07-29) measured a weighted false-static rate of **14.7%**
   (~53 implied videos; 9 confirmed in sample, 7.2 nominal h), concentrated
   in labeled-action overlays (4 of 10) whose fill-based state indication
   defeats the pairwise comparator; pending Bryan's confirmation of the
   nine strips. Evidence:
   `results/wild/reclassify-fleet-20260728T1500Z/deprioritized-sample-20260729/`.

## Models and experiments

### Baselines and local diagnostics

- `[x]` 2-frame, past-only, and centered-window baselines.
- `[x]` Gap-aware temporal window construction and `input_active` filtering.
- `[x]` Chance, persistence, state, and transition-aware metric surfaces.
- `[x]` Controlled label-jitter and overlay-degradation experiments.
- `[x]` State-ambiguity and future-divergence diagnostics.

### Mapped NitroGen transfer

- `[x]` Curated 13.45-hour three-seed comparison after the corrected own-v3
  rerun: macro AP 0.1700 → 0.1941 and exact event F1 0.0784 → 0.0920 on
  the local development split. The mapped zero-shot arm consumed no own-data
  shards and therefore did not require a rerun.
- `[x]` Curated 40.61-hour scale diagnostic. Fixed-endpoint AP improved on
  average, while exact and ±2-frame event F1 remained flat.
- `[x]` 25.7M frozen-feature capacity probe. State F1 improved; exact timing
  did not.
- `[x]` 36.9M end-to-end visual model. Macro AP reached 0.2461, while exact
  event F1 fell to 0.0764.
- `[x]` NitroGen-only unseen-video holdout. Macro AP reached 0.2435 versus
  0.1924 prevalence; exact event F1 was 0.0127.
- `[x]` 112.95M end-to-end curated-corpus run. Selected AP reached 0.2318
  versus 0.2461 for 36.9M; exact event F1 changed from 0.0764 to 0.0810.
  Prediction sidecars and the selected checkpoint were archived with hashes.
- `[x]` Matched full-corpus scale comparison after feature validation. Both
  arms and their mapped-label evaluations are complete and validated.

### NitroGen label corrections and VPT-topology runs

- `[x]` Vertical-axis mapping incident recorded, repaired from raw
  controller arrays (corrected-v2), and independently verified over all
  32,037,600 rows (2026-08-01).
- `[x]` All-seven-key mapping audit: dash starvation and grab-fallback
  defects found; per-action resolved bind sets built under the
  upstream-preference policy (mapper v3/v3.1) and the v3 corpus rebuilt
  and published with independent verification (2026-08-02). 227 of 630
  bind entries remain policy-resolved pending final human review, so v3
  results are described as resolved mapped labels, not individually
  human-verified bindings.
- `[x]` Corrected-v2 unflagged92 retrain and the from-scratch resolved-v3
  full-210 production run complete with byte-identical duplicate
  inference passes on every evaluation (2026-08-03).
- `[x]` Nine retained checkpoints scored once each on identical Wild7
  support
  ([scorecard](results/idm/vpt_wild7_checkpoint_parity_v1/scorecard.json)).
- `[x]` Wild7 adopted as the primary deployment gate; resolved-v3 is the
  frozen comparison baseline (0.6165 equal-video / 0.6334 row-weighted
  macro AP, dash 0.5630 over the 0.3855 bar).

### Evaluation gates

- `[x]` New promoted development results retain prediction sidecars and
  checkpoint hashes. The original three-seed frozen Tier-B sidecars are
  archived; historical-commit regeneration reproduced every 0.5-threshold
  decision and all report metrics within CPU/GPU floating-point tolerance.
- `[x]` Selected and fixed-final endpoints are evaluated separately where
  checkpoint objectives can disagree.
- `[x]` Build and validate B1 frozen features; matched 36.9M/112.95M
  end-to-end development evaluation is archived with sidecars.
- `[x]` Score both full-corpus models on B1 as a development diagnostic;
  selected/final reports and prediction sidecars are archived and validated.
- `[x]` Freeze thresholds and model selection before the untouched capture.
- `[x]` Report the untouched engine-truth result with per-session support,
  prevalence, AP, state F1, exact event F1, and ±2-frame event F1.

## Documentation and publication

- `[x]` Public project overview and standalone research narrative.
- `[x]` Public contributor guide and durable research roadmap.
- `[x]` Complete public-documentation sanitation and archive only durable
  historical findings.
- `[x]` Replace live host addresses, provider identifiers, credential paths,
  and balances with provider-neutral reproducibility notes.
- `[x]` Add third-party and data notices.
- `[ ]` Select and add a repository software license (owner decision).
- `[x]` License selected and added: MIT (2026-07-27).

## Next sequence

1. **COMPLETE 2026-07-28:** rerun the six primary own-data models on the
   rebuilt own-v3 shards and content-validated corrected feature cache. All
   six val-A reports and aligned sidecars passed exact-support validation;
   checkpoint hashes were registered and published under content-addressed
   completion-last prefixes. Scratch macro AP changed 0.1735 -> 0.1700;
   Tier-B-init changed 0.1938 -> 0.1998 but remained seed-variable, and
   neither family improved oracle +/-2-frame timing. See
   [the corrected own-v3 report](results/idm/OWN_V3_RERUN.md).
2. **Score the untouched engine-truth test (PRE-REGISTERED; GPU lane owner:
   Codex).** Session `rec_20260727_220000_test` was recorded and sealed
   2026-07-27 (53,280 frames, 0.34% drops, Chapter 6, leak scan clean); its
   shard is staged in the durable object store under
   `shards/test-untouched-v1/`. Protocol, fixed before any number is seen:
   evaluate the frozen checkpoints exactly as listed in the results
   directory's checkpoint hash records — the three engine-truth-only seeds,
   the three mapped-zero-shot seeds, both end-to-end models (36.9M,
   112.95M), and both full-corpus models (103.41 h, 148.32 h) — one pass
   each, with frozen val-A per-key thresholds (no thresholds fit on this
   session, no checkpoint reselection), reporting macro/per-key AP, collar-0
   and ±2 transition F1, and both key-state accuracy readings with their
   trivial baselines. Results are reported whatever they say; this session
   is never evaluated twice under different settings. Status 2026-07-28,
   COMPLETE: the private preflight audit found three blocking input gaps;
   the owner resolved all three in the private working record, the
   mechanical prerequisites passed (full-corpus val-A threshold freeze;
   every checkpoint reproduced its committed val-A record under
   training-era code, predictions bitwise for the end-to-end pair), and
   the single pass ran — ten models, no retries, no strikes, all
   integrity checks passing. Headline on the input-active maximal
   support: best macro AP 0.2377 (36.9M end-to-end) against 0.1515
   prevalence chance; mapped-supervision families above chance, own-data
   seeds near it (their training shards predate the mask fix; item 1's
   completed clean reruns were correctly not added retroactively to this
   spent battery); collar-0 event F1 0.020–0.034
   against a 0.0054 shuffled-event anchor; micro accuracy below
   always-released for every model. Transfer to unseen engine-truth
   content is far below the development-split numbers; reported as
   found. Tables and stored predictions are in the results directory's
   untouched-test reports; the session is spent and is never evaluated
   again. The pre-registered four-chapter battery was sealed and scored
   the same evening (sixteen models, one pass; all pre-scoring acceptance
   conditions satisfied, with the documented capture deviations): pooled
   best macro AP 0.2667 vs ~0.16 chance, above-prevalence signal on every
   chapter with per-chapter difficulty varying around the Chapter 6 point,
   own-data families near chance on all chapters including their anchor,
   timing 4–8× luck but at most ~0.05 everywhere. The six own-v3 rows are
   protocol-locked diagnostics (evaluator-semantics mismatch, detailed in
   the battery report); headline conclusions rest on the ten original
   frozen models.
   Results are in the results directory's untouched-battery reports; those
   four sessions are also spent.
3. Install `gitleaks` and run the strict public-release verifier (the
   figure-hash pin was refreshed 2026-07-27); the tooling never pushes
   automatically.
5. Repair or reject the engine-counter-reset alignment in
   `rec_20260725_015612` and recompute its drop-rate denominator; add
   monotonicity/reset checks to the session validator.
6. Test a transition-aligned objective and checkpoint rule.
7. Consolidate the technical report, figures, limitations, and artifact map.
8. **Pseudo-labeling and behavior-cloning program, stage 0** (plan drafted
   2026-07-30:
   [results/idm/PSEUDO_LABEL_BC_PLAN.md](results/idm/PSEUDO_LABEL_BC_PLAN.md);
   nothing provisioned or admitted yet). Concurrent with the IDM ladder, in
   this order: (a) GPU inference benchmark for the VPT-small labeler on one
   H100, both window geometries — the receipt every labeling cost estimate
   is waiting on; (b) the deterministic rollout harness (input injection in
   the engine-truth mod, lockstep frame stepping, in-loop masked
   observation capture, replay-diff determinism validation) — the program's
   wall-clock long pole and prerequisite for any promote decision; (c) the
   ~32M causal pilot policy with its causality test and real-data smoke;
   (d) freeze the promote-study preregistration (margins, seeds, arms,
   baselines) and sign the `pseudo_v1` tier spec; (e) run the Phase 6
   promote pilot on the frozen 66-video / 26.9278 h labeler-unseen promote
   reserve (receipt in `results/idm/vpt_promote_reserve_v1/`) once any IDM
   passes the labeler-eligibility gate; the wider admitted population
   (161.9664 h per the immutable build receipts) remains the tier A
   labeling substrate. Census acquisition tranches (capped, CPU
   only) may start ahead of the gate on Bryan's approval.
9. **Public push for the resolved-v3 result (in preparation).** The
   release bundle per the gate report's plan: the incident narrative,
   the v3 configuration and checkpoint identity, the Wild7 comparison
   tables, the secondary val-A result, and reproducible scoring
   commands. Content is collected in the working repository; the export
   and push happen only after owner review.

## Known issues

- **[RESOLVED 2026-07-28] Input-overlay mask undershoot (found 2026-07-26;
  geometry, shards, and guard fixed 2026-07-27; primary model reruns
  complete).** The declared mask rect
  missed the top ~23 px of the rendered overlay cells on the
  built-in-display rigs, and the entire cell row on the pre-crop 1710×1112
  session; a readable answer-key sliver survived in the 1710-family
  own-data shards (single-feature AUC 1.000 for `left` in a training
  shard). Measurement against engine truth showed the 2560×1440 sessions
  were covered all along, so val-A and the E1/E2-era shards never
  contained the leak; the leaked sessions were `rec_20260725_015612` and
  `rec_20260725_021338` (train), `rec_20260725_025853` (val-B), B1, and
  the calibration session. Fixed on 2026-07-27: session manifests carry
  measured per-family rects with the superseded rects preserved,
  `data.mask_coverage` fail-closes any shard build whose rect does not
  cover its rendered widget, and `data/shards_v2` was rebuilt (frame
  counts and splits unchanged, new shard hashes) and probed clean — masked
  zones identically zero in every frame; the adjacent band separates keys
  at gameplay level only (AUC 0.53–0.61 for the previously leaked
  sessions, down from 0.91–0.94). The prior shards are retained at
  `data/shards_v2_leaked_20260726/` as legacy provenance and are no longer
  training inputs. The six direct reruns reused the content-validated
  corrected own-v3 cache: scratch macro AP changed -0.0035 and oracle
  +/-2-frame F1 changed -0.0058, while Tier-B-init AP changed +0.0060 and
  oracle +/-2 changed -0.0017. The defect therefore changed some natural-
  threshold behavior but did not explain own-only near-chance ranking or
  reveal a timing improvement. See
  [results/idm/OWN_V3_RERUN.md](results/idm/OWN_V3_RERUN.md); full geometry
  record remains in the findings log.

## Authoritative reports

- [Results summary](results/idm/SUMMARY.md)
- [NitroGen-only holdout](results/idm/NITROGEN_HOLDOUT.md)
- [Full-corpus audit](results/idm/CORPUS_AUDIT.md)
- [Training-data policy](results/idm/TRAINING_DATA_POLICY.md)
- [Dataset card](data/dataset_card.md)
- [Wild20 production path](harvest/WILD20.md)
- [Curated engineering lessons](docs/engineering-lessons.md)
