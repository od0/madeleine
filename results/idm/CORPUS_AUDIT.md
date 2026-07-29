# Full NitroGen Corpus Audit

Status as of 2026-07-26: source inventory, feature eligibility, action
mapping, frozen features,
and the complete shard/manifests contract validated. Current training state is
tracked in `../../PROGRESS.md`; visual-continuity QA is intentionally
non-blocking. One provenance limitation applies to the deep-validation report;
see "Validator provenance limitation" below.

## Source and label continuity

The durable S3-compatible object-store prefix has 234 verified objects totaling
237,033,327,002 bytes: 232 video files totaling 237,033,179,956 bytes plus two
provenance reports totaling 147,046 bytes. Of those 232 videos, 211 pass the
whole-video metadata gate for the nominal 60-Hz label grid; 21 are explicitly
rejected as nonuniform, rate-ineligible, or not marked 1:1. The accepted set
contains 194 native-CFR and 17 timestamp-resampled sources.

The earlier recovery census recorded 244 successful downloads. Twelve of those
videos are absent from the durable archive for a reason not captured in the
tracked evidence. Ten belonged to the historical 221-video eligible pool and
carried 41.9833 label-hours; subtracting them yields the current strict 211-
video, 150.9167-hour population.

The 211 accepted videos contain 32,598,000 labeled frames, or 150.9167 hours.
Their known label runs provide 149.2322 hours of valid 382-raw-frame targets:
98.8838 percent retention.  This supports keeping videos with occasional
chunk gaps and preventing windows from crossing those gaps, rather than
discarding whole videos.

The paused 49-video visual sample found no decoder-process failures, but did
find one important cadence mismatch: `v1097557936` decoded 235,382 of 414,000
expected labeled frames (56.86 percent).  The other 48 sampled videos decoded
at least 99.43 percent.  Its nominal stream rate is 60 fps but its decoded
average is 33.89 fps.  A complete cheap metadata pass found 17 such nominal-60
sources outside a 0.1-fps tolerance.  Feature generation now timestamp-samples
those sources onto the nominal 60-Hz label grid and records decoder mode and
bounded tail fill in the final manifest.

## Full action mapping

The source-action pass completed and was independently validated against every
expected file and row:

| Quantity | Result |
|---|---:|
| Videos/reports | 211 / 211 |
| Label chunks | 27,165 / 27,165 |
| Label rows | 32,598,000 |
| Skipped chunks | 0 |
| Declared grid | 60 Hz for every chunk |
| Label hours | 150.9167 |

Mapping confidence is a more important admission issue than capture gaps:

| Mapping cohort | Videos | Label hours |
|---|---:|---:|
| Unflagged bind inference | 93 | 106.00 |
| Flagged bind inference | 118 | 44.92 |
| Axis-sign determinate | 30 | 45.51 |
| Axis-sign indeterminate/fallback | 181 | 105.41 |

Median bind confidence is 0.487 (p10 0.418, p90 0.629; min 0, max 0.856).
The 118 flagged videos fall back to a deliberately broad prior mapping: dash
is the OR of west/east and grab is the OR of both triggers and both shoulders.
Those labels preserve all source data but are materially noisier than the
single-button mappings inferred for unflagged videos.

## Training decision (adopted 2026-07-26)

The adopted decision was to generate and retain features for all 211 videos,
with immutable per-video quality metadata, and not to silently reduce the
stored or default training corpus. The build has since completed: 1,554
hard-linked FP16 feature shards covering 32,598,000 train-ready frames
(150.9167 hours) across all 211 accepted videos, 194 native 60-Hz and 17
timestamp-resampled, deep-validated with zero errors (see "Feature-build
result" below). For the scale comparison, the complete 150.92-hour cohort is
the default population, and the 106.00-hour unflagged list (93 videos, 1,078
sessions) is preserved as the matched bind-noise diagnostic.  That comparison
is now complete: the all-valid training arm (210 videos, 148.32 hours after
holding out `y4n`) slightly exceeds the unflagged arm in macro AP on both the
mapped holdout (0.2723 versus 0.2693) and B1 (0.2713 versus 0.2603). This
supports retaining all-valid as the default AP corpus. It does not prove the
fallback mappings are clean because volume and binding cohort change together,
per-key effects are mixed, and B1 timing favors unflagged.

Axis-sign indeterminacy is carried as a second quality dimension rather
than used as an immediate hard filter.  Many videos simply lack enough d-pad
co-votes to infer sign, so indeterminate does not prove the fallback direction
is wrong.  Visual continuity is a third independent annotation, not an
admission gate: label continuity, mapping confidence, and frozen/static imagery
must not be collapsed into one generic notion of data quality.

The adopted operating policy is documented in `TRAINING_DATA_POLICY.md`.
Short repeated frames are tolerated; known missing chunks split runs; no whole
video is discarded for a sparse gap.  The 98.8838-percent long-context
retention makes an exhaustive visual scan too low-value to block model work.

## Feature-build result

- Visual masked-frame scan: the preserved partial artifact contains 49/211
  successful videos with zero scan errors; completing it is optional.
- Frozen ResNet-18 features: the completed build covers all 211 accepted videos
  without waiting for the optional visual scan. It contains 1,554 hard-linked
  FP16 shards and 32,598,000 train-ready frames (150.9167 hours); 194 videos
  used native 60-Hz decoding and 17 used timestamp resampling.
- Deep validation checked every shard header and hard link, both cohort/session
  lists, the 27,165-row chunk index, decoder provenance, confidence/continuity
  metadata, and exact frame/hour totals. It found zero build failures, skipped
  or truncated frames, tail imputation, or temporary artifacts.
- The all-valid population retains 1,554 sessions. The higher-confidence
  unflagged list retains 1,078 sessions across 93 videos. The common mapped
  holdout is removed from each list only by the training launcher.

Machine-readable evidence, all in this directory in the private working
repository (the public export excludes these files):
`corpus_contiguity_metadata.json`, `full_mapping_validation.json`,
`full_corpus_features_validation.json`,
`full_corpus_features_validation_provenance.json`, and the 49-row
`corpus_contiguity_visual.partial.jsonl`. Partial visual
results are non-blocking evidence; a final ranking will be appended only if the
optional scan later completes.

## Validator provenance limitation

The deep-validation report cannot be tied to a tracked validator version
through its own embedded commit field. The report's embedded Git SHA
(`d13ef3e`) identifies the remote worktree base, not the validator itself: the
validator script was untracked on that node when it ran, and that commit is
not reachable from `main`.
`full_corpus_features_validation_provenance.json` (private working
repository) records this limitation, the report's SHA-256
(`fa5948124011dfae7f0d1ca29bb0e17dc4f2e89dd86c16aa189d7b5e7d6b5b91`), and the
executed validator's SHA-256
(`9da6dcbc086c79a86906ab7ddeb0cd52ab415b32ff89001f4432616cc1f14788`). Those
validator bytes are byte-identical to the version tracked at private-repository
commit `10e7062`. The copy currently at
`experiments/validate_full_corpus_features.py` (part of the public export)
differs from the executed bytes only in default path arguments that were
neutralized for public release; the validation logic is unchanged. The tie
between report and validator therefore rests on the recorded byte hashes, not
on Git history.

The private working tree retains the chronological engineering record;
publicly useful failure rules are summarized in
[the curated engineering lessons](../../docs/engineering-lessons.md).
