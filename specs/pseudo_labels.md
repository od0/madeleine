# Pseudo-Label Sidecar and Provenance Contract — v1

This is the interface contract for labels produced by a model rather than by
the engine or by a mapped foreign source. It extends the label-kind taxonomy
of [session_format.md](session_format.md): `truth.parquet` remains reserved
for engine truth, `labels_native.parquet` with `label_kind: "mapped"` remains
the mapped foreign channel, and this document defines the third kind,
`label_kind: "pseudo"`. Pseudo-labels are inferred supervision. They are
never written to `truth.parquet`, never merged into `labels_native.parquet`,
and their hours are reported beside, never summed with, mapped-label and
engine-truth hours.

The consuming program plan is
[results/idm/PSEUDO_LABEL_BC_PLAN.md](../results/idm/PSEUDO_LABEL_BC_PLAN.md).

## Versioning

- The sidecar schema identifier is `madeleine.pseudo-labels.v1`. Any change
  to array names, dtypes, semantics, or required manifest keys requires a
  version bump here with a migration note. Readers refuse an unknown schema
  identifier rather than parse best-effort.
- Relabeling a session with a different labeler checkpoint, inference
  config, or code version produces a new immutable artifact under a new
  prefix keyed by the labeler checkpoint hash. Existing sidecars are never
  overwritten.

## Files

A pseudo-labeled session adds to the session directory:

```
labels_pseudo.npz        # arrays, schema madeleine.pseudo-labels.v1
labels_pseudo.json       # provenance manifest for the sidecar
```

The session's own `manifest.json` is unchanged; the sidecar manifest carries
all pseudo-label metadata so that admission state and label state stay
independently auditable.

## labels_pseudo.npz

One row per labeled source row, canonical key order
(`left, right, up, down, jump, dash, grab`):

| array | dtype | shape | semantics |
| --- | --- | --- | --- |
| `y_prob` | float16 | `[N, 7]` | per-key pressed probability; the primary payload |
| `y_state` | uint8 | `[N, 7]` | `y_prob` thresholded at the frozen per-key operating points; convenience only |
| `source_row_index` | int64 | `[N]` | row index on the session's declared label grid |
| `source_pts_s` | float64 | `[N]` | presentation timestamp of the labeled frame |
| `continuity_id` | int32 | `[N]` | contiguous-run identifier; windows never crossed run boundaries during inference |
| `coverage` | uint8 | `[N]` | 1 where the row was center-supported by a retained window; rows outside retained spans are absent, not zero-filled |

Probabilities are primary so that downstream training may use soft targets
and thresholds may be revisited without relabeling. `y_state` embeds no
information not derivable from `y_prob` plus the manifest's thresholds.

## labels_pseudo.json

Required keys:

- `schema`: `"madeleine.pseudo-labels.v1"`.
- `label_kind`: `"pseudo"`.
- `grid_hz`: the label grid (60, or 20 with an explicit `phase` list for a
  decimated surface). Rates are never mixed within one sidecar and never
  interpolated.
- `labeler`: checkpoint SHA-256, model config hash, training run ID, the
  results report that qualified the checkpoint under the entry gate, and the
  inference code commit.
- `inference`: window length, stride, retained positions, phases, batch
  size, dtype, device.
- `thresholds`: the frozen per-key operating points with the file and hash
  they were frozen from, plus any calibration map identity (none in v1
  unless a calibration artifact is named).
- `upstream`: hashes of the fetch packet, boundary artifact, mask evidence,
  viewport verdict, and the batch acceptance artifact that admitted the
  video under the `pseudo_v1` tier.
- `seen_in_idm_training`: boolean, per this labeler — true when the source
  video appeared in the labeler's own training data. Comparison studies use
  unseen video only; corpora report seen and unseen hours separately.
- `coverage_summary`: labeled rows, total rows, and per-run edge losses.
- `license`: the license posture inherited by this artifact (v1 default:
  non-commercial, propagated from the NitroGen CC BY-NC 4.0 chain).
- `completed`: written last, after every array and key above is final.

## Integrity rules

- The sidecar is invalid without its manifest, and the manifest is invalid
  if any referenced hash fails verification.
- A builder consuming pseudo-labels refuses sessions whose sidecar schema,
  labeler hash, or tier acceptance artifact it does not recognize.
- Quality-control quarantines record the sidecar as quarantined; nothing is
  silently deleted.
