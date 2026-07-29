# Frozen Session Format — v1

This is the interface contract between the instrumentation track and the research
track. It is frozen: changes require a version bump here and a migration note, and
nothing downstream may assume anything this document does not state.

## Versioning and evolution

- `manifest.json` `format_version` identifies the session format; this document
  defines version `"1"`. Any change to the tables, sentinel semantics, or
  manifest keys requires bumping `format_version` here together with a
  migration note. Readers should refuse an unknown `format_version` rather
  than parse best-effort.
- The recording mod's version is recorded separately in `env.mod`
  (`"InputTruth x.y.z"`). Sessions recorded with mod 0.1.0 predate the state
  fields: their `truth.parquet` carries assembler-written placeholders
  (`input_active = true`, `room_id = ""`, zeroed numeric state) rather than
  measured state — see `theo/g1_assemble.py` (`build_truth`). Those
  placeholders are distinct from the player-absent sentinels defined in
  [specs/field_semantics.md](field_semantics.md).

## Canonical key order

Everywhere keys appear — truth columns, model heads, mapped foreign labels, overlay
parser output — the order is:

```
left, right, up, down, jump, dash, grab
```

## Directory layout

```
sessions/<session_id>/
  manifest.json
  video.mkv                # recorded sessions only
  truth.parquet            # recorded sessions only — engine truth
  alignment.parquet        # recorded sessions only
  labels_native.parquet    # foreign (nitrogen | wild) sessions only — mapped labels
```

### Naming rule (structural, enforced by the validator)

`truth.parquet` is **reserved for engine truth**. A session whose manifest declares
`provenance.source != "recorded"` must not contain a file named `truth.parquet`,
and carries its labels in `labels_native.parquet` with `label_kind: "mapped"` and an
explicit `grid_hz`. Evaluation ground truth is engine truth only; curves computed
against mapped labels are relative measurements and must be labeled as such.

## truth.parquet — one row per engine frame

| column        | type    | semantics |
|---------------|---------|-----------|
| frame_idx     | int64   | Dense and monotonic from mod init; it does not reset across deaths, room loads, or level restarts, but it does reset if the game process restarts mid-session. Resets are treated as stream boundaries by downstream consumers; the validator does not currently enforce monotonicity (at least one committed session contains a reset and passes). Gaps within a stream are an integrity violation. This is the same counter rendered in the on-screen frame-index strip. |
| left … grab   | bool ×7 | Bound action state (canonical key order), sampled once per engine update. |
| input_active  | bool    | False when player input is disconnected from the avatar: menus, pause, death animation, room transitions, cutscenes. See [specs/field_semantics.md](field_semantics.md). |
| room_id       | string  | Current room name (level session key). |
| pos_x, pos_y  | float64 | Player position, level-global pixels (not room-local). |
| speed_x, speed_y | float64 | Player speed, px/s, engine convention. |
| dash_count    | int32   | Dashes currently available. |
| stamina       | float64 | Engine stamina value. |
| on_ground     | bool    | Engine ground flag. |
| death         | bool    | True only on the frame the death **triggers** (edge event), not during the animation or respawn. |
| session_id    | string  | Matches manifest.session_id. |

Full per-field semantics, including values during death/transition/pause frames
and the player-absent sentinels (float `NaN`; `dash_count = -1`, since the
column is integral), are pinned in
[specs/field_semantics.md](field_semantics.md); that table is part of this
contract.

## alignment.parquet — one row per video frame

| column                 | type   | semantics |
|------------------------|--------|-----------|
| video_frame_idx        | int64  | 0-based index in video.mkv (CFR, so index ⇔ time). |
| engine_frame_idx       | int64  | Decoded from the rendered strip; stored as −1 (not meaningful) on every row whose decode_status is not "ok". |
| decode_status          | string | `ok` \| `unreadable` \| `out_of_session`. Unreadable is **never** inferred, interpolated, or smoothed. |
| is_duplicate           | bool   | `engine_frame_idx == previous readable engine_frame_idx`. |
| preceded_by_drop_count | int32  | `max(0, engine_frame_idx − previous readable engine_frame_idx − 1)`. |

Video frames before the first readable index or after the last readable index are
`out_of_session`.

## video.mkv encode contract

- **Constant frame rate, 60 fps.** VFR is forbidden: video_frame_idx semantics
  depend on it. Capture drops surface as explicit duplicate frames, which the
  alignment accounting detects exactly.
- Codec/pixel format pinned per session in the manifest; default: libx264,
  crf 16, preset veryfast, yuv420p. The frame-index strip is luma-only and sized
  to survive 4:2:0 ([specs/frameindex_encoding.md](frameindex_encoding.md));
  the synthetic encode fixture (`data/toy_sessions.py` rendered strips, decoded
  through the transcode tests in `tests/test_frameindex.py`) adjudicates any
  settings change before real sessions use it.

## Grids

- Recorded sessions live on the 60 Hz engine grid.
- Foreign labels stay on their **native** grid (one label per source frame;
  `grid_hz = chunk_size / duration`), declared per stream in the manifest and in
  the parquet metadata. Resampling between grids happens in exactly one
  place: `data/precompute_features.py` performs timestamp-aware resampling
  onto the 60 Hz grid, while `data/build_dataset.py` rejects non-60-Hz
  input outright rather than resampling.

## manifest.json

```jsonc
{
  "format_version": "1",
  "session_id": "…",                       // matches directory name
  "created_at": "ISO-8601",
  "env": { "game": "…", "everest": "…", "mod": "InputTruth x.y.z" },
  "capture": { "tool": "ffmpeg-avfoundation|obs|…", "requested_fps": 60,
               "achieved_fps": 59.98, "encode": "libx264 crf16 veryfast yuv420p",
               "resolution": [w, h] },
  "streams": { "video": "video.mkv",
               "truth": "truth.parquet" /* or "labels": "labels_native.parquet" */,
               "alignment": "alignment.parquet",
               "overlay_style": "none|input-display|nohboard" },
  "grid": { "engine_hz": 60 /* recorded */, "grid_hz": 60.0 /* foreign */ },
  "label_kind": "engine_truth|mapped",
  "masked_regions": [
    // "space" and "applied" are free-form declarations; the assembler
    // (theo/g1_assemble.py) currently writes
    // "capture-pixel (post-encode frame)" and
    // "not-applied (masking happens in build_dataset)".
    { "name": "frame_index_strip", "space": "capture_pixels",
      "applied": "post_crop", "rect_px": [x, y, w, h],
      "rect_norm": [x0, y0, x1, y1] }
    // + overlay rect on own data ("input_overlay",
    //   specs/overlay_spec.md); + controller-widget rect on foreign video
  ],
  "integrity": { "video_frames": N, "duplicates": N, "drops": N,
                 "sha256": { "video.mkv": "…", "truth.parquet": "…" } },
  "actions": { "keys": ["left","right","up","down","jump","dash","grab"] },
  "provenance": { "source": "recorded|nitrogen|wild", "origin_url": null,
                  "mapping_report": null }
}
```

Every masked region declares its coordinate space and whether the rect applies
before or after any crop/scale. `sha256` is over raw file bytes, listed per file.
Any model-facing frame must have all masked regions applied. Enforcement is
split: the session validator hard-requires only the `frame_index_strip`
entry; overlay masks are policy applied at shard construction, and the
validator does not currently cross-check declared overlay geometry against
rendered content (the 2026-07-26 mask-coverage defect passed validation for
exactly this reason).

Note (2026-07-26): a mask-coverage defect in assembled manifests — the
capture-pixel `input_overlay` rects undershot the rendered overlay cells —
left a readable sliver of the input overlay in own-data training shards. The
declared rects themselves verified zero; the coverage was incomplete. No
transferable benefit was observed on held-out sessions, but this does not rule
out training distortion; corrected manifests, rebuilt shards, and re-runs are
queued. See `report/findings_log.md` (2026-07-26 entry) and the note in
[specs/overlay_spec.md](overlay_spec.md).

## Public-export status (2026-07-26)

In the public repository, this document, the other specs, the
repository code, and the session index `data/sessions/INDEX.md` — which records
each recorded session's role/split assignment — are exported. Session
directories themselves (`manifest.json`, `video.mkv`, `truth.parquet`,
`alignment.parquet`, `labels_native.parquet`, and the raw mod CSV) are not
exported and exist only in the private working repository; frozen experiment
configs under `experiments/` export as `.py`/`.json`.
