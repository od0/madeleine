# Game instrumentation

`InputTruth/` is the `granny` instrumentation: a C# [Everest](https://everestapi.github.io/)
mod for *Celeste* that produces the project's engine truth. Per engine frame
(60 Hz) it records the seven controls and player state to a raw CSV log, and
it can render three instruments into the game image:

- a machine-readable **frame-index strip** (binary cells + checksum), always
  rendered, decoded from every captured video frame for clock-free
  video-to-engine alignment;
- an opaque **input overlay** showing the true key state, always rendered —
  the reference instrument for the label-degradation experiments;
- a translucent **wild-style action HUD** imitating a speedrunner's
  composited overlay — a calibration target for the wild-overlay decoder.
  This instrument is off by default and is enabled only through the Mod
  Options toggle "Wild-style translucent overlay (calibration only)"
  (`InputTruthSettings.WildOverlayEnabled`). Each session's `meta.json`
  declares whether it was on (`wild_overlay`) and, when on, its logical mask
  rectangle (`wild_overlay_rect_logical`); a toggle flipped mid-session is
  recorded as `wild_overlay_toggled_mid_session` so the assembler can refuse
  the session instead of masking a region that was only sometimes drawn.

The directory and assembly keep the compatibility name `InputTruth` because
recorded session metadata and installed capture setups reference it; the
component's project name is `granny`. The mod version is 0.2.0, declared
identically in `InputTruth/everest.yaml` and
`InputTruthModule.ModVersion`; the Everest manifest depends on
`EverestCore` 1.5935.0.

## Build and install

`InputTruth.csproj` targets `net8.0`, so building requires the .NET 8 SDK
plus an Everest-patched Celeste install to reference (`Celeste.dll`,
`MMHOOK_Celeste.dll`, and `FNA.dll`). The game path is read from the
`CELESTE_GAME_PATH` environment variable and defaults to
`/Applications/Celeste.app/Contents/Resources`; the build fails with an
explicit error if the three assemblies are not found there.

```bash
CELESTE_GAME_PATH=/path/to/celeste-resources \
  dotnet build -c Release granny/InputTruth/InputTruth.csproj
```

The build writes `InputTruth/bin/InputTruth.dll`, matching the `DLL` entry
in `everest.yaml`. Install by placing (or symlinking) the `InputTruth/`
directory — `everest.yaml` plus `bin/InputTruth.dll` — into Celeste's
`Mods/` directory.

The assembly also runs as a console tool for geometry introspection:
`dotnet run --project granny/InputTruth -- --print-vectors N ...` prints
frame-index strip patterns, and `--print-overlay-cells` prints the input
overlay cell rectangles as JSON.

## Output: raw mod CSV, not assembled sessions

Each run creates a session directory named `rec_YYYYMMDD_HHMMSS` under
`$INPUTTRUTH_OUT` (default `~/madeleine_sessions/inputtruth`) containing:

- `truth_raw.csv` — one row per engine frame: `frame_idx`, the seven
  controls (`left`, `right`, `up`, `down`, `jump`, `dash`, `grab`),
  `input_active`, `room_id`, position, speed, `dash_count`, `stamina`,
  `on_ground`, and `death`;
- `meta.json` — session id, mod/Everest/Celeste versions, overlay
  declarations, and wall-clock timestamps.

This raw CSV is not an assembled session. The session format —
`truth.parquet`, `alignment.parquet`, `manifest.json`, and captured video —
is produced by the capture and assembly tools in [`theo/`](../theo)
according to [`specs/session_format.md`](../specs/session_format.md).
