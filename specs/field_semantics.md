# truth.parquet Field Semantics — v1

This table is part of the frozen session format
([specs/session_format.md](session_format.md)). The InputTruth mod (version
0.2.0, `InputTruthModule.ModVersion`, also declared in
`granny/InputTruth/everest.yaml`) implements it verbatim; downstream state
mining and frame filtering depend on every row of it. "Engine source" names are
the Celeste/Monocle members the value is read from, once per engine update,
after the update completes.

Implementing code:

- field capture: `granny/InputTruth/Source/InputTruthModule.cs`
  (`CaptureFrame`; `on_ground` is read through a compiled field accessor,
  `CreatePlayerOnGroundGetter`);
- row layout and CSV encoding: `granny/InputTruth/Source/InputFrameState.cs`
  and `granny/InputTruth/Source/TruthCsvWriter.cs`;
- CSV-to-Parquet conversion: `theo/g1_assemble.py` (`build_truth`), writing the
  Arrow schema in `data/schema.py` (`TRUTH_SCHEMA`).

| field | engine source | units / frame | during death animation | during room transition | during pause/menu | during cutscene |
|---|---|---|---|---|---|---|
| frame_idx | mod counter, incremented every `Engine.Update` | count from mod init; never resets | increments | increments | increments | increments |
| left/right/up/down | `Input.MoveX/MoveY` sign (bound aim/move state) | bool | recorded as held | recorded as held | recorded as held | recorded as held |
| jump | `Input.Jump.Check` | bool | recorded | recorded | recorded | recorded |
| dash | `Input.Dash.Check || Input.CrouchDash.Check` | bool | recorded | recorded | recorded | recorded |
| grab | `Input.Grab.Check` | bool | recorded | recorded | recorded | recorded |
| input_active | derived (below) | bool | **false** | **false** | **false** | **false** |
| room_id | `Level.Session.Level` | string | last room | source room until swap completes | unchanged | unchanged |
| pos_x/pos_y | `Player.Position` | level-global px | last value before player entity removed, then NaN | updates | frozen | updates |
| speed_x/speed_y | `Player.Speed` | px/s | NaN once player removed | updates | frozen | updates |
| dash_count | `Player.Dashes` | int | −1 once removed (integral; never NaN) | updates | frozen | updates |
| stamina | `Player.Stamina` | float | NaN once removed | updates | frozen | updates |
| on_ground | `Player.onGround` (reflection ok) | bool | false once removed | updates | frozen | updates |
| death | `Everest.Events.Player.OnDie` | bool, edge | **true only on the trigger frame** | false | false | false |
| session_id | assembler (see below) | string | — | — | — | — |

## Position is level-global

The mod records `Player.Position` unmodified — no room-local offset is applied
(`InputTruthModule.CaptureFrame`). Celeste entity positions are map-global, so
`pos_x`/`pos_y` are continuous across room transitions rather than resetting
per room. Verified 2026-07-26 on `rec_20260725_160450_b1`: across consecutive
`room_id` changes the position advances by ~1 px, e.g. `02-a → 02-b` at
(2589, −126) → (2589, −127).

## input_active

`input_active = playing && !paused && !transitioning && !cutscene && player exists`,
concretely: a `Level` scene is active, `Level.Paused == false`,
`Level.Transitioning == false`, `Level.InCutscene == false`, and the `Player`
entity is present and not dead. Keys are still **recorded** while
`input_active == false` (the player may be buffering inputs); consumers decide
what to do with those frames — the recorder never drops rows.

## Player-absent convention

When the player entity does not exist (death animation after removal, some
transitions): numeric player fields are NaN (dash_count uses −1 since it is
integral), booleans are false, and `input_active` is false. NaN/−1 are the
explicit "no player" markers; zero is never used as a placeholder.

## Non-Level scenes

Menus, overworld, save select: `room_id = ""`, player fields take the
player-absent convention, `input_active = false`, keys recorded as pressed.

## CSV encoding and Parquet conversion

The mod writes `truth_raw.csv` (`TruthCsvWriter.cs`) with 18 columns:
`frame_idx,left,right,up,down,jump,dash,grab,input_active,room_id,pos_x,pos_y,speed_x,speed_y,dash_count,stamina,on_ground,death`.
There is **no `session_id` column in the CSV**. Encoding:

- booleans are `1`/`0`;
- floats use .NET round-trip (`"R"`) invariant-culture formatting, so the
  player-absent sentinel serializes as the literal `NaN`;
- `dash_count` is a plain integer; the player-absent sentinel is the literal
  `-1`;
- `room_id` is CSV-quoted only when it contains a comma, quote, CR, or LF;
- UTF-8 without BOM, LF line endings.

`theo/g1_assemble.py` (`build_truth`) converts the CSV to `truth.parquet`
against `data/schema.py` `TRUTH_SCHEMA`: float columns are `float64` (values
originate as single-precision floats in the mod), `dash_count` is `int32`, and
`NaN`/`-1` sentinels are parsed as written. The `session_id` column is appended
at assembly time from the session directory name; the mod records its own
launch id separately in `meta.json`, and the two can differ when the operator
names the assembled session (e.g. `rec_20260725_160450_b1`).

Sessions recorded with mod 0.1.0 (the pre-state skeleton) have a CSV without
the state columns (detected by the absence of `input_active`); the assembler
reads `frame_idx` and the seven keys and fills the state columns with
documented placeholders (`input_active = true`, `room_id = ""`, zeroed numeric
state, `on_ground`/`death` false). Those placeholders are **not** the
player-absent sentinels; check `manifest.json` `env.mod` before interpreting
state fields.
