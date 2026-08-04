"""Quick label-only audit: what does 20 Hz decimation censor in engine truth?

Measures, per key, how many press events a 20 Hz (every-3rd-frame) view of
the native 60 Hz engine-truth labels would lose entirely or reduce to a
single sampled frame. This is the fast engine-truth slice of the aliasing
audit described in results/idm/VPT_TEMPORAL_RATE_ENGINEERING_NOTE.md
(section "Recommended next experiments", item A.5); the full three-modality
version lives alongside it.

Definitions:
  press EVENT  maximal run of consecutive pressed frames at native 60 Hz
               (runs never cross frames where the key is not pressed).
  LOST         under phase phi in {0,1,2}, the event contains zero native
               rows with frame_idx % 3 == phi.
  BLIP         the event contains exactly one such sampled row.

Reported loss/blip rates are averaged over all three phases so the answer
does not depend on which alignment a derived training set happened to use.

Sample policy: the final 54,000 frames (15 min at 60 Hz) of each listed
development-era session — the gameplay-dense region; early minutes of some
captures are idle. Sealed battery and untouched-test sessions are excluded
on principle: this audit must not touch spent or reserved surfaces.
"""
import numpy as np
import pyarrow.parquet as pq

SESSIONS = [
    "sessions/rec_20260724_031839_take2/truth.parquet",
    "sessions/rec_20260725_021338/truth.parquet",
]
KEYS = ["left", "right", "up", "down", "jump", "dash", "grab"]
N = 54_000  # 15 minutes at 60 Hz


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of maximal True runs."""
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def audit(path: str) -> None:
    full = pq.read_table(path)
    t = full.slice(full.num_rows - N, N)
    frame_idx = np.asarray(t["frame_idx"])
    active = np.asarray(t["input_active"]).astype(bool)
    print(f"session={path} frames={len(frame_idx)} "
          f"active={active.sum()} ({100 * active.mean():.1f}%)\n")
    print(f"{'key':>5} {'events':>6} {'1f':>5} {'2f':>5} {'3-5f':>6} "
          f"{'6-10f':>6} {'>10f':>6} {'<3f%':>6} {'lost%':>6} "
          f"{'blip%':>6} {'medms':>6}")
    tot = {"events": 0, "lost": 0.0, "blip": 0.0, "sub3": 0}
    for k in KEYS:
        m = np.asarray(t[k]).astype(bool) & active
        ev = runs(m)
        if not ev:
            print(f"{k:>5} {0:>6}")
            continue
        dur = np.array([e - s + 1 for s, e in ev])
        lost = blip = 0.0
        for phase in range(3):
            n_in = np.array([np.sum(frame_idx[s:e + 1] % 3 == phase)
                             for s, e in ev])
            lost += np.mean(n_in == 0)
            blip += np.mean(n_in == 1)
        lost, blip = 100 * lost / 3, 100 * blip / 3
        b = [np.sum(dur == 1), np.sum(dur == 2),
             np.sum((dur >= 3) & (dur <= 5)),
             np.sum((dur >= 6) & (dur <= 10)), np.sum(dur > 10)]
        print(f"{k:>5} {len(ev):>6} {b[0]:>5} {b[1]:>5} {b[2]:>6} "
              f"{b[3]:>6} {b[4]:>6} {100 * np.mean(dur < 3):>6.1f} "
              f"{lost:>6.1f} {blip:>6.1f} "
              f"{np.median(dur) / 60 * 1000:>6.0f}")
        tot["events"] += len(ev)
        tot["lost"] += lost * len(ev)
        tot["blip"] += blip * len(ev)
        tot["sub3"] += int(np.sum(dur < 3))
    e = tot["events"]
    print(f"\nALL   events={e}  sub-3-frame={100 * tot['sub3'] / e:.1f}%  "
          f"phase-avg lost={tot['lost'] / e:.1f}%  "
          f"blip={tot['blip'] / e:.1f}%\n")


if __name__ == "__main__":
    print("1 native frame = 16.7 ms; 20 Hz samples every 50 ms; "
          "sub-3-frame events are <50 ms taps\n")
    for s in SESSIONS:
        audit(s)
