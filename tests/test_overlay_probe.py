from __future__ import annotations

import numpy as np

from harvest.overlay_probe import (
    MIN_CELLS,
    bimodal_static_mask,
    build_probe_command,
    detect_overlay,
    find_cells,
)
from harvest.fetch_wild import FetchPolicy

H, W, T = 180, 320, 240
RNG = np.random.default_rng(0)


def test_probe_command_is_serial_polite_and_deno_backed(tmp_path) -> None:
    policy = FetchPolicy(
        yt_dlp_path="/opt/wildenv/bin/yt-dlp",
        deno_path="/usr/local/bin/deno",
    )
    command = build_probe_command(
        "https://example.invalid/watch?v=x", tmp_path / "probe.mp4", 10, 6, policy
    )
    assert command[0] == "/opt/wildenv/bin/yt-dlp"
    assert command[command.index("--js-runtimes") + 1] == "deno:/usr/local/bin/deno"
    assert command[command.index("--concurrent-fragments") + 1] == "1"
    assert "--sleep-requests" in command


def _game_frames() -> np.ndarray:
    """Scrolling textured content: everything moves, nothing is bimodal."""
    field = RNG.integers(0, 255, (H, W * 3), dtype=np.uint8)
    return np.stack([field[:, t : t + W] for t in range(T)])


def _with_overlay(frames: np.ndarray, n_cells: int = 7, duty: float = 0.3,
                  x0: int = 8, y0: int = 150) -> np.ndarray:
    """Paint a static panel with n_cells toggling between two levels."""
    out = frames.copy()
    panel_w = n_cells * 14 + 6
    out[:, y0 - 3 : y0 + 17, x0 - 3 : x0 + panel_w] = 10        # static panel
    for cell in range(n_cells):
        cx = x0 + cell * 14
        pressed = RNG.random(T) < duty
        for t in range(T):
            out[t, y0 : y0 + 14, cx : cx + 10] = 240 if pressed[t] else 40
    return out


def test_game_only_has_no_overlay() -> None:
    report = detect_overlay(_game_frames())
    assert report["has_overlay"] is False
    assert report["n_cells"] < MIN_CELLS


def test_overlay_is_detected_and_cells_located() -> None:
    report = detect_overlay(_with_overlay(_game_frames()))
    assert report["has_overlay"] is True
    assert report["n_cells"] >= MIN_CELLS
    # The panel must be found near where it was painted, not across the screen.
    x, y, w, h = report["panel_rect"]
    assert 0 <= x <= 20 and 140 <= y <= 165
    assert w < W * 0.5 and h < H * 0.5


def test_static_but_unimodal_region_is_not_a_cell() -> None:
    """A letterbox bar is static; it must not read as an overlay."""
    frames = _game_frames()
    frames[:, 160:180, :] = 0
    report = detect_overlay(frames)
    assert report["has_overlay"] is False


def test_never_pressed_decoration_is_rejected() -> None:
    """Cells that never toggle are decoration, not keys (duty gate)."""
    frames = _with_overlay(_game_frames(), duty=0.0)
    assert detect_overlay(frames)["has_overlay"] is False


def test_always_pressed_decoration_is_rejected() -> None:
    frames = _with_overlay(_game_frames(), duty=1.0)
    assert detect_overlay(frames)["has_overlay"] is False


def test_bimodal_mask_marks_cells_and_not_panel_background() -> None:
    frames = _with_overlay(_game_frames())
    mask, duty = bimodal_static_mask(frames)
    assert mask[155, 12]                       # inside a toggling cell
    assert not mask[148, 12]                   # static panel background
    assert 0.0 < duty[155, 12] < 1.0


def test_cells_are_rectangular_and_plausibly_sized() -> None:
    cells = find_cells(_with_overlay(_game_frames()))
    assert len(cells) >= MIN_CELLS
    for _, _, w, h in cells:
        assert 3 <= w <= W * 0.2 and 3 <= h <= H * 0.2


# --- behavioural identification: input HUD vs timer vs splits ---

from harvest.overlay_probe import cell_dynamics, score_input_hud  # noqa: E402


def _panel(kind: str) -> np.ndarray:
    """Three panels that differ only in how their cells behave over time."""
    out = _game_frames()
    x0, y0, n = 8, 150, 7
    out[:, y0 - 3 : y0 + 17, x0 - 3 : x0 + n * 14 + 6] = 10
    for cell in range(n):
        cx = x0 + cell * 14
        if kind == "input_hud":
            # Real keys are HELD for many frames (16-40 at 60Hz, measured on
            # our own sessions), not re-rolled every frame — so build runs.
            # Each cell gets its own hold/release lengths, giving the unequal
            # duty cycles that distinguish keys from digits.
            state = np.zeros(T, dtype=bool)
            hold = 6 + 4 * cell
            gap = 40 - 4 * cell
            t = 2 * cell
            while t < T:
                state[t : t + hold] = True
                t += hold + gap
        elif kind == "timer":
            # Every cell churns identically and fast, like digits.
            state = (np.arange(T) // (2 + cell % 2)) % 2 == 0
        else:                                   # splits: static for the window
            state = np.zeros(T, dtype=bool)
        for t in range(T):
            out[t, y0 : y0 + 14, cx : cx + 10] = 240 if state[t] else 40
    return out


def test_input_hud_is_identified_by_behaviour() -> None:
    frames = _panel("input_hud")
    report = detect_overlay(frames, fps=60.0)
    assert report["has_input_hud"] is True
    assert report["hud_panel"]["n_active_cells"] >= 3
    assert report["hud_panel"]["duty_spread"] >= 0.03


def test_timer_like_panel_is_rejected() -> None:
    """Uniform, fast, equal-duty cells are digits — not keys."""
    report = detect_overlay(_panel("timer"), fps=60.0)
    assert report["has_input_hud"] is False


def test_static_splits_panel_is_rejected() -> None:
    report = detect_overlay(_panel("splits"), fps=60.0)
    assert report["has_input_hud"] is False


def test_cell_dynamics_measures_rate_and_duty() -> None:
    frames = _panel("input_hud")
    cells = find_cells(frames)
    stats = cell_dynamics(frames, cells, fps=60.0)
    assert len(stats) == len(cells)
    assert all(0.0 <= s["duty"] <= 1.0 for s in stats)
    assert any(s["transitions_per_s"] > 0 for s in stats)


def test_score_input_hud_needs_several_unequal_cells() -> None:
    uniform = [{"transitions_per_s": 2.0, "duty": 0.5} for _ in range(6)]
    assert score_input_hud(uniform)["is_input_hud"] is False      # no spread
    varied = [{"transitions_per_s": 2.0, "duty": 0.05 + 0.15 * i} for i in range(5)]
    assert score_input_hud(varied)["is_input_hud"] is True
