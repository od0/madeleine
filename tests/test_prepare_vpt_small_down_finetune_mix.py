from __future__ import annotations

from experiments.prepare_vpt_small_down_finetune_mix import select_replay_records


def _record(session_id: str, windows: int) -> dict[str, object]:
    return {"session_id": session_id, "windows": windows}


def test_replay_selection_is_deterministic_and_bounds_targeted_fraction() -> None:
    replay = [_record(f"replay-{index:02d}", 100) for index in range(30)]
    selected = select_replay_records(
        replay,
        targeted_windows=100,
        maximum_targeted_fraction=0.05,
        salt="ridge-down-finetune-v1",
    )
    assert sum(int(record["windows"]) for record in selected) == 1900
    assert selected == select_replay_records(
        list(reversed(replay)),
        targeted_windows=100,
        maximum_targeted_fraction=0.05,
        salt="ridge-down-finetune-v1",
    )


def test_replay_selection_refuses_insufficient_support() -> None:
    try:
        select_replay_records(
            [_record("only", 100)],
            targeted_windows=100,
            maximum_targeted_fraction=0.05,
            salt="ridge-down-finetune-v1",
        )
    except ValueError as error:
        assert "cannot satisfy" in str(error)
    else:
        raise AssertionError("insufficient replay support was accepted")
