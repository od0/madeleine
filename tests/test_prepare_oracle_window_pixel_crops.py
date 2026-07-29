from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.oracle_window_localization import OracleExample
from experiments.prepare_oracle_window_pixel_crops import (
    downsample_rgb_area,
    extract_crops,
    validate_cache_archive,
)


def _example(*, crop_start: int, offset: int = 0) -> OracleExample:
    return OracleExample(
        split="train",
        session_id="session-a",
        run_index=0,
        array_index=crop_start + 8 + offset,
        engine_frame_idx=100 + crop_start + 8 + offset,
        head_index=0,
        key_index=0,
        event_type_index=0,
        offset=offset,
        crop_start=crop_start,
        candidate_start=crop_start + 8,
        block_id="session-a:run0:block0",
    )


def test_integer_area_downsampling_is_exact_and_channel_preserving() -> None:
    pixels = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    pixels[:, :2, :2] = np.asarray([1, 2, 3], dtype=np.uint8)
    pixels[:, :2, 2:] = np.asarray([4, 5, 6], dtype=np.uint8)
    pixels[:, 2:, :2] = np.asarray([7, 8, 9], dtype=np.uint8)
    pixels[:, 2:, 2:] = np.asarray([10, 11, 12], dtype=np.uint8)
    observed = downsample_rgb_area(pixels, output_size=2)
    assert observed.dtype == np.uint8
    assert observed.tolist() == [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]


def test_extract_crops_uses_only_declared_rows() -> None:
    frames = np.zeros((40, 4, 4, 3), dtype=np.uint8)
    frames[:] = np.arange(40, dtype=np.uint8)[:, None, None, None]
    examples = [_example(crop_start=0), _example(crop_start=8)]
    observed = extract_crops(
        frames, examples, crop_frames=32, output_size=2, batch_examples=1
    )
    assert observed.shape == (2, 32, 2, 2, 3)
    assert np.array_equal(observed[0, :, 0, 0, 0], np.arange(32))
    assert np.array_equal(observed[1, :, 0, 0, 0], np.arange(8, 40))


def test_cache_validator_rejects_changed_example_identity(tmp_path: Path) -> None:
    example = _example(crop_start=0, offset=3)
    rgb = np.zeros((1, 32, 2, 2, 3), dtype=np.uint8)
    from experiments.oracle_window_localization import _example_arrays

    path = tmp_path / "train.npz"
    np.savez(path, rgb=rgb, **_example_arrays([example]))
    receipt = validate_cache_archive(
        path, expected_examples=[example], crop_frames=32, output_size=2
    )
    assert receipt["examples"] == 1

    changed = replace(example, engine_frame_idx=example.engine_frame_idx + 1)
    with pytest.raises(ValueError, match="metadata changed"):
        validate_cache_archive(
            path, expected_examples=[changed], crop_frames=32, output_size=2
        )


def test_downsampling_rejects_nondivisible_geometry() -> None:
    with pytest.raises(ValueError, match="divisible"):
        downsample_rgb_area(np.zeros((1, 5, 5, 3), np.uint8), output_size=2)
