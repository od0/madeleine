from pathlib import Path

import numpy as np
import torch

from experiments.eval_gru_wild7 import (
    context_supported,
    load_or_encode_features,
    pixel_predictions,
    selected_stream_metadata,
)


def test_context_supported_requires_full_centered_stride_three_window() -> None:
    engine = np.arange(500, dtype=np.int64)
    rows = np.asarray([188, 189, 307, 308], dtype=np.int64)

    supported = context_supported(
        rows,
        engine,
        window=128,
        frame_stride=3,
    )

    assert supported.tolist() == [False, True, True, False]


def test_context_supported_never_crosses_engine_gap() -> None:
    engine = np.concatenate(
        (np.arange(400, dtype=np.int64), np.arange(1_000, 1_400, dtype=np.int64))
    )
    rows = np.asarray([200, 399, 400, 589], dtype=np.int64)

    supported = context_supported(
        rows,
        engine,
        window=128,
        frame_stride=3,
    )

    assert supported.tolist() == [True, False, False, True]


def test_selected_stream_metadata_drops_empty_streams() -> None:
    reference = {
        "session_lengths": np.asarray([3, 2, 4], dtype=np.int64),
        "session_ids": np.asarray(["a", "b", "c"]),
    }
    selected = np.asarray(
        [False, True, True, False, False, True, False, True, False],
        dtype=bool,
    )

    lengths, ids = selected_stream_metadata(reference, selected)

    assert lengths.tolist() == [2, 2]
    assert ids.tolist() == ["a", "c"]


class _FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, frames: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.full((len(frames), 512), self.calls, dtype=np.float16)


def test_feature_cache_is_reused(tmp_path: Path) -> None:
    frames = np.zeros((4, 128, 128, 3), dtype=np.uint8)
    cache = tmp_path / "features.npy"
    encoder = _FakeEncoder()

    first = load_or_encode_features(encoder, frames, cache)
    second = load_or_encode_features(encoder, frames, cache)

    assert encoder.calls == 1
    assert np.array_equal(first, second)
    assert cache.is_file()


class _FakePixelModel:
    window = 3
    frame_stride = 1
    window_mode = "centered"

    def forward_segment(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        frames = batch["frames"]
        target_values = frames[:, 1:-1, 0, 0, 0]
        return target_values.unsqueeze(-1).repeat(1, 1, 7)


def test_pixel_predictions_preserve_requested_order() -> None:
    frames = np.zeros((12, 128, 128, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(12, dtype=np.uint8)
    targets = np.asarray([7, 4], dtype=np.int64)

    probability = pixel_predictions(
        _FakePixelModel(),
        frames,
        targets,
        dense_target_count=8,
        device=torch.device("cpu"),
    )

    expected = torch.sigmoid(torch.tensor([7 / 255, 4 / 255])).numpy()
    assert np.allclose(probability[:, 0], expected)
    assert np.allclose(probability, probability[:, :1])
