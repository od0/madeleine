"""Train Badeline from explicit session-level split lists."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

# Must be set before the first CUDA context exists; required for
# deterministic cuBLAS matmuls under use_deterministic_algorithms(True).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from badeline.model import BadelineIDM, INPUT_CONFIGS, WINDOW_MODES
from data.schema import KEY_ORDER


@dataclass(frozen=True)
class SessionArrays:
    session_id: str
    frames: np.ndarray
    keys: np.ndarray
    engine_frame_idx: np.ndarray | None = None
    input_active: np.ndarray | None = None

    def __post_init__(self) -> None:
        frame_count = len(self.frames)
        engine_frame_idx = (
            np.arange(frame_count, dtype=np.int64)
            if self.engine_frame_idx is None
            else np.asarray(self.engine_frame_idx, dtype=np.int64)
        )
        input_active = (
            np.ones(frame_count, dtype=np.uint8)
            if self.input_active is None
            else np.asarray(self.input_active, dtype=np.uint8)
        )
        if engine_frame_idx.shape != (frame_count,):
            raise ValueError("engine_frame_idx must have shape [N]")
        if input_active.shape != (frame_count,):
            raise ValueError("input_active must have shape [N]")
        object.__setattr__(self, "engine_frame_idx", engine_frame_idx)
        object.__setattr__(self, "input_active", input_active)


def read_session_ids(path: str | Path) -> list[str]:
    """Read an explicit session-id list without discovering other sessions."""

    ids = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not ids:
        raise ValueError(f"empty session list: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate session id in list: {path}")
    return ids


def validate_splits(train_ids: Sequence[str], val_ids: Sequence[str]) -> None:
    overlap = sorted(set(train_ids).intersection(val_ids))
    if overlap:
        raise ValueError(f"overlapping split: {', '.join(overlap)}")


def target_offset(window: int, window_mode: str) -> int:
    """Return the target's offset within a contiguous input window."""

    if window < 1:
        raise ValueError("window must be at least 1")
    if window_mode == "centered":
        # For even windows, use the left-middle frame so future evidence remains.
        return (window - 1) // 2
    if window_mode == "past_only":
        return window - 1
    raise ValueError(f"window_mode must be one of {WINDOW_MODES}")


def load_session(
    data_dir: str | Path,
    session_id: str,
    *,
    precomputed_features: bool = False,
) -> SessionArrays:
    """Load exactly one requested NPZ shard."""

    shard = Path(data_dir) / f"{session_id}.npz"
    if not shard.is_file():
        raise FileNotFoundError(f"missing requested session shard: {shard}")

    with np.load(shard, allow_pickle=False) as archive:
        visual_array = "features" if precomputed_features else "frames"
        required = {
            visual_array, "keys", "engine_frame_idx", "input_active", "session_id"
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{shard}: missing arrays: {sorted(missing)}")
        frames = archive[visual_array]
        keys = archive["keys"]
        engine_frame_idx = archive["engine_frame_idx"]
        input_active = archive["input_active"]
        stored_id_array = archive["session_id"]

    if stored_id_array.size != 1:
        raise ValueError(f"{shard}: session_id must contain one string")
    stored_id = str(stored_id_array.reshape(()).item())
    if stored_id != session_id:
        raise ValueError(
            f"{shard}: stored session_id {stored_id!r} does not match {session_id!r}"
        )
    if precomputed_features:
        if frames.dtype not in (np.float16, np.float32) or frames.ndim != 2:
            raise ValueError(f"{shard}: features must be float16/32 [N,D]")
    else:
        if frames.dtype != np.uint8 or frames.ndim != 4:
            raise ValueError(f"{shard}: frames must be uint8 [N,128,128,3]")
        if frames.shape[1:] != (128, 128, 3):
            raise ValueError(f"{shard}: frames must have shape [N,128,128,3]")
    if keys.dtype != np.uint8 or keys.ndim != 2:
        raise ValueError(f"{shard}: keys must be uint8 [N,7]")
    if keys.shape != (frames.shape[0], len(KEY_ORDER)):
        raise ValueError(f"{shard}: keys must have shape [N,{len(KEY_ORDER)}]")

    if engine_frame_idx.dtype != np.int64:
        raise ValueError(f"{shard}: engine_frame_idx must be int64 [N]")
    if input_active.dtype != np.uint8:
        raise ValueError(f"{shard}: input_active must be uint8 [N]")

    return SessionArrays(
        stored_id, frames, keys, engine_frame_idx, input_active
    )


def contiguous_runs(engine_frame_idx: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open array ranges with strictly consecutive frame indices."""

    frame_idx = np.asarray(engine_frame_idx, dtype=np.int64)
    if frame_idx.ndim != 1:
        raise ValueError("engine_frame_idx must be one-dimensional")
    if not len(frame_idx):
        return []
    delta = np.diff(frame_idx)
    # A capture may span a mod/game relaunch and therefore reset the rendered
    # engine counter. Resets and accidental duplicate indices are boundaries,
    # just like positive gaps; no temporal window may cross any of them.
    boundaries = np.flatnonzero(delta != 1) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(frame_idx)]))
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


class WindowedSessionDataset(Dataset[dict[str, torch.Tensor]]):
    """Contiguous within-session windows with mode-dependent target alignment.

    ``frame_stride`` samples the visual window on a dilated grid while keeping
    adjacent training targets one engine frame apart.  At 60 Hz, 128 samples
    with stride 3 span 382 raw frames, or about 6.37 seconds.
    """

    def __init__(
        self,
        sessions: Sequence[SessionArrays],
        *,
        window: int,
        window_mode: str,
        input_config: str,
        history_len: int,
        history_gap: int = 0,
        active_targets_only: bool = True,
        transition_weight: float = 1.0,
        precomputed_features: bool = False,
        frame_stride: int = 1,
    ) -> None:
        if input_config not in INPUT_CONFIGS:
            raise ValueError(f"input_config must be one of {INPUT_CONFIGS}")
        if input_config == "state_meta":
            raise NotImplementedError("state_meta lands with real data")
        if history_len < 1:
            raise ValueError("history_len must be at least 1")
        self.sessions = list(sessions)
        self.window = window
        self.window_mode = window_mode
        self.input_config = input_config
        self.history_len = history_len
        # A gap between the history window and the target frame. With gap=0 the
        # "history" arm is a persistence baseline (keys[t-1] predicts keys[t] at
        # ~0.91 macro-AP because inputs are held for 16-40 frames at 60Hz). A
        # gap makes it measure a genuine policy/route prior instead of copying.
        self.history_gap = max(0, int(history_gap))
        self.active_targets_only = bool(active_targets_only)
        self.transition_weight = float(transition_weight)
        self.precomputed_features = bool(precomputed_features)
        self.frame_stride = int(frame_stride)
        if self.transition_weight < 1.0:
            raise ValueError("transition_weight must be at least 1")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        self.target_in_window = target_offset(window, window_mode)
        self.frame_span = (self.window - 1) * self.frame_stride + 1

        self._locations: list[tuple[int, int, int]] = []
        for session_index, session in enumerate(self.sessions):
            assert session.engine_frame_idx is not None
            assert session.input_active is not None
            for run_start, run_end in contiguous_runs(session.engine_frame_idx):
                for start in range(run_start, run_end - self.frame_span + 1):
                    target_index = (
                        start + self.target_in_window * self.frame_stride
                    )
                    if self.active_targets_only and not session.input_active[target_index]:
                        continue
                    self._locations.append((session_index, start, run_start))
        if not self._locations:
            raise ValueError(
                f"no contiguous {window}-frame window has an active target"
            )

    def __len__(self) -> int:
        return len(self._locations)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        session_index, start, run_start = self._locations[index]
        session = self.sessions[session_index]
        target_index = start + self.target_in_window * self.frame_stride

        example: dict[str, torch.Tensor] = {
            "target": torch.from_numpy(
                session.keys[target_index].astype(np.float32, copy=True)
            )
        }
        loss_weight = np.ones(len(KEY_ORDER), dtype=np.float32)
        if target_index > run_start and self.transition_weight > 1.0:
            changed = session.keys[target_index] != session.keys[target_index - 1]
            loss_weight[changed] = self.transition_weight
        example["loss_weight"] = torch.from_numpy(loss_weight)
        if self.input_config in ("pixels", "pixels_plus_history"):
            visual = torch.from_numpy(
                session.frames[
                    start : start + self.frame_span : self.frame_stride
                ].copy()
            )
            if self.precomputed_features:
                example["features"] = visual.to(dtype=torch.float32)
            else:
                example["frames"] = visual.permute(0, 3, 1, 2).to(
                    dtype=torch.float32
                ).div_(255.0)

        if self.input_config in ("history", "pixels_plus_history"):
            history = np.zeros(
                (self.history_len, len(KEY_ORDER)), dtype=np.float32
            )
            history_end = max(run_start, target_index - self.history_gap)
            history_start = max(run_start, history_end - self.history_len)
            available = session.keys[history_start:history_end]
            if len(available):
                history[-len(available) :] = available
            example["history"] = torch.from_numpy(history)

        return example


def history_block(
    keys: np.ndarray,
    target_indices: Sequence[int],
    history_len: int,
    history_gap: int,
    floor: int = 0,
) -> np.ndarray:
    """Per-target gapped key history, left-padded with zeros: [S, L, 7]."""

    block = np.zeros(
        (len(target_indices), history_len, keys.shape[1]), dtype=np.float32
    )
    for row, target_index in enumerate(target_indices):
        end = max(floor, int(target_index) - history_gap)
        start = max(floor, end - history_len)
        available = keys[start:end]
        if len(available):
            block[row, -len(available):] = available
    return block


class SegmentSessionDataset(Dataset[dict[str, torch.Tensor]]):
    """Contiguous segments of S windows sharing per-frame work (brief v3.2).

    One item carries the contiguous raw span under S consecutive windows;
    the model's ``forward_segment`` encodes each frame once and assembles
    possibly dilated windows by index. The windowed dataset above encodes every
    frame once PER WINDOW — a 16x inflation at window 16 — and survives only
    for history-only configs and the equivalence test. Trailing windows that
    do not fill a segment are dropped (training coverage; eval sweeps
    sessions exactly via badeline.eval's segment loop).
    """

    def __init__(
        self,
        sessions: Sequence[SessionArrays],
        *,
        window: int,
        window_mode: str,
        input_config: str,
        history_len: int,
        history_gap: int = 0,
        segment_windows: int = 48,
        active_targets_only: bool = True,
        transition_weight: float = 1.0,
        precomputed_features: bool = False,
        frame_stride: int = 1,
    ) -> None:
        if input_config not in INPUT_CONFIGS:
            raise ValueError(f"input_config must be one of {INPUT_CONFIGS}")
        if segment_windows < 1:
            raise ValueError("segment_windows must be at least 1")
        self.sessions = list(sessions)
        self.window = window
        self.input_config = input_config
        self.history_len = history_len
        self.history_gap = max(0, int(history_gap))
        self.segment_windows = segment_windows
        self.active_targets_only = bool(active_targets_only)
        self.transition_weight = float(transition_weight)
        self.precomputed_features = bool(precomputed_features)
        self.frame_stride = int(frame_stride)
        if self.transition_weight < 1.0:
            raise ValueError("transition_weight must be at least 1")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        self.target_in_window = target_offset(window, window_mode)
        self.frame_span = (self.window - 1) * self.frame_stride + 1

        self._locations: list[tuple[int, int, int]] = []
        for session_index, session in enumerate(self.sessions):
            assert session.engine_frame_idx is not None
            assert session.input_active is not None
            for run_start, run_end in contiguous_runs(session.engine_frame_idx):
                eligible: list[int] = []
                for start in range(run_start, run_end - self.frame_span + 1):
                    target_index = (
                        start + self.target_in_window * self.frame_stride
                    )
                    active = bool(session.input_active[target_index])
                    if not self.active_targets_only or active:
                        eligible.append(start)
                    elif eligible:
                        self._append_segments(session_index, run_start, eligible)
                        eligible = []
                if eligible:
                    self._append_segments(session_index, run_start, eligible)
        if not self._locations:
            raise ValueError(
                "no session is long enough for one "
                f"{segment_windows}-window segment"
            )

    def _append_segments(
        self, session_index: int, run_start: int, eligible: Sequence[int]
    ) -> None:
        """Append full segments from one consecutive sequence of valid starts."""

        for offset in range(0, len(eligible) - self.segment_windows + 1,
                            self.segment_windows):
            starts = eligible[offset : offset + self.segment_windows]
            if starts[-1] - starts[0] != self.segment_windows - 1:
                raise AssertionError("segment window starts are not consecutive")
            self._locations.append((session_index, starts[0], run_start))

    def __len__(self) -> int:
        return len(self._locations)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        session_index, start, run_start = self._locations[index]
        session = self.sessions[session_index]
        count = self.segment_windows
        target_indices = [
            start + s + self.target_in_window * self.frame_stride
            for s in range(count)
        ]
        example: dict[str, torch.Tensor] = {
            "target": torch.from_numpy(
                session.keys[
                    target_indices[0] : target_indices[0] + count
                ].astype(np.float32, copy=True)
            )
        }
        loss_weight = np.ones((count, len(KEY_ORDER)), dtype=np.float32)
        if self.transition_weight > 1.0:
            target_array = np.asarray(target_indices)
            can_compare = target_array > run_start
            changed = np.zeros_like(loss_weight, dtype=bool)
            changed[can_compare] = (
                session.keys[target_array[can_compare]]
                != session.keys[target_array[can_compare] - 1]
            )
            loss_weight[changed] = self.transition_weight
        example["loss_weight"] = torch.from_numpy(loss_weight)
        if self.input_config in ("pixels", "pixels_plus_history"):
            visual = torch.from_numpy(
                session.frames[
                    start : start + count + self.frame_span - 1
                ].copy()
            )
            if self.precomputed_features:
                example["features"] = visual.to(dtype=torch.float32)
            else:
                example["frames"] = visual.permute(0, 3, 1, 2).to(
                    dtype=torch.float32
                ).div_(255.0)
        if self.input_config in ("history", "pixels_plus_history"):
            example["history"] = torch.from_numpy(history_block(
                session.keys, target_indices,
                self.history_len, self.history_gap,
                floor=run_start,
            ))
        return example


class _CyclingSourcePool:
    """Deterministically shuffle and cycle one source's segment indices."""

    def __init__(self, indices: Sequence[int], *, seed: int) -> None:
        if not indices:
            raise ValueError("source sampling pool must not be empty")
        self.indices = [int(index) for index in indices]
        if len(self.indices) != len(set(self.indices)):
            raise ValueError("source sampling pool contains duplicate items")
        self.generator = torch.Generator().manual_seed(int(seed))
        self.order: list[int] = []
        self.cursor = 0
        self.draw_counts = {index: 0 for index in self.indices}
        self.completed_pool_passes = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        permutation = torch.randperm(
            len(self.indices), generator=self.generator
        ).tolist()
        self.order = [self.indices[position] for position in permutation]
        self.cursor = 0

    def draw(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("source draw count must be non-negative")
        drawn: list[int] = []
        while len(drawn) < count:
            available = len(self.order) - self.cursor
            take = min(count - len(drawn), available)
            block = self.order[self.cursor : self.cursor + take]
            drawn.extend(block)
            self.cursor += take
            for index in block:
                self.draw_counts[index] += 1
            if self.cursor == len(self.order):
                self.completed_pool_passes += 1
                self._reshuffle()
        return drawn


def _derived_sampling_seed(seed: int, name: str) -> int:
    payload = f"{int(seed)}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class DeterministicSourceBatchSampler(Sampler[list[int]]):
    """Emit an exact, preregistered source mixture for a fixed run.

    Each source owns an independently shuffled pool of segment items. A pool
    is consumed without replacement, reshuffled with its private deterministic
    generator when exhausted, and then cycled. The step cycle fixes exact
    counts rather than merely matching proportions in expectation.

    This sampler is deliberately one-shot. Report-grade training consumes the
    exact declared number of steps once; a second iterator would silently add
    another exposure epoch and is therefore refused.
    """

    def __init__(
        self,
        source_items: Mapping[str, Sequence[int]],
        *,
        step_cycle: Sequence[Mapping[str, int]],
        steps: int,
        seed: int,
        expected_cycle_steps: int,
        expected_cycle_items: int,
        source_session_counts: Mapping[str, int] | None = None,
    ) -> None:
        if steps < 1:
            raise ValueError("source sampling requires at least one step")
        if expected_cycle_steps < 1 or expected_cycle_items < 1:
            raise ValueError("source sampling cycle expectations must be positive")
        if len(step_cycle) != expected_cycle_steps:
            raise ValueError("source sampling cycle step count changed")
        if steps % expected_cycle_steps:
            raise ValueError(
                "source sampling steps must be divisible by the cycle length"
            )

        source_names = set(source_items)
        if not source_names:
            raise ValueError("source sampling requires at least one source")
        normalized_cycle: list[dict[str, int]] = []
        batch_items: int | None = None
        for row in step_cycle:
            if set(row) != source_names:
                raise ValueError(
                    "every source sampling row must name the exact source set"
                )
            normalized = {name: int(row[name]) for name in sorted(source_names)}
            if any(value < 0 for value in normalized.values()):
                raise ValueError("source sampling counts must be non-negative")
            row_items = sum(normalized.values())
            if row_items < 1:
                raise ValueError("source sampling rows must not be empty")
            if batch_items is None:
                batch_items = row_items
            elif row_items != batch_items:
                raise ValueError("source sampling batch size changed within cycle")
            normalized_cycle.append(normalized)
        assert batch_items is not None
        if sum(sum(row.values()) for row in normalized_cycle) != expected_cycle_items:
            raise ValueError("source sampling cycle item count changed")

        flattened = [
            int(index)
            for indices in source_items.values()
            for index in indices
        ]
        if len(flattened) != len(set(flattened)):
            raise ValueError("a dataset item belongs to multiple source pools")
        self.source_names = sorted(source_names)
        self.step_cycle = normalized_cycle
        self.steps = int(steps)
        self.seed = int(seed)
        self.batch_items = batch_items
        self.expected_cycle_steps = int(expected_cycle_steps)
        self.expected_cycle_items = int(expected_cycle_items)
        self.source_session_counts = {
            name: int((source_session_counts or {}).get(name, 0))
            for name in self.source_names
        }
        self.pools = {
            name: _CyclingSourcePool(
                source_items[name],
                seed=_derived_sampling_seed(self.seed, f"source:{name}"),
            )
            for name in self.source_names
        }
        cycle_repetitions = self.steps // len(self.step_cycle)
        self.scheduled_draws = {
            name: cycle_repetitions
            * sum(row[name] for row in self.step_cycle)
            for name in self.source_names
        }
        if any(value < 1 for value in self.scheduled_draws.values()):
            raise ValueError("every declared source must be scheduled")
        self.actual_draws = {name: 0 for name in self.source_names}
        self.steps_emitted = 0
        self._iterated = False
        self.batch_generator = torch.Generator().manual_seed(
            _derived_sampling_seed(self.seed, "batch-order")
        )

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        if self._iterated:
            raise RuntimeError("source batch sampler is one-shot")
        self._iterated = True
        for step in range(self.steps):
            row = self.step_cycle[step % len(self.step_cycle)]
            batch: list[int] = []
            for name in self.source_names:
                count = row[name]
                batch.extend(self.pools[name].draw(count))
                self.actual_draws[name] += count
            if len(batch) != self.batch_items:
                raise AssertionError("source sampling emitted wrong batch size")
            order = torch.randperm(
                len(batch), generator=self.batch_generator
            ).tolist()
            self.steps_emitted += 1
            yield [batch[position] for position in order]

    def receipt(self, *, require_complete: bool) -> dict[str, Any]:
        if require_complete:
            if self.steps_emitted != self.steps:
                raise RuntimeError("source sampling did not emit every scheduled step")
            if self.actual_draws != self.scheduled_draws:
                raise RuntimeError("source sampling actual draws changed")

        sources: dict[str, Any] = {}
        for name in self.source_names:
            pool = self.pools[name]
            counts = list(pool.draw_counts.values())
            unique = sum(value > 0 for value in counts)
            actual = self.actual_draws[name]
            sources[name] = {
                "session_count": self.source_session_counts[name],
                "segment_items": len(pool.indices),
                "scheduled_draws": self.scheduled_draws[name],
                "actual_draws": actual,
                "unique_segment_items_drawn": unique,
                "repeat_draws": max(0, actual - unique),
                "effective_pool_passes": actual / len(pool.indices),
                "completed_pool_passes": pool.completed_pool_passes,
                "minimum_draws_per_item": min(counts),
                "maximum_draws_per_item": max(counts),
                "mean_draws_per_item": actual / len(pool.indices),
            }
        return {
            "format_version": "madeleine.source-balanced-batch.v1",
            "seed": self.seed,
            "cycle_steps": self.expected_cycle_steps,
            "cycle_items": self.expected_cycle_items,
            "batch_items": self.batch_items,
            "scheduled_steps": self.steps,
            "actual_steps": self.steps_emitted,
            "step_cycle": self.step_cycle,
            "complete": self.steps_emitted == self.steps
            and self.actual_draws == self.scheduled_draws,
            "sources": sources,
        }


def build_source_batch_sampler(
    dataset: Dataset,
    train_ids: Sequence[str],
    config: Mapping[str, Any],
    *,
    steps: int,
    seed: int,
    expected_batch_items: int,
) -> DeterministicSourceBatchSampler | None:
    """Build the optional exact-mixture sampler, rejecting fuzzy membership."""

    raw = config.get("source_sampling")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("source_sampling must be an object")
    if not isinstance(dataset, SegmentSessionDataset):
        raise ValueError("source_sampling requires the segment dataset path")
    if raw.get("format_version") != "madeleine.source-balanced-batch.v1":
        raise ValueError("unsupported source_sampling format_version")
    if int(raw.get("expected_steps", -1)) != steps:
        raise ValueError("source_sampling expected_steps changed")

    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, Mapping) or not raw_sources:
        raise ValueError("source_sampling.sources must be a non-empty object")
    train_set = set(train_ids)
    session_source: dict[str, str] = {}
    source_session_counts: dict[str, int] = {}
    for raw_name, raw_ids in raw_sources.items():
        name = str(raw_name)
        if not name or not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError("every source must list at least one session")
        source_session_counts[name] = len(raw_ids)
        for raw_id in raw_ids:
            session_id = str(raw_id)
            if session_id in session_source:
                raise ValueError("a training session belongs to multiple sources")
            session_source[session_id] = name
    if set(session_source) != train_set:
        missing = sorted(train_set - set(session_source))
        extra = sorted(set(session_source) - train_set)
        raise ValueError(
            f"source_sampling session membership mismatch: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    source_items: dict[str, list[int]] = {
        name: [] for name in source_session_counts
    }
    for item_index, (session_index, _, _) in enumerate(dataset._locations):
        session_id = dataset.sessions[session_index].session_id
        source_items[session_source[session_id]].append(item_index)
    empty = sorted(name for name, items in source_items.items() if not items)
    if empty:
        raise ValueError(f"source_sampling sources have no segment items: {empty}")

    step_cycle = raw.get("step_cycle")
    if not isinstance(step_cycle, list) or not step_cycle:
        raise ValueError("source_sampling.step_cycle must be a non-empty list")
    sampler = DeterministicSourceBatchSampler(
        source_items,
        step_cycle=step_cycle,
        steps=steps,
        seed=seed,
        expected_cycle_steps=int(raw.get("cycle_steps", -1)),
        expected_cycle_items=int(raw.get("cycle_items", -1)),
        source_session_counts=source_session_counts,
    )
    if sampler.batch_items != expected_batch_items:
        raise ValueError("source_sampling batch size differs from config")
    return sampler


def resolve_positive_weight(
    config: Mapping[str, Any],
    train_arrays: Sequence[SessionArrays],
) -> list[float] | None:
    """Resolve computed or explicitly frozen BCE positive weights.

    ``frozen_positive_weight`` is opt-in and keyed by the canonical key order.
    It is useful for corpus-mixture ablations where recomputing class weights
    would change the objective at the same time as the sampled examples.
    """

    class_balance = bool(config.get("class_balance", False))
    frozen = config.get("frozen_positive_weight")
    if frozen is not None:
        if not class_balance:
            raise ValueError(
                "frozen_positive_weight requires class_balance=true"
            )
        if not isinstance(frozen, Mapping) or set(frozen) != set(KEY_ORDER):
            raise ValueError(
                "frozen_positive_weight must name the exact canonical key set"
            )
        maximum = float(config.get("class_balance_max", 20.0))
        values = [float(frozen[key]) for key in KEY_ORDER]
        if not np.all(np.isfinite(values)):
            raise ValueError("frozen_positive_weight must be finite")
        if any(value < 1.0 or value > maximum for value in values):
            raise ValueError(
                "frozen_positive_weight must lie in [1, class_balance_max]"
            )
        return values
    if not class_balance:
        return None

    positive = np.zeros(len(KEY_ORDER), dtype=np.float64)
    active_count = 0
    for session in train_arrays:
        assert session.input_active is not None
        active = session.input_active.astype(bool)
        positive += session.keys[active].sum(axis=0)
        active_count += int(active.sum())
    maximum = float(config.get("class_balance_max", 20.0))
    if maximum < 1.0:
        raise ValueError("class_balance_max must be at least 1")
    ratio = (active_count - positive) / np.maximum(positive, 1.0)
    return np.clip(ratio, 1.0, maximum).tolist()


def _move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    inputs = {
        name: value.to(device)
        for name, value in batch.items()
        if name in ("frames", "features", "history")
    }
    targets = batch["target"].to(device)
    loss_weight = batch.get("loss_weight", torch.ones_like(targets)).to(device)
    return inputs, targets, loss_weight


@torch.no_grad()
def evaluate_per_key(
    model: BadelineIDM,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> list[float]:
    was_training = model.training
    model.eval()
    total = torch.zeros(len(KEY_ORDER), dtype=torch.float64)
    count = 0
    for batch in loader:
        inputs, targets, _ = _move_batch(batch, device)
        logits = (
            model.forward_segment(inputs) if targets.ndim == 3 else model(inputs)
        )
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        flat = losses.detach().cpu().to(torch.float64).reshape(-1, len(KEY_ORDER))
        total += flat.sum(dim=0)
        count += flat.shape[0]
    model.train(was_training)
    if count == 0:
        raise ValueError("cannot evaluate an empty dataset")
    return (total / count).tolist()


def _loss_dict(values: Sequence[float]) -> dict[str, float]:
    return {key: float(value) for key, value in zip(KEY_ORDER, values, strict=True)}


def _cycle(
    loader: DataLoader[dict[str, torch.Tensor]],
) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def _write_log(
    log_file: Any,
    *,
    step: int,
    train_values: Sequence[float] | None,
    val_values: Sequence[float],
) -> None:
    record = {
        "step": step,
        "train_bce_per_key": (
            _loss_dict(train_values) if train_values is not None else None
        ),
        "val_bce_per_key": _loss_dict(val_values),
    }
    log_file.write(json.dumps(record, sort_keys=True) + "\n")
    log_file.flush()


def _git_describe() -> str:
    declared_commit = os.environ.get("MADELEINE_SOURCE_COMMIT")
    if declared_commit is not None:
        if len(declared_commit) != 40 or any(
            character not in "0123456789abcdef"
            for character in declared_commit
        ):
            raise ValueError(
                "MADELEINE_SOURCE_COMMIT must be a full lowercase Git SHA"
            )
        return f"{declared_commit}-declared"
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=10, check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _shard_hashes(data_dir: Path, session_ids: Sequence[str]) -> dict[str, str]:
    """sha256 per shard, cached by (size, mtime) in the data dir."""
    cache_path = data_dir / "shard_hashes.json"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            cache = {}
    hashes: dict[str, str] = {}
    for sid in session_ids:
        path = data_dir / f"{sid}.npz"
        stat = path.stat()
        entry = cache.get(sid)
        if entry and entry["size"] == stat.st_size and entry["mtime"] == stat.st_mtime:
            hashes[sid] = entry["sha256"]
            continue
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        hashes[sid] = digest.hexdigest()
        cache[sid] = {"size": stat.st_size, "mtime": stat.st_mtime, "sha256": hashes[sid]}
    try:
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    except OSError:
        pass
    return hashes


def _run_metadata(
    *, config: dict, device: torch.device, seed: int, data_dir: Path,
    train_ids: Sequence[str], val_ids: Sequence[str],
    initialized_from: Path | None = None,
    positive_weight: Sequence[float] | None = None,
) -> dict:
    meta: dict[str, Any] = {
        "git": _git_describe(),
        "argv": sys.argv,
        "seed": seed,
        "config": config,
        "device": str(device),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "num_workers": 0,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "shard_sha256": _shard_hashes(Path(data_dir), [*train_ids, *val_ids]),
        "split": {"train": list(train_ids), "val": list(val_ids)},
        "initialized_from": str(initialized_from) if initialized_from else None,
        "positive_weight": (
            _loss_dict(positive_weight) if positive_weight is not None else None
        ),
    }
    if device.type == "cuda":
        meta.update({
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        })
    return meta


def run_training(
    *,
    data_dir: str | Path,
    train_sessions: str | Path,
    val_sessions: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    max_steps: int | None = None,
    device_name: str | None = None,
    seed_override: int | None = None,
    init_checkpoint: str | Path | None = None,
) -> Path:
    train_ids = read_session_ids(train_sessions)
    val_ids = read_session_ids(val_sessions)
    validate_splits(train_ids, val_ids)

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    if seed_override is not None:
        config["seed"] = int(seed_override)

    window = int(config.get("window", 2))
    window_mode = str(config.get("window_mode", "centered"))
    input_config = str(config.get("input_config", "pixels"))
    history_len = int(config.get("history_len", 8))
    history_gap = int(config.get("history_gap", 0))
    active_targets_only = bool(config.get("active_targets_only", True))
    transition_weight = float(config.get("transition_weight", 1.0))
    precomputed_features = bool(config.get("precomputed_features", False))
    frame_stride = int(config.get("frame_stride", 1))
    batch_size = int(config.get("batch_size", 32))
    eval_batch_size = int(config.get("eval_batch_size", batch_size))
    eval_interval = int(config.get("eval_interval", 50))
    learning_rate = float(config.get("learning_rate", 1e-4))
    encoder_learning_rate = float(
        config.get("encoder_learning_rate", learning_rate)
    )
    optimizer_name = str(config.get("optimizer", "adam")).lower()
    weight_decay = float(config.get("weight_decay", 0.0))
    linear_lr_decay = bool(config.get("linear_lr_decay", False))
    steps = int(max_steps if max_steps is not None else config.get("max_steps", 300))
    seed = int(config.get("seed", 0))
    if batch_size < 1 or eval_batch_size < 1:
        raise ValueError("batch sizes must be at least 1")
    if eval_interval < 1 or steps < 0:
        raise ValueError("eval_interval must be positive and max_steps non-negative")
    if learning_rate <= 0 or encoder_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if optimizer_name not in ("adam", "adamw"):
        raise ValueError("optimizer must be 'adam' or 'adamw'")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    if device_name is None:
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but is not available")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    device = torch.device(device_name)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Strict determinism on CUDA (report-grade runs); warn-only elsewhere —
    # MPS lacks deterministic implementations for some ops, and local MPS
    # runs are never headline numbers.
    torch.use_deterministic_algorithms(True, warn_only=(device_name != "cuda"))
    if device_name == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Only the IDs explicitly named above are ever resolved or loaded.
    train_arrays = [
        load_session(
            data_dir, session_id,
            precomputed_features=precomputed_features,
        )
        for session_id in train_ids
    ]
    val_arrays = [
        load_session(
            data_dir, session_id,
            precomputed_features=precomputed_features,
        )
        for session_id in val_ids
    ]
    # Segment path (brief v3.2) whenever pixels are involved: unique frames
    # encode once per step instead of once per window. History-only configs
    # keep the windowed path — their per-window work is a flatten+linear.
    segment_windows = int(config.get("segment_windows", 48))
    use_segments = input_config in ("pixels", "pixels_plus_history")
    dataset_kwargs = dict(
        window=window,
        window_mode=window_mode,
        input_config=input_config,
        history_len=history_len,
        history_gap=history_gap,
        active_targets_only=active_targets_only,
        transition_weight=transition_weight,
        precomputed_features=precomputed_features,
        frame_stride=frame_stride,
    )
    if use_segments:
        train_dataset: Dataset = SegmentSessionDataset(
            train_arrays, segment_windows=segment_windows, **dataset_kwargs
        )
        val_dataset: Dataset = SegmentSessionDataset(
            val_arrays, segment_windows=segment_windows, **dataset_kwargs
        )
        # batch_size counts windows per optimizer step, approximately
        # preserved under segmentation.
        batch_size = max(1, round(batch_size / segment_windows))
        eval_batch_size = max(1, round(eval_batch_size / segment_windows))
    else:
        train_dataset = WindowedSessionDataset(train_arrays, **dataset_kwargs)
        val_dataset = WindowedSessionDataset(val_arrays, **dataset_kwargs)

    source_batch_sampler = build_source_batch_sampler(
        train_dataset,
        train_ids,
        config,
        steps=steps,
        seed=seed,
        expected_batch_items=batch_size,
    )
    if source_batch_sampler is None:
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=source_batch_sampler,
            num_workers=0,
        )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = BadelineIDM(config).to(device)
    initialized_from = Path(init_checkpoint) if init_checkpoint else None
    if initialized_from is not None:
        state = torch.load(initialized_from, map_location="cpu", weights_only=True)
        weights = state.get("model_state_dict", state.get("model", state))
        model.load_state_dict(weights)

    encoder_parameters: list[nn.Parameter] = []
    if model.frame_encoder is not None:
        encoder_parameters = [
            parameter
            for parameter in model.frame_encoder.features.parameters()
            if parameter.requires_grad
        ]
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    other_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_parameter_ids
    ]
    parameter_groups: list[dict[str, Any]] = [
        {"params": other_parameters, "lr": learning_rate}
    ]
    if encoder_parameters:
        parameter_groups.append({
            "params": encoder_parameters,
            "lr": encoder_learning_rate,
        })
    optimizer_class = (
        torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    )
    optimizer = optimizer_class(parameter_groups, weight_decay=weight_decay)
    scheduler = None
    if linear_lr_decay and steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda completed: max(0.0, 1.0 - completed / steps),
        )
    positive_weight = resolve_positive_weight(config, train_arrays)
    positive_weight_tensor: torch.Tensor | None = None
    if positive_weight is not None:
        positive_weight_tensor = torch.tensor(
            positive_weight, dtype=torch.float32, device=device
        )

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_meta = _run_metadata(
        config=config, device=device, seed=seed, data_dir=Path(data_dir),
        train_ids=train_ids, val_ids=val_ids,
        initialized_from=initialized_from,
        positive_weight=positive_weight,
    )
    if source_batch_sampler is not None:
        run_meta["source_sampling"] = source_batch_sampler.receipt(
            require_complete=False
        )
    (output / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    log_path = output / "log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_file:
        step_zero_train = (
            evaluate_per_key(model, train_eval_loader, device)
            if bool(config.get("initial_train_eval", True))
            else None
        )
        step_zero_val = evaluate_per_key(model, val_loader, device)
        _write_log(
            log_file,
            step=0,
            train_values=step_zero_train,
            val_values=step_zero_val,
        )

        running_loss = torch.zeros(len(KEY_ORDER), dtype=torch.float64)
        running_examples = 0
        # Small-data runs memorize: train loss collapses while val loss turns
        # upward. Evaluating the FINAL weights would then measure memorization,
        # not generalization, so keep the best-val snapshot and report that.
        best_val = float(sum(step_zero_val) / len(step_zero_val))
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_step = 0
        batches = _cycle(train_loader)
        for step in range(1, steps + 1):
            batch = next(batches)
            inputs, targets, loss_weight = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = (
                model.forward_segment(inputs) if targets.ndim == 3
                else model(inputs)
            )
            per_element = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            if positive_weight_tensor is None:
                objective = per_element
            else:
                objective = per_element * (
                    1.0 + targets * (positive_weight_tensor - 1.0)
                )
            objective = objective * loss_weight
            objective.mean().backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            flat_loss = per_element.detach().cpu().to(torch.float64).reshape(
                -1, len(KEY_ORDER)
            )
            running_loss += flat_loss.sum(dim=0)
            running_examples += flat_loss.shape[0]

            if step % eval_interval == 0 or step == steps:
                train_values = (running_loss / running_examples).tolist()
                val_values = evaluate_per_key(model, val_loader, device)
                _write_log(
                    log_file,
                    step=step,
                    train_values=train_values,
                    val_values=val_values,
                )
                mean_val = float(sum(val_values) / len(val_values))
                if mean_val < best_val:
                    best_val = mean_val
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    best_step = step
                running_loss.zero_()
                running_examples = 0

    source_sampling_receipt: dict[str, Any] | None = None
    if source_batch_sampler is not None:
        source_sampling_receipt = source_batch_sampler.receipt(
            require_complete=True
        )
        (output / "source_sampling_receipt.json").write_text(
            json.dumps(source_sampling_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run_meta["source_sampling"] = source_sampling_receipt
        (output / "run_meta.json").write_text(
            json.dumps(run_meta, indent=2), encoding="utf-8"
        )

    checkpoint = {
        "config": config,
        "key_order": list(KEY_ORDER),
        "model_state_dict": best_state,          # best-val, not final
        "final_state_dict": model.state_dict(),  # kept for the memorization story
        "steps": steps,
        "best_val_step": best_step,
        "best_val_mean_bce": best_val,
        "initialized_from": str(initialized_from) if initialized_from else None,
        "positive_weight": positive_weight,
        "source_sampling_receipt": source_sampling_receipt,
    }
    model_path = output / "model.pt"
    torch.save(checkpoint, model_path)
    return model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--train-sessions", required=True, type=Path)
    parser.add_argument("--val-sessions", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    parser.add_argument("--seed", type=int, help="overrides config seed")
    parser.add_argument(
        "--init", type=Path,
        help="initialize from a previous model.pt (for clean fine-tuning)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_training(
        data_dir=args.data,
        train_sessions=args.train_sessions,
        val_sessions=args.val_sessions,
        config_path=args.config,
        out_dir=args.out,
        max_steps=args.max_steps,
        device_name=args.device,
        seed_override=args.seed,
        init_checkpoint=args.init,
    )


if __name__ == "__main__":
    main()
