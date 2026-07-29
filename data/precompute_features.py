"""Precompute the frozen visual backbone used by Badeline.

The takeover transfer set is large enough that storing masked 128px RGB
frames would dominate both disk and training time.  This builder runs the
*same* frozen ImageNet ResNet-18 backbone as ``FrozenImageNetFrameEncoder``
once and stores its 512-dimensional global-average-pooled output as float16.
The trainable projection, feature deltas, temporal model, and heads remain in
Badeline and are therefore still learned per experiment.

Two explicit entry points are provided:

``foreign``
    Decode a mapped NitroGen video, mask its controller overlay before
    resizing, and emit the same contiguous 10-minute parts as the pixel
    shard builder.  Native CFR sources retain their decoded frame order;
    variable-rate sources are sampled by timestamp onto the 60-Hz label grid.

``shards``
    Convert already-audited local RGB NPZ shards.  Keys, engine indices,
    activity flags, and session IDs are copied exactly.

Output NPZs are deliberately uncompressed.  Float16 backbone features are
about 1 KiB/frame, so compression saves little but makes repeated training
loads much more CPU-intensive.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from data.build_dataset import (
    FOREIGN_GRID_HZ,
    MAX_PART_FRAMES,
    MIN_RUN_FRAMES,
    _foreign_runs,
    _run_keys,
)
from data.schema import KEY_ORDER
from nitrogen.mask import video_mask_rect

BACKBONE_FEATURE_DIM = 512
FRAME_SIZE = 128
MAX_SEQUENTIAL_GAP = 6_000  # 100 s at 60 Hz; avoids expensive random seeks
MAX_RESAMPLED_TAIL_REPEAT = 3
PIPE_READ_BYTES = 8 * 1024 * 1024


def _nominal_timeline_frames(
    decoded_frames: int, average_fps: float
) -> tuple[bool, int]:
    """Return whether timestamp resampling is needed and the 60-Hz extent."""

    if decoded_frames < 1 or average_fps <= 0:
        raise ValueError("video frame count and average fps must be positive")
    resample = abs(average_fps - FOREIGN_GRID_HZ) > 0.1
    if not resample:
        return False, decoded_frames
    duration_s = decoded_frames / average_fps
    return True, int(round(duration_s * FOREIGN_GRID_HZ))


def _decode_resampled_part(
    video_path: Path,
    *,
    start_frame: int,
    end_frame: int,
    mask_xyxy: tuple[int, int, int, int],
) -> tuple[np.ndarray, int]:
    """Decode one nominal-time part at 60 Hz, repeating only a short tail.

    FFmpeg's ``fps`` filter repeats or drops decoded images according to their
    timestamps.  This is the intended imputation for sources whose nominal
    stream rate is 60 Hz but whose actual decoded cadence is variable.
    """

    frame_count = end_frame - start_frame
    if frame_count < 1:
        raise ValueError("resampled part must contain at least one frame")
    x0, y0, x1, y1 = mask_xyxy
    filters = (
        "setpts=PTS-STARTPTS,"
        f"fps=fps={FOREIGN_GRID_HZ:g}:round=near,"
        f"scale={FRAME_SIZE}:{FRAME_SIZE}:flags=area,format=rgb24,"
        f"drawbox=x={x0}:y={y0}:w={x1-x0}:h={y1-y0}:"
        "color=black:t=fill"
    )
    command = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-ss", f"{start_frame / FOREIGN_GRID_HZ:.9f}",
        "-i", str(video_path),
        "-t", f"{frame_count / FOREIGN_GRID_HZ:.9f}",
        "-an", "-vf", filters,
        "-frames:v", str(frame_count),
        "-vsync", "0", "-pix_fmt", "rgb24",
        "-f", "rawvideo", "pipe:1",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stderr_tail: deque[bytes] = deque(maxlen=64)

    def drain_stderr() -> None:
        while chunk := process.stderr.read(4096):
            stderr_tail.append(chunk)

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()
    frames = np.empty(
        (frame_count, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8
    )
    target = memoryview(frames).cast("B")
    offset = 0
    while offset < len(target):
        chunk = process.stdout.read(min(PIPE_READ_BYTES, len(target) - offset))
        if not chunk:
            break
        target[offset : offset + len(chunk)] = chunk
        offset += len(chunk)
    extra_bytes = 0
    while chunk := process.stdout.read(PIPE_READ_BYTES):
        extra_bytes += len(chunk)
    return_code = process.wait()
    stderr_thread.join()
    stderr = b"".join(stderr_tail).decode("utf-8", errors="replace")
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg exited {return_code}: {stderr[-500:].strip()}"
        )
    if extra_bytes:
        raise RuntimeError(
            f"ffmpeg emitted {extra_bytes} extra raw-video bytes"
        )
    frame_bytes = FRAME_SIZE * FRAME_SIZE * 3
    if offset % frame_bytes:
        raise RuntimeError(f"partial RGB frame: {offset % frame_bytes} bytes")
    decoded = offset // frame_bytes
    missing = frame_count - decoded
    if decoded == 0 or missing > MAX_RESAMPLED_TAIL_REPEAT:
        raise RuntimeError(
            f"timestamp resample produced {decoded}/{frame_count} frames"
        )
    if missing:
        frames[decoded:] = frames[decoded - 1]
    return frames, missing


class FrozenResNet18Features(nn.Module):
    """The frozen portion of ``badeline.model.FrozenImageNetFrameEncoder``."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.features.requires_grad_(False)
        self.features.eval()
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def train(self, mode: bool = True) -> "FrozenResNet18Features":
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        normalized = (frames - self.image_mean) / self.image_std
        return self.features(normalized).mean(dim=(-2, -1))


class FeatureEncoder:
    """Batched uint8-RGB to CPU float16 feature conversion."""

    def __init__(self, device: str, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.batch_size = batch_size
        self.model = FrozenResNet18Features().eval().to(self.device)

    @torch.inference_mode()
    def encode(self, frames_rgb: np.ndarray) -> np.ndarray:
        if frames_rgb.dtype != np.uint8 or frames_rgb.ndim != 4:
            raise ValueError("frames must be uint8 [N,H,W,3]")
        if frames_rgb.shape[-1] != 3:
            raise ValueError("frames must have three RGB channels")
        output = np.empty(
            (len(frames_rgb), BACKBONE_FEATURE_DIM), dtype=np.float16
        )
        for start in range(0, len(frames_rgb), self.batch_size):
            stop = min(start + self.batch_size, len(frames_rgb))
            batch = (
                torch.from_numpy(frames_rgb[start:stop].copy())
                .permute(0, 3, 1, 2)
                .to(device=self.device, dtype=torch.float32)
                .div_(255.0)
            )
            features = self.model(batch)
            output[start:stop] = features.to(torch.float16).cpu().numpy()
        return output


def _write_feature_shard(
    path: Path,
    *,
    features: np.ndarray,
    keys: np.ndarray,
    engine_frame_idx: np.ndarray,
    input_active: np.ndarray,
    session_id: str,
) -> None:
    """Write one complete shard atomically, leaving no plausible partial NPZ."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        features=np.asarray(features, dtype=np.float16),
        keys=np.asarray(keys, dtype=np.uint8),
        engine_frame_idx=np.asarray(engine_frame_idx, dtype=np.int64),
        input_active=np.asarray(input_active, dtype=np.uint8),
        session_id=np.asarray(session_id),
    )
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _valid_feature_shard(
    path: Path,
    session_id: str,
    keys: np.ndarray,
    engine_frame_idx: np.ndarray,
    input_active: np.ndarray,
) -> bool:
    """Return true only for a complete, schema-valid resumable output."""

    frame_count = len(keys)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                archive["features"].shape == (frame_count, BACKBONE_FEATURE_DIM)
                and archive["features"].dtype == np.float16
                and np.array_equal(archive["keys"], keys)
                and np.array_equal(archive["engine_frame_idx"], engine_frame_idx)
                and np.array_equal(archive["input_active"], input_active)
                and str(archive["session_id"].reshape(()).item()) == session_id
            )
    except (OSError, ValueError, KeyError):
        return False


def convert_shard(
    source: Path, out_dir: Path, encoder: FeatureEncoder
) -> dict:
    """Convert one audited RGB frame shard without changing its supervision."""

    with np.load(source, allow_pickle=False) as archive:
        required = {
            "frames", "keys", "engine_frame_idx", "input_active", "session_id"
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{source}: missing arrays {sorted(missing)}")
        frames = archive["frames"]
        keys = archive["keys"]
        engine_frame_idx = archive["engine_frame_idx"]
        input_active = archive["input_active"]
        session_id = str(archive["session_id"].reshape(()).item())

    if frames.dtype != np.uint8 or frames.ndim != 4:
        raise ValueError(f"{source}: frames must be uint8 [N,H,W,3]")
    if keys.shape != (len(frames), len(KEY_ORDER)):
        raise ValueError(f"{source}: keys do not align with frames")
    output = out_dir / f"{session_id}.npz"
    if _valid_feature_shard(
        output, session_id, keys, engine_frame_idx, input_active
    ):
        return {
            "session_id": session_id,
            "frames": int(len(frames)),
            "source": str(source),
            "npz": output.name,
            "resumed": True,
        }
    _write_feature_shard(
        output,
        features=encoder.encode(frames),
        keys=keys,
        engine_frame_idx=engine_frame_idx,
        input_active=input_active,
        session_id=session_id,
    )
    return {
        "session_id": session_id,
        "frames": int(len(frames)),
        "source": str(source),
        "npz": output.name,
        "resumed": False,
    }


def _foreign_parts(
    *,
    chunk_frames: Path,
    mapped_video_dir: Path,
    video_id: str,
    video_frames: int,
) -> tuple[list[tuple[int, int, np.ndarray]], int, int, list[list[dict]]]:
    """Plan parts exactly as ``data.build_dataset.build_foreign_video``."""

    runs = _foreign_runs(chunk_frames, mapped_video_dir, video_id)
    parts: list[tuple[int, int, np.ndarray]] = []
    skipped_short = 0
    truncated = 0
    for run in runs:
        keys = _run_keys(run)
        start, end = run[0]["start_frame"], run[-1]["end_frame"]
        if end > video_frames:
            truncated += end - video_frames
            end = video_frames
            keys = keys[: max(0, end - start)]
        for part_start in range(start, end, MAX_PART_FRAMES):
            part_end = min(part_start + MAX_PART_FRAMES, end)
            if part_end - part_start < MIN_RUN_FRAMES:
                skipped_short += part_end - part_start
                continue
            parts.append((
                part_start,
                part_end,
                keys[part_start - start : part_end - start],
            ))
    return parts, skipped_short, truncated, runs


def build_foreign_video(
    *,
    video_path: Path,
    video_id: str,
    mapped_root: Path,
    chunk_frames: Path,
    chunk_index: Path,
    out_dir: Path,
    encoder: FeatureEncoder,
    worker_index: int = 0,
    worker_count: int = 1,
) -> dict:
    """Decode, mask, resize, and featurize one mapped 60 Hz video."""

    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError("worker index must lie in [0, worker count)")
    mapped_video_dir = mapped_root / video_id
    if not mapped_video_dir.is_dir():
        raise SystemExit(f"{video_id}: no mapped labels at {mapped_video_dir}")
    mapping_report = json.loads(
        (mapped_video_dir / "mapping_report.json").read_text()
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"{video_id}: cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    resample_timestamps, timeline_frames = _nominal_timeline_frames(
        n_video, fps
    )
    decoder_mode = (
        "ffmpeg_timestamp_resample_60hz"
        if resample_timestamps
        else "opencv_native_60hz"
    )

    rect = video_mask_rect(chunk_index, video_id, (width, height))
    rx0, ry0, rx1, ry1 = rect
    sx0 = max(0, int(rx0 / width * FRAME_SIZE) - 1)
    sy0 = max(0, int(ry0 / height * FRAME_SIZE) - 1)
    sx1 = min(FRAME_SIZE, int(np.ceil(rx1 / width * FRAME_SIZE)) + 1)
    sy1 = min(FRAME_SIZE, int(np.ceil(ry1 / height * FRAME_SIZE)) + 1)
    parts, skipped_short, truncated, runs = _foreign_parts(
        chunk_frames=chunk_frames,
        mapped_video_dir=mapped_video_dir,
        video_id=video_id,
        video_frames=timeline_frames,
    )

    session_rows = []
    resumed_parts = 0
    imputed_tail_frames = 0
    cursor = 0
    worker_start = len(parts) * worker_index // worker_count
    worker_end = len(parts) * (worker_index + 1) // worker_count
    for part_index, (start, end, keys) in enumerate(parts):
        # Contiguous ranges are intentional. OpenCV's MP4 random seek may
        # decode forward from a distant keyframe; round-robin assignment
        # multiplied that cost by every part. Each worker now pays at most one
        # large initial seek, then decodes nearby retained ranges in order.
        if not worker_start <= part_index < worker_end:
            continue
        session_id = f"{video_id}__r{part_index:03d}"
        output = out_dir / f"{session_id}.npz"
        decode_sidecar = out_dir / f"{session_id}.decode.json"
        engine_frame_idx = np.arange(start, end, dtype=np.int64)
        input_active = np.ones(end - start, dtype=np.uint8)
        valid_output = _valid_feature_shard(
            output, session_id, keys, engine_frame_idx, input_active
        )
        decode_metadata = None
        if decode_sidecar.is_file():
            try:
                candidate = json.loads(decode_sidecar.read_text())
                if (
                    isinstance(candidate, dict)
                    and candidate.get("decoder_mode") == decoder_mode
                    and candidate.get("source_frame_range") == [start, end]
                    and isinstance(candidate.get("imputed_tail_frames"), int)
                ):
                    decode_metadata = candidate
            except (OSError, ValueError, TypeError):
                pass
        # A VFR shard without its decode sidecar may have been written by the
        # old frame-index path.  Rebuild it rather than claim timestamp
        # resampling that did not occur.
        if valid_output and (not resample_timestamps or decode_metadata):
            part_imputed = 0
            if decode_metadata is not None:
                part_imputed = int(decode_metadata["imputed_tail_frames"])
            imputed_tail_frames += part_imputed
            resumed_parts += 1
            session_rows.append({
                "session_id": session_id,
                "frames": int(end - start),
                "source_frame_range": [int(start), int(end)],
                "npz": output.name,
                "decoder_mode": decoder_mode,
                "imputed_tail_frames": part_imputed,
            })
            # The decoder was not advanced. Force an explicit seek before the
            # next part rather than pretending its physical cursor moved.
            cursor = -1
            continue
        if resample_timestamps:
            frames, part_imputed = _decode_resampled_part(
                video_path,
                start_frame=start,
                end_frame=end,
                mask_xyxy=(sx0, sy0, sx1, sy1),
            )
            # The OpenCV cursor has no relationship to the nominal-time path.
            cursor = -1
        else:
            part_imputed = 0
            if cursor >= 0 and 0 < start - cursor <= MAX_SEQUENTIAL_GAP:
                while cursor < start:
                    if not cap.grab():
                        raise SystemExit(
                            f"{video_id}: ended at {cursor} before part {start}"
                        )
                    cursor += 1
            elif cursor != start:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
                    raise SystemExit(
                        f"{video_id}: failed to seek to frame {start}"
                    )
                reported = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                if reported != start:
                    raise SystemExit(
                        f"{video_id}: seek landed at {reported}, expected {start}"
                    )
                cursor = start
            frames = np.empty(
                (end - start, FRAME_SIZE, FRAME_SIZE, 3), np.uint8
            )
            for index in range(end - start):
                ok, frame = cap.read()
                if not ok:
                    raise SystemExit(
                        f"{video_id}: decode failed at frame {cursor}"
                    )
                cursor += 1
                frame[ry0:ry1, rx0:rx1] = 0
                small = cv2.resize(
                    frame, (FRAME_SIZE, FRAME_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
                small[sy0:sy1, sx0:sx1] = 0
                frames[index] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if int(frames[:, sy0:sy1, sx0:sx1].max(initial=0)) != 0:
            raise AssertionError(f"{video_id}: controller mask is not black")

        imputed_tail_frames += part_imputed
        _write_feature_shard(
            output,
            features=encoder.encode(frames),
            keys=keys,
            engine_frame_idx=engine_frame_idx,
            input_active=input_active,
            session_id=session_id,
        )
        # Write provenance only after the atomic feature replace.  A crash in
        # encoding must not leave a sidecar that could bless an older VFR shard
        # produced by the pre-resampling frame-index path.
        _write_json_atomic(decode_sidecar, {
            "decoder_mode": decoder_mode,
            "imputed_tail_frames": part_imputed,
            "source_frame_range": [int(start), int(end)],
        })
        session_rows.append({
            "session_id": session_id,
            "frames": int(end - start),
            "source_frame_range": [int(start), int(end)],
            "npz": output.name,
            "decoder_mode": decoder_mode,
            "imputed_tail_frames": part_imputed,
        })
    cap.release()

    return {
        "video_id": video_id,
        "label_kind": "mapped",
        "grid_hz": FOREIGN_GRID_HZ,
        "video": {
            "path": str(video_path),
            "fps": fps,
            "frames": n_video,
            "average_fps": fps,
            "resolution_wh": [width, height],
            "decoded_frames": n_video,
            "nominal_timeline_frames": timeline_frames,
        },
        "decoder_mode": decoder_mode,
        "imputed_tail_frames": int(imputed_tail_frames),
        "mask_rect_xyxy": [int(value) for value in rect],
        "bind_confidence": mapping_report["confidence"],
        "bind_map": mapping_report["bind_map"],
        "end_frame_conventions": sorted({
            row["end_convention"] for run in runs for row in run
        }),
        "runs": len(runs),
        "parts": session_rows,
        "skipped_short_frames": int(skipped_short),
        "tail_truncated_frames": int(truncated),
        "resumed_parts": int(resumed_parts),
        "worker": {"index": int(worker_index), "count": int(worker_count)},
    }


def _common_encoder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    shards = subparsers.add_parser("shards", help="convert local RGB NPZ shards")
    shards.add_argument("--inputs", type=Path, nargs="+", required=True)
    shards.add_argument("--out", type=Path, required=True)
    _common_encoder_args(shards)

    foreign = subparsers.add_parser("foreign", help="build one mapped video")
    foreign.add_argument("--video", type=Path, required=True)
    foreign.add_argument("--video-id", required=True)
    foreign.add_argument("--mapped-root", type=Path, required=True)
    foreign.add_argument("--chunk-frames", type=Path, required=True)
    foreign.add_argument("--chunk-index", type=Path, required=True)
    foreign.add_argument("--out", type=Path, required=True)
    foreign.add_argument("--worker-index", type=int, default=0)
    foreign.add_argument("--worker-count", type=int, default=1)
    _common_encoder_args(foreign)

    args = parser.parse_args()
    encoder = FeatureEncoder(args.device, args.batch_size)
    common = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "format": "resnet18_imagenet_avgpool_float16_v1",
        "backbone_feature_dim": BACKBONE_FEATURE_DIM,
        "frame_size": FRAME_SIZE,
    }
    if args.command == "shards":
        reports = [convert_shard(path, args.out, encoder) for path in args.inputs]
        _write_manifest(
            args.out / "feature_build_manifest.json",
            {**common, "source_kind": "audited_rgb_shards", "sessions": reports},
        )
        # Preserve the engine-truth/split contract used by badeline.eval when
        # every source shard came from one audited build.  Foreign feature
        # directories intentionally never receive this manifest, so the eval
        # driver continues to refuse treating mapped labels as engine truth.
        source_parents = {path.resolve().parent for path in args.inputs}
        if len(source_parents) == 1:
            source_manifest = next(iter(source_parents)) / "build_manifest.json"
            if source_manifest.is_file():
                engine_manifest = json.loads(source_manifest.read_text())
                engine_manifest["visual_representation"] = common["format"]
                engine_manifest["backbone_feature_dim"] = BACKBONE_FEATURE_DIM
                engine_manifest["source_build_manifest"] = str(source_manifest)
                _write_manifest(args.out / "build_manifest.json", engine_manifest)
        print(json.dumps(reports, indent=2))
    else:
        report = build_foreign_video(
            video_path=args.video,
            video_id=args.video_id,
            mapped_root=args.mapped_root,
            chunk_frames=args.chunk_frames,
            chunk_index=args.chunk_index,
            out_dir=args.out,
            encoder=encoder,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
        )
        manifest_name = (
            "feature_build_manifest.json"
            if args.worker_count == 1
            else (
                f"feature_build_manifest.worker_{args.worker_index:02d}"
                f"_of_{args.worker_count:02d}.json"
            )
        )
        _write_manifest(
            args.out / manifest_name,
            {**common, "source_kind": "mapped_foreign_video", "videos": [report]},
        )
        print(json.dumps({
            "video_id": report["video_id"],
            "parts": len(report["parts"]),
            "frames": sum(part["frames"] for part in report["parts"]),
        }, indent=2))


if __name__ == "__main__":
    main()
