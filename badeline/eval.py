"""Evaluation driver: which data reaches which metrics.

Orchestrator-owned — this file routes data, and data routing is where the
traps live. It enforces:
- engine-truth only: the shard directory's build_manifest.json must declare
  engine_hz 60 and a session-unit split; anything else is refused. Foreign
  (mapped) labels never enter here — curves against them are relative
  measurements produced by a different, explicitly-labeled path.
- explicit session lists: no session is evaluated unless named.
- per-key metrics only, via badeline.metrics (which structurally contains no
  aggregate accuracy).

Reports two variants: all frames, and input_active-only (0.1.0-era sessions
carry placeholder-true input_active; their variants coincide and the report
says so).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from badeline.metrics import summarize
from badeline.model import BadelineIDM
from badeline.train import (
    contiguous_runs,
    history_block,
    load_session,
    read_session_ids,
    target_offset,
)


def _ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ndarray_to_list(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def evaluate(
    model: BadelineIDM,
    model_config: dict,
    data_dir: Path,
    session_ids: list[str],
    device: str,
    batch_size: int = 64,
    preds_out: Path | None = None,
    fixed_transition_thresholds: dict[str, float] | None = None,
) -> dict:
    model.eval().to(device)
    config = model_config
    offset = target_offset(
        int(config.get("window", 2)), config.get("window_mode", "centered")
    )
    all_true, all_prob, all_active = [], [], []
    stream_lengths: list[int] = []
    stream_ids: list[str] = []
    window = int(config.get("window", 2))
    frame_stride = int(config.get("frame_stride", 1))
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    frame_span = (window - 1) * frame_stride + 1
    input_config = config.get("input_config", "pixels")
    history_len = int(config.get("history_len", 8))
    history_gap = int(config.get("history_gap", 0))
    uses_pixels = input_config in ("pixels", "pixels_plus_history")
    uses_history = input_config in ("history", "pixels_plus_history")
    precomputed_features = bool(config.get("precomputed_features", False))
    # Segment sweep (brief v3.2): encode each unique frame once and assemble
    # windows by index; exact coverage of every window in session order.
    segment_span = 512
    for sid in session_ids:
        arrays = load_session(
            data_dir, sid, precomputed_features=precomputed_features
        )
        assert arrays.engine_frame_idx is not None
        assert arrays.input_active is not None
        for run_index, (run_start, run_end) in enumerate(
            contiguous_runs(arrays.engine_frame_idx)
        ):
            n_windows = run_end - run_start - frame_span + 1
            if n_windows < 1:
                continue
            probs_chunks = []
            with torch.no_grad():
                for relative_start in range(0, n_windows, segment_span):
                    count = min(segment_span, n_windows - relative_start)
                    start = run_start + relative_start
                    inputs: dict[str, torch.Tensor] = {}
                    if uses_pixels:
                        block = arrays.frames[
                            start : start + count + frame_span - 1
                        ]
                        if precomputed_features:
                            inputs["features"] = (
                                torch.from_numpy(block.copy())
                                .to(dtype=torch.float32).unsqueeze(0).to(device)
                            )
                        else:
                            inputs["frames"] = (
                                torch.from_numpy(block.copy())
                                .permute(0, 3, 1, 2).to(dtype=torch.float32)
                                .div_(255.0).unsqueeze(0).to(device)
                            )
                    if uses_history:
                        target_indices = [
                            start + s + offset * frame_stride
                            for s in range(count)
                        ]
                        inputs["history"] = torch.from_numpy(history_block(
                            arrays.keys, target_indices, history_len, history_gap,
                            floor=run_start,
                        )).unsqueeze(0).to(device)
                    logits = model.forward_segment(inputs)
                    probs_chunks.append(
                        torch.sigmoid(logits)[0].to(torch.float32).cpu().numpy()
                    )
            probs = np.concatenate(probs_chunks)
            target_start = run_start + offset * frame_stride
            keys = arrays.keys[target_start : target_start + len(probs)].astype(bool)
            active = arrays.input_active[
                target_start : target_start + len(probs)
            ].astype(bool)
            all_true.append(keys)
            all_prob.append(probs)
            all_active.append(active)
            stream_lengths.append(len(probs))
            stream_ids.append(f"{sid}__stream{run_index:03d}")

    if not all_true:
        raise ValueError("no contiguous evaluation window in requested sessions")

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)
    active = np.concatenate(all_active)
    # Per-contiguous-stream lengths: events never cross a capture gap or session.
    lengths = stream_lengths

    if preds_out is not None:
        # Sidecar so any future metric change is a local re-score, not a
        # GPU re-inference. float32 probs: ~1 MB per 35k-frame session.
        preds_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            preds_out,
            y_true=y_true.astype(np.uint8),
            y_prob=y_prob.astype(np.float32),
            input_active=active.astype(np.uint8),
            session_lengths=np.asarray(lengths, dtype=np.int64),
            session_ids=np.asarray(stream_ids),
        )

    report = {
        "all_frames": {
            "n": int(len(y_true)),
            "metrics": summarize(
                y_true, y_prob, boundaries=lengths,
                fixed_transition_thresholds=fixed_transition_thresholds,
            ),
        },
        "input_active_only": {
            "n": int(active.sum()),
            "metrics": summarize(
                y_true, y_prob, boundaries=lengths, active=active,
                fixed_transition_thresholds=fixed_transition_thresholds,
            ),
        },
        "input_active_is_placeholder": bool(active.all()),
    }
    return _ndarray_to_list(report)


def select_checkpoint_state(
    checkpoint: dict[str, object], weights: str
) -> object:
    """Select the declared training endpoint without silently substituting one."""

    if weights == "final":
        key = "final_state_dict"
    elif weights == "selected":
        key = "model_state_dict" if "model_state_dict" in checkpoint else "model"
    else:
        raise ValueError(f"unsupported checkpoint weights: {weights}")
    if key not in checkpoint:
        raise KeyError(f"checkpoint does not contain {key}")
    return checkpoint[key]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="training output dir (model.pt + config)")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--sessions", type=Path, required=True, help="explicit session-id list file")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--weights",
        choices=("selected", "final"),
        default="selected",
        help="evaluate the selected checkpoint or the fixed final endpoint",
    )
    ap.add_argument(
        "--transition-thresholds-from", type=Path,
        help=(
            "development eval JSON whose input-active oracle thresholds are "
            "applied as fixed thresholds here"
        ),
    )
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    build_manifest = json.loads((args.data / "build_manifest.json").read_text())
    if build_manifest["grid"].get("engine_hz") != 60:
        raise SystemExit("eval refuses non-engine-truth shards")
    if build_manifest["split"].get("unit") != "session":
        raise SystemExit("eval refuses non-session-unit builds")

    config = json.loads((args.run / "config.json").read_text())
    model = BadelineIDM(config)
    state = torch.load(args.run / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(select_checkpoint_state(state, args.weights))

    session_ids = read_session_ids(args.sessions)
    fixed_thresholds = None
    if args.transition_thresholds_from is not None:
        development = json.loads(args.transition_thresholds_from.read_text())
        tuned = development["input_active_only"]["metrics"][
            "transition_f1_oracle"
        ]
        fixed_thresholds = {
            key: float(tuned[key]["threshold"]) for key in model.key_order
        }
    preds_out = args.out.with_name(args.out.stem + "_preds.npz")
    report = evaluate(
        model, config, args.data, session_ids, args.device,
        preds_out=preds_out,
        fixed_transition_thresholds=fixed_thresholds,
    )
    report["run"] = str(args.run)
    report["sessions"] = session_ids
    report["weights"] = args.weights
    report["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    if args.transition_thresholds_from is not None:
        report["fixed_transition_threshold_source"] = str(
            args.transition_thresholds_from
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    ap_summary = {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in report["input_active_only"]["metrics"]["per_key_ap"].items()
    }
    print(json.dumps({"per_key_ap(input_active)": ap_summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
