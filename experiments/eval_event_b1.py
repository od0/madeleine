"""Score an event-latch checkpoint on frozen B1 engine-truth development data.

B1 cannot affect fitting, checkpoint selection, or threshold selection.  This
entry point accepts only the declared B1 validation session from a validated
60-Hz session-unit build and disables all oracle thresholds in the evaluator.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from badeline.eval import select_checkpoint_state
from badeline.eval_event import evaluate_event_latch
from badeline.event_model import EventLatchIDM
from badeline.train import read_session_ids


B1_SESSION_ID = "rec_20260725_160450_b1"


def validate_b1_manifest(data_dir: Path, session_ids: list[str]) -> dict[str, Any]:
    """Fail closed unless the input is exactly the frozen B1 engine build."""

    manifest_path = data_dir / "build_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("B1 build manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("B1 build manifest must be an object")
    if manifest.get("grid", {}).get("engine_hz") != 60:
        raise ValueError("B1 evaluation requires 60-Hz engine truth")
    split = manifest.get("split", {})
    if split.get("unit") != "session" or split.get("train") != []:
        raise ValueError("B1 evaluation requires a validation-only session split")
    expected = [B1_SESSION_ID]
    if split.get("val") != expected or session_ids != expected:
        raise ValueError("B1 evaluation accepts only the frozen B1 session")
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 1:
        raise ValueError("B1 manifest must describe exactly one session")
    record = sessions[0]
    if record.get("session_id") != B1_SESSION_ID or record.get("frames") != 53762:
        raise ValueError("B1 manifest session identity or frame support changed")
    if record.get("input_active_frames") != 37898:
        raise ValueError("B1 input-active support changed")
    if manifest.get("visual_representation") != "resnet18_imagenet_avgpool_float16_v1":
        raise ValueError("B1 visual representation changed")
    if manifest.get("backbone_feature_dim") != 512:
        raise ValueError("B1 feature dimension changed")
    return manifest


def _contains_oracle_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            "oracle" in str(key).lower() or _contains_oracle_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_oracle_key(item) for item in value)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--weights", choices=("selected", "final"), required=True
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    session_ids = read_session_ids(args.sessions)
    validate_b1_manifest(args.data, session_ids)
    if not (args.data / f"{B1_SESSION_ID}.npz").is_file():
        raise SystemExit("frozen B1 feature shard is missing")

    config = json.loads((args.run / "config.json").read_text())
    if not config.get("event_latch"):
        raise SystemExit("run config is not marked as an event-latch model")
    checkpoint = torch.load(
        args.run / "model.pt", map_location="cpu", weights_only=True
    )
    if "onset_positive_weight" not in checkpoint or "release_positive_weight" not in checkpoint:
        raise SystemExit("checkpoint lacks event positive-weight provenance")
    model = EventLatchIDM(config)
    model.load_state_dict(select_checkpoint_state(checkpoint, args.weights))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    predictions = args.out.with_name(args.out.stem + "_preds.npz")
    report = evaluate_event_latch(
        model,
        config,
        args.data,
        session_ids,
        args.device,
        preds_out=predictions,
        onset_positive_weight=checkpoint["onset_positive_weight"],
        release_positive_weight=checkpoint["release_positive_weight"],
        allow_oracle_thresholds=False,
    )
    report.update(
        {
            "run": str(args.run),
            "sessions": session_ids,
            "weights": args.weights,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "label_kind": "engine_truth_development_b1",
            "label_notice": (
                "Frozen engine-truth development evidence; not an untouched "
                "test result."
            ),
            "b1_policy": {
                "used_for_training": False,
                "used_for_checkpoint_selection": False,
                "used_for_threshold_fitting": False,
                "decode_threshold_source": "fixed_predeclared_policy",
                "event_deweighting_source": "training_checkpoint_positive_weights",
            },
            "predictions": str(predictions),
        }
    )
    if _contains_oracle_key(report):
        raise SystemExit("B1 report unexpectedly contains oracle threshold output")
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    state_ap = report["input_active_only"]["metrics"]["per_key_ap"]
    print(
        json.dumps(
            {
                "frames": report["input_active_only"]["n"],
                "state_macro_ap": float(np.nanmean(list(state_ap.values()))),
                "weights": args.weights,
                "label_kind": report["label_kind"],
                "threshold_tuning": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
