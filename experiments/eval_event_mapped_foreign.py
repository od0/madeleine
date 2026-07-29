"""Evaluate an event-latch IDM on explicit mapped NitroGen sessions.

This entry point is intentionally separate from engine-truth evaluation.  Its
reports measure in-distribution agreement with noisy mapped labels and must not
be presented as local engine-truth transfer performance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from badeline.eval import select_checkpoint_state
from badeline.eval_event import evaluate_event_latch
from badeline.event_model import EventLatchIDM
from badeline.train import read_session_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--weights", choices=("selected", "final"), default="final"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    session_ids = read_session_ids(args.sessions)
    missing = [
        session_id
        for session_id in session_ids
        if not (args.data / f"{session_id}.npz").is_file()
    ]
    if missing:
        raise SystemExit(f"missing mapped-foreign shards: {missing[:5]}")

    config = json.loads((args.run / "config.json").read_text())
    if not config.get("event_latch"):
        raise SystemExit("run config is not marked as an event-latch model")
    checkpoint = torch.load(
        args.run / "model.pt", map_location="cpu", weights_only=True
    )
    model = EventLatchIDM(config)
    model.load_state_dict(select_checkpoint_state(checkpoint, args.weights))
    if "onset_positive_weight" not in checkpoint or "release_positive_weight" not in checkpoint:
        raise SystemExit("checkpoint lacks event positive-weight provenance")

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
    )
    with np.load(predictions, allow_pickle=False) as archive:
        truth = archive["y_true"].astype(bool)
        active = archive["input_active"].astype(bool)
        active_truth = truth[active]
    report.update(
        {
            "run": str(args.run),
            "sessions": session_ids,
            "weights": args.weights,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "label_kind": "mapped_foreign_nitrogen",
            "label_notice": (
                "Relative agreement with noisy mapped NitroGen labels; not "
                "engine-truth and not a local-transfer result."
            ),
            "chance_ap_by_prevalence": {
                key: float(active_truth[:, index].mean())
                for index, key in enumerate(model.key_order)
            },
            "predictions": str(predictions),
        }
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    state_ap = report["input_active_only"]["metrics"]["per_key_ap"]
    state_decisions = report["input_active_only"]["decision_metrics"]
    latch_decisions = report["input_active_only"]["latch_decision_metrics"]
    print(
        json.dumps(
            {
                "frames": report["input_active_only"]["n"],
                "state_macro_ap": float(np.nanmean(list(state_ap.values()))),
                "state_micro_accuracy": state_decisions[
                    "key_state_micro_accuracy"
                ],
                "latch_micro_accuracy": latch_decisions[
                    "key_state_micro_accuracy"
                ],
                "latch_joint_accuracy": latch_decisions[
                    "joint_exact_match_accuracy"
                ],
                "weights": args.weights,
                "label_kind": report["label_kind"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
