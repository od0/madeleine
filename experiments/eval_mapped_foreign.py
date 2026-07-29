"""Evaluate an IDM against explicitly labeled mapped-foreign sessions.

This is deliberately separate from :mod:`badeline.eval`, whose command-line
boundary accepts engine truth only.  Metrics from this script measure
in-distribution agreement with noisy mapped NitroGen labels and must never be
presented as engine-truth or local-transfer performance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from badeline.eval import evaluate
from badeline.model import BadelineIDM
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
    if not session_ids:
        raise SystemExit("mapped-foreign evaluation requires sessions")
    missing = [sid for sid in session_ids if not (args.data / f"{sid}.npz").is_file()]
    if missing:
        raise SystemExit(f"missing mapped-foreign shards: {missing[:5]}")

    config = json.loads((args.run / "config.json").read_text())
    checkpoint = torch.load(
        args.run / "model.pt", map_location="cpu", weights_only=True
    )
    state_key = (
        "final_state_dict" if args.weights == "final" else "model_state_dict"
    )
    if state_key not in checkpoint:
        raise SystemExit(f"checkpoint does not contain {state_key}")
    model = BadelineIDM(config)
    model.load_state_dict(checkpoint[state_key])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    predictions = args.out.with_name(args.out.stem + "_preds.npz")
    report = evaluate(
        model, config, args.data, session_ids, args.device,
        preds_out=predictions,
    )
    with np.load(predictions, allow_pickle=False) as archive:
        truth = archive["y_true"].astype(bool)
        active = archive["input_active"].astype(bool)
        active_truth = truth[active]
    report.update({
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
    })
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    ap = report["input_active_only"]["metrics"]["per_key_ap"]
    print(json.dumps({
        "frames": report["input_active_only"]["n"],
        "macro_ap": float(np.mean(list(ap.values()))),
        "chance_macro_ap": float(np.mean(
            list(report["chance_ap_by_prevalence"].values())
        )),
        "weights": args.weights,
        "label_kind": report["label_kind"],
    }, indent=2))


if __name__ == "__main__":
    main()
