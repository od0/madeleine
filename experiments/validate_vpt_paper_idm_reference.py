#!/usr/bin/env python3
"""Validate the Celeste paper-IDM graph against the official tensor receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch

from badeline.vpt_paper_idm import (
    VPTPaperIDM,
    VPTPaperIDMConfig,
    named_parameter_shapes,
    parameter_inventory,
)


AUDIT_SCHEMA = "madeleine.vpt-paper-idm-reference-graph-audit.v1"


def validate(receipt_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "madeleine.vpt-paper-idm-official-artifacts.v1":
        raise ValueError("unsupported official artifact receipt")
    tensors = receipt["weights_artifact"]["inventory"]["tensors"]
    official_shapes = {
        item["name"]: item["shape"] for item in tensors if item["name"].startswith("net.")
    }
    original_head = [item for item in tensors if item["name"].startswith("pi_head.")]
    if len(official_shapes) + len(original_head) != len(tensors):
        raise ValueError("official archive contains tensors outside net and pi_head")

    with torch.device("meta"):
        model = VPTPaperIDM(VPTPaperIDMConfig())
    observed_shapes = {
        name: shape
        for name, shape in named_parameter_shapes(model).items()
        if name.startswith("net.")
    }
    missing = sorted(set(official_shapes) - set(observed_shapes))
    extra = sorted(set(observed_shapes) - set(official_shapes))
    shape_mismatches = [
        {
            "name": name,
            "official": official_shapes[name],
            "reconstruction": observed_shapes[name],
        }
        for name in sorted(set(official_shapes) & set(observed_shapes))
        if official_shapes[name] != observed_shapes[name]
    ]
    if missing or extra or shape_mismatches:
        raise ValueError(
            f"released body mismatch: missing={missing}, extra={extra}, shapes={shape_mismatches}"
        )

    official_total = sum(int(item["elements"]) for item in tensors)
    official_body = sum(
        int(item["elements"]) for item in tensors if item["name"].startswith("net.")
    )
    official_head = sum(int(item["elements"]) for item in original_head)
    model_inventory = parameter_inventory(model)
    if official_total != 482_330_046:
        raise ValueError(f"unexpected released parameter count: {official_total}")
    if official_body != 482_076_032:
        raise ValueError(f"unexpected released body count: {official_body}")
    if model_inventory["components"].get("net") != official_body:
        raise ValueError("reconstructed body total differs from official body")
    if model_inventory["total"] != 482_133_390:
        raise ValueError("unexpected Celeste reconstruction parameter count")

    net_args = receipt["model_artifact"]["metadata"]["network_args"]
    expected_args = {
        "attention_heads": 32,
        "attention_mask_style": "none",
        "attention_memory_size": 128,
        "hidsize": 4096,
        "impala_width": 16,
        "n_recurrence_layers": 2,
        "pointwise_ratio": 4,
        "pointwise_use_activation": False,
        "recurrence_is_residual": True,
        "recurrence_type": "transformer",
        "timesteps": 128,
    }
    for field, expected in expected_args.items():
        if net_args.get(field) != expected:
            raise ValueError(f"released config mismatch for {field}: {net_args.get(field)!r}")
    if receipt["model_artifact"]["metadata"]["pi_head_options"] != {"temperature": 4}:
        raise ValueError("released action temperature is not four")
    if not receipt["upstream_source"].get("demo_warning_verified"):
        raise ValueError("public-demo non-original-code warning was not verified")

    return {
        "schema_version": AUDIT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_receipt": {
            "path": receipt_path.as_posix(),
            "model_sha256": receipt["model_artifact"]["sha256"],
            "weights_sha256": receipt["weights_artifact"]["sha256"],
            "source_commit": receipt["upstream_source"]["commit"],
        },
        "result": "pass",
        "body_name_shape_match": {
            "matched_tensors": len(official_shapes),
            "missing": missing,
            "extra": extra,
            "shape_mismatches": shape_mismatches,
        },
        "parameter_counts": {
            "released_minecraft_total": official_total,
            "released_minecraft_head": official_head,
            "released_body": official_body,
            "celeste_seven_binary_head": model_inventory["components"]["pi_head"],
            "celeste_reconstruction_total": model_inventory["total"],
        },
        "paper_reconciliation": {
            "appendix_literal_widths": [64, 128, 128],
            "appendix_literal_flattened_width": 32_768,
            "paper_stated_flattened_width": 131_072,
            "released_widths": [256, 512, 512],
            "released_flattened_width": 131_072,
            "appendix_literal_transformer_blocks": 4,
            "released_transformer_blocks": 2,
            "paper_stated_parameters_approximate": 500_000_000,
            "released_minecraft_parameters": official_total,
        },
        "forward_semantics": {
            "attention_mask": "none",
            "attention_logit_scale": "1 / head_dimension",
            "relative_attention_basis_shape": [10, 0],
            "relative_attention_effect": "identically zero but tensors retained",
            "post_transformer_dense": "active lastlayer followed by final_ln",
            "post_transformer_dense_basis": (
                "pinned public MinecraftPolicy base path and released nonempty weights; "
                "the public InverseActionNet demo computes then discards lastlayer and is explicitly "
                "warned by OpenAI not to be the original IDM code"
            ),
            "private_training_code_bit_exactness": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = validate(args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
