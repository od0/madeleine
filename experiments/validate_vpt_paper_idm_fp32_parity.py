#!/usr/bin/env python3
"""Prove FP32 component parity with OpenAI's pinned public VPT graph.

The official weight pickle is never executed. Selected tensor arrays are read
through the restricted inventory produced by
``inspect_vpt_paper_idm_artifacts.py``. The upstream Python modules are imported
only after every pinned source file matches the tracked SHA-256 receipt.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badeline.vpt_paper_idm import (
    VPTPaperIDMConfig,
    _FanInLayer,
    _ImageProcess,
    _TransformerBlock,
)
from experiments.inspect_vpt_paper_idm_artifacts import (
    materialize_weight_arrays,
    sha256_file,
)


PARITY_SCHEMA = "madeleine.vpt-paper-idm-fp32-public-graph-parity.v1"


def _verify_upstream_source(source_root: Path, receipt: Mapping[str, Any]) -> None:
    upstream = receipt["upstream_source"]
    for record in upstream["files"]:
        path = source_root / record["path"]
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"pinned upstream source size mismatch: {record['path']}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"pinned upstream source hash mismatch: {record['path']}")


def _import_upstream(source_root: Path) -> tuple[type[nn.Module], type[nn.Module], type[nn.Module]]:
    """Import only the audited component modules, with inert gym3 type stubs."""

    gym3 = types.ModuleType("gym3")
    gym3_types = types.ModuleType("gym3.types")
    for name in ("DictType", "Discrete", "Real", "TensorType", "ValType"):
        setattr(gym3_types, name, type(name, (), {}))
    gym3.types = gym3_types  # type: ignore[attr-defined]
    sys.modules.setdefault("gym3", gym3)
    sys.modules.setdefault("gym3.types", gym3_types)
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)
    from lib.impala_cnn import ImpalaCNN  # type: ignore[import-not-found]
    from lib.util import FanInInitReLULayer, ResidualRecurrentBlock  # type: ignore[import-not-found]

    return ImpalaCNN, FanInInitReLULayer, ResidualRecurrentBlock


def _arrays_for_prefix(
    weights: Path,
    receipt: Mapping[str, Any],
    prefix: str,
) -> dict[str, np.ndarray]:
    weights_record = receipt["weights_artifact"]
    names = [
        item["name"]
        for item in weights_record["inventory"]["tensors"]
        if item["name"].startswith(prefix)
    ]
    if not names:
        raise ValueError(f"official inventory has no tensors below {prefix}")
    return dict(
        materialize_weight_arrays(
            weights,
            expected_sha256=weights_record["sha256"],
            inventory=weights_record["inventory"],
            names=names,
        )
    )


def _load_prefix(module: nn.Module, arrays: Mapping[str, np.ndarray], prefix: str) -> None:
    state = {
        name.removeprefix(prefix): torch.from_numpy(array)
        for name, array in arrays.items()
    }
    result = module.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"component load mismatch: {result}")


def compare_outputs(
    label: str,
    reproduction: torch.Tensor,
    upstream: torch.Tensor,
    *,
    atol: float = 2e-5,
    rtol: float = 2e-5,
) -> dict[str, Any]:
    if reproduction.shape != upstream.shape:
        raise ValueError(f"{label} output shape mismatch")
    if not torch.isfinite(reproduction).all() or not torch.isfinite(upstream).all():
        raise ValueError(f"{label} produced nonfinite output")
    reproduction_value = reproduction.detach()
    upstream_value = upstream.detach()
    difference = (reproduction_value - upstream_value).abs()
    maximum_absolute = float(difference.max()) if difference.numel() else 0.0
    denominator = upstream_value.abs().clamp_min(1e-8)
    maximum_relative = float((difference / denominator).max()) if difference.numel() else 0.0
    passed = bool(
        torch.allclose(reproduction_value, upstream_value, atol=atol, rtol=rtol)
    )
    if not passed:
        raise ValueError(
            f"{label} FP32 parity failed: max_abs={maximum_absolute}, "
            f"max_rel={maximum_relative}"
        )
    return {
        "label": label,
        "shape": list(reproduction.shape),
        "atol": atol,
        "rtol": rtol,
        "max_abs": maximum_absolute,
        "max_rel": maximum_relative,
        "exact": bool(torch.equal(reproduction_value, upstream_value)),
        "reproduction_sha256": hashlib.sha256(
            reproduction.detach().contiguous().numpy().tobytes()
        ).hexdigest(),
        "upstream_sha256": hashlib.sha256(
            upstream.detach().contiguous().numpy().tobytes()
        ).hexdigest(),
        "result": "pass",
    }


def _parity_loss(output: torch.Tensor) -> torch.Tensor:
    value = output.float()
    return value.square().mean() + 0.125 * value.mean()


def compare_parameter_gradients(
    label: str,
    reproduction: nn.Module,
    upstream: nn.Module,
    *,
    atol: float = 5e-5,
    rtol: float = 5e-5,
) -> dict[str, Any]:
    reproduction_parameters = dict(reproduction.named_parameters())
    upstream_parameters = dict(upstream.named_parameters())
    if reproduction_parameters.keys() != upstream_parameters.keys():
        raise ValueError(f"{label} parameter-name mismatch during gradient parity")

    reproduction_hash = hashlib.sha256()
    upstream_hash = hashlib.sha256()
    maximum_absolute = 0.0
    maximum_relative = 0.0
    exact = True
    byte_exact = True
    elements = 0
    for name in reproduction_parameters:
        reproduction_gradient = reproduction_parameters[name].grad
        upstream_gradient = upstream_parameters[name].grad
        if reproduction_gradient is None or upstream_gradient is None:
            raise ValueError(f"{label} missing gradient for {name}")
        if reproduction_gradient.shape != upstream_gradient.shape:
            raise ValueError(f"{label} gradient shape mismatch for {name}")
        if not torch.isfinite(reproduction_gradient).all() or not torch.isfinite(
            upstream_gradient
        ).all():
            raise ValueError(f"{label} nonfinite gradient for {name}")
        difference = (reproduction_gradient - upstream_gradient).abs()
        if difference.numel():
            maximum_absolute = max(maximum_absolute, float(difference.max()))
            denominator = upstream_gradient.abs().clamp_min(1e-8)
            maximum_relative = max(
                maximum_relative, float((difference / denominator).max())
            )
        if not torch.allclose(
            reproduction_gradient, upstream_gradient, atol=atol, rtol=rtol
        ):
            raise ValueError(
                f"{label} FP32 gradient parity failed for {name}: "
                f"max_abs={maximum_absolute}, max_rel={maximum_relative}"
            )
        exact = exact and bool(torch.equal(reproduction_gradient, upstream_gradient))
        reproduction_bytes = (
            reproduction_gradient.detach().contiguous().cpu().numpy().tobytes()
        )
        upstream_bytes = upstream_gradient.detach().contiguous().cpu().numpy().tobytes()
        byte_exact = byte_exact and reproduction_bytes == upstream_bytes
        name_bytes = name.encode("utf-8")
        reproduction_hash.update(len(name_bytes).to_bytes(4, "big"))
        reproduction_hash.update(name_bytes)
        reproduction_hash.update(reproduction_bytes)
        upstream_hash.update(len(name_bytes).to_bytes(4, "big"))
        upstream_hash.update(name_bytes)
        upstream_hash.update(upstream_bytes)
        elements += reproduction_gradient.numel()

    return {
        "label": label,
        "matched_tensors": len(reproduction_parameters),
        "matched_elements": elements,
        "atol": atol,
        "rtol": rtol,
        "max_abs": maximum_absolute,
        "max_rel": maximum_relative,
        "exact": exact,
        "byte_exact": byte_exact,
        "byte_difference_explanation": (
            None if byte_exact else "numeric identity with differing signed-zero bits"
        ),
        "reproduction_sha256": reproduction_hash.hexdigest(),
        "upstream_sha256": upstream_hash.hexdigest(),
        "result": "pass",
    }


def validate(weights: Path, receipt_path: Path, source_root: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "madeleine.vpt-paper-idm-official-artifacts.v1":
        raise ValueError("unsupported official-artifact receipt")
    _verify_upstream_source(source_root, receipt)
    ImpalaCNN, FanInInitReLULayer, ResidualRecurrentBlock = _import_upstream(source_root)
    config = VPTPaperIDMConfig(activation_checkpointing=False)
    generator = torch.Generator(device="cpu").manual_seed(20260729)
    components: list[dict[str, Any]] = []

    prefix = "net.conv3d_layer."
    arrays = _arrays_for_prefix(weights, receipt, prefix)
    reproduction_conv = _FanInLayer(
        3,
        128,
        kind="conv3d",
        kernel_size=(5, 1, 1),
        padding=(2, 0, 0),
    ).eval()
    upstream_conv = FanInInitReLULayer(
        3,
        128,
        layer_type="conv3d",
        kernel_size=(5, 1, 1),
        padding=(2, 0, 0),
    ).eval()
    _load_prefix(reproduction_conv, arrays, prefix)
    _load_prefix(upstream_conv, arrays, prefix)
    conv_input = torch.randn(1, 3, 7, 9, 11, generator=generator)
    reproduction_output = reproduction_conv(conv_input)
    upstream_output = upstream_conv(conv_input)
    component = compare_outputs("conv3d", reproduction_output, upstream_output)
    _parity_loss(reproduction_output).backward()
    _parity_loss(upstream_output).backward()
    component["parameter_gradient_parity"] = compare_parameter_gradients(
        "conv3d", reproduction_conv, upstream_conv
    )
    components.append(component)
    del arrays, reproduction_conv, upstream_conv, conv_input, reproduction_output, upstream_output
    gc.collect()

    prefix = "net.img_process."
    arrays = _arrays_for_prefix(weights, receipt, prefix)
    reproduction_image = _ImageProcess(config).eval()

    class UpstreamImageProcess(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cnn = ImpalaCNN(
                inshape=[128, 128, 128],
                chans=[256, 512, 512],
                outsize=256,
                nblock=2,
                init_norm_kwargs={"batch_norm": False, "group_norm_groups": 1},
                dense_init_norm_kwargs={"batch_norm": False, "layer_norm": True},
                first_conv_norm=True,
                post_pool_groups=1,
            )
            self.linear = FanInInitReLULayer(
                256,
                4096,
                layer_type="linear",
                layer_norm=True,
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.linear(self.cnn(value))

    upstream_image = UpstreamImageProcess().eval()
    _load_prefix(reproduction_image, arrays, prefix)
    _load_prefix(upstream_image, arrays, prefix)
    image_input = torch.randn(1, 128, 128, 128, generator=generator)
    reproduction_output = reproduction_image(image_input)
    upstream_output = upstream_image(image_input.permute(0, 2, 3, 1).unsqueeze(1))[:, 0]
    component = compare_outputs(
        "impala_and_frame_projection", reproduction_output, upstream_output
    )
    _parity_loss(reproduction_output).backward()
    _parity_loss(upstream_output).backward()
    component["parameter_gradient_parity"] = compare_parameter_gradients(
        "impala_and_frame_projection", reproduction_image, upstream_image
    )
    components.append(component)
    del arrays, reproduction_image, upstream_image, image_input, reproduction_output, upstream_output
    gc.collect()

    prefix = "net.recurrent_layer.blocks.0."
    arrays = _arrays_for_prefix(weights, receipt, prefix)
    reproduction_block = _TransformerBlock(config, init_scale=2**-0.5).eval()
    upstream_block = ResidualRecurrentBlock(
        hidsize=4096,
        timesteps=128,
        init_scale=2**-0.5,
        recurrence_type="transformer",
        is_residual=True,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        attention_heads=32,
        attention_memory_size=128,
        attention_mask_style="none",
        block_number=0,
    ).eval()
    _load_prefix(reproduction_block, arrays, prefix)
    _load_prefix(upstream_block, arrays, prefix)
    transformer_input = torch.randn(1, 5, 4096, generator=generator)
    first = torch.zeros(1, 5, dtype=torch.bool)
    state = upstream_block.r.initial_state(1)
    reproduction_output = reproduction_block(transformer_input)
    upstream_output, _ = upstream_block(transformer_input, first, state)
    component = compare_outputs(
        "attention_and_pointwise_block",
        reproduction_output,
        upstream_output,
        atol=5e-5,
        rtol=5e-5,
    )
    _parity_loss(reproduction_output).backward()
    _parity_loss(upstream_output).backward()
    component["parameter_gradient_parity"] = compare_parameter_gradients(
        "attention_and_pointwise_block",
        reproduction_block,
        upstream_block,
        atol=1e-4,
        rtol=1e-4,
    )
    components.append(component)
    del arrays, reproduction_block, upstream_block, transformer_input, reproduction_output, upstream_output, state
    gc.collect()

    last_prefix = "net.lastlayer."
    final_prefix = "net.final_ln."
    last_arrays = _arrays_for_prefix(weights, receipt, last_prefix)
    final_arrays = _arrays_for_prefix(weights, receipt, final_prefix)
    reproduction_last = _FanInLayer(4096, 4096, kind="linear", layer_norm=True).eval()
    reproduction_final = nn.LayerNorm(4096).eval()
    upstream_last = FanInInitReLULayer(4096, 4096, layer_type="linear", layer_norm=True).eval()
    upstream_final = nn.LayerNorm(4096).eval()
    _load_prefix(reproduction_last, last_arrays, last_prefix)
    _load_prefix(upstream_last, last_arrays, last_prefix)
    _load_prefix(reproduction_final, final_arrays, final_prefix)
    _load_prefix(upstream_final, final_arrays, final_prefix)
    reproduction_post = nn.Sequential(reproduction_last, reproduction_final)
    upstream_post = nn.Sequential(upstream_last, upstream_final)
    last_input = torch.randn(1, 5, 4096, generator=generator)
    reproduction_output = reproduction_post(last_input)
    upstream_output = upstream_post(last_input)
    component = compare_outputs(
        "post_transformer_dense_and_layernorm",
        reproduction_output,
        upstream_output,
    )
    _parity_loss(reproduction_output).backward()
    _parity_loss(upstream_output).backward()
    component["parameter_gradient_parity"] = compare_parameter_gradients(
        "post_transformer_dense_and_layernorm", reproduction_post, upstream_post
    )
    components.append(component)

    return {
        "schema_version": PARITY_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": "pass",
        "scope": "component-level FP32 parity with pinned public graph; not private-trainer bit exactness",
        "official_weights": {
            "bytes": weights.stat().st_size,
            "sha256": receipt["weights_artifact"]["sha256"],
        },
        "upstream_source": {
            "commit": receipt["upstream_source"]["commit"],
            "all_tracked_file_hashes_verified_before_import": True,
        },
        "security": {
            "weights_pickle_executed": False,
            "materializer": "tracked inert inventory plus direct ZIP storage-member reads",
        },
        "components": components,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(args.weights, args.receipt, args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
