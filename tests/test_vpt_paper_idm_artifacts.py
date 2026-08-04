from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import pickle

import pytest
import torch

from experiments.inspect_vpt_paper_idm_artifacts import (
    UnsafePickleGlobalError,
    inspect_weights,
    normalize_model_metadata,
    safe_load_model,
    source_inventory,
)


def _plain_model_metadata() -> dict[str, object]:
    return {
        "version": 1361,
        "model": {
            "function": "ypt.model.inverse_action_model:create",
            "args": {
                "net": {
                    "function": "ypt.model.inverse_action_model:InverseActionNet",
                    "args": {
                        "hidsize": 4096,
                        "attention_heads": 32,
                        "n_recurrence_layers": 2,
                    },
                },
                "pi_head_opts": {"temperature": 4},
            },
        },
        "extra_args": {"ac_space": {}, "ob_space": {}},
    }


def test_model_loader_accepts_data_only_pickle_and_normalizes_architecture(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture.model"
    path.write_bytes(pickle.dumps(_plain_model_metadata(), protocol=4))

    normalized = normalize_model_metadata(safe_load_model(path))

    assert normalized["version"] == 1361
    assert normalized["network_args"] == {
        "attention_heads": 32,
        "hidsize": 4096,
        "n_recurrence_layers": 2,
    }
    assert normalized["pi_head_options"] == {"temperature": 4}


def test_model_loader_rejects_unallowlisted_globals_without_calling_them(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unsafe.model"
    path.write_bytes(pickle.dumps(eval, protocol=4))

    with pytest.raises(UnsafePickleGlobalError, match="builtins.eval"):
        safe_load_model(path)


def test_weights_inventory_reconstructs_inert_tensors_and_archive_spans(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture.weights"
    state = OrderedDict(
        [
            ("layer.weight", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
            ("layer.bias", torch.arange(2, dtype=torch.float32)),
        ]
    )
    torch.save(state, path)

    inventory = inspect_weights(path)

    assert inventory["tensor_count"] == 2
    assert inventory["unique_storage_count"] == 2
    assert inventory["named_tensor_elements"] == 8
    assert inventory["unique_storage_bytes"] == 32
    assert inventory["storage_alias_elements"] == 0
    assert inventory["unreferenced_storage_members"] == []
    weight, bias = inventory["tensors"]
    assert weight["name"] == "layer.weight"
    assert weight["shape"] == [2, 3]
    assert weight["stride"] == [3, 1]
    assert weight["storage_byte_span"] == [0, 24]
    assert weight["archive_byte_span"][1] - weight["archive_byte_span"][0] == 24
    assert bias["storage_byte_span"] == [0, 8]


def test_source_inventory_requires_upstream_non_original_code_warning(
    tmp_path: Path,
) -> None:
    required = (
        "README.md",
        "inverse_dynamics_model.py",
        "run_inverse_dynamics_model.py",
        "lib/action_head.py",
        "lib/impala_cnn.py",
        "lib/masked_attention.py",
        "lib/mlp.py",
        "lib/policy.py",
        "lib/torch_util.py",
        "lib/util.py",
        "lib/xf.py",
    )
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    (tmp_path / "run_inverse_dynamics_model.py").write_text(
        "# NOTE: this is _not_ the original code of IDM!\n", encoding="utf-8"
    )

    inventory = source_inventory(tmp_path, "abc123")

    assert inventory["commit"] == "abc123"
    assert inventory["demo_warning_verified"] is True
    assert len(inventory["files"]) == len(required)

    (tmp_path / "run_inverse_dynamics_model.py").write_text(
        "# warning removed\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="non-original-code warning"):
        source_inventory(tmp_path, "abc123")
