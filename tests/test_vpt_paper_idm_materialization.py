from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
import pickle
import zipfile

import numpy as np
import pytest
import torch

from experiments.inspect_vpt_paper_idm_artifacts import (
    inspect_weights,
    materialize_weight_arrays,
    sha256_file,
)


def test_safe_materializer_recovers_exact_arrays_without_torch_load(tmp_path: Path) -> None:
    path = tmp_path / "fixture.weights"
    state = OrderedDict(
        [
            ("layer.weight", torch.arange(12, dtype=torch.float32).reshape(3, 4)),
            ("layer.bias", torch.tensor([1.25, -2.5, 7.0], dtype=torch.float32)),
        ]
    )
    torch.save(state, path)
    inventory = inspect_weights(path)

    arrays = materialize_weight_arrays(
        path,
        expected_sha256=sha256_file(path),
        inventory=inventory,
        names=["layer.bias", "layer.weight"],
    )

    assert list(arrays) == ["layer.bias", "layer.weight"]
    np.testing.assert_array_equal(arrays["layer.weight"], state["layer.weight"].numpy())
    np.testing.assert_array_equal(arrays["layer.bias"], state["layer.bias"].numpy())
    assert arrays["layer.weight"].flags.c_contiguous
    assert arrays["layer.weight"].flags.writeable


def test_safe_materializer_rejects_archive_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "fixture.weights"
    torch.save(OrderedDict([("x", torch.ones(2))]), path)
    inventory = inspect_weights(path)

    with pytest.raises(ValueError, match="weights SHA-256 mismatch"):
        materialize_weight_arrays(
            path,
            expected_sha256="0" * 64,
            inventory=inventory,
        )


def test_safe_materializer_never_opens_data_pickle(tmp_path: Path) -> None:
    path = tmp_path / "hostile.weights"
    payload = np.array([3.0, -4.0], dtype="<f4").tobytes()
    hostile_pickle = pickle.dumps(eval, protocol=4)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", hostile_pickle)
        archive.writestr("archive/data/0", payload)
    with path.open("rb") as raw, zipfile.ZipFile(raw) as archive:
        info = archive.getinfo("archive/data/0")
        raw.seek(info.header_offset)
        header = raw.read(30)
        filename_length = int.from_bytes(header[26:28], "little")
        extra_length = int.from_bytes(header[28:30], "little")
        start = info.header_offset + 30 + filename_length + extra_length
    inventory = {
        "tensors": [
            {
                "name": "x",
                "shape": [2],
                "stride": [1],
                "dtype": "float32",
                "elements": 2,
                "storage_elements": 2,
                "storage_byte_span": [0, 8],
                "archive_member": "archive/data/0",
                "archive_byte_span": [start, start + 8],
            }
        ]
    }

    arrays = materialize_weight_arrays(
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        inventory=inventory,
    )

    np.testing.assert_array_equal(arrays["x"], np.array([3.0, -4.0], dtype=np.float32))
