#!/usr/bin/env python3
"""Safely inventory the official OpenAI VPT IDM artifacts.

The released ``.model`` and ``.weights`` files are pickle-bearing formats.
This module never imports the packages named by those pickles and never calls
arbitrary pickle globals.  A deliberately small allowlist reconstructs only
inert gym type metadata and inert tensor/storage references.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import pickle
import struct
from typing import Any, BinaryIO, Callable, Mapping, Sequence
import zipfile

import numpy as np


RECEIPT_SCHEMA = "madeleine.vpt-paper-idm-official-artifacts.v1"
UPSTREAM_WARNING = "this is _not_ the original code of IDM"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UnsafePickleGlobalError(pickle.UnpicklingError):
    """Raised when an artifact asks the restricted loader for a global."""


class _InertGymType:
    """State-only stand-in for the three gym3 types in ``4x_idm.model``."""

    pickle_type = "gym3.types.Unknown"

    def __new__(cls, *args: object, **kwargs: object) -> "_InertGymType":
        del args, kwargs
        return super().__new__(cls)

    def __setstate__(self, state: object) -> None:
        if not isinstance(state, dict):
            raise pickle.UnpicklingError("gym type state must be a dictionary")
        self.state = state


def _gym_type(name: str) -> type[_InertGymType]:
    return type(
        f"Inert{name}",
        (_InertGymType,),
        {"pickle_type": f"gym3.types.{name}"},
    )


_GYM_GLOBALS = {
    ("gym3.types", "DictType"): _gym_type("DictType"),
    ("gym3.types", "TensorType"): _gym_type("TensorType"),
    ("gym3.types", "Discrete"): _gym_type("Discrete"),
}


class _ModelUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        allowed = _GYM_GLOBALS.get((module, name))
        if allowed is None:
            raise UnsafePickleGlobalError(f"forbidden model pickle global: {module}.{name}")
        return allowed

    def persistent_load(self, pid: object) -> object:
        raise pickle.UnpicklingError(f"model pickle has unexpected persistent id: {pid!r}")


def safe_load_model(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = _ModelUnpickler(handle).load()
        if handle.read(1):
            raise pickle.UnpicklingError("trailing bytes after model pickle")
    if not isinstance(value, dict):
        raise pickle.UnpicklingError("model artifact must contain a dictionary")
    return value


def _normalize_inert(value: object) -> object:
    if isinstance(value, _InertGymType):
        state = getattr(value, "state", {})
        return {
            "pickle_type": value.pickle_type,
            "state": _normalize_inert(state),
        }
    if isinstance(value, dict):
        return {str(key): _normalize_inert(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_inert(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported normalized model value: {type(value)!r}")


def normalize_model_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        model = payload["model"]
        model_args = model["args"]
        net = model_args["net"]
        net_args = net["args"]
        pi_head_opts = model_args["pi_head_opts"]
    except (KeyError, TypeError) as exc:
        raise ValueError("released model metadata has an unexpected structure") from exc
    if not isinstance(net_args, dict) or not isinstance(pi_head_opts, dict):
        raise ValueError("network and action-head options must be dictionaries")
    return {
        "version": int(payload["version"]),
        "model_function": str(model["function"]),
        "network_function": str(net["function"]),
        "network_args": _normalize_inert(net_args),
        "pi_head_options": _normalize_inert(pi_head_opts),
        "action_space": _normalize_inert(payload["extra_args"]["ac_space"]),
        "observation_space": _normalize_inert(payload["extra_args"]["ob_space"]),
    }


@dataclass(frozen=True)
class _StorageType:
    pickle_global: str
    dtype: str
    itemsize: int


@dataclass(frozen=True)
class StorageRef:
    storage_type: _StorageType
    key: str
    location: str
    elements: int


@dataclass(frozen=True)
class TensorRef:
    storage: StorageRef
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    requires_grad: bool

    @property
    def elements(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    @property
    def storage_byte_span(self) -> tuple[int, int]:
        start = self.storage_offset * self.storage.storage_type.itemsize
        if not self.shape or any(dimension == 0 for dimension in self.shape):
            return start, start
        last = self.storage_offset
        for dimension, stride in zip(self.shape, self.stride, strict=True):
            if dimension < 0 or stride < 0:
                raise ValueError("negative tensor dimensions/strides are unsupported")
            last += (dimension - 1) * stride
        return start, (last + 1) * self.storage.storage_type.itemsize


_STORAGE_GLOBALS: dict[tuple[str, str], _StorageType] = {}
for _name, _dtype, _itemsize in (
    ("FloatStorage", "float32", 4),
    ("DoubleStorage", "float64", 8),
    ("HalfStorage", "float16", 2),
    ("BFloat16Storage", "bfloat16", 2),
    ("LongStorage", "int64", 8),
    ("IntStorage", "int32", 4),
    ("ShortStorage", "int16", 2),
    ("CharStorage", "int8", 1),
    ("ByteStorage", "uint8", 1),
    ("BoolStorage", "bool", 1),
):
    _STORAGE_GLOBALS[("torch", _name)] = _StorageType(
        pickle_global=f"torch.{_name}", dtype=_dtype, itemsize=_itemsize
    )


def _rebuild_tensor(
    storage: StorageRef,
    storage_offset: int,
    shape: Sequence[int],
    stride: Sequence[int],
    *tail: object,
) -> TensorRef:
    requires_grad = bool(tail[0]) if tail else False
    return TensorRef(
        storage=storage,
        storage_offset=int(storage_offset),
        shape=tuple(int(item) for item in shape),
        stride=tuple(int(item) for item in stride),
        requires_grad=requires_grad,
    )


def _rebuild_parameter(tensor: TensorRef, requires_grad: bool, *tail: object) -> TensorRef:
    del tail
    return TensorRef(
        storage=tensor.storage,
        storage_offset=tensor.storage_offset,
        shape=tensor.shape,
        stride=tensor.stride,
        requires_grad=bool(requires_grad),
    )


_REBUILD_GLOBALS: dict[tuple[str, str], Callable[..., object] | type[OrderedDict]] = {
    ("collections", "OrderedDict"): OrderedDict,
    ("torch._utils", "_rebuild_tensor"): _rebuild_tensor,
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
}


class _WeightsUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        allowed = _STORAGE_GLOBALS.get((module, name))
        if allowed is None:
            allowed = _REBUILD_GLOBALS.get((module, name))
        if allowed is None:
            raise UnsafePickleGlobalError(f"forbidden weights pickle global: {module}.{name}")
        return allowed

    def persistent_load(self, pid: object) -> StorageRef:
        if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage":
            raise pickle.UnpicklingError(f"unexpected weights persistent id: {pid!r}")
        _, storage_type, key, location, elements, *view = pid
        if view:
            raise pickle.UnpicklingError("storage views are not supported in official inventory")
        if not isinstance(storage_type, _StorageType):
            raise pickle.UnpicklingError("persistent storage has an unknown type")
        return StorageRef(
            storage_type=storage_type,
            key=str(key),
            location=str(location),
            elements=int(elements),
        )


def _zip_data_offset(handle: BinaryIO, info: zipfile.ZipInfo) -> int:
    handle.seek(info.header_offset)
    header = handle.read(30)
    if len(header) != 30:
        raise ValueError(f"truncated local ZIP header: {info.filename}")
    fields = struct.unpack("<IHHHHHIIIHH", header)
    signature, filename_length, extra_length = fields[0], fields[-2], fields[-1]
    if signature != 0x04034B50:
        raise ValueError(f"invalid local ZIP header: {info.filename}")
    return info.header_offset + 30 + filename_length + extra_length


def inspect_weights(path: Path) -> dict[str, Any]:
    with path.open("rb") as raw, zipfile.ZipFile(raw) as archive:
        infos = {info.filename: info for info in archive.infolist()}
        pickle_members = [name for name in infos if name.endswith("/data.pkl")]
        if len(pickle_members) != 1:
            raise ValueError(f"expected one data.pkl in weights archive, found {pickle_members}")
        data_pickle = pickle_members[0]
        prefix = data_pickle[: -len("data.pkl")]
        state = _WeightsUnpickler(io.BytesIO(archive.read(data_pickle))).load()
        if not isinstance(state, (dict, OrderedDict)):
            raise ValueError("weights data.pkl must contain a state dictionary")

        tensors: list[dict[str, Any]] = []
        storages: dict[str, StorageRef] = {}
        named_elements = 0
        for name, tensor in state.items():
            if not isinstance(name, str) or not isinstance(tensor, TensorRef):
                raise ValueError(f"unexpected state entry: {name!r} -> {type(tensor)!r}")
            member_name = f"{prefix}data/{tensor.storage.key}"
            info = infos.get(member_name)
            if info is None:
                raise ValueError(f"missing storage member for tensor {name}: {member_name}")
            expected_storage_bytes = tensor.storage.elements * tensor.storage.storage_type.itemsize
            if info.file_size != expected_storage_bytes:
                raise ValueError(
                    f"storage size mismatch for {name}: {info.file_size} != {expected_storage_bytes}"
                )
            span = tensor.storage_byte_span
            if span[1] > info.file_size:
                raise ValueError(f"tensor {name} exceeds storage member {member_name}")
            data_offset = _zip_data_offset(raw, info)
            tensors.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "stride": list(tensor.stride),
                    "dtype": tensor.storage.storage_type.dtype,
                    "requires_grad": tensor.requires_grad,
                    "elements": tensor.elements,
                    "storage_key": tensor.storage.key,
                    "storage_elements": tensor.storage.elements,
                    "storage_byte_span": list(span),
                    "archive_member": member_name,
                    "archive_byte_span": [data_offset, data_offset + info.compress_size],
                    "archive_compression": info.compress_type,
                }
            )
            named_elements += tensor.elements
            previous = storages.setdefault(tensor.storage.key, tensor.storage)
            if previous != tensor.storage:
                raise ValueError(f"inconsistent metadata for storage {tensor.storage.key}")

        unique_storage_elements = sum(storage.elements for storage in storages.values())
        unique_storage_bytes = sum(
            storage.elements * storage.storage_type.itemsize for storage in storages.values()
        )
        archive_storage_members = {
            name for name in infos if name.startswith(f"{prefix}data/")
        }
        referenced_storage_members = {
            f"{prefix}data/{storage_key}" for storage_key in storages
        }
        unreferenced = sorted(archive_storage_members - referenced_storage_members)
        missing_references = sorted(referenced_storage_members - archive_storage_members)
        if unreferenced or missing_references:
            raise ValueError(
                "weights storage inventory is not exact: "
                f"unreferenced={unreferenced}, missing={missing_references}"
            )
        return {
            "archive_prefix": prefix,
            "data_pickle_member": data_pickle,
            "tensor_count": len(tensors),
            "unique_storage_count": len(storages),
            "named_tensor_elements": named_elements,
            "unique_storage_elements": unique_storage_elements,
            "unique_storage_bytes": unique_storage_bytes,
            "storage_alias_elements": named_elements - unique_storage_elements,
            "unreferenced_storage_members": unreferenced,
            "tensors": tensors,
        }


def materialize_weight_arrays(
    path: Path,
    *,
    expected_sha256: str,
    inventory: Mapping[str, Any],
    names: Sequence[str] | None = None,
) -> OrderedDict[str, np.ndarray]:
    """Materialize selected FP32 tensors without executing the archive pickle.

    The trusted input is the tracked, independently generated inert inventory.
    The full archive hash is checked before any bytes are exposed as arrays.
    Tensor bytes are then read directly from their declared ZIP storage members;
    ``data.pkl`` is never opened by this function.
    """

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"weights SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    tensors = inventory.get("tensors")
    if not isinstance(tensors, list):
        raise ValueError("weights inventory does not contain a tensor list")
    by_name: dict[str, Mapping[str, Any]] = {}
    for item in tensors:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise ValueError("weights inventory has a malformed tensor record")
        name = str(item["name"])
        if name in by_name:
            raise ValueError(f"duplicate tensor in weights inventory: {name}")
        by_name[name] = item

    selected = list(by_name) if names is None else list(names)
    if len(selected) != len(set(selected)):
        raise ValueError("requested materialization names are not unique")
    missing = sorted(set(selected) - set(by_name))
    if missing:
        raise KeyError(f"requested tensors are absent from inventory: {missing}")

    result: OrderedDict[str, np.ndarray] = OrderedDict()
    with path.open("rb") as raw, zipfile.ZipFile(raw) as archive:
        infos = {info.filename: info for info in archive.infolist()}
        for name in selected:
            item = by_name[name]
            if item.get("dtype") != "float32":
                raise ValueError(f"safe materializer supports FP32 only: {name}")
            member = item.get("archive_member")
            if not isinstance(member, str) or member not in infos:
                raise ValueError(f"declared archive member is missing for {name}")
            info = infos[member]
            declared_archive_span = item.get("archive_byte_span")
            observed_archive_span = [
                _zip_data_offset(raw, info),
                _zip_data_offset(raw, info) + info.compress_size,
            ]
            if declared_archive_span != observed_archive_span:
                raise ValueError(f"archive byte span changed for {name}")
            expected_storage_bytes = int(item["storage_elements"]) * 4
            if info.file_size != expected_storage_bytes:
                raise ValueError(f"storage byte count changed for {name}")
            payload = archive.read(member)  # ZIP CRC is checked by zipfile.
            if len(payload) != expected_storage_bytes:
                raise ValueError(f"short storage payload for {name}")

            shape = tuple(int(value) for value in item["shape"])
            strides = tuple(int(value) * 4 for value in item["stride"])
            span = item.get("storage_byte_span")
            if not isinstance(span, list) or len(span) != 2:
                raise ValueError(f"malformed storage byte span for {name}")
            offset = int(span[0])
            if int(span[1]) > len(payload):
                raise ValueError(f"tensor storage span exceeds payload for {name}")
            array = np.ndarray(
                shape=shape,
                dtype=np.dtype("<f4"),
                buffer=payload,
                offset=offset,
                strides=strides,
            ).copy(order="C")
            if int(array.size) != int(item["elements"]):
                raise ValueError(f"materialized element count changed for {name}")
            result[name] = array
    return result


def source_inventory(root: Path, commit: str) -> dict[str, Any]:
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
    files = []
    for relative in required:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    demo = (root / "run_inverse_dynamics_model.py").read_text(encoding="utf-8")
    if UPSTREAM_WARNING not in demo:
        raise ValueError("upstream demo no longer contains its non-original-code warning")
    return {
        "repository": "https://github.com/openai/Video-Pre-Training",
        "commit": commit,
        "demo_warning_verified": True,
        "demo_warning": UPSTREAM_WARNING,
        "files": files,
    }


def build_receipt(
    *, model_path: Path, weights_path: Path, source_root: Path, source_commit: str
) -> dict[str, Any]:
    model_payload = safe_load_model(model_path)
    return {
        "schema_version": RECEIPT_SCHEMA,
        "authority_order": [
            "released 4x_idm.model and complete 4x_idm.weights inventory",
            "VPT paper Appendix D where consistent with both artifacts",
            "pinned public repository only where released artifacts are silent",
        ],
        "claims": {
            "reproduction_scope": "public-artifact architecture and paper training recipe",
            "bit_exact_private_training_code": False,
            "reason": "OpenAI did not release the original IDM training code",
        },
        "model_artifact": {
            "url": "https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.model",
            "bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
            "metadata": normalize_model_metadata(model_payload),
        },
        "weights_artifact": {
            "url": "https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.weights",
            "bytes": weights_path.stat().st_size,
            "sha256": sha256_file(weights_path),
            "inventory": inspect_weights(weights_path),
        },
        "upstream_source": source_inventory(source_root, source_commit),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = build_receipt(
        model_path=args.model,
        weights_path=args.weights,
        source_root=args.source_root,
        source_commit=args.source_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
