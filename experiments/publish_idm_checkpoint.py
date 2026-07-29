"""Publish one retained IDM checkpoint to durable R2, marker last.

The destination is content addressed and must be empty.  Only ``model.pt``
and its deterministic manifest are uploaded before the completion marker.
Every remote object is streamed back to verify its byte count and SHA-256;
R2 ETags are deliberately ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence


# Bucket identity is operational and stays out of Git (the contributor guide
# requires placeholders or environment variables for host and account
# identifiers). Real publishes set MADELEINE_R2_BUCKET_URI; the placeholder
# default keeps the command line structurally complete in dry contexts.
R2_BUCKET_URI = os.environ.get("MADELEINE_R2_BUCKET_URI", "r2:<bucket>")

MANIFEST_VERSION = "madeleine.idm-checkpoint-manifest.v1"
COMPLETION_VERSION = "madeleine.idm-checkpoint-complete.v1"
ARTIFACT_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _run(args: Sequence[str], *, capture: bool = False) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(args),
        check=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def remote_inventory(remote: str) -> list[str]:
    completed = _run(
        ("rclone", "lsf", remote, "--files-only", "--format", "p"),
        capture=True,
    )
    return sorted(
        line.strip()
        for line in completed.stdout.decode("utf-8").splitlines()
        if line.strip()
    )


def remote_sha256(remote: str) -> tuple[str, int]:
    process = subprocess.Popen(
        ["rclone", "cat", remote],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
        size += len(block)
        digest.update(block)
    stderr = process.stderr.read() if process.stderr is not None else b""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(
            f"rclone cat failed for {remote}: {stderr.decode('utf-8', 'replace')}"
        )
    if size == 0:
        raise ValueError(f"remote object is absent or empty: {remote}")
    return digest.hexdigest(), size


def _copy_verified(local: Path, remote: str) -> None:
    _run(
        (
            "rclone",
            "copyto",
            str(local),
            remote,
            "--immutable",
            "--transfers",
            "1",
        )
    )
    remote_hash, remote_size = remote_sha256(remote)
    if remote_hash != sha256_file(local) or remote_size != local.stat().st_size:
        raise ValueError(f"remote verification failed: {remote}")


def publish(
    *,
    run: Path,
    artifact_id: str,
    role: str,
    remote_root: str,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    if not ARTIFACT_ID.fullmatch(artifact_id):
        raise ValueError("artifact_id must be lowercase hyphen-separated tokens")
    if not role.strip():
        raise ValueError("role must not be empty")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_checkpoint_sha256):
        raise ValueError("expected checkpoint SHA-256 is malformed")

    checkpoint = run / "model.pt"
    config = run / "config.json"
    run_meta = run / "run_meta.json"
    for path in (checkpoint, config, run_meta):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required run artifact is absent or empty: {path}")
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != expected_checkpoint_sha256:
        raise ValueError(
            "checkpoint SHA-256 changed: "
            f"expected {expected_checkpoint_sha256}, got {checkpoint_hash}"
        )

    manifest_path = run / "checkpoint-manifest.json"
    completion_path = run / "checkpoint_complete.json"
    if manifest_path.exists() or completion_path.exists():
        raise FileExistsError("local checkpoint publication receipts already exist")
    manifest = {
        "artifact_id": artifact_id,
        "checkpoint": {
            "bytes": checkpoint.stat().st_size,
            "filename": checkpoint.name,
            "sha256": checkpoint_hash,
        },
        "format_version": MANIFEST_VERSION,
        "metadata_hashes": {
            "config_sha256": sha256_file(config),
            "run_meta_sha256": sha256_file(run_meta),
        },
        "publication_policy": (
            "private durable backup; hash-addressed; checkpoint bytes are not "
            "part of the public repository"
        ),
        "role": role,
    }
    _write_json(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    completion = {
        "artifact_id": artifact_id,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_hash,
        "format_version": COMPLETION_VERSION,
        "manifest_sha256": manifest_hash,
        "payload_object_count": 2,
        "verification": (
            "full remote byte-stream SHA-256 and byte count; completion marker "
            "uploaded last"
        ),
    }
    _write_json(completion_path, completion)

    destination = f"{remote_root.rstrip('/')}/{artifact_id}/{checkpoint_hash}"
    if remote_inventory(destination):
        raise FileExistsError(f"immutable destination is not empty: {destination}")
    _copy_verified(checkpoint, f"{destination}/model.pt")
    _copy_verified(manifest_path, f"{destination}/{manifest_path.name}")
    if remote_inventory(destination) != ["checkpoint-manifest.json", "model.pt"]:
        raise ValueError("remote payload inventory differs before completion")

    _copy_verified(completion_path, f"{destination}/{completion_path.name}")
    expected_inventory = [
        "checkpoint-manifest.json",
        "checkpoint_complete.json",
        "model.pt",
    ]
    if remote_inventory(destination) != expected_inventory:
        raise ValueError("remote inventory differs after completion")
    for path in (checkpoint, manifest_path, completion_path):
        remote_hash, remote_size = remote_sha256(f"{destination}/{path.name}")
        if remote_hash != sha256_file(path) or remote_size != path.stat().st_size:
            raise ValueError(f"final remote verification failed: {path.name}")
    return {**completion, "object_prefix": destination}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument(
        "--remote-root", default=f"{R2_BUCKET_URI}/runs/idm/v1"
    )
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = publish(
        run=args.run,
        artifact_id=args.artifact_id,
        role=args.role,
        remote_root=args.remote_root,
        expected_checkpoint_sha256=args.expect_checkpoint_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
