#!/usr/bin/env python3
"""Seal C1/E1/E2 captures using label-only integrity and support evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

from data.mask_coverage import coverage_violations, measure_mask_coverage
from data.schema import KEY_ORDER
from data.validate_session import validate_session
from experiments.eval_vpt_small import sha256_file
from experiments.probe_shard_leak import probe_shard


SCHEMA_VERSION = "madeleine.vpt-small-calibration-capture-receipt.v1"
MIN_ACTIVE_MINUTES = 15.0
MIN_RUNS_PER_KEY = 25
MAX_DROP_RATE = 0.02
MAX_MARGIN_AUC = 0.90


def count_positive_state_runs(
    keys: np.ndarray,
    input_active: np.ndarray,
    engine_frame_idx: np.ndarray,
    room_id: np.ndarray,
) -> dict[str, int]:
    keys = np.asarray(keys, dtype=bool)
    active = np.asarray(input_active, dtype=bool)
    engine = np.asarray(engine_frame_idx, dtype=np.int64)
    room = np.asarray(room_id).astype(str)
    if keys.ndim != 2 or keys.shape[1] != len(KEY_ORDER):
        raise ValueError("keys must have shape [N,7]")
    if any(array.shape != (len(keys),) for array in (active, engine, room)):
        raise ValueError("support arrays are not aligned")
    if len(keys) == 0:
        raise ValueError("support is empty")
    eligible = np.zeros(len(keys), dtype=bool)
    eligible[1:] = (
        active[1:]
        & active[:-1]
        & (engine[1:] == engine[:-1] + 1)
        & (room[1:] == room[:-1])
    )
    onsets = np.zeros_like(keys, dtype=bool)
    onsets[1:] = keys[1:] & ~keys[:-1] & eligible[1:, None]
    return {key: int(onsets[:, column].sum()) for column, key in enumerate(KEY_ORDER)}


def _derived_common_support(
    derived_root: Path, session_id: str
) -> dict[str, np.ndarray]:
    manifest_path = derived_root / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("phases") != [0, 1, 2]:
        raise ValueError("capture derivation requires exact phases [0,1,2]")
    records = [record for record in manifest["records"] if record["session_id"] == session_id]
    if sorted(record["phase"] for record in records) != [0, 1, 2]:
        raise ValueError("capture derivation does not contain exactly three phase records")
    collected: dict[str, list[np.ndarray]] = {
        "source_row": [], "engine_idx": [], "truth": [], "active": [], "continuity": []
    }
    for record in records:
        directory = derived_root / f"{session_id}__p{record['phase']}"
        starts = np.load(directory / "window_start.npy", allow_pickle=False)
        selected = np.concatenate(
            [np.arange(int(start) + 32, int(start) + 96, dtype=np.int64) for start in starts]
        ) if len(starts) else np.empty(0, dtype=np.int64)
        if len(selected) != len(np.unique(selected)):
            if manifest.get("center_overlap_policy") != "base-first-stable-tail-fill":
                raise RuntimeError(f"overlapping center support in {directory}")
            selected = np.unique(selected)
        arrays = {
            "source_row": np.load(directory / "source_row_index.npy", mmap_mode="r", allow_pickle=False),
            "engine_idx": np.load(directory / "source_engine_frame_idx.npy", mmap_mode="r", allow_pickle=False),
            "truth": np.load(directory / "keys.npy", mmap_mode="r", allow_pickle=False),
            "active": np.load(directory / "input_active.npy", mmap_mode="r", allow_pickle=False),
            "continuity": np.load(directory / "continuity_id.npy", mmap_mode="r", allow_pickle=False),
        }
        for name, value in arrays.items():
            collected[name].append(np.asarray(value[selected]))
    merged = {name: np.concatenate(parts) for name, parts in collected.items()}
    order = np.argsort(merged["source_row"], kind="stable")
    merged = {name: value[order] for name, value in merged.items()}
    if len(merged["source_row"]) != len(np.unique(merged["source_row"])):
        raise RuntimeError("three-phase common support contains duplicate source rows")
    if not np.all(np.isfinite(merged["truth"])):
        raise RuntimeError("common-support truth is nonfinite")
    merged["manifest_sha256"] = np.asarray(sha256_file(manifest_path))
    return merged


def _rooms_for_engine(session_dir: Path, engine: np.ndarray) -> np.ndarray:
    truth = pq.read_table(session_dir / "truth.parquet", columns=["frame_idx", "room_id"])
    frames = np.asarray(truth["frame_idx"].to_pylist(), dtype=np.int64)
    rooms = np.asarray(truth["room_id"].to_pylist()).astype(str)
    lookup = {int(frame): room for frame, room in zip(frames, rooms, strict=True)}
    missing = [int(frame) for frame in engine if int(frame) not in lookup]
    if missing:
        raise RuntimeError(f"{len(missing)} common-support engine rows lack room identity")
    return np.asarray([lookup[int(frame)] for frame in engine])


def _strict_margin_auc(report: dict[str, Any]) -> float:
    values: list[float] = []
    for region_name in ("input_overlay", "wild_overlay"):
        region = report.get(region_name)
        if not region:
            continue
        for value in region["band_above_auc_per_key"].values():
            if value is not None:
                values.append(max(float(value), 1.0 - float(value)))
    return max(values, default=0.5)


def _require_committed(path: Path, repo: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo.resolve())
    except ValueError as error:
        raise RuntimeError(f"frozen prerequisite is outside the repository: {path}") from error
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)], cwd=repo, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=repo,
    ).returncode == 0
    if not clean:
        raise RuntimeError(f"frozen prerequisite differs from HEAD: {relative}")
    return {"path": str(relative), "sha256": sha256_file(resolved)}


def _source_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def seal_capture(
    *,
    role: str,
    session_dir: Path,
    shard_root: Path,
    derived_root: Path,
    chapter_sid: str,
    checkpoint: str,
    out: Path,
    repo: Path,
    frozen_calibrator: Path | None = None,
    frozen_eval_command: Path | None = None,
) -> dict[str, Any]:
    session_id = session_dir.name
    prerequisites: dict[str, Any] = {}
    if role in {"e1", "e2"}:
        if frozen_calibrator is None or frozen_eval_command is None:
            raise RuntimeError(f"{role.upper()} cannot be opened before calibrator and command freeze")
        prerequisites["calibrator"] = _require_committed(frozen_calibrator, repo)
        prerequisites["evaluation_command"] = _require_committed(frozen_eval_command, repo)

    violations = list(validate_session(session_dir))
    manifest_path = session_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("session_id") != session_id:
        violations.append("session manifest identity differs")
    integrity = manifest.get("integrity", {})
    video_frames = int(integrity.get("video_frames", 0))
    drops = int(integrity.get("drops", 0))
    drop_rate = drops / video_frames if video_frames else float("inf")
    if drop_rate > MAX_DROP_RATE:
        violations.append(f"capture drop rate {drop_rate:.6f} exceeds {MAX_DROP_RATE:.6f}")

    coverage_report = measure_mask_coverage(session_dir)
    violations.extend(coverage_violations(coverage_report))
    shard_path = shard_root / f"{session_id}.npz"
    if not shard_path.is_file():
        raise FileNotFoundError(f"fresh capture shard is missing: {shard_path}")
    leak_report = probe_shard(shard_path, manifest_path)
    for region_name in ("input_overlay", "wild_overlay"):
        if region_name in leak_report and not leak_report[region_name]["zone_is_all_zero"]:
            violations.append(f"{region_name} masked zone is not identically zero")
    max_margin_auc = _strict_margin_auc(leak_report)
    if max_margin_auc >= MAX_MARGIN_AUC:
        violations.append(f"adjacent-band leak AUC {max_margin_auc:.6f} reaches {MAX_MARGIN_AUC:.6f}")

    support = _derived_common_support(derived_root, session_id)
    room_id = _rooms_for_engine(session_dir, support["engine_idx"])
    run_counts = count_positive_state_runs(
        support["truth"], support["active"], support["engine_idx"], room_id
    )
    active_minutes = float(np.asarray(support["active"], dtype=bool).sum() / 60.0 / 60.0)
    if active_minutes < MIN_ACTIVE_MINUTES:
        violations.append(f"common-support active minutes {active_minutes:.6f} below {MIN_ACTIVE_MINUTES:.6f}")
    for key, count in run_counts.items():
        if count < MIN_RUNS_PER_KEY:
            violations.append(f"{key} positive state runs {count} below {MIN_RUNS_PER_KEY}")
    boundaries = np.flatnonzero(
        (np.diff(support["source_row"]) != 1)
        | (np.diff(support["engine_idx"]) != 1)
        | (room_id[1:] != room_id[:-1])
    ) + 1
    stream_lengths = np.diff(np.concatenate(([0], boundaries, [len(room_id)]))).astype(np.int64)
    support_payload = {
        "source_row_index": np.asarray(support["source_row"], dtype=np.int64).tolist(),
        "source_engine_frame_idx": np.asarray(support["engine_idx"], dtype=np.int64).tolist(),
        "stream_lengths": stream_lengths.tolist(),
        "room_id": room_id.tolist(),
    }
    support_hash = hashlib_sha256_json(support_payload)

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": _source_commit(repo),
        "role": role,
        "decision": "accepted" if not violations else "rejected",
        "violations": violations,
        "route": {"chapter_sid": chapter_sid, "checkpoint": checkpoint, "template": "vpt-small-calibration-v2-supported-route"},
        "session": {"session_id": session_id, "directory": str(session_dir), "manifest_sha256": sha256_file(manifest_path), "truth_sha256": sha256_file(session_dir / "truth.parquet"), "alignment_sha256": sha256_file(session_dir / "alignment.parquet"), "video_sha256": sha256_file(session_dir / "video.mkv")},
        "capture_integrity": {"video_frames": video_frames, "drops": drops, "drop_rate": drop_rate, "drop_gate": MAX_DROP_RATE, "validator_violations": list(validate_session(session_dir))},
        "mask_coverage": coverage_report,
        "leak_probe": {"report": leak_report, "max_symmetric_margin_auc": max_margin_auc, "gate": MAX_MARGIN_AUC},
        "shard": {"path": str(shard_path), "bytes": shard_path.stat().st_size, "sha256": sha256_file(shard_path), "build_manifest_sha256": sha256_file(shard_root / "build_manifest.json")},
        "derived": {"root": str(derived_root), "build_manifest_sha256": str(support["manifest_sha256"].item()), "complete_sha256": sha256_file(derived_root / "complete.json")},
        "support": {"rows": len(room_id), "active_rows": int(np.asarray(support["active"], dtype=bool).sum()), "active_minutes": active_minutes, "positive_state_runs": run_counts, "segments": len(stream_lengths), "rooms": sorted(set(room_id.tolist())), "support_sha256": support_hash},
        "frozen_prerequisites": prerequisites,
        "model_accessed": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if violations:
        raise RuntimeError(f"capture {session_id} rejected: {violations}")
    return receipt


def hashlib_sha256_json(value: Any) -> str:
    import hashlib
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--role", choices=("c1", "e1", "e2"), required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--chapter-sid", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--frozen-calibrator", type=Path)
    parser.add_argument("--frozen-eval-command", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = seal_capture(
        role=args.role,
        session_dir=args.session_dir,
        shard_root=args.shard_root,
        derived_root=args.derived_root,
        chapter_sid=args.chapter_sid,
        checkpoint=args.checkpoint,
        out=args.out,
        repo=args.repo,
        frozen_calibrator=args.frozen_calibrator,
        frozen_eval_command=args.frozen_eval_command,
    )
    print(json.dumps({"role": receipt["role"], "decision": receipt["decision"], "support": receipt["support"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
