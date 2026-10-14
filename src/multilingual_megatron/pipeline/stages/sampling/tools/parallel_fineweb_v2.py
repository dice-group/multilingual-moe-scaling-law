#!/usr/bin/env python3
"""Parallel, deterministic FineWeb acquisition for large language pools.

Workers transform disjoint pinned Parquet files into source-file-owned split
fragments.  The merge reads compact ordered indexes to determine the exact
sequential stopping point, then concatenates complete fragments and truncates
only the terminal source file.  Expensive Parquet decoding and JSON encoding
therefore scale across Slurm ranks while the final merge remains deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    assign_validation_reserve,
    canonical_json_sha256,
    sha256_file,
    stable_document_id,
    verify_plan_hash,
)
from multilingual_megatron.pipeline.stages.sampling.tools import bootstrap_fineweb_v2 as base


SCHEMA_VERSION = 1
WORKER_TOOL_ID = "fineweb_ab_v2_parallel_worker_v1"
MERGE_TOOL_ID = "fineweb_ab_v2_parallel_merge_v1"
FRAGMENT_NAMES = ("train.jsonl", "validation_reserve.jsonl", "ordered_index.jsonl")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    base._atomic_write_json(path, payload)


def _plan_hash(plan: Mapping[str, Any]) -> str:
    candidate = dict(plan)
    expected = candidate.pop("plan_sha256", None)
    if expected != canonical_json_sha256(candidate):
        raise ValueError("parallel cache plan hash mismatch")
    return str(expected)


def _adapter(path_to_index: Mapping[str, int]):
    def adapt(reader: Any, data: dict[str, Any], path: str, row: int) -> dict[str, Any]:
        payload = base._fineweb_parquet_adapter(reader, data, path, row)
        source_path = str(path)
        if source_path not in path_to_index:
            raise RuntimeError(f"reader returned unplanned source path: {source_path}")
        payload["metadata"]["parallel_source_index"] = int(path_to_index[source_path])
        return payload

    return adapt


def _write_line(handle: Any, payload: Mapping[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    handle.write("\n")


def run_worker(
    *,
    cache_plan_path: Path,
    cache_root: Path,
    fragment_root: Path,
    source_lock_path: Path,
    rank: int,
    world_size: int,
    seed: int,
    validation_fraction: float,
) -> dict[str, Any]:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid parallel rank/world size")
    plan = _load(cache_plan_path.resolve())
    plan_sha = _plan_hash(plan)
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("parallel cache plan has no entries")
    assigned = [row for row in entries if int(row["index"]) % world_size == rank]
    if not assigned:
        raise ValueError(f"rank {rank} has no assigned source files")

    source_lock = _load(source_lock_path.resolve())
    source = base._source_identity(source_lock, "eng_Latn")
    if source.dataset != plan.get("dataset") or source.revision != plan.get("revision"):
        raise ValueError("parallel cache plan differs from the pinned English source")
    path_to_index = {str(row["path"]): int(row["index"]) for row in assigned}
    paths_file = fragment_root / "worker_paths" / f"rank_{rank:03d}.txt"
    base._atomic_write_text(paths_file, "".join(f"{row['path']}\n" for row in assigned))

    try:
        from datatrove.pipeline.readers import ParquetReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("datatrove is required for parallel FineWeb sampling") from exc
    reader = ParquetReader(
        str(cache_root.resolve()),
        paths_file=str(paths_file.resolve()),
        read_metadata=True,
        file_progress=False,
        doc_progress=False,
        shuffle_files=False,
        adapter=_adapter(path_to_index),
    )
    expected_sha = {str(row["path"]): str(row["lfs_sha256"]) for row in assigned}
    state: dict[int, dict[str, Any]] = {}

    def get_state(index: int) -> dict[str, Any]:
        if index in state:
            return state[index]
        root = fragment_root / "files" / f"file_{index:05d}"
        root.mkdir(parents=True, exist_ok=False)
        handles = {
            name: (root / name).open("w", encoding="utf-8") for name in FRAGMENT_NAMES
        }
        state[index] = {
            "root": root,
            "handles": handles,
            "seen": set(),
            "raw": 0,
            "bytes": 0,
            "train": 0,
            "validation": 0,
            "duplicates": 0,
            "path": next(str(row["path"]) for row in assigned if int(row["index"]) == index),
        }
        return state[index]

    for document in reader.run(rank=0, world_size=1):
        metadata = document.metadata if isinstance(document.metadata, dict) else {}
        index = int(metadata["parallel_source_index"])
        current = get_state(index)
        current["raw"] += 1
        text = str(document.text)
        original_id = str(document.id)
        url = str(metadata.get("url", "") or "")
        stable_id = stable_document_id(
            source_dataset=source.dataset,
            source_revision=source.revision,
            subset=source.subset,
            source_id=original_id,
            url=url,
            text=text,
        )
        if stable_id in current["seen"]:
            current["duplicates"] += 1
            continue
        current["seen"].add(stable_id)
        utf8_bytes = len(text.encode("utf-8"))
        current["bytes"] += utf8_bytes
        split = (
            "validation_reserve"
            if assign_validation_reserve(
                document_id=stable_id,
                seed=seed,
                validation_fraction=validation_fraction,
            )
            else "train"
        )
        current["validation" if split == "validation_reserve" else "train"] += 1
        record = base._record(
            text=text,
            stable_id=stable_id,
            original_id=original_id,
            url=url,
            source=source,
            source_file=current["path"],
            source_file_sha256=expected_sha[current["path"]],
            split=split,
            artifact_role=split,
            rehydration_multiplier=1,
            train_rehydration_multiplier=1,
            cluster_size=1,
        )
        _write_line(current["handles"][f"{split}.jsonl"], record)
        _write_line(
            current["handles"]["ordered_index.jsonl"],
            {
                "document_id": stable_id,
                "split": split,
                "utf8_bytes": utf8_bytes,
            },
        )

    reports = []
    for index in sorted(state):
        current = state[index]
        for handle in current["handles"].values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        outputs = {
            name: {
                "sha256": sha256_file(current["root"] / name),
                "bytes": (current["root"] / name).stat().st_size,
            }
            for name in FRAGMENT_NAMES
        }
        report = {
            "schema_version": SCHEMA_VERSION,
            "tool_id": WORKER_TOOL_ID,
            "status": "complete",
            "plan_sha256": plan_sha,
            "rank": rank,
            "world_size": world_size,
            "source_index": index,
            "source_path": current["path"],
            "source_file_sha256": expected_sha[current["path"]],
            "counts": {
                "raw_documents": current["raw"],
                "unique_documents": len(current["seen"]),
                "unique_utf8_bytes": current["bytes"],
                "train_documents": current["train"],
                "validation_documents": current["validation"],
                "duplicates_within_file": current["duplicates"],
            },
            "outputs": outputs,
        }
        report["report_sha256"] = canonical_json_sha256(report)
        _atomic_json(current["root"] / "fragment_report.json", report)
        reports.append(report["report_sha256"])
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "tool_id": WORKER_TOOL_ID,
        "status": "complete",
        "plan_sha256": plan_sha,
        "rank": rank,
        "world_size": world_size,
        "source_indices": sorted(state),
        "fragment_report_sha256": reports,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    _atomic_json(fragment_root / "receipts" / f"rank_{rank:03d}.json", receipt)
    return receipt


def _copy_lines(source: Path, target: Any, count: int | None = None) -> int:
    copied = 0
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if count is not None and copied >= count:
                break
            target.write(line)
            copied += 1
    return copied


def _canonical_id_hash(ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, value in enumerate(sorted(ids)):
        if index:
            digest.update(b",")
        digest.update(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    digest.update(b"]")
    return digest.hexdigest()


def merge_fragments(
    *,
    cache_plan_path: Path,
    cache_root: Path,
    fragment_root: Path,
    output_root: Path,
    source_lock_path: Path,
    rehydration_weights_path: Path,
    source_inventory_path: Path,
    language_map_path: Path,
    sampling_plan_path: Path | None,
    world_size: int,
    seed: int,
    validation_fraction: float,
    safety_margin: float,
    settings_override: base.BootstrapSettings | None = None,
    production_target_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = _load(cache_plan_path.resolve())
    plan_sha = _plan_hash(plan)
    if settings_override is None:
        if sampling_plan_path is None:
            raise ValueError("sampling_plan_path is required without a test override")
        sampling_plan = _load(sampling_plan_path.resolve())
        if not verify_plan_hash(sampling_plan):
            raise ValueError("sampling plan hash mismatch")
        settings, production_target = base._derive_production_settings(
            base=base.BootstrapSettings(seed=seed, validation_fraction=validation_fraction),
            language="eng_Latn",
            sampling_plan=sampling_plan,
            safety_margin=safety_margin,
        )
        production_target["sampling_plan_file_sha256"] = sha256_file(sampling_plan_path)
    else:
        settings = settings_override
        settings.validate()
        production_target = dict(production_target_override or {"test_override": True})
    entries = plan["entries"]
    reports: list[dict[str, Any]] = []
    for rank in range(world_size):
        receipt = _load(fragment_root / "receipts" / f"rank_{rank:03d}.json")
        expected = receipt.pop("receipt_sha256", None)
        if expected != canonical_json_sha256(receipt) or receipt.get("plan_sha256") != plan_sha:
            raise RuntimeError(f"invalid parallel worker receipt {rank}")
    for entry in entries:
        root = fragment_root / "files" / f"file_{int(entry['index']):05d}"
        report = _load(root / "fragment_report.json")
        expected = report.pop("report_sha256", None)
        if expected != canonical_json_sha256(report):
            raise RuntimeError(f"fragment report hash mismatch: {root}")
        report["report_sha256"] = expected
        for name in FRAGMENT_NAMES:
            if sha256_file(root / name) != report["outputs"][name]["sha256"]:
                raise RuntimeError(f"fragment output hash mismatch: {root / name}")
        reports.append(report)

    selected: list[tuple[dict[str, Any], int, int]] = []
    seen: set[str] = set()
    train_ids: set[str] = set()
    validation_ids: set[str] = set()
    calibration_ids: list[str] = []
    total_bytes = 0
    raw_documents = 0
    duplicates = 0
    targets_met = False
    for report in reports:
        root = fragment_root / "files" / f"file_{int(report['source_index']):05d}"
        train_count = validation_count = 0
        with (root / "ordered_index.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                raw_documents += 1
                document_id = str(row["document_id"])
                if document_id in seen:
                    duplicates += 1
                    raise RuntimeError(
                        "cross-source duplicate detected; refusing fast merge because "
                        "sequential equivalence would require record-level rewriting"
                    )
                seen.add(document_id)
                total_bytes += int(row["utf8_bytes"])
                if row["split"] == "train":
                    train_ids.add(document_id)
                    train_count += 1
                    if len(calibration_ids) < settings.calibration_documents:
                        calibration_ids.append(document_id)
                else:
                    validation_ids.add(document_id)
                    validation_count += 1
                if (
                    len(seen) >= settings.target_unique_documents
                    and total_bytes >= settings.target_utf8_bytes
                ):
                    targets_met = True
                    break
        selected.append((report, train_count, validation_count))
        if targets_met:
            break
    if not targets_met:
        raise RuntimeError(
            f"parallel cache prefix is too small: documents={len(seen)} "
            f"bytes={total_bytes}"
        )
    if train_ids & validation_ids:
        raise RuntimeError("parallel merge produced train/validation overlap")
    within_file_duplicates = sum(
        int(report["counts"]["duplicates_within_file"])
        for report, _train, _validation in selected
    )
    raw_documents += within_file_duplicates
    duplicates += within_file_duplicates

    language_dir = output_root / "eng_Latn"
    if language_dir.exists():
        raise RuntimeError(f"parallel merge output already exists: {language_dir}")
    language_dir.mkdir(parents=True)
    temporary: dict[str, Path] = {}
    handles: dict[str, Any] = {}
    try:
        for name in base.OUTPUT_NAMES:
            fd, raw = tempfile.mkstemp(prefix=f".{name}.", dir=language_dir)
            temporary[name] = Path(raw)
            handles[name] = os.fdopen(fd, "w", encoding="utf-8")
        calibration_count = 0
        for report, train_count, validation_count in selected:
            root = fragment_root / "files" / f"file_{int(report['source_index']):05d}"
            copied_train = _copy_lines(root / "train.jsonl", handles["train.jsonl"], train_count)
            copied_validation = _copy_lines(
                root / "validation_reserve.jsonl",
                handles["validation_reserve.jsonl"],
                validation_count,
            )
            if copied_train != train_count or copied_validation != validation_count:
                raise RuntimeError("fragment truncation count mismatch")
            if calibration_count < settings.calibration_documents:
                with (root / "train.jsonl").open("r", encoding="utf-8") as source_handle:
                    for line in source_handle:
                        if calibration_count >= settings.calibration_documents:
                            break
                        row = json.loads(line)
                        row["artifact_role"] = "calibration"
                        _write_line(handles["calibration.jsonl"], row)
                        calibration_count += 1
        for handle in handles.values():
            handle.flush(); os.fsync(handle.fileno()); handle.close()
        handles.clear()
        for name in base.OUTPUT_NAMES:
            os.replace(temporary[name], language_dir / name)
    finally:
        for handle in handles.values():
            handle.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)

    source_lock = _load(source_lock_path)
    source = base._source_identity(source_lock, "eng_Latn")
    file_entries = []
    for entry in entries:
        path = str(entry["path"])
        local = cache_root / path
        file_entries.append(
            {
                "path": path,
                "order_sha256": hashlib.sha256(f"{seed}\0eng_Latn\0{path}".encode()).hexdigest(),
                "content_sha256": str(entry["lfs_sha256"]),
                "bytes": local.stat().st_size,
            }
        )
    paths_manifest: dict[str, Any] = {
        "schema_version": base.SCHEMA_VERSION,
        "tool_id": base.TOOL_ID,
        "mode": "local_fixture",
        "dataset": source.dataset,
        "revision": source.revision,
        "subset": source.subset,
        "seed": seed,
        "data_folder": str(cache_root.resolve()),
        "files": file_entries,
        "ordered_paths_sha256": canonical_json_sha256([row["path"] for row in file_entries]),
    }
    paths_manifest["manifest_sha256"] = base._manifest_hash(paths_manifest)
    _atomic_json(language_dir / "paths_manifest.json", paths_manifest)
    base._atomic_write_text(
        language_dir / "paths.txt", "".join(f"{row['path']}\n" for row in file_entries)
    )
    settings_payload = asdict(settings)
    settings_payload["validation_candidate_fraction"] = validation_fraction
    inventory_sha = sha256_file(source_inventory_path)
    availability_policy = {
        "inventory_limited": False,
        "requested_settings": asdict(settings),
        "effective_settings": asdict(settings),
    }
    run_config = {
        "schema_version": base.SCHEMA_VERSION,
        "tool_id": base.TOOL_ID,
        "language": "eng_Latn",
        "source": asdict(source),
        "source_lock_sha256": sha256_file(source_lock_path),
        "rehydration_weights_sha256": sha256_file(rehydration_weights_path),
        "rehydration_policy_source": "english_fineweb_no_rehydration",
        "rehydration_thresholds_sha256": canonical_json_sha256({"1": 1}),
        "language_map_sha256": sha256_file(language_map_path),
        "source_inventory_sha256": inventory_sha,
        "paths_manifest_sha256": paths_manifest["manifest_sha256"],
        "settings": settings_payload,
        "validation_candidate_contract": {
            "membership_namespace": "validation-reserve-v1",
            "excluded_from_tokenizer_training": True,
            "unused_candidates_return_to_model_training_selection_pool": True,
            "preserved_train_rehydration_multiplier": True,
        },
        "availability_policy": availability_policy,
        "production_target": production_target,
        "parallel_acquisition": {
            "tool_id": MERGE_TOOL_ID,
            "world_size": world_size,
            "cache_plan_sha256": plan_sha,
            "ordered_merge": True,
            "cross_source_duplicates": duplicates,
        },
    }
    outputs = {
        name: {
            "path": name,
            "sha256": sha256_file(language_dir / name),
            "bytes": (language_dir / name).stat().st_size,
        }
        for name in base.OUTPUT_NAMES
    }
    report: dict[str, Any] = {
        "schema_version": base.SCHEMA_VERSION,
        "tool_id": base.TOOL_ID,
        "status": "complete",
        "acquisition_status": "targets_reached",
        "run_config": run_config,
        "run_config_sha256": canonical_json_sha256(run_config),
        "lineage": {
            "source_lock_path": str(source_lock_path.resolve()),
            "rehydration_weights_path": str(rehydration_weights_path.resolve()),
            "language_map_path": str(language_map_path.resolve()),
            "paths_manifest": "paths_manifest.json",
            "paths_manifest_sha256": paths_manifest["manifest_sha256"],
            "ordered_paths_sha256": paths_manifest["ordered_paths_sha256"],
            "read_source_file_sha256": {
                str(row["source_path"]): str(row["source_file_sha256"])
                for row, _train, _validation in selected
            },
        },
        "counts": {
            "unique_source_documents": len(seen),
            "raw_source_documents_read": raw_documents,
            "unique_source_utf8_bytes": total_bytes,
            "duplicate_source_documents_skipped": duplicates,
            "unique_train_documents": len(train_ids),
            "unique_validation_documents": len(validation_ids),
            "unique_validation_candidate_documents": len(validation_ids),
            "declared_train_documents_with_rehydration": len(train_ids),
            "emitted_validation_documents": len(validation_ids),
            "calibration_documents": calibration_count,
            "train_validation_overlap": 0,
        },
        "targets": {
            "requested": asdict(settings),
            "effective_after_inventory_cap": asdict(settings),
            "inventory_limited": False,
            "source_exhausted": False,
            "pinned_inventory_exhausted": False,
            "unique_document_shortfall": 0,
            "utf8_byte_shortfall": 0,
            "calibration_requirement": settings.calibration_documents,
        },
        "hashes": {
            "unique_source_ids_sha256": _canonical_id_hash(seen),
            "unique_train_ids_sha256": _canonical_id_hash(train_ids),
            "unique_validation_ids_sha256": _canonical_id_hash(validation_ids),
            "unique_validation_candidate_ids_sha256": _canonical_id_hash(validation_ids),
            "calibration_ids_sha256": _canonical_id_hash(calibration_ids),
        },
        "rehydration": {
            "enabled": False,
            "identity_override": False,
            "policy_source": "english_fineweb_no_rehydration",
            "thresholds_sha256": canonical_json_sha256({"1": 1}),
            "replica_count_histogram": {"1": len(train_ids)},
            "validation_candidate_train_replica_count_histogram": {"1": len(validation_ids)},
            "cluster_size_histogram": {"1": len(seen)},
        },
        "outputs": outputs,
    }
    report["report_payload_sha256"] = canonical_json_sha256(report)
    _atomic_json(language_dir / "bootstrap_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--cache-plan", type=Path, required=True)
    worker.add_argument("--cache-root", type=Path, required=True)
    worker.add_argument("--fragment-root", type=Path, required=True)
    worker.add_argument("--source-lock", type=Path, required=True)
    worker.add_argument("--rank", type=int, default=-1)
    worker.add_argument("--world-size", type=int, default=32)
    worker.add_argument("--seed", type=int, default=42)
    worker.add_argument("--validation-fraction", type=float, default=0.30)
    merge = sub.add_parser("merge")
    merge.add_argument("--cache-plan", type=Path, required=True)
    merge.add_argument("--cache-root", type=Path, required=True)
    merge.add_argument("--fragment-root", type=Path, required=True)
    merge.add_argument("--output-root", type=Path, required=True)
    merge.add_argument("--source-lock", type=Path, required=True)
    merge.add_argument("--rehydration-weights", type=Path, required=True)
    merge.add_argument("--source-inventory", type=Path, required=True)
    merge.add_argument("--language-map", type=Path, required=True)
    merge.add_argument("--sampling-plan", type=Path, required=True)
    merge.add_argument("--world-size", type=int, default=32)
    merge.add_argument("--seed", type=int, default=42)
    merge.add_argument("--validation-fraction", type=float, default=0.30)
    merge.add_argument("--safety-margin", type=float, default=1.10)
    args = parser.parse_args()
    if args.command == "worker":
        rank = args.rank if args.rank >= 0 else int(os.environ["SLURM_ARRAY_TASK_ID"])
        result = run_worker(
            cache_plan_path=args.cache_plan,
            cache_root=args.cache_root,
            fragment_root=args.fragment_root,
            source_lock_path=args.source_lock,
            rank=rank,
            world_size=args.world_size,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
        )
    else:
        result = merge_fragments(
            cache_plan_path=args.cache_plan,
            cache_root=args.cache_root,
            fragment_root=args.fragment_root,
            output_root=args.output_root,
            source_lock_path=args.source_lock,
            rehydration_weights_path=args.rehydration_weights,
            source_inventory_path=args.source_inventory,
            language_map_path=args.language_map,
            sampling_plan_path=args.sampling_plan,
            world_size=args.world_size,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            safety_margin=args.safety_margin,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
