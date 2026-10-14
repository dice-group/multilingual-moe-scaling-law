#!/usr/bin/env python3
"""Run the audited FineWeb finalizer over balanced language shards."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
    verify_plan_hash,
)
from multilingual_megatron.pipeline.stages.sampling.tools import (
    finalize_fineweb_v2_corpus as finalizer,
)


SCHEMA_VERSION = 1
PLAN_TOOL_ID = "fineweb_ab_v2_parallel_finalizer_plan_v1"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_payload(payload: Mapping[str, Any], key: str) -> str:
    candidate = dict(payload)
    expected = candidate.pop(key, None)
    if expected != canonical_json_sha256(candidate):
        raise ValueError(f"payload hash mismatch: {key}")
    return str(expected)


def _resolve_entry_path(manifest_path: Path, raw: Any) -> Path:
    path = Path(str(raw))
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _subset_bootstrap(
    original: Mapping[str, Any], manifest_path: Path, languages: list[str]
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for language in languages:
        row = json.loads(json.dumps(original["languages"][language]))
        for split in ("train", "validation_reserve"):
            row[split]["path"] = str(
                _resolve_entry_path(manifest_path, row[split]["path"])
            )
        acquisition = row.get("acquisition", {})
        if acquisition.get("report_path"):
            acquisition["report_path"] = str(
                _resolve_entry_path(manifest_path, acquisition["report_path"])
            )
        selected[language] = row
    payload = {
        "schema_version": original["schema_version"],
        "provenance": original.get("provenance", {}),
        "languages": selected,
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def _subset_plan(original: Mapping[str, Any], languages: list[str]) -> dict[str, Any]:
    chosen = set(languages)
    payload = json.loads(json.dumps(original))
    payload.pop("plan_sha256", None)
    payload["physical_pool"] = {
        language: original["physical_pool"][language] for language in languages
    }
    tiers: dict[str, Any] = {}
    for tier_name, raw in original["tiers"].items():
        tier = json.loads(json.dumps(raw))
        tier_languages = [str(value) for value in raw.get("languages", []) if str(value) in chosen]
        tier["languages"] = tier_languages
        for field in (
            "probabilities",
            "target_exposure_tokens",
            "availability_tokens",
        ):
            if isinstance(raw.get(field), dict):
                tier[field] = {
                    language: raw[field][language]
                    for language in tier_languages
                    if language in raw[field]
                }
        tiers[tier_name] = tier
    payload["tiers"] = tiers
    payload["plan_sha256"] = canonical_json_sha256(payload)
    return payload


def prepare(
    *,
    bootstrap_manifest_path: Path,
    sampling_plan_path: Path,
    work_root: Path,
    workers: int,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    bootstrap_manifest_path = bootstrap_manifest_path.resolve()
    sampling_plan_path = sampling_plan_path.resolve()
    bootstrap = _load(bootstrap_manifest_path)
    sampling_plan = _load(sampling_plan_path)
    if not verify_plan_hash(sampling_plan):
        raise ValueError("sampling plan hash mismatch")
    languages = sorted(map(str, sampling_plan["physical_pool"]))
    if set(languages) != set(map(str, bootstrap["languages"])):
        raise ValueError("bootstrap and sampling-plan languages differ")
    weighted: list[tuple[int, str]] = []
    for language in languages:
        row = bootstrap["languages"][language]
        size = sum(
            _resolve_entry_path(bootstrap_manifest_path, row[split]["path"]).stat().st_size
            for split in ("train", "validation_reserve")
        )
        weighted.append((size, language))
    assignments: list[list[str]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for size, language in sorted(weighted, reverse=True):
        rank = min(range(workers), key=lambda value: (loads[value], value))
        assignments[rank].append(language)
        loads[rank] += size
    if any(not values for values in assignments):
        raise ValueError("worker count exceeds language count")
    for rank, rank_languages in enumerate(assignments):
        rank_root = work_root / "inputs" / f"rank_{rank:03d}"
        _write(
            rank_root / "bootstrap_manifest.json",
            _subset_bootstrap(bootstrap, bootstrap_manifest_path, sorted(rank_languages)),
        )
        _write(
            rank_root / "sampling_plan.json",
            _subset_plan(sampling_plan, sorted(rank_languages)),
        )
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_id": PLAN_TOOL_ID,
        "workers": workers,
        "languages": languages,
        "assignments": {str(rank): sorted(values) for rank, values in enumerate(assignments)},
        "estimated_input_bytes_by_rank": {str(rank): loads[rank] for rank in range(workers)},
        "inputs": {
            "bootstrap_manifest": {
                "path": str(bootstrap_manifest_path),
                "sha256": sha256_file(bootstrap_manifest_path),
            },
            "sampling_plan": {
                "path": str(sampling_plan_path),
                "file_sha256": sha256_file(sampling_plan_path),
                "payload_sha256": sampling_plan["plan_sha256"],
            },
        },
    }
    plan["plan_sha256"] = canonical_json_sha256(plan)
    _write(work_root / "parallel_finalizer_plan.json", plan)
    return plan


def run_worker(
    *,
    work_root: Path,
    tokenizer_model_path: Path,
    rank: int,
    workers: int,
    tokenization_workers: int = 1,
    index_work_root: Path | None = None,
) -> dict[str, Any]:
    plan = _load(work_root / "parallel_finalizer_plan.json")
    plan_sha = _verify_payload(plan, "plan_sha256")
    if int(plan["workers"]) != workers or not 0 <= rank < workers:
        raise ValueError("parallel finalizer rank/world mismatch")
    rank_root = work_root / "inputs" / f"rank_{rank:03d}"
    output = work_root / "ranks" / f"rank_{rank:03d}" / "release"
    manifest = finalizer.finalize_corpus(
        bootstrap_manifest_path=rank_root / "bootstrap_manifest.json",
        tokenizer_model_path=tokenizer_model_path,
        sampling_plan_path=rank_root / "sampling_plan.json",
        output_dir=output,
        max_train_shortfall_fraction=0.0,
        validation_target_fraction=0.2,
        tokenization_workers=tokenization_workers,
        index_work_root=index_work_root,
    )
    if not manifest.get("ready"):
        raise RuntimeError(f"rank {rank} finalizer is not ready")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "tool_id": PLAN_TOOL_ID,
        "status": "complete",
        "plan_sha256": plan_sha,
        "rank": rank,
        "workers": workers,
        "languages": plan["assignments"][str(rank)],
        "corpus_manifest_sha256": sha256_file(output / "corpus_manifest.json"),
        "hash_manifest_sha256": sha256_file(output / "hash_manifest.json"),
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    _write(work_root / "receipts" / f"rank_{rank:03d}.json", receipt)
    return receipt


def _hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)


def _verify_rank_release(
    *, release: Path, receipt: Mapping[str, Any], rank: int
) -> dict[str, Any]:
    manifest_path = release / "corpus_manifest.json"
    hash_manifest_path = release / "hash_manifest.json"
    if sha256_file(manifest_path) != receipt["corpus_manifest_sha256"]:
        raise ValueError(f"rank {rank} corpus manifest changed")
    if sha256_file(hash_manifest_path) != receipt["hash_manifest_sha256"]:
        raise ValueError(f"rank {rank} hash manifest changed")
    rank_manifest = _load(manifest_path)
    _verify_payload(rank_manifest, "manifest_sha256")
    hash_manifest = _load(hash_manifest_path)
    _verify_payload(hash_manifest, "hash_manifest_payload_sha256")
    if (
        hash_manifest.get("corpus_manifest_payload_sha256")
        != rank_manifest["manifest_sha256"]
    ):
        raise ValueError(f"rank {rank} corpus/hash-manifest payload mismatch")
    for relative, expected in hash_manifest.get("files", {}).items():
        candidate = (release / str(relative)).resolve()
        try:
            candidate.relative_to(release.resolve())
        except ValueError as error:
            raise ValueError(f"rank {rank} artifact escapes release: {relative}") from error
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError(f"rank {rank} release artifact changed: {relative}")
    return rank_manifest


def merge(
    *, work_root: Path, output_dir: Path, tokenizer_model_path: Path
) -> dict[str, Any]:
    plan = _load(work_root / "parallel_finalizer_plan.json")
    plan_sha = _verify_payload(plan, "plan_sha256")
    workers = int(plan["workers"])
    rank_manifests: list[tuple[Path, dict[str, Any]]] = []
    observed_languages: set[str] = set()
    for rank in range(workers):
        receipt = _load(work_root / "receipts" / f"rank_{rank:03d}.json")
        _verify_payload(receipt, "receipt_sha256")
        if receipt.get("plan_sha256") != plan_sha:
            raise ValueError(f"rank {rank} receipt belongs to another plan")
        if int(receipt.get("rank", -1)) != rank or int(receipt.get("workers", -1)) != workers:
            raise ValueError(f"rank {rank} receipt rank/world mismatch")
        release = work_root / "ranks" / f"rank_{rank:03d}" / "release"
        rank_manifest = _verify_rank_release(
            release=release, receipt=receipt, rank=rank
        )
        rank_languages = set(map(str, rank_manifest["languages"]))
        if rank_languages != set(map(str, plan["assignments"][str(rank)])):
            raise ValueError(f"rank {rank} language assignment mismatch")
        if observed_languages & rank_languages:
            raise ValueError("parallel finalizer language overlap")
        observed_languages |= rank_languages
        rank_manifests.append((release, rank_manifest))
    if observed_languages != set(map(str, plan["languages"])):
        raise ValueError("parallel finalizer language coverage is incomplete")
    if output_dir.exists():
        raise ValueError(f"final output already exists: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    staging.mkdir(parents=True)
    try:
        languages: dict[str, Any] = {}
        validation_inputs: dict[str, Any] = {}
        for release, rank_manifest in rank_manifests:
            for language, payload in rank_manifest["languages"].items():
                languages[language] = payload
                for relative in (
                    payload["train"]["output"]["path"],
                    payload["validation"]["output"]["path"],
                    payload["train"]["unigram_counts"]["path"],
                ):
                    _hardlink(release / relative, staging / relative)
            prefix = _load(release / rank_manifest["validation_prefix_inputs"]["path"])
            validation_inputs.update(prefix["inputs_by_language"])
        prefix_path = staging / "validation_prefix_inputs.json"
        _write(
            prefix_path,
            {
                "schema_version": SCHEMA_VERSION,
                "tokenizer_model_sha256": sha256_file(tokenizer_model_path),
                "append_eod": True,
                "inputs_by_language": dict(sorted(validation_inputs.items())),
            },
        )
        original_bootstrap = Path(plan["inputs"]["bootstrap_manifest"]["path"])
        original_sampling = Path(plan["inputs"]["sampling_plan"]["path"])
        template = rank_manifests[0][1]
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "release_id": "fineweb_ab_v2_finalized",
            "release_status": "ready",
            "ready": True,
            "blocking_languages": {
                "needs_incremental_source_docs": [],
                "needs_incremental_validation_inventory": [],
                "needs_incremental_validation_docs": [],
            },
            "selection_contract": template["selection_contract"],
            "inputs": {
                "bootstrap_manifest": {
                    "path": str(original_bootstrap),
                    "sha256": sha256_file(original_bootstrap),
                    "provenance": _load(original_bootstrap).get("provenance", {}),
                },
                "sampling_plan": {
                    "path": str(original_sampling),
                    "file_sha256": sha256_file(original_sampling),
                    "payload_sha256": _load(original_sampling)["plan_sha256"],
                },
                "tokenizer_model": template["inputs"]["tokenizer_model"],
            },
            "validation_prefix_inputs": finalizer._relative_artifact(prefix_path, staging),
            "languages": dict(sorted(languages.items())),
            "parallel_finalization": {
                "tool_id": PLAN_TOOL_ID,
                "workers": workers,
                "plan_sha256": plan_sha,
                "language_overlap": 0,
                "language_count": len(languages),
            },
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        _write(staging / "corpus_manifest.json", manifest)
        files = {
            str(path.relative_to(staging)): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        hash_manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "algorithm": "sha256",
            "corpus_manifest_payload_sha256": manifest["manifest_sha256"],
            "files": files,
        }
        hash_manifest["hash_manifest_payload_sha256"] = canonical_json_sha256(hash_manifest)
        _write(staging / "hash_manifest.json", hash_manifest)
        staging.replace(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--bootstrap-manifest", type=Path, required=True)
    prep.add_argument("--sampling-plan", type=Path, required=True)
    prep.add_argument("--work-root", type=Path, required=True)
    prep.add_argument("--workers", type=int, default=32)
    worker = sub.add_parser("worker")
    worker.add_argument("--work-root", type=Path, required=True)
    worker.add_argument("--tokenizer-model", type=Path, required=True)
    worker.add_argument("--rank", type=int, default=-1)
    worker.add_argument("--workers", type=int, default=32)
    worker.add_argument("--tokenization-workers", type=int, default=1)
    worker.add_argument("--index-work-root", type=Path)
    combine = sub.add_parser("merge")
    combine.add_argument("--work-root", type=Path, required=True)
    combine.add_argument("--output-dir", type=Path, required=True)
    combine.add_argument("--tokenizer-model", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            bootstrap_manifest_path=args.bootstrap_manifest,
            sampling_plan_path=args.sampling_plan,
            work_root=args.work_root,
            workers=args.workers,
        )
    elif args.command == "worker":
        rank = args.rank if args.rank >= 0 else int(os.environ["SLURM_ARRAY_TASK_ID"])
        result = run_worker(
            work_root=args.work_root,
            tokenizer_model_path=args.tokenizer_model,
            rank=rank,
            workers=args.workers,
            tokenization_workers=args.tokenization_workers,
            index_work_root=args.index_work_root,
        )
    else:
        result = merge(
            work_root=args.work_root,
            output_dir=args.output_dir,
            tokenizer_model_path=args.tokenizer_model,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
