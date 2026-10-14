"""Checkpoint-retention gate for completed A/B-v2 trajectories."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

from .artifacts import _tree_usage, _verified_receipt_evidence
from .common import (
    Paths,
    WorkflowError,
    _atomic_json,
    _file_sha256,
    _job,
    _load_object,
)
from .gate_evaluation import _validate_full_evaluation_artifacts


def _run_checkpoint_prune(
    manifest: Mapping[str, Any], job: Mapping[str, Any], paths: Paths
) -> None:
    arguments = job.get("arguments", {})
    source_jobs = [str(value) for value in arguments.get("source_jobs", [])]
    retained_jobs = {str(value) for value in arguments.get("retained_jobs", [])}
    if not source_jobs or retained_jobs.intersection(source_jobs):
        raise WorkflowError("checkpoint prune source/retained job contract is invalid")

    evidence: dict[str, Any] = {}
    for evidence_job in [str(value) for value in arguments.get("evidence_jobs", [])]:
        evidence[evidence_job] = _verified_receipt_evidence(manifest, evidence_job)
        evidence_row = _job(manifest, evidence_job)
        if (
            evidence_row.get("action") == "per_language_eval"
            and evidence[evidence_job]["status"] == "complete"
        ):
            evidence[evidence_job]["evaluation_artifacts"] = (
                _validate_full_evaluation_artifacts(
                    paths.jobs / evidence_job / "evaluation_manifest.json"
                )
            )
    source_results: dict[str, Any] = {}
    roots: list[tuple[str, Path]] = []
    for source_job in source_jobs:
        evidence[source_job] = _verified_receipt_evidence(manifest, source_job)
        result_path = paths.jobs / source_job / "result.json"
        payload = _load_object(result_path)
        if payload.get("status") == "not_required":
            source_results[source_job] = {
                "path": str(result_path),
                "sha256": _file_sha256(result_path),
                "status": "not_required",
            }
            continue
        if payload.get("status") != "complete":
            raise WorkflowError(f"refusing to prune incomplete source job {source_job}")
        source_results[source_job] = {
            "path": str(result_path),
            "sha256": _file_sha256(result_path),
            "status": "complete",
        }
        source_root = (paths.jobs / source_job).resolve()
        for checkpoint_root in sorted(source_root.rglob("checkpoints")):
            resolved = checkpoint_root.resolve()
            try:
                resolved.relative_to(source_root)
            except ValueError as exc:
                raise WorkflowError(
                    f"checkpoint root escapes source job: {resolved}"
                ) from exc
            if checkpoint_root.is_symlink() or resolved.name != "checkpoints":
                raise WorkflowError(
                    f"unsafe checkpoint prune target: {checkpoint_root}"
                )
            if resolved.is_dir():
                roots.append((source_job, resolved))

    before_bytes = 0
    before_files = 0
    root_records: list[dict[str, Any]] = []
    for source_job, root in roots:
        size, files = _tree_usage(root)
        before_bytes += size
        before_files += files
        root_records.append(
            {
                "source_job": source_job,
                "path": str(root),
                "bytes_before": size,
                "files_before": files,
            }
        )
    for _source_job, root in roots:
        shutil.rmtree(root)
    after_bytes = sum(_tree_usage(root)[0] for _source_job, root in roots)
    if after_bytes != 0 or any(root.exists() for _source_job, root in roots):
        raise WorkflowError(
            "checkpoint pruning did not remove every declared checkpoint root"
        )
    report = {
        "schema_version": 1,
        "status": "complete",
        "job_id": job["job_id"],
        "retention_scope": arguments.get("retention_scope"),
        "source_results": source_results,
        "retained_jobs": sorted(retained_jobs),
        "verified_dependency_evidence": evidence,
        "checkpoint_roots": root_records,
        "files_removed": before_files,
        "bytes_before": before_bytes,
        "bytes_after": after_bytes,
        "bytes_freed": before_bytes - after_bytes,
        "preserved_artifacts": "configs, logs, result/evaluation manifests, metrics, and provenance",
    }
    _atomic_json(paths.control / "prune" / f"{job['job_id']}.json", report)
