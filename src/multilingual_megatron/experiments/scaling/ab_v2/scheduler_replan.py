"""Archive only explicitly safe terminal attempts before a source-level replan."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping

from .common import (
    DATA_ARTIFACT_SEAL_NAME,
    GPU_ESTIMATE_NAME,
    MANIFEST_NAME,
    RETRYABLE_TERMINAL_STATES,
    STATE_NAME,
    WorkflowError,
    _atomic_json,
    _load_object,
)
from .scheduler_accounting import _query_states
from .scheduler_state import _normalized_slurm_state
from .variant import active_variant


def _archive_terminal_pre_gpu_attempt_for_replan(
    *, output_root: Path, manifest_path: Path, state_path: Path
) -> Path:
    """Preserve a terminal pre-GPU failure before replacing its manifest.

    A manifest normally remains immutable once it has a submission state.  The
    narrow recovery exception covers only the scalar CPU gates before any GPU
    allocation.  A source-level repair changes the manifest's frozen code
    hash, so preserve every diagnostic artifact under
    ``superseded_pre_gpu_attempts`` and rerun those cheap gates from the new
    manifest.  A passing sealed audit may be superseded only when its sole
    dependent reader smoke has terminally failed; the corpus remains sealed
    and no H100 job has been submitted.  Every later stage remains immutable.
    """

    state = _load_object(state_path)
    jobs = state.get("jobs")
    allowed_jobs = {
        "storage_preflight",
        "data_audit_gate",
        "data_reuse_gate",
        "data_megatron_reader_smoke",
    }
    if not isinstance(jobs, dict) or not jobs or not set(jobs) <= allowed_jobs:
        raise WorkflowError(
            "cannot force-replan after submitted jobs unless every job is a "
            "terminal pre-GPU data gate"
        )
    slurm_job_ids = {
        job_id: str(raw.get("slurm_job_id", ""))
        for job_id, raw in jobs.items()
        if isinstance(raw, dict)
    }
    if not all(slurm_job_ids.values()):
        raise WorkflowError("submitted pre-GPU data gate lacks a Slurm job ID")
    observed = _query_states(slurm_job_ids.values())
    terminal_states = RETRYABLE_TERMINAL_STATES | {"COMPLETED"}
    observed_by_job = {
        job_id: observed.get(slurm_job_id, {})
        for job_id, slurm_job_id in slurm_job_ids.items()
    }
    if any(
        _normalized_slurm_state(row.get("state")) not in terminal_states
        for row in observed_by_job.values()
    ):
        raise WorkflowError(
            "cannot force-replan while a pre-GPU data gate is not terminal"
        )
    receipts_dir = output_root / "control" / "receipts"
    receipt_paths = list(receipts_dir.glob("*.json")) if receipts_dir.exists() else []
    receipt_names = {path.name for path in receipt_paths}
    allowed_receipts = {
        "storage_preflight.json",
        "data_audit_gate.json",
        "data_reuse_gate.json",
    }
    if not receipt_names <= allowed_receipts:
        raise WorkflowError(
            "cannot force-replan after an unexpected pre-GPU receipt exists; preserve the frozen workflow"
        )
    seal_path = output_root / "control" / DATA_ARTIFACT_SEAL_NAME
    sealed_reader_recovery = seal_path.exists() or "data_audit_gate.json" in receipt_names
    reused_reader_recovery = "data_reuse_gate.json" in receipt_names
    if sealed_reader_recovery:
        audit_state = _normalized_slurm_state(
            observed_by_job.get("data_audit_gate", {}).get("state")
        )
        reader_state = _normalized_slurm_state(
            observed_by_job.get("data_megatron_reader_smoke", {}).get("state")
        )
        if (
            not seal_path.is_file()
            or "data_audit_gate.json" not in receipt_names
            or audit_state != "COMPLETED"
            or reader_state not in RETRYABLE_TERMINAL_STATES
        ):
            raise WorkflowError(
                "cannot force-replan a sealed corpus unless a completed audit and "
                "its failed reader smoke are the only terminal pre-GPU evidence"
            )
    if reused_reader_recovery:
        reuse_state = _normalized_slurm_state(
            observed_by_job.get("data_reuse_gate", {}).get("state")
        )
        reader_state = _normalized_slurm_state(
            observed_by_job.get("data_megatron_reader_smoke", {}).get("state")
        )
        if (
            reuse_state != "COMPLETED"
            or reader_state not in RETRYABLE_TERMINAL_STATES
        ):
            raise WorkflowError(
                "cannot force-replan a reused sealed corpus unless its completed "
                "reuse gate and failed reader smoke are the only terminal pre-GPU evidence"
            )
    archive = (
        output_root
        / "control"
        / "superseded_pre_gpu_attempts"
        / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    archive.mkdir(parents=True, exist_ok=False)
    movable = (
        (manifest_path, archive / MANIFEST_NAME),
        (state_path, archive / STATE_NAME),
        (output_root / "control" / GPU_ESTIMATE_NAME, archive / GPU_ESTIMATE_NAME),
        (output_root / "control" / "storage_preflight.json", archive / "storage_preflight.json"),
        (output_root / "control" / "data_audit.json", archive / "data_audit.json"),
        (output_root / "control" / "data_reuse_evidence.json", archive / "data_reuse_evidence.json"),
        (output_root / "control" / "megatron_data_smoke.json", archive / "megatron_data_smoke.json"),
        (seal_path, archive / DATA_ARTIFACT_SEAL_NAME),
        (receipts_dir, archive / "receipts"),
        (output_root / "control" / "sbatch", archive / "sbatch"),
        (output_root / "control" / "submitted_sbatch", archive / "submitted_sbatch"),
        (output_root / "logs" / "storage_preflight.controller.log", archive / "storage_preflight.controller.log"),
        (output_root / "logs" / "data_audit_gate.controller.log", archive / "data_audit_gate.controller.log"),
        (output_root / "logs" / "data_reuse_gate.controller.log", archive / "data_reuse_gate.controller.log"),
        (output_root / "logs" / "data_megatron_reader_smoke.controller.log", archive / "data_megatron_reader_smoke.controller.log"),
    )
    moved: list[str] = []
    for source, target in movable:
        if source.exists() or source.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            moved.append(str(target.relative_to(output_root)))
    _atomic_json(
        archive / "supersession.json",
        {
            "schema_version": 1,
            "reason": (
                "terminal_reader_smoke_failure_after_sealed_audit_before_any_gpu"
                if sealed_reader_recovery
                else (
                    "terminal_reader_smoke_failure_after_reused_sealed_corpus_gate"
                    if reused_reader_recovery
                    else "terminal_pre_gpu_data_gate_failure_before_data_seal"
                )
            ),
            "prior_manifest_sha256": state.get("manifest_sha256"),
            "pre_gpu_jobs": {
                job_id: {
                    "slurm_job_id": slurm_job_ids[job_id],
                    "state": row.get("state"),
                    "exit_code": row.get("exit_code"),
                }
                for job_id, row in observed_by_job.items()
            },
            "moved_artifacts": moved,
        },
    )
    return archive
def _archive_terminal_sentinel_startup_attempt_for_replan(
    *, output_root: Path, manifest_path: Path, state_path: Path
) -> Path:
    """Archive a source-level sentinel startup failure before any training.

    This is deliberately narrower than ordinary retry.  It permits a new
    code-bound manifest only when reusable-data pilot receipts passed, every
    submitted GPU sentinel failed before producing a training result, and all
    downstream sentinel jobs were cancelled without starting.  It is the safe
    recovery path for an orchestration bug such as a Python ``NameError``;
    training, checkpoint, evaluation, and production evidence remain
    immutable once materialized.
    """

    prior_manifest = _load_object(manifest_path)
    planned_jobs = {
        str(job["job_id"]): job
        for job in prior_manifest.get("jobs", [])
        if isinstance(job, Mapping) and isinstance(job.get("job_id"), str)
    }
    state = _load_object(state_path)
    state_jobs = state.get("jobs")
    if not isinstance(state_jobs, dict) or not state_jobs:
        raise WorkflowError("sentinel startup recovery requires a non-empty workflow state")
    if not set(state_jobs) <= set(planned_jobs):
        raise WorkflowError("workflow state contains a job absent from the frozen manifest")

    pilot_actions = {
        "storage_preflight",
        "reuse_sealed_data_verify",
        "megatron_data_smoke",
        "data_audit",
    }
    sentinel_actions = {
        "training_sentinel",
        "per_language_eval",
        "sentinel_gate",
        "checkpoint_prune",
    } | set(active_variant().sentinel_startup_recovery_actions)
    if any(
        str(planned_jobs[job_id].get("action", "")) not in pilot_actions | sentinel_actions
        for job_id in state_jobs
    ):
        raise WorkflowError(
            "cannot force-replan after a sentinel startup failure when a later workflow stage was submitted"
        )

    slurm_job_ids = {
        job_id: str(raw.get("slurm_job_id", ""))
        for job_id, raw in state_jobs.items()
        if isinstance(raw, Mapping)
    }
    if len(slurm_job_ids) != len(state_jobs) or not all(slurm_job_ids.values()):
        raise WorkflowError("sentinel startup recovery requires Slurm IDs for every submitted job")
    observed = _query_states(slurm_job_ids.values())
    observed_by_job = {
        job_id: observed.get(slurm_job_id, {})
        for job_id, slurm_job_id in slurm_job_ids.items()
    }
    terminal_states = RETRYABLE_TERMINAL_STATES | {"COMPLETED"}
    if any(
        _normalized_slurm_state(row.get("state")) not in terminal_states
        for row in observed_by_job.values()
    ):
        raise WorkflowError(
            "cannot force-replan while a sentinel startup or dependent job is not terminal"
        )
    for job_id in state_jobs:
        if str(planned_jobs[job_id].get("action", "")) in pilot_actions and _normalized_slurm_state(
            observed_by_job[job_id].get("state")
        ) != "COMPLETED":
            raise WorkflowError(
                "sentinel startup recovery requires completed reusable-data pilot gates"
            )

    sentinels = [
        job_id
        for job_id in state_jobs
        if str(planned_jobs[job_id].get("action", "")) == "training_sentinel"
    ]
    if not sentinels:
        raise WorkflowError("sentinel startup recovery requires at least one submitted sentinel")
    for job_id in sentinels:
        if _normalized_slurm_state(observed_by_job[job_id].get("state")) != "FAILED":
            raise WorkflowError(
                "sentinel startup recovery requires every submitted GPU sentinel to fail before training"
            )
        job_root = output_root / "jobs" / job_id
        if (job_root / "result.json").is_file() or any(
            job_root.rglob("training_run_manifest.json")
        ):
            raise WorkflowError(
                "cannot force-replan after a sentinel produced a training result or manifest"
            )
    for job_id in state_jobs:
        action = str(planned_jobs[job_id].get("action", ""))
        if action in sentinel_actions - {"training_sentinel"} and _normalized_slurm_state(
            observed_by_job[job_id].get("state")
        ) != "CANCELLED":
            raise WorkflowError(
                "sentinel startup recovery requires unstarted dependent sentinel jobs to be cancelled"
            )

    receipts_dir = output_root / "control" / "receipts"
    receipt_paths = list(receipts_dir.glob("*.json")) if receipts_dir.exists() else []
    allowed_receipts = {
        "storage_preflight.json",
        "data_audit_gate.json",
        "data_reuse_gate.json",
        "data_megatron_reader_smoke.json",
    }
    if any(path.name not in allowed_receipts for path in receipt_paths):
        raise WorkflowError(
            "cannot force-replan after a sentinel startup failure with non-pilot receipts"
        )

    archive = (
        output_root
        / "control"
        / "superseded_sentinel_startup_attempts"
        / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    archive.mkdir(parents=True, exist_ok=False)
    movable = (
        (manifest_path, archive / MANIFEST_NAME),
        (state_path, archive / STATE_NAME),
        (output_root / "control" / GPU_ESTIMATE_NAME, archive / GPU_ESTIMATE_NAME),
        (output_root / "control" / "storage_preflight.json", archive / "storage_preflight.json"),
        (output_root / "control" / "data_reuse_evidence.json", archive / "data_reuse_evidence.json"),
        (output_root / "control" / "megatron_data_smoke.json", archive / "megatron_data_smoke.json"),
        (receipts_dir, archive / "receipts"),
        (output_root / "control" / "sbatch", archive / "sbatch"),
        (output_root / "control" / "submitted_sbatch", archive / "submitted_sbatch"),
        (output_root / "logs", archive / "logs"),
        (output_root / "jobs", archive / "jobs"),
        (output_root / "runtime", archive / "runtime"),
    )
    moved: list[str] = []
    for source, target in movable:
        if source.exists() or source.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            moved.append(str(target.relative_to(output_root)))
    _atomic_json(
        archive / "supersession.json",
        {
            "schema_version": 1,
            "reason": "terminal_sentinel_startup_failure_before_any_training_artifact",
            "prior_manifest_sha256": state.get("manifest_sha256"),
            "sentinel_jobs": {
                job_id: {
                    "slurm_job_id": slurm_job_ids[job_id],
                    "state": observed_by_job[job_id].get("state"),
                    "exit_code": observed_by_job[job_id].get("exit_code"),
                }
                for job_id in sorted(sentinels)
            },
            "moved_artifacts": moved,
        },
    )
    return archive
