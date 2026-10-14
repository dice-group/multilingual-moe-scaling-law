"""Validation of sealed Stage 1 workflow manifests."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .common import (
    FIXED_PEAK_LEARNING_RATE,
    WORKFLOW_SCHEMA_VERSION,
    WorkflowError,
    _abs,
    _file_sha256,
    _load_object,
    _payload_sha256,
    _recovery_authorization,
    _rel_or_abs,
    _resource_gpu_count,
    _tracked_git_provenance_identity,
    _workflow_source_files,
)
from .contract_timing import _parse_slurm_duration
from .job_validation import _validate_dag, _validate_lr_contract
from .variant import active_variant


def _manifest_without_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate = dict(payload)
    candidate.pop("manifest_sha256", None)
    return candidate


def validate_manifest(
    payload: Mapping[str, Any], *, verify_inputs: bool = True
) -> dict[str, Any]:
    variant = active_variant()
    schema_version = payload.get("schema_version")
    if schema_version not in {1, WORKFLOW_SCHEMA_VERSION}:
        raise WorkflowError("unexpected workflow manifest schema")
    expected = str(payload.get("manifest_sha256", ""))
    actual = _payload_sha256(_manifest_without_hash(payload))
    if expected != actual:
        raise WorkflowError(
            f"workflow manifest hash mismatch: expected {expected}, got {actual}"
        )
    recorded_variant = payload.get("workflow_variant")
    if recorded_variant is not None and recorded_variant != variant.manifest_descriptor():
        raise WorkflowError(
            "workflow manifest variant differs from the active explicit variant"
        )
    recorded_module = payload.get("workflow_module")
    if recorded_module is not None and recorded_module != variant.worker_module:
        raise WorkflowError("workflow manifest worker module differs from its variant")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise WorkflowError("workflow manifest has no jobs list")
    authorization = _recovery_authorization(payload)
    if schema_version == 1:
        try:
            fixed_lr = float(payload["fixed_peak_learning_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError("historical manifest lacks its fixed LR") from exc
        if not math.isclose(
            fixed_lr, FIXED_PEAK_LEARNING_RATE, rel_tol=0.0, abs_tol=0.0
        ):
            raise WorkflowError("historical manifest fixed LR drifted")
        prescription = None
    else:
        prescription = payload.get("learning_rate_prescription")
        if not isinstance(prescription, Mapping):
            raise WorkflowError("workflow manifest lacks its LR prescription")
    _validate_lr_contract(jobs, prescription=prescription)
    _validate_dag(jobs)
    walltime_contract = payload.get("walltime_contract")
    if walltime_contract is not None:
        if not isinstance(walltime_contract, Mapping):
            raise WorkflowError("manifest walltime contract must be an object")
        if walltime_contract.get("mode") == "ordinary_queue_forecast":
            _validate_ordinary_queue_walltimes(payload, jobs, walltime_contract)
            walltime_contract = None
    if walltime_contract is not None:
        _validate_reservation_walltimes(jobs, walltime_contract)
    source_drift_advisories: list[dict[str, str]] = []
    if verify_inputs:
        source_drift_advisories = _validate_fixed_inputs(payload, authorization)
    if any("--exclusive" in str(row) for row in jobs):
        raise WorkflowError("workflow manifest contains forbidden --exclusive")
    return {
        "status": "ok",
        "manifest_sha256": expected,
        "jobs": len(jobs),
        "role_counts": payload.get("role_counts", {}),
        "go_no_go": payload.get("gpu_hour_estimate", {}).get("go_no_go"),
        "source_drift_policy": "record_and_continue_never_blocks_execution",
        "source_drift_advisories": source_drift_advisories,
    }


def _validate_ordinary_queue_walltimes(
    payload: Mapping[str, Any],
    jobs: list[dict[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    multiplier = float(contract["predicted_time_multiplier"])
    additive = float(contract["additive_safety_hours"])
    minimum_hours = float(contract["minimum_job_hours"])
    maximum_hours = float(contract["maximum_job_hours"])
    quantum = int(contract["walltime_rounding_minutes"]) * 60
    assigned = 0
    requested_gpu_hours = 0.0
    runtime = _load_object(Path(str(payload["runtime_path"])))
    for job in jobs:
        if "predicted_wall_hours" not in job:
            continue
        assigned += 1
        predicted = float(job["predicted_wall_hours"])
        expected_hours = max(
            minimum_hours,
            min(maximum_hours, predicted * multiplier + additive),
        )
        expected_seconds = math.ceil(expected_hours * 3600 / quantum) * quantum
        requested = _parse_slurm_duration(job.get("slurm_time_limit"))
        minimum = _parse_slurm_duration(job.get("slurm_time_min"))
        if requested != expected_seconds or minimum != requested:
            raise WorkflowError(
                f"{job['job_id']} ordinary-queue walltime contract drifted"
            )
        requested_gpu_hours += requested / 3600.0 * _resource_gpu_count(
            runtime, str(job["resource"])
        )
    if assigned != int(contract.get("assigned_gpu_jobs", -1)):
        raise WorkflowError("ordinary-queue assigned-job count mismatch")
    if not math.isclose(
        requested_gpu_hours,
        float(contract["maximum_initial_requested_gpu_hours"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise WorkflowError("ordinary-queue requested GPU-hour envelope drifted")


def _validate_reservation_walltimes(
    jobs: list[dict[str, Any]], contract: Mapping[str, Any]
) -> None:
    margin = float(contract["desired_walltime_safety_margin_hours"])
    training_minimum_margin = float(
        contract["training_time_min_safety_margin_hours"]
    )
    evaluation_minimum_margin = float(
        contract["evaluation_time_min_safety_margin_hours"]
    )
    maximum = _parse_slurm_duration(contract["partition_max_time"])
    capped = set(contract.get("partition_capped_jobs", []))
    assigned = 0
    for job in jobs:
        if "predicted_wall_hours" not in job:
            continue
        assigned += 1
        job_id = str(job["job_id"])
        requested = _parse_slurm_duration(job.get("slurm_time_limit"))
        minimum = _parse_slurm_duration(job.get("slurm_time_min"))
        predicted = float(job["predicted_wall_hours"])
        minimum_margin = (
            evaluation_minimum_margin
            if job.get("action") == "per_language_eval"
            else training_minimum_margin
        )
        if requested > maximum:
            raise WorkflowError(f"{job_id} exceeds the partition walltime maximum")
        if minimum > requested:
            raise WorkflowError(f"{job_id} time-min exceeds its desired time limit")
        if minimum + 1e-9 < (predicted + minimum_margin) * 3600:
            raise WorkflowError(
                f"{job_id} time-min lacks its {minimum_margin:g}-hour safety margin"
            )
        required = (predicted + margin) * 3600
        if job_id not in capped and requested + 1e-9 < required:
            raise WorkflowError(
                f"{job_id} lacks the frozen {margin:g}-hour walltime safety margin"
            )
        if job_id in capped and requested != maximum:
            raise WorkflowError(
                f"{job_id} is marked capped but does not use the maximum"
            )
    if assigned != int(contract.get("assigned_gpu_jobs", -1)):
        raise WorkflowError("walltime contract assigned-job count mismatch")


def _validate_fixed_inputs(
    payload: Mapping[str, Any], authorization: Mapping[str, Any] | None
) -> list[dict[str, str]]:
    fixed_inputs = payload.get("fixed_input_sha256", {})
    if not isinstance(fixed_inputs, Mapping):
        raise WorkflowError("workflow manifest fixed inputs must be an object")
    required_sources = {_rel_or_abs(path) for path in _workflow_source_files()}
    missing_sources = required_sources - set(fixed_inputs)
    if missing_sources:
        raise WorkflowError(
            "workflow manifest predates the complete modular source bundle: "
            + ", ".join(sorted(missing_sources))
        )
    authorized_changes = (
        dict(authorization["allowed_fixed_input_changes"])
        if authorization is not None
        else {}
    )
    observed_changes: set[str] = set()
    source_drift_advisories: list[dict[str, str]] = []
    for raw_path, digest in fixed_inputs.items():
        path = _abs(str(raw_path))
        if not path.is_file():
            raise WorkflowError(f"frozen workflow input is missing: {path}")
        actual_digest = _file_sha256(path)
        if actual_digest != digest:
            grant = authorized_changes.get(str(raw_path))
            authorized = (
                isinstance(grant, Mapping)
                and grant.get("launch_sha256") == digest
                and grant.get("execution_sha256") == actual_digest
            )
            is_source_code = (
                str(raw_path) in required_sources
                or path.suffix in {".py", ".pyi", ".sh"}
            )
            if authorized:
                observed_changes.add(str(raw_path))
            elif not is_source_code:
                raise WorkflowError(
                    f"frozen workflow input changed: {path}; expected {digest}, "
                    f"got {actual_digest}"
                )
            if is_source_code:
                source_drift_advisories.append(
                    {
                        "classification": "source_code_drift",
                        "path": str(raw_path),
                        "launch_sha256": str(digest),
                        "execution_sha256": actual_digest,
                        "authorization": (
                            "fixed_input_recovery" if authorized else "advisory_only"
                        ),
                    }
                )
    if authorization is not None and observed_changes != set(authorized_changes):
        raise WorkflowError(
            "recovery fixed-input changes differ from the authorization: "
            f"observed={sorted(observed_changes)} "
            f"authorized={sorted(authorized_changes)}"
        )
    if "git_provenance" in payload:
        expected_provenance = payload.get("git_provenance")
        if not isinstance(expected_provenance, Mapping):
            raise WorkflowError("workflow manifest git provenance is malformed")
        _tracked_git_provenance_identity(expected_provenance)
    return sorted(source_drift_advisories, key=lambda row: row["path"])
