"""Render, submit, reconcile, and collect the A/B-v2 Slurm DAG."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from multilingual_megatron.pipeline.executor import SlurmExecutor

from .artifacts import (
    _has_completed_job_evidence,
    _quarantine_job_receipts,
    _receipt_path,
    _retry_history_entry,
)
from .common import (
    DATA_ARTIFACT_SEAL_NAME,
    GPU_ESTIMATE_NAME,
    MANIFEST_NAME,
    Paths,
    RECOVERY_AUTHORIZATION_ENV,
    RECOVERY_AUTHORIZATION_SHA256_ENV,
    REPO_ROOT,
    REQUIRED_DATA_ENV_KEYS,
    RETRYABLE_TERMINAL_STATES,
    STATE_NAME,
    SUBMISSION_PHASES,
    WorkflowError,
    _array_indices,
    _atomic_json,
    _is_h100_resource,
    _load_object,
    _submission_phase,
    _submit_slot_count,
)
from .contract import (
    build_manifest,
    validate_manifest,
)
from .gate_evaluation import _validation_report_from_eval
from .scheduler_selection import _ordered_submission_candidates
from .variant import active_policy, active_variant


def _resource_executor(
    runtime: Mapping[str, Any],
    resource_name: str,
    *,
    time_limit: str | None = None,
    time_min: str | None = None,
) -> SlurmExecutor:
    raw = runtime["resources"][resource_name]
    extra_sbatch_flags = [
        str(value) for value in raw.get("extra_sbatch_flags", [])
    ]
    if time_min is not None:
        extra_sbatch_flags.append(f"--time-min={time_min}")
    return SlurmExecutor(
        partition=str(raw["partition"]),
        gres=str(raw["gres"]),
        mem=str(raw["mem"]),
        time_limit=str(time_limit or raw["time"]),
        cpus_per_task=int(raw["cpus_per_task"]),
        module_init=str(raw["module_init"]),
        modules=[str(value) for value in raw.get("modules", [])],
        conda_base=str(raw.get("conda_base", "none")),
        conda_env=str(raw.get("conda_env", "none")),
        poll_interval=30,
        extra_sbatch_flags=extra_sbatch_flags,
    )


def _runner_command(
    manifest_path: Path, job_id: str, python_executable: str = "python"
) -> list[str]:
    workflow_module = "multilingual_megatron.experiments.scaling.ab_v2.workflow"
    if manifest_path.is_file():
        payload = _load_object(manifest_path)
        requested = payload.get("workflow_module")
        if requested is not None:
            if not isinstance(requested, str) or not requested.strip():
                raise WorkflowError("manifest workflow_module must be a non-empty string")
            workflow_module = requested
    return [
        python_executable,
        "-m",
        workflow_module,
        "execute-job",
        "--manifest",
        str(manifest_path.resolve()),
        "--job-id",
        job_id,
    ]


def _job_env(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, str]:
    policy = active_policy()
    env = {
        # The CPU reader smoke imports Megatron's real IndexedDataset class.
        # Bridge training adds this path inside its runner, but the standalone
        # smoke has to expose the vendored package before it starts Python.
        "PYTHONPATH": (
            f"{REPO_ROOT / 'external' / 'Megatron-Bridge' / '3rdparty' / 'Megatron-LM'}:"
            "src:."
        ),
        "PROOF_PIPELINE_REPO_ROOT": str(REPO_ROOT),
        "PROOF_PIPELINE_STRUCTURED_LOGS": "1",
        "PROOF_AB_V2_MANIFEST_SHA256": str(manifest["manifest_sha256"]),
        "PROOF_EXPECTED_PYTHON_EXECUTABLE": str(runtime["bridge_python_executable"]),
    }
    for key in (
        RECOVERY_AUTHORIZATION_ENV,
        RECOVERY_AUTHORIZATION_SHA256_ENV,
    ):
        value = os.getenv(key, "").strip()
        if value:
            env[key] = value
    if _is_h100_resource(job["resource"]):
        frozen_env = runtime.get("training", {}).get("runtime_env")
        if (
            not isinstance(frozen_env, dict)
            or set(frozen_env) != policy.required_training_env_keys
        ):
            raise WorkflowError(
                "cannot render H100 job without the exact frozen runtime env"
            )
        env.update({str(key): str(value) for key, value in frozen_env.items()})
    else:
        data_env = runtime.get("data", {}).get("runtime_env")
        if not isinstance(data_env, dict) or set(data_env) != REQUIRED_DATA_ENV_KEYS:
            raise WorkflowError(
                "cannot render data job without frozen data runtime env"
            )
        env.update({str(key): str(value) for key, value in data_env.items()})
    variant_env = active_variant().job_environment_transform(manifest, job, runtime)
    if not isinstance(variant_env, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in variant_env.items()
    ):
        raise WorkflowError("workflow variant returned an invalid job environment")
    env.update(variant_env)
    return env


def render_scripts(manifest_path: Path, manifest: Mapping[str, Any]) -> None:
    output_root = Path(str(manifest["output_root"]))
    scripts = output_root / "control" / "sbatch"
    if scripts.exists():
        shutil.rmtree(scripts)
    logs = output_root / "logs"
    runtime = _load_object(Path(str(manifest["runtime_path"])))
    for job in manifest["jobs"]:
        executor = _resource_executor(
            runtime,
            str(job["resource"]),
            time_limit=(
                str(job["slurm_time_limit"])
                if job.get("slurm_time_limit")
                else None
            ),
            time_min=(
                str(job["slurm_time_min"]) if job.get("slurm_time_min") else None
            ),
        )
        job_id = str(job["job_id"])
        script = executor._build_job_script(  # noqa: SLF001 - deterministic preview uses the real executor renderer
            _runner_command(
                manifest_path,
                job_id,
                str(runtime["bridge_python_executable"]),
            ),
            _job_env(manifest, job, runtime),
            job_id,
            str(
                logs / f"{job_id}.%A_%a.log"
                if job.get("array")
                else logs / f"{job_id}.%j.log"
            ),
        )
        target = scripts / f"{job_id}.sbatch.sh"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(script, encoding="utf-8")


from .scheduler_replan import (
    _archive_terminal_pre_gpu_attempt_for_replan,
    _archive_terminal_sentinel_startup_attempt_for_replan,
)
def plan_workflow(
    *,
    spec: Path,
    runtime: Path,
    output_root: Path,
    force: bool = False,
    build_manifest_fn: Callable[..., dict[str, Any]] = build_manifest,
) -> Path:
    manifest = build_manifest_fn(
        spec_path=spec, runtime_path=runtime, output_root=output_root
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "control" / MANIFEST_NAME
    if manifest_path.exists():
        existing = _load_object(manifest_path)
        if existing == manifest:
            render_scripts(manifest_path, manifest)
            _atomic_json(
                output_root / "control" / GPU_ESTIMATE_NAME,
                manifest["gpu_hour_estimate"],
            )
            return manifest_path
        if not force:
            raise WorkflowError(
                f"refusing to replace a different workflow manifest: {manifest_path}; use --force-plan"
            )
        state_path = output_root / "control" / STATE_NAME
        if state_path.exists() and _load_object(state_path).get("jobs"):
            try:
                _archive_terminal_pre_gpu_attempt_for_replan(
                    output_root=output_root,
                    manifest_path=manifest_path,
                    state_path=state_path,
                )
            except WorkflowError as pre_gpu_error:
                try:
                    _archive_terminal_sentinel_startup_attempt_for_replan(
                        output_root=output_root,
                        manifest_path=manifest_path,
                        state_path=state_path,
                    )
                except WorkflowError as sentinel_error:
                    raise WorkflowError(
                        "cannot force-replan this submitted workflow: "
                        f"pre-GPU recovery rejected it ({pre_gpu_error}); "
                        f"sentinel-startup recovery rejected it ({sentinel_error})"
                    ) from sentinel_error
        receipts_dir = output_root / "control" / "receipts"
        if receipts_dir.exists() and any(receipts_dir.glob("*.json")):
            raise WorkflowError(
                "cannot force-replan after any execution receipts exist"
            )
        submitted_scripts = output_root / "control" / "submitted_sbatch"
        if submitted_scripts.exists():
            shutil.rmtree(submitted_scripts)
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        output_root / "control" / GPU_ESTIMATE_NAME, manifest["gpu_hour_estimate"]
    )
    render_scripts(manifest_path, manifest)
    return manifest_path


from .scheduler_state import (
    _empty_state,
    _load_state,
    _normalized_slurm_state,
)
def submit_workflow(
    *,
    manifest_path: Path,
    retry_failed: bool = False,
    dry_run: bool = False,
    through_stage: str | None = None,
    selected_job_ids: Iterable[str] | None = None,
    validate_manifest_fn: Callable[..., dict[str, Any]] = validate_manifest,
) -> dict[str, Any]:
    manifest = _load_object(manifest_path)
    validate_manifest_fn(manifest)
    estimate = manifest["gpu_hour_estimate"]
    if estimate.get("go_no_go") != "go":
        raise WorkflowError(
            f"GPU-hour go/no-go is {estimate.get('go_no_go')}; "
            f"worst_case={estimate.get('worst_case_estimated_gpu_hours')} limit={estimate.get('gpu_hour_limit')}"
        )
    runtime = _load_object(Path(str(manifest["runtime_path"])))
    retry_policy = runtime.get("scheduling", {}).get("retry_policy")
    maximum_infrastructure_retries: int | None = None
    if retry_policy is not None:
        if (
            not isinstance(retry_policy, Mapping)
            or retry_policy.get("manual_only") is not True
            or retry_policy.get("scientific_or_contract_failures_retryable") is not False
        ):
            raise WorkflowError("runtime retry policy drifted")
        maximum_infrastructure_retries = int(
            retry_policy.get("maximum_infrastructure_retries", 0)
        )
        if maximum_infrastructure_retries != 2:
            raise WorkflowError("runtime must cap infrastructure retries at two")
    output_root = Path(str(manifest["output_root"]))
    state_path = output_root / "control" / STATE_NAME
    state = _load_state(state_path, manifest)

    if through_stage is not None and through_stage not in SUBMISSION_PHASES:
        raise WorkflowError(f"unknown submission stage {through_stage!r}")
    target_index = (
        SUBMISSION_PHASES.index(through_stage)
        if through_stage is not None
        else len(SUBMISSION_PHASES) - 1
    )
    ordered_candidates, jobs_by_id = _ordered_submission_candidates(
        manifest, selected_job_ids
    )
    eligible = [
        job
        for job in ordered_candidates
        if SUBMISSION_PHASES.index(_submission_phase(job)) <= target_index
    ]
    invalid_retry_receipts: set[str] = set()
    prior_completion_required = bool(
        manifest.get("staged_submission", {}).get(
            "prior_phase_completion_required", True
        )
    )
    if through_stage is not None and target_index > 0 and prior_completion_required:
        incomplete_prior = [
            str(job["job_id"])
            for job in manifest["jobs"]
            if SUBMISSION_PHASES.index(_submission_phase(job)) < target_index
            and not _has_completed_job_evidence(manifest, state, job)
        ]
        if incomplete_prior:
            raise WorkflowError(
                f"cannot enter submission stage {through_stage}: prior phase lacks complete "
                "receipt/state evidence: "
                + ", ".join(incomplete_prior[:8])
                + (" ..." if len(incomplete_prior) > 8 else "")
            )
    jobs_to_submit: list[Mapping[str, Any]] = []
    for candidate in eligible:
        candidate_id = str(candidate["job_id"])
        prior = state["jobs"].get(candidate_id, {})
        prior_state = _normalized_slurm_state(prior.get("state"))
        try:
            if _has_completed_job_evidence(manifest, state, candidate):
                continue
        except WorkflowError:
            if not (
                retry_failed
                and (
                    prior_state == "COMPLETED"
                    or prior_state in RETRYABLE_TERMINAL_STATES
                )
            ):
                raise
            invalid_retry_receipts.add(candidate_id)
        if prior_state == "COMPLETED":
            if retry_failed:
                jobs_to_submit.append(candidate)
            continue
        if prior.get("slurm_job_id") and prior_state not in RETRYABLE_TERMINAL_STATES:
            continue
        if prior_state in RETRYABLE_TERMINAL_STATES and not retry_failed:
            continue
        jobs_to_submit.append(candidate)
    ready_only = bool(runtime.get("scheduling", {}).get("submit_ready_only", False))
    if ready_only:
        ready_jobs: list[Mapping[str, Any]] = []
        for candidate in jobs_to_submit:
            dependencies_ready = True
            for dependency in candidate.get("dependencies", []):
                dependency_row = jobs_by_id[str(dependency)]
                try:
                    if not _has_completed_job_evidence(
                        manifest, state, dependency_row
                    ):
                        dependencies_ready = False
                        break
                except WorkflowError:
                    if not (retry_failed and str(dependency) in invalid_retry_receipts):
                        raise
                    dependencies_ready = False
                    break
            if dependencies_ready:
                ready_jobs.append(candidate)
        jobs_to_submit = ready_jobs
    pending_array_jobs = [job for job in jobs_to_submit if job.get("array")]
    if through_stage is not None and len(pending_array_jobs) > 1:
        raise WorkflowError(
            "staged submission would queue multiple 200-task arrays at once; submit and "
            "complete the earlier phase first: "
            + ", ".join(str(job["job_id"]) for job in pending_array_jobs)
        )
    submit_slots = sum(_submit_slot_count(job) for job in jobs_to_submit)
    submit_limit = int(runtime.get("max_submit_slots_per_invocation", 500) or 0)
    if submit_limit <= 0 or submit_slots > submit_limit:
        raise WorkflowError(
            f"submission would consume {submit_slots} Slurm submit slots; limit is {submit_limit}; "
            "array elements count individually"
        )

    submission_ids = {str(row["job_id"]) for row in jobs_to_submit}
    for job in eligible:
        job_id = str(job["job_id"])
        if ready_only and job_id not in submission_ids:
            continue
        prior = state["jobs"].get(job_id, {})
        prior_state = _normalized_slurm_state(prior.get("state"))
        try:
            if _has_completed_job_evidence(manifest, state, job):
                continue
        except WorkflowError:
            if not (
                retry_failed
                and (
                    prior_state == "COMPLETED"
                    or prior_state in RETRYABLE_TERMINAL_STATES
                )
            ):
                raise
        if prior_state == "COMPLETED" and not retry_failed:
            raise WorkflowError(
                f"{job_id} is Slurm COMPLETED but lacks complete hash-valid receipt evidence; "
                "pass --retry-failed for an exact resubmission"
            )
        if prior.get("slurm_job_id") and prior_state not in RETRYABLE_TERMINAL_STATES:
            if prior_state != "COMPLETED":
                continue
        if prior_state in RETRYABLE_TERMINAL_STATES and not retry_failed:
            raise WorkflowError(
                f"{job_id} previously failed; pass --retry-failed to resubmit"
            )

        dependency_ids: list[str] = []
        for dependency in job.get("dependencies", []):
            dependency_state = state["jobs"].get(str(dependency), {})
            dependency_job_id = str(dependency_state.get("slurm_job_id", ""))
            dependency_row = jobs_by_id[str(dependency)]
            try:
                dependency_complete = _has_completed_job_evidence(
                    manifest, state, dependency_row
                )
            except WorkflowError:
                if not (retry_failed and str(dependency) in invalid_retry_receipts):
                    raise
                dependency_complete = False
            if dependency_complete:
                continue
            if dependency_state.get("state") == "COMPLETED":
                raise WorkflowError(
                    f"cannot trust Slurm COMPLETED for {dependency} without complete hash-valid receipts"
                )
            if not dependency_job_id:
                raise WorkflowError(
                    f"cannot submit {job_id}: dependency {dependency} has neither a receipt nor a Slurm job ID"
                )
            dependency_ids.append(dependency_job_id)
        dependency_spec = (
            "afterok:" + ":".join(dependency_ids) if dependency_ids else None
        )
        retrying = bool(prior.get("slurm_job_id")) and (
            prior_state == "COMPLETED" or prior_state in RETRYABLE_TERMINAL_STATES
        )
        quarantined: list[str] = []
        history = list(prior.get("submission_history", []))
        if retrying and retry_failed:
            if maximum_infrastructure_retries is not None:
                if prior_state == "COMPLETED":
                    raise WorkflowError(
                        f"{job_id} has invalid completed evidence; contract failures are not retryable"
                    )
                if len(history) >= maximum_infrastructure_retries:
                    raise WorkflowError(
                        f"{job_id} exhausted its two manually authorized infrastructure retries"
                    )
            if not dry_run:
                quarantined = _quarantine_job_receipts(manifest, job)
            reason = (
                "hash-invalid or incomplete receipt evidence"
                if job_id in invalid_retry_receipts or prior_state == "COMPLETED"
                else f"retryable Slurm terminal state {prior_state}"
            )
            history.append(
                _retry_history_entry(
                    prior=prior,
                    job=job,
                    quarantined_receipts=quarantined,
                    reason=reason,
                    output_root=output_root,
                )
            )
            if not dry_run:
                state["jobs"][job_id] = {**prior, "submission_history": history}
                _atomic_json(state_path, state)
        executor = _resource_executor(
            runtime,
            str(job["resource"]),
            time_limit=(
                str(job["slurm_time_limit"])
                if job.get("slurm_time_limit")
                else None
            ),
            time_min=(
                str(job["slurm_time_min"]) if job.get("slurm_time_min") else None
            ),
        )
        log_path = output_root / "logs" / f"{job_id}.controller.log"
        slurm_output = (
            output_root
            / "logs"
            / (f"{job_id}.%A_%a.log" if job.get("array") else f"{job_id}.%j.log")
        )
        job_env = _job_env(manifest, job, runtime)
        if retrying and retry_failed:
            job_env["PROOF_AB_V2_RETRY_FAILED"] = "1"
        kwargs = {
            "cmd": _runner_command(
                manifest_path,
                job_id,
                str(runtime["bridge_python_executable"]),
            ),
            "env": job_env,
            "dry_run": dry_run,
            "log_path": log_path,
            "dependency": dependency_spec,
            "job_name": f"{str(runtime.get('workflow_job_prefix', 'abv2'))}_{job_id}",
            "script_dir": output_root / "control" / "submitted_sbatch",
            "slurm_output": slurm_output,
        }
        if job.get("array"):
            slurm_id = executor.submit_array(array=str(job["array"]), **kwargs)
        else:
            slurm_id = executor.submit(**kwargs)
        if dry_run:
            state["jobs"][job_id] = {
                "slurm_job_id": f"dryrun_{len(state['jobs']) + 1}",
                "state": "DRY_RUN",
                "resource": job["resource"],
                "array": job.get("array"),
                "dependencies": list(job.get("dependencies", [])),
                "slurm_time_limit": job.get("slurm_time_limit"),
                "slurm_time_min": job.get("slurm_time_min"),
                "submission_history": history,
            }
            continue
        if not slurm_id:
            raise WorkflowError(f"Slurm returned no job ID for {job_id}")
        state["jobs"][job_id] = {
            "slurm_job_id": str(slurm_id),
            "state": "SUBMITTED",
            "resource": job["resource"],
            "array": job.get("array"),
            "dependencies": list(job.get("dependencies", [])),
            "slurm_time_limit": job.get("slurm_time_limit"),
            "slurm_time_min": job.get("slurm_time_min"),
            "submission_history": history,
        }
        _atomic_json(state_path, state)
    state["last_submission"] = {
        "through_stage": through_stage or "all_direct_api",
        "submit_slots": submit_slots,
        "array_elements_count_individually": True,
        "ready_only": ready_only,
        "selected_job_count": (
            len(eligible) if selected_job_ids is not None else None
        ),
    }
    if not dry_run:
        _atomic_json(state_path, state)
    return state


from .scheduler_accounting import (
    _allocated_gpus_from_tres,
    _array_accounting_rollup,
    _parse_sacct_output,
    _query_states,
)
def reconcile_workflow(
    manifest_path: Path,
    *,
    validate_manifest_fn: Callable[..., dict[str, Any]] = validate_manifest,
) -> dict[str, Any]:
    manifest = _load_object(manifest_path)
    validate_manifest_fn(manifest)
    state_path = Path(str(manifest["output_root"])) / "control" / STATE_NAME
    state = _load_state(state_path, manifest)
    lookup = {
        str(value.get("slurm_job_id")): job_id
        for job_id, value in state["jobs"].items()
        if value.get("slurm_job_id")
    }
    jobs_by_id = {str(job["job_id"]): job for job in manifest["jobs"]}
    observed = _query_states(lookup)
    for slurm_id, job_id in lookup.items():
        parent = observed.get(slurm_id)
        tasks = {
            canonical_job_id: status
            for canonical_job_id, status in observed.items()
            if canonical_job_id.startswith(f"{slurm_id}_")
        }
        job = jobs_by_id[job_id]
        if job.get("array"):
            expected_task_ids = {
                f"{slurm_id}_{task_id}"
                for task_id in _array_indices(str(job["array"]))
            }
            aggregate = _array_accounting_rollup(
                parent=parent,
                tasks=tasks,
                expected_task_ids=expected_task_ids,
            )
        else:
            aggregate = dict(parent) if parent else {}
        if aggregate.get("state"):
            state["jobs"][job_id].update(
                {
                    "state": aggregate["state"],
                    "exit_code": aggregate.get("exit_code", ""),
                }
            )
        state["jobs"][job_id]["slurm_accounting"] = {
            "parent": parent or {},
            "tasks": dict(sorted(tasks.items())),
            "aggregate": aggregate,
            "allocated_rows": "array_tasks" if tasks else "parent",
        }
    _atomic_json(state_path, state)
    return state


def _logical_h100_runtime(
    job: Mapping[str, Any], paths: Paths
) -> tuple[str, float] | None:
    action = str(job.get("action", ""))
    job_id = str(job["job_id"])
    if action in {
        "training_sentinel",
        "prescribed_lr_schedule_probe",
        "production_train",
    }:
        result = _load_object(paths.jobs / job_id / "result.json")
        if result.get("status") == "not_required":
            return "training", 0.0
        seconds = float(result.get("end_to_end_seconds", math.nan))
        if (
            result.get("status") != "complete"
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise WorkflowError(f"{job_id} lacks measured logical training runtime")
        return "training", seconds
    if action == "per_language_eval":
        path = paths.jobs / job_id / "evaluation_manifest.json"
        payload = _load_object(path)
        if payload.get("status") == "not_required":
            return "evaluation", 0.0
        # Verify the immutable evaluator/report identity before trusting timing.
        payload, _report, _report_path = _validation_report_from_eval(path)
        seconds = float(
            payload.get("timing", {}).get("internal_total_seconds", math.nan)
        )
        if not math.isfinite(seconds) or seconds <= 0:
            raise WorkflowError(f"{job_id} lacks measured logical evaluation runtime")
        return "evaluation", seconds
    return None


def collect_workflow(
    manifest_path: Path,
    *,
    validate_manifest_fn: Callable[..., dict[str, Any]] = validate_manifest,
) -> dict[str, Any]:
    manifest = _load_object(manifest_path)
    state = reconcile_workflow(
        manifest_path,
        validate_manifest_fn=validate_manifest_fn,
    )
    by_id = {str(row["job_id"]): row for row in manifest["jobs"]}
    status_counts = Counter(
        str(value.get("state", "NOT_SUBMITTED")) for value in state["jobs"].values()
    )
    missing_receipts: list[str] = []
    invalid_receipts: list[str] = []
    accounting_errors: list[str] = []
    failed: list[str] = []
    for job_id, value in state["jobs"].items():
        normalized_state = _normalized_slurm_state(value.get("state"))
        if normalized_state in RETRYABLE_TERMINAL_STATES:
            failed.append(job_id)
        if normalized_state == "COMPLETED":
            try:
                complete = _has_completed_job_evidence(manifest, state, by_id[job_id])
            except WorkflowError as exc:
                invalid_receipts.append(f"{job_id}: {exc}")
                continue
            if not complete:
                array_spec = by_id[job_id].get("array")
                if array_spec:
                    for task_id in _array_indices(str(array_spec)):
                        if not _receipt_path(manifest, job_id, str(task_id)).is_file():
                            missing_receipts.append(f"{job_id}_{task_id}")
                else:
                    missing_receipts.append(job_id)
    paths = Paths(Path(str(manifest["output_root"])))
    per_job_accounting: dict[str, Any] = {}
    allocated_h100_hours = 0.0
    logical_training_seconds = 0.0
    logical_evaluation_seconds = 0.0
    for job_id, value in state["jobs"].items():
        job = by_id[job_id]
        if not _is_h100_resource(job.get("resource")):
            continue
        slurm_accounting = value.get("slurm_accounting", {})
        task_rows = (
            slurm_accounting.get("tasks", {})
            if isinstance(slurm_accounting, dict)
            else {}
        )
        if isinstance(task_rows, dict) and task_rows:
            allocation_rows = list(task_rows.values())
            allocation_source = "array_tasks"
        else:
            parent = (
                slurm_accounting.get("parent", {})
                if isinstance(slurm_accounting, dict)
                else {}
            )
            allocation_rows = [parent] if isinstance(parent, dict) and parent else []
            allocation_source = "parent"
        allocated = sum(
            float(row.get("allocated_gpu_hours", 0.0)) for row in allocation_rows
        )
        allocated_h100_hours += allocated
        logical_kind: str | None = None
        logical_seconds: float | None = None
        if _normalized_slurm_state(value.get("state")) == "COMPLETED":
            try:
                logical = _logical_h100_runtime(job, paths)
            except (OSError, TypeError, ValueError, WorkflowError) as exc:
                accounting_errors.append(f"{job_id}: {exc}")
            else:
                if logical is not None:
                    logical_kind, logical_seconds = logical
                    if logical_kind == "training":
                        logical_training_seconds += logical_seconds
                    else:
                        logical_evaluation_seconds += logical_seconds
        per_job_accounting[job_id] = {
            "slurm_job_id": value.get("slurm_job_id"),
            "allocation_source": allocation_source,
            "allocated_h100_hours": allocated,
            "logical_kind": logical_kind,
            "logical_runtime_seconds": logical_seconds,
            "slurm_accounting": slurm_accounting,
        }
    logical_h100_hours = (
        logical_training_seconds + logical_evaluation_seconds
    ) / 3600.0
    h100_accounting = {
        "definition": (
            "allocated sums ElapsedRaw*allocated GPUs over scalar parent rows or "
            "array task rows, never both and never .batch/.extern steps; logical sums "
            "hash-valid result end-to-end training runtime and evaluator internal runtime"
        ),
        "allocated_h100_hours": allocated_h100_hours,
        "logical_training_hours": logical_training_seconds / 3600.0,
        "logical_evaluation_hours": logical_evaluation_seconds / 3600.0,
        "logical_total_h100_hours": logical_h100_hours,
        "allocation_minus_logical_h100_hours": allocated_h100_hours
        - logical_h100_hours,
        "per_job": per_job_accounting,
    }
    report = {
        "schema_version": 1,
        "manifest_sha256": manifest["manifest_sha256"],
        "submitted_jobs": len(state["jobs"]),
        "total_jobs": len(manifest["jobs"]),
        "state_counts": dict(sorted(status_counts.items())),
        "failed_jobs": sorted(failed),
        "completed_without_receipt": sorted(missing_receipts),
        "invalid_receipts": sorted(invalid_receipts),
        "accounting_errors": sorted(accounting_errors),
        "h100_hour_accounting": h100_accounting,
        "status": (
            "failed"
            if failed or missing_receipts or invalid_receipts or accounting_errors
            else "ok"
        ),
        "gpu_hour_estimate": manifest["gpu_hour_estimate"],
    }
    _atomic_json(
        Path(str(manifest["output_root"])) / "control" / "collect_report.json", report
    )
    return report
