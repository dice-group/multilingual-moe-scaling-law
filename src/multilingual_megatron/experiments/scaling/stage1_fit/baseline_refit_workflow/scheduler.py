"""Render and submit the complete baseline-refit dependency graph once."""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..execution import read_json, write_json_once, write_text_once
from .contracts import (
    FINAL_SCHEMA,
    BaselineRefitError,
    build_workflow,
    control_root,
    gate_path,
    load_profile,
    load_workflow,
    repository_root,
    require_committed_workflow_sources,
    uncommitted_workflow_sources,
    validate_workflow,
    verify_seal,
    workflow_path,
)
from .planning import build_initial_manifests, write_index
from .state import load_state, state_path, submission_lock, write_state


Submitter = Callable[[Path, str | None], str]


def _prepared(profile_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    profile = load_profile(profile_path)
    initial = build_initial_manifests(profile)
    workflow = build_workflow(profile, initial)
    validate_workflow(workflow)
    return profile, initial, workflow


def inspect(profile_path: Path) -> dict[str, Any]:
    _, _, workflow = _prepared(profile_path)
    changed = uncommitted_workflow_sources(workflow["workflow_source"])
    return {
        "workflow_sha256": workflow["workflow_sha256"],
        "output_root": workflow["output_root"],
        "models": workflow["models"],
        "folds": workflow["folds"],
        "inputs": workflow["inputs"],
        "model_outer_parameter_counts": workflow["model_outer_parameter_counts"],
        "claim_rule": workflow["claim_rule"],
        "registered_optimizer_starts": workflow["registered_optimizer_starts"],
        "array_tasks": workflow["array_tasks"],
        "array_jobs": sum(job["kind"] == "array" for job in workflow["jobs"]),
        "gate_jobs": sum(job["kind"] == "gate" for job in workflow["jobs"]),
        "slurm_jobs": len(workflow["jobs"]),
        "largest_array": max(int(job["task_count"] or 0) for job in workflow["jobs"]),
        "workflow_sources_committed": not changed,
        "uncommitted_workflow_sources": changed,
        "dag": [{key: job[key] for key in (
            "job_id", "kind", "model", "fold", "array", "dependencies",
            "dependency_mode", "time_limit", "memory",
        )} for job in workflow["jobs"]],
        "submitted": False,
    }


def _script(
    workflow: Mapping[str, Any],
    workflow_manifest: Path,
    job: Mapping[str, Any],
    *,
    attempt_id: str,
) -> str:
    profile = workflow["profile"]
    log_root = Path(str(workflow["output_root"])) / "logs" / "attempts" / attempt_id
    suffix = "%A_%a" if job["kind"] == "array" else "%j"
    job_name = "s1br-" + str(job["job_id"]).replace(".", "-")
    directives = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --cpus-per-task={job['cpus_per_task']}",
        f"#SBATCH --mem={job['memory']}",
        f"#SBATCH --time={job['time_limit']}",
        f"#SBATCH --output={log_root / (str(job['job_id']).replace('.', '-') + '-' + suffix + '.out')}",
        "#SBATCH --export=NIL",
    ]
    if job["array"]:
        directives.append(f"#SBATCH --array={job['array']}")
    for key, flag in (("partition", "partition"), ("account", "account"), ("reservation", "reservation")):
        if profile.get(key):
            directives.append(f"#SBATCH --{flag}={profile[key]}")
    threads = int(profile["blas_threads"] if job["kind"] == "array" else 1)
    body = [
        "set -euo pipefail",
        f"cd {shlex.quote(str(repository_root()))}",
        f"export PYTHONPATH={shlex.quote('src:.')}",
        *[f"export {name}={threads}" for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        )],
        " ".join(shlex.quote(value) for value in (
            str(profile["python_executable"]),
            "-m",
            "multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit",
            "execute",
            "--workflow-manifest",
            str(workflow_manifest),
            "--job-id",
            str(job["job_id"]),
        )),
    ]
    return "\n".join((*directives, "", *body, ""))


def _default_submitter(script: Path, dependency: str | None) -> str:
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency={dependency}")
    command.append(str(script))
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or getattr(error, "stdout", None) or str(error)
        raise BaselineRefitError(f"sbatch failed: {str(detail).strip()}") from error
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise BaselineRefitError(f"sbatch returned an invalid job id: {result.stdout.strip()!r}")
    return job_id


def _ensure_new_or_matching_root(workflow: Mapping[str, Any]) -> None:
    output = Path(str(workflow["output_root"]))
    manifest = workflow_path(output)
    if (output / "receipts" / "finalization.json").exists():
        raise BaselineRefitError("baseline-refit output root is already finalized")
    if output.exists() and any(output.iterdir()) and not manifest.exists():
        raise BaselineRefitError(f"refusing non-empty unowned output root: {output}")
    if manifest.exists():
        existing = load_workflow(manifest)
        if existing["workflow_sha256"] != workflow["workflow_sha256"]:
            raise BaselineRefitError("output root belongs to another baseline-refit workflow")


def submit(profile_path: Path, *, submitter: Submitter = _default_submitter,
        require_committed: bool = True) -> dict[str, Any]:
    if os.environ.get("SLURM_JOB_ID"):
        raise BaselineRefitError("refusing nested Slurm submission from an allocation")
    _, initial, workflow = _prepared(profile_path)
    if require_committed:
        require_committed_workflow_sources(workflow["workflow_source"])
    _ensure_new_or_matching_root(workflow)
    output = Path(str(workflow["output_root"]))
    manifest_path = workflow_path(output)
    write_json_once(manifest_path, workflow)
    write_index(workflow, name="m0_ols", manifests=initial)

    with submission_lock(workflow):
        state = load_state(workflow)
        write_state(state_path(workflow), state)
        attempt_id = str(state["attempt_id"])
        script_root = control_root(workflow) / "submitted_sbatch" / attempt_id
        (output / "logs" / "attempts" / attempt_id).mkdir(parents=True, exist_ok=True)
        for job in workflow["jobs"]:
            job_id = str(job["job_id"])
            prior = state["jobs"].get(job_id)
            if isinstance(prior, Mapping) and prior.get("slurm_job_id"):
                continue
            dependencies = []
            for logical_parent in job["dependencies"]:
                parent = state["jobs"].get(str(logical_parent))
                if not isinstance(parent, Mapping) or not parent.get("slurm_job_id"):
                    raise BaselineRefitError(
                        f"cannot submit {job_id}: dependency {logical_parent} has no Slurm id"
                    )
                dependencies.append(str(parent["slurm_job_id"]))
            dependency = None
            if dependencies:
                dependency = str(job["dependency_mode"]) + ":" + ":".join(dependencies)
            script_path = script_root / f"{job_id.replace('.', '-')}.sbatch"
            write_text_once(script_path, _script(
                workflow, manifest_path, job, attempt_id=attempt_id,
            ))
            slurm_id = submitter(script_path, dependency)
            if not str(slurm_id).isdigit():
                raise BaselineRefitError(f"submitter returned invalid Slurm id for {job_id}")
            state["jobs"][job_id] = {
                "slurm_job_id": str(slurm_id),
                "dependency": dependency,
                "script": str(script_path),
                "kind": job["kind"],
            }
            write_state(state_path(workflow), state)
    return {
        "workflow_manifest": str(manifest_path),
        "workflow_sha256": workflow["workflow_sha256"],
        "attempt_id": state["attempt_id"],
        "submitted_jobs": len(state["jobs"]),
        "final_job_id": state["jobs"]["gate.final"]["slurm_job_id"],
        "submitted": True,
    }


def _scheduler_states(job_ids: Sequence[str]) -> tuple[dict[str, str], str | None]:
    if not job_ids:
        return {}, None
    try:
        result = subprocess.run(
            ["sacct", "-j", ",".join(job_ids), "--format=JobIDRaw,State", "--noheader", "--parsable2"],
            check=False, capture_output=True, text=True,
        )
    except OSError as error:
        return {}, str(error)
    if result.returncode != 0:
        return {}, (result.stderr or result.stdout).strip()
    states: dict[str, str] = {}
    requested = set(job_ids)
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 2 and parts[0].strip() in requested:
            states[parts[0].strip()] = parts[1].strip().split()[0].split("+")[0]
    return states, None


def status(profile_path: Path) -> dict[str, Any]:
    profile = load_profile(profile_path)
    manifest_path = workflow_path(Path(str(profile["output_root"])))
    if not manifest_path.exists():
        return {"status": "not_submitted", "workflow_manifest": str(manifest_path)}
    workflow = load_workflow(manifest_path, verify_live_files=False)
    live_error = None
    try:
        validate_workflow(workflow, verify_live_files=True)
    except BaselineRefitError as error:
        live_error = str(error)
    state = load_state(workflow)
    scheduler, scheduler_error = _scheduler_states([
        str(value["slurm_job_id"]) for value in state["jobs"].values()
        if isinstance(value, Mapping) and value.get("slurm_job_id")
    ])
    gates = {}
    blocked = []
    for gate_id in ("gate.scale", "gate.m0", "gate.baselines", "gate.m1", "gate.final"):
        path = gate_path(workflow, gate_id)
        if not path.exists():
            continue
        receipt = read_json(path)
        verify_seal(receipt, "gate_sha256", label=gate_id)
        gates[gate_id] = {"status": receipt.get("status"), "path": str(path)}
        if receipt.get("status") == "blocked":
            blocked.append(gate_id)
    final_path = Path(str(workflow["output_root"])) / "receipts" / "finalization.json"
    final = read_json(final_path) if final_path.exists() else None
    if final is not None:
        verify_seal(final, "receipt_sha256", label="finalization")
        if final.get("schema_version") != FINAL_SCHEMA:
            raise BaselineRefitError("finalization schema changed")
    failure_states = {
        "BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL",
        "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT",
    }
    failed_jobs = [logical for logical, value in state["jobs"].items()
        if scheduler.get(str(value.get("slurm_job_id"))) in failure_states]
    observed = "complete" if final is not None else (
        "blocked" if blocked else (
            "scheduler_failed" if failed_jobs else (
                "partially_submitted" if len(state["jobs"]) < len(workflow["jobs"]) else "running"
            )
        )
    )
    jobs = {
        logical: {
            **dict(value),
            "scheduler_state": scheduler.get(str(value.get("slurm_job_id"))),
        }
        for logical, value in state["jobs"].items()
    }
    return {
        "status": observed,
        "workflow_sha256": workflow["workflow_sha256"],
        "attempt_id": state["attempt_id"],
        "submitted_jobs": len(jobs),
        "expected_jobs": len(workflow["jobs"]),
        "gates": gates,
        "blocked_gates": blocked,
        "failed_jobs": failed_jobs,
        "finalization": None if final is None else str(final_path),
        "strict_baseline_win": None if final is None else final.get("strict_baseline_win"),
        "live_contract_error": live_error,
        "scheduler_error": scheduler_error,
        "jobs": jobs,
    }
