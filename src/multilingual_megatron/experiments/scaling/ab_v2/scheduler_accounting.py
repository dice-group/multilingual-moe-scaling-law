"""Parse Slurm accounting and derive array-level scheduler state."""

from __future__ import annotations

import subprocess
from typing import Any, Iterable, Mapping

from .common import RETRYABLE_TERMINAL_STATES, WorkflowError
from .scheduler_state import _normalized_slurm_state


def _allocated_gpus_from_tres(raw: str) -> int:
    values: dict[str, int] = {}
    for item in str(raw).split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            values[key.strip()] = int(value)
        except ValueError:
            continue
    typed = [
        value
        for key, value in values.items()
        if key.startswith("gres/gpu:") or key.startswith("gpu:")
    ]
    if typed:
        return sum(typed)
    for key in ("gres/gpu", "gpu"):
        if key in values:
            return values[key]
    return 0


def _parse_sacct_output(output: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        job_id = parts[0].strip()
        job_id_raw = parts[1].strip()
        if not job_id or "." in job_id:
            continue
        try:
            elapsed_seconds = int(parts[4] or 0)
        except ValueError:
            elapsed_seconds = 0
        alloc_tres = parts[5].strip()
        allocated_gpus = _allocated_gpus_from_tres(alloc_tres)
        rows[job_id] = {
            "job_id": job_id,
            "job_id_raw": job_id_raw,
            "state": parts[2].split()[0] if parts[2].strip() else "",
            "exit_code": parts[3].strip(),
            "elapsed_seconds": elapsed_seconds,
            "alloc_tres": alloc_tres,
            "allocated_gpus": allocated_gpus,
            "allocated_gpu_hours": elapsed_seconds * allocated_gpus / 3600.0,
        }
    return rows


def _array_accounting_rollup(
    *,
    parent: Mapping[str, Any] | None,
    tasks: Mapping[str, Mapping[str, Any]],
    expected_task_ids: set[str],
) -> dict[str, Any]:
    """Derive array state from canonical task IDs without counting allocations twice."""

    counts = {
        "observed_task_count": len(tasks),
        "expected_task_count": len(expected_task_ids),
    }

    def task_sort_key(value: str) -> tuple[int, int | str]:
        suffix = value.rsplit("_", 1)[-1]
        return (0, int(suffix)) if suffix.isdigit() else (1, suffix)

    observed_task_ids = set(tasks)
    ordered_task_ids = sorted(tasks, key=task_sort_key)
    normalized_task_states = {
        task_id: _normalized_slurm_state(tasks[task_id].get("state"))
        for task_id in ordered_task_ids
    }
    failed_task_ids = [
        task_id
        for task_id in ordered_task_ids
        if normalized_task_states[task_id] in RETRYABLE_TERMINAL_STATES
    ]
    active_state_priority = (
        "RUNNING",
        "COMPLETING",
        "CONFIGURING",
        "PENDING",
        "REQUEUED",
        "REQUEUE_FED",
        "RESIZING",
        "SIGNALING",
        "STAGE_OUT",
        "SUSPENDED",
    )
    active_task = next(
        (
            (task_id, active_state)
            for active_state in active_state_priority
            for task_id in ordered_task_ids
            if normalized_task_states[task_id] == active_state
        ),
        None,
    )
    parent_state = _normalized_slurm_state(parent.get("state")) if parent else ""
    terminal_states = RETRYABLE_TERMINAL_STATES | {"COMPLETED"}
    nonterminal_task = next(
        (
            (task_id, normalized_task_states[task_id])
            for task_id in ordered_task_ids
            if normalized_task_states[task_id]
            and normalized_task_states[task_id] not in terminal_states
        ),
        None,
    )

    if failed_task_ids:
        failure = tasks[failed_task_ids[0]]
        failure_detail = {
            "failed_task_ids": failed_task_ids,
            "first_failed_task_job_id": failed_task_ids[0],
        }
        # A single task may fail while other array elements are still active. Keep
        # the aggregate non-terminal until Slurm reports complete task coverage or
        # a terminal parent, otherwise --retry-failed could overlap the live array.
        if active_task is not None:
            active_task_id, active_state = active_task
            return {
                "state": active_state,
                "exit_code": str(tasks[active_task_id].get("exit_code", "")),
                "source": "array_task_active_with_failures",
                "active_task_job_id": active_task_id,
                **failure_detail,
                **counts,
            }
        if parent and parent_state and parent_state not in terminal_states:
            return {
                **parent,
                "source": "array_parent_active_with_failures",
                **failure_detail,
                **counts,
            }
        if nonterminal_task is not None:
            nonterminal_task_id, nonterminal_state = nonterminal_task
            return {
                "state": nonterminal_state,
                "exit_code": str(tasks[nonterminal_task_id].get("exit_code", "")),
                "source": "array_task_nonterminal_with_failures",
                "nonterminal_task_job_id": nonterminal_task_id,
                **failure_detail,
                **counts,
            }
        accounting_is_terminal = (
            bool(expected_task_ids)
            and expected_task_ids <= observed_task_ids
            and all(
                normalized_task_states[task_id] in terminal_states
                for task_id in expected_task_ids
            )
        ) or parent_state in terminal_states
        if accounting_is_terminal:
            return {
                "state": str(failure.get("state", "")),
                "exit_code": str(failure.get("exit_code", "")),
                "source": "array_task_failure",
                "task_job_id": failed_task_ids[0],
                **failure_detail,
                **counts,
            }
        return {
            "source": "incomplete_array_task_accounting_with_failures",
            **failure_detail,
            **counts,
        }

    if expected_task_ids and expected_task_ids <= observed_task_ids and all(
        _normalized_slurm_state(task.get("state")) == "COMPLETED"
        for task in tasks.values()
        if str(task.get("job_id", "")) in expected_task_ids
    ):
        return {
            "state": "COMPLETED",
            "exit_code": "0:0",
            "source": "all_array_tasks_completed",
            **counts,
        }

    if parent and parent_state == "COMPLETED" and not (
        expected_task_ids and expected_task_ids <= observed_task_ids
    ):
        return {
            "source": "incomplete_array_task_accounting_with_completed_parent",
            "parent": dict(parent),
            **counts,
        }

    if parent:
        return {**parent, "source": "array_parent", **counts}

    if active_task is not None:
        active_task_id, active_state = active_task
        task = tasks[active_task_id]
        return {
            "state": active_state,
            "exit_code": str(task.get("exit_code", "")),
            "source": "array_task_active",
            "active_task_job_id": active_task_id,
            **counts,
        }
    if nonterminal_task is not None:
        nonterminal_task_id, nonterminal_state = nonterminal_task
        return {
            "state": nonterminal_state,
            "exit_code": str(tasks[nonterminal_task_id].get("exit_code", "")),
            "source": "array_task_nonterminal",
            "nonterminal_task_job_id": nonterminal_task_id,
            **counts,
        }
    return {"source": "incomplete_array_task_accounting", **counts}


def _query_states(job_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    ids = sorted({str(value) for value in job_ids if str(value)})
    if not ids:
        return {}
    result = subprocess.run(
        [
            "sacct",
            "-j",
            ",".join(ids),
            "--format=JobID,JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES",
            "--noheader",
            "--parsable2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise WorkflowError(f"sacct failed: {(result.stderr or result.stdout).strip()}")
    return _parse_sacct_output(result.stdout)
