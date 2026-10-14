"""Archive-quality Slurm accounting for Stage 1 A--D study jobs.

This module deliberately separates three concerns:

* discover every numeric Slurm attempt associated with a logical workflow job;
* acquire the original ``sacct --json`` records in bounded batches; and
* normalize a compact analysis row without discarding the source record.

The normalized values use the study-wide provenance vocabulary:
``observed``, ``configured``, ``planned``, and ``derived``.  Epoch timestamps
are retained as UTC epoch seconds and also rendered as explicit ``Z`` strings.
"""

from __future__ import annotations

import json
import math
import re
import shlex
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "multilingual_megatron_stage1_ad_slurm_accounting/v1"
PROVENANCE_CLASSES = ("observed", "configured", "planned", "derived")
_NUMERIC_JOB_ID = re.compile(r"^[0-9]+$")
_SLURM_LOG = re.compile(
    r"^(?P<logical_job_id>.+)\.(?P<slurm_job_id>[0-9]+)"
    r"(?:_(?P<array_task_id>[0-9]+))?\.log$"
)
_LOGICAL_ID_KEYS = (
    "logical_job_id",
    "canonical_job_id",
    "canonical_id",
    "job_id",
)
_SLURM_ID_KEYS = (
    "slurm_job_id",
    "scheduler_job_id",
    "submitted_job_id",
    "slurm_id",
)
_RECEIPT_CONTAINER_KEYS = {
    "jobs",
    "attempts",
    "history",
    "submission_history",
    "records",
    "results",
    "receipts",
    "tasks",
    "entries",
    "items",
    "nested",
}


class SlurmAccountingError(RuntimeError):
    """Raised when attempt discovery or accounting cannot be sealed safely."""

    def __init__(
        self,
        message: str,
        *,
        report: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.report = dict(report) if report is not None else None


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SlurmAccountingError(f"missing JSON source: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SlurmAccountingError(f"invalid JSON source {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SlurmAccountingError(f"JSON source must contain an object: {path}")
    return payload


def _json_pointer_part(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _numeric_job_id(value: Any) -> str | None:
    candidate = str(value or "").strip()
    return candidate if _NUMERIC_JOB_ID.fullmatch(candidate) else None


def _logical_job_id(value: Any) -> str | None:
    candidate = str(value or "").strip()
    if not candidate or _NUMERIC_JOB_ID.fullmatch(candidate):
        return None
    return candidate


def _receipt_pairs(
    value: Any,
    *,
    pointer: str = "",
    logical_context: str | None = None,
) -> list[tuple[str, str, str]]:
    """Return ``(logical ID, numeric Slurm ID, JSON pointer)`` triples."""

    if isinstance(value, list):
        pairs: list[tuple[str, str, str]] = []
        for index, child in enumerate(value):
            pairs.extend(
                _receipt_pairs(
                    child,
                    pointer=f"{pointer}/{index}",
                    logical_context=logical_context,
                )
            )
        return pairs
    if not isinstance(value, Mapping):
        return []

    local_logical = next(
        (
            candidate
            for key in _LOGICAL_ID_KEYS
            if (candidate := _logical_job_id(value.get(key))) is not None
        ),
        logical_context,
    )
    local_slurm: tuple[str, str] | None = None
    for key in _SLURM_ID_KEYS:
        numeric = _numeric_job_id(value.get(key))
        if numeric is not None:
            local_slurm = (numeric, key)
            break

    pairs: list[tuple[str, str, str]] = []
    if local_logical is not None and local_slurm is not None:
        numeric, key = local_slurm
        pairs.append(
            (
                local_logical,
                numeric,
                f"{pointer}/{_json_pointer_part(key)}",
            )
        )

    for key, child in value.items():
        if not isinstance(child, (Mapping, list)):
            continue
        child_context = (
            local_logical
            if str(key) in _RECEIPT_CONTAINER_KEYS
            else (_logical_job_id(key) or local_logical)
        )
        pairs.extend(
            _receipt_pairs(
                child,
                pointer=f"{pointer}/{_json_pointer_part(str(key))}",
                logical_context=child_context,
            )
        )
    return pairs


def discover_slurm_attempts(run_roots: Sequence[Path | str]) -> list[dict[str, Any]]:
    """Discover all numeric Slurm attempts for logical jobs under run roots.

    The workflow state is authoritative for the current attempt and explicit
    retry history.  Hashable execution receipts and resolved ``*.JOBID.log``
    names provide independent observed evidence and recover attempts from
    offload/recovery roots.  Repeated evidence is merged, never discarded.
    """

    evidence_by_key: dict[
        tuple[str, str, str], dict[tuple[str, str, str], dict[str, str]]
    ] = defaultdict(dict)
    current_attempts: set[tuple[str, str, str]] = set()

    def add(
        *,
        run_root: Path,
        logical_job_id: str,
        slurm_job_id: str,
        kind: str,
        path: Path,
        detail: str,
        current: bool = False,
    ) -> None:
        root_string = str(run_root)
        key = (root_string, logical_job_id, slurm_job_id)
        source = {
            "kind": kind,
            "path": str(path.resolve()),
            "detail": detail,
            "provenance_class": "observed",
        }
        evidence_by_key[key][(kind, source["path"], detail)] = source
        if current:
            current_attempts.add(key)

    normalized_roots = sorted({Path(root).resolve() for root in run_roots})
    for run_root in normalized_roots:
        if not run_root.is_dir():
            raise SlurmAccountingError(f"run root is not a directory: {run_root}")

        state_path = run_root / "control" / "workflow_state.json"
        if state_path.is_file():
            state = _load_object(state_path)
            jobs = state.get("jobs")
            if not isinstance(jobs, Mapping):
                raise SlurmAccountingError(
                    f"workflow state jobs must be an object: {state_path}"
                )
            for raw_logical, raw_state in jobs.items():
                logical = _logical_job_id(raw_logical)
                if logical is None or not isinstance(raw_state, Mapping):
                    continue
                current_id = _numeric_job_id(raw_state.get("slurm_job_id"))
                if current_id is not None:
                    add(
                        run_root=run_root,
                        logical_job_id=logical,
                        slurm_job_id=current_id,
                        kind="workflow_state_current",
                        path=state_path,
                        detail=(f"/jobs/{_json_pointer_part(logical)}/slurm_job_id"),
                        current=True,
                    )
                history = raw_state.get("submission_history", [])
                if history is None:
                    history = []
                if not isinstance(history, list):
                    raise SlurmAccountingError(
                        f"submission history must be a list for {logical}: {state_path}"
                    )
                for index, item in enumerate(history):
                    if not isinstance(item, Mapping):
                        continue
                    historical_id = _numeric_job_id(item.get("slurm_job_id"))
                    if historical_id is None:
                        continue
                    add(
                        run_root=run_root,
                        logical_job_id=logical,
                        slurm_job_id=historical_id,
                        kind="workflow_state_history",
                        path=state_path,
                        detail=(
                            f"/jobs/{_json_pointer_part(logical)}"
                            f"/submission_history/{index}/slurm_job_id"
                        ),
                    )

        control = run_root / "control"
        receipt_paths: set[Path] = set()
        receipt_root = control / "receipts"
        if receipt_root.is_dir():
            receipt_paths.update(receipt_root.rglob("*.json"))
        if control.is_dir():
            receipt_paths.update(control.rglob("*receipt*.json"))
        for receipt_path in sorted(receipt_paths):
            payload = _load_object(receipt_path)
            for logical, slurm_id, pointer in _receipt_pairs(payload):
                add(
                    run_root=run_root,
                    logical_job_id=logical,
                    slurm_job_id=slurm_id,
                    kind="execution_receipt",
                    path=receipt_path,
                    detail=pointer,
                )

        logs = run_root / "logs"
        if logs.is_dir():
            for log_path in sorted(logs.glob("*.log")):
                match = _SLURM_LOG.fullmatch(log_path.name)
                if match is None:
                    continue
                add(
                    run_root=run_root,
                    logical_job_id=match.group("logical_job_id"),
                    slurm_job_id=match.group("slurm_job_id"),
                    kind="resolved_slurm_log",
                    path=log_path,
                    detail=(
                        "filename"
                        if match.group("array_task_id") is None
                        else f"filename_array_task={match.group('array_task_id')}"
                    ),
                )

    grouped: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in evidence_by_key:
        grouped[(key[0], key[1])].append(key)

    attempts: list[dict[str, Any]] = []
    for group_key in sorted(grouped):
        keys = sorted(grouped[group_key], key=lambda item: int(item[2]))
        for index, key in enumerate(keys, start=1):
            run_root, logical, slurm_id = key
            attempts.append(
                {
                    "attempt_uid": f"{run_root}::{logical}::{slurm_id}",
                    "run_root": run_root,
                    "logical_job_id": logical,
                    "slurm_job_id": slurm_id,
                    "attempt_index": index,
                    "attempt_count": len(keys),
                    "is_current_attempt": key in current_attempts,
                    "discovery_provenance_class": "observed",
                    "discovery_sources": sorted(
                        evidence_by_key[key].values(),
                        key=lambda row: (
                            row["kind"],
                            row["path"],
                            row["detail"],
                        ),
                    ),
                }
            )
    return attempts


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )


def query_sacct_json(
    job_ids: Sequence[str | int],
    *,
    batch_size: int = 200,
    runner: Callable[[Sequence[str]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Query allocation records with ``sacct --json`` in bounded batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ids = sorted(
        {
            numeric
            for value in job_ids
            if (numeric := _numeric_job_id(value)) is not None
        },
        key=int,
    )
    if len(ids) != len({str(value) for value in job_ids}):
        invalid = sorted(
            {str(value) for value in job_ids if _numeric_job_id(value) is None}
        )
        if invalid:
            raise SlurmAccountingError(
                f"non-numeric Slurm job IDs cannot be queried: {invalid}"
            )
    execute = runner or _default_runner
    payloads: list[dict[str, Any]] = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        command = ["sacct", "--json", "-X", "--jobs", ",".join(batch)]
        result = execute(command)
        if isinstance(result, Mapping):
            payload = dict(result)
        else:
            returncode = int(getattr(result, "returncode", 0))
            stdout = str(
                getattr(result, "stdout", result if isinstance(result, str) else "")
            )
            stderr = str(getattr(result, "stderr", ""))
            if returncode != 0:
                raise SlurmAccountingError(
                    f"sacct failed for {batch[0]}..{batch[-1]}: "
                    f"{(stderr or stdout).strip()}"
                )
            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError as exc:
                raise SlurmAccountingError(
                    f"sacct returned invalid JSON for {batch[0]}..{batch[-1]}: {exc}"
                ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
            raise SlurmAccountingError(
                f"sacct JSON lacks a jobs list for {batch[0]}..{batch[-1]}"
            )
        if payload.get("errors"):
            raise SlurmAccountingError(
                f"sacct JSON reports errors for {batch[0]}..{batch[-1]}: "
                f"{payload['errors']}"
            )
        payloads.append(payload)
    return payloads


def _number(value: Any) -> float | int | None:
    if isinstance(value, Mapping):
        if value.get("infinite") is True or value.get("set") is False:
            return None
        value = value.get("number")
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _epoch(value: Any) -> int | None:
    number = _number(value)
    if number is None or number <= 0:
        return None
    return int(number)


def _duration(value: Any) -> float | int | None:
    number = _number(value)
    if number is None or number < 0:
        return None
    return number


def _utc_string(epoch: int | None) -> str | None:
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except (OverflowError, OSError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _tres_map(value: Any) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    if not isinstance(value, list):
        return result
    for item in value:
        if not isinstance(item, Mapping):
            continue
        tres_type = str(item.get("type", "")).strip()
        name = str(item.get("name", "")).strip()
        count = _number(item.get("count"))
        if not tres_type or count is None:
            continue
        key = f"{tres_type}/{name}" if name else tres_type
        prior = result.get(key, 0)
        total = float(prior) + float(count)
        result[key] = int(total) if total.is_integer() else total
    return dict(sorted(result.items()))


def _gpu_counts(
    tres: Mapping[str, float | int],
) -> tuple[float | int, dict[str, float | int]]:
    typed: dict[str, float | int] = {}
    for prefix in ("gres/gpu:", "gpu/"):
        for key, value in tres.items():
            if key.startswith(prefix):
                typed[key.removeprefix(prefix)] = value
    generic = tres.get("gres/gpu", tres.get("gpu"))
    if generic is not None:
        total = generic
    else:
        total_float = sum((float(value) for value in typed.values()), 0.0)
        total = int(total_float) if total_float.is_integer() else total_float
    return total, dict(sorted(typed.items()))


def _nested(mapping: Any, *keys: str) -> Any:
    value = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _dependency_spec(submit_line: Any) -> str | None:
    text = str(submit_line or "").strip()
    if not text:
        return None
    try:
        parts = shlex.split(text)
    except ValueError:
        parts = text.split()
    for index, part in enumerate(parts):
        if part.startswith("--dependency="):
            return part.split("=", 1)[1] or None
        if part == "--dependency" and index + 1 < len(parts):
            return parts[index + 1] or None
    return None


def _difference(
    later: int | None,
    earlier: int | None,
    *,
    label: str,
    warnings: list[str],
) -> int | None:
    if later is None or earlier is None:
        return None
    difference = later - earlier
    if difference < 0:
        warnings.append(f"{label} is negative ({difference}s)")
        return None
    return difference


def _exit_fields(value: Any) -> tuple[str | None, int | None, int | None, str | None]:
    if not isinstance(value, Mapping):
        return None, None, None, None
    statuses = _string_list(value.get("status"))
    return_code = _number(_nested(value, "return_code"))
    signal_id = _number(_nested(value, "signal", "id"))
    signal_name = _nested(value, "signal", "name")
    code = None
    if return_code is not None or signal_id is not None:
        code = f"{int(return_code or 0)}:{int(signal_id or 0)}"
    return (
        code,
        int(return_code) if return_code is not None else None,
        int(signal_id) if signal_id is not None else None,
        str(signal_name) if signal_name else (statuses[0] if statuses else None),
    )


def normalize_sacct_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one Slurm JSON allocation record and retain it verbatim."""

    job_id = _numeric_job_id(raw.get("job_id"))
    if job_id is None:
        raise SlurmAccountingError(
            f"sacct job record lacks a numeric job_id: {raw.get('job_id')!r}"
        )

    array_parent_number = _number(_nested(raw, "array", "job_id"))
    array_task_set = _nested(raw, "array", "task_id", "set") is True
    array_task_number = (
        _number(_nested(raw, "array", "task_id", "number")) if array_task_set else None
    )
    array_parent = (
        str(int(array_parent_number))
        if array_parent_number is not None and array_parent_number > 0
        else None
    )
    array_task = str(int(array_task_number)) if array_task_number is not None else None
    record_id = (
        f"{array_parent}_{array_task}"
        if array_parent is not None and array_task is not None
        else job_id
    )

    time_record = raw.get("time") if isinstance(raw.get("time"), Mapping) else {}
    submit_epoch = _epoch(_nested(time_record, "submission"))
    eligible_epoch = _epoch(_nested(time_record, "eligible"))
    start_epoch = _epoch(_nested(time_record, "start"))
    end_epoch = _epoch(_nested(time_record, "end"))
    elapsed_seconds = _duration(_nested(time_record, "elapsed"))
    planned_wait_seconds = _duration(_nested(time_record, "planned"))
    warnings: list[str] = []

    pre_eligibility_wait = _difference(
        eligible_epoch,
        submit_epoch,
        label="pre-eligibility wait",
        warnings=warnings,
    )
    queue_wait = _difference(
        start_epoch,
        eligible_epoch,
        label="scheduler queue wait",
        warnings=warnings,
    )
    total_wait = _difference(
        start_epoch,
        submit_epoch,
        label="total wait",
        warnings=warnings,
    )
    wall_clock_runtime = _difference(
        end_epoch,
        start_epoch,
        label="wall-clock runtime",
        warnings=warnings,
    )

    dependency = _dependency_spec(raw.get("submit_line"))
    dependency_ids = (
        re.findall(r"(?<![0-9])[0-9]+(?:_[0-9]+)?", dependency)
        if dependency is not None
        else []
    )
    dependency_wait = pre_eligibility_wait if dependency is not None else None

    allocated_tres = _tres_map(_nested(raw, "tres", "allocated"))
    requested_tres = _tres_map(_nested(raw, "tres", "requested"))
    allocated_gpus, allocated_gpu_types = _gpu_counts(allocated_tres)
    requested_gpus, requested_gpu_types = _gpu_counts(requested_tres)
    allocated_gpu_hours = (
        float(elapsed_seconds) * float(allocated_gpus) / 3600.0
        if elapsed_seconds is not None
        else None
    )

    states = _string_list(_nested(raw, "state", "current"))
    state = states[0].split()[0].rstrip("+") if states else ""
    reason = _nested(raw, "state", "reason")
    exit_code, return_code, signal_id, exit_status = _exit_fields(raw.get("exit_code"))
    derived_exit_code, derived_return_code, derived_signal_id, derived_status = (
        _exit_fields(raw.get("derived_exit_code"))
    )
    reservation = (
        raw.get("reservation") if isinstance(raw.get("reservation"), Mapping) else {}
    )

    normalized: dict[str, Any] = {
        "slurm_record_id": record_id,
        "slurm_job_id": job_id,
        "array_parent_slurm_job_id": array_parent,
        "array_task_id": array_task,
        "job_name": str(raw.get("name", "")),
        "user": str(raw.get("user", "")),
        "account": str(raw.get("account", "")),
        "cluster": str(raw.get("cluster", "")),
        "partition": str(raw.get("partition", "")),
        "qos": str(raw.get("qos", "")),
        "qos_requested": str(raw.get("qosreq", "")),
        "reservation_name": str(reservation.get("name", "")),
        "reservation_requested": str(reservation.get("requested", "")),
        "node_list": str(raw.get("nodes", "")),
        "node_count": _number(raw.get("allocation_nodes")),
        "constraints": str(raw.get("constraints", "")),
        "working_directory": str(raw.get("working_directory", "")),
        "stdout_path": str(raw.get("stdout_expanded", raw.get("stdout", ""))),
        "state": state,
        "state_flags": states,
        "state_reason": str(reason or ""),
        "exit_code": exit_code,
        "exit_return_code": return_code,
        "exit_signal_id": signal_id,
        "exit_status": exit_status,
        "derived_exit_code": derived_exit_code,
        "derived_exit_return_code": derived_return_code,
        "derived_exit_signal_id": derived_signal_id,
        "derived_exit_status": derived_status,
        "submit_time_epoch_utc": submit_epoch,
        "eligible_time_epoch_utc": eligible_epoch,
        "start_time_epoch_utc": start_epoch,
        "end_time_epoch_utc": end_epoch,
        "submit_time_utc": _utc_string(submit_epoch),
        "eligible_time_utc": _utc_string(eligible_epoch),
        "start_time_utc": _utc_string(start_epoch),
        "end_time_utc": _utc_string(end_epoch),
        "elapsed_seconds": elapsed_seconds,
        "slurm_planned_wait_seconds": planned_wait_seconds,
        "pre_eligibility_wait_seconds": pre_eligibility_wait,
        "dependency_wait_seconds": dependency_wait,
        "scheduler_queue_wait_seconds": queue_wait,
        "total_wait_seconds": total_wait,
        "wall_clock_runtime_seconds": wall_clock_runtime,
        "dependency_spec": dependency,
        "dependency_job_ids": dependency_ids,
        "allocated_tres": allocated_tres,
        "requested_tres": requested_tres,
        "allocated_cpus": allocated_tres.get("cpu"),
        "requested_cpus": requested_tres.get("cpu"),
        "allocated_memory_mib": allocated_tres.get("mem"),
        "requested_memory_mib": requested_tres.get("mem"),
        "allocated_gpus": allocated_gpus,
        "requested_gpus": requested_gpus,
        "allocated_gpu_type_counts": allocated_gpu_types,
        "requested_gpu_type_counts": requested_gpu_types,
        "allocated_gpu_hours": allocated_gpu_hours,
        "normalization_warnings": warnings,
        "raw_sacct_record": dict(raw),
    }

    observed = {
        "slurm_record_id",
        "slurm_job_id",
        "array_parent_slurm_job_id",
        "array_task_id",
        "job_name",
        "user",
        "account",
        "cluster",
        "partition",
        "qos",
        "reservation_name",
        "node_list",
        "node_count",
        "working_directory",
        "stdout_path",
        "state",
        "state_flags",
        "state_reason",
        "exit_code",
        "exit_return_code",
        "exit_signal_id",
        "exit_status",
        "derived_exit_code",
        "derived_exit_return_code",
        "derived_exit_signal_id",
        "derived_exit_status",
        "submit_time_epoch_utc",
        "eligible_time_epoch_utc",
        "start_time_epoch_utc",
        "end_time_epoch_utc",
        "elapsed_seconds",
        "slurm_planned_wait_seconds",
        "allocated_tres",
        "allocated_cpus",
        "allocated_memory_mib",
        "allocated_gpus",
        "allocated_gpu_type_counts",
        "raw_sacct_record",
    }
    configured = {
        "qos_requested",
        "reservation_requested",
        "constraints",
        "dependency_spec",
        "dependency_job_ids",
        "requested_tres",
        "requested_cpus",
        "requested_memory_mib",
        "requested_gpus",
        "requested_gpu_type_counts",
    }
    derived = set(normalized) - observed - configured - {"field_provenance"}
    normalized["field_provenance"] = {
        key: (
            "observed"
            if key in observed
            else "configured"
            if key in configured
            else "derived"
        )
        for key in normalized
    }
    if not derived:
        raise AssertionError("normalization must include derived fields")
    return normalized


def _normalized_records(
    payloads: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for payload in payloads:
        if payload.get("errors"):
            raise SlurmAccountingError(
                f"injected sacct JSON reports errors: {payload['errors']}"
            )
        jobs = payload.get("jobs")
        if not isinstance(jobs, list):
            raise SlurmAccountingError("injected sacct JSON lacks a jobs list")
        for raw in jobs:
            if not isinstance(raw, Mapping):
                raise SlurmAccountingError("sacct jobs must contain objects")
            records.append(normalize_sacct_record(raw))
    return records


def collect_slurm_accounting(
    run_roots: Sequence[Path | str],
    *,
    injected_sacct_payloads: Sequence[Mapping[str, Any]] | None = None,
    runner: Callable[[Sequence[str]], Any] | None = None,
    batch_size: int = 200,
    missing_policy: str = "error",
) -> dict[str, Any]:
    """Discover attempts, acquire accounting, and return a self-contained report.

    ``missing_policy="error"`` is the publication-release default.  The
    ``"flag"`` mode returns an incomplete report so historical gaps can be
    documented explicitly rather than silently dropped.
    """

    if missing_policy not in {"error", "flag"}:
        raise ValueError("missing_policy must be 'error' or 'flag'")
    attempts = discover_slurm_attempts(run_roots)
    requested_ids = sorted(
        {attempt["slurm_job_id"] for attempt in attempts},
        key=int,
    )
    if injected_sacct_payloads is None:
        payloads = query_sacct_json(
            requested_ids,
            batch_size=batch_size,
            runner=runner,
        )
        acquisition = "sacct_json"
    else:
        payloads = [dict(payload) for payload in injected_sacct_payloads]
        acquisition = "injected_sacct_json"

    normalized_records = _normalized_records(payloads)
    by_record_id: dict[str, dict[str, Any]] = {}
    tasks_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in normalized_records:
        record_id = str(record["slurm_record_id"])
        if record_id in by_record_id:
            if (
                by_record_id[record_id]["raw_sacct_record"]
                != record["raw_sacct_record"]
            ):
                raise SlurmAccountingError(f"conflicting sacct records for {record_id}")
            continue
        by_record_id[record_id] = record
        parent = record.get("array_parent_slurm_job_id")
        if parent:
            tasks_by_parent[str(parent)].append(record)

    missing: list[dict[str, str]] = []
    collected_attempts: list[dict[str, Any]] = []
    for attempt in attempts:
        slurm_id = str(attempt["slurm_job_id"])
        parent = by_record_id.get(slurm_id)
        tasks = sorted(
            tasks_by_parent.get(slurm_id, []),
            key=lambda row: int(str(row["array_task_id"])),
        )
        observed = parent is not None or bool(tasks)
        if not observed:
            missing.append(
                {
                    "attempt_uid": str(attempt["attempt_uid"]),
                    "logical_job_id": str(attempt["logical_job_id"]),
                    "slurm_job_id": slurm_id,
                    "run_root": str(attempt["run_root"]),
                }
            )
        allocation_rows = tasks if tasks else ([parent] if parent is not None else [])
        allocated_gpu_hours = sum(
            float(row.get("allocated_gpu_hours") or 0.0) for row in allocation_rows
        )
        collected_attempts.append(
            {
                **attempt,
                "accounting_status": "observed" if observed else "missing",
                "accounting_status_provenance_class": "derived",
                "allocation_accounting_source": (
                    "array_tasks" if tasks else "parent" if parent is not None else None
                ),
                "allocated_gpu_hours": allocated_gpu_hours if observed else None,
                "allocated_gpu_hours_provenance_class": (
                    "derived" if observed else None
                ),
                "parent_record": parent,
                "array_task_records": tasks,
            }
        )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete" if not missing else "incomplete",
        "provenance_classes": list(PROVENANCE_CLASSES),
        "collection": {
            "acquisition": acquisition,
            "sacct_command_contract": "sacct --json -X --jobs ID[,ID...]",
            "batch_size": batch_size,
            "raw_batches_retained": True,
            "missing_policy": missing_policy,
            "collected_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        },
        "run_roots": sorted({attempt["run_root"] for attempt in attempts}),
        "counts": {
            "attempts": len(attempts),
            "accounted_attempts": len(attempts) - len(missing),
            "missing_attempts": len(missing),
            "normalized_sacct_records": len(normalized_records),
            "raw_sacct_batches": len(payloads),
        },
        "attempts": collected_attempts,
        "missing_accounting": missing,
        "unmatched_sacct_records": sorted(
            set(by_record_id)
            - set(requested_ids)
            - {
                str(record["slurm_record_id"])
                for rows in tasks_by_parent.values()
                for record in rows
            }
        ),
        "raw_sacct_batches": payloads,
    }
    if missing and missing_policy == "error":
        preview = ", ".join(row["slurm_job_id"] for row in missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise SlurmAccountingError(
            f"missing sacct accounting for {len(missing)} attempts: {preview}{suffix}",
            report=report,
        )
    return report
