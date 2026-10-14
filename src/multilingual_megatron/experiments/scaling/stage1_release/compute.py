"""Classify and summarize physical scheduler compute for Stage 1 A--D."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

from multilingual_megatron.experiments.scaling.stage1_release.contracts import (
    Stage1ReleaseError,
    _json_cell,
)


_SYSTEM_PATTERN = re.compile(
    r"(probe|throughput|profile|nsys|sentinel|health|preflight|smoke)",
    re.IGNORECASE,
)
_DATA_PATTERN = re.compile(
    r"(sample|token|audit|corpus|data[_-]|materiali[sz]e|prepare)",
    re.IGNORECASE,
)
_CONTROL_PATTERN = re.compile(
    r"(control|ledger|summary|manifest|plan|freeze|cancel|guard)",
    re.IGNORECASE,
)


def _candidate_run_root(path_text: str) -> str | None:
    path = Path(path_text).resolve()
    parts = path.parts
    if "pipeline_runs" not in parts:
        return None
    index = parts.index("pipeline_runs")
    if index + 1 >= len(parts):
        return None
    return str(Path(*parts[: index + 2]))


def _category(
    *,
    logical_job_id: str,
    run_root: str,
    selected_validation: set[tuple[str, str]],
) -> str:
    if logical_job_id.startswith(("prod_trunk_", "prod_branch_")):
        return "core_training"
    if logical_job_id.startswith("prod_eval_"):
        key = (str(Path(run_root).resolve()), logical_job_id)
        return (
            "selected_validation"
            if key in selected_validation
            else "duplicate_or_recovery_validation"
        )
    if _SYSTEM_PATTERN.search(logical_job_id):
        return "probe_or_systems_optimization"
    if _DATA_PATTERN.search(logical_job_id):
        return "data_preparation"
    if _CONTROL_PATTERN.search(logical_job_id):
        return "cpu_control"
    return "other"


def classify_compute_records(
    *,
    attempts: Sequence[Mapping[str, Any]],
    allocations: Sequence[Mapping[str, Any]],
    validation_candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach mutually exclusive workload categories to accounting records."""

    selected_validation: set[tuple[str, str]] = set()
    for candidate in validation_candidates:
        if not candidate.get("selected"):
            continue
        run_root = _candidate_run_root(
            str(candidate.get("candidate_manifest_path", ""))
        )
        if run_root is not None:
            selected_validation.add((run_root, str(candidate["readout_id"])))

    classified_attempts: list[dict[str, Any]] = []
    category_by_attempt: dict[str, str] = {}
    for source in attempts:
        row = dict(source)
        category = _category(
            logical_job_id=str(row["logical_job_id"]),
            run_root=str(row["run_root"]),
            selected_validation=selected_validation,
        )
        row["workload_category"] = category
        row["workload_category_provenance_class"] = "derived"
        row["workload_category_method"] = (
            "Deterministic logical-job prefix and selected-candidate run-root "
            "join; categories describe workload purpose, not scientific "
            "independence."
        )
        attempt_uid = str(row["attempt_uid"])
        category_by_attempt[attempt_uid] = category
        classified_attempts.append(row)

    classified_allocations: list[dict[str, Any]] = []
    for source in allocations:
        row = dict(source)
        attempt_uid = str(row["attempt_uid"])
        if attempt_uid not in category_by_attempt:
            raise Stage1ReleaseError(
                f"Allocation has no classified attempt: {attempt_uid}"
            )
        row["workload_category"] = category_by_attempt[attempt_uid]
        row["workload_category_provenance_class"] = "derived"
        classified_allocations.append(row)
    return classified_attempts, classified_allocations


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, Mapping) else {}
    return {}


def _reconcile_workload_categories(categories: Sequence[str], *, label: str) -> str:
    unique = {str(category) for category in categories}
    informative = unique - {"other"}
    if not informative:
        return "other"
    if len(informative) == 1:
        return next(iter(informative))
    validation = {"selected_validation", "duplicate_or_recovery_validation"}
    if informative.issubset(validation):
        return (
            "selected_validation"
            if "selected_validation" in informative
            else "duplicate_or_recovery_validation"
        )
    raise Stage1ReleaseError(
        f"Conflicting workload categories for physical {label}: {sorted(unique)}"
    )


def build_physical_compute_category_summary(
    *,
    attempts: Sequence[Mapping[str, Any]],
    allocations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize physical records once, never once per evidence copy/readout."""

    unique_allocations: dict[str, dict[str, Any]] = {}
    for source in allocations:
        record_id = str(source["slurm_record_id"])
        row = dict(source)
        prior = unique_allocations.get(record_id)
        if prior is not None:
            identity_fields = (
                "slurm_job_id",
                "array_task_id",
                "state",
                "elapsed_seconds",
                "allocated_gpus",
                "allocated_gpu_hours",
            )
            if any(prior.get(key) != row.get(key) for key in identity_fields):
                raise Stage1ReleaseError(
                    f"Conflicting physical Slurm record {record_id}"
                )
            categories = [
                *prior.get("_evidence_categories", [prior["workload_category"]]),
                str(row["workload_category"]),
            ]
            prior["_evidence_categories"] = categories
            prior["_evidence_mapping_count"] = (
                int(prior.get("_evidence_mapping_count", 1)) + 1
            )
            prior["workload_category"] = _reconcile_workload_categories(
                categories,
                label=f"Slurm record {record_id}",
            )
            continue
        row["_evidence_categories"] = [str(row["workload_category"])]
        row["_evidence_mapping_count"] = 1
        unique_allocations[record_id] = row

    unique_attempts: dict[str, dict[str, Any]] = {}
    for source in attempts:
        attempt_key = str(source.get("slurm_job_id") or source["attempt_uid"])
        row = dict(source)
        prior = unique_attempts.get(attempt_key)
        if prior is None:
            row["_evidence_categories"] = [str(row["workload_category"])]
            row["_evidence_mapping_count"] = 1
            unique_attempts[attempt_key] = row
            continue
        categories = [
            *prior.get("_evidence_categories", [prior["workload_category"]]),
            str(row["workload_category"]),
        ]
        prior["_evidence_categories"] = categories
        prior["_evidence_mapping_count"] = (
            int(prior.get("_evidence_mapping_count", 1)) + 1
        )
        prior["workload_category"] = _reconcile_workload_categories(
            categories,
            label=f"Slurm attempt {attempt_key}",
        )
        if row.get("accounting_status") == "observed":
            prior["accounting_status"] = "observed"

    attempts_by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in unique_attempts.values():
        attempts_by_category[str(row["workload_category"])].append(row)
    allocations_by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in unique_allocations.values():
        allocations_by_category[str(row["workload_category"])].append(row)

    categories = sorted(set(attempts_by_category) | set(allocations_by_category))
    rows: list[dict[str, Any]] = []
    for category in [*categories, "all_physical_records"]:
        category_attempts = (
            list(unique_attempts.values())
            if category == "all_physical_records"
            else attempts_by_category.get(category, [])
        )
        category_allocations = (
            list(unique_allocations.values())
            if category == "all_physical_records"
            else allocations_by_category.get(category, [])
        )
        states: dict[str, int] = defaultdict(int)
        gpu_types: dict[str, int] = defaultdict(int)
        for allocation in category_allocations:
            states[str(allocation.get("state") or "UNKNOWN")] += 1
            for name, count in _as_mapping(
                allocation.get("allocated_gpu_type_counts")
            ).items():
                gpu_types[str(name)] += int(count)
        successful = [
            row
            for row in category_allocations
            if str(row.get("state", "")).upper() == "COMPLETED"
        ]
        rows.append(
            {
                "workload_category": category,
                "discovered_attempts": len(category_attempts),
                "attempt_evidence_mappings": sum(
                    int(row.get("_evidence_mapping_count", 1))
                    for row in category_attempts
                ),
                "duplicate_attempt_evidence_mappings": sum(
                    int(row.get("_evidence_mapping_count", 1)) - 1
                    for row in category_attempts
                ),
                "accounted_attempts": sum(
                    row.get("accounting_status") == "observed"
                    for row in category_attempts
                ),
                "missing_accounting_attempts": sum(
                    row.get("accounting_status") != "observed"
                    for row in category_attempts
                ),
                "unique_physical_allocation_records": len(category_allocations),
                "allocation_evidence_mappings": sum(
                    int(row.get("_evidence_mapping_count", 1))
                    for row in category_allocations
                ),
                "duplicate_allocation_evidence_mappings": sum(
                    int(row.get("_evidence_mapping_count", 1)) - 1
                    for row in category_allocations
                ),
                "allocated_gpu_hours_all_states": sum(
                    float(row.get("allocated_gpu_hours") or 0.0)
                    for row in category_allocations
                ),
                "allocated_gpu_hours_completed": sum(
                    float(row.get("allocated_gpu_hours") or 0.0) for row in successful
                ),
                "allocated_gpu_hours_noncompleted": sum(
                    float(row.get("allocated_gpu_hours") or 0.0)
                    for row in category_allocations
                    if row not in successful
                ),
                "allocated_cpu_hours_all_states": sum(
                    float(row.get("allocated_cpus") or 0.0)
                    * float(row.get("elapsed_seconds") or 0.0)
                    / 3600.0
                    for row in category_allocations
                ),
                "states_json": _json_cell(dict(sorted(states.items()))),
                "allocated_gpu_type_counts_json": _json_cell(
                    dict(sorted(gpu_types.items()))
                ),
                "deduplication_key": "slurm_record_id",
                "category_reconciliation": (
                    "Identical physical records discovered through multiple roots "
                    "are counted once. Generic 'other' evidence defers to one "
                    "specific category; selected/recovery validation resolves to "
                    "selected. Other cross-purpose conflicts fail closed."
                ),
                "accounting_semantics": (
                    "Each normalized physical job or array-task record is "
                    "counted once across all evidence paths. Allocated hours "
                    "measure allocations, not application utilization."
                ),
                "provenance_class": "derived",
            }
        )
    return rows


__all__ = [
    "build_physical_compute_category_summary",
    "classify_compute_records",
]
