"""Explicit, dependency-safe job selection for workflow submission."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .common import WorkflowError
from .job_validation import _topological


def _ordered_submission_candidates(
    manifest: Mapping[str, Any], selected_job_ids: Iterable[str] | None
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    jobs_by_id = {str(job["job_id"]): job for job in manifest["jobs"]}
    if selected_job_ids is None:
        return _topological(manifest["jobs"]), jobs_by_id

    selected = [str(job_id) for job_id in selected_job_ids]
    if len(selected) != len(set(selected)):
        raise WorkflowError("selected submission order contains duplicate job IDs")
    unknown = [job_id for job_id in selected if job_id not in jobs_by_id]
    if unknown:
        raise WorkflowError(
            "selected submission order contains unknown jobs: "
            + ", ".join(unknown[:8])
        )
    selected_set = set(selected)
    earlier: set[str] = set()
    for job_id in selected:
        dependencies = {
            str(value) for value in jobs_by_id[job_id].get("dependencies", [])
        }
        late = dependencies & selected_set - earlier
        if late:
            raise WorkflowError(
                f"selected submission order places {job_id} before dependencies: "
                + ", ".join(sorted(late))
            )
        earlier.add(job_id)
    return [jobs_by_id[job_id] for job_id in selected], jobs_by_id
