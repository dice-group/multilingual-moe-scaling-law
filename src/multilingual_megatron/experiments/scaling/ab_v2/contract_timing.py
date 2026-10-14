"""Conservative Slurm walltime policy for frozen workflow manifests."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .common import WorkflowError, _is_h100_resource


def _parse_slurm_duration(value: object) -> int:
    """Parse a Slurm duration into seconds without accepting ambiguous units."""

    raw = str(value).strip()
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", raw)
    if match is None:
        raise WorkflowError(f"invalid Slurm duration {raw!r}; expected [D-]HH:MM:SS")
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    if minutes >= 60 or seconds >= 60 or (days and hours >= 24):
        raise WorkflowError(f"invalid Slurm duration fields in {raw!r}")
    total = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    if total <= 0:
        raise WorkflowError("Slurm duration must be positive")
    return total


def _format_slurm_duration(total_seconds: int) -> str:
    if total_seconds <= 0:
        raise WorkflowError("Slurm duration must be positive")
    days, remainder = divmod(int(total_seconds), 24 * 3600)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _apply_conservative_job_time_limits(
    jobs: Sequence[dict[str, Any]],
    runtime: Mapping[str, Any],
    estimate: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Freeze completion-oriented per-job walltimes from measured forecasts."""

    policy = runtime.get("scheduling")
    if policy is None:
        return None
    if not isinstance(policy, Mapping):
        raise WorkflowError("runtime scheduling policy must be an object")
    if policy.get("mode") == "ordinary_queue_forecast":
        multiplier = float(policy["predicted_time_multiplier"])
        additive_hours = float(policy["additive_safety_hours"])
        minimum_hours = float(policy["minimum_job_hours"])
        maximum_hours = float(policy["maximum_job_hours"])
        quantum_seconds = int(policy["walltime_rounding_minutes"]) * 60
        if (
            multiplier < 1.0
            or additive_hours < 0.0
            or minimum_hours <= 0.0
            or maximum_hours < minimum_hours
            or maximum_hours > 168.0
            or quantum_seconds <= 0
        ):
            raise WorkflowError("ordinary-queue walltime policy is invalid")
        estimates = {
            str(row["job_id"]): row
            for row in estimate.get("job_estimates", [])
            if isinstance(row, Mapping) and row.get("job_id")
        }
        assigned = 0
        requested_gpu_hours = 0.0
        for job in jobs:
            if not _is_h100_resource(job.get("resource")):
                continue
            row = estimates.get(str(job["job_id"]))
            if row is None:
                continue
            gpu_count = int(row["gpu_count"])
            predicted_hours = float(row["predicted_gpu_hours"]) / gpu_count
            requested_hours = max(
                minimum_hours,
                min(maximum_hours, predicted_hours * multiplier + additive_hours),
            )
            requested_seconds = math.ceil(
                requested_hours * 3600 / quantum_seconds
            ) * quantum_seconds
            job["predicted_wall_hours"] = predicted_hours
            job["slurm_time_limit"] = _format_slurm_duration(requested_seconds)
            job["slurm_time_min"] = _format_slurm_duration(requested_seconds)
            assigned += 1
            requested_gpu_hours += requested_seconds / 3600.0 * gpu_count
        return {
            "mode": "ordinary_queue_forecast",
            "policy": "multiplied_prediction_plus_additive_margin_bounded_per_job",
            "predicted_time_multiplier": multiplier,
            "additive_safety_hours": additive_hours,
            "minimum_job_hours": minimum_hours,
            "maximum_job_hours": maximum_hours,
            "walltime_rounding_minutes": quantum_seconds // 60,
            "assigned_gpu_jobs": assigned,
            "maximum_initial_requested_gpu_hours": requested_gpu_hours,
            "automatic_retries": False,
        }
    safety_hours = float(policy.get("desired_walltime_safety_margin_hours", 96.0))
    if safety_hours <= 72.0:
        raise WorkflowError("desired walltime safety margin must be greater than 72 hours")
    training_minimum_margin = float(
        policy.get("training_time_min_safety_margin_hours", 24.0)
    )
    evaluation_minimum_margin = float(
        policy.get("evaluation_time_min_safety_margin_hours", 6.0)
    )
    reservation_tail_mode = bool(policy.get("reservation_tail_time_min", False))
    minimum_training_margin = 1.0 if reservation_tail_mode else 24.0
    minimum_evaluation_margin = 1.0 if reservation_tail_mode else 6.0
    if training_minimum_margin < minimum_training_margin:
        raise WorkflowError(
            "training time-min safety margin is below the active reservation policy"
        )
    if evaluation_minimum_margin < minimum_evaluation_margin:
        raise WorkflowError(
            "evaluation time-min safety margin is below the active reservation policy"
        )
    quantum_seconds = int(policy.get("walltime_rounding_minutes", 15)) * 60
    if quantum_seconds <= 0:
        raise WorkflowError("walltime rounding quantum must be positive")
    maximum_seconds = _parse_slurm_duration(policy.get("partition_max_time", "7-00:00:00"))
    if maximum_seconds > 7 * 24 * 3600:
        raise WorkflowError("gpu_h100 partition maximum is seven days")
    reservation_start = datetime.fromisoformat(str(policy["reservation_start"]))
    reservation_end = datetime.fromisoformat(str(policy["reservation_end"]))
    if reservation_start.tzinfo is None or reservation_end.tzinfo is None:
        raise WorkflowError("reservation timestamps must be timezone-aware")
    if reservation_end <= reservation_start:
        raise WorkflowError("reservation end must follow reservation start")
    if (reservation_end - reservation_start).total_seconds() > maximum_seconds:
        raise WorkflowError("reservation duration exceeds the partition maximum")

    estimates = {
        str(row["job_id"]): row
        for row in estimate.get("job_estimates", [])
        if isinstance(row, Mapping) and row.get("job_id")
    }
    assigned = 0
    capped: list[str] = []
    for job in jobs:
        if not _is_h100_resource(job.get("resource")):
            continue
        row = estimates.get(str(job["job_id"]))
        if row is None:
            continue
        gpu_count = int(row["gpu_count"])
        gpu_hours = float(row["predicted_gpu_hours"])
        predicted_hours = gpu_hours / gpu_count
        requested_seconds = math.ceil(
            (predicted_hours + safety_hours) * 3600 / quantum_seconds
        ) * quantum_seconds
        if requested_seconds > maximum_seconds:
            requested_seconds = maximum_seconds
            capped.append(str(job["job_id"]))
        minimum_margin = (
            evaluation_minimum_margin
            if job.get("action") == "per_language_eval"
            else training_minimum_margin
        )
        minimum_seconds = math.ceil(
            (predicted_hours + minimum_margin) * 3600 / quantum_seconds
        ) * quantum_seconds
        minimum_seconds = min(minimum_seconds, requested_seconds)
        job["predicted_wall_hours"] = predicted_hours
        job["slurm_time_limit"] = _format_slurm_duration(requested_seconds)
        job["slurm_time_min"] = _format_slurm_duration(minimum_seconds)
        assigned += 1
    return {
        "policy": (
            "measured_prediction_plus_desired_margin_with_reservation_safe_time_min"
        ),
        "desired_walltime_safety_margin_hours": safety_hours,
        "training_time_min_safety_margin_hours": training_minimum_margin,
        "evaluation_time_min_safety_margin_hours": evaluation_minimum_margin,
        "reservation_tail_time_min": reservation_tail_mode,
        "walltime_rounding_minutes": quantum_seconds // 60,
        "partition_max_time": _format_slurm_duration(maximum_seconds),
        "reservation_start": reservation_start.astimezone(timezone.utc).isoformat(),
        "reservation_end": reservation_end.astimezone(timezone.utc).isoformat(),
        "assigned_gpu_jobs": assigned,
        "partition_capped_jobs": capped,
    }
