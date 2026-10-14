"""Shared primitives for constructing Stage 1 workflow jobs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import (
    FIXED_PEAK_LEARNING_RATE,
    WorkflowError,
    _abs,
    _prescribed_peak_learning_rate,
    _rel_or_abs,
)
from .variant import active_variant


def _add_job(
    jobs: list[dict[str, Any]],
    *,
    job_id: str,
    stage: str,
    action: str,
    resource: str,
    dependencies: Sequence[str] = (),
    array: str | None = None,
    arguments: Mapping[str, Any] | None = None,
    outputs: Sequence[Path | str] = (),
    gpu_tokens: int = 0,
) -> str:
    args = dict(arguments or {})
    resolved_resource = active_variant().resource_selector(resource, args)
    if not isinstance(resolved_resource, str) or not resolved_resource.strip():
        raise WorkflowError(f"{job_id} variant resolved an invalid resource")
    if any(row["job_id"] == job_id for row in jobs):
        raise WorkflowError(f"duplicate workflow job ID: {job_id}")
    if any(not isinstance(value, str) or not value.strip() for value in dependencies):
        raise WorkflowError(f"{job_id} contains an empty or non-string dependency")
    jobs.append(
        {
            "job_id": job_id,
            "stage": stage,
            "action": action,
            "resource": resolved_resource,
            "dependencies": list(dependencies),
            "array": array,
            "arguments": args,
            "outputs": [_rel_or_abs(_abs(value)) for value in outputs],
            "gpu_tokens": int(gpu_tokens),
        }
    )
    return job_id


def _training_contract_arguments(
    prescription: Mapping[str, Any] | None,
    *,
    ladder: str,
    experts: int,
    training_seed: int,
    lr_multiplier: float = 1.0,
) -> dict[str, Any]:
    if int(training_seed) <= 0:
        raise WorkflowError("training seed must be positive")
    if prescription is None:
        if not math.isclose(float(lr_multiplier), 1.0, rel_tol=0.0, abs_tol=0.0):
            raise WorkflowError("historical fixed-LR jobs do not allow LR multipliers")
        return {
            "learning_rate_policy": "fixed_across_all_cells",
            "peak_lr": FIXED_PEAK_LEARNING_RATE,
            "model_seed": int(training_seed),
            "dataset_random_seed": int(training_seed),
        }
    base = _prescribed_peak_learning_rate(
        prescription, ladder=ladder, experts=experts
    )
    return {
        "learning_rate_policy": "jointmoe_paper_rounded_coefficients_k_invariant",
        "base_peak_lr": base,
        "lr_multiplier": float(lr_multiplier),
        "peak_lr": _prescribed_peak_learning_rate(
            prescription,
            ladder=ladder,
            experts=experts,
            multiplier=lr_multiplier,
        ),
        "model_seed": int(training_seed),
        "dataset_random_seed": int(training_seed),
    }
