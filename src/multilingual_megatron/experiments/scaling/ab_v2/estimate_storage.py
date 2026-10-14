"""Conservative full-state checkpoint storage estimate for A/B-v2."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .common import EXPERTS, LANGUAGE_COUNTS, WorkflowError
from .estimates import (
    _full_state_checkpoint_count,
    _model_parameter_upper_bound,
    _planned_checkpoint_count_for_job,
)
from .variant import active_policy


def _storage_estimate(
    jobs: Sequence[Mapping[str, Any]], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    policy = runtime.get("storage")
    if not isinstance(policy, dict):
        raise WorkflowError("runtime storage policy must be an object")
    bytes_per_parameter = int(policy["checkpoint_bytes_per_parameter"])
    overhead = float(policy["checkpoint_serialization_overhead_factor"])
    non_checkpoint = int(policy["non_checkpoint_peak_reserve_bytes"])
    free_headroom = int(policy["post_allocation_free_headroom_bytes"])
    if (
        bytes_per_parameter <= 0
        or overhead < 1
        or non_checkpoint < 3 * 1024**4
        or free_headroom < 3 * 1024**4
    ):
        raise WorkflowError(
            "storage policy must reserve positive checkpoint bytes and at least 3 TiB each "
            "for retained data and post-allocation free headroom"
        )

    checkpoint_bytes = {
        (ladder, experts): int(
            math.ceil(
                _model_parameter_upper_bound(ladder, experts)
                * bytes_per_parameter
                * overhead
            )
        )
        for ladder in active_policy().ladders
        for experts in EXPERTS
    }
    sentinel_peak = 0
    lr_verification_peak = 0
    lr_diagnostics_peak = 0
    for row in jobs:
        arguments = row.get("arguments", {})
        if row.get("action") == "training_sentinel":
            sentinel_peak += checkpoint_bytes[
                (str(arguments["ladder"]), int(arguments["experts"]))
            ]
        if row.get("action") in {
            "fixed_lr_schedule_probe",
            "prescribed_lr_schedule_probe",
        }:
            lr_verification_peak += checkpoint_bytes[
                (str(arguments["ladder"]), int(arguments["experts"]))
            ]
        if row.get("action") == "production_train" and arguments.get(
            "purpose"
        ) == "lr_diagnostic":
            lr_diagnostics_peak += _planned_checkpoint_count_for_job(
                row, runtime
            ) * checkpoint_bytes[
                (str(arguments["ladder"]), int(arguments["experts"]))
            ]

    production_peak = 0
    production_checkpoint_counts: dict[str, Any] = {}
    production_wave_peaks: dict[str, Any] = {}
    production_wave_ids = {
        str(row["job_id"])
        for row in jobs
        if row.get("arguments", {}).get("production_wave") is not None
    }
    production_training_jobs = [
        row
        for row in jobs
        if row.get("action") == "production_train"
        and row.get("arguments", {}).get("purpose") != "lr_diagnostic"
    ]
    retained_final_bytes = 0
    if production_training_jobs and not runtime.get("probe_only_unmeasured"):
        for wave_index, (ladder, experts) in enumerate(
            (
                (ladder, experts)
                for ladder in active_policy().ladders
                for experts in EXPERTS
            ),
            start=1,
        ):
            plan_count = _full_state_checkpoint_count(
                runtime,
                ladder=ladder,
                experts=experts,
                horizons=runtime["training"]["checkpoint_horizons_tokens"][ladder],
            )
            key = f"{ladder}/E{experts}"
            production_checkpoint_counts[key] = plan_count
            transient = (
                len(LANGUAGE_COUNTS)
                * plan_count["total"]
                * checkpoint_bytes[(ladder, experts)]
            )
            if production_wave_ids:
                wave_peak = retained_final_bytes + transient
                production_wave_peaks[key] = {
                    "index": wave_index,
                    "retained_before_bytes": retained_final_bytes,
                    "transient_wave_bytes": transient,
                    "peak_bytes": wave_peak,
                }
                production_peak = max(production_peak, wave_peak)
                retained_final_bytes += (
                    len(LANGUAGE_COUNTS) * checkpoint_bytes[(ladder, experts)]
                )
            else:
                production_peak += transient
    confirmation_final_bytes = sum(
        checkpoint_bytes[(str(row["arguments"]["ladder"]), int(row["arguments"]["experts"]))]
        for row in jobs
        if str(row.get("job_id", "")).startswith("lrconfirm_branch_")
    )
    retain_all = bool(policy.get("retain_all_checkpoints", False))
    checkpoint_peak = (
        sentinel_peak + lr_verification_peak + lr_diagnostics_peak + production_peak
        if retain_all
        else max(sentinel_peak, lr_verification_peak, lr_diagnostics_peak, production_peak)
    )
    required_free = checkpoint_peak + non_checkpoint + free_headroom
    return {
        "schema_version": 2,
        "method": "conservative_full_state_parameter_bytes_with_phase_barriers",
        "checkpoint_bytes_per_parameter": bytes_per_parameter,
        "checkpoint_serialization_overhead_factor": overhead,
        "per_checkpoint_bytes": {
            f"{ladder}/E{experts}": checkpoint_bytes[(ladder, experts)]
            for ladder in active_policy().ladders
            for experts in EXPERTS
        },
        "phase_checkpoint_peak_bytes": {
            "sentinel": sentinel_peak,
            "lr_verification": lr_verification_peak,
            "lr_diagnostics": lr_diagnostics_peak,
            "production": production_peak,
        },
        "production_storage_schedule": (
            "sequential_ladder_expert_waves"
            if production_wave_ids
            else (
                "all_trunks_concurrently_eligible_retain_all"
                if retain_all and production_training_jobs
                else "historical_all_cells_before_summary"
            )
        ),
        "production_wave_peak_bytes": production_wave_peaks,
        "retained_largest_d_checkpoint_bytes": retained_final_bytes,
        "retained_lr_confirmation_final_checkpoint_bytes": confirmation_final_bytes,
        "prescribed_lr_schedule_probe_checkpoint_count": len(
            [
                row
                for row in jobs
                if row.get("action") == "prescribed_lr_schedule_probe"
            ]
        ),
        "lr_diagnostic_training_job_count": len(
            [
                row
                for row in jobs
                if row.get("action") == "production_train"
                and row.get("arguments", {}).get("purpose") == "lr_diagnostic"
            ]
        ),
        "production_checkpoint_counts_per_cell": production_checkpoint_counts,
        "checkpoint_peak_bytes": checkpoint_peak,
        "retained_data_and_other_artifacts_reserve_bytes": non_checkpoint,
        "required_post_allocation_free_headroom_bytes": free_headroom,
        "required_available_bytes_at_preflight": required_free,
        "data_retention_policy": "retain all raw/final/tokenized data through result validation",
        "checkpoint_retention_policy": (
            "retain every sentinel, LR probe, trunk, branch, and final checkpoint"
            if retain_all
            else (
            "after each prospective (ladder, E) wave, prune trunks and smaller-D "
            "branches; retain only the largest-D final branch for each cell"
            if production_wave_ids
            else (
                "prune each 1.6B trunk after hash-valid evaluation and retain every "
                "2B LR-confirmation final checkpoint through manual review"
                if policy.get("retain_confirmation_final_checkpoints")
                else "retain only the largest-D final branch for each cell after summary"
            )
            )
        ),
        "go_no_go": "requires_live_statvfs_preflight",
    }
