"""Validate immutable A/B-v2 DAG, LR, and checkpoint-retention contracts."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .common import (
    EXPERTS,
    FIXED_PEAK_LEARNING_RATE,
    JOINTMOE_PAPER_LR_ACTIVE_COEFFICIENT,
    JOINTMOE_PAPER_LR_EXPERT_COEFFICIENT,
    JOINTMOE_PAPER_LR_INTERCEPT,
    LANGUAGE_COUNTS,
    PRIMARY_TRAINING_SEED,
    REPLICATION_TRAINING_SEEDS,
    WorkflowError,
    _prescribed_peak_learning_rate,
)
from .jobs import _production_wave_id
from .variant import active_policy, active_variant


def _validate_dag(jobs: Sequence[Mapping[str, Any]]) -> None:
    by_id = {str(row["job_id"]): row for row in jobs}
    if len(by_id) != len(jobs):
        raise WorkflowError("workflow contains duplicate job IDs")
    for job_id, row in by_id.items():
        for dependency in row.get("dependencies", []):
            if dependency not in by_id:
                raise WorkflowError(f"{job_id} has unknown dependency {dependency}")
        resource = str(row.get("resource", ""))
        if not re.fullmatch(r"(?:cpu(?:_[a-z0-9]+)*|h100(?:_[1-9][0-9]*)?)", resource):
            raise WorkflowError(f"{job_id} has invalid resource class {resource!r}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(job_id: str) -> None:
        if job_id in visited:
            return
        if job_id in visiting:
            raise WorkflowError(f"workflow dependency cycle at {job_id}")
        visiting.add(job_id)
        for dependency in by_id[job_id].get("dependencies", []):
            visit(str(dependency))
        visiting.remove(job_id)
        visited.add(job_id)

    for job_id in by_id:
        visit(job_id)


def _validate_fixed_lr_contract(jobs: Sequence[Mapping[str, Any]]) -> None:
    training_actions = {
        "training_sentinel",
        "fixed_lr_schedule_probe",
        "production_train",
    }
    forbidden_actions = {
        "lr_candidate_train",
        "lr_select",
        "lr_heldout_gate",
        "lr_stability_probe",
        "lr_stability_gate",
        "prescribed_lr_schedule_probe",
        "prescribed_lr_schedule_gate",
    }
    for row in jobs:
        action = str(row.get("action", ""))
        if action in forbidden_actions:
            raise WorkflowError(f"non-historical LR action remains: {action}")
        if action not in training_actions:
            continue
        arguments = row.get("arguments", {})
        try:
            job_lr = float(arguments["peak_lr"])
            model_seed = int(arguments["model_seed"])
            dataset_seed = int(arguments["dataset_random_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(
                f"historical training job {row.get('job_id')} lacks its LR/seed contract"
            ) from exc
        if (
            arguments.get("learning_rate_policy") != "fixed_across_all_cells"
            or not math.isclose(job_lr, FIXED_PEAK_LEARNING_RATE, rel_tol=0.0, abs_tol=0.0)
            or model_seed != PRIMARY_TRAINING_SEED
            or dataset_seed != PRIMARY_TRAINING_SEED
        ):
            raise WorkflowError(f"historical training job {row.get('job_id')} drifted")


def _validate_production_checkpoint_waves(
    jobs: Sequence[Mapping[str, Any]],
) -> None:
    """Verify the storage-bounding production wave chain."""

    by_id = {str(row["job_id"]): row for row in jobs}
    expected_wave_ids = [
        _production_wave_id(ladder, experts)
        for ladder in active_policy().ladders
        for experts in EXPERTS
    ]
    observed_wave_ids = sorted(
        job_id
        for job_id, row in by_id.items()
        if row.get("arguments", {}).get("production_wave") is not None
    )
    if not observed_wave_ids:
        trunk_jobs = {
            job_id: row
            for job_id, row in by_id.items()
            if job_id.startswith("prod_trunk_")
        }
        evaluation_ids = [
            str(row["job_id"])
            for row in jobs
            if str(row["job_id"]).startswith("prod_eval_")
        ]
        if "production_lr_evidence_gate" in by_id:
            expected_entry_dependencies = ["production_lr_evidence_gate"]
        elif "data_megatron_reader_smoke" in by_id:
            expected_entry_dependencies = ["data_megatron_reader_smoke"]
        else:
            expected_entry_dependencies = []
        if not trunk_jobs or any(
            row.get("dependencies") != expected_entry_dependencies
            for row in trunk_jobs.values()
        ):
            raise WorkflowError("retain-all production trunks are not concurrently gated")
        if any(row.get("action") == "checkpoint_prune" for row in jobs):
            raise WorkflowError("retain-all production workflow contains a prune job")
        summary = by_id.get("final_summary", {})
        if (
            summary.get("dependencies") != evaluation_ids
            or summary.get("arguments", {}).get("evaluation_jobs") != evaluation_ids
        ):
            raise WorkflowError("retain-all production summary dependency drifted")
        return
    if sorted(expected_wave_ids) != observed_wave_ids:
        raise WorkflowError("prospective production checkpoint waves drifted")

    previous_gate = "retention_prune_lr_diagnostics"
    all_evaluations: list[str] = []
    horizons = {
        "A": (500_000_000, 1_000_000_000, 2_000_000_000, 4_000_000_000, 8_000_000_000),
        "B": (
            500_000_000,
            1_000_000_000,
            2_000_000_000,
            4_000_000_000,
            8_000_000_000,
            16_000_000_000,
        ),
    }
    for wave_index, (ladder, experts) in enumerate(
        (
            (ladder, experts)
            for ladder in active_policy().ladders
            for experts in EXPERTS
        ),
        start=1,
    ):
        wave_id = _production_wave_id(ladder, experts)
        wave = by_id[wave_id]
        d_max = horizons[ladder][-1]
        expected_evaluations = [
            f"prod_eval_{ladder.lower()}_e{experts}_k{k:03d}_d{d_tokens}"
            for k in LANGUAGE_COUNTS
            for d_tokens in horizons[ladder]
        ]
        expected_sources = [
            job_id
            for k in LANGUAGE_COUNTS
            for job_id in [
                f"prod_trunk_{ladder.lower()}_e{experts}_k{k:03d}",
                *[
                    f"prod_branch_{ladder.lower()}_e{experts}_k{k:03d}_d{d_tokens}"
                    for d_tokens in horizons[ladder]
                    if d_tokens != d_max
                ],
            ]
        ]
        expected_retained = [
            f"prod_branch_{ladder.lower()}_e{experts}_k{k:03d}_d{d_max}"
            for k in LANGUAGE_COUNTS
        ]
        arguments = wave.get("arguments", {})
        if (
            wave.get("action") != "checkpoint_prune"
            or wave.get("stage") != "production"
            or wave.get("dependencies") != expected_evaluations
            or arguments.get("source_jobs") != expected_sources
            or arguments.get("retained_jobs") != expected_retained
            or arguments.get("evidence_jobs") != expected_evaluations
            or arguments.get("production_wave")
            != {"index": wave_index, "ladder": ladder, "experts": experts}
        ):
            raise WorkflowError(f"production checkpoint wave {wave_id} drifted")
        for k in LANGUAGE_COUNTS:
            trunk_id = f"prod_trunk_{ladder.lower()}_e{experts}_k{k:03d}"
            if by_id[trunk_id].get("dependencies") != [previous_gate]:
                raise WorkflowError(f"production wave dependency drifted for {trunk_id}")
        previous_gate = wave_id
        all_evaluations.extend(expected_evaluations)

    summary = by_id.get("final_summary", {})
    if (
        summary.get("dependencies") != [previous_gate]
        or summary.get("arguments", {}).get("evaluation_jobs") != all_evaluations
        or "retention_prune_production_after_summary" in by_id
    ):
        raise WorkflowError("storage-safe production summary dependency drifted")


def _validate_lr_contract(
    jobs: Sequence[Mapping[str, Any]], *, prescription: Mapping[str, Any] | None
) -> None:
    if prescription is None:
        _validate_fixed_lr_contract(jobs)
        return
    expected_coefficients = (
        JOINTMOE_PAPER_LR_INTERCEPT,
        JOINTMOE_PAPER_LR_ACTIVE_COEFFICIENT,
        JOINTMOE_PAPER_LR_EXPERT_COEFFICIENT,
    )
    observed_coefficients = (
        prescription.get("intercept"),
        prescription.get("active_parameter_coefficient"),
        prescription.get("expert_coefficient"),
    )
    if observed_coefficients != expected_coefficients:
        raise WorkflowError("manifest does not use the printed JointMoE coefficients")
    for ladder in active_policy().ladders:
        try:
            count = int(prescription["N_act_nonembed_by_ladder"][ladder])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(f"manifest lacks {ladder} N_act_nonembed") from exc
        for experts in EXPERTS:
            expected = math.exp(
                JOINTMOE_PAPER_LR_INTERCEPT
                + JOINTMOE_PAPER_LR_ACTIVE_COEFFICIENT * math.log(count)
                + JOINTMOE_PAPER_LR_EXPERT_COEFFICIENT * math.log(experts)
            )
            observed = _prescribed_peak_learning_rate(
                prescription, ladder=ladder, experts=experts
            )
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=0.0):
                raise WorkflowError(f"manifest prescribed LR drifted for {ladder}/E{experts}")
    training_actions = {
        "training_sentinel",
        "prescribed_lr_schedule_probe",
        "production_train",
    }
    forbidden_actions = {
        "lr_candidate_train",
        "lr_select",
        "lr_heldout_gate",
        "lr_stability_probe",
        "lr_stability_gate",
        "fixed_lr_schedule_probe",
        "fixed_lr_schedule_gate",
    }
    if any(
        row.get("action") == "production_train"
        and row.get("arguments", {}).get("purpose") != "lr_diagnostic"
        for row in jobs
    ):
        topology_validator = active_variant().production_topology_validator
        if topology_validator is None:
            _validate_production_checkpoint_waves(jobs)
        else:
            topology_validator(jobs)
    for row in jobs:
        action = str(row.get("action", ""))
        if action in forbidden_actions:
            raise WorkflowError(f"obsolete LR calibration action remains: {action}")
        if action not in training_actions:
            continue
        try:
            arguments = row["arguments"]
            ladder = str(arguments["ladder"])
            experts = int(arguments["experts"])
            base_lr = float(arguments["base_peak_lr"])
            multiplier = float(arguments["lr_multiplier"])
            job_lr = float(row["arguments"]["peak_lr"])
            model_seed = int(arguments["model_seed"])
            dataset_seed = int(arguments["dataset_random_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(
                f"training job {row.get('job_id')} lacks its LR/seed contract"
            ) from exc
        expected_base = _prescribed_peak_learning_rate(
            prescription, ladder=ladder, experts=experts
        )
        expected_lr = expected_base * multiplier
        if (
            arguments.get("learning_rate_policy")
            != "jointmoe_paper_rounded_coefficients_k_invariant"
            or not math.isclose(base_lr, expected_base, rel_tol=0.0, abs_tol=0.0)
            or not math.isclose(job_lr, expected_lr, rel_tol=0.0, abs_tol=0.0)
        ):
            raise WorkflowError(
                f"training job {row.get('job_id')} violates the prescribed LR law"
            )
        diagnostic = arguments.get("purpose") == "lr_diagnostic"
        confirmation = (
            diagnostic
            and arguments.get("diagnostic_protocol")
            == "stage1_jointmoe_lr_confirmation_v1"
        )
        if confirmation:
            candidate_id = str(arguments.get("lr_candidate_id", ""))
            candidate_kind = str(arguments.get("lr_candidate_kind", ""))
            factor = arguments.get("jointmoe_factor")
            expected_candidates = {
                "x025": 0.25,
                "x050": 0.5,
                "x100": 1.0,
                "x200": 2.0,
                "x400": 4.0,
            }
            if candidate_id in expected_candidates:
                expected_multiplier = expected_candidates[candidate_id]
                if (
                    candidate_kind != "jointmoe_factor"
                    or factor != expected_multiplier
                    or not math.isclose(
                        multiplier,
                        expected_multiplier,
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                ):
                    raise WorkflowError(
                        f"training job {row.get('job_id')} has invalid factor lineage"
                    )
            elif candidate_id == "fixed2e4":
                if (
                    candidate_kind != "historical_fixed_control"
                    or factor is not None
                    or not math.isclose(job_lr, 2e-4, rel_tol=0.0, abs_tol=0.0)
                ):
                    raise WorkflowError(
                        f"training job {row.get('job_id')} has invalid fixed-control lineage"
                    )
            else:
                raise WorkflowError(
                    f"training job {row.get('job_id')} has unknown LR-confirmation candidate"
                )
            allowed_multipliers = {multiplier}
            allowed_seeds = {PRIMARY_TRAINING_SEED}
        else:
            allowed_multipliers = {0.5, 1.0, 2.0} if diagnostic else {1.0}
            if diagnostic:
                allowed_seeds = set(REPLICATION_TRAINING_SEEDS)
            else:
                configured = active_variant().production_training_seeds
                allowed_seeds = (
                    {PRIMARY_TRAINING_SEED}
                    if configured is None
                    else {int(seed) for seed in configured}
                )
        if multiplier not in allowed_multipliers:
            raise WorkflowError(
                f"training job {row.get('job_id')} has an unauthorized LR multiplier"
            )
        if model_seed != dataset_seed or model_seed not in allowed_seeds:
            raise WorkflowError(
                f"training job {row.get('job_id')} has an unauthorized training seed"
            )


def _topological(jobs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_id = {str(row["job_id"]): row for row in jobs}
    ordered: list[Mapping[str, Any]] = []
    emitted: set[str] = set()
    while len(ordered) < len(jobs):
        ready = [
            row
            for row in jobs
            if str(row["job_id"]) not in emitted
            and set(str(value) for value in row.get("dependencies", [])).issubset(
                emitted
            )
        ]
        if not ready:
            raise WorkflowError("workflow DAG cannot be topologically ordered")
        for row in ready:
            ordered.append(by_id[str(row["job_id"])])
            emitted.add(str(row["job_id"]))
    return ordered
