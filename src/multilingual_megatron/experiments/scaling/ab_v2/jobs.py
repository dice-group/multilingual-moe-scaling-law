"""Construct and validate the immutable A/B-v2 job DAG."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

from .common import (
    DATA_ARTIFACT_SEAL_NAME,
    EXPERTS,
    FIXED_LR_SCHEDULE_PROBE_CELLS,
    FIXED_LR_SCHEDULE_PROBE_TOKENS,
    FIXED_LR_SCHEDULE_WARMUP_STEPS,
    FIXED_LR_SCHEDULE_WARMUP_TOKENS,
    FIXED_PEAK_LEARNING_RATE,
    JOINTMOE_PAPER_LR_ACTIVE_COEFFICIENT,
    JOINTMOE_PAPER_LR_EXPERT_COEFFICIENT,
    JOINTMOE_PAPER_LR_INTERCEPT,
    LANGUAGE_COUNTS,
    PRESCRIBED_LR_SCHEDULE_PROBE_TOKENS,
    PRESCRIBED_LR_SCHEDULE_WARMUP_STEPS,
    PRESCRIBED_LR_SCHEDULE_WARMUP_TOKENS,
    PRIMARY_TRAINING_SEED,
    Paths,
    REPLICATION_TRAINING_SEEDS,
    WorkflowError,
    _prescribed_peak_learning_rate,
)
from .estimates import (
    _batch_layout,
    _expected_wsd_branch_endpoint,
)
from .job_core import _add_job, _training_contract_arguments
from .job_data import _add_megatron_data_smoke, _data_jobs
from .job_sentinels import _sentinel_jobs
from .variant import active_policy


def _fixed_lr_verification_jobs(
    jobs: list[dict[str, Any]],
    *,
    sentinel_gate: str,
    paths: Paths,
    runtime: Mapping[str, Any],
) -> str:
    """Build only the verification probes frozen by historical C/D contracts."""

    probe_jobs: list[str] = []
    for ladder, experts, k in active_policy().fixed_lr_schedule_probe_cells:
        probe_id = f"fixed_lr_schedule_probe_{ladder.lower()}_e{experts}_k{k:03d}"
        micro_batch_size, data_parallel_size, gradient_accumulation_steps = _batch_layout(
            runtime,
            ladder=ladder,
            experts=experts,
            global_batch_size=32,
        )
        probe_jobs.append(
            _add_job(
                jobs,
                job_id=probe_id,
                stage="lr_verification",
                action="fixed_lr_schedule_probe",
                resource="h100",
                dependencies=(sentinel_gate,),
                arguments={
                    "ladder": ladder,
                    "experts": experts,
                    "K": k,
                    **_training_contract_arguments(
                        None,
                        ladder=ladder,
                        experts=experts,
                        training_seed=PRIMARY_TRAINING_SEED,
                    ),
                    "tokens": FIXED_LR_SCHEDULE_PROBE_TOKENS,
                    "purpose": "fixed_lr_cross_k_early_schedule_verification",
                    "fit_observation": False,
                    "constant_after_warmup": True,
                    "warmup_tokens": FIXED_LR_SCHEDULE_WARMUP_TOKENS,
                    "warmup_steps": FIXED_LR_SCHEDULE_WARMUP_STEPS,
                    "total_steps": 2_560,
                    "global_batch_size": 32,
                    "micro_batch_size": micro_batch_size,
                    "data_parallel_size": data_parallel_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "data_tier": f"K{k}",
                },
                outputs=(paths.jobs / probe_id / "result.json",),
                gpu_tokens=FIXED_LR_SCHEDULE_PROBE_TOKENS,
            )
        )
    verification_gate = _add_job(
        jobs,
        job_id="fixed_lr_schedule_gate",
        stage="lr_verification",
        action="fixed_lr_schedule_gate",
        resource="cpu",
        dependencies=tuple(probe_jobs),
        arguments={
            "probe_jobs": probe_jobs,
            "peak_lr": FIXED_PEAK_LEARNING_RATE,
            "protocol": (
                "historical fixed 2e-4 peak LR; exact 130,023,424-token linear "
                "warmup then constant stable window; verification only"
            ),
            "fit_observation": False,
            "selection_mutation_allowed": False,
        },
        outputs=(paths.control / "fixed_lr_schedule_gate.json",),
    )
    _add_job(
        jobs,
        job_id="retention_prune_fixed_lr_schedule_probes",
        stage="lr_verification",
        action="checkpoint_prune",
        resource="cpu",
        dependencies=(verification_gate,),
        arguments={
            "source_jobs": probe_jobs,
            "evidence_jobs": [verification_gate],
            "retention_scope": "fixed_lr_schedule_probe_checkpoints_after_gate",
        },
        outputs=(
            paths.control
            / "prune"
            / "retention_prune_fixed_lr_schedule_probes.json",
        ),
    )
    return verification_gate


def _prescribed_lr_verification_jobs(
    jobs: list[dict[str, Any]],
    *,
    sentinel_gate: str,
    paths: Paths,
    lr_prescription: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> str:
    probe_jobs: list[str] = []
    probe_cells = active_policy().prescribed_lr_schedule_probe_cells
    if not probe_cells:
        raise WorkflowError("prescribed-LR workflow has no schedule-probe cells")
    for ladder, experts, k in probe_cells:
        probe_id = f"prescribed_lr_schedule_probe_{ladder.lower()}_e{experts}_k{k:03d}"
        micro_batch_size, data_parallel_size, gradient_accumulation_steps = _batch_layout(
            runtime,
            ladder=ladder,
            experts=experts,
            global_batch_size=32,
        )
        probe_jobs.append(
            _add_job(
                jobs,
                job_id=probe_id,
                stage="lr_verification",
                action="prescribed_lr_schedule_probe",
                resource="h100",
                dependencies=(sentinel_gate,),
                arguments={
                    "ladder": ladder,
                    "experts": experts,
                    "K": k,
                    **_training_contract_arguments(
                        lr_prescription,
                        ladder=ladder,
                        experts=experts,
                        training_seed=PRIMARY_TRAINING_SEED,
                    ),
                    "tokens": PRESCRIBED_LR_SCHEDULE_PROBE_TOKENS,
                    "purpose": "prescribed_lr_cross_k_early_schedule_verification",
                    "fit_observation": False,
                    "constant_after_warmup": True,
                    "warmup_tokens": PRESCRIBED_LR_SCHEDULE_WARMUP_TOKENS,
                    "warmup_steps": PRESCRIBED_LR_SCHEDULE_WARMUP_STEPS,
                    "total_steps": 2_560,
                    "global_batch_size": 32,
                    "micro_batch_size": micro_batch_size,
                    "data_parallel_size": data_parallel_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "data_tier": f"K{k}",
                },
                outputs=(paths.jobs / probe_id / "result.json",),
                gpu_tokens=PRESCRIBED_LR_SCHEDULE_PROBE_TOKENS,
            )
        )
    verification_gate = _add_job(
        jobs,
        job_id="prescribed_lr_schedule_gate",
        stage="lr_verification",
        action="prescribed_lr_schedule_gate",
        resource="cpu",
        dependencies=tuple(probe_jobs),
        arguments={
            "probe_jobs": probe_jobs,
            "protocol": (
                "JointMoE paper-prescribed cell LR; exact 130,023,424-token linear "
                "warmup then constant stable window; LR and K extremes; verification only"
            ),
            "fit_observation": False,
            "selection_mutation_allowed": False,
        },
        outputs=(paths.control / "prescribed_lr_schedule_gate.json",),
    )
    if bool(runtime.get("storage", {}).get("retain_all_checkpoints", False)):
        return verification_gate
    return _add_job(
        jobs,
        job_id="retention_prune_prescribed_lr_schedule_probes",
        stage="lr_verification",
        action="checkpoint_prune",
        resource="cpu",
        dependencies=(verification_gate,),
        arguments={
            "source_jobs": probe_jobs,
            "evidence_jobs": [verification_gate],
            "retention_scope": "prescribed_lr_schedule_probe_checkpoints_after_gate",
        },
        outputs=(
            paths.control
            / "prune"
            / "retention_prune_prescribed_lr_schedule_probes.json",
        ),
    )


def _diagnostic_multiplier_id(multiplier: float) -> str:
    value = int(round(float(multiplier) * 100))
    if not math.isclose(float(multiplier), value / 100.0, rel_tol=0.0, abs_tol=0.0):
        raise WorkflowError(f"diagnostic LR multiplier cannot be named exactly: {multiplier}")
    return f"m{value:03d}"


def _production_lr_evidence_jobs(
    jobs: list[dict[str, Any]],
    *,
    verification_gate: str,
    paths: Paths,
    runtime: Mapping[str, Any],
) -> str:
    """Bind production to a reviewed, immutable LR-confirmation result."""

    evidence = runtime["training"].get("production_lr_evidence")
    if not isinstance(evidence, Mapping):
        raise WorkflowError("production runtime lacks training.production_lr_evidence")
    required = {
        "source_gate_path",
        "source_gate_file_sha256",
        "source_gate_payload_sha256",
        "source_recovery_path",
        "source_recovery_file_sha256",
        "source_manifest_sha256",
        "observed_decision",
        "authorization_decision",
        "selected_rule",
        "deferred_boundary_cell",
    }
    if set(evidence) != required:
        raise WorkflowError("production LR-evidence contract keys drifted")
    if (
        evidence["observed_decision"] != "inconclusive_boundary"
        or evidence["authorization_decision"] != "authorized_with_disclosed_boundary"
        or evidence["selected_rule"] != "jointmoe_x1_aggregate_a_priori"
        or evidence["deferred_boundary_cell"] != "B/E1/K1:x4"
    ):
        raise WorkflowError("production LR-evidence scientific decision drifted")
    return _add_job(
        jobs,
        job_id="production_lr_evidence_gate",
        stage="lr_diagnostics",
        action="production_lr_evidence_gate",
        resource="cpu",
        dependencies=(verification_gate,),
        arguments=dict(evidence),
        outputs=(paths.control / "production_lr_evidence_gate.json",),
    )


def _lr_diagnostic_jobs(
    jobs: list[dict[str, Any]],
    *,
    verification_gate: str,
    paths: Paths,
    lr_prescription: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> str:
    policy = active_policy()
    config = runtime["training"]["learning_rate_diagnostics"]
    horizon = int(config["horizon_tokens"])
    corner_experts = tuple(int(value) for value in config["corner_experts"])
    corner_ks = tuple(int(value) for value in config["corner_language_counts"])
    trajectories: list[tuple[str, int, int, int, float, tuple[str, ...]]] = []
    for ladder in policy.ladders:
        for experts in corner_experts:
            for k in corner_ks:
                for seed in REPLICATION_TRAINING_SEEDS:
                    roles = ["seed_replication"]
                    if ladder == "B" and seed == PRIMARY_TRAINING_SEED:
                        roles.append("lr_transfer_baseline")
                    trajectories.append((ladder, experts, k, seed, 1.0, tuple(roles)))
    for ladder in config["neighbor_ladders"]:
        for experts in corner_experts:
            for k in corner_ks:
                for multiplier in config["neighbor_multipliers"]:
                    trajectories.append(
                        (
                            str(ladder),
                            experts,
                            k,
                            PRIMARY_TRAINING_SEED,
                            float(multiplier),
                            ("lr_transfer_neighbor",),
                        )
                    )
    identity = [row[:5] for row in trajectories]
    if len(identity) != len(set(identity)):
        raise WorkflowError("LR diagnostic trajectory identities are not unique")

    evaluation_jobs: list[str] = []
    prune_sources: list[str] = []
    for ladder, experts, k, seed, multiplier, diagnostic_roles in trajectories:
        suffix = (
            f"{ladder.lower()}_e{experts}_k{k:03d}_s{seed}_"
            f"{_diagnostic_multiplier_id(multiplier)}"
        )
        trajectory_id = f"lrdiag_{suffix}"
        contract = _training_contract_arguments(
            lr_prescription,
            ladder=ladder,
            experts=experts,
            training_seed=seed,
            lr_multiplier=multiplier,
        )
        endpoint = _expected_wsd_branch_endpoint(
            runtime,
            ladder=ladder,
            experts=experts,
            nominal_horizon_tokens=horizon,
        )
        common = {
            "trajectory_id": trajectory_id,
            "diagnostic_roles": list(diagnostic_roles),
            "checkpoint_horizons_tokens": [horizon],
            "ladder": ladder,
            "experts": experts,
            "K": k,
            "D": horizon,
            "expected_endpoint": endpoint,
            **contract,
        }
        trunk_id = f"lrdiag_trunk_{suffix}"
        branch_id = f"lrdiag_branch_{suffix}"
        eval_id = f"lrdiag_eval_{suffix}"
        _add_job(
            jobs,
            job_id=trunk_id,
            stage="lr_diagnostics",
            action="production_train",
            resource="h100",
            dependencies=(verification_gate,),
            arguments={"role": "trunk", "purpose": "lr_diagnostic", **common},
            outputs=(paths.jobs / trunk_id / "result.json",),
            gpu_tokens=int(0.8 * horizon),
        )
        _add_job(
            jobs,
            job_id=branch_id,
            stage="lr_diagnostics",
            action="production_train",
            resource="h100",
            dependencies=(trunk_id,),
            arguments={
                "role": "branch",
                "parent_job_id": trunk_id,
                "purpose": "lr_diagnostic",
                **common,
            },
            outputs=(paths.jobs / branch_id / "result.json",),
            gpu_tokens=horizon - int(0.8 * horizon),
        )
        _add_job(
            jobs,
            job_id=eval_id,
            stage="lr_diagnostics",
            action="per_language_eval",
            resource="h100",
            dependencies=(branch_id,),
            arguments={
                "training_job_id": branch_id,
                "purpose": "lr_diagnostic",
                **common,
            },
            outputs=(paths.jobs / eval_id / "evaluation_manifest.json",),
        )
        evaluation_jobs.append(eval_id)
        prune_sources.extend((trunk_id, branch_id))

    gate_id = _add_job(
        jobs,
        job_id="lr_diagnostic_gate",
        stage="lr_diagnostics",
        action="lr_diagnostic_gate",
        resource="cpu",
        dependencies=tuple(evaluation_jobs),
        arguments={
            "evaluation_jobs": evaluation_jobs,
            "horizon_tokens": horizon,
            "endpoint_contract": "nominal horizon plus exact batch-aligned endpoint",
            "allowed_nominal_token_undershoot": 0,
            "systematic_block_count": int(config["systematic_block_count"]),
            "automatic_learning_rate_mutation": False,
        },
        outputs=(paths.control / "lr_diagnostic_gate.json",),
    )
    return _add_job(
        jobs,
        job_id="retention_prune_lr_diagnostics",
        stage="lr_diagnostics",
        action="checkpoint_prune",
        resource="cpu",
        dependencies=(gate_id,),
        arguments={
            "source_jobs": prune_sources,
            "evidence_jobs": [gate_id, *evaluation_jobs],
            "retention_scope": "all_lr_diagnostic_checkpoints_after_gate",
        },
        outputs=(paths.control / "prune" / "retention_prune_lr_diagnostics.json",),
    )


def _production_wave_id(ladder: str, experts: int) -> str:
    return f"retention_prune_production_wave_{ladder.lower()}_e{experts}"


def _production_jobs(
    jobs: list[dict[str, Any]],
    *,
    diagnostic_gate: str,
    lr_prescription: Mapping[str, Any] | None,
    runtime: Mapping[str, Any],
    paths: Paths,
) -> str:
    policy = active_policy()
    all_evaluations: list[str] = []
    prune_sources: list[str] = []
    retained_jobs: list[str] = []
    horizons = runtime["training"]["checkpoint_horizons_tokens"]
    retain_all = bool(runtime.get("storage", {}).get("retain_all_checkpoints", False))
    storage_waves = lr_prescription is not None and not retain_all
    previous_wave_gate = diagnostic_gate
    for wave_index, (ladder, experts) in enumerate(
        ((ladder, experts) for ladder in policy.ladders for experts in EXPERTS),
        start=1,
    ):
        wave_evaluations: list[str] = []
        wave_prune_sources: list[str] = []
        wave_retained_jobs: list[str] = []
        trunk_gate = previous_wave_gate if storage_waves else diagnostic_gate
        for k in LANGUAGE_COUNTS:
            smaller_branches: list[str] = []
            trunk_id = f"prod_trunk_{ladder.lower()}_e{experts}_k{k:03d}"
            d_max = int(horizons[ladder][-1])
            contract = _training_contract_arguments(
                lr_prescription,
                ladder=ladder,
                experts=experts,
                training_seed=PRIMARY_TRAINING_SEED,
            )
            _add_job(
                jobs,
                job_id=trunk_id,
                stage="production",
                action="production_train",
                resource="h100",
                dependencies=(trunk_gate,),
                arguments={
                    "role": "trunk",
                    "ladder": ladder,
                    "experts": experts,
                    "K": k,
                    "D": d_max,
                    **contract,
                },
                outputs=(paths.jobs / trunk_id / "result.json",),
                gpu_tokens=int(0.8 * d_max),
            )
            for d_tokens in horizons[ladder]:
                branch_id = (
                    f"prod_branch_{ladder.lower()}_e{experts}_"
                    f"k{k:03d}_d{int(d_tokens)}"
                )
                _add_job(
                    jobs,
                    job_id=branch_id,
                    stage="production",
                    action="production_train",
                    resource="h100",
                    dependencies=(trunk_id,),
                    arguments={
                        "role": "branch",
                        "parent_job_id": trunk_id,
                        "ladder": ladder,
                        "experts": experts,
                        "K": k,
                        "D": int(d_tokens),
                        **contract,
                    },
                    outputs=(paths.jobs / branch_id / "result.json",),
                    gpu_tokens=int(0.2 * int(d_tokens)),
                )
                eval_id = branch_id.replace("prod_branch_", "prod_eval_", 1)
                _add_job(
                    jobs,
                    job_id=eval_id,
                    stage="validation",
                    action="per_language_eval",
                    resource="h100",
                    dependencies=(branch_id,),
                    arguments={
                        "training_job_id": branch_id,
                        "ladder": ladder,
                        "experts": experts,
                        "K": k,
                        "D": int(d_tokens),
                        "purpose": "production",
                        "model_seed": PRIMARY_TRAINING_SEED,
                        "dataset_random_seed": PRIMARY_TRAINING_SEED,
                    },
                    outputs=(paths.jobs / eval_id / "evaluation_manifest.json",),
                )
                all_evaluations.append(eval_id)
                wave_evaluations.append(eval_id)
                if int(d_tokens) != d_max:
                    smaller_branches.append(branch_id)
            wave_prune_sources.extend([trunk_id, *smaller_branches])
            wave_retained_jobs.append(
                f"prod_branch_{ladder.lower()}_e{experts}_k{k:03d}_d{d_max}"
            )

        if storage_waves:
            wave_id = _production_wave_id(ladder, experts)
            previous_wave_gate = _add_job(
                jobs,
                job_id=wave_id,
                stage="production",
                action="checkpoint_prune",
                resource="cpu",
                dependencies=tuple(wave_evaluations),
                arguments={
                    "source_jobs": wave_prune_sources,
                    "retained_jobs": wave_retained_jobs,
                    "evidence_jobs": wave_evaluations,
                    "production_wave": {
                        "index": wave_index,
                        "ladder": ladder,
                        "experts": experts,
                    },
                    "retention_scope": (
                        f"after all {ladder}/E{experts} evaluations, prune its trunks "
                        "and smaller-D branches; retain four largest-D final branches"
                    ),
                },
                outputs=(paths.control / "prune" / f"{wave_id}.json",),
            )
        else:
            prune_sources.extend(wave_prune_sources)
            retained_jobs.extend(wave_retained_jobs)

    summary_id = _add_job(
        jobs,
        job_id="final_summary",
        stage="summary",
        action="final_summary",
        resource="cpu",
        dependencies=(
            (previous_wave_gate,) if storage_waves else tuple(all_evaluations)
        ),
        arguments={"evaluation_jobs": all_evaluations},
        outputs=(paths.control / "final_summary.json",),
    )
    if storage_waves:
        return summary_id
    if retain_all:
        return summary_id
    return _add_job(
        jobs,
        job_id="retention_prune_production_after_summary",
        stage="retention",
        action="checkpoint_prune",
        resource="cpu",
        dependencies=(summary_id,),
        arguments={
            "source_jobs": prune_sources,
            "retained_jobs": retained_jobs,
            "evidence_jobs": [summary_id, *all_evaluations],
            "retention_scope": (
                "after the complete hash-validated "
                f"{len(all_evaluations)}-result summary, prune every trunk and smaller-D "
                f"branch; retain {len(retained_jobs)} largest-D final branches"
            ),
        },
        outputs=(
            paths.control / "prune" / "retention_prune_production_after_summary.json",
        ),
    )


from .job_validation import (
    _topological,
    _validate_dag,
    _validate_fixed_lr_contract,
    _validate_lr_contract,
    _validate_production_checkpoint_waves,
)
