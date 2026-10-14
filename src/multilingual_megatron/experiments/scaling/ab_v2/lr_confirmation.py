"""Own the Stage-1 B-ladder learning-rate confirmation campaign contract."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .common import Paths, WorkflowError, _prescribed_peak_learning_rate
from .estimates import _expected_wsd_branch_endpoint
from .jobs import _add_job, _training_contract_arguments


PROTOCOL = "stage1_jointmoe_lr_confirmation_v1"
EXPECTED_DIAGNOSTICS = {
    "protocol": PROTOCOL,
    "horizon_tokens": 2_000_000_000,
    "ladder": "B",
    "experts": [1, 16, 32],
    "language_counts": [1, 12, 200],
    "multipliers": [0.25, 0.5, 1.0, 2.0, 4.0],
    "fixed_control_peak_lr": 0.0002,
    "training_seed": 1234,
    "tie_absolute_tolerance": 1e-12,
    "minimum_x1_cell_wins_or_ties": 5,
}
BASE_DIAGNOSTICS = {
    "horizon_tokens": 2_000_000_000,
    "corner_experts": [1, 32],
    "corner_language_counts": [1, 200],
    "neighbor_ladders": ["B"],
    "neighbor_multipliers": [0.5, 2.0],
    "systematic_block_count": 3,
}


def _validate_lr_diagnostic_mode(
    diagnostics: object,
    *,
    legacy_fixed_lr: bool,
    variant_identifier: str,
    authorized_production: bool = False,
) -> bool:
    confirmation = (
        isinstance(diagnostics, Mapping) and diagnostics.get("protocol") == PROTOCOL
    )
    if legacy_fixed_lr and diagnostics is not None:
        raise WorkflowError("historical fixed-LR runtime must not add LR diagnostics")
    if not legacy_fixed_lr:
        ladder = str(diagnostics.get("ladder", "")) if confirmation else ""
        expected_confirmation = dict(EXPECTED_DIAGNOSTICS)
        if ladder == "A":
            expected_confirmation.update({"ladder": "A"})
        expected = expected_confirmation if confirmation else BASE_DIAGNOSTICS
        authorized_confirmation_variants = {
            "stage1_lr_confirmation_v1",
            "stage1_lr_confirmation_a_h100_v1",
        }
        if authorized_production and diagnostics is None:
            return False
        if diagnostics != expected:
            raise WorkflowError("runtime LR-diagnostic contract drifted")
        if confirmation != (variant_identifier in authorized_confirmation_variants):
            raise WorkflowError(
                "LR-confirmation runtime must use its dedicated workflow variant"
            )
    return confirmation


def _lr_confirmation_jobs(
    jobs: list[dict[str, Any]],
    *,
    verification_gate: str,
    paths: Paths,
    lr_prescription: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> str:
    """Build a preregistered ladder-specific 54-trajectory H100 confirmation DAG."""

    config = runtime["training"]["learning_rate_diagnostics"]
    horizon = int(config["horizon_tokens"])
    ladder = str(config["ladder"])
    experts_axis = tuple(int(value) for value in config["experts"])
    k_axis = tuple(int(value) for value in config["language_counts"])
    multipliers = tuple(float(value) for value in config["multipliers"])
    fixed_control = float(config["fixed_control_peak_lr"])
    seed = int(config["training_seed"])
    protocol = str(config["protocol"])
    if protocol != PROTOCOL:
        raise WorkflowError("unexpected LR-confirmation protocol")

    candidates: list[tuple[str, str, float | None]] = [
        (f"x{int(round(multiplier * 100)):03d}", "jointmoe_factor", multiplier)
        for multiplier in multipliers
    ]
    candidates.append(("fixed2e4", "historical_fixed_control", None))
    if len(candidates) != len({candidate[0] for candidate in candidates}):
        raise WorkflowError("LR-confirmation candidate IDs are not unique")

    evaluation_jobs: list[str] = []
    for experts in experts_axis:
        base_lr = _prescribed_peak_learning_rate(
            lr_prescription, ladder=ladder, experts=experts
        )
        for k in k_axis:
            for candidate_id, candidate_kind, factor in candidates:
                effective_multiplier = (
                    float(factor) if factor is not None else fixed_control / base_lr
                )
                contract = _training_contract_arguments(
                    lr_prescription,
                    ladder=ladder,
                    experts=experts,
                    training_seed=seed,
                    lr_multiplier=effective_multiplier,
                )
                suffix = f"{ladder.lower()}_e{experts}_k{k:03d}_s{seed}_{candidate_id}"
                trajectory_id = f"lrconfirm_{suffix}"
                endpoint = _expected_wsd_branch_endpoint(
                    runtime,
                    ladder=ladder,
                    experts=experts,
                    nominal_horizon_tokens=horizon,
                )
                common = {
                    "trajectory_id": trajectory_id,
                    "diagnostic_roles": ["stage1_lr_confirmation"],
                    "diagnostic_protocol": protocol,
                    "lr_candidate_id": candidate_id,
                    "lr_candidate_kind": candidate_kind,
                    "jointmoe_factor": factor,
                    "fixed_control_peak_lr": fixed_control if factor is None else None,
                    "checkpoint_horizons_tokens": [horizon],
                    "ladder": ladder,
                    "experts": experts,
                    "K": k,
                    "D": horizon,
                    "expected_endpoint": endpoint,
                    **contract,
                }
                trunk_id = f"lrconfirm_trunk_{suffix}"
                branch_id = f"lrconfirm_branch_{suffix}"
                eval_id = f"lrconfirm_eval_{suffix}"
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
                if not bool(runtime.get("storage", {}).get("retain_all_checkpoints", False)):
                    _add_job(
                        jobs,
                        job_id=f"retention_prune_{trajectory_id}",
                        stage="lr_diagnostics",
                        action="checkpoint_prune",
                        resource="cpu",
                        dependencies=(eval_id,),
                        arguments={
                            "source_jobs": [trunk_id],
                            "retained_jobs": [branch_id],
                            "evidence_jobs": [eval_id],
                            "retention_scope": (
                                "prune_1p6b_trunk_after_hash_valid_eval_retain_2b_final"
                            ),
                        },
                        outputs=(
                            paths.control
                            / "prune"
                            / f"retention_prune_{trajectory_id}.json",
                        ),
                    )

    expected = len(experts_axis) * len(k_axis) * len(candidates)
    if len(evaluation_jobs) != expected:
        raise WorkflowError("LR-confirmation trajectory cardinality drifted")
    return _add_job(
        jobs,
        job_id="lr_confirmation_gate",
        stage="lr_diagnostics",
        action="lr_diagnostic_gate",
        resource="cpu",
        dependencies=tuple(evaluation_jobs),
        arguments={
            "evaluation_jobs": evaluation_jobs,
            "horizon_tokens": horizon,
            "expected_trajectories": expected,
            "analysis_protocol": protocol,
            "ladder": ladder,
            "endpoint_contract": "nominal horizon plus exact batch-aligned endpoint",
            "allowed_nominal_token_undershoot": 0,
            "automatic_learning_rate_mutation": False,
            "manual_production_authorization_required": True,
            "tie_absolute_tolerance": float(config["tie_absolute_tolerance"]),
            "minimum_x1_cell_wins_or_ties": int(
                config["minimum_x1_cell_wins_or_ties"]
            ),
        },
        outputs=(paths.control / "lr_confirmation_gate.json",),
    )


def _lr_confirmation_role_counts(
    jobs: Sequence[Mapping[str, Any]], *, prescribed_probe_count: int
) -> tuple[dict[str, int], dict[str, int]]:
    counts = {
        "prescribed_lr_schedule_probes": sum(
            row["action"] == "prescribed_lr_schedule_probe" for row in jobs
        ),
        "lr_confirmation_trajectories": sum(
            str(row["job_id"]).startswith("lrconfirm_eval_") for row in jobs
        ),
        "lr_confirmation_training_jobs": sum(
            str(row["job_id"]).startswith(("lrconfirm_trunk_", "lrconfirm_branch_"))
            for row in jobs
        ),
        "lr_confirmation_evaluations": sum(
            str(row["job_id"]).startswith("lrconfirm_eval_") for row in jobs
        ),
        "production_trunks": 0,
        "production_branches": 0,
        "production_evaluations": 0,
    }
    expected = {
        "prescribed_lr_schedule_probes": prescribed_probe_count,
        "lr_confirmation_trajectories": 54,
        "lr_confirmation_training_jobs": 108,
        "lr_confirmation_evaluations": 54,
        "production_trunks": 0,
        "production_branches": 0,
        "production_evaluations": 0,
    }
    return counts, expected


def _lr_confirmation_campaign_descriptor(runtime: Mapping[str, Any]) -> dict[str, Any]:
    config = runtime["training"]["learning_rate_diagnostics"]
    retain_all = bool(runtime.get("storage", {}).get("retain_all_checkpoints", False))
    return {
        "protocol": PROTOCOL,
        "ladder": str(config["ladder"]),
        "trajectory_count": 54,
        "production_jobs_present": False,
        "manual_production_authorization_required": True,
        "checkpoint_retention": (
            "retain every trunk and final 2B checkpoint through manual scientific review"
            if retain_all
            else "retain all final 2B checkpoints through manual scientific review"
        ),
    }
