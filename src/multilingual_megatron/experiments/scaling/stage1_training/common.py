"""Shared contract for clean C/D JointMoE-x1 production reruns."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

from multilingual_megatron.experiments.scaling.ab_v2.common import (
    REPO_ROOT,
    SUBMISSION_PHASES,
    WorkflowError,
    _file_sha256,
    _load_object,
    _paths_for_manifest,
    _payload_sha256,
    _rel_or_abs,
    _submission_phase,
    _submit_slot_count,
)
from multilingual_megatron.experiments.scaling.ab_v2.contract import (
    _apply_conservative_job_time_limits,
    _manifest_without_hash,
)
from multilingual_megatron.experiments.scaling.ab_v2.estimates import (
    _gpu_estimate,
    _storage_estimate,
)
from multilingual_megatron.experiments.scaling.ab_v2.gate_production_lr_evidence import (
    validate_production_lr_evidence,
)
from multilingual_megatron.experiments.scaling.ab_v2.job_validation import (
    _validate_dag,
)


JOINTMOE_X1_PRESCRIPTION = {
    "source": "JointMoE paper arXiv:2502.05172, printed rounded-coefficient law",
    "formula": "exp(8.39 - 0.81 * ln(N_act_nonembed) - 0.25 * ln(E))",
    "intercept": 8.39,
    "active_parameter_coefficient": -0.81,
    "expert_coefficient": -0.25,
    "parameter_count_basis": "paper-style active non-embedding parameter count N_act_nonembed",
    "constant_across_K": True,
    "released_code_full_precision_coefficients_used": False,
    "per_cell_selection_allowed": False,
}

LR_ANALYSIS_EVIDENCE = {
    "analysis_script_path": "result_analysis/stage1_lr_confirmation_analysis.py",
    "analysis_script_sha256": "20eaa5648b2647524145e356d5e238f4e9e5149b3e040880dbe022a6c76d5b58",
    "analysis_manifest_path": "result_analysis/stage1_lr_confirmation_h100_v1/analysis_manifest.json",
    "analysis_manifest_sha256": "b8bd437a2b5f81d4cd34868efe26163dbe1ee40f7ace63ed3b19fda8b6765d17",
    "statistical_script_path": "result_analysis/stage1_lr_confirmation_panel_statistics.py",
    "statistical_script_sha256": "fe2b5edc8936fb029c013cbdf6513a236d089d96dcc86ec356264e4af9807243",
    "comparison_summary_path": "result_analysis/stage1_lr_confirmation_h100_v1/comparison_summary.json",
    "comparison_summary_sha256": "141e479adca5900ab0d50c47a07780c18c08a59e0852a092ef47fd94f8198c9b",
}


def production_lr_decision(ladder: str) -> dict[str, Any]:
    return {
        "status": "authorized_b_ladder_confirmation_only",
        "selected_rule": "jointmoe_x1_aggregate_a_priori",
        "decision_panel": (
            "B ladder at 2B tokens on A100 and H100; "
            "six prescriptions over nine matched E/K cells"
        ),
        "a_ladder_role": "post_decision_diagnostic_not_selection_evidence",
        "target_ladder": ladder,
        "automatic_per_cell_lr_selection": False,
        "c_or_d_lr_probe_required": False,
        "inference_scope": (
            "Prospective uniform JointMoE-x1 prescription for the clean C/D rerun; "
            "not a claim of per-cell, seed-general, long-horizon, or compute-optimal superiority"
        ),
        "disclosed_boundary": (
            "B/E1/K1 selected x4 on H100, and x1 versus x0.5 is not significant "
            "in the conservative two-sided Holm-corrected comparison"
        ),
    }


def validate_production_spec(
    *,
    repo_root: Path,
    spec_path: Path,
    ladder: str,
    parent_relative_path: str,
    parent_sha256: str,
    parent_validator: Callable[..., Any],
) -> dict[str, Any]:
    """Validate a minimal LR-only overlay over one historical ladder spec."""

    spec_path = spec_path.resolve()
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot read JointMoE production spec {spec_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise WorkflowError("JointMoE production spec must be a JSON object")
    inherited = raw.get("inherits")
    expected_parent = {
        "spec_path": parent_relative_path,
        "spec_sha256": parent_sha256,
    }
    if isinstance(inherited, dict) and "historical_spec_sha256" in inherited:
        expected_parent["historical_spec_sha256"] = inherited["historical_spec_sha256"]
    if inherited != expected_parent:
        raise WorkflowError(f"{ladder} JointMoE parent-spec binding drifted")
    parent = (spec_path.parent / parent_relative_path).resolve()
    if _file_sha256(parent) != parent_sha256:
        raise WorkflowError(f"{ladder} historical parent spec changed")
    parent_validator(repo_root=repo_root, spec_path=parent)

    expected_overlay = {
        "inherits": inherited,
        "schema_version": 3,
        "status_date": "2026-08-22",
        "scope": f"Clean Stage-1 {ladder} production rerun with the B-selected JointMoE x1 LR rule",
        "training_policy": {
            "peak_learning_rate": {"__delete__": True},
            "replication_training_seeds": [1234, 5678, 91011],
            "replication_seed_scope": (
                "workflow compatibility only; production uses seed 1234 and no seed-general claim"
            ),
            "learning_rate_prescription": JOINTMOE_X1_PRESCRIPTION,
        },
        "production_lr_decision": production_lr_decision(ladder),
        "lr_analysis_evidence": LR_ANALYSIS_EVIDENCE,
    }
    if raw != expected_overlay:
        raise WorkflowError(f"{ladder} JointMoE production overlay drifted")

    resolved = _load_object(spec_path)
    if resolved.get("schema_version") != 3:
        raise WorkflowError(f"{ladder} JointMoE production schema drifted")
    training = resolved.get("training_policy")
    if (
        not isinstance(training, Mapping)
        or "peak_learning_rate" in training
        or training.get("learning_rate_prescription") != JOINTMOE_X1_PRESCRIPTION
    ):
        raise WorkflowError(f"{ladder} JointMoE LR-only overlay did not resolve exactly")
    ladders = resolved.get("axes", {}).get("ladders", [])
    if len(ladders) != 1 or ladders[0].get("id") != ladder:
        raise WorkflowError(f"{ladder} production overlay changed the ladder axis")
    return resolved


def _refresh_manifest(manifest: dict[str, Any], runtime: Mapping[str, Any]) -> None:
    jobs = manifest["jobs"]
    manifest["stage_counts"] = dict(
        sorted(Counter(str(row["stage"]) for row in jobs).items())
    )
    phase_jobs = {phase: [] for phase in SUBMISSION_PHASES}
    phase_slots = {phase: 0 for phase in SUBMISSION_PHASES}
    for row in jobs:
        phase = _submission_phase(row)
        phase_jobs[phase].append(str(row["job_id"]))
        phase_slots[phase] += _submit_slot_count(row)
    manifest["staged_submission"]["phase_jobs"] = phase_jobs
    manifest["staged_submission"]["phase_submit_slots"] = phase_slots
    manifest["staged_submission"]["prior_phase_completion_required"] = False
    paths = _paths_for_manifest(manifest)
    estimate = _gpu_estimate(jobs, runtime, paths)
    manifest["gpu_hour_estimate"] = estimate
    walltime = _apply_conservative_job_time_limits(jobs, runtime, estimate)
    if walltime is None:
        manifest.pop("walltime_contract", None)
    else:
        manifest["walltime_contract"] = walltime
    manifest["storage_estimate"] = _storage_estimate(jobs, runtime)
    _validate_dag(jobs)


def adopt_synchronous_preflight(
    manifest: dict[str, Any],
    runtime: Mapping[str, Any],
    *,
    systems_evidence_paths: tuple[Path, ...],
) -> None:
    """Remove preproduction Slurm gates after validating their evidence locally."""

    evidence = runtime.get("training", {}).get("production_lr_evidence")
    if not isinstance(evidence, Mapping):
        raise WorkflowError("production runtime lacks B-ladder LR evidence")
    evidence_report = validate_production_lr_evidence(evidence)
    evidence_file_keys = (
        ("analysis_script_path", "analysis_script_sha256"),
        ("analysis_manifest_path", "analysis_manifest_sha256"),
        ("statistical_script_path", "statistical_script_sha256"),
        ("comparison_summary_path", "comparison_summary_sha256"),
    )
    for path_key, sha_key in evidence_file_keys:
        analysis_path = (REPO_ROOT / LR_ANALYSIS_EVIDENCE[path_key]).resolve()
        if _file_sha256(analysis_path) != LR_ANALYSIS_EVIDENCE[sha_key]:
            raise WorkflowError(f"LR analysis evidence hash drifted: {analysis_path}")

    original_jobs = list(manifest["jobs"])
    retained: list[dict[str, Any]] = []
    removed: list[str] = []
    for original in original_jobs:
        job = dict(original)
        if str(job.get("stage")) not in {"production", "validation", "summary"}:
            removed.append(str(job["job_id"]))
            continue
        if str(job["job_id"]).startswith("prod_trunk_"):
            if job.get("dependencies") != ["production_lr_evidence_gate"]:
                raise WorkflowError(
                    f"unexpected production entry dependency for {job['job_id']}"
                )
            job["dependencies"] = []
        retained.append(job)
    manifest["jobs"] = retained
    manifest["role_counts"] = {
        "prescribed_lr_schedule_probes": 0,
        "production_trunks": sum(
            str(row["job_id"]).startswith("prod_trunk_") for row in retained
        ),
        "production_branches": sum(
            str(row["job_id"]).startswith("prod_branch_") for row in retained
        ),
        "production_evaluations": sum(
            str(row["job_id"]).startswith("prod_eval_") for row in retained
        ),
    }
    manifest["synchronous_preflight_contract"] = {
        "status": "required_before_first_sbatch",
        "slurm_gate_jobs_removed": sorted(removed),
        "checks": [
            "manifest and fixed-input hashes",
            "current HEAD equals manifest launch commit; dirty status is informational",
            "sealed corpus and tokenizer evidence",
            "B-ladder LR evidence and disclosed boundary",
            "C/D systems-evidence files",
            "filesystem bytes and inode headroom as advisory-only observations",
            "Slurm partition, account, and command availability",
        ],
        "production_trunk_dependencies": [],
        "git_policy": (
            "Git cleanliness and source-code hash drift are recorded advisories and never "
            "gate execution; non-code scientific fixed-input hashes remain fail-closed"
        ),
        "lr_evidence": evidence_report,
    }
    manifest["checkpoint_resume_policy"] = {
        "granularity": "planned full-state segment endpoints",
        "periodic_mid_segment_checkpoints": False,
        "maximum_planned_segment_walltime_hours": 168,
        "infrastructure_recovery": "resume exact x1 trajectory from latest hash-valid full state",
        "scientific_failure_recovery": "blocked pending documented replan; no automatic LR mutation",
    }

    fixed = dict(manifest["fixed_input_sha256"])
    evidence_paths = {
        Path(str(evidence["source_gate_path"])),
        Path(str(evidence["source_recovery_path"])),
        *systems_evidence_paths,
        *(Path(str(LR_ANALYSIS_EVIDENCE[path_key])) for path_key, _ in evidence_file_keys),
    }
    for path in evidence_paths:
        resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
        fixed[_rel_or_abs(resolved)] = _file_sha256(resolved)
    manifest["fixed_input_sha256"] = dict(sorted(fixed.items()))
    _refresh_manifest(manifest, runtime)
    manifest["storage_estimate"]["go_no_go"] = (
        "informational_advisory_never_blocks_submission"
    )
    manifest["storage_estimate"]["capacity_enforcement"] = "advisory_only"
    manifest["manifest_sha256"] = _payload_sha256(_manifest_without_hash(manifest))


def validate_clean_production_manifest(
    payload: Mapping[str, Any], *, ladder: str
) -> None:
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 241:
        raise WorkflowError(f"clean {ladder} manifest must contain exactly 241 jobs")
    role_counts = {
        "prescribed_lr_schedule_probes": 0,
        "production_trunks": 16,
        "production_branches": 112,
        "production_evaluations": 112,
    }
    if payload.get("role_counts") != role_counts:
        raise WorkflowError(f"clean {ladder} production role cardinality drifted")
    forbidden_actions = {
        "storage_preflight",
        "reuse_sealed_data_verify",
        "megatron_data_smoke",
        "training_sentinel",
        "sentinel_gate",
        "prescribed_lr_schedule_probe",
        "prescribed_lr_schedule_gate",
        "production_lr_evidence_gate",
        "checkpoint_prune",
    }
    if any(str(row.get("action")) in forbidden_actions for row in jobs):
        raise WorkflowError(f"clean {ladder} manifest contains a forbidden gate or probe")
    trunks = [row for row in jobs if str(row["job_id"]).startswith("prod_trunk_")]
    if any(row.get("dependencies") != [] for row in trunks):
        raise WorkflowError(f"clean {ladder} trunks must be released after synchronous preflight")
    if any(
        row.get("arguments", {}).get("lr_multiplier") != 1.0
        for row in jobs
        if row.get("action") == "production_train"
    ):
        raise WorkflowError(f"clean {ladder} manifest contains a non-x1 training job")
    if any(
        key in row.get("arguments", {})
        for row in jobs
        for key in ("resume_from", "load_checkpoint", "historical_checkpoint")
    ):
        raise WorkflowError(f"clean {ladder} manifest references a historical checkpoint")
    preflight = payload.get("synchronous_preflight_contract")
    if not isinstance(preflight, Mapping) or preflight.get("status") != "required_before_first_sbatch":
        raise WorkflowError(f"clean {ladder} synchronous preflight contract is missing")
    storage_estimate = payload.get("storage_estimate", {})
    if (
        storage_estimate.get("go_no_go")
        != "informational_advisory_never_blocks_submission"
        or storage_estimate.get("capacity_enforcement") != "advisory_only"
    ):
        raise WorkflowError(f"clean {ladder} storage capacity must be advisory-only")
    if payload.get("checkpoint_resume_policy", {}).get(
        "maximum_planned_segment_walltime_hours"
    ) != 168:
        raise WorkflowError(f"clean {ladder} recovery walltime contract drifted")
    for row in jobs:
        if not str(row.get("resource", "")).startswith("h100"):
            continue
        if row.get("slurm_time_limit") != row.get("slurm_time_min"):
            raise WorkflowError(f"{row['job_id']} allows a shortened TimeMin")
        if float(row.get("predicted_wall_hours", 0.0)) <= 0:
            raise WorkflowError(f"{row['job_id']} lacks a positive walltime prediction")
    prescription = payload.get("learning_rate_prescription", {})
    rates = prescription.get("base_peak_learning_rate_by_ladder_and_experts", {}).get(
        ladder, {}
    )
    count = int(prescription.get("N_act_nonembed_by_ladder", {}).get(ladder, 0))
    for experts in (1, 4, 16, 32):
        expected = math.exp(8.39 - 0.81 * math.log(count) - 0.25 * math.log(experts))
        if not math.isclose(float(rates.get(str(experts), 0.0)), expected, rel_tol=0.0, abs_tol=0.0):
            raise WorkflowError(f"clean {ladder}/E{experts} LR literal drifted")


def validate_lr_only_runtime(
    runtime: Mapping[str, Any], *, historical_runtime_path: Path
) -> None:
    """Prove that the executable training contract differs only in peak LR."""

    historical_runtime_path = (
        historical_runtime_path.resolve()
        if historical_runtime_path.is_absolute()
        else (REPO_ROOT / historical_runtime_path).resolve()
    )
    historical = _load_object(historical_runtime_path)
    if runtime.get("data") != historical.get("data"):
        raise WorkflowError("JointMoE production changed the frozen data runtime")
    new_training = runtime.get("training")
    old_training = historical.get("training")
    if not isinstance(new_training, Mapping) or not isinstance(old_training, Mapping):
        raise WorkflowError("training runtime contract is missing")
    exact_fields = (
        "context_length",
        "experts",
        "language_counts",
        "micro_batch_size",
        "evaluation_micro_batch_size",
        "checkpoint_horizons_tokens",
        "data_parallel_size_by_ladder_and_experts",
        "cuda_graph_impl_by_experts",
        "runtime_env",
    )
    for field in exact_fields:
        if new_training.get(field) != old_training.get(field):
            raise WorkflowError(
                f"JointMoE production changed scientific runtime field training.{field}"
            )
    old_optimizer = dict(old_training.get("optimizer", {}))
    old_optimizer.pop("peak_learning_rate", None)
    if new_training.get("optimizer") != old_optimizer:
        raise WorkflowError("JointMoE production changed optimizer settings beyond peak LR")
    if "peak_learning_rate" in new_training.get("optimizer", {}):
        raise WorkflowError("JointMoE runtime must derive peak LR from the prescription")
    if new_training.get("sentinels") != []:
        raise WorkflowError("JointMoE C/D runtime must not schedule new sentinel probes")
    if "learning_rate_diagnostics" in new_training:
        raise WorkflowError("JointMoE C/D runtime must not schedule LR diagnostics")
    storage = runtime.get("storage", {})
    if storage.get("enforcement") != "informational_soft_limit":
        raise WorkflowError("JointMoE C/D storage capacity must never gate submission")
    new_seeds = runtime.get("seeds", {})
    old_seeds = historical.get("seeds", {})
    if (
        new_seeds.get("data") != old_seeds.get("data")
        or new_seeds.get("primary_training") != old_seeds.get("model")
    ):
        raise WorkflowError("JointMoE production changed the primary seed contract")
    for resource in runtime.get("resources", {}).values():
        flags = [str(value) for value in resource.get("extra_sbatch_flags", [])]
        if any(
            token in flag
            for flag in flags
            for token in ("--reservation", "--nodelist", "--hold", "--exclude")
        ):
            raise WorkflowError("JointMoE ordinary-queue runtime contains a stale placement flag")


__all__ = [
    "JOINTMOE_X1_PRESCRIPTION",
    "LR_ANALYSIS_EVIDENCE",
    "adopt_synchronous_preflight",
    "production_lr_decision",
    "validate_clean_production_manifest",
    "validate_lr_only_runtime",
    "validate_production_spec",
]
