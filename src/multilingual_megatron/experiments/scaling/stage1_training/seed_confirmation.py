#!/usr/bin/env python3
"""Stage-1 B-ladder 3x3 seed-confirmation side study."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.experiments.scaling.ab_v2 import api as _api
from multilingual_megatron.experiments.scaling.ab_v2.artifact_data import (
    _verify_workflow_data,
)
from multilingual_megatron.experiments.scaling.ab_v2.common import (
    EXPECTED_MICRO_BATCH_SIZE,
    REPO_ROOT,
    REQUIRED_TRAINING_ENV_KEYS,
    SUBMISSION_PHASES,
    Paths,
    WorkflowError,
    _file_sha256,
    _load_object,
    _paths_for_manifest,
    _payload_sha256,
    _rel_or_abs,
    _submission_phase,
    _submit_slot_count,
)
from multilingual_megatron.experiments.scaling.ab_v2.common_provenance import (
    _fixed_hashes,
)
from multilingual_megatron.experiments.scaling.ab_v2.contract import (
    _manifest_without_hash,
)
from multilingual_megatron.experiments.scaling.ab_v2.contract_timing import (
    _apply_conservative_job_time_limits,
)
from multilingual_megatron.experiments.scaling.ab_v2.estimates import (
    _gpu_estimate,
    _model_parameter_upper_bound,
    _planned_checkpoint_count_for_job,
)
from multilingual_megatron.experiments.scaling.ab_v2.gate_production_lr_evidence import (
    validate_production_lr_evidence,
)
from multilingual_megatron.experiments.scaling.ab_v2.job_validation import (
    _validate_dag,
)
from multilingual_megatron.experiments.scaling.ab_v2.variant import (
    WorkflowPolicy,
    WorkflowVariant,
    variant_scope,
)
from multilingual_megatron.experiments.scaling.stage1_training.ab_production import DEFAULT_SPEC as TEMPLATE_SPEC
from multilingual_megatron.experiments.scaling.stage1_training.ab_production import VARIANT as TEMPLATE_VARIANT
from multilingual_megatron.experiments.scaling.stage1_training.ab_production import validate_production_spec
from multilingual_megatron.experiments.scaling.stage1_training.common import LR_ANALYSIS_EVIDENCE


DEFAULT_SPEC = Path(
    "configs/scaling_law/multilingual_moe_ab_v2/study_spec_seed_confirmation_v1.json"
)
DEFAULT_RUNTIME = Path(
    "configs/scaling_law/multilingual_moe_ab_v2/workflow_runtime_seed_confirmation_v1.json"
)
TEMPLATE_RUNTIME = Path(
    "configs/scaling_law/multilingual_moe_ab_v2/workflow_runtime_jointmoe_x1_production_v1.json"
)
DEFAULT_OUTPUT = Path("outputs/pipeline_runs/stage1_b_seed_confirmation_v1")
BASELINE_MANIFEST = Path(
    "outputs/pipeline_runs/stage1_ab_jointmoe_x1_production_v1/control/launch_manifest.json"
)
WORKFLOW_MODULE = (
    "multilingual_megatron.experiments.scaling.stage1_training.seed_confirmation"
)

PARENT_SPEC_SHA256 = "788105aff6f93fa2f0e4a501c864f216c2ac783da2d5e1b0bcbb9fe7a4863585"
PARENT_RUNTIME_SHA256 = "48c0b1e5179a23db106e73cc2f90d88f693669a9cbb6c2a2f19499bc8fe9f12c"
BASELINE_MANIFEST_FILE_SHA256 = (
    "e743d1ef772eef3be167a0c996a49dcf3af0860eae0d1b3560032696198a3d87"
)
BASELINE_WORKFLOW_SHA256 = (
    "188548a71e12a36d9878cf01f13c6a5f50d8be1142ac8a6671966844935bed34"
)

PRIMARY_SEED = 1234
NEW_SEEDS = (5678, 91011)
COMPLETE_SEEDS = (PRIMARY_SEED, *NEW_SEEDS)
SELECTED_EXPERTS = (1, 16, 32)
SELECTED_LANGUAGE_COUNTS = (1, 12, 200)
SUBMISSION_EXPERT_ORDER = (32, 16, 1)
SUBMISSION_LANGUAGE_ORDER = (200, 12, 1)
HORIZONS = (
    500_000_000,
    1_000_000_000,
    2_000_000_000,
    4_000_000_000,
    8_000_000_000,
    16_000_000_000,
)
DESIGN_ID = "stage1_b_seed_confirmation_v1"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot read seed-confirmation config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WorkflowError(f"seed-confirmation config must be an object: {path}")
    return payload


def _validate_design(design: Mapping[str, Any]) -> None:
    if set(design) != {
        "status",
        "ladder",
        "model_size_label",
        "jointmoe_lr_N_act_nonembed",
        "experts",
        "language_counts",
        "checkpoint_horizons_tokens",
        "seeds",
        "trajectory_counts",
        "new_job_counts",
        "learning_rate_contract",
        "baseline_reuse",
        "frozen_factors",
        "analysis_contract",
    }:
        raise WorkflowError("seed-confirmation design keys drifted")
    if (
        design.get("status") != "predeclared_before_launch"
        or design.get("ladder") != "B"
        or int(design.get("jointmoe_lr_N_act_nonembed", 0)) != 92_012_544
        or tuple(design.get("experts", ())) != SELECTED_EXPERTS
        or tuple(design.get("language_counts", ())) != SELECTED_LANGUAGE_COUNTS
        or tuple(design.get("checkpoint_horizons_tokens", ())) != HORIZONS
    ):
        raise WorkflowError("seed-confirmation factorial panel drifted")
    if design.get("seeds") != {
        "corpus_construction": 42,
        "existing_primary_training": PRIMARY_SEED,
        "new_training": list(NEW_SEEDS),
        "complete_panel": list(COMPLETE_SEEDS),
        "training_seed_scope": (
            "model initialization and deterministic training-data ordering; corpus "
            "membership, tokenizer, language mixture, and held-out validation remain fixed"
        ),
    }:
        raise WorkflowError("seed-confirmation randomness contract drifted")
    if design.get("trajectory_counts") != {
        "reused_primary": 9,
        "new": 18,
        "complete": 27,
        "readouts_per_trajectory": 6,
        "complete_readouts": 162,
    }:
        raise WorkflowError("seed-confirmation trajectory counts drifted")
    if design.get("new_job_counts") != {
        "training_trunks": 18,
        "cooldown_branches": 108,
        "heldout_evaluations": 108,
        "gpu_jobs": 234,
        "cpu_summaries": 1,
    }:
        raise WorkflowError("seed-confirmation job counts drifted")
    if design.get("learning_rate_contract") != {
        "rule": "JointMoE printed rounded-coefficient x1 law",
        "factor": 1.0,
        "constant_across_K": True,
        "automatic_selection_or_adaptation": False,
    }:
        raise WorkflowError("seed-confirmation LR contract drifted")
    baseline = design.get("baseline_reuse")
    if baseline != {
        "output_root": "outputs/pipeline_runs/stage1_ab_jointmoe_x1_production_v1",
        "launch_manifest_path": str(BASELINE_MANIFEST),
        "launch_manifest_file_sha256": BASELINE_MANIFEST_FILE_SHA256,
        "workflow_manifest_sha256": BASELINE_WORKFLOW_SHA256,
        "reuse_policy": (
            "consume the nine canonical seed-1234 B trajectories after their hash-valid "
            "receipts complete; never recompute them for this side study"
        ),
    }:
        raise WorkflowError("seed-confirmation baseline binding drifted")
    analysis = design.get("analysis_contract")
    if not isinstance(analysis, Mapping) or (
        analysis.get("primary_metric") != "equal-language adjusted held-out loss"
        or analysis.get("replication_unit")
        != "independent training seed within a fixed E/K cell"
        or analysis.get("inference_scope") != "selected B-ladder anchors only"
        or "checkpoints, readouts, or language rows" not in str(
            analysis.get("significance_policy", "")
        )
        or set(analysis.get("forbidden_claims", ()))
        != {
            "full-grid seed invariance",
            "C/D seed robustness",
            "JointMoE x1 compute optimality",
            "statistical significance inferred from nested readouts or language rows",
        }
    ):
        raise WorkflowError("seed-confirmation analysis boundary drifted")


def validate_seed_confirmation_spec(
    *, repo_root: Path, spec_path: Path, **kwargs: Any
) -> dict[str, Any]:
    del kwargs
    spec_path = spec_path.resolve()
    raw = _read_json(spec_path)
    if set(raw) != {"inherits", "status_date", "scope", "seed_confirmation_design"}:
        raise WorkflowError("seed-confirmation spec overlay keys drifted")
    expected_parent = {
        "spec_path": "study_spec_jointmoe_x1_production_v1.json",
        "spec_sha256": PARENT_SPEC_SHA256,
    }
    inherited = raw.get("inherits")
    if isinstance(inherited, dict) and "historical_spec_sha256" in inherited:
        expected_parent["historical_spec_sha256"] = inherited["historical_spec_sha256"]
    if inherited != expected_parent:
        raise WorkflowError("seed-confirmation parent-spec binding drifted")
    parent = (spec_path.parent / expected_parent["spec_path"]).resolve()
    if _file_sha256(parent) != PARENT_SPEC_SHA256:
        raise WorkflowError("seed-confirmation parent spec changed")
    validate_production_spec(repo_root=repo_root, spec_path=parent)
    if (
        raw.get("status_date") != "2026-08-23"
        or raw.get("scope")
        != "Predeclared Stage-1 B-ladder seed-confirmation side study for the JointMoE x1 main-grid prescription"
    ):
        raise WorkflowError("seed-confirmation scope metadata drifted")
    design = raw.get("seed_confirmation_design")
    if not isinstance(design, Mapping):
        raise WorkflowError("seed-confirmation design is missing")
    _validate_design(design)
    resolved = _load_object(spec_path)
    prescription = resolved.get("training_policy", {}).get(
        "learning_rate_prescription", {}
    )
    if (
        prescription.get("formula")
        != "exp(8.39 - 0.81 * ln(N_act_nonembed) - 0.25 * ln(E))"
        or prescription.get("constant_across_K") is not True
        or prescription.get("per_cell_selection_allowed") is not False
    ):
        raise WorkflowError("seed-confirmation parent no longer resolves JointMoE x1")
    return resolved


def _validate_runtime(runtime_path: Path) -> dict[str, Any]:
    runtime_path = runtime_path.resolve()
    raw = _read_json(runtime_path)
    if set(raw) != {
        "inherits",
        "output_root",
        "workflow_attempt_id",
        "workflow_job_prefix",
        "scheduling",
        "storage",
    }:
        raise WorkflowError("seed-confirmation runtime overlay keys drifted")
    expected_parent = {
        "spec_path": "workflow_runtime_jointmoe_x1_production_v1.json",
        "spec_sha256": PARENT_RUNTIME_SHA256,
    }
    inherited = raw.get("inherits")
    if isinstance(inherited, dict) and "historical_spec_sha256" in inherited:
        expected_parent["historical_spec_sha256"] = inherited["historical_spec_sha256"]
    if inherited != expected_parent:
        raise WorkflowError("seed-confirmation parent-runtime binding drifted")
    parent_path = (runtime_path.parent / expected_parent["spec_path"]).resolve()
    if _file_sha256(parent_path) != PARENT_RUNTIME_SHA256:
        raise WorkflowError("seed-confirmation parent runtime changed")
    if (
        raw.get("output_root") != str(DEFAULT_OUTPUT)
        or raw.get("workflow_attempt_id") != DESIGN_ID
        or raw.get("workflow_job_prefix") != "bseed1"
        or raw.get("scheduling")
        != {
            "mode": "ordinary_queue_forecast",
            "predicted_time_multiplier": 1.35,
            "additive_safety_hours": 1.0,
            "minimum_job_hours": 2.0,
            "maximum_job_hours": 24.0,
            "walltime_rounding_minutes": 15,
            "timing_evidence": (
                "completed seed-1234 B trunks required 9.47-14.78 end-to-end hours; "
                "new limits retain at least 35 percent model margin plus one hour"
            ),
            "retry_policy": {
                "manual_only": True,
                "maximum_infrastructure_retries": 2,
                "scientific_or_contract_failures_retryable": False,
            },
        }
        or raw.get("storage")
        != {
            "enforcement": "informational_soft_limit",
            "retain_all_checkpoints": True,
            "production_launch_policy": (
                "all_18_new_seed_trunks_concurrently_eligible_after_synchronous_preflight"
            ),
        }
    ):
        raise WorkflowError("seed-confirmation runtime policy drifted")
    resolved = _load_object(runtime_path)
    parent = _load_object(parent_path)
    for field in (
        "study_id",
        "bridge_python_executable",
        "data",
        "seeds",
        "resources",
        "training",
        "performance_budget",
        "max_submit_slots_per_invocation",
    ):
        if resolved.get(field) != parent.get(field):
            raise WorkflowError(
                f"seed-confirmation runtime changed frozen parent field {field}"
            )
    if resolved["seeds"] != {
        "data": 42,
        "primary_training": PRIMARY_SEED,
        "replication_training": list(COMPLETE_SEEDS),
    }:
        raise WorkflowError("seed-confirmation runtime seed inventory drifted")
    if tuple(resolved["training"]["checkpoint_horizons_tokens"]["B"]) != HORIZONS:
        raise WorkflowError("seed-confirmation B readouts drifted")
    return resolved


def _baseline_job_ids() -> tuple[str, ...]:
    result: list[str] = []
    for experts in SELECTED_EXPERTS:
        for k in SELECTED_LANGUAGE_COUNTS:
            result.append(f"prod_trunk_b_e{experts}_k{k:03d}")
            for horizon in HORIZONS:
                result.append(
                    f"prod_branch_b_e{experts}_k{k:03d}_d{horizon}"
                )
                result.append(f"prod_eval_b_e{experts}_k{k:03d}_d{horizon}")
    return tuple(result)


def _validate_baseline_manifest() -> dict[str, Any]:
    path = (REPO_ROOT / BASELINE_MANIFEST).resolve()
    if _file_sha256(path) != BASELINE_MANIFEST_FILE_SHA256:
        raise WorkflowError("primary-seed baseline launch manifest changed")
    payload = _load_object(path)
    if payload.get("manifest_sha256") != BASELINE_WORKFLOW_SHA256:
        raise WorkflowError("primary-seed baseline workflow identity changed")
    by_id = {str(row["job_id"]): row for row in payload.get("jobs", [])}
    expected_ids = set(_baseline_job_ids())
    if not expected_ids.issubset(by_id):
        raise WorkflowError("primary-seed baseline lacks a selected B trajectory")
    for job_id in expected_ids:
        row = by_id[job_id]
        arguments = row.get("arguments", {})
        if job_id.startswith(("prod_trunk_", "prod_branch_")) and (
            arguments.get("model_seed") != PRIMARY_SEED
            or arguments.get("dataset_random_seed") != PRIMARY_SEED
            or arguments.get("lr_multiplier") != 1.0
        ):
            raise WorkflowError(f"primary-seed baseline contract drifted for {job_id}")
    return {
        "launch_manifest_path": str(path),
        "launch_manifest_file_sha256": BASELINE_MANIFEST_FILE_SHA256,
        "workflow_manifest_sha256": BASELINE_WORKFLOW_SHA256,
        "primary_seed": PRIMARY_SEED,
        "selected_trajectories": 9,
        "selected_training_and_evaluation_jobs": len(expected_ids),
        "completion_requirement": (
            "hash-valid canonical receipts are required only for final combined analysis, "
            "not for launching independent additional seeds"
        ),
    }


def _seeded_id(original_id: str, seed: int) -> str:
    if not original_id.startswith("prod_"):
        raise WorkflowError(f"cannot seed-qualify template job {original_id}")
    return f"seed_{original_id.removeprefix('prod_')}_s{seed}"


def _clone_job(
    template: Mapping[str, Any], *, seed: int, output_root: Path
) -> dict[str, Any]:
    job = deepcopy(dict(template))
    original_id = str(template["job_id"])
    job_id = _seeded_id(original_id, seed)
    job["job_id"] = job_id
    arguments = dict(job.get("arguments", {}))
    arguments["model_seed"] = seed
    arguments["dataset_random_seed"] = seed
    arguments["seed_confirmation_design_id"] = DESIGN_ID
    arguments["replication_role"] = "additional_independent_training_seed"
    if original_id.startswith("prod_trunk_"):
        job["dependencies"] = []
        output_name = "result.json"
    elif original_id.startswith("prod_branch_"):
        parent = str(arguments["parent_job_id"])
        arguments["parent_job_id"] = _seeded_id(parent, seed)
        job["dependencies"] = [arguments["parent_job_id"]]
        output_name = "result.json"
    elif original_id.startswith("prod_eval_"):
        training = str(arguments["training_job_id"])
        arguments["training_job_id"] = _seeded_id(training, seed)
        arguments["purpose"] = "seed_confirmation"
        job["dependencies"] = [arguments["training_job_id"]]
        output_name = "evaluation_manifest.json"
    else:
        raise WorkflowError(f"unexpected seed template job {original_id}")
    job["arguments"] = arguments
    job["outputs"] = [
        _rel_or_abs(output_root / "jobs" / job_id / output_name)
    ]
    return job


def _seed_job_topology(jobs: Sequence[Mapping[str, Any]]) -> None:
    by_id = {str(row["job_id"]): row for row in jobs}
    expected_trunks: set[str] = set()
    expected_branches: set[str] = set()
    expected_evaluations: set[str] = set()
    for experts in SELECTED_EXPERTS:
        for k in SELECTED_LANGUAGE_COUNTS:
            for seed in NEW_SEEDS:
                trunk = f"seed_trunk_b_e{experts}_k{k:03d}_s{seed}"
                expected_trunks.add(trunk)
                for horizon in HORIZONS:
                    branch = (
                        f"seed_branch_b_e{experts}_k{k:03d}_d{horizon}_s{seed}"
                    )
                    evaluation = (
                        f"seed_eval_b_e{experts}_k{k:03d}_d{horizon}_s{seed}"
                    )
                    expected_branches.add(branch)
                    expected_evaluations.add(evaluation)
                    if by_id.get(branch, {}).get("dependencies") != [trunk]:
                        raise WorkflowError(f"seed branch dependency drifted: {branch}")
                    if by_id.get(evaluation, {}).get("dependencies") != [branch]:
                        raise WorkflowError(
                            f"seed evaluation dependency drifted: {evaluation}"
                        )
    if {
        job_id
        for job_id in by_id
        if job_id.startswith("seed_trunk_")
    } != expected_trunks:
        raise WorkflowError("seed-confirmation trunk identities drifted")
    if {
        job_id
        for job_id in by_id
        if job_id.startswith("seed_branch_")
    } != expected_branches:
        raise WorkflowError("seed-confirmation branch identities drifted")
    if {
        job_id
        for job_id in by_id
        if job_id.startswith("seed_eval_")
    } != expected_evaluations:
        raise WorkflowError("seed-confirmation evaluation identities drifted")
    if any(by_id[job_id].get("dependencies") != [] for job_id in expected_trunks):
        raise WorkflowError("new seed trunks must be concurrently eligible")
    summary = by_id.get("seed_confirmation_summary")
    if (
        len(by_id) != 235
        or summary is None
        or summary.get("action") != "final_summary"
        or set(summary.get("dependencies", ())) != expected_evaluations
        or set(summary.get("arguments", {}).get("evaluation_jobs", ()))
        != expected_evaluations
    ):
        raise WorkflowError("seed-confirmation summary topology drifted")
    for job_id in expected_trunks | expected_branches:
        row = by_id[job_id]
        arguments = row.get("arguments", {})
        seed = int(arguments.get("model_seed", 0))
        if (
            row.get("action") != "production_train"
            or arguments.get("ladder") != "B"
            or int(arguments.get("experts", 0)) not in SELECTED_EXPERTS
            or int(arguments.get("K", 0)) not in SELECTED_LANGUAGE_COUNTS
            or seed not in NEW_SEEDS
            or int(arguments.get("dataset_random_seed", 0)) != seed
            or arguments.get("lr_multiplier") != 1.0
            or arguments.get("seed_confirmation_design_id") != DESIGN_ID
        ):
            raise WorkflowError(f"seed training contract drifted: {job_id}")


POLICY = WorkflowPolicy(
    default_spec=DEFAULT_SPEC,
    default_runtime=DEFAULT_RUNTIME,
    default_output=DEFAULT_OUTPUT,
    ladders=("B",),
    expected_micro_batch_size=EXPECTED_MICRO_BATCH_SIZE,
    required_sentinel_probes=frozenset(),
    fixed_lr_schedule_probe_cells=(),
    required_training_env_keys=frozenset(REQUIRED_TRAINING_ENV_KEYS),
    spec_validator=validate_seed_confirmation_spec,
    prescribed_lr_schedule_probe_cells=(),
)
VARIANT = WorkflowVariant(
    identifier=DESIGN_ID,
    worker_module=WORKFLOW_MODULE,
    policy=POLICY,
    production_training_seeds=NEW_SEEDS,
    production_topology_validator=_seed_job_topology,
    authorized_production=True,
    provenance_files=(Path(__file__),),
)


def _evidence_paths(runtime: Mapping[str, Any]) -> tuple[Path, ...]:
    evidence = runtime.get("training", {}).get("production_lr_evidence")
    if not isinstance(evidence, Mapping):
        raise WorkflowError("seed-confirmation runtime lacks LR evidence")
    validate_production_lr_evidence(evidence)
    result = [
        Path(str(evidence["source_gate_path"])),
        Path(str(evidence["source_recovery_path"])),
        BASELINE_MANIFEST,
    ]
    for path_key in (
        "analysis_script_path",
        "analysis_manifest_path",
        "statistical_script_path",
        "comparison_summary_path",
    ):
        result.append(Path(str(LR_ANALYSIS_EVIDENCE[path_key])))
    return tuple(result)


def _seed_storage_estimate(
    jobs: Sequence[Mapping[str, Any]], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    policy = runtime.get("storage")
    if not isinstance(policy, Mapping):
        raise WorkflowError("seed-confirmation storage policy is missing")
    bytes_per_parameter = int(policy["checkpoint_bytes_per_parameter"])
    overhead = float(policy["checkpoint_serialization_overhead_factor"])
    checkpoint_bytes = {
        experts: int(
            math.ceil(
                _model_parameter_upper_bound("B", experts)
                * bytes_per_parameter
                * overhead
            )
        )
        for experts in SELECTED_EXPERTS
    }
    counts: dict[str, int] = {}
    checkpoint_peak = 0
    for row in jobs:
        if row.get("action") != "production_train":
            continue
        arguments = row["arguments"]
        experts = int(arguments["experts"])
        count = _planned_checkpoint_count_for_job(row, runtime)
        checkpoint_peak += count * checkpoint_bytes[experts]
        key = (
            f"B/E{experts}/K{int(arguments['K']):03d}/"
            f"S{int(arguments['model_seed'])}"
        )
        counts[key] = counts.get(key, 0) + count
    non_checkpoint = int(policy["non_checkpoint_peak_reserve_bytes"])
    free_headroom = int(policy["post_allocation_free_headroom_bytes"])
    return {
        "schema_version": 1,
        "method": "actual_seed-qualified_jobs_times_conservative_full-state_checkpoint_bytes",
        "capacity_enforcement": "advisory_only",
        "go_no_go": "informational_advisory_never_blocks_submission",
        "checkpoint_bytes_per_parameter": bytes_per_parameter,
        "checkpoint_serialization_overhead_factor": overhead,
        "per_checkpoint_bytes": {
            f"B/E{experts}": checkpoint_bytes[experts]
            for experts in SELECTED_EXPERTS
        },
        "full_state_checkpoint_counts_per_trajectory": dict(sorted(counts.items())),
        "checkpoint_peak_bytes": checkpoint_peak,
        "retained_data_and_other_artifacts_reserve_bytes": non_checkpoint,
        "required_post_allocation_free_headroom_bytes": free_headroom,
        "required_available_bytes_at_preflight": (
            checkpoint_peak + non_checkpoint + free_headroom
        ),
        "checkpoint_retention_policy": (
            "retain all new-seed trunks, branch readouts, and final checkpoints through "
            "the seed-robustness analysis"
        ),
    }


def _refresh_manifest(
    manifest: dict[str, Any], runtime: Mapping[str, Any]
) -> None:
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
    manifest["staged_submission"] = {
        "phase_order": list(SUBMISSION_PHASES),
        "phase_jobs": phase_jobs,
        "phase_submit_slots": phase_slots,
        "max_submit_slots_per_invocation": int(
            runtime["max_submit_slots_per_invocation"]
        ),
        "array_elements_count_individually": True,
        "prior_phase_completion_required": False,
    }
    paths = _paths_for_manifest(manifest)
    estimate = _gpu_estimate(jobs, runtime, paths)
    manifest["gpu_hour_estimate"] = estimate
    walltime = _apply_conservative_job_time_limits(jobs, runtime, estimate)
    if walltime is None:
        manifest.pop("walltime_contract", None)
    else:
        manifest["walltime_contract"] = walltime
    manifest["storage_estimate"] = _seed_storage_estimate(jobs, runtime)
    _validate_dag(jobs)


def build_manifest(
    *, spec_path: Path, runtime_path: Path, output_root: Path
) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    runtime_path = runtime_path.resolve()
    output_root = output_root.resolve()
    validate_seed_confirmation_spec(repo_root=REPO_ROOT, spec_path=spec_path)
    runtime = _validate_runtime(runtime_path)
    baseline_reference = _validate_baseline_manifest()

    template = _api.build_manifest(
        spec_path=(REPO_ROOT / TEMPLATE_SPEC).resolve(),
        runtime_path=runtime_path,
        output_root=output_root,
        variant=TEMPLATE_VARIANT,
    )
    template_jobs = {str(row["job_id"]): row for row in template["jobs"]}
    jobs: list[dict[str, Any]] = []
    evaluations: list[str] = []
    for experts in SUBMISSION_EXPERT_ORDER:
        for k in SUBMISSION_LANGUAGE_ORDER:
            for seed in NEW_SEEDS:
                trunk_template = template_jobs[
                    f"prod_trunk_b_e{experts}_k{k:03d}"
                ]
                jobs.append(
                    _clone_job(trunk_template, seed=seed, output_root=output_root)
                )
                for horizon in HORIZONS:
                    branch_template = template_jobs[
                        f"prod_branch_b_e{experts}_k{k:03d}_d{horizon}"
                    ]
                    evaluation_template = template_jobs[
                        f"prod_eval_b_e{experts}_k{k:03d}_d{horizon}"
                    ]
                    jobs.append(
                        _clone_job(branch_template, seed=seed, output_root=output_root)
                    )
                    evaluation = _clone_job(
                        evaluation_template, seed=seed, output_root=output_root
                    )
                    jobs.append(evaluation)
                    evaluations.append(str(evaluation["job_id"]))
    jobs.append(
        {
            "job_id": "seed_confirmation_summary",
            "stage": "summary",
            "action": "final_summary",
            "resource": "cpu",
            "dependencies": list(evaluations),
            "array": None,
            "arguments": {
                "evaluation_jobs": list(evaluations),
                "seed_confirmation_design_id": DESIGN_ID,
            },
            "outputs": [
                _rel_or_abs(output_root / "control" / "seed_confirmation_summary.json")
            ],
            "gpu_tokens": 0,
        }
    )

    manifest = dict(template)
    manifest.update(
        {
            "workflow_variant": VARIANT.manifest_descriptor(),
            "workflow_module": WORKFLOW_MODULE,
            "output_root": str(output_root),
            "spec_path": str(spec_path),
            "runtime_path": str(runtime_path),
            "jobs": jobs,
            "role_counts": {
                "seed_confirmation_trunks": 18,
                "seed_confirmation_branches": 108,
                "seed_confirmation_evaluations": 108,
                "seed_confirmation_summaries": 1,
            },
            "seed_confirmation_campaign": _read_json(spec_path)[
                "seed_confirmation_design"
            ],
            "primary_seed_baseline_reference": baseline_reference,
            "submission_preflight_contract": {
                "status": "synchronous_manifest_validation_before_first_sbatch",
                "checks": [
                    "manifest and non-code fixed-input hashes",
                    "sealed corpus and tokenizer identities",
                    "JointMoE x1 LR evidence and disclosed B/E1/K1 boundary",
                    "primary seed-1234 baseline manifest identity",
                    "exact 18-trajectory seed/E/K identity and dependency DAG",
                    "Slurm command and partition availability",
                    "filesystem bytes and inode headroom as advisory observations",
                ],
                "git_policy": (
                    "commit and dirty status are recorded for provenance but never block "
                    "submission or execution; scientific non-code input hashes remain fail-closed"
                ),
                "storage_policy": "advisory_only_never_blocks_submission",
            },
            "checkpoint_resume_policy": {
                "granularity": "planned full-state segment endpoints",
                "periodic_mid_segment_checkpoints": False,
                "maximum_planned_segment_walltime_hours": 24,
                "infrastructure_recovery": (
                    "resume the exact same seed and JointMoE-x1 trajectory from its latest "
                    "hash-valid full state"
                ),
                "scientific_failure_recovery": (
                    "blocked pending documented review; no automatic seed, LR, data, or "
                    "configuration mutation"
                ),
            },
        }
    )
    cells = manifest.get("batch_ramp_micro_batch_and_gradient_accumulation", {}).get(
        "cells", {}
    )
    if isinstance(cells, Mapping):
        manifest["batch_ramp_micro_batch_and_gradient_accumulation"] = {
            **manifest["batch_ramp_micro_batch_and_gradient_accumulation"],
            "cells": {
                key: value for key, value in cells.items() if str(key).startswith("B/")
            },
        }
    with variant_scope(VARIANT):
        fixed = _fixed_hashes(spec_path, runtime_path)
    for evidence_path in _evidence_paths(runtime):
        path = (
            evidence_path.resolve()
            if evidence_path.is_absolute()
            else (REPO_ROOT / evidence_path).resolve()
        )
        fixed[_rel_or_abs(path)] = _file_sha256(path)
    manifest["fixed_input_sha256"] = dict(sorted(fixed.items()))
    _refresh_manifest(manifest, runtime)
    manifest["manifest_sha256"] = _payload_sha256(_manifest_without_hash(manifest))
    return manifest


def validate_manifest(
    payload: dict[str, Any], *, verify_inputs: bool = True
) -> dict[str, Any]:
    result = _api.validate_manifest(
        payload, verify_inputs=verify_inputs, variant=VARIANT
    )
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise WorkflowError("seed-confirmation manifest lacks jobs")
    _seed_job_topology(jobs)
    if payload.get("role_counts") != {
        "seed_confirmation_trunks": 18,
        "seed_confirmation_branches": 108,
        "seed_confirmation_evaluations": 108,
        "seed_confirmation_summaries": 1,
    }:
        raise WorkflowError("seed-confirmation role counts drifted")
    if payload.get("stage_counts") != {
        "production": 126,
        "summary": 1,
        "validation": 108,
    }:
        raise WorkflowError("seed-confirmation stage counts drifted")
    if any(
        row.get("action")
        in {
            "checkpoint_prune",
            "storage_preflight",
            "training_sentinel",
            "prescribed_lr_schedule_probe",
            "production_lr_evidence_gate",
        }
        for row in jobs
    ):
        raise WorkflowError("seed-confirmation manifest contains a gate, probe, or prune")
    if any(
        int(row.get("arguments", {}).get("model_seed", 0)) == PRIMARY_SEED
        for row in jobs
        if row.get("action") == "production_train"
    ):
        raise WorkflowError("seed-confirmation manifest recomputes seed 1234")
    if payload.get("primary_seed_baseline_reference") != _validate_baseline_manifest():
        raise WorkflowError("seed-confirmation baseline descriptor drifted")
    preflight = payload.get("submission_preflight_contract")
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("status")
        != "synchronous_manifest_validation_before_first_sbatch"
        or "never block" not in str(preflight.get("git_policy", ""))
        or preflight.get("storage_policy") != "advisory_only_never_blocks_submission"
    ):
        raise WorkflowError("seed-confirmation preflight policy drifted")
    storage = payload.get("storage_estimate", {})
    if (
        storage.get("go_no_go")
        != "informational_advisory_never_blocks_submission"
        or storage.get("capacity_enforcement") != "advisory_only"
    ):
        raise WorkflowError("seed-confirmation storage unexpectedly gates execution")
    output_root = Path(str(payload["output_root"])).resolve()
    for row in jobs:
        for raw_output in row.get("outputs", []):
            output = (REPO_ROOT / str(raw_output)).resolve()
            try:
                output.relative_to(output_root)
            except ValueError as exc:
                raise WorkflowError(
                    f"seed-confirmation output escapes its root: {output}"
                ) from exc
        if str(row.get("resource", "")).startswith("h100"):
            if (
                row.get("slurm_time_limit") != row.get("slurm_time_min")
                or float(row.get("predicted_wall_hours", 0.0)) <= 0.0
            ):
                raise WorkflowError(
                    f"seed-confirmation job lacks conservative walltime: {row['job_id']}"
                )
    runtime = _validate_runtime(Path(str(payload["runtime_path"])))
    _evidence_paths(runtime)
    if verify_inputs and not os.getenv("SLURM_JOB_ID"):
        _verify_workflow_data(payload, _paths_for_manifest(payload))
    return {
        **result,
        "new_trajectories": 18,
        "new_gpu_jobs": 234,
        "reused_primary_trajectories": 9,
        "complete_seed_panel_trajectories": 27,
        "scientific_scope": "selected B-ladder seed robustness only",
    }


def main(argv: Sequence[str] | None = None) -> int:
    return _api.main(
        argv,
        variant=VARIANT,
        build_manifest_fn=build_manifest,
        validate_manifest_fn=validate_manifest,
    )


if __name__ == "__main__":
    raise SystemExit(main())
