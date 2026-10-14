#!/usr/bin/env python3
"""D-ladder adapter for probe-gated, promoted shared workflow execution.

The D contract is expressed as immutable variant data plus D-owned promotion
logic.  It does not patch A/B implementation modules or rely on import order.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.experiments.scaling.ab_v2 import api as _workflow
from multilingual_megatron.experiments.scaling.ab_v2.common import (
    REPO_ROOT,
    SUBMISSION_PHASES,
    WorkflowError,
    _abs,
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
from multilingual_megatron.experiments.scaling.ab_v2.jobs import _add_job, _validate_dag
from multilingual_megatron.experiments.scaling.ab_v2.policy import AB_POLICY
from multilingual_megatron.experiments.scaling.ab_v2.variant import (
    WorkflowVariant,
    variant_scope,
)
from multilingual_megatron.experiments.scaling.stage1_training.d_v1.spec import validate_d_v1_spec


DEFAULT_SPEC = Path("configs/scaling_law/multilingual_moe_d_v1/study_spec.json")
DEFAULT_RUNTIME = Path(
    "configs/scaling_law/multilingual_moe_d_v1/workflow_runtime_probe.json"
)
DEFAULT_OUTPUT = Path("outputs/pipeline_runs/h100_d_v1_probe")
WORKFLOW_MODULE = "multilingual_megatron.experiments.scaling.stage1_training.d_v1.workflow"
D_LADDERS = ("D",)
D_MBS = {"D": {"1": 16, "4": 16, "16": 8, "32": 8}}
D_REQUIRED_SENTINELS = {
    ("early_correctness", "D", 1, 32),
    ("early_correctness", "D", 32, 32),
    ("mid_ramp", "D", 32, 64),
    ("plateau_memory_perf", "D", 4, 128),
    ("plateau_memory_perf", "D", 16, 128),
    ("plateau_memory_perf", "D", 32, 128),
    *{
        (f"long_gbs{global_batch_size}", "D", experts, global_batch_size)
        for experts in (1, 4, 16, 32)
        for global_batch_size in (32, 64, 128)
    },
}
D_LR_PROBES = (("D", 1, 1), ("D", 32, 200))
D_EP_RUNTIME_ENV = {
    "PROOF_BIND_LOCAL_RANK_CPUS": "socket",
    "PROOF_ENABLE_MOE_A2A_CPU_SPLIT_LISTS": "1",
}


def _replace_override(overrides: Sequence[str], value: str) -> list[str]:
    key = value.split("=", 1)[0]
    return [item for item in overrides if item.split("=", 1)[0] != key] + [value]


def _d_gpu_count(experts: int) -> int:
    if experts == 16:
        return 2
    if experts == 32:
        return 4
    return 1


def _d_resource_selector(resource: str, arguments: Mapping[str, Any]) -> str:
    if resource == "h100" and str(arguments.get("ladder", "")) == "D":
        count = _d_gpu_count(int(arguments.get("experts", 0) or 0))
        if count > 1:
            return f"h100_{count}"
    return resource


def _d_job_environment(
    _manifest: Mapping[str, Any], job: Mapping[str, Any], _runtime: Mapping[str, Any]
) -> Mapping[str, str]:
    arguments = job.get("arguments", {})
    if (
        isinstance(arguments, Mapping)
        and str(arguments.get("ladder", "")) == "D"
        and int(arguments.get("experts", 0) or 0) in {16, 32}
    ):
        return D_EP_RUNTIME_ENV
    return {}


def _d_training_overrides(overrides: list[str], *, experts: int, **_kwargs: Any) -> list[str]:
    overrides = _replace_override(overrides, "model.cuda_graph_impl=none")
    if experts in {16, 32}:
        ep = _d_gpu_count(experts)
        for value in (
            f"model.expert_model_parallel_size={ep}",
            "model.moe_token_dispatcher_type=alltoall",
            "model.overlap_moe_expert_parallel_comm=true",
            "model.delay_wgrad_compute=true",
            "optimizer.use_distributed_optimizer=true",
            "ddp.use_distributed_optimizer=true",
        ):
            overrides = _replace_override(overrides, value)
    if any("precision_aware" in item or "exp_avg_dtype" in item for item in overrides):
        raise WorkflowError("D execution must retain ordinary FP32 AdamW moments")
    return overrides


def _d_profile_transform(
    profile: dict[str, Any], *, experts: int, **_kwargs: Any
) -> dict[str, Any]:
    nproc = _d_gpu_count(experts)
    if int(profile["training"]["nproc_per_node"]) != nproc:
        raise WorkflowError("D profile process count differs from its execution contract")
    if nproc > 1:
        profile["runtime"]["env"].update(D_EP_RUNTIME_ENV)
    return profile


def _d_evaluation_spec(
    spec: dict[str, Any], *, arguments: Mapping[str, Any], **_kwargs: Any
) -> dict[str, Any]:
    experts = int(arguments["experts"])
    if experts in {16, 32}:
        spec["model_recipe"]["expected_model"].update(
            {
                "expert_model_parallel_size": _d_gpu_count(experts),
                "moe_token_dispatcher_type": "alltoall",
            }
        )
    return spec


def _d_throughput_evidence_required(runtime: Mapping[str, Any]) -> bool:
    """Probe-only D jobs establish evidence; promoted D requires it."""

    return runtime.get("probe_only_unmeasured") is not True


D_POLICY = replace(
    AB_POLICY,
    default_spec=DEFAULT_SPEC,
    default_runtime=DEFAULT_RUNTIME,
    default_output=DEFAULT_OUTPUT,
    ladders=D_LADDERS,
    expected_micro_batch_size=D_MBS,
    required_sentinel_probes=frozenset(D_REQUIRED_SENTINELS),
    fixed_lr_schedule_probe_cells=D_LR_PROBES,
    required_training_env_keys=frozenset(
        set(AB_POLICY.required_training_env_keys) | {"NCCL_GRAPH_REGISTER"}
    ),
    spec_validator=validate_d_v1_spec,
)
D_VARIANT = WorkflowVariant(
    identifier="d_v1",
    worker_module=WORKFLOW_MODULE,
    policy=D_POLICY,
    resource_selector=_d_resource_selector,
    job_environment_transform=_d_job_environment,
    training_overrides_transform=_d_training_overrides,
    profile_transform=_d_profile_transform,
    evaluation_spec_transform=_d_evaluation_spec,
    processes_for_experts=_d_gpu_count,
    throughput_evidence_required=_d_throughput_evidence_required,
    provenance_files=(
        Path(__file__),
        Path(__file__).with_name("spec.py"),
        Path(__file__).with_name("readiness.py"),
        Path(__file__).with_name("__init__.py"),
    ),
)


def _parent_inputs(path: Path) -> list[Path]:
    inputs: list[Path] = []
    current = path.resolve()
    while True:
        raw = json.loads(current.read_text(encoding="utf-8"))
        inherited = raw.get("inherits")
        if inherited is None:
            return inputs
        if not isinstance(inherited, str) or not inherited.strip():
            raise WorkflowError(f"invalid inherited runtime path in {current}")
        parent = Path(inherited)
        current = (
            parent.resolve()
            if parent.is_absolute()
            else (current.parent / parent).resolve()
        )
        if current in inputs:
            raise WorkflowError("runtime inheritance cycle")
        inputs.append(current)


def _refresh_derived_plan_fields(
    manifest: dict[str, Any], runtime: Mapping[str, Any]
) -> None:
    """Recompute controller views after a D-owned graph transformation."""

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


def _apply_production_waves(
    manifest: dict[str, Any], runtime: Mapping[str, Any]
) -> None:
    """Serialize D around two hash-gated storage-safe production waves."""

    jobs = manifest["jobs"]
    by_id = {str(row["job_id"]): row for row in jobs}
    wave_cells = {
        "d1": {
            (experts, k)
            for experts in (1, 4, 16)
            for k in (1, 12, 50, 200)
        }
        | {(32, 1)},
        "d2": {(32, 12), (32, 50), (32, 200)},
    }
    prior_prune: str | None = None
    wave_records: list[dict[str, Any]] = []
    for wave_name, cells in wave_cells.items():
        trunks = [
            row
            for row in jobs
            if str(row["job_id"]).startswith("prod_trunk_d_")
            and (int(row["arguments"]["experts"]), int(row["arguments"]["K"]))
            in cells
        ]
        if prior_prune is not None:
            for trunk in trunks:
                trunk["dependencies"] = [prior_prune]
        branches = [
            row
            for row in jobs
            if str(row["job_id"]).startswith("prod_branch_d_")
            and (int(row["arguments"]["experts"]), int(row["arguments"]["K"]))
            in cells
        ]
        evaluations = [
            row
            for row in jobs
            if str(row["job_id"]).startswith("prod_eval_d_")
            and (int(row["arguments"]["experts"]), int(row["arguments"]["K"]))
            in cells
        ]
        retained = [
            str(row["job_id"])
            for row in branches
            if int(row["arguments"]["D"]) == 32_000_000_000
        ]
        prune_sources = [str(row["job_id"]) for row in trunks] + [
            str(row["job_id"])
            for row in branches
            if str(row["job_id"]) not in retained
        ]
        summary_id = f"wave_summary_{wave_name}"
        _add_job(
            jobs,
            job_id=summary_id,
            stage="summary",
            action="final_summary",
            resource="cpu",
            dependencies=tuple(str(row["job_id"]) for row in evaluations),
            arguments={"evaluation_jobs": [str(row["job_id"]) for row in evaluations]},
            outputs=(
                Path(str(manifest["output_root"]))
                / "control"
                / "waves"
                / f"{wave_name}_summary.json",
            ),
        )
        prune_id = f"retention_prune_{wave_name}"
        _add_job(
            jobs,
            job_id=prune_id,
            stage="retention",
            action="checkpoint_prune",
            resource="cpu",
            dependencies=(summary_id,),
            arguments={
                "source_jobs": prune_sources,
                "retained_jobs": retained,
                "evidence_jobs": [summary_id, *[str(row["job_id"]) for row in evaluations]],
                "retention_scope": (
                    f"D {wave_name}: retain only the largest-D branch for each cell "
                    "before releasing the next storage wave"
                ),
            },
            outputs=(
                Path(str(manifest["output_root"]))
                / "control"
                / "prune"
                / f"{prune_id}.json",
            ),
        )
        prior_prune = prune_id
        wave_records.append(
            {
                "wave": wave_name,
                "cells": [
                    {"experts": experts, "K": k}
                    for experts, k in sorted(cells)
                ],
                "trunks": [str(row["job_id"]) for row in trunks],
                "branches": [str(row["job_id"]) for row in branches],
                "evaluations": [str(row["job_id"]) for row in evaluations],
                "summary": summary_id,
                "prune": prune_id,
                "retained_jobs": retained,
            }
        )

    by_id["retention_prune_production_after_summary"]["dependencies"] = [
        "final_summary",
        str(prior_prune),
    ]
    manifest["production_waves"] = wave_records
    _refresh_derived_plan_fields(manifest, runtime)

    storage = dict(manifest["storage_estimate"])
    per_checkpoint = {
        int(key.split("E", 1)[1]): int(value)
        for key, value in storage["per_checkpoint_bytes"].items()
    }
    checkpoint_counts = {
        int(key.split("E", 1)[1]): int(value["total"])
        for key, value in storage["production_checkpoint_counts_per_cell"].items()
    }
    retained_bytes = 0
    wave_storage: list[dict[str, Any]] = []
    for wave_name, cells in wave_cells.items():
        active_bytes = sum(
            checkpoint_counts[experts] * per_checkpoint[experts]
            for experts, _k in cells
        )
        peak_bytes = retained_bytes + active_bytes
        retained_after = retained_bytes + sum(
            per_checkpoint[experts] for experts, _k in cells
        )
        wave_storage.append(
            {
                "wave": wave_name,
                "retained_before_bytes": retained_bytes,
                "active_checkpoint_bytes": active_bytes,
                "checkpoint_peak_bytes": peak_bytes,
                "retained_after_bytes": retained_after,
            }
        )
        retained_bytes = retained_after
    production_peak = max(row["checkpoint_peak_bytes"] for row in wave_storage)
    checkpoint_peak = max(
        int(storage["phase_checkpoint_peak_bytes"]["sentinel"]),
        int(storage["phase_checkpoint_peak_bytes"]["lr_verification"]),
        production_peak,
    )
    storage.update(
        {
            "schema_version": 2,
            "method": "two_wave_full_state_peak_with_hash_gated_pruning",
            "production_waves": wave_storage,
            "phase_checkpoint_peak_bytes": {
                **storage["phase_checkpoint_peak_bytes"],
                "production": production_peak,
            },
            "checkpoint_peak_bytes": checkpoint_peak,
            "required_available_bytes_at_preflight": (
                checkpoint_peak
                + int(storage["retained_data_and_other_artifacts_reserve_bytes"])
                + int(storage["required_post_allocation_free_headroom_bytes"])
            ),
        }
    )
    manifest["storage_estimate"] = storage


def _reuse_promoted_preproduction_evidence(
    manifest: dict[str, Any], runtime: Mapping[str, Any]
) -> None:
    """Replace duplicate D probe jobs with hash-bound promotion evidence."""

    jobs = list(manifest["jobs"])
    removed = {
        str(row["job_id"])
        for row in jobs
        if _submission_phase(row) in {"sentinel", "lr_verification"}
    }
    retained: list[dict[str, Any]] = []
    for original in jobs:
        job = dict(original)
        job_id = str(job["job_id"])
        if job_id in removed:
            continue
        removed_dependencies = removed.intersection(
            str(dependency) for dependency in job.get("dependencies", [])
        )
        if removed_dependencies:
            if (
                job_id.startswith("prod_trunk_")
                and removed_dependencies == {"fixed_lr_schedule_gate"}
            ):
                job["dependencies"] = ["data_megatron_reader_smoke"]
            else:
                raise WorkflowError(
                    f"{job_id} unexpectedly depends on promoted probe jobs: "
                    + ", ".join(sorted(removed_dependencies))
                )
        retained.append(job)
    manifest["jobs"] = retained
    manifest["promoted_preproduction_evidence"] = {
        "contract": (
            "D-specific hash-bound runtime promotion replaces duplicate sentinel "
            "and fixed-LR implementation probes; production-local storage and data "
            "reader preflights remain mandatory"
        ),
        "promotion_receipt_path": str(
            Path(str(runtime["production_promotion_receipt_path"])).resolve()
        ),
        "reused_phases": ["sentinel", "lr_verification"],
        "removed_job_ids": sorted(removed),
        "production_entry_gate": "data_megatron_reader_smoke",
    }
    manifest["role_counts"] = {
        "fixed_lr_schedule_probes": 0,
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
    _refresh_derived_plan_fields(manifest, runtime)


def _remove_probe_only_production(
    manifest: dict[str, Any], runtime: Mapping[str, Any]
) -> None:
    jobs = [
        row for row in manifest["jobs"] if _submission_phase(row) != "production"
    ]
    manifest["jobs"] = jobs
    manifest["role_counts"] = {
        "fixed_lr_schedule_probes": sum(
            row["action"] == "fixed_lr_schedule_probe" for row in jobs
        ),
        "production_trunks": 0,
        "production_branches": 0,
        "production_evaluations": 0,
    }
    _refresh_derived_plan_fields(manifest, runtime)


def build_manifest(
    *,
    spec_path: Path,
    runtime_path: Path,
    output_root: Path,
    skip_promotion_validation: bool = False,
) -> dict[str, Any]:
    runtime = _load_object(runtime_path)
    if runtime.get("probe_only_unmeasured") is not True and not skip_promotion_validation:
        from multilingual_megatron.experiments.scaling.stage1_training.d_v1.readiness import validate_runtime_promotion

        receipt_path = Path(str(runtime.get("production_promotion_receipt_path", "")))
        validate_runtime_promotion(
            runtime_path=runtime_path.resolve(), receipt_path=receipt_path.resolve()
        )
    with variant_scope(D_VARIANT):
        manifest = _workflow.build_manifest(
            spec_path=spec_path,
            runtime_path=runtime_path,
            output_root=output_root,
            variant=D_VARIANT,
        )
        if runtime.get("probe_only_unmeasured") is True:
            _remove_probe_only_production(manifest, runtime)
        else:
            _reuse_promoted_preproduction_evidence(manifest, runtime)
            _apply_production_waves(manifest, runtime)
        contract = dict(manifest["resolved_training_profile_contract"])
        contract.update(
            {
                "source": "configs/scaling_law/multilingual_moe_d_v1/study_spec.json",
                "cuda_graph_impl": "none",
                "d_execution": "E1/E4 EP1; E16 EP2; E32 EP4; all multi-rank paths use all-to-all",
                "optimizer_moments": "ordinary FP32 AdamW defaults",
                "probe_only_unmeasured": bool(runtime.get("probe_only_unmeasured")),
            }
        )
        manifest["resolved_training_profile_contract"] = contract
        fixed = dict(manifest["fixed_input_sha256"])
        for path in _parent_inputs(runtime_path):
            fixed[_rel_or_abs(path)] = _file_sha256(path)
        if runtime.get("probe_only_unmeasured") is not True:
            promotion = Path(str(runtime["production_promotion_receipt_path"]))
            fixed[_rel_or_abs(promotion)] = _file_sha256(promotion)
            evidence = runtime.get("performance_budget", {}).get(
                "training_throughput_evidence", {}
            )
            for descriptor in evidence.values():
                if isinstance(descriptor, Mapping) and descriptor.get("report"):
                    report = Path(str(descriptor["report"]))
                    fixed[_rel_or_abs(report)] = _file_sha256(report)
        manifest["fixed_input_sha256"] = dict(sorted(fixed.items()))
        manifest["production_generation_enabled"] = not bool(
            runtime.get("probe_only_unmeasured")
        )
        manifest["manifest_sha256"] = _payload_sha256(_manifest_without_hash(manifest))
        return manifest


def validate_manifest(
    payload: Mapping[str, Any], *, verify_inputs: bool = True
) -> dict[str, Any]:
    if payload.get("study_id") != "multilingual_moe_d_v1":
        raise WorkflowError("D workflow manifest has the wrong study_id")
    if payload.get("workflow_module") != WORKFLOW_MODULE:
        raise WorkflowError("D workflow manifest must launch the D worker module")
    validate_d_v1_spec(repo_root=REPO_ROOT, spec_path=Path(str(payload["spec_path"])))
    result = _workflow.validate_manifest(
        dict(payload), verify_inputs=verify_inputs, variant=D_VARIANT
    )
    runtime = _load_object(Path(str(payload["runtime_path"])))
    expected_dp = {"D": {"1": 1, "4": 1, "16": 2, "32": 4}}
    if runtime.get("training", {}).get("data_parallel_size_by_ladder_and_experts") != expected_dp:
        raise WorkflowError("D must freeze DP=1/1/2/4 for E1/E4/E16/E32")
    if bool(runtime.get("probe_only_unmeasured")) == bool(
        payload.get("production_generation_enabled")
    ):
        raise WorkflowError("D probe/production enablement binding is inconsistent")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    if arguments and arguments[0] == "submit" and "--through-stage" in arguments:
        stage = arguments[arguments.index("--through-stage") + 1]
        manifest_path = Path(arguments[arguments.index("--manifest") + 1])
        manifest = _load_object(_abs(manifest_path))
        if stage == "production" and manifest.get("production_generation_enabled") is not True:
            print(
                "[ERROR] D production is blocked until D-specific probe evidence is promoted",
                file=sys.stderr,
            )
            return 2
    return _workflow.main(arguments, variant=D_VARIANT)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["D_VARIANT", "build_manifest", "main", "validate_manifest"]
