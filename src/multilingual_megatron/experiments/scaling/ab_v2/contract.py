"""Build and validate the immutable A/B-v2 launch manifest."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifact_data import _verify_data_artifact_seal
from .common import (
    DATA_ARTIFACT_SEAL_NAME,
    EXPERTS,
    FIXED_PEAK_LEARNING_RATE,
    LANGUAGE_COUNTS,
    MANIFEST_NAME,
    PRIMARY_TRAINING_SEED,
    Paths,
    REPLICATION_TRAINING_SEEDS,
    REPO_ROOT,
    REQUIRED_DATA_ENV_KEYS,
    SUBMISSION_PHASES,
    VALIDATION_CANDIDATE_FRACTION,
    VALIDATION_TARGET_FRACTION,
    WORKFLOW_SCHEMA_VERSION,
    WorkflowError,
    _abs,
    _file_sha256,
    _fixed_hashes,
    _git_provenance,
    _is_h100_resource,
    _learning_rate_prescription,
    _load_object,
    _payload_sha256,
    _rel_or_abs,
    _resource_gpu_count,
    _submission_phase,
    _submit_slot_count,
    _workflow_source_files,
)
from .estimates import (
    _batch_layout,
    _gpu_estimate,
    _performance_budget,
    _storage_estimate,
    _training_data_parallel_size,
    _validate_training_throughput_schedule,
    _verify_training_throughput_evidence,
)
from . import jobs as job_builders
from . import lr_confirmation
from .job_data import _data_jobs
from .job_validation import _validate_dag, _validate_lr_contract
from .variant import active_policy, active_variant


def build_manifest(
    *,
    spec_path: Path,
    runtime_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    policy = active_policy()
    variant = active_variant()
    spec_path = spec_path.resolve()
    runtime_path = runtime_path.resolve()
    output_root = output_root.resolve()
    policy.spec_validator(repo_root=REPO_ROOT, spec_path=spec_path)
    spec = _load_object(spec_path)
    runtime = _load_object(runtime_path)
    training_policy = spec.get("training_policy", {})
    legacy_fixed_lr = not isinstance(
        training_policy.get("learning_rate_prescription")
        if isinstance(training_policy, Mapping)
        else None,
        Mapping,
    )
    lr_prescription = None if legacy_fixed_lr else _learning_rate_prescription(spec)
    expected_runtime_schema = 1 if legacy_fixed_lr else 2
    if (
        runtime.get("schema_version") != expected_runtime_schema
        or runtime.get("study_id") != spec.get("study_id")
    ):
        raise WorkflowError("runtime template does not match the frozen A/B-v2 study")
    runtime_data = runtime.get("data")
    validation_policy = spec.get("validation_policy")
    if not isinstance(runtime_data, dict) or not isinstance(validation_policy, dict):
        raise WorkflowError(
            "runtime data and spec validation policy must both be objects"
        )
    runtime_optimizer = runtime.get("training", {}).get("optimizer")
    if not isinstance(runtime_optimizer, dict):
        raise WorkflowError("runtime training.optimizer must be an object")
    if not isinstance(runtime.get("performance_budget"), Mapping):
        raise WorkflowError("production runtime requires an explicit performance_budget")
    _performance_budget(runtime)
    _validate_training_throughput_schedule(runtime)
    if variant.throughput_evidence_required(runtime):
        _verify_training_throughput_evidence(runtime)
    if legacy_fixed_lr:
        try:
            runtime_peak_lr = float(runtime_optimizer["peak_learning_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError("historical runtime lacks fixed peak LR") from exc
        if not math.isclose(
            runtime_peak_lr, FIXED_PEAK_LEARNING_RATE, rel_tol=0.0, abs_tol=0.0
        ):
            raise WorkflowError("historical runtime fixed peak LR drifted")
    elif "peak_learning_rate" in runtime_optimizer:
        raise WorkflowError(
            "runtime optimizer must not override the JointMoE paper LR prescription"
        )
    data_mode = str(runtime_data.get("mode", "fresh"))
    if data_mode not in {"fresh", "adopt_k12_migration", "reuse_sealed_k12"}:
        raise WorkflowError(f"unsupported A/B-v2 data mode: {data_mode}")
    adoption_predecessor_root: Path | None = None
    if data_mode == "adopt_k12_migration":
        raw_predecessor = runtime_data.get("adoption_predecessor_root")
        if not isinstance(raw_predecessor, str) or not raw_predecessor.strip():
            raise WorkflowError(
                "adopt_k12_migration requires data.adoption_predecessor_root"
            )
        adoption_predecessor_root = _abs(raw_predecessor)
        if adoption_predecessor_root == output_root:
            raise WorkflowError("K12 successor output must differ from its predecessor")
    sealed_corpus: dict[str, Any] | None = None
    if data_mode == "reuse_sealed_k12":
        raw_source = runtime_data.get("sealed_corpus_root")
        if not isinstance(raw_source, str) or not raw_source.strip():
            raise WorkflowError("reuse_sealed_k12 requires data.sealed_corpus_root")
        source_root = _abs(raw_source)
        if source_root == output_root:
            raise WorkflowError("sealed corpus source must differ from the new workflow root")
        source_manifest_path = source_root / "control" / MANIFEST_NAME
        source_seal_path = source_root / "control" / DATA_ARTIFACT_SEAL_NAME
        if not source_manifest_path.is_file() or not source_seal_path.is_file():
            raise WorkflowError("sealed corpus source lacks its manifest or data seal")
        source_manifest = _load_object(source_manifest_path)
        source_seal = _load_object(source_seal_path)
        if source_manifest.get("output_root") != str(source_root):
            raise WorkflowError("sealed corpus source manifest has another output root")
        if source_seal.get("workflow_manifest_sha256") != source_manifest.get(
            "manifest_sha256"
        ):
            raise WorkflowError("sealed corpus source seal is not bound to its manifest")
        sealed_corpus = {
            "root": str(source_root),
            "launch_manifest": str(source_manifest_path),
            "launch_manifest_sha256": _file_sha256(source_manifest_path),
            "workflow_manifest_sha256": source_manifest.get("manifest_sha256"),
            "data_seal": str(source_seal_path),
            "data_seal_sha256": _file_sha256(source_seal_path),
            "data_seal_payload_sha256": source_seal.get("seal_payload_sha256"),
        }
        source_paths = Paths(source_root)
        _verify_data_artifact_seal(source_manifest, source_paths)
    candidate_fraction = runtime_data.get("validation_candidate_fraction")
    target_fraction = runtime_data.get("validation_target_fraction")
    spec_candidate_fraction = validation_policy.get(
        "pre_tokenizer_candidate_reserve_fraction"
    )
    spec_target_fraction = validation_policy.get("final_validation_token_fraction")
    if candidate_fraction != VALIDATION_CANDIDATE_FRACTION:
        raise WorkflowError(
            "runtime validation_candidate_fraction must be frozen at 0.30"
        )
    if target_fraction != VALIDATION_TARGET_FRACTION:
        raise WorkflowError(
            "runtime validation_target_fraction must be frozen at 0.20"
        )
    if candidate_fraction <= target_fraction:
        raise WorkflowError(
            "validation candidate reserve must exceed the final validation target"
        )
    if (
        candidate_fraction != spec_candidate_fraction
        or target_fraction != spec_target_fraction
    ):
        raise WorkflowError(
            "runtime validation fractions differ from the frozen spec validation policy"
        )
    expected_seeds = (
        {"data": 42, "model": PRIMARY_TRAINING_SEED}
        if legacy_fixed_lr
        else {
            "data": 42,
            "primary_training": PRIMARY_TRAINING_SEED,
            "replication_training": list(REPLICATION_TRAINING_SEEDS),
        }
    )
    if runtime["seeds"] != expected_seeds:
        raise WorkflowError("A/B-v2 training/corpus seed policy drifted")
    if tuple(runtime["training"]["experts"]) != EXPERTS:
        raise WorkflowError("runtime expert axis differs from the frozen study")
    if tuple(runtime["training"]["language_counts"]) != LANGUAGE_COUNTS:
        raise WorkflowError("runtime language axis differs from the frozen study")
    diagnostics = runtime["training"].get("learning_rate_diagnostics")
    is_authorized_production = variant.authorized_production
    is_lr_confirmation = lr_confirmation._validate_lr_diagnostic_mode(
        diagnostics,
        legacy_fixed_lr=legacy_fixed_lr,
        variant_identifier=variant.identifier,
        authorized_production=is_authorized_production,
    )
    if runtime["training"].get("micro_batch_size") != policy.expected_micro_batch_size:
        raise WorkflowError(
            "runtime MBS map must be A/E*=64, B/E1,E4,E16=64, and B/E32=32"
        )
    runtime_env = runtime["training"].get("runtime_env")
    if (
        not isinstance(runtime_env, dict)
        or set(runtime_env) != policy.required_training_env_keys
    ):
        raise WorkflowError(
            "runtime training env must contain exactly the frozen clean key set"
        )
    if (
        runtime_env.get("CUDA_DEVICE_MAX_CONNECTIONS") != "8"
        or runtime_env.get("NVTE_FUSED_ATTN") != "1"
        or runtime_env.get("LD_LIBRARY_PATH") != "${CONDA_PREFIX}/lib"
        or any(runtime_env.get(key) == "/tmp" for key in ("TMPDIR", "TMP", "TEMP"))
        or any(key in runtime_env for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
    ):
        raise WorkflowError(
            "runtime training env contains a legacy or unapproved value"
        )
    data_env = runtime["data"].get("runtime_env")
    if not isinstance(data_env, dict) or set(data_env) != REQUIRED_DATA_ENV_KEYS:
        raise WorkflowError(
            "runtime data env must contain exactly the frozen clean key set"
        )
    if any(key in data_env for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise WorkflowError("runtime data env must not force offline mode")
    sentinels = runtime["training"].get("sentinels")
    if not isinstance(sentinels, list):
        raise WorkflowError("runtime sentinels must be a list")
    sentinel_probes = [
        (
            str(row.get("kind", "")),
            str(row.get("ladder", "")),
            int(row.get("experts", 0)),
            int(row.get("global_batch_size", 0)),
        )
        for row in sentinels
        if isinstance(row, dict)
    ]
    if len(sentinel_probes) != len(set(sentinel_probes)):
        raise WorkflowError("runtime sentinel probes must be unique")
    missing_sentinels = sorted(policy.required_sentinel_probes - set(sentinel_probes))
    if missing_sentinels:
        raise WorkflowError(
            f"runtime lacks required production-path sentinels: {missing_sentinels}"
        )
    for row in sentinels:
        ladder = str(row["ladder"])
        experts_int = int(row["experts"])
        experts = str(experts_int)
        gbs = int(row["global_batch_size"])
        if gbs not in {32, 64, 128}:
            raise WorkflowError(
                f"sentinel {ladder}/E{experts}/GBS{gbs} has invalid production batch path"
            )
        # Resolve the actual Megatron batch tuple rather than validating only
        # GBS/MBS.  Expert-parallel C/E32 runs have DP=2, so a configuration
        # can divide by MBS while still failing the true GBS=MBS*DP*GA rule.
        mbs, dp, grad_accumulation_steps = _batch_layout(
            runtime,
            ladder=ladder,
            experts=experts_int,
            global_batch_size=gbs,
        )
        if mbs * dp * grad_accumulation_steps != gbs:
            raise WorkflowError(
                f"sentinel {ladder}/E{experts}/GBS{gbs} does not resolve to an exact MBS/DP/GA tuple"
            )
    expected_module_init = Path("/etc/profile.d/z00_lmod.sh")
    expected_conda_env = Path(str(runtime["bridge_python_executable"])).parent.parent
    for resource_name, resource in runtime["resources"].items():
        resource_flags = [str(flag) for flag in resource.get("extra_sbatch_flags", [])]
        if any("--exclusive" in flag for flag in resource_flags):
            raise WorkflowError("A/B-v2 forbids --exclusive")
        if "--export=NIL" not in resource_flags or any(
            flag.startswith("--export=") and flag != "--export=NIL"
            for flag in resource_flags
        ):
            raise WorkflowError("every A/B-v2 resource must use only --export=NIL")
        module_init = Path(str(resource.get("module_init", "")))
        if not module_init.is_absolute():
            raise WorkflowError(
                "every A/B-v2 resource requires an absolute module_init path"
            )
        if module_init != expected_module_init:
            raise WorkflowError(
                "every A/B-v2 resource must use the frozen HPC Lmod profile init"
            )
        conda_env = Path(str(resource.get("conda_env", "")))
        if conda_env != expected_conda_env:
            raise WorkflowError(
                "every A/B-v2 resource must activate the absolute frozen Bridge environment"
            )
        if _is_h100_resource(resource_name):
            _resource_gpu_count(runtime, str(resource_name))
            if resource.get("partition") != "gpu_h100":
                raise WorkflowError(
                    f"GPU resource {resource_name!r} must use partition gpu_h100"
                )
        elif str(resource.get("gres", "none")).lower() not in {"none", "cpu"}:
            raise WorkflowError(f"CPU resource {resource_name!r} must not request GPUs")

    paths = Paths(
        output_root,
        data_root=(Path(str(sealed_corpus["root"])) if sealed_corpus else None),
    )
    jobs: list[dict[str, Any]] = []
    data_gate = _data_jobs(jobs, paths, runtime)
    sentinel_gate = job_builders._sentinel_jobs(
        jobs,
        data_gate=data_gate,
        runtime=runtime,
        paths=paths,
        lr_prescription=lr_prescription,
    )
    if legacy_fixed_lr:
        production_gate = job_builders._fixed_lr_verification_jobs(
            jobs,
            sentinel_gate=sentinel_gate,
            paths=paths,
            runtime=runtime,
        )
    else:
        if is_authorized_production and not policy.prescribed_lr_schedule_probe_cells:
            lr_verification_gate = sentinel_gate
        else:
            lr_verification_gate = job_builders._prescribed_lr_verification_jobs(
                jobs,
                sentinel_gate=sentinel_gate,
                paths=paths,
                lr_prescription=lr_prescription,
                runtime=runtime,
            )
        if is_lr_confirmation:
            production_gate = lr_confirmation._lr_confirmation_jobs(
                jobs,
                verification_gate=lr_verification_gate,
                paths=paths,
                lr_prescription=lr_prescription,
                runtime=runtime,
            )
        elif is_authorized_production:
            production_gate = job_builders._production_lr_evidence_jobs(
                jobs,
                verification_gate=lr_verification_gate,
                paths=paths,
                runtime=runtime,
            )
        else:
            production_gate = job_builders._lr_diagnostic_jobs(
                jobs,
                verification_gate=lr_verification_gate,
                paths=paths,
                lr_prescription=lr_prescription,
                runtime=runtime,
            )
    if not is_lr_confirmation:
        job_builders._production_jobs(
            jobs,
            diagnostic_gate=production_gate,
            lr_prescription=lr_prescription,
            runtime=runtime,
            paths=paths,
        )
    _validate_lr_contract(jobs, prescription=lr_prescription)
    _validate_dag(jobs)
    undefined_resources = sorted(
        {str(row["resource"]) for row in jobs} - set(runtime["resources"])
    )
    if undefined_resources:
        raise WorkflowError(
            "workflow jobs reference undefined resource classes: "
            + ", ".join(undefined_resources)
        )
    counts = Counter(str(row["stage"]) for row in jobs)
    expected_trunks = (
        0 if is_lr_confirmation else len(policy.ladders) * len(EXPERTS) * len(LANGUAGE_COUNTS)
    )
    expected_branches = sum(
        len(runtime["training"]["checkpoint_horizons_tokens"][ladder])
        * len(EXPERTS)
        * len(LANGUAGE_COUNTS)
        for ladder in policy.ladders
    ) if not is_lr_confirmation else 0
    production_role_counts = {
        "production_trunks": expected_trunks,
        "production_branches": expected_branches,
        "production_evaluations": expected_branches,
    }
    if legacy_fixed_lr:
        role_counts = {
            "fixed_lr_schedule_probes": sum(
                row["action"] == "fixed_lr_schedule_probe" for row in jobs
            ),
            **production_role_counts,
        }
        expected_role_counts = {
            "fixed_lr_schedule_probes": len(policy.fixed_lr_schedule_probe_cells),
            **production_role_counts,
        }
    elif is_lr_confirmation:
        role_counts, expected_role_counts = lr_confirmation._lr_confirmation_role_counts(
            jobs,
            prescribed_probe_count=len(policy.prescribed_lr_schedule_probe_cells),
        )
    elif is_authorized_production:
        role_counts = {
            "prescribed_lr_schedule_probes": sum(
                row["action"] == "prescribed_lr_schedule_probe" for row in jobs
            ),
            **production_role_counts,
        }
        expected_role_counts = {
            "prescribed_lr_schedule_probes": len(
                policy.prescribed_lr_schedule_probe_cells
            ),
            **production_role_counts,
        }
    else:
        role_counts = {
            "prescribed_lr_schedule_probes": sum(
                row["action"] == "prescribed_lr_schedule_probe" for row in jobs
            ),
            "lr_diagnostic_trajectories": sum(
                row["job_id"].startswith("lrdiag_eval_") for row in jobs
            ),
            "lr_diagnostic_seed_trajectories": sum(
                row["job_id"].startswith("lrdiag_eval_")
                and "seed_replication"
                in row.get("arguments", {}).get("diagnostic_roles", [])
                for row in jobs
            ),
            "lr_diagnostic_neighbor_trajectories": sum(
                row["job_id"].startswith("lrdiag_eval_")
                and "lr_transfer_neighbor"
                in row.get("arguments", {}).get("diagnostic_roles", [])
                for row in jobs
            ),
            **production_role_counts,
        }
        expected_role_counts = {
            "prescribed_lr_schedule_probes": len(
                policy.prescribed_lr_schedule_probe_cells
            ),
            "lr_diagnostic_trajectories": 32,
            "lr_diagnostic_seed_trajectories": 24,
            "lr_diagnostic_neighbor_trajectories": 8,
            **production_role_counts,
        }
    if role_counts != expected_role_counts:
        raise WorkflowError(
            f"unexpected frozen workflow cardinality: {role_counts}; expected {expected_role_counts}"
        )

    phase_jobs: dict[str, list[str]] = {phase: [] for phase in SUBMISSION_PHASES}
    phase_slots: dict[str, int] = {phase: 0 for phase in SUBMISSION_PHASES}
    for row in jobs:
        phase = _submission_phase(row)
        phase_jobs[phase].append(str(row["job_id"]))
        phase_slots[phase] += _submit_slot_count(row)
    submit_limit = int(runtime["max_submit_slots_per_invocation"])
    oversized = {
        phase: slots for phase, slots in phase_slots.items() if slots > submit_limit
    }
    if oversized:
        raise WorkflowError(
            f"submission phase exceeds the frozen Slurm slot limit: {oversized}"
        )

    manifest: dict[str, Any] = {
        "schema_version": 1 if legacy_fixed_lr else WORKFLOW_SCHEMA_VERSION,
        "study_id": str(spec["study_id"]),
        "workflow_variant": variant.manifest_descriptor(),
        "workflow_module": variant.worker_module,
        "output_root": str(output_root),
        "repo_root": str(REPO_ROOT),
        "spec_path": str(spec_path),
        "runtime_path": str(runtime_path),
        "fresh_data_only": data_mode == "fresh",
        "data_materialization": {
            "mode": data_mode,
            "predecessor_root": (
                str(adoption_predecessor_root)
                if adoption_predecessor_root is not None
                else None
            ),
            "contract": (
                "hash-verified K12 migration into a distinct successor root; canonical "
                "scientific audit and new workflow-bound seal required before H100"
                if data_mode == "adopt_k12_migration"
                else (
                    "source K12 corpus remains immutable; every data consumer verifies the "
                    "source workflow manifest and data-artifact seal before use"
                    if data_mode == "reuse_sealed_k12"
                    else "fresh end-to-end v2 corpus construction"
                )
            ),
            "sealed_corpus": sealed_corpus,
        },
        "nested_slurm_forbidden": True,
        **(
            {"fixed_peak_learning_rate": FIXED_PEAK_LEARNING_RATE}
            if legacy_fixed_lr
            else {"learning_rate_prescription": lr_prescription}
        ),
        "fixed_input_sha256": _fixed_hashes(spec_path, runtime_path),
        "git_provenance": _git_provenance(),
        "stage_counts": dict(sorted(counts.items())),
        "role_counts": role_counts,
        "staged_submission": {
            "phase_order": list(SUBMISSION_PHASES),
            "phase_jobs": phase_jobs,
            "phase_submit_slots": phase_slots,
            "max_submit_slots_per_invocation": submit_limit,
            "array_elements_count_individually": True,
            "prior_phase_completion_required": not is_lr_confirmation,
        },
        "batch_ramp_micro_batch_and_gradient_accumulation": _batch_ramp_accounting(
            runtime
        ),
        "checkpoint_resume_policy": {
            "granularity": "planned_segment_endpoints_only",
            "periodic_mid_segment_checkpoints": False,
            "maximum_planned_segment_walltime_hours": 7,
            "reason": (
                "bounded segments plus completion-oriented per-job limits make "
                "endpoint-only recovery compatible with checkpoint retention"
            ),
        },
        "resolved_training_profile_contract": {
            "runtime_env": dict(runtime["training"]["runtime_env"]),
            "slurm_export_mode": "NIL",
            "module_init": str(runtime["resources"]["h100"]["module_init"]),
            "expected_python_executable": str(runtime["bridge_python_executable"]),
            "cuda_preflight": True,
            "optional_dependencies": {
                "modelopt": "real package required and proven in every training log",
                "nvrx": "real inprocess retained; incompatible async checkpoint modules disabled",
            },
            "cuda_graph_impl": "none",
            "checkpoint_async_save": False,
            "dense_transformer_engine_op_fuser": False,
            "moe_transformer_engine_op_fuser": True,
            "legacy_runtime_env_forbidden": [
                "CUDA/13",
                "TMPDIR=/tmp",
                "CUDA_DEVICE_MAX_CONNECTIONS=1",
                "--signal",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            ],
            "source": "configs/scaling/h100_ab_ladder_launch_readiness.yaml",
        },
        "jobs": jobs,
    }
    if is_lr_confirmation:
        manifest["lr_confirmation_campaign"] = (
            lr_confirmation._lr_confirmation_campaign_descriptor(runtime)
        )
    estimate = _gpu_estimate(jobs, runtime, paths)
    manifest["gpu_hour_estimate"] = estimate
    walltime_contract = _apply_conservative_job_time_limits(jobs, runtime, estimate)
    if walltime_contract is not None:
        manifest["walltime_contract"] = walltime_contract
    manifest["storage_estimate"] = _storage_estimate(jobs, runtime)
    manifest["manifest_sha256"] = _payload_sha256(manifest)
    return manifest


def _batch_ramp_accounting(runtime: Mapping[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for ladder in active_policy().ladders:
        for experts in EXPERTS:
            plateau_mbs = int(
                runtime["training"]["micro_batch_size"][ladder][str(experts)]
            )
            data_parallel_size = _training_data_parallel_size(
                runtime, ladder=ladder, experts=experts
            )
            segments: list[dict[str, int]] = []
            for gbs in (32, 64, 128):
                from multilingual_megatron.pipeline.stages.training.batch_policy import (
                    derive_segment_micro_batch_size,
                )

                mbs = derive_segment_micro_batch_size(
                    plateau_micro_batch_size=plateau_mbs,
                    global_batch_size=gbs,
                    data_parallel_size=data_parallel_size,
                )
                if gbs % (mbs * data_parallel_size):
                    raise WorkflowError(
                        f"GBS {gbs} is not divisible by MBS {mbs} and DP {data_parallel_size}"
                    )
                segments.append(
                    {
                        "global_batch_sequences": gbs,
                        "micro_batch_sequences": mbs,
                        "gradient_accumulation_steps": gbs // (mbs * data_parallel_size),
                    }
                )
            rows[f"{ladder}/E{experts}"] = {
                "plateau_micro_batch_sequences": plateau_mbs,
                "data_parallel_size": data_parallel_size,
                "segments": segments,
            }
    return {
        "sequence_length": 2048,
        "rule": "gradient_accumulation_steps = GBS / (MBS * DP)",
        "cells": rows,
    }


from .contract_timing import (
    _apply_conservative_job_time_limits,
    _format_slurm_duration,
    _parse_slurm_duration,
)
from .contract_validation import _manifest_without_hash, validate_manifest
