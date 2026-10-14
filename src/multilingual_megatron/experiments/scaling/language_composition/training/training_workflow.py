#!/usr/bin/env python3
"""Stage-1 K12 language-composition sensitivity workflow."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.experiments.scaling.ab_v2 import api as _api
from multilingual_megatron.experiments.scaling.ab_v2.artifact_data import (
    _verify_workflow_data,
)
from multilingual_megatron.experiments.scaling.ab_v2.common import (
    EXPECTED_MICRO_BATCH_SIZE,
    REQUIRED_TRAINING_ENV_KEYS,
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
from multilingual_megatron.experiments.scaling.ab_v2.common_provenance import (
    _fixed_hashes,
)
from multilingual_megatron.experiments.scaling.ab_v2.contract_timing import (
    _apply_conservative_job_time_limits,
)
from multilingual_megatron.experiments.scaling.ab_v2.contract_validation import (
    _manifest_without_hash,
)
from multilingual_megatron.experiments.scaling.ab_v2.estimates import _gpu_estimate
from multilingual_megatron.experiments.scaling.ab_v2.job_validation import _validate_dag
from multilingual_megatron.experiments.scaling.ab_v2.variant import (
    WorkflowPolicy,
    WorkflowVariant,
    variant_scope,
)
from multilingual_megatron.experiments.scaling.stage1_training.ab_production import DEFAULT_SPEC as TEMPLATE_SPEC
from multilingual_megatron.experiments.scaling.stage1_training.ab_production import VARIANT as TEMPLATE_VARIANT

from multilingual_megatron.experiments.scaling.language_composition.study_contracts import (
    ARM_GROUPINGS,
    DESIGN_ID,
    EXPERTS,
    REPO_ROOT,
    SEEDS,
    SELECTION_MANIFEST,
)
from multilingual_megatron.experiments.scaling.language_composition.data.data_artifacts import (
    ARM_MANIFEST_RELATIVE,
    AUDIT_RELATIVE,
    COHORTS_RELATIVE,
    K196_VALIDATION_RELATIVE,
)
from multilingual_megatron.experiments.scaling.language_composition.training.training_actions import (
    custom_action_handler,
    storage_estimate,
    verify_authorization,
)
from multilingual_megatron.experiments.scaling.language_composition.training.training_contracts import (
    DEFAULT_OUTPUT,
    DEFAULT_RUNTIME,
    DEFAULT_SPEC,
    evaluation_spec_transform,
    profile_transform,
    reference_results,
    validate_runtime,
    validate_study_spec,
)
from multilingual_megatron.experiments.scaling.language_composition.training.training_jobs import build_jobs, validate_topology
from multilingual_megatron.experiments.scaling.language_composition.training.production_smoke_evidence import (
    EVIDENCE_GATE_JOB_ID,
    fixed_input_paths as production_smoke_fixed_input_paths,
)


WORKFLOW_MODULE = (
    "multilingual_megatron.experiments.scaling."
    "language_composition.training.training_workflow"
)


POLICY = WorkflowPolicy(
    default_spec=DEFAULT_SPEC,
    default_runtime=DEFAULT_RUNTIME,
    default_output=DEFAULT_OUTPUT,
    ladders=("B",),
    expected_micro_batch_size=EXPECTED_MICRO_BATCH_SIZE,
    required_sentinel_probes=frozenset(),
    fixed_lr_schedule_probe_cells=(),
    required_training_env_keys=frozenset(REQUIRED_TRAINING_ENV_KEYS),
    spec_validator=validate_study_spec,
    prescribed_lr_schedule_probe_cells=(),
)
VARIANT = WorkflowVariant(
    identifier=DESIGN_ID,
    worker_module=WORKFLOW_MODULE,
    policy=POLICY,
    profile_transform=profile_transform,
    evaluation_spec_transform=evaluation_spec_transform,
    production_training_seeds=SEEDS,
    production_topology_validator=validate_topology,
    custom_action_handler=custom_action_handler,
    sentinel_startup_recovery_actions=frozenset(
        {"language_composition_smoke_gate", EVIDENCE_GATE_JOB_ID}
    ),
    authorized_production=True,
    provenance_files=(
        *tuple(sorted(Path(__file__).resolve().parents[1].rglob("*.py"))),
    ),
)


def _validate_template_only_dag(jobs: Sequence[Mapping[str, Any]]) -> None:
    """The custom workflow replaces all production waves after templating."""

    _validate_dag(jobs)


TEMPLATE_BUILD_VARIANT = replace(
    TEMPLATE_VARIANT, production_topology_validator=_validate_template_only_dag
)


def _smoke_reference_evidence(runtime: Mapping[str, Any]) -> dict[str, Any]:
    source = runtime["language_composition"]["reference_sources"]
    root = Path(str(source["primary_seed_root"])).resolve()
    launch_path = root / "control/launch_manifest.json"
    launch = _load_object(launch_path)
    if (
        _file_sha256(launch_path) != source["primary_seed_launch_file_sha256"]
        or launch.get("manifest_sha256") != source["primary_seed_manifest_sha256"]
    ):
        raise WorkflowError("smoke reference launch identity changed")
    rows: dict[str, Any] = {}
    expected_rates = runtime["language_composition"]["smoke"][
        "reference_median_tokens_per_second"
    ]
    steady_window = int(
        runtime["language_composition"]["smoke"][
            "steady_state_window_observations"
        ]
    )
    maximum_cv_pct = float(
        runtime["language_composition"]["smoke"]["maximum_step_time_cv_pct"]
    )
    for experts in (1, 32):
        job_id = f"prod_trunk_b_e{experts}_k012"
        result_path = root / "jobs" / job_id / "result.json"
        receipt_path = root / "control/receipts" / f"{job_id}.json"
        result = _load_object(result_path)
        receipt = _load_object(receipt_path)
        receipt_candidate = dict(receipt)
        receipt_hash = receipt_candidate.pop("receipt_sha256", None)
        result_hash = _file_sha256(result_path)
        segments = result.get("training_health", {}).get("segments", [])
        selected = [
            row
            for row in segments
            if row.get("status") == "ok"
            and row.get("metrics", {}).get("global_batch_size") == 32
        ]
        if not selected:
            raise WorkflowError(f"smoke reference has no healthy GBS32 segment: {job_id}")
        observed = float(selected[0]["metrics"]["median_tokens_per_second"])
        observed_window = int(selected[0]["metrics"].get("steps_counted", 0))
        observed_cv_pct = float(
            selected[0]["metrics"].get("step_time_cv_pct", math.inf)
        )
        if (
            result.get("status") != "complete"
            or result.get("arguments", {}).get("K") != 12
            or result.get("arguments", {}).get("experts") != experts
            or receipt_hash != _payload_sha256(receipt_candidate)
            or receipt.get("status") != "complete"
            or receipt.get("job_id") != job_id
            or result_hash not in set(receipt.get("output_sha256", {}).values())
            or not math.isclose(
                observed,
                float(expected_rates[str(experts)]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or observed_window != steady_window
            or not math.isfinite(observed_cv_pct)
            or observed_cv_pct >= maximum_cv_pct
        ):
            raise WorkflowError(f"smoke reference evidence drifted: {job_id}")
        rows[str(experts)] = {
            "source_job_id": job_id,
            "median_tokens_per_second": observed,
            "steady_state_window_observations": observed_window,
            "step_time_cv_pct": observed_cv_pct,
            "result": {"path": str(result_path), "sha256": result_hash},
            "receipt": {
                "path": str(receipt_path),
                "sha256": _file_sha256(receipt_path),
                "payload_sha256": receipt_hash,
            },
        }
    return {
        "source_launch": {
            "path": str(launch_path),
            "sha256": _file_sha256(launch_path),
            "manifest_sha256": launch["manifest_sha256"],
        },
        "comparison": (
            "matched B/K12/seed1234 first healthy GBS32 trunk segment; "
            "late-window median after long real-data warm-up"
        ),
        "minimum_fraction": 0.85,
        "steady_state_window_observations": steady_window,
        "maximum_step_time_cv_pct": maximum_cv_pct,
        "experts": rows,
    }


def _k196_validation_stats(paths: Any) -> tuple[int, int, str]:
    path = paths.tokenized / K196_VALIDATION_RELATIVE
    payload = _load_object(path)
    descriptors = payload.get("splits", {}).get("validation", {}).get(
        "prefix_artifacts_by_language"
    )
    if not isinstance(descriptors, Mapping) or len(descriptors) != 196:
        raise WorkflowError("composition validation manifest is not K196")
    tokens = 0
    for language, rows in descriptors.items():
        if not isinstance(rows, list) or not rows:
            raise WorkflowError(f"K196 validation has no tokens for {language}")
        tokens += sum(int(row["tokens_including_eod"]) for row in rows)
    if tokens <= 0:
        raise WorkflowError("K196 validation token count is not positive")
    return tokens, len(descriptors), _file_sha256(path)


def _composition_gpu_estimate(
    jobs: Sequence[Mapping[str, Any]], runtime: Mapping[str, Any], paths: Any
) -> dict[str, Any]:
    estimate = _gpu_estimate(jobs, runtime, paths)
    tokens, language_count, validation_hash = _k196_validation_stats(paths)
    job_by_id = {str(row["job_id"]): row for row in jobs}
    forward_total = 0.0
    overhead_total = 0.0
    evaluation_count = 0
    budget = runtime["performance_budget"]
    for row in estimate["job_estimates"]:
        if row.get("action") != "per_language_eval":
            continue
        job = job_by_id[str(row["job_id"])]
        ladder = str(job["arguments"]["ladder"])
        rate = float(row["aggregate_tokens_per_second"])
        gpu_count = int(row["gpu_count"])
        forward = tokens / rate / 3600.0 * gpu_count
        overhead = (
            float(budget["evaluation_setup_seconds"][ladder])
            + language_count
            * float(budget["evaluation_residual_seconds_per_language"][ladder])
        ) / 3600.0 * gpu_count
        row.update(
            {
                "tokens": tokens,
                "language_count": language_count,
                "forward_gpu_hours": forward,
                "evaluation_overhead_gpu_hours": overhead,
                "predicted_gpu_hours": forward + overhead,
            }
        )
        forward_total += forward
        overhead_total += overhead
        evaluation_count += 1
    if evaluation_count != 162:
        raise WorkflowError("GPU estimate does not contain 162 K196 evaluations")
    components = dict(estimate["by_component_gpu_hours"])
    components["validation_forward"] = forward_total
    components["validation_overhead"] = overhead_total
    total = math.fsum(components.values())
    estimate.update(
        {
            "method": "exact_K196_validation_plus_batch_schedule_aware_H100_forecast",
            "base_estimated_gpu_hours": total,
            "worst_case_estimated_gpu_hours": total,
            "by_component_gpu_hours": dict(sorted(components.items())),
            "evaluation_jobs_counted": evaluation_count,
            "validation_exact_token_manifest_sha256": validation_hash,
            "validation_exact_tokens_by_K": {"K196": tokens},
            "validation_language_counts_by_K": {"K196": language_count},
            "validation_tokens": {
                "language_composition": evaluation_count * tokens,
                "lr_diagnostics": 0,
                "sentinel_integration": 0,
            },
            "budget_policy": "informational_no_hard_cap",
            "gpu_hour_limit": None,
            "go_no_go": "go",
        }
    )
    return estimate


def _fixed_inputs(
    *,
    spec_path: Path,
    runtime_path: Path,
    runtime: Mapping[str, Any],
    paths: Any,
    references: Sequence[Mapping[str, Any]],
    smoke_reference: Mapping[str, Any],
) -> dict[str, str]:
    with variant_scope(VARIANT):
        fixed = _fixed_hashes(spec_path, runtime_path)
    additions: set[Path] = {
        (REPO_ROOT / SELECTION_MANIFEST).resolve(),
        *((REPO_ROOT / path).resolve() for path in ARM_GROUPINGS.values()),
        *(paths.tokenized / path for path in ARM_MANIFEST_RELATIVE.values()),
        paths.tokenized / K196_VALIDATION_RELATIVE,
        paths.tokenized / COHORTS_RELATIVE,
        paths.tokenized / AUDIT_RELATIVE,
    }
    evidence = runtime["performance_budget"]["training_throughput_evidence"]
    additions.add(Path(str(evidence["GBS32"]["report"])).resolve())
    additions.add(Path(str(evidence["GBS128"]["report"])).resolve())
    additions.add(
        Path(str(evidence["GBS64"]["B/E32_report"])).resolve()
    )
    for reference in references:
        additions.update(Path(value).resolve() for value in reference["fixed_files"])
    additions.add(Path(str(smoke_reference["source_launch"]["path"])).resolve())
    for descriptor in smoke_reference["experts"].values():
        additions.add(Path(str(descriptor["result"]["path"])).resolve())
        additions.add(Path(str(descriptor["receipt"]["path"])).resolve())
    production_smoke_evidence = runtime["language_composition"].get(
        "production_smoke_evidence"
    )
    if isinstance(production_smoke_evidence, Mapping):
        additions.update(
            production_smoke_fixed_input_paths(production_smoke_evidence)
        )
    for path in additions:
        if path.is_symlink() or not path.is_file():
            raise WorkflowError(f"language-composition fixed input is missing: {path}")
        fixed[_rel_or_abs(path)] = _file_sha256(path)
    return dict(sorted(fixed.items()))


def _reuses_production_smokes(runtime: Mapping[str, Any]) -> bool:
    return isinstance(
        runtime["language_composition"].get("production_smoke_evidence"), Mapping
    )


def _role_counts(runtime: Mapping[str, Any]) -> dict[str, int]:
    reused = _reuses_production_smokes(runtime)
    counts = {
        "data_preflights": 3,
        "bounded_smokes": 0 if reused else 4,
        "smoke_and_authorization_gates": 2,
        "smoke_cleanup": 0 if reused else 1,
        "new_training_trunks": 18,
        "new_cooldown_branches": 108,
        "new_arm_evaluations": 108,
        "reference_reevaluations": 54,
        "trajectory_cleanups": 18,
        "summaries": 1,
    }
    if reused:
        counts["reused_bounded_smokes"] = 4
    return counts


def _authorization_declaration(runtime: Mapping[str, Any]) -> dict[str, Any]:
    reused = _reuses_production_smokes(runtime)
    return {
        "status": "not_authorized_at_plan_time",
        "authorization_file": runtime["language_composition"][
            "production_authorization_file"
        ],
        "gate_job": "language_composition_authorization_gate",
        "policy": (
            "immutable attempt-3 smokes are reused with one explicitly disclosed "
            "timing-CV boundary; no production job may run until the evidence gate "
            "and a hash-bound authorization both pass"
            if reused
            else (
                "four smokes are authorized; no production job may run until an "
                "explicit authorization binds this manifest, data seal, and passed "
                "smoke gate"
            )
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
    if any(
        value > int(runtime["max_submit_slots_per_invocation"])
        for value in phase_slots.values()
    ):
        raise WorkflowError("language-composition phase exceeds Slurm submit-slot limit")
    manifest["staged_submission"] = {
        "phase_order": list(SUBMISSION_PHASES),
        "phase_jobs": phase_jobs,
        "phase_submit_slots": phase_slots,
        "max_submit_slots_per_invocation": int(
            runtime["max_submit_slots_per_invocation"]
        ),
        "array_elements_count_individually": True,
        "prior_phase_completion_required": True,
    }
    paths = _paths_for_manifest(manifest)
    estimate = _composition_gpu_estimate(jobs, runtime, paths)
    manifest["gpu_hour_estimate"] = estimate
    walltime = _apply_conservative_job_time_limits(jobs, runtime, estimate)
    if walltime is None:
        manifest.pop("walltime_contract", None)
    else:
        manifest["walltime_contract"] = walltime
    manifest["storage_estimate"] = storage_estimate(runtime)
    _validate_dag(jobs)


def build_manifest(
    *, spec_path: Path, runtime_path: Path, output_root: Path
) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    runtime_path = runtime_path.resolve()
    output_root = output_root.resolve()
    validate_study_spec(repo_root=REPO_ROOT, spec_path=spec_path)
    runtime = validate_runtime(runtime_path, require_data=True)
    references = reference_results(runtime)
    smoke_reference = _smoke_reference_evidence(runtime)
    template = _api.build_manifest(
        spec_path=(REPO_ROOT / TEMPLATE_SPEC).resolve(),
        runtime_path=runtime_path,
        output_root=output_root,
        variant=TEMPLATE_BUILD_VARIANT,
    )
    jobs = build_jobs(
        template_jobs=template["jobs"],
        runtime=runtime,
        output_root=output_root,
        references=references,
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
            "role_counts": _role_counts(runtime),
            "language_composition_design": _raw_design(spec_path),
            "reference_reuse": references,
            "smoke_reference_evidence": smoke_reference,
            "production_authorization": _authorization_declaration(runtime),
            "claim_boundary": (
                "three independent seeds; nested horizons and language rows are descriptive, "
                "not significance replicates"
            ),
        }
    )
    cells = manifest.get("batch_ramp_micro_batch_and_gradient_accumulation", {}).get(
        "cells", {}
    )
    if isinstance(cells, Mapping):
        manifest["batch_ramp_micro_batch_and_gradient_accumulation"] = {
            **manifest["batch_ramp_micro_batch_and_gradient_accumulation"],
            "cells": {
                key: value
                for key, value in cells.items()
                if key in {"B/E1", "B/E16", "B/E32"}
            },
        }
    paths = _paths_for_manifest(manifest)
    manifest["fixed_input_sha256"] = _fixed_inputs(
        spec_path=spec_path,
        runtime_path=runtime_path,
        runtime=runtime,
        paths=paths,
        references=references,
        smoke_reference=smoke_reference,
    )
    _refresh_manifest(manifest, runtime)
    manifest["manifest_sha256"] = _payload_sha256(_manifest_without_hash(manifest))
    validate_manifest(manifest, verify_inputs=True)
    return manifest


def _raw_design(spec_path: Path) -> dict[str, Any]:
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    return dict(raw["language_composition_design"])


def validate_manifest(
    payload: dict[str, Any], *, verify_inputs: bool = True
) -> dict[str, Any]:
    result = _api.validate_manifest(
        payload, verify_inputs=verify_inputs, variant=VARIANT
    )
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise WorkflowError("language-composition manifest lacks jobs")
    validate_topology(jobs)
    runtime_path = Path(str(payload["runtime_path"])).resolve()
    runtime = validate_runtime(runtime_path, require_data=True)
    reused_smokes = _reuses_production_smokes(runtime)
    expected_roles = _role_counts(runtime)
    if payload.get("role_counts") != expected_roles:
        raise WorkflowError("language-composition role counts drifted")
    if payload.get("language_composition_design") != _raw_design(
        Path(str(payload["spec_path"])).resolve()
    ):
        raise WorkflowError("language-composition manifest design drifted")
    if payload.get("production_authorization") != _authorization_declaration(runtime):
        raise WorkflowError("language-composition authorization declaration drifted")
    expected_stage_counts = {
        "data": 2,
        "production": 145,
        "sentinel": 1 if reused_smokes else 6,
        "storage": 1,
        "summary": 1,
        "validation": 162,
    }
    if payload.get("stage_counts") != expected_stage_counts:
        raise WorkflowError("language-composition stage counts drifted")
    references = reference_results(runtime)
    if payload.get("reference_reuse") != references:
        raise WorkflowError("language-composition reference inventory drifted")
    if payload.get("smoke_reference_evidence") != _smoke_reference_evidence(runtime):
        raise WorkflowError("language-composition smoke reference drifted")
    estimate = payload.get("gpu_hour_estimate", {})
    if (
        estimate.get("go_no_go") != "go"
        or estimate.get("gpu_hour_limit") is not None
        or estimate.get("training_jobs_counted")
        != (126 if reused_smokes else 130)
        or estimate.get("evaluation_jobs_counted") != 162
        or estimate.get("validation_language_counts_by_K") != {"K196": 196}
    ):
        raise WorkflowError("language-composition GPU estimate drifted")
    if payload.get("storage_estimate") != storage_estimate(runtime):
        raise WorkflowError("language-composition storage estimate drifted")
    phase_slots = payload.get("staged_submission", {}).get("phase_submit_slots")
    if not isinstance(phase_slots, Mapping) or (
        int(phase_slots.get("sentinel", -1)) != (1 if reused_smokes else 6)
        or int(phase_slots.get("production", -1)) != 308
    ):
        raise WorkflowError("language-composition submission phases drifted")
    if verify_inputs:
        _verify_workflow_data(payload, _paths_for_manifest(payload))
    return {
        **result,
        "design_id": DESIGN_ID,
        "jobs": 312 if reused_smokes else 317,
        "authorized_smoke_gpu_jobs": 0 if reused_smokes else 4,
        "reused_smoke_gpu_jobs": 4 if reused_smokes else 0,
        "production_gpu_jobs": 288,
        "production_authorized": False,
        "complete_trajectories_after_run": 27,
        "K196_evaluations": 162,
    }


FRESH_SMOKE_JOB_IDS = (
    "storage_preflight",
    "data_reuse_gate",
    "data_megatron_reader_smoke",
    "smoke_compact_b_e1_s1234",
    "smoke_compact_b_e32_s1234",
    "smoke_diverse_b_e1_s1234",
    "smoke_diverse_b_e32_s1234",
    "language_composition_smoke_gate",
    "retention_prune_sentinels",
)


def _preproduction_job_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    job_ids = {str(row["job_id"]) for row in manifest.get("jobs", [])}
    if EVIDENCE_GATE_JOB_ID in job_ids:
        return (
            "storage_preflight",
            "data_reuse_gate",
            "data_megatron_reader_smoke",
            EVIDENCE_GATE_JOB_ID,
        )
    return FRESH_SMOKE_JOB_IDS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    plan.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    plan.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    plan.add_argument("--force-plan", action="store_true")
    for command in ("validate", "submit-smokes", "submit-production", "status"):
        child = sub.add_parser(command)
        child.add_argument("--manifest", type=Path, required=True)
        if command.startswith("submit-"):
            child.add_argument("--dry-run", action="store_true")
            child.add_argument("--retry-failed", action="store_true")
    execute = sub.add_parser("execute-job", help=argparse.SUPPRESS)
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--job-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            path = _api.plan_workflow(
                spec=_abs_repo(args.spec),
                runtime=_abs_repo(args.runtime),
                output_root=_abs_repo(args.output_root),
                force=args.force_plan,
                variant=VARIANT,
                build_manifest_fn=build_manifest,
            )
            manifest = _load_object(path)
            print(
                json.dumps(
                    {"manifest": str(path), **validate_manifest(manifest)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        manifest_path = _abs_repo(args.manifest)
        manifest = _load_object(manifest_path)
        if args.command == "validate":
            print(json.dumps(validate_manifest(manifest), indent=2, sort_keys=True))
            return 0
        if args.command == "submit-smokes":
            preproduction_job_ids = _preproduction_job_ids(manifest)
            state = _api.submit_workflow(
                manifest_path=manifest_path,
                retry_failed=args.retry_failed,
                dry_run=args.dry_run,
                through_stage=None,
                selected_job_ids=preproduction_job_ids,
                variant=VARIANT,
                validate_manifest_fn=validate_manifest,
            )
            print(json.dumps(state, indent=2, sort_keys=True))
            return 0
        if args.command == "submit-production":
            validate_manifest(manifest)
            verify_authorization(
                manifest,
                _paths_for_manifest(manifest),
                validate_runtime(Path(str(manifest["runtime_path"])), require_data=True),
            )
            preproduction_job_ids = set(_preproduction_job_ids(manifest))
            selected = [
                str(row["job_id"])
                for row in manifest["jobs"]
                if str(row["job_id"]) not in preproduction_job_ids
            ]
            state = _api.submit_workflow(
                manifest_path=manifest_path,
                retry_failed=args.retry_failed,
                dry_run=args.dry_run,
                through_stage="production",
                selected_job_ids=selected,
                variant=VARIANT,
                validate_manifest_fn=validate_manifest,
            )
            print(json.dumps(state, indent=2, sort_keys=True))
            return 0
        if args.command == "status":
            state = _api.reconcile_workflow(
                manifest_path,
                variant=VARIANT,
                validate_manifest_fn=validate_manifest,
            )
            print(json.dumps(state, indent=2, sort_keys=True))
            return 0
        if args.command == "execute-job":
            def validate_execution(payload: dict[str, Any]) -> dict[str, Any]:
                from multilingual_megatron.experiments.scaling.language_composition.training.scheduling import apply_scheduling_amendment

                result = validate_manifest(payload)
                apply_scheduling_amendment(payload, _paths_for_manifest(payload).control)
                return result

            _api.execute_job(
                manifest_path,
                args.job_id,
                variant=VARIANT,
                validate_manifest_fn=validate_execution,
            )
            return 0
    except (WorkflowError, OSError, ValueError, KeyError) as exc:
        print(f"[language-composition] {exc}", file=sys.stderr)
        return 2
    raise AssertionError(args.command)


def _abs_repo(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
