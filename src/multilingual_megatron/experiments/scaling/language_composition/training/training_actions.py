"""Custom gates and the preregistered language-composition summary."""

from __future__ import annotations

import math
import os
import statistics
from pathlib import Path
from typing import Any, Mapping

from multilingual_megatron.experiments.scaling.ab_v2.artifact_data import (
    _verify_workflow_data,
)
from multilingual_megatron.experiments.scaling.ab_v2.artifacts import (
    _verified_receipt_evidence,
)
from multilingual_megatron.experiments.scaling.ab_v2.common import (
    Paths,
    WorkflowError,
    _abs,
    _atomic_json,
    _file_sha256,
    _load_object,
    _payload_sha256,
)
from multilingual_megatron.experiments.scaling.ab_v2.estimates import (
    _model_parameter_upper_bound,
)
from multilingual_megatron.experiments.scaling.ab_v2.gate_evaluation import (
    _validate_full_evaluation_artifacts,
    _validation_report_from_eval,
)

from multilingual_megatron.experiments.scaling.language_composition.study_contracts import (
    DESIGN_ID,
    EXPERTS,
    HORIZONS,
    SEEDS,
    SMOKE_EXPERTS,
    SMOKE_GLOBAL_BATCH_SIZE,
    SMOKE_MAX_STEP_TIME_CV_PCT,
    SMOKE_STEADY_STATE_WINDOW_OBSERVATIONS,
    SMOKE_TOKENS,
)
from multilingual_megatron.experiments.scaling.language_composition.data.data_artifacts import ARM_MANIFEST_RELATIVE, COHORTS_RELATIVE
from multilingual_megatron.experiments.scaling.language_composition.training.production_smoke_evidence import (
    AUTHORIZATION_SCHEMA as DISCLOSED_BOUNDARY_AUTHORIZATION_SCHEMA,
    DECISION as DISCLOSED_BOUNDARY_DECISION,
    EVIDENCE_GATE_FILENAME,
    run_evidence_gate,
)


AUTHORIZATION_SCHEMA = "stage1_language_composition_authorization/v1"


def storage_estimate(runtime: Mapping[str, Any]) -> dict[str, Any]:
    storage = runtime["storage"]
    bytes_per_parameter = int(storage["checkpoint_bytes_per_parameter"])
    overhead = float(storage["checkpoint_serialization_overhead_factor"])
    per_checkpoint = {
        experts: int(
            math.ceil(
                _model_parameter_upper_bound("B", experts)
                * bytes_per_parameter
                * overhead
            )
        )
        for experts in EXPERTS
    }
    one_terminal_set = sum(per_checkpoint.values())
    # Each trajectory creates eight trunk segment checkpoints and six branch
    # endpoints before its cleanup.  The five prior waves retain one 16B
    # checkpoint per expert at the sixth-wave peak.
    current_wave_checkpoints_per_expert = 14
    current_wave_bytes = current_wave_checkpoints_per_expert * one_terminal_set
    prior_terminal_bytes = (len(SEEDS) * 2 - 1) * one_terminal_set
    checkpoint_peak = current_wave_bytes + prior_terminal_bytes
    non_checkpoint = int(storage["composition_non_checkpoint_reserve_bytes"])
    final_headroom = int(storage["composition_final_free_headroom_bytes"])
    required_inodes = int(storage["composition_required_available_inodes"])
    return {
        "schema_version": 1,
        "method": "one_arm_seed_wave_conservative_full_state_checkpoint_bound",
        "capacity_enforcement": "hard_gate_at_smoke_and_production_authorization",
        "checkpoint_bytes_per_parameter": bytes_per_parameter,
        "checkpoint_serialization_overhead_factor": overhead,
        "per_checkpoint_bytes": {
            f"B/E{experts}": per_checkpoint[experts] for experts in EXPERTS
        },
        "current_wave_checkpoint_count_per_expert": current_wave_checkpoints_per_expert,
        "maximum_current_wave_checkpoint_bytes": current_wave_bytes,
        "maximum_prior_terminal_checkpoint_bytes": prior_terminal_bytes,
        "checkpoint_peak_bytes": checkpoint_peak,
        "retained_terminal_checkpoint_bytes": len(SEEDS) * 2 * one_terminal_set,
        "non_checkpoint_peak_reserve_bytes": non_checkpoint,
        "required_final_free_headroom_bytes": final_headroom,
        "required_available_bytes_at_preflight": checkpoint_peak
        + non_checkpoint
        + final_headroom,
        "required_available_inodes_at_preflight": required_inodes,
        "retention_policy": (
            "one compact/diverse arm-seed wave at a time; E1/E16/E32 in parallel; "
            "prune trunks and D<16B branches after six K196 evaluations"
        ),
    }


def _run_smoke_gate(
    manifest: Mapping[str, Any], job: Mapping[str, Any], paths: Paths
) -> None:
    arguments = job["arguments"]
    fraction = float(arguments["minimum_reference_throughput_fraction"])
    baselines = arguments["reference_median_tokens_per_second"]
    steady_window = int(arguments["steady_state_window_observations"])
    maximum_cv_pct = float(arguments["maximum_step_time_cv_pct"])
    if (
        steady_window != SMOKE_STEADY_STATE_WINDOW_OBSERVATIONS
        or maximum_cv_pct != SMOKE_MAX_STEP_TIME_CV_PCT
    ):
        raise WorkflowError("language-composition steady-state contract drifted")
    evidence: dict[str, Any] = {}
    errors: list[str] = []
    for job_id in arguments["smoke_jobs"]:
        receipt = _verified_receipt_evidence(manifest, str(job_id))
        result_path = paths.jobs / str(job_id) / "result.json"
        result = _load_object(result_path)
        result_arguments = result.get("arguments")
        health = result.get("sentinel_health")
        metrics = health.get("metrics") if isinstance(health, Mapping) else None
        experts = (
            int(result_arguments.get("experts", 0))
            if isinstance(result_arguments, Mapping)
            else 0
        )
        baseline = float(baselines.get(str(experts), math.nan))
        try:
            median = (
                float(metrics.get("median_tokens_per_second", math.nan))
                if isinstance(metrics, Mapping)
                else math.nan
            )
            steps_counted = (
                int(metrics.get("steps_counted", 0))
                if isinstance(metrics, Mapping)
                else 0
            )
            step_cv_pct = (
                float(metrics.get("step_time_cv_pct", math.nan))
                if isinstance(metrics, Mapping)
                else math.nan
            )
        except (TypeError, ValueError):
            median = math.nan
            steps_counted = 0
            step_cv_pct = math.nan
        threshold = baseline * fraction
        arm = (
            str(result_arguments.get("composition_arm", ""))
            if isinstance(result_arguments, Mapping)
            else ""
        )
        expected_manifest = (
            paths.tokenized / ARM_MANIFEST_RELATIVE[arm]
            if arm in ARM_MANIFEST_RELATIVE
            else None
        )
        row_errors: list[str] = []
        if (
            result.get("status") != "complete"
            or not isinstance(result_arguments, Mapping)
            or result_arguments.get("composition_arm") not in {"compact", "diverse"}
            or experts not in set(SMOKE_EXPERTS)
            or int(result_arguments.get("tokens", 0)) != SMOKE_TOKENS
            or result_arguments.get("sentinel_kind")
            != "language_composition_long_real_data"
            or int(result_arguments.get("global_batch_size", 0))
            != SMOKE_GLOBAL_BATCH_SIZE
            or int(result.get("actual_end_tokens", 0)) < SMOKE_TOKENS
            or result.get("checkpoint_state", {}).get("valid_full_state") is not True
        ):
            row_errors.append("result/checkpoint contract")
        if (
            expected_manifest is None
            or not expected_manifest.is_file()
            or result.get("data_manifest") != str(expected_manifest.resolve())
            or result.get("data_manifest_sha256")
            != _file_sha256(expected_manifest.resolve())
        ):
            row_errors.append("composition data-manifest identity")
        if (
            not isinstance(health, Mapping)
            or health.get("status") != "ok"
            or health.get("errors") != []
            or not isinstance(metrics, Mapping)
            or metrics.get("finite_loss_and_gradients") is not True
            or int(metrics.get("lm_loss_lines", 0)) <= 0
            or int(metrics.get("grad_norm_lines", 0)) <= 0
            or int(metrics.get("max_skipped_iterations", -1)) != 0
            or int(metrics.get("max_nan_iterations", -1)) != 0
            or int(metrics.get("nonfinite_lm_loss_count", -1)) != 0
            or int(metrics.get("nonfinite_grad_norm_count", -1)) != 0
        ):
            row_errors.append("training health")
        if experts == 32 and isinstance(metrics, Mapping) and (
            float(metrics.get("max_dropped_assignment_rate", math.nan)) != 0.0
            or float(metrics.get("max_tokens_with_zero_kept_experts", math.nan)) != 0.0
        ):
            row_errors.append("MoE routing/drop health")
        if (
            steps_counted != steady_window
            or not math.isfinite(step_cv_pct)
            or step_cv_pct >= maximum_cv_pct
        ):
            row_errors.append("steady-state stability")
        if not math.isfinite(median) or median < threshold:
            row_errors.append("throughput floor")
        if row_errors:
            errors.append(f"{job_id}: {', '.join(row_errors)}")
        evidence[str(job_id)] = {
            "receipt": receipt,
            "result": {"path": str(result_path), "sha256": _file_sha256(result_path)},
            "composition_arm": result_arguments.get("composition_arm")
            if isinstance(result_arguments, Mapping)
            else None,
            "experts": experts,
            "median_tokens_per_second": median if math.isfinite(median) else None,
            "steady_state_window_observations": steps_counted,
            "required_steady_state_window_observations": steady_window,
            "step_time_cv_pct": step_cv_pct if math.isfinite(step_cv_pct) else None,
            "maximum_step_time_cv_pct": maximum_cv_pct,
            "reference_median_tokens_per_second": (
                baseline if math.isfinite(baseline) else None
            ),
            "minimum_fraction": fraction,
            "threshold_tokens_per_second": (
                threshold if math.isfinite(threshold) else None
            ),
            "throughput_fraction": median / baseline
            if math.isfinite(median) and baseline > 0
            else None,
            "errors": row_errors,
        }
    report: dict[str, Any] = {
        "schema": "stage1_language_composition_smoke_gate/v2",
        "status": "passed" if not errors else "failed",
        "design_id": DESIGN_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "contract": {
            "real_data": True,
            "arms": ["compact", "diverse"],
            "experts": list(SMOKE_EXPERTS),
            "seed": 1234,
            "tokens_per_job": SMOKE_TOKENS,
            "global_batch_size": SMOKE_GLOBAL_BATCH_SIZE,
            "steady_state_window_observations": steady_window,
            "maximum_step_time_cv_pct": maximum_cv_pct,
            "minimum_reference_throughput_fraction": fraction,
            "production_authorization_automatic": False,
        },
        "evidence": evidence,
        "errors": errors,
        "next_step": (
            "review smoke evidence and create an explicit hash-bound production authorization"
            if not errors
            else "hold production and diagnose the failed smoke contract"
        ),
    }
    report["payload_sha256"] = _payload_sha256(report)
    _atomic_json(paths.control / "language_composition_smoke_gate.json", report)
    if errors:
        raise WorkflowError("language-composition smoke gate failed: " + "; ".join(errors))


def verify_authorization(
    manifest: Mapping[str, Any], paths: Paths, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    authorization_path = _abs(
        str(runtime["language_composition"]["production_authorization_file"])
    )
    authorization = _load_object(authorization_path)
    candidate = dict(authorization)
    embedded = candidate.pop("authorization_sha256", None)
    smoke_path = paths.control / "language_composition_smoke_gate.json"
    sealed = manifest.get("data_materialization", {}).get("sealed_corpus")
    disclosed = runtime["language_composition"].get("production_smoke_evidence")
    if isinstance(disclosed, Mapping):
        evidence_path = paths.control / EVIDENCE_GATE_FILENAME
        evidence_gate = _load_object(evidence_path)
        expected_keys = {
            "schema",
            "decision",
            "design_id",
            "manifest_sha256",
            "data_seal_sha256",
            "smoke_evidence_gate_sha256",
            "smoke_evidence_gate_payload_sha256",
            "source_manifest_payload_sha256",
            "source_smoke_gate_sha256",
            "source_smoke_gate_payload_sha256",
            "waived_errors",
            "waiver_rationale",
            "authorized_scope",
            "authorized_by",
            "authorized_at_utc",
            "authorization_sha256",
        }
        if (
            embedded != _payload_sha256(candidate)
            or set(authorization) != expected_keys
            or authorization.get("schema")
            != DISCLOSED_BOUNDARY_AUTHORIZATION_SCHEMA
            or authorization.get("decision") != DISCLOSED_BOUNDARY_DECISION
            or authorization.get("design_id") != DESIGN_ID
            or authorization.get("manifest_sha256")
            != manifest.get("manifest_sha256")
            or not isinstance(sealed, Mapping)
            or authorization.get("data_seal_sha256")
            != sealed.get("data_seal_sha256")
            or authorization.get("smoke_evidence_gate_sha256")
            != _file_sha256(evidence_path)
            or authorization.get("smoke_evidence_gate_payload_sha256")
            != evidence_gate.get("payload_sha256")
            or evidence_gate.get("status") != DISCLOSED_BOUNDARY_DECISION
            or evidence_gate.get("decision") != DISCLOSED_BOUNDARY_DECISION
            or authorization.get("source_manifest_payload_sha256")
            != disclosed.get("source_manifest_payload_sha256")
            or authorization.get("source_smoke_gate_sha256")
            != disclosed.get("source_smoke_gate_file_sha256")
            or authorization.get("source_smoke_gate_payload_sha256")
            != disclosed.get("source_smoke_gate_payload_sha256")
            or authorization.get("waived_errors") != disclosed.get("waived_errors")
            or authorization.get("waiver_rationale") != disclosed.get("rationale")
            or authorization.get("authorized_scope")
            != "18 new trajectories and 162 K196 evaluations"
            or authorization.get("authorized_by") != disclosed.get("authorized_by")
            or not str(authorization.get("authorized_at_utc", "")).endswith("Z")
        ):
            raise WorkflowError(
                "production authorization is absent, invalid, or out of scope"
            )
        return {
            "path": str(authorization_path),
            "file_sha256": _file_sha256(authorization_path),
            "authorization_sha256": embedded,
            "decision": authorization["decision"],
            "waived_errors": list(authorization["waived_errors"]),
            "authorized_by": authorization["authorized_by"],
            "authorized_at_utc": authorization["authorized_at_utc"],
        }

    smoke = _load_object(smoke_path)
    if (
        embedded != _payload_sha256(candidate)
        or authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("decision") != "authorized"
        or authorization.get("design_id") != DESIGN_ID
        or authorization.get("manifest_sha256") != manifest.get("manifest_sha256")
        or not isinstance(sealed, Mapping)
        or authorization.get("data_seal_sha256") != sealed.get("data_seal_sha256")
        or authorization.get("smoke_gate_sha256") != _file_sha256(smoke_path)
        or authorization.get("smoke_gate_payload_sha256") != smoke.get("payload_sha256")
        or smoke.get("status") != "passed"
        or authorization.get("authorized_scope")
        != "18 new trajectories and 162 K196 evaluations"
        or not str(authorization.get("authorized_by", "")).strip()
        or not str(authorization.get("authorized_at_utc", "")).endswith("Z")
    ):
        raise WorkflowError("production authorization is absent, invalid, or out of scope")
    return {
        "path": str(authorization_path),
        "file_sha256": _file_sha256(authorization_path),
        "authorization_sha256": embedded,
        "authorized_by": authorization["authorized_by"],
        "authorized_at_utc": authorization["authorized_at_utc"],
    }


def _run_authorization_gate(
    manifest: Mapping[str, Any], paths: Paths, runtime: Mapping[str, Any]
) -> None:
    _verify_workflow_data(manifest, paths)
    authorization = verify_authorization(manifest, paths, runtime)
    estimate = manifest.get("storage_estimate")
    if not isinstance(estimate, Mapping):
        raise WorkflowError("language-composition manifest lacks a storage estimate")
    paths.root.mkdir(parents=True, exist_ok=True)
    stats = os.statvfs(paths.root)
    available_bytes = int(stats.f_bavail * stats.f_frsize)
    available_inodes = int(stats.f_favail)
    required_bytes = int(estimate["required_available_bytes_at_preflight"])
    required_inodes = int(estimate["required_available_inodes_at_preflight"])
    errors = []
    if available_bytes < required_bytes:
        errors.append(
            f"available bytes {available_bytes} are below required {required_bytes}"
        )
    if available_inodes < required_inodes:
        errors.append(
            f"available inodes {available_inodes} are below required {required_inodes}"
        )
    report: dict[str, Any] = {
        "schema": "stage1_language_composition_authorization_gate/v1",
        "status": "passed" if not errors else "failed",
        "design_id": DESIGN_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "authorization": authorization,
        "filesystem": {
            "path": str(paths.root.resolve()),
            "available_bytes": available_bytes,
            "required_available_bytes": required_bytes,
            "remaining_bytes_after_reservation": available_bytes - required_bytes,
            "available_inodes": available_inodes,
            "required_available_inodes": required_inodes,
            "remaining_inodes_after_reservation": available_inodes - required_inodes,
        },
        "storage_estimate_sha256": _payload_sha256(estimate),
        "errors": errors,
    }
    report["payload_sha256"] = _payload_sha256(report)
    target = paths.control / "language_composition_authorization_gate.json"
    _atomic_json(target, report)
    if errors:
        raise WorkflowError("language-composition authorization gate failed: " + "; ".join(errors))


def _metrics_for_evaluation(
    manifest: Mapping[str, Any], paths: Paths, job_id: str
) -> tuple[dict[str, Any], dict[str, float]]:
    receipt = _verified_receipt_evidence(manifest, job_id)
    path = paths.jobs / job_id / "evaluation_manifest.json"
    validation = _validate_full_evaluation_artifacts(path)
    payload, _report, _report_path = _validation_report_from_eval(path)
    values: dict[str, float] = {}
    for language, descriptor in payload["languages"].items():
        artifact = Path(str(descriptor["path"]))
        if not artifact.is_absolute():
            artifact = path.parent / artifact
        row = _load_object(artifact.resolve())
        values[str(language)] = float(row["atlas_tao_unigram_adjusted_loss"])
    return {"receipt": receipt, "validation": validation}, values


def _cohort_means(
    language_values: Mapping[str, float], cohorts: Mapping[str, list[str]]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name, languages in cohorts.items():
        missing = [language for language in languages if language not in language_values]
        if missing:
            raise WorkflowError(
                f"evaluation lacks cohort {name} languages: {missing[:4]}"
            )
        result[name] = (
            math.fsum(language_values[language] for language in languages)
            / len(languages)
            if languages
            else None
        )
    return result


def _run_summary(
    manifest: Mapping[str, Any], job: Mapping[str, Any], paths: Paths
) -> None:
    arguments = job["arguments"]
    evaluation_ids = [
        *arguments["new_evaluation_jobs"],
        *arguments["reference_evaluation_jobs"],
    ]
    if len(evaluation_ids) != 162 or len(set(evaluation_ids)) != 162:
        raise WorkflowError("language-composition summary requires 162 unique evaluations")
    cohorts_payload = _load_object(paths.tokenized / COHORTS_RELATIVE)
    cohorts = {
        name: list(raw["languages"])
        for name, raw in cohorts_payload["cohorts"].items()
    }
    rows: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {}
    keyed: dict[tuple[str, int, int, int], dict[str, float | None]] = {}
    jobs_by_id = {str(row["job_id"]): row for row in manifest["jobs"]}
    for job_id in evaluation_ids:
        row = jobs_by_id[str(job_id)]
        args = row["arguments"]
        arm = str(args["composition_arm"])
        seed = int(args["model_seed"])
        experts = int(args["experts"])
        horizon = int(args["D"])
        validation, language_values = _metrics_for_evaluation(
            manifest, paths, str(job_id)
        )
        cohort_values = _cohort_means(language_values, cohorts)
        key = (arm, seed, experts, horizon)
        if key in keyed:
            raise WorkflowError(f"duplicate language-composition evaluation cell: {key}")
        keyed[key] = cohort_values
        rows.append(
            {
                "job_id": job_id,
                "arm": arm,
                "seed": seed,
                "experts": experts,
                "horizon_tokens": horizon,
                "cohort_equal_language_atlas_tao_adjusted_loss": cohort_values,
            }
        )
        evidence[str(job_id)] = validation
    expected = {
        (arm, seed, experts, horizon)
        for arm in ("reference", "compact", "diverse")
        for seed in SEEDS
        for experts in EXPERTS
        for horizon in HORIZONS
    }
    if set(keyed) != expected:
        raise WorkflowError("language-composition evaluation factorial is incomplete")

    contrasts_by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        seed_rows: list[dict[str, float | int]] = []
        for seed in SEEDS:
            gains = {
                arm: float(keyed[(arm, seed, 1, horizon)]["unseen_177"])
                - float(keyed[(arm, seed, 32, horizon)]["unseen_177"])
                for arm in ("reference", "compact", "diverse")
            }
            seed_rows.append(
                {
                    "seed": seed,
                    "reference_moe_gain": gains["reference"],
                    "compact_moe_gain": gains["compact"],
                    "diverse_moe_gain": gains["diverse"],
                    "paired_diverse_minus_compact_gain": gains["diverse"]
                    - gains["compact"],
                }
            )
        paired = [
            float(row["paired_diverse_minus_compact_gain"]) for row in seed_rows
        ]
        contrasts_by_horizon[str(horizon)] = {
            "seedwise": seed_rows,
            "paired_contrast_summary": {
                "mean": statistics.mean(paired),
                "sample_standard_deviation": statistics.stdev(paired),
                "range": [min(paired), max(paired)],
                "n_independent_seeds": len(paired),
            },
        }
    report: dict[str, Any] = {
        "schema": "stage1_language_composition_summary/v1",
        "status": "complete",
        "design_id": DESIGN_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "metric": "equal-language ATLAS-Tao unigram-adjusted held-out loss",
        "primary": {
            "cohort": "unseen_177",
            "horizon_tokens": HORIZONS[-1],
            "contrast": "(E1-E32)_diverse - (E1-E32)_compact",
            **contrasts_by_horizon[str(HORIZONS[-1])],
        },
        "descriptive_all_horizons": contrasts_by_horizon,
        "cell_rows": sorted(
            rows,
            key=lambda row: (
                row["arm"],
                row["seed"],
                row["experts"],
                row["horizon_tokens"],
            ),
        ),
        "evaluation_evidence": evidence,
        "claim_boundary": (
            "three seeds are the replication units; language rows and nested readouts "
            "are not independent significance replicates; no hypothesis test is reported"
        ),
    }
    report["summary_sha256"] = _payload_sha256(report)
    _atomic_json(paths.control / "language_composition_summary.json", report)


def custom_action_handler(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    paths: Paths,
    runtime: Mapping[str, Any],
) -> bool:
    action = str(job["action"])
    if action == "language_composition_smoke_gate":
        _run_smoke_gate(manifest, job, paths)
    elif action == "language_composition_smoke_evidence_gate":
        run_evidence_gate(manifest, job, paths)
    elif action == "language_composition_authorization_gate":
        _run_authorization_gate(manifest, paths, runtime)
    elif action == "language_composition_summary":
        _run_summary(manifest, job, paths)
    else:
        return False
    return True
