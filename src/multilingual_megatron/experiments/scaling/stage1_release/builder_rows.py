"""Deterministic Stage 1 release row assembly and scientific-scope QA."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from multilingual_megatron.experiments.scaling.stage1_release.contracts import (
    EXPECTED_COUNTS,
    LADDER_COUNTS,
    ReleaseBuildSpec,
    Stage1ReleaseError,
    _json_cell,
    _load_json,
    _sha256,
)
from multilingual_megatron.experiments.scaling.stage1_release.builder_common import _utc_now


CORPUS_MANIFEST = Path(
    "outputs/pipeline_runs/h100_ab_v2_k12_fixed_lr_attempt3/"
    "corpus/release_32b/corpus_manifest.json"
)

CD_LEDGER = Path("result_analysis/stage1_cd_analysis/cross_root_result_ledger.json")

ANALYSIS_REGISTRY_PATHS = (
    Path("result_analysis/stage1_cd_20260723/freeze_receipt.json"),
    Path("result_analysis/stage1_cd_20260723/simulation_receipt.json"),
    Path("result_analysis/stage1_cd_20260723/protocol_amendment.json"),
    Path("result_analysis/stage1_cd_20260723/frozen_launched_d_study_spec.json"),
    Path(
        "result_analysis/stage1_cd_20260723/"
        "analysis_first_pause_cancellation_manifest_20260730.json"
    ),
    Path(
        "result_analysis/stage1_cd_20260723/"
        "analysis_first_pause_cancellation_receipt_20260730.json"
    ),
    Path("result_analysis/stage1_cd_analysis/analysis_summary.json"),
    Path("result_analysis/stage1_cd_analysis/cross_root_result_ledger.json"),
    Path("result_analysis/stage1_cd_result_analysis.ipynb"),
    Path("result_analysis/fit_input_validation_loss_joined.csv"),
)


def _training_configuration_rows(
    training_jobs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in training_jobs:
        path = Path(str(job["training_manifest_path"])).resolve()
        payload = _load_json(path)
        metadata = payload.get("metadata", {})
        run_cfg = payload.get("run_cfg", {})
        train = run_cfg.get("train", {}) if isinstance(run_cfg, Mapping) else {}
        policy = train.get("batch_policy", {}) if isinstance(train, Mapping) else {}
        rows.append(
            {
                "training_job_id": job["training_job_id"],
                "trajectory_id": job["trajectory_id"],
                "ladder": job["ladder"],
                "experts": job["experts"],
                "language_count": job["language_count"],
                "job_role": job["job_role"],
                "recipe": payload.get("recipe") or run_cfg.get("recipe"),
                "backend": payload.get("backend") or run_cfg.get("backend"),
                "sequence_length": metadata.get("seq_length"),
                "micro_batch_size": metadata.get("micro_batch_size"),
                "global_batch_size": metadata.get("global_batch_size"),
                "train_iterations": metadata.get("train_iters"),
                "evaluation_interval": metadata.get("eval_interval"),
                "evaluation_iterations": metadata.get("eval_iters"),
                "target_token_budget": metadata.get("target_token_budget_tokens"),
                "planning_method": metadata.get("planning_method"),
                "schedule_mode": policy.get("mode"),
                "warmup_tokens": policy.get("warmup_tokens"),
                "decay_fraction": policy.get("decay_fraction"),
                "data_parallel_size": policy.get("data_parallel_size"),
                "data_manifest_path": run_cfg.get("data_manifest"),
                "data_manifest_sha256": run_cfg.get("data_manifest_sha256"),
                "model_name_or_path": payload.get("model_name_or_path"),
                "tokenized_path": payload.get("tokenized_path"),
                "training_manifest_path": str(path),
                "training_manifest_sha256": _sha256(path),
                "resolved_overrides_json": _json_cell(run_cfg.get("overrides", [])),
                "batch_policy_json": _json_cell(policy),
                "run_configuration_json": _json_cell(run_cfg),
                "training_metadata_json": _json_cell(metadata),
                "dataset_configuration_json": _json_cell(payload.get("dataset", {})),
                "tokenization_configuration_json": _json_cell(
                    payload.get("tokenization", {})
                ),
                "artifact_configuration_json": _json_cell(payload.get("artifacts", {})),
                "field_provenance_json": _json_cell(
                    {
                        "training_manifest_path": "observed",
                        "training_manifest_sha256": "derived",
                        "recipe": "configured",
                        "backend": "configured",
                        "sequence_length": "configured",
                        "micro_batch_size": "configured",
                        "global_batch_size": "configured",
                        "train_iterations": "configured",
                        "evaluation_interval": "configured",
                        "evaluation_iterations": "configured",
                        "target_token_budget": "configured",
                        "planning_method": "configured",
                        "schedule_mode": "configured",
                        "warmup_tokens": "configured",
                        "decay_fraction": "configured",
                        "data_parallel_size": "configured",
                    }
                ),
                "provenance_class": "configured",
            }
        )
    return rows


def _dataset_rows(
    corpus_manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corpus = _load_json(corpus_manifest_path)
    manifest_sha = _sha256(corpus_manifest_path)
    release_rows = [
        {
            "dataset_release_id": corpus.get("release_id"),
            "schema_version": corpus.get("schema_version"),
            "release_status": corpus.get("release_status"),
            "ready": corpus.get("ready"),
            "language_count": len(corpus.get("languages", {})),
            "manifest_path": str(corpus_manifest_path.resolve()),
            "manifest_file_sha256": manifest_sha,
            "embedded_manifest_sha256": corpus.get("manifest_sha256"),
            "selection_contract_json": _json_cell(corpus.get("selection_contract", {})),
            "inputs_json": _json_cell(corpus.get("inputs", {})),
            "validation_prefix_inputs_json": _json_cell(
                corpus.get("validation_prefix_inputs", {})
            ),
            "blocking_languages_json": _json_cell(corpus.get("blocking_languages", [])),
            "provenance_class": "observed",
        }
    ]
    language_rows: list[dict[str, Any]] = []
    for language, record in sorted(corpus.get("languages", {}).items()):
        train = record.get("train", {})
        validation = record.get("validation", {})
        train_output = train.get("output", {})
        validation_output = validation.get("output", {})
        coverage = record.get("source_coverage", {})
        language_rows.append(
            {
                "dataset_release_id": corpus.get("release_id"),
                "language": language,
                "exact_unique_language_tokens_including_eod": record.get(
                    "exact_unique_language_tokens_including_eod"
                ),
                "unique_language_tokens_lower_bound_including_eod": record.get(
                    "observed_unique_language_tokens_lower_bound_including_eod"
                ),
                "train_status": train.get("status"),
                "train_target_tokens": train.get("target_tokens"),
                "train_selected_tokens": train.get("selected_tokens"),
                "train_shortfall_tokens": train.get("shortfall_tokens"),
                "train_emitted_documents": train.get("emitted_documents"),
                "train_path": train_output.get("path"),
                "train_sha256": train_output.get("sha256"),
                "validation_status": validation.get("status"),
                "validation_target_tokens": validation.get("target_tokens"),
                "validation_selected_tokens": validation.get("selected_tokens"),
                "validation_shortfall_tokens": validation.get("shortfall_tokens"),
                "validation_emitted_documents": validation.get("emitted_documents"),
                "validation_path": validation_output.get("path"),
                "validation_sha256": validation_output.get("sha256"),
                "complete_source_scan": coverage.get("complete_source_scan"),
                "acquisition_status": coverage.get("acquisition_status"),
                "source_coverage_report_path": coverage.get("report_path"),
                "source_coverage_report_payload_sha256": coverage.get(
                    "report_payload_sha256"
                ),
                "language_manifest_record_json": _json_cell(record),
                "corpus_manifest_path": str(corpus_manifest_path.resolve()),
                "corpus_manifest_sha256": manifest_sha,
                "provenance_class": "observed",
            }
        )
    if len(language_rows) != 200:
        raise Stage1ReleaseError(
            f"Expected 200 corpus language records, got {len(language_rows)}"
        )
    return release_rows, language_rows


def _analysis_registry(
    root: Path,
    paths: Sequence[Path] = ANALYSIS_REGISTRY_PATHS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    role_by_name = {
        "freeze_receipt.json": "predeclared_freeze_receipt",
        "simulation_receipt.json": "prelaunch_simulation_receipt",
        "protocol_amendment.json": "protocol_amendment",
        "frozen_launched_d_study_spec.json": "frozen_d_protocol",
        "analysis_first_pause_cancellation_manifest_20260730.json": (
            "queued_job_cancellation_manifest"
        ),
        "analysis_first_pause_cancellation_receipt_20260730.json": (
            "queued_job_cancellation_receipt"
        ),
        "analysis_summary.json": "prior_cd_descriptive_analysis",
        "cross_root_result_ledger.json": "canonical_cd_result_ledger",
        "stage1_cd_result_analysis.ipynb": "prior_cd_analysis_notebook",
        "fit_input_validation_loss_joined.csv": "prior_fit_input_export",
        "analysis_manifest.json": "learning_rate_analysis_manifest",
        "comparison_summary.json": "learning_rate_cross_cluster_summary",
        "STATISTICAL_ANALYSIS.md": "learning_rate_statistical_analysis",
    }
    for relative in paths:
        path = (root / relative).resolve()
        rows.append(
            {
                "analysis_artifact_id": relative.name,
                "artifact_role": role_by_name.get(relative.name, "analysis"),
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": _sha256(path) if path.is_file() else None,
                "status": "observed" if path.is_file() else "missing",
                "scope_disclosure": (
                    "Registered historical analysis/protocol artifact; its "
                    "conclusions are not automatically promoted to the new "
                    "A-D release."
                ),
                "sequence_order": None,
                "depends_on": None,
                "provenance_class": "observed",
            }
        )
    return rows


def _jointmoe_analysis_registry(
    root: Path,
    build_spec: ReleaseBuildSpec,
) -> list[dict[str, Any]]:
    """Register LR/seed evidence without mixing it into the canonical fit panel."""

    rows = _analysis_registry(root, build_spec.analysis_registry_paths)
    missing = [row["path"] for row in rows if not row["exists"]]
    if missing:
        raise Stage1ReleaseError(
            "JointMoE analysis artifacts are missing: " + ", ".join(missing)
        )
    manifest_rows = [
        row for row in rows if row["artifact_role"] == "learning_rate_analysis_manifest"
    ]
    comparison_rows = [
        row
        for row in rows
        if row["artifact_role"] == "learning_rate_cross_cluster_summary"
    ]
    if len(manifest_rows) != 1 or len(comparison_rows) != 1:
        raise Stage1ReleaseError(
            "JointMoE release requires exactly one LR analysis manifest and summary"
        )
    lr_manifest_path = Path(str(manifest_rows[0]["path"]))
    lr_manifest = _load_json(lr_manifest_path)
    if (
        lr_manifest.get("schema") != "stage1_lr_confirmation_analysis_manifest/v1"
        or lr_manifest.get("status") != "complete"
        or not isinstance(lr_manifest.get("artifacts"), Mapping)
    ):
        raise Stage1ReleaseError(f"LR analysis manifest is incomplete: {lr_manifest_path}")
    for name, expected_sha256 in lr_manifest["artifacts"].items():
        artifact_path = lr_manifest_path.parent / str(name)
        if not artifact_path.is_file() or _sha256(artifact_path) != str(expected_sha256):
            raise Stage1ReleaseError(f"LR analysis artifact hash drift: {artifact_path}")
    generator_path = root / str(lr_manifest.get("generator", ""))
    if (
        not generator_path.is_file()
        or _sha256(generator_path) != str(lr_manifest.get("generator_sha256", ""))
    ):
        raise Stage1ReleaseError(f"LR analysis generator hash drift: {generator_path}")
    comparison_path = Path(str(comparison_rows[0]["path"]))
    comparison = _load_json(comparison_path)
    conclusion = comparison.get("scientific_conclusion", {})
    if (
        comparison.get("schema")
        != "stage1_lr_confirmation_cross_cluster_analysis/v1"
        or comparison.get("status") != "complete"
        or not isinstance(conclusion, Mapping)
        or conclusion.get("aggregate_leader_h100") != "x100"
        or conclusion.get("aggregate_leader_a100") != "x100"
        or conclusion.get("h100_preregistered_decision")
        != "inconclusive_boundary"
    ):
        raise Stage1ReleaseError(
            f"LR comparison conclusion or scope changed: {comparison_path}"
        )
    if build_spec.seed_confirmation_summary is not None:
        path = (root / build_spec.seed_confirmation_summary).resolve()
        summary = _load_json(path)
        results = summary.get("results")
        seeds = sorted(
            {
                int(row.get("model_seed", -1))
                for row in results
                if isinstance(row, Mapping)
            }
        ) if isinstance(results, list) else []
        dataset_seeds = sorted(
            {
                int(row.get("dataset_random_seed", -1))
                for row in results
                if isinstance(row, Mapping)
            }
        ) if isinstance(results, list) else []
        seed_cells = {
            seed: {
                (int(row["experts"]), int(row["K"]))
                for row in results
                if isinstance(row, Mapping) and int(row.get("model_seed", -1)) == seed
            }
            for seed in seeds
        }
        if (
            summary.get("status") != "complete"
            or summary.get("errors") != []
            or not isinstance(results, list)
            or int(summary.get("observed_results", -1))
            != int(summary.get("expected_results", -2))
            or len(results) != 108
            or seeds != [5678, 91011]
            or dataset_seeds != seeds
            or any(
                cells
                != {
                    (experts, language_count)
                    for experts in (1, 16, 32)
                    for language_count in (1, 12, 200)
                }
                for cells in seed_cells.values()
            )
        ):
            raise Stage1ReleaseError(
                f"Seed-confirmation sidecar is incomplete or changed: {path}"
            )
        rows.append(
            {
                "analysis_artifact_id": "stage1_b_seed_confirmation_v1",
                "artifact_role": "separate_seed_robustness_sidecar",
                "path": str(path),
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "status": "observed_complete",
                "scope_disclosure": (
                    "Two additional seeds for nine B-ladder cells. These rows are "
                    "not part of the 64-trajectory primary scaling-law fit."
                ),
                "sequence_order": None,
                "depends_on": None,
                "provenance_class": "observed",
            }
        )
    rows.append(
        {
            "analysis_artifact_id": "stage1_scaling_fit_jointmoe_x1_v1",
            "artifact_role": "structured_holdout_fit",
            "path": str(
                (root / "result_analysis/stage1_scaling_fit_jointmoe_x1_v1").resolve()
            ),
            "exists": False,
            "size_bytes": None,
            "sha256": None,
            "status": "pending_release_bound_fit",
            "scope_disclosure": (
                "Primary model selection must use the three frozen independent "
                "trajectory holdouts before the selected all-data refit."
            ),
            "sequence_order": 1,
            "depends_on": None,
            "provenance_class": "planned",
        }
    )
    return rows


def _flatten_slurm_report(
    report: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attempt_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    allocation_seen: set[tuple[str, str]] = set()
    for attempt in report.get("attempts", []):
        row = {
            key: value
            for key, value in attempt.items()
            if key not in {"parent_record", "array_task_records", "evidence"}
        }
        row["evidence_json"] = _json_cell(attempt.get("evidence", []))
        row["parent_record_json"] = _json_cell(attempt.get("parent_record"))
        row["array_task_records_json"] = _json_cell(
            attempt.get("array_task_records", [])
        )
        attempt_rows.append(row)
        allocation_source = attempt.get("array_task_records") or (
            [attempt.get("parent_record")]
            if attempt.get("parent_record") is not None
            else []
        )
        for allocation in allocation_source:
            record_id = str(allocation["slurm_record_id"])
            key = (str(attempt["attempt_uid"]), record_id)
            if key in allocation_seen:
                continue
            allocation_seen.add(key)
            allocation_row = dict(allocation)
            raw = allocation_row.pop("raw_sacct_record", {})
            provenance = allocation_row.pop("field_provenance", {})
            allocation_rows.append(
                {
                    "allocation_uid": (f"{attempt['attempt_uid']}:{record_id}"),
                    "attempt_uid": attempt["attempt_uid"],
                    "logical_job_id": attempt["logical_job_id"],
                    "run_root": attempt["run_root"],
                    **allocation_row,
                    "raw_sacct_record_json": _json_cell(raw),
                    "field_provenance_json": _json_cell(provenance),
                    "provenance_class": "observed",
                }
            )
    return attempt_rows, allocation_rows


def _compute_summary_rows(
    *,
    training_jobs: Sequence[Mapping[str, Any]],
    training_summary: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    allocations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summary_by_id = {str(row["training_job_id"]): row for row in training_summary}
    attempts_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in attempts:
        attempts_by_key[(str(row["run_root"]), str(row["logical_job_id"]))].append(row)
    allocations_by_attempt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in allocations:
        allocations_by_attempt[str(row["attempt_uid"])].append(row)

    rows: list[dict[str, Any]] = []
    for job in training_jobs:
        key = (str(Path(str(job["run_root"])).resolve()), str(job["training_job_id"]))
        job_attempts = attempts_by_key.get(key, [])
        job_allocations = [
            allocation
            for attempt in job_attempts
            for allocation in allocations_by_attempt.get(
                str(attempt["attempt_uid"]), []
            )
        ]
        states = sorted(
            {
                str(allocation.get("state", ""))
                for allocation in job_allocations
                if allocation.get("state")
            }
        )
        gpu_types: Counter[str] = Counter()
        for allocation in job_allocations:
            raw_types = allocation.get("allocated_gpu_type_counts")
            if isinstance(raw_types, str):
                try:
                    raw_types = json.loads(raw_types)
                except json.JSONDecodeError:
                    raw_types = {}
            if isinstance(raw_types, Mapping):
                gpu_types.update(
                    {str(name): int(count) for name, count in raw_types.items()}
                )
        accounted = [
            row for row in job_attempts if row.get("accounting_status") == "observed"
        ]
        metric = summary_by_id.get(str(job["training_job_id"]), {})
        rows.append(
            {
                "training_job_id": job["training_job_id"],
                "trajectory_id": job["trajectory_id"],
                "ladder": job["ladder"],
                "experts": job["experts"],
                "language_count": job["language_count"],
                "job_role": job["job_role"],
                "parent_training_job_id": job["parent_training_job_id"],
                "attempt_count": len(job_attempts),
                "accounted_attempt_count": len(accounted),
                "missing_accounting_attempt_count": (
                    len(job_attempts) - len(accounted)
                ),
                "slurm_states_json": _json_cell(states),
                "allocated_gpu_type_counts_across_attempts_json": _json_cell(
                    dict(sorted(gpu_types.items()))
                ),
                "allocated_gpu_hours_including_retries": sum(
                    float(row.get("allocated_gpu_hours") or 0.0)
                    for row in job_allocations
                )
                if job_allocations
                else None,
                "allocated_cpu_hours_including_retries": sum(
                    float(row.get("allocated_cpus") or 0.0)
                    * float(row.get("elapsed_seconds") or 0.0)
                    / 3600.0
                    for row in job_allocations
                )
                if job_allocations
                else None,
                "first_submit_time_utc": min(
                    (
                        str(row["submit_time_utc"])
                        for row in job_allocations
                        if row.get("submit_time_utc")
                    ),
                    default=None,
                ),
                "first_start_time_utc": min(
                    (
                        str(row["start_time_utc"])
                        for row in job_allocations
                        if row.get("start_time_utc")
                    ),
                    default=None,
                ),
                "last_end_time_utc": max(
                    (
                        str(row["end_time_utc"])
                        for row in job_allocations
                        if row.get("end_time_utc")
                    ),
                    default=None,
                ),
                "total_scheduler_queue_wait_hours": sum(
                    float(row.get("scheduler_queue_wait_seconds") or 0.0) / 3600.0
                    for row in job_allocations
                )
                if job_allocations
                else None,
                "result_end_to_end_hours": float(job["end_to_end_seconds"]) / 3600.0,
                "mean_tokens_per_second": metric.get("mean_tokens_per_second"),
                "mean_model_tflops_per_gpu": metric.get("mean_model_tflops_per_gpu"),
                "mean_mfu_pct_of_configured_peak": metric.get(
                    "mean_mfu_pct_of_configured_peak"
                ),
                "mean_gpu_utilization_pct": metric.get("mean_gpu_utilization_pct"),
                "board_energy_estimate_kwh": metric.get("board_energy_estimate_kwh"),
                "board_energy_coverage_fraction": metric.get(
                    "board_energy_coverage_fraction"
                ),
                "configured_resource_json": _json_cell(job.get("resource")),
                "planned_wall_hours": job.get("predicted_wall_hours"),
                "compute_accounting_scope": (
                    "Allocated Slurm GPU-hours sum all observed attempts, "
                    "including retries. Board energy integrates observed GPU "
                    "board telemetry only and excludes facility overhead."
                ),
                "field_provenance_json": _json_cell(
                    {
                        "attempt_count": "derived",
                        "allocated_gpu_hours_including_retries": "derived",
                        "configured_resource_json": "configured",
                        "planned_wall_hours": "planned",
                        "result_end_to_end_hours": "derived",
                        "mean_tokens_per_second": "derived",
                        "board_energy_estimate_kwh": "derived",
                    }
                ),
                "provenance_class": "derived",
            }
        )
    return rows


def _runtime_parameter_rows(
    training_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    rows: list[dict[str, Any]] = []
    for row in training_metrics:
        if row.get("event_type") != "parameter_count":
            continue
        key = (
            row.get("training_job_id"),
            row.get("segment_id"),
            row.get("parameter_scope"),
            row.get("parameter_count"),
            row.get("tensor_model_parallel_rank"),
            row.get("pipeline_model_parallel_rank"),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                field: row.get(field)
                for field in (
                    "training_job_id",
                    "trajectory_id",
                    "ladder",
                    "experts",
                    "language_count",
                    "segment_id",
                    "segment_kind",
                    "parameter_scope",
                    "parameter_count",
                    "parameter_count_billions_reported",
                    "tensor_model_parallel_rank",
                    "pipeline_model_parallel_rank",
                    "source_path",
                    "source_line_number",
                    "provenance_class",
                )
            }
        )
    return rows


def _validity_and_qa_rows(
    *,
    validation: Any,
    initial_validity: Sequence[Mapping[str, Any]],
    slurm_report: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validity = [dict(row) for row in initial_validity]
    candidates_by_readout: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in validation.validation_candidates:
        candidates_by_readout[str(row["readout_id"])].append(row)
    for readout_id, candidates in sorted(candidates_by_readout.items()):
        if len(candidates) <= 1:
            continue
        validity.append(
            {
                "object_type": "validation_readout",
                "object_id": readout_id,
                "issue_type": "duplicate_evaluation_candidates",
                "severity": "information",
                "verdict": "valid_semantically_equivalent",
                "evidence_path": _json_cell(
                    [row["candidate_manifest_path"] for row in candidates]
                ),
                "details": (
                    f"{len(candidates)} valid candidates share the selected "
                    "checkpoint and sufficient-statistics identity."
                ),
                "provenance_class": "derived",
            }
        )
    qa: list[dict[str, Any]] = []

    def add(
        check_id: str,
        *,
        observed: Any,
        expected: Any,
        status: str,
        severity: str = "error",
        details: str = "",
    ) -> None:
        qa.append(
            {
                "check_id": check_id,
                "observed": observed,
                "expected": expected,
                "status": status,
                "severity": severity,
                "details": details,
                "checked_at_utc": _utc_now(),
                "provenance_class": "derived",
            }
        )

    readout_counts = Counter(str(row["ladder"]) for row in tables["readouts"])
    add(
        "canonical_readout_ladder_counts",
        observed=_json_cell(dict(sorted(readout_counts.items()))),
        expected=_json_cell(LADDER_COUNTS),
        status="pass" if dict(readout_counts) == LADDER_COUNTS else "fail",
    )
    exact_count_checks = {
        "independent_trajectory_count": (
            len(tables["trajectories"]),
            EXPECTED_COUNTS["trajectories"],
        ),
        "readout_count": (
            len(tables["readouts"]),
            EXPECTED_COUNTS["readouts"],
        ),
        "validation_language_count": (
            len(tables["validation_language"]),
            EXPECTED_COUNTS["validation_language"],
        ),
        "trained_language_exposure_count": (
            len(tables["data_exposure"]),
            EXPECTED_COUNTS["trained_language"],
        ),
        "production_training_job_count": (
            len(tables["training_jobs"]),
            464,
        ),
        "training_summary_job_count": (
            len(tables["training_summary"]),
            464,
        ),
        "corpus_language_count": (
            len(tables["corpus_languages"]),
            200,
        ),
    }
    for check_id, (observed, expected) in exact_count_checks.items():
        add(
            check_id,
            observed=observed,
            expected=expected,
            status="pass" if observed == expected else "fail",
        )
    role_counts = Counter(
        str(row["metric_role"]) for row in tables["validation_language"]
    )
    expected_roles = {
        "trained": EXPECTED_COUNTS["trained_language"],
        "zero_shot": EXPECTED_COUNTS["zero_shot_language"],
    }
    add(
        "validation_metric_role_counts",
        observed=_json_cell(dict(sorted(role_counts.items()))),
        expected=_json_cell(expected_roles),
        status="pass" if dict(role_counts) == expected_roles else "fail",
    )
    selected_candidates = sum(
        bool(row.get("selected")) for row in tables["validation_candidates"]
    )
    add(
        "one_selected_candidate_per_readout",
        observed=selected_candidates,
        expected=EXPECTED_COUNTS["readouts"],
        status=(
            "pass" if selected_candidates == EXPECTED_COUNTS["readouts"] else "fail"
        ),
    )
    training_ids = {str(row["training_job_id"]) for row in tables["training_jobs"]}
    summary_ids = {str(row["training_job_id"]) for row in tables["training_summary"]}
    add(
        "training_summary_key_coverage",
        observed=len(training_ids & summary_ids),
        expected=len(training_ids),
        status="pass" if training_ids == summary_ids else "fail",
    )
    missing_accounting = int(slurm_report.get("counts", {}).get("missing_attempts", 0))
    add(
        "slurm_accounting_completeness",
        observed=missing_accounting,
        expected=0,
        status="pass" if missing_accounting == 0 else "warning",
        severity="warning",
        details=(
            "Historical gaps are explicitly retained; a warning does not "
            "silently impute compute."
        ),
    )
    add(
        "independent_seed_replication",
        observed=1,
        expected="disclosed limitation",
        status="disclosed_limitation",
        severity="warning",
        details=(
            "Each trajectory has one configured model/data seed. Checkpoint "
            "and language observations are repeated measures, not replicates."
        ),
    )
    if any(row["status"] == "fail" for row in qa):
        failed = [row["check_id"] for row in qa if row["status"] == "fail"]
        raise Stage1ReleaseError("Release QA failed: " + ", ".join(failed))
    return validity, qa


def _validation_scope_rows(
    readouts: Sequence[Mapping[str, Any]],
    validation_language: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build and validate the explicit ladder-by-tier evaluation coverage matrix."""

    readout_by_id = {str(row["readout_id"]): row for row in readouts}
    observations: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in validation_language:
        observations[str(row["readout_id"])].append(row)
    if set(observations) != set(readout_by_id):
        raise Stage1ReleaseError(
            "Validation-scope matrix does not cover exactly the canonical readouts"
        )
    cells: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for readout_id, readout in sorted(readout_by_id.items()):
        rows = observations[readout_id]
        roles = Counter(str(row["metric_role"]) for row in rows)
        language_count = int(readout["language_count"])
        expected_trained = language_count
        expected_zero_shot = 199 if language_count == 1 else 0
        if roles != Counter(
            {"trained": expected_trained, "zero_shot": expected_zero_shot}
        ):
            raise Stage1ReleaseError(
                f"Unexpected validation scope for {readout_id}: {dict(roles)}"
            )
        cells[(str(readout["ladder"]), language_count)].append(
            {
                "readout_id": readout_id,
                "trained": roles["trained"],
                "zero_shot": roles["zero_shot"],
                "languages": {str(row["language"]) for row in rows},
            }
        )
    result: list[dict[str, Any]] = []
    for (ladder, language_count), members in sorted(cells.items()):
        trained_counts = {int(row["trained"]) for row in members}
        zero_shot_counts = {int(row["zero_shot"]) for row in members}
        if len(trained_counts) != 1 or len(zero_shot_counts) != 1:
            raise Stage1ReleaseError(
                f"Inconsistent validation scope in {ladder}/K{language_count}"
            )
        trained = next(iter(trained_counts))
        zero_shot = next(iter(zero_shot_counts))
        languages = set().union(*(row["languages"] for row in members))
        result.append(
            {
                "validation_scope_id": f"{ladder}:K{language_count}",
                "ladder": ladder,
                "language_count": language_count,
                "readout_count": len(members),
                "trained_languages_per_readout": trained,
                "zero_shot_languages_per_readout": zero_shot,
                "validation_languages_per_readout": trained + zero_shot,
                "trained_language_observations": trained * len(members),
                "zero_shot_language_observations": zero_shot * len(members),
                "unique_validation_languages": len(languages),
                "scope_status": (
                    "all_200_languages_trained_plus_zero_shot"
                    if language_count == 1
                    else "trained_tier_only_no_zero_shot"
                ),
                "provenance_class": "derived",
            }
        )
    if len(result) != 16:
        raise Stage1ReleaseError(
            f"Expected 16 A-D/K validation-scope cells, got {len(result)}"
        )
    return result


def _release_summary_rows(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in ("A", "B", "C", "D", "ALL"):
        selected = (
            list(tables["readouts"])
            if scope == "ALL"
            else [row for row in tables["readouts"] if row["ladder"] == scope]
        )
        trajectory_ids = {str(row["trajectory_id"]) for row in selected}
        jobs = [
            row
            for row in tables["training_jobs"]
            if scope == "ALL" or row["ladder"] == scope
        ]
        compute = [
            row
            for row in tables["compute_summary"]
            if scope == "ALL" or row["ladder"] == scope
        ]
        language_rows = [
            row
            for row in tables["validation_language"]
            if scope == "ALL" or row["ladder"] == scope
        ]
        gpu_hours_values = [
            float(row["allocated_gpu_hours_including_retries"])
            for row in compute
            if row.get("allocated_gpu_hours_including_retries") is not None
        ]
        energy_values = [
            float(row["board_energy_estimate_kwh"])
            for row in compute
            if row.get("board_energy_estimate_kwh") is not None
        ]
        rows.append(
            {
                "scope": scope,
                "independent_trajectories": len(trajectory_ids),
                "checkpoint_readouts": len(selected),
                "production_training_jobs": len(jobs),
                "validation_language_observations": len(language_rows),
                "trained_language_observations": sum(
                    row.get("metric_role") == "trained" for row in language_rows
                ),
                "zero_shot_language_observations": sum(
                    row.get("metric_role") == "zero_shot" for row in language_rows
                ),
                "allocated_gpu_hours_including_retries": (
                    sum(gpu_hours_values) if gpu_hours_values else None
                ),
                "training_jobs_with_gpu_hour_accounting": len(gpu_hours_values),
                "gpu_board_energy_estimate_kwh": (
                    sum(energy_values) if energy_values else None
                ),
                "training_jobs_with_board_energy_estimate": len(energy_values),
                "statistical_unit_disclosure": (
                    "Trajectory is the independent experimental unit; "
                    "checkpoint and language rows are repeated measures."
                ),
                "provenance_class": "derived",
            }
        )
    return rows


def _jointmoe_learning_rate_qa(
    *,
    studies: Sequence[Mapping[str, Any]],
    model_configs: Sequence[Mapping[str, Any]],
    training_jobs: Sequence[Mapping[str, Any]],
    training_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    modes = {str(row.get("learning_rate_mode")) for row in studies}
    prescriptions = {
        str(row.get("learning_rate_prescription_json")) for row in studies
    }
    if modes != {"prescribed_per_model_configuration"} or len(prescriptions) != 1:
        raise Stage1ReleaseError(
            "JointMoE release requires one per-model learning-rate prescription"
        )
    prescription = json.loads(next(iter(prescriptions)))
    expected_prescription = {
        "intercept": 8.39,
        "active_parameter_coefficient": -0.81,
        "expert_coefficient": -0.25,
        "constant_across_K": True,
        "parameter_count_basis": (
            "paper-style active non-embedding parameter count N_act_nonembed"
        ),
    }
    for key, expected in expected_prescription.items():
        if prescription.get(key) != expected:
            raise Stage1ReleaseError(
                f"JointMoE learning-rate prescription mismatch for {key}: "
                f"{prescription.get(key)!r} != {expected!r}"
            )

    model_by_cell: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in model_configs:
        key = (str(row["ladder"]), int(row["experts"]))
        if key in model_by_cell:
            raise Stage1ReleaseError(f"Duplicate model configuration for {key}")
        model_by_cell[key] = row
    expected_cells = {(ladder, experts) for ladder in "ABCD" for experts in (1, 4, 16, 32)}
    if set(model_by_cell) != expected_cells:
        raise Stage1ReleaseError(
            "JointMoE learning-rate audit expected the complete 16 model cells"
        )

    job_lrs: dict[tuple[str, int], set[float]] = defaultdict(set)
    job_counts: Counter[tuple[str, int]] = Counter()
    job_contracts: dict[str, tuple[str, float]] = {}
    for row in training_jobs:
        key = (str(row["ladder"]), int(row["experts"]))
        job_id = str(row["training_job_id"])
        value = row.get("peak_learning_rate")
        if key not in expected_cells or value is None:
            raise Stage1ReleaseError(
                f"Missing JointMoE peak learning rate for training job {job_id}"
            )
        learning_rate = float(value)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise Stage1ReleaseError(
                f"Invalid JointMoE peak learning rate for {job_id}"
            )
        if job_id in job_contracts:
            raise Stage1ReleaseError(f"Duplicate JointMoE training job: {job_id}")
        job_contracts[job_id] = (str(row["trajectory_id"]), learning_rate)
        job_lrs[key].add(learning_rate)
        job_counts[key] += 1
    if set(job_lrs) != expected_cells:
        raise Stage1ReleaseError("Training jobs do not cover all JointMoE model cells")

    observed: dict[str, dict[str, Any]] = {}
    for ladder, experts in sorted(expected_cells):
        key = (ladder, experts)
        if len(job_lrs[key]) != 1:
            raise Stage1ReleaseError(
                f"Peak learning rate is not constant across K/jobs for {ladder}:E{experts}"
            )
        nonembedding = int(
            model_by_cell[key]["declared_active_nonembedding_parameters"]
        )
        expected_lr = math.exp(
            float(prescription["intercept"])
            + float(prescription["active_parameter_coefficient"])
            * math.log(nonembedding)
            + float(prescription["expert_coefficient"]) * math.log(experts)
        )
        configured_lr = next(iter(job_lrs[key]))
        if not math.isclose(configured_lr, expected_lr, rel_tol=1e-12, abs_tol=0.0):
            raise Stage1ReleaseError(
                f"Configured peak learning rate does not match JointMoE x1 for "
                f"{ladder}:E{experts}: {configured_lr} != {expected_lr}"
            )
        observed[f"{ladder}:E{experts}"] = {
            "declared_active_nonembedding_parameters": nonembedding,
            "experts": experts,
            "configured_peak_learning_rate": configured_lr,
            "jointmoe_x1_peak_learning_rate": expected_lr,
            "training_job_count": job_counts[key],
        }

    runtime_max_by_job: dict[str, float] = {}
    runtime_rows = 0
    for row in training_metrics:
        value = row.get("learning_rate")
        if value is None:
            continue
        job_id = str(row["training_job_id"])
        if job_id not in job_contracts:
            raise Stage1ReleaseError(
                f"Runtime learning-rate row references unknown job: {job_id}"
            )
        runtime_lr = float(value)
        if not math.isfinite(runtime_lr) or runtime_lr < 0.0:
            raise Stage1ReleaseError(
                f"Invalid runtime learning-rate observation for {job_id}: {runtime_lr}"
            )
        runtime_rows += 1
        runtime_max_by_job[job_id] = max(
            runtime_lr,
            runtime_max_by_job.get(job_id, -math.inf),
        )
    missing_runtime_jobs = sorted(set(job_contracts) - set(runtime_max_by_job))
    if missing_runtime_jobs:
        raise Stage1ReleaseError(
            "Missing runtime learning-rate observations for jobs: "
            + ", ".join(missing_runtime_jobs[:10])
        )

    job_peak_ratios: list[float] = []
    trajectory_runtime_max: dict[str, float] = {}
    trajectory_contract: dict[str, float] = {}
    for job_id, (trajectory_id, configured_lr) in job_contracts.items():
        runtime_max = runtime_max_by_job[job_id]
        ratio = runtime_max / configured_lr
        if ratio > 1.0 + 1e-6:
            raise Stage1ReleaseError(
                f"Runtime learning rate exceeds configured peak for {job_id}: "
                f"{runtime_max} > {configured_lr}"
            )
        job_peak_ratios.append(ratio)
        trajectory_runtime_max[trajectory_id] = max(
            runtime_max,
            trajectory_runtime_max.get(trajectory_id, -math.inf),
        )
        prior = trajectory_contract.setdefault(trajectory_id, configured_lr)
        if prior != configured_lr:
            raise Stage1ReleaseError(
                f"Trajectory has inconsistent configured peak LR: {trajectory_id}"
            )
    trajectory_peak_ratios = [
        trajectory_runtime_max[trajectory_id] / configured_lr
        for trajectory_id, configured_lr in trajectory_contract.items()
    ]
    if any(not math.isclose(ratio, 1.0, rel_tol=1e-5, abs_tol=0.0) for ratio in trajectory_peak_ratios):
        raise Stage1ReleaseError(
            "At least one trajectory did not reach its configured JointMoE peak LR"
        )
    return {
        "check_id": "jointmoe_x1_learning_rate_contract",
        "observed": observed,
        "expected": {
            "model_cells": 16,
            "formula": prescription["formula"],
            "constant_across_K": True,
            "multiplier": 1.0,
        },
        "status": "pass",
        "severity": "error",
        "details": (
            "All production trunk/branch jobs use the exact JointMoE x1 peak "
            "learning rate for their ladder/E cell, constant across K. Runtime "
            f"logs cover all {len(job_contracts)} jobs ({runtime_rows} LR rows); "
            "every trajectory reaches its configured peak within logging precision "
            f"(trajectory ratio range {min(trajectory_peak_ratios):.9f}-"
            f"{max(trajectory_peak_ratios):.9f}; job max ratio range "
            f"{min(job_peak_ratios):.9f}-{max(job_peak_ratios):.9f})."
        ),
        "checked_at_utc": _utc_now(),
        "provenance_class": "derived",
    }


def _jointmoe_scope_disclosures() -> list[dict[str, str]]:
    return [
        {
            "disclosure_id": "learning_rate_evidence_scope",
            "category": "hyperparameter selection",
            "severity": "limitation",
            "title": "Learning-rate confirmation is B-ladder evidence",
            "statement": (
                "JointMoE x1 is the equal-cell aggregate leader on the H100 "
                "and A100 B-ladder confirmations. The original H100 "
                "preregistered gate remained inconclusive because x4 won the "
                "E1/K1 boundary cell."
            ),
            "analysis_consequence": (
                "Treat x1 as the study-wide prescription selected before this "
                "clean rerun, not as a per-cell, seed-general, compute-optimal, "
                "or universally optimal learning rate."
            ),
        },
        {
            "disclosure_id": "seed_confirmation_scope",
            "category": "statistical scope",
            "severity": "limitation",
            "title": "Additional seeds cover nine B-ladder cells",
            "statement": (
                "The B-ladder E={1,16,32} by K={1,12,200} robustness grid has "
                "two additional independent training seeds (108 readouts). The "
                "remaining primary A-D trajectories have one configured seed."
            ),
            "analysis_consequence": (
                "Report the seed sidecar as bounded robustness evidence and do "
                "not pool its readouts into the 64-trajectory scaling-law fit or "
                "claim full-grid seed-general inference."
            ),
        },
    ]
