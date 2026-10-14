"""Unchanged orchestration for building and sealing a Stage 1 release."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import platform
import sys
from typing import Any

from multilingual_megatron.experiments.scaling.stage1_release.execution_specs import build_execution_spec_rows
from multilingual_megatron.experiments.scaling.stage1_release.contracts import (
    PROVENANCE_CLASSES,
    SCHEMA,
    ReleaseBuildSpec,
    Stage1ReleaseError,
    _canonical_bytes,
    _json_cell,
    _model_and_trajectory_rows,
    _primary_roots,
    _production_jobs,
    _sha256,
    _source_registry,
    _write_json,
    historical_release_build_spec,
)
from multilingual_megatron.experiments.scaling.stage1_release.builder_common import _assign_qa_issue_ids
from multilingual_megatron.experiments.scaling.stage1_release.builder_common import _canonical_qa_evidence_cell
from multilingual_megatron.experiments.scaling.stage1_release.builder_common import _run_text
from multilingual_megatron.experiments.scaling.stage1_release.builder_common import _safe_version
from multilingual_megatron.experiments.scaling.stage1_release.builder_common import _utc_now
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _assert_catalog_contracts
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _fallback_catalog
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _fallback_checklist
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _fallback_readme
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _file_inventory
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _presentation_helpers
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _snapshot_generation_code
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _write_fallback_excel
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _write_parquet_tables
from multilingual_megatron.experiments.scaling.stage1_release.builder_rows import _analysis_registry
from multilingual_megatron.experiments.scaling.stage1_release.builder_rows import _compute_summary_rows
from multilingual_megatron.experiments.scaling.stage1_release.builder_rows import _dataset_rows
from multilingual_megatron.experiments.scaling.stage1_release.builder_rows import _flatten_slurm_report
from multilingual_megatron.experiments.scaling.stage1_release.builder_rows import _jointmoe_analysis_registry
from multilingual_megatron.experiments.scaling.stage1_release.builder_rows import _jointmoe_learning_rate_qa
from multilingual_megatron.experiments.scaling.stage1_release.builder_rows import _jointmoe_scope_disclosures
from multilingual_megatron.experiments.scaling.stage1_release.builder_rows import _release_summary_rows
from multilingual_megatron.experiments.scaling.stage1_release.builder_rows import _runtime_parameter_rows
from multilingual_megatron.experiments.scaling.stage1_release.builder_rows import _training_configuration_rows
from multilingual_megatron.experiments.scaling.stage1_release.builder_rows import _validation_scope_rows
from multilingual_megatron.experiments.scaling.stage1_release.builder_rows import _validity_and_qa_rows
from multilingual_megatron.experiments.scaling.stage1_release.compute import build_physical_compute_category_summary
from multilingual_megatron.experiments.scaling.stage1_release.compute import classify_compute_records
from multilingual_megatron.experiments.scaling.stage1_release.data import build_data_exposure_rows
from multilingual_megatron.experiments.scaling.stage1_release.data import build_routing_rows
from multilingual_megatron.experiments.scaling.stage1_release.data import build_training_tables
from multilingual_megatron.experiments.scaling.stage1_release.provenance import append_analysis_protocol_states
from multilingual_megatron.experiments.scaling.stage1_release.provenance import build_launch_provenance
from multilingual_megatron.experiments.scaling.stage1_release.relations import validate_release_relations
from multilingual_megatron.experiments.scaling.stage1_release.training_enrichment import build_readout_compute_attribution
from multilingual_megatron.experiments.scaling.stage1_release.training_enrichment import enrich_model_configs
from multilingual_megatron.experiments.scaling.stage1_release.training_enrichment import enrich_training_summaries
from multilingual_megatron.experiments.scaling.stage1_release.training_enrichment import validate_production_graph
from multilingual_megatron.experiments.scaling.stage1_release.validation_qa import build_language_dimensions
from multilingual_megatron.experiments.scaling.stage1_release.validation_qa import build_validation_cohorts
from multilingual_megatron.experiments.scaling.stage1_release.validation_qa import reconstruct_and_verify_aggregates
from multilingual_megatron.experiments.scaling.stage1_release.validation_qa import validate_duplicate_candidate_routing
from multilingual_megatron.experiments.scaling.stage1_release.validation_qa import validate_routing_rows
from multilingual_megatron.experiments.scaling.stage1_release.validation_qa import validate_validation_identities
from multilingual_megatron.experiments.scaling.stage1_release.slurm import collect_slurm_accounting
from multilingual_megatron.experiments.scaling.stage1_release.validation import build_direct_cd_ledger
from multilingual_megatron.experiments.scaling.stage1_release.validation import build_stage1_ad_validation


def build_release(
    *,
    repo_root: Path,
    output_dir: Path,
    slurm_missing_policy: str = "flag",
    build_spec: ReleaseBuildSpec | None = None,
    build_spec_path: Path | None = None,
) -> dict[str, Any]:
    """Build and seal the complete analytical release."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    definition = historical_release_build_spec() if build_spec is None else build_spec
    release_name = definition.release_name
    if output_dir.exists():
        raise Stage1ReleaseError(
            f"Refusing to overwrite existing release: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = output_dir.with_name(f".{output_dir.name}.building-{os.getpid()}")
    if build_dir.exists():
        raise Stage1ReleaseError(f"Stale build directory exists: {build_dir}")
    build_dir.mkdir(parents=True)
    receipts = build_dir / "receipts"
    receipts.mkdir(parents=True)

    built_at = _utc_now()
    primary_roots = _primary_roots(repo_root, definition.primary_root_names)
    if definition.cd_source == "direct_final_summaries":
        ledger_path = receipts / "direct_cd_final_summary_ledger.json"
        _write_json(
            ledger_path,
            build_direct_cd_ledger(
                cd_result_roots={
                    "C": primary_roots["C"],
                    "D": primary_roots["D"],
                }
            ),
        )
    else:
        if definition.cd_ledger is None:
            raise Stage1ReleaseError("cross-root source mode lacks a C/D ledger")
        ledger_path = (repo_root / definition.cd_ledger).resolve()
    corpus_manifest_path = (repo_root / definition.corpus_manifest).resolve()
    validation = build_stage1_ad_validation(
        ab_result_root=primary_roots["AB"],
        cd_ledger_path=ledger_path,
    )
    source_roots, run_roots = _source_registry(
        root=repo_root,
        primary_roots=primary_roots,
        ledger_path=ledger_path,
    )
    studies, ladder_specs = build_execution_spec_rows(
        repo_root=repo_root,
        primary_roots=primary_roots,
    )
    launch_provenance, execution_inputs, launch_qa = build_launch_provenance(
        repo_root=repo_root,
        primary_roots=primary_roots,
    )
    model_configs, trajectories = _model_and_trajectory_rows(
        readouts=validation.readouts,
        ladder_specs=ladder_specs,
    )
    training_jobs, initial_validity = _production_jobs(primary_roots)
    for row in training_jobs:
        row["field_provenance_json"] = _json_cell(
            {
                "status": "observed",
                "actual_end_tokens": "observed",
                "end_to_end_seconds": "observed",
                "training_manifest_path": "observed",
                "resource": "configured",
                "target_tokens": "configured",
                "peak_learning_rate": "configured",
                "predicted_wall_hours": "planned",
                "planned_gpu_tokens": "planned",
            }
        )
    production_graph = validate_production_graph(
        training_jobs,
        validation.readouts,
    )
    training_configs = _training_configuration_rows(training_jobs)
    training = build_training_tables(training_jobs)
    training_summary, hardware_contract = enrich_training_summaries(
        training_jobs,
        training["training_summary"],
        repo_root,
    )
    runtime_parameters = _runtime_parameter_rows(training["training_metrics"])
    model_configs = enrich_model_configs(model_configs, runtime_parameters)

    data_exposure = build_data_exposure_rows(
        validation=validation,
        corpus_manifest_path=corpus_manifest_path,
    )
    for row in data_exposure:
        exact = row.get("exact_unique_language_tokens_including_eod")
        row["unique_exposure_denominator_status"] = (
            "exact" if exact not in (None, 0) else "censored_lower_bound_not_used"
        )
        row["effective_exposure_passes_status"] = (
            "exact" if exact not in (None, 0) else "not_computable_from_lower_bound"
        )
    routing_layers = build_routing_rows(validation.validation_language)
    dataset_release, corpus_languages = _dataset_rows(corpus_manifest_path)
    language_dimensions = build_language_dimensions(repo_root, corpus_languages)
    validation_cohorts = list(
        build_validation_cohorts(validation.validation_aggregates)
    )
    aggregate_qa = reconstruct_and_verify_aggregates(
        validation.validation_language,
        validation.validation_aggregates,
    )
    identity_qa = validate_validation_identities(validation.validation_language)
    routing_qa = validate_routing_rows(routing_layers, model_configs)
    duplicate_routing_qa = validate_duplicate_candidate_routing(
        validation.validation_candidates
    )

    slurm_report = collect_slurm_accounting(
        run_roots,
        missing_policy=slurm_missing_policy,
    )
    _write_json(receipts / "slurm_accounting.json", slurm_report)
    _write_json(receipts / "production_graph_validation.json", production_graph)
    _write_json(
        receipts / "validation_qa_summaries.json",
        {
            "aggregate_reconstruction": aggregate_qa.summary,
            "validation_identity": identity_qa.summary,
            "routing_conservation": routing_qa.summary,
            "duplicate_candidate_routing": duplicate_routing_qa.summary,
        },
    )
    job_attempts, slurm_allocations = _flatten_slurm_report(slurm_report)
    job_attempts, slurm_allocations = classify_compute_records(
        attempts=job_attempts,
        allocations=slurm_allocations,
        validation_candidates=validation.validation_candidates,
    )
    physical_compute_categories = build_physical_compute_category_summary(
        attempts=job_attempts,
        allocations=slurm_allocations,
    )
    compute_summary = _compute_summary_rows(
        training_jobs=training_jobs,
        training_summary=training_summary,
        attempts=job_attempts,
        allocations=slurm_allocations,
    )
    compute_attribution = build_readout_compute_attribution(
        validation.readouts,
        training_jobs,
        training["training_segments"],
        training["training_metrics"],
        compute_summary,
    )
    tables: dict[str, list[dict[str, Any]]] = {
        "studies": studies,
        "source_roots": source_roots,
        "launch_provenance": launch_provenance,
        "execution_inputs": execution_inputs,
        "model_configs": model_configs,
        "trajectories": trajectories,
        "readouts": list(validation.readouts),
        "validation_aggregates": list(validation.validation_aggregates),
        "validation_language": list(validation.validation_language),
        "validation_scope": _validation_scope_rows(
            validation.readouts,
            validation.validation_language,
        ),
        "validation_cohorts": validation_cohorts,
        "validation_candidates": list(validation.validation_candidates),
        "validation_artifacts": list(validation.artifacts),
        "validation_provenance": list(validation.provenance),
        "training_jobs": training_jobs,
        "training_configs": training_configs,
        "training_segments": training["training_segments"],
        "training_metrics": training["training_metrics"],
        "runtime_parameters": runtime_parameters,
        "training_summary": training_summary,
        "gpu_samples": training["gpu_samples"],
        "training_artifacts": training["training_artifacts"],
        "production_artifact_verifications": [
            {**row, "provenance_class": "derived"}
            for row in production_graph["artifact_verifications"]
        ],
        "readout_training_bindings": [
            {**row, "provenance_class": "derived"}
            for row in production_graph["readout_bindings"]
        ],
        "hardware_contract": hardware_contract,
        "dataset_release": dataset_release,
        "corpus_languages": corpus_languages,
        "languages": list(language_dimensions.languages),
        "language_tier_membership": list(language_dimensions.memberships),
        "language_metadata_sources": list(language_dimensions.sources),
        "data_exposure": data_exposure,
        "routing_layers": routing_layers,
        "job_attempts": job_attempts,
        "slurm_allocations": slurm_allocations,
        "physical_compute_categories": physical_compute_categories,
        "compute_summary": compute_summary,
        **compute_attribution,
        "analysis_registry": (
            _jointmoe_analysis_registry(repo_root, definition)
            if definition.cd_source == "direct_final_summaries"
            else append_analysis_protocol_states(
                _analysis_registry(repo_root, definition.analysis_registry_paths)
            )
        ),
    }
    validity_events, qa_issues = _validity_and_qa_rows(
        validation=validation,
        initial_validity=initial_validity,
        slurm_report=slurm_report,
        tables=tables,
    )
    qa_issues.extend(launch_qa)
    qa_issues.extend(production_graph["checks"])
    qa_issues.extend(language_dimensions.qa)
    qa_issues.extend(aggregate_qa.checks)
    qa_issues.extend(identity_qa.checks)
    qa_issues.extend(routing_qa.checks)
    qa_issues.extend(duplicate_routing_qa.checks)
    if definition.cd_source == "direct_final_summaries":
        qa_issues.append(
            _jointmoe_learning_rate_qa(
                studies=studies,
                model_configs=model_configs,
                training_jobs=training_jobs,
                training_metrics=training["training_metrics"],
            )
        )
    tables["validity_events"] = validity_events
    qa_issues.extend(
        validate_release_relations(
            tables,
            expected_validation_candidate_count=(
                definition.expected_validation_candidate_count
            ),
            expected_duplicate_candidate_count=(
                definition.expected_duplicate_candidate_count
            ),
        )
    )
    for qa_row in qa_issues:
        for evidence_field in ("observed", "expected"):
            if evidence_field in qa_row:
                qa_row[evidence_field] = _canonical_qa_evidence_cell(
                    qa_row[evidence_field]
                )
    qa_issues = _assign_qa_issue_ids(qa_issues)
    tables["qa_issues"] = qa_issues
    tables["release_summary"] = _release_summary_rows(tables)

    dataframes, table_metadata = _write_parquet_tables(
        tables=tables,
        directory=build_dir / "tables",
    )
    release_state = (
        "analysis_ready"
        if slurm_report["counts"]["missing_attempts"] == 0
        else "analysis_ready_with_explicit_slurm_accounting_gaps"
    )
    jointmoe_release = definition.cd_source == "direct_final_summaries"
    analysis_status = (
        "jointmoe_x1_training_and_validation_complete_structured_fit_pending"
        if jointmoe_release
        else "pending_frozen_cd_m1_score_then_locked_ad_refit"
    )
    analysis_boundary = (
        "The 64-trajectory JointMoE-x1 A-D panel is complete and sealed. "
        "Primary model selection on the three frozen independent trajectory "
        "holdouts and the subsequent selected-law all-data refit remain pending."
        if jointmoe_release
        else (
            "The normalized release is ready for descriptive and protocol-"
            "respecting analysis. Frozen C/D M1 scoring and the subsequent "
            "declared A-D refit remain pending and must occur in that order."
        )
    )
    analysis_next_action = (
        "Run and audit M0, M1, and M2 on the three frozen independent "
        "trajectory holdouts; apply the paired one-standard-error selection "
        "rule; then refit only the selected law on all 64 trajectories."
        if jointmoe_release
        else (
            "Use the release descriptively until the frozen C/D M1 score is "
            "sealed, then perform the declared A-D refit."
        )
    )
    release_metadata = {
        "schema": SCHEMA,
        "release_name": release_name,
        "status": release_state,
        "built_at_utc": built_at,
        "source_snapshot_at_utc": slurm_report["collection"]["collected_at_utc"],
        "repo_root": str(repo_root),
        "path_portability": {
            "canonical_release_paths": "relative_to_release_root",
            "absolute_source_paths": "build_host_provenance_only",
            "portable_source_mapping": "tables/source_roots.parquet",
        },
        "independent_trajectories": len(trajectories),
        "checkpoint_readouts": len(validation.readouts),
        "validation_language_observations": len(validation.validation_language),
        "production_training_jobs": len(training_jobs),
        "missing_slurm_attempts": slurm_report["counts"]["missing_attempts"],
        "slurm_accounting_status": slurm_report["status"],
        "provenance_classes": list(PROVENANCE_CLASSES),
        "canonical_format": "Apache Parquet with Zstandard compression",
        "statistical_unit": "trajectory_id",
        "configured_seed_replicates_per_trajectory": 1,
        "scientific_scope": definition.scientific_scope,
        "confirmatory_analysis_status": analysis_status,
        "confirmatory_analysis_complete": False,
        "analysis_boundary": analysis_boundary,
        "analysis_next_action": analysis_next_action,
    }
    table_counts = {name: len(frame) for name, frame in dataframes.items()}
    missing_checkpoints = [
        row
        for row in production_graph["readout_bindings"]
        if row.get("checkpoint_iteration_exists_now") is False
    ]
    missing_checkpoints_by_ladder = dict(
        sorted(Counter(str(row["ladder"]) for row in missing_checkpoints).items())
    )
    disclosures = [
        {
            "disclosure_id": "analysis_ready_not_confirmatory_complete",
            "category": "analysis status",
            "severity": "important",
            "title": "Analysis-ready does not mean confirmatory-complete",
            "statement": analysis_boundary,
            "analysis_consequence": analysis_next_action,
        },
        *(_jointmoe_scope_disclosures() if jointmoe_release else []),
        {
            "disclosure_id": "validation_scope_by_tier",
            "category": "evaluation scope",
            "severity": "important",
            "title": "Zero-shot validation is available only for K1",
            "statement": (
                "Every K1 readout evaluates all 200 languages: one trained and "
                "199 zero-shot. K12, K50, and K200 readouts evaluate only their "
                "12, 50, or 200 trained languages and contain no zero-shot rows."
            ),
            "analysis_consequence": (
                "Use validation_scope before comparing tiers. Cross-tier "
                "zero-shot comparisons are unavailable; only K1 supports a "
                "trained-versus-zero-shot contrast."
            ),
        },
        {
            "disclosure_id": "checkpoint_preservation",
            "category": "artifact preservation",
            "severity": "important",
            "title": "Some historical checkpoint directories are absent now",
            "statement": (
                f"{len(missing_checkpoints)} of 400 readout checkpoint directories "
                "are not present in the current build-host filesystem "
                f"({missing_checkpoints_by_ladder}). Their selected evaluation "
                "reports, manifests, sufficient statistics, and hashes remain "
                "preserved in the release."
            ),
            "analysis_consequence": (
                "Existing outcomes remain analyzable, but checkpoint-level "
                "re-evaluation or new benchmarks for those readouts require "
                "checkpoint restoration or retraining. Current presence is not "
                "used as evidence that execution was valid."
            ),
        },
        {
            "disclosure_id": (
                "structured_holdout_fit_pending"
                if jointmoe_release
                else "confirmatory_sequence_pending"
            ),
            "category": "analysis status",
            "severity": "important",
            "title": (
                "Structured-holdout fitting is pending"
                if jointmoe_release
                else "Confirmatory sequence is pending"
            ),
            "statement": analysis_boundary,
            "analysis_consequence": (
                "Do not select a scaling-law form from all-data fit quality. "
                "Run the frozen holdouts first and keep terminal/random/seed "
                "views secondary."
                if jointmoe_release
                else (
                    "Treat current A-D summaries as descriptive; do not label "
                    "this release confirmatory-complete."
                )
            ),
        },
        {
            "disclosure_id": "physical_vs_attributed_compute",
            "category": "compute accounting",
            "severity": "important",
            "title": "Physical and attributed compute are distinct",
            "statement": (
                "Per-readout attribution includes the relevant shared trunk "
                "prefix, whereas physical totals count each training job and "
                "Slurm allocation once."
            ),
            "analysis_consequence": (
                "Never sum logical per-readout attribution to estimate total "
                "physical study compute."
            ),
        },
        {
            "disclosure_id": "legacy_language_metadata",
            "category": "data provenance",
            "severity": "important",
            "title": "Legacy language metadata is partial",
            "statement": (
                "Family, name, and resource metadata are joined only through "
                "a unique ISO-639-3 prefix; production scripts remain "
                "authoritative and unmatched values stay null."
            ),
            "analysis_consequence": (
                "Use the explicit join method and stale-script flag in language "
                "group analyses; do not impute the 20 unmatched languages."
            ),
        },
    ]
    helpers = _presentation_helpers()
    if helpers is None:
        catalog = _fallback_catalog(dataframes)
        checklist = _fallback_checklist(release_metadata)
        excel_metadata = _write_fallback_excel(
            build_dir / "stage1_abcd_data_workbook.xlsx",
            dataframes,
            release_metadata,
            catalog,
            checklist,
        )
        excel_metadata["path"] = "stage1_abcd_data_workbook.xlsx"
        readme = _fallback_readme(release_metadata, catalog, checklist)
        presentation_backend = "built_in_fallback"
    else:
        (
            build_table_catalog,
            build_neurips_checklist,
            render_readme,
            write_excel_workbook,
        ) = helpers
        catalog = build_table_catalog(dataframes)
        _assert_catalog_contracts(catalog)
        release_metadata["data_dictionary_contract_status"] = "pass"
        excel_metadata = write_excel_workbook(
            build_dir / "stage1_abcd_data_workbook.xlsx",
            dataframes,
            release_metadata,
            catalog=catalog,
            checklist=None,
            disclosures=disclosures,
        )
        excel_metadata["path"] = "stage1_abcd_data_workbook.xlsx"
        omissions = excel_metadata.get("omissions", [])
        checklist = build_neurips_checklist(
            release_metadata,
            table_counts=table_counts,
            disclosures=disclosures,
            excel_omissions=omissions,
        )
        readme = render_readme(
            release_metadata,
            catalog,
            checklist,
            table_counts=table_counts,
            disclosures=disclosures,
            excel_omissions=omissions,
        )
        presentation_backend = "stage1_ad_release_presentation"
    (build_dir / "README.md").write_text(readme, encoding="utf-8")
    _write_json(
        build_dir / "data_dictionary.json",
        {
            "schema": f"{SCHEMA}/data_dictionary",
            "generated_at_utc": _utc_now(),
            "tables": catalog,
        },
    )
    _write_json(
        build_dir / "neurips_checklist.json",
        {
            "schema": f"{SCHEMA}/neurips_checklist",
            "generated_at_utc": _utc_now(),
            "items": checklist,
        },
    )
    _write_json(
        build_dir / "qa_report.json",
        {
            "schema": f"{SCHEMA}/qa",
            "status": "pass",
            "generated_at_utc": _utc_now(),
            "checks": qa_issues,
        },
    )
    _write_json(
        build_dir / "excel_workbook_receipt.json",
        {
            "schema": f"{SCHEMA}/excel_workbook",
            **excel_metadata,
        },
    )

    generation_code_inventory = _snapshot_generation_code(
        repo_root,
        build_dir,
        () if build_spec_path is None else (build_spec_path,),
    )
    git_commit = _run_text(["git", "rev-parse", "HEAD"], cwd=repo_root)
    git_status = _run_text(["git", "status", "--porcelain=v1"], cwd=repo_root)
    manifest: dict[str, Any] = {
        **release_metadata,
        "status": release_state,
        "presentation_backend": presentation_backend,
        "source_contract": {
            "primary_run_roots": {
                key: str(value) for key, value in primary_roots.items()
            },
            "all_discovered_run_roots": [str(path) for path in run_roots],
            "cd_source": definition.cd_source,
            "cd_ledger_path": str(ledger_path),
            "cd_ledger_sha256": _sha256(ledger_path),
            "corpus_manifest_path": str(corpus_manifest_path),
            "corpus_manifest_sha256": _sha256(corpus_manifest_path),
            "seed_confirmation_summary_path": (
                None
                if definition.seed_confirmation_summary is None
                else str((repo_root / definition.seed_confirmation_summary).resolve())
            ),
            "seed_confirmation_summary_sha256": (
                None
                if definition.seed_confirmation_summary is None
                else _sha256((repo_root / definition.seed_confirmation_summary).resolve())
            ),
            "validation_candidate_contract": {
                "expected_candidates": (
                    definition.expected_validation_candidate_count
                ),
                "expected_duplicates_beyond_one_per_readout": (
                    definition.expected_duplicate_candidate_count
                ),
            },
        },
        "table_inventory": table_metadata,
        "release_build": {
            "entrypoint": (
                "multilingual_megatron.experiments.scaling.stage1_release.builder"
            ),
            "repo_root_argument": ".",
            "output_argument": f"result_analysis/{release_name}",
            "build_spec_path": (
                None if build_spec_path is None else str(build_spec_path.resolve())
            ),
            "build_spec_sha256": (
                None if build_spec_path is None else _sha256(build_spec_path.resolve())
            ),
            "slurm_missing_policy": slurm_missing_policy,
            "command": (
                "PYTHONPATH=src:. python -m multilingual_megatron.experiments."
                "scaling.stage1_release.builder "
                + (
                    ""
                    if build_spec_path is None
                    else f"--build-spec {build_spec_path} "
                )
                + "--output "
                f"result_analysis/{release_name} --slurm-missing-policy "
                f"{slurm_missing_policy}"
            ),
        },
        "generation_code_inventory": generation_code_inventory,
        "excel_workbook": excel_metadata,
        "slurm_accounting": {
            "status": slurm_report["status"],
            "counts": slurm_report["counts"],
            "collected_at_utc": slurm_report["collection"]["collected_at_utc"],
            "receipt": "receipts/slurm_accounting.json",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": _safe_version("pandas"),
            "pyarrow": _safe_version("pyarrow"),
            "openpyxl": _safe_version("openpyxl"),
        },
        "git": {
            "commit": git_commit,
            "worktree_clean": git_status == "",
            "status_porcelain_sha256": (
                None
                if git_status is None
                else __import__("hashlib")
                .sha256(git_status.encode("utf-8"))
                .hexdigest()
            ),
            "disclosure": (
                "The manifest records the commit and a hash of dirty-state "
                "paths. Exact local release-generator and focused-test source "
                "bytes are copied under code/ and hash-sealed; unrelated user "
                "changes are not embedded."
            ),
        },
    }
    manifest["generated_file_inventory"] = _file_inventory(build_dir)
    manifest["manifest_integrity_sha256"] = (
        __import__("hashlib").sha256(_canonical_bytes(manifest)).hexdigest()
    )
    _write_json(build_dir / "release_manifest.json", manifest)
    build_dir.rename(output_dir)
    return {
        "release_dir": str(output_dir),
        "manifest": str(output_dir / "release_manifest.json"),
        "tables": {name: metadata["rows"] for name, metadata in table_metadata.items()},
        "slurm": slurm_report["counts"],
        "status": release_state,
    }
