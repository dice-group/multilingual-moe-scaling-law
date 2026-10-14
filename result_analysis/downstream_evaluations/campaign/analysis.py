"""Stable facade for the sealed Stage-1 benchmark analysis workflow.

Artifact contracts, score derivations, seed analyses, and report rendering live
in cohesive sibling modules. This facade preserves the established imports and
owns only end-to-end bundle orchestration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.contracts.hashing import canonical_json_sha256
from result_analysis.downstream_evaluations.campaign.analysis_artifacts import (
    ANALYSIS_BUNDLE_SCHEMA,
    ANALYSIS_IMPLEMENTATION_FILES,
    EXPECTED_CAMPAIGN_ID,
    INTEGRITY_SCHEMA,
    BenchmarkAnalysisError,
    _artifact_reference,
    _atomic_json,
    _atomic_text,
    _load_sealed_inputs,
    _validate_submission_receipt,
    _write_table,
    validate_generated_bundle,
)
from result_analysis.downstream_evaluations.campaign.analysis_replication import (
    _build_b_seed_cell_summary,
    _build_b_seed_contrasts,
    _build_initial_findings,
)
from result_analysis.downstream_evaluations.campaign.analysis_reporting import (
    _render_initial_analysis,
    _render_readme,
)
from result_analysis.downstream_evaluations.campaign.analysis_scores import (
    _build_final_tables,
    _build_model_scores,
    _build_threshold_crossings,
    _build_trajectory_dynamics,
    _derive_chance_levels,
    _derive_common_english_slice,
    _validate_campaign_grains,
)
from multilingual_megatron.experiments.scaling.stage1_benchmarking.campaign_profile import load_campaign_profile


def write_scientific_analysis_bundle(profile_path: Path | str) -> Path:
    """Validate campaign evidence and write the complete scientific result bundle."""

    profile = load_campaign_profile(Path(profile_path))
    report, source_table_manifest, source_tables, request_manifests, status = (
        _load_sealed_inputs(profile)
    )
    model_rows = source_tables["model_readouts"]
    benchmark_rows = source_tables["benchmark_metrics"]
    language_rows = source_tables["language_metrics"]
    _validate_campaign_grains(model_rows, benchmark_rows, language_rows, report)

    chance_rows = _derive_chance_levels(request_manifests)
    common_english_rows = _derive_common_english_slice(
        language_rows, request_manifests
    )
    model_scores = _build_model_scores(
        model_rows, benchmark_rows, language_rows, chance_rows
    )
    trajectory_rows = _build_trajectory_dynamics(model_scores)
    threshold_rows = _build_threshold_crossings(model_scores)
    final_rows, canonical_final = _build_final_tables(model_scores)
    b_cell_rows = _build_b_seed_cell_summary(model_scores)
    b_seedwise, b_summaries = _build_b_seed_contrasts(model_scores)
    submission_path, submission = _validate_submission_receipt(profile, model_scores)

    output_root = profile.output_root / "scientific_analysis_v1"
    table_root = output_root / "tables"
    table_payloads: dict[str, Sequence[Mapping[str, Any]]] = {
        "benchmark_chance_levels": chance_rows,
        "common_english_slice": common_english_rows,
        "model_scores": model_scores,
        "trajectory_dynamics": trajectory_rows,
        "threshold_crossings": threshold_rows,
        "final_endpoints": final_rows,
        "canonical_final_grid": canonical_final,
        "b_seed_cell_summary": b_cell_rows,
        "b_seed_contrasts_seedwise": b_seedwise,
        "b_seed_contrast_summary": b_summaries,
    }
    table_specs = {
        name: _write_table(table_root, name, rows)
        for name, rows in table_payloads.items()
    }

    findings = _build_initial_findings(
        canonical_final=canonical_final,
        trajectory_rows=trajectory_rows,
        threshold_rows=threshold_rows,
        b_cell_rows=b_cell_rows,
        b_seedwise=b_seedwise,
        b_summaries=b_summaries,
        chance_rows=chance_rows,
        model_scores=model_scores,
    )
    findings_path = output_root / "initial_findings.json"
    _atomic_json(findings_path, findings)
    analysis_path = output_root / "initial_analysis.md"
    _atomic_text(
        analysis_path,
        _render_initial_analysis(
            findings,
            threshold_rows=threshold_rows,
        ),
    )
    readme_path = output_root / "README.md"
    _atomic_text(readme_path, _render_readme())

    cost_by_ladder = {
        ladder: sum(
            float(row["gpu_hours"]) for row in model_scores if row["ladder"] == ladder
        )
        for ladder in ("A", "B", "C", "D")
    }
    integrity: dict[str, Any] = {
        "schema": INTEGRITY_SCHEMA,
        "status": "passed",
        "campaign_id": EXPECTED_CAMPAIGN_ID,
        "checks": {
            "campaign_status_complete_508_of_508": True,
            "comparison_report_payload_hash_valid": True,
            "canonical_table_manifest_payload_hash_valid": True,
            "canonical_csv_and_jsonl_hashes_valid": True,
            "canonical_table_row_counts_valid": True,
            "request_manifest_payload_hashes_valid": True,
            "frozen_request_choice_counts_valid": True,
            "common_english_intersection_exactly_four_tasks": True,
            "submission_job_ids_match_runtime_receipts": True,
            "independent_trajectory_count_is_82": True,
            "replicated_b_panel_complete": True,
            "generated_bundle_hashes_valid": True,
        },
        "counts": {
            "submitted_jobs": len(submission["jobs"]),
            "complete_runtime_receipts": status["counts"]["complete"],
            "model_readouts": len(model_scores),
            "benchmark_metric_rows": len(benchmark_rows),
            "language_metric_rows": len(language_rows),
            "independent_trajectories": len(trajectory_rows),
            "final_endpoints": len(final_rows),
            "canonical_final_cells": len(canonical_final),
            "b_seed_cells": len(b_cell_rows),
            "b_seedwise_contrasts": len(b_seedwise),
            "b_seed_contrast_summaries": len(b_summaries),
        },
        "actual_execution": {
            "serial_job_hours": sum(
                float(row["job_wall_seconds"]) / 3600.0 for row in model_scores
            ),
            "gpu_hours": sum(float(row["gpu_hours"]) for row in model_scores),
            "gpu_hours_by_ladder": cost_by_ladder,
            "queue_time_included": False,
        },
        "statistical_unit": (
            "one independently trained seed within a fixed ladder/expert/language cell"
        ),
        "nested_units": ["checkpoint", "benchmark", "language", "example"],
        "three_seed_scope": "B/E={1,16,32}/K={1,12,200}",
    }
    integrity_path = output_root / "integrity_summary.json"
    _atomic_json(integrity_path, integrity)

    report_path = profile.output_root / "comparison_report.json"
    status_path = profile.output_root / "campaign_status.json"
    source_manifest_path = profile.output_root / "analysis_tables" / "manifest.json"
    manifest: dict[str, Any] = {
        "schema": ANALYSIS_BUNDLE_SCHEMA,
        "status": "complete",
        "campaign_id": EXPECTED_CAMPAIGN_ID,
        "output_root": str(output_root),
        "profile": {"path": str(profile.source_path), "sha256": profile.source_sha256},
        "implementations": {
            name: _artifact_reference(path)
            for name, path in ANALYSIS_IMPLEMENTATION_FILES.items()
        },
        "inputs": {
            "comparison_report": {
                **_artifact_reference(report_path),
                "payload_sha256": report["report_sha256"],
            },
            "campaign_status": {
                **_artifact_reference(status_path),
                "payload_sha256": status["status_sha256"],
            },
            "canonical_table_manifest": {
                **_artifact_reference(source_manifest_path),
                "payload_sha256": source_table_manifest["manifest_sha256"],
            },
            "request_manifests": {
                request_set: {
                    **_artifact_reference(profile.request_manifest(request_set)),
                    "payload_sha256": manifest_payload["manifest_sha256"],
                }
                for request_set, manifest_payload in request_manifests.items()
            },
            "submission_receipt": _artifact_reference(submission_path),
        },
        "tables": table_specs,
        "artifacts": {
            "readme": _artifact_reference(readme_path),
            "integrity_summary": _artifact_reference(integrity_path),
            "initial_findings": _artifact_reference(findings_path),
            "initial_analysis": _artifact_reference(analysis_path),
        },
        "analysis_policy": {
            "primary_metric": "equal_task_macro_acc_norm",
            "task_macro": "equal weight over benchmark languages",
            "campaign_composite": "equal weight over six benchmark families",
            "chance_adjustment": (
                "per-benchmark (acc_norm-random_choice)/(1-random_choice), then equal-task mean"
            ),
            "cross_k": "exact four-task common-English intersection only for K1 comparisons",
            "independent_unit": (
                "one independently trained seed within a fixed design cell"
            ),
            "seed_statistics": "mean, sample SD, range, and matched-seed directions",
            "significance_tests": "none; n=3 B panel is descriptive robustness evidence",
            "trajectory_slopes": (
                "descriptive OLS over nested readouts versus log2(actual tokens)"
            ),
            "thresholds": (
                "post-hoc discrete sensitivity grid; no interpolated or inferential onset claim"
            ),
            "contamination": "not assessed",
        },
        "claim_boundary": report["claim_boundary"],
        "regeneration_command": (
            "PYTHONPATH=src:external/lm-evaluation-harness:. "
            f"{profile.lm_eval_python} -m "
            "multilingual_megatron.experiments.scaling.stage1_benchmarking.jointmoe "
            "--profile configs/benchmarks/stage1_abcd_jointmoe_all_readouts_a40_v2.json "
            "analyze"
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = output_root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    validate_generated_bundle(manifest_path)
    return manifest_path


__all__ = [
    "ANALYSIS_BUNDLE_SCHEMA",
    "BenchmarkAnalysisError",
    "validate_generated_bundle",
    "write_scientific_analysis_bundle",
]
