#!/usr/bin/env python3
"""Orchestrate the sealed Stage 1 A--D analytical data release.

The release is intentionally split into:

* canonical typed Parquet tables under ``tables/``;
* exact acquisition receipts and immutable-source hashes under ``receipts/``;
* a human-oriented Excel view and README; and
* a manifest whose integrity hash covers the manifest payload and every
  generated release artifact except the manifest file itself.

The builder never edits source run artifacts.  It also keeps independent
training trajectories, repeated checkpoint readouts, and per-language
observations as separate statistical grains.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from multilingual_megatron.experiments.scaling.stage1_release.execution_specs import build_execution_spec_rows
from multilingual_megatron.experiments.scaling.stage1_release.contracts import (
    EXPECTED_COUNTS,
    LADDER_COUNTS,
    PROVENANCE_CLASSES,
    RELEASE_NAME,
    SCHEMA,
    ReleaseBuildSpec,
    Stage1ReleaseError,
    _canonical_bytes,
    _json_cell,
    _load_json,
    _model_and_trajectory_rows,
    _primary_roots,
    _production_jobs,
    _require_analysis_stack,
    _sha256,
    _source_registry,
    _write_json,
    historical_release_build_spec,
    load_release_build_spec,
)
from multilingual_megatron.experiments.scaling.stage1_release.builder_common import _assign_qa_issue_ids
from multilingual_megatron.experiments.scaling.stage1_release.builder_common import _canonical_qa_evidence_cell
from multilingual_megatron.experiments.scaling.stage1_release.builder_common import _columnar_records
from multilingual_megatron.experiments.scaling.stage1_release.builder_common import _json_safe
from multilingual_megatron.experiments.scaling.stage1_release.builder_common import _run_text
from multilingual_megatron.experiments.scaling.stage1_release.builder_common import _safe_version
from multilingual_megatron.experiments.scaling.stage1_release.builder_common import _utc_now
from multilingual_megatron.experiments.scaling.stage1_release.builder_orchestration import build_release
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _assert_catalog_contracts
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _fallback_catalog
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _fallback_checklist
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _fallback_readme
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _file_inventory
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _presentation_helpers
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _snapshot_generation_code
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _write_fallback_excel
from multilingual_megatron.experiments.scaling.stage1_release.builder_outputs import _write_parquet_tables
from multilingual_megatron.experiments.scaling.stage1_release.builder_rows import ANALYSIS_REGISTRY_PATHS
from multilingual_megatron.experiments.scaling.stage1_release.builder_rows import CD_LEDGER
from multilingual_megatron.experiments.scaling.stage1_release.builder_rows import CORPUS_MANIFEST
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the sealed Stage 1 A-D analytical release."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--build-spec",
        type=Path,
        help="Versioned source binding for a non-historical release.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="New release directory; existing paths are never overwritten.",
    )
    parser.add_argument(
        "--slurm-missing-policy",
        choices=("error", "flag"),
        default="flag",
        help="Fail or explicitly flag historical sacct gaps.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    build_spec_path = None
    if args.build_spec is not None:
        build_spec_path = (
            args.build_spec.resolve()
            if args.build_spec.is_absolute()
            else (repo_root / args.build_spec).resolve()
        )
        build_spec = load_release_build_spec(build_spec_path)
    else:
        build_spec = historical_release_build_spec()
    output_argument = args.output or Path("result_analysis") / build_spec.release_name
    output = (
        output_argument.resolve()
        if output_argument.is_absolute()
        else (repo_root / output_argument).resolve()
    )
    result = build_release(
        repo_root=repo_root,
        output_dir=output,
        slurm_missing_policy=args.slurm_missing_policy,
        build_spec=build_spec,
        build_spec_path=build_spec_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
