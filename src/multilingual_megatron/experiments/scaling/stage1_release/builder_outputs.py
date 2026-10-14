"""Typed table writing, presentation fallbacks, and release sealing helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
from typing import Any

from multilingual_megatron.experiments.scaling.stage1_release.contracts import (
    SCHEMA,
    Stage1ReleaseError,
    _require_analysis_stack,
    _sha256,
)
from multilingual_megatron.experiments.scaling.stage1_release.builder_common import _columnar_records


def _write_parquet_tables(
    *,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pd, _ = _require_analysis_stack()
    directory.mkdir(parents=True, exist_ok=True)
    dataframes: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    empty_table_columns = {
        "validity_events": (
            "object_type",
            "object_id",
            "issue_type",
            "severity",
            "verdict",
            "evidence_path",
            "details",
            "provenance_class",
        ),
    }
    for name, records in tables.items():
        columnar = _columnar_records(list(records))
        frame = pd.DataFrame.from_records(
            columnar,
            columns=(empty_table_columns.get(name) if not columnar else None),
        )
        path = directory / f"{name}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        reread = pd.read_parquet(path)
        if len(reread) != len(frame) or list(reread.columns) != list(frame.columns):
            raise Stage1ReleaseError(f"Parquet round-trip mismatch for {name}")
        dataframes[name] = frame
        metadata[name] = {
            "path": f"tables/{name}.parquet",
            "rows": len(frame),
            "columns": list(frame.columns),
            "column_dtypes": {
                str(column): str(dtype) for column, dtype in frame.dtypes.items()
            },
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    for name in (
        "release_summary",
        "trajectories",
        "readouts",
        "validation_aggregates",
        "training_summary",
        "compute_summary",
        "qa_issues",
    ):
        frame = dataframes[name]
        frame.to_csv(directory / f"{name}.csv", index=False)
    return dataframes, metadata


def _fallback_catalog(dataframes: Mapping[str, Any]) -> dict[str, Any]:
    purposes = {
        "studies": "Frozen/configured study protocols by ladder.",
        "source_roots": "Primary and recovery run roots included in discovery.",
        "model_configs": "Architecture configurations; one row per ladder/E.",
        "trajectories": "Independent experimental units; one row per A-D/E/K.",
        "readouts": "Canonical checkpoint readouts; repeated within trajectory.",
        "validation_aggregates": "Reported aggregate/cohort validation metrics.",
        "validation_language": "Per-language validation sufficient statistics.",
        "validation_candidates": "All canonical and duplicate evaluation candidates.",
        "validation_artifacts": "Hash inventory for validation artifacts.",
        "validation_provenance": "Validation lineage edges.",
        "training_jobs": "Production trunk and branch execution jobs.",
        "training_configs": "Full resolved configuration snapshot per training job.",
        "training_segments": "Segment-level training execution lineage.",
        "training_metrics": "Parsed training iterations/parameters/memory/TFLOPS.",
        "runtime_parameters": "Deduplicated runtime parameter observations.",
        "training_summary": "Training and telemetry aggregates per execution job.",
        "gpu_samples": "Raw normalized nvidia-smi samples.",
        "training_artifacts": "Hash inventory for training logs/manifests/telemetry.",
        "dataset_release": "Corpus release contract and manifest identity.",
        "corpus_languages": "Per-language corpus selection and validation inventory.",
        "data_exposure": "Exact realized trained-language exposure per readout.",
        "routing_layers": "Per-language, per-layer MoE routing diagnostics.",
        "job_attempts": "Every discovered logical-to-Slurm attempt mapping.",
        "slurm_allocations": "Normalized allocated Slurm job/task records.",
        "compute_summary": "Attempt-aware compute and utilization summary per job.",
        "validity_events": "Exclusions, duplicates, recoveries, and validity verdicts.",
        "analysis_registry": "Existing protocol and analysis artifacts with hashes.",
        "qa_issues": "Machine-checkable release assertions and disclosures.",
        "release_summary": "Compact A-D and total release counts/compute.",
    }
    return {
        name: {
            "table": name,
            "purpose": purposes.get(name, "Canonical release table."),
            "rows": len(frame),
            "columns": [
                {
                    "name": str(column),
                    "dtype": str(frame.dtypes[column]),
                    "description": (
                        "Normalized field; see table purpose, README, and "
                        "source lineage columns."
                    ),
                }
                for column in frame.columns
            ],
        }
        for name, frame in dataframes.items()
    }


def _presentation_helpers() -> tuple[Any, Any, Any, Any] | None:
    try:
        from multilingual_megatron.experiments.scaling.stage1_release.presentation import build_neurips_checklist
        from multilingual_megatron.experiments.scaling.stage1_release.presentation import build_table_catalog
        from multilingual_megatron.experiments.scaling.stage1_release.presentation import render_readme
        from multilingual_megatron.experiments.scaling.stage1_release.presentation import write_excel_workbook
    except ImportError:
        return None
    return (
        build_table_catalog,
        build_neurips_checklist,
        render_readme,
        write_excel_workbook,
    )


def _fallback_checklist(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "item": "Experimental details and reproducibility",
            "status": "satisfied",
            "evidence": "studies, model_configs, training_configs, trajectories",
        },
        {
            "item": "Compute resources and accounting",
            "status": (
                "satisfied"
                if metadata["missing_slurm_attempts"] == 0
                else "partial_with_explicit_missingness"
            ),
            "evidence": "job_attempts, slurm_allocations, compute_summary",
        },
        {
            "item": "Data and evaluation provenance",
            "status": "satisfied",
            "evidence": (
                "dataset_release, corpus_languages, data_exposure, "
                "validation_artifacts, validation_provenance"
            ),
        },
        {
            "item": "Independent replications and uncertainty",
            "status": "disclosed_limitation",
            "evidence": (
                "One configured seed per trajectory; checkpoints/languages "
                "are repeated measures."
            ),
        },
        {
            "item": "Limitations and exclusions",
            "status": "satisfied",
            "evidence": "validity_events, qa_issues, README",
        },
    ]


def _fallback_readme(
    metadata: Mapping[str, Any],
    catalog: Mapping[str, Any],
    checklist: Sequence[Mapping[str, Any]],
) -> str:
    table_lines = "\n".join(
        f"- `{name}`: {entry['rows']:,} rows — {entry['purpose']}"
        for name, entry in catalog.items()
    )
    checklist_lines = "\n".join(
        f"- **{row['item']}** — `{row['status']}`: {row['evidence']}"
        for row in checklist
    )
    return f"""# Stage 1 A–D central analytical release

Schema: `{SCHEMA}`

Built at `{metadata["built_at_utc"]}` from immutable, hash-recorded source
artifacts. Canonical data are the typed Parquet files under `tables/`; the
Excel workbook is a derived convenience view.

## Statistical interpretation

The independent experimental unit is one A–D × expert-count × language-count
trajectory (64 total). The 400 checkpoint readouts and 46,200 per-language
validation rows are repeated observations, not independent replications. Each
trajectory uses one configured model/data seed, so seed uncertainty is not
identified. Use within-trajectory methods or cluster inference by trajectory;
do not treat language or checkpoint rows as IID seeds.

Nominal token budgets identify planned ladder readouts. `actual_tokens` is the
observed checkpoint exposure and must be used for continuous compute/exposure
analysis. Trunk and branch jobs are preserved separately; shared trunks must
not be counted once per branch.

## Measurement boundaries

Slurm GPU-hours include every observed attempt, including retries. Missing
historical accounting is explicit rather than imputed. GPU energy is
trapezoidal integration of observed board power only; it excludes host,
network, storage, cooling, and facility overhead. MFU is absent where no
hardware-peak denominator was configured.

Validation selection retains every valid duplicate candidate and its semantic
sufficient-statistics identity. Trained and zero-shot languages are distinct
roles. The routing table is applicable only to sparse-MoE readouts.

## Tables

{table_lines}

## NeurIPS-style completeness

{checklist_lines}

## Recommended analysis entry points

Start with `tables/release_summary.parquet`, `tables/trajectories.parquet`,
`tables/readouts.parquet`, and `tables/validation_aggregates.parquet`. Join
fine-grained tables by `trajectory_id`, `readout_id`, `training_job_id`,
`attempt_uid`, and artifact hashes as appropriate. Consult
`data_dictionary.json`, `qa_report.json`, and `receipts/slurm_accounting.json`
before inferential analysis.
"""


def _write_fallback_excel(
    path: Path,
    dataframes: Mapping[str, Any],
    metadata: Mapping[str, Any],
    catalog: Mapping[str, Any],
    checklist: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pd, _ = _require_analysis_stack()
    selections = {
        "Release": "release_summary",
        "QA": "qa_issues",
        "NeurIPS": None,
        "Studies": "studies",
        "Models": "model_configs",
        "Trajectories": "trajectories",
        "Readouts": "readouts",
        "ValidationAgg": "validation_aggregates",
        "ValidationLang": "validation_language",
        "TrainingSummary": "training_summary",
        "Compute": "compute_summary",
        "Attempts": "job_attempts",
        "Exposure": "data_exposure",
        "Validity": "validity_events",
        "Analyses": "analysis_registry",
    }
    written: dict[str, int] = {}
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, table_name in selections.items():
            frame = (
                pd.DataFrame.from_records(list(checklist))
                if table_name is None
                else dataframes[table_name]
            )
            if len(frame) > 1_048_575:
                raise Stage1ReleaseError(f"Excel sheet {sheet} would be truncated")
            frame.to_excel(writer, sheet_name=sheet, index=False)
            worksheet = writer.book[sheet]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)
            for column in worksheet.columns:
                letter = column[0].column_letter
                width = min(
                    55,
                    max(
                        10,
                        max(
                            len(str(cell.value)) if cell.value is not None else 0
                            for cell in column[:200]
                        )
                        + 2,
                    ),
                )
                worksheet.column_dimensions[letter].width = width
            written[sheet] = len(frame)
    return {
        "sheets": written,
        "canonical_source": "Parquet tables; workbook is derived.",
        "truncated": False,
    }


def _assert_catalog_contracts(catalog: Sequence[Mapping[str, Any]]) -> None:
    """Fail a release when any available table has a stale or invalid contract."""

    errors: list[str] = []
    available = [row for row in catalog if row.get("available") is True]
    names = [str(row.get("table_name")) for row in available]
    duplicate_names = sorted(
        name for name, count in Counter(names).items() if count != 1
    )
    if duplicate_names:
        errors.append(f"duplicate catalog entries: {duplicate_names}")
    for row in available:
        name = str(row.get("table_name"))
        if row.get("category") == "uncatalogued":
            errors.append(f"{name}: uncatalogued")
        if row.get("missing_key_columns"):
            errors.append(
                f"{name}: missing required columns {row['missing_key_columns']}"
            )
        if row.get("missing_primary_key_columns"):
            errors.append(
                f"{name}: missing primary-key columns "
                f"{row['missing_primary_key_columns']}"
            )
        if row.get("primary_key_valid") is not True:
            errors.append(
                f"{name}: invalid primary key; null rows="
                f"{row.get('primary_key_null_row_count')}, duplicate rows="
                f"{row.get('primary_key_duplicate_row_count')}"
            )
    if errors:
        raise Stage1ReleaseError(
            "Publication catalog contract validation failed: " + "; ".join(errors)
        )


def _snapshot_generation_code(
    repo_root: Path,
    release_dir: Path,
    extra_sources: Sequence[Path] = (),
) -> list[dict[str, Any]]:
    """Copy and hash the exact local generator and focused-test source bytes."""

    scaling_dir = repo_root / "src/multilingual_megatron/experiments/scaling"
    sources = {
        *(scaling_dir / "stage1_release").rglob("*.py"),
        *(repo_root / "tests").glob("test_stage1_ad_*.py"),
        repo_root / "scripts/build_stage1_ad_analysis_notebook.py",
        *(Path(path) for path in extra_sources),
    }
    inventory: list[dict[str, Any]] = []
    for source in sorted(path.resolve() for path in sources if path.is_file()):
        relative = source.relative_to(repo_root)
        destination = release_dir / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        inventory.append(
            {
                "source_repo_path": str(relative),
                "release_snapshot_path": str(destination.relative_to(release_dir)),
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    if not inventory:
        raise Stage1ReleaseError("No release-generation source files were found")
    return inventory


def _file_inventory(release_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(release_dir.rglob("*")):
        if not path.is_file() or path.name == "release_manifest.json":
            continue
        rows.append(
            {
                "path": str(path.relative_to(release_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows
