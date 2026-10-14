"""Build and validate the sealed Stage-1 benchmark visualization bundle.

This module is the stable public facade. Pure table derivations and rendering
live in sibling modules; this file owns input verification, artifact writing,
manifest construction, and the single ``visualize`` workflow.
"""

from __future__ import annotations

import csv
import io
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.contracts.hashing import canonical_json_sha256
from result_analysis.downstream_evaluations.campaign import (
    paper_plots,
    visualization_data,
    visualization_plots,
)
from result_analysis.downstream_evaluations.campaign.analysis_artifacts import (
    BenchmarkAnalysisError,
    EXPECTED_CAMPAIGN_ID,
    _artifact_reference,
    _atomic_json,
    _atomic_text,
    _load_object,
    _read_jsonl,
    _resolved_reference,
    _verify_payload_hash,
    validate_generated_bundle,
)
from multilingual_megatron.experiments.scaling.stage1_benchmarking.campaign_profile import (
    load_campaign_profile,
)
from result_analysis.downstream_evaluations.campaign.paper_plots import (
    PAPER_FIGURE_FILES,
    plot_b_seed_task_curves,
    plot_composite_slope_map,
    plot_full_grid_compute_curves,
    plot_task_signal,
)
from result_analysis.downstream_evaluations.campaign.visualization_data import (
    LANGUAGE_COUNTS,
    build_b_seed_task_curve_rows,
    build_scaling_plot_rows,
    build_slope_summary_rows,
    build_task_endpoint_rows,
    build_task_slope_rows,
)
from result_analysis.downstream_evaluations.campaign.visualization_plots import (
    plot_scaling_grid,
    plot_slope_heatmap,
)


VISUALIZATION_SCHEMA = "stage1_jointmoe_benchmark_visualization_bundle/v2"
VISUALIZATION_SUMMARY_SCHEMA = "stage1_jointmoe_benchmark_visualization_summary/v2"
AUDIT_FIGURE_FILES = {
    "absolute_grid": "benchmark_scaling_grid_absolute",
    "change_grid": "benchmark_scaling_grid_change",
    "slope_heatmap": "benchmark_scaling_slope_heatmap",
}
FIGURE_FILES = {**AUDIT_FIGURE_FILES, **PAPER_FIGURE_FILES}
TABLE_FILES = {
    "scaling_plot_data": "benchmark_scaling_plot_data",
    "slope_summary": "benchmark_scaling_slope_summary",
    "task_endpoint_summary": "benchmark_task_endpoint_summary",
    "task_slope_summary": "benchmark_task_slope_summary",
    "b_seed_task_curves": "benchmark_b_seed_task_curves",
}
IMPLEMENTATION_FILES = {
    "bundle": Path(__file__).resolve(),
    "data": Path(visualization_data.__file__).resolve(),
    "audit_plots": Path(visualization_plots.__file__).resolve(),
    "paper_plots": Path(paper_plots.__file__).resolve(),
}

# Preserve the existing import facade while keeping one implementation owner.
_build_scaling_plot_rows = build_scaling_plot_rows
_build_slope_summary_rows = build_slope_summary_rows


def _write_rows(
    root: Path, name: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not rows:
        raise BenchmarkAnalysisError(f"visualization table {name} has no rows")
    columns = list(rows[0])
    if any(set(row) != set(columns) for row in rows):
        raise BenchmarkAnalysisError(f"visualization table {name} column drift")
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    json.dumps(value, allow_nan=False, sort_keys=True)
                    if isinstance(value, (list, tuple, dict))
                    else value
                )
                for key, value in row.items()
            }
        )
    csv_path = root / f"{name}.csv"
    _atomic_text(csv_path, csv_buffer.getvalue())
    jsonl_path = root / f"{name}.jsonl"
    _atomic_text(
        jsonl_path,
        "".join(
            json.dumps(dict(row), allow_nan=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )
    return {
        "rows": len(rows),
        "columns": columns,
        "csv": _artifact_reference(csv_path),
        "jsonl": _artifact_reference(jsonl_path),
    }


def _load_table_from_manifest(
    manifest: Mapping[str, Any], name: str, *, label: str
) -> list[dict[str, Any]]:
    tables = manifest.get("tables")
    if not isinstance(tables, dict) or name not in tables:
        raise BenchmarkAnalysisError(f"{label} lacks table {name}")
    spec = tables[name]
    if not isinstance(spec, dict):
        raise BenchmarkAnalysisError(f"invalid {label} table contract for {name}")
    columns = spec.get("columns")
    reference = spec.get("jsonl")
    if not isinstance(columns, list) or not isinstance(reference, dict):
        raise BenchmarkAnalysisError(f"invalid {label} JSONL contract for {name}")
    path_value = reference.get("path")
    if not isinstance(path_value, str):
        raise BenchmarkAnalysisError(f"{label} table {name} lacks JSONL path")
    path = _resolved_reference(
        reference,
        expected_path=Path(path_value),
        label=f"{label} {name} JSONL",
    )
    rows = _read_jsonl(path, columns=columns, label=f"{label} {name}")
    if len(rows) != int(spec.get("rows", -1)):
        raise BenchmarkAnalysisError(f"{label} table {name} row-count drift")
    return rows


def _load_analysis_table(
    analysis_manifest: Mapping[str, Any], name: str
) -> list[dict[str, Any]]:
    return _load_table_from_manifest(analysis_manifest, name, label="analysis")


def _load_source_table(
    analysis_manifest: Mapping[str, Any], name: str
) -> list[dict[str, Any]]:
    inputs = analysis_manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise BenchmarkAnalysisError("analysis manifest lacks inputs")
    reference = inputs.get("canonical_table_manifest")
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise BenchmarkAnalysisError("analysis lacks canonical table manifest")
    manifest_path = _resolved_reference(
        reference,
        expected_path=Path(str(reference["path"])),
        label="canonical table manifest",
    )
    source_manifest = _load_object(manifest_path, label="canonical table manifest")
    if (
        source_manifest.get("campaign_id") != EXPECTED_CAMPAIGN_ID
        or source_manifest.get("status") != "complete"
    ):
        raise BenchmarkAnalysisError("canonical table manifest identity drift")
    _verify_payload_hash(
        source_manifest,
        field="manifest_sha256",
        label="canonical table manifest",
    )
    if reference.get("payload_sha256") != source_manifest.get("manifest_sha256"):
        raise BenchmarkAnalysisError("analysis/source table payload hash drift")
    return _load_table_from_manifest(source_manifest, name, label="source analysis")


def _build_visualization_summary(
    plot_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_languages: dict[str, Any] = {}
    for languages in LANGUAGE_COUNTS:
        selected = [
            row for row in trajectory_rows if int(row["languages"]) == languages
        ]
        slopes = [
            float(row["primary_ols_slope_per_actual_token_doubling"]) * 100.0
            for row in selected
        ]
        by_languages[f"K{languages}"] = {
            "independent_trajectories": len(selected),
            "positive_slopes": sum(value > 0.0 for value in slopes),
            "negative_slopes": sum(value < 0.0 for value in slopes),
            "zero_slopes": sum(value == 0.0 for value in slopes),
            "median_slope_pp_per_actual_token_doubling": statistics.median(slopes),
            "min_slope_pp_per_actual_token_doubling": min(slopes),
            "max_slope_pp_per_actual_token_doubling": max(slopes),
        }
    all_slopes = [
        float(row["primary_ols_slope_per_actual_token_doubling"]) * 100.0
        for row in trajectory_rows
    ]
    return {
        "schema": VISUALIZATION_SUMMARY_SCHEMA,
        "campaign_id": EXPECTED_CAMPAIGN_ID,
        "status": "complete",
        "counts": {
            "independent_trajectories": len(trajectory_rows),
            "aggregated_grid_readouts": len(plot_rows),
            "positive_trajectory_slopes": sum(value > 0.0 for value in all_slopes),
            "negative_trajectory_slopes": sum(value < 0.0 for value in all_slopes),
        },
        "slope_summary_by_languages": by_languages,
        "visual_encoding": {
            "audit_grids": "raw equal-task macro acc_norm within each request set",
            "compact_compute": (
                "chance-adjusted score change from each trajectory's 0.5B readout"
            ),
            "task_signal": "task accuracy minus frozen random-choice accuracy",
            "ribbon": (
                "observed min-max over three independent B seeds only for "
                "E={1,16,32}, K={1,12,200}"
            ),
            "single_seed": "all remaining grid cells",
        },
        "claim_boundary": (
            "Descriptive learning-curve visualization. Checkpoints are nested "
            "readouts; only training seeds within fixed cells are independent "
            "replications. Logical compute excludes physical-system overhead."
        ),
    }


def _validate_table_specs(
    output_root: Path, tables: Mapping[str, Any]
) -> None:
    if set(tables) != set(TABLE_FILES):
        raise BenchmarkAnalysisError("visualization table inventory drift")
    for name, stem in TABLE_FILES.items():
        spec = tables[name]
        if not isinstance(spec, dict):
            raise BenchmarkAnalysisError(f"invalid visualization table {name}")
        columns = spec.get("columns")
        if not isinstance(columns, list) or not columns:
            raise BenchmarkAnalysisError(f"visualization table {name} lacks columns")
        for extension in ("csv", "jsonl"):
            reference = spec.get(extension)
            if not isinstance(reference, dict):
                raise BenchmarkAnalysisError(
                    f"visualization table {name} lacks {extension}"
                )
            _resolved_reference(
                reference,
                expected_path=output_root / "figures" / f"{stem}.{extension}",
                label=f"visualization table {name}/{extension}",
            )
        rows = _read_jsonl(
            output_root / "figures" / f"{stem}.jsonl",
            columns=columns,
            label=f"visualization table {name}",
        )
        if len(rows) != int(spec.get("rows", -1)):
            raise BenchmarkAnalysisError(f"visualization table {name} row drift")


def validate_visualization_bundle(manifest_path: Path | str) -> dict[str, Any]:
    """Validate every input, table, figure, and implementation reference."""

    path = Path(manifest_path)
    payload = _load_object(path, label="benchmark visualization manifest")
    if (
        payload.get("schema") != VISUALIZATION_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("campaign_id") != EXPECTED_CAMPAIGN_ID
    ):
        raise BenchmarkAnalysisError("benchmark visualization manifest identity drift")
    _verify_payload_hash(payload, field="manifest_sha256", label="visualization bundle")
    output_root_value = payload.get("output_root")
    if not isinstance(output_root_value, str):
        raise BenchmarkAnalysisError("visualization bundle lacks output_root")
    output_root = Path(output_root_value).resolve()
    if path.resolve() != output_root / "visualization_manifest.json":
        raise BenchmarkAnalysisError("visualization manifest/output_root drift")

    implementations = payload.get("implementations")
    if not isinstance(implementations, dict) or set(implementations) != set(
        IMPLEMENTATION_FILES
    ):
        raise BenchmarkAnalysisError("visualization implementation inventory drift")
    for name, expected_path in IMPLEMENTATION_FILES.items():
        reference = implementations[name]
        if not isinstance(reference, dict):
            raise BenchmarkAnalysisError(f"invalid visualization implementation {name}")
        _resolved_reference(
            reference,
            expected_path=expected_path,
            label=f"visualization implementation {name}",
        )

    input_reference = payload.get("input_analysis_manifest")
    if not isinstance(input_reference, dict):
        raise BenchmarkAnalysisError("visualization bundle lacks analysis input")
    analysis_path = _resolved_reference(
        input_reference,
        expected_path=output_root / "manifest.json",
        label="visualization input analysis manifest",
    )
    analysis_manifest = validate_generated_bundle(analysis_path)
    if input_reference.get("payload_sha256") != analysis_manifest.get(
        "manifest_sha256"
    ):
        raise BenchmarkAnalysisError("visualization/analysis payload hash drift")

    figures = payload.get("figures")
    if not isinstance(figures, dict) or set(figures) != set(FIGURE_FILES):
        raise BenchmarkAnalysisError("visualization figure inventory drift")
    for name, stem in FIGURE_FILES.items():
        formats = figures[name]
        if not isinstance(formats, dict) or set(formats) != {"pdf", "png"}:
            raise BenchmarkAnalysisError(f"visualization format drift for {name}")
        for extension in ("pdf", "png"):
            reference = formats[extension]
            if not isinstance(reference, dict):
                raise BenchmarkAnalysisError(f"invalid {name}/{extension} reference")
            figure_path = _resolved_reference(
                reference,
                expected_path=output_root / "figures" / f"{stem}.{extension}",
                label=f"{name} {extension}",
            )
            if reference.get("bytes") != figure_path.stat().st_size:
                raise BenchmarkAnalysisError(f"visualization size drift for {name}")

    tables = payload.get("tables")
    if not isinstance(tables, dict):
        raise BenchmarkAnalysisError("visualization bundle lacks tables")
    _validate_table_specs(output_root, tables)
    summary_reference = payload.get("visualization_summary")
    if not isinstance(summary_reference, dict):
        raise BenchmarkAnalysisError("visualization bundle lacks summary")
    _resolved_reference(
        summary_reference,
        expected_path=output_root / "figures" / "visualization_summary.json",
        label="visualization summary",
    )
    return payload


def write_benchmark_scaling_visualizations(profile_path: Path | str) -> Path:
    """Render and seal audit and compact paper figures through one workflow."""

    profile = load_campaign_profile(Path(profile_path))
    analysis_root = profile.output_root / "scientific_analysis_v1"
    analysis_manifest_path = analysis_root / "manifest.json"
    analysis_manifest = validate_generated_bundle(analysis_manifest_path)
    model_scores = _load_analysis_table(analysis_manifest, "model_scores")
    trajectory_rows = _load_analysis_table(analysis_manifest, "trajectory_dynamics")
    chance_rows = _load_analysis_table(analysis_manifest, "benchmark_chance_levels")
    benchmark_rows = _load_source_table(analysis_manifest, "benchmark_metrics")

    tables: dict[str, list[dict[str, Any]]] = {
        "scaling_plot_data": build_scaling_plot_rows(model_scores),
        "slope_summary": build_slope_summary_rows(trajectory_rows),
        "task_endpoint_summary": build_task_endpoint_rows(
            benchmark_rows, chance_rows
        ),
        "task_slope_summary": build_task_slope_rows(
            benchmark_rows, model_scores, chance_rows
        ),
        "b_seed_task_curves": build_b_seed_task_curve_rows(
            benchmark_rows, model_scores, chance_rows
        ),
    }
    figure_root = analysis_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    table_specs = {
        name: _write_rows(figure_root, TABLE_FILES[name], rows)
        for name, rows in tables.items()
    }
    summary = _build_visualization_summary(
        tables["scaling_plot_data"], trajectory_rows
    )
    summary_path = figure_root / "visualization_summary.json"
    _atomic_json(summary_path, summary)

    figures = {
        "absolute_grid": plot_scaling_grid(
            tables["scaling_plot_data"],
            figure_root=figure_root,
            stem=AUDIT_FIGURE_FILES["absolute_grid"],
            field_prefix="primary_score",
            title="Benchmark scaling across the Stage-1 grid: absolute score",
            ylabel="Equal-task macro acc_norm (%)",
        ),
        "change_grid": plot_scaling_grid(
            tables["scaling_plot_data"],
            figure_root=figure_root,
            stem=AUDIT_FIGURE_FILES["change_grid"],
            field_prefix="change_from_0p5b",
            title="Benchmark scaling across the Stage-1 grid: change from 0.5B",
            ylabel="Change from 0.5B (pp)",
        ),
        "slope_heatmap": plot_slope_heatmap(
            tables["slope_summary"],
            figure_root=figure_root,
            stem=AUDIT_FIGURE_FILES["slope_heatmap"],
        ),
        "task_signal": plot_task_signal(
            tables["task_endpoint_summary"],
            tables["task_slope_summary"],
            figure_root=figure_root,
            stem=PAPER_FIGURE_FILES["task_signal"],
        ),
        "b_seed_task_curves": plot_b_seed_task_curves(
            tables["b_seed_task_curves"],
            figure_root=figure_root,
            stem=PAPER_FIGURE_FILES["b_seed_task_curves"],
        ),
        "full_grid_compute_curves": plot_full_grid_compute_curves(
            tables["scaling_plot_data"],
            figure_root=figure_root,
            stem=PAPER_FIGURE_FILES["full_grid_compute_curves"],
        ),
        "composite_slope_map": plot_composite_slope_map(
            tables["slope_summary"],
            tables["scaling_plot_data"],
            figure_root=figure_root,
            stem=PAPER_FIGURE_FILES["composite_slope_map"],
        ),
    }

    manifest: dict[str, Any] = {
        "schema": VISUALIZATION_SCHEMA,
        "status": "complete",
        "campaign_id": EXPECTED_CAMPAIGN_ID,
        "output_root": str(analysis_root),
        "profile": {"path": str(profile.source_path), "sha256": profile.source_sha256},
        "input_analysis_manifest": {
            **_artifact_reference(analysis_manifest_path),
            "payload_sha256": analysis_manifest["manifest_sha256"],
        },
        "implementations": {
            name: _artifact_reference(path)
            for name, path in IMPLEMENTATION_FILES.items()
        },
        "tables": table_specs,
        "figures": figures,
        "visualization_summary": _artifact_reference(summary_path),
        "data_contract": {
            "model_score_rows": len(model_scores),
            "benchmark_metric_rows": len(benchmark_rows),
            "independent_trajectories": len(trajectory_rows),
            "aggregated_grid_readouts": len(tables["scaling_plot_data"]),
            "slope_grid_cells": len(tables["slope_summary"]),
            "task_endpoint_cells": len(tables["task_endpoint_summary"]),
            "task_slope_cells": len(tables["task_slope_summary"]),
            "replicated_task_curve_cells": len(tables["b_seed_task_curves"]),
            "logical_compute": "6 * active_parameters * actual_tokens",
            "replicated_cells": (
                "B/E={1,16,32}/K={1,12,200}: three-seed mean and observed min-max"
            ),
            "other_cells": "single seed 1234; descriptive only",
        },
        "claim_boundary": summary["claim_boundary"],
        "regeneration_command": (
            "PYTHONPATH=src:external/lm-evaluation-harness:. "
            "conda run -n bridge-train python -m "
            "multilingual_megatron.experiments.scaling.stage1_benchmarking.jointmoe "
            "--profile configs/benchmarks/"
            "stage1_abcd_jointmoe_all_readouts_a40_v2.json visualize"
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = analysis_root / "visualization_manifest.json"
    _atomic_json(manifest_path, manifest)
    validate_visualization_bundle(manifest_path)
    return manifest_path


__all__ = [
    "VISUALIZATION_SCHEMA",
    "_build_scaling_plot_rows",
    "_build_slope_summary_rows",
    "validate_visualization_bundle",
    "write_benchmark_scaling_visualizations",
]
