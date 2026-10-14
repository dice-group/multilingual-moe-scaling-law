"""Human-readable reports for the Stage-1 benchmark analysis bundle."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.3f}%"


def _pp(value: Any) -> str:
    return f"{100.0 * float(value):+.3f} pp"


def _sd_pp(value: Any) -> str:
    return f"{100.0 * float(value):.3f} pp"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _header in headers) + " |",
        *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
    ]


def _render_initial_analysis(
    findings: Mapping[str, Any],
    *,
    threshold_rows: Sequence[Mapping[str, Any]],
) -> str:
    a_rows = findings["a_ladder_8b"]
    a_table = _markdown_table(
        ("Experts", "Languages", "8B primary", "0.5B to 8B", "OLS/doubling"),
        [
            (
                row["experts"],
                row["languages"],
                _pct(row["final_primary_score"]),
                _pp(row["endpoint_delta"]),
                _pp(row["slope_per_actual_token_doubling"]),
            )
            for row in a_rows
        ],
    )
    e32_rows = findings["canonical_e32_final_by_ladder"]
    e32_table = _markdown_table(
        ("Ladder", "Tokens", "K1", "K12", "K50", "K200"),
        [
            (
                row["ladder"],
                f"{float(row['final_tokens']) / 1e9:g}B",
                _pct(row["k1_primary_score"]),
                _pct(row["k12_primary_score"]),
                _pct(row["k50_primary_score"]),
                _pct(row["k200_primary_score"]),
            )
            for row in e32_rows
        ],
    )

    final_contrasts = findings["b_three_seed_final_contrasts"]
    e_contrasts = sorted(
        (
            row
            for row in final_contrasts
            if row["contrast_family"] == "experts_e32_minus_e1"
        ),
        key=lambda row: int(row["languages"]),
    )
    expert_table = _markdown_table(
        ("Languages", "Mean E32-E1", "Sample SD", "Range", "Seed directions"),
        [
            (
                f"K{row['languages']}",
                _pp(row["mean_estimate"]),
                _sd_pp(row["sample_sd"]),
                f"{_pp(row['min_estimate'])} to {_pp(row['max_estimate'])}",
                (
                    f"+{row['positive_seeds']} / -{row['negative_seeds']} / "
                    f"0={row['zero_seeds']}"
                ),
            )
            for row in e_contrasts
        ],
    )
    k12_contrasts = sorted(
        (
            row
            for row in final_contrasts
            if row["contrast_family"] == "languages_k200_minus_k12"
        ),
        key=lambda row: int(row["experts"]),
    )
    k1_contrasts = sorted(
        (
            row
            for row in final_contrasts
            if row["contrast_family"]
            == "languages_k200_minus_k1_common_english"
        ),
        key=lambda row: int(row["experts"]),
    )
    language_table = _markdown_table(
        ("Experts", "K200-K12 multilingual", "K200-K1 common English"),
        [
            (
                f"E{k12['experts']}",
                f"{_pp(k12['mean_estimate'])} (SD {_sd_pp(k12['sample_sd'])})",
                f"{_pp(k1['mean_estimate'])} (SD {_sd_pp(k1['sample_sd'])})",
            )
            for k12, k1 in zip(k12_contrasts, k1_contrasts, strict=True)
        ],
    )

    primary_zero = [
        row
        for row in threshold_rows
        if row["score_scope"] == "primary_six_task" and row["threshold"] == 0.0
    ]
    zero_status = Counter(str(row["status"]) for row in primary_zero)
    cost = findings["observed_cost"]
    direction_totals = findings["b_direction_totals_across_all_six_horizons"]
    k1_experts = direction_totals["e32_minus_e1"]["K1"]
    k12_experts = direction_totals["e32_minus_e1"]["K12"]
    k200_experts = direction_totals["e32_minus_e1"]["K200"]
    common_cost_e16 = direction_totals["k200_minus_k1_common_english"]["E16"]
    common_cost_e32 = direction_totals["k200_minus_k1_common_english"]["E32"]

    lines = [
        "# Stage-1 JointMoE all-readout benchmark: initial analysis",
        "",
        "Status: complete, sealed, and exploratory except for the bounded three-seed B robustness panel.",
        "",
        "## Evidence and estimand",
        "",
        (
            "All 508 checkpoint jobs completed and map to 82 independently trained "
            "trajectories. The primary score is the equal-weight mean of six benchmark "
            "macros after equal weighting of languages within each benchmark. The "
            "task-equal random-choice baseline derived from the frozen requests is "
            f"{_pct(findings['metric_contract']['primary_random_choice_baseline'])}."
        ),
        "",
        (
            "K1 uses an English-only request set; K>=12 uses the multilingual request "
            "set. Their six-task composites are therefore not a cross-K estimand. "
            "K200-vs-K1 results below use only the exact four-task common-English slice."
        ),
        "",
        (
            f"Measured execution cost was {float(cost['gpu_hours']):.2f} A40 GPU-hours "
            f"across {cost['jobs']} jobs ({float(cost['serial_job_hours']):.2f} serial "
            "job-hours); queue time is excluded."
        ),
        "",
        "## 1. The A ladder shows substantial English movement by 8B, but weak multilingual movement",
        "",
        *a_table,
        "",
        (
            "At K1, E16 and E32 gain materially from the first 0.5B readout to 8B; "
            "the K>=12 endpoint changes are much smaller and not uniformly positive. "
            "Because A has one seed per cell, this is evidence of observed learning-curve "
            "movement, not seed-general capability."
        ),
        "",
        "## 2. Larger canonical models improve K1 more clearly than multilingual macros",
        "",
        *e32_table,
        "",
        (
            "Within each K column the frozen requests match, but ladder changes jointly "
            "alter active parameters and terminal token budget. This table is a descriptive "
            "scale trajectory, not a causal parameter-only or token-only contrast."
        ),
        "",
        "## 3. B-seed confirmation shows a recurring English E32 advantage, not a stable multilingual one",
        "",
        *expert_table,
        "",
        (
            "Across all six B horizons and three seeds, E32-E1 is positive in "
            f"{k1_experts['positive']}/{k1_experts['comparisons']} K1 comparisons, "
            f"{k12_experts['positive']}/{k12_experts['comparisons']} K12 comparisons, "
            f"and {k200_experts['positive']}/{k200_experts['comparisons']} K200 comparisons. "
            "The final multilingual means are small relative to their across-seed SDs and "
            "their seed-wise directions are not stable."
        ),
        "",
        "## 4. More languages show an English trade-off, without a consistent K200 macro gain",
        "",
        *language_table,
        "",
        (
            "For the common-English K200-K1 contrast, all 18 E16 and all 18 E32 "
            f"seed/horizon comparisons are negative ({common_cost_e16['negative']}/18 "
            f"and {common_cost_e32['negative']}/18). This is a consistent observed "
            "trade-off under the coupled corpus-mixture/K intervention, not proof that "
            "language count alone caused it. K200-vs-K12 multilingual differences have "
            "mixed signs and no stable overall advantage."
        ),
        "",
        "## 5. The campaign does not identify a unique capability-onset token count",
        "",
        (
            f"At the weakest post-hoc threshold (chance-adjusted score >= 0), "
            f"{zero_status['already_at_or_above_at_first_readout']}/{len(primary_zero)} "
            "trajectories are already at or above threshold at the first 0.5B readout. "
            "Those onsets are left-censored. Higher thresholds are retained as a "
            "sensitivity grid with discrete crossing brackets, not interpolated onset "
            "estimates."
        ),
        "",
        "## Claim boundary",
        "",
        *[f"- {item}" for item in findings["claim_boundary"]],
        "",
        "## Next analysis",
        "",
        (
            "Use trajectory-clustered longitudinal models or seed-wise curve summaries "
            "for formal scaling analysis. Per-task and per-language decomposition should "
            "be treated as nested diagnostics, with multiplicity control if used for "
            "confirmatory claims."
        ),
        "",
    ]
    return "\n".join(lines)


def _render_readme() -> str:
    rows = [
        ("benchmark_chance_levels", "Choice counts and chance levels derived from requests"),
        ("common_english_slice", "Exact cross-request-set English intersection"),
        ("model_scores", "Checkpoint-level composites; nested descriptive rows"),
        ("trajectory_dynamics", "One learning-curve summary per training trajectory"),
        ("threshold_crossings", "Post-hoc discrete sensitivity thresholds"),
        ("final_endpoints", "One terminal checkpoint per training trajectory"),
        ("canonical_final_grid", "The 64 seed-1234 A-D endpoint cells"),
        ("b_seed_cell_summary", "Mean, sample SD, and range over three B seeds"),
        ("b_seed_contrasts_seedwise", "Matched seed-wise B contrasts"),
        ("b_seed_contrast_summary", "Three-seed summaries and direction stability"),
    ]
    lines = [
        "# Scientific analysis bundle",
        "",
        "This directory is generated from hash-validated campaign artifacts. Do not edit it by hand.",
        "",
        "The statistical unit is one independent training seed within a fixed design cell. Checkpoints, tasks, languages, and examples are nested observations.",
        "",
        "## Files",
        "",
        "- `manifest.json`: hashes, lineage, table grains, and claim policy.",
        "- `integrity_summary.json`: completed-cohort and source-validation checks.",
        "- `initial_findings.json`: machine-readable first-pass findings.",
        "- `initial_analysis.md`: human-readable bounded interpretation.",
        "- `tables/*.csv` and `tables/*.jsonl`: paired analysis tables.",
        "",
        "| Table | Purpose |",
        "| --- | --- |",
        *[f"| `{name}` | {purpose} |" for name, purpose in rows],
        "",
        "Regenerate with the documented `stage1_jointmoe_benchmark analyze` command. The command fails closed if a source hash, row count, request choice count, common-language intersection, or seed panel drifts.",
        "",
    ]
    return "\n".join(lines)


__all__ = ["_render_initial_analysis", "_render_readme"]
