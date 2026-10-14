"""Three-seed contrasts and bounded findings for Stage-1 benchmarks."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from result_analysis.downstream_evaluations.campaign.analysis_artifacts import (
    EXPECTED_CAMPAIGN_ID,
    EXPECTED_TRAJECTORIES,
    FINDINGS_SCHEMA,
    POST_HOC_THRESHOLDS,
    REPLICATED_B_EXPERTS,
    REPLICATED_B_LANGUAGES,
    REPLICATED_B_SEEDS,
    REPLICATED_B_TOKENS,
    BenchmarkAnalysisError,
)
from result_analysis.downstream_evaluations.campaign.analysis_scores import (
    _direction,
    _mean,
    _sample_summary,
)


def _replicated_b_rows(
    model_scores: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in model_scores
        if row["ladder"] == "B"
        and int(row["experts"]) in REPLICATED_B_EXPERTS
        and int(row["languages"]) in REPLICATED_B_LANGUAGES
        and int(row["seed"]) in REPLICATED_B_SEEDS
    ]


def _build_b_seed_cell_summary(
    model_scores: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in _replicated_b_rows(model_scores):
        grouped[
            (
                int(row["experts"]),
                int(row["languages"]),
                int(row["nominal_tokens"]),
            )
        ].append(row)
    expected_keys = {
        (experts, languages, tokens)
        for experts in REPLICATED_B_EXPERTS
        for languages in REPLICATED_B_LANGUAGES
        for tokens in REPLICATED_B_TOKENS
    }
    if set(grouped) != expected_keys:
        raise BenchmarkAnalysisError("B seed-summary cell inventory drift")

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        experts, languages, tokens = key
        members = sorted(grouped[key], key=lambda row: int(row["seed"]))
        if tuple(int(row["seed"]) for row in members) != REPLICATED_B_SEEDS:
            raise BenchmarkAnalysisError(f"B seed identity drift for cell {key}")
        primary = _sample_summary([float(row["primary_score"]) for row in members])
        primary_adjusted = _sample_summary(
            [float(row["primary_chance_adjusted_score"]) for row in members]
        )
        common = _sample_summary(
            [float(row["common_english_score"]) for row in members]
        )
        common_adjusted = _sample_summary(
            [float(row["common_english_chance_adjusted_score"]) for row in members]
        )
        rows.append(
            {
                "ladder": "B",
                "experts": experts,
                "languages": languages,
                "nominal_tokens": tokens,
                "seeds": list(REPLICATED_B_SEEDS),
                "n_seeds": primary["n_seeds"],
                "primary_mean": primary["mean"],
                "primary_sample_sd": primary["sample_sd"],
                "primary_min": primary["min"],
                "primary_max": primary["max"],
                "primary_range": primary["range"],
                "primary_chance_adjusted_mean": primary_adjusted["mean"],
                "primary_chance_adjusted_sample_sd": primary_adjusted["sample_sd"],
                "common_english_mean": common["mean"],
                "common_english_sample_sd": common["sample_sd"],
                "common_english_min": common["min"],
                "common_english_max": common["max"],
                "common_english_range": common["range"],
                "common_english_chance_adjusted_mean": common_adjusted["mean"],
                "common_english_chance_adjusted_sample_sd": common_adjusted[
                    "sample_sd"
                ],
                "evidence_label": "three_seed_descriptive_no_significance_test",
            }
        )
    if len(rows) != 54:
        raise BenchmarkAnalysisError("B seed-summary table must contain 54 cells")
    return rows


def _contrast_row(
    *,
    family: str,
    metric: str,
    comparison_scope: str,
    experts: int | str,
    languages: int | str,
    nominal_tokens: int,
    seed: int,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    score_field: str,
) -> dict[str, Any]:
    estimate = float(left[score_field]) - float(right[score_field])
    return {
        "contrast_family": family,
        "metric": metric,
        "comparison_scope": comparison_scope,
        "experts": experts,
        "languages": languages,
        "nominal_tokens": nominal_tokens,
        "seed": seed,
        "left_model_id": left["model_id"],
        "right_model_id": right["model_id"],
        "left_score": left[score_field],
        "right_score": right[score_field],
        "estimate_left_minus_right": estimate,
        "direction": _direction(estimate),
        "evidence_label": "matched_seed_descriptive",
    }


def _build_b_seed_contrasts(
    model_scores: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    panel = {
        (
            int(row["experts"]),
            int(row["languages"]),
            int(row["nominal_tokens"]),
            int(row["seed"]),
        ): row
        for row in _replicated_b_rows(model_scores)
    }
    if len(panel) != 162:
        raise BenchmarkAnalysisError("replicated B contrast panel must contain 162 rows")

    seedwise: list[dict[str, Any]] = []
    for languages in REPLICATED_B_LANGUAGES:
        for tokens in REPLICATED_B_TOKENS:
            for seed in REPLICATED_B_SEEDS:
                seedwise.append(
                    _contrast_row(
                        family="experts_e32_minus_e1",
                        metric="equal_task_macro_acc_norm",
                        comparison_scope=(
                            "same frozen English six-task requests"
                            if languages == 1
                            else "same frozen multilingual six-task requests"
                        ),
                        experts="32-1",
                        languages=languages,
                        nominal_tokens=tokens,
                        seed=seed,
                        left=panel[(32, languages, tokens, seed)],
                        right=panel[(1, languages, tokens, seed)],
                        score_field="primary_score",
                    )
                )
    for experts in REPLICATED_B_EXPERTS:
        for tokens in REPLICATED_B_TOKENS:
            for seed in REPLICATED_B_SEEDS:
                seedwise.append(
                    _contrast_row(
                        family="languages_k200_minus_k12",
                        metric="equal_task_macro_acc_norm",
                        comparison_scope="same frozen multilingual six-task requests",
                        experts=experts,
                        languages="200-12",
                        nominal_tokens=tokens,
                        seed=seed,
                        left=panel[(experts, 200, tokens, seed)],
                        right=panel[(experts, 12, tokens, seed)],
                        score_field="primary_score",
                    )
                )
                seedwise.append(
                    _contrast_row(
                        family="languages_k200_minus_k1_common_english",
                        metric="common_english_four_task_acc_norm",
                        comparison_scope="exact four-task common-English intersection",
                        experts=experts,
                        languages="200-1",
                        nominal_tokens=tokens,
                        seed=seed,
                        left=panel[(experts, 200, tokens, seed)],
                        right=panel[(experts, 1, tokens, seed)],
                        score_field="common_english_score",
                    )
                )
    if len(seedwise) != 162:
        raise BenchmarkAnalysisError("B seed-wise contrast table must contain 162 rows")

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seedwise:
        grouped[
            (
                row["contrast_family"],
                row["metric"],
                row["comparison_scope"],
                row["experts"],
                row["languages"],
                row["nominal_tokens"],
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        members = sorted(grouped[key], key=lambda row: int(row["seed"]))
        if tuple(int(row["seed"]) for row in members) != REPLICATED_B_SEEDS:
            raise BenchmarkAnalysisError(f"contrast seed identity drift for {key}")
        values = [float(row["estimate_left_minus_right"]) for row in members]
        summary = _sample_summary(values)
        directions = Counter(_direction(value) for value in values)
        nonzero_directions = {
            direction for direction in ("positive", "negative") if directions[direction]
        }
        stable = (
            next(iter(nonzero_directions))
            if len(nonzero_directions) == 1 and directions["zero"] == 0
            else "mixed_or_tied"
        )
        summaries.append(
            {
                "contrast_family": key[0],
                "metric": key[1],
                "comparison_scope": key[2],
                "experts": key[3],
                "languages": key[4],
                "nominal_tokens": key[5],
                "seeds": list(REPLICATED_B_SEEDS),
                "n_seeds": summary["n_seeds"],
                "mean_estimate": summary["mean"],
                "sample_sd": summary["sample_sd"],
                "min_estimate": summary["min"],
                "max_estimate": summary["max"],
                "range": summary["range"],
                "positive_seeds": directions["positive"],
                "negative_seeds": directions["negative"],
                "zero_seeds": directions["zero"],
                "stable_direction_across_all_three_seeds": stable,
                "evidence_label": "three_seed_descriptive_no_significance_test",
            }
        )
    if len(summaries) != 54:
        raise BenchmarkAnalysisError("B contrast-summary table must contain 54 rows")
    return seedwise, summaries


def _endpoint_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    result = {
        (str(row["ladder"]), int(row["experts"]), int(row["languages"])): row
        for row in rows
    }
    if len(result) != len(rows):
        raise BenchmarkAnalysisError("canonical endpoint lookup is not unique")
    return result


def _direction_totals(
    rows: Sequence[Mapping[str, Any]], *, family: str, selector: Mapping[str, Any]
) -> dict[str, int]:
    selected = [
        row
        for row in rows
        if row["contrast_family"] == family
        and all(row[key] == value for key, value in selector.items())
    ]
    directions = Counter(str(row["direction"]) for row in selected)
    return {
        "comparisons": len(selected),
        "positive": directions["positive"],
        "negative": directions["negative"],
        "zero": directions["zero"],
    }


def _build_initial_findings(
    *,
    canonical_final: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    threshold_rows: Sequence[Mapping[str, Any]],
    b_cell_rows: Sequence[Mapping[str, Any]],
    b_seedwise: Sequence[Mapping[str, Any]],
    b_summaries: Sequence[Mapping[str, Any]],
    chance_rows: Sequence[Mapping[str, Any]],
    model_scores: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    endpoints = _endpoint_lookup(canonical_final)
    a_dynamics = {
        (int(row["experts"]), int(row["languages"])): row
        for row in trajectory_rows
        if row["ladder"] == "A" and int(row["seed"]) == 1234
    }
    a_rows: list[dict[str, Any]] = []
    for experts in (1, 4, 16, 32):
        for languages in (1, 12, 50, 200):
            endpoint = endpoints[("A", experts, languages)]
            dynamics = a_dynamics[(experts, languages)]
            a_rows.append(
                {
                    "experts": experts,
                    "languages": languages,
                    "final_tokens": endpoint["nominal_tokens"],
                    "final_primary_score": endpoint["primary_score"],
                    "endpoint_delta": dynamics["primary_endpoint_delta"],
                    "slope_per_actual_token_doubling": dynamics[
                        "primary_ols_slope_per_actual_token_doubling"
                    ],
                }
            )

    e32_ladder_rows = [
        {
            "ladder": ladder,
            "final_tokens": endpoints[(ladder, 32, 1)]["nominal_tokens"],
            "k1_primary_score": endpoints[(ladder, 32, 1)]["primary_score"],
            "k12_primary_score": endpoints[(ladder, 32, 12)]["primary_score"],
            "k50_primary_score": endpoints[(ladder, 32, 50)]["primary_score"],
            "k200_primary_score": endpoints[(ladder, 32, 200)]["primary_score"],
        }
        for ladder in ("A", "B", "C", "D")
    ]

    b_final_cells = [
        dict(row)
        for row in b_cell_rows
        if int(row["nominal_tokens"]) == 16_000_000_000
    ]
    b_final_contrasts = [
        dict(row)
        for row in b_summaries
        if int(row["nominal_tokens"]) == 16_000_000_000
    ]
    onset_counts = Counter(str(row["status"]) for row in threshold_rows)
    score_values = [float(row["primary_score"]) for row in model_scores]
    actual_gpu_hours = sum(float(row["gpu_hours"]) for row in model_scores)
    actual_serial_job_hours = sum(
        float(row["job_wall_seconds"]) / 3600.0 for row in model_scores
    )

    return {
        "schema": FINDINGS_SCHEMA,
        "status": "initial_descriptive_analysis_complete",
        "campaign_id": EXPECTED_CAMPAIGN_ID,
        "evidence_scope": {
            "independent_training_trajectories": EXPECTED_TRAJECTORIES,
            "nested_checkpoint_readouts": 508,
            "three_seed_scope": "B/E={1,16,32}/K={1,12,200}",
            "all_other_grid_cells": "single-seed exploratory",
        },
        "metric_contract": {
            "primary": "equal task weight after equal-language macro acc_norm",
            "primary_random_choice_baseline": _mean(
                float(row["random_choice_accuracy"]) for row in chance_rows
            ),
            "common_english": "four exact benchmark-language intersections",
            "thresholds": list(POST_HOC_THRESHOLDS),
            "threshold_status": "post-hoc descriptive sensitivity only",
        },
        "observed_cost": {
            "jobs": 508,
            "serial_job_hours": actual_serial_job_hours,
            "gpu_hours": actual_gpu_hours,
        },
        "score_range": {"min": min(score_values), "max": max(score_values)},
        "a_ladder_8b": a_rows,
        "canonical_e32_final_by_ladder": e32_ladder_rows,
        "b_three_seed_final_cells": b_final_cells,
        "b_three_seed_final_contrasts": b_final_contrasts,
        "b_direction_totals_across_all_six_horizons": {
            "e32_minus_e1": {
                f"K{languages}": _direction_totals(
                    b_seedwise,
                    family="experts_e32_minus_e1",
                    selector={"languages": languages},
                )
                for languages in REPLICATED_B_LANGUAGES
            },
            "k200_minus_k12": {
                f"E{experts}": _direction_totals(
                    b_seedwise,
                    family="languages_k200_minus_k12",
                    selector={"experts": experts},
                )
                for experts in REPLICATED_B_EXPERTS
            },
            "k200_minus_k1_common_english": {
                f"E{experts}": _direction_totals(
                    b_seedwise,
                    family="languages_k200_minus_k1_common_english",
                    selector={"experts": experts},
                )
                for experts in REPLICATED_B_EXPERTS
            },
        },
        "threshold_status_counts": dict(sorted(onset_counts.items())),
        "claim_boundary": [
            "No p-values or confidence intervals are reported from n=3 seeds.",
            "Checkpoint, task, language, and example rows are not independent replicates.",
            "K1 and K>=12 six-task composites use different request sets and are not compared directly.",
            "K200-vs-K1 uses only the exact four-task common-English intersection.",
            "Threshold crossings are discrete, post-hoc descriptions, not capability-onset estimates.",
            "No pretraining-corpus benchmark-contamination audit is available.",
        ],
    }




__all__ = [
    "_build_b_seed_cell_summary",
    "_build_b_seed_contrasts",
    "_build_initial_findings",
    "_contrast_row",
    "_direction_totals",
    "_endpoint_lookup",
    "_replicated_b_rows",
]
