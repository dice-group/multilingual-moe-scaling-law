"""Pure table derivations for Stage-1 benchmark visualizations.

The functions in this module accept already validated rows and return plain
Python records.  They neither discover files nor write artifacts, which keeps
the scientific grain and seed aggregation independently testable.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from result_analysis.downstream_evaluations.campaign.analysis_artifacts import (
    BenchmarkAnalysisError,
    REPLICATED_B_EXPERTS,
    REPLICATED_B_LANGUAGES,
)
from result_analysis.downstream_evaluations.campaign.analysis_scores import (
    _ols_slope,
)
from multilingual_megatron.experiments.scaling.stage1_benchmarking.campaign_profile import (
    BENCHMARK_ORDER,
)


LADDERS = ("A", "B", "C", "D")
EXPERTS = (1, 4, 16, 32)
LANGUAGE_COUNTS = (1, 12, 50, 200)
EXPECTED_READOUTS_BY_LADDER = {"A": 5, "B": 6, "C": 7, "D": 7}


def _expected_seed_count(ladder: str, experts: int, languages: int) -> int:
    return (
        3
        if ladder == "B"
        and experts in REPLICATED_B_EXPERTS
        and languages in REPLICATED_B_LANGUAGES
        else 1
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise BenchmarkAnalysisError("cannot average an empty visualization cell")
    return statistics.fmean(values)


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    """Match pandas' default linear quantile for a finite one-dimensional sample."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise BenchmarkAnalysisError("cannot calculate a quantile of no values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def build_scaling_plot_rows(
    model_scores: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate grid readouts at the declared training-seed grain.

    Raw and chance-adjusted scores are derived together so audit and compact
    paper figures cannot silently implement different aggregation rules.
    """

    if len(model_scores) != 508:
        raise BenchmarkAnalysisError("visualization expects 508 model-score rows")
    trajectories: dict[tuple[str, int, int, int], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in model_scores:
        key = (
            str(row["ladder"]),
            int(row["experts"]),
            int(row["languages"]),
            int(row["seed"]),
        )
        trajectories[key].append(row)
    if len(trajectories) != 82:
        raise BenchmarkAnalysisError("visualization trajectory count drift")

    baselines: dict[tuple[str, int, int, int], tuple[float, float]] = {}
    for key, readouts in trajectories.items():
        ordered = sorted(readouts, key=lambda row: int(row["readout_index"]))
        if len(ordered) != EXPECTED_READOUTS_BY_LADDER[key[0]]:
            raise BenchmarkAnalysisError(f"visualization readout count drift for {key}")
        if int(ordered[0]["nominal_tokens"]) != 500_000_000:
            raise BenchmarkAnalysisError(f"visualization baseline drift for {key}")
        baselines[key] = (
            float(ordered[0]["primary_score"]),
            float(ordered[0]["primary_chance_adjusted_score"]),
        )

    grouped: dict[tuple[str, int, int, int], list[dict[str, float | int]]] = (
        defaultdict(list)
    )
    for row in model_scores:
        trajectory_key = (
            str(row["ladder"]),
            int(row["experts"]),
            int(row["languages"]),
            int(row["seed"]),
        )
        primary = float(row["primary_score"])
        adjusted = float(row["primary_chance_adjusted_score"])
        primary_baseline, adjusted_baseline = baselines[trajectory_key]
        active_parameters = float(row["active_parameters"])
        actual_tokens = float(row["actual_tokens"])
        grouped[
            (
                trajectory_key[0],
                trajectory_key[1],
                trajectory_key[2],
                int(row["nominal_tokens"]),
            )
        ].append(
            {
                "seed": trajectory_key[3],
                "primary": primary,
                "adjusted": adjusted,
                "primary_change": primary - primary_baseline,
                "adjusted_change_pp": 100.0 * (adjusted - adjusted_baseline),
                "active_parameters": active_parameters,
                "logical_compute_eflops": (
                    6.0 * active_parameters * actual_tokens / 1e18
                ),
            }
        )

    expected_cells = {
        (ladder, experts, languages)
        for ladder in LADDERS
        for experts in EXPERTS
        for languages in LANGUAGE_COUNTS
    }
    if {(key[0], key[1], key[2]) for key in grouped} != expected_cells:
        raise BenchmarkAnalysisError("visualization grid-cell inventory drift")

    rows: list[dict[str, Any]] = []
    for key in sorted(
        grouped,
        key=lambda value: (
            LADDERS.index(value[0]),
            LANGUAGE_COUNTS.index(value[2]),
            EXPERTS.index(value[1]),
            value[3],
        ),
    ):
        ladder, experts, languages, nominal_tokens = key
        values = sorted(grouped[key], key=lambda item: int(item["seed"]))
        expected_seeds = _expected_seed_count(ladder, experts, languages)
        if len(values) != expected_seeds:
            raise BenchmarkAnalysisError(
                f"visualization seed-count drift for {(ladder, experts, languages)}"
            )
        primary = [float(item["primary"]) for item in values]
        adjusted = [float(item["adjusted"]) for item in values]
        primary_changes = [float(item["primary_change"]) for item in values]
        adjusted_changes = [float(item["adjusted_change_pp"]) for item in values]
        active_parameters = [float(item["active_parameters"]) for item in values]
        compute = [float(item["logical_compute_eflops"]) for item in values]
        rows.append(
            {
                "ladder": ladder,
                "experts": experts,
                "languages": languages,
                "nominal_tokens": nominal_tokens,
                "seeds": [int(item["seed"]) for item in values],
                "n_seeds": len(values),
                "active_parameters": _mean(active_parameters),
                "logical_compute_eflops": _mean(compute),
                "primary_score_mean": _mean(primary),
                "primary_score_min": min(primary),
                "primary_score_max": max(primary),
                "primary_chance_adjusted_score_mean": _mean(adjusted),
                "primary_chance_adjusted_score_min": min(adjusted),
                "primary_chance_adjusted_score_max": max(adjusted),
                "change_from_0p5b_mean": _mean(primary_changes),
                "change_from_0p5b_min": min(primary_changes),
                "change_from_0p5b_max": max(primary_changes),
                "chance_adjusted_change_from_0p5b_pp_mean": _mean(
                    adjusted_changes
                ),
                "chance_adjusted_change_from_0p5b_pp_min": min(adjusted_changes),
                "chance_adjusted_change_from_0p5b_pp_max": max(adjusted_changes),
                "evidence_label": (
                    "three_seed_mean_observed_range"
                    if len(values) == 3
                    else "single_seed_descriptive"
                ),
            }
        )
    if len(rows) != 400:
        raise BenchmarkAnalysisError("visualization plot-data row count drift")
    return rows


def build_slope_summary_rows(
    trajectory_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize raw and chance-adjusted slopes without inflating n."""

    if len(trajectory_rows) != 82:
        raise BenchmarkAnalysisError("slope visualization expects 82 trajectories")
    grouped: dict[tuple[str, int, int], list[tuple[float, float]]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[
            (str(row["ladder"]), int(row["experts"]), int(row["languages"]))
        ].append(
            (
                100.0 * float(row["primary_ols_slope_per_actual_token_doubling"]),
                100.0
                * float(
                    row[
                        "primary_chance_adjusted_ols_slope_per_actual_token_doubling"
                    ]
                ),
            )
        )
    expected = {
        (ladder, experts, languages)
        for ladder in LADDERS
        for experts in EXPERTS
        for languages in LANGUAGE_COUNTS
    }
    if set(grouped) != expected:
        raise BenchmarkAnalysisError("slope visualization grid-cell inventory drift")

    rows: list[dict[str, Any]] = []
    for ladder in LADDERS:
        for languages in LANGUAGE_COUNTS:
            for experts in EXPERTS:
                values = grouped[(ladder, experts, languages)]
                expected_seeds = _expected_seed_count(ladder, experts, languages)
                if len(values) != expected_seeds:
                    raise BenchmarkAnalysisError(
                        f"slope seed-count drift for {(ladder, experts, languages)}"
                    )
                raw = [value[0] for value in values]
                adjusted = [value[1] for value in values]
                rows.append(
                    {
                        "ladder": ladder,
                        "experts": experts,
                        "languages": languages,
                        "n_trajectories": len(values),
                        "mean_slope_pp_per_actual_token_doubling": _mean(raw),
                        "min_slope_pp_per_actual_token_doubling": min(raw),
                        "max_slope_pp_per_actual_token_doubling": max(raw),
                        "mean_chance_adjusted_slope_pp_per_actual_token_doubling": _mean(
                            adjusted
                        ),
                        "min_chance_adjusted_slope_pp_per_actual_token_doubling": min(
                            adjusted
                        ),
                        "max_chance_adjusted_slope_pp_per_actual_token_doubling": max(
                            adjusted
                        ),
                        "positive_trajectories": sum(value > 0.0 for value in raw),
                        "evidence_label": (
                            "three_seed_descriptive_mean"
                            if len(values) == 3
                            else "single_seed_descriptive"
                        ),
                    }
                )
    if len(rows) != 64:
        raise BenchmarkAnalysisError("slope summary must contain 64 grid cells")
    return rows


def _model_fields(
    model_scores: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[float, float]]:
    fields: dict[str, tuple[float, float]] = {}
    for row in model_scores:
        model_id = str(row["model_id"])
        value = (float(row["active_parameters"]), float(row["actual_tokens"]))
        previous = fields.setdefault(model_id, value)
        if previous != value:
            raise BenchmarkAnalysisError(f"model-field drift for {model_id}")
    if len(fields) != 508:
        raise BenchmarkAnalysisError("model-field inventory drift")
    return fields


def _chance_by_task(
    chance_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    chance = {
        str(row["benchmark"]): float(row["random_choice_accuracy"])
        for row in chance_rows
    }
    if tuple(chance) != BENCHMARK_ORDER:
        raise BenchmarkAnalysisError("benchmark chance-level order drift")
    return chance


def build_task_endpoint_rows(
    benchmark_rows: Sequence[Mapping[str, Any]],
    chance_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize final task signal across the 16 canonical cells per K."""

    chance = _chance_by_task(chance_rows)
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    selected = 0
    for row in benchmark_rows:
        if str(row["source_cohort"]) != "canonical_release" or not bool(
            row["is_final_readout"]
        ):
            continue
        benchmark = str(row["benchmark"])
        languages = int(row["languages"])
        grouped[(benchmark, languages)].append(
            100.0 * (float(row["macro_acc_norm"]) - chance[benchmark])
        )
        selected += 1
    if selected != 64 * len(BENCHMARK_ORDER):
        raise BenchmarkAnalysisError("unexpected canonical final-task support")

    rows: list[dict[str, Any]] = []
    for benchmark in BENCHMARK_ORDER:
        for languages in LANGUAGE_COUNTS:
            values = grouped[(benchmark, languages)]
            if len(values) != 16:
                raise BenchmarkAnalysisError(
                    f"final-task cell drift for {(benchmark, languages)}"
                )
            rows.append(
                {
                    "benchmark": benchmark,
                    "K": languages,
                    "random_choice_accuracy": chance[benchmark],
                    "n": len(values),
                    "mean": _mean(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
            )
    return rows


def build_task_slope_rows(
    benchmark_rows: Sequence[Mapping[str, Any]],
    model_scores: Sequence[Mapping[str, Any]],
    chance_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize task-level within-trajectory gains per compute doubling."""

    model_fields = _model_fields(model_scores)
    chance = _chance_by_task(chance_rows)
    grouped: dict[tuple[str, int, str], list[tuple[float, float]]] = defaultdict(
        list
    )
    for row in benchmark_rows:
        if str(row["source_cohort"]) != "canonical_release":
            continue
        benchmark = str(row["benchmark"])
        active_parameters, actual_tokens = model_fields[str(row["model_id"])]
        compute = 6.0 * active_parameters * actual_tokens / 1e18
        margin = 100.0 * (float(row["macro_acc_norm"]) - chance[benchmark])
        grouped[
            (str(row["trajectory_id"]), int(row["languages"]), benchmark)
        ].append((math.log2(compute), margin))
    if len(grouped) != 64 * len(BENCHMARK_ORDER):
        raise BenchmarkAnalysisError("unexpected task-trajectory support")

    slopes_by_cell: dict[tuple[int, str], list[float]] = defaultdict(list)
    for (_trajectory_id, languages, benchmark), values in grouped.items():
        ordered = sorted(values)
        slopes_by_cell[(languages, benchmark)].append(
            _ols_slope(
                [value[0] for value in ordered],
                [value[1] for value in ordered],
            )
        )

    rows: list[dict[str, Any]] = []
    for benchmark in BENCHMARK_ORDER:
        for languages in LANGUAGE_COUNTS:
            values = slopes_by_cell[(languages, benchmark)]
            if len(values) != 16:
                raise BenchmarkAnalysisError(
                    f"task-slope cell drift for {(benchmark, languages)}"
                )
            rows.append(
                {
                    "benchmark": benchmark,
                    "K": languages,
                    "n": len(values),
                    "positive": sum(value > 0.0 for value in values),
                    "median": statistics.median(values),
                    "q25": _linear_quantile(values, 0.25),
                    "q75": _linear_quantile(values, 0.75),
                }
            )
    return rows


def build_b_seed_task_curve_rows(
    benchmark_rows: Sequence[Mapping[str, Any]],
    model_scores: Sequence[Mapping[str, Any]],
    chance_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate bounded B/E1,E32 task curves at the training-seed grain."""

    model_fields = _model_fields(model_scores)
    chance = _chance_by_task(chance_rows)
    grouped: dict[tuple[str, int, int, int], list[tuple[int, float, float]]] = (
        defaultdict(list)
    )
    for row in benchmark_rows:
        experts = int(row["experts"])
        languages = int(row["languages"])
        if (
            str(row["ladder"]) != "B"
            or experts not in (1, 32)
            or languages not in (1, 12, 200)
        ):
            continue
        benchmark = str(row["benchmark"])
        active_parameters, actual_tokens = model_fields[str(row["model_id"])]
        compute = 6.0 * active_parameters * actual_tokens / 1e18
        margin = 100.0 * (float(row["macro_acc_norm"]) - chance[benchmark])
        grouped[(benchmark, languages, experts, int(row["nominal_tokens"]))].append(
            (int(row["seed"]), compute, margin)
        )

    rows: list[dict[str, Any]] = []
    for key in sorted(
        grouped,
        key=lambda value: (
            BENCHMARK_ORDER.index(value[0]),
            LANGUAGE_COUNTS.index(value[1]),
            EXPERTS.index(value[2]),
            value[3],
        ),
    ):
        benchmark, languages, experts, nominal_tokens = key
        values = grouped[key]
        seeds = {value[0] for value in values}
        if len(values) != 3 or len(seeds) != 3:
            raise BenchmarkAnalysisError(f"B task-curve seed drift for {key}")
        compute = [value[1] for value in values]
        margins = [value[2] for value in values]
        rows.append(
            {
                "benchmark": benchmark,
                "K": languages,
                "E": experts,
                "nominal_tokens": nominal_tokens,
                "n_seeds": len(seeds),
                "compute_eflops": _mean(compute),
                "mean": _mean(margins),
                "minimum": min(margins),
                "maximum": max(margins),
            }
        )
    if len(rows) != len(BENCHMARK_ORDER) * 3 * 2 * 6:
        raise BenchmarkAnalysisError("unexpected replicated benchmark-curve support")
    return rows


__all__ = [
    "EXPECTED_READOUTS_BY_LADDER",
    "EXPERTS",
    "LADDERS",
    "LANGUAGE_COUNTS",
    "build_b_seed_task_curve_rows",
    "build_scaling_plot_rows",
    "build_slope_summary_rows",
    "build_task_endpoint_rows",
    "build_task_slope_rows",
]
