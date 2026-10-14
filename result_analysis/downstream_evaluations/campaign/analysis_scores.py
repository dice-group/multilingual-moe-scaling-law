"""Core score and learning-curve derivations for Stage-1 benchmarks."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from itertools import chain
from typing import Any, Iterable, Mapping, Sequence

from multilingual_megatron.contracts.hashing import canonical_json_sha256
from result_analysis.downstream_evaluations.campaign.analysis_artifacts import (
    COMMON_ENGLISH_SLICE,
    EXPECTED_FINAL_READOUTS,
    EXPECTED_LADDERS,
    EXPECTED_SOURCE_COHORTS,
    EXPECTED_TRAJECTORIES,
    FLOAT_TOLERANCE,
    POST_HOC_THRESHOLDS,
    REPLICATED_B_EXPERTS,
    REPLICATED_B_LANGUAGES,
    REPLICATED_B_SEEDS,
    REPLICATED_B_TOKENS,
    BenchmarkAnalysisError,
)
from multilingual_megatron.experiments.scaling.stage1_benchmarking.campaign_profile import (
    BENCHMARK_ORDER,
)


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(float(value) for value in values)
    if not materialized:
        raise BenchmarkAnalysisError("cannot average an empty collection")
    return sum(materialized) / len(materialized)


def _sample_summary(values: Sequence[float]) -> dict[str, Any]:
    materialized = [float(value) for value in values]
    if len(materialized) < 2:
        raise BenchmarkAnalysisError("sample summaries require at least two seeds")
    return {
        "n_seeds": len(materialized),
        "mean": statistics.fmean(materialized),
        "sample_sd": statistics.stdev(materialized),
        "min": min(materialized),
        "max": max(materialized),
        "range": max(materialized) - min(materialized),
    }


def _ols_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise BenchmarkAnalysisError("OLS slope requires aligned repeated readouts")
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator <= 0:
        raise BenchmarkAnalysisError("OLS slope has no token variation")
    return sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    ) / denominator


def _direction(value: float) -> str:
    if value > FLOAT_TOLERANCE:
        return "positive"
    if value < -FLOAT_TOLERANCE:
        return "negative"
    return "zero"


def _chance_adjusted(score: float, chance: float) -> float:
    if not 0.0 <= chance < 1.0:
        raise BenchmarkAnalysisError(f"invalid chance level: {chance}")
    return (float(score) - chance) / (1.0 - chance)


def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "model_id",
            "trajectory_id",
            "readout_id",
            "ladder",
            "experts",
            "languages",
            "seed",
            "source_cohort",
            "replication_role",
            "request_set",
            "nominal_tokens",
            "actual_tokens",
            "checkpoint_step",
            "readout_index",
            "readout_count",
            "is_final_readout",
            "active_parameters",
            "configured_total_parameters",
            "peak_lr",
            "world_size",
            "slurm_job_id",
            "job_wall_seconds",
            "gpu_count",
            "gpu_hours",
        )
    }



def _derive_chance_levels(
    manifests: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Derive random-choice baselines from the frozen request multiplicities."""

    by_request_set: dict[str, dict[str, tuple[int, int, int]]] = {}
    for request_set, manifest in manifests.items():
        requests = manifest.get("requests")
        if not isinstance(requests, list) or not requests:
            raise BenchmarkAnalysisError(f"{request_set} request list is empty")
        choices_by_doc: dict[tuple[str, str, int], set[int]] = defaultdict(set)
        for request in requests:
            if not isinstance(request, dict):
                raise BenchmarkAnalysisError(f"{request_set} contains a non-object request")
            try:
                key = (
                    str(request["benchmark"]),
                    str(request["task"]),
                    int(request["doc_id"]),
                )
                choice_index = int(request["choice_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise BenchmarkAnalysisError(
                    f"invalid request identity in {request_set}"
                ) from exc
            if choice_index in choices_by_doc[key]:
                raise BenchmarkAnalysisError(
                    f"duplicate choice index for frozen document {key}"
                )
            choices_by_doc[key].add(choice_index)

        counts_by_benchmark: dict[str, set[int]] = defaultdict(set)
        docs_by_benchmark: Counter[str] = Counter()
        requests_by_benchmark: Counter[str] = Counter()
        for (benchmark, _task, _doc_id), indices in choices_by_doc.items():
            if indices != set(range(len(indices))):
                raise BenchmarkAnalysisError(
                    f"non-contiguous choice indices for {benchmark} frozen document"
                )
            counts_by_benchmark[benchmark].add(len(indices))
            docs_by_benchmark[benchmark] += 1
            requests_by_benchmark[benchmark] += len(indices)
        if set(counts_by_benchmark) != set(BENCHMARK_ORDER):
            raise BenchmarkAnalysisError(f"{request_set} benchmark inventory drift")
        derived: dict[str, tuple[int, int, int]] = {}
        for benchmark in BENCHMARK_ORDER:
            choice_counts = counts_by_benchmark[benchmark]
            if len(choice_counts) != 1:
                raise BenchmarkAnalysisError(
                    f"{request_set}/{benchmark} has variable choice counts"
                )
            derived[benchmark] = (
                next(iter(choice_counts)),
                docs_by_benchmark[benchmark],
                requests_by_benchmark[benchmark],
            )
        by_request_set[request_set] = derived

    rows: list[dict[str, Any]] = []
    for benchmark in BENCHMARK_ORDER:
        english = by_request_set["english"][benchmark]
        multilingual = by_request_set["multilingual"][benchmark]
        if english[0] != multilingual[0]:
            raise BenchmarkAnalysisError(
                f"choice-count drift across request sets for {benchmark}"
            )
        rows.append(
            {
                "benchmark": benchmark,
                "choice_count": english[0],
                "random_choice_accuracy": 1.0 / english[0],
                "english_documents": english[1],
                "english_requests": english[2],
                "multilingual_documents": multilingual[1],
                "multilingual_requests": multilingual[2],
                "derivation": "frozen request count per task/document",
            }
        )
    expected_choices = {
        "belebele_cloze": 4,
        "okapi_hellaswag": 4,
        "xstorycloze": 2,
        "xcopa": 2,
        "uhura_arc_easy": 4,
        "global_mmlu_cloze": 4,
    }
    if {row["benchmark"]: row["choice_count"] for row in rows} != expected_choices:
        raise BenchmarkAnalysisError("frozen benchmark choice-count contract drift")
    return rows


def _validate_campaign_grains(
    model_rows: Sequence[Mapping[str, Any]],
    benchmark_rows: Sequence[Mapping[str, Any]],
    language_rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> None:
    model_ids = [str(row["model_id"]) for row in model_rows]
    if len(set(model_ids)) != 508:
        raise BenchmarkAnalysisError("model-readout IDs are not unique")
    if Counter(str(row["source_cohort"]) for row in model_rows) != Counter(
        EXPECTED_SOURCE_COHORTS
    ):
        raise BenchmarkAnalysisError("source-cohort counts drifted")
    if Counter(str(row["ladder"]) for row in model_rows) != Counter(EXPECTED_LADDERS):
        raise BenchmarkAnalysisError("ladder readout counts drifted")
    if len({str(row["trajectory_id"]) for row in model_rows}) != EXPECTED_TRAJECTORIES:
        raise BenchmarkAnalysisError("independent trajectory count drifted")
    if sum(bool(row["is_final_readout"]) for row in model_rows) != EXPECTED_FINAL_READOUTS:
        raise BenchmarkAnalysisError("final-readout count drifted")

    report_models = report.get("models")
    if not isinstance(report_models, list) or len(report_models) != 508:
        raise BenchmarkAnalysisError("comparison-report model inventory drift")
    report_by_id = {
        str(row.get("model_id")): row for row in report_models if isinstance(row, dict)
    }
    if set(report_by_id) != set(model_ids):
        raise BenchmarkAnalysisError("comparison report and model table differ")
    model_by_id = {str(row["model_id"]): row for row in model_rows}
    for model_id, model in model_by_id.items():
        source = report_by_id[model_id]
        for table_field, report_field in (
            ("trajectory_id", "trajectory_id"),
            ("ladder", "ladder"),
            ("experts", "experts"),
            ("languages", "languages"),
            ("seed", "seed"),
            ("nominal_tokens", "nominal_tokens"),
            ("actual_tokens", "actual_tokens"),
            ("metrics_sha256", "metrics_sha256"),
            ("runtime_receipt_sha256", "runtime_receipt_sha256"),
        ):
            if model[table_field] != source.get(report_field):
                raise BenchmarkAnalysisError(
                    f"report/model-table {table_field} drift for {model_id}"
                )

    benchmark_counts = Counter(str(row["model_id"]) for row in benchmark_rows)
    if set(benchmark_counts) != set(model_ids) or set(benchmark_counts.values()) != {6}:
        raise BenchmarkAnalysisError("benchmark rows are not six-per-readout")
    language_counts = Counter(str(row["model_id"]) for row in language_rows)
    if set(language_counts) != set(model_ids):
        raise BenchmarkAnalysisError("language metrics do not cover every readout")
    slice_counts = Counter(
        (str(row["request_set"]), str(row["benchmark"]), str(row["language"]))
        for row in language_rows
    )
    unique_slices = Counter(request_set for request_set, _benchmark, _language in slice_counts)
    if unique_slices != Counter({"english": 6, "multilingual": 196}):
        raise BenchmarkAnalysisError("request-set language-slice inventory drift")
    for model_id, count in language_counts.items():
        request_set = str(model_by_id[model_id]["request_set"])
        expected = unique_slices[request_set]
        if count != expected:
            raise BenchmarkAnalysisError(
                f"language-row count drift for {model_id}: expected {expected}, got {count}"
            )
    for row in chain(benchmark_rows, language_rows):
        model_id = str(row["model_id"])
        model = model_by_id[model_id]
        for field in (
            "trajectory_id",
            "ladder",
            "experts",
            "languages",
            "seed",
            "request_set",
            "nominal_tokens",
            "actual_tokens",
        ):
            if row[field] != model[field]:
                raise BenchmarkAnalysisError(
                    f"nested metric/model {field} drift for {model_id}"
                )
        metric_fields = (
            ("macro_acc", "macro_acc_norm")
            if "macro_acc" in row
            else ("acc", "acc_norm")
        )
        if any(not 0.0 <= float(row[field]) <= 1.0 for field in metric_fields):
            raise BenchmarkAnalysisError(f"out-of-range benchmark metric for {model_id}")

    replicated = [
        row
        for row in model_rows
        if row["ladder"] == "B"
        and int(row["experts"]) in REPLICATED_B_EXPERTS
        and int(row["languages"]) in REPLICATED_B_LANGUAGES
    ]
    observed_panel = {
        (
            int(row["experts"]),
            int(row["languages"]),
            int(row["nominal_tokens"]),
            int(row["seed"]),
        )
        for row in replicated
    }
    expected_panel = {
        (experts, languages, tokens, seed)
        for experts in REPLICATED_B_EXPERTS
        for languages in REPLICATED_B_LANGUAGES
        for tokens in REPLICATED_B_TOKENS
        for seed in REPLICATED_B_SEEDS
    }
    if observed_panel != expected_panel or len(replicated) != len(expected_panel):
        raise BenchmarkAnalysisError("replicated B benchmark panel is incomplete")


def _derive_common_english_slice(
    language_rows: Sequence[Mapping[str, Any]],
    request_manifests: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_request_set: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in language_rows:
        by_request_set[str(row["request_set"])].add(
            (str(row["benchmark"]), str(row["language"]))
        )
    intersection = by_request_set["english"] & by_request_set["multilingual"]
    if intersection != set(COMMON_ENGLISH_SLICE):
        raise BenchmarkAnalysisError(
            f"common-English benchmark intersection drift: {sorted(intersection)}"
        )
    requests_by_set_and_slice: dict[
        str, dict[tuple[str, str], list[Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for request_set, manifest in request_manifests.items():
        for request in manifest["requests"]:
            key = (str(request["benchmark"]), str(request["language"]))
            if key in COMMON_ENGLISH_SLICE:
                requests_by_set_and_slice[request_set][key].append(request)

    rows: list[dict[str, Any]] = []
    for benchmark, language in COMMON_ENGLISH_SLICE:
        key = (benchmark, language)
        english_requests = sorted(
            requests_by_set_and_slice["english"][key],
            key=lambda row: str(row["request_id"]),
        )
        multilingual_requests = sorted(
            requests_by_set_and_slice["multilingual"][key],
            key=lambda row: str(row["request_id"]),
        )
        if not english_requests or english_requests != multilingual_requests:
            raise BenchmarkAnalysisError(
                f"common-English frozen request content differs for {key}"
            )
        rows.append(
            {
                "benchmark": benchmark,
                "language": language,
                "included_for_cross_k": True,
                "english_requests": len(english_requests),
                "multilingual_requests": len(multilingual_requests),
                "request_content_sha256": canonical_json_sha256(english_requests),
                "identical_frozen_requests": True,
                "reason": (
                    "exact benchmark-language key and request content present in both "
                    "frozen request sets"
                ),
            }
        )
    return rows


def _build_model_scores(
    model_rows: Sequence[Mapping[str, Any]],
    benchmark_rows: Sequence[Mapping[str, Any]],
    language_rows: Sequence[Mapping[str, Any]],
    chance_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    chance = {
        str(row["benchmark"]): float(row["random_choice_accuracy"])
        for row in chance_rows
    }
    benchmarks_by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in benchmark_rows:
        benchmarks_by_model[str(row["model_id"])].append(row)
    language_by_model: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = defaultdict(
        dict
    )
    for row in language_rows:
        model_id = str(row["model_id"])
        key = (str(row["benchmark"]), str(row["language"]))
        if key in language_by_model[model_id]:
            raise BenchmarkAnalysisError(
                f"duplicate benchmark-language row for {model_id}: {key}"
            )
        language_by_model[model_id][key] = row

    results: list[dict[str, Any]] = []
    for model in model_rows:
        model_id = str(model["model_id"])
        task_rows = benchmarks_by_model[model_id]
        task_by_name = {str(row["benchmark"]): row for row in task_rows}
        if tuple(task_by_name) != BENCHMARK_ORDER:
            raise BenchmarkAnalysisError(f"benchmark order drift for {model_id}")
        raw_scores = [float(task_by_name[name]["macro_acc_norm"]) for name in BENCHMARK_ORDER]
        primary = _mean(raw_scores)
        if not math.isclose(
            primary,
            float(model["equal_task_macro_acc_norm"]),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise BenchmarkAnalysisError(f"primary composite mismatch for {model_id}")
        primary_chance = _mean(chance[name] for name in BENCHMARK_ORDER)
        primary_adjusted = _mean(
            _chance_adjusted(float(task_by_name[name]["macro_acc_norm"]), chance[name])
            for name in BENCHMARK_ORDER
        )

        common_lookup = language_by_model[model_id]
        if any(key not in common_lookup for key in COMMON_ENGLISH_SLICE):
            raise BenchmarkAnalysisError(f"incomplete common-English slice for {model_id}")
        common_scores = [
            float(common_lookup[key]["acc_norm"]) for key in COMMON_ENGLISH_SLICE
        ]
        common_chances = [chance[benchmark] for benchmark, _language in COMMON_ENGLISH_SLICE]
        results.append(
            {
                **_identity(model),
                "primary_metric": "equal_task_macro_acc_norm",
                "primary_score": primary,
                "primary_random_choice_baseline": primary_chance,
                "primary_chance_adjusted_score": primary_adjusted,
                "common_english_task_count": len(COMMON_ENGLISH_SLICE),
                "common_english_score": _mean(common_scores),
                "common_english_random_choice_baseline": _mean(common_chances),
                "common_english_chance_adjusted_score": _mean(
                    _chance_adjusted(score, task_chance)
                    for score, task_chance in zip(
                        common_scores, common_chances, strict=True
                    )
                ),
                "evidence_label": (
                    "three_seed_descriptive"
                    if model["ladder"] == "B"
                    and int(model["experts"]) in REPLICATED_B_EXPERTS
                    and int(model["languages"]) in REPLICATED_B_LANGUAGES
                    else "single_seed_exploratory"
                ),
            }
        )
    return results


def _build_trajectory_dynamics(
    model_scores: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_trajectory: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in model_scores:
        by_trajectory[str(row["trajectory_id"])].append(row)
    rows: list[dict[str, Any]] = []
    for trajectory_id in sorted(by_trajectory):
        readouts = sorted(
            by_trajectory[trajectory_id], key=lambda row: int(row["readout_index"])
        )
        identity_fields = ("ladder", "experts", "languages", "seed", "request_set")
        for field in identity_fields:
            if len({row[field] for row in readouts}) != 1:
                raise BenchmarkAnalysisError(
                    f"trajectory {trajectory_id} changes {field} across readouts"
                )
        expected_indices = list(range(1, len(readouts) + 1))
        if [int(row["readout_index"]) for row in readouts] != expected_indices:
            raise BenchmarkAnalysisError(f"trajectory {trajectory_id} readout index drift")
        if not bool(readouts[-1]["is_final_readout"]) or any(
            bool(row["is_final_readout"]) for row in readouts[:-1]
        ):
            raise BenchmarkAnalysisError(f"trajectory {trajectory_id} final marker drift")
        actual_tokens = [float(row["actual_tokens"]) for row in readouts]
        if actual_tokens != sorted(actual_tokens) or len(set(actual_tokens)) != len(readouts):
            raise BenchmarkAnalysisError(f"trajectory {trajectory_id} token order drift")
        log2_tokens = [math.log2(value) for value in actual_tokens]
        primary = [float(row["primary_score"]) for row in readouts]
        primary_adjusted = [
            float(row["primary_chance_adjusted_score"]) for row in readouts
        ]
        common = [float(row["common_english_score"]) for row in readouts]
        common_adjusted = [
            float(row["common_english_chance_adjusted_score"]) for row in readouts
        ]
        primary_peak_index = max(range(len(primary)), key=primary.__getitem__)
        common_peak_index = max(range(len(common)), key=common.__getitem__)
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "ladder": readouts[0]["ladder"],
                "experts": readouts[0]["experts"],
                "languages": readouts[0]["languages"],
                "seed": readouts[0]["seed"],
                "request_set": readouts[0]["request_set"],
                "evidence_label": readouts[0]["evidence_label"],
                "readouts": len(readouts),
                "initial_nominal_tokens": readouts[0]["nominal_tokens"],
                "final_nominal_tokens": readouts[-1]["nominal_tokens"],
                "initial_actual_tokens": readouts[0]["actual_tokens"],
                "final_actual_tokens": readouts[-1]["actual_tokens"],
                "initial_primary_score": primary[0],
                "final_primary_score": primary[-1],
                "primary_endpoint_delta": primary[-1] - primary[0],
                "primary_ols_slope_per_actual_token_doubling": _ols_slope(
                    log2_tokens, primary
                ),
                "primary_chance_adjusted_ols_slope_per_actual_token_doubling": (
                    _ols_slope(log2_tokens, primary_adjusted)
                ),
                "primary_peak_score": primary[primary_peak_index],
                "primary_peak_nominal_tokens": readouts[primary_peak_index][
                    "nominal_tokens"
                ],
                "initial_common_english_score": common[0],
                "final_common_english_score": common[-1],
                "common_english_endpoint_delta": common[-1] - common[0],
                "common_english_ols_slope_per_actual_token_doubling": _ols_slope(
                    log2_tokens, common
                ),
                "common_english_chance_adjusted_ols_slope_per_actual_token_doubling": (
                    _ols_slope(log2_tokens, common_adjusted)
                ),
                "common_english_peak_score": common[common_peak_index],
                "common_english_peak_nominal_tokens": readouts[common_peak_index][
                    "nominal_tokens"
                ],
                "positive_primary_steps": sum(
                    current > previous + FLOAT_TOLERANCE
                    for previous, current in zip(primary, primary[1:])
                ),
                "negative_primary_steps": sum(
                    current < previous - FLOAT_TOLERANCE
                    for previous, current in zip(primary, primary[1:])
                ),
                "zero_primary_steps": sum(
                    math.isclose(current, previous, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE)
                    for previous, current in zip(primary, primary[1:])
                ),
            }
        )
    if len(rows) != EXPECTED_TRAJECTORIES:
        raise BenchmarkAnalysisError("trajectory-dynamics row count drift")
    return rows


def _build_threshold_crossings(
    model_scores: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_trajectory: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in model_scores:
        by_trajectory[str(row["trajectory_id"])].append(row)
    rows: list[dict[str, Any]] = []
    scopes = (
        ("primary_six_task", "primary_chance_adjusted_score"),
        ("common_english_four_task", "common_english_chance_adjusted_score"),
    )
    for trajectory_id in sorted(by_trajectory):
        readouts = sorted(
            by_trajectory[trajectory_id], key=lambda row: int(row["readout_index"])
        )
        for scope, field in scopes:
            values = [float(row[field]) for row in readouts]
            for threshold in POST_HOC_THRESHOLDS:
                crossing_index = next(
                    (index for index, value in enumerate(values) if value >= threshold),
                    None,
                )
                if crossing_index is None:
                    status = "not_observed"
                    previous_tokens = readouts[-1]["nominal_tokens"]
                    first_tokens = None
                    first_score = None
                elif crossing_index == 0:
                    status = "already_at_or_above_at_first_readout"
                    previous_tokens = None
                    first_tokens = readouts[0]["nominal_tokens"]
                    first_score = values[0]
                else:
                    status = "first_observed_crossing_within_window"
                    previous_tokens = readouts[crossing_index - 1]["nominal_tokens"]
                    first_tokens = readouts[crossing_index]["nominal_tokens"]
                    first_score = values[crossing_index]
                rows.append(
                    {
                        "trajectory_id": trajectory_id,
                        "ladder": readouts[0]["ladder"],
                        "experts": readouts[0]["experts"],
                        "languages": readouts[0]["languages"],
                        "seed": readouts[0]["seed"],
                        "score_scope": scope,
                        "threshold": threshold,
                        "threshold_scale": "chance_adjusted_acc_norm",
                        "status": status,
                        "previous_nominal_tokens": previous_tokens,
                        "first_crossing_nominal_tokens": first_tokens,
                        "first_crossing_score": first_score,
                        "evidence_label": "post_hoc_descriptive_sensitivity",
                    }
                )
    expected = EXPECTED_TRAJECTORIES * len(scopes) * len(POST_HOC_THRESHOLDS)
    if len(rows) != expected:
        raise BenchmarkAnalysisError("threshold-crossing row count drift")
    return rows


def _build_final_tables(
    model_scores: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_rows = [dict(row) for row in model_scores if bool(row["is_final_readout"])]
    final_rows.sort(
        key=lambda row: (
            str(row["ladder"]),
            int(row["experts"]),
            int(row["languages"]),
            int(row["seed"]),
        )
    )
    if len(final_rows) != EXPECTED_FINAL_READOUTS:
        raise BenchmarkAnalysisError("final endpoint table is incomplete")
    canonical = [
        row
        for row in final_rows
        if int(row["seed"]) == 1234 and row["source_cohort"] == "canonical_release"
    ]
    if len(canonical) != 64:
        raise BenchmarkAnalysisError("canonical final grid is not 64 design cells")
    return final_rows, canonical




__all__ = [
    "_build_final_tables",
    "_build_model_scores",
    "_build_threshold_crossings",
    "_build_trajectory_dynamics",
    "_chance_adjusted",
    "_derive_chance_levels",
    "_derive_common_english_slice",
    "_direction",
    "_identity",
    "_mean",
    "_ols_slope",
    "_sample_summary",
    "_validate_campaign_grains",
]
