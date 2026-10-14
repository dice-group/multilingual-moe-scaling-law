from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from result_analysis.downstream_evaluations.campaign.analysis_artifacts import (  # noqa: E402
    BenchmarkAnalysisError,
    COMMON_ENGLISH_SLICE,
    REPLICATED_B_EXPERTS,
    REPLICATED_B_LANGUAGES,
    REPLICATED_B_SEEDS,
    REPLICATED_B_TOKENS,
)
from result_analysis.downstream_evaluations.campaign.analysis_replication import (  # noqa: E402
    _build_b_seed_contrasts,
)
from result_analysis.downstream_evaluations.campaign.analysis_scores import (  # noqa: E402
    _build_threshold_crossings,
    _derive_chance_levels,
    _derive_common_english_slice,
    _sample_summary,
)
from result_analysis.downstream_evaluations.campaign.analysis_plots import (  # noqa: E402
    _build_scaling_plot_rows,
    _build_slope_summary_rows,
)
from result_analysis.downstream_evaluations.campaign.visualization_data import (  # noqa: E402
    build_b_seed_task_curve_rows,
    build_task_endpoint_rows,
    build_task_slope_rows,
)
from multilingual_megatron.experiments.scaling.stage1_benchmarking.campaign_profile import (  # noqa: E402
    BENCHMARK_ORDER,
)


CHOICES = {
    "belebele_cloze": 4,
    "okapi_hellaswag": 4,
    "xstorycloze": 2,
    "xcopa": 2,
    "uhura_arc_easy": 4,
    "global_mmlu_cloze": 4,
}


def _request_manifest() -> dict[str, object]:
    requests = []
    for benchmark in BENCHMARK_ORDER:
        for choice_index in range(CHOICES[benchmark]):
            requests.append(
                {
                    "benchmark": benchmark,
                    "task": f"{benchmark}_en",
                    "language": "en",
                    "doc_id": 0,
                    "choice_index": choice_index,
                    "request_id": f"{benchmark}-{choice_index}",
                }
            )
    return {"requests": requests}


def _replicated_b_panel() -> list[dict[str, object]]:
    rows = []
    for experts in REPLICATED_B_EXPERTS:
        for languages in REPLICATED_B_LANGUAGES:
            for tokens in REPLICATED_B_TOKENS:
                for seed_index, seed in enumerate(REPLICATED_B_SEEDS):
                    rows.append(
                        {
                            "model_id": f"b-e{experts}-k{languages}-d{tokens}-s{seed}",
                            "ladder": "B",
                            "experts": experts,
                            "languages": languages,
                            "nominal_tokens": tokens,
                            "seed": seed,
                            "primary_score": (
                                0.35
                                + experts * 0.001
                                + languages * 0.00001
                                + seed_index * 0.0001
                            ),
                            "common_english_score": (
                                0.40
                                + experts * 0.001
                                - languages * 0.0001
                                + seed_index * 0.0001
                            ),
                        }
                    )
    return rows


READOUTS_BY_LADDER = {
    "A": (500_000_000, 1_000_000_000, 2_000_000_000, 4_000_000_000, 8_000_000_000),
    "B": (
        500_000_000,
        1_000_000_000,
        2_000_000_000,
        4_000_000_000,
        8_000_000_000,
        16_000_000_000,
    ),
    "C": (
        500_000_000,
        1_000_000_000,
        2_000_000_000,
        4_000_000_000,
        8_000_000_000,
        16_000_000_000,
        32_000_000_000,
    ),
    "D": (
        500_000_000,
        1_000_000_000,
        2_000_000_000,
        4_000_000_000,
        8_000_000_000,
        16_000_000_000,
        32_000_000_000,
    ),
}


def _visualization_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model_scores = []
    trajectory_rows = []
    for ladder, tokens_by_readout in READOUTS_BY_LADDER.items():
        for experts in (1, 4, 16, 32):
            for languages in (1, 12, 50, 200):
                seeds = (
                    REPLICATED_B_SEEDS
                    if ladder == "B"
                    and experts in REPLICATED_B_EXPERTS
                    and languages in REPLICATED_B_LANGUAGES
                    else (1234,)
                )
                for seed_index, seed in enumerate(seeds):
                    trajectory_id = f"{ladder}-e{experts}-k{languages}-s{seed}"
                    slope = 0.001 * (1 + seed_index + experts / 32)
                    trajectory_rows.append(
                        {
                            "trajectory_id": trajectory_id,
                            "ladder": ladder,
                            "experts": experts,
                            "languages": languages,
                            "seed": seed,
                            "primary_ols_slope_per_actual_token_doubling": slope,
                            "primary_chance_adjusted_ols_slope_per_actual_token_doubling": (
                                1.5 * slope
                            ),
                        }
                    )
                    for readout_index, nominal_tokens in enumerate(
                        tokens_by_readout, start=1
                    ):
                        model_id = f"{trajectory_id}-d{nominal_tokens}"
                        model_scores.append(
                            {
                                "model_id": model_id,
                                "trajectory_id": trajectory_id,
                                "ladder": ladder,
                                "experts": experts,
                                "languages": languages,
                                "seed": seed,
                                "readout_index": readout_index,
                                "nominal_tokens": nominal_tokens,
                                "is_final_readout": (
                                    readout_index == len(tokens_by_readout)
                                ),
                                "source_cohort": (
                                    "canonical_release"
                                    if seed == 1234
                                    else "b_seed_confirmation"
                                ),
                                "actual_tokens": nominal_tokens,
                                "active_parameters": {
                                    "A": 93_000_000,
                                    "B": 190_000_000,
                                    "C": 349_000_000,
                                    "D": 590_000_000,
                                }[ladder],
                                "primary_score": (
                                    0.34
                                    + experts * 0.0001
                                    + languages * 0.000001
                                    + seed_index * 0.00001
                                    + slope * (readout_index - 1)
                                ),
                                "primary_chance_adjusted_score": (
                                    0.02
                                    + experts * 0.00015
                                    + languages * 0.0000015
                                    + seed_index * 0.000015
                                    + 1.5 * slope * (readout_index - 1)
                                ),
                            }
                        )
    return model_scores, trajectory_rows


def _task_visualization_fixture(
    model_scores: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    chance = {
        "belebele_cloze": 0.25,
        "okapi_hellaswag": 0.25,
        "xstorycloze": 0.50,
        "xcopa": 0.50,
        "uhura_arc_easy": 0.25,
        "global_mmlu_cloze": 0.25,
    }
    chance_rows = [
        {"benchmark": benchmark, "random_choice_accuracy": chance[benchmark]}
        for benchmark in BENCHMARK_ORDER
    ]
    benchmark_rows = []
    for row in model_scores:
        for task_index, benchmark in enumerate(BENCHMARK_ORDER):
            benchmark_rows.append(
                {
                    **row,
                    "benchmark": benchmark,
                    "macro_acc_norm": (
                        chance[benchmark]
                        + 0.01 * (task_index + 1)
                        + 0.002 * int(row["readout_index"])
                    ),
                }
            )
    return benchmark_rows, chance_rows


class Stage1JointMoEBenchmarkAnalysisTest(unittest.TestCase):
    def test_paper_tables_share_the_validated_visualization_grain(self) -> None:
        model_scores, _trajectory_rows = _visualization_fixture()
        benchmark_rows, chance_rows = _task_visualization_fixture(model_scores)

        endpoints = build_task_endpoint_rows(benchmark_rows, chance_rows)
        task_slopes = build_task_slope_rows(
            benchmark_rows, model_scores, chance_rows
        )
        seed_curves = build_b_seed_task_curve_rows(
            benchmark_rows, model_scores, chance_rows
        )

        self.assertEqual(len(endpoints), 24)
        self.assertEqual({row["n"] for row in endpoints}, {16})
        self.assertEqual(len(task_slopes), 24)
        self.assertEqual({row["n"] for row in task_slopes}, {16})
        self.assertTrue(all(row["positive"] == 16 for row in task_slopes))
        self.assertEqual(len(seed_curves), 216)
        self.assertEqual({row["n_seeds"] for row in seed_curves}, {3})

    def test_visualization_rows_preserve_seed_and_readout_grain(self) -> None:
        model_scores, trajectory_rows = _visualization_fixture()
        self.assertEqual(len(model_scores), 508)
        self.assertEqual(len(trajectory_rows), 82)

        plot_rows = _build_scaling_plot_rows(model_scores)
        slope_rows = _build_slope_summary_rows(trajectory_rows)
        self.assertEqual(len(plot_rows), 400)
        self.assertEqual(len(slope_rows), 64)

        replicated = next(
            row
            for row in plot_rows
            if row["ladder"] == "B"
            and row["experts"] == 32
            and row["languages"] == 200
            and row["nominal_tokens"] == 16_000_000_000
        )
        self.assertEqual(replicated["n_seeds"], 3)
        self.assertEqual(replicated["seeds"], list(REPLICATED_B_SEEDS))
        self.assertGreater(
            replicated["primary_score_max"], replicated["primary_score_min"]
        )
        self.assertEqual(replicated["active_parameters"], 190_000_000)
        self.assertGreater(replicated["logical_compute_eflops"], 0.0)
        self.assertGreater(
            replicated["chance_adjusted_change_from_0p5b_pp_mean"], 0.0
        )

        single_seed = next(
            row
            for row in plot_rows
            if row["ladder"] == "B"
            and row["experts"] == 4
            and row["languages"] == 200
            and row["nominal_tokens"] == 16_000_000_000
        )
        self.assertEqual(single_seed["n_seeds"], 1)
        self.assertEqual(single_seed["evidence_label"], "single_seed_descriptive")

        replicated_slope = next(
            row
            for row in slope_rows
            if row["ladder"] == "B"
            and row["experts"] == 32
            and row["languages"] == 200
        )
        self.assertEqual(replicated_slope["n_trajectories"], 3)
        self.assertEqual(replicated_slope["positive_trajectories"], 3)
        self.assertGreater(
            replicated_slope[
                "mean_chance_adjusted_slope_pp_per_actual_token_doubling"
            ],
            replicated_slope["mean_slope_pp_per_actual_token_doubling"],
        )

    def test_sample_summary_uses_sample_sd_and_range(self) -> None:
        summary = _sample_summary([0.1, 0.2, 0.3])
        self.assertEqual(summary["n_seeds"], 3)
        self.assertTrue(math.isclose(summary["mean"], 0.2))
        self.assertTrue(math.isclose(summary["sample_sd"], 0.1))
        self.assertTrue(math.isclose(summary["range"], 0.2))

    def test_chance_levels_are_derived_from_frozen_request_multiplicity(self) -> None:
        rows = _derive_chance_levels(
            {"english": _request_manifest(), "multilingual": _request_manifest()}
        )
        self.assertEqual([row["benchmark"] for row in rows], list(BENCHMARK_ORDER))
        self.assertEqual(
            {row["benchmark"]: row["choice_count"] for row in rows}, CHOICES
        )
        self.assertTrue(
            math.isclose(
                sum(float(row["random_choice_accuracy"]) for row in rows) / 6.0,
                1.0 / 3.0,
            )
        )

        broken = _request_manifest()
        broken["requests"].append(dict(broken["requests"][0]))  # type: ignore[index,union-attr]
        with self.assertRaisesRegex(BenchmarkAnalysisError, "duplicate choice"):
            _derive_chance_levels(
                {"english": broken, "multilingual": _request_manifest()}
            )

    def test_common_english_slice_requires_identical_frozen_requests(self) -> None:
        language_rows = []
        requests = {"english": [], "multilingual": []}
        for benchmark, language in COMMON_ENGLISH_SLICE:
            for request_set in ("english", "multilingual"):
                language_rows.append(
                    {
                        "request_set": request_set,
                        "benchmark": benchmark,
                        "language": language,
                    }
                )
                requests[request_set].append(
                    {
                        "benchmark": benchmark,
                        "language": language,
                        "request_id": f"{benchmark}-{language}",
                    }
                )
        language_rows.extend(
            [
                {
                    "request_set": "english",
                    "benchmark": "okapi_hellaswag",
                    "language": "en",
                },
                {
                    "request_set": "multilingual",
                    "benchmark": "okapi_hellaswag",
                    "language": "de",
                },
            ]
        )
        rows = _derive_common_english_slice(
            language_rows,
            {
                "english": {"requests": requests["english"]},
                "multilingual": {"requests": requests["multilingual"]},
            },
        )
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["identical_frozen_requests"] for row in rows))

        requests["multilingual"][0] = {
            **requests["multilingual"][0],
            "request_id": "drifted",
        }
        with self.assertRaisesRegex(BenchmarkAnalysisError, "request content differs"):
            _derive_common_english_slice(
                language_rows,
                {
                    "english": {"requests": requests["english"]},
                    "multilingual": {"requests": requests["multilingual"]},
                },
            )

    def test_b_contrasts_are_matched_by_seed_and_summarized_at_n_three(self) -> None:
        seedwise, summaries = _build_b_seed_contrasts(_replicated_b_panel())
        self.assertEqual(len(seedwise), 162)
        self.assertEqual(len(summaries), 54)
        self.assertTrue(all(row["n_seeds"] == 3 for row in summaries))
        expert = next(
            row
            for row in summaries
            if row["contrast_family"] == "experts_e32_minus_e1"
            and row["languages"] == 12
            and row["nominal_tokens"] == 16_000_000_000
        )
        self.assertTrue(math.isclose(expert["mean_estimate"], 0.031))
        self.assertEqual(expert["stable_direction_across_all_three_seeds"], "positive")

    def test_threshold_crossings_are_discrete_and_left_censored(self) -> None:
        rows = []
        for trajectory_index in range(82):
            for readout_index, value in enumerate((-0.01, 0.06), start=1):
                rows.append(
                    {
                        "trajectory_id": f"trajectory-{trajectory_index}",
                        "ladder": "A",
                        "experts": 1,
                        "languages": 1,
                        "seed": 1234,
                        "readout_index": readout_index,
                        "nominal_tokens": 500_000_000 * readout_index,
                        "primary_chance_adjusted_score": value,
                        "common_english_chance_adjusted_score": value + 0.20,
                    }
                )
        crossings = _build_threshold_crossings(rows)
        self.assertEqual(len(crossings), 656)
        primary_zero = next(
            row
            for row in crossings
            if row["trajectory_id"] == "trajectory-0"
            and row["score_scope"] == "primary_six_task"
            and row["threshold"] == 0.0
        )
        self.assertEqual(primary_zero["status"], "first_observed_crossing_within_window")
        self.assertEqual(primary_zero["previous_nominal_tokens"], 500_000_000)
        self.assertEqual(primary_zero["first_crossing_nominal_tokens"], 1_000_000_000)
        common_zero = next(
            row
            for row in crossings
            if row["trajectory_id"] == "trajectory-0"
            and row["score_scope"] == "common_english_four_task"
            and row["threshold"] == 0.0
        )
        self.assertEqual(
            common_zero["status"], "already_at_or_above_at_first_readout"
        )


if __name__ == "__main__":
    unittest.main()
