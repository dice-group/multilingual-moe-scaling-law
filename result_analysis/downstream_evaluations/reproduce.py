"""Recompute downstream contrasts and answer transitions from saved scores."""
from pathlib import Path

import numpy as np
import pandas as pd

from result_analysis import evidence as io
from result_analysis.checks import Checks
from . import contrasts, accounting, plots


def compare(actual, expected, keys, checks, name):
    actual, expected = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    checks.require(name+" membership", actual.index.equals(expected.index))
    columns = [c for c in actual.select_dtypes(include="number") if c in expected]
    a, b = actual[columns].to_numpy(float), expected[columns].to_numpy(float)
    checks.require(name+" missingness", np.array_equal(np.isnan(a), np.isnan(b)))
    checks.require(name+" finite scores", not np.isinf(a).any() and not np.isinf(b).any())
    mask = np.isfinite(a) & np.isfinite(b)
    checks.close(name+" values", a[mask], b[mask], atol=1e-10)


def run(evidence, output):
    evidence.require_current()
    output = io.output_directory(output, evidence)
    checks = Checks()
    frame = evidence.table("benchmark/language_metrics_with_validation.csv")
    checks.require("unique endpoint-language cells", not frame.duplicated(["model_id", "benchmark", "language"]).any())
    checks.require("complete 14 task / 82 model suite", frame.benchmark.nunique() == 14
                   and frame.groupby("benchmark").model_id.nunique().eq(82).all())
    checks.require("same task items across models", frame.groupby(["benchmark", "language"]).samples.nunique().eq(1).all())
    context = evidence.json("study/primary_fits/policy_context.json")
    tables = contrasts.aggregate(frame, context["cohorts"]["common12"])
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    interaction_keys = ["benchmark", "ladder", "seed", "panel", "cohort", "metric", "E_low", "E_high"]
    compare(tables["coverage_interactions"], evidence.table("benchmark/coverage_interactions.csv"),
            interaction_keys, checks, "all cohort/metric/expert interactions")
    compare(tables["cohort_metrics"], evidence.table("benchmark/cohort_metrics.csv"),
            contrasts.IDENTITY+["panel", "cohort"], checks, "all language-balanced endpoint scores")
    interactions = tables["coverage_interactions"]
    primary = interactions.query("cohort == 'common12' and metric == 'primary' and E_low == 1 and E_high == 32")
    headline = primary[primary.seed.eq(1234)].copy()
    seeds = primary[primary.ladder.eq("B")].copy()
    for name, table in (("headline_interactions", headline), ("b_seed_interactions", seeds)):
        compare(table, evidence.table(f"benchmark/report_final/tables/{name}.csv"),
                ["benchmark", "ladder", "seed"], checks, name)
        table.to_csv(output / f"{name}.csv", index=False)
    # 'primary' for external-text tasks is bits/byte. The paper's 22--58%
    # and 20--62% comparisons specifically use nats/token (external_nll).
    losses = interactions.query("cohort == 'common12' and metric == 'external_nll' and E_low == 1 and E_high == 32").copy()
    losses["attenuation_percent"] = 100*losses.interaction/losses.coverage_penalty_low
    checks.require("external NLL interactions all positive", losses.interaction.gt(0).all())
    reference = evidence.table("benchmark/report_final/tables/external_validation_loss_comparison.csv")
    compare(losses[["benchmark", "ladder", "seed", "interaction", "attenuation_percent"]], reference,
            ["benchmark", "ladder", "seed"], checks, "paper external NLL attenuation")
    byte = interactions.query("cohort == 'common12' and metric == 'bits_per_byte' and E_low == 1 and E_high == 32")
    checks.require("byte-normalized direction at all sizes and seeds", byte.interaction.gt(0).all())
    losses.to_csv(output / "external_loss_attenuation.csv", index=False)
    answers, transitions, means = accounting.aggregate(evidence.table("benchmark/examples/language_examples.csv"),
                                                       evidence.table("benchmark/examples/paired_answer_changes.csv"), checks)
    checks.require("all 48 answer-transition comparisons", len(answers) == 48)
    compare(answers, evidence.table("transfer/score_and_transition_accounting.csv"),
            ["benchmark", "ladder", "seed"], checks, "answer-transition accounting")
    answers.to_csv(output / "score_and_transition_accounting.csv", index=False)
    transitions.to_csv(output / "language_macro_transitions.csv", index=False)
    means.to_csv(output / "example_diagnostic_means.csv", index=False)
    # These intervals and confusion diagnostics are sealed upstream reductions,
    # not a bootstrap or a GPU evaluation silently rerun by this CPU command.
    for source, destination in (("benchmark/examples/paired_bootstrap.csv", "retained_conditional_intervals.csv"),
                                ("benchmark/report_final/tables/classification_overview.csv", "retained_classification_overview.csv"),
                                ("benchmark/report_final/tables/coverage_inventory.csv", "retained_coverage_inventory.csv")):
        evidence.table(source).to_csv(output / destination, index=False)
    for receipt in ("benchmark/analysis_receipt.json", "benchmark/acceptance_receipt.json",
                    "benchmark/report_final/verification.json", "benchmark/examples/verification.json", "transfer/verification.json"):
        evidence.json(receipt)
    intervals = evidence.table("benchmark/examples/paired_bootstrap.csv")
    cells = frame.set_index(["benchmark", "ladder", "seed", "E", "K", "language"])
    estimates, saved_estimates = [], []
    for row in intervals[intervals.contrast.eq("interaction")].itertuples():
        values = [cells.loc[(row.benchmark, row.ladder, row.seed, e, k, row.language), row.metric]
                  for e, k in ((1, 12), (32, 12), (1, 200), (32, 200))]
        sign = -1 if row.benchmark in contrasts.LOSS else 100
        estimates.append(sign*(values[3]-values[2]-values[1]+values[0]))
        saved_estimates.append(row.estimate)
    checks.close("retained bootstrap point estimates", estimates, saved_estimates, atol=1e-10)
    plots.render(headline, losses[losses.seed.eq(1234)], answers, output)
    ranges = losses[losses.seed.eq(1234)].groupby("benchmark").attenuation_percent.agg(["min", "max"])
    claims = {"benchmarks": int(frame.benchmark.nunique()), "model_endpoints": int(frame.model_id.nunique()),
              "headline_comparisons": len(headline), "answer_accounting_comparisons": len(answers),
              "external_loss_attenuation_percent": ranges.to_dict("index"), "checks_passed": checks.save(output),
              "reproduction_scope": "language-level endpoint reductions to all contrasts and paired-answer accounting; original GPU scores and conditional bootstrap draws not regenerated",
              "caveats": ["equal corpus-language weights after averaging varieties", "fixed benchmark-supported common12 cohorts",
                          "endpoints match total tokens, not target exposure", "accuracy interaction is task-dependent",
                          "three B seeds do not replicate the full grid", "retained intervals condition on the trained models"]}
    return io.finish(output, evidence, "external_evaluation", claims,
                     [Path(__file__), *[Path(m.__file__) for m in (contrasts, accounting, plots, io)]])
