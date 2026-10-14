"""Observed and fitted capacity allocation, with distinct predictor attribution."""
from pathlib import Path

import pandas as pd

from result_analysis import evidence as io
from result_analysis import learning_curves as curves
from result_analysis.checks import Checks
from . import fitted, observed, plots


def run(evidence, output):
    contract = evidence.require_current()
    output = io.output_directory(output, evidence)
    checks = Checks()
    checks.require("allocation predictor attribution", contract["allocation_predictor"] == "M6-ND")
    context = evidence.json("study/primary_fits/policy_context.json")
    rows = evidence.table("study/primary_fits/fit_rows.parquet")
    geometry = evidence.table("scoring/grid.parquet")[["ladder", "E", "N_act", "N_total"]].drop_duplicates()
    switches, allocation = observed.empirical_switches(rows, geometry, context["cohorts"]["common12"], context["penalty"])
    reference = evidence.table("study/paper_exports/allocation_crossovers.csv").query(
        "cap == 400000000 and K in [12,200] and method in ['smooth','linear_log','pchip_log']")
    keys = ["cap", "K", "method", "below", "above"]
    checks.close("observed allocation switches", switches.set_index(keys).sort_index().F,
                 reference.set_index(keys).sort_index().F, atol=1e-9)
    switches.to_csv(output / "empirical_switches.csv", index=False)
    allocation.to_csv(output / "empirical_allocation.csv", index=False)
    decisions, metrics = observed.policy_scores(evidence.table("replay/policy_predictions.parquet"),
                                               evidence.table("replay/policy_observed.parquet"), context["cohorts"])
    keys = ["model", "cohort", "reference", "cap"]
    reference = evidence.table("replay/policy_metrics.csv").set_index(keys).sort_index()
    actual = metrics.set_index(keys).sort_index()
    checks.require("all policy score populations", actual.index.equals(reference.index))
    checks.close("all policy scores", actual, reference[actual.columns], atol=2e-12)
    decisions.to_csv(output / "policy_decisions.csv", index=False)
    metrics.to_csv(output / "policy_scores.csv", index=False)
    winners, fitted_claims = fitted.run(evidence, output, checks, geometry)
    # RN13/RN14 surface searches remain separate sealed sensitivity evidence.
    # This fast command does not silently repeat their numerical size searches.
    sensitivities = evidence.table("replay/continuous_switches.csv")
    sensitivities.to_csv(output / "retained_repetition_switches.csv", index=False)
    ratios = {}
    for model, group in sensitivities.query("cap == 5100000000 and before == 32 and after == 16").groupby("model"):
        part = group.set_index("K").FLOPs
        if 1 in part.index and 200 in part.index:
            ratios[model] = float(part.loc[200]/part.loc[1])
    primary = switches[switches.method.eq("smooth")].set_index("K").F
    plots.render(winners, allocation, output)
    claims = {"main_fitted_allocation": fitted_claims,
              "observed_switch_EFLOPs": {str(k): float(v) for k, v in primary.items()},
              "observed_switch_delay": float(primary.loc[200]/primary.loc[12]),
              "retained_repetition_switch_ratios": ratios, "checks_passed": checks.save(output),
              "reproduction_scope": "observed switches and policy scores recalculated; M6-ND surface reoptimized with frozen coefficients; RN13/RN14 surface sensitivities retained, not reoptimized",
              "caveats": ["M6-ND allocation is not attributed to RN14", "native-K means are different language populations",
                          "observed switches use the same 12 languages and bounded support",
                          "dashed fitted surfaces extrapolate token horizons; logical FLOPs are not GPU-hours"]}
    return io.finish(output, evidence, "capacity_allocation", claims,
        [Path(m.__file__) for m in (curves, fitted, observed, plots, io, fitted.model)] + [Path(__file__)])
