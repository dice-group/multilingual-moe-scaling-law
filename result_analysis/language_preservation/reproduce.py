"""Recompute preservation policies and per-language losses within measured support."""
from pathlib import Path

import pandas as pd

from result_analysis import evidence as io
from result_analysis import learning_curves as curves
from result_analysis.checks import Checks
from . import estimators, plots
from .analysis import summarize


def run(evidence, output):
    evidence.require_current()
    output = io.output_directory(output, evidence)
    checks = Checks()
    context = evidence.json("study/primary_fits/policy_context.json")
    rows = evidence.table("study/primary_fits/fit_rows.parquet")
    policies = evidence.table("study/paper_exports/expansion_policies.csv").query(
        "base_K == 50 and target_K == 200 and cap == 2200000000 and baseline_F == 2 and method == 'smooth' and tolerance == 0")
    checks.require("exact preservation objectives", set(policies.objective) == {"mean", "all"} and len(policies) == 2)
    languages = context["cohorts"]["incumbent50"]
    geometry = evidence.table("scoring/grid.parquet")[["ladder", "E", "N_act", "N_total"]].drop_duplicates()
    searched, candidates = estimators.search_policies(rows, geometry, languages, context["penalty"])
    a, b = searched.set_index("objective").sort_index(), policies.set_index("objective").sort_index()
    checks.require("searched preservation recipe identities", a.config.equals(b.config)
                   and a.baseline_config.equals(b.baseline_config))
    checks.close("searched preservation compute", a.compute_multiplier, b.compute_multiplier, atol=1e-10)
    searched.to_csv(output / "preservation_policies.csv", index=False)
    candidates.to_csv(output / "preservation_candidates.csv", index=False)
    policies = searched
    actual = estimators.preservation_at_saved_policies(rows, policies, languages, context["penalty"])
    reference = evidence.table("study/paper_exports/preservation_languages.csv")
    keys = ["objective", "language"]
    paired = actual.merge(reference, on=keys, how="outer", validate="one_to_one", indicator=True, suffixes=("", "_saved"))
    checks.require("all 100 language/objective comparisons", len(paired) == 100 and paired._merge.eq("both").all())
    checks.close("per-language preservation losses", paired.loss_delta, paired.loss_delta_saved,
                 source="study/paper_exports/preservation_languages.csv")
    actual = actual.merge(reference[keys+["name", "resource"]], on=keys, validate="one_to_one")
    actual.to_csv(output / "language_preservation.csv", index=False)
    summary = summarize(actual)
    summary.to_csv(output / "preservation_summary.csv")
    # Target-exposure matching is a different objective from loss preservation.
    dilution = evidence.table("study/paper_exports/expansion_dilution.csv").query(
        "base_K == 50 and target_K == 200 and cap == 2200000000 and baseline_F == 2")
    checks.require("one exposure-preservation comparison", len(dilution) == 1)
    base = policies.iloc[0]
    size, experts = base.baseline_config.split("/E")
    shares = rows[rows.ladder.eq(size) & rows.E.eq(int(experts)) & rows.K.isin([50, 200])
                  & rows.language.isin(languages)].sort_values("D_total").groupby(["K", "language"]).tail(1)
    shares = shares.assign(share=shares.D_t/shares.D_total).pivot(index="language", columns="K", values="share")
    checks.require("paired exposure inventory", len(shares) == 50 and shares.notna().all().all())
    ratios = shares[50]/shares[200]
    # Small rounding drift in realized batch shares is reported, not hidden.
    exposure = float(dilution.exposure_multiplier_median.iloc[0])
    checks.close("exposure ratio from recorded shares", ratios.median(), exposure, atol=5e-4,
                 source="study/primary_fits/fit_rows.parquet", scope="realized batch-discretized shares")
    pd.DataFrame({"language": ratios.index, "exposure_multiplier": ratios.values}).to_csv(output / "exposure_ratios.csv", index=False)
    mean = actual[actual.objective.eq("mean")]
    worst, best = mean.loc[mean.loss_delta.idxmax()], mean.loc[mean.loss_delta.idxmin()]
    checks.close("paper mean/all compute multipliers", summary.loc[["mean", "all"], "compute_multiplier"],
                 [1.92, 2.32], atol=.005, source="paper/my_content.tex")
    checks.require("paper protected counts", summary.loc["mean", "protected_languages"] == 18
                   and summary.loc["all", "protected_languages"] == 50, source="paper/my_content.tex")
    plots.render(actual, summary, exposure, output)
    claims = {"preservation": summary.to_dict("index"), "exposure_compute_multiplier": exposure,
              "worst_mean_preservation": {"language": worst.language, "name": worst["name"], "loss_delta": float(worst.loss_delta)},
              "best_mean_preservation": {"language": best.language, "name": best["name"], "loss_delta": float(best.loss_delta)},
              "checks_passed": checks.save(output),
              "reproduction_scope": "headline preservation policies re-searched across supported measured recipes, then all per-language losses and compute multipliers recalculated",
              "caveats": ["same incumbent 50 languages", "bounded smoothing within measured support",
                          "all-language policy is feasible, not a global minimum", "logical FLOPs, not GPU-hours"]}
    return io.finish(output, evidence, "language_preservation", claims,
                     [Path(__file__), Path(estimators.__file__), Path(curves.__file__), Path(plots.__file__), Path(io.__file__)])
