"""Verify routing-table calculations and render the existing diagnostic figures."""
from pathlib import Path

from result_analysis import evidence as io
from result_analysis.checks import Checks
from multilingual_megatron.experiments.scaling.stage1_routing_repair import reporting, report_figures
from . import analysis


def run(evidence, output):
    output = io.output_directory(output, evidence)
    checks = Checks()
    tables = {name: evidence.table(f"routing/tables/{name}.csv") for name in
              ("checkpoint_summary", "language_metrics", "coverage_contrasts", "loss_associations")}
    calculated = analysis.analyze(tables["checkpoint_summary"], tables["language_metrics"])
    for name, keys, values in (
        ("coverage_contrasts", ["ladder", "experts", "seed", "tokens_seen", "cohort", "K_from", "K_to", "metric"], ["delta"]),
        ("loss_associations", ["id", "cohort", "predictor"], ["n_languages", "spearman_rho"]),
    ):
        actual = calculated[name].set_index(keys).sort_index()
        saved = tables[name].set_index(keys).sort_index()
        checks.require(name+" population", actual.index.equals(saved.index) and actual.index.is_unique)
        for column in values:
            checks.require(name+" missingness "+column, actual[column].isna().equals(saved[column].isna()))
            valid = actual[column].notna()
            checks.close(name+" "+column, actual.loc[valid, column], saved.loc[valid, column])
        calculated[name].to_csv(output / f"{name}.csv", index=False)
    release = evidence.json("routing/release_manifest.json")
    evidence.json("routing/analysis_spec.json")
    checkpoints = tables["checkpoint_summary"].id.nunique()
    checks.require("checkpoint inventory", checkpoints == release["audit"]["checkpoints"])
    # Intervals come from paired sequence-block draws, not endpoint subtraction.
    tables["coverage_contrasts"].to_csv(output / "coverage_contrasts_with_saved_intervals.csv", index=False)
    tables["checkpoint_summary"].to_csv(output / "checkpoint_summary.csv", index=False)
    report_figures.render_figures(tables, output)
    claims = {"checkpoints": checkpoints, "coverage_contrasts": len(calculated["coverage_contrasts"]),
              "loss_associations": len(calculated["loss_associations"]), "checks_passed": checks.save(output),
              "reproduction_scope": "coverage deltas and loss associations recomputed from saved summaries; raw counts and bootstrap draws not replayed",
              "caveats": ["exploratory associations, not causal evidence", "fixed language cohorts with equal weights",
                          "checkpoints match total-token horizons, not target exposure",
                          "saved intervals condition on checkpoints; languages/layers are not training replicates"]}
    return io.finish(output, evidence, "routing_analysis", claims,
                     [Path(m.__file__) for m in (analysis, reporting, report_figures, io)] + [Path(__file__)])
