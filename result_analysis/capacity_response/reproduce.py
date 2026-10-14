from pathlib import Path

import numpy as np
import pandas as pd

from result_analysis import evidence as io
from result_analysis.checks import Checks
from result_analysis.capacity_response import tables as estimators
from . import plots
from . import response as dynamics


def run(evidence: io.Evidence, output: Path) -> dict:
    output = io.output_directory(output, evidence)
    current = evidence.analysis.get("version") == "stage1-paper-v2"
    checks = Checks()
    rows = evidence.table("study/primary_fits/fit_rows.parquet")
    if len(rows) != 26300 or rows.trajectory_id.nunique() != 64:
        raise ValueError("unexpected primary-grid population")
    if rows.duplicated(["trajectory_id", "readout_id", "language"]).any():
        raise ValueError("duplicate observation")
    points = estimators.matched_pressure(rows)
    estimators.validate_primary_support(points)
    ladder, summary = estimators.summarize(points)
    saved = evidence.table("study/paper_exports/primary_pressure_points.csv")
    keys = ["ladder", "E", "language", "target_readout_id"]
    joined = points.merge(saved, on=keys, validate="one_to_one", how="outer", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("primary matched-exposure membership changed")
    gap = float(abs(joined.pressure - joined.adjusted_delta).max())
    np.testing.assert_allclose(joined.pressure, joined.adjusted_delta, atol=2e-11, rtol=0)
    points.to_csv(output / "matched_exposure_points.csv", index=False)
    ladder.to_csv(output / "pressure_by_size.csv", index=False)
    summary.to_csv(output / "pressure_summary.csv", index=False)

    nested = []
    for old, new in estimators.COVERAGE_CONTRASTS:
        part, _ = estimators.summarize(estimators.matched_pressure(rows, old, new))
        nested.append(part.assign(reference_K=old, target_K=new))
    nested = pd.concat(nested, ignore_index=True)
    nested.to_csv(output / "nested_expansions.csv", index=False)

    # Seed, LR, benchmark and composition controls retain their own populations.
    seeds = evidence.table("study/paper_exports/seed_attenuation_by_seed.csv")
    np.testing.assert_allclose(1 - seeds.E32_pressure / seeds.E1_pressure,
                               seeds.relative_attenuation, atol=2e-11, rtol=0)
    seeds.to_csv(output / "seed_robustness.csv", index=False)
    fixed_lr = evidence.table("study/paper_exports/fixed_lr_pressure_summary.csv")
    fixed_lr.to_csv(output / "fixed_lr_summary.csv", index=False)
    reference_response = evidence.table("study/main_figure_inputs/matched_response.csv")
    response = dynamics.matched_response(rows, reference_response)
    for name in ("response", "compute_ratio_min", "compute_ratio_max"):
        np.testing.assert_allclose(response[name], reference_response[name], atol=2e-11, rtol=0)
    response.to_csv(output / "matched_response_curves.csv", index=False)
    crossings = dynamics.crossovers(response)
    gains = estimators.expert_gain(rows, set(rows.loc[rows.K.eq(12), "language"]), (12, 200))
    saved_gains = evidence.table("study/main_figure_inputs/expert_gains.csv")
    gain_keys = ["ladder", "K", "D_total", "N_act"]
    gain_check = gains.merge(saved_gains, on=gain_keys, how="outer", validate="one_to_one", indicator=True)
    if not gain_check._merge.eq("both").all():
        raise ValueError("expert-gain population changed")
    np.testing.assert_allclose(gain_check.gain_x, gain_check.gain_y, atol=2e-11, rtol=0)
    gains["FLOPs"] = 6*gains.N_act*gains.D_total
    gains.to_csv(output / "equal_compute_expert_gains.csv", index=False)
    plots.render(ladder, summary, response, gains, output)
    means = summary.set_index("E").pressure
    claims = {
        "primary_observations": len(rows), "training_trajectories": 64,
        "dependent_readouts": len(rows[["trajectory_id", "readout_id"]].drop_duplicates()),
        "matched_rows": len(points), "matched_languages": points.language.nunique(),
        "pressure_by_E": {str(e): float(value) for e, value in means.items()},
        "attenuation_E1_to_E32": float(1 - means.loc[32] / means.loc[1]),
        "saved_point_max_abs_difference": gap,
        "size_D_crossing_FLOPs": crossings,
        "size_D_crossing_delay": crossings["32"]/crossings["1"],
        "response_curve_max_abs_difference": float(abs(response.response-reference_response.response).max()),
        "seed_attenuation": seeds[["seed", "relative_attenuation"]].to_dict("records"),
        "scope": "K12 to K200, common12, actual D_t; within-size averages then equal-size mean",
        "caveat": "primary grid has one seed; three B-size seeds are a separate control, not full-grid replication",
    }
    if current:
        from . import supporting
        checks.close("primary matched-point replay", joined.pressure, joined.adjusted_delta, atol=2e-11)
        checks.close("paper size-balanced pressure", means.loc[[1, 4, 16, 32]], [.0796, .0648, .0321, .0135], atol=5.001e-5)
        claims["supporting"] = supporting.run(evidence, output, checks, rows)
        claims["checks_passed"] = checks.save(output)
        claims["reproduction_scope"] = "observations to primary/nested responses, native capacity gains, resource expansion and composition; seed points reaggregated; LR/routing summaries retained"
    return io.finish(output, evidence, "capacity_response", claims,
                     [Path(__file__), Path(estimators.__file__), Path(dynamics.__file__), Path(plots.__file__), Path(io.__file__),
                      *([Path(supporting.__file__)] if current else [])])
