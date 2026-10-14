"""Explore measured-recipe allocation with explicit cohorts, caps and coverage."""
import numpy as np

from . import observed


def from_parquet(data):
    """Default observed allocation, recomputed from checkpoint losses."""
    import pandas as pd
    rows = pd.read_parquet(data / "main_grid.parquet").query("metric_role == 'trained'")
    geometry = rows[["ladder", "E", "N_act", "N_total"]].drop_duplicates()
    languages = sorted(rows.loc[rows.K.eq(12), "language"].unique())
    return analyze(rows, geometry, languages=languages, coverages=(12, 200), cap=4e8,
                   penalty=5.62341325190349e-7, methods=("smooth",))


def analyze(rows, geometry, *, languages, coverages, cap, penalty,
            methods=("smooth", "linear_log", "pchip_log"), grid_points=181):
    """Return loss curves and bracketed switches within shared observed support.

    Select one seed and supply recorded active/stored geometry. No law or paper
    crossover is assumed. No switch is a valid result; finite grid resolution
    can miss short-lived winner intervals, so vary grid_points for sensitivity.
    fitted.optimum separately supports fixed-law allocation experiments.
    """
    languages, coverages = tuple(languages), tuple(coverages)
    data = rows[rows.K.isin(coverages) & rows.language.isin(languages)]
    if data.empty or set(data.K) != set(coverages):
        raise ValueError("missing requested coverage panel")
    sizes = data[["ladder", "E", "N_act"]].drop_duplicates().merge(
        geometry[["ladder", "E", "N_act"]], on=["ladder", "E"], how="left",
        validate="many_to_one", suffixes=("_observed", "_geometry"))
    if not np.isfinite(sizes[["N_act_observed", "N_act_geometry"]]).all().all() or not np.allclose(
            sizes.N_act_observed, sizes.N_act_geometry, rtol=0, atol=0):
        raise ValueError("geometry does not match observed active sizes")
    switches, losses = observed.empirical_switches(data, geometry, languages, penalty,
        cap=cap, coverages=coverages, methods=methods, grid_points=grid_points)
    return {"switches": switches, "loss_curves": losses}


def run(evidence, output):
    """Compatibility entrypoint for the separate published-result reproduction."""
    from .reproduce import run as reproduce
    return reproduce(evidence, output)
