"""Explore feasible mean/all-language preservation across measured recipes."""
from . import estimators


def from_parquet(data):
    """Default original-50 preservation, recomputed from checkpoint losses."""
    import pandas as pd
    rows = pd.read_parquet(data / "main_grid.parquet").query("metric_role == 'trained'")
    geometry = rows[["ladder", "E", "N_act", "N_total"]].drop_duplicates()
    languages = sorted(rows.loc[rows.K.eq(50), "language"].unique())
    return analyze(rows, geometry, languages=languages, base_k=50, target_k=200,
                   cap=2.2e9, baseline_f=2., penalty=5.62341325190349e-7)


def summarize(losses, *, tolerance=1e-9):
    """Summarize nested language changes, not independent training replicates."""
    return losses.assign(protected=losses.loss_delta.le(tolerance)).groupby("objective").agg(
        compute_multiplier=("compute_multiplier", "first"), protected_languages=("protected", "sum"),
        languages=("language", "size"), mean_delta=("loss_delta", "mean"), worst_delta=("loss_delta", "max"))


def analyze(rows, geometry, *, languages, base_k, target_k, cap, baseline_f,
            penalty, objectives=("mean", "all")):
    """Return feasible policies, every candidate's status and per-language losses.

    baseline_f is in logical EFLOPs. Select one seed first. An unreached
    objective remains absent from policies and explicit in candidate statuses;
    it is not an extrapolated answer or a global impossibility claim.
    """
    import pandas as pd
    import numpy as np
    languages = tuple(languages)
    sizes = rows.loc[rows.K.isin([base_k, target_k]) & rows.language.isin(languages),
                     ["ladder", "E", "N_act"]].drop_duplicates().merge(
        geometry[["ladder", "E", "N_act"]], on=["ladder", "E"], how="left",
        validate="many_to_one", suffixes=("_observed", "_geometry"))
    if sizes.empty or not np.isfinite(sizes[["N_act_observed", "N_act_geometry"]]).all().all() or not np.allclose(
            sizes.N_act_observed, sizes.N_act_geometry, rtol=0, atol=0):
        raise ValueError("geometry does not match observed active sizes")
    policies, candidates = estimators.search_policies(rows, geometry, tuple(languages), penalty,
        cap=cap, baseline_f=baseline_f, base_k=base_k, target_k=target_k,
        objectives=objectives, require_reached=False)
    if policies.empty:
        losses = pd.DataFrame(columns=["objective", "language", "reference_loss",
            "expanded_loss", "loss_delta", "compute_multiplier"])
    else:
        losses = estimators.preservation_at_saved_policies(rows, policies, tuple(languages), penalty)
    return {"policies": policies, "candidates": candidates,
            "language_deltas": losses, "summary": summarize(losses)}


def run(evidence, output):
    """Compatibility entrypoint for the separate published-result reproduction."""
    from .reproduce import run as reproduce
    return reproduce(evidence, output)
