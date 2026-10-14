"""Canonical observed-allocation and fixed-prediction policy estimators."""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from result_analysis import learning_curves as curves


def policy_scores(predictions, observed, cohorts, *, caps=(.4e9, .7e9, 2.2e9)):
    """Select by predicted loss; measure regret on the same feasible observations."""
    if predictions.duplicated(["model", "config", "language", "FLOPs"]).any():
        raise ValueError("duplicate policy prediction")
    if observed.duplicated(["reference", "config", "language", "FLOPs"]).any():
        raise ValueError("duplicate policy observation")
    if not np.isfinite(predictions[["prediction", "FLOPs", "N_total"]]).all().all() or not np.isfinite(observed[["observed", "FLOPs"]]).all().all():
        raise ValueError("nonfinite policy values")
    decisions = []
    for cohort, languages in cohorts.items():
        for frame, keys in ((predictions, ["model", "config", "FLOPs"]),
                            (observed, ["reference", "config", "FLOPs"])):
            part = frame[frame.language.isin(languages)]
            if part.empty or not part.groupby(keys).language.nunique().eq(len(languages)).all():
                raise ValueError("incomplete policy language cohort")
        truth = observed[observed.language.isin(languages)].groupby(["config", "FLOPs", "reference"], as_index=False)["observed"].mean()
        predicted = predictions[predictions.language.isin(languages)].groupby(["model", "config", "FLOPs", "N_total"], as_index=False).prediction.mean()
        # Missing candidates would change the decision set and understate regret.
        keys = ["config", "FLOPs"]
        for _, model in predicted.groupby("model"):
            for _, reference in truth.groupby("reference"):
                paired = model[keys].merge(reference[keys], on=keys, how="outer",
                                           validate="one_to_one", indicator=True)
                if not paired._merge.eq("both").all():
                    raise ValueError("policy candidate populations differ")
        joined = predicted.merge(truth, on=["config", "FLOPs"], validate="many_to_many")
        for cap in caps:
            for (model, reference, budget), group in joined[joined.N_total.le(cap)].groupby(["model", "reference", "FLOPs"]):
                chosen = group.loc[group.prediction.idxmin()]
                best = group.loc[group.observed.idxmin()]
                decisions.append(dict(model=model, cohort=cohort, reference=reference, cap=cap,
                    FLOPs=budget, chosen=chosen.config, best=best.config,
                    regret=chosen.observed-best.observed, correct=chosen.config == best.config,
                    candidate_count=len(group)))
    decisions = pd.DataFrame(decisions)
    metrics = decisions.groupby(["model", "cohort", "reference", "cap"], as_index=False).agg(
        mean_regret=("regret", "mean"), max_regret=("regret", "max"),
        correct_choices=("correct", "sum"), n_budgets=("regret", "size"), winner_accuracy=("correct", "mean"))
    return decisions, metrics


def empirical_switches(rows, geometry, languages, penalty, *, cap=400000000,
                       coverages=(12, 200), methods=("smooth", "linear_log", "pchip_log"), grid_points=181):
    """Search supported measured recipes; transitions are bracketed on the chosen grid."""
    if not languages or len(set(languages)) != len(languages) or grid_points < 2:
        raise ValueError("unique nonempty language cohort and at least two grid points required")
    if not np.isfinite(cap) or cap <= 0 or not coverages or not methods:
        raise ValueError("positive cap, coverages and interpolation methods required")
    if geometry.duplicated(["ladder", "E"]).any() or not np.isfinite(geometry[["N_act", "N_total"]]).all().all():
        raise ValueError("invalid or duplicate architecture geometry")
    if not geometry[["N_act", "N_total"]].gt(0).all().all():
        raise ValueError("positive architecture sizes required")
    feasible = geometry[geometry.N_total.le(cap)].sort_values(["N_total", "ladder", "E"])
    if feasible.empty:
        raise ValueError("no feasible measured architecture under cap")
    transitions, plot_rows = [], []
    for coverage in coverages:
        recipes = []
        for config in feasible.itertuples():
            group = rows[rows.ladder.eq(config.ladder) & rows.E.eq(config.E)
                         & rows.K.eq(coverage) & rows.language.isin(languages)]
            fits = [curves.LearningCurve.fit(g.sort_values("D_total").D_total.to_numpy(),
                    g.sort_values("D_total").adjusted_loss.to_numpy(), penalty)
                    for _, g in group.groupby("language")]
            if len(fits) != len(languages):
                raise ValueError("incomplete allocation cohort")
            recipes.append((f"{config.ladder}/E{config.E}", config.N_act, fits))
        lower = max(6*n*fit.x[0]/1e18 for _, n, fits in recipes for fit in fits)
        upper = min(6*n*fit.x[-1]/1e18 for _, n, fits in recipes for fit in fits)
        if lower >= upper:
            raise ValueError("no common observed compute support")
        budgets = np.geomspace(lower, upper, grid_points)
        for method in methods:
            def macro(recipe, compute):
                _, n, fits = recipe
                return np.mean([fit(np.asarray(compute)*1e18/(6*n), method) for fit in fits], axis=0)
            losses = np.stack([macro(recipe, budgets) for recipe in recipes])
            winners = losses.argmin(axis=0)
            for i, recipe in enumerate(recipes):
                plot_rows.extend(dict(K=coverage, method=method, F=float(f), config=recipe[0], loss=float(loss))
                                 for f, loss in zip(budgets, losses[i]))
            for j in np.flatnonzero(np.diff(winners) != 0):
                before, after = recipes[winners[j]], recipes[winners[j+1]]
                crossing = brentq(lambda f: macro(before, f)-macro(after, f), budgets[j], budgets[j+1])
                transitions.append(dict(cap=cap, K=coverage, method=method, F=crossing,
                                        below=before[0], above=after[0]))
    return (pd.DataFrame(transitions, columns=["cap", "K", "method", "F", "below", "above"]),
            pd.DataFrame(plot_rows))
