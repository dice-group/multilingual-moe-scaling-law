"""Supported restoration policies across interpolation methods and objectives."""
import numpy as np
import pandas as pd
from scipy.interpolate import PPoly
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib.lines import Line2D
from result_analysis.learning_curves import invert_monotone, first_feasible_piecewise


def restore(recipes, reference, cohort, method, objective, tolerance=0.):
    candidates = []
    for recipe in recipes:
        def change(f):
            delta = recipe.values(f, cohort, method)-reference
            return float(delta.mean() if objective == "mean" else delta.max())
        if method == "smooth":
            compute, status = invert_monotone(change, tolerance, recipe.fmin, recipe.fmax)
        else:
            positions = recipe.indices(cohort)
            if method == "pchip_log":
                coefficients = recipe.pchip.c[..., positions].copy()
            elif method == "linear_log":
                y = recipe.loss[:, positions]
                slopes = np.diff(y, axis=0)/np.diff(np.log(recipe.tokens))[:, None]
                coefficients = np.stack([slopes, y[:-1]])
            else:
                raise ValueError("unregistered inversion estimator")
            coefficients[-1] -= reference+tolerance
            if objective == "mean":
                coefficients = coefficients.mean(axis=-1, keepdims=True)
            polynomial = PPoly(coefficients, np.log(recipe.tokens), extrapolate=False)
            log_tokens, status = first_feasible_piecewise(polynomial)
            compute = None if log_tokens is None else float(np.exp(log_tokens)*6*recipe.n/1e18)
            if compute is not None and change(compute) > tolerance+2e-10:
                raise ValueError("piecewise inverse failed joint-preservation check")
        candidates.append((compute, status, recipe))
    reached = [row for row in candidates if row[0] is not None]
    if not reached:
        return dict(status="unreached", F=np.nan, config=None, unknown_continuations="")
    f, status, recipe = min(reached, key=lambda row: (row[0], row[2].stored, row[2].name))
    unresolved = [c.name for compute, _, c in candidates if compute is None and c.fmax < f]
    delta = recipe.values(f, cohort, method)-reference
    return dict(status=status, F=f, config=recipe.name, N_act=recipe.n, E=recipe.e,
                D_total=f*1e18/(6*recipe.n), unknown_continuations=",".join(unresolved),
                mean_delta=float(delta.mean()), worst_delta=float(delta.max()),
                protected_fraction=float(np.mean(delta <= 1e-9)))


def expansion_analysis(bank, cohorts, *, caps):
    records, distribution = [], []
    for base_k, target_k in ((12, 50), (50, 200)):
        cohort = cohorts[base_k]
        for cap in (caps[0], caps[-1]):
            old = sorted([c for (_, k), c in bank.items() if k == base_k and c.stored <= cap], key=lambda c: (c.stored, c.name))
            new = sorted([c for (_, k), c in bank.items() if k == target_k and c.stored <= cap], key=lambda c: (c.stored, c.name))
            lower, upper = max(c.fmin for c in old+new), min(c.fmax for c in old+new)
            grid = sorted(set(np.geomspace(lower, upper, 37)) | {2., 4.})
            for method in ("smooth", "linear_log", "pchip_log"):
                for f in grid:
                    baseline = min(old, key=lambda c: (c.macro(f, cohort, method), c.stored))
                    reference = baseline.values(f, cohort, method)
                    for objective, tolerance in (("mean", 0.), ("all", 0.), ("all", .01)):
                        target = restore(new, reference, cohort, method, objective, tolerance)
                        records.append(dict(base_K=base_k, target_K=target_k, cap=cap, method=method,
                            baseline_F=f, baseline_config=baseline.name, baseline_N_act=baseline.n,
                            objective=objective, tolerance=tolerance,
                            compute_multiplier=target["F"]/f,
                            N_act_multiplier=target.get("N_act", np.nan)/baseline.n,
                            D_total_multiplier=target.get("D_total", np.nan)/(f*1e18/(6*baseline.n)), **target))
                        if objective == "mean" and target["config"] and method == "smooth":
                            delta = bank[target["config"], target_k].values(target["F"], cohort)-reference
                            for language, d in zip(cohort, delta):
                                distribution.append(dict(base_K=base_k, target_K=target_k, cap=cap,
                                    baseline_F=f, language=language, loss_delta=d,
                                    unknown_continuations=target["unknown_continuations"]))
    return pd.DataFrame(records), pd.DataFrame(distribution)


def plot_expansion(policies, distribution, *, caps):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.1), layout="constrained")
    objectives = [("mean", 0., "#c74757", "Preserve incumbent mean"),
                  ("all", 0., "#44316f", "Preserve every incumbent"),
                  ("all", .01, "#258698", "Every incumbent within 0.01 nats")]
    for row, cap in enumerate((caps[0], caps[-1])):
        for column, base_k in enumerate((12, 50)):
            ax = axes[row, column]
            for objective, tolerance, color, _ in objectives:
                data = policies.loc[policies.cap.eq(cap) & policies.base_K.eq(base_k)
                                    & policies.objective.eq(objective) & policies.tolerance.eq(tolerance)]
                pivot = data.pivot(index="baseline_F", columns="method", values="compute_multiplier")
                primary = data.loc[data.method.eq("smooth")].set_index("baseline_F").loc[pivot.index]
                complete = primary.status.eq("bracketed") & primary.unknown_continuations.eq("")
                ax.fill_between(pivot.index, pivot.min(axis=1), pivot.max(axis=1), color=color, alpha=.13, linewidth=0)
                ax.plot(pivot.index, pivot.smooth, color=color, linestyle="--", linewidth=1.5)
                ax.plot(pivot.index, pivot.smooth.where(complete), color=color, linewidth=2)
            target_k = 50 if base_k == 12 else 200
            ax.set(title=f"K{base_k} → K{target_k} · cap {cap/1e9:g}B", xscale="log",
                   xlabel="Reference training compute (EFLOPs)", ylabel="Required compute / reference compute")
            ticks = [f for f in (1.2, 2, 3, 4) if pivot.index.min() <= f <= pivot.index.max()]
            ax.set_xticks(ticks, labels=ticks)
            ax.xaxis.set_minor_locator(NullLocator())
    fig.legend(handles=[Line2D([], [], color=c, label=label) for _, _, c, label in objectives]+[
        Line2D([], [], color=".4", linestyle="--", label="Shorter trajectories leave optimum unresolved")],
        loc="outside upper center", ncol=2, frameon=False, fontsize=8)
    cost_figure = fig

    group = distribution.loc[distribution.base_K.eq(50) & distribution.cap.eq(caps[-1])]
    pivot = group.pivot(index="baseline_F", columns="language", values="loss_delta")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    axes[0].fill_between(pivot.index, pivot.min(axis=1), pivot.max(axis=1), color="#c74757", alpha=.12, label="Range over 50 languages")
    axes[0].fill_between(pivot.index, pivot.quantile(.1, axis=1), pivot.quantile(.9, axis=1), color="#c74757", alpha=.25, label="10th–90th percentiles")
    axes[0].plot(pivot.index, pivot.median(axis=1), color="#c74757", label="Median language")
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set(ylabel="Incumbent-language loss change (nats/token)", title="Equal mean quality can conceal regressions")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].plot(pivot.index, 100*pivot.le(1e-9).mean(axis=1), color="#44316f", linewidth=2,
                 label="At least previous quality")
    axes[1].plot(pivot.index, 100*pivot.le(.01+1e-9).mean(axis=1), color="#258698", linewidth=2,
                 label="Within 0.01 nats of previous quality")
    axes[1].set(ylim=(0, 105), ylabel="Incumbent languages (%)", title="K50 → K200 · mean-preserving allocation")
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")
    for ax in axes:
        ax.set(xscale="log", xlabel="Reference compute (EFLOPs); stored cap 2.2B")
        ax.set_xticks([2, 3, 4], labels=[2, 3, 4])
        ax.xaxis.set_minor_locator(NullLocator())
    return cost_figure, fig


def restoration_policy(curves, base_k, target_k, budget, cap, method):
    from result_analysis.compute_allocation.frontier import best_at, invert_loss
    base = best_at(curves, base_k, budget, cap, method)
    details = []
    for (config, k), curve in curves.items():
        if k != target_k or curve.N_total.iloc[0] > cap:
            continue
        compute, status = invert_loss(curve, base["loss"], method)
        details.append(dict(config=config, compute=compute, status=status))
    reached = [row for row in details if row["compute"] is not None]
    prefix = dict(base_K=base_k, target_K=target_k, baseline_F=budget,
                  parameter_cap=cap, method=method,
                  base_config=base["config"], base_loss=base["loss"],
                  base_N_act=base["N_act"], base_N_total=base["N_total"], base_D_total=base["D_total"],
                  feasible_configurations=len(details), unreached_configurations=len(details)-len(reached))
    if not reached:
        return dict(**prefix, status="not_reached_by_any_observed_configuration")
    winner = min(reached, key=lambda r: (r["compute"], r["config"]))
    curve = curves[winner["config"], target_k]
    compute = winner["compute"]
    # Detect whether an unobserved continuation could undercut this winner.
    exhausted_below_winner = [r["config"] for r in details if r["compute"] is None
                              and curves[r["config"], target_k].F.max() < compute]
    return dict(**prefix, status=winner["status"], target_config=winner["config"],
                target_F=compute, compute_multiplier=compute/budget,
                target_N_act=int(curve.N_act.iloc[0]), target_N_total=int(curve.N_total.iloc[0]),
                N_act_multiplier=curve.N_act.iloc[0]/base["N_act"],
                stored_multiplier=curve.N_total.iloc[0]/base["N_total"],
                D_total_multiplier=compute/budget*base["N_act"]/curve.N_act.iloc[0],
                unknown_continuations_below_winner=",".join(exhausted_below_winner))
