"""Full-trajectory pressure, allocation, and expansion-cost analysis.

Run as a module from the repository root. Sealed observations and fitted-model
artifacts are read-only. Figures, tables, a gallery, and a provenance manifest
are written under this package's ignored outputs/smooth_analysis directory.
"""

from itertools import combinations
from importlib.metadata import version
from pathlib import Path
import html
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator, PPoly
from scipy.optimize import brentq

from result_analysis.evidence import sha256 as sha256_file
from result_analysis.capacity_response.tables import load_primary
from result_analysis.capacity_response.sensitivity import coverage_gap_sensitivity, plot_coverage_gap_sensitivity
from result_analysis.language_preservation.sensitivity import expansion_analysis, plot_expansion
from .frontier import prepare_curves
from result_analysis.learning_curves import RecipeCurve, LearningCurve, PENALTIES, horizon_cv, invert_monotone, allocation_metrics, first_feasible_piecewise


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs/smooth_analysis"
EXPERT_COLORS = {1: "#44316f", 4: "#258698", 16: "#d28b22", 32: "#c74757"}
METHODS = ("smooth", "linear_log", "pchip_log", "smooth_low", "smooth_high", "smooth_unconstrained")
CAPS = (400_000_000, 700_000_000, 2_200_000_000)




def save_table(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / f"{name}.csv", index=False)
    return frame


def save_figure(fig, name):
    for suffix in ("pdf", "png"):
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if suffix == "pdf" else {}
        fig.savefig(OUT / f"{name}.{suffix}", dpi=190, bbox_inches="tight", **kwargs)
    plt.close(fig)


def tune_smoothing(frame, cohort):
    _, groups = prepare_curves(frame, cohort)
    curves = {str(key): (g.D_total.to_numpy(float), g.loss.to_numpy(float)) for key, g in groups.items()}
    penalty, scores = horizon_cv(curves)
    save_table("smoothing_cv", scores)
    # Outer omissions are the common interior horizons, removed from EVERY
    # trajectory before tuning lambda using the remaining interior readouts.
    shared = sorted(set.intersection(*[set(x[1:-1]) for x, _ in curves.values()]))
    outer = []
    for horizon in shared:
        training = {key: (x[x != horizon], y[x != horizon]) for key, (x, y) in curves.items()}
        chosen, _ = horizon_cv(training)
        for key, (x, y) in curves.items():
            keep = x != horizon
            fitted = LearningCurve.fit(x[keep], y[keep], chosen)
            truth = float(y[x == horizon][0])
            for method in ("smooth", "linear_log", "pchip_log"):
                outer.append(dict(trajectory=key, omitted_tokens=horizon, method=method,
                                  penalty=chosen, truth=truth,
                                  prediction=float(fitted(horizon, method))))
    outer = save_table("nested_horizon_validation", outer)
    outer["squared_error"] = (outer.prediction-outer.truth)**2
    summary = outer.groupby("method").squared_error.mean().pow(.5).rename("rmse").reset_index()
    save_table("nested_horizon_validation_summary", summary)
    index = list(PENALTIES).index(penalty)
    strengths = dict(smooth=penalty, smooth_low=PENALTIES[max(0, index-1)],
                     smooth_high=PENALTIES[min(len(PENALTIES)-1, index+1)])
    return strengths, summary




def allocation_analysis(bank, cohort):
    records, transitions = [], []
    for cap in CAPS:
        for k in (12, 50, 200):
            recipes = sorted([c for (_, coverage), c in bank.items() if coverage == k and c.stored <= cap],
                             key=lambda c: (c.stored, c.name))
            lower, upper = max(c.fmin for c in recipes), min(c.fmax for c in recipes)
            grid = np.geomspace(lower, upper, 181)
            for method in METHODS:
                loss = np.stack([c.macro(grid, cohort, method) for c in recipes])
                winners = loss.argmin(axis=0)
                for i, recipe in enumerate(recipes):
                    for j, f in enumerate(grid):
                        regret = float(loss[i, j]-loss[:, j].min())
                        records.append(dict(cap=cap, K=k, method=method, F=f,
                            config=recipe.name, N_act=recipe.n, E=recipe.e, N_total=recipe.stored,
                            D_total=f*1e18/(6*recipe.n), loss=loss[i, j], regret=regret,
                            winner=i == winners[j], near_005=regret <= .005,
                            near_010=regret <= .01, near_020=regret <= .02))
                for j in np.flatnonzero(np.diff(winners) != 0):
                    a, b = recipes[winners[j]], recipes[winners[j+1]]
                    root = brentq(lambda f: a.macro(f, cohort, method)-b.macro(f, cohort, method), grid[j], grid[j+1])
                    transitions.append(dict(cap=cap, K=k, method=method, F=root,
                                            below=a.name, above=b.name))
    return save_table("allocation_regret", records), save_table("allocation_crossovers", transitions)






def heldout_validation(frame, bank, cohort):
    """Post-hoc policy metrics on the existing untouched K200 trajectory fold.

    The primary scoring reference is log-linear interpolation of held-out
    observations. No smoothing is fitted to predictor training/held-out rows.
    """
    from multilingual_megatron.experiments.scaling.stage1_fit import fit
    fit_root = ROOT / "result_analysis/stage1_scaling_fit_jointmoe_x1_v1"
    report_path = fit_root / "reports/candidate_selection.json"
    report = json.loads(report_path.read_text())
    sources = {str(report_path.relative_to(ROOT)): sha256_file(report_path)}
    receipt_path = fit_root / "input" / Path(report["input_receipt"]["path"]).name
    if sha256_file(receipt_path) != report["input_receipt"]["file_sha256"]:
        raise ValueError("fit input receipt changed")
    receipt = json.loads(receipt_path.read_text())
    input_path = fit_root / "input" / Path(receipt["fit_rows_path"]).name
    if sha256_file(input_path) != receipt["fit_rows_file_sha256"]:
        raise ValueError("fit input table changed")
    sources[str(receipt_path.relative_to(ROOT))] = sha256_file(receipt_path)
    sources[str(input_path.relative_to(ROOT))] = sha256_file(input_path)
    keys = ["trajectory_id", "readout_id", "language"]
    columns = ["N_act", "D_t", "D_total", "E", "K", "adjusted_loss"]
    joined = frame[keys+columns].merge(pd.read_parquet(input_path)[keys+columns], on=keys,
                                      suffixes=("_sealed", "_fit"), validate="one_to_one")
    if len(joined) != len(frame) or any(not np.allclose(joined[c+"_sealed"], joined[c+"_fit"],
            rtol=1e-12, atol=1e-12) for c in columns):
        raise ValueError("sealed observations differ from fitted-model inputs")
    metrics, decisions, margins = [], [], []
    grid = np.geomspace(max(c.fmin for c in bank.values()), min(c.fmax for c in bank.values()), 81)
    recipes = sorted([c for (_, k), c in bank.items() if k == 200], key=lambda c: (c.stored, c.name))
    observed = np.stack([c.macro(grid, cohort, "linear_log") for c in recipes])
    for source in report["source_results"]:
        if source["fold_id"] != "K200_common_languages":
            continue
        path = fit_root / "results" / Path(source["path"]).name
        result = json.loads(path.read_text())
        if result["result_sha256"] != source["sha256"] or not result["selected"]["success"]:
            raise ValueError("held-out result identity or completion changed")
        if result["fit_rows_content_sha256"] != receipt["fit_rows_content_sha256"]:
            raise ValueError("held-out fit uses a different input table")
        sources[str(path.relative_to(ROOT))] = sha256_file(path)
        membership = result["membership"]
        membership_path = fit_root / "input" / Path(membership["path"]).name
        if sha256_file(membership_path) != membership["file_sha256"]:
            raise ValueError("held-out membership changed")
        sources[str(membership_path.relative_to(ROOT))] = sha256_file(membership_path)
        if not set(cohort) <= set(membership["score_languages"]):
            raise ValueError("unregistered validation language")
        if set(membership["train_trajectory_ids"]) & set(membership["test_trajectory_ids"]):
            raise ValueError("trajectory leakage")
        if not all(c.trajectory in membership["test_trajectory_ids"] for c in recipes):
            raise ValueError("allocation candidate was not held out")
        shared = fit.SharedParameters(**result["selected"]["shared_parameters"])
        coefficients = pd.DataFrame(result["selected"]["language_coefficients"]).set_index("language").loc[cohort]
        predictions = []
        for c in recipes:
            # Use the exact realized exposure mapping at each evaluation point.
            tokens = grid*1e18/(6*c.n)
            exposure = np.stack([np.exp(np.interp(np.log(tokens), np.log(c.tokens), np.log(c.exposure[:, c.positions[l]]))) for l in cohort], axis=1)
            values = fit.predict(source["model"], shared, c.n, exposure, c.e, 200,
                                 coefficients.L_inf.to_numpy(), coefficients.A.to_numpy(), coefficients.B.to_numpy())
            predictions.append(values.mean(axis=1))
        predicted = np.stack(predictions)
        for a, b in combinations(range(len(recipes)), 2):
            for j, f in enumerate(grid):
                margins.append(dict(model=source["model"], a=recipes[a].name, b=recipes[b].name,
                                    F=f, observed_margin=observed[a, j]-observed[b, j],
                                    predicted_margin=predicted[a, j]-predicted[b, j]))
        for cap in CAPS:
            indices = [i for i, c in enumerate(recipes) if c.stored <= cap]
            score = allocation_metrics(observed[indices], predicted[indices])
            metrics.append(dict(model=source["model"], cap=cap, cohort_size=len(cohort),
                fold=source["fold_id"], candidate_count=len(indices), compute_points=len(grid),
                mean_regret=float(score["regret"].mean()), max_regret=float(score["regret"].max()),
                winner_accuracy=float(score["winner_matches"].mean()),
                within_001=float(score["within_tolerance"].mean()),
                pairwise_margin_rmse=score["pairwise_margin_rmse"],
                resolved_pairwise_accuracy=score["resolved_pairwise_accuracy"]))
            for j, f in enumerate(grid):
                decisions.append(dict(model=source["model"], cap=cap, F=f,
                    chosen=recipes[indices[score["chosen_indices"][j]]].name, regret=score["regret"][j]))
    sources[str(Path(fit.__file__).relative_to(ROOT))] = sha256_file(Path(fit.__file__))
    return save_table("heldout_allocation_metrics", metrics), save_table("heldout_decisions", decisions), save_table("heldout_pairwise_margins", margins), sources




def plot_allocations(table, bank):
    fig, axes = plt.subplots(3, 3, figsize=(12.8, 10.0), layout="constrained")
    for row, cap in enumerate(CAPS):
        for column, k in enumerate((12, 50, 200)):
            ax = axes[row, column]
            selected = table.loc[table.cap.eq(cap) & table.K.eq(k) & table.method.eq("smooth")]
            recipes = selected[["config", "N_total", "N_act", "E"]].drop_duplicates().sort_values(["N_total", "config"])
            pivot = selected.pivot(index="config", columns="F", values="regret").loc[recipes.config]
            x = np.log2(pivot.columns.to_numpy(float))
            edges = np.r_[x[0]-(x[1]-x[0])/2, (x[:-1]+x[1:])/2, x[-1]+(x[-1]-x[-2])/2]
            artist = ax.pcolormesh(edges, np.arange(len(recipes)+1)-.5, pivot,
                                   cmap="YlOrRd", vmin=0, vmax=.08, rasterized=True)
            # Recipes are categorical: outline qualifying intervals within each
            # row, never interpolate a contour between different architectures.
            for i, regrets in enumerate(pivot.to_numpy()):
                mask = regrets <= .01
                starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
                stops = np.flatnonzero(mask & ~np.r_[mask[1:], False])+1
                for start, stop in zip(starts, stops):
                    ax.add_patch(plt.Rectangle((edges[start], i-.43), edges[stop]-edges[start], .86,
                                               facecolor="none", edgecolor="#127f87", linewidth=1.0))
            ax.plot(x, pivot.to_numpy().argmin(axis=0), color="black", linewidth=.7, marker=".",
                    markersize=2, drawstyle="steps-post")
            for i, config in enumerate(recipes.config):
                c = bank[config, k]
                checkpoints = 6*c.n*c.tokens/1e18
                checkpoints = checkpoints[(checkpoints >= pivot.columns.min()) & (checkpoints <= pivot.columns.max())]
                ax.scatter(np.log2(checkpoints), np.full(len(checkpoints), i), marker="|", s=25, color=".4", linewidth=.7)
            ax.set(yticks=np.arange(len(recipes)), yticklabels=[f"{n/1e6:.0f}M / E{e}" for n,e in zip(recipes.N_act,recipes.E)],
                   title=f"K{k} · stored cap {cap/1e9:g}B")
            ticks = [f for f in (1.2, 2, 3, 4) if pivot.columns.min() <= f <= pivot.columns.max()]
            ax.set_xticks(np.log2(ticks), labels=[f"{f:g}" for f in ticks])
            ax.tick_params(axis="y", labelsize=7)
            if row == 2:
                ax.set_xlabel("Logical compute (EFLOPs)")
    fig.colorbar(artist, ax=axes, shrink=.65, extend="max", label="Loss above best supported recipe (nats/token)")
    fig.legend(handles=[Line2D([], [], color="black", marker=".", label="Selected recipe"),
                        Line2D([], [], color="#127f87", label="Outlined: within 0.01 nats"),
                        Line2D([], [], marker="|", linestyle="none", color=".4", label="Measured checkpoints")],
               loc="outside upper center", ncol=3, frameon=False, fontsize=8)
    save_figure(fig, "fig_allocation_regret")


def plot_crossovers(table):
    fig, ax = plt.subplots(figsize=(7.6, 4.2), layout="constrained")
    for k, color in ((12, "#44316f"), (50, "#258698"), (200, "#c74757")):
        group = table.loc[table.cap.eq(CAPS[0]) & table.K.eq(k) & table.config.isin(["A/E16", "B/E4"])]
        values = group.pivot(index=["method", "F"], columns="config", values="loss")
        margin = (values["A/E16"]-values["B/E4"]).unstack("method")
        ax.fill_between(margin.index, margin.min(axis=1), margin.max(axis=1), color=color, alpha=.15, linewidth=0)
        ax.plot(margin.index, margin.smooth, color=color, linewidth=2, label=f"K{k}")
    ax.axhline(0, color="black", linewidth=.8)
    ax.text(.02, .96, "190M / E4 has lower loss", transform=ax.transAxes, va="top", fontsize=9)
    ax.text(.34, .04, "93M / E16 has lower loss", transform=ax.transAxes, va="bottom", fontsize=9)
    ax.set(xscale="log", xlabel="Logical compute (EFLOPs)",
           ylabel="Loss: 93M/E16 minus 190M/E4 (nats/token)",
           title="Broader coverage shifts the active-versus-conditional tradeoff")
    ax.set_xticks([1.2, 2, 3, 4], labels=[1.2, 2, 3, 4])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, "fig_coverage_allocation_boundary")




def plot_validation(decisions):
    colors = {"M0": "#777777", "M1": "#258698", "M2": "#c74757"}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True, layout="constrained")
    for ax, cap in zip(axes, CAPS):
        for model, group in decisions.loc[decisions.cap.eq(cap)].groupby("model"):
            ax.plot(group.F, group.regret, color=colors[model], label=model, linewidth=1.8)
        ax.axhline(.01, color="black", linestyle=":", linewidth=.8)
        ax.set(title=f"Stored cap {cap/1e9:g}B", xlabel="Logical compute (EFLOPs)", xscale="log")
        ax.set_xticks([2, 3, 4], labels=[2, 3, 4])
        ax.xaxis.set_minor_locator(NullLocator())
    axes[0].set_ylabel("Held-out loss above best observed choice")
    axes[-1].legend(frameon=False)
    fig.suptitle("Allocation on held-out K200 trajectories · 50 incumbent languages", fontsize=11)
    save_figure(fig, "fig_heldout_allocation_regret")


def reversal_audit(bank):
    records = []
    for recipe in bank.values():
        differences = np.diff(recipe.loss, axis=0)
        for horizon, position in zip(*np.where(differences > 0)):
            records.append(dict(config=recipe.name, K=recipe.k, language=recipe.languages[position],
                earlier_tokens=recipe.tokens[horizon], later_tokens=recipe.tokens[horizon+1],
                earlier_loss=recipe.loss[horizon, position], later_loss=recipe.loss[horizon+1, position],
                increase=differences[horizon, position]))
    table = save_table("observed_language_reversals", records)
    examples = table.sort_values("increase", ascending=False).drop_duplicates("language").head(3)
    fig, axes = plt.subplots(1, len(examples), figsize=(11.4, 3.8), layout="constrained", squeeze=False)
    for ax, row in zip(axes[0], examples.itertuples(index=False)):
        recipe = bank[row.config, row.K]
        position = recipe.positions[row.language]
        grid = np.geomspace(recipe.tokens[0], recipe.tokens[-1], 201)
        ax.plot(recipe.tokens/1e9, recipe.loss[:, position], "o", color="#c74757", markersize=4)
        for method, color, style in (("smooth", "#258698", "-"), ("smooth_unconstrained", "#d28b22", "--")):
            ax.plot(grid/1e9, recipe.at_tokens(grid, method)[:, position], color=color, linestyle=style)
        ax.set(title=f"{row.language} · {row.config} · K{row.K}", xlabel="Training tokens (B)", xscale="log")
        ax.set_xticks([1, 2, 4, 8, 16, 32], labels=[1, 2, 4, 8, 16, 32])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.text(.97, .93, f"Largest observed rise\n+{row.increase:.3f} nats", transform=ax.transAxes,
                va="top", ha="right", fontsize=8)
    axes[0, 0].set_ylabel("Adjusted validation loss")
    fig.legend(handles=[Line2D([], [], color="#c74757", marker="o", linestyle="none", label="Observed checkpoints"),
        Line2D([], [], color="#258698", label="Monotone smoothing"),
        Line2D([], [], color="#d28b22", linestyle="--", label="Unconstrained smoothing")],
        loc="outside upper center", ncol=3, frameon=False, fontsize=8)
    save_figure(fig, "fig_language_curve_reversals")
    return table


def write_report(strengths, cv, standardized, crossovers, allocations, policies, metrics, reversals):
    summary = standardized.loc[standardized.method.eq("smooth")].groupby(["target_K", "E"]).log_exposure_average.mean().unstack("E")
    crossing = crossovers.loc[crossovers.cap.eq(CAPS[0])].pivot(index="K", columns="method", values="F")
    anchor = policies.loc[policies.base_K.eq(50) & policies.cap.eq(CAPS[-1])
                          & policies.baseline_F.eq(2.) & policies.tolerance.eq(0.)]
    near = allocations.loc[allocations.method.eq("smooth")].groupby(["cap", "K", "F"]).near_010.sum()
    texts = []
    for k, row in summary.iterrows():
        texts.append(f"- K12→K{k}: common-support mean residual is {row[1]:.4f} nats at E1 and {row[32]:.4f} at E32. "
                     f"These are equal-ladder averages integrated over the same log-exposure interval, rather than unequal endpoint inventories.")
    for k, row in crossing.iterrows():
        texts.append(f"- At the 0.4B stored cap, the A/E16→B/E4 crossover for K{k} is {row['smooth']:.3f} EFLOPs; "
                     f"the six-estimator range is {row.min():.3f}–{row.max():.3f}. This range is sensitivity, not a confidence interval.")
    ratios = crossing.loc[200]/crossing.loc[12]
    texts.append(f"- The more conditional allocation remains preferred up to {ratios['smooth']:.2f}× as much compute at K200 as at K12 "
                 f"({ratios.min():.2f}–{ratios.max():.2f} across estimators), at this cap and on this mixture path.")
    for objective in ("mean", "all"):
        group = anchor.loc[anchor.objective.eq(objective)].set_index("method")
        row = group.loc["smooth"]
        texts.append(f"- K50→K200 from 2 EFLOPs under a 2.2B cap, preserving {objective}: {row.compute_multiplier:.3f}× compute "
                     f"({group.compute_multiplier.min():.3f}–{group.compute_multiplier.max():.3f} across interpolation/smoothing), "
                     f"{row.baseline_config}→{row.config}; status={row.status}; unresolved continuations={row.unknown_continuations or 'none'}.")
    mean_anchor = anchor.loc[anchor.method.eq("smooth") & anchor.objective.eq("mean")].iloc[0]
    texts.append(f"- At that smoothed mean-preservation point, {mean_anchor.protected_fraction*100:.0f}% of incumbents individually match or improve; "
                 f"the largest regression is {mean_anchor.worst_delta:.4f} nats/token. The individual-protection optimization compares all feasible recipes.")
    texts.append(f"- Across the declared cap/coverage/compute grid, {100*(near>1).mean():.1f}% of locations have multiple recipes within 0.01 nats of the best. "
                 "This is descriptive grid coverage, not a population probability.")
    texts.append(f"- {len(reversals[['config','K','language']].drop_duplicates())} of 1,792 analyzed language/recipe curves contain an observed increase; "
                 f"the largest is +{reversals.increase.max():.4f} nats. These are retained in the interpolation/unconstrained sensitivity and audited separately, not declared measurement noise.")
    for row in metrics.loc[metrics.model.eq("M2")].itertuples(index=False):
        texts.append(f"- Existing held-out M2 at cap {row.cap/1e9:g}B: mean/max decision regret {row.mean_regret:.4f}/{row.max_regret:.4f} nats, "
                     f"{row.within_001*100:.1f}% of evaluated budgets within 0.01 nats of the observed optimum.")
    at_cap = metrics.loc[metrics.cap.eq(CAPS[-1])].set_index("model")
    texts.append(f"- At the 2.2B cap, M1 has lower held-out allocation regret than M2 ({at_cap.loc['M1','mean_regret']:.4f} versus "
                 f"{at_cap.loc['M2','mean_regret']:.4f} mean nats), despite the original overall RMSE selection favoring M2. "
                 "The predictor's intended decision task needs its own evaluation.")
    body = """# Results 1 and 3: full-trajectory analysis

This is a descriptive reanalysis of the sealed repaired-LR Stage 1 release: 64
training trajectories, 400 nested readouts, one main-grid seed. The independent
seed panel remains separate. No manuscript source or Result 2 fit is changed.

## What the new analysis says

""" + "\n".join(texts) + f"""

The larger story is conditional: experts reduce the exposure-controlled
penalty across the studied curves, while longer training can reveal a positive
residual even at E32. Broader language coverage moves a compute-dependent
allocation boundary; it does not imply that every budget should use a higher
expert count. Mean-preserving expansion and protection of individual languages
are different policy objectives.

## Estimation and evidence boundaries

The selected smoothing penalty is **{strengths['smooth']:.8g}**. Each language's
learning curve uses six cubic B-spline coefficients constrained to be
nonincreasing, with a penalty on integrated squared curvature in normalized
log tokens. The strength is shared; losses and K/E interactions are not pooled
across recipes. Full-curve smoothing reduces dependence on adjacent readouts,
but it does not estimate untested active sizes or expert counts.

The strength is selected by interior-readout CV on 48 common-12 mean curves,
with equal trajectory weights. A separate nested check removes each common
interior horizon from every trajectory, tunes using only remaining horizons,
and predicts the omitted values. This tests within-trajectory reconstruction,
not new-trajectory or seed generalization. Nested RMSEs (nats/token):
""" + "\n".join(f"- {row.method}: {row.rmse:.6f}" for row in cv.itertuples(index=False)) + """

PCHIP is slightly more accurate than the regularized smoother in this nested
check. Smoothing is therefore not claimed as a prediction-accuracy win over
PCHIP; the scientific argument relies on conclusions surviving both methods.

Pressure is computed from paired language losses at the same realized D_t.
The horizontal axis is the amount of K12 training giving that exposure, not a
shared language-token count or D_total/K. All experts use common panel support;
the summary also restricts every active-size ladder to the same exposure
interval. Checkpoint markers are target observations minus an interpolated
reference. Bands include two neighboring smoothing penalties plus log-linear
and PCHIP interpolation, plus an unconstrained smooth fit using the same
penalty. They are estimator sensitivity, not confidence bands. The monotone
primary curve is an assumption about underlying learning; it cannot establish
that observed individual-language reversals are noise. The reversal audit
shows where this assumption matters.

Allocation uses the common 12 languages, all cap-feasible observed recipes,
and continuous compute evaluation within their shared trained support. Regret
is additional loss relative to the best of those recipes. The 0.005/0.01/0.02
nat tolerances are explicit practical-sensitivity thresholds, not statistical
significance tests. Compute is logical F=6*N_act*D_total; stored parameters are
an explicit constraint. Communication, latency, and hardware efficiency are
not modeled. Curves end when comparison support ends. A boundary at E32 does
not identify optimal sparsity beyond E32.

Expansion uses every incumbent: 12 languages for K12→K50 and 50 for K50→K200.
At each reference compute, the old recipe minimizes incumbent mean loss.
The target preserves that same mean, every individual reference loss, or every
reference loss within 0.01 nats. It jointly searches observed target recipes.
Different estimators also change the estimated reference quality; their spread
is full-procedure sensitivity. Dashed segments identify shorter trajectories
whose unobserved continuation might beat the reported feasible solution.
These are observed-support minima, not global optima or seed guarantees.
For interpolated curves that rise again, every polynomial root is searched to
find the first jointly feasible preservation point. A first crossing does not
promise that arbitrarily longer training will retain that quality.

The policy validation reads the existing frozen M0/M1/M2 K200 holdout fits.
All 16 target trajectories were held out; all 50 scoring languages are in the
registered fold. Primary regret is scored against log-linear held-out curves.
The comparison adds no model fitting and does not replace the original model
selection protocol. Repeated budgets and overlapping recipe pairs are not
independent observations. These post-hoc diagnostics should guide future
validation of the author's revised predictor, not be retuned into a new test.

## Figure reading order

1. **Pressure across training:** exposure-controlled effect and horizon/size
   dependence; gray tails extend beyond support shared by all four ladders.
2. **Coverage and allocation boundary:** the clearest multilingual allocation
   insight, with complete feasible-set support and estimator sensitivity.
3. **Allocation regret:** consequential choices versus nearly tied alternatives.
4. **Language expansion cost:** preservation objective, memory, and quality
   level all affect the required compute multiplier.
5. **Incumbent distribution:** average preservation can conceal regressions;
   shading is variation across languages, not estimation uncertainty.
6. **Held-out allocation regret:** whether the stored predictor supports advice.
7. **Individual-language reversals:** a check on monotonic smoothing assumptions.

For a main-paper revision, prioritize figures 1, 2, and 4. Keep the regret maps,
language distribution, and policy diagnostics as supporting evidence. Replace
continuous architecture recommendations only after the revised pooled predictor
passes decision validation and an architecture-to-storage mapping is declared.
No exact 2×-language transition is measured by the current tiers.

## Reproduce and inspect

Run `PYTHONPATH=src:. conda run -n bridge-train python -m
result_analysis.compute_allocation.trajectory_analysis` from the repository
root. All outputs are in this folder. `manifest.json` binds the release, code,
fit sources, and output hashes. CSV files retain curves, censoring, validation
predictions, and decision metrics. `index.html` is the figure gallery.

The implementation uses [SciPy B-splines](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html)
and [bounded linear least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html).
"""
    (OUT / "ANALYSIS.md").write_text(body)
    figures = [
        ("fig_pressure_continuous", "1. When does expert capacity reduce the penalty?", "Full learning curves at matched realized language exposure. Bands show estimator sensitivity; dots are checkpoint-anchored contrasts."),
        ("fig_coverage_allocation_boundary", "2. How does coverage change allocation?", "At a 0.4B stored cap, increasing coverage moves the boundary between 93M/E16 and 190M/E4. Every curve stays within complete comparison support."),
        ("fig_allocation_regret", "3. How costly is a different recipe?", "Cyan outlines mark recipe intervals within 0.01 nats of the best. A selected winner may have nearly equivalent alternatives. Rows are discrete architectures."),
        ("fig_language_expansion_cost", "4. What does adding languages cost?", "Compare mean preservation, individual protection, and a declared tolerance across quality levels. Dashed curves have unresolved shorter-trajectory continuations."),
        ("fig_incumbent_distribution", "5. Does every language keep its quality?", "All 50 incumbents, at the mean-preserving K50→K200 allocation. The shaded distributions are across languages, not confidence intervals."),
        ("fig_heldout_allocation_regret", "6. Can the stored model choose well?", "Post-hoc policy scoring on the frozen K200 trajectory holdout. The dotted line is a 0.01-nat practical tolerance."),
        ("fig_language_curve_reversals", "7. Can smoothing hide a real reversal?", "Three languages with the largest observed increases, selected descriptively. Monotone and unconstrained fits expose the assumption; these one-seed reversals need confirmation."),
    ]
    cards = "\n".join(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p>'
        f'<a href="{stem}.pdf"><img src="{stem}.png" alt="{html.escape(title)}"></a>'
        f'<p><a href="{stem}.pdf">Download PDF</a> · <a href="{stem}.png">Open PNG</a></p></section>'
        for stem, title, caption in figures)
    insight_list = "".join(f"<li>{html.escape(line[2:])}</li>" for line in texts)
    gallery = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 1: multilingual pressure and compute allocation</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;background:#f6f5f2;color:#20232b;margin:0}}main{{max-width:1180px;margin:auto;padding:36px 24px}}h1{{font-size:32px;line-height:1.2}}h2{{font-size:22px}}section{{background:white;border:1px solid #deddd9;border-radius:10px;padding:24px;margin:28px 0}}img{{width:100%;height:auto}}a{{color:#176f80}}li{{margin:.6em 0}}.subtitle{{color:#545967}}details{{margin:24px 0}}code{{font-size:13px}}</style>
<main><h1>Multilingual pressure and compute allocation</h1><p class="subtitle">Stage 1 · full-trajectory descriptive analysis · repaired-LR release</p>
<p><a href="ANALYSIS.md">Analysis and methods</a> · <a href="manifest.json">Provenance manifest</a></p>
<details open><summary><strong>Measured findings and interpretation</strong></summary><ul>{insight_list}</ul></details>
{cards}<p>One main-grid seed; readouts are nested. Logical compute excludes systems overhead. Curve bands show sensitivity, and unobserved architecture choices are not validated.</p></main></html>'''
    (OUT / "index.html").write_text(gallery)
    return texts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    frame = load_primary(ROOT)
    cohorts = {k: sorted(frame.loc[frame.K.eq(k), "language"].unique()) for k in (12, 50)}
    print("Selecting smoothing strength and evaluating omitted horizons", flush=True)
    strengths, cv = tune_smoothing(frame, cohorts[12])
    print("Selected strengths", strengths, flush=True)
    data = frame.loc[frame.K.isin([12, 50, 200]) & frame.language.isin(cohorts[50])]
    bank = {(f"{ladder}/E{e}", k): RecipeCurve(group, strengths)
            for (ladder, e, k), group in data.groupby(["ladder", "E", "K"])}
    print("Fitted", len(bank), "recipe matrices", flush=True)
    pressure, support, anchors, standardized = coverage_gap_sensitivity(frame, bank, cohorts[12], expert_counts=EXPERT_COLORS, methods=METHODS)
    for name, table in (("pressure_curves", pressure), ("pressure_support", support),
                        ("pressure_checkpoint_anchors", anchors), ("pressure_common_support_summary", standardized)):
        save_table(name, table)
    print("Derived coverage-gap sensitivity curves", flush=True)
    allocation, crossovers = allocation_analysis(bank, cohorts[12])
    print("Derived allocation regions and regret", flush=True)
    policies, distribution = expansion_analysis(bank, cohorts, caps=CAPS)
    save_table("expansion_policies", policies)
    save_table("incumbent_loss_changes", distribution)
    print("Derived expansion costs and individual protection", flush=True)
    metrics, decisions, margins, sources = heldout_validation(frame, bank, cohorts[50])
    print("Evaluated frozen held-out allocation predictions", flush=True)
    save_figure(plot_coverage_gap_sensitivity(pressure, support, anchors, expert_colors=EXPERT_COLORS), "fig_pressure_continuous")
    plot_allocations(allocation, bank)
    plot_crossovers(allocation)
    cost_figure, distribution_figure = plot_expansion(policies, distribution, caps=CAPS)
    save_figure(cost_figure, "fig_language_expansion_cost")
    save_figure(distribution_figure, "fig_incumbent_distribution")
    plot_validation(decisions)
    reversals = reversal_audit(bank)
    texts = write_report(strengths, cv, standardized, crossovers, allocation, policies, metrics, reversals)
    manifest = dict(status="post-hoc full-trajectory descriptive analysis and frozen-model policy diagnostic",
        release_manifest_sha256=sha256_file(ROOT / "result_analysis/stage1_abcd_jointmoe_x1_v1/release_manifest.json"),
        code_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in [Path(__file__),
            ROOT/"result_analysis/learning_curves.py", ROOT/"result_analysis/capacity_response/tables.py",
            ROOT/"result_analysis/capacity_response/sensitivity.py", ROOT/"result_analysis/language_preservation/sensitivity.py",
            Path(__file__).with_name("frontier.py")]},
        software_versions={name: version(name) for name in ("numpy", "pandas", "scipy", "matplotlib")},
        fit_sources=sources, smoothing_strengths=strengths, cohorts=cohorts,
        near_optimal_tolerances=[.005, .01, .02], heldout_policy_fold="K200_common_languages",
        counts=dict(pressure_rows=len(pressure), allocation_rows=len(allocation), expansion_policies=len(policies),
                    heldout_pairwise_rows=len(margins), main_trajectories=64, main_readouts=400),
        uncertainty="estimator sensitivity only; no independent-seed confidence intervals",
        output_sha256={p.name: sha256_file(p) for p in sorted(OUT.iterdir())
                       if p.suffix in (".csv", ".pdf", ".png", ".md", ".html")})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("\n".join(texts), flush=True)
    print("Gallery:", OUT / "index.html", flush=True)


if __name__ == "__main__":
    main()
