"""Observed-support compute and stored-capacity allocation review, 2026-09-05.

The default review interpolates each observed architecture without a fitted law.
An optional pooled-fit preview reads the existing audited M2 snapshot without
refitting it. Outputs are exploratory and do not replace manuscript artifacts.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from result_analysis.evidence import sha256 as sha256_file
from result_analysis.capacity_response.tables import load_primary
from result_analysis.language_preservation.sensitivity import restoration_policy

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs/compute_allocation_review_20260905"
EXPERTS = (1, 4, 16, 32)
DISPLAY_CAPS = (0.2, 0.4, 0.7, 1.2, 2.2, 5.1, 9.8)
COLORS = {1: "#39316b", 4: "#337e9c", 16: "#db8c25", 32: "#c34d53"}


def prepare_curves(frame, cohort, coverages=(12, 50, 200)):
    rows = frame.loc[frame.K.isin(coverages) & frame.language.isin(cohort)]
    rows = rows.groupby(["ladder", "E", "K", "D_total"], as_index=False).agg(
        loss=("adjusted_loss", "mean"), N_act=("N_act", "first"),
        N_total=("N_total", "first"), n_languages=("language", "nunique"),
    )
    if not rows.n_languages.eq(len(cohort)).all():
        raise ValueError("incomplete cohort")
    rows["F"] = 6 * rows.N_act.astype(float) * rows.D_total.astype(float) / 1e18
    curves = {}
    for (ladder, e, k), group in rows.groupby(["ladder", "E", "K"]):
        group = group.sort_values("F")
        if not np.all(np.diff(group.loss) < 0):
            raise ValueError("nonmonotone loss curve")
        curves[(f"{ladder}/E{e}", k)] = group
    return rows, curves


def loss_at(curve, compute, method="linear_log"):
    if compute < curve.F.min() or compute > curve.F.max():
        raise ValueError("compute outside observed curve")
    if method == "pchip_log":
        return float(PchipInterpolator(np.log(curve.F), curve.loss)(np.log(compute)))
    return float(np.interp(np.log(compute), np.log(curve.F), curve.loss))


def invert_loss(curve, target, method="linear_log"):
    if target >= curve.loss.iloc[0]:
        return float(curve.F.iloc[0]), "already_reached_at_first_readout"
    if target < curve.loss.iloc[-1]:
        return None, "not_reached_within_support"
    if method == "pchip_log":
        fn = PchipInterpolator(np.log(curve.F), curve.loss)
        compute = float(np.exp(brentq(lambda x: fn(x)-target,
                                     np.log(curve.F.min()), np.log(curve.F.max()))))
    else:
        compute = float(np.exp(np.interp(target, curve.loss.to_numpy()[::-1],
                                        np.log(curve.F.to_numpy()[::-1]))))
    # A separate forward evaluation verifies the inversion result.
    if not np.isclose(loss_at(curve, compute, method), target, rtol=0, atol=2e-11):
        raise ValueError("loss inversion failed forward verification")
    return compute, "bracketed"


def best_at(curves, k, budget, cap, method="linear_log"):
    candidates = []
    for (config, coverage), curve in curves.items():
        if coverage != k or curve.N_total.iloc[0] > cap:
            continue
        if not curve.F.min() <= budget <= curve.F.max():
            continue
        candidates.append((loss_at(curve, budget, method), curve.N_total.iloc[0], config))
    if not candidates:
        raise ValueError("no supported feasible configurations")
    ordered = sorted(candidates)
    loss, stored, config = ordered[0]
    curve = curves[config, k]
    return dict(config=config, loss=loss, N_total=int(stored), E=int(curve.E.iloc[0]),
                N_act=int(curve.N_act.iloc[0]), D_total=budget*1e18/(6*curve.N_act.iloc[0]),
                runner_up_gap=ordered[1][0]-loss if len(ordered)>1 else None)




def save(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(OUT/f"{name}.{extension}", bbox_inches="tight", dpi=190)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_primary(ROOT)
    cohort = set(frame.loc[frame.K.eq(12), "language"])
    observations, curves = prepare_curves(frame, cohort)
    observations.to_csv(OUT/"common12_observed_curves.csv", index=False)
    lower = max(c.F.min() for c in curves.values())
    upper = min(c.F.max() for c in curves.values())
    grid = sorted(set(np.geomspace(lower, upper, 41)) | {2., 4.})
    caps = sorted(set(observations.N_total) | {int(c*1e9) for c in DISPLAY_CAPS})
    decisions = []
    for method in ["linear_log", "pchip_log"]:
        for k in (12, 50, 200):
            for cap in caps:
                for compute in grid:
                    decisions.append(dict(K=k, parameter_cap=cap, F=compute, method=method,
                                          **best_at(curves, k, compute, cap, method)))
    decisions = pd.DataFrame(decisions)
    decisions.to_csv(OUT/"allocation_decisions.csv", index=False)
    recipes = decisions.loc[decisions.F.isin([2., 4.]) & decisions.parameter_cap.isin(
        [int(c*1e9) for c in DISPLAY_CAPS]) & decisions.method.eq("linear_log")]
    recipes.to_csv(OUT/"display_budget_recipes.csv", index=False)
    transitions = []
    capped_curves = [c for c in curves.values() if c.N_total.iloc[0]<=400_000_000]
    cap_lower = max(c.F.min() for c in capped_curves)
    cap_upper = min(c.F.max() for c in capped_curves)
    for method in ["linear_log","pchip_log"]:
        for k in [12,50,200]:
            a,b = curves["A/E16",k],curves["B/E4",k]
            fn = lambda f: loss_at(a,f,method)-loss_at(b,f,method)
            crossing = brentq(fn,cap_lower,cap_upper)
            # Validate the observed pair is the full feasible-set winner on either side.
            before=best_at(curves,k,crossing*.999,400_000_000,method)["config"]
            after=best_at(curves,k,crossing*1.001,400_000_000,method)["config"]
            if (before,after)!=("A/E16","B/E4"):
                raise ValueError("pair crossing is not the feasible-set decision transition")
            transitions.append(dict(K=k,method=method,F_transition=crossing,
                                    below="A/E16",above="B/E4",parameter_cap=400_000_000,
                                    support_min=cap_lower,support_max=cap_upper))
    pd.DataFrame(transitions).to_csv(OUT/"allocation_transition_compute.csv",index=False)
    policies = []
    for method in ["linear_log", "pchip_log"]:
        for base_k, target_k in [(12, 50), (50, 200), (12, 200)]:
            for budget in [2., 4.]:
                for cap in caps:
                    policies.append(restoration_policy(curves, base_k, target_k, budget, cap, method))
    policies = pd.DataFrame(policies)
    policies.to_csv(OUT/"restoration_policies.csv", index=False)

    # Expand K50-to-K200 scoring to every incumbent language as a sensitivity.
    cohort50 = set(frame.loc[frame.K.eq(50), "language"])
    _, curves50 = prepare_curves(frame, cohort50, (50,200))
    full50 = pd.DataFrame([
        restoration_policy(curves50,50,200,budget,int(cap*1e9),method)
        for method in ["linear_log","pchip_log"] for budget in [2.,4.] for cap in DISPLAY_CAPS
    ])
    full50.to_csv(OUT/"common50_restoration_sensitivity.csv",index=False)

    # Inspect individual languages at the common-50 mean-restoration choice.
    anchor = full50.loc[full50.baseline_F.eq(2.) & full50.parameter_cap.eq(2_200_000_000)
                        & full50.method.eq("linear_log")].iloc[0]
    per_language = []
    for language in sorted(cohort50):
        individual = frame.loc[frame.language.eq(language) & frame.ladder.eq("B")
                               & frame.E.eq(32) & frame.K.isin([50,200])].copy()
        individual["F"] = 6*individual.N_act.astype(float)*individual.D_total.astype(float)/1e18
        base_curve = individual.loc[individual.K.eq(50)].sort_values("F").rename(columns={"adjusted_loss":"loss"})
        target_curve = individual.loc[individual.K.eq(200)].sort_values("F").rename(columns={"adjusted_loss":"loss"})
        if not np.all(np.diff(target_curve.loss)<0):
            raise ValueError("individual protection requires monotone curves")
        reference = loss_at(base_curve,2.)
        target = loss_at(target_curve,anchor.target_F)
        protected_compute,status = invert_loss(target_curve,reference)
        per_language.append(dict(language=language,reference_loss=reference,
                                 delta_at_mean_restoration=target-reference,
                                 protection_F=protected_compute,status=status))
    per_language = pd.DataFrame(per_language)
    if not np.isclose(per_language.delta_at_mean_restoration.mean(),0,atol=2e-11,rtol=0):
        raise ValueError("common-50 aggregation does not reproduce mean restoration")
    per_language.to_csv(OUT/"common50_individual_protection.csv",index=False)
    cost_grid = np.linspace(1.,2.7,81)
    individual_changes = []
    for language in sorted(cohort50):
        individual = frame.loc[frame.language.eq(language) & frame.ladder.eq("B")
                               & frame.E.eq(32) & frame.K.eq(200)].copy().sort_values("D_total")
        compute = 6*individual.N_act.to_numpy(float)*individual.D_total.to_numpy(float)/1e18
        reference = per_language.loc[per_language.language.eq(language),"reference_loss"].iloc[0]
        individual_changes.append(np.interp(np.log(cost_grid*2),np.log(compute),individual.adjusted_loss)-reference)
    changes = np.asarray(individual_changes)
    preservation = pd.DataFrame(dict(compute_multiplier=cost_grid,
                                     mean_delta=changes.mean(axis=0),worst_delta=changes.max(axis=0)))
    preservation.to_csv(OUT/"common50_preservation_curves.csv",index=False)

    # Same absolute target loss when comparing dense-only and expert-allowed costs.
    # The target is each dense-only K50 optimum at 2 EFLOPs, under the same cap.
    quality_matched = []
    dense = {key:curve for key,curve in curves50.items() if curve.E.iloc[0]==1}
    for cap,method in [(c,m) for c in DISPLAY_CAPS for m in ["linear_log","pchip_log"]]:
        baseline = best_at(dense,50,2.,int(cap*1e9),method)
        for family, pool in [("dense_only",dense),("experts_allowed",curves50)]:
            winners = {}
            for k in [50,200]:
                options = []
                for (config,coverage), curve in pool.items():
                    if coverage!=k or curve.N_total.iloc[0]>cap*1e9: continue
                    compute,status=invert_loss(curve,baseline["loss"],method)
                    if compute is not None: options.append((compute,config,status))
                winners[k]=min(options) if options else None
            if all(winners.values()):
                f50,c50,s50=winners[50];f200,c200,s200=winners[200]
                quality_matched.append(dict(parameter_cap=cap*1e9,family=family,method=method,
                                           loss_target=baseline["loss"],F50=f50,F200=f200,
                                           K50_config=c50,K200_config=c200,
                                           K50_status=s50,K200_status=s200,
                                           compute_multiplier=f200/f50))
    pd.DataFrame(quality_matched).to_csv(OUT/"quality_matched_dense_sparse_costs.csv",index=False)

    # At the primary optimized target, isolate changes allowed to the baseline.
    routes = []
    for cap in [int(c*1e9) for c in DISPLAY_CAPS]:
        base = best_at(curves, 12, 2., cap)
        base_ladder, base_e = base["config"].split("/E")
        for rule in ["tokens_only", "active_and_tokens", "experts_and_tokens", "joint"]:
            subset = {(config, k): curve for (config, k), curve in curves.items()
                      if (rule == "joint" or
                          rule == "tokens_only" and config == base["config"] or
                          rule == "active_and_tokens" and config.endswith(f"/E{base_e}") or
                          rule == "experts_and_tokens" and config.startswith(f"{base_ladder}/"))}
            # All rules retain the original baseline winner; target loss is identical.
            row = restoration_policy(subset, 12, 200, 2., cap, "linear_log")
            if row["base_config"] != base["config"]:
                raise ValueError("route comparison changed baseline")
            routes.append(dict(rule=rule, **row))
    routes = pd.DataFrame(routes)
    routes.to_csv(OUT/"allowed_allocation_routes.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    palette = ListedColormap([COLORS[e] for e in EXPERTS])
    norm = BoundaryNorm(np.arange(-.5,4.5), palette.N)
    # Discrete rows are selected caps, not a continuous memory axis.
    for ax, k in zip(axes, [12, 50, 200]):
        selected = decisions.loc[decisions.K.eq(k) & decisions.method.eq("linear_log")]
        data = []
        for cap in DISPLAY_CAPS:
            group = selected.loc[selected.parameter_cap.eq(int(cap*1e9))].set_index("F").loc[grid]
            data.append([EXPERTS.index(e) for e in group.E])
        x = np.log2(grid)
        edges = np.r_[x[0]-(x[1]-x[0])/2, (x[:-1]+x[1:])/2, x[-1]+(x[-1]-x[-2])/2]
        ax.pcolormesh(edges, np.arange(len(DISPLAY_CAPS)+1)-.5, data, cmap=palette, norm=norm)
        for i, cap in enumerate(DISPLAY_CAPS):
            group = selected.loc[selected.parameter_cap.eq(int(cap*1e9))].set_index("F").loc[grid]
            alternate = decisions.loc[decisions.K.eq(k) & decisions.parameter_cap.eq(int(cap*1e9))
                                      & decisions.method.eq("pchip_log")].set_index("F").loc[grid]
            configs = list(group.config)
            for j,(primary,other) in enumerate(zip(configs,alternate.config)):
                if primary!=other:
                    ax.add_patch(Rectangle((edges[j],i-.5),edges[j+1]-edges[j],1.,
                                           facecolor="none",edgecolor="white",hatch="////",linewidth=0))
            starts = [0] + [j for j in range(1,len(configs)) if configs[j]!=configs[j-1]] + [len(configs)]
            for a,b in zip(starts[:-1], starts[1:]):
                if b-a >= 6:
                    label=f"{group.N_act.iloc[a]/1e6:.0f}M / E{group.E.iloc[a]}"
                    ax.text((edges[a]+edges[b])/2, i, label,
                            color="white", ha="center", va="center", fontsize=8, clip_on=True,
                            bbox=dict(facecolor="black", alpha=.15, edgecolor="none", pad=1))
        ax.set(title=f"K{k}", xticks=np.log2([2.,3.,4.]), xticklabels=[2,3,4],
               xlabel="Logical training compute (EFLOPs)", yticks=range(len(DISPLAY_CAPS)),
               yticklabels=[f"{c:g}" for c in DISPLAY_CAPS])
    axes[0].set_ylabel("Stored-parameter cap (B; selected caps)")
    fig.legend(handles=[Patch(color=COLORS[e], label=f"E{e}") for e in EXPERTS]+
               [Patch(facecolor="white",edgecolor="gray",hatch="////",label="Interpolation-sensitive")],
               loc="outside upper center", ncol=5, frameon=False)
    save(fig, "fig_compute_capacity_map")

    fig, axes = plt.subplots(1,2,figsize=(10,3.9),layout="constrained")
    selected = policies.loc[policies.baseline_F.eq(2.) & policies.method.eq("linear_log")]
    axes[0].plot(cost_grid,preservation.mean_delta,color="#c34d53",label="Mean of 50 incumbent languages")
    axes[0].plot(cost_grid,preservation.worst_delta,color="#39316b",label="Worst incumbent-language change")
    for ratio,color,label in [(anchor.target_F/2,"#c34d53","Mean restored"),
                              (per_language.protection_F.max()/2,"#39316b","All 50 restored")]:
        axes[0].scatter([ratio],[0],color=color,zorder=4)
        mean_label=label=="Mean restored"
        axes[0].annotate(f"{label}\n{ratio:.2f}x compute",(ratio,0),
                         xytext=(-5,-35) if mean_label else (6,14),
                         textcoords="offset points",ha="right" if mean_label else "left",
                         fontsize=8,color=color)
    axes[0].set(xlabel="Compute / K50 reference compute",ylabel="K200 minus K50 loss (nats/token)",
                title="(a) K50 to K200, fixed 190M / E32")
    axes[0].axhline(0,color="black",linewidth=.6,linestyle="--")
    axes[0].legend(fontsize=8)
    p2 = recipes.loc[recipes.parameter_cap.eq(400_000_000) & recipes.K.isin([12,200])]
    for k,color in [(12,"#39316b"),(200,"#c34d53")]:
        group = p2.loc[p2.K.eq(k)].sort_values("F")
        axes[1].plot(group.F,group.N_act/1e6,"o-",color=color,label=f"K{k}")
        for row in group.itertuples(index=False):
            axes[1].annotate(f"E{row.E}; {row.D_total/1e9:.2f}B tokens",
                             (row.F,row.N_act/1e6),xytext=(0,8 if k==12 else -17),
                             textcoords="offset points",fontsize=8,ha="center")
    axes[1].set(xlim=(1.65,4.35),ylim=(60,220),xticks=[2,4],xlabel="Logical compute (EFLOPs)",
                ylabel="Selected active parameters (M)",title="(b) At a 0.4B stored-parameter cap")
    axes[1].legend(fontsize=8,loc="lower right")
    save(fig,"fig_compute_expansion_and_recipes")

    # Give all options at one fixed cost/coverage so diminishing value is visible.
    fig, axes = plt.subplots(1,2,figsize=(10,3.8),sharey=True,layout="constrained")
    for ax,k in zip(axes,[12,200]):
        fixed = []
        for (config,coverage),curve in curves.items():
            if coverage != k: continue
            fixed.append(dict(config=config,E=int(curve.E.iloc[0]),
                              stored=curve.N_total.iloc[0]/1e9,loss=loss_at(curve,2.)))
        fixed = pd.DataFrame(fixed).sort_values("stored")
        best = float("inf"); frontier = []
        for row in fixed.itertuples(index=False):
            if row.loss < best:
                frontier.append(row); best=row.loss
        ax.step([r.stored for r in frontier],[r.loss for r in frontier],where="post",color="black",alpha=.45)
        for e,color in COLORS.items():
            group=fixed.loc[fixed.E.eq(e)]
            ax.scatter(group.stored,group.loss,color=color,s=40,label=f"E{e}",zorder=3)
        for row in frontier:
            offset = {"A/E4":(-7,8),"B/E1":(10,-13),"A/E16":(-13,12),"B/E4":(12,-13)}.get(row.config,(0,7))
            ax.annotate(row.config,(row.stored,row.loss),xytext=offset,textcoords="offset points",fontsize=7,ha="center")
        ax.set(xscale="log",xlabel="Configured stored parameters (B)",title=f"K{k}, 2 logical EFLOPs")
    axes[0].set_ylabel("Common-12 adjusted loss (lower is better)")
    axes[1].legend(fontsize=8)
    save(fig,"fig_storage_compute_pareto")

    paired = decisions.pivot(index=["K","parameter_cap","F"], columns="method", values="config")
    manifest = dict(status="post-hoc observed-support allocation review; no law fitting or policy validation",
                    release_manifest_sha256=sha256_file(ROOT / "result_analysis/stage1_abcd_jointmoe_x1_v1/release_manifest.json"),source_sha256=sha256_file(Path(__file__)),
                    loader_sha256=sha256_file(ROOT/"result_analysis/capacity_response/tables.py"),
                    cohort=sorted(cohort),common_compute_support=[lower,upper],
                    decision_rows=len(decisions),interpolation_choice_disagreements=int(
                        paired.linear_log.ne(paired.pchip_log).sum()),
                    interpolation_choice_comparisons=len(paired),
                    common50_mean_restoration_F=float(anchor.target_F),
                    common50_regressing_at_mean_match=int(per_language.delta_at_mean_restoration.gt(1e-10).sum()),
                    common50_max_regression=float(per_language.delta_at_mean_restoration.max()),
                    common50_fixed_B_E32_all_language_restoration_F=float(per_language.protection_F.max())
                    if per_language.protection_F.notna().all() else None,
                    output_sha256={p.name:sha256_file(p) for p in sorted(OUT.iterdir())
                                   if p.suffix in (".csv",".pdf",".png")})
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Common compute support",lower,upper)
    print(recipes.loc[recipes.parameter_cap.isin([400_000_000,2_200_000_000])][
        ["K","parameter_cap","F","config","N_act","D_total","runner_up_gap"]].to_string(index=False))
    print(selected.loc[selected.parameter_cap.isin([int(c*1e9) for c in DISPLAY_CAPS])][
        ["base_K","target_K","parameter_cap","base_config","target_config","compute_multiplier",
         "N_act_multiplier","D_total_multiplier","status","unknown_continuations_below_winner"]].to_string(index=False))
    print("Interpolation choices differ",manifest["interpolation_choice_disagreements"],"/",len(paired))


def pooled_fit_preview():
    """Contrast interpolation with an existing all-data predictor, without fitting.

    The diagnostic uses every checkpoint in the displayed trajectories and a
    predictor fitted across the release. It is deliberately separate from the
    observed-support manifest and from the author's ongoing model selection.
    """
    from matplotlib.lines import Line2D
    from multilingual_megatron.experiments.scaling.stage1_fit import fit

    destination = OUT / "pooled_fit_preview"
    destination.mkdir(parents=True, exist_ok=True)
    fit_root = ROOT / "result_analysis/stage1_scaling_fit_jointmoe_x1_v1"
    report_path = fit_root / "reports/l26_e32_m2_prediction.json"
    report = json.loads(report_path.read_text())
    sources = {str(report_path.relative_to(ROOT)): sha256_file(report_path)}

    def verified(entry):
        original = Path(entry["path"])
        path = fit_root / original.parent.name / original.name
        digest = sha256_file(path)
        if digest != entry["file_sha256"]:
            raise ValueError(f"fit source changed: {path}")
        sources[str(path.relative_to(ROOT))] = digest
        return json.loads(path.read_text())

    verified(report["candidate_report"])
    result = verified(report["all_data_result"])
    audit = verified(report["all_data_audit"])
    if (result["fit_identity"] != dict(bound_version=0, fold_id="all_data",
            model="M2", objective="huber", replicate_id=None)
            or result["status"] != "reduced" or not result["selected"]["success"]
            or audit["status"] != "complete"
            or audit["result_sha256"] != result["result_sha256"]):
        raise ValueError("unexpected all-data fit identity or incomplete audit")
    selected = result["selected"]
    parameters = fit.SharedParameters(**selected["shared_parameters"])
    coefficients = {row["language"]: row for row in selected["language_coefficients"]}
    frame = load_primary(ROOT)
    cohort = sorted(frame.loc[frame.K.eq(12), "language"].unique())
    observed, curves = prepare_curves(frame, cohort)
    exposure = (frame.loc[frame.K.isin([12, 50, 200]) & frame.language.isin(cohort)]
                .assign(share=lambda x: x.D_t / x.D_total)
                .groupby(["K", "language"]).share.agg(["min", "max", "mean"]))
    if ((exposure["max"]-exposure["min"])/exposure["mean"]).max() > 3e-4:
        raise ValueError("exposure share varies across readouts")

    def prediction(config, k, compute):
        curve = curves[config, k]
        compute = np.asarray(compute, dtype=float)
        if np.any(compute < curve.F.min()) or np.any(compute > curve.F.max()):
            raise ValueError("preview prediction outside observed trajectory support")
        n = float(curve.N_act.iloc[0])
        e = int(curve.E.iloc[0])
        tokens = compute * 1e18 / (6*n)
        values = []
        for language in cohort:
            c = coefficients[language]
            values.append(fit.predict("M2", parameters, n,
                tokens*float(exposure.loc[(k, language), "mean"]), e, k,
                c["L_inf"], c["A"], c["B"]))
        return np.mean(values, axis=0)

    # In-sample residuals are diagnostics, never a held-out validation score.
    residuals = observed.copy()
    residuals["config"] = residuals.ladder + "/E" + residuals.E.astype(str)
    residuals["pooled_prediction"] = [float(prediction(r.config, r.K, r.F))
        for r in residuals.itertuples(index=False)]
    residuals["residual"] = residuals.loss - residuals.pooled_prediction
    residuals.to_csv(destination / "checkpoint_residuals.csv", index=False)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.6), layout="constrained")
    configs = ("A/E16", "B/E4")
    architecture_colors = {"A/E16": "#db8c25", "B/E4": "#337e9c"}
    method_styles = {"linear_log": "--", "pchip_log": ":", "pooled_M2": "-"}
    evaluations, crossings = [], []
    for column, k in enumerate((12, 50, 200)):
        upper_ax, lower_ax = axes[:, column]
        for config in configs:
            curve = curves[config, k]
            grid = np.geomspace(curve.F.min(), curve.F.max(), 201)
            upper_ax.plot(grid, prediction(config, k, grid),
                          color=architecture_colors[config], linewidth=1.8)
            upper_ax.scatter(curve.F, curve.loss, color=architecture_colors[config],
                             s=23, edgecolors="white", linewidth=.4, zorder=3)
        feasible = [c for (_, coverage), c in curves.items()
                    if coverage == k and c.N_total.iloc[0] <= 400_000_000]
        lower = max(c.F.min() for c in feasible)
        upper = min(c.F.max() for c in feasible)
        grid = np.geomspace(lower, upper, 301)
        for method, style in method_styles.items():
            def margin(f):
                if method == "pooled_M2":
                    return float(prediction(configs[0], k, f)-prediction(configs[1], k, f))
                return loss_at(curves[configs[0], k], f, method)-loss_at(curves[configs[1], k], f, method)
            delta = np.asarray([margin(f) for f in grid])
            lower_ax.plot(grid, delta, style, color="#39316b",
                          linewidth=1.9 if method == "pooled_M2" else 1.2,
                          alpha=1 if method == "pooled_M2" else .65)
            brackets = np.flatnonzero(delta[:-1]*delta[1:] < 0)
            roots = [float(brentq(margin, grid[i], grid[i+1])) for i in brackets]
            crossings.append(dict(K=k, method=method, lower_F=lower, upper_F=upper,
                crossover_F=roots[0] if len(roots)==1 else None, n_crossings=len(roots),
                margin_at_2=margin(2.), margin_at_4=margin(4.)))
            for f, d in zip(grid, delta):
                evaluations.append(dict(K=k, method=method, F=f, loss_A16_minus_B4=d))
        lower_ax.axhline(0, color="black", linewidth=.7)
        lower_ax.set(xlim=(lower, upper), xscale="log", xticks=[1.2, 2, 3, 4],
                     xticklabels=[1.2, 2, 3, 4], xlabel="Logical compute (EFLOPs)")
        upper_ax.set(title=f"K{k}", xscale="log", xlabel="Logical compute (EFLOPs)")
    axes[0, 0].set_ylabel("Common-12 adjusted loss\n(lower is better)")
    axes[1, 0].set_ylabel("Loss: 93M/E16 minus 190M/E4\n(negative favors 93M/E16)")
    fig.legend(handles=[
        Line2D([], [], color=architecture_colors[c], marker="o", linestyle="none",
               label=f"{c}: observed checkpoints") for c in configs] + [
        Line2D([], [], color="#39316b", linestyle=style,
               label={"linear_log": "Log-linear interpolation",
                      "pchip_log": "PCHIP interpolation",
                      "pooled_M2": "Existing pooled M2 fit"}[method])
        for method, style in method_styles.items()],
        loc="outside upper center", ncol=3, frameon=False, fontsize=8)
    for suffix in ("pdf", "png"):
        fig.savefig(destination / f"fig_pooled_loss_and_allocation.{suffix}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(evaluations).to_csv(destination / "allocation_margin_curves.csv", index=False)
    boundary = pd.DataFrame(crossings)
    boundary.to_csv(destination / "crossover_comparison.csv", index=False)
    manifest = dict(status="descriptive existing-fit preview; no refit or held-out decision validation",
        release_manifest_sha256=sha256_file(ROOT / "result_analysis/stage1_abcd_jointmoe_x1_v1/release_manifest.json"),
        source_sha256=sha256_file(Path(__file__)), evaluator_sha256=sha256_file(Path(fit.__file__)),
        fit_sources=sources, fit_identity=result["fit_identity"],
        result_sha256=result["result_sha256"], cohort=cohort,
        uncertainty="none; line differences are estimator sensitivity, not confidence intervals",
        crossover_scope="pairwise A/E16 versus B/E4 inside the common 0.4B-cap feasible support",
        output_sha256={p.name: sha256_file(p) for p in sorted(destination.iterdir())
                       if p.suffix in (".csv", ".pdf", ".png")})
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(boundary.to_string(index=False))
    print("In-sample mean-curve RMSE", float(np.sqrt(np.mean(residuals.residual**2))))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled-fit-preview", action="store_true",
                        help="read the existing audited all-data fit; do not refit")
    if parser.parse_args().pooled_fit_preview:
        pooled_fit_preview()
    else:
        main()
