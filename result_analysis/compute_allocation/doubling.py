"""Frozen-fit size/data allocation interpretation; no fitting or new training.

Run: python -m result_analysis.compute_allocation.doubling --config PATH
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from result_analysis import evidence as io


def regime(d, u):
    return np.where(2*d <= u, "before", np.where(d < u, "crossing", "repeated"))


def bracket(values, target):
    values = np.asarray(values, float)
    exact = np.flatnonzero(np.isclose(values, target, rtol=1e-10, atol=0))
    if len(exact):
        return int(exact[0]), int(exact[0]), "observed"
    j = int(np.clip(np.searchsorted(values, target), 1, len(values)-1))
    status = "interpolated" if values[0] < target < values[-1] else "extrapolated"
    return j-1, j, status


def stored_size(geometry, experts, target):
    g = geometry[geometry.E.eq(experts)].sort_values("N_act")
    lo, hi, status = bracket(g.N_act, target)
    a, b = g.iloc[lo], g.iloc[hi]
    fraction = 0. if lo == hi else (target-a.N_act)/(b.N_act-a.N_act)
    return float(a.N_total+fraction*(b.N_total-a.N_total)), status, (a.ladder, b.ladder)


def exposure_status(group, target):
    values = group.D_t.to_numpy()
    if np.any(np.isclose(values, target, rtol=1e-8, atol=0)):
        return "observed"
    return "interpolated" if values.min() <= target <= values.max() else "extrapolated"


def combined_status(statuses):
    return max(statuses, key={"observed": 0, "interpolated": 1, "extrapolated": 2}.get)


def build_panel(grid, coverages):
    languages = sorted(set.intersection(*(set(grid.loc[grid.K.eq(k), "language"]) for k in coverages)))
    geometry = grid[["ladder", "E", "N_act", "N_total"]].drop_duplicates()
    if geometry.duplicated(["ladder", "E"]).any():
        raise ValueError("ambiguous architecture counts")
    panel = grid[grid.K.isin(coverages)&grid.language.isin(languages)].copy().reset_index(drop=True)
    if not panel.groupby(["trajectory_id", "readout_id"]).language.nunique().eq(len(languages)).all():
        raise ValueError("incomplete common cohort")
    groups = {key: g.sort_values("D_t") for key, g in grid.groupby(["ladder", "E", "K", "language"])}
    # Validate that D_t doubling corresponds to a duration increase at fixed sampling.
    for _, group in panel.groupby(["trajectory_id", "language"]):
        fractions = group.D_t/group.D_total
        np.testing.assert_allclose(fractions, fractions.iloc[0], rtol=3e-4, atol=0)
        if group.U_t.nunique() != 1:
            raise ValueError("inventory changes within trajectory")
    annotations = []
    for row in panel.itertuples():
        total, size_status, ladders = stored_size(geometry, row.E, 2*row.N_act)
        support = [size_status]
        for ladder in set(ladders):
            g = groups[(ladder, row.E, row.K, row.language)]
            np.testing.assert_allclose(g.U_t, row.U_t, rtol=1e-10)
            np.testing.assert_allclose(g.D_t/g.D_total, row.D_t/row.D_total, rtol=3e-4)
            support.append(exposure_status(g, row.D_t))
        data_status = exposure_status(groups[(row.ladder,row.E,row.K,row.language)], 2*row.D_t)
        size_status = combined_status(support)
        annotations.append(dict(doubled_N_total=total, size_support=size_status,
            data_support=data_status, support=combined_status([size_status,data_status])))
    panel = pd.concat([panel,pd.DataFrame(annotations)],axis=1)
    panel["regime"] = regime(panel.D_t, panel.U_t)
    panel["base_logical_compute"] = 6*panel.N_act*panel.D_total
    panel["doubled_size_stored_ratio"] = panel.doubled_N_total/panel.N_total
    return panel, geometry, languages, groups


def aggregate(frame, keys, values):
    # Nested weighting avoids giving longer trajectories extra weight.
    per_readout = frame.groupby(keys+["trajectory_id","readout_id"], observed=True)[values].mean().reset_index()
    per_config = per_readout.groupby(keys+["trajectory_id"], observed=True)[values].mean().reset_index()
    summary = per_config.groupby(keys, observed=True)[values].mean().reset_index()
    counts = frame.groupby(keys, observed=True).agg(rows=("language","size"),
        languages=("language","nunique"), configurations=("trajectory_id","nunique")).reset_index()
    return summary.merge(counts,on=keys,validate="one_to_one")


def interpolate_loss(group, target, method):
    if exposure_status(group,target) == "extrapolated":
        return np.nan
    x, y = np.log(group.D_t.to_numpy()), group.adjusted_loss.to_numpy()
    return float(np.interp(np.log(target),x,y) if method == "linear_log" else
                 PchipInterpolator(x,y,extrapolate=False)(np.log(target)))


def observed_pairs(panel, geometry, groups, fits, predict):
    records = []
    for row in panel.itertuples():
        g = geometry[geometry.E.eq(row.E)].sort_values("N_act")
        larger = g[g.N_act > row.N_act]
        if larger.empty:
            continue
        target = larger.iloc[0]
        ratio = target.N_act/row.N_act
        same = groups[(row.ladder,row.E,row.K,row.language)]
        other = groups[(target.ladder,row.E,row.K,row.language)]
        if "extrapolated" in (exposure_status(same,ratio*row.D_t), exposure_status(other,row.D_t)):
            continue
        point = panel.loc[[row.Index]].copy()
        big = point.assign(N_act=target.N_act,N_total=target.N_total)
        more = point.assign(D_t=ratio*point.D_t,D_total=ratio*point.D_total)
        for form, fit in fits.items():
            delta = float(predict(fit,form,more).prediction.iloc[0]-predict(fit,form,big).prediction.iloc[0])
            for method in ("linear_log","pchip_log"):
                truth = interpolate_loss(same,ratio*row.D_t,method)-interpolate_loss(other,row.D_t,method)
                records.append(dict(model=form,K=row.K,language=row.language,
                    trajectory_id=row.trajectory_id,readout_id=row.readout_id,
                    architecture_pair=f"{row.ladder}->{target.ladder}", compute_ratio=ratio,
                    method=method,predicted_advantage=delta,observed_advantage=truth,
                    agreement=float(np.sign(delta)==np.sign(truth)),
                    regret=max(0., -truth) if delta>0 else max(0.,truth),
                    near_tie=abs(truth)<.001))
    return pd.DataFrame(records)


def refit_predictions(result, name, rows):
    from multilingual_megatron.experiments.scaling.stage1_fit import fit
    from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws import model, replay
    from multilingual_megatron.experiments.scaling.stage1_fit.law_search import model as shared
    from multilingual_megatron.experiments.scaling.stage1_fit.problems import problem
    regularization = result["regularization"]
    if "strength" in regularization:
        regularization = fit.M5M6Regularization(**regularization).as_dict()
    current = model.Model(name, problem(rows, regularization), rows.U_t.to_numpy(),
                          stored_parameters=rows.N_total.to_numpy())
    current.require_positive_exponents = False
    first = shared.predict(current, result["selected"])
    np.testing.assert_allclose(first, replay.predict(name, result["selected"], rows), atol=1e-10, rtol=0)
    return rows.assign(prediction=first)


def run(config_path):
    from types import SimpleNamespace
    from multilingual_megatron.experiments.scaling.stage1_fit.artifacts import binding
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    fits, inputs = {}, [binding(config_path)]
    if cfg.get("refit_plan"):
        from multilingual_megatron.experiments.scaling.stage1_fit.paper_refit_workflow import inputs as refit_io, runner, recovery
        refit_path = Path(cfg["refit_plan"]).resolve()
        refit = runner.load(refit_path)
        report_path = Path(cfg["refit_report"]).resolve()
        report = refit_io.read_sealed(report_path)
        if report["plan_sha256"] != refit["sha256"]:
            raise ValueError("foreign refit report")
        overrides = recovery.verified_overrides(refit_path,
            refit_io.local_path(refit_path.parent, report["recovery"]) if report.get("recovery") else None)
        inputs.extend([binding(refit_path), binding(report_path)])
        input_path = refit_io.local_path(refit_path.parent, refit["input"])
        grid = refit_io.read_training(input_path, "all_grid")[0]
        manifest = refit_io.load(input_path)
        inputs.extend([binding(input_path), binding(refit_io.local_path(input_path.parent, manifest["scoring"]["grid.parquet"]))])
        numerical_sources = [binding(refit_io.local_path(refit_path.parent, p)) for p in refit["source"]]
        reporting = SimpleNamespace(predictions=refit_predictions)
        for form in cfg["forms"]:
            task, = [t for t in refit["tasks"] if t["stage"] == "certify" and
                     (t["model"], t["fold"], t["arm"]) == (form, "all_grid", "lbfgsb_box")]
            index = task["id"]
            result_path, result = overrides.get(index, (refit_path.parent / "results" / f"{index:05d}.json",
                                                         runner.receipt(refit_path, refit, index)))
            if not result.get("certified"):
                raise ValueError(f"uncertified replacement fit: {form}")
            fits[form] = result
            inputs.append(binding(result_path))
    else:
        path = Path(cfg["campaign_plan"]).resolve()
        # Historical mode retains its original immutable campaign contract.
        sys.path.insert(0,str(path.parent/"source"))
        campaign = importlib.import_module("stage1_fit.coverage_laws.campaign")
        reporting = importlib.import_module("stage1_fit.coverage_laws.reporting")
        plan = campaign.load(path,all_inputs=True)
        inputs.extend([binding(path),plan["scoring"]["grid.parquet"]])
        numerical_sources = plan["source"]
        for form in cfg["forms"]:
            task, fit = reporting.fitted_result(path.parent,plan,form,"all_grid")
            if not fit.get("certified"):
                raise ValueError(f"uncertified all-grid fit: {form}")
            fits[form] = fit
            inputs.append(binding(campaign.result_path(path.parent,task)))
        grid = pd.read_parquet(plan["scoring"]["grid.parquet"]["path"])
    panel, geometry, languages, groups = build_panel(grid,cfg["coverages"])
    output = Path(cfg["output_root"]).resolve()
    output.mkdir(parents=True,exist_ok=False)
    source = [binding(Path(__file__).resolve()),binding(Path(io.__file__).resolve()),*numerical_sources]
    io.write_json(output/"plan.json",dict(config=cfg,inputs=inputs,source=source,languages=languages,
        rows=len(panel),no_fitting=True,coefficients_frozen=True,
        sampling_fraction_tolerance=3e-4,
        maximum_sampling_fraction_relative_span=float(panel.assign(f=panel.D_t/panel.D_total).groupby(["K","language"]).f.agg(lambda x:x.max()/x.min()-1).max()),
        excluded="K1 excluded to retain common12; no L26 counterfactual or fresh-data intervention"))
    panel.to_parquet(output/"panel.parquet",index=False)
    geometry.to_csv(output/"architecture_mapping.csv",index=False)
    frames = []
    for form, fit in fits.items():
        base = reporting.predictions(fit,form,panel).prediction.to_numpy()
        big = panel.assign(N_act=2*panel.N_act,N_total=panel.doubled_N_total)
        more = panel.assign(D_t=2*panel.D_t,D_total=2*panel.D_total)
        size = reporting.predictions(fit,form,big).prediction.to_numpy()
        data = reporting.predictions(fit,form,more).prediction.to_numpy()
        frames.append(panel.assign(model=form,base_prediction=base,size_prediction=size,
            data_prediction=data,gain_size=base-size,gain_data=base-data,
            advantage=data-size,size_preferred=(data>size).astype(float),
            size_harm=(size>base).astype(float),data_harm=(data>base).astype(float)))
    predictions = pd.concat(frames,ignore_index=True)
    predictions.to_parquet(output/"predictions.parquet",index=False)
    values = ["gain_size","gain_data","advantage","size_preferred","size_harm","data_harm"]
    summary = aggregate(predictions,["model","K","regime","support"],values)
    summary.to_csv(output/"summary.csv",index=False)
    pairs = observed_pairs(panel,geometry,groups,fits,reporting.predictions)
    pairs.to_parquet(output/"observed_pairs.parquet",index=False)
    pairs_summary = aggregate(pairs,["model","K","method"],
        ["predicted_advantage","observed_advantage","agreement","regret","near_tie"])
    pairs_summary.to_csv(output/"observed_pair_summary.csv",index=False)
    key = ["trajectory_id","readout_id","language","K","regime","support"]
    wide = predictions.pivot(index=key,columns="model",values="advantage")
    robust = wide.reset_index()[key].copy()
    robust["all_agree"] = ((wide>0).all(axis=1)|(wide<0).all(axis=1)).to_numpy().astype(float)
    robust["robust_size"] = (wide.min(axis=1)>cfg["near_tie_nats"]).to_numpy().astype(float)
    robust["robust_data"] = (wide.max(axis=1)<-cfg["near_tie_nats"]).to_numpy().astype(float)
    robust["spread"] = (wide.max(axis=1)-wide.min(axis=1)).to_numpy()
    aggregate(robust,["K","regime","support"],["all_agree","robust_size","robust_data","spread"]).to_csv(output/"robustness.csv",index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,k in zip(axes,cfg["coverages"]):
        for i,form in enumerate(cfg["forms"]):
            s = summary[(summary.model==form)&(summary.K==k)&(summary.support!="extrapolated")]
            s = s.set_index("regime").reindex(["before","crossing","repeated"])
            ax.plot(np.arange(3)+(i-1.5)*.04,s.advantage,marker="o",label=form)
        ax.axhline(0,color="black",lw=.7); ax.set_title(f"K={k}")
        ax.set_xticks(range(3),["Before","Crossing","Repeated"],rotation=20)
    axes[0].set_ylabel("Size gain − data gain (nats)\npositive favors size")
    axes[1].legend(fontsize=6)
    fig.suptitle("Model-implied doubling gains within joint support (interpolated)")
    fig.tight_layout(); fig.savefig(output/"doubling.pdf"); fig.savefig(output/"doubling.png",dpi=160); plt.close(fig)
    for entry in inputs+source:
        if io.sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("input or source changed during analysis")
    io.write_json(output/"receipt.json",dict(completed=True,inputs=inputs,source=source,
        artifacts={p.name:io.sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
        languages=languages,rows=len(panel),prediction_rows=len(predictions),observed_pair_rows=len(pairs),
        caveats=["developmental all-grid predictions, not independent validation",
                 "continuous architecture interpolation is not an observed trainable model",
                 "constant unique inventory; duration doubling, not fresh data",
                 "logical compute is not GPU-hours; memory feasibility not established",
                 "no statistical significance or uncertainty intervals"] ))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    run(parser.parse_args().config)
