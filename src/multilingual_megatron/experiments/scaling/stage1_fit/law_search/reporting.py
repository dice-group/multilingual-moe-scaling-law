"""Inner-only selector, then a separate seven-case scorecard and sensitivity report."""
from __future__ import annotations

from ..scoring import coefficients_predict
from ..scoring import checked_score
from ..scoring import predict_result

from pathlib import Path

import numpy as np
import pandas as pd

from .. import execution as io
from .. import fit
from . import data
from . import model
from . import replay
from ..atlas_comparison_workflow import model as native
from ..data import canonical_json_sha256
from .protocol import INNER
from .protocol import PRIMARY
from .protocol import SECONDARY
from .protocol import PROTOCOL
from .protocol import choose_procedure
from ..artifacts import binding
from ..artifacts import verify
from ..scoring import score


def load_result(root,plan,task_id):
    path = root/"results"/f"{task_id}.json"
    if not path.exists():
        return dict(certified=False,error="missing fit receipt",task_id=task_id)
    result = io.read_json(path)
    task = next(t for t in plan["tasks"] if t["id"] == task_id)
    if result["plan_sha256"] != plan["sha256"] or result["task_sha256"] != canonical_json_sha256(task):
        raise ValueError("fit receipt identity mismatch")
    for entry in result.get("dependencies",[]):
        verify(entry)
    if task["kind"]=="native":
        from .recovery import overlay
        result=overlay(root,plan,task_id,result)
    return result


def select(plan_path):
    from .state import load_plan
    plan = load_plan(plan_path)
    root = Path(plan_path).resolve().parent
    if (root/"selection.json").exists():
        raise ValueError("global procedure already frozen; no reselection")
    if any((root/"results"/f"{t['id']}.json").exists() for t in plan["tasks"] if t.get("model") == "selected"):
        raise ValueError("outer candidate fit already exists before selection")
    records,inputs = [],[]
    for fold in INNER:
        paths = {Path(e["path"]).name:verify(e) for e in plan["inner_scoring"][fold]}
        rows = pd.read_parquet(paths["rows.parquet"])
        with np.load(paths["sources.npz"],allow_pickle=False) as arrays:
            sources = model.Sources(**dict(arrays))
        inputs += plan["inner_scoring"][fold]
        for task in (t for t in plan["tasks"] if t["stage"] == "inner" and t["fold"] == fold):
            result = load_result(root,plan,task["id"])
            if not result.get("certified"):
                raise ValueError(f"inner task {task['id']} is missing or uncertified; no outer/L26 selection")
            pred = rows[data.KEYS+["adjusted_loss"]].copy()
            pred["prediction"] = coefficients_predict(result,rows,sources)
            records.append(dict(model=task["model"],tau=task["tau"],fold=fold,
                                certified=True,**checked_score(pred,rows)))
            inputs.append(binding(root/"results"/f"{task['id']}.json"))
    chosen = choose_procedure(records)
    result = chosen | dict(plan_sha256=plan["sha256"],inner_metrics=records,inputs=inputs,
        frozen_before_outer_scoring=True,outer_or_endpoint_responses_read=False,
        evidence=PROTOCOL["evidence"])
    io.write_json_once(root/"selection.json",result)
    print(chosen["chosen"])


def paired_bootstrap(candidate,baseline,replicates=10000):
    keys = data.KEYS
    a = candidate.set_index(keys).sort_index()
    b = baseline.set_index(keys).loc[a.index]
    if not np.isfinite(a.prediction).all() or not np.isfinite(b.prediction).all():
        return None
    squared = pd.DataFrame(dict(a=(a.prediction-a.adjusted_loss)**2,b=(b.prediction-b.adjusted_loss)**2)).reset_index()
    means = squared.groupby("trajectory_id")[["a","b"]].mean().to_numpy()
    rng = np.random.default_rng(PROTOCOL["seed"])
    sampled = rng.integers(len(means),size=(replicates,len(means)))
    errors = np.sqrt(means[sampled].mean(axis=1))
    gap = errors[:,1]-errors[:,0]
    return dict(unit="whole configuration",replicates=replicates,baseline_minus_candidate_ci95=np.quantile(gap,[.025,.975]).tolist(),
                probability_candidate_better=float(np.mean(gap>0)))


def endpoint_sensitivity(candidate,baseline,rows):
    a = candidate.set_index("language").loc[rows.language]
    b = baseline.set_index("language").loc[rows.language]
    squared = np.column_stack(((a.prediction-a.adjusted_loss)**2,(b.prediction-b.adjusted_loss)**2))
    if not np.isfinite(squared).all():
        return dict(complete=False)
    loo = np.sqrt((squared.sum(axis=0)-squared)/(len(squared)-1))
    strata = {}
    for name,values in (("family",rows.family.fillna("missing-singleton") if "family" in rows else rows.language),
                        ("script",rows.language.str.rsplit("_",n=1).str[-1]),
                        ("repetition",pd.Series(np.where(rows.D_t/rows.U_t>4,"R>4","R<=4")))):
        for label in sorted(set(values)):
            mask = np.asarray(values==label)
            rmse = np.sqrt(squared[mask].mean(axis=0))
            strata[f"{name}:{label}"] = dict(languages=int(mask.sum()),candidate_rmse=float(rmse[0]),baseline_rmse=float(rmse[1]))
    return dict(complete=True,scope="language-cohort sensitivity within ONE endpoint run; not seed confidence",
        leave_one_language_out_gap_range=[float((loo[:,1]-loo[:,0]).min()),float((loo[:,1]-loo[:,0]).max())],strata=strata)


def report(plan_path):
    from .state import load_plan
    from . import reproduction as law_search_reproduction
    plan = load_plan(plan_path,all_inputs=True)
    root = Path(plan_path).resolve().parent
    selection = io.read_json(root/"selection.json")
    if selection["plan_sha256"] != plan["sha256"] or not selection["frozen_before_outer_scoring"]:
        raise ValueError("outer scoring requires the immutable global selection")
    for entry in selection["inputs"]:
        verify(entry)
    output = root/"report"
    if output.exists():
        raise ValueError("scorecard is create-only")
    grid = pd.read_parquet(root/"scoring/grid.parquet")
    endpoint = pd.read_parquet(root/"scoring/endpoint.parquet")
    contract = io.read_json(root/"scoring/exposure_contract.json")
    inv = {(r["language"],r["K"]):r["U"] for r in contract["inventories"]}
    metadata = contract["metadata"]
    historic = pd.read_parquet(root/"scoring/baseline_grid.parquet")
    historic_endpoint = pd.read_csv(root/"scoring/baseline_endpoint.csv").rename(columns={"observed":"adjusted_loss"})
    historic_endpoint["trajectory_id"],historic_endpoint["readout_id"] = endpoint.trajectory_id.iloc[0],endpoint.readout_id.iloc[0]
    metrics,predictions,fit_summaries,inputs,ambiguity = [],[],[],[binding(root/"selection.json")],[]
    resource_accounting=root/"resource_accounting.json"
    if resource_accounting.exists():
        if io.read_json(resource_accounting)["plan_sha256"] != plan["sha256"]:
            raise ValueError("foreign CPU accounting receipt")
        inputs.append(binding(resource_accounting))
    for fold in (*PRIMARY,"all_grid",*SECONDARY):
        cases = (["grid","L26"] if fold in PRIMARY else ["L26"] if fold == "all_grid" else ["secondary"])
        tasks = [t for t in plan["tasks"] if t["fold"] == fold and
                 t["stage"] in ("baselines","final","secondary")]
        results = {t["id"]:load_result(root,plan,t["id"]) for t in tasks}
        for task_id,value in results.items():
            fit_summaries.append(dict(task_id=task_id,fold=fold,certified=bool(value.get("certified")),
                error=value.get("error"),geometry=value.get("geometry"),timed_out=value.get("timed_out",False)))
            path=root/"results"/f"{task_id}.json"
            if path.exists(): inputs.append(binding(path))
            if value.get("box_recovery_receipt"):
                inputs.append(value["box_recovery_receipt"])
        for case in cases:
            if case == "L26":
                languages = sorted(metadata) if fold == "all_grid" else plan["memberships"][fold]["score_languages"]
                rows = endpoint[endpoint.language.isin(languages)].reset_index(drop=True)
                universe = endpoint
            else:
                rows=data.subset(grid,plan["memberships"][fold],"test"); universe=grid
            universe=universe.drop(columns=list(data.RESPONSES & set(universe.columns)))
            if fold in PRIMARY:
                old = historic_endpoint if case == "L26" else historic
                for name,group in old[old.fold.eq(fold)].groupby("model"):
                    if name not in {*plan["protocol"]["shared"],*(s["name"] for s in plan["native_specs"])}:
                        continue
                    records = group[data.KEYS+["adjusted_loss","prediction"]].copy()
                    value=checked_score(records,rows,case=="L26")
                    metrics.append(dict(case=case,fold=fold,model=name,role="baseline",certified=True,**value))
                    predictions.append(records.assign(case=case,fold=fold,model=name,role="baseline"))
            for task in tasks:
                result = results[task["id"]]
                role = "candidate" if task["id"].startswith("selected-") else "control" if task["id"].startswith("control-") else "baseline"
                name = "selected" if role == "candidate" else task["id"].split("-")[1] if role == "control" else task.get("model",task.get("spec",{}).get("name"))
                records = predict_result(result,rows,universe,inv,metadata)
                metrics.append(dict(case=case,fold=fold,model=name,role=role,certified=bool(result.get("certified")),
                                    **checked_score(records,rows,case=="L26")))
                predictions.append(records.assign(case=case,fold=fold,model=name,role=role))
                if role == "candidate" and result.get("selected"):
                    basin_predictions = [predict_result(result,rows,universe,inv,metadata,selected=a).prediction.to_numpy()
                                         for a in result.get("near_equivalent",[])]
                    if basin_predictions:
                        spread = np.ptp(np.stack(basin_predictions),axis=0)
                        ambiguity.append(dict(case=case,fold=fold,basins=len(basin_predictions),max_prediction_range=float(spread.max())))
    frame = pd.DataFrame(metrics)
    all_predictions = pd.concat(predictions,ignore_index=True)
    gates,uncertainty = [],[]
    required = frame[frame.case.ne("secondary")]
    for (case,fold),group in required.groupby(["case","fold"]):
        candidate = group[group.role.eq("candidate")].iloc[0]
        baselines = group[group.role.eq("baseline")]
        eligible = baselines[baselines.complete & baselines.certified]
        best = eligible.loc[eligible.rmse.idxmin()] if len(eligible) else None
        complete = len(baselines)==55 and bool((baselines.complete & baselines.certified).all())
        win = bool(candidate.complete and candidate.certified and best is not None and candidate.rmse<best.rmse-1e-6)
        gates.append(dict(case=case,fold=fold,baseline_models=len(baselines),all_baselines_complete=complete,
            best_baseline=None if best is None else best.model,candidate_rmse=float(candidate.rmse) if candidate.complete else None,
            baseline_rmse=None if best is None else float(best.rmse),win_among_complete=win,audited_all_baseline_win=win and complete,
            unresolved_baselines=baselines.loc[~(baselines.complete & baselines.certified),"model"].tolist()))
        if best is not None and candidate.complete:
            local=all_predictions[all_predictions.case.eq(case)&all_predictions.fold.eq(fold)]
            a,b=local[local.role.eq("candidate")],local[local.role.eq("baseline")&local.model.eq(best.model)]
            uncertainty.append(dict(case=case,fold=fold,**(paired_bootstrap(a,b) if case=="grid" else
                endpoint_sensitivity(a,b,endpoint[endpoint.language.isin(a.language)].reset_index(drop=True)))))
    stability=[]
    for task in (t for t in plan["tasks"] if t["stage"]=="stability"):
        result=load_result(root,plan,task["id"])
        entry=dict(task=task["id"],kind=task["kind"],certified=bool(result.get("certified")),rho=task.get("rho"),
                   objective=(result.get("selected") or {}).get("objective"),geometry=result.get("geometry"))
        if result.get("selected"):
            covariates=endpoint.drop(columns=list(data.RESPONSES & set(endpoint.columns)))
            forecast=predict_result(result,endpoint,covariates,inv,metadata)
            entry.update(checked_score(forecast,endpoint,True))
            predictions.append(forecast.assign(case="stability_L26",fold=task["id"],model=result["model"],role="diagnostic"))
        stability.append(entry)
        path=root/"results"/f"{task['id']}.json"
        if path.exists(): inputs.append(binding(path))
    complete=len(gates)==7 and all(g["all_baselines_complete"] for g in gates) and all(s["certified"] for s in fit_summaries+stability)
    wins=sum(g["audited_all_baseline_win"] for g in gates)
    fragile=any((s.get("geometry") or {}).get("contacts") or
                ((s.get("geometry") or {}).get("nuisance_adjusted_rank",0)<(s.get("geometry") or {}).get("free_shared",0))
                for s in fit_summaries if s["task_id"].startswith("selected-"))
    fragile=fragile or any(u.get("baseline_minus_candidate_ci95",[1.])[0]<=0 for u in uncertainty)
    outcome="incomplete" if not complete else "partial" if wins<7 else "fragile" if fragile else "achieved"
    output.mkdir()
    frame.to_csv(output/"metrics.csv",index=False)
    pd.DataFrame(gates).to_csv(output/"gates.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(output/"predictions.parquet",index=False)
    result=dict(status=outcome,plan_sha256=plan["sha256"],selected=selection["chosen"],required_cases=7,audited_wins=wins,
        gates=gates,fit_summaries=fit_summaries,uncertainty=uncertainty,near_equivalent_prediction_ranges=ambiguity,
        stability=stability,unresolved_native_outside_certified_panel=plan["unresolved"],
        evidence=PROTOCOL["evidence"],fresh_confirmation="proposal only; not performed",inputs=inputs,
        reporter_sources=[binding(Path(module.__file__)) for module in (data,model,replay,native,fit,law_search_reproduction)],
        reporter_source=binding(Path(__file__)),
        reporting_amendment="Native log-space replay, independent new shared/pooled-fit replay, and separately audited roundoff-only box repairs; no law, bound, or tolerance change")
    io.write_json_once(output/"report.json",result)
    print(dict(status=outcome,audited_wins=wins,selected=selection["chosen"]))
