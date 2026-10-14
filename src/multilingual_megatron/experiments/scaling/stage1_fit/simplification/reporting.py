"""Complete scorecard for the frozen compact-law screen; never selects a law."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import execution as core
from .. import fit
from .. import repetition_campaign as reuse
from . import campaign
from . import model
from ..repetition_reporting import KEYS
from ..scoring import score


def report(plan_path):
    plan = campaign.load(plan_path)
    root = Path(plan_path).resolve().parent
    output = root/"report"
    if output.exists():
        raise ValueError("report is create-only")
    endpoint = pd.read_parquet(root/"endpoint.parquet")
    if "unigram_nll" not in endpoint:
        endpoint["unigram_nll"] = -endpoint.unigram_log_probability_sum/endpoint.token_count
    baseline_grid = pd.read_parquet(root/"baseline_grid.parquet")
    baseline_endpoint = pd.read_csv(root/"baseline_endpoint.csv").rename(columns={"observed": "adjusted_loss"})
    baseline_endpoint["trajectory_id"] = endpoint.trajectory_id.iloc[0]
    baseline_endpoint["readout_id"] = endpoint.readout_id.iloc[0]
    recent = pd.read_parquet(root/"recent_predictions.parquet")
    recent_metrics = pd.read_csv(root/"recent_metrics.csv")
    records, emitted, fits, inputs, horizon = [], [], [], [], []
    for cell in plan["cells"]:
        fold, name = cell["fold"], cell["model"]
        path = root/"cells"/f"{cell['cell_id']:03d}.json"
        if path.exists():
            result = core.read_json(path)
            if any(result[k] != cell[k] for k in ("cell_id", "fold", "model")) or result["plan_sha256"] != plan["sha256"]:
                raise ValueError("cell identity changed")
            inputs.append(reuse.binding(path))
        else:
            result = {"selected": None, "status": "missing", "audit": {}, "successful_starts": 0}
        selected = result["selected"]
        healthy = result["status"] == "healthy"
        for attempt in sorted((root/"attempts").glob(f"{cell['cell_id']:03d}-*.json")):
            record = core.read_json(attempt)
            if record["plan_sha256"] != plan["sha256"] or record["cell_id"] != cell["cell_id"]:
                raise ValueError("attempt identity changed")
            inputs.append(reuse.binding(attempt))
        coordinates = dict(zip(model.mapping(name)[1], selected["coordinates"])) if selected else {}
        if selected:
            coordinates["theta"] = (coordinates.get("phi_e32", coordinates["phi_e1"])-coordinates["phi_e1"])/np.log(32.)
        audit = result["audit"]
        fits.append({"fold": fold, "model": name, "status": result["status"],
                     "successful_starts": result["successful_starts"], "shared_parameters": len(cell["bounds"]),
                     "nuisance_parameters": 3*len(selected["language_coefficients"]) if selected else None,
                     "training_objective": selected["objective"] if selected else None,
                     "basin_members": audit.get("basin_members", 0),
                     "maximum_audit_prediction_difference": audit.get("maximum_prediction_difference"),
                     "projected_gradient_inf": audit.get("projected_gradient_inf"), **coordinates})
        testing = pd.read_parquet(root/"packets"/f"{fold}-test.parquet")
        target = endpoint[endpoint.language.isin(cell["membership"]["score_languages"])].copy()
        for kind, reference, old in (("grid", testing, baseline_grid), ("L26_69B", target, baseline_endpoint)):
            new = reference[KEYS+["adjusted_loss"]].copy()
            new["prediction"] = model.predict(reuse.problem(reference, cell["regularization"]), selected, name) if selected else np.nan
            new["fold"], new["model"], new["case"] = fold, name, kind
            new["weight"] = fit.trajectory_weights(reference.trajectory_id.tolist()) if kind == "grid" else 1/len(reference)
            new["D_total"] = reference.D_total.to_numpy()
            emitted.append(new)
            candidates = [(name, new, "candidate", healthy)]
            if name == model.MODELS[0]:
                candidates += [(baseline, group, "baseline", True) for baseline, group in old[old.fold.eq(fold)].groupby("model")]
                for baseline, group in recent[recent.fold.eq(fold) & recent.case.eq(kind)].groupby("model"):
                    status = recent_metrics[recent_metrics.fold.eq(fold) & recent_metrics.case.eq(kind) & recent_metrics.model.eq(baseline)]
                    if len(status) != 1:
                        raise ValueError("ambiguous recent comparator status")
                    candidates.append((baseline, group, "recent_candidate", bool(status.iloc[0].numerically_audited)))
            for predictor, frame, role, audited in candidates:
                value = score(frame, reference, endpoint=kind != "grid")
                records.append({"case": kind, "fold": fold, "model": predictor, "role": role,
                                "numerically_audited": audited, **value})
                if kind == "grid" and fold == "largest_N_act" and value["complete"] and (
                        role != "baseline" or predictor in ("J0-ND", "M5-ND", "M6-ND", "M6")):
                    aligned = frame.set_index(KEYS).loc[reference.set_index(KEYS).index]
                    error = aligned.prediction.to_numpy()-reference.adjusted_loss.to_numpy()
                    weights = fit.trajectory_weights(reference.trajectory_id.tolist())
                    for duration in sorted(reference.D_total.unique()):
                        mask = reference.D_total.to_numpy() == duration
                        w = weights[mask]/weights[mask].sum()
                        horizon.append({"fold": fold, "model": predictor, "D_total": int(duration),
                                        "rmse": float(np.sqrt(w @ error[mask]**2)),
                                        "bias": float(w @ error[mask]), "rows": int(mask.sum())})
    metrics = pd.DataFrame(records)
    gates = []
    for (kind, fold), group in metrics.groupby(["case", "fold"]):
        for scope, roles in (("original_51", ["baseline"]), ("including_recent", ["baseline", "recent_candidate"])):
            eligible = group[group.role.isin(roles) & group.complete & group.numerically_audited]
            best = eligible.loc[eligible.rmse.idxmin()]
            for row in group[group.role.eq("candidate")].itertuples():
                win = bool(row.complete and row.rmse < best.rmse-plan["protocol"]["score_win_tolerance_nats"])
                gates.append({"case": kind, "fold": fold, "scope": scope, "model": row.model,
                              "best_comparator": best.model, "comparator_rmse": best.rmse,
                              "candidate_rmse": row.rmse if row.complete else None,
                              "numerical_win": win, "audited_win": win and bool(row.numerically_audited)})
    primary = metrics[metrics.case.eq("grid")]
    means = []
    for name, group in primary.groupby("model"):
        complete = len(group) == 3 and group.complete.all()
        means.append({"model": name, "complete": bool(complete),
                      "mean_rmse": float(group.rmse.mean()) if complete else None,
                      "numerically_audited": bool(complete and group.numerically_audited.all())})
    output.mkdir()
    tables = {"metrics.csv": metrics, "gates.csv": pd.DataFrame(gates), "fits.csv": pd.DataFrame(fits),
              "mean_metrics.csv": pd.DataFrame(means), "horizon_metrics.csv": pd.DataFrame(horizon)}
    for filename, frame in tables.items():
        frame.to_csv(output/filename, index=False)
    pd.concat(emitted, ignore_index=True).to_parquet(output/"predictions.parquet", index=False)
    wins = {scope: {name: sum(g["audited_win"] for g in gates if g["scope"] == scope and g["model"] == name)
                   for name in model.MODELS} for scope in ("original_51", "including_recent")}
    result = {"plan_sha256": plan["sha256"], "status": "complete_screen" if all(f["status"] == "healthy" for f in fits) else "screen_with_unresolved_fits",
              "audited_wins_out_of_six": wins, "law_selected": False, "inputs": inputs,
              "all_cases_target_demonstrated": any(v == 6 for v in wins["including_recent"].values()),
              "evidence": plan["protocol"]["scientific_status"], "deferred": plan["protocol"]["deferred"],
              "outputs": [reuse.binding(output/n) for n in (*tables, "predictions.parquet")]}
    core.write_json_once(output/"report.json", result)
    return {k: v for k, v in result.items() if k not in ("inputs", "outputs")}
