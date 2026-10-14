"""Full-coverage primary and L26 scorecard; never selects a law using outcomes."""
from __future__ import annotations

from .scoring import score

from pathlib import Path

import numpy as np
import pandas as pd

from . import execution as core, fit, repetition_campaign as campaign, repetition_model as model

KEYS = ["trajectory_id", "readout_id", "language"]


def report(plan_path):
    plan = campaign.load(plan_path)
    root = Path(plan_path).resolve().parent
    output = root/"report"
    if output.exists():
        raise ValueError("report is create-only")
    grid = pd.read_parquet(root/"training_grid.parquet")
    endpoint = pd.read_parquet(root/"endpoint.parquet")
    if "unigram_nll" not in endpoint:
        endpoint["unigram_nll"] = -endpoint.unigram_log_probability_sum/endpoint.token_count
    historical = pd.read_parquet(root/"baseline_grid.parquet")
    endpoint_baselines = pd.read_csv(root/"baseline_endpoint.csv").rename(columns={"observed": "adjusted_loss"})
    endpoint_baselines["trajectory_id"], endpoint_baselines["readout_id"] = endpoint.trajectory_id.iloc[0], endpoint.readout_id.iloc[0]
    records, emitted, summaries, inputs = [], [], [], []
    for cell in plan["cells"]:
        fold, name = cell["fold"], cell["model"]
        attempts = []
        for task in plan["tasks"]:
            if task["cell_id"] != cell["cell_id"]:
                continue
            path = root/"fit"/f"{task['task_id']:03d}.json"
            value = core.read_json(path)
            if value["plan_sha256"] != plan["sha256"] or value["task_id"] != task["task_id"]:
                raise ValueError("attempt identity changed")
            inputs.append(campaign.binding(path))
            attempts.append(value)
        audit_path = root/"audit"/f"{cell['cell_id']:03d}.json"
        audit = core.read_json(audit_path)
        if audit["plan_sha256"] != plan["sha256"] or audit["cell_id"] != cell["cell_id"]:
            raise ValueError("audit identity changed")
        inputs.append(campaign.binding(audit_path))
        try:
            chosen = campaign.selected_attempt(attempts)
        except ValueError:
            chosen = None
        healthy = bool(chosen and audit.get("success") and audit.get("basin_members", 0) >= 2
                       and audit.get("shared_prediction_jacobian_rank") == audit.get("free_shared_coordinates")
                       and audit.get("selected_task_id") == chosen["task_id"])
        summaries.append({"fold": fold, "model": name, "numerically_audited": healthy,
            "successful_starts": sum(bool(a["success"]) for a in attempts),
            "rate": chosen["coordinates"][15] if chosen else None,
            "rho": chosen["coordinates"][16] if chosen and len(chosen["coordinates"]) == 17 else 0.,
            "zeta": chosen["coordinates"][14] if chosen else None})
        testing = campaign.partition(grid, cell["membership"], "test")
        target = endpoint[endpoint.language.isin(cell["membership"]["score_languages"])].copy()
        for kind, reference, old in (("grid", testing, historical), ("L26_69B", target, endpoint_baselines)):
            new = reference[KEYS+["adjusted_loss"]].copy()
            new["prediction"] = (model.predict(campaign.problem(reference, cell["regularization"]),
                reference.U_t.to_numpy(), chosen) if chosen else np.nan)
            new["fold"], new["model"], new["case"] = fold, name, kind
            emitted.append(new)
            records.append({"case": kind, "fold": fold, "model": name, "role": "candidate",
                            "numerically_audited": healthy, **score(new, reference, endpoint=kind != "grid")})
            if name == plan["protocol"]["models"][0]:
                for baseline, group in old[old.fold.eq(fold)].groupby("model"):
                    records.append({"case": kind, "fold": fold, "model": baseline, "role": "baseline",
                                    "numerically_audited": True, **score(group, reference, endpoint=kind != "grid")})
    metrics = pd.DataFrame(records).sort_values(["case", "fold", "rmse", "model"])
    gates = []
    for (kind, fold), group in metrics.groupby(["case", "fold"]):
        baselines = group[group.role.eq("baseline") & group.complete]
        best = baselines.loc[baselines.rmse.idxmin()]
        for row in group[group.role.eq("candidate")].itertuples():
            numerical_win = bool(row.complete and row.rmse < best.rmse-plan["protocol"]["score_win_tolerance_nats"])
            gates.append({"case": kind, "fold": fold, "model": row.model, "best_baseline": best.model,
                "baseline_rmse": float(best.rmse), "candidate_rmse": float(row.rmse) if row.complete else None,
                "numerical_win": numerical_win, "audited_win": numerical_win and bool(row.numerically_audited)})
    output.mkdir()
    metrics.to_csv(output/"metrics.csv", index=False)
    pd.DataFrame(gates).to_csv(output/"gates.csv", index=False)
    pd.DataFrame(summaries).to_csv(output/"fits.csv", index=False)
    pd.concat(emitted, ignore_index=True).to_parquet(output/"predictions.parquet", index=False)
    wins = {name: sum(g["audited_win"] for g in gates if g["model"] == name) for name in plan["protocol"]["models"]}
    result = {"status": "target_not_demonstrated", "plan_sha256": plan["sha256"],
        "primary_and_endpoint_audited_wins": wins, "cases_per_law": len(gates)//len(plan["protocol"]["models"]),
        "deferred_requirements": plan["protocol"]["deferred"], "law_selected": False,
        "evidence": plan["protocol"]["scientific_status"], "inputs": inputs,
        "outputs": [campaign.binding(output/n) for n in ("metrics.csv", "gates.csv", "fits.csv", "predictions.parquet")]}
    core.write_json_once(output/"report.json", result)
    print(pd.DataFrame(gates).to_string(index=False))
