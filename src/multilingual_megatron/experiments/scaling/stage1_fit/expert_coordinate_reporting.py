"""Side-study-only reporting after numerical certification; no paper-table writes."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import execution as core, expert_coordinate_side_study as s
from .m6_size_duration_workflow import model as m, planning as p, reporting as r


def predictions(plan, cell, selected):
    manifest = plan["folds"][cell["fold"]]["manifest"]
    frame = pd.read_parquet(manifest["input"]["fit_rows_path"])
    membership, frames = manifest["membership"], []
    for split in ("train", "test"):
        rows = frame[frame.trajectory_id.isin(membership[f"{split}_trajectory_ids"])].copy()
        if membership[f"{split}_readout_ids"]:
            rows = rows[rows.readout_id.isin(membership[f"{split}_readout_ids"])]
        if split == "test" and membership["score_languages"]:
            rows = rows[rows.language.isin(membership["score_languages"])]
        rows["prediction"] = m.predict(s.coordinate_problem(r.features(rows), cell["coordinate"]), selected)
        rows = r.diagnosis.annotate(rows, model=f"side:{cell['model']}:{cell['coordinate']}",
            fold=cell["fold"], split=split, delta=manifest["fit"]["delta"])
        frames.append(rows.assign(law=cell["model"], coordinate=cell["coordinate"]))
    return pd.concat(frames, ignore_index=True)


def policy_metrics(plan, selected):
    """Same frozen joint-story policy grid, physical resources and interpolants.

    Only this side-study predictor callback differs. Repeated compute points are
    decisions on dependent curves, not independent samples or new training runs.
    """
    from scipy.interpolate import PchipInterpolator
    from result_analysis.compute_allocation import policy_validation as j
    rows = pd.read_parquet(p.diagnosis.bound_file(plan["decision_inputs"]))
    recipes = []
    for (ladder, experts), group in rows.groupby(["ladder", "E"]):
        recipes.append(dict(name=f"{ladder}/E{experts}", n=float(group.N_act.iloc[0]),
            stored=float(group.N_total.iloc[0]), e=int(experts), rows=group,
            lower=6*group.N_act.iloc[0]*group.D_total.min()/1e18,
            upper=6*group.N_act.iloc[0]*group.D_total.max()/1e18))
    recipes.sort(key=lambda c: (c["stored"], c["name"]))
    grid = np.geomspace(max(c["lower"] for c in recipes), min(c["upper"] for c in recipes), 81)
    languages = sorted(plan["cohorts"]["incumbent50"])
    observed = {method: [] for method in ("linear_log", "pchip_log", "smooth")}
    predicted = {name: [] for name in selected}
    for recipe in recipes:
        query = grid*1e18/(6*recipe["n"])
        features, truth = [], {method: [] for method in observed}
        for language in languages:
            group = recipe["rows"].loc[recipe["rows"].language.eq(language)].sort_values("D_total")
            tokens, loss = group.D_total.to_numpy(), group.adjusted_loss.to_numpy()
            components = j.interpolate_exposure(tokens, group[j.COMPONENTS].to_numpy(), query)
            f = pd.DataFrame(components, columns=j.COMPONENTS)
            f["N_act"], f["D_total"], f["E"], f["K"], f["language"] = recipe["n"], query, recipe["e"], 200, language
            features.append(f)
            truth["linear_log"].append(np.interp(np.log(query), np.log(tokens), loss))
            truth["pchip_log"].append(PchipInterpolator(np.log(tokens), loss, extrapolate=False)(np.log(query)))
            truth["smooth"].append(j.LearningCurve.fit(tokens, loss, plan["smoothing_penalty"])(query))
        features = r.features(pd.concat(features, ignore_index=True))
        for method in observed:
            observed[method].append(np.asarray(truth[method]))
        for name, (coordinate, coefficients) in selected.items():
            predicted[name].append(m.predict(s.coordinate_problem(features, coordinate), coefficients).reshape(len(languages), len(grid)))
    metrics, decisions = [], []
    for cohort_name, cohort in plan["cohorts"].items():
        positions = [languages.index(language) for language in cohort]
        for method, values in observed.items():
            truth = np.asarray(values)[:, positions].mean(axis=1)
            for name, values in predicted.items():
                prediction = np.asarray(values)[:, positions].mean(axis=1)
                for cap in j.CAPS:
                    indices = [i for i, c in enumerate(recipes) if c["stored"] <= cap]
                    score = j.allocation_metrics(truth[indices], prediction[indices])
                    metrics.append(dict(model=name, fold=j.FOLD, cohort=cohort_name, reference=method, cap=cap,
                        candidate_count=len(indices), compute_points=len(grid), mean_regret=float(score["regret"].mean()),
                        max_regret=float(score["regret"].max()), winner_accuracy=float(score["winner_matches"].mean()),
                        pairwise_margin_rmse=score["pairwise_margin_rmse"], resolved_pairwise_accuracy=score["resolved_pairwise_accuracy"]))
                    for k, budget in enumerate(grid):
                        chosen = indices[score["chosen_indices"][k]]
                        decisions.append(dict(model=name, cohort=cohort_name, reference=method, cap=cap,
                            F=float(budget), chosen=recipes[chosen]["name"], regret=float(score["regret"][k])))
    return pd.DataFrame(metrics), pd.DataFrame(decisions)


def report(plan_path):
    plan = s.load(plan_path)
    reduction = s.receipt(plan, "reduction.json")
    audits = [s.receipt(plan, f"audits/{i:02d}.json") for i in range(72)]
    for i, value in enumerate(audits):
        if value["reduction_sha256"] != reduction["sha256"] or value["task"] != reduction["audit_tasks"][i]:
            raise ValueError("audit identity changed")
    statuses = []
    for cell, reduced in zip(plan["cells"], reduction["cells"]):
        subset = audits[cell["cell_id"]*4:cell["cell_id"]*4+4]
        statuses.append(dict(cell_id=cell["cell_id"], law=cell["model"], coordinate=cell["coordinate"], fold=cell["fold"],
            successful_starts=reduced["successful"], healthy_audits=sum(v["status"] == "healthy" for v in subset),
            known_minimum_passed=reduced["known_minimum"]["passed"],
            status="certified" if reduced["known_minimum"]["passed"] and all(v["status"] == "healthy" for v in subset) else "unresolved"))
    frames, variants, selected = [], {}, {}
    for law in s.PROTOCOL["models"]:
        for coordinate in s.PROTOCOL["coordinates"]:
            key = f"side:{law}:{coordinate}"
            cells = [c for c in plan["cells"] if (c["model"], c["coordinate"]) == (law, coordinate)]
            certified = all(statuses[c["cell_id"]]["status"] == "certified" for c in cells)
            variants[key] = "certified" if certified else "unresolved"
            if certified:
                for cell in cells:
                    coefficients = reduction["cells"][cell["cell_id"]]["selected"]
                    frames.append(predictions(plan, cell, coefficients))
                    if cell["fold"] == "K200_common_languages":
                        selected[key] = (coordinate, coefficients)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    comparisons, metrics = {}, {}
    if frames:
        heldout = frame[frame.split.eq("test")]
        for name, group in heldout.groupby("model"):
            per = group.groupby(["fold", "trajectory_id"]).squared_error.mean().groupby("fold").mean().pow(.5)
            metrics[name] = per.to_dict() | {"mean": float(per.mean())}
        for law in s.PROTOCOL["models"]:
            keys = [f"side:{law}:{coordinate}" for coordinate in s.PROTOCOL["coordinates"]]
            if all(variants[k] == "certified" for k in keys):
                # Reuse the existing paired trajectory bootstrap, changing labels only.
                pair = frame[frame.model.isin(keys)].copy()
                pair["model"] = pair.model.map({keys[0]: "raw", keys[1]: m.MODEL})
                stats = r.metrics_and_bootstrap(pair, s.PROTOCOL["bootstrap_replicates"], s.PROTOCOL["bootstrap_seed"])
                comparisons[law] = dict(raw=keys[0], transformed=keys[1],
                    mean_delta_hat_minus_raw=stats["paired_intervals"][0] | {"candidate": keys[1], "reference": keys[0]},
                    bootstrap_unit=stats["unit"], interpretation=stats["interpretation"],
                    fold_delta_hat_minus_raw={f: metrics[keys[1]][f]-metrics[keys[0]][f] for f in p.FOLDS})
    output = Path(plan["output_root"])/"reports"
    if output.exists():
        raise ValueError("side-study reports are create-only")
    output.mkdir()
    pd.DataFrame(statuses).to_csv(output/"numerical_status.csv", index=False)
    paths = [output/"numerical_status.csv"]
    if frames:
        frame.to_parquet(output/"predictions.parquet", index=False)
        r.analysis.residual_groups(frame).to_csv(output/"residual_groups.csv", index=False)
        horizons = heldout.groupby(["model", "fold", "E", "K", "D_total", "trajectory_id"])[["squared_error", "residual"]].mean()
        horizons = horizons.groupby(["model", "fold", "E", "K", "D_total"]).mean().reset_index()
        horizons["rmse"] = np.sqrt(horizons.squared_error)
        horizons.to_csv(output/"horizon_metrics.csv", index=False)
        horizons.query("E == 32 and D_total == 32000442368").to_csv(output/"e32_long_tails.csv", index=False)
        policy, decisions = policy_metrics(plan, selected)
        policy.to_csv(output/"policy_metrics.csv", index=False)
        decisions.to_csv(output/"policy_decisions.csv", index=False)
        paths += [output/name for name in ("predictions.parquet", "residual_groups.csv", "horizon_metrics.csv", "e32_long_tails.csv", "policy_metrics.csv", "policy_decisions.csv")]
    # Recheck byte-identical historical evidence after all new calculations.
    for binding in plan["protected"]:
        p.diagnosis.bound_file(binding)
    lines = ["# Expert-coordinate SIDE STUDY", "", s.PROTOCOL["evidence"]+".",
        "", "Historical results and paper tables are untouched. Unresolved variants are not scored.", "",
        "| Side-study law | Coordinate | Largest N | E32 | K200 | Mean |", "|---|---|---:|---:|---:|---:|"]
    for name, values in metrics.items():
        _, law, coordinate = name.split(":")
        lines.append(f"| {law} | {coordinate} | "+" | ".join(f"{values[f]:.8f}" for f in (*p.FOLDS, "mean"))+" |")
    lines += ["", "Unresolved: "+", ".join(name for name, status in variants.items() if status == "unresolved"), "",
        "A coordinate-only transformation cannot add training-horizon dependence. "
        "E1/E32 are anchored, so endpoint prediction changes come from refitting. "
        "Only four expert levels are observed; the published saturation constants were not re-estimated. "
        "Bootstrap intervals are descriptive and conditional on the fitted laws, not independent-seed or search-corrected evidence.", ""]
    core.write_text_once(output/"summary.md", "\n".join(lines))
    paths.append(output/"summary.md")
    return p.write_sealed(output/"report.json", dict(plan_sha256=plan["sha256"], reduction_sha256=reduction["sha256"],
        status="complete" if all(v == "certified" for v in variants.values()) else "completed_with_unresolved_variants",
        evidence=s.PROTOCOL["evidence"], variants=variants, cells=statuses, metrics=metrics, comparisons=comparisons,
        audit_sha256=[v["sha256"] for v in audits], outputs=[p.binding(path) for path in paths],
        historical_results_changed=False, protected_files_verified=len(plan["protected"]),
        heldout_scores_computed=bool(frames), main_result_promotion=False))
