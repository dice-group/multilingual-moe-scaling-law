"""Bind the revised paper's prediction and decision evidence without refitting.

Historical smoothing outputs and all fit campaigns remain immutable inputs.
The K200 decision test uses only held-out recipes, exact registered cohorts,
and one exposure mapping for every predictor. This is development validation,
not a new prospective test. Repeated budgets are not independent samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from result_analysis.capacity_response.tables import load_primary
from result_analysis.language_preservation.diagnostics import dilution_reference
from result_analysis.evidence import sha256 as sha256_file
from result_analysis.learning_curves import LearningCurve, allocation_metrics

ROOT = Path(__file__).resolve().parents[2]
SMOOTH = ROOT/"result_analysis/stage1_multilingual_pressure/outputs/smooth_analysis"
PARENT = ROOT/"result_analysis/stage1_m6_size_duration_v3/plan.json"
ABLATIONS = ROOT/"result_analysis/stage1_size_duration_ablations_v1"
OUT = ROOT/"result_analysis/stage1_joint_story_v1"
KEYS = ["trajectory_id", "readout_id", "language"]
COMPONENTS = ["D_t", "D_family", "D_script", "D_other"]
CAPS = (.4e9, .7e9, 2.2e9)
FOLD = "K200_common_languages"


def interpolate_exposure(tokens, components, query):
    """Interpolate cumulative counts jointly; zero components are legitimate."""
    tokens, components, query = np.asarray(tokens), np.asarray(components), np.asarray(query)
    if tokens.ndim != 1 or components.shape != (len(tokens), 4) or np.any(np.diff(tokens) <= 0):
        raise ValueError("invalid exposure trajectory shape or order")
    if not np.isfinite(components).all() or np.any(components < 0) or not np.allclose(components.sum(axis=1), tokens, rtol=1e-12, atol=1e-6):
        raise ValueError("exposure components do not reconstruct total tokens")
    if np.any(query < tokens[0]*(1-1e-12)) or np.any(query > tokens[-1]*(1+1e-12)):
        raise ValueError("exposure query outside trained support")
    query = np.clip(query, tokens[0], tokens[-1])
    result = np.stack([np.interp(query, tokens, components[:, j]) for j in range(4)], axis=-1)
    if not np.allclose(result.sum(axis=-1), query, rtol=1e-12, atol=1e-6):
        raise ValueError("interpolation broke exposure accounting")
    return result


def model_prediction(name, selected, rows):
    from multilingual_megatron.experiments.scaling.stage1_fit import fit
    from multilingual_megatron.experiments.scaling.stage1_fit.m6_size_duration_workflow import model as nd, reporting
    if name in ("M6-ND", "J0-ND", "M5-ND"):
        return nd.predict(reporting.features(rows), selected)
    coefficients = {v["language"]: v for v in selected["language_coefficients"]}
    if not set(rows.language) <= set(coefficients):
        raise ValueError("predictor lacks registered scoring-language coefficients")
    nuisance = [coefficients[l] for l in rows.language]
    shared = fit.SharedParameters(**selected["shared_parameters"]) if "shared_parameters" in selected else fit.shared_from_outer(name, selected["coordinates"])
    kwargs = {k: rows[k].to_numpy() for k in COMPONENTS[1:]} if name == "M6" else {}
    return fit.predict(name, shared, rows.N_act, rows.D_t, rows.E, rows.K,
        [v["L_inf"] for v in nuisance], [v["A"] for v in nuisance], [v["B"] for v in nuisance], **kwargs)


def load_evidence(ablation_root=ABLATIONS):
    from multilingual_megatron.experiments.scaling.stage1_fit import size_duration_ablations as ab
    from multilingual_megatron.experiments.scaling.stage1_fit.m6_size_duration_workflow import planning as p, execution as e
    plan, parent = ab.load(ablation_root/"plan.json")
    report = p.read_sealed(ablation_root/"report.json")
    if report["plan_sha256"] != plan["sha256"]:
        raise ValueError("ablation report belongs to a different plan")
    sources = [p.binding(ablation_root/"plan.json"), p.binding(ablation_root/"report.json"), report["predictions"],
               p.binding(PARENT), p.binding(Path(parent["output_root"])/"reports/report.json")]
    predictions = pd.read_parquet(p.diagnosis.bound_file(report["predictions"]))
    main = p.read_sealed(Path(parent["output_root"])/"reports/report.json")
    if main["decision"]["passed"] is not False or not main["numerical_audit_healthy"]:
        raise ValueError("historical scientific/numerical status changed")
    comparison_path = next(p.diagnosis.bound_file(v) for v in plan["bindings"] if Path(v["path"]).name == "primary_baseline_comparison.json")
    comparison = json.loads(comparison_path.read_text())
    selected = {}
    for source in comparison["source_results"]:
        if source["fold"] == FOLD:
            path = p.diagnosis.bound_file(source)
            selected[source["model"]] = json.loads(path.read_text())["selected"]
            sources.append(p.binding(path))
    selected["M6"] = parent["folds"][FOLD]["m6_selected"]
    reduction = e.read_artifact(parent, "reduction.json")
    selected["M6-ND"] = reduction["folds"][FOLD]["selected"]
    sources.append(p.binding(Path(parent["output_root"])/"reduction.json"))
    for binding in report["cells"]:
        cell = p.read_sealed(p.diagnosis.bound_file(binding))
        sources.append(binding)
        if cell["fold"] == FOLD and report["models"][cell["model"]] == "complete":
            if cell["status"] != "healthy":
                raise ValueError("uncertified ablation cannot enter policy comparison")
            selected[cell["model"]] = cell["selected"]
    manifest = parent["folds"][FOLD]["manifest"]
    membership = manifest["membership"]
    if set(membership["train_trajectory_ids"]) & set(membership["test_trajectory_ids"]):
        raise ValueError("trajectory leakage")
    rows = predictions.query("model == 'M6-ND' and fold == @FOLD and split == 'test'").copy()
    if rows.duplicated(KEYS).any() or set(rows.trajectory_id) != set(membership["test_trajectory_ids"]):
        raise ValueError("incomplete or duplicated held-out recipe inventory")
    sealed = load_primary(ROOT)
    matched = rows.merge(sealed[KEYS+["N_total", "N_act", "D_t", "D_total", "adjusted_loss"]], on=KEYS,
                         suffixes=("", "_sealed"), validate="one_to_one")
    if len(matched) != len(rows):
        raise ValueError("heldout rows missing from sealed training release")
    for column in ("N_act", "D_t", "D_total", "adjusted_loss"):
        if not np.allclose(matched[column], matched[column+"_sealed"], rtol=1e-12, atol=1e-12):
            raise ValueError("fit and sealed observations disagree")
    rows = matched.drop(columns=[c for c in matched if c.endswith("_sealed")])
    for name, coefficients in selected.items():
        reference = predictions.query("model == @name and fold == @FOLD and split == 'test'")[KEYS+["prediction"]]
        order = rows[KEYS].merge(reference, on=KEYS, validate="one_to_one")
        if len(order) != len(rows) or not np.allclose(model_prediction(name, coefficients, rows), order.prediction, rtol=0, atol=1e-10):
            raise ValueError(f"checkpoint prediction parity failed: {name}")
    smooth = json.loads((SMOOTH/"manifest.json").read_text())
    sources.append(p.binding(SMOOTH/"manifest.json"))
    for name, digest in smooth["output_sha256"].items():
        if sha256_file(SMOOTH/name) != digest:
            raise ValueError(f"smoothing source drift: {name}")
    # Input hashes above bind the saved results. Their producer's code hashes
    # describe that run, not the current implementation used for a new replay.
    cohorts = {"common12": smooth["cohorts"]["12"], "incumbent50": membership["score_languages"]}
    if len(cohorts["incumbent50"]) != 50 or not set(cohorts["common12"]) <= set(cohorts["incumbent50"]):
        raise ValueError("unexpected registered scoring cohorts")
    return report, predictions, rows, selected, cohorts, smooth, sources


def decision_analysis(rows, selected, cohorts, penalty, *, predictor=model_prediction):
    """Replay the frozen decision grid; an explicit predictor changes only the law."""
    recipes = []
    for (ladder, experts), group in rows.groupby(["ladder", "E"]):
        recipes.append(dict(name=f"{ladder}/E{experts}", n=float(group.N_act.iloc[0]),
            stored=float(group.N_total.iloc[0]), e=int(experts), rows=group,
            lower=6*group.N_act.iloc[0]*group.D_total.min()/1e18,
            upper=6*group.N_act.iloc[0]*group.D_total.max()/1e18))
    recipes.sort(key=lambda v: (v["stored"], v["name"]))
    grid = np.geomspace(max(c["lower"] for c in recipes), min(c["upper"] for c in recipes), 81)
    observed = {name: [] for name in ("linear_log", "pchip_log", "smooth")}
    predicted = {name: [] for name in selected}
    languages = sorted(cohorts["incumbent50"])
    for recipe in recipes:
        query = grid*1e18/(6*recipe["n"])
        features, truth = [], {name: [] for name in observed}
        for language in languages:
            group = recipe["rows"].loc[recipe["rows"].language.eq(language)].sort_values("D_total")
            tokens, loss = group.D_total.to_numpy(), group.adjusted_loss.to_numpy()
            components = interpolate_exposure(tokens, group[COMPONENTS].to_numpy(), query)
            f = pd.DataFrame(components, columns=COMPONENTS)
            f["N_act"], f["D_total"], f["E"], f["K"], f["language"] = recipe["n"], query, recipe["e"], 200, language
            features.append(f)
            truth["linear_log"].append(np.interp(np.log(query), np.log(tokens), loss))
            truth["pchip_log"].append(PchipInterpolator(np.log(tokens), loss, extrapolate=False)(np.log(query)))
            truth["smooth"].append(LearningCurve.fit(tokens, loss, penalty)(query))
        features = pd.concat(features, ignore_index=True)
        for method in observed:
            observed[method].append(np.asarray(truth[method]))
        for name, coefficients in selected.items():
            predicted[name].append(predictor(name, coefficients, features).reshape(len(languages), len(grid)))
    observed = {k: np.asarray(v) for k, v in observed.items()}
    predicted = {k: np.asarray(v) for k, v in predicted.items()}
    metrics, decisions = [], []
    for cohort_name, cohort in cohorts.items():
        positions = [languages.index(l) for l in cohort]
        for method, values in observed.items():
            truth = values[:, positions].mean(axis=1)
            for name, values in predicted.items():
                prediction = values[:, positions].mean(axis=1)
                for cap in CAPS:
                    indices = [i for i, c in enumerate(recipes) if c["stored"] <= cap]
                    score = allocation_metrics(truth[indices], prediction[indices])
                    metrics.append(dict(model=name, fold=FOLD, cohort=cohort_name, cohort_size=len(cohort),
                        reference=method, cap=cap, candidate_count=len(indices), compute_points=len(grid),
                        mean_regret=float(score["regret"].mean()), max_regret=float(score["regret"].max()),
                        winner_accuracy=float(score["winner_matches"].mean()),
                        within_005=float(np.mean(score["regret"] <= .005)),
                        within_010=float(np.mean(score["regret"] <= .01)),
                        within_020=float(np.mean(score["regret"] <= .02)),
                        pairwise_margin_rmse=score["pairwise_margin_rmse"], resolved_pairwise_accuracy=score["resolved_pairwise_accuracy"]))
                    for j, f in enumerate(grid):
                        chosen = indices[score["chosen_indices"][j]]
                        decisions.append(dict(model=name, fold=FOLD, cohort=cohort_name, reference=method,
                            cap=cap, F=float(f), chosen=recipes[chosen]["name"], regret=score["regret"][j],
                            observed_loss=truth[chosen, j], best_observed_loss=truth[indices, j].min(),
                            predicted_loss=prediction[chosen, j]))
    return pd.DataFrame(metrics), pd.DataFrame(decisions)




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--ablations", type=Path, default=ABLATIONS)
    args = parser.parse_args()
    report, predictions, rows, selected, cohorts, smooth, sources = load_evidence(args.ablations)
    print("Verified all checkpoint predictions and scoring memberships", flush=True)
    metrics, decisions = decision_analysis(rows, selected, cohorts, smooth["smoothing_strengths"]["smooth"])
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output/"policy_metrics.csv", index=False)
    decisions.to_csv(args.output/"policy_decisions.csv", index=False)
    pd.DataFrame.from_dict(report["statistics"]["metrics"], orient="index").rename_axis("model").reset_index().to_csv(args.output/"holdout_metrics.csv", index=False)
    heldout = predictions.query("split == 'test'")
    horizons = heldout.groupby(["model", "fold", "D_total", "trajectory_id"])[["squared_error", "residual"]].mean().groupby(["model", "fold", "D_total"]).mean().reset_index()
    horizons["rmse"] = np.sqrt(horizons.squared_error)
    horizons.to_csv(args.output/"horizon_metrics.csv", index=False)
    dilution_reference(load_primary(ROOT), pd.read_csv(SMOOTH/"expansion_policies.csv")).to_csv(args.output/"expansion_dilution.csv", index=False)
    for name in ("pressure_curves", "pressure_support", "pressure_common_support_summary", "allocation_crossovers",
                 "allocation_regret", "expansion_policies", "incumbent_loss_changes", "nested_horizon_validation_summary",
                 "observed_language_reversals"):
        source = SMOOTH/f"{name}.csv"
        (args.output/source.name).write_bytes(source.read_bytes())
        sources.append(dict(path=str(source), file_sha256=sha256_file(source)))
    manifest = dict(schema_version="stage1_joint_story/v1", status="complete", ablation_status=report["models"],
        source_files=sources, code_sha256={str(Path(__file__).relative_to(ROOT)): sha256_file(Path(__file__)),
                                          "result_analysis/learning_curves.py": sha256_file(ROOT/"result_analysis/learning_curves.py")},
        units=dict(loss="nats per validation token", F="EFLOPs, 6*N_act*D_total", cap="stored parameter count"),
        cohorts=cohorts, grid="81 logarithmically spaced budgets on all-recipe common support",
        exposure_mapping="piecewise-linear cumulative component counts in total tokens, shared by every predictor",
        independent_units="64 main-grid trajectories; repeated budgets and checkpoints are dependent",
        evidence="post-diagnosis internal validation; no prospective or causal interpretation",
        output_sha256={f.name: sha256_file(f) for f in sorted(args.output.glob("*.csv"))})
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(metrics.query("cohort == 'incumbent50' and reference == 'linear_log'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
